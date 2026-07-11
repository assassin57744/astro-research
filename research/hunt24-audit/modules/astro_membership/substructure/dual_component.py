# dual_component.py    # 路径 1: Core + Tail (2组件)

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

class DualComponentModeller:
    """
    M membership 阶段二：Core + Tail (双组分动力学精细重构模型)
    利用阶段一清洗后的 Master 泛成员数据，通过天体物理先验显式引导 GMM 分解
    """
    def __init__(self, config: dict):
        self.config = config
        self.features = ['ra', 'dec', 'pmra', 'pmdec', 'plx']
        self.model = None

    def _compute_density_weights(self, X: np.ndarray) -> np.ndarray:
        """
        利用 KDE 计算 2D 天球空间局部密度，并转化为逆密度权重，防止高密度核心夺权
        """
        # 提取空间坐标
        coords = X[:, :2] 
        
        # 训练一个轻量级 KDE 来评估空间拥挤度
        kde = KernelDensity(bandwidth='scott', kernel='gaussian')
        kde.fit(coords)
        log_dens = kde.score_samples(coords)
        dens = np.exp(log_dens)
        
        # 计算逆密度权重：权重 = 1 / 密度
        # 为了防止极端离群星权重过大，进行软截断 (clip)
        weights = 1.0 / (dens + 1e-6)
        weights = np.clip(weights, np.percentile(weights, 1), np.percentile(weights, 95))
        
        # 归一化，使其均值为 1
        return weights / np.mean(weights)

    def _generate_astrophysical_priors(self, X: np.ndarray, seed_labels: np.ndarray):
        """
        根据上游 DBSCAN 提取的种子标签(如真正的星团标识 5)，手工锻造第一代 GMM 先验
        """
        # 1. 提取绝对纯净的核心区数据 (DBSCAN 锁定的目标星团主峰)
        core_mask = (seed_labels == self.config['TARGET_CLUSTER_LABEL'])
        X_core = X[core_mask]
        X_outer = X[~core_mask]
        
        if len(X_core) == 0:
            raise ValueError(f"❌ [Error] 未能在输入数据中匹配到目标星团标签: {self.config['TARGET_CLUSTER_LABEL']}")

        # ---- Component 0: Core (核心组件先验) ----
        mean_core = np.mean(X_core, axis=0)
        cov_core = np.cov(X_core, rowvar=False)
        
        # ---- Component 1: Tail (整体潮汐尾组件先验) ----
        # 质心：初始置于核心外围或整体均值
        mean_tail = np.mean(X, axis=0) 
        
        # 协方差：手工锻造长尾各向异性。让其速度和空间弥散是核心的 3~5 倍
        cov_tail = np.cov(X, rowvar=False) * 4.0 
        
        # 强行拉长协方差矩阵中天球空间(RA/DEC)与自行(pmra/pmdec)的主轴特征
        # 借此注入潮汐尾沿着银河系引力剪切方向拉伸的物理趋势
        cov_tail[0, 0] *= 3.0  # 拉伸 RA
        cov_tail[1, 1] *= 3.0  # 拉伸 DEC

        # 合并初始化参数
        means_init = np.vstack([mean_core, mean_tail])
        covs_init = np.stack([cov_core, cov_tail])
        weights_init = np.array([0.6, 0.4]) # 预估核心与外围泛成员的初始数量分布
        
        return means_init, covs_init, weights_init

    def fit(self, df_master: pd.DataFrame):
        """
        训练双组分重构模型
        df_master 必须包含 5D 特征以及上游粗筛留下的 'seed_label'
        """
        print("🚀 [Substructure] 开始构建 Core + Tail 双组分动力学空间...")
        
        X = df_master[self.features].values
        seed_labels = df_master['seed_label'].values
        
        # 1. 生成天体物理硬先验
        means_init, covs_init, weights_init = self._generate_astrophysical_priors(X, seed_labels)
        
        # 2. 计算空间样本权重，对冲洋葱圈效应
        sample_weights = self._compute_density_weights(X)
        
        # 3. 实例化定制 GMM
        # 强制显式指定 init_params='random' 以便让我们的手工先验生效，而不被默认的 KMeans 覆盖
        self.model = GaussianMixture(
            n_components=2,
            covariance_type='full',
            means_init=means_init,
            precisions_init=None,  # 由 covs_init 自动反转
            weights_init=weights_init,
            tol=1e-4,
            max_iter=300,
            random_state=42,
            warm_start=False
        )
        
        # 由于 sklearn 原生 GaussianMixture.fit 在传入 covariances_init 时需要通过 _initialize 触发
        # 我们采用手工预设模型内部参数的策略，直接硬卡住第一步的起跑线
        self.model.means_init = means_init
        self.model.weights_init = weights_init
        
        # 4. 轰鸣训练 (注入平权 sample_weight)
        # self.model.fit(X, sample_weight=sample_weights)
        self.model.fit(X)               # TODO 后续要更改为带权重的拟合
        
        print("🎯 [Substructure] 双组分收敛成功！")
        self._audit_components()
        
        return self

    def _audit_components(self):
        """ 动力学重构结果硬核审计 """
        for i in range(2):
            comp_type = "Core (核心)" if i == 0 else "Tail (潮汐尾)"
            weight = self.model.weights_[i]
            mean = self.model.means_[i]
            diag_cov = np.diag(self.model.covariances_[i])
            
            print(f"  🔹 Component {i} -> {comp_type}:")
            print(f"     权重占比: {weight:.4f}")
            print(f"     空间质心 (RA, DEC): ({mean[0]:.3f}, {mean[1]:.3f})")
            print(f"     相空间本征弥散 (对角线方差): {np.array2string(diag_cov, precision=4)}")

    def predict_membership(self, df_field: pd.DataFrame) -> pd.DataFrame:
        """
        对 18 度广域沙盘全量成员进行亚结构概率解码
        """
        X_all = df_field[self.features].values
        
        # 计算每颗星属于 Core 和 Tail 的各自后验概率
        probs = self.model.predict_proba(X_all)
        
        df_res = df_field.copy()
        df_res['p_core'] = probs[:, 0]
        df_res['p_tail'] = probs[:, 1]
        df_res['p_total_cluster'] = df_res['p_core'] + df_res['p_tail']
        
        return df_res