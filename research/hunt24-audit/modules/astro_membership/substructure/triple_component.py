# triple_component.py  # 路径 2: Core + Leading + Trailing (3组件)

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

class TripleComponentModeller:
    """
    M membership 阶段二：Core + Leading Tail + Trailing Tail (三组件动力学精细解剖模型)
    利用运动学主轴投影，手工切分前导与后随先验，引导 GMM 完美捕捉非线性不连续长尾
    """
    def __init__(self, config: dict):
        self.config = config
        self.features = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']
        self.model = None

    def _compute_density_weights(self, X: np.ndarray) -> np.ndarray:
        """ 空间逆密度平权算子，强行撑大外围微弱长尾的似然贡献 """
        coords = X[:, :2]
        kde = KernelDensity(bandwidth='scott', kernel='gaussian')
        kde.fit(coords)
        log_dens = kde.score_samples(coords)
        dens = np.exp(log_dens)
        
        weights = 1.0 / (dens + 1e-6)
        weights = np.clip(weights, np.percentile(weights, 1), np.percentile(weights, 95))
        return weights / np.mean(weights)

    def _generate_triple_priors(self, X: np.ndarray, seed_labels: np.ndarray):
        """
        基于天体动力学各向异性，通过运动学主轴投影，分裂出前导尾与后随尾的硬初始化矩阵
        """
        # 1. 锁死绝对纯净的核心
        core_mask = (seed_labels == self.config['TARGET_CLUSTER_LABEL'])
        X_core = X[core_mask]
        X_outer = X[~core_mask]
        
        if len(X_core) == 0:
            raise ValueError(f"❌ [Error] 未能在输入中匹配到目标星团标签: {self.config['TARGET_CLUSTER_LABEL']}")

        # ---- Component 0: Core (核心组件基准) ----
        mean_core = np.mean(X_core, axis=0)
        cov_core = np.cov(X_core, rowvar=False)

        # ---- 运动学主轴解耦：寻找潮汐撕裂方向 ----
        # 计算外围星相对于核心的自行偏差
        delta_pm = X_outer[:, 2:4] - mean_core[2:4]
        
        # 利用 SVD 提取外围速度场的主特征向量（即潮汐力拉伸的主轴方向）
        U, S, Vt = np.linalg.svd(delta_pm, full_matrices=False)
        primary_axis = Vt[0, :] # 2D 速度空间的主要撕裂矢量
        
        # 将每颗外围星投影到这条运动学主轴上
        projections = np.dot(delta_pm, primary_axis)
        
        # 顺着主轴投影的正负，暴力切分前导阵营与后随阵营
        leading_mask = (projections >= 0)
        X_leading_init = X_outer[leading_mask]
        X_trailing_init = X_outer[~leading_mask]

        # ---- Component 1: Leading Tail (前导尾先验) ----
        if len(X_leading_init) > 5:
            mean_leading = np.mean(X_leading_init, axis=0)
            cov_leading = np.cov(X_leading_init, rowvar=False) * 2.0
        else:
            # 兜底：若前导样本过稀疏，从核心质心向前推移
            mean_leading = mean_core.copy()
            mean_leading[0] += 0.5 # 空间 RA 正向外推
            cov_leading = np.cov(X, rowvar=False) * 3.0
            
        # ---- Component 2: Trailing Tail (后随尾先验) ----
        if len(X_trailing_init) > 5:
            mean_trailing = np.mean(X_trailing_init, axis=0)
            cov_trailing = np.cov(X_trailing_init, rowvar=False) * 2.0
        else:
            # 兜底：向核心质心反向外推
            mean_trailing = mean_core.copy()
            mean_trailing[0] -= 0.5
            cov_trailing = np.cov(X, rowvar=False) * 3.0

        # 强行对两条尾巴的空间协方差进行方向性各向异性注入
        cov_leading[0, 0] *= 4.0; cov_leading[1, 1] *= 4.0
        cov_trailing[0, 0] *= 4.0; cov_trailing[1, 1] *= 4.0

        # 组合三组先验
        means_init = np.vstack([mean_core, mean_leading, mean_trailing])
        covs_init = np.stack([cov_core, cov_leading, cov_trailing])
        weights_init = np.array([0.5, 0.25, 0.25]) # 预估核心占一半，两条尾巴各分四分之一
        
        return means_init, covs_init, weights_init

    def fit(self, df_master: pd.DataFrame):
        """ 训练三组分重构模型 """
        print("🚀 [Substructure] 开始构建 Core + Leading + Trailing 三组分相空间流动...")
        
        X = df_master[self.features].values
        seed_labels = df_master['seed_label'].values
        
        # 1. 抽取运动学主轴非对称先验
        means_init, covs_init, weights_init = self._generate_triple_priors(X, seed_labels)
        
        # 2. 计算密度权重
        sample_weights = self._compute_density_weights(X)
        
        # 3. 构建三组分高斯混合模型
        self.model = GaussianMixture(
            n_components=3,
            covariance_type='full',
            means_init=means_init,
            weights_init=weights_init,
            tol=1e-5, # 3组件形态更复杂，收敛精度调高半个数量级
            max_iter=400,
            random_state=42,
            warm_start=False
        )
        
        self.model.means_init = means_init
        self.model.weights_init = weights_init
        
        # 4. 轰鸣拟合
        self.model.fit(X, sample_weight=sample_weights)
        
        print("🎯 [Substructure] 三组分亚结构精细解剖收敛成功！")
        self._audit_components()
        
        return self

    def _audit_components(self):
        """ 动力学重构结果硬核审计 """
        types = ["Core (核心区)", "Leading Tail (前导尾)", "Trailing Tail (后随尾)"]
        for i in range(3):
            weight = self.model.weights_[i]
            mean = self.model.means_[i]
            diag_cov = np.diag(self.model.covariances_[i])
            
            print(f"  🔥 Component {i} -> {types[i]}:")
            print(f"     数量权重: {weight:.4f}")
            print(f"     质心位置 (RA, DEC): ({mean[0]:.3f}, {mean[1]:.3f})")
            print(f"     速度质心 (pmra, pmdec): ({mean[2]:.3f}, {mean[3]:.3f})")
            print(f"     5D 弥散方差: {np.array2string(diag_cov, precision=4)}")

    def predict_membership(self, df_广域: pd.DataFrame) -> pd.DataFrame:
        """ 广域 18 度大沙盘解算与打标 """
        X_all = df_广域[self.features].values
        probs = self.model.predict_proba(X_all)
        
        df_res = df_广域.copy()
        df_res['p_core'] = probs[:, 0]
        df_res['p_leading'] = probs[:, 1]
        df_res['p_trailing'] = probs[:, 2]
        # 泛成员总体概率 = 核心 + 前导 + 后随
        df_res['p_total_cluster'] = probs[:, 0] + probs[:, 1] + probs[:, 2]
        
        return df_res