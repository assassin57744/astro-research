# identity.py          # 路径 0: 保持不变

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

class IdentityComponentModeller:
    """
    M membership 阶段二：Identity (单组分基准动力学重构模型)
    针对年轻或高度紧凑、无显著潮汐尾的星团，构建标准的全局束缚超椭球。
    与双组分、三组分模型保持完全对齐的 API 架构。
    """
    def __init__(self, config: dict):
        self.config = config
        self.features = ['ra', 'dec', 'pmra', 'pmdec', 'plx']
        self.model = None

    def _compute_density_weights(self, X: np.ndarray) -> np.ndarray:
        """ 
        空间密度平权算子。
        在单组分中，主要用于轻度压制核心极端致密区的权重，
        使得全局高斯椭球的主轴能更健康地向外围晕区（Halo）泛成员拉伸，防止过度拟合核心核心区。
        """
        coords = X[:, :2]
        kde = KernelDensity(bandwidth='scott', kernel='gaussian')
        kde.fit(coords)
        log_dens = kde.score_samples(coords)
        dens = np.exp(log_dens)
        
        weights = 1.0 / (dens + 1e-6)
        # 单组分不需要过于激进的对冲，将上限控制在 85% 分位数，保持核心的主导性
        weights = np.clip(weights, np.percentile(weights, 1), np.percentile(weights, 85))
        return weights / np.mean(weights)

    def _generate_identity_priors(self, X: np.ndarray, seed_labels: np.ndarray):
        """
        利用上游高纯度种子洗出第一代最纯净的星团核心相空间质心与协方差基准
        """
        core_mask = (seed_labels == self.config['TARGET_CLUSTER_LABEL'])
        X_core = X[core_mask]
        
        if len(X_core) == 0:
            raise ValueError(f"❌ [Error] 未能在输入中匹配到目标星团标签: {self.config['TARGET_CLUSTER_LABEL']}")

        # 提取核心纯净种子的统计特征作为唯一的组件初始化
        mean_init = np.mean(X_core, axis=0).reshape(1, -1)
        cov_init = np.cov(X_core, rowvar=False).reshape(1, self.model_dims, self.model_dims) if hasattr(self, 'model_dims') else np.cov(X_core, rowvar=False).reshape(1, 5, 5)
        weights_init = np.array([1.0])
        
        return mean_init, cov_init, weights_init

    def fit(self, df_master: pd.DataFrame):
        """ 训练单组分基准模型 """
        print("🚀 [Substructure] 开始构建 Identity 单组分全局本征空间...")
        
        X = df_master[self.features].values
        seed_labels = df_master['seed_label'].values
        self.model_dims = X.shape[1]
        
        # 1. 抽取核心硬原子先验
        means_init, covs_init, weights_init = self._generate_identity_priors(X, seed_labels)
        
        # 2. 计算空间密度权重
        sample_weights = self._compute_density_weights(X)
        
        # 3. 实例化单组分 GMM
        self.model = GaussianMixture(
            n_components=1,
            covariance_type='full',
            means_init=means_init,
            weights_init=weights_init,
            tol=1e-4,
            max_iter=200,
            random_state=42,
            warm_start=False
        )
        
        self.model.means_init = means_init
        self.model.weights_init = weights_init
        
        # 4. 轰鸣拟合
        # self.model.fit(X, sample_weight=sample_weights)
        self.model.fit(X)           # TODO 后续必须修改
        
        print("🎯 [Substructure] 单组分基准模型收敛成功！")
        self._audit_components()
        
        return self

    def _audit_components(self):
        """ 动力学重构结果硬核审计 """
        mean = self.model.means_[0]
        diag_cov = np.diag(self.model.covariances_[0])
        
        print(f"  🔹 Component 0 -> Global Cluster Entity (全局星团实体):")
        print(f"     权重占比: {self.model.weights_[0]:.4f} (Locked)")
        print(f"     空间质心 (RA, DEC): ({mean[0]:.3f}, {mean[1]:.3f})")
        print(f"     速度质心 (pmra, pmdec): ({mean[2]:.3f}, {mean[3]:.3f})")
        print(f"     5D 本征弥散方差: {np.array2string(diag_cov, precision=4)}")

    def predict_membership(self, df_广域: pd.DataFrame) -> pd.DataFrame:
        """ 广域 18 度大沙盘本征成员概率解算 """
        X_all = df_广域[self.features].values
        
        # 对于单组分，predict_proba 必然输出全 1 矩阵，无法提供渐变概率
        # 因此我们必须利用 score_samples 获取绝对对数似然，再映射回连续概率空间
        log_likelihood = self.model.score_samples(X_all)
        
        df_res = df_广域.copy()
        # 存储绝对对数似然值，供下游消歧模块做卡方或拐点截断
        df_res['log_p_identity'] = log_likelihood
        
        # 为了保持接口统一，派发一个归一化的相对成员概率（在单模型视角下等于 1.0）
        df_res['p_core'] = 1.0
        df_res['p_tail'] = 0.0
        df_res['p_leading'] = 0.0
        df_res['p_trailing'] = 0.0
        df_res['p_total_cluster'] = 1.0 
        
        return df_res