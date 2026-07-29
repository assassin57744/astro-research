# threshold.py         # 策略 2: 卡方阈值截断 与 自适应导数拐点截断双模算子

import numpy as np
import pandas as pd  # 确保通用数据操作安全
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler  # 引入管线标准归一化器
from modules.membership.base import BaseDisambiguation

# 引入管线标准日志器，保持与其他 Stage 日志行为高度一致
import logging

logger = logging.getLogger(__name__)


class ThresholdGmmDisambiguation(BaseDisambiguation):

    def __init__(self, sigma_cutoff=3.0, method="chi2", **kwargs):
        self.sigma_cutoff = sigma_cutoff
        self.method = method  # "chi2" 或 "knee"

    def _find_knee_threshold(self, scores, initial_threshold):
        """
        利用对数似然分布的一阶与二阶导数寻找背景噪声平台的自适应断层拐点
        """
        # 1. 过滤掉极度远离核心的垃圾背景区，对核心与背景交界处的对数似然进行网格化密度估计
        valid_scores = scores[scores > initial_threshold * 2.0]
        if len(valid_scores) < 100:
            return initial_threshold
            
        counts, bin_edges = np.histogram(valid_scores, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # 2. 计算一阶导数 (梯度) 和 二阶导数 (曲率)
        grad = np.gradient(counts)
        smooth_grad = np.convolve(grad, np.ones(5)/5, mode='same')  # 均值平滑
        curvature = np.gradient(smooth_grad)
        
        # 3. 寻找从高密度主峰向低密度背景平台过渡时，一阶导数急剧减缓、二阶导数出现显著正峰（拐弯）的临界点
        # 扫描范围限定在初筛硬卡位附近
        search_idx = np.where(bin_centers < initial_threshold)[0]
        if len(search_idx) == 0:
            return initial_threshold
            
        # 在搜索区间内寻找曲率最大（弯曲最剧烈）的局部断层
        knee_idx = search_idx[np.argmax(curvature[search_idx])]
        knee_threshold = bin_centers[knee_idx]
        
        return knee_threshold

    def fit_predict(self, df_all, df_seeds, features):
        # 🛡️ 增量优化：构建特征完整的数据子集，防止 NaN 引发底层 GMM 算子崩溃
        df_field_clean = df_all.dropna(subset=features).copy()
        df_seeds_clean = df_seeds.dropna(subset=features).copy()

        # 🌟 核心修复 1：引入数学空间归一化，使用全域背景数据的方差结构作为统一的度量基准
        scaler = StandardScaler()
        scaler.fit(df_field_clean[features])

        X_all_scaled = scaler.transform(df_field_clean[features])
        X_seeds_scaled = scaler.transform(df_seeds_clean[features])

        gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
        # 使用标准化后的种子集锁定星团核心形态
        gmm.fit(X_seeds_scaled)

        n_features = len(features)
        # 修正高维相空间下的卡方切分点映射：高维椭球等效积分直接由 sigma_cutoff 转换为对应的 CDF 分位数
        # 对于多维高斯分布，马氏距离平方服从自由度为 n_features 的卡方分布
        chi2_cutoff = chi2.ppf(
            2.0 * chi2.cdf(self.sigma_cutoff, df=1) - 1.0, df=n_features
        )

        log_det_cov = np.log(np.linalg.det(gmm.covariances_[0]))
        score_threshold = -0.5 * (
            chi2_cutoff + n_features * np.log(2 * np.pi) + log_det_cov
        )

        # 使用标准化后的全量矩阵计算对数似然场
        scores = gmm.score_samples(X_all_scaled)

        # 🚀 动态潜力解绑：如果启用 knee 模式，基于非参数似然导数微调卡位线
        if self.method == "knee":
            score_threshold = self._find_knee_threshold(scores, score_threshold)

        df_result = df_all.copy()

        # 初始化默认输出状态（兜底拦截：默认概率为 0，且非数组成员）
        df_result["prob"] = 0.0
        df_result["is_member"] = False

        # 🌟 核心修复 2 & 3：利用原始行索引，将干净样本算出的确定性结果精准刷回主输出表
        # 将 PDF 密度转换为管线统一的成员归属概率尺度（内部 1.0，外部 0.0）
        df_result.loc[df_field_clean.index, "is_member"] = scores >= score_threshold
        df_result.loc[df_field_clean.index, "prob"] = np.where(
            scores >= score_threshold, 1.0, 0.0
        )

        # 精准记录洗涤前后成员星数量变化变化，契合管线标准埋点规范
        n_before = len(df_all)
        n_after = int(df_result["is_member"].sum())
        logger.info(
            f"[Threshold Wash] Features: {n_features}D | Cutoff: {self.sigma_cutoff} sigma ({chi2_cutoff:.2f}) | Method: {self.method} | Total: {n_before} -> Members: {n_after} | Washed out: {n_before - n_after}"
        )

        return df_result
