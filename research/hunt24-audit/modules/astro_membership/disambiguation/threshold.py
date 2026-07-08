# threshold.py         # 策略 2: 卡方阈值截断

import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from ..base import BaseDisambiguation

# 引入管线标准日志器，保持与其他 Stage 日志行为高度一致
import logging

logger = logging.getLogger(__name__)


class ThresholdGmmDisambiguation(BaseDisambiguation):
    def __init__(self, sigma_cutoff=3.0, **kwargs):
        self.sigma_cutoff = sigma_cutoff

    def fit_predict(self, df_all, df_seeds, features):
        X_all = df_all[features].values
        X_seeds = df_seeds[features].values

        gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
        gmm.fit(X_seeds)

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

        scores = gmm.score_samples(X_all)

        df_result = df_all.copy()
        # 优化概率计算：防止极小值引发的下溢，同时避免盲目归一化破坏多模型贝叶斯链的概率标度
        df_result["prob"] = np.exp(np.clip(scores, -708, 708))
        df_result["is_member"] = scores >= score_threshold

        # 精准记录洗涤前后成员星数量变化变化，契合管线标准埋点规范
        n_before = len(df_all)
        n_after = int(df_result["is_member"].sum())
        logger.info(
            f"[Threshold Wash] Features: {n_features}D | Cutoff: {self.sigma_cutoff} sigma ({chi2_cutoff:.2f}) | Total: {n_before} -> Members: {n_after} | Washed out: {n_before - n_after}"
        )

        return df_result
