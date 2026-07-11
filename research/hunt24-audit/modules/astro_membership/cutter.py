# modules/astro_membership/cutter.py
# --------------------------------==================--------------------------------
# DensityFieldCutter: 广域大沙盘高维流形密度场自适应裁剪与终极打标算子
# --------------------------------==================--------------------------------

import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2

logger = logging.getLogger(__name__)


class DensityFieldCutter:
    """
    无模型依赖的高维标量场自适应裁剪器。
    专门吃入上游解剖引擎（Identity/Dual/Triple）沉淀的对数似然或综合概率流形，
    利用卡方等效积分与非参数密度导数拐点（Knee-Point）定位终极物理斩杀线。
    """

    def __init__(self, sigma_cutoff=3.0, method="chi2", **kwargs):
        self.sigma_cutoff = sigma_cutoff
        self.method = method  # "chi2" 或 "knee"

    def _find_knee_threshold(self, scores, initial_threshold):
        """利用对数似然分布的一阶与二阶导数寻找背景噪声平台的自适应断层拐点"""
        # 1. 过滤掉极度远离核心的垃圾背景区，对核心与背景交界处的对数似然进行网格化密度估计
        valid_scores = scores[scores > initial_threshold * 2.0]
        if len(valid_scores) < 100:
            return initial_threshold

        counts, bin_edges = np.histogram(valid_scores, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # 2. 计算一阶导数 (梯度) 和 二阶导数 (曲率)
        grad = np.gradient(counts)
        smooth_grad = np.convolve(grad, np.ones(5) / 5, mode="same")  # 均值平滑
        curvature = np.gradient(smooth_grad)

        # 3. 寻找从高密度主峰向低密度背景平台过渡时，一阶导数急剧减缓、二阶导数出现显著正峰的临界点
        search_idx = np.where(bin_centers < initial_threshold)[0]
        if len(search_idx) == 0:
            return initial_threshold

        # 在搜索区间内寻找曲率最大（弯曲最剧烈）的局部断层
        knee_idx = search_idx[np.argmax(curvature[search_idx])]
        knee_threshold = bin_centers[knee_idx]

        return knee_threshold

    def cut_field(self, df_all, target_score_col, features):
        """
        全天区密度场核心裁剪函数
        :param df_all: 已经由上游解剖引擎注入了似然场的大沙盘数据表
        :param target_score_col: 统一指定的斩杀目标列名（例如 'log_p_identity' 或 'p_total_cluster'）
        :param features: 参与重构的特征列表，用于卡方高维自由度解算
        :return: 经过自适应断层去噪后，精准刷回主表行索引的终极成员星资产表
        """
        # 🛡️ 确保裁剪目标列和行索引数据完整，防止 NaN 引发百分位数崩溃
        df_field_clean = df_all.dropna(subset=[target_score_col]).copy()
        scores = df_field_clean[target_score_col].values

        n_features = len(features)

        # 修正高维相空间下的卡方切分点映射：高维等效积分直接由 sigma_cutoff 转换为对应的 CDF 分位数
        chi2_cutoff = chi2.ppf(
            2.0 * chi2.cdf(self.sigma_cutoff, df=1) - 1.0, df=n_features
        )

        # 基于卡方高维等效积分，计算全局标量场统计分位数的硬截断理论线
        target_cdf = 2.0 * chi2.cdf(self.sigma_cutoff, df=1) - 1.0
        score_threshold = np.percentile(scores, (1.0 - target_cdf) * 100)

        # 🚀 动态潜力解绑：如果启用 knee 模式，基于非参数似然导数微调自适应断层线
        if self.method == "knee":
            score_threshold = self._find_knee_threshold(scores, score_threshold)

        df_result = df_all.copy()

        # 初始化默认输出状态（兜底拦截：默认概率为 0，且非数组成员）
        df_result["prob"] = 0.0
        df_result["is_member"] = False

        # 🌟 利用原始行索引，将干净样本算出的确定性结果精准刷回主输出表
        df_result.loc[df_field_clean.index, "is_member"] = (
            scores >= score_threshold
        )
        df_result.loc[df_field_clean.index, "prob"] = np.where(
            scores >= score_threshold, 1.0, 0.0
        )

        # 精准记录洗涤前后成员星数量变化变化，契合管线标准埋点规范
        n_before = len(df_all)
        n_after = int(df_result["is_member"].sum())
        logger.info(
            f"[Density Cut] Features: {n_features}D | Cutoff: {self.sigma_cutoff} sigma ({chi2_cutoff:.2f}) | "
            f"Target Col: {target_score_col} | Method: {self.method} | "
            f"Total: {n_before} -> Members: {n_after} | Washed out: {n_before - n_after}"
        )

        return df_result