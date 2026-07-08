# -*- coding: utf-8 -*-
"""
modules/cluster_seed_extractor.py

🎯 星团种子提取器（无监督前置粗筛引擎）。
定位：Pipeline 的前置粗筛 Stage。
重构要点：
  1. 剥离了复杂的 KDE 数理算法和标准化逻辑，完全托管给 astro_membership 子包内的通用内核 helpers。
  2. 保持对外业务接口 `extract_seeds` 的完全兼容，确保上游主工作流零污染。
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN

# 引入物理/数理辅助模块中的通用核心自适应算子
from modules.astro_membership.helpers import calculate_adaptive_eps_kde


class ClusterSeedExtractor:
    """
    🎯 星团种子提取器（无监督粗筛引擎）。

    物理语义：
      1. 通过调用 helpers 中的高斯核密度估计（KDE）重采样，自适应感知天区本征恒星背景密度，动态解算最佳邻域半径 EPS。
      2. 运行高凝聚度硬截断（DBSCAN），大范围剔除银河系野星噪声，提取高纯度的成员星种子（Seeds），
         为下游高阶概率精筛模型（如 BayesianGmmDisambiguation）构建可靠的初始状态底座。
    """

    def __init__(self, cluster_profile: dict = None, **kwargs):
        """
        初始化粗筛引擎，100% 严格对接 config.py 中 CLUSTERS 字典的字段名。

        参数:
        - cluster_profile (dict): 来自 config.CLUSTERS["M45"] 等星团的专属配置 Profile。
        """
        self.logger = logging.getLogger("AstroPipeline.ClusterSeedExtractor")

        # 🛡️ 兜底群落防线
        profile = cluster_profile or {}

        # 1. 严格映射 config.py 中的 "cluster_algo" 字段
        self.cluster_algo = str(
            profile.get("cluster_algo", kwargs.get("cluster_algo", "dbscan"))
        ).lower()

        # 2. 严格映射 config.py 中的 "dbscan_eps" 字段（支持 "auto" 或 浮点数）
        self.dbscan_eps = profile.get("dbscan_eps", kwargs.get("dbscan_eps", "auto"))

        # 3. 严格映射 config.py 中的 "dbscan_min_samples" 字段作为基础物理凝聚门限
        self.min_pts = profile.get("dbscan_min_samples", kwargs.get("min_pts", 9))

        # 4. 严格映射 config.py 中的 "hdbscan_min_cluster_size"
        self.hdbscan_min_cluster_size = profile.get(
            "hdbscan_min_cluster_size", kwargs.get("hdbscan_min_cluster_size", 15)
        )

        # 5. 严格映射 config.py 中的 "hdbscan_min_samples"
        self.hdbscan_min_samples = profile.get(
            "hdbscan_min_samples", kwargs.get("hdbscan_min_samples", None)
        )

        # 固定蒙特卡洛 KDE 重采样模拟深度（默认 30 次以达到科学无偏收敛）
        self.num_simulations = kwargs.get("num_simulations", 30)

    def extract_seeds(
        self, field_stars_df: pd.DataFrame, features: list
    ) -> pd.DataFrame:
        """
        从输入的混乱初始星表中，自适应提取出高纯度的凝聚种子（支持 DBSCAN 与 HDBSCAN 策略）。

        参数:
        - field_stars_df (pd.DataFrame): 某星团靶场的初始全量观测星表（包含 id 和相空间各特征列）。
        - features (list): 参与计算的相空间特征列名列表。

        返回:
        - pd.DataFrame: 筛选出的高纯度星团种子星子集，保持原数据帧的所有物理列，并附加 'cluster_label' 列。
        """
        # 浅拷贝防止破坏外层原始星表
        working_df = field_stars_df.copy()

        # 1. 过滤由于高维特征残缺导致的 NaN 样本
        clean_mask = working_df[features].notna().all(axis=1)
        X_raw = working_df.loc[clean_mask, features].values

        if len(X_raw) == 0:
            self.logger.warning(
                "⚠️ 没有有效恒星样本满足相空间特征完整性，拒绝提取种子，返回空表。"
            )
            return pd.DataFrame(columns=field_stars_df.columns)

        # 统一执行特征空间归一化
        X_mean = X_raw.mean(axis=0)
        X_std = np.where(X_raw.std(axis=0) == 0, 1.0, X_raw.std(axis=0))
        X_scaled = (X_raw - X_mean) / X_std

        # --------------------------------==================--------------------------------
        # 🌟 2. 依据配置策略进行流控路由（DBSCAN 与 HDBSCAN 分流）
        # --------------------------------==================--------------------------------
        if self.cluster_algo == "hdbscan":
            self.logger.info(
                f"🧬 [HDBSCAN 粗筛] 启动变密度层级树剪枝 | "
                f"min_cluster_size: {self.hdbscan_min_cluster_size} | min_samples: {self.hdbscan_min_samples}"
            )

            # 🛡️ 物理微扰防线：注入微小噪声打破浮点数可能因高密度平局导致的树构建崩溃
            X_input = X_scaled.astype(np.float64, order="C") + np.random.normal(
                0, 1e-9, X_scaled.shape
            )

            db = HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,  # 完美透传 config.py 配置
                cluster_selection_method="eom",  # Excess of Mass 算法，完美拟合恒星团质量函数
                n_jobs=-1,
            ).fit(X_input)
            labels = db.labels_

        else:
            # 🚀 DBSCAN 核心分支：动态解析 dbscan_eps 字段
            # 经典 DBSCAN + Castro-Ginard (2018) 先验定标路径
            actual_eps = self.dbscan_eps

            if str(actual_eps).lower() == "auto":
                # 🌟 情况 A: 显式配置了 "auto"，调度重采样解算
                self.logger.info(
                    "⏳ [DBSCAN 自适应] 检测到 dbscan_eps='auto'，正在启动 KDE 银河系背景重采样模拟..."
                )
                actual_eps = calculate_adaptive_eps_kde(
                    X_raw=X_raw,
                    min_pts=self.min_pts,
                    num_simulations=self.num_simulations,
                )
                self.logger.info(
                    f"🤖 [KDE 动态定标成功] 逆向求解出物理噪声隔离边界: EPS = {actual_eps:.4f}"
                )
            else:
                # 🌟 情况 B: 配置了硬编码的物理数值（例如 0.25），直接解析并跳过重采样
                try:
                    actual_eps = float(actual_eps)
                    self.logger.info(
                        f"🎿 [DBSCAN 静态经验值] 跳过 KDE 模拟，直接采用配置指定的物理半径: EPS = {actual_eps:.4f}"
                    )
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"⚠️ 无法解析配置中的 dbscan_eps 值 '{actual_eps}'，强行回退至默认自适应 KDE 定标..."
                    )
                    actual_eps = calculate_adaptive_eps_kde(
                        X_raw,
                        min_pts=self.min_pts,
                        num_simulations=self.num_simulations,
                    )

            self.logger.info(
                f"🗜️ 正在运行无监督物理截断扫描 (EPS={actual_eps:.4f}, minPts={self.min_pts})..."
            )
            db = DBSCAN(eps=actual_eps, min_samples=self.min_pts, n_jobs=-1).fit(
                X_scaled
            )
            labels = db.labels_

        # --------------------------------==================--------------------------------
        # 3. 统一数据收拢与高纯核心析取
        # --------------------------------==================--------------------------------
        # 初始化并将聚类标签安全映射回工作 DataFrame 中
        working_df["cluster_label"] = -1
        working_df.loc[clean_mask, "cluster_label"] = labels

        if labels.max() < 0:
            self.logger.warning(
                "💥 核心拦截：当前天区未发现满足物理凝聚的超密度实体，前置种子库枯竭！"
            )
            return pd.DataFrame(columns=field_stars_df.columns)

        # 寻找点数最多的那个核心簇（非背景野星 -1）
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        best_cluster_label = unique_labels[np.argmax(counts)]

        # 析取
        df_seeds = working_df[working_df["cluster_label"] == best_cluster_label].copy()

        self.logger.info(
            f"🎯 [{self.cluster_algo.upper()} 种子粗筛成功] 目标星团标识: {best_cluster_label} | "
            f"大范围剔除银盘背景野星噪声后，成功出库高纯度初始种子星: {len(df_seeds)} 颗。"
        )

        return df_seeds
