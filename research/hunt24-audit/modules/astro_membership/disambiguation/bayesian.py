# -*- coding: utf-8 -*-
"""
modules/astro_membership/disambiguation/bayesian.py

独立子包策略：基于双模型贝叶斯对抗与极大似然 EMA 迭代的成员星歧义消除算法。
专门用于处理银盘背景复杂、噪声严重的星团靶场（如 M44, M67）。
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from modules.astro_membership.base import BaseDisambiguation


class BayesianGmmDisambiguation(BaseDisambiguation):
    """
    基于种子引导 & 高维贝叶斯对抗 EM 迭代的洗涤算子。

    去除了旧版本中一阶矩中心化等特征工程逻辑，专注于纯粹矩阵上的密度修剪与似然估计收敛。
    """

    def __init__(
        self,
        cluster_algo: str = "dbscan",
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 100,
        hdbscan_min_cluster_size: int = 15,
        hdbscan_min_samples: Any = None,
        hdbscan_eps: float = 0.1,
        max_iter: int = 250,
        tol: float = 1e-6,
        member_threshold: float = 0.2,
        **kwargs,
    ):
        """
        初始化贝叶斯对抗洗涤算子。

        所有控制超参均由主管线根据 config.CLUSTERS 的定义动态透传。
        """
        self.logger = logging.getLogger(f"AstroPipeline.AstroMembership.{__name__}")

        self.cluster_algo = cluster_algo.lower()
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_eps = hdbscan_eps

        self.max_iter = max_iter
        self.tol = tol
        self.member_threshold = member_threshold

    def fit_predict(
        self, df_all: pd.DataFrame, df_seeds: pd.DataFrame, features: List[str]
    ) -> pd.DataFrame:
        """
        洗涤接口：消除背景歧义并为全量天体打上成员概率标签。

        Args:
            df_all (pd.DataFrame): 经过特征工程前置对齐后的全量天区观测数据。
            df_seeds (pd.DataFrame): 经过无监督或外部物理先验初筛的高纯度种子数据集。
            features (List[str]): 参与高维相空间拟合的特征列名定义（由 MEMBERSHIP_FEATURES 指定）。

        Returns:
            pd.DataFrame: 包含两列 ['id', 'prob'] 的洗涤打标结果。
        """
        self.logger.info(
            f"🧬 [BayesianGMM] 启动高维洗涤内核 | 特征空间维度: {len(features)}D -> {features}"
        )

        # --------------------------------==================--------------------------------
        # 1. 数据完整性高纯洗涤 (不再包含任何物理坐标转换)
        # --------------------------------==================--------------------------------
        df_field_clean = df_all.dropna(subset=features).copy()
        df_seeds_clean = (
            df_seeds.dropna(subset=features).drop_duplicates(subset=features).copy()
        )

        n_seeds = len(df_seeds_clean)
        if n_seeds < self.dbscan_min_samples:
            self.logger.warning(
                f"⚠️ [内核警告] 有效种子星数量 ({n_seeds}) 低于修剪阈值 {self.dbscan_min_samples}。将跳过密度修剪。"
            )
            use_density_prune = False
        else:
            use_density_prune = True

        use_density_prune = False  # 强制关闭密度修剪，直接使用全量种子集拟合 GMM

        # --------------------------------==================--------------------------------
        # 2. 数学空间归一化 (Standardization)
        # --------------------------------==================--------------------------------
        scaler = StandardScaler()
        # 统一使用全域背景数据的方差结构作为数学空间的度量基准
        scaler.fit(df_field_clean[features])

        X_field_scaled = scaler.transform(df_field_clean[features])
        X_seeds_scaled = scaler.transform(df_seeds_clean[features])

        # --------------------------------==================--------------------------------
        # 3. 🌌 构建全域背景似然场模型 (Field Model)
        # --------------------------------==================--------------------------------
        field_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        field_model.fit(X_field_scaled)

        # --------------------------------==================--------------------------------
        # 4. 🎯 稳健密度修剪：剥离种子集中的银盘噪声，锁定星团核心 (Cluster Model)
        # --------------------------------==================--------------------------------
        X_core = X_seeds_scaled  # 默认回退状态

        if use_density_prune:
            if self.cluster_algo == "hdbscan":
                # 🛡️ 注入 1e-8 极微小高斯噪声以打破高维空间由于浮点数精度导致的数值平局，防止 HDBSCAN 崩溃
                X_input = X_seeds_scaled.astype(np.float64, order="C")
                X_input += np.random.normal(0, 1e-8, X_input.shape)

                try:
                    db = HDBSCAN(
                        min_cluster_size=self.hdbscan_min_cluster_size,
                        min_samples=self.hdbscan_min_samples,
                        cluster_selection_epsilon=self.hdbscan_eps,
                        copy=True,
                    ).fit(X_input)
                except TypeError as te:
                    if (
                        "converted to Python scalars" in str(te)
                        and self.hdbscan_eps > 0
                    ):
                        self.logger.warning(
                            "💥 HDBSCAN epsilon 发生树转换崩溃，强行降级 epsilon=0.0 重新解算..."
                        )
                        db = HDBSCAN(
                            min_cluster_size=self.hdbscan_min_cluster_size,
                            min_samples=self.hdbscan_min_samples,
                            cluster_selection_epsilon=0.0,
                            copy=True,
                        ).fit(X_input)
                    else:
                        raise te
                algo_name = "HDBSCAN"
            else:
                self.logger.info(
                    f"🧬 [DBSCAN 粗筛] 启动密度修剪 | eps: {self.dbscan_eps} | min_samples: {self.dbscan_min_samples}"
                )
                db = DBSCAN(
                    eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
                ).fit(X_seeds_scaled)
                algo_name = "DBSCAN"

            labels = db.labels_

            if np.all(labels == -1):
                self.logger.warning(
                    f"⚠️ [内核拦截] {algo_name} 密度溃缩，未捕获凝聚核心，回退至全量种子星。"
                )
            else:
                unique_labels, counts = np.unique(
                    labels[labels != -1], return_counts=True
                )
                best_cluster_idx = np.argmax(counts) if len(counts) > 0 else -1
                n_best = counts[best_cluster_idx] if best_cluster_idx != -1 else 0

                # 🛡️ 安全拦截：若核心样本占比低于 20%，说明发生了过度污染或严重的密度过度切分
                if n_best < (n_seeds * 0.2):
                    self.logger.warning(
                        f"⚠️ [内核解构] {algo_name} 捕获的核心过小 ({n_best}/{n_seeds})，回退至全量种子星。"
                    )
                else:
                    best_cluster = unique_labels[best_cluster_idx]
                    cluster_mask = labels == best_cluster
                    X_core = X_seeds_scaled[cluster_mask]
                    self.logger.info(
                        f"🎯 [核心锁定] {algo_name} 修剪成功：剔除野星 {np.sum(labels == -1)} 颗，沉淀核心样本 {len(X_core)} 颗。"
                    )

        # 拟合高纯度星团成员先验高斯
        cluster_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        cluster_model.fit(X_core)

        # --------------------------------==================--------------------------------
        # 5. 🔮 运动学/动力学混合空间递归对抗收敛迭代 (EMA 推理)
        # --------------------------------==================--------------------------------
        total_stars = len(X_field_scaled)
        p_cl = np.exp(cluster_model.score_samples(X_field_scaled))
        p_fi = np.exp(field_model.score_samples(X_field_scaled))

        # 初始成员星空间密度混合权重期望 f
        f_current = len(X_core) / total_stars
        self.logger.info(f"🔮 贝叶斯迭代初始成员密度估计 (Initial f): {f_current:.5f}")

        probs = np.zeros(total_stars)

        f_floor = f_current * 0.2  # 🌟 动态护栏：根据每个星团的初始本征规模，自适应定制保护底线

        for iteration in range(1, self.max_iter + 1):
            num = p_cl * f_current
            den = num + p_fi * (1.0 - f_current)
            probs = num / (den + 1e-15)  # 注入微小 eps 防止分母零溃缩

            f_new = max(np.mean(probs), f_floor) # 保护底线，防止过度收敛导致成员权重被完全抹平
            diff = abs(f_new - f_current)

            if iteration % 20 == 0 or diff < self.tol:
                self.logger.debug(
                    f"🔄 [贝叶斯迭代] 步数 {iteration:03d} | 当前空间权重 f = {f_new:.6f} | delta = {diff:.2e}"
                )

            if diff < self.tol:
                self.logger.info(
                    f"✅ [贝叶斯收敛] 极大似然递归成功收敛！迭代总步数 = {iteration} | 最终收敛权重 f = {f_new:.6f}"
                )
                break
            f_current = f_new
        else:
            self.logger.warning(
                f"⚠️ [收敛警告] 达到最大安全步数 ({self.max_iter}) 未完全收敛。delta: {diff:.2e}"
            )

        # --------------------------------==================--------------------------------
        # 6. 打包打标输出结果
        # --------------------------------==================--------------------------------
        # 仅针对干净的数据帧打标
        df_result_clean = pd.DataFrame(
            {"id": df_field_clean["id"].to_numpy(), "prob": probs}
        )

        # 🛡️ 鲁棒性防线：如果有因 NaN 被清洗掉的野星，通过外连接补充回原始输入帧中，概率强制赋 0.0
        if len(df_result_clean) < len(df_all):
            df_all_ids = df_all[["id"]].copy()
            df_final = df_all_ids.merge(df_result_clean, on="id", how="left").fillna(
                {"prob": 0.0}
            )
            n_dropped = len(df_all) - len(df_result_clean)
            self.logger.info(
                f"🧹 管道补全：由于输入特征残缺，有 {n_dropped} 颗野星已在对齐阶段被安全隔离并归零。"
            )
            return df_final

        return df_result_clean
