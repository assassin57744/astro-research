# -*- coding: utf-8 -*-
"""
modules/membership/disambiguation/bayesian.py

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

from modules.membership.base import BaseDisambiguation


class BayesianGmmDisambiguation(BaseDisambiguation):
    """
    基于种子引导 & 高维贝叶斯对抗 EM 迭代的洗涤算子。

    解耦 fit() 与 predict() 阶段，支持局部天区（如 Orbital Tube 空间管）参数拟合
    与全域天区（Target Field）概率推理推导。
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
        tol: float = 1e-4,
        member_threshold: float = 0.2,
        enable_subsampling: bool = False,
        subsampling_limit: int = 100000,
        bayesian_ema_rtol: float = 1e-3,
        **kwargs,
    ):
        """
        初始化贝叶斯对抗洗涤算子。
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
        self.enable_subsampling = enable_subsampling
        self.subsampling_limit = subsampling_limit
        self.bayesian_ema_rtol = bayesian_ema_rtol

    def fit(
        self,
        df_field: pd.DataFrame,
        df_seeds: pd.DataFrame,
        features: List[str],
        use_density_prune: bool = False,
    ) -> Dict[str, Any]:
        """
        【拟合阶段】在输入天区（如 Channel B 空间管或全天区）内训练背景/成员 GMM 模型，
        并通过 EMA 迭代求解高信噪比的先验物理密度 f。

        Args:
            df_field (pd.DataFrame): 拟合基准天区数据（Channel A 传 df_target_final，Channel B 传 df_tube_full）。
            df_seeds (pd.DataFrame): 引导种子数据集。
            features (List[str]): 参与高维相空间拟合的特征列名。
            use_density_prune (bool): 是否启用 DBSCAN/HDBSCAN 种子集密度修剪。

        Returns:
            Dict[str, Any]: 包含训练好的 StandardScaler, GMM 模型及收敛先验权重 f 的参数包。
        """
        self.logger.info(
            f"🧬 [BayesianGMM.fit] 启动拟合内核 | 拟合天区基数: {len(df_field)} 颗 | 特征空间: {len(features)}D -> {features}"
        )

        df_field_clean = df_field.dropna(subset=features).copy()
        df_seeds_clean = (
            df_seeds.dropna(subset=features).drop_duplicates(subset=features).copy()
        )

        n_seeds = len(df_seeds_clean)
        if use_density_prune and n_seeds < self.dbscan_min_samples:
            self.logger.warning(
                f"⚠️ [内核警告] 有效种子星数量 ({n_seeds}) 低于修剪阈值 {self.dbscan_min_samples}。跳过密度修剪。"
            )
            use_density_prune = False

        # -----------------------------------------------------------------
        # 1. 数学空间归一化 (以当前拟合天区 df_tube 的方差结构为度量基准)
        # -----------------------------------------------------------------
        scaler = StandardScaler()
        scaler.fit(df_field_clean[features])

        X_field_scaled = scaler.transform(df_field_clean[features])
        X_seeds_scaled = scaler.transform(df_seeds_clean[features])

        # -----------------------------------------------------------------
        # 2. 🌌 构建局部背景似然场模型 (Field Model)
        # -----------------------------------------------------------------
        field_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        X_fit = X_field_scaled
        if self.enable_subsampling and len(X_field_scaled) > self.subsampling_limit:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_field_scaled), self.subsampling_limit, replace=False)
            X_fit = X_field_scaled[idx]
            self.logger.info(
                f"⚡ [场模型降采样] {len(X_field_scaled)} → {self.subsampling_limit} | 协方差类型: full"
            )
        field_model.fit(X_fit)

        # -----------------------------------------------------------------
        # 3. 🎯 种子集密度修剪与星团核心锁定 (Cluster Model)
        # -----------------------------------------------------------------
        X_core = X_seeds_scaled

        if use_density_prune:
            if self.cluster_algo == "hdbscan":
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
                    if "converted to Python scalars" in str(te) and self.hdbscan_eps > 0:
                        self.logger.warning(
                            "💥 HDBSCAN epsilon 发生树转换崩溃，降级 epsilon=0.0 重新解算..."
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

        cluster_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        cluster_model.fit(X_core)

        # -----------------------------------------------------------------
        # 4. 🔮 极大似然递归对抗收敛迭代 (EMA 推理求解 f)
        # -----------------------------------------------------------------
        total_stars = len(X_field_scaled)
        p_cl = np.exp(cluster_model.score_samples(X_field_scaled))
        p_fi = np.exp(field_model.score_samples(X_field_scaled))

        f_current = len(X_core) / total_stars
        self.logger.info(f"🔮 贝叶斯迭代初始成员密度估计 (Initial f): {f_current:.5f}")

        # 绝对下限：防止 EM 坍缩至 0，同时避免与 f_current 自锁
        f_floor = max(1.0 / total_stars, 0.0001)
        f_current = max(f_current, f_floor)
        self.logger.info(f"🔮 迭代保护底线 (f_floor): {f_floor:.6f}")

        for iteration in range(1, self.max_iter + 1):
            num = p_cl * f_current
            den = num + p_fi * (1.0 - f_current)
            probs = num / (den + 1e-15)

            f_new = max(np.mean(probs), f_floor)
            diff = abs(f_new - f_current)

            # 收敛判定：仅用相对变化（f 量级变化极大，绝对 tol 不适用）
            rel_diff = diff / max(f_current, f_floor)
            converged = rel_diff < self.bayesian_ema_rtol

            if iteration % 20 == 0 or converged:
                self.logger.info(
                    f"🔄 [贝叶斯迭代] 步数 {iteration:03d} | 当前空间权重 f = {f_new:.6f} "
                    f"| Δabs = {diff:.2e} | Δrel = {rel_diff:.3e}"
                )

            if converged:
                f_current = f_new  # 同步最新值再退出
                self.logger.info(
                    f"✅ [贝叶斯收敛] 极大似然递归成功收敛！迭代总步数 = {iteration} | 最终收敛权重 f = {f_current:.6f}"
                )
                break
            f_current = f_new
        else:
            self.logger.warning(
                f"⚠️ [收敛警告] 达到最大安全步数 ({self.max_iter}) 未完全收敛。delta: {diff:.2e}"
            )

        # 打包返回物理模型与收敛出的先验 f
        return {
            "scaler": scaler,
            "field_model": field_model,
            "cluster_model": cluster_model,
            "f_converged": f_current,
        }

    def predict(
        self,
        df_all: pd.DataFrame,
        model_params: Dict[str, Any],
        features: List[str],
    ) -> pd.DataFrame:
        """
        【推理阶段】使用拟合好的物理模型与先验密度 f，对目标天区（如全天区 223 万天体）进行后验概率推导。

        Args:
            df_all (pd.DataFrame): 待推导的目标天区数据集。
            model_params (Dict[str, Any]): fit() 阶段产出的模型参数包。
            features (List[str]): 参与洗涤的特征列名。

        Returns:
            pd.DataFrame: 包含 ['id', 'prob'] 的打标结果。
        """
        self.logger.info(
            f"🔮 [BayesianGMM.predict] 启动全域推理 | 目标天区: {len(df_all)} 颗 | 使用先验 f={model_params['f_converged']:.6f}"
        )

        scaler = model_params["scaler"]
        field_model = model_params["field_model"]
        cluster_model = model_params["cluster_model"]
        f_target = model_params["f_converged"]

        df_field_clean = df_all.dropna(subset=features).copy()
        X_field_scaled = scaler.transform(df_field_clean[features])

        # 算似然
        p_cl = np.exp(cluster_model.score_samples(X_field_scaled))
        p_fi = np.exp(field_model.score_samples(X_field_scaled))

        # 贝叶斯公式推导
        num = p_cl * f_target
        den = num + p_fi * (1.0 - f_target)
        probs = num / (den + 1e-15)

        df_result_clean = pd.DataFrame(
            {
                "id": df_field_clean["id"].to_numpy(), 
                "prob": probs,
                "p_cl_raw": p_cl,
                "p_tail_raw": p_fi
            }
        )

        # 🛡️ 特征残缺野星兜底隔离
        if len(df_result_clean) < len(df_all):
            df_all_ids = df_all[["id"]].copy()
            df_final = df_all_ids.merge(df_result_clean, on="id", how="left").fillna(
                {"prob": 0.0, "p_cl_raw": 0.0, "p_tail_raw": 0.0}
            )
            n_dropped = len(df_all) - len(df_result_clean)
            self.logger.info(
                f"🧹 管道补全：由于输入特征残缺，有 {n_dropped} 颗野星已在对齐阶段被安全隔离并归零。"
            )
            return df_final

        return df_result_clean

    def fit_predict(
        self,
        df_all: pd.DataFrame,
        df_seeds: pd.DataFrame,
        features: List[str],
        use_density_prune: bool = False,
    ) -> pd.DataFrame:
        """
        【快捷接口】拟合与推理天区一致时的快捷调用（如 Channel A Core 识别）。
        """
        model_params = self.fit(
            df_field=df_all,
            df_seeds=df_seeds,
            features=features,
            use_density_prune=use_density_prune,
        )
        return self.predict(df_all=df_all, model_params=model_params, features=features)