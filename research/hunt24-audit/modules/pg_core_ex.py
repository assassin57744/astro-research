import logging
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config as cfg
from config import MEMBER_SAMPLE_THRESHOLD, STD_COLS
from utils.decorators import astro_checkpoint


@dataclass
class GMMModelParamsEx:
    """
    模型参数容器（封测版）：保存从 PriorGMMEx 训练阶段提取的所有物理先验知识。

    Attributes:
        cluster_model (GaussianMixture): 拟合好的星团成员高斯分布模型。
        field_model (GaussianMixture): 拟合好的背景场高斯分布模型。
        scaler (StandardScaler): 特征标准化处理器，用于保持推理时的一致性。
        n_core_samples (int): 参与星团核心模型训练的有效样本数。
        dim_mode (str): 特征空间维度模式（如 '3d', '6d_p'）。
        features_used (list): 实际参与计算的特征列名。
        center_coords (dict): 用于中心化修正的各维度锚点值。
    """

    cluster_model: GaussianMixture
    field_model: GaussianMixture
    scaler: StandardScaler
    n_core_samples: int
    dim_mode: str
    features_used: list
    center_coords: dict = field(default_factory=dict)


class PriorGMMEx:
    """
    种子星引导的高维先验迭代成员判定算法（PriorGMM 实验性增强版）。

    该算法采用无状态设计，支持从 3D 到 6D 的全相空间特征转换，
    通过高质种子星（Seeds）提取星团核心分布先验，并在全域样本中通过递归极大似然估计实现成员概率收敛。
    """

    def __init__(self, config=None):
        """初始化测试内核。"""
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}.Experimental")
        self.config = config or {}

        # 确定特征维度模式，无缝对接升级后的特征工程
        self.dim_mode = self.config.get("dim_mode", "3d").lower()

        # 动态映射 6 大特征空间对应的特征列名，严格匹配大一统规范（短键名 plx 与 rv）
        feature_map = {
            "2d": ["pmra", "pmdec"],
            "3d": ["pmra", "pmdec", "plx"],
            "5d": ["ra", "dec", "pmra", "pmdec", "plx"],
            "6d_o": ["ra", "dec", "pmra", "pmdec", "plx", "rv"],
            "5d_h": ["l", "b", "pm_l_cosb", "pm_b", "plx"],
            "3d_v": ["U", "V", "W"],
            "6d_p": ["X", "Y", "Z", "U", "V", "W"],
        }

        if self.dim_mode not in feature_map:
            self.logger.error(
                f"❌ [🧪测试核] 无法识别未知的 dim_mode: '{self.dim_mode}'"
            )
            raise ValueError(f"❌ 未知的特征维度模式: '{self.dim_mode}'")

        self.features = feature_map[self.dim_mode]

        # 提取聚类超参数
        self.dbscan_eps = self.config.get("dbscan_eps", 0.3)
        self.dbscan_min_samples = self.config.get("dbscan_min_samples", 10)

        self.logger.info(
            f"🧪 [PriorGMMEx] 实验性内核加载成功 | 模式: {self.dim_mode.upper()} | 维度轴: {self.features}"
        )

    def _apply_adaptive_centering(
        self, df: pd.DataFrame, centers_dict: dict = None
    ) -> tuple:
        """
        [私有方法] 全相空间自适应坐标中心化处理。
        
        通过减去中值，消除绝对坐标（如 RA/Dec 或大数值 X/Y/Z）对 StandardScaler 方差拉伸的负面影响，
        提升标准化后特征空间的数值稳定性。
        """
        df_copy = df.copy()
        out_centers = {}

        # 需要进行一阶矩中心化处理的特征对
        coordinate_pairs = [("ra", "dec"), ("l", "b"), ("x", "y", "z")]

        for group in coordinate_pairs:
            if all(col in self.features for col in group):
                for col in group:
                    if centers_dict and col in centers_dict:
                        center_val = centers_dict[col]
                    else:
                        center_val = float(df_copy[col].median())

                    df_copy[col] = df_copy[col] - center_val
                    out_centers[col] = center_val

                self.logger.debug(
                    f"🧪 [物理核中心化] 坐标组 {group} 修正完成。坐标锚点: { {k: round(v, 4) for k, v in out_centers.items()} }"
                )
                break  # 匹配到一个坐标组即可

        return df_copy, out_centers

    def fit(self, df_seeds: pd.DataFrame, df_field: pd.DataFrame) -> GMMModelParamsEx:
        """
        训练接口：利用种子星与全域背景构建高斯混合先验模型。

        Args:
            df_seeds (pd.DataFrame): 经过高质量筛选的种子星集合。
            df_field (pd.DataFrame): 包含背景与成员的全域背景数据。

        Returns:
            GMMModelParamsEx: 包含模型状态的参数对象。
        """
        self.logger.info(
            f"--- 🧪 [PriorGMMEx] 开始拟合先验模型 ({self.dim_mode.upper()}) ---"
        )

        # --------------------------------==================--------------------------------
        # 1. 字段非空清洗
        # --------------------------------==================--------------------------------
        n_field_orig = len(df_field)
        df_field = df_field.dropna(subset=self.features)
        if len(df_field) < n_field_orig:
            self.logger.warning(f" 字段清洗：剔除全域背景中 {n_field_orig - len(df_field)} 颗不完备星。")

        n_seeds_orig = len(df_seeds)
        df_seeds = df_seeds.dropna(subset=self.features)
        if len(df_seeds) < n_seeds_orig:
            self.logger.warning(f"🧹 字段清洗：剔除种子星集中 {n_seeds_orig - len(df_seeds)} 颗不完备星。")

        self.logger.info(f"📊 训练样本集: 种子星={len(df_seeds)} 颗 | 全域背景={len(df_field)} 颗")

        if len(df_seeds) < self.dbscan_min_samples:
            raise ValueError(
                f"❌ [内核溃缩] 有效种子星数量 ({len(df_seeds)}) 低于 DBSCAN 启动阈值 {self.dbscan_min_samples}"
            )

        # --------------------------------==================--------------------------------
        # 2. 自适应空间零点锚定与特征标准化
        # ------------------------------------------------==--------------------------------
        df_field_proc, center_coords = self._apply_adaptive_centering(df_field)
        df_seeds_proc, _ = self._apply_adaptive_centering(
            df_seeds, centers_dict=center_coords
        )

        scaler = StandardScaler()
        scaler.fit(df_field_proc[self.features])
        X_field_scaled = scaler.transform(df_field_proc[self.features])
        X_seeds_scaled = scaler.transform(df_seeds_proc[self.features])

        # --------------------------------==================--------------------------------
        # 3. 构建全域背景模型 (Field Model)
        # --------------------------------==================--------------------------------
        field_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        field_model.fit(X_field_scaled)
        self.logger.debug("🌌 全域背景场模型 (Field Model) 拟合完成。")

        # --------------------------------==================--------------------------------
        # 4. 稳健密度修剪与星团核心先验模型 (Cluster Model)
        # --------------------------------==================--------------------------------
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(
            X_seeds_scaled
        )
        labels = db.labels_

        if np.all(labels == -1):
            self.logger.warning(
                f"⚠️ [内核拦截] DBSCAN 发生高维密度溃缩。将跳过密度修剪，回退为全量种子星构建分布。"
            )
            X_core = X_seeds_scaled
        else:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) == 0:
                X_core = X_seeds_scaled
            else:
                best_cluster = unique_labels[np.argmax(counts)]
                X_core = X_seeds_scaled[labels == best_cluster]
                self.logger.info(
                    f"🎯 [内核拦截] DBSCAN 完成：识别到噪声 {np.sum(labels == -1)} 颗，提取核心样本 {len(X_core)} 颗。"
                )

        cluster_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        cluster_model.fit(X_core)
        self.logger.info(
            f"✨ [内核验证] 星团先验模型拟合成功。归一化中心偏移: \n{ {f: round(v, 4) for f, v in zip(self.features, cluster_model.means_[0])} }"
        )

        return GMMModelParamsEx(
            cluster_model=cluster_model,
            field_model=field_model,
            scaler=scaler,
            n_core_samples=len(X_core),
            dim_mode=self.dim_mode,
            features_used=self.features,
            center_coords=center_coords,
        )

    def predict(
        self,
        df_predict: pd.DataFrame,
        params: GMMModelParamsEx,
        max_iter: int = 250,
        tol: float = 1e-6,
    ) -> pd.DataFrame:
        """
        推理接口：执行基于先验的递归极大似然收敛计算。

        Args:
            df_predict (pd.DataFrame): 待判定的天体数据帧。
            params (GMMModelParamsEx): 训练好的模型参数对象。
            max_iter (int): 最大收敛迭代步数。
            tol (float): 收敛停止的容差阈值。

        Returns:
            pd.DataFrame: 包含 ID 和计算得出的成员概率（prob）的结果集。
        """
        total_stars = len(df_predict)
        self.logger.info(f"--- 🧪 [PriorGMMEx] 启动递归推理流程 | 目标量: {total_stars} ---")

        # --------------------------------==================--------------------------------
        # 1. 物理坐标对齐与标准化映射
        # --------------------------------==================--------------------------------
        df_proc, _ = self._apply_adaptive_centering(
            df_predict, centers_dict=params.center_coords
        )
        X_scaled = params.scaler.transform(df_proc[params.features_used])

        # --------------------------------==================--------------------------------
        # 2. 计算高维似然（Likelihood）
        # --------------------------------==================--------------------------------
        p_cl = np.exp(params.cluster_model.score_samples(X_scaled))
        p_fi = np.exp(params.field_model.score_samples(X_scaled))

        # --------------------------------==================--------------------------------
        # 3. 先验混合空间递归优化迭代
        # --------------------------------==================--------------------------------
        f_current = params.n_core_samples / total_stars
        self.logger.info(f"🔮 初始星团成员密度比估计 (Initial f): {f_current:.5f}")

        iteration = 0
        probs = np.zeros(total_stars)

        for i in range(max_iter):
            iteration = i + 1
            num = p_cl * f_current
            den = num + p_fi * (1.0 - f_current)
            probs = num / (den + 1e-15)

            f_new = np.mean(probs)
            diff = abs(f_new - f_current)

            if iteration % 10 == 0 or diff < tol:
                self.logger.debug(
                    f"🔄 [测试核迭代] 步数 {iteration:03d} | 空间密度 f = {f_new:.6f} | delta = {diff:.2e}"
                )

            if diff < tol:
                self.logger.info(
                    f"✅ [测试核收敛] 极大似然递归成功收敛！总步数 = {iteration} | 最终混合权重 f = {f_new:.6f}"
                )
                break
            f_current = f_new
        else:
            self.logger.warning(
                f"⚠️ [测试核警告] 达到最大安全迭代步数 ({max_iter}) 仍未完全收敛。残留 delta: {diff:.2e}"
            )

        # --------------------------------==================--------------------------------
        # 4. 成员星最终划分统计与数据收拢
        # --------------------------------==================--------------------------------
        high_prob_mask = probs > MEMBER_SAMPLE_THRESHOLD
        n_members = np.sum(high_prob_mask)
        self.logger.info(
            f"✨ [测试核收工] 判定任务顺利结束 | 成功掘出星团高置信度成员星: {n_members} 颗。"
        )
        self.logger.info(f"--- 🧪 PriorGMMEx 管道内核运行结束 ---")

        return pd.DataFrame(
            {
                STD_COLS.get("ID", "id"): df_predict["id"].to_numpy(),
                STD_COLS.get("PROB", "prob"): probs,
            }
        )
