# -*- coding: utf-8 -*-
# blind.py         # 策略 3: 全量自适应盲搜空间洗涤

import numpy as np
import logging
import pandas as pd  # 确保通用数据操作安全
from sklearn.mixture import GaussianMixture
from ..base import BaseDisambiguation

logger = logging.getLogger(__name__)


class BlindGmmDisambiguation(BaseDisambiguation):
    def __init__(self, n_components=3, covariance_type="full", **kwargs):
        # n_components 默认设为 3，分别用于捕捉银盘背景、弥散流群和紧凑星团过密度
        self.n_components = n_components
        self.covariance_type = covariance_type

        # 解耦提取业务相关的特征映射，防止算法层与硬编码字段（如赤道坐标 ra, dec）强耦合
        # 通过 kwargs 动态接收上游管线（如 config 映射）注入的通用物理字段名称
        self.spatial_cols = kwargs.get("spatial_cols", ["ra", "dec"])
        self.scale_col = kwargs.get("scale_col", "plx")
        self.default_roi = kwargs.get("default_roi", 4.0)

    def fit_predict(self, df_all, df_seeds, features):
        # 核心物理诉求：无先验引导（Blind Search），直接对全量靶场天区 df_all 进行多维相空间解算

        # 物理优化：利用天区中心的空间几何先验进行核密度粗筛（Warming Filter），剔除边缘高噪背景星，强行提升高维相空间的星团信号信噪比
        df_work = df_all.copy()

        # 🌟 保留并扩展：构建特征完整的数据子集，防止 NaN 引发 GMM 底层 C 算子崩溃
        df_clean = df_work[df_work[features].notna().all(axis=1)].copy()

        # 验证当前数据集是否包含配置的空间几何列，避免硬编码打破业务无关属性
        has_spatial = all(col in df_work.columns for col in self.spatial_cols)

        if has_spatial:
            # 动态提取空间物理中心（自适应适配各种坐标系，如赤道坐标 ra/dec 或银道坐标 l/b）
            centers = [df_work[col].mean() for col in self.spatial_cols]

            # 动态计算不同星团的空间粗筛半径（ROI Radius），彻底告别一刀切硬编码
            # 物理优先原则：若输入数据包含尺度特征（如视差），则依据物理潮汐外延（~15 pc 边界）动态推导投影角度
            if (
                self.scale_col in df_work.columns
                and df_work[self.scale_col].median() > 0
            ):
                median_scale = df_work[self.scale_col].median()
                dist_pc = 1000.0 / median_scale
                # 依据天体物理动力学外延半径（考虑潮汐尾迹），自适应转换为天球投影的角度边界
                roi_radius = np.clip((15.0 / dist_pc) * (180.0 / np.pi) * 2.5, 0.5, 8.0)
            else:
                # 缺乏尺度先验时的保守兜底策略（根据全量天区特征标准差自适应收拢至其中心核区）
                roi_radius = np.clip(
                    np.std(df_work[self.spatial_cols[0]]) * 0.5, 1.0, self.default_roi
                )

            # 计算恒星到物理中心的高维几何距离，建立视场截断掩码以优化无监督多成分拟合的样本信噪比
            sq_dist = np.zeros(len(df_work))
            for i, col in enumerate(self.spatial_cols):
                sq_dist += (df_work[col] - centers[i]) ** 2
            spatial_mask = np.sqrt(sq_dist) <= roi_radius

            # 🌟 修复与对齐：基于特征完整的 df_clean 提取参与拟合的物理矩阵，对齐空间和相空间
            clean_spatial_mask = (
                spatial_mask.loc[df_clean.index]
                if isinstance(spatial_mask, pd.Series)
                else spatial_mask[df_work[features].notna().all(axis=1)]
            )
            X_fit_raw = df_clean.loc[clean_spatial_mask, features].values

            # 【增量日志】精确追踪大天区高噪声样本的空间降维状态与自适应视场收拢尺度
            logger.info(
                f"[Blind Search ROI] Geometry Centers: {list(np.round(centers, 4))} | Calculated ROI Radius: {roi_radius:.4f} deg | Fit Subsample: {len(X_fit_raw)} / {len(df_work)}"
            )
        else:
            X_fit_raw = df_clean[features].values
            clean_spatial_mask = np.ones(len(df_clean), dtype=bool)
            roi_radius = 0.0
            # 【增量日志】当缺失空间特征时及时发出警告，并清晰指示算法退化流向
            logger.warning(
                "[Blind Search ROI] Missing spatial columns for ROI warm start. Falling back to full region un-weighted fitting."
            )

        # 🌟 修复与对齐：统一从特征干净的底座提取全量计算矩阵
        X_all_raw = df_clean[features].values

        # 🌟 核心增量逻辑：特征标度标准化（对齐整个 pipeline 的数学度量空间，防止大方差特征霸权）
        X_mean = X_all_raw.mean(axis=0)
        X_std = np.where(X_all_raw.std(axis=0) == 0, 1.0, X_all_raw.std(axis=0))
        X_fit = (X_fit_raw - X_mean) / X_std
        X_all = (X_all_raw - X_mean) / X_std

        # 构建全量空间的无监督概率多成分混合模型（使用提升信噪比后的 X_fit 进行训练）
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=42,
        )
        gmm.fit(X_fit)

        # 背景与星团的自适应剥离：
        # 物理准则：星团在相空间中表现为极高凝聚度（协方差矩阵行列式极小，即体极小）
        # 计算每个高斯成分的协方差矩阵行列式（或其对数），以此评估成分的致密程度
        if self.covariance_type == "full":
            dets = [np.linalg.det(cov) for cov in gmm.covariances_]
        elif self.covariance_type == "diag":
            dets = [np.prod(cov) for cov in gmm.covariances_]
        else:
            # 兼容其他协方差类型
            dets = [
                (
                    np.linalg.det(gmm.covariances_[i])
                    if gmm.covariances_.ndim == 3
                    else np.linalg.det(gmm.covariances_)
                )
                for i in range(self.n_components)
            ]

        # 锁定体积最小（行列式最小）、最紧凑的核心高斯分量作为星团实体成分
        cluster_component_idx = np.argmin(dets)

        # 【增量日志】打印所有高斯成分的超体积测度，以便对无监督解算下的凝聚度一致性进行科学审计
        logger.debug(
            f"[Blind Search Component Volume] Covariance Determinants for all {self.n_components} components: {[f'{d:.2e}' for d in dets]}"
        )

        # 计算全量恒星属于各个成分的后验响应概率（Responsibility）
        # responsibilities 形状为 (n_samples, n_components)
        responsibilities = gmm.predict_proba(X_all)

        # 提取目标星团核心分量的归属概率作为成员概率底座
        prob_cluster = responsibilities[:, cluster_component_idx]

        # 维持与 BaseDisambiguation 接口严格一致的输出结构
        df_result = df_all.copy()

        # 初始化默认输出状态
        df_result["prob"] = 0.0
        df_result["is_member"] = False

        # 🌟 利用原始行索引，将计算出的有效响应精准同步回主输出表，彻底规避 NaN 带来的错位和 ValueError
        df_result.loc[df_clean.index, "prob"] = np.clip(prob_cluster, 0.0, 1.0)

        # 基于响应概率建立初步硬截断（可由下游或配置联合控制，默认取响应主导权或特定阈值）
        # 这里采用最大后验概率（MAP）物理准则：若当前星团分量响应度在所有成分中最高，则初步判定为数组成员
        df_result.loc[df_clean.index, "is_member"] = (
            np.argmax(responsibilities, axis=1) == cluster_component_idx
        )

        # 标准管线埋点与日志规范，升级日志以支持追踪自适应 ROI 半径
        n_before = len(df_all)
        n_members = int(df_result["is_member"].sum())
        logger.info(
            f"[Blind Search] Components: {self.n_components} | Adaptive ROI Radius: {roi_radius:.2f} deg | Cluster Comp Index: {cluster_component_idx} | Total Tensors: {n_before} -> Extracted Candidates: {n_members}"
        )

        return df_result
