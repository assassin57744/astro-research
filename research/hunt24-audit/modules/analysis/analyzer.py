"""
analyzer.py - AstroAnalyzer 门面主控类
专注于 Phase 6 的生命周期编排，调度纯 Python 绘图引擎与评估器。
"""
import logging
import numpy as np
from pathlib import Path

import config as cfg
from modules.analysis.plotter import AstroPlotter
from modules.analysis.evaluator import AuditEvaluator


class _PCAMeta:
    """轻量级 PCA 代理，供解耦后的绘图引擎反解管顶点。"""
    __slots__ = ("cluster_center_", "tube_width_", "_components", "_mean")

    def __init__(self, center, components, mean, tube_width):
        self.cluster_center_ = center
        self._components = components
        self._mean = mean
        self.tube_width_ = tube_width

    def inverse_transform(self, X):
        return np.dot(X, self._components) + self._mean


class AstroAnalyzer:
    """Phase 6 报告与可视化控制器（纯 Python 实现）"""

    def __init__(self, db_instance, ctx=None, target_cluster=None, target_category=None, mode="3d"):
        self.db = db_instance
        if ctx is not None:
            self.ctx = ctx
            self.target_cluster = ctx.cluster_id
            self.target_category = ctx.category
            self.mode = ctx.feature_space
        else:
            self.ctx = None
            self.target_cluster = target_cluster
            self.target_category = target_category
            self.mode = mode

        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        self.plotter = AstroPlotter(logger=self.logger)
        self.evaluator = AuditEvaluator(db_instance=self.db, logger=self.logger)

    def run_phase6(self, post_result: dict, audit_result: dict) -> dict:
        """Phase 6 统一入口（由 workflow._report_phase 委派调用）"""
        from modules.reporter import render_final_report
        from modules.result_logger import resolve_eps, resolve_min_samples, update_dbscan_csv

        ctx = self.ctx
        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id

        # 1. 动态补充参数配置
        gmm_cfg = ctx.state.gmm_config.copy()
        if "eps" in ctx.algo_params:
            gmm_cfg["DBSCAN_EPS"] = ctx.algo_params["eps"]
        elif ctx.state.computed_dbscan_eps:
            gmm_cfg["DBSCAN_EPS"] = ctx.state.computed_dbscan_eps
        if "min_samples" in ctx.algo_params:
            gmm_cfg["DBSCAN_MIN_SAMPLES"] = ctx.algo_params["min_samples"]

        # 2. 渲染文本报告
        summary = render_final_report(
            ctx.cluster_id, ctx.category, ctx.feature_space, ctx.algorithm,
            cluster_cfg, gmm_cfg, post_result, audit_result,
            audit_result.get("deep_stats_pg_only", {}),
            audit_result.get("deep_stats_ref_only", {}),
            audit_result.get("deep_stats_matched", {}),
            audit_result.get(f"deep_stats_{ctx.category}", {}),
            audit_result.get("deep_stats_pg_algo", {}),
            self.logger,
        )

        # 3. 更新实验结果 CSV
        try:
            csv_path = cfg.RESULTS_DIR / "实验结果记录(DBSCAN).csv"
            eps = ctx.state.computed_dbscan_eps or resolve_eps(ctx.algo_params, cluster_cfg, gmm_cfg)
            min_s = resolve_min_samples(ctx.algo_params, cluster_cfg, gmm_cfg)
            update_dbscan_csv(
                csv_path=csv_path, cluster=ctx.cluster_id, eps=eps, min_samples=min_s,
                cross_stats=audit_result.get("stats", {}),
                deep_stats_matched=audit_result.get("deep_stats_matched", {}),
                deep_stats_pg_only=audit_result.get("deep_stats_pg_only", {}),
                deep_stats_ref_only=audit_result.get("deep_stats_ref_only", {}),
            )
        except Exception:
            self.logger.warning("[ResultLog] 记录实验 CSV 出现异常 (不影响主流程)", exc_info=True)

        # 4. 可视化诊断渲染（纯 Python 原生逻辑）
        if not ctx.skip_viz:
            self.run_all_diagnostics()

        return summary

    def run_all_diagnostics(self):
        """调度纯 Python 原生渲染所有 3 张诊断图表（增加真实数据兼容性）。"""
        master = self.ctx.state.master_table
        self.logger.info(f"🎨 [Phase 6] 启动 Python 原生可视化分析: {master}")

        state = self.ctx.state
        
        # 1. 查询 master 表实际拥有的所有列名 (转为小写比较)
        cols_df = self.db.query(f"SELECT * FROM {master} LIMIT 1")
        existing_cols = set(cols_df.columns)

        # 2. 空间管掩模图 (Spatial Tube Mask)
        if state.tube_pca_center is not None and {"ra", "dec", "in_tube"}.issubset(existing_cols):
            df_all = self.db.query(f"SELECT ra, dec FROM {master} WHERE ra IS NOT NULL AND dec IS NOT NULL")
            df_seeds = self.db.query(f"SELECT ra, dec FROM {master} WHERE seed_type IS NOT NULL AND seed_type != ''")
            df_tube = self.db.query(f"SELECT ra, dec FROM {master} WHERE in_tube = TRUE OR in_tube = 'true'")

            if not df_tube.empty:
                pca_proxy = _PCAMeta(
                    center=np.array(state.tube_pca_center),
                    components=state.tube_pca_components,
                    mean=state.tube_pca_mean,
                    tube_width=state.tube_width,
                )
                out_path = self.plotter.render_tube_mask(
                    df_all, df_seeds, df_tube, pca_proxy,
                    self.target_cluster, state.tube_length, state.tube_width
                )
                self.logger.info(f"💾 [{self.target_cluster}] 空间管掩模图已保存至: {out_path}")
        else:
            self.logger.info("⏩ [Phase 6] 跳过空间管掩模图（PCA 元数据或坐标列缺失）。")

        # 3. 三通道概率分布直方图 (Channel Probabilities)
        prob_cols = {"prob", "core_prob", "tail_prob"}
        if prob_cols.issubset(existing_cols):
            df_res = self.db.query(f"SELECT prob, core_prob, tail_prob FROM {master}")
            out_path = self.plotter.render_channel_probs(df_res, self.target_cluster)
            self.logger.info(f"💾 [{self.target_cluster}] 概率分布直方图已保存至: {out_path}")
        else:
            missing = prob_cols - existing_cols
            self.logger.info(f"⏩ [Phase 6] 跳过概率分布直方图（缺失列: {missing}）。")

        # 4. 交叉比对诊断三连图 (Cross-Match Diagnostics)
        x_tag_col = cfg.MASTER_COLS.get("X_MATCH", "x_match_tag")
        diag_cols = {"ra", "dec", "pmra", "pmdec", "mag", "color", x_tag_col}
        
        if diag_cols.issubset(existing_cols):
            # 将缺失的 x_match_tag 填补为 Field，避免绘图丢点
            df_cross = self.db.query(
                f"SELECT ra, dec, pmra, pmdec, mag, color, COALESCE({x_tag_col}, 'Field') as {x_tag_col} "
                f"FROM {master} WHERE ra IS NOT NULL AND dec IS NOT NULL"
            )
            star_cluster = getattr(self.ctx, "star_cluster", None)
            out_path = self.plotter.render_cross_match_diagnostics(
                df_cross, self.target_cluster, star_cluster=star_cluster
            )
            self.logger.info(f"💾 [{self.target_cluster}] 交叉比对诊断三连图已保存至: {out_path}")
        else:
            missing = diag_cols - existing_cols
            self.logger.info(f"⏩ [Phase 6] 跳过交叉比对诊断图（缺失列: {missing}）。")