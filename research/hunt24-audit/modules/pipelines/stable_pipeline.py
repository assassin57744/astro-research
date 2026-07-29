# modules/pipelines/stable_pipeline.py
import logging
import pandas as pd
import config as cfg
from modules.pipelines.core.prior_gmm import PriorGMM  # 👈 更新后的内核导入路径

class StablePipelineRunner:
    """稳定生产轨执行器 (PriorGMM 老轨)"""

    def __init__(self, db_instance, logger: logging.Logger = None):
        self.db = db_instance
        self.logger = logger or logging.getLogger(f"AstroPipeline.{self.__class__.__name__}")

    def run(self, ctx, df_all_field: pd.DataFrame, df_seed_field: pd.DataFrame) -> pd.DataFrame:
        """执行老轨 PriorGMM 拟合与预测。"""
        self.logger.warning("🔒 [Compute] 执行 PriorGMM 稳定生产轨")

        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id
        cluster_cfg["dim_mode"] = ctx.feature_space

        engine = PriorGMM(config=cluster_cfg)
        model_params = engine.fit(df_seed_field, df_all_field)
        return engine.predict(df_all_field, model_params)