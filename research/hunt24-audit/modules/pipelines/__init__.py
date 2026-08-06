# modules/pipelines/__init__.py
import logging
import pandas as pd

def run_pipeline(ctx, db_instance, df_all_field: pd.DataFrame, df_seed_field: pd.DataFrame) -> pd.DataFrame:
    """根据算法配置自动分发至稳定轨或实验轨执行器"""
    logger = logging.getLogger("AstroPipeline.PipelineDispatcher")
    use_experimental = ctx.state.gmm_config.get("use_experimental", False)

    if use_experimental:
        from modules.pipelines.exp_pipeline import ExperimentalPipelineRunner
        runner = ExperimentalPipelineRunner(db_instance=db_instance, logger=logger)
    else:
        from modules.pipelines.stable_pipeline import StablePipelineRunner
        runner = StablePipelineRunner(db_instance=db_instance, logger=logger)

    return runner.run(ctx, df_all_field, df_seed_field)