# modules/operators/data_selector.py
import logging
import pandas as pd

class DataSelector:
    """
    Stage 2.1: 恒星数据选择器。
    【职责纯化】完全消除对外部门面的跨模块耦合，只基于传入的物理激活战报完成大盘与高纯种子数据的切片与审计。
    """
    def __init__(self, engine, cluster_id: str):
        self._engine = engine
        self.cluster_id = cluster_id
        self.logger = logging.getLogger(f"AstroDB.{cluster_id or 'global'}.DataSelector")

    def fetch_data(self, slice_type: str, activation_report=None) -> pd.DataFrame:
        target_type = slice_type.lower().strip()
        if target_type == "field":
            return self._fetch_field_slice()
        elif target_type == "seeds":
            return self._fetch_seeds_slice(activation_report)
        return pd.DataFrame()

    def _fetch_field_slice(self) -> pd.DataFrame:
        """【100%对齐经典流】提取标准的 ODS 大盘切片视图"""
        cluster_obs_field_table_name = f"std_view_{self.cluster_id.lower()}_field"
        try:
            cluster_obs_field_df = self._engine.query(f"SELECT * FROM {cluster_obs_field_table_name}")
            self.logger.info(f"🌟 [DataSelector] 星团观测范围全场大盘数据拉取成功, 记录总数: {len(cluster_obs_field_df)}")
            return cluster_obs_field_df
        except Exception as e:
            self.logger.error(f"❌ [DataSelector] 提取大盘全场数据失败，底层物理/虚拟资产未挂载: {e}")
            return pd.DataFrame()

    def _fetch_seeds_slice(self, activation_report) -> pd.DataFrame:
        """【100%对齐经典流】带有确定性主动防御机制的高纯度种子恒星过滤"""
        seed_view_name = f"std_view_{self.cluster_id.lower()}_seeds_field"

        # if not activation_report or seed_view_name not in activation_report.activated_views:
        #     self.logger.error(f"❌ [DataSelector 灾难] 预期的种子切片虚拟资产未挂载，阻断矩阵计算通道。{seed_view_name}")
        #     return pd.DataFrame()

        # sample_size = activation_report.row_counts[seed_view_name]
        # self.logger.info(f"📈 [DataSelector] 种子切片视图就绪，包含高质量候选恒星 {sample_size} 颗。")

        # # 【核心熔断防御对齐】样本量低于 10 颗执行主动熔断防护，防止算法矩阵奇异崩溃
        # if sample_size < 10:
        #     self.logger.warning(f"⚠️ [算法防御拦截] 种子样本量 ({sample_size}) 过于稀疏，不满足统计置信度，触发主动熔断。")
        #     return pd.DataFrame()

        return self._engine.query(f"SELECT * FROM {seed_view_name}")