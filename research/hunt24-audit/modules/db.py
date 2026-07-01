"""
modules/db.py

AstroDBFacade：数仓门面层。
Workflow 和其他业务模块仅与此类对话，屏蔽底层 DuckDB 和算子实现细节。
"""

import logging
import pandas as pd
from pathlib import Path
from .engine import AstroEngine
from .operators.asset_manager import AssetManager, SchemaAligner
from .operators.data_selector import DataSelector
from .operators.param_registry import ParamRegistry

from .config_manager import ClusterConfigManager
import config as cfg
from config import DerivedActivationResult

from dataclasses import dataclass, field
from typing import List, Dict

class AstroDBFacade:
    """
    数仓门面 (Facade)。
    统一暴露数仓操作接口，Workflow 通过此类访问所有数据库功能。
    """

    def __init__(self, cluster_id: str = None):
        self.cluster_id = cluster_id
        self.logger = logging.getLogger(f"AstroDB.{cluster_id or 'global'}")

        # 初始化底层引擎
        self._engine = AstroEngine()

        # 依赖注入算子
        self._asset_mgr = AssetManager(self._engine, self.cluster_id)
        self._aligner = SchemaAligner(self._engine, self.cluster_id)
        self._registry = ParamRegistry(self._engine)
        self._data_selector = DataSelector(self._engine, self.cluster_id) if cluster_id else None

        self._config_mgr = ClusterConfigManager(db_instance=self._engine)

        self.logger.info(f"✨ AstroDBFacade 门面已就绪，集群: {cluster_id}")

    # ==========================================================================
    # 上下文管理器协议
    # ==========================================================================

    def __enter__(self) -> "AstroDBFacade":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception as e:
            self.logger.error(f"关闭数据库连接时发生异常: {e}")
        return False

    # ==========================================================================
    # 核心业务 API（Stage 调度接口）
    # ==========================================================================

    def activate_base_assets(self):
        """Stage 1.2: 挂载所有原始数据资产"""
        self.logger.info("正在进行资产挂载 (Stage 1.2)...")
        self._asset_mgr.mount_all(self.cluster_id)

    def activate_derived_assets(self, runtime_context):
        return self._asset_mgr.align_derived_assets(self.cluster_id,runtime_context)
    def _activate_derived_assets(
        self, runtime_context: dict
    ) -> List[str]:
        """
        接收 workflow 传入的上下文，驱动底层 SchemaAligner 算子完成派生切片视图的动态挂载。

        Returns:
            
        """
        self.logger.info(
            f"🚀 AstroDBFacade: 正在激活星团 [{self.cluster_id}] 的动态派生视图..."
        )

        # 1. 驱动内部对齐算子刷写 DuckDB
        # 此时底层 SchemaAligner 会去读 config.py 中带有 pre_filters 的资产（如 m45_seeds_field）
        assets = cfg.MANIFEST

        # self.logger.info(f"[debug] cl_assets : {assets}" )

        # 让算子执行实际的 SQL 挂载（保持算子职责纯粹）
        self._asset_mgr.align_derived_assets(self.cluster_id, runtime_context)

        for asset_key, asset_cfg in assets.items():
            if (
                hasattr(asset_cfg, "pre_filters")
                and asset_cfg.pre_filters
                and asset_key.lower().startswith(f"{self.cluster_id.lower()}_")
            ):
                view_name = f"std_view_{asset_cfg.id}"

                # 记录注册关系
                self._activated_views.append(view_name)
                self._view_manifest[view_name] = f"std_view_{asset_cfg.base_idx}"

                # 动态审计：查询当前切片出的种子纯度/样本量
                try:
                    count_df = self._engine.query(
                        f"SELECT COUNT(*) as cnt FROM {view_name}"
                    )
                    self._row_counts[view_name] = int(count_df["cnt"].iloc[0])
                except Exception as query_err:
                    self._row_counts[view_name] = 0
                    # 💡 打印详细报错和当前上下文中实际拥有的变量
                    self.logger.error(
                        f"❌ 审计视图 {view_name} 失败！错误原因: {query_err}",
                        exc_info=True,
                    )
                    self.logger.info(
                        f"📊 当前传递进来的运行时上下文大盘为: {runtime_context}"
                    )

        self.logger.info(f"✨ 动态视图激活完毕报告: {self._row_counts}")
        return self._activated_views

    def align_schemas(self, manifest=None, cluster_cfg=None):
        """Stage 1.5: 资产清洗与物理视图对齐（STD → STX → ALN）"""
        self.logger.info("正在进行 schema 对齐与视图注册 (Stage 1.5)...")
        self._aligner.align_all(self.cluster_id)

    def init_cluster_params(self, param_recon_mode):
        """获取/重构星团物理真身 DNA"""
        self.logger.info(f"正在获取/重构星团真身 DNA (模式: {param_recon_mode})...")
        res = self._registry.resolve(self.cluster_id, param_recon_mode)
        self._registry._save_refined_params(self.cluster_id,res)
        return res
    
    def get_cluster_data(self, slice_type: str) -> pd.DataFrame:
        """
        【一体化门面转发】根据切片类型获取指定星团的标准化数据集。
        :param slice_type: 可选 'field' (拉取全场景观测大盘) 或 'seeds' (拉取通过数据审计的高纯度种子)
        """
        if not self._data_selector:
            self.logger.error(f"❌ 全局无目标上下文门面无法调用带有天体约束的获取数据接口。")
            return pd.DataFrame()
            
        # 统一调度，并将自身(self)传回，供内部提取 seeds 时调度数仓底座
        return self._data_selector.fetch_data(slice_type=slice_type)    

    def update_kinematic_identity(self, data):
        """滚动更新真身数据"""
        self.logger.info("正在持久化星团真身更新...")
        self._registry.update(self.cluster_id, data)

    def _prepare_runtime_context(self, extra_context: dict = None) -> dict:
        """
        【纯函数门面接口】
        1. 自动调用 ClusterConfigManager 级联检索当前星团核心物理参数（CENTER_RA, CENTER_DEC, PLX_REF 等）。
        2. 结合 config.py 的控制参数以及外部传入的临时覆盖参数组装完整的 runtime_context。
        3. 将完整的上下文字典返回给上层 Workflow，不产生任何数仓副作用。
        """
        # 🛡️ 防御性初始化，确保任何提前返回分支都返回标准字典
        runtime_context = {
            "CENTER_RA": 0.0,
            "CENTER_DEC": 0.0,
            "PLX_REF": 1.0,
            "SEED_RADIUS": 1.0,
            "SEED_PLX_LIM": 1.0,
            "SEED_MAX_MAG": 18.0,
        }

        if not self.cluster_id:
            self.logger.warning("⚠️ 全局会话未指定 cluster_id，返回空运行时上下文。")
            return extra_context or runtime_context

        self.logger.info(
            f"🔮 AstroDBFacade: 正在为星团 [{self.cluster_id}] 级联提取运行时切片上下文..."
        )

        # 1. 🔍 利用 ClusterConfigManager 级联提取核心物理参数
        try:
            cluster_dna = {
                "CENTER_RA": self._config_mgr.get_param(
                    self.cluster_id.upper(), "CENTER_RA"
                ),
                "CENTER_DEC": self._config_mgr.get_param(
                    self.cluster_id.upper(), "CENTER_DEC"
                ),
                "PLX_REF": self._config_mgr.get_param(
                    self.cluster_id.upper(), "PLX_REF"
                ),
            }
        except Exception as e:
            self.logger.error(f"❌ 级联检索星团 DNA 物理特征失败: {e}", exc_info=True)
            # 出现异常时返回传入的覆盖参数或空字典，确保不崩
            return extra_context or runtime_context

        # 2. 📑 提取 config.py 里属于该星团的静态控制参数作为兜底
        cluster_cfg = cfg.CLUSTERS.get(self.cluster_id.upper())
        global_render_cfg = {}
        if cluster_cfg:
            global_render_cfg = {
                "SEED_RADIUS": getattr(cluster_cfg, "SEED_RADIUS", 1.0),
                "SEED_PLX_LIM": getattr(cluster_cfg, "SEED_PLX_LIM", 1.0),
                "SEED_MAX_MAG": getattr(cluster_cfg, "SEED_MAX_MAG", 18.0),
            }

        # 3. 🔀 通过 Python 字典解包进行多路合并 (静态配置 <- 级联物理DNA)
        runtime_context = {**global_render_cfg, **cluster_dna}

        # 如果外部传入了临时的覆盖控制参数，合并进来
        if extra_context:
            runtime_context.update(extra_context)

        self.logger.debug(f"🔍 最终刷新的 SQL 模板变量大盘: {runtime_context}")

        # 🚀 【核心修复】显式返回组装好的完整上下文字典！
        return runtime_context

    # ==========================================================================
    # 直接数据访问 API（兼容旧代码的核心接口）
    # ==========================================================================

    def execute(self, sql: str, params=None):
        """执行 SQL 命令（不返回 DataFrame）"""
        return self._engine.execute(sql, params)

    def query(self, sql: str) -> pd.DataFrame:
        """执行 SQL 查询并返回 DataFrame"""
        return self._engine.query(sql)

    def get_view_df(self, view_name: str) -> pd.DataFrame:
        """提供受控的数据读取通道"""
        return self._engine.query(f"SELECT * FROM {view_name}")

    def get_row_count(self, table_or_view: str) -> int:
        """获取表/视图的行数"""
        try:
            result = self._engine.execute(
                f"SELECT COUNT(*) FROM {table_or_view}"
            ).fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    # ==========================================================================
    # 视图 / 表注册 API
    # ==========================================================================

    # TODO: 未来移到 engine 中
    def register_view_from_sql(self, view_name: str, sql: str) -> None:
        """将 SQL 查询注册为数据库视图"""
        self._engine.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
        self.logger.debug(f"已注册 SQL 视图: {view_name}")

    # ==========================================================================
    # Master 宽表 API（算法结果追踪）
    # ==========================================================================

    def init_master_table(self, table_name: str, df_base: pd.DataFrame) -> None:
        """
        初始化 Master 宽表。
        以 df_base 的 id 列为主键骨架，补充所有 MASTER_COLS 的空列。
        """
        id_col = cfg.STD_COLS["ID"]
        if id_col not in df_base.columns:
            self.logger.error(
                f"❌ df_base 中缺少 id 列 '{id_col}'，无法初始化 Master 表。"
            )
            return

        df_master = df_base[[id_col]].copy()
        # 补充 MASTER_COLS 空列
        for col_key, col_name in cfg.MASTER_COLS.items():
            if col_name not in df_master.columns:
                df_master[col_name] = None

        self.register_table_from_df(table_name, df_master)
        self.logger.info(
            f"✅ Master 宽表 [{table_name}] 已初始化，基底行数: {len(df_master)}"
        )

    def tag_master_table(self, table_name: str, df_updates: pd.DataFrame) -> None:
        """
        将 df_updates 中的标签列回灌到 Master 表。
        df_updates 必须包含 id 列，其余列为需要更新的标签列。
        """
        id_col = cfg.STD_COLS["ID"]
        if id_col not in df_updates.columns or df_updates.empty:
            return

        tag_cols = [c for c in df_updates.columns if c != id_col]
        if not tag_cols:
            return

        self._engine.conn.register("_tmp_updates", df_updates)
        try:
            for col in tag_cols:
                sql = f"""
                    UPDATE {table_name}
                    SET {col} = u.{col}
                    FROM _tmp_updates u
                    WHERE {table_name}.{id_col} = u.{id_col}
                """
                self._engine.execute(sql)
            self.logger.debug(
                f"✅ Master 表 [{table_name}] 标签列 {tag_cols} 回灌完毕，更新行数: {len(df_updates)}"
            )
        finally:
            self._engine.execute("DROP VIEW IF EXISTS _tmp_updates")

    # ==========================================================================
    # 数仓持久化 API
    # ==========================================================================

    def save_to_warehouse(
        self, table_or_view: str, storage_type: str = "snapshots", filename: str = None
    ) -> None:
        """将表或视图持久化为 Parquet 文件"""
        sub_dir = cfg.DATA_DIR / "warehouse" / storage_type
        sub_dir.mkdir(parents=True, exist_ok=True)

        fname = filename if filename else table_or_view
        output_path = sub_dir / f"{fname}.parquet"

        self._engine.execute(
            f"COPY (SELECT * FROM {table_or_view}) TO '{output_path.as_posix()}' (FORMAT PARQUET)"
        )
        self.logger.info(f"💾 已持久化: {table_or_view} → {output_path}")

    # ==========================================================================
    # 辅助与基础设施接口
    # ==========================================================================

    def close(self):
        """关闭数据库连接"""
        self._engine.close()
        self.logger.info("💾 AstroDBFacade 连接已关闭。")
