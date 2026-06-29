"""
modules/db.py

AstroDBFacade：数仓门面层。
Workflow 和其他业务模块仅与此类对话，屏蔽底层 DuckDB 和算子实现细节。
"""
import logging
import pandas as pd
from pathlib import Path
from .engine import AstroEngine
from .operators import AssetManager, SchemaAligner, IdentityRegistry
import config as cfg


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
        self._asset_mgr = AssetManager(self._engine)
        self._aligner = SchemaAligner(self._engine)
        self._registry = IdentityRegistry(self._engine)

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

    def prepare_assets(self):
        """Stage 1.2: 挂载所有原始数据资产"""
        self.logger.info("正在进行资产挂载 (Stage 1.2)...")
        self._asset_mgr.mount_all(self.cluster_id)

    def align_schemas(self, manifest=None, cluster_cfg=None):
        """Stage 1.5: 资产清洗与物理视图对齐（STD → STX → ALN）"""
        self.logger.info("正在进行 schema 对齐与视图注册 (Stage 1.5)...")
        self._aligner.data_standardize_all(self.cluster_id, manifest=manifest, cluster_cfg=cluster_cfg)

    def get_kinematic_identity(self, mode="static"):
        """获取/重构星团物理真身 DNA"""
        self.logger.info(f"正在获取/重构星团真身 DNA (模式: {mode})...")
        return self._registry.resolve(self.cluster_id, mode)

    def update_kinematic_identity(self, data):
        """滚动更新真身数据"""
        self.logger.info("正在持久化星团真身更新...")
        self._registry.update(self.cluster_id, data)

    # ==========================================================================
    # 直接数据访问 API（兼容旧代码的核心接口）
    # ==========================================================================

    def execute(self, sql: str, params=None):
        """执行 SQL 命令（不返回 DataFrame）"""
        return self._engine.execute(sql, params)

    def query(self, sql: str) -> pd.DataFrame:
        """执行 SQL 查询并返回 DataFrame"""
        return self._engine.query(sql)

    def get_view(self, view_name: str) -> pd.DataFrame:
        """提供受控的数据读取通道"""
        return self._engine.query(f"SELECT * FROM {view_name}")

    def get_row_count(self, table_or_view: str) -> int:
        """获取表/视图的行数"""
        try:
            result = self._engine.execute(f"SELECT COUNT(*) FROM {table_or_view}").fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    def table_exists(self, name: str) -> bool:
        """检查表或视图是否存在"""
        try:
            result = self._engine.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
            ).fetchone()
            if result and result[0] > 0:
                return True
            # 也检查视图
            result2 = self._engine.execute(
                f"SELECT COUNT(*) FROM information_schema.views WHERE view_name = '{name}'"
            ).fetchone()
            return bool(result2 and result2[0] > 0)
        except Exception:
            return False

    # ==========================================================================
    # 视图 / 表注册 API
    # ==========================================================================

    def register_view_from_sql(self, view_name: str, sql: str) -> None:
        """将 SQL 查询注册为数据库视图"""
        self._engine.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
        self.logger.debug(f"已注册 SQL 视图: {view_name}")

    def register_table_from_df(self, table_name: str, df: pd.DataFrame) -> None:
        """将 DataFrame 物化为 DuckDB 物理表（支持 CREATE OR REPLACE）"""
        self._engine.conn.register("_tmp_df", df)
        self._engine.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
        self._engine.execute("DROP VIEW IF EXISTS _tmp_df")
        self.logger.debug(f"已物化物理表: {table_name}")

    def register_view_from_df(self, view_name: str, df: pd.DataFrame) -> None:
        """将 DataFrame 注册为内存视图（不产生物理拷贝）"""
        self._engine.conn.register(view_name, df)
        self.logger.debug(f"已注册内存视图: {view_name}")

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
            self.logger.error(f"❌ df_base 中缺少 id 列 '{id_col}'，无法初始化 Master 表。")
            return

        df_master = df_base[[id_col]].copy()
        # 补充 MASTER_COLS 空列
        for col_key, col_name in cfg.MASTER_COLS.items():
            if col_name not in df_master.columns:
                df_master[col_name] = None

        self.register_table_from_df(table_name, df_master)
        self.logger.info(f"✅ Master 宽表 [{table_name}] 已初始化，基底行数: {len(df_master)}")

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
            self.logger.debug(f"✅ Master 表 [{table_name}] 标签列 {tag_cols} 回灌完毕，更新行数: {len(df_updates)}")
        finally:
            self._engine.execute("DROP VIEW IF EXISTS _tmp_updates")

    # ==========================================================================
    # 数仓持久化 API
    # ==========================================================================

    def save_to_warehouse(self, table_or_view: str, storage_type: str = "snapshots", filename: str = None) -> None:
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