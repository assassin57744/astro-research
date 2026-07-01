# engine.py
import duckdb
import pandas as pd
import logging
import config as cfg
from pathlib import Path

class AstroEngine:
    def __init__(self):
        self.logger = logging.getLogger("AstroEngine")
        self.data_root = cfg.DATA_DIR
        self.db_path = self.data_root / "warehouse" / "astrodb_internal.db"
        
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"💾 正在连接数据库: {self.db_path}")
            self.conn = duckdb.connect(str(self.db_path))
        except Exception as e:
            self.logger.error(f"❌ 数据库连接失败: {e}")
            raise e

    def execute(self, sql: str, params=None):
        """执行 SQL 命令"""
        return self.conn.execute(sql, params)

    def query(self, sql: str):
        """执行查询并返回 DataFrame"""
        return self.conn.sql(sql).df()

    def close(self):
        """显式关闭连接"""
        if hasattr(self, 'conn'):
            self.conn.close()
            self.logger.info("💾 数据库连接已安全关闭")

# ==================== modules/engine.py ====================

    def table_exists(self, name: str) -> bool:
        """检查表或视图是否存在"""
        try:
            result = self.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
            ).fetchone()
            if result and result[0] > 0:
                return True
            # 也检查视图
            result2 = self.execute(
                f"SELECT COUNT(*) FROM information_schema.views WHERE view_name = '{name}'"
            ).fetchone()
            return bool(result2 and result2[0] > 0)
        except Exception:
            return False    
    
    def register_table_from_df(self, table_name: str, df: pd.DataFrame) -> None:
        """【原子持久化】将 DataFrame 物化为 DuckDB 物理表（支持 CREATE OR REPLACE）"""
        # 使用独立的桥接名字，防止与业务视图冲突
        tmp_bridge = f"_tmp_table_bridge_{table_name}"
        self.conn.register(tmp_bridge, df)
        try:
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {tmp_bridge}")
        finally:
            self.conn.execute(f"DROP VIEW IF EXISTS {tmp_bridge}")

    def register_view_from_df(self, view_name: str, df: pd.DataFrame) -> None:
        """【原子虚拟化】将 DataFrame 注册为内存视图（非持久化、不产生物理拷贝）"""
        self.conn.register(view_name, df)

    def unregister_view(self, view_name: str) -> None:
        """显式销毁内存视图，及时释放 Python 内存引用"""
        self.conn.execute(f"DROP VIEW IF EXISTS {view_name}")

    # --- 增加上下文管理器支持 ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()