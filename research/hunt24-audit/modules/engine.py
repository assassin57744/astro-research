# engine.py
import duckdb
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

    # --- 增加上下文管理器支持 ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()