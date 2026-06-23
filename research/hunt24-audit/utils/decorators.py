import functools
import logging

logger = logging.getLogger("AstroPipeline.Checkpoint")

def astro_checkpoint(cache_table_template: str, force_refresh: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            import config as cfg
            db = getattr(self, "db", None)
            
            # 动态渲染缓存表名，注入星团 ID、审计对象及维度模式
            cluster_id = getattr(self, "target_cluster", "m45").lower()
            category = getattr(self, "target_category", "hunt").lower()
            mode = getattr(self, "mode", "5d").lower()
            algo = getattr(self, "algo", "dbscan").lower()

            cache_table_name = cache_table_template.format(
                cluster=cluster_id, category=category, mode=mode, algo=algo
            )

            cache_exists = db.con.execute(
                f"SELECT count(*) FROM information_schema.tables "
                f"WHERE table_name = '{cache_table_name}'"
            ).fetchone()[0] > 0

            if not force_refresh and cache_exists:
                logger.info(f"💾 [Checkpoint] 直接复用已有缓存表: `{cache_table_name}`")
                return cache_table_name # 直接返回表名

            # 2. 如果没有缓存，或者强制刷新，执行原函数
            logger.info(f"⚡ [Checkpoint] 执行流水线生成数据...")
            actual_table_name = func(self, *args, **kwargs) # 这里得到的是生成的物理表名

            # 3. 将结果持久化为缓存表
            logger.info(f"💾 [Checkpoint] 将结果 `{actual_table_name}` 备份至缓存: `{cache_table_name}`")
            db.con.execute(f"DROP TABLE IF EXISTS {cache_table_name}")
            db.con.execute(f"CREATE TABLE {cache_table_name} AS SELECT * FROM {actual_table_name}")
            
            return cache_table_name

        return wrapper
    return decorator