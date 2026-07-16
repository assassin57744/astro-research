import functools
import logging

logger = logging.getLogger("AstroPipeline.Checkpoint")


def _resolve_checkpoint_keys(self, args):
    """从 ctx（args[0]）或 self 回退中提取缓存键值。"""
    # 优先从第一个位置参数（RunContext）读取
    ctx = args[0] if args else None
    if ctx is not None and hasattr(ctx, "cluster_id"):
        return (
            ctx.cluster_id.lower(),
            ctx.category.lower(),
            ctx.feature_space.lower(),
            ctx.algorithm.lower(),
        )
    # 回退：兼容未传 ctx 的旧调用路径
    return (
        getattr(self, "target_cluster", "m45").lower(),
        getattr(self, "target_category", "hunt").lower(),
        getattr(self, "feature_space", "5d").lower(),
        getattr(self, "algo", "dbscan").lower(),
    )


def astro_checkpoint(cache_table_template: str, force_refresh: bool = False):
    """轻量级缓存装饰器：仅缓存被装饰函数的返回值（表名）。

    缓存键由 RunContext（args[0]）的 cluster_id / category /
    feature_space / algorithm 组成，不再依赖 self 上的镜像属性。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            db = getattr(self, "db", None)
            cluster_id, category, mode, algo = _resolve_checkpoint_keys(self, args)

            cache_table_name = cache_table_template.format(
                cluster=cluster_id, category=category, mode=mode, algo=algo
            )

            cache_exists = db.con.execute(
                f"SELECT count(*) FROM information_schema.tables "
                f"WHERE table_name = '{cache_table_name}'"
            ).fetchone()[0] > 0

            if not force_refresh and cache_exists:
                logger.info(f"💾 [Checkpoint] 直接复用已有缓存表: `{cache_table_name}`")
                return cache_table_name

            logger.info(f"⚡ [Checkpoint] 执行流水线生成数据...")
            actual_table_name = func(self, *args, **kwargs)

            logger.info(
                f"💾 [Checkpoint] 将结果 `{actual_table_name}` 备份至缓存: "
                f"`{cache_table_name}`"
            )
            db.con.execute(f"DROP TABLE IF EXISTS {cache_table_name}")
            db.con.execute(
                f"CREATE TABLE {cache_table_name} AS SELECT * FROM {actual_table_name}"
            )

            return cache_table_name

        return wrapper
    return decorator