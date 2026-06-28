# modules/actions.py
import logging
import copy
from typing import TYPE_CHECKING, List, Any
# 🚀 利用 TYPE_CHECKING 斩断运行时循环导入锁，同时保留 IDE 的丝滑强类型补全
if TYPE_CHECKING:
    from config import ClusterConfig, CatalogConfig

# 初始化日志
logger = logging.getLogger(f"AstroPipeline.{__name__}")


class StdActions:
    """
    STD (Standardization) 层动作集
    职责: 处理字段重命名映射、类型转换, 并支持初步的数据截断(Pre-filtering)。
    """

    @staticmethod
    def _apply_pre_filter(sql_base: str, data_cfg: "CatalogConfig", cl_cfg: "ClusterConfig" = None, data_idx: str = None, manifest: Any = None) -> str:
        """
        [内部辅助方法]：根据配置决定是否在标准化阶段进行初步截断。
        
        参数:
            sql_base: 基础 SQL 查询语句。
            cfg: 当前资产的注册配置对象 (CatalogRegistryConfig)。
            ctx: 包含完备天体物理参数的强类型上下文实例 (ClusterTargetConfig)。
        返回:   
            拼接并渲染预过滤条件后的完整 SQL 语句。
        """
        # 1. 强类型对齐：直接提取强类型对象的属性
        filters = data_cfg.pre_filters if hasattr(data_cfg, "pre_filters") else data_cfg.get("pre_filters", [])
        
        if not filters:
            return sql_base
        
        logger.info(f"🚀 [Pre-Filter Engine] 捕获到原始资产过滤模板数: {len(filters)} 条")
        logger.debug(f"当前过滤模板上下文: {cl_cfg}, 原始条件: {filters}")

        # 2. 核心修复：如果上下文 ctx 存在，进行安全的强类型动态渲染
        if cl_cfg:
            # 为了应对 f.format(**ctx) 的解包，同时拿到特定外部星表的自适应 CAT_NAME：
            # 1. 动态生成当前上下文投影字典
            ctx_projection = {k: cl_cfg[k] for k in cl_cfg.keys()}

            # 🌟 命名自适应融合：实时动态注入当前外部星表源对应的专属物理表别名 (如 m45 -> Melotte_22)
            if data_idx and hasattr(cl_cfg, "get_cat_name"):
                ctx_projection["CAT_NAME"] = cl_cfg.get_cat_name(data_idx, manifest)

            rendered_filters = []
            for f in filters:
                if isinstance(f, str):
                    try:
                        # 🌟 由于类实现了映射协议，可以直接对强类型对象 local_ctx 进行 ** 解包
                        # 此时 f 内部的 {CENTER_RA}, {SEED_RADIUS}, {id} 将被完美动态替换
                        # 原生解包，此时不管是物理常数还是多星表自适应名称，完美命中
                        rendered_filters.append(f.format(**ctx_projection))
                    except KeyError as e:
                        logger.error(
                            f"❌ 过滤模板 '{f}' 渲染故障！"
                            f"星团物理库 [{cl_cfg.KEY_ID}] 中缺乏先验参数: {e}. "
                            f"可用的物理常数有: {list(ctx_projection.keys())}"
                        )
                        raise
                else:
                    rendered_filters.append(f)
            
            filters = rendered_filters
            logger.info(f"✅ 动态物理参数注入完成，实际应用过滤条件: {filters}")

        # 将多个过滤条件拼接，例如: ["mag < 19", "parallax > 0"] -> "mag < 19 AND parallax > 0"
        # 3. 执行最终的 SQL 构建逻辑（根据您的架构，通常是 WHERE 条件组合）
        if filters:
            filter_clause = " AND ".join(filters)
            if "WHERE" in sql_base.upper():
                sql_base = f"{sql_base} AND ({filter_clause})"
            else:
                sql_base = f"{sql_base} WHERE {filter_clause}"
        
        return sql_base
    
    @staticmethod
    def default(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        最简动作：直接从物理表映射为视图，同样支持 pre_filters。
        """
        # 🚀 彻底类型化：全面切换为对象属性访问 cfg.raw_table / cfg.std_view
        base_sql = f"SELECT * FROM {cfg.raw_table}"
        final_sql = StdActions._apply_pre_filter(base_sql, cfg, ctx, data_idx=k_ref, manifest=manifest)

        db.register_view_from_sql(cfg.std_view, final_sql)
        logger.info(f"[{k_ref}] STD: Default view created (with optional pre-filtering).")

    @staticmethod
    def std_mapping(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        核心动作：根据配置中的 fields 键进行字段映射，并支持初步截断。
        """
        raw_table = cfg.raw_table
        std_view = cfg.std_view
        mapping_dict = cfg.fields if hasattr(cfg, "fields") else cfg.get("fields", {})

        # 使用 scalar 或 fetchone 快速获取 count
        count_before = db.get_row_count(raw_table)

        # 1. 构造基础 Mapping SQL
        if mapping_dict:
            select_items = []
            for new_col, old_col in mapping_dict.items():

                safe_old = f'"{old_col}"'

                # 针对 ID 类字段进行强制类型转换，确保跨表 JOIN 时的兼容性
                if "id" in new_col.lower():
                    select_items.append(f"CAST({safe_old} AS BIGINT) AS {new_col}")
                else:
                    select_items.append(f"{safe_old} AS {new_col}")
            select_clause = ", ".join(select_items)
        else:
            select_clause = "*"

        base_sql = f"SELECT {select_clause} FROM {raw_table}"

        # 2. 传递完备线索进入内部拦截器进行初步截断
        final_sql = StdActions._apply_pre_filter(base_sql, cfg, ctx, data_idx=k_ref, manifest=manifest)
        logger.debug(f" [{k_ref}] 预拦截器: {final_sql}")

        # 3. 创建视图
        db.register_view_from_sql(std_view, final_sql)

        # 5. 获取过滤后的记录数（查询刚创建的视图）
        count_after = db.get_row_count(std_view)
        dropped = count_before - count_after

        logger.info(f"[{k_ref}] STD: Standardized with mapping and pre-filtering.")
        logger.info(f"[{k_ref}] STD: 数据量变化: {count_before:,} -> {count_after:,} (预过滤掉: {dropped:,})")

    @staticmethod
    def pass_through(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        [新增] STD 逻辑透传：将原始物理表直接映射为标准化视图。
        适用于字段名已经完全符合标准(ra, dec, id 等)的原始数据。
        """
        raw_table = cfg.raw_table
        std_view = cfg.std_view

        base_sql = f"SELECT * FROM {raw_table}"
        final_sql = StdActions._apply_pre_filter(base_sql, cfg, ctx, data_idx=k_ref, manifest=manifest)

        db.register_view_from_sql(std_view, final_sql)
        logger.info(f"[{k_ref}] STD: Identity pass-through (with optional pre-filtering).")


class StxActions:
    """
    STX (Extension) 层动作集
    职责: 处理跨表关联(ID 桥接)或逻辑透传。
    """

    @staticmethod
    def pass_through(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """逻辑透传：将 std 层直接映射到 stx 层"""
        query = f"SELECT * FROM {cfg.std_view}"
        db.register_view_from_sql(cfg.stx_view, query)
        logger.info(f"[{k_ref}] STX: Identity pass-through.")

    @staticmethod
    def bridge_dr2_to_dr3(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        核心动作：通过索引表将 DR2 ID 桥接到标准 ID (DR3)
        使用 EXCLUDE 语法移除旧的 id_dr2，保持结果集干净。
        """
        v_src = cfg.std_view
        v_dest = cfg.stx_view
        
        # 针对大盘 manifest 容器的强类型/字典自适应兼容
        if hasattr(manifest, "dr2idx"):
            v_idx = manifest.dr2idx.std_view
        else:
            v_idx = manifest["dr2idx"]["std_view"]

        query = f"""
            SELECT 
                idx.id AS id, 
                ref.* EXCLUDE (id_dr2)
            FROM {v_src} AS ref
            LEFT JOIN {v_idx} AS idx ON CAST(ref.id_dr2 AS BIGINT) = CAST(idx.id_dr2 AS BIGINT)
        """
        db.register_view_from_sql(v_dest, query)
        logger.info(f"[{k_ref}] STX: Bridged DR2 to DR3 using {v_idx}.")

    @staticmethod
    def fill_prob(db, k_ref, cfg, manifest, ctx=None):
        """
        [预留] 复杂逻辑示例：基于某个概率模型为 STX 层添加一个新的字段。
        该方法目前未被调用，仅作为未来扩展的示例。
        """
        v_src = cfg["std_view"]
        v_dest = cfg["stx_view"]

        query = f"""
            SELECT 
                *, 
                -- 这里的 prob_model 是一个假设的函数，需要在数据库中预先定义
                1 AS prob
            FROM {v_src}
        """
        db.register_view_from_sql(v_dest, query)
        logger.info(
            f"[{k_ref}] STX: Added membership probability field using a hypothetical model."
        )


class AlnActions:
    """
    ALN (Alignment) 层动作集
    职责：进行针对特定星团的空间切片。
    """

    @staticmethod
    def by_cluster_geometry(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        空间对齐：基于哈弗辛公式进行球面圆裁剪。
        """
        # 🚀 彻底类型化：摒弃 "id" not in ctx 的字典判定，直接判定强类型对象及其属性
        if not ctx or not ctx.id:
            raise ValueError(f"Action 'aln' requires strong-typed ctx with valid '.id' property for {k_ref}")

        v_src = cfg.stx_view
        # 🌟 优雅属性访问：ctx.id 自动映射出小写形式
        v_dest = cfg.aln_view.format(cluster=ctx.id)

        ra = ctx.CENTER_RA
        dec = ctx.CENTER_DEC
        radius = getattr(ctx, "RADIUS", 5.0)  # 如果类里包含字段则直接 ctx.RADIUS

        # --- 1. 获取裁剪前的记录个数 ---
        count_before = db.get_row_count(v_src)

        query = f"""
            SELECT * FROM {v_src}
            WHERE haversine_distance({ra}, {dec}, ra, dec) <= {radius}
        """
        db.register_view_from_sql(v_dest, query)

        # --- 2. 获取裁剪后的记录个数 ---
        count_after = db.get_row_count(v_dest)

        logger.info(f"[{k_ref}] ALN: Aligned to {ctx.KEY_ID} within {radius} deg.")
        # 记录数据量变化
        logger.info(f"[{k_ref}] ALN: 数据量变化: {count_before:,} -> {count_after:,} (裁剪掉: {count_before - count_after:,})")

        logger.debug(f"[{k_ref}] ALN: 裁剪SQL:\n{query}")

        # 结构预览
        preview = db.query(f"SELECT * FROM {v_dest} LIMIT 5")
        logger.debug(f"{v_dest} 结构预览:\n{preview}")

    @staticmethod
    def by_rectangle_geometry(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        矩形裁剪：基于 RA/Dec 的范围进行筛选。
        ctx 需包含: ra_min, ra_max, dec_min, dec_max
        """
        if not ctx:
            raise ValueError(f"Action 'aln' requires strong-typed ctx for {k_ref}")

        v_src = cfg.stx_view
        v_dest = cfg.aln_view.format(cluster=ctx.id)

        # --- 1. 获取裁剪前的记录个数 ---
        count_before = db.get_row_count(v_src)

        # 🚀 彻底类型化：ctx['RA_MIN'] 清理为强对象属性 ctx.RA_MIN
        query = f"""
            SELECT * FROM {v_src}
            WHERE ra BETWEEN {ctx.RA_MIN} AND {ctx.RA_MAX}
              AND dec BETWEEN {ctx.DEC_MIN} AND {ctx.DEC_MAX}
        """
        db.register_view_from_sql(v_dest, query)

        # --- 2. 获取裁剪后的记录个数 ---
        count_after = db.get_row_count(v_dest)
        logger.info(f"[{k_ref}] ALN: Rectangle cropped to {ctx.KEY_ID}.")
        logger.info(f"[{k_ref}] ALN: 数据量变化: {count_before:,} -> {count_after:,} (裁剪掉: {count_before - count_after:,})")

        logger.debug(f"[{k_ref}] ALN: 裁剪SQL:\n{query}")

        # 结构预览
        preview = db.query(f"SELECT * FROM {v_dest} LIMIT 5")
        logger.debug(f"{v_dest} 结构预览:\n{preview}")

    @staticmethod
    def pass_through(db, k_ref: str, cfg: "CatalogConfig", manifest: Any, ctx: "ClusterConfig" = None) -> None:
        """
        ALN 逻辑透传：跳过地理裁剪，直接将 STX 视图映射为 ALN 视图。
        适用于参考用的全局星表（如全天背景星），不针对特定集群裁剪。
        """
        if not ctx:
            # 即使不裁剪，也需要 ctx 来渲染视图名称中的 tag
            raise ValueError(f"Action 'aln_pass' requires strong-typed ctx to resolve view names for {k_ref}")

        v_src = cfg.stx_view
        # 🌟 彻底解决此前报错的隐藏大坑，让强类型对象生命周期贯穿始终
        v_dest = cfg.aln_view.format(cluster=ctx.id)

        query = f"SELECT * FROM {v_src}"
        db.register_view_from_sql(v_dest, query)

        logger.info(f"[{k_ref}] ALN: Geometry pass-through to {v_dest}.")
