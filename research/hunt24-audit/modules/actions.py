import logging

# 初始化日志
logger = logging.getLogger(f"AstroPipeline.{__name__}")


class StdActions:
    """
    STD (Standardization) 层动作集
    职责:处理字段重命名映射、类型转换, 并支持初步的数据截断(Pre-filtering)。
    """

    @staticmethod
    def _apply_pre_filter(sql_base, cfg, ctx=None):
        """
        [内部辅助方法]：根据配置决定是否在标准化阶段进行初步截断。
        用于被该类下的其他公开方法调用，提高代码复用性。
        """
        filters = cfg.get("pre_filters", [])
        if not filters:
            return sql_base

        # 核心修复：如果存在上下文（ctx），则动态渲染过滤模板（例如将 {CAT_NAME} 替换为对应的星团名）
        if ctx:
            filters = [f.format(**ctx) if isinstance(f, str) else f for f in filters]

        # 将多个过滤条件拼接，例如: ["mag < 19", "parallax > 0"] -> "mag < 19 AND parallax > 0"
        where_clause = " AND ".join(filters)
        logger.info(f"🔍 应用预过滤条件: {where_clause}")
        
        # 使用嵌套查询，确保过滤逻辑作用于处理后的字段名或原始表
        return f"SELECT * FROM ({sql_base}) WHERE {where_clause}"

    @staticmethod
    def default(db, k_ref, cfg, manifest, ctx=None):
        """
        最简动作：直接从物理表映射为视图，同样支持 pre_filters。
        """
        base_sql = f"SELECT * FROM {cfg['raw_table']}"
        final_sql = StdActions._apply_pre_filter(base_sql, cfg, ctx)

        db.register_view_from_sql(cfg["std_view"], final_sql)
        logger.info(
            f"[{k_ref}] STD: Default view created (with optional pre-filtering)."
        )

    @staticmethod
    def std_mapping(db, k_ref, cfg, manifest, ctx=None):
        """
        核心动作：根据配置中的 fields 键进行字段映射，并支持初步截断。
        """
        raw_table = cfg["raw_table"]
        std_view = cfg["std_view"]
        mapping_dict = cfg.get("fields", {})

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

        # 2. 调用内部拦截器进行初步截断
        final_sql = StdActions._apply_pre_filter(base_sql, cfg, ctx)
        logger.debug(f" [{k_ref}] 预拦截器: {final_sql}")

        # 3. 创建视图
        db.register_view_from_sql(std_view, final_sql)

        # 5. 获取过滤后的记录数（查询刚创建的视图）
        count_after = db.get_row_count(std_view)
        dropped = count_before - count_after

        logger.info(f"[{k_ref}] STD: Standardized with mapping and pre-filtering.")
        logger.info(
            f"[{k_ref}] STD: 数据量变化: {count_before:,} -> {count_after:,} (预过滤掉: {dropped:,})"
        )

    @staticmethod
    def pass_through(db, k_ref, cfg, manifest, ctx=None):
        """
        [新增] STD 逻辑透传：将原始物理表直接映射为标准化视图。
        适用于字段名已经完全符合标准(ra, dec, id 等)的原始数据。
        """
        raw_table = cfg["raw_table"]
        std_view = cfg["std_view"]

        base_sql = f"SELECT * FROM {raw_table}"
        final_sql = StdActions._apply_pre_filter(base_sql, cfg, ctx)

        db.register_view_from_sql(std_view, final_sql)
        logger.info(
            f"[{k_ref}] STD: Identity pass-through (with optional pre-filtering)."
        )


class StxActions:
    """
    STX (Extension) 层动作集
    职责: 处理跨表关联(ID 桥接)或逻辑透传。
    """

    @staticmethod
    def pass_through(db, k_ref, cfg, manifest, ctx=None):
        """逻辑透传：将 std 层直接映射到 stx 层"""
        query = f"SELECT * FROM {cfg['std_view']}"
        db.register_view_from_sql(cfg["stx_view"], query)
        logger.info(f"[{k_ref}] STX: Identity pass-through.")

    @staticmethod
    def bridge_dr2_to_dr3(db, k_ref, cfg, manifest, ctx=None):
        """
        核心动作：通过索引表将 DR2 ID 桥接到标准 ID (DR3)
        使用 EXCLUDE 语法移除旧的 id_dr2，保持结果集干净。
        """
        v_src = cfg["std_view"]
        v_dest = cfg["stx_view"]
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
    def by_cluster_geometry(db, k_ref, cfg, manifest, ctx=None):
        """
        空间对齐：基于哈弗辛公式进行球面圆裁剪。
        """
        if not ctx or "id" not in ctx:
            raise ValueError(f"Action 'aln' requires ctx with 'id' for {k_ref}")

        v_src = cfg["stx_view"]
        # 动态渲染视图名称 (例如将 {cluster} 替换为 m45)
        v_dest = cfg["aln_view"].format(cluster=ctx["id"].lower())

        ra = ctx["CENTER_RA"]
        dec = ctx["CENTER_DEC"]
        radius = ctx.get("RADIUS", 5.0)

        # --- 1. 获取裁剪前的记录个数 ---
        count_before = db.get_row_count(v_src)

        query = f"""
            SELECT * FROM {v_src}
            WHERE haversine_distance({ra}, {dec}, ra, dec) <= {radius}
        """
        db.register_view_from_sql(v_dest, query)

        # --- 2. 获取裁剪后的记录个数 ---
        count_after = db.get_row_count(v_dest)

        logger.info(f"[{k_ref}] ALN: Aligned to {ctx['NAME']} within {radius} deg.")
        # 记录数据量变化
        logger.info(
            f"[{k_ref}] ALN: 数据量变化: {count_before} -> {count_after} (裁剪掉: {count_before - count_after})"
        )

        logger.debug(f"[{k_ref}] ALN: 裁剪SQL:\n{query}")

        # 结构预览
        preview = db.query(f"SELECT * FROM {v_dest} LIMIT 5")
        logger.debug(f"{v_dest} 结构预览:\n{preview}")

    @staticmethod
    def by_rectangle_geometry(db, k_ref, cfg, manifest, ctx):
        """
        矩形裁剪：基于 RA/Dec 的范围进行筛选。
        ctx 需包含: ra_min, ra_max, dec_min, dec_max
        """
        if not ctx:
            raise ValueError(f"Action 'aln' requires ctx for {k_ref}")

        v_src = cfg["stx_view"]
        v_dest = cfg["aln_view"].format(cluster=ctx["id"].lower())

        # --- 1. 获取裁剪前的记录个数 ---
        count_before = db.get_row_count(v_src)

        # 逻辑：RA 和 Dec 都在指定的闭区间内
        query = f"""
            SELECT * FROM {v_src}
            WHERE ra BETWEEN {ctx['RA_MIN']} AND {ctx['RA_MAX']}
              AND dec BETWEEN {ctx['DEC_MIN']} AND {ctx['DEC_MAX']}
        """
        db.register_view_from_sql(v_dest, query)

        # --- 2. 获取裁剪后的记录个数 ---
        count_after = db.get_row_count(v_dest)

        logger.info(f"[{k_ref}] ALN: Rectangle cropped to {ctx['NAME']}.")
        # 增加数量变化的日志
        logger.info(
            f"[{k_ref}] ALN: 数据量变化: {count_before} -> {count_after} (裁剪掉: {count_before - count_after})"
        )

        logger.debug(f"[{k_ref}] ALN: 裁剪SQL:\n{query}")

        # 结构预览
        preview = db.query(f"SELECT * FROM {v_dest} LIMIT 5")
        logger.debug(f"{v_dest} 结构预览:\n{preview}")

    @staticmethod
    def pass_through(db, k_ref, cfg, manifest, ctx=None):
        """
        [新增] ALN 逻辑透传：跳过地理裁剪，直接将 STX 视图映射为 ALN 视图。
        适用于参考用的全局星表（如全天背景星），不针对特定集群裁剪。
        """
        if not ctx:
            # 即使不裁剪，也需要 ctx 来渲染视图名称中的 tag
            raise ValueError(
                f"Action 'aln_pass' requires ctx to resolve view names for {k_ref}"
            )

        v_src = cfg["stx_view"]
        v_dest = cfg["aln_view"].format(cluster=ctx["id"].lower())

        query = f"SELECT * FROM {v_src}"
        db.register_view_from_sql(v_dest, query)

        logger.info(f"[{k_ref}] ALN: Geometry pass-through to {v_dest}.")
