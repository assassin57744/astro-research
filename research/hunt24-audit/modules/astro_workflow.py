import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
from astroquery.simbad import Simbad  # pylint: disable=unused-import
from utils.decorators import astro_checkpoint  # 1. 导入装饰器
# 扁平化后的内部导入
from modules.astro_db import AstroDB
from modules.pg_core import PriorGMM
from modules.pg_core_ex import PriorGMMEx  # 🧪 引入独立测试核
from modules.validator import UnifiedMemberValidator
from modules.transformer import AstroTransformer

import config as cfg
from config import (
    IDX_CG20,
    IDX_HEYL,
    IDX_HUNT,
    IDX_DR2IDX,
    IDX_IDS_SIMBAD,
    IDX_GMM,
    CLUSTERS,
    MANIFEST as DATA,
    GMM_CONFIG,
    MEMBER_SAMPLE_THRESHOLD,
    STD_COLS,
    TMPL,
    GOLDEN_SAMPLE_THRESHOLD,
)


class AstroWorkflow:
    """天文数据处理工作流编排引擎。

    该类作为流水线的核心调度器，负责编排数据库交互、特征转换、算法模型训练与推理、
    以及基于文献的多维度自动化审计流程。

    Attributes:
        db (AstroDB): 绑定的数据库实例，用于执行 SQL 和管理视图。
        logger (logging.Logger): 专属于工作流模块的日志记录器。
        manifest (dict): 来源于数据库实例的数据配置清单。
    """

    def __init__(self, db_instance, target_cluster=None, target_category=None):
        """初始化工作流实例。

        Args:
            db_instance (AstroDB): 活跃的 AstroDB 数据库对象。
            target_cluster (str): 当前处理的星团 ID。
            target_category (str): 当前审计的类别。
        """
        self.db = db_instance
        self.target_cluster = target_cluster
        self.target_category = target_category
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        self.manifest = getattr(db_instance, "data_manifest", {})

    def _process_conflicts(self, dr2_view, bridge_view):
        """[私有方法] 检测并打印 DR3 ID 与 DR2 ID 之间的多对一冲突。

        Args:
            dr2_view (str): 原始 DR2 数据视图。
            bridge_view (str): 包含 ID 映射关系的桥接表视图。
        """
        conflict_query = f"""
            SELECT 
                nb.id AS dr3_id, 
                count(*) as match_count,
                list(ref.id_dr2) as original_dr2_list,
                list(ref.prob) as prob_list
            FROM {dr2_view} AS ref
            JOIN {bridge_view} AS nb ON ref.id_dr2 = nb.id_dr2
            GROUP BY nb.id
            HAVING count(*) > 1
        """
        df_conflicts = self.db.query(conflict_query)
        if not df_conflicts.empty:
            num_conflicts = len(df_conflicts)
            total_affected = df_conflicts["match_count"].sum()
            self.logger.warning(
                f"⚠️ 检测到 {num_conflicts} 组多对一匹配 (共涉及 {total_affected} 条记录)"
            )

            for _, row in df_conflicts.head(5).iterrows():
                self.logger.debug(
                    f"  - DR3 {row['dr3_id']}: 关联了 DR2 {row['original_dr2_list']} "
                    f"概率分别为 {row['prob_list']}"
                )
        else:
            self.logger.info("✅ 未发现多对一冲突，匹配关系为 1:1")

    def get_subset_view(self, t_source, tag="default", conditions=None):
        """基于特定条件从源表创建子集视图。

        Args:
            t_source (str): 源表名。
            tag (str): 用于标识子集的标签名。
            conditions (str, optional): SQL WHERE 子句条件。

        Returns:
            str: 注册成功的视图名称。
        """
        cluster_name = CLUSTERS[self.target_cluster]["NAME"]
        v_subset = cfg.TMPL.V_RES_SUB.format(cluster=cluster_name, tag=tag)
        sql = f"SELECT * FROM {t_source}"
        if conditions:
            sql += f" WHERE {conditions}"

        self.db.register_view_from_sql(v_subset, sql)
        count = self.db.get_row_count(v_subset)
        self.logger.info(f"✅ 子集视图 [{v_subset}] 已注册 (Tag: {tag}, 记录数: {count})")
        return v_subset

    def get_view_missing_sources(self, t_source, k_ref):
        """构建“缺失源”分析视图：即参考文献中存在但算法未发现的星源。

        Args:
            t_source (str): 算法结果视图。
            k_ref (str): 参考星表键值。

        Returns:
            str: 注册成功的缺失源视图名称（v_miss_...）。
        """
        v_result = cfg.TMPL.V_MISS.format(idx=k_ref)
        self.logger.info(
            f"正在构建 {t_source} 与 {k_ref} 的漏检源分析视图: {v_result}"
        )

        v_ref = DATA[k_ref]["stx_view"]
        col_prob = DATA[k_ref]["col_prob"]

        ctx = CLUSTERS[self.target_cluster]
        pmra_ref = ctx.get("PMRA_REF", 0.0)
        pmdec_ref = ctx.get("PMDEC_REF", 0.0)

        query = f"""
            SELECT ref.id, ref.pmra, ref.pmdec, ref.plx, ref.prob as {col_prob},
                   SQRT(POWER(ref.pmra - {pmra_ref}, 2) + POWER(ref.pmdec - ({pmdec_ref}), 2)) AS pm_dist_sq
            FROM {v_ref} AS ref
            WHERE ref.id NOT IN (SELECT id FROM {t_source})
        """
        self.db.register_view_from_sql(v_result, query)
        return v_result

    def build_analysis_ready_view_ex(self, t_gmm_prob, k_refs):
        """构建分析大宽表。

        汇集 SeedGMM 结果与多个参考星表的概率数据，采用纯 SQL (DuckDB) 逻辑实现，
        避免中间态 DataFrame 的类型转换与内存占用。

        Args:
            t_gmm_prob (str): SeedGMM 核心结果视图名。
            k_refs (list[str]): 参考星表关键字列表 (如 ['heyl', 'hunt'])。

        Returns:
            str: 注册成功的分析大宽表视图名。
        """
        col_id = STD_COLS["ID"]

        # 1. 初始 SQL：以 SeedGMM 的结果作为左表基准
        # 我们使用 CTE (Common Table Expression) 结构，方便动态链式 JOIN
        sql = f"SELECT * FROM {t_gmm_prob}"

        # 2. 动态构建多层 LEFT JOIN
        for k in k_refs:
            col_prob_alias = DATA[k]["col_prob"]
            t_ref_view = DATA[k]["stx_view"]

            join_part = f"""
                LEFT JOIN (
                    SELECT {col_id}, prob AS {col_prob_alias} 
                    FROM {t_ref_view}
                ) AS ref_{k} USING ({col_id})
            """
            sql = f"SELECT base.*, ref_{k}.{col_prob_alias} FROM ({sql}) AS base {join_part}"

        cluster_name = CLUSTERS[self.target_cluster]["NAME"]
        view_name = TMPL.V_ALL.format(cluster=cluster_name)

        # 统一使用标准化视图注册接口
        self.db.register_view_from_sql(view_name, sql)
        count = self.db.get_row_count(view_name)
        self.logger.info(f"分析宽表已通过纯 SQL 构建: {view_name}，共计行数: {count}")
        return view_name

    def align_reference_to_window(self, k_ref):
        """[动态物理窗口对齐] 根据参考星表的物理边界自动裁剪目标天区数据。

        通过分析参考星表的特征极值（RA/Dec/PM/Plx/Mag），构建一个紧凑的对比窗口。

        Args:
            k_ref (str): 参考星表键值。

        Returns:
            str: 对齐后的 aln 视图名称（aln_...）。
        """
        cur_cfg = DATA[k_ref]
        field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
        gaia_cfg = DATA[field_idx]
        v_stx_gaia = gaia_cfg["stx_view"]
        v_result = gaia_cfg["aln_view"].format(prefix=k_ref)
        ref_view_name = cur_cfg["stx_view"]

        if ref_view_name is None:
            self.logger.error(f"❌ 无法找到 ref_key '{k_ref}' 对应的视图")
            raise ValueError(f"Invalid ref_key: {k_ref}")

        # 动态获取参考星表在该星团下的多维度极值边界
        bounds_sql = f"""
        SELECT 
            MIN({STD_COLS['RA']}) as min_ra, MAX({STD_COLS['RA']}) as max_ra,
            MIN({STD_COLS['DEC']}) as min_dec, MAX({STD_COLS['DEC']}) as max_dec,
            MIN({STD_COLS['PMRA']}) as min_pmra, MAX({STD_COLS['PMRA']}) as max_pmra,
            MIN({STD_COLS['PMDEC']}) as min_pmdec, MAX({STD_COLS['PMDEC']}) as max_pmdec,
            MIN({STD_COLS['PLX']}) as min_plx, MAX({STD_COLS['PLX']}) as max_plx,
            MIN({STD_COLS['MAG']}) as min_mag, MAX({STD_COLS['MAG']}) as max_mag
        FROM {ref_view_name}
        """
        bounds = self.db.query(bounds_sql).iloc[0]

        pre_count = self.db.query(f"SELECT COUNT(*) FROM {v_stx_gaia}").iloc[0, 0]

        sql_filter = f"""
        SELECT * FROM {v_stx_gaia}
        WHERE 
            {STD_COLS['RA']} BETWEEN {bounds['min_ra']} AND {bounds['max_ra']}
            AND {STD_COLS['DEC']} BETWEEN {bounds['min_dec']} AND {bounds['max_dec']}
            AND {STD_COLS['PMRA']} BETWEEN {bounds['min_pmra']} AND {bounds['max_pmra']}
            AND {STD_COLS['PMDEC']} BETWEEN {bounds['min_pmdec']} AND {bounds['max_pmdec']}
            AND {STD_COLS['PLX']} BETWEEN {bounds['min_plx']} AND {bounds['max_plx']}
            AND {STD_COLS['MAG']} BETWEEN {bounds['min_mag']} AND {bounds['max_mag']}
        """

        self.db.register_view_from_sql(v_result, sql_filter)

        post_count = self.db.query(f"SELECT COUNT(*) FROM {v_result}").iloc[0, 0]
        dropped_count = pre_count - post_count

        self.logger.info(f"📐 动态窗口对齐报告 [{v_stx_gaia} -> {v_result}]:")
        self.logger.info(f"   - 原始记录数: {pre_count}")
        self.logger.info(
            f"   - 过滤后记录数: {post_count} (对齐范围: Mag[{bounds['min_mag']:.1f}, {bounds['max_mag']:.1f}], Plx[{bounds['min_plx']:.1f}, {bounds['max_plx']:.1f}])"
        )

        if dropped_count > 0:
            self.logger.warning(
                f"   - 注意：有 {dropped_count} 颗星因超出动态边界被截断 (通常是因为参考星表包含了部分超大范围源)"
            )

        return v_result

    def align_reference_to_cluster(self, k_ref):
        """基于别名注册表从参考星表中提取特定星团的成员子集。

        Args:
            k_ref (str): 参考星表键值。

        Returns:
            str: 提取后的 stx 视图名称。
        """
        cluster_info = CLUSTERS.get(self.target_cluster)
        if not cluster_info:
            self.logger.error(
                f"❌ config.py 中未找到 TARGET_CLUSTER_ID: {self.target_cluster}"
            )
            return None

        names_to_search = [cluster_info["NAME"]]
        if "ALIAS_NAME" in cluster_info:
            names_to_search.extend(cluster_info["ALIAS_NAME"].values())

        # 3. 格式化为 SQL 的 IN 子句内容: 'Pleiades', 'Melotte_22'
        in_values = ", ".join([f"'{n}'" for n in names_to_search])

        # 4. 获取数据血缘配置
        ref_cfg = DATA[k_ref]
        v_ref = ref_cfg["std_view"]
        v_result = ref_cfg["stx_view"]

        sql = f"SELECT * FROM {v_ref} WHERE {STD_COLS['CLUSTER']} IN ({in_values})"
        try:
            self.db.register_view_from_sql(v_result, sql)
            count = self.db.query(f"SELECT COUNT(*) FROM {v_result}").iloc[0, 0]

            if count > 0:
                self.logger.info(f"✅ 提取成功: {k_ref} -> {v_result}")
                self.logger.info(f"   - 匹配别名: [{in_values}]")
                self.logger.info(f"   - 成员数量: {count}")
            else:
                self.logger.warning(
                    f"⚠️ 警告: 在 {k_ref} 中未找到匹配 {in_values} 的任何记录"
                )
        except Exception as e:
            self.logger.error(f"❌ 提取参考星表子集时发生异常: {str(e)}")
            raise e
        return v_result

    def data_standardize(self, idx_data, cfg_data, manifest, ctx=None):
        """核心标准化调度算法。

        Args:
            idx_data (str): 数据源索引。
            cfg_data (dict): 数据配置字典。
            manifest (dict): 全局清单。
            ctx (dict, optional): 包含星团几何信息的上下文。
        """
        self.logger.info(f"🚀 正在执行数据标准化, 当前表:{idx_data} ")
        actions = cfg_data.get("actions", {})

        for layer in ["std", "stx", "aln"]: 
            if layer in actions:
                self.logger.debug(f"  ∟ 正在执行层级动作: {layer.upper()}")
                action_func = actions[layer]
                action_func(self.db, idx_data, cfg_data, self.manifest, ctx)

    def _get_seeds(self, idx_data, cfg_src, manifest, ctx=None, required_features=None):
        """从指定数据源的标准视图中提取高质量种子星 (RUWE < 1.4)。

        Args:
            idx_data (str): 数据键。
            cfg_src (dict): 配置字典。
            required_features (list): 必须具备的物理特征列。

        Returns:
            pd.DataFrame: 种子星结果集。
        """
        v_src = cfg_src["aln_view"]
        query = f"SELECT * FROM {v_src}"
        df_raw = self.db.query(query)
        
        # 🚀 优化：仅针对当前运行模式所需的特征执行 dropna
        # 这样在 2D 模式下，即便视差 (plx) 缺失，只要自行 (pm) 还在，种子星就不会被丢弃。
        if required_features:
            df_seeds = df_raw.dropna(subset=required_features).copy()
        else:
            df_seeds = df_raw.dropna().copy()
            
        self.logger.info(f"从数据源 [{v_src}] 提取了 {len(df_seeds)} 颗种子星")
        return df_seeds

    def _get_target(self, idx_data, cfg_src, manifest, ctx=None):
        """获取并清洗目标天区数据。

        Args:
            idx_data (str): 数据源索引。
            cfg_src (dict): 数据源配置。

        Returns:
            pd.DataFrame: 有效的天体特征数据。
        """
        cfg_source = manifest[idx_data]
        v_aln = cfg_source["aln_view"]

        sql = f"SELECT * FROM {v_aln}"
        df_target = self.db.query(sql)

        raw_count = len(df_target)
        self.logger.info(f"从视图 [{v_aln}] 读取到原始数据: {raw_count} 颗")

        clean_count = len(df_target)
        if clean_count < raw_count:
            self.logger.warning(
                f"数据预检：剔除了 {raw_count - clean_count} 颗含有 NaN 的无效星。"
                f"进入后续 Pipe 的有效源: {clean_count}"
            )

        return df_target

    def prepare_field_data(self, ref_tables, ctx_cluster):
        """[批量调度] 执行参考星表的层级标准化过程（STD -> STX -> ALN）。

        Args:
            ref_tables (list[str]): 待处理的参考表键名列表。
            ctx_cluster (dict): 当前星团上下文。
        """
        total = len(ref_tables)
        for i, k in enumerate(ref_tables, 1):
            self.logger.info(f"📋 [{i}/{total}] 正在标准化参考星表: {k}")
            self.data_standardize(
                idx_data=k,
                cfg_data=DATA[k],
                manifest=self.manifest,
                ctx=ctx_cluster,
            )

    def post_pipeline(self, t_main_results):
        """算法后处理流水线：生成成员子集视图并计算统计摘要。

        Args:
            t_main_results (str): 算法结果总表名称。

        Returns:
            dict: 包含处理状态、视图名及候选者统计量的字典。
        """
        cluster_name = CLUSTERS[self.target_cluster]["NAME"]
        self.logger.info(
            f"[{cluster_name}] 启动 post_pipeline: 正在生成成员子集视图..."
        )
        try:
            condi_golden = f"{STD_COLS['PROB']} >= {GOLDEN_SAMPLE_THRESHOLD}"
            v_golden = self.get_subset_view(
                t_main_results, tag="golden_members", conditions=condi_golden
            )

            condi_candidates = f"{STD_COLS['PROB']} > {MEMBER_SAMPLE_THRESHOLD}"
            v_candidates = self.get_subset_view(
                t_main_results, tag="candidates", conditions=condi_candidates
            )

            stats_sql = f"""
                SELECT 
                    count(*) FILTER (WHERE {condi_golden} ) as n_golden,
                    count(*) FILTER (WHERE {condi_candidates} ) as n_candidates
                FROM {t_main_results}
            """
            stats = self.db.execute(stats_sql).fetchone()

            self.logger.info("=" * 60)
            self.logger.info(f"📊 [{cluster_name}] Post-Pipeline 后处理数据审计摘要:")
            self.logger.info(f"  🔹 金种子视图 ({v_golden}): {stats[0]} 颗")
            self.logger.info(f"  🔹 候选者视图 ({v_candidates}): {stats[1]} 颗")
            self.logger.info("=" * 60)

            return {
                "status": "success",
                "v_golden": v_golden,
                "v_candidates": v_candidates,
                "stats": {"n_golden": stats[0], "n_candidates": stats[1]},
            }
        except Exception as e:
            self.logger.info(f"Error in post_pipeline: {str(e)}")
            return {"status": "error", "message": str(e)}

    def prepare_audit_data(self, v_source, v_target):
        """预处理审计数据：执行算法候选者与审计目标之间的交叉匹配。

        Args:
            v_source (str): 算法候选者视图名称。
            v_target (str): 审计目标（文献星表）视图名称。

        Returns:
            dict: 包含审计子视图集 (audit_views) 及统计结果 (stats) 的字典。
        """
        if not self._verify_audit_target_exists(v_target):
            self.logger.warning(f"审计目标表 '{v_target}' 不存在。")
            return {
                "status": "warning",
                "message": f"审计目标表 '{v_target}' 不存在，无法执行交叉审计。",
            }

        self.logger.info(f"⚡ 发现审计目标表 '{v_target}'，开始交叉比对...")

        v_cross_audit = self._execute_cross_match_join(v_source, v_target)

        audit_views = self._create_audit_subviews(v_cross_audit)

        stats_cross = self._calculate_audit_stats(audit_views)

        self._log_audit_summary(v_target, v_cross_audit, audit_views, stats_cross)

        return {
            "status": "success",
            "audit_views": audit_views,
            "stats": {"cross_audit_counts": stats_cross},
        }

    def _verify_audit_target_exists(self, v_target: str) -> bool:
        """检查审计目标表在数据库中是否存在。

        Args:
            v_target (str): 目标表名。

        Returns:
            bool: 存在则返回 True。
        """
        if not self.db:
            return False
        sql = f"SELECT 1 FROM information_schema.tables WHERE table_name = '{v_target}'"
        return self.db.con.execute(sql).fetchone() is not None

    def _create_audit_subviews(self, v_cross_audit: str) -> dict:
        """[私有方法] 根据交叉比对分类标签将总表切分为审计子视图。

        Args:
            v_cross_audit (str): 交叉比对大宽表名称。

        Returns:
            dict: 各分类子视图名称映射。
        """
        self.logger.info("📦 正在按标签切分独立审计子视图...")
        
        v_audit_base = TMPL.V_ADT.format(
            category=self.target_category, 
            cluster=self.target_cluster.lower()
        )
        views = {
            "v_cross_audit_total": v_cross_audit,
            "v_matched": f"{v_audit_base}_matched",
            "v_target_only": f"{v_audit_base}_{self.target_category}_only",
            f"v_{IDX_GMM}_only": f"{v_audit_base}_{IDX_GMM}_only"
        }

        category_map = {
            "v_matched": "Matched",
            "v_target_only": "Dismissed by PG",
            f"v_{IDX_GMM}_only": "PG Only"
        }

        for key, category in category_map.items():
            view_name = views[key]
            sql = f"SELECT * FROM {v_cross_audit} WHERE intersection_category = '{category}'"
            self.db.register_view_from_sql(view_name, sql)
            
        return views

    def _calculate_audit_stats(self, audit_views: dict) -> dict:
        """[私有方法] 计算各审计子视图中的天体数量。

        Args:
            audit_views (dict): 包含各子视图名称的字典。

        Returns:
            dict: 统计量映射（如 matched: 120）。
        """
        stats = {}
        keys_to_count = {
            "matched": audit_views["v_matched"],
            f"{self.target_category}_only": audit_views["v_target_only"],
            f"{IDX_GMM}_only": audit_views[f"v_{IDX_GMM}_only"]
        }

        for stat_key, view_name in keys_to_count.items():
            stats[stat_key] = self.db.get_row_count(view_name)
        return stats

    def _log_audit_summary(self, v_target, v_total, views, stats):
        """[私有方法] 向控制台输出审计阶段的统计简报。

        Args:
            v_target (str): 对比参考星表。
            v_total (str): 交叉总视图。
            views (dict): 子视图字典。
            stats (dict): 统计数值字典。
        """
        s_target_key = f"{self.target_category}_only"
        s_gmm_key = f"{IDX_GMM}_only"

        msg = [
            f"{'='*60}",
            f"📊 [交叉审计报告] 参考星表: {v_target}",
            f"  🔗 聚合分析视图: {v_total}",
            f"  🟢 双方评估一致:    {stats['matched']} 颗",
            f"  🟡 算法漏检 (Recall): {stats[s_target_key]} 颗",
            f"  ✨ 算法新增候选:    {stats[s_gmm_key]} 颗",
            f"{'='*60}"
        ]
        for line in msg:
            self.logger.info(line)

    def audit_literature_via_simbad(self, df_subset, label="discovery"):
        """[科研审计] 对特定数据子集执行 SIMBAD 文献核实。

        Args:
            df_subset (pd.DataFrame): 待审计数据子集，需包含 'id' 列。
            label (str): 报告标签，用于区分导出文件名。

        Returns:
            pd.DataFrame: 包含 SIMBAD 文献比对结果的报告。
        """
        if df_subset is None or df_subset.empty:
            self.logger.warning(f"⚠️ 审计子集 [{label}] 为空，跳过文献核实流程。")
            return None

        self.logger.info(f"🔍 启动文献审计子流程: [{label}] (规模: {len(df_subset)} 颗星)")

        validator = UnifiedMemberValidator(
            cluster_id=self.target_cluster, 
            db_instance=self.db
        )

        # 批量同步 SIMBAD 缓存
        star_ids = df_subset["id"].astype(str).tolist()
        df_audit_raw = validator.sync_simbad_cache(star_ids)

        # 结果合并
        df_audit_raw = df_audit_raw.rename(columns={"gaia_dr3_id": "id"})
        df_audit_raw["id"] = df_audit_raw["id"].astype(df_subset["id"].dtype)
        
        final_report = pd.merge(df_subset, df_audit_raw, on="id", how="left")

        cluster_name = CLUSTERS[self.target_cluster]["NAME"]
        report_name = f"Literature_Audit_{label}_{cluster_name}_{datetime.now().strftime('%H%M%S')}.csv"
        output_path = cfg.EXPORT_DIR / report_name
        
        try:
            final_report.to_csv(output_path, index=False)
            self.logger.info(f"✅ SIMBAD 文献审计报告已导出至: {output_path}")
        except Exception as e:
            self.logger.error(f"❌ 导出审计报告失败: {str(e)}")

        return final_report

    def run_audit(self, target):
        """驱动完整审计管线：涵盖数据补全、文献预热、物理校验与结果导出。

        流程包含：数据预处理(补全特征)、文献缓存预热(SIMBAD批量查询)、深度物理核实以及结果落库。

        Args:
            target (str): 待审计的目标视图名称（通常是算法发现的新源）。

        Returns:
            str: 审计报告表名称。
        """
        self.logger.info(f"🔍 🎬 [Workflow] 开始对 {target} 进行身份审计...")

        try:
            v_audit_input = self.pre_audit(target)
            if not v_audit_input:
                self.logger.error("❌ 审计预处理失败，管线熔断。")
                return None

            validator = UnifiedMemberValidator(
                cluster_id=self.target_cluster, db_instance=self.db
            )

            self._warm_up_literature_cache(validator, v_audit_input)

            audit_report_df = validator.run_full_audit_ex(v_audit_input)

            return self._save_audit_report(target, audit_report_df)

        except Exception as e:
            self.logger.error(f"❌ [Workflow] 审计流程运行期间发生严重故障: {str(e)}", exc_info=True)
            raise e

    def _warm_up_literature_cache(self, validator : UnifiedMemberValidator, v_source):
        """[私有方法] 提取视图中所有天体 ID 并触发文献缓存预热。

        利用 DuckDB 的 ANTI JOIN 在数据库侧直接计算差集，仅提取本地缺失的 ID，
        显著提升百万级数据下的预热效率。

        Args:
            validator (UnifiedMemberValidator): 验证器实例。
            v_source (str): 包含待验证 ID 的视图名。
        """
        cache_table = validator.cache_table
        # 找出在 v_source 中存在但 cache_table 中没有的 ID
        sql_missing = f"""
            SELECT DISTINCT CAST(v.id AS VARCHAR) as id
            FROM {v_source} v
            ANTI JOIN {cache_table} c ON CAST(v.id AS VARCHAR) = c.gaia_dr3_id
        """
        
        self.logger.info(f"🔍 正在检索 [{v_source}] 中缺失的文献缓存记录...")
        df_missing = self.db.con.execute(sql_missing).df()
        ids_to_sync = df_missing["id"].tolist()

        if not ids_to_sync:
            self.logger.info("✅ 缓存对齐完成：所有源均已在本地缓存中，跳过网络同步。")
            return

        self.logger.info(f"🌐 正在为 {len(ids_to_sync)} 个缺失源启动增量 SIMBAD 预热...")
        validator.sync_simbad_cache(ids_to_sync)

    def _save_audit_report(self, target, df_report):
        """[私有方法] 将审计结果固化至数据库物理表。

        Args:
            target (str): 原始目标名称（用于派生表名）。
            df_report (pd.DataFrame): 审计结果数据帧。

        Returns:
            str: 固化后的表名。
        """
        if df_report is None or df_report.empty:
            self.logger.warning("⚠️ 验证器返回的审计数据为空，跳过落库流程。")
            return None

        output_table = f"{target}_audited"
        self.db.register_table_from_df(output_table, df_report)
        
        self.logger.info(f"✅ [Workflow] 审计报告已保存至数据库表: {output_table}")
        return output_table

    def pre_audit(self, v_target):
        """审计前准备：补全物理参数。

        Args:
            v_target (str): 算法结果视图。

        Returns:
            str: 审计输入视图名。
        """
        self.logger.info(f"🔧 正在准备审计数据视图...")

        try:
            # 调用 DB 层提供的标准化审计输入视图构建接口
            field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
            t_base = DATA[field_idx]['stx_view']
            v_result = self.db.register_audit_input_view(v_target, t_base)
            self.logger.info(f"✅ 审计数据准备完成，输入视图: {v_result}")
            return v_result
        except Exception as e:
            self.logger.error(f"❌ 审计数据准备失败: {str(e)}")
            return None

    def post_audit(self, t_audit_report):
        """审计后处理：挖掘误报与遗漏样本的物理分布特征。

        Args:
            t_audit_report (str): 审计报告表名。
        """
        self.logger.info(f"🔍 正在对审计结果进行深入分析...")

        if t_audit_report is None:
            self.logger.warning("⚠️ 审计报告表名为空，无法执行后续分析。")
            return

        try:
            sql_analysis = f"""
                SELECT 
                    audit_result,
                    AVG(plx) AS avg_plx,
                    AVG(pmra) AS avg_pmra,
                    AVG(pmdec) AS avg_pmdec,
                    AVG(mag) AS avg_mag,
                    COUNT(*) AS count
                FROM {t_audit_report}
                GROUP BY audit_result
            """
            analysis_results = self.db.query(sql_analysis)
            self.logger.info(f"📊 审计结果物理特征分析:\n{analysis_results}")

        except Exception as e:
            self.logger.error(f"❌ 审计后分析失败: {str(e)}")

    def _execute_cross_match_join(self, v_src: str, v_target: str) -> str:
        """[私有方法] 执行核心交叉审计连接。

        Args:
            v_src: 算法候选者视图。
            v_target: 审计目标视图。

        Returns:
            str: 交叉审计总视图名。
        """
        v_audit_base = TMPL.V_ADT.format(category=self.target_category, cluster=self.target_cluster.lower())
        v_cross_audit = v_audit_base + "_cross"

        self.logger.info(f"💾 正在构建全外连接总表视图: {v_cross_audit}")

        # 利用 COALESCE 保证 ID 绝不丢失；通过 CASE WHEN 划分三类归属
        cross_audit_query = f"""
            WITH pg_set AS (
                SELECT id, {STD_COLS['PROB']} AS prob FROM {v_src}
            ),
            target_set AS (
                SELECT id FROM {v_target}
            )
            SELECT 
                COALESCE(p.id, h.id) AS id,
                p.prob,
                CASE 
                    WHEN p.id IS NOT NULL AND h.id IS NOT NULL THEN 'Matched'
                    WHEN p.id IS NULL AND h.id IS NOT NULL     THEN 'Dismissed by PG'
                    WHEN p.id IS NOT NULL AND h.id IS NULL     THEN 'PG Only'
                END AS intersection_category
            FROM pg_set p
            FULL OUTER JOIN target_set h ON p.id = h.id;
        """

        # 标准化注册
        self.db.register_view_from_sql(v_cross_audit, cross_audit_query)

        total_rows = self.db.get_row_count(v_cross_audit)
        self.logger.info(f"✅ 交叉比对大宽表构建成功: {v_cross_audit} (记录数: {total_rows})")
        return v_cross_audit

    def _parse_pipeline_config(self) -> tuple[dict, str, list[str]]:
        """[私有方法] 原子拆解：解析 GMM 配置项与特征空间。

        Returns:
            tuple: (配置字典, 运行模式字符串, 特征列名列表)。
        """
        feature_map = GMM_CONFIG.get("feature_map", {})
        self.logger.debug(f"当前 GMM_CONFIG 中的 feature_map 配置:\n {feature_map}")

        current_mode = GMM_CONFIG.get("dim_mode", "3d")
        if current_mode not in feature_map:
            raise ValueError(
                f"未知的运行模式: {current_mode}，请核实 feature_map 配置。"
            )

        required_features = feature_map[current_mode]
        self.logger.info(
            f"🌌 当前运行模式: [{current_mode}], 所需核心特征空间: {required_features}"
        )
        return GMM_CONFIG, current_mode, required_features

    def _load_raw_astrometry_data(
        self, ctx_cluster, required_features=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """[私有方法] 原子拆解：加载全域天区与种子星原始数据。

        Returns:
            tuple: (全域数据 DataFrame, 种子星数据 DataFrame)。
        """
        field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
        df_target_raw = self._get_target(
            idx_data=field_idx,
            cfg_src=DATA[field_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
        )
        self.logger.info(f"目标天区 source 原始数量: {len(df_target_raw)}")

        seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
        df_seeds_raw = self._get_seeds(
            idx_data=seed_idx,
            cfg_src=DATA[seed_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
            required_features=required_features
        )
        self.logger.info(f"目标天区 seeds 原始数量: {len(df_seeds_raw)}")
        return df_target_raw, df_seeds_raw

    def _transform_and_bridge_features(
        self, df_raw: pd.DataFrame, ctx_cluster, mode: str, required_features: list[str]
    ) -> pd.DataFrame:
        """[私有方法] 特征转换网关：将原始坐标转换为目标物理维度特征。

        Args:
            df_raw: 原始 DataFrame。
            ctx_cluster: 星团上下文。
            mode: 运行模式 (e.g., '3d', '6d_p')。
            required_features: 所需特征列名列表。

        Returns:
            pd.DataFrame: 扩展后的 DataFrame。
        """
        if df_raw is None:
            self.logger.error(
                "❌ [Bridge] 输入的原始 DataFrame 为 None，无法进行特征转换！"
            )
            return None

        cluster_rv = ctx_cluster.get("RV_REF", None)
        c_ra = ctx_cluster.get("CENTER_RA", None)
        c_dec = ctx_cluster.get("CENTER_DEC", None)
        cluster_center = (
            (c_ra, c_dec) if (c_ra is not None and c_dec is not None) else None
        )

        transformer = AstroTransformer(
            cluster_rv=cluster_rv, cluster_center_icrs=cluster_center
        )
        # TODO: transformer.ingest_external_rv_data(df_raw)
        X_array = transformer.fit_transform(df_raw, mode=mode)

        if X_array.shape[1] != len(required_features):
            raise KeyError(
                f"Transformer 转换矩阵列数与配置不匹配！"
            )

        cols_upper = [col.upper() for col in required_features]
        cols_lower = [col.lower() for col in required_features]

        # 提取转换后的特征矩阵 (在此之前不得删除原始列)
        df_features = pd.DataFrame(X_array, columns=required_features, index=df_raw.index)

        existing_dup_cols = [col for col in df_raw.columns if col in (cols_upper + cols_lower)]
        if existing_dup_cols:
            self.logger.info(
                f"🔄 [Bridge] 模式 [{mode}] 触发列名防重机制，从原始表中移除了已存在的列: {existing_dup_cols}"
            )
            df_raw = df_raw.drop(columns=existing_dup_cols)
        df_extended = pd.concat([df_raw, df_features], axis=1)
        return df_extended

    def _defensive_nan_purge(
        self, df_extended: pd.DataFrame, required_features: list[str], label: str
    ) -> pd.DataFrame:
        """[私有方法] 原子拆解：特征清洗，剔除指定特征列中含 NaN 的记录。

        Args:
            df_extended: 特征转换后的 DataFrame。
            required_features: 必须具备的特征列。
            label: 用于日志记录的标签名。

        Returns:
            pd.DataFrame: 清洗后的纯净数据。
        """
        if df_extended is None:
            self.logger.error(
                f"❌ [数据清洗 - {label}] 数据为空，无法进行无效值过滤。"
            )
            return pd.DataFrame()

        initial_count = len(df_extended)

        df_clean = df_extended.dropna(subset=required_features).copy()
        dropped = initial_count - len(df_clean)

        if dropped > 0:
            self.logger.warning(
                f"⚠️ [防御性过滤 - {label}]: 剔除了 {dropped} 颗特征不完整(含NaN)的天体，剩余有效样本: {len(df_clean)}。"
            )
        else:
            self.logger.info(f"✅ [数据预检 - {label}] 样本特征完备，共计 {len(df_clean)} 颗星。")
        return df_clean

    @astro_checkpoint(cache_table_name="cache_full_pipeline_result", force_refresh=True)
    def run_pipeline(self, ctx_cluster):
        """驱动核心 GMM 计算流水线：执行双轨制内核推理并固化结果。

        Args:
            ctx_cluster (dict): 星团上下文环境。

        Returns:
            str: 算法结果在数据库中的固化表名。
        """
        gmm_cfg, current_mode, required_features = self._parse_pipeline_config()

        use_experimental = gmm_cfg.get("use_experimental", False)
        kernel_name = "PriorGMMEx" if use_experimental else "PriorGMM"
        self.logger.info(f"🧪 [双轨制触发] 当前任务分配至内核 [{kernel_name}] 运行。")
        engine = (
            PriorGMMEx(config=gmm_cfg) if use_experimental else PriorGMM(config=gmm_cfg)
        )

        self.logger.info("📡 正在准备特征工程输入数据...")
        df_target_raw, df_seeds_raw = self._load_raw_astrometry_data(ctx_cluster)

        self.logger.info(f"⚡ 正在转换特征空间为 [{current_mode.upper()}]...")
        df_target_ext = self._transform_and_bridge_features(
            df_target_raw, ctx_cluster, current_mode, required_features
        )
        df_seeds_ext = self._transform_and_bridge_features(
            df_seeds_raw, ctx_cluster, current_mode, required_features
        )

        self.logger.info("🧹 正在执行特征清洗与 NaN 防御...")
        df_target_final = self._defensive_nan_purge(
            df_target_ext, required_features, label="Target全域"
        )
        df_seeds_final = self._defensive_nan_purge(
            df_seeds_ext, required_features, label="Seeds种子星"
        )

        self.logger.info(f"🔥 开始驱动 {kernel_name} 引擎计算...")
        params = engine.fit(df_seeds_final, df_target_final)
        df_prob = engine.predict(df_target_final, params)

        cluster_name = CLUSTERS[self.target_cluster]["NAME"]
        v_res_pg = cfg.TMPL.T_RES_SG.format(cluster=cluster_name)
        self.db.register_table_from_df(v_res_pg, df_prob)
        self.db.save_to_warehouse(v_res_pg)

        self.logger.debug(df_prob.head(5))
        return v_res_pg
