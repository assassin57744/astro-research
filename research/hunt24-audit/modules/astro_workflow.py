# -*- coding: utf-8 -*-
"""
modules/workflow.py

天文数据处理工作流编排引擎。
集成了双轨制（稳定旧轨/实验新轨）精筛洗涤机制，支持高维贝叶斯对抗与自适应种子粗筛。
该模块负责调度从原始数据同步、特征工程变换到聚类消歧与多维度物理审计的全流程。
"""

import logging
import pandas as pd
import numpy as np
from astroquery.simbad import Simbad  # pylint: disable=unused-import

from utils.decorators import astro_checkpoint
from modules.astro_db import AstroDB
from modules.pg_core import PriorGMM
from modules.validator import UnifiedMemberValidator
from modules.transformer import AstroTransformer
from modules.reporter import render_final_report, render_all_modes_comparison
from modules.cluster import StarCluster

# 🚀 实验新轨核心模块引入
from modules.cluster_seed_extractor import ClusterSeedExtractor
from modules.astro_membership.disambiguation.bayesian import BayesianGmmDisambiguation
from modules.astro_membership.disambiguation.threshold import ThresholdGmmDisambiguation
from modules.astro_membership.disambiguation.blind import BlindGmmDisambiguation

from modules.astro_membership.substructure.identity import IdentityComponentModeller
from modules.astro_membership.substructure.dual_component import DualComponentModeller
from modules.astro_membership.substructure.triple_component import TripleComponentModeller

from modules.astro_membership.phase2_orchestrator import Phase2Orchestrator
from modules.astro_membership.cutter import DensityFieldCutter

import config as cfg
from config import (
    CLUSTERS,
    MANIFEST,
    GMM_CONFIG,
    MEMBER_SAMPLE_THRESHOLD,
    STD_COLS,
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
        t_master (str): 动态生成的 Master 结果表名，用于固化中间状态。
    """

    def __init__(
        self,
        db_instance: AstroDB | None = None,
        target_cluster=None,
        target_category=None,
        feature_space="5d",
        algo="dbscan",
    ):
        """初始化工作流实例。

        Args:
            db_instance (AstroDB | None): 活跃原生数据库实例，传入 None 时内部自动构建。
            target_cluster (str): 目标星团唯一标识符（如 'M41', 'M45'
            target_category (str): 目标文献比对类别（如 'hunt'）。
            mode (str): 动力学特征空间维度模式 ('3d', '5d', '6d')。
            algo (str): 核心聚类识别算法标识（用于实验新轨多态路由）。
        """
        if db_instance is None:
            self.logger_init = logging.getLogger(f"AstroPipeline.Init")
            self.logger_init.info("🛠️ 未检测到活跃 DB 实例，正在初始化默认 AstroDB 引擎...")
            self.db = AstroDB(manifest=cfg.MANIFEST)
            self._owned_db = True
        else:
            self.db = db_instance
            self._owned_db = False

        self.target_cluster = target_cluster
        self.target_category = target_category
        self.feature_space = feature_space
        self.algo = algo
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        
        self.manifest = getattr(self.db, "data_manifest", {})
        
        # 构建当前运行环境下的全局 Master 表名标识符
        self.t_master = cfg.TMPL.T_MASTER.format(
            cluster=self.target_cluster.lower(),
            category=self.target_category,
            mode=self.feature_space,
            algo=self.algo,
        )
        
        self.logger.info(f"✅ 工作流实例初始化完成: Cluster={target_cluster}, Mode={feature_space}, Algo={algo}")
        self.logger.info(f"📍 目标 Master 状态表: {self.t_master}")

    def data_standardize(self, idx_data, cfg_data, manifest, ctx=None):
        """核心标准化调度算法。将原始星表转化为标准化的中间视图 (STD -> STX -> ALN)。

        Args:
            idx_data (str): 数据源索引键名。
            cfg_data (dict): 数据源配置字典（含 actions）。
            manifest (dict): 全局数据清单。
            ctx (dict, optional): 包含星团几何信息的上下文（如中心坐标、参考距离）。
        """
        self.logger.info(f"🚀 [Standardize] 正在处理源: {idx_data}")

        # 🛡️ 核心重构：解析 CAT_NAME 优先级。确保在跨星表比对时，不同源对同一星团的命名对齐。
        local_ctx = ctx.copy() if ctx else {}
        cluster_id = local_ctx.get("id")

        # 默认 CAT_NAME 使用星团的 ID_NAME (Gaia 官方名) 或 NAME (常用名)
        local_ctx.setdefault(
            "CAT_NAME", local_ctx.get("ID_NAME", local_ctx.get("NAME"))
        )

        if cluster_id:
            adapter = getattr(cfg, "CATALOG_NAMING_ADAPTER", {})
            override_name = adapter.get(idx_data, {}).get(cluster_id)
            if override_name:
                self.logger.info(f"  ∟ 📝 发现命名适配规则: {idx_data} -> {cluster_id} 使用别名 [{override_name}]")
                local_ctx["CAT_NAME"] = override_name

        actions = cfg_data.get("actions", {})
        for layer in ["std", "stx", "aln"]:
            if layer in actions:
                self.logger.debug(f"  ∟ 执行层级转换: {layer.upper()}")
                action_func = actions[layer]
                # 将修正后的上下文传给执行层（如 modules/transformer.py 中的标准化函数）
                action_func(self.db, idx_data, cfg_data, self.manifest, local_ctx)
        
        self.logger.info(f"✅ 数据源 [{idx_data}] 标准化链路执行完毕。")

    def _get_seeds(self, idx_data, src, manifest, ctx=None, required_features=None):
        """从指定数据源的标准视图中提取高质量种子星。

        根据不同星团的配置执行初步筛选，并为这些天体打上 'raw_seed' 标签。

        Args:
            idx_data (str): 数据源键名。
            src (dict): 配置字典。
            required_features (list, optional): 必须具备的物理特征列（用于提前清洗缺失值）。

        Returns:
            pd.DataFrame: 种子星结果集。
        """
        v_src = src["aln_view"]
        self.logger.info(f"🔍 [Seeds] 正在从视图 [{v_src}] 提取初始种子星...")
        
        query = f"SELECT * FROM {v_src}"
        df_raw = self.db.query(query)
        self.logger.info(f"  ∟ 原始读取: {len(df_raw)} 颗")

        # 🚀 仅针对当前运行模式所需的特征执行 dropna
        if required_features:
            # 过滤掉在当前 2D/3D/5D 模式下缺失核心参数的样本
            # 派生特征（如 l, b, U, V, W）此时尚未生成，将在 Transformer 转换后的 _defensive_nan_purge 中处理
            available_features = [f for f in required_features if f in df_raw.columns]
            df_seeds = df_raw.dropna(subset=available_features).copy()
            dropped = len(df_raw) - len(df_seeds)
            if dropped > 0:
                self.logger.info(f"  ∟ 物理缺损过滤: 剔除 {dropped} 颗缺失关键特征 [{available_features}] 的天体")
        else:
            df_seeds = df_raw.dropna().copy()

        self.logger.info(f"✅ 最终提取有效种子星: {len(df_seeds)} 颗")

        # 🚀 初始化 Master 表中的种子标签：初步标记为 'raw_seed'，后续会被 GMM 细化为 Core/Noise
        df_tag = df_seeds[[cfg.STD_COLS["ID"]]].copy()
        df_tag["seed_type"] = "raw_seed"
        self.db.tag_master_table(self.t_master, df_tag)
        return df_seeds

    def _get_target(self, idx_data, cfg_src, manifest, ctx=None):
        """获取并清洗目标天区（靶场）数据。

        Args:
            idx_data (str): 数据源索引。
            cfg_src (dict): 数据源配置。

        Returns:
            pd.DataFrame: 目标天区的全量天体特征数据。
        """
        cfg_source = manifest[idx_data]
        v_aln = cfg_source["aln_view"]

        self.logger.info(f"🎯 [Target] 正在加载靶场天区数据: {v_aln}")
        sql = f"SELECT * FROM {v_aln}"
        df_target = self.db.query(sql)

        raw_count = len(df_target)
        self.logger.info(f"✅ 靶场数据读取完成，共计 {raw_count} 颗源。")

        return df_target

    def data_standardize_all(self, ref_tables, ctx_cluster):
        """[批量调度] 执行参考星表的层级标准化过程。

        Args:
            ref_tables (list[str]): 待处理的参考表（如 Gaia, Lit, Simbad）键名列表。
            ctx_cluster (dict): 当前星团物理上下文。
        """
        total = len(ref_tables)
        self.logger.info(f"📋 开始执行批量标准化任务 (共 {total} 个数据源)...")
        for i, k in enumerate(ref_tables, 1):
            self.logger.info(f"  ∟ [{i}/{total}] 正在同步并标准化参考星表: {k}")
            self.data_standardize(
                idx_data=k,
                cfg_data=MANIFEST[k],
                manifest=self.manifest,
                ctx=ctx_cluster,
            )
        self.logger.info("✅ 所有相关数据源标准化对齐完成。")

    def post_pgmm(self, t_main_results):
        """算法后处理流水线：生成成员子集视图并计算统计摘要。

        将算法产出的连续概率 (0-1) 离散化为金种子 (Golden)、候选者 (Candidate) 等逻辑标签。

        Args:
            t_main_results (str): 算法结果固化表名称 (t_master)。
        """
        self.logger.info(f"📊 [{self.target_cluster}] 启动 Post-Pipeline: 执行主表状态同步与统计...")
        try:
            # 1. 动态增加成员分类标签列 (如果不存在)
            self.db.execute(
                f"ALTER TABLE {self.t_master} ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE"
            )
            self.db.execute(
                f"ALTER TABLE {self.t_master} ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE"
            )

            # 2. 直接在 Master 表中执行状态标记
            condi_golden = f"{STD_COLS['PROB']} >= {GOLDEN_SAMPLE_THRESHOLD}"
            condi_candidates = f"{STD_COLS['PROB']} > {MEMBER_SAMPLE_THRESHOLD}"

            self.db.execute(
                f"UPDATE {self.t_master} SET is_golden = TRUE WHERE {condi_golden}"
            )
            self.db.execute(
                f"UPDATE {self.t_master} SET is_candidate = TRUE WHERE {condi_candidates}"
            )

            # 3. 执行多维度数据统计摘要
            stats_sql = f"""
                SELECT 
                    count(*) FILTER (WHERE is_golden = TRUE) AS n_golden,
                    count(*) FILTER (WHERE is_candidate = TRUE) AS n_candidates,
                    count(*) FILTER (WHERE seed_type = 'raw_seed') AS n_seeds,
                    count(*) FILTER (WHERE density_status = 'core') AS n_seed_core,
                    count(*) FILTER (WHERE density_status = 'noise') AS n_seed_noise
                FROM {self.t_master}
            """
            stats = self.db.execute(stats_sql).fetchone()
            n_golden, n_candidates, n_seeds, n_seed_core, n_seed_noise = stats

            self.logger.info("=" * 65)
            self.logger.info(f"📈 [{self.target_cluster}] 算法识别统计摘要:")
            self.logger.info(f"  🔹 高置信金种子星 (P >= {GOLDEN_SAMPLE_THRESHOLD}): {n_golden} 颗")
            self.logger.info(f"  🔹 成员星候选总数 (P > {MEMBER_SAMPLE_THRESHOLD}): {n_candidates} 颗")
            self.logger.info(f"  🔹 原始输入种子星 (Total Seeds): {n_seeds} 颗")
            self.logger.info(f"  🔹 种子集中核心样本 (Core Clusters): {n_seed_core} 颗")
            self.logger.info(f"  🔹 种子集中被判定为背景的样本 (Noise): {n_seed_noise} 颗")
            self.logger.info("=" * 65)

            # 为后续深度审计步骤提供一个逻辑上的“候选者”入口视图
            v_candidates = f"v_candidates_{self.target_cluster.lower()}"
            self.db.register_view_from_sql(
                v_candidates, f"SELECT * FROM {self.t_master} WHERE is_candidate = TRUE"
            )

            return {
                "status": "success",
                "v_candidates": v_candidates,
                "stats": {
                    "n_golden": n_golden,
                    "n_candidates": n_candidates,
                    "n_seeds": n_seeds,
                    "n_seed_core": n_seed_core,
                    "n_seed_noise": n_seed_noise,
                },
            }
        except Exception as e:
            self.logger.error(f"❌ Post-Pipeline 发生异常: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def prepare_audit_data(self, v_source, v_target):
        """预处理审计数据：执行算法候选者与文献比对目标之间的交叉匹配。

        该步骤会识别出：
        1. Matched: 算法与文献均承认的成员。
        2. PG Only: 算法新发现、文献未记载的成员（审计重点）。
        3. Ref Only: 文献记载但算法未通过的成员（查漏补缺）。

        Args:
            v_source (str): 算法候选者视图名称。
            v_target (str): 审计参考目标（文献星表）视图名称。

        Returns:
            dict: 包含统计结果和审计分流视图名。
        """
        if not self._verify_audit_target_exists(v_target):
            self.logger.warning(f"⚠️ 审计目标视图 '{v_target}' 在 DB 中不存在，跳过交叉比对。")
            return {
                "status": "warning",
                "message": f"审计目标表 '{v_target}' 不存在，无法执行交叉审计。",
            }

        self.logger.info(f"⚡ [CrossMatch] 发现审计参考源 '{v_target}'，启动多表联查...")

        # 🚀 [混合模式重构] 在 Master 表中直接更新交叉匹配标签列 (x_match_tag)
        col_x = cfg.MASTER_COLS["X_MATCH"]
        sql_cross = f"""
            SELECT 
                COALESCE(m.id, h.id) as id,
                CASE 
                    WHEN m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} AND h.id IS NOT NULL THEN 'Matched'
                    WHEN m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} AND h.id IS NULL     THEN 'PG Only'
                    WHEN (m.id IS NULL OR m.prob <= {cfg.MEMBER_SAMPLE_THRESHOLD} OR m.prob IS NULL) 
                        AND h.id IS NOT NULL THEN 'Ref Only'
                END as {col_x}
            FROM {self.t_master} m
            FULL OUTER JOIN {v_target} h ON m.id = h.id
            WHERE m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} OR h.id IS NOT NULL
        """
        df_x = self.db.query(sql_cross)
        self.db.tag_master_table(self.t_master, df_x)

        self.logger.info(f"📥 交叉比对结果已同步至 Master 表 [{self.t_master}]。")

        # 为深度审计准备分流视图
        v_audit_pg_only = f"v_tmp_audit_pg_only"
        v_audit_ref_only = f"v_tmp_audit_ref_only"
        self.db.register_view_from_sql(
            v_audit_pg_only, f"SELECT * FROM {self.t_master} WHERE {col_x} = 'PG Only'"
        )
        self.db.register_view_from_sql(
            v_audit_ref_only,
            f"SELECT * FROM {self.t_master} WHERE {col_x} = 'Ref Only'",
        )

        # 统计分布并打印
        st_sql = f"SELECT {col_x}, count(*) FROM {self.t_master} WHERE {col_x} IS NOT NULL GROUP BY {col_x}"
        stats_raw = self.db.execute(st_sql).fetchall()
        stats_cross = {row[0]: row[1] for row in stats_raw}
        
        self.logger.info("-" * 65)
        self.logger.info(f"📊 [交叉审计分流统计]")
        self.logger.info(f"    ✅ 两者匹配 (Matched):  {stats_cross.get('Matched', 0)}")
        self.logger.info(f"    🆕 算法独有 (PG Only): {stats_cross.get('PG Only', 0)}")
        self.logger.info(f"    🔻 文献独有 (Ref Only): {stats_cross.get('Ref Only', 0)}")
        self.logger.info("-" * 65)

        return {
            "status": "success",
            "v_audit_pg_only": v_audit_pg_only,
            "v_audit_ref_only": v_audit_ref_only,
            "stats": stats_cross,
        }

    def _verify_audit_target_exists(self, v_target: str) -> bool:
        """检查审计目标表在数据库中是否存在。

        Args:
            v_target (str): 目标表名.

        Returns:
            bool: 存在则返回 True。
        """
        if not self.db:
            return False
        sql = f"SELECT 1 FROM information_schema.tables WHERE table_name = '{v_target}'"
        return self.db.con.execute(sql).fetchone() is not None

    def run_audit(self, target, audit_type="default"):
        """驱动完整深度审计管线：涵盖特征补全、文献预热、物理校验与结果导出。

        流程包含：数据预处理(补全特征)、文献缓存预热(SIMBAD批量查询)、深度物理核实以及结果落库。

        Args:
            target (str): 待审计的目标视图名称。
            audit_type (str): 审计分流标识 (pg_only/ref_only)。

        Returns:
            str: 审计报告视图名称。
        """
        self.logger.info(f"🔍 🎬 [Audit] 启动深度物理审计流程 -> 目标: [{target}] 类型: [{audit_type}]")

        try:
            # 1. 补全缺失的物理参数 (BP-RP, Gmag 等用于 CMD 校验的参数)
            v_audit_input = self.pre_audit(target)
            if not v_audit_input:
                self.logger.error("❌ 审计预处理(物理特征补全)失败，审计管线熔断。")
                return None

            # 2. 初始化验证器引擎
            validator = UnifiedMemberValidator(
                cluster_id=self.target_cluster, db_instance=self.db, mode=self.feature_space
            )

            # 3. 预热文献缓存 (SIMBAD/Parent ID 批量查询)
            self._warm_up_literature_cache(validator, v_audit_input)

            # 4. 执行多维度全量审计 (Isochrone Consistency + Literature Crossmatch)
            self.logger.info("⚡ 正在执行物理一致性校验 (CMD Residuals) 与文献身份核实...")
            audit_report_df = validator.run_full_audit_ex(v_audit_input)

            # 5. [混合模式重构] 审计结果回灌 Master 表
            self.logger.info(f"📥 正在将深度审计结果 ({len(audit_report_df)} 颗) 同步回灌至 Master 表...")
            self.db.tag_master_table(self.t_master, audit_report_df)

            # 6. 生成特定分流的报告视图
            v_report = f"{self.t_master}_{audit_type}_audited_report"
            col_x = cfg.MASTER_COLS["X_MATCH"]
            x_match_val = (
                "PG Only"
                if audit_type == "pg_only"
                else "Ref Only" if audit_type == "ref_only" else None
            )

            if x_match_val:
                sql_filter = f"SELECT * FROM {self.t_master} WHERE audit_status IS NOT NULL AND {col_x} = '{x_match_val}'"
            else:
                sql_filter = f"SELECT * FROM {self.t_master} WHERE audit_status IS NOT NULL"

            self.db.register_view_from_sql(v_report, sql_filter)
            self.logger.info(f"✅ 深度审计完成。生成结果报表视图: {v_report}")
            return v_report

        except Exception as e:
            self.logger.error(f"❌ [Audit] 流程运行期间发生严重故障: {str(e)}", exc_info=True)
            raise e

    def _warm_up_literature_cache(self, validator: UnifiedMemberValidator, v_source):
        """[私有方法] 提取视图中所有天体 ID 并触发文献缓存预热。

        采用 LEFT JOIN 策略识别本地缺失或 parent 字段尚未拉取的记录。

        Args:
            validator (UnifiedMemberValidator): 验证器实例。
            v_source (str): 包含待验证 ID 的视图名。
        """
        cache_table = validator.cache_table
        self.logger.info(f"🌐 [Cache] 正在检查本地文献缓存状态 (表: {cache_table})...")

        # 确保缓存表已经存在且具备 parent 字段（用于动态演进）
        res = self.db.con.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = ?)",
            [cache_table.lower()],
        ).fetchone()

        if res and res[0]:
            col_info = self.db.con.execute(f"PRAGMA table_info({cache_table})").df()
            if "parent" not in col_info["name"].values:
                self.logger.warning(f"⚠️ [Schema] 本地缓存表缺失 `parent` 字段，正在执行 DDL 补全...")
                try:
                    self.db.con.execute(f"ALTER TABLE {cache_table} ADD COLUMN parent VARCHAR;")
                except Exception as ddl_err:
                    self.logger.error(f"❌ 动态追加 parent 列失败: {str(ddl_err)}")

        # 识别需要从 SIMBAD 网络同步的 ID (从未拉取过，或者 parent 字段为空/None)
        sql_missing = f"""
            SELECT DISTINCT CAST(v.id AS VARCHAR) as id
            FROM {v_source} v
            LEFT JOIN {cache_table} c ON CAST(v.id AS VARCHAR) = c.gaia_dr3_id
            WHERE c.gaia_dr3_id IS NULL 
               OR c.parent IS NULL 
               OR TRIM(c.parent) = '' 
               OR LOWER(TRIM(c.parent)) = 'none'
        """

        df_missing = self.db.con.execute(sql_missing).df()
        ids_to_sync = df_missing["id"].tolist()

        if not ids_to_sync:
            self.logger.info("✅ 缓存预检通过：所有目标天体均已在本地缓存库中。")
            return

        self.logger.info(f"🌐 检测到 {len(ids_to_sync)} 条记录缺失或需要刷新文献 parent 信息，启动增量同步...")
        validator.sync_simbad_cache(ids_to_sync)
        self.logger.info(f"✅ 文献缓存预热同步任务执行完毕。")

    def pre_audit(self, v_target):
        """审计前准备：关联 STX 标准化视图以补全物理光度参数。
        Args:
            v_target (str): 算法结果视图。

        Returns:
            str: 审计输入视图名。
        """
        self.logger.info(f"🔧 [Pre-Audit] 正在关联物理背景表以补全光度学特征...")
        try:
            field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
            t_base = MANIFEST[field_idx]["stx_view"]
            # 调用 DB 层接口将算法 ID 列表与原始星表字段联查
            v_result = self.db.register_audit_input_view(v_target, t_base)
            self.logger.info(f"  ∟ 关联成功。物理特征输入源: {v_result}")
            return v_result
        except Exception as e:
            self.logger.error(f"❌ 审计数据准备失败: {str(e)}")
            return None

    def _parse_pipeline_config(self) -> tuple[dict, list[str]]:
        """[私有方法] 原子拆解：解析当前模式下的 GMM 配置项与对应的物理特征空间。
        Returns:
            tuple: (配置字典, 特征列名列表)。
        """
        gmm_cfg = GMM_CONFIG.copy()
        current_mode = self.feature_space
        gmm_cfg["dim_mode"] = current_mode

        feature_map = gmm_cfg.get("feature_map", {})
        if current_mode not in feature_map:
            raise ValueError(f"❌ 未知的运行模式: [{current_mode}]，请检查 config.py 中的 feature_map 配置。")

        required_features = feature_map[current_mode]
        self.logger.info(f"🌌 [Pipeline] 当前运行模式: [{current_mode}] | 特征维度: {len(required_features)}D")
        self.logger.info(f"  ∟ 选定物理坐标空间: {required_features}")
        return gmm_cfg, required_features

    def _transform_and_bridge_features(
        self, df_raw: pd.DataFrame, ctx_cluster, mode: str, required_features: list[str]
    ) -> pd.DataFrame:
        """[私有方法] 特征转换网关：将原始坐标(RA/Dec/Plx/PM)转换为物理特征(如 L/B/U/V/W)。"""
        if df_raw is None or df_raw.empty:
            self.logger.error("❌ [Transform] 输入数据为空，无法执行特征工程。")
            return None

        # 提取星团中心作为变换参考系
        cluster_rv = ctx_cluster.get("RV_REF", None)
        c_ra = ctx_cluster.get("CENTER_RA", None)
        c_dec = ctx_cluster.get("CENTER_DEC", None)
        cluster_center = (c_ra, c_dec) if (c_ra is not None and c_dec is not None) else None

        self.logger.info(f"⚡ [Transform] 启动 AstroTransformer 物理坐标变换 (Mode: {mode})...")
        transformer = AstroTransformer(
            cluster_rv=cluster_rv, cluster_center_icrs=cluster_center
        )
        # TODO: transformer.ingest_external_rv_data(df_raw)
        
        # 执行转换 (ICRS -> Cartesian/Galactic/Velocity Space)
        X_array = transformer.fit_transform(df_raw, mode=mode)

        if X_array.shape[1] != len(required_features):
            raise KeyError(f"🚨 [Error] Transformer 转换矩阵列数({X_array.shape[1]})与预期配置({len(required_features)})不匹配！")

        # 特征映射与重复列清理
        df_features = pd.DataFrame(X_array, columns=required_features, index=df_raw.index)
        existing_dup_cols = [col for col in df_raw.columns if col in required_features]
        if existing_dup_cols:
            self.logger.debug(f"  ∟ 发现并清理重名特征列: {existing_dup_cols}")
            df_raw = df_raw.drop(columns=existing_dup_cols)
            
        df_extended = pd.concat([df_raw, df_features], axis=1)
        self.logger.info(f"✅ 特征转换完成。新增特征: {required_features}")
        return df_extended

    def _defensive_nan_purge(
        self, df_extended: pd.DataFrame, required_features: list[str], label: str
    ) -> pd.DataFrame:
        """[私有方法] 原子拆解：特征清洗，剔除指定特征列中含 NaN 的物理残损记录。"""
        if df_extended is None or df_extended.empty:
            return pd.DataFrame()

        initial_count = len(df_extended)
        df_clean = df_extended.dropna(subset=required_features).copy()
        dropped = initial_count - len(df_clean)

        if dropped > 0:
            self.logger.warning(
                f"⚠️ [DataPurge - {label}]: 剔除 {dropped} 颗物理特征不完整(NaN)的天体，"
                f"有效样本占比: {len(df_clean)/initial_count:.1%}"
            )
        else:
            self.logger.info(f"✅ [DataPurge - {label}] 物理特征完备，共 {len(df_clean)} 颗。")
        return df_clean

    @astro_checkpoint(
        cache_table_template="cache_{cluster}_{category}_{mode}_{algo}_res",
        force_refresh=True,
    )
    def run_pgmm(self, ctx_cluster):
        """驱动核心精筛计算流水线：支持一阶段新旧轨灰度并线。

        支持双轨控制：
          - 稳定旧轨 (use_experimental=0): 100% 还原传统生产管线行为。
          - 实验新轨 (use_experimental=1/2): 启用多态策略工厂、自适应洗涤或亚结构解剖（潮汐尾）。
        Args:
            ctx_cluster (dict): 星团上下文环境，包含星团专有的 Profile 参数。

        Returns:
            str: 一阶段算法结果在数据库中的固化总表名 (self.t_master)。
        """
        # 1. 确定运行模式与特征空间
        _, required_features = self._parse_pipeline_config()

        # 2. 获取靶场天体并初始化 Master 状态表
        field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
        df_target_raw = self._get_target(field_idx, MANIFEST[field_idx], self.manifest, ctx_cluster)
        
        self.logger.info(f"📦 初始化 Master 数据库记录表: {self.t_master}")
        self.db.init_master_table(self.t_master, df_target_raw)

        # 获取实验性功能全局开关
        use_experimental = GMM_CONFIG.get("use_experimental", 0)

        if use_experimental == 0:
            # =========================================================================
            # 🔒 【稳定旧轨】：统一执行 PriorGMM 老轨行为
            # =========================================================================
            self.logger.warning("🔒 [Engine] 当前处于稳定生产模式：启用 PriorGMM 经典内核行为")
            
            # A. 提取外部静态种子
            seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
            df_seeds_raw = self._get_seeds(seed_idx, MANIFEST[seed_idx], self.manifest, ctx_cluster, required_features)

            # B. 统一特征工程
            df_target_ext = self._transform_and_bridge_features(df_target_raw, ctx_cluster, self.feature_space, required_features)
            df_seeds_ext = self._transform_and_bridge_features(df_seeds_raw, ctx_cluster, self.feature_space, required_features)

            # C. 防御性清洗
            df_target_final = self._defensive_nan_purge(df_target_ext, required_features, "TargetField")
            df_seeds_purge = self._defensive_nan_purge(df_seeds_ext, required_features, "Seeds")

            # D. 驱动 PriorGMM
            engine = PriorGMM(params=self, ctx_cluster=ctx_cluster)
            self.logger.info(f"🧠 正在执行经典 GMM 拟合与推演...")
            engine.fit(df_target_final, df_seeds_purge, required_features)
            df_res = engine.predict(df_target_final, required_features)

        elif use_experimental in (1, 2):
            # =========================================================================
            # 🚀🌌 【实验轨道并线】：启用高阶洗涤与多态亚结构模型
            # =========================================================================
            self.logger.info(f"⚡ [Engine] 已激活实验性并线轨道 (Option={use_experimental})")

            # A. 统一特征变换
            df_target_ext = self._transform_and_bridge_features(df_target_raw, ctx_cluster, self.feature_space, required_features)
            df_target_final = self._defensive_nan_purge(df_target_ext, required_features, "TargetField")

            # B. 提取粗筛种子
            seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
            df_seeds_raw = self._get_seeds(seed_idx, MANIFEST[seed_idx], self.manifest, ctx_cluster, required_features)
            df_seeds_ext = self._transform_and_bridge_features(df_seeds_raw, ctx_cluster, self.feature_space, required_features)
            df_seeds_purge = self._defensive_nan_purge(df_seeds_ext, required_features, "Seeds")
            
            # C. 🧬 [Phase 1] 调度 ClusterSeedExtractor 自适应洗涤高纯度种子星
            self.logger.info("🧬 [Phase 1] 正在通过空间密度特征洗涤自适应种子...")
            extractor = ClusterSeedExtractor(cluster_profile=ctx_cluster)
            df_seeds_final = extractor.extract_seeds(df_seeds_purge, required_features)

            if df_seeds_final is None or df_seeds_final.empty:
                raise ValueError("❌ [Extraction Error] 种子星洗涤失败：未能从背景噪声中凝聚出有效种子。")

            # 字段对齐桥接
            if "cluster_label" in df_seeds_final.columns and "seed_label" not in df_seeds_final.columns:
                df_seeds_final = df_seeds_final.rename(columns={"cluster_label": "seed_label"})

            actual_best_label = df_seeds_final["seed_label"].iloc[0]
            ctx_cluster["TARGET_CLUSTER_LABEL"] = actual_best_label
            self.logger.info(f"✅ [Phase 1] 种子洗涤完成: {len(df_seeds_final)} 颗 | 识别目标簇标签: [{actual_best_label}]")

            # 🎯 【Phase 2 / 二阶段分流】
            if use_experimental == 1:
                # 轨道 1：多态策略工厂模式
                strategy_name = ctx_cluster.get("STRATEGY", GMM_CONFIG.get("default_strategy", "bayesian")).lower()
                self.logger.info(f"🚀 [Phase 2] 激活多态策略工厂: [{strategy_name.upper()}]")

                strategy_params = ctx_cluster.get("STRATEGY_PARAMS", {}).get(strategy_name, {})
                strategy_kwargs = {**strategy_params, "cluster_algo": self.algo, "spatial_cols": ["ra", "dec"], "scale_col": "plx"}

                STRATEGY_CLASSES = {"bayesian": BayesianGmmDisambiguation, "threshold": ThresholdGmmDisambiguation, "blind": BlindGmmDisambiguation}
                if strategy_name not in STRATEGY_CLASSES:
                    raise ValueError(f"❌ 未知的策略类型 [{strategy_name}]")
                
                engine = STRATEGY_CLASSES[strategy_name](**strategy_kwargs)
                df_res = engine.fit_predict(df_target_final, df_seeds_final, required_features)
                self.logger.info(f"✅ 算法内核推演完成。---{use_experimental}---计算结果样例: {df_res.head(10)}...")

            elif use_experimental == 2:
                # 轨道 2：多态亚结构解剖与密度场裁剪
                path_mode = ctx_cluster.get("SUBSTRUCTURE_PATH_MODE", 0)
                self.logger.info(f"🌌 [Phase 2] 激活亚结构解剖轨道 (Path_Mode: {path_mode})")

                SUBSTRUCTURE_MODELS = {
                    0: IdentityComponentModeller,  # 年轻紧凑无尾基准椭球
                    1: DualComponentModeller,  # 核心 + 潮汐长尾
                    2: TripleComponentModeller,  # 核心 + 前导尾 + 后随尾非对称撕裂
                }

                if path_mode not in SUBSTRUCTURE_MODELS:
                    raise ValueError(f"❌ 未知的亚结构路径模式 [{path_mode}]")

                modeller = SUBSTRUCTURE_MODELS[path_mode](config=ctx_cluster)
                self.logger.info("⚡ 正在执行多组分物理模型拟合 (Core + Tail Support)...")
                self.logger.info(f"⚡ 当前组分数目: {path_mode+1} | 当前组分先验数据个数: { len(df_seeds_final) }")
                modeller.fit(df_seeds_final) 
                df_demarcated = modeller.predict_membership(df_target_final)
                self.logger.info(f"⚡ 模型计算概率对象的总数: {len(df_demarcated)}")

                # 处理裁剪逻辑 (Chi2 截断或拐点精筛)
                if path_mode == 0:
                    # 单组分卡方硬裁剪
                    self.logger.info("📐 执行单组分 [Identity] 5D 空间卡方严格裁剪...")
                    gmm = modeller.model
                    inv_covariance = np.linalg.inv(gmm.covariances_[0])
                    delta = df_demarcated[required_features].values - gmm.means_[0]
                    mahalanobis_sq = np.sum(np.dot(delta, inv_covariance) * delta, axis=1)
                    
                    from scipy.stats import chi2
                    quantile = ctx_cluster.get("CUTTER_CHI2_QUANTILE", 0.995)
                    chi2_threshold = chi2.ppf(quantile, df=len(required_features))
                    
                    is_member_mask = (mahalanobis_sq <= chi2_threshold)
                    df_res = df_target_final.copy()
                    df_res["prob"] = 0.0
                    df_res.loc[is_member_mask, "prob"] = 1.0
                    member_count = int(is_member_mask.sum())
                    self.logger.info(f"📐 卡方门槛(3-Sigma 准则): {chi2_threshold:.2f} | 锁定成员: {member_count}")
                else:
                    # 多组分长尾解耦裁剪
                    self.logger.info("🚧 建立多组分物理隔离墙 (7-Sigma Barrier)...")
                    gmm = modeller.model
                    inv_core_cov = np.linalg.inv(gmm.covariances_[0])
                    delta_core = df_demarcated[required_features].values - gmm.means_[0]
                    mahalanobis_core_sq = np.sum(np.dot(delta_core, inv_core_cov) * delta_core, axis=1)
                    
                    from scipy.stats import chi2
                    max_physical_bound = chi2.ppf(0.999999, df=len(required_features))
                    passed_barrier_mask = (mahalanobis_core_sq <= max_physical_bound)
                    df_candidates = df_demarcated[passed_barrier_mask].copy()
                    
                    self.logger.info(f"🛡️ 拦截完成: 过滤掉非本征杂散背景，剩余物理候选: {len(df_candidates)}")
                    
                    df_res = df_target_final.copy()
                    df_res["prob"] = 0.0
                    member_count = 0

                    if not df_candidates.empty:
                        cutter_mode = ctx_cluster.get("CUTTER_MODE", "knee").lower()
                        if cutter_mode == "chi2":
                            quantile = ctx_cluster.get("CUTTER_CHI2_QUANTILE", 0.995)
                            chi2_threshold = chi2.ppf(quantile, df=len(required_features))
                            final_member_mask = (mahalanobis_core_sq <= chi2_threshold)
                            df_res.loc[final_member_mask, "prob"] = 1.0
                            member_count = int(df_res["prob"].sum())
                        else:
                            self.logger.info("✂️ 正在调度 DensityFieldCutter 执行自适应拐点(Knee)裁剪...")
                            cutter = DensityFieldCutter(config=ctx_cluster)
                            df_final_audit = cutter.cut_field(df_candidates, "p_total_cluster", required_features)
                            true_members = df_final_audit[df_final_audit["is_member"] == True]
                            df_res.loc[true_members.index, "prob"] = 1.0
                            member_count = len(true_members)

                    # 补全多维亚结构概率字段
                    for col in ["p_total_cluster", "p_core", "p_tail", "log_p_identity"]:
                        if col in df_demarcated.columns:
                            df_res[col] = df_demarcated[col]
                
                self.logger.info(f"🎯 轨道 2 截断结算完毕！最终锁定 [{member_count}] 颗本征成员。")
        else:
            raise ValueError(f"❌ 未知的实验轨道开关值: {use_experimental}。")

        # =========================================================================
        # 🤝 【统一安全回灌通道】
        # =========================================================================
        if df_res is None or df_res.empty:
            raise ValueError("❌ 算法推断异常：结果集为空。")
        
        self.logger.info(f"✅ 算法内核推演完成。正在将概率 (prob) 同步至 Master 表...")
        
        updates = df_res[[cfg.STD_COLS["ID"], "prob"]].copy()
        self.db.tag_master_table(self.t_master, updates)

        return self.t_master

    def run(self, reconstruct_mode="file", result_mode="brief"):
        """一键驱动完整的端到端管线（对外的唯一核心接口）。"""
        self.logger.info("=" * 70)
        self.logger.info(f"🔄 [Workflow Start] Cluster: {self.target_cluster} | Mode: {self.feature_space}")
        self.logger.info("=" * 70)
        
        try:
            # [1/5] 数据同步
            self.logger.info("📦 [Phase 1/5] 正在执行数据源物理同步...")
            self.db.import_raw(target_cluster_id=self.target_cluster, force=False)

            # [2/5] 数据对齐
            ctx_cluster = cfg.CLUSTERS[self.target_cluster].copy()
            ctx_cluster["id"] = self.target_cluster
            ref_tables = [ctx_cluster["FIELD_IDX"], ctx_cluster["SEED_IDX"], self.target_category, cfg.IDX_DR2IDX, cfg.IDX_IDS_SIMBAD]
            
            self.logger.info(f"📐 [Phase 2/5] 正在执行多表标准化对齐...")
            self.data_standardize_all(ref_tables, ctx_cluster)

            # [2.5/5] 星团领域实体参数重建
            self.logger.info(f"🌌 [Phase 2.5/5] 正在重建星团领域实体模型物理状态...")
            cl = StarCluster(self.target_cluster, db_instance=self.db)
            success = cl.load_or_reconstruct_parameters(mode=reconstruct_mode)
            if not success:
                self.logger.error(f"❌ 无法初始化星团 {self.target_cluster} 的物理资产，管线终止。")
                return None
            self.logger.info(f"  ∟ 模型就绪。参考视差: {cl.plx_ref:.4f} mas | 反演距离: {1000.0 / cl.plx_ref:.1f} pc")

            # 执行核心计算子管线
            return self._run_compute_pipeline(ctx_cluster, result_mode)
            
        except Exception:
            self.logger.error("🚨 [Workflow Crash] 流水线运行期间发生严重崩溃", exc_info=True)
            raise
        finally:
            if self._owned_db:
                self.db.close()
                self.logger.info("🔒 数据库连接已安全释放。")

    def _run_compute_pipeline(self, ctx_cluster: dict, result_mode: str) -> dict | None:
        """执行 GMM → 后处理 → 交叉审计 → 深度审计 → 导出 → 报告 计算阶段。"""
        # [3/5] GMM 成员识别
        self.logger.info(f"🧠 [Phase 3/5] 启动 GMM 成员识别内核 (Algo: {self.algo})...")
        t_result = self.run_pgmm(ctx_cluster)
        self.logger.info(f"  ∟ 算法推断完成，中间状态固化于表: {t_result}")

        # [4/5] 后处理
        self.logger.info("📊 [Phase 4/5] 正在生成候选成员视图并合成统计宽表...")
        v_all = self.post_pgmm(t_result)
        if v_all.get("status") != "success":
            return None

        # [5/5] 交叉审计
        self.logger.info(f"⚖️ [Phase 5/5] 正在与参考文献 [{self.target_category}] 进行交叉审计...")
        target_aln_view = self.manifest[self.target_category]["aln_view"].format(cluster=self.target_cluster.lower())
        audit_res = self.prepare_audit_data(v_all["v_candidates"], target_aln_view)

        if audit_res.get("status") != "success":
            self.logger.warning(f"⚠️ 交叉比对不完整: {audit_res.get('message')}。尝试输出降级报告...")
            return render_final_report(self.target_cluster, self.target_category, self.feature_space, self.algo, ctx_cluster, v_all, audit_res, {}, {}, self.logger)

        # 深度审计（针对分流后的 PG Only / Ref Only）
        v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref = self._execute_deep_audits(audit_res)

        # 数据资产导出
        self._export_if_needed(audit_res, v_final_pg, v_final_ref, result_mode)

        # 渲染最终物理审计报告
        self.logger.info("🏁 工作流全部阶段执行完毕，正在生成汇总报告...")
        return render_final_report(
            self.target_cluster, self.target_category, self.feature_space, self.algo,
            ctx_cluster, v_all, audit_res, deep_stats_pg, deep_stats_ref, self.logger,
        )

    def _execute_deep_audits(self, audit_res: dict) -> tuple:
        """[深度审计调度] 分别处理算法独有候选和文献独有候选。"""
        v_audit_pg_only = audit_res.get("v_audit_pg_only")
        v_audit_ref_only = audit_res.get("v_audit_ref_only")
        x_stats = audit_res.get("stats", {})

        # PG Only 深度审计
        if not v_audit_pg_only or x_stats.get("PG Only", 0) == 0:
            self.logger.info("⏭️ 跳过 PG Only 深度审计 (无候选星)。")
            v_final_pg, deep_stats_pg = None, {}
        else:
            v_final_pg, deep_stats_pg = self._run_deep_audit(v_audit_pg_only, "pg_only")

        # Ref Only 深度审计
        if not v_audit_ref_only or x_stats.get("Ref Only", 0) == 0:
            self.logger.info("⏭️ 跳过 Ref Only 深度审计 (无候选星)。")
            v_final_ref, deep_stats_ref = None, {}
        else:
            v_final_ref, deep_stats_ref = self._run_deep_audit(v_audit_ref_only, "ref_only")

        return v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref

    def _run_deep_audit(self, v_audit_view: str, audit_type: str) -> tuple:
        """对特定视图执行深度物理校验。"""
        v_result = self.run_audit(target=v_audit_view, audit_type=audit_type)
        if not v_result:
            return None, {}
        sql = f"SELECT audit_status, count(*) FROM {v_result} WHERE audit_status IS NOT NULL GROUP BY audit_status"
        stats = dict(self.db.con.execute(sql).fetchall())
        return v_result, stats

    def _export_if_needed(self, audit_res, v_final_pg, v_final_ref, result_mode):
        """按需将关键中间件导出为 CSV。"""
        if result_mode != "detailed":
            self.logger.info("⏩ 跳过物理文件导出 (通过 CLI 参数禁用)。")
            return

        self.logger.info("💾 [Export] 正在导出详细审计资产 (CSV)...")
        export_base = cfg.TMPL.FILE_EXPORT_BASE.format(cluster=self.target_cluster, category=self.target_category, mode=self.feature_space, algo=self.algo)
        
        # 导出 Master 全量结果
        self.db.export_table(self.t_master, export_dir=cfg.RESULTS_DIR)

        # 导出分流审计报告
        if v_final_pg:
            self.db.export_table(v_final_pg, filename=cfg.TMPL.FILE_DEEP_AUDIT.format(base=export_base + "_pg_only"), format="csv", export_dir=cfg.RESULTS_DIR)
        if v_final_ref:
            self.db.export_table(v_final_ref, filename=cfg.TMPL.FILE_DEEP_AUDIT.format(base=export_base + "_ref_only"), format="csv", export_dir=cfg.RESULTS_DIR)
        
        self.logger.info(f"✅ 导出完成，结果存放于: {cfg.RESULTS_DIR}")

    @staticmethod
    def run_all_modes(target_cluster_id: str, target_category: str, algo: str, result_mode: str, reconstruct_mode: str = "file") -> None:
        """[批量入口] 循环所有特征空间模式。"""
        logger = logging.getLogger("AstroPipeline.Batch")
        valid_modes = list(cfg.GMM_CONFIG["feature_map"].keys())
        db = AstroDB(manifest=cfg.MANIFEST)

        logger.info(f"🔄 [Batch Run] 启动星团 {target_cluster_id} 的全模式分析: {valid_modes}")

        ctx_cluster = cfg.CLUSTERS[target_cluster_id].copy()
        ctx_cluster["id"] = target_cluster_id
        ref_tables = [ctx_cluster["FIELD_IDX"], ctx_cluster["SEED_IDX"], target_category, cfg.IDX_DR2IDX, cfg.IDX_IDS_SIMBAD]

        try:
            # 共享数据准备
            db.import_raw(target_cluster_id=target_cluster_id, force=False)
            wf_setup = AstroWorkflow(db, target_cluster_id, target_category, valid_modes[0], algo)
            wf_setup.data_standardize_all(ref_tables, ctx_cluster)

            cl = StarCluster(target_cluster_id, db_instance=db)
            cl.load_or_reconstruct_parameters(mode=reconstruct_mode)

            all_results = []
            for mode in valid_modes:
                wf = AstroWorkflow(db, target_cluster_id, target_category, mode, algo)
                summary = wf._run_compute_pipeline(ctx_cluster, result_mode)
                if summary:
                    all_results.append(summary)

            render_all_modes_comparison(all_results, logger)
        finally:
            db.close()
