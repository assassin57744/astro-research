# -*- coding: utf-8 -*-
"""
modules/workflow.py

天文数据处理工作流编排引擎。
集成了双轨制（稳定旧轨/实验新轨）精筛洗涤机制，支持高维贝叶斯对抗与自适应种子粗筛。
"""

import logging
import pandas as pd
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
    """

    def __init__(
        self,
        db_instance: AstroDB | None = None,
        target_cluster=None,
        target_category=None,
        mode="3d",
        algo="dbscan",
    ):
        """初始化工作流实例。

        Args:
            db_instance (AstroDB | None): 活跃原生数据库实例，传入 None 时内部自动构建。
            target_cluster (str): 目标星团唯一标识符（不区分大小写）。
            target_category (str): 目标文献比对类别。
            mode (str): 动力学特征空间维度模式 ('3d', '5d', '6d')。
            algo (str): 核心聚类识别算法（用于实验新轨多态路由）。
        """
        if db_instance is None:
            self.db = AstroDB(manifest=cfg.MANIFEST)
            self._owned_db = True
        else:
            self.db = db_instance
            self._owned_db = False

        self.target_cluster = target_cluster
        self.target_category = target_category
        self.mode = mode
        self.algo = algo  # 🎯 传递给实验多态策略的无监督算法标识
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        self.manifest = getattr(self.db, "data_manifest", {})
        self.t_master = cfg.TMPL.T_MASTER.format(
            cluster=self.target_cluster.lower(),
            category=self.target_category,
            mode=self.mode,
            algo=self.algo,
        )

    def data_standardize(self, idx_data, cfg_data, manifest, ctx=None):
        """核心标准化调度算法。

        Args:
            idx_data (str): 数据源索引。
            cfg_data (dict): 数据配置字典。
            manifest (dict): 全局清单。
            ctx (dict, optional): 包含星团几何信息的上下文。
        """
        self.logger.info(f"🚀 正在执行数据标准化, 当前源: {idx_data}")

        # 🛡️ 核心重构：确保 CAT_NAME 的解析优先级
        local_ctx = ctx.copy() if ctx else {}
        cluster_id = local_ctx.get("id")

        # 默认 CAT_NAME 使用星团的 ID_NAME 或 NAME
        local_ctx.setdefault(
            "CAT_NAME", local_ctx.get("ID_NAME", local_ctx.get("NAME"))
        )

        if cluster_id:
            adapter = getattr(cfg, "CATALOG_NAMING_ADAPTER", {})
            override_name = adapter.get(idx_data, {}).get(cluster_id)
            if override_name:
                local_ctx["CAT_NAME"] = override_name

        actions = cfg_data.get("actions", {})
        for layer in ["std", "stx", "aln"]:
            if layer in actions:
                self.logger.debug(f"  ∟ 正在执行层级动作: {layer.upper()}")
                action_func = actions[layer]
                # 将修正后的上下文传给执行层
                action_func(self.db, idx_data, cfg_data, self.manifest, local_ctx)

    def _get_seeds(self, idx_data, src, manifest, ctx=None, required_features=None):
        """从指定数据源的标准视图中提取高质量种子星 (条件筛选基于不同星团的配置)。

        Args:
            idx_data (str): 数据键。
            src (dict): 配置字典.
            required_features (list): 必须具备的物理特征列。

        Returns:
            pd.DataFrame: 种子星结果集。
        """
        v_src = src["aln_view"]
        query = f"SELECT * FROM {v_src}"
        df_raw = self.db.query(query)

        self.logger.info(f"从数据源 [{v_src}] 读取到原始种子星数据: {len(df_raw)} 颗")

        # 🚀 仅针对当前运行模式所需的特征执行 dropna
        # 这样在 2D 模式下，即便视差 (plx) 缺失，只要自行 (pm) 还在，种子星就不会被丢弃。
        if required_features:
            # 🚀 [Bugfix] 仅对当前存在的特征执行清洗。
            # 派生特征（如 l, b, U, V, W）此时尚未生成，将在 Transformer 转换后的 _defensive_nan_purge 中处理
            available_features = [f for f in required_features if f in df_raw.columns]
            df_seeds = df_raw.dropna(subset=available_features).copy()
        else:
            df_seeds = df_raw.dropna().copy()

        self.logger.info(f"从数据源 [{v_src}] 提取了 {len(df_seeds)} 颗种子星")

        # 🚀 初始化种子标签：先全部标记为 'raw_seed'
        df_tag = df_seeds[[cfg.STD_COLS["ID"]]].copy()
        df_tag["seed_type"] = "raw_seed"
        self.db.tag_master_table(self.t_master, df_tag)
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

        return df_target

    def data_standardize_all(self, ref_tables, ctx_cluster):
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
                cfg_data=MANIFEST[k],
                manifest=self.manifest,
                ctx=ctx_cluster,
            )

    def post_pgmm(self, t_main_results):
        """算法后处理流水线：生成成员子集视图并计算统计摘要。

        Args:
            t_main_results (str): 算法结果总表名称。
        """
        self.logger.info(
            f"[{self.target_cluster}] 启动 post_pipeline: 执行主表标签状态同步..."
        )
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

            # 3. 统计摘要
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

            self.logger.info("=" * 60)
            self.logger.info(f"📊 [{self.target_cluster}] Master 表后处理标签同步完成:")
            self.logger.info(f"  🔹 高置信金种子星 (is_golden): {n_golden} 颗")
            self.logger.info(f"  🔹 成员星候选总数 (is_candidate): {n_candidates} 颗")
            self.logger.info(f"  🔹 原始输入种子星 (Seeds): {n_seeds} 颗")
            self.logger.info(f"  🔹 种子集核心样本 (Core): {n_seed_core} 颗")
            self.logger.info("=" * 60)

            # 为后续步骤提供一个逻辑上的“候选者”入口视图
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

        # 🚀 [混合模式重构] 直接在 Master 表更新 x_match_tag
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

        self.logger.info(f"🚀 直接在 Master 表更新 x_match_tag 完成.")
        self.logger.info(
            f"🚀 - Master 表记录总数: {self.db.get_row_count(self.t_master)}"
        )
        self.logger.info(f"🚀 - Master 表更新记录总数: {df_x.shape[0]}")

        # 为深度审计准备输入视图（PG Only 和 Ref Only）
        v_audit_pg_only = f"v_tmp_audit_pg_only"
        v_audit_ref_only = f"v_tmp_audit_ref_only"
        self.db.register_view_from_sql(
            v_audit_pg_only, f"SELECT * FROM {self.t_master} WHERE {col_x} = 'PG Only'"
        )
        self.db.register_view_from_sql(
            v_audit_ref_only,
            f"SELECT * FROM {self.t_master} WHERE {col_x} = 'Ref Only'",
        )

        # 统计并日志
        st_sql = f"SELECT {col_x}, count(*) FROM {self.t_master} WHERE {col_x} IS NOT NULL GROUP BY {col_x}"
        stats_raw = self.db.execute(st_sql).fetchall()
        stats_cross = {row[0]: row[1] for row in stats_raw}
        self.logger.info(f"=" * 60)
        self.logger.info(f"📊 [交叉审计] 结果")
        self.logger.info(f"    Matched: {stats_cross.get('Matched', 0)}")
        self.logger.info(f"    PG Only: {stats_cross.get('PG Only', 0)}")
        self.logger.info(f"    Ref Only: {stats_cross.get('Ref Only', 0)}")
        self.logger.info(f"=" * 60)

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
        """驱动完整审计管线：涵盖数据补全、文献预热、物理校验与结果导出。

        流程包含：数据预处理(补全特征)、文献缓存预热(SIMBAD批量查询)、深度物理核实以及结果落库。

        Args:
            target (str): 待审计的目标视图名称（通常是算法发现的新源）。
            audit_type (str): 审计类型标识 (例如 'pg_only' 或 'ref_only')，用于隔离输出视图。

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
                cluster_id=self.target_cluster, db_instance=self.db, mode=self.mode
            )

            self._warm_up_literature_cache(validator, v_audit_input)

            audit_report_df = validator.run_full_audit_ex(v_audit_input)

            # 🚀 [混合模式重构] 审计结果回灌 Master 表
            self.logger.info(f"📥 正在将深度审计结果同步至 Master 表...")
            self.db.tag_master_table(self.t_master, audit_report_df)

            # 🚀 [混合模式] 审计完成后，返回 Master 表的一个逻辑视图作为“审计报告”
            # 这样既不需要创建新物理表，又能保证返回的内容仅包含审计过的星源
            v_report = f"{self.t_master}_{audit_type}_audited_report"
            # 这里的 WHERE 条件不仅判断 audit_status 不为空，还要限定对应交叉匹配类型的星源
            # 从而保证 PG Only 的报告里只有 PG Only，Ref Only 的报告里只有 Ref Only
            col_x = cfg.MASTER_COLS["X_MATCH"]
            x_match_val = (
                "PG Only"
                if audit_type == "pg_only"
                else "Ref Only" if audit_type == "ref_only" else None
            )

            if x_match_val:
                sql_filter = f"SELECT * FROM {self.t_master} WHERE audit_status IS NOT NULL AND {col_x} = '{x_match_val}'"
            else:
                sql_filter = (
                    f"SELECT * FROM {self.t_master} WHERE audit_status IS NOT NULL"
                )

            self.db.register_view_from_sql(v_report, sql_filter)
            return v_report

        except Exception as e:
            self.logger.error(
                f"❌ [Workflow] 审计流程运行期间发生严重故障: {str(e)}", exc_info=True
            )
            raise e

    def _warm_up_literature_cache(self, validator: UnifiedMemberValidator, v_source):
        """[私有方法] 提取视图中所有天体 ID 并触发文献缓存预热。

        利用 DuckDB 的 ANTI JOIN 在数据库侧直接计算差集，仅提取本地缺失的 ID，
        显著提升百万级数据下的预热效率。

        Args:
            validator (UnifiedMemberValidator): 验证器实例。
            v_source (str): 包含待验证 ID 的视图名。
        """
        cache_table = validator.cache_table

        # 确保缓存表已经存在（若完全不存在则直接跳过，后续的 CREATE TABLE 会处理）
        res = self.db.con.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = ?)",
            [cache_table.lower()],
        ).fetchone()

        if res and res[0]:
            # 2. 检测并动态追加 parent 列
            col_info = self.db.con.execute(f"PRAGMA table_info({cache_table})").df()
            if "parent" not in col_info["name"].values:
                self.logger.warning(
                    f"⚠️ [提前演进] 发现本地缓存表 `{cache_table}` 缺失 `parent` 字段，正在提前触发动态追加..."
                )
                try:
                    self.db.con.execute(
                        f"ALTER TABLE {cache_table} ADD COLUMN parent VARCHAR;"
                    )
                    self.logger.info(
                        f"✅ [提前演进成功] 已成功为表 `{cache_table}` 补齐 `parent` 数据通道。"
                    )
                except Exception as ddl_err:
                    self.logger.error(f"❌ 动态追加 parent 列失败: {str(ddl_err)}")

        # 找出在 v_source 中存在但 cache_table 中没有的 ID
        # sql_missing = f"""
        #     SELECT DISTINCT CAST(v.id AS VARCHAR) as id
        #     FROM {v_source} v
        #     ANTI JOIN {cache_table} c ON CAST(v.id AS VARCHAR) = c.gaia_dr3_id
        # """

        # 🛠️ 【核心修改】将原本的 ANTI JOIN 转换为 LEFT JOIN + WHERE 条件
        # 判定刷新条件：1. 本地缓存表压根没有这个 ID (c.gaia_dr3_id IS NULL)
        #              2. 本地虽有这个 ID，但 parent 字段未被拉取或为占位符 (c.parent IS NULL OR c.parent = 'None')
        # 找出本地缺失、或者 parent 字段为空白的记录
        sql_missing = f"""
            SELECT DISTINCT CAST(v.id AS VARCHAR) as id
            FROM {v_source} v
            LEFT JOIN {cache_table} c ON CAST(v.id AS VARCHAR) = c.gaia_dr3_id
            WHERE c.gaia_dr3_id IS NULL 
               OR c.parent IS NULL 
               OR TRIM(c.parent) = '' 
               OR LOWER(TRIM(c.parent)) = 'none'
        """

        self.logger.info(f"🔍 正在检索 [{v_source}] 中缺失的文献缓存记录...")
        df_missing = self.db.con.execute(sql_missing).df()
        ids_to_sync = df_missing["id"].tolist()

        if not ids_to_sync:
            self.logger.info("✅ 缓存对齐完成：所有源均已在本地缓存中，跳过网络同步。")
            return

        self.logger.info(
            f"🌐 正在为 {len(ids_to_sync)} 个缺失源启动增量 SIMBAD 预热..."
        )
        validator.sync_simbad_cache(ids_to_sync)

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
            t_base = MANIFEST[field_idx]["stx_view"]
            v_result = self.db.register_audit_input_view(v_target, t_base)
            self.logger.info(f"✅ 审计数据准备完成，输入视图: {v_result}")
            return v_result
        except Exception as e:
            self.logger.error(f"❌ 审计数据准备失败: {str(e)}")
            return None

    def _parse_pipeline_config(self) -> tuple[dict, list[str]]:
        """[私有方法] 原子拆解：解析 GMM 配置项与特征空间。

        Returns:
            tuple: (配置字典, 特征列名列表)。
        """
        # 采用局部副本，防止污染全局配置
        gmm_cfg = GMM_CONFIG.copy()
        current_mode = self.mode
        gmm_cfg["dim_mode"] = current_mode

        feature_map = gmm_cfg.get("feature_map", {})
        self.logger.debug(f"当前 GMM_CONFIG 中的 feature_map 配置:\n {feature_map}")

        if current_mode not in feature_map:
            raise ValueError(
                f"未知的运行模式: {current_mode}，请核实 feature_map 配置。"
            )

        required_features = feature_map[current_mode]
        self.logger.info(
            f"🌌 当前运行模式: [{current_mode}], 所需核心特征空间: {required_features}"
        )
        return gmm_cfg, required_features

    def _transform_and_bridge_features(
        self, df_raw: pd.DataFrame, ctx_cluster, mode: str, required_features: list[str]
    ) -> pd.DataFrame:
        """[私有方法] 特征转换网关：将原始坐标转换为目标物理维度特征。

        Args:
            df_raw: 原始 DataFrame。
            ctx_cluster: 星团上下文。
            mode: 运行模式 (e.g., '3d', '6d_p') rotate。
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
            raise KeyError(f"Transformer 转换矩阵列数与配置不匹配！")

        cols_upper = [col.upper() for col in required_features]
        cols_lower = [col.lower() for col in required_features]

        # 提取转换后的特征矩阵 (在此之前不得删除原始列)
        df_features = pd.DataFrame(
            X_array, columns=required_features, index=df_raw.index
        )

        existing_dup_cols = [
            col for col in df_raw.columns if col in (cols_upper + cols_lower)
        ]
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
            self.logger.error(f"❌ [数据清洗 - {label}] 数据为空，无法进行无效值过滤。")
            return pd.DataFrame()

        initial_count = len(df_extended)

        df_clean = df_extended.dropna(subset=required_features).copy()
        dropped = initial_count - len(df_clean)

        if dropped > 0:
            self.logger.warning(
                f"⚠️ [防御性过滤 - {label}]: 剔除了 {dropped} 颗特征不完整(含NaN)的天体，"
                f"剩余有效样本: {len(df_clean)}。"
            )
        else:
            self.logger.info(
                f"✅ [数据预检 - {label}] 样本特征完备，共计 {len(df_clean)} 颗星。"
            )
        return df_clean

    @astro_checkpoint(
        cache_table_template="cache_{cluster}_{category}_{mode}_{algo}_res",
        force_refresh=True,
    )
    def run_pgmm(self, ctx_cluster):
        """驱动核心精筛计算流水线：支持一阶段新旧轨灰度并线。

        运行高斯混合模型（GMM）成员星消歧判定管线。
        
        支持双轨控制：
          - 稳定旧轨 (use_experimental=False): 维持老 PriorGMM 行为，依赖外部静态种子表。
          - 实验新轨 (use_experimental=True): 启用 ClusterSeedExtractor 洗涤自适应高纯度种子，
            并统一路由至精简后的一阶段核心 2-Component 贝叶斯消歧器 (BayesianGmmDisambiguation)。

        Args:
            ctx_cluster (dict): 星团上下文环境，包含星团专有的 Profile 参数。

        Returns:
            str: 一阶段算法结果在数据库中的固化总表名 (self.t_master)。
        """
        # 1. 确定运行模式与特征空间需求
        # required_features = self._get_required_features()
        _, required_features = self._parse_pipeline_config()
        self.logger.info(f"📊 当前管线请求的特征空间: {required_features}")

        # 2. 获取并提取全量靶场数据 (Target Field)
        field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
        df_target_raw = self._get_target(
            idx_data=field_idx,
            cfg_src=MANIFEST[field_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
        )

        # 3. 初始化 Master 状态大表
        self.db.init_master_table(self.t_master, df_target_raw)

        # 获取实验性功能全局开关标志
        use_experimental = GMM_CONFIG.get("use_experimental", 0)

        if use_experimental == 0:
            # =========================================================================
            # 🔒 【稳定旧轨】：100% 还原传统生产管线行为
            # =========================================================================
            self.logger.warning("🔒 [双轨分流] 当前处于稳定生产模式：统一执行 PriorGMM 老轨行为")
            
            # A. 通过传统黑盒方法获取外部物理种子表数据
            seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
            df_seeds_raw = self._get_seeds(
                idx_data=seed_idx,
                src=MANIFEST[seed_idx],
                manifest=self.manifest,
                ctx=ctx_cluster,
                required_features=required_features,
            )

            # B. 统一进行高维特征转换（新老共用原 workflow 的私有桥接方法）
            current_mode = self.mode
            df_target_ext = self._transform_and_bridge_features(
                df_target_raw, ctx_cluster, current_mode, required_features
            )
            df_seeds_ext = self._transform_and_bridge_features(
                df_seeds_raw, ctx_cluster, current_mode, required_features
            )

            # C. 统一执行 NaN 缺损防御性清洗
            df_target_final = self._defensive_nan_purge(
                df_target_ext, required_features, label="Target_field"
            )
            df_seeds_purge = self._defensive_nan_purge(
                df_seeds_ext, required_features, label="Seeds"
            )

            # D. 驱动原始老内核进行拟合与推演
            engine = PriorGMM(params=self, ctx_cluster=ctx_cluster)
            engine.fit(df_target_final, df_seeds_purge, required_features)
            df_res = engine.predict(df_target_final, required_features)

        elif use_experimental in (1, 2):
            # =========================================================================
            # 🚀🌌 【实验轨道并线】：1 与 2 前期完全穿透共用，后期动态分流
            # =========================================================================
            self.logger.info(f"⚡ [双轨分流] 已激活实验性并线轨道。当前模式值: [use_experimental={use_experimental}]")

            # -------------------------------------------------------------------------
            # 🧬 【Phase 1 / 一阶段】：全量共用、完全穿透（等同于 C++ case 1 不加 break）
            # -------------------------------------------------------------------------
            # A. 靶场全量天区特征变换与防御性清洗
            current_mode = self.mode
            df_target_ext = self._transform_and_bridge_features(
                df_target_raw, ctx_cluster, current_mode, required_features
            )
            df_target_final = self._defensive_nan_purge(
                df_target_ext, required_features, label="Target_field"
            )

            # B. 获取并清洗用于粗筛种子的基础物理星表
            seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
            df_seeds_raw = self._get_seeds(
                idx_data=seed_idx,
                src=MANIFEST[seed_idx],
                manifest=self.manifest,
                ctx=ctx_cluster,
                required_features=required_features,
            )
            df_seeds_ext = self._transform_and_bridge_features(
                df_seeds_raw, ctx_cluster, current_mode, required_features
            )
            df_seeds_purge = self._defensive_nan_purge(
                df_seeds_ext, required_features, label="Seeds"
            )
            
            # C. 调度 ClusterSeedExtractor 从背景噪声中洗涤高纯度种子
            self.logger.info("🧬 [Phase 1] 正在驱动自适应 ClusterSeedExtractor 洗涤高纯度种子...")
            extractor = ClusterSeedExtractor(cluster_profile=ctx_cluster)
            df_seeds_final = extractor.extract_seeds(
                field_stars_df=df_seeds_purge,
                features=required_features
            )

            if df_seeds_final is None or df_seeds_final.empty:
                raise ValueError("❌ 种子星粗筛危机：ClusterSeedExtractor 未能凝聚出任何有效种子星！")
            
            self.logger.info(f"✅ [Phase 1] 沉淀完成。共洗出 {len(df_seeds_final)} 颗高纯度种子星。")

            # -------------------------------------------------------------------------
            # 🎯 【Phase 2 / 二阶段】：根据 use_experimental 的值进行后期分流结算
            # -------------------------------------------------------------------------
            if use_experimental == 1:
                # =====================================================================
                # 轨道 1：多态策略工厂模式（Bayesian / Threshold / Blind）
                # =====================================================================
                strategy_name = ctx_cluster.get(
                    "STRATEGY", GMM_CONFIG.get("default_strategy", "bayesian")
                ).lower()
                self.logger.info(f"🚀 [Phase 2] 激活多态策略工厂。当前激活策略: [{strategy_name.upper()}]")

                strategy_params = ctx_cluster.get("STRATEGY_PARAMS", {}).get(strategy_name, {})
                strategy_kwargs = {**strategy_params}
                strategy_kwargs.setdefault("cluster_algo", self.algo)
                strategy_kwargs.setdefault("spatial_cols", ["ra", "dec"])
                strategy_kwargs.setdefault("scale_col", "plx")

                STRATEGY_CLASSES = {
                    "bayesian": BayesianGmmDisambiguation,
                    "threshold": ThresholdGmmDisambiguation,
                    "blind": BlindGmmDisambiguation,
                }

                if strategy_name not in STRATEGY_CLASSES:
                    raise ValueError(f"❌ 实验程序错误: 未知的策略类型 [{strategy_name}]")
                
                engine = STRATEGY_CLASSES[strategy_name](**strategy_kwargs)
                df_res = engine.fit_predict(df_target_final, df_seeds_final, required_features)

            elif use_experimental == 2:
                # =====================================================================
                # 轨道 2：多态亚结构解剖（Identity / Dual / Triple） + 密度场自适应裁剪
                # =====================================================================
                if 'seed_label' not in df_seeds_final.columns:
                    raise KeyError("❌ 接口契约破裂：轨道 2 必须要求一阶段输出包含 'seed_label' 列！")

                path_mode = ctx_cluster.get("SUBSTRUCTURE_PATH_MODE", 0)
                self.logger.info(f"🌌 [Phase 2] 激活细分亚结构解剖流。当前路径模式: [路径 {path_mode}]")

                SUBSTRUCTURE_MODELS = {
                    0: IdentityComponentModeller,  # 年轻紧凑无尾基准椭球
                    1: DualComponentModeller,      # 核心 + 潮汐长尾
                    2: TripleComponentModeller,    # 核心 + 前导尾 + 后随尾非对称撕裂
                }

                if path_mode not in SUBSTRUCTURE_MODELS:
                    raise ValueError(f"❌ 实验轨道 2 错误: 未知的亚结构路径模式 [{path_mode}]")

                # 模型训练与多维概率投影
                modeller = SUBSTRUCTURE_MODELS[path_mode](config=ctx_cluster)
                self.logger.info("⚡ 正在将一阶段洗涤种子注入二阶段混合模型进行重构拟合...")
                modeller.fit(df_seeds_final) 
                df_demarcated = modeller.predict_membership(df_target_final)

                # 调用斩杀算子进行场裁剪
                target_score_col = "log_p_identity" if path_mode == 0 else "p_total_cluster"
                cutter = DensityFieldCutter(config=ctx_cluster)
                self.logger.info(f"✂️ 正在激活 DensityFieldCutter，基于评分列 [{target_score_col}] 计算断层边界...")
                
                df_final_audit = cutter.cut_field(
                    df_all=df_demarcated, 
                    target_score_col=target_score_col, 
                    features=required_features
                )

                # 原始行索引精准映射（保证回灌格式契约）
                df_res = df_target_final.copy()
                df_res["prob"] = 0.0  
                member_mask = (df_final_audit["is_member"] == True)
                df_res.loc[df_final_audit[member_mask].index, "prob"] = 1.0
                
                self.logger.info(f"🎯 轨道 2 自适应截断最终锁定 [{int(df_res['prob'].sum())}] 颗高本征成员星。")
        
        else:
            raise ValueError(f"❌ 未知的实验轨道开关值: {use_experimental}. 请检查 GMM_CONFIG['use_experimental'] 配置。")

        # =========================================================================
        # 🤝 【统一安全回灌通道】：严格顺应底层只有 id 与 prob 的真实物理 Facts
        # =========================================================================
        if df_res is None or df_res.empty:
            raise ValueError("❌ 算法内核异常：策略返回或缓存读取的 DataFrame 为空！")
        
        self.logger.info(f"✅ 算法内核计算完成，生成结果集共计 {len(df_res)} 颗天体。")
        self.logger.info("📥 正在将精筛洗涤概率结果同步至 Master 表...")
        
        # 严防硬编码臆造字段带来的 KeyError，新旧版本策略一律通过本通道安全同步
        updates = df_res[[cfg.STD_COLS["ID"], "prob"]].copy()
        self.db.tag_master_table(self.t_master, updates)

        return self.t_master
    
    def _________run_pgmm_bak(self, ctx_cluster):
        """驱动核心精筛计算流水线：支持实验双轨制开关。
        
        若 config.GMM_CONFIG["use_experimental"] 为 False，执行原始稳定版 PriorGMM 内核；
        若为 True，则激活重构后的多态策略工厂实验内核。

        Args:
            ctx_cluster (dict): 星团上下文环境，包含星团专有的 Profile 参数。

        Returns:
            str: 算法结果在数据库中的固化总表名 (self.t_master)。
        """
        # 1. 解析基础管线模式与特征空间
        gmm_cfg, required_features = self._parse_pipeline_config()

        use_experimental = gmm_cfg.get("use_experimental", False)
        kernel_name = "PriorGMMEx" if use_experimental else "PriorGMM"
        self.logger.info(f"🧪 [双轨制触发] 当前任务分配至内核 [{kernel_name}] 运行。")
        engine = (
            PriorGMMEx(config=gmm_cfg) if use_experimental else PriorGMM(config=gmm_cfg)
        )

        self.logger.info("📡 正在准备特征工程输入数据...")

        # 2. 获取并提取全量靶场数据 (Target Field)
        field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
        df_target_raw = self._get_target(
            idx_data=field_idx,
            cfg_src=MANIFEST[field_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
        )

        # 3. 初始化 Master 状态大表
        self.db.init_master_table(self.t_master, df_target_raw)

        # 4. 获取种子星数据
        seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
        df_seeds_raw = self._get_seeds(
            idx_data=seed_idx,
            src=MANIFEST[seed_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
            required_features=required_features,
        )
        
        # 5. 特征多维相空间高维转换 (ICRS 坐标转换为 3D/6D 等物理模式)
        current_mode = self.mode
        self.logger.info(f"⚡ 正在转换特征空间为 [{current_mode.upper()}]...")
        df_target_ext = self._transform_and_bridge_features(
            df_target_raw, ctx_cluster, current_mode, required_features
        )
        df_seeds_ext = self._transform_and_bridge_features(
            df_seeds_raw, ctx_cluster, current_mode, required_features
        )

        # 6. 特征清洗与 NaN 缺损防御性拦截
        self.logger.info("🧹 正在执行特征清洗与 NaN 防御...")
        df_target_final = self._defensive_nan_purge(
            df_target_ext, required_features, label="Target_field"
        )
        df_seeds_final = self._defensive_nan_purge(
            df_seeds_ext, required_features, label="Seeds"
        )

        self.logger.info(f"🔥 开始驱动 {kernel_name} 引擎计算...")

        # 🚀 [性能优化] 针对千万级背景样本的下采样策略
        # 只有在 config.py 中显式开启且样本量超过门限时才执行
        enable_sub = GMM_CONFIG.get("enable_subsampling", False)
        sub_limit = GMM_CONFIG.get("subsampling_limit", 500000)

        df_target_for_fit = df_target_final
        if enable_sub and len(df_target_final) > sub_limit:
            self.logger.info(
                f"🚀 [性能优化] 背景样本量巨大 ({len(df_target_final)}), "
                f"正在下采样至 {sub_limit} 用于模型拟合..."
            )
            df_target_for_fit = df_target_final.sample(n=sub_limit, random_state=42)

        params = engine.fit(df_seeds_final, df_target_for_fit)

        # 🚀 标记种子星类型 (Core/Noise)
        if hasattr(params, "df_seeds_classified"):
            updates = params.df_seeds_classified[[cfg.STD_COLS["ID"], "density_status"]]
            self.db.tag_master_table(self.t_master, updates)

        df_prob = engine.predict(df_target_final, params)
        # 🚀 直接将概率回灌 Master 表，不再创建独立的 pgmm_xxx 表
        self.db.tag_master_table(self.t_master, df_prob)

        # 🚀 [混合模式] 固化 Master 表当前状态并作为结果返回
        # 此处 master 表的数据还不完整, 不是合适导出的时机
        # self.db.save_to_warehouse(self.t_master)
        return self.t_master


        # =========================================================================
        # 🛡️ 【第二阶段双轨控制】：安全分流判定
        # =========================================================================

    def run(self, reconstruct_mode="file", result_mode="brief"):
        """一键驱动完整的端到端管线（单模式，对外的唯一核心接口）。"""
        self.logger.info(f"🔄 启动闭环工作流: {self.target_cluster} [{self.mode}]")
        try:
            # [1/5] 数据同步
            self.logger.info("📦 [1/5] 正在同步物理数据源...")
            self.db.import_raw(target_cluster_id=self.target_cluster, force=False)

            # [2/5] 数据对齐
            ctx_cluster = cfg.CLUSTERS[self.target_cluster].copy()
            ctx_cluster["id"] = self.target_cluster
            ref_tables = [
                ctx_cluster["FIELD_IDX"],
                ctx_cluster["SEED_IDX"],
                self.target_category,
                cfg.IDX_DR2IDX,
                cfg.IDX_IDS_SIMBAD,
            ]
            self.logger.info(
                f"📐 [2/5] 正在执行数据对齐 ({self.target_cluster}, 特征空间: {self.mode})..."
            )
            self.data_standardize_all(ref_tables, ctx_cluster)
            self.logger.info("✅ 数据准备阶段完成。")

            # [2.5/5] 星团领域实体参数重建
            self.logger.info(f"🌌 [2.5/5] 载入目标星团领域实体模型: {self.target_cluster}")
            cl = StarCluster(self.target_cluster, db_instance=self.db)
            success = cl.load_or_reconstruct_parameters(mode=reconstruct_mode)
            self.logger.info(
                f"✅ 星团领域模型物理状态就绪。当前反演距离: {1000.0 / cl.plx_ref:.1f} pc"
            )
            if not success:
                self.logger.error(
                    f"❌ 无法初始化星团 {self.target_cluster} 的物理资产，管线终止。"
                )
                return None

            return self._run_compute_pipeline(ctx_cluster, result_mode)
        except Exception:
            self.logger.error("❌ 流水线在运行期间发生严重崩溃", exc_info=True)
            raise
        finally:
            if self._owned_db:
                self.db.close()
                self.logger.info("🔒 数据库连接已释放。")

    def _run_compute_pipeline(self, ctx_cluster: dict, result_mode: str) -> dict | None:
        """执行 GMM → 后处理 → 交叉审计 → 深度审计 → 导出 → 报告 计算阶段。

        假定数据导入、标准化和星团参数重建已由调用方完成。
        """
        # [3/5] GMM 成员识别
        self.logger.info(
            f"🧠 [3/5] 启动 GMM 成员识别内核 (特征空间: {self.mode}, 算法: {self.algo})..."
        )
        t_result = self.run_pgmm(ctx_cluster)
        self.logger.info(f"✨ 算法推断完成，结果表: {t_result}")

        # [4/5] 后处理
        self.logger.info("📊 [4/5] 正在合成 analysis 宽表并提取候选成员视图...")
        v_all = self.post_pgmm(t_result)
        if v_all.get("status") != "success":
            self.logger.error(f"❌ 后处理流程失败: {v_all.get('message')}")
            return None
        self.logger.info("✅ 数据处理流程结束，转入交叉审计阶段。")

        # [5/5] 交叉审计
        self.logger.info(
            f"⚖️ [5/5] 执行多源文献交叉审计, 参考类别: {self.target_category}"
        )
        target_aln_view = self.manifest[self.target_category]["aln_view"].format(
            cluster=self.target_cluster.lower()
        )
        audit_res = self.prepare_audit_data(v_all["v_candidates"], target_aln_view)

        if audit_res.get("status") != "success":
            self.logger.warning(
                f"⚠️ 交叉比对审计未完全成功: {audit_res.get('message')}"
            )
            # 即使审计不完整也尝试出报告
            return render_final_report(
                self.target_cluster, self.target_category, self.mode, self.algo,
                ctx_cluster, v_all, audit_res, {}, {}, self.logger,
            )

        self.logger.info(
            f"✅ 交叉审计完成，"
            f"PG Only: {audit_res.get('v_audit_pg_only')}, "
            f"Ref Only: {audit_res.get('v_audit_ref_only')}"
        )

        # 深度审计
        v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref = (
            self._execute_deep_audits(audit_res)
        )

        # 导出
        self._export_if_needed(audit_res, v_final_pg, v_final_ref, result_mode)

        # 报告
        return render_final_report(
            self.target_cluster, self.target_category, self.mode, self.algo,
            ctx_cluster, v_all, audit_res, deep_stats_pg, deep_stats_ref, self.logger,
        )

    # =========================================================================
    # 深度审计
    # =========================================================================

    def _execute_deep_audits(self, audit_res: dict) -> tuple:
        """对交叉比对产生的 PG Only / Ref Only 候选分别执行深度审计。

        Returns:
            (v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref)
        """
        v_audit_pg_only = audit_res.get("v_audit_pg_only")
        v_audit_ref_only = audit_res.get("v_audit_ref_only")
        x_stats = audit_res.get("stats", {})

        # PG Only 深度审计
        if not v_audit_pg_only or x_stats.get("PG Only", 0) == 0:
            self.logger.warning("⚠️ 未找到算法独有候选 (PG Only)，跳过 PG Only 深度审计。")
            v_final_pg, deep_stats_pg = None, {}
        else:
            self.logger.info(f"🔍 准备 PG Only 深度审计，目标视图: {v_audit_pg_only}")
            v_final_pg, deep_stats_pg = self._run_deep_audit(
                v_audit_pg_only, "pg_only"
            )

        # Ref Only 深度审计
        if not v_audit_ref_only or x_stats.get("Ref Only", 0) == 0:
            self.logger.warning(
                "⚠️ 未找到文献独有候选 (Ref Only)，跳过 Ref Only 深度审计。"
            )
            v_final_ref, deep_stats_ref = None, {}
        else:
            self.logger.info(f"🔍 准备 Ref Only 深度审计，目标视图: {v_audit_ref_only}")
            v_final_ref, deep_stats_ref = self._run_deep_audit(
                v_audit_ref_only, "ref_only"
            )

        return v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref

    def _run_deep_audit(self, v_audit_view: str, audit_type: str) -> tuple:
        """对单个候选视图执行深度审计。

        Returns:
            (report_view_name | None, {audit_status: count})
        """
        v_result = self.run_audit(target=v_audit_view, audit_type=audit_type)
        if not v_result:
            return None, {}

        sql = (
            f"SELECT audit_status, count(*) FROM {v_result} "
            f"WHERE audit_status IS NOT NULL GROUP BY audit_status"
        )
        stats = dict(self.db.con.execute(sql).fetchall())
        return v_result, stats

    # =========================================================================
    # 结果导出
    # =========================================================================

    def _export_if_needed(self, audit_res, v_final_pg, v_final_ref, result_mode):
        """按需将管线产出物导出为 CSV/Parquet 文件。"""
        if result_mode != "detailed":
            self.logger.info("⏩ 跳过物理文件导出 (通过 CLI 参数禁用)。")
            return

        self.logger.info("💾 [Export] 正在执行耗时的数据资产导出任务...")

        export_base = cfg.TMPL.FILE_EXPORT_BASE.format(
            cluster=self.target_cluster,
            category=self.target_category,
            mode=self.mode,
            algo=self.algo,
        )

        self.db.export_table(self.t_master, export_dir=cfg.RESULTS_DIR)

        if v_final_pg:
            self.db.export_table(
                v_final_pg,
                filename=cfg.TMPL.FILE_DEEP_AUDIT.format(
                    base=export_base + "_pg_only"
                ),
                format="csv",
                export_dir=cfg.RESULTS_DIR,
            )

        if v_final_ref:
            self.db.export_table(
                v_final_ref,
                filename=cfg.TMPL.FILE_DEEP_AUDIT.format(
                    base=export_base + "_ref_only"
                ),
                format="csv",
                export_dir=cfg.RESULTS_DIR,
            )

        self.logger.info("✅ 结果导出完成。")

    # =========================================================================
    # 全模式批量运行
    # =========================================================================

    @staticmethod
    def run_all_modes(
        target_cluster_id: str,
        target_category: str,
        algo: str,
        result_mode: str,
        reconstruct_mode: str = "file",
    ) -> None:
        """循环所有特征空间模式，共享数据准备，产出汇总对比报告。"""
        logger = logging.getLogger("AstroPipeline")
        valid_modes = list(cfg.GMM_CONFIG["feature_map"].keys())
        db = AstroDB(manifest=cfg.MANIFEST)

        # 星团上下文（所有模式共享）
        ctx_cluster = cfg.CLUSTERS[target_cluster_id].copy()
        ctx_cluster["id"] = target_cluster_id
        ref_tables = [
            ctx_cluster["FIELD_IDX"],
            ctx_cluster["SEED_IDX"],
            target_category,
            cfg.IDX_DR2IDX,
            cfg.IDX_IDS_SIMBAD,
        ]

        try:
            # --- 一次性数据准备（所有模式共享）---
            logger.info("📦 正在同步物理数据源...")
            db.import_raw(target_cluster_id=target_cluster_id, force=False)

            wf_setup = AstroWorkflow(
                db, target_cluster_id, target_category, valid_modes[0], algo
            )
            wf_setup.data_standardize_all(ref_tables, ctx_cluster)
            logger.info("✅ 数据准备阶段完成（全模式共享）。")

            # 星团物理参数重建（所有模式共享）
            logger.info(f"🌌 载入目标星团领域实体模型: {target_cluster_id}")
            cl = StarCluster(target_cluster_id, db_instance=db)
            cl.load_or_reconstruct_parameters(mode=reconstruct_mode)
            logger.info(
                f"✅ 星团领域模型物理状态就绪。当前反演距离: {1000.0 / cl.plx_ref:.1f} pc"
            )

            # --- 逐模式执行计算管线 ---
            all_results = []
            for mode in valid_modes:
                wf = AstroWorkflow(db, target_cluster_id, target_category, mode, algo)
                summary = wf._run_compute_pipeline(ctx_cluster, result_mode)
                if summary:
                    all_results.append(summary)

            render_all_modes_comparison(all_results, logger)
        finally:
            db.close()
            logger.info("🔒 数据库连接已释放。")
