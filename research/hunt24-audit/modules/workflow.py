import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils.decorators import astro_checkpoint

from modules.db import AstroDB
from modules.pg_core import PriorGMM
from modules.validator import UnifiedMemberValidator
from modules.transformer import AstroTransformer
from modules.reporter import render_final_report, render_all_modes_comparison
from modules.cluster import StarCluster

import config as cfg


# =============================================================================
# 📦 运行上下文数据类（单次运行的不可变调度上下文）
# =============================================================================

@dataclass
class RunContext:
    """单次运行的不可变调度上下文。

    只包含管线调度参数，不包含子模块细节。
    扩展方式：通过 algo_params / audit_params / seed_params 字典自由注入新参数，
    无需修改本数据类定义。新增参数类别时仅需增加一个命名空间 dict 字段。
    """
    cluster_id: str
    category: str             # "hunt", "cg20", etc.
    feature_space: str        # "2d", "5d", "6d_p", etc.
    algorithm: str            # "dbscan", "hdbscan"
    result_mode: str          # "brief" | "detailed"
    param_source: str         # "file" | "db"

    algo_params: dict = field(default_factory=dict)
    audit_params: dict = field(default_factory=dict)
    seed_params: dict = field(default_factory=dict)

    star_cluster: Any = None
    gmm_config: dict = field(default_factory=dict)
    required_features: list = field(default_factory=list)
    master_table: str = ""
    seed_stats: dict = field(default_factory=dict)  # raw_count / clean_count / refined_count


# =============================================================================
# AstroWorkflow 主类
# =============================================================================

class AstroWorkflow:
    """天文数据处理工作流编排引擎。

    方法层次：
      - 一级 PUBLIC:  run()
      - 二级 阶段调度: _execute_single_pipeline() / _prepare_shared_data() /
                     _finalize_context() / _compute_members() / _post_process() /
                     _audit_phase() / _export_phase() / _report_phase()
      - 三级 功能单元: _standardize_ref_tables() / _load_and_transform_field() /
                     _load_and_transform_seeds() / _run_stable_pipeline() /
                     _run_experimental_pipeline() / _cross_match_with_literature() /
                     _run_audit_pipeline() / 等
    """

    def __init__(self, db_instance: AstroDB | None = None):
        """初始化工作流实例。

        Args:
            db_instance (AstroDB | None): 活跃的 AstroDB 数据库对象。
        """
        if db_instance is None:
            self.db = AstroDB(manifest=cfg.MANIFEST)
            self._owned_db = True
        else:
            self.db = db_instance
            self._owned_db = False

        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        self.manifest = getattr(self.db, "data_manifest", {})

    # =========================================================================
    # 🟢 一级：PUBLIC API
    # =========================================================================


    def run(
        self,
        clusters: list[str],
        categories: list[str],
        feature_spaces: list[str],
        algorithms: list[str],
        result_mode: str = "brief",
        param_source: str = "file",
        algo_params_override: dict | None = None,
        audit_params_override: dict | None = None,
        seed_params_override: dict | None = None,
    ) -> list[dict]:
        """🚀 批量多模式运行入口。

        对 clusters × categories × feature_spaces × algorithms 笛卡尔积
        of the each combination independently execute the complete pipeline, returns a summary results list.
        同一个 cluster + category 的数据准备只执行一次。
        """
        results = []
        total = len(clusters) * len(categories) * len(feature_spaces) * len(algorithms)
        count = 0

        for cluster_id in clusters:
            for category in categories:
                # ── 共享数据准备 ──
                ctx_base = RunContext(
                    cluster_id=cluster_id,
                    category=category,
                    feature_space="",
                    algorithm="",
                    result_mode=result_mode,
                    param_source=param_source,
                )
                self.logger.info(f"📦 [Batch] 准备共享数据: {cluster_id}/{category}")
                self._prepare_shared_data(ctx_base)

                for fs in feature_spaces:
                    for algo in algorithms:
                        count += 1
                        self.logger.info(
                            f"⚡ [Batch] [{count}/{total}] "
                            f"{cluster_id} / {category} / {fs} / {algo}"
                        )
                        ctx = RunContext(
                            cluster_id=cluster_id,
                            category=category,
                            feature_space=fs,
                            algorithm=algo,
                            result_mode=result_mode,
                            param_source=param_source,
                            algo_params=(algo_params_override or {}).copy(),
                            audit_params=(audit_params_override or {}).copy(),
                            seed_params=(seed_params_override or {}).copy(),
                        )
                        ctx.star_cluster = ctx_base.star_cluster

                        try:
                            summary = self._execute_single_pipeline(
                                ctx, skip_data_prep=True
                            )
                            if summary:
                                results.append(summary)
                        except Exception:
                            self.logger.error(
                                f"❌ [Batch] [{cluster_id}/{fs}/{algo}] 执行失败",
                                exc_info=True,
                            )

        self._render_batch_summary(results)
        return results


    # =========================================================================
    # 🟡 二级：阶段调度器
    # =========================================================================

    def _execute_single_pipeline(
        self, ctx: RunContext, skip_data_prep: bool = False
    ) -> dict | None:
        """[核心调度器] 串联完整管线 6 个阶段。"""
        # Phase 1: 数据准备
        if not skip_data_prep:
            self._prepare_shared_data(ctx)
        self._finalize_context(ctx)

        # Phase 2: GMM 成员识别
        self.logger.info(
            f"🧠 [Phase 2] GMM 成员识别: {ctx.cluster_id} [{ctx.feature_space}]"
        )
        result_table = self._compute_members(ctx)
        if not result_table:
            self.logger.error("❌ [Phase 2] 成员识别失败")
            return None

        # Phase 3: 后处理
        self.logger.info("📊 [Phase 3] 后处理...")
        post_result = self._post_process(ctx, result_table)
        if post_result.get("status") != "success":
            self.logger.error(f"❌ [Phase 3] 后处理失败: {post_result.get('message')}")
            return None

        # Phase 4: 审计
        self.logger.info(f"⚖️ [Phase 4] 交叉审计, 参考类别: {ctx.category}")
        audit_result = self._audit_phase(ctx, post_result)

        # Phase 5: 导出
        self._export_phase(ctx, audit_result)

        # Phase 6: 报告
        return self._report_phase(ctx, post_result, audit_result)

    def _prepare_shared_data(self, ctx: RunContext):
        """执行可跨特征空间复用的数据准备：数据导入 + 星团实体 + 标准化。"""
        self.logger.info(f"📦 [Phase 1] 数据准备: {ctx.cluster_id}")

        self.db.import_raw(target_cluster=ctx.cluster_id, force=False)

        ctx.star_cluster = StarCluster(
            ctx.cluster_id, db_instance=self.db, param_source=ctx.param_source
        )

        # 🚀 必须在 load_or_reconstruct_parameters 之前执行标准化，
        # 因为参数重建（config_manager）依赖 aln 视图（如 aln_hunt_m44）已存在。
        self._standardize_ref_tables(ctx)

        success = ctx.star_cluster.load_or_reconstruct_parameters(
            param_source=ctx.param_source
        )
        if not success:
            raise RuntimeError(f"无法初始化星团 {ctx.cluster_id} 的物理资产")
        self.logger.info(
            f"✅ 星团领域模型就绪。"
            f"距离: {1000.0 / ctx.star_cluster.get_param('PLX_REF'):.1f} pc"
        )

        self.logger.info("✅ [Phase 1] 数据准备阶段完成。")

    def _finalize_context(self, ctx: RunContext):
        """填充依赖于特征空间/算法的上下文属性。"""
        gmm_cfg = cfg.GMM_CONFIG.copy()
        gmm_cfg["dim_mode"] = ctx.feature_space

        fmap = gmm_cfg.get("feature_map", {})
        if ctx.feature_space not in fmap:
            raise ValueError(f"未知的特征空间: {ctx.feature_space}")

        ctx.gmm_config = gmm_cfg
        ctx.required_features = fmap[ctx.feature_space]
        ctx.master_table = cfg.TMPL.T_MASTER.format(
            cluster=ctx.cluster_id.lower(),
            category=ctx.category,
            feature_space=ctx.feature_space,
            algo=ctx.algorithm,
        )

    # =========================================================================
    # 🔵 三级：功能单元
    # =========================================================================

    # ── 数据标准化 ──

    def _standardize_ref_tables(self, ctx: RunContext):
        """标准化所有参考星表。"""
        cl = ctx.star_cluster
        ref_tables = [
            cl.get_param("FIELD_IDX"),
            cl.get_param("SEED_IDX"),
            ctx.category,
            cfg.IDX_DR2IDX,
            cfg.IDX_IDS_SIMBAD,
        ]
        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id

        for k in ref_tables:
            self._data_standardize(
                idx_data=k,
                cfg_data=cfg.MANIFEST[k],
                manifest=self.manifest,
                ctx=cluster_cfg,
            )

    def _data_standardize(self, idx_data, cfg_data, manifest, ctx=None):
        """核心标准化调度算法（保留原有逻辑）。"""
        self.logger.info(f"🚀 [Process] 正在执行数据标准化, 当前源: {idx_data}")

        local_ctx = ctx.copy() if ctx else {}
        cluster_id = local_ctx.get("id")
        local_ctx.setdefault("CAT_NAME", local_ctx.get("ID_NAME", local_ctx.get("NAME")))

        if cluster_id:
            adapter = getattr(cfg, "CATALOG_NAMING_ADAPTER", {})
            override_name = adapter.get(idx_data, {}).get(cluster_id)
            if override_name:
                local_ctx["CAT_NAME"] = override_name

        actions = cfg_data.get("actions", {})
        for layer in ["std", "stx", "aln"]:
            if layer in actions:
                self.logger.debug(f"  ∟ [Process] 正在执行层级动作: {layer.upper()}")
                action_func = actions[layer]
                action_func(self.db, idx_data, cfg_data, self.manifest, local_ctx)

    # ── 特征工程 ──

    def _load_and_transform_field(self, ctx: RunContext) -> pd.DataFrame:
        """加载靶场数据 → 特征转换 → NaN清洗。"""
        field_idx = ctx.star_cluster.get_param("FIELD_IDX")
        cfg_source = self.manifest[field_idx]
        v_aln = cfg_source["aln_view"]

        df_raw = self.db.query(f"SELECT * FROM {v_aln}")
        self.logger.info(f"📋 [Process] 从视图 [{v_aln}] 读取目标天区数据: {len(df_raw)} 颗")

        df_ext = self._transform_and_bridge_features(
            df_raw, ctx.feature_space, ctx.required_features, ctx.star_cluster
        )
        return self._defensive_nan_purge(df_ext, ctx.required_features, label="Target_field")

    def _load_and_transform_seeds(self, ctx: RunContext) -> pd.DataFrame:
        """加载种子数据 → 特征转换 → NaN清洗。"""
        seed_idx = ctx.star_cluster.get_param("SEED_IDX")
        src = self.manifest[seed_idx]
        v_src = src["aln_view"]

        df_raw = self.db.query(f"SELECT * FROM {v_src}")
        ctx.seed_stats["raw_count"] = len(df_raw)
        self.logger.info(f"📋 [Process] 从视图 [{v_src}] 读取原始种子星: {len(df_raw)} 颗")

        available_features = [f for f in ctx.required_features if f in df_raw.columns]
        df_seeds = (
            df_raw.dropna(subset=available_features).copy()
            if available_features
            else df_raw.dropna().copy()
        )

        df_tag = df_seeds[[cfg.STD_COLS["ID"]]].copy()
        df_tag["seed_type"] = "raw_seed"
        self.db.tag_master_table(ctx.master_table, df_tag)

        self.logger.info(f"✅ [Process] 种子星提取完成，有效样本: {len(df_seeds)} 颗")

        df_ext = self._transform_and_bridge_features(
            df_seeds, ctx.feature_space, ctx.required_features, ctx.star_cluster
        )
        df_clean = self._defensive_nan_purge(df_ext, ctx.required_features, label="Seeds")
        ctx.seed_stats["clean_count"] = len(df_clean)
        return df_clean

    def _transform_and_bridge_features(
        self, df_raw: pd.DataFrame, feature_space: str, required_features: list[str],
        star_cluster: Any = None,
    ) -> pd.DataFrame:
        """特征转换网关。"""
        if df_raw is None:
            self.logger.error("❌ [Compute] 输入的原始 DataFrame 为 None！")
            return None

        cl = star_cluster
        cluster_rv = cl.get_param("RV_REF", None)
        c_ra = cl.get_param("CENTER_RA", None)
        c_dec = cl.get_param("CENTER_DEC", None)
        cluster_center = (
            (c_ra, c_dec) if (c_ra is not None and c_dec is not None) else None
        )

        transformer = AstroTransformer(
            cluster_rv=cluster_rv, cluster_center_icrs=cluster_center
        )
        X_array = transformer.fit_transform(df_raw, feature_space=feature_space)

        if X_array.shape[1] != len(required_features):
            raise KeyError(f"Transformer 转换矩阵列数与配置不匹配！")

        cols_upper = [col.upper() for col in required_features]
        cols_lower = [col.lower() for col in required_features]

        df_features = pd.DataFrame(X_array, columns=required_features, index=df_raw.index)

        dup_cols = [col for col in df_raw.columns if col in (cols_upper + cols_lower)]
        if dup_cols:
            self.logger.info(f"🔄 [Compute] 模式 [{feature_space}] 移除重复列: {dup_cols}")
            df_raw = df_raw.drop(columns=dup_cols)
        return pd.concat([df_raw, df_features], axis=1)

    def _defensive_nan_purge(
        self, df_extended: pd.DataFrame, required_features: list[str], label: str
    ) -> pd.DataFrame:
        """特征清洗。"""
        if df_extended is None:
            self.logger.error(f"❌ [Compute] [{label}] 数据为空！")
            return pd.DataFrame()

        initial_count = len(df_extended)
        df_clean = df_extended.dropna(subset=required_features).copy()
        dropped = initial_count - len(df_clean)

        if dropped > 0:
            self.logger.warning(
                f"⚠️ [Compute] [防御性过滤 - {label}]: 剔除 {dropped} 颗, "
                f"剩余 {len(df_clean)}。"
            )
        else:
            self.logger.info(f"✅ [Compute] [数据预检 - {label}] 共计 {len(df_clean)} 颗。")
        return df_clean

    # ── GMM 成员识别 ──

    @astro_checkpoint(
        cache_table_template="cache_{cluster}_{category}_{mode}_{algo}_res",
        force_refresh=True,
    )
    def _compute_members(self, ctx: RunContext) -> str | None:
        """统一的成员识别调度器。"""
        self.logger.info(f"📊 [Compute] 管线请求的特征空间: {ctx.required_features}")

        df_target_final = self._load_and_transform_field(ctx)

        # 🚀 必须先建表，再加载种子（_load_and_transform_seeds 内部会调用 tag_master_table 回灌标签）
        self.db.init_master_table(ctx.master_table, df_target_final)
        df_seeds_final = self._load_and_transform_seeds(ctx)

        use_experimental = ctx.gmm_config.get("use_experimental", False)
        if not use_experimental:
            df_res = self._run_stable_pipeline(ctx, df_target_final, df_seeds_final)
        else:
            df_res = self._run_experimental_pipeline(ctx, df_target_final, df_seeds_final)

        if df_res is None or df_res.empty:
            raise ValueError("❌ [Compute] 算法内核异常：结果 DataFrame 为空！")

        self.logger.info(f"✅ [Compute] 算法内核计算完成，结果集共计 {len(df_res)} 颗天体。")
        self.logger.info("📥 [Compute] 正在将概率结果同步至 Master 表...")

        updates = df_res[[cfg.STD_COLS["ID"], "prob"]].copy()
        self.db.tag_master_table(ctx.master_table, updates)

        return ctx.master_table

    def _run_stable_pipeline(
        self, ctx: RunContext, df_target_final: pd.DataFrame, df_seeds_final: pd.DataFrame
    ) -> pd.DataFrame:
        """稳定生产轨：使用传统 PriorGMM。"""
        self.logger.warning("🔒 [Compute] 稳定生产模式：执行 PriorGMM 老轨行为")

        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id].copy()
        cluster_cfg["id"] = ctx.cluster_id

        engine = PriorGMM(params=self, ctx_cluster=cluster_cfg)
        engine.fit(df_target_final, df_seeds_final, ctx.required_features)
        return engine.predict(df_target_final, ctx.required_features)

    def _run_experimental_pipeline(
        self, ctx: RunContext, df_target_final: pd.DataFrame, df_seeds_final: pd.DataFrame
    ) -> pd.DataFrame:
        """实验新轨：ClusterSeedExtractor + 多态策略工厂。"""
        strategy_name = ctx.algo_params.get("strategy", "bayesian")
        self.logger.info(f"🚀 [Compute] 实验性多态管线。策略: [{strategy_name.upper()}]")

        self.logger.info("🧬 [Compute] 正在调度 ClusterSeedExtractor...")
        from modules.seed_extractor import ClusterSeedExtractor

        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id
        extractor = ClusterSeedExtractor(cluster_profile=cluster_cfg)
        df_seeds_refined = extractor.extract_seeds(
            field_stars_df=df_seeds_final,
            features=ctx.required_features,
        )

        if df_seeds_refined is None or df_seeds_refined.empty:
            raise ValueError("❌ [Compute] ClusterSeedExtractor 未能凝聚出有效种子星！")
        ctx.seed_stats["refined_count"] = len(df_seeds_refined)
        self.logger.info(f"✅ [Compute] 种子星粗筛成功！共 {len(df_seeds_refined)} 颗。")

        strategy_params = (
            cfg.CLUSTERS[ctx.cluster_id.upper()]
            .get("STRATEGY_PARAMS", {})
            .get(strategy_name, {})
        )
        strategy_kwargs = {**strategy_params}
        strategy_kwargs.setdefault("spatial_cols", ["ra", "dec"])
        strategy_kwargs.setdefault("scale_col", "plx")

        for key in ("eps", "min_samples", "sigma_cutoff"):
            if key in ctx.algo_params:
                strategy_kwargs[key] = ctx.algo_params[key]

        from modules.astro_membership.disambiguation.bayesian import BayesianGmmDisambiguation
        from modules.astro_membership.disambiguation.threshold import ThresholdGmmDisambiguation
        from modules.astro_membership.disambiguation.blind import BlindGmmDisambiguation

        STRATEGY_CLASSES = {
            "bayesian": BayesianGmmDisambiguation,
            "threshold": ThresholdGmmDisambiguation,
            "blind": BlindGmmDisambiguation,
        }

        if strategy_name not in STRATEGY_CLASSES:
            raise ValueError(f"未知的策略类型 [{strategy_name}]")

        engine_class = STRATEGY_CLASSES[strategy_name]
        self.logger.info(f"✅ [Compute] 已路由至策略类 [{engine_class.__name__}]")
        engine = engine_class(**strategy_kwargs)
        return engine.fit_predict(df_target_final, df_seeds_refined, ctx.required_features)

    # ── 后处理 ──

    def _post_process(self, ctx: RunContext, t_main_results: str) -> dict:
        """算法后处理流水线。"""
        self.logger.info(f"📊 [Process] [{ctx.cluster_id}] 启动后处理...")
        try:
            self.db.execute(
                f"ALTER TABLE {ctx.master_table} "
                f"ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE"
            )
            self.db.execute(
                f"ALTER TABLE {ctx.master_table} "
                f"ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE"
            )

            condi_golden = f"{cfg.STD_COLS['PROB']} >= {cfg.GOLDEN_SAMPLE_THRESHOLD}"
            condi_candidates = f"{cfg.STD_COLS['PROB']} > {cfg.MEMBER_SAMPLE_THRESHOLD}"

            self.db.execute(
                f"UPDATE {ctx.master_table} SET is_golden = TRUE WHERE {condi_golden}"
            )
            self.db.execute(
                f"UPDATE {ctx.master_table} SET is_candidate = TRUE WHERE {condi_candidates}"
            )

            stats_sql = f"""
                SELECT 
                    count(*) FILTER (WHERE is_golden = TRUE) AS n_golden,
                    count(*) FILTER (WHERE is_candidate = TRUE) AS n_candidates,
                    count(*) FILTER (WHERE seed_type = 'raw_seed') AS n_seeds,
                    count(*) FILTER (WHERE density_status = 'core') AS n_seed_core,
                    count(*) FILTER (WHERE density_status = 'noise') AS n_seed_noise
                FROM {ctx.master_table}
            """
            stats = self.db.execute(stats_sql).fetchone()
            n_golden, n_candidates, n_seeds, n_seed_core, n_seed_noise = stats

            self.logger.info("=" * 60)
            self.logger.info(f"📊 [Process] [{ctx.cluster_id}] 后处理标签同步完成:")
            self.logger.info(f"  🔹 高置信金种子星 (is_golden): {n_golden} 颗")
            self.logger.info(f"  🔹 成员星候选总数 (is_candidate): {n_candidates} 颗")
            self.logger.info(f"  🔹 原始输入种子星 (Seeds): {n_seeds} 颗")
            self.logger.info(f"  🔹 种子集核心样本 (Core): {n_seed_core} 颗")
            self.logger.info("=" * 60)

            v_candidates = f"v_candidates_{ctx.cluster_id.lower()}"
            self.db.register_view_from_sql(
                v_candidates,
                f"SELECT * FROM {ctx.master_table} WHERE is_candidate = TRUE",
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
                    **ctx.seed_stats,
                },
            }
        except Exception as e:
            self.logger.error(f"❌ [Process] 后处理失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ── 审计 ──

    def _audit_phase(self, ctx: RunContext, post_result: dict) -> dict:
        """审计阶段：交叉比对 + 深度审计。"""
        target_aln_view = self.manifest[ctx.category]["aln_view"].format(
            cluster=ctx.cluster_id.lower()
        )
        audit_res = self._cross_match_with_literature(
            ctx, target_aln_view
        )

        if audit_res.get("status") != "success":
            self.logger.warning(f"⚠️ [Audit] 交叉比对未完全成功: {audit_res.get('message')}")
            return audit_res

        self.logger.info("✅ [Audit] 交叉审计比对完成。")
        deep_stats_pg, deep_stats_ref = self._execute_deep_audits(ctx, audit_res)
        audit_res["deep_stats_pg"] = deep_stats_pg
        audit_res["deep_stats_ref"] = deep_stats_ref
        return audit_res

    def _cross_match_with_literature(
        self, ctx: RunContext, v_target: str
    ) -> dict:
        """交叉比对（保留原有逻辑）。"""
        if not self._verify_audit_target_exists(v_target):
            self.logger.warning(f"⚠️ [Audit] 审计目标表 '{v_target}' 不存在。")
            return {"status": "warning", "message": f"审计目标表 '{v_target}' 不存在"}

        self.logger.info(f"⚡ [Audit] 开始交叉比对: {v_target}")

        col_x = cfg.MASTER_COLS["X_MATCH"]
        sql_cross = f"""
            SELECT 
                COALESCE(m.id, h.id) as id,
                CASE 
                    WHEN m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD}
                         AND h.id IS NOT NULL THEN 'Matched'
                    WHEN m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD}
                         AND h.id IS NULL     THEN 'PG Only'
                    WHEN (m.id IS NULL OR m.prob <= {cfg.MEMBER_SAMPLE_THRESHOLD}
                         OR m.prob IS NULL)
                         AND h.id IS NOT NULL THEN 'Ref Only'
                END as {col_x}
            FROM {ctx.master_table} m
            FULL OUTER JOIN {v_target} h ON m.id = h.id
            WHERE m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} OR h.id IS NOT NULL
        """
        df_x = self.db.query(sql_cross)
        self.db.tag_master_table(ctx.master_table, df_x)

        self.logger.info(f"✅ [Audit] Master 表 x_match_tag 更新完成。")

        v_audit_pg_only = f"v_tmp_audit_pg_only_{ctx.master_table}"
        v_audit_ref_only = f"v_tmp_audit_ref_only_{ctx.master_table}"
        self.db.register_view_from_sql(
            v_audit_pg_only,
            f"SELECT * FROM {ctx.master_table} WHERE {col_x} = 'PG Only'",
        )
        self.db.register_view_from_sql(
            v_audit_ref_only,
            f"SELECT * FROM {ctx.master_table} WHERE {col_x} = 'Ref Only'",
        )

        st_sql = (
            f"SELECT {col_x}, count(*) FROM {ctx.master_table} "
            f"WHERE {col_x} IS NOT NULL GROUP BY {col_x}"
        )
        stats_raw = self.db.execute(st_sql).fetchall()
        stats_cross = {row[0]: row[1] for row in stats_raw}

        self.logger.info("=" * 60)
        self.logger.info("📊 [Audit] [交叉比对结果统计]")
        self.logger.info(f"    Matched: {stats_cross.get('Matched', 0)}")
        self.logger.info(f"    PG Only: {stats_cross.get('PG Only', 0)}")
        self.logger.info(f"    Ref Only: {stats_cross.get('Ref Only', 0)}")
        self.logger.info("=" * 60)

        return {
            "status": "success",
            "v_audit_pg_only": v_audit_pg_only,
            "v_audit_ref_only": v_audit_ref_only,
            "stats": stats_cross,
        }

    def _verify_audit_target_exists(self, v_target: str) -> bool:
        """检查审计目标表是否存在。"""
        if not self.db:
            return False
        sql = f"SELECT 1 FROM information_schema.tables WHERE table_name = '{v_target}'"
        return self.db.con.execute(sql).fetchone() is not None

    def _execute_deep_audits(self, ctx: RunContext, audit_res: dict) -> tuple:
        """对 PG Only / Ref Only 执行深度审计。"""
        v_audit_pg = audit_res.get("v_audit_pg_only")
        v_audit_ref = audit_res.get("v_audit_ref_only")
        x_stats = audit_res.get("stats", {})

        deep_stats_pg = {}
        if v_audit_pg and x_stats.get("PG Only", 0) > 0:
            _, deep_stats_pg = self._run_deep_audit(ctx, v_audit_pg, "pg_only")
        else:
            self.logger.warning("⚠️ [Audit] 无 PG Only 候选，跳过深度审计。")

        deep_stats_ref = {}
        if v_audit_ref and x_stats.get("Ref Only", 0) > 0:
            _, deep_stats_ref = self._run_deep_audit(ctx, v_audit_ref, "ref_only")
        else:
            self.logger.warning("⚠️ [Audit] 无 Ref Only 候选，跳过深度审计。")

        return deep_stats_pg, deep_stats_ref

    def _run_deep_audit(self, ctx: RunContext, v_audit_view: str, audit_type: str) -> tuple:
        """对单个候选视图执行深度审计。"""
        v_result = self._run_audit_pipeline(ctx, target=v_audit_view, audit_type=audit_type)
        if not v_result:
            return None, {}

        sql = (
            f"SELECT audit_status, count(*) FROM {v_result} "
            f"WHERE audit_status IS NOT NULL GROUP BY audit_status"
        )
        stats = dict(self.db.con.execute(sql).fetchall())
        return v_result, stats

    def _run_audit_pipeline(self, ctx: RunContext, target: str, audit_type: str = "default") -> str | None:
        """驱动完整审计管线（原 run_audit 重命名）。"""
        self.logger.info(f"🔍 🎬 [Audit] 开始对 {target} 进行身份审计...")

        try:
            v_audit_input = self._pre_audit(ctx, target)
            if not v_audit_input:
                self.logger.error("❌ [Audit] 审计预处理失败")
                return None

            validator = UnifiedMemberValidator(
                cluster=ctx.star_cluster,
                db_instance=self.db,
                feature_space=ctx.feature_space,
            )

            self._warm_up_literature_cache(validator, v_audit_input)

            audit_report_df = validator.run(v_audit_input)

            self.logger.info("📥 [Audit] 正在将深度审计结果同步至 Master 表...")
            self.db.tag_master_table(ctx.master_table, audit_report_df)

            v_report = f"{ctx.master_table}_{audit_type}_audited_report"
            col_x = cfg.MASTER_COLS["X_MATCH"]
            x_match_val = (
                "PG Only"
                if audit_type == "pg_only"
                else "Ref Only" if audit_type == "ref_only" else None
            )

            if x_match_val:
                sql_filter = (
                    f"SELECT * FROM {ctx.master_table} "
                    f"WHERE audit_status IS NOT NULL AND {col_x} = '{x_match_val}'"
                )
            else:
                sql_filter = f"SELECT * FROM {ctx.master_table} WHERE audit_status IS NOT NULL"

            self.db.register_view_from_sql(v_report, sql_filter)
            return v_report

        except Exception as e:
            self.logger.error(f"❌ [Audit] 审计流程故障: {str(e)}", exc_info=True)
            return None

    def _pre_audit(self, ctx: RunContext, v_target: str) -> str | None:
        """审计前准备：补全物理参数。"""
        self.logger.info("🔧 [Audit] 正在准备审计数据视图...")
        try:
            field_idx = cfg.CLUSTERS[ctx.cluster_id]["FIELD_IDX"]
            t_base = cfg.MANIFEST[field_idx]["stx_view"]
            v_result = self.db.register_audit_input_view(v_target, t_base)
            self.logger.info(f"✅ [Audit] 审计数据准备完成，输入视图: {v_result}")
            return v_result
        except Exception as e:
            self.logger.error(f"❌ [Audit] 审计数据准备失败: {str(e)}")
            return None

    def _warm_up_literature_cache(self, validator: UnifiedMemberValidator, v_source: str):
        """SIMBAD 文献缓存预热。

        若本地缓存表尚不存在（首次运行），将所有候选 ID 一次性送入
        sync_simbad_cache（内部负责建表 + 网络同步 + 持久化）。
        """
        cache_table = validator.cache_table
        table_exists = (
            self.db.con.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables WHERE table_name = ?"
                ")",
                [cache_table.lower()],
            ).fetchone()[0]
            > 0
        )

        if not table_exists:
            self.logger.info(
                f"🔍 [Audit] 本地 SIMBAD 缓存表 `{cache_table}` 不存在，"
                f"将对 [{v_source}] 全量启动网络同步..."
            )
            df_all = self.db.con.execute(
                f"SELECT DISTINCT CAST(id AS VARCHAR) as id FROM {v_source}"
            ).df()
            ids_to_sync = df_all["id"].tolist()
        else:
            # 已在缓存表中的条目均为有效结果（含 parent="None" 即"SIMBAD 无数据"）
            sql_missing = f"""
                SELECT DISTINCT CAST(v.id AS VARCHAR) as id
                FROM {v_source} v
                LEFT JOIN {cache_table} c ON CAST(v.id AS VARCHAR) = c.gaia_dr3_id
                WHERE c.gaia_dr3_id IS NULL
            """
            self.logger.info(f"🔍 [Audit] 正在检索 [{v_source}] 中缺失的文献缓存记录...")
            df_missing = self.db.con.execute(sql_missing).df()
            ids_to_sync = df_missing["id"].tolist()

        if not ids_to_sync:
            self.logger.info("✅ [Audit] 缓存对齐完成：所有源均在本地缓存中。")
            return

        self.logger.info(
            f"🌐 [Network] 正在为 {len(ids_to_sync)} 个缺失源启动增量 SIMBAD 预热同步..."
        )
        validator.sync_simbad_cache(ids_to_sync)

    # ── 导出 ──

    def _export_phase(self, ctx: RunContext, audit_result: dict):
        """按需导出结果。"""
        if ctx.result_mode != "detailed":
            self.logger.info("⏩ [Export] 跳过物理文件导出。")
            return

        self.logger.info("💾 [Export] 正在执行数据资产导出任务...")

        export_base = cfg.TMPL.FILE_EXPORT_BASE.format(
            cluster=ctx.cluster_id,
            category=ctx.category,
            mode=ctx.feature_space,
            algo=ctx.algorithm,
        )

        self.db.export_table(ctx.master_table, export_dir=cfg.RESULTS_DIR)

        if audit_result.get("v_audit_pg_only"):
            self.db.export_table(
                audit_result["v_audit_pg_only"],
                filename=cfg.TMPL.FILE_DEEP_AUDIT.format(base=export_base + "_pg_only"),
                format="csv",
                export_dir=cfg.RESULTS_DIR,
            )

        if audit_result.get("v_audit_ref_only"):
            self.db.export_table(
                audit_result["v_audit_ref_only"],
                filename=cfg.TMPL.FILE_DEEP_AUDIT.format(base=export_base + "_ref_only"),
                format="csv",
                export_dir=cfg.RESULTS_DIR,
            )

        self.logger.info("✅ [Export] 结果导出完成。")

    # ── 报告 ──

    def _report_phase(self, ctx: RunContext, post_result: dict, audit_result: dict) -> dict:
        """生成最终报告并返回绩效摘要。"""
        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id

        return render_final_report(
            ctx.cluster_id,
            ctx.category,
            ctx.feature_space,
            ctx.algorithm,
            cluster_cfg,
            post_result,
            audit_result,
            audit_result.get("deep_stats_pg", {}),
            audit_result.get("deep_stats_ref", {}),
            self.logger,
        )

    def _render_batch_summary(self, all_results: list[dict]):
        """批量运行汇总报告。"""
        if not all_results:
            self.logger.warning("⚠️ 无有效结果，跳过汇总报告。")
            return

        from collections import defaultdict

        by_cluster = defaultdict(list)
        for r in all_results:
            by_cluster[r.get("cluster", "?")].append(r)

        for cluster_id, cluster_results in by_cluster.items():
            cluster_results.sort(key=lambda x: (x.get("mode", ""), x.get("algo", "")))
            render_all_modes_comparison(cluster_results, self.logger)

        self.logger.info(
            f"🏁 [Batch] 全量执行完成。共 {len(all_results)} 个组合产出有效结果。"
        )


