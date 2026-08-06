import logging
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from sklearn.decomposition import PCA

from utils.decorators import astro_checkpoint

from modules.db import AstroDB
from modules.pipelines.core.prior_gmm import PriorGMM
from modules.validator import UnifiedMemberValidator
from modules.transformer import AstroTransformer
from modules.reporter import render_all_modes_comparison
from modules.cluster import StarCluster

import config as cfg

# =============================================================================
# 📦 管线运行时状态（可变，在管线执行过程中逐步填充）
# =============================================================================


@dataclass
class PipelineState:
    """管线执行过程中逐步计算填充的可变状态。

    与 RunContext 的不可变身份字段分离，避免"声称不可变却处处修改"的设计矛盾。
    """

    gmm_config: dict = field(default_factory=dict)
    required_features: list = field(default_factory=list)
    master_table: str = ""
    seed_stats: dict = field(
        default_factory=dict
    )  # raw_count / clean_count / refined_count
    computed_dbscan_eps: str | None = None  # 运行时 KDE 解算的真实 eps（非 "auto"）

    # ── Phase 6 可视化：空间管 PCA 元数据（从 _build_spatial_tube 提取）──
    tube_pca_center: tuple | None = None       # (x0, y0) 管质心（赤道坐标 ra,dec）
    tube_pca_components: "np.ndarray | None" = None  # 2×2 分量矩阵
    tube_pca_mean: "np.ndarray | None" = None  # PCA fit 均值
    tube_width: float | None = None             # 管自适应半宽 (°)
    tube_length: float | None = None            # 管半长 (°)


# =============================================================================
# 📦 运行上下文数据类（单次运行的调度身份 + 可变状态引用）
# =============================================================================


@dataclass
class RunContext:
    """单次管线运行的调度上下文。

    身份字段（构造时确定，运行中不变）:
      - cluster_id, category, feature_space, algorithm, result_mode
      - param_source: 参数来源（"file" | "db"），用于构造 StarCluster
      - algo_params / audit_params / seed_params: 算法/审计/种子参数覆盖

    可变状态:
      - state: PipelineState，在管线各阶段逐步填充
      - star_cluster: 星团领域实体（Phase 1 创建后赋值）
    """

    cluster_id: str
    category: str  # "hunt", "cg20", etc.
    feature_space: str  # "2d", "5d", "6d_p", etc.
    algorithm: str  # "dbscan", "hdbscan"
    result_mode: str  # "brief" | "detailed"
    param_source: str  # "file" | "db" — 用于构造 StarCluster，之后以 star_cluster.param_source 为准
    skip_viz: bool = False  # 跳过可视化分析（Phase 6 子步骤）

    algo_params: dict = field(default_factory=dict)
    audit_params: dict = field(default_factory=dict)
    seed_params: dict = field(default_factory=dict)

    star_cluster: StarCluster | None = None
    state: PipelineState = field(default_factory=PipelineState)


# =============================================================================
# AstroWorkflow 主类
# =============================================================================


class AstroWorkflow:
    """天文数据处理工作流编排引擎。

    方法层次：
      - 一级 PUBLIC:  run() / _register_union_view()
      - 二级 阶段调度: _execute_single_pipeline() / _prepare_shared_data() /
                     _finalize_context() / _compute_members() / _post_process() /
                     _audit_phase() / _export_phase() / _report_phase()
      - 三级 功能单元: _standardize_ref_tables() / _load_and_transform_field() /
                     _load_and_transform_seeds() / _run_stable_pipeline() /
                     _run_experimental_pipeline() / _run_cross_match() / [A]
                     _run_phys_lit_fusion() / _audit_xmatch_subsets() / [B+C+D]
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
        self._field_cache = {}  # 场星数据跨 mode 缓存

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
        skip_viz: bool = False,
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
                    skip_viz=skip_viz,
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
                            skip_viz=skip_viz,
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
        self._register_union_view(results)
        return results

    # =========================================================================
    # 🟡 二级：阶段调度器
    # =========================================================================

    def _execute_single_pipeline(
        self, ctx: RunContext, skip_data_prep: bool = False
    ) -> dict | None:
        """[核心调度器] 串联完整管线 6 个阶段。"""
        # Phase 1: 数据准备与状态初始化
        self.logger.info(f"📦 [Phase 1] 数据准备与状态初始化: {ctx.cluster_id}")
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
        self.logger.info(f"⚖️ [Phase 4] 交叉审计, 参考星表: {ctx.category}")
        audit_result = self._audit_phase(ctx, post_result)

        # Phase 5: 导出
        self._export_phase(ctx, audit_result)

        # Phase 6: 报告与可视化
        return self._report_phase(ctx, post_result, audit_result)

    def _prepare_shared_data(self, ctx: RunContext):
        """执行可跨特征空间复用的数据准备：数据导入 + 星团实体 + 标准化。"""
        # self.logger.info(f"📦 [Phase 1] 数据准备: {ctx.cluster_id}")
        self.logger.info(f"💾 [DataPrep] 开始加载与标准化星团数据: {ctx.cluster_id}")

        self.db.import_raw(target_cluster=ctx.cluster_id, force=False)

        ctx.star_cluster = StarCluster(
            ctx.cluster_id, db_instance=self.db, param_source=ctx.param_source
        )

        # 🚀 必须在 load_or_reconstruct_parameters 之前执行标准化，
        # 因为参数重建（config_manager）依赖 aln 视图（如 aln_hunt_m44）已存在。
        self._standardize_ref_tables(ctx)

        success = ctx.star_cluster.load_or_reconstruct_parameters()
        if not success:
            raise RuntimeError(f"无法初始化星团 {ctx.cluster_id} 的物理资产")
        self.logger.info(
            f"✅ 星团领域模型就绪。"
            f"距离: {1000.0 / ctx.star_cluster.get_param('PLX_REF'):.1f} pc"
        )

        self.logger.info("✅ [DataPrep] 加载与标准化星团数据加载完成。")

    def _finalize_context(self, ctx: RunContext):
        """填充依赖于特征空间/算法的上下文属性。"""
        self.logger.info(f"📦 [DataPrep] 填充依赖于特征空间/算法的上下文属性: {ctx.cluster_id}")
        gmm_cfg = cfg.GMM_CONFIG.copy()
        gmm_cfg["FEATURE_SPACE"] = ctx.feature_space

        fmap = gmm_cfg.get("feature_map", {})
        if ctx.feature_space not in fmap:
            raise ValueError(f"未知的特征空间: {ctx.feature_space}")

        ctx.state.gmm_config = gmm_cfg
        ctx.state.required_features = fmap[ctx.feature_space]
        ctx.state.master_table = cfg.TMPL.T_MASTER.format(
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
                self.logger.debug(f"  ∟ [Process] 正在执行层级动作: {layer.upper()}")
                action_func = actions[layer]
                action_func(self.db, idx_data, cfg_data, self.manifest, local_ctx)

    # ── 特征工程 ──

    

    def _load_and_transform_field(self, ctx: RunContext) -> pd.DataFrame:
        """加载靶场数据 → 特征转换 → NaN清洗。"""
        field_idx = ctx.star_cluster.get_param("FIELD_IDX")
        cfg_source = self.manifest[field_idx]
        v_aln = cfg_source["aln_view"]

        # 缓存：同一 cluster 多次 mode 运行不复读 SQL
        cache_key = f"raw_{ctx.cluster_id}"
        df_raw = self._field_cache.get(cache_key)
        if df_raw is None:
            # 探测视图中实际存在的列，仅 SELECT 交集
            view_cols = {
                row[0] for row in self.db.con.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{v_aln}'"
                ).fetchall()
            }
            # 选择最小必要列
            cols = [c for c in cfg.MIN_FIELD_COLS if c in view_cols]
            cols_str = ', '.join(cols)
            df_raw = self.db.query(f"SELECT {cols_str} FROM {v_aln}")
            self._field_cache[cache_key] = df_raw
            self.logger.info(
                f"📋 [Process] 从视图 [{v_aln}] 读取目标天区数据: "
                f"{len(df_raw)} 颗 ({len(df_raw.columns)} 列)"
            )

        # 实例化封装后的 Transformer 并一键处理
        transformer = self._get_transformer_instance(ctx)
        return transformer.process_features(
            df_raw=df_raw,
            feature_space=ctx.feature_space,
            required_features=ctx.state.required_features,
            label="Target_field"
        )

    def _load_and_transform_seeds(self, ctx: RunContext) -> pd.DataFrame:
        """加载种子数据 → 特征转换 → NaN清洗。"""
        seed_idx = ctx.star_cluster.get_param("SEED_IDX")
        src = self.manifest[seed_idx]
        v_src = src["aln_view"]

        df_raw = self.db.query(f"SELECT * FROM {v_src}")
        ctx.state.seed_stats["raw_count"] = len(df_raw)
        self.logger.info(
            f"📋 [Process] 从视图 [{v_src}] 读取原始种子星: {len(df_raw)} 颗"
        )

        available_features = [
            f for f in ctx.state.required_features if f in df_raw.columns
        ]
        df_seeds = (
            df_raw.dropna(subset=available_features).copy()
            if available_features
            else df_raw.dropna().copy()
        )

        self.logger.info(f"✅ [Process] 种子星提取完成，有效样本: {len(df_seeds)} 颗")

        transformer = self._get_transformer_instance(ctx)
        df_clean = transformer.process_features(
            df_raw=df_seeds,
            feature_space=ctx.feature_space,
            required_features=ctx.state.required_features,
            label="Seeds"
        )
        
        ctx.state.seed_stats["clean_count"] = len(df_clean)

        # 🎯 标签回写必须在特征清洗之后，确保只标记最终实际使用的种子
        df_tag = df_clean[[cfg.STD_COLS["ID"]]].copy()
        df_tag["seed_type"] = "raw_seed"
        self.db.tag_master_table(ctx.state.master_table, df_tag)

        return df_clean

    

    def _get_transformer_instance(self, ctx: RunContext) -> AstroTransformer:
        """[辅助方法] 根据上下文星团资产，初始化配置好的 AstroTransformer。"""
        cl = ctx.star_cluster
        cluster_rv = cl.get_param("RV_REF", None)
        c_ra = cl.get_param("CENTER_RA", None)
        c_dec = cl.get_param("CENTER_DEC", None)
        cluster_center = (
            (c_ra, c_dec) if (c_ra is not None and c_dec is not None) else None
        )
        return AstroTransformer(
            cluster_rv=cluster_rv, cluster_center_icrs=cluster_center
        )

    # ── GMM 成员识别 ──

    @astro_checkpoint(
        cache_table_template="cache_{cluster}_{category}_{mode}_{algo}_res",
        force_refresh=True,
    )
    def _compute_members(self, ctx: RunContext) -> str | None:
        """统一的成员识别调度器 (Phase 2)"""
        self.logger.info(f"📊 [Compute] 管线请求的特征空间: {ctx.state.required_features}")

        # 1. 准备数据
        df_target_final = self._load_and_transform_field(ctx)
        self.db.init_master_table(ctx.state.master_table, df_target_final)
        df_seeds_final = self._load_and_transform_seeds(ctx)

        # 2. 统一委托给 pipelines 模块分发执行（常规轨 vs 实验轨）
        from modules.pipelines import run_pipeline
        df_res = run_pipeline(
            ctx=ctx,
            db_instance=self.db,
            df_all_field=df_target_final,
            df_seed_field=df_seeds_final
        )

        if df_res is None or df_res.empty:
            raise ValueError("❌ [Compute] 算法内核异常：结果 DataFrame 为空！")

        self.logger.info(f"✅ [Compute] 算法内核计算完成，结果集共计 {len(df_res)} 颗天体。")
        self.logger.info("📥 [Compute] 正在将概率结果同步至 Master 表...")

        # 🚀  计算金标成员和普通成员
        df_res["is_golden"] = df_res["prob"] >= cfg.THRESHOLD_GOLDEN
        df_res["is_candidate"] = df_res["prob"] > cfg.THRESHOLD_BASE

        # 🚀 确保数据库表里有这两列 (把 ALTER TABLE 搬到这里)
        self.db.execute(
            f"ALTER TABLE {ctx.state.master_table} "
            f"ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE"
        )
        self.db.execute(
            f"ALTER TABLE {ctx.state.master_table} "
            f"ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE"
        )

        # 3. 回灌 Master 表（概率及多通道属性）
        update_cols = [cfg.STD_COLS["ID"], "prob", "is_golden", "is_candidate"]
        for extra in ("core_prob", "tail_prob", "source"):
            if extra in df_res.columns:
                update_cols.append(extra)
                
        updates = df_res[update_cols].copy()
        self.db.tag_master_table(ctx.state.master_table, updates)

        return ctx.state.master_table

    # def _run_stable_pipeline(
    #     self,
    #     ctx: RunContext,
    #     df_target_final: pd.DataFrame,
    #     df_seeds_final: pd.DataFrame,
    # ) -> pd.DataFrame:
    #     """稳定生产轨：使用传统 PriorGMM。"""
    #     self.logger.warning("🔒 [Compute] 稳定生产模式：执行 PriorGMM 老轨行为")

    #     cluster_cfg = cfg.CLUSTERS[ctx.cluster_id].copy()
    #     cluster_cfg["id"] = ctx.cluster_id
    #     cluster_cfg["FEATURE_SPACE"] = ctx.feature_space

    #     engine = PriorGMM(config=cluster_cfg)
    #     model_params = engine.fit(df_seeds_final, df_target_final)
    #     return engine.predict(df_target_final, model_params)
    
    # def _run_experimental_pipeline(
    #     self, ctx: RunContext, df_all_field: pd.DataFrame, df_seed_field: pd.DataFrame
    # ) -> pd.DataFrame:
    #     """
    #     实验新轨：委托给独立执行器 ExperimentalPipelineRunner 执行双通道策略。
    #     """
    #     # ⚠️ 局部导入以避免潜在的循环依赖
    #     from modules.pipelines.experimental_pipeline import ExperimentalPipelineRunner
        
    #     self.logger.info("⚙️ [Workflow] 将管线执行权移交至 ExperimentalPipelineRunner")
    #     runner = ExperimentalPipelineRunner(db_instance=self.db, logger=self.logger)
        
    #     # 纯净的输入输出交互
    #     return runner.run(ctx, df_all_field, df_seed_field)

    # ── 后处理 ──

    def _post_process(self, ctx: RunContext, t_main_results: str) -> dict:
        """算法后处理流水线。"""
        self.logger.info(f"📊 [Process] [{ctx.cluster_id}] 启动后处理...")
        try:
            # self.db.execute(
            #     f"ALTER TABLE {ctx.state.master_table} "
            #     f"ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE"
            # )
            # self.db.execute(
            #     f"ALTER TABLE {ctx.state.master_table} "
            #     f"ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE"
            # )

            # condi_golden = f"{cfg.STD_COLS['PROB']} >= {cfg.THRESHOLD_GOLDEN}"
            # condi_candidates = f"{cfg.STD_COLS['PROB']} > {cfg.THRESHOLD_BASE}"

            # self.db.execute(
            #     f"UPDATE {ctx.state.master_table} SET is_golden = TRUE WHERE {condi_golden}"
            # )
            # self.db.execute(
            #     f"UPDATE {ctx.state.master_table} SET is_candidate = TRUE WHERE {condi_candidates}"
            # )

            stats_sql = f"""
                SELECT 
                    count(*) FILTER (WHERE is_golden = TRUE) AS n_golden,
                    count(*) FILTER (WHERE is_candidate = TRUE) AS n_candidates,
                    count(*) FILTER (WHERE seed_type = 'refined_seed') AS n_seeds_refined,
                    count(*) FILTER (WHERE density_status = 'core') AS n_seed_core,
                    count(*) FILTER (WHERE density_status = 'noise') AS n_seed_noise
                FROM {ctx.state.master_table}
            """
            stats = self.db.execute(stats_sql).fetchone()
            n_golden, n_candidates, n_seeds_refined, n_seed_core, n_seed_noise = stats

            # 种子计数从 seed_stats 取（不受 tag_master_table UPDATE 限制影响）
            sd = ctx.state.seed_stats
            n_seeds_raw = sd.get("raw_count", 0)
            n_seeds_clean = sd.get("clean_count", 0)

            self.logger.info("=" * 60)
            self.logger.info(f"📊 [Process] [{ctx.cluster_id}] 后处理标签同步完成:")
            self.logger.info(f"  🔹 高置信金种子星 (is_golden): {n_golden} 颗")
            self.logger.info(f"  🔹 成员星候选总数 (is_candidate): {n_candidates} 颗")
            self.logger.info(f"  🔹 原始种子星目录总数 (Raw): {n_seeds_raw} 颗")
            self.logger.info(f"  🔹 有效输入种子星 (Clean): {n_seeds_clean} 颗")
            self.logger.info(f"  🔹 DBSCAN 精炼种子星 (Refined): {n_seeds_refined} 颗")
            self.logger.info(f"  🔹 种子集核心样本 (Core): {n_seed_core} 颗")
            self.logger.info("=" * 60)

            v_candidates = cfg.TMPL.V_CANDIDATES.format(
                cluster=ctx.cluster_id.lower(),
                category=ctx.category,
                feature_space=ctx.feature_space,
                algo=ctx.algorithm,
            )
            self.db.register_view_from_sql(
                v_candidates,
                f"SELECT * FROM {ctx.state.master_table} WHERE is_candidate = TRUE",
            )

            return {
                "status": "success",
                "v_candidates": v_candidates,
                "stats": {
                    "n_golden": n_golden,
                    "n_candidates": n_candidates,
                    "n_seeds_raw": n_seeds_raw,
                    "n_seeds_clean": n_seeds_clean,
                    "n_seed_core": n_seed_core,
                    "n_seed_noise": n_seed_noise,
                    "n_seeds_refined": n_seeds_refined,
                    **ctx.state.seed_stats,
                },
            }
        except Exception as e:
            self.logger.error(f"❌ [Process] 后处理失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ── 审计 ──
    # Phase 4 分为三个核心审计 + 一个交叉比对辅助步骤:
    #   [A] 交叉比对     (_run_cross_match)     — PG 结果 vs 参考星表
    #   [B] 物理审计     (_run_phys_lit_fusion)  — 卡方/加权惩罚 (Chi2Upmask/Residual/WeightedPenalty)
    #   [C] 文献审计     (同上, validator.run 内部) — SIMBAD 文献共识
    #   [D] 融合决策     (同上, validator.run 内部) — 物理 × 文献 → audit_status

    def _audit_phase(self, ctx: RunContext, post_result: dict) -> dict:
        """审计阶段：[A] 交叉比对 → [B+C+D] 物理+文献+融合。"""
        audit_res = self._run_cross_match(ctx)
        if audit_res.get("status") != "success":
            self.logger.warning(
                f"⚠️ [Audit] 交叉比对未完全成功: {audit_res.get('message')}"
            )
            return audit_res

        self.logger.info("✅ [Audit] 交叉比对完成。")
        audit_stats = self._audit_xmatch_subsets(ctx, audit_res)
        audit_res.update(audit_stats)
        return audit_res

    # ── [A] 交叉比对 ──

    def _run_cross_match(self, ctx: RunContext) -> dict:
        """4a. PG/Ref 交叉比对：标记匹配状态，区分 Matched/PG Only/Ref Only。"""
        v_target = self.manifest[ctx.category]["aln_view"].format(
            cluster=ctx.cluster_id.lower()
        )

        # 检查审计目标表是否存在
        if not self.db:
            self.logger.warning(f"⚠️ [Audit] 审计目标表 '{v_target}' 不存在。")
            return {"status": "warning", "message": f"审计目标表 '{v_target}' 不存在"}
        exists = self.db.con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [v_target],
        ).fetchone() is not None
        if not exists:
            self.logger.warning(f"⚠️ [Audit] 审计目标表 '{v_target}' 不存在。")
            return {"status": "warning", "message": f"审计目标表 '{v_target}' 不存在"}

        self.logger.info(f"⚡ [Audit] 开始交叉比对: {v_target}")

        col_x = cfg.MASTER_COLS["X_MATCH"]
        sql_cross = f"""
            SELECT 
                COALESCE(m.id, h.id) as id,
                CASE 
                    WHEN m.prob > {cfg.THRESHOLD_BASE}
                         AND h.id IS NOT NULL THEN 'Matched'
                    WHEN m.prob > {cfg.THRESHOLD_BASE}
                         AND h.id IS NULL     THEN 'PG Only'
                    WHEN (m.id IS NULL OR m.prob <= {cfg.THRESHOLD_BASE}
                         OR m.prob IS NULL)
                         AND h.id IS NOT NULL THEN 'Ref Only'
                END as {col_x}
            FROM {ctx.state.master_table} m
            FULL OUTER JOIN {v_target} h ON m.id = h.id
            WHERE m.prob > {cfg.THRESHOLD_BASE} OR h.id IS NOT NULL
        """
        df_x = self.db.query(sql_cross)
        self.db.tag_master_table(ctx.state.master_table, df_x)

        self.logger.info(f"✅ [Audit] Master 表 x_match_tag 更新完成。")

        v_audit_pg_only = f"v_tmp_audit_pg_only_{ctx.state.master_table}"
        v_audit_ref_only = f"v_tmp_audit_ref_only_{ctx.state.master_table}"
        self.db.register_view_from_sql(
            v_audit_pg_only,
            f"SELECT * FROM {ctx.state.master_table} WHERE {col_x} = 'PG Only'",
        )
        self.db.register_view_from_sql(
            v_audit_ref_only,
            f"SELECT * FROM {ctx.state.master_table} WHERE {col_x} = 'Ref Only'",
        )

        st_sql = (
            f"SELECT {col_x}, count(*) FROM {ctx.state.master_table} "
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

        # 新增：创建双方共识成员视图 (Matched)
        v_audit_matched = f"v_tmp_audit_matched_{ctx.state.master_table}"
        self.db.register_view_from_sql(
            v_audit_matched,
            f"SELECT * FROM {ctx.state.master_table} WHERE {col_x} = 'Matched'",
        )

        return {
            "status": "success",
            "v_audit_pg_only": v_audit_pg_only,
            "v_audit_ref_only": v_audit_ref_only,
            "v_audit_matched": v_audit_matched,
            "stats": stats_cross,
        }

    # ── [B+C+D] 物理审计 + 文献审计 + 融合决策 ──

    def _audit_xmatch_subsets(self, ctx: RunContext, cross_result: dict) -> dict:
        """对三个原子子集（PG Only / Ref Only / Matched）分别执行 [B]物理 + [C]文献 + [D]融合 审计。

        组合集（Category = Matched+Ref Only, PG Algo = Matched+PG Only）
        通过简单相加原子子集的 deep_stats 获得，避免重复运行 Validator。
        """
        audit_stats = {}
        for subset_type, label in [("pg_only", "PG Only"), ("ref_only", "Ref Only"),
                                    ("matched", "Matched")]:
            view = cross_result.get(f"v_audit_{subset_type}")
            count = cross_result.get("stats", {}).get(label, 0)
            if view and count > 0:
                v_result, stats = self._run_phys_lit_fusion(ctx, view, subset_type)
                audit_stats[f"deep_stats_{subset_type}"] = stats
            else:
                self.logger.warning(f"⚠️ [Audit] 无 {label} 候选，跳过审计。")

        # ── 组合集：通过原子子集相加获得 ──
        deep_pg = audit_stats.get("deep_stats_pg_only", {})
        deep_ref = audit_stats.get("deep_stats_ref_only", {})
        deep_matched = audit_stats.get("deep_stats_matched", {})
        all_keys = {"Confirmed Member", "New Candidate", "Literature Only", "Contamination"}

        if deep_matched and deep_ref:
            cat = ctx.category
            audit_stats[f"deep_stats_{cat}"] = {
                k: deep_matched.get(k, 0) + deep_ref.get(k, 0)
                for k in all_keys
            }
        if deep_matched and deep_pg:
            audit_stats["deep_stats_pg_algo"] = {
                k: deep_matched.get(k, 0) + deep_pg.get(k, 0)
                for k in all_keys
            }

        return audit_stats

    def _run_phys_lit_fusion(
        self, ctx: RunContext, target: str, audit_type: str = "default"
    ) -> tuple:
        """[B+C+D] 对单个候选子集执行物理审计 + 文献审计 + 融合决策，返回 (view_name, stats_dict)。

        内部由 UnifiedMemberValidator.run() 串联三个子阶段:
          ① 物理一致性审计 — 卡方检验 (Chi2Upmask/Chi2Residual) 或加权惩罚 (WeightedPenalty)
          ② 文献共识审计   — SIMBAD 语义树匹配 (LiteratureAuditor)
          ③ 融合决策       — 物理 × 文献 → Confirmed Member / New Candidate / Literature Only / Contamination
        """
        self.logger.info(f"🔍 🎬 [Audit] 开始 [{audit_type}] 审计: {target}")

        try:
            v_audit_input = self._prepare_audit_input(ctx, target)
            if not v_audit_input:
                self.logger.error("❌ [Audit] 审计预处理失败")
                return None, {}

            validator = UnifiedMemberValidator(
                cluster=ctx.star_cluster,
                db_instance=self.db,
                feature_space=ctx.feature_space,
            )

            # ── ① 文献审计前置：SIMBAD 缓存预热 ──
            self._warm_up_literature_cache(validator, v_audit_input)

            # ── ② UnifiedMemberValidator.run()
            #       内部自动串联: 物理审计 → 文献审计 → 融合决策
            self.logger.info(
                f"⚖️ [Audit] 启动 Validator: "
                f"物理审计({cfg.VALIDATION_STRATEGY}) + 文献审计(SIMBAD) + 融合决策"
            )
            audit_report_df = validator.run(v_audit_input)

            self.logger.info("📥 [Audit] 正在将审计结果同步至 Master 表...")
            self.db.tag_master_table(ctx.state.master_table, audit_report_df)

            v_report = f"{ctx.state.master_table}_{audit_type}_audited_report"
            col_x = cfg.MASTER_COLS["X_MATCH"]
            x_match_val = (
                "PG Only"
                if audit_type == "pg_only"
                else "Ref Only" if audit_type == "ref_only" else None
            )

            if x_match_val:
                sql_filter = (
                    f"SELECT * FROM {ctx.state.master_table} "
                    f"WHERE audit_status IS NOT NULL AND {col_x} = '{x_match_val}'"
                )
            else:
                # target 是已限定范围的视图（如 v_tmp_audit_hunt_...）
                # 用 INNER JOIN 确保统计只包含该视图内的行，而非全表
                sql_filter = (
                    f"SELECT m.* FROM {ctx.state.master_table} m "
                    f"INNER JOIN {target} t ON m.id = t.id "
                    f"WHERE m.audit_status IS NOT NULL"
                )

            self.db.register_view_from_sql(v_report, sql_filter)

            # 统计审计结果
            sql = (
                f"SELECT audit_status, count(*) FROM {v_report} "
                f"WHERE audit_status IS NOT NULL GROUP BY audit_status"
            )
            stats = dict(self.db.con.execute(sql).fetchall())
            return v_report, stats

        except Exception as e:
            self.logger.error(f"❌ [Audit] 审计流程故障: {str(e)}", exc_info=True)
            return None, {}

    def _prepare_audit_input(self, ctx: RunContext, v_target: str) -> str | None:
        """审计前准备：补全物理参数并注册审计输入视图。"""
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

    def _warm_up_literature_cache(
        self, validator: UnifiedMemberValidator, v_source: str
    ):
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
            self.logger.info(
                f"🔍 [Audit] 正在检索 [{v_source}] 中缺失的文献缓存记录..."
            )
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

        self.db.export_table(ctx.state.master_table, export_dir=cfg.RESULTS_DIR)

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
                filename=cfg.TMPL.FILE_DEEP_AUDIT.format(
                    base=export_base + "_ref_only"
                ),
                format="csv",
                export_dir=cfg.RESULTS_DIR,
            )

        self.logger.info("✅ [Export] 结果导出完成。")

    # ── 报告 ──

    def _report_phase(
        self, ctx: RunContext, post_result: dict, audit_result: dict
    ) -> dict:
        """Phase 6: 报告与可视化 — 委托给 AstroAnalyzer。"""
        from modules.analysis import AstroAnalyzer
        return AstroAnalyzer(self.db, ctx=ctx).run_phase6(post_result, audit_result)

    def _register_union_view(self, all_results: list[dict]):
        """注册跨星团 UNION ALL 联合视图。

        在所有 pipeline 完成后，将各星团 master 表拼接为统一查询入口。
        单星团运行时只有一段，多星团时自动合并。
        视图名包含 category/feature_space/algo 维度，避免批量运行时冲突。
        """
        seen_tables: set[str] = set()
        union_parts: list[str] = []
        canonical = all_results[0] if all_results else {}
        for r in all_results:
            cluster = r.get("cluster", "?")
            tbl = cfg.TMPL.T_MASTER.format(
                cluster=cluster.lower(),
                category=r.get("category", "hunt"),
                feature_space=r.get("mode", "5d_h").lower(),
                algo=r.get("algo", "dbscan").lower(),
            )
            exists = self.db.con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [tbl]
            ).fetchone() is not None
            if exists and tbl not in seen_tables:
                seen_tables.add(tbl)
                union_parts.append(
                    f"SELECT '{cluster}' AS cluster_id, t.* FROM {tbl} t"
                )

        if not union_parts:
            self.logger.warning("⚠️ [UnionView] 无有效 master 表，跳过联合视图。")
            return

        v_union = cfg.TMPL.V_UNION.format(
            category=canonical.get("category", "hunt"),
            feature_space=canonical.get("mode", "5d_h").lower(),
            algo=canonical.get("algo", "dbscan").lower(),
        )
        sql = (
            f"CREATE OR REPLACE VIEW {v_union} AS\n"
            + "\nUNION ALL\n".join(union_parts)
        )
        self.db.execute(sql)
        self.logger.info(
            f"🌐 [UnionView] 跨星团联合视图已注册: {v_union} "
            f"({len(union_parts)} 个星团)"
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
