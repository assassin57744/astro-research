import logging
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from sklearn.decomposition import PCA

from utils.decorators import astro_checkpoint

from modules.db import AstroDB
from modules.pg_core import PriorGMM
from modules.validator import UnifiedMemberValidator
from modules.transformer import AstroTransformer
from modules.reporter import render_final_report, render_all_modes_comparison
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
      - 一级 PUBLIC:  run()
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

        success = ctx.star_cluster.load_or_reconstruct_parameters()
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

        df_raw = self.db.query(f"SELECT * FROM {v_aln}")
        self.logger.info(
            f"📋 [Process] 从视图 [{v_aln}] 读取目标天区数据: {len(df_raw)} 颗"
        )

        df_ext = self._transform_and_bridge_features(
            df_raw, ctx.feature_space, ctx.state.required_features, ctx.star_cluster
        )
        return self._defensive_nan_purge(
            df_ext, ctx.state.required_features, label="Target_field"
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

        df_ext = self._transform_and_bridge_features(
            df_seeds, ctx.feature_space, ctx.state.required_features, ctx.star_cluster
        )
        df_clean = self._defensive_nan_purge(
            df_ext, ctx.state.required_features, label="Seeds"
        )
        ctx.state.seed_stats["clean_count"] = len(df_clean)

        # 🎯 标签回写必须在特征清洗之后，确保只标记最终实际使用的种子
        df_tag = df_clean[[cfg.STD_COLS["ID"]]].copy()
        df_tag["seed_type"] = "raw_seed"
        self.db.tag_master_table(ctx.state.master_table, df_tag)

        return df_clean

    def _transform_and_bridge_features(
        self,
        df_raw: pd.DataFrame,
        feature_space: str,
        required_features: list[str],
        star_cluster: StarCluster | None = None,
    ) -> pd.DataFrame:
        """特征转换网关。"""
        if df_raw is None:
            self.logger.error("❌ [Compute] 输入的原始 DataFrame 为 None！")
            return None

        self.logger.debug(
            f"🚀 [Compute] 正在执行特征转换，特征空间: {required_features}"
        )

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

        df_features = pd.DataFrame(
            X_array, columns=required_features, index=df_raw.index
        )

        dup_cols = [col for col in df_raw.columns if col in (cols_upper + cols_lower)]
        if dup_cols:
            self.logger.info(
                f"🔄 [Compute] 模式 [{feature_space}] 移除重复列: {dup_cols}"
            )
            df_raw = df_raw.drop(columns=dup_cols)
        return pd.concat([df_raw, df_features], axis=1)

    def _defensive_nan_purge(
        self, df_extended: pd.DataFrame, required_features: list[str], label: str
    ) -> pd.DataFrame:
        """特征清洗。"""
        if df_extended is None or df_extended.empty:
            self.logger.error(f"❌ [Compute] [{label}] 数据为空！")
            return pd.DataFrame()

        initial_count = len(df_extended)

        # 🌟 防护 1：校验 required_features 是否存在于 DataFrame 中
        missing_cols = [f for f in required_features if f not in df_extended.columns]
        if missing_cols:
            self.logger.error(f"❌ [Compute] [{label}] 缺失必要特征列: {missing_cols}")
            raise KeyError(f"Missing required feature columns: {missing_cols}")

        # 🌟 防护 2：打印具体是哪一列导致了 NaN 剔除（极其关键的诊断日志）
        nan_counts = df_extended[required_features].isna().sum()
        culprit_cols = nan_counts[nan_counts > 0].to_dict()
        if culprit_cols:
            self.logger.warning(
                f"🔍 [Compute] [{label}] 特征列 NaN 分布细目: {culprit_cols}"
            )

        # 执行核心 Dropna
        df_clean = df_extended.dropna(subset=required_features).copy()
        dropped = initial_count - len(df_clean)

        if dropped > 0:
            self.logger.warning(
                f"⚠️ [Compute] [防御性过滤 - {label}]: 剔除 {dropped} 颗, "
                f"剩余 {len(df_clean)}。"
            )
        else:
            self.logger.info(
                f"✅ [Compute] [数据预检 - {label}] 共计 {len(df_clean)} 颗。"
            )

        return df_clean

    # ── GMM 成员识别 ──

    @astro_checkpoint(
        cache_table_template="cache_{cluster}_{category}_{mode}_{algo}_res",
        force_refresh=True,
    )
    def _compute_members(self, ctx: RunContext) -> str | None:
        """统一的成员识别调度器。"""
        self.logger.info(
            f"📊 [Compute] 管线请求的特征空间: {ctx.state.required_features}"
        )

        df_target_final = self._load_and_transform_field(ctx)
        self.logger.info(
            f"✅ [Compute] 算法内核计算完成，目标天区共计 {len(df_target_final)} 颗天体。top N: \n{df_target_final.head()}"
        )

        # 🚀 必须先建表，再加载种子（_load_and_transform_seeds 内部会调用 tag_master_table 回灌标签）
        self.db.init_master_table(ctx.state.master_table, df_target_final)
        df_seeds_final = self._load_and_transform_seeds(ctx)
        self.logger.info(
            f"✅ [Compute] 算法内核计算完成，种子星共计 {len(df_seeds_final)} 颗。top N: \n{df_seeds_final.head()}"
        )

        use_experimental = ctx.state.gmm_config.get("use_experimental", False)
        if not use_experimental:
            df_res = self._run_stable_pipeline(ctx, df_target_final, df_seeds_final)
        else:
            df_res = self._run_experimental_pipeline(
                ctx, df_target_final, df_seeds_final
            )

        if df_res is None or df_res.empty:
            raise ValueError("❌ [Compute] 算法内核异常：结果 DataFrame 为空！")

        self.logger.info(
            f"✅ [Compute] 算法内核计算完成，结果集共计 {len(df_res)} 颗天体。"
        )
        self.logger.info("📥 [Compute] 正在将概率结果同步至 Master 表...")

        # 回灌概率及分通道信息（若存在 core_prob / tail_prob / source 则一并写入）
        update_cols = [cfg.STD_COLS["ID"], "prob"]
        for extra in ("core_prob", "tail_prob", "source"):
            if extra in df_res.columns:
                update_cols.append(extra)
        updates = df_res[update_cols].copy()
        self.db.tag_master_table(ctx.state.master_table, updates)

        return ctx.state.master_table

    def _run_stable_pipeline(
        self,
        ctx: RunContext,
        df_target_final: pd.DataFrame,
        df_seeds_final: pd.DataFrame,
    ) -> pd.DataFrame:
        """稳定生产轨：使用传统 PriorGMM。"""
        self.logger.warning("🔒 [Compute] 稳定生产模式：执行 PriorGMM 老轨行为")

        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id].copy()
        cluster_cfg["id"] = ctx.cluster_id
        cluster_cfg["dim_mode"] = ctx.feature_space

        engine = PriorGMM(config=cluster_cfg)
        model_params = engine.fit(df_seeds_final, df_target_final)
        return engine.predict(df_target_final, model_params)

    def _run_experimental_pipeline(
        self, ctx: RunContext, df_all: pd.DataFrame, df_seeds: pd.DataFrame
    ) -> pd.DataFrame:
        """实验新轨：Core 核心识别 + Spatial Tube 潮汐尾捕捉 + 分层互斥决策融合。"""
        strategy_name = ctx.algo_params.get("strategy", "bayesian")
        self.logger.info(
            f"🚀 [Compute] 启动双通道分层捕捉管线 | 策略: [{strategy_name.upper()}]"
        )

        # 🌟 物理空间全景诊断
        self.logger.info(
            f"🔍 [Target Field Diagnosis] 传入天体总量: {len(df_all)} 颗 | "
            f"RA 范围: [{df_all['ra'].min():.2f}°, {df_all['ra'].max():.2f}°] | "
            f"Dec 范围: [{df_all['dec'].min():.2f}°, {df_all['dec'].max():.2f}°] | "
            f"l 范围: [{df_all['l'].min():.2f}°, {df_all['l'].max():.2f}°]"
        )

        # ---------------------------------------------------------
        # 1. 第一阶段：种子星精炼 (ClusterSeedExtractor)
        # ---------------------------------------------------------
        self.logger.info("🧬 [Compute] 正在调度 ClusterSeedExtractor...")
        from modules.seed_extractor import ClusterSeedExtractor

        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id
        extractor = ClusterSeedExtractor(cluster_profile=cluster_cfg)
        df_seeds_core = extractor.extract_seeds(
            seed_field_df=df_seeds,
            features=ctx.state.required_features,
        )

        if df_seeds_core is None or df_seeds_core.empty:
            raise ValueError("❌ [Compute] ClusterSeedExtractor 未能凝聚出有效种子星！")

        # 记录种子统计信息并回写 Master 表
        ctx.state.seed_stats["refined_count"] = len(df_seeds_core)
        self.logger.info(f"✅ [Compute] 种子星粗筛成功！共 {len(df_seeds_core)} 颗。")

        df_tag_refined = df_seeds_core[[cfg.STD_COLS["ID"]]].copy()
        df_tag_refined["seed_type"] = "refined_seed"
        self.db.tag_master_table(ctx.state.master_table, df_tag_refined)

        # ---------------------------------------------------------
        # 2. 第二阶段：实例化消歧引擎策略
        # ---------------------------------------------------------
        strategy_params = (
            cfg.CLUSTERS[ctx.cluster_id.upper()]
            .get("STRATEGY_PARAMS", {})
            .get(strategy_name, {})
        )
        strategy_kwargs = {**strategy_params}
        strategy_kwargs.setdefault("spatial_cols", ["l", "b"])
        strategy_kwargs.setdefault("scale_col", "plx")

        for key in ("eps", "min_samples", "sigma_cutoff"):
            if key in ctx.algo_params:
                strategy_kwargs[key] = ctx.algo_params[key]
                self.logger.info(f"✅ [Compute] 已覆盖设置策略参数 [{key}]")

        from modules.astro_membership.disambiguation.bayesian import (
            BayesianGmmDisambiguation,
        )
        from modules.astro_membership.disambiguation.blind import (
            BlindGmmDisambiguation,
        )
        from modules.astro_membership.disambiguation.threshold import (
            ThresholdGmmDisambiguation,
        )

        STRATEGY_CLASSES = {
            "bayesian": BayesianGmmDisambiguation,
            "threshold": ThresholdGmmDisambiguation,
            "blind": BlindGmmDisambiguation,
        }

        if strategy_name not in STRATEGY_CLASSES:
            raise ValueError(f"未知的策略类型 [{strategy_name}]")

        engine_class = STRATEGY_CLASSES[strategy_name]
        self.logger.info(f"🧠 [Compute] 路由至消歧拟合引擎 [{engine_class.__name__}]")
        engine = engine_class(**strategy_kwargs)

        # ---------------------------------------------------------
        # 3. 通道 A：抓取星团 Core 核心成员
        # ---------------------------------------------------------
        self.logger.info("🎯 [Channel A] 启动星团 Core 核心区域抓取...")
        df_res_core = engine.fit_predict(
            df_all, df_seeds_core, ctx.state.required_features
        )
        prob_col = cfg.STD_COLS["PROB"]

        # ---------------------------------------------------------
        # 4. 通道 B：构建类内空间管，抓取 Tidal Tail / 弥散外围成员
        # ---------------------------------------------------------
        self.logger.info("📐 [Channel B] 启动 Tidal Tail 空间管构建与抓取...")
        from utils.tube import plot_spatial_tube

        cl = ctx.star_cluster

        # 🌟 4.1 自适应管尺寸 (含 PCA 15 度 Hard Cap 防护)
        tidal_radius_pc = cl.get_param("TIDAL_RADIUS", None)
        distance_pc = cl.get_param("DISTANCE_PC", None)
        if tidal_radius_pc and distance_pc and distance_pc > 0:
            angular_tidal = np.degrees(tidal_radius_pc / distance_pc)
            # 管长乘数：可从 config 覆盖（默认 1.6），M45 等扩散星团需更大值
            length_mult = cl.get_param("TUBE_LENGTH_MULTIPLIER", 1.6)
            # 管宽乘数：可从 config 覆盖（默认 0.3）
            width_mult = cl.get_param("TUBE_WIDTH_MULTIPLIER", 0.3)

            raw_length_deg = angular_tidal * length_mult
            # 🌟 增加 Hard Cap 防护，防止 M45 等近邻星团空间管过长膨胀
            max_half_length_deg = cl.get_param("MAX_TUBE_HALF_LENGTH", 15.0)
            length_deg = min(raw_length_deg, max_half_length_deg)

            width_deg = max(1.0, angular_tidal * width_mult)
            self.logger.info(
                f"📐 [Adaptive Tube] TIDAL_RADIUS={tidal_radius_pc}pc, "
                f"DISTANCE={distance_pc:.0f}pc → 角尺度={angular_tidal:.2f}° → "
                f"管长={length_deg:.1f}°(硬门限截断 $\\le${max_half_length_deg}°), "
                f"管宽={width_deg:.1f}°"
            )
        else:
            length_deg = cl.get_param("TUBE_LENGTH", 5.0)
            width_deg = cl.get_param("TUBE_WIDTH", 1.0)
            self.logger.info(
                f"📐 [Fallback Tube] 物理参数缺失，使用配置值: "
                f"管长={length_deg}°, 管宽={width_deg}°"
            )

        # 纯净 5D 特征切片提取
        df_all_clean = df_all[ctx.state.required_features].copy()
        df_seeds_clean = df_seeds_core[ctx.state.required_features].copy()

        # 🌟 4.2 构造空间管切片
        df_tube_clean, pca_model = self._build_spatial_tube(
            df_all_clean,
            df_seeds_clean,
            length_deg=length_deg,
            width_deg=width_deg,
        )

        # 绘制质检图
        plot_spatial_tube(
            df_all=df_all_clean,
            df_seeds=df_seeds_clean,
            df_tube=df_tube_clean,
            pca=pca_model,
            length_deg=length_deg,
            width_deg=pca_model.tube_width_,
            cluster_id=ctx.cluster_id,
            output_dir=cfg.ANALYSIS_DIR,
        )

        # 🌟 4.3 运动学 Sigma Clip (优先选用 Channel A 算出的高纯度 Core 作为基准源)
        pm_cols = ["pm_l_cosb", "pm_b", "plx"]
        sigma_clip = cl.get_param("TUBE_SIGMA_CLIP", 5.0)

        # 提取高置信 Core 天体的 ID 集合
        core_high_conf_mask = df_res_core[prob_col] >= cfg.THRESHOLD_HIGH_CONF
        pure_core_ids = set(df_res_core.loc[core_high_conf_mask, "id"].values)

        # 🌟 核心修复：基于 id 字段从 df_all 中安全提取 pure core 5D 特征
        if len(pure_core_ids) >= 20:
            core_mask_in_all = df_all["id"].isin(pure_core_ids)
            ref_df = df_all.loc[core_mask_in_all, ctx.state.required_features]
            ref_label = f"Pure Core (GMM P >= {cfg.THRESHOLD_HIGH_CONF})"
        else:
            ref_df = df_seeds_clean
            ref_label = "Refined Seeds (Fallback)"

        if sigma_clip > 0 and not df_tube_clean.empty and not ref_df.empty:
            ref_vals = ref_df[pm_cols].values.astype(np.float64)

            # 使用矩估计并注入正则化因子，防止伪逆奇异
            ref_mean = ref_vals.mean(axis=0)
            ref_cov = np.cov(ref_vals, rowvar=False) + np.eye(3) * 1e-6
            inv_cov = np.linalg.pinv(ref_cov)

            tube_vals = df_tube_clean[pm_cols].values.astype(np.float64)
            diff = tube_vals - ref_mean
            md_sq = np.sum((diff @ inv_cov) * diff, axis=1)

            n_before = len(df_tube_clean)
            clip_mask = md_sq <= sigma_clip**2
            df_tube_clean = df_tube_clean[clip_mask].copy()

            self.logger.info(
                f"📐 [Tube Sigma Clip] 基准源: {ref_label} ({len(ref_df)} 颗) | "
                f"σ_clip={sigma_clip} (Mahalanobis) | "
                f"过滤: {n_before} → {len(df_tube_clean)} 颗 "
                f"(剔除 {(n_before - len(df_tube_clean)) / n_before * 100:.1f}%)"
            )

        # 提取管内完整天体视图
        valid_tube_indices = df_all.index.intersection(df_tube_clean.index)
        df_tube_full = df_all.loc[valid_tube_indices].copy()

        for feat in ctx.state.required_features:
            if feat in df_tube_clean.columns:
                df_tube_full[feat] = df_tube_clean.loc[valid_tube_indices, feat]

        self.logger.info(
            f"📊 [Channel B] 空间管内捕获有效候选天体: {len(df_tube_full)} 颗"
        )

        # 🌟 4.4 防御性熔断：如管内为空，退回 Core 结果
        if df_tube_full.empty:
            self.logger.warning(
                "⚠️ [Channel B] 空间管内未捕获到任何有效天体，跳过 Tidal Tail 洗涤，返回 Core 结果。"
            )
            df_res_core["core_prob"] = df_res_core[prob_col]
            df_res_core["tail_prob"] = 0.0
            df_res_core["source"] = "core"
            return df_res_core

        # 从去核管星中构建 Tail 种子模板
        core_prob_map = df_res_core.set_index("id")[prob_col]
        tube_ids_from_idx = df_all.loc[df_tube_clean.index, "id"]
        tube_core_probs = tube_ids_from_idx.map(core_prob_map)

        tail_template_mask = tube_core_probs < cfg.THRESHOLD_BASE
        df_tail_template = df_tube_clean[tail_template_mask].copy()
        n_removed = len(df_tube_clean) - len(df_tail_template)

        self.logger.info(
            f"📐 [Tail Template] σ={sigma_clip} 管星: {len(df_tube_clean)} → "
            f"{len(df_tail_template)} 颗 (剔除 {n_removed} 颗 Core 成员)"
        )

        # 🌟 4.5 两阶段推导 (管内 fit + 管内 predict)
        tail_params = engine.fit(
            df_tube=df_tube_full,
            df_seeds=df_tail_template,
            features=ctx.state.required_features,
            use_density_prune=False,
        )

        df_res_tail = engine.predict(
            df_all=df_tube_full,
            model_params=tail_params,
            features=ctx.state.required_features,
        )

        self.logger.info(
            f"📊 [Channel B] 管内(σ={sigma_clip})全量: {len(df_tube_full)} 颗 | "
            f"Tail 候选(prob>{cfg.THRESHOLD_BASE}): {(df_res_tail[prob_col] > cfg.THRESHOLD_BASE).sum()} 颗 | "
            f"Tail 候选(prob>{cfg.THRESHOLD_HIGH_CONF}): {(df_res_tail[prob_col] > cfg.THRESHOLD_HIGH_CONF).sum()} 颗"
        )

        # ---------------------------------------------------------
        # 5. 第五阶段：核心（Core）与潮汐尾（Tail）分层决策融合 (Hierarchical Union)
        # ---------------------------------------------------------
        m_thresh = cfg.THRESHOLD_HIGH_CONF  # 成员判定统一使用 0.5 门限

        df_res_final = df_res_core.copy()
        df_res_final["core_prob"] = df_res_core[prob_col].values
        df_res_final["tail_prob"] = 0.0

        # 仅更新处于 Spatial Tube 管内天体的 tail_prob
        tail_prob_map = df_res_tail.set_index("id")[prob_col]
        tube_ids_set = set(df_tube_full["id"].values)
        is_in_tube_mask = df_res_final["id"].isin(tube_ids_set)

        df_res_final.loc[is_in_tube_mask, "tail_prob"] = (
            df_res_final.loc[is_in_tube_mask, "id"]
            .map(tail_prob_map)
            .fillna(0.0)
            .values
        )

        # 🌟 核心分层互斥逻辑：
        # 1. Channel A 优先：只要 core_prob >= 0.5，继承 core_prob，标为 'core'
        # 2. Channel B 接管：仅当 (core_prob < 0.5) 且 (在管内) 且 (tail_prob >= 0.5)，继承 tail_prob，标为 'tail'
        # 3. 双方均符合：若 core_prob >= 0.5 且 tail_prob >= 0.5，标为 'both'，概率继承 core_prob

        is_core_member = df_res_final["core_prob"] >= m_thresh
        is_tail_member = df_res_final["tail_prob"] >= m_thresh

        # 默认使用 Core 概率作为主概率
        df_res_final[prob_col] = df_res_final["core_prob"].values

        # 触发 Channel B 救回接管（仅对 core 被拒绝且 tail 认可的天体覆盖概率）
        tail_takeover_mask = (~is_core_member) & is_in_tube_mask & is_tail_member
        df_res_final.loc[tail_takeover_mask, prob_col] = df_res_final.loc[
            tail_takeover_mask, "tail_prob"
        ].values

        # 标记成员来源 metadata
        df_res_final["source"] = "field"
        df_res_final.loc[is_core_member & (~is_tail_member), "source"] = "core"
        df_res_final.loc[tail_takeover_mask, "source"] = "tail"
        df_res_final.loc[is_core_member & is_tail_member, "source"] = "both"

        # 统计并打印最终结果
        n_total_candidates = (df_res_final[prob_col] >= m_thresh).sum()
        n_src_core = (df_res_final["source"] == "core").sum()
        n_src_tail = (df_res_final["source"] == "tail").sum()
        n_src_both = (df_res_final["source"] == "both").sum()

        self.logger.info(
            f"🎯 [Union Complete] 核心与潮汐尾分层融合完成！"
            f"全区高置信成员星总数 (prob >= {m_thresh}): {n_total_candidates} 颗"
        )
        self.logger.info(
            f"📋 [Source Breakdown] 来源分布: "
            f"core_only={n_src_core} | tail_only(接管救回)={n_src_tail} | both={n_src_both} | "
            f"total_member={(n_src_core + n_src_tail + n_src_both)}"
        )

        # 🌟 绘制三通道概率分布直方图
        from utils.tube import plot_prob_distributions

        plot_prob_distributions(
            df_res_final,
            cluster_id=ctx.cluster_id,
            output_dir=cfg.ANALYSIS_DIR,
        )

        return df_res_final

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    def _build_spatial_tube(
        self,
        df_all: pd.DataFrame,
        df_seeds: pd.DataFrame,
        length_deg: float = 5.0,
        width_deg: float = 1.0,
        coord_cols: list[str] = None,
        sigma_multiplier: float = 3.0,
        anisotropy_threshold: float = 0.60,
    ) -> tuple[pd.DataFrame, object]:
        """基于标准的物理正交切面投影（Tangential Projection）锁定空间主轴，并在全量天区中切割出轨道管。

        防退化逻辑：
        - 若种子星形态存在各向异性（PCA 第一主成分方差占比 >= anisotropy_threshold），使用 PCA
        空间主轴；
        - 若种子星呈圆团状/各向同性（PCA 方差占比 < anisotropy_threshold），降级使用【自行矢量 (PM
        Vector)】作为主轴，
        确保空间管方向严格贴合天体物理上的切向运动轨道。
        """
        # 🌟 1. 自动寻找/校验空间坐标列
        if coord_cols is None:
            if "l" in df_seeds.columns and "b" in df_seeds.columns:
                coord_cols = ["l", "b"]
            elif "ra" in df_seeds.columns and "dec" in df_seeds.columns:
                coord_cols = ["ra", "dec"]
            else:
                # 兼容全小写与大写
                cols_lower = {col.lower(): col for col in df_seeds.columns}
                if "l" in cols_lower and "b" in cols_lower:
                    coord_cols = [cols_lower["l"], cols_lower["b"]]
                elif "ra" in cols_lower and "dec" in cols_lower:
                    coord_cols = [cols_lower["ra"], cols_lower["dec"]]
                else:
                    raise KeyError(
                        "未在种子星 DataFrame 中找到 ['l', 'b'] 或 ['ra', 'dec'] 空间坐标列！"
                    )

        col_x, col_y = coord_cols[0], coord_cols[1]
        self.logger.info(f"🌟 [Compute] 空间坐标列: {coord_cols}")
        self.logger.info(f"🌟 [Compute] 空间主轴长度：{length_deg}°")
        self.logger.info(f"🌟 [Compute] 空间主轴宽度：{width_deg}°")

        # 🌟 2. 精确计算种子星物理质心 (x0, y0)
        # 1. 计算种子星质心

        # 2. 精确计算种子星物理质心 (x0, y0)
        y0 = float(np.mean(df_seeds[col_y]))
        x_rad = np.radians(df_seeds[col_x])
        x0_rad = np.arctan2(np.mean(np.sin(x_rad)), np.mean(np.cos(x_rad)))
        x0 = float(np.degrees(x0_rad) % 360.0)

        # 3. 转换为相对质心的正交切面平面坐标 (X_proj, Y_proj)
        dx_seeds = (df_seeds[col_x] - x0 + 180.0) % 360.0 - 180.0
        x_seeds_proj = dx_seeds * np.cos(np.radians(y0))
        y_seeds_proj = df_seeds[col_y] - y0
        coords_seeds_centered = np.column_stack((x_seeds_proj, y_seeds_proj))

        # 4. 拟合 PCA 模型并评估各向异性
        pca = PCA(n_components=2)
        pca.fit(coords_seeds_centered)
        explained_ratio = float(pca.explained_variance_ratio_[0])

        # 🌟 5. 动态探测自行列名（兼容银道 pm_l_cosb/pm_b 与赤道 pmra/pmdec，兼容大小写）
        cols_map = {col.lower(): col for col in df_seeds.columns}

        pm_x_col = next(
            (
                cols_map[k]
                for k in [
                    "pm_l_cosb",
                    "pml_cosb",
                    "pm_l",
                    "pmra",
                    "pmra_cosdec",
                    "pm_ra",
                ]
                if k in cols_map
            ),
            None,
        )
        pm_y_col = next(
            (cols_map[k] for k in ["pm_b", "pmb", "pmdec", "pm_dec"] if k in cols_map),
            None,
        )

        has_pm_cols = (pm_x_col is not None) and (pm_y_col is not None)

        # 🌟 6. 核心防退化重构：检查 Seeds 形态是否发生各向同性退化
        if explained_ratio < anisotropy_threshold and has_pm_cols:
            pm_x_mean = float(df_seeds[pm_x_col].mean())
            pm_y_mean = float(df_seeds[pm_y_col].mean())
            pm_speed = np.hypot(pm_x_mean, pm_y_mean)

            if pm_speed > 1e-3:
                # 计算自行矢量的夹角
                pm_angle = np.arctan2(pm_y_mean, pm_x_mean)

                # 强制覆盖 PCA 的旋转矩阵 components_ (Row 0 为主轴方向，Row 1 为横轴正交方向)
                pca.components_ = np.array(
                    [
                        [np.cos(pm_angle), np.sin(pm_angle)],
                        [-np.sin(pm_angle), np.cos(pm_angle)],
                    ]
                )
                self.logger.warning(
                    f"⚠️ [Tube Degeneration Guard] 种子星形态呈各向同性 (PCA 占比 {explained_ratio:.2f} < {anisotropy_threshold})，"
                    f"自动切换至【自行矢量 (PM Vector)】引导主轴！"
                    f"使用自行列: [{pm_x_col}, {pm_y_col}] | 均值: pm_x={pm_x_mean:.2f}, pm_y={pm_y_mean:.2f} "
                    f"(切向轨道角={np.degrees(pm_angle):.1f}°)"
                )
            else:
                self.logger.info(
                    f"📐 [Tube Alignment] 种子形态 PCA 方差占比: {explained_ratio:.3f} (自行速度过小，沿用 2D PCA)"
                )
        else:
            self.logger.info(
                f"📐 [Tube Alignment] 种子形态 PCA 方差占比: {explained_ratio:.3f} "
                f"({'形态各向异性良好，采用 2D PCA' if explained_ratio >= anisotropy_threshold else '缺失自行数据，沿用 2D PCA'})"
            )

        # 7. 计算自适应管宽 (取种子横轴散布的 3σ，下限 0.8°，上限 width_deg)
        coords_seeds_pca = pca.transform(coords_seeds_centered)
        seed_cross_std = float(np.std(coords_seeds_pca[:, 1]))
        actual_width = float(
            np.clip(sigma_multiplier * seed_cross_std, a_min=0.8, a_max=width_deg)
        )

        self.logger.info(
            f"📐 [Adaptive Width] 种子 PCA 横轴 σ={seed_cross_std:.3f}° | "
            f"实际生效半宽={actual_width:.3f}° (配置上限={width_deg}°)"
        )

        # 8. 全量天区正交切面投影与旋转
        dx_all = (df_all[col_x] - x0 + 180.0) % 360.0 - 180.0
        x_all_proj = dx_all * np.cos(np.radians(y0))
        y_all_proj = df_all[col_y] - y0
        coords_all_centered = np.column_stack((x_all_proj, y_all_proj))

        coords_pca = pca.transform(coords_all_centered)

        pca_long = pd.Series(coords_pca[:, 0], index=df_all.index, name="pca_long")
        pca_cross = pd.Series(coords_pca[:, 1], index=df_all.index, name="pca_cross")

        # 9. 构建切片掩模
        tube_mask = (
            (pca_long.abs() <= length_deg)
            & (pca_cross.abs() <= actual_width)
            & (~pca_long.isna())
        )

        df_tube = df_all[tube_mask].copy()
        df_tube["pca_long"] = pca_long[tube_mask]
        df_tube["pca_cross"] = pca_cross[tube_mask]

        # 10. 附带绘制与反解所需的关键元数据
        pca.cluster_center_ = np.array([x0, y0])
        pca.tube_width_ = actual_width

        self.logger.info(
            f"🔍 [Tube Internal] 质心: ({x0:.2f}°, {y0:.2f}°) | "
            f"管长半长: ±{length_deg:.1f}° | 半宽: ±{actual_width:.2f}° | "
            f"切片锁定天体: {len(df_tube)} 颗"
        )

        return df_tube, pca

    # ── 后处理 ──

    def _post_process(self, ctx: RunContext, t_main_results: str) -> dict:
        """算法后处理流水线。"""
        self.logger.info(f"📊 [Process] [{ctx.cluster_id}] 启动后处理...")
        try:
            self.db.execute(
                f"ALTER TABLE {ctx.state.master_table} "
                f"ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE"
            )
            self.db.execute(
                f"ALTER TABLE {ctx.state.master_table} "
                f"ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE"
            )

            condi_golden = f"{cfg.STD_COLS['PROB']} >= {cfg.THRESHOLD_GOLDEN}"
            condi_candidates = f"{cfg.STD_COLS['PROB']} > {cfg.THRESHOLD_BASE}"

            self.db.execute(
                f"UPDATE {ctx.state.master_table} SET is_golden = TRUE WHERE {condi_golden}"
            )
            self.db.execute(
                f"UPDATE {ctx.state.master_table} SET is_candidate = TRUE WHERE {condi_candidates}"
            )

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

            v_candidates = f"v_candidates_{ctx.cluster_id.lower()}"
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
        """生成最终报告并返回绩效摘要。"""
        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id

        return render_final_report(
            ctx.cluster_id,
            ctx.category,
            ctx.feature_space,
            ctx.algorithm,
            cluster_cfg,
            ctx.state.gmm_config,
            post_result,
            audit_result,
            audit_result.get("deep_stats_pg_only", {}),
            audit_result.get("deep_stats_ref_only", {}),
            audit_result.get("deep_stats_matched", {}),
            audit_result.get(f"deep_stats_{ctx.category}", {}),
            audit_result.get("deep_stats_pg_algo", {}),
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
