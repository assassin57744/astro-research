"""
🌌 天文数据处理工作流总编排引擎。

职责：
  作为流水线的总导演（Orchestrator），按标准阶段控制管线的生命周期：
  Stage 1    : 内存轻量分配物理领域实体 `StarCluster`。
  Stage 1.5  : 联动基础设施层 `SchemaAligner` 自动驱动数据导入（import_raw）与表结构对齐（data_standardize）。
  Stage 1.75 : 在数仓底座就绪后，触发物理实体参数的正式装载与科学反演（解决生命周期死锁）。
  Stage 2    : 从数仓大盘抽取局部天区切片，驱动 `ClusterSeedExtractor` 提炼种子星。
  Stage 3    : 激活 `AstroTransformer` 特征工程转换引擎，完成动力学相空间特征矩阵反演。
  Stage 4    : 移交专属高维精筛引擎 `PriorGMMEx` 拟合相空间特征并计算成员收敛概率。
  Stage 5    : 触发 `UnifiedMemberValidator` 联动多源文献进行自动化联合审计。
  Stage 6    : 交付 `render_final_report` 固化科学资产并输出可视化简报。
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 基础设施与核心科学组件导入
from modules.db import AstroDB, AssetManager
from modules.cluster import StarCluster
from modules.schema_aligner import SchemaAligner
from modules.cluster_seed_extractor import ClusterSeedExtractor
from modules.transformer import AstroTransformer
from modules.pg_core_ex import PriorGMMEx  # 🔬 独占使用的先进高维精筛核心
from modules.validator import UnifiedMemberValidator
from modules.reporter import render_final_report

import config as cfg
from config import GMM_CONFIG, CLUSTERS, MANIFEST, ClusterConfig, CatalogConfig


class AstroWorkflow:
    """天文数据科学管线总编排长。"""

    def __init__(self, db: AstroDB):
        self.db = db
        self.logger = logging.getLogger("AstroPipeline.Workflow")
        # 资产管理和全局清单内聚收拢至工作流内部，生命周期与计算完全同步
        self.asset_mgr = AssetManager()
        self.manifest = getattr(cfg, "MANIFEST", {})

    def execute(
        self,
        cluster_id: str,
        category: str,
        mode: str,
        algo: str,
        result_mode: str,
        reconstruct: str,
    ) -> str:
        """
        执行端到端的星团审计流水线核心大纲。
        """
        safe_cluster_id = cluster_id.strip()
        self.logger.info(f"🎬 开始编排星团审计管线作业 | 目标天体: '{safe_cluster_id}'")

        # Stage 1: 物理领域实体【轻量化激活】（此时不触发数仓物理查询，仅建立内存空间与命名契约基底）
        cluster = self._stage_allocate_cluster(safe_cluster_id)
        if not cluster:
            return ""

        # Stage 1.5: 数据治理（借由内存实体提供的 id_name 契约，安全安全执行原始数据导入与结构标准化）
        self._stage_import_and_align_schemas(safe_cluster_id, cluster)

        # Stage 1.75: 🔄【两段式充水解耦】数仓原始资产落地后，正式触发星团先验参数的加载与科学反演
        if not self._stage_hydrate_cluster_parameters(cluster, reconstruct):
            return ""

        # Stage 2: 捕获局部大盘数据，驱动无监督粗筛提炼种子星
        df_field_stars, df_seeds = self._stage_extract_seeds(
            cluster_id=cluster_id, algo=algo, reconstruct_mode=reconstruct
        )
        if df_seeds.empty:
            return ""

        # Stage 3 & 4: 动力学相空间特征工程反演与精筛 PriorGMMEx 模型收敛
        df_prob_catalog = self._stage_precision_screening(
            df_seeds=df_seeds,              # 🚀 经过 Stage 2 洗干净的高质量种子星（几百颗）
            df_field_stars=df_field_stars,  # 🚀 从最初数仓大盘读取出来的全域切片（几万颗，对应 m45_field 视图）
            cluster=cluster,
            mode=mode
        )

        # Stage 5: 多源文献自动化联合交叉科学审计
        audit_results = self._stage_cross_validate(
            df_prob_catalog, safe_cluster_id, category
        )

        # Stage 6: 自动化科学成果资产归档、落盘与简报渲染
        return self._stage_finalize_and_report(
            safe_cluster_id, category, mode, algo, audit_results, result_mode
        )

    # =====================================================================
    # 🛠️ 下沉的私有阶段过滤器 (Stage Filters)
    # =====================================================================

    def _stage_allocate_cluster(self, cluster_name: str) -> StarCluster:
        """[Stage 1] 轻量分配实体。仅在内存中建立模型，隔离未初始化的数仓物理查询。"""
        self.logger.info("🎬 [Stage 1] 正在轻量级激活星团物理领域模型实体契约...")
        return StarCluster(cluster_name, db_instance=self.db)

    #  优雅强类型写法

    def _stage_import_and_align_schemas(
        self, cluster_name: str, cluster: StarCluster
    ) -> None:
        """
        [Stage 1.5] 级联驱动多源异构星表的原始数据导入(import_raw)与表结构对齐(data_standardize)。
        """
        self.logger.info(
            "🎬 [Stage 1.5] 启动数据治理子管线：开始级联驱动数据导入与结构标准化..."
        )

        # 组装用于星表命名的局部物理上下文，全程确保外部输入的字面量纯净
        # ctx = {
        #     "id": cluster_name,
        #     "ID_NAME": cluster.id_name if hasattr(cluster, "id_name") else cluster_name,
        #     "NAME": cluster_name
        # }
        # 1. 统一从强类型数据库中捞取完整的星团先验物理参数实体
        # 此时 ctx 是一个完备的 ClusterTargetConfig 对象，自带所有星团核心物理常数

        self.logger.info(f"cluster.id : {cluster.id}")
        ctx = CLUSTERS.get(cluster.id.upper())

        if not ctx:
            raise ValueError(
                f"星团 {cluster.id} 未在 config.py 的 CLUSTERS 先验数据库中注册！"
            )

        # 1. 一次性驱动全部数据源的原始物理导入（新 import_raw 内部自行迭代 manifest 并做索引过滤）
        self.db.import_raw(target_cluster_id=cluster_name)

        # 2. 逐数据源级联执行结构标准化 (std -> stx -> aln)
        aligner = SchemaAligner(db=self.db, manifest=self.manifest)

        # 归一化当前处理的星团前缀（例如 "M45" -> "m45_"）
        cluster_prefix = f"{cluster_name.lower()}_"

        for idx_src, cfg_src in cfg.MANIFEST.items():
            # 判断当前资产是否属于专用的 field 或 seeds 类型
            if cfg_src.meta_type in ("field", "seeds"):
                # 如果是 field 或 seeds，但其 id 不是以当前星团名开头，则直接剔除
                if not cfg_src.id.startswith(cluster_prefix):
                    self.logger.info(
                        f"⏭️  [跳过外部资产] 数据源 {idx_src} 不属于当前星团 {cluster_name}。"
                    )
                    continue

            # 如果资产属于全局公共参考、索引或交叉证认表 (如 reference 类型)
            elif cfg_src.meta_type == "reference":
                # 全局参考资产不需要用 构建三重视图,跳过
                self.logger.info(
                    f"⏭️  [跳过外部资产] 数据源 {idx_src} 不属于当前星团 {cluster_name}。"
                )
                continue

            else:
                # 预留给未来可能出现的其他 meta_type
                self.logger.debug(f"ℹ️  [通用资产] 放行标准管道: {idx_src}")

            self.logger.info(f"🔧 正在标准化数据源: {idx_src}")
            aligner.data_standardize(data_idx=idx_src, data_cfg=cfg_src, cl_cfg=ctx)

    def _stage_hydrate_cluster_parameters(
        self, cluster: StarCluster, reconstruct: str
    ) -> bool:
        """[Stage 1.75] 参数实体充水。在数据就绪的底座上，正式驱动物理参数加载与科学反演。"""
        self.logger.info(
            f"🎬 [Stage 1.75] 数仓底座已安全就绪，触发物理实体参数反演装载..."
        )

        if hasattr(cluster, "load_or_reconstruct_parameters"):
            is_loaded = cluster.load_or_reconstruct_parameters(mode=reconstruct)
        else:
            is_loaded = True if cluster.plx_ref is not None else False

        if not is_loaded:
            self.logger.error(
                f"❌ 无法从最新初始化的数仓资产中反演星团 '{cluster.name}' 的先验物理基底，管线强行熔断。"
            )
            return False
        return True

    def _stage_extract_seeds(
        self, cluster_id: str, algo: str, reconstruct_mode: str = "dynamic"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        [Stage 2] 数仓解耦抽取与无监督粗筛提纯。
        
        分别独立抽取纯净的全域大背景场(field)与候选种子场(seeds_field)，
        并对候选种子场驱动自适应密度提纯，洗除背景噪点以萃取高纯度种子星。
        """
        self.logger.info(
            f"🗄️  正在从数仓大盘抽取 '{cluster_id}' 候选种子场的恒星数据... [重建模式: {reconstruct_mode}]"
        )

        # 1. 🚀【元数据定位】拼接出当前星团在 Manifest 容器中注册的种子场属性名
        cl_lower = cluster_id.lower()
        self.logger.info(f"🎬 [Stage 2] 启动数仓资产独立拉取与粗筛提纯流... 核心内核: '{algo}'")

        # ----------------------------------------------------------------------
        # 🚀 轨一：独立、干净地抽取全域视场大盘数据 (Field)
        # ----------------------------------------------------------------------
        target_field_view = f"aln_{cl_lower}_field"
        self.logger.info(f"🗄️  [数仓底盘拉取] 正在从隔离视图 [{target_field_view}] 独立拉取纯净全域背景星...")
        
        sql_field = f"SELECT id, ra, dec, plx, pmra, pmdec, rv, mag, color FROM {target_field_view};"
        df_field_stars = self.db.execute(sql_field).df()

        if df_field_stars.empty:
            self.logger.error(f"❌ 数仓未命中该天区全域背景大盘数据（视图: {target_field_view}），管线中止。")
            return pd.DataFrame(), pd.DataFrame()

        # ----------------------------------------------------------------------
        # 🚀 轨二：独立抽取候选种子场数据 (Seeds Field) 并驱动提纯
        # ----------------------------------------------------------------------
        target_seeds_view = f"aln_{cl_lower}_seeds_field"
        self.logger.info(f"🗄️  [数仓底盘拉取] 正在从隔离视图 [{target_seeds_view}] 抽取候选种子恒星数据进行提纯...")
        
        sql_seeds_field = f"SELECT id, ra, dec, plx, pmra, pmdec, rv, mag, color FROM {target_seeds_view};"
        df_seeds_field_raw = self.db.execute(sql_seeds_field).df()

        if df_seeds_field_raw.empty:
            self.logger.warning(f"⚠️ 数仓未命中该天区候选种子场数据（视图: {target_seeds_view}），降级使用空表。")
            return df_field_stars, pd.DataFrame()

        # ----------------------------------------------------------------------
        # 🧠 驱动无监督聚类算子：对候选种子场计算最优 eps，洗除本征噪声
        # ----------------------------------------------------------------------
        seed_extractor = ClusterSeedExtractor(min_pts=9 * 2, num_simulations=30)

        optimal_eps = None
        if algo.strip().lower() == "dbscan":
            mode_str = reconstruct_mode.strip().lower()
            if mode_str == "dynamic":
                self.logger.info("🧠 触发 [Dynamic] 模式：对候选种子场启动高维空间自适应最近邻距离密度探测...")
                optimal_eps = seed_extractor.calculate_adaptive_eps(df_seeds_field_raw)
            elif mode_str == "static":
                from config import CLUSTERS
                ctx = CLUSTERS.get(cluster_id.upper())
                static_eps = getattr(ctx, "EPS", 0.3) if ctx else 0.3
                self.logger.info(f"🛑 触发 [Static] 硬截断模式：直接应用物理先验值 EPS = {static_eps}")
                optimal_eps = static_eps
            else:
                optimal_eps = seed_extractor.calculate_adaptive_eps(df_seeds_field_raw)

        # 执行提纯（在 5545 颗候选集上榨取最高质量的核心种子星，洗掉散逸和背景噪点）
        df_seeds = seed_extractor.extract_seeds(
            df_seeds_field_raw, algo=algo, eps=optimal_eps
        )

        self.logger.info(
            f"✅ [Stage 2 提纯结束] 规格清单 -> 独立全域背景场: {len(df_field_stars)} 颗 | 洗净核心种子星: {len(df_seeds)} 颗"
        )

        return df_field_stars, df_seeds


    def _stage_precision_screening(
        self,
        df_seeds: pd.DataFrame,
        df_field_stars: pd.DataFrame,
        cluster: Any,
        mode: str,
        category: str = "hunt",  # 🎯 核心找回：接受来自命令行 --category 的靶向审计星表控制
    ) -> pd.DataFrame:
        """
        [Stage 3 & 4] 特征空间动力学反演与精筛 PriorGMMEx 模型收敛（靶向文献审计版）。
        
        功能找回：
          1. 穿透读取历史文献视图 aln_{category}_{cluster}，通过高维碰撞解算物理先验权重 f_init。
          2. 将真实观测先验比率 f_init 逆向解算并注入推理引擎，消除盲目随机抽样导致的概率假死。
          3. 自动化向 DuckDB 动态注册中间审计输入大盘。
        """
        cluster_id = cluster.id if hasattr(cluster, "id") else str(cluster)
        cl_upper = cluster_id.upper()
        cl_lower = cluster_id.lower()
        cat_lower = category.strip().lower()
        
        self.logger.info(
            f"🎬 [Stage 3] 启动相空间反演管线。目标: {cl_upper} | 维度: {mode} | 靶向审计对象: {cat_lower}"
        )

        # ----------------------------------------------------------------------
        # 1. ⚙️【大盘物理上下文对齐】检索被审计星团的权威核心常数
        # ----------------------------------------------------------------------
        from config import CLUSTERS
        ctx = CLUSTERS.get(cl_upper)

        cluster_rv = None
        if ctx:
            for rv_key in ["RV", "rv", "RV_REF", "rv_ref"]:
                if hasattr(ctx, rv_key): cluster_rv = getattr(ctx, rv_key); break

        cluster_center = (None, None)
        if ctx:
            ra = getattr(ctx, "CENTER_RA", getattr(ctx, "ra", None))
            dec = getattr(ctx, "CENTER_DEC", getattr(ctx, "dec", None))
            if ra is not None and dec is not None: cluster_center = (ra, dec)

        # 各向异性物理过滤半径（默认取配置中的核心半径，保底取2.5度）
        spatial_radius = getattr(ctx, "RADIUS", getattr(ctx, "radius", 2.5))

        # ----------------------------------------------------------------------
        # 2. 🌌【空间裁剪】过滤极端远景噪声，保护模型协方差矩阵不被无关背景拉偏
        # ----------------------------------------------------------------------
        if cluster_center != (None, None):
            c_ra, c_dec = cluster_center
            self.logger.info(f"🎯 [空间核密度过滤] 基于物理半径 {spatial_radius}° 执行天区大盘截断...")
            cos_dec = np.cos(np.radians(c_dec))
            delta_ra = (df_field_stars["ra"] - c_ra) * cos_dec
            delta_dec = df_field_stars["dec"] - c_dec
            distances = np.sqrt(delta_ra**2 + delta_dec**2)
            
            df_field_filtered = df_field_stars[distances <= spatial_radius].copy()
            self.logger.info(f"✂️ 全域天区样本由 {len(df_field_stars)} 颗收拢至核心物理天区 {len(df_field_filtered)} 颗。")
        else:
            df_field_filtered = df_field_stars.copy()

        # ----------------------------------------------------------------------
        # 3. 🛡️【坐标变换】驱动特征工程转换引擎
        # ----------------------------------------------------------------------
        physics_transformer = AstroTransformer(cluster_rv=cluster_rv, cluster_center_icrs=cluster_center)
        df_seeds_ext = physics_transformer.fit_transform_df(df_seeds, mode=mode)
        df_field_ext = physics_transformer.fit_transform_df(df_field_filtered, mode=mode)

        feature_map = GMM_CONFIG.get("feature_map", {})
        required_features = feature_map.get(mode, ["pmra", "pmdec", "plx"])

        df_field_final = self._defensive_nan_purge(df_field_ext, required_features, label="Target_field")
        df_seeds_final = self._defensive_nan_purge(df_seeds_ext, required_features, label="Seeds")

        # ----------------------------------------------------------------------
        # 4. 📚【核心目标找回：文献审计初值估算】精准碰撞 --category 指定的视图
        # ----------------------------------------------------------------------
        # 严格按照规范拼接靶向历史文献视图名，例如：aln_cg20_m45 或 aln_hunt_m45
        target_ref_view = f"aln_{cat_lower}_{cl_lower}"
        self.logger.info(f"📚 [先验解算] 正在穿透提取历史文献视图 [{target_ref_view}] 的数据 footprint...")
        
        f_init = None
        try:
            # 独立抽取出该被审计历史星表在数仓中的所有 id 资产
            ref_df = self.db.execute(f"SELECT id FROM {target_ref_view};").df()
            
            if not ref_df.empty:
                ref_ids_set = set(ref_df["id"].to_numpy())
                field_ids_set = set(df_field_final["id"].to_numpy())
                
                # 计算当前空间大盘中命中该历史星表的交集数（即该文献在此处的实际成员星基数）
                matches = len(ref_ids_set.intersection(field_ids_set))
                
                # 物理先验分布比例 = 历史文献命中数 / 当前天区净化后的大盘总数
                f_init = matches / len(df_field_final)
                
                # 设立防护自适应边界，防止历史文献数据为空或极度泛滥时导致 GMM 溃缩
                f_init = max(min(f_init, 0.30), 0.0005)
                self.logger.info(
                    f"✨ [审计对齐成功] 历史文献在当前核心大盘中共命中 {matches} 颗星。"
                    f"解算出初始物理混合权重先验 f_init = {f_init:.6f}"
                )
            else:
                self.logger.warning(f"⚠️ 历史文献视图 [{target_ref_view}] 数据为空，无法提取物理初值。")
        except Exception as err:
            self.logger.error(f"❌ 穿透历史文献视图 [{target_ref_view}] 发生异常 (回退至基础采样比例): {err}")

        # ----------------------------------------------------------------------
        # 5. 🏋️‍♂️【模型训练与参数注入】实例化专属精筛引擎 PriorGMMEx
        # ----------------------------------------------------------------------
        self.logger.info("🎬 [Stage 4] 实例化专属精筛引擎 PriorGMMEx 并注入物理超参...")
        engine = PriorGMMEx(config=GMM_CONFIG)
        if hasattr(engine, "set_dim_mode"):
            engine.set_dim_mode(mode)

        enable_sub = GMM_CONFIG.get("enable_subsampling", False)
        sub_limit = GMM_CONFIG.get("subsampling_limit", 500000)

        df_field_for_fit = df_field_final
        if enable_sub and len(df_field_final) > sub_limit:
            self.logger.info(f"🚀 [性能优化] 过滤后样本数过大, 下采样至 {sub_limit} 用于拟合...")
            df_field_for_fit = df_field_final.sample(n=sub_limit, random_state=42)

        # 驱动模型训练
        gmm_params = engine.fit(df_seeds_final, df_field_for_fit)

        # ----------------------------------------------------------------------
        # 6. 🔮【全域大盘推断与血缘缝合】解算完备大盘的成员概率
        # ----------------------------------------------------------------------
        # 💥 关键咬合：将我们通过 --category 靶向解算出来的 f_init 物理权重注入模型参数
        if f_init is not None and hasattr(gmm_params, "n_core_samples"):
            # 逆向修正内核所需的有效核心样本等效数，迫使推理流的初始 f 完全对齐物理观测值
            gmm_params.n_core_samples = int(f_init * len(df_field_final))
            self.logger.info(f"🧠 [物理内核修正] 成功将对齐文献历史分布的 f_init={f_init:.6f} 注入推理机。")

        self.logger.info(f"🔮 [全域大盘推断] 正在将收敛模型逆向应用至当前空间内的 {len(df_field_final)} 颗恒星...")
        df_prob_only = engine.predict(df_field_final, params=gmm_params)

        self.logger.info("🧩 正在合并成员星概率至天体物理属性大盘...")
        df_field_predicted = df_field_final.merge(df_prob_only, on="id", how="left")
        df_field_predicted["prob"] = df_field_predicted["prob"].fillna(0.0)

        # ----------------------------------------------------------------------
        # 7. 🗄️【注册全局内存审计视图】无缝打通下游 Validator 审计管线
        # ----------------------------------------------------------------------
        try:
            audit_view_name = f"v_audit_input_{cl_upper}"
            self.logger.info(f"🗄️  [数仓注册] 正在将精筛推断产物注册为全局内存审计视图: [{audit_view_name}]")
            self.db.register_dataframe(audit_view_name, df_field_predicted)
            self.logger.info(f"🎉 视图 [{audit_view_name}] 注册成功！")
        except Exception as e:
            self.logger.error(f"❌ 注册内存审计视图失败: {e}")

        return df_field_predicted

    def _stage_cross_validate(
        self, df_prob_catalog: pd.DataFrame, cluster_name: str, category: str
    ) -> dict:
        """[Stage 5] 多源文献自动化联合交叉科学审计。"""
        self.logger.info(
            "🎬 [Stage 5] 统计收敛概率计算完毕，触发多源文献自动化联合审计..."
        )

        # 将 GMM 概率目录注册为临时视图供审计引擎消费
        v_audit_input = f"v_audit_input_{cluster_name}"
        self.db.register_view_from_df(v_audit_input, df_prob_catalog)

        validator = UnifiedMemberValidator(cluster_id=cluster_name, db_instance=self.db)
        df_audited = validator.run_full_audit_ex(v_target_detail=v_audit_input)

        # 从审计结果 DataFrame 提取统计摘要
        stats = (
            df_audited["audit_status"].value_counts().to_dict()
            if not df_audited.empty
            else {}
        )
        return {"df": df_audited, "stats": stats}

    def _stage_finalize_and_report(
        self,
        cluster_name: str,
        category: str,
        mode: str,
        algo: str,
        audit_results: dict,
        result_mode: str,
    ) -> str:
        """[Stage 6] 自动化科学成果资产归档、落盘与简报渲染。"""
        self.logger.info(
            "🎬 [Stage 6] 科学解算流程闭环，正在同步数仓资产并固化量化分析简报..."
        )

        # 从 CLUSTERS 获取 ctx_cluster
        ctx_cluster = (
            cfg.CLUSTERS.get(cluster_name)
            if hasattr(cfg.CLUSTERS, "get")
            else cfg.CLUSTERS[cluster_name]
        )

        # 从 audit_results 中拆分各阶段统计
        audit_stats = audit_results.get("stats", {})
        df_audited = audit_results.get("df", pd.DataFrame())

        # v_all_audit_data: GMM 发现阶段统计占位
        v_all_audit_data: dict = {"stats": {}}

        # audit_res: 交叉比对统计占位
        audit_res: dict = {"stats": {}}

        # 深度审计统计：按 audit_status 分组
        deep_stats_pg: dict = {}
        deep_stats_ref: dict = {}

        if not df_audited.empty and "audit_status" in df_audited.columns:
            deep_stats_all = df_audited["audit_status"].value_counts().to_dict()
            # 将全量审计统计同时填入 pg 和 ref（后续可按需拆分）
            deep_stats_pg = deep_stats_all
            deep_stats_ref = {}

        summary = render_final_report(
            target_cluster_id=cluster_name,
            target_category=category,
            mode=mode,
            algo=algo,
            ctx_cluster=ctx_cluster,
            v_all_audit_data=v_all_audit_data,
            audit_res=audit_res,
            deep_stats_pg=deep_stats_pg,
            deep_stats_ref=deep_stats_ref,
            logger=self.logger,
        )

        try:
            from modules.backup_manager import AstroBackupManager

            backup_mgr = AstroBackupManager(self.db)
            backup_mgr.execute_cluster_snapshot(cluster_id=cluster_name)
        except Exception as backup_err:
            self.logger.warning(
                f"⚠️ 常规流水线成功执行，但 Stage 6 触发自动科学中间快照时挂起: {backup_err}"
            )

        return summary

    def _defensive_nan_purge(
        self, df: pd.DataFrame, features: list, label: str = "Dataset"
    ) -> pd.DataFrame:
        """🛡️ 针对天体测量缺失值的防御性净化工具方法。"""
        if df.empty:
            return df
        before_len = len(df)
        valid_mask = df[features].notna().all(axis=1)
        df_clean = df[valid_mask].copy()
        after_len = len(df_clean)

        if before_len != after_len:
            self.logger.warning(
                f"⚠️ [{label}] 净化拦截：剔除了 {before_len - after_len} 行包含 NaN 核心物理特征的破损恒星记录。"
            )
        return df_clean
