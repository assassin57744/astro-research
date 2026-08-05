# modules/pipelines/experimental_pipeline.py
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import config as cfg

class ExperimentalPipelineRunner:
    """
    双通道分层捕捉实验管线执行器 (Dual-Channel Experimental Pipeline)
    
    采用三层架构设计：
    1. 准备层 (Preparation Layer)
    2. 执行层 (Execution Layer: Channel A & Channel B)
    3. 决策融合层 (Decision & Fusion Layer)
    """

    def __init__(self, db_instance, logger: logging.Logger = None):
        self.db = db_instance
        self.logger = logger or logging.getLogger(f"AstroPipeline.{self.__class__.__name__}")

    def run(self, ctx, df_all_field: pd.DataFrame, df_seed_field: pd.DataFrame) -> pd.DataFrame:
        """主入口：执行 Core + Tail 双通道提取与融合。"""
        strategy_name = ctx.algo_params.get("strategy", "bayesian")
        self.logger.info(f"🚀 [Compute] 启动双通道分层捕捉管线 | 策略: [{strategy_name.upper()}]")
        self.logger.info(
            f"🔍 [Target Field] 传入天体总量: {len(df_all_field)} 颗 | "
            f"RA: [{df_all_field['ra'].min():.2f}°, {df_all_field['ra'].max():.2f}°] | "
            f"Dec: [{df_all_field['dec'].min():.2f}°, {df_all_field['dec'].max():.2f}°]"
        )

        # ---------------------------------------------------------
        # 1. 准备层 (Preparation Layer)
        # ---------------------------------------------------------
        df_core_seeds = self._extract_refined_seeds(ctx, df_seed_field)
        engine = self._create_disambiguation_engine(ctx, strategy_name)

        # ---------------------------------------------------------
        # 2. 执行层：Channel A (Core 核心成员提取)
        # ---------------------------------------------------------
        self.logger.info("🎯 [Channel A] 启动星团 Core 核心区域抓取...")
        df_res_core = engine.fit_predict(df_all_field, df_core_seeds, ctx.state.required_features)

        # ---------------------------------------------------------
        # 3. 执行层：Channel B (Spatial Tube 潮汐尾提取 & 扩展种子策略)
        # ---------------------------------------------------------
        df_res_tail, df_tube_full = self._run_channel_b_tail(
            ctx, engine, df_all_field, df_core_seeds, df_res_core
        )

        # ---------------------------------------------------------
        # 4. 决策融合层 (Fusion & Decision Layer)
        # ---------------------------------------------------------
        return self._fuse_core_and_tail_channels(df_res_core, df_res_tail, df_tube_full)

    # =========================================================================
    # 🧩 内部 Helper 模块
    # =========================================================================

    def _extract_refined_seeds(self, ctx, df_seed_field: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("🧬 [Compute] 正在调度 ClusterSeedExtractor...")
        from modules.seed_extractor import ClusterSeedExtractor

        cluster_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()].copy()
        cluster_cfg["id"] = ctx.cluster_id

        extractor = ClusterSeedExtractor(cluster_profile=cluster_cfg)
        df_core_seeds = extractor.extract_seeds(
            df_field=df_seed_field,
            features=ctx.state.required_features,
        )

        if df_core_seeds is None or df_core_seeds.empty:
            raise ValueError("❌ [Compute] ClusterSeedExtractor 未能凝聚出有效种子星！")

        ctx.state.seed_stats["refined_count"] = len(df_core_seeds)
        if extractor.computed_eps is not None:
            ctx.state.computed_dbscan_eps = f"{extractor.computed_eps:.4f}"

        self.logger.info(f"✅ [Compute] Core 种子星精炼成功！共 {len(df_core_seeds)} 颗。")

        # ① 标记种子类型
        df_tag_refined = df_core_seeds[[cfg.STD_COLS["ID"]]].copy()
        df_tag_refined["seed_type"] = "refined_seed"
        self.db.tag_master_table(ctx.state.master_table, df_tag_refined)

        # ② 回灌无监督聚类标签（全体输入恒星的 DBSCAN/HDBSCAN 标签）
        if extractor.df_labeled_ is not None:
            df_label_tag = extractor.df_labeled_[[cfg.STD_COLS["ID"], "cluster_label"]].copy()
            df_label_tag = df_label_tag.rename(columns={"cluster_label": cfg.MASTER_COLS["SEED_CLUSTER_LABEL"]})
            df_label_tag[cfg.MASTER_COLS["SEED_CLUSTER_LABEL"]] = df_label_tag[cfg.MASTER_COLS["SEED_CLUSTER_LABEL"]].astype(str)
            self.db.tag_master_table(ctx.state.master_table, df_label_tag)
            self.logger.info(
                f"📋 [Master] 已回灌 {len(df_label_tag)} 条聚类标签至 seed_cluster_label 列 "
                f"(簇数: {extractor.df_labeled_['cluster_label'].nunique()})"
            )

        return df_core_seeds

    def _create_disambiguation_engine(self, ctx, strategy_name: str):
        from modules.pipelines.core.disambiguation.bayesian import BayesianGmmDisambiguation
        from modules.pipelines.core.disambiguation.blind import BlindGmmDisambiguation
        from modules.pipelines.core.disambiguation.threshold import ThresholdGmmDisambiguation

        STRATEGY_CLASSES = {
            "bayesian": BayesianGmmDisambiguation,
            "threshold": ThresholdGmmDisambiguation,
            "blind": BlindGmmDisambiguation,
        }

        if strategy_name not in STRATEGY_CLASSES:
            raise ValueError(f"未知的策略类型 [{strategy_name}]")

        cl_cfg = cfg.CLUSTERS[ctx.cluster_id.upper()]
        gmm_cfg = ctx.state.gmm_config
        strategy_params = cl_cfg.get("STRATEGY_PARAMS", {}).get(strategy_name, {})
        strategy_kwargs = {**strategy_params}
        strategy_kwargs.setdefault("spatial_cols", ["l", "b"])
        strategy_kwargs.setdefault("scale_col", "plx")

        for key in ("enable_subsampling", "subsampling_limit", "bayesian_ema_rtol"):
            strategy_kwargs.setdefault(key, cl_cfg.get(key, gmm_cfg.get(key)))

        for key in ("eps", "min_samples", "sigma_cutoff"):
            if key in ctx.algo_params:
                strategy_kwargs[key] = ctx.algo_params[key]
                self.logger.info(f"✅ [Compute] 已覆盖设置策略参数 [{key}]")

        engine_class = STRATEGY_CLASSES[strategy_name]
        self.logger.info(f"🧠 [Compute] 路由至消歧拟合引擎 [{engine_class.__name__}]")
        return engine_class(**strategy_kwargs)

    def _run_channel_b_tail(self, ctx, engine, df_all_field, df_core_seeds, df_res_core):
        self.logger.info("📐 [Channel B] 启动 Tidal Tail 空间管构建与抓取...")
        cl = ctx.star_cluster

        # B1. 空间管几何切割
        length_deg, width_deg = self._compute_adaptive_tube_dimensions(cl)
        tube_cols = list(dict.fromkeys([*ctx.state.required_features, "ra", "dec"]))
        tube_cols = [c for c in tube_cols if c in df_all_field.columns]

        df_tube_features, pca_model = self._build_spatial_tube(
            df_all_field[tube_cols], df_core_seeds[tube_cols], length_deg, width_deg
        )
        self._update_tube_pca_state(ctx, pca_model, length_deg)

        # 动力学 Mahalanobis 剪裁
        df_tube_features = self._apply_kinematic_sigma_clip(
            ctx, cl, df_all_field, df_core_seeds, df_res_core, df_tube_features
        )

        valid_indices = df_all_field.index.intersection(df_tube_features.index)
        df_tube_full = df_all_field.loc[valid_indices].copy()
        for col in ("pca_long", "pca_cross"):
            if col in df_tube_features.columns:
                df_tube_full[col] = df_tube_features.loc[valid_indices, col]

        df_tube_tag = df_tube_full[[cfg.STD_COLS["ID"], "pca_long", "pca_cross"]].copy()
        df_tube_tag["in_tube"] = True
        self.db.tag_master_table(ctx.state.master_table, df_tube_tag)

        if df_tube_full.empty:
            self.logger.warning("⚠️ [Channel B] 空间管内无有效天体，跳过 Tail 拟合。")
            return pd.DataFrame(), df_tube_full

        # B2. Tail 种子生成策略 (具化解耦)
        tail_seed_strategy = ctx.algo_params.get("tail_seed_strategy", "de_coring")
        df_tail_seeds = self._extract_tail_seeds_by_strategy(
            tail_seed_strategy, df_tube_features, df_all_field, df_core_seeds, df_res_core, ctx
        )

        # B3. Channel B 拟合与预测
        tail_params = engine.fit(
            df_field=df_tube_full,
            df_seeds=df_tail_seeds,
            features=ctx.state.required_features,
            use_density_prune=False,
        )
        df_res_tail = engine.predict(
            df_all=df_tube_full,
            model_params=tail_params,
            features=ctx.state.required_features,
        )

        return df_res_tail, df_tube_full

    def _extract_tail_seeds_by_strategy(self, strategy_name, df_tube_features, df_all_field, df_core_seeds, df_res_core, ctx):
        """Tail 种子策略分发路由器"""
        prob_col = cfg.STD_COLS["PROB"]
        if strategy_name == "de_coring":
            core_prob_map = df_res_core.set_index("id")[prob_col]
            tube_ids_from_idx = df_all_field.loc[df_tube_features.index, "id"]
            tube_core_probs = tube_ids_from_idx.map(core_prob_map)

            tail_seed_mask = tube_core_probs < cfg.THRESHOLD_BASE
            df_tail_seeds = df_tube_features[tail_seed_mask].copy()

            n_removed = len(df_tube_features) - len(df_tail_seeds)
            self.logger.info(
                f"📐 [Tail Seed Strategy: De-Coring] 管内源: {len(df_tube_features)} 颗 → "
                f"提取 Tail 种子: {len(df_tail_seeds)} 颗 (剔除 Core 成员 {n_removed} 颗)"
            )
            return df_tail_seeds
        elif strategy_name == "tube_density":
            raise NotImplementedError("策略 'tube_density' 正在开发中...")
        else:
            raise ValueError(f"未知的 Tail 种子策略: [{strategy_name}]")

    def _compute_adaptive_tube_dimensions(self, cl):
        tidal_radius_pc = cl.get_param("TIDAL_RADIUS", None)
        distance_pc = cl.get_param("DISTANCE_PC", None)

        if tidal_radius_pc and distance_pc and distance_pc > 0:
            angular_tidal = np.degrees(tidal_radius_pc / distance_pc)
            length_mult = cl.get_param("TUBE_LENGTH_MULTIPLIER", 1.6)
            width_mult = cl.get_param("TUBE_WIDTH_MULTIPLIER", 0.3)

            raw_length_deg = angular_tidal * length_mult
            max_half_length = cl.get_param("MAX_TUBE_HALF_LENGTH", 15.0)
            length_deg = min(raw_length_deg, max_half_length)
            width_deg = max(1.0, angular_tidal * width_mult)
        else:
            length_deg = cl.get_param("TUBE_LENGTH", 5.0)
            width_deg = cl.get_param("TUBE_WIDTH", 1.0)

        return length_deg, width_deg

    def _apply_kinematic_sigma_clip(self, ctx, cl, df_all_field, df_core_seeds, df_res_core, df_tube_features):
        sigma_clip = cl.get_param("TUBE_SIGMA_CLIP", 5.0)
        if sigma_clip <= 0 or df_tube_features.empty:
            return df_tube_features

        pm_cols_map = {c.lower(): c for c in df_all_field.columns}
        x_col = next((pm_cols_map[k] for k in ["pm_l_cosb", "pmra"] if k in pm_cols_map), None)
        y_col = next((pm_cols_map[k] for k in ["pm_b", "pmdec"] if k in pm_cols_map), None)
        pm_cols = [x_col, y_col, "plx"] if x_col and y_col else ["plx"]

        prob_col = cfg.STD_COLS["PROB"]
        core_high_conf_mask = df_res_core[prob_col] >= cfg.THRESHOLD_HIGH_CONF
        pure_core_ids = set(df_res_core.loc[core_high_conf_mask, "id"].values)

        if len(pure_core_ids) >= 20:
            core_mask = df_all_field["id"].isin(pure_core_ids)
            ref_df = df_all_field.loc[core_mask, ctx.state.required_features]
            ref_label = f"Pure Core (P >= {cfg.THRESHOLD_HIGH_CONF})"
        else:
            ref_df = df_core_seeds
            ref_label = "Refined Seeds (Fallback)"

        avail_pm = [c for c in pm_cols if c in ref_df.columns]
        if len(avail_pm) < 3:
            return df_tube_features

        ref_vals = ref_df[avail_pm].values.astype(np.float64)
        n_dim = len(avail_pm)
        ref_mean = ref_vals.mean(axis=0)
        ref_cov = np.cov(ref_vals, rowvar=False) + np.eye(n_dim) * 1e-6
        inv_cov = np.linalg.pinv(ref_cov)

        tube_vals = df_tube_features[avail_pm].values.astype(np.float64)
        diff = tube_vals - ref_mean
        md_sq = np.sum((diff @ inv_cov) * diff, axis=1)

        clip_mask = md_sq <= sigma_clip**2
        df_clipped = df_tube_features[clip_mask].copy()
        
        self.logger.info(
            f"📐 [Tube Sigma Clip] 基准源: {ref_label} | "
            f"σ_clip={sigma_clip} | 过滤: {len(df_tube_features)} → {len(df_clipped)} 颗"
        )
        return df_clipped

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
            if "ra" in df_seeds.columns and "dec" in df_seeds.columns:
                coord_cols = ["ra", "dec"]
            elif "l" in df_seeds.columns and "b" in df_seeds.columns:
                coord_cols = ["l", "b"]
            else:
                # 兼容全小写与大写
                cols_lower = {col.lower(): col for col in df_seeds.columns}
                if "ra" in cols_lower and "dec" in cols_lower:
                    coord_cols = [cols_lower["ra"], cols_lower["dec"]]
                elif "l" in cols_lower and "b" in cols_lower:
                    coord_cols = [cols_lower["l"], cols_lower["b"]]
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

    def _update_tube_pca_state(self, ctx, pca_model, length_deg):
        ctx.state.tube_pca_center = (
            float(pca_model.cluster_center_[0]), float(pca_model.cluster_center_[1])
        )
        ctx.state.tube_pca_components = pca_model.components_.copy()
        ctx.state.tube_pca_mean = pca_model.mean_.copy() if hasattr(pca_model, "mean_") else np.zeros(2)
        ctx.state.tube_width = float(pca_model.tube_width_)
        ctx.state.tube_length = length_deg

    def _fuse_core_and_tail_channels(self, df_res_core, df_res_tail, df_tube_full):
        m_thresh = cfg.THRESHOLD_HIGH_CONF
        prob_col = cfg.STD_COLS["PROB"]

        df_res_final = df_res_core.copy()
        df_res_final["core_prob"] = df_res_core[prob_col].values
        df_res_final["tail_prob"] = 0.0

        if not df_res_tail.empty and not df_tube_full.empty:
            tail_prob_map = df_res_tail.set_index("id")[prob_col]
            tube_ids_set = set(df_tube_full["id"].values)
            is_in_tube = df_res_final["id"].isin(tube_ids_set)
            df_res_final.loc[is_in_tube, "tail_prob"] = df_res_final.loc[is_in_tube, "id"].map(tail_prob_map).fillna(0.0).values

        is_core_member = df_res_final["core_prob"] >= m_thresh
        is_tail_member = df_res_final["tail_prob"] >= m_thresh

        df_res_final[prob_col] = df_res_final["core_prob"].values
        tail_takeover_mask = (~is_core_member) & is_tail_member
        df_res_final.loc[tail_takeover_mask, prob_col] = df_res_final.loc[tail_takeover_mask, "tail_prob"].values

        df_res_final["source"] = "field"
        df_res_final.loc[is_core_member & (~is_tail_member), "source"] = "core"
        df_res_final.loc[tail_takeover_mask, "source"] = "tail"
        df_res_final.loc[is_core_member & is_tail_member, "source"] = "both"

        return df_res_final