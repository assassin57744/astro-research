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

    def __init__(self, db_instance, target_cluster=None, target_category=None, mode="3d"):
        """初始化工作流实例。

        Args:
            db_instance (AstroDB): 活跃的 AstroDB 数据库对象。
            target_cluster (str): 当前处理的星团 ID。
            target_category (str): 当前审计的类别。
            mode (str): 算法执行模式。
        """
        self.db = db_instance
        self.target_cluster = target_cluster
        self.target_category = target_category
        self.mode = mode
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        self.manifest = getattr(db_instance, "data_manifest", {})
        self.t_master = cfg.TMPL.T_MASTER.format(
            cluster=self.target_cluster.lower(),
            category=self.target_category,
            mode=self.mode
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
        local_ctx.setdefault("CAT_NAME", local_ctx.get("ID_NAME", local_ctx.get("NAME")))

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
            # 🚀 [Bugfix] 仅对当前存在的特征执行清洗。
            # 派生特征（如 l, b, U, V, W）此时尚未生成，将在 Transformer 转换后的 _defensive_nan_purge 中处理
            available_features = [f for f in required_features if f in df_raw.columns]
            df_seeds = df_raw.dropna(subset=available_features).copy()
        else:
            df_seeds = df_raw.dropna().copy()
            
        self.logger.info(f"从数据源 [{v_src}] 提取了 {len(df_seeds)} 颗种子星")

        # 🚀 [混合模式] 初始化种子标签：先全部标记为 'raw_seed'
        df_tag = df_seeds[[cfg.STD_COLS['ID']]].copy()
        df_tag['seed_type'] = 'raw_seed'
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
        """
        self.logger.info(f"[{self.target_cluster}] 启动 post_pipeline: 执行主表标签状态同步...")
        try:
            # 1. 动态增加成员分类标签列 (如果不存在)
            self.db.execute(f"ALTER TABLE {self.t_master} ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE")
            self.db.execute(f"ALTER TABLE {self.t_master} ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE")

            # 2. 直接在 Master 表中执行状态标记
            condi_golden = f"{STD_COLS['PROB']} >= {GOLDEN_SAMPLE_THRESHOLD}"
            condi_candidates = f"{STD_COLS['PROB']} > {MEMBER_SAMPLE_THRESHOLD}"

            self.db.execute(f"UPDATE {self.t_master} SET is_golden = TRUE WHERE {condi_golden}")
            self.db.execute(f"UPDATE {self.t_master} SET is_candidate = TRUE WHERE {condi_candidates}")

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
            self.db.register_view_from_sql(v_candidates, f"SELECT * FROM {self.t_master} WHERE is_candidate = TRUE")

            return {
                "status": "success",
                "v_candidates": v_candidates,
                "stats": {
                    "n_golden": n_golden,
                    "n_candidates": n_candidates,
                    "n_seeds": n_seeds,
                    "n_seed_core": n_seed_core,
                    "n_seed_noise": n_seed_noise
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
        col_x = cfg.MASTER_COLS['X_MATCH']
        sql_cross = f"""
            SELECT 
                m.id,
                CASE 
                    WHEN m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} AND h.id IS NOT NULL THEN 'Matched'
                    WHEN m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} AND h.id IS NULL     THEN 'PG Only'
                    WHEN (m.prob <= {cfg.MEMBER_SAMPLE_THRESHOLD} OR m.prob IS NULL) AND h.id IS NOT NULL THEN 'Ref Only'
                END as {col_x}
            FROM {self.t_master} m
            LEFT JOIN {v_target} h ON m.id = h.id
            WHERE m.prob > {cfg.MEMBER_SAMPLE_THRESHOLD} OR h.id IS NOT NULL
        """
        df_x = self.db.query(sql_cross)
        self.db.tag_master_table(self.t_master, df_x)

        # 为深度审计准备输入视图（仅针对 PG Only 部分）
        v_audit_target = f"v_tmp_audit_pg_only"
        self.db.register_view_from_sql(v_audit_target, f"SELECT * FROM {self.t_master} WHERE {col_x} = 'PG Only'")

        # 统计并日志
        st_sql = f"SELECT {col_x}, count(*) FROM {self.t_master} WHERE {col_x} IS NOT NULL GROUP BY {col_x}"
        stats_raw = self.db.execute(st_sql).fetchall()
        stats_cross = {row[0]: row[1] for row in stats_raw}
        self.logger.info(f"📊 [交叉审计] Matched: {stats_cross.get('Matched', 0)} | PG Only: {stats_cross.get('PG Only', 0)} | Ref Only: {stats_cross.get('Ref Only', 0)}")

        return {
            "status": "success",
            "v_audit_target": v_audit_target,
            "stats": stats_cross
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
            
            # 🚀 [混合模式重构] 审计结果回灌 Master 表
            self.logger.info(f"📥 正在将深度审计结果同步至 Master 表...")
            self.db.tag_master_table(self.t_master, audit_report_df)

            # 🚀 [混合模式] 审计完成后，返回 Master 表的一个逻辑视图作为“审计报告”
            # 这样既不需要创建新物理表，又能保证返回的内容仅包含审计过的星源
            v_report = f"{self.t_master}_audited_report"
            self.db.register_view_from_sql(v_report, f"SELECT * FROM {self.t_master} WHERE audit_status IS NOT NULL")
            return v_report

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

    def _parse_pipeline_config(self) -> tuple[dict, str, list[str]]:
        """[私有方法] 原子拆解：解析 GMM 配置项与特征空间。

        Returns:
            tuple: (配置字典, 运行模式字符串, 特征列名列表)。
        """
        # 采用局部副本，防止污染全局配置
        gmm_cfg = GMM_CONFIG.copy()
        current_mode = self.mode
        gmm_cfg["dim_mode"] = current_mode

        feature_map = gmm_cfg.get("feature_map", {})
        self.logger.debug(f"当前 GMM_CONFIG 中的 feature_map 配置:\n {feature_map}")

        if current_mode not in feature_map:
            raise ValueError(f"未知的运行模式: {current_mode}，请核实 feature_map 配置。")

        required_features = feature_map[current_mode]
        self.logger.info(
            f"🌌 当前运行模式: [{current_mode}], 所需核心特征空间: {required_features}"
        )
        return gmm_cfg, current_mode, required_features

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

    @astro_checkpoint(cache_table_template="cache_{cluster}_{category}_{mode}_res", force_refresh=True)
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
        
        # 🚀 [核心修复] 调整初始化顺序：必须先初始化 Master 表，因为 _get_seeds 会尝试更新它
        field_idx = CLUSTERS[self.target_cluster]["FIELD_IDX"]
        df_target_raw = self._get_target(
            idx_data=field_idx,
            cfg_src=DATA[field_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
        )
        
        # 🚀 [混合模式] 步骤 1: 初始化主表
        self.db.init_master_table(self.t_master, df_target_raw)

        # 步骤 2: 获取种子星数据 (此时 tag_master_table 可以安全执行)
        seed_idx = CLUSTERS[self.target_cluster]["SEED_IDX"]
        df_seeds_raw = self._get_seeds(
            idx_data=seed_idx,
            cfg_src=DATA[seed_idx],
            manifest=self.manifest,
            ctx=ctx_cluster,
            required_features=required_features
        )

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
        
        # 🚀 [混合模式] 步骤 2: 标记种子星类型 (Core/Noise)
        if hasattr(params, 'df_seeds_classified'):
            updates = params.df_seeds_classified[[cfg.STD_COLS['ID'], 'density_status']]
            self.db.tag_master_table(self.t_master, updates)

        df_prob = engine.predict(df_target_final, params)
        # 🚀 [混合模式] 步骤 3: 直接将概率回灌 Master 表，不再创建独立的 pgmm_xxx 表
        self.db.tag_master_table(self.t_master, df_prob)

        # 🚀 [混合模式] 固化 Master 表当前状态并作为结果返回
        self.db.save_to_warehouse(self.t_master)
        return self.t_master
