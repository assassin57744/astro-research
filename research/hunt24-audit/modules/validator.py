# modules/validator.py
import logging
import numpy as np
import pandas as pd
import config as cfg
from config import CLUSTERS, MANIFEST, IDX_IDS_SIMBAD
from modules.cluster import StarCluster
from modules.auditor import create_auditor, LiteratureAuditor, BasePhysicalAuditor

class UnifiedMemberValidator:
    """
    统一天体成员验证与多维度交叉审计引擎。
    集成物理模型约束与文献大数据验证，提供以下核心功能：
    1. 彻底解耦物理实体层，由绑定的 StarCluster 自适应驱动测光演化（CMD 约束）与逆协方差空间计算。
    2. 执行多维动力学残差审计（自行与视差各向异性或 3D 银道坐标系）及空间分布校验。
    3. 联动数仓执行全自动文献交叉审计。
    4. 内置高性能本地缓存型 SIMBAD 批量查询接口，支持百万级数据增量同步。
    Attributes:
        cluster_id (str): 星团标识符（如 'M45'）。
        db (duckdb.Connect): 绑定的数据库实例。
        mode (str): 动力学审计维度模式 (2d, 3d_v, 5d, 6d_p 等)。
    """

    def __init__(self, cluster: StarCluster, feature_space="5d", db_instance=None):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        cluster_id = cluster.id
        if cluster_id not in CLUSTERS:
            raise ValueError(f"❌ 星团 {cluster_id} 不在配置文件中！")
        self.cluster_id = cluster_id.upper()
        self.feature_space = feature_space
        self.db = db_instance
        self.cluster_obj = cluster
        self.cluster_name = CLUSTERS[cluster_id]["ID_NAME"]
        self.cache_table = MANIFEST[IDX_IDS_SIMBAD]["raw_table"]
        # 物理审计器（策略由 cfg.VALIDATION_STRATEGY 决定，构造时一次性选定）
        strategy = getattr(cfg, "VALIDATION_STRATEGY", "chi2_upmask")
        self._phys_auditor = create_auditor(
            strategy, self.cluster_obj, self.logger,
            cluster_id=self.cluster_id, dim_mode=feature_space,
        )
        # 文献审计器
        self._lit_auditor = LiteratureAuditor(
            self.cluster_obj, self.cluster_id, self.logger
        )

    def run(self, v_target_detail: str) -> pd.DataFrame:
        """驱动多维度深度审计管线：加载 → 空间投影 → 物理审计 → 文献审计 → 融合决策。"""
        self.logger.info(f"🎬 [Validator] 启动审计管线，目标视图: {v_target_detail}")

        # ── 1. 数据加载 ──
        audit_matrix = self._load_audit_data(v_target_detail)
        if audit_matrix.empty:
            return audit_matrix

        # ── 2. 空间投影 ──
        self._compute_spatial_features(audit_matrix)

        # ── 3. 物理一致性审计 ──
        self._audit_physical_consistency(audit_matrix)

        # ── 4. 文献共识审计 ──
        consensus_df = self._audit_literature_consistency(audit_matrix)

        # ── 5. 融合决策 ──
        audit_matrix = self._fuse_audit_decisions(audit_matrix, consensus_df)

        # ── 6. 诊断备注 ──
        audit_matrix = self._apply_diagnostic_notes(audit_matrix)

        self._print_audit_summary(audit_matrix)
        return audit_matrix

    # ── 子步骤 ──

    def _load_audit_data(self, v_target_detail: str) -> pd.DataFrame:
        """1. 执行 SQL 获取审计数据，空表时补齐字段并提前退出。"""
        sql = self._build_audit_sql(v_target_detail)
        df = self.db.execute(sql).df()
        self.logger.info(f"🔍 [Validator] 获取到 {len(df)} 条样本。")
        self.logger.debug(f"🔍 [Validator] 审计数据: \n{df.head(5)}")
        for cc in df.columns:
            self.logger.debug(f"🔍 [Validator] 审计数据 columns : {cc}")

        if not df.empty:
            return df

        self.logger.warning(
            f"⚠️ [Validator] 目标视图 {v_target_detail} 未提取到任何样本，跳过后续矩阵计算。"
        )
        for col in ["distance_to_center", "cmd_residual", "is_phys_consistent",
                     "audit_status", "audit_note"]:
            df[col] = pd.Series(dtype=object)
        return df

    def _compute_spatial_features(self, df: pd.DataFrame) -> None:
        """2. 空间投影距离计算。直接修改 df。"""
        cluster_dist = self.cluster_obj.get_param("DISTANCE_PC", 100.0)
        df["distance_to_center"] = cluster_dist * np.radians(df["sep_deg"])

    def _fuse_audit_decisions(self, df: pd.DataFrame,
                               consensus_df: pd.DataFrame) -> pd.DataFrame:
        """5. 融合决策：物理 × 文献 → audit_status。"""
        is_phys = df["is_phys_consistent"]
        is_lit = consensus_df["is_lit_consensus"]
        df["audit_status"] = np.select(
            [
                is_phys & is_lit,                         # Confirmed Member
                is_phys & ~is_lit,                         # New Candidate
                ~is_phys & is_lit,                         # Literature Only
                ~is_phys & ~is_lit,                        # Contamination
            ],
            ["Confirmed Member", "New Candidate",
             "Literature Only", "Contamination"],
            default="Contamination",
        )
        return df

    def _apply_diagnostic_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """6. Literature Only 星追加诊断备注。"""
        df["audit_note"] = "N/A"
        mask = df["audit_status"] == "Literature Only"
        if not mask.any():
            return df

        ruwe = df["ruwe"] if "ruwe" in df.columns else pd.Series(1.0, index=df.index)
        tidal_radius = self.cluster_obj.get_param("TIDAL_RADIUS", 10.0)

        if "pmra_residual" in df.columns and "pmdec_residual" in df.columns:
            pm_outlier = (df["pmra_residual"] > cfg.PHYS_LIT_PM_LIMIT) | (
                df["pmdec_residual"] > cfg.PHYS_LIT_PM_LIMIT)
        else:
            pm_outlier = df["pm_residual"] > cfg.PHYS_LIT_PM_LIMIT

        df.loc[mask, "audit_note"] = np.select(
            [
                pm_outlier[mask] & (df.loc[mask, "cmd_residual"] > cfg.PHYS_LIT_CMD_LIMIT),
                df.loc[mask, "distance_to_center"] > tidal_radius,
                ruwe[mask] > cfg.AUDIT_RUWE_LIMIT,
            ],
            ["CMD Outlier", "Tidal Tail Member", "Gaia Data Quality Issue"],
            default="Standard Literature Entry",
        )
        return df

    def _build_audit_sql(self, v_target_input: str) -> str:
        """构建审计核心 SQL 语句，使用面向 Cluster 的物理参数替换凌乱的硬编码 config 获取"""
        t_simbad_aligned = self.cache_table
        # 全部收归实体属性代理，干净利落
        c_ra = self.cluster_obj.get_param("CENTER_RA")
        c_dec = self.cluster_obj.get_param("CENTER_DEC")
        plx = self.cluster_obj.get_param("PLX_REF")
        pmra = self.cluster_obj.get_param("PMRA_REF")
        pmdec = self.cluster_obj.get_param("PMDEC_REF")
        lit_cols = "sim.main_id, sim.ids, sim.parent"  # These columns are expected from the cache table
        lit_join = f"LEFT JOIN {t_simbad_aligned} sim ON CAST(p.id AS VARCHAR) = sim.gaia_dr3_id"
        return f"""
        WITH physical_stage AS (
            SELECT *,
                   haversine_distance({c_ra}, {c_dec}, ra, dec) as sep_deg,
                   ABS(plx - {plx}) AS plx_residual,
                   ABS(pmra - {pmra}) AS pmra_residual,
                   ABS(pmdec - {pmdec}) AS pmdec_residual,
                   SQRT(POWER(pmra - {pmra}, 2) + POWER(pmdec - {pmdec}, 2)) AS pm_residual
            FROM {v_target_input}
        )
        SELECT p.*, {lit_cols} FROM physical_stage p {lit_join}
        """

    def _audit_physical_consistency(self, df: pd.DataFrame) -> None:
        """[委托] 物理一致性审计，由构造时选定的策略执行。直接修改 df。"""
        if df.empty:
            return
        # 剔除运动学 NaN 行（利用 BasePhysicalAuditor 的静态方法，然后 in-place 删除）
        cleaned = BasePhysicalAuditor._filter_nan(df, self.feature_space)
        if cleaned is not df:
            df.drop(index=df.index.difference(cleaned.index), inplace=True)
        if df.empty:
            return
        self.logger.info(
            f"🔍 [PhysAudit] 启动物理一致性审计。样本: {len(df)}, 维度: {self.feature_space}"
        )
        # 初始化审计列
        df["cmd_residual"] = np.nan
        df["kine_chi2"] = np.nan
        df["plx_chi2"] = np.nan
        df["rv_chi2"] = np.nan
        df["cmd_chi2"] = np.nan
        df["total_integrated_chi2"] = 0.0
        df["total_dof"] = 0
        self._phys_auditor.audit(df)
        if "id_str" in df.columns:
            df.drop(columns=["id_str"], inplace=True)
        self._phys_auditor._log_stats(df)

    def _audit_literature_consistency(self, audit_matrix: pd.DataFrame) -> pd.DataFrame:
        """[委托] 文献共识审计，由构造时绑定的 LiteratureAuditor 执行。"""
        return self._lit_auditor.audit(audit_matrix)

    def _print_audit_summary(self, df: pd.DataFrame):
        """控制台高亮输出最终统计摘要"""
        total = len(df)
        stats = df["audit_status"].value_counts().to_dict()
        # 提取矩阵核心计数 (TP/FP/FN/TN)
        tp = stats.get("Confirmed Member", 0)  # 物理(+) & 文献(+)
        fp = stats.get("New Candidate", 0)  # 物理(+) & 文献(-)
        fn = stats.get("Literature Only", 0)  # 物理(-) & 文献(+)
        tn = stats.get("Contamination", 0)  # 物理(-) & 文献(-)
        # 计算边缘汇总 (Marginal Totals)
        phys_pos_total = tp + fp
        phys_neg_total = fn + tn
        lit_pos_total = tp + fn
        lit_neg_total = fp + tn
        # 1. 打印基础摘要
        self.logger.info("=" * 72)
        self.logger.info(
            f"📊 [验证结果摘要] 星团: {self.cluster_name} | 样本总数: {total}"
        )
        self.logger.info(f"  ✨ 物理验证通过总数 (TP+FP): {phys_pos_total}")
        self.logger.info(f"      --其中文献也证实 (TP):    {tp}")
        self.logger.info(f"      --其中文献缺失 (FP):      {fp}")
        self.logger.info(f"  ⚠️ 文献通过但物理偏离 (FN): {fn}")
        self.logger.info(f"  ❌ 双验证均未通过(背景污染):  {tn}")
        # 2. 打印二维判别矩阵 (Contingency Matrix)
        self.logger.info("-" * 72)
        self.logger.info("深度审计判别矩阵 (物理检查 vs 文献共识):")
        self.logger.info("-" * 72)
        self.logger.info(
            f"{'':<18} | {'文献证实 (+)':<12} | {'文献缺失 (-)':<12} | 物理汇总"
        )  # header
        self.logger.info(
            f"{'物理符合 (+)':<14} | {tp:<16} | {fp:<16} | {phys_pos_total}"
        )
        self.logger.info(
            f"{'物理偏离 (-)':<14} | {fn:<16} | {tn:<16} | {phys_neg_total}"
        )
        self.logger.info("-" * 72)
        self.logger.info(
            f"{'文献汇总':<14} | {lit_pos_total:<16} | {lit_neg_total:<16} | {total}"
        )  # footer
        self.logger.info("=" * 72)
    # =========================================================================
    # 🌟 高性能批量跨网络与本地缓存的星团成员判定引擎
    # =========================================================================

    def sync_simbad_cache(self, source_ids, chunk_size: int = 500) -> pd.DataFrame:
        """
        🚀 [高性能接口] 调用 AstroDB 提供的 SIMBAD 缓存同步服务。
        Args:
            source_ids (list | Series): 需要同步的 Gaia DR3 源 ID。
            chunk_size (int): 网络批量同步的文件片大小。
        Returns:
            pd.DataFrame: 包含文献对齐信息的完整数据帧。
        """
        # Call AstroDB's new SIMBAD sync method
        df_final_merged = self.db.sync_simbad_cache(
            source_ids=source_ids,
            cache_table_name=self.cache_table,
            prefix="Gaia DR3 ",  # SIMBAD typically expects "Gaia DR3 ID" format
            chunk_size=chunk_size,
        )
        if not df_final_merged.empty:
            consensus_df = self._lit_auditor.audit(df_final_merged)
            df_final_merged = pd.concat([df_final_merged, consensus_df], axis=1)
        return df_final_merged
