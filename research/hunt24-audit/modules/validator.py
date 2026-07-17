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
            strategy, self.cluster_obj, self.logger, self.cluster_id
        )
        # 文献审计器
        self._lit_auditor = LiteratureAuditor(
            self.cluster_obj, self.cluster_id, self.logger
        )

    def run(self, v_target_detail: str) -> pd.DataFrame:
        """
        驱动多维度深度审计管线。
        包括：
        1. 数据库侧执行物理残差预计算与文献对齐。
        2. 空间分布分析（距离中心投影）。
        3. 测光演化残差（CMD）分析。
        4. 综合物理与文献证据链，判定成员身份。
        Args:
            v_target_detail (str): 包含详细物理参数的输入视图名。
        Returns:
            pd.DataFrame: 包含审计结果（audit_status）与物理指标的完整数据帧。
        """
        self.logger.info(f"🎬 [Validator] 启动审计管线，目标视图: {v_target_detail}")
        # 1. 构建并执行 SQL 获取基础数据
        sql = self._build_audit_sql(v_target_detail)
        audit_matrix = self.db.execute(sql).df()
        self.logger.debug(f"🔍 [Validator] 获取到 {len(audit_matrix)} 条样本。")
        self.logger.debug(f"🔍 [Validator] 审计数据: \n{audit_matrix.head(5)}")
        for cc in audit_matrix.columns:
            self.logger.debug(f"🔍 [Validator] 审计数据 columns : {cc}")
        # 🚀 【防御机制】如果数据集为空，直接初始化结构并提前退出，防止下游 KeyError
        if audit_matrix.empty:
            self.logger.warning(
                f"⚠️ [Validator] 目标视图 {v_target_detail} 未提取到任何样本，跳过后续矩阵计算。"
            )
            # 补齐关键列名，确保下游合并或读取时不会崩溃
            empty_cols = [
                "distance_to_center",
                "cmd_residual",
                "is_phys_consistent",
                "audit_status",
                "audit_note",
            ]
            for col in empty_cols:
                audit_matrix[col] = pd.Series(dtype=object)
            return audit_matrix
        # 2. 空间投影距离计算 (利用从 Cluster 绑定的 CfgMgr 动态拉取的视距离参数)
        cluster_dist = self.cluster_obj.get_param("DISTANCE_PC", 100.0)
        audit_matrix["distance_to_center"] = cluster_dist * np.radians(
            audit_matrix["sep_deg"]
        )
        # 3. 物理一致性审计：验证观测数据是否符合星团物理规律 (送入重构后的 StarCluster 对象)
        audit_matrix = self._audit_physical_consistency(audit_matrix)
        # 4. 文献共识审计
        consensus_df = self._lit_auditor.audit(audit_matrix)
        is_lit_consensus = consensus_df["is_lit_consensus"]
        is_phys_consistent = audit_matrix["is_phys_consistent"]
        # 5. 向量化最终规则决策
        conditions = [
            (is_phys_consistent == True) & (is_lit_consensus == True),
            (is_phys_consistent == True) & (is_lit_consensus == False),
            (is_phys_consistent == False)
            & (is_lit_consensus == True),  # Literature Only
            (is_phys_consistent == False)
            & (is_lit_consensus == False),  # Contamination
        ]
        choices = [
            "Confirmed Member",
            "New Candidate",
            "Literature Only",
            "Contamination",
        ]
        audit_matrix["audit_status"] = np.select(
            conditions, choices, default="Contamination"
        )
        # 6. 向量化细化诊断备注
        audit_matrix["audit_note"] = "N/A"
        mask_lit = audit_matrix["audit_status"] == "Literature Only"
        if mask_lit.any():
            # 使用 np.select 进一步优化备注生成逻辑
            ruwe_col = (
                audit_matrix["ruwe"]
                if "ruwe" in audit_matrix.columns
                else pd.Series(1.0, index=audit_matrix.index)
            )
            # 仅对符合 Literature Only 条件的子集进行条件判定，确保结果长度对齐
            df_lit = audit_matrix[mask_lit]
            # 安全兼容一维及解耦后的二维自行残差判定备注
            if "pmra_residual" in df_lit.columns and "pmdec_residual" in df_lit.columns:
                pm_outlier_cond = (df_lit["pmra_residual"] > cfg.PHYS_LIT_PM_LIMIT) | (
                    df_lit["pmdec_residual"] > cfg.PHYS_LIT_PM_LIMIT
                )
            else:
                pm_outlier_cond = df_lit["pm_residual"] > cfg.PHYS_LIT_PM_LIMIT
            # 动态获取当前星团的潮汐半径限制
            tidal_radius = self.cluster_obj.get_param("TIDAL_RADIUS", 10.0)
            conds = [
                pm_outlier_cond & (df_lit["cmd_residual"] > cfg.PHYS_LIT_CMD_LIMIT),
                (df_lit["distance_to_center"] > tidal_radius),
                (ruwe_col[mask_lit] > cfg.AUDIT_RUWE_LIMIT),
            ]
            choices = ["CMD Outlier", "Tidal Tail Member", "Gaia Data Quality Issue"]
            audit_matrix.loc[mask_lit, "audit_note"] = np.select(
                conds, choices, default="Standard Literature Entry"
            )
        self._print_audit_summary(audit_matrix)
        return audit_matrix

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

    def _audit_physical_consistency(self, audit_matrix: pd.DataFrame) -> pd.DataFrame:
        """[委托] 物理一致性审计，由构造时选定的策略执行。"""
        if audit_matrix.empty:
            return audit_matrix
        dim_mode = self.feature_space
        audit_matrix = BasePhysicalAuditor._filter_nan(audit_matrix, dim_mode)
        if audit_matrix.empty:
            return audit_matrix
        self.logger.info(
            f"🔍 [PhysAudit] 启动物理一致性审计。样本: {len(audit_matrix)}, 维度: {dim_mode}"
        )
        # 初始化审计列
        audit_matrix["cmd_residual"] = np.nan
        audit_matrix["kine_chi2"] = np.nan
        audit_matrix["plx_chi2"] = np.nan
        audit_matrix["rv_chi2"] = np.nan
        audit_matrix["cmd_chi2"] = np.nan
        audit_matrix["total_integrated_chi2"] = 0.0
        audit_matrix["total_dof"] = 0
        if hasattr(self._phys_auditor, "audit_with_dim"):
            audit_matrix = self._phys_auditor.audit_with_dim(audit_matrix, dim_mode)
        else:
            audit_matrix = self._phys_auditor.audit(audit_matrix)
        if "id_str" in audit_matrix.columns:
            audit_matrix.drop(columns=["id_str"], inplace=True)
        self._phys_auditor._log_stats(audit_matrix)
        return audit_matrix

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
