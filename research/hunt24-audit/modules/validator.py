import os
import sys
import logging
import duckdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import re

import config as cfg
from config import (
    CLUSTERS,
    STD_COLS,
    MANIFEST,
    IDX_IDS_SIMBAD
)  # 导入核心配置


class UnifiedMemberValidator:
    """
    统一天体成员验证与多维度交叉审计引擎。

    集成物理模型约束与文献大数据验证，提供以下核心功能：
    1. 动态解析 PARSEC 等龄线模型，构建基于测光演化的非线性插值器（CMD 约束）。
    2. 执行多维动力学残差审计（自行与视差）及空间分布校验。
    3. 联动数仓执行全自动文献交叉审计。
    4. 内置高性能本地缓存型 SIMBAD 批量查询接口，支持百万级数据增量同步。

    Attributes:
        cluster_id (str): 星团标识符（如 'M45'）。
        db (AstroDB): 绑定的数据库实例。
        config (dict): 从 CLUSTERS 中提取的物理先验配置。
    """

    def __init__(self, cluster_id, db_instance=None, cache_dir=None):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

        if cluster_id not in CLUSTERS:
            raise ValueError(f"❌ 星团 {cluster_id} 不在配置文件中！")

        self.cluster_id = cluster_id
        self.db = db_instance
        self.config = CLUSTERS[cluster_id]
        self.cluster_name = self.config["ID_NAME"]

        # 数据库持久化缓存配置
        self.cache_table = MANIFEST[IDX_IDS_SIMBAD]["raw_table"] 
        self.isochrone_df = None
        self.cmd_interpolator = None
        self.color_min = None
        self.color_max = None

        # 驱动完整物理边界加载
        self._setup_physical_constraints()

        self.tidal_radius = self.config.get(
            "TIDAL_RADIUS", 10.0
        )  # 默认为 10.0，防止意外崩溃

    def _setup_physical_constraints(self):
        """
        🚀 【物理边界自适应解析引擎】
        完整实现：动态定位物理模型文件，自适应清洗并解析 PARSEC 模型表头，
        自动计算距离模数与星际消光修正，建立非线性 CMD 演化插值控制网格。
        """
        self.logger.info(
            f"🧬 [物理引擎] 正在加载星团 {self.cluster_name} 的测光演化约束模型..."
        )

        # 1. 动态提取配置文件中的等龄线路径
        iso_file_name = self.config.get("ISO_FILE", "")
        if not iso_file_name:
            self.logger.warning(
                f"⚠️ 星团 {self.cluster_name} 未配置 ISO_FILE。将跳过测光演化审计。"
            )
            return

        # 支持绝对路径与项目相对路径的健壮解析
        iso_path = Path(iso_file_name)
        if not iso_path.is_absolute():
            iso_path = self.db.dirs["raw"] / "oapd" / iso_file_name

        if not iso_path.exists():
            self.logger.error(f"❌ [Validator] 未找到等龄线模型文件: {iso_path}")
            return

        # 2. 动态检测并提取 PARSEC 等龄线文件的真实数据表头
        self.logger.debug(f"📖 正在解析物理模型文件头: {iso_path.name}")
        header_line = None
        data_start_idx = 0

        with open(iso_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                clean_line = line.strip()
                if not clean_line:
                    continue
                if (
                    clean_line.startswith("#")
                    and "Gmag" in clean_line
                    and "G_BPmag" in clean_line
                ):
                    header_line = clean_line.lstrip("#").strip().split()
                    data_start_idx = idx + 1
                    break
                elif not clean_line.startswith("#"):
                    data_start_idx = idx
                    break

        # 3. 加载并对齐列名
        try:
            if header_line:
                self.isochrone_df = pd.read_csv(
                    iso_path,
                    skiprows=data_start_idx,
                    sep=r"\s+",
                    names=header_line,
                    comment="#",
                )
            else:
                self.isochrone_df = pd.read_csv(iso_path, sep=r"\s+", comment="#")

            self.logger.info(f"✅ [Validator] 成功加载等龄线模型 ({len(self.isochrone_df)} 演化步长)")
        except Exception as e:
            self.logger.error(f"❌ [Validator] 解析等龄线文件失败: {str(e)}")
            return

        # 4. 动态识别 Gaia 测光波段字段
        col_mapping = {}
        for col in self.isochrone_df.columns:
            if col.lower() in ["gmag", "g"]:
                col_mapping["G"] = col
            if col.lower() in ["g_bpmag", "bpmag", "bp"]:
                col_mapping["BP"] = col
            if col.lower() in ["g_rpmag", "rpmag", "rp"]:
                col_mapping["RP"] = col

        if len(col_mapping) < 3:
            self.logger.error(f"❌ [Validator] 物理文件波段缺失: {col_mapping}")
            return

        # 5. 执行视距离模数与红化消光修正
        # 公式: m_v - M_v = 5 * log10(d) - 5 + A_v
        try:
            distance = self.config.get("DISTANCE_PC", 100.0)  # 默认单位: pc
            extinction_g = self.config.get("EXT_AG", 0.0)  # G波段视消光
            
            # 🧪 增强逻辑：如果配置缺失 E_BP_RP，根据 EXT_AG 自动按比例估算
            extinction_bp_rp = self.config.get("E_BP_RP")
            if extinction_bp_rp is None:
                extinction_bp_rp = extinction_g * 0.52 # 基于 Gaia DR3 经典红化律估算

            distance_modulus = 5.0 * np.log10(distance) - 5.0

            # 物理网格整体向观测红化视空间平移
            model_g = (
                self.isochrone_df[col_mapping["G"]].values
                + distance_modulus
                + extinction_g
            )
            model_bp = self.isochrone_df[col_mapping["BP"]].values
            model_rp = self.isochrone_df[col_mapping["RP"]].values
            model_color = (model_bp - model_rp) + extinction_bp_rp

            # 6. ✨ 核心：构建非线性单调一维连续空间插值器
            # 按照颜色轴排序，确保插值函数能够完美收敛
            sort_idx = np.argsort(model_color)
            sorted_color = model_color[sort_idx]
            sorted_g = model_g[sort_idx]

            # 过滤由于物理演化极端膨胀阶段导致的输入非单调重复噪点
            unique_idx = np.unique(sorted_color, return_index=True)[1]

            self.color_min = sorted_color[unique_idx].min()
            self.color_max = sorted_color[unique_idx].max()

            # 使用线性边界外推防御机制（bounds_error=False）建立插值函数
            self.cmd_interpolator = interp1d(
                sorted_color[unique_idx],
                sorted_g[unique_idx],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            self.logger.debug(f"🎨 [Validator] CMD 插值网格构建成功，色指数区间: [{self.color_min:.2f}, {self.color_max:.2f}]")
            self.logger.debug(f"🎨 [成员校验器] CMD 插值网格构建成功，色指数区间: [{self.color_min:.2f}, {self.color_max:.2f}]")
        except Exception as math_err:
            self.logger.error(f"❌ [Validator] 构建 CMD 插值矩阵失败: {str(math_err)}")

    def run_full_audit_ex(self, v_target_detail: str) -> pd.DataFrame:
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
        sql = self._build_audit_sql(v_target_detail) # This SQL will now join with the cache table managed by AstroDB
        audit_df = self.db.execute(sql).df()

        # 2. 空间预处理 - 专门处理几何和空间关系
        audit_df = self._preprocess_spatial_data(audit_df)

        # 2. 执行物理指标校验
        audit_df = self._calculate_phys_metrics(audit_df)

        # 🚀 优化：使用向量化操作代替 apply(axis=1) 以显著提升百万级数据处理效率

        # 3. 批量文献标签判定
        lit_metrics = self._evaluate_literature_labels_batch(audit_df)
        lit_confirmed = lit_metrics["is_literature_member"]
        phys_pass = audit_df["phys_audit_pass"]

        # 4. 向量化最终规则决策 (np.select 代替 apply)
        conditions = [
            (phys_pass == True) & (lit_confirmed == True),
            (phys_pass == True) & (lit_confirmed == False),
            (phys_pass == False) & (lit_confirmed == True), # Literature Only
            (phys_pass == False) & (lit_confirmed == False) # Contamination
        ]
        choices = [
            "Confirmed Member", 
            "New Candidate", 
            "Literature Only", 
            "Contamination"
        ]
        audit_df["audit_status"] = np.select(conditions, choices, default="Contamination")

        # 5. 向量化细化诊断备注 (处理 Literature Only 子集)
        audit_df["audit_note"] = "N/A"
        mask_lit = audit_df["audit_status"] == "Literature Only"

        if mask_lit.any():
            # 批量计算备注
            notes = pd.Series("", index=audit_df.index)
            notes.loc[mask_lit & (audit_df["pm_residual"] < cfg.PHYS_LIT_PM_LIMIT) & (audit_df["cmd_residual"] > cfg.PHYS_LIT_CMD_LIMIT)] += "CMD Outlier | "
            notes.loc[mask_lit & (audit_df["distance_to_center"] > self.tidal_radius)] += "Tidal Tail Member | "

            ruwe_col = audit_df["ruwe"] if "ruwe" in audit_df.columns else pd.Series(1.0, index=audit_df.index)
            notes.loc[mask_lit & (ruwe_col > cfg.AUDIT_RUWE_LIMIT)] += "Gaia Data Quality Issue | "
            
            # 清理末尾分隔符并填充默认值
            final_notes = notes.str.rstrip(" | ").replace("", "Standard Literature Entry")
            audit_df.loc[mask_lit, "audit_note"] = final_notes

        self._print_audit_summary(audit_df)

        return audit_df

    def _build_audit_sql(self, v_target_input: str) -> str:
        """
        [私有方法] 构建审计核心 SQL 语句，执行特征对齐与文献 Join。

        Args:
            v_target_input (str): 输入源视图。

        Returns:
            str: 组装好的 SQL 语句。
        """
        t_simbad_aligned = self.cache_table
        
        plx, pmra, pmdec = (
            self.config.get("PLX_REF", 1.0),
            self.config.get("PMRA_REF", 0.0),
            self.config.get("PMDEC_REF", 0.0),
        )

        lit_cols = (
            "sim.main_id, sim.ids" # These columns are expected from the cache table
        )
        lit_join = (
            f"LEFT JOIN {t_simbad_aligned} sim ON CAST(p.id AS VARCHAR) = sim.gaia_dr3_id"
        )

        return f"""
        WITH physical_stage AS (
            SELECT *,
                   ABS(plx - {plx}) AS plx_residual,
                   SQRT(POWER(pmra - {pmra}, 2) + POWER(pmdec - {pmdec}, 2)) AS pm_residual
            FROM {v_target_input}
        )
        SELECT p.*, {lit_cols} FROM physical_stage p {lit_join}
        """

    def _calculate_phys_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """天体物理学稳健审计版本"""
        if df.empty:
            return df

        # 1. 确保基础残差列存在
        if "cmd_residual" not in df.columns:
            df["cmd_residual"] = np.nan

        pm_penalty = df["pm_residual"] / self.config.get("PM_RADIUS", 3.0)
        plx_penalty = df["plx_residual"] / self.config.get("PLX_ERROR", 0.5)

        if self.cmd_interpolator and all(c in df.columns for c in ["color", "mag"]):
            theoretical_g = self.cmd_interpolator(df["color"].values)
            df["cmd_residual"] = np.abs(df["mag"].values - theoretical_g)

        cmd_dev_scale = self.config.get("CMD_DEV", 0.8)
        cmd_penalty = df["cmd_residual"].fillna(cmd_dev_scale) / cmd_dev_scale

        pm_score = np.clip(pm_penalty, 0, 2.5)
        plx_score = np.clip(plx_penalty, 0, 2.5)
        cmd_score = np.clip(cmd_penalty, 0, 2.5)

        kinematics_valid = (pm_score < 2.0) & (plx_score < 2.0)

        w = cfg.PHYS_VERIFY_WEIGHTS
        weighted_penalty = (pm_score * w["pm"]) + (plx_score * w["plx"]) + (cmd_score * w["cmd"])
        score_valid = (
            weighted_penalty < cfg.PHYS_VERIFY_PENALTY_LIMIT
        )  # 给予外围边缘星更大的生存空间

        df["phys_audit_pass"] = kinematics_valid & score_valid

        passed_df = df[df["phys_audit_pass"]]
        if not passed_df.empty:
            self.logger.info(
                f"📈 [PhysAudit] 统计: 通过率 {len(passed_df)}/{len(df)} | "
                f"PM残差均值: {passed_df['pm_residual'].mean():.2f} | "
                f"CMD残差均值: {passed_df['cmd_residual'].mean():.2f}"
            )

        return df

    def run_full_audit(self, v_target_input: str) -> pd.DataFrame:
        """[旧版本/兼容性入口] 以数仓为核心进行全量天体多阶段过滤。"""
        self.logger.info(f"🎬 [Validator] 启动传统审计流，视图: {v_target_input}...")

        if not self.db:
            raise RuntimeError("❌ 错误: 必须绑定活跃的 AstroDB 实例。")

        table_exists = self.db.table_exists(self.cache_table)

        if not table_exists:
            self.logger.warning(f"⚠️ [Validator] 未检测到文献对齐表 `{self.cache_table}`。")

        plx_ref = self.config.get("PLX_REF", 1.0)
        pmra_ref = self.config.get("PMRA_REF", 0.0)
        pmdec_ref = self.config.get("PMDEC_REF", 0.0)

        str_re = r"(?:Gaia DR[23]\s+)?([0-9]+)"
        t_simbad_aligned = self.cache_table # Use the validator's specific cache table

        lit_cols = (
            "sim.main_id AS simbad_preferred_name, sim.ids AS simbad_all_aliases"
        )
        lit_join = (
            f"LEFT JOIN {t_simbad_aligned} sim "
            f"ON CAST(p.id AS VARCHAR) = sim.gaia_dr3_id"
        )

        # 3. 使用多行字符串构建结构化的 SQL，保持层次感
        sql_audit_matrix = f"""
        WITH physical_stage AS (
            SELECT 
                *,
                ABS(plx - {plx_ref}) AS plx_residual,
                SQRT(POWER(pmra - {pmra_ref}, 2) + POWER(pmdec - {pmdec_ref}, 2)) AS pm_residual
            FROM {v_target_input}
        ),
        literature_stage AS (
            SELECT 
                p.*,
                {lit_cols}
            FROM physical_stage p
            {lit_join}
        )
        SELECT * FROM literature_stage;
        """

        try:
            audit_base_df = self.db.con.execute(sql_audit_matrix).df()
        except Exception as e:
            self.logger.error(f"❌ [Validator] 运行 SQL 审计失败: {str(e)}")
            raise e

        phys_pm_gate = audit_base_df["pm_residual"] <= self.config.get("PM_RADIUS", 3.0)
        phys_plx_gate = audit_base_df["plx_residual"] <= self.config.get(
            "PLX_ERROR", 0.5
        )

        if (
            self.cmd_interpolator is not None
            and "color" in audit_base_df.columns
            and "mag" in audit_base_df.columns
        ):
            # 批量获取理论等龄线视星等值
            theoretical_g = self.cmd_interpolator(audit_base_df["color"].values)
            cmd_residuals = np.abs(audit_base_df["mag"].values - theoretical_g)

            audit_base_df["cmd_residual"] = cmd_residuals
            phys_cmd_gate = (
                (cmd_residuals <= self.config.get("CMD_DEV", 0.8))
                & (audit_base_df["color"] >= self.color_min)
                & (audit_base_df["color"] <= self.color_max)
            )
        else:
            audit_base_df["cmd_residual"] = np.nan
            phys_cmd_gate = True

        audit_base_df["phys_audit_pass"] = phys_pm_gate & phys_plx_gate & phys_cmd_gate
        audit_base_df["audit_status"] = audit_base_df.apply(
            self._finalize_status_rules, axis=1
        )
        self._print_audit_summary(audit_base_df)

        return audit_base_df

    def _finalize_status_rules(self, row) -> str:
        """⚙️ 【星团身份判定多路决策树规则】"""
        phys_pass = row.get("phys_audit_pass", False)
        literature_results = self.verify_literature_membership(row)
        literature_confirmed = literature_results["is_literature_confirmed"]

        if phys_pass:
            if literature_confirmed:
                return "Confirmed Member"
            else:
                return "New Candidate"
        else:
            if literature_confirmed:
                return "Literature Only"
            else:
                return "Contamination"

    def verify_literature_membership(self, row) -> dict:
        """
        [单条校验] 验证天体别名集合是否包含目标星团关键字。

        Args:
            row (dict | pd.Series): 包含 simbad_all_aliases 的数据行。

        Returns:
            dict: 包含判定结果（confirmed, potential 等）。
        """
        def normalize_name(name):
            if pd.isna(name) or str(name).strip().upper() == "NONE":
                return ""
            return re.sub(r"\s+", " ", str(name)).strip().upper()

        simbad_aliases = normalize_name(row.get("simbad_all_aliases"))
        simbad_preferred = normalize_name(row.get("simbad_preferred_name"))

        cluster_keyword = self.cluster_name.replace("_", " ").upper()
        cluster_keyword = re.sub(r"\s+", " ", cluster_keyword).strip()

        is_strict_match = (cluster_keyword in simbad_aliases) or (
            cluster_keyword in simbad_preferred
        )
        is_potential_match = "CL*" in simbad_preferred

        # self.logger.info(
        #     f"🔍 文献验证: {row.get('gaia_dr3_id')} - Strict: {is_strict_match}, Potential: {is_potential_match}"
        # )
        # self.logger.info(f"   验证星团关键词: '{cluster_keyword}'")

        return {
            "confirmed": is_strict_match,
            "potential": is_potential_match,
            "is_literature_confirmed": is_strict_match or is_potential_match,
        }

    def _print_audit_summary(self, df: pd.DataFrame):
        """控制台高亮输出最终统计摘要"""
        total = len(df)
        stats = df["audit_status"].value_counts().to_dict()
        
        msg = [ # Use f-strings for cleaner formatting
            f"{'='*70}",
            f"📊 [审计摘要] 星团: {self.cluster_name} | 样本总数: {total}",
            f"  🔹 确认成员 (物理+文献一致):  {stats.get('Confirmed Member', 0)}",
            f"  ✨ 算法新发现 (仅物理符合):   {stats.get('New Candidate', 0)}",
            f"  ⚠️ 文献孤儿星 (仅文献收录):   {stats.get('Literature Only', 0)}",
            f"  ❌ 背景污染噪点:             {stats.get('Contamination', 0)}",
            f"{'='*70}"
        ]
        for line in msg:
            self.logger.info(line)

    def validate_member(self, star_row) -> bool:
        """向后兼容单星测光与动力学快检接口"""
        return bool(star_row.get("phys_audit_pass", False))

    def _refine_literature_status(self, row) -> str:
        """
        生成天体身份审计的深度诊断备注。

        用于解释为什么某个被文献标记为成员的天体在物理上被拒绝。
        """
        notes = []

        if row["pm_residual"] < cfg.PHYS_LIT_PM_LIMIT and row["cmd_residual"] > cfg.PHYS_LIT_CMD_LIMIT:
            notes.append("CMD Outlier")

        if row.get("distance_to_center", 0) > self.tidal_radius:
            notes.append("Tidal Tail Member")

        if row.get("ruwe", 1.0) > cfg.AUDIT_RUWE_LIMIT:
            notes.append("Gaia Data Quality Issue")

        if not notes:
            return "Standard Literature Entry"
        return " | ".join(notes)

    def _preprocess_spatial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """[私有方法] 计算每个源到星团中心的物理投影距离 (单位: pc)。"""
        if df.empty:
            return df

        center = SkyCoord(
            ra=self.config["CENTER_RA"] * u.deg, dec=self.config["CENTER_DEC"] * u.deg
        )
        stars = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

        # 2. 计算天球角距离（先拿到弧度值，方便后续物理转换）
        separation_rad = stars.separation(center).radian

        cluster_distance_pc = self.config.get("DISTANCE_PC") or 136.2

        df["distance_to_center"] = cluster_distance_pc * separation_rad
        return df

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
            prefix="Gaia DR3 ", # SIMBAD typically expects "Gaia DR3 ID" format
            chunk_size=chunk_size
        )

        if not df_final_merged.empty:
            membership_metrics = self._evaluate_literature_labels_batch(df_final_merged)
            df_final_merged = pd.concat([df_final_merged, membership_metrics], axis=1)

        return df_final_merged

    def _evaluate_literature_labels_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [私有方法] 利用向量化字符串操作批量判定文献身份标签。
        """
        if df.empty:
            return pd.DataFrame(columns=["is_literature_member", "match_type"])

        cluster_keyword = self.cluster_name.replace("_", " ").upper()
        cluster_keyword = re.sub(r"\s+", " ", cluster_keyword).strip()

        def clean_series(series):
            return series.fillna("").astype(str).str.upper() \
                         .str.replace(r"\s+", " ", regex=True) \
                         .str.strip() \
                         .replace("NONE", "")

        norm_aliases = clean_series(df["ids"])
        norm_preferred = clean_series(df["main_id"])

        is_strict = norm_aliases.str.contains(cluster_keyword, regex=False, na=False) | \
                    norm_preferred.str.contains(cluster_keyword, regex=False, na=False)
        
        is_potential = norm_preferred.str.contains("CL*", regex=False, na=False)

        is_literature_member = is_strict | is_potential

        match_type = pd.Series("Unmatched", index=df.index)
        match_type.mask(is_potential, "Potential", inplace=True)
        match_type.mask(is_strict, "Strict", inplace=True)

        return pd.DataFrame({
            "is_literature_member": is_literature_member,
            "match_type": match_type
        })

def run_simbad_batch_audit():
    # 2. 配置高亮日志，以便直观观测“本地缓存命中”与“CDS 远程跨网打包下载”的细节
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    )
    logger = logging.getLogger("AstroPipeline.BatchRunner")

    # 3. 声明数仓路径与目标星表视图信息
    db_path = r"D:/git/repo/Alapha/cluster_audit/data/warehouse/astrodb_internal.db"
    target_table = "v_audit_hunt_melotte_22_pgmm_only_audited"
    id_column = "id"

    logger.info(f"📂 正在建立与本地 DuckDB 数仓的瞬时连接: {db_path}")
    if not os.path.exists(db_path):
        logger.error(f"❌ 找不到指定的数据库文件，请核对路径！")
        return

    # 4. 从 DuckDB 中高效拉取待审计的输入源
    try:
        # 使用 context manager 确保连接用完即释放
        with duckdb.connect(db_path, read_only=True) as con:
            # 仅提取需要的 id 列，避免拖带大字段造成内存震荡
            query_sql = (
                f"SELECT {id_column} FROM {target_table} WHERE {id_column} IS NOT NULL"
            )
            input_df = con.execute(query_sql).df()

        total_sources = len(input_df)
        logger.info(
            f"📊 成功从视图 `{target_table}` 中装载了 {total_sources} 颗待验证恒星。"
        )

        if total_sources == 0:
            logger.warning("⚠️ 视图内无可用的天体 ID，审计流程中止。")
            return

        # 将输入列转换为纯数字字符串列表，适配 SIMBAD 接口
        candidate_ids = input_df[id_column].astype(str).tolist()

    except Exception as db_err:
        logger.error(f"❌ 从 DuckDB 读取目标数据集失败: {str(db_err)}")
        return

    # 5. 实例化验证器
    # 这里传入 Melotte_22 对应的配置 ID（根据你之前 verify_literature_membership 的逻辑，
    # 验证器内部会提取该星团的中心参数与等龄线，并自动创建本地缓存文件：simbad_local_archive.csv）
    cluster_key = "M45"  # 请根据你的 config.CLUSTERS 中的具体键名微调
    logger.info(f"🧬 正在初始化星团 [{cluster_key}] 的统一成员验证引擎...")

    # 如果 UnifiedMemberValidator 初始化需要传入活动 db 实例，可在此处构造传入。
    # 如果纯粹调用文献接口，db_instance 可留空或传 None
    validator = UnifiedMemberValidator(
        cluster_id=cluster_key, db_instance=con
    )  # Pass the database connection

    # 6. 执行高频低耗批量交叉审计（内部自动完成：区分缓存 -> 远程打包单次请求 -> 回灌本地）
    logger.info(
        "⚡ 启动批量文献穿透审计引擎（通过 AstroDB 提供的服务，支持分片与增量缓存）..."
    )
    audit_report_df = validator.sync_simbad_cache(candidate_ids)

    # 7. 控制台格式化对齐打印穿透报告结果
    logger.info(
        "========================================================================"
    )
    logger.info(f"🏁 本轮批量文献审计穿透报告抽样（共 {len(audit_report_df)} 行）:")
    with pd.option_context(
        "display.max_rows",
        200,
        "display.max_columns",
        None,
        "display.width",
        1000,
        "display.max_colwidth",
        50,  # 限制别名显示长度，防止打印过宽
    ):
        print(
            audit_report_df[
                [
                    "gaia_dr3_id",
                    "main_id",
                    "cache_hit",
                    "is_literature_member",
                    "match_type",
                ]
            ].head(200)
        )
    logger.info(
        "========================================================================"
    )

    # 8. （可选）如果你需要把本次通过文献审计的信息写回或另存为 CSV 供科研作图/写论文：
    output_csv = Path(db_path).parent / "melotte_22_simbad_audited_report.csv"
    audit_report_df.to_csv(output_csv, index=False)
    logger.info(f"💾 审计穿透完整矩阵报告已成功转储至本地磁盘: {output_csv}")


if __name__ == "__main__":
    run_simbad_batch_audit()
