import os
import sys
import logging
import duckdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astroquery.simbad import Simbad
import astropy.units as u
from pathlib import Path
import re

import config as cfg
from config import CLUSTERS, STD_COLS, MANIFEST, IDX_IDS_SIMBAD  # 导入核心配置


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

    def __init__(self, cluster_id, mode="5d", db_instance=None, cache_dir=None):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

        if cluster_id not in CLUSTERS:
            raise ValueError(f"❌ 星团 {cluster_id} 不在配置文件中！")

        self.cluster_id = cluster_id
        self.mode = mode
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

        self.tidal_radius = self._get_config_with_warning("TIDAL_RADIUS", 10.0)

    def _get_config_with_warning(self, key: str, default_val):
        """
        获取星团配置项的值。如果配置项缺失，则记录警告日志并返回默认值，以防止潜在的精度或逻辑缺陷。
        """
        if key not in self.config:
            self.logger.warning(
                f"⚠️ [ConfigFallback] 星团 {self.cluster_name} ({self.cluster_id}) 配置文件中未检测到关键参数 '{key}'！"
                f"将自动降级使用默认值 {default_val}。这可能会影响物理一致性审计的准确性，建议在 config.py 中补齐配置。"
            )
            return default_val
        return self.config[key]

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

        # 2. 稳健解析等龄线文件 (恢复至之前的兼容算法，解决 astropy 无法识别格式的问题)
        try:
            # PARSEC 等龄线表头通常位于注释行的最后一行，手动提取以确保列名映射准确
            with open(iso_path, "r", encoding="utf-8") as f:
                col_names = None
                last_comment_idx = 0
                for idx, line in enumerate(f):
                    if line.startswith("#"):
                        last_comment_idx = idx
                        # 记录最后一行包含关键字段的注释作为列名
                        if "Gmag" in line and "G_BPmag" in line:
                            col_names = line.lstrip("#").strip().split()
                    elif not line.strip():
                        continue
                    else:
                        break

            # 使用 pandas 高效加载数据，sep=r'\s+' 处理变长空格，comment='#' 跳过所有注释行
            if col_names:
                self.isochrone_df = pd.read_csv(
                    iso_path,
                    sep=r"\s+",
                    comment="#",
                    names=col_names,
                    skiprows=last_comment_idx,
                    engine="python",
                )
            else:
                self.isochrone_df = pd.read_csv(iso_path, sep=r"\s+", comment="#")

            self.logger.info(
                f"✅ [Validator] 成功加载等龄线模型 ({len(self.isochrone_df)} 演化步长)"
            )
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
            distance = self._get_config_with_warning(
                "DISTANCE_PC", 100.0
            )  # 默认单位: pc
            extinction_g = self._get_config_with_warning("EXT_AG", 0.0)  # G波段视消光

            # 🧪 增强逻辑：如果配置缺失 E_BP_RP，根据 EXT_AG 自动按比例估算
            extinction_bp_rp = self.config.get("E_BP_RP")
            if extinction_bp_rp is None:
                extinction_bp_rp = extinction_g * cfg.REDDENING_RATIO_BP_RP

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
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            self.logger.debug(
                f"🎨 [Validator] CMD 插值网格构建成功，色指数区间: [{self.color_min:.2f}, {self.color_max:.2f}]"
            )
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
        sql = self._build_audit_sql(
            v_target_detail
        )  # This SQL will now join with the cache table managed by AstroDB
        audit_matrix = self.db.execute(sql).df()

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

        # 2. 空间投影距离计算 (基于 SQL 返回 of sep_deg)
        cluster_dist = self._get_config_with_warning("DISTANCE_PC", 100.0)
        audit_matrix["distance_to_center"] = cluster_dist * np.radians(
            audit_matrix["sep_deg"]
        )

        # 2. 物理一致性审计：验证观测数据是否符合星团物理规律
        audit_matrix = self._audit_physical_consistency(audit_matrix)

        # 3. 文献共识审计：验证身份记录是否与科研结论一致
        consensus_df = self._audit_literature_consensus(audit_matrix)

        is_lit_consensus = consensus_df["is_lit_consensus"]
        is_phys_consistent = audit_matrix["is_phys_consistent"]

        # 4. 向量化最终规则决策 (np.select 代替 apply)
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

        # 5. 向量化细化诊断备注 (处理 Literature Only 子集)
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
                pm_outlier_cond = (df_lit["pmra_residual"] > cfg.PHYS_LIT_PM_LIMIT) | (df_lit["pmdec_residual"] > cfg.PHYS_LIT_PM_LIMIT)
            else:
                pm_outlier_cond = df_lit["pm_residual"] > cfg.PHYS_LIT_PM_LIMIT

            conds = [
                pm_outlier_cond & (df_lit["cmd_residual"] > cfg.PHYS_LIT_CMD_LIMIT),
                (df_lit["distance_to_center"] > self.tidal_radius),
                (ruwe_col[mask_lit] > cfg.AUDIT_RUWE_LIMIT),
            ]
            choices = ["CMD Outlier", "Tidal Tail Member", "Gaia Data Quality Issue"]
            audit_matrix.loc[mask_lit, "audit_note"] = np.select(
                conds, choices, default="Standard Literature Entry"
            )

        self._print_audit_summary(audit_matrix)

        return audit_matrix

    def _build_audit_sql(self, v_target_input: str) -> str:
        """
        [私有方法] 构建审计核心 SQL 语句，执行特征对齐与文献 Join。

        Args:
            v_target_input (str): 输入源视图。

        Returns:
            str: 组装好的 SQL 语句。
        """
        t_simbad_aligned = self.cache_table

        c_ra = self._get_config_with_warning("CENTER_RA", None)
        c_dec = self._get_config_with_warning("CENTER_DEC", None)
        plx, pmra, pmdec = (
            self._get_config_with_warning("PLX_REF", 1.0),
            self._get_config_with_warning("PMRA_REF", 0.0),
            self._get_config_with_warning("PMDEC_REF", 0.0),
        )

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
        """物理一致性审计：通过多维残差与解耦的椭球惩罚系统评估天体与星团模型的匹配度。"""
        if audit_matrix.empty:
            return audit_matrix

        # 1. 环境上下文初始化
        # dim_mode = self._get_config_with_warning("DIM_MODE", "3d").lower()
        dim_mode = self.mode
        is_2d = (dim_mode == "2d")
        is_physical_v = (dim_mode in ["3d_v", "6d_p"])
        audit_matrix["cmd_residual"] = np.nan  # 初始化残差列
        
        self.logger.info(
            f"🔍 [PhysAudit] 启动物理一致性审计。样本总数: {len(audit_matrix)}, 维度模式: {dim_mode}"
        )

        # 2. 残差计算阶段 (各维度独立计算)
        penalties = {}

        # =====================================================================
        # A. 动力学残差 (自行解耦正椭圆约束 或 银道三维速度椭球约束)
        # =====================================================================
        if is_physical_v and all(c in audit_matrix.columns for c in ["u", "v", "w"]):
            # 🚀 3D速度空间解耦（银道坐标系各向异性约束）
            v_ref = self._get_config_with_warning("UVW_REF", np.zeros(3))
            
            # 分别提取三个方向各自的物理弥散标准差
            u_err = self._get_config_with_warning("U_ERROR", 2.5)
            v_err = self._get_config_with_warning("V_ERROR", 1.8)
            w_err = self._get_config_with_warning("W_ERROR", 1.2)

            # 计算各轴物理速度残差
            u_res = audit_matrix["u"] - v_ref[0]
            v_res = audit_matrix["v"] - v_ref[1]
            w_res = audit_matrix["w"] - v_ref[2]

            # 通过卡方椭球算子融合3D空间速度惩罚分
            penalties["kinematics"] = np.sqrt(
                (u_res / u_err) ** 2 + (v_res / v_err) ** 2 + (w_res / w_err) ** 2
            )
            
            # 为向下兼容和归一化阶段注入影子评分
            audit_matrix["pm_score"] = penalties["kinematics"]
            
            self.logger.info(
                f"  ⚡ [PhysAudit] 3D 银道空间速度审计完成（解耦椭球边界）。参考速度 UVW_REF: {v_ref}, "
                f"配置弥散 [U,V,W]_ERROR: [{u_err}, {v_err}, {w_err}] km/s, "
                f"平均综合速度惩罚分: {penalties['kinematics'].mean():.3f}"
            )
        else:
            # 🚀 2D 自行空间解耦（赤经/赤纬正椭圆约束）
            pmra_disp = self._get_config_with_warning("PMRA_DISPERSION", 3.0)
            pmdec_disp = self._get_config_with_warning("PMDEC_DISPERSION", 3.0)

            # 正椭圆卡方算子组合惩罚分：Score = sqrt((Δpmra/σ_pmra)^2 + (Δpmdec/σ_pmdec)^2)
            penalties["pm"] = np.sqrt(
                (audit_matrix["pmra_residual"] / pmra_disp) ** 2 + 
                (audit_matrix["pmdec_residual"] / pmdec_disp) ** 2
            )
            
            # 确保下游 Kinematics Gate 能顺利读取
            audit_matrix["pm_score"] = penalties["pm"]
            
            self.logger.info(
                f"  ⚡ [PhysAudit] 二维自行空间审计完成（解耦正椭圆边界）。"
                f"容忍度 [PMRA, PMDEC]_DISPERSION: [{pmra_disp}, {pmdec_disp}] mas/yr, "
                f"平均综合自行惩罚分: {penalties['pm'].mean():.3f}"
            )

        # =====================================================================
        # B. 视差残差 (仅 3D+)
        # =====================================================================
        if not is_2d:
            plx_err = self._get_config_with_warning("PLX_ERROR", 0.5)
            penalties["plx"] = audit_matrix["plx_residual"] / plx_err
            self.logger.info(
                f"  ⚡ [PhysAudit] 视差残差计算完成。视差基准不确定度 PLX_ERROR: {plx_err} mas, "
                f"平均残差: {audit_matrix['plx_residual'].mean():.3f} mas, 平均惩罚分: {penalties['plx'].mean():.3f}"
            )

        # =====================================================================
        # C. 视向速度残差 (如果列存在)
        # =====================================================================
        if "rv" in audit_matrix.columns:
            rv_ref = self._get_config_with_warning("RV_REF", 0.0)
            rv_err = self._get_config_with_warning("RV_ERROR", 5.0)
            rv_res = (audit_matrix["rv"] - rv_ref).abs()
            penalties["rv"] = rv_res / rv_err
            self.logger.info(
                f"  ⚡ [PhysAudit] 视向速度残差计算完成。RV_REF: {rv_ref} km/s, RV_ERROR: {rv_err} km/s, "
                f"平均残差: {rv_res.mean():.3f} km/s, 平均惩罚分: {penalties['rv'].mean():.3f}"
            )

        # =====================================================================
        # D. 测光演化残差 (CMD)
        # =====================================================================
        if self.cmd_interpolator and all(c in audit_matrix.columns for c in ["color", "mag"]):
            raw_res = audit_matrix["mag"].values - self.cmd_interpolator(audit_matrix["color"].values)
            
            # 非对称修正：联星方向(负)权重减半；超出范围权重增加 1.5 倍
            cmd_res = np.where(raw_res < 0, -raw_res * 0.5, raw_res)
            out_mask = (audit_matrix["color"] < self.color_min) | (audit_matrix["color"] > self.color_max)
            cmd_res[out_mask] *= 1.5
            audit_matrix["cmd_residual"] = cmd_res  # 持久化残差
            
            cmd_dev = self._get_config_with_warning("CMD_DEV", 0.8)
            penalties["cmd"] = pd.Series(cmd_res, index=audit_matrix.index) / cmd_dev
            self.logger.info(
                f"  ⚡ [PhysAudit] 测光演化残差计算完成。演化偏离限制 CMD_DEV: {cmd_dev} mag, "
                f"平均残差: {cmd_res.mean():.3f} mag, 平均惩罚分: {penalties['cmd'].mean():.3f}"
            )

        # =====================================================================
        # 3. 评分归一化与硬门槛判定 (Kinematics Gate)
        # =====================================================================
        fill_values = {"pm": 2.5, "plx": 2.5, "rv": 1.0, "cmd": 2.5}
        for key in ["pm", "plx", "rv", "cmd"]:
            if key in penalties:
                audit_matrix[f"{key}_score"] = np.clip(penalties[key], 0, 2.5).fillna(fill_values[key])
            elif f"{key}_score" not in audit_matrix.columns:
                audit_matrix[f"{key}_score"] = fill_values[key]
        self.logger.info("  ⚡ [PhysAudit] 维度评分区间截断与空值填充完成。")

        # 4. 硬门槛判定 (Kinematics Gate)
        # 核心逻辑：自行和视差(若存在)必须通过初步筛选，且 RV 不能偏离过大
        kine_score_limit = self._get_config_with_warning("KINE_SCORE_LIMIT", 2.0)
        kine_valid = audit_matrix["pm_score"] < kine_score_limit

        if not is_2d:
            kine_valid &= audit_matrix["plx_score"] < kine_score_limit
        if "rv" in audit_matrix.columns:
            kine_valid &= audit_matrix["rv"].isna() | (audit_matrix["rv_score"] < kine_score_limit)

        self.logger.info(
            f"  ⚡ [PhysAudit] 动力学门槛筛选完成。门限值: {kine_score_limit}, 通过度: {kine_valid.sum()}/{len(audit_matrix)}"
        )

        # =====================================================================
        # 5. 权重动态分配与最终决策
        # =====================================================================
        base_weights = cfg.PHYS_VERIFY_WEIGHTS.copy()
        if "rv" in penalties:
            base_weights["rv"] = 0.2  # 6D 模式下赋予 RV 20% 权重

        # 动态过滤当前算得的有效维度
        active_dims = [k for k in ["pm", "plx", "rv", "cmd"] if k in penalties or k == "pm"]
        w = {k: base_weights[k] for k in active_dims if k in base_weights}
        w_sum = sum(w.values())
        w = {k: v / w_sum for k, v in w.items()}

        audit_matrix["weighted_penalty"] = sum(audit_matrix[f"{k}_score"] * w[k] for k in w)
        
        # 综合判定定格
        score_valid = audit_matrix["weighted_penalty"] < cfg.PHYS_VERIFY_PENALTY_LIMIT
        audit_matrix["is_phys_consistent"] = kine_valid & score_valid

        # 7. 统计输出
        self._log_phys_audit_stats(audit_matrix)
        return audit_matrix

    def _log_phys_audit_stats(self, df):
        """内部工具：输出物理审计统计日志"""
        passed = df[df["is_phys_consistent"]]
        if not passed.empty:
            self.logger.info(
                f"📈 [PhysAudit] 统计: 通过率 {len(passed)}/{len(df)} | "
                f"综合自行/速度惩罚分均值: {passed['pm_score'].mean():.2f} | "
                f"CMD残差均值: {passed['cmd_residual'].mean():.2f} | "
                f"平均加权总惩罚分: {passed['weighted_penalty'].mean():.2f}"
            )

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
            consensus_df = self._audit_literature_consensus(df_final_merged)
            df_final_merged = pd.concat([df_final_merged, consensus_df], axis=1)

        return df_final_merged

    def _audit_literature_consensus(self, audit_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        [重构版] 基于 SIMBAD 实时父级语义树的权威审计引擎。
        """
        if audit_matrix.empty:
            return pd.DataFrame(columns=["is_lit_consensus", "match_type"])

        self.logger.debug(
            f" 🟢 [emantic Audit]----------待审计数据---------------\n {audit_matrix}"
        )

        # 1. 动态构建星团的规范化核心词及缩写别名网
        keywords = []
        for key in ["NAME", "SIMBAD_NAME", "ID_NAME", "CAT_NAME"]:
            val = self.config.get(key)
            if val:
                val_upper = str(val).upper()
                keywords.append(val_upper)
                keywords.append(val_upper.replace("_", " "))
                keywords.append(val_upper.replace("_", ""))
                keywords.append(re.sub(r"\s+", "", val_upper.replace("_", " ")))

        # 针对常见的前缀提取数字生成缩写 (如 Melotte 22 -> Mel 22, Mel.22; NGC 2632 -> NGC 2632)
        for kw in list(keywords):
            match_num = re.search(r"\d+", kw)
            if match_num:
                num = match_num.group()
                if "MELOTTE" in kw or "MEL" in kw:
                    keywords.extend([f"MEL {num}", f"MEL.{num}", f"MELOTTE {num}"])
                elif "MESSIER" in kw or kw.startswith("M") or "M" in kw:
                    if "MELOTTE" not in kw:
                        keywords.extend([f"M {num}", f"M{num}", f"MESSIER {num}"])
                elif "NGC" in kw:
                    keywords.extend([f"NGC {num}", f"NGC{num}"])

        # 星团ID前缀解析
        match_id_num = re.search(r"\d+", self.cluster_id)
        if match_id_num:
            num = match_id_num.group()
            if self.cluster_id.startswith("M"):
                keywords.extend([f"M {num}", f"M{num}", f"MESSIER {num}"])
            elif self.cluster_id.startswith("NGC"):
                keywords.extend([f"NGC {num}", f"NGC{num}"])

        # 预设已知星团的特殊别名 (安全防御)
        special_aliases = {
            "M45": ["PLEIADES", "SUBARU", "MELOTTE 22", "MEL 22", "M 45", "M45"],
            "M44": ["PRAESEPE", "BEEHIVE", "NGC 2632", "NGC2632", "M 44", "M44"],
            "Mel25": ["HYADES", "MELOTTE 25", "MEL 25", "MEL.25"],
            "Mel111": ["COMA BERENICES", "COMA BER", "MELOTTE 111", "MEL 111"],
        }
        if self.cluster_id in special_aliases:
            keywords.extend(special_aliases[self.cluster_id])

        # keywords = list(set([k.strip().upper() for k in keywords if k.strip()]))
        keywords = sorted(
            list(set([k.strip().upper() for k in keywords if k.strip()])),
            key=len,
            reverse=True,
        )
        self.logger.info(
            f"🧬 [Semantic Audit] 激活语义审计核心，检索空间词网: {keywords}"
        )

        # 2. 向量化 Parent 审计 (优先使用已缓存的 parent 关系)
        is_parent_match = pd.Series(False, index=audit_matrix.index, dtype=bool)
        # kw_pattern = "|".join(re.escape(k) for k in keywords)
        # parent_pattern = rf"(?:^|\||\s)(?:{kw_pattern})(?:\b|\||$)"
        # 核心重构：简化正则，忽略严格边界，重点在于匹配关键词的“存在性”
        # 使用 re.IGNORECASE 替代预先的 .str.upper()，增强容错
        # 将所有空格替换为 \s+，这样能匹配 1 个或多个空格
        kw_pattern = "|".join([re.escape(k).replace(r"\ ", r"\s+") for k in keywords])
        # 只要字符串包含该关键词序列，无论前后是什么
        parent_pattern = rf"{kw_pattern}"

        def clean_text(s):
            """将多空格压缩为单个空格，确保匹配一致性"""
            return re.sub(r"\s+", " ", str(s).strip().upper())

        if "parent" in audit_matrix.columns:
            parent_series = audit_matrix["parent"].fillna("NONE").apply(clean_text)
            # parent_series = parent_series.replace(r"\s+", " ", regex=True)
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                1000,
                "display.max_colwidth",
                None,
            ):
                self.logger.debug(
                    f"🔎 [Semantic Audit] 检索父级名字：\n{parent_series}"
                )

            is_parent_match = parent_series.str.contains(
                parent_pattern, regex=True, case=False, na=False
            )
        else:
            # 💡 容错与向后兼容：如果输入中缺失 'parent' 字段，则执行按天体查询 Simbad 的兜底逻辑，并给出警告
            self.logger.warning(
                "⚠️ 输入矩阵中缺失 'parent' 字段，将执行实时网络家谱查询（这会非常慢！）"
            )
            unique_mids = [
                m
                for m in audit_matrix["main_id"].unique()
                if pd.notna(m) and str(m).strip().upper() not in ["", "NONE"]
            ]
            mid_to_parent_match = {}
            for mid in unique_mids:
                try:
                    hierarchy = Simbad.query_hierarchy(mid, hierarchy="parents")
                    if hierarchy is not None:
                        parent_names = [str(p["main_id"]).upper() for p in hierarchy]
                        for kw in keywords:
                            if any(kw in p for p in parent_names):
                                mid_to_parent_match[mid] = True
                                break
                except Exception as e:
                    self.logger.warning(f"⚠️ [实时家谱审计] 天体 {mid} 查询失败: {e}")
            is_parent_match = (
                audit_matrix["main_id"]
                .map(mid_to_parent_match)
                .fillna(False)
                .astype(bool)
            )

        # 3. 基于正则表达式的“成员/子天体”语法边界匹配 (兜底逻辑)
        is_strict = pd.Series(False, index=audit_matrix.index, dtype=bool)
        norm_aliases = audit_matrix["ids"].fillna("").apply(clean_text)
        norm_preferred = audit_matrix["main_id"].fillna("").apply(clean_text)

        strict_pattern = (
            # rf"(?:^|\||\s)(?:CL\*|NAME|NAME\s+)?(?:{kw_pattern})(?:\b|\||$)"
            rf"(?:^|\||\s)(?:{kw_pattern})(?:\s*\||$)"
        )
        is_strict = norm_aliases.str.contains(
            strict_pattern, regex=True, na=False, case=False
        ) | norm_preferred.str.contains(
            strict_pattern, regex=True, na=False, case=False
        )

        # 4. 聚合判定
        is_potential = norm_preferred.str.contains(
            r"^CL\*", regex=True, na=False
        ).astype(bool)
        is_literature_member = (is_parent_match | is_strict | is_potential).astype(bool)

        # 5. 生成匹配类型矩阵，使用 np.select 彻底规避类型与 mask inplace 警告
        condlist = [is_parent_match, is_strict, is_potential]
        choicelist = [
            "Parent Relation Confirmed",
            "Strict Name Bound Match",
            "Potential Cluster Member Form",
        ]
        match_type = pd.Series(
            np.select(condlist, choicelist, default="Unmatched"),
            index=audit_matrix.index,
            dtype=object,
        )

        return pd.DataFrame(
            {"is_lit_consensus": is_literature_member, "match_type": match_type}
        )


def run_simbad_batch_audit():
    # 2. 配置高亮日志，以便直观观测“本地缓存命中”与“CDS 远程跨网打包下载”的细节
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    )
    logger = logging.getLogger("AstroPipeline.BatchRunner")

    # 3. 声明数仓路径与目标星表视图信息
    db_path = str(cfg.DATA_DIR / "warehouse" / "astrodb_internal.db")
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
                    "is_lit_consensus",
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
