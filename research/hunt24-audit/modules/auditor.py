# modules/auditor.py
"""物理一致性审计策略模块。

提供可插拔的物理验证策略，每种策略封装完整的审计管线：
  - Chi2UpmaskAuditor      (默认): 卡方检验 + pyUPMASK CMD,  p ≥ cfg.THRESHOLD_MEMBERSHIP_PROB
  - Chi2ResidualAuditor    (实验A): 卡方检验 + 等龄线插值残差, p ≥ cfg.ALPHA_CHI2_PVALUE
  - WeightedPenaltyAuditor (实验B): 启发式加权惩罚分

工厂函数 create_auditor() 根据 cfg.VALIDATION_STRATEGY 自动路由。
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import config as cfg
from modules.cluster import StarCluster
import modules.pyUPMASK.pyUPMASK as upmask_mod


# =============================================================================
# 抽象基类
# =============================================================================

class BasePhysicalAuditor(ABC):
    """物理审计策略抽象基类。"""

    def __init__(self, cluster: StarCluster, logger: logging.Logger, dim_mode: str = "5d_h"):
        self.cluster = cluster
        self.logger = logger
        self.dim_mode = dim_mode

    # -------------------------------------------------------------------
    # 共享：NaN 过滤
    # -------------------------------------------------------------------

    @staticmethod
    def _filter_nan(df: pd.DataFrame, dim_mode: str) -> pd.DataFrame:
        """剔除运动学关键字段含 NaN 的行，返回过滤后的副本。"""
        required = ["ra", "dec", "pmra", "pmdec"]
        if dim_mode != "2d":
            required.append("plx")
        existing = [c for c in required if c in df.columns]
        nan_mask = df[existing].isna().any(axis=1)
        if nan_mask.any():
            return df[~nan_mask].copy()
        return df

    # -------------------------------------------------------------------
    # 共享：统计日志
    # -------------------------------------------------------------------

    def _log_stats(self, df: pd.DataFrame):
        """输出物理审计通过率与关键指标均值。"""
        passed = df[df["is_phys_consistent"]]
        if passed.empty:
            return
        self.logger.info(
            f"📈 [PhysAudit] 统计: 通过率 {len(passed)}/{len(df)} | "
            f"综合自行/速度惩罚分均值: {passed['pm_score'].mean():.2f} | "
            f"CMD残差均值: {passed['cmd_residual'].mean():.2f} | "
            f"平均加权总惩罚分: {passed['weighted_penalty'].mean():.2f}"
        )

    # -------------------------------------------------------------------
    # 抽象入口（所有子类必须实现 audit）
    # -------------------------------------------------------------------

    @abstractmethod
    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行物理一致性审计，返回带 is_phys_consistent 列的 DataFrame。

        Args:
            df: 输入数据帧（含运动学、测光等字段）
        """
        ...


# =============================================================================
# 卡方系列共享基类
# =============================================================================

class _Chi2Auditor(BasePhysicalAuditor, ABC):
    """卡方检验系列策略的共享基类：运动学 / 视差 / RV / 决策逻辑。"""

    p_threshold: float = cfg.ALPHA_CHI2_PVALUE  # 子类覆盖

    # -------------------------------------------------------------------
    # 共享：维度计算
    # -------------------------------------------------------------------

    def _compute_kine_chi2(self, df: pd.DataFrame):
        """动力学卡方：3D 速度或 2D 自行马氏距离。

        使用当前子集的 PM/UVW 中位数作为动态参考值（而非固定配置值），
        使得审计能自适应不同子集（PG Only / Ref Only）的 PM 分布差异。
        Ref Only 星往往是更弥散的外围成员，PM 质心可能与核心不同，
        使用动态参考可避免系统性偏差导致的 χ² 膨胀。

        注意：5d_h 模式（银道特征空间 pm_l_cosb/pm_b）在审计时不做特殊处理。
        原因：
          1. SQL (_build_audit_sql) 已通过 stx_view 计算出赤道 pmra_residual/pmdec_residual，
             审计矩阵始终包含 pmra/pmdec 列，无论特征空间是赤道还是银道。
          2. pm_inv_cov 基于 PMRA_DISPERSION/PMDEC_DISPERSION 构建，是赤道坐标系下的。
          3. 赤道 ↔ 银道是纯坐标系旋转，马氏距离在正交旋转下保持不变，
             因此 χ² 值相同，无需为银道模式单独实现。
          总之：审计和聚类是解耦的 —— 聚类用银道特征空间，审计统一用赤道自行残差。
        """
        if all(c in df.columns for c in ["u", "v", "w"]):
            uvw_ref = df[["u", "v", "w"]].median().values
            u_res = df[["u", "v", "w"]].values - uvw_ref
            uvw_inv_cov = self.cluster.uvw_inv_cov
            kine = np.einsum("ni,ij,nj->n", u_res, uvw_inv_cov, u_res)
            dof = 3
            self.logger.debug(
                f"  🎯 [Kine] 动态 UVW 参考: ({uvw_ref[0]:.2f}, {uvw_ref[1]:.2f}, {uvw_ref[2]:.2f}) km/s"
            )
        else:
            pmra_ref = df["pmra"].median()
            pmdec_ref = df["pmdec"].median()
            pm_res = df[["pmra", "pmdec"]].values - np.array([pmra_ref, pmdec_ref])
            pm_inv_cov = self.cluster.pm_inv_cov
            kine = np.einsum("ni,ij,nj->n", pm_res, pm_inv_cov, pm_res)
            dof = 2
            self.logger.debug(
                f"  🎯 [Kine] 动态 PM 参考: pmra={pmra_ref:.3f}, pmdec={pmdec_ref:.3f} mas/yr"
            )

        df["kine_chi2"] = kine
        df["total_integrated_chi2"] += kine
        df["total_dof"] += dof

    def _compute_plx_chi2(self, df: pd.DataFrame):
        """视差卡方 (DoF=1)，仅 3D+ 模式生效。"""
        plx_err = self.cluster.get_param("PLX_ERROR", 1.0)
        plx_chi2 = (df["plx_residual"] / plx_err) ** 2
        df["plx_chi2"] = plx_chi2
        df["total_integrated_chi2"] += plx_chi2
        df["total_dof"] += 1

    def _compute_rv_chi2(self, df: pd.DataFrame):
        """视向速度卡方 (DoF=1)，仅对有 RV 的行生效。"""
        if "rv" not in df.columns:
            return
        rv_err = self.cluster.get_param("RV_ERROR", 5.0)
        rv_ref = self.cluster.get_param("RV_REF", 0.0)
        has_rv = df["rv"].notna()
        if not has_rv.any():
            return
        rv_chi2 = ((df.loc[has_rv, "rv"] - rv_ref) / rv_err) ** 2
        df.loc[has_rv, "rv_chi2"] = rv_chi2
        df.loc[has_rv, "total_integrated_chi2"] += rv_chi2.values
        df.loc[has_rv, "total_dof"] += 1

    # -------------------------------------------------------------------
    # 共享：决策
    # -------------------------------------------------------------------

    def _decide_chi2(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一卡方假设检验决策。"""
        dof_safe = df["total_dof"].clip(lower=1)
        df["global_cluster_probability"] = chi2.sf(
            df["total_integrated_chi2"], dof_safe
        )
        df["is_phys_consistent"] = (
            df["global_cluster_probability"] >= self.p_threshold
        )

        self.logger.info(f"📊 [卡方检验报告] p 阈值 = {self.p_threshold}（上尾检验）:")
        for dof_val in sorted(df["total_dof"].unique()):
            dof_mask = df["total_dof"] == dof_val
            if not dof_mask.any():
                continue
            _dof = max(int(dof_val), 1)
            chi2_cutoff = chi2.ppf(1 - self.p_threshold, _dof)  # 上尾临界
            passed = (dof_mask & df["is_phys_consistent"]).sum()
            total = dof_mask.sum()
            self.logger.info(
                f"  -> DoF={_dof}: χ²临界(上尾)={chi2_cutoff:.3f}, "
                f"通过={passed}/{total} ({passed/total*100:.1f}%)"
            )

        # 诊断：未通过星的各分量均值
        failed = df[~df["is_phys_consistent"]]
        if not failed.empty:
            parts = []
            for col, label in [("kine_chi2", "PM"), ("plx_chi2", "PLX"),
                               ("rv_chi2", "RV"), ("cmd_chi2", "CMD")]:
                if col in failed.columns and failed[col].notna().any():
                    parts.append(f"{label}={failed[col].mean():.2f}")
            self.logger.info(
                f"  💡 [诊断] 未通过星 χ² 分量均值: {' | '.join(parts)}"
            )
        return df

    # -------------------------------------------------------------------
    # 抽象：CMD 子策略
    # -------------------------------------------------------------------

    @abstractmethod
    def _compute_cmd(self, df: pd.DataFrame):
        """CMD 维度卡方填充 (各子策略独立实现)。"""
        ...

    # -------------------------------------------------------------------
    # 模板方法
    # -------------------------------------------------------------------

    def _run_chi2_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """卡方系列共享管线：维度计算 → CMD → 决策 → 后处理。"""
        is_2d = self.dim_mode == "2d"

        # 日志诊断：输出当前弥散度参数，用于排查 χ² 偏离问题
        self.logger.info(
            f"📐 [Chi2 弥散度参数] "
            f"PMRA_DISP={self.cluster.get_param('PMRA_DISPERSION', 'N/A'):.3f}, "
            f"PMDEC_DISP={self.cluster.get_param('PMDEC_DISPERSION', 'N/A'):.3f}, "
            f"PLX_ERR={self.cluster.get_param('PLX_ERROR', 'N/A'):.3f}, "
            f"CMD_DEV={self.cluster.get_param('CMD_DEV', 'N/A'):.3f}"
        )

        self._compute_kine_chi2(df)
        if not is_2d:
            self._compute_plx_chi2(df)
        self._compute_rv_chi2(df)

        self._compute_cmd(df)
        df = self._decide_chi2(df)

        # 向后兼容字段
        dof_safe = df["total_dof"].clip(lower=1)
        df["weighted_penalty"] = df["total_integrated_chi2"] / dof_safe
        df["pm_score"] = df["total_integrated_chi2"]

        return df


# =============================================================================
# 策略 A：chi2_upmask (默认)
# =============================================================================

class Chi2UpmaskAuditor(_Chi2Auditor):
    """卡方检验 + pyUPMASK CMD 聚类概率 → χ²。"""

    p_threshold = cfg.ALPHA_CHI2_PVALUE

    def __init__(self, cluster: StarCluster, logger: logging.Logger, cluster_id: str, dim_mode: str = "5d_h"):
        super().__init__(cluster, logger, dim_mode)
        self.cluster_id = cluster_id

    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._run_chi2_pipeline(df)

    def _compute_cmd(self, df: pd.DataFrame):
        """pyUPMASK 聚类概率 → χ²。"""
        if not all(c in df.columns for c in ["ra", "dec", "color", "mag"]):
            return

        df["upmask_prob"] = 0.0
        upmask_cols = ["ra", "dec", "color", "mag"]

        # 加载外部星表
        ext_csv_path = Path(cfg.GAIA_INPUT_DIR) / f"{self.cluster_id.lower()}_pyUPMASK.csv"
        self.logger.info(f"📥 [PhysAudit] 尝试加载外部星表: {ext_csv_path}")
        ext_gaia_ids: set = set()
        if ext_csv_path.exists():
            try:
                ext_df = pd.read_csv(ext_csv_path)
                id_col = next((c for c in ext_df.columns if "source" in c.lower()), None)
                if id_col:
                    ext_gaia_ids = set(ext_df[id_col].dropna().astype(str).unique())
                    self.logger.info(
                        f"📥 [PhysAudit] 外部星表加载成功: {len(ext_gaia_ids)} 颗基准星。"
                    )
            except Exception as e:
                self.logger.error(f"❌ [PhysAudit] 外部星表加载失败: {e}")

        df["id_str"] = df["id"].astype(str)
        is_in_external = df["id_str"].isin(ext_gaia_ids)

        valid_cmd = df[upmask_cols].notna().all(axis=1)
        if not valid_cmd.any():
            return

        valid_df = df[valid_cmd].copy()
        valid_idx = valid_df.index
        n_iter = int(self.cluster.get_param("UPMASK_ITERATIONS", 20))
        max_cl = int(self.cluster.get_param("UPMASK_MAX_CLUSTERS", 5))

        probs_all = upmask_mod.dataProcess(
            ID=valid_idx.values,
            xy=valid_df[["ra", "dec"]].values,
            data=valid_df[["color", "mag"]].values,
            data_err=valid_df[["color_err", "mag_err"]].values,
            verbose=0, OL_runs=n_iter, parallel_flag=False, parallel_procs=1,
            resampleFlag=True, PCAflag=False, PCAdims=2,
            GUMM_flag=False, GUMM_perc=None, KDEP_flag=False,
            IL_runs=5, N_membs=10, N_cl_max=max_cl,
            clust_method="KMeans", clRjctMethod="rkfunc",
            C_thresh=0.05, cl_method_pars={},
        )
        upmask_prob = np.mean(probs_all, axis=0) if len(probs_all) > 1 else probs_all[0]
        df.loc[valid_idx, "upmask_prob"] = upmask_prob

        cmd_chi2 = -2.0 * np.log(upmask_prob + 1e-10)
        if "color_excess" in df.columns and "color_excess_sigma" in df.columns:
            outlier = np.abs(df["color_excess"]) > (5.0 * df["color_excess_sigma"])
            cmd_chi2[outlier[valid_cmd].values] += 1.0

        df.loc[valid_idx, "cmd_residual"] = cmd_chi2
        df.loc[valid_idx, "cmd_chi2"] = cmd_chi2

        # 仅 PG Only 星累加 CMD 卡方
        pg_only_valid = (~is_in_external) & valid_cmd
        if pg_only_valid.any():
            df.loc[pg_only_valid, "total_integrated_chi2"] += df.loc[pg_only_valid, "cmd_residual"]
            df.loc[pg_only_valid, "total_dof"] += 1


# =============================================================================
# 策略 B：chi2_cmd_residual (实验 A)
# =============================================================================

class Chi2ResidualAuditor(_Chi2Auditor):
    """卡方检验 + 等龄线插值残差 → χ²。"""

    p_threshold = cfg.ALPHA_CHI2_PVALUE

    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._run_chi2_pipeline(df)

    def _compute_cmd(self, df: pd.DataFrame):
        """等龄线插值残差 → χ²。"""
        if not all(c in df.columns for c in ["color", "mag"]):
            return

        cmd_interp = self.cluster.cmd_interpolator
        c_min, c_max = self.cluster.cmd_color_bounds

        if cmd_interp is None:
            self.logger.warning("⚠️ [PhysAudit] cmd_interpolator 未就绪，跳过 CMD 卡方。")
            return

        raw_res = df["mag"].values - cmd_interp(df["color"].values)
        cmd_res = np.where(raw_res < 0, -raw_res * 0.5, raw_res)
        out_mask = (df["color"] < c_min) | (df["color"] > c_max)
        cmd_res[out_mask] *= 1.5

        cmd_dev = self.cluster.get_param("CMD_DEV", 0.8)
        cmd_chi2 = (cmd_res / cmd_dev) ** 2

        df["cmd_residual"] = cmd_res
        df["cmd_chi2"] = cmd_chi2
        df["total_integrated_chi2"] += cmd_chi2
        df["total_dof"] += 1

        self.logger.info(
            f"  ⚡ [PhysAudit] CMD 插值残差 χ² 完成。CMD_DEV={cmd_dev} mag, "
            f"平均残差: {cmd_res.mean():.3f} mag, 平均 χ²: {cmd_chi2.mean():.3f}"
        )


# =============================================================================
# 策略 C：weighted_penalty (实验 B)
# =============================================================================

class WeightedPenaltyAuditor(BasePhysicalAuditor):
    """启发式加权惩罚分验证。"""

    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        is_2d = self.dim_mode == "2d"
        is_physical_v = self.dim_mode in ["3d_v", "6d_p"]

        penalties: dict = {}
        df["cmd_residual"] = np.nan

        # ── A. 动力学 ──
        if is_physical_v and all(c in df.columns for c in ["u", "v", "w"]):
            v_ref = self.cluster.get_param("UVW_REF", np.zeros(3))
            penalties["pm"] = np.sqrt(
                ((df["u"] - v_ref[0]) / self.cluster.get_param("U_ERROR", 2.5)) ** 2
                + ((df["v"] - v_ref[1]) / self.cluster.get_param("V_ERROR", 1.8)) ** 2
                + ((df["w"] - v_ref[2]) / self.cluster.get_param("W_ERROR", 1.2)) ** 2
            )
        else:
            pmra_disp = self.cluster.get_param("PMRA_DISPERSION", 3.0)
            pmdec_disp = self.cluster.get_param("PMDEC_DISPERSION", 3.0)
            penalties["pm"] = np.sqrt(
                (df["pmra_residual"] / pmra_disp) ** 2
                + (df["pmdec_residual"] / pmdec_disp) ** 2
            )
        df["pm_score"] = penalties["pm"]

        # ── B. 视差 (仅 3D+) ──
        if not is_2d:
            plx_err = self.cluster.get_param("PLX_ERROR", 0.5)
            penalties["plx"] = df["plx_residual"] / plx_err

        # ── C. 视向速度 ──
        if "rv" in df.columns:
            rv_ref = self.cluster.get_param("RV_REF", 0.0)
            rv_err = self.cluster.get_param("RV_ERROR", 5.0)
            penalties["rv"] = (df["rv"] - rv_ref).abs() / rv_err

        # ── D. CMD ──
        if self.cluster.cmd_interpolator is not None and all(
            c in df.columns for c in ["color", "mag"]
        ):
            raw_res = df["mag"].values - self.cluster.cmd_interpolator(df["color"].values)
            cmd_res = np.where(raw_res < 0, -raw_res * 0.5, raw_res)
            c_min, c_max = self.cluster.cmd_color_bounds
            out_mask = (df["color"] < c_min) | (df["color"] > c_max)
            cmd_res[out_mask] *= 1.5
            df["cmd_residual"] = np.abs(cmd_res)
            cmd_dev = self.cluster.get_param("CMD_DEV", 0.8)
            penalties["cmd"] = np.abs(cmd_res) / cmd_dev

        # ── 归一化 + 硬门槛 ──
        fill_vals = {"pm": 2.5, "plx": 2.5, "rv": 1.0, "cmd": 2.5}
        for key in ["pm", "plx", "rv", "cmd"]:
            col = f"{key}_score"
            if key in penalties:
                p = penalties[key]
                if isinstance(p, np.ndarray):
                    p = pd.Series(p, index=df.index)
                df[col] = p.clip(0, 2.5).fillna(fill_vals[key])
            elif col not in df.columns:
                df[col] = fill_vals[key]

        kine_limit = self.cluster.get_param("KINE_SCORE_LIMIT", 2.0)
        kine_valid = df["pm_score"] < kine_limit
        if not is_2d:
            kine_valid &= df["plx_score"] < kine_limit
        if "rv" in df.columns:
            kine_valid &= df["rv"].isna() | (df["rv_score"] < kine_limit)

        # ── 加权融合 ──
        base_w = cfg.PHYS_VERIFY_WEIGHTS.copy()
        if "rv" in penalties:
            base_w["rv"] = 0.2
        active = [k for k in ["pm", "plx", "rv", "cmd"] if k in penalties or k == "pm"]
        w = {k: base_w[k] for k in active if k in base_w}
        w_sum = sum(w.values())
        w = {k: v / w_sum for k, v in w.items()}

        df["weighted_penalty"] = sum(df[f"{k}_score"] * w[k] for k in w)
        score_valid = df["weighted_penalty"] < cfg.PHYS_VERIFY_PENALTY_LIMIT
        df["is_phys_consistent"] = kine_valid & score_valid

        self.logger.info(
            f"  ⚡ [WeightedPenalty] 动力学硬门槛={kine_limit}, "
            f"加权阈值={cfg.PHYS_VERIFY_PENALTY_LIMIT}, "
            f"通过={df['is_phys_consistent'].sum()}/{len(df)}"
        )
        return df


# =============================================================================
# 工厂函数
# =============================================================================

def create_auditor(strategy: str, cluster: StarCluster, logger: logging.Logger,
                   cluster_id: str | None = None,
                   dim_mode: str = "5d_h") -> BasePhysicalAuditor:
    """根据策略名创建对应的物理审计器实例。

    Args:
        strategy: "chi2_upmask" | "chi2_cmd_residual" | "weighted_penalty"
        cluster: 星团物理实体
        logger: 日志记录器
        cluster_id: 星团 ID（chi2_upmask 策略需要，用于 pyUPMASK 文件路径构造）
        dim_mode: 维度模式 ("2d", "5d", "5d_h", "3d_v", "6d_p" 等)
    """
    _strategies = {
        "chi2_upmask": lambda: Chi2UpmaskAuditor(cluster, logger, cluster_id or cluster.id, dim_mode),
        "chi2_cmd_residual": lambda: Chi2ResidualAuditor(cluster, logger, dim_mode=dim_mode),
        "weighted_penalty": lambda: WeightedPenaltyAuditor(cluster, logger, dim_mode=dim_mode),
    }
    factory = _strategies.get(strategy)
    if factory is None:
        raise ValueError(f"未知的物理验证策略: {strategy}，可选: {list(_strategies)}")
    return factory()


# =============================================================================
# 文献审计：关键词构建 + 语义匹配
# =============================================================================

import re
from astroquery.simbad import Simbad


def build_cluster_keywords(cluster: StarCluster, cluster_id: str,
                           logger: logging.Logger) -> list:
    """动态构建星团的规范化核心词及缩写别名网（纯函数）。

    从配置参数 (NAME, SIMBAD_NAME, ID_NAME, CAT_NAME)、星团 ID 前缀、
    及预设特殊别名中提取并派生所有可能的别名变体。

    Args:
        cluster: 星团物理实体（提供 get_param 访问配置）
        cluster_id: 星团标识符
        logger: 日志记录器

    Returns:
        去重并按长度降序排列的关键词列表，最长的优先匹配。
    """
    keywords: list[str] = []
    for key in ["NAME", "SIMBAD_NAME", "ID_NAME", "CAT_NAME"]:
        val = cluster.get_param(key)
        if val:
            val_upper = str(val).upper()
            keywords.append(val_upper)
            keywords.append(val_upper.replace("_", " "))
            keywords.append(val_upper.replace("_", ""))
            keywords.append(re.sub(r"\s+", "", val_upper.replace("_", " ")))

    # 针对常见的前缀提取数字生成缩写
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
    match_id_num = re.search(r"\d+", cluster_id)
    if match_id_num:
        num = match_id_num.group()
        if cluster_id.startswith("M"):
            keywords.extend([f"M {num}", f"M{num}", f"MESSIER {num}"])
        elif cluster_id.startswith("NGC"):
            keywords.extend([f"NGC {num}", f"NGC{num}"])

    # 预设已知星团的特殊别名
    special_aliases = {
        "M45": ["PLEIADES", "SUBARU", "MELOTTE 22", "MEL 22", "M 45", "M45"],
        "M44": ["PRAESEPE", "BEEHIVE", "NGC 2632", "NGC2632", "M 44", "M44"],
        "Mel25": ["HYADES", "MELOTTE 25", "MEL 25", "MEL.25"],
        "Mel111": ["COMA BERENICES", "COMA BER", "MELOTTE 111", "MEL 111"],
    }
    if cluster_id in special_aliases:
        keywords.extend(special_aliases[cluster_id])

    keywords = sorted(
        list({k.strip().upper() for k in keywords if k.strip()}),
        key=len,
        reverse=True,
    )
    logger.info(
        f"🧬 [Semantic Audit] 激活语义审计核心，检索空间词网: {keywords}"
    )
    return keywords


class LiteratureAuditor:
    """文献共识审计器。

    基于 SIMBAD 实时父级语义树及已知星团特殊别名的权威语义树过滤算法，
    判断数据帧中的天体是否在已知文献中被归类为该星团成员。
    """

    def __init__(self, cluster: StarCluster, cluster_id: str, logger: logging.Logger):
        self.cluster = cluster
        self.cluster_id = cluster_id
        self.logger = logger

    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行文献共识审计。

        Args:
            df: 包含 main_id, ids, parent 列的数据帧

        Returns:
            含 is_lit_consensus 和 match_type 两列的 DataFrame。
        """
        if df.empty:
            return pd.DataFrame(columns=["is_lit_consensus", "match_type"])

        self.logger.debug(
            f" 🟢 [Semantic Audit]----------待审计数据---------------\n {df}"
        )

        keywords = build_cluster_keywords(
            self.cluster, self.cluster_id, self.logger
        )

        def clean_text(s):
            if pd.isna(s):
                return ""
            return re.sub(r"\s+", " ", str(s).strip().upper())

        # 构建规范正则模式
        kw_pattern = "|".join(
            [re.escape(k).replace(r"\ ", r"\s+") for k in keywords if k]
        )

        # 1. Parent 严格边界审计
        parent_pattern = rf"\b(?:{kw_pattern})\b"
        if "parent" in df.columns:
            parent_series = df["parent"].fillna("NONE").apply(clean_text)
            is_parent_match = parent_series.str.contains(
                parent_pattern, regex=True, case=False, na=False
            )
        else:
            is_parent_match = self._fallback_parent_lookup(
                df, kw_pattern, clean_text
            )

        # 2. 名称严格边界匹配 (Match Preferred or Aliases)
        norm_aliases = df["ids"].fillna("").apply(clean_text)
        norm_preferred = df["main_id"].fillna("").apply(clean_text)
        strict_pattern = rf"(?:^|\||\s)(?:{kw_pattern})(?:\s*\||\s|$)"

        is_strict = norm_aliases.str.contains(
            strict_pattern, regex=True, na=False, case=False
        ) | norm_preferred.str.contains(
            strict_pattern, regex=True, na=False, case=False
        )

        # 3. 约束后的 Potential 判定 (必须同时符合 CL* 前缀 + 当前星团关键词)
        is_potential = norm_preferred.str.contains(
            r"^CL\*", regex=True, na=False
        ) & norm_preferred.str.contains(
            rf"\b(?:{kw_pattern})\b", regex=True, na=False, case=False
        )

        # 聚合判定
        is_literature_member = (
            is_parent_match | is_strict | is_potential
        ).astype(bool)

        match_type = pd.Series(
            np.select(
                [is_parent_match, is_strict, is_potential],
                [
                    "Parent Relation Confirmed",
                    "Strict Name Bound Match",
                    "Potential Cluster Member Form",
                ],
                default="Unmatched",
            ),
            index=df.index,
            dtype=object,
        )

        return pd.DataFrame(
            {
                "is_lit_consensus": is_literature_member,
                "match_type": match_type,
            }
        )

    def _fallback_parent_lookup(
        self, df: pd.DataFrame, kw_pattern: str, clean_text_fn
    ) -> pd.Series:
        """实时 SIMBAD 家谱查询兜底（当输入缺失 parent 列时）。"""
        self.logger.warning(
            "⚠️ 输入矩阵中缺失 'parent' 字段，将执行实时网络家谱查询（这会非常慢！）"
        )
        norm_mids = df["main_id"].apply(clean_text_fn)
        unique_mids = [m for m in norm_mids.unique() if m and m != "NONE"]

        mid_to_match: dict = {}
        for mid in unique_mids:
            mid_to_match[mid] = False
            try:
                hierarchy = Simbad.query_hierarchy(mid, hierarchy="parents")
                if hierarchy is not None:
                    parent_names = [
                        clean_text_fn(p["main_id"]) for p in hierarchy
                    ]
                    for p in parent_names:
                        if re.search(
                            rf"\b(?:{kw_pattern})\b", p, flags=re.IGNORECASE
                        ):
                            mid_to_match[mid] = True
                            break
            except Exception as e:
                self.logger.warning(
                    f"⚠️ [实时家谱审计] 天体 {mid} 查询失败: {e}"
                )

        return norm_mids.map(mid_to_match).fillna(False).astype(bool)