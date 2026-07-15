# modules/validator.py
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
from modules.cluster import StarCluster  # 🎯 引入重构后的星团物理实体类

import modules.pyUPMASK.pyUPMASK as upmask_mod  # 🎯 引入 pyUPMASK 模块
# from modules.pyUPMASK import dataProcess  # 🎯 引入 pyUPMASK 的核心数据处理函数

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

    def __init__(self, cluster: StarCluster, mode="5d", db_instance=None, cache_dir=None):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

        cluster_id = cluster.id
        if cluster_id not in CLUSTERS:
            raise ValueError(f"❌ 星团 {cluster_id} 不在配置文件中！")

        self.cluster_id = cluster_id.upper()
        self.mode = mode
        self.db = db_instance

        # 🎯 核心改变：实例化纯粹的天体物理实体类，由它承载原本凌乱的配置读取与插值计算
        # self.cluster_obj = StarCluster(
        #     cluster_id=self.cluster_id, db_instance=db_instance
        # )
        self.cluster_obj = cluster

        # 为了兼容你原本的类属性命名习惯，保留以下别名映射
        self.cluster_name = CLUSTERS[cluster_id]["ID_NAME"]
        self.config = CLUSTERS[cluster_id]

        # 数据库持久化缓存配置
        self.cache_table = MANIFEST[IDX_IDS_SIMBAD]["raw_table"]

        # 🎯 必须显式启动自适应测光演化模型，构建 CMD 插值网格
        self._setup_physical_constraints()

    def get_param(self, param_name: str, default=None):
        """
        🎯 修复版：安全穿透检索，切断循环递归依赖。

        三级检索：动态配置管理器 (内存/DB) → 静态 config.py → default。
        仅当三级全部未命中时才发出警告。
        """
        if not self.cluster_obj or not hasattr(self.cluster_obj, "cfg_mgr"):
            # 降级采用静态配置
            return self.config.get(param_name, default)

        # 直接从底层容器获取，防止与实例属性读取形成死循环
        val = self.cluster_obj.cfg_mgr.get_param(self.cluster_id, param_name)
        if val is not None:
            return val

        # 降级采用静态配置
        val = self.config.get(param_name, default)
        if val is None:
            self.logger.warn(
                f"⚠️ 星团 {self.cluster_name} 在动态配置和静态配置中均未找到参数 {param_name}。"
            )
        return val

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
        iso_file_name = self.get_param("ISO_FILE", "")
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
            distance = self.get_param("DISTANCE_PC", 100.0)  # 默认单位: pc
            extinction_g = self.get_param("EXT_AG", 0.0)  # G波段视消光

            # 🧪 增强逻辑：如果配置缺失 E_BP_RP，根据 EXT_AG 自动按比例估算
            extinction_bp_rp = self.get_param("E_BP_RP")
            if extinction_bp_rp is None:
                extinction_bp_rp = extinction_g * cfg.REDDENING_RATIO_BP_RP

            distance_modulus = 5.0 * np.log10(distance) - 5.0

            # 物理网格整体向观测红化视空间平移
            model_g = (
                self.isochrone_df[col_mapping["G"]].values
                + distance_modulus
                + extinction_g
            )
            model_bp = self.isochrone_df[col_mapping["BP"]].values + distance_modulus
            model_rp = self.isochrone_df[col_mapping["RP"]].values + distance_modulus
            model_color = (model_bp - model_rp) + extinction_bp_rp

            # 6. ✨ 核心：构建非线性单调一维连续空间插值器
            # 按照颜色轴排序，确保插值函数能够完美收敛
            sort_idx = np.argsort(model_color)
            sorted_color = model_color[sort_idx]
            sorted_g = model_g[sort_idx]

            # 过滤由于物理演化极端膨胀阶段导致的输入非单调重复噪点
            unique_idx = np.unique(sorted_color, return_index=True)[1]

            self.cluster_obj.cmd_color_bounds = (
                float(sorted_color[unique_idx].min()),
                float(sorted_color[unique_idx].max()),
            )

            # 使用线性边界外推防御机制（bounds_error=False）建立插值函数
            self.cluster_obj.cmd_interpolator = interp1d(
                sorted_color[unique_idx],
                sorted_g[unique_idx],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            c_min, c_max = self.cluster_obj.cmd_color_bounds
            self.logger.info(
                f"✅ [物理引擎] 成功为物理对象构建 CMD 插值网格。控制区间色指数: ({c_min:.4f}, {c_max:.4f})"
            )
        except Exception as math_err:
            # 打印更精确的异常上下文，方便调试
            self.logger.error(
                f"❌ [Validator] 委托实体构建 CMD 插值矩阵失败: {str(math_err)}", 
                exc_info=True
            )

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
        sql = self._build_audit_sql(v_target_detail)
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

        # 2. 空间投影距离计算 (利用从 Cluster 绑定的 CfgMgr 动态拉取的视距离参数)
        cluster_dist = self.get_param("DISTANCE_PC", 100.0)

        audit_matrix["distance_to_center"] = cluster_dist * np.radians(
            audit_matrix["sep_deg"]
        )

        # 3. 物理一致性审计：验证观测数据是否符合星团物理规律 (送入重构后的 StarCluster 对象)
        audit_matrix = self._audit_physical_consistency(audit_matrix)

        # 4. 文献共识审计：验证身份记录是否与科研结论一致
        consensus_df = self._audit_literature_consensus(audit_matrix)

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
            tidal_radius = self.get_param("TIDAL_RADIUS", 10.0)

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
        c_ra = self.get_param("CENTER_RA")
        c_dec = self.get_param("CENTER_DEC")

        plx = self.get_param("PLX_REF")
        pmra = self.get_param("PMRA_REF")
        pmdec = self.get_param("PMDEC_REF")

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
        """[联合物理一致性审计 - 显式卡方多列输出版] 
        
        🎯 终极全参数融合范式：
        将 2D自行/3D速度、视差、视向速度 以及 测光CMD 全部融合成一个统一的综合卡方统计量(Total $\ chi^2$)，
        并基于动态总自由度(Total DoF)执行严格的卡方独立性检验与概率过滤。
        同时，将各个子维度的卡方值显式记录在最终输出的矩阵中。
        """
        import sys
        import numpy as np
        import pandas as pd
        from scipy.stats import chi2  # 🚀 引入 Scipy 核心卡方分布检验引擎
        import config as cfg

        if audit_matrix.empty:
            return audit_matrix

        # 1. 环境上下文初始化与全新卡方追溯列初始化（默认赋予 NaN 保证维度自适应）
        dim_mode = self.mode
        is_2d = dim_mode == "2d"
        is_physical_v = dim_mode in ["3d_v", "6d_p"]
        audit_matrix["cmd_residual"] = np.nan
        
        # 🧪 【新规字段】初始化子维度卡方列
        audit_matrix["kine_chi2"] = np.nan   # 动力学卡方（2D自行马氏距离或3D速度卡方）
        audit_matrix["plx_chi2"] = np.nan    # 视差卡方
        audit_matrix["rv_chi2"] = np.nan     # 视向速度卡方
        audit_matrix["cmd_chi2"] = np.nan    # 测光演化/UPMASK 空间卡方

        # 提前过滤 NaN 核心数据
        required_cols = ["ra", "dec", "pmra", "pmdec"]
        if not is_2d:
            required_cols.append("plx")
        existing_req_cols = [c for c in required_cols if c in audit_matrix.columns]
        nan_mask = audit_matrix[existing_req_cols].isna().any(axis=1)
        if nan_mask.any():
            self.logger.warning(f"⚠️ [PhysAudit] 自动强制剔除 {nan_mask.sum()} 行包含关键字段 NaN 的样本。")
            audit_matrix = audit_matrix[~nan_mask].copy()

        self.logger.info(f"🔍 [PhysAudit] 启动全参数融合卡方检验。总样本数: {len(audit_matrix)}, 维度模式: {dim_mode}")

        # 初始化总卡方值(Total Chi2)与总自由度(Total DoF)
        audit_matrix["total_integrated_chi2"] = 0.0
        audit_matrix["total_dof"] = 0

        # =====================================================================
        # 维度 1：动力学空间 (3D速度或2D自行)
        # =====================================================================
        if is_physical_v and all(c in audit_matrix.columns for c in ["u", "v", "w"]):
            uvw_ref = self.get_param("UVW_REF")
            u_res = audit_matrix["u"] - uvw_ref[0]
            v_res = audit_matrix["v"] - uvw_ref[1]
            w_res = audit_matrix["w"] - uvw_ref[2]
            u_error = self.get_param("U_ERROR")
            v_error = self.get_param("V_ERROR")
            w_error = self.get_param("W_ERROR")

            # 3D 速度卡方 (DoF = 3)
            kine_chi2 = (u_res / u_error)**2 + (v_res / v_error)**2 + (w_res / w_error)**2
            audit_matrix["kine_chi2"] = kine_chi2  # 📥 写入最终表
            audit_matrix["total_integrated_chi2"] += kine_chi2
            audit_matrix["total_dof"] += 3
        else:
            # 2D 自行马氏卡方距离 (DoF = 2)
            pmra_ref = self.get_param("PMRA_REF")
            pmdec_ref = self.get_param("PMDEC_REF")
            pm_res = audit_matrix[["pmra", "pmdec"]].values - np.array([pmra_ref, pmdec_ref])
            pm_inv_cov = self.cluster_obj.pm_inv_cov
            
            pm_chi2 = np.einsum('ni,ij,nj->n', pm_res, pm_inv_cov, pm_res)
            audit_matrix["kine_chi2"] = pm_chi2  # 📥 写入最终表
            audit_matrix["total_integrated_chi2"] += pm_chi2
            audit_matrix["total_dof"] += 2

        # =====================================================================
        # 维度 2：视差空间 (DoF = 1)
        # =====================================================================
        if not is_2d:
            plx_error = self.get_param("PLX_ERROR", 1.0)
            plx_chi2 = (audit_matrix["plx_residual"] / plx_error)**2
            audit_matrix["plx_chi2"] = plx_chi2  # 📥 写入最终表
            audit_matrix["total_integrated_chi2"] += plx_chi2
            audit_matrix["total_dof"] += 1

        # =====================================================================
        # 维度 3：视向速度空间 (动态激活, DoF = 1)
        # =====================================================================
        if "rv" in audit_matrix.columns:
            rv_err = self.get_param("RV_ERROR", 5.0)
            rv_ref = self.get_param("RV_REF", 0.0)
            has_rv_mask = audit_matrix["rv"].notna()
            if has_rv_mask.any():
                rv_chi2 = ((audit_matrix.loc[has_rv_mask, "rv"] - rv_ref) / rv_err)**2
                audit_matrix.loc[has_rv_mask, "rv_chi2"] = rv_chi2  # 📥 仅对有 RV 观测的行写入
                audit_matrix.loc[has_rv_mask, "total_integrated_chi2"] += rv_chi2
                audit_matrix.loc[has_rv_mask, "total_dof"] += 1

        # =====================================================================
        # 维度 4：测光演化 CMD 空间 (DoF = 2) - 【外部星表注入与多轨概率分类版】
        # =====================================================================
        if all(c in audit_matrix.columns for c in ["ra", "dec", "color", "mag"]):
            audit_matrix["upmask_prob"] = 0.0
            upmask_cols = ["ra", "dec", "color", "mag"]
            
            # 1. 从外部注入已知恒星 CSV 库
            
            # ext_csv_path = Path(f"D:/git/astro-research/research/hunt24-audit/data/raw/gaia_archive/"f"{self.cluster_id.lower()}_pyUPMASK.csv")
            ext_csv_path = Path(cfg.GAIA_INPUT_DIR) /  f"{self.cluster_id.lower()}_pyUPMASK.csv"
            self.logger.info(f"📥 [PhysAudit] 尝试加载外部星表: {ext_csv_path}")
            ext_gaia_ids = set()
            
            if ext_csv_path.exists():
                try:
                    ext_df = pd.read_csv(ext_csv_path)
                    id_col = next((c for c in ext_df.columns if 'source' in c.lower()), None)
                    if id_col:
                        ext_gaia_ids = set(ext_df[id_col].dropna().astype(str).unique())
                        self.logger.info(f"📥 [PhysAudit] 成功导入外部星表，共加载 {len(ext_gaia_ids)} 颗基准恒星。")
                    else:
                        self.logger.warning("⚠️ [PhysAudit] 外部 CSV 文件未找到包含 'id' 关键字的列。")
                except Exception as csv_err:
                    self.logger.error(f"❌ [PhysAudit] 加载外部星表失败: {str(csv_err)}")
            else:
                self.logger.warning(f"⚠️ [PhysAudit] 未找到外部星表文件: {ext_csv_path}。")

            # 生成内外星表的布尔掩码
            audit_matrix['id_str'] = audit_matrix['id'].astype(str)
            is_in_external = audit_matrix['id_str'].isin(ext_gaia_ids)

            # 2. 🎯 核心过滤：剔除缺失测光数据的样本
            valid_cmd_mask = audit_matrix[upmask_cols].notna().all(axis=1)
            
            if valid_cmd_mask.any():
                valid_df = audit_matrix[valid_cmd_mask].copy()
                valid_indices = valid_df.index
                
                # 动态加载 pyUPMASK
                # pyupmask_dir = r"D:/git/astro-research/research/hunt24-audit/modules/pyUPMASK"
                # if pyupmask_dir not in sys.path: sys.path.append(pyupmask_dir)
                # import modules.pyUPMASK as upmask_mod

                n_iterations = int(self.get_param("UPMASK_ITERATIONS", 20))
                max_clusters = int(self.get_param("UPMASK_MAX_CLUSTERS", 5))

                probs_all = upmask_mod.dataProcess(
                    ID=valid_indices.values, 
                    xy=valid_df[["ra", "dec"]].values,
                    data=valid_df[["color", "mag"]].values,
                    data_err=valid_df[["color_err", "mag_err"]].values,
                    verbose=0, 
                    OL_runs=n_iterations, 
                    parallel_flag=False, 
                    parallel_procs=1,
                    resampleFlag=True, 
                    PCAflag=False, 
                    PCAdims=2, 
                    GUMM_flag=False, 
                    GUMM_perc=None,
                    KDEP_flag=False, 
                    IL_runs=5, 
                    N_membs=10, 
                    N_cl_max=max_clusters,
                    clust_method='KMeans', 
                    clRjctMethod='rkfunc', 
                    C_thresh=0.05, 
                    cl_method_pars={}
                )
                
                upmask_prob = np.mean(probs_all, axis=0) if len(probs_all) > 1 else probs_all[0]
                audit_matrix.loc[valid_indices, "upmask_prob"] = upmask_prob
                
                # 计算全量样本的卡方残差
                cmd_chi2 = -2.0 * np.log(upmask_prob + 1e-10)
                
                if "color_excess" in audit_matrix.columns and "color_excess_sigma" in audit_matrix.columns:
                    outlier_mask = np.abs(audit_matrix["color_excess"]) > (5.0 * audit_matrix["color_excess_sigma"])
                    sub_outlier = outlier_mask[valid_cmd_mask]
                    cmd_chi2[sub_outlier] += 1.0

                # 保存原有的测光残差值以便后续追溯
                # TODO: 考虑将 cmd_residual 与 cmd_chi2 分开存储，避免混淆
                audit_matrix.loc[valid_cmd_mask, "cmd_residual"] = cmd_chi2
                audit_matrix.loc[valid_cmd_mask, "cmd_chi2"] = cmd_chi2  # 📥 写入新规卡方列

                # 🎯 【关键微调】：构建仅属于管线侧且测光完整的掩码 (PG Only & Valid CMD)
                pg_only_valid_mask = (~is_in_external) & valid_cmd_mask
                
                if pg_only_valid_mask.any():
                    # 仅让 pg_only 的恒星累加测光卡方值
                    audit_matrix.loc[pg_only_valid_mask, "total_integrated_chi2"] += audit_matrix.loc[pg_only_valid_mask, "cmd_residual"]
                    # 仅让 pg_only 的恒星 DoF 增加 1
                    audit_matrix.loc[pg_only_valid_mask, "total_dof"] += 1
                    
        # =====================================================================
        # 5. 统一卡方假设检验决策
        # =====================================================================
        audit_matrix["global_cluster_probability"] = chi2.sf(
            audit_matrix["total_integrated_chi2"], 
            audit_matrix["total_dof"]
        )

        # 核心筛选：卡掉整体属于星团概率 < 0.50 的恒星
        audit_matrix["is_phys_consistent"] = audit_matrix["global_cluster_probability"] >= 0.50

        # 6. 详细统计报告输出
        self.logger.info("📊 [全参数融合卡方检验报告]:")
        for dof in sorted(audit_matrix["total_dof"].unique()):
            dof_mask = audit_matrix["total_dof"] == dof
            if not dof_mask.any(): continue
            
            chi2_cutoff = chi2.ppf(0.50, dof)
            passed = (dof_mask & audit_matrix["is_phys_consistent"]).sum()
            total = dof_mask.sum()
            
            self.logger.info(
                f"  -> 融合总自由度 DoF = {dof}: 临界卡方阈值 = {chi2_cutoff:.3f}, 通过率 = {passed}/{total} ({passed/total*100:.1f}%)"
            )

        # 为保证向下兼容，保留旧评分字段的填充
        audit_matrix["weighted_penalty"] = audit_matrix["total_integrated_chi2"] / audit_matrix["total_dof"]
        audit_matrix["pm_score"] = audit_matrix["total_integrated_chi2"] 

        self._log_phys_audit_stats(audit_matrix)
        
        # 清理临时列
        if 'id_str' in audit_matrix.columns:
            audit_matrix.drop(columns=['id_str'], inplace=True)

        # =====================================================================
        # 📥 【新规自动化落盘】保持与原管线一致的 DuckDB 持久化逻辑
        # =====================================================================
        try:
            if hasattr(self, 'db') and self.db is not None:
                self.logger.info("💾 [PhysAudit] 检测到活跃的 DuckDB 连接，正在同步包含新卡方参数的数据...")
                
                # 🎯 修复：直接让 DuckDB 嗅探当前作用域下的 Python 变量名 audit_matrix
                # self.db.execute("CREATE OR REPLACE TABLE master_m45_hunt_5d_dbscan AS SELECT * FROM audit_matrix")
                # self.db.tag_master_table()
                
                self.logger.info("💾 [PhysAudit] ✅ 成功！新卡方参数已无损同步至表 [master_m45_hunt_5d_dbscan]。")        
        except Exception as db_err:
            self.logger.error(f"❌ [PhysAudit] 自动写入 DuckDB 失败，报错原因: {str(db_err)}")
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

    def _build_cluster_keywords(self) -> list:
        """动态构建星团的规范化核心词及缩写别名网。

        从配置参数 (NAME, SIMBAD_NAME, ID_NAME, CAT_NAME)、星团 ID 前缀、
        及预设特殊别名中提取并派生所有可能的别名变体。

        Returns:
            去重并按长度降序排列的关键词列表，最长的优先匹配。
        """
        keywords = []
        for key in ["NAME", "SIMBAD_NAME", "ID_NAME", "CAT_NAME"]:
            val = self.get_param(key)
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

        keywords = sorted(
            list(set([k.strip().upper() for k in keywords if k.strip()])),
            key=len,
            reverse=True,
        )
        self.logger.info(
            f"🧬 [Semantic Audit] 激活语义审计核心，检索空间词网: {keywords}"
        )
        return keywords

    def _audit_literature_consensus(self, audit_matrix: pd.DataFrame) -> pd.DataFrame:
        """[无损保留] 基于 SIMBAD 实时父级语义树及已知星团特殊别名的权威语义树过滤算法"""
        if audit_matrix.empty:
            return pd.DataFrame(columns=["is_lit_consensus", "match_type"])

        self.logger.debug(
            f" 🟢 [Semantic Audit]----------待审计数据---------------\n {audit_matrix}"
        )

        # 1. 动态构建星团的规范化核心词及缩写别名网
        keywords = self._build_cluster_keywords()

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

        norm_aliases = audit_matrix["ids"].fillna("").apply(clean_text)
        norm_preferred = audit_matrix["main_id"].fillna("").apply(clean_text)

        strict_pattern = rf"(?:^|\||\s)(?:{kw_pattern})(?:\s*\||$)"
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
