# modules/cluster.py
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from modules.config_manager import ClusterConfigManager
import config as cfg

class StarCluster:
    """
    描述星团物理特征与理论演化模型的实体对象（Data & Physical Identity）。
    
    该类作为星团物理状态的“单一事实来源”，负责：
    1. 管理和缓存核心物理参数 (RA, Dec, Plx, PM, RV)。
    2. 维护运动学逆协方差矩阵（用于成员判定）。
    3. 解析等龄线理论模型并构建 CMD 插值 DNA。
    """

    # 定义需要自动从配置同步到内存属性的核心字段
    CORE_PHYSICAL_PARAMS = [
        "CENTER_RA", "CENTER_DEC", "PLX_REF", "PLX_ERROR",
        "PMRA_REF", "PMDEC_REF", "PMRA_ERROR", "PMDEC_ERROR",
        "PMRA_DISPERSION", "PMDEC_DISPERSION", "PM_CORR",
        "RV_REF", "RV_ERROR", "DISTANCE_PC", "ISO_FILE", "E_BP_RP", "EXT_AG"
    ]

    def __init__(self, cluster_id: str, db_instance=None, param_source=None):
        self.id = cluster_id.upper()
        self.db = db_instance
        self.param_source = param_source
        self.logger = logging.getLogger(f"AstroPipeline.cluster.{self.id}")
        
        # 1. 绑定配置管理器（处理 DB 与静态 Config 的分层检索）
        self.cfg_mgr = ClusterConfigManager(db_instance=db_instance, param_source=param_source)

        # 2. 初始化关键容器
        self.pm_inv_cov = None
        self.cmd_interpolator = None
        self.cmd_color_bounds = (0.0, 3.5) # 默认安全边界

        # 3. 初始同步：从底座装载物理参数
        self.logger.info(f"🌌 [Domain] 正在初始化星团实体模型: {self.id}")
        self._hydrate_from_config()

    def get_param(self, param_name: str, default=None):
        """[透传接口] 通过配置管理器安全获取参数（支持动态覆盖）"""
        return self.cfg_mgr.get_param(self.id, param_name, default)

    # =====================================================================
    # 核心物理状态同步 (Hydration)
    # =====================================================================

    def _hydrate_from_config(self) -> bool:
        """从底座配置管理器中将最新的物理参数同步到对象的内存属性中"""
        self.logger.debug(f"🧬 [Physical] 正在执行物理参数同步 (Hydration)...")
        try:
            # 1. 自动化属性映射 (例如: PLX_REF -> self.plx_ref)
            for param in self.CORE_PHYSICAL_PARAMS:
                val = self.get_param(param)
                setattr(self, param.lower(), val)

            # 2. 处理 UVW 矢量参考
            raw_uvw = self.get_param("UVW_REF")
            self.uvw_ref = np.array(raw_uvw) if raw_uvw is not None else np.zeros(3)

            # 3. 重新构建运动学逆协方差矩阵
            self.pm_inv_cov = self._load_pm_inverse_covariance()

            # 4. 触发测光演化 DNA (等龄线) 的解析与构建
            self._setup_cmd_constraints()
            
            return self.plx_ref is not None
        except Exception as e:
            self.logger.error(f"❌ [Physical] 参数装载并同步至内存状态时崩溃: {e}", exc_info=True)
            return False

    def load_or_reconstruct_parameters(self, param_source: str = "file") -> bool:
        """
        🚀 [富领域行为] 统一负责星团物理属性的装载或自适应重建。
        
        Args:
            mode: "db" 代表启动高精度物理资产反演引擎进行重建；
                  "file" 代表从静态配置文件直接加载。
        """
        if param_source == "db":
            self.logger.info(f"🧬 [Domain] 触发 [{self.id}] 相空间物理参数的自适应反演与自我重建...")
            # 让实体对象自己调用底座去重建自己
            recon_res = self.cfg_mgr.reconstruct_cl_params_from_db(self.id)
            if not recon_res:
                self.logger.warning("⚠️ [Domain] 历史数据重建失败，将降级加载静态参数。")
            
            # 重建完成后，必须刷新当前对象的内部物理状态
            return self._hydrate_from_config()
        else:
            # 常规模式，直接同步配置
            return self._hydrate_from_config()

    # =====================================================================
    # 内部模型计算与构建逻辑
    # =====================================================================

    def _load_pm_inverse_covariance(self):
        """基于自行弥散度构建 2D 逆协方差矩阵"""
        self.logger.debug(f"📐 [Physical] 正在构建运动学协方差矩阵...")
        try:
            # 优先检查是否有显式的全矩阵定义
            full_cov = self.get_param("PM_COVARIANCE_MATRIX")
            if full_cov is not None:
                return np.linalg.inv(np.array(full_cov))

            # 降级：利用弥散度构建各向异性对角阵
            ra_disp = getattr(self, "pmra_dispersion", None) or 1.0
            dec_disp = getattr(self, "pmdec_dispersion", None) or 1.0
            
            # 考虑相关系数 (pm_corr)
            corr = getattr(self, "pm_corr", 0.0) or 0.0
            cov_matrix = np.array([
                [ra_disp**2, corr * ra_disp * dec_disp],
                [corr * ra_disp * dec_disp, dec_disp**2]
            ])
            return np.linalg.inv(cov_matrix)
        except Exception as e:
            self.logger.warning(f"⚠️ [Physical] 逆协方差矩阵构建失败，降级为单位阵。原因: {e}")
            return np.eye(2)

    def _setup_cmd_constraints(self):
        """解析理论模型文件，应用距离模数平移并构建 CMD 插值器"""
        iso_file = getattr(self, "iso_file", None)
        if not iso_file:
            self.logger.warning(f"⚠️ [Model] 未配置 ISO_FILE，跳过测光演化模型构建。")
            return

        # 路径解析：支持绝对路径及 data/raw/oapd 相对路径
        iso_path = Path(iso_file)
        if not iso_path.is_absolute():
            iso_path = cfg.DATA_DIR / "raw" / "oapd" / iso_file

        if not iso_path.exists():
            self.logger.error(f"❌ [Model] 找不到等龄线模型文件: {iso_path}")
            return

        self.logger.info(f"🧬 [Model] 正在解析等龄线模型: {iso_path.name}")
        try:
            # 1. 稳健读取：提取注释行末尾作为表头
            with open(iso_path, "r") as f:
                header_line = ""
                for line in f:
                    if line.startswith("#"): header_line = line
                    else: break
            
            cols = header_line.lstrip("#").split()
            df_iso = pd.read_csv(iso_path, sep=r"\s+", comment="#", names=cols)

            # 2. 识别 Gaia 波段字段
            g_col = next(c for c in df_iso.columns if c.lower() in ["gmag", "g"])
            bp_col = next(c for c in df_iso.columns if c.lower() in ["g_bpmag", "bpmag"])
            rp_col = next(c for c in df_iso.columns if c.lower() in ["g_rpmag", "rpmag"])

            # 3. 物理空间平移 (距离模数 + 红化修正)
            dist_pc = getattr(self, "distance_pc", 100.0)
            ext_ag = getattr(self, "ext_ag", 0.0)
            ebprp = getattr(self, "e_bp_rp", 0.0)
            
            dist_mod = 5.0 * np.log10(dist_pc) - 5.0
            model_g = df_iso[g_col].values + dist_mod + ext_ag
            model_color = (df_iso[bp_col] - df_iso[rp_col]).values + ebprp

            # 4. 构建单调三次样条插值器
            sort_idx = np.argsort(model_color)
            u_color, u_idx = np.unique(model_color[sort_idx], return_index=True)
            u_g = model_g[sort_idx][u_idx]

            self.cmd_color_bounds = (float(u_color.min()), float(u_color.max()))
            self.cmd_interpolator = interp1d(
                u_color, u_g, kind="cubic", bounds_error=False, fill_value="extrapolate"
            )
            self.logger.info(f"✅ [Model] CMD DNA 构建完成。有效区间: {self.cmd_color_bounds}")
        except Exception as e:
            self.logger.error(f"❌ [Model] 构建 CMD 插值器失败: {e}")
            self.cmd_interpolator = None