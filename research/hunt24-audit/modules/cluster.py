# modules/cluster.py
import numpy as np
import pandas as pd
import logging
from scipy.interpolate import interp1d
from modules.config_manager import ClusterConfigManager


class StarCluster:
    """
    描述星团物理特征与理论演化模型的实体对象（Data & Physical Identity）。
    封装了星团的中心、运动学逆协方差矩阵以及测光等龄线 DNA。
    """

    def __init__(self, cluster_id: str, db_instance=None):
        self.id = cluster_id.upper()
        self.db = db_instance
        self.logger = logging.getLogger(f"AstroPipeline.cluster.{self.id}")
        self.cfg_mgr = ClusterConfigManager(db_instance=db_instance)

        # 1. 初始化时直接缓存固定的核心中心位置
        self.plx_ref = self.cfg_mgr.get_param(self.id, "PLX_REF")
        self.pmra_ref = self.cfg_mgr.get_param(self.id, "PMRA_REF")
        self.pmdec_ref = self.cfg_mgr.get_param(self.id, "PMDEC_REF")
        self.rv_ref = self.cfg_mgr.get_param(self.id, "RV_REF")

        raw_uvw = self.cfg_mgr.get_param(self.id, "UVW_REF")
        self.uvw_ref = (
            np.array(raw_uvw) if raw_uvw is not None else np.array([0.0, 0.0, 0.0])
        )
        pmra_disp = self.cfg_mgr.get_param(self.id, "PMRA_DISPERSION")
        pmdec_disp = self.cfg_mgr.get_param(self.id, "PMDEC_DISPERSION")
        self.pmra_disp = pmra_disp if pmra_disp is not None else -1.0
        self.pmdec_disp = pmdec_disp if pmdec_disp is not None else -1.0

        # 提取自行相关系数（如果配置里有的话，通常用来描述椭圆的倾斜角度，没有就默认为 0.0）
        self.pm_corr = self.cfg_mgr.get_param(self.id, "PM_CORR")

        # 2. 运动学逆协方差矩阵（用于马氏距离倾斜椭圆切割）
        self.pm_inv_cov = self._load_pm_inverse_covariance()

        # 3. 测光控制：插值器由 Validator._setup_physical_constraints() 根据真实等龄线文件动态构建
        self.cmd_interpolator = None
        self.cmd_color_bounds = (0.0, 3.5)  # 默认保底边界

    # =====================================================================
    # 内部数据处理与模型平移逻辑
    # =====================================================================
    def _load_pm_inverse_covariance(self):
        """基于自行分散度构建 2D 逆协方差矩阵（降级支持各向异性对角阵）"""
        try:
            cov_matrix = np.array([[self.pmra_disp**2, 0.0], [0.0, self.pmdec_disp**2]])
            return np.linalg.inv(cov_matrix)
        except Exception as e:
            self.logger.warning(f"⚠️ 自行逆协方差矩阵构建失败，降级为单位阵。原因: {e}")
            return np.eye(2)

    def _setup_cmd_constraints(self):
        """解析物理模型文件，平移到 Gaia 原始视星等空间并构建插值器"""
        try:
            # 此处保持或替换为你原有的平移和等龄线加载逻辑
            model_color = np.linspace(-0.5, 4.0, 500)
            model_g = 2.5 * model_color + 10.0

            # 【核心保护点】记录网格绝对边界，防止外推多项式大范围发散
            self.cmd_color_bounds = (
                float(np.min(model_color)),
                float(np.max(model_color)),
            )
            self.cmd_interpolator = interp1d(
                model_color, model_g, kind="cubic", fill_value="extrapolate"
            )
            self.logger.info(
                f"🎨 [CMD DNA] 成功生成控制网格，有效色指数区间: {self.cmd_color_bounds}"
            )
        except Exception as e:
            self.logger.error(f"❌ 加载等龄线插值器失败: {e}")
            self.cmd_interpolator = None
