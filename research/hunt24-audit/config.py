# config.py
"""
hunt24-audit 科学管线全局配置蓝图中枢 v2.1 (重构优化版)

[架构组织分水岭]
========================= 外部开放配置区 =========================
1. 系统路径与环境配置 (Paths & Environment)
2. 科学计算门限与物理常数 (Thresholds & Physics)
3. 星团物理先验配置数据库 (Pure Cluster Physical Database)
4. 文献及标准字段映射中枢 (Schemas, Fields & Naming Adapters)
5. 算法流水线配置与盲搜空间元数据大盘 (Pipeline & GMM Config)

========================= 内部流转架构区 =========================
6. 管线统一多态资产注册架构 (Unified Asset Core Models)
7. 数据资产静态大盘模型定义 (_StaticManifestHolder)
8. 全局单例多态清单容器 (Manifest Container Engine)

========================= 外部调用标准接口 =======================
9. 全局单例标准对外入口网关 (Global Standard API Entry)
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 内部动作模块导入
from modules.actions import StdActions, StxActions, AlnActions

# =================================================================
# 1. 系统路径与环境配置 (Paths & Environment)
# =================================================================
# [说明] 规范整个科学管线的存储拓扑，所有对外输出和输入均在此锚定。
BASE_DIR = Path(__file__).resolve().parent.resolve()
LOG_DIR = (BASE_DIR / "logs").resolve()
DATA_DIR = (BASE_DIR / "data").resolve()
ANALYSIS_DIR = (BASE_DIR / "analysis").resolve()
RESULTS_DIR = (ANALYSIS_DIR / "results").resolve()

# 数据清洗子目录
RAW_DIR = (DATA_DIR / "raw").resolve()
BACKUP_DIR = (DATA_DIR / "backups").resolve()
EXPORT_DIR = (DATA_DIR / "exports").resolve()
INTERNAL_DIR = (DATA_DIR / "internal").resolve()

# 天文科学数据源输入子目录
GAIA_INPUT_DIR = (RAW_DIR / "gaia_archive").resolve()
VIZIER_INPUT_DIR = (RAW_DIR / "vizier").resolve()
SIMBAD_INPUT_DIR = (RAW_DIR / "simbad").resolve()
OAPD_INPUT_DIR = (RAW_DIR / "oapd").resolve()
DOWNLOAD_DIR = RAW_DIR

# 外部数据源（Gaia Archive）云端认证凭证
GAIA_USER = os.getenv("GAIA_USER", "jli21")
GAIA_PWD = os.getenv("GAIA_PWD")


# =================================================================
# 2. 科学计算门限与物理常数 (Thresholds & Physics)
# =================================================================
# [说明] 核心天体物理筛选门限。业务层在进行成员身份判定、残差过滤时直接调取。
MEMBER_SAMPLE_THRESHOLD = 0.2
GOLDEN_SAMPLE_THRESHOLD = 0.8

AUDIT_PROB_HIGH = 0.7            # 成员身份判定高门限（置信度高）
AUDIT_PROB_LOW = 0.3             # 成员身份判定低门限（视作背景噪点）
AUDIT_RUWE_LIMIT = 1.4           # Gaia 天体测量质量极限门限（超过此值天体测量不可靠）
AUDIT_PLX_RESIDUAL_LIMIT = 1.0   # 视差残差允许度精度门限 (单位: mas)
AUDIT_MAG_LIMIT_HUNT24 = 19.0    # Hunt2024 文献深度参考视星等极限线

# 天体物理验证权重与细分容忍度配置
PHYS_VERIFY_WEIGHTS = {
    "pm": 0.4,                   # 自行（Proper Motion）权重
    "plx": 0.4,                  # 视差（Parallax）权重
    "cmd": 0.2,                  # 色等图（Color-Magnitude Diagram）权重
}
PHYS_VERIFY_PENALTY_LIMIT = 1.1
PHYS_LIT_PM_LIMIT = 1.5
PHYS_LIT_CMD_LIMIT = 3.0
REDDENING_RATIO_BP_RP = 0.52     # E(BP-RP) / A_G 消光比例系数


# =================================================================
# 3. 星团物理先验配置数据库 (Pure Cluster Physical Database)
# =================================================================
# [说明] 目标研究星团的已知物理先验参数仓库。对外提供纯粹的天体物理常数查询。

@dataclass
class ClusterConfig:
    KEY_ID: str = ""
    CENTER_RA: float = 0.0          # 星团中心赤经
    CENTER_DEC: float = 0.0         # 星团中心赤纬
    RADIUS: float = 0.0             # 整体搜索半径
    RA_MIN: float = 0.0
    RA_MAX: float = 0.0
    DEC_MIN: float = 0.0
    DEC_MAX: float = 0.0
    MAX_MAG: float = 21.0
    CORE_RADIUS: float = 0.0         # 核心半径
    HALF_MASS_RADIUS: float = 0.0    # 半质量半径
    R_HALF_LIGHT: float = 0.0       # 半光半径
    TIDAL_RADIUS: float = 0.0        # 潮汐半径
    DISTANCE_PC: float = 0.0         # 距离 (pc)
    DISTANCE_MODULUS: float = 0.0    # 距离模数
    AV: float = 0.0                 # 视向消光
    EXT_AG: float = 0.0
    E_BP_RP: float = 0.0            # 色余
    PLX_REF: float = 0.0            # 视差参考先验值
    PLX_ERROR: float = 0.0
    PMRA_REF: float = 0.0           # 赤经方向自行先验值
    PMDEC_REF: float = 0.0          # 赤纬方向自行先验值
    PMRA_DISPERSION: float = 0.0
    PMDEC_DISPERSION: float = 0.0
    PM_CORR: float = 0.0
    PM_RADIUS: float = 0.0
    RV_REF: float = 0.0             # 视向速度先验值
    RV_ERROR: float = 0.0
    UVW_REF: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # 空间速度三维矢量
    V_ERROR: float = 2.0
    UVW_ERROR: float = 2.0
    U_ERROR: float = 2.0
    W_ERROR: float = 1.0
    CMD_REF: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    CMD_DEV: float = 0.8
    KINE_SCORE_LIMIT: float = 2.0
    SEED_RADIUS: float = 0.0         # 种子选取空间半径
    SEED_PLX_LIM: float = 0.0        # 种子选取视差容忍度
    SEED_MAX_MAG: float = 0.0        # 种子星等上限值
    SEED_MAX_RUWE: float = 1.4       # 种子天体测量质量上限
    SEED_PM_LIM: float = 0.0         # 种子自行容忍度

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    @property
    def id(self) -> str:
        return self.KEY_ID.lower()

    def keys(self):
        return list(self.__dict__.keys()) + ["id", "CAT_NAME"]

    def __getitem__(self, key):
        if key == "id":
            return self.id
        if key == "CAT_NAME":
            return self.KEY_ID
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    @property
    def FIELD_IDX(self) -> str:
        return f"{self.KEY_ID.lower()}_field"

    @property
    def SEED_IDX(self) -> str:
        return f"{self.KEY_ID.lower()}_seeds_field"

    @property
    def ISO_FILE(self) -> str:
        return CLUSTER_REGISTRY.get(self.KEY_ID).ISO_FILE if CLUSTER_REGISTRY.get(self.KEY_ID) else "default.dat"

    @property
    def ID_NAME(self) -> str:
        meta = CLUSTER_REGISTRY.get(self.KEY_ID)
        return meta.ID_NAME if meta else self.KEY_ID.lower()

    @property
    def CAT_NAME(self) -> str:
        meta = CLUSTER_REGISTRY.get(self.KEY_ID)
        return meta.CAT_NAME if meta else self.KEY_ID

    @property
    def SIMBAD_NAME(self) -> str:
        meta = CLUSTER_REGISTRY.get(self.KEY_ID)
        return meta.SIMBAD_NAME if meta else self.KEY_ID

    @property
    def NAME(self) -> str:
        meta = CLUSTER_REGISTRY.get(self.KEY_ID)
        return meta.NAME if meta else self.KEY_ID

    def get_cat_name(self, idx_data: str, manifest=None) -> str:
        adapter = CATALOG_NAMING_ADAPTER
        override_name = adapter.get(idx_data, {}).get(self.id)
        if override_name:
            return override_name
        return self.KEY_ID


CLUSTERS = {
    "M45": ClusterConfig(CENTER_RA=56.61398997432307, CENTER_DEC=24.09029596042996, RADIUS=17.78, RA_MIN=44.0, RA_MAX=66.0, DEC_MIN=16.0, DEC_MAX=36.0, CORE_RADIUS=1.3, HALF_MASS_RADIUS=3.5, R_HALF_LIGHT=3.0, TIDAL_RADIUS=10.0, DISTANCE_PC=136.2, DISTANCE_MODULUS=5.66, EXT_AG=0.12, E_BP_RP=0.06, PLX_REF=7.329881851789656, PLX_ERROR=0.5, PMRA_REF=19.816076644561296, PMDEC_REF=-45.02481613280063, PMRA_DISPERSION=1.5, PMDEC_DISPERSION=1.2, PM_CORR=-0.03740541790723895, PM_RADIUS=3.0, RV_REF=5.63, RV_ERROR=5.0, UVW_REF=np.array([-6.05, -28.02, -14.34]), UVW_ERROR=2.0, U_ERROR=2.2, V_ERROR=1.6, W_ERROR=1.0, SEED_RADIUS=2.0, SEED_PLX_LIM=1.5, SEED_MAX_MAG=18.0, SEED_MAX_RUWE=1.2),
    "M44": ClusterConfig(CENTER_RA=130.1, CENTER_DEC=19.7, RADIUS=11.90, RA_MIN=120.0, RA_MAX=140.0, DEC_MIN=10.0, DEC_MAX=30.0, CORE_RADIUS=0.8, HALF_MASS_RADIUS=3.9, R_HALF_LIGHT=3.5, TIDAL_RADIUS=12.0, DISTANCE_PC=187.0, DISTANCE_MODULUS=6.36, EXT_AG=0.05, E_BP_RP=0.03, PLX_REF=5.35, PLX_ERROR=0.4, PMRA_REF=-36.0, PMDEC_REF=-12.9, RV_REF=35.0, RV_ERROR=5.0, UVW_REF=np.array([-34.5, -21.2, -6.8]), V_ERROR=2.0, PM_RADIUS=4.0, CMD_DEV=0.6, SEED_RADIUS=2.0, SEED_PLX_LIM=1.2, SEED_MAX_MAG=18.0, SEED_MAX_RUWE=1.4),
    "Mel25": ClusterConfig(CENTER_RA=66.75, CENTER_DEC=15.87, RADIUS=59.31, RA_MIN=50.0, RA_MAX=85.0, DEC_MIN=0.0, DEC_MAX=32.0, CORE_RADIUS=2.7, HALF_MASS_RADIUS=4.1, R_HALF_LIGHT=3.1, TIDAL_RADIUS=10.0, DISTANCE_PC=46.7, DISTANCE_MODULUS=3.35, AV=0.02, EXT_AG=0.01, E_BP_RP=0.01, PLX_REF=21.41, PLX_ERROR=2.5, PMRA_REF=101.10, PMDEC_REF=-28.50, RV_REF=39.10, UVW_REF=np.array([-42.24, -19.11, -1.45]), V_ERROR=3.0, RV_ERROR=5.0, PM_RADIUS=12.0, CMD_DEV=0.6, SEED_RADIUS=8.0, SEED_PLX_LIM=2.0, SEED_MAX_MAG=16.0, SEED_MAX_RUWE=1.2),
    "Mel111": ClusterConfig(CENTER_RA=186.6, CENTER_DEC=26.1, RADIUS=42.61, RA_MIN=175.0, RA_MAX=198.0, DEC_MIN=15.0, DEC_MAX=37.0, CORE_RADIUS=1.5, HALF_MASS_RADIUS=4.5, R_HALF_LIGHT=3.8, TIDAL_RADIUS=15.0, DISTANCE_PC=86.0, DISTANCE_MODULUS=4.67, EXT_AG=0.02, E_BP_RP=0.01, PLX_REF=11.60, PLX_ERROR=1.5, PMRA_REF=-12.11, PMDEC_REF=-9.01, RV_REF=-1.0, RV_ERROR=5.0, UVW_REF=np.array([-1.7, -6.1, -1.3]), V_ERROR=2.5, PM_RADIUS=5.0, CMD_DEV=0.6, SEED_RADIUS=5.0, SEED_PLX_LIM=1.5, SEED_MAX_MAG=16.0, SEED_MAX_RUWE=1.2),
    "M67": ClusterConfig(CENTER_RA=132.83, CENTER_DEC=11.82, RADIUS=2.5, RA_MIN=128.0, RA_MAX=138.0, DEC_MIN=7.0, DEC_MAX=17.0, CORE_RADIUS=1.2, HALF_MASS_RADIUS=4.5, R_HALF_LIGHT=3.8, TIDAL_RADIUS=16.0, DISTANCE_PC=850.0, DISTANCE_MODULUS=9.65, EXT_AG=0.10, E_BP_RP=0.05, PLX_REF=1.17, PLX_ERROR=0.2, PMRA_REF=-10.96, PMDEC_REF=-2.94, RV_REF=33.7, RV_ERROR=3.0, UVW_REF=np.array([-21.4, -25.2, -15.1]), V_ERROR=1.5, PM_RADIUS=1.5, CMD_DEV=0.5, SEED_RADIUS=1.5, SEED_PLX_LIM=0.4, SEED_MAX_MAG=20.0, SEED_MAX_RUWE=1.4, SEED_PM_LIM=2.5),
    "M13": ClusterConfig(CENTER_RA=250.42, CENTER_DEC=36.46, RADIUS=3.28, RA_MIN=248.0, RA_MAX=253.0, DEC_MIN=34.5, DEC_MAX=38.5, CORE_RADIUS=1.3, HALF_MASS_RADIUS=3.5, R_HALF_LIGHT=3.2, TIDAL_RADIUS=43.0, DISTANCE_PC=7100.0, DISTANCE_MODULUS=14.25, EXT_AG=0.04, E_BP_RP=0.02, PLX_REF=0.14, PLX_ERROR=0.1, PMRA_REF=-3.18, PMDEC_REF=-2.57, RV_REF=-244.2, RV_ERROR=10.0, UVW_REF=np.array([58.0, -241.0, 10.0]), V_ERROR=10.0, PM_RADIUS=1.0, CMD_DEV=0.4, SEED_RADIUS=0.8, SEED_PLX_LIM=0.5, SEED_MAX_MAG=20.5, SEED_MAX_RUWE=1.4, SEED_PM_LIM=2.0),
    "M41": ClusterConfig(CENTER_RA=101.50, CENTER_DEC=-20.75, RADIUS=2.53, RA_MIN=96.0, RA_MAX=107.0, DEC_MIN=-25.0, DEC_MAX=-15.0, CORE_RADIUS=1.5, HALF_MASS_RADIUS=4.0, R_HALF_LIGHT=3.6, TIDAL_RADIUS=12.0, DISTANCE_PC=710.0, DISTANCE_MODULUS=9.25, EXT_AG=0.05, E_BP_RP=0.03, PLX_REF=1.41, PLX_ERROR=0.3, PMRA_REF=-1.55, PMDEC_REF=-1.05, RV_REF=34.0, RV_ERROR=5.0, UVW_REF=np.array([-10.5, -20.2, -5.1]), V_ERROR=2.0, CMD_DEV=0.6, SEED_RADIUS=2.0, SEED_PLX_LIM=0.9, SEED_MAX_MAG=18.0, SEED_MAX_RUWE=1.4),
}

# 物理先验数据库的 Key 自动绑定赋值
for k, obj in CLUSTERS.items():
    obj.KEY_ID = k


# =================================================================
# 4. 文献架构图谱与标准字段映射中枢 (Schemas, Fields & Naming Adapters)
# =================================================================
# [说明] 统管多源异构星表（VizieR 各文献、Gaia、Simbad）字段别名映射的转换适配。

CATALOG_NAMING_ADAPTER = {
    "zerj": {"M45": "Melotte 22"},
}

# 管线内部清洗后的标准列名映射规范
STD_COLS = {
    "ID": "id", "ID_DR2": "id_dr2", "RA": "ra", "DEC": "dec", "PMRA": "pmra", "PMDEC": "pmdec",
    "PLX": "plx", "MAG": "mag", "COLOR": "color", "RV": "rv", "RUWE": "ruwe", "PROB": "prob",
    "GMM_PROB": "gmm_prob", "REF_PROB": "r_prob", "CLUSTER": "cluster",
}

# Master 主宽表结果中存储的多模态审计标记字段列名
MASTER_COLS = {
    "SEED_TYPE": "seed_type", "DENSITY_TAG": "density_status", "GMM_PROB": "prob",
    "X_MATCH": "x_match_tag", "AUDIT": "audit_status", "AUDIT_NOTE": "audit_note",
}

# 文献来源短标识别名常量
IDX_FIELD_CLUSTER_M45, IDX_FIELD_CLUSTER_M45_SEEDS = "m45_field", "m45_seeds_field"
IDX_FIELD_CLUSTER_M44, IDX_FIELD_CLUSTER_M44_SEEDS = "m44_field", "m44_seeds_field"
IDX_FIELD_CLUSTER_MEL25, IDX_FIELD_CLUSTER_MEL25_SEEDS = "mel25_field", "mel25_seeds_field"
IDX_FIELD_CLUSTER_MEL111, IDX_FIELD_CLUSTER_MEL111_SEEDS = "mel111_field", "mel111_seeds_field"
IDX_FIELD_CLUSTER_M67, IDX_FIELD_CLUSTER_M67_SEEDS = "m67_field", "m67_seeds_field"
IDX_FIELD_CLUSTER_M13, IDX_FIELD_CLUSTER_M13_SEEDS = "m13_field", "m13_seeds_field"
IDX_FIELD_CLUSTER_M41, IDX_FIELD_CLUSTER_M41_SEEDS = "m41_field", "m41_seeds_field"

IDX_CG20, IDX_HEYL, IDX_ZERJ, IDX_RISB, IDX_HUNT = "cg20", "heyl", "zerj", "risb", "hunt"
IDX_DR2IDX, IDX_IDS_SIMBAD, IDX_GMM = "dr2idx", "ids_simbad", "pgmm"

@dataclass(frozen=True)
class ClusterIdentity:
    NAME: str
    SIMBAD_NAME: str
    ID_NAME: str
    CAT_NAME: str
    ISO_FILE: str


CLUSTER_REGISTRY: Dict[str, ClusterIdentity] = {
    "M45": ClusterIdentity(NAME="Pleiades", SIMBAD_NAME="Cl Melotte 22", ID_NAME="melotte_22", CAT_NAME="Melotte_22", ISO_FILE="pleiades_126myr.dat"),
    "M44": ClusterIdentity(NAME="Praesepe", SIMBAD_NAME="Cl NGC 2632", ID_NAME="Melotte_88", CAT_NAME="NGC_2632", ISO_FILE="praesepe_700myr.dat"),
    "Mel25": ClusterIdentity(NAME="Hyades", SIMBAD_NAME="Cl Melotte 25", ID_NAME="melotte_25", CAT_NAME="Melotte_25", ISO_FILE="hyades_650myr.dat"),
    "Mel111": ClusterIdentity(NAME="ComaBer", SIMBAD_NAME="Cl Melotte 111", ID_NAME="melotte_111", CAT_NAME="Melotte_111", ISO_FILE="mel111_500myr.dat"),
    "M67": ClusterIdentity(NAME="M67", SIMBAD_NAME="Cl NGC 2682", ID_NAME="ngc_2682", CAT_NAME="NGC_2682", ISO_FILE="m67_4000myr.dat"),
    "M13": ClusterIdentity(NAME="M13", SIMBAD_NAME="Cl NGC 6205", ID_NAME="ngc_6205", CAT_NAME="NGC_6205", ISO_FILE="m13_12gyr.dat"),
    "M41": ClusterIdentity(NAME="M41", SIMBAD_NAME="Cl NGC 2287", ID_NAME="ngc_2287", CAT_NAME="NGC_2287", ISO_FILE="m41_240myr.dat"),
}

@dataclass(frozen=True)
class LiteratureSchema:
    """外部异构文献数据表头结构定义模型 (Schema)"""
    id: Optional[str] = None; id_dr2: Optional[str] = None; ra: Optional[str] = None; dec: Optional[str] = None
    pmra: Optional[str] = None; pmdec: Optional[str] = None; plx: Optional[str] = None; plx_err: Optional[str] = None
    mag: Optional[str] = None; color: Optional[str] = None; ruwe: Optional[str] = None; rv: Optional[str] = None
    prob: Optional[str] = None; cluster: Optional[str] = None; main_id: Optional[str] = None; ids: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, default)
        return val if val is not None else default

    def keys(self): return [k for k, v in self.__dict__.items() if v is not None]
    def items(self): return [(k, v) for k, v in self.__dict__.items() if v is not None]


LITERATURE_SCHEMAS: Dict[str, LiteratureSchema] = {
    "gaiadr3": LiteratureSchema(id="source_id", ra="ra", dec="dec", pmra="pmra", pmdec="pmdec", plx="parallax", plx_err="parallax_error", mag="phot_g_mean_mag", color="bp_rp", ruwe="ruwe", rv="radial_velocity"),
    "dr2idx": LiteratureSchema(id="dr3_source_id", id_dr2="dr2_source_id"),
    "ids_simbad": LiteratureSchema(id="gaia_dr3_id", main_id="main_id", ids="ids"),
    "hunt": LiteratureSchema(id="GaiaDR3", prob="Prob", ra="RA_ICRS", dec="DE_ICRS", pmra="pmRA", pmdec="pmDE", plx="Plx", rv="RV", mag="Gmag", color="BP-RP", cluster="Name"),
    "zerj": LiteratureSchema(id="GaiaDR3", ra="RA_ICRS", dec="DE_ICRS", mag="Gmag", cluster="Cluster"),
    "risb": LiteratureSchema(id="GaiaDR3", ra="RA_ICRS", dec="DE_ICRS", pmra="pmRA", pmdec="pmDE", plx="plx", mag="Gmag", color="BP-RP", cluster="Cluster"),
    "cg20": LiteratureSchema(id_dr2="Source", prob="Proba", ra="RA_ICRS", dec="DE_ICRS", pmra="pmRA", pmdec="pmDE", plx="Plx", mag="Gmag", color="BP-RP", cluster="Cluster"),
    "heyl": LiteratureSchema(id="GaiaEDR3", ra="RA_ICRS", dec="DE_ICRS", pmra="pmRA", pmdec="pmDE", plx="Plx", mag="Gmag", color="Bp-Rp"),
}

# 外部物理星等列名称、文件导出的物理模板常量
class TMPL:
    T_RAW = "raw_{idx}"
    V_STD = "std_{idx}"
    V_STX = "stx_{idx}"
    T_ALN = "aln_{idx}"
    T_ALN_EX = "aln_{idx}_{cluster}"
    T_TBL = "tbl_{idx}"
    T_RES_SG = "pgmm_{cluster}_{category}_{mode}_{algo}"
    T_MASTER = "master_{cluster}_{category}_{mode}_{algo}"
    V_RES_SUB = "v_pgmm_{cluster}_{category}_{mode}_{algo}_{tag}"
    V_ALL = "v_wide_{cluster}_{category}_{mode}_{algo}"
    V_DIFF = "v_diff_{cluster}_{category}_{mode}_{algo}_vs_{idx}"
    V_NEW = "v_new_{cluster}_{category}_{mode}_{algo}_vs_{idx}"
    V_MISS = "v_miss_{cluster}_{category}_{mode}_{algo}_vs_{idx}"
    V_ADT = "v_audit_{category}_{cluster}_{mode}_{algo}"
    V_ADT_INPUT = "v_audit_input_{src}"
    V_AUDITED = "{src}_audited"
    V_ADT_HUNT24 = "audit_report_hunt24_by_{src}"
    COL_PROB = "{idx}_prob"
    FILE_FITS = "Pleiades_{category}_vs_{idx}.fits"
    FILE_REPORT = "Validation_Report_{idx}_{date}.csv"
    FILE_LIT_REPORT = "Lit_Audit_{label}_{cluster}_{timestamp}.csv"
    FILE_PLOT = "{cluster}_{category}_{mode}_{prefix}_{timestamp}.png"
    FILE_NEW_CANDIDATES = "new_candidates_vs_{ref}.csv"
    FILE_EXPORT_BASE = "{cluster}_{category}_{mode}_{algo}"
    FILE_SEEDS = "{base}_seeds"
    FILE_SEEDS_CORE = "{base}_seeds_core"
    FILE_SEEDS_NOISE = "{base}_seeds_noise"
    FILE_CROSS_SUMMARY = "{base}_cross_summary"
    FILE_DEEP_AUDIT = "{base}_deep_audit"
    FILE_FINAL_REPORT = "{base}_final_report.txt"
    FILE_MISS_MAG_DIST = "hunt24_missing_mag_dist.png"


# =================================================================
# 5. 算法流水线配置与盲搜空间元数据大盘 (Pipeline & GMM Config)
# =================================================================
# [说明] 核心高维聚类（GMM / DBSCAN / HDBSCAN）特征搜索空间的参数控制。

@dataclass(frozen=True)
class _GmmFeatureSpace:
    """管线内部专用的多维运动特征空间结构体（隐藏内部实现，规范流转契约）"""
    d2: List[str] = field(default_factory=lambda: ["pmra", "pmdec"])
    d3: List[str] = field(default_factory=lambda: ["pmra", "pmdec", "plx"])
    d5: List[str] = field(default_factory=lambda: ["ra", "dec", "pmra", "pmdec", "plx"])
    d6_o: List[str] = field(default_factory=lambda: ["ra", "dec", "pmra", "pmdec", "plx", "rv"])
    d5_h: List[str] = field(default_factory=lambda: ["l", "b", "pm_l_cosb", "pm_b", "plx"])
    d3_v: List[str] = field(default_factory=lambda: ["U", "V", "W"])
    d6_p: List[str] = field(default_factory=lambda: ["X", "Y", "Z", "U", "V", "W"])

    def get(self, key: str, default: Any = None) -> Any:
        norm_key = f"d{key}" if not key.startswith("d") else key
        return getattr(self, norm_key, default)


@dataclass(frozen=True)
class PipelineAlgorithmConfig:
    FULL_NAME: str = "PriorGMM"
    SHORT_NAME: str = "pg"
    feature_map: _GmmFeatureSpace = field(default_factory=_GmmFeatureSpace)
    dim_mode: str = "5d"
    ruwe_limit: float = 1.4
    cluster_algo: str = "dbscan"
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 100
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    hdbscan_cluster_selection_epsilon: float = 0.1
    gmm_covariance_type: str = "full"
    max_iter: int = 20
    tol: float = 1e-5
    use_experimental: bool = True
    enable_subsampling: bool = False
    subsampling_limit: int = 500000

    def __getitem__(self, key: str) -> Any:
        """保持外部算法与评估器组件直接通过字典形式传参时的完美无损解包"""
        if hasattr(self, key):
            val = getattr(self, key)
            if key == "feature_map" and isinstance(val, _GmmFeatureSpace):
                return {
                    "2d": val.d2, "3d": val.d3, "5d": val.d5, "6d_o": val.d6_o,
                    "5d_h": val.d5_h, "3d_v": val.d3_v, "6d_p": val.d6_p
                }
            return val
        raise KeyError(f"Configuration key '{key}' does not exist in GMM pipeline engine.")

    def get(self, key: str, default: Any = None) -> Any:
        try: return self[key]
        except KeyError: return default


GMM_CONFIG = PipelineAlgorithmConfig()


# =================================================================
# 6. 管线统一多态资产注册架构 (Unified Asset Core Models) — [ 内部架构 ]
# =================================================================
# [说明] 此区域为强类型系统核心架构骨架，定义多态资产的行为契约，外部业务开发无需频繁改动。

class AssetType(Enum):
    OBS_FIELD = "obs_field"       # 第一类：目标星团大范围观测大盘
    LIT_CATALOG = "lit_catalog"   # 第二类：来自 VizieR / 文献的历史审计星表
    DYNAMIC_CACHE = "dyn_cache"   # 第三类：动态在线资产缓存


@dataclass
class BaseAssetConfig:
    """所有数据资产的顶级抽象基类"""
    id: str
    asset_type: AssetType
    base_idx: Optional[str] = None
    pre_filters: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None: return val
        # 兼容旧代码对远程库、表 ID 的硬编码字典契约检索
        if key == "remote_catalog_id" and hasattr(self, "remote_catalog_id"):
            return getattr(self, "remote_catalog_id")
        if key == "remote_table_id" and hasattr(self, "remote_table_id"):
            return getattr(self, "remote_table_id")
        raise KeyError(f"'{key}' not found in AssetConfig '{self.id}'")

    def __contains__(self, key: str) -> bool: return hasattr(self, key)

    @property
    def name(self) -> str: return TMPL.T_TBL.format(idx=self.id)
    @property
    def raw_table(self) -> str: return TMPL.T_RAW.format(idx=self.base_idx if self.base_idx else self.id)
    @property
    def std_view(self) -> str: return TMPL.V_STD.format(idx=self.id)
    @property
    def stx_view(self) -> str: return TMPL.V_STX.format(idx=self.id)
    @property
    def aln_view(self) -> str: return TMPL.T_ALN.format(idx=self.id)
    @property
    def sync_mode(self) -> str: return "HYBRID"
    @property
    def col_prob(self) -> Optional[str]:
        return TMPL.COL_PROB.format(idx=self.id) if self.asset_type == AssetType.LIT_CATALOG else None

    @property
    def meta_type(self) -> str:
        """无损向下兼容旧管线对资产类型标签（seeds / field / catalog）的分类判定契约"""
        idx_lower = self.id.lower()
        if "seeds" in idx_lower or self.sync_mode == "VIRTUAL": return "seeds"
        if "field" in idx_lower or self.asset_type == AssetType.OBS_FIELD: return "field"
        return "catalog"


@dataclass
class ObsFieldAssetConfig(BaseAssetConfig):
    """观测数据资产配置模型 (如 Gaia DR3)"""
    asset_type: AssetType = AssetType.OBS_FIELD
    provider: str = "local_file"
    sync_mode: str = "OFFLINE"
    fields_key: str = "gaia_base"

    @property
    def file_pattern(self) -> str:
        cluster_short = self.id.split("_")[0]
        return f"gaia_archive/gaiadr3_{cluster_short}_wide.parquet"

    @property
    def fields(self) -> LiteratureSchema:
        return LITERATURE_SCHEMAS.get(self.fields_key, LiteratureSchema())

    @property
    def actions(self) -> Dict[str, Any]:
        return {"std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through}


@dataclass
class LitCatalogAssetConfig(BaseAssetConfig):
    """VizieR 文献/历史审计目标星表资产配置模型"""
    asset_type: AssetType = AssetType.LIT_CATALOG
    provider: str = "vizier"
    sync_mode: str = "HYBRID"
    file_pattern: str = ""
    remote_catalog_id: Optional[str] = None
    remote_table_id: Optional[str] = None
    fields_key: str = "gaia_base"
    raw_filters: List[str] = field(default_factory=list)
    actions: Dict[str, Any] = field(default_factory=lambda: {
        "std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through
    })
    extra_params: Optional[Dict[str, Any]] = None

    @property
    def fields(self) -> LiteratureSchema:
        return LITERATURE_SCHEMAS.get(self.fields_key, LiteratureSchema())


@dataclass
class DynamicCacheAssetConfig(BaseAssetConfig):
    """网络快照缓存（Cache）资产配置模型"""
    asset_type: AssetType = AssetType.DYNAMIC_CACHE
    index_column: str = "source_id"
    cache_strategy: str = "APPEND_ONLY"
    provider: str = "local_file"
    sync_mode: str = "VIRTUAL"

    @property
    def file_pattern(self) -> str: return f"internal/cache_{self.id}.parquet"
    @property
    def fields(self) -> LiteratureSchema: return LITERATURE_SCHEMAS.get(self.id, LiteratureSchema())
    @property
    def actions(self) -> Dict[str, Any]: return {"std": None, "stx": None, "aln": None}


# =================================================================
# 7. 全局资产标准容器大盘网关 (Global Asset Manifest Gateway)
# =================================================================
# [说明] 彻底合并与精简掉 _StaticManifestHolder 和 ManifestContainer 两个结构体类。
# 统一收拢为标准的 Python 强类型字典，配合 Literal 签名保护 Key 不被意外拼错。

AssetKey = Literal[
    "m45_field", "m44_field", "mel25_field", "mel111_field", "m67_field", "m13_field", "m41_field",
    "m45_seeds_field", "m44_seeds_field", "mel25_seeds_field", "mel111_seeds_field", "m67_seeds_field", "m13_seeds_field", "m41_seeds_field",
    "hunt", "zerj", "risb", "cg20", "heyl", "ids_simbad", "dr2idx"
]

MANIFEST: Dict[AssetKey, BaseAssetConfig] = {
    # --- 基础大范围观测星等资产 (Local Obs Fields) ---
    "m45_field": ObsFieldAssetConfig(id="m45_field"),
    "m44_field": ObsFieldAssetConfig(id="m44_field"),
    "mel25_field": ObsFieldAssetConfig(id="mel25_field"),
    "mel111_field": ObsFieldAssetConfig(id="mel111_field"),
    "m67_field": ObsFieldAssetConfig(id="m67_field"),
    "m13_field": ObsFieldAssetConfig(id="m13_field"),
    "m41_field": ObsFieldAssetConfig(id="m41_field"),

    # --- 动态过滤种子 View/Seeds 资产 ---
    "m45_seeds_field": BaseAssetConfig(id="m45_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m45_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),
    "m44_seeds_field": BaseAssetConfig(id="m44_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m44_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),
    "mel25_seeds_field": BaseAssetConfig(id="mel25_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="mel25_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),
    "mel111_seeds_field": BaseAssetConfig(id="mel111_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="mel111_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),
    "m67_seeds_field": BaseAssetConfig(id="m67_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m67_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),
    "m13_seeds_field": BaseAssetConfig(id="m13_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m13_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),
    "m41_seeds_field": BaseAssetConfig(id="m41_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m41_field", pre_filters=["haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}", "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2", "mag < {SEED_MAX_MAG}"]),

    # --- 文献历史审计资产 ---
    "hunt": LitCatalogAssetConfig(id="hunt", remote_catalog_id="J/A+A/686/A42", fields_key="hunt", file_pattern="vizier/vizier_hunt24_full.parquet", raw_filters=["cluster = '{CAT_NAME}'"]),
    "zerj": LitCatalogAssetConfig(id="zerj", remote_catalog_id="J/A+A/686/A42", fields_key="zerj", file_pattern="vizier/vizier_zerj23.parquet", raw_filters=["cluster = '{CAT_NAME}'"]),
    "risb": LitCatalogAssetConfig(id="risb", remote_catalog_id="J/A+A/694/A258", fields_key="risb", file_pattern="vizier/vizier_risb25.vot", raw_filters=["cluster = '{CAT_NAME}'"]),
    "cg20": LitCatalogAssetConfig(id="cg20", remote_catalog_id="J/A+A/633/A99", fields_key="cg20", file_pattern="vizier/vizier_cg2020_full.parquet", raw_filters=["cluster = '{CAT_NAME}'"], actions={"std": StdActions.std_mapping, "stx": StxActions.bridge_dr2_to_dr3, "aln": AlnActions.pass_through}),
    "heyl": LitCatalogAssetConfig(id="heyl", remote_catalog_id="J/ApJ/926/132", remote_table_id="table4", fields_key="heyl", file_pattern="vizier/vizier_heyl22_*.parquet", actions={"std": StdActions.std_mapping, "stx": StxActions.fill_prob, "aln": AlnActions.pass_through}),

    # --- 内部基础架构高速缓存及交叉比对资产 ---
    "ids_simbad": DynamicCacheAssetConfig(id="ids_simbad", index_column="gaia_dr3_id"),
    "dr2idx": DynamicCacheAssetConfig(id="dr2idx", index_column="dr3_source_id"),
}


# =================================================================
# 8. 外部调用辅助兼容层网关 (Helper Interface)
# =================================================================

def get_assets_for_cluster(cluster_id: str) -> Dict[str, BaseAssetConfig]:
    """无损获取与特定星团匹配的所有强类型配置实例，平滑适配旧代码的成员方法契约"""
    cluster_upper = cluster_id.upper()
    matched_assets_dict = {}
    for k, v in MANIFEST.items():
        if cluster_upper in k.upper() or k in ["hunt", "zerj", "risb", "cg20", "heyl", "ids_simbad", "dr2idx"]:
            matched_assets_dict[k] = v
    return matched_assets_dict

# 将业务过滤接口动态挂载至字典实例上，支持上层管线对旧 ManifestContainer 容器方法的透明调用
MANIFEST.get_assets_for_cluster = get_assets_for_cluster  # type: ignore