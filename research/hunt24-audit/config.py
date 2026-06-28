# config.py
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 内部模块导入
from modules.actions import StdActions, StxActions, AlnActions

# =================================================================
# 1. 系统路径与环境配置 (Paths & Environment)
# =================================================================
BASE_DIR = Path(__file__).resolve().parent.resolve()
LOG_DIR = (BASE_DIR / "logs").resolve()
DATA_DIR = (BASE_DIR / "data").resolve()
ANALYSIS_DIR = (BASE_DIR / "analysis").resolve()
RESULTS_DIR = (ANALYSIS_DIR / "results").resolve()

RAW_DIR = (DATA_DIR / "raw").resolve()
BACKUP_DIR = (DATA_DIR / "backups").resolve()
EXPORT_DIR = (DATA_DIR / "exports").resolve()
INTERNAL_DIR = (DATA_DIR / "internal").resolve()

# 数据源子目录
GAIA_INPUT_DIR = (RAW_DIR / "gaia_archive").resolve()
VIZIER_INPUT_DIR = (RAW_DIR / "vizier").resolve()
SIMBAD_INPUT_DIR = (RAW_DIR / "simbad").resolve()
OAPD_INPUT_DIR = (RAW_DIR / "oapd").resolve()
DOWNLOAD_DIR = RAW_DIR

# Gaia Archive 认证信息
GAIA_USER = os.getenv("GAIA_USER", "jli21")
GAIA_PWD = os.getenv("GAIA_PWD")


# =================================================================
# 2. 科学计算门限与物理常数 (Thresholds & Physics)
# =================================================================
MEMBER_SAMPLE_THRESHOLD = 0.2
GOLDEN_SAMPLE_THRESHOLD = 0.8

AUDIT_PROB_HIGH = 0.7  # 成员身份判定高门限
AUDIT_PROB_LOW = 0.3  # 成员身份判定低门限（背景噪点）
AUDIT_RUWE_LIMIT = 1.4  # Gaia 天体测量质量门限
AUDIT_PLX_RESIDUAL_LIMIT = 1.0  # 视差残差允许度 (mas)
AUDIT_MAG_LIMIT_HUNT24 = 19.0  # Hunt2024 文献深度参考线

# 物理验证权重与细分容忍度
PHYS_VERIFY_WEIGHTS = {
    "pm": 0.4,
    "plx": 0.4,
    "cmd": 0.2,
}  # TODO: 需要调整到与星团相关, 且考虑到不同维度的加权
PHYS_VERIFY_PENALTY_LIMIT = 1.1
PHYS_LIT_PM_LIMIT = 1.5
PHYS_LIT_CMD_LIMIT = 3.0
REDDENING_RATIO_BP_RP = 0.52  # E(BP-RP) / A_G 比例系数 (基于 Gaia DR3 经验红化律)


# =================================================================
# 3. 命名规范、标准列与模板中枢 (Naming & Adapters)
# =================================================================
CATALOG_NAMING_ADAPTER = {
    "zerj": {"M45": "Melotte 22"},
}

STD_COLS = {
    "ID": "id",
    "ID_DR2": "id_dr2",
    "RA": "ra",
    "DEC": "dec",
    "PMRA": "pmra",  # 对应 pmra_cosdec (μ*α), 单位 mas/yr
    "PMDEC": "pmdec",  # 对应 pmdec (μδ), 单位 mas/yr
    "PLX": "plx",
    "MAG": "mag",
    "COLOR": "color",
    "RV": "rv",
    "RUWE": "ruwe",
    "PROB": "prob",
    "GMM_PROB": "gmm_prob",
    "REF_PROB": "r_prob",
    "CLUSTER": "cluster",
}

# Master 表专用标签列名
MASTER_COLS = {
    "SEED_TYPE": "seed_type",  # raw_seed
    "DENSITY_TAG": "density_status",  # core / noise
    "GMM_PROB": "prob",  # 算法计算概率
    "X_MATCH": "x_match_tag",  # Matched / PG_Only / Ref_Only
    "AUDIT": "audit_status",  # Confirmed / Candidate / Contamination
    "AUDIT_NOTE": "audit_note",  # 审计备注（如：视差偏离、暗端漏检等）
}


class TMPL:
    # --- 数据库表/视图名 ---
    T_RAW = "raw_{idx}"  # L1: 原始物理表
    V_STD = "std_{idx}"  # L2: 标准化视图
    V_STX = "stx_{idx}"  # L2+: 标准化视图增强
    T_ALN = "aln_{idx}"  # L3: 物理对齐表
    T_ALN_EX = "aln_{idx}_{cluster}"  # L3+: 物理对齐表增强, tag用于区分不同的对齐版本
    T_TBL = "tbl_{idx}"  # 表格逻辑名称

    # --- 算法结果与分析 ---
    T_RES_SG = "pgmm_{cluster}_{category}_{mode}_{algo}"  # SeedGMM 原始产出
    T_MASTER = "master_{cluster}_{category}_{mode}_{algo}"  # [混合模式] 状态跟踪宽表
    V_RES_SUB = "v_pgmm_{cluster}_{category}_{mode}_{algo}_{tag}"  # 结果子集视图名模板
    V_ALL = "v_wide_{cluster}_{category}_{mode}_{algo}"  # 集成所有参考星表的分析大宽表
    V_DIFF = "v_diff_{cluster}_{category}_{mode}_{algo}_vs_{idx}"  # 分歧源
    V_NEW = "v_new_{cluster}_{category}_{mode}_{algo}_vs_{idx}"  # 全新发现源
    V_MISS = "v_miss_{cluster}_{category}_{mode}_{algo}_vs_{idx}"  # 漏检源
    V_ADT = "v_audit_{category}_{cluster}_{mode}_{algo}"  # 审计专用视图
    V_ADT_INPUT = "v_audit_input_{src}"  # 审计输入增强视图
    V_AUDITED = "{src}_audited"  # 审计完成后的物化表名
    V_ADT_HUNT24 = "audit_report_hunt24_by_{src}"  # 针对 Hunt24 的专项审计结果

    # --- 动态列名 ---
    COL_PROB = "{idx}_prob"

    # --- 导出文件名 ---
    FILE_FITS = "Pleiades_{category}_vs_{idx}.fits"
    FILE_REPORT = "Validation_Report_{idx}_{date}.csv"
    FILE_LIT_REPORT = "Lit_Audit_{label}_{cluster}_{timestamp}.csv"  # 文献审计报告
    FILE_PLOT = "{cluster}_{category}_{mode}_{prefix}_{timestamp}.png"  # 诊断图表文件名
    FILE_NEW_CANDIDATES = "new_candidates_vs_{ref}.csv"  # 新候选者验证清单
    FILE_EXPORT_BASE = "{cluster}_{category}_{mode}_{algo}"  # 导出文件名的基本前缀
    FILE_SEEDS = "{base}_seeds"  # 全量种子星导出文件名模板
    FILE_SEEDS_CORE = "{base}_seeds_core"  # 核心种子星导出文件名模板
    FILE_SEEDS_NOISE = "{base}_seeds_noise"  # 噪声种子星导出文件名模板
    FILE_CROSS_SUMMARY = "{base}_cross_summary"  # 交叉比对汇总文件名
    FILE_DEEP_AUDIT = "{base}_deep_audit"  # 深度审计报告文件名
    FILE_FINAL_REPORT = "{base}_final_report.txt"  # 最终执行摘要文件名
    FILE_MISS_MAG_DIST = "hunt24_missing_mag_dist.png"  # 漏检源星等分布图


# =================================================================
# 4. 【数据资产原子实体】强类型物理常量、星相Schema与资产元数据大盘
# =================================================================
# 4.1 核心字段与种子集 ID
IDX_FIELD_CLUSTER_M45, IDX_FIELD_CLUSTER_M45_SEEDS = "m45_field", "m45_seeds_field"
IDX_FIELD_CLUSTER_M44, IDX_FIELD_CLUSTER_M44_SEEDS = "m44_field", "m44_seeds_field"
IDX_FIELD_CLUSTER_MEL25, IDX_FIELD_CLUSTER_MEL25_SEEDS = (
    "mel25_field",
    "mel25_seeds_field",
)
IDX_FIELD_CLUSTER_MEL111, IDX_FIELD_CLUSTER_MEL111_SEEDS = (
    "mel111_field",
    "mel111_seeds_field",
)
IDX_FIELD_CLUSTER_M67, IDX_FIELD_CLUSTER_M67_SEEDS = "m67_field", "m67_seeds_field"
IDX_FIELD_CLUSTER_M13, IDX_FIELD_CLUSTER_M13_SEEDS = "m13_field", "m13_seeds_field"
IDX_FIELD_CLUSTER_M41, IDX_FIELD_CLUSTER_M41_SEEDS = "m41_field", "m41_seeds_field"

# 4.2 参考文献星表 ID
IDX_CG20 = "cg20"
IDX_HEYL = "heyl"
IDX_ZERJ = "zerj"
IDX_RISB = "risb"
IDX_HUNT = "hunt"

# 4.3 基础设施与算法输出 ID
IDX_DR2IDX = "dr2idx"
IDX_IDS_SIMBAD = "ids_simbad"
IDX_GMM = "pgmm"


# 🌟 4.1 星团等时线与多系统全称命名元数据大盘 (强类型重构)
@dataclass(frozen=True)
class ClusterIdentityMeta:
    """星团多套命名规范系统与等时线实体的强类型元数据契约"""

    NAME: str
    SIMBAD_NAME: str
    ID_NAME: str
    CAT_NAME: str
    ISO_FILE: str

    def get(self, key: str, default: Any = None) -> Any:
        """无损兼容下层可能通过字典格式调用的旧代码"""
        return getattr(self, key, default)


CLUSTER_IDENTITY_ARRAY: Dict[str, ClusterIdentityMeta] = {
    "M45": ClusterIdentityMeta(
        NAME="Pleiades",
        SIMBAD_NAME="Cl Melotte 22",
        ID_NAME="melotte_22",
        CAT_NAME="Melotte_22",
        ISO_FILE="pleiades_126myr.dat",
    ),
    "M44": ClusterIdentityMeta(
        NAME="Praesepe",
        SIMBAD_NAME="Cl NGC 2632",
        ID_NAME="Melotte_88",
        CAT_NAME="NGC_2632",
        ISO_FILE="praesepe_700myr.dat",
    ),
    "Mel25": ClusterIdentityMeta(
        NAME="Hyades",
        SIMBAD_NAME="Cl Melotte 25",
        ID_NAME="melotte_25",
        CAT_NAME="Melotte_25",
        ISO_FILE="hyades_650myr.dat",
    ),
    "Mel111": ClusterIdentityMeta(
        NAME="ComaBer",
        SIMBAD_NAME="Cl Melotte 111",
        ID_NAME="melotte_111",
        CAT_NAME="Melotte_111",
        ISO_FILE="mel111_500myr.dat",
    ),
    "M67": ClusterIdentityMeta(
        NAME="M67",
        SIMBAD_NAME="Cl NGC 2682",
        ID_NAME="ngc_2682",
        CAT_NAME="NGC_2682",
        ISO_FILE="m67_4000myr.dat",
    ),
    "M13": ClusterIdentityMeta(
        NAME="M13",
        SIMBAD_NAME="Cl NGC 6205",
        ID_NAME="ngc_6205",
        CAT_NAME="NGC_6205",
        ISO_FILE="m13_12gyr.dat",
    ),
    "M41": ClusterIdentityMeta(
        NAME="M41",
        SIMBAD_NAME="Cl NGC 2287",
        ID_NAME="ngc_2287",
        CAT_NAME="NGC_2287",
        ISO_FILE="m41_240myr.dat",
    ),
}


# =================================================================
# 6. 数据清单 (Manifest Registry)
# =================================================================
# 6.1 字段映射模板
FIELDS_VIZIER = {
    "id": "Source",
    "ra": "RA_ICRS",
    "dec": "DE_ICRS",
    "pmra": "pmRA",
    "pmdec": "pmDE",
    "plx": "Plx",
    "plx_err": "e_Plx",
    "mag": "Gmag",
    "color": "BP-RP",
    "ruwe": "RUWE",
    "rv": "RV",
}

FIELDS_GAIA_ARCHIVE = {
    "id": "source_id",
    "ra": "ra",
    "dec": "dec",
    "pmra": "pmra",
    "pmdec": "pmdec",
    "plx": "parallax",
    "plx_err": "parallax_error",
    "mag": "phot_g_mean_mag",
    "color": "bp_rp",
    "ruwe": "ruwe",
    "rv": "radial_velocity",
}


# 🌌 4.2 外部文献与基础设施 Schema 物理列名字段映射契约大盘 (强类型重构)
@dataclass(frozen=True)
class LiteratureSchemaMeta:
    """各星表异构数据列名在管线中标准化的物理映射契约"""

    id: Optional[str] = None
    id_dr2: Optional[str] = None
    ra: Optional[str] = None
    dec: Optional[str] = None
    pmra: Optional[str] = None
    pmdec: Optional[str] = None
    plx: Optional[str] = None
    plx_err: Optional[str] = None
    mag: Optional[str] = None
    color: Optional[str] = None
    ruwe: Optional[str] = None
    rv: Optional[str] = None
    prob: Optional[str] = None
    cluster: Optional[str] = None
    main_id: Optional[str] = None
    ids: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """模拟旧 Dict 的 .get() 表现，确保 pd.DataFrame.rename(columns=fields) 完美无感运行"""
        val = getattr(self, key, default)
        return val if val is not None else default

    def keys(self):
        """支持字典解包或循环遍历，过滤掉 None 的键值对"""
        return [k for k, v in self.__dict__.items() if v is not None]

    def items(self):
        """支持遍历映射对行为"""
        return [(k, v) for k, v in self.__dict__.items() if v is not None]


LITERATURE_FIELDS_ARRAY: Dict[str, LiteratureSchemaMeta] = {
    "gaia_base": LiteratureSchemaMeta(
        id="source_id",
        ra="ra",
        dec="dec",
        pmra="pmra",
        pmdec="pmdec",
        plx="parallax",
        plx_err="parallax_error",
        mag="phot_g_mean_mag",
        color="bp_rp",
        ruwe="ruwe",
        rv="radial_velocity",
    ),
    "dr2idx": LiteratureSchemaMeta(id="dr3_source_id", id_dr2="dr2_source_id"),
    "ids_simbad": LiteratureSchemaMeta(id="gaia_dr3_id", main_id="main_id", ids="ids"),
    "hunt": LiteratureSchemaMeta(
        id="GaiaDR3",
        prob="Prob",
        ra="RA_ICRS",
        dec="DE_ICRS",
        pmra="pmRA",
        pmdec="pmDE",
        plx="Plx",
        rv="RV",
        mag="Gmag",
        color="BP-RP",
        cluster="Name",
    ),
    "zerj": LiteratureSchemaMeta(
        id="GaiaDR3", ra="RA_ICRS", dec="DE_ICRS", mag="Gmag", cluster="Cluster"
    ),
    "risb": LiteratureSchemaMeta(
        id="GaiaDR3",
        ra="RA_ICRS",
        dec="DE_ICRS",
        pmra="pmRA",
        pmdec="pmDE",
        plx="plx",
        mag="Gmag",
        color="BP-RP",
        cluster="Cluster",
    ),
    "cg20": LiteratureSchemaMeta(
        id_dr2="Source",
        prob="Proba",
        ra="RA_ICRS",
        dec="DE_ICRS",
        pmra="pmRA",
        pmdec="pmDE",
        plx="Plx",
        mag="Gmag",
        color="BP-RP",
        cluster="Cluster",
    ),
    "heyl": LiteratureSchemaMeta(
        id="GaiaEDR3",
        ra="RA_ICRS",
        dec="DE_ICRS",
        pmra="pmRA",
        pmdec="pmDE",
        plx="Plx",
        mag="Gmag",
        color="Bp-Rp",
    ),
}


# 🚀 4.3 星表基础设施与文献数据资产总管盘 (Dataclass 强类型版)
@dataclass(frozen=True)
class AssetMeta:
    """外部星表与管线数据资产的强类型结构契约定义"""

    file_pattern: str
    provider: Literal["local_file", "gaia", "vizier"] = "local_file"
    sync_mode: Literal["HYBRID", "FORCE_REMOTE", "OFFLINE", "VIRTUAL"] = "HYBRID"
    storage_path: Optional[str] = "snapshots"
    is_static: bool = True
    use_ex_aln: bool = True
    remote_cat: Optional[str] = None
    remote_tab: Optional[str] = "members"
    fields_key: str = "gaia_base"
    raw_filters: List[str] = field(default_factory=list)
    actions: Dict[str, Any] = field(
        default_factory=lambda: {
            "std": StdActions.std_mapping,
            "stx": StxActions.pass_through,
            "aln": AlnActions.pass_through,
        }
    )
    extra_params: Optional[Dict[str, Any]] = None

    def get(self, key: str, default: Any = None) -> Any:
        """保持对原有旧代码 Dict 字典消费行为的无损向下兼容接口"""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        if key == "extra_params" and self.extra_params is None:
            return {}
        return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


LITERATURE_METADATA_ARRAY: Dict[str, AssetMeta] = {
    "field": AssetMeta(
        file_pattern=f"{GAIA_INPUT_DIR.name}/gaiadr3_{{lower_key}}_wide.parquet",
        provider="local_file",
        is_static=False,
        use_ex_aln=False,
    ),
    "seeds": AssetMeta(
        file_pattern="",
        provider="internal_view",
        sync_mode="VIRTUAL",
        is_static=False,
        use_ex_aln=False,
        raw_filters=[
            "haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}*5/{SEED_RADIUS}",
            "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}*2",
            "mag < {SEED_MAX_MAG}",
            # "ruwe < {SEED_MAX_RUWE}",
        ],
    ),
    "dr2idx": AssetMeta(
        file_pattern=f"{GAIA_INPUT_DIR.name}/gaia_gaiadr3_dr2_neighbourhood_Melotte_22_r*",
        is_static=False,
        use_ex_aln=False,
        remote_cat="gaiadr3",
        remote_tab="dr2_neighbourhood",
        fields_key="dr2idx",
    ),
    "ids_simbad": AssetMeta(
        file_pattern=f"{SIMBAD_INPUT_DIR.name}/local_ids_simbad.parquet",
        is_static=False,
        use_ex_aln=False,
        fields_key="ids_simbad",
        actions={"std": None, "stx": None, "aln": None},
        extra_params={"id_col": "id", "prefix": "Gaia DR3 ", "optional": True},
    ),
    "hunt": AssetMeta(
        file_pattern=f"{VIZIER_INPUT_DIR.name}/vizier_hunt24_full.parquet",
        remote_cat="J/A+A/686/A42",
        fields_key="hunt",
        raw_filters=["cluster = '{CAT_NAME}'"],
    ),
    "zerj": AssetMeta(
        file_pattern=f"{VIZIER_INPUT_DIR.name}/vizier_zerj23.parquet",
        remote_cat="J/A+A/686/A42",
        fields_key="zerj",
        raw_filters=["cluster = '{CAT_NAME}'"],
    ),
    "risb": AssetMeta(
        file_pattern=f"{VIZIER_INPUT_DIR.name}/vizier_risb25.vot",
        remote_cat="J/A+A/694/A258",
        fields_key="risb",
        raw_filters=["cluster = '{CAT_NAME}'"],
    ),
    "cg20": AssetMeta(
        file_pattern=f"{VIZIER_INPUT_DIR.name}/vizier_cg2020_full.parquet",
        remote_cat="J/A+A/633/A99",
        fields_key="cg20",
        raw_filters=["cluster = '{CAT_NAME}'"],
        actions={
            "std": StdActions.std_mapping,
            "stx": StxActions.bridge_dr2_to_dr3,
            "aln": AlnActions.pass_through,
        },
    ),
    "heyl": AssetMeta(
        file_pattern=f"{VIZIER_INPUT_DIR.name}/vizier_heyl22_*.parquet",
        remote_cat="J/ApJ/926/132",
        remote_tab="table4",
        fields_key="heyl",
        actions={
            "std": StdActions.std_mapping,
            "stx": StxActions.fill_prob,
            "aln": AlnActions.pass_through,
        },
    ),
}


# =================================================================
# 5. 强类型多态资产注册配置架构 (Unified Catalog & Cache Registry Architecture)
# =================================================================

class AssetType(Enum):
    OBS_FIELD = "obs_field"       # 第一类：目标星团大范围观测大盘 (如 Gaia DR3)
    LIT_CATALOG = "lit_catalog"   # 第二类：来自 VizieR / 文献的历史审计星表
    DYNAMIC_CACHE = "dyn_cache"   # 第三类：动态在线资产缓存 (SIMBAD/VizieR/Archive)


@dataclass
class BaseAssetConfig:
    """所有数据资产的顶级抽象基类"""
    id: str
    asset_type: AssetType
    base_idx: Optional[str] = None
    pre_filters: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """无损兼容旧代码的 dict-style .get() 访问"""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    def __getitem__(self, key: str) -> Any:
        """无损兼容旧代码的 bracket 访问（如 task_cfg['remote_catalog_id']）"""
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        # 针对旧代码期望从特定子类属性取值的行为进行路由垫片防御
        if key == "remote_catalog_id" and hasattr(self, "remote_cat"):
            return getattr(self, "remote_cat")
        if key == "remote_table_id" and hasattr(self, "remote_tab"):
            return getattr(self, "remote_tab")
        raise KeyError(f"'{key}' not found in AssetConfig '{self.id}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    @property
    def name(self) -> str:
        return TMPL.T_TBL.format(idx=self.id)

    @property
    def raw_table(self) -> str:
        return TMPL.T_RAW.format(idx=self.base_idx if self.base_idx else self.id)

    @property
    def std_view(self) -> str:
        return TMPL.V_STD.format(idx=self.id)

    @property
    def col_prob(self) -> Optional[str]:
        # 只有特定的外部文献对比表才会在对齐层暴露出特有的概率列
        return TMPL.COL_PROB.format(idx=self.id) if self.asset_type == AssetType.LIT_CATALOG else None


# 🚀 5.1 第一类：全大盘观测数据资产配置模型
@dataclass
class ObsFieldAssetConfig(BaseAssetConfig):
    """专门处理类似 Gaia DR3 百万级非结构化大天区观测数据的配置实体"""
    asset_type: AssetType = AssetType.OBS_FIELD
    provider: str = "local_file"
    sync_mode: str = "OFFLINE"
    fields_key: str = "gaia_base"

    @property
    def file_pattern(self) -> str:
        cluster_short = self.id.split("_")[0]
        return f"gaia_archive/gaiadr3_{cluster_short}_wide.parquet"

    @property
    def fields(self) -> LiteratureSchemaMeta:
        return LITERATURE_FIELDS_ARRAY.get(self.fields_key, LiteratureSchemaMeta())

    @property
    def actions(self) -> Dict[str, Any]:
        return {
            "std": StdActions.std_mapping,
            "stx": StxActions.pass_through,
            "aln": AlnActions.pass_through,
        }


# 🚀 5.2 第二类：VizieR 文献/历史审计目标星表资产配置模型
@dataclass
class LitCatalogAssetConfig(BaseAssetConfig):
    """专门处理来自 VizieR 显式下载、用于 Stage 5 事后交叉审计的历史星表配置实体"""
    asset_type: AssetType = AssetType.LIT_CATALOG
    meta_type: str = "hunt"  # 对应原 LITERATURE_METADATA_ARRAY 中的 key
    provider: str = "vizier"
    sync_mode: str = "HYBRID"

    @property
    def meta(self) -> AssetMeta:
        return LITERATURE_METADATA_ARRAY.get(self.meta_type, AssetMeta(file_pattern=""))

    @property
    def remote_catalog_id(self) -> Optional[str]:
        return self.meta.remote_cat

    @property
    def remote_table_id(self) -> Optional[str]:
        return self.meta.remote_tab

    @property
    def file_pattern(self) -> str:
        return self.meta.file_pattern

    @property
    def fields(self) -> LiteratureSchemaMeta:
        return LITERATURE_FIELDS_ARRAY.get(self.meta.fields_key, LiteratureSchemaMeta())

    @property
    def actions(self) -> Dict[str, Any]:
        return self.meta.actions


# 🚀 5.3 第三类：动态在线网络快照缓存（Cache）资产配置模型
@dataclass
class DynamicCacheAssetConfig(BaseAssetConfig):
    """
    【本版核心重构】专门处理在管线运行中，从 SIMBAD / VizieR / Gaia Archive 动态获取的持久化快照缓存。
    特点：总体数据量巨大、管线单次请求量极小、本征不经常变化。
    """
    asset_type: AssetType = AssetType.DYNAMIC_CACHE
    index_column: str = "source_id"               # 核心索引主键 (如：gaia_dr3_source_id)
    cache_strategy: str = "APPEND_ONLY"          # 缓存写入策略 (只允许增量追加，不许篡改)
    provider: str = "local_file"
    sync_mode: str = "VIRTUAL"                   # 在未发生 Cache Miss 前对算法域不可见

    @property
    def file_pattern(self) -> str:
        # 映射到内部专用持久化拓扑路径下
        return f"internal/cache_{self.id}.parquet"

    @property
    def fields(self) -> LiteratureSchemaMeta:
        # 检索专门为缓存实体注册的 Schema（如原代码中的 ids_simbad、dr2idx 等）
        return LITERATURE_FIELDS_ARRAY.get(self.id, LiteratureSchemaMeta())

    @property
    def actions(self) -> Dict[str, Any]:
        # 缓存数据的核心物理语义是快照恢复，Stage 1.5 阶段执行纯粹的无状态透传或原子合并
        return {
            "std": None,
            "stx": None,
            "aln": None
        }


# =================================================================
# 5.4 全局注册清单资产中枢实体类 (提供 100% 静态防御与双向兼容)
# =================================================================
@dataclass(frozen=True)
class ManifestContainer:
    """全局无损数据大盘，清晰划分三类资产语义空间"""

    # --- 1. 第一类：各天区全大盘观测观测数据资产 (Obs Field Layer) ---
    m45_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m45_field"))
    m44_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m44_field"))
    mel25_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="mel25_field"))
    mel111_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="mel111_field"))
    m67_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m67_field"))
    m13_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m13_field"))
    m41_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m41_field"))

    # --- 2. 虚拟衍生的粗筛种子层 (Seeds Extractor - 依赖观测大盘) ---
    m45_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m45_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m45_field"))
    m44_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m44_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m44_field"))
    mel25_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="mel25_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="mel25_field"))
    mel111_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="mel111_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="mel111_field"))
    m67_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m67_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m67_field"))
    m13_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m13_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m13_field"))
    m41_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m41_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m41_field"))

    # --- 3. 第二类：来自 Vizier 的显式历史文献审计星表 (Literature Catalog Layer) ---
    hunt: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="hunt", meta_type="hunt"))
    zerj: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="zerj", meta_type="zerj"))
    risb: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="risb", meta_type="risb"))
    cg20: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="cg20", meta_type="cg20"))
    heyl: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="heyl", meta_type="heyl"))

    # --- 4. 第三类：高频动态在线资产缓存库 (Dynamic System Cache Layer) ---
    # 以 gaia dr3 source_id 为主键，缓存 SIMBAD 天体别名、多路标识符交叉表 (parents, ids 等)
    ids_simbad: DynamicCacheAssetConfig = field(default_factory=lambda: DynamicCacheAssetConfig(id="ids_simbad", index_column="gaia_dr3_id"))
    # 跨版本交叉索引缓存 (DR2 到 DR3 邻域对齐映射链)
    dr2idx: DynamicCacheAssetConfig = field(default_factory=lambda: DynamicCacheAssetConfig(id="dr2idx", index_column="dr3_source_id"))

    def __getitem__(self, key: str) -> BaseAssetConfig:
        """核心兼容垫片：保证旧代码的 MANIFEST[idx] 语法在多态体系下完美运行"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Asset '{key}' not registered in pipeline manifest container.")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def items(self):
        return [(k, v) for k, v in vars(self).items() if isinstance(v, BaseAssetConfig)]

    def keys(self):
        return [k for k, _ in self.items()]

    def values(self):
        return [v for _, v in self.items()]


# 实例化全局单例强类型清单对象
MANIFEST = ManifestContainer()


# =================================================================
# 6. 星团物理先验配置数据库 (Pure Cluster Physical Database)
# =================================================================
@dataclass
class ClusterConfig:
    KEY_ID: str = ""
    CENTER_RA: float = 0.0
    CENTER_DEC: float = 0.0
    RADIUS: float = 0.0
    RA_MIN: float = 0.0
    RA_MAX: float = 0.0
    DEC_MIN: float = 0.0
    DEC_MAX: float = 0.0
    MAX_MAG: float = 21.0
    CORE_RADIUS: float = 0.0
    HALF_MASS_RADIUS: float = 0.0
    R_HALF_LIGHT: float = 0.0
    TIDAL_RADIUS: float = 0.0
    DISTANCE_PC: float = 0.0
    DISTANCE_MODULUS: float = 0.0
    AV: float = 0.0
    EXT_AG: float = 0.0
    E_BP_RP: float = 0.0
    PLX_REF: float = 0.0
    PLX_ERROR: float = 0.0
    PMRA_REF: float = 0.0
    PMDEC_REF: float = 0.0
    PMRA_DISPERSION: float = 0.0
    PMDEC_DISPERSION: float = 0.0
    PM_CORR: float = 0.0
    PM_RADIUS: float = 0.0
    RV_REF: float = 0.0
    RV_ERROR: float = 0.0
    UVW_REF: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    V_ERROR: float = 2.0
    UVW_ERROR: float = 2.0
    U_ERROR: float = 2.0
    W_ERROR: float = 1.0
    CMD_REF: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    CMD_DEV: float = 0.8
    KINE_SCORE_LIMIT: float = 2.0
    SEED_RADIUS: float = 0.0
    SEED_PLX_LIM: float = 0.0
    SEED_MAX_MAG: float = 0.0
    SEED_MAX_RUWE: float = 1.4
    SEED_PM_LIM: float = 0.0

    def get(self, key: str, default: Any = None) -> Any:
        """兼容旧代码 dict-style .get() 访问（如 analysis.py 的 ctx.get('PLX_REF')）"""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    @property
    def id(self) -> str:
        """向上/下层历史遗留代码提供无感兼容，访问 .id 实际返回小写的 KEY_ID"""
        return self.KEY_ID.lower()

    def keys(self):
        # 暴露所有物理字段名以及动态的 id 属性
        return list(self.__dict__.keys()) + ["id", "CAT_NAME"]

    def __getitem__(self, key):
        if key == "id":
            return self.id
        if key == "CAT_NAME":
            # 默认兜底，由于解包时无法直接传外部 idx_data 参数，这里直接返回基础大写名称
            # 真正的精细别名渲染会在 _apply_pre_filter 内部实时修正
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
        return (
            CLUSTER_IDENTITY_ARRAY.get(self.KEY_ID).ISO_FILE
            if CLUSTER_IDENTITY_ARRAY.get(self.KEY_ID)
            else "default.dat"
        )

    @property
    def ID_NAME(self) -> str:
        meta = CLUSTER_IDENTITY_ARRAY.get(self.KEY_ID)
        return meta.ID_NAME if meta else self.KEY_ID.lower()

    @property
    def CAT_NAME(self) -> str:
        meta = CLUSTER_IDENTITY_ARRAY.get(self.KEY_ID)
        return meta.CAT_NAME if meta else self.KEY_ID

    @property
    def SIMBAD_NAME(self) -> str:
        meta = CLUSTER_IDENTITY_ARRAY.get(self.KEY_ID)
        return meta.SIMBAD_NAME if meta else self.KEY_ID

    @property
    def NAME(self) -> str:
        meta = CLUSTER_IDENTITY_ARRAY.get(self.KEY_ID)
        return meta.NAME if meta else self.KEY_ID

    # 🌟 彻底类型化：将命名自适应策略内聚到对象内部
    def get_cat_name(self, idx_data: str, manifest=None) -> str:
        """
        根据当前驱动的外部星表资产源 (idx_data)，动态解析其在这个星表语境下的自适应表名。
        """
        # 1. 尝试从全局配置或传入的 manifest 中读取命名适配器映射表
        # 假设旧的命名适配器存放在全局或 manifest 对象的 CATALOG_NAMING_ADAPTER 属性中
        adapter = getattr(manifest, "CATALOG_NAMING_ADAPTER", {}) if manifest else {}

        # 2. 路由检索：看当前资产源下，针对当前星团是否注册了别名覆写
        # 比如：adapter.get("m45_seeds_field", {}).get("m45") -> "Melotte_22"
        override_name = adapter.get(idx_data, {}).get(self.id)
        if override_name:
            return override_name

        # 3. 兜底策略：若无专属别名，则使用规范化大写的 KEY_ID (或小写的 id，根据您数仓建表的规范统一)
        return self.KEY_ID


_RAW_TARGETS = {
    "M45": ClusterConfig(
        CENTER_RA=56.61398997432307,
        CENTER_DEC=24.09029596042996,
        RADIUS=17.78,
        RA_MIN=44.0,
        RA_MAX=66.0,
        DEC_MIN=16.0,
        DEC_MAX=36.0,
        CORE_RADIUS=1.3,
        HALF_MASS_RADIUS=3.5,
        R_HALF_LIGHT=3.0,
        TIDAL_RADIUS=10.0,
        DISTANCE_PC=136.2,
        DISTANCE_MODULUS=5.66,
        EXT_AG=0.12,
        E_BP_RP=0.06,
        PLX_REF=7.329881851789656,
        PLX_ERROR=0.5,
        PMRA_REF=19.816076644561296,
        PMDEC_REF=-45.02481613280063,
        PMRA_DISPERSION=1.5,
        PMDEC_DISPERSION=1.2,
        PM_CORR=-0.03740541790723895,
        PM_RADIUS=3.0,
        RV_REF=5.63,
        RV_ERROR=5.0,
        UVW_REF=np.array([-6.05, -28.02, -14.34]),
        UVW_ERROR=2.0,
        U_ERROR=2.2,
        V_ERROR=1.6,
        W_ERROR=1.0,
        SEED_RADIUS=2.0,
        SEED_PLX_LIM=1.5,
        SEED_MAX_MAG=18.0,
        SEED_MAX_RUWE=1.2,
    ),
    "M44": ClusterConfig(
        CENTER_RA=130.1,
        CENTER_DEC=19.7,
        RADIUS=11.90,
        RA_MIN=120.0,
        RA_MAX=140.0,
        DEC_MIN=10.0,
        DEC_MAX=30.0,
        CORE_RADIUS=0.8,
        HALF_MASS_RADIUS=3.9,
        R_HALF_LIGHT=3.5,
        TIDAL_RADIUS=12.0,
        DISTANCE_PC=187.0,
        DISTANCE_MODULUS=6.36,
        EXT_AG=0.05,
        E_BP_RP=0.03,
        PLX_REF=5.35,
        PLX_ERROR=0.4,
        PMRA_REF=-36.0,
        PMDEC_REF=-12.9,
        RV_REF=35.0,
        RV_ERROR=5.0,
        UVW_REF=np.array([-34.5, -21.2, -6.8]),
        V_ERROR=2.0,
        PM_RADIUS=4.0,
        CMD_DEV=0.6,
        SEED_RADIUS=2.0,
        SEED_PLX_LIM=1.2,
        SEED_MAX_MAG=18.0,
        SEED_MAX_RUWE=1.4,
    ),
    "Mel25": ClusterConfig(
        CENTER_RA=66.75,
        CENTER_DEC=15.87,
        RADIUS=59.31,
        RA_MIN=50.0,
        RA_MAX=85.0,
        DEC_MIN=0.0,
        DEC_MAX=32.0,
        CORE_RADIUS=2.7,
        HALF_MASS_RADIUS=4.1,
        R_HALF_LIGHT=3.1,
        TIDAL_RADIUS=10.0,
        DISTANCE_PC=46.7,
        DISTANCE_MODULUS=3.35,
        AV=0.02,
        EXT_AG=0.01,
        E_BP_RP=0.01,
        PLX_REF=21.41,
        PLX_ERROR=2.5,
        PMRA_REF=101.10,
        PMDEC_REF=-28.50,
        RV_REF=39.10,
        UVW_REF=np.array([-42.24, -19.11, -1.45]),
        V_ERROR=3.0,
        RV_ERROR=5.0,
        PM_RADIUS=12.0,
        CMD_DEV=0.6,
        SEED_RADIUS=8.0,
        SEED_PLX_LIM=2.0,
        SEED_MAX_MAG=16.0,
        SEED_MAX_RUWE=1.2,
    ),
    "Mel111": ClusterConfig(
        CENTER_RA=186.6,
        CENTER_DEC=26.1,
        RADIUS=42.61,
        RA_MIN=175.0,
        RA_MAX=198.0,
        DEC_MIN=15.0,
        DEC_MAX=37.0,
        CORE_RADIUS=1.5,
        HALF_MASS_RADIUS=4.5,
        R_HALF_LIGHT=3.8,
        TIDAL_RADIUS=15.0,
        DISTANCE_PC=86.0,
        DISTANCE_MODULUS=4.67,
        EXT_AG=0.02,
        E_BP_RP=0.01,
        PLX_REF=11.60,
        PLX_ERROR=1.5,
        PMRA_REF=-12.11,
        PMDEC_REF=-9.01,
        RV_REF=-1.0,
        RV_ERROR=5.0,
        UVW_REF=np.array([-1.7, -6.1, -1.3]),
        V_ERROR=2.5,
        PM_RADIUS=5.0,
        CMD_DEV=0.6,
        SEED_RADIUS=5.0,
        SEED_PLX_LIM=1.5,
        SEED_MAX_MAG=16.0,
        SEED_MAX_RUWE=1.2,
    ),
    "M67": ClusterConfig(
        CENTER_RA=132.83,
        CENTER_DEC=11.82,
        RADIUS=2.5,
        RA_MIN=128.0,
        RA_MAX=138.0,
        DEC_MIN=7.0,
        DEC_MAX=17.0,
        CORE_RADIUS=1.2,
        HALF_MASS_RADIUS=4.5,
        R_HALF_LIGHT=3.8,
        TIDAL_RADIUS=16.0,
        DISTANCE_PC=850.0,
        DISTANCE_MODULUS=9.65,
        EXT_AG=0.10,
        E_BP_RP=0.05,
        PLX_REF=1.17,
        PLX_ERROR=0.2,
        PMRA_REF=-10.96,
        PMDEC_REF=-2.94,
        RV_REF=33.7,
        RV_ERROR=3.0,
        UVW_REF=np.array([-21.4, -25.2, -15.1]),
        V_ERROR=1.5,
        PM_RADIUS=1.5,
        CMD_DEV=0.5,
        SEED_RADIUS=1.5,
        SEED_PLX_LIM=0.4,
        SEED_MAX_MAG=20.0,
        SEED_MAX_RUWE=1.4,
        SEED_PM_LIM=2.5,
    ),
    "M13": ClusterConfig(
        CENTER_RA=250.42,
        CENTER_DEC=36.46,
        RADIUS=3.28,
        RA_MIN=248.0,
        RA_MAX=253.0,
        DEC_MIN=34.5,
        DEC_MAX=38.5,
        CORE_RADIUS=1.3,
        HALF_MASS_RADIUS=3.5,
        R_HALF_LIGHT=3.2,
        TIDAL_RADIUS=43.0,
        DISTANCE_PC=7100.0,
        DISTANCE_MODULUS=14.25,
        EXT_AG=0.04,
        E_BP_RP=0.02,
        PLX_REF=0.14,
        PLX_ERROR=0.1,
        PMRA_REF=-3.18,
        PMDEC_REF=-2.57,
        RV_REF=-244.2,
        RV_ERROR=10.0,
        UVW_REF=np.array([58.0, -241.0, 10.0]),
        V_ERROR=10.0,
        PM_RADIUS=1.0,
        CMD_DEV=0.4,
        SEED_RADIUS=0.8,
        SEED_PLX_LIM=0.5,
        SEED_MAX_MAG=20.5,
        SEED_MAX_RUWE=1.4,
        SEED_PM_LIM=2.0,
    ),
    "M41": ClusterConfig(
        CENTER_RA=101.50,
        CENTER_DEC=-20.75,
        RADIUS=2.53,
        RA_MIN=96.0,
        RA_MAX=107.0,
        DEC_MIN=-25.0,
        DEC_MAX=-15.0,
        CORE_RADIUS=1.5,
        HALF_MASS_RADIUS=4.0,
        R_HALF_LIGHT=3.6,
        TIDAL_RADIUS=12.0,
        DISTANCE_PC=710.0,
        DISTANCE_MODULUS=9.25,
        EXT_AG=0.05,
        E_BP_RP=0.03,
        PLX_REF=1.41,
        PLX_ERROR=0.3,
        PMRA_REF=-1.55,
        PMDEC_REF=-1.05,
        RV_REF=34.0,
        RV_ERROR=5.0,
        UVW_REF=np.array([-10.5, -20.2, -5.1]),
        V_ERROR=2.0,
        CMD_DEV=0.6,
        SEED_RADIUS=2.0,
        SEED_PLX_LIM=0.9,
        SEED_MAX_MAG=18.0,
        SEED_MAX_RUWE=1.4,
    ),
}

CLUSTERS: Dict[str, ClusterConfig] = {}
for k, obj in _RAW_TARGETS.items():
    object.__setattr__(obj, "KEY_ID", k)
    CLUSTERS[k] = obj


# =================================================================
# 7. 算法流水线配置与盲搜空间元数据大盘 (Pipeline & GMM Config去类型化)
# =================================================================
@dataclass(frozen=True)
class GmmFeatureSpace:
    """聚类与盲搜空间的物理维度强类型字段集定义"""

    d2: List[str] = field(default_factory=lambda: ["pmra", "pmdec"])
    d3: List[str] = field(default_factory=lambda: ["pmra", "pmdec", "plx"])
    d5: List[str] = field(default_factory=lambda: ["ra", "dec", "pmra", "pmdec", "plx"])
    d6_o: List[str] = field(
        default_factory=lambda: ["ra", "dec", "pmra", "pmdec", "plx", "rv"]
    )
    d5_h: List[str] = field(
        default_factory=lambda: ["l", "b", "pm_l_cosb", "pm_b", "plx"]
    )
    d3_v: List[str] = field(default_factory=lambda: ["U", "V", "W"])
    d6_p: List[str] = field(default_factory=lambda: ["X", "Y", "Z", "U", "V", "W"])

    def get(self, key: str, default: Any = None) -> Any:
        """支持旧代码用 2d/3d 字符串格式获取"""
        norm_key = f"d{key}" if not key.startswith("d") else key
        return getattr(self, norm_key, default)


@dataclass(frozen=True)
class PipelineAlgorithmConfig:
    """统计混合模型与机器学习管线的全局强类型只读配置实体"""

    FULL_NAME: str = "SeedGMM"
    SHORT_NAME: str = "SG"
    feature_map: GmmFeatureSpace = field(default_factory=GmmFeatureSpace)
    dim_mode: str = "5d"
    ruwe_limit: float = 1.4
    cluster_algo: str = "dbscan"
    dbscan_eps: float = 0.3 # Castro-Ginard 的算法结果: 0.2231
    dbscan_min_samples: int = 100 # Castro-Ginard 的算法输入: 18
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
        """🌟 核心兼容垫片：保证外部机器学习算法组件通过字典形式传参时完全兼容"""
        if hasattr(self, key):
            val = getattr(self, key)
            if key == "feature_map" and isinstance(val, GmmFeatureSpace):
                # 兼容旧代码直接在 feature_map 字典上二级取值的行为
                return {
                    "2d": val.d2,
                    "3d": val.d3,
                    "5d": val.d5,
                    "6d_o": val.d6_o,
                    "5d_h": val.d5_h,
                    "3d_v": val.d3_v,
                    "6d_p": val.d6_p,
                }
            return val
        raise KeyError(
            f"Configuration key '{key}' does not exist in GMM pipeline engine."
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# 全局单例算法强类型配置中枢
GMM_CONFIG = PipelineAlgorithmConfig()
