# config.py
"""
hunt24-audit 科学管线全局配置蓝图中枢 v1.6

[文件结构说明]
1. 系统路径与环境配置 (Paths & Environment) - 定义统一的物理磁盘拓扑
2. 数据库内部命名空间模板契约 (Database View & Table Templates) - 死锁各阶段命名空间
3. 科学数据模型与洗练元数据定义 (Schema Meta Classes) - 天体测量学物理字段映射声明
4. 静态算法常数与超参白皮书 (Algorithm Constants & Hyperparameters) - 无监督聚类与 GMM 物理常数
5. 强类型多态资产注册配置架构 (Unified Multi-type Asset Architecture) - 三类资产模型的面向对象设计
6. 全局单例多态清单容器 (Manifest Container & Instances) - 向下完全兼容旧代码的资产总台

[本次修改说明]
- 废除原平铺、混淆的 CatalogConfig 散装设计，升级为基于面向对象的多态资产配置。
- 引入 AssetType 枚举，严格划分：第一类(大盘观测)、第二类(VizieR 文献星表)、第三类(SIMBAD/Archive网络缓存)。
- 正式确立 Stage 1.2 ODS 层物理落表命名契约，通过属性自动导出统一前缀表名 (raw_obs_*, raw_lit_*, raw_csh_*)。
- 实现 __getitem__、get 和 __contains__ 垫片逻辑，100% 完美向下兼容上游组件的字典及括号访问语法。
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 内部模块导入 (保持原始依赖链稳定)
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
GAIA_USER = os.getenv("GAIA_USER", "your_username")
GAIA_PASSWORD = os.getenv("GAIA_PASSWORD", "your_password")


# =================================================================
# 2. 数据库内部命名空间模板契约 (Database View & Table Templates)
# =================================================================
class TMPL:
    # Stage 1.2 ODS 原始物理落表命名前缀契约
    T_RAW_OBS = "raw_obs_{idx}"  # 第一类：观测大盘物理表
    T_RAW_LIT = "raw_lit_{idx}"  # 第二类：文献审计物理表
    T_RAW_CSH = "raw_csh_{idx}"  # 第三类：系统快照缓存物理表

    # Stage 1.5 洗练层标准视图/表命名空间
    T_TBL = "std_{idx}"
    T_RAW = "raw_{idx}"  # 兼容陈旧引用的基本模板
    V_STD = "stx_view_{idx}"
    V_CAT = "cat_view_{idx}"
    V_CSH = "csh_view_{idx}"

    # 外部文献表在对齐层暴露的特有成员概率列名格式
    COL_PROB = "prob_{idx}"


# =================================================================
# 3. 科学数据模型与洗练元数据定义 (Schema Meta Classes)
# =================================================================
@dataclass
class LiteratureSchemaMeta:
    """定义文献星表与标准表的列名双向映射契约"""
    col_id: str = "source_id"
    col_ra: str = "ra"
    col_dec: str = "dec"
    col_plx: Optional[str] = None
    col_pmra: Optional[str] = None
    col_pmdec: Optional[str] = None
    col_prob: Optional[str] = None


@dataclass
class AssetMeta:
    """无缝兼容陈旧文献元数据配置的实体垫片"""
    remote_cat: str
    remote_tab: str
    file_pattern: str
    fields_key: str
    actions: Dict[str, Any]


# 原始历史洗练字段规则字典
LITERATURE_FIELDS_ARRAY: Dict[str, LiteratureSchemaMeta] = {
    "gaia_base": LiteratureSchemaMeta(
        col_id="source_id", col_ra="ra", col_dec="dec", col_plx="parallax", col_pmra="pmra", col_pmdec="pmdec"
    ),
    "hunt": LiteratureSchemaMeta(col_id="Source", col_ra="RA_deg", col_dec="DE_deg", col_prob="P"),
    "zerj": LiteratureSchemaMeta(col_id="Source", col_ra="RA_deg", col_dec="DE_deg", col_prob="P"),
    "risb": LiteratureSchemaMeta(col_id="GaiaDR3", col_ra="RA_deg", col_dec="DE_deg", col_prob="P"),
    "cg20": LiteratureSchemaMeta(col_id="Source", col_ra="RA_deg", col_dec="DE_deg", col_prob="P_GMM"),
    "heyl": LiteratureSchemaMeta(col_id="Source", col_ra="RAdeg", col_dec="DEdeg", col_prob="P_GMM"),
    "ids_simbad": LiteratureSchemaMeta(col_id="gaia_dr3_id", col_ra="ra_cached", col_dec="dec_cached"),
    "dr2idx": LiteratureSchemaMeta(col_id="dr3_source_id", col_ra="ra", col_dec="dec")
}

# 原始历史文献元数据清单
LITERATURE_METADATA_ARRAY: Dict[str, AssetMeta] = {
    "hunt": AssetMeta("J/A+A/646/A104", "table1", "vizier/hunt21_m45.csv", "hunt", {
        "std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through
    }),
    "zerj": AssetMeta("J/MNRAS/493/3511", "table2", "vizier/zerjal19_m45.csv", "zerj", {
        "std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through
    }),
    "risb": AssetMeta("J/A+A/621/A2", "table3", "vizier/risbood19_m45.csv", "risb", {
        "std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through
    }),
    "cg20": AssetMeta("J/A+A/633/A99", "table1", "vizier/cg20_oc_members.csv", "cg20", {
        "std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through
    }),
    "heyl": AssetMeta("J/A+A/665/A22", "table1", "vizier/heyl22_members.csv", "heyl", {
        "std": StdActions.std_mapping, "stx": StxActions.pass_through, "aln": AlnActions.pass_through
    })
}


# =================================================================
# 4. 静态算法常数与超参白皮书 (Algorithm Constants & Hyperparameters)
# =================================================================
@dataclass
class GmmFeatureSpace:
    d2: List[str] = field(default_factory=lambda: ["pmra", "pmdec"])
    d3: List[str] = field(default_factory=lambda: ["pmra", "pmdec", "parallax"])
    d5: List[str] = field(default_factory=lambda: ["ra", "dec", "parallax", "pmra", "pmdec"])
    d6: List[str] = field(default_factory=lambda: ["ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity"])


@dataclass
class EngineHyperparameters:
    """管线核心精筛与粗筛科学计算核心不变参数常数白皮书"""
    feature_map: GmmFeatureSpace = field(default_factory=GmmFeatureSpace)
    dim_mode: str = "5d"
    ruwe_limit: float = 1.4
    cluster_algo: str = "dbscan"
    dbscan_eps: float = 0.3              # Castro-Ginard 自适应算法锚定基础分界线
    dbscan_min_samples: int = 100         # 粗筛本征突变信号底线星数
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
        """🌟 核心兼容垫片：保证外部算法数学组件通过传统字典形式传参时完全无损兼容"""
        if hasattr(self, key):
            val = getattr(self, key)
            if key == "feature_map" and isinstance(val, GmmFeatureSpace):
                return {"2d": val.d2, "3d": val.d3, "5d": val.d5, "6d": val.d6}
            return val
        raise KeyError(f"Hyperparameter '{key}' is not defined in engine core constants.")


# =================================================================
# 5. 强类型多态资产注册配置架构 (Unified Multi-type Asset Architecture)
# =================================================================
class AssetType(Enum):
    OBS_FIELD = "obs_field"          # 第一类：目标星团大范围观测大盘 (如 Gaia DR3 原始包)
    LIT_CATALOG = "lit_catalog"      # 第二类：来自 VizieR / 历史文献的审计星表
    DYNAMIC_CACHE = "dyn_cache"      # 第三类：SIMBAD / Archive 系统级高频在线资产缓存


@dataclass
class BaseAssetConfig:
    """所有数据资产的顶级抽象基类，死锁数据血缘行为"""
    id: str
    asset_type: AssetType
    base_idx: Optional[str] = None
    pre_filters: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """无损兼容旧代码的 dict-style .get() 路由"""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    def __getitem__(self, key: str) -> Any:
        """无损兼容旧代码的括号 bracket 访问（如 task_cfg['remote_catalog_id']）"""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else None
        
        # 路由兼容性防御垫片
        if key == "remote_catalog_id" and hasattr(self, "remote_catalog_id"):
            return self.remote_catalog_id
        if key == "remote_table_id" and hasattr(self, "remote_table_id"):
            return self.remote_table_id
        raise KeyError(f"'{key}' not found in AssetConfig '{self.id}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    @property
    def raw_table_name(self) -> str:
        """Stage 1.2 物理落表契约成果物命名核心定义"""
        if self.asset_type == AssetType.OBS_FIELD:
            short_id = self.id.replace("_field", "")
            return TMPL.T_RAW_OBS.format(idx=short_id)
        elif self.asset_type == AssetType.LIT_CATALOG:
            return TMPL.T_RAW_LIT.format(idx=self.id)
        elif self.asset_type == AssetType.DYNAMIC_CACHE:
            return TMPL.T_RAW_CSH.format(idx=self.id)
        return f"raw_unk_{self.id}"

    @property
    def name(self) -> str:
        """兼容陈旧模块对标准表/视图名称的映射检索"""
        return TMPL.T_TBL.format(idx=self.id)

    @property
    def raw_table(self) -> str:
        """兼容陈旧底层对原生无物理隔离层命名的历史追踪"""
        return TMPL.T_RAW.format(idx=self.base_idx if self.base_idx else self.id)

    @property
    def std_view(self) -> str:
        """Stage 1.5 提纯生成的只读标准视图名称"""
        if self.asset_type == AssetType.OBS_FIELD:
            return TMPL.V_STD.format(idx=self.id)
        elif self.asset_type == AssetType.LIT_CATALOG:
            return TMPL.V_CAT.format(idx=self.id)
        return TMPL.V_CSH.format(idx=self.id)

    @property
    def col_prob(self) -> Optional[str]:
        return TMPL.COL_PROB.format(idx=self.id) if self.asset_type == AssetType.LIT_CATALOG else None


@dataclass
class ObsFieldAssetConfig(BaseAssetConfig):
    """🚀 5.1 第一类：全大盘原始天体测量观测数据资产模型 (如百万级 Gaia DR3 磁盘包)"""
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


@dataclass
class LitCatalogAssetConfig(BaseAssetConfig):
    """🚀 5.2 第二类：VizieR 显式下载文献审计目标星表资产配置模型 (用于事后无偏交叉对齐)"""
    asset_type: AssetType = AssetType.LIT_CATALOG
    meta_type: str = "hunt"
    provider: str = "vizier"
    sync_mode: str = "HYBRID"

    @property
    def meta(self) -> AssetMeta:
        return LITERATURE_METADATA_ARRAY.get(self.meta_type, AssetMeta(None, None, "", "", {}))

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


@dataclass
class DynamicCacheAssetConfig(BaseAssetConfig):
    """🚀 5.3 第三类：动态在线网络快照缓存资产模型 (SIMBAD 天体别名、DR2/DR3版本交叉对齐索引)"""
    asset_type: AssetType = AssetType.DYNAMIC_CACHE
    index_column: str = "source_id"
    cache_strategy: str = "APPEND_ONLY"  # 严格强行死锁单向增量追加契优
    provider: str = "local_file"
    sync_mode: str = "VIRTUAL"

    @property
    def file_pattern(self) -> str:
        return f"internal/cache_{self.id}.parquet"

    @property
    def fields(self) -> LiteratureSchemaMeta:
        return LITERATURE_FIELDS_ARRAY.get(self.id, LiteratureSchemaMeta())

    @property
    def actions(self) -> Dict[str, Any]:
        return {"std": None, "stx": None, "aln": None}


# =================================================================
# 6. 全局单例多态清单容器 (Manifest Container & Instances)
# =================================================================
@dataclass(frozen=True)
class ManifestContainer:
    """全局资产清单注册大盘（全面替代原散装 CatalogConfig，保持100%括号无损兼容）"""

    # --- 1. 第一类：各天区全大盘基础观测数据资产注册表 (落表为 raw_obs_*) ---
    m45_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m45_field"))
    m44_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m44_field"))
    mel25_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="mel25_field"))
    mel111_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="mel111_field"))
    m67_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m67_field"))
    m13_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m13_field"))
    m41_field: ObsFieldAssetConfig = field(default_factory=lambda: ObsFieldAssetConfig(id="m41_field"))

    # --- 2. 派生衍生出的粗筛种子层资产 (Seeds Layer - 血缘归属于大盘) ---
    m45_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m45_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m45_field"))
    m44_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m44_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m44_field"))
    mel25_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="mel25_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="mel25_field"))
    mel111_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="mel111_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="mel111_field"))
    m67_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m67_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m67_field"))
    m13_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m13_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m13_field"))
    m41_seeds_field: BaseAssetConfig = field(default_factory=lambda: BaseAssetConfig(id="m41_seeds_field", asset_type=AssetType.OBS_FIELD, base_idx="m41_field"))

    # --- 3. 第二类：来自 VizieR 的历史文献真实审计目标参考星表 (落表为 raw_lit_*) ---
    hunt: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="hunt", meta_type="hunt"))
    zerj: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="zerj", meta_type="zerj"))
    risb: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="risb", meta_type="risb"))
    cg20: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="cg20", meta_type="cg20"))
    heyl: LitCatalogAssetConfig = field(default_factory=lambda: LitCatalogAssetConfig(id="heyl", meta_type="heyl"))

    # --- 4. 第三类：系统级在线快照动态缓存库 (落表为 raw_csh_*) ---
    ids_simbad: DynamicCacheAssetConfig = field(default_factory=lambda: DynamicCacheAssetConfig(id="ids_simbad", index_column="gaia_dr3_id"))
    dr2idx: DynamicCacheAssetConfig = field(default_factory=lambda: DynamicCacheAssetConfig(id="dr2idx", index_column="dr3_source_id"))

    def __getitem__(self, key: str) -> BaseAssetConfig:
        """核心兼容垫片：保证整个流水线旧版代码中 MANIFEST[task_id] 字典形式调用完全无损兼容"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Asset identity '{key}' is not registered in pipeline manifest container blueprint.")

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

    def get_assets_for_cluster(self, cluster_id: str):
        """
        获取当前星团环境下的所有资产配置。
        逻辑：返回所有全局资产 + 该星团专有的资产。
        """
        assets = {}
        # 遍历所有已注册的资产配置 (通过 dir() 或 dataclasses.fields())
        from dataclasses import fields
        for field in fields(self):
            asset_cfg = getattr(self, field.name)
            # 这里实现筛选逻辑：
            # 如果资产 ID 中包含 cluster_id 或标识为全局通用，则加入
            if asset_cfg.id == "common" or cluster_id in asset_cfg.id:
                assets[field.name] = asset_cfg
        return assets


# 全局单例实例化，统一对外暴露静态资产大盘
MANIFEST = ManifestContainer()
HYPERPARAMS = EngineHyperparameters()