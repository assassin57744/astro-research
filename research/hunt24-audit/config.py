# config.py
import os
import numpy as np
from pathlib import Path
from typing import TypedDict, Dict, Any

# 扁平化导入：假设 actions.py 已移动到 modules/ 根目录
from modules.actions import StdActions, StxActions, AlnActions


# -----------------------------------------------------------------
# 运行时上下文 (Runtime Context)
# -----------------------------------------------------------------
# 获取默认目标（仅用于 CLI 默认值，不建议在代码中修改这些变量）
DEFAULT_CLUSTER = os.getenv("ASTRO_TARGET_CLUSTER", "M13")
DEFAULT_CATEGORY = os.getenv("ASTRO_TARGET_CATEGORY", "hunt")

# -----------------------------------------------------------------
# 路径与门限定义
# -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.resolve()
LOG_DIR = (BASE_DIR / "logs").resolve()
DATA_DIR = (BASE_DIR / "data").resolve()
RAW_DIR = (DATA_DIR / "raw").resolve()
EXPORT_DIR = (DATA_DIR / "exports").resolve()
INTERNAL_DIR = (DATA_DIR / "internal").resolve()

# 数据来源子目录配置
GAIA_INPUT_DIR = (RAW_DIR / "gaia_archive").resolve()
VIZIER_INPUT_DIR = (RAW_DIR / "vizier").resolve()
SIMBAD_INPUT_DIR = (RAW_DIR / "simbad").resolve()
OAPD_INPUT_DIR = (RAW_DIR / "oapd").resolve()

DOWNLOAD_DIR = RAW_DIR # 配合精确路径匹配，统一以 raw 为基准

# Gaia Archive 认证信息
GAIA_USER = os.getenv("GAIA_USER", "jli21")
GAIA_PWD = os.getenv("GAIA_PWD") # 严禁硬编码密码

# -----------------------------------------------------------------
# 核心指示符与列名定义
# -----------------------------------------------------------------

# 成员识别目标区域
IDX_FIELD_CLUSTER_M45 = "m45_field"
IDX_FIELD_CLUSTER_M45_SEEDS = "m45_seeds_field"

IDX_FIELD_CLUSTER_M44 = "m44_field"
IDX_FIELD_CLUSTER_M44_SEEDS = "m44_seeds_field"

IDX_FIELD_CLUSTER_MEL25 = "mel25_field"
IDX_FIELD_CLUSTER_MEL25_SEEDS = "mel25_seeds_field"

IDX_FIELD_CLUSTER_MEL111 = "mel111_field"
IDX_FIELD_CLUSTER_MEL111_SEEDS = "mel111_seeds_field"

IDX_FIELD_CLUSTER_M67 = "m67_field"
IDX_FIELD_CLUSTER_M67_SEEDS = "m67_seeds_field"

IDX_FIELD_CLUSTER_M13 = "m13_field"
IDX_FIELD_CLUSTER_M13_SEEDS = "m13_seeds_field"

IDX_FIELD_CLUSTER_M41 = "m41_field"
IDX_FIELD_CLUSTER_M41_SEEDS = "m41_seeds_field"

# 参考星表唯一识别码 (整个系统通过这些 Index 来流通)
IDX_CG20 = "cg20"
IDX_HEYL = "heyl"
IDX_ZERJ = "zerj"
IDX_RISB = "risb"
IDX_HUNT = "hunt"

# 辅助表 (用于本地缓存)
IDX_DR2IDX = "dr2idx"
IDX_IDS_SIMBAD = "ids_simbad"
IDX_GMM = "pgmm"  # 自定义算法产出表标识

STD_COLS = {
    "ID": "id",
    "ID_DR2": "id_dr2",
    "RA": "ra",
    "DEC": "dec",
    "PMRA": "pmra",
    "PMDEC": "pmdec",
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


class TMPL:
    """数据血缘命名规范模板类。

    L1 (raw_): 原始数据层，保持原始列名
    L2 (std_): 标准化视图，列名对齐，初步清洗
    L2+(stx_): 标准化视图ex, 概率补齐, dr2id -> dr3id等
    L3 (aln_): 物理窗口对齐层，裁剪到特定天区
    """

    # --- 数据库表/视图名 ---
    T_RAW = "raw_{idx}"  # L1: 原始物理表
    V_STD = "std_{idx}"  # L2: 标准化视图
    V_STX = "stx_{idx}"  # L2+: 标准化视图增强
    T_ALN = "aln_{idx}"  # L3: 物理对齐表
    T_ALN_EX = "aln_{idx}_{cluster}"  # L3+: 物理对齐表增强, tag用于区分不同的对齐版本
    T_TBL = "tbl_{idx}"  # 表格逻辑名称

    # --- 算法结果与分析 ---
    T_RES_SG = "pgmm_{cluster}"  # SeedGMM 原始产出
    V_RES_SUB = "v_res_sg_{cluster}_{tag}"  # 结果子集视图名模板,tag说明子集的特征
    V_ALL = "v_wide_{cluster}"  # 集成所有参考星表的分析大宽表
    V_DIFF = "v_diff_vs_{idx}"  # 分歧源
    V_NEW = "v_new_vs_{idx}"  # 全新源
    V_MISS = "v_miss_vs_{idx}"  # 缺失源
    V_ADT = "v_audit_{category}_{cluster}"  # 审计专用视图，包含交叉比对的结果和分析字段

    # --- 动态列名 ---
    COL_PROB = "{idx}_prob"

    # --- 导出文件名 ---
    FILE_FITS = "Pleiades_{category}_vs_{idx}.fits"
    FILE_REPORT = "Validation_Report_{idx}_{date}.csv"


CLUSTERS = {
    "M45": {
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_M45,
        "SEED_IDX": IDX_FIELD_CLUSTER_M45_SEEDS,
        "NAME": "Pleiades",
        "ID_NAME": "melotte_22",
        "CAT_NAME": "Melotte_22",  # 在hunt2024中，Pleiades被标记为Melotte_22
        "ISO_FILE": "pleiades_126myr.dat",  # 昴星团参考等龄线约为 126 Myr (年轻星团)
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 56.75,
        "CENTER_DEC": 24.12,
        "RADIUS": 17.78,  # 天区搜索半径，单位：度 (覆盖整个星团及周边背景)
        "RA_MIN": 44.0,
        "RA_MAX": 66.0,
        "DEC_MIN": 16.0,
        "DEC_MAX": 36.0,
        "MAX_MAG": 21.0,  # 探测暗端截止视星等,大于hunt24的星团最大值: 20.620213
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 1.3,  # 单位：pc
        "HALF_MASS_RADIUS": 3.5,  # 单位：pc
        "half_light_radius": 3.0,  # 单位：pc (保持原有小写变量名对齐)
        "TIDAL_RADIUS": 10.0,  # 单位：pc
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 136.2,
        "DISTANCE_MODULUS": 5.66,  # 本征距离模数 (5 * log10(136.2) - 5)
        "EXT_AG": 0.12,  # Gaia G波段消光
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        # 观测空间运动学参考值
        "PLX_REF": 7.33,  # 平均视差 (mas)，对应距离约 136.2 pc
        "PMRA_REF": 20.10,  # 平均 pmRA (mas/yr)
        "PMDEC_REF": -45.40,  # 平均 pmDEC (mas/yr)
        # 物理空间运动学参考值（新增：用于支持 3d_v 和 6d_p 的高阶转换）
        "RV_REF": 5.63,  # 标准系统视向速度 (km/s)，用于暗星虚拟投影填充
        "UVW_REF": np.array([-6.05, -28.02, -14.34]),  # 标准三维空间速度基准
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 3.0,  # 自行半径容忍度 (mas/yr)，参考 Hunt2024 Figure 3 分布范围
        "PLX_ERROR": 0.5,  # 视差误差容忍度 (mas)
        "CMD_DEV": 0.8,  # CMD 偏离容忍度 (mag)
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 2.0,
        "SEED_PLX_LIM": 1.5,  # PLX_ERROR * 3
        "SEED_MAX_MAG": 18.0,
        "SEED_MAX_RUWE": 1.2,
    },
    # Praesepe, 又名 M44 或 NGC 2632，是一个距离较近的开放星团，年龄约为 600-750 Myr
    "M44": {
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_M44,
        "SEED_IDX": IDX_FIELD_CLUSTER_M44_SEEDS,
        "NAME": "Praesepe",  # Praesepe、蜂巢星团(Beehive)、鬼宿星团、NGC 2632
        "ID_NAME": "Melotte_88",
        "CAT_NAME": "NGC_2632",  # 在hunt2024中，Praesepe被标记为NGC_2632
        "ISO_FILE": "praesepe_700myr.dat",  # 鬼星团参考年龄约为 600~750 Myr
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 130.1,
        "CENTER_DEC": 19.7,
        "RADIUS": 11.90,  # 天区搜索半径，单位：度
        "RA_MIN": 120.0,
        "RA_MAX": 140.0,
        "DEC_MIN": 10.0,
        "DEC_MAX": 30.0,
        "MAX_MAG": 21.0,  # 探测暗端截止视星等, 大于hunt24的星团最大值: 20.663906
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 0.8,  # 单位：pc
        "HALF_MASS_RADIUS": 3.9,  # 单位：pc
        "half_light_radius": 3.5,  # 单位：pc
        "TIDAL_RADIUS": 12.0,  # 单位：pc
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 187.0,
        "DISTANCE_MODULUS": 6.36,  # 本征距离模数 (5 * log10(187.0) - 5)
        "EXT_AG": 0.05,
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        # 观测空间运动学参考值
        "PLX_REF": 5.35,  # 平均视差 (mas)，对应距离约 187.0 pc
        "PMRA_REF": -36.0,  # 平均 pmRA (mas/yr)
        "PMDEC_REF": -12.9,  # 平均 pmDEC (mas/yr)
        # 物理空间运动学参考值
        "RV_REF": 35.0,  # 标准系统视向速度 (km/s)
        "UVW_REF": np.array([-34.5, -21.2, -6.8]),  # 标准三维空间速度基准
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 4.0,  # 自行半径容忍度 (mas/yr)
        "PLX_ERROR": 0.4,  # 视差误差/弥散容忍度 (mas)
        "CMD_DEV": 0.6,  # CMD 偏离容忍度 (mag)
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 2.0,
        "SEED_PLX_LIM": 1.2,  # PLX_ERROR * 3
        "SEED_MAX_MAG": 18.0,
        "SEED_MAX_RUWE": 1.4,
    },
    # Melotte 25, 又名 Hyades (毕星团)，是距离太阳最近的开放星团，年龄约为 625-650 Myr
    "Mel25": {  # 范围有点大, 第二批识别目标, 主要用于测试管线在不同环境下的适应性和鲁棒性
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_MEL25,
        "SEED_IDX": IDX_FIELD_CLUSTER_MEL25_SEEDS,
        "NAME": "Hyades",
        "ID_NAME": "melotte_25",
        "CAT_NAME": "Melotte_25",  # 在hunt2024中，Hyades被标记为Melotte_25
        "ISO_FILE": "hyades_650myr.dat",  # 毕星团参考年龄约为 625~650 Myr
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 66.75,
        "CENTER_DEC": 15.87,
        "RADIUS": 59.31,  # 天区搜索半径，单位：度 (近距离星团需要大天区粗筛)
        "RA_MIN": 50.0,
        "RA_MAX": 85.0,
        "DEC_MIN": 0.0,
        "DEC_MAX": 32.0,
        "MAX_MAG": 21.0,  # 探测暗端截止视星等, hunt24的星团最大值: 20.608107
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 2.7,  # 单位：pc (约 8.8 光年)
        "HALF_MASS_RADIUS": 4.1,  # 单位：pc
        "half_light_radius": 3.1,  # 单位：pc (保持原有小写变量名)
        "TIDAL_RADIUS": 10.0,  # 单位：pc (经典重力潮汐半径，外部流失星形成延展星流)
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 46.7,
        "DISTANCE_MODULUS": 3.35,  # 本征距离模数 (5 * log10(46.7) - 5)
        "AV": 0.02,  # V波段尘埃消光 (位于本地泡内，尘埃消光极低，近乎为0)
        "EXT_AG": 0.01,  # Gaia G波段消光
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        # 观测空间运动学参考值
        "PLX_REF": 21.41,  # 平均视差 (mas)，对应距离约 46.7 pc
        "PMRA_REF": 101.10,  # 平均 pmRA (mas/yr) —— 典型的大自行星团
        "PMDEC_REF": -28.50,  # 平均 pmDEC (mas/yr)
        # 物理空间运动学参考值（新增：用于支持 3d_v 和 6d_p 的高阶转换）
        "RV_REF": 39.10,  # 标准系统视向速度 (km/s)，用于暗星虚拟投影填充
        "UVW_REF": np.array([-42.24, -19.11, -1.45]),  # 标准三维空间速度基准
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 12.0,  # 自行半径容忍度 (mas/yr)，离得太近导致自行发散严重
        "PLX_ERROR": 2.5,  # 视差误差/弥散容忍度 (mas)
        "CMD_DEV": 0.6,  # CMD 偏离容忍度 (mag) (主序带非常窄且干净)
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 8.0,
        "SEED_PLX_LIM": 2.0,
        "SEED_MAX_MAG": 16.0,
        "SEED_MAX_RUWE": 1.2,
    },
    # Melotte 111, 又名 Coma Berenices Cluster (后发座星团)，是一个距离较近的开放星团，年龄约为 450-500 Myr
    "Mel111": {  # 范围很大, 作为第二批识别目标,
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_MEL111,
        "SEED_IDX": IDX_FIELD_CLUSTER_MEL111_SEEDS,
        "NAME": "ComaBer",  # 后发座星团 (Coma Berenices Cluster)
        "ID_NAME": "melotte_111",
        "CAT_NAME": "Melotte_111",
        "ISO_FILE": "mel111_500myr.dat",  # 参考年龄约为 450~500 Myr
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 186.6,
        "CENTER_DEC": 26.1,
        "RADIUS": 42.61,  # 距离极近 (86pc)，视角范围非常大
        "RA_MIN": 175.0,
        "RA_MAX": 198.0,
        "DEC_MIN": 15.0,
        "DEC_MAX": 37.0,
        "MAX_MAG": 21.0,  # 探测暗端截止视星等, 大于hunt24的星团最大值: 20.357265
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 1.5,  # 单位：pc
        "HALF_MASS_RADIUS": 4.5,
        "half_light_radius": 3.8,
        "TIDAL_RADIUS": 15.0,  # 作为一个弥散星团，其动力学边界较宽
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 86.0,
        "DISTANCE_MODULUS": 4.67,  # 5 * log10(86.0) - 5
        "EXT_AG": 0.02,  # 高银纬天区，消光极低
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        "PLX_REF": 11.60,
        "PMRA_REF": -12.11,
        "PMDEC_REF": -9.01,
        "RV_REF": -1.0,  # 系统视向速度接近于0
        "UVW_REF": np.array([-1.7, -6.1, -1.3]),
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 5.0,  # 自行散布容忍度
        "PLX_ERROR": 1.5,  # 视差绝对误差容忍度
        "CMD_DEV": 0.6,
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 5.0,
        "SEED_PLX_LIM": 1.5,
        "SEED_MAX_MAG": 16.0,
        "SEED_MAX_RUWE": 1.2,
    },
    "M67": {
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_M67,
        "SEED_IDX": IDX_FIELD_CLUSTER_M67_SEEDS,
        "NAME": "M67",
        "ID_NAME": "ngc_2682",
        "CAT_NAME": "NGC_2682",
        "ISO_FILE": "m67_4000myr.dat",
        "DIM_MODE": "2d",             # 🚀 经验证，2D 模式对 M67 的召回率提升巨大
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 132.83,
        "CENTER_DEC": 11.82,
        "RADIUS": 2.5, # 实际数据下载半径为2.5,理论半径2.16
        "RA_MIN": 128.0,
        "RA_MAX": 138.0,
        "DEC_MIN": 7.0,
        "DEC_MAX": 17.0,
        "MAX_MAG": 21.0,  # hunt24的星团最大值: 20.584248
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 1.2,  # 单位：pc
        "HALF_MASS_RADIUS": 4.5,
        "half_light_radius": 3.8,
        "TIDAL_RADIUS": 16.0,
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 850.0,
        "DISTANCE_MODULUS": 9.65,  # 5 * log10(850) - 5
        "EXT_AG": 0.10,  # 低银纬，但消光相对适中
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        "PLX_REF": 1.17,  # 平均视差约 1.1-1.2 mas
        "PMRA_REF": -10.96,  # 典型自行
        "PMDEC_REF": -2.94,
        "RV_REF": 33.7,  # 视向速度显著
        "UVW_REF": np.array([-21.4, -25.2, -15.1]),
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 1.5,  # 远距离星团自行弥散极小
        "PLX_ERROR": 0.2,  # 视差容忍度收紧
        "CMD_DEV": 0.5,
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 1.5,       # 继续扩大以包含更多外围种子
        "SEED_PLX_LIM": 0.4,       # 放宽视差限制以找回更多潜在种子
        "SEED_MAX_MAG": 20.0,      # 🚀 添加缺失的星等门限，解决 KeyError
        "SEED_MAX_RUWE": 1.4,
        "SEED_PM_LIM": 2.5,        # 进一步放宽，确保拥挤区样本不因测量弥散被踢出
    },
    "M13": {  # 致密球状星团，距离较远，年龄极老，具有非常不同的物理和动力学特征，作为挑战性测试对象
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_M13,
        "SEED_IDX": IDX_FIELD_CLUSTER_M13_SEEDS,
        "NAME": "M13",  # 武仙座球状星团 (Hercules Globular Cluster)
        "ID_NAME": "ngc_6205",
        "CAT_NAME": "NGC_6205",
        "ISO_FILE": "m13_12gyr.dat",  # 极老星族，参考年龄约 11.5~12.0 Gyr
        "DIM_MODE": "2d",             # 🚀 经验证，2D 模式对 M13 的召回率提升巨大
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 250.42,
        "CENTER_DEC": 36.46,
        "RADIUS": 3.28,  # 远距离致密天体，1.5度已足够覆盖潮汐半径外围
        "RA_MIN": 248.0,
        "RA_MAX": 253.0,
        "DEC_MIN": 34.5,
        "DEC_MAX": 38.5,
        "MAX_MAG": 21.0,  # 20.520828是hunt24的星团最大值, 但考虑到球状星团中存在大量低质量的暗矮星，且我们希望测试管线在极端环境下的表现，因此将容忍度放宽到21.0
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 1.3,  # 单位：pc (核心致密)
        "HALF_MASS_RADIUS": 3.5,  # 单位：pc
        "half_light_radius": 3.2,
        "TIDAL_RADIUS": 43.0,  # 球状星团的潮汐半径通常较大
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 7100.0,
        "DISTANCE_MODULUS": 14.25,  # 5 * log10(7100) - 5
        "EXT_AG": 0.04,  # 高银纬，消光极低
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        "PLX_REF": 0.14,  # 极小视差 (约 7.1 kpc)
        "PMRA_REF": -3.18,  # 典型球状星团的高本征运动
        "PMDEC_REF": -2.57,
        "RV_REF": -244.2,  # 极高的负向视向速度，特征显著
        "UVW_REF": np.array([58.0, -241.0, 10.0]),  # 银晕轨道的典型运动学
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 1.0,  # 远距离天体自行离散度极小
        "PLX_ERROR": 0.1,  # 视差门限需非常严苛
        "CMD_DEV": 0.4,  # 球状星团主序带极其狭窄
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 0.8,       # 扩大范围，球状星团外围成员很多
        "SEED_PLX_LIM": 0.5,       # 考虑到 7kpc 的误差，放宽视差窗口
        "SEED_MAX_MAG": 20.5,      # 尽可能包含到 Hunt24 的深度
        "SEED_MAX_RUWE": 1.4,
        "SEED_PM_LIM": 2.0,        # 进一步放宽自行限制，对抗拥挤区的测量噪声
    },
    "M41": {
        # ==============================================================================
        # 1. 基础元数据 (Metadata)
        # ==============================================================================
        "FIELD_IDX": IDX_FIELD_CLUSTER_M41,
        "SEED_IDX": IDX_FIELD_CLUSTER_M41_SEEDS,
        "NAME": "M41",  # 小蜂巢星团 (Little Beehive Cluster)
        "ID_NAME": "ngc_2287",
        "CAT_NAME": "NGC_2287",
        "ISO_FILE": "m41_240myr.dat",  # 参考年龄约 200~240 Myr
        # ==============================================================================
        # 2. 天区截取与几何边界 (Spatial & Sky Boundary)
        # ==============================================================================
        "CENTER_RA": 101.50,
        "CENTER_DEC": -20.75,
        "RADIUS": 2.53,  # 核心覆盖范围约 1 度，取 2 度以包含晕族成员
        "RA_MIN": 96.0,
        "RA_MAX": 107.0,
        "DEC_MIN": -25.0,
        "DEC_MAX": -15.0,
        "MAX_MAG": 21.0,
        # ==============================================================================
        # 3. 空间物理结构半径 (Structural Radii)
        # ==============================================================================
        "CORE_RADIUS": 1.5,  # 单位：pc
        "HALF_MASS_RADIUS": 4.0,
        "half_light_radius": 3.6,
        "TIDAL_RADIUS": 12.0,
        # ==============================================================================
        # 4. 距离与消光物理环境 (Physics & Extinction Environment)
        # ==============================================================================
        "DISTANCE_PC": 710.0,
        "DISTANCE_MODULUS": 9.25,  # 5 * log10(710) - 5
        "EXT_AG": 0.05,  # 消光较低
        # ==============================================================================
        # 5. 天体测量与动力学先验基准 (Astrometry & Kinematics Priors)
        # ==============================================================================
        "PLX_REF": 1.41,  # 对应 ~710 pc
        "PMRA_REF": -1.55,  # 较小的自行特征
        "PMDEC_REF": -1.05,
        "RV_REF": 34.0,
        "UVW_REF": np.array([-10.5, -20.2, -5.1]),
        # ==============================================================================
        # 6. 经验约束与质量容忍度 (Tolerance & Quality Control)
        # ==============================================================================
        "PM_RADIUS": 2.0,  # 较远星团，自行散布较小
        "PLX_ERROR": 0.3,
        "CMD_DEV": 0.6,
        # --- 种子筛选动态配置 ---
        "SEED_RADIUS": 2.0,
        "SEED_PLX_LIM": 0.9,  # PLX_ERROR * 3
        "SEED_MAX_MAG": 18.0,
        "SEED_MAX_RUWE": 1.4,
    },
}

# -----------------------------------------------------------------
# 数据清单 (Manifest Registry)
# -----------------------------------------------------------------

# --- 字段映射模板 ---
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


def _make_gaia_entry(idx, file_pattern, fields=FIELDS_GAIA_ARCHIVE, pre_filters=None):
    """生成 Gaia 数据源标准配置项的辅助函数"""
    return {
        "name": TMPL.T_TBL.format(idx=idx),
        "idx": idx,
        "raw_table": TMPL.T_RAW.format(idx=idx),
        "std_view": TMPL.V_STD.format(idx=idx),
        "stx_view": TMPL.V_STX.format(idx=idx),
        "aln_view": TMPL.T_ALN.format(idx=idx),
        "sync_mode": "HYBRID",
        "provider": "local_file",
        "params": {
            "file_pattern": f"{GAIA_INPUT_DIR.name}/{file_pattern}",
            "storage_path": "snapshots",
        },
        "fields": fields,
        "pre_filters": pre_filters or [],
        "actions": {
            "std": StdActions.std_mapping,
            "stx": StxActions.pass_through,
            "aln": AlnActions.pass_through,
        },
    }


def _make_seed_entry(idx, base_idx, with_pm=False, pre_filters=None, fields=FIELDS_GAIA_ARCHIVE):
    """
    生成种子集配置项的辅助函数。
    种子集被视为基础字段表（Wide-field）的一个逻辑子集（View），不再对应物理文件。
    """
    if pre_filters is None:
        pre_filters = [
            "haversine_distance({CENTER_RA}, {CENTER_DEC}, ra, dec) < {SEED_RADIUS}",
            "abs(plx - {PLX_REF}) < {SEED_PLX_LIM}",
            "mag < {SEED_MAX_MAG}",
            "ruwe < {SEED_MAX_RUWE}",
        ]
        if with_pm:
            pre_filters.append(
                "abs(pmra - {PMRA_REF}) < {SEED_PM_LIM} AND abs(pmdec - {PMDEC_REF}) < {SEED_PM_LIM}"
            )

    return {
        "name": TMPL.T_TBL.format(idx=idx),
        "idx": idx,
        "base_idx": base_idx,
        # 关键变化：直接引用基础字段的原始表名，实现表结构与数据源复用
        "raw_table": TMPL.T_RAW.format(idx=base_idx),
        "std_view": TMPL.V_STD.format(idx=idx),
        "stx_view": TMPL.V_STX.format(idx=idx),
        "aln_view": TMPL.T_ALN.format(idx=idx),
        "sync_mode": "VIRTUAL",  # 标记为虚拟同步，DB 导入阶段将跳过物理文件检查
        "provider": "internal_view",
        "fields": fields,  # 种子视图必须与基础字段表的列映射保持一致
        "pre_filters": pre_filters,
        "actions": {
            "std": StdActions.std_mapping,
            "stx": StxActions.pass_through,
            "aln": AlnActions.pass_through,
        },
    }


def _make_catalog_entry(
    idx,
    file_pattern,
    fields,
    remote_cat=None,
    remote_tab="members",
    pre_filters=None,
    stx=StxActions.pass_through,
    use_ex_aln=True,
    **extra_params
):
    """生成外部参考星表 (Membership Catalog) 标准配置项的辅助函数"""
    params = {"storage_path": "snapshots", "file_pattern": f"{VIZIER_INPUT_DIR.name}/{file_pattern}"}
    params.update(extra_params)

    entry = {
        "name": TMPL.T_TBL.format(idx=idx),
        "idx": idx,
        "raw_table": TMPL.T_RAW.format(idx=idx),
        "std_view": TMPL.V_STD.format(idx=idx),
        "stx_view": TMPL.V_STX.format(idx=idx),
        "aln_view": (
            TMPL.T_ALN_EX.format(idx=idx, cluster="{cluster}")
            if use_ex_aln
            else TMPL.T_ALN.format(idx=idx)
        ),
        "sync_mode": "HYBRID",
        "provider": "local_file",
        "remote_catalog_id": remote_cat,
        "remote_table_id": remote_tab,
        "params": params,
        "fields": fields,
        "actions": {
            "std": StdActions.std_mapping,
            "stx": stx,
            "aln": AlnActions.pass_through,
        },
    }
    if use_ex_aln:
        entry["col_prob"] = TMPL.COL_PROB.format(idx=idx)
    if pre_filters:
        entry["pre_filters"] = pre_filters
    return entry


MANIFEST = {
    # ==============================================================================
    # 1. Target Cluster Data (Gaia DR3 Source Fields & Seed Samples)
    # ==============================================================================

    # --- M45: Pleiades ---
    IDX_FIELD_CLUSTER_M45: _make_gaia_entry(
        IDX_FIELD_CLUSTER_M45,
        "gaiadr3_m45_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_M45_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_M45_SEEDS,
        IDX_FIELD_CLUSTER_M45,
    ),

    # --- M44: Praesepe ---
    IDX_FIELD_CLUSTER_M44: _make_gaia_entry(
        IDX_FIELD_CLUSTER_M44,
        "gaiadr3_m44_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_M44_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_M44_SEEDS,
        IDX_FIELD_CLUSTER_M44,
    ),

    # --- Mel 25: Hyades ---
    IDX_FIELD_CLUSTER_MEL25: _make_gaia_entry(
        IDX_FIELD_CLUSTER_MEL25,
        "gaiadr3_mel25_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_MEL25_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_MEL25_SEEDS,
        IDX_FIELD_CLUSTER_MEL25,
    ),
    # --- Mel111: Coma Berenices ---
    IDX_FIELD_CLUSTER_MEL111: _make_gaia_entry(
        IDX_FIELD_CLUSTER_MEL111,
        "gaiadr3_mel111_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_MEL111_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_MEL111_SEEDS,
        IDX_FIELD_CLUSTER_MEL111,
    ),
    # --- M67: NGC 2682 ---
    IDX_FIELD_CLUSTER_M67: _make_gaia_entry(
        IDX_FIELD_CLUSTER_M67,
        "gaiadr3_m67_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_M67_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_M67_SEEDS,
        IDX_FIELD_CLUSTER_M67,
        with_pm=True,
    ),
    # --- M13: NGC 6205 (Globular Cluster) ---
    IDX_FIELD_CLUSTER_M13: _make_gaia_entry(
        IDX_FIELD_CLUSTER_M13,
        "gaiadr3_m13_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_M13_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_M13_SEEDS,
        IDX_FIELD_CLUSTER_M13,
        with_pm=True,
    ),
    # --- M41: NGC 2287 (Little Beehive) ---
    IDX_FIELD_CLUSTER_M41: _make_gaia_entry(
        IDX_FIELD_CLUSTER_M41,
        "gaiadr3_m41_wide.parquet",
    ),
    IDX_FIELD_CLUSTER_M41_SEEDS: _make_seed_entry(
        IDX_FIELD_CLUSTER_M41_SEEDS,
        IDX_FIELD_CLUSTER_M41,
    ),
    # ==============================================================================
    # 2. Infrastructure & Utility Indexes (Bridge Tables, Metadata)
    # ==============================================================================
    IDX_DR2IDX: {
        "name": TMPL.T_TBL.format(idx=IDX_DR2IDX),
        "idx": IDX_DR2IDX,
        "raw_table": TMPL.T_RAW.format(idx=IDX_DR2IDX),
        "std_view": TMPL.V_STD.format(idx=IDX_DR2IDX),
        # 程序中没有进行二次清理,stx与std保持一致
        "stx_view": TMPL.V_STX.format(idx=IDX_DR2IDX),
        "aln_view": TMPL.T_ALN.format(idx=IDX_DR2IDX),
        "sync_mode": "HYBRID",
        "provider": "local_file",
        "remote_catalog_id": "gaiadr3",
        "remote_table_id": "dr2_neighbourhood",
        "params": {
            "file_pattern": f"{GAIA_INPUT_DIR.name}/gaia_gaiadr3_dr2_neighbourhood_Melotte_22_r*",
            "storage_path": "snapshots",
        },
        "fields": {"id": "dr3_source_id", "id_dr2": "dr2_source_id"},
        "actions": {
            "std": StdActions.std_mapping,
            "stx": StxActions.pass_through,
            "aln": AlnActions.pass_through,
        },
    },
    IDX_IDS_SIMBAD: {
        "name": TMPL.T_TBL.format(idx=IDX_IDS_SIMBAD),
        "idx": IDX_IDS_SIMBAD,
        "raw_table": TMPL.T_RAW.format(idx=IDX_IDS_SIMBAD),
        "sync_mode": "HYBRID",
        "provider": "local_file",
        "params": {
            "storage_path": "snapshots",
            "file_pattern": f"{SIMBAD_INPUT_DIR.name}/local_ids_simbad.parquet",
            "id_col": "id",
            "prefix": "Gaia DR3 ",
            "optional": True,
        },
        "fields": {"id": "gaia_dr3_id", "main_id": "main_id", "ids": "ids"},
    },
    # ==============================================================================
    # 3. External Reference Catalogs (Literature Membership for Audit)
    # ==============================================================================
    IDX_HUNT: _make_catalog_entry(
        IDX_HUNT,
        "vizier_hunt24_full.parquet",
        fields={
            "id": "GaiaDR3",
            "prob": "Prob",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "pmra": "pmRA",
            "pmdec": "pmDE",
            "plx": "Plx",
            "mag": "Gmag",
            "color": "BP-RP",
            "cluster": "Name",
        },
        remote_cat="J/A+A/686/A42",
        pre_filters=["cluster = '{CAT_NAME}'"],
    ),
    IDX_ZERJ: _make_catalog_entry(
        IDX_ZERJ,
        "vizier_zerj23.parquet",
        fields={
            "id": "GaiaDR3",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "mag": "Gmag",
            "cluster": "Cluster",
        },
        remote_cat="J/A+A/686/A42",
        pre_filters=["cluster = '{CAT_NAME}'"],
    ),
    IDX_RISB: _make_catalog_entry(
        IDX_RISB,
        "vizier_risb25.vot",
        fields={
            "id": "GaiaDR3",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "pmra": "pmRA",
            "pmdec": "pmDE",
            "plx": "plx",
            "mag": "Gmag",
            "color": "BP-RP",
            "cluster": "Cluster",
        },
        remote_cat="J/A+A/694/A258",
        pre_filters=["cluster = '{CAT_NAME}'"],
    ),
    IDX_CG20: _make_catalog_entry(
        IDX_CG20,
        "vizier_cg2020_full.parquet",
        fields={
            "id_dr2": "Source",
            "prob": "Proba",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "pmra": "pmRA",
            "pmdec": "pmDE",
            "plx": "Plx",
            "mag": "Gmag",
            "color": "BP-RP",
            "cluster": "Cluster",
        },
        remote_cat="J/A+A/633/A99",
        stx=StxActions.bridge_dr2_to_dr3,
        pre_filters=["cluster = '{CAT_NAME}'"],
    ),
    IDX_HEYL: _make_catalog_entry(
        IDX_HEYL,
        "vizier_heyl22_*.parquet",
        fields={
            "id": "GaiaEDR3",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "pmra": "pmRA",
            "pmdec": "pmDE",
            "plx": "Plx",
            "mag": "Gmag",
            "color": "Bp-Rp",
        },
        remote_cat="J/ApJ/926/132",
        remote_tab="table4",
        stx=StxActions.fill_prob,
    ),
}

# -----------------------------------------------------------------
# 算法配置与物理常数
# -----------------------------------------------------------------
GMM_CONFIG = {
    "FULL_NAME": "SeedGMM",
    "SHORT_NAME": "SG",
    "feature_map": {
        "2d": ["pmra", "pmdec"],
        "3d": ["pmra", "pmdec", "plx"],
        "5d": ["ra", "dec", "pmra", "pmdec", "plx"],
        "6d_o": ["ra", "dec", "pmra", "pmdec", "plx", "rv"],
        # --- Hunt 2024 文献混合空间模式 ---
        "5d_h": ["l", "b", "pm_l_cosb", "pm_b", "plx"],  # 银道 5D 盲搜空间
        # --- 物理直角空间模式 (Physical) ---
        "3d_v": ["U", "V", "W"],  # 物理速度空间（纯动力学）
        "6d_p": ["X", "Y", "Z", "U", "V", "W"],  # 完整物理相空间
    },
    "dim_mode": "3d",
    "ruwe_limit": 1.4,
    "dbscan_eps": 0.3,  # 从 0.3 调大，补偿高维空间距离
    "dbscan_min_samples": 100,  # 3d模型为100
    "gmm_covariance_type": "full",
    "max_iter": 20,
    "tol": 1e-5,
    "use_experimental": True,  # 启用实验性功能，如基于近邻的智能初始化
}

MEMBER_SAMPLE_THRESHOLD = 0.2
GOLDEN_SAMPLE_THRESHOLD = 0.8  # 补全缺失的门限值
