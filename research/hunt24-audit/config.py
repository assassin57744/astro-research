# config.py
import os
from pathlib import Path
from typing import TypedDict, Dict, Any
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

RAW_DIR    = (DATA_DIR / "raw").resolve()
BACKUP_DIR = (DATA_DIR / "backups").resolve()
EXPORT_DIR = (DATA_DIR / "exports").resolve()
INTERNAL_DIR = (DATA_DIR / "internal").resolve()

# 数据源子目录
GAIA_INPUT_DIR   = (RAW_DIR / "gaia_archive").resolve()
VIZIER_INPUT_DIR = (RAW_DIR / "vizier").resolve()
SIMBAD_INPUT_DIR = (RAW_DIR / "simbad").resolve()
OAPD_INPUT_DIR   = (RAW_DIR / "oapd").resolve()
DOWNLOAD_DIR     = RAW_DIR 

# Gaia Archive 认证信息
GAIA_USER = os.getenv("GAIA_USER", "jli21")
GAIA_PWD  = os.getenv("GAIA_PWD") 


# =================================================================
# 2. 科学计算门限与物理常数 (Thresholds & Physics)
# =================================================================
MEMBER_SAMPLE_THRESHOLD = 0.2
GOLDEN_SAMPLE_THRESHOLD = 0.8

AUDIT_PROB_HIGH = 0.7            # 成员身份判定高门限
AUDIT_PROB_LOW  = 0.3            # 成员身份判定低门限（背景噪点）
AUDIT_RUWE_LIMIT = 1.4           # Gaia 天体测量质量门限
AUDIT_PLX_RESIDUAL_LIMIT = 1.0   # 视差残差允许度 (mas)
AUDIT_MAG_LIMIT_HUNT24 = 19.0    # Hunt2024 文献深度参考线

# 物理验证权重与细分容忍度
PHYS_VERIFY_WEIGHTS = {"pm": 0.4, "plx": 0.4, "cmd": 0.2}   # TODO: 需要调整到与星团相关, 且考虑到不同维度的加权
PHYS_VERIFY_PENALTY_LIMIT = 1.1
PHYS_LIT_PM_LIMIT = 1.5
PHYS_LIT_CMD_LIMIT = 3.0
REDDENING_RATIO_BP_RP = 0.52  # E(BP-RP) / A_G 比例系数 (基于 Gaia DR3 经验红化律)


# =================================================================
# 3. 命名规范、模板与适配器 (Naming & Adapters)
# =================================================================
CATALOG_NAMING_ADAPTER = {
    "zerj": {"M45": "Melotte 22"},
}

STD_COLS = {
    "ID": "id", "ID_DR2": "id_dr2",
    "RA": "ra", "DEC": "dec", 
    "PMRA": "pmra",  # 对应 pmra_cosdec (μ*α), 单位 mas/yr
    "PMDEC": "pmdec", # 对应 pmdec (μδ), 单位 mas/yr
    "PLX": "plx", "MAG": "mag", "COLOR": "color",
    "RV": "rv", "RUWE": "ruwe", "PROB": "prob",
    "GMM_PROB": "gmm_prob",
    "REF_PROB": "r_prob",
    "CLUSTER": "cluster",
}

# Master 表专用标签列名
MASTER_COLS = {
    "SEED_TYPE": "seed_type",   # raw_seed
    "DENSITY_TAG": "density_status", # core / noise
    "GMM_PROB": "prob",         # 算法计算概率
    "X_MATCH": "x_match_tag",   # Matched / PG_Only / Ref_Only
    "AUDIT": "audit_status",    # Confirmed / Candidate / Contamination
    "AUDIT_NOTE": "audit_note", # 审计备注（如：视差偏离、暗端漏检等）
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
    T_MASTER = "master_{cluster}_{category}_{mode}_{algo}" # [混合模式] 状态跟踪宽表
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
# 4. 数据注册键 (Registry Keys - IDX)
# =================================================================
# 4.1 核心字段与种子集 ID
IDX_FIELD_CLUSTER_M45      , IDX_FIELD_CLUSTER_M45_SEEDS      = "m45_field"    , "m45_seeds_field"
IDX_FIELD_CLUSTER_M44      , IDX_FIELD_CLUSTER_M44_SEEDS      = "m44_field"    , "m44_seeds_field"
IDX_FIELD_CLUSTER_MEL25    , IDX_FIELD_CLUSTER_MEL25_SEEDS    = "mel25_field"  , "mel25_seeds_field"
IDX_FIELD_CLUSTER_MEL111   , IDX_FIELD_CLUSTER_MEL111_SEEDS   = "mel111_field" , "mel111_seeds_field"
IDX_FIELD_CLUSTER_M67      , IDX_FIELD_CLUSTER_M67_SEEDS      = "m67_field"    , "m67_seeds_field"
IDX_FIELD_CLUSTER_M13      , IDX_FIELD_CLUSTER_M13_SEEDS      = "m13_field"    , "m13_seeds_field"
IDX_FIELD_CLUSTER_M41      , IDX_FIELD_CLUSTER_M41_SEEDS      = "m41_field"    , "m41_seeds_field"

# 4.2 参考文献星表 ID
IDX_CG20 = "cg20"
IDX_HEYL = "heyl"
IDX_ZERJ = "zerj"
IDX_RISB = "risb"
IDX_HUNT = "hunt"

# 4.3 基础设施与算法输出 ID
IDX_DR2IDX     = "dr2idx"
IDX_IDS_SIMBAD = "ids_simbad"
IDX_GMM        = "pgmm" 


# =================================================================
# 5. 星团物理先验配置 (Cluster Configurations)
# =================================================================
CLUSTERS = {
    "M45": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_M45,
        "SEED_IDX": IDX_FIELD_CLUSTER_M45_SEEDS,
        "NAME": "Pleiades",
        "SIMBAD_NAME": "Cl Melotte 22",
        "ID_NAME": "melotte_22",
        "CAT_NAME": "Melotte_22",
        "ISO_FILE": "pleiades_126myr.dat",
        # "CENTER_RA": 56.75, "CENTER_DEC": 24.12, "RADIUS": 17.78,
        # 基于hunt24的星团成员的均值
        "CENTER_RA": 56.61398997432307, "CENTER_DEC": 24.09029596042996, "RADIUS": 17.78, 
        "RA_MIN": 44.0, "RA_MAX": 66.0, "DEC_MIN": 16.0, "DEC_MAX": 36.0, "MAX_MAG": 21.0,
        "CORE_RADIUS": 1.3,  # 单位：pc
        "HALF_MASS_RADIUS": 3.5,  # 单位：pc
        "half_light_radius": 3.0,  # 单位：pc (保持原有小写变量名对齐)
        "TIDAL_RADIUS": 10.0,  # 单位：pc
        "DISTANCE_PC": 136.2,
        "DISTANCE_MODULUS": 5.66,
        "EXT_AG": 0.12,  # Gaia G波段消光
        "E_BP_RP": 0.06,  # 对应色余 E(BP-RP)
        # "PLX_REF": 7.33, "PMRA_REF": 20.10, "PMDEC_REF": -45.40,
        # 基于hunt24的星团成员的均值
        "PLX_REF": 7.329881851789656, 
        "PLX_ERROR": 0.5,  # 视差误差容忍度 (mas)
        "PMRA_REF": 19.816076644561296, "PMDEC_REF": -45.02481613280063,
        "PM_RADIUS": 3.0,  # 自行空间容忍度 (mas/yr)，参考 Hunt2024 Figure 3 分布范围
        "RV_REF": 5.63,
        "RV_ERROR": 5.0, # 视向速度容忍度 (km/s)        
        "UVW_REF": np.array([-6.05, -28.02, -14.34]),
        "V_ERROR": 2.0,  # 速度空间容忍度 (km/s)
        
        "CMD_DEV": 0.8,  # CMD 偏离容忍度 (mag)
        "KINE_SCORE_LIMIT": 2.0, # 动力学硬门槛     # TODO: 可以细化到分pm, plx, cmd, rv
        "SEED_RADIUS": 2.0, # 单位：deg, 源星种子搜索半径(第一次实验取值:2.0, 第二次实验取值:1.2)
        "SEED_PLX_LIM": 1.5,# 单位：mas, 源星种子搜索视差容忍度(第一次实验取值:1.5, 第二次实验取值:0.5)
        "SEED_MAX_MAG": 18.0, # 源星种子搜索最大亮度限制(第一次实验取值:18.0, 第二次实验取值:15.0)
        "SEED_MAX_RUWE": 1.2,
    },
    "M44": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_M44,
        "SEED_IDX": IDX_FIELD_CLUSTER_M44_SEEDS,
        "NAME": "Praesepe",
        "ID_NAME": "Melotte_88",
        "CAT_NAME": "NGC_2632",
        "ISO_FILE": "praesepe_700myr.dat",
        "CENTER_RA": 130.1, "CENTER_DEC": 19.7, "RADIUS": 11.90,
        "RA_MIN": 120.0, "RA_MAX": 140.0, "DEC_MIN": 10.0, "DEC_MAX": 30.0, "MAX_MAG": 21.0,
        "CORE_RADIUS": 0.8,  # 单位：pc
        "HALF_MASS_RADIUS": 3.9,  # 单位：pc
        "half_light_radius": 3.5,  # 单位：pc
        "TIDAL_RADIUS": 12.0,  # 单位：pc
        "DISTANCE_PC": 187.0,
        "DISTANCE_MODULUS": 6.36,
        "EXT_AG": 0.05,
        "E_BP_RP": 0.03,  # 补齐色余
        "PLX_REF": 5.35, "PMRA_REF": -36.0, "PMDEC_REF": -12.9,
        "RV_REF": 35.0,
        "UVW_REF": np.array([-34.5, -21.2, -6.8]),
        "V_ERROR": 2.0,
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 4.0,  # 自行半径容忍度 (mas/yr)
        "PLX_ERROR": 0.4,  # 视差误差/弥散容忍度 (mas)
        "CMD_DEV": 0.6,  # CMD 偏离容忍度 (mag)
        "SEED_RADIUS": 2.0,
        "SEED_PLX_LIM": 1.2,
        "SEED_MAX_MAG": 18.0,
        "SEED_MAX_RUWE": 1.4,
    },
    "Mel25": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_MEL25,
        "SEED_IDX": IDX_FIELD_CLUSTER_MEL25_SEEDS,
        "NAME": "Hyades",
        "ID_NAME": "melotte_25",
        "CAT_NAME": "Melotte_25",
        "ISO_FILE": "hyades_650myr.dat",
        "CENTER_RA": 66.75, "CENTER_DEC": 15.87, "RADIUS": 59.31,
        "RA_MIN": 50.0, "RA_MAX": 85.0, "DEC_MIN": 0.0, "DEC_MAX": 32.0, "MAX_MAG": 21.0,
        "CORE_RADIUS": 2.7,  # 单位：pc (约 8.8 光年)
        "HALF_MASS_RADIUS": 4.1,  # 单位：pc
        "half_light_radius": 3.1,  # 单位：pc (保持原有小写变量名)
        "TIDAL_RADIUS": 10.0,  # 单位：pc (经典重力潮汐半径，外部流失星形成延展星流)
        "DISTANCE_PC": 46.7,
        "DISTANCE_MODULUS": 3.35,
        "AV": 0.02,  # V波段尘埃消光 (位于本地泡内，尘埃消光极低，近乎为0)
        "EXT_AG": 0.01,  # Gaia G波段消光
        "E_BP_RP": 0.01,
        "PLX_REF": 21.41, "PMRA_REF": 101.10, "PMDEC_REF": -28.50,
        "RV_REF": 39.10,
        "UVW_REF": np.array([-42.24, -19.11, -1.45]),
        "V_ERROR": 3.0, # 毕宿星团极其靠近，投影效应导致的速度残差容忍度需放宽
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 12.0,  # 自行半径容忍度 (mas/yr)，离得太近导致自行发散严重
        "PLX_ERROR": 2.5,  # 视差误差/弥散容忍度 (mas)
        "CMD_DEV": 0.6,  # CMD 偏离容忍度 (mag) (主序带非常窄且干净)
        "SEED_RADIUS": 8.0,
        "SEED_PLX_LIM": 2.0,
        "SEED_MAX_MAG": 16.0,
        "SEED_MAX_RUWE": 1.2,
    },
    "Mel111": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_MEL111,
        "SEED_IDX": IDX_FIELD_CLUSTER_MEL111_SEEDS,
        "NAME": "ComaBer",
        "ID_NAME": "melotte_111",
        "CAT_NAME": "Melotte_111",
        "ISO_FILE": "mel111_500myr.dat",
        "CENTER_RA": 186.6, "CENTER_DEC": 26.1, "RADIUS": 42.61,
        "RA_MIN": 175.0, "RA_MAX": 198.0, "DEC_MIN": 15.0, "DEC_MAX": 37.0, "MAX_MAG": 21.0,
        "CORE_RADIUS": 1.5,  # 单位：pc
        "HALF_MASS_RADIUS": 4.5,
        "half_light_radius": 3.8,
        "TIDAL_RADIUS": 15.0,  # 作为一个弥散星团，其动力学边界较宽
        "DISTANCE_PC": 86.0,
        "DISTANCE_MODULUS": 4.67,
        "EXT_AG": 0.02,  # 高银纬天区，消光极低
        "E_BP_RP": 0.01,
        "PLX_REF": 11.60, "PMRA_REF": -12.11, "PMDEC_REF": -9.01,
        "RV_REF": -1.0,
        "UVW_REF": np.array([-1.7, -6.1, -1.3]),
        "V_ERROR": 2.5,
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 5.0,  # 自行散布容忍度
        "PLX_ERROR": 1.5,  # 视差绝对误差容忍度
        "CMD_DEV": 0.6,
        "SEED_RADIUS": 5.0,
        "SEED_PLX_LIM": 1.5,
        "SEED_MAX_MAG": 16.0,
        "SEED_MAX_RUWE": 1.2,
    },
    "M67": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_M67,
        "SEED_IDX": IDX_FIELD_CLUSTER_M67_SEEDS,
        "NAME": "M67",
        "ID_NAME": "ngc_2682",
        "CAT_NAME": "NGC_2682",
        "ISO_FILE": "m67_4000myr.dat",
        "DIM_MODE": "2d",
        "CENTER_RA": 132.83, "CENTER_DEC": 11.82, "RADIUS": 2.5,
        "RA_MIN": 128.0, "RA_MAX": 138.0, "DEC_MIN": 7.0, "DEC_MAX": 17.0, "MAX_MAG": 21.0,
        "CORE_RADIUS": 1.2,  # 单位：pc
        "HALF_MASS_RADIUS": 4.5,
        "half_light_radius": 3.8,
        "TIDAL_RADIUS": 16.0,
        "DISTANCE_PC": 850.0,
        "DISTANCE_MODULUS": 9.65,
        "EXT_AG": 0.10,
        "E_BP_RP": 0.05,
        "PLX_REF": 1.17, "PMRA_REF": -10.96, "PMDEC_REF": -2.94,
        "RV_REF": 33.7,
        "UVW_REF": np.array([-21.4, -25.2, -15.1]),
        "V_ERROR": 1.5, # 古老星团成员分布较为凝聚
        "RV_ERROR": 3.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 1.5,  # 远距离星团自行弥散极小
        "PLX_ERROR": 0.2,  # 视差容忍度收紧
        "CMD_DEV": 0.5,
        "SEED_RADIUS": 1.5,       # 继续扩大以包含更多外围种子
        "SEED_PLX_LIM": 0.4,       # 放宽视差限制以找回更多潜在种子
        "SEED_MAX_MAG": 20.0,
        "SEED_MAX_RUWE": 1.4,
        "SEED_PM_LIM": 2.5,
    },
    "M13": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_M13,
        "SEED_IDX": IDX_FIELD_CLUSTER_M13_SEEDS,
        "NAME": "M13",
        "ID_NAME": "ngc_6205",
        "CAT_NAME": "NGC_6205",
        "ISO_FILE": "m13_12gyr.dat",
        "DIM_MODE": "2d",
        "CENTER_RA": 250.42, "CENTER_DEC": 36.46, "RADIUS": 3.28,
        "RA_MIN": 248.0, "RA_MAX": 253.0, "DEC_MIN": 34.5, "DEC_MAX": 38.5, "MAX_MAG": 21.0,
        "CORE_RADIUS": 1.3,  # 单位：pc (核心致密)
        "HALF_MASS_RADIUS": 3.5,  # 单位：pc
        "half_light_radius": 3.2,
        "TIDAL_RADIUS": 43.0,  # 球状星团的潮汐半径通常较大
        "DISTANCE_PC": 7100.0,
        "DISTANCE_MODULUS": 14.25,
        "EXT_AG": 0.04,
        "E_BP_RP": 0.02,
        "PLX_REF": 0.14, "PMRA_REF": -3.18, "PMDEC_REF": -2.57,
        "RV_REF": -244.2,
        "UVW_REF": np.array([58.0, -241.0, 10.0]),  # 银晕轨道的典型运动学
        "V_ERROR": 10.0, # 球状星团内部速度弥散度极高
        "RV_ERROR": 10.0,
        "KINE_SCORE_LIMIT": 2.5,
        "PM_RADIUS": 1.0,  # 远距离天体自行离散度极小
        "PLX_ERROR": 0.1,  # 视差门限需非常严苛
        "CMD_DEV": 0.4,  # 球状星团主序带极其狭窄
        "SEED_RADIUS": 0.8, "SEED_PLX_LIM": 0.5,
        "SEED_MAX_MAG": 20.5,
        "SEED_MAX_RUWE": 1.4,
        "SEED_PM_LIM": 2.0,
    },
    "M41": {
        "FIELD_IDX": IDX_FIELD_CLUSTER_M41,
        "SEED_IDX": IDX_FIELD_CLUSTER_M41_SEEDS,
        "NAME": "M41",
        "ID_NAME": "ngc_2287",
        "CAT_NAME": "NGC_2287",
        "ISO_FILE": "m41_240myr.dat",
        "CENTER_RA": 101.50, "CENTER_DEC": -20.75, "RADIUS": 2.53,
        "RA_MIN": 96.0, "RA_MAX": 107.0, "DEC_MIN": -25.0, "DEC_MAX": -15.0, "MAX_MAG": 21.0,
        "CORE_RADIUS": 1.5,  # 单位：pc
        "HALF_MASS_RADIUS": 4.0,
        "half_light_radius": 3.6,
        "TIDAL_RADIUS": 12.0,
        "DISTANCE_PC": 710.0,
        "DISTANCE_MODULUS": 9.25,
        "EXT_AG": 0.05,  # 消光较低
        "E_BP_RP": 0.03,
        "PLX_REF": 1.41, "PMRA_REF": -1.55, "PMDEC_REF": -1.05,
        "RV_REF": 34.0,
        "UVW_REF": np.array([-10.5, -20.2, -5.1]),
        "V_ERROR": 2.0,
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 2.0,  # 较远星团，自行散布较小
        "PLX_ERROR": 0.3,
        "CMD_DEV": 0.6,
        "SEED_RADIUS": 2.0,
        "SEED_PLX_LIM": 0.9,  # PLX_ERROR * 3
        "SEED_MAX_MAG": 18.0,
        "SEED_MAX_RUWE": 1.4,
    },
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

# 6.2 配置辅助函数
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

# 6.3 核心数据清单
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

# =================================================================
# 7. 算法流水线配置 (Pipeline & GMM Configuration)
# =================================================================
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
    "cluster_algo": "dbscan", # 可选: dbscan, hdbscan
    "dbscan_eps": 0.3,  # 从 0.3 调大，补偿高维空间距离
    "dbscan_min_samples": 100,  # 3d模型为100
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 5,          # 提高门限，使种子核心更凝聚，减少杂质
    "hdbscan_cluster_selection_epsilon": 0.1, # 进一步降低合并容忍度，只保留最高密度的核心部分
    "gmm_covariance_type": "full",
    "max_iter": 20,
    "tol": 1e-5,
    "use_experimental": True,  # 启用实验性功能，如基于近邻的智能初始化
    "enable_subsampling": False,  # 是否启用背景下采样优化，以加速模型拟合
    "subsampling_limit": 500000, # 下采样触发门限及目标样本量
}
