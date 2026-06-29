# config.py
import os
from pathlib import Path
from typing import TypedDict, Dict, Any
import numpy as np

# 鍐呴儴妯″潡瀵煎叆
from modules.actions import StdActions, StxActions, AlnActions


# =================================================================
# 1. 绯荤粺璺緞涓庣幆澧冮厤缃?(Paths & Environment)
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

# 鏁版嵁婧愬瓙鐩綍
GAIA_INPUT_DIR   = (RAW_DIR / "gaia_archive").resolve()
VIZIER_INPUT_DIR = (RAW_DIR / "vizier").resolve()
SIMBAD_INPUT_DIR = (RAW_DIR / "simbad").resolve()
OAPD_INPUT_DIR   = (RAW_DIR / "oapd").resolve()
DOWNLOAD_DIR     = RAW_DIR 

# Gaia Archive 璁よ瘉淇℃伅
GAIA_USER = os.getenv("GAIA_USER", "jli21")
GAIA_PWD  = os.getenv("GAIA_PWD") 


# =================================================================
# 2. 绉戝璁＄畻闂ㄩ檺涓庣墿鐞嗗父鏁?(Thresholds & Physics)
# =================================================================
MEMBER_SAMPLE_THRESHOLD = 0.2
GOLDEN_SAMPLE_THRESHOLD = 0.8

AUDIT_PROB_HIGH = 0.7            # 鎴愬憳韬唤鍒ゅ畾楂橀棬闄?AUDIT_PROB_LOW  = 0.3            # 鎴愬憳韬唤鍒ゅ畾浣庨棬闄愶紙鑳屾櫙鍣偣锛?AUDIT_RUWE_LIMIT = 1.4           # Gaia 澶╀綋娴嬮噺璐ㄩ噺闂ㄩ檺
AUDIT_PLX_RESIDUAL_LIMIT = 1.0   # 瑙嗗樊娈嬪樊鍏佽搴?(mas)
AUDIT_MAG_LIMIT_HUNT24 = 19.0    # Hunt2024 鏂囩尞娣卞害鍙傝€冪嚎

# 鐗╃悊楠岃瘉鏉冮噸涓庣粏鍒嗗蹇嶅害
PHYS_VERIFY_WEIGHTS = {"pm": 0.4, "plx": 0.4, "cmd": 0.2}   # TODO: 闇€瑕佽皟鏁村埌涓庢槦鍥㈢浉鍏? 涓旇€冭檻鍒颁笉鍚岀淮搴︾殑鍔犳潈
PHYS_VERIFY_PENALTY_LIMIT = 1.1
PHYS_LIT_PM_LIMIT = 1.5
PHYS_LIT_CMD_LIMIT = 3.0
REDDENING_RATIO_BP_RP = 0.52  # E(BP-RP) / A_G 姣斾緥绯绘暟 (鍩轰簬 Gaia DR3 缁忛獙绾㈠寲寰?


# =================================================================
# 3. 鍛藉悕瑙勮寖銆佹ā鏉夸笌閫傞厤鍣?(Naming & Adapters)
# =================================================================
CATALOG_NAMING_ADAPTER = {
    "zerj": {"M45": "Melotte 22"},
}

STD_COLS = {
    "ID": "id", "ID_DR2": "id_dr2",
    "RA": "ra", "DEC": "dec", 
    "PMRA": "pmra",  # 瀵瑰簲 pmra_cosdec (渭*伪), 鍗曚綅 mas/yr
    "PMDEC": "pmdec", # 瀵瑰簲 pmdec (渭未), 鍗曚綅 mas/yr
    "PLX": "plx", "MAG": "mag", "COLOR": "color",
    "RV": "rv", "RUWE": "ruwe", "PROB": "prob",
    "GMM_PROB": "gmm_prob",
    "REF_PROB": "r_prob",
    "CLUSTER": "cluster",
}

# Master 琛ㄤ笓鐢ㄦ爣绛惧垪鍚?MASTER_COLS = {
    "SEED_TYPE": "seed_type",   # raw_seed
    "DENSITY_TAG": "density_status", # core / noise
    "GMM_PROB": "prob",         # 绠楁硶璁＄畻姒傜巼
    "X_MATCH": "x_match_tag",   # Matched / PG_Only / Ref_Only
    "AUDIT": "audit_status",    # Confirmed / Candidate / Contamination
    "AUDIT_NOTE": "audit_note", # 瀹¤澶囨敞锛堝锛氳宸亸绂汇€佹殫绔紡妫€绛夛級
}

class TMPL:
    # --- 鏁版嵁搴撹〃/瑙嗗浘鍚?---
    T_RAW = "raw_{idx}"  # L1: 鍘熷鐗╃悊琛?    V_STD = "std_{idx}"  # L2: 鏍囧噯鍖栬鍥?    V_STX = "stx_{idx}"  # L2+: 鏍囧噯鍖栬鍥惧寮?    T_ALN = "aln_{idx}"  # L3: 鐗╃悊瀵归綈琛?    T_ALN_EX = "aln_{idx}_{cluster}"  # L3+: 鐗╃悊瀵归綈琛ㄥ寮? tag鐢ㄤ簬鍖哄垎涓嶅悓鐨勫榻愮増鏈?    T_TBL = "tbl_{idx}"  # 琛ㄦ牸閫昏緫鍚嶇О

    # --- 绠楁硶缁撴灉涓庡垎鏋?---
    T_RES_SG = "pgmm_{cluster}_{category}_{mode}_{algo}"  # SeedGMM 鍘熷浜у嚭
    T_MASTER = "master_{cluster}_{category}_{mode}_{algo}" # [娣峰悎妯″紡] 鐘舵€佽窡韪琛?    V_RES_SUB = "v_pgmm_{cluster}_{category}_{mode}_{algo}_{tag}"  # 缁撴灉瀛愰泦瑙嗗浘鍚嶆ā鏉?    V_ALL = "v_wide_{cluster}_{category}_{mode}_{algo}"  # 闆嗘垚鎵€鏈夊弬鑰冩槦琛ㄧ殑鍒嗘瀽澶у琛?    V_DIFF = "v_diff_{cluster}_{category}_{mode}_{algo}_vs_{idx}"  # 鍒嗘婧?    V_NEW = "v_new_{cluster}_{category}_{mode}_{algo}_vs_{idx}"  # 鍏ㄦ柊鍙戠幇婧?    V_MISS = "v_miss_{cluster}_{category}_{mode}_{algo}_vs_{idx}"  # 婕忔婧?    V_ADT = "v_audit_{category}_{cluster}_{mode}_{algo}"  # 瀹¤涓撶敤瑙嗗浘
    V_ADT_INPUT = "v_audit_input_{src}"  # 瀹¤杈撳叆澧炲己瑙嗗浘
    V_AUDITED = "{src}_audited"  # 瀹¤瀹屾垚鍚庣殑鐗╁寲琛ㄥ悕
    V_ADT_HUNT24 = "audit_report_hunt24_by_{src}"  # 閽堝 Hunt24 鐨勪笓椤瑰璁＄粨鏋?
    # --- 鍔ㄦ€佸垪鍚?---
    COL_PROB = "{idx}_prob"

    # --- 瀵煎嚭鏂囦欢鍚?---
    FILE_FITS = "Pleiades_{category}_vs_{idx}.fits"
    FILE_REPORT = "Validation_Report_{idx}_{date}.csv"
    FILE_LIT_REPORT = "Lit_Audit_{label}_{cluster}_{timestamp}.csv"  # 鏂囩尞瀹¤鎶ュ憡
    FILE_PLOT = "{cluster}_{category}_{mode}_{prefix}_{timestamp}.png"  # 璇婃柇鍥捐〃鏂囦欢鍚?    FILE_NEW_CANDIDATES = "new_candidates_vs_{ref}.csv"  # 鏂板€欓€夎€呴獙璇佹竻鍗?    FILE_EXPORT_BASE = "{cluster}_{category}_{mode}_{algo}"  # 瀵煎嚭鏂囦欢鍚嶇殑鍩烘湰鍓嶇紑
    FILE_SEEDS = "{base}_seeds"  # 鍏ㄩ噺绉嶅瓙鏄熷鍑烘枃浠跺悕妯℃澘
    FILE_SEEDS_CORE = "{base}_seeds_core"  # 鏍稿績绉嶅瓙鏄熷鍑烘枃浠跺悕妯℃澘
    FILE_SEEDS_NOISE = "{base}_seeds_noise"  # 鍣０绉嶅瓙鏄熷鍑烘枃浠跺悕妯℃澘
    FILE_CROSS_SUMMARY = "{base}_cross_summary"  # 浜ゅ弶姣斿姹囨€绘枃浠跺悕
    FILE_DEEP_AUDIT = "{base}_deep_audit"  # 娣卞害瀹¤鎶ュ憡鏂囦欢鍚?    FILE_FINAL_REPORT = "{base}_final_report.txt"  # 鏈€缁堟墽琛屾憳瑕佹枃浠跺悕
    FILE_MISS_MAG_DIST = "hunt24_missing_mag_dist.png"  # 婕忔婧愭槦绛夊垎甯冨浘


# =================================================================
# 4. 鏁版嵁娉ㄥ唽閿?(Registry Keys - IDX)
# =================================================================
# 4.1 鏍稿績瀛楁涓庣瀛愰泦 ID
IDX_FIELD_CLUSTER_M45      , IDX_FIELD_CLUSTER_M45_SEEDS      = "m45_field"    , "m45_seeds_field"
IDX_FIELD_CLUSTER_M44      , IDX_FIELD_CLUSTER_M44_SEEDS      = "m44_field"    , "m44_seeds_field"
IDX_FIELD_CLUSTER_MEL25    , IDX_FIELD_CLUSTER_MEL25_SEEDS    = "mel25_field"  , "mel25_seeds_field"
IDX_FIELD_CLUSTER_MEL111   , IDX_FIELD_CLUSTER_MEL111_SEEDS   = "mel111_field" , "mel111_seeds_field"
IDX_FIELD_CLUSTER_M67      , IDX_FIELD_CLUSTER_M67_SEEDS      = "m67_field"    , "m67_seeds_field"
IDX_FIELD_CLUSTER_M13      , IDX_FIELD_CLUSTER_M13_SEEDS      = "m13_field"    , "m13_seeds_field"
IDX_FIELD_CLUSTER_M41      , IDX_FIELD_CLUSTER_M41_SEEDS      = "m41_field"    , "m41_seeds_field"

# 4.2 鍙傝€冩枃鐚槦琛?ID
IDX_CG20 = "cg20"
IDX_HEYL = "heyl"
IDX_ZERJ = "zerj"
IDX_RISB = "risb"
IDX_HUNT = "hunt"

# 4.3 鍩虹璁炬柦涓庣畻娉曡緭鍑?ID
IDX_DR2IDX     = "dr2idx"
IDX_IDS_SIMBAD = "ids_simbad"
IDX_GMM        = "pgmm" 


# =================================================================
# 5. 鏄熷洟鐗╃悊鍏堥獙閰嶇疆 (Cluster Configurations)
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
        # 鍩轰簬hunt24鐨勬槦鍥㈡垚鍛樼殑鍧囧€?        "CENTER_RA": 56.61398997432307, "CENTER_DEC": 24.09029596042996, "RADIUS": 17.78, 
        "RA_MIN": 44.0, "RA_MAX": 66.0, "DEC_MIN": 16.0, "DEC_MAX": 36.0, 
        "MAX_MAG": 21.0,
        "CORE_RADIUS": 1.3,  # 鍗曚綅锛歱c
        "HALF_MASS_RADIUS": 3.5,  # 鍗曚綅锛歱c
        "R_HALF_LIGHT": 3.0,  # 鍗曚綅锛歱c (淇濇寔鍘熸湁灏忓啓鍙橀噺鍚嶅榻?
        "TIDAL_RADIUS": 10.0,  # 鍗曚綅锛歱c
        "DISTANCE_PC": 136.2,
        "DISTANCE_MODULUS": 5.66,
        "EXT_AG": 0.12,  # Gaia G娉㈡娑堝厜
        "E_BP_RP": 0.06,  # 瀵瑰簲鑹蹭綑 E(BP-RP)
        # "PLX_REF": 7.33, "PMRA_REF": 20.10, "PMDEC_REF": -45.40,   
        "PLX_REF": 7.329881851789656,   # source: 鍩轰簬hunt24鐨勬槦鍥㈡垚鍛樼殑鍧囧€?        "PLX_ERROR": 0.5,  # 瑙嗗樊璇樊瀹瑰繊搴?(mas)
        "PMRA_REF": 19.816076644561296, "PMDEC_REF": -45.02481613280063,
        "PMRA_DISPERSION": 1.5, "PMDEC_DISPERSION": 1.2, # 鑷绌洪棿鍒嗘暎搴?(mas/yr), 闈?3d_v/6d_p 妯″紡涓嬬敓鏁?        "PM_RADIUS": 3.0,  # 鑷绌洪棿瀹瑰繊搴?(mas/yr)锛宻ource: Hunt2024 Figure 3 鍒嗗竷鑼冨洿
        "RV_REF": 5.63,
        "RV_ERROR": 5.0, # 瑙嗗悜閫熷害瀹瑰繊搴?(km/s)        
        "UVW_REF": np.array([-6.05, -28.02, -14.34]),
        "UVW_ERROR": 2.0,  # 閫熷害绌洪棿瀹瑰繊搴?(km/s)
        "U_ERROR": 2.2, "V_ERROR": 1.6, "W_ERROR": 1.0, # 閫熷害绌洪棿鍒嗘暎搴?(km/s), 浠?3d_v/6d_p 妯″紡涓嬬敓鏁?        "CMD_REF": np.array([0.0, 0.0, 0.0]),
        "CMD_DEV": 0.8,  # CMD 鍋忕瀹瑰繊搴?(mag)
        "KINE_SCORE_LIMIT": 2.0, # 鍔ㄥ姏瀛︾‖闂ㄦ     # TODO: 鍙互缁嗗寲鍒板垎pm, plx, cmd, rv
        "SEED_RADIUS": 2.0, # 鍗曚綅锛歞eg, 婧愭槦绉嶅瓙鎼滅储鍗婂緞(绗竴娆″疄楠屽彇鍊?2.0, 绗簩娆″疄楠屽彇鍊?1.2)
        "SEED_PLX_LIM": 1.5,# 鍗曚綅锛歮as, 婧愭槦绉嶅瓙鎼滅储瑙嗗樊瀹瑰繊搴?绗竴娆″疄楠屽彇鍊?1.5, 绗簩娆″疄楠屽彇鍊?0.5)
        "SEED_MAX_MAG": 18.0, # 婧愭槦绉嶅瓙鎼滅储鏈€澶т寒搴﹂檺鍒?绗竴娆″疄楠屽彇鍊?18.0, 绗簩娆″疄楠屽彇鍊?15.0)
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
        "CORE_RADIUS": 0.8,  # 鍗曚綅锛歱c
        "HALF_MASS_RADIUS": 3.9,  # 鍗曚綅锛歱c
        "half_light_radius": 3.5,  # 鍗曚綅锛歱c
        "TIDAL_RADIUS": 12.0,  # 鍗曚綅锛歱c
        "DISTANCE_PC": 187.0,
        "DISTANCE_MODULUS": 6.36,
        "EXT_AG": 0.05,
        "E_BP_RP": 0.03,  # 琛ラ綈鑹蹭綑
        "PLX_REF": 5.35, "PMRA_REF": -36.0, "PMDEC_REF": -12.9,
        "RV_REF": 35.0,
        "UVW_REF": np.array([-34.5, -21.2, -6.8]),
        "V_ERROR": 2.0,
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 4.0,  # 鑷鍗婂緞瀹瑰繊搴?(mas/yr)
        "PLX_ERROR": 0.4,  # 瑙嗗樊璇樊/寮ユ暎瀹瑰繊搴?(mas)
        "CMD_DEV": 0.6,  # CMD 鍋忕瀹瑰繊搴?(mag)
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
        "CORE_RADIUS": 2.7,  # 鍗曚綅锛歱c (绾?8.8 鍏夊勾)
        "HALF_MASS_RADIUS": 4.1,  # 鍗曚綅锛歱c
        "half_light_radius": 3.1,  # 鍗曚綅锛歱c (淇濇寔鍘熸湁灏忓啓鍙橀噺鍚?
        "TIDAL_RADIUS": 10.0,  # 鍗曚綅锛歱c (缁忓吀閲嶅姏娼睈鍗婂緞锛屽閮ㄦ祦澶辨槦褰㈡垚寤跺睍鏄熸祦)
        "DISTANCE_PC": 46.7,
        "DISTANCE_MODULUS": 3.35,
        "AV": 0.02,  # V娉㈡灏樺焹娑堝厜 (浣嶄簬鏈湴娉″唴锛屽皹鍩冩秷鍏夋瀬浣庯紝杩戜箮涓?)
        "EXT_AG": 0.01,  # Gaia G娉㈡娑堝厜
        "E_BP_RP": 0.01,
        "PLX_REF": 21.41, "PMRA_REF": 101.10, "PMDEC_REF": -28.50,
        "RV_REF": 39.10,
        "UVW_REF": np.array([-42.24, -19.11, -1.45]),
        "V_ERROR": 3.0, # 姣曞鏄熷洟鏋佸叾闈犺繎锛屾姇褰辨晥搴斿鑷寸殑閫熷害娈嬪樊瀹瑰繊搴﹂渶鏀惧
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 12.0,  # 鑷鍗婂緞瀹瑰繊搴?(mas/yr)锛岀寰楀お杩戝鑷磋嚜琛屽彂鏁ｄ弗閲?        "PLX_ERROR": 2.5,  # 瑙嗗樊璇樊/寮ユ暎瀹瑰繊搴?(mas)
        "CMD_DEV": 0.6,  # CMD 鍋忕瀹瑰繊搴?(mag) (涓诲簭甯﹂潪甯哥獎涓斿共鍑€)
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
        "CORE_RADIUS": 1.5,  # 鍗曚綅锛歱c
        "HALF_MASS_RADIUS": 4.5,
        "half_light_radius": 3.8,
        "TIDAL_RADIUS": 15.0,  # 浣滀负涓€涓讥鏁ｆ槦鍥紝鍏跺姩鍔涘杈圭晫杈冨
        "DISTANCE_PC": 86.0,
        "DISTANCE_MODULUS": 4.67,
        "EXT_AG": 0.02,  # 楂橀摱绾ぉ鍖猴紝娑堝厜鏋佷綆
        "E_BP_RP": 0.01,
        "PLX_REF": 11.60, "PMRA_REF": -12.11, "PMDEC_REF": -9.01,
        "RV_REF": -1.0,
        "UVW_REF": np.array([-1.7, -6.1, -1.3]),
        "V_ERROR": 2.5,
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 5.0,  # 鑷鏁ｅ竷瀹瑰繊搴?        "PLX_ERROR": 1.5,  # 瑙嗗樊缁濆璇樊瀹瑰繊搴?        "CMD_DEV": 0.6,
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
        "CORE_RADIUS": 1.2,  # 鍗曚綅锛歱c
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
        "V_ERROR": 1.5, # 鍙よ€佹槦鍥㈡垚鍛樺垎甯冭緝涓哄嚌鑱?        "RV_ERROR": 3.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 1.5,  # 杩滆窛绂绘槦鍥㈣嚜琛屽讥鏁ｆ瀬灏?        "PLX_ERROR": 0.2,  # 瑙嗗樊瀹瑰繊搴︽敹绱?        "CMD_DEV": 0.5,
        "SEED_RADIUS": 1.5,       # 缁х画鎵╁ぇ浠ュ寘鍚洿澶氬鍥寸瀛?        "SEED_PLX_LIM": 0.4,       # 鏀惧瑙嗗樊闄愬埗浠ユ壘鍥炴洿澶氭綔鍦ㄧ瀛?        "SEED_MAX_MAG": 20.0,
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
        "CORE_RADIUS": 1.3,  # 鍗曚綅锛歱c (鏍稿績鑷村瘑)
        "HALF_MASS_RADIUS": 3.5,  # 鍗曚綅锛歱c
        "half_light_radius": 3.2,
        "TIDAL_RADIUS": 43.0,  # 鐞冪姸鏄熷洟鐨勬疆姹愬崐寰勯€氬父杈冨ぇ
        "DISTANCE_PC": 7100.0,
        "DISTANCE_MODULUS": 14.25,
        "EXT_AG": 0.04,
        "E_BP_RP": 0.02,
        "PLX_REF": 0.14, "PMRA_REF": -3.18, "PMDEC_REF": -2.57,
        "RV_REF": -244.2,
        "UVW_REF": np.array([58.0, -241.0, 10.0]),  # 閾舵檿杞ㄩ亾鐨勫吀鍨嬭繍鍔ㄥ
        "V_ERROR": 10.0, # 鐞冪姸鏄熷洟鍐呴儴閫熷害寮ユ暎搴︽瀬楂?        "RV_ERROR": 10.0,
        "KINE_SCORE_LIMIT": 2.5,
        "PM_RADIUS": 1.0,  # 杩滆窛绂诲ぉ浣撹嚜琛岀鏁ｅ害鏋佸皬
        "PLX_ERROR": 0.1,  # 瑙嗗樊闂ㄩ檺闇€闈炲父涓ヨ嫑
        "CMD_DEV": 0.4,  # 鐞冪姸鏄熷洟涓诲簭甯︽瀬鍏剁嫮绐?        "SEED_RADIUS": 0.8, "SEED_PLX_LIM": 0.5,
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
        "CORE_RADIUS": 1.5,  # 鍗曚綅锛歱c
        "HALF_MASS_RADIUS": 4.0,
        "half_light_radius": 3.6,
        "TIDAL_RADIUS": 12.0,
        "DISTANCE_PC": 710.0,
        "DISTANCE_MODULUS": 9.25,
        "EXT_AG": 0.05,  # 娑堝厜杈冧綆
        "E_BP_RP": 0.03,
        "PLX_REF": 1.41, "PMRA_REF": -1.55, "PMDEC_REF": -1.05,
        "RV_REF": 34.0,
        "UVW_REF": np.array([-10.5, -20.2, -5.1]),
        "V_ERROR": 2.0,
        "RV_ERROR": 5.0,
        "KINE_SCORE_LIMIT": 2.0,
        "PM_RADIUS": 2.0,  # 杈冭繙鏄熷洟锛岃嚜琛屾暎甯冭緝灏?        "PLX_ERROR": 0.3,
        "CMD_DEV": 0.6,
        "SEED_RADIUS": 2.0,
        "SEED_PLX_LIM": 0.9,  # PLX_ERROR * 3
        "SEED_MAX_MAG": 18.0,
        "SEED_MAX_RUWE": 1.4,
    },
}


# =================================================================
# 6. 鏁版嵁娓呭崟 (Manifest Registry)
# =================================================================
# 6.1 瀛楁鏄犲皠妯℃澘
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

# 6.2 閰嶇疆杈呭姪鍑芥暟
def _make_gaia_entry(idx, file_pattern, fields=FIELDS_GAIA_ARCHIVE, pre_filters=None):
    """鐢熸垚 Gaia 鏁版嵁婧愭爣鍑嗛厤缃」鐨勮緟鍔╁嚱鏁?""
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
    鐢熸垚绉嶅瓙闆嗛厤缃」鐨勮緟鍔╁嚱鏁般€?    绉嶅瓙闆嗚瑙嗕负鍩虹瀛楁琛紙Wide-field锛夌殑涓€涓€昏緫瀛愰泦锛圴iew锛夛紝涓嶅啀瀵瑰簲鐗╃悊鏂囦欢銆?    """
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
        # 鍏抽敭鍙樺寲锛氱洿鎺ュ紩鐢ㄥ熀纭€瀛楁鐨勫師濮嬭〃鍚嶏紝瀹炵幇琛ㄧ粨鏋勪笌鏁版嵁婧愬鐢?        "raw_table": TMPL.T_RAW.format(idx=base_idx),
        "std_view": TMPL.V_STD.format(idx=idx),
        "stx_view": TMPL.V_STX.format(idx=idx),
        "aln_view": TMPL.T_ALN.format(idx=idx),
        "sync_mode": "VIRTUAL",  # 鏍囪涓鸿櫄鎷熷悓姝ワ紝DB 瀵煎叆闃舵灏嗚烦杩囩墿鐞嗘枃浠舵鏌?        "provider": "internal_view",
        "fields": fields,  # 绉嶅瓙瑙嗗浘蹇呴』涓庡熀纭€瀛楁琛ㄧ殑鍒楁槧灏勪繚鎸佷竴鑷?        "pre_filters": pre_filters,
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
    """鐢熸垚澶栭儴鍙傝€冩槦琛?(Membership Catalog) 鏍囧噯閰嶇疆椤圭殑杈呭姪鍑芥暟"""
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

# 6.3 鏍稿績鏁版嵁娓呭崟
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
        # 绋嬪簭涓病鏈夎繘琛屼簩娆℃竻鐞?stx涓巗td淇濇寔涓€鑷?        "stx_view": TMPL.V_STX.format(idx=IDX_DR2IDX),
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
            "rv": "RV",
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
# 7. 绠楁硶娴佹按绾块厤缃?(Pipeline & GMM Configuration)
# =================================================================
GMM_CONFIG = {
    "FULL_NAME": "SeedGMM",
    "SHORT_NAME": "SG",
    "feature_map": {
        "2d": ["pmra", "pmdec"],
        "3d": ["pmra", "pmdec", "plx"],
        "5d": ["ra", "dec", "pmra", "pmdec", "plx"],
        "6d_o": ["ra", "dec", "pmra", "pmdec", "plx", "rv"],
        # --- Hunt 2024 鏂囩尞娣峰悎绌洪棿妯″紡 ---
        "5d_h": ["l", "b", "pm_l_cosb", "pm_b", "plx"],  # 閾堕亾 5D 鐩叉悳绌洪棿
        # --- 鐗╃悊鐩磋绌洪棿妯″紡 (Physical) ---
        "3d_v": ["U", "V", "W"],  # 鐗╃悊閫熷害绌洪棿锛堢函鍔ㄥ姏瀛︼級
        "6d_p": ["X", "Y", "Z", "U", "V", "W"],  # 瀹屾暣鐗╃悊鐩哥┖闂?    },
    "dim_mode": "3d",
    "ruwe_limit": 1.4,
    "cluster_algo": "dbscan", # 鍙€? dbscan, hdbscan
    "dbscan_eps": 0.3,  # 浠?0.3 璋冨ぇ锛岃ˉ鍋块珮缁寸┖闂磋窛绂?    "dbscan_min_samples": 100,  # 3d妯″瀷涓?00
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 5,          # 鎻愰珮闂ㄩ檺锛屼娇绉嶅瓙鏍稿績鏇村嚌鑱氾紝鍑忓皯鏉傝川
    "hdbscan_cluster_selection_epsilon": 0.1, # 杩涗竴姝ラ檷浣庡悎骞跺蹇嶅害锛屽彧淇濈暀鏈€楂樺瘑搴︾殑鏍稿績閮ㄥ垎
    "gmm_covariance_type": "full",
    "max_iter": 20,
    "tol": 1e-5,
    "use_experimental": True,  # 鍚敤瀹為獙鎬у姛鑳斤紝濡傚熀浜庤繎閭荤殑鏅鸿兘鍒濆鍖?    "enable_subsampling": False,  # 鏄惁鍚敤鑳屾櫙涓嬮噰鏍蜂紭鍖栵紝浠ュ姞閫熸ā鍨嬫嫙鍚?    "subsampling_limit": 500000, # 涓嬮噰鏍疯Е鍙戦棬闄愬強鐩爣鏍锋湰閲?}
