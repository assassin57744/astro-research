from pathlib import Path

# =============================================================================
# 1. 基础路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# 数据下载目录 (Gaia/Vizier 原始文件)
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"
# 内部状态目录 (保存下载进度、Job ID 等)
INTERNAL_DIR = PROJECT_ROOT / "data" / "internal"


# =============================================================================
# 2. 数据源凭据 (建议从环境变量读取以保证安全)
# =============================================================================
GAIA_USER = "your_gaia_username"  # 替换为你的 Gaia Archive 用户名
GAIA_PWD = "your_gaia_password"   # 替换为你的 Gaia Archive 密码


# =============================================================================
# 3. 字段映射配置 (用于统一不同引擎的查询列)
# =============================================================================
# Gaia Archive ADQL 字段映射
FIELDS_GAIA_ARCHIVE = {
    "id": "source_id",
    "ra": "ra",
    "dec": "dec",
    "mag": "phot_g_mean_mag",
    "pmra": "pmra",
    "pmdec": "pmdec",
    "parallax": "parallax",
    "bp_rp": "bp_rp"
}

# VizieR (I/355/gaiadr3) 字段映射
FIELDS_VIZIER = {
    "id": "Source",
    "ra": "RA_ICRS",
    "dec": "DE_ICRS",
    "mag": "Gmag",
    "pmra": "pmRA",
    "pmdec": "pmDE",
    "parallax": "Plx",
    "bp_rp": "BP-RP"
}


# =============================================================================
# 4. 星表清单与 Manifest 配置
# =============================================================================
# Manifest 索引定义
IDX_GAIA = 0
IDX_HUNT = 1

MANIFEST = {
    IDX_GAIA: {
        "name": "Gaia DR3 Source",
        "params": {
            "file_pattern": "{cluster_id}_gaia_dr3.parquet"
        }
    },
    IDX_HUNT: {
        "name": "Hunt 2024 Reference",
        "raw_table": "hunt_2024_raw",
        "fields": {
            "ra": "ra",
            "dec": "dec",
            "mag": "g_mag",
            "cluster": "cluster_name"
        }
    }
}

# =============================================================================
# 5. 科学目标配置 (星团列表)
# =============================================================================
CLUSTERS = {
    "M45": {
        "NAME": "Pleiades",
        "CAT_NAME": "Pleiades",      # 对应文献或 Simbad 中的名字
        "CENTER_RA": 56.75,
        "CENTER_DEC": 24.1167,
        "RADIUS": 3.0,              # 下载半径 (度)
        "MAX_MAG": 19.0,            # 视星等上限
        "FIELD_IDX": IDX_GAIA       # 指定对应的下载规则
    },
    "M44": {
        "NAME": "Praesepe",
        "CAT_NAME": "NGC_2632",
        "CENTER_RA": 130.1,
        "CENTER_DEC": 19.66,
        "RADIUS": 2.5,
        "MAX_MAG": 18.5,
        "FIELD_IDX": IDX_GAIA
    }
}