import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# 修正导入路径
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.db import AstroDB
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AstroUtils.RadiusCalculator")

def compute_hunt_metrics(db, cluster_id):
    """
    根据 Hunt 2024 的成员分布计算统计特征（半径与星等）并给出建议值。
    """
    cluster_cfg = cfg.CLUSTERS.get(cluster_id)
    if not cluster_cfg:
        logger.error(f"❌ config.py 中未找到星团 {cluster_id}")
        return None

    # 获取 Hunt24 的标准化视图名
    hunt_cfg = cfg.MANIFEST[cfg.IDX_HUNT]
    t_hunt_raw = hunt_cfg.raw_table
    fields = hunt_cfg.fields

    # 提取物理表中的原始列名
    raw_ra = fields.get("ra") if hasattr(fields, "get") else fields.ra
    raw_dec = fields.get("dec") if hasattr(fields, "get") else fields.dec
    raw_mag = fields.get("mag") if hasattr(fields, "get") else fields.mag
    raw_cluster = fields.get("cluster") if hasattr(fields, "get") else fields.cluster
    cat_name = cluster_cfg.CAT_NAME

    # 1. 直接从原始物理表提取样本，确保没有经过 std 视图的预过滤截断
    query = f"SELECT \"{raw_ra}\" as ra, \"{raw_dec}\" as dec, \"{raw_mag}\" as mag FROM {t_hunt_raw} WHERE \"{raw_cluster}\" = '{cat_name}'"
    df_members = db.query(query)

    if df_members.empty:
        logger.warning(f"⚠️ 在 {t_hunt_raw} 中未找到星团 {cat_name} 的成员，请确保数据已导入。")
        return None

    # 2. 计算角距离
    center_coord = SkyCoord(ra=cluster_cfg.CENTER_RA*u.deg, dec=cluster_cfg.CENTER_DEC*u.deg)
    member_coords = SkyCoord(ra=df_members["ra"].values*u.deg, dec=df_members["dec"].values*u.deg)
    
    # 得到所有成员到中心的距离
    distances = center_coord.separation(member_coords).degree
    
    r_max = np.max(distances)
    r_mean = np.mean(distances)
    r_95 = np.percentile(distances, 95) # 95% 的成员覆盖半径
    
    # 3. 计算星等分布 (最暗的星)
    mag_max = np.max(df_members["mag"].dropna().values)

    # 4. 给出建议值 (带 Buffer)
    suggested = max(r_max * 1.2, r_max + 0.5)
    suggested_mag = np.ceil(mag_max + 0.4) # 向上取整，并留出约 0.5 的余量

    logger.info(f"📊 [星团: {cluster_id} / {cluster_cfg.NAME}]")
    logger.info(f"   ∟ 已知成员数: {len(df_members)}")
    logger.info(f"   ∟ [半径统计] 最大: {r_max:.3f}° | 95%: {r_95:.3f}° | 均值: {r_mean:.3f}°")
    logger.info(f"   ∟ [星等统计] 最暗: {mag_max:.3f}")
    logger.info(f"   ∟ [对比] 半径 (Cur: {cluster_cfg.RADIUS:.2f} -> Sug: {suggested:.2f})")
    logger.info(f"   ∟ [对比] 星等 (Cur: {getattr(cluster_cfg, 'MAX_MAG', 'N/A')} -> Sug: {suggested_mag:.1f})")

    return {
        "cluster": cluster_id,
        "r_max": r_max,
        "suggested_r": suggested,
        "current_r": cluster_cfg.RADIUS,
        "mag_max": mag_max,
        "suggested_mag": suggested_mag,
        "current_mag": getattr(cluster_cfg, "MAX_MAG", 0)
    }

if __name__ == "__main__":
    db = AstroDB(manifest=cfg.MANIFEST)
    
    # 确保 Hunt24 数据已载入 DuckDB
    # 检查原始物理表是否存在
    if not db.table_exists(cfg.MANIFEST[cfg.IDX_HUNT].raw_table):
        logger.info("📡 正在导入参考星表数据...")
        db.import_raw()

    results = []
    logger.info("🚀 开始扫描 Hunt24 文献以校准天区半径...")
    for cid in cfg.CLUSTERS.keys():
        res = compute_hunt_metrics(db, cid)
        if res:
            results.append(res)

    # 汇总输出
    print("\n" + "="*80)
    header = f"{'Cluster':<8} | {'Cur_R':<6} | {'Act_R':<6} | {'Sug_R':<6} | {'Cur_M':<6} | {'Act_M':<6} | {'Sug_M':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['cluster']:<8} | {r['current_r']:<6.2f} | {r['r_max']:<6.2f} | {r['suggested_r']:<6.2f} | {r['current_mag']:<6.1f} | {r['mag_max']:<6.1f} | {r['suggested_mag']:<6.1f}"
        print(row)
    print("="*80)
