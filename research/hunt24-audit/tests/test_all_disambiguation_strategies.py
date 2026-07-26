# -*- coding: utf-8 -*-
"""
test_all_disambiguation_strategies.py

🎯 针对三个精筛策略（Bayesian, Threshold, Blind）的多元相空间集成测试脚本。
物理语义：验证不同科学假设下的精筛策略在数据结构、特征处理、NaN 拦截和接口契约上的一致性。
"""

import os
import sys
from pathlib import Path

# =================================================================
# 🛡️ 动态定位并注入项目根目录到 sys.path，支持跨目录无痛执行
# =================================================================
current_file = Path(__file__).resolve()
project_root = current_file.parent if (current_file.parent / "config.py").exists() else current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import numpy as np
import pandas as pd

# 显式导入三大重构完备的精筛策略算子
from modules.membership.disambiguation.bayesian import BayesianGmmDisambiguation
from modules.membership.disambiguation.threshold import ThresholdGmmDisambiguation
from modules.membership.disambiguation.blind import BlindGmmDisambiguation

# 配置测试专用控制台高可读日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AstroPipeline.AllStrategiesTest")


def generate_synthetic_gaia_field() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成高度仿真、包含 5D 相空间物理量纲的综合巡天测试数据
    特征：包含高凝聚星团核心、弥散银盘背景野星、以及带有残缺特征（NaN）的噪声星
    """
    np.random.seed(2026)  # 锁定数理随机基准
    n_field = 1500
    n_cluster = 100
    n_nan = 50
    
    logger.info(f"🔮 正在构建综合天区实体。背景野星: {n_field} | 核心星团星: {n_cluster} | 特征残缺噪声星: {n_nan}")
    
    # 1. 构造广袤弥散的银盘背景 (高方差自行，宽泛视差)
    df_field = pd.DataFrame({
        "id": np.arange(1, n_field + 1),
        "ra": np.random.uniform(56.0, 57.5, n_field),
        "dec": np.random.uniform(23.5, 24.5, n_field),
        "pmra": np.random.normal(0.0, 15.0, n_field),
        "pmdec": np.random.normal(-10.0, 15.0, n_field),
        "plx": np.random.uniform(1.0, 5.0, n_field),
    })
    
    # 2. 注入高密实的近邻星团核心 (极小方差凝聚，视差高度统一)
    cluster_start_id = n_field + 1
    df_cluster = pd.DataFrame({
        "id": np.arange(cluster_start_id, cluster_start_id + n_cluster),
        "ra": np.random.normal(56.75, 0.2, n_cluster),
        "dec": np.random.normal(24.0, 0.2, n_cluster),
        "pmra": np.random.normal(19.5, 0.3, n_cluster),
        "pmdec": np.random.normal(-44.8, 0.3, n_cluster),
        "plx": np.random.normal(7.34, 0.08, n_cluster),
    })
    
    # 3. 注入现实巡天中不可避免的特征残缺恶性样本 (包含 NaN)
    nan_start_id = cluster_start_id + n_cluster
    df_nan = pd.DataFrame({
        "id": np.arange(nan_start_id, nan_start_id + n_nan),
        "ra": np.random.uniform(56.0, 57.5, n_nan),
        "dec": np.random.uniform(23.5, 24.5, n_nan),
        "pmra": np.random.normal(0.0, 15.0, n_nan),
        "pmdec": [np.nan if i % 2 == 0 else 1.2 for i in range(n_nan)],  # 强行注入部分自行 NaN
        "plx": [np.nan if i % 3 == 0 else 3.4 for i in range(n_nan)],    # 强行注入部分视差 NaN
    })
    
    # 组合为全量靶场大表 df_all
    df_all = pd.concat([df_field, df_cluster, df_nan], ignore_index=True)
    
    # 模拟完美无瑕的前置种子筛选：精准拿到绝大部分高质量星团核心作为下游先验
    df_seeds = df_cluster.head(85).copy()
    df_seeds["cluster_label"] = 0
    
    return df_all, df_seeds


def run_pipeline_integration():
    """执行三大精筛算子全链路测试演练"""
    logger.info("🎬 开始执行精筛解超空间全策略集成测试流水线...\n" + "="*70)
    
    # 生成测试用物理底座
    df_all, df_seeds = generate_synthetic_gaia_field()
    
    # 统一拟合洗涤的特征相空间：3D（自行 + 视差）
    test_features = ["pmra", "pmdec", "plx"]
    
    # --------------------------------==================--------------------------------
    # 🧪 策略 1 测试：BayesianGmmDisambiguation (双模型极大似然对抗)
    # --------------------------------==================--------------------------------
    logger.info("\n🛰️ [策略 1 演练] 初始化 Bayesian 极大似然洗涤算子...")
    strategy_bayesian = BayesianGmmDisambiguation(
        cluster_algo="dbscan", dbscan_eps=0.3, dbscan_min_samples=5, tol=1e-6
    )
    res_bayesian = strategy_bayesian.fit_predict(df_all, df_seeds, test_features)
    
    # 严格契约断言检查
    assert len(res_bayesian) == len(df_all), "❌ Bayesian 策略输出总行数错位！"
    assert "prob" in res_bayesian.columns, "❌ Bayesian 策略缺少统一概率标度 'prob' 列！"
    # 科学合理性检查：由于 NaN 星未参与推演，其对应的概率必须被安全置 0
    nan_ids = df_all[df_all[test_features].isna().any(axis=1)]["id"]
    assert (res_bayesian.loc[res_bayesian["id"].isin(nan_ids), "prob"] == 0.0).all(), "❌ Bayesian 策略未成功拦截 NaN 样本！"
    logger.info(f"✅ [策略 1 成功] Bayesian 对抗收敛完备。判定高归属度恒星数: {len(res_bayesian[res_bayesian['prob'] > 0.5])} 颗。")

    # --------------------------------==================--------------------------------
    # 🧪 策略 2 测试：ThresholdGmmDisambiguation (卡方多元超椭球硬截断)
    # --------------------------------==================--------------------------------
    logger.info("\n📐 [策略 2 演练] 初始化 Threshold 卡方高维积分截断算子...")
    strategy_threshold = ThresholdGmmDisambiguation(sigma_cutoff=3.0)
    res_threshold = strategy_threshold.fit_predict(df_all, df_seeds, test_features)
    
    # 严格契约断言检查
    assert len(res_threshold) == len(df_all), "❌ Threshold 策略输出总行数错位！"
    assert "is_member" in res_threshold.columns, "❌ Threshold 策略缺少硬打标 'is_member' 列！"
    assert (res_threshold.loc[res_threshold["id"].isin(nan_ids), "prob"] == 0.0).all(), "❌ Threshold 策略未成功拦截 NaN 样本！"
    logger.info(f"✅ [策略 2 成功] Threshold 卡方解算完备。通过椭球边界硬洗涤出的数组成员: {res_threshold['is_member'].sum()} 颗。")

    # --------------------------------==================--------------------------------
    # 🧪 策略 3 测试：BlindGmmDisambiguation (空间 ROI 自适应无先验盲搜)
    # --------------------------------==================--------------------------------
    logger.info("\n🦅 [策略 3 演练] 初始化 Blind 空间自适应全域多成分盲搜算子...")
    strategy_blind = BlindGmmDisambiguation(
        n_components=3, covariance_type="full", spatial_cols=["ra", "dec"], scale_col="plx"
    )
    res_blind = strategy_blind.fit_predict(df_all, df_seeds, test_features)
    
    # 严格契约断言检查
    assert len(res_blind) == len(df_all), "❌ Blind 策略输出总行数错位！"
    assert "is_member" in res_blind.columns, "❌ Blind 策略缺少硬打标 'is_member' 列！"
    assert (res_blind.loc[res_blind["id"].isin(nan_ids), "prob"] == 0.0).all(), "❌ Blind 策略未成功拦截 NaN 样本并置零！"
    logger.info(f"✅ [策略 3 成功] Blind 盲搜内核完备。自适应剥离高噪声银盘后析出的潜在数组成员: {res_blind['is_member'].sum()} 颗。")

    # --------------------------------==================--------------------------------
    # 🎉 终极总结
    # --------------------------------==================--------------------------------
    logger.info("\n" + "="*70 + "\n🏆 🎉 恭喜！第一阶段重构的三大精筛核心策略算法 100% 通过集成测试全栈大闭环！\n" + "="*70)


if __name__ == "__main__":
    run_pipeline_integration()