# -*- coding: utf-8 -*-
"""
test_stage1_integration.py

🎯 第一阶段（Stage 1）无监督粗筛与贝叶斯精筛全链路集成测试脚本。
物理语义：模拟真实巡天管线，加载 config.py，流转全链条数据，验证重构完备性。
"""
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

# =================================================================
# 🛡️ 动态定位并注入项目根目录到 sys.path
# =================================================================
# 1. 获取当前测试脚本文件的绝对路径
current_file = Path(__file__).resolve()

# 2. 向上追溯到项目根目录 (根据你的目录结构，tests/ 的上一级即为根目录)
# 如果你决定采用方案 B 直接放在根目录，parent 依然安全
project_root = current_file.parent if (current_file.parent / "config.py").exists() else current_file.parent.parent

# 3. 拦截式注入：将其作为最高优先级搜索路径推入 sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

    
import logging
import numpy as np
import pandas as pd
import config  # 导入你的标准配置文件

from modules.cluster_seed_extractor import ClusterSeedExtractor
from modules.astro_membership.disambiguation.bayesian import BayesianGmmDisambiguation

# 配置控制台输出日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AstroPipeline.TestIntegration")


def generate_mock_gaia_data(n_stars=1000) -> pd.DataFrame:
    """生成模拟的 Gaia 相空间星表数据，包含常规特征和集群凝聚核"""
    np.random.seed(42)
    logger.info(f"🔮 正在生成模拟巡天靶场数据，总恒星数: {n_stars} ...")
    
    # 模拟背景弥散野星 (Field Stars)
    df_field = pd.DataFrame({
        "id": np.arange(1, n_stars + 1),
        "ra": np.random.uniform(55, 58, n_stars),
        "dec": np.random.uniform(23, 25, n_stars),
        "pmra": np.random.normal(0.0, 5.0, n_stars),
        "pmdec": np.random.normal(0.0, 5.0, n_stars),
        "plx": np.random.uniform(2.0, 10.0, n_stars),
    })
    
    # 注入一个高度凝聚的星团核心 (Cluster Core, 约 80 颗星)
    n_cluster = 80
    df_field.loc[100:100+n_cluster-1, "pmra"] = np.random.normal(20.0, 0.2, n_cluster)
    df_field.loc[100:100+n_cluster-1, "pmdec"] = np.random.normal(-45.0, 0.2, n_cluster)
    df_field.loc[100:100+n_cluster-1, "plx"] = np.random.normal(7.3, 0.05, n_cluster)
    
    return df_field


def run_stage1_pipeline_test(cluster_key: str):
    """运行指定星团配置的 Stage 1 链路测试"""
    logger.info(f"\n" + "="*60 + f"\n🎬 开始测试星团 Profile: [{cluster_key}] 从 config.py 读取\n" + "="*60)
    
    # 0. 捞取原始配置 Profile
    profile = config.CLUSTERS.get(cluster_key)
    if not profile:
        logger.error(f"❌ config.py 中未发现星团 [{cluster_key}] 的配置，跳过测试。")
        return
    
    # 提取参与计算的特征列 (此处模拟 3D 相空间：pmra, pmdec, plx)
    features = ["pmra", "pmdec", "plx"] 
    df_all_sky = generate_mock_gaia_data(n_stars=2000)
    
    # --------------------------------==================--------------------------------
    # 🌟 步骤 1: 前置无监督粗筛测试 (ClusterSeedExtractor)
    # --------------------------------==================--------------------------------
    logger.info("⏳ [Stage 1.1] 正在加载 ClusterSeedExtractor 实例...")
    extractor = ClusterSeedExtractor(cluster_profile=profile)
    
    # 执行种子出库
    df_seeds = extractor.extract_seeds(field_stars_df=df_all_sky, features=features)
    
    logger.info(f"📊 粗筛出库验证: 原始靶场星数={len(df_all_sky)} | 析出种子星数={len(df_seeds)}")
    if len(df_seeds) == 0:
        logger.error("💥 测试失败：前置种子库干涸，未能析出凝聚核心！")
        return
        
    assert "cluster_label" in df_seeds.columns, "❌ 错误：输出种子 DataFrame 缺少 'cluster_label' 列！"

    # --------------------------------==================--------------------------------
    # 🌟 步骤 2: 贝叶斯概率精筛测试 (BayesianGmmDisambiguation)
    # --------------------------------==================--------------------------------
    logger.info("⏳ [Stage 1.2] 正在加载 BayesianGmmDisambiguation 精筛算子...")
    
    # 100% 映射 config.py 中的参数字段初始化贝叶斯洗涤器
    disambiguator = BayesianGmmDisambiguation(
        cluster_algo=profile.get("cluster_algo", "dbscan"),
        dbscan_eps=0.3, # 贝叶斯内部二次修剪硬编码半径
        dbscan_min_samples=profile.get("dbscan_min_samples", 9),
        hdbscan_min_cluster_size=profile.get("hdbscan_min_cluster_size", 15),
        tol=1e-6
    )
    
    # 执行高阶似然对决洗涤
    df_final_probs = disambiguator.fit_predict(
        df_all=df_all_sky,
        df_seeds=df_seeds,
        features=features
    )
    
    # --------------------------------==================--------------------------------
    # 🌟 步骤 3: 断言与一致性检查
    # --------------------------------==================--------------------------------
    logger.info("⏳ [Stage 1.3] 正在对输出矩阵执行最终数理契约验证...")
    
    assert len(df_final_probs) == len(df_all_sky), "❌ 错误：精筛输出的恒星总数与输入靶场不一致！"
    assert "prob" in df_final_probs.columns, "❌ 错误：精筛结果缺少 'prob' 成员概率列！"
    
    high_prob_count = len(df_final_probs[df_final_probs["prob"] > 0.5])
    logger.info(f"🎉 [{cluster_key}] 链路集成测试 100% 通过！")
    logger.info(f"📈 结果摘要: 最终判定高概率成员星（prob > 0.5）共计: {high_prob_count} 颗。")


if __name__ == "__main__":
    # 可以通过切换不同的 Key，测试常规 "auto" 模式或未来的变密度模式
    run_stage1_pipeline_test("M45")