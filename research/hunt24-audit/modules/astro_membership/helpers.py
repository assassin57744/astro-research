# -*- coding: utf-8 -*-
"""
modules/astro_membership/helpers.py

🧠 天体测量学与统计学底层核心辅助算子。
定位：astro_membership 子包内专用的数理内核库，与上游业务流完全解耦。
"""

import logging
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors

logger = logging.getLogger("AstroPipeline.AstroMembership.Helpers")


def calculate_adaptive_eps_kde(
    X_raw: np.ndarray, 
    min_pts: int = 9, 
    num_simulations: int = 30
) -> float:
    """
    基于 Castro-Ginard et al. (2018) 论文逻辑，通过全量背景天区的 KDE 蒙特卡洛重采样，
    逆向解算出能安全消除银河系随机统计涨落的最优物理邻域半径 EPS。

    Args:
        X_raw (np.ndarray): 原始特征矩阵，形状为 (n_samples, n_features)。
                            可以是天球相空间坐标，也可以是直角坐标/速度。
        min_pts (int): 构成密集核心所需的最小恒星点数（DBSCAN 的 min_samples）。默认 9。
        num_simulations (int): KDE 随机背景重采样模拟的次数。默认 30。

    Returns:
        float: 自动定标出的物理边界半径 EPS。
    """
    n_samples, n_features = X_raw.shape
    if n_samples <= min_pts:
        logger.warning(f"样本数 ({n_samples}) 小于等于 min_pts ({min_pts})，自适应 KDE 无法定标，回退至经验值 0.3")
        return 0.3

    # 1. 内部标度标准化（Standardization）
    # 确保不同量纲（如角度、视差、自行）在数学空间中具有平等的权重
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)  # 🛡️ 防止方差为 0 导致除零溃缩
    X_scaled = (X_raw - X_mean) / X_std

    # 2. 拟合本征背景场的非参数高斯核密度估计（KDE）模型
    # 采用 Scott's rule 动态确定高维空间平滑带宽 (Bandwidth)
    bandwidth = n_samples ** (-1.0 / (n_features + 4))
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian", metric="euclidean")
    kde.fit(X_scaled)

    # 3. 蒙特卡洛重采样模拟：探查完全无物理凝聚的随机背景场在统计学上的“密度最高峰”
    sim_eps_list = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*delayed.*should be used with.*Parallel.*")
        for i in range(num_simulations):
            X_sim = kde.sample(n_samples=n_samples, random_state=42 + i)
            nbrs = NearestNeighbors(n_neighbors=min_pts, n_jobs=-1).fit(X_sim)
            distances, _ = nbrs.kneighbors(X_sim)
            sim_eps_list.append(np.min(distances[:, min_pts - 1]))

    # 4. 科学沉淀：取多次模拟上限的均值，作为斩断一切随机噪声、筛选真实星团实体的 EPS 屏障
    optimal_eps = float(np.mean(sim_eps_list))
    
    # 🛡️ 极端边界防御
    if optimal_eps <= 0 or np.isnan(optimal_eps):
        logger.warning("自适应解算出的 EPS 异常，回退至默认物理经验值 0.3")
        return 0.3
        
    return optimal_eps


def estimate_adaptive_dbscan_eps_knee(X_raw: np.ndarray, k: int = 10) -> float:
    """
    通过 k-距离图（k-distance graph）的几何最大弯曲点（拐点检测），
    快速估计 DBSCAN 的自适应 eps 邻域半径。
    相比 KDE 重采样，该方法不进行模拟，计算速度极快，适合超大样本。

    Args:
        X_raw (np.ndarray): 原始特征矩阵 (n_samples, n_features)。
        k (int): 邻域期望星数，通常与 DBSCAN 的 min_samples 挂钩。默认 10。

    Returns:
        float: 基于最大曲率弯曲点解算出的特征空间距离 EPS。
    """
    n_samples, n_features = X_raw.shape
    if n_samples <= k:
        return 0.3
        
    # 1. 内部标度标准化
    X_mean = X_raw.mean(axis=0)
    X_std = np.where(X_raw.std(axis=0) == 0, 1.0, X_raw.std(axis=0))
    X_scaled = (X_raw - X_mean) / X_std
    
    # 2. 计算每个点到其第 k 个近邻的欧氏距离
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    
    # 3. 对所有近邻距离进行升序排序
    k_distances = np.sort(distances[:, k - 1])
    n_points = len(k_distances)
    
    # 4. 几何弯曲最大化法（Knee Detection）
    # 连接排序曲线的起点和终点构成基准直线，寻找曲线上正交投影距离该直线最远的点
    x_coords = np.arange(n_points)
    start_point = np.array([0, k_distances[0]])
    end_point = np.array([n_points - 1, k_distances[-1]])
    
    line_vec = end_point - start_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    # 向量化计算所有样本点到基准直线的正交几何距离
    vecs = np.column_stack((x_coords, k_distances)) - start_point
    proj_lengths = np.dot(vecs, line_vec_norm)
    proj_vecs = np.outer(proj_lengths, line_vec_norm)
    dist_to_line = np.linalg.norm(vecs - proj_vecs, axis=1)
    
    # 5. 抓取弯曲拐点对应的 k-distance 作为自适应门限
    knee_idx = np.argmax(dist_to_line)
    adaptive_eps = float(k_distances[knee_idx])
    
    if adaptive_eps <= 0 or np.isnan(adaptive_eps):
        return 0.3
        
    return adaptive_eps