"""
🎯 星团自适应超参数解算器（Castro-Ginard 论文算法纯净版）。

核心定位：
  - 单一职责：专注于根据输入的背景视场大盘（field_stars_df），利用 KDE 重采样解算 DBSCAN 的自适应最优领域半径 eps。
  - 功能不纯剔除：移除了内部执行聚类、动态路由算法（DBSCAN/HDBSCAN）等流水线编排逻辑。
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity


class CastroGinardEpsEstimator:
    """基于 Castro-Ginard et al. (2018) 算法的自适应最优 EPS 解算引擎。"""

    def __init__(self, min_pts: int = 9, num_simulations: int = 30):
        """
        参数:
        - min_pts (int): 构成密集核心星团所需的最小恒星点数（minPts）。
        - num_simulations (int): KDE 随机背景重采样的迭代次数。
        """
        self.logger = logging.getLogger("AstroPipeline.CastroGinardEpsEstimator")
        self.min_pts = int(min_pts)
        self.num_simulations = int(num_simulations)

        # 实际参与密度感知的物理相空间特征维度
        self.features = ["ra", "dec", "plx", "pmra", "pmdec"]

    def calculate_adaptive_eps(self, df_seeds_field: pd.DataFrame) -> float:
        """
        🧬 动态感知局部天区背景密度，反演自适应最优领域半径 EPS。

        还原论文思路：
          - Step A: 解算真实星表相空间的 (minPts - 1) 阶最近邻距离（k-NND）分布的极小值，感知高密度突变包络。
          - Step B: 通过高斯核密度估计（KDE）重构无结构的平滑噪声统计底盘，计算背景随机聚集期望值。
          - Step C: 算术平均调和两极，在“提高捕获率”与“防御野星污染”之间达成最优物理平衡。
        
        输入参数:
          - df_seeds_field (pd.DataFrame): 目标星团及周边局部视场的恒星原始大盘。
        
        返回:
          - float: 自适应计算得出的物理最优领域半径 eps。如果样本不足则返回 -1.0。
        """
        # 剔除无效观测值，确保相空间矩阵完整
        clean_df = df_seeds_field[self.features].dropna()
        n_samples = len(clean_df)

        if n_samples <= self.min_pts:
            self.logger.warning(
                f"⚠️ 当前视场天区恒星样本量 ({n_samples}) 少于 minPts ({self.min_pts})，无法执行自适应调参！"
            )
            return -1.0

        self.logger.info(
            f"📡 [密度感知] 启动本征噪声线动态解算 | 样本量: {n_samples} | minPts: {self.min_pts}"
        )

        # 1. 特征矩阵标准化（消除度、mas/yr、mas 之间的跨数量级标度残差，使欧氏距离具备物理意义）
        X = clean_df.values
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std = np.where(X_std == 0, 1.0, X_std)  # 防御分母为 0
        X_scaled = (X - X_mean) / X_std

        # 论文设定：k 阶最近邻的 k = minPts - 1
        k = self.min_pts - 1

        # 🚀 步骤 A: 计算真实星表的 k-NND 突变谷值
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        eps_knn = float(np.min(distances[:, k]))
        self.logger.debug(f"📊 [Step A] 真实相空间最近邻特征谷值 eps_knn = {eps_knn:.6f}")

        # 🚀 步骤 B: 通过高斯 KDE 随机重采样构建平滑统计大盘
        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(X_scaled)
        rand_min_distances = []

        for i in range(self.num_simulations):
            # 模拟生成一个完全无物理星团、只有本征随机涨落的平滑伪星表
            X_rand = kde.sample(n_samples=n_samples, random_state=i)
            nbrs_rand = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_rand)
            dist_rand, _ = nbrs_rand.kneighbors(X_rand)
            rand_min_distances.append(np.min(dist_rand[:, k]))

        # 对多次平滑背景期望取算术平均
        eps_rand = float(np.mean(rand_min_distances))
        self.logger.debug(f"🎲 [Step B] {self.num_simulations} 次高斯底盘无结构随机期望 eps_rand = {eps_rand:.6f}")

        # 🚀 步骤 C: 算术平均调和
        optimal_eps = (eps_knn + eps_rand) / 2.0
        self.logger.info(f"🎯 [Step C] Castro-Ginard 调和完成！获得该天区自适应最优超参数: EPS = {optimal_eps:.4f}")

        return optimal_eps