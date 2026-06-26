# modules/cluster_seed_extractor.py
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN


class ClusterSeedExtractor:
    """
    🎯 星团种子提取器（无监督粗筛引擎）。

    定位：Pipeline 的前置粗筛 Stage。
    物理语义：
      1. 基于 Castro-Ginard et al. (2018) 论文逻辑，自适应感知天区本征恒星背景密度，动态解算最佳领域半径 EPS。
      2. 运行精巧的几何空间硬截断（DBSCAN），大范围剔除银河系野星噪声，提取高纯度的成员星种子（Seeds），
         为下游高阶概率精筛模型（如 PriorGMMEx）构建可靠的初始状态底座。
    """

    def __init__(self, min_pts: int = 9, num_simulations: int = 30):
        """
        参数:
        - min_pts (int): 构成密集核心星团所需的最小恒星点数（minPts）。论文推荐 5 ~ 9 颗星。
        - num_simulations (int): KDE 随机背景重采样的迭代次数，用于完全消除天区内随机统计涨落带来的误差。
        """
        self.logger = logging.getLogger("AstroPipeline.ClusterSeedExtractor")
        self.min_pts = min_pts
        self.num_simulations = num_simulations

        # 实际参与密度感知的物理相空间特征维度
        self.features = ["ra", "dec", "plx", "pmra", "pmdec"]

    def calculate_adaptive_eps(self, field_stars_df: pd.DataFrame) -> float:
        """
        🧬 [核心物理算法] 动态感知局部天区背景密度，反演自适应的最佳领域半径 Epsilon (EPS)。

        还原论文思路：
          - Step A: 解算真实星表相空间的 (minPts - 1) 阶最近邻距离（k-NND）分布的极小值，感知高密度突变包络。
          - Step B: 通过高斯核密度估计（KDE）重构无结构的平滑噪声统计底盘，计算背景随机聚集期望值。
          - Step C: 算术平均调和两极，在“提高捕获率”与“防御野星污染”之间达成最优物理平衡。
        """
        # 剔除无效观测值，确保相空间矩阵完整
        clean_df = field_stars_df[self.features].dropna()
        n_samples = len(clean_df)

        if n_samples <= self.min_pts:
            self.logger.warning(
                f"⚠️ 当前视场天区恒星样本量 ({n_samples}) 少于 minPts ({self.min_pts})，无法执行自适应调参！"
            )
            return -1.0

        self.logger.info(
            f"📡 [密度感知] 启动本征噪声线动态解算 | 样本量: {n_samples} | minPts: {self.min_pts}"
        )

        # 1. 特征矩阵标准化（核心工程防御：消除度、mas/yr、mas 之间的跨数量级标度残差，使欧氏距离具备物理意义）
        X = clean_df.values
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        # 防止分母为 0 异常
        X_std = np.where(X_std == 0, 1.0, X_std)
        X_scaled = (X - X_mean) / X_std

        # 论文设定：k 阶最近邻的 k = minPts - 1
        k = self.min_pts - 1

        # 🚀 步骤 A: 计算真实星表的 k-NND 突变谷值
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        # 提取真实样本到其第 k 个最近邻居的特征距离
        eps_knn = float(np.min(distances[:, k]))
        self.logger.debug(
            f"📊 [Step A] 真实相空间最近邻特征谷值 eps_knn = {eps_knn:.6f}"
        )

        # 🚀 步骤 B: 通过高斯 KDE 随机重采样构建平滑统计大盘
        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(X_scaled)
        rand_min_distances = []

        for i in range(self.num_simulations):
            # 模拟生成一个完全无物理星团、只有本征随机涨落的平滑伪星表
            X_rand = kde.sample(n_samples=n_samples, random_state=i)
            nbrs_rand = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(
                X_rand
            )
            dist_rand, _ = nbrs_rand.kneighbors(X_rand)
            rand_min_distances.append(np.min(dist_rand[:, k]))

        # 对 30 次平滑背景期望取算术平均
        eps_rand = float(np.mean(rand_min_distances))
        self.logger.debug(
            f"🎲 [Step B] {self.num_simulations} 次高斯底盘无结构随机期望 eps_rand = {eps_rand:.6f}"
        )

        # 🚀 步骤 C: 算术平均调和
        optimal_eps = (eps_knn + eps_rand) / 2.0
        self.logger.info(
            f"🎯 [Step C] Castro-Ginard 调和完成！获得该天区自适应最优超参数: EPS = {optimal_eps:.4f}"
        )

        return optimal_eps

    def extract_seeds(self, field_stars_df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """
        🏃‍♂️ 执行无监督密度切割，剔除噪声，提取种子星。

        参数:
        - field_stars_df: 包含局部视场恒星全大盘的原始 DataFrame。
        - eps: 由 calculate_adaptive_eps 动态反演得到的自适应半径。

        返回:
        - pd.DataFrame: 提取出的高纯度种子星副本（df_seeds），附带 'seed_label' 列。
        """
        if eps <= 0:
            self.logger.error("❌ 输入的 EPS 参数非法，拒绝提取种子，返回空表。")
            return pd.DataFrame(columns=field_stars_df.columns)

        # 浅拷贝防止破坏外层原始星表
        working_df = field_stars_df.copy()

        # 保持与自适应解算时 100% 一致的标准化特征标度
        clean_mask = working_df[self.features].notna().all(axis=1)
        X_raw = working_df.loc[clean_mask, self.features].values

        if len(X_raw) == 0:
            self.logger.warning("⚠️ 没有有效恒星样本满足相空间特征完整性，无法提取。")
            return pd.DataFrame(columns=field_stars_df.columns)

        X_scaled = (X_raw - X_raw.mean(axis=0)) / np.where(
            X_raw.std(axis=0) == 0, 1.0, X_raw.std(axis=0)
        )

        # 运行 DBSCAN 进行硬性几何相空间密度识别
        self.logger.info(
            f"🗜️ 正在运行无监督物理截断扫描 (EPS={eps:.4f}, minPts={self.min_pts})..."
        )
        db = DBSCAN(eps=eps, min_samples=self.min_pts, n_jobs=-1).fit(X_scaled)

        # 标签回填（不满足 clean_mask 的默认填充为 -1 噪声）
        labels = np.full(len(working_df), -1, dtype=int)
        labels[clean_mask] = db.labels_
        working_df["seed_label"] = labels

        # 过滤提取：丢弃 -1 (噪声野星)，只保留密集核心集群（labels >= 0）
        df_seeds = working_df[working_df["seed_label"] >= 0].copy()

        # 物理资产产量审计
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        self.logger.info(
            f"🎉 粗筛扫描结束！共捕获高密度核心聚集群 x {n_clusters}，成功分离提取出 {len(df_seeds)} 颗高纯度种子星。"
        )

        return df_seeds
