import logging
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity

import config as cfg
from config import MEMBER_SAMPLE_THRESHOLD, STD_COLS
from utils.decorators import astro_checkpoint



@dataclass
class GMMModelParamsEx:
    """
    模型参数容器（封测版）：保存从 PriorGMMEx 训练阶段提取的所有物理先验知识。

    Attributes:
        cluster_model (GaussianMixture): 拟合好的星团成员高斯分布模型。
        field_model (GaussianMixture): 拟合好的背景场高斯分布模型。
        scaler (StandardScaler): 特征标准化处理器，用于保持推理时的一致性。
        n_core_samples (int): 参与星团核心模型训练的有效样本数。
        dim_mode (str): 特征空间维度模式（如 '3d', '6d_p'）。
        features_used (list): 实际参与计算的特征列名。
        df_seeds_classified (pd.DataFrame): 包含分类标签(Core/Noise)的种子星副本。
        center_coords (dict): 用于中心化修正的各维度锚点值。
    """

    cluster_model: GaussianMixture
    field_model: GaussianMixture
    scaler: StandardScaler
    n_core_samples: int
    dim_mode: str
    features_used: list
    df_seeds_classified: pd.DataFrame = None
    center_coords: dict = field(default_factory=dict)


class PriorGMM:
    """
    种子星引导的高维先验迭代成员判定算法（PriorGMM 实验性增强版）。

    该算法采用无状态设计，支持从 3D 到 6D 的全相空间特征转换，
    通过高质种子星（Seeds）提取星团核心分布先验，并在全域样本中通过递归极大似然估计实现成员概率收敛。
    """

    def __init__(self, config=None):
        """初始化测试内核。"""
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}.Experimental")
        self.config = config or {}

        # 确定特征维度模式，无缝对接升级后的特征工程
        self.dim_mode = self.config.get("dim_mode", "3d").lower()

        # 动态映射 6 大特征空间对应的特征列名，严格匹配大一统规范（短键名 plx 与 rv）
        feature_map = {
            "2d": ["pmra", "pmdec"],
            "3d": ["pmra", "pmdec", "plx"],
            "5d": ["ra", "dec", "pmra", "pmdec", "plx"],
            "6d_o": ["ra", "dec", "pmra", "pmdec", "plx", "rv"],
            "5d_h": ["l", "b", "pm_l_cosb", "pm_b", "plx"],
            "3d_v": ["U", "V", "W"],
            "6d_p": ["X", "Y", "Z", "U", "V", "W"],
        }

        if self.dim_mode not in feature_map:
            self.logger.error(
                f"❌ [🧪测试核] 无法识别未知的 dim_mode: '{self.dim_mode}'"
            )
            raise ValueError(f"❌ 未知的特征维度模式: '{self.dim_mode}'")

        self.features = feature_map[self.dim_mode]

        # 提取聚类超参数
        self.cluster_algo = self.config.get("cluster_algo", "dbscan").lower()
        self.dbscan_eps = self.config.get("dbscan_eps", 0.3)
        self.dbscan_min_samples = self.config.get("dbscan_min_samples", 10)
        self.hdbscan_min_cluster_size = self.config.get("hdbscan_min_cluster_size", 15)
        self.hdbscan_min_samples = self.config.get("hdbscan_min_samples", None)
        self.hdbscan_eps = self.config.get("hdbscan_cluster_selection_epsilon", 0.0)

        self.logger.info(
            f"🧪 [PriorGMMEx] 实验性内核加载成功 | 模式: {self.dim_mode.upper()} | 算法: {self.cluster_algo.upper()} | 维度轴: {self.features}"
        )

    def _apply_adaptive_centering(
        self, df: pd.DataFrame, centers_dict: dict = None
    ) -> tuple:
        """
        [私有方法] 全相空间自适应坐标中心化处理。
        
        通过减去中值，消除绝对坐标（如 RA/Dec 或大数值 X/Y/Z）对 StandardScaler 方差拉伸的负面影响，
        提升标准化后特征空间的数值稳定性。
        """
        df_copy = df.copy()
        out_centers = {}

        # 需要进行一阶矩中心化处理的特征对
        coordinate_pairs = [("ra", "dec"), ("l", "b"), ("x", "y", "z")]

        for group in coordinate_pairs:
            if all(col in self.features for col in group):
                for col in group:
                    if centers_dict and col in centers_dict:
                        center_val = centers_dict[col]
                    else:
                        center_val = float(df_copy[col].median())

                    df_copy[col] = df_copy[col] - center_val
                    out_centers[col] = center_val

                self.logger.debug(
                    f"🧪 [物理核中心化] 坐标组 {group} 修正完成。坐标锚点: { {k: round(v, 4) for k, v in out_centers.items()} }"
                )
                break  # 匹配到一个坐标组即可

        return df_copy, out_centers
    
    def _calculate_adaptive_eps(self, df_seeds_field: pd.DataFrame) -> float:
        """
        🧬 [DBSCAN 专属物理算法] 动态感知局部天区背景密度，反演自适应最优领域半径 EPS。

        还原论文思路：
          - Step A: 解算真实星表相空间的 (minPts - 1) 阶最近邻距离（k-NND）分布的极小值，感知高密度突变包络。
          - Step B: 通过高斯核密度估计（KDE）重构无结构的平滑噪声统计底盘，计算背景随机聚集期望值。
          - Step C: 算术平均调和两极，在“提高捕获率”与“防御野星污染”之间达成最优物理平衡。
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
            nbrs_rand = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_rand)
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

    def fit(self, df_seeds: pd.DataFrame, df_field: pd.DataFrame) -> GMMModelParamsEx:
        """
        训练接口：利用种子星与全域背景构建高斯混合先验模型。

        Args:
            df_seeds (pd.DataFrame): 经过高质量筛选的种子星集合。
            df_field (pd.DataFrame): 包含背景与成员的全域背景数据。

        Returns:
            GMMModelParamsEx: 包含模型状态的参数对象。
        """
        self.logger.info(
            f"--- 🧪 [PriorGMMEx] 开始拟合先验模型 ({self.dim_mode.upper()}) ---"
        )

        # --------------------------------==================--------------------------------
        # 1. 字段非空清洗
        # --------------------------------==================--------------------------------
        n_field_orig = len(df_field)
        df_field = df_field.dropna(subset=self.features)
        if len(df_field) < n_field_orig:
            self.logger.warning(f" 字段清洗：剔除全域背景中 {n_field_orig - len(df_field)} 颗不完备星。")

        n_seeds_orig = len(df_seeds)
        df_seeds = df_seeds.dropna(subset=self.features)
        if len(df_seeds) < n_seeds_orig:
            self.logger.warning(f"🧹 字段清洗：剔除种子星集中 {n_seeds_orig - len(df_seeds)} 颗不完备星。")

        # 🛡️ 防御性编程：剔除特征空间中的完全重复点，防止 HDBSCAN 在处理重合点且 epsilon > 0 时
        # 触发 tree_to_labels 的 Cython 标量转换错误 (scikit-learn 内部 Bug)
        df_seeds = df_seeds.drop_duplicates(subset=self.features).copy()

        self.logger.info(f"📊 训练样本集: 种子星={len(df_seeds)} 颗 | 全域背景={len(df_field)} 颗")

        if len(df_seeds) < self.dbscan_min_samples:
            raise ValueError(
                f"❌ [内核溃缩] 有效种子星数量 ({len(df_seeds)}) 低于 DBSCAN 启动阈值 {self.dbscan_min_samples}"
            )

        # --------------------------------==================--------------------------------
        # 2. 自适应空间零点锚定与特征标准化
        # ------------------------------------------------==--------------------------------
        df_field_proc, center_coords = self._apply_adaptive_centering(df_field)
        df_seeds_proc, _ = self._apply_adaptive_centering(
            df_seeds, centers_dict=center_coords
        )

        scaler = StandardScaler()
        scaler.fit(df_field_proc[self.features])
        X_field_scaled = scaler.transform(df_field_proc[self.features])
        X_seeds_scaled = scaler.transform(df_seeds_proc[self.features])

        # --------------------------------==================--------------------------------
        # 3. 构建全域背景模型 (Field Model)
        # --------------------------------==================--------------------------------
        field_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        field_model.fit(X_field_scaled)
        self.logger.debug("🌌 全域背景场模型 (Field Model) 拟合完成。")

        # --------------------------------==================--------------------------------
        # 4. 稳健密度修剪与星团核心先验模型 (Cluster Model)
        # --------------------------------==================--------------------------------
        if self.cluster_algo == "hdbscan":
            # 🛡️ [Bugfix] 解决 sklearn HDBSCAN 的 tree_to_labels 崩溃问题
            # 原因：当 epsilon > 0 且数据中存在极近点（距离平局）时，内部 Cython 树遍历会触发类型转换错误。
            X_input = X_seeds_scaled.astype(np.float64, order='C')
            # 注入稍微明显的噪声 (1e-8) 以彻底打破数值平局
            X_input += np.random.normal(0, 1e-8, X_input.shape)

            try:
                db = HDBSCAN(
                    min_cluster_size=self.hdbscan_min_cluster_size,
                    min_samples=self.hdbscan_min_samples,
                    cluster_selection_epsilon=self.hdbscan_eps,
                    copy=True
                ).fit(X_input)
            except TypeError as te:
                # 如果依然触发该特定 Bug，强制降级 epsilon 为 0 以保全管线运行
                if "converted to Python scalars" in str(te) and self.hdbscan_eps > 0:
                    self.logger.warning("💥 HDBSCAN epsilon 搜索算法崩溃，正在启动兜底方案：禁用 epsilon 搜索并重试...")
                    db = HDBSCAN(
                        min_cluster_size=self.hdbscan_min_cluster_size,
                        min_samples=self.hdbscan_min_samples,
                        cluster_selection_epsilon=0.0,
                        copy=True
                    ).fit(X_input)
                else:
                    raise te

            algo_name = "HDBSCAN"
        else:
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(
                X_seeds_scaled
            )
            algo_name = "DBSCAN"
            
        labels = db.labels_
        
        # 🏷️ 无论是否触发防护机制，先记录原始聚类结果以便后续导出分析
        df_seeds_classified = df_seeds.copy()
        df_seeds_classified['density_status'] = 'noise'

        if np.all(labels == -1):
            self.logger.warning(
                f"⚠️ [内核拦截] {algo_name} 发生高维密度溃缩。将跳过密度修剪，回退为全量种子星构建分布。"
            )
            X_core = X_seeds_scaled
        else:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            n_seeds = len(X_seeds_scaled)
            
            # 找到最大的聚类
            best_cluster_idx = np.argmax(counts) if len(counts) > 0 else -1
            n_best = counts[best_cluster_idx] if best_cluster_idx != -1 else 0
            
            # 🛡️ 引入拦截防护机制：如果核心样本占比低于 20%，视为“过度清洗”或“密度溃缩”
            if n_best < (n_seeds * 0.2):
                self.logger.warning(
                    f"⚠️ [内核警告] {algo_name} 拦截过于激进 ({n_best}/{n_seeds})，"
                    f"可能发生密度溃缩。自动启动防护机制：改用全量种子星构建先验。"
                )
                X_core = X_seeds_scaled
            else:
                best_cluster = unique_labels[best_cluster_idx]
                cluster_mask = labels == best_cluster
                X_core = X_seeds_scaled[cluster_mask]
                # 标记该簇为核心成员
                df_seeds_classified.loc[df_seeds_classified.index[cluster_mask], 'density_status'] = 'core'

                self.logger.info(
                    f"🎯 [内核拦截] {algo_name} 完成：识别到噪声 {np.sum(labels == -1)} 颗，提取核心样本 {len(X_core)} 颗。"
                )

        cluster_model = GaussianMixture(
            n_components=1, covariance_type="full", random_state=42
        )
        cluster_model.fit(X_core)
        self.logger.info(
            f"✨ [内核验证] 星团先验模型拟合成功。归一化中心偏移: { {f: round(v, 4) for f, v in zip(self.features, cluster_model.means_[0])} }"
        )

        return GMMModelParamsEx(
            cluster_model=cluster_model,
            field_model=field_model,
            scaler=scaler,
            n_core_samples=len(X_core),
            dim_mode=self.dim_mode,
            features_used=self.features,
            df_seeds_classified=df_seeds_classified,
            center_coords=center_coords,
        )

    def predict(
        self,
        df_predict: pd.DataFrame,
        params: GMMModelParamsEx,
        max_iter: int = 250,
        tol: float = 1e-6,
    ) -> pd.DataFrame:
        """
        推理接口：执行基于先验的递归极大似然收敛计算。

        Args:
            df_predict (pd.DataFrame): 待判定的天体数据帧。
            params (GMMModelParamsEx): 训练好的模型参数对象。
            max_iter (int): 最大收敛迭代步数。
            tol (float): 收敛停止的容差阈值。

        Returns:
            pd.DataFrame: 包含 ID 和计算得出的成员概率（prob）的结果集。
        """
        total_stars = len(df_predict)
        self.logger.info(f"--- 🧪 [PriorGMMEx] 启动递归推理流程 | 目标量: {total_stars} ---")

        # --------------------------------==================--------------------------------
        # 1. 物理坐标对齐与标准化映射
        # --------------------------------==================--------------------------------
        df_proc, _ = self._apply_adaptive_centering(
            df_predict, centers_dict=params.center_coords
        )
        X_scaled = params.scaler.transform(df_proc[params.features_used])

        # --------------------------------==================--------------------------------
        # 2. 计算高维似然（Likelihood）
        # --------------------------------==================--------------------------------
        p_cl = np.exp(params.cluster_model.score_samples(X_scaled))
        p_fi = np.exp(params.field_model.score_samples(X_scaled))

        # --------------------------------==================--------------------------------
        # 3. 先验混合空间递归优化迭代
        # --------------------------------==================--------------------------------
        f_current = params.n_core_samples / total_stars
        self.logger.info(f"🔮 初始星团成员密度比估计 (Initial f): {f_current:.5f}")

        iteration = 0
        probs = np.zeros(total_stars)

        for i in range(max_iter):
            iteration = i + 1
            num = p_cl * f_current
            den = num + p_fi * (1.0 - f_current)
            probs = num / (den + 1e-15)

            f_new = np.mean(probs)
            diff = abs(f_new - f_current)

            if iteration % 10 == 0 or diff < tol:
                self.logger.debug(
                    f"🔄 [测试核迭代] 步数 {iteration:03d} | 空间密度 f = {f_new:.6f} | delta = {diff:.2e}"
                )

            if diff < tol:
                self.logger.info(
                    f"✅ [测试核收敛] 极大似然递归成功收敛！总步数 = {iteration} | 最终混合权重 f = {f_new:.6f}"
                )
                break
            f_current = f_new
        else:
            self.logger.warning(
                f"⚠️ [测试核警告] 达到最大安全迭代步数 ({max_iter}) 仍未完全收敛。残留 delta: {diff:.2e}"
            )

        # --------------------------------==================--------------------------------
        # 4. 成员星最终划分统计与数据收拢
        # --------------------------------==================--------------------------------
        high_prob_mask = probs > MEMBER_SAMPLE_THRESHOLD
        n_members = np.sum(high_prob_mask)
        self.logger.info(
            f"✨ [测试核收工] 判定任务顺利结束 | 成功掘出星团高置信度成员星: {n_members} 颗。"
        )
        self.logger.info(f"--- 🧪 PriorGMMEx 管道内核运行结束 ---")

        return pd.DataFrame(
            {
                STD_COLS.get("ID", "id"): df_predict["id"].to_numpy(),
                STD_COLS.get("PROB", "prob"): probs,
            }
        )
