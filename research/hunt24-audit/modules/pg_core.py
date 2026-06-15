import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from utils.decorators import astro_checkpoint  # 1. 导入装饰器

import config as cfg
from config import (
    STD_COLS,
)

@dataclass
class GMMModelParams:
    """
    模型参数容器：保存从训练阶段提取的所有先验知识。
    将模型状态（参数）与执行逻辑（类）解耦，便于保存和复用。
    """
    cluster_model: GaussianMixture
    field_model: GaussianMixture
    scaler: StandardScaler
    n_core_samples: int
    center_coords: tuple = (0.0, 0.0)

class PriorGMM:
    """
    PriorGMM: 种子星引导的先验迭代成员判定算法。
    
    接口设计遵循“无状态”原则：
    - fit(): 输入数据，产出 GMMModelParams 对象。
    - predict(): 接收数据与参数对象，产出概率结果。
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")
        self.config = config or {}        
        # 确定特征维度模式
        self.dim_mode = self.config.get("dim_mode", "3d").lower()

        feature_map = {
            "5d": ["ra", "dec", "pmra", "pmdec", "plx"],
            "4d": ["pmra", "pmdec", "plx", "color"],
            "3d": ["pmra", "pmdec", "plx"],
            "2d": ["pmra", "pmdec"],
        }
        self.features = feature_map.get(self.dim_mode, feature_map["3d"])
        
        # DBSCAN 超参数，用于从种子星提取核心
        self.dbscan_eps = self.config.get("dbscan_eps", 0.3)
        self.dbscan_min_samples = self.config.get("dbscan_min_samples", 10)

        self.logger.info(f"PriorGMM 初始化完成 (维度: {self.dim_mode.upper()}, 特征: {self.features})")

    def _preprocess_5d(self, df, center=None):
        """[私有方法] 处理 5D 坐标偏移，消除绝对坐标值对标准化的影响。"""
        if self.dim_mode != "5d":
            return df, (0.0, 0.0)
        
        df_copy = df.copy()
        if center is None:
            center = (df_copy["ra"].median(), df_copy["dec"].median())
            self.logger.debug(f"5D 坐标中心确定为: RA={center[0]:.4f}, Dec={center[1]:.4f}")
            
        df_copy["ra"] -= center[0]
        df_copy["dec"] -= center[1]
        return df_copy, center

    # @astro_checkpoint(cache_table_name="cache_full_pipeline_result")
    def fit(self, df_seeds: pd.DataFrame, df_field: pd.DataFrame) -> GMMModelParams:
        """
        训练接口：构建背景场模型与星团核心模型。
        
        - df_seeds: 经过高质量筛选(如 RUWE < 1.4)的种子星数据。
        - df_field: 目标天区的全量背景数据。
        """

        # --- 强制清洗全域数据 ---
        n_orig_field = len(df_field)
        df_field = df_field.dropna(subset=self.features).copy()
        n_clean_field = len(df_field)
        
        if n_orig_field > n_clean_field:
            self.logger.warning(f"从全域背景中剔除了 {n_orig_field - n_clean_field} 颗含有 NaN 的无效星")

        df_seeds = df_seeds.dropna(subset=self.features).copy()

        self.logger.info(f"--- PriorGMM 训练开始 ({self.dim_mode.upper()}) ---")
        self.logger.info(f"输入统计: 种子星={len(df_seeds)} 颗, 全域背景={len(df_field)} 颗")

        # 1. 坐标中心化处理
        df_all, center = self._preprocess_5d(df_field)
        df_seeds_proc, _ = self._preprocess_5d(df_seeds, center=center)

        # 2. 训练特征标准化器
        scaler = StandardScaler()
        scaler.fit(df_all[self.features])
        self.logger.debug(f"特征均值: {dict(zip(self.features, scaler.mean_))}")
        X_all_scaled = scaler.transform(df_all[self.features])

        # 3. 训练全域背景模型 (Field Model)
        field_model = GaussianMixture(n_components=1, covariance_type="full")
        field_model.fit(X_all_scaled)
        self.logger.debug("背景场 GMM (Field Model) 训练完成")

        # 4. 训练星团核心模型 (Cluster Model)
        X_seeds_scaled = scaler.transform(df_seeds_proc[self.features])
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(X_seeds_scaled)

        self.logger.info(f"DBSCAN 聚类结果: {np.unique(db.labels_, return_counts=True)} (标签: 颗数)")
        
        X_core = X_seeds_scaled[db.labels_ != -1]
        n_noise = np.sum(db.labels_ == -1)
        self.logger.info(f"DBSCAN 完成: 提取核心 {len(X_core)} 颗, 识别噪声 {n_noise} 颗")

        if len(X_core) < self.dbscan_min_samples:
            self.logger.error(f"核心样本不足: 找到 {len(X_core)} < 最小阈值 {self.dbscan_min_samples}")
            raise ValueError(f"DBSCAN 无法在 {self.dim_mode} 空间找到足够的聚类核心星。")

        cluster_model = GaussianMixture(n_components=1, covariance_type="full")
        cluster_model.fit(X_core)
        self.logger.info(f"星团核心 GMM (Cluster Model) 训练完成。中心位置(归一化): {cluster_model.means_[0]}")

        self.logger.info(f"模型构建成功。核心星样本数: {len(X_core)}")
        return GMMModelParams(
            cluster_model=cluster_model,
            field_model=field_model,
            scaler=scaler,
            n_core_samples=len(X_core),
            center_coords=center
        )

    def predict(self, df_predict: pd.DataFrame, params: GMMModelParams, 
                max_iter=250, tol=1e-6) -> pd.DataFrame:
        """
        推理接口：基于先验参数进行递归迭代，计算最终成员概率。
        
        - params: fit() 阶段产出的 GMMModelParams 对象。
        """
        self.logger.info(f"--- PriorGMM 推理开始 ---")
        self.logger.info(f"待判定目标总数: {len(df_predict)}")
        # 1. 数据预处理与标准化
        df_proc, _ = self._preprocess_5d(df_predict, center=params.center_coords)
        X_scaled = params.scaler.transform(df_proc[self.features])

        # 2. 计算 Likelihood (似然值)
        p_cl = np.exp(params.cluster_model.score_samples(X_scaled))
        p_fi = np.exp(params.field_model.score_samples(X_scaled))
        self.logger.debug(f"似然值计算完成 (P_cl max: {p_cl.max():.2e}, P_fi max: {p_fi.max():.2e})")

        # 3. 递归优化权重 f (星团星占比)
        f_current = params.n_core_samples / len(X_scaled)
        self.logger.info(f"初始成员占比估计 (Initial f): {f_current:.4f}")
        
        iteration = 0
        for i in range(max_iter):
            iteration = i + 1
            num = p_cl * f_current
            den = num + p_fi * (1 - f_current)
            probs = num / (den + 1e-15) # 避免分母为零
            
            f_new = np.mean(probs)
            diff = abs(f_new - f_current)

            # 每 5 次迭代打印一次中间状态，或者在最后一次迭代打印
            if iteration % 5 == 0 or diff < tol:
                self.logger.debug(f"迭代 {iteration:02d}: f = {f_new:.6f}, delta = {diff:.2e}")

            if diff < tol:
                self.logger.info(f"收敛成功: 迭代次数={iteration}, 最终权重 f={f_new:.6f}")
                break
            f_current = f_new
        else:
            self.logger.warning(f"达到最大迭代次数({max_iter})仍未收敛。最后 delta: {diff:.2e}")
        
        # 结果统计
        n_members = np.sum(probs > cfg.MEMBER_SAMPLE_THRESHOLD)
        self.logger.info(f"判定结果: 高概率成员(P>{cfg.MEMBER_SAMPLE_THRESHOLD})共 {n_members} 颗")
        self.logger.info(f"--- PriorGMM 任务结束 ---")
        return pd.DataFrame({
            STD_COLS["ID"]: df_predict["id"],
            STD_COLS["PROB"]: probs
        })