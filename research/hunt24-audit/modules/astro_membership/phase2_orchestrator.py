# modules/astro_membership/phase2_orchestrator.py
# --------------------------------==================--------------------------------
# Phase 2: M Membership 广域沙盘消歧与多组分动力学亚结构重构顶层编排器
# --------------------------------==================--------------------------------

import logging
import os
import numpy as np
import pandas as pd

# 🌟 完美的扁平化平级与子包相对导入
from .cutter import DensityFieldCutter
from .substructure.identity import IdentityComponentModeller
from .substructure.dual_component import DualComponentModeller
from .substructure.triple_component import TripleComponentModeller

logger = logging.getLogger(__name__)


class Phase2Orchestrator:
    """Phase 2 顶层流控中心

    负责动态调度单/双/三组分流形解剖引擎，并结合自适应标量场裁剪器完成广域大沙盘打标
    """

    def __init__(self, config: dict):
        self.config = config
        self.features = ["ra", "dec", "pmra", "pmdec", "parallax"]
        self._route_substructure_model()

    def _route_substructure_model(self):
        """策略动态路由：根据静态配置或从历史文献重建的物理特征，派发底层的解剖引擎"""
        # 支持三种解构路径：0-Identity, 1-Dual (Core+Tail), 2-Triple (Core+L+T)
        path_mode = self.config.get("SUBSTRUCTURE_PATH_MODE", 0)

        if path_mode == 0:
            logger.info(
                "🎯 [Route] 检测到无显著潮汐尾紧凑星团，路由至路径 0: Identity (单组分基准)"
            )
            self.modeller = IdentityComponentModeller(self.config)
        elif path_mode == 1:
            logger.info(
                "🎯 [Route] 检测到常规长尾潮汐撕裂，路由至路径 1: DualComponent (双组分核心+尾)"
            )
            self.modeller = DualComponentModeller(self.config)
        elif path_mode == 2:
            logger.info(
                "🎯 [Route] 检测到巨型球团/强非对称战场，路由至路径 2: TripleComponent (三组分前导+后随)"
            )
            self.modeller = TripleComponentModeller(self.config)
        else:
            raise ValueError(
                f"❌ [Route Error] 未知的亚结构重构路径模式: {path_mode}"
            )

    def run_pipeline(
        self, df_master: pd.DataFrame, df_field: pd.DataFrame
    ) -> pd.DataFrame:
        """全量编排核心函数

        :param df_master: 一阶段 PriorGMM 沉淀的高纯度核心种子及历史文献参数重建资产
        :param df_field: 广域大沙盘未消歧洗涤的几百万颗背景野星大表
        :return: 经过亚结构剥离与自适应断层去噪后的终极成员星资产表
        """
        logger.info(
            f"⚡ [Phase 2] 启动二阶段轰鸣。Master 种子数: {len(df_master)} | 广域大沙盘输入: {len(df_field)}"
        )

        # --------------------------------==================--------------------------------
        # 1. 动力学重构：激活底层解剖引擎，吃入 Master 种子进行 5D 相空间重构
        # --------------------------------==================--------------------------------
        self.modeller.fit(df_master)

        # --------------------------------==================--------------------------------
        # 2. 广域大沙盘概率/似然解算（投影至全域背景大沙盘）
        # --------------------------------==================--------------------------------
        logger.info("🔮 正在将亚结构流形本征基因投影至全域背景大沙盘...")
        df_demarcated = self.modeller.predict_membership(df_field)

        # --------------------------------==================--------------------------------
        # 3. 闭环消歧截断：调用无模型依赖的自适应密度场裁剪算子
        # --------------------------------==================--------------------------------
        sigma_cutoff = self.config.get("THRESHOLD_SIGMA", 3.0)
        wash_method = self.config.get("THRESHOLD_METHOD", "chi2")  # 'chi2' 或 'knee'

        logger.info(
            f"🛡️ 调起终极后处理斩杀算子 [Method: {wash_method} | Sigma: {sigma_cutoff}]"
        )
        cutter = DensityFieldCutter(
            sigma_cutoff=sigma_cutoff, method=wash_method
        )

        # 动态决定斩杀的目标似然/概率资产列
        # 路径 0（Identity）直接斩杀绝对对数似然；多组分路径斩杀联合概率或 p_total_cluster
        path_mode = self.config.get("SUBSTRUCTURE_PATH_MODE", 0)
        target_score_col = (
            "log_p_identity" if path_mode == 0 else "p_total_cluster"
        )

        # 核心轰鸣：直接裁剪广域大沙盘标量密度场，完成终极打标
        df_final_audit = cutter.cut_field(
            df_all=df_demarcated,
            target_score_col=target_score_col,
            features=self.features,
        )

        # --------------------------------==================--------------------------------
        # 4. 最终状态审计与管线对齐
        # --------------------------------==================--------------------------------
        n_final_members = int(df_final_audit["is_member"].sum())
        logger.info(
            f"🏆 [Phase 2 Complete] 二阶段亚结构重构与终极去噪洗涤合龙！"
        )
        logger.info(
            f"   最终截留成员星数 (is_member==True): {n_final_members} 颗"
        )

        # 针对多组分路径，额外打印亚结构泛成员的宏观动力学分布
        if "p_leading" in df_final_audit.columns and n_final_members > 0:
            member_mask = df_final_audit["is_member"] == True
            n_leading = int(
                (
                    df_final_audit.loc[member_mask, "p_leading"]
                    > df_final_audit.loc[member_mask, "p_trailing"]
                ).sum()
            )
            logger.info(
                f"   运动学形态学精细分解 -> 潮汐前导尾阵营: {n_leading} 颗 | 后随尾阵营: {n_final_members - n_leading} 颗"
            )

        return df_final_audit