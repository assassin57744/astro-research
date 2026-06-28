"""
基础设施层：多源文献星表拓扑与结构规范化对齐器。

职责：
  1. 承接并实现原管线内部的多源星表结构标准化调度算法（data_standardize）。
  2. 通过数据驱动模型（Actions 路由），级联驱动跨文献星表的 STD（清洗）、STX（扩展）、ALN（联合对齐）层级动作。

大小写与字面量防御规范：
  - 传入的 ctx (上下文) 和其中的 id (星团名) 全程保持用户在命令行输入的原始字面量形态。
  - 仅在 adapter 映射检索或配置路由时临时做匹配，绝不在全局执行盲目的覆盖性大小写赋值，捍卫原始血缘。
"""

import logging
import config as cfg
from modules.db import AstroDB
import copy
from config import ClusterConfig, CatalogConfig

logger = logging.getLogger("AstroPipeline.SchemaAligner")


class SchemaAligner:
    """数仓拓扑规范化与异构文献资产对齐引擎。"""

    def __init__(self, db: AstroDB, manifest: dict = None):
        """
        注入唯一的数仓底座，并可选绑定全局清单（manifest）。
        """
        self.db = db
        # 兼容处理全局清单
        self.manifest = manifest if manifest is not None else {}
        self.logger = logging.getLogger("AstroPipeline.SchemaAligner")

    def data_standardize(
        self, data_idx: str, data_cfg: CatalogConfig, cl_cfg: ClusterConfig = None
    ) -> None:
        """
        核心数据表结构标准化调度算法（Schema Standardization）。
        """
        self.logger.info(
            f"🚀 [Schema Aligner] 正在执行数据结构标准化，当前源: {data_idx}"
        )

        # 🛡️ 核心防线：浅拷贝克隆强类型上下文，防止多源循环时状态污染
        local_cl_cfg = copy.copy(cl_cfg) if cl_cfg else None

        # 从资产配置中安全提取原子清洗动作流
        actions = (
            data_cfg.actions
            if hasattr(data_cfg, "actions")
            else data_cfg.get("actions", {})
        )
        if not actions:
            return

        # 打印一下当前的命名自适应解析状态（仅作 debug 审计）
        if local_cl_cfg:
            actual_cat_name = local_cl_cfg.get_cat_name(data_idx, self.manifest)
            self.logger.debug(
                f" ∟ 命名自适应适配完成: 资产 [{data_idx}] -> 目标物理库映射名为: {actual_cat_name}"
            )

        # 级联调度层级清洗动作
        for layer in ["std", "stx", "aln"]:
            if layer in actions:
                action_func = actions[layer]
                # 🚀 纯正的强类型对象 local_ctx 带着完整的生命周期和自适应方法流向下游
                action_func(self.db, data_idx, data_cfg, self.manifest, local_cl_cfg)

        self.logger.info(f"✅ 源 [{data_idx}] 的数据结构对齐与拓扑清洗完毕。")
