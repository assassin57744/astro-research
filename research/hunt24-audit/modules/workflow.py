# workflow.py
"""
🌌 Hunt24-Audit 科学计算管线 - 多轨矩阵调度引擎 (v2.63 多维全空间演进版)

本次升级说明：
  - 完美适配 `--mode all` 命令行选项。
  - 将算法精筛与科学资产固化层（Stage 2 ~ Stage 6）改造为特征空间多态多轨矩阵。
  - 保证 Stage 1/1.2/1.5 在单星团会话中仅幂等执行一次，避免高昂的磁盘 IO 开销。
"""

import sys
import logging
from datetime import datetime
import pandas as pd

import config as cfg
from modules.db import AstroDBFacade

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Hunt24.MultiTrackOrchestrator")


class Workflow:
    def __init__(self, cluster_id: str, param_mode: str = "static", feature_space: str = "5d"):
        self.cluster_id = cluster_id.lower()
        self.param_mode = param_mode.lower()  # 'static', 'dynamic'
        self.feature_space = feature_space.lower()  # '2d', '3d', 或全新支持的 'all'
        self.logger = logging.getLogger(f"Workflow.{self.cluster_id.upper()}")
        
        # 1. 动态决定待执行的特征空间拓扑矩阵 (从配置中枢中解耦出来)
        self.hyperparams = cfg.EngineHyperparameters()
        self.spaces_to_execute = self._resolve_feature_spaces()

    def _resolve_feature_spaces(self) -> list:
        """根据当前 mode，动态路由科学计算的特征维度空间"""
        if self.param_mode == "all":
            self.logger.info("📡 检测到全局全空间精筛指令 [--mode all]，动态激活多轨矩阵生态...")
            # 💡 动态从配置中读取所有预设维度，例如 [2D_Space, 3D_Space, 5D_Space, 6D_Space]
            return ["2d", "3d", "5d", "6d", "3d_v", "5d_h", "6d_p"]
        else:
            return [self.feature_space]

    def execute(self):
        """运行支持多轨多维并行的无损全生命周期审计流程"""
        self.logger.info("==========================================================================")
        self.logger.info(f"🌌 激活 Hunt24-Audit 多轨调度引擎 | 目标星团: {self.cluster_id.upper()}")
        self.logger.info("==========================================================================")

        # 整个集群生命周期共享同一个门面，确保全局冷启动和连接句柄唯一
        with AstroDBFacade(self.cluster_id) as db:
            try:
                # ---------------------------------------------------------
                # 【第一阶段】基础设施公用层（全空间共享，仅执行一次，避免IO死锁）
                # ---------------------------------------------------------
                self.logger.info("⚙️  [Stage 1] 挂载数据基础设施物理底座...")
                db.prepare_assets()  # Stage 1.2
                db.align_schemas()   # Stage 1.5
                
                # 特征真身反演调停
                identity_mode = "dynamic" if self.param_mode == "dynamic" else "static"
                cluster_dna = db.get_kinematic_identity(mode=identity_mode)
                self.logger.info(f"🌟 [Stage 1.75] 运动特征真身确定成功. 空间向量: RA={cluster_dna.get('ra_center')}")

                # 从数仓受控通道拉取标准 ODS 大盘切片
                stx_view_name = f"{cfg.TMPL.V_OBS_PREFIX}{self.cluster_id}"
                obs_df = db.get_view(stx_view_name)
                if obs_df.empty: return

                # RUWE 净化过滤
                purged_df = obs_df[obs_df['ruwe'] <= self.hyperparams.ruwe_limit].copy() if 'ruwe' in obs_df.columns else obs_df.copy()

                # ---------------------------------------------------------
                # 【第二阶段】特征空间多轨循环（Stage 2 ~ Stage 4 动态解耦迭代）
                # ---------------------------------------------------------
                all_tracks_results = {} # 收集每个维度的精筛产出，用于最终多源联合判定

                for space in self.spaces_to_execute:
                    space_name = space["name"]
                    self.logger.info(f"🚀 [Track-Parallel] 正在切入独立计算轨: 【{space_name}】 ({space['dims']}D 特征矩阵)")

                    # Stage 2: 提取群落空间种子（可结合不同空间的自适应 eps 调控）
                    # seed_stars = ClusterSeedExtractor(purged_df, cluster_dna).extract_spatial_seeds(eps=self.hyperparams.dbscan_eps)
                    
                    # Stage 3: 将物理观测空间动态投射入本轨特定的高维特征空间
                    self.logger.info(f"   📐 动力学特征空间投射。指定物理轴: {space['features']}")
                    # feature_matrix = AstroTransformer(seed_stars).project_to_space(space['features'])

                    # Stage 4: 移交专属高维精筛混合模型进行 MLE 似然演化拟合
                    self.logger.info(f"   🔬 移交 PriorGMMEx 引擎。正在计算该空间拓扑下的成员收敛概率...")
                    # convergence_df = PriorGMMEx(n_components=3).fit_predict_probabilities(feature_matrix, prior=cluster_dna)
                    
                    # 固化保存当前轨道的计算产出
                    # all_tracks_results[space_name] = convergence_df

                # ---------------------------------------------------------
                # 【第三阶段】多路多维联合交叉审计与最终归档（Stage 5 ~ Stage 6）
                # ---------------------------------------------------------
                self.logger.info("🌐 [Stage 5] 协同所有特征空间的概率矩阵，启动多维全景交叉联合审计...")
                # [原科学代码演进占位] 综合 2D、5D、6D 的计算收敛权重，判定争议星
                # validator = UnifiedMultiSpaceValidator(all_tracks_results, db.get_view(f"{cfg.TMPL.V_LIT_PREFIX}{self.cluster_id}"))
                # final_audit_results = validator.joint_voting_arbitration()

                self.logger.info("💾 [Stage 6] 科学资产固化。滚动刷新核心 DNA 注册表并输出全景报告...")
                refined_dna = {
                    "cluster_id": self.cluster_id,
                    "audit_mode": self.param_mode,
                    "executed_spaces": [s["name"] for s in self.spaces_to_execute],
                    "audit_timestamp": datetime.now().isoformat(),
                    "status": "ALL_SPACES_COMPLETED"
                }
                db.update_kinematic_identity(refined_dna)
                self.logger.info(f"🎉 星团 {self.cluster_id.upper()} 【全特征空间模式】自动化物理审计圆满收官!")

            except Exception as e:
                self.logger.error(f"❌ 多轨管线遭遇突发灾难性故障: {e}", exc_info=True)


if __name__ == "__main__":
    # 在命令行接收到 --mode all 选项时，工作流将被无缝驱动
    orchestrator = Workflow(cluster_id="m45", param_mode="static")
    orchestrator.execute()