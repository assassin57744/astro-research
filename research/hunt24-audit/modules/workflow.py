"""
🌌 天文数据处理工作流总编排引擎（文献靶向审计订正版）。

职责：
  作为流水线的总导演（Orchestrator），按标准阶段控制管线的生命周期：
  Stage 1    : 内存轻量分配并初始化物理领域实体 `StarCluster`。
  Stage 1.5  : 联动基础设施层 `SchemaAligner` 自动驱动数据异构星表结构对齐（data_standardize）。
  Stage 1.75 : 在数仓底座就绪后，触发物理实体参数的正式同步与动态反演（解决生命周期死锁）。
  Stage 2    : 从数仓大盘抽取局部天区切片，驱动 `ClusterSeedExtractor` 提炼种子星。
  Stage 3    : 激活 `AstroTransformer` 特征工程转换引擎，完成动力学相空间特征矩阵反演。
  Stage 4    : 移交专属高维精筛引擎 `PriorGMMEx` 拟合相空间特征并计算成员收敛概率。
  Stage 5    : 触发 `UnifiedMemberValidator` 联动多源文献进行自动化联合审计（多路分流审计）。
  Stage 6    : 交付 `render_final_report` 固化科学资产并输出可视化简报。
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 基础设施与核心科学组件导入
from modules.db import AstroDB, AssetManager
from modules.cluster import StarCluster
from modules.schema_aligner import SchemaAligner
from modules.eps_estimator import CastroGinardEpsEstimator
from modules.transformer import AstroTransformer
from modules.pg_core import PriorGMM
from modules.validator import UnifiedMemberValidator
from modules.reporter import render_final_report

import config as cfg
from config import (
    MANIFEST,
    CLUSTERS,
    GMM_CONFIG,
    STD_COLS,
    MASTER_COLS,
    MEMBER_SAMPLE_THRESHOLD,
    GOLDEN_SAMPLE_THRESHOLD,
)


class AstroWorkflow:
    """天文数据处理工作流编排引擎。

    该类作为流水线的核心调度器，负责编排数仓底座、特征转换、先验高斯混合模型（PriorGMMEx）推理、
    以及多群体靶向文献比对与深度联合审计链路。
    """

    def __init__(self, db: AstroDB):
        """初始化工作流引擎。

        Args:
            db (AstroDB): 实例化的基础设施数仓底座。
        """
        self.db = db
        self.logger = logging.getLogger("AstroPipeline.Workflow")
        # 🎯 传入当前的全局配置文件清单给 SchemaAligner 初始化
        self.aligner = SchemaAligner(self.db, manifest=cfg.MANIFEST)
        self.extractor = CastroGinardEpsEstimator(min_pts=9, num_simulations=30, db=self.db)
        self.logger.info("⚙️ [Workflow Init] 工作流调度引擎初始化成功，已绑定物理数仓中介与 Schema 拓扑对齐器。")

    def execute(
        self,
        cluster_id: str,
        category: str,
        mode: str,
        algo: str,
        result_mode: str = "brief",
        reconstruct: str = "static",
    ) -> str:
        """执行端到端天体测量星团成员精筛与靶向文献漏报/误报深度审计。"""
        self.logger.info("=" * 80)
        self.logger.info(f"🎬 启动工作流引擎总调度总线...")
        self.logger.info(f"   目标星团: {cluster_id} | 维度模式: {mode} | 初筛算法: {algo}")
        self.logger.info(f"   对账文献: {category} | 资产装载策略: {reconstruct}")
        self.logger.info("=" * 80)

        # ----------------------------------------------------------------------
        # Stage 1: 实例化星团物理领域实体
        # ----------------------------------------------------------------------
        self.logger.info(f"⚡ [Stage 1] 正在轻量创建星团物理领域对象，锁定标识符: '{cluster_id}'")
        cluster = StarCluster(cluster_id=cluster_id, db_instance=self.db)
        cl_context = CLUSTERS.get(cluster_id.upper(), {})
        self.logger.info(f" ∟ 提取的星团静态元数据上下文: {cl_context}")

        # TODO: 此处初始化数据不全, 需要按照 field, seeds, reference和others四类加载 
        # ----------------------------------------------------------------------
        # Stage 1.5: 驱动资产层级标准化异构星表规范化对齐
        # ----------------------------------------------------------------------
        # 从该星团的物理上下文映射中，提取运行所需的初筛种子和全域背景参考星表
        field_idx = cl_context.get("FIELD_IDX") if isinstance(cl_context, dict) else getattr(cl_context, "FIELD_IDX", None)
        seed_idx = cl_context.get("SEED_IDX") if isinstance(cl_context, dict) else getattr(cl_context, "SEED_IDX", None)
        
        ref_tables = list(set([field_idx, seed_idx]))
        total_tables = len(ref_tables)
        
        self.logger.info(f"🛠️ [Stage 1.5] 启动资产层级标准化异构星表对齐流，共需处理 {total_tables} 个资产...")
        for i, tbl_idx in enumerate(ref_tables, 1):
            if not tbl_idx:
                self.logger.warning(f" ⚠️ 索引映射位置 [{i}] 发现未分配的资产标号，跳过。")
                continue
            self.logger.info(f" 📋 [{i}/{total_tables}] 正在调用 SchemaAligner 标准化参考资产: '{tbl_idx}'")
            self.aligner.data_standardize(
                data_idx=tbl_idx,
                data_cfg=MANIFEST[tbl_idx],
                cl_cfg=cl_context
            )
        self.logger.info("✅ [Stage 1.5] 全域异构多源星表结构拓扑与标准字段映射完全对齐。")

        # ----------------------------------------------------------------------
        # Stage 1.75: 🎯 订正：对齐 cluster.py 真实公开的 hydrate_from_db 方法名
        # ----------------------------------------------------------------------
        self.logger.info(f"🔄 [Stage 1.75] 激活星团科学物理先验参数装载，执行方法: cluster.hydrate_from_db(strategy='{reconstruct}')...")
        hydrate_success = cluster.load_or_reconstruct_parameters(mode=reconstruct)
        
        # 实时打印同步或反演出来的核心物理参数状态，确保持续透明
        self.logger.info(f" ∟ 参数装载状态: {hydrate_success} | 物理特征清单:")
        self.logger.info(f"   ▪ 视差先验 (plx_ref): {getattr(cluster, 'plx_ref', None)} mas")
        self.logger.info(f"   ▪ 赤经自行先验 (pmra_ref): {getattr(cluster, 'pmra_ref', None)} mas/yr")
        self.logger.info(f"   ▪ 赤纬自行先验 (pmdec_ref): {getattr(cluster, 'pmdec_ref', None)} mas/yr")
        self.logger.info(f"   ▪ 视向速度先验 (rv_ref): {getattr(cluster, 'rv_ref', None)} km/s")
        self.logger.info(f"✅ [Stage 1.75] 星团领域实体物理上下文装载与反演完毕: {cluster}")

        # ----------------------------------------------------------------------
        # Stage 2: 从数仓抽取全天区背景大盘，提炼高质量无监督种子星
        # ----------------------------------------------------------------------
        self.logger.info(f"🌱 [Stage 2] 开始从数仓底座提取目标局部大盘数据...")
        # 从配置清单定位清洗对齐后的标准视图 (stx_view)
        t_master = MANIFEST[field_idx]["stx_view"]
        
        # 提取当前星团天区的背景场恒星大盘数据
        self.logger.info(f" ∟ 正在加载数仓主大盘表数据: '{t_master}'")
        df_field_stars = self.db.execute(f"SELECT * FROM {t_master}").df()
        
        # ✨ 完美契约调用：无缝传参，直接捕获解耦后的 3 元组返回值
        self.logger.info(f" ∟ 移交 ClusterSeedExtractor 空间切片提炼粗筛种子群体...")
        # TODO: 此处后续逻辑需要优化 t_master 没有必要传入又传出
        df_seeds, df_field_stars, t_master = self.extractor.extract_seeds(
            df_seeds_field=df_field_stars, 
            algo=algo,
            t_master=t_master
        )
        
        self.logger.info(
            f" ∟ 提炼完毕。高置信度动力学种子数: {len(df_seeds)} 颗 | "
            f"背景场待筛样本总数: {len(df_field_stars)} 颗 | 目标大盘主表: '{t_master}'"
        )

        # ----------------------------------------------------------------------
        # Stage 3 & 4: 特征空间动力学反演与精筛 PriorGMMEx 先验模型推理推断
        # ----------------------------------------------------------------------
        self.logger.info("🔬 [Stage 3 & 4] 进入多维运动学相空间动力学反演与高维极大似然估计推断...")
        df_predicted_field = self._stage_precision_screening(
            df_seeds=df_seeds,
            df_field_stars=df_field_stars,
            cluster=cluster,
            mode=mode,
            category=category,
            t_master=t_master
        )
        self.logger.info(f"✅ 后验概率矩阵反演求解完成，概率记录维度: {df_predicted_field.shape}")

        # ----------------------------------------------------------------------
        # Stage 4.5: 标签同步与后处理（高置信金标与普通成员打标）
        # ----------------------------------------------------------------------
        self.logger.info("📊 [Stage 4.5] 激活推断产物后处理，同步 Master 表分类标签...")
        post_res = self._post_process_labels(t_master=t_master, cluster_name=cluster_id)
        if post_res.get("status") != "success":
            self.logger.error("❌ 后处理标签同步遭遇未预期中断，管线熔断。")
            return ""

        v_candidates = post_res["v_candidates"]
        self.logger.info(f" ∟ 分类结果快照成功投递至临时候选者视图: '{v_candidates}'")

        # ----------------------------------------------------------------------
        # Stage 5: 靶向文献交叉比对与多路分流深度物理审计
        # ----------------------------------------------------------------------
        # 拼接靶向历史文献在数仓中的标准化对齐视图名（如 aln_cg20_m45 或 aln_hunt_m45）
        target_aln_view = f"aln_{category.lower()}_{cluster_id.lower()}"
        self.logger.info(f"⚖️ [Stage 5] 启动历史交叉碰撞对账。靶向对账视图: [{target_aln_view}]")
        
        audit_res = self._prepare_audit_data(
            t_master=t_master, v_source=v_candidates, v_target=target_aln_view
        )

        # 初始化深度验证统计容器
        deep_stats_pg = {}
        deep_stats_ref = {}

        if audit_res.get("status") == "success":
            stats_cross = audit_res.get("stats", {})
            self.logger.info(f" ∟ 账目切片咬合成功。开启多路分流物理边界审计分支流...")
            
            # 🔗 子群体深度验证一：对潜在的历史漏报成员（PG Only）执行深度验证
            if stats_cross.get("PG Only", 0) > 0:
                v_pg_only = audit_res["v_audit_pg_only"]
                self.logger.info(f"🕵️‍♂️ [分流审计一] 发现算法独有成员（潜在历史漏报）{stats_cross['PG Only']} 颗，触发等龄线网格与 SIMBAD 追溯...")
                v_report_pg = self._run_deep_validator(
                    t_master=t_master, target_view=v_pg_only, cluster_name=cluster_id, mode=mode, audit_type="pg_only"
                )
                if v_report_pg:
                    deep_stats_pg = self.db.execute(
                        f"SELECT audit_status, count(*) FROM {t_master} WHERE {MASTER_COLS['X_MATCH']} = 'PG Only' AND audit_status IS NOT NULL GROUP BY audit_status"
                    ).df().set_index("audit_status").to_dict().get("count(*)", {})
                    self.logger.info(f"   ∟ 潜在漏报源多源交叉核验细分对账单: {deep_stats_pg}")
            else:
                self.logger.info("✅ 未发现算法独有的潜在漏报成员。")

            # 🔗 子群体深度验证二：对潜在的历史误报成员（Ref Only）执行深度验证
            if stats_cross.get("Ref Only", 0) > 0:
                v_ref_only = audit_res["v_audit_ref_only"]
                self.logger.info(f"🕵️‍♂️ [分流审计二] 发现文献独有成员（潜在历史误报）{stats_cross['Ref Only']} 颗，触发特征空间偏离度审计...")
                v_report_ref = self._run_deep_validator(
                    t_master=t_master, target_view=v_ref_only, cluster_name=cluster_id, mode=mode, audit_type="ref_only"
                )
                if v_report_ref:
                    deep_stats_ref = self.db.execute(
                        f"SELECT audit_status, count(*) FROM {t_master} WHERE {MASTER_COLS['X_MATCH']} = 'Ref Only' AND audit_status IS NOT NULL GROUP BY audit_status"
                    ).df().set_index("audit_status").to_dict().get("count(*)", {})
                    self.logger.info(f"   ∟ 潜在误报源偏离度核验细分对账单: {deep_stats_ref}")
            else:
                self.logger.info("✅ 未发现历史文献独有的潜在误报成员。")
        else:
            self.logger.warning(f"⚠️ 无法咬合历史对账单视图 [{target_aln_view}]，跳过深度交叉验证分支。")

        # ----------------------------------------------------------------------
        # Stage 6: 固化科学资产并输出可视化简报
        # ----------------------------------------------------------------------
        self.logger.info("📋 [Stage 6] 正在构建完备的多源文献审计最终科学成果报告...")
        # 为了保证下游报告组件可用，将完整带权大盘作为 v_all_audit_data 传入
        v_all_audit_data = f"v_audit_input_{cluster_id.upper()}"
        
        summary = render_final_report(
            target_cluster_id=cluster_id,
            target_category=category,
            mode=mode,
            algo=algo,
            ctx_cluster=cl_context,
            v_all_audit_data=v_all_audit_data,
            audit_res=audit_res,
            deep_stats_pg=deep_stats_pg,
            deep_stats_ref=deep_stats_ref,
            logger=self.logger,
        )

        try:
            self.logger.info("💾 [Snapshot] 正在保存当前星团资产级数据快照，固化中间数仓成果...")
            from modules.backup_manager import AstroBackupManager
            backup_mgr = AstroBackupManager(self.db)
            backup_mgr.execute_cluster_snapshot(cluster_id=cluster_id)
            self.logger.info("✅ 科学资产版本快照固化完毕。")
        except Exception as backup_err:
            self.logger.warning(f"⚠️ 资产快照保存失败，但不影响主链路交付: {backup_err}")

        self.logger.info(f"🎉 星团 '{cluster_id}' 成员精筛与靶向交叉审计管线成功闭环。")
        return summary

    def _stage_precision_screening(
        self,
        df_seeds: pd.DataFrame,
        df_field_stars: pd.DataFrame,
        cluster: StarCluster,
        mode: str,
        category: str,
        t_master: str,
    ) -> pd.DataFrame:
        """[Stage 3 & 4] 特征空间动力学反演与精筛 PriorGMMEx 模型推理。"""
        self.logger.info("🎬 [Stage 3] 激活天体测量特征工程转换引擎...")

        # 1. 提取物理常数并初始化 AstroTransformer
        cluster_rv = cluster.rv_ref if hasattr(cluster, "rv_ref") else None
        cluster_center = (cluster.ra_ref, cluster.dec_ref) if hasattr(cluster, "ra_ref") else (None, None)

        physics_transformer = AstroTransformer(
            cluster_rv=cluster_rv,
            cluster_center_icrs=cluster_center,
        )

        self.logger.info(" ∟ 正在对种子群体执行动力学坐标与协方差矩阵特征映射...")
        df_seeds_ext = physics_transformer.fit_transform_df(df_seeds, mode=mode)
        self.logger.info(" ∟ 正在对全天区背景场执行动力学坐标与协方差矩阵特征映射...")
        df_field_ext = physics_transformer.fit_transform_df(df_field_stars, mode=mode)

        feature_map = GMM_CONFIG.get("feature_map", {})
        required_features = feature_map.get(mode, ["pmra", "pmdec", "plx"])
        self.logger.info(f"🔬 当前相空间维度模式 [{mode}] 提取的核心物理特征: {required_features}")

        # 防御性脱空清洗
        df_field_final = self._defensive_nan_purge(df_field_ext, required_features, label="Target_field")
        df_seeds_final = self._defensive_nan_purge(df_seeds_ext, required_features, label="Seeds")

        # 混合先验初值反演碰撞
        target_ref_view = f"aln_{category.lower()}_{cluster.id.lower()}"
        f_init = None
        try:
            self.logger.info(f"🔭 正在检测对账历史视图 [{target_ref_view}] 以反演核心混合初值先验比例...")
            ref_df = self.db.execute(f"SELECT id FROM {target_ref_view};").df()
            if not ref_df.empty:
                ref_ids_set = set(ref_df["id"].to_numpy())
                field_ids_set = set(df_field_final["id"].to_numpy())
                matches = len(ref_ids_set.intersection(field_ids_set))
                f_init = matches / len(df_field_final)
                f_init = max(min(f_init, 0.30), 0.0005)  # 设立物理安全边界保护
                self.logger.info(f"✨ [先验初值反演] 历史星表 [{target_ref_view}] 共命中 {matches} 颗星。解算 f_init = {f_init:.6f}")
        except Exception as err:
            self.logger.warning(f"⚠️ 提取对账历史 footprint 比例失败: {err}")

        # 3. 实例化精筛测试核引擎
        self.logger.info("🎬 [Stage 4] 实例化高维精筛混合推理机 PriorGMMEx...")
        engine = PriorGMMEx(config=GMM_CONFIG)
        if hasattr(engine, "set_dim_mode"):
            engine.set_dim_mode(mode)

        enable_sub = GMM_CONFIG.get("enable_subsampling", False)
        sub_limit = GMM_CONFIG.get("subsampling_limit", 500000)

        df_field_for_fit = df_field_final
        if enable_sub and len(df_field_final) > sub_limit:
            self.logger.info(f"🚀 [下采样过滤] 背景场恒星过载 ({len(df_field_final)}), 采样前 {sub_limit} 颗进行快速拟合估计...")
            df_field_for_fit = df_field_final.sample(n=sub_limit, random_state=42)

        self.logger.info("🚜 正在拟合高斯混合先验空间概率场密度 (fit)...")
        gmm_params = engine.fit(df_seeds_final, df_field_for_fit)
        
        if f_init is not None and hasattr(gmm_params, "n_core_samples"):
            gmm_params.n_core_samples = int(f_init * len(df_field_final))
            self.logger.info(f"🧠 [内核修正] 成功将对齐指定文献分布的 f_init={f_init:.6f} 注入推理机。")

        self.logger.info(f"🔮 [全域推断] 正在对当前全域视场 ({len(df_field_final)} 颗恒星) 解算成员收敛概率...")
        df_prob_only = engine.predict(df_field_final, params=gmm_params)

        # 6. 数据血缘归拢并安全回刷至主表大盘
        df_field_predicted = df_field_final.merge(df_prob_only, on="id", how="left")
        df_field_predicted["prob"] = df_field_predicted["prob"].fillna(0.0)

        self.logger.info(f"🖋️ 正在将推断的连续成员概率向数仓大盘主表 [{t_master}] 刷写同步...")
        self.db.tag_master_table(t_master, df_prob_only)

        # 7. 向全局数仓注册隔离大盘视图，打通后续 Validator 的主表特征级联读取
        try:
            audit_view_name = f"v_audit_input_{cluster.id.upper()}"
            self.db.register_dataframe(audit_view_name, df_field_predicted)
            self.logger.info(f"📡 成功在数仓底层注册全景属性分析中转视图: '{audit_view_name}'")
        except Exception as ev:
            self.logger.error(f"❌ 注册中间属性视图失败: {ev}")

        return df_field_predicted

    # --------------------------------------------------------------------------
    # 💥【全功能找回：后处理、交叉碰撞对账、多路深度验证子模块】
    # --------------------------------------------------------------------------
    def _post_process_labels(self, t_master: str, cluster_name: str) -> dict:
        """同步主表普通候选者与高置信金标成员标签，并注册为算法候选视图。"""
        try:
            self.logger.info(f" ∟ 开始在主表 {t_master} 动态追加物理分类打标列...")
            self.db.execute(f"ALTER TABLE {t_master} ADD COLUMN IF NOT EXISTS is_golden BOOLEAN DEFAULT FALSE")
            self.db.execute(f"ALTER TABLE {t_master} ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT FALSE")

            condi_golden = f"{STD_COLS['PROB']} >= {GOLDEN_SAMPLE_THRESHOLD}"
            condi_candidates = f"{STD_COLS['PROB']} > {MEMBER_SAMPLE_THRESHOLD}"

            self.db.execute(f"UPDATE {t_master} SET is_golden = TRUE WHERE {condi_golden}")
            self.db.execute(f"UPDATE {t_master} SET is_candidate = TRUE WHERE {condi_candidates}")

            stats_sql = f"""
                SELECT 
                    count(*) FILTER (WHERE is_golden = TRUE) AS n_golden,
                    count(*) FILTER (WHERE is_candidate = TRUE) AS n_candidates
                FROM {t_master}
            """
            n_golden, n_candidates = self.db.execute(stats_sql).fetchone()

            self.logger.info(f"    🔹 高置信金标成员门限 (>= {GOLDEN_SAMPLE_THRESHOLD}): {n_golden} 颗")
            self.logger.info(f"    🔹 算法推断成员星门限 (> {MEMBER_SAMPLE_THRESHOLD}): {n_candidates} 颗")

            v_candidates = f"v_candidates_{cluster_name.lower()}"
            self.db.register_view_from_sql(v_candidates, f"SELECT * FROM {t_master} WHERE is_candidate = TRUE")
            return {"status": "success", "v_candidates": v_candidates}
        except Exception as e:
            self.logger.error(f"❌ 后处理打标故障: {e}")
            return {"status": "error", "message": str(e)}

    def _prepare_audit_data(self, t_master: str, v_source: str, v_target: str) -> dict:
        """联动预测候选视图与指定历史账目文献表交叉碰撞，解构 PG Only / Ref Only / Matched 子群体。"""
        if not self._verify_view_exists(v_target):
            self.logger.warning(f"⚠️ 靶向审计对账历史视图 '{v_target}' 不存在，跳过高维交叉对账。")
            return {"status": "warning", "message": f"视图 {v_target} 不存在"}

        col_x = MASTER_COLS["X_MATCH"]
        self.db.execute(f"ALTER TABLE {t_master} ADD COLUMN IF NOT EXISTS {col_x} VARCHAR")
        self.db.execute(f"UPDATE {t_master} SET {col_x} = NULL")

        # 高维外连接交叉碰撞分析 SQL
        sql_cross = f"""
            SELECT 
                COALESCE(m.id, h.id) as id,
                CASE 
                    WHEN m.prob > {MEMBER_SAMPLE_THRESHOLD} AND h.id IS NOT NULL THEN 'Matched'
                    WHEN m.prob > {MEMBER_SAMPLE_THRESHOLD} AND h.id IS NULL     THEN 'PG Only'
                    WHEN (m.id IS NULL OR m.prob <= {MEMBER_SAMPLE_THRESHOLD} OR m.prob IS NULL) 
                        AND h.id IS NOT NULL THEN 'Ref Only'
                END as {col_x}
            FROM {t_master} m
            FULL OUTER JOIN {v_target} h ON m.id = h.id
            WHERE m.prob > {MEMBER_SAMPLE_THRESHOLD} OR h.id IS NOT NULL
        """
        self.logger.info(" ∟ 驱动 FULL OUTER JOIN 执行两代星表全集合对账对账单计算...")
        df_x = self.db.query(sql_cross)
        self.db.tag_master_table(t_master, df_x)

        # 为下游 Validator 分流审计创建独立的内存子视图隔离天区
        v_audit_pg_only = "v_tmp_audit_pg_only"
        v_audit_ref_only = "v_tmp_audit_ref_only"
        self.db.register_view_from_sql(v_audit_pg_only, f"SELECT * FROM {t_master} WHERE {col_x} = 'PG Only'")
        self.db.register_view_from_sql(v_audit_ref_only, f"SELECT * FROM {t_master} WHERE {col_x} = 'Ref Only'")

        st_sql = f"SELECT {col_x}, count(*) FROM {t_master} WHERE {col_x} IS NOT NULL GROUP BY {col_x}"
        stats_raw = self.db.execute(st_sql).fetchall()
        stats_cross = {row[0]: row[1] for row in stats_raw}
        
        self.logger.info(
            f"📊 [交叉碰撞报告] Matched(双方一致): {stats_cross.get('Matched', 0)} 颗 | "
            f"PG Only(算法漏报): {stats_cross.get('PG Only', 0)} 颗 | "
            f"Ref Only(文献误报): {stats_cross.get('Ref Only', 0)} 颗"
        )
        return {
            "status": "success",
            "v_audit_pg_only": v_audit_pg_only,
            "v_audit_ref_only": v_audit_ref_only,
            "stats": stats_cross,
        }

    def _run_deep_validator(
        self, t_master: str, target_view: str, cluster_name: str, mode: str, audit_type: str
    ) -> str:
        """驱动深度联合验证器，对各路争议群体执行多源文献追溯与等龄线演化网格核验。"""
        try:
            # 补全中间所需的多维天体物理特征属性列并注册输入源
            field_idx = CLUSTERS[cluster_name.upper()]["FIELD_IDX"]
            t_base = MANIFEST[field_idx]["stx_view"]
            v_audit_input = self.db.register_audit_input_view(target_view, t_base)

            if not v_audit_input:
                self.logger.error(f"❌ 无法映射底层物理拓展视图，分支 [{audit_type}] 联合审计熔断。")
                return ""

            validator = UnifiedMemberValidator(
                cluster_id=cluster_name, db_instance=self.db, mode=mode
            )

            # 联动网络：自动触发增量互联网 SIMBAD 文献追溯缓存预热
            self._warm_up_literature_cache(validator, v_audit_input)
            
            self.logger.info(f" ∟ 正在启动科学网格核验模型计算分支 [{audit_type}] (run_full_audit_ex)...")
            audit_report_df = validator.run_full_audit_ex(v_audit_input)
            
            self.logger.info(f" ∟ 正在回写细分决策状态标签到主表...")
            self.db.tag_master_table(t_master, audit_report_df)

            # 将本路审计完备的报告结果注册为隔离可供导出的视图
            v_report = f"{t_master}_{audit_type}_audited_report"
            col_x = MASTER_COLS["X_MATCH"]
            x_match_val = "PG Only" if audit_type == "pg_only" else "Ref Only"
            
            sql_filter = f"SELECT * FROM {t_master} WHERE audit_status IS NOT NULL AND {col_x} = '{x_match_val}'"
            self.db.register_view_from_sql(v_report, sql_filter)
            return v_report
        except Exception as e:
            self.logger.error(f"❌ 驱动多路分支 [{audit_type}] 深度联合审计时遭遇故障: {e}", exc_info=True)
            return ""

    def _warm_up_literature_cache(self, validator: UnifiedMemberValidator, v_source: str):
        """提取子群体中缺失的源 ID 并完成 SIMBAD 增量文献缓存网络对齐。"""
        cache_table = validator.cache_table
        res = self.db.con.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = ?)",
            [cache_table.lower()],
        ).fetchone()

        if res and res[0]:
            col_info = self.db.con.execute(f"PRAGMA table_info({cache_table})").df()
            if "parent" not in col_info["name"].values:
                try:
                    self.db.con.execute(f"ALTER TABLE {cache_table} ADD COLUMN parent VARCHAR;")
                    self.logger.info(f"🛠️ [DDL Dynamic] 成功为本地缓存表 [{cache_table}] 追加 'parent' 列元数据拓扑。")
                except Exception as ddl_err:
                    self.logger.error(f"❌ 动态追加 parent 缓存物理列失败: {ddl_err}")

        sql_missing = f"""
            SELECT DISTINCT CAST(v.id AS VARCHAR) as id
            FROM {v_source} v
            LEFT JOIN {cache_table} c ON CAST(v.id AS VARCHAR) = c.gaia_dr3_id
            WHERE c.gaia_dr3_id IS NULL 
               OR c.parent IS NULL 
               OR TRIM(c.parent) = '' 
               OR LOWER(TRIM(c.parent)) = 'none'
        """
        df_missing = self.db.con.execute(sql_missing).df()
        ids_to_sync = df_missing["id"].tolist()

        if not ids_to_sync:
            self.logger.info("✅ 争议子群体文献缓存极度完备，跳过增量网络请求。")
            return

        self.logger.info(f"🌐 正在为 {len(ids_to_sync)} 个争议源启动增量本地/网络 SIMBAD 文献追溯同步...")
        validator.sync_simbad_cache(ids_to_sync)

    def _verify_view_exists(self, view_name: str) -> bool:
        """检查数仓中指定的视图或表是否存在。"""
        try:
            sql = f"SELECT 1 FROM information_schema.tables WHERE table_name = '{view_name.lower()}'"
            return self.db.con.execute(sql).fetchone() is not None
        except Exception:
            return False

    def _defensive_nan_purge(self, df: pd.DataFrame, features: list, label: str = "Dataset") -> pd.DataFrame:
        """🛡️ 针对天体测量缺失值的防御性净化工具方法。"""
        if df.empty:
            return df
        before_len = len(df)
        df_clean = df.dropna(subset=features).copy()
        dropped = before_len - len(df_clean)
        if dropped > 0:
            self.logger.warning(f"⚠️ [防御性过滤 - {label}]: 剔除了 {dropped} 颗多维物理特征缺失的天体记录（保留样本数: {len(df_clean)}）。")
        return df_clean
