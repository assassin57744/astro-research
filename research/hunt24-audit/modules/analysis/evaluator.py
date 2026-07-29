"""
evaluator.py - AuditEvaluator: 交叉比对与算法评估引擎
专注于本地集合运算、文献 Recall 计算及差异源筛选。
"""
import logging
import pandas as pd

import config as cfg
from config import STD_COLS, TMPL
from modules.analysis.simbad_client import SimbadClient


class AuditEvaluator:
    """天体物理交叉审计与算法性能评估器"""

    def __init__(self, db_instance, logger=None):
        self.db = db_instance
        self.logger = logger or logging.getLogger("AstroPipeline.Evaluator")
        # 组合独立出来的 SIMBAD 客户端
        self.simbad_client = SimbadClient(logger=self.logger)

    def run_cross_match_analysis(self, result_table: str, ref_table: str, cluster_id: str, category: str, mode: str) -> str:
        """[交叉匹配] 提取我方高概率但文献低概率的候选源，注册为差异视图。"""
        self.logger.info(f"🚀 开始与参考表 [{ref_table}] 进行匹配分析...")

        prob_col = STD_COLS["PROB"]
        ref_prob_col = TMPL.COL_PROB.format(idx=category)

        discrepancy_view = TMPL.V_DIFF.format(
            cluster=cluster_id.lower(),
            category=category,
            mode=mode,
            idx=ref_table,
        )
        sql = f"""
            SELECT * FROM {result_table}
            WHERE {prob_col} > {cfg.AUDIT_PROB_HIGH} 
              AND ({ref_prob_col} < {cfg.AUDIT_PROB_LOW} OR {ref_prob_col} IS NULL)
        """
        self.db.register_view_from_sql(discrepancy_view, sql)
        count = self.db.get_row_count(discrepancy_view)
        self.logger.info(f"🔍 识别到 {count} 个潜在新成员候选，已存入视图: {discrepancy_view}")

        return discrepancy_view

    def analyze_audit_performance(self, t_audit_report: str) -> dict:
        """[算法评估] 计算 Recall 召回率和新发现数量。"""
        self.logger.info(f"📊 正在计算算法性能指标: {t_audit_report}")

        sql = f"""
            SELECT audit_status, count(*) as count, avg(mag) as avg_mag
            FROM {t_audit_report}
            GROUP BY audit_status
        """
        df_stats = self.db.query(sql).set_index("audit_status")

        hits = df_stats.get("count", {}).get("Confirmed Member", 0)
        misses = df_stats.get("count", {}).get("Literature Only", 0)
        total_lit = hits + misses
        recall = (hits / total_lit * 100) if total_lit > 0 else 0.0
        new_candidates = df_stats.get("count", {}).get("New Candidate", 0)

        self.logger.info("=" * 50)
        self.logger.info(f"📈 算法文献召回率: {recall:.2f}% ({hits}/{total_lit})")
        self.logger.info(f"✨ 物理符合但文献未收录的新源: {new_candidates} 颗")
        self.logger.info("=" * 50)

        return {
            "recall": recall,
            "new_discoveries": new_candidates,
            "missing_count": misses,
        }

    def audit_literature_via_simbad(self, df_target: pd.DataFrame, output_name: str, top_n: int = None) -> pd.DataFrame:
        """委派给独立的 SimbadClient 进行外部文献核验与导出。"""
        return self.simbad_client.export_audit_csv(df_target, output_name, top_n=top_n)