"""
simbad_client.py - SimbadClient: SIMBAD 数据库文献与天体核查客户端
专注于处理 astroquery.simbad 网络查询、ID 格式化与响应数据结构化。
"""
import logging
import pandas as pd
from pathlib import Path
from astroquery.simbad import Simbad

import config as cfg
from config import STD_COLS


class SimbadClient:
    """SIMBAD 外部数据库服务客户端"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("AstroPipeline.SimbadClient")

    def query_sources(self, df_target: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
        """
        批量查询目标天体的 SIMBAD 发表文献量与最新文献 Bibcode。

        Args:
            df_target (pd.DataFrame): 包含 'id' 列的天体数据帧
            top_n (int, optional): 仅查询前 N 个高概率天体

        Returns:
            pd.DataFrame: 包含 SIMBAD 交叉核验结果的 DataFrame
        """
        if df_target is None or df_target.empty:
            self.logger.warning("输入的查询数据集为空，跳过 SIMBAD 查询。")
            return pd.DataFrame()

        prob_col = STD_COLS.get("PROB", "prob")
        if top_n and prob_col in df_target.columns:
            df_target = df_target.sort_values(prob_col, ascending=False).head(top_n)

        star_ids = df_target["id"].tolist()
        self.logger.info(f"🌐 [SIMBAD API] 准备向 SIMBAD 发起 {len(star_ids)} 颗天体的文献核实...")

        # 配置 astroquery.simbad 字段
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields("p_count")

        results = []
        for star_id in star_ids:
            formatted_id = f"Gaia DR3 {star_id}" if isinstance(star_id, (int, float)) else str(star_id)
            try:
                result_table = Simbad.query_object(formatted_id)
                bib_table = Simbad.query_bibobj(formatted_id)

                if result_table is not None:
                    p_count = result_table[0]["P_COUNT"]
                    latest_bib = bib_table["bibcode"][0] if bib_table is not None else "None"
                    results.append({
                        "id": star_id,
                        "simbad_name": formatted_id,
                        "pub_count": p_count,
                        "latest_ref": latest_bib,
                        "simbad_status": "Found",
                    })
                else:
                    results.append({
                        "id": star_id,
                        "simbad_name": formatted_id,
                        "pub_count": 0,
                        "latest_ref": "None",
                        "simbad_status": "Not Found",
                    })
            except Exception as e:
                self.logger.error(f"❌ [SIMBAD API] 查询 {formatted_id} 发生错误: {e}")
                results.append({
                    "id": star_id,
                    "simbad_name": formatted_id,
                    "pub_count": -1,
                    "latest_ref": "Error",
                    "simbad_status": "Error",
                })

        df_simbad = pd.DataFrame(results)
        final_report = pd.merge(df_target, df_simbad, on="id", how="left")
        return final_report

    def export_audit_csv(self, df_target: pd.DataFrame, output_name: str, top_n: int = None) -> Path:
        """查询并直接将 SIMBAD 审计报告导出为 CSV 文件。"""
        final_report = self.query_sources(df_target, top_n=top_n)
        if final_report.empty:
            return None

        output_path = cfg.EXPORT_DIR / f"simbad_audit_{output_name}.csv"
        final_report.to_csv(output_path, index=False)
        self.logger.info(f"💾 [SIMBAD API] 审计报告已保存至: {output_path}")
        return output_path