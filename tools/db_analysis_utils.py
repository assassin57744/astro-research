import logging
import pandas as pd

def process_id_conflicts(
    db_instance,
    logger_instance: logging.Logger,
    dr2_view: str,
    bridge_view: str,
    id_col_dr3: str = "id",
    id_col_dr2_ref: str = "id_dr2",
    id_col_dr2_bridge: str = "id_dr2",
    prob_col: str = "prob"
) -> pd.DataFrame:
    """
    检测并返回 DR3 ID 与 DR2 ID 之间的多对一冲突。

    Args:
        db_instance: 数据库连接实例 (需提供 .query() 方法返回 Pandas DataFrame)。
        logger_instance: 日志记录器实例。
        dr2_view (str): 原始 DR2 数据视图名。
        bridge_view (str): 包含 ID 映射关系的桥接表视图名。
        id_col_dr3 (str): 桥接表中 DR3 ID 的列名。
        id_col_dr2_ref (str): 原始 DR2 视图中 DR2 ID 的列名。
        id_col_dr2_bridge (str): 桥接表中 DR2 ID 的列名。
        prob_col (str): 原始 DR2 视图中概率的列名。

    Returns:
        pd.DataFrame: 包含冲突信息的 DataFrame，如果无冲突则为空。
    """
    conflict_query = f"""
        SELECT
            nb.{id_col_dr3} AS dr3_id,
            count(*) as match_count,
            list(ref.{id_col_dr2_ref}) as original_dr2_list,
            list(ref.{prob_col}) as prob_list
        FROM {dr2_view} AS ref
        JOIN {bridge_view} AS nb ON ref.{id_col_dr2_ref} = nb.{id_col_dr2_bridge}
        GROUP BY nb.{id_col_dr3}
        HAVING count(*) > 1
    """
    df_conflicts = db_instance.query(conflict_query)

    if not df_conflicts.empty:
        num_conflicts = len(df_conflicts)
        total_affected = df_conflicts["match_count"].sum()
        logger_instance.warning(f"⚠️ 检测到 {num_conflicts} 组多对一匹配 (共涉及 {total_affected} 条记录)")
        for _, row in df_conflicts.head(5).iterrows():
            logger_instance.debug(f"  - DR3 {row['dr3_id']}: 关联了 DR2 {row['original_dr2_list']} 概率分别为 {row['prob_list']}")
    else:
        logger_instance.info("✅ 未发现多对一冲突，匹配关系为 1:1")

    return df_conflicts