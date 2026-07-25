"""实验结果记录模块：将 DBSCAN 管线的中间结果写入 CSV 记录表。

以 (星团, 类型, eps, min_samples) 四元组为键，支持不同参数组合独立记录，
互不覆盖。完全独立于主干管线，异常不影响主流程。
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# CSV 中每个星团的交叉比对子行标签
_CROSS_MATCH_LABELS = ["matched", "pg only", "ref only"]


# ── 辅助 ──

def _pad_row(row: list[str], max_col: int) -> None:
    """补齐行到至少 max_col+1 个单元格。"""
    while len(row) <= max_col:
        row.append("")


# ── 参数解析 ──

def _resolve_param(
    algo_key: str,
    cluster_key: str,
    gmm_key: str,
    algo_params: dict,
    cluster_cfg: dict,
    gmm_cfg: dict,
) -> str:
    """按优先级解析参数值。

    优先级: algo_params[algo_key] > cluster_cfg[cluster_key] > gmm_cfg[gmm_key]
    """
    if algo_key in algo_params:
        return str(algo_params[algo_key])
    if cluster_key in cluster_cfg:
        return str(cluster_cfg[cluster_key])
    if gmm_key in gmm_cfg:
        return str(gmm_cfg[gmm_key])
    return ""


def resolve_eps(algo_params: dict, cluster_cfg: dict, gmm_cfg: dict) -> str:
    return _resolve_param("eps", "DBSCAN_EPS", "DBSCAN_EPS",
                          algo_params, cluster_cfg, gmm_cfg)


def resolve_min_samples(algo_params: dict, cluster_cfg: dict, gmm_cfg: dict) -> str:
    return _resolve_param("min_samples", "DBSCAN_MIN_SAMPLES", "DBSCAN_MIN_SAMPLES",
                          algo_params, cluster_cfg, gmm_cfg)


# ── 公开 API ──

def update_dbscan_csv(
    csv_path: str | Path,
    cluster: str,
    eps: str,
    min_samples: str,
    cross_stats: dict,
    deep_stats_matched: dict,
    deep_stats_pg_only: dict,
    deep_stats_ref_only: dict,
) -> None:
    """更新 DBSCAN 实验结果 CSV。

    以 (星团, 类型, eps, min_samples) 为键：
      1. 精确匹配 → 更新结果列
      2. 空模板行（星团+类型匹配，eps/min_samples 为空）→ 填充
      3. 都不匹配 → 追加新行

    Args:
        csv_path: CSV 文件完整路径。
        cluster: 星团标识，如 "M45"。
        eps: 本次使用的 eps 值（字符串）。
        min_samples: 本次使用的 min_samples 值（字符串）。
        cross_stats: {"Matched": N, "PG Only": N, "Ref Only": N}。
        deep_stats_*: 各子集深度审计统计。
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.warning(f"⚠️ [ResultLog] CSV 不存在: {csv_path}，跳过。")
        return

    try:
        # ── 读取 ──
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows: list[list[str]] = list(reader)

        if len(rows) < 1:
            logger.warning("⚠️ [ResultLog] CSV 为空，跳过。")
            return

        header = rows[0]

        # 定位列索引
        try:
            col_type = header.index("类型")
            col_eps = header.index("eps")
            col_ms = header.index("min_samples")
            col_cross = header.index("交叉比对结果")
            col_phys = header.index("通过物理验证个数")
        except ValueError as e:
            logger.warning(f"⚠️ [ResultLog] CSV 缺少预期列头: {e}，跳过。")
            return

        max_col = max(col_type, col_eps, col_ms, col_cross, col_phys)

        # ── 计算物理验证通过数 ──
        def _phys_pass(ds: dict) -> int:
            return ds.get("Confirmed Member", 0) + ds.get("New Candidate", 0)

        phys_map = {
            "matched": _phys_pass(deep_stats_matched),
            "pg only": _phys_pass(deep_stats_pg_only),
            "ref only": _phys_pass(deep_stats_ref_only),
        }

        # ── 逐类型处理 ──
        updated = False

        for label in _CROSS_MATCH_LABELS:
            cross_val = {
                "matched": cross_stats.get("Matched", ""),
                "pg only": cross_stats.get("PG Only", ""),
                "ref only": cross_stats.get("Ref Only", ""),
            }[label]
            phys_val = phys_map[label]

            # Pass 1: 精确匹配 (星团, 类型, eps, min_samples)
            exact_row = None
            for i, row in enumerate(rows[1:], start=1):
                _pad_row(row, max_col)
                rc = (row[0] or "").strip()
                rt = (row[col_type] or "").strip().lower()
                re = (row[col_eps] or "").strip()
                rm = (row[col_ms] or "").strip()
                if (rc.upper() == cluster.upper() and rt == label
                        and re == eps and rm == min_samples):
                    exact_row = i
                    break

            if exact_row is not None:
                rows[exact_row][col_cross] = str(cross_val)
                rows[exact_row][col_phys] = str(phys_val)
                updated = True
                continue

            # Pass 2: 空模板行 (星团+类型匹配, eps/min_samples 为空)
            blank_row = None
            for i, row in enumerate(rows[1:], start=1):
                _pad_row(row, max_col)
                rc = (row[0] or "").strip()
                rt = (row[col_type] or "").strip().lower()
                re = (row[col_eps] or "").strip()
                rm = (row[col_ms] or "").strip()
                if (rc.upper() == cluster.upper() and rt == label
                        and re == "" and rm == ""):
                    blank_row = i
                    break

            if blank_row is not None:
                rows[blank_row][col_eps] = eps
                rows[blank_row][col_ms] = min_samples
                rows[blank_row][col_cross] = str(cross_val)
                rows[blank_row][col_phys] = str(phys_val)
                updated = True
                continue

            # Pass 3: 追加新行
            new_row = [""] * (max_col + 1)
            new_row[0] = cluster
            new_row[col_type] = label
            new_row[col_eps] = eps
            new_row[col_ms] = min_samples
            new_row[col_cross] = str(cross_val)
            new_row[col_phys] = str(phys_val)
            rows.append(new_row)
            updated = True

        if not updated:
            logger.warning(
                f"⚠️ [ResultLog] 未找到星团 {cluster} 的可更新行，跳过。"
            )
            return

        # ── 写回 ──
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        logger.info(
            f"📝 [ResultLog] 已更新 {csv_path.name}: "
            f"cluster={cluster}, eps={eps}, min_samples={min_samples}, "
            f"Matched={cross_stats.get('Matched','')}, "
            f"PG={cross_stats.get('PG Only','')}, "
            f"Ref={cross_stats.get('Ref Only','')}"
        )

    except Exception:
        logger.warning(
            "⚠️ [ResultLog] 更新 CSV 时发生异常（已静默，不影响主流程）",
            exc_info=True,
        )
