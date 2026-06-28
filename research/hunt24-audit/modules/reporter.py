"""报告格式化模块：生成管线执行报告和多模式对比表。

本模块仅负责文本格式化和文件持久化，不依赖 AstroWorkflow 或 AstroDB。
"""

import logging
from pathlib import Path

import config as cfg

# =============================================================================
# 内部格式化 helper
# =============================================================================


def _format_algo_params(algo: str) -> list[str]:
    """格式化 GMM 算法参数配置块。"""
    lines = ["  [算法参数配置]"]
    if algo == "dbscan":
        lines.append(f"      - DBSCAN eps: {cfg.GMM_CONFIG.get('dbscan_eps', 'N/A')}")
        lines.append(
            f"      - DBSCAN min_samples: {cfg.GMM_CONFIG.get('dbscan_min_samples', 'N/A')}"
        )
    elif algo == "hdbscan":
        lines.append(
            f"      - HDBSCAN min_cluster_size: {cfg.GMM_CONFIG.get('hdbscan_min_cluster_size', 'N/A')}"
        )
        lines.append(
            f"      - HDBSCAN min_samples: {cfg.GMM_CONFIG.get('hdbscan_min_samples', 'N/A')}"
        )
        lines.append(
            f"      - HDBSCAN cluster_selection_epsilon: {cfg.GMM_CONFIG.get('hdbscan_cluster_selection_epsilon', 'N/A')}"
        )
    lines.append(
        f"      - GMM covariance_type: {cfg.GMM_CONFIG.get('gmm_covariance_type', 'N/A')}"
    )
    lines.append(f"      - GMM max_iter: {cfg.GMM_CONFIG.get('max_iter', 'N/A')}")
    lines.append(f"      - GMM tol: {cfg.GMM_CONFIG.get('tol', 'N/A')}")
    lines.append(
        f"      - use_experimental: {cfg.GMM_CONFIG.get('use_experimental', 'N/A')}"
    )
    lines.append(
        f"      - enable_subsampling: {cfg.GMM_CONFIG.get('enable_subsampling', 'N/A')}"
    )
    if cfg.GMM_CONFIG.get("enable_subsampling"):
        lines.append(
            f"      - subsampling_limit: {cfg.GMM_CONFIG.get('subsampling_limit', 'N/A')}"
        )
    return lines


def _format_pipeline_params(ctx_cluster) -> list[str]:
    """格式化管线筛选参数块。"""
    return [
        "  [Pipeline 筛选参数]",
        f"      - PM_RADIUS: {getattr(ctx_cluster, 'PM_RADIUS', 'N/A')} mas/yr",
        f"      - PLX_ERROR: {getattr(ctx_cluster, 'PLX_ERROR', 'N/A')} mas",
        f"      - RV_ERROR: {getattr(ctx_cluster, 'RV_ERROR', 'N/A')} km/s",
        f"      - CMD_DEV: {getattr(ctx_cluster, 'CMD_DEV', 'N/A')} mag",
        f"      - KINE_SCORE_LIMIT: {getattr(ctx_cluster, 'KINE_SCORE_LIMIT', 'N/A')}",
        f"      - SEED_RADIUS: {getattr(ctx_cluster, 'SEED_RADIUS', 'N/A')} pc",
        f"      - SEED_PLX_LIM: {getattr(ctx_cluster, 'SEED_PLX_LIM', 'N/A')} mas",
        f"      - SEED_MAX_MAG: {getattr(ctx_cluster, 'SEED_MAX_MAG', 'N/A')}",
        f"      - SEED_MAX_RUWE: {getattr(ctx_cluster, 'SEED_MAX_RUWE', 'N/A')}",
    ]


def _format_gmm_stats(p_stats: dict) -> list[str]:
    """格式化 GMM 算法发现阶段统计。"""
    return [
        "  [1] 算法发现阶段 (GMM Inference):",
        f"      - 原始输入种子数量 (Seeds): {p_stats.get('n_seeds', 0)}",
        f"      - 成员星候选总数: {p_stats.get('n_candidates', 0)}",
        f"      - 高置信金种子星: {p_stats.get('n_golden', 0)}",
        f"      - 种子集核心样本 (Core): {p_stats.get('n_seed_core', 0)}",
        f"      - 种子集噪声剔除 (Noise): {p_stats.get('n_seed_noise', 0)}",
    ]


def _format_cross_match_stats(a_stats: dict) -> list[str]:
    """格式化交叉比对统计块。"""
    ref_missed = a_stats.get("Ref Only", 0)
    matched = a_stats.get("Matched", 0)
    ref_total = ref_missed + matched
    miss_rate = (ref_missed / ref_total * 100) if ref_total > 0 else 0

    return [
        "  [2] 审计目标交叉比对 (Cross Match):",
        f"      - 双方共识成员 (Matched):   {matched} 颗",
        f"      - 算法漏检 (Recall/Missed): {ref_missed} 颗 (漏检率: {miss_rate:.2f}%)",
        f"      - 算法独有候选 (PG Only):   {a_stats.get('PG Only', 0)}",
    ]


def _format_contingency_matrix(
    label: str,
    lit_pos_label: str,
    lit_neg_label: str,
    deep_stats: dict,
) -> list[str]:
    """格式化单个 2×2 审计判别矩阵。

    Args:
        label: 矩阵标题标签 (如 "PG Only (算法独有候选)")
        lit_pos_label: 文献正面标签 (如 "文献证实 (+)")
        lit_neg_label: 文献负面标签 (如 "文献缺失 (-)")
        deep_stats: audit_status → count 映射
    """
    tp = deep_stats.get("Confirmed Member", 0)
    fp = deep_stats.get("New Candidate", 0)
    fn = deep_stats.get("Literature Only", 0)
    tn = deep_stats.get("Contamination", 0)
    total = tp + fp + fn + tn

    return [
        "      " + "-" * 72,
        f"      {label} 审计判别矩阵 (物理检查 vs 文献共识):",
        "      " + "-" * 72,
        f"      {'':<18} | {lit_pos_label:<12} | {lit_neg_label:<12} | 物理汇总",
        f"      {'物理符合 (+)':<14} | {tp:<16} | {fp:<16} | {tp + fp}",
        f"      {'物理偏离 (-)':<14} | {fn:<16} | {tn:<16} | {fn + tn}",
        "      " + "-" * 72,
        f"      {'文献汇总':<14} | {tp + fn:<16} | {fp + tn:<16} | {total}",
        "      " + "-" * 72,
    ]


def _format_deep_audit_section(
    section_num: int,
    label: str,
    lit_pos_label: str,
    deep_stats: dict,
) -> list[str]:
    """格式化单个深度审计结果段（统计摘要 + 列联表）。

    Returns:
        格式化行列表；若 deep_stats 为空则返回空列表。
    """
    if not deep_stats:
        return []

    total = sum(deep_stats.values())
    tp = deep_stats.get("Confirmed Member", 0)
    fp = deep_stats.get("New Candidate", 0)
    fn = deep_stats.get("Literature Only", 0)
    tn = deep_stats.get("Contamination", 0)
    new_finds = tp + fp
    precision = (new_finds / total * 100) if total > 0 else 0

    lines = [
        f"  [{section_num}] 深度审计结果 - {label}:",
        f"      - 深度审计样本总数: {total} 颗",
        f"      - 双重确认成员 (物理+文献): {tp} 颗",
        f"      - 物理验证通过 (New Discovery): {new_finds} 颗 (发现准确率: {precision:.2f}%)",
        f"      - 确认为背景噪点 (Contamination): {tn}",
        f"      - 仅文献收录 (Lit. Only):       {fn}",
    ]
    lines += _format_contingency_matrix(label, lit_pos_label, "文献缺失 (-)", deep_stats)
    return lines


# =============================================================================
# 公开 API
# =============================================================================


def build_summary(
    target_cluster_id: str,
    mode: str,
    algo: str,
    v_all_audit_data: dict,
    audit_res: dict,
    deep_stats_pg: dict,
) -> dict:
    """从各阶段结果字典中提取统一的绩效摘要。

    供多模式汇总对比使用，避免与 render_final_report 中的统计提取重复。
    """
    a_stats = audit_res.get("stats", {})
    matched = a_stats.get("Matched", 0)
    ref_missed = a_stats.get("Ref Only", 0)
    ref_total = ref_missed + matched

    summary = {
        "cluster": target_cluster_id,
        "mode": mode.upper(),
        "algo": algo.upper(),
        "candidates": v_all_audit_data.get("stats", {}).get("n_candidates", 0),
        "golden": v_all_audit_data.get("stats", {}).get("n_golden", 0),
        "seed_core": v_all_audit_data.get("stats", {}).get("n_seed_core", 0),
        "matched": matched,
        "pg_only": a_stats.get("PG Only", 0),
        "ref_only": a_stats.get("Ref Only", 0),
        "recall": (matched / ref_total * 100) if ref_total > 0 else 0,
        "new_finds": 0,
        "precision": 0.0,
    }

    if deep_stats_pg:
        total_audited_pg = sum(deep_stats_pg.values())
        new_finds_pg = deep_stats_pg.get("Confirmed Member", 0) + deep_stats_pg.get(
            "New Candidate", 0
        )
        summary["new_finds"] = new_finds_pg
        summary["precision"] = (
            (new_finds_pg / total_audited_pg * 100) if total_audited_pg > 0 else 0
        )

    return summary


def render_final_report(
    target_cluster_id: str,
    target_category: str,
    mode: str,
    algo: str,
    ctx_cluster,
    v_all_audit_data: dict,
    audit_res: dict,
    deep_stats_pg: dict,
    deep_stats_ref: dict,
    logger: logging.Logger,
) -> dict:
    """构建、打印并持久化管线最终执行报告。

    Returns:
        build_summary() 产出的绩效摘要 dict。
    """
    used_features = cfg.GMM_CONFIG["feature_map"].get(mode, [])

    report_lines = [
        "=" * 65,
        f"🏁 [管线执行最终报告 - {target_cluster_id}]",
        f"  🔹 目标星团: {target_cluster_id} ({ctx_cluster.NAME})",
        f"  🔹 执行模式: {mode.upper()} -> 物理特征空间: {used_features}",
        f"  🔹 聚类算法: {algo.upper()}",
        f"  🔹 审计参考: {target_category}",
        "-" * 65,
    ]

    # 算法参数 + 管线筛选参数
    report_lines += _format_algo_params(algo)
    report_lines.append("-" * 65)
    report_lines += _format_pipeline_params(ctx_cluster)
    report_lines.append("-" * 65)

    # GMM 发现阶段统计
    report_lines += _format_gmm_stats(v_all_audit_data.get("stats", {}))

    # 交叉比对统计
    report_lines += _format_cross_match_stats(audit_res.get("stats", {}))

    # PG Only 深度审计
    report_lines += _format_deep_audit_section(
        3, "PG Only (算法独有候选)", "文献证实 (+)", deep_stats_pg
    )

    # Ref Only 深度审计
    report_lines += _format_deep_audit_section(
        4, "Ref Only (文献独有候选)", "文献一致 (+)", deep_stats_ref
    )

    # 页脚
    report_lines.append("-" * 65)
    report_lines.append(f"  ✅ 任务状态: 成功完成 | 资产导出路径: {cfg.RESULTS_DIR}")
    report_lines.append("=" * 65)

    # 输出到日志
    for line in report_lines:
        logger.info(line)

    # 持久化到文件
    export_base = cfg.TMPL.FILE_EXPORT_BASE.format(
        cluster=target_cluster_id, category=target_category, mode=mode, algo=algo
    )
    report_path = cfg.RESULTS_DIR / cfg.TMPL.FILE_FINAL_REPORT.format(base=export_base)
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info(f"📄 最终报告副本已保存至: {report_path}")
    except Exception as e:
        logger.error(f"❌ 无法保存最终报告副本: {e}")

    return build_summary(
        target_cluster_id, mode, algo, v_all_audit_data, audit_res, deep_stats_pg
    )


def render_all_modes_comparison(
    all_results: list[dict],
    logger: logging.Logger,
) -> None:
    """打印多模式算法运行的汇总对照报告。"""
    if not all_results:
        return

    all_results.sort(key=lambda x: x["mode"])

    logger.info("\n" + "═" * 125)
    logger.info(
        f" 🏆 [全模式算法绩效汇总对照表] - 目标星团: {all_results[0]['cluster']}"
    )
    logger.info("-" * 125)

    header = (
        f"{'MODE':<8} | {'ALGO':<8} | {'CANDIDATES':<12} | {'CORE':<6} | {'GOLDEN':<10} | {'MATCHED':<10} | "
        f"{'PG ONLY':<10} | {'RECALL':<12} | {'NEW DISCOVERY':<15} | {'PRECISION'}"
    )
    logger.info(header)
    logger.info("-" * 125)

    for res in all_results:
        line = (
            f"{res['mode']:<8} | {res['algo']:<8} | {res['candidates']:<12} | {res['seed_core']:<6} | "
            f"{res['golden']:<10} | "
            f"{res['matched']:<10} | {res['pg_only']:<10} | {res['recall']:>10.2f}% | "
            f"{res['new_finds']:<15} | {res['precision']:>9.2f}%"
        )
        logger.info(line)

    logger.info("═" * 125)
    logger.info(
        " 💡 注: RECALL 基于文献已知成员的找回率; PRECISION 基于算法独有源通过物理深度审计的比例。\n"
    )
