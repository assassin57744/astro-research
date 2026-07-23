"""报告格式化模块：生成管线执行报告和多模式对比表。

本模块仅负责文本格式化和文件持久化，不依赖 AstroWorkflow 或 AstroDB。
"""

import logging
from pathlib import Path

import config as cfg

# =============================================================================
# 内部格式化 helper
# =============================================================================


def _format_algo_params(algo: str, gmm_config: dict) -> list[str]:
    """格式化 GMM 算法参数配置块（使用实际运行时的配置，非全局默认值）。"""
    lines = ["  [算法参数配置]"]
    if algo == "dbscan":
        lines.append(f"      - DBSCAN eps: {gmm_config.get('dbscan_eps', 'N/A')}")
        lines.append(
            f"      - DBSCAN min_samples: {gmm_config.get('dbscan_min_samples', 'N/A')}"
        )
    elif algo == "hdbscan":
        lines.append(
            f"      - HDBSCAN min_cluster_size: {gmm_config.get('hdbscan_min_cluster_size', 'N/A')}"
        )
        lines.append(
            f"      - HDBSCAN min_samples: {gmm_config.get('hdbscan_min_samples', 'N/A')}"
        )
        lines.append(
            f"      - HDBSCAN cluster_selection_epsilon: {gmm_config.get('hdbscan_cluster_selection_epsilon', 'N/A')}"
        )
    lines.append(
        f"      - GMM covariance_type: {gmm_config.get('gmm_covariance_type', 'N/A')}"
    )
    lines.append(f"      - GMM max_iter: {gmm_config.get('max_iter', 'N/A')}")
    lines.append(f"      - GMM tol: {gmm_config.get('tol', 'N/A')}")
    lines.append(
        f"      - use_experimental: {gmm_config.get('use_experimental', 'N/A')}"
    )
    lines.append(
        f"      - enable_subsampling: {gmm_config.get('enable_subsampling', 'N/A')}"
    )
    if gmm_config.get("enable_subsampling"):
        lines.append(
            f"      - subsampling_limit: {gmm_config.get('subsampling_limit', 'N/A')}"
        )
    return lines


def _format_pipeline_params(ctx_cluster: dict) -> list[str]:
    """格式化管线筛选参数块。"""
    return [
        "  [Pipeline 筛选参数]",
        f"      - PM_RADIUS: {ctx_cluster.get('PM_RADIUS', 'N/A')} mas/yr",
        f"      - PLX_ERROR: {ctx_cluster.get('PLX_ERROR', 'N/A')} mas",
        f"      - RV_ERROR: {ctx_cluster.get('RV_ERROR', 'N/A')} km/s",
        f"      - CMD_DEV: {ctx_cluster.get('CMD_DEV', 'N/A')} mag",
        f"      - KINE_SCORE_LIMIT: {ctx_cluster.get('KINE_SCORE_LIMIT', 'N/A')}",
        f"      - SEED_RADIUS: {ctx_cluster.get('SEED_RADIUS', 'N/A')} pc",
        f"      - SEED_PLX_LIM: {ctx_cluster.get('SEED_PLX_LIM', 'N/A')} mas",
        f"      - SEED_MAX_MAG: {ctx_cluster.get('SEED_MAX_MAG', 'N/A')}",
        f"      - SEED_MAX_RUWE: {ctx_cluster.get('SEED_MAX_RUWE', 'N/A')}",
    ]


def _format_gmm_stats(p_stats: dict) -> list[str]:
    """格式化 GMM 算法发现阶段统计。"""
    lines = [
        "  [1] 算法发现阶段 (GMM Inference):",
        f"      - 原始输入种子数量 (Seeds): {p_stats.get('n_seeds', 0)}",
        f"      - 成员星候选总数: {p_stats.get('n_candidates', 0)}",
        f"      - 高置信金种子星: {p_stats.get('n_golden', 0)}",
        f"      - 种子集核心样本 (Core): {p_stats.get('n_seed_core', 0)}",
        f"      - 种子集噪声剔除 (Noise): {p_stats.get('n_seed_noise', 0)}",
    ]
    # 种子管线各级过滤统计
    if p_stats.get("raw_count"):
        lines.append(f"      - 种子星 (aln视图原始): {p_stats['raw_count']}")
    if p_stats.get("clean_count"):
        lines.append(f"      - 种子星 (NaN清洗后):   {p_stats['clean_count']}")
    if p_stats.get("refined_count"):
        lines.append(f"      - 种子星 (DBSCAN精炼):  {p_stats['refined_count']}")
    return lines


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
    pass_label: str = "New Discovery",
    pass_rate_label: str = "发现准确率",
) -> list[str]:
    """格式化单个深度审计结果段（统计摘要 + 列联表）。

    Args:
        pass_label: 物理验证通过行的标签，PG Only 用 "New Discovery"，
                    参考星审计 (Ref Only / Category) 用 "物理验证通过"。
        pass_rate_label: 通过率指标标签，对应 pass_label 的比率名称。

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
    pass_rate = (new_finds / total * 100) if total > 0 else 0

    lines = [
        f"  [{section_num}] 深度审计结果 - {label}:",
        f"      - 深度审计样本总数: {total} 颗",
        f"      - 双重确认成员 (物理+文献): {tp} 颗",
        f"      - 物理验证通过 ({pass_label}): {new_finds} 颗 ({pass_rate_label}: {pass_rate:.2f}%)",
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
    deep_stats_category: dict | None = None,
) -> dict:
    """从各阶段结果字典中提取统一的绩效摘要。

    供多模式汇总对比使用，避免与 render_final_report 中的统计提取重复。
    """
    a_stats = audit_res.get("stats", {})
    matched = a_stats.get("Matched", 0)
    ref_missed = a_stats.get("Ref Only", 0)
    ref_total = ref_missed + matched
    p_stats = v_all_audit_data.get("stats", {})

    summary = {
        "cluster": target_cluster_id,
        "mode": mode.upper(),
        "algo": algo.upper(),
        "candidates": p_stats.get("n_candidates", 0),
        "golden": p_stats.get("n_golden", 0),
        "seed_core": p_stats.get("n_seed_core", 0),
        "seeds_raw": p_stats.get("raw_count", 0),
        "seeds_clean": p_stats.get("clean_count", 0),
        "seeds_refined": p_stats.get("refined_count", 0),
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

    if deep_stats_category:
        summary["deep_stats_category"] = deep_stats_category

    return summary


def render_final_report(
    target_cluster_id: str,
    target_category: str,
    mode: str,
    algo: str,
    ctx_cluster: dict,
    gmm_config: dict,
    v_all_audit_data: dict,
    audit_res: dict,
    deep_stats_pg: dict,
    deep_stats_ref: dict,
    deep_stats_matched: dict,
    deep_stats_category: dict,
    deep_stats_pg_algo: dict,
    logger: logging.Logger,
) -> dict:
    """构建、打印并持久化管线最终执行报告。

    Args:
        gmm_config: 实际运行时使用的 GMM 配置（含 dim_mode 等运行时覆盖）。
            ── 注意：feature_map 也必须从中读取，确保与 ctx.state.required_features 一致。

    Returns:
        build_summary() 产出的绩效摘要 dict。
    """
    used_features = gmm_config["feature_map"].get(mode, [])

    report_lines = [
        "=" * 65,
        f"🏁 [管线执行最终报告 - {target_cluster_id}]",
        f"  🔹 目标星团: {target_cluster_id} ({target_cluster_id})",
        f"  🔹 执行模式: {mode.upper()} -> 物理特征空间: {used_features}",
        f"  🔹 聚类算法: {algo.upper()}",
        f"  🔹 审计参考: {target_category}",
        "-" * 65,
    ]

    # 算法参数 + 管线筛选参数
    report_lines += _format_algo_params(algo, gmm_config)
    report_lines.append("-" * 65)
    report_lines += _format_pipeline_params(ctx_cluster)
    report_lines.append("-" * 65)

    # GMM 发现阶段统计
    report_lines += _format_gmm_stats(v_all_audit_data.get("stats", {}))

    # 交叉比对统计
    report_lines += _format_cross_match_stats(audit_res.get("stats", {}))

    # ── 统一基准：参考星表总数 (Matched + Ref Only) ──
    _matched = audit_res.get("stats", {}).get("Matched", 0)
    _ref_only = audit_res.get("stats", {}).get("Ref Only", 0)
    _pg_only = audit_res.get("stats", {}).get("PG Only", 0)
    _ref_total = _matched + _ref_only

    def _add_coverage(lines, deep_stats, subset_label, subset_total):
        """在判别矩阵之前插入覆盖率与双重确认覆盖率两行。"""
        total = sum(deep_stats.values()) if deep_stats else 0
        tp = deep_stats.get("Confirmed Member", 0) if deep_stats else 0
        sub_cov = (total / subset_total * 100) if subset_total > 0 else 0
        ref_cov = (total / _ref_total * 100) if _ref_total > 0 else 0
        dual_cov = (tp / _ref_total * 100) if _ref_total > 0 else 0
        lines.insert(3, f"      - 双重确认覆盖率: {tp}/{_ref_total} ({dual_cov:.2f}%)")
        lines.insert(3, f"      - {subset_label}可审率: {total}/{subset_total} ({sub_cov:.2f}%)  |  占参考星全量: {total}/{_ref_total} ({ref_cov:.2f}%)")

    # ── [3] PG Only ──
    pg_lines = list(_format_deep_audit_section(
        3, "PG Only (算法独有候选)", "文献证实 (+)", deep_stats_pg,
    ))
    if pg_lines and deep_stats_pg and _pg_only > 0:
        _add_coverage(pg_lines, deep_stats_pg, "PG Only ", _pg_only)
    report_lines += pg_lines

    # ── [4] Ref Only ──
    ref_lines = list(_format_deep_audit_section(
        4, "Ref Only (文献独有候选)", "文献一致 (+)", deep_stats_ref,
        pass_label="物理验证通过", pass_rate_label="物理通过率",
    ))
    if ref_lines and deep_stats_ref and _ref_only > 0:
        _add_coverage(ref_lines, deep_stats_ref, "Ref Only", _ref_only)
    report_lines += ref_lines

    # ── [5] Matched ──
    matched_lines = list(_format_deep_audit_section(
        5, "Matched (双方共识成员)", "文献证实 (+)", deep_stats_matched,
    ))
    if matched_lines and deep_stats_matched and _matched > 0:
        _add_coverage(matched_lines, deep_stats_matched, "共识成员", _matched)
    report_lines += matched_lines

    # ── [6] Category 全量参考星 (Matched + Ref Only) ──
    cat_lines = list(_format_deep_audit_section(
        6, f"{target_category} 全量参考星 (Matched+Ref Only)", "文献一致 (+)", deep_stats_category,
        pass_label="物理验证通过", pass_rate_label="物理通过率",
    ))
    if cat_lines and deep_stats_category and _ref_total > 0:
        cat_audited = sum(deep_stats_category.values())
        tp_cat = deep_stats_category.get("Confirmed Member", 0)
        cov = (cat_audited / _ref_total * 100) if _ref_total > 0 else 0
        dual_cov = (tp_cat / _ref_total * 100) if _ref_total > 0 else 0
        cat_lines.insert(3, f"      - 双重确认覆盖率: {tp_cat}/{_ref_total} ({dual_cov:.2f}%)")
        cat_lines.insert(3, f"      - 参考星覆盖率: {cat_audited}/{_ref_total} ({cov:.2f}%)")
    report_lines += cat_lines

    # ── [7] PG Algo 全量 (Matched + PG Only) ──
    pg_algo_lines = list(_format_deep_audit_section(
        7, "PG 算法发现全量 (Matched+PG Only)", "文献证实 (+)", deep_stats_pg_algo,
    ))
    if pg_algo_lines and deep_stats_pg_algo and _ref_total > 0:
        pg_algo_audited = sum(deep_stats_pg_algo.values())
        tp_pg_algo = deep_stats_pg_algo.get("Confirmed Member", 0)
        cov = (pg_algo_audited / _ref_total * 100) if _ref_total > 0 else 0
        dual_cov = (tp_pg_algo / _ref_total * 100) if _ref_total > 0 else 0
        pg_algo_lines.insert(3, f"      - 双重确认覆盖率: {tp_pg_algo}/{_ref_total} ({dual_cov:.2f}%)")
        pg_algo_lines.insert(3, f"      - 算法候选覆盖率: {pg_algo_audited}/{_ref_total} ({cov:.2f}%)")
    report_lines += pg_algo_lines

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
        target_cluster_id, mode, algo, v_all_audit_data, audit_res, deep_stats_pg,
        deep_stats_category=deep_stats_category,
    )


def render_all_modes_comparison(
    all_results: list[dict],
    logger: logging.Logger,
) -> None:
    """打印多模式算法运行的汇总对照报告。"""
    if not all_results:
        return

    all_results.sort(key=lambda x: x["mode"])

    logger.info("═" * 158)
    logger.info(
        f" 🏆 [全模式算法绩效汇总对照表] - 目标星团: {all_results[0]['cluster']}"
    )
    logger.info("-" * 158)

    header = (
        f"{'MODE':<8} | {'ALGO':<8} | {'SEEDS_RAW':<10} | {'SEEDS_CLN':<10} | {'SEEDS_REF':<10} | "
        f"{'CANDIDATES':<12} | {'GOLDEN':<10} | {'MATCHED':<10} | "
        f"{'PG ONLY':<10} | {'RECALL':<12} | {'NEW DISCOVERY':<15} | {'PRECISION':<10}"
    )
    logger.info(header)
    logger.info("-" * 158)

    for res in all_results:
        line = (
            f"{res['mode']:<8} | {res['algo']:<8} | {res['seeds_raw']:<10} | {res['seeds_clean']:<10} | "
            f"{res['seeds_refined']:<10} | "
            f"{res['candidates']:<12} | {res['golden']:<10} | "
            f"{res['matched']:<10} | {res['pg_only']:<10} | {res['recall']:>10.2f}% | "
            f"{res['new_finds']:<15} | {res['precision']:>9.2f}%"
        )
        logger.info(line)

    logger.info("═" * 158)
    logger.info(
        " 💡 注: RECALL 基于文献已知成员的找回率; PRECISION 基于算法独有源通过物理深度审计的比例。\n"
    )
