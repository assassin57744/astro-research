"""星团成员审计管线入口。

编排 Gaia DR3 数据导入、GMM 成员识别、多源文献交叉审计的端到端流程。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.astro_db import AstroDB, AssetManager
from modules.astro_workflow import AstroWorkflow
from modules.reporter import render_final_report, render_all_modes_comparison

import config as cfg

logger = logging.getLogger("AstroPipeline")


# =============================================================================
# 日志初始化
# =============================================================================


def setup_logging(level: int = logging.INFO) -> None:
    """初始化双通道（文件 + 控制台）日志系统。"""
    log_dir = Path(cfg.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / datetime.now().strftime("astro_run_%Y%m%d.log")

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname).4s] %(filename)18s:%(lineno)4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"📝 日志系统就绪。文件存放在: {log_path}")


# =============================================================================
# CLI 参数解析
# =============================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Astro Research Pipeline - 天文数据分析与自动审计命令行工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "cluster",
        type=str,
        nargs="?",
        help="目标星团名称 (运行管线模式下必填, 例如: M45, M44, M67)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="hunt",
        choices=["cg20", "heyl", "zerj", "risb", "hunt"],
        help="审计对比的参考星表类别",
    )
    valid_modes = list(cfg.GMM_CONFIG["feature_map"].keys())
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        choices=valid_modes + ["all"],
        help="GMM 算法的特征维度模式，使用 'all' 将循环执行所有模式",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=cfg.GMM_CONFIG.get("cluster_algo", "dbscan"),
        choices=["dbscan", "hdbscan"],
        help="种子集预处理使用的聚类算法",
    )
    parser.add_argument(
        "--result",
        type=str,
        default="brief",
        choices=["brief", "detailed"],
        help="结果产出等级: brief (精简, 仅日志及轻量报告) 或 detailed (详细, 导出全量 CSV 资产)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="控制台日志输出级别",
    )

    maint_group = parser.add_argument_group("资产维护命令 (Maintenance)")
    maint_group.add_argument(
        "--backup", action="store_true", help="手动备份 data/raw 目录"
    )
    maint_group.add_argument(
        "--restore",
        type=str,
        nargs="?",
        const="A",
        choices=["A", "B"],
        default=None,
        help="从备份中恢复数据 (A 或 B, 默认 A)",
    )
    maint_group.add_argument(
        "--query-backup", action="store_true", help="查询当前备份库的状态"
    )

    return parser.parse_args()


# =============================================================================
# 维护命令处理
# =============================================================================


def handle_maintenance(args: argparse.Namespace) -> bool:
    """处理资产维护子命令。返回 True 表示已执行维护操作。"""
    if args.query_backup:
        AssetManager().query_backup_assets()
        return True

    if args.backup:
        AssetManager().manage_backup_assets()
        return True

    if args.restore is not None:
        AssetManager().restore_backup_assets(target=args.restore)
        return True

    return False


# =============================================================================
# 参数验证与上下文准备
# =============================================================================


def _validate_cluster(cluster_str: str | None) -> str:
    """校验并标准化星团名称。失败时直接退出进程。"""
    if not cluster_str:
        sys.exit("错误: 运行管线模式必须提供 cluster 参数。使用 --help 查看维护命令。")

    cluster_input = cluster_str.upper()
    cluster_map = {k.upper(): k for k in cfg.CLUSTERS.keys()}

    if cluster_input not in cluster_map:
        logger.error(
            f"❌ 未知的星团名称: '{cluster_str}'。可选范围: {list(cfg.CLUSTERS.keys())}"
        )
        sys.exit(1)

    return cluster_map[cluster_input]


def _prepare_context(cluster_id: str, category: str) -> tuple[dict, dict, list[str]]:
    """准备星团上下文、数据清单和待处理的参考表列表。

    Returns:
        (ctx_cluster, data_manifest, ref_tables)
    """
    ctx_cluster = cfg.CLUSTERS[cluster_id].copy()
    ctx_cluster["id"] = cluster_id
    data_manifest = cfg.MANIFEST

    ref_tables = [
        ctx_cluster["FIELD_IDX"],
        ctx_cluster["SEED_IDX"],
        category,
        cfg.IDX_DR2IDX,
        cfg.IDX_IDS_SIMBAD,
    ]

    kernel_type = (
        "PriorGMMEx (Experimental)"
        if cfg.GMM_CONFIG.get("use_experimental")
        else "PriorGMM (Standard)"
    )
    logger.info(
        f"🚀 初始化管道组件完成。星团: {cluster_id}, "
        f"内核: {kernel_type}, 算法: {cfg.GMM_CONFIG.get('cluster_algo', 'dbscan').upper()}, "
        f"种子处理: {cfg.GMM_CONFIG.get('cluster_algo', 'dbscan')}, "
        f"审计对象: {category}"
    )

    return ctx_cluster, data_manifest, ref_tables


# =============================================================================
# 审计辅助
# =============================================================================


def _run_deep_audit(
    wf: AstroWorkflow, v_audit_view: str, audit_type: str
) -> tuple[str | None, dict]:
    """对单个候选视图执行深度审计。

    Args:
        wf: AstroWorkflow 实例。
        v_audit_view: 待审计的视图名。
        audit_type: 审计类型标识 ("pg_only" 或 "ref_only")。

    Returns:
        (report_view_name | None, {audit_status: count})
    """
    v_result = wf.run_audit(target=v_audit_view, audit_type=audit_type)

    if not v_result:
        return None, {}

    sql = (
        f"SELECT audit_status, count(*) FROM {v_result} "
        f"WHERE audit_status IS NOT NULL GROUP BY audit_status"
    )
    stats = dict(wf.db.con.execute(sql).fetchall())
    return v_result, stats


def _perform_cross_audit(
    wf: AstroWorkflow,
    v_all_audit_data: dict,
    data_manifest: dict,
    target_cluster_id: str,
) -> tuple[dict, str | None, str | None, dict, dict]:
    """执行交叉比对和深度审计。

    Returns:
        (audit_res, v_final_pg_audited, v_final_ref_audited, deep_stats_pg, deep_stats_ref)
    """
    logger.info(f"⚖️ [5/5] 执行多源文献交叉审计, 参考类别: {wf.target_category}")

    v_candidates = v_all_audit_data["v_candidates"]
    target_aln_view = data_manifest[wf.target_category]["aln_view"].format(
        cluster=target_cluster_id.lower()
    )

    audit_res = wf.prepare_audit_data(v_candidates, target_aln_view)
    if audit_res.get("status") != "success":
        logger.warning(f"⚠️ 交叉比对审计未完全成功: {audit_res.get('message')}")
        return audit_res, None, None, {}, {}

    v_audit_pg_only = audit_res.get("v_audit_pg_only")
    v_audit_ref_only = audit_res.get("v_audit_ref_only")

    logger.info(
        f"✅ 交叉审计完成，PG Only: {v_audit_pg_only}, Ref Only: {v_audit_ref_only}"
    )

    # PG Only 深度审计
    if not v_audit_pg_only or audit_res.get("stats", {}).get("PG Only", 0) == 0:
        logger.warning("⚠️ 未找到算法独有候选 (PG Only)，跳过 PG Only 深度审计。")
        v_final_pg, deep_stats_pg = None, {}
    else:
        logger.info(f"🔍 准备 PG Only 深度审计，目标视图: {v_audit_pg_only}")
        v_final_pg, deep_stats_pg = _run_deep_audit(wf, v_audit_pg_only, "pg_only")

    # Ref Only 深度审计
    if not v_audit_ref_only or audit_res.get("stats", {}).get("Ref Only", 0) == 0:
        logger.warning("⚠️ 未找到文献独有候选 (Ref Only)，跳过 Ref Only 深度审计。")
        v_final_ref, deep_stats_ref = None, {}
    else:
        logger.info(f"🔍 准备 Ref Only 深度审计，目标视图: {v_audit_ref_only}")
        v_final_ref, deep_stats_ref = _run_deep_audit(wf, v_audit_ref_only, "ref_only")

    return audit_res, v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref


# =============================================================================
# 结果导出
# =============================================================================


def _export_pipeline_results(
    db: AstroDB,
    wf: AstroWorkflow,
    audit_res: dict,
    v_final_pg_audited: str | None,
    v_final_ref_audited: str | None,
    target_cluster_id: str,
    target_category: str,
    mode: str,
    algo: str,
    do_export: bool = False,
) -> None:
    """将管线产出物导出为 CSV/Parquet 文件。"""
    if not do_export:
        logger.info("⏩ 跳过物理文件导出 (通过 CLI 参数禁用)。")
        return

    logger.info("💾 [Export] 正在执行耗时的数据资产导出任务...")

    export_base = cfg.TMPL.FILE_EXPORT_BASE.format(
        cluster=target_cluster_id, category=target_category, mode=mode, algo=algo
    )

    db.export_table(wf.t_master, export_dir=cfg.RESULTS_DIR)

    if v_final_pg_audited:
        db.export_table(
            v_final_pg_audited,
            filename=cfg.TMPL.FILE_DEEP_AUDIT.format(base=export_base + "_pg_only"),
            format="csv",
            export_dir=cfg.RESULTS_DIR,
        )

    if v_final_ref_audited:
        db.export_table(
            v_final_ref_audited,
            filename=cfg.TMPL.FILE_DEEP_AUDIT.format(base=export_base + "_ref_only"),
            format="csv",
            export_dir=cfg.RESULTS_DIR,
        )

    logger.info("✅ 结果导出完成。")


# =============================================================================
# 核心计算管线
# =============================================================================


def _run_single_mode(
    db: AstroDB,
    wf: AstroWorkflow,
    ctx_cluster: dict,
    data_manifest: dict,
    target_cluster_id: str,
    target_category: str,
    mode: str,
    algo: str,
    result_mode: str,
) -> dict | None:
    """执行单模式的 GMM → 后处理 → 交叉审计 → 导出 → 报告 全流程。

    假定数据导入和标准化已由调用方完成。

    Returns:
        绩效摘要 dict；失败时返回 None。
    """
    # [3/5] GMM 成员识别
    logger.info(f"🧠 [3/5] 启动 GMM 成员识别内核 (模式: {wf.mode}, 算法: {wf.algo})...")
    t_result = wf.run_pgmm(ctx_cluster)
    logger.info(f"✨ 算法推断完成，结果表: {t_result}")

    # [4/5] 后处理
    logger.info("📊 [4/5] 正在合成分析宽表并提取候选成员视图...")
    v_all = wf.post_pgmm(t_result)
    if v_all.get("status") != "success":
        logger.error(f"❌ 后处理流程失败: {v_all.get('message')}")
        return None
    logger.info("✅ 数据处理流程结束，转入交叉审计阶段。")

    # [5/5] 交叉审计
    audit_res, v_final_pg, v_final_ref, deep_stats_pg, deep_stats_ref = (
        _perform_cross_audit(wf, v_all, data_manifest, target_cluster_id)
    )

    # 导出
    _export_pipeline_results(
        db, wf, audit_res, v_final_pg, v_final_ref,
        target_cluster_id, target_category, mode, algo,
        do_export=(result_mode == "detailed"),
    )

    # 报告
    return render_final_report(
        target_cluster_id, target_category, mode, algo, ctx_cluster,
        v_all, audit_res, deep_stats_pg, deep_stats_ref, logger,
    )


# =============================================================================
# 公开入口
# =============================================================================


def run_pipeline(
    target_cluster_id: str,
    target_category: str,
    mode: str = "5d",
    algo: str = "dbscan",
    db: AstroDB = None,
    result_mode: str = "brief",
) -> dict | None:
    """驱动端到端的天文数据分析与审计管线。

    负责编排数据导入、空间对齐、GMM 算法处理、交叉验证及最终审计的完整流程。
    """
    db_provided = db is not None
    if db is None:
        db = AstroDB(manifest=cfg.MANIFEST)

    ctx_cluster, data_manifest, ref_tables = _prepare_context(
        target_cluster_id, target_category
    )
    wf = AstroWorkflow(db, target_cluster_id, target_category, mode, algo)

    try:
        # [1/5] 数据同步
        logger.info("📦 [1/5] 正在同步物理数据源 (跳过虚拟视图)...")
        db.import_raw(target_cluster_id=target_cluster_id, force=False)

        # [2/5] 数据对齐
        logger.info(
            f"📐 [2/5] 正在执行数据对齐与虚拟视图创建 "
            f"({target_cluster_id}, 模式: {mode})..."
        )
        wf.prepare_field_data(ref_tables, ctx_cluster)
        logger.info("✅ 数据准备阶段完成。")

        return _run_single_mode(
            db, wf, ctx_cluster, data_manifest,
            target_cluster_id, target_category, mode, algo, result_mode,
        )
    except Exception as e:
        logger.error(f"❌ 流水线在运行期间发生严重崩溃: {str(e)}", exc_info=True)
        raise
    finally:
        if not db_provided:
            db.close()
            logger.info("🔒 数据库连接已释放。")


def _run_all_modes(
    target_cluster_id: str,
    target_category: str,
    algo: str,
    result_mode: str,
) -> None:
    """循环所有特征空间模式，产出汇总对比报告。"""
    db = AstroDB(manifest=cfg.MANIFEST)
    valid_modes = list(cfg.GMM_CONFIG["feature_map"].keys())

    ctx_cluster, data_manifest, ref_tables = _prepare_context(
        target_cluster_id, target_category
    )

    try:
        # 一次性数据导入与标准化（所有模式共享）
        logger.info("📦 正在同步物理数据源...")
        db.import_raw(target_cluster_id=target_cluster_id, force=False)

        wf_setup = AstroWorkflow(db, target_cluster_id, target_category, valid_modes[0], algo)
        wf_setup.prepare_field_data(ref_tables, ctx_cluster)
        logger.info("✅ 数据准备阶段完成（全模式共享）。")

        all_results = []
        for mode in valid_modes:
            wf = AstroWorkflow(db, target_cluster_id, target_category, mode, algo)
            summary = _run_single_mode(
                db, wf, ctx_cluster, data_manifest,
                target_cluster_id, target_category, mode, algo, result_mode,
            )
            if summary:
                all_results.append(summary)

        render_all_modes_comparison(all_results, logger)
    finally:
        db.close()
        logger.info("🔒 数据库连接已释放。")


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    """主程序入口：参数解析 → 环境初始化 → 管线执行。"""
    args = parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=numeric_level if isinstance(numeric_level, int) else logging.INFO)

    if handle_maintenance(args):
        return

    target_cluster_id = _validate_cluster(args.cluster)

    logger.info(
        f"🚀 启动分析管道 - 目标: {target_cluster_id}, 模式: {args.mode}, 参考: {args.category}"
    )

    if args.mode == "all":
        _run_all_modes(target_cluster_id, args.category, args.algo, args.result)
    else:
        run_pipeline(
            target_cluster_id, args.category, args.mode, args.algo,
            result_mode=args.result,
        )


if __name__ == "__main__":
    main()
