import logging
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# 按照扁平化结构更新导入路径
from modules.astro_db import AstroDB
from modules.astro_workflow import AstroWorkflow

import config as cfg

# 初始化模块级记录器，确保所有函数共享同一日志上下文
logger = logging.getLogger("AstroPipeline")


class NameShortenerFilter(logging.Filter):
    """
    日志记录过滤器：用于将冗长的模块路径简化为 “文件名:行号” 格式。
    """

    def filter(self, record):
        # 按照用户要求：只打印 文件名和行号 (例如: validator.py:456)
        # record.filename 获取文件名，record.lineno 获取调用处的行号
        record.call_path = f"{record.filename}:{record.lineno}"
        
        return True


def setup_logging(log_dir=cfg.LOG_DIR, level=logging.INFO):
    """
    初始化全局日志系统，支持双通道（文件与控制台）输出。

    Args:
        log_dir (Path | str): 日志文件的存储目录。默认为配置中的 LOG_DIR。

    Returns:
        logging.Logger: 配置好的 'AstroPipeline' 根记录器实例。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = datetime.now().strftime("astro_run_%Y%m%d.log")
    log_path = log_dir / log_filename

    # 此处 logger 已在模块级定义，直接设置级别
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，捕获所有消息

    shortener = NameShortenerFilter()

    # 定义全局日志格式：时间 - 调用路径(固定22宽) - 级别 - 消息
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname).4s] %(call_path)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    fh.addFilter(shortener)
    logger.addHandler(fh)

    # 控制台处理器通常设置为 INFO，避免 DEBUG 级别的冗余信息干扰实时监控
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    ch.addFilter(shortener)
    logger.addHandler(ch)

    logger.info(f"📝 日志系统就绪。文件存放在: {log_path}")
    logger.debug("已激活名称精简过滤器以优化日志输出。")
    return logger


def _initialize_pipeline_components(
    target_cluster_id: str, target_category: str, mode: str
):
    """初始化核心管道组件和配置。"""
    cfg.GMM_CONFIG["dim_mode"] = mode
    ctx_cluster = cfg.CLUSTERS[target_cluster_id].copy()
    ctx_cluster["id"] = target_cluster_id
    data_manifest = cfg.MANIFEST

    db = AstroDB(manifest=data_manifest)
    wf = AstroWorkflow(
        db, target_cluster=target_cluster_id, target_category=target_category
    )

    ref_tables = [
        ctx_cluster["FIELD_IDX"],
        ctx_cluster["SEED_IDX"],
        cfg.IDX_DR2IDX,
        cfg.IDX_CG20,
        cfg.IDX_HEYL,
        cfg.IDX_HUNT,
        cfg.IDX_ZERJ,
        cfg.IDX_RISB,
    ]
    logger.info(
        f"🚀 初始化管道组件完成。目标星团: {target_cluster_id}, 模式: {mode}, 审计参考: {target_category}"
    )
    return db, wf, ctx_cluster, data_manifest, ref_tables


def _ingest_and_prepare_data(
    db: AstroDB,
    wf: AstroWorkflow,
    target_cluster_id: str,
    ref_tables: list,
    ctx_cluster: dict,
):
    """处理原始数据摄取、标准化和虚拟视图创建。"""
    logger.info(f"📦 [1/5] 正在同步物理数据源 (跳过虚拟视图)...")
    db.import_raw(target_cluster_id=target_cluster_id, force=False)

    logger.info(
        f"📐 [2/5] 正在执行数据对齐与虚拟视图创建 ({target_cluster_id}, 模式: {cfg.GMM_CONFIG['dim_mode']})..."
    )
    wf.prepare_field_data(ref_tables, ctx_cluster)
    logger.info("✅ 数据准备阶段完成。")


def _execute_gmm_algorithm(wf: AstroWorkflow, ctx_cluster: dict) -> str:
    """执行核心 GMM 成员识别算法。"""
    logger.info(
        f"🧠 [3/5] 启动 GMM 成员识别内核 ({cfg.GMM_CONFIG['dim_mode']} 模式)..."
    )
    t_result = wf.run_pipeline(ctx_cluster)
    logger.info(f"✨ 算法推断完成，结果表: {t_result}")
    return t_result


def _post_process_results(wf: AstroWorkflow, t_result: str) -> dict:
    """执行后处理和候选者分类。"""
    logger.info("📊 [4/5] 正在合成分析宽表并提取候选成员视图...")
    v_all_audit_data = wf.post_pipeline(t_result)
    if v_all_audit_data.get("status") != "success":
        logger.error(f"❌ 后处理流程失败: {v_all_audit_data.get('message')}")
        raise RuntimeError(f"后处理流程失败: {v_all_audit_data.get('message')}")
    logger.info("✅ 数据处理流程结束，转入交叉审计阶段。")
    return v_all_audit_data


def _perform_cross_audit(
    wf: AstroWorkflow,
    v_all_audit_data: dict,
    data_manifest: dict,
    target_cluster_id: str,
) -> tuple[dict, str, dict]:
    """执行交叉验证和新发现的深度审计。"""
    logger.info(f"⚖️ [5/5] 执行多源文献交叉审计, 参考类别: {wf.target_category}")

    v_audit_ref = v_all_audit_data["v_candidates"]  # 获取算法产出的候选成员视图名
    target_aln_view = data_manifest[wf.target_category]["aln_view"].format(
        cluster=target_cluster_id.lower()
    )
    audit_res = wf.prepare_audit_data(v_audit_ref, target_aln_view)
    if audit_res.get("status") != "success":
        logger.warning(f"⚠️ 交叉比对审计未完全成功: {audit_res.get('message')}")
        # 如果交叉比对不成功，深度审计将无法进行，返回默认值
        return audit_res, None, {}

    gmm_only_key = f"v_{cfg.IDX_GMM}_only"
    v_audit_target = audit_res.get("audit_views", {}).get(gmm_only_key)

    v_final_audited = None
    deep_stats = {}
    if not v_audit_target:
        logger.warning(f"⚠️ 未找到名为 {gmm_only_key} 的审计视图，跳过深度审计。")
    else:
        logger.info(f"🔍 准备深度审计，目标视图: {v_audit_target}")
        v_final_audited = wf.run_audit(target=v_audit_target)

        # 获取深度审计统计汇总
        if v_final_audited:
            st_sql = f"SELECT audit_status, count(*) FROM {v_final_audited} GROUP BY audit_status"
            deep_stats = dict(wf.db.con.execute(st_sql).fetchall())
    return audit_res, v_final_audited, deep_stats


def _export_pipeline_results(
    db: AstroDB,
    audit_res: dict,
    v_final_audited: str,
    target_cluster_id: str,
    target_category: str,
    mode: str,
):
    """将管道结果导出到指定目录。"""
    logger.info("💾 [Export] 启动数据资产导出模块...")

    # 构造标准化的导出文件名前缀，包含星团 ID、审计目录及执行模式
    # 生成示例: M45_hunt_3d_cross_summary.csv
    export_base = f"{target_cluster_id}_{target_category}_{mode}"

    # A. 导出交叉比对汇总宽表：包含 Gaia Source ID, 算法概率以及双方一致性分类标签
    v_cross_total = audit_res.get("audit_views", {}).get("v_cross_audit_total")
    if v_cross_total:
        db.export_table(
            v_cross_total,
            filename=f"{export_base}_cross_summary",
            format="csv",
            export_dir=cfg.RESULTS_DIR,
        )

    # B. 导出深度审计详细报告：包含物理验证状态 (audit_status)、残差指标以及 SIMBAD 文献证据
    if v_final_audited:
        db.export_table(
            v_final_audited,
            filename=f"{export_base}_deep_audit",
            format="csv",
            export_dir=cfg.RESULTS_DIR,
        )
    logger.info("✅ 结果导出完成。")


def _log_final_report(
    target_cluster_id: str,
    target_category: str,
    mode: str,
    ctx_cluster: dict,
    v_all_audit_data: dict,
    audit_res: dict,
    deep_stats: dict,
):
    """记录并保存最终的合并执行报告。"""
    used_features = cfg.GMM_CONFIG["feature_map"].get(mode, [])

    report_lines = [
        "=" * 65,
        f"🏁 [管线执行最终报告 - {target_cluster_id}]",
        f"  🔹 目标星团: {target_cluster_id} ({ctx_cluster['NAME']})",
        f"  🔹 执行模式: {mode.upper()} -> 物理特征空间: {used_features}",
        f"  🔹 审计参考: {target_category}",
        "-" * 65
    ]

    p_stats = v_all_audit_data.get("stats", {})
    report_lines.append("  [1] 算法发现阶段 (GMM Inference):")
    report_lines.append(f"      - 成员星候选总数: {p_stats.get('n_candidates', 0)}")
    report_lines.append(f"      - 高置信金种子星: {p_stats.get('n_golden', 0)}")

    a_stats = audit_res.get("stats", {}).get("cross_audit_counts", {})
    ref_missed = a_stats.get(f"{target_category}_only", 0)
    matched = a_stats.get("matched", 0)
    ref_total = ref_missed + matched
    miss_rate = (ref_missed / ref_total * 100) if ref_total > 0 else 0

    report_lines.append("  [2] 文献交叉比对 (Cross Match):")
    report_lines.append(f"      - 双方共识成员 (Matched):   {matched} 颗")
    report_lines.append(f"      - 算法漏检 (Recall/Missed): {ref_missed} 颗 (漏检率: {miss_rate:.2f}%)")
    report_lines.append(f"      - 算法独有候选 (PG Only):   {a_stats.get('pgmm_only', 0)}")

    if deep_stats:
        total_audited = sum(deep_stats.values())
        confirmed_both = deep_stats.get("Confirmed Member", 0)
        new_pure_candidates = deep_stats.get("New Candidate", 0)
        new_finds = confirmed_both + new_pure_candidates
        discovery_precision = (
            (new_finds / total_audited * 100) if total_audited > 0 else 0
        )

        report_lines.append("  [3] 深度审计结果 (Deep Validation):")
        report_lines.append(f"      - 深度审计样本总数: {total_audited} 颗")
        report_lines.append(f"      - 双重确认成员 (物理+文献): {confirmed_both} 颗")
        report_lines.append(f"      - 物理验证通过 (New Discovery): {new_finds} 颗 (发现准确率: {discovery_precision:.2f}%)")
        report_lines.append(f"      - 确认为背景噪点 (Contamination): {deep_stats.get('Contamination', 0)}")
        report_lines.append(f"      - 仅文献收录 (Lit. Only):       {deep_stats.get('Literature Only', 0)}")

    report_lines.append("-" * 65)
    report_lines.append(f"  ✅ 任务状态: 成功完成 | 资产导出路径: {cfg.RESULTS_DIR}")
    report_lines.append("=" * 65)

    # 1. 打印到日志系统
    for line in report_lines:
        logger.info(line)

    # 2. 保存到分析结果目录
    export_base = f"{target_cluster_id}_{target_category}_{mode}"
    report_path = cfg.RESULTS_DIR / f"{export_base}_final_report.txt"

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info(f"📄 最终报告副本已保存至: {report_path}")
    except Exception as e:
        logger.error(f"❌ 无法保存最终报告副本: {e}")


def run_pipeline(target_cluster_id: str, target_category: str, mode: str):
    """
    驱动端到端的天文数据分析与审计管线。

    该函数负责编排数据导入、空间对齐、GMM 算法处理、交叉验证及最终审计的完整流程。

    Args:
        target_cluster_id (str): 目标星团标识符（例如 'M45'）。
        target_category (str): 用于审计比对的参考星表类别（例如 'hunt'）。
        mode (str): GMM 执行模式（如 '3d', '6d_p'）。
    """
    db, wf, ctx_cluster, data_manifest, ref_tables = _initialize_pipeline_components(
        target_cluster_id, target_category, mode
    )

    try:
        _ingest_and_prepare_data(db, wf, target_cluster_id, ref_tables, ctx_cluster)

        t_result = _execute_gmm_algorithm(wf, ctx_cluster)

        v_all_audit_data = _post_process_results(wf, t_result)

        audit_res, v_final_audited, deep_stats = _perform_cross_audit(
            wf, v_all_audit_data, data_manifest, target_cluster_id
        )

        _export_pipeline_results(
            db, audit_res, v_final_audited, target_cluster_id, target_category, mode
        )

        _log_final_report(
            target_cluster_id,
            target_category,
            mode,
            ctx_cluster,
            v_all_audit_data,
            audit_res,
            deep_stats,
        )

    except Exception as e:
        logger.error(f"❌ 流水线在运行期间发生严重崩溃: {str(e)}", exc_info=True)
        raise e
    finally:
        db.close()
        logger.info("🔒 数据库连接已释放。")


if __name__ == "__main__":
    # 1. 命令行参数解析配置
    parser = argparse.ArgumentParser(
        description="Astro Research Pipeline - 天文数据分析与自动审计命令行工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 星团 ID (必填，位置参数)
    parser.add_argument("cluster", type=str, help="目标星团名称 (例如: M45, M44, M67)")

    # 审计对象 (可选，缺省 hunt)
    parser.add_argument(
        "--category",
        type=str,
        default="hunt",
        choices=["cg20", "heyl", "zerj", "risb", "hunt"],
        help="审计对比的参考星表类别",
    )

    # 执行模式 (可选，缺省 3d)
    valid_modes = list(cfg.GMM_CONFIG["feature_map"].keys())
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        choices=valid_modes,
        help="GMM 算法的特征维度模式",
    )

    # 日志级别 (可选，缺省 INFO)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="控制台日志输出级别",
    )

    args = parser.parse_args()

    # 2. 初始化日志系统 (根据参数设置级别)
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    logger = setup_logging(level=numeric_level)

    # 2. 参数校验与预处理
    # 星团名不区分大小写匹配
    cluster_input = args.cluster.upper()
    cluster_map = {k.upper(): k for k in cfg.CLUSTERS.keys()}

    if cluster_input not in cluster_map:
        logger.error(
            f"❌ 未知的星团名称: '{args.cluster}'。可选范围: {list(cfg.CLUSTERS.keys())}"
        )
        sys.exit(1)

    target_cluster_id = cluster_map[cluster_input]

    logger.info(
        f"🚀 启动天文数据分析管道 - 目标: {target_cluster_id}, 模式: {args.mode}, 审计参考: {args.category}"
    )
    run_pipeline(target_cluster_id, args.category, args.mode)
