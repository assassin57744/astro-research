"""星团成员审计管线入口。

负责 CLI 参数解析、日志初始化、维护命令分发，并将管线执行委托给 AstroWorkflow。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.db import AssetManager
from modules.workflow import AstroWorkflow

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

    # 统一日志格式，增加对齐
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

    logger.info(f"📝 [System] 日志系统就绪。执行快照: {log_path.name}")


# =============================================================================
# CLI 参数解析
# =============================================================================


def _parse_key_value_pairs(raw: list[str]) -> dict[str, Any]:
    """将 ['eps=0.3', 'strategy=bayesian'] 解析为 {'eps': 0.3, 'strategy': 'bayesian'}。"""
    result: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        low = v.lower()
        if low in ("true", "false"):
            result[k] = low == "true"
        elif low in ("none", "auto"):
            result[k] = low
        else:
            try:
                result[k] = float(v) if "." in v else int(v)
            except ValueError:
                result[k] = v
    return result


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
        default="5d",
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
        help="结果产出等级: brief (精简) 或 detailed (导出全量资产)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="控制台日志输出级别",
    )
    parser.add_argument(
        "--reconstruct",
        type=str,
        default="file",
        choices=["file", "db"],
        help="配置来源: file (静态) 或 db (历史重建)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        default=False,
        help="跳过 Phase 7 可视化分析阶段（空间管掩模图、概率分布直方图等）",
    )

    # 🚀 扩展入口：算法/审计/种子参数覆盖
    parser.add_argument(
        "--algo-params",
        type=str,
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="算法微调参数，例：--algo-params eps=0.5 strategy=threshold",
    )
    parser.add_argument(
        "--audit-params",
        type=str,
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="审计微调参数，例：--audit-params skip_simbad=true",
    )
    parser.add_argument(
        "--seed-params",
        type=str,
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="种子集微调参数，例：--seed-params radius_override=3.0",
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
    """处理资产维护子命令。"""
    manager = AssetManager()
    if args.query_backup:
        logger.info("🔍 [System] 正在查询备份资产状态...")
        manager.query_backup_assets()
        return True

    if args.backup:
        logger.info("📦 [System] 正在启动手动备份流程...")
        manager.manage_backup_assets()
        return True

    if args.restore is not None:
        logger.warning(f"🔄 [System] 正在准备从备份 [{args.restore}] 执行强制恢复...")
        manager.restore_backup_assets(target=args.restore)
        return True

    return False


# =============================================================================
# 参数验证
# =============================================================================


def _validate_cluster(cluster_str: str | None) -> str:
    """校验并标准化星团名称。"""
    if not cluster_str:
        logger.error("❌ [System] 运行模式缺失必填参数: cluster")
        sys.exit(1)

    cluster_input = cluster_str.upper()
    cluster_map = {k.upper(): k for k in cfg.CLUSTERS.keys()}

    if cluster_input not in cluster_map:
        logger.error(
            f"❌ [System] 未知的星团名称: '{cluster_str}'。可选范围: {list(cfg.CLUSTERS.keys())}"
        )
        sys.exit(1)

    return cluster_map[cluster_input]


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    """主程序入口：参数解析 → 环境初始化 → 委托 AstroWorkflow 执行。"""
    args = parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(
        level=numeric_level if isinstance(numeric_level, int) else logging.INFO
    )

    # 1. 处理维护模式
    if handle_maintenance(args):
        logger.info("✅ [System] 维护任务处理完成，程序退出。")
        return

    # 2. 准备管线参数
    target_cluster_ids = []
    if args.cluster.lower() == "all":
        target_cluster_ids = list(cfg.CLUSTERS.keys())
    else:
        target_cluster_id = _validate_cluster(args.cluster)
        target_cluster_ids = [target_cluster_id]

    # 确定特征空间列表
    if args.mode == "all":
        feature_spaces = list(cfg.GMM_CONFIG["feature_map"].keys())
    else:
        feature_spaces = [args.mode]

    # 将键值对参数解析为 dict
    algo_params = _parse_key_value_pairs(args.algo_params)
    audit_params = _parse_key_value_pairs(args.audit_params)
    seed_params = _parse_key_value_pairs(args.seed_params)

    logger.info(f"🚀 [Startup] 启动分析管道 - 审计目标: {args.category}")
    logger.info(
        f"📊 [Startup] 运行模式: {args.mode} | 审计范围: {target_cluster_ids} | 算法: {args.algo}"
    )


    

    # 统一使用 run_batch（单模式也走同一代码路径）
    wf = AstroWorkflow(db_instance=None)
    wf.run(
        clusters=target_cluster_ids,
        categories=[args.category],
        feature_spaces=feature_spaces,
        algorithms=[args.algo],
        result_mode=args.result,
        param_source=args.reconstruct,
        skip_viz=args.skip_viz,
        algo_params_override=algo_params,
        audit_params_override=audit_params,
        seed_params_override=seed_params,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n🛑 [System] 用户手动中断了管线运行 (Ctrl+C)。")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"💥 [System] 发生未捕获的致命崩溃: {e}", exc_info=True)
        sys.exit(1)

