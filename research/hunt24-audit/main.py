"""星团成员审计管线入口。

负责 CLI 参数解析、日志初始化、维护命令分发，并将管线执行委托给 AstroWorkflow。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.astro_db import AssetManager
from modules.astro_workflow import AstroWorkflow

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
        help="结果产出等级: brief (精简, 仅日志及轻量报告) 或 detailed (详细, 导出全量 CSV 资产)",
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
        help="配置来源: file (静态 config.py) 或 db (从历史数据重建物理参数)",
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
# 参数验证
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

    if handle_maintenance(args):
        return

    target_cluster_id = _validate_cluster(args.cluster)

    logger.info(
        f"🚀 启动分析管道 - 审计范围: {target_cluster_id}, "
        f"精筛特征空间: {args.mode}, 审计对象: {args.category}"
    )

    if args.mode == "all":
        AstroWorkflow.run_all_modes(
            target_cluster_id=target_cluster_id,
            target_category=args.category,
            algo=args.algo,
            result_mode=args.result,
            reconstruct_mode=args.reconstruct,
        )
    else:
        wf = AstroWorkflow(
            db_instance=None,
            target_cluster=target_cluster_id,
            target_category=args.category,
            mode=args.mode,
            algo=args.algo,
        )
        wf.run(
            reconstruct_mode=args.reconstruct,
            result_mode=args.result,
        )


if __name__ == "__main__":
    main()
