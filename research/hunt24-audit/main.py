"""星团成员审计管线入口。

负责 CLI 参数解析、日志初始化、维护命令分发，并将管线执行委托给 AstroWorkflow。
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

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

    # 未来可以在这里随意添加新参数，Workflow 会自动吸收
    # parser.add_argument("--new-feature", type=float, default=0.5)

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
    target_cluster_id = _validate_cluster(args.cluster)

    # 🌟 [设计模式] 接口解耦方案：
    # 将 args 命名空间转换为字典，并注入已校验的星团 ID。
    # 这样 Workflow 的构造函数无需随着 CLI 参数的增加而修改。
    workflow_params = vars(args).copy()
    workflow_params["target_cluster"] = target_cluster_id

    logger.info(f"🚀 [Startup] 启动分析管道 - 审计目标: {args.category}")
    logger.info(f"📊 [Startup] 运行模式: {args.mode} | 审计范围: {target_cluster_id} | 算法: {args.algo}")

    if args.mode == "all":
        # 暂不支持
        logger.error("❌ [System] 暂不支持批量模式。")
        raise NotImplementedError
        # 批量模式也建议统一接受配置字典
        # AstroWorkflow.run_all_modes(**workflow_params)
    else:
        # 🌟 动态解包传入所有参数
        wf = AstroWorkflow(db_instance=None, **workflow_params)
        # wf.init_data()
        # ctx_cluster = cfg.CLUSTERS[target_cluster_id].copy()
        # ctx_cluster["id"] = target_cluster_id
        wf.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n🛑 [System] 用户手动中断了管线运行 (Ctrl+C)。")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"💥 [System] 发生未捕获的致命崩溃: {e}", exc_info=True)
        sys.exit(1)
