"""
星团成员审计管线入口（优雅架构解耦版）。

配置语义校准：
  -t, --target-clusters : 目标星团唯一标识符（如 'M45', 'Mel25', 多个逗号分隔，使用 'all' 扫描全量底座）
  -c, --category        : 审计对比的参考星表类别（如 'hunt', 'cg20'），缺省为 'hunt'
  -m, --mode            : 精筛阶段 GMM 的相空间维度计算模式 (如 '3d', '5d', '6d_p', 使用 'all' 循环所有模式)
  -a, --algo            : 初筛阶段 种子星提取所使用的算法 (如 'dbscan', 'hdbscan')

大小写防御设计规范：
  - 严禁在全局或赋值阶段对任何输入字符串参数执行覆写性的 .upper() 或 .lower()。
  - 所有变量保持用户输入的原始字面量状态形态，仅在下游检索或逻辑分支时临时转换。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# 导入已经具备多轨调度能力的 Workflow（或未来演进的批量调度引擎）
from modules.workflow import Workflow
from modules.db import AstroDBFacade
import config as cfg

logger = logging.getLogger()


# =============================================================================
# 日志系统初始化
# =============================================================================


def setup_logging(level: int = logging.INFO) -> None:
    """
    初始化双通道（文件 + 控制台）日志系统。
    保持高精度调试日志持续输出至本地磁盘，控制台根据输入阈值过滤。
    """
    log_dir = Path(cfg.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / datetime.now().strftime("astro_run_%Y%m%d.log")

    # 👇 【新增这部分】清除根日志器上可能残存的默认 Handler
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname).4s] %(filename)18s:%(lineno)-4d | %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # 磁盘文件通道（持久化 Debug 级别）
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台终端通道（动态过滤级别）
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"📝 日志系统就绪。文件存放在: {log_path}")


# =============================================================================
# 数仓维护拦截流
# =============================================================================


def handle_maintenance(args: argparse.Namespace) -> bool:
    """
    数仓高危维护机制安全关卡。
    如果激活了 --maintenance，将拦截后续计算，直接导向索引重建与快照备份。
    """
    if not args.maintenance:
        return False

    logger.warning(
        "⚠️ [Security Warning] 检测到维护流激活指令，常规科学计算管线已被阻断拦截。"
    )
    db = AstroDBFacade()
    try:
        from modules.backup_manager import execute_pipeline_maintenance

        intercepted = execute_pipeline_maintenance(args)  # TODO: 待完善
        return intercepted
    except Exception as e:
        logger.error(f"❌ 维护流执行期间遭遇非预期崩溃: {e}")
        return False
    finally:
        db.close()


# =============================================================================
# 核心重构：统一处理 all 参数选项及输入清洗
# =============================================================================


def resolve_runtime_matrix(
    clusters_input: str, feature_spaces_input: str
) -> Tuple[List[str], List[str]]:
    """
    【架构优化】统一收拢并解析命令行中关于 'all' 预设的多态展开逻辑。

    职责：
      1. 纯化清洗星团输入序列，处理 'all' 全量扫描。
      2. 纯化清洗特征相空间模式输入，处理 'all' 多轨联跑。
      3. 全程保持用户输入字面量的原始形态，严禁全局大小写覆写赋值。

    返回:
      tuple: (展开后的星团列表, 展开后的空间维度模式列表)
    """
    if not clusters_input or not clusters_input.strip():
        logger.error(
            "❌ 命令行解析错误: 未提供有效的星团标识符 (-t / --target-clusters)。"
        )
        sys.exit(1)

    # ---- 1. 动态展开目标星团序列 ----
    # 规范所有输入为小写
    raw_clusters = [
        tok.strip().lower() for tok in clusters_input.split(",") if tok.strip()
    ]

    if "all" in [t for t in raw_clusters]:
        try:
            resolved_clusters = list(cfg.CLUSTERS.keys())
            logger.info(
                f"🌌 [Global Search] 检测到全局扫描指令 'all'，已从物理先验大盘动态装载全量目标天体: {resolved_clusters}"
            )
        except AttributeError as e:
            # 抛出具体的自定义异常，把原始错误包裹进去
            raise AttributeError(
                "config.py 中未发现 CLUSTER_PRESETS 先验定义大盘，无法展开 'all' 指令。"
            ) from e
    else:
        resolved_clusters = raw_clusters

    # ---- 2. 动态展开特征空间模式 ----
    # 规范所有输入为小写
    raw_feature_spaces = [
        tok.strip().lower() for tok in feature_spaces_input.split(",") if tok.strip()
    ]

    if "all" in [t for t in raw_feature_spaces]:
        raw_feature_spaces = [
            "2d",
            "3d",
            "5d",
            "6d_o",
            "3d_v",
            "5d_h",
            "6d_p",
        ]  # 默认循环执行的经典核心相空间子集
        logger.info(
            f"📊 [Mode Poly] 检测到模式全扫指令 'all'，将级联循环特征子空间: {raw_feature_spaces}"
        )
    else:
        resolved_feature_spaces = raw_feature_spaces

    return resolved_clusters, resolved_feature_spaces


# =============================================================================
# 命令行接口定义
# =============================================================================


def parse_args() -> argparse.Namespace:
    """定义 Gaia DR3 星团成员审计管线的全量选项参数解析器。"""
    parser = argparse.ArgumentParser(description="基于Gaia DR3 数据的星表成员审计")

    # 核心资产定位参数组
    parser.add_argument(
        "-t",
        "--target-clusters",
        default="m45",
        required=True,
        help="目标星团唯一标识符（例如: M45, M44, Mel25, all），多个星团用逗号分隔，使用 'all' 循环扫描所有预设星团",
    )
    parser.add_argument(
        "-c",
        "--category",
        type=str,
        choices=["cg20", "heyl", "zerj", "risb", "hunt"],
        default="hunt",
        help="审计对比的参考文献文献星表类别 (缺省: hunt)",
    )
    valid_feature_spaces = [
        "2d",
        "3d",
        "5d",
        "6d_o",
        "5d_h",
        "3d_v",
        "6d_p",
    ]  # list(cfg.GMM_CONFIG["feature_map"].keys())
    # 算法与超参数矩阵参数组
    parser.add_argument(
        "-f",
        "--feature-spaces",
        default="5d",
        choices=valid_feature_spaces + ["all"],
        help="精筛阶段 GMM 相空间特征维度计算模式，使用 'all' 将循环执行所有受支持的空间模式",
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="dbscan",
        help="粗筛阶段 种子星提取所选用的无监督密度聚类算法 (例如: dbscan, hdbscan)",
    )

    # 资产固化与环境流控参数组
    parser.add_argument(
        "--result",
        type=str,
        default="brief",
        choices=["brief", "detailed"],
        help="结果产出等级: brief (精简, 仅日志及轻量报告) 或 detailed (详细, 导出全量 CSV 资产)",
    )

    parser.add_argument(
        "-p",
        "--param-recon-mode",
        default="static",
        choices=["static", "dynamic"],
        help="星团运动参数的装载策略: 'static'使用静态配置; 'dynamic'根据文献星表重建",
    )
    parser.add_argument(
        "--log",
        default="info",
        help="控制台终端输出的日志分级限制 (debug, info, warning, error)",
    )
    parser.add_argument(
        "-m",
        "--maintenance",
        action="store_true",
        help="激活底座数仓备份、恢复或日常索引维护流",
    )

    return parser.parse_args()


# =============================================================================
# 主程序控制流
# =============================================================================


def main() -> None:
    """主程序总控制流：极简外壳（参数解析 → 矩阵组装 → 移交工作流引擎）。"""
    args = parse_args()

    # 1. 日志系统级联初始化
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    setup_logging(level=log_level if isinstance(log_level, int) else logging.INFO)

    # 2. 维护模式安全拦截
    if handle_maintenance(args):
        return

    # 3. 统一解析并组装运行时矩阵
    clusters_to_audit, feature_spaces_to_run = resolve_runtime_matrix(
        args.target_clusters, args.feature_spaces
    )

    logger.info(
        f"🚀 [Pipeline Control] 运行时上下文解析就绪。目标星团总数: {len(clusters_to_audit)} | 空间维度总数: {len(feature_spaces_to_run)}"
    )
    logger.info(
        f"📌 [Global Config] 初筛算法 (-a): '{args.algo}' | 参考对比星表 (-c): '{args.category}' | 星团参数来源 (-p): '{args.param_recon_mode}'"
    )

    try:
        wf = Workflow(
            cluster_ids=clusters_to_audit,
            param_recon_mode=args.param_recon_mode,
            feature_spaces=feature_spaces_to_run,
        )
        wf.execute()
    except Exception as e:
        logger.error(
            f"💥 [Global Crash] 管线执行期间遭遇非预期灾难性击穿: {e}", exc_info=True
        )
    finally:
        # db.close()
        logger.info("🔒 基础设施层：DuckDB 数据库连接已安全释放。")


if __name__ == "__main__":
    main()
