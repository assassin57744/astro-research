"""
星团成员审计管线入口（终极订正版）。

配置语义校准：
  -m, --mode : 精筛阶段 GMM 的相空间维度计算模式 (如 '3d', '5d', '6d_p')
  -a, --algo : 初筛阶段 种子星提取所使用的算法 (如 'dbscan', 'hdbscan')

大小写防御设计规范：
  - 严禁在全局或赋值阶段对任何输入字符串参数执行覆写性的 .upper() 或 .lower()。
  - 所有变量保持用户输入的原始字面量状态形态，仅在下游检索或逻辑分支时临时转换。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.workflow import Workflow
from modules.db import AstroDBFacade
import config as cfg

logger = logging.getLogger("AstroPipeline")


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

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname).4s] %(filename)18s:%(lineno)4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
# 命令行参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """配置并解析命令行依赖输入参数。"""
    parser = argparse.ArgumentParser(
        description="Gaia DR3 星团成员概率反演与多源文献联合审计管线大盘"
    )
    parser.add_argument(
        "-c", "--cluster", 
        required=True, 
        help="目标星团唯一标识符（例如: M45, M44, Mel25, Mel111, M67, M13, M41）"
    )
    parser.add_argument(
        "-cat", "--category",
        type=str,
        default="hunt",
        choices=["cg20", "heyl", "zerj", "risb", "hunt"],
        help="审计对比的参考星表类别",
    )
    valid_modes =["2d", "3d", "5d", "6d_o", "5d_h", "3d_v", "6d_p"] #list(cfg.GMM_CONFIG["feature_map"].keys())
    parser.add_argument(
        "-m", "--mode", 
        default="5d", 
        choices=valid_modes + ["all"],
        help="精筛阶段 GMM 相空间特征维度计算模式 (例如: 3d, 5d, 5d_h, 3d_v, 6d_p), 使用 'all' 将循环执行所有模式"
		
    )
    parser.add_argument(
        "-a", "--algo", 
        default="dbscan", 
        help="初筛阶段 种子星提取所选用的无监督密度聚类算法 (例如: dbscan, hdbscan)"
    )
    parser.add_argument(
        "--result",
        type=str,
        default="brief",
        choices=["brief", "detailed"],
        help="结果产出等级: brief (精简, 仅日志及轻量报告) 或 detailed (详细, 导出全量 CSV 资产)",
    )

    parser.add_argument(
        "-p", "--params", 
        default="static", 
        choices=["static", "dynamic"],
        help="物理先验资产的装载策略: 'static'使用静态底座; 'dynamic'启动自适应历史反演"
    )
    parser.add_argument(
        "--log", 
        default="info", 
        help="控制台终端输出的日志分级限制 (debug, info, warning, error)"
    )
    parser.add_argument(
        "-mt", "--maintenance", 
        action="store_true", 
        help="激活底座数仓备份、恢复或日常索引维护流"
    )
    return parser.parse_args()


# =============================================================================
# 维护与防御性校验
# =============================================================================

def handle_maintenance(args: argparse.Namespace) -> bool:
    """处理数仓维护与灾备恢复指令。若激活则阻断后续管线。"""
    if not args.maintenance:
        return False

    db = AstroDBFacade(manifest=cfg.MANIFEST)
    try:
        from modules.backup_manager import AstroBackupManager
        backup_mgr = AstroBackupManager(db)
        # 💡 参数保持字面量原样状态注入响应器
        intercepted = backup_mgr.handle_cli_maintenance(args)
        return intercepted
    except Exception as e:
        logger.error(f"❌ 维护流执行期间遭遇非预期崩溃: {e}")
        return False
    finally:
        db.close()


def _validate_cluster_input(cluster_input: str) -> str:
    """
    对输入的星团标识符进行防御性清洗。
    保持原始输入，仅剔除首尾空白，不在此处执行大范围的大小写转换赋值。
    """
    if not cluster_input or not cluster_input.strip():
        logger.error("❌ 命令行解析错误: 未提供有效的星团标识符 (-c / --cluster)。")
        sys.exit(1)
    return cluster_input.strip()


def main() -> None:
    """主程序总控制流：命令行解析 → 环境就绪 → 启动管道。"""
    args = parse_args()

    # 1. 日志系统级联初始化
    numeric_level = getattr(logging, args.log.upper(), logging.INFO)
    setup_logging(level=numeric_level if isinstance(numeric_level, int) else logging.INFO)

    # 2. 维护模式安全拦截
    if handle_maintenance(args):
        return

    # 3. 提取并纯化星团名（变量全程保持输入形态，不转换为大写）
    target_cluster_id = _validate_cluster_input(args.cluster)

    logger.info(
        f"🌌 [Pipeline Control] 成功锁定目标天体: '{target_cluster_id}' | "
        f"初筛算法 (-a): '{args.algo}' | 精筛维度模式 (-m): '{args.mode}'"
    )
    
    # 4. 实例化数仓中介并移交控制权
    # db = AstroDBFacade(manifest=cfg.MANIFEST)
    db = AstroDBFacade(cluster_id=target_cluster_id)
    try:
        wf = Workflow(cluster_id=target_cluster_id)
        wf.execute()
    except Exception as e:
        logger.error(f"❌ 管线在执行期间遭遇核心故障阻断: {e}", exc_info=True)
    finally:
        db.close()
        logger.info("🔒 基础设施层：DuckDB 数据库连接已安全释放。")


if __name__ == "__main__":
    main()
