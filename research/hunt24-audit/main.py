import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# 按照扁平化结构更新导入路径
from modules.astro_db import AstroDB
from modules.pg_core import PriorGMM
from modules.astro_workflow import AstroWorkflow

import config as cfg

# 初始化模块级记录器，确保所有函数共享同一日志上下文
logger = logging.getLogger("AstroPipeline")


class NameShortenerFilter(logging.Filter):
    """
    日志记录过滤器：用于精简记录器名称并合成紧凑的调用路径。

    主要功能：
    1. 提取模块叶节点名称（例如：将 'AstroPipeline.modules.seed_core' 简化为 'seed_core'）。
    2. 合成 'call_path' 属性，将简短名与函数名结合，便于在日志中进行等宽对齐美化。
    """

    def filter(self, record):
        # 提取模块层级的最后一级，避免长路径破坏日志排版
        if "." in record.name:
            short_name = record.short_name = record.name.split(".")[-1]
        else:
            short_name = record.short_name = record.name

        # 预合成调用路径 (例如: "astro_workflow.run_pipeline")
        # 在日志中使用固定宽度对齐，极大提升可读性
        record.call_path = f"{short_name}.{record.funcName}"

        return True


def setup_logging(log_dir=cfg.LOG_DIR):
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

    # 定义全局日志格式：时间 - 调用路径(固定45宽) - 级别 - 消息
    formatter = logging.Formatter(
        "%(asctime)s - %(call_path)-45s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    fh.addFilter(shortener)
    logger.addHandler(fh)

    # 控制台处理器通常设置为 INFO，避免 DEBUG 级别的冗余信息干扰实时监控
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    ch.addFilter(shortener)
    logger.addHandler(ch)

    logger.info(f"📝 日志系统就绪。文件存放在: {log_path}")
    logger.debug("已激活名称精简过滤器以优化日志输出。")
    return logger


def run_pipeline(target_cluster_id: str, target_category: str):
    """
    驱动端到端的天文数据分析与审计管线。

    该函数负责编排数据导入、空间对齐、GMM 算法处理、交叉验证及最终审计的完整流程。

    Args:
        target_cluster_id (str): 目标星团标识符（例如 'M45'）。
        target_category (str): 用于审计比对的参考星表类别（例如 'hunt'）。
    """
    ctx_cluster = cfg.CLUSTERS[target_cluster_id].copy()
    ctx_cluster["id"] = target_cluster_id  # 显式注入 ID，供 Actions 模块使用
    data_manifest = cfg.MANIFEST

    db = AstroDB(manifest=data_manifest)

    # 定义流水线运行所需的参考星表列表，包括基础星场、种子点及各文献比对表
    ref_tables = [
        ctx_cluster["FIELD_IDX"],  # 动态获取当前星团的全量数据索引
        ctx_cluster["SEED_IDX"],  # 动态获取当前星团的种子源索引
        cfg.IDX_DR2IDX,  # Gaia DR2-DR3 转换索引, 配合IDX_CG20使用
        cfg.IDX_CG20,  # Cantat-Gaudin (2020) 参考表
        cfg.IDX_HEYL,  # Heyl (2022) 参考表
        cfg.IDX_HUNT,  # Hunt (2024) 参考表
        cfg.IDX_ZERJ,  # Zerjal (2023) 参考表
        cfg.IDX_RISB,  # Risb (2023) 参考表
    ]

    try:
        # 1. 基础数据层同步 (L1: Raw Data Layer)
        # 此时 db 会根据 sync_mode 自动跳过标注为 VIRTUAL 的种子集表
        logger.info(f"📦 [1/5] 正在同步物理数据源 (跳过虚拟视图)...")
        db.import_raw(target_cluster_id=target_cluster_id, force=False) 

        # 在底层物理环境就绪后，初始化工作流引擎
        wf = AstroWorkflow(db, target_cluster=target_cluster_id, target_category=target_category)
        
        logger.info(
            f"📐 [2/5] 正在执行数据对齐与虚拟视图创建 ({target_cluster_id})..."
        )
        # prepare_field_data 内部会检测 VIRTUAL 模式，
        # 并使用 ctx_cluster 渲染 config 中的 pre_filters（如 M45 的 5度半径筛选）
        wf.prepare_field_data(ref_tables, ctx_cluster)

        # 2. 执行核心概率推断算法
        logger.info(
            f"🧠 [3/5] 启动 GMM 成员识别内核 ({cfg.GMM_CONFIG['dim_mode']} 模式)..."
        )
        t_result = wf.run_pipeline(ctx_cluster)
        logger.info(f"✨ 算法推断完成，结果表: {t_result}")

        # 3. 后处理与成员分类
        logger.info("📊 [4/5] 正在合成分析宽表并提取候选成员视图...")
        v_all_audit_data = wf.post_pipeline(t_result)
        if v_all_audit_data.get("status") != "success":
            logger.error(f"❌ 后处理流程失败: {v_all_audit_data.get('message')}")
            return

        logger.info("✅ 数据处理流程结束，转入交叉审计阶段。")
        v_audit_ref = v_all_audit_data["v_candidates"]  # 获取算法产出的候选成员视图名

        # 4. 交叉比对与审计
        logger.info(f"⚖️ [5/5] 执行多源文献交叉审计, 参考类别: {target_category}")

        # 动态渲染对齐视图名称，将占位符 {cluster} 替换为实际执行的星团 ID
        target_aln_view = data_manifest[target_category]["aln_view"].format(
            cluster=target_cluster_id.lower()
        )
        audit_res = wf.prepare_audit_data(v_audit_ref, target_aln_view)
        if audit_res.get("status") != "success":
            logger.warning(f"⚠️ 交叉比对审计未完全成功: {audit_res.get('message')}")
            return

        # 5. 发现挖掘：提取“仅被我方算法识别，而文献未包含”的源（New Discovery Candidates）
        gmm_only_key = f"v_{cfg.IDX_GMM}_only"
        v_audit_target = audit_res.get("audit_views", {}).get(gmm_only_key)

        if not v_audit_target:
            logger.warning(f"⚠️ 未找到名为 {gmm_only_key} 的审计视图，跳过深度审计。")
        else:
            logger.info(f"🔍 准备深度审计，目标视图: {v_audit_target}")
            wf.run_audit(target=v_audit_target)

        logger.info(f"🏁 任务执行完毕，星团 {target_cluster_id} 的完整流水线已结束。")
    except Exception as e:
        logger.error(f"❌ 流水线在运行期间发生严重崩溃: {str(e)}", exc_info=True)
        raise e
    finally:
        db.close()
        logger.info("🔒 数据库连接已释放。")


if __name__ == "__main__":
    logger = setup_logging()

    # 从环境变量或 config 默认值获取初始目标
    target_cluster = os.getenv("ASTRO_TARGET_CLUSTER", cfg.DEFAULT_CLUSTER)
    target_cat = os.getenv("ASTRO_TARGET_CATEGORY", cfg.DEFAULT_CATEGORY)
    
    logger.info(
        f"🚀 启动天文数据分析管道 - 目标: {target_cluster}, 审计参考: {target_cat}"
    )
    run_pipeline(target_cluster, target_cat)
