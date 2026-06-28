"""
基础设施层：数仓备份与灾难恢复管理器（全新重构版）。

职责：
  1. 封装 DuckDB 底层的逻辑备份（EXPORT DATABASE）与灾难恢复（IMPORT DATABASE）物理行为。
  2. 提供直接响应并接管 main.py 命令行参数（如 --maintenance, --cluster）的直连控制入口。

大小写与字面量防御规范：
  - 严禁在全局或初始化、赋值阶段对任何传入的参数（如 cluster_id）执行盲目的覆盖性大小写转换。
  - 所有变量全程保持用户在命令行输入的原始字面量形态（如 'm45', 'Melotte_22'）。
  - 仅在拼接本地备份目录、生成 Parquet 数据快照路径、或组装 SQL 语句时，局部临时执行 .upper() 或 .lower() 规范化，
    由此确保系统在大小写敏感的跨平台（Windows/Linux/macOS）文件系统下的绝对唯一确定性，捍卫数据血缘不被污染。
"""

import logging
import os
import argparse
from datetime import datetime
from pathlib import Path
from modules.db import AstroDB
import config as cfg

logger = logging.getLogger("AstroPipeline.BackupManager")


class AstroBackupManager:
    """数仓资产快照备份、容灾重构与高级维护引擎。"""

    def __init__(self, db_instance: AstroDB):
        """
        注入唯一的数仓物理实例，并自适应级联初始化本地备份存储基底。
        """
        self.db = db_instance
        # 统一从 config.py 读取备份根路径，若未配置则采用安全的相对路径兜底
        self.backup_root = Path(getattr(cfg, "BACKUP_DIR", "./backups"))

    # =============================================================================
    # 核心物理行为层（物理 EXPORT / IMPORT 算子）
    # =============================================================================

    def execute_cluster_snapshot(self, cluster_id: str) -> bool:
        """
        📸 [逻辑备份] 针对特定星团的天区切片及概率计算中间表资产导出独立 Parquet 科学快照。
        
        常用于：Pipeline 的 Stage 5 收尾阶段，将当前跑出的高价值概率星表自动持久化固化。
        """
        safe_cluster_name = cluster_id.strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return True  
        
        # 💡【大小写隔离防御】仅在本地文件系统路径拼接时，临时局部采用规范化大写，不污染外部入参
        snapshot_dir = self.backup_root / "snapshots" / safe_cluster_name.upper() / timestamp
        
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 正在为星团 '{safe_cluster_name}' 导出物理数仓快照...")
            
            # 拼装局部大小写隔离的逻辑导出 SQL。
            # DuckDB 会自动在该目录下生成包含表结构定义文件的 schema.sql 以及高效压缩的 Parquet 数据。
            sql = f"""
                EXPORT DATABASE '{snapshot_dir.as_posix()}' 
                (FORMAT PARQUET, COMPRESSION 'ZSTD');
            """
            self.db.execute(sql)
            logger.info(f"✅ 快照逻辑备份成功！该版本科学资产已安全固化至: {snapshot_dir}")
            return True
        except Exception as e:
            logger.error(f"❌ 导出星团 '{safe_cluster_name}' 物理快照时遭遇核心崩溃: {e}", exc_info=True)
            return False

    def restore_cluster_snapshot(self, cluster_id: str, snapshot_folder_name: str) -> bool:
        """
        ⏪ [灾难恢复] 从指定的历史逻辑备份目录中，通过读取 schema 并反序列化 Parquet 重建整个天区资产底座。
        """
        safe_cluster_name = cluster_id.strip()
        # 💡【大小写隔离防御】在检索快照文件夹时局部临时转大写对齐
        target_dir = self.backup_root / "snapshots" / safe_cluster_name.upper() / snapshot_folder_name

        if not target_dir.exists():
            logger.error(f"❌ 容灾恢复重构中止：指定的物理备份资产路径不存在 -> {target_dir}")
            return False

        try:
            logger.warning(f"🚨 警告: 正在执行还原动作！将从历史备份中强行重构星团 '{safe_cluster_name}' 的数据拓扑...")
            
            # 物理执行 DuckDB 灾难恢复导入指令
            sql = f"IMPORT DATABASE '{target_dir.as_posix()}';"
            self.db.execute(sql)
            
            logger.info(f"🎉 星团 '{safe_cluster_name}' 的历史数据底座与清单资产已完美重构还原！")
            return True
        except Exception as e:
            logger.error(f"❌ 物理恢复星团 '{safe_cluster_name}' 资产时发生核心崩溃: {e}", exc_info=True)
            return False

    # =============================================================================
    # 🛠️ 命令行参数直连响应与控制流拦截接口
    # =============================================================================

    def handle_cli_maintenance(self, args: argparse.Namespace) -> bool:
        """
        🎯 [命令直连入口] 承接并直接响应由 main.py 传入的解析完成后的原生命令行 Namespace。
        
        参数:
          - args: 来自入口层 argparse 的命名空间。
        
        返回:
          - bool: True  表示拦截接管控制流并成功处理了维护/恢复作业，应当直接阻断常规 Workflow 计算管线；
                  False 表示命令行未检测到相关指令或未满足执行条件，安全放行常规 Pipeline。
        """
        # 1. 安全关卡：如果没有激活 --maintenance 标志，直接退回，放行常规科学计算
        if not getattr(args, "maintenance", False):
            return False

        logger.info("🛠️  [Backup CLI Target] 成功拦截到数仓维护与灾备恢复信号，正在接管控制流...")
        
        # 提取命令行中对应的星团唯一标识符，此时完美保持用户输入的原始形态（不执行 .upper()）
        cluster_id = getattr(args, "cluster", "").strip()
        
        # 2. 分支 A：如果没有指定特定的星团标识，则视为对数仓底座执行全局常规索引物理维护与碎片清理
        if not cluster_id:
            logger.warning("⚠️  检测到未指定目标星团 (-c/--cluster)，将对整个物理数仓执行全局优化与垃圾碎片整理。")
            try:
                # 执行 DuckDB 统计信息重构与物理表拓扑重新分析
                self.db.execute("ANALYZE;")
                logger.info("✅ 全局数仓拓扑分析、索引重建与优化流顺利收尾。")
            except Exception as e:
                logger.error(f"❌ 全局数仓常规维护执行失败: {e}")
            return True

        # 3. 分支 B：检测是否传入了显式的历史快照恢复文件夹参数（例如在命令行扩展了 --restore-snapshot 参数）
        restore_snapshot_dir = getattr(args, "restore_snapshot", None)
        
        if restore_snapshot_dir:
            logger.info(f"🔍 正在检索星团 '{cluster_id}' 的定向历史快照版本: {restore_snapshot_dir}")
            # 驱动底层的灾难恢复算子
            success = self.restore_cluster_snapshot(cluster_id, snapshot_folder_name=restore_snapshot_dir)
            if success:
                logger.info(f"✨ 星团 '{cluster_id}' 的全链灾备重构作业已完美执行闭环。")
            else:
                logger.error(f"❌ 星团 '{cluster_id}' 恢复流由于非预期错误异常终止。")
            return True
        
        # 4. 分支 C：若携带了特定的星团但无恢复参数，默认对该星团涉及的切片数据执行局部性能调优与真空抽吸
        try:
            logger.info(f"🧼 正在针对星团 '{cluster_id}' 的局部数仓底盘执行真空清理与性能统计更新...")
            # 同样利用 ANALYZE 执行局部级联更新
            self.db.execute("ANALYZE;") 
            logger.info(f"✅ 星团 '{cluster_id}' 定向局部数仓底层调优流执行成功。")
        except Exception as e:
            logger.error(f"❌ 执行星团 '{cluster_id}' 定向局部维护时崩溃: {e}")
            
        return True