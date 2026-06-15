import logging
import os
import signal
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

# 修正导入路径：确保脚本直接运行时能找到项目根目录下的 config.py
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

import config as cfg

logger = logging.getLogger("AstroPipeline.VizierFetcher")

class VizierDataFetcher:
    """
    Vizier 数据获取器：从 CDS VizieR 获取 Gaia DR3 数据作为 Gaia Archive 的快速替代方案。
    
    Gaia Archive (ESA) 的异步查询虽然稳健，但在高峰期排队时间极长。
    VizieR 对于中等规模（半径 < 5度）的天区通常可以实现秒级响应。
    """

    def __init__(self, output_dir=cfg.DOWNLOAD_DIR, timeout=120):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = cfg.INTERNAL_DIR / "fetch_state_vizier.json"
        self.state = self._load_state()
        self.interrupted = False
        
        # 从配置中提取 VizieR 侧的原始列名
        self.columns = list(cfg.FIELDS_VIZIER.values())
        
        # 初始化 Vizier 客户端：
        # row_limit=-1 确保获取所有匹配行
        # columns 为配置中定义的列表
        self.v = Vizier(row_limit=-1, columns=self.columns, timeout=timeout)
        self.catalog = "I/355/gaiadr3" # Gaia DR3 在 VizieR 的官方编号
        
        logger.debug(f"🛠️  Vizier 客户端初始化完成。目录: {self.catalog}, 超时: {timeout}s")
        logger.debug(f"📋 待获取字段列表: {self.columns}")

        # 注册中断信号处理
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # 用于模拟异步的线程池
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        logger.debug(f"💾 正在保存状态至: {self.state_file}")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=4)

    def _handle_interrupt(self, signum, frame):
        logger.warning("\n🛑 接收到用户中断信号 (Ctrl+C)。正在尝试保存当前状态并安全退出...")
        self.interrupted = True
        self._save_state()

    def fetch_cluster(self, cluster_id: str, mode="async"):
        """
        使用 VizieR 引擎获取特定星团的天区数据。支持 sync 和 async 两种模式。
        """
        if self.interrupted:
            return

        if cluster_id not in cfg.CLUSTERS:
            logger.error(f"未在配置中找到星团: {cluster_id}")
            return

        cluster_meta = cfg.CLUSTERS[cluster_id]
        manifest_key = cluster_meta["FIELD_IDX"]
        manifest_entry = cfg.MANIFEST[manifest_key]
        
        filename = manifest_entry["params"]["file_pattern"]
        dest_path = self.output_dir / filename

        # 如果文件已存在则跳过
        if dest_path.exists():
            logger.info(f"⏩ 目标文件已存在，跳过任务: {dest_path.name}")
            return

        # 检查是否在上次运行中被中断（可选，主要用于日志提示）
        if self.state.get(cluster_id) == "RUNNING":
            logger.info(f"🔄 发现上次运行中断记录: {cluster_id}，正在恢复下载...")

        self.state[cluster_id] = "RUNNING"
        self._save_state()

        logger.info(f"📡 [VizieR] 启动查询: {cluster_id}")
        logger.info(f"   ∟ 中心坐标: RA={cluster_meta['CENTER_RA']}, Dec={cluster_meta['CENTER_DEC']}")
        logger.info(f"   ∟ 检索半径: {cluster_meta['RADIUS']} deg | 最大视星等: {cluster_meta['MAX_MAG']}")
        
        # 准备查询参数
        coord = SkyCoord(
            ra=cluster_meta['CENTER_RA'], 
            dec=cluster_meta['CENTER_DEC'], 
            unit=(u.deg, u.deg), 
            frame='icrs'
        )
        radius = cluster_meta['RADIUS'] * u.deg
        
        # 设置动态过滤器：VizieR 侧 Gaia DR3 的星等列名通常为 Gmag
        mag_col = cfg.FIELDS_VIZIER.get("mag", "Gmag")
        self.v.column_filters = {mag_col: f"<{cluster_meta['MAX_MAG']}"}
        logger.debug(f"🔍 应用列过滤器: {self.v.column_filters}")

        try:
            start_time = time.time()
            
            if mode == "sync":
                # 原有的同步阻塞模式
                result = self.v.query_region(coord, radius=radius, catalog=self.catalog)
            else:
                # 模拟异步模式：在线程池中运行，主线程轮询以响应中断
                future = self._executor.submit(
                    self.v.query_region, coord, radius=radius, catalog=self.catalog
                )
                
                while not future.done():
                    if self.interrupted:
                        logger.warning(f"🛑 任务 [{cluster_id}] 在等待期间被用户取消。")
                        return
                    time.sleep(0.5) # 每 0.5 秒轮询一次状态
                
                result = future.result()

            elapsed = time.time() - start_time

            if self.interrupted:
                return

            if not result or len(result) == 0:
                logger.error(f"❌ VizieR 响应为空。请核对目录编号 ({self.catalog}) 或网络连接。")
                return

            table = result[0]
            logger.info(f"📥 数据拉取成功! 耗时: {elapsed:.2f}s")
            logger.info(f"   ∟ 样本规模: {len(table)} 行 | 特征数: {len(table.columns)}")
            logger.debug(f"   ∟ 返回列名: {table.colnames}")

            # 使用临时文件确保写入的原子性，支持真正的“断点续传”可靠性
            temp_path = dest_path.with_suffix(".tmp")
            logger.debug(f"💾 正在写入临时文件: {temp_path.name}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            table.write(temp_path, format='parquet', overwrite=True)
            
            # 写入成功后更名并更新状态
            logger.debug(f"🚚 原子性更名: {temp_path.name} -> {dest_path.name}")
            os.replace(temp_path, dest_path)
            
            if cluster_id in self.state:
                del self.state[cluster_id]
            self._save_state()
            
            logger.info(f"✅ {cluster_id} 数据已成功落地。")

        except Exception as e:
            self._save_state()
            logger.error(f"❌ 运行期间发生异常 [{cluster_id}]: {str(e)}", exc_info=True)

    def fetch_all_configured(self):
        """批量处理配置中的所有目标"""
        all_targets = list(cfg.CLUSTERS.keys())
        # 过滤掉已存在的文件，确保进度预测是针对实际需要执行的任务
        targets = [cid for cid in all_targets if not (self.output_dir / cfg.MANIFEST[cfg.CLUSTERS[cid]["FIELD_IDX"]]["params"]["file_pattern"]).exists()]
        
        total = len(targets)
        if total == 0:
            logger.info("✨ 所有配置的星团数据已存在，无需下载。")
            return

        logger.info(f"🚀 开始执行批量 VizieR 获取任务，剩余目标数量: {total}")
        start_batch_time = time.time()

        for i, cid in enumerate(targets, 1):
            if self.interrupted:
                break
                
            self.fetch_cluster(cid)
            
            # 计算进度与 ETA
            elapsed = time.time() - start_batch_time
            avg_time = elapsed / i
            eta_seconds = avg_time * (total - i)
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)
            
            logger.info(f"📊 进度: [{i}/{total}] | 单个均耗: {avg_time:.1f}s | 预计剩余: {eta_min}m {eta_sec}s")
            
            # 短暂休眠以对服务器友好
            time.sleep(1)
        
        if self.interrupted:
            logger.info("🏁 批量下载任务已被人工中断，JobID/状态已持久化。")
        else:
            logger.info("🏁 恭喜！所有配置的星团数据已全部获取并落地。")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    fetcher = VizierDataFetcher()
    fetcher.fetch_all_configured()
