import logging
import os
import json
import signal
import sys
import time
from pathlib import Path

# 修正导入路径：确保脚本直接运行时能找到项目根目录下的 config.py
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from astroquery.gaia import Gaia
import config as cfg

logger = logging.getLogger("AstroPipeline.DataFetcher")

class GaiaDataFetcher:
    """
    Gaia 数据获取器：负责从 Gaia Archive 异步下载星团大天区数据。
    """

    def __init__(self, output_dir=cfg.DOWNLOAD_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = cfg.INTERNAL_DIR / "fetch_state_gaia.json"
        self.state = self._load_state()
        self.interrupted = False

        # 执行用户登录以获取更高配额与作业保留时长
        self._login()

        # 设置 Gaia 默认输出格式
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" 
        
        # 注册中断信号处理
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _login(self):
        if cfg.GAIA_USER and cfg.GAIA_PWD:
            try:
                logger.info(f"🔑 正在登录 Gaia Archive (用户: {cfg.GAIA_USER})...")
                Gaia.login(user=cfg.GAIA_USER, password=cfg.GAIA_PWD)
                logger.info("✅ 登录成功，已获得认证会话。")
            except Exception as e:
                logger.error(f"❌ Gaia 登录失败: {str(e)}")

    def _handle_interrupt(self, signum, frame):
        logger.warning("\n🛑 接收到用户中断信号 (Ctrl+C)。正在尝试保存状态并退出...")
        self.interrupted = True
        self._save_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def fetch_cluster(self, cluster_id: str):
        """
        获取指定星团的数据。
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

        if dest_path.exists():
            logger.info(f"⏩ 文件已存在，跳过下载: {dest_path}")
            return

        job = None
        job_id = self.state.get(cluster_id)

        try:
            if job_id:
                logger.info(f"🔍 发现历史作业记录 {job_id}，尝试恢复...")
                try:
                    job = Gaia.get_job(job_id)
                except Exception:
                    logger.warning(f"⚠️ 作业 {job_id} 在服务器上已过期或无效，将重新提交。")
                    job_id = None

            if not job_id:
                logger.info(f"📡 正在提交新异步查询: {cluster_id} (半径: {cluster_meta['RADIUS']} deg)")
                cols = ", ".join(cfg.FIELDS_GAIA_ARCHIVE.values())
                adql = f"""
                SELECT {cols}
                FROM gaiadr3.gaia_source
                WHERE 1=CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE('ICRS', {cluster_meta['CENTER_RA']}, {cluster_meta['CENTER_DEC']}, {cluster_meta['RADIUS']})
                )
                AND phot_g_mean_mag < {cluster_meta['MAX_MAG']}
                """
                job = Gaia.launch_job_async(adql, name=f"fetch_{cluster_id}")
                self.state[cluster_id] = job.jobid
                self._save_state()

            # 🛰️ 实时轮询作业状态，让用户看到进度
            while True:
                if self.interrupted:
                    logger.warning(f"🛑 检测到中断信号，作业 {job.jobid} 将保留在服务器。")
                    return

                try:
                    status = job.get_phase()
                    if status == 'COMPLETED':
                        logger.info(f"✨ 作业 {job.jobid} 已经在服务器端完成计算。")
                        break
                    elif status in ['ERROR', 'ABORTED']:
                        logger.error(f"❌ 作业 {job.jobid} 运行失败，服务器返回状态: {status}")
                        if cluster_id in self.state: 
                            del self.state[cluster_id]
                            self._save_state()
                        return
                    logger.info(f"⏳ 正在等待服务器执行 [{cluster_id}]... 当前状态: {status}")
                except Exception as poll_err:
                    # 容错处理：状态查询失败（如网络抖动）不应导致任务失败
                    logger.warning(f"⚠️ 无法获取作业状态: {poll_err}。将在 30 秒后重试...")
                    time.sleep(20) # 额外多等一会儿

                time.sleep(10)

            # 此时作业已完成，开始从服务器下载数据流
            logger.info(f"📥 正在从服务器下载结果集并转换为内存对象...")
            results = job.get_results()

            # 核心改进：一旦 results 拿到手，哪怕检测到 self.interrupted 也要强行存盘
            # 因为 results 的获取可能已经耗费了很长时间的网络 IO
            logger.info(f"📥 下载完成，获取到 {len(results)} 条记录。正在执行原子化持久化...")
            
            temp_path = dest_path.with_suffix(".tmp")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            results.write(temp_path, format='parquet', overwrite=True)
            os.replace(temp_path, dest_path)

            if self.interrupted:
                logger.warning(f"🌙 数据已保存，但在后续清理状态前接收到中断信号。")
                return

            # 下载成功后清理状态
            if cluster_id in self.state:
                del self.state[cluster_id]
                self._save_state()
            
            logger.info(f"✅ {cluster_id} 数据已成功落地。")

        except KeyboardInterrupt:
            self.interrupted = True
            logger.warning("\n🖐️ 操作被用户中断。")
            self._save_state()
        except Exception as e:
            logger.error(f"❌ 获取 {cluster_id} 数据失败: {str(e)}")

    def fetch_all_configured(self):
        """
        循环下载 config.CLUSTERS 中定义的所有星团。
        """
        all_clusters = list(cfg.CLUSTERS.keys())
        # 过滤已存在文件
        target_clusters = [cid for cid in all_clusters if not (self.output_dir / cfg.MANIFEST[cfg.CLUSTERS[cid]["FIELD_IDX"]]["params"]["file_pattern"]).exists()]
        
        total = len(target_clusters)
        if total == 0:
            logger.info("✨ 所有配置的星团数据已存在。")
            return

        logger.info(f"🚀 开始批量获取 Gaia 数据，共 {total} 个新任务...")
        start_batch_time = time.time()

        for i, cid in enumerate(target_clusters, 1):
            if self.interrupted:
                break
            self.fetch_cluster(cid)
            
            elapsed = time.time() - start_batch_time
            avg_time = elapsed / i
            eta_seconds = avg_time * (total - i)
            
            logger.info(f"📊 批量进度: [{i}/{total}] | 平均每任务耗时: {avg_time:.1f}s | 预计还需: {int(eta_seconds//60)}分{int(eta_seconds%60)}秒")

            # 任务间稍作停顿，避免触发 Gaia API 的并发限制
            if not self.interrupted:
                time.sleep(1)

        if self.interrupted:
            logger.info("🏁 任务已中断，未完成的 JobID 已保存。")
            sys.exit(0)
        else:
            logger.info("🏁 所有下载任务已完成。")

if __name__ == "__main__":
    # 独立运行时配置日志
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    fetcher = GaiaDataFetcher()
    # fetcher.fetch_all_configured()
    # 这两个比较小
    # fetcher.fetch_cluster("M45")
    # fetcher.fetch_cluster("M44")
    # 这两个范围太大, 暂时不取
    # fetcher.fetch_cluster("MEL25")
    # fetcher.fetch_cluster("MEL111")
    # fetcher.fetch_cluster("M67") # 实际下载半径2.5度
    # fetcher.fetch_cluster("M13")
    fetcher.fetch_cluster("M41")

# ================================================================================
# Cluster  | Cur_R  | Act_R  | Sug_R  | Cur_M  | Act_M  | Sug_M 
# --------------------------------------------------------------
# M45      | 17.78  | 14.82  | 17.78  | 21.0   | 20.6   | 22.0  
# M44      | 11.90  | 9.92   | 11.90  | 21.0   | 20.7   | 22.0  
# Mel25    | 59.31  | 49.43  | 59.31  | 21.0   | 20.6   | 22.0  
# Mel111   | 42.61  | 35.50  | 42.61  | 21.0   | 20.4   | 21.0  
# M67      | 2.16   | 1.66   | 2.16   | 21.0   | 20.6   | 21.0  
# M13      | 3.28   | 2.73   | 3.28   | 21.0   | 20.5   | 21.0  
# M41      | 2.53   | 2.03   | 2.53   | 19.0   | 20.0   | 21.0  
# ================================================================================
