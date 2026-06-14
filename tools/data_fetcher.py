import logging
import os
import json
import signal
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# 修正导入路径：确保脚本直接运行时能找到项目根目录下的 config.py
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

logger = logging.getLogger("AstroPipeline.DataFetcher")


@dataclass
class QueryCriteria:
    """定义统一的科学检索标准"""

    ra: float
    dec: float
    radius: float
    max_mag: float
    fields: Dict[str, str]


class BaseEngine(ABC):
    """数据引擎基类，定义统一的查询接口"""

    @abstractmethod
    def fetch_data(
        self,
        task_id: str,
        criteria: QueryCriteria,
        job_id: Optional[str],
        interrupted_check: Callable[[], bool],
    ) -> Optional[tuple[Any, str]]:
        pass

    @abstractmethod
    def get_job_id(self, cluster_id: str, state: Dict) -> Optional[str]:
        pass


class GaiaEngine(BaseEngine):
    """Gaia Archive 引擎：支持异步作业和断点恢复"""

    def __init__(self, user=None, password=None):
        if user and password:
            try:
                logger.info(f"🔑 正在登录 Gaia Archive (用户: {user})...")
                Gaia.login(user=user, password=password)
            except Exception as e:
                logger.error(f"❌ Gaia 登录失败: {str(e)}")
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    def get_job_id(self, cluster_id: str, state: Dict) -> Optional[str]:
        return state.get(cluster_id)

    def fetch_data(
        self,
        task_id: str,
        criteria: QueryCriteria,
        job_id: Optional[str],
        interrupted_check: Callable[[], bool],
    ) -> Optional[Any]:
        job = None

        if job_id:
            try:
                job = Gaia.get_job(job_id)
            except Exception:
                logger.warning(f"⚠️ 作业 {job_id} 无效，将重新提交。")

        if not job:
            adql = self._build_adql(criteria)
            job = Gaia.launch_job_async(adql, name=f"fetch_{task_id}")

        # 轮询状态
        while True:
            if interrupted_check():
                return None
            try:
                status = job.get_phase()
                if status == "COMPLETED":
                    break
                if status in ["ERROR", "ABORTED"]:
                    logger.error(f"❌ Gaia 作业失败: {status}")
                    return None
                logger.info(f"⏳ Gaia 执行中 [{task_id}]... 状态: {status}")
            except Exception as e:
                logger.warning(f"⚠️ 状态获取失败，稍后重试: {e}")
                time.sleep(10)
            time.sleep(10)

        return job.get_results(), job.jobid

    def _build_adql(self, c: QueryCriteria):
        cols = ", ".join(c.fields.values())
        return f"""
        SELECT {cols} FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {c.ra}, {c.dec}, {c.radius}))
        AND phot_g_mean_mag < {c.max_mag}
        """


class VizierEngine(BaseEngine):
    """VizieR 引擎：通过线程池模拟异步以响应中断"""

    def __init__(self, fields: Dict[str, str], timeout=120):
        self.v = Vizier(row_limit=-1, columns=list(fields.values()), timeout=timeout)
        self.catalog = "I/355/gaiadr3"
        self.mag_col = fields.get("mag", "Gmag")
        self._executor = ThreadPoolExecutor(max_workers=1)

    def get_job_id(self, cluster_id: str, state: Dict) -> Optional[str]:
        return "RUNNING" if state.get(cluster_id) else None

    def fetch_data(
        self,
        task_id: str,
        criteria: QueryCriteria,
        job_id: Optional[str],
        interrupted_check: Callable[[], bool],
    ) -> Optional[Any]:
        coord = SkyCoord(ra=criteria.ra * u.deg, dec=criteria.dec * u.deg, frame="icrs")
        self.v.column_filters = {self.mag_col: f"<{criteria.max_mag}"}

        future = self._executor.submit(
            self.v.query_region,
            coord,
            radius=criteria.radius * u.deg,
            catalog=self.catalog,
        )

        while not future.done():
            if interrupted_check():
                return None
            time.sleep(0.5)

        result = future.result()
        if result and len(result) > 0:
            return result[0], "DONE"
        return None


class AstroDataFetcher:
    """
    通用天文数据调度器
    """

    def __init__(
        self, engine: BaseEngine, output_dir: str, state_dir: str, state_name: str
    ):
        """初始化调度器，state_dir 为持久化中间状态的目录"""
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(state_dir) / f"fetch_state_{state_name}.json"
        self.state = self._load_state()
        self.interrupted = False

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        logger.warning("\n🛑 接收到用户中断信号 (Ctrl+C)。正在尝试保存状态并退出...")
        self.interrupted = True
        self._save_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def fetch(self, task_id: str, criteria: QueryCriteria, dest_path: Path):
        """执行单个拉取任务"""
        if dest_path.exists():
            logger.info(f"⏩ 跳过已存在文件: {dest_path.name}")
            return

        job_id = self.engine.get_job_id(task_id, self.state)

        try:
            fetch_res = self.engine.fetch_data(
                task_id, criteria, job_id, lambda: self.interrupted
            )
            if not fetch_res:
                return

            results, new_job_id = fetch_res
            if new_job_id:
                self.state[task_id] = new_job_id
                self._save_state()

            logger.info(f"📥 正在持久化 {task_id} ({len(results)} 行)...")
            temp_path = dest_path.with_suffix(".tmp")
            results.write(temp_path, format="parquet", overwrite=True)
            os.replace(temp_path, dest_path)

            if task_id in self.state:
                del self.state[task_id]
                self._save_state()
            logger.info(f"✅ {task_id} 完成。")
        except Exception as e:
            logger.error(f"❌ {task_id} 失败: {e}")


if __name__ == "__main__":
    # 只有作为脚本直接运行时，才引入项目特定的输入配置
    import tools.data_fetcher_input as cfg

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    choice = sys.argv[1] if len(sys.argv) > 1 else "gaia"

    if choice == "gaia":
        engine = GaiaEngine(cfg.GAIA_USER, cfg.GAIA_PWD)
    else:
        engine = VizierEngine(cfg.FIELDS_VIZIER)

    # 初始化时传入具体的目录配置
    fetcher = AstroDataFetcher(engine, cfg.DOWNLOAD_DIR, cfg.INTERNAL_DIR, choice)

    # 原 fetch_all_configured 的逻辑移至此处，保持 AstroDataFetcher 类的纯粹性
    targets = [
        cid
        for cid in cfg.CLUSTERS.keys()
        if not (
            fetcher.output_dir
            / cfg.MANIFEST[cfg.CLUSTERS[cid]["FIELD_IDX"]]["params"]["file_pattern"]
        ).exists()
    ]

    if not targets:
        logger.info("✨ 所有配置的星团数据已存在。")
    else:
        logger.info(f"🚀 启动批量任务，共 {len(targets)} 个...")
        for cid in targets:
            if fetcher.interrupted:
                break
            c = cfg.CLUSTERS[cid]
            criteria = QueryCriteria(
                ra=c["CENTER_RA"],
                dec=c["CENTER_DEC"],
                radius=c["RADIUS"],
                max_mag=c["MAX_MAG"],
                fields=(
                    cfg.FIELDS_GAIA_ARCHIVE
                    if isinstance(engine, GaiaEngine)
                    else cfg.FIELDS_VIZIER
                ),
            )
            path = (
                fetcher.output_dir
                / cfg.MANIFEST[c["FIELD_IDX"]]["params"]["file_pattern"]
            )
            fetcher.fetch(cid, criteria, path)
            time.sleep(1)
        logger.info("🏁 批量下载结束。")
