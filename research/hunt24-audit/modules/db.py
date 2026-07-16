import os
import duckdb
import certifi
import ssl
import logging
import pandas as pd
import re
import time
import random
import sys
import shutil
import json
from datetime import datetime
from astroquery.simbad import Simbad
from pathlib import Path

from config import STD_COLS
import config as cfg

class AstroDB:
    def __init__(self, manifest=None, db_path=None):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

        # 统一使用 config.py 中已经定义好的 Path 对象
        self.data_root = cfg.DATA_DIR

        self.dirs = {
            "data_root": self.data_root,
            "raw": self.data_root / "raw",
            "warehouse": self.data_root / "warehouse",
        }

        if db_path is None:
            db_path = self.dirs["warehouse"] / "astrodb_internal.db"

        # 💡 判断是否是全新创建（冷启动）
        is_new_db = not db_path.exists()
        if is_new_db:
            # 🛡️ 核心修复：确保数据库所在的目录（warehouse）物理存在，防止 duckdb.connect 抛出路径未找到异常
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.warning(
                f"🆕 [Startup] 未检测到本地数据库，正在预期位置初始化新空库: {db_path.absolute()}"
            )
        else:
            self.logger.info(f"💾 [Startup] 底层数据库安全对接成功: {db_path.name}")

        # 确认存在后，再安全连接
        self.con = duckdb.connect(database=str(db_path))
        self._connection_active = True
        self.data_manifest = manifest

        self._setup_db_macros()

        # ✨ 如果是全新的数据库，在此处执行数据结构/数仓初始化
        if is_new_db:
            self._init_empty_database()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _init_empty_database(self):
        """
        初始化数据库基础元数据表。
        """
        self.logger.info("⚡ [Schema] 正在初始化数据库基础架构...")
        try:
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS t_db_metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.con.execute(
                "INSERT INTO t_db_metadata (key, value) VALUES ('db_version', '1.0.0');"
            )

            self.logger.info("✅ [Schema] 数据库初始化完毕。")
        except Exception as e:
            self.logger.error(f"❌ [Schema] 初始化数据库基础表结构失败: {str(e)}")
            raise e

    # --- 引导与同步逻辑 ---

    def register_view_from_sql(self, view_name, sql_query):
        """将 SQL 查询逻辑注册为视图。这是 Actions 类的核心支撑方法。"""
        try:
            self.con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql_query}")
            self.logger.debug(f"✅ [Registry] 逻辑视图注册成功: {view_name}")
        except Exception as e:
            self.logger.error(f"❌ [Registry] 注册视图 {view_name} 失败: {str(e)}")
            raise e

    def _get_physics_fields(self, view_name):
        """
        从元数据中提取需要数值化的物理字段。
        """
        config = self.data_manifest.get(view_name, {})
        fields = config.get("fields", {})
        # 定义不需要转为数值的“黑名单”角色
        non_numeric_roles = {"id", "id_dr2", "cluster"}

        return [
            orig_name
            for std_key, orig_name in fields.items()
            if std_key not in non_numeric_roles
        ]

    def _load_and_merge_local_files(self, file_list):
        """加载并合并本地天文常用格式文件 (FITS/CSV/Parquet/VOTable)。"""
        if not file_list:
            return pd.DataFrame()

        dfs = []
        for f in file_list:
            if f.endswith(".fits"):
                from astropy.table import Table

                dfs.append(Table.read(f).to_pandas())
            elif f.endswith(".csv"):
                dfs.append(pd.read_csv(f))
            elif f.endswith(".parquet"):
                dfs.append(pd.read_parquet(f))
            elif f.endswith(".vot"):
                from astropy.table import Table

                # 使用 astropy 核心解析器读取虚拟天文台标准格式
                table_vo = Table.read(f, format="votable")
                dfs.append(table_vo.to_pandas())

        return pd.concat(dfs, ignore_index=True)

    def _standardize_dataframe(self, df, numeric_cols=None):
        """
        通用数据标准化清洗：解码字节码、处理掩码。
        如果提供了 numeric_cols，则强制执行数值转换。
        """
        if df.empty:
            return df

        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].apply(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )
                except Exception as e:
                    # 如果该 object 列包含无法 decode 的其他复杂对象，安全跳过
                    pass

            # 额外防线：把字符类型的 None、NaN 或空掩码统一转换为标准字符串空值，规避后续类型冲突
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna("").astype(str).str.strip()

            # 强制数值转换 (修正原代码参数失效问题)
            if numeric_cols and col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _get_enrichment_sql(self, v_src, t_base, threshold=None):
        """[私有] 构造数据增强 (JOIN Gaia 物理参数) 的通用 SQL。"""
        id_col = STD_COLS["ID"]
        prob_col = STD_COLS["PROB"]
        where_clause = f"WHERE res.{prob_col} > {threshold}" if threshold is not None else ""
        return f"""
            SELECT res.*, phys.* EXCLUDE ({id_col}) 
            FROM {v_src} AS res 
            JOIN {t_base} AS phys ON res.{id_col} = phys.{id_col}
            -- {where_clause}
        """

    def enrich_with_gaia_data(self, v_target, t_base, needed_fields=None):
        """自动补充 Gaia 物理参数并返回 DataFrame。"""
        if self.get_row_count(v_target) == 0:
            self.logger.warning(f"⚠️ [Compute] {v_target} 为空，跳过增强步骤。")
            return pd.DataFrame()

        sql = self._get_enrichment_sql(v_target, t_base)
        self.logger.info(f"🧪 [Compute] 正在从 {t_base} 为视图 {v_target} 补充物理参数...")
        res_df = self.query(sql)

        if needed_fields:
            actual_cols = res_df.columns.tolist()
            for f in needed_fields:
                if f not in actual_cols:
                    self.logger.error(f"❌ [Compute] 数据增强失败：表 {t_base} 中未找到字段 {f}")
                    raise KeyError(f"Missing field: {f}")
        return res_df

    def register_audit_input_view(self, v_src, t_base, threshold=None):
        """准备增强后的审计输入视图 (含物理参数补全与概率过滤)。"""
        threshold = threshold if threshold is not None else cfg.MEMBER_SAMPLE_THRESHOLD
        v_audit_input = cfg.TMPL.V_ADT_INPUT.format(src=v_src)

        sql = self._get_enrichment_sql(v_src, t_base, threshold=threshold)
        self.logger.debug(f"📋 [Registry] 正在注册审计输入视图: {v_audit_input} (源: {v_src})")
        self.register_view_from_sql(v_audit_input, sql)

        count = self.get_row_count(v_audit_input)
        self.logger.info(f"✅ [Registry] 审计输入视图 [{v_audit_input}] 准备就绪 (记录数: {count:,})")
        return v_audit_input

    def _execute_sync_task(self, task_cfg, result_path):
        """内部同步分发器：支持 local_file, vizier, gaia"""
        v_result = task_cfg.get("raw_table", "unknown_view")
        self.logger.info(
            f"📡 [Network] 执行同步任务: {v_result} (Provider: {task_cfg.get('provider')})"
        )
        provider = task_cfg.get("provider")
        params = task_cfg.get("params", {})

        physics_cols = self._get_physics_fields(v_result)

        if not provider:
            raise ValueError("Task 必须包含 'provider' 字段")

        # 1. 本地分片文件合并
        if provider == "local_file":
            pattern = params.get("file_pattern")
            # 使用 Path.rglob 进行更优雅的递归搜索
            matches = [str(p) for p in self.dirs["raw"].rglob(pattern)]
            if not matches:
                raise FileNotFoundError(f"❌ [Network] 在 raw 子目录下未找到匹配文件: {pattern}")
            matches.sort()
            self.logger.info(f"📚 [Network] 匹配到 {len(matches)} 个分片，准备合并...")
            df = self._load_and_merge_local_files(matches)

        # 2. Vizier 服务
        elif provider == "vizier":
            from astroquery.vizier import Vizier

            full_catalog = (
                f"{task_cfg['remote_catalog_id']}/{task_cfg['remote_table_id']}"
            )

            criteria = {
                k: v
                for k, v in params.items()
                if k not in ["file_pattern", "storage_path"]
            }

            v = Vizier(row_limit=-1, columns=["**"])
            res = v.query_constraints(catalog=full_catalog, **criteria)
            if res:
                df = res[0].to_pandas()
            else:
                self.logger.error(
                    f"🌌 [Network] Vizier 数据抓取失败. full_catalog: {full_catalog}; criteria: {criteria}"
                )

        # 3. Gaia Archive ADQL 查询
        elif provider == "gaia":
            from astroquery.gaia import Gaia

            query = params.get("query")
            if not query:
                raise ValueError("Gaia 任务需提供 'query' 参数。")

            self.logger.info(f"📡 [Network] 正在向 Gaia Archive 发送 ADQL 查询...")
            job = Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()
            # physics_cols = self._get_physics_fields(task['std_view'])
            # self.logger.info(f"🌌 Gaia 数据已落地: {target_path}")

        elif provider == "simbad":
            df = self._fetch_simbad_data(params)

        df = self._standardize_dataframe(df, numeric_cols=physics_cols)
        # 转存到 'snapshots' 目录
        df.to_parquet(result_path, index=False)
        self.logger.info(f"💾 [Network] 远程/外部同步完成，已固化: {result_path.name}")

    # --- 注册接口 ---

    def register_view_from_file(self, view_name, file_path):
        """从本地文件注册视图。"""
        abs_path = Path(file_path).resolve().as_posix()
        self.con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM '{abs_path}'"
        )
        self.logger.debug(f"✅ [Registry] 文件视图已挂载: {view_name}")

    def register_table_from_df(self, table_name, df):
        """将 DataFrame 物化为 DuckDB 物理表 (支持 UPSERT)"""
        self.con.register("_tmp_df", df)
        self.con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df"
        )
        self.con.execute("DROP VIEW IF EXISTS _tmp_df")
        self.logger.debug(f"✅ [Registry] 已物化物理表: {table_name}")

    def register_view_from_df(self, view_name, df):
        """将 DataFrame 注册为临时视图 (不产生物理拷贝)"""
        self.con.register(view_name, df)
        self.logger.debug(f"✅ [Registry] 已注册内存视图: {view_name}")

    def save_to_warehouse(self, table_or_view, storage_type="snapshots", filename=None):
        """将表或视图持久化为 Parquet 文件。"""
        sub_dir = self.dirs["warehouse"] / storage_type
        sub_dir.mkdir(parents=True, exist_ok=True)

        fname = filename if filename else table_or_view
        path = (sub_dir / f"{fname}.parquet").resolve()

        try:
            # 方式 A：如果数据量巨大，仍想用 COPY TO，请先 unlink 目标
            if path.exists():
                path.unlink()
            
            # 方式 B（推荐）：使用 Pandas/PyArrow 写入
            df = self.query(f"SELECT * FROM {table_or_view}")
            df.to_parquet(path, index=False)
            
            self.logger.info(f"💾 [Storage] 已将资产 {table_or_view} 固化至: {path.name}")
        except Exception as e:
            self.logger.error(f"❌ [Storage] 固化资产 {table_or_view} 失败: {e}")
            
        return path

    def export_table(self, table_name, filename=None, format="fits", export_dir=cfg.EXPORT_DIR):
        """导出表为指定格式。"""
        output_name = filename if filename else table_name
        output_path = Path(export_dir) / f"{output_name}.{format}"

        # 确保导出目录及其父目录存在，防止 DuckDB 报错找不到路径
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "fits":
            from astropy.table import Table

            df = self.query(f"SELECT * FROM {table_name}")
            Table.from_pandas(df).write(str(output_path), overwrite=True)
        else:
            self.con.execute(
                f"COPY {table_name} TO '{output_path.as_posix()}' (FORMAT {format.upper()})"
            )
        self.logger.info(f"💾 [Storage] 导出资产: {output_path.name} ({format.upper()}) -> {output_path.parent}")

    def batch_export(self, table_names, export_dir=cfg.EXPORT_DIR):
        self.logger.info(f"📦 [Storage] 启动批量导出任务，目标目录: {export_dir}")
        for name in table_names:
            self.export_table(name, export_dir=export_dir)

    def close(self):
        if hasattr(self, "con") and self._connection_active:
            self.con.close()
            self._connection_active = False
            self.logger.info("🔌 [System] AstroDB 数据库连接已安全关闭。")

    def query(self, sql_query):
        """执行 SQL 并返回 Pandas DataFrame。"""
        try:
            return self.con.sql(sql_query).df()
        except Exception as e:
            self.logger.error(f"🔍 [Query] 查询执行失败: {e}\nSQL: {sql_query}")
            return pd.DataFrame()

    def execute(self, sql):
        """通用 SQL 执行入口。"""
        try:
            self.logger.debug(f"🏗️ [Schema] Executing SQL: {sql}")
            return self.con.execute(sql)
        except Exception as e:
            self.logger.error(f"❌ [Schema] SQL Execution Failed: {e}")
            self.logger.error(f"Failed SQL: {sql}")
            raise e

    def list_resources(self):
        """打印当前库中所有表和视图。"""
        # 获取所有物理表和视图
        df_tables = self.con.execute("SHOW TABLES").df()
        if df_tables.empty:
            self.logger.info("📁 [Query] 当前数据库为空。")
            return

        self.logger.info(
            f"📁 [Query] 当前数据库资源清单 (共 {len(df_tables)} 个):\n{df_tables}"
        )
        return df_tables

    def get_table_schema(self, table_name):
        """获取表结构。"""
        try:
            schema = self.con.execute(f"DESCRIBE {table_name}").df()
            self.logger.info(
                f"📊 [Query] 表 {table_name} 的字段结构:\n{schema[['column_name', 'column_type']]}"
            )
        except Exception as e:
            self.logger.error(f"❌ [Query] 无法获取表 {table_name} 的结构: {e}")

    # def _setup_spatial_macros(self):
    #     """注册球面距离计算宏 (Haversine Formula)。"""
    #     self.logger.info("📐 正在注册空间计算宏: haversine_distance (单位: Degree)")

    #     # 针对天文学应用，直接返回度数（Degree）是最合理的
    #     sql = """
    #     CREATE OR REPLACE MACRO haversine_distance(ra1, dec1, ra2, dec2) AS (
    #         DEGREES(2 * ASIN(SQRT(
    #             POW(SIN(RADIANS(dec2 - dec1) / 2), 2) +
    #             COS(RADIANS(dec1)) * COS(RADIANS(dec2)) *
    #             POW(SIN(RADIANS(ra2 - ra1) / 2), 2)
    #         )))
    #     );
    #     """
    #     self.con.execute(sql)

    def _setup_db_macros(self):
        """注册天文学相关的计算宏（空间距离与色余修正）。"""
        self.logger.info("📐 [Schema] 正在注册空间计算宏: haversine_distance (单位: Degree)")
        self.logger.info("✨ [Schema] 正在注册色余与测光误差修正宏: calc_corrected_color_excess, calc_corrected_color_excess_sigma, calc_e_color")

        sql = """
        -- 1. 球面距离计算宏 (Haversine Formula)
        CREATE OR REPLACE MACRO haversine_distance(ra1, dec1, ra2, dec2) AS (
            DEGREES(2 * ASIN(SQRT(
                POW(SIN(RADIANS(dec2 - dec1) / 2), 2) +
                COS(RADIANS(dec1)) * COS(RADIANS(dec2)) *
                POW(SIN(RADIANS(ra2 - ra1) / 2), 2)
            )))
        );

        -- 2. 修正后的色余计算宏
        CREATE OR REPLACE MACRO calc_corrected_color_excess(formal_color_excess) AS (
            CASE 
                WHEN formal_color_excess < 0.5 
                    THEN formal_color_excess - 1.154360 + 0.033772 * formal_color_excess + 0.032277 * POW(formal_color_excess, 2)
                WHEN formal_color_excess >= 0.5 AND formal_color_excess < 4.0 
                    THEN formal_color_excess - 1.162004 + 0.011464 * formal_color_excess + 0.049255 * POW(formal_color_excess, 2) - 0.005879 * POW(formal_color_excess, 3)
                WHEN formal_color_excess >= 4.0 
                    THEN formal_color_excess - 1.057572 + 0.260015 * formal_color_excess - 0.049302 * POW(formal_color_excess, 2) + 0.002879 * POW(formal_color_excess, 3)
                ELSE NULL
            END
        );

        -- 3. 色余标准差计算宏
        CREATE OR REPLACE MACRO calc_corrected_color_excess_sigma(gmag) AS (
            0.0059898 + 8.817481e-12 * POW(gmag, 7.618399)
        );

        -- 4. 颜色综合误差计算宏
        CREATE OR REPLACE MACRO calc_e_color(e_bpmag, e_rpmag) AS (
            SQRT(POW(e_bpmag, 2) + POW(e_rpmag, 2))
        );
        """
        
        # DuckDB 支持在单个 execute() 中执行多条以分号分隔的 SQL 语句
        self.con.execute(sql)

    def get_row_count(self, name, column=None):
        """

        Args:
        Returns:
            int: 记录数量
        """
        try:
            target = f"COUNT(DISTINCT {column})" if column else "COUNT(*)"
            sql = f"SELECT {target} FROM {name}"
            result = self.con.execute(sql).fetchone()
            count = result[0] if result else 0
            return count
        except Exception as e:
            self.logger.error(f"❌ [Query] 无法获取 [{name}] 的计数: {e}")
            return 0

    def table_exists(self, name):
        """检查表或视图是否存在。"""
        try:
            sql = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{name}'"
            res = self.con.execute(sql).fetchone()
            return res[0] > 0 if res else False
        except Exception:
            return False

    def _rotate_backups(self, file_path: Path, limit: int = 3, is_pre_write: bool = True):
        """
        备份文件并循环保留最近 N 次执行的数据。
        
        Args:
            file_path (Path): 要备份的主文件路径。
            limit (int): 保留的备份文件数量。
            is_pre_write (bool): 如果为 True，表示在写入新文件前进行备份（备份当前有效文件）。
                                 如果为 False，表示在写入新文件后，将新文件作为最新备份。
        """
        if not file_path.exists():
            return

        # 删除最旧的备份
        oldest_bak = file_path.parent / f"{file_path.stem}.bak{limit}{file_path.suffix}"
        if oldest_bak.exists():
            oldest_bak.unlink()

        # 滚动现有备份
        for i in range(limit - 1, 0, -1): # 从 limit-1 到 1
            src = file_path.parent / f"{file_path.stem}.bak{i}{file_path.suffix}"
            dst = file_path.parent / f"{file_path.stem}.bak{i+1}{file_path.suffix}"
            if src.exists():
                src.rename(dst)

    def drop_table(self, name):
        """删除物理表（如果存在）。"""
        try:
            self.con.execute(f"DROP TABLE IF EXISTS {name}")
            self.logger.debug(f"✅ [Schema] 已删除物理表: {name}")
        except Exception as e:
            self.logger.error(f"❌ [Schema] 删除表 {name} 失败: {e}")

    def drop_view(self, name):
        """删除视图（如果存在）。"""
        try:
            self.con.execute(f"DROP VIEW IF EXISTS {name}")
            self.logger.debug(f"✅ [Schema] 已删除视图: {name}")
        except Exception as e:
            self.logger.error(f"❌ [Schema] 删除视图 {name} 失败: {e}")

    def query_subset(self, table_name, where_clause=None):
        """查询子集并确保 ID 精度。"""
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"

        df = pd.read_sql(query, self.con)
        _id = STD_COLS["ID"]
        if _id in df.columns:
            df[_id] = df[_id].astype("Int64")
        return df

    def import_raw(self, target_cluster=None, force=False):
        if not self.data_manifest:
            self.logger.warning("⚠️ [Startup] 未检测到 Data Manifest。")
            return

        self.logger.info(f"🚀 [Startup] 开始引导 AstroDB 数据环境 (星团: {target_cluster or 'ALL'})...")

        # ── 构建过滤白名单 ──
        # 当指定 target_cluster 时，只处理以下 MANIFEST 条目：
        #   1. 目标星团的 FIELD_IDX / SEED_IDX
        #   2. 不属于任何活跃星团的共享条目（参考星表、桥接表等）
        target_indices = set()
        all_active_cluster_indices = set()
        if target_cluster and target_cluster in cfg.CLUSTERS:
            target_indices.add(cfg.CLUSTERS[target_cluster].get("FIELD_IDX"))
            target_indices.add(cfg.CLUSTERS[target_cluster].get("SEED_IDX"))
        for cid in cfg.CLUSTERS:
            all_active_cluster_indices.add(cfg.CLUSTERS[cid].get("FIELD_IDX"))
            all_active_cluster_indices.add(cfg.CLUSTERS[cid].get("SEED_IDX"))

        for k, config in self.data_manifest.items():
            # 指定了目标星团时：跳过属于其他活跃星团的数据源
            if target_cluster and k in all_active_cluster_indices and k not in target_indices:
                continue

            mode = config.get("sync_mode", "HYBRID")

            # [核心重构]：如果模式是 VIRTUAL，说明该项是一个逻辑视图（如种子集），
            # 它直接引用 base_idx 对应的 raw 表，不需要物理文件同步。
            if mode == "VIRTUAL":
                self.logger.info(f"🌌 [Startup] 跳过虚拟同步项: {k} (将作为逻辑视图处理)")
                continue

            t_raw = config.get("raw_table")  # 从 manifest 获取表名
            params = config.get("params", {})
            storage_path = params.get("storage_path", "snapshots")

            result_path = self.dirs["warehouse"] / storage_path / f"{k}.parquet"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = result_path.exists()
            table_exists = self.table_exists(t_raw)

            should_sync = False
            if mode == "FORCE_REMOTE":
                should_sync = True
            elif mode == "HYBRID" and not file_exists:
                # 标记为 optional 的数据源（如 SIMBAD 缓存）首次缺失属正常
                if params.get("optional"):
                    self.logger.info(
                        f"ℹ️ [Startup] 可选数据源 {k} 尚不存在，首次网络查询时将自动创建。"
                    )
                    continue
                should_sync = True
            elif mode == "OFFLINE" and not file_exists:
                self.logger.error(f"❌ [Startup] 离线任务缺失物理文件: {k} (路径: {result_path})")
                continue

            if should_sync:
                self.logger.info(f"🔄 [Network] 正在下载并转换数据: {k} -> {result_path.name}")
                try:
                    self._execute_sync_task(config, result_path)
                    file_exists = True  # 同步成功后更新状态
                except Exception as e:
                    # 指定了目标星团时，非目标条目的同步失败降级为 WARNING
                    if target_cluster and k not in target_indices:
                        self.logger.warning(f"⚠️ [Network] 非目标条目同步失败 (可忽略): {k} - {e}")
                    else:
                        self.logger.error(f"❌ [Network] 同步 {k} 失败: {e}")
                    continue
            else:
                self.logger.info(f"✅ [Local] 数据文件 {k}.parquet 已存在，跳过同步。")

            # 从 .parquet 中注册数据库
            if force or not table_exists:
                self.logger.info(f"📋 [Registry] 正在注册数据库表: {t_raw}")
                calc_custom_cols = False
                # str = f"raw_{target_cluster_id}_field".lower()
                # self.logger.info(f"🔍 检查是否需要计算自定义列: {str} == {t_raw} ?")
                if t_raw.lower() == f"raw_{target_cluster}_field".lower() and target_cluster:
                    calc_custom_cols = True    
                self.register_table_from_file(t_raw, result_path, calc_custom_cols=calc_custom_cols)
            else:
                self.logger.info(f"✅ [Registry] 表 {t_raw} 已在内存中就绪。")

        self.logger.info("✨ [Startup] AstroDB L1 原始数据环境导入完成。")

    def register_table_from_file(self, table_name, file_path, calc_custom_cols=False):
        """将 Parquet 文件物化为 DuckDB 物理表。"""
        abs_path = Path(file_path).resolve().as_posix()
        try:
            if calc_custom_cols:
                sql = f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT *, 
                    calc_corrected_color_excess(color_excess) AS corrected_color_excess,
                    calc_corrected_color_excess_sigma(gmag) AS corrected_color_excess_sigma,
                    calc_e_color(e_bpmag, e_rpmag) AS color_err 
                    FROM read_parquet('{abs_path}')
                    """
                # 仅在首次注册时添加自定义列
                self.con.execute(sql)
                self.logger.debug(f"🧪 [Compute] 已物化物理表(含自定义列): {table_name}")
            else:
                self.con.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{abs_path}')"
                )
            count = self.get_row_count(table_name)
            self.logger.info(f"📦 [Registry] 已物化物理表: {table_name} (行数: {count:,})")
            self.logger.debug(f"📦 [Registry] 源文件: {abs_path}")
        except Exception as e:
            self.logger.error(f"❌ [Registry] 物化物理表 {table_name} 失败: {e}")

    def init_master_table(self, table_name, df_base):
        """初始化 Master 状态宽表。"""
        self.logger.info(f"🏗️  [Master] 初始化状态追踪表: {table_name}")
        df_init = df_base[[cfg.STD_COLS['ID']]].copy()
        self.register_table_from_df(table_name, df_init)

        # 🚀 [核心修复] 定义 Master 表字段的物理类型映射
        # 默认全部为 VARCHAR，但算法概率列必须为 DOUBLE 以支持后续的数值比较和逻辑运算
        type_map = {
            cfg.MASTER_COLS['GMM_PROB']: "DOUBLE",
        }

        # 预添加标准状态列
        for col in cfg.MASTER_COLS.values():
            col_type = type_map.get(col, "VARCHAR")
            self.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")

    def tag_master_table(self, master_name, df_updates, key_col='id'):
        """ 执行增量标签/结果回灌。"""
        if df_updates is None or df_updates.empty:
            return

        # 🚀 获取目标表已有的列，仅更新匹配的列，防止物理参数（如 ra/dec）被误灌回 Master 表导致报错
        dest_cols = self.con.execute(f"PRAGMA table_info('{master_name}')").df()['name'].tolist()

        update_cols = [c for c in df_updates.columns if c != key_col and c in dest_cols]

        if not update_cols:
            self.logger.info(f"⚠️ [Master] {master_name} 没有匹配的列需要更新 (跳过)")
            return

        temp_name = f"tmp_tag_{int(time.time())}_{random.randint(0, 1000)}"
        self.register_view_from_df(temp_name, df_updates)

        set_clause = ", ".join([f"{c} = src.{c}" for c in update_cols])

        sql = f"""
            UPDATE {master_name} AS dest
            SET {set_clause}
            FROM {temp_name} AS src
            WHERE dest.{key_col} = src.{key_col}
        """
        try:
            self.execute(sql)
            self.logger.info(f"✅ [Master] 成功回灌 {len(df_updates)} 条状态至 {master_name}")
        finally:
            self.con.unregister(temp_name)

    def _fetch_simbad_data(self, params):
        """
        从 SIMBAD 批量获取天体别名。
        params 需包含:
            - id_list_sql: 从哪个内部表提取原始 ID 的 SQL
            - id_col: ID 所在的列名
            - prefix: (可选) 补全前缀，如 'Gaia DR3 '
        """
        id_sql = params.get("id_list_sql") or ""
        id_col = params.get("id_col", "id")
        prefix = params.get("prefix", "")

        # 核心逻辑：如果 SQL 执行失败（比如表还没创建）或者没有定义 SQL，返回空 Schema
        try:
            if not id_sql:
                raise ValueError("No SQL")
            target_ids = self.con.execute(id_sql).df()[id_col].tolist()
        except Exception:
            self.logger.warning("⚠️ [Network] Simbad 依赖源未就绪或未定义，将初始化空缓存表。")
            return pd.DataFrame(columns=[id_col, "main_id", "ids"])

        formatted_ids = [f"{prefix}{idx}" for idx in target_ids]
        if not formatted_ids:
            return pd.DataFrame(columns=[id_col, "main_id", "ids"])

        self.logger.info(f"📡 [Network] 正在从 SIMBAD 查询 {len(formatted_ids)} 个天体的别名...")

        Simbad.reset_votable_fields()
        Simbad.add_votable_fields("ids")

        result_table = Simbad.query_objects(formatted_ids)

        if result_table is None:
            self.logger.warning("⚠️ [Network] SIMBAD 未返回任何匹配数据。")
            return pd.DataFrame(columns=[id_col, "ids"])

        df = result_table.to_pandas()
        if "MAIN_ID" in df.columns:
            df[id_col] = target_ids

        return df

    # --- SIMBAD 网络工具 ---

    def _bypass_ssl_verification(self):
        """
        SSL 校验绕过补丁，确保内网环境下的 TAP 服务连接。
        """
        try:
            import ssl
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
            _create_unverified_https_context = ssl._create_unverified_context
        except (AttributeError, ImportError):
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
            self.logger.debug("🌐 [System] SIMBAD SSL 校验已绕过。")

    def query_simbad_target_info(self, gaia_dr3_id: str) -> dict:
        """
        查询指定天体的完整别名与元数据。
        """
        self._bypass_ssl_verification()
        self.logger.info(f"🌐 [Network] 正在从远端 CDS SIMBAD 检索单个天体: {gaia_dr3_id} ...")

        Simbad.reset_votable_fields()
        Simbad.add_votable_fields("ids")

        try:
            result_table = Simbad.query_object(gaia_dr3_id)
            if result_table is None:
                return None

            df = result_table.to_pandas()
            df = self._standardize_dataframe(df)

            target_dict = df.to_dict(orient="records")[0]
            return target_dict

        except Exception as e:
            self.logger.error(
                f"❌ [Network] 从 SIMBAD 检索单星数据时发生网络或解析异常: {str(e)}"
            )
            return None

    # =========================================================================
    # 🌟 高性能批量跨网络与本地数据库同步服务
    # =========================================================================

    def sync_simbad_cache(
        self, source_ids, cache_table_name, prefix="Gaia DR3 ", chunk_size: int = 500
    ) -> pd.DataFrame:
        """
        🚀 [核心接口] 跨网络与本地数据库同步 SIMBAD 缓存 (支持 Parent 自动补全)。
        已针对 Windows 环境下的文件 IO 进行了健壮性加固。
        """
        import shutil
        import os

        # 1. 动态 DDL 升维防御 (确保 parent 列存在)
        if self.table_exists(cache_table_name):
            col_info = self.con.execute(f"PRAGMA table_info({cache_table_name})").df()
            if "parent" not in col_info["name"].values:
                self.logger.warning(f"🏗️ [Schema] 发现本地缓存表 `{cache_table_name}` 缺失 `parent` 字段，正在追加...")
                try:
                    self.con.execute(f"ALTER TABLE {cache_table_name} ADD COLUMN parent VARCHAR;")
                except Exception as ddl_err:
                    self.logger.error(f"❌ [Schema] 动态追加 parent 列失败: {str(ddl_err)}")

        df_input = self._normalize_input_ids(source_ids)
        self.con.register("temp_sync_input", df_input)

        # 2. 创建缓存表（如果不存在）
        self.execute(f"""
            CREATE TABLE IF NOT EXISTS {cache_table_name} (
                gaia_dr3_id VARCHAR PRIMARY KEY,
                main_id VARCHAR,
                ids VARCHAR,
                parent VARCHAR
            )
        """)

        # 3. 检查本地命中情况
        df_cached = self.con.execute(f"""
            SELECT i.gaia_dr3_id, c.main_id, c.ids, c.parent
            FROM temp_sync_input i
            JOIN {cache_table_name} c ON i.gaia_dr3_id = c.gaia_dr3_id
        """).df()

        # 识别需要补全 parent 的行
        mask_needs_repair = df_cached['parent'].isna() | \
                            (df_cached['parent'] == '') | \
                            (df_cached['parent'].str.lower() == 'none')
        
        df_valid_cached = df_cached[~mask_needs_repair].copy()
        df_valid_cached["cache_hit"] = True
        
        # 4. 获取需要在线同步或修复的 ID 列表
        df_missing = self.con.execute(f"""
            SELECT DISTINCT i.gaia_dr3_id
            FROM temp_sync_input i
            ANTI JOIN {cache_table_name} c ON i.gaia_dr3_id = c.gaia_dr3_id
        """).df()
        
        ids_to_repair = df_cached.loc[mask_needs_repair, "gaia_dr3_id"].tolist()
        ids_to_fetch = df_missing["gaia_dr3_id"].tolist() + ids_to_repair
        
        self.con.unregister("temp_sync_input")

        # 5. 执行网络同步
        df_online_results = pd.DataFrame(columns=["gaia_dr3_id", "main_id", "ids", "parent", "cache_hit"])
        if ids_to_fetch:
            self.logger.info(f"📡 [Network] 检测到 {len(ids_to_fetch)} 个源需要在线同步/修复。")
            df_online_results = self._perform_online_sync(ids_to_fetch, cache_table_name, prefix, chunk_size)
            df_online_results["cache_hit"] = False
        else:
            self.logger.info("✅ [SimbadCache] 所有请求均在本地缓存命中。")

        # 6. 合并结果
        df_final_merged = pd.concat([df_valid_cached, df_online_results], ignore_index=True)

        self.logger.info(
            f"🎯 [SimbadCache] 内存同步完成: "
            f"命中有效缓存 {len(df_valid_cached)} | "
            f"修复/新增同步 {len(df_online_results)} 颗"
        )

        # 7. 🌟 物理文件同步：加固后的文件保存逻辑
        if not df_online_results.empty or not df_valid_cached.empty:
            simbad_cfg = self.data_manifest.get(cfg.IDX_IDS_SIMBAD, {})
            rel_path = simbad_cfg.get("params", {}).get("file_pattern")
            
            if rel_path:
                source_file = self.dirs["raw"] / rel_path
                try:
                    # 确保文件夹存在
                    source_file.parent.mkdir(parents=True, exist_ok=True)

                    # A. 备份当前旧文件
                    if source_file.exists():
                        self._rotate_backups(source_file)

                    # B. 生成临时文件 (使用 Pandas 写入，绕过 DuckDB COPY 限制)
                    temp_file = source_file.parent / f"{source_file.stem}_tmp{source_file.suffix}"
                    
                    # 清理可能残留的临时文件
                    if temp_file.exists():
                        os.remove(temp_file)

                    # 从数据库拉取最新全量缓存
                    df_all_cache = self.con.execute(f"SELECT * FROM {cache_table_name}").df()
                    df_all_cache.to_parquet(temp_file, index=False)

                    # C. 原子化替换文件
                    if temp_file.exists():
                        # 在 Windows 上，显式删除目标文件后再 rename 是最稳妥的
                        if source_file.exists():
                            os.remove(source_file)
                        shutil.move(str(temp_file), str(source_file))
                        self.logger.info(f"💾 [Storage] 已同步更新 SIMBAD 原始数据源并保留备份: {source_file.name}")
                    else:
                        self.logger.error(f"❌ [Storage] 写入失败：临时文件 {temp_file} 未能生成。")

                except Exception as io_err:
                    self.logger.error(f"❌ [Storage] 同步到物理文件时发生错误: {str(io_err)}")

            # 2. 更新数仓快照
            try:
                self.save_to_warehouse(cache_table_name, storage_type="snapshots", filename=cfg.IDX_IDS_SIMBAD)
            except Exception as e:
                self.logger.error(f"❌ [Storage] 更新仓库快照失败: {e}")

        return df_final_merged

    def _normalize_input_ids(self, source_ids) -> pd.DataFrame:
        """[私有方法] 规范化输入 ID。"""
        if isinstance(source_ids, (int, str)):
            ids = pd.Series([str(source_ids)])
        else:
            ids = pd.Series(source_ids).astype(str).unique()
        return pd.DataFrame({"gaia_dr3_id": ids})

    def _perform_online_sync(self, ids_missing: list, table_name: str, prefix: str, chunk_size: int) -> pd.DataFrame:
        """[私有方法] 分批次从远程 SIMBAD 同步数据并回灌缓存。"""
        self.logger.info(f"🌐 [Network] 正在从 CDS 增量抓取 {len(ids_missing)} 个源...")
        self._bypass_ssl_verification()

        online_records = []
        for i in range(0, len(ids_missing), chunk_size):
            if self._check_user_interrupt():
                self.logger.warning("🛑 [Network] 接收到人工中断信号，正在保存已完成批次...")
                break

            chunk_ids = ids_missing[i : i + chunk_size]
            batch_results = self._fetch_simbad_batch(chunk_ids, prefix)

            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                self._save_local_cache_incremental(df_batch, table_name)
                online_records.extend(batch_results)
                self.logger.info(f"   ∟ [Network] 进度: {len(online_records)}/{len(ids_missing)}")

            if i + chunk_size < len(ids_missing):
                time.sleep(random.uniform(0.5, 1.5))

        df_online = pd.DataFrame(online_records)
        if not df_online.empty:
            df_online["cache_hit"] = False
        else:
            # 🌟 确保存储列结构即使为空时也能完美对齐下游合并
            df_online = pd.DataFrame(columns=["gaia_dr3_id", "main_id", "ids", "parent", "cache_hit"])
        return df_online

    def _fetch_simbad_batch(self, chunk_ids: list, prefix: str) -> list:
        """执行批次网络请求，并实时调用 query_hierarchy 补全父节点。"""
        query_names = [f"{prefix}{mid}" for mid in chunk_ids]
        # 匹配 Gaia ID 的正则表达式，用于从 SIMBAD 的 user_specified_id 回溯
        id_pattern = re.compile(r"(\d+)$")

        batch_results = []

        try:
            Simbad.reset_votable_fields()
            Simbad.add_votable_fields("ids")
            # 注意：此处不再试图通过 query_objects 获取 parents，因为它不支持
            table = Simbad.query_objects(query_names)

            if table is not None:
                for row in table:
                    match = id_pattern.search(str(row["user_specified_id"]))
                    if not match: continue
                    gid = match.group(1)
                    
                    main_id = str(row["main_id"])
                    is_empty = (main_id.strip().upper() in ["NONE", ""])
                    ids = row["ids"].decode("utf-8") if isinstance(row["ids"], bytes) else str(row["ids"])
                    
                    # 🛠️ 【核心逻辑】：如果存在有效 main_id，实时查询家谱补全 parent
                    parent_val = "None"
                    if not is_empty:
                        try:
                            # 调用官方家谱查询接口
                            hierarchy = Simbad.query_hierarchy(main_id, hierarchy='parents', detailed_hierarchy=False)
                            if hierarchy is not None and len(hierarchy) > 0:
                                # 提取所有父节点名称并用 | 分隔
                                parent_list = [str(p['main_id']) for p in hierarchy]
                                parent_val = " | ".join(parent_list)
                                self.logger.info(f"🌟 [Network] 家谱查询成功 {main_id}: {parent_val}")
                        except Exception as sub_e:
                            self.logger.warning(f"⚠️ [Network] 家谱查询失败 {main_id}: {sub_e}")

                    batch_results.append({
                        "gaia_dr3_id": gid,
                        "main_id": "None" if is_empty else main_id,
                        "ids": "None" if is_empty else ids,
                        "parent": parent_val
                    })
            
            # 补全未命中的 ID
            found_ids = {res["gaia_dr3_id"] for res in batch_results}
            for mid in chunk_ids:
                if str(mid) not in found_ids:
                    batch_results.append({"gaia_dr3_id": str(mid), "main_id": "None", "ids": "None", "parent": "None"})

        except Exception as e:
            self.logger.error(f"❌ [Network] 网络请求失败: {str(e)}")

        return batch_results

    def _save_local_cache_incremental(self, df_new: pd.DataFrame, table_name: str):
        """增量更新数据库表。"""
        # 🛡️ 安全防御：确保存储前 DataFrame 结构中具有 parent 属性
        if "parent" not in df_new.columns:
            df_new["parent"] = "None"

        self.con.register("tmp_inc", df_new)
        # 修正：DuckDB 在没有明确 PRIMARY KEY/UNIQUE 约束时不支持 INSERT OR REPLACE。
        # 由于缓存表可能在 import_raw 阶段通过 Parquet 重新物化（导致约束丢失），此处采用 DELETE + INSERT 模式实现 UPSERT。
        self.execute(f"DELETE FROM {table_name} WHERE gaia_dr3_id IN (SELECT gaia_dr3_id FROM tmp_inc)")
        
        # 🌟 核心修改：显式指定目标表和数据源的列名对齐，防止因 SELECT * 造成字段顺序错位或缺失崩溃
        self.execute(f"""
            INSERT INTO {table_name} (gaia_dr3_id, main_id, ids, parent) 
            SELECT gaia_dr3_id, main_id, ids, parent FROM tmp_inc
        """)
        
        self.con.unregister("tmp_inc")

    def _check_user_interrupt(self) -> bool:
        """非阻塞检查用户中断 (仅限终端模式)。"""
        if not sys.stdin.isatty(): return False
        try:
            if os.name == 'nt':
                import msvcrt
                if msvcrt.kbhit(): return msvcrt.getch().lower() == b'q'
            else:
                import select
                if select.select([sys.stdin], [], [], 0)[0]: return sys.stdin.read(1).lower() == 'q'
        except:
            pass
        return False


class AssetManager:
    """
    专门负责 RAW 数据资产维护的管理器。
    与持久化数据库解耦，仅在需要时使用内存数据库进行元数据统计。
    """
    def __init__(self):
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

    def query_backup_assets(self):
        """查询并打印当前备份资产信息。"""
        meta_file = cfg.BACKUP_DIR / "metadata.json"
        if not meta_file.exists():
            print("\n[备份查询] ❌ 目前没有任何手动备份记录。")
            return

        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        print(f"\n{'='*105}")
        print(f"{'ID':<8} | {'备份时间':<20} | {'文件数':<8} | {'总记录数'}")
        print(f"{'-'*105}")
        for key in ['A', 'B']:
            info = meta.get(key)
            if info:
                print(f"备份 {key:<5} | {info['timestamp']:<20} | {info['file_count']:<8} | {info['record_count']:,}")
                
                # 打印详细的文件组成明细
                file_details = info.get("file_details", {})
                if file_details:
                    print(f"  ∟ 📂 数据资产清单 (RAW):")
                    # 按文件名排序，方便用户对照星团
                    for fname in sorted(file_details.keys()):
                        count = file_details[fname]
                        records_str = f"{count:,}" if count > 0 else "[不可读取/非表格]"
                        print(f"    - {fname:<72} | {records_str:>15} 记录")
                
                if info.get("db_included"):
                    print(f"  ∟ 🗄️ 包含内部数据库快照 (astrodb_internal.db)")
                print(f"{'-'*105}")
            else:
                print(f"备份 {key:<5} | {'无数据':<20} | {'-':<8} | -")
                print(f"{'-'*105}")
        print(f"{'='*105}\n")

    def manage_backup_assets(self):
        """执行资产手动备份。"""
        raw_dir = cfg.RAW_DIR
        db_file = cfg.DATA_DIR / "warehouse" / "astrodb_internal.db" # 定义 db_file
        backup_root = cfg.BACKUP_DIR
        backup_root.mkdir(parents=True, exist_ok=True)

        path_a, path_b = backup_root / "raw_backup_A", backup_root / "raw_backup_B"
        meta_file = backup_root / "metadata.json"

        meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except: meta = {}

        self.logger.info("📦 [Storage] 正在执行资产手动备份轮转 (含 RAW 数据与内部数据库)...")

        if path_b.exists(): shutil.rmtree(path_b)
        if path_a.exists():
            path_a.rename(path_b)
            meta['B'] = meta.get('A')
            self.logger.info("  ∟ [Storage] 已将原有 [备份A] 顺延至 [备份B]")

        # 创建 A 目录并建立子结构
        path_a.mkdir(parents=True)
        shutil.copytree(raw_dir, path_a / "raw")
        
        db_included = False
        if db_file.exists():
            shutil.copy2(db_file, path_a / "astrodb_internal.db")
            db_included = True
            self.logger.info(f"  🗄️ [Storage] 已同步备份内部数据库文件")
        
        file_details, total_records = {}, 0
        # 使用内存连接，避免触发项目数据库初始化
        with duckdb.connect(":memory:") as temp_con:
            for p in (path_a / "raw").rglob("*"):
                if not p.is_file():
                    continue
                
                rel_name = p.relative_to(path_a / "raw").as_posix()
                ext = p.suffix.lower()
                count = 0
                
                try:
                    if ext == ".parquet":
                        count = temp_con.execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0]
                    elif ext == ".csv":
                        # DuckDB 自动探测 CSV 结构并统计行数
                        count = temp_con.execute(f"SELECT count(*) FROM read_csv_auto('{p.as_posix()}')").fetchone()[0]
                    elif ext in [".fits", ".vot"]:
                        # 针对 FITS 和 VOTable，使用项目中已有的 astropy 逻辑
                        from astropy.table import Table
                        count = len(Table.read(p))
                    else:
                        # 非结构化数据文件，不统计记录数
                        count = 0
                    
                    file_details[rel_name] = count
                    total_records += count
                    self.logger.info(f"  📄 [Storage] 备份文件明细: {rel_name:<30} | 记录数: {count:,}")
                except Exception:
                    file_details[rel_name] = 0
                    self.logger.info(f"  📄 [Storage] 备份文件明细: {rel_name:<30} | 记录数: [不可读取]")

        meta['A'] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": total_records,
            "file_count": sum(1 for _ in path_a.rglob("*") if _.is_file()),
            "file_details": file_details,
            "db_included": db_included
        }

        temp_meta = meta_file.with_suffix(".tmp")
        with open(temp_meta, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        temp_meta.replace(meta_file)

        self.logger.info(f"✅ [Storage] 手动备份完成：[备份A] 已更新。总记录数(RAW): {total_records:,}")

    def restore_backup_assets(self, target='A'):
        """从备份恢复数据。"""
        target_dir = cfg.BACKUP_DIR / f"raw_backup_{target.upper()}"
        raw_dir = cfg.RAW_DIR
        db_file = cfg.DATA_DIR / "warehouse" / "astrodb_internal.db"

        if not target_dir.exists():
            self.logger.error(f"❌ [Storage] 恢复失败：指定的 [{target_dir.name}] 不存在")
            return False

        # 增加二次确认
        print(f"\n⚠️  警告: 正在从 [{target_dir.name}] 恢复数据!")
        print(f"这将会彻底删除当前的 {raw_dir} 目录以及内部数据库文件。")
        confirm = input("确定要继续吗? (y/N): ")
        if confirm.lower() != 'y':
            print("🚫 恢复操作已取消。")
            return False

        self.logger.warning(f"🔄 [Storage] 正在从 [{target_dir.name}] 强制覆盖恢复数据资产...")
        
        # 兼容性恢复逻辑：判断备份中是否含有 'raw' 子目录
        if (target_dir / "raw").exists():
            if raw_dir.exists(): shutil.rmtree(raw_dir)
            shutil.copytree(target_dir / "raw", raw_dir)
            
            db_backup = target_dir / "astrodb_internal.db"
            if db_backup.exists():
                db_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(db_backup, db_file)
                self.logger.info(f"  ∟ [Storage] 内部数据库文件恢复完成")
        else:
            # 处理旧版备份结构
            if raw_dir.exists(): shutil.rmtree(raw_dir)
            shutil.copytree(target_dir, raw_dir)

        self.logger.info(f"✅ [Storage] 数据恢复成功！来源: {target_dir.name}")
        return True