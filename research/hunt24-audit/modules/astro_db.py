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
            self.logger.warning(
                f"🆕 未检测到本地数据库，正在预期位置初始化新空库: {db_path.absolute()}"
            )
        else:
            self.logger.info(f"💾 底层数据库安全对接成功: {db_path}")

        # 确认存在后，再安全连接
        self.con = duckdb.connect(database=str(db_path))
        self._connection_active = True
        self.data_manifest = manifest

        self._setup_spatial_macros()

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
        self.logger.info("⚡ 正在初始化数据库基础架构...")
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

            self.logger.info("✅ 数据库初始化完毕。")
        except Exception as e:
            self.logger.error(f"❌ 初始化数据库基础表结构失败: {str(e)}")
            raise e

    # --- 引导与同步逻辑 ---

    def register_view_from_sql(self, view_name, sql_query):
        """将 SQL 查询逻辑注册为视图。这是 Actions 类的核心支撑方法。"""
        try:
            self.con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql_query}")
            self.logger.debug(f"✅ 逻辑视图注册成功: {view_name}")
        except Exception as e:
            self.logger.error(f"❌ 注册视图 {view_name} 失败: {str(e)}")
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
            {where_clause}
        """

    def enrich_with_gaia_data(self, v_target, t_base, needed_fields=None):
        """自动补充 Gaia 物理参数并返回 DataFrame。"""
        if self.get_row_count(v_target) == 0:
            self.logger.warning(f"{v_target} 为空，跳过增强步骤。")
            return pd.DataFrame()

        sql = self._get_enrichment_sql(v_target, t_base)
        self.logger.info(f"正在从 {t_base} 为视图 {v_target} 补充物理参数...")
        res_df = self.query(sql)

        if needed_fields:
            actual_cols = res_df.columns.tolist()
            for f in needed_fields:
                if f not in actual_cols:
                    self.logger.error(f"❌ 数据增强失败：表 {t_base} 中未找到字段 {f}")
                    raise KeyError(f"Missing field: {f}")
        return res_df

    def register_audit_input_view(self, v_src, t_base, threshold=None):
        """准备增强后的审计输入视图 (含物理参数补全与概率过滤)。"""
        threshold = threshold if threshold is not None else cfg.MEMBER_SAMPLE_THRESHOLD
        v_audit_input = f"v_audit_input_{v_src}"

        sql = self._get_enrichment_sql(v_src, t_base, threshold=threshold)
        self.register_view_from_sql(v_audit_input, sql)

        count = self.get_row_count(v_audit_input)
        self.logger.info(f"✅ 审计输入视图 [{v_audit_input}] 准备就绪 (记录数: {count})")
        return v_audit_input

    def _execute_sync_task(self, task_cfg, result_path):
        """内部同步分发器：支持 local_file, vizier, gaia"""
        v_result = task_cfg.get("raw_table", "unknown_view")
        self.logger.info(
            f"📡 执行同步任务: {v_result} (Provider: {task_cfg.get('provider')})"
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
                raise FileNotFoundError(f"在 raw 子目录下未找到匹配文件: {pattern}")
            matches.sort()
            self.logger.info(f"📚 匹配到 {len(matches)} 个分片，准备合并...")
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
                    f"🌌 Vizier 数据抓取失败. full_catalog: {full_catalog}; criteria: {criteria}"
                )

        # 3. Gaia Archive ADQL 查询
        elif provider == "gaia":
            from astroquery.gaia import Gaia

            query = params.get("query")
            if not query:
                raise ValueError("Gaia 任务需提供 'query' 参数。")

            job = Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()
            # physics_cols = self._get_physics_fields(task['std_view'])
            # self.logger.info(f"🌌 Gaia 数据已落地: {target_path}")

        elif provider == "simbad":
            df = self._fetch_simbad_data(params)

        df = self._standardize_dataframe(df, numeric_cols=physics_cols)
        df.to_parquet(result_path, index=False)

    # --- 注册接口 ---

    def register_view_from_file(self, view_name, file_path):
        """从本地文件注册视图。"""
        abs_path = Path(file_path).resolve().as_posix()
        self.con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM '{abs_path}'"
        )

    def register_table_from_df(self, table_name, df):
        """将 DataFrame 物化为 DuckDB 物理表 (支持 UPSERT)"""
        self.con.register("_tmp_df", df)
        self.con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df"
        )
        self.con.execute("DROP VIEW IF EXISTS _tmp_df")
        self.logger.debug(f"已物化物理表: {table_name}")

    def register_view_from_df(self, view_name, df):
        """将 DataFrame 注册为临时视图 (不产生物理拷贝)"""
        self.con.register(view_name, df)
        self.logger.debug(f"已注册内存视图: {view_name}")

    def save_to_warehouse(self, table_or_view, storage_type="snapshots"):
        """将表或视图持久化为 Parquet 文件。"""
        sub_dir = self.dirs["warehouse"] / storage_type
        sub_dir.mkdir(parents=True, exist_ok=True)

        path = (sub_dir / f"{table_or_view}.parquet").resolve()

        # DuckDB 在 Windows 上也推荐使用 Posix 风格路径 (/)
        self.con.execute(f"COPY {table_or_view} TO '{path.as_posix()}' (FORMAT PARQUET)")
        self.logger.info(f"💾 已将资产 {table_or_view} 固化至: {path}")
        return path

    def export_table(self, table_name, filename=None, format="fits"):
        """导出表为指定格式。"""
        output_name = filename if filename else table_name
        output_path = cfg.EXPORT_DIR / f"{output_name}.{format}"
        if format.lower() == "fits":
            from astropy.table import Table

            df = self.query(f"SELECT * FROM {table_name}")
            Table.from_pandas(df).write(str(output_path), overwrite=True)
        else:
            self.con.execute(
                f"COPY {table_name} TO '{output_path.as_posix()}' (FORMAT {format.upper()})"
            )

    def batch_export(self, table_names):
        for name in table_names:
            self.export_table(name)

    def close(self):
        if hasattr(self, "con") and self._connection_active:
            self.con.close()
            self._connection_active = False

    def query(self, sql_query):
        """执行 SQL 并返回 Pandas DataFrame。"""
        try:
            return self.con.sql(sql_query).df()
        except Exception as e:
            self.logger.error(f"🔍 查询执行失败: {e}\nSQL: {sql_query}")
            return pd.DataFrame()

    def execute(self, sql):
        """通用 SQL 执行入口。"""
        try:
            self.logger.debug(f"Executing SQL: {sql}")
            return self.con.execute(sql)
        except Exception as e:
            self.logger.error(f"❌ SQL Execution Failed: {e}")
            self.logger.error(f"Failed SQL: {sql}")
            raise e

    def list_resources(self):
        """打印当前库中所有表和视图。"""
        # 获取所有物理表和视图
        df_tables = self.con.execute("SHOW TABLES").df()
        if df_tables.empty:
            self.logger.info("当前数据库为空。")
            return

        self.logger.info(
            f"📁 当前数据库资源清单 (共 {len(df_tables)} 个):\n{df_tables}"
        )
        return df_tables

    def get_table_schema(self, table_name):
        """获取表结构。"""
        try:
            schema = self.con.execute(f"DESCRIBE {table_name}").df()
            self.logger.info(
                f"📊 表 {table_name} 的字段结构:\n{schema[['column_name', 'column_type']]}"
            )
        except Exception as e:
            self.logger.error(f"无法获取表 {table_name} 的结构: {e}")

    def _setup_spatial_macros(self):
        """注册球面距离计算宏 (Haversine Formula)。"""
        self.logger.info("📐 正在注册空间计算宏: haversine_distance (单位: Degree)")

        # 针对天文学应用，直接返回度数（Degree）是最合理的
        sql = """
        CREATE OR REPLACE MACRO haversine_distance(ra1, dec1, ra2, dec2) AS (
            DEGREES(2 * ASIN(SQRT(
                POW(SIN(RADIANS(dec2 - dec1) / 2), 2) +
                COS(RADIANS(dec1)) * COS(RADIANS(dec2)) *
                POW(SIN(RADIANS(ra2 - ra1) / 2), 2)
            )))
        );
        """
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
            self.logger.error(f"无法获取 [{name}] 的计数: {e}")
            return 0

    def table_exists(self, name):
        """检查表或视图是否存在。"""
        try:
            sql = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{name}'"
            res = self.con.execute(sql).fetchone()
            return res[0] > 0 if res else False
        except Exception:
            return False

    def drop_table(self, name):
        """删除物理表（如果存在）。"""
        try:
            self.con.execute(f"DROP TABLE IF EXISTS {name}")
            self.logger.debug(f"已删除物理表: {name}")
        except Exception as e:
            self.logger.error(f"删除表 {name} 失败: {e}")

    def drop_view(self, name):
        """删除视图（如果存在）。"""
        try:
            self.con.execute(f"DROP VIEW IF EXISTS {name}")
            self.logger.debug(f"已删除视图: {name}")
        except Exception as e:
            self.logger.error(f"删除视图 {name} 失败: {e}")

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

    def import_raw(self, target_cluster_id=None, force=False):
        if not self.data_manifest:
            self.logger.warning("⚠️ 未检测到 Data Manifest。")
            return

        self.logger.info("🚀 开始引导 AstroDB 数据环境...")

        # 核心优化：提取当前任务相关的索引，过滤无关星团的加载
        target_indices = set()
        other_clusters_indices = set()
        if target_cluster_id and target_cluster_id in cfg.CLUSTERS:
            target_indices.add(cfg.CLUSTERS[target_cluster_id].get("FIELD_IDX"))
            target_indices.add(cfg.CLUSTERS[target_cluster_id].get("SEED_IDX"))
            
            for cid, cinfo in cfg.CLUSTERS.items():
                if cid != target_cluster_id:
                    other_clusters_indices.add(cinfo.get("FIELD_IDX"))
                    other_clusters_indices.add(cinfo.get("SEED_IDX"))

        for k, config in self.data_manifest.items():
            # 如果该项属于其他星团的数据源（且不是当前目标的必要项），则跳过
            if k in other_clusters_indices and k not in target_indices:
                continue

            mode = config.get("sync_mode", "HYBRID")
            
            # [核心重构]：如果模式是 VIRTUAL，说明该项是一个逻辑视图（如种子集），
            # 它直接引用 base_idx 对应的 raw 表，不需要物理文件同步。
            if mode == "VIRTUAL":
                self.logger.info(f"🌌 跳过虚拟同步项: {k} (将作为逻辑视图处理)")
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
                should_sync = True
            elif mode == "OFFLINE" and not file_exists:
                self.logger.error(f"❌ 离线任务缺失物理文件: {k} (路径: {result_path})")
                continue

            if should_sync:
                self.logger.info(f"🔄 正在同步数据: {k} -> {result_path}")
                try:
                    self._execute_sync_task(config, result_path)
                    file_exists = True  # 同步成功后更新状态
                except Exception as e:
                    self.logger.error(f"❌ 同步 {k} 失败: {e}")
                    continue
            else:
                self.logger.debug(f"⏭️  数据文件 {k}.parquet 已存在，跳过下载。")

            if force or not table_exists:
                self.logger.info(f"📋 正在注册数据库表: {t_raw}")
                self.register_table_from_file(t_raw, result_path)
            else:
                self.logger.info(f"✅ 表 {t_raw} 已在内存中就绪，无需重新注册。")

        self.logger.info("✨ AstroDB L1 原始数据环境导入完成。")

    def register_table_from_file(self, table_name, file_path):
        """将 Parquet 文件物化为 DuckDB 物理表。"""
        abs_path = Path(file_path).resolve().as_posix()
        try:
            self.con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{abs_path}')"
            )
            count = self.get_row_count(table_name)
            self.logger.info(f"📦 已物化物理表: {table_name} (行数: {count})")
        except Exception as e:
            self.logger.error(f"❌ 物化物理表 {table_name} 失败: {e}")

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
            self.logger.warning("⚠️ [SimbadProvider] 依赖源未就绪或未定义，将初始化空缓存表结构。")
            return pd.DataFrame(columns=[id_col, "main_id", "ids"])

        formatted_ids = [f"{prefix}{idx}" for idx in target_ids]
        if not formatted_ids:
            return pd.DataFrame(columns=[id_col, "main_id", "ids"])

        self.logger.info(f"正在从 SIMBAD 查询 {len(formatted_ids)} 个天体的别名...")

        Simbad.reset_votable_fields()
        Simbad.add_votable_fields("ids")

        result_table = Simbad.query_objects(formatted_ids)

        if result_table is None:
            self.logger.warning("SIMBAD 未返回任何匹配数据。")
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
            self.logger.debug("🌐 SIMBAD SSL 校验已绕过。")

    def query_simbad_target_info(self, gaia_dr3_id: str) -> dict:
        """
        查询指定天体的完整别名与元数据。
        """
        self._bypass_ssl_verification()
        self.logger.info(f"🌐 正在从远端 CDS SIMBAD 检索单个天体: {gaia_dr3_id} ...")

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
                f"❌ 从 SIMBAD 检索单星数据时发生网络或解析异常: {str(e)}"
            )
            return None

    # =========================================================================
    # 🌟 高性能批量跨网络与本地数据库同步服务
    # =========================================================================

    def sync_simbad_cache(self, source_ids, cache_table_name, prefix="Gaia DR3 ", chunk_size: int = 500) -> pd.DataFrame:
        """
        🚀 [核心接口] 跨网络与本地数据库同步 SIMBAD 缓存。
        
        采用 DuckDB 数据库侧集合运算，支持增量同步与网络熔断机制。
        """
        df_input = self._normalize_input_ids(source_ids)
        self.con.register("temp_sync_input", df_input)
        
        # 1. 创建缓存表（如果不存在）
        self.execute(f"""
            CREATE TABLE IF NOT EXISTS {cache_table_name} (
                gaia_dr3_id VARCHAR PRIMARY KEY,
                main_id VARCHAR,
                ids VARCHAR
            )
        """)

        # 2. 检查本地命中
        df_cached_hits = self.con.execute(f"""
            SELECT i.gaia_dr3_id, c.main_id, c.ids
            FROM temp_sync_input i
            JOIN {cache_table_name} c ON i.gaia_dr3_id = c.gaia_dr3_id
        """).df()
        df_cached_hits["cache_hit"] = True

        # 3. 找出本地缺失项
        df_missing = self.con.execute(f"""
            SELECT DISTINCT i.gaia_dr3_id
            FROM temp_sync_input i
            ANTI JOIN {cache_table_name} c ON i.gaia_dr3_id = c.gaia_dr3_id
        """).df()
        
        ids_missing = df_missing["gaia_dr3_id"].tolist()
        self.con.unregister("temp_sync_input")

        df_online_results = pd.DataFrame(columns=["gaia_dr3_id", "main_id", "ids", "cache_hit"])

        # 4. 执行网络同步
        if ids_missing:
            df_online_results = self._perform_online_sync(ids_missing, cache_table_name, prefix, chunk_size)

        # 5. 合并结果
        df_final_merged = pd.concat([df_cached_hits, df_online_results], ignore_index=True)
        
        self.logger.info(
            f"🎯 [SimbadCache] 同步完成: 总计 {len(df_input)} 颗 | "
            f"命中缓存: {len(df_cached_hits)} | 增量下载: {len(df_online_results)}"
        )

        # ✨ 关键逻辑：自动将增量更新后的结果持久化到 Parquet
        # 这样下次 import_raw 运行时，加载的就是合并后的最新数据
        if not df_online_results.empty:
            self.save_to_warehouse(cache_table_name, storage_type="snapshots")

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
        self.logger.info(f"🌐 [SimbadSync] 正在从 CDS 增量抓取 {len(ids_missing)} 个源...")
        self._bypass_ssl_verification()
        
        online_records = []
        for i in range(0, len(ids_missing), chunk_size):
            if self._check_user_interrupt():
                self.logger.warning("🛑 [SimbadSync] 接收到人工中断信号，正在保存已完成批次...")
                break

            chunk_ids = ids_missing[i : i + chunk_size]
            batch_results = self._fetch_simbad_batch(chunk_ids, prefix)
            
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                self._save_local_cache_incremental(df_batch, table_name)
                online_records.extend(batch_results)
                self.logger.info(f"   ∟ 进度: {len(online_records)}/{len(ids_missing)}")

            if i + chunk_size < len(ids_missing):
                time.sleep(random.uniform(0.5, 1.5))

        df_online = pd.DataFrame(online_records)
        if not df_online.empty:
            df_online["cache_hit"] = False
        return df_online

    def _fetch_simbad_batch(self, chunk_ids: list, prefix: str) -> list:
        """执行单批次网络请求。"""
        query_names = [f"{prefix}{mid}" for mid in chunk_ids]
        # 匹配 Gaia ID 的正则，用于从 SIMBAD 的 user_specified_id 回溯
        id_pattern = re.compile(r"(\d+)$")
        
        batch_results = []
        found_ids = set()

        try:
            Simbad.reset_votable_fields()
            Simbad.add_votable_fields("ids")
            table = Simbad.query_objects(query_names)
            
            if table is not None:
                for row in table:
                    # 提取原始 ID
                    match = id_pattern.search(str(row["user_specified_id"]))
                    if not match: continue
                    
                    gid = match.group(1)
                    found_ids.add(gid)
                    
                    main_id = str(row["main_id"])
                    ids = row["ids"].decode("utf-8") if isinstance(row["ids"], bytes) else str(row["ids"])
                    
                    is_empty = (main_id.strip().upper() in ["NONE", ""])
                    batch_results.append({
                        "gaia_dr3_id": gid,
                        "main_id": "None" if is_empty else main_id,
                        "ids": "None" if is_empty else ids
                    })
            
            # 补全未找到的项
            for mid in chunk_ids:
                if str(mid) not in found_ids:
                    batch_results.append({"gaia_dr3_id": str(mid), "main_id": "None", "ids": "None"})
                    
        except Exception as e:
            self.logger.error(f"❌ [SimbadSync] 网络请求失败: {str(e)}")
            
        return batch_results

    def _save_local_cache_incremental(self, df_new: pd.DataFrame, table_name: str):
        """增量更新数据库表。"""
        self.con.register("tmp_inc", df_new)
        # 修正：DuckDB 在没有明确 PRIMARY KEY/UNIQUE 约束时不支持 INSERT OR REPLACE。
        # 由于缓存表可能在 import_raw 阶段通过 Parquet 重新物化（导致约束丢失），此处采用 DELETE + INSERT 模式实现 UPSERT。
        self.execute(f"DELETE FROM {table_name} WHERE gaia_dr3_id IN (SELECT gaia_dr3_id FROM tmp_inc)")
        self.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_inc")
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
        except: pass
        return False
