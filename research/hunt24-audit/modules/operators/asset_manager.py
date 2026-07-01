# assest_manager.py
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging
from astropy.coordinates import SkyCoord
import astropy.units as u
import config as cfg

class AssetManager:
    """Stage 1.2: 负责外部资产挂载与入库"""
    def __init__(self, engine, cluster_id=None):
        self._engine = engine
        # 建立层级 Logger，例如 AstroDB.m45.AssetManager
        ctx = cluster_id or 'global'
        self.logger = logging.getLogger(f"AstroDB.{ctx}.AssetManager")

    def mount_all(self, cluster_id):
        """挂载该星团所需的所有原始资产 (raw_*)"""        
        
        # 从 MANIFEST 获取该星团的资产清单
        cl_assets, g_assets = cfg.MANIFEST.get_assets_from_config(cluster_id)
        # self.logger.info(f"星团相关资产: {cl_assets};星团无关资产 : {g_assets}")
        self._mount_assests(cl_assets)
        self._mount_assests(g_assets)

    def _mount_assests(self, assets):
        for asset_key, asset_cfg in assets.items():
            # 1. 确定物理文件路径
            file_path = self._resolve_path(asset_cfg)
            
            # 2. 生成规范的物理表名 (例如 raw_lic_hunt)
            table_name = f"raw_{asset_key}"
            
            # 3. 执行入库操作
            self._import_to_db(file_path, table_name)

    def _resolve_path(self, asset_cfg):
        # 封装路径拼接逻辑
        return cfg.DATA_DIR / "raw" / asset_cfg.file_pattern

    def _import_to_db(self, file_path, table_name, force_import = False):
        """
        将不同格式的星表文件吸纳为 DuckDB 物理表
        支持格式: .parquet, .csv, .vot, .fit, .fits
        """

        if self._engine.table_exists(table_name) and not force_import:
            self.logger.info(f"表 {table_name} 已经存在, 跳过引入过程")
            return

        file_path_str = str(file_path)
        # 获取小写的统一后缀名
        ext = Path(file_path).suffix.lower()
        
        # 1. 针对 DuckDB 原生支持的格式，直接使用 SQL 提升性能（免去内存转换）
        native_readers = {
            '.parquet': f"read_parquet('{file_path_str}')",
            '.csv': f"read_csv_auto('{file_path_str}')",
        }
        
        if ext in native_readers:
            reader_expr = native_readers[ext]
            sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {reader_expr}"
            self._engine.execute(sql)
            self.logger.debug(f"⚡ [Native] 已通过 SQL 导入表 {table_name} ({ext})")
            
        # 2. 针对天文学专用格式（VOTable, FITS），通过底层一键落表完成持久化
        elif ext in ['.vot', '.fit', '.fits']:
            astro_df = self._load_astro_file_to_df(file_path_str, ext)
            
            # 【终极优雅】因为算子需要持久化，直接调用一键落表
            self._engine.register_table_from_df(table_name, astro_df)
            self.logger.debug(f"🪐 [Astropy] 已通过驱动原子接口物化落表 {table_name} ({ext})")
                
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_astro_file_to_df(self, file_path, ext):
        """辅助方法：将 VOTable 或 FITS 转换为 Pandas DataFrame"""
        try:
            from astropy.table import Table
        except ImportError:
            raise ImportError(
                "Refactoring requires 'astropy' to support VOTable and FITS formats. "
                "Please run: pip install astropy"
            )

        # 显式指定天文学格式，防止部分非常规后缀无法识别
        if ext == '.vot':
            astro_table = Table.read(file_path, format='votable')
        elif ext in ['.fit', '.fits']:
            astro_table = Table.read(file_path, format='fits')
        else:
            raise ValueError(f"Invalid astronomical format: {ext}")
            
        # 转换为 Pandas DataFrame
        # 提示：Astropy 会将掩码（Masked）星等或不确定度自动转换为具有 NaN 的 Float 序列，DuckDB 会完美对齐为 NULL
        return astro_table.to_pandas()    
    
    def ______activate_derived_assets(self, cluster_id, runtime_context: any) -> any:
        """
        Stage 1.5: 派生虚拟资产级联装配控制流。
        【从 db.py 迁移合并至此】根据运行时上下文，动态编译并挂载面向具体业务（如种子切片、大盘演化）的 SQL 虚拟化视图。
        """
        if not self._engine:
            self.logger.error("❌ 引擎未就绪，无法执行派生资产虚拟化挂载。")
            return cfg.DerivedActivationResult()

        if not cluster_id:
            self.logger.warning("⚠️ 全局未指定特定天体标识符，跳过级联派生视图挂载。")
            return cfg.DerivedActivationResult()

        self.logger.info(f"🎬 [AssetManager] 激活底座虚拟化引擎，开始级联装配星团 '{cluster_id}' 的派生科学资产视图...")

        cluster_id_lower = cluster_id.lower()
        activated_views = []
        row_counts = {}

        # -----------------------------------------------------------------------------
        # 核心编译矩阵：定义需要动态挂载的视图大盘拓扑结构
        # -----------------------------------------------------------------------------
        view_templates = {
            # 1. 基础物理空间对齐剪裁视图
            f"std_view_{cluster_id_lower}_field": f"""
                SELECT 
                    id, ra, dec, plx, pmra, pmdec, color,
                    plx_error, pmra_error, pmdec_error
                FROM raw_lic_hunt
                WHERE cluster_id = '{self.cluster_id.upper()}'
                  AND plx IS NOT NULL
            """,
            
            # 2. 运行时动态业务切片视图（如：注入种子星提取的高纯度约束条件）
            f"std_view_{cluster_id_lower}_seeds_field": f"""
                SELECT * FROM std_view_{cluster_id_lower}_field
                WHERE phot_g_mean_mag <= {runtime_context.dynamic_parameters.get('BRIGHT_LIMIT', 15.0)}
                  AND parallax >= {runtime_context.dynamic_parameters.get('PLX_MIN', 0.1)}
            """
        }

        # -----------------------------------------------------------------------------
        # 执行资产物化/虚拟化流，并滚动审计各节点的数据流吞吐量
        # -----------------------------------------------------------------------------
        try:
            for view_name, view_sql in view_templates.items():
                # 级联编译并创建/覆盖虚拟视图
                compile_sql = f"CREATE OR REPLACE VIEW {view_name} AS {view_sql}"
                self._engine.execute(compile_sql)
                
                # 动态审计当前数据节点的流水量（吞吐行数统计）
                count_res = self._engine.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()
                current_count = count_res[0] if count_res else 0
                
                # 记录装配状态大盘
                activated_views.append(view_name)
                row_counts[view_name] = current_count
                
                self.logger.info(
                    f"  ↳ 💾 [Asset Virtualized] 派生资产节点 {view_name:<35} 装配成功 | 级联节点吞吐量: {current_count} 行"
                )

            # 同步刷新当前算子内部的状态寄存器
            self.activated_views = activated_views
            self.row_counts = row_counts

            self.logger.info(f"✨ [AssetManager] 派生虚拟化大盘级联就绪。成功挂载节点数: {len(activated_views)}")
            
            # 返回标准的资产激活报告供给下游（如 SeedSelector）进行熔断审计
            return cfg.DerivedActivationResult(
                activated_views=activated_views,
                row_counts=row_counts,
                dynamic_parameters=runtime_context.dynamic_parameters
            )

        except Exception as e:
            self.logger.error(f"💥 [AssetManager 灾难] 派生虚拟资产级联装配流遭遇物理击穿: {e}", exc_info=True)
            # 发生灾难时返回空报告，触发下游算法级联安全熔断
            return cfg.DerivedActivationResult()
        
    def align_derived_assets(self, cluster_id, runtime_context: dict, asset_type_filter=None):
        """
        [阶段二] 通用动态派生视图注册算子
        
        Args:
            cluster_id: 星团ID
            runtime_context: 运行时上下文字典（包含 MLE 反演出的 DNA 参数、全局控制变量等）
            asset_type_filter: 可选，指定只注册某种特定类型的派生视图（例如未来按需加载）
        """
        self.logger.info(f"🔄 启动通用动态派生视图管道... 目标星团: {cluster_id}")
        # cl_assets, _ = cfg.MANIFEST.get_assets_from_config(cluster_id)
        cl_assets = cfg.MANIFEST
        
        # 组装完整的渲染上下文（将全局静态配置作为兜底，运行时的动态上下文覆盖进去）
        global_render_cfg = {
            # "SEED_MAX_MAG": getattr(cfg, "SEED_MAX_MAG", 18.0),
            "SEED_RADIUS": cfg.CLUSTERS[cluster_id.upper()].SEED_RADIUS,
            "SEED_PLX_LIM": cfg.CLUSTERS[cluster_id.upper()].SEED_PLX_LIM,
            "SEED_MAX_MAG": cfg.CLUSTERS[cluster_id.upper()].SEED_MAX_MAG,
            # 未来可以在这里扩展更多全局派生控制参数
        }
        # 运行时的 DNA 等高优先级参数覆盖全局配置
        render_context = {**global_render_cfg, **runtime_context}

        for asset_key, asset_cfg in cl_assets.items():
            # 1. 检查通用条件：是否有预过滤条件
            if not (hasattr(asset_cfg, 'pre_filters') and asset_cfg.pre_filters):
                continue
                
            # 2. 可选：按资产类型或标签过滤（为未来更精细的控制做准备）
            if asset_type_filter and getattr(asset_cfg, 'asset_type', None) != asset_type_filter:
                continue

            # self.logger.info(f"[debug]asset_key:{asset_key}")
            if not asset_key.startswith(f"{cluster_id}_") :
                continue
                
            # 3. 核心通用渲染与物理/视图化物化
            derived_sql = self._build_derived_view_sql(asset_cfg, render_context)
            if derived_sql:
                self._engine.execute(derived_sql)
                self.logger.info(f"🚀 动态派生视图 [std_view_{asset_cfg.id}] 注册成功")

    def _build_derived_view_sql(self, asset_cfg, render_context: dict):
        """通用的 SQL 模板渲染引擎"""
        base_view_name = f"std_view_{asset_cfg.base_idx}"
        target_view_name = f"std_view_{asset_cfg.id}"
        
        rendered_filters = []
        for filter_tpl in asset_cfg.pre_filters:
            try:
                # 依靠 Python 的标准强类型字符串格式化，支持任意自定义标签的动态替换
                rendered_filter = filter_tpl.format(**render_context)
                rendered_filters.append(rendered_filter)
            except KeyError as e:
                self.logger.error(f"❌ 派生视图 [{target_view_name}] 过滤条件解析失败，缺少运行上下文参数: {e} | 模板: {filter_tpl}")
                return None

        where_clause = " AND ".join([f"({f})" for f in rendered_filters])
        sql = f"""
            CREATE OR REPLACE VIEW {target_view_name} AS 
            SELECT * FROM {base_view_name} 
            WHERE {where_clause}
        """
        return sql

        

class SchemaAligner:
    """Stage 1.5: 负责 ODS -> DW 的洗练与视图注册"""
    def __init__(self, engine, cluster_id):
        self.engine = engine
        ctx = cluster_id or 'global'
        self.logger = logging.getLogger(f"AstroDB.{ctx}.SchemaAligner")

    def align_all(self, cluster_id):
        """扫描所有已落表的 raw_* 物理表，自动创建标准视图"""
        # 1. 获取该星团的所有原始表清单, 包含星团相关和全局数据资产
        cl_assets, g_assets = cfg.MANIFEST.get_assets_from_config(cluster_id)

        align_assets = cl_assets | g_assets
        
        for table in align_assets.keys():
            # 2. 根据表前缀决定视图逻辑 (raw_ -> std_view_ 等)
            view_sql = self._build_view_sql(table)
            self.logger.debug(f"[debug] view_sql : {view_sql}")
            if view_sql:
                self.engine.execute(view_sql)
                self.logger.info(f"✨ 视图注册完成: {table}")

    

    def _build_derived_view_sql(self, asset_cfg, render_context: dict):
        """通用的 SQL 模板渲染引擎"""
        base_view_name = f"std_view_{asset_cfg.base_idx}"
        target_view_name = f"std_view_{asset_cfg.id}"
        
        rendered_filters = []
        for filter_tpl in asset_cfg.pre_filters:
            try:
                # 依靠 Python 的标准强类型字符串格式化，支持任意自定义标签的动态替换
                rendered_filter = filter_tpl.format(**render_context)
                rendered_filters.append(rendered_filter)
            except KeyError as e:
                self.logger.error(f"❌ 派生视图 [{target_view_name}] 过滤条件解析失败，缺少运行上下文参数: {e} | 模板: {filter_tpl}")
                return None

        where_clause = " AND ".join([f"({f})" for f in rendered_filters])
        sql = f"""
            CREATE OR REPLACE VIEW {target_view_name} AS 
            SELECT * FROM {base_view_name} 
            WHERE {where_clause}
        """
        return sql

    def _get_raw_tables(self, cluster_id):
        """查询 DuckDB 系统表，获取当前集群相关的原始表"""
        sql = f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'raw_%_{cluster_id}_field'"
        return [row[0] for row in self.engine.execute(sql).fetchall()]

    # def _build_view_sql(self, table_name):
    #     """根据表前缀生成 SQL 映射逻辑"""
    #     # 核心逻辑：读取 config.py 中的字段映射映射表 (例如 STD_COLS)
    #     if "raw_obs_" in table_name:
    #         view_name = table_name.replace("raw_obs_", "stx_view_")
    #         # 这里调用 config.py 中定义的映射模板
    #         mapping = ", ".join([f"{k} AS {v}" for k, v in cfg.STD_COLS.items()])
    #         return f"CREATE OR REPLACE VIEW {view_name} AS SELECT {mapping} FROM {table_name}"
            
    #     elif "raw_lit_" in table_name:
    #         view_name = table_name.replace("raw_lit_", "cat_view_")
    #         return f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {table_name}"
            
    #     return None
    
    def _build_view_sql(self, table_name):
        """根据表id生成 SQL 映射逻辑"""
        # 核心逻辑：读取 config.py 中的字段映射映射表 (例如 ASSET_SCHEMAS)
        view_name = f"std_view_{table_name}"
        raw_name = f"raw_{table_name}"

        # f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {table_name}"
        # 2. 从 config.py 的 LITERATURE_SCHEMAS 中获取对应的 Schema 实例
        st = cfg.MANIFEST[table_name].schemas_type
        schema = cfg.ASSET_SCHEMAS.get(st)

        
        asset_type = cfg.MANIFEST[table_name].asset_type
        # if table_name.lower() == "hunt".lower()


        # self.logger.info(f"++++{asset_type}++++")

        if asset_type == cfg.AssetType.LIT_CATALOG and cfg.MANIFEST[table_name].raw_filters: 
            rendered_filters = cfg.MANIFEST[table_name].raw_filters[0].format(CAT_NAME="Melotte_22")
            
            where_clause = " AND ".join([rendered_filters])
            self.logger.info(f"[where_clause]++++{where_clause}++++")
        else:
            where_clause = "TRUE"
        
        # self.logger.info(f"{table_name}|rendered_filters : {rendered_filters}")

        # where_clause = " AND ".join([f"({f})" for f in rendered_filters])
        
        if schema:
            # 3. 动态构建映射逻辑
            # schema.items() 返回 (标准字段名, 原始表格列名)，如 ('id', 'GaiaDR3')
            # 构造 SQL 时需要反过来：'原始列名' AS '标准字段名'
            # 使用双引号包裹字段名，防止类似 'BP-RP' 的连字符导致 SQL 语法报错
            mapping_list = [f'"{origin_col}" AS "{std_field}"' for std_field, origin_col in schema.items()]
            mapping = ", ".join(mapping_list)
            return f"CREATE OR REPLACE VIEW {view_name} AS SELECT {mapping} FROM {raw_name} WHERE {where_clause}"
        else:
            # 如果配置中没定义该文献表的 Schema，可以选择降级为 SELECT * 或抛出异常
            return f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {raw_name} WHERE {where_clause}"