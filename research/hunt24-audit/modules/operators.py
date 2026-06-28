# operators.py
import pandas as pd
import json

import config as cfg


class AssetManager:
    """Stage 1.2: 负责外部资产挂载与入库"""
    def __init__(self, engine):
        self.engine = engine

    def mount_all(self, cluster_id):
        """挂载该星团所需的所有原始资产 (raw_*)"""
        # 从 MANIFEST 获取该星团的资产清单
        assets = cfg.MANIFEST.get_assets_for_cluster(cluster_id)
        
        for asset_key, asset_cfg in assets.items():
            # 1. 确定物理文件路径
            file_path = self._resolve_path(asset_cfg)
            
            # 2. 生成规范的物理表名 (例如 raw_obs_m45)
            table_name = f"raw_{asset_cfg.meta_type}_{cluster_id}"
            
            # 3. 执行入库操作
            self._import_to_db(file_path, table_name)

    def _resolve_path(self, asset_cfg):
        # 封装路径拼接逻辑
        return cfg.DATA_DIR / "raw" / f"{asset_cfg.id}.parquet"

    def _import_to_db(self, file_path, table_name):
        # 核心：将文件吸纳为物理表
        # 这里通过 engine 执行 SQL，不再直接操作 duckdb.conn
        sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')"
        self.engine.execute(sql)

class SchemaAligner:
    """Stage 1.5: 负责 ODS -> DW 的洗练与视图注册"""
    def __init__(self, engine):
        self.engine = engine

    def align_all(self, cluster_id):
        """扫描所有已落表的 raw_* 物理表，自动创建标准视图"""
        # 1. 获取该星团的所有原始表清单
        raw_tables = self._get_raw_tables(cluster_id)
        
        for table in raw_tables:
            # 2. 根据表前缀决定视图逻辑 (raw_obs_ -> stx_view_ 等)
            view_sql = self._build_view_sql(table)
            if view_sql:
                self.engine.execute(view_sql)
                print(f"✨ 视图注册完成: {table.replace('raw_', '')}")

    def _get_raw_tables(self, cluster_id):
        """查询 DuckDB 系统表，获取当前集群相关的原始表"""
        sql = f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'raw_%_{cluster_id}'"
        return [row[0] for row in self.engine.execute(sql).fetchall()]

    def _build_view_sql(self, table_name):
        """根据表前缀生成 SQL 映射逻辑"""
        # 核心逻辑：读取 config.py 中的字段映射映射表 (例如 STD_COLS)
        if "raw_obs_" in table_name:
            view_name = table_name.replace("raw_obs_", "stx_view_")
            # 这里调用 config.py 中定义的映射模板
            mapping = ", ".join([f"{k} AS {v}" for k, v in cfg.STD_COLS.items()])
            return f"CREATE OR REPLACE VIEW {view_name} AS SELECT {mapping} FROM {table_name}"
            
        elif "raw_lit_" in table_name:
            view_name = table_name.replace("raw_lit_", "cat_view_")
            return f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {table_name}"
            
        return None


class IdentityRegistry:
    """Stage 1.1: 负责星团物理特征真身 (Identity) 的注册、反演与持久化"""
    def __init__(self, engine):
        self.engine = engine

    def resolve(self, cluster_id, mode):
        """
        核心调度方法：
        如果 mode='static'，直接读取真身表；
        如果 mode='dynamic'，触发 MLE 数学反演并滚动更新。
        """
        if mode == "dynamic":
            return self._dynamic_reconstruct(cluster_id)
        
        # 默认 static 模式：从物理表中装载 DNA
        sql = f"SELECT * FROM cluster_kinematic_identity WHERE cluster_id = '{cluster_id}'"
        return self.engine.query(sql).iloc[0]

    def _dynamic_reconstruct(self, cluster_id):
        """
        物理 DNA 重构算子：
        1. 提取文献明细视图 (cat_view_*)
        2. 执行 MLE/高斯拟合数学反演
        3. 写回真身表并返回最新 DNA
        """
        # 1. 提取参考文献样本 (此处逻辑对应白皮书中的交叉审计底牌)
        lit_df = self.engine.query(f"SELECT * FROM cat_view_hunt WHERE cluster_id = '{cluster_id}'")
        
        # 2. 执行数学反演 (例如：计算协方差矩阵、质心漂移)
        # 此处简化逻辑，实际项目中会调用 scipy.optimize 或 sklearn.mixture
        new_dna = self._perform_mle_inversion(lit_df)
        
        # 3. 滚动更新真身表 (Update Registry)
        self._update_identity_table(cluster_id, new_dna)
        
        return new_dna

    def _perform_mle_inversion(self, df):
        """在这里放置重型数学反演逻辑，不再干扰外部业务"""
        # (此处为伪代码，实际执行 MLE 反演计算)
        return {"ra_center": ..., "pmra_center": ..., "cov": [...]}
    
    def update(self, cluster_id, data):
        """Workflow 调用的公有更新接口"""
        self.logger.info(f"正在更新真身 DNA: {cluster_id}")
        self._update_identity_table(cluster_id, data)

    def _update_identity_table(self, cluster_id, dna):
        """执行 SQL 更新操作"""
        # update_sql = f"UPDATE cluster_kinematic_identity SET ... WHERE cluster_id = '{cluster_id}'"
        # self.engine.execute(update_sql)
        pass