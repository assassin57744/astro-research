# modules/config_manager.py
import logging
import json
import numpy as np
import duckdb
import config as cfg  # 引入项目根目录下的静态 config.py


class ClusterConfigManager:
    """
    星团动态配置与参数资产管理器。
    支持：内存缓存 -> DuckDB自适应表检索 -> config.py静态兜底 的三级级联查找。
    """

    def __init__(self, db_instance=None):
        self.logger = logging.getLogger("AstroPipeline.ConfigManager")
        self.db = db_instance
        self._runtime_cache = {}

    def get_param(self, cluster_id: str, param_name: str):
        """核心物理参数级联检索接口"""
        cluster_id = cluster_id.upper()
        param_name = param_name.upper()

        # 层级 1：检查内存运行时缓存
        if (
            cluster_id in self._runtime_cache
            and param_name in self._runtime_cache[cluster_id]
        ):
            return self._runtime_cache[cluster_id][param_name]

        # 层级 2：检索先前由 Refiner 提炼并安全落盘的 DuckDB 科学资产
        if self.db:
            try:
                self._ensure_config_table_exists()
                query_sql = f"""
                    SELECT param_value, is_array 
                    FROM cluster_refined_params 
                    WHERE cluster_id = '{cluster_id}' AND param_name = '{param_name}'
                """
                res = self.db.execute(query_sql).fetchone()
                if res:
                    val_str, is_array = res[0], res[1]
                    val = np.array(json.loads(val_str)) if is_array else float(val_str)
                    self._set_memory_cache(cluster_id, param_name, val)
                    return val
            except Exception as e:
                self.logger.debug(
                    f"🛈 资产表未命中或检索失败，转向静态配置降级。原因: {e}"
                )
                pass

        # 层级 3：平滑降级到项目根目录下的静态 config.py
        if hasattr(cfg, "CLUSTERS") and cluster_id in cfg.CLUSTERS:
            static_cluster_cfg = cfg.CLUSTERS[cluster_id]
            if param_name in static_cluster_cfg:
                val = static_cluster_cfg[param_name]
                self._set_memory_cache(cluster_id, param_name, val)
                return val

        self.logger.error(
            f"❌ [ConfigError] 无法在任意域中匹配到星团 {cluster_id} 的物理参数: '{param_name}'"
        )
        return None

    def save_refined_params(self, cluster_id: str, refined_config: dict):
        """将自适应进化参数批量 Upsert 写入 DuckDB，并强刷内存缓存"""
        cluster_id = cluster_id.upper()
        if not self.db:
            self.logger.warning(
                f"⚠️ 未检测到有效数仓连接，{cluster_id} 自适应参数将只在内存级生效（不落盘）。"
            )
            for k, v in refined_config.items():
                self._set_memory_cache(cluster_id, k, v)
            return

        try:
            self._ensure_config_table_exists()
            for k, v in refined_config.items():
                k = k.upper()
                is_array = isinstance(v, np.ndarray)
                val_str = json.dumps(v.tolist()) if is_array else str(v)

                upsert_sql = f"""
                    INSERT OR REPLACE INTO cluster_refined_params (cluster_id, param_name, param_value, is_array)
                    VALUES ('{cluster_id}', '{k}', '{val_str}', {is_array})
                """
                self.db.execute(upsert_sql)
                self._set_memory_cache(cluster_id, k, v)
            self.logger.info(
                f"💾 [ConfigManager] 星团 {cluster_id} 动态提炼的物理资产已安全持久化至 DuckDB 配置底座。"
            )
        except Exception as e:
            self.logger.error(
                f"❌ [ConfigError] 持久化自适应参数时发生 DuckDB 写入阻塞: {e}"
            )

    def _ensure_config_table_exists(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS cluster_refined_params (
                cluster_id VARCHAR,
                param_name VARCHAR,
                param_value VARCHAR,
                is_array BOOLEAN,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (cluster_id, param_name)
            )
        """)

    def _set_memory_cache(self, cluster_id, param_name, value):
        if cluster_id not in self._runtime_cache:
            self._runtime_cache[cluster_id] = {}
        self._runtime_cache[cluster_id][param_name] = value

    def reconstruct_cluster_params_from_db(self, cluster_id: str) -> dict:
        """
        🚀 【物理资产动态重建引擎】
        
        从现有 db 数据库的 hunt24 原始星表中，反演、重建该星团的核心运动学与几何特征参数。
        """
        cluster_key = cluster_id.upper()
        self.logger.info(f"🧬 [资产管理器] 启动物理先验重建管线，目标星团: {cluster_key}")

        # 1. 嗅探数仓中是否存在 hunt24 基础星表数据
        # 假设你的历史星表表名为 'hunt24_base' 或在 manifest 中有定义，这里以 'hunt24' 为例
        hunt_table = "hunt24" 
        
        # 检查表是否存在
        try:
            table_check = self.db.execute(
                f"SELECT * FROM information_schema.tables WHERE table_name = '{hunt_table}'"
            ).df()
            if table_check.empty:
                self.logger.warning(f"⚠️ 未在数仓中找到历史星表 '{hunt_table}'，无法执行反演重建。")
                return {}
        except Exception as e:
            self.logger.error(f"❌ 探测数仓基础表失败: {str(e)}")
            return {}

        # 2. 构建向量化鲁棒统计聚合 SQL
        # 使用 MEDIAN 替代 AVG 防止被极端外围星/流星拉偏基准；使用标准差计算弥散
        sql = f"""
            SELECT 
                COUNT(*) as member_count,
                MEDIAN(ra) as center_ra,
                MEDIAN(dec) as center_dec,
                MEDIAN(plx) as plx_ref,
                MEDIAN(pmra) as pmra_ref,
                MEDIAN(pmdec) as pmdec_ref,
                STDDEV(pmra) as pmra_disp,
                STDDEV(pmdec) as pmdec_disp,
                COVAR_POP(pmra, pmdec) as pm_covar
            FROM {hunt_table}
            WHERE UPPER(cluster_name) = '{cluster_key}'
               OR UPPER(cluster_id) = '{cluster_key}'
        """
        
        try:
            stats_df = self.db.execute(sql).df()
            if stats_df.empty or stats_df["member_count"].iloc[0] == 0:
                self.logger.warning(f"⚠️ 数仓历史星表 '{hunt_table}' 中未检索到星团 {cluster_key} 的任何历史成员，重建跳过。")
                return {}
            
            row = stats_df.iloc[0]
            member_count = int(row["member_count"])
            self.logger.info(f"📊 成功提取历史资产，包含高确信度历史成员星共: {member_count} 颗")

            # 3. 计算物理衍生 DNA 资产
            plx_median = float(row["plx_ref"])
            # 视差转换为距离 (pc)，防止除以 0 导致溢出
            distance_pc = 1000.0 / plx_median if plx_median > 0 else 100.0
            
            # 计算自行相关系数 PM_CORR = cov(x,y) / (std(x) * std(y))
            pmra_disp = float(row["pmra_disp"]) if pd.notna(row["pmra_disp"]) else 0.5
            pmdec_disp = float(row["pmdec_disp"]) if pd.notna(row["pmdec_disp"]) else 0.5
            pm_covar = float(row["pm_covar"]) if pd.notna(row["pm_covar"]) else 0.0
            
            pm_corr = 0.0
            if pmra_disp > 0 and pmdec_disp > 0:
                pm_corr = float(np.clip(pm_covar / (pmra_disp * pmdec_disp), -0.99, 0.99))

            # 4. 组装重构后的标准先验字典
            reconstructed_params = {
                "ID_NAME": cluster_key,
                "CENTER_RA": float(row["center_ra"]),
                "CENTER_DEC": float(row["center_dec"]),
                "PLX_REF": plx_median,
                "PMRA_REF": float(row["pmra_ref"]),
                "PMDEC_REF": float(row["pmdec_ref"]),
                "PMRA_DISPERSION": max(pmra_disp, 0.1),  # 设定 0.1 mas/yr 的物理保底本征弥散
                "PMDEC_DISPERSION": max(pmdec_disp, 0.1),
                "PM_CORR": pm_corr,
                "DISTANCE_PC": distance_pc,
                "PLX_ERROR": 0.2,       # Gaia DR3 标准系统误差保底
                "CMD_DEV": 0.8,         # 默认经验测光容忍度
                "RECONSTRUCTED": True   # 资产打上动态重构标签
            }

            self.logger.info(
                f"✅ [重构成功] 反演物理先验: 距离={distance_pc:.1f}pc, "
                f"自行椭圆弥散=({reconstructed_params['PMRA_DISPERSION']:.2f}, {reconstructed_params['PMDEC_DISPERSION']:.2f}) mas/yr, "
                f"倾斜相关角={pm_corr:.3f}"
            )

            # 5. 自动写回内存配置库或持久化层
            if not hasattr(self, "_dynamic_cache"):
                self._dynamic_cache = {}
            self._dynamic_cache[cluster_key] = reconstructed_params
            
            return reconstructed_params

        except Exception as ex:
            self.logger.error(f"❌ 从数仓历史数据重建星团先验参数失败: {str(ex)}", exc_info=True)
            return {}
