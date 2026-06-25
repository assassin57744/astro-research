# modules/config_manager.py
import logging
import json
import numpy as np
import duckdb
import pandas as pd
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
                is_array = isinstance(v, np.ndarray) or isinstance(
                    v, list
                )  # 兼容 list 形式的 UVW_REF
                val_str = (
                    json.dumps(v.tolist())
                    if isinstance(v, np.ndarray)
                    else (json.dumps(v) if is_array else str(v))
                )

                # 🎯 显式将 updated_at 载入覆盖序列，强迫引擎刷新时间戳底座
                upsert_sql = f"""
                    INSERT OR REPLACE INTO cluster_refined_params (cluster_id, param_name, param_value, is_array, updated_at)
                    VALUES ('{cluster_id}', '{k}', '{val_str}', {is_array}, CURRENT_TIMESTAMP)
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

    def reconstruct_cl_params_from_db(
            self, 
            cluster_id: str, 
            input_data: pd.DataFrame = None, 
            metric: str = "median", 
            use_gaia_raw: bool = False
        ) -> dict:
        """
        🚀 【物理资产动态重建与多指标/多数据源质量校验引擎】

        功能：
        1. [生产流]：若 input_data 为 None，自动从 DuckDB 数仓中提取指定星团的历史参考星表进行反演。
        2. [校验流]：若传入 input_data (DataFrame)，则直接对该数据集进行动力学鲁棒重建，用于校验和评估外部算法的输出质量。

        参数：
        - cluster_id: 星团唯一标识符。
        - input_data: 可选。外部算法注入的 DataFrame，用于沙箱校验。如果传入，则直接基于传入数据计算。
        - metric: 统计算子选择，"median" (缺省) 或 "mean"。
        - use_gaia_raw: 开关。False (缺省) 使用 hunt 视图数据；True 使用 hunt 的样本束缚联查 Gaia 原始物理量。
        """
        cluster_id = cluster_id.upper()
        cluster_key = (
            self.get_param(cluster_id, "CAT_NAME") if not input_data else cluster_id
        )
        metric_mode = metric.lower()
        
        self.logger.info(
            f"🧬 [资产校验引擎] 启动重建管线 | 目标: {cluster_key} | "
            f"算子: {metric_mode.upper()} | 原始数据开关: {use_gaia_raw}"
        )

        # 1. 动态输入分支路由
        if input_data is not None:
            self.logger.info("📥 [沙箱校验模式] 检测到外部算法注入数据，跳过数仓检索...")
            stars_df = input_data.copy()
        else:
            self.logger.info("🗄️ [数仓生产模式] 开始动态构建多源自适应 SQL 检索...")
            hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(cluster=cluster_id.lower())
            
            try:
                table_check = self.db.execute(
                    f"SELECT * FROM information_schema.tables WHERE table_name = '{hunt_table}'"
                ).df()
                if table_check.empty:
                    self.logger.warning(f"⚠️ 未在数仓中找到历史星表 '{hunt_table}'，无法执行反演重建。")
                    return {}
                
                # 🎯 【核心修改】：依据开关，动态组装 SQL 查询
                if use_gaia_raw:
                    self.logger.info("📡 [数据源：Gaia 原始仓] 启动 source_id 拓扑联查，正在提取 Gaia 核心流原始运动学向量...")
                    # 假设统一关联的原始表为 gaia_dr3，利用 hunt 表作为成员星样本过滤器
                    sql = f"""
                        SELECT 
                            g.ra, g.dec, g.parallax as plx, g.pmra, g.pmdec, g.radial_velocity as rv 
                        FROM {hunt_table} h
                        INNER JOIN gaia_dr3 g ON h.source_id = g.source_id
                        WHERE h.cluster = '{cluster_key}'
                    """
                else:
                    self.logger.info("🧹 [数据源：Hunt 清洗仓] 正在直接提取清洗/改正后的流水线视图数据...")
                    sql = f"SELECT ra, dec, plx, pmra, pmdec, rv FROM {hunt_table} WHERE cluster = '{cluster_key}'"
                
                stars_df = self.db.execute(sql).df()
            except Exception as e:
                self.logger.error(f"❌ 探测或动态多源读取数仓基础表失败: {str(e)}")
                return {}

        # 2. 核心样本基底完整性校验
        member_count = len(stars_df)
        if stars_df.empty or member_count < 3:
            self.logger.warning(f"⚠️ 有效测试样本过少 ({member_count} 颗)，无法构建统计学反演。")
            return {}
        
        # 统一列名规范化
        stars_df.columns = [col.lower() for col in stars_df.columns]
        
        ra_vals = stars_df["ra"].values
        dec_vals = stars_df["dec"].values
        plx_vals = stars_df["plx"].values
        pmra_vals = stars_df["pmra"].values
        pmdec_vals = stars_df["pmdec"].values

        # 3. 核心统计解析
        if metric_mode == "mean":
            center_ra = float(np.mean(ra_vals))
            center_dec = float(np.mean(dec_vals))
            plx_ref = float(np.mean(plx_vals))
            pmra_ref = float(np.mean(pmra_vals))
            pmdec_ref = float(np.mean(pmdec_vals))

            # 传统标准差弥散
            pmra_disp = max(float(np.std(pmra_vals)), 0.1)
            pmdec_disp = max(float(np.std(pmdec_vals)), 0.1)

            # 传统皮尔逊相关系数
            cov_matrix = np.cov(pmra_vals, pmdec_vals)
            pm_corr = 0.0
            if pmra_disp > 0 and pmdec_disp > 0 and cov_matrix.ndim == 2:
                pm_corr = float(
                    np.clip(cov_matrix[0, 1] / (pmra_disp * pmdec_disp), -0.99, 0.99)
                )
        else:
            # 保持原有高鲁棒性的中位数与 MAD
            center_ra = float(np.median(ra_vals))
            center_dec = float(np.median(dec_vals))
            plx_ref = float(np.median(plx_vals))
            pmra_ref = float(np.median(pmra_vals))
            pmdec_ref = float(np.median(pmdec_vals))

            pmra_mad = np.median(np.abs(pmra_vals - pmra_ref))
            pmdec_mad = np.median(np.abs(pmdec_vals - pmdec_ref))
            pmra_disp = max(float(1.4826 * pmra_mad), 0.1)
            pmdec_disp = max(float(1.4826 * pmdec_mad), 0.1)

            # 鲁棒相关系数计算
            rem_ra = pmra_vals - pmra_ref
            rem_dec = pmdec_vals - pmdec_ref
            robust_covar = float(np.median(rem_ra * rem_dec))
            pm_corr = 0.0
            if pmra_disp > 0 and pmdec_disp > 0:
                pm_corr = float(
                    np.clip(robust_covar / (pmra_disp * pmdec_disp), -0.99, 0.99)
                )

        # --- 📐 视差转换为距离 (pc)  ---
        if plx_ref > 0:
            if metric_mode == "mean":
                # 1. 传统平均值分支：引入泰勒级数二阶偏置修正
                plx_disp = np.std(plx_vals)
                rel_err = plx_disp / plx_ref

                if rel_err < 0.2:  # 相对色散在 20% 以内时，泰勒修正极度精准且安全
                    distance_pc = (1000.0 / plx_ref) * (1.0 + rel_err**2)
                    self.logger.debug(
                        f"📐 [物理修正] Mean 分支触发非线性视差偏置修正: +{rel_err**2:.4f}"
                    )
                else:
                    # 相对误差过大时（如极远或极暗星团），传统平均值发散，被迫降级为硬转换
                    distance_pc = 1000.0 / plx_ref
            else:
                # 2. 默认中位数分支：完美共变，直接硬转换，无任何数学偏差
                distance_pc = 1000.0 / plx_ref
        else:
            distance_pc = 100.0  # 兜底边界

        # 4. 组装重构后的标准先验字典
        reconstructed_params = {
            "ID_NAME": cluster_key,
            "CENTER_RA": center_ra,
            "CENTER_DEC": center_dec,
            "PLX_REF": plx_ref,
            "PMRA_REF": pmra_ref,
            "PMDEC_REF": pmdec_ref,
            "PMRA_DISPERSION": pmra_disp,
            "PMDEC_DISPERSION": pmdec_disp,
            "PM_CORR": pm_corr,
            "DISTANCE_PC": distance_pc,
            "PLX_ERROR": 0.2,
            "CMD_DEV": 0.8,
            "RECONSTRUCTED": True,
        }

        # --- 🎯 6. 动态追加：调用 3D UVW 速度反演引擎 (同步向下传递开关与 metric) ---
        uvw_data = self.reconstruct_cl_params_ex_from_db(
            cluster_id, input_data=stars_df, metric=metric_mode, use_gaia_raw=use_gaia_raw
        )
        if uvw_data:
            reconstructed_params.update(uvw_data)
            self.logger.info(
                f"🧬 [高维动力学] 3D UVW/RV 资产向量矩阵基于 [{metric_mode.upper()}] 重建成功"
            )
        else:
            # 缺乏 6D 数据时的平滑动力学降级
            reconstructed_params["UVW_REF"] = [-6.05, -28.02, -14.34]
            reconstructed_params["U_ERROR"] = 2.2
            reconstructed_params["V_ERROR"] = 1.6
            reconstructed_params["W_ERROR"] = 1.0
            reconstructed_params["RV_REF"] = 5.63
            reconstructed_params["RV_ERROR"] = 5.0

        self.logger.info(
            f"✅ [重建/审计流执行完毕] 算子={metric_mode.upper()} | 距离={distance_pc:.1f}pc, "
            f"自行椭圆弥散=({pmra_disp:.3f}, {pmdec_disp:.3f}) mas/yr, "
            f"倾斜相关角={pm_corr:.3f}"
        )

        # 7. 持久化路由决策：只有在使用原始清洗库生成的标准生产流模式下，才允许写回底座，防沙箱污染
        if input_data is None and not use_gaia_raw:
            self.save_refined_params(cluster_key, reconstructed_params)
        else:
            self.logger.info("🚫 [沙箱保护] 当前运行于测试数据源或外部注入模式，反演资产仅在内存中返回。")

        return reconstructed_params

    def reconstruct_cl_params_ex_from_db(
        self, 
        cluster_id: str, 
        input_data: pd.DataFrame = None, 
        metric: str = "median", 
        use_gaia_raw: bool = False
    ) -> tuple:
        """
        🚀 【3D 运动学资产重建引擎 - 多源自适应多源数据校验版】
        """
        cluster_key = cluster_id.upper()
        metric_mode = metric.lower()
        
        if input_data is not None:
            stars_df = input_data.copy()
            if "rv" in stars_df.columns:
                stars_df = stars_df[stars_df["rv"].notna() & ~np.isnan(stars_df["rv"]) & (stars_df["plx"] > 0)]
        else:
            hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(cluster=cluster_id.lower())
            
            # 🎯 【核心修改】：依据数据源开关，动态组装 6D 动力学 SQL 查询
            if use_gaia_raw:
                sql = f"""
                    SELECT 
                        g.ra, g.dec, g.parallax as plx, g.pmra, g.pmdec, g.radial_velocity as rv 
                    FROM {hunt_table} h
                    INNER JOIN gaia_dr3 g ON h.source_id = g.source_id
                    WHERE h.cluster = '{self.get_param(cluster_id, "CAT_NAME")}' 
                      AND g.parallax > 0 AND g.radial_velocity IS NOT NULL AND NOT isnan(g.radial_velocity)
                """
            else:
                sql = f"""
                    SELECT ra, dec, plx, pmra, pmdec, rv FROM {hunt_table}
                    WHERE cluster = '{self.get_param(cluster_id, "CAT_NAME")}' 
                      AND plx > 0 AND rv IS NOT NULL AND NOT isnan(rv)
                """
                
            try:
                stars_df = self.db.execute(sql).df()
            except Exception as e:
                self.logger.error(f"❌ 读取 6D 动力学多源数仓失败: {e}")
                return None
        
        if len(stars_df) < 3:
            self.logger.warning(
                f"⚠️ 星团 {cluster_key} 包含视向速度的有效 6D 样本过少 ({len(stars_df)}颗)，放弃 3D 速度反演。"
            )
            return None

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            # 天体物理直角坐标拓扑旋转
            coords = SkyCoord(
                ra=stars_df["ra"].values * u.deg,
                dec=stars_df["dec"].values * u.deg,
                distance=(1000.0 / stars_df["plx"].values) * u.pc,
                pm_ra_cosdec=stars_df["pmra"].values * (u.mas / u.yr),
                pm_dec=stars_df["pmdec"].values * (u.mas / u.yr),
                radial_velocity=stars_df["rv"].values * (u.km / u.s),
                frame="icrs"
            )

            galactic_frame = coords.galactic
            if hasattr(galactic_frame, 'v_x'):
                u_arr = galactic_frame.v_x.to(u.km / u.s).value
                v_arr = galactic_frame.v_y.to(u.km / u.s).value
                w_arr = galactic_frame.v_z.to(u.km / u.s).value
            elif hasattr(galactic_frame, 'velocity'):
                u_arr = galactic_frame.velocity.d_x.to(u.km / u.s).value
                v_arr = galactic_frame.velocity.d_y.to(u.km / u.s).value
                w_arr = galactic_frame.velocity.d_z.to(u.km / u.s).value
            else:
                cartesian_vel = galactic_frame.data.differentials['s']
                u_arr = cartesian_vel.d_x.to(u.km / u.s).value
                v_arr = cartesian_vel.d_y.to(u.km / u.s).value
                w_arr = cartesian_vel.d_z.to(u.km / u.s).value

            rv_values = stars_df["rv"].values

            # 4. 根据指定的基准指标路由解算质心三维运动学和空间色散度
            if metric_mode == "mean":
                u_ref = float(np.mean(u_arr))
                v_ref = float(np.mean(v_arr))
                w_ref = float(np.mean(w_arr))

                # 传统标准差估计弥散
                u_disp = max(float(np.std(u_arr)), 0.8)
                v_disp = max(float(np.std(v_arr)), 0.8)
                w_disp = max(float(np.std(w_arr)), 0.8)

                rv_ref = float(np.mean(rv_values))
                rv_error = max(float(np.std(rv_values)), 1.0)
            else:
                # 高抗噪中位数范式
                u_ref = float(np.median(u_arr))
                v_ref = float(np.median(v_arr))
                w_ref = float(np.median(w_arr))

                # MAD 鲁棒误差解算
                u_mad = np.median(np.abs(u_arr - u_ref))
                v_mad = np.median(np.abs(v_arr - v_ref))
                w_mad = np.median(np.abs(w_arr - w_ref))

                u_disp = max(float(1.4826 * u_mad), 0.8)
                v_disp = max(float(1.4826 * v_mad), 0.8)
                w_disp = max(float(1.4826 * w_mad), 0.8)

                rv_ref = float(np.median(rv_values))
                rv_mad = np.median(np.abs(rv_values - rv_ref))
                rv_error = max(float(1.4826 * rv_mad), 1.0)

            return {
                "UVW_REF": [u_ref, v_ref, w_ref],
                "U_ERROR": u_disp,
                "V_ERROR": v_disp,
                "W_ERROR": w_disp,
                "RV_REF": rv_ref,
                "RV_ERROR": rv_error,
                "KINE_6D_COUNT": len(stars_df)
            }
        except Exception as ex:
            self.logger.error(f"❌ 空间动力学矩阵反演崩溃: {str(ex)}", exc_info=True)
            return None
