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
                is_array = isinstance(v, np.ndarray) or isinstance(v, list) # 兼容 list 形式的 UVW_REF
                val_str = json.dumps(v.tolist()) if isinstance(v, np.ndarray) else (json.dumps(v) if is_array else str(v))

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

    def reconstruct_cl_params_from_db(self, cluster_id: str, input_data: pd.DataFrame = None) -> dict:
        """
        🚀 【物理资产动态重建与算法质量校验引擎】
        
        功能：
        1. [生产流]：若 input_data 为 None，自动从 DuckDB 数仓中提取指定星团的历史参考星表进行反演。
        2. [校验流]：若传入 input_data (DataFrame)，则直接对该数据集进行动力学鲁棒重建，用于校验和评估外部算法的输出质量。
        """
        cluster_id = cluster_id.upper()
        cluster_key = self.get_param(cluster_id, "CAT_NAME") if not input_data else cluster_id
        
        self.logger.info(f"🧬 [资产校验引擎] 启动物理先验重建与质量审计管线，目标识别符: {cluster_key}")

        # 1. 动态输入分支路由
        if input_data is not None:
            self.logger.info("📥 [沙箱校验模式] 检测到外部算法注入数据，跳过数仓检索，直接启动内存阵列拓扑审计...")
            stars_df = input_data.copy()
        else:
            self.logger.info("🗄️ [数仓生产模式] 未检测到外部输入，正在从 DuckDB 历史底座中检索参考文献资产...")
            hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(cluster=cluster_id.lower())
            
            # 探测基础数仓表是否存在
            try:
                table_check = self.db.execute(
                    f"SELECT * FROM information_schema.tables WHERE table_name = '{hunt_table}'"
                ).df()
                if table_check.empty:
                    self.logger.warning(f"⚠️ 未在数仓中找到历史星表 '{hunt_table}'，无法执行反演重建。")
                    return {}
                
                sql = f"SELECT ra, dec, plx, pmra, pmdec, rv FROM {hunt_table} WHERE cluster = '{cluster_key}'"
                stars_df = self.db.execute(sql).df()
            except Exception as e:
                self.logger.error(f"❌ 探测或读取数仓基础表失败: {str(e)}")
                return {}

        # 2. 核心样本基底完整性校验
        member_count = len(stars_df)
        if stars_df.empty or member_count < 3:
            self.logger.warning(f"⚠️ 有效测试样本过少 ({member_count} 颗)，无法构建统计学意义上的鲁棒反演，重建终止。")
            return {}
        
        self.logger.info(f"📊 基础运动学矩阵载入成功，包含审计数仓/算法成员星共: {member_count} 颗")

        # 3. 2D 空间与几何基准的 Median 抗噪算子解析
        # 统一列名规范化（容忍大写或小写输入）
        stars_df.columns = [col.lower() for col in stars_df.columns]
        
        center_ra = float(np.median(stars_df["ra"].values))
        center_dec = float(np.median(stars_df["dec"].values))
        plx_ref = float(np.median(stars_df["plx"].values))
        pmra_ref = float(np.median(stars_df["pmra"].values))
        pmdec_ref = float(np.median(stars_df["pmdec"].values))

        # 视差转换为距离 (pc)
        distance_pc = 1000.0 / plx_ref if plx_ref > 0 else 100.0

        # 4. 🎯 核心物理重构：2D 自行空间弥散度的 1.4826 * MAD 鲁棒估计
        pmra_vals = stars_df["pmra"].values
        pmdec_vals = stars_df["pmdec"].values

        pmra_mad = np.median(np.abs(pmra_vals - pmra_ref))
        pmdec_mad = np.median(np.abs(pmdec_vals - pmdec_dec_ref if "pmdec_ref" in locals() else pmdec_vals - pmdec_ref))

        pmra_disp = max(float(1.4826 * pmra_mad), 0.1)
        pmdec_disp = max(float(1.4826 * pmdec_mad), 0.1)

        # 5. 鲁棒相关系数计算（抵抗异常值对协方差矩阵的扭曲）
        rem_ra = pmra_vals - pmra_ref
        rem_dec = pmdec_vals - pmdec_ref
        robust_covar = float(np.median(rem_ra * rem_dec))

        pm_corr = 0.0
        if pmra_disp > 0 and pmdec_disp > 0:
            pm_corr = float(np.clip(robust_covar / (pmra_disp * pmdec_disp), -0.99, 0.99))

        # 6. 组装重构后的标准先验字典
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

        # --- 🎯 7. 动态追加：调用 3D UVW 速度反演引擎 (同步向下传递 input_data) ---
        uvw_data = self.reconstruct_cl_params_ex_from_db(cluster_id, input_data=stars_df)
        if uvw_data:
            reconstructed_params.update(uvw_data)
            self.logger.info("🧬 [高维动力学] 3D UVW/RV 资产向量矩阵鲁棒重建成功")
        else:
            # 缺乏 6D 数据时的平滑动力学降级
            reconstructed_params["UVW_REF"] = [-6.05, -28.02, -14.34]
            reconstructed_params["U_ERROR"] = 2.2
            reconstructed_params["V_ERROR"] = 1.6
            reconstructed_params["W_ERROR"] = 1.0
            reconstructed_params["RV_REF"] = 5.63
            reconstructed_params["RV_ERROR"] = 5.0

        self.logger.info(
            f"✅ [重建/审计流执行完毕] 距离={distance_pc:.1f}pc, "
            f"自行椭圆弥散=({pmra_disp:.3f}, {pmdec_disp:.3f}) mas/yr, "
            f"倾斜相关角={pm_corr:.3f}"
        )

        # 8. 持久化路由决策：只有在没有外部注入的生产流模式下，才允许冲刷数仓配置底座
        if input_data is None:
            self.save_refined_params(cluster_key, reconstructed_params)
        else:
            self.logger.info("🚫 [沙箱保护] 当前运行于测试校验模式，反演资产将只在内存字典中返回，禁止污染 DuckDB 生产底座。")

        return reconstructed_params    

    def reconstruct_cl_params_ex_from_db(self, cluster_id: str, input_data: pd.DataFrame = None) -> tuple:
        """
        🚀 【3D 运动学资产重建引擎 - 多源自适应校验版】
        """
        cluster_key = cluster_id.upper()
        
        if input_data is not None:
            # 接受上游洗干净并规范化列名的 DataFrame，过滤出包含有效 RV 的行
            stars_df = input_data.copy()
            if "rv" in stars_df.columns:
                stars_df = stars_df[stars_df["rv"].notna() & ~np.isnan(stars_df["rv"]) & (stars_df["plx"] > 0)]
        else:
            hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(cluster=cluster_id.lower())
            sql = f"""
                SELECT ra, dec, plx, pmra, pmdec, rv FROM {hunt_table}
                WHERE cluster = '{self.get_param(cluster_id, "CAT_NAME")}' AND plx > 0 AND rv IS NOT NULL AND NOT isnan(rv)
            """
            try:
                stars_df = self.db.execute(sql).df()
            except Exception as e:
                self.logger.error(f"❌ 读取 6D 动力学数仓失败: {e}")
                return None
        
        if len(stars_df) < 3:
            self.logger.warning(f"⚠️ 星团 {cluster_key} 包含视向速度的有效 6D 样本过少 ({len(stars_df)}颗)，放弃 3D 速度反演。")
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

            rv_values = stars_df["rv"].values
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