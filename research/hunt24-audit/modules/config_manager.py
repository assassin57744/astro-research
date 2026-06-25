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

    def reconstruct_cl_params_from_db(self, cluster_id: str) -> dict:
        """
        🚀 【物理资产动态重建引擎】

        从现有 db 数据库的 hunt24 原始星表中，反演、重建该星团的核心运动学与几何特征参数。
        """
        cluster_key = self.get_param(cluster_id, "CAT_NAME")
        self.logger.info(f"获取星团在星表中的名称(区分大小写): {cluster_key}")
        self.logger.info(
            f"🧬 [资产管理器] 启动物理先验重建管线，目标星团: {cluster_key}"
        )

        # 1. 嗅探数仓中是否存在 Hunt 参考文献星表数据
        hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(
            cluster=cluster_id.lower()
        )  # 从 manifest 动态解析表名 (e.g., "aln_hunt_m45")

        # 检查表是否存在
        try:
            table_check = self.db.execute(
                f"SELECT * FROM information_schema.tables WHERE table_name = '{hunt_table}'"
            ).df()
            if table_check.empty:
                self.logger.warning(
                    f"⚠️ 未在数仓中找到历史星表 '{hunt_table}'，无法执行反演重建。"
                )
                return {}
        except Exception as e:
            self.logger.error(f"❌ 探测数仓基础表失败: {str(e)}")
            return {}

        # 2. 从数仓中全量拉取该星团的运动学核心切片，交由 Python 进行高鲁棒性统计解析
        sql = f"""
            SELECT ra, dec, plx, pmra, pmdec
            FROM {hunt_table}
            WHERE (cluster) = '{cluster_key}'
        """

        try:
            stars_df = self.db.execute(sql).df()
            member_count = len(stars_df)
            if stars_df.empty or member_count < 3:
                self.logger.warning(
                    f"⚠️ 数仓历史星表 '{hunt_table}' 中未检索到星团 {cluster_key} 的任何历史成员，重建跳过。"
                )
                return {}

            self.logger.info(
                f"📊 成功提取历史资产，包含高确信度历史成员星共: {member_count} 颗"
            )

            # 3. 2D 空间与几何基准的中位数抗噪算子解析
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
            pmdec_mad = np.median(np.abs(pmdec_vals - pmdec_ref))

            # 恢复等效标准差，并设定 0.1 mas/yr 的开放星团物理本征弥散保底上限
            pmra_disp = max(float(1.4826 * pmra_mad), 0.1)
            pmdec_disp = max(float(1.4826 * pmdec_mad), 0.1)

            # 5. 鲁棒相关系数计算（利用中位数残差乘积来抵抗异常值对协方差的扭曲）
            rem_ra = pmra_vals - pmra_ref
            rem_dec = pmdec_vals - pmdec_ref
            robust_covar = float(np.median(rem_ra * rem_dec))

            pm_corr = 0.0
            if pmra_disp > 0 and pmdec_disp > 0:
                pm_corr = float(
                    np.clip(robust_covar / (pmra_disp * pmdec_disp), -0.99, 0.99)
                )

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
                "PLX_ERROR": 0.2,  # Gaia DR3 标准系统误差保底
                "CMD_DEV": 0.8,  # 默认经验测光容忍度
                "RECONSTRUCTED": True,  # 资产打上动态重构标签
            }

            # --- 🎯 动态追加：调用 3D UVW 速度反演引擎 ---
            uvw_data = self.reconstruct_cl_params_ex_from_db(cluster_id)
            if uvw_data:
                # 完美并入资产配置库中
                reconstructed_params.update(uvw_data)
                self.logger.info("数据库重建UVW成功")
            else:
                # 如果没有足够的 rv 数据来重建 3D 速度，给定一套标准的太阳邻域运动学平滑降级基准
                reconstructed_params["UVW_REF"] = [-6.05, -28.02, -14.34]
                reconstructed_params["U_ERROR"] = 2.2
                reconstructed_params["V_ERROR"] = 1.6
                reconstructed_params["W_ERROR"] = 1.0
                reconstructed_params["RV_REF"] = 5.63
                reconstructed_params["RV_ERROR"] = 5.0
                self.logger.warning(
                    f"数据库重建UVW失败, 使用缺省值. | UVW_REF = [-6.05, -28.02, -14.34] | UVW_ERR = [2.2, 1.6, 1.0] | RV_REF = 5.63 | RV_ERROR = 5.0"
                )

            self.logger.info(
                f"✅ [重构成功] 反演物理先验: 距离={distance_pc:.1f}pc, "
                f"自行椭圆弥散=({reconstructed_params['PMRA_DISPERSION']:.3f}, {reconstructed_params['PMDEC_DISPERSION']:.3f}) mas/yr, "
                f"倾斜相关角={pm_corr:.3f}"
            )

            # 5. 持久化至 DuckDB 配置底座并同步刷新内存缓存
            self.save_refined_params(cluster_key, reconstructed_params)

            return reconstructed_params

        except Exception as ex:
            self.logger.error(
                f"❌ 从数仓历史数据重建星团先验参数失败: {str(ex)}", exc_info=True
            )
            return {}


    def reconstruct_cl_params_ex_from_db(self, cluster_id: str) -> tuple:
        """
        🚀 【3D 运动学资产重建引擎】
        
        从现有 db 数据库的 hunt24 星表中，反演、重建该星团在银道直角坐标系中的三轴速度基准 (U_REF, V_REF, W_REF)
        以及各轴的速度弥散度 (U_ERROR, V_ERROR, W_ERROR)。
        """
        self.logger.debug(f"cluster_id : {cluster_id}")
        cluster_key = self.get_param(cluster_id, "CAT_NAME")
        self.logger.info(f"🌌 [动力学引擎] 启动 3D 空间速度 UVW & RV 鲁棒反演管线...")

        hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(
            cluster=cluster_id.lower()
        )  # 从 manifest 动态解析表名 (e.g., "aln_hunt_m45")
        
        # 1. 提取拥有完整 6D 空间信息（必须包含视向速度 rv）的高确信度历史成员
        # Gaia DR3 中很多星没有 rv，所以我们必须用 WHERE rv IS NOT NULL 过滤
        sql = f"""
            SELECT ra, dec, plx, pmra, pmdec, rv
            FROM {hunt_table}
            WHERE cluster = '{cluster_key}' 
              AND plx > 0 
              AND rv IS NOT NULL
              AND NOT isnan(rv)
        """
        
        try:
            stars_df = self.db.execute(sql).df()
            if len(stars_df) < 3:
                self.logger.warning(
                    f"⚠️ 星团 {cluster_key} 包含视向速度的有效样本过少 ({len(stars_df)}颗)，"
                    f"无法完成高精度 3D 速度反演。将平滑降级。"
                )
                return None
            
            self.logger.info(f"📊 提取到带有完整 6D 动力的核心成员星共: {len(stars_df)} 颗，正在流转空间坐标变换...")

            # 2. 利用 Astropy 构建天体物理坐标拓扑
            # 这样写可以完美自动处理 Johnson-Soderblom 变换矩阵的所有岁差和赤道-银道旋转细节
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            # 将 Gaia 观测资产打包为 SkyCoord 向量
            coords = SkyCoord(
                ra=stars_df["ra"].values * u.deg,
                dec=stars_df["dec"].values * u.deg,
                distance=(1000.0 / stars_df["plx"].values) * u.pc,
                pm_ra_cosdec=stars_df["pmra"].values * (u.mas / u.yr),
                pm_dec=stars_df["pmdec"].values * (u.mas / u.yr),
                radial_velocity=stars_df["rv"].values * (u.km / u.s),
                frame="icrs"
            )

            # 3. 💥 一键穿透变换到银道直角坐标系（Galactic LS_XYZ / UVW 速度空间）
            # astropy 的 velocity_components 直接吐出 U, V, W (单位: km/s)
            # 注：Astropy 默认的 U 轴正方向符合右手系定义（指向银心）
            # 🎯 兼容性重构：自适应探测 Astropy 的 3D 直角速度组件 API
            galactic_frame = coords.galactic
            
            if hasattr(galactic_frame, 'v_x'):
                # 现代 / 标准 Astropy 规范：直接作为直角坐标速度属性暴露
                u_arr = galactic_frame.v_x.to(u.km / u.s).value
                v_arr = galactic_frame.v_y.to(u.km / u.s).value
                w_arr = galactic_frame.v_z.to(u.km / u.s).value
            elif hasattr(galactic_frame, 'velocity'):
                # 部分过渡版本的表达形式
                u_arr = galactic_frame.velocity.d_x.to(u.km / u.s).value
                v_arr = galactic_frame.velocity.d_y.to(u.km / u.s).value
                w_arr = galactic_frame.velocity.d_z.to(u.km / u.s).value
            else:
                # 终极保底：强行解构其微分表示层 (Differentials)
                cartesian_vel = galactic_frame.data.differentials['s']
                u_arr = cartesian_vel.d_x.to(u.km / u.s).value
                v_arr = cartesian_vel.d_y.to(u.km / u.s).value
                w_arr = cartesian_vel.d_z.to(u.km / u.s).value

            # 4. 采用具有鲁棒抗噪特征的 MEDIAN 算子锁定星团集体运动中心
            u_ref = float(np.median(u_arr))
            v_ref = float(np.median(v_arr))
            w_ref = float(np.median(w_arr))

            # 5. 🎯 核心物理重构：使用 1.4826 * MAD 强制剔除联星与外围噪星对速度弥散度的恶性扩散
            u_mad = np.median(np.abs(u_arr - u_ref))
            v_mad = np.median(np.abs(v_arr - v_ref))
            w_mad = np.median(np.abs(w_arr - w_ref))
            
            # 给定 0.8 km/s 的物理本征弥散下限保底
            u_disp = max(float(1.4826 * u_mad), 0.8)
            v_disp = max(float(1.4826 * v_mad), 0.8)
            w_disp = max(float(1.4826 * w_mad), 0.8)

            # 6. 🎯 核心物理重构：同理修复视向速度（RV）的先验弥散
            rv_values = stars_df["rv"].values
            rv_ref = float(np.median(rv_values))
            
            rv_mad = np.median(np.abs(rv_values - rv_ref))
            rv_error = max(float(1.4826 * rv_mad), 1.0)  # 1.0 km/s 物理保底

            uvw_results = {
                "UVW_REF": [u_ref, v_ref, w_ref],
                "U_ERROR": u_disp,
                "V_ERROR": v_disp,
                "W_ERROR": w_disp,
                "RV_REF": rv_ref,
                "RV_ERROR": rv_error,
                "KINE_6D_COUNT": len(stars_df)
            }

            self.logger.info(
                f"✅ [UVW & RV 重建成功] 星团 {cluster_key} 核心动力学参数更新:\n"
                f"   🪐 UVW_REF 基准 (km/s) -> U: {u_ref:.2f}, V: {v_ref:.2f}, W: {w_ref:.2f}\n"
                f"   📡 RV_REF  基准 (km/s) -> {rv_ref:.2f} (本征弥散 RV_ERROR: {rv_error:.2f} km/s)\n"
                f"   📈 速度三轴弥散 (km/s) -> U_err: {u_disp:.2f}, V_err: {v_disp:.2f}, W_err: {w_disp:.2f}"
            )
            
            return uvw_results

        except Exception as ex:
            self.logger.error(f"❌ 反演 UVW 空间动力学矩阵发生非预期崩溃: {str(ex)}", exc_info=True)
            return None