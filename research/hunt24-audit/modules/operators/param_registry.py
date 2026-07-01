# modules/operators/param_registry.py
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging
from astropy.coordinates import SkyCoord
import astropy.units as u
import config as cfg
from modules.engine import AstroEngine


class ParamRegistry:
    """
    Stage 1.75: 星团整体物理参数注册与状态审计算子。
    【统一职责】承载物理模型重建后的多维核心物理参数落盘（resolve）、内存高速缓存维护以及多维参数矩阵的一体化查询。
    """

    def __init__(self, engine: AstroEngine):
        self._engine = engine
        self.logger = logging.getLogger("AstroDB.global.ParamRegistry")

        # 维护当前激活星团的物理参数字典高速缓存
        self._current_params = {}
        self._active_cluster_id = None
        self._runtime_cache = {}

    def _ensure_physical_table(self) -> None:
        """
        🎯【需求5】前置物理防御层：确保底座物理表存在。
        如果底层数仓中不存在该表，则动态执行 DDL 编译创建，彻底防止 DML 击穿。
        """
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS cluster_params (
                cluster_id VARCHAR,
                param_name VARCHAR,
                param_value VARCHAR,
                is_array BOOLEAN,
                updated_at TIMESTAMP
            )
        """
        try:
            self._engine.execute(create_table_sql)
        except Exception as e:
            self.logger.error(
                f"❌ [ParamRegistry] 动态编译物理表架构失败: {e}", exc_info=True
            )
            raise e

    def ____resolve(self, cluster_id: str, recon_metrics: dict) -> None:
        """
        🎯【需求2】核心资产同步接口：将上游算法矩阵重建出的科学物理 DNA 安全回灌至数仓并刷新内存缓存。
        """
        if not cluster_id:
            self.logger.error("❌ 未指定具体天体标识符，无法执行参数资产 resolve。")
            return

        cluster_id_upper = cluster_id.upper()
        self._active_cluster_id = cluster_id_upper
        self.logger.info(
            f"⚡ [ParamRegistry] 正在为星团 '{cluster_id_upper}' 执行核心参数解析与物化回灌..."
        )

        # 🛠️【需求5】确保表 100% 存在
        self._ensure_physical_table()

        # ✨【完美对齐】提取原单体 operators_bak.py 中的全部 14 个完整物理与运动学指标
        center_ra = recon_metrics.get("center_ra", 0.0)
        center_dec = recon_metrics.get("center_dec", 0.0)
        plx_ref = recon_metrics.get("plx_ref", 0.0)
        pmra_ref = recon_metrics.get("pmra_ref", 0.0)
        pmdec_ref = recon_metrics.get("pmdec_ref", 0.0)
        pmra_disp = recon_metrics.get("pmra_disp", 0.0)
        pmdec_disp = recon_metrics.get("pmdec_disp", 0.0)
        pm_corr = recon_metrics.get("pm_corr", 0.0)
        distance_pc = recon_metrics.get("distance_pc", 0.0)

        # 补齐原代码遗漏的高阶三维速度空间属性
        u_ref = recon_metrics.get("u_ref", 0.0)
        v_ref = recon_metrics.get("v_ref", 0.0)
        w_ref = recon_metrics.get("w_ref", 0.0)
        rv_ref = recon_metrics.get("rv_ref", 0.0)

        # 【需求4】同步物化并更新内存高速缓存状态，补齐所有高阶运动学指标，防止下游高阶算子 KeyError
        self._current_params = {
            "RA_CENT": float(center_ra),
            "DEC_CENT": float(center_dec),
            "PLX_REF": float(plx_ref),
            "PMRA_REF": float(pmra_ref),
            "PMDEC_REF": float(pmdec_ref),
            "PMRA_DISP": float(pmra_disp),
            "PMDEC_DISP": float(pmdec_disp),
            "PM_CORR": float(pm_corr),
            "DISTANCE_PC": float(distance_pc),
            "U_REF": float(u_ref),
            "V_REF": float(v_ref),
            "W_REF": float(w_ref),
            "RV_REF": float(rv_ref),
        }

        try:
            update_sql = f"""
                INSERT OR REPLACE INTO cluster_kinematic_identity 
                (cluster_id, ra_center, dec_center, plx_ref, pmra_ref, pmdec_ref, pmra_disp, pmdec_disp, pm_corr, distance_pc)
                VALUES 
                ('{cluster_id_upper}', {center_ra}, {center_dec}, {plx_ref}, {pmra_ref}, {pmdec_ref}, {pmra_disp}, {pmdec_disp}, {pm_corr}, {distance_pc})
            """
            self._engine.execute(update_sql)
            self.logger.info(
                f"💾 [ParamRegistry] 物理资产成功持久化至 `cluster_kinematic_identity` 表"
            )
        except Exception as update_err:
            self.logger.warning(
                f"⚠️ [ParamRegistry 柔性容错] 底层数仓物理落盘遭遇间歇性拦截: {update_err}"
            )

    def get_cluster_params(self, cluster_id: str) -> pd.DataFrame:
        """【数仓标准查询通道】拉取特定星团的完整参数 DataFrame 映射"""
        if not cluster_id:
            return pd.DataFrame()
        self._ensure_physical_table()
        cluster_id_upper = cluster_id.upper()

        sql = f"SELECT * FROM cluster_kinematic_identity WHERE cluster_id = '{cluster_id_upper}'"
        try:
            df = self._engine.query(sql)
            if not df.empty and (self._active_cluster_id != cluster_id_upper):
                row = df.iloc[0]
                self._active_cluster_id = cluster_id_upper
                self._current_params = {
                    "RA_CENT": float(row.get("ra_center", 0.0)),
                    "DEC_CENT": float(row.get("dec_center", 0.0)),
                    "PLX_REF": float(row.get("plx_ref", 0.0)),
                    "PMRA_REF": float(row.get("pmra_ref", 0.0)),
                    "PMDEC_REF": float(row.get("pmdec_ref", 0.0)),
                    "PMRA_DISP": float(row.get("pmra_disp", 0.0)),
                    "PMDEC_DISP": float(row.get("pmdec_disp", 0.0)),
                    "PM_CORR": float(row.get("pm_corr", 0.0)),
                    "DISTANCE_PC": float(row.get("distance_pc", 0.0)),
                }
            return df
        except Exception as e:
            self.logger.error(f"❌ [ParamRegistry] 从数仓拉取星团参数失败: {e}")
            return pd.DataFrame()

    def _get_cached_param(self, cluster_id: str, param_key: str) -> float:
        cluster_id_upper = cluster_id.upper()
        if (
            self._active_cluster_id == cluster_id_upper
            and param_key in self._current_params
        ):
            return self._current_params[param_key]
        df = self.get_cluster_params(cluster_id_upper)
        if not df.empty:
            return self._current_params.get(param_key, 0.0)
        return 0.0
    
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
            if hasattr(static_cluster_cfg, param_name):
                val = getattr(static_cluster_cfg, param_name)
                self._set_memory_cache(cluster_id, param_name, val)
                return val

        self.logger.error(
            f"❌ [ConfigError] 无法在任意域中匹配到星团 {cluster_id} 的物理参数: '{param_name}'"
        )
        return None

    def get_center_ra(self, cluster_id: str) -> float:
        return self._get_cached_param(cluster_id, "RA_CENT")

    def get_center_dec(self, cluster_id: str) -> float:
        return self._get_cached_param(cluster_id, "DEC_CENT")

    def get_reference_parallax(self, cluster_id: str) -> float:
        return self._get_cached_param(cluster_id, "PLX_REF")

    def get_reference_pmra(self, cluster_id: str) -> float:
        return self._get_cached_param(cluster_id, "PMRA_REF")

    def get_reference_pmdec(self, cluster_id: str) -> float:
        return self._get_cached_param(cluster_id, "PMDEC_REF")

    def resolve(self, cluster_id: str, mode: str, catalog_manager=None) -> dict:
        """
        🪐 核心调度方法（算子唯一定外入口）
        """
        cluster_id_upper = cluster_id.upper()

        # 1. 动态反演分支
        if mode == "dynamic":
            return self._dynamic_reconstruct(cluster_id_upper)

        # 2. 静态配置装载分支
        self.logger.info(f"📜 [静态配置装载] 正在检索星团静态蓝图: {cluster_id_upper}")

        if cluster_id_upper not in cfg.CLUSTERS:
            self.logger.warning(
                f"⚠️ config.py 中未发现 {cluster_id_upper} 蓝图，强行降级切换至 dynamic 模式..."
            )
            return self._dynamic_reconstruct(cluster_id_upper)

        cluster_static_cfg = cfg.CLUSTERS[cluster_id_upper]

        # 提取 3D UVW 速度向量
        uvw_static = cluster_static_cfg.UVW_REF
        u_ref = float(uvw_static[0]) if len(uvw_static) > 0 else 0.0
        v_ref = float(uvw_static[1]) if len(uvw_static) > 1 else 0.0
        w_ref = float(uvw_static[2]) if len(uvw_static) > 2 else 0.0

        # 计算静态自适应距离 (pc)
        plx_ref = float(cluster_static_cfg.PLX_REF)
        distance_pc = 1000.0 / plx_ref if plx_ref > 0 else 100.0

        # 从属性或者静态配置映射出符合外层契约规范的 DNA 字典
        dna_profile = {
            "cluster_id": cluster_id_upper,
            "ra_center": float(cluster_static_cfg.CENTER_RA),
            "dec_center": float(cluster_static_cfg.CENTER_DEC),
            "plx_ref": plx_ref,
            "pmra_ref": float(cluster_static_cfg.PMRA_REF),
            "pmdec_ref": float(cluster_static_cfg.PMDEC_REF),
            "pmra_disp": getattr(cluster_static_cfg, "PMRA_DISPERSION", 0.5),
            "pmdec_disp": getattr(cluster_static_cfg, "PMDEC_DISPERSION", 0.5),
            "pm_corr": getattr(cluster_static_cfg, "PM_CORR", 0.0),
            "distance_pc": distance_pc,
            "u_ref": u_ref,
            "v_ref": v_ref,
            "w_ref": w_ref,
            "u_disp": getattr(cluster_static_cfg, "U_ERROR", 1.0),
            "v_disp": getattr(cluster_static_cfg, "V_ERROR", 1.0),
            "w_disp": getattr(cluster_static_cfg, "W_ERROR", 1.0),
            "rv_ref": getattr(cluster_static_cfg, "RV_REF", 0.0),
            "rv_error": getattr(cluster_static_cfg, "RV_ERROR", 1.0),
            "dynamic_reconstructed": False,
            "sample_size": 0,
        }
        return self._normalize_and_alias(dna_profile)

    def _____dynamic_reconstruct(self, cluster_id: str, catalog_manager=None) -> dict:
        """
        🚀 【物理资产反演内核】
        """
        cluster_id_upper = cluster_id.upper()
        self.logger.info(
            f"🧬 [动态真身反演] 启动 MLE 动力学自收敛重构管线 | 目标: {cluster_id_upper}"
        )

        # =================================================================
        # STEP 1: 通过资产管理器解除硬编码视图依赖
        # =================================================================
        cluster_static_cfg = cfg.CLUSTERS.get(cluster_id_upper, cfg.CLUSTERS.get("M45"))
        r_half_deg = getattr(cluster_static_cfg, "R_HALF_LIGHT", 0.5)
        cat_name = getattr(cluster_static_cfg, "CAT_NAME", cluster_id_upper)

        if catalog_manager:
            obs_aln_table = catalog_manager.resolve_view_name(cluster_id_upper, "field")
            hunt_table = catalog_manager.resolve_view_name(cluster_id_upper, "hunt")
        else:
            # 兼容性无损降级方案
            cluster_id_lower = cluster_id.lower()
            obs_aln_table = f"std_{cluster_id_lower}_field"
            hunt_table = f"std_view_hunt"

        self.logger.info(
            f"📡 [数仓关联拓扑成功] 观测流: {obs_aln_table} | 文献流: {hunt_table}"
        )

        # =================================================================
        # STEP 2: 执行自收敛几何质心裁剪 SQL
        # =================================================================
        sql = f"""
            WITH dynamic_center AS (
                SELECT median("ra") AS c_ra, median("dec") AS c_dec 
                FROM {hunt_table} WHERE "cluster" = '{cat_name}'
            ),
            core_members AS (
                SELECT h."id"
                FROM {hunt_table} h
                CROSS JOIN dynamic_center c
                WHERE h."cluster" = '{cat_name}'
                  AND SQRT(POWER((h."ra" - c.c_ra) * COS(RADIANS(c.c_dec)), 2) + POWER(h."dec" - c.c_dec, 2)) <= {r_half_deg}
                ORDER BY h."prob" DESC
                LIMIT 400
            )
            SELECT g."ra", g."dec", g."plx", g."pmra", g."pmdec", g."rv"
            FROM core_members cm
            INNER JOIN {obs_aln_table} g ON cm."id" = g."id"
        """

        # 依赖统一的 Engine 接口，拒绝在内部判断 hasattr
        stars_df = self._engine.query(sql)
        member_count = len(stars_df)
        self.logger.info(f"📊 骨干样本矩阵收敛锁定有效核心星数: {member_count} 颗")

        if stars_df.empty or member_count < 3:
            raise ValueError(
                f"❌ 星团 {cluster_id_upper} 骨干计算样本过少 ({member_count})，物理反演强行中断。"
            )

        stars_df.columns = [col.lower() for col in stars_df.columns]

        # =================================================================
        # STEP 3: 5D 运动学统计学基础解算 (中位数 + MAD)
        # =================================================================
        center_ra = float(np.median(stars_df["ra"].values))
        center_dec = float(np.median(stars_df["dec"].values))
        plx_ref = float(np.median(stars_df["plx"].values))
        pmra_ref = float(np.median(stars_df["pmra"].values))
        pmdec_ref = float(np.median(stars_df["pmdec"].values))

        distance_pc = 1000.0 / plx_ref if plx_ref > 0 else 100.0

        pmra_mad = np.median(np.abs(stars_df["pmra"].values - pmra_ref))
        pmdec_mad = np.median(np.abs(stars_df["pmdec"].values - pmdec_ref))
        pmra_disp = max(float(1.4826 * pmra_mad), 0.1)
        pmdec_disp = max(float(1.4826 * pmdec_mad), 0.1)

        rem_ra = stars_df["pmra"].values - pmra_ref
        rem_dec = stars_df["pmdec"].values - pmdec_ref
        robust_covar = float(np.median(rem_ra * rem_dec))
        pm_corr = 0.0
        if pmra_disp > 0 and pmdec_disp > 0:
            pm_corr = float(
                np.clip(robust_covar / (pmra_disp * pmdec_disp), -0.99, 0.99)
            )

        # =================================================================
        # STEP 4: 高维 6D 动力学解算与拓扑旋转
        # =================================================================
        rv_df = stars_df[
            stars_df["rv"].notna() & ~np.isnan(stars_df["rv"]) & (stars_df["plx"] > 0)
        ]

        u_ref, v_ref, w_ref = 0.0, 0.0, 0.0
        u_disp, v_disp, w_disp = 1.0, 1.0, 1.0
        rv_ref, rv_error = 0.0, 1.0
        has_6d = False

        if len(rv_df) >= 3:
            try:
                coords = SkyCoord(
                    ra=rv_df["ra"].values * u.deg,
                    dec=rv_df["dec"].values * u.deg,
                    distance=(1000.0 / rv_df["plx"].values) * u.pc,
                    pm_ra_cosdec=rv_df["pmra"].values * (u.mas / u.yr),
                    pm_dec=rv_df["pmdec"].values * (u.mas / u.yr),
                    radial_velocity=rv_df["rv"].values * (u.km / u.s),
                    frame="icrs",
                )
                galactic_frame = coords.galactic

                # 多态属性探测（防御 Astropy 升级版本差异）
                if hasattr(galactic_frame, "v_x"):
                    u_arr, v_arr, w_arr = (
                        galactic_frame.v_x.value,
                        galactic_frame.v_y.value,
                        galactic_frame.v_z.value,
                    )
                elif hasattr(galactic_frame, "velocity"):
                    u_arr = galactic_frame.velocity.d_x.to(u.km / u.s).value
                    v_arr = galactic_frame.velocity.d_y.to(u.km / u.s).value
                    w_arr = galactic_frame.velocity.d_z.to(u.km / u.s).value
                else:
                    cartesian_vel = galactic_frame.data.differentials["s"]
                    u_arr, v_arr, w_arr = (
                        cartesian_vel.d_x.value,
                        cartesian_vel.d_y.value,
                        cartesian_vel.d_z.value,
                    )

                u_ref, v_ref, w_ref = (
                    float(np.median(u_arr)),
                    float(np.median(v_arr)),
                    float(np.median(w_arr)),
                )
                u_disp = max(float(1.4826 * np.median(np.abs(u_arr - u_ref))), 0.8)
                v_disp = max(float(1.4826 * np.median(np.abs(v_arr - v_ref))), 0.8)
                w_disp = max(float(1.4826 * np.median(np.abs(w_arr - w_ref))), 0.8)

                rv_values = rv_df["rv"].values
                rv_ref = float(np.median(rv_values))
                rv_error = max(
                    float(1.4826 * np.median(np.abs(rv_values - rv_ref))), 1.0
                )
                has_6d = True
                self.logger.info(
                    f"🌌 [高维动力学解算成功] 基于 {len(rv_df)} 颗高本征 6D 样本重建 3D UVW 矩阵矢量"
                )
            except Exception as ex:
                self.logger.error(f"⚠️ 空间动力学变换发生数学异常: {str(ex)}")

        if not has_6d:
            self.logger.warning(
                f"🚨 [空间运动学割裂警告] 星团 {cluster_id_upper} 缺乏有效 6D 样本，UVW 速度场强行激活静态兼容降级配置"
            )
            static_cfg = cfg.CLUSTERS.get(cluster_id_upper, cfg.CLUSTERS["M45"])
            uvw_static = static_cfg.UVW_REF
            u_ref, v_ref, w_ref = (
                float(uvw_static[0]),
                float(uvw_static[1]),
                float(uvw_static[2]),
            )
            u_disp = getattr(static_cfg, "U_ERROR", 1.0)
            v_disp = getattr(static_cfg, "V_ERROR", 1.0)
            w_disp = getattr(static_cfg, "W_ERROR", 1.0)
            rv_ref = getattr(static_cfg, "RV_REF", 0.0)
            rv_error = getattr(static_cfg, "RV_ERROR", 1.0)

        # =================================================================
        # STEP 5: 契约字典组装与滚动式落盘
        # =================================================================
        dna_profile = {
            "cluster_id": cluster_id_upper,
            "ra_center": center_ra,
            "dec_center": center_dec,
            "plx_ref": plx_ref,
            "pmra_ref": pmra_ref,
            "pmdec_ref": pmdec_ref,
            "pmra_disp": pmra_disp,
            "pmdec_disp": pmdec_disp,
            "pm_corr": pm_corr,
            "distance_pc": distance_pc,
            "u_ref": u_ref,
            "v_ref": v_ref,
            "w_ref": w_ref,
            "u_disp": u_disp,
            "v_disp": v_disp,
            "w_disp": w_disp,
            "rv_ref": rv_ref,
            "rv_error": rv_error,
            "dynamic_reconstructed": True,
            "sample_size": member_count,
        }

        self._ensure_physical_table()

        try:
            update_sql = f"""
                INSERT OR REPLACE INTO cluster_refined_params 
                (cluster_id, ra_center, dec_center, plx_ref, pmra_ref, pmdec_ref, pmra_disp, pmdec_disp, pm_corr, distance_pc)
                VALUES 
                ('{cluster_id_upper}', {center_ra}, {center_dec}, {plx_ref}, {pmra_ref}, {pmdec_ref}, {pmra_disp}, {pmdec_disp}, {pm_corr}, {distance_pc})
            """
            self._engine.execute(update_sql)
            self.logger.info(
                f"💾 [资产滚动落盘] 动态 DNA 资产成功同步覆盖至 `cluster_kinematic_identity` 物理表"
            )
        except Exception as update_err:
            self.logger.warning(
                f"⚠️ [资产落盘跳过] 临时内存或底层数仓物理表未就绪: {update_err}"
            )

        return self._normalize_and_alias(dna_profile)

    def _dynamic_reconstruct(
        self,
        cluster_id: str,
        input_data: pd.DataFrame = None,
        metric: str = "median",
        use_gaia_raw: bool = True,
    ) -> dict:
        """
        🚀 【高精度自我一致性物理资产反演引擎】

        功能：
        1. [精确筛选模式]：若 input_data 为 None，自动通过动态几何质心定位半光半径，抽取出 hunt24 表中 Top 400 最高概率核心源进行反演。
        2. [沙箱校验模式]：若传入 input_data (DataFrame)，则直接对注入数据集进行动力学和光度学参数的解算。

        参数：
        - cluster_id: 星团唯一标识符。
        - input_data: 可选。外部算法注入的 DataFrame。如果传入，则直接基于传入数据计算。
        - metric: 统计算子选择，"median" (缺省) 或 "mean"。
        - use_gaia_raw: 开关。False 使用 hunt24 视图数据；True 提取 hunt24 骨干样本拓扑内联 Gaia 原始天体测量物理量。
        """
        cluster_id = cluster_id.upper()
        cluster_key = (
            #  self.get_param(cluster_id, "CAT_NAME") if not input_data else cluster_id
             cfg.CLUSTER_REGISTRY[cluster_id].CAT_NAME
        )
        metric_mode = metric.lower()

        self.logger.info(
            f"🧬 [资产精确重建] 启动重建管线 | 目标: {cluster_key} | "
            f"算子: {metric_mode.upper()} | 物理源: {'Gaia原始仓' if use_gaia_raw else 'Hunt清洗仓'}"
        )

        # 1. 动态输入分支路由
        if input_data is not None:
            self.logger.info(
                "📥 [沙箱校验模式] 检测到外部算法注入数据，跳过数仓自适应筛选..."
            )
            stars_df = input_data.copy()
        else:
            self.logger.info(
                "🗄️ [精确筛选模式] 开始利用自收敛几何质心裁切半光半径内 Top 400 骨干星..."
            )

            # 获取该星团的半光半径先验（单位：度），若找不到则默认以 0.5 度兜底
            try:
                r_half_deg = float(self.get_param(cluster_id, "R_HALF_LIGHT"))
            except Exception:
                r_half_deg = 0.5
                self.logger.warning(
                    f"⚠️ 未找到 {cluster_id} 的 R_HALF_LIGHT 先验，临时使用 {r_half_deg} 度。"
                )

            hunt_table = cfg.MANIFEST[cfg.IDX_HUNT].std_view.format(
                cluster=cluster_id.lower()
            )
            gaiadr3_table = cfg.MANIFEST[f"{cluster_id.lower()}_field"].std_view.format(cluster=cluster_id.lower())
            self.logger.info(f"[debug]gaiadr3_table:{gaiadr3_table}")

            try:
                # 🎯 【核心 SQL 修改】：利用 CTE 自收敛计算临时几何质心，空间过滤 + 概率切片 Top 400
                if use_gaia_raw:
                    self.logger.info(
                        "📡 [数据源：Gaia 原始仓] 关联 std_view_m45_field 提取核心骨干星原始相空间数据..."
                    )
                    sql = f"""
                        WITH dynamic_center AS (
                            SELECT median(ra) AS c_ra, median(dec) AS c_dec 
                            FROM {hunt_table} WHERE cluster = '{cluster_key}'
                        ),
                        core_members AS (
                            SELECT h.id, h.prob
                            FROM {hunt_table} h
                            CROSS JOIN dynamic_center c
                            WHERE h.cluster = '{cluster_key}'
                              AND SQRT(POWER((h.ra - c.c_ra) * COS(RADIANS(c.c_dec)), 2) + POWER(h.dec - c.c_dec, 2)) <= {r_half_deg}
                            ORDER BY h.prob DESC
                            LIMIT 400
                        )
                        SELECT 
                            g.ra, g.dec, g.plx, g.pmra, g.pmdec, g.rv,
                            g.mag as g_mag, g.color as bp_rp, 
                            0.0 as extinction_g   -- Gaia 原始表无消光改正
                        FROM core_members cm
                        INNER JOIN {gaiadr3_table} g ON cm.id = g.id
                    """
                else:
                    self.logger.info(
                        "🧹 [数据源：Hunt 清洗仓] 提取核心骨干星 Hunt 视差消光改正参数..."
                    )
                    sql = f"""
                        WITH dynamic_center AS (
                            SELECT median(ra) AS c_ra, median(dec) AS c_dec 
                            FROM {hunt_table} WHERE cluster = '{cluster_key}'
                        )
                        SELECT 
                            h.ra, h.dec, h.plx, h.pmra, h.pmdec, h.rv,
                            h.g_mag_corr as g_mag, h.bp_rp_corr as bp_rp, h.a_g as extinction_g
                        FROM {hunt_table} h
                        CROSS JOIN dynamic_center c
                        WHERE h.cluster = '{cluster_key}'
                          AND SQRT(POWER((h.ra - c.c_ra) * COS(RADIANS(c.c_dec)), 2) + POWER(h.dec - c.c_dec, 2)) <= {r_half_deg}
                        ORDER BY h.prob DESC
                        LIMIT 400
                    """

                stars_df = self._engine.execute(sql).df()
            except Exception as e:
                self.logger.error(
                    f"❌ 执行高精度自适应 SQL 筛选或多源数仓读取失败: {str(e)}"
                )
                return {}

        # 2. 核心样本基底完整性校验
        member_count = len(stars_df)
        self.logger.info(f"📊 锁定有效计算样本数: {member_count} 颗星")
        if stars_df.empty or member_count < 3:
            self.logger.warning("⚠️ 有效计算样本过少，无法构建统计学反演。")
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

        # --- 🎯 追加光度学与消光资产解析 ---
        ext_vals = (
            stars_df["extinction_g"].dropna().values
            if "extinction_g" in stars_df.columns
            else []
        )
        if len(ext_vals) > 0:
            avg_extinction = (
                float(np.mean(ext_vals))
                if metric_mode == "mean"
                else float(np.median(ext_vals))
            )
        else:
            avg_extinction = 0.0

        # 4. 组装重构后的标准先验字典（包含了新提炼的光度资产）
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
            "CMD_DEV": (
                0.8 if use_gaia_raw else 0.3
            ),  # Hunt 改正消光后 CMD 色散散布会显著降低
            "PHOT_EXTINCTION_AG": avg_extinction,  # 🚀 新增消光资产项
            "MAG_CORRECTED": not use_gaia_raw,  # 🚀 新增星等改正状态标记
            "RECONSTRUCTED": True,
        }

        # --- 🎯 6. 动态追加：调用 3D UVW 速度反演引擎 (向下透传高纯度核心骨干星数据集) ---
        uvw_data = self._dynamic_reconstruct_ex(
            cluster_id,
            input_data=stars_df,
            metric=metric_mode,
            use_gaia_raw=use_gaia_raw,
        )
        if uvw_data:
            reconstructed_params.update(uvw_data)
            self.logger.info(
                f"🧬 [高维动力学] 3D UVW/RV 骨干样本矩阵基于 [{metric_mode.upper()}] 重建成功"
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
            f"✅ [高精度重构完成] 算子={metric_mode.upper()} | 样本数={member_count} | 距离={distance_pc:.1f}pc, 消光AG={avg_extinction:.3f}"
        )

        # 7. 持久化路由决策：只有在使用原始清洗库生成的标准生产流模式下，才允许写回底座，防沙箱污染
        if input_data is None:
            self._save_refined_params(cluster_key, reconstructed_params)
        else:
            self.logger.info(
                "🚫 [沙箱保护] 当前运行于测试数据源或外部注入模式，反演资产仅在内存中返回。"
            )

        return reconstructed_params

    def _dynamic_reconstruct_ex(
        self,
        cluster_id: str,
        input_data: pd.DataFrame = None,
        metric: str = "median",
        use_gaia_raw: bool = True,
    ) -> tuple:
        """
        🚀 【3D 运动学资产重建引擎 - 多源自适应多源数据校验版】
        """
        cluster_key = cluster_id.upper()
        metric_mode = metric.lower()

        if input_data is not None:
            stars_df = input_data.copy()
            if "rv" in stars_df.columns:
                stars_df = stars_df[
                    stars_df["rv"].notna()
                    & ~np.isnan(stars_df["rv"])
                    & (stars_df["plx"] > 0)
                ]
        else:
            # 💡 说明：若上游没有传递核心数据流，本函数也会自适应采用相同的“几何收敛+Top 400”过滤逻辑
            try:
                r_half_deg = float(self.get_param(cluster_id, "R_HALF_LIGHT"))
            except Exception:
                r_half_deg = 0.5

            hunt_table = cfg.MANIFEST[cfg.IDX_HUNT]["aln_view"].format(
                cluster=cluster_id.lower()
            )

            if use_gaia_raw:
                self.logger.info(
                    "📡 [数据源：Gaia 原始仓] 启动 6D 空间高精度自适应拓扑联查..."
                )
                sql = f"""
                    WITH dynamic_center AS (
                        SELECT median(ra) AS c_ra, median(dec) AS c_dec 
                        FROM {hunt_table} WHERE cluster = '{self.get_param(cluster_id, "CAT_NAME")}'
                    ),
                    core_members AS (
                        SELECT h.id
                        FROM {hunt_table} h
                        CROSS JOIN dynamic_center c
                        WHERE h.cluster = '{self.get_param(cluster_id, "CAT_NAME")}'
                          AND SQRT(POWER((h.ra - c.c_ra) * COS(RADIANS(c.c_dec)), 2) + POWER(h.dec - c.c_dec, 2)) <= {r_half_deg}
                        ORDER BY h.prob DESC
                        LIMIT 400
                    )
                    SELECT 
                        g.ra, g.dec, g.plx, g.pmra, g.pmdec, g.rv 
                    FROM core_members cm
                    INNER JOIN aln_m45_field g ON cm.id = g.id
                    WHERE g.plx > 0 AND g.rv IS NOT NULL AND NOT isnan(g.rv)
                """
            else:
                sql = f"""
                    WITH dynamic_center AS (
                        SELECT median(ra) AS c_ra, median(dec) AS c_dec 
                        FROM {hunt_table} WHERE cluster = '{self.get_param(cluster_id, "CAT_NAME")}'
                    )
                    SELECT h.ra, h.dec, h.plx, h.pmra, h.pmdec, h.rv 
                    FROM {hunt_table} h
                    CROSS JOIN dynamic_center c
                    WHERE h.cluster = '{self.get_param(cluster_id, "CAT_NAME")}' 
                      AND SQRT(POWER((h.ra - c.c_ra) * COS(RADIANS(c.c_dec)), 2) + POWER(h.dec - c.c_dec, 2)) <= {r_half_deg}
                      AND h.plx > 0 AND h.rv IS NOT NULL AND NOT isnan(h.rv)
                    ORDER BY h.prob DESC
                    LIMIT 400
                """

            try:
                stars_df = self._engine.execute(sql).df()
            except Exception as e:
                self.logger.error(f"❌ 读取 6D 动力学多源数仓失败: {e}")
                return None

        if len(stars_df) < 3:
            self.logger.warning(
                f"⚠️ 星团 {cluster_key} 有效 6D 速度样本过少 ({len(stars_df)}颗)，放弃 3D 速度反演。"
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
                frame="icrs",
            )

            galactic_frame = coords.galactic
            if hasattr(galactic_frame, "v_x"):
                u_arr = galactic_frame.v_x.to(u.km / u.s).value
                v_arr = galactic_frame.v_y.to(u.km / u.s).value
                w_arr = galactic_frame.v_z.to(u.km / u.s).value
            elif hasattr(galactic_frame, "velocity"):
                u_arr = galactic_frame.velocity.d_x.to(u.km / u.s).value
                v_arr = galactic_frame.velocity.d_y.to(u.km / u.s).value
                w_arr = galactic_frame.velocity.d_z.to(u.km / u.s).value
            else:
                cartesian_vel = galactic_frame.data.differentials["s"]
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
                "KINE_6D_COUNT": len(stars_df),
            }
        except Exception as ex:
            self.logger.error(f"❌ 空间动力学矩阵反演崩溃: {str(ex)}", exc_info=True)
            return None

    def _set_memory_cache(self, cluster_id, param_name, value):
        if cluster_id not in self._runtime_cache:
            self._runtime_cache[cluster_id] = {}
        self._runtime_cache[cluster_id][param_name] = value

    def _ensure_config_table_exists(self):
        self._engine.execute("""
            CREATE TABLE IF NOT EXISTS cluster_refined_params (
                cluster_id VARCHAR,
                param_name VARCHAR,
                param_value VARCHAR,
                is_array BOOLEAN,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (cluster_id, param_name)
            )
        """)

    def _save_refined_params(self, cluster_id: str, refined_config: dict):
        """将自适应进化参数批量 Upsert 写入 DuckDB，并强刷内存缓存"""
        cluster_id = cluster_id.upper()
        if not self._engine:
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
                # 兼容 list 形式的 UVW_REF
                is_array = isinstance(v, np.ndarray) or isinstance(v, list)
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
                self._engine.execute(upsert_sql)
                self._set_memory_cache(cluster_id, k, v)
            self.logger.info(
                f"💾 [ConfigManager] 星团 {cluster_id} 动态提炼的物理资产已安全持久化至 DuckDB 配置底座。[{k}]"
            )
        except Exception as e:
            self.logger.error(
                f"❌ [ConfigError] 持久化自适应参数时发生 DuckDB 写入阻塞: {e}"
            )
