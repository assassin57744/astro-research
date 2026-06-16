import logging
import time
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic


class AstroTransformer:
    def __init__(self, cluster_rv: float = None, cluster_center_icrs: tuple = None):
        """
        天体测量特征工程转换引擎 (遵循 Gaia DPAC 官方 RV 验证规范)。

        Args:
            cluster_rv (float, optional): 星团先验标准视向速度 (km/s)。用于缺失 RV 时的几何填充。
            cluster_center_icrs (tuple, optional): 星团天球中心锚点 (RA, Dec)，用于物理空间转换。
        """
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

        self.cluster_rv = cluster_rv
        if cluster_center_icrs is not None:
            self.cluster_center_ra = cluster_center_icrs[0]
            self.cluster_center_dec = cluster_center_icrs[1]
        else:
            self.cluster_center_ra = None
            self.cluster_center_dec = None

        self.logger.info(
            f"🚌 [AstroTransformer] 引擎初始化完成 | 模式: 动力学自适应 | "
            f"先验 RV: {self.cluster_rv} km/s | 锚点: ({self.cluster_center_ra}, {self.cluster_center_dec})"
        )

    def ingest_external_rv_data(
        self, df: pd.DataFrame, source_path: str, source_type: str = "hunt"
    ):
        """
        [预处理层] 自动化集成外部多源 RV 数据（如 Hunt 或 CU6 补充星表）。

        Args:
            df (pd.DataFrame): 原始恒星数据帧。
            source_path (str): 外部数据文件路径。
            source_type (str): 数据源标识（'hunt' | 'cu6'）。

        Returns:
            pd.DataFrame: 注入 RV 字段后的数据帧。
        """

        return df  # 占位：后续实现将直接返回注入了外部 RV 数据的 DataFrame

        # 读取外部数据
        # 使用 sep=None 和 engine='python' 自动探测 CSV 或 TSV 分隔符
        external_df = pd.read_csv(source_path, sep=None, engine="python")

        # 定义不同来源的列映射字典
        # 确保外部表至少包含 'source_id' 和对应的 RV 列
        mapping = {
            "hunt": {"col": "hunt_rv", "target": "rv_hunt"},
            "cu6": {"col": "rv_cu6", "target": "rv_cu6"},
        }

        if source_type not in mapping:
            raise ValueError(
                f"❌ 不支持的数据源类型: {source_type}，请选择 'hunt' 或 'cu6'"
            )

        target_col = mapping[source_type]["target"]
        source_col = mapping[source_type]["col"]

        # 确保 source_id 类型一致（防止 str 和 int 的 mismatch）
        df["source_id"] = df["source_id"].astype(str)
        external_df["source_id"] = external_df["source_id"].astype(str)

        # 执行左连接 (Left Join)
        df = df.merge(
            external_df[["source_id", source_col]], on="source_id", how="left"
        )

        # 重命名列以匹配 _compute_expected_rv 中的预定逻辑
        if source_col != target_col:
            df = df.rename(columns={source_col: target_col})

        self.logger.info(f"✅ 成功注入外部 RV 数据源: {source_type} (列: {target_col})")
        return df

    def _compute_expected_rv(self, df: pd.DataFrame) -> np.ndarray:
        """
        [私有方法] 极致动力学四级递进 RV 融合算子 (遵循 Katz et al. 2022 规范)。
        
        通过多级过滤与补全（原生观测 -> 官方补充 -> 文献先验 -> 几何投影），
        确保速度特征空间在物理上的完备性与纯净度。
        """
        assert (
            self.cluster_center_ra is not None and self.cluster_center_dec is not None
        ), "❌ 缺陷拦截: 启动物理直角坐标系时，必须传入有效的 cluster_center_icrs 物理锚点位置！"

        total_stars = len(df)

        # 1. 抽取第一级：原生 Gaia DR3 主表观测值
        rv_native = (
            df["rv"].to_numpy() if "rv" in df.columns else np.full(total_stars, np.nan)
        )

        # 💡 依据官方验证论文 (Katz et al. 2022) 的质量审计：
        # 如果原生观测误差过大，或者光谱有效观测周期数太少，说明存在严重的双星污染或互相关伪峰。
        if "rv_error" in df.columns or "radial_velocity_error" in df.columns:
            err_col = (
                "rv_error" if "rv_error" in df.columns else "radial_velocity_error"
            )
            rv_err = df[err_col].to_numpy()
            # 过滤掉测量误差大于 10 km/s 的极端高噪点，强制降级到下一级
            bad_rv_mask = rv_err > 10.0
            bad_count = np.sum(bad_rv_mask & (~np.isnan(rv_native)))
            if bad_count > 0:
                rv_native[bad_rv_mask] = np.nan
                self.logger.debug(f"🛡️ [RV审计] 拦截 {bad_count} 颗高误差(>10km/s)原生源，已标记为待补全。")

        # 2. 抽取第二级：Gaia 官方主管团队 (CU6/Blomme et al.) 针对热星/暗端发布的扩展补充表
        cu6_col = (
            "rv_cu6"
            if "rv_cu6" in df.columns
            else ("rv_hot" if "rv_hot" in df.columns else None)
        )
        rv_cu6 = df[cu6_col].to_numpy() if cu6_col else np.full(total_stars, np.nan)

        # 3. 抽取第三级：文献外部先验值（如 Emily L. Hunt 2024 等）
        hunt_col = (
            "rv_hunt"
            if "rv_hunt" in df.columns
            else ("hunt_rv" if "hunt_rv" in df.columns else None)
        )
        rv_hunt = df[hunt_col].to_numpy() if hunt_col else np.full(total_stars, np.nan)

        # ==============================================================================
        # 四级金字塔过滤渗透逻辑
        # ==============================================================================
        rv_final = np.full(total_stars, np.nan)

        # 【第一级】：高质主表原生观测覆盖
        is_native_valid = ~np.isnan(rv_native)
        rv_final[is_native_valid] = rv_native[is_native_valid]

        # 【第二级】：官方 CU6 补充/热星专属通道修正覆盖
        need_cu6_fill = np.isnan(rv_final) & (~np.isnan(rv_cu6))
        rv_final[need_cu6_fill] = rv_cu6[need_cu6_fill]

        # 【第三级】：文献先验 Hunt 建模填充
        need_hunt_fill = np.isnan(rv_final) & (~np.isnan(rv_hunt))
        rv_final[need_hunt_fill] = rv_hunt[need_hunt_fill]

        # 统计经过官方与文献深度挖掘后，最终仍旧缺失的数量
        is_still_nan = np.isnan(rv_final)
        nan_count = np.sum(is_still_nan)

        if nan_count == 0:
            self.logger.info(
                f"⚡ RV 全部闭环覆盖 | 总星数: {total_stars} [1.主表高质原生: {np.sum(is_native_valid)} | "
                f"2.CU6官方热星补充: {np.sum(need_cu6_fill)} | 3.Hunt先验: {np.sum(need_hunt_fill)}]"
            )
            return rv_final

        # 【第四级】：无可奈何情况下的几何投影反算保底 (v_r = V_cluster * cos(lambda))
        if self.cluster_rv is None:
            self.logger.warning("⚠️ [AstroTransformer] 未配置先验 RV_REF，缺失值将强制填充为 0.0。")
            rv_final[is_still_nan] = 0.0
        else:
            ra_rad = np.radians(df["ra"].to_numpy())
            dec_rad = np.radians(df["dec"].to_numpy())
            c_ra_rad = np.radians(self.cluster_center_ra)
            c_dec_rad = np.radians(self.cluster_center_dec)

            cos_lambda = np.sin(dec_rad) * np.sin(c_dec_rad) + np.cos(dec_rad) * np.cos(
                c_dec_rad
            ) * np.cos(ra_rad - c_ra_rad)

            rv_expected = self.cluster_rv * cos_lambda
            rv_final[is_still_nan] = rv_expected[is_still_nan]

        # ==============================================================================
        # 动力学多源联合审计日志
        # ==============================================================================
        self.logger.info(
            f"🔮 [RV补全报告] 原生: {np.sum(is_native_valid)} | CU6补充: {np.sum(need_cu6_fill)} | "
            f"Hunt先验: {np.sum(need_hunt_fill)} | 几何投影保底: {nan_count} ({ (nan_count/total_stars)*100 :.1f}%)"
        )
        return rv_final

    def fit_transform(self, df: pd.DataFrame, mode: str) -> np.ndarray:
        """
        核心物理转换网关。
        
        接收原生观测数据，根据指定的维度模式（3D 到 6D），执行相空间转换与特征工程，
        产出标准化的 N x D 特征矩阵。

        Args:
            df (pd.DataFrame): 输入观测数据。
            mode (str): 转换模式（'3d', '5d', '6d_o', '5d_h', '3d_v', '6d_p'）。

        Returns:
            np.ndarray: 纯净特征矩阵。
        """
        start_time = time.time()
        total_stars = len(df)
        mode = mode.lower()

        self.logger.info(f"🚀 [AstroTransformer] 启动特征转换 | 模式: {mode.upper()} | 样本: {total_stars}")

        # --------------------------------==================--------------------------------
        # 阶段一：字段归一化与预检
        # --------------------------------==================--------------------------------
        # 注意：在 Gaia 和本系统中，pmra 统一指代 pmra_cosdec (μ*α)，即已包含 cos(dec) 修正。
        # 对于大尺度星团，建议使用 6d_p 或 3d_v 模式以消除球面投影畸变。
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        if "radial_velocity" in df.columns and "rv" not in df.columns:
            df = df.rename(columns={"radial_velocity": "rv"})
        elif "rv" not in df.columns:
            df["rv"] = np.nan

        if "parallax" in df.columns and "plx" not in df.columns:
            df = df.rename(columns={"parallax": "plx"})
        elif "plx" not in df.columns:
            df["plx"] = np.nan

        # 防御性补全：如果由于上游视图缺失导致 df 为空且无列，强制补全核心字段以防 KeyError
        if "pmra" not in df.columns:
            df["pmra"] = np.nan
        if "pmdec" not in df.columns:
            df["pmdec"] = np.nan

        # 补全自行(Proper Motion)字段的常见别名映射 (兼容 pmra_cosdec, pm_ra, pmde, pm_dec 等)
        pm_map = {"pmra_cosdec": "pmra", "pm_ra": "pmra", "pmde": "pmdec", "pm_dec": "pmdec"}
        for old_col, new_col in pm_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # --------------------------------==================--------------------------------
        # 阶段二：按 6 大特征相空间模式分流处理
        # --------------------------------==================--------------------------------
        if mode == "3d":
            res = df[["pmra", "pmdec", "plx"]].to_numpy()

        elif mode == "2d":
            res = df[["pmra", "pmdec"]].to_numpy()

        elif mode == "5d":
            res = df[["ra", "dec", "pmra", "pmdec", "plx"]].to_numpy()

        elif mode == "6d_o":
            rv_processed = self._compute_expected_rv(df)
            res = np.column_stack(
                (
                    df[["ra", "dec", "pmra", "pmdec", "plx"]].to_numpy(),
                    rv_processed,
                )
            )

        elif mode == "5d_h":
            sc = SkyCoord(
                ra=df["ra"].to_numpy() * u.deg,
                dec=df["dec"].to_numpy() * u.deg,
                pm_ra_cosdec=df["pmra"].to_numpy() * u.mas / u.yr,
                pm_dec=df["pmdec"].to_numpy() * u.mas / u.yr,
                frame="icrs",
            )
            galactic_frame = sc.galactic

            l = galactic_frame.l.degree
            b = galactic_frame.b.degree
            pm_l_cosb = galactic_frame.pm_l_cosb.value
            pm_b = galactic_frame.pm_b.value
            plx = df["plx"].to_numpy()

            res = np.column_stack((l, b, pm_l_cosb, pm_b, plx))

        elif mode in ["3d_v", "6d_p"]:
            rv_processed = self._compute_expected_rv(df)

            bad_plx_mask = df["plx"].to_numpy() <= 0.1
            bad_plx_count = np.sum(bad_plx_mask)
            if bad_plx_count > 0:
                self.logger.warning(f"🛡️ [视差拦截] {bad_plx_count} 颗星 plx <= 0.1 mas，已强制截断以防计算发散。")

            plx_safe = np.where(bad_plx_mask, 0.1, df["plx"].to_numpy())
            distance_pc = 1000.0 / plx_safe

            self.logger.debug(f"✨ 视差安全映射完成 | 距离跨度: {distance_pc.min():.1f} - {distance_pc.max():.1f} pc")

            sc = SkyCoord(
                ra=df["ra"].to_numpy() * u.deg,
                dec=df["dec"].to_numpy() * u.deg,
                distance=distance_pc * u.pc,
                pm_ra_cosdec=df["pmra"].to_numpy() * u.mas / u.yr,
                pm_dec=df["pmdec"].to_numpy() * u.mas / u.yr,
                radial_velocity=rv_processed * u.km / u.s,
                frame="icrs",
            )

            self.logger.debug(
                f"✨ 天球坐标构建完成 | RA range: [{df['ra'].min():.2f}, {df['ra'].max():.2f}] deg, "
                f"Dec range: [{df['dec'].min():.2f}, {df['dec'].max():.2f}] deg"
            )

            gal_cart = sc.transform_to(Galactic())
            self.logger.debug("✨ [AstroTransformer] 成功完成 ICRS -> Galactic 坐标系变换。")

            u_vel = gal_cart.velocity.d_x.value
            v_vel = gal_cart.velocity.d_y.value
            w_vel = gal_cart.velocity.d_z.value

            self.logger.debug(
                f"✨ 物理空间转换完成 | 速度空间统计: "
                f"U [{u_vel.min():.2f}, {u_vel.max():.2f}] km/s, "
                f"V [{v_vel.min():.2f}, {v_vel.max():.2f}] km/s, "
                f"W [{w_vel.min():.2f}, {w_vel.max():.2f}] km/s"
            )

            if mode == "3d_v":
                res = np.column_stack((u_vel, v_vel, w_vel))
            else:
                x_pos = gal_cart.cartesian.x.value
                y_pos = gal_cart.cartesian.y.value
                z_pos = gal_cart.cartesian.z.value
                res = np.column_stack((x_pos, y_pos, z_pos, u_vel, v_vel, w_vel))

            self.logger.debug(
                f"✨ 最终特征矩阵构建完成 | 模式: '{mode}' | 矩阵形状: {res.shape}"
            )

        else:
            self.logger.error(f"❌ 特征工程无法识别未知的 dim_mode: '{mode}'")
            raise ValueError(f"❌ 未知的特征维度模式 (dim_mode): '{mode}'。")

        elapsed_time = time.time() - start_time
        self.logger.info(f"✨ 特征矩阵构建成功 | Shape: {res.shape} | 耗时: {elapsed_time:.2f}s")
        return res
