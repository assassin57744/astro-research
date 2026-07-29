"""
plotter.py - AstroPlotter: 纯 Python ggplot 风格天文学可视化渲染引擎
专注于空间掩模图、三通道概率分布图及交叉比对诊断图表的高质量静默渲染。
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from pathlib import Path

import config as cfg

# 静默渲染后端
matplotlib.use("Agg")


class AstroPlotter:
    """天文学可视化绘制器（统一治理 Python 原生绘图与 ggplot 视觉风格）"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("AstroPipeline.Plotter")
        # 全局应用 ggplot 主题美学
        plt.style.use("ggplot")

    # ── 1. 空间管掩模图 (Spatial Tube Mask) ──

    def render_tube_mask(self, df_all: pd.DataFrame, df_seeds: pd.DataFrame, 
                         df_tube: pd.DataFrame, pca_proxy, cluster_id: str, 
                         length_deg: float, width_deg: float) -> Path:
        """重建空间管掩模图（统一使用赤道坐标 ra, dec，ggplot 经典配色与高对比度分层）。"""
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        center = getattr(pca_proxy, "cluster_center_", np.array([0.0, 0.0]))
        x0, y0 = float(center[0]), float(center[1])

        def _tangent_proj(df):
            if df.empty or "ra" not in df.columns or "dec" not in df.columns:
                return np.array([]), np.array([])
            dl = (df["ra"] - x0 + 180.0) % 360.0 - 180.0
            return dl * np.cos(np.radians(y0)), df["dec"] - y0

        x_all_p, y_all_p = _tangent_proj(df_all)
        x_tube_p, y_tube_p = _tangent_proj(df_tube)
        x_seeds_p, y_seeds_p = _tangent_proj(df_seeds)

        x_all, y_all = x0 + x_all_p, y0 + y_all_p
        x_tube, y_tube = x0 + x_tube_p, y0 + y_tube_p
        x_seeds, y_seeds = x0 + x_seeds_p, y0 + y_seeds_p

        # 图层 1: 背景场星密度网格 (Hexbin)
        if len(x_all) > 0:
            hb = ax.hexbin(
                x_all, y_all, gridsize=300, cmap="Greys",
                norm=LogNorm(vmin=1, vmax=max(10, len(df_all) // 5000)),
                alpha=0.6, zorder=1,
            )
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r"Stellar Density ($\log_{10} N$)", fontsize=12)

        # 图层 2: PCA 锚点 (Seeds) — 缩小尺寸并调高透明度，防止形成红块遮挡管内天体
        if len(x_seeds) > 0:
            ax.scatter(
                x_seeds, y_seeds, s=4.0, c="#e74c3c", alpha=0.5,
                edgecolors="none", label="Seeds (PCA Anchor)", zorder=2
            )

        # 图层 3: 管内天体 (Tube Filtered) — 使用高亮蓝并放置在上方层级
        if len(x_tube) > 0:
            ax.scatter(
                x_tube, y_tube, s=3.0, c="#00a8ff", alpha=0.7,
                label="Tube Filtered", zorder=3
            )

        # 图层 4: 潮汐管界限 Polygon — 使用白色底层 + 高亮橙色双重描边
        actual_width = getattr(pca_proxy, "tube_width_", width_deg)
        corners_pca = np.array([
            [-length_deg, -actual_width],
            [length_deg, -actual_width],
            [length_deg, actual_width],
            [-length_deg, actual_width],
        ])
        corners_proj = pca_proxy.inverse_transform(corners_pca) + np.array([x0, y0])

        # 4.1 白色底层描边（增加暗色/彩色背景下的对比度）
        ax.add_patch(Polygon(
            corners_proj, closed=True, fill=False,
            edgecolor="white", linewidth=4.0, linestyle="-",
            zorder=4
        ))
        # 4.2 顶层高亮鲜橙色虚线框
        ax.add_patch(Polygon(
            corners_proj, closed=True, fill=False,
            edgecolor="#f39c12", linewidth=2.5, linestyle="--",
            label=f"Tidal Tube (±{length_deg:.1f}°)", zorder=5
        ))

        # 视图边距与坐标轴翻转
        if len(x_all_p) > 0:
            view_margin = float(np.max(np.hypot(x_all_p, y_all_p))) * 1.03
        else:
            view_margin = max(length_deg, actual_width) * 1.2

        ax.set_xlim(x0 + view_margin, x0 - view_margin)
        ax.set_ylim(y0 - view_margin, y0 + view_margin)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_xaxis()  # 天文惯例：赤经 (RA) 增加方向向左 (东)

        ax.set_xlabel("RA (deg)", fontsize=12, labelpad=8)
        ax.set_ylabel("Dec (deg)", fontsize=12, labelpad=8)
        ax.set_title(
            f"{cluster_id.upper()} Spatial Masking: {length_deg * 2:.1f}° Tidal Tube Cut over Field Stars",
            fontsize=14, pad=12, fontweight="bold",
        )

        legend = ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
        for handle in legend.legend_handles:
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([30.0])

        plt.tight_layout()
        output_dir = Path(cfg.ANALYSIS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cluster_id.lower()}_spatial_tube.jpg"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ── 2. 三通道概率分布图 (Channel Probabilities Distribution) ──

    def render_channel_probs(self, df_res: pd.DataFrame, cluster_id: str) -> Path:
        """渲染 prob / core_prob / tail_prob 三通道概率分布直方图。"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

        titles = [
            ("prob", "Union Prob", "#3498db"),
            ("core_prob", "Core Channel Prob", "#e67e22"),
            ("tail_prob", "Tail Channel Prob", "#2ecc71"),
        ]

        for ax, (col, title, color) in zip(axes, titles):
            if col not in df_res.columns:
                ax.text(0.5, 0.5, f"Missing column: {col}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=13, fontweight="bold")
                continue

            data = df_res[col].dropna()
            ax.hist(data, bins=80, range=(0, 1), color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.axvline(x=0.2, color="#e74c3c", linestyle="--", linewidth=1.0, label="Member threshold (0.2)")
            ax.axvline(x=0.5, color="#c0392b", linestyle="--", linewidth=1.0, label="High-conf threshold (0.5)")
            ax.set_xlabel("Probability", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_yscale("log")
            ax.legend(fontsize=8, framealpha=0.8, loc="upper center")
            ax.text(
                0.95, 0.95,
                f"N={len(data):,}\nμ={data.mean():.3f}\nmed={data.median():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
            )

        fig.suptitle(
            f"{cluster_id.upper()} — Probability Distribution: Core vs Tail Channels",
            fontsize=15, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        output_dir = Path(cfg.ANALYSIS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cluster_id.lower()}_prob_dist.jpg"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ── 3. 交叉比对诊断三连图 (Cross-Match Diagnostics) ──

    def render_cross_match_diagnostics(self, df: pd.DataFrame, cluster_id: str, star_cluster=None) -> Path:
        """绘制交叉比对诊断三连图：CMD + 赤道位置 + 自行矢量图 (VPD)。"""
        x_tag = cfg.MASTER_COLS["X_MATCH"]

        colors = {
            "Matched":  ("#3498db", "Matched"),
            "PG Only":  ("#e74c3c", "PG Only"),
            "Ref Only": ("#2ecc71", "Ref Only"),
        }

        fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=150)
        fig.suptitle(
            f"{cluster_id.upper()} — Cross-Match Diagnostics",
            fontsize=15, fontweight="bold", y=1.02,
        )

        # 1. CMD (颜色-星等图)
        ax = axes[0]
        cmd_mask = (df["color"] > -0.5) & (df["color"] < 4.5) & (df["mag"] < 22)
        df_cmd = df[cmd_mask]
        for tag, (c, label) in colors.items():
            sub = df_cmd[df_cmd[x_tag] == tag]
            if not sub.empty:
                ax.scatter(sub["color"], sub["mag"], c=c, s=14, alpha=0.8,
                           edgecolors="none", label=f"{label} ({len(sub)})", zorder=3)
        if star_cluster is not None:
            self._overlay_isochrones(ax, star_cluster, df_cmd)
        ax.invert_yaxis()
        ax.set_xlabel("G_BP − G_RP", fontsize=12)
        ax.set_ylabel("G (mag)", fontsize=12)
        ax.set_title("CMD", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.8, loc="upper right")

        # 2. 赤道位置 (RA/Dec)
        ax = axes[1]
        ax.scatter(df["ra"], df["dec"], c="lightgrey", s=1, alpha=0.15, zorder=1)
        for tag, (c, label) in colors.items():
            sub = df[df[x_tag] == tag]
            if not sub.empty:
                ax.scatter(sub["ra"], sub["dec"], c=c, s=14, alpha=0.8,
                           edgecolors="none", label=f"{label} ({len(sub)})", zorder=3)
        ax.invert_xaxis()
        cos_dec = np.cos(np.radians(abs(df["dec"].mean())))
        ax.set_aspect(1.0 / cos_dec, adjustable="datalim")
        ax.set_xlabel("RA (deg)", fontsize=12)
        ax.set_ylabel("Dec (deg)", fontsize=12)
        ax.set_title("Spatial Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.8, loc="upper right")

        # 3. 自行矢量图 (VPD)
        ax = axes[2]
        for tag, (c, label) in colors.items():
            sub = df[df[x_tag] == tag]
            if not sub.empty:
                ax.scatter(sub["pmra"], sub["pmdec"], c=c, s=14, alpha=0.8,
                           edgecolors="none", label=f"{label} ({len(sub)})", zorder=3)
        ax.set_xlabel("pmra (mas/yr)", fontsize=12)
        ax.set_ylabel("pmdec (mas/yr)", fontsize=12)
        ax.set_title("Proper Motion VPD", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.8, loc="upper right")

        plt.tight_layout()
        output_dir = Path(cfg.ANALYSIS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cluster_id.lower()}_cross_match_diag.jpg"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ── 辅助方法：叠加等龄线 ──

    def _overlay_isochrones(self, ax, star_cluster, df_cmd=None):
        """在 CMD 图上叠加星团理论等龄线。"""
        iso_df = getattr(star_cluster, "isochrone_df", None)
        if iso_df is None or iso_df.empty:
            return

        dist_pc = getattr(star_cluster, "distance_pc", 100.0)
        ext_ag = getattr(star_cluster, "ext_ag", 0.0)
        ebprp = getattr(star_cluster, "e_bp_rp", ext_ag * cfg.REDDENING_RATIO_BP_RP)
        dist_mod = 5.0 * np.log10(dist_pc) - 5.0

        col_map = {}
        for col in iso_df.columns:
            c = col.lower()
            if c in ("gmag", "g"):
                col_map["G"] = col
            if c in ("g_bpmag", "bpmag", "bp"):
                col_map["BP"] = col
            if c in ("g_rpmag", "rpmag", "rp"):
                col_map["RP"] = col
        if len(col_map) < 3:
            return

        age_col = next((c for c in iso_df.columns if c.lower() in ("logage", "logageyr")), None)
        if age_col is None:
            return

        model_g = iso_df[col_map["G"]].values + dist_mod + ext_ag
        model_color = iso_df[col_map["BP"]].values - iso_df[col_map["RP"]].values + ebprp

        if df_cmd is not None and len(df_cmd) > 0:
            c_min, c_max = df_cmd["color"].min() - 0.2, df_cmd["color"].max() + 0.2
            m_min, m_max = df_cmd["mag"].min() - 1.0, df_cmd["mag"].max() + 1.0
        else:
            c_min, c_max = model_color.min(), model_color.max()
            m_min, m_max = model_g.min(), model_g.max()

        ages = sorted(iso_df[age_col].unique())
        step = max(1, len(ages) // 5)
        for log_age in ages[::step]:
            mask = iso_df[age_col] == log_age
            mc, mg = model_color[mask], model_g[mask]
            seg = (mc >= c_min) & (mc <= c_max) & (mg >= m_min) & (mg <= m_max)
            if seg.sum() < 5 or seg.sum() / max(len(mc), 1) < 0.05:
                continue
            ax.plot(mc[seg], mg[seg], "k-", linewidth=1.2, alpha=0.6, zorder=5)