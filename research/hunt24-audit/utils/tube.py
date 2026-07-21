import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use("Agg")  # 保证静默执行
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm
import os


def plot_spatial_tube(
    df_all: pd.DataFrame,
    df_seeds: pd.DataFrame,
    df_tube: pd.DataFrame,
    pca,
    length_deg: float = 18.0,
    width_deg: float = 1.5,
    output_dir: str = "./",
    cluster_id: str = "M45",  # 👈 新增：动态传入星团名称
):
    """
    自适应绘制星团潮汐尾 PCA 空间轨道管掩模并静默导出。
    """
    print(f"🎨 正在静默渲染 [{cluster_id}] 物理对齐场星背景与轨道管掩模...")

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    # 1. 坐标投影 (l*cos(b), b)
    x_all = df_all["l"] * np.cos(np.radians(df_all["b"]))
    y_all = df_all["b"]

    x_tube = df_tube["l"] * np.cos(np.radians(df_tube["b"]))
    y_tube = df_tube["b"]

    x_seeds = df_seeds["l"] * np.cos(np.radians(df_seeds["b"]))
    y_seeds = df_seeds["b"]

    # 2. 场星密度背景
    hb = ax.hexbin(
        x_all,
        y_all,
        gridsize=300,
        cmap="Greys",
        norm=LogNorm(vmin=1, vmax=1000),
        alpha=0.6,
        zorder=1,
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Stellar Density (log$_{10}$ N)", fontsize=12)

    # 3. 候选星与种子星
    ax.scatter(
        x_tube,
        y_tube,
        s=0.5,
        c="dodgerblue",
        alpha=0.4,
        label="Tube Filtered",
        zorder=2,
    )
    ax.scatter(
        x_seeds,
        y_seeds,
        s=3,
        c="crimson",
        alpha=0.8,
        label="Seeds (PCA Anchor)",
        zorder=3,
    )

    # 4. 动态绘制 PCA 轨道管边界
    corners_pca = np.array(
        [
            [-length_deg, -width_deg],
            [length_deg, -width_deg],
            [length_deg, width_deg],
            [-length_deg, width_deg],
        ]
    )
    # corners_sky = pca.inverse_transform(corners_pca)
    # 反解 PCA 矩形顶点时加上质心偏移：
    corners_sky = pca.inverse_transform(corners_pca) + getattr(pca, 'cluster_center_', np.array([0, 0]))

    tube_patch = Polygon(
        corners_sky,
        closed=True,
        fill=False,
        edgecolor="darkorange",
        linewidth=2.5,
        linestyle="--",
        label=f"Tidal Tube (±{length_deg}°)",  # 👈 动态标注实际长度
        zorder=4,
    )
    ax.add_patch(tube_patch)

    # 5. 修饰
    ax.invert_xaxis()
    ax.set_aspect("equal")

    ax.set_xlim(corners_sky[:, 0].max() + 3, corners_sky[:, 0].min() - 3)
    ax.set_ylim(corners_sky[:, 1].min() - 3, corners_sky[:, 1].max() + 3)

    ax.set_xlabel(r"Galactic Longitude $l \cos(b)$ (deg)", fontsize=14)
    ax.set_ylabel(r"Galactic Latitude $b$ (deg)", fontsize=14)
    # 👈 动态渲染标题
    ax.set_title(
        f"{cluster_id.upper()} Spatial Masking: {length_deg*2:.1f}° Tidal Tube Cut over Field Stars",
        fontsize=16,
        pad=15,
    )

    legend = ax.legend(loc="upper right", frameon=True, fontsize=12)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([30.0])

    plt.tight_layout()

    # 6. 动态文件名输出 (例如 analysis/m67_spatial_tube.jpg)
    # output_dir = os.path.join(os.getcwd(), "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{cluster_id.lower()}_spatial_tube.jpg")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 [{cluster_id}] 成果图已保存至: {output_path}")
