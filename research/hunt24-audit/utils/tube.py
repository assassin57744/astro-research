import os
import matplotlib

matplotlib.use("Agg")  # 保证静默执行
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


def plot_spatial_tube(
    df_all: pd.DataFrame,
    df_seeds: pd.DataFrame,
    df_tube: pd.DataFrame,
    pca,
    length_deg: float = 18.0,
    width_deg: float = 1.5,
    output_dir: str = "./",
    cluster_id: str = "M45",
):
    """自适应绘制星团潮汐尾 PCA 空间轨道管掩模并静默导出。

    已彻底解决：
    1. 正交切面投影反解与坐标量纲对齐；
    2. 基于包络半径的物理等距对称视场锁定 (完美保证 1:1 绝对正圆背景，无形变/剪裁)；
    3. Matplotlib adjustable='box' 视场警告与轴比例冲突消除。
    """
    print(
        f"🎨 正在静默渲染 [{cluster_id}] 物理对齐场星背景与轨道管掩模..."
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # 1. 提取质心参考原点 (x0, y0)
    center = getattr(pca, "cluster_center_", np.array([0.0, 0.0]))
    x0, y0 = float(center[0]), float(center[1])  # x0 = l0_deg, y0 = b0_deg

    # 2. 统一构建正交切面天球坐标 (X_proj = (l - l0)*cos(b0), Y_proj = b - b0)
    def compute_tangent_proj(df):
        if df.empty or "l" not in df.columns or "b" not in df.columns:
            return np.array([]), np.array([])
        # 考虑 0/360 跨越的相对经度差
        dl = (df["l"] - x0 + 180.0) % 360.0 - 180.0
        x_proj = dl * np.cos(np.radians(y0))
        y_proj = df["b"] - y0
        return x_proj, y_proj

    x_all_p, y_all_p = compute_tangent_proj(df_all)
    x_tube_p, y_tube_p = compute_tangent_proj(df_tube)
    x_seeds_p, y_seeds_p = compute_tangent_proj(df_seeds)

    # 换算回相对于质心的物理绝对角度，保证 X 轴代表真实物理角尺度 offset (deg)
    x_all, y_all = x0 + x_all_p, y0 + y_all_p
    x_tube, y_tube = x0 + x_tube_p, y0 + y_tube_p
    x_seeds, y_seeds = x0 + x_seeds_p, y0 + y_seeds_p

    # 3. 场星密度背景
    if len(x_all) > 0:
        hb = ax.hexbin(
            x_all,
            y_all,
            gridsize=300,
            cmap="Greys",
            norm=LogNorm(vmin=1, vmax=max(10, len(df_all) // 5000)),
            alpha=0.6,
            zorder=1,
        )
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"Stellar Density ($\log_{10} N$)", fontsize=12)

    # 4. 候选星与种子星
    if len(x_tube) > 0:
        ax.scatter(
            x_tube,
            y_tube,
            s=1.5,
            c="dodgerblue",
            alpha=0.5,
            label="Tube Filtered",
            zorder=2,
        )
    if len(x_seeds) > 0:
        ax.scatter(
            x_seeds,
            y_seeds,
            s=8,
            c="crimson",
            alpha=0.8,
            label="Seeds (PCA Anchor)",
            zorder=3,
        )

    # 5. 精确反解 PCA 轨道管 4 个顶点
    actual_width = getattr(pca, "tube_width_", width_deg)
    corners_pca = np.array(
        [
            [-length_deg, -actual_width],
            [length_deg, -actual_width],
            [length_deg, actual_width],
            [-length_deg, actual_width],
        ]
    )

    # 通过 PCA 逆变换得到正交切面相对投影坐标 (X_proj, Y_proj)
    corners_proj = pca.inverse_transform(corners_pca)
    # 加回物理质心 (x0, y0)，还原为天球投影绝对度数
    corners_sky = corners_proj + np.array([x0, y0])

    tube_patch = Polygon(
        corners_sky,
        closed=True,
        fill=False,
        edgecolor="darkorange",
        linewidth=2.5,
        linestyle="--",
        label=f"Tidal Tube (±{length_deg:.1f}°)",
        zorder=4,
    )
    ax.add_patch(tube_patch)

    # 6. 基于全量场星数据 (df_all) 的真实物理半径自动锁定 1:1 正圆形视场
    # 计算所有场星相对于质心 (x0, y0) 的最大切面投影半径
    if len(x_all_p) > 0:
        # 天球切面物理半径 r = sqrt(X_proj^2 + Y_proj^2)
        field_radius = float(np.max(np.hypot(x_all_p, y_all_p)))
        # 留出 3% 的微小边距，防止 Hexbin 边缘贴死坐标轴
        view_margin = field_radius * 1.03
    else:
        view_margin = max(length_deg, actual_width) * 1.2

    # 以质心 (x0, y0) 为中心，向四周对称扩展 view_margin，构成完全正方形的数据边界
    # 在 1:1 (equal aspect) 下渲染，内部的 Hexbin 场星必然呈现为无剪裁的绝对正圆形！
    ax.set_xlim(x0 + view_margin, x0 - view_margin)
    ax.set_ylim(y0 - view_margin, y0 + view_margin)

    # 锁定 1:1 绝对物理数据宽高比 (box 模式)
    ax.set_aspect("equal", adjustable="box")

    # 修饰与坐标轴标注
    ax.set_xlabel(r"Galactic Longitude $l \cos(b)$ (deg)", fontsize=14)
    ax.set_ylabel(r"Galactic Latitude $b$ (deg)", fontsize=14)
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

    # 7. 静默保存导出
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{cluster_id.lower()}_spatial_tube.jpg"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 [{cluster_id}] 成果图已保存至: {output_path}")


def plot_prob_distributions(
    df_res: pd.DataFrame,
    cluster_id: str = "M45",
    output_dir: str = "./",
):
    """绘制 prob / core_prob / tail_prob 三个概率分布直方图。"""
    print(
        f"📊 正在渲染 [{cluster_id}] 三通道概率分布直方图..."
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    titles = [
        ("prob", "Union Prob", "#4C72B0"),
        ("core_prob", "Core Channel Prob", "#DD8452"),
        ("tail_prob", "Tail Channel Prob", "#55A868"),
    ]

    for ax, (col, title, color) in zip(axes, titles):
        if col not in df_res.columns:
            ax.text(
                0.5,
                0.5,
                f"Missing: {col}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=13)
            continue

        data = df_res[col].dropna()
        ax.hist(
            data,
            bins=80,
            range=(0, 1),
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(
            x=0.2,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="Member threshold (0.2)",
        )
        ax.axvline(
            x=0.5,
            color="darkred",
            linestyle="--",
            linewidth=1.0,
            label="High-conf threshold (0.5)",
        )
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=8, framealpha=0.7)
        ax.text(
            0.95,
            0.95,
            f"N={len(data):,}\nμ={data.mean():.3f}\nmed={data.median():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8
            ),
        )

    fig.suptitle(
        f"{cluster_id.upper()} — Probability Distribution: Core vs Tail Channels",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{cluster_id.lower()}_prob_dist.jpg"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 [{cluster_id}] 概率分布直方图已保存至: {output_path}")