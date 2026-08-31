import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 读取 CSV
# ============================================================

csv_file = r"D:\git\astro-research\research\hunt24-audit\analysis\M67\m67_full.csv"
df = pd.read_csv(csv_file)


# ============================================================
# 2. 自动寻找字段名
# ============================================================

def find_column(df, candidates):
    """从候选字段名中找到实际存在的列"""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"没有找到字段，候选字段为: {candidates}\n"
        f"当前 CSV 字段为:\n{df.columns.tolist()}"
    )


# G 星等
gmag_col = find_column(
    df,
    [
        "Gmag",
        "phot_g_mean_mag",
        "gmag",
        "g_mag",
        "mag",
    ]
)

# 视差误差
plx_err_col = find_column(
    df,
    [
        "e_Plx",
        "parallax_error",
        "plx_err",
        "plx_error",
    ]
)

# CMD chi2
cmd_chi2_col = find_column(
    df,
    [
        "cmd_chi2",
        "CMD_CHI2",
    ]
)


print(f"Gmag column       : {gmag_col}")
print(f"Parallax err column: {plx_err_col}")
print(f"CMD chi2 column    : {cmd_chi2_col}")


# ============================================================
# 3. 去掉绘图所需字段中的 NaN / inf
# ============================================================

plot_df = df[
    [gmag_col, plx_err_col, cmd_chi2_col]
].copy()

plot_df = plot_df.replace(
    [float("inf"), float("-inf")],
    pd.NA
).dropna()


# ============================================================
# 4. 筛选 cmd_chi2 < 0.5
# ============================================================

cmd_mask = plot_df[cmd_chi2_col] < 0.5
cmd_good = plot_df[cmd_mask]

print()
print(f"All stars          : {len(plot_df):,}")
print(f"cmd_chi2 < 0.5     : {len(cmd_good):,}")
print(
    f"Fraction           : "
    f"{len(cmd_good) / len(plot_df) * 100:.2f}%"
)


# ============================================================
# 5. 绘图
# ============================================================

plt.figure(figsize=(10, 7))

# 所有恒星
plt.scatter(
    plot_df[gmag_col],
    plot_df[plx_err_col],
    s=10,
    alpha=1,
    label=f"All stars ({len(plot_df):,})"
)

# cmd_chi2 < 0.5 的恒星
plt.scatter(
    cmd_good[gmag_col],
    cmd_good[plx_err_col],
    s=15,
    alpha=1,
    label=f"Prob < 0.5 ({len(cmd_good):,})"
)

plt.xlabel("G magnitude")
plt.ylabel("Parallax error (mas)")
plt.title("Parallax Error vs G Magnitude of M67")

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

# 保存图片
plt.savefig(
    r"D:\git\astro-research\research\hunt24-audit\analysis\parallax_error_vs_gmag.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()