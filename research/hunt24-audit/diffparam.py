import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 参数
# ============================================================

csv_files = [
    r"D:\git\astro-research\research\hunt24-audit\analysis\M44\different_parameter\seed_10.csv",
    r"D:\git\astro-research\research\hunt24-audit\analysis\M44\different_parameter\seed_20.csv",
    r"D:\git\astro-research\research\hunt24-audit\analysis\M44\different_parameter\seed_40.csv",
    r"D:\git\astro-research\research\hunt24-audit\analysis\M44\different_parameter\seed_80.csv",
]

# 根据你的实际列名修改
ID_COL = "id"
RA_COL = "ra"
DEC_COL = "dec"

# ============================================================
# 2. 读取四个 CSV
# ============================================================

dfs = []

for i, file in enumerate(csv_files, start=1):

    df = pd.read_csv(
        file,
        dtype={ID_COL: "string"}
    )

    # 检查必要字段
    required_cols = [ID_COL, RA_COL, DEC_COL]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{file} 缺少字段: {missing}\n"
            f"实际字段: {df.columns.tolist()}"
        )

    # 数值化 RA / Dec
    df[RA_COL] = pd.to_numeric(
        df[RA_COL],
        errors="coerce"
    )

    df[DEC_COL] = pd.to_numeric(
        df[DEC_COL],
        errors="coerce"
    )

    # 去掉没有 ID 或坐标的记录
    df = df.dropna(
        subset=[ID_COL, RA_COL, DEC_COL]
    ).copy()

    # 记录来源
    df["source_file"] = f"CSV{i}"
    df["source_filename"] = file

    dfs.append(df)

    print(
        f"CSV{i}: {file} -> {len(df):,} stars"
    )


# ============================================================
# 3. 合并四张表
# ============================================================

all_df = pd.concat(
    dfs,
    ignore_index=True
)

print()
print(
    f"四个 CSV 合计记录数: {len(all_df):,}"
)


# ============================================================
# 4. 按 ID 判断重复
#
# 注意：
# 不是简单 count()，而是计算某个 ID 出现于几个不同 CSV。
# 这样即使同一 CSV 内部意外出现重复行，也不会误判为
# “跨星表重复”。
# ============================================================

id_file_count = (
    all_df
    .groupby(ID_COL)["source_file"]
    .nunique()
)

all_df["n_files"] = (
    all_df[ID_COL]
    .map(id_file_count)
)


# 至少出现在两个不同 CSV 中 -> repeated
all_df["is_repeated"] = (
    all_df["n_files"] >= 2
)


# ============================================================
# 5. 分成 Repeated / Unique
# ============================================================

repeated_records = all_df[
    all_df["is_repeated"]
].copy()

unique_df = all_df[
    ~all_df["is_repeated"]
].copy()


# ============================================================
# 6. 重复恒星只保留一行用于绘图
#
# RA / Dec 采用各 CSV 的平均值。
# 如果四张表都是 Gaia 数据，同一 source_id 的坐标通常
# 应该几乎完全一致。
# ============================================================

repeated_stars = (
    repeated_records
    .groupby(ID_COL, as_index=False)
    .agg({
        RA_COL: "mean",
        DEC_COL: "mean",
        "n_files": "first",
    })
)


# ============================================================
# 7. 统计
# ============================================================

n_unique = len(unique_df)
n_repeated = len(repeated_stars)

print()
print("========== 统计 ==========")

print(
    f"只出现于一个 CSV 的恒星: "
    f"{n_unique:,}"
)

print(
    f"出现在 >=2 个 CSV 的恒星: "
    f"{n_repeated:,}"
)


print()
print("重复恒星按出现文件数分类:")

for n in range(2, 5):

    count = (
        repeated_stars["n_files"] == n
    ).sum()

    print(
        f"出现在 {n} 个 CSV 中: "
        f"{count:,}"
    )


# ============================================================
# 8. 保存数据
# ============================================================

# 每颗重复恒星只保存一行
repeated_stars.to_csv(
    "repeated_stars.csv",
    index=False
)

# 只出现一次的恒星
unique_df.to_csv(
    "unique_stars.csv",
    index=False
)

# 如果还想知道某颗重复星具体出现在哪些 CSV，
# 把所有重复记录也保存一份
repeated_records.to_csv(
    "repeated_records_all.csv",
    index=False
)


# ============================================================
# 9. RA-Dec 图
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


# ------------------------------------------------------------
# Unique
# ------------------------------------------------------------

ax.scatter(
    unique_df[RA_COL],
    unique_df[DEC_COL],
    s=20,
    label=f"Unique ({len(unique_df):,})",
)


# ------------------------------------------------------------
# Repeated
# 每颗重复恒星只画一次
# ------------------------------------------------------------

ax.scatter(
    repeated_stars[RA_COL],
    repeated_stars[DEC_COL],
    s=10,
    label=f"Repeated ({len(repeated_stars):,})",
)


# ============================================================
# 10. 图形设置
# ============================================================

ax.set_xlabel(
    "Right Ascension (deg)",
    fontsize=12
)

ax.set_ylabel(
    "Declination (deg)",
    fontsize=12
)

ax.set_title(
    "RA–Dec Distribution: Repeated vs Unique Stars",
    fontsize=14
)

ax.grid(
    alpha=0.3
)

ax.legend(
    fontsize=10
)

# 天文学常用画法：
# RA 从左向右减小
ax.invert_xaxis()

plt.tight_layout()


# ============================================================
# 11. 保存图片
# ============================================================

plt.savefig(
    r"D:\git\astro-research\research\hunt24-audit\analysis\ra_dec_repeated_unique.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()