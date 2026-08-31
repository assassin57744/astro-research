import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. 文件名：这里改成你的文件名
# =====================================================
CSV1 = r"D:\git\astro-research\research\hunt24-audit\analysis\mel22_left.csv"
CSV2 = r"D:\git\astro-research\research\hunt24-audit\analysis\mel22_right.csv"
CSV3 = r"D:\git\astro-research\research\hunt24-audit\analysis\mel22_own.csv"

# 坐标列名：这里改成你的 RA / Dec 列名
RA_COL = "ra"
DEC_COL = "dec"

# =====================================================
# 2. 匹配方式
#    推荐优先使用唯一ID
#    可选：
#       MATCH_MODE = "id"
#       MATCH_MODE = "radec"
# =====================================================
MATCH_MODE = "id"
ID_COL = "id"

# 如果没有唯一ID，只能按 ra/dec 匹配
RADEC_DECIMALS = 8

# 输出图文件（矢量图 PDF）
OUTPUT_PDF = r"D:\git\astro-research\research\hunt24-audit\analysis\ra_dec_membership_compare.pdf"


# =====================================================
# 工具函数
# =====================================================
def check_required_columns(df, df_name, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} 缺少列: {missing}\n"
            f"{df_name} 当前列名为: {list(df.columns)}"
        )


def build_match_key(df, mode="id", id_col="source_id", ra_col="ra", dec_col="dec", decimals=8):
    """
    生成用于判断“是否是同一个成员”的 key
    """
    if mode == "id":
        if id_col not in df.columns:
            raise ValueError(
                f"选择了 MATCH_MODE='id'，但列 {id_col} 不存在。\n"
                f"当前列名：{list(df.columns)}"
            )
        return df[id_col].astype(str)

    elif mode == "radec":
        if ra_col not in df.columns or dec_col not in df.columns:
            raise ValueError(
                f"选择了 MATCH_MODE='radec'，但缺少 {ra_col} 或 {dec_col}。"
            )
        ra_round = pd.to_numeric(df[ra_col], errors="coerce").round(decimals)
        dec_round = pd.to_numeric(df[dec_col], errors="coerce").round(decimals)
        return ra_round.astype(str) + "_" + dec_round.astype(str)

    else:
        raise ValueError("MATCH_MODE 只能是 'id' 或 'radec'")


# =====================================================
# 3. 读取 CSV
# =====================================================
df1 = pd.read_csv(CSV1)
df2 = pd.read_csv(CSV2)
df3 = pd.read_csv(CSV3)

for name, df in [("CSV1", df1), ("CSV2", df2), ("CSV3", df3)]:
    check_required_columns(df, name, [RA_COL, DEC_COL])


# =====================================================
# 4. 合并 csv1 和 csv2：只取并集，不做 merge
# =====================================================
df12_union = pd.concat([df1, df2], ignore_index=True)

df12_union["_match_key"] = build_match_key(
    df12_union,
    mode=MATCH_MODE,
    id_col=ID_COL,
    ra_col=RA_COL,
    dec_col=DEC_COL,
    decimals=RADEC_DECIMALS
)

# 去重：只保留第一条，不拼接其他数据
df12_union = df12_union.drop_duplicates(subset=["_match_key"], keep="first").reset_index(drop=True)

df3 = df3.copy()
df3["_match_key"] = build_match_key(
    df3,
    mode=MATCH_MODE,
    id_col=ID_COL,
    ra_col=RA_COL,
    dec_col=DEC_COL,
    decimals=RADEC_DECIMALS
)

# 第三个 CSV 自身也去重，避免重复点
df3 = df3.drop_duplicates(subset=["_match_key"], keep="first").reset_index(drop=True)


# =====================================================
# 5. 计算三种状态
# =====================================================
keys_union = set(df12_union["_match_key"])
keys_csv3 = set(df3["_match_key"])

# 共同成员
common_keys = keys_union & keys_csv3

# csv1 和 csv2 的独有成员
only_union_keys = keys_union - keys_csv3

# csv3 的独有成员
only_csv3_keys = keys_csv3 - keys_union

df_common = df12_union[df12_union["_match_key"].isin(common_keys)].copy()
df_only_union = df12_union[df12_union["_match_key"].isin(only_union_keys)].copy()
df_only_csv3 = df3[df3["_match_key"].isin(only_csv3_keys)].copy()


# =====================================================
# 6. 画 RA-Dec 图
#    三种状态：
#    1. 共同成员
#    2. CSV1+CSV2 独有成员
#    3. CSV3 独有成员
# =====================================================

fig, ax = plt.subplots(figsize=(8, 6))

# 1. CSV1+CSV2 独有成员
ax.scatter(
    df_only_union[RA_COL],
    df_only_union[DEC_COL],
    s=16,
    alpha=0.80,
    label=f"Only in Splitted Field ({len(df_only_union)})",
    zorder=2
)

# 2. CSV3 独有成员
ax.scatter(
    df_only_csv3[RA_COL],
    df_only_csv3[DEC_COL],
    s=16,
    alpha=0.80,
    label=f"Only in Constructed Pipeline ({len(df_only_csv3)})",
    zorder=3
)

# 3. 共同成员
# 放在最后画，使共同成员不会被另外两类点遮住
ax.scatter(
    df_common[RA_COL],
    df_common[DEC_COL],
    s=22,
    alpha=0.90,
    label=f"Matched ({len(df_common)})",
    zorder=4
)

# 坐标轴
ax.set_xlabel("RA (deg)", fontsize=12)
ax.set_ylabel("Dec (deg)", fontsize=12)

ax.set_title(
    "RA-Dec Membership Comparison",
    fontsize=13
)

# 图例中自动显示每类数量
ax.legend(
    loc="best",
    fontsize=10,
    frameon=True
)

ax.grid(
    True,
    linestyle=":",
    alpha=0.3
)

# 如果希望符合天文图常见习惯：
# RA 从左向右减小
# ax.invert_xaxis()

plt.tight_layout()

# 输出真正的矢量 PDF
fig.savefig(
    OUTPUT_PDF,
    format="pdf",
    bbox_inches="tight"
)

plt.close(fig)

print(f"共同成员: {len(df_common)}")
print(f"CSV1+CSV2 独有成员: {len(df_only_union)}")
print(f"CSV3 独有成员: {len(df_only_csv3)}")
print(f"PDF 已保存: {OUTPUT_PDF}")


# =====================================================
# 7. 打印统计信息
# =====================================================
print("处理完成。")
print(f"CSV1 行数: {len(df1)}")
print(f"CSV2 行数: {len(df2)}")
print(f"CSV3 行数: {len(df3)}")
print(f"CSV1+CSV2 并集行数: {len(df12_union)}")
print(f"共同成员数: {len(df_common)}")
print(f"CSV1+CSV2 独有成员数: {len(df_only_union)}")
print(f"CSV3 独有成员数: {len(df_only_csv3)}")
print(f"矢量图 PDF 已保存为: {OUTPUT_PDF}")