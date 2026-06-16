import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.patches import Circle  # 核心引入
from astroquery.simbad import Simbad

import config as cfg  # config.py 位于 modules/ 的上一级目录
from config import (  # config.py 位于 modules/ 的上一级目录
    MANIFEST as DATA,
    STD_COLS,
    TMPL,
)


class AstroAnalyzer:
    def __init__(self, db_instance, target_cluster=None, target_category=None, mode="3d"):
        self.db = db_instance
        self.target_cluster = target_cluster
        self.target_category = target_category
        self.mode = mode
        self.logger = logging.getLogger(f"AstroPipeline.{__name__}")

        # 定义标准特征组，便于自动增强
        self.FEATURE_GROUPS = {
            "proper_motion": ["pmra", "pmdec"],
            "photometry": ["mag", "color"],
            "astrometry": ["plx", "ruwe"],
        }

    def _save_plot(self, fig, prefix, key_ref=None):
        """[内部工具] 统一保存图表到指定目录并关闭画布。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cluster_id = self.target_cluster
        if cluster_id not in cfg.CLUSTERS:
            self.logger.error(f"❌ 未知的星团 ID: {cluster_id}")
            return
        
        # 优化文件名：增加 mode 和 category 标识，便于溯源
        category = key_ref or self.target_category or "none"
        filename = TMPL.FILE_PLOT.format(
            cluster=cluster_id,
            category=category,
            mode=self.mode,
            prefix=prefix,
            timestamp=timestamp,
        )

        # 扁平化处理：直接使用导出根目录，不再创建 plots 子目录
        plot_dir = cfg.EXPORT_DIR
        plot_dir.mkdir(parents=True, exist_ok=True)

        save_path = plot_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"📊 图表已保存至: {save_path}")

    def _get_bg_data(self, t_background, fields):
        """[内部工具] 获取背景参考数据。"""
        field_str = ", ".join(fields)
        self.logger.debug(f"正在提取背景场样本: {t_background}")
        return self.db.query(f"SELECT {field_str} FROM {t_background}")

    # --- 2. 科学诊断绘图流 (原子函数) ---

    def plot_proper_motion_vpd(self, v_target, t_source, key_ref=None, ax=None):
        """
        绘制矢量点图 (Vector Point Diagram, VPD)。

        展示天体自行的分布，背景为全域背景星。

        Args:
            v_target (str): 候选成员视图。
            t_source (str): 物理参数参考源表。
            key_ref (str, optional): 文献对比参考键。
            ax (matplotlib.axes.Axes, optional): 绘图轴对象。
        """
        if key_ref is None:
            key_ref = self.target_category

        ref_cfg = DATA.get(key_ref)
        if not ref_cfg:
            self.logger.warning(f"⚠️ [VPD] 未找到参考星表配置: {key_ref}，将不绘制背景。")
            t_ref = None
            col_prob = None
        else:
            t_ref = ref_cfg.get("stx_view")
            col_prob = TMPL.COL_PROB.format(idx=key_ref)

        df_enriched = self.db.enrich_with_gaia_data(
            v_target, t_source, needed_fields=self.FEATURE_GROUPS["proper_motion"]
        )

        # 第二步：安全的日志打印（只打印存在的列）
        self.logger.debug(f"准备绘制 VPD，数据规模: {len(df_enriched)}")

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 6))

        # 绘制背景
        if t_ref:
            df_bg = self._get_bg_data(t_ref, self.FEATURE_GROUPS["proper_motion"])
            ax.scatter(
                df_bg["pmra"],
                df_bg["pmdec"],
                c="lightgray",
                s=1,
                alpha=0.3,
                label=f"Field Stars (n={len(df_bg)})",
            )

        sc = None  # 初始化散点图对象

        label_target = f"Candidates (n={len(df_enriched)})"

        if col_prob not in df_enriched.columns or pd.isna(df_enriched[col_prob]).all():
            self.logger.warning(
                f"列 {col_prob} 缺失或为空，将使用单一颜色绘制 VPD 目标点。"
            )
            ax.scatter(
                df_enriched["pmra"],
                df_enriched["pmdec"],
                c="crimson",
                s=20,
                edgecolors="k",
                zorder=5,
                label=label_target,
            )
        else:
            # 使用我方概率与文献概率的差异进行热力着色
            sg_prob_col = STD_COLS["PROB"]
            diff = (df_enriched[sg_prob_col] - df_enriched[col_prob].fillna(0)).abs()
            sc = ax.scatter(
                df_enriched["pmra"],
                df_enriched["pmdec"],
                c=diff,
                cmap="Reds",
                s=40,
                edgecolors="k",
                zorder=5,
            )

        ax.set_xlabel("pmra (mas/yr)")
        ax.set_ylabel("pmdec (mas/yr)")
        ax.set_title("Proper Motion VPD")
        ax.legend(loc="best")

        if sc is not None:
            ax.figure.colorbar(sc, ax=ax, label="Probability Discrepancy")

        if standalone:
            plt.tight_layout()
            self._save_plot(fig, "VPD", key_ref)

    def plot_cmd(self, v_target, t_source=None, key_ref=None, ax=None):
        """
        绘制颜色-星等图 (Color-Magnitude Diagram, CMD)。

        Args:
            v_target (str): 候选成员视图。
            t_source (str): 物理参数源表。
            key_ref (str, optional): 参考背景星源。
        """
        t_ref = DATA.get(key_ref, {}).get("stx_view") if key_ref else None
        df = self.db.enrich_with_gaia_data(
            v_target, t_source, needed_fields=self.FEATURE_GROUPS["photometry"]
        )

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 9))

        if t_ref:
            df_bg = self._get_bg_data(t_ref, self.FEATURE_GROUPS["photometry"])
            ax.scatter(
                df_bg["color"],
                df_bg["mag"],
                c="gray",
                s=1.5,
                alpha=0.2,
                label=f"Field Stars (n={len(df_bg)})",
            )
        # self.logger.debug(f"cmd目标数据示例:\n{df.head(30)}")
        label_target = f"Candidates (n={len(df)})"
        ax.scatter(
            df["color"],
            df["mag"],
            c="crimson",
            marker="*",
            s=30,
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
            label=label_target,
        )

        ax.invert_yaxis()
        ax.set_xlabel("G_BP - G_RP")
        ax.set_ylabel("G (mag)")
        ax.legend(loc="best")
        ax.set_title("Color-Magnitude Diagram")

        if standalone:
            plt.tight_layout()
            self._save_plot(fig, "CMD", key_ref)

    def plot_probability_consensus(self, v_target, ax=None):
        """
        绘制概率共识对比图 (SG Prob vs Ref Prob)。
        """
        ref_prob_col = STD_COLS.get("REF_PROB", "r_prob")
        sg_prob_col = STD_COLS["PROB"]

        show_plot = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        query = f"SELECT {sg_prob_col}, {ref_prob_col} FROM {v_target}"
        df = self.db.query(query)

        if df.empty:
            self.logger.warning("概率共识图数据为空。")
            return

        if ref_prob_col not in df.columns or pd.isna(df[ref_prob_col]).all():
            self.logger.warning(f"列 {ref_prob_col} 缺失或全空，无法绘制概率共识图。")
            ax.text(
                0.5,
                0.5,
                "No Reference Probabilities",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            return

        ax.scatter(
            df[sg_prob_col],
            df[ref_prob_col],
            c="royalblue",
            alpha=0.5,
            edgecolors="k",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Consensus")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("My Prob")
        ax.set_ylabel("Ref Prob")
        ax.set_title("Probability Consensus")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        if show_plot:
            self._save_plot(fig, "Consensus")

    # --- 3. 业务逻辑与套件 ---

    def plot_diagnostic_suite(self, v_target, t_source, key_ref=None):
        """
        [一键套件] 生成科学诊断仪表盘图表 (VPD + CMD + Consensus)。
        """
        if key_ref is None:
            key_ref = self.target_category or cfg.DEFAULT_CATEGORY

        t_ref = DATA.get(key_ref, {}).get("stx_view", "none")

        self.logger.info(f"🚀 正在生成诊断仪表盘: {v_target} vs {t_ref}...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f"Diagnostic Dashboard: {v_target} (Ref: {key_ref})",
            fontsize=16,
            fontweight="bold",
        )

        self.plot_probability_consensus(v_target, ax=axes[0])
        self.plot_proper_motion_vpd(
            v_target, t_source=t_source, key_ref=key_ref, ax=axes[1]
        )
        self.plot_cmd(v_target, t_source=t_source, key_ref=key_ref, ax=axes[2])

        plt.tight_layout()
        self._save_plot(fig, "Dashboard", key_ref)

    def run_cross_match_analysis(self, result_table, ref_table):
        """
        [整合接口] 阶段一：执行交叉匹配，产出分歧样本清单。

        此方法作为 Actions 生成宽表后的第一步分析，专门提取我方高概率但文献低概率的“惊喜”源。
        """
        self.logger.info(f"🚀 开始与参考表 [{ref_table}] 进行匹配分析...")

        # 1. 准备就绪视图
        view_name = result_table
        prob_col = STD_COLS["PROB"]
        cat = self.target_category
        ref_prob_col = TMPL.COL_PROB.format(idx=cat)

        # 2. 提取分歧样本：增加集群、类别和模式前缀防止覆盖
        discrepancy_view = TMPL.V_DIFF.format(
            cluster=self.target_cluster.lower(),
            category=cat,
            mode=self.mode,
            idx=ref_table,
        )
        sql = f"""
            SELECT * FROM {view_name}
            WHERE {prob_col} > {cfg.AUDIT_PROB_HIGH} AND ({ref_prob_col} < {cfg.AUDIT_PROB_LOW} OR {ref_prob_col} IS NULL)
        """
        self.db.register_view_from_sql(discrepancy_view, sql)
        count = self.db.get_row_count(discrepancy_view)
        self.logger.info(
            f"🔍 识别到 {count} 个潜在新成员候选，已存入视图: {discrepancy_view}"
        )

        return discrepancy_view

    def vld_and_exp_discoveries(self, v_candidates, key_ref, validator):
        """阶段二：执行物理参数验证并保存结果"""
        t_ref = DATA[key_ref]["stx_view"]
        len_candidates = self.db.get_row_count(v_candidates)
        self.logger.info(f"🚀 开始对 {len_candidates} 个候选进行验证")
        if len_candidates <= 0:
            self.logger.info(f"✨ 无需验证：与 [{t_ref}] 无显著分歧。")
            return None

        # 1. 诊断图组 (可视化验证)
        seed_idx = cfg.CLUSTERS[self.target_cluster]["SEED_IDX"]
        t_gaia = DATA[seed_idx]["stx_view"]
        self.plot_diagnostic_suite(v_candidates, t_gaia, key_ref)

        # 2. 物理审核
        df_to_audit = self.db.enrich_with_gaia_data(v_candidates, t_gaia)

        # 此处假设 validator 支持 run_full_audit_ex 批量模式
        df_audit_report = validator.run_full_audit_ex(v_candidates)

        # 3. 统计并保存
        self.report_validation_results(df_audit_report, t_ref)
        return df_audit_report

    def report_validation_results(self, df_audit_report, t_ref):
        """审计结果汇总报告。"""
        output_path = cfg.EXPORT_DIR / TMPL.FILE_NEW_CANDIDATES.format(ref=t_ref)
        df_audit_report.to_csv(output_path, index=False)

        passed = len(
            df_audit_report[df_audit_report["audit_status"] == "Confirmed Member"]
        )
        self.logger.info(
            f"💾 验证报告已固化: {output_path} (通过数: {passed}/{len(df_audit_report)})"
        )

    # --- 算法性能诊断 ---

    def analyze_audit_performance(self, t_audit_report):
        """
        [整合接口] 综合审计性能评估。

        从审计报告表中一次性提取：召回率、新发现统计、歧见分析及漏判诊断。
        """
        self.logger.info(f"📊 正在生成算法性能综合报告: {t_audit_report}")

        sql = f"""
            SELECT
                audit_status,
                count(*) as count,
                avg(mag) as avg_mag
            FROM {t_audit_report}
            GROUP BY audit_status
        """
        df_stats = self.db.query(sql).set_index("audit_status")

        # 1. 召回率计算 (Recall)
        hits = df_stats.get("count", {}).get("Confirmed Member", 0)
        misses = df_stats.get("count", {}).get("Literature Only", 0)
        total_lit = hits + misses
        recall = (hits / total_lit * 100) if total_lit > 0 else 0

        # 2. 新发现统计 (Precision/New Discovery)
        new_candidates = df_stats.get("count", {}).get("New Candidate", 0)

        # 3. 输出汇总日志
        self.logger.info(f"{'='*50}")
        self.logger.info(f"📈 算法文献召回率: {recall:.2f}% ({hits}/{total_lit})")
        self.logger.info(f"✨ 物理符合但文献未收录的新源: {new_candidates} 颗")

        if misses > 0:
            avg_miss_mag = df_stats.get("avg_mag", {}).get("Literature Only", 0)
            self.logger.info(
                f"🔍 漏判源平均星等: {avg_miss_mag:.2f} (建议检查暗端灵敏度)"
            )

        self.logger.info(f"{'='*50}")

        return {
            "recall": recall,
            "new_discoveries": new_candidates,
            "missing_count": misses,
        }

    def plot_spatial_comparison(self, v_target, key_ref=None, show_radius=True):
        """
        [整合接口] 绘制空间分布对比图。

        统一了背景星、文献成员和我方新发现候选者的空间可视化。
        """
        if key_ref is None:
            key_ref = self.target_category

        ref_table = DATA[key_ref]["stx_view"]
        self.logger.info(f"正在生成空间分布对比图: {v_target} vs {ref_table}")

        seed_idx = cfg.CLUSTERS[self.target_cluster]["SEED_IDX"]
        gaia_table = DATA[seed_idx]["stx_view"]
        df = self.db.enrich_with_gaia_data(v_target, gaia_table)
        df_ref = self.db.query(f"SELECT * FROM {ref_table}")

        fig, ax = plt.subplots(figsize=(10, 8))

        # 核心逻辑：区分“共有”与“新增”
        col_id = STD_COLS["ID"]
        mine_ids = df[col_id].astype("Int64")
        ref_ids = df_ref[col_id].astype("Int64")

        df_new = df[~mine_ids.isin(ref_ids)]
        df_common = df[mine_ids.isin(ref_ids)]

        # 绘图层
        ax.scatter(
            df_ref["ra"],
            df_ref["dec"],
            c="lightgray",
            s=1,
            alpha=0.5,
            label=f"Ref: {key_ref}",
        )
        ax.scatter(
            df_common["ra"],
            df_common["dec"],
            c="royalblue",
            s=3,
            label=f"Common (n={len(df_common)})",
        )
        ax.scatter(
            df_new["ra"],
            df_new["dec"],
            c="orangered",
            s=10,
            edgecolors="white",
            linewidths=0.3,
            zorder=5,
            label=f"New Candidates (n={len(df_new)})",
        )

        if show_radius:
            ctx = cfg.CLUSTERS[cfg.target_cluster]
            center_ra = ctx["CENTER_RA"]
            center_dec = ctx["CENTER_DEC"]
            radius = ctx.get("RADIUS", 5.0)
            circle = Circle(
                (center_ra, center_dec),
                radius,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(circle)

        # 4. 修饰图表
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(f"Spatial Distribution: {self.target_cluster.upper()} ({self.mode.upper()})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_xaxis()
        ax.legend(loc="best", markerscale=3)
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        self._save_plot(fig, "Spatial_Comparison", key_ref)

    def plot_probability_distribution(self, view_name, bins=20):
        """
        生成带数值标签的成员概率分布直方图。
        """
        # 1. 获取数据
        prob_col = STD_COLS["PROB"]
        query = f"SELECT {prob_col} FROM {view_name}"
        df = self.db.query(query)

        if df.empty:
            self.logger.warning(f"视图 {view_name} 中没有数据。")
            return

        # 2. 绘图并接收返回值
        fig, ax = plt.subplots(figsize=(12, 7))
        counts, edges, patches = plt.hist(
            df[prob_col],
            bins=bins,
            range=(0, 1),
            color="#3498db",
            edgecolor="white",
            alpha=0.8,
        )

        for count, edge in zip(counts, edges):
            if count > 0:  # 只标注有数据的柱子
                x_pos = edge + (edges[1] - edges[0]) / 2
                y_pos = count

                ax.text(
                    x_pos,
                    y_pos + (max(counts) * 0.01),  # 稍微向上偏移 1% 的高度
                    f"{int(count)}",
                    ha="center",  # 水平居中
                    va="bottom",  # 垂直基准在线方
                    fontsize=10,
                    color="#2c3e50",
                    fontweight="bold",
                )

        # 3. 美化与保存
        plt.title(
            f"Membership Probability Distribution - {self.target_cluster.upper()}",
            fontsize=14,
            pad=20,
        )
        plt.xlabel("Probability (SeedGMM Score)", fontsize=12)
        plt.ylabel("Number of Sources", fontsize=12)
        plt.ylim(0, max(counts) * 1.1)  # 增加 10% 的纵向空间防止标签被顶框遮挡
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        # 添加门限参考线
        plt.axvline(
            x=cfg.MEMBER_SAMPLE_THRESHOLD,
            color="#e74c3c",
            linestyle="--",
            label=f"Threshold ({cfg.MEMBER_SAMPLE_THRESHOLD})",
        )
        plt.legend()

        self._save_plot(fig, "Prob_Dist", view_name)
        return fig

    def audit_missing_sources(self, df_missing):
        self.logger.info("--- 开始回溯审计漏检源 ---")

        # 统计 RUWE
        bad_data = df_missing[df_missing["ruwe"] > cfg.AUDIT_RUWE_LIMIT]
        self.logger.info(f"数据质量差 (RUWE > {cfg.AUDIT_RUWE_LIMIT}): {len(bad_data)} 个 (建议维持剔除)")

        # 统计视差残差
        ctx = cfg.CLUSTERS.get(self.target_cluster, {})
        plx_ref = ctx.get("PLX_REF", 0.0)
        plx_dist = (df_missing["plx"] - plx_ref).abs()
        wrong_dist = df_missing[plx_dist > cfg.AUDIT_PLX_RESIDUAL_LIMIT]  # 距离偏差超过 ~20pc
        self.logger.info(f"距离显著偏离星团中心: {len(wrong_dist)} 个 (建议维持剔除)")

        # 剩余的源：可能是你的模型“漏网之鱼”
        candidates_to_reconsider = df_missing[
            ~df_missing["id"].isin(bad_data["id"])
            & ~df_missing["id"].isin(wrong_dist["id"])
        ]
        self.logger.info(
            f"潜在遗漏的高质量成员: {len(candidates_to_reconsider)} 个 (需重点核实)"
        )

    def cross_match_report(self, t_target, k_ref):
        """
        [接口 2] 展示层：生成包含双方总数对比的统计报告
        修正版：采用 SQL 集合逻辑计算底数，解决由于前置过滤导致的统计不一致问题。
        """
        # t_ref = TMPL.T_ALN.format(prefix=k_ref, cluster=CLUSTER_NAME)
        v_stx_ref = TMPL.V_STX.format(prefix=k_ref)
        col_prob = TMPL.COL_PROB.format(prefix=k_ref)
        self.logger.info(f"正在生成 {t_target} 与 {v_stx_ref} 的交叉匹配报告...")

        # --- 1. SQL 物理对账（确保总数逻辑自洽） ---
        # 使用 COUNT(DISTINCT id) 确保不受重复行干扰

        # 我方视图总数 (A)
        total_my = self.db.get_row_count(t_target, column="id")

        # 参考表总数 (B)
        ref_total_count = self.db.get_row_count(v_stx_ref, column="id")

        # 物理交集数 (A ∩ B)
        # 只有 ID 同时存在于两张表才算 Overlap
        total_overlap = self.db.con.execute(f"""
            SELECT COUNT(DISTINCT a.id) 
            FROM {t_target} a 
            INNER JOIN {v_stx_ref} b ON a.id = b.id
        """).fetchone()[0]
        # 注意：此处 JOIN count 逻辑较为特殊，暂维持 execute

        # 物理漏检数 (B - A): 对方有，但我方视图里没有
        not_in_me = self.db.con.execute(f"""
            SELECT COUNT(DISTINCT id) FROM {v_stx_ref} 
            WHERE id NOT IN (SELECT id FROM {t_target})
        """).fetchone()[0]
        # 物理新增数 (A - B): 我方有，但对方没收录
        not_in_ref = total_my - total_overlap

        sg_prob_col = STD_COLS["PROB"]

        # --- 2. 加载数据进行概率分布分析（仅针对 Overlap 样本） ---
        self.logger.info(f"正在从分析视图 {t_target} 加载数据帧进行分类统计...")
        df = self.db.query(f"SELECT * FROM {t_target}")

        # 筛选出含有参考表概率的行（Overlap 样本）
        df_overlap = df[df[col_prob].notna()]

        # 基于【交集样本】进行共识分类统计
        both_high = len(
            df_overlap[(df_overlap[sg_prob_col] >= cfg.AUDIT_PROB_HIGH) & (df_overlap[col_prob] >= cfg.AUDIT_PROB_HIGH)]
        )
        both_low = len(
            df_overlap[(df_overlap[sg_prob_col] < cfg.AUDIT_PRO_LOW) & (df_overlap[col_prob] < cfg.AUDIT_PROB_LOW)]
        )

        # 歧见统计：ID 都在，但概率判定冲突
        only_me = len(
            df_overlap[(df_overlap[sg_prob_col] >= cfg.AUDIT_PROB_HIGH) & (df_overlap[col_prob] < cfg.AUDIT_PROB_LOW)]
        )
        only_ref = len(
            df_overlap[(df_overlap[sg_prob_col] < cfg.AUDIT_PROB_LOW) & (df_overlap[col_prob] >= cfg.AUDIT_PROB_HIGH)]
        )

        # --- 3. 生成报告文本 ---
        report = f"""
        {'='*45}
                算法共识分析报告
        {'='*45}
        [数据源概览]
        我方总源数 ({t_target}):    {total_my}
        参考表总源数 ({v_stx_ref}): {ref_total_count}
        双方共同收录 (Overlap): {total_overlap}
        
        [收录差异]
        ❓ 参考星表未收录 (New):    {not_in_ref}  (仅在我方表中)
        🔍 我方算法未收录 (Missing):    {not_in_me}  (仅在参考表中)
        
        [分类共识]
        ✅ 双方共识成员 (Prob >= 0.7):  {both_high}
        ✅ 双方共识背景 (Prob < 0.3):   {both_low}

        [算法歧见]
        ⭐ 仅由 SeedGMM 认定-SG>.7; REF<0.3:    {only_me}
        ❌ 仅由 {k_ref} 认定-SG<0.3; REF>0.7:   {only_ref}
        {'='*45}
        """
        self.logger.info(report)

        return {
            "only_me_high": only_me,
            "both_high": both_high,
            "total_overlap": total_overlap,
            "missing_count": not_in_me,
            "new_count": not_in_ref,
        }

    def audit_literature_via_simbad(self, v_subset, top_n=None):
        """
        [科研核实] 自动从分析子集中提取 ID 并通过 SIMBAD 查询文献背景。

        Args:
            v_subset (str): 子集视图名。
            top_n (int, optional): 限制查询数量。
        """
        df_target = self.db.query(f"SELECT * FROM {v_subset}")

        if df_target is None or df_target.empty:
            self.logger.warning(f"子集 {v_subset} 为空，取消 SIMBAD 查询。")
            return None

        if top_n:
            df_target = df_target.sort_values(STD_COLS["PROB"], ascending=False).head(
                top_n
            )

        star_ids = df_target["id"].tolist()
        self.logger.info(f"🔍 准备对 {len(star_ids)} 颗星进行 SIMBAD 文献核实...")

        # 2. 配置 SIMBAD (只配置一次)
        Simbad.reset_votable_fields()  # 重置默认字段防止冲突
        Simbad.add_votable_fields("p_count")

        results = []
        for star_id in star_ids:
            formatted_id = (
                f"Gaia DR3 {star_id}" if isinstance(star_id, (int, float)) else star_id
            )

            try:
                result_table = Simbad.query_object(formatted_id)
                bib_table = Simbad.query_bibobj(formatted_id)

                if result_table is not None:
                    p_count = result_table[0]["P_COUNT"]
                    latest_bib = (
                        bib_table["bibcode"][0] if bib_table is not None else "None"
                    )

                    results.append(
                        {
                            "id": star_id,
                            "simbad_name": formatted_id,
                            "pub_count": p_count,
                            "latest_ref": latest_bib,
                            "status": "Found",
                        }
                    )
                else:
                    results.append(
                        {
                            "id": star_id,
                            "pub_count": 0,
                            "latest_ref": "None",
                            "status": "Not Found",
                        }
                    )

            except Exception as e:
                self.logger.error(f"查询 {formatted_id} 时发生网络或解析错误: {e}")
                results.append(
                    {
                        "id": star_id,
                        "pub_count": -1,
                        "latest_ref": "Error",
                        "status": "Error",
                    }
                )

        # 3. 转换为 DataFrame 并合并回原物理参数
        df_simbad = pd.DataFrame(results)

        # 将 SIMBAD 结果与原始宽表中的物理参数（如 plx, pm, sg_prob）合并
        final_report = pd.merge(df_target, df_simbad, on="id", how="left")

        # 4. 保存报告
        output_path = cfg.EXPORT_DIR / f"simbad_audit_{v_subset}.csv"
        final_report.to_csv(output_path, index=False)
        self.logger.info(f"✅ SIMBAD 审计完成！报告已保存至: {output_path}")

        return final_report

    def analyze_hunt24_missing_sources(self, t_audit_report):
        """
        分析 Hunt24 遗漏的源在星等(G)上的分布特征。
        """
        self.logger.info(f"📊 开始分析 Hunt24 遗漏源的分布特征...")

        # 1. 从审计表中提取被标记为“遗漏”的源，并关联 Gaia 物理参数
        field_idx = cfg.CLUSTERS[self.target_cluster]["FIELD_IDX"]
        t_base = DATA[field_idx]["stx_view"]

        prob_col = STD_COLS["PROB"]
        query = f"""
            SELECT a.id, a.{prob_col}, g.mag as phot_g_mean_mag
            FROM {t_audit_report} a
            JOIN {t_base} g ON a.id = g.id
            WHERE a.audit_status = 'Literature Only'
        """
        self.logger.debug(f"执行 SQL 查询以获取 Hunt24 遗漏源的星等分布:\n{query}")

        df_missing = self.db.query(query)

        if df_missing.empty:
            self.logger.warning("⚠️ 没有发现 Hunt24 遗漏的源，跳过分析。")
            return

        # 2. 统计星等分布
        total_missing = len(df_missing)
        dark_samples = df_missing[df_missing["phot_g_mean_mag"] > cfg.AUDIT_MAG_LIMIT_HUNT24]
        dark_count = len(dark_samples)
        percentage = (dark_count / total_missing) * 100 if total_missing > 0 else 0

        self.logger.info(f"📈 遗漏源统计摘要:")
        self.logger.info(f"   - 总计遗漏: {total_missing} 颗")
        self.logger.info(f"   - 其中暗端 (G > {cfg.AUDIT_MAG_LIMIT_HUNT24}): {dark_count} 颗")
        self.logger.info(f"   - 暗端占比: {percentage:.2f}%")

        if percentage > 50:
            self.logger.info(
                "🚀 结论确认：你的算法在暗端（小质量端）确实比 Hunt24 更具灵敏度！"
            )

        # 3. 可视化星等直方图
        plt.figure(figsize=(10, 6))
        plt.hist(
            df_missing["phot_g_mean_mag"],
            bins=20,
            alpha=0.7,
            color="teal",
            edgecolor="black",
        )
        plt.axvline(
            19, color="red", linestyle="--", label="G = 19 (Hunt24 Sensitivity Limit)"
        )
        plt.title(
            f"Magnitude Distribution of Sources Missed by Hunt24 (N={total_missing})"
        )
        plt.xlabel("G Magnitude")
        plt.ylabel("Count")
        plt.legend()

        # 保存图表
        # 扁平化处理：将原本散落在 reports 下的内容统一收纳到数据导出目录
        save_path = cfg.EXPORT_DIR / TMPL.FILE_MISS_MAG_DIST
        plt.savefig(save_path)
        self.logger.info(f"💾 星等分布直方图已保存至: {save_path}")

        return df_missing

    def _audit_hunt24_catalogue(self, t_candidates):
        """
        [私有方法] 以 Hunt24 为主体进行审计，识别其误报与遗漏
        :param t_candidates: PG算法生成的候选者表名 (作为审计工具)
        """
        from config import IDX_HUNT

        v_hunt = DATA[IDX_HUNT]["aln_view"]

        # 结果表名：标识是对 Hunt24 的审计结果
        t_audit_report = TMPL.V_ADT_HUNT24.format(src=t_candidates)
        self.logger.info(f"🔍 正在审计 Hunt24 星表... 使用 {t_candidates} 作为校验准则")

        # 核心逻辑：以 Hunt24 (h) 为主，对比 PG候选 (c)
        # 逻辑点：
        # - h 有, c 有 -> 双方一致
        # - h 有, c 无 -> Hunt24 可能误报 (或者 PG 算法太严)
        # - h 无, c 有 -> Hunt24 遗漏 (PG 算法的新发现)
        sql_audit = f"""
            CREATE TABLE {t_audit_report} AS
            SELECT 
                COALESCE(h.id, c.id) AS id,
                h.prob AS hunt24_prob,
                c.sg_prob AS pg_prob,
                CASE 
                    WHEN h.id IS NOT NULL AND c.id IS NOT NULL THEN '双方评估一致'
                    WHEN h.id IS NOT NULL AND c.id IS NULL THEN 'Hunt24可能误报(需要物理复核)'
                    WHEN h.id IS NULL AND c.id IS NOT NULL THEN 'Hunt24遗漏(PG算法新发现)'
                END AS audit_result
            FROM {v_hunt} as h
            FULL OUTER JOIN {t_candidates} c ON h.id = c.id
            WHERE h.id IS NOT NULL OR c.id IS NOT NULL
        """

        try:
            self.db.drop_table(t_audit_report)
            self.db.execute(sql_audit)

            # 统计审计结果
            self._print_audit_stats(t_audit_report)
            # 返回审计结果表名以供后续分析使用
            return t_audit_report

        except Exception as e:
            self.logger.error(f"❌ Hunt24 审计流程失败: {str(e)}")
            return None
