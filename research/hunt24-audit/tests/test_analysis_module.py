"""
test_analysis_module.py - analysis 子包单元与集成测试套件
依赖环境: pytest, duckdb, pandas, matplotlib
运行方式: pytest tests/test_analysis_module.py -v
"""
import sys
from pathlib import Path

# 将工程根目录添加到 sys.path 中（定位到 tests 的上一级目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import pytest
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

import config as cfg
from modules.analysis.plotter import AstroPlotter
from modules.analysis.evaluator import AuditEvaluator
from modules.analysis.simbad_client import SimbadClient
from modules.analysis.analyzer import AstroAnalyzer, _PCAMeta

# 指定你的数据库文件绝对路径
DB_PATH = r"D:\git\repo\astro-research\research\hunt24-audit\data\warehouse\astrodb_internal.db"


# =============================================================================
# 🛠️ Pytest Fixtures (测试上下文与数据库连接)
# =============================================================================

@pytest.fixture(scope="module")
def db_conn():
    """建立与真实数据库文件的只读连接（如果文件不存在，则创建内存镜像数据避免 CI 崩溃）。"""
    if os.path.exists(DB_PATH):
        # 建立数据库连接（以 read_only 模式防误涂改）
        conn = duckdb.connect(DB_PATH, read_only=True)
        # 封装简易的 query 接口匹配系统内部 DuckDB 包装器行为
        class DBWrapper:
            def __init__(self, raw_conn):
                self.con = raw_conn
            def query(self, sql):
                return self.con.execute(sql).df()
            def get_row_count(self, table_name, column="id"):
                return self.con.execute(f"SELECT COUNT({column}) FROM {table_name}").fetchone()[0]
            def register_view_from_sql(self, view_name, sql):
                self.con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")

        wrapper = DBWrapper(conn)
        yield wrapper
        conn.close()
    else:
        # 降级：若路径在测试机上不存在，自动生成 Mock 数据库包装器
        mock_db = MagicMock()
        mock_df = pd.DataFrame({
            "id": [101, 102, 103, 104],
            "ra": [10.5, 10.6, 10.7, 10.8],
            "dec": [20.1, 20.2, 20.3, 20.4],
            "pmra": [-1.2, -1.5, -1.1, -1.0],
            "pmdec": [3.4, 3.2, 3.5, 3.1],
            "mag": [12.5, 14.1, 15.2, 18.0],
            "color": [0.8, 1.1, 1.3, 1.9],
            "prob": [0.95, 0.85, 0.20, 0.05],
            "core_prob": [0.90, 0.80, 0.10, 0.01],
            "tail_prob": [0.05, 0.05, 0.10, 0.04],
            "in_tube": [True, True, False, False],
            "seed_type": ["PCA", None, None, None],
            "x_match_tag": ["Matched", "PG Only", "Ref Only", "Matched"]
        })
        mock_db.query.return_value = mock_df
        mock_db.con.execute.return_value.fetchall.return_value = [(col,) for col in mock_df.columns]
        mock_db.con.execute.return_value.fetchone.return_value = [4]
        yield mock_db


@pytest.fixture(scope="module")
def sample_master_table(db_conn):
    """自动检测数据库中是否存在有效的 master 表或视图，若无则寻找候选表。"""
    if hasattr(db_conn, "con"):
        tables = [r[0] for r in db_conn.con.execute("SHOW TABLES").fetchall()]
        masters = [t for t in tables if "master" in t.lower()]
        if masters:
            return masters[0]
        elif tables:
            return tables[0]
    return "master_table"


@pytest.fixture
def mock_run_context(sample_master_table):
    """构建 Phase 6 运行所需的 Mock RunContext。"""
    ctx = MagicMock()
    ctx.cluster_id = "M44"
    ctx.category = "hunt24"
    ctx.feature_space = "3d"
    ctx.algorithm = "dbscan"
    ctx.skip_viz = False
    
    # State Mock
    ctx.state.master_table = sample_master_table
    ctx.state.tube_pca_center = [130.1, 19.6]
    ctx.state.tube_pca_components = np.eye(2)
    ctx.state.tube_pca_mean = np.array([0.0, 0.0])
    ctx.state.tube_width = 1.5
    ctx.state.tube_length = 5.0
    ctx.state.computed_dbscan_eps = 0.35
    ctx.state.gmm_config = {"DBSCAN_EPS": 0.3, "DBSCAN_MIN_SAMPLES": 10}
    ctx.algo_params = {"eps": 0.35, "min_samples": 8}
    
    return ctx


# =============================================================================
# 1. 单元测试: AstroPlotter (纯 Python 可视化引擎)
# =============================================================================

class TestAstroPlotter:
    """测试 AstroPlotter 的 3 张核心诊断图绘制功能。"""

    def test_render_tube_mask(self, tmp_path):
        plotter = AstroPlotter()
        
        # 构造模拟数据
        df_all = pd.DataFrame({"ra": [130.0, 130.2, 130.5], "dec": [19.5, 19.6, 19.8]})
        df_seeds = pd.DataFrame({"ra": [130.1], "dec": [19.6]})
        df_tube = pd.DataFrame({"ra": [130.0, 130.2], "dec": [19.5, 19.6]})
        
        pca_proxy = _PCAMeta(
            center=np.array([130.1, 19.6]),
            components=np.eye(2),
            mean=np.array([0.0, 0.0]),
            tube_width=1.5
        )

        with patch.object(cfg, "ANALYSIS_DIR", tmp_path):
            output_path = plotter.render_tube_mask(
                df_all, df_seeds, df_tube, pca_proxy, 
                cluster_id="m44", length_deg=5.0, width_deg=1.5
            )
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert "spatial_tube" in output_path.name

    def test_render_channel_probs(self, tmp_path):
        plotter = AstroPlotter()
        df_res = pd.DataFrame({
            "prob": [0.9, 0.8, 0.1, 0.05],
            "core_prob": [0.85, 0.75, 0.05, 0.01],
            "tail_prob": [0.05, 0.05, 0.05, 0.04]
        })

        with patch.object(cfg, "ANALYSIS_DIR", tmp_path):
            output_path = plotter.render_channel_probs(df_res, cluster_id="m44")
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert "prob_dist" in output_path.name

    def test_render_cross_match_diagnostics(self, tmp_path):
        plotter = AstroPlotter()
        df_cross = pd.DataFrame({
            "ra": [130.0, 130.1, 130.2],
            "dec": [19.5, 19.6, 19.7],
            "pmra": [-1.2, -1.5, 0.5],
            "pmdec": [3.4, 3.2, -0.1],
            "mag": [12.0, 14.5, 18.0],
            "color": [0.8, 1.2, 2.1],
            cfg.MASTER_COLS["X_MATCH"]: ["Matched", "PG Only", "Ref Only"]
        })

        with patch.object(cfg, "ANALYSIS_DIR", tmp_path):
            output_path = plotter.render_cross_match_diagnostics(df_cross, cluster_id="m44")
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert "cross_match_diag" in output_path.name


# =============================================================================
# 2. 单元测试: AuditEvaluator & SimbadClient (评估与 API 客户端)
# =============================================================================

class TestAuditEvaluator:
    """测试评估引擎的指标计算功能。"""

    def test_analyze_audit_performance(self, db_conn):
        evaluator = AuditEvaluator(db_instance=db_conn)
        
        # Mock 在数据库中存在审计结果表
        with patch.object(db_conn, "query") as mock_query:
            mock_df = pd.DataFrame({
                "count": [80, 20, 15],
                "avg_mag": [13.2, 16.5, 18.1]
            }, index=["Confirmed Member", "Literature Only", "New Candidate"])
            mock_query.return_value.set_index.return_value = mock_df

            res = evaluator.analyze_audit_performance("test_audit_table")
            assert "recall" in res
            assert res["recall"] == 80.0  # 80 / (80 + 20) * 100
            assert res["new_discoveries"] == 15
            assert res["missing_count"] == 20

    @patch("modules.analysis.simbad_client.Simbad.query_object")
    @patch("modules.analysis.simbad_client.Simbad.query_bibobj")
    def test_simbad_client_query(self, mock_bib, mock_obj, tmp_path):
        """测试 SIMBAD API 客户端的解析与 CSV 导出。"""
        client = SimbadClient()
        df_target = pd.DataFrame({"id": [123456789], "prob": [0.98]})
        
        # Mock SIMBAD 网络返回结果
        mock_obj.return_value = [{"P_COUNT": 5}]
        mock_bib.return_value = {"bibcode": ["2024A&A...123A"]}

        with patch.object(cfg, "EXPORT_DIR", tmp_path):
            out_path = client.export_audit_csv(df_target, output_name="unit_test")
            assert out_path is not None
            assert out_path.exists()
            
            df_out = pd.read_csv(out_path)
            assert "pub_count" in df_out.columns
            assert df_out.iloc[0]["pub_count"] == 5
            assert df_out.iloc[0]["latest_ref"] == "2024A&A...123A"


# =============================================================================
# 3. 集成测试: AstroAnalyzer (门面主控)
# =============================================================================

class TestAstroAnalyzerIntegration:
    """对 AstroAnalyzer 门面类进行全流程集成测试（模拟 Phase 6 触发）。"""

    @patch("modules.reporter.render_final_report")
    @patch("modules.result_logger.update_dbscan_csv")
    def test_run_phase6_pipeline(self, mock_update_csv, mock_render_report, db_conn, mock_run_context, tmp_path):
        mock_render_report.return_value = {"summary": "Phase 6 Complete"}
        
        analyzer = AstroAnalyzer(db_instance=db_conn, ctx=mock_run_context)
        
        post_result = {"status": "success"}
        audit_result = {
            "stats": {"matched": 100},
            "deep_stats_matched": {},
            "deep_stats_pg_only": {},
            "deep_stats_ref_only": {}
        }

        # 将导出路径指向临时目录，避免污染生产环境
        with patch.object(cfg, "ANALYSIS_DIR", tmp_path), patch.object(cfg, "RESULTS_DIR", tmp_path):
            summary = analyzer.run_phase6(post_result, audit_result)
            
            # 1. 验证报告与日志写入是否被正常委派调用
            assert mock_render_report.called
            assert mock_update_csv.called
            assert summary == {"summary": "Phase 6 Complete"}

    def test_run_all_diagnostics_directly(self, db_conn, mock_run_context):
        """在真实 master 数据表上直接运行全量可视化渲染，并强制刷新成果图。"""
        import time
        from pathlib import Path

        if not hasattr(db_conn, "con"):
            pytest.skip("未找到真实数据库文件，跳过真实数据画图测试。")

        tables = [r[0] for r in db_conn.con.execute("SHOW TABLES").fetchall()]
        master_tables = [t for t in tables if "master" in t.lower()]
        
        if not master_tables:
            pytest.skip("数据库中未查查找包含 master 的数据表。")
            
        # real_master = master_tables[0]
        real_master = "master_m41_hunt_5d_h_dbscan"
        print(f"\n[真实运行] 当前绑定的真实 master 表: {real_master}")

        # 1. 动态对齐真实 M41 的赤道坐标中心 (~101.8°, -20.7°) 及星团名
        mock_run_context.cluster_id = "M41"
        mock_run_context.state.master_table = real_master
        mock_run_context.state.tube_pca_center = [101.8, -20.7]  # 根据你给的数据 RA ~ 101.8, Dec ~ -22.0 调整
        mock_run_context.state.tube_pca_mean = np.array([0.0, 0.0])
        mock_run_context.state.tube_pca_components = np.eye(2)
        mock_run_context.state.tube_length = 3.0
        mock_run_context.state.tube_width = 1.0

        output_dir = Path(cfg.ANALYSIS_DIR)

        # 2. 删除今天的旧 M41 文件，验证是否真的生成了全新图表
        for old_pic in output_dir.glob("m41_*.jpg"):
            try:
                old_pic.unlink()
                print(f"[清理旧图] 已成功删除旧图: {old_pic.name}")
            except Exception:
                pass

        # 3. 运行诊断渲染
        analyzer = AstroAnalyzer(db_instance=db_conn, ctx=mock_run_context)
        analyzer.run_all_diagnostics()

        # 4. 打印全新的 M41 图表文件修改时间
        new_files = list(output_dir.glob("m41_*.jpg"))
        print(f"\n[运行成功] 输出目录: {output_dir}")
        print(f"[全新生成] 本次生成的 m41 图片数量: {len(new_files)}")
        for f in new_files:
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
            print(f"  └── 成果图: {f.name} | 修改时间: {mtime} | 文件大小: {f.stat().st_size / 1024:.1f} KB")

        assert len(new_files) > 0, f"未能为 {real_master} 生成全新的 M41 诊断图！"