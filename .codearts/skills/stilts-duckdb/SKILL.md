# Skill: STILTS & DuckDB 天文数据处理技能

## 1. 默认数据源与工具路径
- 本地 DuckDB 数据库文件：astrodb_internal-main.db
- 中间桥梁管道脚本：.codearts/rules/stilts-duckdb/run_pipeline.py

---

## 2. 工作流程与交互规范
当用户提出需要对 D:/git/repo/astro-research/research/hunt24-audit/data/warehouse/astrodb_internal.db 数据库中的星表进行数据处理、天球坐标交叉匹配（如 tmatch2）、全天或区域天区画图（如 plot2sky）时，请按以下三个步骤进行响应：

1. 构建 DuckDB SQL：根据需求编写高效过滤的 SQL 语句（务必在 SQL 端优先筛选天区、星等、自行等字段，以减少导出开销）。
2. 构建 STILTS 命令行：使用 {input_file} 代表从 DuckDB 导出的临时文件占位符，用 {output_file} 代表最终处理结果占位符。
3. 输出一键运行指令：组装并提供调用 run_pipeline.py 的可直接执行脚本。

---

## 3. 常用场景与命令行模板

### 场景一：天球坐标交叉匹配 (tmatch2)
从 astrodb_internal.db 筛选目标天区/星等的数据，与外部 FITS 参考星表进行天球坐标匹配（例如 1.0 角秒匹配半径）。

运行命令：
python .codearts/rules/stilts-duckdb/run_pipeline.py \
  --db astrodb_internal.db \
  --sql "SELECT ra, dec, phot_g_mean_mag FROM stars WHERE phot_g_mean_mag < 18" \
  --cmd "stilts tmatch2 in1={input_file} ifmt1=csv ra1=ra dec1=dec in2=ref_catalog.fits ifmt2=fits ra2=RA dec2=DEC matcher=sky params=1.0 omode=out out={output_file} ofmt=fits" \
  --out matched_stars.fits


### 场景二：天区分布与坐标绘图 (plot2sky)
提取指定坐标范围的天体数据，使用 STILTS 绘制全天（如 Mollweide 投影）或局部区域分布图。

运行命令：
python .codearts/rules/stilts-duckdb/run_pipeline.py \
  --db astrodb_internal.db \
  --sql "SELECT ra, dec FROM stars WHERE ra BETWEEN 50 AND 60 AND dec BETWEEN 20 AND 30" \
  --cmd "stilts plot2sky in={input_file} ifmt=csv lon=ra lat=dec projection=mollweide layer1=mark shape1=open_circle color1=blue out={output_file} ofmt=png" \
  --out sky_map.png


### 场景三：多参数数据过滤与格式转换 (tpipe)
使用 STILTS 表达式对导出数据进行过滤并转存为高精度 FITS 文件。

运行命令：
python .codearts/rules/stilts-duckdb/run_pipeline.py \
  --db astrodb_internal.db \
  --sql "SELECT ra, dec, pmra, pmdec, parallax FROM stars WHERE parallax > 2.0" \
  --cmd "stilts tpipe in={input_file} ifmt=csv cmd='select parallax>5.0' omode=out out={output_file} ofmt=fits" \
  --out filtered_parallax.fits