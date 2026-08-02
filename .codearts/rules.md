# 项目星表处理规范 (STILTS)

当用户提出关于 FITS、VOTable、Parquet 等星表数据的过滤、筛选、交叉匹配或清洗需求时：
1. **禁止直接加载全量数据**：严禁直接使用 Python Pandas/Astropy 将数 GB 的原始数据全量读入内存，以防 OOM。
2. **优先使用 STILTS 命令行**：必须优先生成 `stilts` 命令行或基于 Python `subprocess` 调用 `stilts` 的代码。
3. **常用表达式规范**：
   - 数据过滤：`stilts tpipe in=<input> cmd="select '<expr>'" cmd="keepcols '<cols>'" out=<output> ofmt=csv/fits`
   - 交叉匹配：`stilts tmatch2 in1=<a> in2=<b> matcher=sky params=1.0 values1="ra dec" values2="ra dec" out=<output>`
4. **输出控制**：如果是在终端运行，输出格式尽量指定为轻量 CSV，并通过终端打印 `describe()` 统计摘要。

- STILTS Jar 包位置：`D:/git/repo/astro-research/tools/stilts/stilts.jar`
- 运行方式：`java -jar D:/git/repo/astro-research/tools/stilts/stilts.jar`