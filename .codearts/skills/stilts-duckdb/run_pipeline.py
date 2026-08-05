import argparse
import os
import subprocess
import sys
import tempfile
import duckdb


def run_pipeline(db_path, sql_query, stilts_cmd_template, output_path):
    """从 DuckDB 提取数据并交给 STILTS 执行命令"""
    if not os.path.exists(db_path):
        print(f"错误: 找不到数据库文件 {db_path}")
        sys.exit(1)

    # 1. 连接本地 DuckDB 并导出临时 CSV/FITS
    with tempfile.NamedTemporaryFile(
        suffix='.csv', delete=False
    ) as tmp_file:
        tmp_csv = tmp_file.name

    print(f"[1/3] 连接 DuckDB ({db_path}) 并执行查询...")
    try:
        con = duckdb.connect(db_path, read_only=True)
        # 使用 DuckDB 原生 COPY 命令快速导出 CSV
        export_sql = (
            f"COPY ({sql_query}) TO '{tmp_csv}' (HEADER, DELIMITER ',');"
        )
        con.execute(export_sql)
        con.close()
        print(f"      数据已导出至临时文件: {tmp_csv}")
    except Exception as e:
        print(f"DuckDB 查询/导出失败: {e}")
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        sys.exit(1)

    # 2. 组装并运行 STILTS 命令
    # 将 {input_file} 替换为生成的临时文件，{output_file} 替换为最终输出
    full_stilts_cmd = stilts_cmd_template.format(
        input_file=tmp_csv, output_file=output_path
    )

    print(f"[2/3] 执行 STILTS 指令:\n      {full_stilts_cmd}")
    try:
        result = subprocess.run(
            full_stilts_cmd, shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"STILTS 执行报错:\n{result.stderr}")
            sys.exit(1)
    finally:
        # 清理临时文件
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

    print(f"[3/3] 执行成功！输出结果已保存至: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STILTS & DuckDB 数据管道驱动'
    )
    parser.add_argument(
        '--db',
        default='astrodb_internal-main.db',
        help='DuckDB 数据库文件路径',
    )
    parser.add_argument('--sql', required=True, help='DuckDB SQL 查询语句')
    parser.add_argument(
        '--cmd', required=True, help='STILTS 命令行模板（包含占位符）'
    )
    parser.add_argument(
        '--out', default='stilts_output.fits', help='输出文件路径'
    )

    args = parser.parse_args()
    run_pipeline(args.db, args.sql, args.cmd, args.out)