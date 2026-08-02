# tools/run_stilts.py
import sys
import subprocess
import pandas as pd
import io
from pathlib import Path

STILTS_CMD = str(Path(__file__).parent / "stilts" / "stilts.bat")

def run_tpipe(input_file, filter_expr, keep_cols, output_file=None):
    command = [
        STILTS_CMD, "tpipe",
        f"in={input_file}",
        f"cmd=select '{filter_expr}'",
        f"cmd=keepcols '{keep_cols}'",
        "ofmt=csv"
    ]
    try:
        res = subprocess.run(command, capture_output=True, check=True)
        df = pd.read_csv(io.BytesIO(res.stdout))
        print(f"✅ 处理完成，筛选出 {len(df)} 行数据。\n")
        print("📊 数据摘要:")
        print(df.describe().to_string())
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n💾 数据已写入: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ STILTS 执行失败: {e.stderr.decode('utf-8')}")

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        out = sys.argv[4] if len(sys.argv) > 4 else None
        run_tpipe(sys.argv[1], sys.argv[2], sys.argv[3], out)