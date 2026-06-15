import pandas as pd
from astropy.table import Table

def votable_to_parquet(vot_path, parquet_path):
    print(f"正在读取 VOTable 文件: {vot_path} ...")
    # 1. 使用 astropy 读取天文学标准的 .vot 文件
    astropy_table = Table.read(vot_path, format='votable')
    
    # 2. 转换为 Pandas DataFrame
    # astropy 会自动处理 Masked Array，将天文学中的无效值转换为 NaN
    df = astropy_table.to_pandas()
    
    # 3. 关键修正：处理字节串 (bytes) 编码
    # 部分天文原始数据（如 Gaia source_id 后的某些标记字符串）在 astropy 中读出为 bytes 类型
    # 如果不转成 str，pyarrow 写入 parquet 时会报错
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # 尝试将字节串解码为标准字符串
                df[col] = df[col].str.decode('utf-8')
            except AttributeError:
                # 如果本身就是字符串或普通对象，跳过
                pass
                
    # 4. 写入高性能的 .parquet 文件
    print(f"正在写入 Parquet 文件: {parquet_path} ...")
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    print("转换成功！")

if __name__ == "__main__":
    # 示例调用（换成你的实际路径）
    vot_path = "D:/git/repo/Alapha/cluster_audit/data/raw/vizier/zerj23.vot"
    parquet_path = "D:/git/repo/Alapha/cluster_audit/data/raw/vizier/processed/zerj23.parquet"
    votable_to_parquet(vot_path, parquet_path)
    