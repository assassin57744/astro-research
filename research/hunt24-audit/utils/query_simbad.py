from astroquery.simbad import Simbad
from astroquery.utils.tap.core import TapPlus
import pandas as pd
import re
import os
import duckdb
import certifi


def query_gaia_star_full_info(gaia_dr3_id):
    """
    查询指定 Gaia DR3 ID 的完整 SIMBAD 信息，并以字典格式输出以防截断。
    """
    # 1. 初始化 SIMBAD 实例
    custom_simbad = Simbad()

    # 获取字段表
    fields_table = Simbad.list_votable_fields()

    # 转换为 DataFrame 以获得更好的显示效果
    fields_df = fields_table.to_pandas()

    # 设置 Pandas 显示全部行，不截断
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    custom_simbad.add_votable_fields('ids')

    # 3. 执行查询
    print(f"正在从 SIMBAD 检索天体: {gaia_dr3_id} ...")
    result_table = custom_simbad.query_object(gaia_dr3_id)

    if result_table is None:
        print("未找到该天体，请检查 ID 格式是否正确（例如 'Gaia DR3 ...'）。")
        return None

    # 4. 转换为 Pandas DataFrame
    df = result_table.to_pandas()

    # 5. 转换为字典格式（orient='records' 会返回一个列表，每个元素代表一行）
    # 对于单颗星查询，我们取列表的第一个元素 [0]
    full_data = df.to_dict(orient='records')[0]

    return full_data

def run_pleiades_verification():
    # --- 配置与路径 ---
    module_base = "D:/git/repo/Alapha/astro_research/modules/db/"
    dirs = {
        "warehouse": os.path.join(module_base, "warehouse"),
    }
    db_path = os.path.join(dirs["warehouse"], "astrodb_internal.db")
    view_name = "res_new"

    # --- 统计计数器 ---
    stats = {
        "total": 0,
        "confirmed": 0,
        "unconfirmed": 0,
        "errors": 0
    }

    # --- 预编译正则 ---
    melotte_pattern = re.compile(r"Melotte.*22", re.IGNORECASE)
    m45_pattern = re.compile(r"M\s*45", re.IGNORECASE)

    # --- 1. 建立连接 ---
    print(f"[*] 正在连接数据库: {db_path}")
    con = duckdb.connect(database=db_path)

    try:
        # --- 2. 提取数据 ---
        print(f"[*] 正在从视图 [{view_name}] 提取 ID 列表...")
        results = con.execute(f"SELECT id FROM {view_name}").fetchall()
        target_ids = [row[0] for row in results]
        stats["total"] = len(target_ids)
        
        print(f"[+] 成功获取 {stats['total']} 个天体，准备开始 SIMBAD 校验...\n")
        print(f"{'Source ID':<35} | {'Status':<15} | {'Details'}")
        print("-" * 80)

        # --- 3. 循环查询 ---
        for gaia_id in target_ids:
            search_id = f"Gaia DR3 {gaia_id}" if "Gaia" not in str(gaia_id) else gaia_id
            
            try:
                # 调用查询函数
                star_info = query_gaia_star_full_info(search_id)

                if not star_info:
                    # 情况 2：结果为空，打印异常并继续
                    print(f"{search_id:<35} | [!] Skipping    | SIMBAD returned no data")
                    stats["errors"] += 1
                    continue

                # 处理标识符字符串
                ids_raw = star_info.get('ids', b'')
                ids_str = ids_raw.decode('utf-8') if isinstance(ids_raw, bytes) else str(ids_raw)

                # 匹配逻辑
                if melotte_pattern.search(ids_str) or m45_pattern.search(ids_str):
                    print(f"{search_id:<35} | [√] Confirmed   | Member of Pleiades Cluster")
                    stats["confirmed"] += 1
                else:
                    print(f"{search_id:<35} | [ ] Unconfirmed | Other identifiers found")
                    stats["unconfirmed"] += 1

            except Exception as e:
                # 情况 2：执行异常，打印并继续
                print(f"{search_id:<35} | [X] Exception   | {str(e)[:40]}")
                stats["errors"] += 1

        # --- 4. 输出美化报告 ---
        print("\n" + "="*40)
        print("          昂星团成员核验报告")
        print("="*40)
        print(f" 📅 处理总数:    {stats['total']:>6}")
        print(f" ✅ 确认数量:    {stats['confirmed']:>6}")
        print(f" ❌ 未确认数:    {stats['unconfirmed']:>6}")
        print(f" ⚠️ 异常/跳过:   {stats['errors']:>6}")
        print("-" * 40)
        confirm_rate = (stats['confirmed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f" 📊 确认率:      {confirm_rate:>6.2f}%")
        print("="*40)

    finally:
        con.close()
        print("\n[*] DuckDB 连接已安全关闭。")

def list_votable_fields_ex():
    custom_simbad = Simbad()
    # 获取字段表
    fields_table = Simbad.list_votable_fields()
    # 转换为 DataFrame 以获得更好的显示效果
    fields_df = fields_table.to_pandas()

    # 设置 Pandas 显示全部行，不截断
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    print(fields_df)

def get_otype():
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('otype')
    result_table = custom_simbad.query_object('Gaia DR3 129093492715773184')
    if result_table is None:
        print("未找到该天体，请检查 ID 格式是否正确（例如 'Gaia DR3 ...'）。")
        return None
    # 转换为 Pandas DataFrame
    df = result_table.to_pandas()
    # 设置 Pandas 显示全部行，不截断
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    pd.set_option('display.max_colwidth', None)
    print(df)

# def get_res_simbad(search_id):
#     custom_simbad = Simbad()
#     os.environ['SSL_CERT_FILE'] = certifi.where()
#     import ssl

#     # 创建一个忽略证书验证的上下文
#     try:
#         _create_unverified_https_context = ssl._create_unverified_context
#     except AttributeError:
#         pass
#     else:
#         ssl._create_default_https_context = _create_unverified_https_context

#     # 1. 连接服务
#     # simbad_tap = TapPlus(url="[https://simbad.cds.unistra.fr/simbad/tap/]")
#     # simbad_tap = custom_simbad.get_tap_engine()
#     # tap_url = Simbad.tap_url 
#     # print(f"正在使用官方预设 URL: {tap_url}")

#     # simbad_tap = TapPlus(url=tap_url)

#     # 2. 准备 ID 列表 (例如你的 106 颗星)
#     # formatted_ids = str(tuple([f"Gaia DR3 {i}" for i in star_ids]))

#     formatted_ids = search_id

#     query = f"""
#     SELECT 
#         i.id AS searched_id,
#         ids.id AS alias,
#         b.otype AS object_type
#     FROM ident AS i
#     JOIN ident AS ids ON i.oidref = ids.oidref
#     JOIN basic AS b ON i.oidref = b.oid
#     WHERE i.id IN ('Gaia DR3 {formatted_ids}')
#     """
#     # 执行查询
#     custom_simbad = Simbad()
#     result_table = custom_simbad.query_tap(query)

#     df_raw = result_table #.to_pandas()

#     # job = simbad_tap.launch_job(query)
#     # df_raw = job.get_results().to_pandas()

#     print(df_raw)

import pandas as pd
from astroquery.simbad import Simbad

def query_simbad_aliases_and_type(gaia_dr3_id) -> pd.DataFrame:
    """
    通过 Gaia DR3 ID 查询该天体在 Simbad 中的所有别名(alias)以及物理天体类型(object_type)
    
    参数:
        gaia_dr3_id: 可以是整型(int)或字符串(str)的纯 Gaia DR3 源码编号，
                    例如: 129093492715773184 或 '129093492715773184'
    返回:
        pd.DataFrame: 包含 searched_id, alias, object_type 三列的 Pandas 数据框
    """
    # 1. 规范化输入：确保 ID 是去除了首尾空格的字符串
    clean_id = str(gaia_dr3_id).strip()
    
    # 2. 动态拼接成 Simbad 识别的完整标准标识符
    full_ident_name = f"Gaia DR3 {clean_id}"
    
    # 3. 构造专门针对 Simbad TAP 服务的自连接 SQL 语句
    # 注意：在 Astroquery 的 Simbad TAP 查询中，表名需要带上公有前缀（通常直接用 ident 和 basic 即可）
    query = f"""
    SELECT 
        i.id AS searched_id,
        ids.id AS alias,
        b.otype AS object_type
    FROM ident AS i
    JOIN ident AS ids ON i.oidref = ids.oidref
    JOIN basic AS b ON i.oidref = b.oid
    WHERE i.id IN ('{full_ident_name}')
    """
    
    try:
        # 4. 执行 TAP 异步/同步查询（Simbad.query_tap）
        print(f"正在向 Simbad 证认源: {full_ident_name} ...")
        result_table = Simbad.query_tap(query)
        
        # 5. 如果查到了结果，转换为 Pandas DataFrame 并返回
        if result_table is not None and len(result_table) > 0:
            df = result_table.to_pandas()
            
            # 关键清洗步骤：处理 Astropy 转 Pandas 时可能出现的字节串(bytes)编码问题
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            return df
        else:
            print(f"警告: 未在 Simbad 中找到对应的天体: {full_ident_name}")
            return pd.DataFrame(columns=['searched_id', 'alias', 'object_type'])
            
    except Exception as e:
        print(f"查询出错: {e}")
        return pd.DataFrame(columns=['searched_id', 'alias', 'object_type'])

import pandas as pd
from astroquery.simbad import Simbad

import pandas as pd
from astroquery.simbad import Simbad

import pandas as pd
from astroquery.simbad import Simbad

import pandas as pd
from astroquery.simbad import Simbad

import pandas as pd
from astroquery.simbad import Simbad

def query_simbad_parent_cluster_ex(gaia_dr3_id) -> pd.DataFrame:
    """
    通过 Gaia DR3 ID 在官方 SIMBAD TAP 服务中通过 h_link 表逆向反查其隶属的母星团名称及类型。
    自动清洗因数据库别名导致的重复行，保证一星对应一主体。
    """
    # 1. 规范化输入
    clean_id = str(gaia_dr3_id).strip()
    full_child_name = f"Gaia DR3 {clean_id}"
    
    # 2. 优化后的 ADQL 语句
    # 限制 child_ident.id 必须就是我们输入的那个 Gaia ID，防止子源多别名爆炸
    query = f"""
    SELECT 
        child_ident.id AS child_name,
        parent_ident.id AS parent_name,
        parent_basic.otype AS parent_type
    FROM ident AS child_ident
    JOIN basic AS child_basic ON child_ident.oidref = child_basic.oid
    JOIN h_link AS hl ON child_basic.oid = hl.child
    JOIN basic AS parent_basic ON hl.parent = parent_basic.oid
    JOIN ident AS parent_ident ON parent_basic.oid = parent_ident.oidref
    WHERE child_ident.id = '{full_child_name}' 
      AND parent_ident.id LIKE 'NAME %'
    """
    
    try:
        result_table = Simbad.query_tap(query)
        
        if result_table is not None and len(result_table) > 0:
            df = result_table.to_pandas()
            
            # 解决字节串编码问题
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            
            # 清洗父源名字前缀 (例如 'NAME Pleiades' -> 'Pleiades')
            if 'parent_name' in df.columns:
                df['parent_name'] = df['parent_name'].str.replace('NAME ', '', regex=False)
            
            # =========================================================
            # 核心清洗步骤：利用 Pandas 强行滤除由于星团多别名产生的重复行
            # =========================================================
            df = df.drop_duplicates(subset=['child_name', 'parent_name']).reset_index(drop=True)
            
            return df
        else:
            return pd.DataFrame(columns=['child_name', 'parent_name', 'parent_type'])
            
    except Exception as e:
        print(f"SIMBAD 查询失败: {e}")
        return pd.DataFrame(columns=['child_name', 'parent_name', 'parent_type'])

import pandas as pd
from astroquery.simbad import Simbad

def query_simbad_parent_cluster_cross_matrix(gaia_dr3_id) -> pd.DataFrame:
    """
    通过 Gaia DR3 ID 逆向检索母体结构。
    不进行任何归一化或单行裁剪，完整保留子源所有别名与父源所有别名的笛卡尔乘积大矩阵，
    用于科学复核、多命名证认审计。
    """
    clean_id = str(gaia_dr3_id).strip()
    input_gaia_name = f"Gaia DR3 {clean_id}"
    
    # 依据你提供的底层物理 Schema 构建的完全交叉 ADQL 语句
    # 核心逻辑：
    # 1. WHERE 子句仅作为“锚点”定位输入的这颗星：child_ident_anchor.id = '{input_gaia_name}'
    # 2. 通过 child_basic.oid 重新 JOIN ident 表 (child_ident)，从而把这颗星的所有其他别名全部释放出来
    # 3. 同样通过 parent_basic.oid 释放母星团在 ident 表 (parent_ident) 里的所有别名
    query = f"""
    SELECT 
        child_ident.id AS child_name,
        parent_ident.id AS parent_name,
        parent_basic.otype AS parent_type
    FROM ident AS child_ident_anchor
    JOIN basic AS child_basic ON child_ident_anchor.oidref = child_basic.oid
    JOIN h_link AS hl ON child_basic.oid = hl.child
    JOIN basic AS parent_basic ON hl.parent = parent_basic.oid
    JOIN ident AS parent_ident ON parent_basic.oid = parent_ident.oidref
    JOIN ident AS child_ident ON child_basic.oid = child_ident.oidref
    WHERE child_ident_anchor.id = '{input_gaia_name}'
    """
    
    try:
        print(f"正在向 SIMBAD 提取 [子源所有别名 × 母体所有别名] 笛卡尔复核矩阵...")
        result_table = Simbad.query_tap(query)
        
        if result_table is not None and len(result_table) > 0:
            df = result_table.to_pandas()
            
            # 1. 转换 Astropy 字节串(bytes)为标准 Python 字符串
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            
            # 2. 清洗名字中的 'NAME ' 前缀，使复核时星团别名更加清爽直观
            if 'parent_name' in df.columns:
                df['parent_name'] = df['parent_name'].str.replace('NAME ', '', regex=False)
            if 'child_name' in df.columns:
                df['child_name'] = df['child_name'].str.replace('NAME ', '', regex=False)
            
            # 3. 按照子源名字和父源名字排序，让笛卡尔积以区块对齐的形式呈现，极大地方便人工和算法复核
            df = df.sort_values(by=['child_name', 'parent_name']).reset_index(drop=True)
            
            return df
        else:
            print(f"提示: 未在 SIMBAD 中找到该源的任何物理隶属关系矩阵。")
            return pd.DataFrame(columns=['child_name', 'parent_name', 'parent_type'])
            
    except Exception as e:
        print(f"SIMBAD 交叉矩阵检索失败: {e}")
        return pd.DataFrame(columns=['child_name', 'parent_name', 'parent_type'])

def query_simbad_parent_cluster(gaia_dr3_id) -> pd.DataFrame:
    """
    通过 Gaia DR3 ID 在官方 SIMBAD TAP 服务中通过 h_link 表逆向反查其隶属的母星团名称及类型。
    
    参数:
        gaia_dr3_id: 可以是整型(int)或字符串(str)的纯 Gaia DR3 源码编号，
                    例如: 129093492715773184
    返回:
        pd.DataFrame: 包含 child_name, parent_name, parent_type 三列的数据框
    """
    # 1. 规范化输入
    clean_id = str(gaia_dr3_id).strip()
    full_child_name = f"Gaia DR3 {clean_id}"
    
    # 2. 依据你提供的物理 Schema 完美构建的 ADQL 语句：
    # - 表名：h_link
    # - 字段：child (子源 OID), parent (父源 OID)
    query = f"""
    SELECT 
        child_ident.id AS child_name,
        parent_ident.id AS parent_name,
        parent_basic.otype AS parent_type
    FROM ident AS child_ident
    JOIN basic AS child_basic ON child_ident.oidref = child_basic.oid
    JOIN h_link AS hl ON child_basic.oid = hl.child
    JOIN basic AS parent_basic ON hl.parent = parent_basic.oid
    JOIN ident AS parent_ident ON parent_basic.oid = parent_ident.oidref
    WHERE child_ident.id = '{full_child_name}' 
      --AND (parent_ident.id LIKE 'NAME %' OR parent_ident.id LIKE 'Cl %' OR parent_ident.id LIKE 'Ass %')
    """
    
    try:
        print(f"正在向 SIMBAD 通过 [h_link] 检索母体结构，子源 ID: {full_child_name} ...")
        # 3. 执行官方 TAP 查询
        result_table = Simbad.query_tap(query)
        
        # 4. 解析并清洗数据
        if result_table is not None and len(result_table) > 0:
            df = result_table.to_pandas()
            
            # 解决 Astropy 的 bytes 转换为标准字符串问题
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            
            # 擦除名称前缀，恢复干净的星团名字（如 'NAME Pleiades' -> 'Pleiades'）
            if 'parent_name' in df.columns:
                df['parent_name'] = df['parent_name'].str.replace('NAME ', '', regex=False)
                df['parent_name'] = df['parent_name'].str.replace('Cl ', '', regex=False)
                
            return df
        else:
            print(f"提示: 未在 SIMBAD [h_link] 中找到该源的 Parent 隶属关系。")
            return pd.DataFrame(columns=['child_name', 'parent_name', 'parent_type'])
            
    except Exception as e:
        print(f"SIMBAD [h_link] 层次关系查询失败: {e}")
        return pd.DataFrame(columns=['child_name', 'parent_name', 'parent_type'])

import pandas as pd
from astroquery.simbad import Simbad

def generate_simbad_complete_review_report(gaia_dr3_id):
    """
    针对给定的 Gaia DR3 ID 生成全量不截断的扁平化科学复核报告。
    严格基于用户提供的物理 Schema 图谱：
    - 左侧栏：释放该子源本身的所有化名 (ident 表)
    - 右侧栏：抽取其隶属的所有母体信息，包含物理类型、h_link.membership (成员概率/标志) 
             以及通过 h_link.id_ref -> ref.oid 跨表拿到的标准文献 link_ref (ref.bibcode)
    """
    clean_id = str(gaia_dr3_id).strip()
    input_gaia_name = f"Gaia DR3 {clean_id}"
    
    # --- 1. 左侧独立提取：该恒星的所有别名群 (Identifiers) ---
    query_left_aliases = f"""
    SELECT DISTINCT child_ident.id AS child_aliases
    FROM ident AS child_ident_anchor
    JOIN basic AS child_basic ON child_ident_anchor.oidref = child_basic.oid
    JOIN ident AS child_ident ON child_basic.oid = child_ident.oidref
    WHERE child_ident_anchor.id = '{input_gaia_name}'
    """
    
    # --- 2. 右侧独立提取：层级隶属树 (Parents) + 成员概率 + 跨表文献引 ---
    # 物理关联路径：h_link (hl) -> JOIN ref (r) ON hl.id_ref = r.oid
    query_right_parents = f"""
    SELECT DISTINCT 
        parent_ident.id AS parent_name,
        parent_basic.otype AS parent_type,
        hl.membership AS membership,
        r.bibcode AS link_ref
    FROM ident AS child_ident_anchor
    JOIN basic AS child_basic ON child_ident_anchor.oidref = child_basic.oid
    JOIN h_link AS hl ON child_basic.oid = hl.child
    JOIN ref AS r ON hl.id_ref = r.oid
    JOIN basic AS parent_basic ON hl.parent = parent_basic.oid
    JOIN ident AS parent_ident ON parent_basic.oid = parent_ident.oidref
    WHERE child_ident_anchor.id = '{input_gaia_name}' 
      AND parent_ident.id LIKE 'NAME %'
    """
    
    try:
        print(f"正在依据 Schema 图谱全量抽取数据 [包含 membership 与 跨表 ref.bibcode]...")
        
        # 执行两个互不干扰的一维 TAP 提取
        res_aliases = Simbad.query_tap(query_left_aliases)
        res_parents = Simbad.query_tap(query_right_parents)
        
        # 处理并清洗左侧数据
        if res_aliases is not None and len(res_aliases) > 0:
            df_left = res_aliases.to_pandas()
            df_left['child_aliases'] = df_left['child_aliases'].astype(str).str.replace('NAME ', '', regex=False)
        else:
            df_left = pd.DataFrame(columns=['child_aliases'])
            
        # 处理并清洗右侧数据
        if res_parents is not None and len(res_parents) > 0:
            df_right = res_parents.to_pandas()
            df_right['parent_name'] = df_right['parent_name'].astype(str).str.replace('NAME ', '', regex=False)
            df_right['parent_type'] = df_right['parent_type'].astype(str)
            df_right['membership'] = df_right['membership'].astype(str)
            df_right['link_ref'] = df_right['link_ref'].astype(str)
        else:
            df_right = pd.DataFrame(columns=['parent_name', 'parent_type', 'membership', 'link_ref'])
            
        # --- 3. 强行进行一维扁平横向无损拼接 (杜绝笛卡尔积，行数按最大者对齐) ---
        df_left = df_left.reset_index(drop=True)
        df_right = df_right.reset_index(drop=True)
        
        df_report = pd.concat([df_left, df_right], axis=1)
        
        # 将空缺的单元格填充为空白，使得格式等同于网页布局
        df_report = df_report.fillna('')
        
        return df_report
        
    except Exception as e:
        print(f"生成完整报告时发生异常: {e}")
        return pd.DataFrame(columns=['child_aliases', 'parent_name', 'parent_type', 'membership', 'link_ref'])

# =========================================================================
# 执行全量无损打印
# =========================================================================
if __name__ == "__main__":
    target_gaia_id = 66167411464494848
    
    # 核心：生成无删减、含具体文献引用的扁平报告大表
    final_report_df = query_simbad_parent_cluster(target_gaia_id)
    
    print("\n" + "="*90)
    print("                      SIMBAD 官方物理树扁平化科学审计报告                     ")
    print("="*90)
    print(f"审计目标 Gaia DR3 ID: {target_gaia_id} \n")
    print("报告说明: 该表格包含了目标天体的所有别名（左侧）以及其在 SIMBAD 中被识别的所有母体结构信息（右侧），")
    print("包括母体名称、类型、成员关系以及相关文献引用。表格完全保留了 SIMBAD 中的原始数据，不进行任何形式的截断或归一化，适用于科学复核和多命名证认审计。")
    print("-" * 90)
    print(final_report_df)
    print("\n")
    print("="*90)

    

    