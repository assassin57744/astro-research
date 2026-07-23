"""Query M45/M41/M67 tail_only stars"""
import duckdb
import numpy as np

con = duckdb.connect("research/hunt24-audit/data/warehouse/astrodb_internal.db", read_only=True)

for cluster in ["m45", "m41", "m67"]:
    tables = con.execute(
        f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%master_{cluster}%5d_h%' AND table_name NOT LIKE '%audit%' AND table_name NOT LIKE '%report%'"
    ).fetchall()
    
    for t in tables:
        tn = t[0]
        cols = [c[0] for c in con.execute(f"DESCRIBE {tn}").fetchall()]
        
        if "source" not in cols or "tail_prob" not in cols:
            continue
            
        has_core_prob = "core_prob" in cols
        has_tail_prob = "tail_prob" in cols
        
        # Get counts
        total = con.execute(f"SELECT COUNT(*) FROM {tn}").fetchone()[0]
        
        r = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE source = 'core' AND prob > 0.5),
                COUNT(*) FILTER (WHERE source = 'tail' AND prob > 0.5),
                COUNT(*) FILTER (WHERE source = 'both' AND prob > 0.5),
                COUNT(*) FILTER (WHERE prob > 0.5)
            FROM {tn}
        """).fetchone()
        
        print(f"\n{'='*60}")
        print(f"Table: {tn}")
        print(f"Total in table: {total}")
        print(f"  core_only(>0.5)={r[0]} | tail_only(>0.5)={r[1]} | both(>0.5)={r[2]} | total(>0.5)={r[3]}")
        
        # Get tail_only stars with spatial data
        if r[1] > 0:
            df = con.execute(f"""
                SELECT m.id, m.prob, m.tail_prob, m.core_prob,
                       a.ra, a.dec, a.mag
                FROM {tn} m
                LEFT JOIN aln_{cluster}_field a ON m.id = a.id
                WHERE m.source = 'tail' AND m.prob > 0.2
                ORDER BY m.prob DESC
            """).df()
            
            print(f"\n  tail_only (prob>0.2) stars: {len(df)}")
            
            if len(df) > 0:
                ra_c = {"m45": 56.61, "m44": 130.1, "m41": 101.5, "m67": 132.85}[cluster]
                dec_c = {"m45": 24.09, "m44": 19.7, "m41": -20.71, "m67": 11.83}[cluster]
                
                sep = np.sqrt((df["ra"] - ra_c)**2 + (df["dec"] - dec_c)**2)
                
                print(f"    RA: [{df['ra'].min():.2f}, {df['ra'].max():.2f}]  σ={df['ra'].std():.2f}")
                print(f"    Dec: [{df['dec'].min():.2f}, {df['dec'].max():.2f}]  σ={df['dec'].std():.2f}")
                print(f"    Mag: [{df['mag'].min():.1f}, {df['mag'].max():.1f}]")
                print(f"    Sep from center: mean={sep.mean():.2f}° median={sep.median():.2f}° max={sep.max():.2f}°")
                print(f"    Prob range: [{df['prob'].min():.4f}, {df['prob'].max():.4f}]")

con.close()
