"""Query M45 tail_only stars spatial distribution"""
import duckdb
import numpy as np

for db_path in [
    "research/hunt24-audit/data/warehouse/astro_research.db",
    "research/hunt24-audit/data/warehouse/astrodb_internal.db",
]:
    con = duckdb.connect(db_path)
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%m45%'"
    ).fetchall()
    if tables:
        print(f"\n=== {db_path} ===")
        for t in tables:
            print(f"  {t[0]}")
    con.close()

# Then find the right one
con = duckdb.connect("research/hunt24-audit/data/warehouse/astrodb_internal.db")
master = None
tables = con.execute(
    "SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%master%m45%'"
).fetchall()
for t in tables:
    tn = t[0]
    cols = [c[0] for c in con.execute(f"DESCRIBE {tn}").fetchall()]
    if 'source' in cols:
        master = tn
        print(f"\nUsing: {master}")
        break

if not master:
    print("No master table found!")
    con.close()
    exit()

# Get tail_only count at various thresholds
print("\n=== tail_only at various prob thresholds ===")
for thresh in [0.2, 0.5, 0.8, 0.9, 0.99]:
    r = con.execute(f"""
        SELECT COUNT(*) FROM {master} 
        WHERE source = 'tail' AND prob > {thresh}
    """).fetchone()
    print(f"  prob > {thresh}: {r[0]}")

# Get tail_only stars with spatial data
print("\n=== tail_only (prob>0.2) ===")
df = con.execute(f"""
    SELECT m.id, m.prob, m.tail_prob, m.core_prob,
           a.ra, a.dec, a.pmra, a.pmdec, a.plx, a.mag
    FROM {master} m
    LEFT JOIN aln_m45_field a ON m.id = a.id
    WHERE m.source = 'tail' AND m.prob > 0.2
    ORDER BY m.prob DESC
""").df()

print(f"Total: {len(df)}")
if len(df) > 0:
    print(f"\nRA: [{df['ra'].min():.2f}, {df['ra'].max():.2f}]  std={df['ra'].std():.2f}")
    print(f"Dec: [{df['dec'].min():.2f}, {df['dec'].max():.2f}]  std={df['dec'].std():.2f}")
    print(f"\nTop 10 by prob:")
    pd_opts = {"max_columns": 20, "width": 150}
    print(df.head(10).to_string(index=False, **pd_opts))

    # Distance from M45 center (56.61, 24.09)
    ra_c, dec_c = 56.61, 24.09
    sep = np.sqrt((df['ra'] - ra_c)**2 + (df['dec'] - dec_c)**2)
    print(f"\nAngular separation from center:")
    print(f"  Mean: {sep.mean():.2f}°  Median: {sep.median():.2f}°  Max: {sep.max():.2f}°")
    print(f"  Within 1.3° (core): {(sep <= 1.3).sum()}")
    print(f"  1.3-5°: {((sep > 1.3) & (sep <= 5)).sum()}")
    print(f"  5-10°: {((sep > 5) & (sep <= 10)).sum()}")
    print(f"  >10°: {(sep > 10).sum()}")
    print(f"  >15°: {(sep > 15).sum()}")

    # Mag distribution
    print(f"\nMag: [{df['mag'].min():.1f}, {df['mag'].max():.1f}]")
    
    # Direction analysis - are they along a preferred axis?
    # Convert to position angle from center
    pa = np.degrees(np.arctan2(df['dec'] - dec_c, df['ra'] - ra_c))
    print(f"\nPosition angle (from center):")
    print(f"  NE quadrant (0-90°): {((pa >= 0) & (pa < 90)).sum()}")
    print(f"  SE quadrant (90-180°): {((pa >= 90) & (pa < 180)).sum()}")
    print(f"  SW quadrant (-180 to -90°): {((pa >= -180) & (pa < -90)).sum()}")
    print(f"  NW quadrant (-90 to 0°): {((pa >= -90) & (pa < 0)).sum()}")

con.close()

