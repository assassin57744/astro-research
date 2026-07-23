import duckdb

dbs = [
    "research/hunt24-audit/data/warehouse/astro_research.db",
    "research/hunt24-audit/data/warehouse/astrodb_internal.db",
]

for db_path in dbs:
    try:
        con = duckdb.connect(db_path)
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%m44%'"
        ).fetchall()
        if tables:
            print(f"\n=== {db_path} ===")
            for t in tables:
                tn = t[0]
                print(f"\nTable: {tn}")
                cols = [c[0] for c in con.execute(f"DESCRIBE {tn}").fetchall()]

                if "source" in cols and "prob" in cols:
                    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
                        sql = f"""
                        SELECT
                            COUNT(*) FILTER (WHERE source = 'core' AND prob > {thresh}) as core_only,
                            COUNT(*) FILTER (WHERE source = 'tail' AND prob > {thresh}) as tail_only,
                            COUNT(*) FILTER (WHERE source = 'both' AND prob > {thresh}) as both,
                            COUNT(*) FILTER (WHERE prob > {thresh}) as total
                        FROM {tn}
                        """
                        r = con.execute(sql).fetchone()
                        print(f"  prob > {thresh:.2f}:  core_only={r[0]:>5} | tail_only={r[1]:>5} | both={r[2]:>5} | total={r[3]:>5}")

                    if "tail_prob" in cols:
                        r = con.execute(f"""
                            SELECT
                                COUNT(*) FILTER (WHERE tail_prob > 0.0 AND tail_prob <= 0.2),
                                COUNT(*) FILTER (WHERE tail_prob > 0.2 AND tail_prob <= 0.5),
                                COUNT(*) FILTER (WHERE tail_prob > 0.5 AND tail_prob <= 0.7),
                                COUNT(*) FILTER (WHERE tail_prob > 0.7 AND tail_prob <= 0.9),
                                COUNT(*) FILTER (WHERE tail_prob > 0.9)
                            FROM {tn} WHERE tail_prob > 0
                        """).fetchone()
                        print(f"  tail_prob bins: 0-0.2={r[0]}  0.2-0.5={r[1]}  0.5-0.7={r[2]}  0.7-0.9={r[3]}  0.9-1.0={r[4]}")
                else:
                    print(f"  Columns: {cols}")
        con.close()
    except Exception as e:
        print(f"{db_path}: {e}")

