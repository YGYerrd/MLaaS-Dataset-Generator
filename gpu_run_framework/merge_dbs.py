import sqlite3, glob, os
from pathlib import Path

# Destination DB
out_db = "federated_merged.db"
if os.path.exists(out_db):
    os.remove(out_db)

# Initialize the merged DB schema from the first shard
first_db = sorted(glob.glob("run_shard_*/federated.db"))[0]
with sqlite3.connect(out_db) as con_out, sqlite3.connect(first_db) as con_in:
    schema = "\n".join(r[0] for r in con_in.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"))
    con_out.executescript(schema)
    con_out.commit()

# Helper: check if run_id looks numeric
def is_numeric_runid(cur):
    cur.execute("SELECT run_id FROM runs LIMIT 1;")
    val = cur.fetchone()
    return val and str(val[0]).isdigit()

# Merge all shard DBs
run_id_offset = 0
for db_path in sorted(glob.glob("run_shard_*/federated.db")):
    print(f"Merging {db_path} ...")
    with sqlite3.connect(db_path) as src, sqlite3.connect(out_db) as dst:
        src.row_factory = sqlite3.Row
        src_cur = src.cursor()
        dst_cur = dst.cursor()

        numeric = is_numeric_runid(src_cur)

        # Fetch all runs
        runs = list(src_cur.execute("SELECT * FROM runs"))
        for r in runs:
            r = dict(r)
            if numeric:
                r["run_id"] = int(r["run_id"]) + run_id_offset
            cols = ",".join(r.keys())
            placeholders = ",".join(["?"] * len(r))
            dst_cur.execute(f"INSERT INTO runs ({cols}) VALUES ({placeholders})", tuple(r.values()))
        dst.commit()

        # Now merge dependent tables
        for table in ("rounds", "client_rounds"):
            rows = list(src_cur.execute(f"SELECT * FROM {table}"))
            for row in rows:
                row = dict(row)
                if numeric:
                    row["run_id"] = int(row["run_id"]) + run_id_offset
                cols = ",".join(row.keys())
                placeholders = ",".join(["?"] * len(row))
                dst_cur.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))
            dst.commit()

        if numeric:
            # update offset to prevent collisions
            max_run = max(int(r["run_id"]) for r in runs)
            run_id_offset += max_run
