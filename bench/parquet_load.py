#!/usr/bin/env python3
"""Parquet load benchmark: JouleDB `COPY t FROM 'x.parquet'` vs DuckDB.

DuckDB writes an N-row hits-like Parquet file into the server's
JOULEDB_COPY_DIR; then we time
  - JouleDB: CREATE TABLE + COPY FROM the file (pgwire), then COUNT/SUM check
  - DuckDB:  CREATE TABLE AS SELECT * FROM read_parquet(file) (in-memory)
and print one markdown table. Run with the server started as
  JOULEDB_COPY_DIR=<dir> jouledb -p <http> ...
"""
import argparse
import os
import statistics
import time

import duckdb
import psycopg

ap = argparse.ArgumentParser()
ap.add_argument("--pg-port", type=int, default=28082)
ap.add_argument("--rows", type=int, default=1_000_000)
ap.add_argument("--dir", required=True)
ap.add_argument("--repeat", type=int, default=3)
a = ap.parse_args()

path = os.path.join(a.dir, f"hits-{a.rows}.parquet")
duckdb.sql(f"""COPY (SELECT i AS id, 'u' || (i % 50000) AS user_id, 'http://e.com/p/' || (i % 9973) AS url,
  (i * 7919) % 1000 / 10.0 AS amount, i % 2 = 0 AS ok, DATE '2026-01-01' + (i % 365)::INT AS day
  FROM range({a.rows}) t(i)) TO '{path}' (FORMAT parquet)""")
size_mb = os.path.getsize(path) / 1e6
want = duckdb.sql(f"SELECT COUNT(*), SUM(amount)::DOUBLE FROM read_parquet('{path}')").fetchone()

c = psycopg.connect(f"host=127.0.0.1 port={a.pg_port} user=jouledb dbname=jouledb", autocommit=True)
jt, dt = [], []
for r in range(a.repeat):
    t = f"pq{r}"
    c.execute(f"CREATE TABLE {t} (id INT, user_id TEXT, url TEXT, amount FLOAT, ok BOOLEAN, day TEXT)")
    t0 = time.perf_counter()
    c.execute(f"COPY {t} FROM '{path}'")
    jt.append(time.perf_counter() - t0)
    got = c.execute(f"SELECT COUNT(*), SUM(amount) FROM {t}").fetchone()
    assert got[0] == want[0] and abs(got[1] - want[1]) < 1e-6 * max(1, abs(want[1])), (got, want)
    c.execute(f"DROP TABLE {t}")
    con = duckdb.connect()
    t0 = time.perf_counter()
    con.sql(f"CREATE TABLE t AS SELECT * FROM read_parquet('{path}')")
    dt.append(time.perf_counter() - t0)
    con.close()

j, d = statistics.median(jt), statistics.median(dt)
print(f"| rows | file MB | JouleDB COPY FROM parquet (s) | DuckDB CTAS read_parquet (s) | ratio | rows/s JouleDB |")
print(f"|---|---|---|---|---|---|")
print(f"| {a.rows:,} | {size_mb:.1f} | {j:.3f} | {d:.3f} | {j/d:.1f}x | {a.rows/j:,.0f} |")
print(f"check: COUNT/SUM match DuckDB ({want[0]}, {want[1]:.1f}); runs jouledb={['%.3f'%x for x in jt]} duckdb={['%.3f'%x for x in dt]}")
