#!/usr/bin/env python3
"""Commit durability cost: single-writer autocommit INSERTs/s, and concurrent
writers (group commit), against a running jouledb (run once with
JOULEDB_SYNC_COMMIT=on, once with off)."""
import argparse, threading, time
import psycopg

ap = argparse.ArgumentParser()
ap.add_argument("--pg-port", type=int, default=28082)
ap.add_argument("--seconds", type=float, default=5)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
dsn = f"host=127.0.0.1 port={a.pg_port} user=jouledb dbname=jouledb"
c = psycopg.connect(dsn, autocommit=True)
c.execute("DROP TABLE IF EXISTS synct")
c.execute("CREATE TABLE synct (w INT, k INT)")

def writer(w, out):
    cc = psycopg.connect(dsn, autocommit=True)
    n, end = 0, time.perf_counter() + a.seconds
    while time.perf_counter() < end:
        cc.execute("INSERT INTO synct VALUES (%s, %s)", (w, n)); n += 1
    out[w] = n

for nt in (1, a.threads):
    out = {}
    ts = [threading.Thread(target=writer, args=(w, out)) for w in range(nt)]
    t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]
    dt = time.perf_counter() - t0
    print(f"writers={nt} commits={sum(out.values())} commits/s={sum(out.values())/dt:.0f}")
