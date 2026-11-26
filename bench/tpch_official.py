#!/usr/bin/env python3
"""
Official 22-Query TPC-H Relational Analytics Benchmark: JouleDB vs DuckDB
=========================================================================
Generates the standard 8-table TPC-H schema (`nation`, `region`, `part`,
`supplier`, `partsupp`, `customer`, `orders`, `lineitem`) via DuckDB's
official `tpch` extension (`CALL dbgen(sf=...)`), exports all 8 tables as
Apache Parquet, ingests them into JouleDB over real TCP PostgreSQL v3
wire protocol (`pgwire`, server-side `COPY ... FROM '...parquet'`), and
executes all **22 official TPC-H queries** (`SELECT query_nr, query FROM tpch_queries()`)
completely unmodified over TCP `pgwire` (`psycopg`) against DuckDB on the
identical Parquet / native dataset.

Methodology & Honesty Rules:
- Every single query Q01..Q22 is executed and reported unfiltered (wins,
  losses, timeouts, or unsupported SQL constructs).
- Numerical/row-count equivalence vs DuckDB is verified for every query.
- On Linux with readable `/sys/class/powercap/intel-rapl:0/energy_uj`,
  `energy_tag = "reported_j"` records real hardware RAPL package energy
  per query for both JouleDB (client + server) and DuckDB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import duckdb
import psycopg


RAPL_CANDIDATES = [
    Path("/sys/class/powercap/intel-rapl:0/energy_uj"),
    Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"),
]

TPCH_TABLES = [
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
]


def detect_rapl() -> Path | None:
    for p in RAPL_CANDIDATES:
        try:
            if p.exists() and os.access(p, os.R_OK):
                _ = int(p.read_text().strip())
                return p
        except Exception:
            continue
    return None


def read_rapl_uj(rapl_path: Path | None) -> int | None:
    if rapl_path is None:
        return None
    try:
        return int(rapl_path.read_text().strip())
    except Exception:
        return None


def rapl_delta_j(u0: int | None, u1: int | None) -> float | None:
    if u0 is None or u1 is None:
        return None
    d = u1 - u0
    if d < 0:
        d += 1 << 32
    return round(d / 1_000_000.0, 6)


def duck_type_to_sql(dtype: str) -> str:
    u = dtype.upper()
    if "INT" in u:
        return "BIGINT"
    if "DECIMAL" in u or "DOUBLE" in u or "FLOAT" in u or "NUMERIC" in u:
        return "FLOAT"
    if "DATE" in u:
        return "TEXT"
    return "TEXT"


def read_proc_mem_mb(pid: int) -> tuple[float | None, float | None]:
    if not sys.platform.startswith("linux"):
        return None, None
    rss_mb = None
    hwm_mb = None
    try:
        for ln in Path(f"/proc/{pid}/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                rss_mb = round(int(ln.split()[1]) / 1024.0, 2)
            elif ln.startswith("VmHWM:"):
                hwm_mb = round(int(ln.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    return rss_mb, hwm_mb


def wait_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for {host}:{port}")


def cells_close(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa = float(a)
        fb = float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            tol = max(1e-2, 1e-4 * max(abs(fa), abs(fb)))
            return abs(fa - fb) <= tol
    except (TypeError, ValueError):
        pass
    sa = str(a).strip()
    sb = str(b).strip()
    # Normalize ISO date prefix if timestamp rendered
    if len(sa) >= 10 and len(sb) == 10 and sa[:10] == sb:
        return True
    if len(sb) >= 10 and len(sa) == 10 and sb[:10] == sa:
        return True
    return sa == sb


def rows_match(duck_rows: list[tuple[Any, ...]], jdb_rows: list[tuple[Any, ...]]) -> tuple[bool, str]:
    if len(duck_rows) != len(jdb_rows):
        return False, f"row_count mismatch: duckdb={len(duck_rows)} jouledb={len(jdb_rows)}"
    for idx, (dr, jr) in enumerate(zip(duck_rows, jdb_rows)):
        if len(dr) != len(jr):
            return False, f"col_count mismatch at row {idx}: duckdb={len(dr)} jouledb={len(jr)}"
        for ci, (dv, jv) in enumerate(zip(dr, jr)):
            if not cells_close(dv, jv):
                return False, f"row {idx} col {ci} mismatch: duckdb={dv!r} jouledb={jv!r}"
    return True, "exact_or_float_tol_match"


def split_sql_statements(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Official 22-Query TPC-H Benchmark: JouleDB vs DuckDB")
    ap.add_argument("--jouledb-bin", required=True, help="Path to release jouledb binary")
    ap.add_argument("--sf", type=float, default=0.1, help="TPC-H scale factor (e.g. 0.1 or 1.0)")
    ap.add_argument("--runs", type=int, default=3, help="Timed runs per query (after 1 warmup if runs>1)")
    ap.add_argument("--query-timeout-s", type=float, default=60.0, help="Per-query timeout in seconds")
    ap.add_argument("--max-rss-gb", type=float, default=36.0, help="Per-query server RSS guard in GB (0=disabled)")
    ap.add_argument("--port", type=int, default=18780, help="Base JouleDB port (pgwire = port + 2)")
    ap.add_argument("--work-dir", default="", help="Optional persistent scratch directory")
    ap.add_argument("--out", required=True, help="Path to output JSON report")
    args = ap.parse_args()

    rapl_path = detect_rapl()
    energy_tag = "reported_j" if rapl_path is not None else "unavailable"

    cleanup_dir = False
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"tpch_sf{args.sf}_")).resolve()
        cleanup_dir = True

    parquet_dir = work_dir / "parquet"
    jdb_data_dir = work_dir / "jdb_data"
    if jdb_data_dir.exists():
        shutil.rmtree(jdb_data_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    jdb_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tpch] Generating official TPC-H SF={args.sf} dataset via DuckDB tpch extension...")
    dcon = duckdb.connect(":memory:")
    dcon.execute("INSTALL tpch; LOAD tpch;")
    t_gen0 = time.perf_counter()
    dcon.execute(f"CALL dbgen(sf={args.sf});")
    gen_s = time.perf_counter() - t_gen0

    schemas: dict[str, list[tuple[str, str]]] = {}
    table_counts: dict[str, int] = {}
    total_rows = 0
    for tbl in TPCH_TABLES:
        cols = dcon.execute(f"DESCRIBE {tbl}").fetchall()
        schemas[tbl] = [(c[0], duck_type_to_sql(c[1])) for c in cols]
        cnt = dcon.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        table_counts[tbl] = int(cnt)
        total_rows += int(cnt)
        pq_path = parquet_dir / f"{tbl}.parquet"
        if not pq_path.exists():
            dcon.execute(f"COPY {tbl} TO '{pq_path}' (FORMAT PARQUET);")

    official_queries: list[tuple[int, str]] = [
        (int(qnr), str(qsql))
        for qnr, qsql in dcon.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr").fetchall()
    ]
    print(f"[tpch] Generated {total_rows:,} total rows across 8 tables in {gen_s:.2f}s ({len(official_queries)} official queries).")

    # DuckDB benchmark over Parquet Views (matching cold/hot Parquet/table analytical setup)
    duck_db_path = work_dir / "duck_tpch.duckdb"
    if duck_db_path.exists():
        duck_db_path.unlink()
    d_bench = duckdb.connect(str(duck_db_path))
    u_di0 = read_rapl_uj(rapl_path)
    t_di0 = time.perf_counter()
    for tbl in TPCH_TABLES:
        pq_path = parquet_dir / f"{tbl}.parquet"
        d_bench.execute(f"CREATE TABLE {tbl} AS SELECT * FROM read_parquet('{pq_path}');")
    duck_ingest_s = round(time.perf_counter() - t_di0, 4)
    duck_ingest_j = rapl_delta_j(u_di0, read_rapl_uj(rapl_path))

    # Start JouleDB server
    env = os.environ.copy()
    env["JOULEDB_COPY_DIR"] = str(parquet_dir)
    env["JOULEDB_QUERY_TIMEOUT_MS"] = str(int(args.query_timeout_s * 1000))
    pg_port = args.port + 2
    def spawn_server() -> subprocess.Popen:
        return subprocess.Popen(
            [
                args.jouledb_bin,
                "--port",
                str(args.port),
                "--data",
                str(jdb_data_dir),
                "--query-timeout",
                str(int(args.query_timeout_s * 1000)),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    proc = spawn_server()

    class _QueryWallTimeout(Exception):
        pass

    def _alarm_handler(_signum: int, _frame: Any) -> None:
        raise _QueryWallTimeout(f"query exceeded wall timeout of {args.query_timeout_s}s")

    signal.signal(signal.SIGALRM, _alarm_handler)

    try:
        wait_port("127.0.0.1", pg_port, timeout_s=30.0)
        conn = psycopg.connect(
            f"host=127.0.0.1 port={pg_port} user=postgres dbname=default",
            autocommit=True,
        )

        # Ingest all 8 Parquet tables over pgwire
        u_ji0 = read_rapl_uj(rapl_path)
        t_ji0 = time.perf_counter()
        per_table_ingest_s: dict[str, float] = {}
        with conn.cursor() as cur:
            for tbl in TPCH_TABLES:
                col_ddl = ", ".join(f"{cname} {ctype}" for cname, ctype in schemas[tbl])
                cur.execute(f"CREATE TABLE {tbl} ({col_ddl});")
                t_t0 = time.perf_counter()
                cur.execute(f"COPY {tbl} FROM '{tbl}.parquet' (FORMAT parquet);")
                per_table_ingest_s[tbl] = round(time.perf_counter() - t_t0, 4)
        jdb_ingest_s = round(time.perf_counter() - t_ji0, 4)
        jdb_ingest_j = rapl_delta_j(u_ji0, read_rapl_uj(rapl_path))
        post_load_rss_mb, _ = read_proc_mem_mb(proc.pid)
        print(f"[tpch] Ingested {total_rows:,} rows into JouleDB (.jcol + Amorphic) in {jdb_ingest_s:.3f}s (DuckDB: {duck_ingest_s:.3f}s, post_load_rss_mb={post_load_rss_mb}).")

        query_reports: dict[str, Any] = {}
        passed = 0
        wins = 0
        jdb_medians: list[float] = []
        duck_medians: list[float] = []
        ratios: list[float] = []

        for qnr, qsql in official_queries:
            qid = f"Q{qnr:02d}"
            stmts = split_sql_statements(qsql)

            # DuckDB reference + timing
            duck_times: list[float] = []
            duck_joules: list[float] = []
            duck_rows: list[tuple[Any, ...]] = []
            total_iters = (1 + args.runs) if args.runs > 1 else 1
            for it in range(total_iters):
                u0 = read_rapl_uj(rapl_path)
                t0 = time.perf_counter()
                last_res: list[tuple[Any, ...]] = []
                for st in stmts:
                    rel = d_bench.execute(st)
                    if st.lstrip().upper().startswith(("SELECT", "WITH")):
                        last_res = rel.fetchall()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                dj = rapl_delta_j(u0, read_rapl_uj(rapl_path))
                if it == 0 and total_iters > 1:
                    duck_rows = last_res
                else:
                    if not duck_rows:
                        duck_rows = last_res
                    duck_times.append(dt_ms)
                    if dj is not None:
                        duck_joules.append(dj)

            duck_med = statistics.median(duck_times)
            duck_j_med = statistics.median(duck_joules) if duck_joules else None

            # JouleDB over TCP pgwire
            jdb_times: list[float] = []
            jdb_joules: list[float] = []
            jdb_rows: list[tuple[Any, ...]] = []
            jdb_err: str | None = None

            for it in range(total_iters):
                u0 = read_rapl_uj(rapl_path)
                t0 = time.perf_counter()
                rss_tripped: list[str] = []
                stop_watch = threading.Event()

                def _rss_watchdog(p: subprocess.Popen, limit_gb: float) -> None:
                    if limit_gb <= 0 or not sys.platform.startswith("linux"):
                        return
                    limit_kb = int(limit_gb * 1024 * 1024)
                    status_file = Path(f"/proc/{p.pid}/status")
                    while not stop_watch.wait(0.15):
                        try:
                            for ln in status_file.read_text().splitlines():
                                if ln.startswith("VmRSS:"):
                                    kb = int(ln.split()[1])
                                    if kb > limit_kb:
                                        rss_tripped.append(
                                            f"server VmRSS ({kb / 1048576.0:.1f} GB) exceeded --max-rss-gb={limit_gb:.1f} GB cap"
                                        )
                                        p.kill()
                                        return
                                    break
                        except Exception:
                            return

                wt = threading.Thread(target=_rss_watchdog, args=(proc, args.max_rss_gb), daemon=True)
                wt.start()
                try:
                    signal.setitimer(signal.ITIMER_REAL, args.query_timeout_s)
                    last_j: list[tuple[Any, ...]] = []
                    with conn.cursor() as cur:
                        for st in stmts:
                            cur.execute(st)
                            if cur.description is not None:
                                last_j = cur.fetchall()
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    stop_watch.set()
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    jj = rapl_delta_j(u0, read_rapl_uj(rapl_path))
                    if it == 0 and total_iters > 1:
                        jdb_rows = last_j
                    else:
                        if not jdb_rows:
                            jdb_rows = last_j
                        jdb_times.append(dt_ms)
                        if jj is not None:
                            jdb_joules.append(jj)
                except _QueryWallTimeout as exc:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    stop_watch.set()
                    jdb_err = rss_tripped[0] if rss_tripped else str(exc)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    proc.kill()
                    proc.wait(timeout=5)
                    proc = spawn_server()
                    wait_port("127.0.0.1", pg_port, timeout_s=30.0)
                    conn = psycopg.connect(
                        f"host=127.0.0.1 port={pg_port} user=postgres dbname=default",
                        autocommit=True,
                    )
                    break
                except Exception as exc:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    stop_watch.set()
                    jdb_err = rss_tripped[0] if rss_tripped else str(exc)
                    # Reconnect if connection broke or server was killed by RSS watchdog
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if rss_tripped:
                        proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            pass
                    if proc.poll() is not None:
                        proc = spawn_server()
                        wait_port("127.0.0.1", pg_port, timeout_s=30.0)
                    conn = psycopg.connect(
                        f"host=127.0.0.1 port={pg_port} user=postgres dbname=default",
                        autocommit=True,
                    )
                    break

            if jdb_err is not None or not jdb_times:
                print(f"  {qid}: ERROR ({jdb_err})", flush=True)
                query_reports[qid] = {
                    "query_nr": qnr,
                    "status": "error",
                    "error": jdb_err,
                    "duckdb_median_ms": round(duck_med, 3),
                }
                continue

            ok, match_note = rows_match(duck_rows, jdb_rows)
            jdb_med = statistics.median(jdb_times)
            jdb_j_med = statistics.median(jdb_joules) if jdb_joules else None
            ratio = jdb_med / max(duck_med, 0.001)

            if ok:
                passed += 1
                jdb_medians.append(jdb_med)
                duck_medians.append(duck_med)
                ratios.append(ratio)
                if jdb_med <= duck_med:
                    wins += 1

            status_str = "PASS" if ok else f"MISMATCH ({match_note})"
            print(
                f"  {qid}: {status_str} | JouleDB={jdb_med:8.2f} ms | DuckDB={duck_med:7.2f} ms | ratio={ratio:6.2f}x | rows={len(jdb_rows)}",
                flush=True,
            )
            query_reports[qid] = {
                "query_nr": qnr,
                "status": "ok" if ok else "mismatch",
                "verification": match_note,
                "row_count": len(jdb_rows),
                "jouledb_median_ms": round(jdb_med, 3),
                "duckdb_median_ms": round(duck_med, 3),
                "ratio_jdb_over_duckdb": round(ratio, 4),
                "jouledb_reported_j": round(jdb_j_med, 6) if jdb_j_med is not None else None,
                "duckdb_reported_j": round(duck_j_med, 6) if duck_j_med is not None else None,
                "energy_tag": energy_tag,
            }

        end_rss_mb, peak_hwm_mb = read_proc_mem_mb(proc.pid)
        geomean_ratio = (
            round(math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios)), 4)
            if ratios
            else None
        )
        report = {
            "benchmark": "TPC-H Official 22-Query Relational Analytics Suite",
            "scale_factor": args.sf,
            "total_rows": total_rows,
            "table_row_counts": table_counts,
            "transport": "TCP pgwire (psycopg3) vs DuckDB native Python C++ API",
            "duckdb_version": duckdb.__version__,
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "energy_tag": energy_tag,
            },
            "memory": {
                "server_post_load_rss_mb": post_load_rss_mb,
                "server_end_rss_mb": end_rss_mb,
                "server_peak_vmhwm_mb": peak_hwm_mb,
            },
            "ingest": {
                "jouledb_parquet_copy_s": jdb_ingest_s,
                "jouledb_per_table_s": per_table_ingest_s,
                "jouledb_reported_j": jdb_ingest_j,
                "duckdb_parquet_load_s": duck_ingest_s,
                "duckdb_reported_j": duck_ingest_j,
                "energy_tag": energy_tag,
            },
            "summary": {
                "queries_total": len(official_queries),
                "queries_verified_exact": passed,
                "jouledb_faster_or_equal_count": wins,
                "jouledb_total_ms": round(sum(jdb_medians), 3),
                "duckdb_total_ms": round(sum(duck_medians), 3),
                "geomean_jdb_over_duckdb": geomean_ratio,
                "median_jdb_over_duckdb": round(statistics.median(ratios), 4) if ratios else None,
            },
            "queries": query_reports,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"\n[tpch] Summary SF={args.sf}: {passed}/{len(official_queries)} verified exact | "
            f"wins={wins}/{len(official_queries)} | total JouleDB={sum(jdb_medians):.1f} ms vs DuckDB={sum(duck_medians):.1f} ms | "
            f"geomean={geomean_ratio}x -> {out_path}"
        )
        conn.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        d_bench.close()
        dcon.close()
        if cleanup_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
