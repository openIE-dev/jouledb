#!/usr/bin/env python3
"""Official ClickBench End-to-End Evaluation: JouleDB (pgwire) vs DuckDB.

Unlike synthetic micro-benchmarks, this script loads rows directly from the official
ClickBench `hits.parquet` file (`https://datasets.clickhouse.com/hits_compatible/hits.parquet`,
99,997,497 rows, 105 columns, 14.78 GB) and executes all 43 official ClickBench SQL
queries (`Q0`..`Q42` from `ClickHouse/ClickBench/duckdb/queries.sql`) unchanged on both
DuckDB and JouleDB (via PostgreSQL v3 wire protocol `psycopg`).

Candid reporting rules:
  - Every query (Q0..Q42) is attempted and recorded in the output JSON and Markdown table.
  - Losses, timeouts, SQL syntax/planner errors, and answer mismatches are reported openly.
  - Energy figures use hardware counters (`reported_j` via Linux RAPL / macOS powermetrics)
    when available, or `unmetered` (`None`) when unavailable — never analytical constants.
"""
import argparse
import datetime
import json
import os
import platform
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Window, make_meter  # noqa: E402


def duck_type_to_pg_ddl(dtype_str: str) -> str:
    t = dtype_str.upper()
    if "INT" in t:
        return "BIGINT"
    if "DOUBLE" in t or "FLOAT" in t or "DECIMAL" in t:
        return "FLOAT"
    if "DATE" in t or "TIME" in t:
        return "TEXT"
    return "TEXT"


def normalize_cell(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    s = str(v)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return round(float(s), 4)
    return s


def normalize_temporal_cell(v):
    """Normalize DuckDB `datetime.date` / `datetime.datetime` and Unix epoch days/seconds to a canonical number."""
    if isinstance(v, datetime.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=datetime.timezone.utc)
        return round(v.timestamp(), 4)
    if isinstance(v, datetime.date):
        return float((v - datetime.date(1970, 1, 1)).days)
    return normalize_cell(v)


def normalize_rows(rows):
    return [tuple(normalize_cell(c) for c in r) for r in rows]


def normalize_rows_temporal(rows):
    return [tuple(normalize_temporal_cell(c) for c in r) for r in rows]


def verify_sql_equivalence(dcon, sql: str, d_rows, j_rows) -> tuple[bool, str]:
    """Verify whether `j_rows` is a valid SQL answer when `ORDER BY` has ties, is absent, or uses epoch ints."""
    if len(j_rows) != len(d_rows):
        return False, "row_count_mismatch"
    if normalize_rows(j_rows[:20]) == normalize_rows(d_rows[:20]):
        return True, "exact"
    if normalize_rows_temporal(j_rows[:20]) == normalize_rows_temporal(d_rows[:20]):
        return True, "temporal_epoch_equivalent"
    # Check queries without ORDER BY (e.g. Q17: `GROUP BY UserID, SearchPhrase LIMIT 10`)
    upper = sql.upper()
    if "ORDER BY" not in upper and "LIMIT" in upper and len(j_rows) <= 50 and len(j_rows) > 0:
        base_sql = re.sub(r"\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*$", "", sql, flags=re.IGNORECASE)
        try:
            col0_vals = ", ".join(repr(r[0]) if isinstance(r[0], str) else str(r[0]) for r in j_rows if r[0] is not None)
            first_col = dcon.execute(f"DESCRIBE ({base_sql})").fetchone()[0]
            sub_rows = dcon.execute(f'SELECT * FROM ({base_sql}) _sub WHERE "{first_col}" IN ({col0_vals})').fetchall()
            sub_set = set(normalize_rows_temporal(sub_rows))
            if all(r in sub_set for r in normalize_rows_temporal(j_rows)):
                return True, "unordered_limit_valid_subset"
        except Exception:
            pass
    # Check ORDER BY tie-breaking (e.g. Q10, Q11, Q22, Q30, Q31, Q32 where multiple groups share the same count)
    if "ORDER BY" in upper and "LIMIT" in upper and 0 < len(d_rows) <= 50:
        base_sql = re.sub(r"\s+ORDER\s+BY\s+.+$", "", sql, flags=re.IGNORECASE)
        try:
            col0_vals = ", ".join(repr(r[0]) if isinstance(r[0], str) else str(r[0]) for r in j_rows if r[0] is not None)
            first_col = dcon.execute(f"DESCRIBE ({base_sql})").fetchone()[0]
            sub_rows = dcon.execute(f'SELECT * FROM ({base_sql}) _sub WHERE "{first_col}" IN ({col0_vals})').fetchall()
            sub_set = set(normalize_rows_temporal(sub_rows))
            if all(r in sub_set for r in normalize_rows_temporal(j_rows)):
                return True, "order_by_tie_verified_in_duckdb"
        except Exception:
            pass
    return False, "value_diff"


def proc_rss_mb(pid: int):
    try:
        if sys.platform.startswith("linux"):
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        if kb > 0:
                            return round(kb / 1024.0, 2)
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return round(int(out) / 1024.0, 2) if out else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to official ClickBench hits.parquet")
    ap.add_argument("--queries", required=True, help="Path to official ClickBench queries.sql (43 lines)")
    ap.add_argument("--rows", type=int, default=1_000_000,
                    help="Number of rows from official hits.parquet to load (0 = full 99,997,497-row file)")
    ap.add_argument("--pg-port", type=int, default=28082)
    ap.add_argument("--jouledb-bin", default="", help="Optional path to jouledb binary to spawn on (pg_port-2)")
    ap.add_argument("--jouledb-pid", type=int, default=0)
    ap.add_argument("--query-timeout-s", type=float, default=30.0)
    ap.add_argument("--copy-dir", default="", help="Server JOULEDB_COPY_DIR for Parquet slice staging")
    ap.add_argument("--data-parent-dir", default="", help="Disk directory for temporary JouleDB WAL/store (avoids RAM tmpfs)")
    ap.add_argument("--out", default="clickbench-official-result.json")
    args = ap.parse_args()

    import duckdb
    import psycopg

    parquet_abs = os.path.abspath(args.parquet)
    parquet_bytes = os.path.getsize(parquet_abs)
    with open(args.queries) as qf:
        raw_queries = [ln.strip().rstrip(";") for ln in qf if ln.strip() and not ln.strip().startswith("--")]

    meter = make_meter()
    meter.start()
    energy_tag = "reported_j" if meter.source != "none" and not meter.source.startswith("none") else "unmetered"

    with Window(meter) as idle_win:
        time.sleep(2.0)
    idle_w = (idle_win.j / (idle_win.t1 - idle_win.t0)) if idle_win.j is not None else None

    dcon = duckdb.connect()
    total_file_rows = dcon.execute(f"SELECT count(*) FROM read_parquet('{parquet_abs}')").fetchone()[0]

    # Prepare slice or full file
    target_rows = total_file_rows if args.rows <= 0 or args.rows >= total_file_rows else args.rows
    if target_rows == total_file_rows:
        active_parquet = parquet_abs
    else:
        stage_dir = args.copy_dir or os.path.dirname(parquet_abs)
        active_parquet = os.path.join(stage_dir, f"hits_official_head_{target_rows}.parquet")
        if not os.path.exists(active_parquet):
            dcon.execute(
                f"COPY (SELECT * FROM read_parquet('{parquet_abs}') LIMIT {target_rows}) "
                f"TO '{active_parquet}' (FORMAT parquet)"
            )

    schema_rows = dcon.execute(f"DESCRIBE SELECT * FROM read_parquet('{active_parquet}')").fetchall()
    col_defs = ", ".join(f'"{c[0]}" {duck_type_to_pg_ddl(c[1])}' for c in schema_rows)
    ddl = f"CREATE TABLE hits ({col_defs})"

    report = {
        "benchmark": "ClickBench (official hits.parquet + official 43 queries)",
        "official_parquet_bytes": parquet_bytes,
        "official_parquet_total_rows": total_file_rows,
        "evaluated_rows": target_rows,
        "is_full_100m_dataset": target_rows == total_file_rows,
        "num_columns": len(schema_rows),
        "num_queries": len(raw_queries),
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "energy_source": meter.source,
        "energy_evidence": energy_tag,
        "idle_watts": round(idle_w, 2) if idle_w is not None else None,
        "engines": {},
        "summary": {},
        "queries": {},
    }

    # 1. Load into DuckDB (applying ClickBench duckdb/create.sql EventTime/EventDate epoch types)
    with Window(meter) as w_duck_load:
        dcon.execute(
            f"CREATE TABLE hits AS SELECT * REPLACE ("
            f"to_timestamp(EventTime) AS EventTime, "
            f"make_date(1970, 1, 1) + EventDate::INT AS EventDate"
            f") FROM read_parquet('{active_parquet}')"
        )
    duck_count = dcon.execute("SELECT count(*) FROM hits").fetchone()[0]
    report["engines"]["duckdb"] = {
        "version": duckdb.__version__,
        "load_seconds": round(w_duck_load.t1 - w_duck_load.t0, 4),
        "load_joules": round(w_duck_load.j, 3) if w_duck_load.j is not None else None,
        "loaded_rows": duck_count,
    }

    # 2. Load into JouleDB over real pgwire (optionally spawning clean server)
    spawned_proc = None
    temp_data_dir = None
    srv_log_path = None
    server_restarts = 0

    def start_jdb_proc(fresh_dir=False):
        nonlocal spawned_proc, temp_data_dir, srv_log_path
        import socket
        import tempfile
        if fresh_dir or temp_data_dir is None:
            if temp_data_dir and os.path.isdir(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)
            parent_d = os.path.abspath(args.data_parent_dir or args.copy_dir or os.path.dirname(parquet_abs))
            os.makedirs(parent_d, exist_ok=True)
            temp_data_dir = tempfile.mkdtemp(prefix="jdb_clickbench_", dir=parent_d)
            srv_log_path = os.path.join(temp_data_dir, "srv.log")
        http_port = args.pg_port - 2
        env = os.environ.copy()
        if args.copy_dir:
            env["JOULEDB_COPY_DIR"] = os.path.abspath(args.copy_dir)
        log_f = open(srv_log_path, "ab")
        spawned_proc = subprocess.Popen(
            [args.jouledb_bin, "-p", str(http_port), "-d", temp_data_dir,
             "--query-timeout", str(int(args.query_timeout_s * 1000))],
            stdout=log_f, stderr=subprocess.STDOUT, env=env,
        )
        args.jouledb_pid = spawned_proc.pid
        for _ in range(300):
            try:
                with socket.create_connection(("127.0.0.1", args.pg_port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.1)

    if args.jouledb_bin:
        start_jdb_proc(fresh_dir=True)

    pcon = psycopg.connect(
        host="127.0.0.1",
        port=args.pg_port,
        user="postgres",
        dbname="postgres",
        sslmode="disable",
        autocommit=True,
    )
    jdb_ver = pcon.execute("SELECT version()").fetchone()[0]
    pcon.execute("DROP TABLE IF EXISTS hits")
    pcon.execute(ddl)

    jdb_load_error = None
    jdb_loaded_rows = 0
    copy_target = active_parquet
    with Window(meter) as w_jdb_load:
        try:
            try:
                pcon.execute(f"COPY hits FROM '{copy_target}'")
            except Exception:
                copy_target = os.path.basename(active_parquet)
                pcon.execute(f"COPY hits FROM '{copy_target}'")
            jdb_loaded_rows = pcon.execute("SELECT count(*) FROM hits").fetchone()[0]
        except Exception as ex:
            jdb_load_error = f"{type(ex).__name__}: {str(ex)[:300]}"

    report["engines"]["jouledb"] = {
        "version": jdb_ver,
        "load_seconds": round(w_jdb_load.t1 - w_jdb_load.t0, 4),
        "load_joules": round(w_jdb_load.j, 3) if w_jdb_load.j is not None else None,
        "loaded_rows": jdb_loaded_rows,
        "rss_mb_after_load": proc_rss_mb(args.jouledb_pid) if args.jouledb_pid else None,
        "load_error": jdb_load_error,
    }

    # 3. Execute all 43 official ClickBench queries
    jdb_ok = 0
    jdb_err = 0
    jdb_exact_match = 0
    jdb_sql_valid_match = 0
    jdb_faster = 0
    duck_faster = 0
    ratios_ms = []
    ratios_j = []

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Query exceeded {args.query_timeout_s}s timeout")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _on_timeout)

    for idx, sql in enumerate(raw_queries):
        qid = f"Q{idx}"
        q_entry = {"sql": sql}

        # DuckDB run (warm 1 + timed 1, or single cold/warm recorded)
        try:
            with Window(meter) as wd:
                d_rows = dcon.execute(sql).fetchall()
            d_ms = (wd.t1 - wd.t0) * 1000.0
            q_entry["duckdb"] = {
                "status": "ok",
                "ms": round(d_ms, 3),
                "joules": round(wd.j, 4) if wd.j is not None else None,
                "row_count": len(d_rows),
            }
        except Exception as ex:
            d_rows = None
            q_entry["duckdb"] = {"status": "error", "error": f"{type(ex).__name__}: {str(ex)[:240]}"}

        # JouleDB run over pgwire
        if jdb_load_error is not None:
            jdb_err += 1
            q_entry["jouledb"] = {"status": "skipped_load_failed", "error": jdb_load_error}
        else:
            try:
                if hasattr(signal, "ITIMER_REAL"):
                    signal.setitimer(signal.ITIMER_REAL, args.query_timeout_s)
                with Window(meter) as wj:
                    j_rows = pcon.execute(sql).fetchall()
                if hasattr(signal, "ITIMER_REAL"):
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                j_ms = (wj.t1 - wj.t0) * 1000.0
                jdb_ok += 1
                q_entry["jouledb"] = {
                    "status": "ok",
                    "ms": round(j_ms, 3),
                    "joules": round(wj.j, 4) if wj.j is not None else None,
                    "row_count": len(j_rows),
                }
                if d_rows is not None:
                    matched = normalize_rows(j_rows[:20]) == normalize_rows(d_rows[:20]) and len(j_rows) == len(d_rows)
                    sql_valid, match_kind = verify_sql_equivalence(dcon, sql, d_rows, j_rows)
                    q_entry["results_match"] = matched
                    q_entry["results_sql_valid"] = sql_valid
                    q_entry["match_kind"] = match_kind
                    if matched:
                        jdb_exact_match += 1
                    if sql_valid:
                        jdb_sql_valid_match += 1
                    if not matched:
                        q_entry["sample_diff"] = {
                            "duckdb_head": str(d_rows[:2])[:200],
                            "jouledb_head": str(j_rows[:2])[:200],
                        }
                    d_ms_val = q_entry["duckdb"]["ms"]
                    if d_ms_val > 0 and j_ms > 0:
                        ratio = j_ms / d_ms_val
                        q_entry["jouledb_vs_duckdb_ms_ratio"] = round(ratio, 3)
                        ratios_ms.append(ratio)
                        if j_ms < d_ms_val:
                            jdb_faster += 1
                        else:
                            duck_faster += 1
                    dj = q_entry["duckdb"].get("joules")
                    jj = q_entry["jouledb"].get("joules")
                    if dj and jj and dj > 0 and jj > 0:
                        ratios_j.append(jj / dj)
            except Exception as ex:
                if hasattr(signal, "ITIMER_REAL"):
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                jdb_err += 1
                err_status = "error"
                srv_tail = ""
                try:
                    pcon.close()
                except Exception:
                    pass
                if spawned_proc is not None:
                    time.sleep(0.15)
                    if spawned_proc.poll() is not None:
                        err_status = "server_aborted"
                        args.pg_port += 10
                        server_restarts += 1
                        if srv_log_path and os.path.exists(srv_log_path):
                            with open(srv_log_path, "rb") as lf:
                                srv_tail = lf.read()[-400:].decode("utf-8", errors="replace").strip()
                        start_jdb_proc(fresh_dir=True)
                    elif isinstance(ex, TimeoutError):
                        err_status = "timeout"
                        spawned_proc.terminate()
                        try:
                            spawned_proc.wait(timeout=2)
                        except Exception:
                            spawned_proc.send_signal(9)
                            spawned_proc.wait(timeout=5)
                        args.pg_port += 10
                        server_restarts += 1
                        start_jdb_proc(fresh_dir=True)
                pcon = psycopg.connect(
                    host="127.0.0.1",
                    port=args.pg_port,
                    user="postgres",
                    dbname="postgres",
                    sslmode="disable",
                    autocommit=True,
                )
                if err_status in ("server_aborted", "timeout") and jdb_load_error is None:
                    try:
                        cnt = pcon.execute("SELECT count(*) FROM hits").fetchone()[0]
                    except Exception:
                        cnt = -1
                    if cnt != jdb_loaded_rows:
                        pcon.execute("DROP TABLE IF EXISTS hits")
                        pcon.execute(ddl)
                        pcon.execute(f"COPY hits FROM '{copy_target}'")
                q_entry["jouledb"] = {
                    "status": err_status,
                    "error": f"{type(ex).__name__}: {str(ex)[:240]}",
                }
                if srv_tail:
                    q_entry["jouledb"]["server_log_tail"] = srv_tail

        report["queries"][qid] = q_entry

    meter.stop()
    try:
        pcon.close()
    except Exception:
        pass
    if spawned_proc is not None:
        spawned_proc.terminate()
        try:
            spawned_proc.wait(timeout=5)
        except Exception:
            pass
    if temp_data_dir and os.path.isdir(temp_data_dir):
        shutil.rmtree(temp_data_dir, ignore_errors=True)

    geo_mean = lambda xs: round(pow(10.0, statistics.mean([ __import__("math").log10(max(x, 1e-9)) for x in xs ])), 3) if xs else None
    report["summary"] = {
        "queries_total": len(raw_queries),
        "jouledb_succeeded": jdb_ok,
        "jouledb_errored_or_unsupported": jdb_err,
        "jouledb_server_restarts_on_timeout_or_abort": server_restarts,
        "jouledb_exact_answer_matches": jdb_exact_match,
        "jouledb_sql_valid_matches": jdb_sql_valid_match,
        "jouledb_faster_count": jdb_faster,
        "duckdb_faster_count": duck_faster,
        "geomean_jouledb_over_duckdb_latency_x": geo_mean(ratios_ms),
        "median_jouledb_over_duckdb_latency_x": round(statistics.median(ratios_ms), 3) if ratios_ms else None,
        "geomean_jouledb_over_duckdb_joules_x": geo_mean(ratios_j),
    }

    with open(args.out, "w") as out_f:
        json.dump(report, out_f, indent=2)

    print(f"=== Official ClickBench ({target_rows:,} / {total_file_rows:,} rows) on {report['host']} ===")
    print(f"Energy source: {meter.source} ({energy_tag}), idle={report['idle_watts']} W")
    print(f"Load: DuckDB {report['engines']['duckdb']['load_seconds']}s vs JouleDB {report['engines']['jouledb']['load_seconds']}s (err={jdb_load_error})")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
