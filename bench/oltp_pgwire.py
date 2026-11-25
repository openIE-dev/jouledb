#!/usr/bin/env python3
"""Concurrent Multi-Client OLTP Benchmark over Real TCP `pgwire`: JouleDB vs PostgreSQL & SQLite WAL.

Evaluates durable transactional OLTP throughput (`tps`), client-side TCP round-trip latency
(`p50`, `p95`, `p99` ms), hardware energy (`reported_j` via Linux RAPL / macOS powermetrics),
and post-run conservation invariants across `1`, `4`, `8`, and `16` concurrent worker clients.

Workload mix per transaction (`--warehouses` * 1,000 accounts, default 100 warehouses = 100,000 accounts):
  - 50% Point Read (`SELECT balance, branch_id FROM accounts WHERE id = ?`)
  - 30% Account Deposit/Update + Ledger Append (`UPDATE accounts SET balance = balance + delta WHERE id = ?` + `INSERT INTO history ...`)
  - 20% Warehouse Aggregate Probe (`SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = ?`)

Post-run invariant check:
  Verifies `SUM(balance) == initial_sum + exact_committed_deltas` and `COUNT(history) == committed_writes`.
"""
import argparse
import json
import os
import platform
import random
import shutil
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Window, make_meter  # noqa: E402


def percentile(sorted_vals, pct: float):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return round(sorted_vals[f], 3)
    return round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 3)


def wait_tcp(host: str, port: int, timeout_s: float = 20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def run_pgwire_suite(
    engine_label: str,
    host: str,
    port: int,
    user: str,
    dbname: str,
    warehouses: int,
    accounts_per_wh: int,
    concurrencies: list[int],
    duration_s: float,
    meter,
):
    import psycopg

    total_accounts = warehouses * accounts_per_wh
    init_bal = 1000

    with psycopg.connect(host=host, port=port, user=user, dbname=dbname, sslmode="disable", autocommit=True) as setup_con:
        ver = setup_con.execute("SELECT version()").fetchone()[0]
        setup_con.execute("DROP TABLE IF EXISTS history")
        setup_con.execute("DROP TABLE IF EXISTS accounts")
        # Populate via standard PostgreSQL `COPY ... FROM STDIN` (matching `pgbench -i`),
        # then create secondary indexes for point and warehouse lookups.
        setup_con.execute(
            "CREATE TABLE accounts (id BIGINT, branch_id BIGINT, balance BIGINT)"
        )
        setup_con.execute(
            "CREATE TABLE history (tx_seq BIGINT, acct_id BIGINT, branch_id BIGINT, delta BIGINT)"
        )

        with Window(meter) as w_load:
            with setup_con.cursor().copy("COPY accounts (id, branch_id, balance) FROM STDIN") as cp:
                for aid in range(1, total_accounts + 1):
                    cp.write_row((aid, (aid - 1) // accounts_per_wh + 1, init_bal))
            setup_con.execute("CREATE INDEX idx_accounts_id ON accounts (id)")
            setup_con.execute("CREATE INDEX idx_accounts_branch ON accounts (branch_id)")

    results_by_c = {}
    cumulative_delta = 0
    cumulative_writes = 0
    tx_counter = 0
    tx_lock = threading.Lock()

    for conc in concurrencies:
        stop_ev = threading.Event()
        barrier = threading.Barrier(conc + 1)
        worker_stats = [None] * conc

        def worker(wid: int):
            nonlocal cumulative_delta, cumulative_writes, tx_counter
            rng = random.Random(20260700 + conc * 100 + wid)
            latencies_ms = []
            reads_ok = 0
            writes_ok = 0
            aggs_ok = 0
            errors = 0
            local_delta = 0
            local_writes = 0

            con = psycopg.connect(
                host=host, port=port, user=user, dbname=dbname, sslmode="disable", autocommit=True
            )
            barrier.wait()
            while not stop_ev.is_set():
                r = rng.random()
                t0 = time.perf_counter()
                try:
                    if r < 0.50:
                        aid = rng.randint(1, total_accounts)
                        con.execute(
                            f"SELECT balance, branch_id FROM accounts WHERE id = {aid}"
                        ).fetchone()
                        reads_ok += 1
                    elif r < 0.80:
                        aid = rng.randint(1, total_accounts)
                        bid = (aid - 1) // accounts_per_wh + 1
                        delta = rng.choice([-50, -20, -10, 10, 20, 50, 100])
                        with tx_lock:
                            tx_counter += 1
                            seq = tx_counter
                        con.execute("BEGIN")
                        con.execute(
                            f"UPDATE accounts SET balance = balance + ({delta}) WHERE id = {aid}"
                        )
                        con.execute(
                            f"INSERT INTO history (tx_seq, acct_id, branch_id, delta) VALUES ({seq}, {aid}, {bid}, {delta})"
                        )
                        con.execute("COMMIT")
                        writes_ok += 1
                        local_writes += 1
                        local_delta += delta
                    else:
                        bid = rng.randint(1, warehouses)
                        con.execute(
                            f"SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = {bid}"
                        ).fetchone()
                        aggs_ok += 1
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                except Exception:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        pass
                    errors += 1

            con.close()
            with tx_lock:
                cumulative_delta += local_delta
                cumulative_writes += local_writes
            worker_stats[wid] = {
                "latencies_ms": latencies_ms,
                "reads_ok": reads_ok,
                "writes_ok": writes_ok,
                "aggs_ok": aggs_ok,
                "errors": errors,
            }

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conc)]
        for t in threads:
            t.start()
        barrier.wait()
        with Window(meter) as w_run:
            time.sleep(duration_s)
            stop_ev.set()
            for t in threads:
                t.join()

        elapsed = w_run.t1 - w_run.t0
        all_lats = []
        r_ok = w_ok = a_ok = errs = 0
        for st in worker_stats:
            if st:
                all_lats.extend(st["latencies_ms"])
                r_ok += st["reads_ok"]
                w_ok += st["writes_ok"]
                a_ok += st["aggs_ok"]
                errs += st["errors"]
        all_lats.sort()
        total_tx = r_ok + w_ok + a_ok
        tps = total_tx / elapsed if elapsed > 0 else 0.0
        mj_per_tx = ((w_run.j * 1000.0) / total_tx) if (w_run.j is not None and total_tx > 0) else None
        results_by_c[f"c{conc}"] = {
            "concurrency": conc,
            "elapsed_s": round(elapsed, 4),
            "transactions_ok": total_tx,
            "point_reads_ok": r_ok,
            "write_txns_ok": w_ok,
            "warehouse_aggs_ok": a_ok,
            "errors": errs,
            "tps": round(tps, 2),
            "p50_ms": percentile(all_lats, 50),
            "p95_ms": percentile(all_lats, 95),
            "p99_ms": percentile(all_lats, 99),
            "joules": round(w_run.j, 4) if w_run.j is not None else None,
            "mj_per_txn": round(mj_per_tx, 4) if mj_per_tx is not None else None,
        }

    # Verify conservation invariant across all accounts and ledger history!
    with psycopg.connect(host=host, port=port, user=user, dbname=dbname, sslmode="disable", autocommit=True) as check_con:
        acct_cnt, acct_sum = check_con.execute("SELECT COUNT(*), SUM(balance) FROM accounts").fetchone()
        hist_cnt, hist_sum = check_con.execute("SELECT COUNT(*), COALESCE(SUM(delta), 0) FROM history").fetchone()

    expected_sum = total_accounts * init_bal + cumulative_delta
    invariant_ok = (
        int(acct_cnt) == total_accounts
        and int(acct_sum) == expected_sum
        and int(hist_cnt) == cumulative_writes
        and int(hist_sum) == cumulative_delta
    )

    return {
        "engine": engine_label,
        "version": ver,
        "protocol": "PostgreSQL v3 wire (psycopg TCP)",
        "load_seconds": round(w_load.t1 - w_load.t0, 4),
        "load_joules": round(w_load.j, 4) if w_load.j is not None else None,
        "invariant_verified": invariant_ok,
        "invariant_details": {
            "accounts_count": int(acct_cnt),
            "expected_balance_sum": expected_sum,
            "actual_balance_sum": int(acct_sum),
            "committed_write_txns": cumulative_writes,
            "history_rows": int(hist_cnt),
        },
        "concurrency_runs": results_by_c,
    }


def run_sqlite_wal_suite(
    db_path: str,
    warehouses: int,
    accounts_per_wh: int,
    concurrencies: list[int],
    duration_s: float,
    meter,
):
    total_accounts = warehouses * accounts_per_wh
    init_bal = 1000
    scon = sqlite3.connect(db_path, timeout=30.0)
    scon.execute("PRAGMA journal_mode=WAL")
    scon.execute("PRAGMA synchronous=NORMAL")
    scon.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, branch_id INTEGER, balance INTEGER)")
    scon.execute("CREATE TABLE history (tx_seq INTEGER, acct_id INTEGER, branch_id INTEGER, delta INTEGER)")
    scon.execute("CREATE INDEX idx_accounts_branch ON accounts (branch_id)")
    with Window(meter) as w_load:
        scon.executemany(
            "INSERT INTO accounts VALUES (?, ?, ?)",
            [(aid, (aid - 1) // accounts_per_wh + 1, init_bal) for aid in range(1, total_accounts + 1)],
        )
        scon.commit()
    scon.close()

    results_by_c = {}
    cumulative_delta = 0
    cumulative_writes = 0
    tx_counter = 0
    tx_lock = threading.Lock()

    for conc in concurrencies:
        stop_ev = threading.Event()
        barrier = threading.Barrier(conc + 1)
        worker_stats = [None] * conc

        def worker(wid: int):
            nonlocal cumulative_delta, cumulative_writes, tx_counter
            rng = random.Random(20260700 + conc * 100 + wid)
            lats = []
            r_ok = w_ok = a_ok = errs = 0
            l_delta = 0
            l_writes = 0
            con = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
            con.execute("PRAGMA synchronous=NORMAL")
            barrier.wait()
            while not stop_ev.is_set():
                r = rng.random()
                t0 = time.perf_counter()
                try:
                    if r < 0.50:
                        aid = rng.randint(1, total_accounts)
                        con.execute("SELECT balance, branch_id FROM accounts WHERE id = ?", (aid,)).fetchone()
                        r_ok += 1
                    elif r < 0.80:
                        aid = rng.randint(1, total_accounts)
                        bid = (aid - 1) // accounts_per_wh + 1
                        delta = rng.choice([-50, -20, -10, 10, 20, 50, 100])
                        with tx_lock:
                            tx_counter += 1
                            seq = tx_counter
                        con.execute("BEGIN IMMEDIATE")
                        con.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (delta, aid))
                        con.execute("INSERT INTO history VALUES (?, ?, ?, ?)", (seq, aid, bid, delta))
                        con.execute("COMMIT")
                        w_ok += 1
                        l_writes += 1
                        l_delta += delta
                    else:
                        bid = rng.randint(1, warehouses)
                        con.execute("SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = ?", (bid,)).fetchone()
                        a_ok += 1
                    lats.append((time.perf_counter() - t0) * 1000.0)
                except Exception:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        pass
                    errs += 1
            con.close()
            with tx_lock:
                cumulative_delta += l_delta
                cumulative_writes += l_writes
            worker_stats[wid] = (lats, r_ok, w_ok, a_ok, errs)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conc)]
        for t in threads:
            t.start()
        barrier.wait()
        with Window(meter) as w_run:
            time.sleep(duration_s)
            stop_ev.set()
            for t in threads:
                t.join()
        elapsed = w_run.t1 - w_run.t0
        all_lats = []
        r_ok = w_ok = a_ok = errs = 0
        for st in worker_stats:
            if st:
                all_lats.extend(st[0])
                r_ok += st[1]
                w_ok += st[2]
                a_ok += st[3]
                errs += st[4]
        all_lats.sort()
        total_tx = r_ok + w_ok + a_ok
        tps = total_tx / elapsed if elapsed > 0 else 0.0
        mj = ((w_run.j * 1000.0) / total_tx) if (w_run.j is not None and total_tx > 0) else None
        results_by_c[f"c{conc}"] = {
            "concurrency": conc,
            "elapsed_s": round(elapsed, 4),
            "transactions_ok": total_tx,
            "point_reads_ok": r_ok,
            "write_txns_ok": w_ok,
            "warehouse_aggs_ok": a_ok,
            "errors": errs,
            "tps": round(tps, 2),
            "p50_ms": percentile(all_lats, 50),
            "p95_ms": percentile(all_lats, 95),
            "p99_ms": percentile(all_lats, 99),
            "joules": round(w_run.j, 4) if w_run.j is not None else None,
            "mj_per_txn": round(mj, 4) if mj is not None else None,
        }

    scon = sqlite3.connect(db_path)
    acct_cnt, acct_sum = scon.execute("SELECT COUNT(*), SUM(balance) FROM accounts").fetchone()
    hist_cnt, hist_sum = scon.execute("SELECT COUNT(*), COALESCE(SUM(delta), 0) FROM history").fetchone()
    scon.close()
    expected_sum = total_accounts * init_bal + cumulative_delta
    return {
        "engine": "sqlite3_wal",
        "version": f"SQLite {sqlite3.sqlite_version} (WAL, synchronous=NORMAL, embedded in-process baseline)",
        "protocol": "Embedded in-process C API (no TCP wire overhead)",
        "load_seconds": round(w_load.t1 - w_load.t0, 4),
        "load_joules": round(w_load.j, 4) if w_load.j is not None else None,
        "invariant_verified": (
            int(acct_cnt) == total_accounts
            and int(acct_sum) == expected_sum
            and int(hist_cnt) == cumulative_writes
            and int(hist_sum) == cumulative_delta
        ),
        "invariant_details": {
            "accounts_count": int(acct_cnt),
            "expected_balance_sum": expected_sum,
            "actual_balance_sum": int(acct_sum),
            "committed_write_txns": cumulative_writes,
            "history_rows": int(hist_cnt),
        },
        "concurrency_runs": results_by_c,
    }


def find_pg_bindir():
    for candidate in ["/usr/lib/postgresql/17/bin", "/usr/lib/postgresql/16/bin", "/usr/lib/postgresql/15/bin", "/opt/homebrew/opt/postgresql@16/bin"]:
        if os.path.exists(os.path.join(candidate, "initdb")) and os.path.exists(os.path.join(candidate, "postgres")):
            return candidate
    initdb = shutil.which("initdb")
    return os.path.dirname(initdb) if initdb else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jouledb-bin", required=True)
    ap.add_argument("--work-dir", default="", help="Disk directory for temporary data dirs (avoids RAM tmpfs)")
    ap.add_argument("--warehouses", type=int, default=100)
    ap.add_argument("--accounts-per-warehouse", type=int, default=1000)
    ap.add_argument("--concurrencies", default="1,4,8,16")
    ap.add_argument("--duration-s", type=float, default=8.0)
    ap.add_argument("--pg-port", type=int, default=29182)
    ap.add_argument("--out", default="oltp-pgwire-result.json")
    args = ap.parse_args()

    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    work_parent = os.path.abspath(args.work_dir or ".")
    os.makedirs(work_parent, exist_ok=True)
    root_tmp = tempfile.mkdtemp(prefix="oltp_bench_", dir=work_parent)

    meter = make_meter()
    meter.start()
    energy_tag = "reported_j" if meter.source != "none" and not meter.source.startswith("none") else "unmetered"

    with Window(meter) as idle_w:
        time.sleep(1.5)
    idle_watts = (idle_w.j / (idle_w.t1 - idle_w.t0)) if idle_w.j is not None else None

    report = {
        "benchmark": "Concurrent Multi-Client OLTP over TCP pgwire (50% Point Read / 30% Durable Balance Update+Insert / 20% Warehouse Aggregate)",
        "warehouses": args.warehouses,
        "accounts_per_warehouse": args.accounts_per_warehouse,
        "total_accounts": args.warehouses * args.accounts_per_warehouse,
        "concurrencies_tested": concs,
        "duration_per_level_s": args.duration_s,
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "energy_source": meter.source,
        "energy_evidence": energy_tag,
        "idle_watts": round(idle_watts, 2) if idle_watts is not None else None,
        "engines": {},
    }

    try:
        # 1. JouleDB over TCP pgwire (durable WAL group-commit enabled!)
        jdb_dir = os.path.join(root_tmp, "jouledb_data")
        os.makedirs(jdb_dir, exist_ok=True)
        http_port = args.pg_port - 2
        jdb_log = open(os.path.join(root_tmp, "jouledb.log"), "wb")
        jdb_proc = subprocess.Popen(
            [args.jouledb_bin, "-p", str(http_port), "-d", jdb_dir],
            stdout=jdb_log, stderr=subprocess.STDOUT,
        )
        try:
            if not wait_tcp("127.0.0.1", args.pg_port, 20.0):
                raise RuntimeError("JouleDB pgwire failed to bind")
            report["engines"]["jouledb_pgwire"] = run_pgwire_suite(
                "jouledb_pgwire",
                "127.0.0.1",
                args.pg_port,
                "postgres",
                "postgres",
                args.warehouses,
                args.accounts_per_warehouse,
                concs,
                args.duration_s,
                meter,
            )
        finally:
            jdb_proc.terminate()
            try:
                jdb_proc.wait(timeout=5)
            except Exception:
                jdb_proc.kill()
            jdb_log.close()

        # 2. PostgreSQL (if installed on host) over TCP pgwire
        pg_bin = find_pg_bindir()
        if pg_bin:
            pg_dir = os.path.join(root_tmp, "pg_data")
            pg_port_real = args.pg_port + 20
            subprocess.run([os.path.join(pg_bin, "initdb"), "-D", pg_dir, "-A", "trust", "-U", "postgres"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pg_proc = subprocess.Popen(
                [os.path.join(pg_bin, "postgres"), "-D", pg_dir, "-p", str(pg_port_real), "-k", root_tmp, "-h", "127.0.0.1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                if wait_tcp("127.0.0.1", pg_port_real, 15.0):
                    report["engines"]["postgresql_pgwire"] = run_pgwire_suite(
                        "postgresql_pgwire",
                        "127.0.0.1",
                        pg_port_real,
                        "postgres",
                        "postgres",
                        args.warehouses,
                        args.accounts_per_warehouse,
                        concs,
                        args.duration_s,
                        meter,
                    )
            finally:
                pg_proc.terminate()
                try:
                    pg_proc.wait(timeout=5)
                except Exception:
                    pg_proc.kill()

        # 3. SQLite 3 WAL baseline
        sq_path = os.path.join(root_tmp, "oltp.sqlite")
        report["engines"]["sqlite3_wal"] = run_sqlite_wal_suite(
            sq_path,
            args.warehouses,
            args.accounts_per_warehouse,
            concs,
            args.duration_s,
            meter,
        )
    finally:
        meter.stop()
        shutil.rmtree(root_tmp, ignore_errors=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
