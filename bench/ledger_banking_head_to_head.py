#!/usr/bin/env python3
"""Head-to-Head Core Banking Double-Entry Ledger Benchmark (NVMe + Hardware RAPL `reported_j`):
JouleDB v0.16.0 (`JWL1` Binary CRC-32 + SHA-256 Chain + Adaptive FIFO Group Commit) vs PostgreSQL 16 (`fsync=on`).

Enforces identical core-banking invariants on both engines over 1,000 customer accounts:
  1. Conservation of Money: every transfer records balanced debit + credit amounts (sum(debits) == sum(credits)).
  2. Hard Overdraft Floor: customer accounts enforce `posted_credits - posted_debits >= 0` (`NO_OVERDRAFT` / `CHECK`).
  3. Native Idempotency: duplicate `idempotency_key` retries (~5% of traffic) return without double-moving funds.
  4. Cryptographic Auditability: SHA-256 digest persisted per transfer (`JWL1` chain in JouleDB; built-in `sha256()` in PG16).
  5. Strict `fsync=on` Durability: physical NVMe directory (`--work-dir /var/tmp`), no `synchronous_commit=off` shortcuts.

Four Apples-to-Apples Regimes:
  - `c=1`:   1 client worker issuing individual durable transfers (`keep-alive` HTTP/1.1 vs persistent `psycopg3` TCP).
  - `c=4`:   4 concurrent workers exercising automatic group-commit coalescing (`JouleDB` FIFO vs `PG16` XLogFlush).
  - `c=16`:  16 concurrent workers under hot-account contention + 5% idempotent retries (`0` overdrafts, `0` double-spends).
  - `batch=500`: Clearing-house settlement batches of 500 transfers per durable `fdatasync` (`POST /api/v1/ledger/transfers/batch`
                 vs PostgreSQL 16 set-based `UNNEST(...)` single-transaction CTE).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Window, make_meter  # noqa: E402


def percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return round(sorted_vals[f], 3)
    return round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 3)


def wait_http(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.05)
    return False


def wait_pg_ready(host: str, port: int, timeout_s: float = 30.0) -> bool:
    import psycopg

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(
                host=host,
                port=port,
                user="postgres",
                dbname="postgres",
                sslmode="disable",
                connect_timeout=2,
            ) as c:
                c.execute("SELECT 1").fetchone()
                return True
        except Exception:
            time.sleep(0.1)
    return False


def find_pg_bindir() -> str | None:
    for candidate in [
        "/usr/lib/postgresql/17/bin",
        "/usr/lib/postgresql/16/bin",
        "/usr/lib/postgresql/15/bin",
        "/opt/homebrew/opt/postgresql@16/bin",
    ]:
        if os.path.exists(os.path.join(candidate, "initdb")) and os.path.exists(
            os.path.join(candidate, "postgres")
        ):
            return candidate
    initdb = shutil.which("initdb")
    return os.path.dirname(initdb) if initdb else None


def docker_available() -> bool:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    res = subprocess.run(
        [docker_bin, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return res.returncode == 0


def summarize_window(
    committed_tx: int,
    replayed_tx: int,
    rejected_tx: int,
    latencies_ms: list[float],
    win: Window,
    idle_watts: float | None,
    batch_size: int = 1,
) -> dict:
    elapsed = max(win.t1 - win.t0, 1e-6)
    total_ops = committed_tx + replayed_tx
    tps = total_ops / elapsed
    lat_sorted = sorted(latencies_ms)
    raw_j = win.j
    inc_j = (
        max(0.0, raw_j - (idle_watts * elapsed))
        if (raw_j is not None and idle_watts is not None)
        else None
    )
    return {
        "batch_size": batch_size,
        "elapsed_s": round(elapsed, 4),
        "committed_transfers": committed_tx,
        "idempotent_replays": replayed_tx,
        "overdraft_or_conflict_rejections": rejected_tx,
        "effective_transfers_per_sec": round(tps, 1),
        "p50_ms": percentile(lat_sorted, 50),
        "p95_ms": percentile(lat_sorted, 95),
        "p99_ms": percentile(lat_sorted, 99),
        "reported_j": round(raw_j, 4) if raw_j is not None else None,
        "incremental_j": round(inc_j, 4) if inc_j is not None else None,
        "mj_per_transfer": round((raw_j * 1000.0) / total_ops, 4)
        if (raw_j is not None and total_ops > 0)
        else None,
    }


def run_jouledb_banking(
    http_port: int,
    num_accounts: int,
    concurrencies: list[int],
    duration_s: float,
    batch_size: int,
    meter,
    idle_watts: float | None,
) -> dict:
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    conn = http.client.HTTPConnection("127.0.0.1", http_port, timeout=30.0)

    def post(path: str, obj: dict) -> tuple[int, dict]:
        body = json.dumps(obj).encode("utf-8")
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw)

    # Create treasury + customer wallets with NO_OVERDRAFT (`debits_must_not_exceed_credits = True`)
    st, _ = post(
        "/api/v1/ledger/accounts",
        {"account_id": "treasury_usd", "ledger_id": 1, "code": 1000, "scale": 2},
    )
    assert st == 201
    for i in range(num_accounts):
        st, _ = post(
            "/api/v1/ledger/accounts",
            {
                "account_id": f"acct_{i:04d}",
                "ledger_id": 1,
                "code": 2000,
                "scale": 2,
                "flags": {"debits_must_not_exceed_credits": True},
            },
        )
        assert st == 201

    # Seed each customer wallet with $10,000.00 (1,000,000 cents) in 2 single-fdatasync batches
    chunk = 500
    for start_i in range(0, num_accounts, chunk):
        seed_reqs = [
            {
                "idempotency_key": f"seed-{i:04d}",
                "debit_account_id": "treasury_usd",
                "credit_account_id": f"acct_{i:04d}",
                "amount_minor": 1_000_000,
            }
            for i in range(start_i, min(start_i + chunk, num_accounts))
        ]
        st, rep = post(
            "/api/v1/ledger/transfers/batch", {"transfers": seed_reqs, "atomic": True}
        )
        assert st == 200 and rep["committed"] == len(seed_reqs)
    conn.close()

    regimes: dict[str, dict] = {}
    for conc in concurrencies:
        stop_ev = threading.Event()
        barrier = threading.Barrier(conc + 1)
        stats_per_worker: list[tuple[int, int, int, list[float]]] = [
            (0, 0, 0, []) for _ in range(conc)
        ]

        def worker(wid: int) -> None:
            rng = random.Random(910_000 + conc * 1000 + wid)
            wconn = http.client.HTTPConnection("127.0.0.1", http_port, timeout=30.0)
            committed = 0
            replayed = 0
            rejected = 0
            lats: list[float] = []
            seq_local = 0
            last_ok_payload: dict | None = None
            barrier.wait()
            while not stop_ev.is_set():
                if last_ok_payload is not None and rng.random() < 0.05:
                    payload = last_ok_payload
                else:
                    a = rng.randrange(num_accounts)
                    b = (a + rng.randrange(1, num_accounts)) % num_accounts
                    seq_local += 1
                    payload = {
                        "idempotency_key": f"jdb-c{conc}-w{wid}-{seq_local}",
                        "debit_account_id": f"acct_{a:04d}",
                        "credit_account_id": f"acct_{b:04d}",
                        "amount_minor": rng.randint(1, 250),
                    }
                raw_req = json.dumps(payload).encode("utf-8")
                t0 = time.perf_counter()
                wconn.request("POST", "/api/v1/ledger/transfers", body=raw_req, headers=headers)
                resp = wconn.getresponse()
                data = json.loads(resp.read())
                lats.append((time.perf_counter() - t0) * 1000.0)
                if resp.status == 200:
                    if data.get("idempotent_replay"):
                        replayed += 1
                    else:
                        committed += 1
                        last_ok_payload = payload
                else:
                    rejected += 1
            wconn.close()
            stats_per_worker[wid] = (committed, replayed, rejected, lats)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
        for t in threads:
            t.start()
        barrier.wait()
        with Window(meter) as win:
            time.sleep(duration_s)
            stop_ev.set()
            for t in threads:
                t.join()

        tot_c = sum(x[0] for x in stats_per_worker)
        tot_r = sum(x[1] for x in stats_per_worker)
        tot_x = sum(x[2] for x in stats_per_worker)
        all_lats = [lat for x in stats_per_worker for lat in x[3]]
        regimes[f"c={conc}"] = summarize_window(
            tot_c, tot_r, tot_x, all_lats, win, idle_watts, batch_size=1
        )

    # Regime D: Clearing-house batch settlement (`batch_size` transfers per durable fdatasync)
    bconn = http.client.HTTPConnection("127.0.0.1", http_port, timeout=30.0)
    rng_b = random.Random(990_001)
    batch_committed = 0
    batch_replayed = 0
    batch_rejected = 0
    batch_lats: list[float] = []
    batch_idx = 0
    with Window(meter) as bwin:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            batch_idx += 1
            items = []
            for j in range(batch_size):
                a = (batch_idx * batch_size + j) % num_accounts
                b = (a + 1 + (j % 7)) % num_accounts
                items.append(
                    {
                        "idempotency_key": f"jdb-batch-{batch_idx}-{j}",
                        "debit_account_id": f"acct_{a:04d}",
                        "credit_account_id": f"acct_{b:04d}",
                        "amount_minor": rng_b.randint(1, 50),
                    }
                )
            raw_b = json.dumps(
                {"transfers": items, "atomic": False, "include_receipts": False}
            ).encode("utf-8")
            t0 = time.perf_counter()
            bconn.request("POST", "/api/v1/ledger/transfers/batch", body=raw_b, headers=headers)
            resp = bconn.getresponse()
            rep = json.loads(resp.read())
            batch_lats.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status == 200, f"batch failed: {rep}"
            batch_committed += rep["committed"]
            batch_replayed += rep["idempotent_replays"]
            batch_rejected += rep["rejected"]

    # Verify full SHA-256 hash chain + global double-entry conservation + zero negative customer balances
    bconn.request("GET", "/api/v1/ledger/verify")
    vresp = bconn.getresponse()
    verify = json.loads(vresp.read())
    bconn.request("GET", "/api/v1/ledger/accounts")
    aresp = bconn.getresponse()
    accts = json.loads(aresp.read())["accounts"]
    bconn.close()

    min_customer_net = min(
        a["posted_credits"] - a["posted_debits"]
        for a in accts
        if a["account_id"].startswith("acct_")
    )
    assert verify["valid"] is True, f"JouleDB verify_chain failed: {verify}"
    assert verify["total_posted_debits"] == verify["total_posted_credits"]
    assert min_customer_net >= 0, f"Customer overdraft detected: {min_customer_net}"

    regimes[f"batch={batch_size}"] = summarize_window(
        batch_committed,
        batch_replayed,
        batch_rejected,
        batch_lats,
        bwin,
        idle_watts,
        batch_size=batch_size,
    )
    return {
        "engine": "JouleDB v0.16.0 HTTP (Pre-Allocated JWL1 + SHA-256 Chain + Adaptive FIFO Group Commit, fsync=on)",
        "invariants_verified": {
            "sha256_chain_valid": verify["valid"],
            "journal_entries_verified": verify["journal_entries_verified"],
            "total_posted_debits": verify["total_posted_debits"],
            "total_posted_credits": verify["total_posted_credits"],
            "min_customer_balance_minor": min_customer_net,
        },
        "regimes": regimes,
    }


def run_jouledb_pgwire_banking(
    pgwire_port: int,
    http_port: int,
    num_accounts: int,
    concurrencies: list[int],
    duration_s: float,
    batch_size: int,
    meter,
    idle_watts: float | None,
) -> dict:
    import psycopg

    with psycopg.connect(
        host="127.0.0.1",
        port=pgwire_port,
        user="postgres",
        dbname="postgres",
        sslmode="disable",
        autocommit=True,
    ) as setup:
        setup.execute("LEDGER CREATE ACCOUNT treasury_usd LEDGER 1 CODE 1000 SCALE 2")
        for i in range(num_accounts):
            setup.execute(
                f"LEDGER CREATE ACCOUNT acct_{i:04d} LEDGER 1 CODE 2000 SCALE 2 NO_OVERDRAFT"
            )
        chunk = 500
        for start_i in range(0, num_accounts, chunk):
            end_i = min(start_i + chunk, num_accounts)
            idems = [f"pgw-seed-{i:04d}" for i in range(start_i, end_i)]
            debits = ["treasury_usd" for _ in range(start_i, end_i)]
            credits = [f"acct_{i:04d}" for i in range(start_i, end_i)]
            amts = [1_000_000 for _ in range(start_i, end_i)]
            c_cnt = setup.execute(
                "SELECT ledger_transfer_batch(%s::text[], %s::text[], %s::text[], %s::bigint[], true)",
                (idems, debits, credits, amts),
            ).fetchone()[0]
            assert int(c_cnt) == len(idems)

    regimes: dict[str, dict] = {}
    for conc in concurrencies:
        stop_ev = threading.Event()
        barrier = threading.Barrier(conc + 1)
        stats_per_worker: list[tuple[int, int, int, list[float]]] = [
            (0, 0, 0, []) for _ in range(conc)
        ]

        def worker(wid: int) -> None:
            rng = random.Random(920_000 + conc * 1000 + wid)
            wconn = psycopg.connect(
                host="127.0.0.1",
                port=pgwire_port,
                user="postgres",
                dbname="postgres",
                sslmode="disable",
                autocommit=True,
            )
            cur = wconn.cursor()
            committed = 0
            replayed = 0
            rejected = 0
            lats: list[float] = []
            seq_local = 0
            last_ok_args: tuple[str, str, str, int] | None = None
            barrier.wait()
            while not stop_ev.is_set():
                is_retry = False
                if last_ok_args is not None and rng.random() < 0.05:
                    call_args = last_ok_args
                    is_retry = True
                else:
                    a = rng.randrange(num_accounts)
                    b = (a + rng.randrange(1, num_accounts)) % num_accounts
                    seq_local += 1
                    call_args = (
                        f"pgw-c{conc}-w{wid}-{seq_local}",
                        f"acct_{a:04d}",
                        f"acct_{b:04d}",
                        rng.randint(1, 250),
                    )
                t0 = time.perf_counter()
                try:
                    cur.execute(
                        "SELECT ledger_transfer(%s, %s, %s, %s)",
                        call_args,
                        prepare=True,
                    )
                    cur.fetchone()
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    if is_retry:
                        replayed += 1
                    else:
                        committed += 1
                        last_ok_args = call_args
                except Exception:
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    rejected += 1
            wconn.close()
            stats_per_worker[wid] = (committed, replayed, rejected, lats)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
        for t in threads:
            t.start()
        barrier.wait()
        with Window(meter) as win:
            time.sleep(duration_s)
            stop_ev.set()
            for t in threads:
                t.join()

        tot_c = sum(x[0] for x in stats_per_worker)
        tot_r = sum(x[1] for x in stats_per_worker)
        tot_x = sum(x[2] for x in stats_per_worker)
        all_lats = [lat for x in stats_per_worker for lat in x[3]]
        regimes[f"c={conc}"] = summarize_window(
            tot_c, tot_r, tot_x, all_lats, win, idle_watts, batch_size=1
        )

    # Regime D: pgwire batch settlement (`SELECT ledger_transfer_batch(...)`)
    with psycopg.connect(
        host="127.0.0.1",
        port=pgwire_port,
        user="postgres",
        dbname="postgres",
        sslmode="disable",
        autocommit=True,
    ) as bconn:
        bcur = bconn.cursor()
        rng_b = random.Random(990_002)
        batch_committed = 0
        batch_rejected = 0
        batch_lats: list[float] = []
        batch_idx = 0
        with Window(meter) as bwin:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                batch_idx += 1
                idems: list[str] = []
                debits: list[str] = []
                credits: list[str] = []
                amts: list[int] = []
                for j in range(batch_size):
                    a = (batch_idx * batch_size + j) % num_accounts
                    b = (a + 1 + (j % 7)) % num_accounts
                    idems.append(f"pgw-batch-{batch_idx}-{j}")
                    debits.append(f"acct_{a:04d}")
                    credits.append(f"acct_{b:04d}")
                    amts.append(rng_b.randint(1, 50))
                t0 = time.perf_counter()
                c_cnt = int(
                    bcur.execute(
                        "SELECT ledger_transfer_batch(%s::text[], %s::text[], %s::text[], %s::bigint[])",
                        (idems, debits, credits, amts),
                        prepare=True,
                    ).fetchone()[0]
                )
                batch_lats.append((time.perf_counter() - t0) * 1000.0)
                batch_committed += c_cnt
                batch_rejected += batch_size - c_cnt

    vconn = http.client.HTTPConnection("127.0.0.1", http_port, timeout=30.0)
    vconn.request("GET", "/api/v1/ledger/verify")
    verify = json.loads(vconn.getresponse().read())
    vconn.request("GET", "/api/v1/ledger/accounts")
    accts = json.loads(vconn.getresponse().read())["accounts"]
    vconn.close()
    min_customer_net = min(
        a["posted_credits"] - a["posted_debits"]
        for a in accts
        if a["account_id"].startswith("acct_")
    )
    assert verify["valid"] is True, f"JouleDB pgwire verify_chain failed: {verify}"
    assert verify["total_posted_debits"] == verify["total_posted_credits"]
    assert min_customer_net >= 0

    regimes[f"batch={batch_size}"] = summarize_window(
        batch_committed,
        0,
        batch_rejected,
        batch_lats,
        bwin,
        idle_watts,
        batch_size=batch_size,
    )
    return {
        "engine": "JouleDB v0.16.0 pgwire (psycopg3 + Pre-Allocated JWL1 + SHA-256 Chain + Adaptive FIFO Group Commit, fsync=on)",
        "invariants_verified": {
            "sha256_chain_valid": verify["valid"],
            "journal_entries_verified": verify["journal_entries_verified"],
            "total_posted_debits": verify["total_posted_debits"],
            "total_posted_credits": verify["total_posted_credits"],
            "min_customer_balance_minor": min_customer_net,
        },
        "regimes": regimes,
    }


def run_postgres_banking(
    pg_port: int,
    num_accounts: int,
    concurrencies: list[int],
    duration_s: float,
    batch_size: int,
    meter,
    idle_watts: float | None,
) -> dict:
    import psycopg

    with psycopg.connect(
        host="127.0.0.1",
        port=pg_port,
        user="postgres",
        dbname="postgres",
        sslmode="disable",
        autocommit=True,
    ) as setup:
        pg_version = setup.execute("SELECT version()").fetchone()[0]
        fsync_setting = setup.execute("SHOW fsync").fetchone()[0]
        sync_commit = setup.execute("SHOW synchronous_commit").fetchone()[0]
        setup.execute("DROP TABLE IF EXISTS ledger_legs")
        setup.execute("DROP TABLE IF EXISTS ledger_transfers")
        setup.execute("DROP TABLE IF EXISTS ledger_accounts")
        setup.execute(
            """
            CREATE TABLE ledger_accounts (
                account_id TEXT PRIMARY KEY,
                posted_debits NUMERIC(38, 0) NOT NULL DEFAULT 0,
                posted_credits NUMERIC(38, 0) NOT NULL DEFAULT 0,
                no_overdraft BOOLEAN NOT NULL DEFAULT FALSE,
                CHECK (NOT no_overdraft OR posted_credits >= posted_debits)
            )
            """
        )
        setup.execute(
            """
            CREATE TABLE ledger_transfers (
                seq BIGSERIAL PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                debit_account_id TEXT NOT NULL,
                credit_account_id TEXT NOT NULL,
                amount_minor NUMERIC(38, 0) NOT NULL CHECK (amount_minor > 0),
                entry_sha256 BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Seed accounts
        with setup.cursor().copy(
            "COPY ledger_accounts (account_id, posted_debits, posted_credits, no_overdraft) FROM STDIN"
        ) as cp:
            cp.write_row(("treasury_usd", num_accounts * 1_000_000, 0, False))
            for i in range(num_accounts):
                cp.write_row((f"acct_{i:04d}", 0, 1_000_000, True))

        # Stored PL/pgSQL atomic double-entry transfer function with deterministic lock ordering
        # (locks min(debit, credit) then max(debit, credit) so concurrent workers NEVER deadlock!)
        setup.execute(
            """
            CREATE OR REPLACE FUNCTION pg_submit_transfer(
                p_idem TEXT,
                p_debit TEXT,
                p_credit TEXT,
                p_amt NUMERIC(38, 0)
            ) RETURNS BOOLEAN LANGUAGE plpgsql AS $$
            DECLARE
                v_existing BIGINT;
                v_first TEXT := LEAST(p_debit, p_credit);
                v_second TEXT := GREATEST(p_debit, p_credit);
            BEGIN
                SELECT seq INTO v_existing FROM ledger_transfers WHERE idempotency_key = p_idem;
                IF FOUND THEN
                    RETURN TRUE; -- idempotent replay
                END IF;

                PERFORM 1 FROM ledger_accounts WHERE account_id = v_first FOR UPDATE;
                PERFORM 1 FROM ledger_accounts WHERE account_id = v_second FOR UPDATE;

                UPDATE ledger_accounts SET posted_debits = posted_debits + p_amt WHERE account_id = p_debit;
                UPDATE ledger_accounts SET posted_credits = posted_credits + p_amt WHERE account_id = p_credit;

                INSERT INTO ledger_transfers (idempotency_key, debit_account_id, credit_account_id, amount_minor, entry_sha256)
                VALUES (
                    p_idem,
                    p_debit,
                    p_credit,
                    p_amt,
                    sha256(convert_to(p_idem || '|' || p_debit || '|' || p_credit || '|' || p_amt::text, 'UTF8'))
                );
                RETURN FALSE;
            END;
            $$;
            """
        )

    regimes: dict[str, dict] = {}
    for conc in concurrencies:
        stop_ev = threading.Event()
        barrier = threading.Barrier(conc + 1)
        stats_per_worker: list[tuple[int, int, int, list[float]]] = [
            (0, 0, 0, []) for _ in range(conc)
        ]

        def pg_worker(wid: int) -> None:
            rng = random.Random(910_000 + conc * 1000 + wid)
            con = psycopg.connect(
                host="127.0.0.1",
                port=pg_port,
                user="postgres",
                dbname="postgres",
                sslmode="disable",
                autocommit=True,
            )
            committed = 0
            replayed = 0
            rejected = 0
            lats: list[float] = []
            seq_local = 0
            last_ok: tuple[str, str, str, int] | None = None
            barrier.wait()
            while not stop_ev.is_set():
                if last_ok is not None and rng.random() < 0.05:
                    idem, d_acct, c_acct, amt = last_ok
                else:
                    a = rng.randrange(num_accounts)
                    b = (a + rng.randrange(1, num_accounts)) % num_accounts
                    seq_local += 1
                    idem = f"pg-c{conc}-w{wid}-{seq_local}"
                    d_acct = f"acct_{a:04d}"
                    c_acct = f"acct_{b:04d}"
                    amt = rng.randint(1, 250)
                t0 = time.perf_counter()
                try:
                    row = con.execute(
                        "SELECT pg_submit_transfer(%s, %s, %s, %s)",
                        (idem, d_acct, c_acct, amt),
                        prepare=True,
                    ).fetchone()
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    if row and row[0]:
                        replayed += 1
                    else:
                        committed += 1
                        last_ok = (idem, d_acct, c_acct, amt)
                except Exception:
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    rejected += 1
            con.close()
            stats_per_worker[wid] = (committed, replayed, rejected, lats)

        threads = [threading.Thread(target=pg_worker, args=(w,)) for w in range(conc)]
        for t in threads:
            t.start()
        barrier.wait()
        with Window(meter) as win:
            time.sleep(duration_s)
            stop_ev.set()
            for t in threads:
                t.join()

        tot_c = sum(x[0] for x in stats_per_worker)
        tot_r = sum(x[1] for x in stats_per_worker)
        tot_x = sum(x[2] for x in stats_per_worker)
        all_lats = [lat for x in stats_per_worker for lat in x[3]]
        regimes[f"c={conc}"] = summarize_window(
            tot_c, tot_r, tot_x, all_lats, win, idle_watts, batch_size=1
        )

    # Regime D: Set-based single-transaction `UNNEST` batch CTE in PostgreSQL 16
    bcon = psycopg.connect(
        host="127.0.0.1",
        port=pg_port,
        user="postgres",
        dbname="postgres",
        sslmode="disable",
        autocommit=True,
    )
    batch_sql = """
        WITH raw_in AS (
            SELECT * FROM UNNEST(%s::text[], %s::text[], %s::text[], %s::numeric[])
                AS t(idem, debit_acct, credit_acct, amt)
        ),
        ins AS (
            INSERT INTO ledger_transfers (idempotency_key, debit_account_id, credit_account_id, amount_minor, entry_sha256)
            SELECT idem, debit_acct, credit_acct, amt,
                   sha256(convert_to(idem || '|' || debit_acct || '|' || credit_acct || '|' || amt::text, 'UTF8'))
            FROM raw_in
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING debit_account_id, credit_account_id, amount_minor
        ),
        deltas AS (
            SELECT acct, SUM(d_amt) AS add_d, SUM(c_amt) AS add_c
            FROM (
                SELECT debit_account_id AS acct, amount_minor AS d_amt, 0::numeric AS c_amt FROM ins
                UNION ALL
                SELECT credit_account_id AS acct, 0::numeric AS d_amt, amount_minor AS c_amt FROM ins
            ) u
            GROUP BY acct
        ),
        upd AS (
            UPDATE ledger_accounts a
            SET posted_debits = a.posted_debits + d.add_d,
                posted_credits = a.posted_credits + d.add_c
            FROM deltas d
            WHERE a.account_id = d.acct
            RETURNING 1
        )
        SELECT (SELECT COUNT(*) FROM ins), (SELECT COUNT(*) FROM upd)
    """
    rng_b = random.Random(990_001)
    batch_committed = 0
    batch_rejected = 0
    batch_lats: list[float] = []
    batch_idx = 0
    with Window(meter) as bwin:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            batch_idx += 1
            idems, debits, credits, amts = [], [], [], []
            for j in range(batch_size):
                a = (batch_idx * batch_size + j) % num_accounts
                b = (a + 1 + (j % 7)) % num_accounts
                idems.append(f"pg-batch-{batch_idx}-{j}")
                debits.append(f"acct_{a:04d}")
                credits.append(f"acct_{b:04d}")
                amts.append(rng_b.randint(1, 50))
            t0 = time.perf_counter()
            try:
                row = bcon.execute(
                    batch_sql, (idems, debits, credits, amts), prepare=True
                ).fetchone()
                batch_lats.append((time.perf_counter() - t0) * 1000.0)
                batch_committed += int(row[0])
            except Exception:
                batch_lats.append((time.perf_counter() - t0) * 1000.0)
                batch_rejected += batch_size

    inv = bcon.execute(
        """
        SELECT
            SUM(posted_debits) = SUM(posted_credits),
            SUM(posted_debits)::text,
            MIN(CASE WHEN no_overdraft THEN posted_credits - posted_debits ELSE 0 END)::text,
            (SELECT COUNT(*) FROM ledger_transfers)
        FROM ledger_accounts
        """
    ).fetchone()
    bcon.close()
    assert inv[0] is True and int(inv[2]) >= 0, f"PostgreSQL invariant check failed: {inv}"

    regimes[f"batch={batch_size}"] = summarize_window(
        batch_committed,
        0,
        batch_rejected,
        batch_lats,
        bwin,
        idle_watts,
        batch_size=batch_size,
    )
    return {
        "engine": pg_version,
        "durability": {"fsync": fsync_setting, "synchronous_commit": sync_commit},
        "invariants_verified": {
            "debits_equal_credits": inv[0],
            "total_posted_minor": inv[1],
            "min_customer_balance_minor": int(inv[2]),
            "journal_rows_persisted": int(inv[3]),
        },
        "regimes": regimes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jouledb-bin", required=True)
    ap.add_argument("--work-dir", default="/var/tmp", help="Physical NVMe directory")
    ap.add_argument("--accounts", type=int, default=1000)
    ap.add_argument("--concurrencies", default="1,4,16")
    ap.add_argument("--duration-s", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--http-port", type=int, default=28480)
    ap.add_argument("--pg-port", type=int, default=28532)
    ap.add_argument("--pg-docker-image", default="postgres:16-alpine")
    ap.add_argument("--out", default="ledger_banking_nvme_rapl_v014.json")
    args = ap.parse_args()

    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    work_parent = os.path.abspath(args.work_dir)
    os.makedirs(work_parent, exist_ok=True)
    root_tmp = tempfile.mkdtemp(prefix="jdb_banking_h2h_", dir=work_parent)

    meter = make_meter()
    meter.start()
    energy_tag = (
        "reported_j"
        if meter.source != "none" and not meter.source.startswith("none")
        else "unmetered"
    )

    with Window(meter) as idle_w:
        time.sleep(1.5)
    idle_watts = (idle_w.j / (idle_w.t1 - idle_w.t0)) if idle_w.j is not None else None

    report: dict = {
        "benchmark": "Core Banking Double-Entry Ledger Head-to-Head (Conservation + NO_OVERDRAFT + Idempotency + SHA-256 + fsync=on)",
        "version": "0.16.0",
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "storage_dir": work_parent,
        "accounts": args.accounts,
        "concurrencies": concs,
        "batch_size": args.batch_size,
        "duration_per_regime_s": args.duration_s,
        "energy_source": meter.source,
        "energy_evidence": energy_tag,
        "idle_watts": round(idle_watts, 2) if idle_watts is not None else None,
        "engines": {},
    }

    try:
        # 1A. JouleDB v0.16.0 (HTTP REST API)
        jdb_dir = os.path.join(root_tmp, "jouledb_data")
        os.makedirs(jdb_dir, exist_ok=True)
        jdb_proc = subprocess.Popen(
            [args.jouledb_bin, "-p", str(args.http_port), "-d", jdb_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "JOULEDB_DISABLE_RATE_LIMIT": "1"},
        )
        try:
            if not wait_http(args.http_port, 20.0):
                raise RuntimeError("JouleDB failed to start HTTP listener")
            report["engines"]["jouledb"] = run_jouledb_banking(
                args.http_port,
                args.accounts,
                concs,
                args.duration_s,
                args.batch_size,
                meter,
                idle_watts,
            )
        finally:
            jdb_proc.terminate()
            try:
                jdb_proc.wait(timeout=5)
            except Exception:
                jdb_proc.kill()

        # 1B. JouleDB v0.16.0 (Native pgwire + psycopg3 — exact wire protocol parity with PostgreSQL 16)
        jdb_pg_dir = os.path.join(root_tmp, "jouledb_pgwire_data")
        os.makedirs(jdb_pg_dir, exist_ok=True)
        jdb_pgwire_port = args.http_port + 2
        jdb_pg_proc = subprocess.Popen(
            [args.jouledb_bin, "-p", str(args.http_port), "-d", jdb_pg_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "JOULEDB_DISABLE_RATE_LIMIT": "1"},
        )
        try:
            if not wait_http(args.http_port, 20.0) or not wait_pg_ready("127.0.0.1", jdb_pgwire_port, 20.0):
                raise RuntimeError("JouleDB failed to start pgwire listener")
            report["engines"]["jouledb_pgwire"] = run_jouledb_pgwire_banking(
                jdb_pgwire_port,
                args.http_port,
                args.accounts,
                concs,
                args.duration_s,
                args.batch_size,
                meter,
                idle_watts,
            )
        finally:
            jdb_pg_proc.terminate()
            try:
                jdb_pg_proc.wait(timeout=5)
            except Exception:
                jdb_pg_proc.kill()

        # 2. PostgreSQL 16 (fsync=on, synchronous_commit=on)
        pg_bin = find_pg_bindir()
        if pg_bin:
            pg_dir = os.path.join(root_tmp, "pg_data")
            subprocess.run(
                [os.path.join(pg_bin, "initdb"), "-D", pg_dir, "-A", "trust", "-U", "postgres"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pg_proc = subprocess.Popen(
                [
                    os.path.join(pg_bin, "postgres"),
                    "-D",
                    pg_dir,
                    "-p",
                    str(args.pg_port),
                    "-k",
                    root_tmp,
                    "-h",
                    "127.0.0.1",
                    "-c",
                    "fsync=on",
                    "-c",
                    "synchronous_commit=on",
                    "-c",
                    "max_connections=100",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                if not wait_pg_ready("127.0.0.1", args.pg_port, 25.0):
                    raise RuntimeError("PostgreSQL native failed to start")
                report["engines"]["postgresql"] = run_postgres_banking(
                    args.pg_port,
                    args.accounts,
                    concs,
                    args.duration_s,
                    args.batch_size,
                    meter,
                    idle_watts,
                )
            finally:
                pg_proc.terminate()
                try:
                    pg_proc.wait(timeout=5)
                except Exception:
                    pg_proc.kill()
        elif docker_available():
            pg_dir = os.path.join(root_tmp, "pg_docker_data")
            os.makedirs(pg_dir, mode=0o777, exist_ok=True)
            cname = f"jdb_bank_pg16_{os.getpid()}"
            subprocess.run(
                ["docker", "pull", "-q", args.pg_docker_image],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    cname,
                    "--network",
                    "host",
                    "-e",
                    "POSTGRES_HOST_AUTH_METHOD=trust",
                    "-e",
                    "PGDATA=/var/lib/postgresql/data/pgdata",
                    "-v",
                    f"{pg_dir}:/var/lib/postgresql/data",
                    args.pg_docker_image,
                    "-p",
                    str(args.pg_port),
                    "-h",
                    "127.0.0.1",
                    "-c",
                    "fsync=on",
                    "-c",
                    "synchronous_commit=on",
                    "-c",
                    "max_connections=100",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                if not wait_pg_ready("127.0.0.1", args.pg_port, 30.0):
                    raise RuntimeError("PostgreSQL Docker failed to start")
                report["engines"]["postgresql"] = run_postgres_banking(
                    args.pg_port,
                    args.accounts,
                    concs,
                    args.duration_s,
                    args.batch_size,
                    meter,
                    idle_watts,
                )
            finally:
                subprocess.run(
                    ["docker", "rm", "-f", cname],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{pg_dir}:/mnt",
                        args.pg_docker_image,
                        "sh",
                        "-c",
                        "rm -rf /mnt/*",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        if "jouledb" in report["engines"] and "postgresql" in report["engines"]:
            j_http = report["engines"]["jouledb"]["regimes"]
            j_pgw = report["engines"].get("jouledb_pgwire", {}).get("regimes", {})
            p_reg = report["engines"]["postgresql"]["regimes"]
            ratios = {}
            for k in j_http:
                if k in p_reg:
                    jtps_http = j_http[k]["effective_transfers_per_sec"]
                    jtps_pgw = j_pgw.get(k, {}).get("effective_transfers_per_sec")
                    ptps = max(p_reg[k]["effective_transfers_per_sec"], 0.1)
                    jmj_http = j_http[k]["mj_per_transfer"]
                    jmj_pgw = j_pgw.get(k, {}).get("mj_per_transfer")
                    pmj = p_reg[k]["mj_per_transfer"]
                    best_tps = max(jtps_http, jtps_pgw or 0.0)
                    best_mj = min(
                        [m for m in (jmj_http, jmj_pgw) if m is not None and m > 0],
                        default=None,
                    )
                    ratios[k] = {
                        "jouledb_http_tps": jtps_http,
                        "jouledb_pgwire_tps": jtps_pgw,
                        "postgresql_tps": ptps,
                        "http_speedup_vs_pg": round(jtps_http / ptps, 2),
                        "pgwire_speedup_vs_pg": round(jtps_pgw / ptps, 2)
                        if jtps_pgw is not None
                        else None,
                        "best_speedup_vs_pg": round(best_tps / ptps, 2),
                        "jouledb_http_mj_per_tx": jmj_http,
                        "jouledb_pgwire_mj_per_tx": jmj_pgw,
                        "postgresql_mj_per_tx": pmj,
                        "energy_reduction_vs_pg_x": round(pmj / best_mj, 2)
                        if (best_mj and pmj and best_mj > 0)
                        else None,
                    }
            report["head_to_head_summary"] = ratios
    finally:
        meter.stop()
        shutil.rmtree(root_tmp, ignore_errors=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
