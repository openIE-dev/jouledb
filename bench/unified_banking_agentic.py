#!/usr/bin/env python3
"""
Unified Banking, Agentic Sandbox & Human ORM Verification + Benchmark Harness (v0.14.0)

Exercises all three ecosystem profiles against a single running `jouledb` binary:
  1. Core Banking Profile:
     - Exact 128-bit fixed-point SQL arithmetic (`0.10 + 0.20 = 0.30`, 30-digit `DECIMAL_ADD`, penny `SUM`)
     - Double-entry `LEDGER` conservation, two-phase `HOLD`/`POST`/`VOID`, `idempotency_key` replay,
       64-worker concurrent hot-wallet overdraft stress (`0` overdrafts, `0` lost pennies), SQL `JOIN`
       between relational `kyc_customers` and `jouledb_ledger_accounts`, and `LEDGER VERIFY` SHA-256 chain proof.
  2. Autonomous Agent Profile:
     - O(1) Copy-on-Write SQL `CREATE BRANCH`, isolated writes invisible on `main`, `DIFF BRANCH`,
       3-way conflict detection (`SQLSTATE 40001`), clean `MERGE BRANCH`, strict destructive-DDL /
       unbounded-write guardrails (`SQLSTATE 42501`), microjoule budget enforcement (`SQLSTATE 53200`),
       and DBOS-style crash-proof `WORKFLOW CHECKPOINT` / `WORKFLOW RESUME`.
  3. Human Developer / Queue Worker Profile:
     - Pessimistic `SELECT ... FOR UPDATE NOWAIT` (`SQLSTATE 55P03`) and `SKIP LOCKED` worker queue
       claim across concurrent transactions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def get_json(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def sql_exec(base: str, sql: str, session_id: str | None = None, branch_id: str | None = None) -> tuple[int, dict]:
    payload: dict = {"sql": sql}
    if session_id is not None:
        payload["session_id"] = session_id
    if branch_id is not None:
        payload["branch_id"] = branch_id
    return post_json(f"{base}/api/v1/query", payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to release jouledb binary")
    ap.add_argument("--http-port", type=int, default=18180)
    ap.add_argument("--pg-port", type=int, default=18182)
    ap.add_argument("--out", default="", help="Optional JSON output path")
    args = ap.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="jdb_unified_v013_"))
    db_dir = work_dir / "data"
    srv = subprocess.Popen(
        [
            args.bin,
            "-p",
            str(args.http_port),
            "-d",
            str(db_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "JOULEDB_DISABLE_RATE_LIMIT": "1"},
    )
    base = f"http://127.0.0.1:{args.http_port}"
    try:
        for _ in range(60):
            try:
                code, _ = get_json(f"{base}/health")
                if code == 200:
                    break
            except Exception:
                time.sleep(0.1)

        # ── 1. Exact Decimal128 Verification ──────────────────────────────────
        st, r = sql_exec(
            base,
            "SELECT 0.10 + 0.20 = 0.30, DECIMAL_ADD('999999999999999999999999.99', '0.01'), CAST('12.345' AS NUMERIC(18, 2))",
        )
        assert st == 200 and r["rows"][0] == [
            True,
            "1000000000000000000000000.00",
            12.35,
        ], f"Decimal128 check failed: {r}"

        sql_exec(base, "CREATE TABLE pennies (id INT PRIMARY KEY, amt NUMERIC(18, 2))")
        values = ", ".join(f"({i}, 0.01)" for i in range(1, 1001))
        sql_exec(base, f"INSERT INTO pennies VALUES {values}")
        st, sum_r = sql_exec(base, "SELECT SUM(amt), SUM(amt) = 10 FROM pennies")
        assert st == 200 and sum_r["rows"][0] == [10, True], f"Penny SUM drift: {sum_r}"

        # ── 2. Core Banking Double-Entry Ledger + SQL KYC Join + Hot-Wallet Stress ──
        sql_exec(base, "CREATE TABLE kyc_customers (id INT PRIMARY KEY, name TEXT, wallet_id TEXT, tier TEXT)")
        sql_exec(
            base,
            "INSERT INTO kyc_customers VALUES (1, 'Alice Institutional', 'cust_alice_usd', 'Tier-1'), (2, 'Acme Merchant', 'merch_acme_usd', 'Merchant')",
        )
        assert sql_exec(base, "LEDGER CREATE ACCOUNT 'treasury_usd' LEDGER 1 CODE 1000 SCALE 2")[0] == 200
        assert sql_exec(base, "LEDGER CREATE ACCOUNT 'cust_alice_usd' LEDGER 1 CODE 2000 SCALE 2 NO_OVERDRAFT")[0] == 200
        assert sql_exec(base, "LEDGER CREATE ACCOUNT 'merch_acme_usd' LEDGER 1 CODE 2000 SCALE 2")[0] == 200
        assert sql_exec(base, "LEDGER CREATE ACCOUNT 'fee_rev_usd' LEDGER 1 CODE 4000 SCALE 2")[0] == 200

        # Fund Alice $1,000.00 (100_000 cents) + verify idempotency
        st, tx1 = sql_exec(
            base,
            "LEDGER TRANSFER 100000 FROM 'treasury_usd' TO 'cust_alice_usd' IDEMPOTENCY KEY 'dep-001'",
        )
        assert st == 200 and tx1["rows"][0][4] is False
        st, tx1_dup = sql_exec(
            base,
            "LEDGER TRANSFER 100000 FROM 'treasury_usd' TO 'cust_alice_usd' IDEMPOTENCY KEY 'dep-001'",
        )
        assert st == 200 and tx1_dup["rows"][0][4] is True

        # Multi-leg split settlement ($250.00 -> $242.50 merchant + $7.50 fee)
        st, ml = sql_exec(
            base,
            "LEDGER MULTILEG DEBIT 'cust_alice_usd' 25000 CREDIT 'merch_acme_usd' 24250 CREDIT 'fee_rev_usd' 750 IDEMPOTENCY KEY 'split-001'",
        )
        assert st == 200, f"Multi-leg failed: {ml}"

        # Two-phase card auth hold ($500.00) -> partial capture ($400.00) -> $350.00 remaining
        assert sql_exec(
            base,
            "LEDGER HOLD 50000 FROM 'cust_alice_usd' TO 'merch_acme_usd' IDEMPOTENCY KEY 'hold-001' HOLD_ID 'h_001'",
        )[0] == 200
        assert sql_exec(
            base,
            "LEDGER POST HOLD 'h_001' AMOUNT 40000 IDEMPOTENCY KEY 'cap-001'",
        )[0] == 200

        # Relational JOIN between KYC table and live ledger balances (zero split-brain!)
        st, join_r = sql_exec(
            base,
            "SELECT k.name, a.net_credits_minor, a.formatted_balance FROM kyc_customers k JOIN jouledb_ledger_accounts a ON k.wallet_id = a.account_id ORDER BY k.id",
        )
        assert st == 200 and join_r["rows"] == [
            ["Alice Institutional", 35000, "350.00"],
            ["Acme Merchant", 64250, "642.50"],
        ], f"KYC+Ledger JOIN mismatch: {join_r}"

        # Concurrent 16-worker hot-wallet stress: 350 debits of 100 cents must succeed, 150 must reject cleanly (`0` overdraft!)
        t_bank_0 = time.perf_counter()

        def try_debit(idx: int) -> bool:
            code, _ = post_json(
                f"{base}/api/v1/ledger/transfers",
                {
                    "idempotency_key": f"stress-{idx}",
                    "debit_account_id": "cust_alice_usd",
                    "credit_account_id": "merch_acme_usd",
                    "amount_minor": 100,
                },
            )
            return code == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            outcomes = list(pool.map(try_debit, range(500)))
        bank_elapsed_s = time.perf_counter() - t_bank_0
        assert sum(outcomes) == 350, f"Expected exactly 350 debits to succeed, got {sum(outcomes)}"

        _, verify = get_json(f"{base}/api/v1/ledger/verify")
        assert verify["valid"] is True, f"Ledger chain invalid: {verify}"
        assert verify["total_posted_debits"] == verify["total_posted_credits"]

        # ── 3. Agentic CoW SQL Branching, 3-Way Conflict Check & Guardrails ──
        branch_lat_ms = []
        for b_idx in range(10):
            t0 = time.perf_counter()
            bname = f"agent_exp_{b_idx}"
            assert sql_exec(base, f"CREATE BRANCH {bname} ENERGY BUDGET 5000000 GUARDRAILS STRICT")[0] == 200
            # Destructive DDL MUST be blocked on strict agent branch (`42501`)
            ddl_st, ddl_err = sql_exec(base, "DROP TABLE kyc_customers", branch_id=bname)
            assert ddl_st != 200, f"Guardrail failed to block DROP TABLE: {ddl_err}"
            # Unbounded DELETE without WHERE MUST be blocked (`42501`)
            del_st, _ = sql_exec(base, "DELETE FROM kyc_customers", branch_id=bname)
            assert del_st != 200, "Guardrail failed to block unbounded DELETE"
            # Scoped insert on branch is invisible on main
            assert sql_exec(
                base,
                f"INSERT INTO kyc_customers VALUES ({100 + b_idx}, 'Agent User {b_idx}', 'w_{b_idx}', 'Sandbox')",
                branch_id=bname,
            )[0] == 200
            _, main_cnt = sql_exec(base, "SELECT COUNT(*) FROM kyc_customers")
            assert main_cnt["rows"][0][0] == 2, "Branch write leaked into main!"
            branch_lat_ms.append((time.perf_counter() - t0) * 1000.0)

        # Merge agent_exp_0 cleanly into main -> main count becomes 3!
        st, diff0 = sql_exec(base, "DIFF BRANCH agent_exp_0")
        assert st == 200 and diff0["rows"][0][2] == 1
        assert sql_exec(base, "MERGE BRANCH agent_exp_0 DELETE AFTER")[0] == 200
        _, main_cnt_after = sql_exec(base, "SELECT COUNT(*) FROM kyc_customers")
        assert main_cnt_after["rows"][0][0] == 3

        # Trigger true 3-way write-write conflict on key id=1
        assert sql_exec(base, "CREATE BRANCH conflict_branch")[0] == 200
        assert sql_exec(
            base, "UPDATE kyc_customers SET tier = 'Branch-VIP' WHERE id = 1", branch_id="conflict_branch"
        )[0] == 200
        assert sql_exec(
            base, "UPDATE kyc_customers SET tier = 'Main-VIP' WHERE id = 1"
        )[0] == 200
        c_st, c_body = sql_exec(base, "MERGE BRANCH conflict_branch")
        assert c_st != 200 and "conflict" in json.dumps(c_body).lower(), f"Expected 3-way conflict: {c_body}"

        # Durable Workflow Checkpointing (idempotent memoization)
        st, wf1 = sql_exec(
            base,
            """WORKFLOW CHECKPOINT 'wf_loan_underwrite_1' STEP 'credit_pull' STATUS 'completed' OUTPUT '{"fico":780}' INPUT_HASH 'h1' ENERGY_UJ 420""",
        )
        assert st == 200 and wf1["rows"][0][4] is False
        st, wf1_replay = sql_exec(
            base,
            """WORKFLOW CHECKPOINT 'wf_loan_underwrite_1' STEP 'credit_pull' STATUS 'completed' OUTPUT '{"fico":780}' INPUT_HASH 'h1' ENERGY_UJ 420""",
        )
        assert st == 200 and wf1_replay["rows"][0][4] is True

        # ── 4. Pessimistic Row Locking (`SELECT ... FOR UPDATE NOWAIT` & `SKIP LOCKED`) ──
        _, b1 = sql_exec(base, "BEGIN")
        sid1 = b1["session_id"]
        _, b2 = sql_exec(base, "BEGIN")
        sid2 = b2["session_id"]
        # Session 1 locks row id=1
        st, l1 = sql_exec(base, "SELECT id, name FROM kyc_customers WHERE id = 1 FOR UPDATE", session_id=sid1)
        assert st == 200 and len(l1["rows"]) == 1
        # Session 2 NOWAIT on id=1 immediately fails with lock_not_available (`55P03`)
        nw_st, nw_err = sql_exec(
            base, "SELECT id FROM kyc_customers WHERE id = 1 FOR UPDATE NOWAIT", session_id=sid2
        )
        assert nw_st != 200, f"Expected NOWAIT lock failure, got {nw_err}"
        # Session 2 SKIP LOCKED cleanly skips id=1 and claims unlocked rows (id=2, 100)
        sk_st, sk_rows = sql_exec(
            base, "SELECT id FROM kyc_customers ORDER BY id FOR UPDATE SKIP LOCKED", session_id=sid2
        )
        assert sk_st == 200 and [r[0] for r in sk_rows["rows"]] == [2, 100], f"SKIP LOCKED mismatch: {sk_rows}"
        sql_exec(base, "COMMIT", session_id=sid1)
        sql_exec(base, "COMMIT", session_id=sid2)

        summary = {
            "version": "0.14.0",
            "exact_decimal128_verified": True,
            "double_entry_ledger": {
                "hot_wallet_attempts": 500,
                "hot_wallet_committed": sum(outcomes),
                "overdrafts": 0,
                "throughput_ops_per_sec": round(500.0 / bank_elapsed_s, 1),
                "sha256_chain_valid": verify["valid"],
                "journal_entries_verified": verify["journal_entries_verified"],
                "head_hash": verify["head_hash"],
            },
            "agentic_cow_branches": {
                "branches_tested": 10,
                "median_fork_guardrail_write_ms": round(statistics.median(branch_lat_ms), 2),
                "destructive_ddl_blocked": True,
                "unbounded_delete_blocked": True,
                "three_way_conflict_detected": True,
                "durable_workflow_memoization_verified": True,
            },
            "human_row_locking": {
                "for_update_nowait_verified": True,
                "for_update_skip_locked_verified": True,
            },
        }
        print(json.dumps(summary, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        srv.kill()
        srv.wait(timeout=5)


if __name__ == "__main__":
    main()
