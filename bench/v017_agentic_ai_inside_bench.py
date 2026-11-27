#!/usr/bin/env python3
"""
JouleDB v0.17.0 Empirical Full-Stack Unification Benchmark Suite
================================================================
Measures live end-to-end metrics across the 5 v0.17.0 pillars against a running
release `jouledb` binary:
  1. `JouleRoute` RTK Context Compression + 4-Tier Quota Failover + Tier-0 LSN LangCache
  2. `Harness Super-Adapters` (`Pi` + `OpenCode` + `Hermes` Rank-32 -> Rank-96 Concat + `.jsyn` Fusion)
  3. `Kinetic Ontology` (`0.55 ms`-class `CREATE BRANCH` Dry-Run vs Atomic `FinancialLedger` Commit + Scoped `Kiro` Powers)
  4. `LanceDB Blob v2` Random-Access Byte-Range Seek vs Full-Shard Read + In-SQL `AI_RANK` / `AI_FILTER` / `AI_EMBED`
  5. `OpenCode` Pre-Execution SQL/Ontology LSP Diagnostics Latency
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict


def free_port() -> int:
    for base in range(24100, 48000, 10):
        ok = True
        for offset in range(6):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", base + offset)) == 0:
                    ok = False
                    break
        if ok:
            return base
    raise RuntimeError("could not find 6 consecutive free localhost ports")


def http_json(method: str, url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "target" / "release" / "jouledb"
    if not bin_path.exists():
        raise SystemExit(f"Release binary not found at {bin_path}; build `cargo build --release --bin jouledb` first.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="jouledb_v017_bench_"))
    http_port = free_port()
    base_url = f"http://127.0.0.1:{http_port}"

    env = os.environ.copy()
    env["JOULEDB_AUTH_ENABLED"] = "false"
    proc = subprocess.Popen(
        [
            str(bin_path),
            "--port",
            str(http_port),
            "--data",
            str(tmp_dir / "data"),
            "--no-jwp",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for health
        for _ in range(80):
            try:
                http_json("GET", f"{base_url}/health")
                break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("jouledb server did not become healthy in time")

        results: Dict[str, Any] = {
            "suite": "jouledb_v017_unification_bench",
            "version": "0.17.0",
            "energy_measurement_class": "analytical_est",
        }

        # ── 1. RTK Context Pruning + 4-Tier Failover + Tier-0 LangCache ──
        noisy_log = (
            "\x1b[32mCompiling tokio v1.44.0\x1b[0m\n" * 40
            + "Downloading serde v1.0.219\n" * 30
            + "========================================================\n" * 12
            + "retry_poll_status: waiting on upstream worker...\n" * 35
            + "```sql\nSELECT account_id, net_credits FROM ledger_accounts WHERE currency = 'USD';\n```\n"
            + "error[23514]: insufficient available credits on account 101"
        )
        t0 = time.perf_counter()
        rtk = http_json("POST", f"{base_url}/api/v1/router/rtk/compress", {"text": noisy_log})
        rtk_us = (time.perf_counter() - t0) * 1e6

        http_json(
            "POST",
            f"{base_url}/api/v1/router/providers/register",
            {
                "provider_id": "sub-claude-pro",
                "vendor": "anthropic-sub",
                "model_name": "jouledb-cascade-v1",
                "tier": "o_auth_subscription",
                "api_key": "sk-ant-oat01-bench-key-7777",
                "rpm_limit": 60,
            },
        )
        http_json("POST", f"{base_url}/api/v1/router/providers/sub-claude-pro/report-429", {})

        t_miss = time.perf_counter()
        c_miss = http_json(
            "POST",
            f"{base_url}/v1/router/chat/completions",
            {"messages": [{"role": "user", "content": noisy_log}], "bypass_lang_cache": True},
        )
        miss_ms = (time.perf_counter() - t_miss) * 1e3

        t_hit = time.perf_counter()
        c_hit = http_json(
            "POST",
            f"{base_url}/v1/router/chat/completions",
            {"messages": [{"role": "user", "content": noisy_log}], "bypass_lang_cache": False},
        )
        hit_ms = (time.perf_counter() - t_hit) * 1e3

        results["pillar1_joule_route"] = {
            "rtk_raw_tokens": rtk["raw_est_tokens"],
            "rtk_compressed_tokens": rtk["compressed_est_tokens"],
            "rtk_reduction_pct": round(rtk["reduction_pct"], 2),
            "rtk_latency_us": round(rtk_us, 1),
            "failover_cold_ms": round(miss_ms, 3),
            "tier0_langcache_hit_ms": round(hit_ms, 3),
            "selected_failover_provider": c_miss["joule_route"]["selected_provider"],
            "tier0_hit_tier": c_hit["joule_route"]["selected_tier"],
        }

        # ── 2. Harness Super-Adapter (3 x Rank-32 -> Rank-96 Concat + .jsyn Fusion) ──
        rows, cols = 256, 256
        for aid, hname, seed in [
            ("pi-r32", "pi", 101),
            ("opencode-r32", "opencode", 202),
            ("hermes-r32", "hermes", 303),
        ]:
            http_json(
                "POST",
                f"{base_url}/api/v1/synapse/super-adapters/register",
                {
                    "adapter_id": aid,
                    "harness_name": hname,
                    "rows": rows,
                    "cols": cols,
                    "rank": 32,
                    "lora_alpha": 64.0,
                    "seed": seed,
                },
            )

        t_comp = time.perf_counter()
        sup = http_json(
            "POST",
            f"{base_url}/api/v1/synapse/super-adapters/compose",
            {
                "super_adapter_id": "qwen38-super-r96",
                "adapter_ids": ["pi-r32", "opencode-r32", "hermes-r32"],
                "mode": "rank_concat",
                "cosine_threshold": 0.15,
            },
        )
        comp_ms = (time.perf_counter() - t_comp) * 1e3
        rec = sup["receipt"]
        results["pillar2_super_adapter"] = {
            "matrix_shape": f"{rows}x{cols}",
            "constituent_ranks": [32, 32, 32],
            "super_adapter_rank": rec["resulting_rank"],
            "max_pairwise_frobenius_cosine": round(rec["pre_audit"]["max_pairwise_cosine"], 6),
            "mean_pairwise_frobenius_cosine": round(rec["pre_audit"]["mean_pairwise_cosine"], 6),
            "all_pairs_orthogonal": rec["pre_audit"]["all_pairs_orthogonal"],
            "audit_and_concat_ms": round(comp_ms, 3),
        }

        # ── 3 & 5. Kinetic Ontology Dry-Run vs Commit + Kiro Power Savings + OpenCode LSP ──
        http_json(
            "POST",
            f"{base_url}/api/v1/query",
            {"sql": "CREATE TABLE treasury_transfers (tx_id INTEGER PRIMARY KEY, status TEXT, cents INTEGER, note TEXT)"},
        )
        for i in range(1, 51):
            note = (
                "urgent vendor wire invoice settlement"
                if i % 5 == 0
                else "routine office stationery expense"
            )
            http_json(
                "POST",
                f"{base_url}/api/v1/query",
                {"sql": f"INSERT INTO treasury_transfers VALUES ({i}, 'PENDING', {i * 100}, '{note}')"},
            )
        http_json(
            "POST",
            f"{base_url}/api/v1/ledger/accounts",
            {"account_id": "1", "ledger_id": 1, "code": 100, "scale": 2},
        )
        http_json(
            "POST",
            f"{base_url}/api/v1/ledger/accounts",
            {"account_id": "2", "ledger_id": 1, "code": 200, "scale": 2},
        )
        http_json(
            "POST",
            f"{base_url}/api/v1/ontology/objects",
            {
                "object_type": "TreasuryTransfer",
                "backing_table": "treasury_transfers",
                "primary_key": "tx_id",
                "domain_power": "treasury",
                "description": "Governed corporate treasury transfer",
                "properties": {"tx_id": "int", "status": "string", "cents": "int"},
            },
        )
        http_json(
            "POST",
            f"{base_url}/api/v1/ontology/actions",
            {
                "action_id": "execute_treasury_wire",
                "domain_power": "treasury",
                "target_object": "TreasuryTransfer",
                "description": "Governed EARS-checked treasury settlement",
                "parameters": {"tx_id": "int", "cents": "int", "src": "int", "dst": "int"},
                "required_roles": ["treasurer"],
                "ears_preconditions": [
                    "WHEN :cents > 1000000 THE SYSTEM SHALL REJECT WITH 23514: exceeds 10k USD cap"
                ],
                "sql_mutations": [
                    "UPDATE treasury_transfers SET status = 'SETTLED', cents = :cents WHERE tx_id = :tx_id"
                ],
                "ledger_hook": {
                    "debit_account_param": "src",
                    "credit_account_param": "dst",
                    "amount_minor_param": "cents",
                    "currency": "USD",
                    "code": 1,
                },
            },
        )

        dry_lats = []
        for _ in range(10):
            t_d = time.perf_counter()
            http_json(
                "POST",
                f"{base_url}/api/v1/ontology/actions/execute_treasury_wire/execute",
                {
                    "actor_id": "kiro-spec-agent",
                    "actor_roles": ["treasurer"],
                    "arguments": {"tx_id": 5, "cents": 5000, "src": 1, "dst": 2},
                    "dry_run_on_branch": True,
                },
            )
            dry_lats.append((time.perf_counter() - t_d) * 1e3)

        t_live = time.perf_counter()
        live_act = http_json(
            "POST",
            f"{base_url}/api/v1/ontology/actions/execute_treasury_wire/execute",
            {
                "actor_id": "kiro-spec-agent",
                "actor_roles": ["treasurer"],
                "arguments": {"tx_id": 5, "cents": 5000, "src": 1, "dst": 2},
                "dry_run_on_branch": False,
            },
        )
        live_ms = (time.perf_counter() - t_live) * 1e3

        power = http_json("GET", f"{base_url}/api/v1/ontology/powers/treasury")
        t_lsp = time.perf_counter()
        diag = http_json(
            "POST",
            f"{base_url}/api/v1/diagnostics/sql",
            {"sql": "UPDATE treasury_transfers SET status = 'BYPASS'"},
        )
        lsp_ms = (time.perf_counter() - t_lsp) * 1e3

        results["pillar3_and_5_ontology_kiro_opencode"] = {
            "branch_dry_run_median_ms": round(sorted(dry_lats)[len(dry_lats) // 2], 3),
            "live_action_plus_ledger_ms": round(live_ms, 3),
            "ledger_transfer_id": live_act["receipt"]["ledger_transfer_id"],
            "scoped_power_prompt_tokens": power["estimated_prompt_tokens"],
            "tokens_saved_vs_59_tool_mcp": power["tokens_saved_vs_full_59_tool_mcp"],
            "prompt_bloat_reduction_pct": round(
                power["tokens_saved_vs_full_59_tool_mcp"] / 14800.0 * 100.0, 2
            ),
            "opencode_sql_lsp_roundtrip_ms": round(lsp_ms, 3),
            "opencode_lsp_codes": [d["code"] for d in diag["diagnostics"]],
        }

        # ── 4. Lance Blob v2 Seek Slice (vs 16 MB Container Read) + In-SQL AI_RANK/AI_FILTER ──
        pack_file = tmp_dir / "multimodal_16mb.pack"
        frame_bytes = b"\x89PNG\r\n\x1a\n" + os.urandom(4096)
        offset = 12 * 1024 * 1024
        with open(pack_file, "wb") as f:
            f.write(b"\x00" * offset)
            f.write(frame_bytes)
            f.write(b"\x00" * (4 * 1024 * 1024))

        sha_hex = hashlib.sha256(frame_bytes).hexdigest()
        http_json(
            "POST",
            f"{base_url}/api/v1/ducklake/tables",
            {
                "schema": "main",
                "table": "media_assets",
                "columns": [["id", "BIGINT", False], ["blob", "BLOB_REF", False]],
                "timestamp_ms": 1000,
            },
        )
        reg = http_json(
            "POST",
            f"{base_url}/api/v1/lance/blobs/register",
            {
                "schema": "main",
                "table": "media_assets",
                "column_name": "blob",
                "row_id": 1,
                "uri": f"file://{pack_file}",
                "byte_offset": offset,
                "byte_length": len(frame_bytes),
                "mime_type": "image/png",
                "sha256_hex": sha_hex,
                "timestamp_ms": 1100,
            },
        )
        blob_id = reg["blob"]["blob_id"]

        t_slice = time.perf_counter()
        fetched = http_bytes(f"{base_url}/api/v1/lance/blobs/{blob_id}/slice")
        slice_ms = (time.perf_counter() - t_slice) * 1e3
        assert fetched == frame_bytes

        t_sql_ai = time.perf_counter()
        ai_q = http_json(
            "POST",
            f"{base_url}/api/v1/query",
            {
                "sql": "SELECT tx_id, AI_RANK('urgent vendor wire settlement', note) AS r FROM treasury_transfers WHERE AI_FILTER(note, 'vendor wire') ORDER BY r DESC LIMIT 5"
            },
        )
        sql_ai_ms = (time.perf_counter() - t_sql_ai) * 1e3

        results["pillar4_ducklake_lance_sql_ai"] = {
            "container_size_bytes": pack_file.stat().st_size,
            "sliced_payload_bytes": len(fetched),
            "io_amplification_avoided_x": round(pack_file.stat().st_size / len(fetched), 1),
            "lance_blob_v2_seek_verify_ms": round(slice_ms, 3),
            "in_sql_ai_rank_filter_50_rows_ms": round(sql_ai_ms, 3),
            "in_sql_matched_rows": len(ai_q["rows"]),
        }

        out_path = repo_root / "bench" / "results" / "v017_unification_bench.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
