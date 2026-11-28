#!/usr/bin/env python3
"""
JouleDB v0.19.0 Empirical Full-Stack Agentic, Rerank, Live-429 & Sub-2ms Lazy CoW Branch Suite
==============================================================================================
Measures live end-to-end metrics against a release `jouledb` binary:
  1. Sub-2ms Lazy CoW Ephemeral SQL Branch Dry-Run Latency (`OnceLock` LSH seed cache +
     zero-DDL deferred-hologram RAM materializer) vs `v0.18.0` (26.135 ms) & `v0.17.0` (34.122 ms)
  2. Live Network `HTTP 429` Catch & Multi-Tier Waterfall Failover (`live_http_dispatch = true`)
     + `RTK` Context Pruning + Tier-0 LSN `LangCache`
  3. Voyage AI / Cohere Rerank v2 Wire-Compatible `POST /v1/rerank` Latency & Throughput
  4. Dynamic Scoped `Kiro` MCP Power (`POST /mcp?power=treasury`) Prompt Token Savings
  5. `Harness Super-Adapters` (`Pi` + `OpenCode` + `Hermes` Rank-32 -> Rank-96 Concat)
     + `LanceDB Blob v2` Random-Access Seek + `OpenCode` Pre-Execution SQL LSP Diagnostics
"""

from __future__ import annotations

import hashlib
import http.client
import http.server
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict


def free_port() -> int:
    for base in range(25100, 48000, 10):
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


def start_mock_upstream(status_code: int, reply_content: str) -> tuple[http.server.HTTPServer, int]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                _ = self.rfile.read(length)
            if status_code == 429:
                body = json.dumps({"error": {"message": "rate limit exceeded"}}).encode()
            else:
                body = json.dumps(
                    {
                        "id": "chatcmpl-mock-ok",
                        "object": "chat.completion",
                        "model": "mock-healthy-v1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": reply_content},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "target" / "release" / "jouledb"
    if not bin_path.exists():
        raise SystemExit(
            f"Release binary not found at {bin_path}; build `cargo build --release --bin jouledb` first."
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="jouledb_v019_bench_"))
    http_port = free_port()
    base_url = f"http://127.0.0.1:{http_port}"

    srv_429, port_429 = start_mock_upstream(429, "")
    srv_200, port_200 = start_mock_upstream(200, "LIVE_FAILOVER_OK")

    env = os.environ.copy()
    env["JOULEDB_AUTH_ENABLED"] = "false"
    env["JOULEDB_DISABLE_RATE_LIMIT"] = "1"
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
        for _ in range(80):
            try:
                http_json("GET", f"{base_url}/health")
                break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("jouledb server did not become healthy in time")

        results: Dict[str, Any] = {
            "suite": "jouledb_v019_sota_agentic_bench",
            "version": "0.19.0",
            "energy_measurement_class": "analytical_est",
        }

        # ── 1. Live HTTP 429 Failover + RTK + Tier-0 LSN LangCache ──
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
                "provider_id": "aaa-tier1-429-upstream",
                "vendor": "mock-sub-primary",
                "model_name": "sub-primary-v1",
                "base_url": f"http://127.0.0.1:{port_429}/v1",
                "tier": "o_auth_subscription",
                "api_key": "sk-sub-primary-7777",
                "live_http_dispatch": True,
                "rpm_limit": 100,
            },
        )
        http_json(
            "POST",
            f"{base_url}/api/v1/router/providers/register",
            {
                "provider_id": "zzz-tier1-healthy-upstream",
                "vendor": "mock-sub-backup",
                "model_name": "sub-backup-v1",
                "base_url": f"http://127.0.0.1:{port_200}/v1",
                "tier": "o_auth_subscription",
                "api_key": "sk-sub-backup-8888",
                "live_http_dispatch": True,
                "rpm_limit": 100,
            },
        )

        t_failover = time.perf_counter()
        c_miss = http_json(
            "POST",
            f"{base_url}/v1/router/chat/completions",
            {"messages": [{"role": "user", "content": noisy_log}], "bypass_lang_cache": True},
        )
        failover_ms = (time.perf_counter() - t_failover) * 1e3

        t_hit = time.perf_counter()
        c_hit = http_json(
            "POST",
            f"{base_url}/v1/router/chat/completions",
            {"messages": [{"role": "user", "content": noisy_log}], "bypass_lang_cache": False},
        )
        hit_ms = (time.perf_counter() - t_hit) * 1e3

        results["pillar1_joule_route_live_failover"] = {
            "rtk_raw_tokens": rtk["raw_est_tokens"],
            "rtk_compressed_tokens": rtk["compressed_est_tokens"],
            "rtk_reduction_pct": round(rtk["reduction_pct"], 2),
            "rtk_latency_us": round(rtk_us, 1),
            "live_http_429_failover_ms": round(failover_ms, 3),
            "tier0_langcache_hit_ms": round(hit_ms, 3),
            "failover_skipped": c_miss["joule_route"]["failover_skipped"],
            "selected_provider": c_miss["joule_route"]["selected_provider"],
            "tier0_hit_tier": c_hit["joule_route"]["selected_tier"],
        }

        # ── 2. Sub-2ms Lazy CoW Ephemeral Branch Dry-Run + SQL Ontology DDL + Scoped MCP Power ──
        for sql in [
            "CREATE TABLE treasury_transfers (tx_id INTEGER PRIMARY KEY, status TEXT, cents INTEGER, note TEXT)",
            "LEDGER CREATE ACCOUNT '1' LEDGER 1 CODE 100 SCALE 2",
            "LEDGER CREATE ACCOUNT '2' LEDGER 1 CODE 200 SCALE 2",
            "CREATE ONTOLOGY OBJECT TreasuryTransfer ON treasury_transfers (PRIMARY KEY tx_id, DOMAIN 'treasury')",
            "CREATE KINETIC ACTION execute_treasury_wire ON TreasuryTransfer DOMAIN 'treasury' ROLE 'treasurer' WHEN 'WHEN :cents > 1000000 THE SYSTEM SHALL REJECT WITH 23514: exceeds 10k USD cap' LEDGER (src, dst, cents, 'USD') AS UPDATE treasury_transfers SET status = 'SETTLED', cents = :cents WHERE tx_id = :tx_id",
        ]:
            http_json("POST", f"{base_url}/api/v1/query", {"sql": sql})

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

        # Warm up `OnceLock` seed state before steady-state measurement
        http_json(
            "POST",
            f"{base_url}/api/v1/ontology/actions/execute_treasury_wire/execute",
            {
                "actor_id": "kiro-spec-agent",
                "actor_roles": ["treasurer"],
                "arguments": {"tx_id": 5, "cents": 5000, "src": "1", "dst": "2"},
                "dry_run_on_branch": True,
            },
        )

        dry_urllib_lats = []
        dry_keepalive_lats = []
        dry_sql_lats = []
        ka_conn = http.client.HTTPConnection("127.0.0.1", http_port, timeout=30)
        ka_body_act = json.dumps(
            {
                "actor_id": "kiro-spec-agent",
                "actor_roles": ["treasurer"],
                "arguments": {"tx_id": 5, "cents": 5000, "src": "1", "dst": "2"},
                "dry_run_on_branch": True,
            }
        ).encode("utf-8")
        ka_body_sql = json.dumps(
            {
                "sql": "EXECUTE ACTION execute_treasury_wire (tx_id = 5, cents = 5000, src = '1', dst = '2') AS 'kiro-sql' ROLE 'treasurer' DRY RUN"
            }
        ).encode("utf-8")

        for _ in range(19):
            t_u = time.perf_counter()
            http_json(
                "POST",
                f"{base_url}/api/v1/ontology/actions/execute_treasury_wire/execute",
                {
                    "actor_id": "kiro-spec-agent",
                    "actor_roles": ["treasurer"],
                    "arguments": {"tx_id": 5, "cents": 5000, "src": "1", "dst": "2"},
                    "dry_run_on_branch": True,
                },
            )
            dry_urllib_lats.append((time.perf_counter() - t_u) * 1e3)

            t_k = time.perf_counter()
            ka_conn.request(
                "POST",
                "/api/v1/ontology/actions/execute_treasury_wire/execute",
                body=ka_body_act,
                headers={"Content-Type": "application/json"},
            )
            _ = ka_conn.getresponse().read()
            dry_keepalive_lats.append((time.perf_counter() - t_k) * 1e3)

            t_s = time.perf_counter()
            ka_conn.request(
                "POST",
                "/api/v1/query",
                body=ka_body_sql,
                headers={"Content-Type": "application/json"},
            )
            _ = ka_conn.getresponse().read()
            dry_sql_lats.append((time.perf_counter() - t_s) * 1e3)
        ka_conn.close()

        t_live = time.perf_counter()
        live_act = http_json(
            "POST",
            f"{base_url}/api/v1/query",
            {
                "sql": "EXECUTE ACTION execute_treasury_wire (tx_id = 5, cents = 5000, src = '1', dst = '2') AS 'kiro-sql' ROLE 'treasurer'"
            },
        )
        live_ms = (time.perf_counter() - t_live) * 1e3

        mcp_scoped = http_json(
            "POST",
            f"{base_url}/mcp?power=treasury",
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        t_lsp = time.perf_counter()
        diag = http_json(
            "POST",
            f"{base_url}/api/v1/diagnostics/sql",
            {"sql": "UPDATE treasury_transfers SET status = 'BYPASS'"},
        )
        lsp_ms = (time.perf_counter() - t_lsp) * 1e3

        med_dry_urllib = sorted(dry_urllib_lats)[len(dry_urllib_lats) // 2]
        med_dry_keepalive = sorted(dry_keepalive_lats)[len(dry_keepalive_lats) // 2]
        med_dry_sql = sorted(dry_sql_lats)[len(dry_sql_lats) // 2]
        results["pillar2_lazy_cow_ram_ontology_and_mcp_power"] = {
            "v017_disk_branch_dry_run_ms": 34.122,
            "v018_pure_ram_dry_run_ms": 26.135,
            "v019_lazy_cow_urllib_dry_run_median_ms": round(med_dry_urllib, 3),
            "v019_lazy_cow_keepalive_dry_run_median_ms": round(med_dry_keepalive, 3),
            "v019_lazy_cow_sql_dry_run_median_ms": round(med_dry_sql, 3),
            "speedup_vs_v017_x": round(34.122 / max(med_dry_urllib, 0.001), 2),
            "speedup_vs_v018_x": round(26.135 / max(med_dry_urllib, 0.001), 2),
            "sql_live_action_plus_ledger_ms": round(live_ms, 3),
            "ledger_transfer_id": live_act["rows"][0][4],
            "mcp_scoped_tool_count": len(mcp_scoped["result"]["tools"]),
            "mcp_scoped_prompt_savings_pct": round(
                mcp_scoped["result"]["_meta"]["jouledb_power"]["savings_pct"], 2
            ),
            "opencode_sql_lsp_roundtrip_ms": round(lsp_ms, 3),
            "opencode_lsp_codes": [d["code"] for d in diag["diagnostics"]],
        }

        # ── 3. Voyage AI / Cohere Rerank v2 (`POST /v1/rerank`) ──
        candidate_docs = [
            f"Document #{i}: general cloud infrastructure telemetry and metrics shard {i}"
            for i in range(64)
        ]
        candidate_docs[19] = (
            "JouleDB Kinetic Ontology enforces EARS preconditions and 128-bit double-entry FinancialLedger transfers."
        )
        rerank_lats = []
        rr_out = {}
        for _ in range(20):
            t_r = time.perf_counter()
            rr_out = http_json(
                "POST",
                f"{base_url}/v1/rerank",
                {
                    "model": "rerank-2.5",
                    "query": "EARS preconditions and double-entry FinancialLedger transfers",
                    "documents": candidate_docs,
                    "top_n": 5,
                    "return_documents": True,
                },
            )
            rerank_lats.append((time.perf_counter() - t_r) * 1e3)

        med_rr_ms = sorted(rerank_lats)[len(rerank_lats) // 2]
        results["pillar3_voyage_cohere_rerank_64docs"] = {
            "candidate_docs": len(candidate_docs),
            "top1_index": rr_out["results"][0]["index"],
            "top1_score": rr_out["results"][0]["relevance_score"],
            "http_roundtrip_median_ms": round(med_rr_ms, 3),
            "engine_elapsed_us": rr_out["joule_receipt"]["elapsed_us"],
            "docs_per_sec": round(len(candidate_docs) / (med_rr_ms / 1e3), 1),
        }

        # ── 4. Super-Adapter Rank-96 Concat + Lance Blob v2 Seek ──
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
                    "rows": 256,
                    "cols": 256,
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

        results["pillar4_super_adapter_and_lance_blob"] = {
            "super_adapter_rank": sup["receipt"]["resulting_rank"],
            "max_pairwise_cosine": round(sup["receipt"]["pre_audit"]["max_pairwise_cosine"], 6),
            "audit_and_concat_ms": round(comp_ms, 3),
            "lance_blob_4kb_seek_ms": round(slice_ms, 3),
        }

        out_dir = repo_root / "bench" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "v019_sota_agentic_bench.json"
        out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        srv_429.shutdown()
        srv_200.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
