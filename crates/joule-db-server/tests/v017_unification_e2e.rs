//! End-to-End Integration Verification for `JouleDB v0.17.0` 5-Pillar Full-Stack Unification:
//! 1. `JouleRoute` 4-Tier Encrypted Quota Waterfall + `HTTP 429` Failover + `RTK` + `LangCache`
//! 2. `Harness Super-Adapters` (`Pi` + `OpenCode` + `Hermes` Rank-96 Concat + `.jsyn` Bitplane Fusion + `AsyncGRPO`)
//! 3. `Kinetic Ontology` (`OBJECT / LINK / ACTION` + `CREATE BRANCH` Dry-Run + `FinancialLedger` + Scoped `Kiro` Powers)
//! 4. `DuckLake` Persistent Catalog + `Lance Namespace` REST + `Blob v2` Zero-Copy Slice + In-SQL `AI_*` Functions
//! 5. `EARS` Executable Spec Compiler + `OpenCode` Pre-Execution SQL/Ontology LSP Diagnostics

use axum::body::Body;
use axum::http::{Request, StatusCode};
use joule_db_server::{Server, ServerConfig};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

async fn post_json(app: &axum::Router, uri: &str, payload: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("build POST request");
    let resp = app.clone().oneshot(req).await.expect("oneshot POST");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    let val = serde_json::from_slice::<Value>(&bytes)
        .unwrap_or_else(|_| json!({ "raw_utf8": String::from_utf8_lossy(&bytes).to_string() }));
    (status, val)
}

async fn get_raw(app: &axum::Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("build GET request");
    let resp = app.clone().oneshot(req).await.expect("oneshot GET");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body")
        .to_vec();
    (status, bytes)
}

#[tokio::test]
async fn test_v017_all_five_pillars_end_to_end() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "jouledb_v017_e2e_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("mkdir");

    let cfg = ServerConfig {
        db_path: tmp_dir.to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    let state = Server::new(cfg).expect("init Server");
    let app = state.router();

    // ── Pillar 1: `JouleRoute` 4-Tier Waterfall + `429` Failover + `RTK` + Tier-0 `LangCache` ──
    let (st, r1) = post_json(
        &app,
        "/api/v1/router/providers/register",
        json!({
            "provider_id": "claude-max-oauth",
            "vendor": "anthropic-sub",
            "model_name": "jouledb-cascade-v1",
            "tier": "o_auth_subscription",
            "api_key": "sk-ant-oat01-secret-subscription-9999",
            "rpm_limit": 30,
            "cost_micro_usd_per_1k_tokens": 0,
            "estimated_uj_per_token": 20
        }),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "register sub provider: {r1}");
    assert_eq!(r1["provider"]["key_preview"], "sk-a...9999");

    // Trip HTTP 429 on Tier 1 (`claude-max-oauth`) -> router must seamlessly fail over to Tier 2 (`local-joulesynapse-jsyn`)!
    let (st429, _) = post_json(
        &app,
        "/api/v1/router/providers/claude-max-oauth/report-429",
        json!({}),
    )
    .await;
    assert_eq!(st429, StatusCode::OK);

    let noisy_prompt = "\x1b[33mCompiling crate_a v0.1.0\x1b[0m\n\
Downloading crate_b v1.0.0\n\
====================================\n\
====================================\n\
Summarize double-entry ledger invariant.";

    let (st_c1, comp1) = post_json(
        &app,
        "/v1/router/chat/completions",
        json!({
            "messages": [{ "role": "user", "content": noisy_prompt }],
            "apply_rtk_compression": true,
            "bypass_lang_cache": true
        }),
    )
    .await;
    assert_eq!(st_c1, StatusCode::OK, "router completion 1: {comp1}");
    assert_eq!(
        comp1["joule_route"]["selected_provider"],
        "local-joulesynapse-jsyn"
    );
    assert!(
        comp1["joule_route"]["rtk_tokens_saved"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    assert!(
        comp1["joule_route"]["failover_skipped"]
            .as_array()
            .expect("skipped array")
            .iter()
            .any(|v| v
                .as_str()
                .unwrap_or("")
                .contains("claude-max-oauth:circuit_open_429"))
    );

    // Second identical call hits Tier 0 LSN-valid `LangCache` (`cost_micro_usd == 0`, `tier0_lsn_lang_cache`)!
    let (st_c2, comp2) = post_json(
        &app,
        "/v1/router/chat/completions",
        json!({
            "messages": [{ "role": "user", "content": noisy_prompt }],
            "apply_rtk_compression": true,
            "bypass_lang_cache": false
        }),
    )
    .await;
    assert_eq!(st_c2, StatusCode::OK);
    assert_eq!(
        comp2["joule_route"]["selected_tier"],
        "tier0_lsn_lang_cache"
    );

    // ── Pillar 2: Orthogonal Rank-Concatenated `Harness Super-Adapters` + `.jsyn` Bitplane Fusion ──
    let dense_64: Vec<f32> = (0..(32 * 32))
        .map(|i| ((i as f32) * 0.09).cos() * 0.25)
        .collect();
    let (st_lb, _) = post_json(
        &app,
        "/api/v1/synapse/littlebit/factorize",
        json!({
            "name": "q_proj_v017",
            "weights": dense_64,
            "rows": 32,
            "cols": 32,
            "rank": 16,
            "itq_iters": 4,
            "max_outliers": 0
        }),
    )
    .await;
    assert_eq!(st_lb, StatusCode::OK);

    for (id, harness, seed) in [
        ("pi-r32", "pi", 101u64),
        ("opencode-r32", "opencode", 202u64),
        ("hermes-r32", "hermes", 303u64),
    ] {
        let (st_reg, _) = post_json(
            &app,
            "/api/v1/synapse/super-adapters/register",
            json!({
                "adapter_id": id,
                "harness_name": harness,
                "rows": 32,
                "cols": 32,
                "rank": 32,
                "lora_alpha": 64.0,
                "seed": seed
            }),
        )
        .await;
        assert_eq!(st_reg, StatusCode::OK);
    }

    let (st_sup, sup_resp) = post_json(
        &app,
        "/api/v1/synapse/super-adapters/compose",
        json!({
            "super_adapter_id": "qwen-super-r96",
            "adapter_ids": ["pi-r32", "opencode-r32", "hermes-r32"],
            "mode": "rank_concat",
            "cosine_threshold": 0.20,
            "fuse_into_littlebit_tensor": "q_proj_v017"
        }),
    )
    .await;
    assert_eq!(st_sup, StatusCode::OK, "compose super adapter: {sup_resp}");
    assert_eq!(sup_resp["receipt"]["resulting_rank"], 96);
    assert_eq!(sup_resp["fused_littlebit_rank"], 16 + 96);

    let (st_grpo, grpo_resp) = post_json(
        &app,
        "/api/v1/synapse/super-adapters/grpo-step",
        json!({
            "episodes": [
                {
                    "episode_id": "ep_pass",
                    "prompt_group_id": "grp_sql_1",
                    "harness_name": "opencode",
                    "prompt": "Transfer 50 USD safely",
                    "steps": [],
                    "final_answer": "done",
                    "raw_reward": 1.0,
                    "grpo_advantage": 0.0
                },
                {
                    "episode_id": "ep_fail",
                    "prompt_group_id": "grp_sql_1",
                    "harness_name": "opencode",
                    "prompt": "Transfer 50 USD safely",
                    "steps": [],
                    "final_answer": "error",
                    "raw_reward": -1.0,
                    "grpo_advantage": 0.0
                }
            ]
        }),
    )
    .await;
    assert_eq!(st_grpo, StatusCode::OK);
    assert!(
        grpo_resp["trl_jsonl"]
            .as_str()
            .unwrap_or("")
            .contains("ep_pass")
    );

    // ── Pillars 3 & 5: Kinetic Ontology + `CREATE BRANCH` Dry-Run + `FinancialLedger` + Powers + LSP ──
    let (st_q1, _) = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "CREATE TABLE wire_requests (wire_id INTEGER PRIMARY KEY, state TEXT, cents INTEGER, memo TEXT)"
        }),
    )
    .await;
    assert_eq!(st_q1, StatusCode::OK);
    let _ = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "INSERT INTO wire_requests VALUES (7, 'QUEUED', 2500, 'vendor invoice settlement')"
        }),
    )
    .await;
    let _ = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "INSERT INTO wire_requests VALUES (8, 'QUEUED', 900, 'quantum physics paper subscription')"
        }),
    )
    .await;

    // Create double-entry ledger accounts
    let (st_la1, _) = post_json(
        &app,
        "/api/v1/ledger/accounts",
        json!({ "account_id": "10", "ledger_id": 1, "code": 100, "scale": 2 }),
    )
    .await;
    assert_eq!(st_la1, StatusCode::CREATED);
    let (st_la2, _) = post_json(
        &app,
        "/api/v1/ledger/accounts",
        json!({ "account_id": "20", "ledger_id": 1, "code": 200, "scale": 2 }),
    )
    .await;
    assert_eq!(st_la2, StatusCode::CREATED);

    let (st_obj, _) = post_json(
        &app,
        "/api/v1/ontology/objects",
        json!({
            "object_type": "WireRequest",
            "backing_table": "wire_requests",
            "primary_key": "wire_id",
            "domain_power": "payments",
            "description": "Governed outbound wire request",
            "properties": { "wire_id": "int", "state": "string", "cents": "int" }
        }),
    )
    .await;
    assert_eq!(st_obj, StatusCode::OK);

    let (st_act, _) = post_json(
        &app,
        "/api/v1/ontology/actions",
        json!({
            "action_id": "settle_wire",
            "domain_power": "payments",
            "target_object": "WireRequest",
            "description": "Settle a wire request with EARS compliance cap and atomic double-entry transfer",
            "parameters": {
                "wire_id": "int",
                "cents": "int",
                "debit_id": "int",
                "credit_id": "int"
            },
            "required_roles": ["payments_officer"],
            "ears_preconditions": [
                "WHEN :cents > 50000 THE SYSTEM SHALL REJECT WITH 23514: wire exceeds automated settlement ceiling"
            ],
            "sql_mutations": [
                "UPDATE wire_requests SET state = 'SETTLED', cents = :cents WHERE wire_id = :wire_id"
            ],
            "ledger_hook": {
                "debit_account_param": "debit_id",
                "credit_account_param": "credit_id",
                "amount_minor_param": "cents",
                "currency": "USD",
                "code": 1
            }
        }),
    )
    .await;
    assert_eq!(st_act, StatusCode::OK);

    // Dry-run inside 0.55 ms isolated CoW branch (`main` stays 'QUEUED')
    let (st_dry, dry_res) = post_json(
        &app,
        "/api/v1/ontology/actions/settle_wire/execute",
        json!({
            "actor_id": "kiro-agent",
            "actor_roles": ["payments_officer"],
            "arguments": { "wire_id": 7, "cents": 3200, "debit_id": 10, "credit_id": 20 },
            "dry_run_on_branch": true
        }),
    )
    .await;
    assert_eq!(st_dry, StatusCode::OK, "dry run: {dry_res}");
    assert_eq!(dry_res["receipt"]["dry_run_on_branch"], true);

    let (_, check_main) = post_json(
        &app,
        "/api/v1/query",
        json!({ "sql": "SELECT state FROM wire_requests WHERE wire_id = 7" }),
    )
    .await;
    assert_eq!(check_main["rows"][0][0], "QUEUED");

    // Live commit -> mutates `main` AND appends SHA-256 double-entry transfer!
    let (st_live, live_res) = post_json(
        &app,
        "/api/v1/ontology/actions/settle_wire/execute",
        json!({
            "actor_id": "kiro-agent",
            "actor_roles": ["payments_officer"],
            "arguments": { "wire_id": 7, "cents": 3200, "debit_id": 10, "credit_id": 20 },
            "dry_run_on_branch": false
        }),
    )
    .await;
    assert_eq!(st_live, StatusCode::OK, "live action: {live_res}");
    assert!(live_res["receipt"]["ledger_transfer_id"].as_str().is_some());

    // Scoped Kiro Power bundle + OpenCode SQL LSP Diagnostics
    let (st_pow, pow_bytes) = get_raw(&app, "/api/v1/ontology/powers/payments").await;
    assert_eq!(st_pow, StatusCode::OK);
    let pow_json: Value = serde_json::from_slice(&pow_bytes).expect("power json");
    assert_eq!(pow_json["mcp_tools"].as_array().expect("tools").len(), 1);
    assert!(
        pow_json["tokens_saved_vs_full_59_tool_mcp"]
            .as_u64()
            .unwrap_or(0)
            > 10_000
    );

    let (st_diag, diag_res) = post_json(
        &app,
        "/api/v1/diagnostics/sql",
        json!({ "sql": "DELETE FROM wire_requests" }),
    )
    .await;
    assert_eq!(st_diag, StatusCode::OK);
    assert!(
        diag_res["diagnostics"]
            .as_array()
            .expect("diags")
            .iter()
            .any(|d| d["code"] == "ONT001")
    );

    // ── Pillar 4: In-SQL `AI_RANK` / `AI_FILTER` + `Lance Namespace` & `Blob v2` Slice ──
    let (st_ai, ai_res) = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "SELECT wire_id, AI_RANK('vendor invoice settlement', memo) AS score FROM wire_requests WHERE AI_FILTER(memo, 'invoice') ORDER BY score DESC"
        }),
    )
    .await;
    assert_eq!(st_ai, StatusCode::OK, "in-sql AI query: {ai_res}");
    assert_eq!(ai_res["rows"][0][0], 7);

    let blob_pack = tmp_dir.join("invoice_scans.pack");
    let pad = b"PACK_HDR_000";
    let pdf_slice = b"%PDF-1.7 WIRE_7_SIGNED_RECEIPT";
    let mut pack_bytes = pad.to_vec();
    pack_bytes.extend_from_slice(pdf_slice);
    std::fs::write(&blob_pack, &pack_bytes).expect("write pack");
    let sha_hex = hex::encode(Sha256::digest(pdf_slice));

    let _ = post_json(
        &app,
        "/api/v1/ducklake/tables",
        json!({
            "schema": "main",
            "table": "wire_attachments",
            "columns": [["wire_id", "BIGINT", false], ["scan_ref", "BLOB_REF", false]],
            "timestamp_ms": 5000
        }),
    )
    .await;

    let (st_breg, breg) = post_json(
        &app,
        "/api/v1/lance/blobs/register",
        json!({
            "schema": "main",
            "table": "wire_attachments",
            "column_name": "scan_ref",
            "row_id": 7,
            "uri": format!("file://{}", blob_pack.display()),
            "byte_offset": pad.len(),
            "byte_length": pdf_slice.len(),
            "mime_type": "application/pdf",
            "sha256_hex": sha_hex,
            "timestamp_ms": 5100
        }),
    )
    .await;
    assert_eq!(st_breg, StatusCode::OK, "register lance blob: {breg}");
    let blob_id = breg["blob"]["blob_id"].as_u64().expect("blob id");

    let (st_slice, slice_bytes) =
        get_raw(&app, &format!("/api/v1/lance/blobs/{blob_id}/slice")).await;
    assert_eq!(st_slice, StatusCode::OK);
    assert_eq!(slice_bytes.as_slice(), pdf_slice);

    let (st_ns, ns_bytes) = get_raw(&app, "/api/v1/lance/v1/namespace/main/table/list").await;
    assert_eq!(st_ns, StatusCode::OK);
    let ns_json: Value = serde_json::from_slice(&ns_bytes).expect("ns json");
    assert!(
        ns_json["tables"]
            .as_array()
            .expect("tables")
            .contains(&json!("wire_attachments"))
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
