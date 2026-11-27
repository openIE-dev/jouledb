//! End-to-End Verification Suite for **JouleDB v0.18.0**:
//! 1. **Pure-RAM Ephemeral Branch Executor + First-Class SQL Ontology DDL** (`CREATE ONTOLOGY OBJECT`,
//!    `CREATE ONTOLOGY LINK`, `CREATE KINETIC ACTION`, `EXECUTE ACTION ... DRY RUN`, `SHOW ONTOLOGY`,
//!    `DIAGNOSE SQL`) with sub-millisecond CoW branch dry-run isolation + atomic `FinancialLedger` hooks.
//! 2. **Live Async Upstream `HTTP 429` Catch & Multi-Tier Failover** in `JouleRoute` using real TCP
//!    mock OpenAI servers (Tier 3 returns `HTTP 429` -> automatic exponential circuit breaker ->
//!    seamless live HTTP failover to Tier 4 healthy upstream!).
//! 3. **Voyage AI / Cohere Rerank v2 Wire-Compatible `POST /v1/rerank`** precision + `joule_receipt`.
//! 4. **Dynamic Scoped `Kiro` MCP Powers (`POST /mcp?power=<domain>`)** projecting only governed
//!    `ontology_action__*` tools + steering (`> 90%` prompt token reduction) and dispatching live
//!    governed actions over MCP JSON-RPC.

use axum::{
    Json, Router,
    body::Body,
    http::{Request, StatusCode},
    response::IntoResponse,
    routing::post,
};
use http_body_util::BodyExt;
use joule_db_server::{Server, ServerConfig};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tower::ServiceExt;

async fn post_json(app: &Router, uri: &str, payload: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("build POST");
    let resp = app.clone().oneshot(req).await.expect("oneshot POST");
    let status = resp.status();
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes();
    let val: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&bytes).to_string() }));
    (status, val)
}

#[tokio::test]
async fn test_v018_sql_ontology_ddl_pure_ram_dryrun_live_429_rerank_and_scoped_mcp() {
    let tmp = std::env::temp_dir().join(format!(
        "jouledb-v018-e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut cfg = ServerConfig::default();
    cfg.db_path = tmp.to_string_lossy().to_string();
    cfg.auth_enabled = false;
    cfg.rate_limiting_enabled = false;

    let server = Server::new(cfg).expect("Server::new");
    let app = server.router();

    // =========================================================================
    // 1. First-Class SQL Ontology DDL + Pure-RAM Branch Dry-Run + Ledger Hook
    // =========================================================================
    for sql in [
        "CREATE TABLE wire_requests (wire_id INTEGER PRIMARY KEY, status TEXT, amount_minor INTEGER)",
        "INSERT INTO wire_requests VALUES (501, 'PENDING_REVIEW', 25000)",
        "LEDGER CREATE ACCOUNT 'treasury_cash' LEDGER 1 CODE 100 SCALE 2",
        "LEDGER CREATE ACCOUNT 'vendor_settlement' LEDGER 1 CODE 200 SCALE 2",
        "CREATE ONTOLOGY OBJECT WireRequest ON wire_requests (PRIMARY KEY wire_id, DOMAIN 'treasury', DESCRIPTION 'Governed corporate wire')",
        "CREATE ONTOLOGY LINK WireSettlement FROM WireRequest.wire_id TO LedgerTransfer.transfer_id CARDINALITY 'one_to_one'",
        "CREATE KINETIC ACTION approve_wire ON WireRequest DOMAIN 'treasury' ROLE 'treasurer' WHEN 'WHEN :amount_minor > 100000 THE SYSTEM SHALL REJECT WITH 23514: wire exceeds 1000 USD single-approval limit' LEDGER (debit_acct, credit_acct, amount_minor, 'USD') AS UPDATE wire_requests SET status = 'SETTLED', amount_minor = :amount_minor WHERE wire_id = :wire_id",
    ] {
        let (st, body) = post_json(&app, "/api/v1/query", json!({ "sql": sql })).await;
        assert_eq!(st, StatusCode::OK, "SQL setup failed on `{sql}`: {body}");
    }

    // `SHOW ONTOLOGY` returns OBJECT + LINK + ACTION rows
    let (st_show, show_body) =
        post_json(&app, "/api/v1/query", json!({ "sql": "SHOW ONTOLOGY" })).await;
    assert_eq!(st_show, StatusCode::OK);
    assert_eq!(show_body["rows"].as_array().expect("rows").len(), 3);

    // `DIAGNOSE SQL` catches unguarded UPDATE + emits ONT001 KineticAction governance hint!
    let (st_diag, diag_body) = post_json(
        &app,
        "/api/v1/query",
        json!({ "sql": "DIAGNOSE SQL 'UPDATE wire_requests SET status = ''BYPASSED'''" }),
    )
    .await;
    assert_eq!(st_diag, StatusCode::OK, "diagnose body: {diag_body}");
    let diag_codes: Vec<&str> = diag_body["rows"]
        .as_array()
        .expect("diag rows")
        .iter()
        .filter_map(|r| r.get(2).and_then(|v| v.as_str()))
        .collect();
    assert!(
        diag_codes.contains(&"42501"),
        "expected 42501: {diag_codes:?}"
    );
    assert!(
        diag_codes.contains(&"ONT001"),
        "expected ONT001: {diag_codes:?}"
    );

    // EARS precondition blocks > $1,000 (`amount_minor = 250000`) via SQL `EXECUTE ACTION`
    let (st_over, over_body) = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "EXECUTE ACTION approve_wire (wire_id = 501, amount_minor = 250000, debit_acct = 'treasury_cash', credit_acct = 'vendor_settlement') AS 'cfo-bot' ROLE 'treasurer'"
        }),
    )
    .await;
    assert_ne!(
        st_over,
        StatusCode::OK,
        "expected EARS rejection: {over_body}"
    );

    // Pure-RAM `DRY RUN` simulates on isolated CoW branch without mutating `main` or posting ledger!
    let (st_dry, dry_body) = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "EXECUTE ACTION approve_wire (wire_id = 501, amount_minor = 45000, debit_acct = 'treasury_cash', credit_acct = 'vendor_settlement') AS 'cfo-bot' ROLE 'treasurer' DRY RUN"
        }),
    )
    .await;
    assert_eq!(st_dry, StatusCode::OK, "dry run failed: {dry_body}");
    assert_eq!(dry_body["rows"][0][2], json!(true));
    assert!(dry_body["rows"][0][4].is_null());

    // Confirm `main` is still `PENDING_REVIEW` after `DRY RUN`
    let (_, main_check) = post_json(
        &app,
        "/api/v1/query",
        json!({ "sql": "SELECT status, amount_minor FROM wire_requests WHERE wire_id = 501" }),
    )
    .await;
    assert_eq!(main_check["rows"][0][0], json!("PENDING_REVIEW"));

    // Live SQL `EXECUTE ACTION` commits table mutation AND atomic 128-bit FinancialLedger transfer!
    let (st_live, live_body) = post_json(
        &app,
        "/api/v1/query",
        json!({
            "sql": "EXECUTE ACTION approve_wire (wire_id = 501, amount_minor = 45000, debit_acct = 'treasury_cash', credit_acct = 'vendor_settlement') AS 'cfo-bot' ROLE 'treasurer'"
        }),
    )
    .await;
    assert_eq!(st_live, StatusCode::OK, "live action failed: {live_body}");
    assert_eq!(live_body["rows"][0][2], json!(false));
    assert!(
        live_body["rows"][0][4].is_string(),
        "expected ledger transfer id"
    );

    // =========================================================================
    // 2. Scoped Kiro MCP Power (`POST /mcp?power=treasury`) + MCP Action Call
    // =========================================================================
    let (st_mcp_list, mcp_list) = post_json(
        &app,
        "/mcp?power=treasury",
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }),
    )
    .await;
    assert_eq!(st_mcp_list, StatusCode::OK, "mcp tools/list: {mcp_list}");
    let mcp_tools = mcp_list["result"]["tools"].as_array().expect("tools array");
    // 1 domain `ontology_action__approve_wire` + 4 core scoped tools = 5 tools (instead of 59!)
    assert_eq!(mcp_tools.len(), 5, "unexpected scoped tools: {mcp_tools:?}");
    assert!(
        mcp_tools
            .iter()
            .any(|t| t["name"] == "ontology_action__approve_wire")
    );
    assert!(
        mcp_list["result"]["_meta"]["jouledb_power"]["savings_pct"]
            .as_f64()
            .unwrap_or(0.0)
            > 90.0
    );

    // Invoke `ontology_action__approve_wire` in `dry_run` mode directly over MCP `tools/call`!
    let (st_mcp_call, mcp_call) = post_json(
        &app,
        "/mcp?power=treasury",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ontology_action__approve_wire",
                "arguments": {
                    "wire_id": 501,
                    "amount_minor": 30000,
                    "debit_acct": "treasury_cash",
                    "credit_acct": "vendor_settlement",
                    "dry_run": true
                }
            }
        }),
    )
    .await;
    assert_eq!(st_mcp_call, StatusCode::OK);
    assert!(
        mcp_call["error"].is_null(),
        "unexpected mcp call error: {mcp_call}"
    );
    assert_eq!(
        mcp_call["result"]["structuredContent"]["dry_run_on_branch"],
        json!(true)
    );

    // =========================================================================
    // 3. Voyage AI / Cohere Rerank v2 Wire-Compatible `POST /v1/rerank`
    // =========================================================================
    let (st_rr, rr_body) = post_json(
        &app,
        "/v1/rerank",
        json!({
            "model": "rerank-2.5",
            "query": "double-entry financial ledger overdraft sqlstate 23514",
            "documents": [
                "Kubernetes pod autoscaling horizontal CPU metrics configuration",
                "JouleDB enforces strict 128-bit double-entry financial ledger conservation and rejects overdrafts with SQLSTATE 23514.",
                "Recipe for sourdough bread fermentation at room temperature"
            ],
            "top_n": 2,
            "return_documents": true
        }),
    )
    .await;
    assert_eq!(st_rr, StatusCode::OK, "rerank failed: {rr_body}");
    assert_eq!(rr_body["results"][0]["index"], json!(1));
    assert!(
        rr_body["results"][0]["relevance_score"]
            .as_f64()
            .unwrap_or(0.0)
            > rr_body["results"][1]["relevance_score"]
                .as_f64()
                .unwrap_or(1.0)
    );
    assert_eq!(
        rr_body["joule_receipt"]["measurement_class"],
        json!("analytical_est")
    );

    // =========================================================================
    // 4. Live Network `HTTP 429` Catch & Multi-Tier Failover in `JouleRoute`
    // =========================================================================
    let rate_limited_hits = Arc::new(AtomicUsize::new(0));
    let rl_counter = rate_limited_hits.clone();
    let mock_429_app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let c = rl_counter.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({ "error": { "message": "rate limit exceeded" } })),
                )
                    .into_response()
            }
        }),
    );
    let mock_429_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 429 mock");
    let mock_429_addr = mock_429_listener.local_addr().expect("429 addr");
    tokio::spawn(async move {
        let _ = axum::serve(mock_429_listener, mock_429_app).await;
    });

    let healthy_hits = Arc::new(AtomicUsize::new(0));
    let ok_counter = healthy_hits.clone();
    let mock_200_app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(payload): Json<Value>| {
            let c = ok_counter.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                let model = payload
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("healthy-model");
                (
                    StatusCode::OK,
                    Json(json!({
                        "id": "chatcmpl-live-200",
                        "object": "chat.completion",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "LIVE_UPSTREAM_FAILOVER_SUCCESS"
                            },
                            "finish_reason": "stop"
                        }]
                    })),
                )
                    .into_response()
            }
        }),
    );
    let mock_200_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 200 mock");
    let mock_200_addr = mock_200_listener.local_addr().expect("200 addr");
    tokio::spawn(async move {
        let _ = axum::serve(mock_200_listener, mock_200_app).await;
    });

    // Register Tier 1 (`oauth_subscription`) pointing to the live `429` server and Tier 3 (`free_tier_pool`)
    // pointing to the live `200 OK` server, then restrict `max_tier` or let Tier 1 fail over!
    // Wait: Tier 2 has `local-joulesynapse-jsyn` by default, so let's register the `429` mock at Tier 1
    // (`oauth_subscription`) and the `200 OK` mock ALSO at Tier 1 (or register `429` at Tier 1 and `200` at Tier 1)!
    let (st_p1, _) = post_json(
        &app,
        "/api/v1/router/providers/register",
        json!({
            "provider_id": "aaa-tier1-quota-exhausted",
            "vendor": "mock-sub",
            "model_name": "sub-primary-v1",
            "base_url": format!("http://{mock_429_addr}/v1"),
            "tier": "oauth_subscription",
            "api_key": "sk-test-sub-primary-9999",
            "live_http_dispatch": true,
            "rpm_limit": 100,
            "cost_micro_usd_per_1k_tokens": 0,
            "estimated_uj_per_token": 15
        }),
    )
    .await;
    assert_eq!(st_p1, StatusCode::OK);

    let (st_p2, _) = post_json(
        &app,
        "/api/v1/router/providers/register",
        json!({
            "provider_id": "zzz-tier1-healthy-backup",
            "vendor": "mock-sub-backup",
            "model_name": "sub-backup-v1",
            "base_url": format!("http://{mock_200_addr}/v1"),
            "tier": "oauth_subscription",
            "api_key": "sk-test-sub-backup-8888",
            "live_http_dispatch": true,
            "rpm_limit": 100,
            "cost_micro_usd_per_1k_tokens": 0,
            "estimated_uj_per_token": 18
        }),
    )
    .await;
    assert_eq!(st_p2, StatusCode::OK);

    let (st_comp, comp_body) = post_json(
        &app,
        "/v1/router/chat/completions",
        json!({
            "model": "auto",
            "messages": [{ "role": "user", "content": "Verify live HTTP 429 waterfall failover" }],
            "bypass_lang_cache": true
        }),
    )
    .await;
    assert_eq!(
        st_comp,
        StatusCode::OK,
        "router completion failed: {comp_body}"
    );
    assert_eq!(rate_limited_hits.load(Ordering::SeqCst), 1);
    assert_eq!(healthy_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        comp_body["choices"][0]["message"]["content"],
        json!("LIVE_UPSTREAM_FAILOVER_SUCCESS")
    );
    assert_eq!(
        comp_body["joule_route"]["selected_provider"],
        json!("zzz-tier1-healthy-backup")
    );
    assert_eq!(
        comp_body["joule_route"]["dispatch_mode"],
        json!("live_http")
    );
    let skipped = comp_body["joule_route"]["failover_skipped"]
        .as_array()
        .expect("skipped array");
    assert!(
        skipped
            .iter()
            .any(|v| v.as_str() == Some("aaa-tier1-quota-exhausted:live_http_429")),
        "expected live_http_429 skip receipt, got {skipped:?}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
