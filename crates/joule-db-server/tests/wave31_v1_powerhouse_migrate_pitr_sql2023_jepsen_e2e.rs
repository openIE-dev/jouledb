//! End-to-End Integration Test for `v0.6.0` Subsystem Smoke Checks:
//! - Multi-format import & multiset BLAKE3 reconciliation (`live_migration`)
//! - Binary TLV/CRC-32 framing & WAL LSN/timestamp PITR (`v1_stability_and_pitr`)
//! - Window frames, 3D CUBE, recursive CTE & runtime AQE (`sql2023_powerhouse`)
//! - Scripted 5-struct in-memory partition unit simulation (`distributed_jepsen_consensus`)
//! - Honest MCP tool registry (`49` tools)

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use joule_db_server::{Server, ServerConfig};
use tower::ServiceExt;

#[tokio::test]
async fn test_wave31_subsystem_smoke_http_mcp_migration_pitr_and_aqe_e2e() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut cfg = ServerConfig::default();
    cfg.db_path = temp.path().to_string_lossy().to_string();
    let server = Server::new(cfg).expect("server init");
    let app = server.router();

    // 1. 6-check in-process subsystem smoke runner over HTTP
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/bench/v1_powerhouse")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let smoke: serde_json::Value = serde_json::from_slice(&body_bytes).expect("json");
    assert_eq!(smoke["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(smoke["checks_executed"], 6);
    assert_eq!(smoke["all_6_smoke_checks_passed"], true);
    assert_eq!(smoke["energy_evidence"], "est_j");

    // 2. Live migration parser + CDC handoff + ML-DSA-65 signature
    let mig_payload = serde_json::json!({
        "migration_id": "e2e-pg-cutover-v060",
        "source_format": "postgres",
        "collection_or_table": "orders",
        "payload": "CREATE TABLE orders (id INT PRIMARY KEY, customer TEXT, total_cents INT);\nINSERT INTO orders (id, customer, total_cents) VALUES (101, 'cern-lab', 450000), (102, 'esa-ops', 890000);",
        "cdc_events": [
            {
                "lsn": 9001,
                "op": "INSERT",
                "record": {
                    "collection_or_table": "orders",
                    "primary_key": "103",
                    "fields": { "id": 103, "customer": "jaxa-tokyo", "total_cents": 620000 }
                }
            }
        ]
    });
    let mig_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/migrate/run")
                .header("content-type", "application/json")
                .body(Body::from(mig_payload.to_string()))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(mig_resp.status(), StatusCode::OK);
    let mig_json: serde_json::Value = serde_json::from_slice(
        &mig_resp.into_body().collect().await.expect("b").to_bytes(),
    )
    .expect("json");
    assert_eq!(mig_json["zero_row_drift"], true);
    assert_eq!(mig_json["final_reconciled_rows"], 3);
    assert_eq!(mig_json["pq_signature_verified"], true);

    // 3. Verify 49 MCP tools via JSON-RPC `tools/list`
    let mcp_list = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/list",
        "params": { "includeAgenticExtensions": true }
    });
    let mcp_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .body(Body::from(mcp_list.to_string()))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(mcp_resp.status(), StatusCode::OK);
    let mcp_json: serde_json::Value = serde_json::from_slice(
        &mcp_resp.into_body().collect().await.expect("b").to_bytes(),
    )
    .expect("json");
    let tools = mcp_json["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 59);

    for tool_name in [
        "migrate.shadow_cutover",
        "core.tlv_and_pitr_smoke",
        "query.window_cube_aqe_smoke",
        "cluster.in_memory_partition_sim",
        "deploy.iac_blueprint",
        "bench.subsystem_smoke",
    ] {
        let call_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": { "name": tool_name, "arguments": {} }
        });
        let call_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from(call_req.to_string()))
                    .expect("req"),
            )
            .await
            .expect("resp");
        assert_eq!(call_resp.status(), StatusCode::OK);
        let call_json: serde_json::Value = serde_json::from_slice(
            &call_resp.into_body().collect().await.expect("b").to_bytes(),
        )
        .expect("json");
        assert!(
            call_json.get("error").is_none() || call_json["error"].is_null(),
            "MCP tool {tool_name} failed: {call_json}"
        );
    }
}
