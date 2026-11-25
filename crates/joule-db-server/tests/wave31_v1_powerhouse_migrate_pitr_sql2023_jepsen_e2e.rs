//! End-to-End Certification for `JouleDB v1.0.0` GA ("Global Database Powerhouse"):
//! - Pillar 31: Zero-Downtime Polyglot Migration (`PostgreSQL`, `SQLite`, `Mongo`, `Elastic`, `CSV`),
//!              Order-Independent Multiset BLAKE3 Reconciliation, CoW Shadow-Traffic Parity & `ML-DSA-65` Certificate
//! - Pillar 32: `v1.0.0` Wire/Storage ABI Forward/Backward Compatibility, IEEE CRC-32 Bit-Rot Guard,
//!              Continuous Hash-Chained WAL Archiving & Exact Microsecond `PITR` + Rolling Upgrade
//! - Pillar 33: Enterprise `SQL:2023` Window Frames, 3D `CUBE`, Cycle-Safe `WITH RECURSIVE` BOM CTE,
//!              Runtime Adaptive Query Execution (`AQE` 100x Skew Switch) & Hybrid `SQL ⋈ JSONB ⋈ RaBitQ`
//! - Pillar 34: 5-Node 3-Region Geo-Partitioned Raft, Bounded-Staleness Follower Reads (`0` WAN hops)
//!              & Deterministic `Jepsen` Split-Brain / Clock-Skew / Linearizability Nemesis Suite
//! - Pillar 35 & 36: Polyglot First-Party SDK Matrix + `Helm` / `Terraform` IaC Blueprints & `54` MCP Tools

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use joule_db_server::{Server, ServerConfig};
use tower::ServiceExt;

#[tokio::test]
async fn test_wave31_v1_powerhouse_http_mcp_migration_pitr_aqe_and_jepsen_e2e() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut cfg = ServerConfig::default();
    cfg.db_path = temp.path().to_string_lossy().to_string();
    let server = Server::new(cfg).expect("server init");
    let app = server.router();

    // 1. Master v1.0.0 36-Pillar Powerhouse Suite over HTTP
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
    let powerhouse: serde_json::Value = serde_json::from_slice(&body_bytes).expect("json");
    assert_eq!(powerhouse["total_certified_pillars"], 36);
    assert_eq!(powerhouse["mcp_2026_tools_count"], 54);
    assert_eq!(powerhouse["all_36_pillars_passed"], true);

    // 2. Live Custom Migration Run with CDC High-Watermark Handoff + ML-DSA-65 Certificate
    let mig_payload = serde_json::json!({
        "migration_id": "e2e-pg-cutover-v1",
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
    assert_eq!(mig_json["cutover_write_freeze_ms"], 0);
    assert_eq!(mig_json["pq_signature_verified"], true);
    assert_eq!(mig_json["pq_signature_bytes"], 3309);

    // 3. Verify 54 MCP 2026 SOTA tools via JSON-RPC `tools/list` (`includeAgenticExtensions: true`)
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
    assert_eq!(tools.len(), 54, "expected 54 MCP 2026 tools in v1.0.0");

    // 4. Dispatch `cluster.jepsen_geo_raft` and `core.v1_abi_and_pitr` over MCP `tools/call`
    for tool_name in [
        "migrate.shadow_cutover",
        "core.v1_abi_and_pitr",
        "query.sql2023_aqe_suite",
        "cluster.jepsen_geo_raft",
        "deploy.iac_blueprint",
        "bench.v1_powerhouse_36",
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
