//! Synthetic Analytic Smoke Checks & MCP Utility Tools E2E Test (`v0.6.0`).
//!
//! Verifies `/api/v1/bench/analytic_suite` (synthetic in-memory ALP/FSST, skewed WCOJ,
//! ASOF, and hybrid RRF smoke checks) and MCP `sql.dialect_repair` + `robotics.stl_trajectory_clamp`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use joule_db_server::{Server, ServerConfig};
use serde_json::{Value, json};
use tower::ServiceExt;

async fn post_json(router: &axum::Router, uri: &str, payload: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("build request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let val: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
    (status, val)
}

#[tokio::test]
async fn test_wave27_synthetic_analytic_smoke_and_mcp_utility_tools_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut cfg = ServerConfig::default();
    cfg.db_path = tmp.path().join("db").to_string_lossy().to_string();
    cfg.auth_enabled = false;
    cfg.rate_limiting_enabled = false;
    cfg.enable_tcp = false;
    cfg.enable_pgwire = false;
    cfg.enable_websocket = false;

    let server = Server::new(cfg).expect("create server");
    let router = server.router();

    let (st_ana, ana_resp) = post_json(
        &router,
        "/api/v1/bench/analytic_suite",
        json!({
            "clickbench_rows": 8192,
            "job_hub_fanout": 120,
            "stac_ticks_per_venue": 1000,
            "beir_docs": 96
        }),
    )
    .await;
    assert_eq!(st_ana, StatusCode::OK, "analytic body: {ana_resp}");
    assert_eq!(ana_resp["all_four_synthetic_smoke_checks_passed"], true);
    assert_eq!(ana_resp["energy_evidence"], "est_j");
    assert!(
        ana_resp["synthetic_columnar_codec_smoke"]["compression_ratio"]
            .as_f64()
            .unwrap()
            > 3.5
    );
    assert!(
        ana_resp["synthetic_skewed_wcoj_smoke"]["intermediate_blowup_reduction_x"]
            .as_f64()
            .unwrap()
            > 15.0
    );
    assert_eq!(
        ana_resp["synthetic_asof_tick_smoke"]["future_leak_violations"]
            .as_u64()
            .unwrap(),
        0
    );

    let (st_mcp_list, mcp_list) = post_json(
        &router,
        "/mcp",
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": { "includeAgenticExtensions": true }
        }),
    )
    .await;
    assert_eq!(st_mcp_list, StatusCode::OK);
    let tools = mcp_list["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 59);

    let (_st_sql, mcp_sql) = post_json(
        &router,
        "/mcp",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "sql.dialect_repair",
                "arguments": {
                    "sql": "SELECT SAFE_DIVIDE(revnue, orders) FROM `sales`",
                    "known_columns": ["revenue", "orders"]
                }
            }
        }),
    )
    .await;
    let repaired = mcp_sql["result"]["structuredContent"]["repaired_sql"]
        .as_str()
        .unwrap();
    assert!(repaired.contains("NULLIF"));
    assert!(repaired.contains("revenue"));

    let (_st_vla, mcp_vla) = post_json(
        &router,
        "/mcp",
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "robotics.stl_trajectory_clamp",
                "arguments": {
                    "start_xyz": [0.0, 0.0, 0.0],
                    "envelope": {
                        "max_force_newtons": 12.0,
                        "max_step_delta_m": 0.2,
                        "keep_out_spheres": [[[0.05, 0.0, 0.0], 0.08]]
                    },
                    "trajectory": [{ "pos_xyz": [0.05, 0.0, 0.0], "force_newtons": 25.0 }]
                }
            }
        }),
    )
    .await;
    assert_eq!(
        mcp_vla["result"]["structuredContent"]["shielded_audit"]["violation_severity_index"]
            .as_f64()
            .unwrap(),
        0.0
    );
}
