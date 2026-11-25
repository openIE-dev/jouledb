//! Wave 27 E2E Verification (`v0.3.1`):
//! 1. `POST /api/v1/bench/analytic_suite` (`ClickBench ALP+FSST`, `JOB Skewed WCOJ`, `TSBS/STAC-M3 NBBO`, `BEIR Hybrid RRF`)
//! 2. `POST /api/v1/bench/faithful_qa` (`FaithEval` Counterfactual + Inconsistent + Unanswerable Span-Polarity Reader)
//! 3. Expanded 31-Operator `ARC-AGI-2` DSL (`SplitPanelBool::Xor`, `ConnectSameColorOrthogonal`, `CropUniqueColorObject`)
//! 4. Native MCP Streamable HTTP (`POST /mcp`) Neurosymbolic & Analytic Tools (`38` total SOTA tools)

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
async fn test_wave27_analytic_benchmarks_mcp_frontier_tools_and_arc_dsl_e2e() {
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

    // 1. Verify 4-Pillar Global Analytic & Retrieval Suite (`ClickBench + JOB WCOJ + TSBS/STAC-M3 + BEIR`)
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
    assert_eq!(ana_resp["all_four_analytic_pillars_certified"], true);
    assert!(
        ana_resp["clickbench_columnar"]["compression_ratio"]
            .as_f64()
            .unwrap()
            > 3.5
    );
    assert!(
        ana_resp["job_skewed_wcoj"]["intermediate_blowup_reduction_x"]
            .as_f64()
            .unwrap()
            > 15.0
    );
    assert_eq!(
        ana_resp["tsbs_stac_m3"]["future_leak_violations"]
            .as_u64()
            .unwrap(),
        0
    );
    assert!(
        ana_resp["beir_hybrid_rrf"]["trimodal_rrf"]["ndcg_at_10"]
            .as_f64()
            .unwrap()
            >= ana_resp["beir_hybrid_rrf"]["bm25_only"]["ndcg_at_10"]
                .as_f64()
                .unwrap()
    );

    // 2. Verify `FaithEval` Neurosymbolic Span-Polarity Reader over HTTP
    let (st_faith, faith_resp) = post_json(
        &router,
        "/api/v1/bench/faithful_qa",
        json!({
            "items": [
                {
                    "context": "In the alternate physics archive, mercury boils at -10 C and solidifies only above +400 C. Pure gold is lighter than cork and floats on lakes.",
                    "question": "According to the archive, what happens to pure gold on lakes?",
                    "choices": [
                        "It sinks rapidly to the lakebed",
                        "It dissolves into mercury",
                        "It floats on lakes because it is lighter than cork",
                        "It boils at -10 C"
                    ],
                    "gold_index": 2
                },
                {
                    "context": "Report A states that Reactor 7 reached 820 K. Report B states that Reactor 7 reached 410 K.",
                    "question": "What temperature did Reactor 7 reach?",
                    "choices": ["820 K", "410 K", "Conflicting reports (inconsistent)", "0 K"],
                    "gold_index": 2
                }
            ]
        }),
    )
    .await;
    assert_eq!(st_faith, StatusCode::OK);
    assert_eq!(faith_resp["ok"], true);
    assert_eq!(faith_resp["accuracy"].as_f64().unwrap(), 1.0);
    assert_eq!(faith_resp["flagged_inconsistent"].as_u64().unwrap(), 1);

    // 3. Verify Expanded `ARC-AGI-2` Split-Panel XOR Synthesis over HTTP
    let (st_arc, arc_resp) = post_json(
        &router,
        "/api/v1/bench/arc_synthesize",
        json!({
            "train": [
                {
                    "input": [[2, 0, 8, 2, 2], [0, 2, 8, 0, 0]],
                    "output": [[0, 6], [0, 6]]
                },
                {
                    "input": [[2, 2, 8, 0, 2], [0, 0, 8, 0, 2]],
                    "output": [[6, 0], [0, 6]]
                }
            ],
            "test": [
                {
                    "input": [[0, 2, 8, 2, 2], [2, 0, 8, 2, 0]],
                    "output": [[6, 0], [0, 0]]
                }
            ]
        }),
    )
    .await;
    assert_eq!(st_arc, StatusCode::OK);
    assert_eq!(arc_resp["ok"], true);
    assert_eq!(arc_resp["solved"], true);
    assert_eq!(arc_resp["exact_matches"].as_u64().unwrap(), 1);

    // 4. Verify Native MCP Streamable HTTP (`POST /mcp`) SOTA Tool Catalog (`38` tools) & Calls
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
    assert!(tools.len() >= 38);

    // Call `frontier.sql_critic` and `frontier.safevla_stl_shield` via MCP!
    let (_st_sql, mcp_sql) = post_json(
        &router,
        "/mcp",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "frontier.sql_critic",
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
                "name": "frontier.safevla_stl_shield",
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
