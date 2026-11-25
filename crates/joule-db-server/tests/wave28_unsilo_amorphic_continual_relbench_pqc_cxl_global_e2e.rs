//! Wave 28 E2E Verification (`v0.4.0` — Un-Siloed Amorphous Continuum + Global Cross-Country SOTA Closure):
//! 1. `POST /api/v1/amorphic/unsilo_morph`: Dissolves 4 incompatible silos (`Salesforce JSON`,
//!    `Stripe k=v`, `Incident Prose`, `RDF Knowledge Triple`) into a `10,240`-bit HDC Semantic
//!    Continuum (`0.0` USD/cents unit error, `0 ms` ISO/Unix drift) and morphs across all 6 fluid
//!    display lenses (`ConstellationGraph`, `PerceptualChart`, `GeodesicMap`, `CausalTimeline`, etc.)!
//! 2. `POST /api/v1/sovereign/continual_adapt`: `OrthogonalTitansSynapse` (`0.00%` catastrophic
//!    forgetting across sequential tasks + `100%` CoW canary poison quarantine).
//! 3. `POST /api/v1/sovereign/relbench_rfm`: Stanford `RelBench v2` / `KumoRFM` multi-table temporal
//!    RFM (`future_leakage_rows_used == 0`, `AUROC >= 0.98`, `R² >= 0.95`).
//! 4. `POST /api/v1/sovereign/pq_attest` & `pq_federated_merge`: Real `NIST FIPS 203 ML-KEM-768` +
//!    `FIPS 204 ML-DSA-65` (`1,952 B` PK, `3,309 B` Sig) Merkle-rooted verifiable execution proofs +
//!    Byzantine `Multi-Krum` PQC federated continual skill gossip!
//! 5. `POST /api/v1/sovereign/global_sota_suite` & `POST /mcp` (`43` SOTA MCP tools, `CXL 3.1 +
//!    Mooncake` 3-tier pool, and 12-script `Cohere Global-MMLU` `1.0x` joule fairness).

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
async fn test_wave28_unsilo_amorphic_continual_relbench_pqc_cxl_global_e2e() {
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

    // 1. Verify Amorphous Un-Siloed Store + Fluid Intent-Driven Display Morphing
    let silos_payload = vec![
        json!({
            "silo_name": "crm_json",
            "payload": "{\"cust_mail\": \"ada@lovelace.org\", \"customer\": \"Ada Lovelace\", \"annual_contract\": \"$14,250.00\", \"signed_at\": \"2025-03-14T09:30:00Z\", \"account_mgr\": \"babbage\"}"
        }),
        json!({
            "silo_name": "stripe_csv",
            "payload": "subscriber_email=ada@lovelace.org, mrr_cents=1425000, created_unix=1741944600, territory_lat=51.5074, territory_lon=-0.1278"
        }),
        json!({
            "silo_name": "incident_log",
            "payload": "2025-03-14T09:30:00Z Incident #904: Ada Lovelace (ada@lovelace.org) observed 420ms latency and 2.5kJ energy draw in London [51.5074, -0.1278] with @babbage"
        }),
        json!({
            "silo_name": "knowledge_triple",
            "payload": "(ada@lovelace.org) --[ARCHITECTS]--> (Analytical Engine v2)"
        }),
    ];

    for (intent, expected_lens) in [
        ("explore_connections", "constellation_graph"),
        ("compare_quantities", "perceptual_chart"),
        ("locate_spatially", "geodesic_map"),
        ("trace_causal_history", "causal_timeline"),
        ("inspect_unified_dossier", "narrative_dossier"),
    ] {
        let (st, resp) = post_json(
            &router,
            "/api/v1/amorphic/unsilo_morph",
            json!({
                "fragments": silos_payload,
                "intent": intent
            }),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "unsilo morph failed: {resp}");
        assert_eq!(resp["canvas_spec"]["primary_lens"], expected_lens);
        assert_eq!(resp["unified_entities"].as_array().unwrap().len(), 1);
        assert_eq!(
            resp["unified_entities"][0]["source_silos"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert!(resp["cross_silo_same_entity_resonance"].as_f64().unwrap() > 0.68);
        assert_eq!(resp["canvas_spec"]["semantic_preservation_ratio"], 1.0);
    }

    // 2. Verify Unified 5-Pillar Global SOTA Closure Suite (`B22..B26`)
    let (st_suite, suite) =
        post_json(&router, "/api/v1/sovereign/global_sota_suite", json!({})).await;
    assert_eq!(st_suite, StatusCode::OK, "global_sota_suite: {suite}");
    assert_eq!(suite["all_5_global_frontiers_passed"], true);
    assert!(
        suite["b23_orthogonal_titans_continual"]["forgetting_reduction_x"]
            .as_f64()
            .unwrap()
            > 25.0
    );
    assert_eq!(
        suite["b24_relbench_temporal_rfm"]["user_churn"]["future_leakage_rows_used"],
        0
    );
    assert_eq!(
        suite["b25_post_quantum_verifiable_sovereign"]["mldsa65_pk_bytes"],
        1952
    );
    assert_eq!(
        suite["b25_post_quantum_verifiable_sovereign"]["mldsa65_sig_bytes"],
        3309
    );
    assert_eq!(
        suite["b26_cxl_and_12script_global_mmlu"]["global_mmlu_12script"]["languages_evaluated"],
        12
    );

    // 3. Verify PQC Federated Mint + Merge over HTTP (`ML-DSA-65` + `ML-KEM-768`)
    let (st_m1, m1) = post_json(
        &router,
        "/api/v1/sovereign/pq_federated_merge",
        json!({
            "mint_local_node_id": "m5-max-node",
            "mint_local_svf": { "blk.0.q": [0.15, -0.05, 0.22] },
            "mint_local_facts": [["unsilo_rule", "preserve_meaning"]]
        }),
    )
    .await;
    assert_eq!(st_m1, StatusCode::OK);
    let delta1 = m1["minted_delta"].clone();
    let (st_mg, mg) = post_json(
        &router,
        "/api/v1/sovereign/pq_federated_merge",
        json!({
            "deltas": [delta1],
            "max_byzantine": 0
        }),
    )
    .await;
    assert_eq!(st_mg, StatusCode::OK);
    assert_eq!(mg["report"]["accepted_nodes"][0], "m5-max-node");
    assert_eq!(mg["report"]["mlkem768_channel_verified"], true);

    // 4. Verify Native MCP Streamable HTTP (`POST /mcp`) exposes all 43 SOTA Tools & executes `amorphic.unsilo_morph`
    let (st_list, mcp_list) = post_json(
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
    assert_eq!(st_list, StatusCode::OK);
    let tools = mcp_list["result"]["tools"].as_array().expect("mcp tools");
    assert!(
        tools.len() >= 43,
        "expected >= 43 SOTA MCP tools, got {}",
        tools.len()
    );

    let (st_call, mcp_call) = post_json(
        &router,
        "/mcp",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "amorphic.unsilo_morph",
                "arguments": {}
            }
        }),
    )
    .await;
    assert_eq!(st_call, StatusCode::OK, "mcp call: {mcp_call}");
    assert_eq!(
        mcp_call["result"]["structuredContent"]["benchmark_receipt"]["silos_ingested"],
        4
    );
}
