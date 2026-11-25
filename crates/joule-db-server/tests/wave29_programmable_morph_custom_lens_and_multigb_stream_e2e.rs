//! Wave 29 (`v0.5.0`) End-to-End Integration Suite:
//! - Pillar 27: Programmable Semantic Morphing Rules (`MorphRuleDsl`), Scientific SI Unit
//!   Harmonization (`°C`/`°F`/`K`, `nm`/`µm`/`m`, `mg`/`g`/`kg`), Kanerva VSA Algebraic Role
//!   Unbinding (`H ⊗ ROLE⁻¹`), and User-Authored Custom Semantic Lenses (`ParetoScatter`, `SankeyLineage`).
//! - Pillar 28: Bounded-Memory Multi-Format Streaming Amorphous Continuum (`JSONL`, `CSV`, `XML`,
//!   `k=v`, `Prose`, `Triples`) + 45-Tool MCP 2026 Registry & Axum HTTP Verification.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use joule_db_server::mcp_transport::all_sota_tool_definitions;
use joule_db_server::{Server, ServerConfig};
use tower::ServiceExt;

#[tokio::test]
async fn test_wave29_programmable_morph_custom_lens_and_streaming_scale_e2e() {
    // 1. Verify >= 45 SOTA MCP 2026 tools registered!
    let tools = all_sota_tool_definitions();
    assert!(tools.len() >= 45, "expected >= 45 SOTA MCP tools in v0.5.0");

    // 2. Verify HTTP `/api/v1/amorphic/morph_rules_and_lens` with custom fragments + Morph DSL + Custom Lens DSL
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut cfg = ServerConfig::default();
    cfg.db_path = tmp.path().join("db").to_string_lossy().to_string();
    cfg.auth_enabled = false;
    cfg.rate_limiting_enabled = false;
    cfg.enable_tcp = false;
    cfg.enable_pgwire = false;
    cfg.enable_websocket = false;
    let server = Server::new(cfg).expect("create server");
    let app = server.router();
    let payload = serde_json::json!({
        "fragments": [
            {
                "silo_name": "fab_json",
                "payload": "{\"eng_mail\":\"ada@lovelace.org\",\"customer\":\"Ada Lovelace\",\"annual_contract\":\"$64,000.00\",\"pulse_energy\":\"4.0kJ\",\"chamber_temp\":\"25°C\",\"latency_ms\":12}"
            },
            {
                "silo_name": "metrology_kv",
                "payload": "operator=ada@lovelace.org, mrr_cents=6400000, ambient_temp=77°F, laser_lambda=532nm"
            },
            {
                "silo_name": "fab_json",
                "payload": "{\"eng_mail\":\"babbage@engine.uk\",\"customer\":\"Charles Babbage\",\"annual_contract\":\"$18,000.00\",\"pulse_energy\":\"9.0kJ\",\"chamber_temp\":\"20°C\",\"latency_ms\":85}"
            },
            {
                "silo_name": "metrology_kv",
                "payload": "operator=babbage@engine.uk, mrr_cents=1800000, ambient_temp=68°F, laser_lambda=1064nm"
            }
        ],
        "morph_dsl": "DERIVE \"efficiency.usd_per_joule\" = \"money.usd\" / \"energy.joules\" UNIT \"USD/J\"\nLINK WHEN \"efficiency.usd_per_joule\" >= 10.0 -> EDGE \"PARETO_LEADER\" TO \"tier1_cluster\"",
        "custom_lens_dsl": "LENS \"Thermodynamic Efficiency vs Latency Pareto\" ENCODING pareto_scatter X=\"duration.seconds\"(min,log) Y=\"efficiency.usd_per_joule\"(max) SIZE=\"money.usd\" MIN_SILOS=2"
    });

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/amorphic/morph_rules_and_lens")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .expect("req"),
        )
        .await
        .expect("http ok");
    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .expect("bytes");
    let v: serde_json::Value = serde_json::from_slice(&body_bytes).expect("json");
    assert_eq!(v["ok"], true);
    assert_eq!(v["morph_report"]["derived_atoms_synthesized"], 2);
    assert_eq!(v["morph_report"]["emergent_edges_synthesized"], 1);
    assert_eq!(
        v["custom_lens"]["pareto_optimal_anchors"],
        serde_json::json!(["ada@lovelace.org"])
    );

    // 3. Verify HTTP `/api/v1/amorphic/stream_corpus_scale`
    let stream_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/amorphic/stream_corpus_scale")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "target_bytes": 3_000_000,
                        "num_unique_entities": 80
                    })
                    .to_string(),
                ))
                .expect("req"),
        )
        .await
        .expect("http stream ok");
    assert_eq!(stream_resp.status(), StatusCode::OK);
    let stream_bytes = axum::body::to_bytes(stream_resp.into_body(), 4 * 1024 * 1024)
        .await
        .expect("bytes");
    let sv: serde_json::Value = serde_json::from_slice(&stream_bytes).expect("json");
    assert_eq!(sv["ok"], true);
    assert_eq!(sv["streaming_receipt"]["files_streamed"], 6);
    assert_eq!(sv["streaming_receipt"]["unique_entities_unified"], 80);
    assert!(
        sv["streaming_receipt"]["max_unit_harmonization_rel_err"]
            .as_f64()
            .unwrap_or(1.0)
            < 1e-6
    );

    // 4. Verify MCP Streamable HTTP (`POST /mcp`) dispatch for Wave 29 tools
    let mcp_resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "amorphic.morph_rules_and_lens",
                            "arguments": {}
                        }
                    })
                    .to_string(),
                ))
                .expect("req"),
        )
        .await
        .expect("mcp http ok");
    assert_eq!(mcp_resp.status(), StatusCode::OK);
    let mcp_bytes = axum::body::to_bytes(mcp_resp.into_body(), 4 * 1024 * 1024)
        .await
        .expect("bytes");
    let mv: serde_json::Value = serde_json::from_slice(&mcp_bytes).expect("json");
    assert!(
        mv["result"]["structuredContent"]["morph_benchmark_receipt"]
            ["vsa_role_unbinding_top1_accuracy"]
            .as_f64()
            .unwrap_or(0.0)
            >= 0.99
    );

    // 5. Production hygiene guard: zero `.unwrap(` or `.expect(` in non-test production code of Wave 29 modules!
    for rel_path in [
        "../joule-db-amorphic/src/morph_rules.rs",
        "../joule-db-amorphic/src/streaming_continuum.rs",
        "../joule-db-viz/src/custom_lens.rs",
    ] {
        let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel_path);
        let src = std::fs::read_to_string(&full).expect("read module source");
        let prod_part = src.split("#[cfg(test)]").next().unwrap_or(&src);
        assert!(
            !prod_part.contains(".unwrap(") && !prod_part.contains(".expect("),
            "found .unwrap() or .expect() in production section of {}",
            full.display()
        );
    }
}
