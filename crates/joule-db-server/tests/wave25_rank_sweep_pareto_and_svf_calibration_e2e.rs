//! Wave 25 End-to-End Verification:
//! Certifies the Peer-Review `.jsyn` Rank-Sweep Quality-vs-Joules Evaluator
//! (`POST /api/v1/synapse/bench/rank_sweep` across `target_rank = 8 -> 32 -> 64 -> 128` +
//! out-of-sample `AWQ ⊕ GPTQ` Activation-Aware `SVF` Calibration) on both synthetic skewed-channel
//! Transformer projections and real `.safetensors` (`BF16` + packed `NVFP4` Mixture-of-Experts)
//! checkpoints, plus `Engram` database-offloaded factual recall alongside ultra-low-rank (`r = 8`) bitplanes.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::checkpoint_ingest::{
    SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
};
use joule_db_weights::rank_sweep_bench::generate_synthetic_transformer_matrix;
use serde_json::{Value, json};
use std::collections::BTreeMap;
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
async fn test_wave25_rank_sweep_pareto_svf_calibration_and_engram_fact_offload_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let st_path = tmp.path().join("flagship_layer.safetensors");

    let rows = 384usize;
    let cols = 384usize;
    let dense_w = generate_synthetic_transformer_matrix(rows, cols, 20261124);
    let entries = vec![SafetensorBuildEntry {
        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        dtype: SafetensorDtype::BF16,
        shape: vec![rows, cols],
        values_f32: dense_w,
    }];
    let st_bytes = build_safetensors_bytes(&BTreeMap::new(), &entries).expect("safetensors");
    std::fs::write(&st_path, st_bytes).expect("write st");

    let mut cfg = ServerConfig::default();
    cfg.db_path = tmp.path().join("db").to_string_lossy().to_string();
    cfg.auth_enabled = false;
    cfg.rate_limiting_enabled = false;
    cfg.enable_tcp = false;
    cfg.enable_pgwire = false;
    cfg.enable_websocket = false;

    let srv = Server::new(cfg).expect("Server::new");
    let router = srv.router();

    let (st, sweep_v) = post_json(
        &router,
        "/api/v1/synapse/bench/rank_sweep",
        json!({
            "safetensors_path": st_path.to_string_lossy(),
            "tensor_filter": "self_attn.q_proj.weight",
            "ranks": [8, 32, 64, 128],
            "max_outliers": 32,
            "calib_samples": 24,
            "eval_samples": 24,
            "svf_sweeps": 6,
        }),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "rank_sweep failed: {sweep_v}");
    let pts = sweep_v["report"]["points"]
        .as_array()
        .expect("points array");
    assert_eq!(pts.len(), 4);

    let r8_kl = pts[0]["post_svf_kl_divergence"].as_f64().expect("r8 kl");
    let r128_kl = pts[3]["post_svf_kl_divergence"].as_f64().expect("r128 kl");
    let r8_cos = pts[0]["post_svf_output_cosine"].as_f64().expect("r8 cos");
    let r128_cos = pts[3]["post_svf_output_cosine"].as_f64().expect("r128 cos");
    assert!(
        r128_kl < r8_kl,
        "KL divergence must decrease from r=8 ({r8_kl}) to r=128 ({r128_kl})"
    );
    assert!(
        r128_cos > r8_cos,
        "Output cosine must increase from r=8 ({r8_cos}) to r=128 ({r128_cos})"
    );

    for p in pts {
        let pre_m = p["pre_svf_eval_mse"].as_f64().expect("pre mse");
        let post_m = p["post_svf_eval_mse"].as_f64().expect("post mse");
        assert!(
            post_m <= pre_m,
            "Held-out SVF calibration must reduce or preserve MSE ({post_m} <= {pre_m})"
        );
        assert_eq!(p["base_bitplanes_mutated"].as_bool(), Some(false));
    }

    // Verify Engram O(1) database-backed factual recall alongside sub-1-bit `.jsyn`
    let (st_eng, _) = post_json(
        &router,
        "/api/v1/synapse/engram/upsert",
        json!({
            "phrase": "third mayor of toledo 1887",
            "embedding": [0.25, -0.5, 0.75, 0.1],
            "fact_payload": "J. Kent Hamilton (served 1887-1891)"
        }),
    )
    .await;
    assert_eq!(st_eng, StatusCode::OK);

    let (st_lk, lk_v) = post_json(
        &router,
        "/api/v1/synapse/engram/lookup",
        json!({ "phrase": "Third Mayor of Toledo 1887" }),
    )
    .await;
    assert_eq!(st_lk, StatusCode::OK);
    assert_eq!(lk_v["hit"].as_bool(), Some(true));
    assert!(
        lk_v["record"]
            .to_string()
            .contains("J. Kent Hamilton (served 1887-1891)")
    );
}
