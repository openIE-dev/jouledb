//! Wave 14 End-to-End Integration Suite:
//! - Full Multi-Layer Transformer Encoder + Pure-Rust `WordPieceTokenizer` Inside `JouleSynapse`
//!   (auto-loading sibling `vocab.txt` / `tokenizer.json` alongside `.safetensors` + `/v1/embeddings`)
//! - Live Hardware Sensor Fusion (`HardwarePowerSample` / `EnergySnapshot`) Into the 6-Rail `nJ` Fabric Ledger

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::checkpoint_ingest::{
    SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use tower::ServiceExt;

fn boot_test_server(dir: &std::path::Path) -> Server {
    let cfg = ServerConfig {
        db_path: dir.to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    Server::new(cfg).expect("boot server")
}

async fn post_json(server: &Server, uri: &str, body: JsonValue) -> (StatusCode, JsonValue) {
    let app = server.router();
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("post req");
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    let parsed: JsonValue = serde_json::from_slice(&bytes).expect("parse json");
    (status, parsed)
}

async fn get_json(server: &Server, uri: &str) -> (StatusCode, JsonValue) {
    let app = server.router();
    let req = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .expect("get req");
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    let parsed: JsonValue = serde_json::from_slice(&bytes).expect("parse json");
    (status, parsed)
}

fn write_synthetic_minilm_checkpoint(model_dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::create_dir_all(model_dir).expect("mkdir model_dir");
    let vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\nthermodynamic\ndatabase\njoule\n##s\nquantum\nlaser\n";
    std::fs::write(model_dir.join("vocab.txt"), vocab).expect("write vocab.txt");

    let d = 16usize;
    let ffn = 32usize;
    let mk_2d = |name: &str, r: usize, c: usize, seed: f32| SafetensorBuildEntry {
        name: name.to_string(),
        dtype: SafetensorDtype::F32,
        shape: vec![r, c],
        values_f32: (0..(r * c))
            .map(|i| ((i as f32) * 0.17 + seed).sin() * 0.3)
            .collect(),
    };
    let mk_1d = |name: &str, n: usize, val: f32| SafetensorBuildEntry {
        name: name.to_string(),
        dtype: SafetensorDtype::F32,
        shape: vec![n],
        values_f32: vec![val; n],
    };

    let mut entries = vec![
        mk_2d("embeddings.word_embeddings.weight", 32, d, 0.1),
        mk_2d("embeddings.position_embeddings.weight", 32, d, 0.2),
        mk_2d("embeddings.token_type_embeddings.weight", 2, d, 0.0),
        mk_1d("embeddings.LayerNorm.weight", d, 1.0),
        mk_1d("embeddings.LayerNorm.bias", d, 0.0),
    ];
    for l in 0..2 {
        let p = format!("encoder.layer.{l}");
        entries.extend([
            mk_2d(&format!("{p}.attention.self.query.weight"), d, d, 0.3 + l as f32),
            mk_1d(&format!("{p}.attention.self.query.bias"), d, 0.0),
            mk_2d(&format!("{p}.attention.self.key.weight"), d, d, 0.4 + l as f32),
            mk_1d(&format!("{p}.attention.self.key.bias"), d, 0.0),
            mk_2d(&format!("{p}.attention.self.value.weight"), d, d, 0.5 + l as f32),
            mk_1d(&format!("{p}.attention.self.value.bias"), d, 0.0),
            mk_2d(&format!("{p}.attention.output.dense.weight"), d, d, 0.6 + l as f32),
            mk_1d(&format!("{p}.attention.output.dense.bias"), d, 0.0),
            mk_1d(&format!("{p}.attention.output.LayerNorm.weight"), d, 1.0),
            mk_1d(&format!("{p}.attention.output.LayerNorm.bias"), d, 0.0),
            mk_2d(&format!("{p}.intermediate.dense.weight"), ffn, d, 0.7 + l as f32),
            mk_1d(&format!("{p}.intermediate.dense.bias"), ffn, 0.0),
            mk_2d(&format!("{p}.output.dense.weight"), d, ffn, 0.8 + l as f32),
            mk_1d(&format!("{p}.output.dense.bias"), d, 0.0),
            mk_1d(&format!("{p}.output.LayerNorm.weight"), d, 1.0),
            mk_1d(&format!("{p}.output.LayerNorm.bias"), d, 0.0),
        ]);
    }
    let bytes = build_safetensors_bytes(&BTreeMap::new(), &entries).expect("build safetensors");
    let st_path = model_dir.join("mini-bert.safetensors");
    std::fs::write(&st_path, bytes).expect("write safetensors");
    st_path
}

#[tokio::test]
async fn test_multilayer_littlebit_transformer_auto_vocab_and_openai_embeddings_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(tmp.path());
    let st_path = write_synthetic_minilm_checkpoint(&tmp.path().join("model_assets"));

    // 1. Ingest `.safetensors` via `file_path` — sibling `vocab.txt` auto-discovered!
    let (status, ingest) = post_json(
        &server,
        "/api/v1/synapse/ingest/safetensors",
        json!({
            "model_id": "mini-bert-2l",
            "file_path": st_path.to_string_lossy(),
            "target_rank": 8,
            "max_outliers": 4
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "ingest resp: {ingest}");
    assert_eq!(ingest["vocab_tokens_loaded"], 10);
    assert_eq!(ingest["transformer_encoder_layers"], 2);

    // 2. Call OpenAI `POST /v1/embeddings` -> executes full 2-layer 4-head LittleBit-2 Transformer!
    let (status, emb_resp) = post_json(
        &server,
        "/v1/embeddings",
        json!({
            "model": "mini-bert-2l",
            "input": [
                "thermodynamic joules database",
                "quantum laser",
                "thermodynamic joules database"
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "emb_resp: {emb_resp}");
    assert_eq!(
        emb_resp["x_jouledb_energy"]["kernel"],
        "littlebit_svf_transformer_2L_4H"
    );

    let v0: Vec<f64> = emb_resp["data"][0]["embedding"]
        .as_array()
        .expect("v0 array")
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0))
        .collect();
    let v1: Vec<f64> = emb_resp["data"][1]["embedding"]
        .as_array()
        .expect("v1 array")
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0))
        .collect();
    let v2: Vec<f64> = emb_resp["data"][2]["embedding"]
        .as_array()
        .expect("v2 array")
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0))
        .collect();

    assert_eq!(v0.len(), 16);
    let norm0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm0 - 1.0).abs() < 1e-4);

    // Exact deterministic match on identical input (`v0 == v2`), distinct on `v1`!
    let cos_same: f64 = v0.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let cos_diff: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
    assert!((cos_same - 1.0).abs() < 1e-5);
    assert!(cos_diff < 0.9999);

    // 3. Direct `/api/v1/synapse/encoder/embed` endpoint exposes token_ids & linear projection count
    let (status, direct) = post_json(
        &server,
        "/api/v1/synapse/encoder/embed",
        json!({
            "model_id": "mini-bert-2l",
            "text": "Thermodynamic Joules"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // [CLS]=2, thermodynamic=4, joule=6, ##s=7, [SEP]=3 -> 5 tokens * 4 QKV/O/FFN1/FFN2? (6 linear/token/layer * 2 layers = 60!)
    assert_eq!(direct["token_ids"], json!([2, 4, 6, 7, 3]));
    assert_eq!(direct["linear_projections_executed"], 60);
}

#[tokio::test]
async fn test_live_hardware_sensor_fusion_into_6_rail_nj_ledger_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(tmp.path());
    let st_path = write_synthetic_minilm_checkpoint(&tmp.path().join("hw_assets"));

    let (status, _) = post_json(
        &server,
        "/api/v1/synapse/ingest/safetensors",
        json!({
            "model_id": "mini-hw-2l",
            "file_path": st_path.to_string_lossy(),
            "target_rank": 8
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Inject live hardware telemetry (18.5 W board power, 30% CPU, 75% GPU, 25% UMA memory)
    let (status, hw_set) = post_json(
        &server,
        "/api/v1/ward/hardware-telemetry",
        json!({
            "board_power_mw": 18_500,
            "cpu_utilization": 0.30,
            "gpu_utilization": 0.75,
            "npu_utilization": 0.05,
            "memory_pressure": 0.25
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(hw_set["active_hardware_sample"]["board_power_mw"], 18_500);

    // Run `/v1/embeddings` and verify `hardware_sensor_fused: true` + non-zero 6-rail ledger!
    let (status, emb) = post_json(
        &server,
        "/v1/embeddings",
        json!({
            "model": "mini-hw-2l",
            "input": "thermodynamic database"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(emb["x_jouledb_energy"]["hardware_sensor_fused"], true);
    assert!(
        emb["x_jouledb_energy"]["aggregate_rails"]["peak_power_mw"]
            .as_u64()
            .unwrap_or(0)
            >= 18_500
    );

    let (status, ledger) = get_json(&server, "/api/v1/ward/ledger").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ledger["hardware_sensor_fused"], true);
    assert_eq!(
        ledger["live_hardware_telemetry"]["board_power_mw"],
        18_500
    );
}
