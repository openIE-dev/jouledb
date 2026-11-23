//! Wave 15 End-to-End Integration Suite:
//! - Full Causal Autoregressive Decoder (`RoPE` + `GQA` + `RMSNorm` + `SwiGLU` + Pure-Rust `BpeTokenizer`)
//! - Sibling `tokenizer.json` (BPE merges + special tokens) & `config.json` Auto-Discovery
//! - Materialized `Mooncake` Prefix KV-Cache Silicon Projection Skip Across Multi-Turn Prompts
//! - Drop-In OpenAI `/v1/chat/completions` & Anthropic `/v1/messages` Causal Generation + Live HW Fusion

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

fn write_synthetic_smollm_checkpoint(model_dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::create_dir_all(model_dir).expect("mkdir");
    let tok_json = r#"{
        "model": {
            "vocab": {
                "<|endoftext|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "j": 3, "o": 4, "u": 5, "l": 6, "e": 7,
                "Ġ": 8, "d": 9, "b": 10,
                "jo": 11, "ul": 12, "joul": 13, "joule": 14,
                "Ġd": 15, "Ġdb": 16
            },
            "merges": [
                "j o", "u l", "jo ul", "joul e", "Ġ d", "Ġd b"
            ]
        }
    }"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).expect("write tokenizer.json");

    let cfg_json = r#"{
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rope_theta": 100000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true
    }"#;
    std::fs::write(model_dir.join("config.json"), cfg_json).expect("write config.json");

    let d = 32usize;
    let kv_d = 16usize;
    let ffn = 64usize;
    let vsize = 32usize;
    let mk_2d = |name: &str, r: usize, c: usize, seed: f32| SafetensorBuildEntry {
        name: name.to_string(),
        dtype: SafetensorDtype::F32,
        shape: vec![r, c],
        values_f32: (0..(r * c))
            .map(|i| ((i as f32) * 0.21 + seed).sin() * 0.25)
            .collect(),
    };
    let mk_1d = |name: &str, n: usize, val: f32| SafetensorBuildEntry {
        name: name.to_string(),
        dtype: SafetensorDtype::F32,
        shape: vec![n],
        values_f32: vec![val; n],
    };

    let mut entries = vec![
        mk_2d("model.embed_tokens.weight", vsize, d, 0.1),
        mk_1d("model.norm.weight", d, 1.0),
    ];
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        entries.extend([
            mk_1d(&format!("{p}.input_layernorm.weight"), d, 1.0),
            mk_2d(&format!("{p}.self_attn.q_proj.weight"), d, d, 0.2 + l as f32),
            mk_2d(&format!("{p}.self_attn.k_proj.weight"), kv_d, d, 0.3 + l as f32),
            mk_2d(&format!("{p}.self_attn.v_proj.weight"), kv_d, d, 0.4 + l as f32),
            mk_2d(&format!("{p}.self_attn.o_proj.weight"), d, d, 0.5 + l as f32),
            mk_1d(&format!("{p}.post_attention_layernorm.weight"), d, 1.0),
            mk_2d(&format!("{p}.mlp.gate_proj.weight"), ffn, d, 0.6 + l as f32),
            mk_2d(&format!("{p}.mlp.up_proj.weight"), ffn, d, 0.7 + l as f32),
            mk_2d(&format!("{p}.mlp.down_proj.weight"), d, ffn, 0.8 + l as f32),
        ]);
    }
    let bytes = build_safetensors_bytes(&BTreeMap::new(), &entries).expect("safetensors");
    let st_path = model_dir.join("smollm-2l.safetensors");
    std::fs::write(&st_path, bytes).expect("write safetensors");
    st_path
}

#[tokio::test]
async fn test_causal_decoder_bpe_mooncake_kv_and_openai_anthropic_disintermediation_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(tmp.path());
    let st_path = write_synthetic_smollm_checkpoint(&tmp.path().join("smol_assets"));

    // 1. Ingest `.safetensors` — sibling `tokenizer.json` (BPE) and `config.json` auto-discovered!
    let (status, ingest) = post_json(
        &server,
        "/api/v1/synapse/ingest/safetensors",
        json!({
            "model_id": "smollm-2l",
            "file_path": st_path.to_string_lossy(),
            "target_rank": 16,
            "max_outliers": 4
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "ingest: {ingest}");
    assert_eq!(ingest["vocab_tokens_loaded"], 17);
    assert_eq!(ingest["bpe_merges_loaded"], 6);
    assert_eq!(ingest["causal_decoder_layers"], 2);

    // 2. Direct `/api/v1/synapse/decoder/generate`: Turn 1 populates Mooncake 8-token prefix block;
    //    Turn 2 hits materialized `(K, V)` cache and skips >=100 prefill projections!
    let long_prompt = "<|im_start|>joule dbjoule dbjoule dbjoule dbjoule<|im_end|>";
    let (status, gen1) = post_json(
        &server,
        "/api/v1/synapse/decoder/generate",
        json!({
            "model_id": "smollm-2l",
            "prompt": long_prompt,
            "max_new_tokens": 5,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "gen1: {gen1}");
    assert_eq!(gen1["num_layers"], 2);
    assert_eq!(gen1["num_q_heads"], 4);
    assert_eq!(gen1["num_kv_heads"], 2);

    let (status, gen2) = post_json(
        &server,
        "/api/v1/synapse/decoder/generate",
        json!({
            "model_id": "smollm-2l",
            "prompt": long_prompt,
            "max_new_tokens": 5,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(gen1["generated_token_ids"], gen2["generated_token_ids"]);
    assert_eq!(gen2["kv_prefix_receipt"]["matched_prefix_tokens"], 8);
    assert!(
        gen2["linear_projections_executed"].as_u64().unwrap_or(9999) + 100
            <= gen1["linear_projections_executed"].as_u64().unwrap_or(0)
    );

    // 3. Inject live hardware telemetry & verify OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`
    //    automatically dispatch `tier4_littlebit_causal_decoder`!
    let _ = post_json(
        &server,
        "/api/v1/ward/hardware-telemetry",
        json!({
            "board_power_mw": 19_200,
            "cpu_utilization": 0.45,
            "gpu_utilization": 0.60,
            "npu_utilization": 0.0,
            "memory_pressure": 0.30
        }),
    )
    .await;

    let (status, chat) = post_json(
        &server,
        "/v1/chat/completions",
        json!({
            "model": "smollm-2l",
            "messages": [{ "role": "user", "content": "joule db" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "chat: {chat}");
    assert_eq!(
        chat["x_jouledb_energy"]["cascade_tier"],
        "tier4_littlebit_causal_decoder"
    );
    assert_eq!(
        chat["x_jouledb_energy"]["kernel"],
        "littlebit_svf_causal_2L_4Q_2KV"
    );

    let (status, anth) = post_json(
        &server,
        "/v1/messages",
        json!({
            "model": "smollm-2l",
            "max_tokens": 16,
            "messages": [{ "role": "user", "content": "joule" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "anth: {anth}");
    assert_eq!(
        anth["x_jouledb_energy"]["cascade_tier"],
        "tier4_littlebit_causal_decoder"
    );
}
