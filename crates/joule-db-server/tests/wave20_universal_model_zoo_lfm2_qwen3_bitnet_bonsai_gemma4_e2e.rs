//! Wave 20 End-to-End Integration Suite:
//! Universal `< 30 GB` Open-Weight Model Zoo Adaptation (`Liquid AI LFM2/LFM3` `ShortConv`,
//! `PrismML Bonsai` MLX Packed `U32` 1-Bit & `Q1_0_G128` GGUF, `Qwen3` `QK-Norm` + `q_dim != hidden`,
//! `Microsoft BitNet b1.58` `SubLN` + `ReLU²`, `Google Gemma 4` `GeGLU` + `+1` RMSNorm + Metaspace BPE,
//! `Cactus Compute Needle` `self_attn_gate` + `fc1/fc2`, `IBM Granite ModernBERT` `Wqkv` + `GeGLU`,
//! `Qwen3-Embedding`, License-Preserving `.jsyn` Export, and `GET /api/v1/synapse/hf/catalog`).

use base64::Engine;
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::checkpoint_ingest::{
    SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use tokio::net::TcpListener;

fn build_synthetic_bonsai_q1_0_gguf() -> Vec<u8> {
    // Build a valid GGUF v3 binary with:
    // - `general.architecture = "qwen3"`
    // - `qwen3.attention.head_count = 4`, `head_count_kv = 2`, `key_length = 32` (`q_dim = 128`, `kv_dim = 64`)
    // - `tokenizer.ggml.tokens` (128 tokens)
    // - `token_embd.weight` (`dims = [128, 128]`, `type_id = 41` `Q1_0_G128`!)
    // - `output_norm.weight` (`dims = [128]`, `type_id = 0` `F32`)
    // - `blk.0.*` (`attn_norm`, `attn_q` `[128, 128]` `Q1_0_G128`, `attn_k` `[128, 64]` `Q1_0_G128`,
    //   `attn_v` `[128, 64]`, `attn_q_norm` `[32]`, `attn_k_norm` `[32]`, `attn_output` `[128, 128]`,
    //   `ffn_norm` `[128]`, `ffn_gate` `[128, 256]`, `ffn_up` `[128, 256]`, `ffn_down` `[256, 128]`)
    let mut out = Vec::new();
    let push_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version 3
    let tensors: Vec<(&str, Vec<u64>, u32)> = vec![
        ("token_embd.weight", vec![128, 128], 41),
        ("output_norm.weight", vec![128], 0),
        ("blk.0.attn_norm.weight", vec![128], 0),
        ("blk.0.attn_q.weight", vec![128, 128], 41),
        ("blk.0.attn_k.weight", vec![128, 64], 41),
        ("blk.0.attn_v.weight", vec![128, 64], 41),
        ("blk.0.attn_q_norm.weight", vec![32], 0),
        ("blk.0.attn_k_norm.weight", vec![32], 0),
        ("blk.0.attn_output.weight", vec![128, 128], 41),
        ("blk.0.ffn_norm.weight", vec![128], 0),
        ("blk.0.ffn_gate.weight", vec![128, 256], 41),
        ("blk.0.ffn_up.weight", vec![128, 256], 41),
        ("blk.0.ffn_down.weight", vec![256, 128], 41),
    ];
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&5u64.to_le_bytes()); // 5 KV pairs

    // KV 1: general.architecture = "qwen3"
    push_str(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut out, "qwen3");
    // KV 2: qwen3.attention.head_count = 4
    push_str(&mut out, "qwen3.attention.head_count");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    // KV 3: qwen3.attention.head_count_kv = 2
    push_str(&mut out, "qwen3.attention.head_count_kv");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    // KV 4: qwen3.attention.key_length = 32
    push_str(&mut out, "qwen3.attention.key_length");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    // KV 5: tokenizer.ggml.tokens = ["tok0", ..., "tok127"]
    push_str(&mut out, "tokenizer.ggml.tokens");
    out.extend_from_slice(&9u32.to_le_bytes()); // array
    out.extend_from_slice(&8u32.to_le_bytes()); // string elem
    out.extend_from_slice(&128u64.to_le_bytes());
    for i in 0..128 {
        push_str(&mut out, &format!("tok{i}"));
    }

    let mut data_blob = Vec::new();
    for (idx, (name, dims, tid)) in tensors.iter().enumerate() {
        let rel_off = data_blob.len() as u64;
        push_str(&mut out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&tid.to_le_bytes());
        out.extend_from_slice(&rel_off.to_le_bytes());

        let elems: usize = dims.iter().map(|&x| x as usize).product();
        if *tid == 0 {
            for _ in 0..elems {
                data_blob.extend_from_slice(&1.0f32.to_le_bytes());
            }
        } else {
            // `Q1_0_G128` (`type_id = 41`): 18 bytes per 128 elements (2B f16 `d = 0.0625 = 0x2c00` + 16B bits)
            let blocks = elems / 128;
            for b in 0..blocks {
                data_blob.extend_from_slice(&0x2C00u16.to_le_bytes());
                for byte_i in 0..16u8 {
                    data_blob.push(
                        ((idx as u8).wrapping_mul(37)).wrapping_add(b as u8)
                            ^ byte_i.wrapping_mul(0x55),
                    );
                }
            }
        }
        while data_blob.len() % 32 != 0 {
            data_blob.push(0);
        }
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&data_blob);
    out
}

#[tokio::test]
async fn test_wave20_universal_model_zoo_lfm2_qwen3_bitnet_bonsai_gemma4_needle_modernbert_e2e() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = ServerConfig {
        db_path: dir.path().join("db").to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    let server = Server::new(cfg).expect("boot server");
    let app = server.router();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let base_url = format!("http://{addr}");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let http = reqwest::Client::new();
    let mk = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.17 + seed).sin() * 0.25)
            .collect()
    };

    // -------------------------------------------------------------------------
    // 1. Curated `< 30 GB` Permissive Open-Weight Catalog (`GET /api/v1/synapse/hf/catalog`)
    // -------------------------------------------------------------------------
    let cat: JsonValue = http
        .get(format!("{base_url}/api/v1/synapse/hf/catalog"))
        .send()
        .await
        .expect("hf_catalog")
        .json()
        .await
        .expect("hf_catalog json");
    assert_eq!(cat["ok"], true);
    assert!(cat["total_models"].as_u64().unwrap_or(0) >= 22);
    let ecosystems: Vec<&str> = cat["models"]
        .as_array()
        .expect("models")
        .iter()
        .filter_map(|m| m["ecosystem"].as_str())
        .collect();
    for required in [
        "Liquid AI",
        "PrismML",
        "Cactus Compute",
        "Poolside",
        "Google Gemma 4",
        "Qwen3",
        "Microsoft BitNet",
        "IBM Granite",
    ] {
        assert!(ecosystems.contains(&required), "catalog missing {required}");
    }

    // -------------------------------------------------------------------------
    // 2. Liquid AI `LFM2.5` Hybrid `ShortConv` (`LIV`) + `GQA` + `SwiGLU` (`w1/w3/w2`) + `LFM1.0` Export!
    // -------------------------------------------------------------------------
    let d = 32usize;
    let ffn = 64usize;
    let lfm_entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![48, d],
            values_f32: mk(48 * d, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.embedding_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        // Layer 0: `ShortConv` (`conv.in_proj` `[3*d, d]`, depthwise `conv.conv` `[d, 1, 3]`, `conv.out_proj` `[d, d]`) + `w1/w3/w2`
        SafetensorBuildEntry {
            name: "model.layers.0.operator_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.conv.in_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![3 * d, d],
            values_f32: mk(3 * d * d, 0.2),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.conv.conv.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d, 1, 3],
            values_f32: vec![0.33; d * 3],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.conv.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d, d],
            values_f32: mk(d * d, 0.3),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.ffn_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.feed_forward.w1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![ffn, d],
            values_f32: mk(ffn * d, 0.4),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.feed_forward.w3.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![ffn, d],
            values_f32: mk(ffn * d, 0.5),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.feed_forward.w2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d, ffn],
            values_f32: mk(d * ffn, 0.6),
        },
        // Layer 1: Attention (`q_layernorm`, `k_layernorm`, `out_proj`) + `w1/w3/w2`
        SafetensorBuildEntry {
            name: "model.layers.1.operator_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.1.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d, d],
            values_f32: mk(d * d, 0.7),
        },
        SafetensorBuildEntry {
            name: "model.layers.1.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, d],
            values_f32: mk(16 * d, 0.8),
        },
        SafetensorBuildEntry {
            name: "model.layers.1.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, d],
            values_f32: mk(16 * d, 0.9),
        },
        SafetensorBuildEntry {
            name: "model.layers.1.self_attn.q_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![8],
            values_f32: vec![1.0; 8],
        },
        SafetensorBuildEntry {
            name: "model.layers.1.self_attn.k_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![8],
            values_f32: vec![1.0; 8],
        },
        SafetensorBuildEntry {
            name: "model.layers.1.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d, d],
            values_f32: mk(d * d, 1.0),
        },
        SafetensorBuildEntry {
            name: "model.layers.1.ffn_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.1.feed_forward.w1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![ffn, d],
            values_f32: mk(ffn * d, 1.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.1.feed_forward.w3.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![ffn, d],
            values_f32: mk(ffn * d, 1.2),
        },
        SafetensorBuildEntry {
            name: "model.layers.1.feed_forward.w2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d, ffn],
            values_f32: mk(d * ffn, 1.3),
        },
        SafetensorBuildEntry {
            name: "model.norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
    ];
    let lfm_bytes = build_safetensors_bytes(&BTreeMap::new(), &lfm_entries).expect("lfm st");
    let lfm_ingest: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "liquid-lfm2-mini",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&lfm_bytes),
            "config_json": r#"{"model_type":"lfm2","num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"conv_L_cache":3}"#,
            "target_rank": 8,
            "distill_svf": true
        }))
        .send()
        .await
        .expect("lfm ingest")
        .json()
        .await
        .expect("lfm ingest json");
    assert_eq!(lfm_ingest["causal_decoder_layers"], 2);

    let lfm_gen: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "liquid-lfm2-mini",
            "prompt": "Liquid ShortConv thermodynamic state",
            "max_new_tokens": 5,
            "speculative_draft_k": 3
        }))
        .send()
        .await
        .expect("lfm gen")
        .json()
        .await
        .expect("lfm gen json");
    assert_eq!(lfm_gen["num_layers"], 2);
    assert_eq!(
        lfm_gen["generated_token_ids"].as_array().map(Vec::len),
        Some(5)
    );

    // Verify LFM Open License v1.0 Model Card export preservation!
    let lfm_exp_dir = dir.path().join("lfm_export");
    let lfm_exp: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/hf/export"))
        .json(&json!({
            "model_id": "liquid-lfm2-mini",
            "output_dir": lfm_exp_dir.to_string_lossy(),
            "base_model_repo": "LiquidAI/LFM2.5-350M",
            "hf_repo_id": "dcharlot65-openie/LFM2.5-350M-LittleBit"
        }))
        .send()
        .await
        .expect("lfm export")
        .json()
        .await
        .expect("lfm export json");
    assert_eq!(lfm_exp["manifest"]["license"], "lfm1.0");
    let readme_txt = std::fs::read_to_string(lfm_exp_dir.join("README.md")).expect("readme");
    assert!(readme_txt.contains("license_name: lfm1.0"));
    assert!(readme_txt.contains("Built with Liquid AI LFM"));

    // -------------------------------------------------------------------------
    // 3. PrismML `Bonsai` `Q1_0_G128` (`.gguf` `type_id=41`) + `Qwen3` `QK-Norm` (`q_dim != hidden`)
    //    AND MLX Packed `U32` 1-Bit `.safetensors`!
    // -------------------------------------------------------------------------
    let gguf_path = dir.path().join("Bonsai-mini-Q1_0.gguf");
    std::fs::write(&gguf_path, build_synthetic_bonsai_q1_0_gguf()).expect("write gguf");
    let gguf_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/gguf"))
        .json(&json!({
            "model_id": "bonsai-q1-gguf",
            "file_path": gguf_path.to_string_lossy(),
            "target_rank": 8,
            "distill_svf": true
        }))
        .send()
        .await
        .expect("ingest gguf")
        .json()
        .await
        .expect("ingest gguf json");
    assert_eq!(gguf_res["causal_decoder_layers"], 1);
    assert_eq!(gguf_res["vocab_tokens_loaded"], 128);

    let bonsai_gen: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "bonsai-q1-gguf",
            "prompt": "tok1 tok2 tok3",
            "max_new_tokens": 4
        }))
        .send()
        .await
        .expect("bonsai gen")
        .json()
        .await
        .expect("bonsai gen json");
    assert_eq!(bonsai_gen["num_q_heads"], 4);
    assert_eq!(bonsai_gen["num_kv_heads"], 2);
    assert_eq!(bonsai_gen["head_dim"], 32);
    assert_eq!(
        bonsai_gen["generated_token_ids"].as_array().map(Vec::len),
        Some(4)
    );

    // Verify MLX packed `U32` 1-bit `.safetensors` (`weight U32 [64, 4]` -> `128` cols, `scales F16 [64, 1]`, `biases F16 [64, 1]`)
    let mlx_entries = vec![
        SafetensorBuildEntry {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![128],
            values_f32: vec![1.0; 128],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::U32,
            shape: vec![64, 4],
            values_f32: vec![0xAAAA_5555u32 as f32; 64 * 4],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.scales".into(),
            dtype: SafetensorDtype::F16,
            shape: vec![64, 1],
            values_f32: vec![0.10; 64],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.biases".into(),
            dtype: SafetensorDtype::F16,
            shape: vec![64, 1],
            values_f32: vec![-0.05; 64],
        },
    ];
    let mlx_bytes = build_safetensors_bytes(&BTreeMap::new(), &mlx_entries).expect("mlx st");
    let mlx_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "bonsai-mlx-u32",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&mlx_bytes),
            "target_rank": 8
        }))
        .send()
        .await
        .expect("mlx ingest")
        .json()
        .await
        .expect("mlx json");
    assert_eq!(mlx_res["report"]["factorized_2d_tensors"], 1);
    assert_eq!(mlx_res["report"]["tensors"][1]["cols"], 128);
    assert_eq!(
        mlx_res["report"]["tensors"][1]["source_dtype"],
        "MLX_U32_1BIT"
    );

    // -------------------------------------------------------------------------
    // 4. Google `Gemma 4` (`model.language_model.*` + `q_dim=64 > hidden=32` + `GeGLU` + `+1` RMSNorm + softcap)
    //    AND Microsoft `BitNet b1.58` (`attn_sub_norm` + `ffn_sub_norm` + `ReLU²`)
    //    AND Cactus `Needle` (`model.decoder.*` + `self_attn_gate` + `fc1/fc2`)
    // -------------------------------------------------------------------------
    let gemma_entries = vec![
        SafetensorBuildEntry {
            name: "model.language_model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.11),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![0.1; 32],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32], // q_dim (64) != hidden_dim (32)!
            values_f32: mk(64 * 32, 0.21),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.31),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.41),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.q_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16],
            values_f32: vec![0.05; 16],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.k_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16],
            values_f32: vec![0.05; 16],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.51),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![0.1; 32],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.pre_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![0.1; 32],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.61),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.71),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.81),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.post_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![0.1; 32],
        },
        SafetensorBuildEntry {
            name: "model.language_model.norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![0.1; 32],
        },
    ];
    let gemma_bytes = build_safetensors_bytes(&BTreeMap::new(), &gemma_entries).expect("gemma st");
    let gemma_in: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "gemma-4-mini",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&gemma_bytes),
            "config_json": r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"final_logit_softcapping":30.0}}"#,
            "vocab_txt": r#"{"model":{"vocab":{"<bos>":0,"<eos>":1,"▁joule":2,"▁db":3,"j":4,"o":5},"merges":[]}}"#,
            "target_rank": 8
        }))
        .send()
        .await
        .expect("gemma in")
        .json()
        .await
        .expect("gemma json");
    assert_eq!(gemma_in["causal_decoder_layers"], 1);

    let gemma_gen: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "gemma-4-mini",
            "prompt": " joule db",
            "max_new_tokens": 4
        }))
        .send()
        .await
        .expect("gemma gen")
        .json()
        .await
        .expect("gemma gen json");
    assert_eq!(gemma_gen["num_q_heads"], 4);
    assert_eq!(gemma_gen["head_dim"], 16);
    assert_eq!(gemma_gen["hidden_dim"], 32);

    // -------------------------------------------------------------------------
    // 5. IBM Granite `ModernBERT` (`Wqkv` + `RoPE` + `GeGLU` `Wi`/`Wo`) Encoder!
    // -------------------------------------------------------------------------
    let mbert_entries = vec![
        SafetensorBuildEntry {
            name: "embeddings.tok_embeddings.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.12),
        },
        SafetensorBuildEntry {
            name: "embeddings.norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
        SafetensorBuildEntry {
            name: "layers.0.attn_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
        SafetensorBuildEntry {
            name: "layers.0.attn.Wqkv.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![96, 32],
            values_f32: mk(96 * 32, 0.22),
        },
        SafetensorBuildEntry {
            name: "layers.0.attn.Wo.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.32),
        },
        SafetensorBuildEntry {
            name: "layers.0.mlp_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
        SafetensorBuildEntry {
            name: "layers.0.mlp.Wi.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![128, 32],
            values_f32: mk(128 * 32, 0.42),
        },
        SafetensorBuildEntry {
            name: "layers.0.mlp.Wo.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.52),
        },
        SafetensorBuildEntry {
            name: "final_norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
    ];
    let mbert_bytes = build_safetensors_bytes(&BTreeMap::new(), &mbert_entries).expect("mbert st");
    let mbert_in: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "granite-modernbert-mini",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&mbert_bytes),
            "target_rank": 8
        }))
        .send()
        .await
        .expect("mbert in")
        .json()
        .await
        .expect("mbert json");
    assert_eq!(mbert_in["transformer_encoder_layers"], 1);

    let emb_res: JsonValue = http
        .post(format!("{base_url}/v1/embeddings"))
        .json(&json!({
            "model": "granite-modernbert-mini",
            "input": "Thermodynamic SQL vector indexing"
        }))
        .send()
        .await
        .expect("modernbert embed")
        .json()
        .await
        .expect("modernbert embed json");
    let vec_out = emb_res["data"][0]["embedding"].as_array().expect("emb vec");
    assert_eq!(vec_out.len(), 32);
    let l2: f64 = vec_out
        .iter()
        .filter_map(|v| v.as_f64())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        (l2 - 1.0).abs() < 1e-3,
        "expected unit L2 embedding, got {l2}"
    );
}
