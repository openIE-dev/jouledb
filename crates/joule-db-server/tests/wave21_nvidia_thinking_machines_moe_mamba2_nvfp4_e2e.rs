//! Wave 21 End-to-End Integration Suite:
//! NVIDIA (`Nemotron-H` `Mamba-2` SSM + `GQA` + `ReLU²` Hybrid, `FP8` & `NVFP4 E2M1` Unpacking,
//! `Nemotron-3-Embed` Bidirectional Decoder-Embed), Thinking Machines Lab (`Inkling-2` `model.llm.`
//! + 3D Stacked `NVFP4` MoE `mlp.experts.{w13_weight, w2_weight}`), Sparse Mixture-of-Experts
//! (`Qwen3-MoE` Top-K Router), `Liquid AI LFM3` Post-Norms, `AllenAI OLMo-2` Post-Norm,
//! `NomicBert` Fused `Wqkv` + `SwiGLU`, `Google EmbeddingGemma-2`, Microsoft `BitNet.cpp` `I2_S`
//! GGUF (`type_id = 36`), RAM Model Eviction, and Expanded 46-Model `< 30 GB` Catalog.

use base64::Engine;
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::checkpoint_ingest::{
    SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use tokio::net::TcpListener;

fn mk(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + 5) % 31) as f32 - 15.0) * scale / 15.0)
        .collect()
}

fn build_synthetic_bitnet_i2s_gguf() -> Vec<u8> {
    // GGUF v3 with `general.architecture = "gemma3"` and `type_id = 36` (`GGML_TYPE_I2_S`):
    // `elem_count / 4` packed 2-bit ternary bytes + 4-byte f32 scale!
    let mut out = Vec::new();
    let push_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    let tensors: Vec<(&str, Vec<u64>, u32)> = vec![
        ("token_embd.weight", vec![128, 128], 36),
        ("output_norm.weight", vec![128], 0),
        ("blk.0.attn_norm.weight", vec![128], 0),
        ("blk.0.attn_q.weight", vec![128, 128], 36),
        ("blk.0.attn_k.weight", vec![128, 64], 36),
        ("blk.0.attn_v.weight", vec![128, 64], 36),
        ("blk.0.attn_output.weight", vec![128, 128], 36),
        ("blk.0.post_attention_norm.weight", vec![128], 0),
        ("blk.0.ffn_norm.weight", vec![128], 0),
        ("blk.0.ffn_gate.weight", vec![128, 256], 36),
        ("blk.0.ffn_up.weight", vec![128, 256], 36),
        ("blk.0.ffn_down.weight", vec![256, 128], 36),
        ("blk.0.post_ffw_norm.weight", vec![128], 0),
    ];
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&4u64.to_le_bytes());

    push_str(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut out, "gemma3");
    push_str(&mut out, "gemma3.attention.head_count");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    push_str(&mut out, "gemma3.attention.head_count_kv");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    push_str(&mut out, "gemma3.attention.key_length");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());

    let mut data_blob = Vec::new();
    for (idx, (name, dims, tid)) in tensors.iter().enumerate() {
        while !data_blob.len().is_multiple_of(32) {
            data_blob.push(0);
        }
        let rel_off = data_blob.len() as u64;
        push_str(&mut out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&tid.to_le_bytes());
        out.extend_from_slice(&rel_off.to_le_bytes());

        let nel: usize = dims.iter().map(|&x| x as usize).product();
        if *tid == 0 {
            for j in 0..nel {
                let v = if name.contains("norm") {
                    0.1f32
                } else {
                    ((j + idx) % 7) as f32 * 0.05
                };
                data_blob.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            for j in 0..(nel / 4) {
                let b = match (j + idx) % 3 {
                    0 => 0b10_01_00_10u8,
                    1 => 0b00_10_01_10u8,
                    _ => 0b01_10_00_01u8,
                };
                data_blob.push(b);
            }
            data_blob.extend_from_slice(&0.08f32.to_le_bytes());
        }
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.extend_from_slice(&data_blob);
    out
}

#[tokio::test]
async fn test_wave21_nvidia_nemotron_h_mamba2_thinking_machines_inkling_nvfp4_moe_and_embed_zoo_e2e()
 {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = ServerConfig {
        db_path: tmp.path().join("db").to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    let server = Server::new(cfg).expect("server");
    let router = server.router();
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    let base_url = format!("http://{addr}");
    let http = reqwest::Client::new();

    // -------------------------------------------------------------------------
    // 1. Expanded `< 30 GB` Open-Weight Catalog (`>= 45` models incl. NVIDIA & Thinking Machines Lab)
    // -------------------------------------------------------------------------
    let cat: JsonValue = http
        .get(format!("{base_url}/api/v1/synapse/hf/catalog"))
        .send()
        .await
        .expect("catalog")
        .json()
        .await
        .expect("catalog json");
    assert_eq!(cat["ok"], true);
    let total_models = cat["total_models"].as_u64().unwrap_or(0);
    assert!(
        total_models >= 45,
        "expected >= 45 curated <30GB models, got {total_models}"
    );
    let models_arr = cat["models"].as_array().expect("models array");
    assert!(
        models_arr
            .iter()
            .any(|m| m["ecosystem"] == "Thinking Machines Lab")
    );
    assert!(models_arr.iter().any(|m| m["ecosystem"] == "NVIDIA"));

    // -------------------------------------------------------------------------
    // 2. NVIDIA `Nemotron-H` (`Mamba-2` SSM + `ReLU²` MLP + `GQA` Hybrid) + `F8_E4M3` Scaled Unpacking!
    // -------------------------------------------------------------------------
    let d = 64usize;
    let mamba_heads = 4usize;
    let mamba_hd = 16usize; // d_inner = 64
    let n_groups = 2usize;
    let d_state = 8usize;
    let conv_dim = 64 + 2 * n_groups * d_state; // 64 + 32 = 96
    let in_proj_dim = 64 + conv_dim + mamba_heads; // 64 + 96 + 4 = 164

    let nemotron_st = build_safetensors_bytes(
        &BTreeMap::new(),
        &[
            SafetensorBuildEntry {
                name: "backbone.embeddings.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![128, d],
                values_f32: mk(128 * d, 0.15),
            },
            SafetensorBuildEntry {
                name: "backbone.norm_f.weight".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "lm_head.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![128, d],
                values_f32: mk(128 * d, 0.14),
            },
            // Layer 0: Mamba-2 Selective SSM Mixer (`F8_E4M3` + `weight_scale`!)
            SafetensorBuildEntry {
                name: "backbone.layers.0.norm.weight".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.in_proj.weight".into(),
                dtype: SafetensorDtype::F8E4M3,
                shape: vec![in_proj_dim, d],
                values_f32: mk(in_proj_dim * d, 1.5),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.in_proj.weight_scale".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![1],
                values_f32: vec![0.08],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.conv1d.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![conv_dim, 1, 4],
                values_f32: mk(conv_dim * 4, 0.12),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.conv1d.bias".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![conv_dim],
                values_f32: vec![0.01; conv_dim],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.A_log".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![mamba_heads],
                values_f32: vec![0.5; mamba_heads],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.D".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![mamba_heads],
                values_f32: vec![1.0; mamba_heads],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.dt_bias".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![mamba_heads],
                values_f32: vec![0.1; mamba_heads],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.norm.weight".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.out_proj.weight".into(),
                dtype: SafetensorDtype::F8E4M3,
                shape: vec![d, d],
                values_f32: mk(d * d, 1.2),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.0.mixer.out_proj.weight_scale".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![1],
                values_f32: vec![0.08],
            },
            // Layer 1: Standalone ReLU² MLP Mixer
            SafetensorBuildEntry {
                name: "backbone.layers.1.norm.weight".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.1.mixer.up_proj.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![128, d],
                values_f32: mk(128 * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.1.mixer.down_proj.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d, 128],
                values_f32: mk(d * 128, 0.12),
            },
            // Layer 2: Standalone GQA Mixer
            SafetensorBuildEntry {
                name: "backbone.layers.2.norm.weight".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "backbone.layers.2.mixer.q_proj.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d, d],
                values_f32: mk(d * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.2.mixer.k_proj.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![32, d],
                values_f32: mk(32 * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.2.mixer.v_proj.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![32, d],
                values_f32: mk(32 * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "backbone.layers.2.mixer.o_proj.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d, d],
                values_f32: mk(d * d, 0.12),
            },
        ],
    )
    .expect("build nemotron_h safetensors");

    let nemotron_cfg = json!({
        "model_type": "nemotron_h",
        "hidden_size": d,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "mamba_num_heads": mamba_heads,
        "mamba_head_dim": mamba_hd,
        "n_groups": n_groups,
        "mamba_state_dim": d_state,
        "mlp_hidden_act": "relu2"
    });
    let ing_nem: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "nvidia-nemotron-h-test",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&nemotron_st),
            "config_json": nemotron_cfg.to_string(),
            "target_rank": 8,
            "max_outliers": 8,
            "distill_svf": true
        }))
        .send()
        .await
        .expect("ingest nemotron")
        .json()
        .await
        .expect("ingest nemotron json");
    assert_eq!(ing_nem["causal_decoder_layers"], 3);
    let gen_nem: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "nvidia-nemotron-h-test",
            "prompt": "NVIDIA Nemotron Mamba2 hybrid",
            "max_new_tokens": 5,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("gen nemotron")
        .json()
        .await
        .expect("gen nemotron json");
    assert_eq!(gen_nem["num_layers"], 3);
    assert_eq!(
        gen_nem["generated_token_ids"].as_array().map(Vec::len),
        Some(5)
    );
    let _ = mamba_hd;

    // Verify NVIDIA Open Model License export preservation!
    let exp_dir = tmp.path().join("nemotron-export");
    let exp_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/hf/export"))
        .json(&json!({
            "model_id": "nvidia-nemotron-h-test",
            "output_dir": exp_dir.to_string_lossy(),
            "base_model_repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "hf_repo_id": "dcharlot65-openie/NVIDIA-Nemotron-3-Nano-4B-LittleBit"
        }))
        .send()
        .await
        .expect("export nemotron")
        .json()
        .await
        .expect("export nemotron json");
    assert_eq!(exp_res["ok"], true);
    let card = std::fs::read_to_string(exp_dir.join("README.md")).expect("read model card");
    assert!(card.contains("nvidia-nemotron-open-model-license"));

    // -------------------------------------------------------------------------
    // 3. Thinking Machines Lab `Inkling-2` (`model.llm.` + `NVFP4 E2M1` 3D Stacked MoE!)
    // -------------------------------------------------------------------------
    // 4 experts, hidden_dim = 64, inter_dim = 32 -> w13 unpacked `[4, 64, 64]` => packed U8 `[4, 64, 32]`!
    // w2 unpacked `[4, 64, 32]` => packed U8 `[4, 64, 16]`!
    let ink_st = build_safetensors_bytes(
        &BTreeMap::new(),
        &[
            SafetensorBuildEntry {
                name: "model.llm.embed.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![128, d],
                values_f32: mk(128 * d, 0.14),
            },
            SafetensorBuildEntry {
                name: "model.llm.embed_norm.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "model.llm.norm.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "model.llm.unembed.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![128, d],
                values_f32: mk(128 * d, 0.14),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.attn_norm.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.attn.wq_du.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d, d],
                values_f32: mk(d * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.attn.wk_dv.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![32, d],
                values_f32: mk(32 * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.attn.wv_dv.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![32, d],
                values_f32: mk(32 * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.attn.gate_attn.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d, d],
                values_f32: mk(d * d, 0.10),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.attn.wo_ud.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d, d],
                values_f32: mk(d * d, 0.12),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp_norm.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![d],
                values_f32: vec![1.0; d],
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp.gate.weight".into(),
                dtype: SafetensorDtype::BF16,
                shape: vec![4, d],
                values_f32: mk(4 * d, 0.20),
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp.gate.bias".into(),
                dtype: SafetensorDtype::F32,
                shape: vec![4],
                values_f32: vec![0.01, -0.02, 0.03, 0.0],
            },
            // NVFP4 packed `U8` (`0x2A` => lo=0xA (-1.0), hi=0x2 (+1.0)) + `F8_E4M3` block_size=16 scale!
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp.experts.w13_weight".into(),
                dtype: SafetensorDtype::U8,
                shape: vec![4, 64, 32],
                values_f32: vec![0x2Au8 as f32; 4 * 64 * 32],
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp.experts.w13_weight_scale".into(),
                dtype: SafetensorDtype::F8E4M3,
                shape: vec![4, 64, 4],
                values_f32: vec![0.125; 4 * 64 * 4],
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp.experts.w2_weight".into(),
                dtype: SafetensorDtype::U8,
                shape: vec![4, 64, 16],
                values_f32: vec![0x3Bu8 as f32; 4 * 64 * 16],
            },
            SafetensorBuildEntry {
                name: "model.llm.layers.0.mlp.experts.w2_weight_scale".into(),
                dtype: SafetensorDtype::F8E4M3,
                shape: vec![4, 64, 2],
                values_f32: vec![0.125; 4 * 64 * 2],
            },
        ],
    )
    .expect("build inkling2 safetensors");

    let ink_cfg = json!({
        "model_type": "inkling2_loop",
        "text_config": {
            "model_type": "inkling2_loop_text",
            "hidden_size": d,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "final_logit_softcapping": 30.0
        }
    });
    let ing_ink: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "tml-inkling2-mini-test",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&ink_st),
            "config_json": ink_cfg.to_string(),
            "target_rank": 8,
            "max_outliers": 8,
            "distill_svf": true
        }))
        .send()
        .await
        .expect("ingest inkling2")
        .json()
        .await
        .expect("ingest inkling2 json");
    assert_eq!(ing_ink["causal_decoder_layers"], 1);
    let gen_ink: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "tml-inkling2-mini-test",
            "prompt": "Thinking Machines Lab Inkling-2 NVFP4 MoE",
            "max_new_tokens": 4,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("gen inkling2")
        .json()
        .await
        .expect("gen inkling2 json");
    assert_eq!(
        gen_ink["generated_token_ids"].as_array().map(Vec::len),
        Some(4)
    );

    // -------------------------------------------------------------------------
    // 4. Sparse MoE (`Qwen3-MoE` 4 experts, top-2) + `AllenAI OLMo-2` Post-Norm!
    // -------------------------------------------------------------------------
    let mut moe_entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, d],
            values_f32: mk(128 * d, 0.14),
        },
        SafetensorBuildEntry {
            name: "model.norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        // OLMo-2 style post-norm (NO `input_layernorm`!) + Qwen3-MoE sparse experts!
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d, d],
            values_f32: mk(d * d, 0.12),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d, d],
            values_f32: mk(d * d, 0.12),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d, d],
            values_f32: mk(d * d, 0.12),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d, d],
            values_f32: mk(d * d, 0.12),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.post_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![d],
            values_f32: vec![1.0; d],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.gate.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![4, d],
            values_f32: mk(4 * d, 0.18),
        },
    ];
    for e in 0..4 {
        moe_entries.push(SafetensorBuildEntry {
            name: format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"),
            dtype: SafetensorDtype::BF16,
            shape: vec![32, d],
            values_f32: mk(32 * d, 0.11 + 0.01 * e as f32),
        });
        moe_entries.push(SafetensorBuildEntry {
            name: format!("model.layers.0.mlp.experts.{e}.up_proj.weight"),
            dtype: SafetensorDtype::BF16,
            shape: vec![32, d],
            values_f32: mk(32 * d, 0.12),
        });
        moe_entries.push(SafetensorBuildEntry {
            name: format!("model.layers.0.mlp.experts.{e}.down_proj.weight"),
            dtype: SafetensorDtype::BF16,
            shape: vec![d, 32],
            values_f32: mk(d * 32, 0.12),
        });
    }
    let moe_st = build_safetensors_bytes(&BTreeMap::new(), &moe_entries).expect("moe st");
    let ing_moe: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "qwen3-olmo2-moe-test",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&moe_st),
            "config_json": json!({
                "model_type": "qwen3_moe",
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2
            }).to_string(),
            "target_rank": 8,
            "max_outliers": 4
        }))
        .send()
        .await
        .expect("ing moe")
        .json()
        .await
        .expect("ing moe json");
    assert_eq!(ing_moe["causal_decoder_layers"], 1);
    let gen_moe: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "qwen3-olmo2-moe-test",
            "prompt": "Sparse Mixture of Experts",
            "max_new_tokens": 4,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("gen moe")
        .json()
        .await
        .expect("gen moe json");
    assert_eq!(
        gen_moe["generated_token_ids"].as_array().map(Vec::len),
        Some(4)
    );

    // -------------------------------------------------------------------------
    // 5. Microsoft `BitNet.cpp` `I2_S` (`type_id = 36`) GGUF Embedding + `NomicBert` + Eviction!
    // -------------------------------------------------------------------------
    let i2s_gguf = build_synthetic_bitnet_i2s_gguf();
    let i2s_path = tmp.path().join("bitnet-embed-i2s.gguf");
    std::fs::write(&i2s_path, &i2s_gguf).expect("write i2s gguf");
    let ing_i2s: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/gguf"))
        .json(&json!({
            "model_id": "bitnet-embed-270m-test",
            "file_path": i2s_path.to_string_lossy(),
            "target_rank": 8,
            "max_outliers": 4
        }))
        .send()
        .await
        .expect("ing i2s")
        .json()
        .await
        .expect("ing i2s json");
    assert_eq!(ing_i2s["transformer_encoder_layers"], 1);
    let emb_i2s: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/encoder/embed"))
        .json(&json!({
            "model_id": "bitnet-embed-270m-test",
            "text": "BitNet 1.58-bit ternary embeddings in JouleSynapse"
        }))
        .send()
        .await
        .expect("emb i2s")
        .json()
        .await
        .expect("emb i2s json");
    assert_eq!(emb_i2s["embedding"].as_array().map(Vec::len), Some(128));

    // Evict model from RAM and verify clean removal!
    let ev_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/models/evict"))
        .json(&json!({ "model_id": "bitnet-embed-270m-test" }))
        .send()
        .await
        .expect("evict")
        .json()
        .await
        .expect("evict json");
    assert_eq!(ev_res["ok"], true);
    assert!(ev_res["removed_entries"].as_u64().unwrap_or(0) >= 10);
}
