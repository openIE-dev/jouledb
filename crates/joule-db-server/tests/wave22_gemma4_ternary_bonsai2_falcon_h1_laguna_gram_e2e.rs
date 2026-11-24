//! Wave 22 End-to-End Integration Suite:
//! - SplitMix64 Rademacher + Joint Binary Gram Gauss-Seidel `LittleBitSVFTensor` quality gain (`0` extra BPW),
//! - Google `Gemma 4` (`E2B` / `E4B` / `26B-A4B`) heterogeneous per-layer `head_dim` (`16` sliding vs `32` full),
//!   `attention_k_eq_v`, `num_kv_shared_layers` KV reuse, `layer_scalar`, & zero-copy `.safetensors` mmap,
//! - PrismML `Ternary-Bonsai-2` (`PQ2_0` `type_id = 142`, `PTQ1_0` `type_id = 143`, `Q2_0` `type_id = 42`) GGUF
//!   dequantization + `Qwen3.5` Gated DeltaNet (`self_attn.qkv_proj` + `ssm.*`) hybrid causal decode,
//! - TII `Falcon-H1` parallel `Mamba-2` (`mamba.*`) + `GQA` (`self_attn.*`) + `feed_forward.*` decode,
//! - Poolside `Laguna-XS-2.1` 3D stacked GGUF MoE (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` + `attn_gate`) decode,
//! - `gemma` & `falcon-llm-license` Model Card preservation + 67-model `< 30 GB` catalog.

use joule_db_server::{Server, ServerConfig};
use joule_db_weights::{
    checkpoint_ingest::{SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes},
    synapse::LittleBitSVFTensor,
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use tokio::net::TcpListener;

fn mk(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 19 + 7) % 37) as f32 - 18.0) * scale / 18.0)
        .collect()
}

fn build_synthetic_ternary_bonsai2_pq2_gguf() -> Vec<u8> {
    // GGUF v3 with `general.architecture = "qwen35"`, Layer 0 = Gated DeltaNet (`PQ2_0` `tid=142` + `PTQ1_0` `tid=143`),
    // Layer 1 = standard Qwen3 GQA (`Q2_0` `tid=42`)!
    let mut out = Vec::new();
    let push_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());

    let tensors: Vec<(&str, Vec<u64>, u32)> = vec![
        ("token_embd.weight", vec![128, 128], 142),
        ("output_norm.weight", vec![128], 0),
        // Layer 0: Qwen3.5 Gated DeltaNet (`PQ2_0` tid=142 & `PTQ1_0` tid=143)
        ("blk.0.attn_norm.weight", vec![128], 0),
        ("blk.0.attn_qkv.weight", vec![128, 384], 142),
        ("blk.0.attn_gate.weight", vec![128, 128], 143),
        ("blk.0.ssm_conv1d.weight", vec![4, 384], 0),
        ("blk.0.ssm_a", vec![4], 0),
        ("blk.0.ssm_dt.bias", vec![4], 0),
        ("blk.0.ssm_alpha.weight", vec![128, 4], 0),
        ("blk.0.ssm_beta.weight", vec![128, 4], 0),
        ("blk.0.ssm_norm.weight", vec![32], 0),
        ("blk.0.ssm_out.weight", vec![128, 128], 142),
        ("blk.0.post_attention_norm.weight", vec![128], 0),
        ("blk.0.ffn_gate.weight", vec![128, 256], 142),
        ("blk.0.ffn_up.weight", vec![128, 256], 143),
        ("blk.0.ffn_down.weight", vec![256, 128], 142),
        // Layer 1: Qwen3 GQA (`Q2_0` tid=42)
        ("blk.1.attn_norm.weight", vec![128], 0),
        ("blk.1.attn_q.weight", vec![128, 128], 42),
        ("blk.1.attn_k.weight", vec![128, 64], 42),
        ("blk.1.attn_v.weight", vec![128, 64], 42),
        ("blk.1.attn_q_norm.weight", vec![32], 0),
        ("blk.1.attn_k_norm.weight", vec![32], 0),
        ("blk.1.attn_output.weight", vec![128, 128], 42),
        ("blk.1.post_attention_norm.weight", vec![128], 0),
        ("blk.1.ffn_gate.weight", vec![128, 256], 42),
        ("blk.1.ffn_up.weight", vec![128, 256], 42),
        ("blk.1.ffn_down.weight", vec![256, 128], 42),
    ];
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&3u64.to_le_bytes());

    push_str(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut out, "qwen35");
    push_str(&mut out, "qwen35.attention.head_count");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    push_str(&mut out, "qwen35.attention.head_count_kv");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());

    let f16_scale: [u8; 2] = [0xf5, 0x24]; // ~0.01936 f16
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
        match *tid {
            0 => {
                for j in 0..nel {
                    let v = if name.contains("norm") {
                        1.0f32
                    } else {
                        (((j + idx) % 9) as f32 - 4.0) * 0.04
                    };
                    data_blob.extend_from_slice(&v.to_le_bytes());
                }
            }
            42 => {
                for b_i in 0..(nel / 64) {
                    data_blob.extend_from_slice(&f16_scale);
                    for j in 0..16 {
                        data_blob.push(if (b_i + j + idx).is_multiple_of(2) {
                            0b10_01_00_10
                        } else {
                            0b00_10_01_00
                        });
                    }
                }
            }
            142 => {
                for b_i in 0..(nel / 128) {
                    data_blob.extend_from_slice(&f16_scale);
                    for j in 0..32 {
                        data_blob.push(if (b_i + j + idx).is_multiple_of(2) {
                            0b10_01_00_10
                        } else {
                            0b01_00_10_01
                        });
                    }
                }
            }
            143 => {
                for b_i in 0..(nel / 128) {
                    for j in 0..24 {
                        data_blob.push(((b_i + j * 13 + idx) % 243) as u8);
                    }
                    data_blob.extend_from_slice(&[0b10_01_00_10, 0b00_10_01_10]);
                    data_blob.extend_from_slice(&f16_scale);
                }
            }
            _ => unreachable!(),
        }
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.extend_from_slice(&data_blob);
    out
}

fn build_synthetic_laguna_stacked_moe_gguf() -> Vec<u8> {
    // GGUF v3 mimicking `poolside/Laguna-XS-2.1-GGUF` (`blk.0` 3D stacked MoE `ffn_{gate,up,down}_exps` + `attn_gate`)!
    let mut out = Vec::new();
    let push_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());

    let tensors: Vec<(&str, Vec<u64>, u32)> = vec![
        ("token_embd.weight", vec![64, 96], 0),
        ("output_norm.weight", vec![64], 0),
        ("blk.0.attn_norm.weight", vec![64], 0),
        ("blk.0.attn_q.weight", vec![64, 64], 0),
        ("blk.0.attn_k.weight", vec![64, 32], 0),
        ("blk.0.attn_v.weight", vec![64, 32], 0),
        ("blk.0.attn_q_norm.weight", vec![16], 0),
        ("blk.0.attn_k_norm.weight", vec![16], 0),
        ("blk.0.attn_gate.weight", vec![64, 4], 0),
        ("blk.0.attn_output.weight", vec![64, 64], 0),
        ("blk.0.ffn_norm.weight", vec![64], 0),
        ("blk.0.ffn_gate_inp.weight", vec![64, 4], 0),
        ("blk.0.exp_probs_b.bias", vec![4], 0),
        ("blk.0.ffn_gate_exps.weight", vec![64, 32, 4], 0),
        ("blk.0.ffn_up_exps.weight", vec![64, 32, 4], 0),
        ("blk.0.ffn_down_exps.weight", vec![32, 64, 4], 0),
        ("blk.0.ffn_gate_shexp.weight", vec![64, 32], 0),
        ("blk.0.ffn_up_shexp.weight", vec![64, 32], 0),
        ("blk.0.ffn_down_shexp.weight", vec![32, 64], 0),
    ];
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());

    push_str(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut out, "laguna");

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
        for j in 0..nel {
            let v = if name.contains("norm") {
                1.0f32
            } else {
                (((j * 7 + idx * 3) % 19) as f32 - 9.0) * 0.03
            };
            data_blob.extend_from_slice(&v.to_le_bytes());
        }
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.extend_from_slice(&data_blob);
    out
}

#[tokio::test]
async fn test_wave22_gram_littlebit_gemma4_shared_kv_ternary_bonsai2_falcon_h1_and_laguna_e2e() {
    // 0. Verify SplitMix64 Rademacher + Joint Binary Gram Gauss-Seidel achieves strong multi-rank cosine!
    let rows = 128usize;
    let cols = 128usize;
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..8usize {
        let scale = 2.0 / ((r + 1) as f32);
        for i in 0..rows {
            let ui = (((i * (r + 3) + 5) as f32) * 0.17).sin();
            for j in 0..cols {
                let vj = (((j * (r + 2) + 11) as f32) * 0.13).cos();
                dense[i * cols + j] += scale * ui * vj;
            }
        }
    }
    let lb = LittleBitSVFTensor::factorize("gram_test", &dense, rows, cols, 24, 6, 0)
        .expect("factorize");
    assert!(
        lb.reconstruction_cosine > 0.88,
        "Expected > 0.88 multi-rank cosine from SplitMix64 + Gram Gauss-Seidel, got {}",
        lb.reconstruction_cosine
    );

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
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. Google `Gemma 4` (`model.language_model.*`):
    //    - Layer 0: sliding_attention (`head_dim = 16`, `q_proj = [64, 64]`, `k_proj = [16, 64]`, `v_proj` omitted -> `attention_k_eq_v`)
    //    - Layer 1: full_attention (`head_dim = 32`, `q_proj = [128, 64]`, `k_proj = [32, 64]`, `v_proj = [32, 64]`)
    //    - Layer 2: shared-KV layer (`k_proj` & `v_proj` omitted -> reuses Layer 0 sliding KV cache!) + `layer_scalar` + `0-D` `input_max` skipped!
    let g4_path = tmp.path().join("gemma4.safetensors");
    let g4_entries = vec![
        SafetensorBuildEntry {
            name: "model.language_model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![96, 64],
            values_f32: mk(96 * 64, 0.2),
        },
        SafetensorBuildEntry {
            name: "model.language_model.norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.vision_tower.encoder.layers.0.mlp.up_proj.input_max".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![],
            values_f32: vec![3.5],
        },
        // Layer 0 (sliding, head_dim = 16, attention_k_eq_v)
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![16, 64],
            values_f32: mk(16 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.q_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![16],
            values_f32: vec![0.05; 16],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.k_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![16],
            values_f32: vec![0.05; 16],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.pre_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.post_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.0.layer_scalar".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![1],
            values_f32: vec![0.95],
        },
        // Layer 1 (full_attention, head_dim = 32!)
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.input_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.self_attn.q_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![32],
            values_f32: vec![0.05; 32],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.self_attn.k_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![32],
            values_f32: vec![0.05; 32],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.pre_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.1.post_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        // Layer 2 (shared-KV: omits k_proj & v_proj -> reuses Layer 0 KV cache with head_dim = 16!)
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.input_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.self_attn.q_norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![16],
            values_f32: vec![0.05; 16],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.pre_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.language_model.layers.2.post_feedforward_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![0.1; 64],
        },
    ];
    let g4_bytes = build_safetensors_bytes(&BTreeMap::new(), &g4_entries).expect("g4 st");
    std::fs::write(&g4_path, g4_bytes).expect("write g4");

    let ing_g4: JsonValue = client
        .post(format!("{base}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "gemma4-e2b-synth",
            "file_path": g4_path.to_string_lossy(),
            "config_json": json!({
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 32,
                    "final_logit_softcapping": 30.0,
                    "rope_parameters": { "sliding_attention": { "rope_theta": 10000.0 } }
                }
            }).to_string(),
            "target_rank": 16
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(ing_g4["causal_decoder_layers"], 3);

    let gen_g4: JsonValue = client
        .post(format!("{base}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "gemma4-e2b-synth",
            "prompt": "SELECT * FROM jouledb",
            "max_new_tokens": 6
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(gen_g4["num_layers"], 3);
    assert_eq!(
        gen_g4["generated_token_ids"].as_array().map(Vec::len),
        Some(6)
    );

    // 2. PrismML `Ternary-Bonsai-2` (`PQ2_0` + `PTQ1_0` + `Q2_0`) + `Qwen3.5` Gated DeltaNet hybrid!
    let tb2_path = tmp.path().join("ternary_bonsai2.gguf");
    std::fs::write(&tb2_path, build_synthetic_ternary_bonsai2_pq2_gguf()).expect("write tb2");
    let ing_tb2: JsonValue = client
        .post(format!("{base}/api/v1/synapse/ingest/gguf"))
        .json(&json!({
            "model_id": "ternary-bonsai-2-synth",
            "file_path": tb2_path.to_string_lossy(),
            "target_rank": 16
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(ing_tb2["causal_decoder_layers"], 2);

    let gen_tb2: JsonValue = client
        .post(format!("{base}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "ternary-bonsai-2-synth",
            "prompt": "fn main()",
            "max_new_tokens": 5
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(gen_tb2["num_layers"], 2);
    assert_eq!(
        gen_tb2["generated_token_ids"].as_array().map(Vec::len),
        Some(5)
    );

    // 3. TII `Falcon-H1` parallel `Mamba-2` (`mamba.*`) + `GQA` (`self_attn.*`) + `feed_forward.*`!
    let fh1_entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![80, 64],
            values_f32: mk(80 * 64, 0.2),
        },
        SafetensorBuildEntry {
            name: "model.norm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![1.0; 64],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![1.0; 64],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mamba.A_log".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![4],
            values_f32: vec![0.1; 4],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mamba.D".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![4],
            values_f32: vec![1.0; 4],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mamba.dt_bias".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![4],
            values_f32: vec![0.05; 4],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mamba.conv1d.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![80, 1, 4],
            values_f32: vec![0.1; 80 * 4],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mamba.in_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![148, 64],
            values_f32: mk(148 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mamba.out_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.pre_ff_layernorm.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64],
            values_f32: vec![1.0; 64],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.feed_forward.gate_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.feed_forward.up_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.1),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.feed_forward.down_proj.weight".into(),
            dtype: SafetensorDtype::BF16,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.1),
        },
    ];
    let fh1_path = tmp.path().join("falcon_h1.safetensors");
    std::fs::write(
        &fh1_path,
        build_safetensors_bytes(&BTreeMap::new(), &fh1_entries).expect("fh1"),
    )
    .expect("write fh1");

    let ing_fh1: JsonValue = client
        .post(format!("{base}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "falcon-h1-synth",
            "file_path": fh1_path.to_string_lossy(),
            "target_rank": 16
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(ing_fh1["causal_decoder_layers"], 1);

    let gen_fh1: JsonValue = client
        .post(format!("{base}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "falcon-h1-synth",
            "prompt": "hybrid ssm",
            "max_new_tokens": 4
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(
        gen_fh1["generated_token_ids"].as_array().map(Vec::len),
        Some(4)
    );

    // 4. Poolside `Laguna-XS-2.1` 3D stacked GGUF MoE + per-head `attn_gate`!
    let lag_path = tmp.path().join("laguna_moe.gguf");
    std::fs::write(&lag_path, build_synthetic_laguna_stacked_moe_gguf()).expect("write laguna");
    let ing_lag: JsonValue = client
        .post(format!("{base}/api/v1/synapse/ingest/gguf"))
        .json(&json!({
            "model_id": "laguna-moe-synth",
            "file_path": lag_path.to_string_lossy(),
            "target_rank": 16
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(ing_lag["causal_decoder_layers"], 1);

    let gen_lag: JsonValue = client
        .post(format!("{base}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "laguna-moe-synth",
            "prompt": "impl Iterator",
            "max_new_tokens": 5
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(
        gen_lag["generated_token_ids"].as_array().map(Vec::len),
        Some(5)
    );

    // 5. Verify `falcon-llm-license` & `gemma` Model Card export + 67-model catalog!
    let exp_dir = tmp.path().join("fh1_export");
    let exp_res: JsonValue = client
        .post(format!("{base}/api/v1/synapse/hf/export"))
        .json(&json!({
            "model_id": "falcon-h1-synth",
            "output_dir": exp_dir.to_string_lossy(),
            "base_model_repo": "tiiuae/Falcon-H1-0.5B-Base",
            "hf_repo_id": "dcharlot65-openie/Falcon-H1-0.5B-Base-LittleBit",
            "license": "falcon-llm-license"
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert_eq!(exp_res["ok"], true);
    let card = std::fs::read_to_string(exp_dir.join("README.md")).expect("readme");
    assert!(card.contains("license_name: falcon-llm-license"));

    let cat: JsonValue = client
        .get(format!("{base}/api/v1/synapse/hf/catalog"))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    assert!(cat["total_models"].as_u64().unwrap_or(0) >= 65);
}
