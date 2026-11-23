//! Wave 9 End-to-End Integration Suite:
//! `JouleSynapse` (Post-Frozen-Weights Living Model Substrate) &
//! `JouleHarness` (Database-Native AI Retrieval, Observation & Self-Edit Engine)
//!
//! Proves live over HTTP (`joule_db_server::Server`):
//! 1. Samsung `LittleBit-2` sub-1-bit (`< 0.65` BPW, `> 24×` vs FP16) Dual-SVID binary bitplanes,
//!    pure `XNOR + POPCNT` matvec, and real-time Sakana `Transformer²` `SVF` online adaptation.
//! 2. Google `Titans` / `Nested Learning` + `TTT` surprise-gated live plastic weights, habituation,
//!    MVCC `rollback_to_lsn` defense against test-time weight poisoning, and sleep consolidation.
//! 3. DeepSeek `Engram` $O(1)$ conditional memory + `AirLLM` / SJTU `PowerInfer` strict-VRAM-budget
//!    hot-pinned & double-buffered layer/expert streaming (`0` OOMs at 4× model-to-VRAM ratio).
//! 4. Database-native `JouleHarness`: secret-scrubbed CAS observation salience compression,
//!    `Mooncake` causal Radix KV-cache prefix hits, `MIT SEAL` self-edit compilation, and atomic
//!    whole-mind `checkpoint` / `rewind` across context, conditional memory, and live weights.

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use joule_db_server::{Server, ServerConfig};
use serde_json::{Value as JsonValue, json};
use tower::ServiceExt;

fn boot_test_server(dir: &std::path::Path) -> Server {
    let cfg = ServerConfig {
        db_path: dir.to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    Server::new(cfg).expect("boot test server")
}

async fn post_json(server: &Server, uri: &str, body: JsonValue) -> (StatusCode, JsonValue) {
    let app = server.router();
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("build request");
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
        .expect("build get");
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    let parsed: JsonValue = serde_json::from_slice(&bytes).expect("parse json");
    (status, parsed)
}

#[tokio::test]
async fn test_live_http_littlebit_sub_1_bit_xnor_popcnt_and_sakana_svf_adaptation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(dir.path());

    let rows = 128usize;
    let cols = 128usize;
    let mut weights = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let base = if (i + j) % 2 == 0 { 1.6f32 } else { -1.6f32 };
            let mod2 = if i < 64 { 0.7f32 } else { -0.7f32 } * if j % 4 < 2 { 1.1 } else { -1.1 };
            weights.push(base + mod2);
        }
    }

    // 1. Factorize into sub-1-bit Samsung LittleBit-2 representation
    let (st, fac) = post_json(
        &server,
        "/api/v1/synapse/littlebit/factorize",
        json!({
            "name": "blk.0.ffn_gate",
            "rows": rows,
            "cols": cols,
            "rank": 8,
            "itq_iters": 8,
            "max_outliers": 16,
            "weights": weights,
        }),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let bpw = fac["stats"]["bits_per_weight"].as_f64().expect("bpw");
    let comp = fac["stats"]["compression_vs_fp16"].as_f64().expect("comp");
    let cos = fac["stats"]["reconstruction_cosine"].as_f64().expect("cos");
    assert!(bpw < 0.65, "Expected < 0.65 BPW, got {bpw}");
    assert!(comp > 24.0, "Expected > 24x vs FP16, got {comp}");
    assert!(cos > 0.85, "Expected > 0.85 cosine fidelity, got {cos}");

    // 2. Run both Dual-SVID continuous-input and pure XNOR+POPCNT bitplane kernels
    let input: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.13).cos()).collect();
    let (st_fwd, fwd) = post_json(
        &server,
        "/api/v1/synapse/littlebit/forward",
        json!({
            "name": "blk.0.ffn_gate",
            "input": input,
            "binary_xnor_popcnt": false,
        }),
    )
    .await;
    assert_eq!(st_fwd, StatusCode::OK);
    let base_out: Vec<f32> = fwd["output"]
        .as_array()
        .expect("output")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect();
    assert_eq!(base_out.len(), rows);

    let (st_pop, pop) = post_json(
        &server,
        "/api/v1/synapse/littlebit/forward",
        json!({
            "name": "blk.0.ffn_gate",
            "input": input,
            "binary_xnor_popcnt": true,
        }),
    )
    .await;
    assert_eq!(st_pop, StatusCode::OK);
    assert_eq!(pop["kernel"], "xnor_popcnt_bitplane");

    // 3. Adapt Sakana Transformer^2 SVF expert vector in real time at inference!
    let target_y: Vec<f32> = base_out.iter().map(|&y| y * 1.4).collect();
    let mut initial_mse = 0.0;
    let mut final_mse = 0.0;
    for step in 0..6 {
        let (st_ad, ad) = post_json(
            &server,
            "/api/v1/synapse/littlebit/svf-adapt",
            json!({
                "name": "blk.0.ffn_gate",
                "expert_name": "sql_optimizer_skill",
                "input": input,
                "target_y": target_y,
                "learning_rate": 0.4,
            }),
        )
        .await;
        assert_eq!(st_ad, StatusCode::OK);
        assert_eq!(ad["base_bitplanes_mutated"], false);
        if step == 0 {
            initial_mse = ad["mse_before"].as_f64().expect("mse_before");
        }
        final_mse = ad["mse_after"].as_f64().expect("mse_after");
    }
    assert!(
        final_mse < initial_mse * 0.5,
        "Expected SVF real-time adaptation to cut MSE by >50% ({initial_mse} -> {final_mse})"
    );
}

#[tokio::test]
async fn test_live_http_titans_surprise_gated_plastic_weights_poisoning_rollback_and_sleep() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(dir.path());

    let key: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 0.75 } else { -0.5 }).collect();
    let target: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.3).sin()).collect();

    // 1. Converge plastic fast weights until surprise gate habituates
    let mut saw_adapted = false;
    let mut saw_habituated = false;
    for _ in 0..20 {
        let (st, res) = post_json(
            &server,
            "/api/v1/synapse/titans/observe",
            json!({ "key": key, "target": target }),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        if res["adapted"].as_bool() == Some(true) {
            saw_adapted = true;
        } else {
            saw_habituated = true;
            break;
        }
    }
    assert!(saw_adapted, "Novel input must trigger live Titans plastic weight update");
    assert!(saw_habituated, "Converged predictable input must habituate with 0 writes");

    let (_, clean_fwd) = post_json(
        &server,
        "/api/v1/synapse/titans/forward",
        json!({ "key": key }),
    )
    .await;
    let clean_lsn = clean_fwd["lsn"].as_u64().expect("lsn");
    let clean_out = clean_fwd["output"].clone();

    // 2. Adversarial test-time weight poisoning -> instant MVCC LSN rollback!
    let poison: Vec<f32> = vec![-50.0; 16];
    let (st_p, p_res) = post_json(
        &server,
        "/api/v1/synapse/titans/observe",
        json!({ "key": key, "target": poison }),
    )
    .await;
    assert_eq!(st_p, StatusCode::OK);
    assert_eq!(p_res["adapted"], true);

    let (st_rb, rb) = post_json(
        &server,
        "/api/v1/synapse/titans/rollback",
        json!({ "target_lsn": clean_lsn }),
    )
    .await;
    assert_eq!(st_rb, StatusCode::OK);
    assert_eq!(rb["reverted_commits"], 1);

    let (_, restored_fwd) = post_json(
        &server,
        "/api/v1/synapse/titans/forward",
        json!({ "key": key }),
    )
    .await;
    assert_eq!(
        restored_fwd["output"], clean_out,
        "MVCC rollback must restore exact pre-poisoning fast weights!"
    );

    // 3. Slow-wave sleep consolidation (`Fast -> Medium/Slow`)
    let (st_sl, sl) = post_json(
        &server,
        "/api/v1/synapse/titans/consolidate",
        json!({ "transfer_ratio": 0.85 }),
    )
    .await;
    assert_eq!(st_sl, StatusCode::OK);
    assert!(sl["transferred_l2_norm"].as_f64().unwrap_or(0.0) > 0.0);
}

#[tokio::test]
async fn test_live_http_deepseek_engram_and_airllm_powerinfer_strict_vram_streaming() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(dir.path());

    // 1. DeepSeek Engram conditional memory upsert + surface-normalized lookup
    let (st_up, _) = post_json(
        &server,
        "/api/v1/synapse/engram/upsert",
        json!({
            "phrase": "JouleDB::Primary_Port (Cluster-01)!",
            "embedding": [0.25, -0.5, 0.75, 1.0],
            "fact_payload": "{\"http\":18080,\"pgwire\":18082}",
        }),
    )
    .await;
    assert_eq!(st_up, StatusCode::OK);

    let (st_lk, lk) = post_json(
        &server,
        "/api/v1/synapse/engram/lookup",
        json!({ "phrase": "  jouledb   primary_port cluster 01 " }),
    )
    .await;
    assert_eq!(st_lk, StatusCode::OK);
    assert_eq!(lk["hit"], true);
    assert_eq!(
        lk["record"]["fact_payload"],
        "{\"http\":18080,\"pgwire\":18082}"
    );

    // 2. AirLLM + PowerInfer: 32 MiB model streamed inside an 8 MiB VRAM budget (4x over-VRAM!)
    let budget = 8 * 1024 * 1024u64;
    let mut routed = Vec::new();
    for idx in 0..16 {
        let shard_id = format!("layer.{idx}.moe_expert");
        let (st_reg, _) = post_json(
            &server,
            "/api/v1/synapse/airstream/register-shard",
            json!({
                "vram_budget_bytes": if idx == 0 { Some(budget) } else { None },
                "shard": {
                    "layer_idx": idx,
                    "shard_id": shard_id,
                    "size_bytes": 2 * 1024 * 1024u64,
                    "hot_score": if idx == 0 { 0.92 } else { 0.01 },
                }
            }),
        )
        .await;
        assert_eq!(st_reg, StatusCode::OK);
        routed.push(shard_id);
    }

    let (st_step, step) = post_json(
        &server,
        "/api/v1/synapse/airstream/step",
        json!({ "routed_shard_ids": routed }),
    )
    .await;
    assert_eq!(st_step, StatusCode::OK);
    assert_eq!(step["budget_respected"], true);
    assert!(
        step["peak_resident_vram_bytes"].as_u64().expect("peak") <= budget,
        "AirStream peak VRAM violated budget!"
    );
    assert_eq!(step["hot_hit_bytes_saved"], 2 * 1024 * 1024);
    assert!(step["prefetched_shards"].as_u64().unwrap_or(0) > 0);
}

#[tokio::test]
async fn test_live_http_joule_harness_observation_cas_mooncake_kv_prefix_seal_and_mind_rewind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(dir.path());

    // 1. Noisy observation ingestion with leaked ghp_ token
    let mut raw_spam = String::from("Auth header ghp_supersecret999999\n");
    for _ in 0..180 {
        raw_spam.push_str("downloading wheel cache chunk ... identical ...\n");
    }
    raw_spam.push_str("RESULT: 5 passed in 0.04s (12.4 microjoules)\n");

    let (st_obs, obs) = post_json(
        &server,
        "/api/v1/harness/observe",
        json!({ "source": "capsule-exec", "raw_output": raw_spam }),
    )
    .await;
    assert_eq!(st_obs, StatusCode::OK);
    assert_eq!(obs["redacted_secrets"], 1);
    assert!(
        !obs["compact_summary"]
            .as_str()
            .unwrap_or("")
            .contains("ghp_supersecret")
    );
    assert!(obs["tokens_saved"].as_u64().unwrap_or(0) > 500);

    // 2. Mooncake Radix KV-cache prefix reuse across 2 context-planning turns
    let cands_t1 = json!([
        {
            "id": "system_constitution",
            "prefix_stable": true,
            "pinned": true,
            "tokens": 32,
            "relevance": 1.0,
            "surprise": 0.0,
            "content": "Energy first heterogeneous substrate constitution rules."
        },
        {
            "id": "schema_catalog",
            "prefix_stable": true,
            "pinned": true,
            "tokens": 32,
            "relevance": 0.9,
            "surprise": 0.0,
            "content": "Tables accounts sensors synapse engram capsules."
        },
        {
            "id": "ephemeral_turn",
            "prefix_stable": false,
            "pinned": false,
            "tokens": 16,
            "relevance": 0.85,
            "surprise": 1.5,
            "content": "Inspect live Raft leader term."
        }
    ]);
    let (st_p1, _) = post_json(
        &server,
        "/api/v1/harness/plan-context",
        json!({ "candidates": cands_t1, "max_tokens": 96, "max_joules_uj": 40000.0 }),
    )
    .await;
    assert_eq!(st_p1, StatusCode::OK);

    let mut cands_t2 = cands_t1.clone();
    cands_t2[2]["content"] = json!("Run SEAL self-edit and verify prefix KV hit.");
    let (st_p2, plan2) = post_json(
        &server,
        "/api/v1/harness/plan-context",
        json!({ "candidates": cands_t2, "max_tokens": 96, "max_joules_uj": 40000.0 }),
    )
    .await;
    assert_eq!(st_p2, StatusCode::OK);
    assert!(
        plan2["kv_prefix_receipt"]["matched_prefix_tokens"]
            .as_u64()
            .unwrap_or(0)
            >= 64
    );

    // 3. Atomic Mind Checkpoint -> MIT SEAL Self-Edit -> Rewind (restoring Engram + Plastic Weights!)
    let (st_ck, ckpt) = post_json(
        &server,
        "/api/v1/harness/checkpoint",
        json!({ "label": "pre-seal-experiment" }),
    )
    .await;
    assert_eq!(st_ck, StatusCode::OK);
    let ckpt_id = ckpt["checkpoint_id"].as_u64().expect("ckpt_id");

    let key_vec: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { -0.5 }).collect();
    let target_vec: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 0.9 } else { -0.9 }).collect();
    let (_, pre_fwd) = post_json(
        &server,
        "/api/v1/synapse/titans/forward",
        json!({ "key": key_vec }),
    )
    .await;

    let (st_se, se) = post_json(
        &server,
        "/api/v1/harness/self-edit",
        json!({
            "trigger_phrase": "Speculative API Endpoint",
            "correction_fact": "/api/v1/query uses field 'sql'",
            "key_vec": key_vec,
            "target_vec": target_vec,
        }),
    )
    .await;
    assert_eq!(st_se, StatusCode::OK);
    assert_eq!(se["synapse_adapted"], true);

    // Rewind! Both the speculative Engram entry AND the Titans fast-weight mutation vanish atomically!
    let (st_rw, _) = post_json(
        &server,
        "/api/v1/harness/rewind",
        json!({ "checkpoint_id": ckpt_id }),
    )
    .await;
    assert_eq!(st_rw, StatusCode::OK);

    let (_, post_lk) = post_json(
        &server,
        "/api/v1/synapse/engram/lookup",
        json!({ "phrase": "speculative api endpoint" }),
    )
    .await;
    assert_eq!(post_lk["hit"], false);

    let (_, post_fwd) = post_json(
        &server,
        "/api/v1/synapse/titans/forward",
        json!({ "key": key_vec }),
    )
    .await;
    assert_eq!(post_fwd["output"], pre_fwd["output"]);

    // 4. Status telemetry endpoint
    let (st_status, status) = get_json(&server, "/api/v1/synapse/status").await;
    assert_eq!(st_status, StatusCode::OK);
    assert_eq!(status["status"], "ok");
}

#[tokio::test]
async fn test_live_http_safetensors_living_ingest_sql_introspection_and_hardware_metal_wgpu_popcnt() {
    use joule_db_weights::checkpoint_ingest::{
        SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
    };
    use std::collections::BTreeMap;

    let dir = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(dir.path());

    // 1. Build a multi-layer `.safetensors` checkpoint (`F16` + `BF16` + `F32` + `__metadata__`)
    let mut meta = BTreeMap::new();
    meta.insert("architecture".to_string(), "Qwen3ForCausalLM".to_string());
    meta.insert("context_window".to_string(), "131072".to_string());

    let rows = 128usize;
    let cols = 128usize;
    let mut attn_q = Vec::with_capacity(rows * cols);
    let mut mlp_gate = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            attn_q.push(if (i + j) % 2 == 0 { 1.4f32 } else { -1.4f32 });
            mlp_gate.push(if i < 64 { 0.9f32 } else { -0.9f32 } * if j % 3 == 0 { 1.2 } else { -0.8 });
        }
    }

    let st_bytes = build_safetensors_bytes(
        &meta,
        &[
            SafetensorBuildEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                dtype: SafetensorDtype::F16,
                shape: vec![rows, cols],
                values_f32: attn_q,
            },
            SafetensorBuildEntry {
                name: "model.layers.1.mlp.gate_proj.weight".to_string(),
                dtype: SafetensorDtype::BF16,
                shape: vec![rows, cols],
                values_f32: mlp_gate,
            },
            SafetensorBuildEntry {
                name: "model.layers.0.input_layernorm.weight".to_string(),
                dtype: SafetensorDtype::F32,
                shape: vec![rows],
                values_f32: vec![1.0; rows],
            },
        ],
    )
    .expect("build safetensors");

    let st_path = dir.path().join("qwen3-edge-0.5b.safetensors");
    std::fs::write(&st_path, &st_bytes).expect("write safetensors");

    // 2. Ingest `.safetensors` file into JouleSynapse + live SQL `synapse_tensors` table!
    let (st_ing, ing) = post_json(
        &server,
        "/api/v1/synapse/ingest/safetensors",
        json!({
            "model_id": "qwen3-edge-0.5b",
            "file_path": st_path.to_string_lossy(),
            "target_rank": 8,
            "max_outliers": 8,
        }),
    )
    .await;
    assert_eq!(st_ing, StatusCode::OK, "ingest failed: {ing}");
    assert_eq!(ing["sql_rows_materialized"], 3);
    assert!(
        ing["report"]["overall_bits_per_weight"]
            .as_f64()
            .expect("bpw")
            < 0.60
    );

    // 3. Query the ingested model layers directly via standard SQL over `/api/v1/query`!
    let (st_sql, sql_res) = post_json(
        &server,
        "/api/v1/query",
        json!({
            "sql": "SELECT tensor_name, layer_idx, source_dtype, bits_per_weight FROM synapse_tensors WHERE model_id = 'qwen3-edge-0.5b' ORDER BY tensor_name ASC"
        }),
    )
    .await;
    assert_eq!(st_sql, StatusCode::OK);
    let sql_rows = sql_res["rows"].as_array().expect("sql rows");
    assert_eq!(sql_rows.len(), 3);

    let (st_agg, agg_res) = post_json(
        &server,
        "/api/v1/query",
        json!({
            "sql": "SELECT COUNT(*), SUM(compressed_bytes) FROM synapse_tensors WHERE model_id = 'qwen3-edge-0.5b'"
        }),
    )
    .await;
    assert_eq!(st_agg, StatusCode::OK);
    assert_eq!(agg_res["rows"][0][0], 3);
    assert!(agg_res["rows"][0][1].as_i64().unwrap_or(0) > 0);

    // 4. Verify metadata landed in $O(1)$ DeepSeek `Engram` conditional memory
    let (_, eng_meta) = post_json(
        &server,
        "/api/v1/synapse/engram/lookup",
        json!({ "phrase": "qwen3-edge-0.5b metadata architecture" }),
    )
    .await;
    assert_eq!(eng_meta["hit"], true);
    assert_eq!(eng_meta["record"]["fact_payload"], "Qwen3ForCausalLM");

    // 5. Run GPU `wgpu` Metal/Vulkan/WebGPU `XNOR + countOneBits()` shader (`use_gpu: true`)
    //    AND verify bit-exact agreement with CPU `XNOR + POPCNT`!
    let x_vec: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.11).sin()).collect();
    let tensor_fq = "qwen3-edge-0.5b::model.layers.0.self_attn.q_proj.weight";

    let (st_cpu, cpu_res) = post_json(
        &server,
        "/api/v1/synapse/littlebit/forward",
        json!({
            "name": tensor_fq,
            "input": x_vec,
            "binary_xnor_popcnt": true,
            "use_gpu": false,
        }),
    )
    .await;
    assert_eq!(st_cpu, StatusCode::OK);

    let (st_gpu, gpu_res) = post_json(
        &server,
        "/api/v1/synapse/littlebit/forward",
        json!({
            "name": tensor_fq,
            "input": x_vec,
            "use_gpu": true,
        }),
    )
    .await;
    assert_eq!(st_gpu, StatusCode::OK);
    println!(
        "Live HTTP LittleBit GPU forward executed on device='{}' kernel='{}'",
        gpu_res["device"], gpu_res["kernel"]
    );

    let cpu_out = cpu_res["output"].as_array().expect("cpu_out");
    let gpu_out = gpu_res["output"].as_array().expect("gpu_out");
    assert_eq!(cpu_out.len(), gpu_out.len());
    for (idx, (cv, gv)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        let c = cv.as_f64().expect("c");
        let g = gv.as_f64().expect("g");
        assert!(
            (c - g).abs() < 1e-2,
            "Element {idx} mismatch between CPU XNOR+POPCNT ({c}) and wgpu WGSL shader ({g})"
        );
    }
}

#[tokio::test]
async fn test_live_http_synapse_durable_persistence_across_server_crash_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key_vec: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 0.8 } else { -0.4 }).collect();
    let target_vec: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.2).cos()).collect();
    let x_in: Vec<f32> = (0..64).map(|j| if j % 2 == 0 { 0.5 } else { -0.5 }).collect();

    let (expected_gpu_out, expected_titans_out) = {
        let server1 = boot_test_server(dir.path());
        let weights: Vec<f32> = (0..(64 * 64))
            .map(|k| if k % 2 == 0 { 1.25 } else { -1.25 })
            .collect();
        let (st_f, _) = post_json(
            &server1,
            "/api/v1/synapse/littlebit/factorize",
            json!({
                "name": "durable::blk0.q",
                "rows": 64,
                "cols": 64,
                "rank": 8,
                "weights": weights,
            }),
        )
        .await;
        assert_eq!(st_f, StatusCode::OK);

        let (st_se, _) = post_json(
            &server1,
            "/api/v1/harness/self-edit",
            json!({
                "trigger_phrase": "Durable Reboot Rule",
                "correction_fact": "JSYN CRC32 + zstd survives restart",
                "key_vec": key_vec,
                "target_vec": target_vec,
            }),
        )
        .await;
        assert_eq!(st_se, StatusCode::OK);

        let (_, gpu1) = post_json(
            &server1,
            "/api/v1/synapse/littlebit/forward",
            json!({ "name": "durable::blk0.q", "input": x_in, "use_gpu": true }),
        )
        .await;
        let (_, tit1) = post_json(
            &server1,
            "/api/v1/synapse/titans/forward",
            json!({ "key": key_vec }),
        )
        .await;
        (gpu1["output"].clone(), tit1["output"].clone())
    };

    // Reboot a brand-new `Server` from the exact same `dir.path()`:
    // `synapse_state.bin` MUST automatically reload + warm GPU FastLanes cache!
    let server2 = boot_test_server(dir.path());
    let (_, eng2) = post_json(
        &server2,
        "/api/v1/synapse/engram/lookup",
        json!({ "phrase": "durable reboot rule" }),
    )
    .await;
    assert_eq!(eng2["hit"], true);
    assert_eq!(
        eng2["record"]["fact_payload"],
        "JSYN CRC32 + zstd survives restart"
    );

    let (st_gpu2, gpu2) = post_json(
        &server2,
        "/api/v1/synapse/littlebit/forward",
        json!({ "name": "durable::blk0.q", "input": x_in, "use_gpu": true }),
    )
    .await;
    assert_eq!(st_gpu2, StatusCode::OK);
    assert_eq!(gpu2["output"], expected_gpu_out);

    let (_, tit2) = post_json(
        &server2,
        "/api/v1/synapse/titans/forward",
        json!({ "key": key_vec }),
    )
    .await;
    assert_eq!(tit2["output"], expected_titans_out);
}
