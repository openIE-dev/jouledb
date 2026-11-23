//! Wave 19 End-to-End Integration Suite:
//! 1. **`JouleKube` — Native Kubernetes Control Plane, `kubectl` Wire API & CNCF AI-Conformance Reconciler**:
//!    - Verifies `GET /version`, `GET /apis`, multi-document `---` YAML ingestion (`POST /api/v1/kube/apply`)
//!      for `Namespace`, `Pod`, `SandboxWarmPool` (`0.0W` standby), `SandboxClaim` (`ThermoCap` macaroon),
//!      `LeaderWorkerSet`, `InferencePool` (`llm-d` EPP), `ClusterQueue` (`Kueue`), and `kagent.dev/v1alpha1` `Agent`.
//!    - Verifies plain SQL querying of `_jouledb_kube_objects` (proving `etcd` disintermediation!),
//!      `kubectl` CRUD + `?watch=true` NDJSON stream, `InferencePool` routing, and `Agent` invocation
//!      where `SandboxClaim` `allow-sql-tables` permits `sensor_logs` and halts `payroll_secrets` (`403`)!
//! 2. **Full `Whisper` Two-Conv + Cross-Attention Seq2Seq Decoder & `SigLIP` Zero-Shot Vision-Text Alignment**
//! 3. **Continuous In-Flight Multi-Request Batching (`generate_batch`) & Hugging Face `.jsyn` Export/Pull**

use base64::Engine;
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::{
    checkpoint_ingest::{SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes},
    multimodal_encoder::{build_pcm16_wav_bytes, build_ppm_p6_bytes},
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use std::f32::consts::PI;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_wave19_joulekube_whisper_seq2seq_continuous_batch_and_hf_jsyn_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = ServerConfig {
        db_path: tmp.path().to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    let server = Server::new(cfg).expect("boot server");
    let app = server.router();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let srv_task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let base_url = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();
    let mk =
        |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * s).sin() * 0.42).collect() };

    // Seed an allowed SQL table (`sensor_logs`) and a forbidden table (`payroll_secrets`)
    for sql in [
        "CREATE TABLE sensor_logs (id INTEGER PRIMARY KEY, watts INTEGER)",
        "INSERT INTO sensor_logs (id, watts) VALUES (1, 18)",
        "CREATE TABLE payroll_secrets (id INTEGER PRIMARY KEY, salary INTEGER)",
    ] {
        let r = http
            .post(format!("{base_url}/api/v1/query"))
            .json(&json!({ "sql": sql }))
            .send()
            .await
            .expect("sql setup");
        assert!(r.status().is_success());
    }

    // -------------------------------------------------------------------------
    // 1. Ingest a Causal Decoder (`tiny-kube-llm`) + Continuous Batching + HF .jsyn Export/Pull
    // -------------------------------------------------------------------------
    let dec_entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.09),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.13),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 32],
            values_f32: mk(16 * 32, 0.17),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 32],
            values_f32: mk(16 * 32, 0.19),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.23),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.29),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.31),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.37),
        },
        SafetensorBuildEntry {
            name: "lm_head.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.41),
        },
    ];
    let dec_st = build_safetensors_bytes(&BTreeMap::new(), &dec_entries).expect("dec st");
    let ing_r: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "tiny-kube-llm",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&dec_st),
            "target_rank": 16,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("ingest dec")
        .json()
        .await
        .expect("json");
    assert_eq!(ing_r["causal_decoder_layers"], 1);

    // Continuous In-Flight Multi-Request Batching (4 concurrent prompts!)
    let batch_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate_batch"))
        .json(&json!({
            "model_id": "tiny-kube-llm",
            "prompts": [
                "SELECT * FROM nodes",
                "SELECT * FROM pods",
                "SELECT * FROM sandboxes",
                "SELECT * FROM queues"
            ],
            "max_new_tokens": 6,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("batch gen")
        .json()
        .await
        .expect("batch json");
    assert_eq!(batch_res["batch_size"], 4);
    assert_eq!(batch_res["total_generated_tokens"], 24);
    assert!(
        batch_res["aggregate_tokens_per_sec"]
            .as_f64()
            .unwrap_or(0.0)
            > 0.0
    );

    // Hugging Face `.jsyn` Export -> Pull Roundtrip
    let hf_dir = tmp.path().join("hf-export-bundle");
    let exp_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/hf/export"))
        .json(&json!({
            "model_id": "tiny-kube-llm",
            "output_dir": hf_dir.to_string_lossy(),
            "base_model_repo": "openIE/tiny-kube-llm",
            "hf_repo_id": "dcharlot65-openie/tiny-kube-llm-LittleBit"
        }))
        .send()
        .await
        .expect("hf export")
        .json()
        .await
        .expect("hf export json");
    assert_eq!(exp_res["ok"], true);
    assert!(hf_dir.join("model.jsyn").exists());
    assert!(hf_dir.join("README.md").exists());

    let pull_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/hf/pull"))
        .json(&json!({
            "model_id": "reloaded-kube-llm",
            "local_dir": hf_dir.to_string_lossy()
        }))
        .send()
        .await
        .expect("hf pull")
        .json()
        .await
        .expect("hf pull json");
    assert_eq!(pull_res["ok"], true);
    assert_eq!(pull_res["imported_2d_tensors"], 9);

    // -------------------------------------------------------------------------
    // 2. `JouleKube` — `kubectl` Handshake, Multi-Doc YAML Apply, SQL `_jouledb_kube_objects` & ThermoCap Agent
    // -------------------------------------------------------------------------
    let ver_res: JsonValue = http
        .get(format!("{base_url}/version"))
        .send()
        .await
        .expect("version")
        .json()
        .await
        .expect("ver json");
    assert!(
        ver_res["gitVersion"]
            .as_str()
            .unwrap_or("")
            .contains("jouledb")
    );

    let apis_res: JsonValue = http
        .get(format!("{base_url}/apis"))
        .send()
        .await
        .expect("apis")
        .json()
        .await
        .expect("apis json");
    assert_eq!(apis_res["kind"], "APIGroupList");

    let k8s_yaml = r#"
apiVersion: v1
kind: Namespace
metadata:
  name: ai-ops
---
apiVersion: v1
kind: Pod
metadata:
  name: telemetry-collector
  namespace: ai-ops
  labels:
    role: collector
spec:
  replicas: 1
---
apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxWarmPool
metadata:
  name: zero-watt-pool
  namespace: ai-ops
spec:
  replicas: 16
---
apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxClaim
metadata:
  name: claim-ops
  namespace: ai-ops
  annotations:
    jouledb.io/energy-budget-nj: "250J"
    jouledb.io/allow-sql-tables: "sensor_logs"
spec:
  warmPoolRef:
    name: zero-watt-pool
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: fabric-main-queue
spec:
  energyQuotaNj: 900000000000
---
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: distributed-swarm
  namespace: ai-ops
spec:
  replicas: 1
---
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: causal-pool
  namespace: ai-ops
spec:
  modelName: reloaded-kube-llm
---
apiVersion: kagent.dev/v1alpha1
kind: Agent
metadata:
  name: ops-guardian
  namespace: ai-ops
spec:
  modelConfig: reloaded-kube-llm
  sandboxClaimRef: claim-ops
"#;

    let apply_res: JsonValue = http
        .post(format!("{base_url}/api/v1/kube/apply"))
        .header("content-type", "application/yaml")
        .body(k8s_yaml)
        .send()
        .await
        .expect("kube apply")
        .json()
        .await
        .expect("kube apply json");
    assert_eq!(apply_res["ok"], true);
    assert_eq!(apply_res["applied_count"], 8);

    // Query Kubernetes cluster state via plain SQL (`_jouledb_kube_objects`)!
    let sql_kube: JsonValue = http
        .post(format!("{base_url}/api/v1/query"))
        .json(&json!({
            "sql": "SELECT kind, name, substrate_primitive FROM _jouledb_kube_objects WHERE namespace = 'ai-ops'"
        }))
        .send()
        .await
        .expect("sql kube")
        .json()
        .await
        .expect("sql kube json");
    let rows = sql_kube["rows"].as_array().expect("rows");
    assert!(
        rows.len() >= 6,
        "expected >= 6 SQL rows in _jouledb_kube_objects, got {}",
        rows.len()
    );

    // `kubectl get pods -n ai-ops -l role=collector` wire check + `?watch=true` NDJSON check
    let pod_list: JsonValue = http
        .get(format!(
            "{base_url}/api/v1/namespaces/ai-ops/pods?labelSelector=role=collector"
        ))
        .send()
        .await
        .expect("get pods")
        .json()
        .await
        .expect("pods json");
    assert_eq!(pod_list["kind"], "PodList");
    assert_eq!(pod_list["items"].as_array().map(Vec::len), Some(1));
    assert_eq!(pod_list["items"][0]["status"]["phase"], "Running");

    let watch_ndjson = http
        .get(format!(
            "{base_url}/api/v1/namespaces/ai-ops/pods?watch=true"
        ))
        .send()
        .await
        .expect("watch")
        .text()
        .await
        .expect("watch text");
    assert!(watch_ndjson.contains("\"ADDED\""));

    // Gateway API Inference Extension (`InferencePool` EPP route -> sub-1-bit `reloaded-kube-llm`!)
    let route_res: JsonValue = http
        .post(format!(
            "{base_url}/api/v1/kube/inference-pools/ai-ops/causal-pool/route"
        ))
        .json(&json!({ "prompt": "Check rack thermals", "max_new_tokens": 5 }))
        .send()
        .await
        .expect("epp route")
        .json()
        .await
        .expect("epp json");
    assert_eq!(route_res["ok"], true);
    assert_eq!(route_res["model"], "reloaded-kube-llm");

    // Declarative `Kagent` invocation confined by `SandboxClaim` (`claim-ops` allows ONLY `sensor_logs`!):
    // 2a. Permitted SQL table (`sensor_logs`) -> `stop_reason == "completed"`, 1 SQL step executed
    let ag_ok: JsonValue = http
        .post(format!(
            "{base_url}/api/v1/kube/agents/ai-ops/ops-guardian/invoke"
        ))
        .json(&json!({
            "goal": "Run SQL: SELECT id, watts FROM sensor_logs",
            "max_turns": 1
        }))
        .send()
        .await
        .expect("agent ok")
        .json()
        .await
        .expect("agent ok json");
    assert_eq!(ag_ok["stop_reason"], "completed");
    assert_eq!(ag_ok["steps"].as_array().map(Vec::len), Some(1));

    // 2b. Unauthorized SQL table (`payroll_secrets`) -> cryptographically halted at Turn 0 by `SandboxClaim` ThermoCap!
    let ag_deny: JsonValue = http
        .post(format!(
            "{base_url}/api/v1/kube/agents/ai-ops/ops-guardian/invoke"
        ))
        .json(&json!({
            "goal": "Run SQL: SELECT id, salary FROM payroll_secrets",
            "max_turns": 1
        }))
        .send()
        .await
        .expect("agent deny")
        .json()
        .await
        .expect("agent deny json");
    assert_eq!(ag_deny["stop_reason"], "halted_by_ward_or_guardrail");
    assert_eq!(ag_deny["steps"].as_array().map(Vec::len), Some(0));

    // -------------------------------------------------------------------------
    // 3. Full `Whisper` Two-Conv + Cross-Attention Seq2Seq Decoder & `SigLIP` Zero-Shot Vision
    // -------------------------------------------------------------------------
    let d_w = 64usize;
    let whisper_entries = vec![
        SafetensorBuildEntry {
            name: "model.encoder.conv1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, 80, 3],
            values_f32: mk(d_w * 240, 0.11),
        },
        SafetensorBuildEntry {
            name: "model.encoder.conv2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w, 3],
            values_f32: mk(d_w * d_w * 3, 0.13),
        },
        SafetensorBuildEntry {
            name: "model.encoder.embed_positions.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, d_w],
            values_f32: mk(64 * d_w, 0.15),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.17),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.19),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.21),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.23),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.fc1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w * 2, d_w],
            values_f32: mk(d_w * 2 * d_w, 0.25),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.fc2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w * 2],
            values_f32: mk(d_w * d_w * 2, 0.27),
        },
        SafetensorBuildEntry {
            name: "model.decoder.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, d_w],
            values_f32: mk(32 * d_w, 0.29),
        },
        SafetensorBuildEntry {
            name: "model.decoder.embed_positions.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, d_w],
            values_f32: mk(64 * d_w, 0.31),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.33),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.35),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.37),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.39),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.encoder_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.41),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.encoder_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.43),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.encoder_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.45),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.encoder_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w],
            values_f32: mk(d_w * d_w, 0.47),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.fc1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w * 2, d_w],
            values_f32: mk(d_w * 2 * d_w, 0.49),
        },
        SafetensorBuildEntry {
            name: "model.decoder.layers.0.fc2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_w, d_w * 2],
            values_f32: mk(d_w * d_w * 2, 0.51),
        },
    ];
    let w_st = build_safetensors_bytes(&BTreeMap::new(), &whisper_entries).expect("whisper st");
    http.post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "whisper-seq2seq-lb",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&w_st),
            "target_rank": 16,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("ingest whisper");

    let samples: Vec<f32> = (0..4_800)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / 16_000.0).sin() * 0.5)
        .collect();
    let wav = build_pcm16_wav_bytes(16_000, &samples);
    let asr_res: JsonValue = http
        .post(format!("{base_url}/v1/audio/transcriptions"))
        .json(&json!({
            "model": "whisper-seq2seq-lb",
            "audio_base64": base64::engine::general_purpose::STANDARD.encode(&wav),
        }))
        .send()
        .await
        .expect("asr")
        .json()
        .await
        .expect("asr json");
    assert_eq!(asr_res["model"], "whisper-seq2seq-lb");
    assert!(!asr_res["text"].as_str().unwrap_or("").is_empty());

    // SigLIP Zero-Shot Vision-Text Alignment
    let sig_entries = vec![
        SafetensorBuildEntry {
            name: "vision_model.embeddings.patch_embedding.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 3, 4, 4],
            values_f32: mk(64 * 48, 0.12),
        },
        SafetensorBuildEntry {
            name: "vision_model.embeddings.position_embedding.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 64],
            values_f32: mk(16 * 64, 0.16),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.18),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.20),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.22),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.24),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.mlp.fc1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.26),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.mlp.fc2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.28),
        },
        SafetensorBuildEntry {
            name: "text_model.embeddings.token_embedding.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.30),
        },
        SafetensorBuildEntry {
            name: "text_model.encoder.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.32),
        },
        SafetensorBuildEntry {
            name: "text_model.encoder.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.34),
        },
        SafetensorBuildEntry {
            name: "text_model.encoder.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.36),
        },
        SafetensorBuildEntry {
            name: "text_model.encoder.layers.0.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.38),
        },
        SafetensorBuildEntry {
            name: "text_model.encoder.layers.0.mlp.fc1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.40),
        },
        SafetensorBuildEntry {
            name: "text_model.encoder.layers.0.mlp.fc2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.42),
        },
        SafetensorBuildEntry {
            name: "logit_scale".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![1],
            values_f32: vec![2.3],
        },
    ];
    let sig_st = build_safetensors_bytes(&BTreeMap::new(), &sig_entries).expect("sig st");
    http.post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "siglip-zs-lb",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&sig_st),
            "target_rank": 16,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("ingest siglip");

    let rgb: Vec<u8> = (0..16 * 16)
        .flat_map(|i| [(i % 255) as u8, 80, 180])
        .collect();
    let ppm = build_ppm_p6_bytes(16, 16, &rgb).expect("ppm");
    let zs_res: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/vision/zero_shot"))
        .json(&json!({
            "model_id": "siglip-zs-lb",
            "image_base64": base64::engine::general_purpose::STANDARD.encode(&ppm),
            "candidate_labels": ["a thermal server rack", "an ocean wave", "a circuit board"]
        }))
        .send()
        .await
        .expect("zs")
        .json()
        .await
        .expect("zs json");
    assert_eq!(zs_res["candidates"].as_array().map(Vec::len), Some(3));
    assert!(!zs_res["best_label"].as_str().unwrap_or("").is_empty());

    srv_task.abort();
}
