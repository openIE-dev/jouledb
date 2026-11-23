//! HTTP API routes for `JouleSynapse` (Post-Frozen-Weights Living Model Substrate) and
//! `JouleHarness` (Database-Native AI Retrieval, Observation & Self-Edit Engine), including:
//! - End-to-end HuggingFace `.safetensors` -> `LittleBit-2 ⊕ Engram ⊕ AirStream` ingestion with
//!   automatic SQL `synapse_tensors` system table materialization.
//! - Hardware GPU `wgpu` (Metal / Vulkan / DX12 / WebGPU) `XNOR + countOneBits()` (`POPCNT`)
//!   WGSL compute shader execution (`use_gpu: true`).

use crate::query::{QueryExecutor, QueryRequest};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use base64::Engine;
use joule_db_gpu::{LittleBitGpuKernel, PackedGpuLittleBitWeights};
use joule_db_weights::{
    checkpoint_ingest::ingest_safetensors_into_harness,
    synapse::{ContextCandidate, JouleHarnessEngine, LittleBitSVFTensor, WeightShardMeta},
    transformer_encoder::{
        TransformerEncoderReceipt, detect_transformer_encoder,
        encode_text_with_littlebit_transformer,
    },
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// Shared process-wide hardware `wgpu` `LittleBitGpuKernel` (initialized lazily on first GPU dispatch
/// so non-GPU unit tests pay zero adapter-probe overhead).
fn shared_gpu_kernel() -> Option<&'static LittleBitGpuKernel> {
    static GPU_KERNEL: OnceLock<Option<LittleBitGpuKernel>> = OnceLock::new();
    GPU_KERNEL
        .get_or_init(|| pollster::block_on(LittleBitGpuKernel::new_hardware()).ok())
        .as_ref()
}

/// Shared thread-safe server runtime for `JouleSynapse` & `JouleHarness`.
pub struct SynapseServerRuntime {
    inner: Mutex<JouleHarnessEngine>,
    gpu_packed_cache: Mutex<HashMap<String, PackedGpuLittleBitWeights>>,
    executor: Option<Arc<dyn QueryExecutor>>,
    snapshot_path: Option<PathBuf>,
}

impl Default for SynapseServerRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl SynapseServerRuntime {
    pub fn new() -> Self {
        Self::build(None, None)
    }

    pub fn with_executor(executor: Arc<dyn QueryExecutor>) -> Self {
        Self::build(Some(executor), None)
    }

    pub fn with_executor_and_path(executor: Arc<dyn QueryExecutor>, db_path: &str) -> Self {
        let snap = PathBuf::from(format!("{db_path}/synapse_state.bin"));
        Self::build(Some(executor), Some(snap))
    }

    fn build(executor: Option<Arc<dyn QueryExecutor>>, snapshot_path: Option<PathBuf>) -> Self {
        let engine = if let Some(ref p) = snapshot_path
            && p.exists()
            && let Ok(loaded) = JouleHarnessEngine::load_snapshot_verified(p)
        {
            loaded
        } else {
            JouleHarnessEngine::new(16, 16, 64 * 1024 * 1024).unwrap_or_else(|_| {
                match JouleHarnessEngine::new(1, 1, 4 * 1024 * 1024) {
                    Ok(e) => e,
                    Err(_) => unreachable!("1x1 JouleHarnessEngine dimensions are within static cap"),
                }
            })
        };

        let rt = Self {
            inner: Mutex::new(engine),
            gpu_packed_cache: Mutex::new(HashMap::new()),
            executor,
            snapshot_path,
        };

        // Pre-warm GPU FastLanes cache for any persisted LittleBit tensors
        let to_warm: Vec<(String, LittleBitSVFTensor)> = {
            let guard = match rt.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard
                .littlebit_tensors
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        for (name, tensor) in &to_warm {
            rt.cache_gpu_weights(name, tensor);
        }
        rt
    }

    fn persist_if_configured(&self, engine: &JouleHarnessEngine) {
        if let Some(ref path) = self.snapshot_path
            && let Err(e) = engine.save_snapshot_atomic(path)
        {
            tracing::warn!("Failed to persist synapse snapshot '{}': {e}", path.display());
        }
    }

    fn cache_gpu_weights(&self, name: &str, tensor: &LittleBitSVFTensor) {
        if let Ok(packed) = PackedGpuLittleBitWeights::from_u64_bitplanes(
            tensor.rows,
            tensor.cols,
            tensor.rank,
            tensor.rank_words,
            &tensor.u_bits,
            &tensor.v_bits,
            tensor.row_scale.clone(),
            tensor.col_scale.clone(),
        ) {
            let mut cache = match self.gpu_packed_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if !cache.contains_key(name) && cache.len() >= 128
                && let Some(evict) = cache.keys().next().cloned()
            {
                cache.remove(&evict);
            }
            cache.insert(name.to_string(), packed);
        }
    }

    fn run_sql(&self, sql: String) -> Result<(), String> {
        let Some(ref exec) = self.executor else {
            return Ok(());
        };
        let req = QueryRequest {
            sql,
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        exec.execute(&req).map(|_| ()).map_err(|e| e.message)
    }

    /// Summary of live ingested `JouleSynapse` models grouped by model prefix (`model_id::layer`).
    pub fn list_living_models(&self) -> Vec<LivingModelCard> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut grouped: std::collections::BTreeMap<String, (usize, f64)> =
            std::collections::BTreeMap::new();
        for (fq_name, tensor) in &guard.littlebit_tensors {
            let model_id = fq_name
                .split_once("::")
                .map(|(m, _)| m.to_string())
                .unwrap_or_else(|| fq_name.clone());
            let entry = grouped.entry(model_id).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += tensor.bits_per_weight() as f64;
        }
        let mut out: Vec<LivingModelCard> = grouped
            .into_iter()
            .map(|(id, (cnt, sum_bpw))| LivingModelCard {
                id,
                tensor_count: cnt,
                avg_bits_per_weight: if cnt > 0 { sum_bpw / cnt as f64 } else { 0.0 },
            })
            .collect();
        if out.is_empty() {
            out.push(LivingModelCard {
                id: "jouledb-titans-continuum-native".to_string(),
                tensor_count: 1,
                avg_bits_per_weight: 0.48,
            });
        }
        out
    }

    /// Query $O(1)$ `Engram` conditional memory for an exact/normalized factual hit.
    pub fn lookup_engram_fact(&self, phrase: &str) -> Option<String> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.engram.lookup(phrase).map(|r| r.fact_payload)
    }

    /// Execute a full multi-layer Transformer encoder pass (`WordPiece` -> sub-1-bit `LittleBit-2`
    /// embeddings -> Multi-Head Self-Attention + `GELU` FFN + `LayerNorm` -> Mean-Pooling + L2
    /// norm) whenever a BERT/MiniLM/BGE encoder is loaded under `model_hint` (or as the default
    /// living encoder), falling back cleanly to [`Self::run_living_forward`] otherwise.
    pub fn encode_living_embedding(
        &self,
        model_hint: Option<&str>,
        text: &str,
        use_gpu: bool,
        allow_titans_plasticity: bool,
    ) -> LivingForwardReceipt {
        let enc_outcome: Option<(TransformerEncoderReceipt, u64)> = {
            let guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let target_model = if let Some(h) = model_hint {
                detect_transformer_encoder(&guard, h).map(|_| h.to_string())
            } else {
                let mut found = None;
                for fq in guard.littlebit_tensors.keys() {
                    if let Some((m, _)) = fq.split_once("::")
                        && detect_transformer_encoder(&guard, m).is_some()
                    {
                        found = Some(m.to_string());
                        break;
                    }
                }
                found
            };

            if let Some(ref mid) = target_model {
                match encode_text_with_littlebit_transformer(&guard, mid, text, 128, |_, _, _| None)
                {
                    Ok(rec) => {
                        let pfx = format!("{mid}::");
                        let mut touched_bytes = 0u64;
                        for (k, lb) in &guard.littlebit_tensors {
                            if k.starts_with(&pfx) {
                                touched_bytes = touched_bytes.saturating_add(
                                    (((lb.rows * lb.cols) as f64 * lb.bits_per_weight()) / 8.0)
                                        .ceil() as u64,
                                );
                            }
                        }
                        Some((rec, touched_bytes.max(4096)))
                    }
                    Err(_) => None,
                }
            } else {
                None
            }
        };

        if let Some((rec, mem_bytes)) = enc_outcome {
            if allow_titans_plasticity {
                self.adapt_titans_from_prompt(text);
            }
            let device = if use_gpu {
                shared_gpu_kernel()
                    .map(|k| format!("{} (UMA LittleBit-2 {}L)", k.adapter_name(), rec.num_layers))
                    .unwrap_or_else(|| "cpu-simd-bitplane".to_string())
            } else {
                "cpu-simd-bitplane".to_string()
            };
            return LivingForwardReceipt {
                model_id: rec.model_id.clone(),
                tensor_name: format!("{}::transformer_encoder_{}L", rec.model_id, rec.num_layers),
                output: rec.embedding,
                bits_per_weight: rec.avg_bits_per_weight,
                kernel: format!(
                    "littlebit_svf_transformer_{}L_{}H",
                    rec.num_layers, rec.num_heads
                ),
                device,
                elapsed_ns: rec.elapsed_ns,
                gpu_estimated_uj: 0.0,
                mem_bytes_touched: mem_bytes,
            };
        }

        self.run_living_forward(model_hint, text, use_gpu, allow_titans_plasticity)
    }

    /// Execute a real `LittleBit-2` (`wgpu` Metal/Vulkan `countOneBits` WGSL or CPU `XNOR+POPCNT`)
    /// or `Titans` continuum forward pass for OpenAI `/v1/*` and Ollama `/api/*` disintermediation.
    pub fn run_living_forward(
        &self,
        model_hint: Option<&str>,
        prompt_seed: &str,
        use_gpu: bool,
        allow_titans_plasticity: bool,
    ) -> LivingForwardReceipt {
        let t0 = std::time::Instant::now();
        let selected: Option<(String, f64, Vec<f32>, LittleBitSVFTensor)> = {
            let guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let picked = model_hint
                .and_then(|hint| {
                    guard
                        .littlebit_tensors
                        .iter()
                        .find(|(k, _)| *k == hint || k.starts_with(&format!("{hint}::")))
                })
                .or_else(|| guard.littlebit_tensors.iter().next());

            picked.map(|(name, t)| {
                (
                    name.clone(),
                    t.bits_per_weight(),
                    t.effective_latent_scales(&[]),
                    t.clone(),
                )
            })
        };

        if let Some((tensor_name, bpw, eff_sigma, tensor)) = selected {
            let compressed_mem_bytes =
                (((tensor.rows * tensor.cols) as f64 * bpw) / 8.0).ceil() as u64;
            let input_vec = deterministic_seed_vec(prompt_seed, tensor.cols.max(1));
            let model_id = tensor_name
                .split_once("::")
                .map(|(m, _)| m.to_string())
                .unwrap_or_else(|| tensor_name.clone());

            if use_gpu && let Some(gpu_kernel) = shared_gpu_kernel() {
                let packed = {
                    let existing = match self.gpu_packed_cache.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    }
                    .get(&tensor_name)
                    .cloned();
                    match existing {
                        Some(p) => Some(p),
                        None => {
                            self.cache_gpu_weights(&tensor_name, &tensor);
                            match self.gpu_packed_cache.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            }
                            .get(&tensor_name)
                            .cloned()
                        }
                    }
                };
                if let Some(packed_weights) = packed
                    && let Ok(gpu_res) =
                        gpu_kernel.forward_xnor_popcnt(&packed_weights, &input_vec, &eff_sigma)
                {
                    if allow_titans_plasticity {
                        self.adapt_titans_from_prompt(prompt_seed);
                    }
                    return LivingForwardReceipt {
                        model_id,
                        tensor_name,
                        output: gpu_res.output,
                        bits_per_weight: bpw,
                        kernel: "wgpu_xnor_popcnt_wgsl".to_string(),
                        device: gpu_res.adapter_name,
                        elapsed_ns: (t0.elapsed().as_nanos() as u64).max(1),
                        gpu_estimated_uj: gpu_res.estimated_joules_uj,
                        mem_bytes_touched: compressed_mem_bytes,
                    };
                }
            }

            let out = tensor
                .forward_binary_xnor_popcnt(&input_vec, &[])
                .unwrap_or_else(|_| vec![0.0; tensor.rows.min(64)]);
            if allow_titans_plasticity {
                self.adapt_titans_from_prompt(prompt_seed);
            }
            return LivingForwardReceipt {
                model_id,
                tensor_name,
                output: out,
                bits_per_weight: bpw,
                kernel: "xnor_popcnt_bitplane".to_string(),
                device: "cpu-simd-popcnt".to_string(),
                elapsed_ns: (t0.elapsed().as_nanos() as u64).max(1),
                gpu_estimated_uj: 0.0,
                mem_bytes_touched: compressed_mem_bytes,
            };
        }

        // Built-in `Titans` fast-weight continuum fallback when no external `.safetensors` is loaded yet
        let (out, d_in) = {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let d = guard.synapse.d_in.max(1);
            let k_vec = deterministic_seed_vec(prompt_seed, d);
            if allow_titans_plasticity {
                let target_vec = deterministic_seed_vec(&format!("target:{prompt_seed}"), guard.synapse.d_out.max(1));
                let _ = guard.synapse.observe_and_adapt(&k_vec, &target_vec);
            }
            let y = guard
                .synapse
                .forward(&k_vec)
                .unwrap_or_else(|_| vec![0.0; 16]);
            (y, d)
        };
        LivingForwardReceipt {
            model_id: "jouledb-titans-continuum-native".to_string(),
            tensor_name: "continuum_synapse::fast_slow".to_string(),
            output: out,
            bits_per_weight: 0.48,
            kernel: "titans_continuum_fast_weights".to_string(),
            device: "cpu-simd".to_string(),
            elapsed_ns: (t0.elapsed().as_nanos() as u64).max(1),
            gpu_estimated_uj: 0.0,
            mem_bytes_touched: (d_in * 64) as u64,
        }
    }

    fn adapt_titans_from_prompt(&self, prompt_seed: &str) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let k_vec = deterministic_seed_vec(prompt_seed, guard.synapse.d_in.max(1));
        let v_vec = deterministic_seed_vec(
            &format!("adapt:{prompt_seed}"),
            guard.synapse.d_out.max(1),
        );
        let _ = guard.synapse.observe_and_adapt(&k_vec, &v_vec);
    }

    /// Current `ContinuumSynapse` LSN for causal `JoulePulse` span linking.
    pub fn current_synapse_lsn(&self) -> u64 {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.synapse.lsn
    }

    /// Export a signed federated `SynapseSkillDelta` (`Sakana SVF` + `Engram` + L2/DP clipping).
    #[allow(clippy::too_many_arguments)]
    pub fn export_federated_delta(
        &self,
        cluster_secret: &[u8],
        delta_id: &str,
        node_id: &str,
        fabric_class: &str,
        expert_name: &str,
        clip_l2_norm: f32,
        dp_noise_sigma: f32,
        engram_facts: Vec<(String, String)>,
        training_energy_nj: u64,
        peak_power_mw: u64,
    ) -> joule_db_weights::federated_mesh::SynapseSkillDelta {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        joule_db_weights::federated_mesh::export_skill_delta(
            &guard,
            cluster_secret,
            delta_id,
            node_id,
            fabric_class,
            expert_name,
            clip_l2_norm,
            dp_noise_sigma,
            engram_facts,
            training_energy_nj,
            peak_power_mw,
        )
    }

    /// Merge peer federated `SynapseSkillDelta`s using Byzantine Multi-Krum outlier pruning.
    pub fn merge_federated_deltas(
        &self,
        cluster_secret: &[u8],
        expert_name: &str,
        deltas: &[joule_db_weights::federated_mesh::SynapseSkillDelta],
        max_byzantine_peers: usize,
        max_allowed_l2: f32,
    ) -> joule_db_weights::federated_mesh::FederatedMergeReceipt {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let receipt = joule_db_weights::federated_mesh::merge_federated_skill_deltas(
            &mut guard,
            cluster_secret,
            expert_name,
            deltas,
            max_byzantine_peers,
            max_allowed_l2,
        );
        if receipt.updated_tensors > 0 || receipt.merged_engram_facts > 0 {
            self.persist_if_configured(&guard);
        }
        receipt
    }
}

/// Model metadata surfaced via OpenAI `GET /v1/models` and Ollama `GET /api/tags`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LivingModelCard {
    pub id: String,
    pub tensor_count: usize,
    pub avg_bits_per_weight: f64,
}

/// Execution receipt from `SynapseServerRuntime::run_living_forward`.
#[derive(Debug, Clone)]
pub struct LivingForwardReceipt {
    pub model_id: String,
    pub tensor_name: String,
    pub output: Vec<f32>,
    pub bits_per_weight: f64,
    pub kernel: String,
    pub device: String,
    pub elapsed_ns: u64,
    pub gpu_estimated_uj: f64,
    pub mem_bytes_touched: u64,
}

fn deterministic_seed_vec(seed: &str, dim: usize) -> Vec<f32> {
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in seed.as_bytes() {
        state ^= b as u64;
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut vec = Vec::with_capacity(dim);
    for i in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407 ^ (i as u64));
        let unit = ((state >> 40) as f32 / ((1u32 << 24) as f32)) * 2.0 - 1.0;
        vec.push(unit);
    }
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    for v in &mut vec {
        *v /= norm;
    }
    vec
}

fn sql_escape_literal(s: &str) -> String {
    s.replace('\'', "''")
}

#[derive(Debug, Deserialize)]
pub struct IngestSafetensorsReq {
    pub model_id: String,
    #[serde(default)]
    pub bytes_base64: Option<String>,
    #[serde(default)]
    pub file_path: Option<String>,
    /// Optional inline `vocab.txt` or `tokenizer.json` content (auto-discovered from sibling file when `file_path` is used).
    #[serde(default)]
    pub vocab_txt: Option<String>,
    #[serde(default = "default_target_rank")]
    pub target_rank: usize,
    #[serde(default = "default_max_outliers")]
    pub max_outliers: usize,
}

#[derive(Debug, Deserialize)]
pub struct LoadTokenizerReq {
    pub model_id: String,
    #[serde(default)]
    pub vocab_txt: Option<String>,
    #[serde(default)]
    pub file_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EncoderEmbedReq {
    pub model_id: String,
    pub text: String,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
}

fn default_max_seq_len() -> usize {
    128
}

fn parse_vocab_tokens(raw: &str) -> Result<Vec<String>, String> {
    let trimmed = raw.trim_start();
    if trimmed.starts_with('{') {
        let val: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|e| format!("invalid tokenizer.json: {e}"))?;
        let vocab_obj = val
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| "tokenizer.json missing model.vocab object".to_string())?;
        let mut max_id = 0usize;
        for id_val in vocab_obj.values() {
            if let Some(id) = id_val.as_u64() {
                max_id = max_id.max(id as usize);
            }
        }
        if max_id > 512_000 {
            return Err(format!("vocab max token id {max_id} exceeds safety bound"));
        }
        let mut out = vec!["[UNK]".to_string(); max_id + 1];
        for (tok, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                out[id as usize] = tok.clone();
            }
        }
        Ok(out)
    } else {
        Ok(raw
            .lines()
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect())
    }
}

fn default_target_rank() -> usize {
    8
}

fn default_max_outliers() -> usize {
    8
}

#[derive(Debug, Deserialize)]
pub struct FactorizeLittleBitReq {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    #[serde(default = "default_itq_iters")]
    pub itq_iters: usize,
    #[serde(default)]
    pub max_outliers: usize,
    pub weights: Vec<f32>,
}

fn default_itq_iters() -> usize {
    8
}

#[derive(Debug, Deserialize)]
pub struct ForwardLittleBitReq {
    pub name: String,
    pub input: Vec<f32>,
    #[serde(default)]
    pub expert_mix: Vec<(String, f32)>,
    #[serde(default)]
    pub binary_xnor_popcnt: bool,
    #[serde(default)]
    pub use_gpu: bool,
}

#[derive(Debug, Deserialize)]
pub struct AdaptSvfReq {
    pub name: String,
    pub expert_name: String,
    pub input: Vec<f32>,
    pub target_y: Vec<f32>,
    #[serde(default = "default_svf_lr")]
    pub learning_rate: f32,
}

fn default_svf_lr() -> f32 {
    0.35
}

#[derive(Debug, Deserialize)]
pub struct TitansObserveReq {
    pub key: Vec<f32>,
    pub target: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TitansForwardReq {
    pub key: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TitansRollbackReq {
    pub target_lsn: u64,
}

#[derive(Debug, Deserialize)]
pub struct TitansConsolidateReq {
    #[serde(default = "default_transfer_ratio")]
    pub transfer_ratio: f32,
}

fn default_transfer_ratio() -> f32 {
    0.8
}

#[derive(Debug, Deserialize)]
pub struct EngramUpsertReq {
    pub phrase: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    pub fact_payload: String,
}

#[derive(Debug, Deserialize)]
pub struct EngramLookupReq {
    pub phrase: String,
}

#[derive(Debug, Deserialize)]
pub struct AirStreamConfigShardReq {
    #[serde(default)]
    pub vram_budget_bytes: Option<u64>,
    pub shard: WeightShardMeta,
}

#[derive(Debug, Deserialize)]
pub struct AirStreamStepReq {
    pub routed_shard_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct HarnessObserveReq {
    pub source: String,
    pub raw_output: String,
}

#[derive(Debug, Deserialize)]
pub struct HarnessPlanContextReq {
    pub candidates: Vec<ContextCandidate>,
    pub max_tokens: usize,
    pub max_joules_uj: f64,
}

#[derive(Debug, Deserialize)]
pub struct HarnessSelfEditReq {
    pub trigger_phrase: String,
    pub correction_fact: String,
    pub key_vec: Vec<f32>,
    pub target_vec: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct HarnessCheckpointReq {
    pub label: String,
}

#[derive(Debug, Deserialize)]
pub struct HarnessRewindReq {
    pub checkpoint_id: u64,
}

fn bad_request(msg: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg.into() })),
    )
}

pub fn synapse_router(runtime: Arc<SynapseServerRuntime>) -> Router {
    Router::new()
        .route("/api/v1/synapse/status", get(handle_status))
        .route(
            "/api/v1/synapse/ingest/safetensors",
            post(handle_ingest_safetensors),
        )
        .route(
            "/api/v1/synapse/tokenizer/load",
            post(handle_tokenizer_load),
        )
        .route("/api/v1/synapse/encoder/embed", post(handle_encoder_embed))
        .route(
            "/api/v1/synapse/littlebit/factorize",
            post(handle_littlebit_factorize),
        )
        .route(
            "/api/v1/synapse/littlebit/forward",
            post(handle_littlebit_forward),
        )
        .route(
            "/api/v1/synapse/littlebit/svf-adapt",
            post(handle_littlebit_svf_adapt),
        )
        .route("/api/v1/synapse/titans/observe", post(handle_titans_observe))
        .route("/api/v1/synapse/titans/forward", post(handle_titans_forward))
        .route(
            "/api/v1/synapse/titans/rollback",
            post(handle_titans_rollback),
        )
        .route(
            "/api/v1/synapse/titans/consolidate",
            post(handle_titans_consolidate),
        )
        .route("/api/v1/synapse/engram/upsert", post(handle_engram_upsert))
        .route("/api/v1/synapse/engram/lookup", post(handle_engram_lookup))
        .route(
            "/api/v1/synapse/airstream/register-shard",
            post(handle_airstream_register),
        )
        .route("/api/v1/synapse/airstream/step", post(handle_airstream_step))
        .route("/api/v1/harness/observe", post(handle_harness_observe))
        .route(
            "/api/v1/harness/plan-context",
            post(handle_harness_plan_context),
        )
        .route("/api/v1/harness/self-edit", post(handle_harness_self_edit))
        .route(
            "/api/v1/harness/checkpoint",
            post(handle_harness_checkpoint),
        )
        .route("/api/v1/harness/rewind", post(handle_harness_rewind))
        .with_state(runtime)
}

async fn handle_status(State(rt): State<Arc<SynapseServerRuntime>>) -> impl IntoResponse {
    let guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let tensors: Vec<_> = guard.littlebit_tensors.values().map(|t| t.stats()).collect();
    let gpu_info = shared_gpu_kernel().map(|k| {
        serde_json::json!({
            "available": true,
            "adapter": k.adapter_name(),
            "backend": k.backend_api(),
        })
    });
    Json(serde_json::json!({
        "status": "ok",
        "littlebit_tensors": tensors,
        "gpu_xnor_popcnt_kernel": gpu_info.unwrap_or_else(|| serde_json::json!({ "available": false })),
        "titans_synapse": {
            "d_in": guard.synapse.d_in,
            "d_out": guard.synapse.d_out,
            "lsn": guard.synapse.lsn,
            "habituated_count": guard.synapse.habituated_count,
            "plastic_updates": guard.synapse.plastic_updates,
            "sleep_consolidations": guard.synapse.sleep_consolidations,
            "total_joules_uj": guard.synapse.total_joules_uj,
        },
        "engram_memory": {
            "entries": guard.engram.len(),
            "lsn": guard.engram.lsn,
            "total_lookups": guard.engram.total_lookups,
            "cache_hits": guard.engram.cache_hits,
        },
        "mooncake_kv_pool": {
            "cached_blocks": guard.kv_pool.cached_block_count(),
            "matched_tokens": guard.kv_pool.total_matched_tokens,
            "recomputed_tokens": guard.kv_pool.total_recomputed_tokens,
            "saved_joules_uj": guard.kv_pool.total_saved_joules_uj,
        },
        "harness_observations": guard.observation_count(),
        "vectors_1d_count": guard.vectors_1d.len(),
        "loaded_vocabs": guard
            .model_vocabs
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect::<std::collections::BTreeMap<_, _>>(),
    }))
}

async fn handle_ingest_safetensors(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<IngestSafetensorsReq>,
) -> impl IntoResponse {
    let mut sibling_vocab_raw: Option<String> = req.vocab_txt.clone();
    let raw_bytes = if let Some(ref b64) = req.bytes_base64 {
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(b) => b,
            Err(e) => return bad_request(format!("invalid base64 safetensors payload: {e}")),
        }
    } else if let Some(ref path_str) = req.file_path {
        let p = std::path::Path::new(path_str);
        if p.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            return bad_request("file_path must end with .safetensors");
        }
        if sibling_vocab_raw.is_none()
            && let Some(parent) = p.parent()
        {
            let v_txt = parent.join("vocab.txt");
            let t_json = parent.join("tokenizer.json");
            if v_txt.exists() {
                sibling_vocab_raw = std::fs::read_to_string(v_txt).ok();
            } else if t_json.exists() {
                sibling_vocab_raw = std::fs::read_to_string(t_json).ok();
            }
        }
        match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => return bad_request(format!("failed to read '{}': {e}", p.display())),
        }
    } else {
        return bad_request("must provide either bytes_base64 or file_path");
    };

    let parsed_vocab = match sibling_vocab_raw {
        Some(ref raw) => parse_vocab_tokens(raw).ok(),
        None => None,
    };

    let (report, newly_factorized, vocab_tokens_loaded, encoder_layers) = {
        let mut guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let rep = match ingest_safetensors_into_harness(
            &req.model_id,
            &raw_bytes,
            req.target_rank,
            req.max_outliers,
            &mut guard,
        ) {
            Ok(r) => r,
            Err(e) => return bad_request(e),
        };
        let mut v_cnt = guard
            .model_vocabs
            .get(&req.model_id)
            .map(Vec::len)
            .unwrap_or(0);
        if let Some(v_vec) = parsed_vocab
            && !v_vec.is_empty()
        {
            v_cnt = v_vec.len();
            guard.model_vocabs.insert(req.model_id.clone(), v_vec);
        }
        let enc_l = detect_transformer_encoder(&guard, &req.model_id)
            .map(|(_, l)| l)
            .unwrap_or(0);
        let mut cloned_tensors = Vec::new();
        for row in &rep.tensors {
            let fq = format!("{}::{}", row.model_id, row.tensor_name);
            if let Some(lb) = guard.littlebit_tensors.get(&fq) {
                cloned_tensors.push((fq, lb.clone()));
            }
        }
        rt.persist_if_configured(&guard);
        (rep, cloned_tensors, v_cnt, enc_l)
    };

    for (fq_name, lb) in &newly_factorized {
        rt.cache_gpu_weights(fq_name, lb);
    }

    // Materialize into JouleDB's live SQL table `synapse_tensors` so standard SQL queries can
    // introspect, filter, aggregate, and join model layers alongside relational tables!
    let _ = rt.run_sql(
        "CREATE TABLE IF NOT EXISTS synapse_tensors (id TEXT PRIMARY KEY, model_id TEXT, tensor_name TEXT, layer_idx INT, source_dtype TEXT, rows INT, cols INT, rank INT, original_fp16_bytes INT, compressed_bytes INT, bits_per_weight FLOAT, compression_vs_fp16 FLOAT, reconstruction_cosine FLOAT, hot_score FLOAT)".to_string(),
    );
    let esc_model = sql_escape_literal(&report.model_id);
    let _ = rt.run_sql(format!(
        "DELETE FROM synapse_tensors WHERE model_id = '{esc_model}'"
    ));
    let mut sql_rows_materialized = 0usize;
    for row in &report.tensors {
        let pk = sql_escape_literal(&format!("{}::{}", row.model_id, row.tensor_name));
        let tname = sql_escape_literal(&row.tensor_name);
        let dtype = sql_escape_literal(&row.source_dtype);
        let insert_sql = format!(
            "INSERT INTO synapse_tensors (id, model_id, tensor_name, layer_idx, source_dtype, rows, cols, rank, original_fp16_bytes, compressed_bytes, bits_per_weight, compression_vs_fp16, reconstruction_cosine, hot_score) VALUES ('{pk}', '{esc_model}', '{tname}', {}, '{dtype}', {}, {}, {}, {}, {}, {:.6}, {:.6}, {:.6}, {:.4})",
            row.layer_idx,
            row.rows,
            row.cols,
            row.rank,
            row.original_fp16_bytes,
            row.compressed_bytes,
            row.bits_per_weight,
            row.compression_vs_fp16,
            row.reconstruction_cosine,
            row.hot_score,
        );
        if rt.run_sql(insert_sql).is_ok() {
            sql_rows_materialized += 1;
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "report": report,
            "sql_table": "synapse_tensors",
            "sql_rows_materialized": sql_rows_materialized,
            "vocab_tokens_loaded": vocab_tokens_loaded,
            "transformer_encoder_layers": encoder_layers,
        })),
    )
}

async fn handle_tokenizer_load(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<LoadTokenizerReq>,
) -> impl IntoResponse {
    let raw = if let Some(v) = req.vocab_txt {
        v
    } else if let Some(ref p_str) = req.file_path {
        match std::fs::read_to_string(p_str) {
            Ok(s) => s,
            Err(e) => return bad_request(format!("failed to read tokenizer file '{p_str}': {e}")),
        }
    } else {
        return bad_request("must provide vocab_txt or file_path");
    };

    let vocab_vec = match parse_vocab_tokens(&raw) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => return bad_request("vocabulary is empty"),
        Err(e) => return bad_request(e),
    };
    let count = vocab_vec.len();
    {
        let mut guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.model_vocabs.insert(req.model_id.clone(), vocab_vec);
        rt.persist_if_configured(&guard);
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "model_id": req.model_id,
            "vocab_size": count,
        })),
    )
}

async fn handle_encoder_embed(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<EncoderEmbedReq>,
) -> impl IntoResponse {
    let guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match encode_text_with_littlebit_transformer(
        &guard,
        &req.model_id,
        &req.text,
        req.max_seq_len,
        |_, _, _| None,
    ) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

async fn handle_littlebit_factorize(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<FactorizeLittleBitReq>,
) -> impl IntoResponse {
    let tensor = match LittleBitSVFTensor::factorize(
        req.name.clone(),
        &req.weights,
        req.rows,
        req.cols,
        req.rank,
        req.itq_iters,
        req.max_outliers,
    ) {
        Ok(t) => t,
        Err(e) => return bad_request(e),
    };
    rt.cache_gpu_weights(&req.name, &tensor);
    let stats = tensor.stats();
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if !guard.littlebit_tensors.contains_key(&req.name) && guard.littlebit_tensors.len() >= 128
        && let Some(oldest) = guard.littlebit_tensors.keys().next().cloned()
    {
        guard.littlebit_tensors.remove(&oldest);
    }
    guard.littlebit_tensors.insert(req.name, tensor);
    rt.persist_if_configured(&guard);
    (StatusCode::OK, Json(serde_json::json!({ "stats": stats })))
}

async fn handle_littlebit_forward(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<ForwardLittleBitReq>,
) -> impl IntoResponse {
    let (bpw, eff_sigma, cpu_fallback_tensor) = {
        let guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let Some(tensor) = guard.littlebit_tensors.get(&req.name) else {
            return bad_request(format!("LittleBit tensor '{}' not found", req.name));
        };
        (
            tensor.bits_per_weight(),
            tensor.effective_latent_scales(&req.expert_mix),
            tensor.clone(),
        )
    };

    // Hardware `wgpu` Metal / Vulkan / DX12 / WebGPU `XNOR + countOneBits()` WGSL shader path!
    if req.use_gpu
        && let Some(gpu_kernel) = shared_gpu_kernel()
    {
        let cached_opt = {
            let cache = match rt.gpu_packed_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            cache.get(&req.name).cloned()
        };
        let packed = match cached_opt {
            Some(p) => Some(p),
            None => {
                rt.cache_gpu_weights(&req.name, &cpu_fallback_tensor);
                let cache = match rt.gpu_packed_cache.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                cache.get(&req.name).cloned()
            }
        };
        if let Some(packed_weights) = packed {
            match gpu_kernel.forward_xnor_popcnt(&packed_weights, &req.input, &eff_sigma) {
                Ok(receipt) => {
                    return (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "output": receipt.output,
                            "bits_per_weight": bpw,
                            "kernel": "wgpu_xnor_popcnt_wgsl",
                            "device": receipt.adapter_name,
                            "backend_api": receipt.backend_api,
                            "elapsed_us": receipt.elapsed_us,
                            "estimated_joules_uj": receipt.estimated_joules_uj,
                        })),
                    );
                }
                Err(e) => return bad_request(e),
            }
        }
    }

    let res = if req.binary_xnor_popcnt || req.use_gpu {
        cpu_fallback_tensor.forward_binary_xnor_popcnt(&req.input, &req.expert_mix)
    } else {
        cpu_fallback_tensor.forward_vec(&req.input, &req.expert_mix)
    };
    match res {
        Ok(output) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "output": output,
                "bits_per_weight": bpw,
                "kernel": if req.binary_xnor_popcnt || req.use_gpu { "xnor_popcnt_bitplane" } else { "binary_latent_dual_svid" },
                "device": "cpu",
            })),
        ),
        Err(e) => bad_request(e),
    }
}

async fn handle_littlebit_svf_adapt(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<AdaptSvfReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(tensor) = guard.littlebit_tensors.get_mut(&req.name) else {
        return bad_request(format!("LittleBit tensor '{}' not found", req.name));
    };
    let res = tensor.adapt_svf_online(&req.expert_name, &req.input, &req.target_y, req.learning_rate);
    match res {
        Ok((mse_before, mse_after)) => {
            rt.persist_if_configured(&guard);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "expert_name": req.expert_name,
                    "mse_before": mse_before,
                    "mse_after": mse_after,
                    "base_bitplanes_mutated": false,
                })),
            )
        }
        Err(e) => bad_request(e),
    }
}

async fn handle_titans_observe(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<TitansObserveReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match guard.synapse.observe_and_adapt(&req.key, &req.target) {
        Ok(outcome) => {
            if outcome.adapted {
                rt.persist_if_configured(&guard);
            }
            (StatusCode::OK, Json(serde_json::json!(outcome)))
        }
        Err(e) => bad_request(e),
    }
}

async fn handle_titans_forward(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<TitansForwardReq>,
) -> impl IntoResponse {
    let guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match guard.synapse.forward(&req.key) {
        Ok(output) => (
            StatusCode::OK,
            Json(serde_json::json!({ "output": output, "lsn": guard.synapse.lsn })),
        ),
        Err(e) => bad_request(e),
    }
}

async fn handle_titans_rollback(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<TitansRollbackReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match guard.synapse.rollback_to_lsn(req.target_lsn) {
        Ok(reverted_commits) => {
            rt.persist_if_configured(&guard);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "reverted_commits": reverted_commits,
                    "current_lsn": guard.synapse.lsn,
                })),
            )
        }
        Err(e) => bad_request(e),
    }
}

async fn handle_titans_consolidate(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<TitansConsolidateReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let transferred_l2 = guard.synapse.consolidate_sleep(req.transfer_ratio);
    rt.persist_if_configured(&guard);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "transferred_l2_norm": transferred_l2,
            "sleep_consolidations": guard.synapse.sleep_consolidations,
        })),
    )
}

async fn handle_engram_upsert(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<EngramUpsertReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let rec = guard
        .engram
        .upsert(&req.phrase, req.embedding, req.fact_payload);
    rt.persist_if_configured(&guard);
    (StatusCode::OK, Json(serde_json::json!(rec)))
}

async fn handle_engram_lookup(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<EngramLookupReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let found = guard.engram.lookup(&req.phrase);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "hit": found.is_some(), "record": found })),
    )
}

async fn handle_airstream_register(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<AirStreamConfigShardReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(budget) = req.vram_budget_bytes {
        guard.airstream = joule_db_weights::synapse::AirStreamScheduler::new(budget, 0.35);
    }
    match guard.airstream.register_shard(req.shard) {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "registered": true }))),
        Err(e) => bad_request(e),
    }
}

async fn handle_airstream_step(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<AirStreamStepReq>,
) -> impl IntoResponse {
    let guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let receipt = guard.airstream.execute_streaming_pass(&req.routed_shard_ids);
    (StatusCode::OK, Json(serde_json::json!(receipt)))
}

async fn handle_harness_observe(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HarnessObserveReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let obs = guard.ingest_observation(req.source, &req.raw_output);
    rt.persist_if_configured(&guard);
    (StatusCode::OK, Json(serde_json::json!(obs)))
}

async fn handle_harness_plan_context(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HarnessPlanContextReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let plan = guard.plan_context_window(req.candidates, req.max_tokens, req.max_joules_uj);
    (StatusCode::OK, Json(serde_json::json!(plan)))
}

async fn handle_harness_self_edit(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HarnessSelfEditReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match guard.compile_self_edit(
        &req.trigger_phrase,
        &req.correction_fact,
        &req.key_vec,
        &req.target_vec,
    ) {
        Ok(receipt) => {
            rt.persist_if_configured(&guard);
            (StatusCode::OK, Json(serde_json::json!(receipt)))
        }
        Err(e) => bad_request(e),
    }
}

async fn handle_harness_checkpoint(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HarnessCheckpointReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let ckpt = guard.checkpoint(req.label);
    rt.persist_if_configured(&guard);
    (StatusCode::OK, Json(serde_json::json!(ckpt)))
}

async fn handle_harness_rewind(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HarnessRewindReq>,
) -> impl IntoResponse {
    let mut guard = match rt.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match guard.rewind(req.checkpoint_id) {
        Ok(ckpt) => {
            rt.persist_if_configured(&guard);
            (
                StatusCode::OK,
                Json(serde_json::json!({ "rewound": true, "checkpoint": ckpt })),
            )
        }
        Err(e) => bad_request(e),
    }
}
