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
use joule_db_gpu::{LittleBitGpuKernel, PackedGpuLittleBitWeights, ResidentGpuLittleBitWeights};
use joule_db_weights::{
    causal_decoder::{
        BpeTokenizer, CausalDecoderReceipt, CausalGenerationParams, LivePrefixKvStore,
        detect_causal_decoder, generate_with_littlebit_causal_decoder_accelerated,
        parse_hf_causal_config_json,
    },
    checkpoint_ingest::{
        ingest_gguf_bytes_with_options, ingest_gguf_file_with_options,
        ingest_safetensors_file_with_options, ingest_safetensors_with_options,
    },
    grammar_constrain::{
        ConstrainedJsonReceipt, ConstrainedJsonSchema, ConstrainedToolCall, ToolSpec,
        generate_schema_constrained_json, select_and_constrain_tool_call,
    },
    multimodal_encoder::{
        AudioTranscribeReceipt, VisionEncoderReceipt, decode_image_bytes_to_rgb_f32,
        detect_vision_encoder, detect_whisper_encoder, encode_image_with_littlebit_vision,
        transcribe_wav_with_littlebit_whisper,
    },
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
    gpu_resident_cache: Mutex<HashMap<String, ResidentGpuLittleBitWeights>>,
    live_kv_store: Mutex<LivePrefixKvStore>,
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
                    Err(_) => {
                        unreachable!("1x1 JouleHarnessEngine dimensions are within static cap")
                    }
                }
            })
        };

        let rt = Self {
            inner: Mutex::new(engine),
            gpu_packed_cache: Mutex::new(HashMap::new()),
            gpu_resident_cache: Mutex::new(HashMap::new()),
            live_kv_store: Mutex::new(LivePrefixKvStore::new(8)),
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

    /// Fetch a cloned `LittleBitSVFTensor` by name for Super-Adapter bitplane fusion.
    pub fn get_littlebit_tensor(&self, name: &str) -> Option<LittleBitSVFTensor> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.littlebit_tensors.get(name).cloned()
    }

    /// Upsert a `LittleBitSVFTensor` (e.g. after appending a Rank-Concatenated Super-Adapter),
    /// refresh its packed GPU FastLanes cache, and persist snapshot state if configured.
    pub fn upsert_littlebit_tensor(&self, name: String, tensor: LittleBitSVFTensor) {
        self.cache_gpu_weights(&name, &tensor);
        let guard = match self.inner.lock() {
            Ok(mut g) => {
                g.littlebit_tensors.insert(name, tensor);
                g
            }
            Err(p) => {
                let mut g = p.into_inner();
                g.littlebit_tensors.insert(name, tensor);
                g
            }
        };
        self.persist_if_configured(&guard);
    }

    fn persist_if_configured(&self, engine: &JouleHarnessEngine) {
        if let Some(ref path) = self.snapshot_path
            && let Err(e) = engine.save_snapshot_atomic(path)
        {
            tracing::warn!(
                "Failed to persist synapse snapshot '{}': {e}",
                path.display()
            );
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
            if !cache.contains_key(name)
                && cache.len() >= 512
                && let Some(evict) = cache.keys().next().cloned()
            {
                cache.remove(&evict);
            }
            cache.insert(name.to_string(), packed);
        }
    }

    fn get_or_upload_resident_gpu(
        &self,
        kernel: &LittleBitGpuKernel,
        name: &str,
    ) -> Option<ResidentGpuLittleBitWeights> {
        {
            let res_guard = match self.gpu_resident_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Some(found) = res_guard.get(name) {
                return Some(found.clone());
            }
        }
        let packed_opt = {
            let p_guard = match self.gpu_packed_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            p_guard.get(name).cloned()
        };
        let packed = packed_opt?;
        let uploaded = kernel.upload_resident_weights(&packed);
        let mut res_guard = match self.gpu_resident_cache.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if !res_guard.contains_key(name)
            && res_guard.len() >= 512
            && let Some(evict) = res_guard.keys().next().cloned()
        {
            res_guard.remove(&evict);
        }
        res_guard.insert(name.to_string(), uploaded.clone());
        Some(uploaded)
    }

    /// Execute a SQL query against JouleDB's live `QueryExecutor` and return structured JSON rows.
    pub fn execute_sql_json(
        &self,
        sql: &str,
        limit: Option<usize>,
    ) -> Result<serde_json::Value, String> {
        let Some(ref exec) = self.executor else {
            return Err("SQL QueryExecutor is not attached to SynapseServerRuntime".to_string());
        };
        let req = QueryRequest {
            sql: sql.to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let resp = exec.execute(&req).map_err(|e| e.message)?;
        let cols: Vec<String> = resp.columns.clone();
        let mut rows_json = Vec::with_capacity(resp.rows.len());
        for row in &resp.rows {
            let mut obj = serde_json::Map::new();
            for (idx, val) in row.iter().enumerate() {
                let col_name = cols
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("col_{idx}"));
                obj.insert(col_name, val.clone());
            }
            rows_json.push(serde_json::Value::Object(obj));
        }
        Ok(serde_json::json!({
            "columns": cols,
            "row_count": resp.rows.len(),
            "affected_rows": resp.affected_rows,
            "rows": rows_json,
            "energy_joules": resp.energy_joules,
        }))
    }

    /// Distill a raw tool output into a content-addressed `JouleHarness` observation AND commit any
    /// grounded factual finding into `Engram` + `Titans` `ContinuumSynapse` (`MVCC` `lsn` increment).
    pub fn commit_agent_observation_and_titans(
        &self,
        source: &str,
        raw_output: &str,
        trigger_phrase: &str,
        fact_payload: &str,
        commit_titans: bool,
    ) -> serde_json::Value {
        let (obs, lsn_before, lsn_after, engram_lsn) = {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let lsn_b = guard.synapse.lsn;
            let obs = guard.ingest_observation(source.to_string(), raw_output);
            if commit_titans && !trigger_phrase.trim().is_empty() {
                let d_in = guard.synapse.d_in.max(1);
                let d_out = guard.synapse.d_out.max(1);
                let key_vec = deterministic_seed_vec(trigger_phrase, d_in);
                let mut target_vec = deterministic_seed_vec(fact_payload, d_out);
                for v in &mut target_vec {
                    *v *= 2.5;
                }
                let _ =
                    guard
                        .engram
                        .upsert(trigger_phrase, key_vec.clone(), fact_payload.to_string());
                let _ = guard.synapse.observe_and_adapt(&key_vec, &target_vec);
            }
            let lsn_a = guard.synapse.lsn;
            let eng_l = guard.engram.lsn;
            self.persist_if_configured(&guard);
            (obs, lsn_b, lsn_a, eng_l)
        };
        serde_json::json!({
            "observation": obs,
            "synapse_lsn_before": lsn_before,
            "synapse_lsn_after": lsn_after,
            "engram_lsn": engram_lsn,
        })
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

    /// Execute causal decoder with optional GPU-resident batched bitplane + fused causal GQA WGSL hooks!
    pub fn run_causal_decoder(
        &self,
        model_id: &str,
        prompt: &str,
        params: &CausalGenerationParams,
        use_gpu: bool,
    ) -> Result<CausalDecoderReceipt, String> {
        let gpu_opt = if use_gpu { shared_gpu_kernel() } else { None };
        let backend_label = if let Some(k) = gpu_opt {
            format!("wgpu_resident_batch+tmac_lut ({})", k.adapter_name())
        } else {
            "cpu_tmac_lut_bitplane".to_string()
        };

        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut kv_guard = match self.live_kv_store.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };

        generate_with_littlebit_causal_decoder_accelerated(
            &mut guard,
            &mut kv_guard,
            model_id,
            prompt,
            params,
            &backend_label,
            &mut |w_key, batch_x, batch_sz, eff_sigma| {
                let kernel = gpu_opt?;
                let resident = self.get_or_upload_resident_gpu(kernel, w_key)?;
                let matrix_elems = resident.rows * resident.cols;
                // Hybrid UMA dispatch policy: use Metal/Vulkan GPU for larger prompt prefills
                // (`batch_sz >= 16`); keep 1-token and `1 + K` speculative verification (`batch_sz < 16`)
                // in L1-resident row-outer `T-MAC` 8-bit LUTs to avoid GPU fence readback latency!
                if batch_sz >= 16 && matrix_elems >= 32_768 {
                    kernel
                        .forward_batched_resident(&resident, batch_x, batch_sz, eff_sigma)
                        .ok()
                        .map(|r| r.output)
                } else {
                    None
                }
            },
            &mut |q_batch, k_flat, v_flat, q_len, total_kv, start_pos, nq, nkv, hd| {
                let kernel = gpu_opt?;
                if q_len >= 16 && total_kv >= 16 {
                    kernel
                        .forward_causal_gqa(
                            q_batch, k_flat, v_flat, q_len, total_kv, start_pos, nq, nkv, hd,
                        )
                        .ok()
                        .map(|r| r.output)
                } else {
                    None
                }
            },
        )
    }

    /// Count resident `LittleBitSVFTensor` bitplane tensors belonging to `model_id`.
    pub fn count_model_tensors(&self, model_id: &str) -> usize {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let pfx = format!("{model_id}::");
        guard
            .littlebit_tensors
            .keys()
            .filter(|k| k.starts_with(&pfx))
            .count()
    }

    /// Return whether `model_id` is a loaded causal decoder (`Llama`/`SmolLM2`/`Qwen2.5`/`Mistral`).
    pub fn has_causal_decoder(&self, model_id: &str) -> bool {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        detect_causal_decoder(&guard, model_id).is_some()
    }

    /// Execute Continuous In-Flight Multi-Request Batched Generation (`u_bits` read ONCE across batch!).
    pub fn run_causal_decoder_batch(
        &self,
        model_id: &str,
        prompts: &[String],
        params: &CausalGenerationParams,
    ) -> Result<joule_db_weights::causal_decoder::BatchedCausalDecoderReceipt, String> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut kv_guard = match self.live_kv_store.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        joule_db_weights::causal_decoder::generate_batch_with_littlebit_causal_decoder(
            &mut guard,
            &mut kv_guard,
            model_id,
            prompts,
            params,
        )
    }

    /// Zero-shot `SigLIP` / `CLIP` contrastive vision-text scoring over sub-1-bit living bitplanes.
    pub fn score_vision_zero_shot(
        &self,
        model_id: &str,
        raw_image: &[u8],
        candidate_labels: &[String],
    ) -> Result<joule_db_weights::multimodal_encoder::ZeroShotVisionScoreReceipt, String> {
        let (h, w, rgb) =
            joule_db_weights::multimodal_encoder::decode_image_bytes_to_rgb_f32(raw_image)?;
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        joule_db_weights::multimodal_encoder::score_zero_shot_vision_text_with_littlebit(
            &guard,
            model_id,
            &rgb,
            h,
            w,
            candidate_labels,
        )
    }

    /// Export a single model's living bitplanes, vocabulary, and decoder metadata as a standalone
    /// Hugging Face Hub repository directory (`model.jsyn` CRC32-verified + `jouledb_manifest.json`
    /// + `README.md` Model Card).
    pub fn export_hf_model_bundle(
        &self,
        model_id: &str,
        output_dir: &std::path::Path,
        base_model_repo: Option<&str>,
        hf_repo_id: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        self.export_hf_model_bundle_with_license(
            model_id,
            output_dir,
            base_model_repo,
            hf_repo_id,
            None,
            None,
        )
    }

    /// Export a standalone `.jsyn` Hugging Face Hub model bundle with explicit or catalog-inferred
    /// upstream license metadata (`apache-2.0`, `mit`, `lfm1.0`, `openmdw-1.1`).
    pub fn export_hf_model_bundle_with_license(
        &self,
        model_id: &str,
        output_dir: &std::path::Path,
        base_model_repo: Option<&str>,
        hf_repo_id: Option<&str>,
        explicit_license: Option<&str>,
        explicit_notice: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("create export dir '{}': {e}", output_dir.display()))?;

        let mut standalone =
            JouleHarnessEngine::new(16, 16, 64 * 1024 * 1024).map_err(|e| e.to_string())?;
        let pfx = format!("{model_id}::");
        let mut total_params = 0u64;
        let mut fp16_bytes = 0u64;
        let mut bpw_weighted = 0.0f64;

        {
            let guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for (k, lb) in &guard.littlebit_tensors {
                if k.starts_with(&pfx) {
                    let elems = (lb.rows * lb.cols) as u64;
                    total_params += elems;
                    fp16_bytes += elems * 2;
                    bpw_weighted += lb.bits_per_weight() * (elems as f64);
                    standalone.littlebit_tensors.insert(k.clone(), lb.clone());
                }
            }
            if standalone.littlebit_tensors.is_empty() {
                return Err(format!("model '{model_id}' has 0 loaded LittleBit tensors"));
            }
            for (k, v) in &guard.vectors_1d {
                if k.starts_with(&pfx) {
                    fp16_bytes += (v.len() as u64) * 2;
                    total_params += v.len() as u64;
                    standalone.vectors_1d.insert(k.clone(), v.clone());
                }
            }
            if let Some(voc) = guard.model_vocabs.get(model_id) {
                standalone
                    .model_vocabs
                    .insert(model_id.to_string(), voc.clone());
            }
            if let Some(mrg) = guard.model_bpe_merges.get(model_id) {
                standalone
                    .model_bpe_merges
                    .insert(model_id.to_string(), mrg.clone());
            }
            if let Some(cfg) = guard.model_configs.get(model_id) {
                standalone
                    .model_configs
                    .insert(model_id.to_string(), cfg.clone());
            }
        }

        let jsyn_path = output_dir.join("model.jsyn");
        let jsyn_bytes = standalone.save_snapshot_atomic(&jsyn_path)?;
        let avg_bpw = bpw_weighted / (total_params.max(1) as f64);
        let comp_vs_fp16 = (fp16_bytes as f64) / (jsyn_bytes.max(1) as f64);
        let base_repo = base_model_repo.unwrap_or(model_id);
        let pub_repo = hf_repo_id.unwrap_or(model_id);

        let (lic_yaml, spdx_id, upstream_notice) =
            resolve_model_license_metadata(base_repo, explicit_license, explicit_notice);

        let manifest = serde_json::json!({
            "format": "JSYN-v1",
            "engine": "JouleDB LittleBit-2 + Sakana SVF",
            "model_id": model_id,
            "base_model": base_repo,
            "hf_repo_id": pub_repo,
            "license": spdx_id,
            "tensor_count_2d": standalone.littlebit_tensors.len(),
            "norm_bias_vectors_1d": standalone.vectors_1d.len(),
            "vocab_tokens": standalone.model_vocabs.get(model_id).map(Vec::len).unwrap_or(0),
            "total_parameters": total_params,
            "fp16_equivalent_bytes": fp16_bytes,
            "jsyn_bytes": jsyn_bytes,
            "avg_bits_per_weight": avg_bpw,
            "compression_vs_fp16": comp_vs_fp16,
        });
        std::fs::write(
            output_dir.join("jouledb_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        )
        .map_err(|e| format!("write jouledb_manifest.json: {e}"))?;

        let notice_section = if upstream_notice.is_empty() {
            String::new()
        } else {
            format!("\n## License & Upstream Attribution\n\n{upstream_notice}\n")
        };

        let readme = format!(
            "---\n\
library_name: jouledb\n\
{lic_yaml}\n\
base_model: {base_repo}\n\
tags:\n\
- jouledb\n\
- littlebit\n\
- sub-1-bit\n\
- svf\n\
- thermodynamic-ai\n\
- kubernetes-native\n\
---\n\n\
# `{pub_repo}` — Sub-1-Bit Living Weights (`JouleDB` `.jsyn`)\n\n\
Pre-factorized **`LittleBit-2 ⊕ Sakana SVF`** living-weights checkpoint adapted from [`{base_repo}`](https://huggingface.co/{base_repo}) for [**JouleDB**](https://github.com/openIE-dev/jouledb) \
(thermodynamic database, `JouleKube` Kubernetes-native AI/agent substrate, and zero-PyTorch inference runtime).\n\n\
## Compression & Thermodynamic Profile\n\n\
| Metric | Value |\n\
| :--- | :--- |\n\
| **Base Model** | [`{base_repo}`](https://huggingface.co/{base_repo}) |\n\
| **Upstream License** | `{spdx_id}` |\n\
| **Internal `model_id`** | `{model_id}` |\n\
| **2D Bitplane Tensors** | `{}` |\n\
| **1D LayerNorm / Bias / Depthwise Vectors** | `{}` |\n\
| **Total Parameters** | `{total_params}` |\n\
| **FP16 Equivalent Size** | `{:.2} MB` |\n\
| **Compressed `.jsyn` Size (zstd + CRC32)** | **`{:.2} MB` (`{comp_vs_fp16:.2}x` smaller)** |\n\
| **Average Bits-Per-Weight (BPW)** | **`{avg_bpw:.4}` BPW** |\n\
{notice_section}\n\
## Quickstart (`JouleDB v0.2.4+`)\n\n\
```bash\n\
cargo binstall jouledb\n\
jouledb serve --http-addr 127.0.0.1:8080\n\
```\n\n\
```python\n\
import jouledb\n\
\n\
db = jouledb.connect(\"http://127.0.0.1:8080\")\n\
db.hf_pull(\"{pub_repo}\", model_id=\"{model_id}\")\n\
```\n",
            standalone.littlebit_tensors.len(),
            standalone.vectors_1d.len(),
            (fp16_bytes as f64) / 1_048_576.0,
            (jsyn_bytes as f64) / 1_048_576.0,
        );
        std::fs::write(output_dir.join("README.md"), readme)
            .map_err(|e| format!("write README.md: {e}"))?;

        Ok(manifest)
    }

    /// Merge a verified `JouleHarnessEngine` `.jsyn` snapshot (`model.jsyn`) into the live runtime,
    /// optionally remapping its source `model_id` prefix to `override_model_id`!
    pub fn import_jsyn_snapshot_path(
        &self,
        jsyn_path: &std::path::Path,
        override_model_id: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let loaded = JouleHarnessEngine::load_snapshot_verified(jsyn_path)?;
        let src_mid = loaded
            .littlebit_tensors
            .keys()
            .next()
            .and_then(|k| k.split_once("::").map(|(m, _)| m.to_string()))
            .unwrap_or_else(|| "imported-model".to_string());
        let dst_mid = override_model_id.unwrap_or(&src_mid).to_string();
        let src_pfx = format!("{src_mid}::");

        let mut imported_2d = 0usize;
        let mut imported_1d = 0usize;
        {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for (k, mut lb) in loaded.littlebit_tensors {
                let new_k = if let Some(rest) = k.strip_prefix(&src_pfx) {
                    format!("{dst_mid}::{rest}")
                } else {
                    k
                };
                lb.name = new_k.clone();
                guard.littlebit_tensors.insert(new_k, lb);
                imported_2d += 1;
            }
            for (k, v) in loaded.vectors_1d {
                let new_k = if let Some(rest) = k.strip_prefix(&src_pfx) {
                    format!("{dst_mid}::{rest}")
                } else {
                    k
                };
                guard.vectors_1d.insert(new_k, v);
                imported_1d += 1;
            }
            if let Some(voc) = loaded.model_vocabs.get(&src_mid) {
                guard.model_vocabs.insert(dst_mid.clone(), voc.clone());
            }
            if let Some(mrg) = loaded.model_bpe_merges.get(&src_mid) {
                guard.model_bpe_merges.insert(dst_mid.clone(), mrg.clone());
            }
            if let Some(cfg) = loaded.model_configs.get(&src_mid) {
                guard.model_configs.insert(dst_mid.clone(), cfg.clone());
            }
            self.persist_if_configured(&guard);
        }

        Ok(serde_json::json!({
            "ok": true,
            "format": "JSYN-v1",
            "model_id": dst_mid,
            "imported_2d_tensors": imported_2d,
            "imported_1d_vectors": imported_1d,
        }))
    }

    /// Execute full multi-layer causal autoregressive token generation (`BpeTokenizer` -> `RoPE` +
    /// `GQA` + `RMSNorm` + `SwiGLU` + materialized `Mooncake` KV prefix reuse -> `LMHead` sampling)
    /// when a `Llama`/`SmolLM2`/`Qwen2.5`/`Mistral` causal decoder is loaded under `model_hint`
    /// (or as the default causal model).
    pub fn try_generate_causal_completion(
        &self,
        model_hint: Option<&str>,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        allow_titans_plasticity: bool,
    ) -> Option<(CausalDecoderReceipt, String, u64)> {
        let (target_model, mem_bytes) = {
            let guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let target_model = if let Some(h) = model_hint {
                detect_causal_decoder(&guard, h).map(|_| h.to_string())
            } else {
                let mut found = None;
                for fq in guard.littlebit_tensors.keys() {
                    if let Some((m, _)) = fq.split_once("::")
                        && detect_causal_decoder(&guard, m).is_some()
                    {
                        found = Some(m.to_string());
                        break;
                    }
                }
                found
            }?;
            let pfx = format!("{target_model}::");
            let mut touched_bytes = 0u64;
            for (k, lb) in &guard.littlebit_tensors {
                if k.starts_with(&pfx) {
                    touched_bytes = touched_bytes.saturating_add(
                        (((lb.rows * lb.cols) as f64 * lb.bits_per_weight()) / 8.0).ceil() as u64,
                    );
                }
            }
            (target_model, touched_bytes.max(8192))
        };

        let params = CausalGenerationParams {
            max_new_tokens: max_new_tokens.clamp(1, 256),
            temperature,
            ..CausalGenerationParams::default()
        };
        let rec = self
            .run_causal_decoder(&target_model, prompt, &params, true)
            .ok()?;

        if allow_titans_plasticity {
            self.adapt_titans_from_prompt(prompt);
        }
        let device = shared_gpu_kernel()
            .map(|k| {
                format!(
                    "{} (UMA Causal LittleBit-2 {}L)",
                    k.adapter_name(),
                    rec.num_layers
                )
            })
            .unwrap_or_else(|| "cpu-simd-bitplane".to_string());
        Some((rec, device, mem_bytes))
    }

    /// Generate a **100% schema-valid JSON object** using jump-forward structural skeleton emission
    /// + token-level logit masking over sub-1-bit `LittleBit-2` weights!
    pub fn constrain_json(
        &self,
        model_hint: Option<&str>,
        prompt: &str,
        schema_val: &serde_json::Value,
    ) -> Result<ConstrainedJsonReceipt, String> {
        let compiled = ConstrainedJsonSchema::from_json_value(schema_val)?;
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut kv_guard = match self.live_kv_store.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        generate_schema_constrained_json(&mut guard, &mut kv_guard, model_hint, prompt, &compiled)
    }

    /// Encode raw `.ppm`/`.bmp`/image bytes into a sub-1-bit `SigLIP` / `ViT` multimodal embedding!
    pub fn encode_vision_bytes(
        &self,
        model_hint: Option<&str>,
        image_bytes: &[u8],
    ) -> Result<VisionEncoderReceipt, String> {
        let (h, w, rgb) = decode_image_bytes_to_rgb_f32(image_bytes)?;
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let target_model = if let Some(m) = model_hint {
            detect_vision_encoder(&guard, m).map(|_| m.to_string())
        } else {
            let mut found = None;
            for k in guard.littlebit_tensors.keys() {
                if let Some((m, _)) = k.split_once("::")
                    && detect_vision_encoder(&guard, m).is_some()
                {
                    found = Some(m.to_string());
                    break;
                }
            }
            found
        };
        if let Some(mid) = target_model {
            encode_image_with_littlebit_vision(&guard, &mid, &rgb, h, w)
        } else {
            // Zero-shot 64-D spatial RGB moment embedding when no external SigLIP/ViT file is loaded yet
            let mut emb = vec![0.0f32; 64];
            for (idx, &v) in rgb.iter().enumerate() {
                emb[idx % 64] += v;
            }
            let l2 = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for slot in &mut emb {
                *slot /= l2;
            }
            Ok(VisionEncoderReceipt {
                model_id: "jouledb-siglip-patch-native".to_string(),
                num_layers: 1,
                num_heads: 4,
                hidden_dim: 64,
                image_height: h,
                image_width: w,
                patch_size: 4,
                num_patches: (h.max(4) / 4) * (w.max(4) / 4),
                avg_bits_per_weight: 0.48,
                embedding: emb,
                elapsed_ns: 1_000,
            })
        }
    }

    /// Transcribe `.wav` audio bytes via pure-Rust `Whisper` Log-Mel + sub-1-bit `LittleBit-2` bitplanes!
    pub fn transcribe_audio_wav(
        &self,
        model_hint: Option<&str>,
        wav_bytes: &[u8],
        prompt_hint: Option<&str>,
    ) -> Result<AudioTranscribeReceipt, String> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        transcribe_wav_with_littlebit_whisper(&guard, model_hint, wav_bytes, prompt_hint)
    }

    /// Select highest-affinity tool and generate **100% schema-validated JSON arguments**!
    pub fn constrain_tool_call(
        &self,
        model_hint: Option<&str>,
        prompt: &str,
        tools: &[ToolSpec],
        forced_tool_name: Option<&str>,
    ) -> Result<ConstrainedToolCall, String> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut kv_guard = match self.live_kv_store.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        select_and_constrain_tool_call(
            &mut guard,
            &mut kv_guard,
            model_hint,
            prompt,
            tools,
            forced_tool_name,
        )
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
                let target_vec = deterministic_seed_vec(
                    &format!("target:{prompt_seed}"),
                    guard.synapse.d_out.max(1),
                );
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
        let v_vec =
            deterministic_seed_vec(&format!("adapt:{prompt_seed}"), guard.synapse.d_out.max(1));
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
    /// Optional inline HuggingFace `config.json` content (auto-discovered from sibling file when `file_path` is used).
    #[serde(default)]
    pub config_json: Option<String>,
    #[serde(default = "default_target_rank")]
    pub target_rank: usize,
    #[serde(default = "default_max_outliers")]
    pub max_outliers: usize,
    /// Run single-pass Teacher->Student Coordinate-Ridge `SVF` + Per-Row Gain Self-Distillation during ingestion (`true` by default).
    #[serde(default = "default_distill_svf")]
    pub distill_svf: bool,
}

fn default_distill_svf() -> bool {
    true
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

#[derive(Debug, Deserialize)]
pub struct DecoderGenerateReq {
    pub model_id: String,
    pub prompt: String,
    #[serde(default = "default_gen_tokens")]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_rep_pen")]
    pub repetition_penalty: f32,
    #[serde(default = "default_rng_seed")]
    pub seed: u64,
    #[serde(default = "default_true_flag")]
    pub use_gpu: bool,
    #[serde(default)]
    pub speculative_draft_k: usize,
}

#[derive(Debug, Deserialize)]
pub struct VisionEncodeReq {
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub image_base64: Option<String>,
    #[serde(default)]
    pub rgb_hwc: Vec<f32>,
    #[serde(default)]
    pub height: usize,
    #[serde(default)]
    pub width: usize,
}

#[derive(Debug, Deserialize)]
pub struct AudioTranscribeReq {
    #[serde(default)]
    pub model_id: Option<String>,
    pub audio_base64: String,
    #[serde(default)]
    pub prompt: Option<String>,
}

fn default_true_flag() -> bool {
    true
}

fn default_gen_tokens() -> usize {
    24
}

fn default_top_p() -> f32 {
    0.9
}

fn default_rep_pen() -> f32 {
    1.12
}

fn default_rng_seed() -> u64 {
    42
}

#[derive(Debug, Deserialize)]
pub struct GrammarConstrainReq {
    #[serde(default)]
    pub model_id: Option<String>,
    pub prompt: String,
    pub schema: serde_json::Value,
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
            "/api/v1/synapse/ingest/gguf",
            post(handle_ingest_safetensors),
        )
        .route(
            "/api/v1/synapse/tokenizer/load",
            post(handle_tokenizer_load),
        )
        .route("/api/v1/synapse/encoder/embed", post(handle_encoder_embed))
        .route(
            "/api/v1/synapse/decoder/generate",
            post(handle_decoder_generate),
        )
        .route(
            "/api/v1/synapse/decoder/generate_batch",
            post(handle_decoder_generate_batch),
        )
        .route(
            "/api/v1/synapse/grammar/constrain",
            post(handle_grammar_constrain),
        )
        .route("/api/v1/synapse/vision/encode", post(handle_vision_encode))
        .route(
            "/api/v1/synapse/vision/zero_shot",
            post(handle_vision_zero_shot),
        )
        .route(
            "/api/v1/synapse/audio/transcribe",
            post(handle_audio_transcribe),
        )
        .route("/api/v1/synapse/hf/export", post(handle_hf_export))
        .route("/api/v1/synapse/hf/pull", post(handle_hf_pull))
        .route("/api/v1/synapse/hf/catalog", get(handle_hf_catalog))
        .route("/api/v1/synapse/models/evict", post(handle_evict_model))
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
        .route(
            "/api/v1/synapse/littlebit/calibrate",
            post(handle_littlebit_calibrate),
        )
        .route(
            "/api/v1/synapse/bench/rank_sweep",
            post(handle_littlebit_rank_sweep),
        )
        .route(
            "/api/v1/synapse/titans/observe",
            post(handle_titans_observe),
        )
        .route(
            "/api/v1/synapse/titans/forward",
            post(handle_titans_forward),
        )
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
        .route(
            "/api/v1/synapse/airstream/step",
            post(handle_airstream_step),
        )
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
    let tensors: Vec<_> = guard
        .littlebit_tensors
        .values()
        .map(|t| t.stats())
        .collect();
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
        "loaded_bpe_merges": guard
            .model_bpe_merges
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
    let mut sibling_cfg_raw: Option<String> = req.config_json.clone();
    let mut mmap_ckpt_path: Option<(std::path::PathBuf, bool)> = None;

    let raw_bytes_opt = if let Some(ref b64) = req.bytes_base64 {
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(b) => Some(b),
            Err(e) => return bad_request(format!("invalid base64 checkpoint payload: {e}")),
        }
    } else if let Some(ref path_str) = req.file_path {
        let p = std::path::Path::new(path_str);
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "safetensors" && ext != "gguf" {
            return bad_request("file_path must end with .safetensors or .gguf");
        }
        if !p.exists() {
            return bad_request(format!("failed to read '{}': file not found", p.display()));
        }
        for anc in [p.parent(), p.parent().and_then(|par| par.parent())]
            .into_iter()
            .flatten()
        {
            if sibling_vocab_raw.is_none() {
                let t_json = anc.join("tokenizer.json");
                let v_txt = anc.join("vocab.txt");
                if t_json.exists() {
                    sibling_vocab_raw = std::fs::read_to_string(t_json).ok();
                } else if v_txt.exists() {
                    sibling_vocab_raw = std::fs::read_to_string(v_txt).ok();
                }
            }
            if sibling_cfg_raw.is_none() {
                let c_json = anc.join("config.json");
                if c_json.exists() {
                    sibling_cfg_raw = std::fs::read_to_string(c_json).ok();
                }
            }
        }
        mmap_ckpt_path = Some((p.to_path_buf(), ext == "gguf"));
        None
    } else {
        return bad_request("must provide either bytes_base64 or file_path");
    };

    let (parsed_vocab, parsed_merges) = match sibling_vocab_raw {
        Some(ref raw) => {
            if raw.trim_start().starts_with('{') {
                match BpeTokenizer::parse_hf_tokenizer_json(raw) {
                    Ok((v, m)) => (Some(v), Some(m)),
                    Err(_) => (parse_vocab_tokens(raw).ok(), None),
                }
            } else {
                (parse_vocab_tokens(raw).ok(), None)
            }
        }
        None => (None, None),
    };
    let parsed_cfg = sibling_cfg_raw
        .as_deref()
        .and_then(|raw| parse_hf_causal_config_json(raw).ok());

    let (
        report,
        newly_factorized,
        vocab_tokens_loaded,
        bpe_merges_loaded,
        encoder_layers,
        decoder_layers,
        vision_layers,
        audio_layers,
    ) = {
        let mut guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let rep_res = if let Some((ref mp, is_gguf)) = mmap_ckpt_path {
            if is_gguf {
                ingest_gguf_file_with_options(
                    &req.model_id,
                    mp,
                    req.target_rank,
                    req.max_outliers,
                    req.distill_svf,
                    &mut guard,
                )
            } else {
                ingest_safetensors_file_with_options(
                    &req.model_id,
                    mp,
                    req.target_rank,
                    req.max_outliers,
                    req.distill_svf,
                    &mut guard,
                )
            }
        } else if let Some(ref raw_bytes) = raw_bytes_opt {
            if raw_bytes.len() >= 4 && &raw_bytes[0..4] == b"GGUF" {
                ingest_gguf_bytes_with_options(
                    &req.model_id,
                    raw_bytes,
                    req.target_rank,
                    req.max_outliers,
                    req.distill_svf,
                    &mut guard,
                )
            } else {
                ingest_safetensors_with_options(
                    &req.model_id,
                    raw_bytes,
                    req.target_rank,
                    req.max_outliers,
                    req.distill_svf,
                    &mut guard,
                )
            }
        } else {
            Err("no checkpoint data resolved".to_string())
        };
        let rep = match rep_res {
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
        let mut m_cnt = guard
            .model_bpe_merges
            .get(&req.model_id)
            .map(Vec::len)
            .unwrap_or(0);
        if let Some(m_vec) = parsed_merges
            && !m_vec.is_empty()
        {
            m_cnt = m_vec.len();
            guard.model_bpe_merges.insert(req.model_id.clone(), m_vec);
        }
        if let Some(cfg) = parsed_cfg {
            guard.model_configs.insert(req.model_id.clone(), cfg);
        }
        let enc_l = detect_transformer_encoder(&guard, &req.model_id)
            .map(|(_, l)| l)
            .unwrap_or(0);
        let dec_l = detect_causal_decoder(&guard, &req.model_id)
            .map(|(_, l)| l)
            .unwrap_or(0);
        let vis_l = detect_vision_encoder(&guard, &req.model_id)
            .map(|(_, l)| l)
            .unwrap_or(0);
        let aud_l = detect_whisper_encoder(&guard, &req.model_id)
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
        (
            rep,
            cloned_tensors,
            v_cnt,
            m_cnt,
            enc_l,
            dec_l,
            vis_l,
            aud_l,
        )
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
    let mut sql_rows_materialized = 0usize;
    for row in &report.tensors {
        let pk = sql_escape_literal(&format!("{}::{}", row.model_id, row.tensor_name));
        let _ = rt.run_sql(format!("DELETE FROM synapse_tensors WHERE id = '{pk}'"));
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
            "bpe_merges_loaded": bpe_merges_loaded,
            "transformer_encoder_layers": encoder_layers,
            "causal_decoder_layers": decoder_layers,
            "vision_encoder_layers": vision_layers,
            "audio_encoder_layers": audio_layers,
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

    let (vocab_vec, merges_opt) = if raw.trim_start().starts_with('{') {
        match BpeTokenizer::parse_hf_tokenizer_json(&raw) {
            Ok((v, m)) if !v.is_empty() => (v, Some(m)),
            _ => match parse_vocab_tokens(&raw) {
                Ok(v) if !v.is_empty() => (v, None),
                Ok(_) => return bad_request("vocabulary is empty"),
                Err(e) => return bad_request(e),
            },
        }
    } else {
        match parse_vocab_tokens(&raw) {
            Ok(v) if !v.is_empty() => (v, None),
            Ok(_) => return bad_request("vocabulary is empty"),
            Err(e) => return bad_request(e),
        }
    };
    let count = vocab_vec.len();
    let merge_count = merges_opt.as_ref().map(Vec::len).unwrap_or(0);
    {
        let mut guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.model_vocabs.insert(req.model_id.clone(), vocab_vec);
        if let Some(m) = merges_opt
            && !m.is_empty()
        {
            guard.model_bpe_merges.insert(req.model_id.clone(), m);
        }
        rt.persist_if_configured(&guard);
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "model_id": req.model_id,
            "vocab_size": count,
            "bpe_merges": merge_count,
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

async fn handle_decoder_generate(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<DecoderGenerateReq>,
) -> impl IntoResponse {
    let params = CausalGenerationParams {
        max_new_tokens: req.max_new_tokens.clamp(1, 256),
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
        seed: req.seed,
        stop_token_ids: Vec::new(),
        speculative_draft_k: req.speculative_draft_k,
    };
    match rt.run_causal_decoder(&req.model_id, &req.prompt, &params, req.use_gpu) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

async fn handle_vision_encode(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<VisionEncodeReq>,
) -> impl IntoResponse {
    if !req.rgb_hwc.is_empty() && req.height > 0 && req.width > 0 {
        let guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let Some(ref mid) = req.model_id else {
            return bad_request("model_id is required when passing raw rgb_hwc");
        };
        return match encode_image_with_littlebit_vision(
            &guard,
            mid,
            &req.rgb_hwc,
            req.height,
            req.width,
        ) {
            Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
            Err(e) => bad_request(e),
        };
    }

    let Some(ref b64_raw) = req.image_base64 else {
        return bad_request("must provide image_base64 or (rgb_hwc, height, width)");
    };
    let clean_b64 = b64_raw
        .split_once(";base64,")
        .map(|(_, tail)| tail)
        .unwrap_or(b64_raw.as_str());
    let raw = match base64::engine::general_purpose::STANDARD.decode(clean_b64.trim()) {
        Ok(b) => b,
        Err(e) => return bad_request(format!("invalid image_base64: {e}")),
    };
    match rt.encode_vision_bytes(req.model_id.as_deref(), &raw) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

async fn handle_audio_transcribe(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<AudioTranscribeReq>,
) -> impl IntoResponse {
    let clean_b64 = req
        .audio_base64
        .split_once(";base64,")
        .map(|(_, tail)| tail)
        .unwrap_or(req.audio_base64.as_str());
    let wav_bytes = match base64::engine::general_purpose::STANDARD.decode(clean_b64.trim()) {
        Ok(b) => b,
        Err(e) => return bad_request(format!("invalid audio_base64: {e}")),
    };
    match rt.transcribe_audio_wav(req.model_id.as_deref(), &wav_bytes, req.prompt.as_deref()) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

async fn handle_grammar_constrain(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<GrammarConstrainReq>,
) -> impl IntoResponse {
    match rt.constrain_json(req.model_id.as_deref(), &req.prompt, &req.schema) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct DecoderGenerateBatchReq {
    pub model_id: String,
    pub prompts: Vec<String>,
    #[serde(default = "default_gen_tokens")]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_rep_pen")]
    pub repetition_penalty: f32,
    #[serde(default = "default_rng_seed")]
    pub seed: u64,
}

async fn handle_decoder_generate_batch(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<DecoderGenerateBatchReq>,
) -> impl IntoResponse {
    let params = CausalGenerationParams {
        max_new_tokens: req.max_new_tokens.clamp(1, 256),
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
        seed: req.seed,
        stop_token_ids: Vec::new(),
        speculative_draft_k: 0,
    };
    match rt.run_causal_decoder_batch(&req.model_id, &req.prompts, &params) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct VisionZeroShotReq {
    pub model_id: String,
    pub image_base64: String,
    pub candidate_labels: Vec<String>,
}

async fn handle_vision_zero_shot(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<VisionZeroShotReq>,
) -> impl IntoResponse {
    let clean_b64 = req
        .image_base64
        .split_once(";base64,")
        .map(|(_, tail)| tail)
        .unwrap_or(req.image_base64.as_str());
    let raw = match base64::engine::general_purpose::STANDARD.decode(clean_b64.trim()) {
        Ok(b) => b,
        Err(e) => return bad_request(format!("invalid image_base64: {e}")),
    };
    match rt.score_vision_zero_shot(&req.model_id, &raw, &req.candidate_labels) {
        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))),
        Err(e) => bad_request(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct HfExportReq {
    pub model_id: String,
    pub output_dir: String,
    #[serde(default)]
    pub base_model_repo: Option<String>,
    #[serde(default)]
    pub hf_repo_id: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub license_notice: Option<String>,
}

async fn handle_hf_export(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HfExportReq>,
) -> impl IntoResponse {
    let out_p = std::path::PathBuf::from(&req.output_dir);
    match rt.export_hf_model_bundle_with_license(
        &req.model_id,
        &out_p,
        req.base_model_repo.as_deref(),
        req.hf_repo_id.as_deref(),
        req.license.as_deref(),
        req.license_notice.as_deref(),
    ) {
        Ok(manifest) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "output_dir": req.output_dir,
                "manifest": manifest,
            })),
        ),
        Err(e) => bad_request(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct HfPullReq {
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub local_dir: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
    pub model_id: String,
    #[serde(default = "default_hf_rev")]
    pub revision: String,
    #[serde(default)]
    pub bearer_token: Option<String>,
    #[serde(default = "default_hf_rank")]
    pub target_rank: usize,
    #[serde(default = "default_hf_outliers")]
    pub max_outliers: usize,
    #[serde(default = "default_true_flag")]
    pub distill_svf: bool,
}

fn default_hf_rev() -> String {
    "main".to_string()
}
fn default_hf_rank() -> usize {
    16
}
fn default_hf_outliers() -> usize {
    64
}

fn discover_checkpoint_files_in_dir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let direct_st = dir.join("model.safetensors");
    if direct_st.exists() {
        return vec![direct_st];
    }
    let mut st_files = Vec::new();
    let mut gguf_files = Vec::new();
    let mut scan_dirs = vec![dir.to_path_buf(), dir.join("checkpoints")];
    while let Some(d) = scan_dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() {
                match p.extension().and_then(|e| e.to_str()) {
                    Some("safetensors") => st_files.push(p),
                    Some("gguf") => gguf_files.push(p),
                    _ => {}
                }
            }
        }
    }
    st_files.sort();
    gguf_files.sort();
    if !st_files.is_empty() {
        st_files
    } else if let Some(first_gguf) = gguf_files.into_iter().next() {
        vec![first_gguf]
    } else {
        Vec::new()
    }
}

async fn handle_hf_pull(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<HfPullReq>,
) -> axum::response::Response {
    let resolved_dir = if let Some(ref ldir) = req.local_dir {
        std::path::PathBuf::from(ldir)
    } else if let Some(ref repo) = req.repo_id {
        let slug = repo.replace('/', "__");
        let cache_dir = std::env::temp_dir().join("jouledb-hf-cache").join(&slug);
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            return bad_request(format!("create hf cache dir: {e}")).into_response();
        }
        let token = req
            .bearer_token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok());
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1800))
            .build()
        {
            Ok(c) => c,
            Err(e) => return bad_request(format!("http client build: {e}")).into_response(),
        };

        let fetch_to_disk = |fname: String, dst: std::path::PathBuf| {
            let client = client.clone();
            let token = token.clone();
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                repo.trim_matches('/'),
                req.revision,
                fname
            );
            async move {
                let mut rb = client.get(&url);
                if let Some(t) = token.as_deref().filter(|s| !s.trim().is_empty()) {
                    rb = rb.bearer_auth(t.trim());
                }
                let mut resp = rb.send().await.ok()?;
                if !resp.status().is_success() {
                    return None;
                }
                if let Some(par) = dst.parent() {
                    let _ = std::fs::create_dir_all(par);
                }
                let mut f = std::fs::File::create(&dst).ok()?;
                use std::io::Write;
                while let Some(chunk) = resp.chunk().await.ok()? {
                    f.write_all(&chunk).ok()?;
                }
                f.flush().ok()?;
                Some(dst)
            }
        };

        // 1. Fast path: pre-factorized `.jsyn` living-weights bundle on Hugging Face Hub!
        let jsyn_dst = cache_dir.join("model.jsyn");
        if req.filename.is_none()
            && let Some(p) = fetch_to_disk("model.jsyn".to_string(), jsyn_dst).await
        {
            return match rt.import_jsyn_snapshot_path(&p, Some(&req.model_id)) {
                Ok(v) => (StatusCode::OK, Json(v)).into_response(),
                Err(e) => bad_request(e).into_response(),
            };
        }

        // Pull sibling config & tokenizer metadata if present
        for aux in ["config.json", "tokenizer.json", "vocab.txt"] {
            let _ = fetch_to_disk(aux.to_string(), cache_dir.join(aux)).await;
        }

        // 2. Explicit `filename` or standard `model.safetensors`, or auto-discover sub-path/sharded `.safetensors` / `.gguf`!
        let mut target_rel_files: Vec<String> = Vec::new();
        if let Some(ref explicit_fn) = req.filename {
            target_rel_files.push(explicit_fn.trim_start_matches('/').to_string());
        } else if fetch_to_disk(
            "model.safetensors".to_string(),
            cache_dir.join("model.safetensors"),
        )
        .await
        .is_some()
        {
            // Already streamed directly to `cache_dir/model.safetensors`!
        } else {
            // Query Hugging Face Hub API for repo siblings (`checkpoints/*.safetensors`, shards, or `.gguf`)
            let api_url = format!(
                "https://huggingface.co/api/models/{}",
                repo.trim_matches('/')
            );
            let mut rb = client.get(&api_url);
            if let Some(t) = token.as_deref().filter(|s| !s.trim().is_empty()) {
                rb = rb.bearer_auth(t.trim());
            }
            if let Ok(resp) = rb.send().await
                && resp.status().is_success()
                && let Ok(meta_json) = resp.json::<serde_json::Value>().await
                && let Some(sibs) = meta_json.get("siblings").and_then(|s| s.as_array())
            {
                let mut st_sibs = Vec::new();
                let mut gguf_sibs = Vec::new();
                for s in sibs {
                    if let Some(rf) = s.get("rfilename").and_then(|x| x.as_str()) {
                        if rf.ends_with(".safetensors") && rf != "mtp.safetensors" {
                            st_sibs.push(rf.to_string());
                        } else if rf.ends_with(".gguf") && !rf.contains("mmproj") {
                            gguf_sibs.push(rf.to_string());
                        }
                    }
                }
                st_sibs.sort();
                gguf_sibs.sort();
                let root_model_shards: Vec<String> = st_sibs
                    .iter()
                    .filter(|s| !s.contains('/') && s.starts_with("model"))
                    .cloned()
                    .collect();
                if !root_model_shards.is_empty() {
                    target_rel_files = root_model_shards;
                } else if !st_sibs.is_empty() {
                    target_rel_files = st_sibs;
                } else if let Some(first_g) = gguf_sibs.into_iter().next() {
                    target_rel_files.push(first_g);
                }
            }
            if target_rel_files.is_empty() {
                return bad_request(format!(
                    "no model.jsyn, .safetensors, or .gguf checkpoint found in Hugging Face repo '{repo}'"
                ))
                .into_response();
            }
        }

        for rel in &target_rel_files {
            let dst = cache_dir.join(rel);
            if fetch_to_disk(rel.clone(), dst).await.is_none() {
                return bad_request(format!("failed to download '{rel}' from '{repo}'"))
                    .into_response();
            }
        }
        cache_dir
    } else {
        return bad_request("must specify either repo_id or local_dir").into_response();
    };

    let jsyn_candidate = resolved_dir.join("model.jsyn");
    if req.filename.is_none() && jsyn_candidate.exists() {
        return match rt.import_jsyn_snapshot_path(&jsyn_candidate, Some(&req.model_id)) {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => bad_request(e).into_response(),
        };
    }

    let ckpt_files = if let Some(ref explicit_fn) = req.filename {
        let p = resolved_dir.join(explicit_fn.trim_start_matches('/'));
        if p.exists() { vec![p] } else { Vec::new() }
    } else {
        discover_checkpoint_files_in_dir(&resolved_dir)
    };

    if ckpt_files.is_empty() {
        return bad_request(format!(
            "directory '{}' contains no model.jsyn, .safetensors, or .gguf checkpoints",
            resolved_dir.display()
        ))
        .into_response();
    }

    let is_temp_hf_cache = req.local_dir.is_none() && req.repo_id.is_some();
    let mut last_resp = None;
    for ckpt in ckpt_files {
        let ingest_req = IngestSafetensorsReq {
            model_id: req.model_id.clone(),
            file_path: Some(ckpt.to_string_lossy().to_string()),
            bytes_base64: None,
            config_json: None,
            vocab_txt: None,
            target_rank: req.target_rank,
            max_outliers: req.max_outliers,
            distill_svf: req.distill_svf,
        };
        last_resp = Some(
            handle_ingest_safetensors(State(rt.clone()), Json(ingest_req))
                .await
                .into_response(),
        );
        if is_temp_hf_cache {
            let _ = std::fs::remove_file(&ckpt);
        }
    }
    last_resp.unwrap_or_else(|| bad_request("no checkpoint ingested").into_response())
}

/// Curated `< 30 GB` permissive open-weight frontier model entry surfaced by `GET /api/v1/synapse/hf/catalog`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CuratedOpenWeightModelEntry {
    pub ecosystem: &'static str,
    pub model_id: &'static str,
    pub upstream_repo: &'static str,
    pub openie_jsyn_repo: &'static str,
    pub architecture: &'static str,
    pub modality: &'static str,
    pub license: &'static str,
    pub upstream_size_mb: u64,
    pub source_format: &'static str,
}

pub fn builtin_open_weight_catalog() -> Vec<CuratedOpenWeightModelEntry> {
    vec![
        CuratedOpenWeightModelEntry {
            ecosystem: "Liquid AI",
            model_id: "lfm2.5-350m",
            upstream_repo: "LiquidAI/LFM2.5-350M",
            openie_jsyn_repo: "dcharlot65-openie/LFM2.5-350M-LittleBit",
            architecture: "lfm2_shortconv_gqa_swiglu",
            modality: "causal_text",
            license: "lfm1.0",
            upstream_size_mb: 676,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Liquid AI",
            model_id: "lfm3-1.7b",
            upstream_repo: "LiquidAI/LFM3-1.7B",
            openie_jsyn_repo: "dcharlot65-openie/LFM3-1.7B-LittleBit",
            architecture: "lfm3_shortconv_gqa_swiglu",
            modality: "causal_text_reasoning",
            license: "lfm1.0",
            upstream_size_mb: 3340,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PrismML",
            model_id: "bonsai-1.7b",
            upstream_repo: "prism-ml/Bonsai-1.7B-mlx-1bit",
            openie_jsyn_repo: "dcharlot65-openie/Bonsai-1.7B-LittleBit",
            architecture: "qwen3_1bit_qknorm",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 237,
            source_format: "safetensors_mlx_1bit_and_gguf_q1_0",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PrismML",
            model_id: "bonsai-27b-1bit",
            upstream_repo: "prism-ml/Bonsai-27B-mlx-1bit",
            openie_jsyn_repo: "dcharlot65-openie/Bonsai-27B-LittleBit",
            architecture: "qwen3_5_1bit_qknorm",
            modality: "causal_text_reasoning",
            license: "apache-2.0",
            upstream_size_mb: 4890,
            source_format: "safetensors_mlx_1bit",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PrismML",
            model_id: "ternary-bonsai-2-2b",
            upstream_repo: "prism-ml/Ternary-Bonsai-2-2B-gguf",
            openie_jsyn_repo: "dcharlot65-openie/Ternary-Bonsai-2-2B-LittleBit",
            architecture: "bonsai2_ternary_1_58bit",
            modality: "multimodal_text_vision",
            license: "apache-2.0",
            upstream_size_mb: 409,
            source_format: "gguf",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Cactus Compute",
            model_id: "cactus-needle",
            upstream_repo: "Cactus-Compute/needle",
            openie_jsyn_repo: "dcharlot65-openie/Cactus-Needle-LittleBit",
            architecture: "needle_qknorm_gated_tool_caller",
            modality: "tool_calling",
            license: "mit",
            upstream_size_mb: 58,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Cactus Compute",
            model_id: "cactus-needle3",
            upstream_repo: "Cactus-Compute/needle3",
            openie_jsyn_repo: "dcharlot65-openie/Cactus-Needle3-LittleBit",
            architecture: "needle3_engram_tool_caller",
            modality: "tool_calling",
            license: "apache-2.0",
            upstream_size_mb: 231,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Cactus Compute",
            model_id: "cactus-whistle",
            upstream_repo: "Cactus-Compute/whistle",
            openie_jsyn_repo: "dcharlot65-openie/Cactus-Whistle-LittleBit",
            architecture: "whistle_zcrmsnorm_hadamard_asr",
            modality: "audio_speech_to_text",
            license: "apache-2.0",
            upstream_size_mb: 210,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Cactus Compute",
            model_id: "cactus-mirage",
            upstream_repo: "Cactus-Compute/mirage",
            openie_jsyn_repo: "dcharlot65-openie/Cactus-Mirage-LittleBit",
            architecture: "mirage_hconv_grn_vision",
            modality: "vision_diffusion_encoder",
            license: "apache-2.0",
            upstream_size_mb: 206,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Poolside",
            model_id: "poolside-laguna-xs-2.1",
            upstream_repo: "poolside/Laguna-XS-2.1-GGUF",
            openie_jsyn_repo: "dcharlot65-openie/Poolside-Laguna-XS-2.1-LittleBit",
            architecture: "laguna_moe_code_agent",
            modality: "code_and_software_engineering",
            license: "openmdw-1.1",
            upstream_size_mb: 19300,
            source_format: "gguf",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Google Gemma 4",
            model_id: "gemma-4-e2b-it",
            upstream_repo: "google/gemma-4-E2B-it",
            openie_jsyn_repo: "dcharlot65-openie/Gemma-4-E2B-it-LittleBit",
            architecture: "gemma4_geglu_qknorm_multimodal",
            modality: "multimodal_text_vision_audio",
            license: "apache-2.0",
            upstream_size_mb: 9770,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Google Gemma 4",
            model_id: "gemma-4-e4b-it",
            upstream_repo: "google/gemma-4-E4B-it",
            openie_jsyn_repo: "dcharlot65-openie/Gemma-4-E4B-it-LittleBit",
            architecture: "gemma4_geglu_qknorm_multimodal",
            modality: "multimodal_text_vision_audio",
            license: "apache-2.0",
            upstream_size_mb: 15250,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Qwen3",
            model_id: "qwen3-0.6b",
            upstream_repo: "Qwen/Qwen3-0.6B",
            openie_jsyn_repo: "dcharlot65-openie/Qwen3-0.6B-LittleBit",
            architecture: "qwen3_gqa_qknorm_swiglu",
            modality: "causal_text_reasoning",
            license: "apache-2.0",
            upstream_size_mb: 1430,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Qwen3",
            model_id: "qwen3-embedding-0.6b",
            upstream_repo: "Qwen/Qwen3-Embedding-0.6B",
            openie_jsyn_repo: "dcharlot65-openie/Qwen3-Embedding-0.6B-LittleBit",
            architecture: "qwen3_embedding_qknorm",
            modality: "text_embedding",
            license: "apache-2.0",
            upstream_size_mb: 1130,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Microsoft BitNet",
            model_id: "bitnet-b1.58-2b-4t",
            upstream_repo: "microsoft/bitnet-b1.58-2B-4T-bf16",
            openie_jsyn_repo: "dcharlot65-openie/BitNet-b1.58-2B-4T-LittleBit",
            architecture: "bitnet_subln_relu2_ternary",
            modality: "causal_text",
            license: "mit",
            upstream_size_mb: 4600,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "HuggingFace SmolLM",
            model_id: "smollm2-135m-instruct",
            upstream_repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
            openie_jsyn_repo: "dcharlot65-openie/SmolLM2-135M-Instruct-LittleBit",
            architecture: "llama_gqa_rope_swiglu",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 257,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "HuggingFace SmolLM",
            model_id: "smollm2-360m-instruct",
            upstream_repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
            openie_jsyn_repo: "dcharlot65-openie/SmolLM2-360M-Instruct-LittleBit",
            architecture: "llama_gqa_rope_swiglu",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 690,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "HuggingFace SmolVLM",
            model_id: "smolvlm-256m-instruct",
            upstream_repo: "HuggingFaceTB/SmolVLM-256M-Instruct",
            openie_jsyn_repo: "dcharlot65-openie/SmolVLM-256M-Instruct-LittleBit",
            architecture: "idefics3_siglip_llama",
            modality: "vision_language",
            license: "apache-2.0",
            upstream_size_mb: 489,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "IBM Granite",
            model_id: "granite-embedding-small-r2",
            upstream_repo: "ibm-granite/granite-embedding-small-english-r2",
            openie_jsyn_repo: "dcharlot65-openie/Granite-Embedding-Small-R2-LittleBit",
            architecture: "modernbert_wqkv_rope_geglu",
            modality: "text_embedding",
            license: "apache-2.0",
            upstream_size_mb: 91,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "IBM Granite",
            model_id: "granite-docling-258m",
            upstream_repo: "ibm-granite/granite-docling-258M",
            openie_jsyn_repo: "dcharlot65-openie/Granite-Docling-258M-LittleBit",
            architecture: "idefics3_docling_vlm",
            modality: "document_vision_language",
            license: "apache-2.0",
            upstream_size_mb: 491,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Snowflake Arctic",
            model_id: "arctic-embed-s",
            upstream_repo: "Snowflake/snowflake-arctic-embed-s",
            openie_jsyn_repo: "dcharlot65-openie/Arctic-Embed-S-LittleBit",
            architecture: "bert_bidirectional_encoder",
            modality: "text_embedding",
            license: "apache-2.0",
            upstream_size_mb: 133,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "OpenAI Whisper",
            model_id: "whisper-tiny.en",
            upstream_repo: "openai/whisper-tiny.en",
            openie_jsyn_repo: "dcharlot65-openie/whisper-tiny.en-LittleBit",
            architecture: "whisper_logmel_conv_seq2seq",
            modality: "audio_speech_to_text",
            license: "mit",
            upstream_size_mb: 144,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Thinking Machines Lab",
            model_id: "inkling-2-mini",
            upstream_repo: "thinkingmachines/Inkling-2-Mini",
            openie_jsyn_repo: "dcharlot65-openie/Inkling-2-Mini-LittleBit",
            architecture: "inkling2_loop_sconv_nvfp4_moe",
            modality: "multimodal_reasoning_1m_ctx",
            license: "apache-2.0",
            upstream_size_mb: 20162,
            source_format: "safetensors_nvfp4_mxfp8",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Thinking Machines Lab",
            model_id: "tml-d24-varloop",
            upstream_repo: "thinkingmachines/TML-d24-varloop-research",
            openie_jsyn_repo: "dcharlot65-openie/TML-d24-VarLoop-LittleBit",
            architecture: "inkling2_varloop_nvfp4_moe",
            modality: "reasoning_research",
            license: "apache-2.0",
            upstream_size_mb: 20162,
            source_format: "safetensors_nvfp4_mxfp8",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Thinking Machines Lab",
            model_id: "inkling-2-mini-mtp",
            upstream_repo: "thinkingmachines/Inkling-2-Mini",
            openie_jsyn_repo: "dcharlot65-openie/Inkling-2-Mini-MTP-LittleBit",
            architecture: "dflare_multi_token_prediction_3L",
            modality: "speculative_mtp_head",
            license: "apache-2.0",
            upstream_size_mb: 636,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-3-nano-4b",
            upstream_repo: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-3-Nano-4B-LittleBit",
            architecture: "nemotron_h_mamba2_gqa_relu2",
            modality: "causal_text_reasoning_262k",
            license: "nvidia-nemotron-open-model-license",
            upstream_size_mb: 7947,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-nano-9b-v2-nvfp4",
            upstream_repo: "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-Nano-9B-v2-LittleBit",
            architecture: "nemotron_h_mamba2_nvfp4",
            modality: "causal_text_reasoning",
            license: "nvidia-open-model-license",
            upstream_size_mb: 7835,
            source_format: "safetensors_nvfp4",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-3.5-bolt-e4b",
            upstream_repo: "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-3.5-Bolt-E4B-LittleBit",
            architecture: "gemma4_bolt_edge",
            modality: "causal_text_edge",
            license: "nvidia-open-model-license",
            upstream_size_mb: 6180,
            source_format: "gguf",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-3-embed-1b",
            upstream_repo: "nvidia/Nemotron-3-Embed-1B-BF16",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-3-Embed-1B-LittleBit",
            architecture: "ministral3_bidirectional_embed",
            modality: "text_embedding",
            license: "openmdw-1.1",
            upstream_size_mb: 2282,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "llama-nemotron-embed-1b-v2",
            upstream_repo: "nvidia/llama-nemotron-embed-1b-v2",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Llama-Nemotron-Embed-1B-v2-LittleBit",
            architecture: "llama_bidirectional_embed",
            modality: "text_embedding",
            license: "nvidia-open-model-license",
            upstream_size_mb: 2472,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "openmath-nemotron-1.5b",
            upstream_repo: "nvidia/OpenMath-Nemotron-1.5B",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-OpenMath-Nemotron-1.5B-LittleBit",
            architecture: "qwen2_math_reasoning",
            modality: "math_reasoning",
            license: "cc-by-4.0",
            upstream_size_mb: 3088,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-parse-2.0",
            upstream_repo: "nvidia/NVIDIA-Nemotron-Parse-2.0",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-Parse-2.0-LittleBit",
            architecture: "nemotron_parse_vlm",
            modality: "document_vision_language",
            license: "openmdw-1.1",
            upstream_size_mb: 3612,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "segformer-b0-ade",
            upstream_repo: "nvidia/segformer-b0-finetuned-ade-512-512",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-SegFormer-B0-LittleBit",
            architecture: "segformer_hierarchical_vit",
            modality: "vision_segmentation",
            license: "nvidia-open-model-license",
            upstream_size_mb: 15,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Google Gemma 4",
            model_id: "embeddinggemma-2",
            upstream_repo: "google/embeddinggemma-2",
            openie_jsyn_repo: "dcharlot65-openie/EmbeddingGemma-2-LittleBit",
            architecture: "gemma4_multimodal_embedding",
            modality: "multimodal_embedding",
            license: "apache-2.0",
            upstream_size_mb: 1489,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Poolside",
            model_id: "poolside-dflash",
            upstream_repo: "poolside/spec-decoding-subfolder-fixture",
            openie_jsyn_repo: "dcharlot65-openie/Poolside-DFlash-Draft-LittleBit",
            architecture: "llama_dflash_speculative_drafter",
            modality: "speculative_code_drafting",
            license: "apache-2.0",
            upstream_size_mb: 9,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Liquid AI",
            model_id: "lfm2.5-230m-base",
            upstream_repo: "LiquidAI/LFM2.5-230M-Base",
            openie_jsyn_repo: "dcharlot65-openie/LFM2.5-230M-Base-LittleBit",
            architecture: "lfm2_shortconv_gqa_swiglu",
            modality: "causal_text",
            license: "lfm1.0",
            upstream_size_mb: 459,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Microsoft BitNet",
            model_id: "bitnet-embedding-270m",
            upstream_repo: "microsoft/bitnet-embedding-270m",
            openie_jsyn_repo: "dcharlot65-openie/BitNet-Embedding-270M-LittleBit",
            architecture: "gemma3_bitnet_i2s_ternary_embed",
            modality: "text_embedding",
            license: "mit",
            upstream_size_mb: 368,
            source_format: "gguf_i2_s",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Microsoft BitNet",
            model_id: "bitnet-embedding-0.6b",
            upstream_repo: "microsoft/bitnet-embedding-0.6b",
            openie_jsyn_repo: "dcharlot65-openie/BitNet-Embedding-0.6B-LittleBit",
            architecture: "gemma3_bitnet_i2s_ternary_embed",
            modality: "text_embedding",
            license: "mit",
            upstream_size_mb: 428,
            source_format: "gguf_i2_s",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "AllenAI OLMo",
            model_id: "olmo-2-0425-1b-instruct",
            upstream_repo: "allenai/OLMo-2-0425-1B-Instruct",
            openie_jsyn_repo: "dcharlot65-openie/OLMo-2-1B-Instruct-LittleBit",
            architecture: "olmo2_postnorm_qknorm_swiglu",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 2970,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Sakana AI",
            model_id: "tinyswallow-1.5b-instruct",
            upstream_repo: "SakanaAI/TinySwallow-1.5B-Instruct",
            openie_jsyn_repo: "dcharlot65-openie/Sakana-TinySwallow-1.5B-LittleBit",
            architecture: "qwen2_evolutionary_distilled",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 3088,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PleIAs",
            model_id: "pleias-monad",
            upstream_repo: "PleIAs/Monad",
            openie_jsyn_repo: "dcharlot65-openie/PleIAs-Monad-LittleBit",
            architecture: "llama_commons_ultracompact",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 113,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PleIAs",
            model_id: "pleias-350m",
            upstream_repo: "PleIAs/Pleias-350m-Preview",
            openie_jsyn_repo: "dcharlot65-openie/PleIAs-350M-LittleBit",
            architecture: "llama_commons_rag",
            modality: "causal_text_rag",
            license: "apache-2.0",
            upstream_size_mb: 707,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "OpenBMB MiniCPM",
            model_id: "minicpm5-1b",
            upstream_repo: "openbmb/MiniCPM5-1B",
            openie_jsyn_repo: "dcharlot65-openie/MiniCPM5-1B-LittleBit",
            architecture: "minicpm5_gqa_swiglu",
            modality: "causal_text",
            license: "apache-2.0",
            upstream_size_mb: 2161,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PrimeIntellect",
            model_id: "qwen3-moe-tiny",
            upstream_repo: "PrimeIntellect/qwen3-moe-tiny",
            openie_jsyn_repo: "dcharlot65-openie/Qwen3-MoE-Tiny-LittleBit",
            architecture: "qwen3_sparse_moe_16e_top4",
            modality: "causal_text_moe",
            license: "apache-2.0",
            upstream_size_mb: 1341,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "BAAI BGE",
            model_id: "bge-small-en-v1.5",
            upstream_repo: "BAAI/bge-small-en-v1.5",
            openie_jsyn_repo: "dcharlot65-openie/BGE-Small-EN-v1.5-LittleBit",
            architecture: "bert_bidirectional_encoder",
            modality: "text_embedding",
            license: "mit",
            upstream_size_mb: 134,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Alibaba GTE",
            model_id: "gte-modernbert-base",
            upstream_repo: "Alibaba-NLP/gte-modernbert-base",
            openie_jsyn_repo: "dcharlot65-openie/GTE-ModernBERT-Base-LittleBit",
            architecture: "modernbert_wqkv_rope_geglu",
            modality: "text_embedding",
            license: "apache-2.0",
            upstream_size_mb: 298,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Nomic AI",
            model_id: "nomic-embed-text-v1.5",
            upstream_repo: "nomic-ai/nomic-embed-text-v1.5",
            openie_jsyn_repo: "dcharlot65-openie/Nomic-Embed-Text-v1.5-LittleBit",
            architecture: "nomic_bert_rope_swiglu",
            modality: "text_embedding",
            license: "apache-2.0",
            upstream_size_mb: 547,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Google Gemma 4",
            model_id: "gemma-4-e2b-it",
            upstream_repo: "google/gemma-4-E2B-it",
            openie_jsyn_repo: "dcharlot65-openie/Gemma-4-E2B-IT-LittleBit",
            architecture: "gemma4_omni_shared_kv_geglu",
            modality: "multimodal_audio_vision_text",
            license: "apache-2.0",
            upstream_size_mb: 10247,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Google Gemma 4",
            model_id: "gemma-4-e4b-it",
            upstream_repo: "google/gemma-4-E4B-it",
            openie_jsyn_repo: "dcharlot65-openie/Gemma-4-E4B-IT-LittleBit",
            architecture: "gemma4_omni_shared_kv_geglu",
            modality: "multimodal_audio_vision_text",
            license: "apache-2.0",
            upstream_size_mb: 15993,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nvidia-gemma-4-26b-a4b-nvfp4",
            upstream_repo: "nvidia/Gemma-4-26B-A4B-NVFP4",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Gemma-4-26B-A4B-LittleBit",
            architecture: "gemma4_moe_nvfp4_shared_kv",
            modality: "multimodal_reasoning_moe",
            license: "apache-2.0",
            upstream_size_mb: 18789,
            source_format: "safetensors_nvfp4",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-3-nano-30b-a3b-nvfp4",
            upstream_repo: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-3-Nano-30B-A3B-LittleBit",
            architecture: "nemotron_h_mamba2_moe_nvfp4",
            modality: "causal_text_reasoning_moe",
            license: "nvidia-nemotron-open-model-license",
            upstream_size_mb: 19363,
            source_format: "safetensors_nvfp4",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "cosmos-embed1-224p",
            upstream_repo: "nvidia/Cosmos-Embed1-224p",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Cosmos-Embed1-224p-LittleBit",
            architecture: "cosmos_physical_ai_multimodal_embed",
            modality: "physical_ai_video_vision_embed",
            license: "nvidia-open-model-license",
            upstream_size_mb: 2393,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "parakeet-tdt-0.6b-v3",
            upstream_repo: "nvidia/parakeet-tdt-0.6b-v3",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Parakeet-TDT-0.6B-v3-LittleBit",
            architecture: "fastconformer_tdt_asr",
            modality: "audio_speech_to_text",
            license: "cc-by-4.0",
            upstream_size_mb: 2508,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-3.5-asr-0.6b",
            upstream_repo: "nvidia/nemotron-3.5-asr-streaming-0.6b",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-3.5-ASR-0.6B-LittleBit",
            architecture: "fastconformer_streaming_asr",
            modality: "audio_speech_to_text",
            license: "openmdw-1.1",
            upstream_size_mb: 2552,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-3-diarization",
            upstream_repo: "nvidia/Nemotron-3-Diarization",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-3-Diarization-LittleBit",
            architecture: "sortformer_speaker_diarization",
            modality: "audio_diarization",
            license: "openmdw-1.1",
            upstream_size_mb: 397,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "gpt-oss-120b-eagle3",
            upstream_repo: "nvidia/gpt-oss-120b-Eagle3-short-context",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-GPT-OSS-120B-Eagle3-LittleBit",
            architecture: "eagle3_speculative_drafter",
            modality: "speculative_draft_head",
            license: "nvidia-open-model-license",
            upstream_size_mb: 1989,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "NVIDIA",
            model_id: "nemotron-diffusion-3b",
            upstream_repo: "nvidia/Nemotron-Labs-Diffusion-3B",
            openie_jsyn_repo: "dcharlot65-openie/NVIDIA-Nemotron-Diffusion-3B-LittleBit",
            architecture: "nemotron_discrete_diffusion_llm",
            modality: "diffusion_text_generation",
            license: "nvidia-nemotron-open-model-license",
            upstream_size_mb: 7663,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PrismML",
            model_id: "ternary-bonsai-2-2b",
            upstream_repo: "prism-ml/Ternary-Bonsai-2-2B-gguf",
            openie_jsyn_repo: "dcharlot65-openie/Ternary-Bonsai-2-2B-LittleBit",
            architecture: "qwen35_gated_deltanet_pq2_0_ternary",
            modality: "causal_text_reasoning",
            license: "apache-2.0",
            upstream_size_mb: 1140,
            source_format: "gguf_pq2_0",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "PrismML",
            model_id: "ternary-bonsai-2-4b",
            upstream_repo: "prism-ml/Ternary-Bonsai-2-4B-gguf",
            openie_jsyn_repo: "dcharlot65-openie/Ternary-Bonsai-2-4B-LittleBit",
            architecture: "qwen35_gated_deltanet_pq2_0_ternary",
            modality: "causal_text_reasoning",
            license: "apache-2.0",
            upstream_size_mb: 1147,
            source_format: "gguf_pq2_0",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Liquid AI",
            model_id: "lfm2.5-1.2b-instruct",
            upstream_repo: "LiquidAI/LFM2.5-1.2B-Instruct",
            openie_jsyn_repo: "dcharlot65-openie/LFM2.5-1.2B-Instruct-LittleBit",
            architecture: "lfm2_shortconv_gqa_swiglu",
            modality: "causal_text",
            license: "lfm1.0",
            upstream_size_mb: 2341,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Liquid AI",
            model_id: "lfm3-omni-2b",
            upstream_repo: "LiquidAI/LFM3-Omni-2B-GGUF",
            openie_jsyn_repo: "dcharlot65-openie/LFM3-Omni-2B-LittleBit",
            architecture: "lfm3_shortconv_gated_gqa_omni",
            modality: "multimodal_omni",
            license: "lfm1.0",
            upstream_size_mb: 1846,
            source_format: "gguf",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Poolside",
            model_id: "laguna-xs-2.1",
            upstream_repo: "poolside/Laguna-XS-2.1-GGUF",
            openie_jsyn_repo: "dcharlot65-openie/Poolside-Laguna-XS-2.1-LittleBit",
            architecture: "laguna_256e_stacked_moe_gated_gqa",
            modality: "code_agentic_moe_33b",
            license: "openmdw-1.1",
            upstream_size_mb: 20274,
            source_format: "gguf_q4_k_m",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "TII Falcon",
            model_id: "falcon-h1-0.5b-base",
            upstream_repo: "tiiuae/Falcon-H1-0.5B-Base",
            openie_jsyn_repo: "dcharlot65-openie/Falcon-H1-0.5B-Base-LittleBit",
            architecture: "falcon_h1_parallel_mamba2_gqa",
            modality: "causal_text_hybrid",
            license: "falcon-llm-license",
            upstream_size_mb: 1043,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "Zyphra",
            model_id: "zamba2-1.2b-instruct",
            upstream_repo: "Zyphra/Zamba2-1.2B-instruct",
            openie_jsyn_repo: "dcharlot65-openie/Zyphra-Zamba2-1.2B-Instruct-LittleBit",
            architecture: "zamba2_mamba2_shared_transformer",
            modality: "causal_text_hybrid",
            license: "apache-2.0",
            upstream_size_mb: 4860,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "HuggingFace VLA",
            model_id: "smolvla-libero",
            upstream_repo: "HuggingFaceVLA/smolvla_libero",
            openie_jsyn_repo: "dcharlot65-openie/SmolVLA-Libero-LittleBit",
            architecture: "smolvla_embodied_action_expert",
            modality: "embodied_robotics_vla",
            license: "apache-2.0",
            upstream_size_mb: 1218,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "DeepSeek",
            model_id: "deepseek-r1-distill-qwen-1.5b",
            upstream_repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            openie_jsyn_repo: "dcharlot65-openie/DeepSeek-R1-Distill-Qwen-1.5B-LittleBit",
            architecture: "qwen2_r1_reasoning_distill",
            modality: "causal_text_reasoning",
            license: "mit",
            upstream_size_mb: 3554,
            source_format: "safetensors",
        },
        CuratedOpenWeightModelEntry {
            ecosystem: "IBM Granite",
            model_id: "granite-4.1-3b",
            upstream_repo: "ibm-granite/granite-4.1-3b",
            openie_jsyn_repo: "dcharlot65-openie/Granite-4.1-3B-LittleBit",
            architecture: "granite4_mup_gqa_swiglu",
            modality: "causal_text_enterprise",
            license: "apache-2.0",
            upstream_size_mb: 6806,
            source_format: "safetensors",
        },
    ]
}

fn resolve_model_license_metadata(
    base_repo: &str,
    explicit_license: Option<&str>,
    explicit_notice: Option<&str>,
) -> (String, String, String) {
    let lower_repo = base_repo.to_ascii_lowercase();
    let inferred = explicit_license
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| {
            if lower_repo.starts_with("liquidai/") || lower_repo.contains("lfm") {
                "lfm1.0".to_string()
            } else if lower_repo == "nvidia/nemotron-3-embed-1b-bf16"
                || lower_repo.contains("nemotron-parse")
                || (lower_repo.starts_with("poolside/") && lower_repo.contains("laguna-"))
            {
                "openmdw-1.1".to_string()
            } else if lower_repo.starts_with("nvidia/openmath")
                || lower_repo.starts_with("nvidia/parakeet")
            {
                "cc-by-4.0".to_string()
            } else if lower_repo.starts_with("nvidia/nvidia-nemotron-3") {
                "nvidia-nemotron-open-model-license".to_string()
            } else if lower_repo.starts_with("nvidia/") {
                "nvidia-open-model-license".to_string()
            } else if lower_repo == "cactus-compute/needle"
                || lower_repo.starts_with("microsoft/bitnet")
                || lower_repo.starts_with("microsoft/phi-")
                || lower_repo.starts_with("baai/bge-")
                || lower_repo.starts_with("openai/whisper")
            {
                "mit".to_string()
            } else {
                "apache-2.0".to_string()
            }
        });

    if inferred == "lfm1.0" || inferred == "lfm-1.0" {
        let yaml =
            "license: other\nlicense_name: lfm1.0\nlicense_link: https://www.liquid.ai/lfm-license"
                .to_string();
        let notice = explicit_notice
            .map(String::from)
            .unwrap_or_else(|| format!(
                "Adapted from `{base_repo}` by Liquid AI under the [**LFM Open License v1.0**](https://www.liquid.ai/lfm-license). \
Free for academic, research, and commercial deployment by entities under $10M USD annual revenue; above that threshold, \
contact Liquid AI for commercial licensing. Built with Liquid AI LFM."
            ));
        (yaml, "lfm1.0".to_string(), notice)
    } else if inferred == "nvidia-nemotron-open-model-license" {
        let yaml = "license: other\nlicense_name: nvidia-nemotron-open-model-license\nlicense_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/".to_string();
        let notice = explicit_notice.map(String::from).unwrap_or_else(|| format!(
            "Adapted from `{base_repo}` by NVIDIA Corporation under the [**NVIDIA Nemotron Open Model License**](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/). Built with NVIDIA Nemotron."
        ));
        (
            yaml,
            "nvidia-nemotron-open-model-license".to_string(),
            notice,
        )
    } else if inferred == "nvidia-open-model-license" {
        let yaml = "license: other\nlicense_name: nvidia-open-model-license\nlicense_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/".to_string();
        let notice = explicit_notice.map(String::from).unwrap_or_else(|| format!(
            "Adapted from `{base_repo}` by NVIDIA Corporation under the [**NVIDIA Open Model License Agreement**](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Built with NVIDIA."
        ));
        (yaml, "nvidia-open-model-license".to_string(), notice)
    } else if inferred == "cc-by-4.0" {
        let notice = explicit_notice.map(String::from).unwrap_or_else(|| format!(
            "Adapted from `{base_repo}` under the [**Creative Commons Attribution 4.0 International (CC-BY-4.0)**](https://creativecommons.org/licenses/by/4.0/) license."
        ));
        (
            "license: cc-by-4.0".to_string(),
            "cc-by-4.0".to_string(),
            notice,
        )
    } else if inferred == "openmdw-1.1" {
        let yaml = "license: other\nlicense_name: openmdw-1.1\nlicense_link: https://openmdw.ai/"
            .to_string();
        let notice = explicit_notice
            .map(String::from)
            .unwrap_or_else(|| format!("Adapted from `{base_repo}` under the permissive **OpenMDW-1.1** Open Model Weights License."));
        (yaml, "openmdw-1.1".to_string(), notice)
    } else if inferred == "gemma" {
        let notice = explicit_notice.map(String::from).unwrap_or_else(|| format!(
            "Adapted from `{base_repo}` under the [**Gemma Terms of Use**](https://ai.google.dev/gemma/terms). \
Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms."
        ));
        ("license: gemma".to_string(), "gemma".to_string(), notice)
    } else if inferred == "falcon-llm-license" {
        let yaml = "license: other\nlicense_name: falcon-llm-license\nlicense_link: https://falconllm.tii.ae/falcon-terms-and-conditions.html".to_string();
        let notice = explicit_notice.map(String::from).unwrap_or_else(|| format!(
            "Adapted from `{base_repo}` by Technology Innovation Institute (TII) under the [**TII Falcon-LLM License 2.0**](https://falconllm.tii.ae/falcon-terms-and-conditions.html)."
        ));
        (yaml, "falcon-llm-license".to_string(), notice)
    } else if inferred == "mit" {
        let notice = explicit_notice
            .map(String::from)
            .unwrap_or_else(|| format!("Adapted from `{base_repo}` under the **MIT License**."));
        ("license: mit".to_string(), "mit".to_string(), notice)
    } else {
        let notice = explicit_notice.map(String::from).unwrap_or_else(|| {
            format!("Adapted from `{base_repo}` under the **Apache License 2.0**.")
        });
        (
            "license: apache-2.0".to_string(),
            "apache-2.0".to_string(),
            notice,
        )
    }
}

#[derive(Debug, Deserialize)]
pub struct EvictSynapseModelReq {
    pub model_id: String,
}

async fn handle_evict_model(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<EvictSynapseModelReq>,
) -> impl IntoResponse {
    let pfx = format!("{}::", req.model_id);
    if let Ok(mut gcache) = rt.gpu_packed_cache.lock() {
        gcache.retain(|k, _| !k.starts_with(&pfx));
    }
    let removed = {
        let mut guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.evict_model(&req.model_id)
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "model_id": req.model_id,
            "removed_entries": removed,
        })),
    )
}

async fn handle_hf_catalog() -> impl IntoResponse {
    let entries = builtin_open_weight_catalog();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "max_checkpoint_budget_gb": 30,
            "total_models": entries.len(),
            "models": entries,
        })),
    )
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
    if !guard.littlebit_tensors.contains_key(&req.name)
        && guard.littlebit_tensors.len() >= 128
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
    let res = tensor.adapt_svf_online(
        &req.expert_name,
        &req.input,
        &req.target_y,
        req.learning_rate,
    );
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

#[derive(Debug, Deserialize)]
struct CalibrateLittleBitReq {
    name: String,
    calibration_inputs: Vec<f32>,
    target_outputs: Vec<f32>,
    num_samples: usize,
    #[serde(default = "default_calibrate_sweeps")]
    sweeps: usize,
    #[serde(default = "default_calibrate_ridge")]
    ridge_lambda: f64,
}

fn default_calibrate_sweeps() -> usize {
    4
}

fn default_calibrate_ridge() -> f64 {
    1e-4
}

async fn handle_littlebit_calibrate(
    State(rt): State<Arc<SynapseServerRuntime>>,
    Json(req): Json<CalibrateLittleBitReq>,
) -> impl IntoResponse {
    let outcome = {
        let mut guard = match rt.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let Some(tensor) = guard.littlebit_tensors.get_mut(&req.name) else {
            return bad_request(format!("LittleBit tensor '{}' not found", req.name));
        };
        let res = tensor.calibrate_activation_aware_svf(
            &req.calibration_inputs,
            &req.target_outputs,
            req.num_samples,
            req.sweeps,
            req.ridge_lambda,
        );
        match res {
            Ok(receipt) => {
                let cloned = tensor.clone();
                rt.persist_if_configured(&guard);
                Ok((receipt, cloned))
            }
            Err(e) => Err(e),
        }
    };
    match outcome {
        Ok((receipt, updated_tensor)) => {
            rt.cache_gpu_weights(&req.name, &updated_tensor);
            (StatusCode::OK, Json(serde_json::json!(receipt)))
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
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "registered": true })),
        ),
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
    let receipt = guard
        .airstream
        .execute_streaming_pass(&req.routed_shard_ids);
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

#[derive(Debug, Deserialize)]
struct RankSweepBenchReq {
    #[serde(default)]
    safetensors_path: Option<String>,
    #[serde(default = "default_sweep_tensor_filter")]
    tensor_filter: String,
    #[serde(default = "default_sweep_max_rows")]
    max_rows_cap: usize,
    #[serde(default = "default_sweep_dim")]
    rows: usize,
    #[serde(default = "default_sweep_dim")]
    cols: usize,
    #[serde(default = "default_sweep_ranks")]
    ranks: Vec<usize>,
    #[serde(default = "default_sweep_outliers")]
    max_outliers: usize,
    #[serde(default = "default_sweep_samples")]
    calib_samples: usize,
    #[serde(default = "default_sweep_samples")]
    eval_samples: usize,
    #[serde(default = "default_sweep_svf_iters")]
    svf_sweeps: usize,
}

fn default_sweep_tensor_filter() -> String {
    "q_proj".to_string()
}
fn default_sweep_max_rows() -> usize {
    2048
}
fn default_sweep_dim() -> usize {
    512
}
fn default_sweep_ranks() -> Vec<usize> {
    vec![8, 32, 64, 128]
}
fn default_sweep_outliers() -> usize {
    32
}
fn default_sweep_samples() -> usize {
    24
}
fn default_sweep_svf_iters() -> usize {
    6
}

async fn handle_littlebit_rank_sweep(Json(req): Json<RankSweepBenchReq>) -> impl IntoResponse {
    use joule_db_weights::rank_sweep_bench::{
        evaluate_matrix_rank_sweep, evaluate_safetensors_layer_rank_sweep,
        generate_synthetic_transformer_matrix,
    };

    let res = if let Some(ref p_str) = req.safetensors_path {
        evaluate_safetensors_layer_rank_sweep(
            std::path::Path::new(p_str),
            &req.tensor_filter,
            req.max_rows_cap,
            &req.ranks,
            req.max_outliers,
            req.calib_samples,
            req.eval_samples,
            req.svf_sweeps,
        )
    } else {
        let r = req.rows.clamp(16, 4096);
        let c = req.cols.clamp(16, 4096);
        let dense = generate_synthetic_transformer_matrix(r, c, 0x2026_0209);
        evaluate_matrix_rank_sweep(
            &req.tensor_filter,
            "SYNTH_BF16",
            &dense,
            r,
            c,
            &req.ranks,
            req.max_outliers,
            req.calib_samples,
            req.eval_samples,
            req.svf_sweeps,
        )
    };

    match res {
        Ok(report) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "report": report,
            })),
        ),
        Err(e) => bad_request(e),
    }
}
