//! LLM-specific runtime — first-class support for containerized LLM inference.
//!
//! The highest-value use case for energy-aware containers: every inference request
//! is metered with joules-per-token, enabling cost comparison across hardware and
//! model architectures.
//!
//! ```bash
//! jouledb run llama3:8b --gpu 0              # auto-detect GGUF/MLX backend
//! jouledb run mistral:7b --npu               # route to Neural Engine
//! jouledb run deepseek-r1:671b --gpu 0,1,2,3 # multi-GPU
//! jouledb models                             # list cached models
//! ```
//!
//! Energy-per-token model:
//! ```text
//! Energy(J) = tokens × FLOPs_per_token × J_per_FLOP × PUE
//! FLOPs_per_token ≈ 2 × params
//! ```

use crate::{AcceleratorKind, InstanceId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::Instant;

/// LLM inference backend type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmBackend {
    /// GGUF format via llama.cpp — CPU/GPU/Metal.
    GGUF,
    /// vLLM — high-throughput GPU serving (PagedAttention).
    VLLM,
    /// TensorRT-LLM — NVIDIA-optimized inference.
    TensorRT,
    /// MLX — Apple Silicon optimized (unified memory).
    MLX,
    /// ONNX Runtime — cross-platform inference.
    ONNX,
    /// Custom backend.
    Custom(String),
}

impl std::fmt::Display for LlmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GGUF => write!(f, "gguf"),
            Self::VLLM => write!(f, "vllm"),
            Self::TensorRT => write!(f, "tensorrt"),
            Self::MLX => write!(f, "mlx"),
            Self::ONNX => write!(f, "onnx"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Known model profile with parameter count and hardware characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Model identifier (e.g. `"llama3-8b"`, `"mistral-7b"`).
    pub model_id: String,
    /// Model family (e.g. `"llama"`, `"mistral"`, `"deepseek"`).
    pub family: String,
    /// Parameter count in billions.
    pub params_b: f64,
    /// Default context length.
    pub context_length: u32,
    /// Preferred backend for this model.
    pub preferred_backend: LlmBackend,
    /// Preferred accelerator kind.
    pub preferred_accelerator: AcceleratorKind,
    /// Minimum VRAM in MB (0 for CPU-only models).
    pub min_vram_mb: u64,
}

/// Running LLM instance with inference telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmInstance {
    /// Instance ID of the underlying container/process.
    pub instance_id: String,
    /// Model being served.
    pub model_id: String,
    /// Inference backend.
    pub backend: LlmBackend,
    /// Accelerator type in use.
    pub accelerator: AcceleratorKind,
    /// Context length configured.
    pub context_length: u32,
    /// Parameter count in billions.
    pub params_b: f64,
}

/// Inference telemetry for a running LLM.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceTelemetry {
    /// Total tokens generated since start.
    pub total_tokens: u64,
    /// Total inference requests served.
    pub total_requests: u64,
    /// Total energy consumed for inference in joules.
    pub total_energy_joules: f64,
    /// Current throughput: tokens per second (exponential moving average).
    pub tokens_per_second: f64,
    /// Energy per token in joules (running average).
    pub joules_per_token: f64,
    /// Energy cost per 1M tokens at grid rate (USD).
    pub cost_per_million_tokens_usd: f64,
}

/// Manages LLM model profiles, running instances, and inference telemetry.
pub struct LlmRuntime {
    /// Directory for cached model weights.
    model_cache: PathBuf,
    /// Known model profiles (model_id → profile).
    model_registry: HashMap<String, ModelProfile>,
    /// Active LLM instances (instance_id → LlmInstance).
    active_models: RwLock<HashMap<String, LlmInstance>>,
    /// Inference telemetry per instance.
    telemetry: RwLock<HashMap<String, InferenceTelemetry>>,
    /// Grid electricity rate (USD per kWh), default 0.12.
    grid_rate_usd_per_kwh: f64,
}

impl LlmRuntime {
    /// Create a new LLM runtime with default model registry.
    pub fn new(model_cache: PathBuf) -> Self {
        std::fs::create_dir_all(&model_cache).ok();
        Self {
            model_cache,
            model_registry: build_default_registry(),
            active_models: RwLock::new(HashMap::new()),
            telemetry: RwLock::new(HashMap::new()),
            grid_rate_usd_per_kwh: 0.12,
        }
    }

    /// Set the grid electricity rate for cost calculations.
    pub fn set_grid_rate(&mut self, usd_per_kwh: f64) {
        self.grid_rate_usd_per_kwh = usd_per_kwh;
    }

    /// Get a model profile by ID.
    pub fn get_profile(&self, model_id: &str) -> Option<&ModelProfile> {
        // Try exact match first
        if let Some(p) = self.model_registry.get(model_id) {
            return Some(p);
        }
        // Try without tag (e.g. "llama3:8b" → "llama3-8b")
        let normalized = model_id.replace(':', "-");
        self.model_registry.get(&normalized)
    }

    /// List all known model profiles.
    pub fn list_profiles(&self) -> Vec<&ModelProfile> {
        self.model_registry.values().collect()
    }

    /// Register an LLM instance (called after `start_workload` for an LLM container).
    pub fn register_instance(&self, instance: LlmInstance) {
        let id = instance.instance_id.clone();
        self.active_models
            .write()
            .unwrap()
            .insert(id.clone(), instance);
        self.telemetry
            .write()
            .unwrap()
            .insert(id, InferenceTelemetry::default());
    }

    /// Deregister an LLM instance (called on stop).
    pub fn deregister_instance(&self, instance_id: &str) {
        self.active_models.write().unwrap().remove(instance_id);
        self.telemetry.write().unwrap().remove(instance_id);
    }

    /// Get the LLM instance if this instance ID is running an LLM.
    pub fn get_instance(&self, instance_id: &str) -> Option<LlmInstance> {
        self.active_models.read().unwrap().get(instance_id).cloned()
    }

    /// List all active LLM instances.
    pub fn list_instances(&self) -> Vec<LlmInstance> {
        self.active_models
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Record completed inference and update telemetry.
    ///
    /// Called by the energy sidecar or inference proxy after each request.
    pub fn record_inference(
        &self,
        instance_id: &str,
        tokens_generated: u64,
        energy_joules: f64,
        duration_secs: f64,
    ) {
        let mut telemetry = self.telemetry.write().unwrap();
        if let Some(t) = telemetry.get_mut(instance_id) {
            t.total_tokens += tokens_generated;
            t.total_requests += 1;
            t.total_energy_joules += energy_joules;

            // Exponential moving average for tokens/sec (alpha = 0.1)
            let instant_tps = if duration_secs > 0.0 {
                tokens_generated as f64 / duration_secs
            } else {
                0.0
            };
            t.tokens_per_second = 0.9 * t.tokens_per_second + 0.1 * instant_tps;

            // Running average for joules/token
            if t.total_tokens > 0 {
                t.joules_per_token = t.total_energy_joules / t.total_tokens as f64;
            }

            // Cost per 1M tokens: J/token × 1M / 3600 / 1000 × rate
            // (convert joules to kWh, then multiply by rate)
            t.cost_per_million_tokens_usd =
                t.joules_per_token * 1_000_000.0 / 3_600_000.0 * self.grid_rate_usd_per_kwh;
        }
    }

    /// Estimate energy per token for a model on a given accelerator.
    ///
    /// Uses the formula: `Energy(J) = 2 × params_B × 1e9 × J_per_FLOP × PUE`
    /// where J_per_FLOP depends on the hardware.
    pub fn estimate_joules_per_token(
        &self,
        model_id: &str,
        accelerator: &AcceleratorKind,
    ) -> Option<f64> {
        let profile = self.get_profile(model_id)?;
        let j_per_flop = joules_per_flop(accelerator);
        let pue = power_usage_effectiveness(accelerator);
        let flops_per_token = 2.0 * profile.params_b * 1e9;
        Some(flops_per_token * j_per_flop * pue)
    }

    /// Get inference telemetry for an instance.
    pub fn get_telemetry(&self, instance_id: &str) -> Option<InferenceTelemetry> {
        self.telemetry.read().unwrap().get(instance_id).cloned()
    }

    /// Get telemetry for all active LLM instances.
    pub fn all_telemetry(&self) -> HashMap<String, InferenceTelemetry> {
        self.telemetry.read().unwrap().clone()
    }

    /// Get the model cache directory path.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.model_cache
    }

    /// Detect the best backend for a model on this platform.
    pub fn detect_backend(&self, model_id: &str) -> LlmBackend {
        if let Some(profile) = self.get_profile(model_id) {
            return profile.preferred_backend.clone();
        }

        // Platform-based heuristic
        let platform = joule_db_energy::detect_platform();
        if platform.cpu_brand.contains("Apple") {
            LlmBackend::MLX
        } else if platform.gpu_available {
            LlmBackend::GGUF // GGUF with CUDA is the most portable GPU option
        } else {
            LlmBackend::GGUF // CPU fallback
        }
    }

    /// Detect the best accelerator for a model on this platform.
    pub fn detect_accelerator(&self, model_id: &str) -> AcceleratorKind {
        if let Some(profile) = self.get_profile(model_id) {
            return profile.preferred_accelerator.clone();
        }

        let platform = joule_db_energy::detect_platform();
        if platform.gpu_available {
            AcceleratorKind::GPU
        } else if platform.npu_available {
            AcceleratorKind::NPU
        } else {
            AcceleratorKind::GPU // Fallback, will use CPU in practice
        }
    }
}

/// Joules per floating-point operation for different accelerator types.
///
/// These are approximate values based on published TDP and TFLOPS specs.
fn joules_per_flop(accelerator: &AcceleratorKind) -> f64 {
    match accelerator {
        // NVIDIA A100: 312 TFLOPS at 300W → ~1e-12 J/FLOP
        // Consumer GPUs are less efficient (~2-5e-12)
        AcceleratorKind::GPU => 2e-12,
        // Apple Neural Engine: ~15.8 TOPS at ~8W → ~5e-13 J/OP
        AcceleratorKind::NPU => 5e-13,
        // Google TPU v4: 275 TFLOPS at 170W → ~6e-13 J/FLOP
        AcceleratorKind::TPU => 6e-13,
        // Groq LPU: 750 TOPS at 300W → ~4e-13 J/OP
        AcceleratorKind::LPU => 4e-13,
        _ => 3e-12, // Conservative default
    }
}

/// Power Usage Effectiveness multiplier for different deployment types.
fn power_usage_effectiveness(accelerator: &AcceleratorKind) -> f64 {
    match accelerator {
        // Local devices: PUE ≈ 1.0 (no datacenter overhead)
        AcceleratorKind::GPU | AcceleratorKind::NPU => 1.0,
        // Cloud/datacenter accelerators: PUE ≈ 1.1-1.4
        AcceleratorKind::TPU => 1.1,
        AcceleratorKind::LPU => 1.1,
        _ => 1.2,
    }
}

/// Build the default model registry with known model profiles.
fn build_default_registry() -> HashMap<String, ModelProfile> {
    let profiles = vec![
        // Llama family
        ModelProfile {
            model_id: "llama3-8b".into(),
            family: "llama".into(),
            params_b: 8.0,
            context_length: 8192,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 6144,
        },
        ModelProfile {
            model_id: "llama3-70b".into(),
            family: "llama".into(),
            params_b: 70.0,
            context_length: 8192,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 40960,
        },
        ModelProfile {
            model_id: "llama3.1-405b".into(),
            family: "llama".into(),
            params_b: 405.0,
            context_length: 131072,
            preferred_backend: LlmBackend::VLLM,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 245760,
        },
        // Mistral family
        ModelProfile {
            model_id: "mistral-7b".into(),
            family: "mistral".into(),
            params_b: 7.0,
            context_length: 32768,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 5120,
        },
        ModelProfile {
            model_id: "mixtral-8x7b".into(),
            family: "mistral".into(),
            params_b: 46.7,
            context_length: 32768,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 26624,
        },
        // Gemma family
        ModelProfile {
            model_id: "gemma2-9b".into(),
            family: "gemma".into(),
            params_b: 9.0,
            context_length: 8192,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 6144,
        },
        ModelProfile {
            model_id: "gemma2-27b".into(),
            family: "gemma".into(),
            params_b: 27.0,
            context_length: 8192,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 16384,
        },
        // Qwen family
        ModelProfile {
            model_id: "qwen2.5-7b".into(),
            family: "qwen".into(),
            params_b: 7.0,
            context_length: 131072,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 5120,
        },
        ModelProfile {
            model_id: "qwen2.5-72b".into(),
            family: "qwen".into(),
            params_b: 72.0,
            context_length: 131072,
            preferred_backend: LlmBackend::VLLM,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 40960,
        },
        // DeepSeek
        ModelProfile {
            model_id: "deepseek-r1-671b".into(),
            family: "deepseek".into(),
            params_b: 671.0,
            context_length: 65536,
            preferred_backend: LlmBackend::VLLM,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 409600,
        },
        ModelProfile {
            model_id: "deepseek-v3-685b".into(),
            family: "deepseek".into(),
            params_b: 685.0,
            context_length: 131072,
            preferred_backend: LlmBackend::VLLM,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 409600,
        },
        // Phi (small models)
        ModelProfile {
            model_id: "phi-3-mini".into(),
            family: "phi".into(),
            params_b: 3.8,
            context_length: 128000,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::NPU,
            min_vram_mb: 2048,
        },
        // Apple MLX-optimized
        ModelProfile {
            model_id: "llama3-8b-mlx".into(),
            family: "llama".into(),
            params_b: 8.0,
            context_length: 8192,
            preferred_backend: LlmBackend::MLX,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 0, // Unified memory
        },
    ];

    profiles
        .into_iter()
        .map(|p| (p.model_id.clone(), p))
        .collect()
}

/// Parse a model reference into (model_id, tag).
///
/// Examples:
/// - `"llama3:8b"` → `("llama3-8b", "8b")`
/// - `"mistral:latest"` → `("mistral-7b", "latest")` (with heuristic)
/// - `"deepseek-r1:671b"` → `("deepseek-r1-671b", "671b")`
pub fn parse_model_ref(input: &str) -> (String, Option<String>) {
    if let Some((name, tag)) = input.split_once(':') {
        let model_id = format!("{}-{}", name, tag);
        (model_id, Some(tag.to_string()))
    } else {
        (input.to_string(), None)
    }
}

/// Check if an input looks like an LLM model reference rather than a container image.
///
/// Heuristic: known model families without registry prefix.
pub fn is_llm_model_ref(input: &str) -> bool {
    let lower = input.to_lowercase();
    let base = lower.split(':').next().unwrap_or(&lower);

    let known_families = [
        "llama",
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "mistral",
        "mixtral",
        "gemma",
        "gemma2",
        "qwen",
        "qwen2",
        "qwen2.5",
        "deepseek",
        "deepseek-r1",
        "deepseek-v2",
        "deepseek-v3",
        "phi",
        "phi-3",
        "phi-4",
        "codellama",
        "code-llama",
        "whisper",
        "stable-diffusion",
        "sdxl",
        "falcon",
        "yi",
        "command-r",
        "solar",
    ];

    known_families
        .iter()
        .any(|f| base == *f || base.starts_with(&format!("{}-", f)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_backend_display() {
        assert_eq!(LlmBackend::GGUF.to_string(), "gguf");
        assert_eq!(LlmBackend::MLX.to_string(), "mlx");
        assert_eq!(LlmBackend::VLLM.to_string(), "vllm");
        assert_eq!(LlmBackend::TensorRT.to_string(), "tensorrt");
        assert_eq!(LlmBackend::Custom("triton".into()).to_string(), "triton");
    }

    #[test]
    fn test_llm_backend_serde() {
        for backend in [
            LlmBackend::GGUF,
            LlmBackend::MLX,
            LlmBackend::VLLM,
            LlmBackend::TensorRT,
        ] {
            let json = serde_json::to_string(&backend).unwrap();
            let parsed: LlmBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, backend);
        }
    }

    #[test]
    fn test_model_profile_registry() {
        let registry = build_default_registry();
        assert!(registry.len() >= 12, "expected at least 12 model profiles");

        let llama = registry.get("llama3-8b").unwrap();
        assert_eq!(llama.params_b, 8.0);
        assert_eq!(llama.preferred_backend, LlmBackend::GGUF);

        let deepseek = registry.get("deepseek-r1-671b").unwrap();
        assert_eq!(deepseek.params_b, 671.0);
        assert_eq!(deepseek.preferred_backend, LlmBackend::VLLM);
    }

    #[test]
    fn test_parse_model_ref() {
        let (id, tag) = parse_model_ref("llama3:8b");
        assert_eq!(id, "llama3-8b");
        assert_eq!(tag.as_deref(), Some("8b"));

        let (id, tag) = parse_model_ref("mistral");
        assert_eq!(id, "mistral");
        assert!(tag.is_none());

        let (id, tag) = parse_model_ref("deepseek-r1:671b");
        assert_eq!(id, "deepseek-r1-671b");
        assert_eq!(tag.as_deref(), Some("671b"));
    }

    #[test]
    fn test_is_llm_model_ref() {
        assert!(is_llm_model_ref("llama3:8b"));
        assert!(is_llm_model_ref("mistral:7b"));
        assert!(is_llm_model_ref("deepseek-r1:671b"));
        assert!(is_llm_model_ref("phi-3"));
        assert!(is_llm_model_ref("qwen2.5:72b"));
        assert!(is_llm_model_ref("gemma2:9b"));

        // Not LLM refs
        assert!(!is_llm_model_ref("nginx:latest"));
        assert!(!is_llm_model_ref("postgres"));
        assert!(!is_llm_model_ref("ghcr.io/user/app:v1"));
    }

    #[test]
    fn test_llm_runtime_new() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = LlmRuntime::new(tmp.path().join("models"));
        assert!(runtime.cache_dir().exists());
        assert!(runtime.list_instances().is_empty());
        assert!(runtime.list_profiles().len() >= 12);
    }

    #[test]
    fn test_llm_runtime_get_profile() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = LlmRuntime::new(tmp.path().join("models"));

        assert!(runtime.get_profile("llama3-8b").is_some());
        assert!(runtime.get_profile("llama3:8b").is_some()); // colon-normalized
        assert!(runtime.get_profile("nonexistent").is_none());
    }

    #[test]
    fn test_llm_runtime_register_deregister() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = LlmRuntime::new(tmp.path().join("models"));

        let instance = LlmInstance {
            instance_id: "test-001".into(),
            model_id: "llama3-8b".into(),
            backend: LlmBackend::GGUF,
            accelerator: AcceleratorKind::GPU,
            context_length: 8192,
            params_b: 8.0,
        };

        runtime.register_instance(instance.clone());
        assert_eq!(runtime.list_instances().len(), 1);
        assert!(runtime.get_instance("test-001").is_some());

        runtime.deregister_instance("test-001");
        assert!(runtime.list_instances().is_empty());
    }

    #[test]
    fn test_inference_telemetry() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = LlmRuntime::new(tmp.path().join("models"));

        let instance = LlmInstance {
            instance_id: "telem-001".into(),
            model_id: "llama3-8b".into(),
            backend: LlmBackend::GGUF,
            accelerator: AcceleratorKind::GPU,
            context_length: 8192,
            params_b: 8.0,
        };

        runtime.register_instance(instance);

        // Record some inference
        runtime.record_inference("telem-001", 100, 0.5, 2.0);
        let t = runtime.get_telemetry("telem-001").unwrap();
        assert_eq!(t.total_tokens, 100);
        assert_eq!(t.total_requests, 1);
        assert!((t.total_energy_joules - 0.5).abs() < 1e-10);
        assert!((t.joules_per_token - 0.005).abs() < 1e-10);
        assert!(t.tokens_per_second > 0.0);
        assert!(t.cost_per_million_tokens_usd > 0.0);

        // Record more
        runtime.record_inference("telem-001", 200, 1.0, 1.0);
        let t = runtime.get_telemetry("telem-001").unwrap();
        assert_eq!(t.total_tokens, 300);
        assert_eq!(t.total_requests, 2);
        assert!((t.total_energy_joules - 1.5).abs() < 1e-10);
        assert!((t.joules_per_token - 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_joules_per_token() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = LlmRuntime::new(tmp.path().join("models"));

        // Llama3-8B on GPU
        let gpu_j = runtime
            .estimate_joules_per_token("llama3-8b", &AcceleratorKind::GPU)
            .unwrap();
        // 2 × 8e9 × 2e-12 × 1.0 = 0.032 J/token
        assert!((gpu_j - 0.032).abs() < 0.001, "GPU J/token = {}", gpu_j);

        // Same model on NPU — should be cheaper
        let npu_j = runtime
            .estimate_joules_per_token("llama3-8b", &AcceleratorKind::NPU)
            .unwrap();
        assert!(
            npu_j < gpu_j,
            "NPU should be more efficient: {} vs {}",
            npu_j,
            gpu_j
        );

        // DeepSeek-R1-671B on GPU
        let big_j = runtime
            .estimate_joules_per_token("deepseek-r1-671b", &AcceleratorKind::GPU)
            .unwrap();
        // 2 × 671e9 × 2e-12 × 1.0 = 2.684 J/token
        assert!(big_j > 2.0, "671B model J/token should be > 2: {}", big_j);
        assert!(big_j > gpu_j * 50.0, "671B should be >50x 8B model");

        // Unknown model
        assert!(
            runtime
                .estimate_joules_per_token("nonexistent", &AcceleratorKind::GPU)
                .is_none()
        );
    }

    #[test]
    fn test_detect_backend() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = LlmRuntime::new(tmp.path().join("models"));

        // Known model: uses profile preference
        assert_eq!(runtime.detect_backend("llama3-8b"), LlmBackend::GGUF);
        assert_eq!(runtime.detect_backend("llama3.1-405b"), LlmBackend::VLLM);
        assert_eq!(runtime.detect_backend("llama3-8b-mlx"), LlmBackend::MLX);

        // Unknown model: platform heuristic
        let backend = runtime.detect_backend("unknown-model");
        // On Apple Silicon this would be MLX, otherwise GGUF
        assert!(
            matches!(backend, LlmBackend::MLX | LlmBackend::GGUF),
            "expected MLX or GGUF, got {:?}",
            backend
        );
    }

    #[test]
    fn test_model_profile_serde() {
        let profile = ModelProfile {
            model_id: "test-7b".into(),
            family: "test".into(),
            params_b: 7.0,
            context_length: 4096,
            preferred_backend: LlmBackend::GGUF,
            preferred_accelerator: AcceleratorKind::GPU,
            min_vram_mb: 5120,
        };
        let json = serde_json::to_string(&profile).unwrap();
        let parsed: ModelProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_id, "test-7b");
        assert_eq!(parsed.params_b, 7.0);
    }

    #[test]
    fn test_llm_instance_serde() {
        let instance = LlmInstance {
            instance_id: "inst-001".into(),
            model_id: "llama3-8b".into(),
            backend: LlmBackend::GGUF,
            accelerator: AcceleratorKind::GPU,
            context_length: 8192,
            params_b: 8.0,
        };
        let json = serde_json::to_string(&instance).unwrap();
        let parsed: LlmInstance = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.instance_id, "inst-001");
        assert_eq!(parsed.backend, LlmBackend::GGUF);
    }

    #[test]
    fn test_inference_telemetry_serde() {
        let t = InferenceTelemetry {
            total_tokens: 1000,
            total_requests: 10,
            total_energy_joules: 5.0,
            tokens_per_second: 50.0,
            joules_per_token: 0.005,
            cost_per_million_tokens_usd: 0.000167,
        };
        let json = serde_json::to_string(&t).unwrap();
        let parsed: InferenceTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_tokens, 1000);
        assert!((parsed.joules_per_token - 0.005).abs() < 1e-10);
    }
}
