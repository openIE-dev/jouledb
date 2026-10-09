//! Apple Neural Engine lane for JouleDB HDC similarity.
//!
//! There is no public API that dispatches work directly to the Apple Neural
//! Engine (ANE). The sanctioned path is Core ML: a model is compiled, loaded
//! with `MLComputeUnits::CPUAndNeuralEngine`, and Core ML's planner decides
//! which device runs each operation. This crate:
//!
//! 1. Generates a Core ML ML Program (`.mlpackage`) for HDC similarity
//!    `scores = Q(n x d, +-1, fp16) . M(k x d, +-1, fp16)^T` in pure Rust
//!    ([`model`]); no Python, no coremltools, no checked-in artifact.
//!    Hamming distance is `(d - score) / 2`. XOR/popcount would not be
//!    eligible for the ANE, so the binary vectors are expressed as +-1 FP16.
//! 2. Compiles it with `MLModel compileModelAtURL`, loads it with
//!    `CPUAndNeuralEngine`, and runs a prediction with Float16
//!    `MLMultiArray` inputs (macOS only).
//! 3. Loads an `MLComputePlan` for the compiled model and reports, per
//!    operation, the preferred and supported compute devices
//!    ([`Placement`]). Callers must use this placement, not an assumption,
//!    to decide whether the Neural Engine ran the similarity op.
//! 4. Optionally samples the `ANE Power` line of `powermetrics` through
//!    `sudo -n` (never prompts) to measure ANE power ([`power`]).
//!
//! Off macOS every entry point returns [`AneError::NotAvailable`]; nothing
//! panics and nothing pretends an ANE ran.

pub mod model;
mod pb;
pub mod power;
pub mod sampler;

#[cfg(target_os = "macos")]
mod coreml;

pub use model::{ModelShape, write_mlpackage};
pub use power::{PowerMeasurement, powermetrics_available};
pub use sampler::{Confidence, PowerRing, PowerSampler, RailSample, Rails, WindowEnergy};

/// Largest d-tile for which FP16 dot products of +-1 vectors are exact.
pub const EXACT_TILE_DIM: usize = 2048;

/// Graph layout of the similarity op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// `matmul(q[n,d], codebook[k,d], transpose_y = true)`.
    MatMul,
    /// Apple ml-ane-transformers layout: `q` as `(1, d, 1, n)` (B, C, 1, S)
    /// and a 1x1 `conv` with weight `(k, d, 1, 1)`.
    ChannelsFirstConv,
}

impl Layout {
    pub fn as_str(&self) -> &'static str {
        match self {
            Layout::MatMul => "matmul",
            Layout::ChannelsFirstConv => "conv1x1_bc1s",
        }
    }
}

/// Device named by `MLComputePlan` for one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlannedDevice {
    /// `MLNeuralEngineComputeDevice`.
    NeuralEngine,
    /// `MLGPUComputeDevice`.
    Gpu,
    /// `MLCPUComputeDevice`.
    Cpu,
    /// The plan did not report a device for the op.
    Unknown,
}

impl PlannedDevice {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlannedDevice::NeuralEngine => "neural_engine",
            PlannedDevice::Gpu => "gpu",
            PlannedDevice::Cpu => "cpu",
            PlannedDevice::Unknown => "unknown",
        }
    }
}

/// MIL operator without its opset prefix: `ios16.matmul` -> `matmul`.
pub fn base_operator(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// `MLComputePlan` result for one ML Program operation.
#[derive(Debug, Clone, PartialEq)]
pub struct OpPlacement {
    /// MIL operator name, e.g. `matmul`, `conv`, `const`.
    pub operator: String,
    pub outputs: Vec<String>,
    /// `preferredComputeDevice`, or `None` when the plan had no usage entry
    /// (Core ML reports none for `const` ops).
    pub preferred: Option<PlannedDevice>,
    pub supported: Vec<PlannedDevice>,
    /// `estimatedCostOfMLProgramOperation.weight`, when Core ML gives one.
    pub estimated_cost: Option<f64>,
}

/// Per-operation device placement from `MLComputePlan`.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub layout: Layout,
    /// Compute units the model was loaded with.
    pub compute_units: &'static str,
    pub ops: Vec<OpPlacement>,
}

impl Placement {
    /// The op that computes the similarity scores (`matmul` / `conv`).
    pub fn similarity_op(&self) -> Option<&OpPlacement> {
        self.ops.iter().find(|op| {
            matches!(base_operator(&op.operator), "matmul" | "conv" | "linear")
                || op.outputs.iter().any(|o| o == model::OUTPUT_NAME)
        })
    }

    /// Device the planner prefers for the similarity op.
    pub fn similarity_device(&self) -> PlannedDevice {
        self.similarity_op()
            .and_then(|op| op.preferred)
            .unwrap_or(PlannedDevice::Unknown)
    }

    /// `true` only when `MLComputePlan` names the Neural Engine for the op.
    pub fn similarity_on_neural_engine(&self) -> bool {
        self.similarity_device() == PlannedDevice::NeuralEngine
    }

    /// One line per op, e.g. `matmul->scores preferred=neural_engine supported=[neural_engine,cpu] cost=0.98`.
    pub fn lines(&self) -> Vec<String> {
        self.ops
            .iter()
            .map(|op| {
                let supported: Vec<&str> = op.supported.iter().map(|d| d.as_str()).collect();
                format!(
                    "{}->{} preferred={} supported=[{}] cost={}",
                    op.operator,
                    op.outputs.join(","),
                    op.preferred.map(|d| d.as_str()).unwrap_or("none"),
                    supported.join(","),
                    op.estimated_cost
                        .map(|c| format!("{c:.4}"))
                        .unwrap_or_else(|| "none".into())
                )
            })
            .collect()
    }

    /// Compact summary for a receipt: `layout:op@device`.
    pub fn summary(&self) -> String {
        let op = self
            .similarity_op()
            .map(|op| op.operator.as_str())
            .unwrap_or("similarity");
        format!(
            "{}:{}@{}",
            self.layout.as_str(),
            op,
            self.similarity_device().as_str()
        )
    }
}

/// When to sample ANE power with `powermetrics` (only ever through
/// `sudo -n`, so it never prompts; skipped when sudo would need a password).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PowerSampling {
    /// Never sample.
    #[default]
    Off,
    /// Sample only when `MLComputePlan` puts the similarity op on the
    /// Neural Engine (ANE power is meaningless for a CPU-placed op).
    OnNeuralEngine,
    /// Always sample (diagnostics).
    Always,
}

/// Options for [`hdc_similarity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdcOptions {
    pub layout: Layout,
    pub power: PowerSampling,
}

impl Default for HdcOptions {
    fn default() -> Self {
        Self {
            layout: Layout::MatMul,
            power: PowerSampling::Off,
        }
    }
}

/// Result of one Core ML HDC similarity run.
#[derive(Debug, Clone)]
pub struct HdcRun {
    /// `n x k` row-major dot products of the +-1 vectors.
    pub scores: Vec<i32>,
    pub n: usize,
    pub k: usize,
    pub d: usize,
    /// Number of d-tiles (each at most [`EXACT_TILE_DIM`]).
    pub tiles: usize,
    /// Placement of the (first) tile's model. All tiles share one graph.
    pub placement: Placement,
    /// Seconds spent generating + compiling models in this call (0 when the
    /// compiled `.mlmodelc` was already on disk or the model was loaded).
    pub compile_s: f64,
    /// One-time setup in this call: compile (if needed) + `MLComputePlan` +
    /// model load, for every tile not already loaded in this process.
    /// `0.0` when every tile was reused.
    pub setup_s: f64,
    /// `true` when this call compiled or loaded at least one model.
    pub setup_performed: bool,
    /// Seconds of per-job work: input conversion, prediction(s), readback.
    /// Excludes setup.
    pub predict_s: f64,
    /// ANE power measurement when sampling was requested (see
    /// [`PowerSampling`]) and powermetrics ran without a prompt.
    pub power: Option<PowerMeasurement>,
    /// Layouts tried, with the device the plan named for each.
    pub attempts: Vec<(Layout, PlannedDevice)>,
}

impl HdcRun {
    /// Hamming distances `(d - dot) / 2`, `n x k` row-major.
    pub fn hamming(&self) -> Vec<u32> {
        self.scores
            .iter()
            .map(|s| hamming_from_dot(*s, self.d))
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AneError {
    #[error("Apple Neural Engine lane not available: {reason}")]
    NotAvailable { reason: String },
    #[error("invalid HDC input: {0}")]
    InvalidInput(String),
    #[error("Core ML: {0}")]
    CoreMl(String),
    #[error("io: {0}")]
    Io(String),
}

impl AneError {
    fn not_macos() -> Self {
        AneError::NotAvailable {
            reason: "Core ML and the Apple Neural Engine exist only on macOS".to_string(),
        }
    }
}

/// `true` on macOS, where Core ML exists. Whether the ANE actually runs a
/// given op is only known from [`Placement`].
pub fn is_available() -> bool {
    cfg!(target_os = "macos")
}

/// Hamming distance of two +-1 vectors of length `d` from their dot product.
pub fn hamming_from_dot(dot: i32, d: usize) -> u32 {
    ((d as i64 - i64::from(dot)) / 2).max(0) as u32
}

/// Check shapes and that every value is +1 or -1. Returns `(n, k)`.
pub fn validate(queries: &[i8], memory: &[i8], d: usize) -> Result<(usize, usize), AneError> {
    if d == 0 {
        return Err(AneError::InvalidInput("d must be at least 1".into()));
    }
    if queries.is_empty() || queries.len() % d != 0 {
        return Err(AneError::InvalidInput(format!(
            "queries length {} is not a positive multiple of d={d}",
            queries.len()
        )));
    }
    if memory.is_empty() || memory.len() % d != 0 {
        return Err(AneError::InvalidInput(format!(
            "memory length {} is not a positive multiple of d={d}",
            memory.len()
        )));
    }
    if let Some(bad) = queries.iter().chain(memory).find(|v| **v != 1 && **v != -1) {
        return Err(AneError::InvalidInput(format!(
            "bipolar HDC values must be +1 or -1, found {bad}"
        )));
    }
    Ok((queries.len() / d, memory.len() / d))
}

/// CPU reference: `n x k` row-major dot products.
pub fn cpu_dot_scores(queries: &[i8], memory: &[i8], d: usize) -> Vec<i32> {
    let n = queries.len() / d.max(1);
    let k = memory.len() / d.max(1);
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        let q = &queries[i * d..(i + 1) * d];
        for j in 0..k {
            let m = &memory[j * d..(j + 1) * d];
            out.push(q.iter().zip(m).map(|(a, b)| i32::from(*a) * i32::from(*b)).sum());
        }
    }
    out
}

/// Stable fingerprint of a model (shape + codebook tile) for the compile cache.
pub(crate) fn fingerprint(shape: &ModelShape, codebook: &[i8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for b in [shape.layout as u8, 0xA5] {
        eat(b);
    }
    for v in [shape.n, shape.d, shape.k] {
        for b in (v as u64).to_le_bytes() {
            eat(b);
        }
    }
    for v in codebook {
        eat(*v as u8);
    }
    model::splitmix(h)
}

/// Copy column range `[start, end)` of a row-major `rows x d` matrix.
pub(crate) fn tile_columns(src: &[i8], d: usize, start: usize, end: usize) -> Vec<i8> {
    let rows = src.len() / d;
    let mut out = Vec::with_capacity(rows * (end - start));
    for r in 0..rows {
        out.extend_from_slice(&src[r * d + start..r * d + end]);
    }
    out
}

/// Run HDC similarity on Core ML with `CPUAndNeuralEngine`.
///
/// `queries` is `n x d` and `memory` is `k x d`, row-major, every value +1
/// or -1. `d` is split into tiles of at most [`EXACT_TILE_DIM`] so FP16
/// arithmetic stays exact; tile scores are summed in i32 on the host.
pub fn hdc_similarity(
    queries: &[i8],
    memory: &[i8],
    d: usize,
    options: HdcOptions,
) -> Result<HdcRun, AneError> {
    #[cfg(target_os = "macos")]
    {
        let (n, k) = validate(queries, memory, d)?;
        coreml::hdc_similarity(queries, memory, n, k, d, options)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (queries, memory, d, options);
        Err(AneError::not_macos())
    }
}

/// Try [`Layout::MatMul`] first. If `MLComputePlan` does not put the
/// similarity op on the Neural Engine, try the ANE-friendly
/// [`Layout::ChannelsFirstConv`] and keep whichever the plan placed on the
/// Neural Engine (or the conv run if neither was). `attempts` records both.
pub fn hdc_similarity_auto(
    queries: &[i8],
    memory: &[i8],
    d: usize,
    power: PowerSampling,
) -> Result<HdcRun, AneError> {
    let first = hdc_similarity(
        queries,
        memory,
        d,
        HdcOptions {
            layout: Layout::MatMul,
            power,
        },
    );
    let first = match first {
        Ok(run) if run.placement.similarity_on_neural_engine() => return Ok(run),
        other => other,
    };
    let second = hdc_similarity(
        queries,
        memory,
        d,
        HdcOptions {
            layout: Layout::ChannelsFirstConv,
            power,
        },
    );
    // Setup paid by either attempt in this call belongs to this call.
    let first_setup = first.as_ref().map(|r| (r.setup_s, r.setup_performed, r.compile_s)).ok();
    let merge = |mut run: HdcRun, other: Option<(f64, bool, f64)>, attempts: Vec<(Layout, PlannedDevice)>| {
        if let Some((s, p, c)) = other {
            run.setup_s += s;
            run.setup_performed |= p;
            run.compile_s += c;
        }
        run.attempts = attempts;
        run
    };
    match (first, second) {
        (Ok(a), Ok(b)) => {
            let mut attempts = a.attempts.clone();
            attempts.extend(b.attempts.iter().copied());
            if b.placement.similarity_on_neural_engine() || !a.placement.similarity_on_neural_engine() {
                Ok(merge(b, first_setup, attempts))
            } else {
                let b_setup = Some((b.setup_s, b.setup_performed, b.compile_s));
                Ok(merge(a, b_setup, attempts))
            }
        }
        (Ok(a), Err(_)) => Ok(a),
        (Err(_), Ok(b)) => Ok(b),
        (Err(e), Err(_)) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_matches_dot_identity() {
        let q = [1i8, -1, 1, 1];
        let m = [1i8, 1, 1, -1, -1, -1, -1, -1];
        let dots = cpu_dot_scores(&q, &m, 4);
        assert_eq!(dots, vec![0, -2]);
        assert_eq!(hamming_from_dot(dots[0], 4), 2);
        assert_eq!(hamming_from_dot(dots[1], 4), 3);
        assert_eq!(hamming_from_dot(4, 4), 0);
    }

    #[test]
    fn validate_rejects_non_bipolar_and_bad_shapes() {
        assert!(matches!(validate(&[1, 0], &[1, 1], 2), Err(AneError::InvalidInput(_))));
        assert!(matches!(validate(&[1, 1, 1], &[1, 1], 2), Err(AneError::InvalidInput(_))));
        assert!(matches!(validate(&[], &[1, 1], 2), Err(AneError::InvalidInput(_))));
        assert_eq!(validate(&[1, -1, 1, 1], &[1, 1], 2).ok(), Some((2, 1)));
    }

    #[test]
    fn tile_columns_slices_rows() {
        let src = [1i8, 2, 3, 4, 5, 6];
        assert_eq!(tile_columns(&src, 3, 1, 3), vec![2, 3, 5, 6]);
    }

    #[test]
    fn placement_reports_similarity_device_truthfully() {
        let mut p = Placement {
            layout: Layout::MatMul,
            compute_units: "cpu_and_neural_engine",
            ops: vec![
                OpPlacement {
                    operator: "const".into(),
                    outputs: vec!["codebook".into()],
                    preferred: None,
                    supported: vec![],
                    estimated_cost: None,
                },
                OpPlacement {
                    operator: "matmul".into(),
                    outputs: vec!["scores".into()],
                    preferred: Some(PlannedDevice::Cpu),
                    supported: vec![PlannedDevice::Cpu, PlannedDevice::NeuralEngine],
                    estimated_cost: Some(1.0),
                },
            ],
        };
        assert_eq!(p.similarity_device(), PlannedDevice::Cpu);
        assert!(!p.similarity_on_neural_engine());
        assert_eq!(p.summary(), "matmul:matmul@cpu");
        p.ops[1].operator = "ios16.matmul".into();
        p.ops[1].outputs = vec!["other".into()];
        assert_eq!(base_operator("ios16.matmul"), "matmul");
        assert_eq!(p.similarity_device(), PlannedDevice::Cpu);
        p.ops[1].preferred = Some(PlannedDevice::NeuralEngine);
        assert!(p.similarity_on_neural_engine());
        p.ops[1].preferred = None;
        assert_eq!(p.similarity_device(), PlannedDevice::Unknown);
        assert!(p.lines()[1].contains("preferred=none"));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn off_macos_is_not_available_and_never_panics() {
        assert!(!is_available());
        let err = hdc_similarity(&[1, -1], &[1, 1], 2, HdcOptions::default()).unwrap_err();
        assert!(matches!(err, AneError::NotAvailable { .. }), "{err}");
        let err = hdc_similarity_auto(&[1, -1], &[1, 1], 2, PowerSampling::Always).unwrap_err();
        assert!(matches!(err, AneError::NotAvailable { .. }), "{err}");
    }
}
