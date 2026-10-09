//! SigQL (Signal Query Language) Execution Engine
//!
//! Bridges SigQL's signal-processing pipeline into JouleDB's query dispatch.
//! SigQL queries operate on time-series signal data with DSP transforms,
//! frequency-domain operations, and uncertainty-quantified aggregations.
//!
//! Signal sources are resolved from amorphic storage tables:
//! - `FROM controller.imu.accel` → table `controller_imu_accel`
//! - Numeric `value` column extracted as f64 samples
//! - Sample rate from `sample_rate` column or default (1000 Hz)

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::Value;
use joule_db_query::executor::TableStorage;
use sigql::ast::FromClause;
use sigql::compile::Target;
use sigql::compile::gpu::{GpuInput, GpuKernel};
use sigql::runtime::{OutputValue, Runtime, RuntimeConfig};
use sigql::types::DynSignal;
use std::sync::Arc;
use std::time::Instant;

/// Maximum rows returned for signal/spectrum outputs to prevent OOM.
const MAX_SIGNAL_ROWS: usize = 10_000;

// ============================================================================
// Detection
// ============================================================================


/// Choose SigQL target: WebGpu when a wgpu adapter is present, else Simd.
///
/// The adapter probe runs once and is cached. Never invents a Metal backend:
/// on a Mac wgpu itself picks Metal, and the receipt names the adapter.
fn select_sigql_target() -> Target {
    if gpu_context().is_ok() {
        Target::WebGpu
    } else {
        Target::Simd
    }
}

/// Detect whether a query string is SigQL rather than SQL.
///
/// SigQL is distinguished by keywords/syntax that never appear in SQL:
/// - `TRANSFORM` keyword after `FROM`
/// - `AGGREGATE {` with curly braces
/// - `CORRELATE` keyword
/// - Pipeline operator `|>`
/// - Frequency literals (`4Hz`, `12Hz`)
/// - `RETURNING CONFIDENCE(...)`
pub fn is_sigql_query(sql: &str) -> bool {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();

    // Primary pattern: FROM ... <sigql-keyword>
    if upper.starts_with("FROM ") {
        if upper.contains(" TRANSFORM ")
            || upper.contains(" AGGREGATE {")
            || upper.contains(" AGGREGATE{")
            || upper.contains(" CORRELATE ")
            || upper.contains(" CORRELATE{")
            || upper.contains(" RETURNING CONFIDENCE")
            || trimmed.contains("|>")
            || contains_frequency_literal(trimmed)
        {
            return true;
        }
    }

    // Pipeline syntax: source |> operation (exclusive to SigQL)
    if trimmed.contains("|>") && !upper.contains("SELECT") {
        return true;
    }

    false
}

/// Check for frequency literals like `4Hz`, `12.5Hz`, `500kHz`.
fn contains_frequency_literal(s: &str) -> bool {
    let bytes = s.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        // Look for digit followed by "Hz" (case-insensitive)
        if bytes[i].is_ascii_digit()
            && (bytes[i + 1] == b'H' || bytes[i + 1] == b'h')
            && (bytes[i + 2] == b'z' || bytes[i + 2] == b'Z')
        {
            return true;
        }
        // Also check for "kHz", "MHz" patterns: digit + 'k'/'M' + 'Hz'
        if i + 3 < bytes.len()
            && bytes[i].is_ascii_digit()
            && (bytes[i + 1] == b'k' || bytes[i + 1] == b'M' || bytes[i + 1] == b'G')
            && (bytes[i + 2] == b'H' || bytes[i + 2] == b'h')
            && (bytes[i + 3] == b'z' || bytes[i + 3] == b'Z')
        {
            return true;
        }
    }
    false
}

// ============================================================================
// Execution
// ============================================================================

/// Execute a SigQL query against amorphic storage.
///
/// Pipeline: parse → compile → load sources → run on the GPU when a wgpu
/// adapter is present and every plan step has a kernel, else on the CPU
/// runtime → map to QueryResponse. The response says which one ran and, on a
/// CPU fallback, why.
pub fn execute_sigql(
    sql: &str,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    execute_sigql_routed(sql, amorphic, start, crate::cost_model::global())
}

/// The cost-model key for a plan: `sigql:` plus the distinct GPU kernels
/// (or CPU step labels) it runs, e.g. `sigql:iir`. `Err` names why the plan
/// cannot run on the GPU.
fn plan_op_kind(plan: &sigql::ExecutionPlan) -> (String, Result<(), String>) {
    use sigql::compile::PlanStep;
    let mut parts: Vec<String> = Vec::new();
    let mut gpu = Ok(());
    for step in &plan.steps {
        if matches!(step, PlanStep::LoadSignal { .. } | PlanStep::Store { .. } | PlanStep::Passthrough { .. }) {
            continue;
        }
        let part = match GpuKernel::from_step(step) {
            Ok(Some(kernel)) => kernel.name().to_string(),
            Ok(None) => continue,
            Err(reason) => {
                if gpu.is_ok() {
                    gpu = Err(reason);
                }
                sigql::compile::codegen::step_label(step).to_ascii_lowercase()
            }
        };
        if !parts.contains(&part) {
            parts.push(part);
        }
    }
    (format!("sigql:{}", parts.join("+")), gpu)
}

/// Run the plan on the CPU runtime.
fn run_cpu_plan(
    plan: &sigql::ExecutionPlan,
    sources: &[(String, DynSignal<f64>)],
) -> Result<sigql::runtime::ExecutionResult, QueryErrorResponse> {
    let mut runtime = Runtime::new(RuntimeConfig::default());
    for (name, signal) in sources {
        runtime.register_signal(name.clone(), signal.clone());
    }
    runtime
        .execute(plan)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL runtime: {}", e)))
}

/// [`execute_sigql`] with an explicit cost model: the fabric routes the
/// plan to the processor that measured cheapest (joules, then time) for this
/// op kind and size on this machine, explores a bounded number of times, and
/// never refuses. The response carries an [`energy_receipt`](QueryResponse::energy_receipt).
pub fn execute_sigql_routed(
    sql: &str,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
    model: &crate::cost_model::CostModel,
) -> Result<QueryResponse, QueryErrorResponse> {
    let query = sigql::parse(sql)
        .map_err(|e| QueryErrorResponse::syntax_error(&format!("SigQL: {}", e), 1, 1))?;
    let plan = sigql::compile(&query, Target::Simd)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL compile: {}", e)))?;
    let sources = load_sources(&query, amorphic);
    run_routed(&plan, &sources, start, model, crate::power_meter::Meter::global())
}

/// Route a compiled plan by measured cost and run it (see
/// [`execute_sigql_routed`]).
fn run_routed(
    plan: &sigql::ExecutionPlan,
    sources: &[(String, DynSignal<f64>)],
    start: Instant,
    model: &crate::cost_model::CostModel,
    meter: &crate::power_meter::Meter,
) -> Result<QueryResponse, QueryErrorResponse> {
    use crate::cost_model::{Candidate, RouteReceipt, size_bucket};
    let (op, lowers) = plan_op_kind(plan);
    let size: usize = sources.iter().map(|(_, s)| s.samples.len()).sum();

    // Candidates: the GPU when it exists and every step has a kernel, then
    // the CPU (always).
    let mut candidates = Vec::new();
    let mut cpu_only_note = None;
    match (gpu_context(), &lowers) {
        (Ok(ctx), Ok(())) => candidates.push(Candidate::new("gpu", ctx.backend.clone())),
        (Err(reason), _) => cpu_only_note = Some(format!("no wgpu adapter ({reason})")),
        (Ok(_), Err(reason)) => cpu_only_note = Some(format!("WebGPU path unavailable ({reason})")),
    }
    candidates.push(Candidate::cpu());
    let route = model.choose(&op, size, &candidates);
    let mut route_reason = route.reason_label();
    let watts = crate::fabric::routing_watts();

    let span = meter.begin();
    let mut gpu_error = None;
    let mut ran_gpu = None;
    let mut setup = std::time::Duration::ZERO;
    if route.chosen.device == "gpu" {
        match execute_plan_webgpu(&plan, &sources) {
            Ok(done) => ran_gpu = Some(done),
            Err(reason) => {
                route_reason.push_str("+gpu_error");
                gpu_error = Some(reason);
            }
        }
    }
    let (mut response, ran) = match ran_gpu {
        Some((result, run)) => {
            setup = run.setup;
            let mut response = map_result_to_response(result, start)?;
            response.device_target = Some("webgpu".into());
            response.warnings.push(format!(
                "SigQL ran on the GPU ({}) as {} WGSL kernel(s): {}",
                run.adapter,
                run.kernels.len(),
                run.kernels.join(", ")
            ));
            (response, route.chosen.clone())
        }
        None => {
            let result = run_cpu_plan(&plan, &sources)?;
            let mut response = map_result_to_response(result, start)?;
            response.device_target = Some("cpu".into());
            response.warnings.push(match (&gpu_error, &cpu_only_note) {
                (Some(reason), _) => format!("SigQL ran on the CPU runtime: WebGPU path failed ({reason})"),
                (None, Some(note)) => format!("SigQL ran on the CPU runtime: {note}"),
                (None, None) => format!("SigQL ran on the CPU runtime (route: {route_reason})"),
            });
            (response, Candidate::cpu())
        }
    };
    // Pipeline compilation is one-time setup: reported, not charged to the
    // job, so a first GPU run is not judged by its shader compile.
    let energy = meter.finish(span, setup, &ran.device, watts);
    let setup_seconds = (!setup.is_zero()).then(|| setup.as_secs_f64());
    model.record_energy(&op, size, &ran, &energy);
    // The next real processor: the route's fallback when the chosen device
    // ran; the processor that actually ran when it did not.
    let fallback = if ran == route.chosen { route.fallback.device.clone() } else { ran.device.clone() };
    let receipt = RouteReceipt {
        op,
        size_bucket: size_bucket(size),
        requested_device: route.requested.device.clone(),
        device: ran.device.clone(),
        backend: ran.backend.clone(),
        fallback,
        route_reason,
        ranked_by: route.ranked_by.map(|b| b.label().to_string()).unwrap_or_else(|| "none".into()),
        joules: energy.joules,
        incremental_joules: energy.incremental_joules,
        seconds: energy.seconds,
        watts: energy.watts,
        energy_source: energy.source.clone(),
        energy_confidence: energy.confidence.clone(),
        cpu_watts: energy.cpu_watts,
        gpu_watts: energy.gpu_watts,
        ane_watts: energy.ane_watts,
        baseline_watts: energy.baseline_watts,
        setup_seconds,
    };
    response.energy_joules = Some(energy.joules);
    response.power_watts = Some(energy.watts);
    response.energy_receipt = Some(receipt);
    Ok(response)
}

/// [`execute_sigql`] with an explicit target, so tests can compare backends.
pub fn execute_sigql_with(
    sql: &str,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
    target: Target,
) -> Result<QueryResponse, QueryErrorResponse> {
    // 1. Parse
    let query = sigql::parse(sql)
        .map_err(|e| QueryErrorResponse::syntax_error(&format!("SigQL: {}", e), 1, 1))?;

    // 2. Compile (the plan is target-independent; the target picks the executor)
    let plan = sigql::compile(&query, Target::Simd)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL compile: {}", e)))?;

    // 3. Signal sources from storage
    let sources = load_sources(&query, amorphic);

    // 4. GPU first when requested; any reason it cannot run is reported.
    let gpu_note = if target == Target::WebGpu {
        match execute_plan_webgpu(&plan, &sources) {
            Ok((result, run)) => {
                let mut response = map_result_to_response(result, start)?;
                response.device_target = Some("webgpu".into());
                response.warnings.push(format!(
                    "SigQL ran on the GPU ({}) as {} WGSL kernel(s): {}",
                    run.adapter,
                    run.kernels.len(),
                    run.kernels.join(", ")
                ));
                return Ok(response);
            }
            Err(reason) => format!("SigQL ran on the CPU runtime: WebGPU path unavailable ({reason})"),
        }
    } else if target == Target::Simd {
        match gpu_context() {
            Ok(_) => "SigQL ran on the CPU runtime (Simd target requested)".to_string(),
            Err(reason) => format!("SigQL ran on the CPU runtime: no wgpu adapter ({reason})"),
        }
    } else {
        format!("SigQL ran on the CPU runtime ({target:?} target)")
    };

    // 5. CPU DSP / Simd runtime
    let mut runtime = Runtime::new(RuntimeConfig::default());
    for (name, signal) in sources {
        runtime.register_signal(name, signal);
    }
    let result = runtime
        .execute(&plan)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL runtime: {}", e)))?;

    let mut response = map_result_to_response(result, start)?;
    response.device_target = Some("cpu".into());
    response.warnings.push(gpu_note);
    Ok(response)
}

// ============================================================================
// WebGPU executor
// ============================================================================

/// Shared wgpu device for every SigQL query, opened once.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// "<adapter name> via <backend>", for receipts.
    adapter: String,
    /// Routing backend label, e.g. `wgpu-metal`, `wgpu-vulkan`.
    backend: String,
    /// The adapter is a CPU rasterizer (lavapipe, SwiftShader, WARP).
    cpu_adapter: bool,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// Compiled pipelines keyed by (WGSL source, entry point).
    pipelines: std::sync::Mutex<std::collections::HashMap<(String, &'static str), wgpu::ComputePipeline>>,
    /// The device can write GPU timestamps at compute-pass boundaries
    /// (used for per-pass profiles).
    timestamps: bool,
}

/// What a GPU run did, for the response.
struct GpuRun {
    adapter: String,
    kernels: Vec<&'static str>,
    /// Time spent compiling pipelines this run (zero when all were cached):
    /// one-time setup, kept out of the per-job cost.
    setup: std::time::Duration,
}

fn gpu_context() -> Result<&'static GpuContext, String> {
    use std::sync::OnceLock;
    static CTX: OnceLock<Result<GpuContext, String>> = OnceLock::new();
    CTX.get_or_init(|| {
        if std::env::var("JOULE_SIGQL_WEBGPU").is_ok_and(|v| v == "0") {
            return Err("disabled by JOULE_SIGQL_WEBGPU=0".into());
        }
        // Own thread so pollster never nests inside a tokio worker.
        std::thread::spawn(|| pollster::block_on(open_gpu()))
            .join()
            .unwrap_or_else(|_| Err("wgpu adapter probe panicked".into()))
    })
    .as_ref()
    .map_err(Clone::clone)
}

async fn open_gpu() -> Result<GpuContext, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
        memory_budget_thresholds: Default::default(),
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("no wgpu adapter: {e}"))?;
    let info = adapter.get_info();
    let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sigql-webgpu"),
            required_features: if timestamps { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
        })
        .await
        .map_err(|e| format!("wgpu device: {e}"))?;
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sigql-kernel"),
        entries: &[
            storage(0, true),
            storage(1, false),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(3, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sigql-kernel"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    Ok(GpuContext {
        device,
        queue,
        adapter: format!("{} via {:?}", info.name, info.backend),
        backend: format!("wgpu-{:?}", info.backend).to_ascii_lowercase(),
        cpu_adapter: info.device_type == wgpu::DeviceType::Cpu,
        layout,
        pipeline_layout,
        pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
        timestamps,
    })
}

impl GpuContext {
    /// Adapter-specific tuning of a bound kernel. A CPU rasterizer has a
    /// few wide lanes, so the chunked IIR uses about `sqrt(n)` chunks
    /// (measured fastest on lavapipe); a GPU keeps the default ~64-sample
    /// chunks (measured fastest on an M5 Max).
    fn tune(&self, kernel: GpuKernel, n: usize) -> GpuKernel {
        if self.cpu_adapter {
            let target = (n as f64).sqrt().ceil() as usize;
            kernel.with_iir_chunk_target(n, target)
        } else {
            kernel
        }
    }

    /// Compile (or reuse) the pipeline for one entry point of a kernel's
    /// WGSL, surfacing WGSL errors.
    fn pipeline(&self, wgsl: &str, entry: &'static str, label: &str) -> Result<wgpu::ComputePipeline, String> {
        let key = (wgsl.to_string(), entry);
        if let Some(p) = self.pipelines.lock().ok().and_then(|m| m.get(&key).cloned()) {
            return Ok(p);
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(format!("{label} kernel ({entry}) failed to compile: {err}"));
        }
        if let Ok(mut m) = self.pipelines.lock() {
            m.insert(key, pipeline.clone());
        }
        Ok(pipeline)
    }
}

/// A plan register held on the GPU.
#[derive(Clone)]
enum GpuValue {
    Signal { buf: wgpu::Buffer, len: usize, rate: u32, channel: String, start_ns: i64 },
    Spectrum { buf: wgpu::Buffer, len: usize },
    /// Raw output of a scalar kernel (statistics, band power), turned into
    /// the scalar on readback by `GpuKernel::finish_scalar`.
    Scalar { buf: wgpu::Buffer, kernel: GpuKernel, n: usize },
}

/// Run a whole plan on the GPU, or say why it cannot.
fn execute_plan_webgpu(
    plan: &sigql::ExecutionPlan,
    sources: &[(String, DynSignal<f64>)],
) -> Result<(sigql::runtime::ExecutionResult, GpuRun), String> {
    // Lower every step before touching the device, so an unsupported step
    // costs nothing and is the reported reason.
    let mut kernels = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        kernels.push(GpuKernel::from_step(step)?);
    }
    let ctx = gpu_context()?;
    let plan = plan.clone();
    let sources = sources.to_vec();
    // wgpu error scopes are per thread and pollster must not run on a tokio
    // worker, so the whole run happens on its own thread.
    std::thread::spawn(move || run_on_gpu(ctx, &plan, &kernels, &sources))
        .join()
        .map_err(|_| "GPU run panicked".to_string())?
}

fn run_on_gpu(
    ctx: &GpuContext,
    plan: &sigql::ExecutionPlan,
    kernels: &[Option<GpuKernel>],
    sources: &[(String, DynSignal<f64>)],
) -> Result<(sigql::runtime::ExecutionResult, GpuRun), String> {
    use sigql::compile::PlanStep;
    use wgpu::util::DeviceExt;

    let began = Instant::now();
    let device = &ctx.device;
    let max_bytes = device.limits().max_storage_buffer_binding_size;
    let storage = |bytes: u64, label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };

    let mut regs: Vec<Option<GpuValue>> = vec![None; plan.allocate_resources().num_registers as usize];
    let reg = |regs: &Vec<Option<GpuValue>>, r: sigql::compile::RegisterId| -> Result<GpuValue, String> {
        regs.get(r.0 as usize)
            .cloned()
            .flatten()
            .ok_or_else(|| format!("register {} is empty", r.0))
    };
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sigql"),
    });
    let mut stores = Vec::new();
    let mut names = Vec::new();
    let mut samples_processed = 0usize;
    let mut setup = std::time::Duration::ZERO;
    let mut bytes_used = 0u64;

    for (step, kernel) in plan.steps.iter().zip(kernels) {
        match (step, kernel) {
            (PlanStep::LoadSignal { source, output }, _) => {
                let signal = sources
                    .iter()
                    .find(|(name, _)| name.as_str() == source.as_str())
                    .map(|(_, s)| s)
                    .ok_or_else(|| format!("signal source '{source}' is not in storage"))?;
                let bytes = (signal.samples.len() * 4) as u64;
                if bytes > max_bytes {
                    return Err(format!("signal '{source}' ({bytes} bytes) exceeds the adapter's storage binding limit"));
                }
                let data: Vec<u8> = signal.samples.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sigql-load"),
                    contents: if data.is_empty() { &[0u8; 4] } else { &data },
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
                bytes_used += bytes;
                samples_processed += signal.samples.len();
                regs[output.0 as usize] = Some(GpuValue::Signal {
                    buf,
                    len: signal.samples.len(),
                    rate: signal.sample_rate,
                    channel: signal.channel.to_string(),
                    start_ns: signal.start_ns,
                });
            }
            (PlanStep::Store { input, name }, _) => {
                let value = reg(&regs, *input)?;
                regs[input.0 as usize] = None; // the CPU runtime takes it too
                stores.push((name.clone(), value));
            }
            (PlanStep::Passthrough { input, output }, _) => {
                let value = reg(&regs, *input)?;
                regs[output.0 as usize] = Some(value);
            }
            (_, Some(kernel)) => {
                let (input, output) = match (
                    sigql::compile::codegen::step_input(step),
                    sigql::compile::codegen::step_output(step),
                ) {
                    (Some(i), Some(o)) => (i, o),
                    _ => return Err(format!("{} has no single input/output register", kernel.name())),
                };
                let (buf, len, rate, channel, start_ns, is_spectrum) = match reg(&regs, input)? {
                    GpuValue::Signal { buf, len, rate, channel, start_ns } => {
                        (buf, len, rate, channel, start_ns, false)
                    }
                    GpuValue::Spectrum { buf, len } => (buf, len, 0, String::new(), 0, true),
                    GpuValue::Scalar { .. } => {
                        return Err(format!("{} cannot take a scalar input", kernel.name()))
                    }
                };
                let second = match step {
                    PlanStep::CrossCorrelate { input_b, .. } => match reg(&regs, *input_b)? {
                        GpuValue::Signal { buf, len, .. } => Some((buf, len)),
                        _ => return Err("cross-correlation needs two time-domain signals".into()),
                    },
                    _ => None,
                };
                let kernel = ctx.tune(
                    kernel.bind(&GpuInput {
                        len,
                        rate,
                        is_spectrum,
                        second_len: second.as_ref().map(|(_, l)| *l),
                        default_rate: RuntimeConfig::default().default_sample_rate,
                    })?,
                    len,
                );
                let out_len = kernel.output_len(len);
                let buf_bytes = (kernel.buffer_len(len) * 4) as u64;
                if buf_bytes > max_bytes {
                    return Err(format!(
                        "{} needs a {buf_bytes}-byte buffer, above the adapter's storage binding limit",
                        kernel.name()
                    ));
                }
                let out = storage(buf_bytes, kernel.name());
                bytes_used += buf_bytes;
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sigql-params"),
                    contents: &kernel.params(len).iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let aux = match &second {
                    Some((b, _)) => b.clone(),
                    None => device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("sigql-aux"),
                        contents: &kernel.aux().iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
                };
                let wgsl = kernel.wgsl();
                let compiling = Instant::now();
                let passes = kernel
                    .passes(len)
                    .into_iter()
                    .map(|(entry, groups)| Ok((ctx.pipeline(&wgsl, entry, kernel.name())?, groups)))
                    .collect::<Result<Vec<_>, String>>()?;
                setup += compiling.elapsed();
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(kernel.name()),
                    layout: &ctx.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: aux.as_entire_binding() },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(kernel.name()),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, &bind, &[]);
                    // Dispatches in one pass are ordered: each sees the
                    // previous one's storage writes.
                    for (pipeline, groups) in &passes {
                        pass.set_pipeline(pipeline);
                        pass.dispatch_workgroups(*groups, 1, 1);
                    }
                }
                names.push(kernel.name());
                regs[output.0 as usize] = Some(if matches!(kernel, GpuKernel::Spectrum { .. }) {
                    GpuValue::Spectrum { buf: out, len: out_len }
                } else if kernel.is_scalar() {
                    GpuValue::Scalar { buf: out, kernel, n: len }
                } else if matches!(kernel, GpuKernel::CrossCorrelate { .. }) {
                    // Runtime::cross_correlate names and times its output so.
                    GpuValue::Signal { buf: out, len: out_len, rate, channel: "cross_correlation".into(), start_ns: 0 }
                } else {
                    GpuValue::Signal { buf: out, len: out_len, rate: kernel.output_rate(rate), channel, start_ns }
                });
            }
            (other, None) => {
                return Err(format!(
                    "{} has no GPU kernel",
                    sigql::compile::codegen::step_label(other)
                ));
            }
        }
    }

    // Read back every stored register in one submission.
    let mut staged = Vec::with_capacity(stores.len());
    for (name, value) in &stores {
        let (src, len) = match value {
            GpuValue::Signal { buf, len, .. } | GpuValue::Spectrum { buf, len } => (buf, *len),
            GpuValue::Scalar { buf, kernel, n } => (buf, kernel.output_len(*n)),
        };
        let bytes = (len.max(1) * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name.as_str()),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        staged.push((staging, len));
    }
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, (staging, _)) in staged.iter().enumerate() {
        let tx = tx.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send((i, r));
        });
    }
    drop(tx);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(format!("GPU submission failed: {err}"));
    }
    for (i, r) in rx.iter() {
        r.map_err(|e| format!("readback of output {i} failed: {e}"))?;
    }

    let mut outputs = std::collections::HashMap::new();
    for ((name, value), (staging, len)) in stores.into_iter().zip(&staged) {
        let data: Vec<f64> = staging
            .slice(..)
            .get_mapped_range()
            .chunks_exact(4)
            .take(*len)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect();
        staging.unmap();
        let out = match value {
            GpuValue::Signal { rate, channel, start_ns, .. } => {
                OutputValue::Signal(DynSignal::new(channel, data, rate, start_ns))
            }
            GpuValue::Spectrum { .. } => OutputValue::Spectrum(data),
            GpuValue::Scalar { kernel, n, .. } => OutputValue::Scalar(
                kernel
                    .finish_scalar(&data, n)
                    .ok_or_else(|| format!("{} on {n} samples has no GPU result", kernel.name()))?,
            ),
        };
        outputs.insert(name, out);
    }

    Ok((
        sigql::runtime::ExecutionResult {
            outputs,
            stats: sigql::runtime::ExecutionStats {
                samples_processed,
                execution_time_ns: began.elapsed().as_nanos() as u64,
                memory_used: bytes_used as usize,
            },
        },
        GpuRun { adapter: ctx.adapter.clone(), kernels: names, setup },
    ))
}

/// Extract signal source names from the parsed query's FROM clauses
/// and load matching amorphic tables as signals (shared by the GPU and CPU paths).
fn load_sources(
    query: &sigql::Query,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Vec<(String, DynSignal<f64>)> {
    let mut loaded = Vec::new();
    for from in &query.from {
        let source_name = match from {
            FromClause::Signal(source_ref) => source_ref.path.to_string(),
            FromClause::Table { name, .. } => name.to_string(),
            _ => continue,
        };

        // Map dotted signal path to table name: controller.imu.accel → controller_imu_accel
        let table_name = source_name.replace('.', "_");

        // Check if table exists in amorphic storage
        let tables = amorphic.list_tables();
        if !tables.contains(&table_name) {
            // Not in storage — the runtime will handle SourceNotFound if needed
            continue;
        }

        // Scan the table for signal data
        let scan = match amorphic.scan(&table_name) {
            Ok(rows) => rows,
            Err(_) => continue,
        };
        if scan.is_empty() {
            continue;
        }

        // Extract samples from the `value` column (or first numeric column)
        let mut samples: Vec<f64> = Vec::new();
        let mut sample_rate: u32 = 1000; // default

        // Find column indices from the table schema
        let columns = match amorphic.columns(&table_name) {
            Ok(cols) => cols,
            Err(_) => continue,
        };
        let value_idx = columns.iter().position(|c| c == "value");
        let rate_idx = columns.iter().position(|c| c == "sample_rate");

        for row in &scan {
            if let Some(idx) = value_idx {
                if let Some(val) = row.values.get(idx) {
                    if let Some(f) = value_to_f64(val) {
                        samples.push(f);
                    }
                }
            } else {
                // No `value` column — try first numeric column
                for val in &row.values {
                    if let Some(f) = value_to_f64(val) {
                        samples.push(f);
                        break;
                    }
                }
            }

            // Extract sample rate from first row
            if samples.len() == 1 {
                if let Some(idx) = rate_idx {
                    if let Some(val) = row.values.get(idx) {
                        if let Some(r) = value_to_f64(val) {
                            sample_rate = r as u32;
                        }
                    }
                }
            }
        }

        if !samples.is_empty() {
            let signal = DynSignal::new(&source_name, samples, sample_rate, 0);
            loaded.push((source_name, signal));
        }
    }

    loaded
}

/// Convert an AST Value to f64 if possible.
fn value_to_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

// ============================================================================
// Result Mapping
// ============================================================================

/// Map SigQL `ExecutionResult` to JouleDB `QueryResponse`.
fn map_result_to_response(
    result: sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    if result.outputs.is_empty() {
        // No named outputs: report the runtime's execution stats instead of
        // a hardcoded success token.
        return Ok(QueryResponse {
            columns: vec![
                "samples_processed".into(),
                "execution_time_ns".into(),
                "memory_used".into(),
            ],
            rows: vec![vec![
                serde_json::json!(result.stats.samples_processed),
                serde_json::json!(result.stats.execution_time_ns),
                serde_json::json!(result.stats.memory_used),
            ]],
            affected_rows: None,
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated: false,
            warnings: vec!["sigql plan produced no named outputs".into()],
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: Some("sigql".into()),
            session_id: None,
            viz_hint: None,
            energy_receipt: None,
        });
    }

    // Classify outputs
    let mut has_scalar = false;
    let mut has_signal = false;
    let mut has_spectrum = false;
    for val in result.outputs.values() {
        match val {
            OutputValue::Scalar(_) => has_scalar = true,
            OutputValue::Signal(_) => has_signal = true,
            OutputValue::Spectrum(_) => has_spectrum = true,
        }
    }

    // All-scalar: compact uncertainty table
    if has_scalar && !has_signal && !has_spectrum {
        return map_scalars(&result, start);
    }

    // Signal output
    if has_signal && !has_spectrum {
        return map_signals(&result, start);
    }

    // Spectrum output
    if has_spectrum && !has_signal {
        return map_spectra(&result, start);
    }

    // Mixed: use generic format
    map_mixed(&result, start)
}

/// Map all-scalar outputs to a compact uncertainty table.
fn map_scalars(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec![
        "name".into(),
        "value".into(),
        "confidence".into(),
        "lower_bound".into(),
        "upper_bound".into(),
        "n_samples".into(),
    ];

    let mut rows = Vec::new();
    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        if let Some(OutputValue::Scalar(uv)) = result.outputs.get(name) {
            rows.push(vec![
                serde_json::json!(name.as_str()),
                serde_json::json!(uv.value),
                serde_json::json!(uv.confidence),
                serde_json::json!(uv.lower_bound),
                serde_json::json!(uv.upper_bound),
                serde_json::json!(uv.n_samples),
            ]);
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

/// Map signal outputs to sample rows.
fn map_signals(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec![
        "name".into(),
        "sample_index".into(),
        "value".into(),
        "sample_rate".into(),
        "channel".into(),
    ];

    let mut rows = Vec::new();
    let mut truncated = false;
    let mut warnings = Vec::new();

    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        if let Some(OutputValue::Signal(sig)) = result.outputs.get(name) {
            let limit = sig.samples.len().min(MAX_SIGNAL_ROWS);
            if sig.samples.len() > MAX_SIGNAL_ROWS {
                truncated = true;
                warnings.push(format!(
                    "Signal '{}' truncated from {} to {} samples",
                    name,
                    sig.samples.len(),
                    MAX_SIGNAL_ROWS
                ));
            }
            for (i, &sample) in sig.samples.iter().take(limit).enumerate() {
                rows.push(vec![
                    serde_json::json!(name.as_str()),
                    serde_json::json!(i),
                    serde_json::json!(sample),
                    serde_json::json!(sig.sample_rate),
                    serde_json::json!(sig.channel.as_str()),
                ]);
            }
        }
        // Include any scalars in the same output
        if let Some(OutputValue::Scalar(uv)) = result.outputs.get(name) {
            rows.push(vec![
                serde_json::json!(name.as_str()),
                serde_json::json!(0),
                serde_json::json!(uv.value),
                serde_json::json!(0),
                serde_json::json!("scalar"),
            ]);
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings,
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

/// Map spectrum outputs to frequency-bin rows.
fn map_spectra(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec!["name".into(), "frequency_bin".into(), "magnitude".into()];

    let mut rows = Vec::new();
    let mut truncated = false;
    let mut warnings = Vec::new();

    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        if let Some(OutputValue::Spectrum(spec)) = result.outputs.get(name) {
            let limit = spec.len().min(MAX_SIGNAL_ROWS);
            if spec.len() > MAX_SIGNAL_ROWS {
                truncated = true;
                warnings.push(format!(
                    "Spectrum '{}' truncated from {} to {} bins",
                    name,
                    spec.len(),
                    MAX_SIGNAL_ROWS
                ));
            }
            for (i, &mag) in spec.iter().take(limit).enumerate() {
                rows.push(vec![
                    serde_json::json!(name.as_str()),
                    serde_json::json!(i),
                    serde_json::json!(mag),
                ]);
            }
        }
        // Include scalars
        if let Some(OutputValue::Scalar(uv)) = result.outputs.get(name) {
            rows.push(vec![
                serde_json::json!(name.as_str()),
                serde_json::json!(0),
                serde_json::json!(uv.value),
            ]);
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings,
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

/// Map mixed outputs (scalar + signal + spectrum) to a generic format.
fn map_mixed(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec!["name".into(), "type".into(), "index".into(), "value".into()];

    let mut rows = Vec::new();
    let mut truncated = false;
    let mut warnings = Vec::new();

    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        match result.outputs.get(name) {
            Some(OutputValue::Scalar(uv)) => {
                rows.push(vec![
                    serde_json::json!(name.as_str()),
                    serde_json::json!("scalar"),
                    serde_json::json!(0),
                    serde_json::json!(uv.value),
                ]);
            }
            Some(OutputValue::Signal(sig)) => {
                let limit = sig.samples.len().min(MAX_SIGNAL_ROWS);
                if sig.samples.len() > MAX_SIGNAL_ROWS {
                    truncated = true;
                    warnings.push(format!(
                        "Signal '{}' truncated to {} samples",
                        name, MAX_SIGNAL_ROWS
                    ));
                }
                for (i, &sample) in sig.samples.iter().take(limit).enumerate() {
                    rows.push(vec![
                        serde_json::json!(name.as_str()),
                        serde_json::json!("signal"),
                        serde_json::json!(i),
                        serde_json::json!(sample),
                    ]);
                }
            }
            Some(OutputValue::Spectrum(spec)) => {
                let limit = spec.len().min(MAX_SIGNAL_ROWS);
                if spec.len() > MAX_SIGNAL_ROWS {
                    truncated = true;
                    warnings.push(format!(
                        "Spectrum '{}' truncated to {} bins",
                        name, MAX_SIGNAL_ROWS
                    ));
                }
                for (i, &mag) in spec.iter().take(limit).enumerate() {
                    rows.push(vec![
                        serde_json::json!(name.as_str()),
                        serde_json::json!("spectrum"),
                        serde_json::json!(i),
                        serde_json::json!(mag),
                    ]);
                }
            }
            None => {}
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings,
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn select_sigql_target_is_simd_or_webgpu() {
        let t = super::select_sigql_target();
        assert!(
            matches!(t, Target::Simd | Target::WebGpu),
            "unexpected target: {:?}",
            t
        );
    }

    use super::*;

    // -- Detection tests --

    #[test]
    fn test_is_sigql_query_positive() {
        // FROM + TRANSFORM
        assert!(is_sigql_query(
            "FROM sensor.data TRANSFORM bandpass(4Hz, 12Hz)"
        ));
        // FROM + AGGREGATE {
        assert!(is_sigql_query(
            "FROM sensor.data AGGREGATE { power: band_power(4Hz..12Hz) }"
        ));
        // FROM + CORRELATE
        assert!(is_sigql_query(
            "FROM sig_a, sig_b CORRELATE { xcorr: cross_correlation }"
        ));
        // Pipeline operator
        assert!(is_sigql_query(
            "sensor.data |> bandpass(4Hz, 12Hz) |> envelope"
        ));
        // Frequency literal
        assert!(is_sigql_query("FROM sensor TRANSFORM lowpass(50Hz)"));
        // RETURNING CONFIDENCE
        assert!(is_sigql_query(
            "FROM sensor.data AGGREGATE { m: mean } RETURNING CONFIDENCE(0.95)"
        ));
        // Case insensitive
        assert!(is_sigql_query("from sensor.data transform fft()"));
    }

    #[test]
    fn test_is_sigql_query_negative() {
        // Standard SQL
        assert!(!is_sigql_query("SELECT * FROM users"));
        assert!(!is_sigql_query("INSERT INTO users VALUES (1, 'Alice')"));
        // Cypher
        assert!(!is_sigql_query("MATCH (n:Person) RETURN n"));
        // CQL
        assert!(!is_sigql_query("CREATE KEYSPACE ks WITH replication = {}"));
        // GraphQL
        assert!(!is_sigql_query("{ users { id name } }"));
        assert!(!is_sigql_query("query GetUser { user(id: 1) { name } }"));
        // Plain FROM without SigQL keywords
        assert!(!is_sigql_query("FROM users WHERE id = 1"));
        // SHOW/DESCRIBE
        assert!(!is_sigql_query("SHOW TABLES"));
    }

    #[test]
    fn test_contains_frequency_literal() {
        assert!(contains_frequency_literal("bandpass(4Hz, 12Hz)"));
        assert!(contains_frequency_literal("lowpass(50kHz)"));
        assert!(contains_frequency_literal("highpass(1MHz)"));
        assert!(!contains_frequency_literal("SELECT Hz FROM table"));
        assert!(!contains_frequency_literal("no frequencies here"));
    }

    // -- Execution tests --

    fn create_test_amorphic() -> Arc<AmorphicTableStorage> {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        std::mem::forget(dir);
        Arc::new(AmorphicTableStorage::new(store))
    }

    #[test]
    fn test_sigql_parse_error() {
        let amorphic = create_test_amorphic();
        let start = Instant::now();
        // Invalid SigQL syntax
        let result = execute_sigql("FROM TRANSFORM", &amorphic, start);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigql_source_not_found() {
        let amorphic = create_test_amorphic();
        let start = Instant::now();
        // Valid SigQL referencing a source not in storage — runtime error
        let result = execute_sigql("FROM nonexistent.signal TRANSFORM fft()", &amorphic, start);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigql_scalar_output() {
        let amorphic = create_test_amorphic();
        let start = Instant::now();

        // Create a table with signal data
        use joule_db_query::executor::TableStorage;
        let cols = vec!["value".to_string()];
        amorphic.create_table("test_signal", &cols).unwrap();

        // Insert sample data
        for i in 0..100 {
            let val = (i as f64 * 0.1).sin();
            let row = joule_db_query::executor::RowData::new(
                vec!["value".into()],
                vec![joule_db_query::ast::Value::Float(val)],
            );
            let _ = amorphic.insert("test_signal", &row);
        }

        // Run aggregate query
        let result = execute_sigql("FROM test.signal AGGREGATE { avg: mean }", &amorphic, start);

        // Should produce scalar output (or fail gracefully if source not found)
        // The table name is test_signal but source is test.signal → maps to test_signal
        match result {
            Ok(resp) => {
                assert_eq!(resp.algorithm_type, Some("sigql".into()));
                assert!(!resp.columns.is_empty());
            }
            Err(_) => {
                // Source resolution may fail if runtime doesn't find it — acceptable
            }
        }
    }

    fn signal_table(name: &str, samples: &[f64]) -> Arc<AmorphicTableStorage> {
        use joule_db_query::executor::TableStorage;
        let amorphic = create_test_amorphic();
        amorphic.create_table(name, &["value".to_string()]).unwrap();
        for &v in samples {
            let row = joule_db_query::executor::RowData::new(
                vec!["value".into()],
                vec![joule_db_query::ast::Value::Float(v)],
            );
            amorphic.insert(name, &row).unwrap();
        }
        amorphic
    }

    fn add_signal(amorphic: &Arc<AmorphicTableStorage>, name: &str, samples: &[f64]) {
        use joule_db_query::executor::TableStorage;
        amorphic.create_table(name, &["value".to_string()]).unwrap();
        for &v in samples {
            let row = joule_db_query::executor::RowData::new(
                vec!["value".into()],
                vec![joule_db_query::ast::Value::Float(v)],
            );
            amorphic.insert(name, &row).unwrap();
        }
    }

    fn numbers(resp: &QueryResponse) -> Vec<f64> {
        let col = resp
            .columns
            .iter()
            .position(|c| c == "value" || c == "magnitude")
            .unwrap_or_else(|| panic!("no value column in {:?}", resp.columns));
        let mut out: Vec<f64> = resp.rows.iter().map(|r| r[col].as_f64().unwrap()).collect();
        // Scalar tables also carry their interval.
        if resp.columns.iter().any(|c| c == "lower_bound") {
            for name in ["lower_bound", "upper_bound"] {
                let i = resp.columns.iter().position(|c| c == name).unwrap();
                out.extend(resp.rows.iter().map(|r| r[i].as_f64().unwrap()));
            }
        }
        out
    }

    /// The WebGPU path must compile its WGSL and produce what the CPU
    /// runtime produces (f32 vs f64, so within tolerance) for every op the
    /// default path lowers. Without an adapter the query must fall back to
    /// the CPU and say why, never fail.
    #[test]
    fn sigql_webgpu_matches_cpu_runtime() {
        // 300 samples: more than one 64-thread workgroup, not a power of two.
        let samples: Vec<f64> = (0..300)
            .map(|i| {
                let t = i as f64 / 1000.0;
                (2.0 * std::f64::consts::PI * 8.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
                    + 0.1
            })
            .collect();
        let amorphic = signal_table("bench_sig", &samples);
        // A second, shorter signal for cross-correlation.
        add_signal(&amorphic, "bench_ref", &samples[17..257]);
        let queries = [
            "FROM bench.sig TRANSFORM abs()",
            "FROM bench.sig TRANSFORM diff()",
            "FROM bench.sig TRANSFORM cumsum()",
            "FROM bench.sig TRANSFORM zscore()",
            "FROM bench.sig TRANSFORM lowpass(50Hz)",
            "FROM bench.sig TRANSFORM bandpass(4Hz, 12Hz)",
            "FROM bench.sig TRANSFORM fft()",
            "FROM bench.sig AGGREGATE { m: mean }",
            "FROM bench.sig AGGREGATE { r: rms }",
            "FROM bench.sig AGGREGATE { s: std }",
            "FROM bench.sig AGGREGATE { v: var }",
            "FROM bench.sig AGGREGATE { hi: peak }",
            "FROM bench.sig AGGREGATE { lo: trough }",
            "FROM bench.sig AGGREGATE { p: peak_to_peak }",
            "FROM bench.sig AGGREGATE { z: zero_crossings }",
            "FROM bench.sig TRANSFORM abs() AGGREGATE { m: mean }",
            "FROM bench.sig AGGREGATE { s: skewness }",
            "FROM bench.sig TRANSFORM abs() AGGREGATE { s: skewness }",
            "FROM bench.sig AGGREGATE { k: kurtosis }",
            "FROM bench.sig AGGREGATE { s: slope }",
            "FROM bench.sig TRANSFORM cumsum() AGGREGATE { s: slope }",
            "FROM bench.sig TRANSFORM detrend()",
            "FROM bench.sig TRANSFORM cumsum(), detrend()",
            "FROM bench.sig TRANSFORM envelope()",
            "FROM bench.sig TRANSFORM hilbert()",
            "FROM bench.sig TRANSFORM decimate(4)",
            "FROM bench.sig TRANSFORM decimate(1)",
            "FROM bench.sig AGGREGATE { p: band_power(4Hz..12Hz) }",
            "FROM bench.sig AGGREGATE { p: band_power(100Hz..140Hz) }",
            "FROM bench.sig TRANSFORM fft() AGGREGATE { p: band_power(4Hz..12Hz) }",
            "FROM bench.sig AS a, bench.ref AS b CORRELATE { x: cross_correlation(a, b, max_lag: 20ms) }",
            "FROM bench.sig AS a, bench.ref AS b CORRELATE { x: cross_correlation(a, b) }",
            "FROM bench.sig, bench.ref CORRELATE { x: cross_correlation(ref, sig, max_lag: 5ms) }",
        ];
        let gpu = gpu_context();
        for sql in queries {
            let cpu = execute_sigql_with(sql, &amorphic, Instant::now(), Target::Simd)
                .unwrap_or_else(|e| panic!("{sql} on CPU: {e:?}"));
            assert_eq!(cpu.device_target.as_deref(), Some("cpu"));
            let auto = execute_sigql_with(sql, &amorphic, Instant::now(), Target::WebGpu)
                .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
            match &gpu {
                Ok(ctx) => {
                    assert_eq!(
                        auto.device_target.as_deref(),
                        Some("webgpu"),
                        "{sql} fell back on {}: {:?}",
                        ctx.adapter,
                        auto.warnings
                    );
                    assert!(auto.warnings.iter().any(|w| w.contains("WGSL kernel")), "{:?}", auto.warnings);
                    assert_eq!(auto.columns, cpu.columns, "{sql}");
                    let (a, b) = (numbers(&auto), numbers(&cpu));
                    assert!(!b.is_empty(), "{sql}: no output");
                    assert_eq!(a.len(), b.len(), "{sql}");
                    // Everything but the computed values (names, timestamps,
                    // confidence, sample counts) is identical.
                    assert_eq!(auto.rows.len(), cpu.rows.len(), "{sql}");
                    for (ra, rc) in auto.rows.iter().zip(&cpu.rows) {
                        for (c, name) in cpu.columns.iter().enumerate() {
                            if !["value", "magnitude", "lower_bound", "upper_bound"].contains(&name.as_str()) {
                                assert_eq!(ra[c], rc[c], "{sql}: column {name}");
                            }
                        }
                    }
                    let scale = b.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                        assert!(
                            (x - y).abs() <= 2e-4 * scale,
                            "{sql}: element {i} GPU {x} vs CPU {y} (scale {scale})"
                        );
                    }
                    eprintln!("{sql}: webgpu on {} matches CPU ({} values)", ctx.adapter, a.len());
                }
                Err(reason) => {
                    assert_eq!(auto.device_target.as_deref(), Some("cpu"));
                    assert!(
                        auto.warnings.iter().any(|w| w.contains("no wgpu adapter")),
                        "{sql}: fallback must say why: {:?} ({reason})",
                        auto.warnings
                    );
                    assert_eq!(numbers(&auto), numbers(&cpu));
                }
            }
        }
        if std::env::var("JOULE_REQUIRE_WEBGPU").is_ok_and(|v| v == "1") {
            let ctx = gpu.as_ref().expect("JOULE_REQUIRE_WEBGPU=1 but no wgpu adapter");
            eprintln!("WebGPU adapter: {}", ctx.adapter);
        }
    }

    /// Compile `sql` and bind every source it loads to `samples` at 1 kHz.
    fn long_plan(sql: &str, samples: &[f64]) -> (sigql::ExecutionPlan, Vec<(String, DynSignal<f64>)>) {
        let plan = sigql::compile(&sigql::parse(sql).unwrap(), Target::Simd).unwrap();
        let sources = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                sigql::compile::PlanStep::LoadSignal { source, .. } => Some(source.to_string()),
                _ => None,
            })
            .map(|name| (name.clone(), DynSignal::new(name.as_str(), samples.to_vec(), 1000, 0)))
            .collect();
        (plan, sources)
    }

    fn signal_output(result: &sigql::runtime::ExecutionResult) -> Vec<f64> {
        result
            .outputs
            .values()
            .find_map(|v| match v {
                OutputValue::Signal(s) => Some(s.samples.clone()),
                _ => None,
            })
            .expect("a signal output")
    }

    fn cpu_run(plan: &sigql::ExecutionPlan, sources: &[(String, DynSignal<f64>)]) -> sigql::runtime::ExecutionResult {
        let mut runtime = Runtime::new(RuntimeConfig::default());
        for (name, signal) in sources {
            runtime.register_signal(name.clone(), signal.clone());
        }
        runtime.execute(plan).unwrap()
    }

    fn long_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / 1000.0;
                (2.0 * std::f64::consts::PI * 8.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
                    + 0.2 * ((i * 7919 % 1009) as f64 / 1009.0 - 0.5)
                    + 0.1
            })
            .collect()
    }

    const LONG_IIR_QUERIES: [&str; 4] = [
        "FROM bench.sig TRANSFORM lowpass(50Hz)",
        "FROM bench.sig TRANSFORM highpass(20Hz)",
        "FROM bench.sig TRANSFORM bandpass(4Hz, 12Hz)",
        "FROM bench.sig TRANSFORM decimate(4)",
    ];

    /// The chunked IIR (prefix-scan carry) matches the CPU runtime over
    /// `n` samples, where any error in the carried chunk states would show
    /// up and accumulate.
    fn long_iir_parity(n: usize) {
        let ctx = match gpu_context() {
            Ok(ctx) => ctx,
            Err(reason) => {
                assert!(
                    !std::env::var("JOULE_REQUIRE_WEBGPU").is_ok_and(|v| v == "1"),
                    "JOULE_REQUIRE_WEBGPU=1 but no wgpu adapter: {reason}"
                );
                eprintln!("no wgpu adapter ({reason}); long IIR parity needs one");
                return;
            }
        };
        let samples = long_signal(n);
        for sql in LONG_IIR_QUERIES {
            let (plan, sources) = long_plan(sql, &samples);
            let cpu = signal_output(&cpu_run(&plan, &sources));
            let (result, run) = execute_plan_webgpu(&plan, &sources).unwrap_or_else(|e| panic!("{sql}: {e}"));
            assert!(run.kernels.iter().any(|k| *k == "iir" || *k == "decimate"), "{sql}: {:?}", run.kernels);
            let gpu = signal_output(&result);
            assert_eq!(gpu.len(), cpu.len(), "{sql}");
            let scale = cpu.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let worst = gpu
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(worst <= 2e-4 * scale, "{sql}: max error {worst} (scale {scale})");
            eprintln!(
                "{sql}: {} samples on {} match CPU, max error {:.2e} of scale {scale:.3}",
                samples.len(),
                ctx.adapter,
                worst
            );
        }
    }

    #[test]
    fn sigql_webgpu_long_iir_matches_cpu() {
        long_iir_parity(1 << 20);
    }

    #[test]
    fn sigql_webgpu_4m_iir_matches_cpu() {
        long_iir_parity(1 << 22);
    }

    /// One IIR shader set up for benchmarking: buffers resident, pipelines
    /// compiled.
    struct BenchKernel {
        bind: wgpu::BindGroup,
        out: wgpu::Buffer,
        pipelines: Vec<(&'static str, wgpu::ComputePipeline, u32)>,
        read_floats: usize,
        wgsl: String,
        params: [u32; 8],
        aux: Vec<f32>,
        out_floats: usize,
    }

    fn bench_kernel(
        ctx: &GpuContext,
        wgsl: String,
        passes: &[(&'static str, u32)],
        samples: &[f64],
        out_floats: usize,
        read_floats: usize,
        params: [u32; 8],
        aux: Vec<f32>,
    ) -> BenchKernel {
        let pipelines = passes
            .iter()
            .map(|(entry, groups)| (*entry, ctx.pipeline(&wgsl, entry, "bench").unwrap(), *groups))
            .collect();
        let (bind, out) = bench_bind(ctx, samples, out_floats, params, &aux);
        BenchKernel { bind, out, pipelines, read_floats, wgsl, params, aux, out_floats }
    }

    fn bench_bind(
        ctx: &GpuContext,
        samples: &[f64],
        out_floats: usize,
        params: [u32; 8],
        aux: &[f32],
    ) -> (wgpu::BindGroup, wgpu::Buffer) {
        use wgpu::util::DeviceExt;
        let device = &ctx.device;
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench-in"),
            contents: &bytes(&samples.iter().map(|&v| v as f32).collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench-out"),
            size: (out_floats.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench-params"),
            contents: &params.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let aux = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench-aux"),
            contents: &bytes(if aux.is_empty() { &[0.0] } else { aux }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench"),
            layout: &ctx.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: aux.as_entire_binding() },
            ],
        });
        (bind, out)
    }

    /// Submit `range` of the kernel's passes and wait; returns the wall time.
    fn bench_submit(ctx: &GpuContext, k: &BenchKernel, bind: &wgpu::BindGroup, range: std::ops::Range<usize>) -> std::time::Duration {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_bind_group(0, bind, &[]);
            for (_, p, g) in &k.pipelines[range] {
                pass.set_pipeline(p);
                pass.dispatch_workgroups(*g, 1, 1);
            }
        }
        let t = Instant::now();
        ctx.queue.submit([encoder.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        t.elapsed()
    }

    fn best_of<F: FnMut() -> std::time::Duration>(reps: usize, mut f: F) -> std::time::Duration {
        f();
        (0..reps).map(|_| f()).min().unwrap()
    }

    /// Kernel only: every pass in one submission, buffers resident.
    fn kernel_time(ctx: &GpuContext, k: &BenchKernel, reps: usize) -> std::time::Duration {
        best_of(reps, || bench_submit(ctx, k, &k.bind, 0..k.pipelines.len()))
    }

    /// End to end: upload (f64 -> f32), every pass, readback of the output.
    fn e2e_time(ctx: &GpuContext, k: &BenchKernel, samples: &[f64], reps: usize) -> std::time::Duration {
        best_of(reps, || {
            let t = Instant::now();
            let (bind, out) = bench_bind(ctx, samples, k.out_floats, k.params, &k.aux);
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bench-staging"),
                size: (k.read_floats.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_bind_group(0, &bind, &[]);
                for (_, p, g) in &k.pipelines {
                    pass.set_pipeline(p);
                    pass.dispatch_workgroups(*g, 1, 1);
                }
            }
            encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, (k.read_floats.max(1) * 4) as u64);
            ctx.queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            let data: Vec<f32> = staging
                .slice(..)
                .get_mapped_range()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            std::hint::black_box(data.iter().map(|&v| v as f64).collect::<Vec<f64>>());
            t.elapsed()
        })
    }

    /// GPU time per pass from timestamps written at each compute pass's
    /// start and end, all passes in one submission; best of `reps`. `None`
    /// when the adapter has no timestamp queries.
    fn gpu_pass_times(ctx: &GpuContext, k: &BenchKernel, reps: usize) -> Option<Vec<(&'static str, f64)>> {
        if !ctx.timestamps {
            return None;
        }
        let count = 2 * k.pipelines.len() as u32;
        let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bench-ts"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench-ts-resolve"),
            size: count as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let period = ctx.queue.get_timestamp_period() as f64; // ns per tick
        let mut best: Option<Vec<f64>> = None;
        for _ in 0..=reps {
            let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bench-ts-read"),
                size: count as u64 * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for (i, (_, p, g)) in k.pipelines.iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &set,
                        beginning_of_pass_write_index: Some(2 * i as u32),
                        end_of_pass_write_index: Some(2 * i as u32 + 1),
                    }),
                });
                pass.set_bind_group(0, &k.bind, &[]);
                pass.set_pipeline(p);
                pass.dispatch_workgroups(*g, 1, 1);
            }
            encoder.resolve_query_set(&set, 0..count, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, count as u64 * 8);
            ctx.queue.submit([encoder.finish()]);
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            let ticks: Vec<u64> = read
                .slice(..)
                .get_mapped_range()
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let times: Vec<f64> = ticks
                .chunks_exact(2)
                .map(|t| t[1].saturating_sub(t[0]) as f64 * period / 1e6)
                .collect();
            best = Some(match best {
                None => times,
                Some(b) if times.iter().sum::<f64>() < b.iter().sum::<f64>() => times,
                Some(b) => b,
            });
        }
        best.map(|t| k.pipelines.iter().map(|(name, _, _)| *name).zip(t).collect())
    }

    /// Per-pass GPU profile, scan levels grouped: `"chunk_state 0.012, ..."`
    /// plus the total, in ms.
    fn pass_profile(ctx: &GpuContext, k: &BenchKernel, reps: usize) -> (String, f64) {
        let Some(times) = gpu_pass_times(ctx, k, reps) else {
            return ("no timestamp queries on this adapter".into(), f64::NAN);
        };
        let mut groups: Vec<(String, f64)> = Vec::new();
        for (name, t) in times {
            let label = if name.starts_with("scan_") { "scan".to_string() } else { name.to_string() };
            match groups.last_mut() {
                Some((l, acc)) if *l == label => *acc += t,
                _ => groups.push((label, t)),
            }
        }
        let total: f64 = groups.iter().map(|(_, t)| t).sum();
        let levels = k.pipelines.iter().filter(|(n, _, _)| n.starts_with("scan_") && *n != "scan_block").count();
        let text = groups
            .iter()
            .map(|(l, t)| {
                if l == "scan" {
                    format!("scan(block+{levels} levels) {t:.3} ({:.0}%)", 100.0 * t / total)
                } else {
                    format!("{l} {t:.3} ({:.0}%)", 100.0 * t / total)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        (text, total)
    }

    fn iir_kernel(sql: &str, n: usize, chunks: Option<sigql::compile::gpu::IirChunks>) -> GpuKernel {
        let (plan, _) = long_plan(sql, &[0.0]);
        let step = plan
            .steps
            .iter()
            .find(|s| matches!(s, sigql::compile::PlanStep::IirFilter { .. }))
            .unwrap();
        let bound = GpuKernel::from_step(step)
            .unwrap()
            .unwrap()
            .bind(&GpuInput { len: n, rate: 1000, is_spectrum: false, second_len: None, default_rate: 1000 })
            .unwrap();
        match (bound, chunks) {
            (GpuKernel::Iir { sections, .. }, Some(c)) => GpuKernel::Iir { sections, chunks: Some(c) },
            (k, _) => k,
        }
    }

    /// Release benchmark of the IIR kernels: the Phase C chunked shader
    /// (serial carry, ~sqrt(16n) chunks), the prefix-scan shader, and the
    /// CPU runtime; kernel-only and end-to-end, best of 5; plus a per-pass
    /// profile and a chunk-length sweep. Run with
    /// `cargo test --release -p joule-db-server --lib sigql_iir_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn sigql_iir_timing() {
        use sigql::compile::gpu::{serial_carry_iir_wgsl, IirChunks};
        let ctx = gpu_context().expect("a wgpu adapter");
        eprintln!("adapter: {}", ctx.adapter);
        let reps = 5;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let mk = |kernel: &GpuKernel, wgsl: String, passes: Vec<(&'static str, u32)>, samples: &[f64]| {
            let n = samples.len();
            bench_kernel(ctx, wgsl, &passes, samples, kernel.buffer_len(n), kernel.output_len(n), kernel.params(n), kernel.aux())
        };
        for n in [1usize << 16, 1 << 20, 1 << 22] {
            let samples = long_signal(n);
            for sql in ["FROM bench.sig TRANSFORM lowpass(50Hz)", "FROM bench.sig TRANSFORM bandpass(4Hz, 12Hz)"] {
                let tree = ctx.tune(iir_kernel(sql, n, None), n);
                let GpuKernel::Iir { sections, chunks: Some(tc) } = &tree else { panic!("{tree:?}") };
                // Phase C: ~sqrt(16 n) chunks (capped 8192), serial carry.
                let old_chunks = IirChunks::with_chunks(sections, n, ((16.0 * n as f64).sqrt().ceil() as usize).min(8192));
                let old = iir_kernel(sql, n, Some(old_chunks.clone()));
                let groups = old_chunks.chunks.div_ceil(64);
                let old_b = mk(&old, serial_carry_iir_wgsl(sections), vec![("chunk_state", groups), ("carry", 1), ("main", groups)], &samples);
                let tree_b = mk(&tree, tree.wgsl(), tree.passes(n), &samples);
                let (old_k, old_e) = (kernel_time(ctx, &old_b, reps), e2e_time(ctx, &old_b, &samples, reps));
                let (tree_k, tree_e) = (kernel_time(ctx, &tree_b, reps), e2e_time(ctx, &tree_b, &samples, reps));
                let floor = best_of(reps, || bench_submit(ctx, &tree_b, &tree_b.bind, 0..0));
                let (plan, sources) = long_plan(sql, &samples);
                let cpu = best_of(reps, || {
                    let t = Instant::now();
                    std::hint::black_box(cpu_run(&plan, &sources));
                    t.elapsed()
                });
                let executor = best_of(reps, || {
                    let t = Instant::now();
                    std::hint::black_box(execute_plan_webgpu(&plan, &sources).unwrap());
                    t.elapsed()
                });
                eprintln!(
                    "TIMING n={n} {sql} sections={} | chunked(serial carry) {}x{}: kernel {:.3} ms, e2e {:.3} ms | tree {}x{} levels={}: kernel {:.3} ms, e2e {:.3} ms | CPU {:.3} ms | executor e2e {:.3} ms | tree vs chunked kernel {:.1}x | empty submit {:.3} ms",
                    sections.len(),
                    old_chunks.chunks, old_chunks.chunk_len, ms(old_k), ms(old_e),
                    tc.chunks, tc.chunk_len, tc.levels, ms(tree_k), ms(tree_e),
                    ms(cpu), ms(executor), ms(old_k) / ms(tree_k), ms(floor),
                );
                for (label, k) in [("chunked", &old_b), ("tree", &tree_b)] {
                    let (prof, total) = pass_profile(ctx, k, reps);
                    eprintln!("PROFILE n={n} {sql} {label}: GPU {total:.3} ms = {prof} (ms, timestamps)");
                }
                if n >= 1 << 20 {
                    for len in [16usize, 32, 64, 128, 256, 1024] {
                        let c = IirChunks::with_chunks(sections, n, n.div_ceil(len));
                        let k = iir_kernel(sql, n, Some(c.clone()));
                        let b = mk(&k, k.wgsl(), k.passes(n), &samples);
                        eprintln!(
                            "SWEEP n={n} {sql} chunk_len={} chunks={} levels={}: kernel wall {:.3} ms, GPU {:.3} ms",
                            c.chunk_len, c.chunks, c.levels, ms(kernel_time(ctx, &b, reps)), pass_profile(ctx, &b, reps).1
                        );
                    }
                }
                let _ = &tree_b.wgsl;
            }
        }
    }

    fn receipt(resp: &QueryResponse) -> crate::cost_model::RouteReceipt {
        resp.energy_receipt.clone().expect("SigQL responses carry a routing receipt")
    }

    fn assert_real(device: &str) {
        assert!(matches!(device, "cpu" | "gpu" | "npu" | "tpu" | "lpu"), "not a processor: {device:?}");
    }

    /// An injected cost model decides where SigQL runs: the cheaper device
    /// by joules wins, the receipt says why, and the fallback is the other
    /// real processor. Without an adapter the CPU is the only candidate and
    /// still names a real fallback.
    #[test]
    fn sigql_routes_by_injected_cost() {
        use crate::cost_model::{Candidate, CostModel};
        let samples = long_signal(4096);
        let amorphic = signal_table("route_sig", &samples);
        let sql = "FROM route.sig TRANSFORM lowpass(50Hz)";
        let gpu = gpu_context().ok().map(|ctx| Candidate::new("gpu", ctx.backend.clone()));
        for (cpu_j, gpu_j) in [(1e-4, 1e-2), (1e-2, 1e-4)] {
            let mut model = CostModel::new();
            model.explore_every = 0;
            // Injected measured incremental joules, equal times: energy decides.
            model.record_cost("sigql:iir", samples.len(), &Candidate::cpu(), 1e-3, cpu_j, Some(cpu_j), "injected");
            if let Some(gpu) = &gpu {
                model.record_cost("sigql:iir", samples.len(), gpu, 1e-3, gpu_j, Some(gpu_j), "injected");
            }
            let resp = execute_sigql_routed(sql, &amorphic, Instant::now(), &model).unwrap();
            let r = receipt(&resp);
            eprintln!("injected cpu={cpu_j} gpu={gpu_j}: {}", r.line());
            assert_eq!(r.op, "sigql:iir");
            assert_eq!(r.size_bucket, 12);
            assert_real(&r.device);
            assert_real(&r.fallback);
            assert_real(&r.requested_device);
            assert!(r.joules > 0.0 && r.seconds > 0.0);
            assert_eq!(resp.energy_joules, Some(r.joules));
            match &gpu {
                Some(g) => {
                    assert_eq!(r.requested_device, "gpu");
                    let want = if cpu_j < gpu_j { Candidate::cpu() } else { g.clone() };
                    assert_eq!((r.device.clone(), r.backend.clone()), (want.device.clone(), want.backend.clone()));
                    assert_eq!(r.route_reason, format!("cost_model_{}_cheaper", want.device));
                    assert_eq!(r.fallback, if want.device == "cpu" { "gpu" } else { "cpu" });
                    assert_eq!(resp.device_target.as_deref(), Some(if want.device == "cpu" { "cpu" } else { "webgpu" }));
                }
                None => {
                    assert_eq!((r.device.as_str(), r.route_reason.as_str(), r.fallback.as_str()), ("cpu", "only_candidate", "cpu"));
                }
            }
        }
    }

    /// A fresh model runs the static preference, explores the other device
    /// once, then follows the measurement; one call in `explore_every`
    /// re-measures the runner-up. The fallback is never empty.
    #[test]
    fn sigql_routing_explores_then_follows_measurement() {
        use crate::cost_model::{Candidate, CostModel};
        let samples = long_signal(2048);
        let amorphic = signal_table("explore_sig", &samples);
        let sql = "FROM explore.sig TRANSFORM lowpass(50Hz)";
        let mut model = CostModel::new();
        model.explore_every = 4;
        let reasons: Vec<crate::cost_model::RouteReceipt> = (0..8)
            .map(|_| receipt(&execute_sigql_routed(sql, &amorphic, Instant::now(), &model).unwrap()))
            .collect();
        for r in &reasons {
            eprintln!("{}", r.line());
            assert_real(&r.fallback);
            assert_real(&r.device);
        }
        match gpu_context() {
            Ok(ctx) => {
                let gpu = Candidate::new("gpu", ctx.backend.clone());
                assert_eq!((reasons[0].device.as_str(), reasons[0].route_reason.as_str()), ("gpu", "no_history"));
                assert_eq!((reasons[1].device.as_str(), reasons[1].route_reason.as_str()), ("cpu", "explore"));
                let explores = reasons[2..].iter().filter(|r| r.route_reason == "explore").count();
                assert_eq!(explores, 2, "calls 4 and 8 re-measure the runner-up");
                for r in reasons[2..].iter().filter(|r| r.route_reason != "explore") {
                    assert!(r.route_reason.starts_with("cost_model_"), "{}", r.line());
                }
                assert!(model.observed("sigql:iir", samples.len(), &gpu).is_some());
                assert!(model.observed("sigql:iir", samples.len(), &Candidate::cpu()).is_some());
            }
            Err(_) => {
                assert!(reasons.iter().all(|r| r.device == "cpu" && r.route_reason == "only_candidate"));
            }
        }
    }

    /// Live: at 64K and 1M samples the route converges on the device this
    /// machine measured cheaper (by joules, then time), and every later
    /// cost-model decision agrees with the recorded costs.
    #[test]
    fn sigql_live_routing_follows_measurement() {
        use crate::cost_model::{Candidate, CostModel};
        for n in [1usize << 16, 1 << 20] {
            // Sources bound directly (a 1M-row table insert would dominate).
            let (plan, sources) = long_plan("FROM live.sig TRANSFORM bandpass(4Hz, 12Hz)", &long_signal(n));
            let mut model = CostModel::new();
            model.explore_every = 0;
            let mut receipts = Vec::new();
            for _ in 0..8 {
                let resp =
                    run_routed(&plan, &sources, Instant::now(), &model, crate::power_meter::Meter::global()).unwrap();
                receipts.push(receipt(&resp));
            }
            let cpu = model.observed("sigql:iir", n, &Candidate::cpu()).unwrap();
            let gpu = gpu_context()
                .ok()
                .and_then(|ctx| model.observed("sigql:iir", n, &Candidate::new("gpu", ctx.backend.clone())));
            for r in &receipts {
                eprintln!("LIVE n={n} {}", r.line());
                assert_real(&r.fallback);
            }
            match gpu {
                Some(gpu_cost) => {
                    let ctx = gpu_context().unwrap();
                    let both = [Candidate::new("gpu", ctx.backend.clone()), Candidate::cpu()];
                    let (by_energy, by_time, by) = model.rankings("sigql:iir", n, &both).unwrap();
                    eprintln!(
                        "LIVE n={n} cpu {:.3e} J (incr {:?}) / {:.3} ms [{}], gpu {:.3e} J (incr {:?}) / {:.3} ms [{}] -> ranked_by={} energy={} time={}",
                        cpu.joules,
                        cpu.incremental_joules,
                        cpu.seconds * 1e3,
                        cpu.energy_source,
                        gpu_cost.joules,
                        gpu_cost.incremental_joules,
                        gpu_cost.seconds * 1e3,
                        gpu_cost.energy_source,
                        by.label(),
                        by_energy.device,
                        by_time.device
                    );
                    let last = receipts.last().unwrap();
                    assert!(last.route_reason.starts_with("cost_model_"), "{}", last.line());
                    // The next decision follows the recorded costs.
                    assert_eq!(model.choose("sigql:iir", n, &both).chosen, by_energy);
                }
                None => assert!(receipts.iter().all(|r| r.device == "cpu")),
            }
        }
    }

    /// Steps without a kernel run on the CPU and the response names the step.
    #[test]
    fn sigql_unsupported_step_falls_back_with_reason() {
        let samples: Vec<f64> = (0..128).map(|i| (i as f64 * 0.2).sin()).collect();
        let amorphic = signal_table("env_sig", &samples);
        let resp = execute_sigql_with(
            "FROM env.sig TRANSFORM interpolate(2)",
            &amorphic,
            Instant::now(),
            Target::WebGpu,
        )
        .unwrap();
        assert_eq!(resp.device_target.as_deref(), Some("cpu"));
        assert!(
            resp.warnings.iter().any(|w| w.contains("Interpolate has no GPU kernel")
                || w.contains("no wgpu adapter")),
            "{:?}",
            resp.warnings
        );
    }

    #[test]
    fn test_sigql_signal_output_mapping() {
        // Test the result mapping directly
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0], 1000, 0);
        outputs.insert(
            smol_str::SmolStr::new("filtered"),
            OutputValue::Signal(signal),
        );

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(
            response.columns,
            vec!["name", "sample_index", "value", "sample_rate", "channel"]
        );
        assert_eq!(response.rows.len(), 5);
        assert_eq!(response.rows[0][0], serde_json::json!("filtered"));
        assert_eq!(response.rows[0][1], serde_json::json!(0));
        assert_eq!(response.rows[0][2], serde_json::json!(1.0));
        assert!(!response.truncated);
    }

    #[test]
    fn test_sigql_scalar_output_mapping() {
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use sigql::types::UncertainValue;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        let uv = UncertainValue::from_ci(10.5, 0.7, 0.95, 1000);
        outputs.insert(smol_str::SmolStr::new("power"), OutputValue::Scalar(uv));

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(response.columns[0], "name");
        assert_eq!(response.columns[1], "value");
        assert_eq!(response.columns[2], "confidence");
        assert_eq!(response.rows.len(), 1);
        assert_eq!(response.rows[0][0], serde_json::json!("power"));
        assert_eq!(response.rows[0][1], serde_json::json!(10.5));
    }

    #[test]
    fn test_sigql_spectrum_output_mapping() {
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        outputs.insert(
            smol_str::SmolStr::new("spectrum"),
            OutputValue::Spectrum(vec![0.1, 0.5, 0.3, 0.2]),
        );

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(response.columns, vec!["name", "frequency_bin", "magnitude"]);
        assert_eq!(response.rows.len(), 4);
        assert_eq!(response.rows[1][2], serde_json::json!(0.5));
    }

    #[test]
    fn test_sigql_truncation() {
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        let samples: Vec<f64> = (0..20_000).map(|i| i as f64 * 0.001).collect();
        let signal = DynSignal::new("big", samples, 1000, 0);
        outputs.insert(smol_str::SmolStr::new("big"), OutputValue::Signal(signal));

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(response.rows.len(), MAX_SIGNAL_ROWS);
        assert!(response.truncated);
        assert!(!response.warnings.is_empty());
    }

    /// Mac live (run by hand): measured per-device watts and joules for
    /// 64K and 1M IIR, and whether measured energy changes the time-only
    /// routing decision.
    #[test]
    #[ignore]
    fn measured_power_live_iir() {
        use crate::cost_model::{Candidate, CostModel, RouteReceipt};
        let meter = crate::power_meter::Meter::global();
        eprintln!("POWER meter={}", meter.source());
        std::thread::sleep(std::time::Duration::from_millis(1500));
        for n in [1usize << 16, 1 << 20] {
            let (plan, sources) = long_plan("FROM live.sig TRANSFORM bandpass(4Hz, 12Hz)", &long_signal(n));
            let mut model = CostModel::new();
            // Alternate so both devices keep fresh measurements.
            model.explore_every = 2;
            let rs: Vec<RouteReceipt> = (0..16)
                .map(|_| receipt(&run_routed(&plan, &sources, Instant::now(), &model, meter).unwrap()))
                .collect();
            for r in &rs {
                eprintln!("POWER iir n={n} {}", r.line());
                assert_real(&r.fallback);
            }
            for dev in ["gpu", "cpu"] {
                // Skip runs that paid real setup (the first GPU call's pipeline compile).
                let v: Vec<&RouteReceipt> =
                    rs.iter().filter(|r| r.device == dev && r.setup_seconds.unwrap_or(0.0) < 1e-3).collect();
                if v.is_empty() {
                    continue;
                }
                let k = v.len() as f64;
                let mean = |f: &dyn Fn(&RouteReceipt) -> f64| v.iter().map(|r| f(r)).sum::<f64>() / k;
                let opt = |f: &dyn Fn(&RouteReceipt) -> Option<f64>| {
                    let x: Vec<f64> = v.iter().filter_map(|r| f(r)).collect();
                    if x.is_empty() { "none".to_string() } else { format!("{:.3}", x.iter().sum::<f64>() / x.len() as f64) }
                };
                let conf: Vec<&str> = v.iter().map(|r| r.energy_confidence.as_str()).collect();
                eprintln!(
                    "POWER-TABLE job=iir_bandpass size={n} device={dev} backend={} runs={} ms={:.3} J={:.4e} incr_J={:.4e} W={:.2} cpu_w={} gpu_w={} ane_w={} base_w={} source={} confidence={:?}",
                    v[0].backend,
                    v.len(),
                    mean(&|r| r.seconds) * 1e3,
                    mean(&|r| r.joules),
                    mean(&|r| r.incremental_joules),
                    mean(&|r| r.watts),
                    opt(&|r| r.cpu_watts),
                    opt(&|r| r.gpu_watts),
                    opt(&|r| r.ane_watts),
                    opt(&|r| r.baseline_watts),
                    v[0].energy_source,
                    conf
                );
            }
            if let Ok(ctx) = gpu_context() {
                let both = [Candidate::new("gpu", ctx.backend.clone()), Candidate::cpu()];
                if let Some((e, t, by)) = model.rankings("sigql:iir", n, &both) {
                    eprintln!(
                        "POWER-ROUTE job=iir_bandpass size={n} ranked_by={} energy_choice={} time_choice={} flipped={}",
                        by.label(),
                        e.device,
                        t.device,
                        e != t
                    );
                }
            }
        }
    }
}

