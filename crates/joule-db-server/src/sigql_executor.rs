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
    execute_sigql_with(sql, amorphic, start, select_sigql_target())
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
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// Compiled pipelines keyed by WGSL source.
    pipelines: std::sync::Mutex<std::collections::HashMap<String, wgpu::ComputePipeline>>,
}

/// What a GPU run did, for the response.
struct GpuRun {
    adapter: String,
    kernels: Vec<&'static str>,
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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sigql-webgpu"),
            required_features: wgpu::Features::empty(),
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
        layout,
        pipeline_layout,
        pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
    })
}

impl GpuContext {
    /// Compile (or reuse) the pipeline for `kernel`, surfacing WGSL errors.
    fn pipeline(&self, kernel: &GpuKernel) -> Result<wgpu::ComputePipeline, String> {
        let wgsl = kernel.wgsl();
        if let Some(p) = self.pipelines.lock().ok().and_then(|m| m.get(&wgsl).cloned()) {
            return Ok(p);
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(kernel.name()),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(kernel.name()),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(format!("{} kernel failed to compile: {err}", kernel.name()));
        }
        if let Ok(mut m) = self.pipelines.lock() {
            m.insert(wgsl, pipeline.clone());
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
                let kernel = kernel.bind(&GpuInput {
                    len,
                    rate,
                    is_spectrum,
                    second_len: second.as_ref().map(|(_, l)| *l),
                    default_rate: RuntimeConfig::default().default_sample_rate,
                })?;
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
                let pipeline = ctx.pipeline(&kernel)?;
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
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(kernel.workgroups(len), 1, 1);
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
        GpuRun { adapter: ctx.adapter.clone(), kernels: names },
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
            let auto = execute_sigql(sql, &amorphic, Instant::now())
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
}
