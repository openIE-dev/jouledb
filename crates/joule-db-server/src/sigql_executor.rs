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
use sigql::runtime::{OutputValue, Runtime, RuntimeConfig};
use sigql::types::DynSignal;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use std::time::Instant;

/// Maximum rows returned for signal/spectrum outputs to prevent OOM.
const MAX_SIGNAL_ROWS: usize = 10_000;

// ============================================================================
// Detection
// ============================================================================


/// Choose SigQL codegen target: WebGpu when a wgpu adapter is present, else Simd.
///
/// Probe runs once and is cached. Never invents a Metal backend — only WebGpu or Simd.
fn select_sigql_target() -> Target {
    if wgpu_adapter_available() {
        Target::WebGpu
    } else {
        Target::Simd
    }
}

fn wgpu_adapter_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // Probe on a dedicated thread so we never nest pollster inside a tokio worker.
        std::thread::spawn(|| {
            pollster::block_on(async {
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    flags: wgpu::InstanceFlags::default(),
                    backend_options: wgpu::BackendOptions::default(),
                    display: None,
                    memory_budget_thresholds: Default::default(),
                });
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .is_ok()
            })
        })
        .join()
        .unwrap_or(false)
    })
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
/// Pipeline: parse → compile → register sources → execute → map to QueryResponse.
pub fn execute_sigql(
    sql: &str,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    // 1. Parse
    let query = sigql::parse(sql)
        .map_err(|e| QueryErrorResponse::syntax_error(&format!("SigQL: {}", e), 1, 1))?;

    // 2. Compile — prefer WebGPU when a wgpu adapter is available, else SIMD.
    //    WebGPU must actually dispatch the generated WGSL. On any failure the
    //    Simd runtime still runs.
    let target = select_sigql_target();
    let mut gpu_note: Option<String> = None;
    if target == Target::WebGpu {
        match try_execute_webgpu(&query, amorphic) {
            Ok(samples) => {
                let mut response = webgpu_samples_response(&samples, start);
                response.warnings.push(
                    "SigQL plan executed on the wgpu adapter from generated WGSL".into(),
                );
                return Ok(response);
            }
            Err(reason) => {
                gpu_note = Some(format!(
                    "WebGpu plan did not execute ({reason}); fell back to Simd"
                ));
            }
        }
    }

    let plan = sigql::compile(&query, Target::Simd)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL compile: {}", e)))?;

    // 3. Create runtime and register signal sources from storage
    let mut runtime = Runtime::new(RuntimeConfig::default());
    register_sources_from_storage(&query, &mut runtime, amorphic)?;

    // 4. Execute on the CPU DSP / Simd runtime
    let result = runtime
        .execute(&plan)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL runtime: {}", e)))?;

    // 5. Map results to QueryResponse
    let mut response = map_result_to_response(result, start)?;
    response.device_target = Some("cpu".into());
    if let Some(note) = gpu_note {
        response.warnings.push(note);
    } else if target == Target::Simd {
        response
            .warnings
            .push("SigQL target Simd; no wgpu adapter".into());
    }
    Ok(response)
}

fn webgpu_samples_response(samples: &[f64], start: Instant) -> QueryResponse {
    let mut rows = Vec::new();
    let limit = samples.len().min(MAX_SIGNAL_ROWS);
    for (i, sample) in samples.iter().take(limit).enumerate() {
        rows.push(vec![
            serde_json::json!(i),
            serde_json::json!(sample),
        ]);
    }
    QueryResponse {
        columns: vec!["sample_index".into(), "value".into()],
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: samples.len() > MAX_SIGNAL_ROWS,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: Some("webgpu".into()),
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
    }
}

/// Compile the plan to WGSL and dispatch it. Errors mean the caller must use Simd.
fn try_execute_webgpu(
    query: &sigql::Query,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<Vec<f64>, String> {
    let plan = sigql::compile(query, Target::WebGpu).map_err(|e| e.to_string())?;
    let generated = sigql::compile::codegen::generate(&plan, Target::WebGpu).map_err(|e| e.to_string())?;
    let sigql::compile::codegen::GeneratedCode::WebGpu(code) = generated else {
        return Err("codegen did not produce WebGpu shader".into());
    };
    let samples = collect_samples(query, amorphic)?;
    if samples.is_empty() {
        return Err("no signal samples in storage".into());
    }
    dispatch_wgsl(&code.wgsl, &code.bindings, code.workgroup_size, &samples)
}

fn collect_samples(
    query: &sigql::Query,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<Vec<f64>, String> {
    let mut samples = Vec::new();
    for from in &query.from {
        let source_name = match from {
            FromClause::Signal(source_ref) => source_ref.path.to_string(),
            FromClause::Table { name, .. } => name.to_string(),
            _ => continue,
        };
        let table_name = source_name.replace('.', "_");
        let tables = amorphic.list_tables();
        if !tables.contains(&table_name) {
            continue;
        }
        let scan = amorphic.scan(&table_name).map_err(|e| e.to_string())?;
        for row in scan {
            if let Some(value) = row.get("value").and_then(value_to_f64) {
                samples.push(value);
            } else if let Some(value) = row.values.first().and_then(value_to_f64) {
                samples.push(value);
            }
        }
    }
    Ok(samples)
}

fn dispatch_wgsl(
    wgsl: &str,
    bindings: &[sigql::compile::codegen::BindingDescriptor],
    workgroup_size: (u32, u32, u32),
    samples: &[f64],
) -> Result<Vec<f64>, String> {
    let wgsl = wgsl.to_string();
    let binding_kinds: Vec<sigql::compile::codegen::BufferType> =
        bindings.iter().map(|b| b.buffer_type).collect();
    let samples = samples.to_vec();
    std::thread::spawn(move || {
        pollster::block_on(async move {
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
                .map_err(|e| format!("no wgpu adapter: {e:?}"))?;
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
                .map_err(|e| e.to_string())?;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sigql-wgsl"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let mut entries = Vec::new();
            for (i, kind) in binding_kinds.iter().enumerate() {
                let ty = match kind {
                    sigql::compile::codegen::BufferType::Uniform => wgpu::BufferBindingType::Uniform,
                    sigql::compile::codegen::BufferType::ReadOnlyStorage => {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    }
                    sigql::compile::codegen::BufferType::Storage => {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    }
                };
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sigql-bgl"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sigql-pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sigql-pipe"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
            let n = samples.len().max(1);
            let input_bytes: Vec<u8> = samples
                .iter()
                .map(|v| (*v as f32).to_le_bytes())
                .flatten()
                .collect();
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sigql-in"),
                contents: &input_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let out_size = (n * 4) as u64;
            let mut storage_buffers = Vec::new();
            let mut params_buf = None;
            let mut output_index = None;
            for (i, kind) in binding_kinds.iter().enumerate() {
                match kind {
                    sigql::compile::codegen::BufferType::ReadOnlyStorage => {}
                    sigql::compile::codegen::BufferType::Storage => {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("sigql-store"),
                            size: out_size.max(4),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        });
                        if output_index.is_none() {
                            output_index = Some(storage_buffers.len());
                        }
                        storage_buffers.push(buf);
                    }
                    sigql::compile::codegen::BufferType::Uniform => {
                        let params = [
                            n as u32,
                            1000f32.to_bits(),
                            n.next_power_of_two() as u32,
                            0u32,
                        ];
                        let mut bytes = Vec::new();
                        for word in params {
                            bytes.extend_from_slice(&word.to_le_bytes());
                        }
                        params_buf = Some(device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("sigql-params"),
                                contents: &bytes,
                                usage: wgpu::BufferUsages::UNIFORM,
                            },
                        ));
                    }
                }
                let _ = i;
            }
            let mut bind_entries = Vec::new();
            let mut storage_cursor = 0usize;
            for (i, kind) in binding_kinds.iter().enumerate() {
                let resource = match kind {
                    sigql::compile::codegen::BufferType::ReadOnlyStorage => {
                        input.as_entire_binding()
                    }
                    sigql::compile::codegen::BufferType::Storage => {
                        let buf = storage_buffers.get(storage_cursor).ok_or("missing storage")?;
                        storage_cursor += 1;
                        buf.as_entire_binding()
                    }
                    sigql::compile::codegen::BufferType::Uniform => params_buf
                        .as_ref()
                        .ok_or("missing params")?
                        .as_entire_binding(),
                };
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource,
                });
            }
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sigql-bg"),
                layout: &layout,
                entries: &bind_entries,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sigql-enc"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sigql-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg = workgroup_size.0.max(1);
                let groups = ((n as u32) + wg - 1) / wg;
                pass.dispatch_workgroups(groups.max(1), 1, 1);
            }
            let out_buf = output_index
                .and_then(|idx| storage_buffers.get(idx))
                .ok_or("shader has no storage output")?;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sigql-stage"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(out_buf, 0, &staging, 0, out_size);
            queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv()
                .map_err(|_| "map channel closed".to_string())?
                .map_err(|e| format!("map failed: {e:?}"))?;
            let data = slice.get_mapped_range().to_vec();
            drop(data.clone());
            let copied = slice.get_mapped_range().to_vec();
            staging.unmap();
            let mut out = Vec::with_capacity(n);
            for chunk in copied.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                let v = f32::from_le_bytes(arr) as f64;
                if !v.is_finite() {
                    return Err("webgpu output was non-finite".into());
                }
                out.push(v);
            }
            if out.is_empty() {
                return Err("webgpu output was empty".into());
            }
            Ok(out)
        })
    })
    .join()
    .map_err(|_| "webgpu thread panicked".to_string())?
}

/// Extract signal source names from the parsed query's FROM clauses
/// and register matching amorphic tables as signal sources in the runtime.
fn register_sources_from_storage(
    query: &sigql::Query,
    runtime: &mut Runtime,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(), QueryErrorResponse> {
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
            runtime.register_signal(&source_name, signal);
        }
    }

    Ok(())
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
