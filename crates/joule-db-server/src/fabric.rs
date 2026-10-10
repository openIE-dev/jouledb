//! Heterogeneous compute dispatch.
//!
//! Routes a numeric job through `HardwareAdvisor` + `ComputeRouter`. Every
//! dispatch smart-routes a preferred device from algorithm affinity and the
//! devices `detect_devices()` actually found, including the Apple Neural Engine
//! on macOS aarch64. Every receipt names a fallback processor. If the preferred
//! device cannot run the kernel, the fallback is the processor that did. If the
//! preferred device ran it, the fallback is the next real processor. Never
//! empty, never "accelerator missing", never a refusal.
//!
//! When a GPU lane is available and the GPU is the chosen backend (or the NPU
//! fallback), the kernel runs on Metal. wgpu is the lane when Metal is not the
//! selected path, including on a Mac where the Metal device is missing. The
//! receipt `backend` is `metal` or `wgpu`, not `cpu`, when that path produces
//! the value.
//!
//! HDC similarity ([`dispatch_hdc_similarity`], [`dispatch_hdc_hamming`])
//! smart-routes to `npu` on macos+aarch64 and tries the Apple Neural Engine
//! lane first. That lane is Core ML (the only sanctioned ANE path): a
//! generated FP16 ML Program computes `Q(+-1) . M(+-1)^T`, loaded with
//! `CPUAndNeuralEngine` (`joule-ane-rt`). The receipt names the device that
//! `MLComputePlan` reports for the similarity op: `npu`/`coreml-ane` only when
//! the plan says Neural Engine, `gpu`/`coreml` or `cpu`/`coreml` when Core ML
//! placed it there. After the ANE lane the next processor is the Metal GPU
//! HDC kernel when a Metal lane exists, else the CPU.
//!
//! Small HDC jobs never touch Core ML. Below [`ANE_MIN_WORK`] (`n*k*d`,
//! measured on an M5 Max) Core ML's planner keeps the op on the CPU, so the
//! job goes straight to the Metal HDC kernel (or the CPU). Above it, the
//! first job in a shape bucket probes Core ML and the `MLComputePlan`
//! placement is cached per process; later jobs in that bucket follow the
//! cached placement. The receipt's `route_reason` says which rule fired.
//! Compile / load / plan time is a one-time setup cost reported as
//! `setup_ms` / `setup_joules` on the call that paid it; per-job `joules`
//! cover only the kernel or prediction. The fallback is always a
//! real processor. The scalar reduction ([`dispatch_sum`]) is not an HDC
//! similarity and keeps its Metal / wgpu / CPU lanes.
//!
//! Energy is a measurement on the receipt. `energy_source` is
//! `powermetrics` only when `powermetrics` (its `ANE Power` line) ran through
//! `sudo -n` (never prompting) while the ANE executed the op; otherwise it
//! is `estimate` (watts snapshot x wall time). This path never refuses or
//! aborts work because of an energy figure or budget.

use joule_db_core::persistence::compute::{
    AggregationType, BufferUsage, ComputeBackend, ComputeOp, CpuComputeBackend,
};
use joule_db_energy::{
    AdaptiveParams, AlgorithmType, ComputeRouter, DeviceTarget, EnergyConfig, EnergySnapshot,
    ExecutionHint, HardwareAdvisor,
};
use joule_ane_rt::PlannedDevice;
use joule_runtime::{AcceleratorKind, detect_devices};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// What ran, where, and how much energy it took.
#[derive(Debug, Clone)]
pub struct EnergyReceipt {
    /// Processor that produced `value`.
    ///
    /// Reduction ([`dispatch_sum`]): `gpu` when Metal or wgpu ran, `cpu` when
    /// the CPU kernel ran, `npu` only when the Neural Engine was requested
    /// and no GPU lane existed (the CPU kernel ran; `backend` says `cpu`).
    /// The ANE does not execute the reduction.
    ///
    /// HDC similarity ([`dispatch_hdc_similarity`]): `npu` only when Core ML
    /// ran it and `MLComputePlan` placed the similarity op on the Neural
    /// Engine (`backend = coreml-ane`); `gpu`/`cpu` with `backend = coreml`
    /// when Core ML placed it there; `gpu`/`metal` or `cpu`/`cpu` otherwise.
    pub device: String,
    /// Smart-routed preferred device: algorithm affinity intersected with
    /// what `detect_devices()` found. CPU is always eligible. Never absent.
    pub requested_device: String,
    /// `coreml-ane`, `coreml`, `metal`, `wgpu`, or `cpu`. The kernel that
    /// produced `value`.
    pub backend: String,
    /// Joules for this dispatch: measured over the job window (all
    /// CPU/GPU/ANE rails, or package counters) or `watts x seconds` for an
    /// estimate. Reporting only.
    pub joules: f64,
    /// Joules above the idle baseline (equals `joules` for an estimate).
    pub incremental_joules: f64,
    /// `joules / seconds` (the snapshot watts, default 5, for an estimate).
    pub watts: f64,
    /// Job seconds the energy covers (setup excluded).
    pub seconds: f64,
    /// `measured`, `interpolated` (shorter than a sample interval, so
    /// apportioned from overlapping samples), or `estimate`.
    pub energy_confidence: String,
    /// Mean per-device watts over the job window.
    pub cpu_watts: Option<f64>,
    pub gpu_watts: Option<f64>,
    pub ane_watts: Option<f64>,
    /// Idle baseline subtracted for `incremental_joules`, watts.
    pub baseline_watts: Option<f64>,
    /// Fallback processor. Always a real device (`cpu`, `gpu`, `npu`, `tpu`,
    /// `lpu`). Never empty.
    ///
    /// When the preferred device cannot run the kernel, this is the processor
    /// that did run (usually `cpu`). When the preferred device ran the kernel,
    /// this is the next real processor: `cpu` after a GPU/NPU/TPU/LPU run, or
    /// another detected device after a CPU run. A CPU-only host still names
    /// `cpu` — there is always a processor.
    pub fallback: String,
    /// Numeric result of the reduction that ran on the chosen backend.
    /// For HDC similarity: the sum of all `n x k` dot products (checksum).
    pub value: f64,
    pub algorithm: String,
    /// Where `joules` came from: `powermetrics`, `rapl`, `emi` (measured), or
    /// `estimate` (watts snapshot x time).
    pub energy_source: String,
    /// `MLComputePlan` summary (`layout:op@device`) when the Core ML lane ran.
    pub placement: Option<String>,
    /// Why this lane was chosen, e.g. `below_ane_threshold`, `ane_probe`,
    /// `ane_placement_cached`, `ane_placement_cached_off_ane`, `force_cpu`.
    pub route_reason: String,
    /// One-time setup paid by this call (Core ML compile / plan / load,
    /// Metal shader compile). `None` when nothing was set up. Not included
    /// in `joules`.
    pub setup_ms: Option<f64>,
    /// `watts x setup time`, always an estimate. `None` with `setup_ms`.
    pub setup_joules: Option<f64>,
}

impl EnergyReceipt {
    /// One-line, key=value rendering for logs and test output.
    pub fn line(&self) -> String {
        let w = |v: Option<f64>| v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "none".into());
        format!(
            "requested_device={} device={} backend={} fallback={} route_reason={} algorithm={} value={} joules={:.6e} incremental_joules={:.6e} seconds={:.6} watts={:.4} energy_source={} energy_confidence={} cpu_w={} gpu_w={} ane_w={} baseline_w={} placement={} setup_ms={} setup_joules={}",
            self.requested_device,
            self.device,
            self.backend,
            self.fallback,
            self.route_reason,
            self.algorithm,
            self.value,
            self.joules,
            self.incremental_joules,
            self.seconds,
            self.watts,
            self.energy_source,
            self.energy_confidence,
            w(self.cpu_watts),
            w(self.gpu_watts),
            w(self.ane_watts),
            w(self.baseline_watts),
            self.placement.as_deref().unwrap_or("none"),
            self.setup_ms
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "none".into()),
            self.setup_joules
                .map(|v| format!("{v:.6e}(estimate)"))
                .unwrap_or_else(|| "none".into())
        )
    }
}

/// GPU command path that can run the numeric kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuLane {
    /// Metal compute shader. Live submits only on macOS.
    Metal,
    /// wgpu aggregate. Used when Metal is not the selected lane.
    Wgpu,
}

/// Core ML lane for the Apple Neural Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AneLane {
    /// Live Core ML with `CPUAndNeuralEngine` (macOS aarch64).
    Live,
    /// Injected descriptor: the CPU reference computes the scores and the
    /// receipt stamps this `MLComputePlan` device, so Linux CI can check
    /// receipts. Never used by the live profile.
    Descriptor(PlannedDevice),
}

/// Smallest HDC job (`n * k * d`) worth sending to Core ML.
///
/// Measured on an Apple M5 Max (macOS 27.0.1, `CPUAndNeuralEngine`, d=2048):
/// `MLComputePlan` placed the similarity op on the CPU for every shape up to
/// n=64, k=256 (33.5M) and on the Neural Engine from n=64, k=1024 (134.2M)
/// upward, for both the matmul and the 1x1-conv layouts. Below this the
/// planner keeps the op on the CPU, so the fabric skips Core ML entirely.
pub const ANE_MIN_WORK: u64 = 64 * 1024 * 2048;

/// Shape bucket for the per-process placement cache: each dimension rounded
/// up to a power of two, `d` per exact FP16 tile, plus the tile count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShapeBucket {
    n: usize,
    k: usize,
    tile_d: usize,
    tiles: usize,
}

impl ShapeBucket {
    fn of(n: usize, k: usize, d: usize) -> Self {
        let tile = joule_ane_rt::EXACT_TILE_DIM;
        Self {
            n: n.next_power_of_two(),
            k: k.next_power_of_two(),
            tile_d: d.min(tile).next_power_of_two(),
            tiles: d.div_ceil(tile),
        }
    }
}

/// `MLComputePlan` similarity-op placement remembered per shape bucket.
type PlacementCache = Arc<Mutex<HashMap<ShapeBucket, PlannedDevice>>>;

fn live_placement_cache() -> PlacementCache {
    static CACHE: OnceLock<PlacementCache> = OnceLock::new();
    Arc::clone(CACHE.get_or_init(|| Arc::new(Mutex::new(HashMap::new()))))
}

fn cached_placement(cache: &PlacementCache, bucket: ShapeBucket) -> Option<PlannedDevice> {
    cache
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&bucket)
        .copied()
}

fn remember_placement(cache: &PlacementCache, bucket: ShapeBucket, device: PlannedDevice) {
    cache
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(bucket, device);
}

/// Accelerators visible to one dispatch. CPU is implicit and always present.
#[derive(Clone)]
struct HostProfile {
    kinds: Vec<AcceleratorKind>,
    /// Core ML / ANE lane for HDC similarity. `None` off macOS aarch64.
    ane: Option<AneLane>,
    /// HDC work (`n*k*d`) below which Core ML is not invoked. Live:
    /// [`ANE_MIN_WORK`].
    ane_min_work: u64,
    /// Per-process placement memory for HDC shape buckets.
    placement_cache: PlacementCache,
    /// `None` means a named GPU still cannot execute (no Metal device and no
    /// wgpu adapter).
    gpu: Option<GpuLane>,
    /// `true` submits to Metal or wgpu. `false` is an injected descriptor:
    /// the lane's f32 reference reduction runs in-process so Linux CI can
    /// check receipts without a Mac or an adapter. `dispatch_sum` is live.
    live_gpu: bool,
}

fn detected_kinds() -> &'static [AcceleratorKind] {
    static KINDS: OnceLock<Vec<AcceleratorKind>> = OnceLock::new();
    KINDS
        .get_or_init(|| detect_devices().into_iter().map(|d| d.kind).collect())
        .as_slice()
}

fn wgpu_adapter_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
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

fn live_gpu_lane(gpu_found: bool) -> Option<GpuLane> {
    if !gpu_found {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        if joule_metal_rt::MetalDevice::system_default().is_ok() {
            return Some(GpuLane::Metal);
        }
    }
    if wgpu_adapter_available() {
        Some(GpuLane::Wgpu)
    } else {
        None
    }
}

fn live_ane_lane(kinds: &[AcceleratorKind]) -> Option<AneLane> {
    let npu_found = kinds.iter().any(|k| *k == AcceleratorKind::NPU);
    if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && npu_found
        && joule_ane_rt::is_available()
    {
        Some(AneLane::Live)
    } else {
        None
    }
}

fn live_profile() -> HostProfile {
    let kinds = detected_kinds().to_vec();
    let gpu_found = kinds.iter().any(|k| *k == AcceleratorKind::GPU);
    HostProfile {
        ane: live_ane_lane(&kinds),
        ane_min_work: ANE_MIN_WORK,
        placement_cache: live_placement_cache(),
        gpu: live_gpu_lane(gpu_found),
        live_gpu: true,
        kinds,
    }
}

fn kind_on(host: &HostProfile, kind: AcceleratorKind) -> bool {
    host.kinds.iter().any(|k| *k == kind)
}

fn device_present(device: DeviceTarget, host: &HostProfile) -> bool {
    match device {
        // Every host has a CPU. Routing never observes "no processor".
        DeviceTarget::Cpu => true,
        DeviceTarget::Gpu => kind_on(host, AcceleratorKind::GPU),
        DeviceTarget::Npu => kind_on(host, AcceleratorKind::NPU),
        DeviceTarget::Tpu => kind_on(host, AcceleratorKind::TPU),
        DeviceTarget::Lpu => kind_on(host, AcceleratorKind::LPU),
    }
}

fn snapshot_for(host: &HostProfile) -> EnergySnapshot {
    let mut snap = EnergySnapshot::default();
    snap.gpu_available = device_present(DeviceTarget::Gpu, host);
    snap.npu_available = device_present(DeviceTarget::Npu, host);
    snap.tpu_available = device_present(DeviceTarget::Tpu, host);
    snap.lpu_available = device_present(DeviceTarget::Lpu, host);
    if snap.power_watts <= 0.0 {
        snap.power_watts = 5.0;
    }
    snap
}

/// Watts the cost model multiplies elapsed time by (the live energy
/// snapshot's figure; 5 W when the platform reports none). An estimate.
pub fn routing_watts() -> f64 {
    snapshot_for(&live_profile()).power_watts
}

fn preferred_from_hints(hints: &[ExecutionHint]) -> Vec<DeviceTarget> {
    let mut out = Vec::new();
    for hint in hints {
        let device = match hint {
            ExecutionHint::PreferGpu => Some(DeviceTarget::Gpu),
            ExecutionHint::PreferNpu => Some(DeviceTarget::Npu),
            ExecutionHint::PreferTpu => Some(DeviceTarget::Tpu),
            ExecutionHint::PreferLpu => Some(DeviceTarget::Lpu),
            ExecutionHint::PreferCpu => Some(DeviceTarget::Cpu),
            _ => None,
        };
        if let Some(device) = device {
            if !out.contains(&device) {
                out.push(device);
            }
        }
    }
    out
}

/// Smart route: advisor affinity hints, algorithm affinity, and only devices
/// `detect_devices()` actually found (CPU always). Apple Neural Engine is an
/// NPU on that list for macos+aarch64.
fn smart_preferred(
    algorithm: AlgorithmType,
    hints: &[ExecutionHint],
    snap: &EnergySnapshot,
    host: &HostProfile,
) -> DeviceTarget {
    let hinted = preferred_from_hints(hints)
        .into_iter()
        .filter(|device| device_present(*device, host))
        .collect::<Vec<_>>();
    let params = AdaptiveParams {
        memtable_threshold: 4 * 1024 * 1024,
        l0_compaction_trigger: 4,
        buffer_pool_capacity: 256,
        write_rate_limit: 0,
        hints: hints.to_vec(),
        preferred_devices: hinted,
        gpu_dispatch_overrides: None,
    };
    let routed = ComputeRouter::route(algorithm, &params, snap).device;
    if device_present(routed, host) {
        routed
    } else {
        DeviceTarget::Cpu
    }
}

/// Next processor after a successful run, or the processor that actually ran
/// when the preferred device could not execute the kernel.
fn fallback_processor(
    preferred_ran: bool,
    executed: DeviceTarget,
    host: &HostProfile,
) -> DeviceTarget {
    if !preferred_ran {
        return executed;
    }
    if executed != DeviceTarget::Cpu {
        return DeviceTarget::Cpu;
    }
    const OTHERS: [DeviceTarget; 4] = [
        DeviceTarget::Npu,
        DeviceTarget::Gpu,
        DeviceTarget::Tpu,
        DeviceTarget::Lpu,
    ];
    OTHERS
        .into_iter()
        .find(|device| device_present(*device, host))
        .unwrap_or(DeviceTarget::Cpu)
}

fn cpu_sum(values: &[f64]) -> Result<f64, String> {
    let mut backend = CpuComputeBackend::new();
    let n = values.len().max(1) as u32;
    let mut bytes = Vec::with_capacity(n as usize * 8);
    if values.is_empty() {
        bytes.extend_from_slice(&0f64.to_le_bytes());
    } else {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let input = backend
        .create_buffer(bytes.len() as u64, BufferUsage::STORAGE_READ, Some("fabric-in"))
        .map_err(|e| e.to_string())?;
    backend
        .write_buffer(input, 0, &bytes)
        .map_err(|e| e.to_string())?;
    let result = backend
        .execute(
            ComputeOp::Aggregate {
                agg_type: AggregationType::Sum,
                num_items: n,
            },
            &[input],
        )
        .map_err(|e| e.to_string())?;
    let out = backend
        .read_buffer(result.output_buffer, 0, 8)
        .map_err(|e| e.to_string())?;
    let arr: [u8; 8] = out
        .as_slice()
        .try_into()
        .map_err(|_| "cpu aggregate returned a short buffer".to_string())?;
    Ok(f64::from_le_bytes(arr))
}

fn wgpu_sum(values: &[f64]) -> Result<f64, String> {
    let owned = values.to_vec();
    std::thread::spawn(move || {
        pollster::block_on(async {
            let mut backend = joule_db_gpu::WgpuComputeBackend::new()
                .await
                .map_err(|e| e.to_string())?;
            let f32s: Vec<f32> = if owned.is_empty() {
                vec![0.0]
            } else {
                owned.iter().map(|v| *v as f32).collect()
            };
            let mut bytes = Vec::with_capacity(f32s.len() * 4);
            for value in &f32s {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let input = backend
                .create_buffer(
                    bytes.len() as u64,
                    BufferUsage::STORAGE_READ,
                    Some("fabric-gpu-in"),
                )
                .map_err(|e| e.to_string())?;
            backend
                .write_buffer(input, 0, &bytes)
                .map_err(|e| e.to_string())?;
            let result = backend
                .execute(
                    ComputeOp::Aggregate {
                        agg_type: AggregationType::Sum,
                        num_items: f32s.len() as u32,
                    },
                    &[input],
                )
                .map_err(|e| e.to_string())?;
            backend.synchronize().map_err(|e| e.to_string())?;
            let out = backend
                .read_buffer(result.output_buffer, 0, 4)
                .map_err(|e| e.to_string())?;
            if out.len() < 4 {
                return Err("wgpu aggregate returned a short buffer".into());
            }
            let arr: [u8; 4] = out[0..4].try_into().unwrap();
            Ok(f32::from_le_bytes(arr) as f64)
        })
    })
    .join()
    .map_err(|_| "wgpu dispatch thread panicked".to_string())?
}

/// Route `values` and reduce them. Always returns a receipt; never an energy error.
///
/// When the smart route prefers the GPU and a GPU lane exists, the
/// process-wide [cost model](crate::cost_model::global) decides between
/// that lane and the CPU from measured joules (then time) for this
/// algorithm and size bucket; `route_reason` says which rule fired.
pub fn dispatch_sum(values: &[f64], algorithm: AlgorithmType, force_cpu: bool) -> EnergyReceipt {
    dispatch_sum_costed(values, algorithm, force_cpu, &live_profile(), crate::cost_model::global())
}

/// [`dispatch_sum_on`] with measured-cost routing between the GPU lane and
/// the CPU. Other preferences (NPU, TPU, LPU, CPU), `force_cpu`, and hosts
/// without a GPU lane keep the smart route unchanged.
fn dispatch_sum_costed(
    values: &[f64],
    algorithm: AlgorithmType,
    force_cpu: bool,
    host: &HostProfile,
    model: &crate::cost_model::CostModel,
) -> EnergyReceipt {
    use crate::cost_model::Candidate;
    let snap = snapshot_for(host);
    let hints = HardwareAdvisor::new(&EnergyConfig::default()).advise(&snap);
    let preferred = smart_preferred(algorithm, &hints, &snap, host);
    let lane = match host.gpu {
        Some(lane) if !force_cpu && preferred == DeviceTarget::Gpu => lane,
        _ => return dispatch_sum_on(values, algorithm, force_cpu, host),
    };
    let gpu = Candidate::new(
        "gpu",
        match lane {
            GpuLane::Metal => "metal",
            GpuLane::Wgpu => "wgpu",
        },
    );
    let op = format!("fabric:sum:{algorithm}");
    let route = model.choose(&op, values.len(), &[gpu, Candidate::cpu()]);
    let mut receipt = dispatch_sum_on(values, algorithm, route.chosen.device == "cpu", host);
    let ran = Candidate::new(receipt.device.clone(), receipt.backend.clone());
    let measured = receipt.energy_source != "estimate";
    model.record_cost(
        &op,
        values.len(),
        &ran,
        receipt.seconds,
        receipt.joules,
        measured.then_some(receipt.incremental_joules),
        &receipt.energy_source,
    );
    receipt.route_reason = route.reason_label();
    if ran.device == route.chosen.device {
        receipt.fallback = route.fallback.device.clone();
    } else {
        receipt.route_reason.push_str(&format!("+{}_error", route.chosen.device));
    }
    receipt
}

fn dispatch_sum_on(
    values: &[f64],
    algorithm: AlgorithmType,
    force_cpu: bool,
    host: &HostProfile,
) -> EnergyReceipt {
    let snap = snapshot_for(host);
    let advisor = HardwareAdvisor::new(&EnergyConfig::default());
    let hints = advisor.advise(&snap);
    let preferred = smart_preferred(algorithm, &hints, &snap, host);

    let meter = meter_for(host);
    let span = meter.begin();
    let (value, backend, reported, executed, preferred_ran) =
        execute(values, preferred, force_cpu, host);
    let fallback = fallback_processor(preferred_ran, executed, host);
    let device = if backend == "cpu" { "cpu" } else { "gpu" };
    let e = meter.finish(span, std::time::Duration::ZERO, device, snap.power_watts);

    EnergyReceipt {
        device: reported.to_string(),
        requested_device: preferred.to_string(),
        backend: backend.to_string(),
        joules: e.joules,
        incremental_joules: e.incremental_joules,
        watts: e.watts,
        seconds: e.seconds,
        energy_confidence: e.confidence.clone(),
        cpu_watts: e.cpu_watts,
        gpu_watts: e.gpu_watts,
        ane_watts: e.ane_watts,
        baseline_watts: e.baseline_watts,
        fallback: fallback.to_string(),
        value,
        algorithm: algorithm.to_string(),
        energy_source: e.source.clone(),
        placement: None,
        route_reason: if force_cpu {
            "force_cpu".to_string()
        } else {
            format!("requested_{preferred}")
        },
        setup_ms: None,
        setup_joules: None,
    }
}

fn gpu_matches_cpu(values: &[f64], gpu_value: f64) -> (bool, f64) {
    let cpu_value = cpu_sum(values).unwrap_or(gpu_value);
    let ok = gpu_value.is_finite()
        && (gpu_value - cpu_value).abs() <= 1e-2 * (1.0 + cpu_value.abs());
    (ok, cpu_value)
}

/// Run the GPU lane. Live Metal submits `joule_metal_rt::sum_f64`. A Metal
/// failure falls through to wgpu. Descriptor mode does not submit; it applies
/// the shader's f32 reduction and stamps `metal` or `wgpu`.
fn run_gpu_lane(values: &[f64], host: &HostProfile) -> Result<(f64, &'static str), String> {
    let lane = host
        .gpu
        .ok_or_else(|| "no gpu execution lane".to_string())?;
    if !host.live_gpu {
        let backend = match lane {
            GpuLane::Metal => "metal",
            GpuLane::Wgpu => "wgpu",
        };
        return Ok((joule_metal_rt::reference_f32_sum(values), backend));
    }
    match lane {
        GpuLane::Metal => match joule_metal_rt::sum_f64(values) {
            Ok(value) => Ok((value, "metal")),
            Err(_) => wgpu_sum(values).map(|value| (value, "wgpu")),
        },
        GpuLane::Wgpu => wgpu_sum(values).map(|value| (value, "wgpu")),
    }
}

/// Run the kernel. `force_cpu` pins execution to the CPU; it does not erase
/// the smart-routed preferred device and it does not refuse the job.
///
/// Return is `(value, backend, reported_device, executed, preferred_ran)`.
fn execute(
    values: &[f64],
    preferred: DeviceTarget,
    force_cpu: bool,
    host: &HostProfile,
) -> (f64, &'static str, DeviceTarget, DeviceTarget, bool) {
    let gpu_choice = preferred == DeviceTarget::Gpu || preferred == DeviceTarget::Npu;
    if !force_cpu && host.gpu.is_some() && gpu_choice {
        match run_gpu_lane(values, host) {
            Ok((gpu_value, backend)) => {
                let (ok, cpu_value) = gpu_matches_cpu(values, gpu_value);
                if ok {
                    // NPU was requested but the kernel ran on the GPU. The
                    // reported device is the GPU. `preferred_ran` stays false
                    // so the fallback processor is that GPU, not CPU.
                    let preferred_ran = preferred == DeviceTarget::Gpu;
                    return (
                        gpu_value,
                        backend,
                        DeviceTarget::Gpu,
                        DeviceTarget::Gpu,
                        preferred_ran,
                    );
                }
                return (cpu_value, "cpu", DeviceTarget::Cpu, DeviceTarget::Cpu, false);
            }
            Err(_) => {
                let cpu_value = cpu_sum(values).unwrap_or(0.0);
                if preferred == DeviceTarget::Npu {
                    return (
                        cpu_value,
                        "cpu",
                        DeviceTarget::Npu,
                        DeviceTarget::Cpu,
                        false,
                    );
                }
                return (cpu_value, "cpu", DeviceTarget::Cpu, DeviceTarget::Cpu, false);
            }
        }
    }

    let cpu_value = cpu_sum(values).unwrap_or(0.0);
    if !force_cpu && matches!(preferred, DeviceTarget::Npu | DeviceTarget::Tpu | DeviceTarget::Lpu)
    {
        // No GPU lane. Name the routed accelerator, run on CPU, and record
        // CPU as the fallback processor that did the work.
        return (cpu_value, "cpu", preferred, DeviceTarget::Cpu, false);
    }

    let preferred_ran = preferred == DeviceTarget::Cpu;
    (
        cpu_value,
        "cpu",
        DeviceTarget::Cpu,
        DeviceTarget::Cpu,
        preferred_ran,
    )
}

/// Result of an HDC similarity dispatch.
#[derive(Debug, Clone)]
pub struct HdcDispatch {
    pub receipt: EnergyReceipt,
    /// `n x k` row-major dot products of the +-1 vectors.
    pub scores: Vec<i32>,
    /// `n x k` row-major Hamming distances, `(d - dot) / 2`.
    pub hamming: Vec<u32>,
    pub n: usize,
    pub k: usize,
    pub d: usize,
    /// Per-op `MLComputePlan` lines when the Core ML lane ran.
    pub plan: Vec<String>,
    /// Core ML layouts tried with the device the plan named for each.
    pub attempts: Vec<String>,
}

/// What the Core ML lane produced.
struct AneOutcome {
    scores: Vec<i32>,
    /// Per-job prediction seconds (excludes setup).
    predict_s: f64,
    /// One-time setup paid by this call, seconds (0 when reused).
    setup_s: f64,
    planned: PlannedDevice,
    summary: String,
    plan: Vec<String>,
    attempts: Vec<String>,
    power: Option<joule_ane_rt::PowerMeasurement>,
}

fn run_ane_lane(
    queries: &[i8],
    memory: &[i8],
    d: usize,
    lane: AneLane,
) -> Result<AneOutcome, String> {
    match lane {
        AneLane::Descriptor(planned) => Ok(AneOutcome {
            scores: joule_ane_rt::cpu_dot_scores(queries, memory, d),
            predict_s: 0.0,
            setup_s: 0.0,
            planned,
            summary: format!("descriptor:matmul@{}", planned.as_str()),
            plan: vec![format!("matmul->scores preferred={} (descriptor)", planned.as_str())],
            attempts: vec![format!("descriptor@{}", planned.as_str())],
            power: None,
        }),
        AneLane::Live => {
            let run = joule_ane_rt::hdc_similarity_auto(
                queries,
                memory,
                d,
                joule_ane_rt::PowerSampling::OnNeuralEngine,
            )
            .map_err(|e| e.to_string())?;
            Ok(AneOutcome {
                predict_s: run.predict_s,
                setup_s: if run.setup_performed { run.setup_s } else { 0.0 },
                planned: run.placement.similarity_device(),
                summary: run.placement.summary(),
                plan: run.placement.lines(),
                attempts: run
                    .attempts
                    .iter()
                    .map(|(l, dev)| format!("{}@{}", l.as_str(), dev.as_str()))
                    .collect(),
                power: run.power,
                scores: run.scores,
            })
        }
    }
}

/// Metal HDC kernel. Descriptor mode applies the CPU reference. Returns
/// `(scores, per-job seconds, setup seconds)`.
fn run_gpu_hdc(
    queries: &[i8],
    memory: &[i8],
    d: usize,
    host: &HostProfile,
) -> Result<(Vec<i32>, f64, f64), String> {
    match host.gpu {
        Some(GpuLane::Metal) if !host.live_gpu => {
            let started = Instant::now();
            let scores = joule_metal_rt::reference_hdc_dot(queries, memory, d);
            Ok((scores, started.elapsed().as_secs_f64(), 0.0))
        }
        Some(GpuLane::Metal) => joule_metal_rt::hdc_dot_i8_timed(queries, memory, d)
            .map(|(scores, t)| (scores, t.kernel_s, t.setup_s))
            .map_err(|e| e.to_string()),
        // No wgpu HDC kernel: the wgpu lane only implements the reduction.
        _ => Err("no metal HDC lane".to_string()),
    }
}

/// Route an HDC similarity job: `queries` is `n x d`, `memory` is `k x d`,
/// row-major, every value +1 or -1.
///
/// Order: Core ML / ANE (when routed to `npu` and the lane exists), then the
/// Metal GPU kernel, then the CPU. Every accelerator result is checked
/// against the CPU reference; a mismatch moves to the next processor. The
/// only error is malformed input. Energy never refuses or aborts work.
pub fn dispatch_hdc_similarity(
    queries: &[i8],
    memory: &[i8],
    d: usize,
    force_cpu: bool,
) -> Result<HdcDispatch, String> {
    dispatch_hdc_on(queries, memory, d, force_cpu, &live_profile())
}

/// [`dispatch_hdc_similarity`] over binary hypervectors. Bit 0 maps to +1
/// and bit 1 to -1, so the reported Hamming distance equals
/// `BinaryHyperVector::hamming_distance`.
pub fn dispatch_hdc_hamming(
    queries: &[joule_db_hdc::BinaryHyperVector],
    memory: &[joule_db_hdc::BinaryHyperVector],
    force_cpu: bool,
) -> Result<HdcDispatch, String> {
    let d = queries
        .first()
        .map(|v| v.dimensions())
        .ok_or_else(|| "no query vectors".to_string())?;
    if queries.iter().chain(memory).any(|v| v.dimensions() != d) {
        return Err(format!("all hypervectors must have {d} dimensions"));
    }
    let to_bipolar = |vs: &[joule_db_hdc::BinaryHyperVector]| -> Vec<i8> {
        let mut out = Vec::with_capacity(vs.len() * d);
        for v in vs {
            for i in 0..d {
                out.push(if v.get_bit(i) { -1 } else { 1 });
            }
        }
        out
    };
    dispatch_hdc_similarity(&to_bipolar(queries), &to_bipolar(memory), d, force_cpu)
}

/// Lane decision for one HDC job, before anything runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HdcRoute {
    try_ane: bool,
    reason: &'static str,
}

/// Size- and history-aware choice of whether to invoke Core ML.
///
/// Only `npu`-routed jobs on a host with the Core ML lane consider it. A
/// cached `MLComputePlan` placement for the job's shape bucket wins: Neural
/// Engine -> run Core ML, anything else -> skip it. With no cache entry, jobs
/// below `host.ane_min_work` skip Core ML (`below_ane_threshold`) and larger
/// jobs probe it (`ane_probe`). Skipping Core ML means the Metal HDC kernel,
/// else the CPU.
fn route_hdc(
    n: usize,
    k: usize,
    d: usize,
    preferred: DeviceTarget,
    force_cpu: bool,
    host: &HostProfile,
) -> HdcRoute {
    let skip = |reason| HdcRoute { try_ane: false, reason };
    if force_cpu {
        return skip("force_cpu");
    }
    match preferred {
        DeviceTarget::Npu => {}
        DeviceTarget::Gpu => return skip("requested_gpu"),
        DeviceTarget::Cpu => return skip("requested_cpu"),
        DeviceTarget::Tpu => return skip("requested_tpu"),
        DeviceTarget::Lpu => return skip("requested_lpu"),
    }
    if host.ane.is_none() {
        return skip("no_ane_lane");
    }
    match cached_placement(&host.placement_cache, ShapeBucket::of(n, k, d)) {
        Some(PlannedDevice::NeuralEngine) => HdcRoute {
            try_ane: true,
            reason: "ane_placement_cached",
        },
        Some(_) => skip("ane_placement_cached_off_ane"),
        None => {
            let work = (n as u64).saturating_mul(k as u64).saturating_mul(d as u64);
            if work < host.ane_min_work {
                skip("below_ane_threshold")
            } else {
                HdcRoute {
                    try_ane: true,
                    reason: "ane_probe",
                }
            }
        }
    }
}

fn dispatch_hdc_on(
    queries: &[i8],
    memory: &[i8],
    d: usize,
    force_cpu: bool,
    host: &HostProfile,
) -> Result<HdcDispatch, String> {
    let (n, k) = joule_ane_rt::validate(queries, memory, d).map_err(|e| e.to_string())?;
    let algorithm = AlgorithmType::Hdc;
    let snap = snapshot_for(host);
    let advisor = HardwareAdvisor::new(&EnergyConfig::default());
    let hints = advisor.advise(&snap);
    let preferred = smart_preferred(algorithm, &hints, &snap, host);

    // CPU reference: the CPU lane's result and the check for every
    // accelerator result. Its time is the CPU lane's per-job time.
    let meter = meter_for(host);
    let cpu_span = meter.begin();
    let cpu_started = Instant::now();
    let cpu_ref = joule_ane_rt::cpu_dot_scores(queries, memory, d);
    let cpu_s = cpu_started.elapsed().as_secs_f64();
    let cpu_end = Instant::now();
    cpu_span.close(cpu_end);
    // GPU kernel window: (span, end, setup to skip).
    let mut gpu_window: Option<(crate::power_meter::Span, Instant, std::time::Duration)> = None;
    // Core ML window (setup skipped), for a Core ML run placed off the ANE.
    let mut ane_window: Option<(crate::power_meter::Span, Instant, std::time::Duration)> = None;

    let decision = route_hdc(n, k, d, preferred, force_cpu, host);
    let bucket = ShapeBucket::of(n, k, d);
    let mut route_reason = decision.reason.to_string();

    // (scores, backend, executed device, preferred_ran, per-job seconds)
    let mut chosen: Option<(Vec<i32>, &'static str, DeviceTarget, bool, f64)> = None;
    let mut ane: Option<AneOutcome> = None;
    let mut setup_s = 0.0;

    if decision.try_ane {
        if let Some(lane) = host.ane {
            let ane_span = meter.begin();
            match run_ane_lane(queries, memory, d, lane) {
                Ok(outcome) => {
                    let ane_end = Instant::now();
                    ane_span.close(ane_end);
                    ane_window = Some((
                        ane_span,
                        ane_end,
                        std::time::Duration::from_secs_f64(outcome.setup_s.max(0.0)),
                    ));
                    setup_s += outcome.setup_s;
                    remember_placement(&host.placement_cache, bucket, outcome.planned);
                    if outcome.scores == cpu_ref {
                        let (device, backend, ran) = match outcome.planned {
                            PlannedDevice::NeuralEngine => (DeviceTarget::Npu, "coreml-ane", true),
                            PlannedDevice::Gpu => (DeviceTarget::Gpu, "coreml", false),
                            // `Unknown` is reported as CPU: with CPUAndNeuralEngine
                            // Core ML can only use CPU or ANE, and ANE is never
                            // claimed without the plan naming it.
                            PlannedDevice::Cpu | PlannedDevice::Unknown => {
                                (DeviceTarget::Cpu, "coreml", false)
                            }
                        };
                        chosen = Some((outcome.scores.clone(), backend, device, ran, outcome.predict_s));
                    } else {
                        route_reason.push_str("+ane_mismatch");
                    }
                    ane = Some(outcome);
                }
                Err(_) => {
                    // Do not retry Core ML for this bucket on every call.
                    remember_placement(&host.placement_cache, bucket, PlannedDevice::Unknown);
                    route_reason.push_str("+ane_error");
                }
            }
        }
    }

    if chosen.is_none()
        && !force_cpu
        && matches!(preferred, DeviceTarget::Npu | DeviceTarget::Gpu)
    {
        let gpu_span = meter.begin();
        if let Ok((scores, kernel_s, gpu_setup_s)) = run_gpu_hdc(queries, memory, d, host) {
            let gpu_end = Instant::now();
            gpu_span.close(gpu_end);
            gpu_window = Some((gpu_span, gpu_end, std::time::Duration::from_secs_f64(gpu_setup_s.max(0.0))));
            setup_s += gpu_setup_s;
            if scores == cpu_ref {
                chosen = Some((
                    scores,
                    "metal",
                    DeviceTarget::Gpu,
                    preferred == DeviceTarget::Gpu,
                    kernel_s,
                ));
            }
        }
    }

    let (scores, backend, executed, preferred_ran, job_s) = chosen.unwrap_or_else(|| {
        let ran = preferred == DeviceTarget::Cpu;
        (cpu_ref, "cpu", DeviceTarget::Cpu, ran, cpu_s)
    });

    let fallback = if executed == DeviceTarget::Npu && preferred_ran {
        // After the ANE: the Metal GPU when a GPU lane exists, else the CPU.
        if host.gpu.is_some() {
            DeviceTarget::Gpu
        } else {
            DeviceTarget::Cpu
        }
    } else {
        fallback_processor(preferred_ran, executed, host)
    };

    // Per-job energy covers only the kernel / prediction. Setup is separate.
    let job_s = job_s.max(1e-9);
    let measured = ane
        .as_ref()
        .filter(|_| backend == "coreml-ane")
        .and_then(|o| o.power.clone());
    let watts_estimate = snap.power_watts.max(0.0);
    let energy = match (measured, backend) {
        (Some(p), _) => match p.rails {
            // Shared sampler: all rails over the prediction loop, per prediction.
            Some(rails) => {
                let per = p.window_s / p.predictions.max(1) as f64;
                crate::power_meter::JobEnergy {
                    source: "powermetrics".into(),
                    confidence: p.confidence.into(),
                    seconds: per,
                    joules: rails.total() * per,
                    incremental_joules: p.incremental_joules_per_prediction.unwrap_or(rails.total() * per),
                    watts: rails.total(),
                    cpu_watts: Some(rails.cpu_w),
                    gpu_watts: Some(rails.gpu_w),
                    ane_watts: Some(rails.ane_w),
                    dram_watts: None,
                    baseline_watts: p
                        .incremental_joules_per_prediction
                        .map(|inc| ((rails.total() * per - inc) / per).max(0.0)),
                }
            }
            // One-off powermetrics run: ANE rail only, no baseline.
            None => crate::power_meter::JobEnergy {
                source: "powermetrics".into(),
                confidence: p.confidence.into(),
                seconds: job_s,
                joules: p.joules_per_prediction,
                incremental_joules: p.joules_per_prediction,
                watts: p.mean_ane_watts,
                cpu_watts: None,
                gpu_watts: None,
                ane_watts: Some(p.mean_ane_watts),
                dram_watts: None,
                baseline_watts: None,
            },
        },
        (None, "metal") => match gpu_window {
            Some((span, end, skip)) => meter.finish_at(span, end, skip, "gpu", watts_estimate),
            None => crate::power_meter::JobEnergy::estimate("gpu", watts_estimate, job_s),
        },
        // Core ML placed the op on the CPU or GPU: no ANE power loop ran,
        // so meter the Core ML call itself.
        (None, "coreml") => match ane_window {
            Some((span, end, skip)) => {
                meter.finish_at(span, end, skip, executed_device_label(executed), watts_estimate)
            }
            None => crate::power_meter::JobEnergy::estimate(executed_device_label(executed), watts_estimate, job_s),
        },
        (None, "cpu") => meter.finish_at(cpu_span, cpu_end, std::time::Duration::ZERO, "cpu", watts_estimate),
        (None, _) => crate::power_meter::JobEnergy::estimate(executed_device_label(executed), watts_estimate, job_s),
    };
    let (joules, watts, energy_source) = (energy.joules, energy.watts, energy.source.clone());
    let (setup_ms, setup_joules) = if setup_s > 0.0 {
        (Some(setup_s * 1000.0), Some(watts_estimate * setup_s))
    } else {
        (None, None)
    };

    let hamming = scores
        .iter()
        .map(|s| joule_ane_rt::hamming_from_dot(*s, d))
        .collect();
    let value = scores.iter().map(|s| f64::from(*s)).sum();
    let (plan, attempts, placement) = match &ane {
        Some(o) => (o.plan.clone(), o.attempts.clone(), Some(o.summary.clone())),
        None => (Vec::new(), Vec::new(), None),
    };

    Ok(HdcDispatch {
        receipt: EnergyReceipt {
            device: executed.to_string(),
            requested_device: preferred.to_string(),
            backend: backend.to_string(),
            joules,
            incremental_joules: energy.incremental_joules,
            watts,
            seconds: energy.seconds,
            energy_confidence: energy.confidence.clone(),
            cpu_watts: energy.cpu_watts,
            gpu_watts: energy.gpu_watts,
            ane_watts: energy.ane_watts,
            baseline_watts: energy.baseline_watts,
            fallback: fallback.to_string(),
            value,
            algorithm: algorithm.to_string(),
            energy_source,
            placement,
            route_reason,
            setup_ms,
            setup_joules,
        },
        scores,
        hamming,
        n,
        k,
        d,
        plan,
        attempts,
    })
}

/// The power meter for `host`: the process meter for a live host; the
/// labelled estimate for an injected descriptor host (nothing real ran).
fn meter_for(host: &HostProfile) -> &'static crate::power_meter::Meter {
    static ESTIMATE: crate::power_meter::Meter = crate::power_meter::Meter::Estimate;
    if host.live_gpu { crate::power_meter::Meter::global() } else { &ESTIMATE }
}

fn executed_device_label(d: DeviceTarget) -> &'static str {
    match d {
        DeviceTarget::Gpu => "gpu",
        DeviceTarget::Npu => "npu",
        _ => "cpu",
    }
}

fn algorithm_for_sql(sql: &str) -> AlgorithmType {
    let upper = sql.to_uppercase();
    if upper.contains("HDC") || upper.contains("HYPERDIM") {
        AlgorithmType::Hdc
    } else if upper.contains("SUM(")
        || upper.contains("AVG(")
        || upper.contains("COUNT(")
        || upper.contains("MIN(")
        || upper.contains("MAX(")
    {
        AlgorithmType::Columnar
    } else {
        AlgorithmType::BTree
    }
}

fn sample_values(response: &crate::query::QueryResponse) -> Vec<f64> {
    let mut values = Vec::new();
    for row in response.rows.iter().take(256) {
        for cell in row {
            if let Some(n) = cell.as_f64() {
                values.push(n);
            } else if let Some(n) = cell.as_i64() {
                values.push(n as f64);
            }
        }
        if values.len() >= 256 {
            break;
        }
    }
    if values.is_empty() {
        values.push(1.0);
    }
    values
}


/// Run a real reduction for this response and record the device plus joules.
///
/// Does not change a successful query into an error. The fallback processor
/// stays on the receipt; it is not turned into an "accelerator missing" warning.
pub fn attach_receipt(response: &mut crate::query::QueryResponse, sql: &str) {
    let values = sample_values(response);
    let algorithm = algorithm_for_sql(sql);
    let receipt = dispatch_sum(&values, algorithm, false);
    if response.device_target.is_none() {
        response.device_target = Some(receipt.device);
    }
    response.energy_joules = Some(receipt.joules);
    if response.power_watts.is_none() {
        response.power_watts = Some(receipt.watts);
    }
    if response.algorithm_type.is_none() {
        response.algorithm_type = Some(receipt.algorithm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_processor(name: &str) -> bool {
        matches!(name, "cpu" | "gpu" | "npu" | "tpu" | "lpu")
    }


    fn assert_processor(name: &str) {
        assert!(
            is_processor(name),
            "expected a real processor, got {name:?}"
        );
        let lower = name.to_ascii_lowercase();
        assert!(
            !lower.contains("missing") && !lower.contains("none") && !lower.contains("accelerator"),
            "processor name must not claim an accelerator is missing: {name}"
        );
    }

    fn m5_max_host() -> HostProfile {
        let ane = joule_runtime::apple_neural_engine_device_for_host(
            "macos",
            "aarch64",
            "Apple M5 Max",
        )
        .expect("injected macos-aarch64 host must yield Apple Neural Engine");
        assert_eq!(ane.kind, AcceleratorKind::NPU);
        assert_eq!(ane.name, "Apple Neural Engine");
        HostProfile {
            kinds: vec![AcceleratorKind::GPU, ane.kind],
            ane: Some(AneLane::Descriptor(PlannedDevice::NeuralEngine)),
            ane_min_work: ANE_MIN_WORK,
            placement_cache: PlacementCache::default(),
            gpu: Some(GpuLane::Metal),
            live_gpu: false,
        }
    }

    fn m5_max_npu_only() -> HostProfile {
        HostProfile {
            kinds: vec![AcceleratorKind::NPU],
            ane: None,
            ane_min_work: ANE_MIN_WORK,
            placement_cache: PlacementCache::default(),
            gpu: None,
            live_gpu: false,
        }
    }

    fn cpu_only_host() -> HostProfile {
        HostProfile {
            kinds: Vec::new(),
            ane: None,
            ane_min_work: ANE_MIN_WORK,
            placement_cache: PlacementCache::default(),
            gpu: None,
            live_gpu: false,
        }
    }

    #[test]
    fn fabric_force_cpu_runs_and_reports_joules() {
        let receipt = dispatch_sum(&[1.0, 2.0, 3.0], AlgorithmType::Columnar, true);
        assert_eq!(receipt.device, "cpu");
        assert_eq!(receipt.backend, "cpu");
        assert!((receipt.value - 6.0).abs() < 1e-9, "value={}", receipt.value);
        assert!(receipt.joules >= 0.0);
        assert_processor(&receipt.requested_device);
        assert_processor(&receipt.fallback);
    }

    /// Injected costs decide between the GPU lane and the CPU for a sum the
    /// smart route sends to the GPU; the receipt says why, and the fallback
    /// is the other real processor.
    #[test]
    fn fabric_sum_follows_injected_costs() {
        use crate::cost_model::{Candidate, CostModel};
        let mut host = m5_max_host();
        host.live_gpu = false; // descriptor lane: no Metal submit needed
        let values: Vec<f64> = (0..4096).map(|i| (i % 7) as f64).collect();
        let snap = snapshot_for(&host);
        let hints = HardwareAdvisor::new(&EnergyConfig::default()).advise(&snap);
        assert_eq!(smart_preferred(AlgorithmType::Scan, &hints, &snap, &host), DeviceTarget::Gpu);
        let op = format!("fabric:sum:{}", AlgorithmType::Scan);
        let gpu = Candidate::new("gpu", "metal");

        let mut model = CostModel::new();
        model.explore_every = 0;
        model.record(&op, values.len(), &Candidate::cpu(), 1e-4, 5e-4);
        model.record(&op, values.len(), &gpu, 1e-3, 5e-3);
        let r = dispatch_sum_costed(&values, AlgorithmType::Scan, false, &host, &model);
        println!("injected cpu cheaper: {}", r.line());
        assert_eq!((r.device.as_str(), r.backend.as_str()), ("cpu", "cpu"));
        assert_eq!(r.route_reason, "cost_model_cpu_cheaper");
        assert_eq!(r.requested_device, "gpu");
        assert_eq!(r.fallback, "gpu");
        assert_eq!(r.value, values.iter().sum::<f64>());

        let mut model = CostModel::new();
        model.explore_every = 0;
        model.record(&op, values.len(), &Candidate::cpu(), 1e-3, 5e-3);
        model.record(&op, values.len(), &gpu, 1e-4, 5e-4);
        let r = dispatch_sum_costed(&values, AlgorithmType::Scan, false, &host, &model);
        println!("injected gpu cheaper: {}", r.line());
        assert_eq!((r.device.as_str(), r.backend.as_str()), ("gpu", "metal"));
        assert_eq!(r.route_reason, "cost_model_gpu_cheaper");
        assert_eq!(r.fallback, "cpu");

        // Fresh model: static preference first, then the CPU is explored.
        let model = CostModel::new();
        let first = dispatch_sum_costed(&values, AlgorithmType::Scan, false, &host, &model);
        let second = dispatch_sum_costed(&values, AlgorithmType::Scan, false, &host, &model);
        assert_eq!((first.device.as_str(), first.route_reason.as_str()), ("gpu", "no_history"));
        assert_eq!((second.device.as_str(), second.route_reason.as_str()), ("cpu", "explore"));
        for r in [&first, &second] {
            assert_processor(&r.fallback);
            assert_processor(&r.requested_device);
        }
        // force_cpu keeps its meaning and skips the model.
        let forced = dispatch_sum_costed(&values, AlgorithmType::Scan, true, &host, &model);
        assert_eq!(forced.route_reason, "force_cpu");
    }

    #[test]
    fn fabric_live_host_always_routes_and_falls_back() {
        let receipt = dispatch_sum(&[1.0, 2.0, 3.0], AlgorithmType::Hdc, false);
        assert!((receipt.value - 6.0).abs() < 1e-2, "value={}", receipt.value);
        assert!(receipt.joules >= 0.0);
        assert_processor(&receipt.requested_device);
        assert_processor(&receipt.device);
        assert_processor(&receipt.fallback);
        assert!(!receipt.backend.is_empty());
        if receipt.requested_device == "npu" && receipt.backend == "cpu" {
            assert_eq!(receipt.device, "npu");
            assert_eq!(
                receipt.fallback, "cpu",
                "ANE present with a CPU kernel still needs a CPU fallback"
            );
        }
    }

    #[test]
    fn m5_max_hdc_routes_npu_and_runs_on_metal_gpu() {
        assert!(joule_db_energy::apple_neural_engine_present("macos", "aarch64"));
        let receipt = dispatch_sum_on(
            &[1.0, 2.0, 3.0],
            AlgorithmType::Hdc,
            false,
            &m5_max_host(),
        );
        assert!((receipt.value - 6.0).abs() < 1e-6, "value={}", receipt.value);
        assert_eq!(receipt.requested_device, "npu");
        assert_eq!(receipt.device, "gpu");
        assert_eq!(receipt.backend, "metal");
        assert_eq!(receipt.fallback, "gpu");
        assert_eq!(receipt.algorithm, "hdc");
        assert!(receipt.joules >= 0.0);
        assert!(receipt.watts > 0.0);
        assert_ne!(receipt.backend, "cpu");
    }

    #[test]
    fn m5_max_scan_gpu_backend_is_metal_and_fallback_is_cpu() {
        let receipt = dispatch_sum_on(
            &[1.0, 2.0, 3.0],
            AlgorithmType::Scan,
            false,
            &m5_max_host(),
        );
        assert!((receipt.value - 6.0).abs() < 1e-6, "value={}", receipt.value);
        assert_eq!(receipt.requested_device, "gpu");
        assert_eq!(receipt.device, "gpu");
        assert_eq!(receipt.backend, "metal");
        assert_eq!(receipt.fallback, "cpu");
        assert_ne!(receipt.backend, "cpu");
    }

    #[test]
    fn m5_max_wgpu_lane_is_wgpu_not_cpu() {
        let mut host = m5_max_host();
        host.gpu = Some(GpuLane::Wgpu);
        let receipt = dispatch_sum_on(&[1.0, 2.0, 3.0], AlgorithmType::Scan, false, &host);
        assert_eq!(receipt.requested_device, "gpu");
        assert_eq!(receipt.device, "gpu");
        assert_eq!(receipt.backend, "wgpu");
        assert_eq!(receipt.fallback, "cpu");
    }

    #[test]
    fn m5_max_npu_without_gpu_lane_falls_back_to_cpu() {
        let receipt = dispatch_sum_on(
            &[1.0, 2.0, 3.0],
            AlgorithmType::Hdc,
            false,
            &m5_max_npu_only(),
        );
        assert!((receipt.value - 6.0).abs() < 1e-9, "value={}", receipt.value);
        assert_eq!(receipt.requested_device, "npu");
        assert_eq!(receipt.device, "npu");
        assert_eq!(receipt.backend, "cpu");
        assert_eq!(receipt.fallback, "cpu");
        assert_eq!(receipt.algorithm, "hdc");
    }

    #[test]
    fn force_cpu_on_m5_still_smart_routes() {
        let receipt = dispatch_sum_on(
            &[1.0, 2.0, 3.0],
            AlgorithmType::Hdc,
            true,
            &m5_max_host(),
        );
        assert_eq!(receipt.requested_device, "npu");
        assert_eq!(receipt.device, "cpu");
        assert_eq!(receipt.backend, "cpu");
        assert_eq!(receipt.fallback, "cpu");
        assert!((receipt.value - 6.0).abs() < 1e-9);
    }

    #[test]
    fn cpu_only_host_routes_cpu_and_falls_back_to_cpu() {
        let receipt = dispatch_sum_on(
            &[1.0, 2.0, 3.0],
            AlgorithmType::Hdc,
            false,
            &cpu_only_host(),
        );
        assert!((receipt.value - 6.0).abs() < 1e-9, "value={}", receipt.value);
        assert_eq!(receipt.requested_device, "cpu");
        assert_eq!(receipt.device, "cpu");
        assert_eq!(receipt.backend, "cpu");
        assert_eq!(receipt.fallback, "cpu");
        assert_eq!(receipt.algorithm, "hdc");
        assert!(receipt.joules >= 0.0);
    }

    #[test]
    fn preferred_gpu_without_kernel_falls_back_to_cpu() {
        let host = HostProfile {
            kinds: vec![AcceleratorKind::GPU],
            ane: None,
            ane_min_work: ANE_MIN_WORK,
            placement_cache: PlacementCache::default(),
            gpu: None,
            live_gpu: true,
        };
        let receipt = dispatch_sum_on(&[1.0, 2.0, 3.0], AlgorithmType::Scan, false, &host);
        assert_eq!(receipt.requested_device, "gpu");
        assert_eq!(receipt.device, "cpu");
        assert_eq!(receipt.backend, "cpu");
        assert_eq!(receipt.fallback, "cpu");
        assert!((receipt.value - 6.0).abs() < 1e-9);
    }

    #[test]
    fn cpu_run_on_m5_max_falls_back_to_the_other_processor() {
        // BTree affinity is CPU only. The preferred device runs, so the
        // fallback is the other real processor (the Neural Engine), not None.
        let receipt = dispatch_sum_on(
            &[1.0, 2.0, 3.0],
            AlgorithmType::BTree,
            false,
            &m5_max_host(),
        );
        assert_eq!(receipt.requested_device, "cpu");
        assert_eq!(receipt.device, "cpu");
        assert_eq!(receipt.backend, "cpu");
        assert_eq!(receipt.fallback, "npu");
    }

    #[test]
    fn fallback_after_gpu_or_npu_run_is_cpu() {
        let host = HostProfile {
            kinds: vec![AcceleratorKind::GPU, AcceleratorKind::NPU],
            ane: None,
            ane_min_work: ANE_MIN_WORK,
            placement_cache: PlacementCache::default(),
            gpu: Some(GpuLane::Wgpu),
            live_gpu: false,
        };
        assert_eq!(
            fallback_processor(true, DeviceTarget::Gpu, &host),
            DeviceTarget::Cpu
        );
        assert_eq!(
            fallback_processor(true, DeviceTarget::Npu, &host),
            DeviceTarget::Cpu
        );
        assert_eq!(
            fallback_processor(false, DeviceTarget::Cpu, &cpu_only_host()),
            DeviceTarget::Cpu
        );
    }

    // ── HDC similarity lane (Core ML / ANE -> Metal -> CPU) ──────────────

    fn bipolar(len: usize, seed: u64) -> Vec<i8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state >> 63 == 0 { 1 } else { -1 }
            })
            .collect()
    }

    fn hdc_case() -> (Vec<i8>, Vec<i8>, usize) {
        let d = 256;
        (bipolar(4 * d, 3), bipolar(16 * d, 5), d)
    }

    fn host_with_ane(planned: PlannedDevice, gpu: Option<GpuLane>) -> HostProfile {
        let mut host = m5_max_host();
        host.ane = Some(AneLane::Descriptor(planned));
        // Tiny test jobs: put every job above the Core ML threshold.
        host.ane_min_work = 0;
        host.gpu = gpu;
        if gpu.is_none() {
            host.kinds = vec![AcceleratorKind::NPU];
        }
        host
    }

    fn assert_hdc_receipt_sane(out: &HdcDispatch, q: &[i8], m: &[i8], d: usize) {
        let want = joule_ane_rt::cpu_dot_scores(q, m, d);
        assert_eq!(out.scores, want);
        assert_eq!(out.hamming.len(), want.len());
        assert_processor(&out.receipt.requested_device);
        assert_processor(&out.receipt.device);
        assert_processor(&out.receipt.fallback);
        assert!(out.receipt.joules >= 0.0);
        assert_eq!(out.receipt.algorithm, "hdc");
        assert!(matches!(
            out.receipt.backend.as_str(),
            "coreml-ane" | "coreml" | "metal" | "cpu"
        ));
        assert!(matches!(
            out.receipt.energy_source.as_str(),
            "estimate" | "powermetrics"
        ));
        if out.receipt.backend == "coreml-ane" {
            assert_eq!(out.receipt.device, "npu");
        }
        if out.receipt.device == "npu" {
            assert_eq!(out.receipt.backend, "coreml-ane", "npu is claimed only for a plan on the Neural Engine");
        }
    }

    #[test]
    fn hdc_ane_on_neural_engine_reports_npu_coreml_ane_then_gpu() {
        let (q, m, d) = hdc_case();
        let host = host_with_ane(PlannedDevice::NeuralEngine, Some(GpuLane::Metal));
        let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_hdc_receipt_sane(&out, &q, &m, d);
        assert_eq!(out.receipt.requested_device, "npu");
        assert_eq!(out.receipt.device, "npu");
        assert_eq!(out.receipt.backend, "coreml-ane");
        assert_eq!(out.receipt.fallback, "gpu");
        assert_eq!(out.receipt.energy_source, "estimate");
        assert!(out.receipt.placement.as_deref().unwrap_or("").ends_with("@neural_engine"));
    }

    #[test]
    fn hdc_ane_without_gpu_lane_falls_back_to_cpu() {
        let (q, m, d) = hdc_case();
        let host = host_with_ane(PlannedDevice::NeuralEngine, None);
        let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_hdc_receipt_sane(&out, &q, &m, d);
        assert_eq!(out.receipt.device, "npu");
        assert_eq!(out.receipt.backend, "coreml-ane");
        assert_eq!(out.receipt.fallback, "cpu");
    }

    #[test]
    fn hdc_coreml_placed_on_cpu_is_reported_as_cpu() {
        let (q, m, d) = hdc_case();
        for planned in [PlannedDevice::Cpu, PlannedDevice::Unknown] {
            let host = host_with_ane(planned, Some(GpuLane::Metal));
            let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
            assert_hdc_receipt_sane(&out, &q, &m, d);
            assert_eq!(out.receipt.requested_device, "npu");
            assert_eq!(out.receipt.device, "cpu");
            assert_eq!(out.receipt.backend, "coreml");
            assert_eq!(out.receipt.fallback, "cpu");
            assert_eq!(out.receipt.energy_source, "estimate");
        }
    }

    #[test]
    fn hdc_coreml_placed_on_gpu_is_reported_as_gpu() {
        let (q, m, d) = hdc_case();
        let host = host_with_ane(PlannedDevice::Gpu, Some(GpuLane::Metal));
        let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_hdc_receipt_sane(&out, &q, &m, d);
        assert_eq!(out.receipt.device, "gpu");
        assert_eq!(out.receipt.backend, "coreml");
        assert_eq!(out.receipt.fallback, "gpu");
    }

    #[test]
    fn hdc_without_ane_lane_runs_metal_gpu() {
        let (q, m, d) = hdc_case();
        let mut host = m5_max_host();
        host.ane = None;
        let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_hdc_receipt_sane(&out, &q, &m, d);
        assert_eq!(out.receipt.requested_device, "npu");
        assert_eq!(out.receipt.device, "gpu");
        assert_eq!(out.receipt.backend, "metal");
        assert_eq!(out.receipt.fallback, "gpu");
        assert!(out.receipt.placement.is_none());
    }

    #[test]
    fn hdc_force_cpu_and_cpu_only_hosts_run_on_cpu() {
        let (q, m, d) = hdc_case();
        let out = dispatch_hdc_on(&q, &m, d, true, &m5_max_host()).expect("dispatch");
        assert_hdc_receipt_sane(&out, &q, &m, d);
        assert_eq!(out.receipt.requested_device, "npu");
        assert_eq!(out.receipt.device, "cpu");
        assert_eq!(out.receipt.backend, "cpu");
        assert_eq!(out.receipt.fallback, "cpu");

        let out = dispatch_hdc_on(&q, &m, d, false, &cpu_only_host()).expect("dispatch");
        assert_hdc_receipt_sane(&out, &q, &m, d);
        assert_eq!(out.receipt.requested_device, "cpu");
        assert_eq!(out.receipt.device, "cpu");
        assert_eq!(out.receipt.fallback, "cpu");
    }

    #[test]
    fn hdc_fallback_is_never_none_on_any_host() {
        let (q, m, d) = hdc_case();
        let planned = [
            PlannedDevice::NeuralEngine,
            PlannedDevice::Gpu,
            PlannedDevice::Cpu,
            PlannedDevice::Unknown,
        ];
        let mut hosts = vec![cpu_only_host(), m5_max_npu_only(), m5_max_host()];
        for p in planned {
            for gpu in [None, Some(GpuLane::Metal), Some(GpuLane::Wgpu)] {
                hosts.push(host_with_ane(p, gpu));
            }
        }
        for host in &hosts {
            for force in [false, true] {
                let out = dispatch_hdc_on(&q, &m, d, force, host).expect("dispatch");
                assert_hdc_receipt_sane(&out, &q, &m, d);
                assert!(!out.receipt.fallback.is_empty());
            }
        }
    }

    #[test]
    fn hdc_rejects_malformed_input_but_not_for_energy() {
        assert!(dispatch_hdc_on(&[1, 0], &[1, 1], 2, false, &cpu_only_host()).is_err());
        assert!(dispatch_hdc_on(&[1, 1, 1], &[1, 1], 2, false, &cpu_only_host()).is_err());
    }

    #[test]
    fn hdc_hamming_matches_binary_hypervector() {
        use joule_db_hdc::BinaryHyperVector;
        let d = 1024;
        let queries: Vec<BinaryHyperVector> =
            (0..3).map(|i| BinaryHyperVector::random(d, 100 + i)).collect();
        let memory: Vec<BinaryHyperVector> =
            (0..9).map(|i| BinaryHyperVector::random(d, 200 + i)).collect();
        let out = dispatch_hdc_hamming(&queries, &memory, false).expect("dispatch");
        for (i, qv) in queries.iter().enumerate() {
            for (j, mv) in memory.iter().enumerate() {
                assert_eq!(out.hamming[i * memory.len() + j], qv.hamming_distance(mv));
            }
        }
        assert_processor(&out.receipt.device);
        assert_processor(&out.receipt.fallback);
        println!("hdc live receipt: {}", out.receipt.line());
    }

    #[test]
    fn sum_receipts_are_estimates() {
        let receipt = dispatch_sum_on(&[1.0, 2.0], AlgorithmType::Scan, false, &m5_max_host());
        assert_eq!(receipt.energy_source, "estimate");
        assert!(receipt.placement.is_none());
        assert!(receipt.line().contains("energy_source=estimate"));
    }

    // ── Live macOS checks (run on the Mac) ───────────────────────────────

    #[cfg(target_os = "macos")]
    #[test]
    fn live_mac_scan_runs_on_metal_gpu_and_matches_cpu() {
        // A real Mac always has the Metal lane. A virtualized runner may not;
        // there the receipt must name the processor that actually ran.
        let metal_lane = live_profile().gpu == Some(GpuLane::Metal);
        if !metal_lane {
            assert!(
                !joule_metal_rt::hardware_required(),
                "{}=1 but the live profile has no Metal lane",
                joule_metal_rt::REQUIRE_HW_ENV
            );
            println!("live scan: no Metal lane on this host; checking the fallback receipt instead");
        }
        for n in [1usize, 255, 256, 257, 4096, 100_003] {
            let values: Vec<f64> = (0..n).map(|i| ((i * 2_654_435_761usize) % 5 + 1) as f64).collect();
            // The smart route without the cost model, so the Metal lane is
            // what runs (cost routing has its own tests).
            let receipt = dispatch_sum_on(&values, AlgorithmType::Scan, false, &live_profile());
            let cpu = cpu_sum(&values).expect("cpu sum");
            println!("live scan n={n}: cpu={cpu} {}", receipt.line());
            assert_eq!(receipt.value, cpu, "integer-valued sums are exact in f32 below 2^24");
            assert_processor(&receipt.requested_device);
            assert_processor(&receipt.device);
            assert_processor(&receipt.fallback);
            if metal_lane {
                assert_eq!(receipt.requested_device, "gpu");
                assert_eq!(receipt.device, "gpu");
                assert_eq!(receipt.backend, "metal");
            } else {
                assert_ne!(receipt.backend, "metal", "no Metal lane, so Metal cannot be claimed");
                match receipt.backend.as_str() {
                    "wgpu" => assert_eq!(receipt.device, "gpu"),
                    "cpu" => assert_eq!(receipt.device, "cpu"),
                    other => panic!("unexpected backend {other:?}"),
                }
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn live_mac_profile_has_the_coreml_lane() {
        let host = live_profile();
        if !kind_on(&host, AcceleratorKind::NPU) && !joule_metal_rt::hardware_required() {
            println!(
                "SKIP live_mac_profile_has_the_coreml_lane: no Neural Engine detected on this host; set {}=1 to fail instead",
                joule_metal_rt::REQUIRE_HW_ENV
            );
            assert_eq!(host.ane, None, "no NPU detected, so no Core ML lane may be claimed");
            return;
        }
        assert_eq!(host.ane, Some(AneLane::Live), "macOS aarch64 with an ANE gets the Core ML lane");
        assert_eq!(host.ane_min_work, ANE_MIN_WORK);
    }

    // ── Size-aware HDC routing ───────────────────────────────────────────

    #[test]
    fn ane_threshold_matches_measured_m5_max_placement() {
        assert_eq!(ANE_MIN_WORK, 64 * 1024 * 2048);
        let host = host_with_ane(PlannedDevice::NeuralEngine, Some(GpuLane::Metal));
        let host = HostProfile { ane_min_work: ANE_MIN_WORK, ..host };
        let route = |n, k, d| route_hdc(n, k, d, DeviceTarget::Npu, false, &host);
        // Measured CPU placements stay off Core ML.
        for (n, k, d) in [(1, 64, 2048), (8, 64, 2048), (64, 256, 2048), (8, 64, 1024)] {
            assert_eq!(route(n, k, d), HdcRoute { try_ane: false, reason: "below_ane_threshold" });
        }
        // Measured Neural Engine placements probe Core ML.
        for (n, k, d) in [(64, 1024, 2048), (256, 1024, 2048), (512, 4096, 2048), (1024, 1024, 2048)] {
            assert_eq!(route(n, k, d), HdcRoute { try_ane: true, reason: "ane_probe" });
        }
    }

    #[test]
    fn small_hdc_job_skips_coreml_and_runs_metal_or_cpu() {
        let (q, m, d) = hdc_case();
        for (gpu, device, backend, fallback) in [
            (Some(GpuLane::Metal), "gpu", "metal", "gpu"),
            (None, "cpu", "cpu", "cpu"),
            (Some(GpuLane::Wgpu), "cpu", "cpu", "cpu"),
        ] {
            let mut host = host_with_ane(PlannedDevice::NeuralEngine, gpu);
            host.ane_min_work = ANE_MIN_WORK;
            let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
            assert_hdc_receipt_sane(&out, &q, &m, d);
            assert_eq!(out.receipt.requested_device, "npu");
            assert_eq!(out.receipt.route_reason, "below_ane_threshold");
            assert_eq!(out.receipt.device, device);
            assert_eq!(out.receipt.backend, backend);
            assert_eq!(out.receipt.fallback, fallback);
            assert!(out.receipt.placement.is_none(), "Core ML must not be invoked");
            assert!(out.plan.is_empty());
            assert!(
                host.placement_cache.lock().map(|c| c.is_empty()).unwrap_or(false),
                "below-threshold jobs do not probe placement"
            );
        }
    }

    #[test]
    fn large_hdc_job_probes_ane_then_honors_cached_neural_engine() {
        let (q, m, d) = hdc_case();
        let host = host_with_ane(PlannedDevice::NeuralEngine, Some(GpuLane::Metal));
        let first = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_eq!(first.receipt.route_reason, "ane_probe");
        assert_eq!(first.receipt.device, "npu");
        assert_eq!(first.receipt.backend, "coreml-ane");
        assert_eq!(
            cached_placement(&host.placement_cache, ShapeBucket::of(4, 16, d)),
            Some(PlannedDevice::NeuralEngine)
        );
        let second = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_hdc_receipt_sane(&second, &q, &m, d);
        assert_eq!(second.receipt.route_reason, "ane_placement_cached");
        assert_eq!(second.receipt.device, "npu");
        assert_eq!(second.receipt.backend, "coreml-ane");
        assert_eq!(second.receipt.fallback, "gpu");
    }

    #[test]
    fn cached_off_ane_placement_skips_coreml_next_time() {
        let (q, m, d) = hdc_case();
        for planned in [PlannedDevice::Cpu, PlannedDevice::Gpu, PlannedDevice::Unknown] {
            let host = host_with_ane(planned, Some(GpuLane::Metal));
            let first = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
            assert_eq!(first.receipt.route_reason, "ane_probe");
            assert_eq!(first.receipt.backend, "coreml");
            assert!(first.receipt.placement.is_some());
            let second = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
            assert_hdc_receipt_sane(&second, &q, &m, d);
            assert_eq!(second.receipt.route_reason, "ane_placement_cached_off_ane");
            assert_eq!(second.receipt.requested_device, "npu");
            assert_eq!(second.receipt.device, "gpu");
            assert_eq!(second.receipt.backend, "metal");
            assert_eq!(second.receipt.fallback, "gpu");
            assert!(second.receipt.placement.is_none(), "Core ML must not be invoked again");
        }
    }

    #[test]
    fn cached_placement_overrides_threshold_and_buckets_are_distinct() {
        let (q, m, d) = hdc_case();
        let mut host = host_with_ane(PlannedDevice::NeuralEngine, Some(GpuLane::Metal));
        host.ane_min_work = ANE_MIN_WORK;
        remember_placement(&host.placement_cache, ShapeBucket::of(4, 16, d), PlannedDevice::NeuralEngine);
        let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert_eq!(out.receipt.route_reason, "ane_placement_cached");
        assert_eq!(out.receipt.backend, "coreml-ane");
        assert_ne!(ShapeBucket::of(64, 256, 2048), ShapeBucket::of(64, 1024, 2048));
        assert_eq!(ShapeBucket::of(60, 1000, 2048), ShapeBucket::of(64, 1024, 2048));
        assert_eq!(ShapeBucket::of(4, 32, 3000).tiles, 2);
    }

    #[test]
    fn route_reasons_for_non_ane_requests() {
        let (q, m, d) = hdc_case();
        let out = dispatch_hdc_on(&q, &m, d, true, &m5_max_host()).expect("dispatch");
        assert_eq!(out.receipt.route_reason, "force_cpu");
        let out = dispatch_hdc_on(&q, &m, d, false, &cpu_only_host()).expect("dispatch");
        assert_eq!(out.receipt.route_reason, "requested_cpu");
        let mut no_lane = m5_max_host();
        no_lane.ane = None;
        let out = dispatch_hdc_on(&q, &m, d, false, &no_lane).expect("dispatch");
        assert_eq!(out.receipt.route_reason, "no_ane_lane");
        assert_eq!(out.receipt.backend, "metal");
    }

    #[test]
    fn descriptor_runs_report_no_setup_and_job_only_estimates() {
        let (q, m, d) = hdc_case();
        let host = host_with_ane(PlannedDevice::NeuralEngine, Some(GpuLane::Metal));
        let out = dispatch_hdc_on(&q, &m, d, false, &host).expect("dispatch");
        assert!(out.receipt.setup_ms.is_none());
        assert!(out.receipt.setup_joules.is_none());
        assert_eq!(out.receipt.energy_source, "estimate");
        assert!(out.receipt.line().contains("setup_ms=none"));
        assert!(out.receipt.line().contains("route_reason=ane_probe"));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn live_mac_hdc_small_large_and_repeat_receipts() {
        use joule_db_hdc::BinaryHyperVector;
        let vectors = |count: usize, d: usize, seed: u64| -> Vec<BinaryHyperVector> {
            (0..count).map(|i| BinaryHyperVector::random(d, seed + i as u64)).collect()
        };
        // Real Macs have both lanes. A virtualized runner may lack either;
        // then the receipts must say what actually ran instead.
        let host = live_profile();
        let npu_detected = kind_on(&host, AcceleratorKind::NPU);
        let metal_lane = host.gpu == Some(GpuLane::Metal);
        let ane_lane = host.ane.is_some();
        let check = |label: &str, q: &[BinaryHyperVector], m: &[BinaryHyperVector]| -> HdcDispatch {
            let out = dispatch_hdc_hamming(q, m, false).expect("dispatch");
            let k = m.len();
            let exact = q.iter().enumerate().all(|(i, qv)| {
                m.iter().enumerate().all(|(j, mv)| out.hamming[i * k + j] == qv.hamming_distance(mv))
            });
            println!("{label} n={} k={} d={} hamming_match={}", q.len(), k, out.d, if exact { "yes" } else { "NO" });
            for line in &out.plan {
                println!("  plan op: {line}");
            }
            println!("  receipt: {}", out.receipt.line());
            assert!(exact);
            assert_processor(&out.receipt.requested_device);
            assert_processor(&out.receipt.device);
            assert_processor(&out.receipt.fallback);
            if npu_detected {
                assert_eq!(out.receipt.requested_device, "npu");
            }
            out
        };

        if !(metal_lane && ane_lane) {
            assert!(
                !joule_metal_rt::hardware_required(),
                "{}=1 but the live profile lacks a lane: metal={metal_lane} coreml={ane_lane}",
                joule_metal_rt::REQUIRE_HW_ENV
            );
            println!("live hdc: metal_lane={metal_lane} coreml_lane={ane_lane}; checking fallback receipts");
        }

        let small = check("small", &vectors(8, 1024, 11_000), &vectors(64, 1024, 12_000));
        assert!(small.receipt.placement.is_none(), "small jobs never invoke Core ML");
        if ane_lane {
            assert_eq!(small.receipt.route_reason, "below_ane_threshold");
        }
        if metal_lane {
            assert_eq!(small.receipt.backend, "metal");
        } else {
            assert_eq!(small.receipt.backend, "cpu");
            assert_eq!(small.receipt.device, "cpu");
        }

        let q = vectors(64, 2048, 21_000);
        let m = vectors(1024, 2048, 22_000);
        let first = check("large_first", &q, &m);
        let second = check("large_second", &q, &m);
        if !ane_lane {
            assert!(
                !first.receipt.route_reason.starts_with("ane_"),
                "no Core ML lane, so no ANE route: {}",
                first.receipt.route_reason
            );
            assert!(first.receipt.placement.is_none(), "no Core ML lane, so no placement");
            assert_ne!(first.receipt.backend, "coreml-ane");
            return;
        }
        assert!(first.receipt.placement.is_some());
        assert!(first.receipt.setup_ms.is_some(), "first large call loads the model");
        assert!(second.receipt.setup_ms.is_none(), "second call reuses the loaded model");
        if first.receipt.backend == "coreml-ane" {
            assert_eq!(second.receipt.route_reason, "ane_placement_cached");
        } else {
            assert_eq!(second.receipt.route_reason, "ane_placement_cached_off_ane");
        }
    }

    /// Mean of a receipt field over the receipts of one device.
    fn power_row(job: &str, size: usize, rs: &[&EnergyReceipt]) {
        if rs.is_empty() {
            return;
        }
        let n = rs.len() as f64;
        let mean = |f: &dyn Fn(&EnergyReceipt) -> f64| rs.iter().map(|r| f(r)).sum::<f64>() / n;
        let opt = |f: &dyn Fn(&EnergyReceipt) -> Option<f64>| {
            let v: Vec<f64> = rs.iter().filter_map(|r| f(r)).collect();
            if v.is_empty() { "none".to_string() } else { format!("{:.3}", v.iter().sum::<f64>() / v.len() as f64) }
        };
        let conf: Vec<&str> = rs.iter().map(|r| r.energy_confidence.as_str()).collect();
        eprintln!(
            "POWER-TABLE job={job} size={size} device={} backend={} runs={} ms={:.3} J={:.4e} incr_J={:.4e} W={:.2} cpu_w={} gpu_w={} ane_w={} base_w={} source={} confidence={:?}",
            rs[0].device,
            rs[0].backend,
            rs.len(),
            mean(&|r| r.seconds) * 1e3,
            mean(&|r| r.joules),
            mean(&|r| r.incremental_joules),
            mean(&|r| r.watts),
            opt(&|r| r.cpu_watts),
            opt(&|r| r.gpu_watts),
            opt(&|r| r.ane_watts),
            opt(&|r| r.baseline_watts),
            rs[0].energy_source,
            conf
        );
    }

    /// Mac live (run by hand): measured CPU/GPU/ANE watts and joules for a
    /// cost-routed GPU sum and an HDC job on the ANE, Metal and CPU.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore]
    fn measured_power_live_fabric() {
        use crate::cost_model::{Candidate, CostModel};
        let meter = crate::power_meter::Meter::global();
        eprintln!("POWER meter={}", meter.source());
        // Let the sampler collect idle history for the baseline.
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let host = live_profile();
        for n in [1usize << 20, 1 << 24] {
            let values: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.5).collect();
            let mut model = CostModel::new();
            model.explore_every = 2;
            let rs: Vec<EnergyReceipt> =
                (0..12).map(|_| dispatch_sum_costed(&values, AlgorithmType::Scan, false, &host, &model)).collect();
            for r in &rs {
                eprintln!("POWER sum n={n} {}", r.line());
                assert_processor(&r.device);
                assert_processor(&r.fallback);
            }
            for dev in ["gpu", "cpu"] {
                power_row("sum", n, &rs.iter().skip(1).filter(|r| r.device == dev).collect::<Vec<_>>());
            }
            let op = format!("fabric:sum:{}", AlgorithmType::Scan);
            if let Some((e, t, by)) = model.rankings(&op, n, &[Candidate::new("gpu", "metal"), Candidate::cpu()]) {
                eprintln!(
                    "POWER-ROUTE job=sum size={n} ranked_by={} energy_choice={} time_choice={} flipped={}",
                    by.label(),
                    e.device,
                    t.device,
                    e != t
                );
            }
        }
        let d = 2048;
        let (q, m) = (bipolar(64 * d, 7), bipolar(1024 * d, 9));
        let mut no_ane = live_profile();
        no_ane.ane = None;
        for _ in 0..2 {
            let ane = dispatch_hdc_on(&q, &m, d, false, &host).expect("ane");
            let metal = dispatch_hdc_on(&q, &m, d, false, &no_ane).expect("metal");
            let cpu = dispatch_hdc_on(&q, &m, d, true, &host).expect("cpu");
            for (label, out) in [("ane", &ane), ("metal", &metal), ("cpu", &cpu)] {
                eprintln!("POWER hdc lane={label} {}", out.receipt.line());
                power_row("hdc_64x1024x2048", 64 * 1024, &[&out.receipt]);
            }
        }
    }
}

