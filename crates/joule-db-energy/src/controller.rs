//! Adaptive storage controller.
//!
//! Consumes `EnergySnapshot` from the monitor and dynamically tunes
//! storage engine parameters (memtable threshold, compaction triggers,
//! buffer pool capacity, write rate limit) and **compute device routing**
//! (preferred device, GPU dispatch thresholds) based on `HardwareAdvisor` hints.
//!
//! This is the feedback loop that makes JouleDB the world's first database
//! that modulates its behavior in response to real hardware telemetry —
//! including heterogeneous compute routing across CPU, GPU, NPU, and TPU.

use crate::EnergyConfig;
use crate::advisor::{ExecutionHint, HardwareAdvisor};
use crate::monitor::EnergySnapshot;
use crate::tracker::DeviceTarget;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Dynamic overrides for GPU dispatch thresholds.
///
/// When the CPU is overloaded and GPU is idle, the controller lowers
/// these thresholds to offload more work to GPU. When GPU is overloaded
/// or thermal pressure mandates CPU-only, thresholds are raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicDispatchThresholds {
    /// Override for similarity search dispatch threshold (static default: 64).
    pub similarity_threshold: usize,
    /// Override for bundle dispatch threshold (static default: 32).
    pub bundle_threshold: usize,
    /// Override for bind dispatch threshold (static default: 128).
    pub bind_threshold: usize,
}

/// Baseline storage parameters captured at startup for recovery.
#[derive(Debug, Clone)]
pub struct AdaptiveBaseline {
    /// Default memtable flush threshold in bytes.
    pub memtable_threshold: usize,
    /// Default L0 compaction trigger (number of SSTables).
    pub l0_compaction_trigger: usize,
    /// Default buffer pool capacity per shard.
    pub buffer_pool_capacity: usize,
}

/// Current effective storage parameters (snapshot for consumers).
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    /// Effective memtable flush threshold in bytes.
    pub memtable_threshold: usize,
    /// Effective L0 compaction trigger.
    pub l0_compaction_trigger: usize,
    /// Effective buffer pool capacity per shard.
    pub buffer_pool_capacity: usize,
    /// Write rate limit in ops/sec (0 = unlimited).
    pub write_rate_limit: u64,
    /// Hints that produced these params.
    pub hints: Vec<ExecutionHint>,
    /// Preferred compute devices in priority order.
    /// Empty means no preference — use algorithm affinity defaults.
    pub preferred_devices: Vec<DeviceTarget>,
    /// Dynamic GPU dispatch threshold overrides.
    /// `None` means use static defaults (64/32/128).
    pub gpu_dispatch_overrides: Option<DynamicDispatchThresholds>,
}

/// Adaptive storage controller.
///
/// Reads hardware state from a shared `EnergySnapshot`, evaluates
/// `HardwareAdvisor` rules, and publishes new storage parameters
/// via lock-free atomics for zero-contention reads from the hot path.
pub struct AdaptiveController {
    advisor: HardwareAdvisor,
    monitor: Arc<RwLock<EnergySnapshot>>,
    // Current effective parameters — atomics for lock-free hot-path reads.
    memtable_threshold: AtomicUsize,
    l0_compaction_trigger: AtomicUsize,
    buffer_pool_capacity: AtomicUsize,
    write_rate_limit: AtomicU64,
    // Device routing — atomics for lock-free hot-path reads.
    // Encoding: 0=None, 1=Cpu, 2=Gpu, 3=Npu, 4=Tpu, 5=Lpu
    primary_device_preference: AtomicU8,
    // Dynamic GPU dispatch thresholds (0 = no override, use static defaults).
    gpu_similarity_threshold: AtomicUsize,
    gpu_bundle_threshold: AtomicUsize,
    gpu_bind_threshold: AtomicUsize,
    // Baseline values for recovery to nominal.
    baseline: AdaptiveBaseline,
    // 0 = static (always return baseline), 1 = adaptive.
    mode: AtomicU8,
}

impl AdaptiveController {
    /// Create a new adaptive controller.
    ///
    /// Starts in adaptive mode. Use `set_mode(false)` for static behavior.
    pub fn new(
        config: &EnergyConfig,
        baseline: AdaptiveBaseline,
        monitor: Arc<RwLock<EnergySnapshot>>,
    ) -> Self {
        Self {
            advisor: HardwareAdvisor::new(config),
            memtable_threshold: AtomicUsize::new(baseline.memtable_threshold),
            l0_compaction_trigger: AtomicUsize::new(baseline.l0_compaction_trigger),
            buffer_pool_capacity: AtomicUsize::new(baseline.buffer_pool_capacity),
            write_rate_limit: AtomicU64::new(0), // unlimited at start
            primary_device_preference: AtomicU8::new(0), // no preference
            gpu_similarity_threshold: AtomicUsize::new(0), // no override
            gpu_bundle_threshold: AtomicUsize::new(0),
            gpu_bind_threshold: AtomicUsize::new(0),
            baseline,
            mode: AtomicU8::new(1), // adaptive by default
            monitor,
        }
    }

    /// Switch between adaptive (true) and static (false) mode.
    pub fn set_mode(&self, adaptive: bool) {
        self.mode
            .store(if adaptive { 1 } else { 0 }, Ordering::Release);
        if !adaptive {
            // Immediately restore baseline
            self.memtable_threshold
                .store(self.baseline.memtable_threshold, Ordering::Release);
            self.l0_compaction_trigger
                .store(self.baseline.l0_compaction_trigger, Ordering::Release);
            self.buffer_pool_capacity
                .store(self.baseline.buffer_pool_capacity, Ordering::Release);
            self.write_rate_limit.store(0, Ordering::Release);
            self.primary_device_preference.store(0, Ordering::Release);
            self.gpu_similarity_threshold.store(0, Ordering::Release);
            self.gpu_bundle_threshold.store(0, Ordering::Release);
            self.gpu_bind_threshold.store(0, Ordering::Release);
        }
    }

    /// Whether the controller is in adaptive mode.
    pub fn is_adaptive(&self) -> bool {
        self.mode.load(Ordering::Acquire) == 1
    }

    /// Get the baseline values.
    pub fn baseline(&self) -> &AdaptiveBaseline {
        &self.baseline
    }

    /// Re-evaluate hardware state and update effective parameters.
    ///
    /// Call this periodically (e.g., every 2s aligned with the energy monitor).
    /// Returns the new effective parameters.
    pub fn tick(&self) -> AdaptiveParams {
        if !self.is_adaptive() {
            return self.static_params();
        }

        let snapshot = self.monitor.read().map(|s| s.clone()).unwrap_or_default();

        let hints = self.advisor.advise(&snapshot);
        let params = self.apply_hints(&hints);

        // Publish atomically — storage params
        self.memtable_threshold
            .store(params.memtable_threshold, Ordering::Release);
        self.l0_compaction_trigger
            .store(params.l0_compaction_trigger, Ordering::Release);
        self.buffer_pool_capacity
            .store(params.buffer_pool_capacity, Ordering::Release);
        self.write_rate_limit
            .store(params.write_rate_limit, Ordering::Release);

        // Publish atomically — device routing
        let primary = params
            .preferred_devices
            .first()
            .map(|d| device_to_u8(*d))
            .unwrap_or(0);
        self.primary_device_preference
            .store(primary, Ordering::Release);
        match &params.gpu_dispatch_overrides {
            Some(ov) => {
                self.gpu_similarity_threshold
                    .store(ov.similarity_threshold, Ordering::Release);
                self.gpu_bundle_threshold
                    .store(ov.bundle_threshold, Ordering::Release);
                self.gpu_bind_threshold
                    .store(ov.bind_threshold, Ordering::Release);
            }
            None => {
                self.gpu_similarity_threshold.store(0, Ordering::Release);
                self.gpu_bundle_threshold.store(0, Ordering::Release);
                self.gpu_bind_threshold.store(0, Ordering::Release);
            }
        }

        params
    }

    /// Read current effective parameters (lock-free, suitable for hot path).
    pub fn current_params(&self) -> AdaptiveParams {
        if !self.is_adaptive() {
            return self.static_params();
        }

        let primary = self.primary_device_preference.load(Ordering::Acquire);
        let preferred_devices = match u8_to_device(primary) {
            Some(d) => vec![d],
            None => vec![],
        };

        let sim_thresh = self.gpu_similarity_threshold.load(Ordering::Acquire);
        let gpu_dispatch_overrides = if sim_thresh > 0 {
            Some(DynamicDispatchThresholds {
                similarity_threshold: sim_thresh,
                bundle_threshold: self.gpu_bundle_threshold.load(Ordering::Acquire),
                bind_threshold: self.gpu_bind_threshold.load(Ordering::Acquire),
            })
        } else {
            None
        };

        AdaptiveParams {
            memtable_threshold: self.memtable_threshold.load(Ordering::Acquire),
            l0_compaction_trigger: self.l0_compaction_trigger.load(Ordering::Acquire),
            buffer_pool_capacity: self.buffer_pool_capacity.load(Ordering::Acquire),
            write_rate_limit: self.write_rate_limit.load(Ordering::Acquire),
            hints: Vec::new(), // Not tracked in atomics
            preferred_devices,
            gpu_dispatch_overrides,
        }
    }

    // --- Private helpers ---

    fn static_params(&self) -> AdaptiveParams {
        AdaptiveParams {
            memtable_threshold: self.baseline.memtable_threshold,
            l0_compaction_trigger: self.baseline.l0_compaction_trigger,
            buffer_pool_capacity: self.baseline.buffer_pool_capacity,
            write_rate_limit: 0,
            hints: vec![ExecutionHint::Normal],
            preferred_devices: vec![],
            gpu_dispatch_overrides: None,
        }
    }

    fn apply_hints(&self, hints: &[ExecutionHint]) -> AdaptiveParams {
        let mut memtable = self.baseline.memtable_threshold;
        let mut l0_trigger = self.baseline.l0_compaction_trigger;
        let mut buffer_cap = self.baseline.buffer_pool_capacity;
        let mut rate_limit: u64 = 0; // unlimited
        let mut preferred_devices: Vec<DeviceTarget> = Vec::new();
        let mut gpu_overrides: Option<DynamicDispatchThresholds> = None;

        for hint in hints {
            match hint {
                ExecutionHint::Normal => {
                    // Keep baseline (already set)
                }
                ExecutionHint::ReduceBatchSize { suggested_factor } => {
                    let factor = *suggested_factor as usize;
                    // Increase memtable threshold → fewer flushes → less I/O energy
                    memtable = memtable.saturating_mul(factor);
                    // Defer compaction → avoid CPU-intensive merge during thermal events
                    l0_trigger = l0_trigger.saturating_add(factor);
                    // Slow writes proportionally
                    let fraction = match factor {
                        4 => 4, // 25% of baseline throughput
                        _ => 2, // 50% of baseline throughput
                    };
                    // Use a reasonable base rate (100K ops/s baseline)
                    let base_rate: u64 = 100_000;
                    let limited = base_rate / fraction as u64;
                    // Take the more restrictive limit
                    if rate_limit == 0 || limited < rate_limit {
                        rate_limit = limited;
                    }
                }
                ExecutionHint::Throttle => {
                    // Hard braking
                    memtable = memtable.saturating_mul(4);
                    l0_trigger = l0_trigger.saturating_add(4);
                    let hard_limit: u64 = 1_000; // 1K ops/s floor
                    if rate_limit == 0 || hard_limit < rate_limit {
                        rate_limit = hard_limit;
                    }
                }
                ExecutionHint::ReduceBufferPool => {
                    // Free memory for the OS
                    buffer_cap = buffer_cap / 2;
                    // Minimum floor: 16 entries per shard
                    if buffer_cap < 16 {
                        buffer_cap = 16;
                    }
                }
                ExecutionHint::PreferGpu => {
                    if !preferred_devices.contains(&DeviceTarget::Gpu) {
                        preferred_devices.push(DeviceTarget::Gpu);
                    }
                    // CPU overloaded, GPU idle → aggressively lower GPU dispatch
                    // thresholds to offload more work
                    gpu_overrides = Some(DynamicDispatchThresholds {
                        similarity_threshold: 16, // down from 64
                        bundle_threshold: 8,      // down from 32
                        bind_threshold: 32,       // down from 128
                    });
                }
                ExecutionHint::PreferNpu => {
                    if !preferred_devices.contains(&DeviceTarget::Npu) {
                        preferred_devices.push(DeviceTarget::Npu);
                    }
                }
                ExecutionHint::PreferTpu => {
                    if !preferred_devices.contains(&DeviceTarget::Tpu) {
                        preferred_devices.push(DeviceTarget::Tpu);
                    }
                }
                ExecutionHint::PreferLpu => {
                    if !preferred_devices.contains(&DeviceTarget::Lpu) {
                        preferred_devices.push(DeviceTarget::Lpu);
                    }
                }
                ExecutionHint::PreferCpu => {
                    if !preferred_devices.contains(&DeviceTarget::Cpu) {
                        preferred_devices.push(DeviceTarget::Cpu);
                    }
                    // GPU overloaded or thermal pressure → raise thresholds to
                    // keep work on CPU (avoid adding GPU heat)
                    if gpu_overrides.is_none() {
                        gpu_overrides = Some(DynamicDispatchThresholds {
                            similarity_threshold: 256, // up from 64
                            bundle_threshold: 128,     // up from 32
                            bind_threshold: 512,       // up from 128
                        });
                    }
                }
            }
        }

        AdaptiveParams {
            memtable_threshold: memtable,
            l0_compaction_trigger: l0_trigger,
            buffer_pool_capacity: buffer_cap,
            write_rate_limit: rate_limit,
            hints: hints.to_vec(),
            preferred_devices,
            gpu_dispatch_overrides: gpu_overrides,
        }
    }
}

// --- Atomic encoding helpers for DeviceTarget ---

fn device_to_u8(d: DeviceTarget) -> u8 {
    match d {
        DeviceTarget::Cpu => 1,
        DeviceTarget::Gpu => 2,
        DeviceTarget::Npu => 3,
        DeviceTarget::Tpu => 4,
        DeviceTarget::Lpu => 5,
    }
}

fn u8_to_device(v: u8) -> Option<DeviceTarget> {
    match v {
        1 => Some(DeviceTarget::Cpu),
        2 => Some(DeviceTarget::Gpu),
        3 => Some(DeviceTarget::Npu),
        4 => Some(DeviceTarget::Tpu),
        5 => Some(DeviceTarget::Lpu),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::ThermalState;

    fn default_baseline() -> AdaptiveBaseline {
        AdaptiveBaseline {
            memtable_threshold: 4 * 1024 * 1024, // 4MB
            l0_compaction_trigger: 4,
            buffer_pool_capacity: 256,
        }
    }

    fn nominal_snapshot() -> EnergySnapshot {
        EnergySnapshot {
            power_watts: 15.0,
            cpu_utilization: 0.3,
            thermal_state: ThermalState::Nominal,
            memory_pressure: 0.2,
            gpu_available: true,
            gpu_utilization: 0.1,
            ..EnergySnapshot::default()
        }
    }

    fn make_controller(snapshot: EnergySnapshot) -> AdaptiveController {
        let monitor = Arc::new(RwLock::new(snapshot));
        let config = EnergyConfig::default();
        AdaptiveController::new(&config, default_baseline(), monitor)
    }

    fn make_controller_with_monitor(
        snapshot: EnergySnapshot,
    ) -> (AdaptiveController, Arc<RwLock<EnergySnapshot>>) {
        let monitor = Arc::new(RwLock::new(snapshot));
        let config = EnergyConfig::default();
        let ctrl = AdaptiveController::new(&config, default_baseline(), monitor.clone());
        (ctrl, monitor)
    }

    #[test]
    fn test_nominal_returns_baseline() {
        let ctrl = make_controller(nominal_snapshot());
        let params = ctrl.tick();

        assert_eq!(params.memtable_threshold, 4 * 1024 * 1024);
        assert_eq!(params.l0_compaction_trigger, 4);
        assert_eq!(params.buffer_pool_capacity, 256);
        assert_eq!(params.write_rate_limit, 0); // unlimited
        assert!(params.hints.contains(&ExecutionHint::Normal));
        assert!(params.preferred_devices.is_empty());
        assert!(params.gpu_dispatch_overrides.is_none());
    }

    #[test]
    fn test_thermal_serious_increases_memtable() {
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Serious;
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        // Serious → ReduceBatchSize{2} → 2× memtable
        assert!(params.memtable_threshold > 4 * 1024 * 1024);
        assert_eq!(params.memtable_threshold, 2 * 4 * 1024 * 1024);
        // Deferred compaction
        assert!(params.l0_compaction_trigger > 4);
    }

    #[test]
    fn test_thermal_critical_full_throttle() {
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Critical;
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        // Critical → Throttle + ReduceBatchSize{4}
        // Multiple hints stack: both Throttle and ReduceBatchSize{4} apply
        assert!(params.memtable_threshold >= 4 * 4 * 1024 * 1024);
        assert!(params.l0_compaction_trigger >= 8); // baseline 4 + at least 4
        assert!(params.write_rate_limit > 0); // rate limited
        assert!(params.write_rate_limit <= 1_000); // hard throttle floor
    }

    #[test]
    fn test_power_envelope_throttles() {
        let mut snap = nominal_snapshot();
        // TDP = 30W, threshold = 0.8 → 24W triggers. 28W > 24W.
        snap.power_watts = 28.0;
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        assert!(params.write_rate_limit > 0);
        assert!(
            params
                .hints
                .iter()
                .any(|h| matches!(h, ExecutionHint::Throttle))
        );
    }

    #[test]
    fn test_memory_pressure_reduces_buffer() {
        let mut snap = nominal_snapshot();
        snap.memory_pressure = 0.85; // > 0.7 threshold
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        assert_eq!(params.buffer_pool_capacity, 128); // 256 / 2
        assert!(params.hints.contains(&ExecutionHint::ReduceBufferPool));
    }

    #[test]
    fn test_recovery_to_baseline() {
        let (ctrl, monitor) = make_controller_with_monitor(EnergySnapshot {
            thermal_state: ThermalState::Critical,
            power_watts: 28.0,
            cpu_utilization: 0.9,
            ..nominal_snapshot()
        });

        // Tick under thermal pressure
        let stressed = ctrl.tick();
        assert!(stressed.write_rate_limit > 0);

        // Recover: set snapshot back to nominal
        {
            let mut snap = monitor.write().unwrap();
            *snap = nominal_snapshot();
        }

        let recovered = ctrl.tick();
        assert_eq!(recovered.memtable_threshold, 4 * 1024 * 1024);
        assert_eq!(recovered.l0_compaction_trigger, 4);
        assert_eq!(recovered.buffer_pool_capacity, 256);
        assert_eq!(recovered.write_rate_limit, 0);
    }

    #[test]
    fn test_mode_toggle() {
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Critical;
        snap.power_watts = 28.0;
        let ctrl = make_controller(snap);

        // Adaptive mode: should modulate
        assert!(ctrl.is_adaptive());
        let adaptive = ctrl.tick();
        assert!(adaptive.write_rate_limit > 0);

        // Switch to static: should return baseline
        ctrl.set_mode(false);
        assert!(!ctrl.is_adaptive());
        let static_params = ctrl.current_params();
        assert_eq!(static_params.memtable_threshold, 4 * 1024 * 1024);
        assert_eq!(static_params.l0_compaction_trigger, 4);
        assert_eq!(static_params.write_rate_limit, 0);

        // Switch back to adaptive
        ctrl.set_mode(true);
        assert!(ctrl.is_adaptive());
    }

    #[test]
    fn test_concurrent_reads() {
        let ctrl = Arc::new(make_controller(nominal_snapshot()));
        let mut handles = Vec::new();

        // Tick once to populate
        ctrl.tick();

        for _ in 0..8 {
            let c = ctrl.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    let p = c.current_params();
                    assert!(p.memtable_threshold > 0);
                    assert!(p.l0_compaction_trigger > 0);
                    assert!(p.buffer_pool_capacity > 0);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_tick_updates_atomics() {
        let (ctrl, monitor) = make_controller_with_monitor(nominal_snapshot());

        // Initial tick — nominal
        ctrl.tick();
        let before = ctrl.current_params();
        assert_eq!(before.write_rate_limit, 0);

        // Change to Critical
        {
            let mut snap = monitor.write().unwrap();
            snap.thermal_state = ThermalState::Critical;
            snap.power_watts = 28.0;
        }

        ctrl.tick();
        let after = ctrl.current_params();
        assert!(after.write_rate_limit > 0);
        assert!(after.memtable_threshold > before.memtable_threshold);
    }

    #[test]
    fn test_buffer_pool_minimum_floor() {
        let baseline = AdaptiveBaseline {
            memtable_threshold: 4 * 1024 * 1024,
            l0_compaction_trigger: 4,
            buffer_pool_capacity: 20, // Very small
        };
        let monitor = Arc::new(RwLock::new(EnergySnapshot {
            memory_pressure: 0.85,
            ..nominal_snapshot()
        }));
        let ctrl = AdaptiveController::new(&EnergyConfig::default(), baseline, monitor);
        let params = ctrl.tick();

        // 20 / 2 = 10, but floor is 16
        assert_eq!(params.buffer_pool_capacity, 16);
    }

    #[test]
    fn test_gpu_offload_hint_no_storage_change() {
        let mut snap = nominal_snapshot();
        snap.cpu_utilization = 0.9;
        snap.gpu_utilization = 0.1;
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        // GPU offload hint should not change storage params
        assert_eq!(params.memtable_threshold, 4 * 1024 * 1024);
        assert_eq!(params.l0_compaction_trigger, 4);
        assert_eq!(params.buffer_pool_capacity, 256);
        assert_eq!(params.write_rate_limit, 0);

        // BUT it should now publish a device preference and lower GPU thresholds
        assert!(params.preferred_devices.contains(&DeviceTarget::Gpu));
        assert!(params.gpu_dispatch_overrides.is_some());
        let ov = params.gpu_dispatch_overrides.unwrap();
        assert_eq!(ov.similarity_threshold, 16);
        assert_eq!(ov.bundle_threshold, 8);
        assert_eq!(ov.bind_threshold, 32);
    }

    #[test]
    fn test_energy_savings_proof() {
        // PROOF: Under thermal stress, adaptive mode reduces I/O work.
        //
        // Fewer flushes = fewer SSTable writes = less disk I/O = less energy.
        // Deferred compaction = less CPU-intensive merging during thermal events.
        // Rate limiting = less work per unit time = lower power draw.

        let mut stressed = nominal_snapshot();
        stressed.thermal_state = ThermalState::Serious;

        let ctrl = make_controller(stressed);
        let params = ctrl.tick();
        let baseline = default_baseline();

        // Proof 1: Memtable threshold increased → fewer flushes
        assert!(
            params.memtable_threshold > baseline.memtable_threshold,
            "adaptive should increase memtable threshold (fewer flushes = less I/O energy)"
        );

        // Proof 2: L0 trigger increased → deferred compaction
        assert!(
            params.l0_compaction_trigger > baseline.l0_compaction_trigger,
            "adaptive should defer compaction (less CPU work during thermal stress)"
        );

        // Proof 3: Write rate limited → less work per second
        assert!(
            params.write_rate_limit > 0,
            "adaptive should rate-limit writes (lower power draw)"
        );
    }

    #[test]
    fn test_static_mode_unchanged() {
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Critical;
        snap.power_watts = 28.0;
        let ctrl = make_controller(snap);

        // Switch to static immediately
        ctrl.set_mode(false);
        let params = ctrl.tick();

        // Static mode must return exact baseline — no modulation
        assert_eq!(params.memtable_threshold, 4 * 1024 * 1024);
        assert_eq!(params.l0_compaction_trigger, 4);
        assert_eq!(params.buffer_pool_capacity, 256);
        assert_eq!(params.write_rate_limit, 0);
        assert!(params.preferred_devices.is_empty());
        assert!(params.gpu_dispatch_overrides.is_none());
    }

    // ========================================================================
    // Heterogeneous compute tests
    // ========================================================================

    #[test]
    fn test_gpu_offload_publishes_preference() {
        // CPU overloaded (0.9), GPU idle (0.1) → PreferGpu → lower thresholds
        let snap = EnergySnapshot {
            cpu_utilization: 0.9,
            gpu_available: true,
            gpu_utilization: 0.1,
            ..nominal_snapshot()
        };
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        assert!(params.preferred_devices.contains(&DeviceTarget::Gpu));
        let ov = params
            .gpu_dispatch_overrides
            .expect("should have GPU threshold overrides");
        assert_eq!(ov.similarity_threshold, 16);
        assert_eq!(ov.bundle_threshold, 8);
        assert_eq!(ov.bind_threshold, 32);

        // Verify lock-free read path agrees
        let read = ctrl.current_params();
        assert!(read.preferred_devices.contains(&DeviceTarget::Gpu));
        assert!(read.gpu_dispatch_overrides.is_some());
    }

    #[test]
    fn test_thermal_prefers_cpu_raises_thresholds() {
        // Serious thermal → PreferCpu (rule 1) → raise GPU thresholds
        let snap = EnergySnapshot {
            thermal_state: ThermalState::Serious,
            ..nominal_snapshot()
        };
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        assert!(params.preferred_devices.contains(&DeviceTarget::Cpu));
        let ov = params
            .gpu_dispatch_overrides
            .expect("should have raised GPU thresholds");
        assert_eq!(ov.similarity_threshold, 256);
        assert_eq!(ov.bundle_threshold, 128);
        assert_eq!(ov.bind_threshold, 512);
    }

    #[test]
    fn test_npu_offload_publishes_preference() {
        // CPU overloaded, NPU idle and available → PreferNpu
        let snap = EnergySnapshot {
            cpu_utilization: 0.9,
            npu_available: true,
            npu_utilization: 0.1,
            // GPU also available and idle → will also get PreferGpu
            gpu_available: true,
            gpu_utilization: 0.1,
            ..nominal_snapshot()
        };
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        assert!(
            params.preferred_devices.contains(&DeviceTarget::Npu),
            "NPU should be in preferred devices"
        );
        // Both GPU and NPU should be preferred since both are idle
        assert!(params.preferred_devices.contains(&DeviceTarget::Gpu));
    }

    #[test]
    fn test_lpu_offload_publishes_preference() {
        // CPU overloaded, LPU idle and available → PreferLpu
        let snap = EnergySnapshot {
            cpu_utilization: 0.9,
            lpu_available: true,
            lpu_utilization: 0.1,
            // GPU also available and idle → will also get PreferGpu
            gpu_available: true,
            gpu_utilization: 0.1,
            ..nominal_snapshot()
        };
        let ctrl = make_controller(snap);
        let params = ctrl.tick();

        assert!(
            params.preferred_devices.contains(&DeviceTarget::Lpu),
            "LPU should be in preferred devices"
        );
        // Both GPU and LPU should be preferred since both are idle
        assert!(params.preferred_devices.contains(&DeviceTarget::Gpu));

        // Verify lock-free read path agrees
        let read = ctrl.current_params();
        // Primary device is the first one published (Gpu since PreferGpu fires first)
        assert!(!read.preferred_devices.is_empty());
    }

    #[test]
    fn test_dynamic_thresholds_recovery() {
        // Start with CPU pressure → GPU thresholds lowered
        let (ctrl, monitor) = make_controller_with_monitor(EnergySnapshot {
            cpu_utilization: 0.9,
            gpu_available: true,
            gpu_utilization: 0.1,
            ..nominal_snapshot()
        });

        let stressed = ctrl.tick();
        assert!(stressed.gpu_dispatch_overrides.is_some());
        assert!(stressed.preferred_devices.contains(&DeviceTarget::Gpu));

        // Recovery: CPU load drops → no more device hints
        {
            let mut snap = monitor.write().unwrap();
            *snap = nominal_snapshot(); // cpu_utilization = 0.3
        }

        let recovered = ctrl.tick();
        assert!(
            recovered.preferred_devices.is_empty(),
            "no device preference after recovery"
        );
        assert!(
            recovered.gpu_dispatch_overrides.is_none(),
            "no threshold overrides after recovery"
        );

        // Verify lock-free read agrees
        let read = ctrl.current_params();
        assert!(read.preferred_devices.is_empty());
        assert!(read.gpu_dispatch_overrides.is_none());
    }
}
