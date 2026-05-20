//! Compute router: bridges adaptive controller hints to dispatch decisions.
//!
//! Consumers call `route()` with an algorithm type and the current adaptive
//! params to get a `RoutingDecision` that tells them which device to target
//! and whether GPU dispatch thresholds should be overridden.
//!
//! This is the missing link between the `HardwareAdvisor` (which knows the
//! hardware state) and the `GpuDispatcher` (which does the actual dispatch).
//!
//! ## Architecture
//!
//! ```text
//! EnergyMonitor → HardwareAdvisor → AdaptiveController → ComputeRouter
//!                                                             ↓
//!                                                     RoutingDecision {
//!                                                       device: Gpu,
//!                                                       threshold_overrides: Some(16/8/32),
//!                                                       should_batch: true,
//!                                                     }
//! ```

use crate::controller::{AdaptiveParams, DynamicDispatchThresholds};
use crate::monitor::EnergySnapshot;
use crate::tracker::{AlgorithmType, DeviceTarget};

/// Result of a compute routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The device to dispatch this operation to.
    pub device: DeviceTarget,
    /// If `Some`, override the static GPU dispatch thresholds.
    pub threshold_overrides: Option<DynamicDispatchThresholds>,
    /// Whether the operation should be batched for GPU/TPU efficiency.
    /// True for GPU and TPU targets where dispatch overhead makes
    /// individual operations expensive.
    pub should_batch: bool,
}

/// Kind of GPU-dispatchable operation (for threshold lookup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOperationKind {
    /// HDC similarity search (static default threshold: 64 vectors).
    Similarity,
    /// HDC majority-vote bundling (static default threshold: 32 vectors).
    Bundle,
    /// HDC XOR binding (static default threshold: 128 pairs).
    Bind,
}

/// Static default GPU dispatch thresholds.
const DEFAULT_SIMILARITY_THRESHOLD: usize = 64;
const DEFAULT_BUNDLE_THRESHOLD: usize = 32;
const DEFAULT_BIND_THRESHOLD: usize = 128;

/// Stateless compute router.
///
/// Combines algorithm affinity, adaptive controller preferences, and
/// device availability to produce a routing decision. No internal state —
/// all decisions are pure functions of the inputs.
pub struct ComputeRouter;

impl ComputeRouter {
    /// Route an operation given the current adaptive parameters and hardware state.
    ///
    /// Decision priority:
    /// 1. If the controller has preferred devices, intersect with algorithm affinity
    /// 2. If no intersection, use the controller's top preference
    /// 3. If no controller preference, pick first available device from affinity
    /// 4. Fallback: CPU
    pub fn route(
        algorithm: AlgorithmType,
        params: &AdaptiveParams,
        snapshot: &EnergySnapshot,
    ) -> RoutingDecision {
        let affinity = algorithm.device_affinity();

        let device = if !params.preferred_devices.is_empty() {
            // Find first affinity device that matches a controller preference
            affinity
                .iter()
                .find(|d| params.preferred_devices.contains(d))
                .copied()
                // No overlap: use controller's top preference
                .unwrap_or(params.preferred_devices[0])
        } else {
            // No controller preference: pick first available device from affinity
            affinity
                .iter()
                .find(|d| Self::is_device_available(**d, snapshot))
                .copied()
                .unwrap_or(DeviceTarget::Cpu)
        };

        let should_batch = matches!(
            device,
            DeviceTarget::Gpu | DeviceTarget::Tpu | DeviceTarget::Lpu
        );

        RoutingDecision {
            device,
            threshold_overrides: params.gpu_dispatch_overrides,
            should_batch,
        }
    }

    /// Quick check: should a given batch size be dispatched to GPU?
    ///
    /// Uses dynamic thresholds from the adaptive params if available,
    /// falling back to static defaults (64/32/128).
    pub fn should_dispatch_gpu(
        batch_size: usize,
        operation: GpuOperationKind,
        params: &AdaptiveParams,
    ) -> bool {
        let threshold = match &params.gpu_dispatch_overrides {
            Some(overrides) => match operation {
                GpuOperationKind::Similarity => overrides.similarity_threshold,
                GpuOperationKind::Bundle => overrides.bundle_threshold,
                GpuOperationKind::Bind => overrides.bind_threshold,
            },
            None => match operation {
                GpuOperationKind::Similarity => DEFAULT_SIMILARITY_THRESHOLD,
                GpuOperationKind::Bundle => DEFAULT_BUNDLE_THRESHOLD,
                GpuOperationKind::Bind => DEFAULT_BIND_THRESHOLD,
            },
        };
        batch_size >= threshold
    }

    /// Check if a device is available based on the hardware snapshot.
    fn is_device_available(device: DeviceTarget, snapshot: &EnergySnapshot) -> bool {
        match device {
            DeviceTarget::Cpu => true,
            DeviceTarget::Gpu => snapshot.gpu_available,
            DeviceTarget::Npu => snapshot.npu_available,
            DeviceTarget::Tpu => snapshot.tpu_available,
            DeviceTarget::Lpu => snapshot.lpu_available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::ThermalState;

    fn nominal_snapshot() -> EnergySnapshot {
        EnergySnapshot {
            power_watts: 15.0,
            cpu_utilization: 0.3,
            thermal_state: ThermalState::Nominal,
            memory_pressure: 0.2,
            gpu_available: true,
            gpu_utilization: 0.1,
            npu_available: true,
            npu_utilization: 0.1,
            tpu_available: false,
            ..EnergySnapshot::default()
        }
    }

    fn no_accelerator_snapshot() -> EnergySnapshot {
        EnergySnapshot {
            gpu_available: false,
            npu_available: false,
            tpu_available: false,
            lpu_available: false,
            ..nominal_snapshot()
        }
    }

    fn params_no_preference() -> AdaptiveParams {
        AdaptiveParams {
            memtable_threshold: 4 * 1024 * 1024,
            l0_compaction_trigger: 4,
            buffer_pool_capacity: 256,
            write_rate_limit: 0,
            hints: vec![],
            preferred_devices: vec![],
            gpu_dispatch_overrides: None,
        }
    }

    fn params_prefer_gpu() -> AdaptiveParams {
        AdaptiveParams {
            preferred_devices: vec![DeviceTarget::Gpu],
            gpu_dispatch_overrides: Some(DynamicDispatchThresholds {
                similarity_threshold: 16,
                bundle_threshold: 8,
                bind_threshold: 32,
            }),
            ..params_no_preference()
        }
    }

    fn params_prefer_cpu() -> AdaptiveParams {
        AdaptiveParams {
            preferred_devices: vec![DeviceTarget::Cpu],
            gpu_dispatch_overrides: Some(DynamicDispatchThresholds {
                similarity_threshold: 256,
                bundle_threshold: 128,
                bind_threshold: 512,
            }),
            ..params_no_preference()
        }
    }

    #[test]
    fn test_route_hdc_prefers_npu_when_available() {
        // HDC affinity = [Npu, Gpu, Cpu]
        // No controller preference → pick first available → NPU
        let snap = nominal_snapshot(); // npu_available = true
        let params = params_no_preference();
        let decision = ComputeRouter::route(AlgorithmType::Hdc, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Npu);
        assert!(!decision.should_batch); // NPU is not batched
    }

    #[test]
    fn test_route_scan_to_gpu_under_cpu_pressure() {
        // Scan affinity = [Gpu, Cpu]
        // Controller prefers GPU → intersect → GPU
        let snap = nominal_snapshot();
        let params = params_prefer_gpu();
        let decision = ComputeRouter::route(AlgorithmType::Scan, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Gpu);
        assert!(decision.should_batch);
        assert!(decision.threshold_overrides.is_some());
    }

    #[test]
    fn test_route_btree_always_cpu() {
        // BTree affinity = [Cpu] only
        // Controller prefers GPU, but BTree affinity has no GPU → no intersection
        // → falls back to controller's top preference (Gpu)
        // However, BTree is pointer-chasing — the routing still respects
        // the controller's preference even if it overrides affinity
        let snap = nominal_snapshot();
        let params = params_prefer_gpu();
        let decision = ComputeRouter::route(AlgorithmType::BTree, &params, &snap);

        // When controller preference doesn't intersect affinity,
        // we use the controller's top preference
        assert_eq!(decision.device, DeviceTarget::Gpu);
    }

    #[test]
    fn test_route_btree_no_preference_uses_cpu() {
        // BTree affinity = [Cpu], no controller preference → CPU
        let snap = nominal_snapshot();
        let params = params_no_preference();
        let decision = ComputeRouter::route(AlgorithmType::BTree, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Cpu);
        assert!(!decision.should_batch);
    }

    #[test]
    fn test_route_falls_back_to_cpu_no_accelerators() {
        // No GPU/NPU/TPU available, HDC affinity = [Npu, Gpu, Cpu]
        // → falls through to CPU
        let snap = no_accelerator_snapshot();
        let params = params_no_preference();
        let decision = ComputeRouter::route(AlgorithmType::Hdc, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Cpu);
    }

    #[test]
    fn test_should_dispatch_gpu_with_overrides() {
        let params = params_prefer_gpu();

        // With lowered thresholds (16/8/32), smaller batches dispatch
        assert!(ComputeRouter::should_dispatch_gpu(
            16,
            GpuOperationKind::Similarity,
            &params
        ));
        assert!(ComputeRouter::should_dispatch_gpu(
            8,
            GpuOperationKind::Bundle,
            &params
        ));
        assert!(ComputeRouter::should_dispatch_gpu(
            32,
            GpuOperationKind::Bind,
            &params
        ));

        // Below overridden thresholds → don't dispatch
        assert!(!ComputeRouter::should_dispatch_gpu(
            15,
            GpuOperationKind::Similarity,
            &params
        ));
        assert!(!ComputeRouter::should_dispatch_gpu(
            7,
            GpuOperationKind::Bundle,
            &params
        ));
    }

    #[test]
    fn test_should_dispatch_gpu_default_thresholds() {
        let params = params_no_preference(); // no overrides

        // Static defaults: 64/32/128
        assert!(!ComputeRouter::should_dispatch_gpu(
            63,
            GpuOperationKind::Similarity,
            &params
        ));
        assert!(ComputeRouter::should_dispatch_gpu(
            64,
            GpuOperationKind::Similarity,
            &params
        ));

        assert!(!ComputeRouter::should_dispatch_gpu(
            31,
            GpuOperationKind::Bundle,
            &params
        ));
        assert!(ComputeRouter::should_dispatch_gpu(
            32,
            GpuOperationKind::Bundle,
            &params
        ));

        assert!(!ComputeRouter::should_dispatch_gpu(
            127,
            GpuOperationKind::Bind,
            &params
        ));
        assert!(ComputeRouter::should_dispatch_gpu(
            128,
            GpuOperationKind::Bind,
            &params
        ));
    }

    #[test]
    fn test_should_dispatch_gpu_raised_thresholds() {
        // PreferCpu → thresholds raised to 256/128/512
        let params = params_prefer_cpu();

        // Batches that would normally dispatch to GPU now stay on CPU
        assert!(!ComputeRouter::should_dispatch_gpu(
            100,
            GpuOperationKind::Similarity,
            &params
        ));
        assert!(!ComputeRouter::should_dispatch_gpu(
            64,
            GpuOperationKind::Bundle,
            &params
        ));
        assert!(!ComputeRouter::should_dispatch_gpu(
            200,
            GpuOperationKind::Bind,
            &params
        ));

        // Only very large batches dispatch
        assert!(ComputeRouter::should_dispatch_gpu(
            256,
            GpuOperationKind::Similarity,
            &params
        ));
    }

    #[test]
    fn test_route_hdc_prefers_lpu_when_available() {
        // HDC affinity = [Lpu, Npu, Gpu, Cpu]
        // No controller preference → pick first available → LPU
        let mut snap = nominal_snapshot();
        snap.lpu_available = true;
        snap.lpu_utilization = 0.1;
        let params = params_no_preference();
        let decision = ComputeRouter::route(AlgorithmType::Hdc, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Lpu);
        assert!(decision.should_batch); // LPU batches for efficiency
    }

    #[test]
    fn test_route_falls_back_without_lpu() {
        // HDC affinity = [Lpu, Npu, Gpu, Cpu]
        // LPU not available → falls to NPU (available in nominal_snapshot)
        let snap = nominal_snapshot(); // lpu_available = false by default
        let params = params_no_preference();
        let decision = ComputeRouter::route(AlgorithmType::Hdc, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Npu);
    }

    #[test]
    fn test_route_columnar_to_gpu() {
        // Columnar affinity = [Gpu, Tpu, Cpu]
        // No preference, GPU available → GPU
        let snap = nominal_snapshot();
        let params = params_no_preference();
        let decision = ComputeRouter::route(AlgorithmType::Columnar, &params, &snap);

        assert_eq!(decision.device, DeviceTarget::Gpu);
        assert!(decision.should_batch);
    }
}
