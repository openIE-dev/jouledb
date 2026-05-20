//! Per-operation energy tracker.
//!
//! RAII guard that estimates energy consumption of database operations.
//! Follows the `HistogramTimer` pattern: starts on creation, records on drop.

use crate::monitor::EnergySnapshot;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Type of database operation being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OperationType {
    Read,
    Write,
    Search,
    Bind,
    Scan,
    Aggregate,
    TernaryMatvec,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Search => write!(f, "search"),
            Self::Bind => write!(f, "bind"),
            Self::Scan => write!(f, "scan"),
            Self::Aggregate => write!(f, "aggregate"),
            Self::TernaryMatvec => write!(f, "ternary_matvec"),
        }
    }
}

/// Target device for the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceTarget {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    /// Language Processing Unit (Groq/xAI inference chips).
    Lpu,
}

impl std::fmt::Display for DeviceTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
            Self::Npu => write!(f, "npu"),
            Self::Tpu => write!(f, "tpu"),
            Self::Lpu => write!(f, "lpu"),
        }
    }
}

/// Algorithm used for the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AlgorithmType {
    Hdc,
    BTree,
    Scan,
    Columnar,
    Holographic,
    TernaryHdc,
}

impl std::fmt::Display for AlgorithmType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hdc => write!(f, "hdc"),
            Self::BTree => write!(f, "btree"),
            Self::Scan => write!(f, "scan"),
            Self::Columnar => write!(f, "columnar"),
            Self::Holographic => write!(f, "holographic"),
            Self::TernaryHdc => write!(f, "ternary_hdc"),
        }
    }
}

impl AlgorithmType {
    /// Returns preferred device targets in priority order for this algorithm.
    /// First element is the most preferred device.
    pub fn device_affinity(&self) -> &'static [DeviceTarget] {
        match self {
            // Sequential pointer-chasing, cache/branch-predictor dependent
            Self::BTree => &[DeviceTarget::Cpu],
            // Embarrassingly parallel row iteration
            Self::Scan => &[DeviceTarget::Gpu, DeviceTarget::Cpu],
            // Dense bit-vector XOR/popcount — LPU deterministic SRAM excels
            Self::Hdc => &[
                DeviceTarget::Lpu,
                DeviceTarget::Npu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu,
            ],
            // Vectorized column-at-a-time aggregation
            Self::Columnar => &[
                DeviceTarget::Gpu,
                DeviceTarget::Tpu,
                DeviceTarget::Lpu,
                DeviceTarget::Cpu,
            ],
            // Tensor-like holographic transforms
            Self::Holographic => &[
                DeviceTarget::Npu,
                DeviceTarget::Lpu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu,
            ],
            // Ternary matrix-vector: 2-bit ops — LPU deterministic dispatch ideal
            Self::TernaryHdc => &[
                DeviceTarget::Lpu,
                DeviceTarget::Npu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu,
            ],
        }
    }
}

/// Recorded energy observation for a completed operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EnergyObservation {
    pub operation: OperationType,
    pub device: DeviceTarget,
    pub algorithm: AlgorithmType,
    pub duration_secs: f64,
    pub estimated_joules: f64,
    pub power_watts_at_start: f64,
}

/// RAII guard that estimates energy consumption on drop.
///
/// Usage:
/// ```ignore
/// let _guard = OperationEnergyTracker::start(
///     &snapshot_handle,
///     OperationType::Search,
///     DeviceTarget::Cpu,
///     AlgorithmType::Hdc,
///     |obs| metrics.record_observation(&obs),
/// );
/// // ... perform operation ...
/// // energy recorded automatically when _guard drops
/// ```
pub struct OperationEnergyTracker {
    snapshot_handle: Arc<RwLock<EnergySnapshot>>,
    start: Instant,
    power_at_start: f64,
    operation: OperationType,
    device: DeviceTarget,
    algorithm: AlgorithmType,
    on_complete: Option<Box<dyn FnOnce(EnergyObservation) + Send>>,
}

impl OperationEnergyTracker {
    /// Start tracking an operation. Energy is estimated and callback invoked on drop.
    pub fn start(
        snapshot_handle: &Arc<RwLock<EnergySnapshot>>,
        operation: OperationType,
        device: DeviceTarget,
        algorithm: AlgorithmType,
        on_complete: impl FnOnce(EnergyObservation) + Send + 'static,
    ) -> Self {
        let power_at_start = snapshot_handle.read().map(|s| s.power_watts).unwrap_or(0.0);

        Self {
            snapshot_handle: snapshot_handle.clone(),
            start: Instant::now(),
            power_at_start,
            operation,
            device,
            algorithm,
            on_complete: Some(Box::new(on_complete)),
        }
    }

    /// Create a tracker without a callback (observation is discarded).
    pub fn start_silent(
        snapshot_handle: &Arc<RwLock<EnergySnapshot>>,
        operation: OperationType,
        device: DeviceTarget,
        algorithm: AlgorithmType,
    ) -> Self {
        let power_at_start = snapshot_handle.read().map(|s| s.power_watts).unwrap_or(0.0);

        Self {
            snapshot_handle: snapshot_handle.clone(),
            start: Instant::now(),
            power_at_start,
            operation,
            device,
            algorithm,
            on_complete: None,
        }
    }

    /// Get the elapsed time since tracking started.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

impl Drop for OperationEnergyTracker {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();

        // Read current utilization for a better energy estimate
        let utilization = self
            .snapshot_handle
            .read()
            .map(|snap| match self.device {
                DeviceTarget::Cpu => snap.cpu_utilization,
                DeviceTarget::Gpu => snap.gpu_utilization,
                DeviceTarget::Npu => snap.npu_utilization,
                DeviceTarget::Tpu => snap.tpu_utilization,
                DeviceTarget::Lpu => snap.lpu_utilization,
            })
            .unwrap_or(0.5);

        // Energy = Power × Time × Utilization_factor
        // Utilization_factor accounts for this query's share of total power
        let joules = self.power_at_start * duration * utilization.max(0.01);

        let observation = EnergyObservation {
            operation: self.operation,
            device: self.device,
            algorithm: self.algorithm,
            duration_secs: duration,
            estimated_joules: joules,
            power_watts_at_start: self.power_at_start,
        };

        if let Some(callback) = self.on_complete.take() {
            callback(observation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(power: f64, cpu: f64) -> Arc<RwLock<EnergySnapshot>> {
        Arc::new(RwLock::new(EnergySnapshot {
            power_watts: power,
            cpu_utilization: cpu,
            ..EnergySnapshot::default()
        }))
    }

    #[test]
    fn test_tracker_records_on_drop() {
        let snapshot = make_snapshot(30.0, 0.5);
        let received = Arc::new(std::sync::Mutex::new(None));
        let received_clone = received.clone();

        {
            let _guard = OperationEnergyTracker::start(
                &snapshot,
                OperationType::Search,
                DeviceTarget::Cpu,
                AlgorithmType::Hdc,
                move |obs| {
                    *received_clone.lock().unwrap() = Some(obs);
                },
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        } // guard drops here

        let obs = received.lock().unwrap().take().unwrap();
        assert_eq!(obs.operation, OperationType::Search);
        assert_eq!(obs.device, DeviceTarget::Cpu);
        assert_eq!(obs.algorithm, AlgorithmType::Hdc);
        assert!(obs.duration_secs >= 0.008); // At least ~8ms
        assert!(obs.estimated_joules > 0.0);
        assert_eq!(obs.power_watts_at_start, 30.0);
    }

    #[test]
    fn test_tracker_silent_no_panic() {
        let snapshot = make_snapshot(30.0, 0.5);
        {
            let _guard = OperationEnergyTracker::start_silent(
                &snapshot,
                OperationType::Read,
                DeviceTarget::Cpu,
                AlgorithmType::BTree,
            );
        }
        // No panic on drop without callback
    }

    #[test]
    fn test_energy_scales_with_power() {
        let low_power = make_snapshot(10.0, 0.5);
        let high_power = make_snapshot(60.0, 0.5);

        let low_obs = Arc::new(std::sync::Mutex::new(None));
        let high_obs = Arc::new(std::sync::Mutex::new(None));

        let low_clone = low_obs.clone();
        let high_clone = high_obs.clone();

        {
            let _g = OperationEnergyTracker::start(
                &low_power,
                OperationType::Read,
                DeviceTarget::Cpu,
                AlgorithmType::BTree,
                move |obs| {
                    *low_clone.lock().unwrap() = Some(obs);
                },
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        {
            let _g = OperationEnergyTracker::start(
                &high_power,
                OperationType::Read,
                DeviceTarget::Cpu,
                AlgorithmType::BTree,
                move |obs| {
                    *high_clone.lock().unwrap() = Some(obs);
                },
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let low_j = low_obs.lock().unwrap().as_ref().unwrap().estimated_joules;
        let high_j = high_obs.lock().unwrap().as_ref().unwrap().estimated_joules;

        // Higher power → more joules (roughly 6x for 60W vs 10W)
        assert!(high_j > low_j);
    }

    #[test]
    fn test_algorithm_device_affinity() {
        // BTree: CPU only
        assert_eq!(AlgorithmType::BTree.device_affinity(), &[DeviceTarget::Cpu]);
        // Scan: GPU first, then CPU
        assert_eq!(
            AlgorithmType::Scan.device_affinity(),
            &[DeviceTarget::Gpu, DeviceTarget::Cpu]
        );
        // Hdc: LPU first (deterministic SRAM), then NPU, GPU, CPU
        assert_eq!(
            AlgorithmType::Hdc.device_affinity(),
            &[
                DeviceTarget::Lpu,
                DeviceTarget::Npu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu
            ]
        );
        // Columnar: GPU first, then TPU, LPU, CPU
        assert_eq!(
            AlgorithmType::Columnar.device_affinity(),
            &[
                DeviceTarget::Gpu,
                DeviceTarget::Tpu,
                DeviceTarget::Lpu,
                DeviceTarget::Cpu
            ]
        );
        // Holographic: NPU first, then LPU, GPU, CPU
        assert_eq!(
            AlgorithmType::Holographic.device_affinity(),
            &[
                DeviceTarget::Npu,
                DeviceTarget::Lpu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu
            ]
        );
        // TernaryHdc: LPU first, then NPU, GPU, CPU
        assert_eq!(
            AlgorithmType::TernaryHdc.device_affinity(),
            &[
                DeviceTarget::Lpu,
                DeviceTarget::Npu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu
            ]
        );
    }

    #[test]
    fn test_operation_type_display() {
        assert_eq!(OperationType::Search.to_string(), "search");
        assert_eq!(DeviceTarget::Gpu.to_string(), "gpu");
        assert_eq!(DeviceTarget::Npu.to_string(), "npu");
        assert_eq!(DeviceTarget::Tpu.to_string(), "tpu");
        assert_eq!(DeviceTarget::Lpu.to_string(), "lpu");
        assert_eq!(AlgorithmType::Hdc.to_string(), "hdc");
        assert_eq!(OperationType::TernaryMatvec.to_string(), "ternary_matvec");
        assert_eq!(AlgorithmType::TernaryHdc.to_string(), "ternary_hdc");
    }
}
