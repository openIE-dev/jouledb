//! Hardware-aware execution advisor.
//!
//! Analyzes hardware state and produces execution hints for the query engine.
//! Rule-based (no ML dependencies) -- inspired by Specter Pro's MacIntelligence
//! bottleneck detection with thresholds tuned for database workloads.

use crate::EnergyConfig;
use crate::monitor::{EnergySnapshot, ThermalState};

/// Execution hint produced by the hardware advisor.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionHint {
    /// All systems nominal, use default strategy.
    Normal,
    /// Prefer CPU execution (thermal pressure or GPU overloaded).
    PreferCpu,
    /// Prefer GPU execution (CPU overloaded, GPU idle).
    PreferGpu,
    /// Prefer NPU execution (CPU overloaded, NPU idle).
    PreferNpu,
    /// Prefer TPU execution (CPU overloaded, TPU idle).
    PreferTpu,
    /// Prefer LPU execution (CPU overloaded, LPU idle).
    PreferLpu,
    /// Reduce batch sizes to lower thermal/power load.
    ReduceBatchSize {
        /// Divisor: 2 = halve, 4 = quarter.
        suggested_factor: u8,
    },
    /// Throttle operations (approaching power envelope or critical thermal).
    Throttle,
    /// Reduce buffer pool size (memory pressure too high).
    ReduceBufferPool,
}

impl std::fmt::Display for ExecutionHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::PreferCpu => write!(f, "prefer_cpu"),
            Self::PreferGpu => write!(f, "prefer_gpu"),
            Self::PreferNpu => write!(f, "prefer_npu"),
            Self::PreferTpu => write!(f, "prefer_tpu"),
            Self::PreferLpu => write!(f, "prefer_lpu"),
            Self::ReduceBatchSize { suggested_factor } => {
                write!(f, "reduce_batch_size_{}", suggested_factor)
            }
            Self::Throttle => write!(f, "throttle"),
            Self::ReduceBufferPool => write!(f, "reduce_buffer_pool"),
        }
    }
}

/// Hardware-aware execution advisor.
///
/// Produces `ExecutionHint`s based on current hardware state.
/// Thread-safe and stateless -- can be shared across request handlers.
pub struct HardwareAdvisor {
    thermal_threshold: ThermalState,
    memory_pressure_threshold: f64,
    power_envelope_threshold: f64,
    default_tdp_watts: f64,
}

impl HardwareAdvisor {
    /// Create an advisor from energy configuration.
    pub fn new(config: &EnergyConfig) -> Self {
        Self {
            thermal_threshold: config.thermal_throttle_threshold,
            memory_pressure_threshold: config.memory_pressure_threshold,
            power_envelope_threshold: config.power_envelope_threshold,
            default_tdp_watts: config.default_tdp_watts,
        }
    }

    /// Analyze the current hardware snapshot and produce execution hints.
    pub fn advise(&self, snapshot: &EnergySnapshot) -> Vec<ExecutionHint> {
        let mut hints = Vec::new();

        // Rule 1: Thermal throttling
        if snapshot.thermal_state >= self.thermal_threshold {
            hints.push(ExecutionHint::PreferCpu);
            hints.push(ExecutionHint::ReduceBatchSize {
                suggested_factor: 2,
            });
        }
        if snapshot.thermal_state == ThermalState::Critical {
            hints.push(ExecutionHint::Throttle);
            hints.push(ExecutionHint::ReduceBatchSize {
                suggested_factor: 4,
            });
        }

        // Rule 2: Power envelope
        if self.default_tdp_watts > 0.0 {
            let power_ratio = snapshot.power_watts / self.default_tdp_watts;
            if power_ratio > self.power_envelope_threshold {
                hints.push(ExecutionHint::Throttle);
            }
        }

        // Rule 3: Memory pressure
        if snapshot.memory_pressure > self.memory_pressure_threshold {
            hints.push(ExecutionHint::ReduceBufferPool);
            hints.push(ExecutionHint::ReduceBatchSize {
                suggested_factor: 2,
            });
        }

        // Rule 4: GPU offloading opportunity
        if snapshot.gpu_available
            && snapshot.gpu_utilization < 0.5
            && snapshot.cpu_utilization > 0.8
        {
            hints.push(ExecutionHint::PreferGpu);
        }

        // Rule 5: GPU overloaded, fall back to CPU
        if snapshot.gpu_available && snapshot.gpu_utilization > 0.9 {
            hints.push(ExecutionHint::PreferCpu);
        }

        // Rule 6: NPU offloading opportunity
        if snapshot.npu_available
            && snapshot.npu_utilization < 0.5
            && snapshot.cpu_utilization > 0.8
        {
            hints.push(ExecutionHint::PreferNpu);
        }

        // Rule 7: TPU offloading opportunity
        if snapshot.tpu_available
            && snapshot.tpu_utilization < 0.5
            && snapshot.cpu_utilization > 0.8
        {
            hints.push(ExecutionHint::PreferTpu);
        }

        // Rule 8: LPU offloading opportunity
        if snapshot.lpu_available
            && snapshot.lpu_utilization < 0.5
            && snapshot.cpu_utilization > 0.8
        {
            hints.push(ExecutionHint::PreferLpu);
        }

        // Rule 9: LPU overloaded, fall back to CPU
        if snapshot.lpu_available && snapshot.lpu_utilization > 0.9 {
            hints.push(ExecutionHint::PreferCpu);
        }

        if hints.is_empty() {
            hints.push(ExecutionHint::Normal);
        }

        hints
    }

    /// Quick check: should we throttle based on this snapshot?
    pub fn should_throttle(&self, snapshot: &EnergySnapshot) -> bool {
        self.advise(snapshot)
            .iter()
            .any(|h| matches!(h, ExecutionHint::Throttle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_advisor() -> HardwareAdvisor {
        HardwareAdvisor::new(&EnergyConfig::default())
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

    #[test]
    fn test_nominal_returns_normal() {
        let advisor = default_advisor();
        let hints = advisor.advise(&nominal_snapshot());
        assert_eq!(hints, vec![ExecutionHint::Normal]);
    }

    #[test]
    fn test_thermal_serious_throttles() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Serious;

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferCpu));
        assert!(hints.contains(&ExecutionHint::ReduceBatchSize {
            suggested_factor: 2,
        }));
    }

    #[test]
    fn test_thermal_critical_hard_throttle() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Critical;

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::Throttle));
        assert!(hints.contains(&ExecutionHint::ReduceBatchSize {
            suggested_factor: 4,
        }));
    }

    #[test]
    fn test_high_power_throttles() {
        let advisor = default_advisor(); // TDP = 30W, threshold = 0.8 → 24W triggers
        let mut snap = nominal_snapshot();
        snap.power_watts = 28.0; // > 24W threshold

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::Throttle));
    }

    #[test]
    fn test_high_memory_pressure() {
        let advisor = default_advisor(); // threshold = 0.7
        let mut snap = nominal_snapshot();
        snap.memory_pressure = 0.85;

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::ReduceBufferPool));
    }

    #[test]
    fn test_gpu_offload_opportunity() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.cpu_utilization = 0.9; // CPU overloaded
        snap.gpu_utilization = 0.1; // GPU idle

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferGpu));
    }

    #[test]
    fn test_gpu_overloaded_prefers_cpu() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.gpu_utilization = 0.95;

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferCpu));
    }

    #[test]
    fn test_should_throttle_helper() {
        let advisor = default_advisor();
        assert!(!advisor.should_throttle(&nominal_snapshot()));

        let mut critical = nominal_snapshot();
        critical.thermal_state = ThermalState::Critical;
        assert!(advisor.should_throttle(&critical));
    }

    #[test]
    fn test_execution_hint_display() {
        assert_eq!(ExecutionHint::Normal.to_string(), "normal");
        assert_eq!(ExecutionHint::Throttle.to_string(), "throttle");
        assert_eq!(ExecutionHint::PreferNpu.to_string(), "prefer_npu");
        assert_eq!(ExecutionHint::PreferTpu.to_string(), "prefer_tpu");
        assert_eq!(ExecutionHint::PreferLpu.to_string(), "prefer_lpu");
        assert_eq!(
            ExecutionHint::ReduceBatchSize {
                suggested_factor: 4
            }
            .to_string(),
            "reduce_batch_size_4"
        );
    }

    #[test]
    fn test_npu_offload_opportunity() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.cpu_utilization = 0.9; // CPU overloaded
        snap.npu_available = true;
        snap.npu_utilization = 0.1; // NPU idle

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferNpu));
    }

    #[test]
    fn test_tpu_offload_opportunity() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.cpu_utilization = 0.9; // CPU overloaded
        snap.tpu_available = true;
        snap.tpu_utilization = 0.1; // TPU idle

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferTpu));
    }

    #[test]
    fn test_npu_overloaded_not_preferred() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.cpu_utilization = 0.9;
        snap.npu_available = true;
        snap.npu_utilization = 0.8; // NPU busy — should NOT prefer NPU

        let hints = advisor.advise(&snap);
        assert!(!hints.contains(&ExecutionHint::PreferNpu));
    }

    #[test]
    fn test_lpu_offload_opportunity() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.cpu_utilization = 0.9; // CPU overloaded
        snap.lpu_available = true;
        snap.lpu_utilization = 0.1; // LPU idle

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferLpu));
    }

    #[test]
    fn test_lpu_overloaded_prefers_cpu() {
        let advisor = default_advisor();
        let mut snap = nominal_snapshot();
        snap.lpu_available = true;
        snap.lpu_utilization = 0.95; // LPU overloaded

        let hints = advisor.advise(&snap);
        assert!(hints.contains(&ExecutionHint::PreferCpu));
    }
}
