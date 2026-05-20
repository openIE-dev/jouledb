//! Hardware energy monitor.
//!
//! Collects system energy metrics at configurable intervals.
//! Provides `EnergySnapshot` for real-time hardware state.

use crate::EnergyConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::platform::{self, PlatformEnergyProvider};
use std::sync::{Arc, RwLock};
use std::time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use sysinfo::System;

/// Thermal state levels (matches macOS machdep.xcpm.thermal_status).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ThermalState {
    Nominal = 0,
    Fair = 1,
    Serious = 2,
    Critical = 3,
}

impl std::fmt::Display for ThermalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThermalState::Nominal => write!(f, "Nominal"),
            ThermalState::Fair => write!(f, "Fair"),
            ThermalState::Serious => write!(f, "Serious"),
            ThermalState::Critical => write!(f, "Critical"),
        }
    }
}

/// Point-in-time hardware energy snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnergySnapshot {
    /// Timestamp of this snapshot (monotonic)
    #[serde(skip)]
    pub timestamp: Option<Instant>,
    /// Current power draw in watts (from battery V×A or estimation)
    pub power_watts: f64,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Current thermal state
    pub thermal_state: ThermalState,
    /// Memory pressure (0.0 - 1.0)
    pub memory_pressure: f64,
    /// Used memory in bytes
    pub memory_used_bytes: u64,
    /// Total memory in bytes
    pub memory_total_bytes: u64,
    /// Whether a GPU is available for compute
    pub gpu_available: bool,
    /// GPU utilization (0.0 - 1.0)
    pub gpu_utilization: f64,
    /// Whether an NPU is available for compute
    pub npu_available: bool,
    /// NPU utilization (0.0 - 1.0)
    pub npu_utilization: f64,
    /// Whether a TPU is available for compute
    pub tpu_available: bool,
    /// TPU utilization (0.0 - 1.0)
    pub tpu_utilization: f64,
    /// Whether an LPU (Language Processing Unit) is available for compute
    pub lpu_available: bool,
    /// LPU utilization (0.0 - 1.0)
    pub lpu_utilization: f64,
    /// Battery charge percentage (None if no battery)
    pub battery_percent: Option<f64>,
    /// Whether the battery is charging
    pub battery_charging: bool,
    /// Swap used in bytes
    pub swap_used_bytes: u64,
    /// Cumulative energy consumed since monitor start (joules)
    pub cumulative_joules: f64,
}

impl Default for EnergySnapshot {
    fn default() -> Self {
        Self {
            timestamp: None,
            power_watts: 0.0,
            cpu_utilization: 0.0,
            thermal_state: ThermalState::Nominal,
            memory_pressure: 0.0,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
            gpu_available: false,
            gpu_utilization: 0.0,
            npu_available: false,
            npu_utilization: 0.0,
            tpu_available: false,
            tpu_utilization: 0.0,
            lpu_available: false,
            lpu_utilization: 0.0,
            battery_percent: None,
            battery_charging: false,
            swap_used_bytes: 0,
            cumulative_joules: 0.0,
        }
    }
}

/// Hardware energy monitor.
///
/// Collects system metrics at regular intervals and makes them available
/// via a shared `EnergySnapshot`. Designed for low overhead (<1% CPU).
#[cfg(not(target_arch = "wasm32"))]
pub struct EnergyMonitor {
    config: EnergyConfig,
    system: System,
    platform: Box<dyn PlatformEnergyProvider>,
    latest: Arc<RwLock<EnergySnapshot>>,
    last_collect_time: Option<Instant>,
    cumulative_joules: f64,
}

#[cfg(not(target_arch = "wasm32"))]
impl EnergyMonitor {
    /// Create a new energy monitor with the given configuration.
    pub fn new(config: EnergyConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let provider = platform::create_provider(config.default_tdp_watts);
        Self {
            config,
            system,
            platform: provider,
            latest: Arc::new(RwLock::new(EnergySnapshot::default())),
            last_collect_time: None,
            cumulative_joules: 0.0,
        }
    }

    /// Collect a single snapshot. Call this periodically.
    pub fn collect(&mut self) -> EnergySnapshot {
        let now = Instant::now();
        self.system.refresh_cpu_usage();
        self.system.refresh_memory();

        let cpu_util = self.system.global_cpu_usage() as f64 / 100.0;

        // Feed CPU utilization to the platform provider (used by fallback)
        self.platform.set_cpu_utilization(cpu_util);

        let power_watts = self.platform.power_watts();
        let thermal_state = self.platform.thermal_state();
        let memory_pressure = self.platform.memory_pressure();
        let gpu_utilization = self.platform.gpu_utilization();
        let gpu_available = self.platform.gpu_available();
        let npu_utilization = self.platform.npu_utilization();
        let npu_available = self.platform.npu_available();
        let tpu_utilization = self.platform.tpu_utilization();
        let tpu_available = self.platform.tpu_available();
        let lpu_utilization = self.platform.lpu_utilization();
        let lpu_available = self.platform.lpu_available();

        // If platform didn't provide memory pressure, estimate from usage
        let effective_memory_pressure = if memory_pressure > 0.0 {
            memory_pressure
        } else if self.system.total_memory() > 0 {
            self.system.used_memory() as f64 / self.system.total_memory() as f64
        } else {
            0.0
        };

        // Estimate power if platform returned 0 (e.g., desktop without battery)
        let effective_power = if power_watts > 0.0 {
            power_watts
        } else {
            self.config.default_tdp_watts * cpu_util.max(0.01)
        };

        // Accumulate energy (joules = watts × seconds)
        if let Some(last_time) = self.last_collect_time {
            let elapsed_secs = now.duration_since(last_time).as_secs_f64();
            self.cumulative_joules += effective_power * elapsed_secs;
        }
        self.last_collect_time = Some(now);

        // Battery info via the battery crate (cross-platform)
        let (battery_percent, battery_charging) = self.read_battery_info();

        let snapshot = EnergySnapshot {
            timestamp: Some(now),
            power_watts: effective_power,
            cpu_utilization: cpu_util,
            thermal_state,
            memory_pressure: effective_memory_pressure,
            memory_used_bytes: self.system.used_memory(),
            memory_total_bytes: self.system.total_memory(),
            gpu_available,
            gpu_utilization,
            npu_available,
            npu_utilization,
            tpu_available,
            tpu_utilization,
            lpu_available,
            lpu_utilization,
            battery_percent,
            battery_charging,
            swap_used_bytes: self.system.used_swap(),
            cumulative_joules: self.cumulative_joules,
        };

        if let Ok(mut latest) = self.latest.write() {
            *latest = snapshot.clone();
        }

        snapshot
    }

    /// Get the latest cached snapshot (non-blocking read).
    pub fn snapshot(&self) -> EnergySnapshot {
        self.latest.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Get the shared snapshot handle for passing to other components.
    pub fn snapshot_handle(&self) -> Arc<RwLock<EnergySnapshot>> {
        self.latest.clone()
    }

    /// Start background collection in a dedicated thread.
    /// Returns the shared snapshot handle and a join handle.
    pub fn start_background(
        mut self,
    ) -> (Arc<RwLock<EnergySnapshot>>, std::thread::JoinHandle<()>) {
        let handle = self.latest.clone();
        let interval_ms = self.config.collection_interval_ms;

        let join = std::thread::Builder::new()
            .name("energy-monitor".to_string())
            .spawn(move || {
                loop {
                    self.collect();
                    std::thread::sleep(std::time::Duration::from_millis(interval_ms));
                }
            })
            .expect("failed to spawn energy monitor thread");

        (handle, join)
    }

    /// Read battery info via the cross-platform battery crate.
    fn read_battery_info(&self) -> (Option<f64>, bool) {
        if let Ok(manager) = battery::Manager::new() {
            if let Ok(mut batteries) = manager.batteries() {
                if let Some(Ok(batt)) = batteries.next() {
                    let percent = batt.state_of_charge().value as f64 * 100.0;
                    let charging = matches!(batt.state(), battery::State::Charging);
                    return (Some(percent), charging);
                }
            }
        }
        (None, false)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let config = EnergyConfig::default();
        let monitor = EnergyMonitor::new(config);
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.power_watts, 0.0); // Before first collect
    }

    #[test]
    fn test_collect_returns_valid_snapshot() {
        let config = EnergyConfig::default();
        let mut monitor = EnergyMonitor::new(config);
        let snapshot = monitor.collect();

        assert!(snapshot.cpu_utilization >= 0.0 && snapshot.cpu_utilization <= 1.0);
        assert!(snapshot.memory_total_bytes > 0);
        assert!(snapshot.memory_used_bytes <= snapshot.memory_total_bytes);
        assert!(snapshot.timestamp.is_some());
    }

    #[test]
    fn test_cumulative_joules_increase() {
        let config = EnergyConfig::default();
        let mut monitor = EnergyMonitor::new(config);

        let snap1 = monitor.collect();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let snap2 = monitor.collect();

        // Cumulative joules should increase (power > 0, time > 0)
        assert!(snap2.cumulative_joules >= snap1.cumulative_joules);
    }

    #[test]
    fn test_snapshot_handle_shared() {
        let config = EnergyConfig::default();
        let mut monitor = EnergyMonitor::new(config);
        let handle = monitor.snapshot_handle();

        monitor.collect();
        let snapshot = handle.read().unwrap();
        assert!(snapshot.memory_total_bytes > 0);
    }

    #[test]
    fn test_thermal_state_display() {
        assert_eq!(ThermalState::Nominal.to_string(), "Nominal");
        assert_eq!(ThermalState::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_thermal_state_ordering() {
        assert!(ThermalState::Critical > ThermalState::Serious);
        assert!(ThermalState::Serious > ThermalState::Fair);
        assert!(ThermalState::Fair > ThermalState::Nominal);
    }
}
