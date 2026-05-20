// Energy measurement code has legitimate patterns that trigger these lints:
// - float_cmp: threshold/zero comparisons in calibration
// - ptr_as_ptr, ptr_cast_constness, borrow_as_ptr: IOKit FFI pointer casts
// - manual_clamp: explicit clamping for clarity in calibration
// - duplicated_attributes: inner #![cfg] on platform-gated modules
#![allow(
    clippy::float_cmp,
    clippy::ptr_as_ptr,
    clippy::ptr_cast_constness,
    clippy::manual_clamp,
    clippy::borrow_as_ptr,
    clippy::duplicated_attributes,
    clippy::ignore_without_reason
)]
//! Joule Energy Runtime
//!
//! This library provides cross-platform energy measurement capabilities.
//!
//! # Platform Support
//!
//! - **Linux**: Uses RAPL (Running Average Power Limit) for Intel/AMD CPUs
//! - **macOS**: Uses IOReport for Apple Silicon (M1/M2/M3/M4/M5) energy monitoring
//! - **Windows**: Uses EMI (Energy Metering Interface) for Windows 10+ with compatible hardware
//!
//! # Example
//!
//! ```no_run
//! use joule_energy_rt::EnergyMonitor;
//!
//! let monitor = EnergyMonitor::start().expect("Failed to start energy monitoring");
//!
//! // ... perform computation ...
//!
//! let metrics = monitor.stop().expect("Failed to stop monitoring");
//! println!("Energy consumed: {:.6} J", metrics.energy_joules);
//! println!("Average power: {:.2} W", metrics.power_watts);
//! ```

pub mod domains;
pub mod error;
pub mod kinematics;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod platform;
pub mod rapl;
pub mod thermal;
#[cfg(target_os = "windows")]
pub mod windows;

pub use domains::{
    BandwidthStats, CpuDomainReader, DomainCorrelations, DomainReader, DomainSample,
    DomainTelemetry, DramDomainReader, EstimatedDomainReader, FrequencyStats, GroqDomainReader,
    GpuDomainReader, HardwareDomain, MultiDomainMonitor, MultiDomainTelemetry,
    NeuronDomainReader, SystemAggregate, TpuDomainReader, UtilizationStats,
};
// Accelerator readers gated behind the "accelerators" feature
#[cfg(feature = "accelerators")]
pub use domains::NvmlDomainReader;
#[cfg(feature = "accelerators")]
pub use domains::LevelZeroDomainReader;
#[cfg(all(target_os = "linux", feature = "accelerators"))]
pub use domains::RocmDomainReader;
#[cfg(all(target_os = "linux", feature = "accelerators"))]
pub use domains::HlmlDomainReader;
pub use error::{Error, Result};
pub use kinematics::{
    DerivativeChain, DerivativeComputer, EfficiencyMetrics, EnergyDerivatives, KinematicConfig,
    KinematicMonitor, KinematicTelemetry, Sample, TelemetryStats, ThermalDerivatives,
    ThermodynamicCoupling,
};
pub use platform::{EnergyReader, create_reader};
pub use thermal::{ThermalMonitor, ThermalState};

use std::time::{Duration, Instant};

/// Energy measurement metrics
#[derive(Debug, Clone)]
pub struct EnergyMetrics {
    pub energy_joules: f64,
    pub power_watts: f64,
    pub duration: Duration,
    pub temp_delta: Option<f64>,
}

/// Energy monitor
pub struct EnergyMonitor {
    reader: Box<dyn EnergyReader>,
    start_energy: f64,
    start_time: Instant,
    start_temp: Option<f64>,
}

impl EnergyMonitor {
    pub fn start() -> Result<Self> {
        let reader = create_reader()?;
        let start_energy = reader.read_energy()?;
        let start_time = Instant::now();
        let start_temp = reader.read_temperature().ok();

        Ok(Self {
            reader,
            start_energy,
            start_time,
            start_temp,
        })
    }

    /// Take an intermediate energy reading without stopping the monitor.
    ///
    /// Returns the cumulative energy consumed (in joules) since `start()` was
    /// called, along with the elapsed duration. This allows sampling energy
    /// consumption at arbitrary intervals during a profiled run.
    pub fn sample(&self) -> Result<EnergyMetrics> {
        let current_energy = self.reader.read_energy()?;
        let duration = self.start_time.elapsed();
        let energy_joules = current_energy - self.start_energy;
        let secs = duration.as_secs_f64();
        let power_watts = if secs > 0.0 {
            energy_joules / secs
        } else {
            0.0
        };

        Ok(EnergyMetrics {
            energy_joules,
            power_watts,
            duration,
            temp_delta: None,
        })
    }

    pub fn stop(&self) -> Result<EnergyMetrics> {
        let end_energy = self.reader.read_energy()?;
        let duration = self.start_time.elapsed();
        let energy_joules = end_energy - self.start_energy;
        let power_watts = energy_joules / duration.as_secs_f64();

        let temp_delta = match (self.start_temp, self.reader.read_temperature().ok()) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        };

        Ok(EnergyMetrics {
            energy_joules,
            power_watts,
            duration,
            temp_delta,
        })
    }
}
