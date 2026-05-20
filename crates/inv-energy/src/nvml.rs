//! NVIDIA GPU energy meter using NVML (NVIDIA Management Library).
//!
//! When the `nvml` feature is enabled, this module uses the `nvml-wrapper`
//! crate to read real GPU power consumption via NVIDIA's NVML API.
//! Without the feature, `NvmlMeter` reports as unavailable and the
//! composite meter falls back to CPU-based estimation.

use crate::meter::{EnergyMeter, EnergyMeterError};
use inv_core::energy::{EnergyReading, EnergySource};

#[cfg(feature = "nvml")]
use inv_core::energy::{Joules, Watts};
#[cfg(feature = "nvml")]
use std::time::{SystemTime, UNIX_EPOCH};

/// NVIDIA GPU energy meter using NVML (NVIDIA Management Library).
///
/// When the `nvml` feature is enabled and NVIDIA drivers are present,
/// reads real-time GPU power consumption. Falls back gracefully on
/// systems without NVIDIA GPUs.
pub struct NvmlMeter {
    available: bool,
    #[cfg(feature = "nvml")]
    nvml: Option<nvml_wrapper::Nvml>,
    #[cfg(feature = "nvml")]
    device_count: u32,
    /// Cumulative energy in microjoules.
    #[cfg(feature = "nvml")]
    cumulative_uj: std::sync::atomic::AtomicU64,
    /// Last reading timestamp in milliseconds.
    #[cfg(feature = "nvml")]
    last_timestamp_ms: std::sync::atomic::AtomicU64,
}

impl Default for NvmlMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl NvmlMeter {
    /// Create a new NVML meter.
    ///
    /// Attempts to initialize NVML and enumerate GPUs. If NVML is not
    /// available (no NVIDIA drivers, or `nvml` feature not enabled),
    /// the meter reports as unavailable.
    pub fn new() -> Self {
        #[cfg(feature = "nvml")]
        {
            match nvml_wrapper::Nvml::init() {
                Ok(nvml) => {
                    let device_count = nvml.device_count().unwrap_or(0);
                    if device_count > 0 {
                        tracing::info!(gpu_count = device_count, "NVML initialized");
                        return Self {
                            available: true,
                            nvml: Some(nvml),
                            device_count,
                            cumulative_uj: std::sync::atomic::AtomicU64::new(0),
                            last_timestamp_ms: std::sync::atomic::AtomicU64::new(0),
                        };
                    }
                    tracing::debug!("NVML initialized but no GPUs found");
                    Self {
                        available: false,
                        nvml: Some(nvml),
                        device_count: 0,
                        cumulative_uj: std::sync::atomic::AtomicU64::new(0),
                        last_timestamp_ms: std::sync::atomic::AtomicU64::new(0),
                    }
                }
                Err(e) => {
                    tracing::debug!("NVML init failed: {e}");
                    Self {
                        available: false,
                        nvml: None,
                        device_count: 0,
                        cumulative_uj: std::sync::atomic::AtomicU64::new(0),
                        last_timestamp_ms: std::sync::atomic::AtomicU64::new(0),
                    }
                }
            }
        }
        #[cfg(not(feature = "nvml"))]
        {
            Self { available: false }
        }
    }
}

impl EnergyMeter for NvmlMeter {
    fn read(&self) -> Result<EnergyReading, EnergyMeterError> {
        #[cfg(feature = "nvml")]
        {
            let nvml = self
                .nvml
                .as_ref()
                .ok_or_else(|| EnergyMeterError::NotSupported("NVML not initialized".into()))?;

            let mut total_milliwatts: u64 = 0;
            for i in 0..self.device_count {
                let device = nvml
                    .device_by_index(i)
                    .map_err(|e| EnergyMeterError::ReadFailed(format!("GPU {i}: {e}")))?;
                let power_mw = device
                    .power_usage()
                    .map_err(|e| EnergyMeterError::ReadFailed(format!("GPU {i} power: {e}")))?;
                total_milliwatts += power_mw as u64;
            }

            let watts = total_milliwatts as f64 / 1000.0;
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Accumulate energy
            let last_ms = self
                .last_timestamp_ms
                .swap(now_ms, std::sync::atomic::Ordering::Relaxed);
            if last_ms > 0 {
                let elapsed_secs = (now_ms - last_ms) as f64 / 1000.0;
                let delta_uj = (watts * elapsed_secs * 1_000_000.0) as u64;
                self.cumulative_uj
                    .fetch_add(delta_uj, std::sync::atomic::Ordering::Relaxed);
            }

            let total_joules = self
                .cumulative_uj
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1_000_000.0;

            Ok(EnergyReading::new(
                Joules::new(total_joules),
                Watts::new(watts),
                now_ms,
            ))
        }
        #[cfg(not(feature = "nvml"))]
        {
            Err(EnergyMeterError::NotSupported(
                "NVML support requires the `nvml` feature".into(),
            ))
        }
    }

    fn name(&self) -> &str {
        "nvml"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn energy_source(&self) -> EnergySource {
        EnergySource::WallPower
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvml_meter_without_gpu() {
        let meter = NvmlMeter::new();
        // On CI / macOS without NVIDIA GPUs, meter is unavailable
        assert_eq!(meter.name(), "nvml");
        assert_eq!(meter.energy_source(), EnergySource::WallPower);
    }

    #[test]
    fn nvml_meter_name() {
        let meter = NvmlMeter::new();
        assert_eq!(meter.name(), "nvml");
    }

    #[test]
    fn nvml_energy_source() {
        let meter = NvmlMeter::new();
        assert_eq!(meter.energy_source(), EnergySource::WallPower);
    }
}
