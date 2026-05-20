//! Platform-specific energy measurement providers.
//!
//! Each platform implements `PlatformEnergyProvider` with the best available
//! hardware APIs. Falls back to estimation on unsupported platforms.

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

pub mod fallback;

use crate::monitor::ThermalState;

/// Platform-specific energy and hardware measurement provider.
pub trait PlatformEnergyProvider: Send {
    /// Current thermal state of the system.
    fn thermal_state(&mut self) -> ThermalState;

    /// Memory pressure as a fraction (0.0 - 1.0).
    fn memory_pressure(&mut self) -> f64;

    /// Current power draw in watts. Returns 0.0 if unavailable.
    fn power_watts(&mut self) -> f64;

    /// GPU utilization as a fraction (0.0 - 1.0).
    fn gpu_utilization(&mut self) -> f64;

    /// Whether a GPU is available for compute.
    fn gpu_available(&self) -> bool;

    /// NPU utilization as a fraction (0.0 - 1.0).
    fn npu_utilization(&mut self) -> f64 {
        0.0
    }

    /// Whether an NPU is available for compute.
    fn npu_available(&self) -> bool {
        false
    }

    /// TPU utilization as a fraction (0.0 - 1.0).
    fn tpu_utilization(&mut self) -> f64 {
        0.0
    }

    /// Whether a TPU is available for compute.
    fn tpu_available(&self) -> bool {
        false
    }

    /// LPU (Language Processing Unit) utilization as a fraction (0.0 - 1.0).
    fn lpu_utilization(&mut self) -> f64 {
        0.0
    }

    /// Whether an LPU is available for compute.
    fn lpu_available(&self) -> bool {
        false
    }

    /// Feed CPU utilization from sysinfo (used by fallback provider).
    fn set_cpu_utilization(&mut self, _util: f64) {}
}

/// Create the best available platform provider for this system.
#[cfg(not(target_arch = "wasm32"))]
pub fn create_provider(default_tdp_watts: f64) -> Box<dyn PlatformEnergyProvider> {
    #[cfg(target_os = "macos")]
    {
        Box::new(macos::MacOsProvider::new())
    }

    #[cfg(target_os = "linux")]
    {
        Box::new(linux::LinuxProvider::new())
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Box::new(fallback::FallbackProvider::new(default_tdp_watts))
    }
}
