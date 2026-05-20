//! Fallback energy provider for unsupported platforms.
//!
//! Estimates power consumption as TDP × CPU utilization.
//! Returns Nominal thermal state and no GPU info.

use crate::monitor::ThermalState;
use crate::platform::PlatformEnergyProvider;

pub struct FallbackProvider {
    default_tdp_watts: f64,
    cpu_utilization: f64,
}

impl FallbackProvider {
    pub fn new(default_tdp_watts: f64) -> Self {
        Self {
            default_tdp_watts,
            cpu_utilization: 0.0,
        }
    }

    /// Called by EnergyMonitor to feed CPU utilization from sysinfo.
    pub fn set_cpu_utilization(&mut self, util: f64) {
        self.cpu_utilization = util;
    }
}

impl PlatformEnergyProvider for FallbackProvider {
    fn thermal_state(&mut self) -> ThermalState {
        ThermalState::Nominal
    }

    fn memory_pressure(&mut self) -> f64 {
        0.0
    }

    fn power_watts(&mut self) -> f64 {
        self.default_tdp_watts * self.cpu_utilization.max(0.01)
    }

    fn gpu_utilization(&mut self) -> f64 {
        0.0
    }

    fn gpu_available(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_estimates_power() {
        let mut provider = FallbackProvider::new(30.0);
        provider.set_cpu_utilization(0.5);
        let watts = provider.power_watts();
        assert!((watts - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_fallback_minimum_power() {
        let mut provider = FallbackProvider::new(30.0);
        provider.set_cpu_utilization(0.0);
        let watts = provider.power_watts();
        // Should use min 0.01 utilization
        assert!((watts - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_fallback_thermal_always_nominal() {
        let mut provider = FallbackProvider::new(30.0);
        assert_eq!(provider.thermal_state(), ThermalState::Nominal);
    }

    #[test]
    fn test_fallback_no_gpu() {
        let mut provider = FallbackProvider::new(30.0);
        assert!(!provider.gpu_available());
        assert_eq!(provider.gpu_utilization(), 0.0);
    }
}
