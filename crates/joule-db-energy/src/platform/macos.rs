//! macOS IOKit energy provider.
//!
//! Reads real hardware data from Apple's System Management Controller:
//! - Battery voltage × amperage for real power draw (watts)
//! - AGXAccelerator PerformanceStatistics for GPU utilization
//! - sysctl for thermal and memory pressure
//!
//! Ported from Specter Pro's TelemetryProvider.

use crate::monitor::ThermalState;
use crate::platform::PlatformEnergyProvider;
use core_foundation::base::{TCFType, TCFTypeRef};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use io_kit_sys::{
    IORegistryEntryCreateCFProperty, IOServiceGetMatchingService, IOServiceMatching,
    kIOMasterPortDefault,
};
use libc::{size_t, sysctlbyname};

pub struct MacOsProvider {
    _private: (),
}

impl MacOsProvider {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Read a sysctl integer value by name.
    fn sysctl_i32(&self, name: &str) -> Option<i32> {
        let mut val: i32 = 0;
        let mut size = std::mem::size_of::<i32>() as size_t;
        let cname = std::ffi::CString::new(name).ok()?;
        unsafe {
            if sysctlbyname(
                cname.as_ptr(),
                &mut val as *mut _ as *mut _,
                &mut size,
                std::ptr::null_mut(),
                0,
            ) == 0
            {
                Some(val)
            } else {
                None
            }
        }
    }

    /// Read battery voltage (V) and amperage (A) from IOKit AppleSmartBattery.
    /// Returns (voltage_volts, amperage_amps).
    fn battery_power_from_iokit(&self) -> Option<f64> {
        unsafe {
            let matching = IOServiceMatching("AppleSmartBattery\0".as_ptr() as *const i8);
            let service = IOServiceGetMatchingService(kIOMasterPortDefault, matching);
            if service == 0 {
                return None;
            }

            let mut voltage = 0.0_f64;
            let mut amperage = 0.0_f64;

            // Read Voltage (millivolts)
            let voltage_key = CFString::new("Voltage");
            let v_props = IORegistryEntryCreateCFProperty(
                service,
                voltage_key.as_concrete_TypeRef(),
                std::ptr::null(),
                0,
            );
            if !v_props.is_null() {
                let num: CFNumber = TCFType::wrap_under_get_rule(v_props as *const _);
                voltage = num.to_f64().unwrap_or(0.0) / 1000.0; // mV → V
            }

            // Read Amperage (milliamps, signed: positive = discharging)
            let amperage_key = CFString::new("Amperage");
            let a_props = IORegistryEntryCreateCFProperty(
                service,
                amperage_key.as_concrete_TypeRef(),
                std::ptr::null(),
                0,
            );
            if !a_props.is_null() {
                let num: CFNumber = TCFType::wrap_under_get_rule(a_props as *const _);
                amperage = num.to_f64().unwrap_or(0.0) / 1000.0; // mA → A
            }

            let watts = (voltage * amperage).abs();
            if watts > 0.0 { Some(watts) } else { None }
        }
    }

    /// Read GPU utilization from AGXAccelerator PerformanceStatistics.
    fn agx_gpu_utilization(&self) -> f64 {
        unsafe {
            let matching = IOServiceMatching("AGXAccelerator\0".as_ptr() as *const i8);
            let service = IOServiceGetMatchingService(kIOMasterPortDefault, matching);
            if service == 0 {
                return 0.0;
            }

            let key = CFString::new("PerformanceStatistics");
            let props = IORegistryEntryCreateCFProperty(
                service,
                key.as_concrete_TypeRef(),
                std::ptr::null(),
                0,
            );
            if props.is_null() {
                return 0.0;
            }

            let dict: CFDictionary = TCFType::wrap_under_get_rule(props as *const _);
            let util_key = CFString::new("Device Utilization %");
            if let Some(val) = dict.find(util_key.as_CFTypeRef()) {
                let num: CFNumber = TCFType::wrap_under_get_rule(val.as_void_ptr() as *const _);
                if let Some(n) = num.to_f32() {
                    return (n as f64) / 100.0; // Normalize to 0.0 - 1.0
                }
            }

            0.0
        }
    }
}

impl PlatformEnergyProvider for MacOsProvider {
    fn thermal_state(&mut self) -> ThermalState {
        match self.sysctl_i32("machdep.xcpm.thermal_status") {
            Some(0) => ThermalState::Nominal,
            Some(1) => ThermalState::Fair,
            Some(2) => ThermalState::Serious,
            Some(3) => ThermalState::Critical,
            _ => ThermalState::Nominal,
        }
    }

    fn memory_pressure(&mut self) -> f64 {
        // kern.memo_status_level returns 0-4 (normal=1, warn=2, critical=4)
        // Normalize: treat as fraction of 4
        match self.sysctl_i32("kern.memo_status_level") {
            Some(level) => (level as f64 / 4.0).clamp(0.0, 1.0),
            None => 0.0,
        }
    }

    fn power_watts(&mut self) -> f64 {
        self.battery_power_from_iokit().unwrap_or(0.0)
    }

    fn gpu_utilization(&mut self) -> f64 {
        self.agx_gpu_utilization()
    }

    fn gpu_available(&self) -> bool {
        // Apple Silicon always has integrated GPU
        true
    }

    fn npu_available(&self) -> bool {
        // All Apple Silicon Macs have the Apple Neural Engine
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macos_provider_creation() {
        let provider = MacOsProvider::new();
        assert!(provider.gpu_available());
    }

    #[test]
    fn test_thermal_state_returns_valid() {
        let mut provider = MacOsProvider::new();
        let state = provider.thermal_state();
        // Should be one of the valid variants (most likely Nominal in test)
        assert!(matches!(
            state,
            ThermalState::Nominal
                | ThermalState::Fair
                | ThermalState::Serious
                | ThermalState::Critical
        ));
    }

    #[test]
    fn test_memory_pressure_in_range() {
        let mut provider = MacOsProvider::new();
        let pressure = provider.memory_pressure();
        assert!(pressure >= 0.0 && pressure <= 1.0);
    }

    #[test]
    fn test_power_watts_non_negative() {
        let mut provider = MacOsProvider::new();
        let watts = provider.power_watts();
        assert!(watts >= 0.0);
    }

    #[test]
    fn test_gpu_utilization_in_range() {
        let mut provider = MacOsProvider::new();
        let util = provider.gpu_utilization();
        assert!(util >= 0.0 && util <= 1.0);
    }
}
