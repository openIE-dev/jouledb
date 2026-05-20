//! Linux sysfs energy provider.
//!
//! Reads hardware data from Linux pseudo-filesystems:
//! - /sys/class/power_supply/*/power_now for power draw
//! - /sys/class/thermal/thermal_zone*/temp for thermal state
//! - /proc/pressure/memory for memory pressure

use crate::monitor::ThermalState;
use crate::platform::PlatformEnergyProvider;
use std::fs;
use std::path::Path;

/// Check if a binary exists in PATH (lightweight which(1) equivalent).
fn which_exists(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(name).is_file()))
        .unwrap_or(false)
}

pub struct LinuxProvider {
    /// Cached: Cloud TPU detected (GCE TPU VM or libtpu device node)
    cloud_tpu_detected: bool,
    /// Cached: LPU runtime detected (Groq inference chip)
    lpu_detected: bool,
}

impl LinuxProvider {
    pub fn new() -> Self {
        Self {
            cloud_tpu_detected: Self::detect_cloud_tpu(),
            lpu_detected: Self::detect_lpu(),
        }
    }

    /// Detect Google Cloud TPU (v4/v5e/v5p/v6e).
    ///
    /// Cloud TPU VMs always set the `TPU_NAME` environment variable.
    /// Additionally, libtpu exposes `/sys/class/accel/accel0`.
    fn detect_cloud_tpu() -> bool {
        // Method 1: GCE TPU VM sets TPU_NAME
        if std::env::var("TPU_NAME").is_ok() {
            return true;
        }
        // Method 2: libtpu device node (distinct from /dev/apex_* which is Coral)
        Path::new("/sys/class/accel/accel0").exists()
    }

    /// Detect Groq LPU inference accelerator.
    ///
    /// Groq's runtime tools (`groq-runtime`, `groqit`) are the
    /// canonical indicator of an LPU-equipped host.
    fn detect_lpu() -> bool {
        // Check if groq-runtime or groqit is in PATH
        which_exists("groq-runtime") || which_exists("groqit")
    }

    /// Read power_now from the first battery power supply (microwatts → watts).
    fn read_power_supply_watts(&self) -> f64 {
        let power_supply = Path::new("/sys/class/power_supply");
        if !power_supply.exists() {
            return 0.0;
        }

        if let Ok(entries) = fs::read_dir(power_supply) {
            for entry in entries.flatten() {
                let path = entry.path();
                let power_now = path.join("power_now");
                if power_now.exists() {
                    if let Ok(content) = fs::read_to_string(&power_now) {
                        if let Ok(microwatts) = content.trim().parse::<f64>() {
                            return microwatts / 1_000_000.0;
                        }
                    }
                }
                // Some systems use voltage_now + current_now instead
                let voltage = path.join("voltage_now");
                let current = path.join("current_now");
                if voltage.exists() && current.exists() {
                    if let (Ok(v_str), Ok(c_str)) =
                        (fs::read_to_string(&voltage), fs::read_to_string(&current))
                    {
                        if let (Ok(v_uv), Ok(c_ua)) =
                            (v_str.trim().parse::<f64>(), c_str.trim().parse::<f64>())
                        {
                            let volts = v_uv / 1_000_000.0;
                            let amps = c_ua / 1_000_000.0;
                            return (volts * amps).abs();
                        }
                    }
                }
            }
        }
        0.0
    }

    /// Read thermal zone temperatures, return max temp in Celsius.
    fn read_max_thermal_temp(&self) -> f64 {
        let thermal_base = Path::new("/sys/class/thermal");
        if !thermal_base.exists() {
            return 0.0;
        }

        let mut max_temp = 0.0_f64;
        if let Ok(entries) = fs::read_dir(thermal_base) {
            for entry in entries.flatten() {
                let temp_path = entry.path().join("temp");
                if temp_path.exists() {
                    if let Ok(content) = fs::read_to_string(&temp_path) {
                        if let Ok(millidegrees) = content.trim().parse::<f64>() {
                            max_temp = max_temp.max(millidegrees / 1000.0);
                        }
                    }
                }
            }
        }
        max_temp
    }

    /// Read memory pressure from /proc/pressure/memory (PSI).
    /// Returns the avg10 "some" value as a fraction (0.0 - 1.0).
    fn read_memory_pressure(&self) -> f64 {
        let psi_path = Path::new("/proc/pressure/memory");
        if !psi_path.exists() {
            return 0.0;
        }

        if let Ok(content) = fs::read_to_string(psi_path) {
            // Format: "some avg10=0.00 avg60=0.00 avg300=0.00 total=0"
            for line in content.lines() {
                if line.starts_with("some ") {
                    if let Some(avg10_part) =
                        line.split_whitespace().find(|s| s.starts_with("avg10="))
                    {
                        if let Some(val_str) = avg10_part.strip_prefix("avg10=") {
                            if let Ok(percent) = val_str.parse::<f64>() {
                                return (percent / 100.0).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }
        }
        0.0
    }
}

impl PlatformEnergyProvider for LinuxProvider {
    fn thermal_state(&mut self) -> ThermalState {
        let temp = self.read_max_thermal_temp();
        if temp >= 95.0 {
            ThermalState::Critical
        } else if temp >= 85.0 {
            ThermalState::Serious
        } else if temp >= 75.0 {
            ThermalState::Fair
        } else {
            ThermalState::Nominal
        }
    }

    fn memory_pressure(&mut self) -> f64 {
        self.read_memory_pressure()
    }

    fn power_watts(&mut self) -> f64 {
        self.read_power_supply_watts()
    }

    fn gpu_utilization(&mut self) -> f64 {
        // Could read from /sys/class/drm/card*/device/gpu_busy_percent (AMD)
        // or nvidia-smi. For now, return 0 (unknown).
        0.0
    }

    fn gpu_available(&self) -> bool {
        Path::new("/sys/class/drm/card0").exists()
    }

    fn npu_available(&self) -> bool {
        // Rockchip RKNN NPU
        Path::new("/sys/class/misc/rknn_npu").exists()
        // Hailo-8 AI accelerator
        || Path::new("/dev/hailo0").exists()
        // Generic accelerator device node (e.g., Samsung/Qualcomm NPU)
        || Path::new("/dev/accel0").exists()
    }

    fn tpu_available(&self) -> bool {
        // Google Coral Edge TPU (PCI or USB)
        let coral = (0..4).any(|i| Path::new(&format!("/dev/apex_{}", i)).exists());
        // Cloud TPU (v4/v5e/v5p/v6e) via TPU_NAME env or libtpu device node
        coral || self.cloud_tpu_detected
    }

    fn lpu_available(&self) -> bool {
        self.lpu_detected
    }

    fn lpu_utilization(&mut self) -> f64 {
        // Groq does not expose sysfs utilization metrics
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linux_provider_creation() {
        let _provider = LinuxProvider::new();
    }

    #[test]
    fn test_thermal_state_from_temp() {
        let mut provider = LinuxProvider::new();
        let state = provider.thermal_state();
        assert!(matches!(
            state,
            ThermalState::Nominal
                | ThermalState::Fair
                | ThermalState::Serious
                | ThermalState::Critical
        ));
    }
}
