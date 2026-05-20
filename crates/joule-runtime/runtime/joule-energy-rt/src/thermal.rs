//! Thermal monitoring and state detection

use crate::error::Result;

#[cfg(not(target_os = "macos"))]
use crate::error::Error;
#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::PathBuf;

/// Thermal state of the CPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Cool,     // < 50C
    Nominal,  // 50-70C
    Elevated, // 70-85C
    Hot,      // 85-95C
    Critical, // > 95C
}

impl ThermalState {
    /// Get thermal state from temperature
    pub fn from_temperature(celsius: f64) -> Self {
        match celsius {
            t if t < 50.0 => ThermalState::Cool,
            t if t < 70.0 => ThermalState::Nominal,
            t if t < 85.0 => ThermalState::Elevated,
            t if t < 95.0 => ThermalState::Hot,
            _ => ThermalState::Critical,
        }
    }

    /// Get energy multiplier for this thermal state
    ///
    /// This represents how much more energy instructions consume
    /// when the CPU is hot vs. nominal temperature.
    pub fn energy_multiplier(self) -> f64 {
        match self {
            ThermalState::Cool => 0.9,     // Slightly less energy when cool
            ThermalState::Nominal => 1.0,  // Baseline
            ThermalState::Elevated => 1.3, // 30% more energy
            ThermalState::Hot => 1.6,      // 60% more energy
            ThermalState::Critical => 2.0, // 2x more energy (thermal throttling)
        }
    }

    /// Check if SIMD should be avoided in this thermal state
    pub fn should_avoid_simd(self) -> bool {
        matches!(self, ThermalState::Hot | ThermalState::Critical)
    }

    /// Check if loop unrolling should be conservative
    pub fn should_limit_unrolling(self) -> bool {
        matches!(
            self,
            ThermalState::Elevated | ThermalState::Hot | ThermalState::Critical
        )
    }
}

/// Thermal monitor backend
#[cfg(any(target_os = "linux", target_os = "macos"))]
enum ThermalBackend {
    /// Linux sysfs thermal zone
    #[cfg(target_os = "linux")]
    Linux { sensor_path: PathBuf },
    /// macOS IOKit thermal reader
    #[cfg(target_os = "macos")]
    MacOS {
        thermal_reader: crate::macos::MacOSThermalReader,
    },
}

/// Thermal monitor
pub struct ThermalMonitor {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    backend: ThermalBackend,
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    _private: (),
}

impl ThermalMonitor {
    /// Create a new thermal monitor
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            // Try different thermal zone paths
            for i in 0..10 {
                let path = PathBuf::from(format!("/sys/class/thermal/thermal_zone{}/temp", i));
                if path.exists() {
                    return Ok(Self {
                        backend: ThermalBackend::Linux { sensor_path: path },
                    });
                }
            }

            Err(Error::Unsupported(
                "No thermal sensors found in /sys/class/thermal/".to_string(),
            ))
        }

        #[cfg(target_os = "macos")]
        {
            let thermal_reader = crate::macos::MacOSThermalReader::new()?;
            Ok(Self {
                backend: ThermalBackend::MacOS { thermal_reader },
            })
        }

        #[cfg(target_os = "windows")]
        {
            // Windows uses WMI for thermal monitoring
            Err(Error::Unsupported(
                "Windows thermal monitoring not yet implemented. Use WMI.".to_string(),
            ))
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Err(Error::Unsupported("Platform not supported".to_string()))
        }
    }

    /// Read current temperature in Celsius
    pub fn read_temperature(&self) -> Result<f64> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            match &self.backend {
                #[cfg(target_os = "linux")]
                ThermalBackend::Linux { sensor_path } => {
                    let content = fs::read_to_string(sensor_path)?;
                    let millidegrees = content
                        .trim()
                        .parse::<f64>()
                        .map_err(|e| Error::Parse(format!("Failed to parse temperature: {}", e)))?;

                    Ok(millidegrees / 1000.0)
                }
                #[cfg(target_os = "macos")]
                ThermalBackend::MacOS { thermal_reader } => thermal_reader.read_cpu_temperature(),
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Err(Error::Unsupported("Platform not supported".to_string()))
        }
    }

    /// Get current thermal state
    pub fn thermal_state(&self) -> Result<ThermalState> {
        let temp = self.read_temperature()?;
        Ok(ThermalState::from_temperature(temp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_state_from_temp() {
        assert_eq!(ThermalState::from_temperature(40.0), ThermalState::Cool);
        assert_eq!(ThermalState::from_temperature(60.0), ThermalState::Nominal);
        assert_eq!(ThermalState::from_temperature(75.0), ThermalState::Elevated);
        assert_eq!(ThermalState::from_temperature(90.0), ThermalState::Hot);
        assert_eq!(
            ThermalState::from_temperature(100.0),
            ThermalState::Critical
        );
    }

    #[test]
    fn test_energy_multiplier() {
        assert_eq!(ThermalState::Cool.energy_multiplier(), 0.9);
        assert_eq!(ThermalState::Nominal.energy_multiplier(), 1.0);
        assert_eq!(ThermalState::Elevated.energy_multiplier(), 1.3);
        assert_eq!(ThermalState::Hot.energy_multiplier(), 1.6);
        assert_eq!(ThermalState::Critical.energy_multiplier(), 2.0);
    }

    #[test]
    #[ignore] // Requires Linux or macOS with thermal sensors
    fn test_read_temperature() {
        let monitor = ThermalMonitor::new().unwrap();
        let temp = monitor.read_temperature().unwrap();
        println!("CPU temperature: {:.1}C", temp);

        // Should be between 20-120°C (Apple Silicon can run warm)
        assert!(temp > 20.0 && temp < 120.0);
    }

    #[test]
    #[ignore] // Requires Linux or macOS with thermal sensors
    fn test_thermal_state() {
        let monitor = ThermalMonitor::new().unwrap();
        let state = monitor.thermal_state().unwrap();
        println!("Thermal state: {:?}", state);
    }
}
