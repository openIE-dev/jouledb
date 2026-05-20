use crate::meter::{EnergyMeter, EnergyMeterError};
#[cfg(not(target_os = "macos"))]
use inv_core::energy::EnergyReading;
#[cfg(target_os = "macos")]
use inv_core::energy::{EnergyReading, Joules, Watts};
use inv_core::energy::{EnergySource, ThermalState};

/// Apple Silicon energy meter using load-average-based estimation.
/// Production version would use IOReport FFI for direct PMC access.
pub struct AppleSiliconMeter {
    available: bool,
    #[cfg(target_os = "macos")]
    cumulative_joules: std::sync::atomic::AtomicU64,
    #[cfg(target_os = "macos")]
    last_timestamp_ms: std::sync::atomic::AtomicU64,
}

impl Default for AppleSiliconMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AppleSiliconMeter {
    pub fn new() -> Self {
        let available = cfg!(target_os = "macos") && cfg!(target_arch = "aarch64");
        Self {
            available,
            #[cfg(target_os = "macos")]
            cumulative_joules: std::sync::atomic::AtomicU64::new(0),
            #[cfg(target_os = "macos")]
            last_timestamp_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    #[cfg(target_os = "macos")]
    fn read_power(&self) -> Result<f64, EnergyMeterError> {
        let mut load: [f64; 3] = [0.0; 3];
        unsafe {
            libc::getloadavg(load.as_mut_ptr(), 3);
        }
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as f64;
        let cpu_utilization = (load[0] / num_cpus).min(1.0);

        // Apple Silicon typical: M1 ~10W idle / ~30W peak, M3 Max ~15W idle / ~60W peak
        let idle_watts = 8.0;
        let peak_watts = 40.0;
        let watts = idle_watts + (peak_watts - idle_watts) * cpu_utilization;
        Ok(watts)
    }
}

impl EnergyMeter for AppleSiliconMeter {
    fn read(&self) -> Result<EnergyReading, EnergyMeterError> {
        if !self.available {
            return Err(EnergyMeterError::NotSupported(
                "Apple Silicon meter only available on macOS aarch64".into(),
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let watts = self.read_power()?;
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            let last_ms = self
                .last_timestamp_ms
                .swap(now_ms, std::sync::atomic::Ordering::Relaxed);

            if last_ms > 0 {
                let elapsed_secs = (now_ms - last_ms) as f64 / 1000.0;
                let delta_uj = (watts * elapsed_secs * 1_000_000.0) as u64;
                self.cumulative_joules
                    .fetch_add(delta_uj, std::sync::atomic::Ordering::Relaxed);
            }

            let total_uj = self
                .cumulative_joules
                .load(std::sync::atomic::Ordering::Relaxed);
            let total_joules = total_uj as f64 / 1_000_000.0;

            Ok(EnergyReading::new(
                Joules::new(total_joules),
                Watts::new(watts),
                now_ms,
            ))
        }

        #[cfg(not(target_os = "macos"))]
        Err(EnergyMeterError::NotSupported(
            "Apple Silicon meter only available on macOS".into(),
        ))
    }

    fn name(&self) -> &str {
        "apple-silicon"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn energy_source(&self) -> EnergySource {
        EnergySource::WallPower
    }

    fn thermal_state(&self) -> ThermalState {
        ThermalState::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_meter_creation() {
        let meter = AppleSiliconMeter::new();
        assert_eq!(meter.name(), "apple-silicon");
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn apple_meter_reads_on_apple_silicon() {
        let meter = AppleSiliconMeter::new();
        let reading = meter.read().unwrap();
        assert!(reading.watts_current.as_f64() > 0.0);
    }
}
