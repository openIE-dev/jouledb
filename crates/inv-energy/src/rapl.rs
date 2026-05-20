use crate::meter::{EnergyMeter, EnergyMeterError};
use inv_core::energy::{EnergyReading, EnergySource};
use std::path::Path;

/// Intel/AMD RAPL (Running Average Power Limit) energy meter.
/// Reads from `/sys/class/powercap/intel-rapl/` on Linux.
pub struct RaplMeter {
    available: bool,
}

impl Default for RaplMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl RaplMeter {
    pub fn new() -> Self {
        let available = cfg!(target_os = "linux") && Path::new(RAPL_ENERGY_PATH).exists();
        Self { available }
    }
}

const RAPL_ENERGY_PATH: &str = "/sys/class/powercap/intel-rapl:0/energy_uj";

impl EnergyMeter for RaplMeter {
    fn read(&self) -> Result<EnergyReading, EnergyMeterError> {
        if !self.available {
            return Err(EnergyMeterError::NotSupported(
                "RAPL not available (not Linux or no powercap)".into(),
            ));
        }

        #[cfg(target_os = "linux")]
        {
            use inv_core::energy::{Joules, Watts};

            let energy_uj = std::fs::read_to_string(RAPL_ENERGY_PATH)
                .map_err(|e| EnergyMeterError::ReadFailed(format!("reading RAPL: {e}")))?
                .trim()
                .parse::<u64>()
                .map_err(|e| EnergyMeterError::ReadFailed(format!("parsing RAPL: {e}")))?;

            let joules = energy_uj as f64 / 1_000_000.0;
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            Ok(EnergyReading::new(
                Joules::new(joules),
                Watts::new(0.0),
                now_ms,
            ))
        }

        #[cfg(not(target_os = "linux"))]
        Err(EnergyMeterError::NotSupported(
            "RAPL only available on Linux".into(),
        ))
    }

    fn name(&self) -> &str {
        "rapl"
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
    fn rapl_meter_platform_detection() {
        let meter = RaplMeter::new();
        if cfg!(not(target_os = "linux")) {
            assert!(!meter.is_available());
        }
        assert_eq!(meter.name(), "rapl");
    }
}
