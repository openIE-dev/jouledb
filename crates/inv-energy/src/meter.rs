use inv_core::energy::{EnergyReading, EnergySource, Joules, ThermalState, Watts};
use std::time::{SystemTime, UNIX_EPOCH};

/// The unified energy meter trait — platform-specific backends implement this.
pub trait EnergyMeter: Send + Sync {
    /// Take a point-in-time energy reading.
    fn read(&self) -> Result<EnergyReading, EnergyMeterError>;

    /// The name of this meter backend (e.g., "rapl", "apple-silicon", "estimation").
    fn name(&self) -> &str;

    /// Whether this meter is available on the current platform.
    fn is_available(&self) -> bool;

    /// The energy source this meter reports for.
    fn energy_source(&self) -> EnergySource;

    /// Current thermal state, if detectable.
    fn thermal_state(&self) -> ThermalState {
        ThermalState::Normal
    }

    /// Battery percentage, if applicable.
    fn battery_pct(&self) -> Option<u8> {
        None
    }
}

/// Composite meter that tries multiple backends and uses the first available.
pub struct CompositeMeter {
    meters: Vec<Box<dyn EnergyMeter>>,
    active_index: Option<usize>,
}

impl CompositeMeter {
    /// Create from a list of meters, selecting the first available one.
    pub fn new(meters: Vec<Box<dyn EnergyMeter>>) -> Self {
        let active_index = meters.iter().position(|m| m.is_available());
        Self {
            meters,
            active_index,
        }
    }

    /// The active meter backend, if any.
    pub fn active(&self) -> Option<&dyn EnergyMeter> {
        self.active_index.map(|i| self.meters[i].as_ref())
    }

    /// The name of the active meter.
    pub fn active_name(&self) -> &str {
        self.active().map(|m| m.name()).unwrap_or("none")
    }
}

impl EnergyMeter for CompositeMeter {
    fn read(&self) -> Result<EnergyReading, EnergyMeterError> {
        match self.active() {
            Some(meter) => meter.read(),
            None => Err(EnergyMeterError::NoMeterAvailable),
        }
    }

    fn name(&self) -> &str {
        self.active_name()
    }

    fn is_available(&self) -> bool {
        self.active_index.is_some()
    }

    fn energy_source(&self) -> EnergySource {
        self.active()
            .map(|m| m.energy_source())
            .unwrap_or(EnergySource::Unknown)
    }

    fn thermal_state(&self) -> ThermalState {
        self.active()
            .map(|m| m.thermal_state())
            .unwrap_or(ThermalState::Normal)
    }

    fn battery_pct(&self) -> Option<u8> {
        self.active().and_then(|m| m.battery_pct())
    }
}

/// An estimation-based fallback meter that estimates energy from CPU load.
/// Used when no hardware meter is available.
pub struct EstimationMeter {
    /// Assumed TDP in watts for the current system.
    tdp_watts: f64,
    /// Baseline idle power as a fraction of TDP (typically 0.1-0.3).
    idle_fraction: f64,
    /// Cumulative joules estimated.
    cumulative_joules: std::sync::atomic::AtomicU64,
    /// Last reading timestamp.
    last_timestamp_ms: std::sync::atomic::AtomicU64,
}

impl EstimationMeter {
    /// Create with a TDP estimate.
    pub fn new(tdp_watts: f64) -> Self {
        Self {
            tdp_watts,
            idle_fraction: 0.2,
            cumulative_joules: std::sync::atomic::AtomicU64::new(0),
            last_timestamp_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Estimate current power draw from CPU load (0.0 to 1.0).
    fn estimate_watts(&self, cpu_load: f64) -> f64 {
        let idle = self.tdp_watts * self.idle_fraction;
        let active = self.tdp_watts * (1.0 - self.idle_fraction) * cpu_load;
        idle + active
    }

    /// Get a rough CPU load estimate. Platform-specific.
    fn cpu_load_estimate(&self) -> f64 {
        // Simple estimation: use system load average if available
        #[cfg(target_os = "macos")]
        {
            let mut load: [f64; 3] = [0.0; 3];
            libc_getloadavg(&mut load);
            (load[0] / num_cpus() as f64).min(1.0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0.3 // Default estimate when we can't measure
        }
    }
}

#[cfg(target_os = "macos")]
fn libc_getloadavg(load: &mut [f64; 3]) {
    unsafe {
        libc::getloadavg(load.as_mut_ptr(), 3);
    }
}

#[cfg(target_os = "macos")]
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

impl EnergyMeter for EstimationMeter {
    fn read(&self) -> Result<EnergyReading, EnergyMeterError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let cpu_load = self.cpu_load_estimate();
        let watts = self.estimate_watts(cpu_load);

        let last_ms = self
            .last_timestamp_ms
            .swap(now_ms, std::sync::atomic::Ordering::Relaxed);

        // Accumulate joules based on elapsed time
        if last_ms > 0 {
            let elapsed_secs = (now_ms - last_ms) as f64 / 1000.0;
            let joules_delta = watts * elapsed_secs;
            let delta_bits = (joules_delta * 1_000_000.0) as u64; // Store as microjoules
            self.cumulative_joules
                .fetch_add(delta_bits, std::sync::atomic::Ordering::Relaxed);
        }

        let total_microjoules = self
            .cumulative_joules
            .load(std::sync::atomic::Ordering::Relaxed);
        let total_joules = total_microjoules as f64 / 1_000_000.0;

        Ok(EnergyReading::new(
            Joules::new(total_joules),
            Watts::new(watts),
            now_ms,
        ))
    }

    fn name(&self) -> &str {
        "estimation"
    }

    fn is_available(&self) -> bool {
        true // Always available as fallback
    }

    fn energy_source(&self) -> EnergySource {
        EnergySource::Unknown
    }
}

/// Detect the best available meter for the current platform.
pub fn detect_meter() -> CompositeMeter {
    let mut meters: Vec<Box<dyn EnergyMeter>> = Vec::new();

    // Platform-specific meters (tried in order of accuracy)
    #[cfg(target_os = "macos")]
    {
        meters.push(Box::new(crate::apple::AppleSiliconMeter::new()));
    }

    #[cfg(target_os = "linux")]
    {
        meters.push(Box::new(crate::rapl::RaplMeter::new()));
    }

    // Estimation fallback (always available)
    meters.push(Box::new(EstimationMeter::new(65.0))); // 65W TDP default

    CompositeMeter::new(meters)
}

#[derive(Debug, thiserror::Error)]
pub enum EnergyMeterError {
    #[error("no energy meter available on this platform")]
    NoMeterAvailable,
    #[error("meter read failed: {0}")]
    ReadFailed(String),
    #[error("meter not supported: {0}")]
    NotSupported(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimation_meter_always_available() {
        let meter = EstimationMeter::new(45.0);
        assert!(meter.is_available());
        assert_eq!(meter.name(), "estimation");
    }

    #[test]
    fn estimation_meter_reads() {
        let meter = EstimationMeter::new(45.0);
        let reading = meter.read().unwrap();
        assert!(reading.watts_current.as_f64() > 0.0);
    }

    #[test]
    fn estimation_meter_accumulates() {
        let meter = EstimationMeter::new(45.0);
        let r1 = meter.read().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let r2 = meter.read().unwrap();
        // Second reading should have more accumulated joules
        assert!(r2.timestamp_ms >= r1.timestamp_ms);
    }

    #[test]
    fn composite_meter_uses_estimation_fallback() {
        let meter = detect_meter();
        assert!(meter.is_available());
        // Should always have at least the estimation fallback
        let reading = meter.read().unwrap();
        assert!(reading.watts_current.as_f64() > 0.0);
    }

    #[test]
    fn composite_meter_empty() {
        let meter = CompositeMeter::new(vec![]);
        assert!(!meter.is_available());
        assert!(meter.read().is_err());
        assert_eq!(meter.active_name(), "none");
    }
}
