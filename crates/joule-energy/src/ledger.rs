//! Cumulative energy tracking per compute unit.
//!
//! Integrates sampled power draws (mW) into accumulated energy (millijoules)
//! using trapezoidal approximation. Tracks CPU, GPU, and NPU independently.
//!
//! All data stays local by default. Optional reporting requires explicit opt-in.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

const SESSION_HISTORY_CAPACITY: usize = 100;

/// Cumulative energy record for one session or reporting window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyRecord {
    pub cpu_mj: f64,
    pub gpu_mj: f64,
    pub npu_mj: f64,
    pub total_mj: f64,
    pub sample_count: u64,
    pub start_timestamp: u64,
    pub last_timestamp: u64,
}

impl Default for EnergyRecord {
    fn default() -> Self {
        let now = unix_timestamp();
        Self {
            cpu_mj: 0.0,
            gpu_mj: 0.0,
            npu_mj: 0.0,
            total_mj: 0.0,
            sample_count: 0,
            start_timestamp: now,
            last_timestamp: now,
        }
    }
}

/// A single power measurement snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSnapshot {
    pub cpu_mw: u64,
    pub gpu_mw: u64,
    pub npu_mw: u64,
    pub timestamp: u64,
}

impl Default for PowerSnapshot {
    fn default() -> Self {
        Self {
            cpu_mw: 0,
            gpu_mw: 0,
            npu_mw: 0,
            timestamp: unix_timestamp(),
        }
    }
}

/// Reporting granularity for optional energy reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportGranularity {
    Session,
    Minute,
    Detailed,
}

/// Configuration for energy reporting. Disabled by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    pub enabled: bool,
    pub anonymize: bool,
    pub granularity: ReportGranularity,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            anonymize: true,
            granularity: ReportGranularity::Session,
        }
    }
}

/// An energy report generated for optional external consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyReport {
    pub device_id: Option<String>,
    pub session: EnergyRecord,
    pub minute_breakdown: Vec<EnergyRecord>,
    pub config: ReportingConfig,
}

/// Always-on cumulative energy tracker.
///
/// Integrates sampled power draws into millijoules using trapezoidal
/// approximation. All data stays local. Optional reporting is disabled
/// by default and requires explicit opt-in.
pub struct EnergyLedger {
    current: EnergyRecord,
    session_history: VecDeque<EnergyRecord>,
    last_power_mw: PowerSnapshot,
    last_sample_time: Instant,
    has_first_sample: bool,
    reporting: ReportingConfig,
    frame_count: u64,
    device_id: String,
}

impl EnergyLedger {
    pub fn new() -> Self {
        Self {
            current: EnergyRecord::default(),
            session_history: VecDeque::with_capacity(SESSION_HISTORY_CAPACITY),
            last_power_mw: PowerSnapshot::default(),
            last_sample_time: Instant::now(),
            has_first_sample: false,
            reporting: ReportingConfig::default(),
            frame_count: 0,
            device_id: String::new(),
        }
    }

    /// Record a power sample. Call this each tick (~1s recommended).
    ///
    /// Integrates power (mW) over elapsed time (seconds) into millijoules:
    /// energy_mj = power_mw * dt_seconds (since mW * s = mJ).
    pub fn record_sample(&mut self, cpu_mw: u64, gpu_mw: u64, npu_mw: u64) {
        let now = Instant::now();
        let timestamp = unix_timestamp();

        if self.has_first_sample {
            let dt_secs = now.duration_since(self.last_sample_time).as_secs_f64();

            // Trapezoidal integration: average of last and current sample.
            let avg_cpu = (self.last_power_mw.cpu_mw as f64 + cpu_mw as f64) / 2.0;
            let avg_gpu = (self.last_power_mw.gpu_mw as f64 + gpu_mw as f64) / 2.0;
            let avg_npu = (self.last_power_mw.npu_mw as f64 + npu_mw as f64) / 2.0;

            let cpu_energy = avg_cpu * dt_secs;
            let gpu_energy = avg_gpu * dt_secs;
            let npu_energy = avg_npu * dt_secs;

            self.current.cpu_mj += cpu_energy;
            self.current.gpu_mj += gpu_energy;
            self.current.npu_mj += npu_energy;
            self.current.total_mj += cpu_energy + gpu_energy + npu_energy;
        } else {
            self.has_first_sample = true;
        }

        self.current.sample_count += 1;
        self.current.last_timestamp = timestamp;

        self.last_power_mw = PowerSnapshot {
            cpu_mw,
            gpu_mw,
            npu_mw,
            timestamp,
        };
        self.last_sample_time = now;
    }

    /// Record a sample from a `joule-db-energy` EnergySnapshot, bridging
    /// the hardware monitor to the application ledger.
    pub fn record_from_snapshot(&mut self, snap: &crate::EnergySnapshot) {
        // EnergySnapshot has total power_watts; split by utilization ratios.
        let total_mw = (snap.power_watts * 1000.0) as u64;
        let total_util = snap.cpu_utilization + snap.gpu_utilization + snap.npu_utilization;

        if total_util < 0.001 {
            self.record_sample(total_mw, 0, 0);
        } else {
            let cpu_mw = ((snap.cpu_utilization / total_util) * total_mw as f64) as u64;
            let gpu_mw = ((snap.gpu_utilization / total_util) * total_mw as f64) as u64;
            let npu_mw = ((snap.npu_utilization / total_util) * total_mw as f64) as u64;
            self.record_sample(cpu_mw, gpu_mw, npu_mw);
        }
    }

    /// Increment the frame counter.
    pub fn tick_frame(&mut self) {
        self.frame_count += 1;
    }

    /// Get the current session energy record.
    pub fn current_session(&self) -> &EnergyRecord {
        &self.current
    }

    /// Average millijoules consumed per frame.
    pub fn energy_per_frame(&self) -> f64 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.current.total_mj / self.frame_count as f64
        }
    }

    /// Take a snapshot and push to history.
    pub fn snapshot(&mut self) {
        if self.session_history.len() >= SESSION_HISTORY_CAPACITY {
            self.session_history.pop_front();
        }
        self.session_history.push_back(self.current.clone());
    }

    pub fn session_history(&self) -> &VecDeque<EnergyRecord> {
        &self.session_history
    }

    pub fn reporting(&self) -> &ReportingConfig {
        &self.reporting
    }

    pub fn set_reporting(&mut self, config: ReportingConfig) {
        self.reporting = config;
    }

    pub fn set_device_id(&mut self, id: String) {
        self.device_id = id;
    }

    /// Generate an energy report. Returns `None` if reporting is disabled.
    pub fn generate_report(&self) -> Option<EnergyReport> {
        if !self.reporting.enabled {
            return None;
        }

        let device_id = if self.reporting.anonymize {
            None
        } else {
            Some(self.device_id.clone())
        };

        let minute_breakdown = match self.reporting.granularity {
            ReportGranularity::Session => Vec::new(),
            ReportGranularity::Minute | ReportGranularity::Detailed => {
                self.session_history.iter().cloned().collect()
            }
        };

        Some(EnergyReport {
            device_id,
            session: self.current.clone(),
            minute_breakdown,
            config: self.reporting.clone(),
        })
    }

    /// Reset the ledger.
    pub fn reset(&mut self) {
        self.current = EnergyRecord::default();
        self.session_history.clear();
        self.has_first_sample = false;
        self.frame_count = 0;
    }
}

impl Default for EnergyLedger {
    fn default() -> Self {
        Self::new()
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_new() {
        let ledger = EnergyLedger::new();
        assert_eq!(ledger.current.total_mj, 0.0);
        assert_eq!(ledger.current.sample_count, 0);
        assert!(!ledger.reporting.enabled);
    }

    #[test]
    fn test_record_sample() {
        let mut ledger = EnergyLedger::new();
        ledger.record_sample(5000, 8000, 500);
        assert_eq!(ledger.current.sample_count, 1);
        assert_eq!(ledger.current.total_mj, 0.0);

        thread::sleep(Duration::from_millis(10));
        ledger.record_sample(5000, 8000, 500);
        assert_eq!(ledger.current.sample_count, 2);
        assert!(ledger.current.total_mj > 0.0);
    }

    #[test]
    fn test_cumulative() {
        let mut ledger = EnergyLedger::new();
        ledger.record_sample(1000, 2000, 100);
        thread::sleep(Duration::from_millis(10));
        ledger.record_sample(1000, 2000, 100);
        let first = ledger.current.total_mj;

        thread::sleep(Duration::from_millis(10));
        ledger.record_sample(1000, 2000, 100);
        assert!(ledger.current.total_mj > first);
    }

    #[test]
    fn test_energy_per_frame() {
        let mut ledger = EnergyLedger::new();
        assert_eq!(ledger.energy_per_frame(), 0.0);

        ledger.record_sample(10000, 20000, 1000);
        thread::sleep(Duration::from_millis(10));
        ledger.record_sample(10000, 20000, 1000);

        for _ in 0..60 {
            ledger.tick_frame();
        }
        assert!(ledger.energy_per_frame() > 0.0);
    }

    #[test]
    fn test_session_history_bounded() {
        let mut ledger = EnergyLedger::new();
        for _ in 0..150 {
            ledger.snapshot();
        }
        assert_eq!(ledger.session_history.len(), SESSION_HISTORY_CAPACITY);
    }

    #[test]
    fn test_reporting_disabled_by_default() {
        let ledger = EnergyLedger::new();
        assert!(ledger.generate_report().is_none());
    }

    #[test]
    fn test_reset() {
        let mut ledger = EnergyLedger::new();
        ledger.record_sample(5000, 8000, 500);
        thread::sleep(Duration::from_millis(10));
        ledger.record_sample(5000, 8000, 500);
        ledger.tick_frame();
        ledger.snapshot();

        assert!(ledger.current.total_mj > 0.0);
        ledger.reset();
        assert_eq!(ledger.current.total_mj, 0.0);
        assert!(ledger.session_history.is_empty());
    }
}
