//! Energy management for intermittent, energy-harvesting devices.
//!
//! Provides checkpoint scheduling and harvest window prediction for
//! devices powered by solar, wind, vibration, thermal, or RF energy sources.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// HarvestSource
// ---------------------------------------------------------------------------

/// The type of energy-harvesting source powering a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HarvestSource {
    /// Photovoltaic / solar energy.
    Solar,
    /// Wind turbine or micro-wind generator.
    Wind,
    /// Piezoelectric or kinetic vibration harvesting.
    Vibration,
    /// Thermoelectric (Peltier/Seebeck) harvesting.
    Thermal,
    /// Radio-frequency energy harvesting.
    Rf,
}

impl fmt::Display for HarvestSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solar => write!(f, "Solar"),
            Self::Wind => write!(f, "Wind"),
            Self::Vibration => write!(f, "Vibration"),
            Self::Thermal => write!(f, "Thermal"),
            Self::Rf => write!(f, "RF"),
        }
    }
}

// ---------------------------------------------------------------------------
// HarvestProfile
// ---------------------------------------------------------------------------

/// Describes the energy-harvesting characteristics of a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarvestProfile {
    /// The type of harvesting source.
    pub source: HarvestSource,
    /// Average harvest power output (milliwatts).
    pub avg_power_mw: f64,
    /// Peak harvest power output (milliwatts).
    pub peak_power_mw: f64,
    /// Variability coefficient: 0.0 = constant, 1.0 = highly variable.
    pub variability: f64,
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A snapshot of device state saved when energy drops below a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Serialised application state.
    pub state_bytes: Vec<u8>,
    /// Energy remaining at the time of checkpoint (millijoules).
    pub energy_at_checkpoint_mj: f64,
    /// Timestamp when the checkpoint was taken (milliseconds).
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// HarvestWindow
// ---------------------------------------------------------------------------

/// A predicted window of time during which harvesting is expected to
/// produce significant energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarvestWindow {
    /// Start of the window (milliseconds).
    pub start_ms: u64,
    /// Duration of the window (milliseconds).
    pub duration_ms: u64,
    /// Expected energy yield during this window (millijoules).
    pub expected_energy_mj: f64,
}

// ---------------------------------------------------------------------------
// IntermittentScheduler
// ---------------------------------------------------------------------------

/// Scheduler for intermittent, energy-harvesting devices.
///
/// Tracks energy readings over time, decides when to checkpoint, and
/// predicts the next high-energy harvest window based on historical data.
pub struct IntermittentScheduler {
    /// Harvest profile describing the device's energy source.
    profile: HarvestProfile,
    /// Trigger a checkpoint when energy drops below this level (mJ).
    checkpoint_threshold_mj: f64,
    /// Historical energy readings: `(timestamp_ms, energy_mj)`.
    history: Vec<(u64, f64)>,
}

impl IntermittentScheduler {
    /// Creates a new scheduler with the given harvest profile and
    /// checkpoint threshold (millijoules).
    pub fn new(profile: HarvestProfile, threshold_mj: f64) -> Self {
        Self {
            profile,
            checkpoint_threshold_mj: threshold_mj,
            history: Vec::new(),
        }
    }

    /// Returns `true` when `current_energy_mj` is at or below the
    /// checkpoint threshold.
    pub fn should_checkpoint(&self, current_energy_mj: f64) -> bool {
        current_energy_mj <= self.checkpoint_threshold_mj
    }

    /// Records a timestamped energy reading for future prediction.
    pub fn record_reading(&mut self, timestamp_ms: u64, energy_mj: f64) {
        self.history.push((timestamp_ms, energy_mj));
    }

    /// Predicts the next harvest window based on historical readings.
    ///
    /// Looks at the history to compute the average inter-peak interval and
    /// average peak energy.  Returns `None` if fewer than two readings are
    /// available.
    pub fn predict_next_window(&self, now_ms: u64) -> Option<HarvestWindow> {
        if self.history.len() < 2 {
            return None;
        }

        // Compute average interval between consecutive readings.
        let mut total_interval: u64 = 0;
        let intervals = self.history.len() - 1;
        for i in 1..self.history.len() {
            total_interval += self.history[i].0.saturating_sub(self.history[i - 1].0);
        }
        let avg_interval = total_interval / intervals as u64;

        // Compute average energy across all readings.
        let avg_energy: f64 =
            self.history.iter().map(|(_, e)| e).sum::<f64>() / self.history.len() as f64;

        let last_ts = self.history.last().map(|(ts, _)| *ts).unwrap_or(now_ms);
        let start = if last_ts + avg_interval > now_ms {
            last_ts + avg_interval
        } else {
            now_ms + avg_interval
        };

        Some(HarvestWindow {
            start_ms: start,
            duration_ms: avg_interval,
            expected_energy_mj: avg_energy,
        })
    }

    /// Creates a checkpoint from the given application state.
    pub fn create_checkpoint(
        &self,
        state: &[u8],
        current_energy_mj: f64,
        timestamp_ms: u64,
    ) -> Checkpoint {
        Checkpoint {
            state_bytes: state.to_vec(),
            energy_at_checkpoint_mj: current_energy_mj,
            timestamp_ms,
        }
    }

    /// Estimates how many milliseconds the device can operate with the
    /// given energy (mJ), based on the profile's average power draw.
    ///
    /// `time_ms = energy_mj / avg_power_mw` because `mJ / mW = ms`.
    pub fn energy_to_time(&self, energy_mj: f64) -> f64 {
        if self.profile.avg_power_mw == 0.0 {
            return 0.0;
        }
        energy_mj / self.profile.avg_power_mw
    }

    /// Returns the number of historical readings recorded.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Clears all historical readings.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> HarvestProfile {
        HarvestProfile {
            source: HarvestSource::Solar,
            avg_power_mw: 5.0,
            peak_power_mw: 20.0,
            variability: 0.7,
        }
    }

    #[test]
    fn should_checkpoint_below_threshold() {
        let scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        assert!(scheduler.should_checkpoint(5.0));
    }

    #[test]
    fn should_checkpoint_above_threshold() {
        let scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        assert!(!scheduler.should_checkpoint(15.0));
    }

    #[test]
    fn should_checkpoint_at_threshold() {
        let scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        assert!(scheduler.should_checkpoint(10.0));
    }

    #[test]
    fn record_reading() {
        let mut scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        scheduler.record_reading(1000, 50.0);
        scheduler.record_reading(2000, 45.0);
        assert_eq!(scheduler.history_len(), 2);
    }

    #[test]
    fn predict_no_history() {
        let scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        assert!(scheduler.predict_next_window(5000).is_none());
    }

    #[test]
    fn predict_with_history() {
        let mut scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        scheduler.record_reading(1000, 50.0);
        scheduler.record_reading(2000, 60.0);
        scheduler.record_reading(3000, 55.0);

        let window = scheduler
            .predict_next_window(3500)
            .expect("should predict a window");
        // Average interval = ((2000-1000) + (3000-2000)) / 2 = 1000
        // Last timestamp = 3000; 3000 + 1000 = 4000 > 3500, so start = 4000
        assert_eq!(window.start_ms, 4000);
        assert_eq!(window.duration_ms, 1000);
        // Average energy = (50 + 60 + 55) / 3 = 55.0
        assert!((window.expected_energy_mj - 55.0).abs() < 1e-10);
    }

    #[test]
    fn create_checkpoint() {
        let scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        let state = b"app-state-bytes";
        let cp = scheduler.create_checkpoint(state, 8.5, 42_000);

        assert_eq!(cp.state_bytes, state.to_vec());
        assert!((cp.energy_at_checkpoint_mj - 8.5).abs() < 1e-10);
        assert_eq!(cp.timestamp_ms, 42_000);
    }

    #[test]
    fn energy_to_time() {
        let scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        // 50 mJ / 5 mW = 10 ms
        let time = scheduler.energy_to_time(50.0);
        assert!((time - 10.0).abs() < 1e-10);
    }

    #[test]
    fn harvest_source_display() {
        assert_eq!(format!("{}", HarvestSource::Solar), "Solar");
        assert_eq!(format!("{}", HarvestSource::Wind), "Wind");
        assert_eq!(format!("{}", HarvestSource::Vibration), "Vibration");
        assert_eq!(format!("{}", HarvestSource::Thermal), "Thermal");
        assert_eq!(format!("{}", HarvestSource::Rf), "RF");
    }

    #[test]
    fn harvest_profile_serialization() {
        let profile = sample_profile();
        let json = serde_json::to_string(&profile).expect("serialize");
        let d: HarvestProfile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(d.source, HarvestSource::Solar);
        assert!((d.avg_power_mw - 5.0).abs() < 1e-10);
        assert!((d.peak_power_mw - 20.0).abs() < 1e-10);
        assert!((d.variability - 0.7).abs() < 1e-10);
    }

    #[test]
    fn clear_history() {
        let mut scheduler = IntermittentScheduler::new(sample_profile(), 10.0);
        scheduler.record_reading(1000, 50.0);
        scheduler.record_reading(2000, 60.0);
        assert_eq!(scheduler.history_len(), 2);

        scheduler.clear_history();
        assert_eq!(scheduler.history_len(), 0);
    }

    #[test]
    fn checkpoint_serialization() {
        let cp = Checkpoint {
            state_bytes: vec![1, 2, 3, 4],
            energy_at_checkpoint_mj: 12.5,
            timestamp_ms: 99_000,
        };
        let json = serde_json::to_string(&cp).expect("serialize");
        let d: Checkpoint = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(d.state_bytes, vec![1, 2, 3, 4]);
        assert!((d.energy_at_checkpoint_mj - 12.5).abs() < 1e-10);
        assert_eq!(d.timestamp_ms, 99_000);
    }
}
