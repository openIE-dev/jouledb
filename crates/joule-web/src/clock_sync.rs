//! Network clock synchronization (NTP-like) — offset estimation via RTT sampling.
//!
//! Implements Cristian's algorithm for server-time offset estimation, running
//! average with outlier rejection, configurable sample count, estimated
//! accuracy/confidence, monotonic adjustment (never go backwards), drift rate
//! estimation, sync quality metric, and periodic re-sync triggering.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Clock sync domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockSyncError {
    /// Not enough samples to produce an estimate.
    InsufficientSamples { have: usize, need: usize },
    /// RTT is negative or implausible.
    InvalidRtt(i64),
    /// Sample rejected as outlier.
    OutlierRejected { rtt_ms: i64, threshold_ms: i64 },
}

impl fmt::Display for ClockSyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSamples { have, need } => {
                write!(f, "insufficient samples: have {have}, need {need}")
            }
            Self::InvalidRtt(rtt) => write!(f, "invalid RTT: {rtt}ms"),
            Self::OutlierRejected { rtt_ms, threshold_ms } => {
                write!(f, "outlier rejected: rtt={rtt_ms}ms, threshold={threshold_ms}ms")
            }
        }
    }
}

impl std::error::Error for ClockSyncError {}

// ── RTT Sample ──────────────────────────────────────────────────

/// A single round-trip time sample with server timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RttSample {
    /// Time the request was sent (local clock, ms).
    pub send_time_ms: i64,
    /// Time the server received the request (server clock, ms).
    pub server_time_ms: i64,
    /// Time the response was received (local clock, ms).
    pub recv_time_ms: i64,
}

impl RttSample {
    pub fn new(send_time_ms: i64, server_time_ms: i64, recv_time_ms: i64) -> Self {
        Self { send_time_ms, server_time_ms, recv_time_ms }
    }

    /// Round-trip time in milliseconds.
    pub fn rtt_ms(&self) -> i64 {
        self.recv_time_ms - self.send_time_ms
    }

    /// Estimated one-way latency (half RTT).
    pub fn one_way_ms(&self) -> i64 {
        self.rtt_ms() / 2
    }

    /// Cristian's offset estimate: server_time - (send + rtt/2).
    pub fn offset_estimate_ms(&self) -> i64 {
        self.server_time_ms - (self.send_time_ms + self.one_way_ms())
    }
}

impl fmt::Display for RttSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RttSample(rtt={}ms, offset={}ms)", self.rtt_ms(), self.offset_estimate_ms())
    }
}

// ── Sync Quality ────────────────────────────────────────────────

/// Quality level of the clock synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SyncQuality {
    /// No samples yet.
    None,
    /// Very few samples, low confidence.
    Poor,
    /// Enough samples, moderate confidence.
    Fair,
    /// Many samples, high confidence.
    Good,
    /// Excellent convergence.
    Excellent,
}

impl fmt::Display for SyncQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::None => "none",
            Self::Poor => "poor",
            Self::Fair => "fair",
            Self::Good => "good",
            Self::Excellent => "excellent",
        };
        write!(f, "{s}")
    }
}

// ── Clock Sync Config ───────────────────────────────────────────

/// Configuration for clock synchronization.
#[derive(Debug, Clone)]
pub struct ClockSyncConfig {
    /// Maximum number of samples to keep in the window.
    pub max_samples: usize,
    /// Minimum samples before producing an estimate.
    pub min_samples: usize,
    /// Outlier rejection: reject samples with RTT > mean + factor * stddev.
    pub outlier_factor: f64,
    /// Maximum acceptable RTT in ms.
    pub max_rtt_ms: i64,
    /// Re-sync interval in ms (trigger re-sync if last sample is older).
    pub resync_interval_ms: i64,
}

impl ClockSyncConfig {
    pub fn new() -> Self {
        Self {
            max_samples: 16,
            min_samples: 3,
            outlier_factor: 2.0,
            max_rtt_ms: 5000,
            resync_interval_ms: 30_000,
        }
    }

    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = n;
        self
    }

    pub fn with_min_samples(mut self, n: usize) -> Self {
        self.min_samples = n;
        self
    }

    pub fn with_outlier_factor(mut self, f: f64) -> Self {
        self.outlier_factor = f;
        self
    }

    pub fn with_max_rtt(mut self, ms: i64) -> Self {
        self.max_rtt_ms = ms;
        self
    }

    pub fn with_resync_interval(mut self, ms: i64) -> Self {
        self.resync_interval_ms = ms;
        self
    }
}

impl Default for ClockSyncConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Clock Sync ──────────────────────────────────────────────────

/// Estimates server time offset from local clock via RTT sampling.
pub struct ClockSync {
    pub config: ClockSyncConfig,
    samples: Vec<RttSample>,
    /// Current estimated offset (server - local) in ms.
    current_offset_ms: i64,
    /// Monotonic floor: offset never decreases below this.
    monotonic_floor_ms: i64,
    /// Last drift estimate (ms per second of wall time).
    drift_rate_ms_per_sec: f64,
    /// Total samples ever added (including rejected).
    total_samples_added: u64,
    total_samples_rejected: u64,
}

impl ClockSync {
    pub fn new() -> Self {
        Self {
            config: ClockSyncConfig::new(),
            samples: Vec::new(),
            current_offset_ms: 0,
            monotonic_floor_ms: i64::MIN,
            drift_rate_ms_per_sec: 0.0,
            total_samples_added: 0,
            total_samples_rejected: 0,
        }
    }

    pub fn with_config(mut self, config: ClockSyncConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a round-trip sample. Returns the updated offset or an error.
    pub fn add_sample(&mut self, sample: RttSample) -> Result<i64, ClockSyncError> {
        let rtt = sample.rtt_ms();
        if rtt < 0 {
            self.total_samples_rejected += 1;
            return Err(ClockSyncError::InvalidRtt(rtt));
        }
        if rtt > self.config.max_rtt_ms {
            self.total_samples_rejected += 1;
            return Err(ClockSyncError::OutlierRejected {
                rtt_ms: rtt,
                threshold_ms: self.config.max_rtt_ms,
            });
        }

        // Outlier rejection based on existing samples.
        if self.samples.len() >= self.config.min_samples {
            let (mean, stddev) = self.rtt_stats();
            let threshold = mean + self.config.outlier_factor * stddev;
            if (rtt as f64) > threshold {
                self.total_samples_rejected += 1;
                return Err(ClockSyncError::OutlierRejected {
                    rtt_ms: rtt,
                    threshold_ms: threshold as i64,
                });
            }
        }

        self.samples.push(sample);
        self.total_samples_added += 1;

        // Evict oldest if over capacity.
        while self.samples.len() > self.config.max_samples {
            self.samples.remove(0);
        }

        self.recompute_offset();
        Ok(self.current_offset_ms)
    }

    /// Recompute offset from all retained samples using weighted average.
    fn recompute_offset(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let old_offset = self.current_offset_ms;

        // Weight inversely by RTT (lower RTT = higher confidence).
        let mut weighted_sum: f64 = 0.0;
        let mut weight_total: f64 = 0.0;
        for s in &self.samples {
            let rtt = s.rtt_ms().max(1) as f64;
            let w = 1.0 / rtt;
            weighted_sum += s.offset_estimate_ms() as f64 * w;
            weight_total += w;
        }

        let new_offset = if weight_total > 0.0 {
            (weighted_sum / weight_total).round() as i64
        } else {
            old_offset
        };

        // Monotonic: never go backwards.
        self.current_offset_ms = new_offset.max(self.monotonic_floor_ms);
        self.monotonic_floor_ms = self.current_offset_ms;

        // Estimate drift.
        if self.samples.len() >= 2 {
            let first = &self.samples[0];
            let last = &self.samples[self.samples.len() - 1];
            let time_span = (last.send_time_ms - first.send_time_ms) as f64;
            if time_span > 0.0 {
                let offset_change = (last.offset_estimate_ms() - first.offset_estimate_ms()) as f64;
                self.drift_rate_ms_per_sec = offset_change / (time_span / 1000.0);
            }
        }
    }

    /// Mean and standard deviation of RTTs in the sample window.
    fn rtt_stats(&self) -> (f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.samples.len() as f64;
        let mean = self.samples.iter().map(|s| s.rtt_ms() as f64).sum::<f64>() / n;
        let variance = self.samples.iter()
            .map(|s| {
                let d = s.rtt_ms() as f64 - mean;
                d * d
            })
            .sum::<f64>() / n;
        (mean, variance.sqrt())
    }

    /// Current estimated offset (server - local) in ms.
    pub fn offset_ms(&self) -> Result<i64, ClockSyncError> {
        if self.samples.len() < self.config.min_samples {
            return Err(ClockSyncError::InsufficientSamples {
                have: self.samples.len(),
                need: self.config.min_samples,
            });
        }
        Ok(self.current_offset_ms)
    }

    /// Convert a local timestamp to estimated server time.
    pub fn to_server_time(&self, local_ms: i64) -> Result<i64, ClockSyncError> {
        Ok(local_ms + self.offset_ms()?)
    }

    /// Convert a server timestamp to estimated local time.
    pub fn to_local_time(&self, server_ms: i64) -> Result<i64, ClockSyncError> {
        Ok(server_ms - self.offset_ms()?)
    }

    /// Estimated accuracy: half the minimum RTT in the window.
    pub fn accuracy_ms(&self) -> Option<i64> {
        self.samples.iter().map(|s| s.rtt_ms()).min().map(|r| r / 2)
    }

    /// Drift rate in ms per second of wall time.
    pub fn drift_rate(&self) -> f64 {
        self.drift_rate_ms_per_sec
    }

    /// Sync quality based on sample count and RTT variance.
    pub fn quality(&self) -> SyncQuality {
        let n = self.samples.len();
        if n == 0 {
            return SyncQuality::None;
        }
        if n < self.config.min_samples {
            return SyncQuality::Poor;
        }
        let (_, stddev) = self.rtt_stats();
        if stddev < 5.0 && n >= self.config.max_samples / 2 {
            SyncQuality::Excellent
        } else if stddev < 20.0 {
            SyncQuality::Good
        } else {
            SyncQuality::Fair
        }
    }

    /// Whether re-sync is needed based on last sample age.
    pub fn needs_resync(&self, current_local_ms: i64) -> bool {
        match self.samples.last() {
            None => true,
            Some(s) => (current_local_ms - s.recv_time_ms) > self.config.resync_interval_ms,
        }
    }

    /// Number of samples in the current window.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Total samples ever processed (accepted + rejected).
    pub fn total_processed(&self) -> u64 {
        self.total_samples_added + self.total_samples_rejected
    }

    /// Reset all samples and estimates.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.current_offset_ms = 0;
        self.monotonic_floor_ms = i64::MIN;
        self.drift_rate_ms_per_sec = 0.0;
    }
}

impl Default for ClockSync {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ClockSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClockSync(samples={}, offset={}ms, quality={})",
            self.samples.len(),
            self.current_offset_ms,
            self.quality(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(send: i64, server: i64, recv: i64) -> RttSample {
        RttSample::new(send, server, recv)
    }

    #[test]
    fn rtt_sample_basics() {
        let s = make_sample(100, 200, 120);
        assert_eq!(s.rtt_ms(), 20);
        assert_eq!(s.one_way_ms(), 10);
    }

    #[test]
    fn rtt_sample_offset_estimate() {
        // send=100, server=210, recv=120 -> rtt=20, offset = 210 - (100+10) = 100
        let s = make_sample(100, 210, 120);
        assert_eq!(s.offset_estimate_ms(), 100);
    }

    #[test]
    fn rtt_sample_display() {
        let s = make_sample(0, 50, 10);
        let d = format!("{s}");
        assert!(d.contains("rtt=10ms"));
    }

    #[test]
    fn clock_sync_insufficient_samples() {
        let cs = ClockSync::new();
        assert!(matches!(cs.offset_ms(), Err(ClockSyncError::InsufficientSamples { .. })));
    }

    #[test]
    fn clock_sync_add_samples_and_offset() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_min_samples(2),
        );
        cs.add_sample(make_sample(0, 105, 10)).unwrap();
        cs.add_sample(make_sample(10, 115, 20)).unwrap();
        let off = cs.offset_ms().unwrap();
        assert!(off > 90 && off < 110, "offset={off}");
    }

    #[test]
    fn clock_sync_reject_negative_rtt() {
        let mut cs = ClockSync::new();
        let res = cs.add_sample(make_sample(100, 200, 50));
        assert!(matches!(res, Err(ClockSyncError::InvalidRtt(_))));
    }

    #[test]
    fn clock_sync_reject_high_rtt() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_max_rtt(100),
        );
        let res = cs.add_sample(make_sample(0, 500, 200));
        assert!(matches!(res, Err(ClockSyncError::OutlierRejected { .. })));
    }

    #[test]
    fn clock_sync_monotonic_offset() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_min_samples(1).with_max_samples(2),
        );
        cs.add_sample(make_sample(0, 200, 10)).unwrap();
        let first = cs.offset_ms().unwrap();
        // Add a sample that would give a lower offset.
        cs.add_sample(make_sample(10, 100, 20)).unwrap();
        let second = cs.offset_ms().unwrap();
        assert!(second >= first, "monotonic violated: {second} < {first}");
    }

    #[test]
    fn clock_sync_to_server_time() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_min_samples(1),
        );
        cs.add_sample(make_sample(0, 100, 10)).unwrap();
        let server_t = cs.to_server_time(50).unwrap();
        assert!(server_t > 50);
    }

    #[test]
    fn clock_sync_to_local_time() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_min_samples(1),
        );
        cs.add_sample(make_sample(0, 100, 10)).unwrap();
        let local_t = cs.to_local_time(150).unwrap();
        assert!(local_t < 150);
    }

    #[test]
    fn clock_sync_accuracy() {
        let mut cs = ClockSync::new();
        assert!(cs.accuracy_ms().is_none());
        cs.add_sample(make_sample(0, 100, 20)).unwrap();
        assert_eq!(cs.accuracy_ms(), Some(10));
    }

    #[test]
    fn clock_sync_quality_none() {
        let cs = ClockSync::new();
        assert_eq!(cs.quality(), SyncQuality::None);
    }

    #[test]
    fn clock_sync_quality_improves() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_min_samples(2).with_max_samples(8),
        );
        cs.add_sample(make_sample(0, 105, 10)).unwrap();
        assert_eq!(cs.quality(), SyncQuality::Poor);
        for i in 1..8 {
            let _ = cs.add_sample(make_sample(i * 10, 105 + i * 10, i * 10 + 10));
        }
        assert!(cs.quality() >= SyncQuality::Fair);
    }

    #[test]
    fn clock_sync_needs_resync() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_resync_interval(1000),
        );
        assert!(cs.needs_resync(0));
        cs.add_sample(make_sample(0, 100, 10)).unwrap();
        assert!(!cs.needs_resync(500));
        assert!(cs.needs_resync(2000));
    }

    #[test]
    fn clock_sync_sample_count() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_max_samples(4),
        );
        for i in 0..6 {
            let _ = cs.add_sample(make_sample(i * 10, 100 + i * 10, i * 10 + 10));
        }
        assert_eq!(cs.sample_count(), 4);
    }

    #[test]
    fn clock_sync_reset() {
        let mut cs = ClockSync::new();
        cs.add_sample(make_sample(0, 100, 10)).unwrap();
        cs.reset();
        assert_eq!(cs.sample_count(), 0);
    }

    #[test]
    fn clock_sync_drift_rate() {
        let mut cs = ClockSync::new().with_config(
            ClockSyncConfig::new().with_min_samples(2),
        );
        cs.add_sample(make_sample(0, 100, 10)).unwrap();
        cs.add_sample(make_sample(1000, 1200, 1010)).unwrap();
        let drift = cs.drift_rate();
        // Some drift expected since offsets differ.
        assert!(drift.abs() < 1000.0);
    }

    #[test]
    fn clock_sync_display() {
        let cs = ClockSync::new();
        let d = format!("{cs}");
        assert!(d.contains("ClockSync"));
    }

    #[test]
    fn config_builder_chain() {
        let cfg = ClockSyncConfig::new()
            .with_max_samples(32)
            .with_min_samples(5)
            .with_outlier_factor(3.0)
            .with_max_rtt(2000)
            .with_resync_interval(60_000);
        assert_eq!(cfg.max_samples, 32);
        assert_eq!(cfg.min_samples, 5);
        assert_eq!(cfg.max_rtt_ms, 2000);
    }

    #[test]
    fn sync_quality_ordering() {
        assert!(SyncQuality::Excellent > SyncQuality::Good);
        assert!(SyncQuality::Good > SyncQuality::Fair);
        assert!(SyncQuality::Fair > SyncQuality::Poor);
        assert!(SyncQuality::Poor > SyncQuality::None);
    }
}
