//! Load testing profiles — virtual users, ramp-up/down, constant load, step
//! function, percentile calculation, throughput/latency tracking.
//!
//! Replaces JS load testing tools (k6, Artillery, autocannon) with a pure-Rust
//! load profile engine for modeling and analyzing load test scenarios.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Load profile errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadProfileError {
    /// Invalid ramp configuration.
    InvalidRamp(String),
    /// No data collected.
    NoData,
    /// Invalid percentile (must be 0..=100).
    InvalidPercentile(u8),
    /// Stage not found.
    StageNotFound(String),
    /// Duration must be positive.
    InvalidDuration(String),
}

impl fmt::Display for LoadProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRamp(msg) => write!(f, "invalid ramp: {msg}"),
            Self::NoData => write!(f, "no data collected"),
            Self::InvalidPercentile(p) => write!(f, "invalid percentile: {p} (must be 0..=100)"),
            Self::StageNotFound(name) => write!(f, "stage not found: {name}"),
            Self::InvalidDuration(msg) => write!(f, "invalid duration: {msg}"),
        }
    }
}

impl std::error::Error for LoadProfileError {}

// ── Load Shape ─────────────────────────────────────────────────

/// A stage in a load profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadStage {
    /// Stage name for identification.
    pub name: String,
    /// Duration of this stage in seconds.
    pub duration_secs: u64,
    /// Target number of virtual users at end of stage.
    pub target_vus: u32,
    /// Shape of the transition within this stage.
    pub shape: StageShape,
}

/// Shape of load within a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageShape {
    /// Linear ramp from previous VU count to target.
    Linear,
    /// Instant jump to target.
    Step,
    /// Hold at target (constant).
    Constant,
}

impl LoadStage {
    /// Create a ramp-up stage.
    pub fn ramp_up(name: impl Into<String>, duration_secs: u64, target_vus: u32) -> Self {
        Self {
            name: name.into(),
            duration_secs,
            target_vus,
            shape: StageShape::Linear,
        }
    }

    /// Create a constant-load stage.
    pub fn constant(name: impl Into<String>, duration_secs: u64, vus: u32) -> Self {
        Self {
            name: name.into(),
            duration_secs,
            target_vus: vus,
            shape: StageShape::Constant,
        }
    }

    /// Create a step-function stage.
    pub fn step(name: impl Into<String>, duration_secs: u64, target_vus: u32) -> Self {
        Self {
            name: name.into(),
            duration_secs,
            target_vus,
            shape: StageShape::Step,
        }
    }
}

// ── Load Profile ───────────────────────────────────────────────

/// A complete load profile composed of sequential stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadProfile {
    /// Profile name.
    pub name: String,
    /// Sequential stages.
    pub stages: Vec<LoadStage>,
    /// Tags for categorization.
    pub tags: HashMap<String, String>,
}

impl LoadProfile {
    /// Create a new empty profile.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            tags: HashMap::new(),
        }
    }

    /// Add a stage.
    pub fn add_stage(&mut self, stage: LoadStage) {
        self.stages.push(stage);
    }

    /// Add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Total duration in seconds.
    pub fn total_duration_secs(&self) -> u64 {
        self.stages.iter().map(|s| s.duration_secs).sum()
    }

    /// Peak VU count across all stages.
    pub fn peak_vus(&self) -> u32 {
        self.stages.iter().map(|s| s.target_vus).max().unwrap_or(0)
    }

    /// Calculate the VU count at a specific elapsed second.
    pub fn vus_at(&self, elapsed_secs: u64) -> u32 {
        let mut offset = 0u64;
        let mut prev_vus = 0u32;

        for stage in &self.stages {
            let stage_end = offset + stage.duration_secs;
            if elapsed_secs < stage_end {
                let within = elapsed_secs - offset;
                return match stage.shape {
                    StageShape::Step => stage.target_vus,
                    StageShape::Constant => stage.target_vus,
                    StageShape::Linear => {
                        if stage.duration_secs == 0 {
                            stage.target_vus
                        } else {
                            let from = prev_vus as f64;
                            let to = stage.target_vus as f64;
                            let ratio = within as f64 / stage.duration_secs as f64;
                            (from + (to - from) * ratio) as u32
                        }
                    }
                };
            }
            prev_vus = stage.target_vus;
            offset = stage_end;
        }

        // After all stages, return last target
        prev_vus
    }

    /// Generate a VU timeline at 1-second resolution.
    pub fn timeline(&self) -> Vec<(u64, u32)> {
        let total = self.total_duration_secs();
        (0..=total).map(|t| (t, self.vus_at(t))).collect()
    }
}

// ── Latency Sample ─────────────────────────────────────────────

/// A single request latency measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencySample {
    /// Timestamp offset from test start in milliseconds.
    pub timestamp_ms: u64,
    /// Response latency in microseconds.
    pub latency_us: u64,
    /// HTTP status code.
    pub status: u16,
    /// Whether the request was successful.
    pub success: bool,
}

// ── Latency Tracker ────────────────────────────────────────────

/// Tracks latency samples and computes statistics.
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    samples: Vec<LatencySample>,
    labels: HashMap<String, Vec<usize>>,
}

impl LatencyTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            labels: HashMap::new(),
        }
    }

    /// Record a sample.
    pub fn record(&mut self, sample: LatencySample) {
        self.samples.push(sample);
    }

    /// Record a sample with a label (e.g., endpoint name).
    pub fn record_labeled(&mut self, label: impl Into<String>, sample: LatencySample) {
        let idx = self.samples.len();
        self.samples.push(sample);
        self.labels.entry(label.into()).or_default().push(idx);
    }

    /// Total number of samples.
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Number of successful requests.
    pub fn success_count(&self) -> usize {
        self.samples.iter().filter(|s| s.success).count()
    }

    /// Number of failed requests.
    pub fn failure_count(&self) -> usize {
        self.samples.iter().filter(|s| !s.success).count()
    }

    /// Get sorted latencies (microseconds) for percentile calculation.
    fn sorted_latencies(&self) -> Vec<u64> {
        let mut lats: Vec<u64> = self.samples.iter().map(|s| s.latency_us).collect();
        lats.sort_unstable();
        lats
    }

    /// Get sorted latencies for a specific label.
    fn sorted_latencies_for(&self, label: &str) -> Vec<u64> {
        let mut lats: Vec<u64> = self.labels
            .get(label)
            .map(|indices| indices.iter().map(|i| self.samples[*i].latency_us).collect())
            .unwrap_or_default();
        lats.sort_unstable();
        lats
    }

    /// Calculate a percentile (0-100) across all samples.
    pub fn percentile(&self, p: u8) -> Result<u64, LoadProfileError> {
        if p > 100 {
            return Err(LoadProfileError::InvalidPercentile(p));
        }
        let sorted = self.sorted_latencies();
        if sorted.is_empty() {
            return Err(LoadProfileError::NoData);
        }
        let idx = ((p as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        Ok(sorted[idx.min(sorted.len() - 1)])
    }

    /// Calculate a percentile for a specific label.
    pub fn percentile_for(&self, label: &str, p: u8) -> Result<u64, LoadProfileError> {
        if p > 100 {
            return Err(LoadProfileError::InvalidPercentile(p));
        }
        let sorted = self.sorted_latencies_for(label);
        if sorted.is_empty() {
            return Err(LoadProfileError::NoData);
        }
        let idx = ((p as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        Ok(sorted[idx.min(sorted.len() - 1)])
    }

    /// Mean latency in microseconds.
    pub fn mean_us(&self) -> Result<f64, LoadProfileError> {
        if self.samples.is_empty() {
            return Err(LoadProfileError::NoData);
        }
        let sum: u64 = self.samples.iter().map(|s| s.latency_us).sum();
        Ok(sum as f64 / self.samples.len() as f64)
    }

    /// Minimum latency in microseconds.
    pub fn min_us(&self) -> Result<u64, LoadProfileError> {
        self.samples.iter().map(|s| s.latency_us).min().ok_or(LoadProfileError::NoData)
    }

    /// Maximum latency in microseconds.
    pub fn max_us(&self) -> Result<u64, LoadProfileError> {
        self.samples.iter().map(|s| s.latency_us).max().ok_or(LoadProfileError::NoData)
    }

    /// Compute throughput (requests per second) over the test duration.
    pub fn throughput_rps(&self) -> Result<f64, LoadProfileError> {
        if self.samples.is_empty() {
            return Err(LoadProfileError::NoData);
        }
        let min_ts = self.samples.iter().map(|s| s.timestamp_ms).min().unwrap_or(0);
        let max_ts = self.samples.iter().map(|s| s.timestamp_ms).max().unwrap_or(0);
        let duration_secs = (max_ts - min_ts) as f64 / 1000.0;
        if duration_secs <= 0.0 {
            return Ok(self.samples.len() as f64);
        }
        Ok(self.samples.len() as f64 / duration_secs)
    }

    /// Error rate (0.0 to 1.0).
    pub fn error_rate(&self) -> Result<f64, LoadProfileError> {
        if self.samples.is_empty() {
            return Err(LoadProfileError::NoData);
        }
        Ok(self.failure_count() as f64 / self.samples.len() as f64)
    }

    /// Generate a summary of all statistics.
    pub fn summary(&self) -> Result<LatencySummary, LoadProfileError> {
        Ok(LatencySummary {
            count: self.count(),
            success_count: self.success_count(),
            failure_count: self.failure_count(),
            mean_us: self.mean_us()?,
            min_us: self.min_us()?,
            max_us: self.max_us()?,
            p50_us: self.percentile(50)?,
            p90_us: self.percentile(90)?,
            p95_us: self.percentile(95)?,
            p99_us: self.percentile(99)?,
            throughput_rps: self.throughput_rps()?,
            error_rate: self.error_rate()?,
        })
    }

    /// Get all labels.
    pub fn labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self.labels.keys().cloned().collect();
        labels.sort();
        labels
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.labels.clear();
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a load test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: usize,
    pub success_count: usize,
    pub failure_count: usize,
    pub mean_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub p50_us: u64,
    pub p90_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub throughput_rps: f64,
    pub error_rate: f64,
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ts: u64, lat: u64, status: u16) -> LatencySample {
        LatencySample {
            timestamp_ms: ts,
            latency_us: lat,
            status,
            success: (200..300).contains(&status),
        }
    }

    #[test]
    fn test_stage_ramp_up() {
        let s = LoadStage::ramp_up("warmup", 30, 100);
        assert_eq!(s.duration_secs, 30);
        assert_eq!(s.target_vus, 100);
        assert_eq!(s.shape, StageShape::Linear);
    }

    #[test]
    fn test_stage_constant() {
        let s = LoadStage::constant("steady", 60, 50);
        assert_eq!(s.shape, StageShape::Constant);
        assert_eq!(s.target_vus, 50);
    }

    #[test]
    fn test_stage_step() {
        let s = LoadStage::step("burst", 10, 200);
        assert_eq!(s.shape, StageShape::Step);
    }

    #[test]
    fn test_profile_total_duration() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::ramp_up("a", 10, 50));
        p.add_stage(LoadStage::constant("b", 30, 50));
        p.add_stage(LoadStage::ramp_up("c", 10, 0));
        assert_eq!(p.total_duration_secs(), 50);
    }

    #[test]
    fn test_profile_peak_vus() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::ramp_up("a", 10, 50));
        p.add_stage(LoadStage::constant("b", 30, 100));
        p.add_stage(LoadStage::ramp_up("c", 10, 0));
        assert_eq!(p.peak_vus(), 100);
    }

    #[test]
    fn test_profile_vus_at_linear_ramp() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::ramp_up("ramp", 100, 100));
        // At 0% through the ramp, should be ~0 (starting from 0)
        assert_eq!(p.vus_at(0), 0);
        // At 50% through, should be ~50
        assert_eq!(p.vus_at(50), 50);
        // At 100%, should be 100
        assert_eq!(p.vus_at(100), 100);
    }

    #[test]
    fn test_profile_vus_at_step() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::step("burst", 10, 200));
        // Step immediately jumps to target
        assert_eq!(p.vus_at(0), 200);
        assert_eq!(p.vus_at(5), 200);
    }

    #[test]
    fn test_profile_vus_at_constant() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::constant("hold", 60, 75));
        assert_eq!(p.vus_at(0), 75);
        assert_eq!(p.vus_at(30), 75);
        assert_eq!(p.vus_at(59), 75);
    }

    #[test]
    fn test_profile_vus_at_multi_stage() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::ramp_up("up", 10, 100));
        p.add_stage(LoadStage::constant("hold", 20, 100));
        p.add_stage(LoadStage::ramp_up("down", 10, 0));
        // During ramp up
        assert_eq!(p.vus_at(5), 50);
        // During hold
        assert_eq!(p.vus_at(15), 100);
        assert_eq!(p.vus_at(29), 100);
        // During ramp down
        assert_eq!(p.vus_at(35), 50);
    }

    #[test]
    fn test_profile_vus_after_all_stages() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::constant("hold", 10, 50));
        // After stages end, returns last target
        assert_eq!(p.vus_at(100), 50);
    }

    #[test]
    fn test_profile_timeline() {
        let mut p = LoadProfile::new("test");
        p.add_stage(LoadStage::constant("hold", 5, 10));
        let timeline = p.timeline();
        assert_eq!(timeline.len(), 6); // 0..=5
        assert!(timeline.iter().all(|&(_, vus)| vus == 10));
    }

    #[test]
    fn test_profile_tags() {
        let p = LoadProfile::new("test")
            .with_tag("env", "staging")
            .with_tag("build", "123");
        assert_eq!(p.tags.get("env").unwrap(), "staging");
    }

    #[test]
    fn test_profile_empty() {
        let p = LoadProfile::new("empty");
        assert_eq!(p.total_duration_secs(), 0);
        assert_eq!(p.peak_vus(), 0);
    }

    #[test]
    fn test_tracker_empty() {
        let t = LatencyTracker::new();
        assert_eq!(t.count(), 0);
        assert!(t.mean_us().is_err());
        assert!(t.percentile(50).is_err());
    }

    #[test]
    fn test_tracker_record_and_count() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 1000, 200));
        t.record(sample(100, 2000, 200));
        t.record(sample(200, 3000, 500));
        assert_eq!(t.count(), 3);
        assert_eq!(t.success_count(), 2);
        assert_eq!(t.failure_count(), 1);
    }

    #[test]
    fn test_tracker_mean() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 1000, 200));
        t.record(sample(100, 3000, 200));
        let mean = t.mean_us().unwrap();
        assert!((mean - 2000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_min_max() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 500, 200));
        t.record(sample(100, 1500, 200));
        t.record(sample(200, 1000, 200));
        assert_eq!(t.min_us().unwrap(), 500);
        assert_eq!(t.max_us().unwrap(), 1500);
    }

    #[test]
    fn test_tracker_percentile() {
        let mut t = LatencyTracker::new();
        for i in 1..=100 {
            t.record(sample(i * 10, i * 100, 200));
        }
        let p50 = t.percentile(50).unwrap();
        // p50 of 100..=10000 step 100 should be ~5000
        assert!(p50 >= 4900 && p50 <= 5100);
        let p99 = t.percentile(99).unwrap();
        assert!(p99 >= 9800);
    }

    #[test]
    fn test_tracker_percentile_invalid() {
        let t = LatencyTracker::new();
        assert!(matches!(t.percentile(101), Err(LoadProfileError::InvalidPercentile(101))));
    }

    #[test]
    fn test_tracker_throughput() {
        let mut t = LatencyTracker::new();
        // 10 requests over 1 second = 10 rps
        for i in 0..10 {
            t.record(sample(i * 100, 500, 200));
        }
        let rps = t.throughput_rps().unwrap();
        assert!(rps > 9.0 && rps < 12.0);
    }

    #[test]
    fn test_tracker_error_rate() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 100, 200));
        t.record(sample(100, 100, 200));
        t.record(sample(200, 100, 500));
        t.record(sample(300, 100, 503));
        let rate = t.error_rate().unwrap();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_labeled() {
        let mut t = LatencyTracker::new();
        t.record_labeled("GET /api", sample(0, 100, 200));
        t.record_labeled("GET /api", sample(100, 200, 200));
        t.record_labeled("POST /api", sample(200, 300, 201));
        assert_eq!(t.labels(), vec!["GET /api", "POST /api"]);
        let p50 = t.percentile_for("GET /api", 50).unwrap();
        assert!(p50 >= 100 && p50 <= 200);
    }

    #[test]
    fn test_tracker_percentile_for_unknown_label() {
        let t = LatencyTracker::new();
        assert!(t.percentile_for("unknown", 50).is_err());
    }

    #[test]
    fn test_tracker_summary() {
        let mut t = LatencyTracker::new();
        for i in 1..=100 {
            t.record(sample(i * 10, i * 100, if i <= 95 { 200 } else { 500 }));
        }
        let summary = t.summary().unwrap();
        assert_eq!(summary.count, 100);
        assert_eq!(summary.success_count, 95);
        assert_eq!(summary.failure_count, 5);
        assert!(summary.p50_us > 0);
        assert!(summary.p95_us > summary.p50_us);
        assert!(summary.p99_us >= summary.p95_us);
    }

    #[test]
    fn test_tracker_clear() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 100, 200));
        t.record_labeled("x", sample(100, 200, 200));
        t.clear();
        assert_eq!(t.count(), 0);
        assert!(t.labels().is_empty());
    }

    #[test]
    fn test_error_display() {
        assert!(format!("{}", LoadProfileError::NoData).contains("no data"));
        assert!(format!("{}", LoadProfileError::InvalidPercentile(150)).contains("150"));
    }

    #[test]
    fn test_single_sample_stats() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 500, 200));
        assert_eq!(t.min_us().unwrap(), 500);
        assert_eq!(t.max_us().unwrap(), 500);
        assert!((t.mean_us().unwrap() - 500.0).abs() < f64::EPSILON);
        assert_eq!(t.percentile(0).unwrap(), 500);
        assert_eq!(t.percentile(100).unwrap(), 500);
    }

    #[test]
    fn test_throughput_single_instant() {
        let mut t = LatencyTracker::new();
        t.record(sample(0, 100, 200));
        // Single request at t=0 => special case
        let rps = t.throughput_rps().unwrap();
        assert_eq!(rps, 1.0);
    }
}
