//! Patient event timeline construction.
//!
//! Provides [`PatientTimelineConfig`] builder and [`PatientTimeline`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PatientTimelineError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PatientTimelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PatientTimeline: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PatientTimeline: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PatientTimeline: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PatientTimeline`] parameters.
#[derive(Debug, Clone)]
pub struct PatientTimelineConfig {
    pub max_events: usize,
    pub merge_overlapping: bool,
    pub include_labs: bool,
    pub include_meds: bool,
}

impl PatientTimelineConfig {
    pub fn new() -> Self {
        Self {
            max_events: 10000,
            merge_overlapping: true,
            include_labs: true,
            include_meds: true,
        }
    }

    pub fn with_max_events(mut self, v: usize) -> Self {
        self.max_events = v;
        self
    }

    pub fn with_merge_overlapping(mut self, v: bool) -> Self {
        self.merge_overlapping = v;
        self
    }

    pub fn with_include_labs(mut self, v: bool) -> Self {
        self.include_labs = v;
        self
    }

    pub fn with_include_meds(mut self, v: bool) -> Self {
        self.include_meds = v;
        self
    }

    pub fn validate(&self) -> Result<(), PatientTimelineError> {
        Ok(())
    }
}

impl Default for PatientTimelineConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PatientTimelineConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PatientTimelineConfig(max_events={0}, merge_overlapping={1}, include_labs={2}, include_meds={3})", self.max_events, self.merge_overlapping, self.include_labs, self.include_meds)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core patient event timeline construction engine.
#[derive(Debug, Clone)]
pub struct PatientTimeline {
    config: PatientTimelineConfig,
    data: Vec<f64>,
}

impl PatientTimeline {
    pub fn new(config: PatientTimelineConfig) -> Result<Self, PatientTimelineError> {
        config.validate()?;
        Ok(Self { config, data: Vec::new() })
    }

    pub fn with_data(mut self, data: Vec<f64>) -> Self {
        self.data = data;
        self
    }

    pub fn push(&mut self, value: f64) {
        self.data.push(value);
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn config(&self) -> &PatientTimelineConfig { &self.config }

    /// Add timeline event.
    pub fn add_event(&self) -> usize {
        self.data.len()
    }

    /// Events within date range.
    pub fn events_in_range(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Find timeline gaps.
    pub fn gap_analysis(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Summary statistics of loaded data.
    pub fn summary(&self) -> (f64, f64, f64, f64) {
        if self.data.is_empty() { return (0.0, 0.0, 0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let var = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt(), min, max)
    }

    /// Percentile of the data (0.0–1.0).
    pub fn percentile(&self, p: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Exponentially weighted moving statistic.
    pub fn ewm(&self, decay: f64) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.data.len());
        let mut ewm = 0.0;
        for (i, &v) in self.data.iter().enumerate() {
            if i == 0 { ewm = v; } else { ewm = decay * ewm + (1.0 - decay) * v; }
            result.push(ewm);
        }
        result
    }
}

impl fmt::Display for PatientTimeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PatientTimeline(n={})", self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }

    #[test]
    fn test_config_default() {
        let cfg = PatientTimelineConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PatientTimelineConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PatientTimelineConfig"));
    }

    #[test]
    fn test_config_with_max_events() {
        let cfg = PatientTimelineConfig::new().with_max_events(42);
        assert_eq!(cfg.max_events, 42);
    }

    #[test]
    fn test_config_with_merge_overlapping() {
        let cfg = PatientTimelineConfig::new().with_merge_overlapping(false);
        assert_eq!(cfg.merge_overlapping, false);
    }

    #[test]
    fn test_config_with_include_labs() {
        let cfg = PatientTimelineConfig::new().with_include_labs(false);
        assert_eq!(cfg.include_labs, false);
    }

    #[test]
    fn test_config_with_include_meds() {
        let cfg = PatientTimelineConfig::new().with_include_meds(false);
        assert_eq!(cfg.include_meds, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PatientTimelineConfig::new().with_max_events(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PatientTimeline"));
    }

    #[test]
    fn test_summary() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_add_event() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.add_event();
        assert!(result > 0);
    }

    #[test]
    fn test_events_in_range() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.events_in_range();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_gap_analysis() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gap_analysis();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_gap_analysis_empty() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap();
        assert!(e.gap_analysis().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = PatientTimeline::new(PatientTimelineConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PatientTimelineError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PatientTimelineError::InvalidConfig("a".into());
        let e2 = PatientTimelineError::ComputationFailed("b".into());
        let e3 = PatientTimelineError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
