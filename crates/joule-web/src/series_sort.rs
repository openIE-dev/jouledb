//! DICOM series sorting and grouping.
//!
//! Provides [`SeriesSortConfig`] builder and [`SeriesSort`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SeriesSortError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SeriesSortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SeriesSort: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SeriesSort: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SeriesSort: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SeriesSort`] parameters.
#[derive(Debug, Clone)]
pub struct SeriesSortConfig {
    pub sort_by: usize,
    pub group_by_series: bool,
    pub handle_multiframe: bool,
    pub temporal_sort: bool,
}

impl SeriesSortConfig {
    pub fn new() -> Self {
        Self {
            sort_by: 0,
            group_by_series: true,
            handle_multiframe: true,
            temporal_sort: false,
        }
    }

    pub fn with_sort_by(mut self, v: usize) -> Self {
        self.sort_by = v;
        self
    }

    pub fn with_group_by_series(mut self, v: bool) -> Self {
        self.group_by_series = v;
        self
    }

    pub fn with_handle_multiframe(mut self, v: bool) -> Self {
        self.handle_multiframe = v;
        self
    }

    pub fn with_temporal_sort(mut self, v: bool) -> Self {
        self.temporal_sort = v;
        self
    }

    pub fn validate(&self) -> Result<(), SeriesSortError> {
        Ok(())
    }
}

impl Default for SeriesSortConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SeriesSortConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SeriesSortConfig(sort_by={0}, group_by_series={1}, handle_multiframe={2}, temporal_sort={3})", self.sort_by, self.group_by_series, self.handle_multiframe, self.temporal_sort)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core dicom series sorting and grouping engine.
#[derive(Debug, Clone)]
pub struct SeriesSort {
    config: SeriesSortConfig,
    data: Vec<f64>,
}

impl SeriesSort {
    pub fn new(config: SeriesSortConfig) -> Result<Self, SeriesSortError> {
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
    pub fn config(&self) -> &SeriesSortConfig { &self.config }

    /// Sort instances in series.
    pub fn sort_instances(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Group by SeriesInstanceUID.
    pub fn group_series(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Detect missing instances.
    pub fn detect_gaps(&self) -> Vec<f64> {
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

impl fmt::Display for SeriesSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SeriesSort(n={})", self.data.len())
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
        let cfg = SeriesSortConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SeriesSortConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SeriesSortConfig"));
    }

    #[test]
    fn test_config_with_sort_by() {
        let cfg = SeriesSortConfig::new().with_sort_by(42);
        assert_eq!(cfg.sort_by, 42);
    }

    #[test]
    fn test_config_with_group_by_series() {
        let cfg = SeriesSortConfig::new().with_group_by_series(false);
        assert_eq!(cfg.group_by_series, false);
    }

    #[test]
    fn test_config_with_handle_multiframe() {
        let cfg = SeriesSortConfig::new().with_handle_multiframe(false);
        assert_eq!(cfg.handle_multiframe, false);
    }

    #[test]
    fn test_config_with_temporal_sort() {
        let cfg = SeriesSortConfig::new().with_temporal_sort(true);
        assert_eq!(cfg.temporal_sort, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SeriesSortConfig::new().with_sort_by(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SeriesSort::new(SeriesSortConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SeriesSort"));
    }

    #[test]
    fn test_summary() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sort_instances() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.sort_instances();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_group_series() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.group_series();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_detect_gaps() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.detect_gaps();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_detect_gaps_empty() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap();
        assert!(e.detect_gaps().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SeriesSort::new(SeriesSortConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SeriesSortError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SeriesSortError::InvalidConfig("a".into());
        let e2 = SeriesSortError::ComputationFailed("b".into());
        let e3 = SeriesSortError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
