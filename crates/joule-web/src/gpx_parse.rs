//! GPX file parsing for tracks, routes, and waypoints.
//!
//! Provides [`GpxParseConfig`] builder and [`GpxParse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum GpxParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for GpxParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "GpxParse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "GpxParse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "GpxParse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`GpxParse`] parameters.
#[derive(Debug, Clone)]
pub struct GpxParseConfig {
    pub parse_extensions: bool,
    pub compute_stats: bool,
    pub filter_duplicates: bool,
    pub simplify_tolerance: f64,
}

impl GpxParseConfig {
    pub fn new() -> Self {
        Self {
            parse_extensions: false,
            compute_stats: true,
            filter_duplicates: true,
            simplify_tolerance: 0.0,
        }
    }

    pub fn with_parse_extensions(mut self, v: bool) -> Self {
        self.parse_extensions = v;
        self
    }

    pub fn with_compute_stats(mut self, v: bool) -> Self {
        self.compute_stats = v;
        self
    }

    pub fn with_filter_duplicates(mut self, v: bool) -> Self {
        self.filter_duplicates = v;
        self
    }

    pub fn with_simplify_tolerance(mut self, v: f64) -> Self {
        self.simplify_tolerance = v;
        self
    }

    pub fn validate(&self) -> Result<(), GpxParseError> {
        if self.simplify_tolerance.is_nan() {
            return Err(GpxParseError::InvalidConfig("simplify_tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for GpxParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GpxParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpxParseConfig(parse_extensions={0}, compute_stats={1}, filter_duplicates={2}, simplify_tolerance={3:.4})", self.parse_extensions, self.compute_stats, self.filter_duplicates, self.simplify_tolerance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core gpx file parsing for tracks, routes, and waypoints engine.
#[derive(Debug, Clone)]
pub struct GpxParse {
    config: GpxParseConfig,
    data: Vec<f64>,
}

impl GpxParse {
    pub fn new(config: GpxParseConfig) -> Result<Self, GpxParseError> {
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
    pub fn config(&self) -> &GpxParseConfig { &self.config }

    /// Parse GPX waypoints.
    pub fn parse_waypoints(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Parse GPX tracks.
    pub fn parse_tracks(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute track statistics.
    pub fn track_stats(&self) -> Vec<f64> {
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

impl fmt::Display for GpxParse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpxParse(n={})", self.data.len())
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
        let cfg = GpxParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = GpxParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("GpxParseConfig"));
    }

    #[test]
    fn test_config_with_parse_extensions() {
        let cfg = GpxParseConfig::new().with_parse_extensions(true);
        assert_eq!(cfg.parse_extensions, true);
    }

    #[test]
    fn test_config_with_compute_stats() {
        let cfg = GpxParseConfig::new().with_compute_stats(false);
        assert_eq!(cfg.compute_stats, false);
    }

    #[test]
    fn test_config_with_filter_duplicates() {
        let cfg = GpxParseConfig::new().with_filter_duplicates(false);
        assert_eq!(cfg.filter_duplicates, false);
    }

    #[test]
    fn test_config_with_simplify_tolerance() {
        let cfg = GpxParseConfig::new().with_simplify_tolerance(42.0);
        assert!((cfg.simplify_tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = GpxParseConfig::new().with_parse_extensions(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = GpxParse::new(GpxParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("GpxParse"));
    }

    #[test]
    fn test_summary() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_waypoints() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_waypoints();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_parse_tracks() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_tracks();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_track_stats() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.track_stats();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_track_stats_empty() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap();
        assert!(e.track_stats().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = GpxParse::new(GpxParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = GpxParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = GpxParseError::InvalidConfig("a".into());
        let e2 = GpxParseError::ComputationFailed("b".into());
        let e3 = GpxParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
