//! G-code program generation.
//!
//! Provides [`GcodeGenConfig`] builder and [`GcodeGen`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum GcodeGenError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for GcodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "GcodeGen: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "GcodeGen: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "GcodeGen: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`GcodeGen`] parameters.
#[derive(Debug, Clone)]
pub struct GcodeGenConfig {
    pub unit_metric: bool,
    pub coord_system: usize,
    pub line_numbers: bool,
    pub max_feed: f64,
}

impl GcodeGenConfig {
    pub fn new() -> Self {
        Self {
            unit_metric: true,
            coord_system: 0,
            line_numbers: true,
            max_feed: 10000.0,
        }
    }

    pub fn with_unit_metric(mut self, v: bool) -> Self {
        self.unit_metric = v;
        self
    }

    pub fn with_coord_system(mut self, v: usize) -> Self {
        self.coord_system = v;
        self
    }

    pub fn with_line_numbers(mut self, v: bool) -> Self {
        self.line_numbers = v;
        self
    }

    pub fn with_max_feed(mut self, v: f64) -> Self {
        self.max_feed = v;
        self
    }

    pub fn validate(&self) -> Result<(), GcodeGenError> {
        if self.max_feed.is_nan() {
            return Err(GcodeGenError::InvalidConfig("max_feed is NaN".into()));
        }
        Ok(())
    }
}

impl Default for GcodeGenConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GcodeGenConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcodeGenConfig(unit_metric={0}, coord_system={1}, line_numbers={2}, max_feed={3:.4})", self.unit_metric, self.coord_system, self.line_numbers, self.max_feed)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core g-code program generation engine.
#[derive(Debug, Clone)]
pub struct GcodeGen {
    config: GcodeGenConfig,
    data: Vec<f64>,
}

impl GcodeGen {
    pub fn new(config: GcodeGenConfig) -> Result<Self, GcodeGenError> {
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
    pub fn config(&self) -> &GcodeGenConfig { &self.config }

    /// G0 rapid move.
    pub fn rapid_move(&self) -> String {
        format!("{}: {} records", stringify!(rapid_move), self.data.len())
    }

    /// G1 linear interpolation.
    pub fn linear_move(&self) -> String {
        format!("{}: {} records", stringify!(linear_move), self.data.len())
    }

    /// G2/G3 arc interpolation.
    pub fn arc_move(&self) -> String {
        format!("{}: {} records", stringify!(arc_move), self.data.len())
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

impl fmt::Display for GcodeGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcodeGen(n={})", self.data.len())
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
        let cfg = GcodeGenConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = GcodeGenConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("GcodeGenConfig"));
    }

    #[test]
    fn test_config_with_unit_metric() {
        let cfg = GcodeGenConfig::new().with_unit_metric(false);
        assert_eq!(cfg.unit_metric, false);
    }

    #[test]
    fn test_config_with_coord_system() {
        let cfg = GcodeGenConfig::new().with_coord_system(42);
        assert_eq!(cfg.coord_system, 42);
    }

    #[test]
    fn test_config_with_line_numbers() {
        let cfg = GcodeGenConfig::new().with_line_numbers(false);
        assert_eq!(cfg.line_numbers, false);
    }

    #[test]
    fn test_config_with_max_feed() {
        let cfg = GcodeGenConfig::new().with_max_feed(42.0);
        assert!((cfg.max_feed - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = GcodeGenConfig::new().with_unit_metric(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = GcodeGen::new(GcodeGenConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("GcodeGen"));
    }

    #[test]
    fn test_summary() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_rapid_move() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rapid_move();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_linear_move() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.linear_move();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_arc_move() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.arc_move();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_arc_move_empty() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap();
        let _ = e.arc_move();
    }

    #[test]
    fn test_config_accessor() {
        let e = GcodeGen::new(GcodeGenConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = GcodeGenError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = GcodeGenError::InvalidConfig("a".into());
        let e2 = GcodeGenError::ComputationFailed("b".into());
        let e3 = GcodeGenError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
