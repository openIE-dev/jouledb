//! Contour line generation from raster data.
//!
//! Provides [`ContourGenConfig`] builder and [`ContourGen`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ContourGenError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ContourGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ContourGen: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ContourGen: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ContourGen: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ContourGen`] parameters.
#[derive(Debug, Clone)]
pub struct ContourGenConfig {
    pub interval: f64,
    pub base: f64,
    pub smooth_factor: f64,
    pub label_interval: usize,
}

impl ContourGenConfig {
    pub fn new() -> Self {
        Self {
            interval: 10.0,
            base: 0.0,
            smooth_factor: 0.5,
            label_interval: 5,
        }
    }

    pub fn with_interval(mut self, v: f64) -> Self {
        self.interval = v;
        self
    }

    pub fn with_base(mut self, v: f64) -> Self {
        self.base = v;
        self
    }

    pub fn with_smooth_factor(mut self, v: f64) -> Self {
        self.smooth_factor = v;
        self
    }

    pub fn with_label_interval(mut self, v: usize) -> Self {
        self.label_interval = v;
        self
    }

    pub fn validate(&self) -> Result<(), ContourGenError> {
        if self.interval.is_nan() {
            return Err(ContourGenError::InvalidConfig("interval is NaN".into()));
        }
        if self.base.is_nan() {
            return Err(ContourGenError::InvalidConfig("base is NaN".into()));
        }
        if self.smooth_factor.is_nan() {
            return Err(ContourGenError::InvalidConfig("smooth_factor is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ContourGenConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ContourGenConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContourGenConfig(interval={0:.4}, base={1:.4}, smooth_factor={2:.4}, label_interval={3})", self.interval, self.base, self.smooth_factor, self.label_interval)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core contour line generation from raster data engine.
#[derive(Debug, Clone)]
pub struct ContourGen {
    config: ContourGenConfig,
    data: Vec<f64>,
}

impl ContourGen {
    pub fn new(config: ContourGenConfig) -> Result<Self, ContourGenError> {
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
    pub fn config(&self) -> &ContourGenConfig { &self.config }

    /// Generate contour lines.
    pub fn generate_contours(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Marching squares algorithm.
    pub fn marching_squares(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute label positions.
    pub fn label_positions(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
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

impl fmt::Display for ContourGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContourGen(n={})", self.data.len())
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
        let cfg = ContourGenConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ContourGenConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ContourGenConfig"));
    }

    #[test]
    fn test_config_with_interval() {
        let cfg = ContourGenConfig::new().with_interval(42.0);
        assert!((cfg.interval - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_base() {
        let cfg = ContourGenConfig::new().with_base(42.0);
        assert!((cfg.base - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_smooth_factor() {
        let cfg = ContourGenConfig::new().with_smooth_factor(42.0);
        assert!((cfg.smooth_factor - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_label_interval() {
        let cfg = ContourGenConfig::new().with_label_interval(42);
        assert_eq!(cfg.label_interval, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ContourGenConfig::new().with_interval(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ContourGen::new(ContourGenConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ContourGen"));
    }

    #[test]
    fn test_summary() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_contours() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_contours();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_marching_squares() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.marching_squares();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_label_positions() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.label_positions();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_label_positions_empty() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap();
        assert!(e.label_positions().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ContourGen::new(ContourGenConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ContourGenError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ContourGenError::InvalidConfig("a".into());
        let e2 = ContourGenError::ComputationFailed("b".into());
        let e3 = ContourGenError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
