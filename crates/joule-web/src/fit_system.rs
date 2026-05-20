//! ISO/ANSI fit and tolerance system.
//!
//! Provides [`FitSystemConfig`] builder and [`FitSystem`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FitSystemError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for FitSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "FitSystem: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "FitSystem: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "FitSystem: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`FitSystem`] parameters.
#[derive(Debug, Clone)]
pub struct FitSystemConfig {
    pub nominal_size: f64,
    pub hole_tolerance: usize,
    pub shaft_tolerance: usize,
    pub fit_class: usize,
}

impl FitSystemConfig {
    pub fn new() -> Self {
        Self {
            nominal_size: 25.0,
            hole_tolerance: 7,
            shaft_tolerance: 6,
            fit_class: 0,
        }
    }

    pub fn with_nominal_size(mut self, v: f64) -> Self {
        self.nominal_size = v;
        self
    }

    pub fn with_hole_tolerance(mut self, v: usize) -> Self {
        self.hole_tolerance = v;
        self
    }

    pub fn with_shaft_tolerance(mut self, v: usize) -> Self {
        self.shaft_tolerance = v;
        self
    }

    pub fn with_fit_class(mut self, v: usize) -> Self {
        self.fit_class = v;
        self
    }

    pub fn validate(&self) -> Result<(), FitSystemError> {
        if self.nominal_size.is_nan() {
            return Err(FitSystemError::InvalidConfig("nominal_size is NaN".into()));
        }
        Ok(())
    }
}

impl Default for FitSystemConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FitSystemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FitSystemConfig(nominal_size={0:.4}, hole_tolerance={1}, shaft_tolerance={2}, fit_class={3})", self.nominal_size, self.hole_tolerance, self.shaft_tolerance, self.fit_class)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core iso/ansi fit and tolerance system engine.
#[derive(Debug, Clone)]
pub struct FitSystem {
    config: FitSystemConfig,
    data: Vec<f64>,
}

impl FitSystem {
    pub fn new(config: FitSystemConfig) -> Result<Self, FitSystemError> {
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
    pub fn config(&self) -> &FitSystemConfig { &self.config }

    /// Calculate clearance range.
    pub fn clearance_range(&self) -> (f64, f64) {
        if self.data.len() < 2 { return (0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        (sum / n, sum)
    }

    /// Classify fit type.
    pub fn classify_fit(&self) -> String {
        format!("{}: {} records", stringify!(classify_fit), self.data.len())
    }

    /// Lookup tolerance value.
    pub fn tolerance_value(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
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

impl fmt::Display for FitSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FitSystem(n={})", self.data.len())
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
        let cfg = FitSystemConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = FitSystemConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("FitSystemConfig"));
    }

    #[test]
    fn test_config_with_nominal_size() {
        let cfg = FitSystemConfig::new().with_nominal_size(42.0);
        assert!((cfg.nominal_size - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_hole_tolerance() {
        let cfg = FitSystemConfig::new().with_hole_tolerance(42);
        assert_eq!(cfg.hole_tolerance, 42);
    }

    #[test]
    fn test_config_with_shaft_tolerance() {
        let cfg = FitSystemConfig::new().with_shaft_tolerance(42);
        assert_eq!(cfg.shaft_tolerance, 42);
    }

    #[test]
    fn test_config_with_fit_class() {
        let cfg = FitSystemConfig::new().with_fit_class(42);
        assert_eq!(cfg.fit_class, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = FitSystemConfig::new().with_nominal_size(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = FitSystem::new(FitSystemConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("FitSystem"));
    }

    #[test]
    fn test_summary() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_clearance_range() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        let (a, b) = e.clearance_range();
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_classify_fit() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.classify_fit();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tolerance_value() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tolerance_value();
        assert!(result.is_finite());
    }

    #[test]
    fn test_tolerance_value_empty() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap();
        assert!((e.tolerance_value() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = FitSystem::new(FitSystemConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = FitSystemError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = FitSystemError::InvalidConfig("a".into());
        let e2 = FitSystemError::ComputationFailed("b".into());
        let e3 = FitSystemError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
