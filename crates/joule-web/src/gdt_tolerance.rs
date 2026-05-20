//! GD&T geometric dimensioning and tolerancing.
//!
//! Provides [`GdtToleranceConfig`] builder and [`GdtTolerance`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum GdtToleranceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for GdtToleranceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "GdtTolerance: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "GdtTolerance: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "GdtTolerance: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`GdtTolerance`] parameters.
#[derive(Debug, Clone)]
pub struct GdtToleranceConfig {
    pub tolerance_value: f64,
    pub tolerance_type: usize,
    pub modifier: usize,
    pub datum_refs: usize,
}

impl GdtToleranceConfig {
    pub fn new() -> Self {
        Self {
            tolerance_value: 0.1,
            tolerance_type: 0,
            modifier: 0,
            datum_refs: 0,
        }
    }

    pub fn with_tolerance_value(mut self, v: f64) -> Self {
        self.tolerance_value = v;
        self
    }

    pub fn with_tolerance_type(mut self, v: usize) -> Self {
        self.tolerance_type = v;
        self
    }

    pub fn with_modifier(mut self, v: usize) -> Self {
        self.modifier = v;
        self
    }

    pub fn with_datum_refs(mut self, v: usize) -> Self {
        self.datum_refs = v;
        self
    }

    pub fn validate(&self) -> Result<(), GdtToleranceError> {
        if self.tolerance_value.is_nan() {
            return Err(GdtToleranceError::InvalidConfig("tolerance_value is NaN".into()));
        }
        Ok(())
    }
}

impl Default for GdtToleranceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GdtToleranceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GdtToleranceConfig(tolerance_value={0:.4}, tolerance_type={1}, modifier={2}, datum_refs={3})", self.tolerance_value, self.tolerance_type, self.modifier, self.datum_refs)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core gd&t geometric dimensioning and tolerancing engine.
#[derive(Debug, Clone)]
pub struct GdtTolerance {
    config: GdtToleranceConfig,
    data: Vec<f64>,
}

impl GdtTolerance {
    pub fn new(config: GdtToleranceConfig) -> Result<Self, GdtToleranceError> {
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
    pub fn config(&self) -> &GdtToleranceConfig { &self.config }

    /// Evaluate flatness tolerance.
    pub fn evaluate_flatness(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Evaluate cylindricity.
    pub fn evaluate_cylindricity(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Compute tolerance zone.
    pub fn tolerance_zone(&self) -> Vec<f64> {
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

impl fmt::Display for GdtTolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GdtTolerance(n={})", self.data.len())
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
        let cfg = GdtToleranceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = GdtToleranceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("GdtToleranceConfig"));
    }

    #[test]
    fn test_config_with_tolerance_value() {
        let cfg = GdtToleranceConfig::new().with_tolerance_value(42.0);
        assert!((cfg.tolerance_value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_tolerance_type() {
        let cfg = GdtToleranceConfig::new().with_tolerance_type(42);
        assert_eq!(cfg.tolerance_type, 42);
    }

    #[test]
    fn test_config_with_modifier() {
        let cfg = GdtToleranceConfig::new().with_modifier(42);
        assert_eq!(cfg.modifier, 42);
    }

    #[test]
    fn test_config_with_datum_refs() {
        let cfg = GdtToleranceConfig::new().with_datum_refs(42);
        assert_eq!(cfg.datum_refs, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = GdtToleranceConfig::new().with_tolerance_value(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("GdtTolerance"));
    }

    #[test]
    fn test_summary() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_flatness() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate_flatness();
        assert!(result.is_finite());
    }

    #[test]
    fn test_evaluate_cylindricity() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate_cylindricity();
        assert!(result.is_finite());
    }

    #[test]
    fn test_tolerance_zone() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tolerance_zone();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tolerance_zone_empty() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap();
        assert!(e.tolerance_zone().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = GdtTolerance::new(GdtToleranceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = GdtToleranceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = GdtToleranceError::InvalidConfig("a".into());
        let e2 = GdtToleranceError::ComputationFailed("b".into());
        let e3 = GdtToleranceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
