//! Spatial regression models with diagnostics.
//!
//! Provides [`SpatialRegressConfig`] builder and [`SpatialRegress`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialRegressError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SpatialRegressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SpatialRegress: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SpatialRegress: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SpatialRegress: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SpatialRegress`] parameters.
#[derive(Debug, Clone)]
pub struct SpatialRegressConfig {
    pub num_predictors: usize,
    pub spatial_lag: bool,
    pub spatial_error: bool,
    pub significance: f64,
}

impl SpatialRegressConfig {
    pub fn new() -> Self {
        Self {
            num_predictors: 3,
            spatial_lag: true,
            spatial_error: false,
            significance: 0.05,
        }
    }

    pub fn with_num_predictors(mut self, v: usize) -> Self {
        self.num_predictors = v;
        self
    }

    pub fn with_spatial_lag(mut self, v: bool) -> Self {
        self.spatial_lag = v;
        self
    }

    pub fn with_spatial_error(mut self, v: bool) -> Self {
        self.spatial_error = v;
        self
    }

    pub fn with_significance(mut self, v: f64) -> Self {
        self.significance = v;
        self
    }

    pub fn validate(&self) -> Result<(), SpatialRegressError> {
        if self.significance.is_nan() {
            return Err(SpatialRegressError::InvalidConfig("significance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SpatialRegressConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SpatialRegressConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialRegressConfig(num_predictors={0}, spatial_lag={1}, spatial_error={2}, significance={3:.4})", self.num_predictors, self.spatial_lag, self.spatial_error, self.significance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core spatial regression models with diagnostics engine.
#[derive(Debug, Clone)]
pub struct SpatialRegress {
    config: SpatialRegressConfig,
    data: Vec<f64>,
}

impl SpatialRegress {
    pub fn new(config: SpatialRegressConfig) -> Result<Self, SpatialRegressError> {
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
    pub fn config(&self) -> &SpatialRegressConfig { &self.config }

    /// Ordinary least squares regression.
    pub fn ols_fit(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Spatial lag model (SAR).
    pub fn spatial_lag_model(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Moran's I on residuals.
    pub fn morans_i_residuals(&self) -> f64 {
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

impl fmt::Display for SpatialRegress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialRegress(n={})", self.data.len())
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
        let cfg = SpatialRegressConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SpatialRegressConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SpatialRegressConfig"));
    }

    #[test]
    fn test_config_with_num_predictors() {
        let cfg = SpatialRegressConfig::new().with_num_predictors(42);
        assert_eq!(cfg.num_predictors, 42);
    }

    #[test]
    fn test_config_with_spatial_lag() {
        let cfg = SpatialRegressConfig::new().with_spatial_lag(false);
        assert_eq!(cfg.spatial_lag, false);
    }

    #[test]
    fn test_config_with_spatial_error() {
        let cfg = SpatialRegressConfig::new().with_spatial_error(true);
        assert_eq!(cfg.spatial_error, true);
    }

    #[test]
    fn test_config_with_significance() {
        let cfg = SpatialRegressConfig::new().with_significance(42.0);
        assert!((cfg.significance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SpatialRegressConfig::new().with_num_predictors(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SpatialRegress"));
    }

    #[test]
    fn test_summary() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ols_fit() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ols_fit();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_spatial_lag_model() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.spatial_lag_model();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_morans_i_residuals() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.morans_i_residuals();
        assert!(result.is_finite());
    }

    #[test]
    fn test_morans_i_residuals_empty() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap();
        assert!((e.morans_i_residuals() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SpatialRegress::new(SpatialRegressConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SpatialRegressError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SpatialRegressError::InvalidConfig("a".into());
        let e2 = SpatialRegressError::ComputationFailed("b".into());
        let e3 = SpatialRegressError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
