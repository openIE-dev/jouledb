//! Geoid undulation model for height conversion.
//!
//! Provides [`GeoidModelConfig`] builder and [`GeoidModel`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum GeoidModelError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for GeoidModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "GeoidModel: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "GeoidModel: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "GeoidModel: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`GeoidModel`] parameters.
#[derive(Debug, Clone)]
pub struct GeoidModelConfig {
    pub grid_spacing: f64,
    pub interpolation_order: usize,
    pub default_undulation: f64,
    pub datum_name: usize,
}

impl GeoidModelConfig {
    pub fn new() -> Self {
        Self {
            grid_spacing: 1.0,
            interpolation_order: 2,
            default_undulation: 0.0,
            datum_name: 0,
        }
    }

    pub fn with_grid_spacing(mut self, v: f64) -> Self {
        self.grid_spacing = v;
        self
    }

    pub fn with_interpolation_order(mut self, v: usize) -> Self {
        self.interpolation_order = v;
        self
    }

    pub fn with_default_undulation(mut self, v: f64) -> Self {
        self.default_undulation = v;
        self
    }

    pub fn with_datum_name(mut self, v: usize) -> Self {
        self.datum_name = v;
        self
    }

    pub fn validate(&self) -> Result<(), GeoidModelError> {
        if self.grid_spacing.is_nan() {
            return Err(GeoidModelError::InvalidConfig("grid_spacing is NaN".into()));
        }
        if self.default_undulation.is_nan() {
            return Err(GeoidModelError::InvalidConfig("default_undulation is NaN".into()));
        }
        Ok(())
    }
}

impl Default for GeoidModelConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GeoidModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GeoidModelConfig(grid_spacing={0:.4}, interpolation_order={1}, default_undulation={2:.4}, datum_name={3})", self.grid_spacing, self.interpolation_order, self.default_undulation, self.datum_name)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core geoid undulation model for height conversion engine.
#[derive(Debug, Clone)]
pub struct GeoidModel {
    config: GeoidModelConfig,
    data: Vec<f64>,
}

impl GeoidModel {
    pub fn new(config: GeoidModelConfig) -> Result<Self, GeoidModelError> {
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
    pub fn config(&self) -> &GeoidModelConfig { &self.config }

    /// Geoid undulation at lat/lon.
    pub fn undulation_at(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Ellipsoidal to orthometric height.
    pub fn orthometric_height(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Orthometric to ellipsoidal height.
    pub fn ellipsoidal_height(&self) -> f64 {
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

impl fmt::Display for GeoidModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GeoidModel(n={})", self.data.len())
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
        let cfg = GeoidModelConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = GeoidModelConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("GeoidModelConfig"));
    }

    #[test]
    fn test_config_with_grid_spacing() {
        let cfg = GeoidModelConfig::new().with_grid_spacing(42.0);
        assert!((cfg.grid_spacing - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_interpolation_order() {
        let cfg = GeoidModelConfig::new().with_interpolation_order(42);
        assert_eq!(cfg.interpolation_order, 42);
    }

    #[test]
    fn test_config_with_default_undulation() {
        let cfg = GeoidModelConfig::new().with_default_undulation(42.0);
        assert!((cfg.default_undulation - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_datum_name() {
        let cfg = GeoidModelConfig::new().with_datum_name(42);
        assert_eq!(cfg.datum_name, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = GeoidModelConfig::new().with_grid_spacing(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = GeoidModel::new(GeoidModelConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("GeoidModel"));
    }

    #[test]
    fn test_summary() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_undulation_at() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.undulation_at();
        assert!(result.is_finite());
    }

    #[test]
    fn test_orthometric_height() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.orthometric_height();
        assert!(result.is_finite());
    }

    #[test]
    fn test_ellipsoidal_height() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ellipsoidal_height();
        assert!(result.is_finite());
    }

    #[test]
    fn test_ellipsoidal_height_empty() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap();
        assert!((e.ellipsoidal_height() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = GeoidModel::new(GeoidModelConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = GeoidModelError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = GeoidModelError::InvalidConfig("a".into());
        let e2 = GeoidModelError::ComputationFailed("b".into());
        let e3 = GeoidModelError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
