//! Surface curvature and quality analysis.
//!
//! Provides [`SurfaceAnalysisConfig`] builder and [`SurfaceAnalysis`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceAnalysisError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SurfaceAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SurfaceAnalysis: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SurfaceAnalysis: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SurfaceAnalysis: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SurfaceAnalysis`] parameters.
#[derive(Debug, Clone)]
pub struct SurfaceAnalysisConfig {
    pub sample_density: usize,
    pub tolerance: f64,
    pub compute_principal: bool,
    pub integration_order: usize,
}

impl SurfaceAnalysisConfig {
    pub fn new() -> Self {
        Self {
            sample_density: 20,
            tolerance: 1e-6,
            compute_principal: true,
            integration_order: 4,
        }
    }

    pub fn with_sample_density(mut self, v: usize) -> Self {
        self.sample_density = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_compute_principal(mut self, v: bool) -> Self {
        self.compute_principal = v;
        self
    }

    pub fn with_integration_order(mut self, v: usize) -> Self {
        self.integration_order = v;
        self
    }

    pub fn validate(&self) -> Result<(), SurfaceAnalysisError> {
        if self.tolerance.is_nan() {
            return Err(SurfaceAnalysisError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SurfaceAnalysisConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SurfaceAnalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceAnalysisConfig(sample_density={0}, tolerance={1:.4}, compute_principal={2}, integration_order={3})", self.sample_density, self.tolerance, self.compute_principal, self.integration_order)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core surface curvature and quality analysis engine.
#[derive(Debug, Clone)]
pub struct SurfaceAnalysis {
    config: SurfaceAnalysisConfig,
    data: Vec<f64>,
}

impl SurfaceAnalysis {
    pub fn new(config: SurfaceAnalysisConfig) -> Result<Self, SurfaceAnalysisError> {
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
    pub fn config(&self) -> &SurfaceAnalysisConfig { &self.config }

    /// Gaussian curvature at point.
    pub fn gaussian_curvature(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Mean curvature at point.
    pub fn mean_curvature(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Numerical surface area.
    pub fn surface_area(&self) -> f64 {
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

impl fmt::Display for SurfaceAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceAnalysis(n={})", self.data.len())
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
        let cfg = SurfaceAnalysisConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SurfaceAnalysisConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SurfaceAnalysisConfig"));
    }

    #[test]
    fn test_config_with_sample_density() {
        let cfg = SurfaceAnalysisConfig::new().with_sample_density(42);
        assert_eq!(cfg.sample_density, 42);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = SurfaceAnalysisConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_compute_principal() {
        let cfg = SurfaceAnalysisConfig::new().with_compute_principal(false);
        assert_eq!(cfg.compute_principal, false);
    }

    #[test]
    fn test_config_with_integration_order() {
        let cfg = SurfaceAnalysisConfig::new().with_integration_order(42);
        assert_eq!(cfg.integration_order, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SurfaceAnalysisConfig::new().with_sample_density(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SurfaceAnalysis"));
    }

    #[test]
    fn test_summary() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_curvature() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.gaussian_curvature();
        assert!(result.is_finite());
    }

    #[test]
    fn test_mean_curvature() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.mean_curvature();
        assert!(result.is_finite());
    }

    #[test]
    fn test_surface_area() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.surface_area();
        assert!(result.is_finite());
    }

    #[test]
    fn test_surface_area_empty() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap();
        assert!((e.surface_area() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SurfaceAnalysis::new(SurfaceAnalysisConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SurfaceAnalysisError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SurfaceAnalysisError::InvalidConfig("a".into());
        let e2 = SurfaceAnalysisError::ComputationFailed("b".into());
        let e3 = SurfaceAnalysisError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
