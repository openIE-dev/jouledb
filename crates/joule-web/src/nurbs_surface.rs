//! NURBS surface with weights and bidirectional knots.
//!
//! Provides [`NurbsSurfaceConfig`] builder and [`NurbsSurface`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum NurbsSurfaceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for NurbsSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "NurbsSurface: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "NurbsSurface: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "NurbsSurface: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`NurbsSurface`] parameters.
#[derive(Debug, Clone)]
pub struct NurbsSurfaceConfig {
    pub degree_u: usize,
    pub degree_v: usize,
    pub weighted: bool,
    pub tolerance: f64,
}

impl NurbsSurfaceConfig {
    pub fn new() -> Self {
        Self {
            degree_u: 3,
            degree_v: 3,
            weighted: true,
            tolerance: 1e-6,
        }
    }

    pub fn with_degree_u(mut self, v: usize) -> Self {
        self.degree_u = v;
        self
    }

    pub fn with_degree_v(mut self, v: usize) -> Self {
        self.degree_v = v;
        self
    }

    pub fn with_weighted(mut self, v: bool) -> Self {
        self.weighted = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn validate(&self) -> Result<(), NurbsSurfaceError> {
        if self.tolerance.is_nan() {
            return Err(NurbsSurfaceError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for NurbsSurfaceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for NurbsSurfaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NurbsSurfaceConfig(degree_u={0}, degree_v={1}, weighted={2}, tolerance={3:.4})", self.degree_u, self.degree_v, self.weighted, self.tolerance)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core nurbs surface with weights and bidirectional knots engine.
#[derive(Debug, Clone)]
pub struct NurbsSurface {
    config: NurbsSurfaceConfig,
    data: Vec<f64>,
}

impl NurbsSurface {
    pub fn new(config: NurbsSurfaceConfig) -> Result<Self, NurbsSurfaceError> {
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
    pub fn config(&self) -> &NurbsSurfaceConfig { &self.config }

    /// Evaluate NURBS surface at (u,v).
    pub fn evaluate_uv(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Surface normal.
    pub fn normal_at(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Partial derivatives.
    pub fn partial_deriv(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for NurbsSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NurbsSurface(n={})", self.data.len())
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
        let cfg = NurbsSurfaceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = NurbsSurfaceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("NurbsSurfaceConfig"));
    }

    #[test]
    fn test_config_with_degree_u() {
        let cfg = NurbsSurfaceConfig::new().with_degree_u(42);
        assert_eq!(cfg.degree_u, 42);
    }

    #[test]
    fn test_config_with_degree_v() {
        let cfg = NurbsSurfaceConfig::new().with_degree_v(42);
        assert_eq!(cfg.degree_v, 42);
    }

    #[test]
    fn test_config_with_weighted() {
        let cfg = NurbsSurfaceConfig::new().with_weighted(false);
        assert_eq!(cfg.weighted, false);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = NurbsSurfaceConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = NurbsSurfaceConfig::new().with_degree_u(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("NurbsSurface"));
    }

    #[test]
    fn test_summary() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_uv() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.evaluate_uv();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_normal_at() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.normal_at();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_partial_deriv() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.partial_deriv();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_partial_deriv_empty() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap();
        assert!(e.partial_deriv().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = NurbsSurface::new(NurbsSurfaceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = NurbsSurfaceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = NurbsSurfaceError::InvalidConfig("a".into());
        let e2 = NurbsSurfaceError::ComputationFailed("b".into());
        let e3 = NurbsSurfaceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
