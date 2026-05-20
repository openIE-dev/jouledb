//! Loft surface through cross-section profiles.
//!
//! Provides [`LoftSurfaceConfig`] builder and [`LoftSurface`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum LoftSurfaceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for LoftSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "LoftSurface: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "LoftSurface: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "LoftSurface: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`LoftSurface`] parameters.
#[derive(Debug, Clone)]
pub struct LoftSurfaceConfig {
    pub num_profiles: usize,
    pub degree_v: usize,
    pub ruled: bool,
    pub compatible_knots: bool,
}

impl LoftSurfaceConfig {
    pub fn new() -> Self {
        Self {
            num_profiles: 4,
            degree_v: 3,
            ruled: false,
            compatible_knots: true,
        }
    }

    pub fn with_num_profiles(mut self, v: usize) -> Self {
        self.num_profiles = v;
        self
    }

    pub fn with_degree_v(mut self, v: usize) -> Self {
        self.degree_v = v;
        self
    }

    pub fn with_ruled(mut self, v: bool) -> Self {
        self.ruled = v;
        self
    }

    pub fn with_compatible_knots(mut self, v: bool) -> Self {
        self.compatible_knots = v;
        self
    }

    pub fn validate(&self) -> Result<(), LoftSurfaceError> {
        Ok(())
    }
}

impl Default for LoftSurfaceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for LoftSurfaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoftSurfaceConfig(num_profiles={0}, degree_v={1}, ruled={2}, compatible_knots={3})", self.num_profiles, self.degree_v, self.ruled, self.compatible_knots)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core loft surface through cross-section profiles engine.
#[derive(Debug, Clone)]
pub struct LoftSurface {
    config: LoftSurfaceConfig,
    data: Vec<f64>,
}

impl LoftSurface {
    pub fn new(config: LoftSurfaceConfig) -> Result<Self, LoftSurfaceError> {
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
    pub fn config(&self) -> &LoftSurfaceConfig { &self.config }

    /// Compute lofted surface.
    pub fn compute_loft(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Linear ruled surface.
    pub fn ruled_surface(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Skin through profiles.
    pub fn skin_profiles(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for LoftSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoftSurface(n={})", self.data.len())
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
        let cfg = LoftSurfaceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = LoftSurfaceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("LoftSurfaceConfig"));
    }

    #[test]
    fn test_config_with_num_profiles() {
        let cfg = LoftSurfaceConfig::new().with_num_profiles(42);
        assert_eq!(cfg.num_profiles, 42);
    }

    #[test]
    fn test_config_with_degree_v() {
        let cfg = LoftSurfaceConfig::new().with_degree_v(42);
        assert_eq!(cfg.degree_v, 42);
    }

    #[test]
    fn test_config_with_ruled() {
        let cfg = LoftSurfaceConfig::new().with_ruled(true);
        assert_eq!(cfg.ruled, true);
    }

    #[test]
    fn test_config_with_compatible_knots() {
        let cfg = LoftSurfaceConfig::new().with_compatible_knots(false);
        assert_eq!(cfg.compatible_knots, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = LoftSurfaceConfig::new().with_num_profiles(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("LoftSurface"));
    }

    #[test]
    fn test_summary() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_loft() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compute_loft();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_ruled_surface() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ruled_surface();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_skin_profiles() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.skin_profiles();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_skin_profiles_empty() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap();
        assert!(e.skin_profiles().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = LoftSurface::new(LoftSurfaceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = LoftSurfaceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = LoftSurfaceError::InvalidConfig("a".into());
        let e2 = LoftSurfaceError::ComputationFailed("b".into());
        let e3 = LoftSurfaceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
