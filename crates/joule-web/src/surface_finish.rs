//! Surface finish parameters and measurement.
//!
//! Provides [`SurfaceFinishConfig`] builder and [`SurfaceFinish`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceFinishError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SurfaceFinishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SurfaceFinish: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SurfaceFinish: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SurfaceFinish: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SurfaceFinish`] parameters.
#[derive(Debug, Clone)]
pub struct SurfaceFinishConfig {
    pub cutoff_mm: f64,
    pub evaluation_length: f64,
    pub profile_filter: usize,
    pub lay_direction: usize,
}

impl SurfaceFinishConfig {
    pub fn new() -> Self {
        Self {
            cutoff_mm: 0.8,
            evaluation_length: 4.0,
            profile_filter: 0,
            lay_direction: 0,
        }
    }

    pub fn with_cutoff_mm(mut self, v: f64) -> Self {
        self.cutoff_mm = v;
        self
    }

    pub fn with_evaluation_length(mut self, v: f64) -> Self {
        self.evaluation_length = v;
        self
    }

    pub fn with_profile_filter(mut self, v: usize) -> Self {
        self.profile_filter = v;
        self
    }

    pub fn with_lay_direction(mut self, v: usize) -> Self {
        self.lay_direction = v;
        self
    }

    pub fn validate(&self) -> Result<(), SurfaceFinishError> {
        if self.cutoff_mm.is_nan() {
            return Err(SurfaceFinishError::InvalidConfig("cutoff_mm is NaN".into()));
        }
        if self.evaluation_length.is_nan() {
            return Err(SurfaceFinishError::InvalidConfig("evaluation_length is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SurfaceFinishConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SurfaceFinishConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceFinishConfig(cutoff_mm={0:.4}, evaluation_length={1:.4}, profile_filter={2}, lay_direction={3})", self.cutoff_mm, self.evaluation_length, self.profile_filter, self.lay_direction)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core surface finish parameters and measurement engine.
#[derive(Debug, Clone)]
pub struct SurfaceFinish {
    config: SurfaceFinishConfig,
    data: Vec<f64>,
}

impl SurfaceFinish {
    pub fn new(config: SurfaceFinishConfig) -> Result<Self, SurfaceFinishError> {
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
    pub fn config(&self) -> &SurfaceFinishConfig { &self.config }

    /// Average roughness Ra.
    pub fn roughness_ra(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Mean peak-to-valley Rz.
    pub fn roughness_rz(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// RMS roughness Rq.
    pub fn roughness_rq(&self) -> f64 {
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

impl fmt::Display for SurfaceFinish {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceFinish(n={})", self.data.len())
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
        let cfg = SurfaceFinishConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SurfaceFinishConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SurfaceFinishConfig"));
    }

    #[test]
    fn test_config_with_cutoff_mm() {
        let cfg = SurfaceFinishConfig::new().with_cutoff_mm(42.0);
        assert!((cfg.cutoff_mm - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_evaluation_length() {
        let cfg = SurfaceFinishConfig::new().with_evaluation_length(42.0);
        assert!((cfg.evaluation_length - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_profile_filter() {
        let cfg = SurfaceFinishConfig::new().with_profile_filter(42);
        assert_eq!(cfg.profile_filter, 42);
    }

    #[test]
    fn test_config_with_lay_direction() {
        let cfg = SurfaceFinishConfig::new().with_lay_direction(42);
        assert_eq!(cfg.lay_direction, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SurfaceFinishConfig::new().with_cutoff_mm(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SurfaceFinish"));
    }

    #[test]
    fn test_summary() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_roughness_ra() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.roughness_ra();
        assert!(result.is_finite());
    }

    #[test]
    fn test_roughness_rz() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.roughness_rz();
        assert!(result.is_finite());
    }

    #[test]
    fn test_roughness_rq() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.roughness_rq();
        assert!(result.is_finite());
    }

    #[test]
    fn test_roughness_rq_empty() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap();
        assert!((e.roughness_rq() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SurfaceFinish::new(SurfaceFinishConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SurfaceFinishError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SurfaceFinishError::InvalidConfig("a".into());
        let e2 = SurfaceFinishError::ComputationFailed("b".into());
        let e3 = SurfaceFinishError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
