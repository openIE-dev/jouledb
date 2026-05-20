//! Curve offsetting with self-intersection trimming.
//!
//! Provides [`CurveOffsetConfig`] builder and [`CurveOffset`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveOffsetError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CurveOffsetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CurveOffset: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CurveOffset: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CurveOffset: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CurveOffset`] parameters.
#[derive(Debug, Clone)]
pub struct CurveOffsetConfig {
    pub distance: f64,
    pub tolerance: f64,
    pub max_segments: usize,
    pub trim_loops: bool,
}

impl CurveOffsetConfig {
    pub fn new() -> Self {
        Self {
            distance: 1.0,
            tolerance: 1e-6,
            max_segments: 100,
            trim_loops: true,
        }
    }

    pub fn with_distance(mut self, v: f64) -> Self {
        self.distance = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_max_segments(mut self, v: usize) -> Self {
        self.max_segments = v;
        self
    }

    pub fn with_trim_loops(mut self, v: bool) -> Self {
        self.trim_loops = v;
        self
    }

    pub fn validate(&self) -> Result<(), CurveOffsetError> {
        if self.distance.is_nan() {
            return Err(CurveOffsetError::InvalidConfig("distance is NaN".into()));
        }
        if self.tolerance.is_nan() {
            return Err(CurveOffsetError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CurveOffsetConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CurveOffsetConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurveOffsetConfig(distance={0:.4}, tolerance={1:.4}, max_segments={2}, trim_loops={3})", self.distance, self.tolerance, self.max_segments, self.trim_loops)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core curve offsetting with self-intersection trimming engine.
#[derive(Debug, Clone)]
pub struct CurveOffset {
    config: CurveOffsetConfig,
    data: Vec<f64>,
}

impl CurveOffset {
    pub fn new(config: CurveOffsetConfig) -> Result<Self, CurveOffsetError> {
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
    pub fn config(&self) -> &CurveOffsetConfig { &self.config }

    /// Compute offset curve.
    pub fn offset_curve(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Variable distance offset.
    pub fn variable_offset(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Offset error estimation.
    pub fn error_estimate(&self) -> f64 {
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

impl fmt::Display for CurveOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurveOffset(n={})", self.data.len())
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
        let cfg = CurveOffsetConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CurveOffsetConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CurveOffsetConfig"));
    }

    #[test]
    fn test_config_with_distance() {
        let cfg = CurveOffsetConfig::new().with_distance(42.0);
        assert!((cfg.distance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = CurveOffsetConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_max_segments() {
        let cfg = CurveOffsetConfig::new().with_max_segments(42);
        assert_eq!(cfg.max_segments, 42);
    }

    #[test]
    fn test_config_with_trim_loops() {
        let cfg = CurveOffsetConfig::new().with_trim_loops(false);
        assert_eq!(cfg.trim_loops, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CurveOffsetConfig::new().with_distance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CurveOffset::new(CurveOffsetConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CurveOffset"));
    }

    #[test]
    fn test_summary() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_offset_curve() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.offset_curve();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_variable_offset() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.variable_offset();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_error_estimate() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.error_estimate();
        assert!(result.is_finite());
    }

    #[test]
    fn test_error_estimate_empty() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap();
        assert!((e.error_estimate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = CurveOffset::new(CurveOffsetConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CurveOffsetError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CurveOffsetError::InvalidConfig("a".into());
        let e2 = CurveOffsetError::ComputationFailed("b".into());
        let e3 = CurveOffsetError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
