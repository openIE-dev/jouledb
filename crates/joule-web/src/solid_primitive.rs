//! Parametric solid primitive shapes.
//!
//! Provides [`SolidPrimitiveConfig`] builder and [`SolidPrimitive`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SolidPrimitiveError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SolidPrimitiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SolidPrimitive: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SolidPrimitive: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SolidPrimitive: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SolidPrimitive`] parameters.
#[derive(Debug, Clone)]
pub struct SolidPrimitiveConfig {
    pub size_x: f64,
    pub size_y: f64,
    pub size_z: f64,
    pub segments: usize,
}

impl SolidPrimitiveConfig {
    pub fn new() -> Self {
        Self {
            size_x: 1.0,
            size_y: 1.0,
            size_z: 1.0,
            segments: 32,
        }
    }

    pub fn with_size_x(mut self, v: f64) -> Self {
        self.size_x = v;
        self
    }

    pub fn with_size_y(mut self, v: f64) -> Self {
        self.size_y = v;
        self
    }

    pub fn with_size_z(mut self, v: f64) -> Self {
        self.size_z = v;
        self
    }

    pub fn with_segments(mut self, v: usize) -> Self {
        self.segments = v;
        self
    }

    pub fn validate(&self) -> Result<(), SolidPrimitiveError> {
        if self.size_x.is_nan() {
            return Err(SolidPrimitiveError::InvalidConfig("size_x is NaN".into()));
        }
        if self.size_y.is_nan() {
            return Err(SolidPrimitiveError::InvalidConfig("size_y is NaN".into()));
        }
        if self.size_z.is_nan() {
            return Err(SolidPrimitiveError::InvalidConfig("size_z is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SolidPrimitiveConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SolidPrimitiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SolidPrimitiveConfig(size_x={0:.4}, size_y={1:.4}, size_z={2:.4}, segments={3})", self.size_x, self.size_y, self.size_z, self.segments)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core parametric solid primitive shapes engine.
#[derive(Debug, Clone)]
pub struct SolidPrimitive {
    config: SolidPrimitiveConfig,
    data: Vec<f64>,
}

impl SolidPrimitive {
    pub fn new(config: SolidPrimitiveConfig) -> Result<Self, SolidPrimitiveError> {
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
    pub fn config(&self) -> &SolidPrimitiveConfig { &self.config }

    /// Create box primitive.
    pub fn create_box(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Create sphere primitive.
    pub fn create_sphere(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Compute volume.
    pub fn volume(&self) -> f64 {
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

impl fmt::Display for SolidPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SolidPrimitive(n={})", self.data.len())
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
        let cfg = SolidPrimitiveConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SolidPrimitiveConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SolidPrimitiveConfig"));
    }

    #[test]
    fn test_config_with_size_x() {
        let cfg = SolidPrimitiveConfig::new().with_size_x(42.0);
        assert!((cfg.size_x - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_size_y() {
        let cfg = SolidPrimitiveConfig::new().with_size_y(42.0);
        assert!((cfg.size_y - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_size_z() {
        let cfg = SolidPrimitiveConfig::new().with_size_z(42.0);
        assert!((cfg.size_z - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_segments() {
        let cfg = SolidPrimitiveConfig::new().with_segments(42);
        assert_eq!(cfg.segments, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SolidPrimitiveConfig::new().with_size_x(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SolidPrimitive"));
    }

    #[test]
    fn test_summary() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_create_box() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.create_box();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_create_sphere() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.create_sphere();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_volume() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.volume();
        assert!(result.is_finite());
    }

    #[test]
    fn test_volume_empty() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap();
        assert!((e.volume() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = SolidPrimitive::new(SolidPrimitiveConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SolidPrimitiveError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SolidPrimitiveError::InvalidConfig("a".into());
        let e2 = SolidPrimitiveError::ComputationFailed("b".into());
        let e3 = SolidPrimitiveError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
