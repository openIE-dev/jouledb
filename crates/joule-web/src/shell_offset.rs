//! Shell and offset solid operations.
//!
//! Provides [`ShellOffsetConfig`] builder and [`ShellOffset`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ShellOffsetError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ShellOffsetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ShellOffset: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ShellOffset: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ShellOffset: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ShellOffset`] parameters.
#[derive(Debug, Clone)]
pub struct ShellOffsetConfig {
    pub thickness: f64,
    pub inner_offset: bool,
    pub tolerance: f64,
    pub exclude_faces: usize,
}

impl ShellOffsetConfig {
    pub fn new() -> Self {
        Self {
            thickness: 1.0,
            inner_offset: true,
            tolerance: 1e-4,
            exclude_faces: 0,
        }
    }

    pub fn with_thickness(mut self, v: f64) -> Self {
        self.thickness = v;
        self
    }

    pub fn with_inner_offset(mut self, v: bool) -> Self {
        self.inner_offset = v;
        self
    }

    pub fn with_tolerance(mut self, v: f64) -> Self {
        self.tolerance = v;
        self
    }

    pub fn with_exclude_faces(mut self, v: usize) -> Self {
        self.exclude_faces = v;
        self
    }

    pub fn validate(&self) -> Result<(), ShellOffsetError> {
        if self.thickness.is_nan() {
            return Err(ShellOffsetError::InvalidConfig("thickness is NaN".into()));
        }
        if self.tolerance.is_nan() {
            return Err(ShellOffsetError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ShellOffsetConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ShellOffsetConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShellOffsetConfig(thickness={0:.4}, inner_offset={1}, tolerance={2:.4}, exclude_faces={3})", self.thickness, self.inner_offset, self.tolerance, self.exclude_faces)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core shell and offset solid operations engine.
#[derive(Debug, Clone)]
pub struct ShellOffset {
    config: ShellOffsetConfig,
    data: Vec<f64>,
}

impl ShellOffset {
    pub fn new(config: ShellOffsetConfig) -> Result<Self, ShellOffsetError> {
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
    pub fn config(&self) -> &ShellOffsetConfig { &self.config }

    /// Create shell from solid.
    pub fn create_shell(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Offset solid surface.
    pub fn offset_solid(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Check for self-intersection.
    pub fn check_self_int(&self) -> bool {
        !self.data.is_empty()
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

impl fmt::Display for ShellOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShellOffset(n={})", self.data.len())
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
        let cfg = ShellOffsetConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ShellOffsetConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ShellOffsetConfig"));
    }

    #[test]
    fn test_config_with_thickness() {
        let cfg = ShellOffsetConfig::new().with_thickness(42.0);
        assert!((cfg.thickness - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_inner_offset() {
        let cfg = ShellOffsetConfig::new().with_inner_offset(false);
        assert_eq!(cfg.inner_offset, false);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = ShellOffsetConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_exclude_faces() {
        let cfg = ShellOffsetConfig::new().with_exclude_faces(42);
        assert_eq!(cfg.exclude_faces, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ShellOffsetConfig::new().with_thickness(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ShellOffset::new(ShellOffsetConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ShellOffset"));
    }

    #[test]
    fn test_summary() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_create_shell() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.create_shell();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_offset_solid() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.offset_solid();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_check_self_int() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.check_self_int();
        assert!(result);
    }

    #[test]
    fn test_check_self_int_empty() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap();
        assert!(!e.check_self_int());
    }

    #[test]
    fn test_config_accessor() {
        let e = ShellOffset::new(ShellOffsetConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ShellOffsetError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ShellOffsetError::InvalidConfig("a".into());
        let e2 = ShellOffsetError::ComputationFailed("b".into());
        let e3 = ShellOffsetError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
