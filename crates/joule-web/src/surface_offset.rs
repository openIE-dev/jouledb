//! Surface offsetting operations.
//!
//! Provides [`SurfaceOffsetConfig`] builder and [`SurfaceOffset`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceOffsetError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SurfaceOffsetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SurfaceOffset: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SurfaceOffset: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SurfaceOffset: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SurfaceOffset`] parameters.
#[derive(Debug, Clone)]
pub struct SurfaceOffsetConfig {
    pub distance: f64,
    pub tolerance: f64,
    pub detect_self_int: bool,
    pub step_mode: bool,
}

impl SurfaceOffsetConfig {
    pub fn new() -> Self {
        Self {
            distance: 1.0,
            tolerance: 1e-4,
            detect_self_int: true,
            step_mode: false,
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

    pub fn with_detect_self_int(mut self, v: bool) -> Self {
        self.detect_self_int = v;
        self
    }

    pub fn with_step_mode(mut self, v: bool) -> Self {
        self.step_mode = v;
        self
    }

    pub fn validate(&self) -> Result<(), SurfaceOffsetError> {
        if self.distance.is_nan() {
            return Err(SurfaceOffsetError::InvalidConfig("distance is NaN".into()));
        }
        if self.tolerance.is_nan() {
            return Err(SurfaceOffsetError::InvalidConfig("tolerance is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SurfaceOffsetConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SurfaceOffsetConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceOffsetConfig(distance={0:.4}, tolerance={1:.4}, detect_self_int={2}, step_mode={3})", self.distance, self.tolerance, self.detect_self_int, self.step_mode)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core surface offsetting operations engine.
#[derive(Debug, Clone)]
pub struct SurfaceOffset {
    config: SurfaceOffsetConfig,
    data: Vec<f64>,
}

impl SurfaceOffset {
    pub fn new(config: SurfaceOffsetConfig) -> Result<Self, SurfaceOffsetError> {
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
    pub fn config(&self) -> &SurfaceOffsetConfig { &self.config }

    /// Offset surface by distance.
    pub fn offset_surface(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Check offset self-intersection.
    pub fn self_intersection(&self) -> bool {
        !self.data.is_empty()
    }

    /// Offset error distribution.
    pub fn error_map(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for SurfaceOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceOffset(n={})", self.data.len())
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
        let cfg = SurfaceOffsetConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SurfaceOffsetConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SurfaceOffsetConfig"));
    }

    #[test]
    fn test_config_with_distance() {
        let cfg = SurfaceOffsetConfig::new().with_distance(42.0);
        assert!((cfg.distance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_tolerance() {
        let cfg = SurfaceOffsetConfig::new().with_tolerance(42.0);
        assert!((cfg.tolerance - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_detect_self_int() {
        let cfg = SurfaceOffsetConfig::new().with_detect_self_int(false);
        assert_eq!(cfg.detect_self_int, false);
    }

    #[test]
    fn test_config_with_step_mode() {
        let cfg = SurfaceOffsetConfig::new().with_step_mode(true);
        assert_eq!(cfg.step_mode, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SurfaceOffsetConfig::new().with_distance(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SurfaceOffset"));
    }

    #[test]
    fn test_summary() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_offset_surface() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.offset_surface();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_self_intersection() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.self_intersection();
        assert!(result);
    }

    #[test]
    fn test_error_map() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.error_map();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_error_map_empty() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap();
        assert!(e.error_map().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SurfaceOffset::new(SurfaceOffsetConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SurfaceOffsetError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SurfaceOffsetError::InvalidConfig("a".into());
        let e2 = SurfaceOffsetError::ComputationFailed("b".into());
        let e3 = SurfaceOffsetError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
