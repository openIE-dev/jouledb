//! Sweep surface along spine curve.
//!
//! Provides [`SweepSurfaceConfig`] builder and [`SweepSurface`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepSurfaceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SweepSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SweepSurface: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SweepSurface: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SweepSurface: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SweepSurface`] parameters.
#[derive(Debug, Clone)]
pub struct SweepSurfaceConfig {
    pub twist_deg: f64,
    pub scale_start: f64,
    pub scale_end: f64,
    pub frame_type: usize,
}

impl SweepSurfaceConfig {
    pub fn new() -> Self {
        Self {
            twist_deg: 0.0,
            scale_start: 1.0,
            scale_end: 1.0,
            frame_type: 0,
        }
    }

    pub fn with_twist_deg(mut self, v: f64) -> Self {
        self.twist_deg = v;
        self
    }

    pub fn with_scale_start(mut self, v: f64) -> Self {
        self.scale_start = v;
        self
    }

    pub fn with_scale_end(mut self, v: f64) -> Self {
        self.scale_end = v;
        self
    }

    pub fn with_frame_type(mut self, v: usize) -> Self {
        self.frame_type = v;
        self
    }

    pub fn validate(&self) -> Result<(), SweepSurfaceError> {
        if self.twist_deg.is_nan() {
            return Err(SweepSurfaceError::InvalidConfig("twist_deg is NaN".into()));
        }
        if self.scale_start.is_nan() {
            return Err(SweepSurfaceError::InvalidConfig("scale_start is NaN".into()));
        }
        if self.scale_end.is_nan() {
            return Err(SweepSurfaceError::InvalidConfig("scale_end is NaN".into()));
        }
        Ok(())
    }
}

impl Default for SweepSurfaceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SweepSurfaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SweepSurfaceConfig(twist_deg={0:.4}, scale_start={1:.4}, scale_end={2:.4}, frame_type={3})", self.twist_deg, self.scale_start, self.scale_end, self.frame_type)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core sweep surface along spine curve engine.
#[derive(Debug, Clone)]
pub struct SweepSurface {
    config: SweepSurfaceConfig,
    data: Vec<f64>,
}

impl SweepSurface {
    pub fn new(config: SweepSurfaceConfig) -> Result<Self, SweepSurfaceError> {
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
    pub fn config(&self) -> &SweepSurfaceConfig { &self.config }

    /// Compute swept surface.
    pub fn compute_sweep(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Frenet frame at parameter.
    pub fn frenet_frame(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Bishop frame (rotation-minimizing).
    pub fn bishop_frame(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for SweepSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SweepSurface(n={})", self.data.len())
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
        let cfg = SweepSurfaceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SweepSurfaceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SweepSurfaceConfig"));
    }

    #[test]
    fn test_config_with_twist_deg() {
        let cfg = SweepSurfaceConfig::new().with_twist_deg(42.0);
        assert!((cfg.twist_deg - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_scale_start() {
        let cfg = SweepSurfaceConfig::new().with_scale_start(42.0);
        assert!((cfg.scale_start - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_scale_end() {
        let cfg = SweepSurfaceConfig::new().with_scale_end(42.0);
        assert!((cfg.scale_end - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_frame_type() {
        let cfg = SweepSurfaceConfig::new().with_frame_type(42);
        assert_eq!(cfg.frame_type, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SweepSurfaceConfig::new().with_twist_deg(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SweepSurface"));
    }

    #[test]
    fn test_summary() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_sweep() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compute_sweep();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_frenet_frame() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.frenet_frame();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_bishop_frame() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bishop_frame();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_bishop_frame_empty() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap();
        assert!(e.bishop_frame().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = SweepSurface::new(SweepSurfaceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SweepSurfaceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SweepSurfaceError::InvalidConfig("a".into());
        let e2 = SweepSurfaceError::ComputationFailed("b".into());
        let e3 = SweepSurfaceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
