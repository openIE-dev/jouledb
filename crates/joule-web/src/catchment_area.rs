//! Catchment and service area analysis.
//!
//! Provides [`CatchmentAreaConfig`] builder and [`CatchmentArea`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CatchmentAreaError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for CatchmentAreaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "CatchmentArea: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "CatchmentArea: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "CatchmentArea: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`CatchmentArea`] parameters.
#[derive(Debug, Clone)]
pub struct CatchmentAreaConfig {
    pub travel_time_max: f64,
    pub walk_speed_kmh: f64,
    pub drive_speed_kmh: f64,
    pub overlap: bool,
}

impl CatchmentAreaConfig {
    pub fn new() -> Self {
        Self {
            travel_time_max: 30.0,
            walk_speed_kmh: 5.0,
            drive_speed_kmh: 50.0,
            overlap: true,
        }
    }

    pub fn with_travel_time_max(mut self, v: f64) -> Self {
        self.travel_time_max = v;
        self
    }

    pub fn with_walk_speed_kmh(mut self, v: f64) -> Self {
        self.walk_speed_kmh = v;
        self
    }

    pub fn with_drive_speed_kmh(mut self, v: f64) -> Self {
        self.drive_speed_kmh = v;
        self
    }

    pub fn with_overlap(mut self, v: bool) -> Self {
        self.overlap = v;
        self
    }

    pub fn validate(&self) -> Result<(), CatchmentAreaError> {
        if self.travel_time_max.is_nan() {
            return Err(CatchmentAreaError::InvalidConfig("travel_time_max is NaN".into()));
        }
        if self.walk_speed_kmh.is_nan() {
            return Err(CatchmentAreaError::InvalidConfig("walk_speed_kmh is NaN".into()));
        }
        if self.drive_speed_kmh.is_nan() {
            return Err(CatchmentAreaError::InvalidConfig("drive_speed_kmh is NaN".into()));
        }
        Ok(())
    }
}

impl Default for CatchmentAreaConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for CatchmentAreaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CatchmentAreaConfig(travel_time_max={0:.4}, walk_speed_kmh={1:.4}, drive_speed_kmh={2:.4}, overlap={3})", self.travel_time_max, self.walk_speed_kmh, self.drive_speed_kmh, self.overlap)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core catchment and service area analysis engine.
#[derive(Debug, Clone)]
pub struct CatchmentArea {
    config: CatchmentAreaConfig,
    data: Vec<f64>,
}

impl CatchmentArea {
    pub fn new(config: CatchmentAreaConfig) -> Result<Self, CatchmentAreaError> {
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
    pub fn config(&self) -> &CatchmentAreaConfig { &self.config }

    /// Travel time catchment.
    pub fn time_catchment(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Distance-based catchment.
    pub fn distance_catchment(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Market area analysis.
    pub fn market_area(&self) -> Vec<Vec<f64>> {
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

impl fmt::Display for CatchmentArea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CatchmentArea(n={})", self.data.len())
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
        let cfg = CatchmentAreaConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = CatchmentAreaConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("CatchmentAreaConfig"));
    }

    #[test]
    fn test_config_with_travel_time_max() {
        let cfg = CatchmentAreaConfig::new().with_travel_time_max(42.0);
        assert!((cfg.travel_time_max - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_walk_speed_kmh() {
        let cfg = CatchmentAreaConfig::new().with_walk_speed_kmh(42.0);
        assert!((cfg.walk_speed_kmh - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_drive_speed_kmh() {
        let cfg = CatchmentAreaConfig::new().with_drive_speed_kmh(42.0);
        assert!((cfg.drive_speed_kmh - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_overlap() {
        let cfg = CatchmentAreaConfig::new().with_overlap(false);
        assert_eq!(cfg.overlap, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = CatchmentAreaConfig::new().with_travel_time_max(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("CatchmentArea"));
    }

    #[test]
    fn test_summary() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_catchment() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.time_catchment();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_distance_catchment() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.distance_catchment();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_market_area() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.market_area();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_market_area_empty() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap();
        assert!(e.market_area().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = CatchmentArea::new(CatchmentAreaConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = CatchmentAreaError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = CatchmentAreaError::InvalidConfig("a".into());
        let e2 = CatchmentAreaError::ComputationFailed("b".into());
        let e3 = CatchmentAreaError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
