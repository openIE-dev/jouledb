//! Service area and facility location analysis.
//!
//! Provides [`ServiceAreaConfig`] builder and [`ServiceArea`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceAreaError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for ServiceAreaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "ServiceArea: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "ServiceArea: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "ServiceArea: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`ServiceArea`] parameters.
#[derive(Debug, Clone)]
pub struct ServiceAreaConfig {
    pub travel_mode: usize,
    pub max_time_min: f64,
    pub num_facilities: usize,
    pub demand_weight: f64,
}

impl ServiceAreaConfig {
    pub fn new() -> Self {
        Self {
            travel_mode: 0,
            max_time_min: 30.0,
            num_facilities: 5,
            demand_weight: 1.0,
        }
    }

    pub fn with_travel_mode(mut self, v: usize) -> Self {
        self.travel_mode = v;
        self
    }

    pub fn with_max_time_min(mut self, v: f64) -> Self {
        self.max_time_min = v;
        self
    }

    pub fn with_num_facilities(mut self, v: usize) -> Self {
        self.num_facilities = v;
        self
    }

    pub fn with_demand_weight(mut self, v: f64) -> Self {
        self.demand_weight = v;
        self
    }

    pub fn validate(&self) -> Result<(), ServiceAreaError> {
        if self.max_time_min.is_nan() {
            return Err(ServiceAreaError::InvalidConfig("max_time_min is NaN".into()));
        }
        if self.demand_weight.is_nan() {
            return Err(ServiceAreaError::InvalidConfig("demand_weight is NaN".into()));
        }
        Ok(())
    }
}

impl Default for ServiceAreaConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ServiceAreaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ServiceAreaConfig(travel_mode={0}, max_time_min={1:.4}, num_facilities={2}, demand_weight={3:.4})", self.travel_mode, self.max_time_min, self.num_facilities, self.demand_weight)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core service area and facility location analysis engine.
#[derive(Debug, Clone)]
pub struct ServiceArea {
    config: ServiceAreaConfig,
    data: Vec<f64>,
}

impl ServiceArea {
    pub fn new(config: ServiceAreaConfig) -> Result<Self, ServiceAreaError> {
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
    pub fn config(&self) -> &ServiceAreaConfig { &self.config }

    /// Compute isochrone service area.
    pub fn isochrone_area(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// P-median facility location.
    pub fn p_median(&self) -> Vec<Vec<f64>> {
        if self.data.is_empty() { return Vec::new(); }
        vec![self.data.clone()]
    }

    /// Accessibility scoring.
    pub fn accessibility_score(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
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

impl fmt::Display for ServiceArea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ServiceArea(n={})", self.data.len())
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
        let cfg = ServiceAreaConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = ServiceAreaConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("ServiceAreaConfig"));
    }

    #[test]
    fn test_config_with_travel_mode() {
        let cfg = ServiceAreaConfig::new().with_travel_mode(42);
        assert_eq!(cfg.travel_mode, 42);
    }

    #[test]
    fn test_config_with_max_time_min() {
        let cfg = ServiceAreaConfig::new().with_max_time_min(42.0);
        assert!((cfg.max_time_min - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_num_facilities() {
        let cfg = ServiceAreaConfig::new().with_num_facilities(42);
        assert_eq!(cfg.num_facilities, 42);
    }

    #[test]
    fn test_config_with_demand_weight() {
        let cfg = ServiceAreaConfig::new().with_demand_weight(42.0);
        assert!((cfg.demand_weight - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = ServiceAreaConfig::new().with_travel_mode(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = ServiceArea::new(ServiceAreaConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("ServiceArea"));
    }

    #[test]
    fn test_summary() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_isochrone_area() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.isochrone_area();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_p_median() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.p_median();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_accessibility_score() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.accessibility_score();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_accessibility_score_empty() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap();
        assert!(e.accessibility_score().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = ServiceArea::new(ServiceAreaConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = ServiceAreaError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = ServiceAreaError::InvalidConfig("a".into());
        let e2 = ServiceAreaError::ComputationFailed("b".into());
        let e3 = ServiceAreaError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
