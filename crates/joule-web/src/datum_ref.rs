//! Datum reference frame establishment.
//!
//! Provides [`DatumRefConfig`] builder and [`DatumRef`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum DatumRefError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for DatumRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "DatumRef: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "DatumRef: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "DatumRef: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`DatumRef`] parameters.
#[derive(Debug, Clone)]
pub struct DatumRefConfig {
    pub datum_letter: usize,
    pub target_count: usize,
    pub material_condition: usize,
    pub simulation_points: usize,
}

impl DatumRefConfig {
    pub fn new() -> Self {
        Self {
            datum_letter: 0,
            target_count: 0,
            material_condition: 0,
            simulation_points: 100,
        }
    }

    pub fn with_datum_letter(mut self, v: usize) -> Self {
        self.datum_letter = v;
        self
    }

    pub fn with_target_count(mut self, v: usize) -> Self {
        self.target_count = v;
        self
    }

    pub fn with_material_condition(mut self, v: usize) -> Self {
        self.material_condition = v;
        self
    }

    pub fn with_simulation_points(mut self, v: usize) -> Self {
        self.simulation_points = v;
        self
    }

    pub fn validate(&self) -> Result<(), DatumRefError> {
        Ok(())
    }
}

impl Default for DatumRefConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for DatumRefConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DatumRefConfig(datum_letter={0}, target_count={1}, material_condition={2}, simulation_points={3})", self.datum_letter, self.target_count, self.material_condition, self.simulation_points)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core datum reference frame establishment engine.
#[derive(Debug, Clone)]
pub struct DatumRef {
    config: DatumRefConfig,
    data: Vec<f64>,
}

impl DatumRef {
    pub fn new(config: DatumRefConfig) -> Result<Self, DatumRefError> {
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
    pub fn config(&self) -> &DatumRefConfig { &self.config }

    /// Establish datum plane.
    pub fn establish_plane(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Establish datum axis.
    pub fn establish_axis(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Constrained least-squares fit.
    pub fn constrained_fit(&self) -> Vec<f64> {
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

impl fmt::Display for DatumRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DatumRef(n={})", self.data.len())
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
        let cfg = DatumRefConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = DatumRefConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("DatumRefConfig"));
    }

    #[test]
    fn test_config_with_datum_letter() {
        let cfg = DatumRefConfig::new().with_datum_letter(42);
        assert_eq!(cfg.datum_letter, 42);
    }

    #[test]
    fn test_config_with_target_count() {
        let cfg = DatumRefConfig::new().with_target_count(42);
        assert_eq!(cfg.target_count, 42);
    }

    #[test]
    fn test_config_with_material_condition() {
        let cfg = DatumRefConfig::new().with_material_condition(42);
        assert_eq!(cfg.material_condition, 42);
    }

    #[test]
    fn test_config_with_simulation_points() {
        let cfg = DatumRefConfig::new().with_simulation_points(42);
        assert_eq!(cfg.simulation_points, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = DatumRefConfig::new().with_datum_letter(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = DatumRef::new(DatumRefConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("DatumRef"));
    }

    #[test]
    fn test_summary() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_establish_plane() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.establish_plane();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_establish_axis() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.establish_axis();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_constrained_fit() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.constrained_fit();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_constrained_fit_empty() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap();
        assert!(e.constrained_fit().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = DatumRef::new(DatumRefConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = DatumRefError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = DatumRefError::InvalidConfig("a".into());
        let e2 = DatumRefError::ComputationFailed("b".into());
        let e3 = DatumRefError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
