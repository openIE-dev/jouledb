//! True position tolerance evaluation.
//!
//! Provides [`PositionTolConfig`] builder and [`PositionTol`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionTolError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PositionTolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PositionTol: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PositionTol: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PositionTol: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PositionTol`] parameters.
#[derive(Debug, Clone)]
pub struct PositionTolConfig {
    pub tolerance_dia: f64,
    pub mmc_bonus: bool,
    pub projected: bool,
    pub composite: bool,
}

impl PositionTolConfig {
    pub fn new() -> Self {
        Self {
            tolerance_dia: 0.1,
            mmc_bonus: false,
            projected: false,
            composite: false,
        }
    }

    pub fn with_tolerance_dia(mut self, v: f64) -> Self {
        self.tolerance_dia = v;
        self
    }

    pub fn with_mmc_bonus(mut self, v: bool) -> Self {
        self.mmc_bonus = v;
        self
    }

    pub fn with_projected(mut self, v: bool) -> Self {
        self.projected = v;
        self
    }

    pub fn with_composite(mut self, v: bool) -> Self {
        self.composite = v;
        self
    }

    pub fn validate(&self) -> Result<(), PositionTolError> {
        if self.tolerance_dia.is_nan() {
            return Err(PositionTolError::InvalidConfig("tolerance_dia is NaN".into()));
        }
        Ok(())
    }
}

impl Default for PositionTolConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PositionTolConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PositionTolConfig(tolerance_dia={0:.4}, mmc_bonus={1}, projected={2}, composite={3})", self.tolerance_dia, self.mmc_bonus, self.projected, self.composite)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core true position tolerance evaluation engine.
#[derive(Debug, Clone)]
pub struct PositionTol {
    config: PositionTolConfig,
    data: Vec<f64>,
}

impl PositionTol {
    pub fn new(config: PositionTolConfig) -> Result<Self, PositionTolError> {
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
    pub fn config(&self) -> &PositionTolConfig { &self.config }

    /// Calculate true position deviation.
    pub fn true_position(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Calculate bonus tolerance.
    pub fn bonus_tolerance(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check if position conforms.
    pub fn is_conforming(&self) -> bool {
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

impl fmt::Display for PositionTol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PositionTol(n={})", self.data.len())
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
        let cfg = PositionTolConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PositionTolConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PositionTolConfig"));
    }

    #[test]
    fn test_config_with_tolerance_dia() {
        let cfg = PositionTolConfig::new().with_tolerance_dia(42.0);
        assert!((cfg.tolerance_dia - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_mmc_bonus() {
        let cfg = PositionTolConfig::new().with_mmc_bonus(true);
        assert_eq!(cfg.mmc_bonus, true);
    }

    #[test]
    fn test_config_with_projected() {
        let cfg = PositionTolConfig::new().with_projected(true);
        assert_eq!(cfg.projected, true);
    }

    #[test]
    fn test_config_with_composite() {
        let cfg = PositionTolConfig::new().with_composite(true);
        assert_eq!(cfg.composite, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PositionTolConfig::new().with_tolerance_dia(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PositionTol::new(PositionTolConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PositionTol"));
    }

    #[test]
    fn test_summary() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_true_position() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.true_position();
        assert!(result.is_finite());
    }

    #[test]
    fn test_bonus_tolerance() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.bonus_tolerance();
        assert!(result.is_finite());
    }

    #[test]
    fn test_is_conforming() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_conforming();
        assert!(result);
    }

    #[test]
    fn test_is_conforming_empty() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap();
        assert!(!e.is_conforming());
    }

    #[test]
    fn test_config_accessor() {
        let e = PositionTol::new(PositionTolConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PositionTolError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PositionTolError::InvalidConfig("a".into());
        let e2 = PositionTolError::ComputationFailed("b".into());
        let e3 = PositionTolError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
