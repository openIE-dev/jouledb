//! Runout tolerance measurement simulation.
//!
//! Provides [`RunoutTolConfig`] builder and [`RunoutTol`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RunoutTolError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RunoutTolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RunoutTol: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RunoutTol: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RunoutTol: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RunoutTol`] parameters.
#[derive(Debug, Clone)]
pub struct RunoutTolConfig {
    pub tolerance_value: f64,
    pub total_runout: bool,
    pub num_sections: usize,
    pub datum_axis_tol: f64,
}

impl RunoutTolConfig {
    pub fn new() -> Self {
        Self {
            tolerance_value: 0.05,
            total_runout: false,
            num_sections: 36,
            datum_axis_tol: 0.01,
        }
    }

    pub fn with_tolerance_value(mut self, v: f64) -> Self {
        self.tolerance_value = v;
        self
    }

    pub fn with_total_runout(mut self, v: bool) -> Self {
        self.total_runout = v;
        self
    }

    pub fn with_num_sections(mut self, v: usize) -> Self {
        self.num_sections = v;
        self
    }

    pub fn with_datum_axis_tol(mut self, v: f64) -> Self {
        self.datum_axis_tol = v;
        self
    }

    pub fn validate(&self) -> Result<(), RunoutTolError> {
        if self.tolerance_value.is_nan() {
            return Err(RunoutTolError::InvalidConfig("tolerance_value is NaN".into()));
        }
        if self.datum_axis_tol.is_nan() {
            return Err(RunoutTolError::InvalidConfig("datum_axis_tol is NaN".into()));
        }
        Ok(())
    }
}

impl Default for RunoutTolConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RunoutTolConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RunoutTolConfig(tolerance_value={0:.4}, total_runout={1}, num_sections={2}, datum_axis_tol={3:.4})", self.tolerance_value, self.total_runout, self.num_sections, self.datum_axis_tol)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core runout tolerance measurement simulation engine.
#[derive(Debug, Clone)]
pub struct RunoutTol {
    config: RunoutTolConfig,
    data: Vec<f64>,
}

impl RunoutTol {
    pub fn new(config: RunoutTolConfig) -> Result<Self, RunoutTolError> {
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
    pub fn config(&self) -> &RunoutTolConfig { &self.config }

    /// Measure circular runout.
    pub fn circular_runout(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Measure total runout.
    pub fn total_runout_val(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Full indicator movement.
    pub fn fim(&self) -> f64 {
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

impl fmt::Display for RunoutTol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RunoutTol(n={})", self.data.len())
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
        let cfg = RunoutTolConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RunoutTolConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RunoutTolConfig"));
    }

    #[test]
    fn test_config_with_tolerance_value() {
        let cfg = RunoutTolConfig::new().with_tolerance_value(42.0);
        assert!((cfg.tolerance_value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_total_runout() {
        let cfg = RunoutTolConfig::new().with_total_runout(true);
        assert_eq!(cfg.total_runout, true);
    }

    #[test]
    fn test_config_with_num_sections() {
        let cfg = RunoutTolConfig::new().with_num_sections(42);
        assert_eq!(cfg.num_sections, 42);
    }

    #[test]
    fn test_config_with_datum_axis_tol() {
        let cfg = RunoutTolConfig::new().with_datum_axis_tol(42.0);
        assert!((cfg.datum_axis_tol - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RunoutTolConfig::new().with_tolerance_value(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RunoutTol::new(RunoutTolConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RunoutTol"));
    }

    #[test]
    fn test_summary() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_circular_runout() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.circular_runout();
        assert!(result.is_finite());
    }

    #[test]
    fn test_total_runout_val() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.total_runout_val();
        assert!(result.is_finite());
    }

    #[test]
    fn test_fim() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.fim();
        assert!(result.is_finite());
    }

    #[test]
    fn test_fim_empty() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap();
        assert!((e.fim() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = RunoutTol::new(RunoutTolConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RunoutTolError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RunoutTolError::InvalidConfig("a".into());
        let e2 = RunoutTolError::ComputationFailed("b".into());
        let e3 = RunoutTolError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
