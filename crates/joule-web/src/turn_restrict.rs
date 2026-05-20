//! Turn restriction modeling for routing.
//!
//! Provides [`TurnRestrictConfig`] builder and [`TurnRestrict`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnRestrictError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for TurnRestrictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "TurnRestrict: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "TurnRestrict: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "TurnRestrict: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`TurnRestrict`] parameters.
#[derive(Debug, Clone)]
pub struct TurnRestrictConfig {
    pub u_turn_penalty: f64,
    pub left_turn_penalty: f64,
    pub right_turn_penalty: f64,
    pub time_dependent: bool,
}

impl TurnRestrictConfig {
    pub fn new() -> Self {
        Self {
            u_turn_penalty: 1000.0,
            left_turn_penalty: 10.0,
            right_turn_penalty: 5.0,
            time_dependent: false,
        }
    }

    pub fn with_u_turn_penalty(mut self, v: f64) -> Self {
        self.u_turn_penalty = v;
        self
    }

    pub fn with_left_turn_penalty(mut self, v: f64) -> Self {
        self.left_turn_penalty = v;
        self
    }

    pub fn with_right_turn_penalty(mut self, v: f64) -> Self {
        self.right_turn_penalty = v;
        self
    }

    pub fn with_time_dependent(mut self, v: bool) -> Self {
        self.time_dependent = v;
        self
    }

    pub fn validate(&self) -> Result<(), TurnRestrictError> {
        if self.u_turn_penalty.is_nan() {
            return Err(TurnRestrictError::InvalidConfig("u_turn_penalty is NaN".into()));
        }
        if self.left_turn_penalty.is_nan() {
            return Err(TurnRestrictError::InvalidConfig("left_turn_penalty is NaN".into()));
        }
        if self.right_turn_penalty.is_nan() {
            return Err(TurnRestrictError::InvalidConfig("right_turn_penalty is NaN".into()));
        }
        Ok(())
    }
}

impl Default for TurnRestrictConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for TurnRestrictConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TurnRestrictConfig(u_turn_penalty={0:.4}, left_turn_penalty={1:.4}, right_turn_penalty={2:.4}, time_dependent={3})", self.u_turn_penalty, self.left_turn_penalty, self.right_turn_penalty, self.time_dependent)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core turn restriction modeling for routing engine.
#[derive(Debug, Clone)]
pub struct TurnRestrict {
    config: TurnRestrictConfig,
    data: Vec<f64>,
}

impl TurnRestrict {
    pub fn new(config: TurnRestrictConfig) -> Result<Self, TurnRestrictError> {
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
    pub fn config(&self) -> &TurnRestrictConfig { &self.config }

    /// Apply turn restriction.
    pub fn apply_restriction(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Determine maneuver type.
    pub fn maneuver_type(&self) -> String {
        format!("{}: {} records", stringify!(maneuver_type), self.data.len())
    }

    /// Check if turn is restricted.
    pub fn is_restricted(&self) -> bool {
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

impl fmt::Display for TurnRestrict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TurnRestrict(n={})", self.data.len())
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
        let cfg = TurnRestrictConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = TurnRestrictConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("TurnRestrictConfig"));
    }

    #[test]
    fn test_config_with_u_turn_penalty() {
        let cfg = TurnRestrictConfig::new().with_u_turn_penalty(42.0);
        assert!((cfg.u_turn_penalty - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_left_turn_penalty() {
        let cfg = TurnRestrictConfig::new().with_left_turn_penalty(42.0);
        assert!((cfg.left_turn_penalty - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_right_turn_penalty() {
        let cfg = TurnRestrictConfig::new().with_right_turn_penalty(42.0);
        assert!((cfg.right_turn_penalty - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_time_dependent() {
        let cfg = TurnRestrictConfig::new().with_time_dependent(true);
        assert_eq!(cfg.time_dependent, true);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = TurnRestrictConfig::new().with_u_turn_penalty(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("TurnRestrict"));
    }

    #[test]
    fn test_summary() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_restriction() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.apply_restriction();
        assert!(result.is_finite());
    }

    #[test]
    fn test_maneuver_type() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.maneuver_type();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_is_restricted() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.is_restricted();
        assert!(result);
    }

    #[test]
    fn test_is_restricted_empty() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap();
        assert!(!e.is_restricted());
    }

    #[test]
    fn test_config_accessor() {
        let e = TurnRestrict::new(TurnRestrictConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = TurnRestrictError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = TurnRestrictError::InvalidConfig("a".into());
        let e2 = TurnRestrictError::ComputationFailed("b".into());
        let e3 = TurnRestrictError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
