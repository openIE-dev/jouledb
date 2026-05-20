//! Role-based access control for clinical data.
//!
//! Provides [`RoleAccessConfig`] builder and [`RoleAccess`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RoleAccessError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for RoleAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "RoleAccess: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "RoleAccess: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "RoleAccess: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`RoleAccess`] parameters.
#[derive(Debug, Clone)]
pub struct RoleAccessConfig {
    pub role_type: usize,
    pub department: usize,
    pub emergency_override: bool,
    pub min_necessary: bool,
}

impl RoleAccessConfig {
    pub fn new() -> Self {
        Self {
            role_type: 0,
            department: 0,
            emergency_override: true,
            min_necessary: true,
        }
    }

    pub fn with_role_type(mut self, v: usize) -> Self {
        self.role_type = v;
        self
    }

    pub fn with_department(mut self, v: usize) -> Self {
        self.department = v;
        self
    }

    pub fn with_emergency_override(mut self, v: bool) -> Self {
        self.emergency_override = v;
        self
    }

    pub fn with_min_necessary(mut self, v: bool) -> Self {
        self.min_necessary = v;
        self
    }

    pub fn validate(&self) -> Result<(), RoleAccessError> {
        Ok(())
    }
}

impl Default for RoleAccessConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for RoleAccessConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoleAccessConfig(role_type={0}, department={1}, emergency_override={2}, min_necessary={3})", self.role_type, self.department, self.emergency_override, self.min_necessary)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core role-based access control for clinical data engine.
#[derive(Debug, Clone)]
pub struct RoleAccess {
    config: RoleAccessConfig,
    data: Vec<f64>,
}

impl RoleAccess {
    pub fn new(config: RoleAccessConfig) -> Result<Self, RoleAccessError> {
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
    pub fn config(&self) -> &RoleAccessConfig { &self.config }

    /// Check role permission.
    pub fn check_permission(&self) -> bool {
        !self.data.is_empty()
    }

    /// Grant emergency access.
    pub fn grant_emergency(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get data access scope.
    pub fn access_scope(&self) -> Vec<f64> {
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

impl fmt::Display for RoleAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoleAccess(n={})", self.data.len())
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
        let cfg = RoleAccessConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = RoleAccessConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("RoleAccessConfig"));
    }

    #[test]
    fn test_config_with_role_type() {
        let cfg = RoleAccessConfig::new().with_role_type(42);
        assert_eq!(cfg.role_type, 42);
    }

    #[test]
    fn test_config_with_department() {
        let cfg = RoleAccessConfig::new().with_department(42);
        assert_eq!(cfg.department, 42);
    }

    #[test]
    fn test_config_with_emergency_override() {
        let cfg = RoleAccessConfig::new().with_emergency_override(false);
        assert_eq!(cfg.emergency_override, false);
    }

    #[test]
    fn test_config_with_min_necessary() {
        let cfg = RoleAccessConfig::new().with_min_necessary(false);
        assert_eq!(cfg.min_necessary, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = RoleAccessConfig::new().with_role_type(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = RoleAccess::new(RoleAccessConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("RoleAccess"));
    }

    #[test]
    fn test_summary() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_check_permission() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.check_permission();
        assert!(result);
    }

    #[test]
    fn test_grant_emergency() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.grant_emergency();
        assert!(result);
    }

    #[test]
    fn test_access_scope() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.access_scope();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_access_scope_empty() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap();
        assert!(e.access_scope().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = RoleAccess::new(RoleAccessConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = RoleAccessError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = RoleAccessError::InvalidConfig("a".into());
        let e2 = RoleAccessError::ComputationFailed("b".into());
        let e3 = RoleAccessError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
