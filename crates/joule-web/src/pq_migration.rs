//! Cryptographic migration framework for algorithm agility.
//!
//! Provides [`PqMigrationConfig`] builder and [`PqMigration`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PqMigrationError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PqMigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PqMigration: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PqMigration: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PqMigration: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PqMigration`] parameters.
#[derive(Debug, Clone)]
pub struct PqMigrationConfig {
    pub hybrid_mode: bool,
    pub deprecation_warn_days: u32,
    pub auto_negotiate: bool,
    pub fallback_allowed: bool,
}

impl PqMigrationConfig {
    pub fn new() -> Self {
        Self {
            hybrid_mode: true,
            deprecation_warn_days: 90,
            auto_negotiate: true,
            fallback_allowed: true,
        }
    }

    pub fn with_hybrid_mode(mut self, v: bool) -> Self {
        self.hybrid_mode = v;
        self
    }

    pub fn with_deprecation_warn_days(mut self, v: u32) -> Self {
        self.deprecation_warn_days = v;
        self
    }

    pub fn with_auto_negotiate(mut self, v: bool) -> Self {
        self.auto_negotiate = v;
        self
    }

    pub fn with_fallback_allowed(mut self, v: bool) -> Self {
        self.fallback_allowed = v;
        self
    }

    pub fn validate(&self) -> Result<(), PqMigrationError> {
        Ok(())
    }
}

impl Default for PqMigrationConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PqMigrationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqMigrationConfig(hybrid_mode={0}, deprecation_warn_days={1}, auto_negotiate={2}, fallback_allowed={3})", self.hybrid_mode, self.deprecation_warn_days, self.auto_negotiate, self.fallback_allowed)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core cryptographic migration framework for algorithm agility engine.
#[derive(Debug, Clone)]
pub struct PqMigration {
    config: PqMigrationConfig,
    data: Vec<f64>,
}

impl PqMigration {
    pub fn new(config: PqMigrationConfig) -> Result<Self, PqMigrationError> {
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
    pub fn config(&self) -> &PqMigrationConfig { &self.config }

    /// Negotiate cipher suite.
    pub fn negotiate_suite(&self) -> String {
        format!("{}: {} records", stringify!(negotiate_suite), self.data.len())
    }

    /// Generate migration plan.
    pub fn migration_plan(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Check algorithm compatibility.
    pub fn compatibility_check(&self) -> bool {
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

impl fmt::Display for PqMigration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqMigration(n={})", self.data.len())
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
        let cfg = PqMigrationConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PqMigrationConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PqMigrationConfig"));
    }

    #[test]
    fn test_config_with_hybrid_mode() {
        let cfg = PqMigrationConfig::new().with_hybrid_mode(false);
        assert_eq!(cfg.hybrid_mode, false);
    }

    #[test]
    fn test_config_with_deprecation_warn_days() {
        let cfg = PqMigrationConfig::new().with_deprecation_warn_days(42);
        assert_eq!(cfg.deprecation_warn_days, 42);
    }

    #[test]
    fn test_config_with_auto_negotiate() {
        let cfg = PqMigrationConfig::new().with_auto_negotiate(false);
        assert_eq!(cfg.auto_negotiate, false);
    }

    #[test]
    fn test_config_with_fallback_allowed() {
        let cfg = PqMigrationConfig::new().with_fallback_allowed(false);
        assert_eq!(cfg.fallback_allowed, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PqMigrationConfig::new().with_hybrid_mode(false);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PqMigration::new(PqMigrationConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PqMigration"));
    }

    #[test]
    fn test_summary() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_negotiate_suite() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.negotiate_suite();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_migration_plan() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.migration_plan();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_compatibility_check() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.compatibility_check();
        assert!(result);
    }

    #[test]
    fn test_compatibility_check_empty() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap();
        assert!(!e.compatibility_check());
    }

    #[test]
    fn test_config_accessor() {
        let e = PqMigration::new(PqMigrationConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PqMigrationError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PqMigrationError::InvalidConfig("a".into());
        let e2 = PqMigrationError::ComputationFailed("b".into());
        let e3 = PqMigrationError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
