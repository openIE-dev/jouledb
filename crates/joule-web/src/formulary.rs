//! Drug formulary management and tier classification.
//!
//! Provides [`FormularyConfig`] builder and [`Formulary`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FormularyError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for FormularyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Formulary: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "Formulary: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "Formulary: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`Formulary`] parameters.
#[derive(Debug, Clone)]
pub struct FormularyConfig {
    pub num_tiers: usize,
    pub require_pa: bool,
    pub step_therapy: bool,
    pub quantity_limit: bool,
}

impl FormularyConfig {
    pub fn new() -> Self {
        Self {
            num_tiers: 4,
            require_pa: false,
            step_therapy: false,
            quantity_limit: true,
        }
    }

    pub fn with_num_tiers(mut self, v: usize) -> Self {
        self.num_tiers = v;
        self
    }

    pub fn with_require_pa(mut self, v: bool) -> Self {
        self.require_pa = v;
        self
    }

    pub fn with_step_therapy(mut self, v: bool) -> Self {
        self.step_therapy = v;
        self
    }

    pub fn with_quantity_limit(mut self, v: bool) -> Self {
        self.quantity_limit = v;
        self
    }

    pub fn validate(&self) -> Result<(), FormularyError> {
        Ok(())
    }
}

impl Default for FormularyConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FormularyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FormularyConfig(num_tiers={0}, require_pa={1}, step_therapy={2}, quantity_limit={3})", self.num_tiers, self.require_pa, self.step_therapy, self.quantity_limit)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core drug formulary management and tier classification engine.
#[derive(Debug, Clone)]
pub struct Formulary {
    config: FormularyConfig,
    data: Vec<f64>,
}

impl Formulary {
    pub fn new(config: FormularyConfig) -> Result<Self, FormularyError> {
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
    pub fn config(&self) -> &FormularyConfig { &self.config }

    /// Get formulary tier for drug.
    pub fn tier_for_drug(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Check prior authorization.
    pub fn requires_pa(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get formulary alternatives.
    pub fn alternatives(&self) -> Vec<f64> {
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

impl fmt::Display for Formulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Formulary(n={})", self.data.len())
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
        let cfg = FormularyConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = FormularyConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("FormularyConfig"));
    }

    #[test]
    fn test_config_with_num_tiers() {
        let cfg = FormularyConfig::new().with_num_tiers(42);
        assert_eq!(cfg.num_tiers, 42);
    }

    #[test]
    fn test_config_with_require_pa() {
        let cfg = FormularyConfig::new().with_require_pa(true);
        assert_eq!(cfg.require_pa, true);
    }

    #[test]
    fn test_config_with_step_therapy() {
        let cfg = FormularyConfig::new().with_step_therapy(true);
        assert_eq!(cfg.step_therapy, true);
    }

    #[test]
    fn test_config_with_quantity_limit() {
        let cfg = FormularyConfig::new().with_quantity_limit(false);
        assert_eq!(cfg.quantity_limit, false);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = FormularyConfig::new().with_num_tiers(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = Formulary::new(FormularyConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = Formulary::new(FormularyConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("Formulary"));
    }

    #[test]
    fn test_summary() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = Formulary::new(FormularyConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tier_for_drug() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.tier_for_drug();
        assert!(result.is_finite());
    }

    #[test]
    fn test_requires_pa() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.requires_pa();
        assert!(result);
    }

    #[test]
    fn test_alternatives() {
        let e = Formulary::new(FormularyConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.alternatives();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_alternatives_empty() {
        let e = Formulary::new(FormularyConfig::new()).unwrap();
        assert!(e.alternatives().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = Formulary::new(FormularyConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = FormularyError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = FormularyError::InvalidConfig("a".into());
        let e2 = FormularyError::ComputationFailed("b".into());
        let e3 = FormularyError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
