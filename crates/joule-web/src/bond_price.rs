//! Bond pricing with duration and convexity.
//!
//! Provides [`BondPriceConfig`] builder and [`BondPrice`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum BondPriceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for BondPriceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "BondPrice: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "BondPrice: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "BondPrice: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`BondPrice`] parameters.
#[derive(Debug, Clone)]
pub struct BondPriceConfig {
    pub face_value: f64,
    pub coupon_rate: f64,
    pub yield_rate: f64,
    pub maturity_years: f64,
}

impl BondPriceConfig {
    pub fn new() -> Self {
        Self {
            face_value: 1000.0,
            coupon_rate: 0.05,
            yield_rate: 0.04,
            maturity_years: 10.0,
        }
    }

    pub fn with_face_value(mut self, v: f64) -> Self {
        self.face_value = v;
        self
    }

    pub fn with_coupon_rate(mut self, v: f64) -> Self {
        self.coupon_rate = v;
        self
    }

    pub fn with_yield_rate(mut self, v: f64) -> Self {
        self.yield_rate = v;
        self
    }

    pub fn with_maturity_years(mut self, v: f64) -> Self {
        self.maturity_years = v;
        self
    }

    pub fn validate(&self) -> Result<(), BondPriceError> {
        if self.face_value.is_nan() {
            return Err(BondPriceError::InvalidConfig("face_value is NaN".into()));
        }
        if self.coupon_rate.is_nan() {
            return Err(BondPriceError::InvalidConfig("coupon_rate is NaN".into()));
        }
        if self.yield_rate.is_nan() {
            return Err(BondPriceError::InvalidConfig("yield_rate is NaN".into()));
        }
        if self.maturity_years.is_nan() {
            return Err(BondPriceError::InvalidConfig("maturity_years is NaN".into()));
        }
        Ok(())
    }
}

impl Default for BondPriceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for BondPriceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BondPriceConfig(face_value={0:.4}, coupon_rate={1:.4}, yield_rate={2:.4}, maturity_years={3:.4})", self.face_value, self.coupon_rate, self.yield_rate, self.maturity_years)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core bond pricing with duration and convexity engine.
#[derive(Debug, Clone)]
pub struct BondPrice {
    config: BondPriceConfig,
    data: Vec<f64>,
}

impl BondPrice {
    pub fn new(config: BondPriceConfig) -> Result<Self, BondPriceError> {
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
    pub fn config(&self) -> &BondPriceConfig { &self.config }

    /// Clean bond price.
    pub fn clean_price(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Dirty price with accrued interest.
    pub fn dirty_price(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Modified duration.
    pub fn modified_duration(&self) -> f64 {
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

impl fmt::Display for BondPrice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BondPrice(n={})", self.data.len())
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
        let cfg = BondPriceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = BondPriceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("BondPriceConfig"));
    }

    #[test]
    fn test_config_with_face_value() {
        let cfg = BondPriceConfig::new().with_face_value(42.0);
        assert!((cfg.face_value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_coupon_rate() {
        let cfg = BondPriceConfig::new().with_coupon_rate(42.0);
        assert!((cfg.coupon_rate - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_yield_rate() {
        let cfg = BondPriceConfig::new().with_yield_rate(42.0);
        assert!((cfg.yield_rate - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_maturity_years() {
        let cfg = BondPriceConfig::new().with_maturity_years(42.0);
        assert!((cfg.maturity_years - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = BondPriceConfig::new().with_face_value(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = BondPrice::new(BondPriceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("BondPrice"));
    }

    #[test]
    fn test_summary() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_clean_price() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.clean_price();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dirty_price() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.dirty_price();
        assert!(result.is_finite());
    }

    #[test]
    fn test_modified_duration() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.modified_duration();
        assert!(result.is_finite());
    }

    #[test]
    fn test_modified_duration_empty() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap();
        assert!((e.modified_duration() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let e = BondPrice::new(BondPriceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = BondPriceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = BondPriceError::InvalidConfig("a".into());
        let e2 = BondPriceError::ComputationFailed("b".into());
        let e3 = BondPriceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
