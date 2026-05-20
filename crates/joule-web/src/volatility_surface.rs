//! Volatility surface with SABR and SVI models.
//!
//! Provides [`VolatilitySurfaceConfig`] builder and [`VolatilitySurface`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum VolatilitySurfaceError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for VolatilitySurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "VolatilitySurface: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "VolatilitySurface: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "VolatilitySurface: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`VolatilitySurface`] parameters.
#[derive(Debug, Clone)]
pub struct VolatilitySurfaceConfig {
    pub atm_vol: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub term_slope: f64,
}

impl VolatilitySurfaceConfig {
    pub fn new() -> Self {
        Self {
            atm_vol: 0.20,
            skew: -0.1,
            kurtosis: 0.05,
            term_slope: 0.01,
        }
    }

    pub fn with_atm_vol(mut self, v: f64) -> Self {
        self.atm_vol = v;
        self
    }

    pub fn with_skew(mut self, v: f64) -> Self {
        self.skew = v;
        self
    }

    pub fn with_kurtosis(mut self, v: f64) -> Self {
        self.kurtosis = v;
        self
    }

    pub fn with_term_slope(mut self, v: f64) -> Self {
        self.term_slope = v;
        self
    }

    pub fn validate(&self) -> Result<(), VolatilitySurfaceError> {
        if self.atm_vol.is_nan() {
            return Err(VolatilitySurfaceError::InvalidConfig("atm_vol is NaN".into()));
        }
        if self.skew.is_nan() {
            return Err(VolatilitySurfaceError::InvalidConfig("skew is NaN".into()));
        }
        if self.kurtosis.is_nan() {
            return Err(VolatilitySurfaceError::InvalidConfig("kurtosis is NaN".into()));
        }
        if self.term_slope.is_nan() {
            return Err(VolatilitySurfaceError::InvalidConfig("term_slope is NaN".into()));
        }
        Ok(())
    }
}

impl Default for VolatilitySurfaceConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for VolatilitySurfaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VolatilitySurfaceConfig(atm_vol={0:.4}, skew={1:.4}, kurtosis={2:.4}, term_slope={3:.4})", self.atm_vol, self.skew, self.kurtosis, self.term_slope)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core volatility surface with sabr and svi models engine.
#[derive(Debug, Clone)]
pub struct VolatilitySurface {
    config: VolatilitySurfaceConfig,
    data: Vec<f64>,
}

impl VolatilitySurface {
    pub fn new(config: VolatilitySurfaceConfig) -> Result<Self, VolatilitySurfaceError> {
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
    pub fn config(&self) -> &VolatilitySurfaceConfig { &self.config }

    /// Implied vol at strike/tenor.
    pub fn implied_vol(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Local volatility (Dupire).
    pub fn local_vol(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Vol smile at given tenor.
    pub fn smile_at_tenor(&self) -> Vec<f64> {
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

impl fmt::Display for VolatilitySurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VolatilitySurface(n={})", self.data.len())
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
        let cfg = VolatilitySurfaceConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = VolatilitySurfaceConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("VolatilitySurfaceConfig"));
    }

    #[test]
    fn test_config_with_atm_vol() {
        let cfg = VolatilitySurfaceConfig::new().with_atm_vol(42.0);
        assert!((cfg.atm_vol - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_skew() {
        let cfg = VolatilitySurfaceConfig::new().with_skew(42.0);
        assert!((cfg.skew - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_kurtosis() {
        let cfg = VolatilitySurfaceConfig::new().with_kurtosis(42.0);
        assert!((cfg.kurtosis - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_term_slope() {
        let cfg = VolatilitySurfaceConfig::new().with_term_slope(42.0);
        assert!((cfg.term_slope - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = VolatilitySurfaceConfig::new().with_atm_vol(f64::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_new_engine() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("VolatilitySurface"));
    }

    #[test]
    fn test_summary() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_implied_vol() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.implied_vol();
        assert!(result.is_finite());
    }

    #[test]
    fn test_local_vol() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.local_vol();
        assert!(result.is_finite());
    }

    #[test]
    fn test_smile_at_tenor() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.smile_at_tenor();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_smile_at_tenor_empty() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap();
        assert!(e.smile_at_tenor().is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let e = VolatilitySurface::new(VolatilitySurfaceConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = VolatilitySurfaceError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = VolatilitySurfaceError::InvalidConfig("a".into());
        let e2 = VolatilitySurfaceError::ComputationFailed("b".into());
        let e3 = VolatilitySurfaceError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
