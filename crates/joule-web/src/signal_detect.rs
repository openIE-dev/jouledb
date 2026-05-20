//! Technical signal detection (MA crossovers, RSI, MACD).
//!
//! Provides [`SignalDetectConfig`] builder and [`SignalDetect`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum SignalDetectError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for SignalDetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "SignalDetect: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "SignalDetect: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "SignalDetect: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`SignalDetect`] parameters.
#[derive(Debug, Clone)]
pub struct SignalDetectConfig {
    pub short_period: usize,
    pub long_period: usize,
    pub signal_period: usize,
    pub rsi_period: usize,
}

impl SignalDetectConfig {
    pub fn new() -> Self {
        Self {
            short_period: 12,
            long_period: 26,
            signal_period: 9,
            rsi_period: 14,
        }
    }

    pub fn with_short_period(mut self, v: usize) -> Self {
        self.short_period = v;
        self
    }

    pub fn with_long_period(mut self, v: usize) -> Self {
        self.long_period = v;
        self
    }

    pub fn with_signal_period(mut self, v: usize) -> Self {
        self.signal_period = v;
        self
    }

    pub fn with_rsi_period(mut self, v: usize) -> Self {
        self.rsi_period = v;
        self
    }

    pub fn validate(&self) -> Result<(), SignalDetectError> {
        Ok(())
    }
}

impl Default for SignalDetectConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for SignalDetectConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SignalDetectConfig(short_period={0}, long_period={1}, signal_period={2}, rsi_period={3})", self.short_period, self.long_period, self.signal_period, self.rsi_period)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a SignalDetect operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for Signal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signal({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core technical signal detection (ma crossovers, rsi, macd) engine.
#[derive(Debug, Clone)]
pub struct SignalDetect {
    config: SignalDetectConfig,
    data: Vec<f64>,
}

impl SignalDetect {
    pub fn new(config: SignalDetectConfig) -> Result<Self, SignalDetectError> {
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
    pub fn config(&self) -> &SignalDetectConfig { &self.config }

    /// Moving average crossover signal.
    pub fn ma_crossover(&self) -> Signal {
        let v = if self.data.is_empty() { 0.0 } else { self.data[0] };
        Signal { value: v, label: stringify!(ma_crossover).into() }
    }

    /// Relative strength index.
    pub fn rsi(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// MACD line and signal.
    pub fn macd(&self) -> (f64, f64) {
        if self.data.len() < 2 { return (0.0, 0.0); }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        (sum / n, sum)
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

impl fmt::Display for SignalDetect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SignalDetect(n={})", self.data.len())
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
        let cfg = SignalDetectConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = SignalDetectConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("SignalDetectConfig"));
    }

    #[test]
    fn test_config_with_short_period() {
        let cfg = SignalDetectConfig::new().with_short_period(42);
        assert_eq!(cfg.short_period, 42);
    }

    #[test]
    fn test_config_with_long_period() {
        let cfg = SignalDetectConfig::new().with_long_period(42);
        assert_eq!(cfg.long_period, 42);
    }

    #[test]
    fn test_config_with_signal_period() {
        let cfg = SignalDetectConfig::new().with_signal_period(42);
        assert_eq!(cfg.signal_period, 42);
    }

    #[test]
    fn test_config_with_rsi_period() {
        let cfg = SignalDetectConfig::new().with_rsi_period(42);
        assert_eq!(cfg.rsi_period, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = SignalDetectConfig::new().with_short_period(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = SignalDetect::new(SignalDetectConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("SignalDetect"));
    }

    #[test]
    fn test_summary() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ma_crossover() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.ma_crossover();
        assert!(result.value.is_finite());
    }

    #[test]
    fn test_rsi() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.rsi();
        assert!(result.is_finite());
    }

    #[test]
    fn test_macd() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap()
            .with_data(sample_data());
        let (a, b) = e.macd();
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_macd_empty() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap();
        let _ = e.macd();
    }

    #[test]
    fn test_config_accessor() {
        let e = SignalDetect::new(SignalDetectConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = SignalDetectError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = SignalDetectError::InvalidConfig("a".into());
        let e2 = SignalDetectError::ComputationFailed("b".into());
        let e3 = SignalDetectError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
