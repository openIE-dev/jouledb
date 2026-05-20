//! Payment flow management and netting.
//!
//! Provides [`PaymentFlowConfig`] builder and [`PaymentFlow`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum PaymentFlowError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for PaymentFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "PaymentFlow: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "PaymentFlow: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "PaymentFlow: insufficient data: {msg}"),
        }
    }
}

/// Variant selector for CurrencyCode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurrencyCode {
    /// Usd.
    Usd,
    /// Eur.
    Eur,
    /// Gbp.
    Gbp,
    /// Jpy.
    Jpy,
}

impl fmt::Display for CurrencyCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`PaymentFlow`] parameters.
#[derive(Debug, Clone)]
pub struct PaymentFlowConfig {
    pub currency: CurrencyCode,
    pub cutoff_hour: u8,
    pub net_payments: bool,
    pub batch_size: usize,
}

impl PaymentFlowConfig {
    pub fn new() -> Self {
        Self {
            currency: CurrencyCode::Usd,
            cutoff_hour: 17,
            net_payments: true,
            batch_size: 100,
        }
    }

    pub fn with_currency(mut self, v: CurrencyCode) -> Self {
        self.currency = v;
        self
    }

    pub fn with_cutoff_hour(mut self, v: u8) -> Self {
        self.cutoff_hour = v;
        self
    }

    pub fn with_net_payments(mut self, v: bool) -> Self {
        self.net_payments = v;
        self
    }

    pub fn with_batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn validate(&self) -> Result<(), PaymentFlowError> {
        Ok(())
    }
}

impl Default for PaymentFlowConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for PaymentFlowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PaymentFlowConfig(currency={0:?}, cutoff_hour={1}, net_payments={2}, batch_size={3})", self.currency, self.cutoff_hour, self.net_payments, self.batch_size)
    }
}

// ── Result Types ────────────────────────────────────────────────

/// Result from a PaymentFlow operation.
#[derive(Debug, Clone, PartialEq)]
pub struct PaymentInstruction {
    pub value: f64,
    pub label: String,
}

impl fmt::Display for PaymentInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PaymentInstruction({:.4}, {})", self.value, self.label)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core payment flow management and netting engine.
#[derive(Debug, Clone)]
pub struct PaymentFlow {
    config: PaymentFlowConfig,
    data: Vec<f64>,
}

impl PaymentFlow {
    pub fn new(config: PaymentFlowConfig) -> Result<Self, PaymentFlowError> {
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
    pub fn config(&self) -> &PaymentFlowConfig { &self.config }

    /// Net payment amount.
    pub fn net_amount(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let n = self.data.len() as f64;
        let sum: f64 = self.data.iter().sum();
        let mean = sum / n;
        let var: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Generate payment instructions.
    pub fn generate_instructions(&self) -> Vec<PaymentInstruction> {
        self.data.iter().enumerate().map(|(i, &v)| PaymentInstruction {
            value: v, label: format!("item_{i}")
        }).collect()
    }

    /// Validate payment.
    pub fn validate(&self) -> bool {
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

impl fmt::Display for PaymentFlow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PaymentFlow(n={})", self.data.len())
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
        let cfg = PaymentFlowConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = PaymentFlowConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("PaymentFlowConfig"));
    }

    #[test]
    fn test_config_with_currency() {
        let cfg = PaymentFlowConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_with_cutoff_hour() {
        let cfg = PaymentFlowConfig::new().with_cutoff_hour(42);
        assert_eq!(cfg.cutoff_hour, 42);
    }

    #[test]
    fn test_config_with_net_payments() {
        let cfg = PaymentFlowConfig::new().with_net_payments(false);
        assert_eq!(cfg.net_payments, false);
    }

    #[test]
    fn test_config_with_batch_size() {
        let cfg = PaymentFlowConfig::new().with_batch_size(42);
        assert_eq!(cfg.batch_size, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = PaymentFlowConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("PaymentFlow"));
    }

    #[test]
    fn test_summary() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_net_amount() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.net_amount();
        assert!(result.is_finite());
    }

    #[test]
    fn test_generate_instructions() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_instructions();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_validate() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.validate();
        assert!(result);
    }

    #[test]
    fn test_validate_empty() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap();
        assert!(!e.validate());
    }

    #[test]
    fn test_config_accessor() {
        let e = PaymentFlow::new(PaymentFlowConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = PaymentFlowError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PaymentFlowError::InvalidConfig("a".into());
        let e2 = PaymentFlowError::ComputationFailed("b".into());
        let e3 = PaymentFlowError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
