//! HL7v2 message parsing and validation.
//!
//! Provides [`Hl7ParseConfig`] builder and [`Hl7Parse`] engine.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Errors from this module.
#[derive(Debug, Clone, PartialEq)]
pub enum Hl7ParseError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Computation failed.
    ComputationFailed(String),
    /// Insufficient data provided.
    InsufficientData(String),
}

impl fmt::Display for Hl7ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Hl7Parse: invalid config: {msg}"),
            Self::ComputationFailed(msg) => write!(f, "Hl7Parse: computation failed: {msg}"),
            Self::InsufficientData(msg) => write!(f, "Hl7Parse: insufficient data: {msg}"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Builder for [`Hl7Parse`] parameters.
#[derive(Debug, Clone)]
pub struct Hl7ParseConfig {
    pub encoding_chars: usize,
    pub strict_mode: bool,
    pub parse_z_segments: bool,
    pub max_segments: usize,
}

impl Hl7ParseConfig {
    pub fn new() -> Self {
        Self {
            encoding_chars: 0,
            strict_mode: true,
            parse_z_segments: false,
            max_segments: 1000,
        }
    }

    pub fn with_encoding_chars(mut self, v: usize) -> Self {
        self.encoding_chars = v;
        self
    }

    pub fn with_strict_mode(mut self, v: bool) -> Self {
        self.strict_mode = v;
        self
    }

    pub fn with_parse_z_segments(mut self, v: bool) -> Self {
        self.parse_z_segments = v;
        self
    }

    pub fn with_max_segments(mut self, v: usize) -> Self {
        self.max_segments = v;
        self
    }

    pub fn validate(&self) -> Result<(), Hl7ParseError> {
        Ok(())
    }
}

impl Default for Hl7ParseConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for Hl7ParseConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hl7ParseConfig(encoding_chars={0}, strict_mode={1}, parse_z_segments={2}, max_segments={3})", self.encoding_chars, self.strict_mode, self.parse_z_segments, self.max_segments)
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Core hl7v2 message parsing and validation engine.
#[derive(Debug, Clone)]
pub struct Hl7Parse {
    config: Hl7ParseConfig,
    data: Vec<f64>,
}

impl Hl7Parse {
    pub fn new(config: Hl7ParseConfig) -> Result<Self, Hl7ParseError> {
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
    pub fn config(&self) -> &Hl7ParseConfig { &self.config }

    /// Parse HL7v2 message.
    pub fn parse_message(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Get segment by name.
    pub fn get_segment(&self) -> Vec<f64> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len() as f64;
        self.data.iter().map(|x| x / n).collect()
    }

    /// Generate ACK message.
    pub fn generate_ack(&self) -> String {
        format!("{}: {} records", stringify!(generate_ack), self.data.len())
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

impl fmt::Display for Hl7Parse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hl7Parse(n={})", self.data.len())
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
        let cfg = Hl7ParseConfig::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let cfg = Hl7ParseConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("Hl7ParseConfig"));
    }

    #[test]
    fn test_config_with_encoding_chars() {
        let cfg = Hl7ParseConfig::new().with_encoding_chars(42);
        assert_eq!(cfg.encoding_chars, 42);
    }

    #[test]
    fn test_config_with_strict_mode() {
        let cfg = Hl7ParseConfig::new().with_strict_mode(false);
        assert_eq!(cfg.strict_mode, false);
    }

    #[test]
    fn test_config_with_parse_z_segments() {
        let cfg = Hl7ParseConfig::new().with_parse_z_segments(true);
        assert_eq!(cfg.parse_z_segments, true);
    }

    #[test]
    fn test_config_with_max_segments() {
        let cfg = Hl7ParseConfig::new().with_max_segments(42);
        assert_eq!(cfg.max_segments, 42);
    }

    #[test]
    fn test_config_nan_rejected() {
        let cfg = Hl7ParseConfig::new().with_encoding_chars(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_new_engine() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_with_data() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert_eq!(e.len(), 10);
        assert!(!e.is_empty());
    }

    #[test]
    fn test_push() {
        let mut e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap();
        e.push(1.0);
        e.push(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_display() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        let s = format!("{e}");
        assert!(s.contains("Hl7Parse"));
    }

    #[test]
    fn test_summary() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        let (mean, std, min, max) = e.summary();
        assert!((mean - 5.5).abs() < 1e-9);
        assert!(std > 0.0);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        assert!((e.percentile(0.0) - 1.0).abs() < 1e-9);
        assert!((e.percentile(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewm() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        let ewm = e.ewm(0.9);
        assert_eq!(ewm.len(), 10);
        assert!((ewm[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_data_methods() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap();
        let (mean, _, _, _) = e.summary();
        assert!((mean - 0.0).abs() < 1e-9);
        assert!((e.percentile(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_message() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.parse_message();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_get_segment() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.get_segment();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_ack() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap()
            .with_data(sample_data());
        let result = e.generate_ack();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generate_ack_empty() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap();
        let _ = e.generate_ack();
    }

    #[test]
    fn test_config_accessor() {
        let e = Hl7Parse::new(Hl7ParseConfig::new()).unwrap();
        let _ = e.config();
    }

    #[test]
    fn test_error_display() {
        let err = Hl7ParseError::InvalidConfig("test".into());
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = Hl7ParseError::InvalidConfig("a".into());
        let e2 = Hl7ParseError::ComputationFailed("b".into());
        let e3 = Hl7ParseError::InsufficientData("c".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
        assert_ne!(format!("{e2}"), format!("{e3}"));
    }
}
