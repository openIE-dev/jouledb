//! Uncertainty Quantification
//!
//! Every SigQL result is an `UncertainValue<T>` that propagates
//! confidence intervals, noise floors, and artifact flags through
//! all computations.

use core::ops::{Add, Div, Mul, Sub};

/// A value with associated uncertainty quantification.
///
/// This is the return type for ALL SigQL aggregations and computations.
/// The uncertainty propagates through operations using standard error calculus.
#[derive(Debug, Clone, Copy)]
pub struct UncertainValue<T> {
    /// The central/point estimate
    pub value: T,
    /// Confidence level (0.0 to 1.0, typically 0.95)
    pub confidence: f64,
    /// Lower bound of confidence interval
    pub lower_bound: T,
    /// Upper bound of confidence interval
    pub upper_bound: T,
    /// Estimated noise floor in the measurement
    pub noise_floor: Option<T>,
    /// Signal-to-noise ratio (dB)
    pub snr_db: Option<f64>,
    /// Number of samples/observations used
    pub n_samples: usize,
    /// Quality flags
    pub flags: QualityFlags,
}

impl<T: Default> Default for UncertainValue<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            confidence: 0.95,
            lower_bound: T::default(),
            upper_bound: T::default(),
            noise_floor: None,
            snr_db: None,
            n_samples: 0,
            flags: QualityFlags::empty(),
        }
    }
}

impl UncertainValue<f64> {
    /// Create from value with symmetric confidence interval
    pub fn from_ci(value: f64, half_width: f64, confidence: f64, n: usize) -> Self {
        Self {
            value,
            confidence,
            lower_bound: value - half_width,
            upper_bound: value + half_width,
            noise_floor: None,
            snr_db: None,
            n_samples: n,
            flags: QualityFlags::empty(),
        }
    }

    /// Create from mean and standard error
    pub fn from_mean_se(mean: f64, std_error: f64, n: usize) -> Self {
        // 95% CI using t-distribution approximation (z=1.96 for large n)
        let z = if n > 30 { 1.96 } else { 2.0 }; // Simplified
        let half_width = z * std_error;
        Self::from_ci(mean, half_width, 0.95, n)
    }

    /// Create from bootstrap samples
    pub fn from_bootstrap(samples: &[f64], confidence: f64) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self::default();
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;

        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = ((n as f64) * alpha).floor() as usize;
        let upper_idx = ((n as f64) * (1.0 - alpha)).ceil() as usize;

        Self {
            value: mean,
            confidence,
            lower_bound: sorted.get(lower_idx).copied().unwrap_or(mean),
            upper_bound: sorted.get(upper_idx.min(n - 1)).copied().unwrap_or(mean),
            noise_floor: None,
            snr_db: None,
            n_samples: n,
            flags: QualityFlags::empty(),
        }
    }

    /// Check if value is statistically significant (CI doesn't include zero)
    pub fn is_significant(&self) -> bool {
        self.lower_bound > 0.0 || self.upper_bound < 0.0
    }

    /// Width of confidence interval
    pub fn ci_width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }

    /// Relative uncertainty (CI width / value)
    pub fn relative_uncertainty(&self) -> Option<f64> {
        if self.value.abs() > f64::EPSILON {
            Some(self.ci_width() / self.value.abs())
        } else {
            None
        }
    }
}

impl UncertainValue<f32> {
    /// Convert to f64 version
    pub fn to_f64(&self) -> UncertainValue<f64> {
        UncertainValue {
            value: self.value as f64,
            confidence: self.confidence,
            lower_bound: self.lower_bound as f64,
            upper_bound: self.upper_bound as f64,
            noise_floor: self.noise_floor.map(|x| x as f64),
            snr_db: self.snr_db,
            n_samples: self.n_samples,
            flags: self.flags,
        }
    }
}

/// Quality flags for uncertainty values
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QualityFlags(u32);

impl QualityFlags {
    pub const ARTIFACT_DETECTED: u32 = 1 << 0;
    pub const LOW_SNR: u32 = 1 << 1;
    pub const CLIPPED: u32 = 1 << 2;
    pub const INTERPOLATED: u32 = 1 << 3;
    pub const FLATLINE: u32 = 1 << 4;
    pub const HIGH_VARIANCE: u32 = 1 << 5;
    pub const EXTRAPOLATED: u32 = 1 << 6;
    pub const APPROXIMATE: u32 = 1 << 7;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn all() -> Self {
        Self(0xFF)
    }

    pub fn set(&mut self, flag: u32) {
        self.0 |= flag;
    }

    pub fn clear(&mut self, flag: u32) {
        self.0 &= !flag;
    }

    pub const fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }

    pub const fn is_clean(&self) -> bool {
        self.0 == 0
    }
}

/// Error propagation modes for combined operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorPropagation {
    /// Assume independent errors (sqrt sum of squares)
    Independent,
    /// Assume correlated errors (linear sum)
    Correlated,
    /// Use bootstrap resampling
    Bootstrap { replicates: usize },
    /// Monte Carlo simulation
    MonteCarlo { samples: usize },
}

/// Result of an uncertain computation that can carry multiple outputs
#[derive(Debug, Clone)]
pub struct UncertainResult {
    /// Named output values
    pub values: Vec<(String, UncertainValue<f64>)>,
    /// Overall confidence
    pub overall_confidence: f64,
    /// Computation metadata
    pub metadata: ComputationMetadata,
}

impl UncertainResult {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            overall_confidence: 0.95,
            metadata: ComputationMetadata::default(),
        }
    }

    pub fn with_value(mut self, name: impl Into<String>, value: UncertainValue<f64>) -> Self {
        self.values.push((name.into(), value));
        self
    }

    pub fn get(&self, name: &str) -> Option<&UncertainValue<f64>> {
        self.values.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

impl Default for UncertainResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about the computation that produced a result
#[derive(Debug, Clone, Default)]
pub struct ComputationMetadata {
    /// Time range analyzed
    pub time_range_ns: Option<(i64, i64)>,
    /// Samples processed
    pub samples_processed: usize,
    /// Computation time (nanoseconds)
    pub compute_time_ns: Option<u64>,
    /// Algorithm used
    pub algorithm: Option<String>,
    /// Version/parameters
    pub parameters: Vec<(String, String)>,
}

// Implement arithmetic with uncertainty propagation for f64

impl Add for UncertainValue<f64> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Independent error propagation: σ² = σ₁² + σ₂²
        let half_width_1 = (self.upper_bound - self.lower_bound) / 2.0;
        let half_width_2 = (rhs.upper_bound - rhs.lower_bound) / 2.0;
        let combined_half_width = (half_width_1.powi(2) + half_width_2.powi(2)).sqrt();

        let value = self.value + rhs.value;

        Self {
            value,
            confidence: self.confidence.min(rhs.confidence),
            lower_bound: value - combined_half_width,
            upper_bound: value + combined_half_width,
            noise_floor: match (self.noise_floor, rhs.noise_floor) {
                (Some(a), Some(b)) => Some((a.powi(2) + b.powi(2)).sqrt()),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            snr_db: None, // Would need recalculation
            n_samples: self.n_samples.min(rhs.n_samples),
            flags: QualityFlags(self.flags.0 | rhs.flags.0),
        }
    }
}

impl Sub for UncertainValue<f64> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // Same propagation as addition for subtraction
        let half_width_1 = (self.upper_bound - self.lower_bound) / 2.0;
        let half_width_2 = (rhs.upper_bound - rhs.lower_bound) / 2.0;
        let combined_half_width = (half_width_1.powi(2) + half_width_2.powi(2)).sqrt();

        let value = self.value - rhs.value;

        Self {
            value,
            confidence: self.confidence.min(rhs.confidence),
            lower_bound: value - combined_half_width,
            upper_bound: value + combined_half_width,
            noise_floor: None,
            snr_db: None,
            n_samples: self.n_samples.min(rhs.n_samples),
            flags: QualityFlags(self.flags.0 | rhs.flags.0),
        }
    }
}

impl Mul for UncertainValue<f64> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Relative error propagation: (σ/μ)² = (σ₁/μ₁)² + (σ₂/μ₂)²
        let rel_err_1 = if self.value.abs() > f64::EPSILON {
            (self.upper_bound - self.lower_bound) / (2.0 * self.value.abs())
        } else {
            0.0
        };
        let rel_err_2 = if rhs.value.abs() > f64::EPSILON {
            (rhs.upper_bound - rhs.lower_bound) / (2.0 * rhs.value.abs())
        } else {
            0.0
        };

        let value = self.value * rhs.value;
        let combined_rel_err = (rel_err_1.powi(2) + rel_err_2.powi(2)).sqrt();
        let half_width = value.abs() * combined_rel_err;

        Self {
            value,
            confidence: self.confidence.min(rhs.confidence),
            lower_bound: value - half_width,
            upper_bound: value + half_width,
            noise_floor: None,
            snr_db: None,
            n_samples: self.n_samples.min(rhs.n_samples),
            flags: QualityFlags(self.flags.0 | rhs.flags.0),
        }
    }
}

impl Div for UncertainValue<f64> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Same relative error propagation as multiplication
        let rel_err_1 = if self.value.abs() > f64::EPSILON {
            (self.upper_bound - self.lower_bound) / (2.0 * self.value.abs())
        } else {
            0.0
        };
        let rel_err_2 = if rhs.value.abs() > f64::EPSILON {
            (rhs.upper_bound - rhs.lower_bound) / (2.0 * rhs.value.abs())
        } else {
            0.0
        };

        let value = if rhs.value.abs() > f64::EPSILON {
            self.value / rhs.value
        } else {
            f64::NAN
        };
        let combined_rel_err = (rel_err_1.powi(2) + rel_err_2.powi(2)).sqrt();
        let half_width = value.abs() * combined_rel_err;

        Self {
            value,
            confidence: self.confidence.min(rhs.confidence),
            lower_bound: value - half_width,
            upper_bound: value + half_width,
            noise_floor: None,
            snr_db: None,
            n_samples: self.n_samples.min(rhs.n_samples),
            flags: QualityFlags(self.flags.0 | rhs.flags.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_ci() {
        let samples: Vec<f64> = (0..1000).map(|i| 10.0 + (i as f64 * 0.001)).collect();
        let result = UncertainValue::from_bootstrap(&samples, 0.95);

        assert!(result.lower_bound < result.value);
        assert!(result.upper_bound > result.value);
        assert_eq!(result.n_samples, 1000);
    }

    #[test]
    fn test_uncertainty_propagation() {
        let a = UncertainValue::from_ci(10.0, 1.0, 0.95, 100);
        let b = UncertainValue::from_ci(5.0, 0.5, 0.95, 100);

        let sum = a + b;
        assert!((sum.value - 15.0).abs() < 0.001);

        // Combined uncertainty should be sqrt(1² + 0.5²) ≈ 1.118
        let expected_half_width = (1.0_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        let actual_half_width = (sum.upper_bound - sum.lower_bound) / 2.0;
        assert!((actual_half_width - expected_half_width).abs() < 0.001);
    }

    #[test]
    fn test_quality_flags() {
        let mut flags = QualityFlags::empty();
        assert!(flags.is_clean());

        flags.set(QualityFlags::LOW_SNR);
        assert!(flags.contains(QualityFlags::LOW_SNR));
        assert!(!flags.contains(QualityFlags::CLIPPED));

        flags.set(QualityFlags::CLIPPED);
        assert!(flags.contains(QualityFlags::LOW_SNR));
        assert!(flags.contains(QualityFlags::CLIPPED));
    }
}
