//! Digital Signal Processing Core
//!
//! This module provides the computational backend for SigQL transforms.
//! All DSP operations are implemented here and called by the runtime.

pub mod correlation;
pub mod envelope;
pub mod fft;
pub mod filter;
pub mod resample;
pub mod statistics;
pub mod window;

use crate::types::{DynSignal, UncertainValue};
use num_complex::Complex64;

/// Sample rate as a simple wrapper (matches types module usage)
pub type SampleRate = crate::types::SampleRate;

/// Result type for DSP operations
pub type DspResult<T> = Result<T, DspError>;

/// DSP operation errors
#[derive(Debug, thiserror::Error)]
pub enum DspError {
    #[error("Signal too short: need {needed} samples, got {got}")]
    SignalTooShort { needed: usize, got: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Sample rate mismatch: {0}Hz vs {1}Hz")]
    SampleRateMismatch(u32, u32),

    #[error("FFT size must be power of 2, got {0}")]
    InvalidFftSize(usize),

    #[error("Filter design failed: {0}")]
    FilterDesignFailed(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Common trait for DSP operations
pub trait DspOperation {
    /// Apply the operation to a signal
    fn apply(&self, signal: &DynSignal<f64>) -> DspResult<DynSignal<f64>>;

    /// Get the latency in samples (for causal operations)
    fn latency_samples(&self) -> usize {
        0
    }

    /// Whether the operation changes sample rate
    fn output_sample_rate(&self, input_rate: SampleRate) -> SampleRate {
        input_rate
    }
}

/// Trait for frequency-domain operations
pub trait SpectralOperation {
    /// Apply operation and return spectrum
    fn apply_spectral(&self, signal: &DynSignal<f64>) -> DspResult<Vec<Complex64>>;
}

/// Trait for aggregation operations
pub trait AggregateOperation {
    /// Compute aggregate with uncertainty
    fn aggregate(&self, signal: &DynSignal<f64>) -> DspResult<UncertainValue<f64>>;
}

// Re-exports
pub use correlation::{coherence, cross_correlate, phase_locking_value};
pub use envelope::{EnvelopeExtractor, HilbertTransform};
pub use fft::{Fft, Ifft, Stft, StftParams};
pub use filter::{BiquadCoeffs, BiquadFilter, FilterDesign, FilterType};
pub use resample::{ResampleMethod, Resampler};
pub use statistics::{compute_kurtosis, compute_mean, compute_rms, compute_std};
pub use window::{WindowType, apply_window};
