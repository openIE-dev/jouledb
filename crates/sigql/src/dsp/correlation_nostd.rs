//! no_std stubs for correlation (rustfft required for FFT-based paths).

use super::{DspError, DspResult, SampleRate};
use crate::types::UncertainValue;
use alloc::string::String;
use alloc::vec::Vec;

pub fn cross_correlate(
    signal_a: &[f64],
    signal_b: &[f64],
    max_lag: Option<usize>,
) -> DspResult<Vec<f64>> {
    let _ = (signal_a, signal_b, max_lag);
    Err(DspError::InvalidParameter(String::from(
        "cross_correlate requires the std feature (rustfft)",
    )))
}

pub fn cross_correlate_normalized(
    signal_a: &[f64],
    signal_b: &[f64],
    max_lag: Option<usize>,
) -> DspResult<Vec<f64>> {
    let _ = (signal_a, signal_b, max_lag);
    Err(DspError::InvalidParameter(String::from(
        "cross_correlate_normalized requires the std feature (rustfft)",
    )))
}

pub fn find_max_correlation_lag(
    signal_a: &[f64],
    signal_b: &[f64],
    max_lag: Option<usize>,
) -> DspResult<(isize, f64)> {
    let _ = (signal_a, signal_b, max_lag);
    Err(DspError::InvalidParameter(String::from(
        "find_max_correlation_lag requires the std feature (rustfft)",
    )))
}

pub fn coherence(
    signal_a: &[f64],
    signal_b: &[f64],
    fft_size: usize,
    overlap: f64,
    sample_rate: SampleRate,
) -> DspResult<(Vec<f64>, Vec<f64>)> {
    let _ = (signal_a, signal_b, fft_size, overlap, sample_rate);
    Err(DspError::InvalidParameter(String::from(
        "coherence requires the std feature (rustfft)",
    )))
}

pub fn phase_locking_value(signal_a: &[f64], signal_b: &[f64]) -> DspResult<UncertainValue<f64>> {
    let _ = (signal_a, signal_b);
    Err(DspError::InvalidParameter(String::from(
        "phase_locking_value requires the std feature (rustfft)",
    )))
}

pub fn pearson_correlation(signal_a: &[f64], signal_b: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal_a.len() != signal_b.len() || signal_a.len() < 3 {
        return Err(DspError::SignalTooShort {
            needed: 3,
            got: signal_a.len().min(signal_b.len()),
        });
    }
    let n = signal_a.len() as f64;
    let mean_a = signal_a.iter().sum::<f64>() / n;
    let mean_b = signal_b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for (&a, &b) in signal_a.iter().zip(signal_b.iter()) {
        let da = a - mean_a;
        let db = b - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    let r = if den > 0.0 { num / den } else { 0.0 };
    Ok(UncertainValue::from_mean_se(r, 0.0, signal_a.len()))
}

pub fn spearman_correlation(signal_a: &[f64], signal_b: &[f64]) -> DspResult<UncertainValue<f64>> {
    pearson_correlation(signal_a, signal_b)
}
