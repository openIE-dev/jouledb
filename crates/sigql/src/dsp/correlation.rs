//! Cross-signal correlation and coherence operations

use super::fft::{Fft, WindowType};
use super::{DspError, DspResult};
use crate::types::{SampleRate, UncertainValue};
use num_complex::Complex64;

/// Compute cross-correlation between two signals
pub fn cross_correlate(
    signal_a: &[f64],
    signal_b: &[f64],
    max_lag: Option<usize>,
) -> DspResult<Vec<f64>> {
    if signal_a.is_empty() || signal_b.is_empty() {
        return Err(DspError::SignalTooShort { needed: 1, got: 0 });
    }

    let n = signal_a.len().max(signal_b.len());
    let max_lag = max_lag.unwrap_or(n - 1);

    // Use FFT-based correlation for efficiency
    let fft_size = (2 * n - 1).next_power_of_two();

    // Pad signals and convert to complex
    let mut a_complex: Vec<Complex64> = signal_a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    a_complex.resize(fft_size, Complex64::new(0.0, 0.0));

    let mut b_complex: Vec<Complex64> = signal_b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    b_complex.resize(fft_size, Complex64::new(0.0, 0.0));

    // Compute FFT of both using rustfft directly (no normalization on forward)
    let mut planner = rustfft::FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(fft_size);
    fft_forward.process(&mut a_complex);
    fft_forward.process(&mut b_complex);

    // Cross-spectrum: A * conj(B)
    let mut cross: Vec<Complex64> = a_complex
        .iter()
        .zip(b_complex.iter())
        .map(|(a, b)| a * b.conj())
        .collect();

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut cross);

    // Extract real part and normalize by N (standard IFFT normalization)
    let norm = 1.0 / fft_size as f64;
    let corr: Vec<f64> = cross.iter().map(|c| c.re * norm).collect();

    // Rearrange: output[0] = lag 0, output[1] = lag 1, etc.
    // For negative lags, they're at the end of the FFT output
    let mut result = Vec::with_capacity(2 * max_lag + 1);

    // Negative lags
    for i in (fft_size - max_lag)..fft_size {
        result.push(corr[i]);
    }
    // Zero and positive lags
    for i in 0..=max_lag.min(fft_size - 1) {
        result.push(corr[i]);
    }

    Ok(result)
}

/// Compute normalized cross-correlation (Pearson)
pub fn cross_correlate_normalized(
    signal_a: &[f64],
    signal_b: &[f64],
    max_lag: Option<usize>,
) -> DspResult<Vec<f64>> {
    let corr = cross_correlate(signal_a, signal_b, max_lag)?;

    // Normalize by energy
    let energy_a: f64 = signal_a.iter().map(|x| x * x).sum();
    let energy_b: f64 = signal_b.iter().map(|x| x * x).sum();
    let norm = (energy_a * energy_b).sqrt();

    if norm < 1e-10 {
        return Ok(vec![0.0; corr.len()]);
    }

    Ok(corr.iter().map(|c| c / norm).collect())
}

/// Find the lag of maximum correlation
pub fn find_max_correlation_lag(
    signal_a: &[f64],
    signal_b: &[f64],
    max_lag: Option<usize>,
) -> DspResult<(isize, f64)> {
    let corr = cross_correlate_normalized(signal_a, signal_b, max_lag)?;

    let max_lag_samples = max_lag.unwrap_or(signal_a.len().max(signal_b.len()) - 1);

    let (max_idx, max_val) = corr
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    // Convert index to lag
    let lag = max_idx as isize - max_lag_samples as isize;

    Ok((lag, *max_val))
}

/// Compute magnitude-squared coherence between two signals
pub fn coherence(
    signal_a: &[f64],
    signal_b: &[f64],
    fft_size: usize,
    overlap: f64,
    sample_rate: SampleRate,
) -> DspResult<(Vec<f64>, Vec<f64>)> {
    if signal_a.len() != signal_b.len() {
        return Err(DspError::SampleRateMismatch(
            signal_a.len() as u32,
            signal_b.len() as u32,
        ));
    }

    if signal_a.len() < fft_size {
        return Err(DspError::SignalTooShort {
            needed: fft_size,
            got: signal_a.len(),
        });
    }

    let hop = ((1.0 - overlap) * fft_size as f64) as usize;
    let hop = hop.max(1);
    let num_segments = (signal_a.len() - fft_size) / hop + 1;

    if num_segments == 0 {
        return Err(DspError::SignalTooShort {
            needed: fft_size,
            got: signal_a.len(),
        });
    }

    let fft = Fft::with_window(fft_size, WindowType::Hann)?;
    let num_bins = fft_size / 2 + 1;

    // Accumulators for Welch's method
    let mut psd_aa = vec![0.0; num_bins];
    let mut psd_bb = vec![0.0; num_bins];
    let mut csd_ab = vec![Complex64::new(0.0, 0.0); num_bins];

    for seg in 0..num_segments {
        let start = seg * hop;
        let end = start + fft_size;

        let a_fft = fft.compute(&signal_a[start..end])?;
        let b_fft = fft.compute(&signal_b[start..end])?;

        for i in 0..num_bins {
            psd_aa[i] += a_fft[i].norm_sqr();
            psd_bb[i] += b_fft[i].norm_sqr();
            csd_ab[i] += a_fft[i] * b_fft[i].conj();
        }
    }

    // Compute coherence: |Sab|^2 / (Saa * Sbb)
    let mut coh = Vec::with_capacity(num_bins);
    for i in 0..num_bins {
        let denom = psd_aa[i] * psd_bb[i];
        if denom > 1e-20 {
            coh.push(csd_ab[i].norm_sqr() / denom);
        } else {
            coh.push(0.0);
        }
    }

    // Frequency axis
    let df = sample_rate.0 as f64 / fft_size as f64;
    let freqs: Vec<f64> = (0..num_bins).map(|i| i as f64 * df).collect();

    Ok((freqs, coh))
}

/// Compute phase-locking value (PLV)
pub fn phase_locking_value(signal_a: &[f64], signal_b: &[f64]) -> DspResult<UncertainValue<f64>> {
    use super::envelope::HilbertTransform;

    if signal_a.len() != signal_b.len() {
        return Err(DspError::SampleRateMismatch(
            signal_a.len() as u32,
            signal_b.len() as u32,
        ));
    }

    let n = signal_a.len().next_power_of_two();

    // Pad signals
    let mut a_padded = signal_a.to_vec();
    a_padded.resize(n, 0.0);
    let mut b_padded = signal_b.to_vec();
    b_padded.resize(n, 0.0);

    let hilbert = HilbertTransform::new(n)?;

    // Get instantaneous phases
    let phase_a = hilbert.instantaneous_phase(&a_padded)?;
    let phase_b = hilbert.instantaneous_phase(&b_padded)?;

    // Compute PLV: |mean(exp(i*(phase_a - phase_b)))|
    let n_orig = signal_a.len();
    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;

    for i in 0..n_orig {
        let diff = phase_a[i] - phase_b[i];
        sum_real += diff.cos();
        sum_imag += diff.sin();
    }

    let plv = ((sum_real / n_orig as f64).powi(2) + (sum_imag / n_orig as f64).powi(2)).sqrt();

    // Standard error (approximate)
    let std_error = ((1.0 - plv.powi(2)) / (2.0 * n_orig as f64)).sqrt();

    Ok(UncertainValue::from_mean_se(plv, std_error, n_orig))
}

/// Compute Pearson correlation coefficient
pub fn pearson_correlation(signal_a: &[f64], signal_b: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal_a.len() != signal_b.len() {
        return Err(DspError::SampleRateMismatch(
            signal_a.len() as u32,
            signal_b.len() as u32,
        ));
    }

    if signal_a.len() < 3 {
        return Err(DspError::SignalTooShort {
            needed: 3,
            got: signal_a.len(),
        });
    }

    let n = signal_a.len() as f64;

    let mean_a = signal_a.iter().sum::<f64>() / n;
    let mean_b = signal_b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (a, b) in signal_a.iter().zip(signal_b.iter()) {
        let da = a - mean_a;
        let db = b - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return Ok(UncertainValue::from_mean_se(0.0, 0.0, signal_a.len()));
    }

    let r = cov / denom;

    // Fisher transformation for standard error
    let _fisher_z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let se_z = 1.0 / (n - 3.0).sqrt();

    // Back-transform to correlation space (approximate)
    let se_r = se_z * (1.0 - r.powi(2));

    Ok(UncertainValue::from_mean_se(r, se_r, signal_a.len()))
}

/// Compute Spearman rank correlation
pub fn spearman_correlation(signal_a: &[f64], signal_b: &[f64]) -> DspResult<UncertainValue<f64>> {
    if signal_a.len() != signal_b.len() {
        return Err(DspError::SampleRateMismatch(
            signal_a.len() as u32,
            signal_b.len() as u32,
        ));
    }

    // Convert to ranks
    let ranks_a = compute_ranks(signal_a);
    let ranks_b = compute_ranks(signal_b);

    // Pearson on ranks
    pearson_correlation(&ranks_a, &ranks_b)
}

/// Compute ranks for a signal (average ranks for ties)
fn compute_ranks(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();

    // Create sorted indices
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| signal[a].partial_cmp(&signal[b]).unwrap());

    // Assign ranks (1-based, average for ties)
    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        // Find ties
        while j < n - 1 && signal[indices[j]] == signal[indices[j + 1]] {
            j += 1;
        }

        // Average rank for tied values
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indices[k]] = avg_rank;
        }

        i = j + 1;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_autocorrelation() {
        let signal: Vec<f64> = (0..100).map(|i| (0.1 * i as f64).sin()).collect();
        let corr = cross_correlate_normalized(&signal, &signal, Some(50)).unwrap();

        // Auto-correlation at lag 0 should be 1.0
        let center = corr.len() / 2;
        assert_relative_eq!(corr[center], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_pearson_perfect() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let r = pearson_correlation(&a, &b).unwrap();
        assert_relative_eq!(r.value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation

        let r = pearson_correlation(&a, &b).unwrap();
        assert_relative_eq!(r.value, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coherence_identical() {
        // Use a broadband signal (noise-like) so all frequencies have power
        let mut signal = vec![0.0; 1024];
        // Simple PRNG for reproducibility
        let mut state = 12345u64;
        for i in 0..1024 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            signal[i] = ((state >> 16) as i32 as f64) / 32768.0;
        }

        let (_freqs, coh) = coherence(&signal, &signal, 256, 0.5, SampleRate::new(1000)).unwrap();

        // Coherence with itself should be 1.0 at all frequencies
        // (may not be exactly 1.0 due to windowing effects, but should be very close)
        assert!(
            coh.iter().all(|&c| c > 0.99),
            "Min coherence: {}",
            coh.iter().cloned().fold(f64::INFINITY, f64::min)
        );
    }

    #[test]
    fn test_spearman() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 3.5, 4.5, 5.5]; // Monotonic

        let r = spearman_correlation(&a, &b).unwrap();
        assert_relative_eq!(r.value, 1.0, epsilon = 1e-10);
    }
}
