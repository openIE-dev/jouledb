//! Bandpass and band-reject filters — pure Rust, no external dependencies.
//!
//! FIR bandpass filter design using windowed-sinc method with selectable
//! window functions (Hamming, Blackman, Kaiser). Band-reject via spectral
//! inversion. Multi-band filters. Frequency response computation. Group delay.

use std::f64::consts::PI;

// ── Window Functions for Filter Design ──────────────────────────

/// Window function type for filter design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterWindow {
    Hamming,
    Blackman,
    Kaiser(f64), // beta parameter
}

fn hamming_window(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0]; }
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

fn blackman_window(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0]; }
    (0..n)
        .map(|i| {
            let x = 2.0 * PI * i as f64 / (n - 1) as f64;
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect()
}

/// Modified Bessel function I0 (zeroth order, first kind).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=25 {
        term *= (x / (2.0 * k as f64)) * (x / (2.0 * k as f64));
        sum += term;
        if term < 1e-15 { break; }
    }
    sum
}

fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n <= 1 { return vec![1.0]; }
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let ratio = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

fn make_filter_window(win: FilterWindow, n: usize) -> Vec<f64> {
    match win {
        FilterWindow::Hamming => hamming_window(n),
        FilterWindow::Blackman => blackman_window(n),
        FilterWindow::Kaiser(beta) => kaiser_window(n, beta),
    }
}

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BandpassError {
    InvalidFrequency(String),
    InvalidOrder(String),
    EmptyInput,
}

impl std::fmt::Display for BandpassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFrequency(s) => write!(f, "invalid frequency: {s}"),
            Self::InvalidOrder(s) => write!(f, "invalid order: {s}"),
            Self::EmptyInput => write!(f, "input signal is empty"),
        }
    }
}

// ── Low-pass sinc kernel ────────────────────────────────────────

/// Design a low-pass FIR filter using windowed sinc.
/// `cutoff`: normalized cutoff frequency (0..0.5, fraction of sample rate).
/// `order`: filter order (number of taps - 1; must be even for symmetric).
fn lowpass_sinc(cutoff: f64, order: usize, window: FilterWindow) -> Vec<f64> {
    let n = order + 1;
    let mid = order as f64 / 2.0;
    let win = make_filter_window(window, n);
    let mut kernel: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - mid;
            let sinc = if x.abs() < 1e-12 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * x).sin() / (PI * x)
            };
            sinc * win[i]
        })
        .collect();

    // Normalize to unity gain at DC
    let sum: f64 = kernel.iter().sum();
    if sum.abs() > 1e-12 {
        for k in kernel.iter_mut() {
            *k /= sum;
        }
    }
    kernel
}

// ── Bandpass ────────────────────────────────────────────────────

/// Design a bandpass FIR filter.
///
/// * `f_low`: lower cutoff (normalized, 0..0.5)
/// * `f_high`: upper cutoff (normalized, 0..0.5)
/// * `order`: filter order (even recommended)
/// * `window`: window function
pub fn design_bandpass(
    f_low: f64,
    f_high: f64,
    order: usize,
    window: FilterWindow,
) -> Result<Vec<f64>, BandpassError> {
    if f_low <= 0.0 || f_high >= 0.5 || f_low >= f_high {
        return Err(BandpassError::InvalidFrequency(format!(
            "need 0 < f_low < f_high < 0.5, got {f_low}, {f_high}"
        )));
    }
    if order < 2 {
        return Err(BandpassError::InvalidOrder("order must be >= 2".into()));
    }

    // Bandpass = high_pass - low_pass (difference of two low-pass kernels)
    let lp_high = lowpass_sinc(f_high, order, window);
    let lp_low = lowpass_sinc(f_low, order, window);

    let bp: Vec<f64> = lp_high.iter().zip(lp_low.iter()).map(|(h, l)| h - l).collect();
    Ok(bp)
}

/// Design a bandpass filter using center frequency and bandwidth.
pub fn design_bandpass_center(
    center: f64,
    bandwidth: f64,
    order: usize,
    window: FilterWindow,
) -> Result<Vec<f64>, BandpassError> {
    let f_low = center - bandwidth / 2.0;
    let f_high = center + bandwidth / 2.0;
    design_bandpass(f_low, f_high, order, window)
}

// ── Band-reject (notch) ────────────────────────────────────────

/// Design a band-reject (notch) FIR filter via spectral inversion.
pub fn design_band_reject(
    f_low: f64,
    f_high: f64,
    order: usize,
    window: FilterWindow,
) -> Result<Vec<f64>, BandpassError> {
    let bp = design_bandpass(f_low, f_high, order, window)?;
    spectral_invert(&bp)
}

/// Spectral inversion: converts bandpass to band-reject (and vice versa).
fn spectral_invert(kernel: &[f64]) -> Result<Vec<f64>, BandpassError> {
    if kernel.is_empty() {
        return Err(BandpassError::EmptyInput);
    }
    let mid = kernel.len() / 2;
    let mut inverted: Vec<f64> = kernel.iter().map(|k| -k).collect();
    inverted[mid] += 1.0;
    Ok(inverted)
}

// ── Multi-band filter ───────────────────────────────────────────

/// Band specification: (low_freq, high_freq) normalized.
#[derive(Debug, Clone, PartialEq)]
pub struct BandSpec {
    pub f_low: f64,
    pub f_high: f64,
}

/// Design a multi-band filter by summing individual bandpass filters.
pub fn design_multiband(
    bands: &[BandSpec],
    order: usize,
    window: FilterWindow,
) -> Result<Vec<f64>, BandpassError> {
    if bands.is_empty() {
        return Err(BandpassError::EmptyInput);
    }
    let n = order + 1;
    let mut combined = vec![0.0; n];
    for band in bands {
        let bp = design_bandpass(band.f_low, band.f_high, order, window)?;
        for (i, &v) in bp.iter().enumerate() {
            combined[i] += v;
        }
    }
    Ok(combined)
}

// ── Apply filter (convolution) ──────────────────────────────────

/// Apply an FIR filter to a signal (direct convolution).
pub fn apply_filter(signal: &[f64], kernel: &[f64]) -> Result<Vec<f64>, BandpassError> {
    if signal.is_empty() || kernel.is_empty() {
        return Err(BandpassError::EmptyInput);
    }
    let out_len = signal.len() + kernel.len() - 1;
    let mut output = vec![0.0; out_len];
    for (i, &s) in signal.iter().enumerate() {
        for (j, &k) in kernel.iter().enumerate() {
            output[i + j] += s * k;
        }
    }
    Ok(output)
}

/// Apply filter with "same" mode: output length = input length (centered).
pub fn apply_filter_same(signal: &[f64], kernel: &[f64]) -> Result<Vec<f64>, BandpassError> {
    let full = apply_filter(signal, kernel)?;
    let offset = kernel.len() / 2;
    Ok(full[offset..offset + signal.len()].to_vec())
}

// ── Frequency response ──────────────────────────────────────────

/// Complex frequency response (magnitude, phase) at `n_points` frequencies.
/// Returns `(frequencies_normalized, magnitudes, phases)`.
pub fn frequency_response(kernel: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_points);
    let mut mags = Vec::with_capacity(n_points);
    let mut phases = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let f = 0.5 * i as f64 / (n_points - 1).max(1) as f64;
        let omega = 2.0 * PI * f;
        let mut real = 0.0;
        let mut imag = 0.0;
        for (k, &h) in kernel.iter().enumerate() {
            real += h * (omega * k as f64).cos();
            imag -= h * (omega * k as f64).sin();
        }
        freqs.push(f);
        mags.push((real * real + imag * imag).sqrt());
        phases.push(imag.atan2(real));
    }
    (freqs, mags, phases)
}

// ── Group delay ─────────────────────────────────────────────────

/// Compute group delay of an FIR filter at `n_points` frequencies.
/// For a linear-phase filter, this is constant = (order) / 2.
pub fn group_delay(kernel: &[f64], n_points: usize) -> Vec<f64> {
    let n = kernel.len();
    let mut delays = Vec::with_capacity(n_points);

    // Weighted kernel: n * h[n]
    let weighted: Vec<f64> = kernel.iter().enumerate().map(|(i, &h)| i as f64 * h).collect();

    for i in 0..n_points {
        let f = 0.5 * i as f64 / (n_points - 1).max(1) as f64;
        let omega = 2.0 * PI * f;

        let mut hr = 0.0;
        let mut hi = 0.0;
        let mut wr = 0.0;
        let mut wi = 0.0;

        for k in 0..n {
            let cos_val = (omega * k as f64).cos();
            let sin_val = (omega * k as f64).sin();
            hr += kernel[k] * cos_val;
            hi -= kernel[k] * sin_val;
            wr += weighted[k] * cos_val;
            wi -= weighted[k] * sin_val;
        }

        let denom = hr * hr + hi * hi;
        if denom > 1e-12 {
            // group delay = Re{ (wr + j*wi) * conj(hr + j*hi) } / |H|^2
            let gd = (wr * hr + wi * hi) / denom;
            delays.push(gd);
        } else {
            delays.push(0.0);
        }
    }
    delays
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_hamming_window_endpoints() {
        let w = hamming_window(64);
        assert_eq!(w.len(), 64);
        assert!(w[0] > 0.07); // Hamming doesn't reach zero
        assert!(w[32] > 0.9); // peak near center
    }

    #[test]
    fn test_blackman_window_endpoints() {
        let w = blackman_window(64);
        assert!(w[0].abs() < 0.01); // near zero at edges
        assert!(w[32] > 0.9);
    }

    #[test]
    fn test_kaiser_window_endpoints() {
        let w = kaiser_window(64, 5.0);
        assert!(w[0] > 0.0);
        assert!(w[32] > w[0]); // center > edge
    }

    #[test]
    fn test_bessel_i0_at_zero() {
        assert!(approx_eq(bessel_i0(0.0), 1.0));
    }

    #[test]
    fn test_lowpass_sinc_dc_gain() {
        let kernel = lowpass_sinc(0.25, 32, FilterWindow::Hamming);
        let sum: f64 = kernel.iter().sum();
        assert!(approx_eq(sum, 1.0), "DC gain = {sum}");
    }

    #[test]
    fn test_design_bandpass_valid() {
        let bp = design_bandpass(0.1, 0.3, 32, FilterWindow::Hamming).unwrap();
        assert_eq!(bp.len(), 33);
    }

    #[test]
    fn test_design_bandpass_invalid_freq() {
        assert!(design_bandpass(0.3, 0.1, 32, FilterWindow::Hamming).is_err());
        assert!(design_bandpass(-0.1, 0.3, 32, FilterWindow::Hamming).is_err());
        assert!(design_bandpass(0.1, 0.6, 32, FilterWindow::Hamming).is_err());
    }

    #[test]
    fn test_design_bandpass_invalid_order() {
        assert!(design_bandpass(0.1, 0.3, 1, FilterWindow::Hamming).is_err());
    }

    #[test]
    fn test_design_bandpass_center() {
        let bp = design_bandpass_center(0.2, 0.1, 32, FilterWindow::Blackman).unwrap();
        assert_eq!(bp.len(), 33);
    }

    #[test]
    fn test_bandpass_passes_in_band() {
        let bp = design_bandpass(0.15, 0.35, 64, FilterWindow::Hamming).unwrap();
        let (_, mags, _) = frequency_response(&bp, 256);
        // Check that passband (around normalized 0.25) has good gain
        let mid_bin = 128; // 0.25 normalized
        assert!(mags[mid_bin] > 0.5, "passband gain = {}", mags[mid_bin]);
    }

    #[test]
    fn test_bandpass_rejects_out_of_band() {
        let bp = design_bandpass(0.2, 0.3, 64, FilterWindow::Blackman).unwrap();
        let (_, mags, _) = frequency_response(&bp, 256);
        // DC (bin 0) should be attenuated
        assert!(mags[0] < 0.1, "stopband DC gain = {}", mags[0]);
    }

    #[test]
    fn test_band_reject() {
        let br = design_band_reject(0.15, 0.35, 64, FilterWindow::Hamming).unwrap();
        let (_, mags, _) = frequency_response(&br, 256);
        // DC should pass
        assert!(mags[0] > 0.5, "DC gain = {}", mags[0]);
        // Notch region should be attenuated
        let notch_bin = 128; // ~0.25
        assert!(mags[notch_bin] < 0.5, "notch gain = {}", mags[notch_bin]);
    }

    #[test]
    fn test_multiband() {
        let bands = vec![
            BandSpec { f_low: 0.05, f_high: 0.15 },
            BandSpec { f_low: 0.30, f_high: 0.45 },
        ];
        let mb = design_multiband(&bands, 64, FilterWindow::Hamming).unwrap();
        assert_eq!(mb.len(), 65);
    }

    #[test]
    fn test_multiband_empty_bands() {
        assert!(design_multiband(&[], 64, FilterWindow::Hamming).is_err());
    }

    #[test]
    fn test_apply_filter_impulse() {
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let kernel = vec![1.0, 2.0, 3.0];
        let out = apply_filter(&signal, &kernel).unwrap();
        assert_eq!(out.len(), 6);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 2.0));
        assert!(approx_eq(out[2], 3.0));
        assert!(approx_eq(out[3], 0.0));
    }

    #[test]
    fn test_apply_filter_same() {
        let signal = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let kernel = vec![1.0, 2.0, 1.0];
        let out = apply_filter_same(&signal, &kernel).unwrap();
        assert_eq!(out.len(), signal.len());
    }

    #[test]
    fn test_apply_filter_empty() {
        assert!(apply_filter(&[], &[1.0]).is_err());
        assert!(apply_filter(&[1.0], &[]).is_err());
    }

    #[test]
    fn test_frequency_response_length() {
        let kernel = lowpass_sinc(0.2, 16, FilterWindow::Hamming);
        let (freqs, mags, phases) = frequency_response(&kernel, 64);
        assert_eq!(freqs.len(), 64);
        assert_eq!(mags.len(), 64);
        assert_eq!(phases.len(), 64);
    }

    #[test]
    fn test_group_delay_linear_phase() {
        // Symmetric FIR → constant group delay = order/2
        let kernel = lowpass_sinc(0.2, 32, FilterWindow::Hamming);
        let gd = group_delay(&kernel, 128);
        let expected = 32.0 / 2.0;
        // Interior points (away from band edges) should be close
        for &d in &gd[10..110] {
            assert!(
                (d - expected).abs() < 1.0,
                "group delay {d} far from expected {expected}"
            );
        }
    }

    #[test]
    fn test_spectral_inversion_roundtrip() {
        let bp = design_bandpass(0.1, 0.3, 32, FilterWindow::Hamming).unwrap();
        let br = spectral_invert(&bp).unwrap();
        let bp_again = spectral_invert(&br).unwrap();
        for (a, b) in bp.iter().zip(bp_again.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_kaiser_beta_effect() {
        // Higher beta → narrower main lobe, more sidelobe suppression
        let w_low = kaiser_window(32, 2.0);
        let w_high = kaiser_window(32, 8.0);
        // At the edge (index 0), higher beta should be smaller
        assert!(w_high[0] < w_low[0]);
    }

    #[test]
    fn test_frequency_response_dc_lowpass() {
        let kernel = lowpass_sinc(0.2, 32, FilterWindow::Hamming);
        let (_, mags, _) = frequency_response(&kernel, 256);
        // DC gain should be ~1.0
        assert!(approx_eq(mags[0], 1.0), "DC gain = {}", mags[0]);
    }
}
