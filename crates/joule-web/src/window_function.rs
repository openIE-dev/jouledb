//! Window functions for signal processing — pure Rust, no external dependencies.
//!
//! Provides standard window functions (rectangular, Hann, Hamming, Blackman,
//! Blackman-Harris, Kaiser, Gaussian, flat-top, Bartlett, Tukey), periodic
//! vs symmetric modes, main lobe / side lobe characterization, COLA verification,
//! ENBW computation, and window correction factors.

use std::f64::consts::PI;

// ── Window Type ─────────────────────────────────────────────────

/// Supported window function types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowKind {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Kaiser(f64),    // beta parameter
    Gaussian(f64),  // sigma parameter
    FlatTop,
    Bartlett,
    Tukey(f64),     // alpha parameter (0 = rectangular, 1 = Hann)
}

/// Mode: symmetric (for analysis) or periodic (for FFT).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowMode {
    Symmetric,
    Periodic,
}

// ── Error type ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum WindowError {
    InvalidSize(String),
    InvalidParameter(String),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSize(s) => write!(f, "invalid window size: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
        }
    }
}

// ── Bessel I0 (for Kaiser) ──────────────────────────────────────

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=30 {
        term *= (x / (2.0 * k as f64)) * (x / (2.0 * k as f64));
        sum += term;
        if term < 1e-15 { break; }
    }
    sum
}

// ── Core window generation ──────────────────────────────────────

/// Generate a window of the given kind, size, and mode.
pub fn generate(kind: WindowKind, size: usize, mode: WindowMode) -> Result<Vec<f64>, WindowError> {
    if size == 0 {
        return Err(WindowError::InvalidSize("size must be > 0".into()));
    }
    if size == 1 {
        return Ok(vec![1.0]);
    }

    // For periodic mode, generate size+1 symmetric then drop the last sample
    let gen_size = match mode {
        WindowMode::Symmetric => size,
        WindowMode::Periodic => size + 1,
    };

    let n = gen_size - 1; // denominator for symmetric formula
    let window = match kind {
        WindowKind::Rectangular => vec![1.0; gen_size],

        WindowKind::Hann => (0..gen_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos()))
            .collect(),

        WindowKind::Hamming => (0..gen_size)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / n as f64).cos())
            .collect(),

        WindowKind::Blackman => (0..gen_size)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / n as f64;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),

        WindowKind::BlackmanHarris => (0..gen_size)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / n as f64;
                0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos()
                    - 0.01168 * (3.0 * x).cos()
            })
            .collect(),

        WindowKind::Kaiser(beta) => {
            if beta < 0.0 {
                return Err(WindowError::InvalidParameter("Kaiser beta must be >= 0".into()));
            }
            let denom = bessel_i0(beta);
            (0..gen_size)
                .map(|i| {
                    let ratio = 2.0 * i as f64 / n as f64 - 1.0;
                    let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
                    bessel_i0(arg) / denom
                })
                .collect()
        }

        WindowKind::Gaussian(sigma) => {
            if sigma <= 0.0 {
                return Err(WindowError::InvalidParameter("Gaussian sigma must be > 0".into()));
            }
            let center = n as f64 / 2.0;
            (0..gen_size)
                .map(|i| {
                    let x = (i as f64 - center) / (sigma * center);
                    (-0.5 * x * x).exp()
                })
                .collect()
        }

        WindowKind::FlatTop => (0..gen_size)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / n as f64;
                1.0 - 1.93 * x.cos() + 1.29 * (2.0 * x).cos()
                    - 0.388 * (3.0 * x).cos() + 0.0322 * (4.0 * x).cos()
            })
            .collect(),

        WindowKind::Bartlett => (0..gen_size)
            .map(|i| {
                1.0 - (2.0 * i as f64 / n as f64 - 1.0).abs()
            })
            .collect(),

        WindowKind::Tukey(alpha) => {
            if alpha < 0.0 || alpha > 1.0 {
                return Err(WindowError::InvalidParameter("Tukey alpha must be in [0, 1]".into()));
            }
            (0..gen_size)
                .map(|i| {
                    let x = i as f64 / n as f64;
                    if alpha == 0.0 {
                        1.0
                    } else if x < alpha / 2.0 {
                        0.5 * (1.0 + (PI * (2.0 * x / alpha - 1.0)).cos())
                    } else if x > 1.0 - alpha / 2.0 {
                        0.5 * (1.0 + (PI * (2.0 * x / alpha - 2.0 / alpha + 1.0)).cos())
                    } else {
                        1.0
                    }
                })
                .collect()
        }
    };

    match mode {
        WindowMode::Symmetric => Ok(window),
        WindowMode::Periodic => Ok(window[..size].to_vec()),
    }
}

// ── Convenience constructors ────────────────────────────────────

pub fn rectangular(size: usize) -> Vec<f64> {
    generate(WindowKind::Rectangular, size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn hann(size: usize) -> Vec<f64> {
    generate(WindowKind::Hann, size, WindowMode::Periodic).unwrap_or_default()
}

pub fn hamming(size: usize) -> Vec<f64> {
    generate(WindowKind::Hamming, size, WindowMode::Periodic).unwrap_or_default()
}

pub fn blackman(size: usize) -> Vec<f64> {
    generate(WindowKind::Blackman, size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn blackman_harris(size: usize) -> Vec<f64> {
    generate(WindowKind::BlackmanHarris, size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn kaiser(size: usize, beta: f64) -> Vec<f64> {
    generate(WindowKind::Kaiser(beta), size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn gaussian(size: usize, sigma: f64) -> Vec<f64> {
    generate(WindowKind::Gaussian(sigma), size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn flat_top(size: usize) -> Vec<f64> {
    generate(WindowKind::FlatTop, size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn bartlett(size: usize) -> Vec<f64> {
    generate(WindowKind::Bartlett, size, WindowMode::Symmetric).unwrap_or_default()
}

pub fn tukey(size: usize, alpha: f64) -> Vec<f64> {
    generate(WindowKind::Tukey(alpha), size, WindowMode::Periodic).unwrap_or_default()
}

// ── Window characterization ─────────────────────────────────────

/// Compute the Equivalent Noise Bandwidth (ENBW) of a window.
/// ENBW = N * sum(w[n]^2) / (sum(w[n]))^2.
pub fn enbw(window: &[f64]) -> f64 {
    let n = window.len() as f64;
    let sum_sq: f64 = window.iter().map(|w| w * w).sum();
    let sum: f64 = window.iter().sum();
    if sum.abs() < 1e-12 { return 0.0; }
    n * sum_sq / (sum * sum)
}

/// Amplitude correction factor: 1 / mean(window).
pub fn amplitude_correction(window: &[f64]) -> f64 {
    let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
    if mean.abs() < 1e-12 { return 0.0; }
    1.0 / mean
}

/// Energy correction factor: sqrt(N / sum(w^2)).
pub fn energy_correction(window: &[f64]) -> f64 {
    let n = window.len() as f64;
    let sum_sq: f64 = window.iter().map(|w| w * w).sum();
    if sum_sq < 1e-12 { return 0.0; }
    (n / sum_sq).sqrt()
}

// ── COLA (Constant Overlap-Add) condition ───────────────────────

/// Check if a window satisfies the Constant Overlap-Add condition
/// for a given hop size. The sum of overlapping windows should be constant.
pub fn is_cola(window: &[f64], hop_size: usize) -> bool {
    if window.is_empty() || hop_size == 0 || hop_size > window.len() {
        return false;
    }
    let n = window.len();
    // Sum contributions over one period
    let mut sums = vec![0.0; hop_size];
    for (i, &w) in window.iter().enumerate() {
        sums[i % hop_size] += w;
    }
    // All sums should be equal (constant)
    let target = sums[0];
    sums.iter().all(|s| (s - target).abs() < 1e-6)
}

/// Find hop sizes that satisfy COLA for a given window, up to `max_hop`.
pub fn cola_hop_sizes(window: &[f64], max_hop: usize) -> Vec<usize> {
    (1..=max_hop.min(window.len()))
        .filter(|hop| is_cola(window, *hop))
        .collect()
}

// ── Main lobe width estimation ──────────────────────────────────

/// Estimate the -3dB main lobe width in bins by computing the DFT of the window.
/// Returns the width in frequency bins (approximate).
pub fn main_lobe_width(window: &[f64]) -> f64 {
    if window.is_empty() { return 0.0; }
    let n = window.len();
    let fft_size = {
        let mut p = 1;
        while p < n * 4 { p <<= 1; }
        p
    };

    // Compute magnitude spectrum via DFT
    let mut max_mag = 0.0_f64;
    let mut mags = Vec::with_capacity(fft_size / 2 + 1);
    for k in 0..=fft_size / 2 {
        let omega = 2.0 * PI * k as f64 / fft_size as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &w) in window.iter().enumerate() {
            re += w * (omega * i as f64).cos();
            im -= w * (omega * i as f64).sin();
        }
        let mag = (re * re + im * im).sqrt();
        max_mag = max_mag.max(mag);
        mags.push(mag);
    }

    if max_mag < 1e-12 { return 0.0; }

    // -3dB threshold
    let threshold = max_mag * 0.5_f64.sqrt();
    let mut width_bins = 0;
    for &m in &mags {
        if m >= threshold {
            width_bins += 1;
        } else {
            break;
        }
    }

    // Convert to original-resolution bins
    2.0 * width_bins as f64 * n as f64 / fft_size as f64
}

/// Estimate peak side lobe level in dB relative to main lobe.
pub fn side_lobe_level_db(window: &[f64]) -> f64 {
    if window.is_empty() { return 0.0; }
    let n = window.len();
    let fft_size = {
        let mut p = 1;
        while p < n * 4 { p <<= 1; }
        p
    };

    let mut mags = Vec::with_capacity(fft_size / 2 + 1);
    for k in 0..=fft_size / 2 {
        let omega = 2.0 * PI * k as f64 / fft_size as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &w) in window.iter().enumerate() {
            re += w * (omega * i as f64).cos();
            im -= w * (omega * i as f64).sin();
        }
        mags.push((re * re + im * im).sqrt());
    }

    let main_peak = mags[0];
    if main_peak < 1e-12 { return 0.0; }

    // Find where main lobe ends (first local minimum)
    let mut main_end = 1;
    for k in 1..mags.len() - 1 {
        if mags[k] <= mags[k - 1] && mags[k] <= mags[k + 1] {
            main_end = k;
            break;
        }
    }

    // Find peak side lobe
    let side_peak = mags[main_end..]
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    if side_peak < 1e-12 {
        return -200.0; // effectively no side lobes
    }
    20.0 * (side_peak / main_peak).log10()
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
    fn test_rectangular() {
        let w = rectangular(8);
        assert_eq!(w.len(), 8);
        assert!(w.iter().all(|v| approx_eq(*v, 1.0)));
    }

    #[test]
    fn test_hann_endpoints() {
        let w = hann(16);
        assert!(approx_eq(w[0], 0.0));
        // Periodic Hann: last sample is near-zero but not exactly zero
        // (the zero falls at the hypothetical index N, outside the window).
        assert!(w[15] < 0.04, "w[15] = {}", w[15]);
        // Peak at center
        assert!(w[8] > 0.99);
    }

    #[test]
    fn test_hamming_endpoints() {
        let w = hamming(16);
        assert!(w[0] > 0.07); // Hamming doesn't reach zero
        assert!(w[8] > 0.99);
    }

    #[test]
    fn test_blackman_endpoints() {
        let w = blackman(32);
        assert!(w[0].abs() < 0.01);
        assert!(w[16] > 0.99);
    }

    #[test]
    fn test_blackman_harris() {
        let w = blackman_harris(32);
        assert_eq!(w.len(), 32);
        assert!(w[16] > 0.99);
    }

    #[test]
    fn test_kaiser_beta_zero() {
        // Kaiser with beta=0 → rectangular
        let w = kaiser(8, 0.0);
        for &v in &w {
            assert!(approx_eq(v, 1.0));
        }
    }

    #[test]
    fn test_kaiser_symmetric() {
        let w = kaiser(16, 5.0);
        for i in 0..8 {
            assert!(approx_eq(w[i], w[15 - i]));
        }
    }

    #[test]
    fn test_gaussian() {
        let w = gaussian(16, 0.4);
        assert_eq!(w.len(), 16);
        assert!(w[8] > w[0]); // center > edge
    }

    #[test]
    fn test_flat_top() {
        let w = flat_top(32);
        assert_eq!(w.len(), 32);
        // Flat-top peaks above 1.0 at center for amplitude accuracy
        assert!(w[16] > 0.9);
    }

    #[test]
    fn test_bartlett_triangle() {
        let w = bartlett(9);
        assert!(approx_eq(w[0], 0.0));
        assert!(approx_eq(w[4], 1.0));
        assert!(approx_eq(w[8], 0.0));
    }

    #[test]
    fn test_tukey_alpha_zero() {
        // alpha=0 → rectangular
        let w = tukey(16, 0.0);
        for &v in &w {
            assert!(approx_eq(v, 1.0));
        }
    }

    #[test]
    fn test_tukey_alpha_one() {
        // alpha=1 → Hann
        let w_tukey = tukey(16, 1.0);
        let w_hann = hann(16);
        for (a, b) in w_tukey.iter().zip(w_hann.iter()) {
            assert!((a - b).abs() < 0.05, "tukey(1.0) vs hann: {a} vs {b}");
        }
    }

    #[test]
    fn test_periodic_mode() {
        let sym = generate(WindowKind::Hann, 8, WindowMode::Symmetric).unwrap();
        let per = generate(WindowKind::Hann, 8, WindowMode::Periodic).unwrap();
        assert_eq!(sym.len(), 8);
        assert_eq!(per.len(), 8);
        // Periodic should differ from symmetric at endpoints
        // Specifically, periodic Hann has w[N-1] != 0
        assert!(per[7] > 0.01); // periodic doesn't end at zero
    }

    #[test]
    fn test_window_size_one() {
        let w = generate(WindowKind::Hann, 1, WindowMode::Symmetric).unwrap();
        assert_eq!(w, vec![1.0]);
    }

    #[test]
    fn test_window_size_zero() {
        assert!(generate(WindowKind::Hann, 0, WindowMode::Symmetric).is_err());
    }

    #[test]
    fn test_enbw_rectangular() {
        let w = rectangular(64);
        let e = enbw(&w);
        assert!(approx_eq(e, 1.0), "ENBW rectangular = {e}");
    }

    #[test]
    fn test_enbw_hann() {
        let w = hann(64);
        let e = enbw(&w);
        // Hann ENBW ≈ 1.5 bins
        assert!((e - 1.5).abs() < 0.1, "ENBW Hann = {e}");
    }

    #[test]
    fn test_amplitude_correction() {
        let w = hann(64);
        let ac = amplitude_correction(&w);
        // Hann amplitude correction ≈ 2.0
        assert!((ac - 2.0).abs() < 0.2, "amplitude correction = {ac}");
    }

    #[test]
    fn test_energy_correction() {
        let w = rectangular(64);
        let ec = energy_correction(&w);
        assert!(approx_eq(ec, 1.0), "energy correction rectangular = {ec}");
    }

    #[test]
    fn test_cola_hann_50_percent() {
        // Hann window with 50% overlap (hop = N/2) satisfies COLA
        let w = hann(8);
        assert!(is_cola(&w, 4), "Hann COLA with hop=N/2");
    }

    #[test]
    fn test_cola_rectangular() {
        // Rectangular with hop=N satisfies COLA (no overlap)
        let w = rectangular(8);
        assert!(is_cola(&w, 8));
    }

    #[test]
    fn test_cola_invalid() {
        let w = hann(8);
        assert!(!is_cola(&w, 0));
        assert!(!is_cola(&[], 4));
    }

    #[test]
    fn test_cola_hop_sizes() {
        let w = hann(8);
        let hops = cola_hop_sizes(&w, 8);
        assert!(hops.contains(&4), "hops = {:?}", hops);
    }

    #[test]
    fn test_main_lobe_width_rectangular() {
        let w = rectangular(32);
        let mlw = main_lobe_width(&w);
        // Rectangular main lobe is ~2 bins wide at -3dB
        assert!(mlw > 0.5 && mlw < 4.0, "rectangular MLW = {mlw}");
    }

    #[test]
    fn test_side_lobe_level_blackman() {
        let w = blackman(64);
        let sll = side_lobe_level_db(&w);
        // Blackman: ~-58 dB side lobe
        assert!(sll < -40.0, "Blackman SLL = {sll} dB");
    }

    #[test]
    fn test_side_lobe_level_rectangular() {
        let w = rectangular(64);
        let sll = side_lobe_level_db(&w);
        // Rectangular: ~-13 dB side lobe
        assert!(sll > -20.0 && sll < -10.0, "Rectangular SLL = {sll} dB");
    }

    #[test]
    fn test_invalid_kaiser_beta() {
        assert!(generate(WindowKind::Kaiser(-1.0), 8, WindowMode::Symmetric).is_err());
    }

    #[test]
    fn test_invalid_gaussian_sigma() {
        assert!(generate(WindowKind::Gaussian(0.0), 8, WindowMode::Symmetric).is_err());
    }

    #[test]
    fn test_invalid_tukey_alpha() {
        assert!(generate(WindowKind::Tukey(-0.1), 8, WindowMode::Symmetric).is_err());
        assert!(generate(WindowKind::Tukey(1.5), 8, WindowMode::Symmetric).is_err());
    }
}
