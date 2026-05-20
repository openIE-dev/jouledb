//! DSP fundamentals — pure-Rust replacement for dsp.js, fft.js.
//!
//! FFT (Cooley-Tukey radix-2), IFFT, convolution, correlation, windowing functions
//! (Hamming, Hann, Blackman), FIR filter design, power spectral density, zero-crossing detection.

use std::f64::consts::PI;

// ── Complex for FFT ───────────────────────────────────────────

/// Minimal complex type for internal FFT use.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    pub fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }

    pub fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }
}

// ── FFT ───────────────────────────────────────────────────────

/// Next power of 2 >= n.
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 { return 1; }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Bit-reverse permutation for FFT.
fn bit_reverse(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// In-place Cooley-Tukey radix-2 FFT.
/// Input length must be a power of 2.
/// `inverse` = true computes IFFT.
fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");
    if n <= 1 { return; }

    bit_reverse(data);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wn = Complex::from_polar(1.0, angle);

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[i + j];
                let v = data[i + j + half].mul(w);
                data[i + j] = u.add(v);
                data[i + j + half] = u.sub(v);
                w = w.mul(wn);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for c in data.iter_mut() {
            *c = c.scale(scale);
        }
    }
}

/// Compute the FFT of a real-valued signal.
/// Pads to next power of 2 if needed.
pub fn fft(signal: &[f64]) -> Vec<Complex> {
    let n = next_power_of_two(signal.len());
    let mut data: Vec<Complex> = signal.iter()
        .map(|x| Complex::new(*x, 0.0))
        .chain(std::iter::repeat(Complex::ZERO))
        .take(n)
        .collect();
    fft_in_place(&mut data, false);
    data
}

/// Compute the FFT of complex input.
pub fn fft_complex(signal: &[Complex]) -> Vec<Complex> {
    let n = next_power_of_two(signal.len());
    let mut data: Vec<Complex> = signal.iter().copied()
        .chain(std::iter::repeat(Complex::ZERO))
        .take(n)
        .collect();
    fft_in_place(&mut data, false);
    data
}

/// Compute the inverse FFT.
pub fn ifft(spectrum: &[Complex]) -> Vec<Complex> {
    let n = next_power_of_two(spectrum.len());
    let mut data: Vec<Complex> = spectrum.iter().copied()
        .chain(std::iter::repeat(Complex::ZERO))
        .take(n)
        .collect();
    fft_in_place(&mut data, true);
    data
}

/// Extract magnitude spectrum from FFT output.
pub fn magnitude_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.magnitude()).collect()
}

/// Extract phase spectrum from FFT output.
pub fn phase_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.phase()).collect()
}

// ── Convolution ───────────────────────────────────────────────

/// Linear convolution via FFT.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_power_of_two(out_len);

    let mut fa: Vec<Complex> = a.iter().map(|x| Complex::new(*x, 0.0))
        .chain(std::iter::repeat(Complex::ZERO))
        .take(n)
        .collect();
    let mut fb: Vec<Complex> = b.iter().map(|x| Complex::new(*x, 0.0))
        .chain(std::iter::repeat(Complex::ZERO))
        .take(n)
        .collect();

    fft_in_place(&mut fa, false);
    fft_in_place(&mut fb, false);

    let mut product: Vec<Complex> = fa.iter().zip(fb.iter())
        .map(|(a_val, b_val)| a_val.mul(*b_val))
        .collect();

    fft_in_place(&mut product, true);

    product.iter().take(out_len).map(|c| c.re).collect()
}

/// Cross-correlation of two signals.
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    // Correlation is convolution with b reversed
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    convolve(a, &b_rev)
}

/// Autocorrelation (cross-correlation of signal with itself).
pub fn autocorrelate(signal: &[f64]) -> Vec<f64> {
    cross_correlate(signal, signal)
}

// ── Windowing functions ───────────────────────────────────────

/// Apply a window function to a signal in-place.
pub fn apply_window(signal: &mut [f64], window: &[f64]) {
    for (s, &w) in signal.iter_mut().zip(window.iter()) {
        *s *= w;
    }
}

/// Generate a Hamming window of length n.
pub fn hamming_window(n: usize) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1.0]; }
    (0..n).map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()).collect()
}

/// Generate a Hann window of length n.
pub fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1.0]; }
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())).collect()
}

/// Generate a Blackman window of length n.
pub fn blackman_window(n: usize) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1.0]; }
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;
    (0..n).map(|i| {
        let x = i as f64 / (n - 1) as f64;
        a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
    }).collect()
}

/// Generate a rectangular window of length n (all ones).
pub fn rectangular_window(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

// ── FIR Filter ────────────────────────────────────────────────

/// Design a low-pass FIR filter using the windowed sinc method.
/// `cutoff` is normalized frequency (0 to 1, where 1 = Nyquist).
/// `order` is the filter order (number of taps - 1, must be even).
pub fn design_lowpass_fir(cutoff: f64, order: usize) -> Vec<f64> {
    let order = if order % 2 != 0 { order + 1 } else { order };
    let n = order + 1;
    let mid = order / 2;

    let mut coeffs = vec![0.0; n];
    for i in 0..n {
        let k = i as f64 - mid as f64;
        if k.abs() < 1e-12 {
            coeffs[i] = cutoff;
        } else {
            coeffs[i] = (PI * cutoff * k).sin() / (PI * k);
        }
    }

    // Apply Hamming window
    let window = hamming_window(n);
    for (c, &w) in coeffs.iter_mut().zip(window.iter()) {
        *c *= w;
    }

    // Normalize so sum = 1
    let sum: f64 = coeffs.iter().sum();
    if sum.abs() > 1e-12 {
        for c in &mut coeffs {
            *c /= sum;
        }
    }

    coeffs
}

/// Design a high-pass FIR filter (spectral inversion of low-pass).
pub fn design_highpass_fir(cutoff: f64, order: usize) -> Vec<f64> {
    let mut coeffs = design_lowpass_fir(cutoff, order);
    // Spectral inversion
    for c in &mut coeffs {
        *c = -*c;
    }
    let mid = coeffs.len() / 2;
    coeffs[mid] += 1.0;
    coeffs
}

/// Apply an FIR filter to a signal (direct convolution).
pub fn apply_fir(signal: &[f64], coeffs: &[f64]) -> Vec<f64> {
    if signal.is_empty() || coeffs.is_empty() {
        return Vec::new();
    }
    let out_len = signal.len();
    let filter_len = coeffs.len();
    let mut output = vec![0.0; out_len];

    for i in 0..out_len {
        let mut sum = 0.0;
        for j in 0..filter_len {
            if i >= j {
                sum += signal[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }
    output
}

// ── Power spectral density ────────────────────────────────────

/// Estimate power spectral density using periodogram method.
pub fn psd(signal: &[f64]) -> Vec<f64> {
    let mut windowed = signal.to_vec();
    let window = hann_window(windowed.len());
    apply_window(&mut windowed, &window);

    let spectrum = fft(&windowed);
    let n = spectrum.len() as f64;

    // Compute |X(f)|^2 / N
    spectrum.iter().map(|c| c.magnitude_sq() / n).collect()
}

/// Estimate PSD using Welch's method (averaged periodograms with overlap).
pub fn psd_welch(signal: &[f64], segment_len: usize, overlap: usize) -> Vec<f64> {
    if signal.len() < segment_len {
        return psd(signal);
    }

    let step = segment_len - overlap;
    let fft_len = next_power_of_two(segment_len);
    let mut avg = vec![0.0; fft_len];
    let mut count = 0usize;
    let window = hann_window(segment_len);

    let mut start = 0;
    while start + segment_len <= signal.len() {
        let mut segment = signal[start..start + segment_len].to_vec();
        apply_window(&mut segment, &window);

        let spectrum = fft(&segment);
        let n = spectrum.len() as f64;
        for (i, c) in spectrum.iter().enumerate() {
            avg[i] += c.magnitude_sq() / n;
        }
        count += 1;
        start += step;
    }

    if count > 0 {
        for v in &mut avg {
            *v /= count as f64;
        }
    }
    avg
}

// ── Zero-crossing detection ───────────────────────────────────

/// Detect zero crossings in a signal. Returns indices where sign changes.
pub fn zero_crossings(signal: &[f64]) -> Vec<usize> {
    let mut crossings = Vec::new();
    for i in 1..signal.len() {
        if (signal[i - 1] > 0.0 && signal[i] <= 0.0)
            || (signal[i - 1] < 0.0 && signal[i] >= 0.0)
            || (signal[i - 1] == 0.0 && signal[i] != 0.0)
        {
            crossings.push(i);
        }
    }
    crossings
}

/// Estimate fundamental frequency from zero crossing rate.
/// `sample_rate` is in Hz.
pub fn zero_crossing_rate(signal: &[f64], sample_rate: f64) -> f64 {
    let crossings = zero_crossings(signal);
    if signal.len() < 2 { return 0.0; }
    crossings.len() as f64 * sample_rate / (2.0 * (signal.len() - 1) as f64)
}

// ── Signal generation (utility) ───────────────────────────────

/// Generate a sine wave.
pub fn generate_sine(frequency: f64, sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    (0..n).map(|i| {
        let t = i as f64 / sample_rate;
        (2.0 * PI * frequency * t).sin()
    }).collect()
}

/// Generate a cosine wave.
pub fn generate_cosine(frequency: f64, sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    (0..n).map(|i| {
        let t = i as f64 / sample_rate;
        (2.0 * PI * frequency * t).cos()
    }).collect()
}

/// RMS (root mean square) of a signal.
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty() { return 0.0; }
    let sum_sq: f64 = signal.iter().map(|x| x * x).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

/// Energy of a signal (sum of squares).
pub fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-8;

    #[test]
    fn next_power_of_two_values() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
    }

    #[test]
    fn fft_dc_signal() {
        let signal = vec![1.0; 8];
        let spectrum = fft(&signal);
        assert!((spectrum[0].re - 8.0).abs() < EPS);
        for i in 1..spectrum.len() {
            assert!(spectrum[i].magnitude() < EPS);
        }
    }

    #[test]
    fn fft_ifft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = fft(&signal);
        let recovered = ifft(&spectrum);
        for (i, &orig) in signal.iter().enumerate() {
            assert!((recovered[i].re - orig).abs() < EPS,
                "Mismatch at {}: got {} expected {}", i, recovered[i].re, orig);
        }
    }

    #[test]
    fn fft_sine_wave() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();
        let spectrum = fft(&signal);
        let mags = magnitude_spectrum(&spectrum);
        // Peak should be at bin 1 (and bin N-1 for negative frequency)
        let peak_idx = mags[..n / 2].iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, 1);
    }

    #[test]
    fn convolution_identity() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0];
        let result = convolve(&signal, &kernel);
        for (i, &v) in signal.iter().enumerate() {
            assert!((result[i] - v).abs() < EPS);
        }
    }

    #[test]
    fn convolution_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = convolve(&a, &b);
        // Expected: [0, 1, 2.5, 4, 1.5]
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < EPS);
        assert!((result[1] - 1.0).abs() < EPS);
        assert!((result[2] - 2.5).abs() < EPS);
        assert!((result[3] - 4.0).abs() < EPS);
        assert!((result[4] - 1.5).abs() < EPS);
    }

    #[test]
    fn cross_correlation_basic() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let result = cross_correlate(&a, &b);
        assert!((result[3] - 1.0).abs() < EPS); // peak at lag 0 (centered)
    }

    #[test]
    fn hamming_window_properties() {
        let w = hamming_window(8);
        assert_eq!(w.len(), 8);
        // Symmetric
        for i in 0..4 {
            assert!((w[i] - w[7 - i]).abs() < EPS);
        }
        // Endpoints should be 0.08
        assert!((w[0] - 0.08).abs() < 0.01);
    }

    #[test]
    fn hann_window_properties() {
        let w = hann_window(8);
        assert_eq!(w.len(), 8);
        assert!((w[0] - 0.0).abs() < EPS);
        assert!((w[7] - 0.0).abs() < EPS);
    }

    #[test]
    fn blackman_window_properties() {
        let w = blackman_window(8);
        assert_eq!(w.len(), 8);
        assert!((w[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn fir_lowpass() {
        let coeffs = design_lowpass_fir(0.5, 20);
        assert_eq!(coeffs.len(), 21);
        let sum: f64 = coeffs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn fir_highpass() {
        let coeffs = design_highpass_fir(0.5, 20);
        assert_eq!(coeffs.len(), 21);
    }

    #[test]
    fn apply_fir_filter() {
        let signal = vec![1.0; 100];
        let coeffs = design_lowpass_fir(0.5, 10);
        let filtered = apply_fir(&signal, &coeffs);
        assert_eq!(filtered.len(), 100);
        // After transient, output should be close to 1.0 for DC signal
        assert!((filtered[50] - 1.0).abs() < 0.01);
    }

    #[test]
    fn psd_basic() {
        let signal = generate_sine(10.0, 1000.0, 0.1);
        let spectrum = psd(&signal);
        assert!(!spectrum.is_empty());
        // Should have a peak somewhere
        let max_val = spectrum.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_val > 0.0);
    }

    #[test]
    fn zero_crossings_sine() {
        let signal = generate_sine(1.0, 100.0, 1.0);
        let crossings = zero_crossings(&signal);
        // 1 Hz sine should cross zero about 2 times per cycle
        assert!(crossings.len() >= 1);
    }

    #[test]
    fn zero_crossing_rate_test() {
        let signal = generate_sine(10.0, 1000.0, 1.0);
        let zcr = zero_crossing_rate(&signal, 1000.0);
        // Should be approximately the frequency
        assert!((zcr - 10.0).abs() < 1.0);
    }

    #[test]
    fn rms_and_energy() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        assert!((rms(&signal) - 1.0).abs() < EPS);
        assert!((energy(&signal) - 4.0).abs() < EPS);
    }

    #[test]
    fn generate_sine_wave() {
        let signal = generate_sine(1.0, 4.0, 1.0);
        assert_eq!(signal.len(), 4);
        assert!((signal[0] - 0.0).abs() < EPS);
        assert!((signal[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn psd_welch_basic() {
        let signal = generate_sine(50.0, 1000.0, 0.5);
        let spectrum = psd_welch(&signal, 128, 64);
        assert!(!spectrum.is_empty());
    }

    #[test]
    fn complex_operations() {
        let a = Complex::new(3.0, 4.0);
        assert!((a.magnitude() - 5.0).abs() < EPS);
        let b = Complex::new(1.0, 2.0);
        let product = a.mul(b);
        assert!((product.re - (-5.0)).abs() < EPS);
        assert!((product.im - 10.0).abs() < EPS);
    }
}
