//! Fast Fourier Transform engine — pure Rust, no external dependencies.
//!
//! Implements the Cooley-Tukey radix-2 decimation-in-time algorithm for
//! power-of-2 length signals. Provides forward/inverse FFT, real-to-complex
//! half-spectrum optimization, magnitude/phase extraction, zero-padding,
//! FFT-based convolution, and Parseval's theorem verification.

use std::f64::consts::PI;

// ── Complex Number ──────────────────────────────────────────────

/// A complex number with `re` (real) and `im` (imaginary) parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn from_polar(mag: f64, phase: f64) -> Self {
        Self {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self { re: self.re * rhs, im: self.im * rhs }
    }
}

// ── Utility helpers ─────────────────────────────────────────────

/// Returns `true` if `n` is a power of two.
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Returns the next power of two >= `n`.
pub fn next_power_of_two(n: usize) -> usize {
    if is_power_of_two(n) {
        return n;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Bit-reversal permutation index for `i` given `log2_n` bits.
fn bit_reverse(i: usize, log2_n: u32) -> usize {
    let mut result = 0usize;
    let mut val = i;
    for _ in 0..log2_n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Applies bit-reversal permutation in-place.
fn bit_reversal_permute(data: &mut [Complex]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let log2_n = (n as f64).log2() as u32;
    for i in 0..n {
        let j = bit_reverse(i, log2_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

// ── FFT Core ────────────────────────────────────────────────────

/// Configuration for the FFT engine.
#[derive(Debug, Clone, PartialEq)]
pub struct FftConfig {
    /// Length of the FFT (must be power of 2).
    pub size: usize,
}

impl FftConfig {
    pub fn new(size: usize) -> Result<Self, FftError> {
        if !is_power_of_two(size) {
            return Err(FftError::NotPowerOfTwo(size));
        }
        Ok(Self { size })
    }
}

/// Errors from FFT operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FftError {
    NotPowerOfTwo(usize),
    LengthMismatch { expected: usize, got: usize },
    EmptyInput,
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPowerOfTwo(n) => write!(f, "FFT size {n} is not a power of 2"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected length {expected}, got {got}")
            }
            Self::EmptyInput => write!(f, "input is empty"),
        }
    }
}

/// Performs in-place Cooley-Tukey radix-2 DIT FFT.
/// `inverse` = true computes the IFFT (with 1/N scaling).
fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    bit_reversal_permute(data);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * 2.0 * PI / len as f64;

        for start in (0..n).step_by(len) {
            for k in 0..half {
                let twiddle = Complex::from_polar(1.0, angle * k as f64);
                let u = data[start + k];
                let v = data[start + k + half] * twiddle;
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for sample in data.iter_mut() {
            *sample = *sample * scale;
        }
    }
}

/// Compute the forward FFT of a complex signal.
pub fn fft(input: &[Complex]) -> Result<Vec<Complex>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    if !is_power_of_two(input.len()) {
        return Err(FftError::NotPowerOfTwo(input.len()));
    }
    let mut data = input.to_vec();
    fft_in_place(&mut data, false);
    Ok(data)
}

/// Compute the inverse FFT of a complex spectrum.
pub fn ifft(input: &[Complex]) -> Result<Vec<Complex>, FftError> {
    if input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    if !is_power_of_two(input.len()) {
        return Err(FftError::NotPowerOfTwo(input.len()));
    }
    let mut data = input.to_vec();
    fft_in_place(&mut data, true);
    Ok(data)
}

/// Real-to-complex FFT with half-spectrum optimization.
/// Input: `N` real samples (N must be power of 2).
/// Output: `N/2 + 1` complex bins (the non-redundant half).
pub fn rfft(real_input: &[f64]) -> Result<Vec<Complex>, FftError> {
    if real_input.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let n = real_input.len();
    if !is_power_of_two(n) {
        return Err(FftError::NotPowerOfTwo(n));
    }
    let complex_input: Vec<Complex> = real_input
        .iter()
        .map(|r| Complex::new(*r, 0.0))
        .collect();
    let full = fft(&complex_input)?;
    Ok(full[..n / 2 + 1].to_vec())
}

// ── Extraction helpers ──────────────────────────────────────────

/// Extract magnitude spectrum from complex FFT output.
pub fn magnitude_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.magnitude()).collect()
}

/// Extract phase spectrum (radians) from complex FFT output.
pub fn phase_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.phase()).collect()
}

/// Extract power spectrum (magnitude squared) from complex FFT output.
pub fn power_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.magnitude_squared()).collect()
}

// ── Zero-padding ────────────────────────────────────────────────

/// Zero-pad a real signal to the next power of two.
pub fn zero_pad_real(signal: &[f64]) -> Vec<f64> {
    let target = next_power_of_two(signal.len());
    let mut padded = signal.to_vec();
    padded.resize(target, 0.0);
    padded
}

/// Zero-pad a complex signal to the next power of two.
pub fn zero_pad_complex(signal: &[Complex]) -> Vec<Complex> {
    let target = next_power_of_two(signal.len());
    let mut padded = signal.to_vec();
    padded.resize(target, Complex::zero());
    padded
}

/// Zero-pad a real signal to exactly `target_len` (must be power of 2).
pub fn zero_pad_to(signal: &[f64], target_len: usize) -> Result<Vec<f64>, FftError> {
    if !is_power_of_two(target_len) {
        return Err(FftError::NotPowerOfTwo(target_len));
    }
    let mut padded = signal.to_vec();
    padded.resize(target_len, 0.0);
    Ok(padded)
}

// ── FFT-based convolution ───────────────────────────────────────

/// FFT-based linear convolution of two real signals.
/// Returns a vector of length `a.len() + b.len() - 1` (may be padded).
pub fn fft_convolve(a: &[f64], b: &[f64]) -> Result<Vec<f64>, FftError> {
    if a.is_empty() || b.is_empty() {
        return Err(FftError::EmptyInput);
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_power_of_two(out_len);

    let mut ca: Vec<Complex> = a.iter().map(|v| Complex::new(*v, 0.0)).collect();
    ca.resize(n, Complex::zero());
    let mut cb: Vec<Complex> = b.iter().map(|v| Complex::new(*v, 0.0)).collect();
    cb.resize(n, Complex::zero());

    let fa = fft(&ca)?;
    let fb = fft(&cb)?;

    let product: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(x, y)| *x * *y).collect();

    let result = ifft(&product)?;
    Ok(result.iter().take(out_len).map(|c| c.re).collect())
}

// ── Parseval's theorem ──────────────────────────────────────────

/// Verify Parseval's theorem: sum(|x[n]|^2) == (1/N) * sum(|X[k]|^2).
/// Returns `(time_energy, freq_energy)`.
pub fn parseval_check(signal: &[Complex]) -> Result<(f64, f64), FftError> {
    let spectrum = fft(signal)?;
    let n = signal.len() as f64;
    let time_energy: f64 = signal.iter().map(|c| c.magnitude_squared()).sum();
    let freq_energy: f64 = spectrum.iter().map(|c| c.magnitude_squared()).sum::<f64>() / n;
    Ok((time_energy, freq_energy))
}

// ── FFT Engine (stateful) ───────────────────────────────────────

/// Stateful FFT engine that pre-computes twiddle factors for a fixed size.
#[derive(Debug, Clone)]
pub struct FftEngine {
    size: usize,
    twiddles_fwd: Vec<Complex>,
    twiddles_inv: Vec<Complex>,
}

impl FftEngine {
    /// Create an engine for the given FFT `size` (must be power of 2).
    pub fn new(size: usize) -> Result<Self, FftError> {
        if !is_power_of_two(size) {
            return Err(FftError::NotPowerOfTwo(size));
        }
        let mut twiddles_fwd = Vec::with_capacity(size / 2);
        let mut twiddles_inv = Vec::with_capacity(size / 2);
        for k in 0..size / 2 {
            let angle = -2.0 * PI * k as f64 / size as f64;
            twiddles_fwd.push(Complex::from_polar(1.0, angle));
            twiddles_inv.push(Complex::from_polar(1.0, -angle));
        }
        Ok(Self { size, twiddles_fwd, twiddles_inv })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Forward FFT using pre-computed twiddles.
    pub fn forward(&self, input: &[Complex]) -> Result<Vec<Complex>, FftError> {
        if input.len() != self.size {
            return Err(FftError::LengthMismatch { expected: self.size, got: input.len() });
        }
        let mut data = input.to_vec();
        self.fft_core(&mut data, &self.twiddles_fwd, false);
        Ok(data)
    }

    /// Inverse FFT using pre-computed twiddles.
    pub fn inverse(&self, input: &[Complex]) -> Result<Vec<Complex>, FftError> {
        if input.len() != self.size {
            return Err(FftError::LengthMismatch { expected: self.size, got: input.len() });
        }
        let mut data = input.to_vec();
        self.fft_core(&mut data, &self.twiddles_inv, true);
        Ok(data)
    }

    fn fft_core(&self, data: &mut [Complex], twiddles: &[Complex], inverse: bool) {
        let n = data.len();
        bit_reversal_permute(data);

        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let tw = twiddles[k * step];
                    let u = data[start + k];
                    let v = data[start + k + half] * tw;
                    data[start + k] = u + v;
                    data[start + k + half] = u - v;
                }
            }
            len <<= 1;
        }

        if inverse {
            let scale = 1.0 / n as f64;
            for sample in data.iter_mut() {
                *sample = *sample * scale;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn complex_approx_eq(a: &Complex, b: &Complex) -> bool {
        approx_eq(a.re, b.re) && approx_eq(a.im, b.im)
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(6));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(100), 128);
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        let sum = a + b;
        assert!(approx_eq(sum.re, 4.0));
        assert!(approx_eq(sum.im, 2.0));
        let diff = a - b;
        assert!(approx_eq(diff.re, 2.0));
        assert!(approx_eq(diff.im, 6.0));
        let prod = a * b;
        // (3+4i)(1-2i) = 3 -6i +4i -8i^2 = 11 -2i
        assert!(approx_eq(prod.re, 11.0));
        assert!(approx_eq(prod.im, -2.0));
    }

    #[test]
    fn test_complex_magnitude_phase() {
        let c = Complex::new(3.0, 4.0);
        assert!(approx_eq(c.magnitude(), 5.0));
        let c2 = Complex::new(1.0, 0.0);
        assert!(approx_eq(c2.phase(), 0.0));
        let c3 = Complex::new(0.0, 1.0);
        assert!(approx_eq(c3.phase(), PI / 2.0));
    }

    #[test]
    fn test_complex_polar() {
        let c = Complex::from_polar(5.0, PI / 4.0);
        assert!(approx_eq(c.magnitude(), 5.0));
        assert!(approx_eq(c.phase(), PI / 4.0));
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex::new(3.0, 4.0);
        let conj = c.conjugate();
        assert!(approx_eq(conj.re, 3.0));
        assert!(approx_eq(conj.im, -4.0));
    }

    #[test]
    fn test_fft_dc_signal() {
        // All ones → DC component = N, rest zero
        let n = 8;
        let input: Vec<Complex> = vec![Complex::new(1.0, 0.0); n];
        let result = fft(&input).unwrap();
        assert!(approx_eq(result[0].re, 8.0));
        assert!(approx_eq(result[0].im, 0.0));
        for k in 1..n {
            assert!(approx_eq(result[k].magnitude(), 0.0));
        }
    }

    #[test]
    fn test_fft_single_frequency() {
        // cos(2*pi*k0*n/N) at k0=1 should peak at bins 1 and N-1
        let n = 8;
        let input: Vec<Complex> = (0..n)
            .map(|i| {
                let val = (2.0 * PI * i as f64 / n as f64).cos();
                Complex::new(val, 0.0)
            })
            .collect();
        let result = fft(&input).unwrap();
        assert!(result[1].magnitude() > 3.0); // should be N/2 = 4
        assert!(result[7].magnitude() > 3.0); // mirror
        for k in [0, 2, 3, 4, 5, 6] {
            assert!(result[k].magnitude() < EPS);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.0, 3.0),
            Complex::new(-1.0, 2.0),
        ];
        let spectrum = fft(&input).unwrap();
        let recovered = ifft(&spectrum).unwrap();
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(a, b));
        }
    }

    #[test]
    fn test_fft_not_power_of_two_error() {
        let input = vec![Complex::zero(); 3];
        assert_eq!(fft(&input), Err(FftError::NotPowerOfTwo(3)));
    }

    #[test]
    fn test_fft_empty_error() {
        let input: Vec<Complex> = vec![];
        assert_eq!(fft(&input), Err(FftError::EmptyInput));
    }

    #[test]
    fn test_rfft_real_cosine() {
        let n = 16;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).cos())
            .collect();
        let result = rfft(&input).unwrap();
        assert_eq!(result.len(), n / 2 + 1);
        // Peak should be at bin 2
        let mags: Vec<f64> = result.iter().map(|c| c.magnitude()).collect();
        assert!(mags[2] > 7.0);
        for (i, &m) in mags.iter().enumerate() {
            if i != 2 {
                assert!(m < EPS, "bin {i} magnitude = {m}");
            }
        }
    }

    #[test]
    fn test_magnitude_spectrum() {
        let spec = vec![Complex::new(3.0, 4.0), Complex::new(0.0, 1.0)];
        let mags = magnitude_spectrum(&spec);
        assert!(approx_eq(mags[0], 5.0));
        assert!(approx_eq(mags[1], 1.0));
    }

    #[test]
    fn test_phase_spectrum() {
        let spec = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let phases = phase_spectrum(&spec);
        assert!(approx_eq(phases[0], 0.0));
        assert!(approx_eq(phases[1], PI / 2.0));
    }

    #[test]
    fn test_power_spectrum() {
        let spec = vec![Complex::new(3.0, 4.0)];
        let pows = power_spectrum(&spec);
        assert!(approx_eq(pows[0], 25.0));
    }

    #[test]
    fn test_zero_pad_real() {
        let sig = vec![1.0, 2.0, 3.0];
        let padded = zero_pad_real(&sig);
        assert_eq!(padded.len(), 4);
        assert!(approx_eq(padded[0], 1.0));
        assert!(approx_eq(padded[3], 0.0));
    }

    #[test]
    fn test_zero_pad_complex() {
        let sig = vec![Complex::new(1.0, 0.0); 5];
        let padded = zero_pad_complex(&sig);
        assert_eq!(padded.len(), 8);
        assert!(complex_approx_eq(&padded[4], &Complex::new(1.0, 0.0)));
        assert!(complex_approx_eq(&padded[7], &Complex::zero()));
    }

    #[test]
    fn test_zero_pad_to() {
        let sig = vec![1.0, 2.0];
        let padded = zero_pad_to(&sig, 8).unwrap();
        assert_eq!(padded.len(), 8);
        assert!(approx_eq(padded[0], 1.0));
        assert!(approx_eq(padded[7], 0.0));
        assert!(zero_pad_to(&sig, 7).is_err());
    }

    #[test]
    fn test_fft_convolve_impulse() {
        // Convolve with impulse → same signal
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0];
        let result = fft_convolve(&a, &b).unwrap();
        assert_eq!(result.len(), 4);
        for (i, &v) in result.iter().enumerate() {
            assert!(approx_eq(v, a[i]));
        }
    }

    #[test]
    fn test_fft_convolve_known() {
        // [1,2,3] * [1,1] = [1,3,5,3]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = fft_convolve(&a, &b).unwrap();
        let expected = vec![1.0, 3.0, 5.0, 3.0];
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*r, *e));
        }
    }

    #[test]
    fn test_parseval_theorem() {
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(-1.0, 0.5),
            Complex::new(0.3, -0.7),
            Complex::new(2.0, 1.0),
        ];
        let (time_e, freq_e) = parseval_check(&signal).unwrap();
        assert!(
            (time_e - freq_e).abs() < 1e-10,
            "Parseval's: time={time_e}, freq={freq_e}"
        );
    }

    #[test]
    fn test_fft_engine_roundtrip() {
        let engine = FftEngine::new(8).unwrap();
        let input: Vec<Complex> = (0..8)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let spectrum = engine.forward(&input).unwrap();
        let recovered = engine.inverse(&spectrum).unwrap();
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(a, b));
        }
    }

    #[test]
    fn test_fft_engine_size_mismatch() {
        let engine = FftEngine::new(8).unwrap();
        let input = vec![Complex::zero(); 4];
        assert!(engine.forward(&input).is_err());
    }

    #[test]
    fn test_fft_engine_invalid_size() {
        assert!(FftEngine::new(7).is_err());
    }

    #[test]
    fn test_fft_linearity() {
        // FFT(a*x + b*y) == a*FFT(x) + b*FFT(y)
        let x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let y: Vec<Complex> = (0..8).map(|i| Complex::new(0.0, i as f64 * 0.5)).collect();
        let a = 2.0;
        let b = 3.0;
        let combined: Vec<Complex> = x.iter().zip(y.iter()).map(|(xi, yi)| *xi * a + *yi * b).collect();
        let fft_combined = fft(&combined).unwrap();
        let fft_x = fft(&x).unwrap();
        let fft_y = fft(&y).unwrap();
        for k in 0..8 {
            let expected = fft_x[k] * a + fft_y[k] * b;
            assert!(complex_approx_eq(&fft_combined[k], &expected));
        }
    }

    #[test]
    fn test_bit_reversal_permutation() {
        // For N=8: 0→0, 1→4, 2→2, 3→6, 4→1, 5→5, 6→3, 7→7
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
    }

    #[test]
    fn test_fft_size_one() {
        let input = vec![Complex::new(42.0, 7.0)];
        let result = fft(&input).unwrap();
        assert!(complex_approx_eq(&result[0], &input[0]));
    }
}
