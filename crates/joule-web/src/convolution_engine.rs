//! Convolution operations for signals — pure Rust, no external dependencies.
//!
//! Provides linear convolution (direct and FFT-based), circular convolution,
//! cross-correlation, auto-correlation, overlap-add and overlap-save for
//! streaming, Wiener deconvolution, matched filtering, convolution reverb,
//! and multi-channel convolution.

use std::f64::consts::PI;

// ── Mini FFT ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Cpx { re: f64, im: f64 }

impl Cpx {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    fn polar(r: f64, t: f64) -> Self { Self { re: r * t.cos(), im: r * t.sin() } }
    fn mag_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
    fn conj(&self) -> Self { Self { re: self.re, im: -self.im } }
}

impl std::ops::Add for Cpx {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { re: self.re + r.re, im: self.im + r.im } }
}
impl std::ops::Sub for Cpx {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { re: self.re - r.re, im: self.im - r.im } }
}
impl std::ops::Mul for Cpx {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self { re: self.re * r.re - self.im * r.im, im: self.re * r.im + self.im * r.re }
    }
}
impl std::ops::Mul<f64> for Cpx {
    type Output = Self;
    fn mul(self, r: f64) -> Self { Self { re: self.re * r, im: self.im * r } }
}
impl std::ops::Div for Cpx {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        let d = r.re * r.re + r.im * r.im;
        Self { re: (self.re * r.re + self.im * r.im) / d, im: (self.im * r.re - self.re * r.im) / d }
    }
}

fn next_pow2(n: usize) -> usize { let mut p = 1; while p < n { p <<= 1; } p }

fn bit_rev(i: usize, bits: u32) -> usize {
    let mut r = 0; let mut v = i;
    for _ in 0..bits { r = (r << 1) | (v & 1); v >>= 1; }
    r
}

fn fft_ip(data: &mut [Cpx], inv: bool) {
    let n = data.len();
    if n <= 1 { return; }
    let lg = (n as f64).log2() as u32;
    for i in 0..n { let j = bit_rev(i, lg); if i < j { data.swap(i, j); } }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let sign = if inv { 1.0 } else { -1.0 };
        let ang = sign * 2.0 * PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..half {
                let tw = Cpx::polar(1.0, ang * k as f64);
                let u = data[s + k];
                let v = data[s + k + half] * tw;
                data[s + k] = u + v;
                data[s + k + half] = u - v;
            }
        }
        len <<= 1;
    }
    if inv { let s = 1.0 / n as f64; for d in data.iter_mut() { *d = *d * s; } }
}

// ── Error type ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ConvError {
    EmptyInput,
    LengthMismatch(String),
    InvalidParameter(String),
}

impl std::fmt::Display for ConvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input is empty"),
            Self::LengthMismatch(s) => write!(f, "length mismatch: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
        }
    }
}

// ── Linear convolution (direct) ─────────────────────────────────

/// Direct linear convolution, O(N*M).
/// Output length = `a.len() + b.len() - 1`.
pub fn convolve_direct(a: &[f64], b: &[f64]) -> Result<Vec<f64>, ConvError> {
    if a.is_empty() || b.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    let out_len = a.len() + b.len() - 1;
    let mut output = vec![0.0; out_len];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            output[i + j] += av * bv;
        }
    }
    Ok(output)
}

// ── FFT-based convolution ───────────────────────────────────────

/// FFT-based linear convolution, O(N log N).
pub fn convolve_fft(a: &[f64], b: &[f64]) -> Result<Vec<f64>, ConvError> {
    if a.is_empty() || b.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_pow2(out_len);

    let mut fa: Vec<Cpx> = a.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    fa.resize(n, Cpx::zero());
    let mut fb: Vec<Cpx> = b.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    fb.resize(n, Cpx::zero());

    fft_ip(&mut fa, false);
    fft_ip(&mut fb, false);

    let mut product: Vec<Cpx> = fa.iter().zip(fb.iter()).map(|(x, y)| *x * *y).collect();
    fft_ip(&mut product, true);

    Ok(product.iter().take(out_len).map(|c| c.re).collect())
}

// ── Circular convolution ────────────────────────────────────────

/// Circular (periodic) convolution. Both signals must be the same length.
pub fn convolve_circular(a: &[f64], b: &[f64]) -> Result<Vec<f64>, ConvError> {
    if a.is_empty() || b.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(ConvError::LengthMismatch(format!(
            "a.len()={} != b.len()={}",
            a.len(),
            b.len()
        )));
    }
    let n_orig = a.len();
    let n = next_pow2(n_orig);

    let mut fa: Vec<Cpx> = a.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    fa.resize(n, Cpx::zero());
    let mut fb: Vec<Cpx> = b.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    fb.resize(n, Cpx::zero());

    fft_ip(&mut fa, false);
    fft_ip(&mut fb, false);

    let mut product: Vec<Cpx> = fa.iter().zip(fb.iter()).map(|(x, y)| *x * *y).collect();
    fft_ip(&mut product, true);

    // Fold back to original length (circular wrap)
    let mut result = vec![0.0; n_orig];
    for (i, c) in product.iter().enumerate() {
        result[i % n_orig] += c.re;
    }
    Ok(result)
}

// ── Cross-correlation ───────────────────────────────────────────

/// Cross-correlation of two signals.
/// R_xy[k] = sum(x[n] * y[n+k]) for lag k.
/// Output length = `a.len() + b.len() - 1`, centered at index `b.len() - 1` (zero lag).
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Result<Vec<f64>, ConvError> {
    if a.is_empty() || b.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    // Cross-correlation = convolution with time-reversed b
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    convolve_fft(a, &b_rev)
}

// ── Auto-correlation ────────────────────────────────────────────

/// Auto-correlation of a signal: cross_correlate(x, x).
pub fn auto_correlate(signal: &[f64]) -> Result<Vec<f64>, ConvError> {
    cross_correlate(signal, signal)
}

// ── Overlap-Add (streaming) ─────────────────────────────────────

/// Streaming convolution state using overlap-add.
#[derive(Debug, Clone)]
pub struct OverlapAdd {
    kernel_fft: Vec<Cpx>,
    kernel_len: usize,
    block_size: usize,
    overlap_buf: Vec<f64>,
}

impl OverlapAdd {
    /// Create an overlap-add processor for the given kernel.
    pub fn new(kernel: &[f64]) -> Result<Self, ConvError> {
        if kernel.is_empty() {
            return Err(ConvError::EmptyInput);
        }
        let m = kernel.len();
        let block_size = next_pow2(m * 2);

        let mut kfft: Vec<Cpx> = kernel.iter().map(|v| Cpx::new(*v, 0.0)).collect();
        kfft.resize(block_size, Cpx::zero());
        fft_ip(&mut kfft, false);

        Ok(Self {
            kernel_fft: kfft,
            kernel_len: m,
            block_size,
            overlap_buf: vec![0.0; m - 1],
        })
    }

    /// Process one block of input. Output length = input length.
    pub fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut buf: Vec<Cpx> = input.iter().map(|v| Cpx::new(*v, 0.0)).collect();
        buf.resize(self.block_size, Cpx::zero());

        fft_ip(&mut buf, false);
        for (b, &k) in buf.iter_mut().zip(self.kernel_fft.iter()) {
            *b = *b * k;
        }
        fft_ip(&mut buf, true);

        let conv_len = n + self.kernel_len - 1;
        let mut result = vec![0.0; conv_len.min(buf.len())];
        for (i, r) in result.iter_mut().enumerate() {
            *r = buf[i].re;
        }

        // Add overlap from previous block
        let overlap_add_len = self.overlap_buf.len().min(result.len());
        for i in 0..overlap_add_len {
            result[i] += self.overlap_buf[i];
        }

        // Save new overlap
        let new_overlap_start = n;
        self.overlap_buf.fill(0.0);
        for i in 0..self.overlap_buf.len() {
            if new_overlap_start + i < result.len() {
                self.overlap_buf[i] = result[new_overlap_start + i];
            }
        }

        result.truncate(n);
        result
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.overlap_buf.fill(0.0);
    }
}

// ── Overlap-Save (streaming) ────────────────────────────────────

/// Streaming convolution state using overlap-save.
#[derive(Debug, Clone)]
pub struct OverlapSave {
    kernel_fft: Vec<Cpx>,
    kernel_len: usize,
    block_size: usize,
    prev_tail: Vec<f64>,
}

impl OverlapSave {
    pub fn new(kernel: &[f64]) -> Result<Self, ConvError> {
        if kernel.is_empty() {
            return Err(ConvError::EmptyInput);
        }
        let m = kernel.len();
        let block_size = next_pow2(m * 4);

        let mut kfft: Vec<Cpx> = kernel.iter().map(|v| Cpx::new(*v, 0.0)).collect();
        kfft.resize(block_size, Cpx::zero());
        fft_ip(&mut kfft, false);

        Ok(Self {
            kernel_fft: kfft,
            kernel_len: m,
            block_size,
            prev_tail: vec![0.0; m - 1],
        })
    }

    /// Process one block. Input length should be >= kernel_len.
    pub fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        // Prepend previous tail
        let mut extended = self.prev_tail.clone();
        extended.extend_from_slice(input);

        let mut buf: Vec<Cpx> = extended.iter().map(|v| Cpx::new(*v, 0.0)).collect();
        buf.resize(self.block_size, Cpx::zero());

        fft_ip(&mut buf, false);
        for (b, &k) in buf.iter_mut().zip(self.kernel_fft.iter()) {
            *b = *b * k;
        }
        fft_ip(&mut buf, true);

        // Discard first kernel_len-1 samples (overlap-save)
        let skip = self.kernel_len - 1;
        let useful: Vec<f64> = buf[skip..skip + input.len()]
            .iter()
            .map(|c| c.re)
            .collect();

        // Save tail for next block
        let tail_start = input.len().saturating_sub(self.kernel_len - 1);
        self.prev_tail = input[tail_start..].to_vec();
        self.prev_tail.resize(self.kernel_len - 1, 0.0);

        useful
    }

    pub fn reset(&mut self) {
        self.prev_tail.fill(0.0);
    }
}

// ── Wiener deconvolution ────────────────────────────────────────

/// Wiener deconvolution: recover original signal from convolved + noise.
/// `observed`: observed signal (convolution of original with `kernel` + noise).
/// `kernel`: known impulse response (convolution kernel).
/// `noise_power`: estimated noise power spectral density (regularization).
pub fn wiener_deconvolution(
    observed: &[f64],
    kernel: &[f64],
    noise_power: f64,
) -> Result<Vec<f64>, ConvError> {
    if observed.is_empty() || kernel.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if noise_power < 0.0 {
        return Err(ConvError::InvalidParameter("noise_power must be >= 0".into()));
    }

    let n = next_pow2(observed.len().max(kernel.len()));

    let mut obs_fft: Vec<Cpx> = observed.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    obs_fft.resize(n, Cpx::zero());
    fft_ip(&mut obs_fft, false);

    let mut ker_fft: Vec<Cpx> = kernel.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    ker_fft.resize(n, Cpx::zero());
    fft_ip(&mut ker_fft, false);

    // Wiener filter: H_inv(f) = conj(K(f)) / (|K(f)|^2 + noise_power)
    let mut result_fft: Vec<Cpx> = obs_fft
        .iter()
        .zip(ker_fft.iter())
        .map(|(y, h)| {
            let h_conj = h.conj();
            let denom = h.mag_sq() + noise_power;
            if denom > 1e-12 {
                *y * h_conj * (1.0 / denom)
            } else {
                Cpx::zero()
            }
        })
        .collect();

    fft_ip(&mut result_fft, true);
    Ok(result_fft.iter().take(observed.len()).map(|c| c.re).collect())
}

// ── Matched filtering ───────────────────────────────────────────

/// Matched filter: correlate signal with known template for detection.
/// Equivalent to cross-correlation, but normalized for detection.
pub fn matched_filter(signal: &[f64], template: &[f64]) -> Result<Vec<f64>, ConvError> {
    if signal.is_empty() || template.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    let result = cross_correlate(signal, template)?;
    // Normalize by template energy
    let template_energy: f64 = template.iter().map(|t| t * t).sum();
    if template_energy < 1e-12 {
        return Ok(result);
    }
    Ok(result.iter().map(|r| r / template_energy.sqrt()).collect())
}

// ── Convolution reverb ──────────────────────────────────────────

/// Apply convolution reverb: convolve a dry signal with an impulse response.
/// `wet_mix`: 0.0 = fully dry, 1.0 = fully wet.
pub fn convolution_reverb(
    dry_signal: &[f64],
    impulse_response: &[f64],
    wet_mix: f64,
) -> Result<Vec<f64>, ConvError> {
    if dry_signal.is_empty() || impulse_response.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    let wet_mix = wet_mix.clamp(0.0, 1.0);
    let wet = convolve_fft(dry_signal, impulse_response)?;

    // Normalize wet signal to match dry signal energy
    let dry_energy: f64 = dry_signal.iter().map(|s| s * s).sum::<f64>().sqrt();
    let wet_energy: f64 = wet.iter().take(dry_signal.len()).map(|s| s * s).sum::<f64>().sqrt();
    let gain = if wet_energy > 1e-12 { dry_energy / wet_energy } else { 1.0 };

    let output_len = wet.len();
    let mut output = vec![0.0; output_len];
    for i in 0..output_len {
        let dry_val = if i < dry_signal.len() { dry_signal[i] } else { 0.0 };
        let wet_val = wet[i] * gain;
        output[i] = (1.0 - wet_mix) * dry_val + wet_mix * wet_val;
    }
    Ok(output)
}

// ── Multi-channel convolution ───────────────────────────────────

/// Convolve each channel independently with the same kernel.
pub fn convolve_multichannel(
    channels: &[Vec<f64>],
    kernel: &[f64],
) -> Result<Vec<Vec<f64>>, ConvError> {
    if channels.is_empty() || kernel.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    channels
        .iter()
        .map(|ch| convolve_fft(ch, kernel))
        .collect()
}

/// Convolve each channel with its own kernel.
pub fn convolve_multichannel_multi_kernel(
    channels: &[Vec<f64>],
    kernels: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ConvError> {
    if channels.is_empty() || kernels.is_empty() {
        return Err(ConvError::EmptyInput);
    }
    if channels.len() != kernels.len() {
        return Err(ConvError::LengthMismatch(format!(
            "channels={} != kernels={}",
            channels.len(),
            kernels.len()
        )));
    }
    channels
        .iter()
        .zip(kernels.iter())
        .map(|(ch, kr)| convolve_fft(ch, kr))
        .collect()
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
    fn test_direct_impulse() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 0.0, 0.0];
        let out = convolve_direct(&a, &b).unwrap();
        assert_eq!(out.len(), 5);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 2.0));
        assert!(approx_eq(out[2], 3.0));
    }

    #[test]
    fn test_direct_known() {
        // [1,2,3] * [1,1] = [1,3,5,3]
        let out = convolve_direct(&[1.0, 2.0, 3.0], &[1.0, 1.0]).unwrap();
        let expected = vec![1.0, 3.0, 5.0, 3.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_direct_empty() {
        assert!(convolve_direct(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_fft_matches_direct() {
        let a: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let b = vec![0.2, 0.6, 0.2];
        let direct = convolve_direct(&a, &b).unwrap();
        let fft_out = convolve_fft(&a, &b).unwrap();
        assert_eq!(direct.len(), fft_out.len());
        for (d, f) in direct.iter().zip(fft_out.iter()) {
            assert!(approx_eq(*d, *f), "direct={d}, fft={f}");
        }
    }

    #[test]
    fn test_fft_empty() {
        assert!(convolve_fft(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_circular_convolution() {
        // Circular convolution of [1,2,3,4] with [1,0,0,0] = [1,2,3,4]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let out = convolve_circular(&a, &b).unwrap();
        for (x, y) in out.iter().zip(a.iter()) {
            assert!(approx_eq(*x, *y));
        }
    }

    #[test]
    fn test_circular_length_mismatch() {
        assert!(convolve_circular(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_cross_correlate_self() {
        let sig = vec![1.0, 0.0, -1.0, 0.0];
        let xcorr = cross_correlate(&sig, &sig).unwrap();
        // Zero-lag (at index b.len()-1 = 3) should be max
        let zero_lag = xcorr[sig.len() - 1];
        for &v in &xcorr {
            assert!(v <= zero_lag + EPS);
        }
    }

    #[test]
    fn test_auto_correlate_symmetry() {
        let sig = vec![1.0, 2.0, 3.0, 2.0];
        let ac = auto_correlate(&sig).unwrap();
        let mid = sig.len() - 1;
        // Auto-correlation is symmetric around zero lag
        for i in 1..sig.len() {
            assert!(
                (ac[mid - i] - ac[mid + i]).abs() < EPS,
                "asymmetry at lag {i}"
            );
        }
    }

    #[test]
    fn test_auto_correlate_peak() {
        let sig: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).sin()).collect();
        let ac = auto_correlate(&sig).unwrap();
        let mid = sig.len() - 1;
        // Zero-lag should be the maximum
        for &v in &ac {
            assert!(v <= ac[mid] + EPS);
        }
    }

    #[test]
    fn test_overlap_add_basic() {
        let kernel = vec![1.0, 0.5, 0.25];
        let mut ola = OverlapAdd::new(&kernel).unwrap();
        let block1 = vec![1.0, 0.0, 0.0, 0.0];
        let out1 = ola.process_block(&block1);
        assert_eq!(out1.len(), 4);
        assert!(approx_eq(out1[0], 1.0));
    }

    #[test]
    fn test_overlap_add_empty_kernel() {
        assert!(OverlapAdd::new(&[]).is_err());
    }

    #[test]
    fn test_overlap_save_basic() {
        let kernel = vec![1.0, 0.5];
        let mut ols = OverlapSave::new(&kernel).unwrap();
        let block = vec![1.0, 2.0, 3.0, 4.0];
        let out = ols.process_block(&block);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_overlap_save_empty_kernel() {
        assert!(OverlapSave::new(&[]).is_err());
    }

    #[test]
    fn test_wiener_deconvolution_identity() {
        // Convolve with [1,0,...] (identity) → deconvolution should recover original
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kernel = vec![1.0];
        let observed = convolve_fft(&original, &kernel).unwrap();
        let recovered = wiener_deconvolution(&observed, &kernel, 0.0).unwrap();
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(approx_eq(*a, *b), "recovered {} != original {}", *b, *a);
        }
    }

    #[test]
    fn test_wiener_deconvolution_smoothing() {
        let original: Vec<f64> = (0..16).map(|i| (i as f64 * 0.5).sin()).collect();
        let kernel = vec![0.25, 0.5, 0.25];
        let observed = convolve_fft(&original, &kernel).unwrap();
        let recovered = wiener_deconvolution(&observed, &kernel, 0.001).unwrap();
        // Should roughly recover the original (with some noise from regularization)
        for i in 2..14 {
            assert!(
                (original[i] - recovered[i]).abs() < 0.5,
                "idx {i}: {} vs {}",
                original[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_wiener_empty() {
        assert!(wiener_deconvolution(&[], &[1.0], 0.0).is_err());
    }

    #[test]
    fn test_wiener_negative_noise() {
        assert!(wiener_deconvolution(&[1.0], &[1.0], -1.0).is_err());
    }

    #[test]
    fn test_matched_filter_detection() {
        // Signal with embedded template
        let template = vec![1.0, -1.0, 1.0, -1.0];
        let mut signal = vec![0.0; 20];
        // Embed template at position 8
        for (i, &t) in template.iter().enumerate() {
            signal[8 + i] = t;
        }
        let mf = matched_filter(&signal, &template).unwrap();
        // Peak should be near the embedding position
        let peak_idx = mf.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        // Peak in cross-correlation should be at embed_pos + template_len - 1
        let expected_peak = 8 + template.len() - 1;
        assert!(
            (peak_idx as i64 - expected_peak as i64).unsigned_abs() < 3,
            "peak at {peak_idx}, expected near {expected_peak}"
        );
    }

    #[test]
    fn test_matched_filter_empty() {
        assert!(matched_filter(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_convolution_reverb_dry() {
        // wet_mix=0 → output = dry signal
        let dry = vec![1.0, 2.0, 3.0, 4.0];
        let ir = vec![1.0, 0.5];
        let out = convolution_reverb(&dry, &ir, 0.0).unwrap();
        for i in 0..dry.len() {
            assert!(approx_eq(out[i], dry[i]));
        }
    }

    #[test]
    fn test_convolution_reverb_wet() {
        let dry = vec![1.0, 0.0, 0.0, 0.0];
        let ir = vec![1.0, 0.5, 0.25];
        let out = convolution_reverb(&dry, &ir, 1.0).unwrap();
        assert!(!out.is_empty());
        // With wet_mix=1.0, first sample should be nonzero
        assert!(out[0].abs() > 0.01);
    }

    #[test]
    fn test_convolution_reverb_empty() {
        assert!(convolution_reverb(&[], &[1.0], 0.5).is_err());
    }

    #[test]
    fn test_multichannel_same_kernel() {
        let channels = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let kernel = vec![1.0, 0.5];
        let out = convolve_multichannel(&channels, &kernel).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_multichannel_multi_kernel() {
        let channels = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let kernels = vec![
            vec![1.0],
            vec![0.5, 0.5],
        ];
        let out = convolve_multichannel_multi_kernel(&channels, &kernels).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_multichannel_mismatch() {
        let channels = vec![vec![1.0]];
        let kernels = vec![vec![1.0], vec![2.0]];
        assert!(convolve_multichannel_multi_kernel(&channels, &kernels).is_err());
    }

    #[test]
    fn test_convolution_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let ab = convolve_fft(&a, &b).unwrap();
        let ba = convolve_fft(&b, &a).unwrap();
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!(approx_eq(*x, *y));
        }
    }

    #[test]
    fn test_overlap_add_reset() {
        let kernel = vec![1.0, 0.5];
        let mut ola = OverlapAdd::new(&kernel).unwrap();
        ola.process_block(&[1.0, 2.0, 3.0]);
        ola.reset();
        // After reset, should behave like fresh
        let out = ola.process_block(&[1.0, 0.0, 0.0]);
        assert!(approx_eq(out[0], 1.0));
    }
}
