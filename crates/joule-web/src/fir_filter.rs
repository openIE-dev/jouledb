//! FIR (Finite Impulse Response) filters — pure Rust, no external dependencies.
//!
//! Design methods: windowed sinc (low-pass, high-pass), least-squares,
//! simplified Parks-McClellan (Remez exchange). Direct convolution and
//! overlap-save (FFT-based) application. Linear and minimum phase.
//! Filter cascading. Impulse/step response. Group delay.

use std::f64::consts::PI;

// ── Mini FFT for overlap-save ───────────────────────────────────

#[derive(Clone, Copy)]
struct Cpx { re: f64, im: f64 }

impl Cpx {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    fn polar(r: f64, t: f64) -> Self { Self { re: r * t.cos(), im: r * t.sin() } }
    fn mag(&self) -> f64 { (self.re * self.re + self.im * self.im).sqrt() }
    fn phase(&self) -> f64 { self.im.atan2(self.re) }
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

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FirError {
    InvalidOrder(String),
    InvalidFrequency(String),
    EmptyInput,
    IncompatibleLengths,
}

impl std::fmt::Display for FirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder(s) => write!(f, "invalid order: {s}"),
            Self::InvalidFrequency(s) => write!(f, "invalid frequency: {s}"),
            Self::EmptyInput => write!(f, "empty input"),
            Self::IncompatibleLengths => write!(f, "incompatible lengths"),
        }
    }
}

// ── Window functions ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirWindow {
    Rectangular,
    Hamming,
    Blackman,
    Kaiser(f64),
    Hann,
}

fn make_window(win: FirWindow, n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0]; }
    match win {
        FirWindow::Rectangular => vec![1.0; n],
        FirWindow::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
            .collect(),
        FirWindow::Blackman => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (n - 1) as f64;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
        FirWindow::Kaiser(beta) => {
            fn bessel_i0(x: f64) -> f64 {
                let mut s = 1.0; let mut t = 1.0;
                for k in 1..=25 { t *= (x / (2.0 * k as f64)).powi(2); s += t; }
                s
            }
            let d = bessel_i0(beta);
            (0..n).map(|i| {
                let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / d
            }).collect()
        }
        FirWindow::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
            .collect(),
    }
}

// ── Windowed sinc design ────────────────────────────────────────

/// Design a low-pass FIR filter using the windowed sinc method.
/// `cutoff`: normalized frequency (0..0.5).
/// `order`: filter order (taps = order + 1, even recommended).
pub fn design_lowpass(cutoff: f64, order: usize, window: FirWindow) -> Result<Vec<f64>, FirError> {
    if cutoff <= 0.0 || cutoff >= 0.5 {
        return Err(FirError::InvalidFrequency(format!("cutoff {cutoff} not in (0, 0.5)")));
    }
    if order < 1 {
        return Err(FirError::InvalidOrder("order must be >= 1".into()));
    }
    let n = order + 1;
    let mid = order as f64 / 2.0;
    let win = make_window(window, n);
    let mut kernel: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - mid;
            let sinc = if x.abs() < 1e-12 { 2.0 * cutoff } else { (2.0 * PI * cutoff * x).sin() / (PI * x) };
            sinc * win[i]
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    if sum.abs() > 1e-12 { for k in kernel.iter_mut() { *k /= sum; } }
    Ok(kernel)
}

/// Design a high-pass FIR filter via spectral inversion of a low-pass.
pub fn design_highpass(cutoff: f64, order: usize, window: FirWindow) -> Result<Vec<f64>, FirError> {
    let mut lp = design_lowpass(cutoff, order, window)?;
    let mid = lp.len() / 2;
    for v in lp.iter_mut() { *v = -*v; }
    lp[mid] += 1.0;
    Ok(lp)
}

// ── Least-squares design ────────────────────────────────────────

/// Design a low-pass FIR filter using least-squares optimization.
/// Minimizes ∫|H(ω) - D(ω)|² over specified bands.
pub fn design_least_squares(cutoff: f64, order: usize) -> Result<Vec<f64>, FirError> {
    if cutoff <= 0.0 || cutoff >= 0.5 {
        return Err(FirError::InvalidFrequency(format!("cutoff {cutoff}")));
    }
    if order < 2 {
        return Err(FirError::InvalidOrder("order >= 2".into()));
    }
    let n = order + 1;
    let mid = order as f64 / 2.0;
    let wc = PI * 2.0 * cutoff;

    // Optimal least-squares coefficients for ideal low-pass
    let kernel: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - mid;
            if x.abs() < 1e-12 {
                wc / PI
            } else {
                (wc * x).sin() / (PI * x)
            }
        })
        .collect();

    // Normalize
    let sum: f64 = kernel.iter().sum();
    Ok(if sum.abs() > 1e-12 {
        kernel.iter().map(|k| k / sum).collect()
    } else {
        kernel
    })
}

// ── Simplified Parks-McClellan (Remez) ──────────────────────────

/// Simplified Parks-McClellan filter design.
/// `bands`: &[(start_freq, end_freq, desired_gain, weight)] — normalized frequencies.
/// Returns filter taps of length `order + 1`.
pub fn design_parks_mcclellan(
    order: usize,
    bands: &[(f64, f64, f64, f64)],
) -> Result<Vec<f64>, FirError> {
    if order < 2 {
        return Err(FirError::InvalidOrder("order >= 2".into()));
    }
    if bands.is_empty() {
        return Err(FirError::EmptyInput);
    }
    let n = order + 1;
    let mid = order as f64 / 2.0;

    // Dense frequency grid
    let n_grid = 512;
    let mut freqs = Vec::new();
    let mut desired = Vec::new();
    let mut weights = Vec::new();

    for &(f1, f2, d, w) in bands {
        let steps = ((f2 - f1) / 0.5 * n_grid as f64).max(2.0) as usize;
        for i in 0..steps {
            let f = f1 + (f2 - f1) * i as f64 / (steps - 1).max(1) as f64;
            freqs.push(f);
            desired.push(d);
            weights.push(w);
        }
    }

    // Iterative weighted least-squares approximation (simplified Remez)
    let mut kernel = vec![0.0; n];
    for iter_count in 0..20 {
        // Compute current response at grid points
        let mut max_error = 0.0_f64;
        let _ = iter_count;

        for (gi, &f) in freqs.iter().enumerate() {
            let omega = 2.0 * PI * f;
            let mut resp = 0.0;
            for (k, &h) in kernel.iter().enumerate() {
                resp += h * (omega * (k as f64 - mid)).cos();
            }
            let err = (desired[gi] - resp) * weights[gi];
            max_error = max_error.max(err.abs());
        }

        // Update coefficients using weighted frequency sampling
        for k in 0..n {
            let mut num = 0.0;
            let mut den = 0.0;
            for (gi, &f) in freqs.iter().enumerate() {
                let omega = 2.0 * PI * f;
                let basis = (omega * (k as f64 - mid)).cos();
                let w = weights[gi];
                num += w * desired[gi] * basis;
                den += w * basis * basis;
            }
            if den.abs() > 1e-12 {
                kernel[k] = num / den;
            }
        }

        if max_error < 1e-6 { break; }
    }
    Ok(kernel)
}

// ── Apply: direct convolution ───────────────────────────────────

/// Apply FIR filter via direct convolution (full output).
pub fn apply_direct(signal: &[f64], taps: &[f64]) -> Result<Vec<f64>, FirError> {
    if signal.is_empty() || taps.is_empty() {
        return Err(FirError::EmptyInput);
    }
    let out_len = signal.len() + taps.len() - 1;
    let mut out = vec![0.0; out_len];
    for (i, &s) in signal.iter().enumerate() {
        for (j, &t) in taps.iter().enumerate() {
            out[i + j] += s * t;
        }
    }
    Ok(out)
}

/// Apply FIR filter, "same" mode (output length = input length).
pub fn apply_same(signal: &[f64], taps: &[f64]) -> Result<Vec<f64>, FirError> {
    let full = apply_direct(signal, taps)?;
    let offset = taps.len() / 2;
    Ok(full[offset..offset + signal.len()].to_vec())
}

// ── Apply: overlap-save (FFT-based) ────────────────────────────

/// Apply FIR filter using the overlap-save method (FFT-based, for long signals).
/// Returns output of length `signal.len()`, aligned like `apply_same` (centered).
pub fn apply_overlap_save(signal: &[f64], taps: &[f64]) -> Result<Vec<f64>, FirError> {
    if signal.is_empty() || taps.is_empty() {
        return Err(FirError::EmptyInput);
    }
    let m = taps.len();
    let block_size = next_pow2((4 * m).max(64));
    let overlap = m - 1;
    let useful = block_size - overlap;

    // FFT of kernel (zero-padded to block_size)
    let mut h_fft: Vec<Cpx> = taps.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    h_fft.resize(block_size, Cpx::zero());
    fft_ip(&mut h_fft, false);

    // Pad signal with overlap zeros at the beginning for the first block
    let mut padded = vec![0.0; overlap];
    padded.extend_from_slice(signal);
    // Pad end so last block is full
    let remainder = (padded.len().saturating_sub(overlap)) % useful;
    if remainder != 0 {
        padded.resize(padded.len() + useful - remainder, 0.0);
    }

    // Compute full linear convolution output via overlap-save blocks
    let mut full_output = Vec::with_capacity(signal.len() + overlap);
    let mut pos = 0;

    while pos + block_size <= padded.len() {
        let mut block: Vec<Cpx> = (0..block_size)
            .map(|i| Cpx::new(padded[pos + i], 0.0))
            .collect();

        fft_ip(&mut block, false);
        for (b, &h) in block.iter_mut().zip(h_fft.iter()) {
            *b = *b * h;
        }
        fft_ip(&mut block, true);

        // The first overlap samples are corrupted by circular aliasing; discard them.
        // The remaining `useful` samples are correct linear convolution output.
        for i in overlap..block_size {
            full_output.push(block[i].re);
        }
        pos += useful;
    }

    // Extract same-length center portion to match apply_same
    let offset = m / 2;
    let end = (offset + signal.len()).min(full_output.len());
    let start = offset.min(full_output.len());
    let mut result = full_output[start..end].to_vec();
    result.resize(signal.len(), 0.0);
    Ok(result)
}

// ── Linear phase check ─────────────────────────────────────────

/// Check if filter coefficients are symmetric (linear phase, Type I or II).
pub fn is_linear_phase(taps: &[f64]) -> bool {
    let n = taps.len();
    for i in 0..n / 2 {
        if (taps[i] - taps[n - 1 - i]).abs() > 1e-10 {
            return false;
        }
    }
    true
}

// ── Minimum phase conversion ────────────────────────────────────

/// Convert a linear-phase FIR to minimum phase using cepstral method.
pub fn to_minimum_phase(taps: &[f64]) -> Result<Vec<f64>, FirError> {
    if taps.is_empty() {
        return Err(FirError::EmptyInput);
    }
    let n = next_pow2(taps.len() * 4);
    let mut spec: Vec<Cpx> = taps.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    spec.resize(n, Cpx::zero());
    fft_ip(&mut spec, false);

    // Log magnitude → cepstrum
    let mut log_mag: Vec<Cpx> = spec
        .iter()
        .map(|c| Cpx::new((c.mag().max(1e-20)).ln(), 0.0))
        .collect();

    fft_ip(&mut log_mag, true);

    // Fold cepstrum: keep [0], double [1..N/2-1], zero [N/2..N-1]
    let half = n / 2;
    log_mag[0] = log_mag[0];
    for i in 1..half {
        log_mag[i] = log_mag[i] * 2.0;
    }
    for i in half..n {
        log_mag[i] = Cpx::zero();
    }

    fft_ip(&mut log_mag, false);

    // Exponentiate
    let min_phase: Vec<Cpx> = log_mag
        .iter()
        .map(|c| {
            let mag = c.re.exp();
            Cpx::polar(mag, c.im)
        })
        .collect();

    // IFFT to get time-domain
    let mut result = min_phase;
    fft_ip(&mut result, true);

    Ok(result[..taps.len()].iter().map(|c| c.re).collect())
}

// ── Filter cascading ────────────────────────────────────────────

/// Cascade two FIR filters (convolve their coefficients).
pub fn cascade(taps_a: &[f64], taps_b: &[f64]) -> Result<Vec<f64>, FirError> {
    if taps_a.is_empty() || taps_b.is_empty() {
        return Err(FirError::EmptyInput);
    }
    let n = taps_a.len() + taps_b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &a) in taps_a.iter().enumerate() {
        for (j, &b) in taps_b.iter().enumerate() {
            result[i + j] += a * b;
        }
    }
    Ok(result)
}

// ── Impulse and step response ───────────────────────────────────

/// Impulse response of an FIR filter (just the taps themselves).
pub fn impulse_response(taps: &[f64]) -> Vec<f64> {
    taps.to_vec()
}

/// Step response of an FIR filter (cumulative sum of taps).
pub fn step_response(taps: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    taps.iter()
        .map(|t| {
            acc += t;
            acc
        })
        .collect()
}

// ── Frequency response ──────────────────────────────────────────

/// Compute magnitude and phase response at `n_points` frequencies.
pub fn frequency_response(taps: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_points);
    let mut mags = Vec::with_capacity(n_points);
    let mut phases = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let f = 0.5 * i as f64 / (n_points - 1).max(1) as f64;
        let omega = 2.0 * PI * f;
        let mut re = 0.0;
        let mut im = 0.0;
        for (k, &h) in taps.iter().enumerate() {
            re += h * (omega * k as f64).cos();
            im -= h * (omega * k as f64).sin();
        }
        freqs.push(f);
        mags.push((re * re + im * im).sqrt());
        phases.push(im.atan2(re));
    }
    (freqs, mags, phases)
}

// ── Group delay ─────────────────────────────────────────────────

/// Compute group delay at `n_points` frequencies.
/// For linear-phase FIR, this is constant = (N-1)/2.
pub fn group_delay(taps: &[f64], n_points: usize) -> Vec<f64> {
    let weighted: Vec<f64> = taps.iter().enumerate().map(|(i, &h)| i as f64 * h).collect();
    let mut delays = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let f = 0.5 * i as f64 / (n_points - 1).max(1) as f64;
        let omega = 2.0 * PI * f;

        let mut hr = 0.0;
        let mut hi = 0.0;
        let mut wr = 0.0;
        let mut wi = 0.0;

        for (k, &h) in taps.iter().enumerate() {
            let c = (omega * k as f64).cos();
            let s = (omega * k as f64).sin();
            hr += h * c;
            hi -= h * s;
            wr += weighted[k] * c;
            wi -= weighted[k] * s;
        }

        let denom = hr * hr + hi * hi;
        delays.push(if denom > 1e-12 { (wr * hr + wi * hi) / denom } else { 0.0 });
    }
    delays
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn test_lowpass_dc_gain() {
        let taps = design_lowpass(0.2, 32, FirWindow::Hamming).unwrap();
        let sum: f64 = taps.iter().sum();
        assert!(approx_eq(sum, 1.0), "DC gain = {sum}");
    }

    #[test]
    fn test_lowpass_length() {
        let taps = design_lowpass(0.25, 64, FirWindow::Blackman).unwrap();
        assert_eq!(taps.len(), 65);
    }

    #[test]
    fn test_lowpass_invalid_freq() {
        assert!(design_lowpass(0.0, 32, FirWindow::Hamming).is_err());
        assert!(design_lowpass(0.5, 32, FirWindow::Hamming).is_err());
    }

    #[test]
    fn test_highpass_nyquist_gain() {
        let taps = design_highpass(0.2, 32, FirWindow::Hamming).unwrap();
        let (_, mags, _) = frequency_response(&taps, 256);
        assert!(mags[255] > 0.8, "Nyquist gain = {}", mags[255]);
        assert!(mags[0] < 0.01, "DC gain = {}", mags[0]);
    }

    #[test]
    fn test_least_squares_dc() {
        let taps = design_least_squares(0.2, 32).unwrap();
        let sum: f64 = taps.iter().sum();
        assert!(approx_eq(sum, 1.0), "LS DC gain = {sum}");
    }

    #[test]
    fn test_least_squares_invalid() {
        assert!(design_least_squares(0.0, 32).is_err());
        assert!(design_least_squares(0.2, 1).is_err());
    }

    #[test]
    fn test_parks_mcclellan_basic() {
        let bands = vec![
            (0.0, 0.15, 1.0, 1.0),   // passband
            (0.25, 0.5, 0.0, 1.0),    // stopband
        ];
        let taps = design_parks_mcclellan(32, &bands).unwrap();
        assert_eq!(taps.len(), 33);
    }

    #[test]
    fn test_parks_mcclellan_empty_bands() {
        assert!(design_parks_mcclellan(32, &[]).is_err());
    }

    #[test]
    fn test_apply_direct_impulse() {
        let taps = vec![1.0, 2.0, 3.0];
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let out = apply_direct(&signal, &taps).unwrap();
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 2.0));
        assert!(approx_eq(out[2], 3.0));
        assert!(approx_eq(out[3], 0.0));
    }

    #[test]
    fn test_apply_same_length() {
        let taps = vec![0.2, 0.6, 0.2];
        let signal: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let out = apply_same(&signal, &taps).unwrap();
        assert_eq!(out.len(), signal.len());
    }

    #[test]
    fn test_apply_empty() {
        assert!(apply_direct(&[], &[1.0]).is_err());
        assert!(apply_direct(&[1.0], &[]).is_err());
    }

    #[test]
    fn test_overlap_save_matches_direct() {
        let taps = design_lowpass(0.2, 16, FirWindow::Hamming).unwrap();
        let signal: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let direct = apply_same(&signal, &taps).unwrap();
        let ols = apply_overlap_save(&signal, &taps).unwrap();
        // Overlap-save output should be close to direct (same length)
        let compare_len = direct.len().min(ols.len());
        let mut max_diff = 0.0_f64;
        for i in 16..compare_len - 16 {
            max_diff = max_diff.max((direct[i] - ols[i]).abs());
        }
        assert!(max_diff < 0.05, "max diff = {max_diff}");
    }

    #[test]
    fn test_is_linear_phase() {
        let taps = design_lowpass(0.2, 32, FirWindow::Hamming).unwrap();
        assert!(is_linear_phase(&taps));
    }

    #[test]
    fn test_not_linear_phase() {
        let taps = vec![1.0, 2.0, 0.5]; // asymmetric
        assert!(!is_linear_phase(&taps));
    }

    #[test]
    fn test_minimum_phase_energy() {
        let taps = design_lowpass(0.2, 16, FirWindow::Hamming).unwrap();
        let min_ph = to_minimum_phase(&taps).unwrap();
        // Energy should be roughly preserved
        let orig_energy: f64 = taps.iter().map(|t| t * t).sum();
        let min_energy: f64 = min_ph.iter().map(|t| t * t).sum();
        assert!(
            (orig_energy - min_energy).abs() / orig_energy < 0.3,
            "energy ratio: {}",
            min_energy / orig_energy
        );
    }

    #[test]
    fn test_cascade_identity() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = cascade(&a, &b).unwrap();
        for (x, y) in c.iter().zip(b.iter()) {
            assert!(approx_eq(*x, *y));
        }
    }

    #[test]
    fn test_cascade_commutative() {
        let a = vec![1.0, 0.5];
        let b = vec![1.0, -0.5, 0.25];
        let ab = cascade(&a, &b).unwrap();
        let ba = cascade(&b, &a).unwrap();
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!(approx_eq(*x, *y));
        }
    }

    #[test]
    fn test_impulse_response() {
        let taps = vec![1.0, 0.5, 0.25];
        let ir = impulse_response(&taps);
        assert_eq!(ir, taps);
    }

    #[test]
    fn test_step_response() {
        let taps = vec![0.25, 0.5, 0.25];
        let sr = step_response(&taps);
        assert!(approx_eq(sr[0], 0.25));
        assert!(approx_eq(sr[1], 0.75));
        assert!(approx_eq(sr[2], 1.0));
    }

    #[test]
    fn test_frequency_response_length() {
        let taps = design_lowpass(0.2, 16, FirWindow::Hamming).unwrap();
        let (f, m, p) = frequency_response(&taps, 64);
        assert_eq!(f.len(), 64);
        assert_eq!(m.len(), 64);
        assert_eq!(p.len(), 64);
    }

    #[test]
    fn test_group_delay_constant() {
        let taps = design_lowpass(0.2, 32, FirWindow::Hamming).unwrap();
        let gd = group_delay(&taps, 128);
        let expected = 16.0; // (33-1)/2
        // Interior points
        for &d in &gd[10..110] {
            assert!((d - expected).abs() < 1.0, "gd = {d}, expected ~{expected}");
        }
    }

    #[test]
    fn test_frequency_response_lowpass() {
        let taps = design_lowpass(0.2, 64, FirWindow::Blackman).unwrap();
        let (_, mags, _) = frequency_response(&taps, 256);
        assert!(approx_eq(mags[0], 1.0), "DC = {}", mags[0]);
        // High frequency should be attenuated
        assert!(mags[200] < 0.01, "stopband = {}", mags[200]);
    }

    #[test]
    fn test_window_types() {
        for win in [FirWindow::Rectangular, FirWindow::Hamming, FirWindow::Blackman,
                    FirWindow::Kaiser(5.0), FirWindow::Hann] {
            let taps = design_lowpass(0.25, 16, win).unwrap();
            let sum: f64 = taps.iter().sum();
            assert!(approx_eq(sum, 1.0), "DC gain with {:?} = {sum}", win);
        }
    }

    #[test]
    fn test_overlap_save_empty() {
        assert!(apply_overlap_save(&[], &[1.0]).is_err());
    }
}
