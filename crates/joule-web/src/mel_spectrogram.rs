//! Mel-scale spectrogram and MFCC — pure Rust, no external dependencies.
//!
//! Provides mel-scale frequency conversion, triangular mel filter banks,
//! power and log-mel spectrograms, MFCCs via DCT-II, and delta/delta-delta
//! features for speech and audio analysis.

use std::f64::consts::PI;

// ── Mel scale conversion ────────────────────────────────────────

/// Convert a frequency in Hz to the mel scale.
pub fn freq_to_mel(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Convert a mel value back to frequency in Hz.
pub fn mel_to_freq(mel: f64) -> f64 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

// ── Mini FFT ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Cpx {
    re: f64,
    im: f64,
}

impl Cpx {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    fn from_polar(mag: f64, phase: f64) -> Self {
        Self { re: mag * phase.cos(), im: mag * phase.sin() }
    }
    fn mag_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
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

fn next_pow2(n: usize) -> usize {
    let mut p = 1; while p < n { p <<= 1; } p
}

fn bit_rev(i: usize, bits: u32) -> usize {
    let mut r = 0; let mut v = i;
    for _ in 0..bits { r = (r << 1) | (v & 1); v >>= 1; }
    r
}

fn fft_forward(data: &mut [Cpx]) {
    let n = data.len();
    if n <= 1 { return; }
    let log2n = (n as f64).log2() as u32;
    for i in 0..n { let j = bit_rev(i, log2n); if i < j { data.swap(i, j); } }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..half {
                let tw = Cpx::from_polar(1.0, ang * k as f64);
                let u = data[s + k];
                let v = data[s + k + half] * tw;
                data[s + k] = u + v;
                data[s + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

fn compute_power_spectrum(signal: &[f64], fft_size: usize) -> Vec<f64> {
    let mut buf: Vec<Cpx> = signal.iter().map(|v| Cpx::new(*v, 0.0)).collect();
    buf.resize(fft_size, Cpx::zero());
    fft_forward(&mut buf);
    let half = fft_size / 2 + 1;
    buf[..half].iter().map(|c| c.mag_sq()).collect()
}

// ── Mel Filter Bank ─────────────────────────────────────────────

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    /// Number of mel filter bands.
    pub n_mels: usize,
    /// Number of MFCC coefficients to keep.
    pub n_mfcc: usize,
    /// FFT size (power of 2).
    pub fft_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Minimum frequency for mel bank.
    pub fmin: f64,
    /// Maximum frequency for mel bank.
    pub fmax: f64,
    /// Window size for analysis (power of 2).
    pub window_size: usize,
    /// Hop size for frame advance.
    pub hop_size: usize,
}

impl MelConfig {
    pub fn new(sample_rate: f64, fft_size: usize) -> Self {
        Self {
            n_mels: 40,
            n_mfcc: 13,
            fft_size,
            sample_rate,
            fmin: 0.0,
            fmax: sample_rate / 2.0,
            window_size: fft_size,
            hop_size: fft_size / 4,
        }
    }

    pub fn with_n_mels(mut self, n: usize) -> Self {
        self.n_mels = n;
        self
    }

    pub fn with_n_mfcc(mut self, n: usize) -> Self {
        self.n_mfcc = n;
        self
    }

    pub fn with_freq_range(mut self, fmin: f64, fmax: f64) -> Self {
        self.fmin = fmin;
        self.fmax = fmax;
        self
    }

    pub fn with_hop_size(mut self, hop: usize) -> Self {
        self.hop_size = hop;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MelError {
    InvalidConfig(String),
    EmptyInput,
}

impl std::fmt::Display for MelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid mel config: {s}"),
            Self::EmptyInput => write!(f, "input is empty"),
        }
    }
}

/// A single triangular mel filter: (start_bin, center_bin, end_bin, weights).
#[derive(Debug, Clone)]
pub struct MelFilter {
    pub start_bin: usize,
    pub center_bin: usize,
    pub end_bin: usize,
    pub weights: Vec<f64>,
}

/// Build a mel filter bank: `n_mels` triangular filters.
pub fn build_mel_filter_bank(config: &MelConfig) -> Result<Vec<MelFilter>, MelError> {
    if config.n_mels == 0 {
        return Err(MelError::InvalidConfig("n_mels must be > 0".into()));
    }
    if config.fmax <= config.fmin {
        return Err(MelError::InvalidConfig("fmax must be > fmin".into()));
    }

    let mel_low = freq_to_mel(config.fmin);
    let mel_high = freq_to_mel(config.fmax);
    let n_points = config.n_mels + 2;
    let half = config.fft_size / 2 + 1;

    // Evenly spaced points on mel scale
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_points - 1) as f64)
        .collect();

    // Convert to FFT bin indices
    let bin_indices: Vec<usize> = mel_points
        .iter()
        .map(|m| {
            let freq = mel_to_freq(*m);
            let bin = (freq * config.fft_size as f64 / config.sample_rate).round() as usize;
            bin.min(half - 1)
        })
        .collect();

    let mut filters = Vec::with_capacity(config.n_mels);
    for i in 0..config.n_mels {
        let start = bin_indices[i];
        let center = bin_indices[i + 1];
        let end = bin_indices[i + 2];

        let filter_len = if end >= start { end - start + 1 } else { 1 };
        let mut weights = vec![0.0; filter_len];

        for k in start..=end {
            let idx = k - start;
            if k <= center && center > start {
                weights[idx] = (k - start) as f64 / (center - start) as f64;
            } else if k > center && end > center {
                weights[idx] = (end - k) as f64 / (end - center) as f64;
            }
        }

        filters.push(MelFilter {
            start_bin: start,
            center_bin: center,
            end_bin: end,
            weights,
        });
    }
    Ok(filters)
}

/// Apply mel filter bank to a power spectrum, returning mel energies.
pub fn apply_mel_filters(power_spec: &[f64], filters: &[MelFilter]) -> Vec<f64> {
    filters
        .iter()
        .map(|filt| {
            let mut energy = 0.0;
            for (i, &w) in filt.weights.iter().enumerate() {
                let bin = filt.start_bin + i;
                if bin < power_spec.len() {
                    energy += power_spec[bin] * w;
                }
            }
            energy
        })
        .collect()
}

// ── Hann window ─────────────────────────────────────────────────

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos()))
        .collect()
}

// ── Mel Spectrogram ─────────────────────────────────────────────

/// Compute power spectrogram frames from a signal.
fn compute_frames(signal: &[f64], config: &MelConfig) -> Vec<Vec<f64>> {
    let window = hann_window(config.window_size);
    let mut frames = Vec::new();
    let mut start = 0;
    while start + config.window_size <= signal.len() {
        let windowed: Vec<f64> = (0..config.window_size)
            .map(|i| signal[start + i] * window[i])
            .collect();
        let ps = compute_power_spectrum(&windowed, config.fft_size);
        frames.push(ps);
        start += config.hop_size;
    }
    frames
}

/// Compute mel spectrogram (mel energies per frame).
pub fn mel_spectrogram(signal: &[f64], config: &MelConfig) -> Result<Vec<Vec<f64>>, MelError> {
    if signal.is_empty() {
        return Err(MelError::EmptyInput);
    }
    let filters = build_mel_filter_bank(config)?;
    let frames = compute_frames(signal, config);
    Ok(frames.iter().map(|ps| apply_mel_filters(ps, &filters)).collect())
}

/// Compute log-mel spectrogram (log of mel energies).
pub fn log_mel_spectrogram(signal: &[f64], config: &MelConfig) -> Result<Vec<Vec<f64>>, MelError> {
    let mel = mel_spectrogram(signal, config)?;
    Ok(mel
        .iter()
        .map(|row| row.iter().map(|e| (e + 1e-10).ln()).collect())
        .collect())
}

// ── DCT-II ──────────────────────────────────────────────────────

/// Type-II Discrete Cosine Transform.
fn dct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut sum = 0.0;
            for (i, &val) in input.iter().enumerate() {
                sum += val * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos();
            }
            sum
        })
        .collect()
}

// ── MFCC ────────────────────────────────────────────────────────

/// Compute MFCCs from a signal. Returns one MFCC vector per frame.
pub fn mfcc(signal: &[f64], config: &MelConfig) -> Result<Vec<Vec<f64>>, MelError> {
    let log_mel = log_mel_spectrogram(signal, config)?;
    Ok(log_mel
        .iter()
        .map(|row| {
            let dct = dct_ii(row);
            dct[..config.n_mfcc.min(dct.len())].to_vec()
        })
        .collect())
}

// ── Delta features ──────────────────────────────────────────────

/// Compute delta (first derivative) features from a feature matrix.
/// Uses a +/- `width` context (default 2).
pub fn delta_features(features: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    let n_frames = features.len();
    if n_frames == 0 {
        return vec![];
    }
    let n_coeffs = features[0].len();
    let w = width.max(1);

    // Denominator: 2 * sum(i^2 for i in 1..=w)
    let denom: f64 = 2.0 * (1..=w).map(|i| (i * i) as f64).sum::<f64>();

    features
        .iter()
        .enumerate()
        .map(|(t, _)| {
            (0..n_coeffs)
                .map(|c| {
                    let mut numerator = 0.0;
                    for i in 1..=w {
                        let prev = if t >= i { t - i } else { 0 };
                        let next = (t + i).min(n_frames - 1);
                        numerator += i as f64 * (features[next][c] - features[prev][c]);
                    }
                    numerator / denom
                })
                .collect()
        })
        .collect()
}

/// Compute delta-delta (second derivative) features.
pub fn delta_delta_features(features: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    let d = delta_features(features, width);
    delta_features(&d, width)
}

/// Append delta and delta-delta to each feature vector.
pub fn append_deltas(features: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    let d1 = delta_features(features, width);
    let d2 = delta_delta_features(features, width);
    features
        .iter()
        .zip(d1.iter())
        .zip(d2.iter())
        .map(|((f, d1v), d2v)| {
            let mut combined = f.clone();
            combined.extend_from_slice(d1v);
            combined.extend_from_slice(d2v);
            combined
        })
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
    fn test_freq_to_mel_zero() {
        assert!(approx_eq(freq_to_mel(0.0), 0.0));
    }

    #[test]
    fn test_freq_to_mel_1000() {
        // 1127 * ln(1 + 1000/700) ≈ 1127 * ln(1.4286) ≈ 1127 * 0.3567 ≈ 401.9
        let mel = freq_to_mel(1000.0);
        assert!((mel - 1127.0 * (1.0_f64 + 1000.0 / 700.0).ln()).abs() < 1e-6);
    }

    #[test]
    fn test_mel_roundtrip() {
        for freq in [0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let mel = freq_to_mel(freq);
            let back = mel_to_freq(mel);
            assert!(approx_eq(freq, back), "roundtrip failed for {freq}");
        }
    }

    #[test]
    fn test_mel_monotonic() {
        let mut prev = freq_to_mel(0.0);
        for f in (100..=8000).step_by(100) {
            let mel = freq_to_mel(f as f64);
            assert!(mel > prev, "mel scale not monotonic at {f}");
            prev = mel;
        }
    }

    #[test]
    fn test_build_filter_bank_dimensions() {
        let config = MelConfig::new(16000.0, 512).with_n_mels(26);
        let filters = build_mel_filter_bank(&config).unwrap();
        assert_eq!(filters.len(), 26);
    }

    #[test]
    fn test_build_filter_bank_zero_mels() {
        let config = MelConfig::new(16000.0, 512).with_n_mels(0);
        assert!(build_mel_filter_bank(&config).is_err());
    }

    #[test]
    fn test_build_filter_bank_invalid_freq_range() {
        let config = MelConfig::new(16000.0, 512).with_freq_range(8000.0, 100.0);
        assert!(build_mel_filter_bank(&config).is_err());
    }

    #[test]
    fn test_filter_weights_nonnegative() {
        let config = MelConfig::new(16000.0, 256).with_n_mels(10);
        let filters = build_mel_filter_bank(&config).unwrap();
        for filt in &filters {
            for &w in &filt.weights {
                assert!(w >= 0.0, "negative weight found");
            }
        }
    }

    #[test]
    fn test_apply_mel_filters_unity() {
        // Flat power spectrum → mel energies should reflect filter shape
        let config = MelConfig::new(8000.0, 64).with_n_mels(4);
        let filters = build_mel_filter_bank(&config).unwrap();
        let flat_spec = vec![1.0; 33]; // fft_size/2 + 1
        let energies = apply_mel_filters(&flat_spec, &filters);
        assert_eq!(energies.len(), 4);
        for &e in &energies {
            assert!(e >= 0.0);
        }
    }

    #[test]
    fn test_mel_spectrogram_dimensions() {
        let config = MelConfig::new(8000.0, 64)
            .with_n_mels(10)
            .with_hop_size(16);
        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.01).sin()).collect();
        let spec = mel_spectrogram(&signal, &config).unwrap();
        // Number of frames = (256 - 64) / 16 + 1 = 13
        let expected_frames = (256 - 64) / 16 + 1;
        assert_eq!(spec.len(), expected_frames);
        for row in &spec {
            assert_eq!(row.len(), 10);
        }
    }

    #[test]
    fn test_mel_spectrogram_empty() {
        let config = MelConfig::new(16000.0, 256);
        assert!(mel_spectrogram(&[], &config).is_err());
    }

    #[test]
    fn test_log_mel_finite() {
        let config = MelConfig::new(8000.0, 64)
            .with_n_mels(8)
            .with_hop_size(16);
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.05).sin()).collect();
        let lm = log_mel_spectrogram(&signal, &config).unwrap();
        for row in &lm {
            for &v in row {
                assert!(v.is_finite(), "non-finite log-mel value");
            }
        }
    }

    #[test]
    fn test_dct_ii_dc() {
        // DCT of constant signal → first coeff is N * value, rest near zero
        let input = vec![1.0; 8];
        let dct = dct_ii(&input);
        assert!(approx_eq(dct[0], 8.0));
        for &v in &dct[1..] {
            assert!(v.abs() < EPS);
        }
    }

    #[test]
    fn test_mfcc_dimensions() {
        let config = MelConfig::new(8000.0, 64)
            .with_n_mels(20)
            .with_n_mfcc(13)
            .with_hop_size(16);
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.03).sin()).collect();
        let coeffs = mfcc(&signal, &config).unwrap();
        for row in &coeffs {
            assert_eq!(row.len(), 13);
        }
    }

    #[test]
    fn test_mfcc_empty() {
        let config = MelConfig::new(16000.0, 256);
        assert!(mfcc(&[], &config).is_err());
    }

    #[test]
    fn test_delta_features_constant() {
        // Constant features → delta should be zero
        let feats = vec![vec![1.0, 2.0, 3.0]; 5];
        let d = delta_features(&feats, 2);
        assert_eq!(d.len(), 5);
        for row in &d {
            for &v in row {
                assert!(v.abs() < EPS, "delta of constant should be 0, got {v}");
            }
        }
    }

    #[test]
    fn test_delta_features_linear() {
        // Linear ramp → delta should be constant
        let feats: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let d = delta_features(&feats, 2);
        // Interior frames should have ~1.0 slope
        for row in &d[3..7] {
            assert!(
                (row[0] - 1.0).abs() < 0.3,
                "expected delta ~1.0, got {}",
                row[0]
            );
        }
    }

    #[test]
    fn test_delta_delta_dimensions() {
        let feats = vec![vec![1.0, 2.0]; 8];
        let dd = delta_delta_features(&feats, 2);
        assert_eq!(dd.len(), 8);
        for row in &dd {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn test_append_deltas_dimensions() {
        let feats = vec![vec![1.0, 2.0, 3.0]; 6];
        let combined = append_deltas(&feats, 2);
        assert_eq!(combined.len(), 6);
        // 3 base + 3 delta + 3 delta-delta = 9
        for row in &combined {
            assert_eq!(row.len(), 9);
        }
    }

    #[test]
    fn test_delta_features_empty() {
        let d = delta_features(&[], 2);
        assert!(d.is_empty());
    }

    #[test]
    fn test_mel_config_builder() {
        let config = MelConfig::new(44100.0, 1024)
            .with_n_mels(80)
            .with_n_mfcc(20)
            .with_freq_range(80.0, 7600.0)
            .with_hop_size(256);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_mfcc, 20);
        assert!(approx_eq(config.fmin, 80.0));
        assert!(approx_eq(config.fmax, 7600.0));
        assert_eq!(config.hop_size, 256);
    }

    #[test]
    fn test_power_spectrum_parseval() {
        // Sum of power spectrum ≈ sum of signal²  × fft_size
        let signal = vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.0];
        let time_energy: f64 = signal.iter().map(|s| s * s).sum();
        let ps = compute_power_spectrum(&signal, 8);
        // Parseval: sum(|X[k]|^2) = N * sum(|x[n]|^2)
        let freq_energy: f64 = ps[0] + 2.0 * ps[1..4].iter().sum::<f64>() + ps[4];
        let ratio = freq_energy / (8.0 * time_energy);
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "Parseval ratio: {ratio}"
        );
    }
}
