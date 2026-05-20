//! Short-Time Fourier Transform — pure Rust, no external dependencies.
//!
//! Sliding-window FFT analysis with configurable window size and hop size.
//! Supports forward STFT (spectrogram generation), inverse STFT via overlap-add,
//! phase vocoder for time-stretching, and Griffin-Lim magnitude-only reconstruction.

use std::f64::consts::PI;

// ── Complex number (local, minimal) ─────────────────────────────

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
        Self { re: mag * phase.cos(), im: mag * phase.sin() }
    }
    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
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

// ── Mini FFT (radix-2 Cooley-Tukey) ────────────────────────────

fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn bit_reverse(i: usize, log2_n: u32) -> usize {
    let mut r = 0usize;
    let mut v = i;
    for _ in 0..log2_n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 { return; }
    let log2_n = (n as f64).log2() as u32;
    for i in 0..n {
        let j = bit_reverse(i, log2_n);
        if i < j { data.swap(i, j); }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let sign = if inverse { 1.0 } else { -1.0 };
        let angle = sign * 2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let tw = Complex::from_polar(1.0, angle * k as f64);
                let u = data[start + k];
                let v = data[start + k + half] * tw;
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
        len <<= 1;
    }
    if inverse {
        let s = 1.0 / n as f64;
        for d in data.iter_mut() { *d = *d * s; }
    }
}

fn fft(input: &[Complex]) -> Vec<Complex> {
    let mut data = input.to_vec();
    fft_in_place(&mut data, false);
    data
}

fn ifft(input: &[Complex]) -> Vec<Complex> {
    let mut data = input.to_vec();
    fft_in_place(&mut data, true);
    data
}

// ── Window functions ────────────────────────────────────────────

/// Window type for STFT analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
}

fn make_window(win_type: WindowType, size: usize) -> Vec<f64> {
    match win_type {
        WindowType::Rectangular => vec![1.0; size],
        WindowType::Hann => (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos()))
            .collect(),
        WindowType::Hamming => (0..size)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / size as f64).cos())
            .collect(),
    }
}

// ── STFT Config ─────────────────────────────────────────────────

/// Configuration for the Short-Time Fourier Transform.
#[derive(Debug, Clone, PartialEq)]
pub struct StftConfig {
    /// Window size in samples (must be power of 2).
    pub window_size: usize,
    /// Hop size in samples (overlap = window_size - hop_size).
    pub hop_size: usize,
    /// Window function to apply.
    pub window_type: WindowType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StftError {
    WindowNotPowerOfTwo(usize),
    HopSizeTooLarge,
    EmptyInput,
    FrameMismatch,
}

impl std::fmt::Display for StftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WindowNotPowerOfTwo(n) => write!(f, "window size {n} not power of 2"),
            Self::HopSizeTooLarge => write!(f, "hop size > window size"),
            Self::EmptyInput => write!(f, "input is empty"),
            Self::FrameMismatch => write!(f, "frame dimensions mismatch"),
        }
    }
}

impl StftConfig {
    pub fn new(window_size: usize, hop_size: usize, window_type: WindowType) -> Result<Self, StftError> {
        if !is_power_of_two(window_size) {
            return Err(StftError::WindowNotPowerOfTwo(window_size));
        }
        if hop_size > window_size || hop_size == 0 {
            return Err(StftError::HopSizeTooLarge);
        }
        Ok(Self { window_size, hop_size, window_type })
    }

    pub fn overlap(&self) -> usize {
        self.window_size - self.hop_size
    }

    pub fn num_frames(&self, signal_len: usize) -> usize {
        if signal_len < self.window_size {
            return 0;
        }
        (signal_len - self.window_size) / self.hop_size + 1
    }
}

// ── STFT Frame ──────────────────────────────────────────────────

/// A single STFT frame containing the complex spectrum.
#[derive(Debug, Clone)]
pub struct StftFrame {
    /// Complex spectrum bins (length = window_size).
    pub spectrum: Vec<Complex>,
}

impl StftFrame {
    pub fn magnitude(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.magnitude()).collect()
    }

    pub fn phase(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.phase()).collect()
    }

    pub fn power(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect()
    }
}

// ── Forward STFT ────────────────────────────────────────────────

/// Compute the STFT of a real signal, returning a vector of frames.
pub fn stft(signal: &[f64], config: &StftConfig) -> Result<Vec<StftFrame>, StftError> {
    if signal.is_empty() {
        return Err(StftError::EmptyInput);
    }
    let window = make_window(config.window_type, config.window_size);
    let num_frames = config.num_frames(signal.len());
    let mut frames = Vec::with_capacity(num_frames);

    for f_idx in 0..num_frames {
        let start = f_idx * config.hop_size;
        let windowed: Vec<Complex> = (0..config.window_size)
            .map(|i| Complex::new(signal[start + i] * window[i], 0.0))
            .collect();
        let spectrum = fft(&windowed);
        frames.push(StftFrame { spectrum });
    }
    Ok(frames)
}

/// Compute spectrogram (time x frequency magnitude matrix).
pub fn spectrogram(signal: &[f64], config: &StftConfig) -> Result<Vec<Vec<f64>>, StftError> {
    let frames = stft(signal, config)?;
    let half = config.window_size / 2 + 1;
    Ok(frames.iter().map(|f| {
        f.spectrum[..half].iter().map(|c| c.magnitude()).collect()
    }).collect())
}

// ── Inverse STFT (Overlap-Add) ──────────────────────────────────

/// Reconstruct a signal from STFT frames using overlap-add.
pub fn istft(frames: &[StftFrame], config: &StftConfig) -> Result<Vec<f64>, StftError> {
    if frames.is_empty() {
        return Err(StftError::EmptyInput);
    }
    let window = make_window(config.window_type, config.window_size);
    let out_len = (frames.len() - 1) * config.hop_size + config.window_size;
    let mut output = vec![0.0; out_len];
    let mut window_sum = vec![0.0; out_len];

    for (f_idx, frame) in frames.iter().enumerate() {
        let time_domain = ifft(&frame.spectrum);
        let start = f_idx * config.hop_size;
        for i in 0..config.window_size {
            output[start + i] += time_domain[i].re * window[i];
            window_sum[start + i] += window[i] * window[i];
        }
    }

    // Normalize by window sum to recover original amplitude
    for i in 0..out_len {
        if window_sum[i] > 1e-10 {
            output[i] /= window_sum[i];
        }
    }
    Ok(output)
}

// ── Phase Vocoder ───────────────────────────────────────────────

/// Time-stretch a signal by `rate` using a phase vocoder.
/// `rate` > 1.0 slows down, < 1.0 speeds up.
pub fn phase_vocoder(
    signal: &[f64],
    config: &StftConfig,
    rate: f64,
) -> Result<Vec<f64>, StftError> {
    if signal.is_empty() {
        return Err(StftError::EmptyInput);
    }
    let frames = stft(signal, config)?;
    if frames.is_empty() {
        return Err(StftError::EmptyInput);
    }

    let n_bins = config.window_size;
    let hop = config.hop_size as f64;
    let synthesis_hop = (hop * rate) as usize;
    let num_out_frames = frames.len();

    // Phase accumulator
    let mut phase_accum = frames[0].phase();
    let mut out_frames = Vec::with_capacity(num_out_frames);

    // First frame keeps original phase
    out_frames.push(frames[0].clone());

    for f_idx in 1..num_out_frames {
        let prev_phase = frames[f_idx - 1].phase();
        let curr_phase = frames[f_idx].phase();
        let curr_mag = frames[f_idx].magnitude();

        let mut new_spectrum = vec![Complex::zero(); n_bins];
        for k in 0..n_bins {
            // Expected phase advance for this bin
            let expected = 2.0 * PI * k as f64 * hop / n_bins as f64;
            // Actual phase difference
            let dp = curr_phase[k] - prev_phase[k] - expected;
            // Wrap to [-pi, pi]
            let wrapped = dp - (dp / (2.0 * PI)).round() * 2.0 * PI;
            // True frequency deviation
            let true_freq = expected + wrapped;
            // Accumulate phase at synthesis hop rate
            phase_accum[k] += true_freq * synthesis_hop as f64 / hop;
            new_spectrum[k] = Complex::from_polar(curr_mag[k], phase_accum[k]);
        }
        out_frames.push(StftFrame { spectrum: new_spectrum });
    }

    // Reconstruct with synthesis hop
    let synth_config = StftConfig {
        window_size: config.window_size,
        hop_size: synthesis_hop.max(1),
        window_type: config.window_type,
    };
    istft(&out_frames, &synth_config)
}

// ── Griffin-Lim ─────────────────────────────────────────────────

/// Griffin-Lim algorithm: reconstruct a signal from magnitude-only spectrogram.
/// `magnitudes`: time x frequency magnitude matrix (each row = half spectrum + 1).
/// `iterations`: number of iterations (typically 30-100).
pub fn griffin_lim(
    magnitudes: &[Vec<f64>],
    config: &StftConfig,
    iterations: usize,
) -> Result<Vec<f64>, StftError> {
    if magnitudes.is_empty() {
        return Err(StftError::EmptyInput);
    }
    let n_frames = magnitudes.len();
    let half = config.window_size / 2 + 1;
    let n_bins = config.window_size;

    // Initialize with random phase (deterministic seed for testing)
    let mut frames: Vec<StftFrame> = magnitudes
        .iter()
        .enumerate()
        .map(|(f_idx, mag_row)| {
            let spectrum: Vec<Complex> = (0..n_bins)
                .map(|k| {
                    let m = if k < half { mag_row.get(k).copied().unwrap_or(0.0) }
                            else { mag_row.get(n_bins - k).copied().unwrap_or(0.0) };
                    // Deterministic initial phase
                    let init_phase = (f_idx * 17 + k * 31) as f64 * 0.1;
                    Complex::from_polar(m, init_phase)
                })
                .collect();
            StftFrame { spectrum }
        })
        .collect();

    let window = make_window(config.window_type, config.window_size);

    for _ in 0..iterations {
        // Reconstruct signal
        let signal = istft(&frames, config)?;

        // Re-analyze
        let new_num = config.num_frames(signal.len());
        let actual_frames = new_num.min(n_frames);
        for f_idx in 0..actual_frames {
            let start = f_idx * config.hop_size;
            let windowed: Vec<Complex> = (0..n_bins)
                .map(|i| {
                    let s = if start + i < signal.len() { signal[start + i] } else { 0.0 };
                    Complex::new(s * window[i], 0.0)
                })
                .collect();
            let spec = fft(&windowed);

            // Keep magnitude from input, take phase from reconstruction
            let new_spec: Vec<Complex> = spec.iter().enumerate().map(|(k, c)| {
                let target_mag = if k < half {
                    magnitudes[f_idx].get(k).copied().unwrap_or(0.0)
                } else {
                    magnitudes[f_idx].get(n_bins - k).copied().unwrap_or(0.0)
                };
                let p = c.phase();
                Complex::from_polar(target_mag, p)
            }).collect();
            frames[f_idx] = StftFrame { spectrum: new_spec };
        }
    }

    istft(&frames, config)
}

// ── Frame processor ─────────────────────────────────────────────

/// Process STFT frames one at a time with a user-supplied closure.
pub fn process_frames<F>(
    signal: &[f64],
    config: &StftConfig,
    mut processor: F,
) -> Result<Vec<f64>, StftError>
where
    F: FnMut(usize, &mut StftFrame),
{
    let mut frames = stft(signal, config)?;
    for (i, frame) in frames.iter_mut().enumerate() {
        processor(i, frame);
    }
    istft(&frames, config)
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
    fn test_stft_config_valid() {
        let cfg = StftConfig::new(256, 64, WindowType::Hann).unwrap();
        assert_eq!(cfg.window_size, 256);
        assert_eq!(cfg.hop_size, 64);
        assert_eq!(cfg.overlap(), 192);
    }

    #[test]
    fn test_stft_config_not_power_of_two() {
        assert!(StftConfig::new(100, 50, WindowType::Hann).is_err());
    }

    #[test]
    fn test_stft_config_hop_too_large() {
        assert!(StftConfig::new(256, 512, WindowType::Hann).is_err());
    }

    #[test]
    fn test_num_frames() {
        let cfg = StftConfig::new(8, 4, WindowType::Rectangular).unwrap();
        assert_eq!(cfg.num_frames(16), 3); // frames at 0, 4, 8
        assert_eq!(cfg.num_frames(8), 1);
        assert_eq!(cfg.num_frames(4), 0);
    }

    #[test]
    fn test_stft_empty_error() {
        let cfg = StftConfig::new(8, 4, WindowType::Hann).unwrap();
        assert!(stft(&[], &cfg).is_err());
    }

    #[test]
    fn test_stft_dc_signal() {
        let cfg = StftConfig::new(8, 4, WindowType::Rectangular).unwrap();
        let signal = vec![1.0; 16];
        let frames = stft(&signal, &cfg).unwrap();
        assert_eq!(frames.len(), 3);
        // DC bin (bin 0) should be ~8.0 for rectangular window
        for frame in &frames {
            assert!(approx_eq(frame.spectrum[0].re, 8.0));
        }
    }

    #[test]
    fn test_stft_frame_magnitude_phase() {
        let cfg = StftConfig::new(4, 2, WindowType::Rectangular).unwrap();
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let frames = stft(&signal, &cfg).unwrap();
        let mag = frames[0].magnitude();
        let _phase = frames[0].phase();
        assert_eq!(mag.len(), 4);
        // DC should be 0 for alternating signal
        assert!(approx_eq(mag[0], 0.0));
    }

    #[test]
    fn test_stft_istft_roundtrip_rectangular() {
        let cfg = StftConfig::new(8, 4, WindowType::Rectangular).unwrap();
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let frames = stft(&signal, &cfg).unwrap();
        let recovered = istft(&frames, &cfg).unwrap();
        // Check that the middle portion matches (edges have boundary effects)
        for i in 4..28 {
            assert!(
                approx_eq(signal[i], recovered[i]),
                "mismatch at {i}: {} vs {}",
                signal[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_stft_istft_roundtrip_hann() {
        let cfg = StftConfig::new(16, 4, WindowType::Hann).unwrap();
        let signal: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let frames = stft(&signal, &cfg).unwrap();
        let recovered = istft(&frames, &cfg).unwrap();
        // Overlap-add with 75% overlap should reconstruct well
        for i in 16..48 {
            assert!(
                (signal[i] - recovered[i]).abs() < 0.05,
                "mismatch at {i}: {} vs {}",
                signal[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_spectrogram_dimensions() {
        let cfg = StftConfig::new(8, 4, WindowType::Hann).unwrap();
        let signal = vec![0.0; 32];
        let spec = spectrogram(&signal, &cfg).unwrap();
        assert_eq!(spec.len(), cfg.num_frames(32));
        // Each row should have window_size/2 + 1 = 5 bins
        for row in &spec {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn test_spectrogram_sine_peak() {
        let cfg = StftConfig::new(16, 8, WindowType::Rectangular).unwrap();
        // Sine at bin 2 (freq = 2/16 of sample rate)
        let signal: Vec<f64> = (0..32)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / 16.0).cos())
            .collect();
        let spec = spectrogram(&signal, &cfg).unwrap();
        // Bin 2 should have the largest magnitude
        for row in &spec {
            let max_bin = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            assert_eq!(max_bin, 2);
        }
    }

    #[test]
    fn test_phase_vocoder_identity() {
        // Rate = 1.0 should produce signal close to original
        let cfg = StftConfig::new(16, 4, WindowType::Hann).unwrap();
        let signal: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let result = phase_vocoder(&signal, &cfg, 1.0).unwrap();
        assert!(!result.is_empty());
        // With rate=1.0, output length should be close to input
        assert!((result.len() as f64 - signal.len() as f64).abs() < 32.0);
    }

    #[test]
    fn test_phase_vocoder_stretch() {
        let cfg = StftConfig::new(16, 8, WindowType::Hann).unwrap();
        let signal: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let stretched = phase_vocoder(&signal, &cfg, 2.0).unwrap();
        // Stretched output should be longer
        assert!(stretched.len() > signal.len());
    }

    #[test]
    fn test_griffin_lim_convergence() {
        let cfg = StftConfig::new(16, 4, WindowType::Hann).unwrap();
        let signal: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let spec = spectrogram(&signal, &cfg).unwrap();
        let reconstructed = griffin_lim(&spec, &cfg, 30).unwrap();
        assert!(!reconstructed.is_empty());
        // After 30 iterations, energy should be positive
        let energy: f64 = reconstructed.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_griffin_lim_empty() {
        let cfg = StftConfig::new(8, 4, WindowType::Hann).unwrap();
        assert!(griffin_lim(&[], &cfg, 10).is_err());
    }

    #[test]
    fn test_process_frames_identity() {
        let cfg = StftConfig::new(8, 4, WindowType::Rectangular).unwrap();
        let signal: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let result = process_frames(&signal, &cfg, |_, _| {
            // No modification → should reconstruct
        }).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_process_frames_zero_out() {
        let cfg = StftConfig::new(8, 4, WindowType::Rectangular).unwrap();
        let signal = vec![1.0; 16];
        let result = process_frames(&signal, &cfg, |_, frame| {
            for c in frame.spectrum.iter_mut() {
                *c = Complex::zero();
            }
        }).unwrap();
        // All frames zeroed → output should be near zero
        for &s in &result {
            assert!(s.abs() < EPS);
        }
    }

    #[test]
    fn test_window_functions() {
        let rect = make_window(WindowType::Rectangular, 8);
        assert!(rect.iter().all(|v| approx_eq(*v, 1.0)));

        let hann = make_window(WindowType::Hann, 8);
        assert!(approx_eq(hann[0], 0.0));
        assert!(approx_eq(hann[4], 1.0)); // peak at center for odd N-1

        let hamming = make_window(WindowType::Hamming, 8);
        assert!(hamming[0] > 0.07); // Hamming doesn't reach zero
    }

    #[test]
    fn test_stft_frame_power() {
        let cfg = StftConfig::new(4, 2, WindowType::Rectangular).unwrap();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let frames = stft(&signal, &cfg).unwrap();
        let power = frames[0].power();
        let mag = frames[0].magnitude();
        for (p, m) in power.iter().zip(mag.iter()) {
            assert!(approx_eq(*p, m * m));
        }
    }

    #[test]
    fn test_istft_empty_error() {
        let cfg = StftConfig::new(8, 4, WindowType::Hann).unwrap();
        assert!(istft(&[], &cfg).is_err());
    }

    #[test]
    fn test_stft_single_frame() {
        let cfg = StftConfig::new(4, 2, WindowType::Rectangular).unwrap();
        let signal = vec![1.0, 0.0, -1.0, 0.0];
        let frames = stft(&signal, &cfg).unwrap();
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_phase_vocoder_empty() {
        let cfg = StftConfig::new(8, 4, WindowType::Hann).unwrap();
        assert!(phase_vocoder(&[], &cfg, 1.0).is_err());
    }
}
