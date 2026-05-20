//! Waveform analysis — peak detection, RMS, zero-crossing, FFT, spectrum, mel scale.
//!
//! Pure Rust implementation of Cooley-Tukey radix-2 DIT FFT for power-of-2 sizes.
//! `WaveformRenderer` downsamples audio to N visual bars for display.

use std::f64::consts::PI;

// ── Peak Detection ──────────────────────────────────────────────

/// Detected peak with sample index and amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    pub index: usize,
    pub amplitude: f32,
}

/// Find local maxima above `threshold` with at least `min_distance` samples apart.
pub fn detect_peaks(samples: &[f32], threshold: f32, min_distance: usize) -> Vec<Peak> {
    let mut peaks = Vec::new();
    if samples.len() < 3 {
        return peaks;
    }

    for i in 1..samples.len() - 1 {
        if samples[i] > samples[i - 1]
            && samples[i] > samples[i + 1]
            && samples[i] >= threshold
        {
            // Check distance from last peak
            if let Some(last) = peaks.last() {
                let last_peak: &Peak = last;
                if i - last_peak.index < min_distance {
                    // Keep the larger peak
                    if samples[i] > last_peak.amplitude {
                        peaks.pop();
                        peaks.push(Peak {
                            index: i,
                            amplitude: samples[i],
                        });
                    }
                    continue;
                }
            }
            peaks.push(Peak {
                index: i,
                amplitude: samples[i],
            });
        }
    }

    peaks
}

// ── RMS Level ───────────────────────────────────────────────────

/// Compute the RMS (root mean square) level of a signal.
pub fn rms_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Compute the peak level (maximum absolute value).
pub fn peak_level(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// Compute RMS level in dB (relative to 1.0).
pub fn rms_db(samples: &[f32]) -> f32 {
    let rms = rms_level(samples);
    if rms <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * rms.log10()
}

// ── Zero-Crossing Rate ─────────────────────────────────────────

/// Count the number of zero crossings in a signal.
pub fn zero_crossing_count(samples: &[f32]) -> usize {
    if samples.len() < 2 {
        return 0;
    }
    let mut count = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0 && samples[i - 1] < 0.0)
            || (samples[i] < 0.0 && samples[i - 1] >= 0.0)
        {
            count += 1;
        }
    }
    count
}

/// Zero-crossing rate (crossings per sample).
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    zero_crossing_count(samples) as f32 / (samples.len() - 1) as f32
}

// ── Waveform Renderer ───────────────────────────────────────────

/// A single bar in the waveform visualization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveformBar {
    pub min: f32,
    pub max: f32,
    pub rms: f32,
}

/// Downsample audio to N visual bars for waveform display.
pub struct WaveformRenderer;

impl WaveformRenderer {
    /// Render `num_bars` bars from the given samples.
    pub fn render(samples: &[f32], num_bars: usize) -> Vec<WaveformBar> {
        if num_bars == 0 || samples.is_empty() {
            return Vec::new();
        }

        let mut bars = Vec::with_capacity(num_bars);
        let samples_per_bar = samples.len() as f64 / num_bars as f64;

        for i in 0..num_bars {
            let start = (i as f64 * samples_per_bar) as usize;
            let end = ((i + 1) as f64 * samples_per_bar) as usize;
            let end = end.min(samples.len());

            if start >= end {
                bars.push(WaveformBar {
                    min: 0.0,
                    max: 0.0,
                    rms: 0.0,
                });
                continue;
            }

            let chunk = &samples[start..end];
            let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
            let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let rms = rms_level(chunk);

            bars.push(WaveformBar { min, max, rms });
        }

        bars
    }
}

// ── FFT (Cooley-Tukey Radix-2 DIT) ─────────────────────────────

/// Complex number for FFT computation.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// Check if n is a power of 2.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Bit-reverse permutation index.
fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Cooley-Tukey radix-2 DIT FFT. Input length must be a power of 2.
/// Returns `None` if length is not a power of 2.
pub fn fft(input: &[Complex]) -> Option<Vec<Complex>> {
    let n = input.len();
    if !is_power_of_two(n) {
        return None;
    }
    if n <= 1 {
        return Some(input.to_vec());
    }

    let bits = (n as f64).log2() as u32;

    // Bit-reversal permutation
    let mut data: Vec<Complex> = (0..n)
        .map(|i| input[bit_reverse(i, bits)])
        .collect();

    // Butterfly operations
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle = -2.0 * PI / size as f64;

        for start in (0..n).step_by(size) {
            for k in 0..half {
                let twiddle = Complex::new(
                    (angle * k as f64).cos(),
                    (angle * k as f64).sin(),
                );
                let a = data[start + k];
                let b = data[start + k + half].mul(twiddle);
                data[start + k] = a.add(b);
                data[start + k + half] = a.sub(b);
            }
        }

        size *= 2;
    }

    Some(data)
}

/// Convenience: compute FFT of real-valued samples (zero-padded to power of 2 if needed).
pub fn fft_real(samples: &[f32]) -> Vec<Complex> {
    let n = samples.len().next_power_of_two();
    let mut input: Vec<Complex> = samples
        .iter()
        .map(|s| Complex::new(*s as f64, 0.0))
        .collect();
    input.resize(n, Complex::new(0.0, 0.0));
    fft(&input).unwrap_or_default()
}

// ── Spectrum ────────────────────────────────────────────────────

/// Magnitude spectrum bin.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumBin {
    pub frequency: f64,
    pub magnitude: f64,
}

/// Compute the magnitude spectrum from FFT output.
pub fn magnitude_spectrum(fft_output: &[Complex], sample_rate: f64) -> Vec<SpectrumBin> {
    let n = fft_output.len();
    let nyquist = n / 2;
    let freq_resolution = sample_rate / n as f64;

    (0..nyquist)
        .map(|i| SpectrumBin {
            frequency: i as f64 * freq_resolution,
            magnitude: fft_output[i].magnitude() / n as f64,
        })
        .collect()
}

/// Find the dominant frequency in a signal.
pub fn dominant_frequency(samples: &[f32], sample_rate: f64) -> Option<f64> {
    let spectrum = magnitude_spectrum(&fft_real(samples), sample_rate);
    // Skip DC (bin 0)
    spectrum
        .iter()
        .skip(1)
        .max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap_or(std::cmp::Ordering::Equal))
        .map(|bin| bin.frequency)
}

// ── Mel Scale ───────────────────────────────────────────────────

/// Convert frequency in Hz to mel scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Generate `num_filters` mel-scale filter bank center frequencies.
pub fn mel_filter_centers(low_hz: f64, high_hz: f64, num_filters: usize) -> Vec<f64> {
    let low_mel = hz_to_mel(low_hz);
    let high_mel = hz_to_mel(high_hz);

    (0..num_filters)
        .map(|i| {
            let mel = low_mel + (high_mel - low_mel) * i as f64 / (num_filters - 1).max(1) as f64;
            mel_to_hz(mel)
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_detection_simple() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, 0.3, 0.8, 0.3, 0.0];
        let peaks = detect_peaks(&samples, 0.5, 1);
        assert_eq!(peaks.len(), 2);
        assert_eq!(peaks[0].index, 2);
        assert_eq!(peaks[0].amplitude, 1.0);
        assert_eq!(peaks[1].index, 6);
    }

    #[test]
    fn peak_detection_min_distance() {
        let samples = vec![0.0, 1.0, 0.5, 0.9, 0.0];
        // With min_distance=3, the two peaks are too close; keep the larger
        let peaks = detect_peaks(&samples, 0.5, 3);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].amplitude, 1.0);
    }

    #[test]
    fn rms_of_dc() {
        let samples = vec![0.5f32; 100];
        let rms = rms_level(&samples);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn rms_of_sine() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let rms = rms_level(&samples);
        // RMS of a sine wave is 1/sqrt(2) ≈ 0.7071
        assert!((rms - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01);
    }

    #[test]
    fn rms_db_test() {
        let samples = vec![1.0f32; 100];
        let db = rms_db(&samples);
        assert!(db.abs() < 0.01); // 0 dB
    }

    #[test]
    fn zero_crossing_sine() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let crossings = zero_crossing_count(&samples);
        // A 440 Hz sine at 44100 Hz should have ~880 crossings per second
        assert!((crossings as i32 - 880).unsigned_abs() < 10);
    }

    #[test]
    fn waveform_renderer_basic() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0) - 1.0).collect();
        let bars = WaveformRenderer::render(&samples, 10);
        assert_eq!(bars.len(), 10);
        // First bar should have negative values, last bar should have positive
        assert!(bars[0].min < 0.0);
        assert!(bars[9].max > 0.0);
    }

    #[test]
    fn fft_impulse() {
        // FFT of an impulse should be flat spectrum
        let mut input = vec![Complex::new(0.0, 0.0); 8];
        input[0] = Complex::new(1.0, 0.0);
        let result = fft(&input).unwrap();
        for bin in &result {
            assert!((bin.magnitude() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn fft_dc() {
        // FFT of a constant signal should have energy only in bin 0
        let input: Vec<Complex> = vec![Complex::new(1.0, 0.0); 8];
        let result = fft(&input).unwrap();
        assert!((result[0].magnitude() - 8.0).abs() < 1e-10);
        for bin in &result[1..] {
            assert!(bin.magnitude() < 1e-10);
        }
    }

    #[test]
    fn fft_non_power_of_two_returns_none() {
        let input = vec![Complex::new(1.0, 0.0); 7];
        assert!(fft(&input).is_none());
    }

    #[test]
    fn mel_round_trip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((back - hz).abs() < 0.01);
    }

    #[test]
    fn mel_filter_centers_monotonic() {
        let centers = mel_filter_centers(0.0, 8000.0, 26);
        assert_eq!(centers.len(), 26);
        for i in 1..centers.len() {
            assert!(centers[i] > centers[i - 1]);
        }
    }

    #[test]
    fn dominant_frequency_detection() {
        // Generate 440 Hz sine
        let sr = 8192.0;
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let freq = dominant_frequency(&samples, sr).unwrap();
        assert!((freq - 440.0).abs() < 2.0, "Detected {} Hz, expected 440 Hz", freq);
    }

    #[test]
    fn peak_level_test() {
        let samples = vec![-0.3, 0.5, -0.8, 0.2];
        assert!((peak_level(&samples) - 0.8).abs() < 1e-6);
    }
}
