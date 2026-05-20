//! Audio analysis — RMS, peak, zero crossing, spectral features, onset/beat detection.
//!
//! Provides signal analysis tools for audio: level metering, spectral analysis
//! (centroid, rolloff), onset detection, simple beat detection via autocorrelation,
//! and loudness estimation (simplified ITU-R BS.1770).

use std::f64::consts::PI;

// ── Level Metering ──────────────────────────────────────────────

/// Calculate RMS (root mean square) level of a buffer.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Calculate peak (maximum absolute value) of a buffer.
pub fn peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
}

/// Calculate crest factor (peak / RMS ratio).
pub fn crest_factor(samples: &[f32]) -> f32 {
    let r = rms(samples);
    if r < 1e-10 {
        return 0.0;
    }
    peak(samples) / r
}

/// Calculate RMS in dB (relative to 1.0).
pub fn rms_db(samples: &[f32]) -> f32 {
    let r = rms(samples);
    if r <= 0.0 {
        return -120.0;
    }
    20.0 * r.log10()
}

/// Calculate peak in dB.
pub fn peak_db(samples: &[f32]) -> f32 {
    let p = peak(samples);
    if p <= 0.0 {
        return -120.0;
    }
    20.0 * p.log10()
}

/// Calculate RMS for overlapping windows, returning one RMS value per hop.
pub fn rms_windowed(samples: &[f32], window_size: usize, hop_size: usize) -> Vec<f32> {
    if samples.is_empty() || window_size == 0 || hop_size == 0 {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut pos = 0;
    while pos + window_size <= samples.len() {
        result.push(rms(&samples[pos..pos + window_size]));
        pos += hop_size;
    }
    result
}

// ── Zero Crossing Rate ─────────────────────────────────────────

/// Calculate the zero crossing rate (crossings per sample).
pub fn zero_crossing_rate(samples: &[f32]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0u64;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f64 / (samples.len() - 1) as f64
}

/// Calculate zero crossing rate for overlapping windows.
pub fn zcr_windowed(samples: &[f32], window_size: usize, hop_size: usize) -> Vec<f64> {
    if samples.is_empty() || window_size == 0 || hop_size == 0 {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut pos = 0;
    while pos + window_size <= samples.len() {
        result.push(zero_crossing_rate(&samples[pos..pos + window_size]));
        pos += hop_size;
    }
    result
}

// ── Spectral Analysis (DFT-based) ──────────────────────────────

/// Compute the magnitude spectrum using a simple DFT (not FFT — O(N^2), suitable
/// for small windows). Returns N/2+1 magnitude bins.
pub fn magnitude_spectrum(samples: &[f32]) -> Vec<f64> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    let half = n / 2 + 1;
    let mut magnitudes = Vec::with_capacity(half);

    for k in 0..half {
        let mut real = 0.0f64;
        let mut imag = 0.0f64;
        for (i, sample) in samples.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
            real += *sample as f64 * angle.cos();
            imag += *sample as f64 * angle.sin();
        }
        magnitudes.push((real * real + imag * imag).sqrt());
    }

    magnitudes
}

/// Compute the power spectrum (magnitude squared).
pub fn power_spectrum(samples: &[f32]) -> Vec<f64> {
    magnitude_spectrum(samples)
        .iter()
        .map(|m| m * m)
        .collect()
}

/// Calculate spectral centroid (weighted mean frequency).
/// Returns frequency in Hz.
pub fn spectral_centroid(samples: &[f32], sample_rate: f64) -> f64 {
    let mags = magnitude_spectrum(samples);
    if mags.is_empty() {
        return 0.0;
    }
    let n = samples.len();
    let freq_resolution = sample_rate / n as f64;

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (k, mag) in mags.iter().enumerate() {
        let freq = k as f64 * freq_resolution;
        weighted_sum += freq * mag;
        total_weight += mag;
    }

    if total_weight < 1e-10 {
        return 0.0;
    }

    weighted_sum / total_weight
}

/// Calculate spectral rolloff — the frequency below which `percentage` (0.0-1.0)
/// of the total spectral energy is contained.
pub fn spectral_rolloff(samples: &[f32], sample_rate: f64, percentage: f64) -> f64 {
    let power = power_spectrum(samples);
    if power.is_empty() {
        return 0.0;
    }
    let n = samples.len();
    let freq_resolution = sample_rate / n as f64;

    let total_energy: f64 = power.iter().sum();
    let threshold = total_energy * percentage.clamp(0.0, 1.0);

    let mut cumulative = 0.0;
    for (k, p) in power.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return k as f64 * freq_resolution;
        }
    }

    (power.len() - 1) as f64 * freq_resolution
}

/// Calculate spectral bandwidth (weighted standard deviation around centroid).
pub fn spectral_bandwidth(samples: &[f32], sample_rate: f64) -> f64 {
    let mags = magnitude_spectrum(samples);
    if mags.is_empty() {
        return 0.0;
    }
    let centroid = spectral_centroid(samples, sample_rate);
    let n = samples.len();
    let freq_res = sample_rate / n as f64;

    let mut weighted_var = 0.0;
    let mut total_weight = 0.0;
    for (k, mag) in mags.iter().enumerate() {
        let freq = k as f64 * freq_res;
        let diff = freq - centroid;
        weighted_var += diff * diff * mag;
        total_weight += mag;
    }

    if total_weight < 1e-10 {
        return 0.0;
    }

    (weighted_var / total_weight).sqrt()
}

/// Calculate spectral flatness (geometric mean / arithmetic mean of power spectrum).
/// Range 0..1 where 1 = white noise, 0 = tonal.
pub fn spectral_flatness(samples: &[f32]) -> f64 {
    let power = power_spectrum(samples);
    if power.is_empty() {
        return 0.0;
    }

    // Filter out zero values
    let nonzero: Vec<f64> = power.iter().copied().filter(|p| *p > 1e-30).collect();
    if nonzero.is_empty() {
        return 0.0;
    }

    let n = nonzero.len() as f64;
    let log_sum: f64 = nonzero.iter().map(|p| p.ln()).sum();
    let geometric_mean = (log_sum / n).exp();
    let arithmetic_mean: f64 = nonzero.iter().sum::<f64>() / n;

    if arithmetic_mean < 1e-30 {
        return 0.0;
    }

    (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
}

// ── Onset Detection ─────────────────────────────────────────────

/// Simple onset detection using spectral flux.
/// Returns a vector of onset strengths per frame.
pub fn onset_detection(
    samples: &[f32],
    window_size: usize,
    hop_size: usize,
) -> Vec<f64> {
    if samples.len() < window_size || hop_size == 0 {
        return Vec::new();
    }

    let mut prev_spectrum: Option<Vec<f64>> = None;
    let mut onsets = Vec::new();
    let mut pos = 0;

    while pos + window_size <= samples.len() {
        let spectrum = magnitude_spectrum(&samples[pos..pos + window_size]);

        if let Some(prev) = &prev_spectrum {
            // Half-wave rectified spectral flux
            let flux: f64 = spectrum
                .iter()
                .zip(prev.iter())
                .map(|(curr, prev_val)| {
                    let diff = curr - prev_val;
                    if diff > 0.0 { diff } else { 0.0 }
                })
                .sum();
            onsets.push(flux);
        } else {
            onsets.push(0.0);
        }

        prev_spectrum = Some(spectrum);
        pos += hop_size;
    }

    onsets
}

/// Pick onset times from onset strength signal using a threshold.
/// Returns frame indices where onsets are detected.
pub fn pick_onsets(onset_strength: &[f64], threshold: f64, min_distance: usize) -> Vec<usize> {
    let mut peaks = Vec::new();
    let mut last_onset: Option<usize> = None;

    for i in 1..onset_strength.len().saturating_sub(1) {
        if onset_strength[i] > threshold
            && onset_strength[i] > onset_strength[i - 1]
            && onset_strength[i] >= onset_strength[i + 1]
        {
            if let Some(last) = last_onset {
                if i - last < min_distance {
                    continue;
                }
            }
            peaks.push(i);
            last_onset = Some(i);
        }
    }

    peaks
}

// ── Beat Detection ──────────────────────────────────────────────

/// Simple beat detection using autocorrelation of onset strength.
/// Returns estimated BPM.
pub fn detect_bpm(samples: &[f32], sample_rate: f64) -> f64 {
    let window = 1024usize;
    let hop = 512usize;

    if samples.len() < window {
        return 0.0;
    }

    let onset_str = onset_detection(samples, window, hop);
    if onset_str.len() < 4 {
        return 0.0;
    }

    // Autocorrelation of onset strength
    let frame_rate = sample_rate / hop as f64;
    let min_lag = (frame_rate * 60.0 / 200.0) as usize; // 200 BPM
    let max_lag = (frame_rate * 60.0 / 40.0) as usize; // 40 BPM
    let max_lag = max_lag.min(onset_str.len() / 2);

    if min_lag >= max_lag {
        return 0.0;
    }

    let mut best_lag = min_lag;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in min_lag..max_lag {
        let mut corr = 0.0;
        let count = onset_str.len() - lag;
        for i in 0..count {
            corr += onset_str[i] * onset_str[i + lag];
        }
        corr /= count as f64;

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_lag == 0 {
        return 0.0;
    }

    // Convert lag to BPM
    let beat_period_seconds = best_lag as f64 / frame_rate;
    60.0 / beat_period_seconds
}

// ── Loudness Estimation (simplified ITU-R BS.1770) ──────────────

/// K-weighting pre-filter stage 1 coefficients (high-shelf boost ~+4dB at high freq).
/// Simplified approximation at 48kHz.
fn k_weight_stage1(sample_rate: f64) -> KWeightCoeffs {
    // High shelf at ~1500 Hz, +4dB
    let fc = 1681.974450955533;
    let g = 3.999843853973347; // dB
    let q_val = 0.7071752369554196;

    let k = (PI * fc / sample_rate).tan();
    let vh = 10.0f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);

    let a0 = 1.0 + k / q_val + k * k;
    let b0 = (1.0 + vb * k / q_val + vh * k * k) / a0;
    let b1 = (2.0 * (vh * k * k - 1.0)) / a0;
    let b2 = (1.0 - vb * k / q_val + vh * k * k) / a0;
    let a1 = (2.0 * (k * k - 1.0)) / a0;
    let a2 = (1.0 - k / q_val + k * k) / a0;

    KWeightCoeffs {
        b0,
        b1,
        b2,
        a1,
        a2,
    }
}

/// K-weighting pre-filter stage 2 (highpass at ~38Hz).
fn k_weight_stage2(sample_rate: f64) -> KWeightCoeffs {
    let fc = 38.13547087602444;
    let q_val = 0.5003270373238773;

    let w0 = 2.0 * PI * fc / sample_rate;
    let alpha = w0.sin() / (2.0 * q_val);
    let cos_w0 = w0.cos();

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    KWeightCoeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

#[derive(Debug, Clone, Copy)]
struct KWeightCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

struct KWeightFilter {
    coeffs: KWeightCoeffs,
    z1: f64,
    z2: f64,
}

impl KWeightFilter {
    fn new(coeffs: KWeightCoeffs) -> Self {
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn tick(&mut self, input: f64) -> f64 {
        let y = self.coeffs.b0 * input + self.z1;
        self.z1 = self.coeffs.b1 * input - self.coeffs.a1 * y + self.z2;
        self.z2 = self.coeffs.b2 * input - self.coeffs.a2 * y;
        y
    }
}

/// Estimate integrated loudness in LUFS (Loudness Units Full Scale).
/// Simplified single-channel implementation of ITU-R BS.1770.
pub fn loudness_lufs(samples: &[f32], sample_rate: f64) -> f64 {
    if samples.is_empty() {
        return -120.0;
    }

    // Apply K-weighting
    let s1_coeffs = k_weight_stage1(sample_rate);
    let s2_coeffs = k_weight_stage2(sample_rate);
    let mut stage1 = KWeightFilter::new(s1_coeffs);
    let mut stage2 = KWeightFilter::new(s2_coeffs);

    let weighted: Vec<f64> = samples
        .iter()
        .map(|s| {
            let s1 = stage1.tick(*s as f64);
            stage2.tick(s1)
        })
        .collect();

    // Mean square
    let ms: f64 = weighted.iter().map(|s| s * s).sum::<f64>() / weighted.len() as f64;

    if ms <= 0.0 {
        return -120.0;
    }

    // LUFS = -0.691 + 10 * log10(ms)
    -0.691 + 10.0 * ms.log10()
}

/// Short-term loudness (over a window, typically 3 seconds).
pub fn loudness_windowed(
    samples: &[f32],
    sample_rate: f64,
    window_secs: f64,
    hop_secs: f64,
) -> Vec<f64> {
    let window_samples = (window_secs * sample_rate) as usize;
    let hop_samples = (hop_secs * sample_rate) as usize;

    if window_samples == 0 || hop_samples == 0 || samples.len() < window_samples {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut pos = 0;
    while pos + window_samples <= samples.len() {
        result.push(loudness_lufs(
            &samples[pos..pos + window_samples],
            sample_rate,
        ));
        pos += hop_samples;
    }
    result
}

// ── Energy ──────────────────────────────────────────────────────

/// Calculate total energy of a signal.
pub fn energy(samples: &[f32]) -> f64 {
    samples.iter().map(|s| (*s as f64) * (*s as f64)).sum()
}

/// Calculate energy per window.
pub fn energy_windowed(samples: &[f32], window_size: usize, hop_size: usize) -> Vec<f64> {
    if samples.is_empty() || window_size == 0 || hop_size == 0 {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut pos = 0;
    while pos + window_size <= samples.len() {
        result.push(energy(&samples[pos..pos + window_size]));
        pos += hop_size;
    }
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, sample_rate: f64, duration_secs: f64) -> Vec<f32> {
        let n = (sample_rate * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin() as f32)
            .collect()
    }

    #[test]
    fn rms_of_sine() {
        let samples = sine_wave(440.0, 44100.0, 1.0);
        let r = rms(&samples);
        // RMS of sine wave = 1/sqrt(2) ~ 0.707
        assert!(
            (r - 0.707).abs() < 0.01,
            "RMS of sine should be ~0.707, got {r}"
        );
    }

    #[test]
    fn peak_of_sine() {
        let samples = sine_wave(440.0, 44100.0, 1.0);
        let p = peak(&samples);
        assert!((p - 1.0).abs() < 0.01, "peak should be ~1.0, got {p}");
    }

    #[test]
    fn rms_of_silence() {
        let samples = vec![0.0f32; 1000];
        assert!(rms(&samples) < 1e-10);
        assert!(rms_db(&samples) < -100.0);
    }

    #[test]
    fn crest_factor_sine() {
        let samples = sine_wave(440.0, 44100.0, 1.0);
        let cf = crest_factor(&samples);
        // Crest factor of sine = sqrt(2) ~ 1.414
        assert!(
            (cf - 1.414).abs() < 0.02,
            "crest factor should be ~1.414, got {cf}"
        );
    }

    #[test]
    fn rms_windowed_basic() {
        let samples = vec![1.0f32; 1000];
        let windowed = rms_windowed(&samples, 100, 100);
        assert_eq!(windowed.len(), 10);
        assert!((windowed[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_crossing_rate_square_wave() {
        // Square wave: alternating +1/-1
        let samples: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let zcr = zero_crossing_rate(&samples);
        // Should have crossing at every sample
        assert!((zcr - 1.0).abs() < 0.01, "ZCR should be ~1.0, got {zcr}");
    }

    #[test]
    fn zero_crossing_rate_dc() {
        let samples = vec![1.0f32; 100];
        let zcr = zero_crossing_rate(&samples);
        assert!(zcr < 0.01, "DC should have no crossings");
    }

    #[test]
    fn zcr_windowed_basic() {
        let samples: Vec<f32> = (0..400)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let windowed = zcr_windowed(&samples, 100, 100);
        assert_eq!(windowed.len(), 4);
    }

    #[test]
    fn spectral_centroid_low_freq() {
        // Low frequency should have low centroid
        let low = sine_wave(100.0, 44100.0, 0.1);
        let high = sine_wave(5000.0, 44100.0, 0.1);
        let c_low = spectral_centroid(&low[..256], 44100.0);
        let c_high = spectral_centroid(&high[..256], 44100.0);
        assert!(
            c_low < c_high,
            "100Hz centroid ({c_low}) should be < 5000Hz centroid ({c_high})"
        );
    }

    #[test]
    fn spectral_rolloff_basic() {
        let samples = sine_wave(440.0, 44100.0, 0.1);
        let rolloff = spectral_rolloff(&samples[..512], 44100.0, 0.85);
        // Most energy should be near 440 Hz
        assert!(rolloff > 100.0 && rolloff < 5000.0, "rolloff = {rolloff}");
    }

    #[test]
    fn spectral_bandwidth_tone_vs_noise() {
        let tone = sine_wave(440.0, 44100.0, 0.05);
        let bw_tone = spectral_bandwidth(&tone[..256], 44100.0);
        // A pure tone should have relatively narrow bandwidth
        assert!(bw_tone < 10000.0, "tone bandwidth = {bw_tone}");
    }

    #[test]
    fn spectral_flatness_tone() {
        let tone = sine_wave(440.0, 44100.0, 0.05);
        let flat = spectral_flatness(&tone[..256]);
        // Pure tone should have low flatness
        assert!(flat < 0.3, "tone flatness should be low, got {flat}");
    }

    #[test]
    fn onset_detection_basic() {
        let mut samples = vec![0.0f32; 4096];
        // Add a sudden burst at frame 1024
        for s in &mut samples[1024..1536] {
            *s = 0.8;
        }
        let onsets = onset_detection(&samples, 512, 256);
        assert!(!onsets.is_empty());
        // There should be a spike near the burst
        let max_onset = onsets.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_onset > 0.0);
    }

    #[test]
    fn pick_onsets_basic() {
        let strength = vec![0.0, 0.1, 0.5, 1.0, 0.3, 0.0, 0.0, 0.8, 1.2, 0.5, 0.0];
        let picks = pick_onsets(&strength, 0.4, 2);
        assert!(!picks.is_empty());
        // Should find at least the peak at index 3 or 8
        assert!(picks.contains(&3) || picks.contains(&8));
    }

    #[test]
    fn detect_bpm_returns_reasonable() {
        // Generate a click track at ~120 BPM (0.5 second intervals)
        let sr = 44100.0;
        let mut samples = vec![0.0f32; (sr * 4.0) as usize]; // 4 seconds
        let interval = (sr * 0.5) as usize; // 120 BPM = 0.5s
        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + 100).min(samples.len());
            for s in &mut samples[pos..end] {
                *s = 1.0;
            }
            pos += interval;
        }
        let bpm = detect_bpm(&samples, sr);
        // BPM detection is approximate; just check it's in a reasonable range
        assert!(
            bpm > 30.0 && bpm < 300.0,
            "BPM should be reasonable, got {bpm}"
        );
    }

    #[test]
    fn loudness_lufs_silence() {
        let samples = vec![0.0f32; 44100];
        let lufs = loudness_lufs(&samples, 44100.0);
        assert!(lufs < -100.0, "silence should be very quiet: {lufs} LUFS");
    }

    #[test]
    fn loudness_lufs_sine() {
        let samples = sine_wave(1000.0, 48000.0, 1.0);
        let lufs = loudness_lufs(&samples, 48000.0);
        // Full scale sine ~-3 LUFS
        assert!(
            lufs > -10.0 && lufs < 0.0,
            "full scale sine should be near -3 LUFS, got {lufs}"
        );
    }

    #[test]
    fn loudness_windowed_basic() {
        let samples = sine_wave(440.0, 44100.0, 3.0);
        let windowed = loudness_windowed(&samples, 44100.0, 0.4, 0.1);
        assert!(!windowed.is_empty());
        // All windows should have similar loudness
        for &l in &windowed {
            assert!(l > -10.0 && l < 5.0, "window loudness = {l}");
        }
    }

    #[test]
    fn energy_basic() {
        let samples = vec![1.0f32; 100];
        let e = energy(&samples);
        assert!((e - 100.0).abs() < 1e-6);
    }

    #[test]
    fn energy_windowed_basic() {
        let samples = vec![1.0f32; 400];
        let windowed = energy_windowed(&samples, 100, 100);
        assert_eq!(windowed.len(), 4);
        assert!((windowed[0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn magnitude_spectrum_dc() {
        let samples = vec![1.0f32; 64];
        let mags = magnitude_spectrum(&samples);
        // DC bin should be large, all others small
        assert!(mags[0] > 50.0);
        for mag in &mags[1..] {
            assert!(*mag < 1.0, "non-DC bin should be small, got {mag}");
        }
    }

    #[test]
    fn peak_db_unity() {
        let samples = vec![1.0f32; 10];
        let db = peak_db(&samples);
        assert!(db.abs() < 0.1, "peak dB of 1.0 should be ~0, got {db}");
    }

    #[test]
    fn empty_samples() {
        assert!(rms(&[]) < 1e-10);
        assert!(peak(&[]) < 1e-10);
        assert!(zero_crossing_rate(&[]) < 1e-10);
        assert!(magnitude_spectrum(&[]).is_empty());
    }
}
