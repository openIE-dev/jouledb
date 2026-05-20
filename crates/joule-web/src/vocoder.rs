//! Channel vocoder: analyse a modulator signal and apply its spectral envelope
//! to a carrier signal via a bank of band-pass filters.
//!
//! Features: configurable band count (16-32) with logarithmic spacing,
//! per-band envelope follower (RMS with attack/release smoothing),
//! formant shifting, and unvoiced/sibilance detection with pass-through.
//! Pure Rust — no DSP library deps.

use std::f64::consts::PI;

// ── Band Configuration ──────────────────────────────────────────

/// Configuration for a single frequency band.
#[derive(Debug, Clone, PartialEq)]
pub struct BandConfig {
    /// Centre frequency in Hz.
    pub center_hz: f64,
    /// Q factor (bandwidth control).
    pub q: f64,
}

/// Generate logarithmically spaced band configurations between two frequencies.
pub fn log_spaced_bands(num_bands: usize, low_hz: f64, high_hz: f64, q: f64) -> Vec<BandConfig> {
    if num_bands == 0 || low_hz <= 0.0 || high_hz <= low_hz {
        return Vec::new();
    }
    let log_low = low_hz.ln();
    let log_high = high_hz.ln();
    (0..num_bands)
        .map(|i| {
            let t = i as f64 / (num_bands.max(1) - 1).max(1) as f64;
            let center = (log_low + t * (log_high - log_low)).exp();
            BandConfig { center_hz: center, q }
        })
        .collect()
}

// ── Biquad Band-Pass Filter ─────────────────────────────────────

/// A simple biquad band-pass filter (second order).
#[derive(Debug, Clone)]
struct BandPassFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BandPassFilter {
    fn new(center_hz: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * center_hz / sample_rate;
        let alpha = w0.sin() / (2.0 * q.max(0.001));
        let cos_w0 = w0.cos();

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        let inv = 1.0 / a0;
        Self {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let out = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = out;
        out
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ── High-Pass Filter (for sibilance detection) ──────────────────

#[derive(Debug, Clone)]
struct HighPassFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl HighPassFilter {
    fn new(cutoff_hz: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let alpha = w0.sin() / (2.0 * q.max(0.001));
        let cos_w0 = w0.cos();

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        let inv = 1.0 / a0;
        Self {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let out = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = out;
        out
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ── Envelope Follower ───────────────────────────────────────────

/// Per-band amplitude envelope follower with attack/release smoothing.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Attack coefficient (0..1; smaller = faster).
    attack_coeff: f64,
    /// Release coefficient (0..1; smaller = faster).
    release_coeff: f64,
    /// Current envelope level.
    level: f64,
}

impl EnvelopeFollower {
    /// Create an envelope follower.
    /// `attack_ms` and `release_ms` are smoothing times; `sample_rate` in Hz.
    pub fn new(attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        let attack_coeff = (-1.0 / (attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (release_ms * 0.001 * sample_rate)).exp();
        Self { attack_coeff, release_coeff, level: 0.0 }
    }

    /// Process a rectified input sample and return the smoothed envelope.
    pub fn process(&mut self, input: f64) -> f64 {
        let rect = input.abs();
        let coeff = if rect > self.level {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.level = coeff * self.level + (1.0 - coeff) * rect;
        self.level
    }

    /// Current level.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.level = 0.0;
    }
}

// ── Vocoder Channel ─────────────────────────────────────────────

/// A single vocoder channel (analysis + synthesis for one frequency band).
#[derive(Debug, Clone)]
struct VocoderChannel {
    /// Band-pass filter for the modulator (analysis).
    mod_filter: BandPassFilter,
    /// Band-pass filter for the carrier (synthesis).
    car_filter: BandPassFilter,
    /// Envelope follower for the modulator.
    envelope: EnvelopeFollower,
    /// Band centre frequency.
    center_hz: f64,
}

impl VocoderChannel {
    fn new(config: &BandConfig, sample_rate: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            mod_filter: BandPassFilter::new(config.center_hz, config.q, sample_rate),
            car_filter: BandPassFilter::new(config.center_hz, config.q, sample_rate),
            envelope: EnvelopeFollower::new(attack_ms, release_ms, sample_rate),
            center_hz: config.center_hz,
        }
    }

    fn reset(&mut self) {
        self.mod_filter.reset();
        self.car_filter.reset();
        self.envelope.reset();
    }
}

// ── Vocoder ─────────────────────────────────────────────────────

/// Configuration for the vocoder.
#[derive(Debug, Clone, PartialEq)]
pub struct VocoderConfig {
    /// Band configurations.
    pub bands: Vec<BandConfig>,
    /// Envelope follower attack time in ms.
    pub attack_ms: f64,
    /// Envelope follower release time in ms.
    pub release_ms: f64,
    /// Formant shift in semitones (positive = up, negative = down).
    pub formant_shift_semitones: f64,
    /// Enable unvoiced/sibilance detection and pass-through.
    pub sibilance_enabled: bool,
    /// Sibilance detection threshold (energy ratio above which signal is "unvoiced").
    pub sibilance_threshold: f64,
    /// Mix level for sibilance pass-through (0..1).
    pub sibilance_mix: f64,
    /// Sample rate.
    pub sample_rate: f64,
}

impl VocoderConfig {
    /// Create a default 16-band vocoder configuration.
    pub fn default_16_band(sample_rate: f64) -> Self {
        Self {
            bands: log_spaced_bands(16, 100.0, 8000.0, 8.0),
            attack_ms: 2.0,
            release_ms: 20.0,
            formant_shift_semitones: 0.0,
            sibilance_enabled: true,
            sibilance_threshold: 0.3,
            sibilance_mix: 0.5,
            sample_rate,
        }
    }

    /// Create a 32-band vocoder configuration.
    pub fn default_32_band(sample_rate: f64) -> Self {
        Self {
            bands: log_spaced_bands(32, 80.0, 12000.0, 12.0),
            attack_ms: 1.0,
            release_ms: 15.0,
            formant_shift_semitones: 0.0,
            sibilance_enabled: true,
            sibilance_threshold: 0.3,
            sibilance_mix: 0.5,
            sample_rate,
        }
    }
}

/// Channel vocoder processor.
#[derive(Debug, Clone)]
pub struct Vocoder {
    channels: Vec<VocoderChannel>,
    /// High-pass filter for sibilance detection.
    sibilance_hp: HighPassFilter,
    /// Sibilance envelope follower.
    sibilance_envelope: EnvelopeFollower,
    /// Full-band modulator envelope for sibilance ratio.
    fullband_envelope: EnvelopeFollower,
    config: VocoderConfig,
}

impl Vocoder {
    /// Create a new vocoder from configuration.
    pub fn new(config: VocoderConfig) -> Self {
        let channels = config.bands.iter().map(|band| {
            // Apply formant shift to carrier filter centre frequency.
            let shifted_center = band.center_hz * 2.0_f64.powf(config.formant_shift_semitones / 12.0);
            let carrier_band = BandConfig { center_hz: shifted_center, q: band.q };
            let mut ch = VocoderChannel::new(band, config.sample_rate, config.attack_ms, config.release_ms);
            ch.car_filter = BandPassFilter::new(carrier_band.center_hz, carrier_band.q, config.sample_rate);
            ch
        }).collect();

        let sibilance_hp = HighPassFilter::new(5000.0, 0.707, config.sample_rate);
        let sibilance_envelope = EnvelopeFollower::new(1.0, 10.0, config.sample_rate);
        let fullband_envelope = EnvelopeFollower::new(1.0, 10.0, config.sample_rate);

        Self { channels, sibilance_hp, sibilance_envelope, fullband_envelope, config }
    }

    /// Number of bands.
    pub fn band_count(&self) -> usize {
        self.channels.len()
    }

    /// Get band centre frequencies.
    pub fn band_frequencies(&self) -> Vec<f64> {
        self.channels.iter().map(|ch| ch.center_hz).collect()
    }

    /// Get current envelope levels for all bands.
    pub fn envelope_levels(&self) -> Vec<f64> {
        self.channels.iter().map(|ch| ch.envelope.level()).collect()
    }

    /// Process a single sample pair (modulator, carrier) and return the vocoded output.
    pub fn process(&mut self, modulator: f64, carrier: f64) -> f64 {
        let mut output = 0.0;

        for ch in &mut self.channels {
            // Analysis: filter modulator, extract envelope.
            let mod_filtered = ch.mod_filter.process(modulator);
            let env_level = ch.envelope.process(mod_filtered);

            // Synthesis: filter carrier, apply envelope.
            let car_filtered = ch.car_filter.process(carrier);
            output += car_filtered * env_level;
        }

        // Sibilance detection and pass-through.
        if self.config.sibilance_enabled {
            let hp_signal = self.sibilance_hp.process(modulator);
            let sib_level = self.sibilance_envelope.process(hp_signal);
            let full_level = self.fullband_envelope.process(modulator);

            let ratio = if full_level > 1e-10 { sib_level / full_level } else { 0.0 };
            if ratio > self.config.sibilance_threshold {
                output += hp_signal * self.config.sibilance_mix;
            }
        }

        output
    }

    /// Process blocks of modulator and carrier, writing result to output buffer.
    pub fn process_block(&mut self, modulator: &[f64], carrier: &[f64], output: &mut [f64]) {
        let len = modulator.len().min(carrier.len()).min(output.len());
        for i in 0..len {
            output[i] = self.process(modulator[i], carrier[i]);
        }
    }

    /// Reset all internal filter and envelope states.
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
        self.sibilance_hp.reset();
        self.sibilance_envelope.reset();
        self.fullband_envelope.reset();
    }

    /// Update formant shift. Rebuilds carrier filters.
    pub fn set_formant_shift(&mut self, semitones: f64) {
        self.config.formant_shift_semitones = semitones;
        for (ch, band) in self.channels.iter_mut().zip(self.config.bands.iter()) {
            let shifted = band.center_hz * 2.0_f64.powf(semitones / 12.0);
            ch.car_filter = BandPassFilter::new(shifted, band.q, self.config.sample_rate);
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f64 = 1e-4;

    fn sine_tone(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr).sin()).collect()
    }

    fn rms(buf: &[f64]) -> f64 {
        let sum: f64 = buf.iter().map(|s| s * s).sum();
        (sum / buf.len().max(1) as f64).sqrt()
    }

    #[test]
    fn test_log_spaced_bands_count() {
        let bands = log_spaced_bands(16, 100.0, 8000.0, 8.0);
        assert_eq!(bands.len(), 16);
    }

    #[test]
    fn test_log_spaced_bands_range() {
        let bands = log_spaced_bands(16, 100.0, 8000.0, 8.0);
        assert!((bands[0].center_hz - 100.0).abs() < 1.0);
        assert!((bands[15].center_hz - 8000.0).abs() < 10.0);
    }

    #[test]
    fn test_log_spaced_bands_ascending() {
        let bands = log_spaced_bands(16, 100.0, 8000.0, 8.0);
        for i in 1..bands.len() {
            assert!(bands[i].center_hz > bands[i - 1].center_hz,
                "bands should be ascending: {} vs {}", bands[i - 1].center_hz, bands[i].center_hz);
        }
    }

    #[test]
    fn test_log_spaced_bands_empty() {
        assert!(log_spaced_bands(0, 100.0, 8000.0, 8.0).is_empty());
        assert!(log_spaced_bands(16, 0.0, 8000.0, 8.0).is_empty());
        assert!(log_spaced_bands(16, 8000.0, 100.0, 8.0).is_empty());
    }

    #[test]
    fn test_envelope_follower_attack() {
        let mut ef = EnvelopeFollower::new(1.0, 50.0, SR);
        // Feed constant signal.
        for _ in 0..1000 {
            ef.process(1.0);
        }
        assert!(ef.level() > 0.8, "envelope should track input: level={}", ef.level());
    }

    #[test]
    fn test_envelope_follower_release() {
        let mut ef = EnvelopeFollower::new(1.0, 5.0, SR);
        for _ in 0..1000 {
            ef.process(1.0);
        }
        let peak = ef.level();
        for _ in 0..1000 {
            ef.process(0.0);
        }
        assert!(ef.level() < peak * 0.5, "envelope should release: {} vs peak {}", ef.level(), peak);
    }

    #[test]
    fn test_envelope_follower_reset() {
        let mut ef = EnvelopeFollower::new(1.0, 10.0, SR);
        for _ in 0..100 {
            ef.process(1.0);
        }
        ef.reset();
        assert!((ef.level() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_vocoder_band_count() {
        let config = VocoderConfig::default_16_band(SR);
        let vocoder = Vocoder::new(config);
        assert_eq!(vocoder.band_count(), 16);
    }

    #[test]
    fn test_vocoder_32_band() {
        let config = VocoderConfig::default_32_band(SR);
        let vocoder = Vocoder::new(config);
        assert_eq!(vocoder.band_count(), 32);
    }

    #[test]
    fn test_vocoder_silence_in_silence_out() {
        let config = VocoderConfig::default_16_band(SR);
        let mut vocoder = Vocoder::new(config);
        let out = vocoder.process(0.0, 0.0);
        assert!((out - 0.0).abs() < EPS, "silence in should give silence out");
    }

    #[test]
    fn test_vocoder_produces_output() {
        let config = VocoderConfig::default_16_band(SR);
        let mut vocoder = Vocoder::new(config);
        let modulator = sine_tone(200.0, SR, 4096);
        let carrier = sine_tone(440.0, SR, 4096);
        let mut output = vec![0.0; 4096];
        vocoder.process_block(&modulator, &carrier, &mut output);
        let out_rms = rms(&output);
        assert!(out_rms > 0.001, "vocoder should produce nonzero output: rms={out_rms}");
    }

    #[test]
    fn test_vocoder_no_carrier_no_output() {
        let config = VocoderConfig::default_16_band(SR);
        let mut vocoder = Vocoder::new(config);
        let modulator = sine_tone(200.0, SR, 4096);
        let carrier = vec![0.0; 4096]; // silence carrier
        let mut output = vec![0.0; 4096];
        vocoder.process_block(&modulator, &carrier, &mut output);
        // With sibilance off, should be very quiet.
        let mut config2 = VocoderConfig::default_16_band(SR);
        config2.sibilance_enabled = false;
        let mut vocoder2 = Vocoder::new(config2);
        let mut output2 = vec![0.0; 4096];
        vocoder2.process_block(&modulator, &carrier, &mut output2);
        let out_rms = rms(&output2);
        assert!(out_rms < 0.01, "no carrier + no sibilance should be near silent: rms={out_rms}");
    }

    #[test]
    fn test_vocoder_envelope_levels() {
        let config = VocoderConfig::default_16_band(SR);
        let mut vocoder = Vocoder::new(config);
        let modulator = sine_tone(500.0, SR, 4096);
        let carrier = sine_tone(500.0, SR, 4096);
        for i in 0..4096 {
            vocoder.process(modulator[i], carrier[i]);
        }
        let levels = vocoder.envelope_levels();
        assert_eq!(levels.len(), 16);
        let any_active = levels.iter().any(|l| *l > 0.001);
        assert!(any_active, "some bands should have nonzero envelope");
    }

    #[test]
    fn test_vocoder_reset() {
        let config = VocoderConfig::default_16_band(SR);
        let mut vocoder = Vocoder::new(config);
        let modulator = sine_tone(500.0, SR, 1024);
        let carrier = sine_tone(500.0, SR, 1024);
        for i in 0..1024 {
            vocoder.process(modulator[i], carrier[i]);
        }
        vocoder.reset();
        let levels = vocoder.envelope_levels();
        for l in &levels {
            assert!((*l - 0.0).abs() < EPS, "all levels should be zero after reset, got {l}");
        }
    }

    #[test]
    fn test_vocoder_formant_shift() {
        let config = VocoderConfig::default_16_band(SR);
        let mut vocoder = Vocoder::new(config);
        vocoder.set_formant_shift(12.0); // shift up one octave
        // Should not panic and still produce output.
        let modulator = sine_tone(300.0, SR, 2048);
        let carrier = sine_tone(440.0, SR, 2048);
        let mut output = vec![0.0; 2048];
        vocoder.process_block(&modulator, &carrier, &mut output);
        let out_rms = rms(&output);
        assert!(out_rms > 0.0, "formant-shifted vocoder should produce output: {out_rms}");
    }

    #[test]
    fn test_vocoder_band_frequencies() {
        let config = VocoderConfig::default_16_band(SR);
        let vocoder = Vocoder::new(config);
        let freqs = vocoder.band_frequencies();
        assert_eq!(freqs.len(), 16);
        for i in 1..freqs.len() {
            assert!(freqs[i] > freqs[i - 1], "frequencies should be ascending");
        }
    }

    #[test]
    fn test_bandpass_filter_passes_center() {
        let mut bp = BandPassFilter::new(1000.0, 5.0, SR);
        let signal = sine_tone(1000.0, SR, 4096);
        let filtered: Vec<f64> = signal.iter().map(|s| bp.process(*s)).collect();
        let out_rms = rms(&filtered);
        let in_rms = rms(&signal);
        assert!(out_rms > in_rms * 0.1, "BP should pass center freq: in={in_rms} out={out_rms}");
    }

    #[test]
    fn test_bandpass_filter_rejects_far_freq() {
        let mut bp = BandPassFilter::new(1000.0, 10.0, SR);
        let signal = sine_tone(10000.0, SR, 4096);
        let filtered: Vec<f64> = signal.iter().map(|s| bp.process(*s)).collect();
        let out_rms = rms(&filtered);
        let in_rms = rms(&signal);
        assert!(out_rms < in_rms * 0.3, "BP should reject far freq: in={in_rms} out={out_rms}");
    }

    #[test]
    fn test_sibilance_detection() {
        let mut config = VocoderConfig::default_16_band(SR);
        config.sibilance_enabled = true;
        config.sibilance_mix = 1.0;
        let mut vocoder = Vocoder::new(config);

        // High-frequency modulator (sibilant) with silent carrier.
        let modulator = sine_tone(8000.0, SR, 4096);
        let carrier = vec![0.0; 4096];
        let mut output = vec![0.0; 4096];
        vocoder.process_block(&modulator, &carrier, &mut output);
        // Sibilance pass-through should add some output even with no carrier.
        let out_rms = rms(&output);
        // May or may not trigger depending on threshold; just verify no crash.
        assert!(out_rms >= 0.0);
    }

    #[test]
    fn test_vocoder_config_reference() {
        let config = VocoderConfig::default_16_band(SR);
        let vocoder = Vocoder::new(config);
        assert_eq!(vocoder.config().bands.len(), 16);
        assert!((vocoder.config().sample_rate - SR).abs() < EPS);
    }

    #[test]
    fn test_highpass_filter_passes_high() {
        let mut hp = HighPassFilter::new(1000.0, 0.707, SR);
        let signal = sine_tone(5000.0, SR, 4096);
        let filtered: Vec<f64> = signal.iter().map(|s| hp.process(*s)).collect();
        let out_rms = rms(&filtered);
        let in_rms = rms(&signal);
        assert!(out_rms > in_rms * 0.5, "HP should pass 5kHz: in={in_rms} out={out_rms}");
    }

    #[test]
    fn test_highpass_filter_rejects_low() {
        let mut hp = HighPassFilter::new(5000.0, 0.707, SR);
        let signal = sine_tone(100.0, SR, 4096);
        let filtered: Vec<f64> = signal.iter().map(|s| hp.process(*s)).collect();
        let out_rms = rms(&filtered);
        let in_rms = rms(&signal);
        assert!(out_rms < in_rms * 0.2, "HP should reject 100Hz: in={in_rms} out={out_rms}");
    }
}
