//! Audio filters for synthesis — biquad and state-variable implementations.
//!
//! Provides low-pass, high-pass, band-pass, band-reject (notch), all-pass,
//! peaking EQ, low-shelf, and high-shelf filters using Robert Bristow-Johnson's
//! Audio EQ Cookbook formulas. Supports cascading for higher-order filtering
//! and cutoff sweeps. Pure Rust — no DSP library deps.

use std::f64::consts::PI;

// ── Filter Type ─────────────────────────────────────────────────

/// Supported biquad filter types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    /// Band-reject / notch filter.
    Notch,
    AllPass,
    /// Peaking EQ with gain_db.
    PeakingEQ { gain_db: f64 },
    /// Low shelf with gain_db.
    LowShelf { gain_db: f64 },
    /// High shelf with gain_db.
    HighShelf { gain_db: f64 },
}

// ── Biquad Coefficients ─────────────────────────────────────────

/// Normalised biquad filter coefficients: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Compute coefficients for the given filter type, cutoff frequency, Q, and sample rate.
    /// Based on Robert Bristow-Johnson's Audio EQ Cookbook.
    pub fn compute(filter_type: FilterType, cutoff_hz: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q.max(0.001));

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            FilterType::LowPass => {
                let b1 = 1.0 - cos_w0;
                let b0 = b1 / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighPass => {
                let b1 = -(1.0 + cos_w0);
                let b0 = (1.0 + cos_w0) / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::AllPass => {
                let b0 = 1.0 - alpha;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 + alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::PeakingEQ { gain_db } => {
                let a_val = 10.0_f64.powf(gain_db / 40.0);
                let b0 = 1.0 + alpha * a_val;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a_val;
                let a0 = 1.0 + alpha / a_val;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a_val;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::LowShelf { gain_db } => {
                let a_val = 10.0_f64.powf(gain_db / 40.0);
                let two_sqrt_a_alpha = 2.0 * a_val.sqrt() * alpha;
                let b0 = a_val * ((a_val + 1.0) - (a_val - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a_val * ((a_val - 1.0) - (a_val + 1.0) * cos_w0);
                let b2 = a_val * ((a_val + 1.0) - (a_val - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a_val + 1.0) + (a_val - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a_val - 1.0) + (a_val + 1.0) * cos_w0);
                let a2 = (a_val + 1.0) + (a_val - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighShelf { gain_db } => {
                let a_val = 10.0_f64.powf(gain_db / 40.0);
                let two_sqrt_a_alpha = 2.0 * a_val.sqrt() * alpha;
                let b0 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a_val * ((a_val - 1.0) + (a_val + 1.0) * cos_w0);
                let b2 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a_val + 1.0) - (a_val - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a_val - 1.0) - (a_val + 1.0) * cos_w0);
                let a2 = (a_val + 1.0) - (a_val - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        // Normalise by a0.
        let inv_a0 = 1.0 / a0;
        BiquadCoeffs {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
        }
    }
}

// ── Biquad Filter ───────────────────────────────────────────────

/// A single second-order (biquad) IIR filter.
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    coeffs: BiquadCoeffs,
    /// Delay line: x[n-1], x[n-2], y[n-1], y[n-2].
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    /// Current filter parameters (for recalculation).
    pub filter_type: FilterType,
    pub cutoff_hz: f64,
    pub q: f64,
    pub sample_rate: f64,
}

impl BiquadFilter {
    /// Create a new biquad filter.
    pub fn new(filter_type: FilterType, cutoff_hz: f64, q: f64, sample_rate: f64) -> Self {
        let coeffs = BiquadCoeffs::compute(filter_type, cutoff_hz, q, sample_rate);
        Self {
            coeffs,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            filter_type,
            cutoff_hz,
            q,
            sample_rate,
        }
    }

    /// Recalculate coefficients (call after changing cutoff/Q/type).
    pub fn update_coefficients(&mut self) {
        self.coeffs = BiquadCoeffs::compute(self.filter_type, self.cutoff_hz, self.q, self.sample_rate);
    }

    /// Reset the filter state (delay line).
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Process a single input sample and return the filtered output.
    pub fn process(&mut self, input: f64) -> f64 {
        let c = &self.coeffs;
        let output = c.b0 * input + c.b1 * self.x1 + c.b2 * self.x2
            - c.a1 * self.y1 - c.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    /// Process a block of samples in-place.
    pub fn process_block(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Get a copy of the current coefficients.
    pub fn coefficients(&self) -> BiquadCoeffs {
        self.coeffs
    }
}

// ── Cascaded Filter (higher order) ──────────────────────────────

/// Two biquads in series to form a 4th-order filter.
#[derive(Debug, Clone)]
pub struct CascadeFilter {
    stages: Vec<BiquadFilter>,
}

impl CascadeFilter {
    /// Create a cascade of `order` identical biquad stages (each is 2nd order).
    pub fn new(filter_type: FilterType, cutoff_hz: f64, q: f64, sample_rate: f64, stages: usize) -> Self {
        let stages_vec = (0..stages.max(1))
            .map(|_| BiquadFilter::new(filter_type, cutoff_hz, q, sample_rate))
            .collect();
        Self { stages: stages_vec }
    }

    /// Process a single sample through all stages.
    pub fn process(&mut self, input: f64) -> f64 {
        let mut val = input;
        for stage in &mut self.stages {
            val = stage.process(val);
        }
        val
    }

    /// Process a block of samples.
    pub fn process_block(&mut self, buffer: &mut [f64]) {
        for stage in &mut self.stages {
            stage.process_block(buffer);
        }
    }

    /// Number of cascaded stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Update cutoff for all stages.
    pub fn set_cutoff(&mut self, cutoff_hz: f64) {
        for stage in &mut self.stages {
            stage.cutoff_hz = cutoff_hz;
            stage.update_coefficients();
        }
    }

    /// Update Q for all stages.
    pub fn set_q(&mut self, q: f64) {
        for stage in &mut self.stages {
            stage.q = q;
            stage.update_coefficients();
        }
    }

    /// Reset all stages.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }
}

// ── Filter Sweep ────────────────────────────────────────────────

/// Linearly interpolates cutoff over a block for smooth filter sweeps.
pub fn sweep_cutoff_block(
    filter: &mut BiquadFilter,
    buffer: &mut [f64],
    start_hz: f64,
    end_hz: f64,
) {
    let len = buffer.len();
    if len == 0 {
        return;
    }
    for (i, sample) in buffer.iter_mut().enumerate() {
        let t = i as f64 / len as f64;
        filter.cutoff_hz = start_hz + (end_hz - start_hz) * t;
        filter.update_coefficients();
        *sample = filter.process(*sample);
    }
}

/// Exponential sweep (more musical).
pub fn sweep_cutoff_exponential(
    filter: &mut BiquadFilter,
    buffer: &mut [f64],
    start_hz: f64,
    end_hz: f64,
) {
    let len = buffer.len();
    if len == 0 || start_hz <= 0.0 || end_hz <= 0.0 {
        return;
    }
    let log_start = start_hz.ln();
    let log_end = end_hz.ln();
    for (i, sample) in buffer.iter_mut().enumerate() {
        let t = i as f64 / len as f64;
        filter.cutoff_hz = (log_start + (log_end - log_start) * t).exp();
        filter.update_coefficients();
        *sample = filter.process(*sample);
    }
}

// ── State-Variable Filter ───────────────────────────────────────

/// A state-variable filter (SVF) that simultaneously outputs LP, HP, BP, and Notch.
#[derive(Debug, Clone)]
pub struct StateVariableFilter {
    pub cutoff_hz: f64,
    pub q: f64,
    pub sample_rate: f64,
    // Internal state.
    ic1eq: f64,
    ic2eq: f64,
}

/// Output of the state-variable filter for a single sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvfOutput {
    pub low_pass: f64,
    pub high_pass: f64,
    pub band_pass: f64,
    pub notch: f64,
}

impl StateVariableFilter {
    /// Create a new SVF.
    pub fn new(cutoff_hz: f64, q: f64, sample_rate: f64) -> Self {
        Self { cutoff_hz, q, sample_rate, ic1eq: 0.0, ic2eq: 0.0 }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    /// Process a single sample, returning all four outputs.
    pub fn process(&mut self, input: f64) -> SvfOutput {
        let g = (PI * self.cutoff_hz / self.sample_rate).tan();
        let k = 1.0 / self.q.max(0.001);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let v3 = input - self.ic2eq;
        let v1 = a1 * self.ic1eq + a2 * v3;
        let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;

        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        SvfOutput {
            low_pass: v2,
            band_pass: v1,
            high_pass: input - k * v1 - v2,
            notch: input - k * v1,
        }
    }

    /// Process a block, returning only the low-pass output.
    pub fn process_block_lp(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample).low_pass;
        }
    }

    /// Process a block, returning only the high-pass output.
    pub fn process_block_hp(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample).high_pass;
        }
    }

    /// Process a block, returning only the band-pass output.
    pub fn process_block_bp(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample).band_pass;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f64 = 1e-3;

    /// Generate a sine tone buffer.
    fn sine_tone(freq: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    /// RMS power of a signal.
    fn rms(signal: &[f64]) -> f64 {
        let sum_sq: f64 = signal.iter().map(|s| s * s).sum();
        (sum_sq / signal.len() as f64).sqrt()
    }

    #[test]
    fn test_lowpass_attenuates_high_freq() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 500.0, 0.707, SR);
        // 5kHz tone should be attenuated
        let mut signal = sine_tone(5000.0, SR, 4096);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms < input_rms * 0.2, "LP should attenuate 5kHz: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_lowpass_passes_low_freq() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 5000.0, 0.707, SR);
        // 100Hz tone should pass through mostly
        let mut signal = sine_tone(100.0, SR, 4096);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 0.8, "LP should pass 100Hz: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_highpass_attenuates_low_freq() {
        let mut filter = BiquadFilter::new(FilterType::HighPass, 5000.0, 0.707, SR);
        let mut signal = sine_tone(100.0, SR, 4096);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms < input_rms * 0.2, "HP should attenuate 100Hz: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_highpass_passes_high_freq() {
        let mut filter = BiquadFilter::new(FilterType::HighPass, 500.0, 0.707, SR);
        let mut signal = sine_tone(5000.0, SR, 4096);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 0.7, "HP should pass 5kHz: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_bandpass_center_freq() {
        let mut filter = BiquadFilter::new(FilterType::BandPass, 1000.0, 5.0, SR);
        // 1kHz tone should pass
        let mut signal = sine_tone(1000.0, SR, 4096);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 0.3, "BP should pass 1kHz: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_notch_rejects_center() {
        let mut filter = BiquadFilter::new(FilterType::Notch, 1000.0, 10.0, SR);
        let mut signal = sine_tone(1000.0, SR, 8192);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms < input_rms * 0.2, "Notch should reject 1kHz: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_allpass_preserves_amplitude() {
        let mut filter = BiquadFilter::new(FilterType::AllPass, 1000.0, 0.707, SR);
        let mut signal = sine_tone(1000.0, SR, 8192);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!((output_rms - input_rms).abs() < input_rms * 0.1,
            "AllPass should preserve amplitude: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_peaking_eq_boost() {
        let mut filter = BiquadFilter::new(FilterType::PeakingEQ { gain_db: 12.0 }, 1000.0, 1.0, SR);
        let mut signal = sine_tone(1000.0, SR, 8192);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 1.5, "PeakingEQ +12dB should boost: in={input_rms} out={output_rms}");
    }

    #[test]
    fn test_cascade_steeper_rolloff() {
        // Single stage LP
        let mut single = BiquadFilter::new(FilterType::LowPass, 500.0, 0.707, SR);
        let mut signal1 = sine_tone(2000.0, SR, 4096);
        single.process_block(&mut signal1);
        let rms1 = rms(&signal1);

        // Two-stage cascade LP (4th order)
        let mut cascade = CascadeFilter::new(FilterType::LowPass, 500.0, 0.707, SR, 2);
        let mut signal2 = sine_tone(2000.0, SR, 4096);
        cascade.process_block(&mut signal2);
        let rms2 = rms(&signal2);

        assert!(rms2 < rms1, "cascade should have steeper rolloff: single={rms1} cascade={rms2}");
    }

    #[test]
    fn test_cascade_stage_count() {
        let cascade = CascadeFilter::new(FilterType::LowPass, 1000.0, 0.707, SR, 3);
        assert_eq!(cascade.stage_count(), 3);
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 1000.0, 0.707, SR);
        for _ in 0..100 {
            filter.process(1.0);
        }
        filter.reset();
        // After reset, processing silence should yield ~0.
        let out = filter.process(0.0);
        assert!((out - 0.0).abs() < EPS);
    }

    #[test]
    fn test_sweep_cutoff_linear() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 200.0, 0.707, SR);
        let mut signal = sine_tone(1000.0, SR, 2048);
        sweep_cutoff_block(&mut filter, &mut signal, 200.0, 5000.0);
        // Should not panic and produce output.
        let output_rms = rms(&signal);
        assert!(output_rms > 0.0, "sweep should produce nonzero output");
    }

    #[test]
    fn test_sweep_cutoff_exponential() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 200.0, 0.707, SR);
        let mut signal = sine_tone(1000.0, SR, 2048);
        sweep_cutoff_exponential(&mut filter, &mut signal, 200.0, 5000.0);
        let output_rms = rms(&signal);
        assert!(output_rms > 0.0, "exp sweep should produce nonzero output");
    }

    #[test]
    fn test_svf_lowpass() {
        let mut svf = StateVariableFilter::new(500.0, 0.707, SR);
        let mut signal = sine_tone(5000.0, SR, 4096);
        let input_rms = rms(&signal);
        svf.process_block_lp(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms < input_rms * 0.3, "SVF LP should attenuate 5kHz");
    }

    #[test]
    fn test_svf_highpass() {
        let mut svf = StateVariableFilter::new(5000.0, 0.707, SR);
        let mut signal = sine_tone(100.0, SR, 4096);
        let input_rms = rms(&signal);
        svf.process_block_hp(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms < input_rms * 0.3, "SVF HP should attenuate 100Hz");
    }

    #[test]
    fn test_svf_bandpass() {
        let mut svf = StateVariableFilter::new(1000.0, 5.0, SR);
        let mut signal = sine_tone(1000.0, SR, 4096);
        let input_rms = rms(&signal);
        svf.process_block_bp(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 0.1, "SVF BP should pass center freq");
    }

    #[test]
    fn test_svf_simultaneous_outputs() {
        let mut svf = StateVariableFilter::new(1000.0, 0.707, SR);
        let out = svf.process(1.0);
        // All outputs should be finite.
        assert!(out.low_pass.is_finite());
        assert!(out.high_pass.is_finite());
        assert!(out.band_pass.is_finite());
        assert!(out.notch.is_finite());
    }

    #[test]
    fn test_svf_reset() {
        let mut svf = StateVariableFilter::new(1000.0, 0.707, SR);
        for _ in 0..100 {
            svf.process(1.0);
        }
        svf.reset();
        let out = svf.process(0.0);
        assert!((out.low_pass).abs() < EPS);
    }

    #[test]
    fn test_cascade_set_cutoff() {
        let mut cascade = CascadeFilter::new(FilterType::LowPass, 1000.0, 0.707, SR, 2);
        cascade.set_cutoff(2000.0);
        // Process to verify it doesn't panic.
        let mut signal = sine_tone(500.0, SR, 256);
        cascade.process_block(&mut signal);
        assert!(rms(&signal) > 0.0);
    }

    #[test]
    fn test_lowshelf_boost() {
        let mut filter = BiquadFilter::new(FilterType::LowShelf { gain_db: 12.0 }, 500.0, 0.707, SR);
        let mut signal = sine_tone(100.0, SR, 8192);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 1.2, "LowShelf +12dB should boost low freq");
    }

    #[test]
    fn test_highshelf_boost() {
        let mut filter = BiquadFilter::new(FilterType::HighShelf { gain_db: 12.0 }, 5000.0, 0.707, SR);
        let mut signal = sine_tone(10000.0, SR, 8192);
        let input_rms = rms(&signal);
        filter.process_block(&mut signal);
        let output_rms = rms(&signal);
        assert!(output_rms > input_rms * 1.2, "HighShelf +12dB should boost high freq");
    }

    #[test]
    fn test_update_coefficients() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 1000.0, 0.707, SR);
        let coeffs1 = filter.coefficients();
        filter.cutoff_hz = 5000.0;
        filter.update_coefficients();
        let coeffs2 = filter.coefficients();
        assert!((coeffs1.b0 - coeffs2.b0).abs() > EPS, "coefficients should change");
    }

    #[test]
    fn test_empty_buffer_sweep() {
        let mut filter = BiquadFilter::new(FilterType::LowPass, 1000.0, 0.707, SR);
        let mut empty: Vec<f64> = Vec::new();
        sweep_cutoff_block(&mut filter, &mut empty, 100.0, 5000.0);
        // Should not panic.
    }
}
