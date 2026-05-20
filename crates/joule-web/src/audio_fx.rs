//! Audio effects processing — filters, delay, reverb, distortion, compressor.
//!
//! All effects implement the `AudioProcessor` trait. They can be chained
//! together with `EffectChain`. Fully headless and testable on native targets.

// ── AudioProcessor trait ────────────────────────────────────────

/// Trait for audio effect processors.
pub trait AudioProcessor: Send {
    /// Process samples in place.
    fn process(&mut self, samples: &mut [f32]);

    /// Reset internal state.
    fn reset(&mut self);

    /// Effect name for debugging.
    fn name(&self) -> &str;
}

// ── Low-Pass Filter (simple RC) ─────────────────────────────────

/// First-order low-pass filter (RC model).
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    cutoff: f32,
    sample_rate: f32,
    alpha: f32,
    prev: f32,
}

impl LowPassFilter {
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        let alpha = Self::compute_alpha(cutoff, sample_rate);
        Self {
            cutoff,
            sample_rate,
            alpha,
            prev: 0.0,
        }
    }

    fn compute_alpha(cutoff: f32, sample_rate: f32) -> f32 {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        dt / (rc + dt)
    }

    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff;
        self.alpha = Self::compute_alpha(cutoff, self.sample_rate);
    }
}

impl AudioProcessor for LowPassFilter {
    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            self.prev += self.alpha * (*s - self.prev);
            *s = self.prev;
        }
    }

    fn reset(&mut self) {
        self.prev = 0.0;
    }

    fn name(&self) -> &str {
        "LowPassFilter"
    }
}

// ── High-Pass Filter ────────────────────────────────────────────

/// First-order high-pass filter.
#[derive(Debug, Clone)]
pub struct HighPassFilter {
    cutoff: f32,
    sample_rate: f32,
    alpha: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighPassFilter {
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        let alpha = Self::compute_alpha(cutoff, sample_rate);
        Self {
            cutoff,
            sample_rate,
            alpha,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    fn compute_alpha(cutoff: f32, sample_rate: f32) -> f32 {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        rc / (rc + dt)
    }

    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff;
        self.alpha = Self::compute_alpha(cutoff, self.sample_rate);
    }
}

impl AudioProcessor for HighPassFilter {
    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let input = *s;
            self.prev_output = self.alpha * (self.prev_output + input - self.prev_input);
            self.prev_input = input;
            *s = self.prev_output;
        }
    }

    fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }

    fn name(&self) -> &str {
        "HighPassFilter"
    }
}

// ── Delay ───────────────────────────────────────────────────────

/// Delay effect with circular buffer and feedback.
#[derive(Debug, Clone)]
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    mix: f32,
}

impl Delay {
    /// Create a delay with the given delay time in seconds.
    pub fn new(delay_seconds: f32, sample_rate: f32, feedback: f32, mix: f32) -> Self {
        let size = (delay_seconds * sample_rate) as usize;
        let size = size.max(1);
        Self {
            buffer: vec![0.0; size],
            write_pos: 0,
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
        }
    }

    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }
}

impl AudioProcessor for Delay {
    fn process(&mut self, samples: &mut [f32]) {
        let len = self.buffer.len();
        for s in samples.iter_mut() {
            let delayed = self.buffer[self.write_pos];
            let input = *s;
            self.buffer[self.write_pos] = input + delayed * self.feedback;
            self.write_pos = (self.write_pos + 1) % len;
            *s = input * (1.0 - self.mix) + delayed * self.mix;
        }
    }

    fn reset(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_pos = 0;
    }

    fn name(&self) -> &str {
        "Delay"
    }
}

// ── Comb Filter (internal for reverb) ───────────────────────────

#[derive(Debug, Clone)]
struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
        }
    }

    fn process_sample(&mut self, input: f32) -> f32 {
        let len = self.buffer.len();
        let output = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = input + output * self.feedback;
        self.write_pos = (self.write_pos + 1) % len;
        output
    }

    fn reset(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_pos = 0;
    }
}

// ── Allpass Filter (internal for reverb) ────────────────────────

#[derive(Debug, Clone)]
struct AllpassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
        }
    }

    fn process_sample(&mut self, input: f32) -> f32 {
        let len = self.buffer.len();
        let buffered = self.buffer[self.write_pos];
        let output = -input + buffered;
        self.buffer[self.write_pos] = input + buffered * self.feedback;
        self.write_pos = (self.write_pos + 1) % len;
        output
    }

    fn reset(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.write_pos = 0;
    }
}

// ── Reverb (Schroeder) ─────────────────────────────────────────

/// Schroeder reverb: 4 parallel comb filters into 2 series allpass filters.
#[derive(Debug, Clone)]
pub struct Reverb {
    combs: [CombFilter; 4],
    allpasses: [AllpassFilter; 2],
    mix: f32,
}

impl Reverb {
    pub fn new(sample_rate: f32, mix: f32) -> Self {
        // Classic Schroeder delay times (in samples at the given sample rate)
        let scale = sample_rate / 44100.0;
        let comb_delays = [
            (1557.0 * scale) as usize,
            (1617.0 * scale) as usize,
            (1491.0 * scale) as usize,
            (1422.0 * scale) as usize,
        ];
        let allpass_delays = [
            (225.0 * scale) as usize,
            (556.0 * scale) as usize,
        ];

        Self {
            combs: [
                CombFilter::new(comb_delays[0], 0.805),
                CombFilter::new(comb_delays[1], 0.827),
                CombFilter::new(comb_delays[2], 0.783),
                CombFilter::new(comb_delays[3], 0.764),
            ],
            allpasses: [
                AllpassFilter::new(allpass_delays[0], 0.7),
                AllpassFilter::new(allpass_delays[1], 0.7),
            ],
            mix: mix.clamp(0.0, 1.0),
        }
    }
}

impl AudioProcessor for Reverb {
    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let input = *s;

            // Sum of parallel comb filters
            let mut comb_out = 0.0f32;
            for comb in &mut self.combs {
                comb_out += comb.process_sample(input);
            }
            comb_out *= 0.25;

            // Series allpass filters
            let mut ap_out = comb_out;
            for ap in &mut self.allpasses {
                ap_out = ap.process_sample(ap_out);
            }

            *s = input * (1.0 - self.mix) + ap_out * self.mix;
        }
    }

    fn reset(&mut self) {
        for c in &mut self.combs {
            c.reset();
        }
        for a in &mut self.allpasses {
            a.reset();
        }
    }

    fn name(&self) -> &str {
        "Reverb"
    }
}

// ── Distortion ──────────────────────────────────────────────────

/// Distortion via tanh waveshaping.
#[derive(Debug, Clone)]
pub struct Distortion {
    /// Drive amount (1.0 = clean, higher = more distortion).
    pub drive: f32,
    /// Output level compensation.
    pub output_gain: f32,
}

impl Distortion {
    pub fn new(drive: f32) -> Self {
        Self {
            drive: drive.max(1.0),
            output_gain: 1.0,
        }
    }
}

impl AudioProcessor for Distortion {
    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = (*s * self.drive).tanh() * self.output_gain;
        }
    }

    fn reset(&mut self) {
        // Stateless effect
    }

    fn name(&self) -> &str {
        "Distortion"
    }
}

// ── Compressor ──────────────────────────────────────────────────

/// Dynamic range compressor.
#[derive(Debug, Clone)]
pub struct Compressor {
    pub threshold: f32,
    pub ratio: f32,
    pub attack: f32,
    pub release: f32,
    sample_rate: f32,
    envelope: f32,
}

impl Compressor {
    pub fn new(threshold: f32, ratio: f32, attack: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            threshold,
            ratio: ratio.max(1.0),
            attack,
            release,
            sample_rate,
            envelope: 0.0,
        }
    }
}

impl AudioProcessor for Compressor {
    fn process(&mut self, samples: &mut [f32]) {
        let attack_coeff = (-1.0 / (self.attack * self.sample_rate)).exp();
        let release_coeff = (-1.0 / (self.release * self.sample_rate)).exp();

        for s in samples.iter_mut() {
            let abs_val = s.abs();

            // Envelope follower
            let coeff = if abs_val > self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * abs_val;

            // Gain computation
            if self.envelope > self.threshold {
                let db_over = 20.0 * (self.envelope / self.threshold).log10();
                let db_reduction = db_over * (1.0 - 1.0 / self.ratio);
                let gain = 10.0f32.powf(-db_reduction / 20.0);
                *s *= gain;
            }
        }
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
    }

    fn name(&self) -> &str {
        "Compressor"
    }
}

// ── Effect Chain ────────────────────────────────────────────────

/// Chain multiple effects in series.
pub struct EffectChain {
    effects: Vec<Box<dyn AudioProcessor>>,
}

impl EffectChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    pub fn add(&mut self, effect: Box<dyn AudioProcessor>) {
        self.effects.push(effect);
    }

    pub fn len(&self) -> usize {
        self.effects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for EffectChain {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioProcessor for EffectChain {
    fn process(&mut self, samples: &mut [f32]) {
        for fx in &mut self.effects {
            fx.process(samples);
        }
    }

    fn reset(&mut self) {
        for fx in &mut self.effects {
            fx.reset();
        }
    }

    fn name(&self) -> &str {
        "EffectChain"
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_attenuates_high_freq() {
        let mut lpf = LowPassFilter::new(100.0, 44100.0);
        // Generate a high-frequency signal (10kHz)
        let mut samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 10000.0 * i as f32 / 44100.0).sin())
            .collect();
        let input_energy: f32 = samples.iter().map(|s| s * s).sum();
        lpf.process(&mut samples);
        let output_energy: f32 = samples.iter().map(|s| s * s).sum();
        assert!(output_energy < input_energy * 0.5, "LPF should attenuate high frequencies");
    }

    #[test]
    fn lowpass_passes_dc() {
        let mut lpf = LowPassFilter::new(1000.0, 44100.0);
        let mut samples = vec![1.0f32; 1000];
        lpf.process(&mut samples);
        // After settling, output should be close to 1.0
        assert!((samples[999] - 1.0).abs() < 0.01);
    }

    #[test]
    fn highpass_removes_dc() {
        let mut hpf = HighPassFilter::new(100.0, 44100.0);
        let mut samples = vec![1.0f32; 2000];
        hpf.process(&mut samples);
        // DC should be attenuated
        assert!(samples[1999].abs() < 0.1);
    }

    #[test]
    fn delay_echoes() {
        let sr = 44100.0;
        let delay_time = 0.01; // 10ms = 441 samples
        let mut delay = Delay::new(delay_time, sr, 0.0, 1.0);
        let mut samples = vec![0.0f32; 1000];
        samples[0] = 1.0; // impulse
        delay.process(&mut samples);
        // The impulse should appear at ~441 samples later
        let delay_samples = (delay_time * sr) as usize;
        assert!(samples[delay_samples].abs() > 0.5);
        // Original position should be near-zero (mix=1.0 means all wet)
        assert!(samples[0].abs() < 0.01);
    }

    #[test]
    fn delay_feedback() {
        let sr = 44100.0;
        let mut delay = Delay::new(0.01, sr, 0.5, 1.0);
        let mut samples = vec![0.0f32; 2000];
        samples[0] = 1.0;
        delay.process(&mut samples);
        let delay_samples = (0.01 * sr) as usize;
        // First echo
        assert!(samples[delay_samples].abs() > 0.5);
        // Second echo (attenuated by feedback)
        assert!(samples[delay_samples * 2].abs() > 0.2);
    }

    #[test]
    fn reverb_adds_tail() {
        let mut reverb = Reverb::new(44100.0, 0.5);
        let mut samples = vec![0.0f32; 4000];
        samples[0] = 1.0;
        reverb.process(&mut samples);
        // Reverb tail should have energy beyond the initial impulse
        let tail_energy: f32 = samples[100..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 0.001);
    }

    #[test]
    fn distortion_clips() {
        let mut dist = Distortion::new(10.0);
        let mut samples = vec![0.0, 0.5, 1.0, -1.0, 2.0];
        dist.process(&mut samples);
        // tanh saturates, so everything should be in [-1, 1]
        for s in &samples {
            assert!(s.abs() <= 1.0);
        }
        // High drive on 2.0 should be close to 1.0
        assert!(samples[4] > 0.99);
    }

    #[test]
    fn distortion_preserves_silence() {
        let mut dist = Distortion::new(5.0);
        let mut samples = vec![0.0; 10];
        dist.process(&mut samples);
        for s in &samples {
            assert_eq!(*s, 0.0);
        }
    }

    #[test]
    fn compressor_reduces_loud() {
        let mut comp = Compressor::new(0.1, 4.0, 0.001, 0.01, 44100.0);
        let mut samples: Vec<f32> = (0..2000)
            .map(|i| 0.8 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let input_peak: f32 = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        comp.process(&mut samples);
        let output_peak: f32 = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(output_peak < input_peak, "Compressor should reduce peaks");
    }

    #[test]
    fn effect_chain_processes_in_order() {
        let mut chain = EffectChain::new();
        chain.add(Box::new(Distortion::new(5.0)));
        chain.add(Box::new(LowPassFilter::new(5000.0, 44100.0)));
        assert_eq!(chain.len(), 2);

        let mut samples: Vec<f32> = (0..500)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        chain.process(&mut samples);
        // Should produce non-zero output
        let energy: f32 = samples.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn lowpass_set_cutoff() {
        let mut lpf = LowPassFilter::new(1000.0, 44100.0);
        lpf.set_cutoff(500.0);
        let mut samples = vec![1.0f32; 100];
        lpf.process(&mut samples);
        // Just verify it doesn't panic and still works
        assert!(samples[99] > 0.0);
    }

    #[test]
    fn reverb_reset() {
        let mut reverb = Reverb::new(44100.0, 1.0);
        let mut samples = vec![1.0f32; 100];
        reverb.process(&mut samples);
        reverb.reset();
        // After reset, processing silence should give silence
        let mut silence = vec![0.0f32; 100];
        reverb.process(&mut silence);
        let energy: f32 = silence.iter().map(|s| s * s).sum();
        assert!(energy < 0.001);
    }
}
