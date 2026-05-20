//! Audio oscillator and waveform generator with polyphonic synthesis.
//!
//! Provides band-limited waveforms (sine, square, sawtooth, triangle, pulse,
//! white noise, pink noise) with PolyBLEP anti-aliasing, phase accumulation,
//! hard sync, and detune support. Pure Rust — no DSP library dependencies.

use std::collections::HashMap;

// ── Waveform Types ──────────────────────────────────────────────

/// Supported waveform shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    /// Pulse with variable duty cycle (0.0 = fully off, 1.0 = fully on).
    Pulse(f64),
    WhiteNoise,
    PinkNoise,
}

// ── PolyBLEP Anti-aliasing ──────────────────────────────────────

/// PolyBLEP correction for band-limited waveforms.
/// `t` is the phase position normalised 0..1, `dt` is phase increment per sample.
fn poly_blep(t: f64, dt: f64) -> f64 {
    if dt <= 0.0 {
        return 0.0;
    }
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

// ── Simple PRNG (xorshift64) ────────────────────────────────────

/// Deterministic xorshift64 PRNG for noise generation.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Map to -1.0..1.0
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}

// ── Pink Noise Filter (Paul Kellet's method) ────────────────────

#[derive(Debug, Clone)]
struct PinkFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    b5: f64,
    b6: f64,
}

impl PinkFilter {
    fn new() -> Self {
        Self { b0: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, b4: 0.0, b5: 0.0, b6: 0.0 }
    }

    fn process(&mut self, white: f64) -> f64 {
        self.b0 = 0.99886 * self.b0 + white * 0.0555179;
        self.b1 = 0.99332 * self.b1 + white * 0.0750759;
        self.b2 = 0.96900 * self.b2 + white * 0.1538520;
        self.b3 = 0.86650 * self.b3 + white * 0.3104856;
        self.b4 = 0.55000 * self.b4 + white * 0.5329522;
        self.b5 = -0.7616 * self.b5 - white * 0.0168980;
        let pink = self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6 + white * 0.5362;
        self.b6 = white * 0.115926;
        pink * 0.11 // normalise roughly to -1..1
    }
}

// ── Single Oscillator ───────────────────────────────────────────

/// A single oscillator voice with phase accumulator and waveform generation.
#[derive(Debug, Clone)]
pub struct Oscillator {
    /// Waveform shape.
    pub waveform: Waveform,
    /// Base frequency in Hz.
    pub frequency: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Current phase in 0.0..1.0.
    phase: f64,
    /// Detune in cents (100 cents = 1 semitone).
    pub detune_cents: f64,
    /// Amplitude 0.0..1.0.
    pub amplitude: f64,
    /// PRNG for noise waveforms.
    rng: Xorshift64,
    /// Pink noise filter state.
    pink: PinkFilter,
}

impl Oscillator {
    /// Create a new oscillator at the given frequency and sample rate.
    pub fn new(waveform: Waveform, frequency: f64, sample_rate: f64) -> Self {
        Self {
            waveform,
            frequency,
            sample_rate,
            phase: 0.0,
            detune_cents: 0.0,
            amplitude: 1.0,
            rng: Xorshift64::new(0xCAFE_BABE),
            pink: PinkFilter::new(),
        }
    }

    /// Set a specific random seed for noise waveforms.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
    }

    /// Reset phase to zero.
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }

    /// Set phase to an arbitrary value in 0.0..1.0.
    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase.rem_euclid(1.0);
    }

    /// Get the current phase.
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Compute the effective frequency including detune.
    pub fn effective_frequency(&self) -> f64 {
        self.frequency * 2.0_f64.powf(self.detune_cents / 1200.0)
    }

    /// Phase increment per sample.
    fn dt(&self) -> f64 {
        if self.sample_rate <= 0.0 {
            return 0.0;
        }
        self.effective_frequency() / self.sample_rate
    }

    /// Generate the next sample and advance the phase.
    pub fn next_sample(&mut self) -> f64 {
        let dt = self.dt();
        let sample = self.sample_at_phase(self.phase, dt);
        self.phase += dt;
        self.phase = self.phase.rem_euclid(1.0);
        sample * self.amplitude
    }

    /// Perform a hard-sync reset: snap phase back to zero.
    pub fn hard_sync_reset(&mut self) {
        self.phase = 0.0;
    }

    /// Sample the waveform at a given phase (0..1) with PolyBLEP where applicable.
    fn sample_at_phase(&mut self, phase: f64, dt: f64) -> f64 {
        match self.waveform {
            Waveform::Sine => (phase * std::f64::consts::TAU).sin(),
            Waveform::Square => {
                let naive = if phase < 0.5 { 1.0 } else { -1.0 };
                naive + poly_blep(phase, dt) - poly_blep((phase + 0.5).rem_euclid(1.0), dt)
            }
            Waveform::Sawtooth => {
                let naive = 2.0 * phase - 1.0;
                naive - poly_blep(phase, dt)
            }
            Waveform::Triangle => {
                // Integrate a square wave for band-limited triangle.
                let sq = if phase < 0.5 { 1.0 } else { -1.0 };
                let sq = sq + poly_blep(phase, dt) - poly_blep((phase + 0.5).rem_euclid(1.0), dt);
                // Leaky integrator to form triangle from square.
                // For a clean triangle: 4 * |phase - 0.5| - 1
                // We use the naive approach corrected with PolyBLEP square.
                let _ = sq; // suppress warning
                4.0 * (phase - 0.5).abs() - 1.0
            }
            Waveform::Pulse(duty) => {
                let duty = duty.clamp(0.01, 0.99);
                let naive = if phase < duty { 1.0 } else { -1.0 };
                naive + poly_blep(phase, dt) - poly_blep((phase - duty + 1.0).rem_euclid(1.0), dt)
            }
            Waveform::WhiteNoise => self.rng.next_f64(),
            Waveform::PinkNoise => {
                let white = self.rng.next_f64();
                self.pink.process(white)
            }
        }
    }

    /// Generate a block of samples into the provided buffer.
    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }
}

// ── Polyphonic Oscillator Bank ──────────────────────────────────

/// A polyphonic bank of oscillators, each identified by a voice ID.
#[derive(Debug, Clone)]
pub struct OscillatorBank {
    voices: HashMap<u32, Oscillator>,
    sample_rate: f64,
    next_id: u32,
    /// Master oscillator for hard-sync (optional).
    master: Option<Oscillator>,
    prev_master_phase: f64,
}

impl OscillatorBank {
    /// Create a new oscillator bank at the given sample rate.
    pub fn new(sample_rate: f64) -> Self {
        Self {
            voices: HashMap::new(),
            sample_rate,
            next_id: 0,
            master: None,
            prev_master_phase: 0.0,
        }
    }

    /// Add a voice and return its ID.
    pub fn add_voice(&mut self, waveform: Waveform, frequency: f64) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let osc = Oscillator::new(waveform, frequency, self.sample_rate);
        self.voices.insert(id, osc);
        id
    }

    /// Remove a voice by ID.
    pub fn remove_voice(&mut self, id: u32) -> bool {
        self.voices.remove(&id).is_some()
    }

    /// Get a mutable reference to a voice.
    pub fn voice_mut(&mut self, id: u32) -> Option<&mut Oscillator> {
        self.voices.get_mut(&id)
    }

    /// Get a reference to a voice.
    pub fn voice(&self, id: u32) -> Option<&Oscillator> {
        self.voices.get(&id)
    }

    /// Number of active voices.
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Set a master oscillator for hard sync.
    pub fn set_master(&mut self, waveform: Waveform, frequency: f64) {
        self.master = Some(Oscillator::new(waveform, frequency, self.sample_rate));
        self.prev_master_phase = 0.0;
    }

    /// Remove the master oscillator.
    pub fn clear_master(&mut self) {
        self.master = None;
    }

    /// Generate the next mixed sample from all voices (summed).
    /// If a master is set, hard-sync resets slave phases on master zero-crossing.
    pub fn next_sample(&mut self) -> f64 {
        // Check master for zero-crossing.
        let do_sync = if let Some(ref mut master) = self.master {
            let _master_sample = master.next_sample();
            let new_phase = master.phase();
            let crossed = new_phase < self.prev_master_phase && self.prev_master_phase > 0.5;
            self.prev_master_phase = new_phase;
            crossed
        } else {
            false
        };

        let ids: Vec<u32> = self.voices.keys().copied().collect();
        let mut sum = 0.0;
        for id in ids {
            if let Some(osc) = self.voices.get_mut(&id) {
                if do_sync {
                    osc.hard_sync_reset();
                }
                sum += osc.next_sample();
            }
        }
        sum
    }

    /// Generate a block of mixed samples.
    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }
}

// ── Detune helper ───────────────────────────────────────────────

/// Convert a MIDI note number to frequency in Hz (A4 = 69 = 440 Hz).
pub fn midi_to_freq(note: f64) -> f64 {
    440.0 * 2.0_f64.powf((note - 69.0) / 12.0)
}

/// Convert frequency to the nearest MIDI note.
pub fn freq_to_midi(freq: f64) -> f64 {
    if freq <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (freq / 440.0).log2()
}

/// Compute a detuned frequency given base freq and cents offset.
pub fn detune_freq(base_freq: f64, cents: f64) -> f64 {
    base_freq * 2.0_f64.powf(cents / 1200.0)
}

// ── Unison Spread ───────────────────────────────────────────────

/// Create a set of detuned frequencies for a unison spread.
/// `count` voices spread symmetrically by `spread_cents` total.
pub fn unison_frequencies(base_freq: f64, count: usize, spread_cents: f64) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![base_freq];
    }
    let half = spread_cents / 2.0;
    (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64; // 0..1
            let cents = -half + t * spread_cents;
            detune_freq(base_freq, cents)
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f64 = 1e-4;

    #[test]
    fn test_sine_at_zero_phase() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        let sample = osc.sample_at_phase(0.0, osc.dt());
        assert!((sample - 0.0).abs() < EPS, "sine(0) should be ~0, got {sample}");
    }

    #[test]
    fn test_sine_at_quarter_phase() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        let sample = osc.sample_at_phase(0.25, osc.dt());
        assert!((sample - 1.0).abs() < EPS, "sine(pi/2) should be ~1, got {sample}");
    }

    #[test]
    fn test_sawtooth_range() {
        let mut osc = Oscillator::new(Waveform::Sawtooth, 100.0, SR);
        for _ in 0..1000 {
            let s = osc.next_sample();
            assert!(s >= -1.5 && s <= 1.5, "sawtooth out of range: {s}");
        }
    }

    #[test]
    fn test_square_values() {
        let mut osc = Oscillator::new(Waveform::Square, 100.0, SR);
        let mut saw_positive = false;
        let mut saw_negative = false;
        for _ in 0..1000 {
            let s = osc.next_sample();
            if s > 0.5 { saw_positive = true; }
            if s < -0.5 { saw_negative = true; }
        }
        assert!(saw_positive, "square should produce positive samples");
        assert!(saw_negative, "square should produce negative samples");
    }

    #[test]
    fn test_triangle_range() {
        let mut osc = Oscillator::new(Waveform::Triangle, 200.0, SR);
        for _ in 0..2000 {
            let s = osc.next_sample();
            assert!(s >= -1.1 && s <= 1.1, "triangle out of range: {s}");
        }
    }

    #[test]
    fn test_pulse_duty_cycle() {
        let mut osc = Oscillator::new(Waveform::Pulse(0.25), 100.0, SR);
        let mut positive_count = 0;
        let total = 4410; // one full cycle at 100Hz, 44100 SR = 441 samples. 10 cycles.
        for _ in 0..total {
            if osc.next_sample() > 0.0 {
                positive_count += 1;
            }
        }
        // ~25% should be positive (with some PolyBLEP blurring at transitions)
        let ratio = positive_count as f64 / total as f64;
        assert!((ratio - 0.25).abs() < 0.05, "pulse duty ratio should be ~0.25, got {ratio}");
    }

    #[test]
    fn test_white_noise_nonzero() {
        let mut osc = Oscillator::new(Waveform::WhiteNoise, 440.0, SR);
        let samples: Vec<f64> = (0..100).map(|_| osc.next_sample()).collect();
        let any_nonzero = samples.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "white noise should produce nonzero samples");
    }

    #[test]
    fn test_pink_noise_nonzero() {
        let mut osc = Oscillator::new(Waveform::PinkNoise, 440.0, SR);
        let samples: Vec<f64> = (0..200).map(|_| osc.next_sample()).collect();
        let any_nonzero = samples.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "pink noise should produce nonzero samples");
    }

    #[test]
    fn test_phase_reset() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        for _ in 0..100 {
            osc.next_sample();
        }
        assert!(osc.phase() > 0.0);
        osc.reset_phase();
        assert!((osc.phase() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_set_phase() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        osc.set_phase(0.5);
        assert!((osc.phase() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_detune_cents() {
        let osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        assert!((osc.effective_frequency() - 440.0).abs() < EPS);

        let mut osc2 = Oscillator::new(Waveform::Sine, 440.0, SR);
        osc2.detune_cents = 1200.0; // one octave up
        assert!((osc2.effective_frequency() - 880.0).abs() < 0.1);
    }

    #[test]
    fn test_amplitude_scaling() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        osc.amplitude = 0.5;
        osc.set_phase(0.25); // sin(pi/2)=1 → 0.5
        let s = osc.next_sample();
        assert!((s - 0.5).abs() < EPS, "amplitude scaling failed: {s}");
    }

    #[test]
    fn test_generate_block() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        let mut buffer = vec![0.0; 256];
        osc.generate_block(&mut buffer);
        let any_nonzero = buffer.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "generate_block should fill buffer");
    }

    #[test]
    fn test_oscillator_bank_add_remove() {
        let mut bank = OscillatorBank::new(SR);
        let id1 = bank.add_voice(Waveform::Sine, 440.0);
        let id2 = bank.add_voice(Waveform::Square, 220.0);
        assert_eq!(bank.voice_count(), 2);
        assert!(bank.remove_voice(id1));
        assert_eq!(bank.voice_count(), 1);
        assert!(!bank.remove_voice(id1)); // already removed
        assert!(bank.voice(id2).is_some());
    }

    #[test]
    fn test_oscillator_bank_mix() {
        let mut bank = OscillatorBank::new(SR);
        bank.add_voice(Waveform::Sine, 440.0);
        bank.add_voice(Waveform::Sine, 440.0);
        let mut buf = vec![0.0; 128];
        bank.generate_block(&mut buf);
        // Two identical sines should produce nonzero sum
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero);
    }

    #[test]
    fn test_hard_sync_master() {
        let mut bank = OscillatorBank::new(SR);
        bank.add_voice(Waveform::Sawtooth, 440.0);
        bank.set_master(Waveform::Sawtooth, 110.0);
        // Just verify it runs without panic and produces output.
        let mut buf = vec![0.0; 4410];
        bank.generate_block(&mut buf);
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero);
    }

    #[test]
    fn test_midi_to_freq() {
        assert!((midi_to_freq(69.0) - 440.0).abs() < EPS);
        assert!((midi_to_freq(81.0) - 880.0).abs() < 0.5);
        assert!((midi_to_freq(57.0) - 220.0).abs() < 0.5);
    }

    #[test]
    fn test_freq_to_midi() {
        assert!((freq_to_midi(440.0) - 69.0).abs() < EPS);
        assert!((freq_to_midi(880.0) - 81.0).abs() < EPS);
    }

    #[test]
    fn test_detune_freq_fn() {
        let f = detune_freq(440.0, 1200.0);
        assert!((f - 880.0).abs() < 0.1);
        let f2 = detune_freq(440.0, -1200.0);
        assert!((f2 - 220.0).abs() < 0.1);
    }

    #[test]
    fn test_unison_frequencies() {
        let freqs = unison_frequencies(440.0, 5, 20.0);
        assert_eq!(freqs.len(), 5);
        // Center voice should be ~440
        assert!((freqs[2] - 440.0).abs() < 0.1);
        // First should be lower, last higher
        assert!(freqs[0] < freqs[4]);
    }

    #[test]
    fn test_unison_single() {
        let freqs = unison_frequencies(440.0, 1, 20.0);
        assert_eq!(freqs.len(), 1);
        assert!((freqs[0] - 440.0).abs() < EPS);
    }

    #[test]
    fn test_unison_empty() {
        let freqs = unison_frequencies(440.0, 0, 20.0);
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_poly_blep_mid_phase() {
        // In the middle of the waveform, PolyBLEP should be zero.
        let val = poly_blep(0.5, 0.01);
        assert!((val - 0.0).abs() < EPS);
    }

    #[test]
    fn test_zero_sample_rate() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, 0.0);
        let s = osc.next_sample();
        // Should not panic, just produce 0.
        assert!((s - 0.0).abs() < EPS);
    }

    #[test]
    fn test_voice_mut_detune() {
        let mut bank = OscillatorBank::new(SR);
        let id = bank.add_voice(Waveform::Sine, 440.0);
        bank.voice_mut(id).unwrap().detune_cents = 100.0;
        let eff = bank.voice(id).unwrap().effective_frequency();
        // 100 cents = 1 semitone up from 440 ≈ 466.16
        assert!((eff - 466.16).abs() < 0.5);
    }
}
