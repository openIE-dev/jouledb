//! Software synthesizer — oscillators, ADSR envelopes, polyphonic voice allocator.
//!
//! Generates audio waveforms (sine, square, sawtooth, triangle) at arbitrary
//! frequencies and sample rates. Supports ADSR envelopes and LFO modulation.
//! Voice allocator manages polyphony with oldest-voice stealing.

use std::f64::consts::PI;

// ── Waveform types ──────────────────────────────────────────────

/// Oscillator waveform shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

/// Generate `num_samples` of the given waveform at `frequency` Hz and `sample_rate`.
/// Phase is in range [0, 1) and is updated in place.
pub fn generate_waveform(
    waveform: Waveform,
    frequency: f64,
    sample_rate: f64,
    num_samples: usize,
    phase: &mut f64,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_samples);
    let phase_inc = frequency / sample_rate;

    for _ in 0..num_samples {
        let sample = match waveform {
            Waveform::Sine => (2.0 * PI * *phase).sin(),
            Waveform::Square => {
                if *phase < 0.5 { 1.0 } else { -1.0 }
            }
            Waveform::Sawtooth => 2.0 * *phase - 1.0,
            Waveform::Triangle => {
                if *phase < 0.5 {
                    4.0 * *phase - 1.0
                } else {
                    3.0 - 4.0 * *phase
                }
            }
        };
        out.push(sample as f32);
        *phase += phase_inc;
        *phase -= (*phase).floor();
    }

    out
}

// ── Oscillator ──────────────────────────────────────────────────

/// A stateful oscillator with phase tracking.
#[derive(Debug, Clone)]
pub struct Oscillator {
    pub waveform: Waveform,
    pub frequency: f64,
    pub sample_rate: f64,
    phase: f64,
}

impl Oscillator {
    pub fn new(waveform: Waveform, frequency: f64, sample_rate: f64) -> Self {
        Self {
            waveform,
            frequency,
            sample_rate,
            phase: 0.0,
        }
    }

    /// Generate the next `num_samples` samples.
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        generate_waveform(
            self.waveform,
            self.frequency,
            self.sample_rate,
            num_samples,
            &mut self.phase,
        )
    }

    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }
}

// ── ADSR Envelope ───────────────────────────────────────────────

/// ADSR envelope generator.
#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    /// Attack time in seconds.
    pub attack: f64,
    /// Decay time in seconds.
    pub decay: f64,
    /// Sustain level (0.0 to 1.0).
    pub sustain_level: f64,
    /// Release time in seconds.
    pub release: f64,
}

/// Current stage of the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeStage {
    Attack,
    Decay,
    Sustain,
    Release,
    Off,
}

/// Envelope state tracker.
#[derive(Debug, Clone)]
pub struct EnvelopeState {
    stage: EnvelopeStage,
    time_in_stage: f64,
    current_level: f64,
    release_start_level: f64,
}

impl EnvelopeState {
    pub fn new() -> Self {
        Self {
            stage: EnvelopeStage::Off,
            time_in_stage: 0.0,
            current_level: 0.0,
            release_start_level: 0.0,
        }
    }

    pub fn stage(&self) -> EnvelopeStage {
        self.stage
    }

    pub fn current_level(&self) -> f64 {
        self.current_level
    }

    /// Trigger note on.
    pub fn note_on(&mut self) {
        self.stage = EnvelopeStage::Attack;
        self.time_in_stage = 0.0;
    }

    /// Trigger note off (begin release).
    pub fn note_off(&mut self) {
        self.release_start_level = self.current_level;
        self.stage = EnvelopeStage::Release;
        self.time_in_stage = 0.0;
    }

    /// Advance the envelope by one sample and return the amplitude.
    pub fn tick(&mut self, envelope: &AdsrEnvelope, sample_rate: f64) -> f64 {
        let dt = 1.0 / sample_rate;

        match self.stage {
            EnvelopeStage::Attack => {
                if envelope.attack <= 0.0 {
                    self.current_level = 1.0;
                    self.stage = EnvelopeStage::Decay;
                    self.time_in_stage = 0.0;
                    // Fall through to decay immediately
                    if envelope.decay <= 0.0 {
                        self.current_level = envelope.sustain_level;
                        self.stage = EnvelopeStage::Sustain;
                    }
                } else {
                    self.current_level = (self.time_in_stage / envelope.attack).min(1.0);
                    self.time_in_stage += dt;
                    if self.current_level >= 1.0 {
                        self.current_level = 1.0;
                        self.stage = EnvelopeStage::Decay;
                        self.time_in_stage = 0.0;
                    }
                }
            }
            EnvelopeStage::Decay => {
                if envelope.decay <= 0.0 {
                    self.current_level = envelope.sustain_level;
                    self.stage = EnvelopeStage::Sustain;
                } else {
                    let progress = (self.time_in_stage / envelope.decay).min(1.0);
                    self.current_level =
                        1.0 - progress * (1.0 - envelope.sustain_level);
                    self.time_in_stage += dt;
                    if progress >= 1.0 {
                        self.current_level = envelope.sustain_level;
                        self.stage = EnvelopeStage::Sustain;
                    }
                }
            }
            EnvelopeStage::Sustain => {
                self.current_level = envelope.sustain_level;
            }
            EnvelopeStage::Release => {
                if envelope.release <= 0.0 {
                    self.current_level = 0.0;
                    self.stage = EnvelopeStage::Off;
                } else {
                    let progress = (self.time_in_stage / envelope.release).min(1.0);
                    self.current_level = self.release_start_level * (1.0 - progress);
                    self.time_in_stage += dt;
                    if progress >= 1.0 {
                        self.current_level = 0.0;
                        self.stage = EnvelopeStage::Off;
                    }
                }
            }
            EnvelopeStage::Off => {
                self.current_level = 0.0;
            }
        }

        self.current_level
    }
}

impl Default for EnvelopeState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Voice Allocator ─────────────────────────────────────────────

/// A single synthesizer voice.
#[derive(Debug, Clone)]
pub struct Voice {
    pub frequency: f64,
    pub oscillator: Oscillator,
    pub envelope_state: EnvelopeState,
    pub note_id: u64,
    pub active: bool,
}

/// Polyphonic voice allocator with oldest-voice stealing.
#[derive(Debug)]
pub struct VoiceAllocator {
    voices: Vec<Voice>,
    max_voices: usize,
    next_note_id: u64,
    sample_rate: f64,
    pub waveform: Waveform,
    pub envelope: AdsrEnvelope,
}

impl VoiceAllocator {
    pub fn new(
        max_voices: usize,
        sample_rate: f64,
        waveform: Waveform,
        envelope: AdsrEnvelope,
    ) -> Self {
        Self {
            voices: Vec::with_capacity(max_voices),
            max_voices,
            next_note_id: 1,
            sample_rate,
            waveform,
            envelope,
        }
    }

    /// Trigger a note on at the given frequency. Returns the note ID.
    pub fn note_on(&mut self, frequency: f64) -> u64 {
        let id = self.next_note_id;
        self.next_note_id += 1;

        if self.voices.len() >= self.max_voices {
            // Steal the oldest voice
            if let Some(oldest) = self.voices.first_mut() {
                oldest.frequency = frequency;
                oldest.oscillator = Oscillator::new(self.waveform, frequency, self.sample_rate);
                oldest.envelope_state = EnvelopeState::new();
                oldest.envelope_state.note_on();
                oldest.note_id = id;
                oldest.active = true;
                // Move to end (newest)
                let v = self.voices.remove(0);
                self.voices.push(v);
                return id;
            }
        }

        let mut env_state = EnvelopeState::new();
        env_state.note_on();
        self.voices.push(Voice {
            frequency,
            oscillator: Oscillator::new(self.waveform, frequency, self.sample_rate),
            envelope_state: env_state,
            note_id: id,
            active: true,
        });

        id
    }

    /// Release a note by frequency (triggers release on the first matching voice).
    pub fn note_off_by_frequency(&mut self, frequency: f64) {
        for voice in &mut self.voices {
            if voice.active && (voice.frequency - frequency).abs() < 0.01 {
                voice.envelope_state.note_off();
                voice.active = false;
                break;
            }
        }
    }

    /// Release a note by its ID.
    pub fn note_off(&mut self, note_id: u64) {
        for voice in &mut self.voices {
            if voice.note_id == note_id {
                voice.envelope_state.note_off();
                voice.active = false;
                break;
            }
        }
    }

    /// Generate `num_samples` by summing all voices.
    pub fn render(&mut self, num_samples: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; num_samples];

        for voice in &mut self.voices {
            let samples = voice.oscillator.generate(num_samples);
            for i in 0..num_samples {
                let env = voice.envelope_state.tick(&self.envelope, self.sample_rate);
                output[i] += samples[i] * env as f32;
            }
        }

        // Remove fully-off voices
        self.voices.retain(|v| v.envelope_state.stage() != EnvelopeStage::Off);

        output
    }

    pub fn active_voice_count(&self) -> usize {
        self.voices.len()
    }
}

// ── LFO ─────────────────────────────────────────────────────────

/// Low Frequency Oscillator for modulation.
#[derive(Debug, Clone)]
pub struct Lfo {
    pub waveform: Waveform,
    pub rate: f64,
    pub depth: f64,
    phase: f64,
    sample_rate: f64,
}

impl Lfo {
    pub fn new(waveform: Waveform, rate: f64, depth: f64, sample_rate: f64) -> Self {
        Self {
            waveform,
            rate,
            depth,
            phase: 0.0,
            sample_rate,
        }
    }

    /// Generate `num_samples` of modulation signal (bipolar, scaled by depth).
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        let mut samples =
            generate_waveform(self.waveform, self.rate, self.sample_rate, num_samples, &mut self.phase);
        for s in &mut samples {
            *s *= self.depth as f32;
        }
        samples
    }

    /// Apply LFO modulation to a frequency, returning modulated frequency per sample.
    pub fn modulate_frequency(
        &mut self,
        base_freq: f64,
        num_samples: usize,
    ) -> Vec<f64> {
        let lfo_samples = self.generate(num_samples);
        lfo_samples
            .iter()
            .map(|s| base_freq * (1.0 + *s as f64))
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_generates_correct_range() {
        let mut phase = 0.0;
        let samples = generate_waveform(Waveform::Sine, 440.0, 44100.0, 1000, &mut phase);
        assert_eq!(samples.len(), 1000);
        for s in &samples {
            assert!(*s >= -1.0 && *s <= 1.0, "sample {} out of range", s);
        }
    }

    #[test]
    fn square_wave_values() {
        let mut phase = 0.0;
        let samples = generate_waveform(Waveform::Square, 1.0, 4.0, 4, &mut phase);
        // At sample_rate=4, freq=1: phase goes 0.0, 0.25, 0.5, 0.75
        assert_eq!(samples[0], 1.0);  // phase 0.0 < 0.5
        assert_eq!(samples[1], 1.0);  // phase 0.25 < 0.5
        assert_eq!(samples[2], -1.0); // phase 0.5 >= 0.5
        assert_eq!(samples[3], -1.0); // phase 0.75 >= 0.5
    }

    #[test]
    fn sawtooth_ramp() {
        let mut phase = 0.0;
        let samples = generate_waveform(Waveform::Sawtooth, 1.0, 8.0, 8, &mut phase);
        // Should ramp from -1 to near +1 over one period
        assert!(samples[0] < samples[7]);
        assert!((samples[0] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn triangle_symmetry() {
        let mut phase = 0.0;
        let samples = generate_waveform(Waveform::Triangle, 1.0, 100.0, 100, &mut phase);
        // Peak should be around sample 25 (phase 0.25 -> value 0.0, phase 0.5 is peak)
        // Actually at phase 0.25 -> 4*0.25 - 1 = 0.0, at phase 0.5 boundary
        for s in &samples {
            assert!(*s >= -1.001 && *s <= 1.001);
        }
    }

    #[test]
    fn oscillator_state() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, 44100.0);
        let s1 = osc.generate(100);
        let s2 = osc.generate(100);
        assert_eq!(s1.len(), 100);
        assert_eq!(s2.len(), 100);
        // Phase should have advanced so s2 continues from where s1 left off
        // They shouldn't be identical
        assert_ne!(s1, s2);
    }

    #[test]
    fn adsr_attack_to_sustain() {
        let env = AdsrEnvelope {
            attack: 0.01,
            decay: 0.01,
            sustain_level: 0.5,
            release: 0.01,
        };
        let mut state = EnvelopeState::new();
        state.note_on();

        let sr = 44100.0;
        // Run through attack + decay
        let mut last = 0.0;
        for _ in 0..2000 {
            last = state.tick(&env, sr);
        }
        // Should be at sustain level
        assert_eq!(state.stage(), EnvelopeStage::Sustain);
        assert!((last - 0.5).abs() < 0.01);
    }

    #[test]
    fn adsr_release_to_off() {
        let env = AdsrEnvelope {
            attack: 0.0,
            decay: 0.0,
            sustain_level: 1.0,
            release: 0.01,
        };
        let mut state = EnvelopeState::new();
        state.note_on();
        // Immediately at sustain
        state.tick(&env, 44100.0);
        assert_eq!(state.stage(), EnvelopeStage::Sustain);

        state.note_off();
        for _ in 0..1000 {
            state.tick(&env, 44100.0);
        }
        assert_eq!(state.stage(), EnvelopeStage::Off);
        assert!((state.current_level()).abs() < 0.001);
    }

    #[test]
    fn voice_allocator_basic() {
        let env = AdsrEnvelope {
            attack: 0.001,
            decay: 0.001,
            sustain_level: 0.8,
            release: 0.001,
        };
        let mut alloc = VoiceAllocator::new(4, 44100.0, Waveform::Sine, env);
        alloc.note_on(440.0);
        alloc.note_on(550.0);
        assert_eq!(alloc.active_voice_count(), 2);
    }

    #[test]
    fn voice_allocator_stealing() {
        let env = AdsrEnvelope {
            attack: 0.0,
            decay: 0.0,
            sustain_level: 1.0,
            release: 10.0, // Long release so voices stay
        };
        let mut alloc = VoiceAllocator::new(2, 44100.0, Waveform::Sine, env);
        alloc.note_on(440.0);
        alloc.note_on(550.0);
        // Third note should steal oldest
        alloc.note_on(660.0);
        assert_eq!(alloc.active_voice_count(), 2);
    }

    #[test]
    fn voice_allocator_render() {
        let env = AdsrEnvelope {
            attack: 0.0,
            decay: 0.0,
            sustain_level: 1.0,
            release: 0.0,
        };
        let mut alloc = VoiceAllocator::new(4, 44100.0, Waveform::Sine, env);
        alloc.note_on(440.0);
        let samples = alloc.render(256);
        assert_eq!(samples.len(), 256);
        // Should have non-zero samples
        let energy: f32 = samples.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn lfo_modulation() {
        let mut lfo = Lfo::new(Waveform::Sine, 5.0, 0.1, 44100.0);
        let freqs = lfo.modulate_frequency(440.0, 100);
        assert_eq!(freqs.len(), 100);
        // Modulated frequencies should be near 440 +/- 10%
        for f in &freqs {
            assert!(*f > 380.0 && *f < 500.0);
        }
    }

    #[test]
    fn voice_note_off_by_id() {
        let env = AdsrEnvelope {
            attack: 0.0,
            decay: 0.0,
            sustain_level: 1.0,
            release: 0.0,
        };
        let mut alloc = VoiceAllocator::new(4, 44100.0, Waveform::Sine, env);
        let id = alloc.note_on(440.0);
        assert_eq!(alloc.active_voice_count(), 1);
        alloc.note_off(id);
        alloc.render(1); // Tick to flush Off voices
        assert_eq!(alloc.active_voice_count(), 0);
    }
}
