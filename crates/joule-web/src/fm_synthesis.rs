//! FM (Frequency Modulation) synthesis engine with configurable operator routing.
//!
//! Implements DX7-style 4-operator FM synthesis with configurable algorithms,
//! modulation index, operator feedback, ratio/detune per operator, velocity
//! and key scaling, and preset patches. Pure Rust — no DSP library deps.

use std::f64::consts::TAU;

// ── Operator ────────────────────────────────────────────────────

/// A single FM operator: oscillator + envelope + level.
#[derive(Debug, Clone)]
pub struct Operator {
    /// Frequency ratio relative to the note frequency.
    pub ratio: f64,
    /// Fine detune in Hz.
    pub detune_hz: f64,
    /// Output level (0.0..1.0).
    pub level: f64,
    /// Modulation index (controls brightness when this op modulates another).
    pub mod_index: f64,
    /// Self-feedback amount (0.0..1.0).
    pub feedback: f64,
    /// Velocity sensitivity (0.0 = ignore velocity, 1.0 = full).
    pub velocity_sensitivity: f64,
    /// Key scaling: positive = brighter at higher notes, 0 = none.
    pub key_scaling: f64,
    /// Simple ADSR-like envelope: attack/decay/sustain/release rates.
    pub attack_rate: f64,
    pub decay_rate: f64,
    pub sustain_level: f64,
    pub release_rate: f64,
    // Internal state.
    phase: f64,
    envelope_level: f64,
    envelope_stage: EnvStage,
    prev_output: f64,
}

/// Simple envelope stages for operator amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EnvStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl Operator {
    /// Create a new operator with default settings.
    pub fn new() -> Self {
        Self {
            ratio: 1.0,
            detune_hz: 0.0,
            level: 1.0,
            mod_index: 1.0,
            feedback: 0.0,
            velocity_sensitivity: 0.5,
            key_scaling: 0.0,
            attack_rate: 0.99,
            decay_rate: 0.999,
            sustain_level: 0.7,
            release_rate: 0.995,
            phase: 0.0,
            envelope_level: 0.0,
            envelope_stage: EnvStage::Idle,
            prev_output: 0.0,
        }
    }

    /// Trigger the operator (note on).
    pub fn note_on(&mut self) {
        self.envelope_stage = EnvStage::Attack;
        self.phase = 0.0;
    }

    /// Release the operator (note off).
    pub fn note_off(&mut self) {
        if self.envelope_stage != EnvStage::Idle {
            self.envelope_stage = EnvStage::Release;
        }
    }

    /// Reset to idle.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.envelope_level = 0.0;
        self.envelope_stage = EnvStage::Idle;
        self.prev_output = 0.0;
    }

    /// Is the operator idle?
    pub fn is_idle(&self) -> bool {
        self.envelope_stage == EnvStage::Idle
    }

    /// Advance the simple envelope by one sample.
    fn advance_envelope(&mut self) {
        match self.envelope_stage {
            EnvStage::Idle => {}
            EnvStage::Attack => {
                self.envelope_level += (1.0 - self.envelope_level) * (1.0 - self.attack_rate);
                if self.envelope_level > 0.999 {
                    self.envelope_level = 1.0;
                    self.envelope_stage = EnvStage::Decay;
                }
            }
            EnvStage::Decay => {
                self.envelope_level -= (self.envelope_level - self.sustain_level) * (1.0 - self.decay_rate);
                if (self.envelope_level - self.sustain_level).abs() < 0.001 {
                    self.envelope_level = self.sustain_level;
                    self.envelope_stage = EnvStage::Sustain;
                }
            }
            EnvStage::Sustain => {
                // Hold at sustain level.
            }
            EnvStage::Release => {
                self.envelope_level *= self.release_rate;
                if self.envelope_level < 0.001 {
                    self.envelope_level = 0.0;
                    self.envelope_stage = EnvStage::Idle;
                }
            }
        }
    }

    /// Compute the operator's output for one sample.
    /// `base_freq` is the note frequency, `modulation` is the phase modulation input,
    /// `velocity` is 0..1, `note_number` is MIDI note for key scaling.
    pub fn process(
        &mut self,
        base_freq: f64,
        sample_rate: f64,
        modulation: f64,
        velocity: f64,
        note_number: f64,
    ) -> f64 {
        self.advance_envelope();

        let freq = base_freq * self.ratio + self.detune_hz;
        let dt = freq / sample_rate;

        // Self-feedback.
        let fb = self.prev_output * self.feedback * TAU;

        // Phase modulation.
        let phase_mod = modulation * self.mod_index * TAU;

        let output = (self.phase * TAU + phase_mod + fb).sin();

        // Apply envelope, velocity, and key scaling.
        let vel_scale = 1.0 - self.velocity_sensitivity * (1.0 - velocity);
        let key_scale = 1.0 + self.key_scaling * (note_number - 60.0) / 60.0;
        let scaled = output * self.level * self.envelope_level * vel_scale * key_scale.max(0.0);

        self.phase += dt;
        self.phase = self.phase.rem_euclid(1.0);
        self.prev_output = scaled;

        scaled
    }

    /// Current envelope level.
    pub fn envelope_level(&self) -> f64 {
        self.envelope_level
    }
}

// ── Algorithm ───────────────────────────────────────────────────

/// Routing topology for 4 operators. Each entry describes which operators
/// modulate which, and which are carriers (output to audio).
#[derive(Debug, Clone, PartialEq)]
pub struct Algorithm {
    /// Name of the algorithm.
    pub name: String,
    /// For each operator (index 0-3), list of operators that modulate it.
    pub modulators: [Vec<usize>; 4],
    /// Which operators are carriers (their output goes to audio).
    pub carriers: Vec<usize>,
}

impl Algorithm {
    /// Algorithm 1: Serial chain 4→3→2→1 (1 is carrier).
    pub fn serial_chain() -> Self {
        Self {
            name: "Serial 4→3→2→1".into(),
            modulators: [
                vec![1],    // op0 modulated by op1
                vec![2],    // op1 modulated by op2
                vec![3],    // op2 modulated by op3
                vec![],     // op3 not modulated
            ],
            carriers: vec![0],
        }
    }

    /// Algorithm 2: Parallel — all 4 operators are carriers (organ-like).
    pub fn parallel() -> Self {
        Self {
            name: "Parallel (all carriers)".into(),
            modulators: [vec![], vec![], vec![], vec![]],
            carriers: vec![0, 1, 2, 3],
        }
    }

    /// Algorithm 3: Stacked pairs — (4→3) + (2→1), ops 1 and 3 are carriers.
    pub fn stacked_pairs() -> Self {
        Self {
            name: "Stacked (4→3) + (2→1)".into(),
            modulators: [
                vec![1],    // op0 modulated by op1
                vec![],     // op1 is modulator only
                vec![3],    // op2 modulated by op3
                vec![],     // op3 is modulator only
            ],
            carriers: vec![0, 2],
        }
    }

    /// Algorithm 4: 4→(1,2,3) — op4 modulates all others.
    pub fn one_modulates_three() -> Self {
        Self {
            name: "4→(1,2,3)".into(),
            modulators: [
                vec![3],    // op0 modulated by op3
                vec![3],    // op1 modulated by op3
                vec![3],    // op2 modulated by op3
                vec![],
            ],
            carriers: vec![0, 1, 2],
        }
    }

    /// Algorithm 5: (3→2→1) + 4 — serial trio with op4 as separate carrier.
    pub fn trio_plus_one() -> Self {
        Self {
            name: "(3→2→1) + 4".into(),
            modulators: [
                vec![1],
                vec![2],
                vec![],
                vec![],
            ],
            carriers: vec![0, 3],
        }
    }

    /// Algorithm 6: (4→3→1) + (4→2) — op4 feeds both chains.
    pub fn branched() -> Self {
        Self {
            name: "(4→3→1) + (4→2)".into(),
            modulators: [
                vec![2],    // op0 modulated by op2
                vec![3],    // op1 modulated by op3
                vec![3],    // op2 modulated by op3
                vec![],
            ],
            carriers: vec![0, 1],
        }
    }
}

// ── FM Synth Voice ──────────────────────────────────────────────

/// A 4-operator FM synthesiser voice.
#[derive(Debug, Clone)]
pub struct FmVoice {
    pub operators: [Operator; 4],
    pub algorithm: Algorithm,
    /// Base note frequency in Hz.
    pub frequency: f64,
    /// MIDI note number (for key scaling).
    pub note_number: f64,
    /// Velocity (0.0..1.0).
    pub velocity: f64,
    /// Sample rate.
    pub sample_rate: f64,
}

impl FmVoice {
    /// Create a new FM voice with the given algorithm.
    pub fn new(algorithm: Algorithm, frequency: f64, sample_rate: f64) -> Self {
        Self {
            operators: [Operator::new(), Operator::new(), Operator::new(), Operator::new()],
            algorithm,
            frequency,
            note_number: 60.0,
            velocity: 1.0,
            sample_rate,
        }
    }

    /// Trigger all operators (note on).
    pub fn note_on(&mut self, velocity: f64) {
        self.velocity = velocity.clamp(0.0, 1.0);
        for op in &mut self.operators {
            op.note_on();
        }
    }

    /// Release all operators (note off).
    pub fn note_off(&mut self) {
        for op in &mut self.operators {
            op.note_off();
        }
    }

    /// Is the voice idle (all operators finished)?
    pub fn is_idle(&self) -> bool {
        self.operators.iter().all(|op| op.is_idle())
    }

    /// Reset all operators.
    pub fn reset(&mut self) {
        for op in &mut self.operators {
            op.reset();
        }
    }

    /// Process one sample.
    pub fn next_sample(&mut self) -> f64 {
        // Compute operator outputs in dependency order.
        // We need two passes: first compute modulation sums, then compute outputs.
        let mut outputs = [0.0_f64; 4];

        // Simple iterative approach: compute each operator with modulation from previous frame.
        // For a 4-op system this is standard practice (one sample delay for feedback paths).
        let prev_outputs = [
            self.operators[0].prev_output,
            self.operators[1].prev_output,
            self.operators[2].prev_output,
            self.operators[3].prev_output,
        ];

        for i in 0..4 {
            let mut modulation = 0.0;
            for &mod_idx in &self.algorithm.modulators[i] {
                if mod_idx < 4 {
                    modulation += prev_outputs[mod_idx];
                }
            }
            outputs[i] = self.operators[i].process(
                self.frequency,
                self.sample_rate,
                modulation,
                self.velocity,
                self.note_number,
            );
        }

        // Sum carrier outputs.
        let mut sum = 0.0;
        for &carrier_idx in &self.algorithm.carriers {
            if carrier_idx < 4 {
                sum += outputs[carrier_idx];
            }
        }

        sum
    }

    /// Generate a block of samples.
    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }
}

// ── Preset Patches ──────────────────────────────────────────────

/// Collection of preset FM patches.
pub struct FmPreset;

impl FmPreset {
    /// Electric piano patch (DX7 "E.PIANO 1" approximation).
    pub fn electric_piano(sample_rate: f64) -> FmVoice {
        let mut voice = FmVoice::new(Algorithm::stacked_pairs(), 440.0, sample_rate);
        // Op 0: carrier, ratio 1, level 0.8
        voice.operators[0].ratio = 1.0;
        voice.operators[0].level = 0.8;
        voice.operators[0].attack_rate = 0.95;
        voice.operators[0].decay_rate = 0.9998;
        voice.operators[0].sustain_level = 0.2;
        // Op 1: modulator for op0, ratio 1, high mod index
        voice.operators[1].ratio = 1.0;
        voice.operators[1].level = 1.0;
        voice.operators[1].mod_index = 2.0;
        voice.operators[1].attack_rate = 0.9;
        voice.operators[1].decay_rate = 0.999;
        voice.operators[1].sustain_level = 0.1;
        // Op 2: carrier, ratio 1, adds brightness
        voice.operators[2].ratio = 1.0;
        voice.operators[2].level = 0.3;
        voice.operators[2].attack_rate = 0.8;
        voice.operators[2].decay_rate = 0.9995;
        voice.operators[2].sustain_level = 0.0;
        // Op 3: modulator for op2
        voice.operators[3].ratio = 14.0;
        voice.operators[3].level = 0.5;
        voice.operators[3].mod_index = 0.5;
        voice.operators[3].attack_rate = 0.8;
        voice.operators[3].decay_rate = 0.998;
        voice.operators[3].sustain_level = 0.0;
        voice
    }

    /// Bass patch.
    pub fn bass(sample_rate: f64) -> FmVoice {
        let mut voice = FmVoice::new(Algorithm::serial_chain(), 110.0, sample_rate);
        voice.operators[0].ratio = 1.0;
        voice.operators[0].level = 1.0;
        voice.operators[1].ratio = 1.0;
        voice.operators[1].mod_index = 3.0;
        voice.operators[1].level = 0.8;
        voice.operators[1].decay_rate = 0.9995;
        voice.operators[1].sustain_level = 0.1;
        voice.operators[2].ratio = 1.0;
        voice.operators[2].mod_index = 0.5;
        voice.operators[2].level = 0.3;
        voice.operators[3].ratio = 2.0;
        voice.operators[3].mod_index = 0.3;
        voice.operators[3].level = 0.2;
        voice
    }

    /// Bright bell patch.
    pub fn bell(sample_rate: f64) -> FmVoice {
        let mut voice = FmVoice::new(Algorithm::serial_chain(), 880.0, sample_rate);
        voice.operators[0].ratio = 1.0;
        voice.operators[0].level = 0.7;
        voice.operators[0].attack_rate = 0.8;
        voice.operators[0].decay_rate = 0.99998;
        voice.operators[0].sustain_level = 0.0;
        voice.operators[1].ratio = 3.5;
        voice.operators[1].mod_index = 5.0;
        voice.operators[1].level = 1.0;
        voice.operators[1].attack_rate = 0.7;
        voice.operators[1].decay_rate = 0.9999;
        voice.operators[1].sustain_level = 0.0;
        voice.operators[2].ratio = 7.0;
        voice.operators[2].mod_index = 2.0;
        voice.operators[2].level = 0.5;
        voice.operators[3].ratio = 1.0;
        voice.operators[3].mod_index = 0.0;
        voice.operators[3].level = 0.0;
        voice
    }

    /// Organ patch (all parallel carriers with integer ratios).
    pub fn organ(sample_rate: f64) -> FmVoice {
        let mut voice = FmVoice::new(Algorithm::parallel(), 261.63, sample_rate);
        voice.operators[0].ratio = 1.0;
        voice.operators[0].level = 1.0;
        voice.operators[1].ratio = 2.0;
        voice.operators[1].level = 0.7;
        voice.operators[2].ratio = 3.0;
        voice.operators[2].level = 0.4;
        voice.operators[3].ratio = 4.0;
        voice.operators[3].level = 0.2;
        voice
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f64 = 1e-4;

    #[test]
    fn test_operator_idle_by_default() {
        let op = Operator::new();
        assert!(op.is_idle());
        assert!((op.envelope_level() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_operator_note_on_starts_attack() {
        let mut op = Operator::new();
        op.note_on();
        assert!(!op.is_idle());
    }

    #[test]
    fn test_operator_envelope_rises() {
        let mut op = Operator::new();
        op.attack_rate = 0.9;
        op.note_on();
        for _ in 0..200 {
            op.process(440.0, SR, 0.0, 1.0, 60.0);
        }
        assert!(op.envelope_level() > 0.5, "envelope should rise, got {}", op.envelope_level());
    }

    #[test]
    fn test_operator_release_decays() {
        let mut op = Operator::new();
        op.attack_rate = 0.5; // fast attack
        op.note_on();
        for _ in 0..500 {
            op.process(440.0, SR, 0.0, 1.0, 60.0);
        }
        op.note_off();
        for _ in 0..5000 {
            op.process(440.0, SR, 0.0, 1.0, 60.0);
        }
        assert!(op.envelope_level() < 0.1, "should decay after release, got {}", op.envelope_level());
    }

    #[test]
    fn test_operator_reset() {
        let mut op = Operator::new();
        op.note_on();
        for _ in 0..100 {
            op.process(440.0, SR, 0.0, 1.0, 60.0);
        }
        op.reset();
        assert!(op.is_idle());
        assert!((op.envelope_level() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_operator_velocity_scaling() {
        let mut op1 = Operator::new();
        op1.velocity_sensitivity = 1.0;
        op1.attack_rate = 0.5;
        op1.note_on();
        // High velocity
        let mut sum_high = 0.0;
        for _ in 0..100 {
            sum_high += op1.process(440.0, SR, 0.0, 1.0, 60.0).abs();
        }

        let mut op2 = Operator::new();
        op2.velocity_sensitivity = 1.0;
        op2.attack_rate = 0.5;
        op2.note_on();
        let mut sum_low = 0.0;
        for _ in 0..100 {
            sum_low += op2.process(440.0, SR, 0.0, 0.2, 60.0).abs();
        }
        assert!(sum_high > sum_low, "high velocity should be louder: high={sum_high} low={sum_low}");
    }

    #[test]
    fn test_algorithm_serial_chain() {
        let alg = Algorithm::serial_chain();
        assert_eq!(alg.carriers, vec![0]);
        assert_eq!(alg.modulators[0], vec![1]);
        assert_eq!(alg.modulators[2], vec![3]);
    }

    #[test]
    fn test_algorithm_parallel() {
        let alg = Algorithm::parallel();
        assert_eq!(alg.carriers.len(), 4);
        for mods in &alg.modulators {
            assert!(mods.is_empty());
        }
    }

    #[test]
    fn test_fm_voice_produces_output() {
        let mut voice = FmVoice::new(Algorithm::serial_chain(), 440.0, SR);
        voice.note_on(1.0);
        let mut buf = vec![0.0; 1024];
        voice.generate_block(&mut buf);
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "FM voice should produce nonzero output");
    }

    #[test]
    fn test_fm_voice_idle_after_release() {
        let mut voice = FmVoice::new(Algorithm::parallel(), 440.0, SR);
        voice.note_on(1.0);
        for _ in 0..500 {
            voice.next_sample();
        }
        voice.note_off();
        // Run enough samples for release to finish.
        for _ in 0..50000 {
            voice.next_sample();
        }
        assert!(voice.is_idle(), "voice should be idle after long release");
    }

    #[test]
    fn test_fm_voice_modulation_changes_timbre() {
        // With modulation (serial chain), output spectrum should differ from parallel.
        let mut voice_mod = FmVoice::new(Algorithm::serial_chain(), 440.0, SR);
        voice_mod.operators[1].mod_index = 5.0;
        voice_mod.note_on(1.0);
        let mut buf_mod = vec![0.0; 4096];
        voice_mod.generate_block(&mut buf_mod);

        let mut voice_no_mod = FmVoice::new(Algorithm::parallel(), 440.0, SR);
        voice_no_mod.note_on(1.0);
        let mut buf_clean = vec![0.0; 4096];
        voice_no_mod.generate_block(&mut buf_clean);

        // Compare — they should differ.
        let diff: f64 = buf_mod.iter().zip(buf_clean.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0, "modulated and clean should differ: diff={diff}");
    }

    #[test]
    fn test_operator_feedback() {
        let mut op_fb = Operator::new();
        op_fb.feedback = 0.5;
        op_fb.attack_rate = 0.5;
        op_fb.note_on();
        let mut sum_fb = 0.0;
        for _ in 0..500 {
            sum_fb += op_fb.process(440.0, SR, 0.0, 1.0, 60.0).abs();
        }

        let mut op_no_fb = Operator::new();
        op_no_fb.feedback = 0.0;
        op_no_fb.attack_rate = 0.5;
        op_no_fb.note_on();
        let mut sum_no_fb = 0.0;
        for _ in 0..500 {
            sum_no_fb += op_no_fb.process(440.0, SR, 0.0, 1.0, 60.0).abs();
        }
        // Feedback adds harmonics, RMS may differ.
        assert!(sum_fb > 0.0 && sum_no_fb > 0.0, "both should produce output");
    }

    #[test]
    fn test_preset_electric_piano() {
        let mut voice = FmPreset::electric_piano(SR);
        voice.note_on(0.8);
        let mut buf = vec![0.0; 2048];
        voice.generate_block(&mut buf);
        let rms: f64 = (buf.iter().map(|s| s * s).sum::<f64>() / buf.len() as f64).sqrt();
        assert!(rms > 0.001, "e-piano preset should produce output");
    }

    #[test]
    fn test_preset_bass() {
        let mut voice = FmPreset::bass(SR);
        voice.note_on(1.0);
        let mut buf = vec![0.0; 2048];
        voice.generate_block(&mut buf);
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "bass preset should produce output");
    }

    #[test]
    fn test_preset_bell() {
        let mut voice = FmPreset::bell(SR);
        voice.note_on(1.0);
        let mut buf = vec![0.0; 2048];
        voice.generate_block(&mut buf);
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "bell preset should produce output");
    }

    #[test]
    fn test_preset_organ() {
        let mut voice = FmPreset::organ(SR);
        voice.note_on(1.0);
        let mut buf = vec![0.0; 2048];
        voice.generate_block(&mut buf);
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.001);
        assert!(any_nonzero, "organ preset should produce output");
    }

    #[test]
    fn test_key_scaling_effect() {
        let mut op_high = Operator::new();
        op_high.key_scaling = 1.0;
        op_high.attack_rate = 0.5;
        op_high.note_on();
        let mut sum_high = 0.0;
        for _ in 0..200 {
            sum_high += op_high.process(440.0, SR, 0.0, 1.0, 96.0).abs(); // high note
        }

        let mut op_low = Operator::new();
        op_low.key_scaling = 1.0;
        op_low.attack_rate = 0.5;
        op_low.note_on();
        let mut sum_low = 0.0;
        for _ in 0..200 {
            sum_low += op_low.process(440.0, SR, 0.0, 1.0, 24.0).abs(); // low note
        }
        assert!(sum_high > sum_low, "high note should be brighter: high={sum_high} low={sum_low}");
    }

    #[test]
    fn test_ratio_affects_pitch() {
        let mut op1 = Operator::new();
        op1.ratio = 1.0;
        op1.attack_rate = 0.1;
        op1.note_on();
        let samples1: Vec<f64> = (0..441).map(|_| op1.process(440.0, SR, 0.0, 1.0, 60.0)).collect();

        let mut op2 = Operator::new();
        op2.ratio = 2.0;
        op2.attack_rate = 0.1;
        op2.note_on();
        let samples2: Vec<f64> = (0..441).map(|_| op2.process(440.0, SR, 0.0, 1.0, 60.0)).collect();

        // Count zero crossings — ratio 2 should have ~double.
        fn zero_crossings(s: &[f64]) -> usize {
            s.windows(2).filter(|w| w[0].signum() != w[1].signum()).count()
        }
        let zc1 = zero_crossings(&samples1);
        let zc2 = zero_crossings(&samples2);
        assert!(zc2 > zc1, "ratio 2 should have more zero crossings: r1={zc1} r2={zc2}");
    }

    #[test]
    fn test_all_algorithms_produce_output() {
        let algorithms = vec![
            Algorithm::serial_chain(),
            Algorithm::parallel(),
            Algorithm::stacked_pairs(),
            Algorithm::one_modulates_three(),
            Algorithm::trio_plus_one(),
            Algorithm::branched(),
        ];
        for alg in algorithms {
            let name = alg.name.clone();
            let mut voice = FmVoice::new(alg, 440.0, SR);
            voice.note_on(1.0);
            let mut buf = vec![0.0; 512];
            voice.generate_block(&mut buf);
            let any_nonzero = buf.iter().any(|s| s.abs() > 0.0001);
            assert!(any_nonzero, "algorithm '{name}' should produce output");
        }
    }
}
