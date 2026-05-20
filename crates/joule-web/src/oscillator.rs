//! Audio oscillators — sine, square, sawtooth, triangle wave generation.
//!
//! Provides configurable oscillators with phase accumulator, wavetable lookup,
//! FM synthesis basics, LFO (low frequency oscillator), and oscillator bank
//! for polyphonic synthesis.

use std::f64::consts::PI;

// ── Waveform Types ──────────────────────────────────────────────

/// Oscillator waveform shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

// ── Oscillator ──────────────────────────────────────────────────

/// A single audio oscillator with phase accumulator.
#[derive(Debug, Clone)]
pub struct Oscillator {
    waveform: Waveform,
    frequency: f64,
    phase: f64,
    amplitude: f64,
    sample_rate: f64,
    /// Phase offset in radians (0..2*PI).
    phase_offset: f64,
    /// Pulse width for square wave (0.0..1.0), default 0.5.
    pulse_width: f64,
}

impl Oscillator {
    /// Create a new oscillator.
    pub fn new(waveform: Waveform, frequency: f64, sample_rate: f64) -> Self {
        Self {
            waveform,
            frequency,
            phase: 0.0,
            amplitude: 1.0,
            sample_rate,
            phase_offset: 0.0,
            pulse_width: 0.5,
        }
    }

    pub fn waveform(&self) -> Waveform {
        self.waveform
    }

    pub fn set_waveform(&mut self, waveform: Waveform) {
        self.waveform = waveform;
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
    }

    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }

    pub fn set_amplitude(&mut self, amp: f64) {
        self.amplitude = amp;
    }

    pub fn phase(&self) -> f64 {
        self.phase
    }

    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase % 1.0;
    }

    pub fn set_phase_offset(&mut self, offset: f64) {
        self.phase_offset = offset;
    }

    pub fn set_pulse_width(&mut self, pw: f64) {
        self.pulse_width = pw.clamp(0.01, 0.99);
    }

    /// Reset phase accumulator to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Generate a single sample and advance the phase.
    pub fn tick(&mut self) -> f32 {
        let t = self.phase + self.phase_offset / (2.0 * PI);
        let t_wrapped = t - t.floor();
        let sample = self.compute_sample(t_wrapped);
        self.advance_phase();
        (sample * self.amplitude) as f32
    }

    /// Generate `count` samples into the output buffer.
    pub fn generate(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s = self.tick();
        }
    }

    /// Generate samples and add to existing buffer content.
    pub fn generate_add(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s += self.tick();
        }
    }

    fn compute_sample(&self, t: f64) -> f64 {
        match self.waveform {
            Waveform::Sine => (t * 2.0 * PI).sin(),
            Waveform::Square => {
                if t < self.pulse_width {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 2.0 * t - 1.0,
            Waveform::Triangle => {
                if t < 0.5 {
                    4.0 * t - 1.0
                } else {
                    3.0 - 4.0 * t
                }
            }
        }
    }

    fn advance_phase(&mut self) {
        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
    }
}

// ── Wavetable Oscillator ────────────────────────────────────────

/// Oscillator that reads from a pre-computed wavetable for efficiency.
#[derive(Debug, Clone)]
pub struct WavetableOscillator {
    table: Vec<f32>,
    phase: f64,
    frequency: f64,
    amplitude: f64,
    sample_rate: f64,
}

impl WavetableOscillator {
    /// Create a wavetable oscillator from a table of one cycle.
    pub fn new(table: Vec<f32>, frequency: f64, sample_rate: f64) -> Self {
        Self {
            table,
            phase: 0.0,
            frequency,
            amplitude: 1.0,
            sample_rate,
        }
    }

    /// Create a sine wavetable of given size.
    pub fn sine(table_size: usize, frequency: f64, sample_rate: f64) -> Self {
        let table: Vec<f32> = (0..table_size)
            .map(|i| {
                let t = i as f64 / table_size as f64;
                (t * 2.0 * PI).sin() as f32
            })
            .collect();
        Self::new(table, frequency, sample_rate)
    }

    /// Create a sawtooth wavetable of given size.
    pub fn sawtooth(table_size: usize, frequency: f64, sample_rate: f64) -> Self {
        let table: Vec<f32> = (0..table_size)
            .map(|i| {
                let t = i as f64 / table_size as f64;
                (2.0 * t - 1.0) as f32
            })
            .collect();
        Self::new(table, frequency, sample_rate)
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
    }

    pub fn set_amplitude(&mut self, amp: f64) {
        self.amplitude = amp;
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    pub fn table_size(&self) -> usize {
        self.table.len()
    }

    /// Generate one sample using linear interpolation.
    pub fn tick(&mut self) -> f32 {
        if self.table.is_empty() {
            return 0.0;
        }
        let table_len = self.table.len() as f64;
        let pos = self.phase * table_len;
        let idx0 = pos.floor() as usize % self.table.len();
        let idx1 = (idx0 + 1) % self.table.len();
        let frac = pos - pos.floor();

        let sample = self.table[idx0] as f64 * (1.0 - frac) + self.table[idx1] as f64 * frac;

        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        (sample * self.amplitude) as f32
    }

    /// Generate `count` samples.
    pub fn generate(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s = self.tick();
        }
    }
}

// ── FM Oscillator ───────────────────────────────────────────────

/// Simple two-operator FM synthesis oscillator.
#[derive(Debug, Clone)]
pub struct FmOscillator {
    /// Carrier oscillator.
    carrier: Oscillator,
    /// Modulator oscillator.
    modulator: Oscillator,
    /// Modulation index (depth of FM).
    mod_index: f64,
}

impl FmOscillator {
    /// Create an FM oscillator with carrier and modulator frequencies.
    pub fn new(
        carrier_freq: f64,
        modulator_freq: f64,
        mod_index: f64,
        sample_rate: f64,
    ) -> Self {
        Self {
            carrier: Oscillator::new(Waveform::Sine, carrier_freq, sample_rate),
            modulator: Oscillator::new(Waveform::Sine, modulator_freq, sample_rate),
            mod_index,
        }
    }

    pub fn set_carrier_frequency(&mut self, freq: f64) {
        self.carrier.set_frequency(freq);
    }

    pub fn set_modulator_frequency(&mut self, freq: f64) {
        self.modulator.set_frequency(freq);
    }

    pub fn set_mod_index(&mut self, index: f64) {
        self.mod_index = index;
    }

    pub fn set_amplitude(&mut self, amp: f64) {
        self.carrier.set_amplitude(amp);
    }

    pub fn reset(&mut self) {
        self.carrier.reset();
        self.modulator.reset();
    }

    /// Generate one FM sample.
    pub fn tick(&mut self) -> f32 {
        // Get modulator output (normalized to [-1, 1])
        let mod_sample = self.modulator.tick() as f64;
        // Apply modulation to carrier phase
        let freq_mod = mod_sample * self.mod_index * self.carrier.frequency();
        let original_freq = self.carrier.frequency();
        self.carrier.set_frequency(original_freq + freq_mod);
        let sample = self.carrier.tick();
        self.carrier.set_frequency(original_freq);
        sample
    }

    /// Generate samples into buffer.
    pub fn generate(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s = self.tick();
        }
    }
}

// ── LFO ─────────────────────────────────────────────────────────

/// Low Frequency Oscillator for modulation purposes.
#[derive(Debug, Clone)]
pub struct Lfo {
    oscillator: Oscillator,
    /// Minimum output value.
    min_value: f64,
    /// Maximum output value.
    max_value: f64,
}

impl Lfo {
    /// Create an LFO with given waveform and rate (Hz).
    pub fn new(waveform: Waveform, rate: f64, sample_rate: f64) -> Self {
        Self {
            oscillator: Oscillator::new(waveform, rate, sample_rate),
            min_value: 0.0,
            max_value: 1.0,
        }
    }

    /// Set the output range of the LFO.
    pub fn set_range(&mut self, min: f64, max: f64) {
        self.min_value = min;
        self.max_value = max;
    }

    pub fn set_rate(&mut self, rate: f64) {
        self.oscillator.set_frequency(rate);
    }

    pub fn set_waveform(&mut self, waveform: Waveform) {
        self.oscillator.set_waveform(waveform);
    }

    pub fn reset(&mut self) {
        self.oscillator.reset();
    }

    /// Get next LFO value mapped to [min_value, max_value].
    pub fn tick(&mut self) -> f32 {
        let raw = self.oscillator.tick() as f64; // -1..1
        let normalized = (raw + 1.0) * 0.5; // 0..1
        let mapped = self.min_value + normalized * (self.max_value - self.min_value);
        mapped as f32
    }

    /// Generate modulation values.
    pub fn generate(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s = self.tick();
        }
    }
}

// ── Oscillator Bank ─────────────────────────────────────────────

/// A bank of oscillators for polyphonic or additive synthesis.
#[derive(Debug, Clone)]
pub struct OscillatorBank {
    oscillators: Vec<Oscillator>,
    sample_rate: f64,
}

impl OscillatorBank {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            oscillators: Vec::new(),
            sample_rate,
        }
    }

    /// Add an oscillator to the bank.
    pub fn add(&mut self, waveform: Waveform, frequency: f64, amplitude: f64) -> usize {
        let mut osc = Oscillator::new(waveform, frequency, self.sample_rate);
        osc.set_amplitude(amplitude);
        self.oscillators.push(osc);
        self.oscillators.len() - 1
    }

    /// Remove an oscillator by index.
    pub fn remove(&mut self, index: usize) -> bool {
        if index < self.oscillators.len() {
            self.oscillators.remove(index);
            true
        } else {
            false
        }
    }

    pub fn count(&self) -> usize {
        self.oscillators.len()
    }

    /// Set frequency of an oscillator by index.
    pub fn set_frequency(&mut self, index: usize, freq: f64) {
        if let Some(osc) = self.oscillators.get_mut(index) {
            osc.set_frequency(freq);
        }
    }

    /// Set amplitude of an oscillator by index.
    pub fn set_amplitude(&mut self, index: usize, amp: f64) {
        if let Some(osc) = self.oscillators.get_mut(index) {
            osc.set_amplitude(amp);
        }
    }

    /// Generate summed output from all oscillators.
    pub fn generate(&mut self, output: &mut [f32]) {
        // Zero out first
        for s in output.iter_mut() {
            *s = 0.0;
        }
        for osc in &mut self.oscillators {
            osc.generate_add(output);
        }
    }

    /// Reset all oscillators.
    pub fn reset(&mut self) {
        for osc in &mut self.oscillators {
            osc.reset();
        }
    }

    /// Create an additive synthesis bank with harmonics.
    /// Each harmonic has frequency = base_freq * (n+1) and amplitude = base_amp / (n+1).
    pub fn additive(
        base_freq: f64,
        base_amp: f64,
        num_harmonics: usize,
        sample_rate: f64,
    ) -> Self {
        let mut bank = Self::new(sample_rate);
        for n in 0..num_harmonics {
            let harmonic = (n + 1) as f64;
            bank.add(Waveform::Sine, base_freq * harmonic, base_amp / harmonic);
        }
        bank
    }
}

// ── Utility ─────────────────────────────────────────────────────

/// Convert MIDI note number to frequency (A4 = 440 Hz).
pub fn midi_to_freq(note: u8) -> f64 {
    440.0 * 2.0f64.powf((note as f64 - 69.0) / 12.0)
}

/// Convert frequency to nearest MIDI note number.
pub fn freq_to_midi(freq: f64) -> u8 {
    let note = 69.0 + 12.0 * (freq / 440.0).log2();
    note.round().clamp(0.0, 127.0) as u8
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;

    #[test]
    fn sine_oscillator_basic() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        let mut buf = vec![0.0f32; 1024];
        osc.generate(&mut buf);
        // Sine should have values in [-1, 1]
        assert!(buf.iter().all(|s| *s >= -1.001 && *s <= 1.001));
        // Should not be all zeros
        assert!(buf.iter().any(|s| s.abs() > 0.1));
    }

    #[test]
    fn square_wave_values() {
        let mut osc = Oscillator::new(Waveform::Square, 1.0, 100.0);
        let mut buf = vec![0.0f32; 100];
        osc.generate(&mut buf);
        // Square wave should be +1 or -1
        for s in &buf {
            assert!((s.abs() - 1.0).abs() < 0.01, "sample = {s}");
        }
    }

    #[test]
    fn sawtooth_range() {
        let mut osc = Oscillator::new(Waveform::Sawtooth, 1.0, 1000.0);
        let mut buf = vec![0.0f32; 1000];
        osc.generate(&mut buf);
        assert!(buf.iter().all(|s| *s >= -1.001 && *s <= 1.001));
        // Sawtooth should start near -1 and ramp up
        assert!(buf[0] < -0.9);
        assert!(buf[499] > -0.1);
    }

    #[test]
    fn triangle_range() {
        let mut osc = Oscillator::new(Waveform::Triangle, 1.0, 1000.0);
        let mut buf = vec![0.0f32; 1000];
        osc.generate(&mut buf);
        assert!(buf.iter().all(|s| *s >= -1.001 && *s <= 1.001));
    }

    #[test]
    fn amplitude_scaling() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        osc.set_amplitude(0.5);
        let mut buf = vec![0.0f32; 1024];
        osc.generate(&mut buf);
        assert!(buf.iter().all(|s| *s >= -0.501 && *s <= 0.501));
    }

    #[test]
    fn frequency_change() {
        let mut osc = Oscillator::new(Waveform::Sine, 100.0, SR);
        assert!((osc.frequency() - 100.0).abs() < 1e-10);
        osc.set_frequency(200.0);
        assert!((osc.frequency() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn phase_reset() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        osc.tick();
        osc.tick();
        assert!(osc.phase() > 0.0);
        osc.reset();
        assert!((osc.phase()).abs() < 1e-10);
    }

    #[test]
    fn pulse_width() {
        let mut osc = Oscillator::new(Waveform::Square, 1.0, 100.0);
        osc.set_pulse_width(0.25);
        let mut buf = vec![0.0f32; 100];
        osc.generate(&mut buf);
        // About 25% should be +1 and 75% should be -1
        let positive: usize = buf.iter().filter(|s| **s > 0.0).count();
        assert!(positive >= 20 && positive <= 30, "positive count = {positive}");
    }

    #[test]
    fn wavetable_sine() {
        let mut wt = WavetableOscillator::sine(1024, 440.0, SR);
        let mut buf = vec![0.0f32; 512];
        wt.generate(&mut buf);
        assert!(buf.iter().all(|s| *s >= -1.001 && *s <= 1.001));
        assert!(buf.iter().any(|s| s.abs() > 0.5));
    }

    #[test]
    fn wavetable_sawtooth() {
        let mut wt = WavetableOscillator::sawtooth(2048, 100.0, SR);
        let mut buf = vec![0.0f32; 512];
        wt.generate(&mut buf);
        assert!(buf.iter().all(|s| *s >= -1.001 && *s <= 1.001));
    }

    #[test]
    fn wavetable_reset() {
        let mut wt = WavetableOscillator::sine(1024, 440.0, SR);
        let s1 = wt.tick();
        wt.tick();
        wt.reset();
        let s2 = wt.tick();
        assert!((s1 - s2).abs() < 1e-4);
    }

    #[test]
    fn fm_oscillator() {
        let mut fm = FmOscillator::new(440.0, 220.0, 2.0, SR);
        let mut buf = vec![0.0f32; 1024];
        fm.generate(&mut buf);
        // FM should produce non-zero output
        assert!(buf.iter().any(|s| s.abs() > 0.1));
        // FM should be bounded
        assert!(buf.iter().all(|s| *s >= -2.0 && *s <= 2.0));
    }

    #[test]
    fn lfo_range() {
        let mut lfo = Lfo::new(Waveform::Sine, 5.0, SR);
        lfo.set_range(0.0, 1.0);
        let mut buf = vec![0.0f32; 44100];
        lfo.generate(&mut buf);
        // LFO output should be in [0, 1]
        assert!(buf.iter().all(|s| *s >= -0.001 && *s <= 1.001));
        // Should have variation
        let min = buf.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.9);
    }

    #[test]
    fn lfo_custom_range() {
        let mut lfo = Lfo::new(Waveform::Triangle, 1.0, 100.0);
        lfo.set_range(200.0, 800.0);
        let mut buf = vec![0.0f32; 100];
        lfo.generate(&mut buf);
        assert!(buf.iter().all(|s| *s >= 199.0 && *s <= 801.0));
    }

    #[test]
    fn oscillator_bank_basic() {
        let mut bank = OscillatorBank::new(SR);
        bank.add(Waveform::Sine, 440.0, 0.5);
        bank.add(Waveform::Sine, 880.0, 0.25);
        assert_eq!(bank.count(), 2);
        let mut buf = vec![0.0f32; 1024];
        bank.generate(&mut buf);
        assert!(buf.iter().any(|s| s.abs() > 0.1));
    }

    #[test]
    fn oscillator_bank_additive() {
        let mut bank = OscillatorBank::additive(220.0, 1.0, 4, SR);
        assert_eq!(bank.count(), 4);
        let mut buf = vec![0.0f32; 1024];
        bank.generate(&mut buf);
        assert!(buf.iter().any(|s| s.abs() > 0.5));
    }

    #[test]
    fn oscillator_bank_remove() {
        let mut bank = OscillatorBank::new(SR);
        bank.add(Waveform::Sine, 440.0, 1.0);
        bank.add(Waveform::Sine, 880.0, 1.0);
        assert!(bank.remove(0));
        assert_eq!(bank.count(), 1);
        assert!(!bank.remove(5));
    }

    #[test]
    fn midi_to_freq_a4() {
        assert!((midi_to_freq(69) - 440.0).abs() < 0.01);
    }

    #[test]
    fn midi_to_freq_c4() {
        assert!((midi_to_freq(60) - 261.63).abs() < 0.1);
    }

    #[test]
    fn freq_to_midi_roundtrip() {
        for note in [36, 48, 60, 69, 72, 84, 96] {
            let freq = midi_to_freq(note);
            let back = freq_to_midi(freq);
            assert_eq!(back, note, "roundtrip failed for note {note}");
        }
    }

    #[test]
    fn generate_add_accumulates() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, SR);
        let mut buf = vec![1.0f32; 16];
        osc.generate_add(&mut buf);
        // All values should be != 1.0 since sine adds to the existing 1.0
        assert!(buf.iter().any(|s| (*s - 1.0).abs() > 0.01));
    }
}
