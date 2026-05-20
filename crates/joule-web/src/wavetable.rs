//! Wavetable synthesis engine with band-limited table generation and morphing.
//!
//! Stores single-cycle waveforms at multiple octaves to prevent aliasing.
//! Supports linear and cubic interpolation, cross-fade morphing between
//! wavetable positions, custom user tables, and multi-wavetable stacking.
//! Pure Rust — no DSP library deps.

use std::f64::consts::{PI, TAU};

// ── Interpolation Mode ──────────────────────────────────────────

/// Sample interpolation method for reading the wavetable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interpolation {
    /// Nearest sample (no interpolation).
    Nearest,
    /// Linear interpolation between two adjacent samples.
    Linear,
    /// Cubic (Hermite) interpolation using four samples.
    Cubic,
}

// ── Single Wavetable ────────────────────────────────────────────

/// A single-cycle waveform stored as a Vec of samples.
#[derive(Debug, Clone)]
pub struct WaveTable {
    /// Waveform data, normalised to -1.0..1.0, one full cycle.
    pub data: Vec<f64>,
}

impl WaveTable {
    /// Create a wavetable from raw sample data.
    pub fn from_samples(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Create a sine wavetable of the given size.
    pub fn sine(size: usize) -> Self {
        let data = (0..size)
            .map(|i| (TAU * i as f64 / size as f64).sin())
            .collect();
        Self { data }
    }

    /// Create a sawtooth wavetable (naive, not band-limited).
    pub fn sawtooth(size: usize) -> Self {
        let data = (0..size)
            .map(|i| 2.0 * i as f64 / size as f64 - 1.0)
            .collect();
        Self { data }
    }

    /// Create a square wavetable (naive).
    pub fn square(size: usize) -> Self {
        let data = (0..size)
            .map(|i| if i < size / 2 { 1.0 } else { -1.0 })
            .collect();
        Self { data }
    }

    /// Create a triangle wavetable.
    pub fn triangle(size: usize) -> Self {
        let data = (0..size)
            .map(|i| {
                let phase = i as f64 / size as f64;
                4.0 * (phase - 0.5).abs() - 1.0
            })
            .collect();
        Self { data }
    }

    /// Generate a band-limited sawtooth using additive synthesis up to `max_harmonic`.
    pub fn band_limited_sawtooth(size: usize, max_harmonic: usize) -> Self {
        let mut data = vec![0.0; size];
        for h in 1..=max_harmonic {
            let sign = if h % 2 == 0 { -1.0 } else { 1.0 };
            for (i, sample) in data.iter_mut().enumerate() {
                *sample += sign * (TAU * h as f64 * i as f64 / size as f64).sin() / h as f64;
            }
        }
        // Normalise.
        let peak = data.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if peak > 0.0 {
            for s in &mut data {
                *s /= peak;
            }
        }
        Self { data }
    }

    /// Generate a band-limited square using additive synthesis (odd harmonics only).
    pub fn band_limited_square(size: usize, max_harmonic: usize) -> Self {
        let mut data = vec![0.0; size];
        let mut h = 1;
        while h <= max_harmonic {
            for (i, sample) in data.iter_mut().enumerate() {
                *sample += (TAU * h as f64 * i as f64 / size as f64).sin() / h as f64;
            }
            h += 2;
        }
        let peak = data.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if peak > 0.0 {
            for s in &mut data {
                *s /= peak;
            }
        }
        Self { data }
    }

    /// Table size.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read a sample at a fractional index using the chosen interpolation.
    pub fn read(&self, index: f64, interp: Interpolation) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let len = self.data.len() as f64;
        let idx = index.rem_euclid(len);

        match interp {
            Interpolation::Nearest => {
                self.data[idx.round() as usize % self.data.len()]
            }
            Interpolation::Linear => {
                let i0 = idx.floor() as usize % self.data.len();
                let i1 = (i0 + 1) % self.data.len();
                let frac = idx - idx.floor();
                self.data[i0] * (1.0 - frac) + self.data[i1] * frac
            }
            Interpolation::Cubic => {
                let i1 = idx.floor() as usize % self.data.len();
                let n = self.data.len();
                let i0 = if i1 == 0 { n - 1 } else { i1 - 1 };
                let i2 = (i1 + 1) % n;
                let i3 = (i1 + 2) % n;
                let frac = idx - idx.floor();
                hermite_interp(self.data[i0], self.data[i1], self.data[i2], self.data[i3], frac)
            }
        }
    }
}

/// Hermite cubic interpolation.
fn hermite_interp(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * t + c2) * t + c1) * t + c0
}

// ── Band-Limited Wavetable Set ──────────────────────────────────

/// A set of wavetables, one per octave, for alias-free playback at any frequency.
#[derive(Debug, Clone)]
pub struct BandLimitedTableSet {
    /// One table per octave. Index 0 = lowest octave (most harmonics).
    tables: Vec<WaveTable>,
    /// Table size (samples per table).
    table_size: usize,
    /// Base frequency of the lowest table.
    base_freq: f64,
    /// Sample rate.
    sample_rate: f64,
}

impl BandLimitedTableSet {
    /// Generate a band-limited sawtooth table set covering all audible octaves.
    pub fn sawtooth(table_size: usize, sample_rate: f64) -> Self {
        let base_freq = 20.0; // lowest fundamental
        let mut tables = Vec::new();
        let mut freq = base_freq;
        while freq < sample_rate / 2.0 {
            let max_harmonic = ((sample_rate / 2.0) / freq).floor() as usize;
            let max_harmonic = max_harmonic.max(1);
            tables.push(WaveTable::band_limited_sawtooth(table_size, max_harmonic));
            freq *= 2.0;
        }
        if tables.is_empty() {
            tables.push(WaveTable::sine(table_size));
        }
        Self { tables, table_size, base_freq, sample_rate }
    }

    /// Generate a band-limited square table set.
    pub fn square(table_size: usize, sample_rate: f64) -> Self {
        let base_freq = 20.0;
        let mut tables = Vec::new();
        let mut freq = base_freq;
        while freq < sample_rate / 2.0 {
            let max_harmonic = ((sample_rate / 2.0) / freq).floor() as usize;
            let max_harmonic = max_harmonic.max(1);
            tables.push(WaveTable::band_limited_square(table_size, max_harmonic));
            freq *= 2.0;
        }
        if tables.is_empty() {
            tables.push(WaveTable::sine(table_size));
        }
        Self { tables, table_size, base_freq, sample_rate }
    }

    /// Number of octave tables.
    pub fn octave_count(&self) -> usize {
        self.tables.len()
    }

    /// Select the appropriate table index for a given playback frequency.
    fn table_index_for_freq(&self, freq: f64) -> usize {
        if freq <= self.base_freq || self.tables.is_empty() {
            return 0;
        }
        let octave = (freq / self.base_freq).log2().floor() as usize;
        octave.min(self.tables.len() - 1)
    }

    /// Read a sample for a given phase (0..1) and frequency, auto-selecting the octave table.
    pub fn read(&self, phase: f64, freq: f64, interp: Interpolation) -> f64 {
        let idx = self.table_index_for_freq(freq);
        let table = &self.tables[idx];
        let index = phase * table.len() as f64;
        table.read(index, interp)
    }
}

// ── Wavetable Oscillator ────────────────────────────────────────

/// A wavetable oscillator that reads from a table set with phase accumulation.
#[derive(Debug, Clone)]
pub struct WavetableOscillator {
    pub frequency: f64,
    pub sample_rate: f64,
    pub interpolation: Interpolation,
    phase: f64,
    pub amplitude: f64,
}

impl WavetableOscillator {
    /// Create a new wavetable oscillator.
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        Self {
            frequency,
            sample_rate,
            interpolation: Interpolation::Linear,
            phase: 0.0,
            amplitude: 1.0,
        }
    }

    /// Current phase (0..1).
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Reset phase.
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }

    /// Generate next sample from a band-limited table set.
    pub fn next_sample(&mut self, tables: &BandLimitedTableSet) -> f64 {
        let sample = tables.read(self.phase, self.frequency, self.interpolation);
        let dt = self.frequency / self.sample_rate;
        self.phase += dt;
        self.phase = self.phase.rem_euclid(1.0);
        sample * self.amplitude
    }

    /// Generate next sample from a single wavetable.
    pub fn next_sample_single(&mut self, table: &WaveTable) -> f64 {
        let index = self.phase * table.len() as f64;
        let sample = table.read(index, self.interpolation);
        let dt = self.frequency / self.sample_rate;
        self.phase += dt;
        self.phase = self.phase.rem_euclid(1.0);
        sample * self.amplitude
    }
}

// ── Wavetable Morph ─────────────────────────────────────────────

/// Morphs (crossfades) between multiple wavetables based on a position parameter.
#[derive(Debug, Clone)]
pub struct WavetableMorph {
    tables: Vec<WaveTable>,
    /// Morph position: 0.0 = first table, 1.0 = last table.
    pub position: f64,
}

impl WavetableMorph {
    /// Create a morph set from a list of wavetables.
    pub fn new(tables: Vec<WaveTable>) -> Self {
        Self { tables, position: 0.0 }
    }

    /// Number of tables in the morph set.
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    /// Read a sample at the given phase (0..1), interpolating between tables
    /// according to the current morph position.
    pub fn read(&self, phase: f64, interp: Interpolation) -> f64 {
        if self.tables.is_empty() {
            return 0.0;
        }
        if self.tables.len() == 1 {
            let idx = phase * self.tables[0].len() as f64;
            return self.tables[0].read(idx, interp);
        }
        let pos = self.position.clamp(0.0, 1.0);
        let scaled = pos * (self.tables.len() - 1) as f64;
        let lo = scaled.floor() as usize;
        let hi = (lo + 1).min(self.tables.len() - 1);
        let frac = scaled - lo as f64;

        let idx_lo = phase * self.tables[lo].len() as f64;
        let idx_hi = phase * self.tables[hi].len() as f64;
        let s_lo = self.tables[lo].read(idx_lo, interp);
        let s_hi = self.tables[hi].read(idx_hi, interp);
        s_lo * (1.0 - frac) + s_hi * frac
    }
}

// ── Wavetable Scanner ───────────────────────────────────────────

/// Scans a morph position over time using a phase accumulator (LFO-style).
#[derive(Debug, Clone)]
pub struct WavetableScanner {
    /// Scan rate in Hz (how fast the morph position oscillates).
    pub rate_hz: f64,
    /// Sample rate.
    pub sample_rate: f64,
    /// Internal phase.
    phase: f64,
    /// Scan depth: 0.0 = no scanning, 1.0 = full range.
    pub depth: f64,
    /// Center position (0..1).
    pub center: f64,
}

impl WavetableScanner {
    /// Create a new scanner.
    pub fn new(rate_hz: f64, sample_rate: f64) -> Self {
        Self { rate_hz, sample_rate, phase: 0.0, depth: 1.0, center: 0.5 }
    }

    /// Advance and return the current morph position (0..1).
    pub fn next_position(&mut self) -> f64 {
        let lfo = (self.phase * TAU).sin() * 0.5 + 0.5; // 0..1
        let pos = self.center + (lfo - 0.5) * self.depth;
        self.phase += self.rate_hz / self.sample_rate;
        self.phase = self.phase.rem_euclid(1.0);
        pos.clamp(0.0, 1.0)
    }

    /// Reset scanner phase.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

// ── Multi-Wavetable Stack ───────────────────────────────────────

/// Stack multiple wavetable oscillators and sum their output.
#[derive(Debug, Clone)]
pub struct WavetableStack {
    oscillators: Vec<WavetableOscillator>,
    /// Gain per oscillator.
    gains: Vec<f64>,
}

impl WavetableStack {
    /// Create a new empty stack.
    pub fn new() -> Self {
        Self { oscillators: Vec::new(), gains: Vec::new() }
    }

    /// Add an oscillator with a gain.
    pub fn add(&mut self, osc: WavetableOscillator, gain: f64) {
        self.oscillators.push(osc);
        self.gains.push(gain);
    }

    /// Number of oscillators in the stack.
    pub fn count(&self) -> usize {
        self.oscillators.len()
    }

    /// Generate the next mixed sample from a single shared table.
    pub fn next_sample(&mut self, table: &WaveTable) -> f64 {
        let mut sum = 0.0;
        for (osc, gain) in self.oscillators.iter_mut().zip(self.gains.iter()) {
            sum += osc.next_sample_single(table) * gain;
        }
        sum
    }

    /// Generate the next mixed sample from a band-limited table set.
    pub fn next_sample_band_limited(&mut self, tables: &BandLimitedTableSet) -> f64 {
        let mut sum = 0.0;
        for (osc, gain) in self.oscillators.iter_mut().zip(self.gains.iter()) {
            sum += osc.next_sample(tables) * gain;
        }
        sum
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f64 = 1e-4;
    const TABLE_SIZE: usize = 2048;

    #[test]
    fn test_sine_table_at_zero() {
        let table = WaveTable::sine(TABLE_SIZE);
        let val = table.read(0.0, Interpolation::Linear);
        assert!((val - 0.0).abs() < EPS, "sine at 0 should be ~0, got {val}");
    }

    #[test]
    fn test_sine_table_at_quarter() {
        let table = WaveTable::sine(TABLE_SIZE);
        let val = table.read(TABLE_SIZE as f64 / 4.0, Interpolation::Linear);
        assert!((val - 1.0).abs() < 0.01, "sine at pi/2 should be ~1, got {val}");
    }

    #[test]
    fn test_sawtooth_table_endpoints() {
        let table = WaveTable::sawtooth(TABLE_SIZE);
        let start = table.read(0.0, Interpolation::Nearest);
        assert!((start - (-1.0)).abs() < 0.01, "saw start should be ~-1, got {start}");
    }

    #[test]
    fn test_square_table_values() {
        let table = WaveTable::square(TABLE_SIZE);
        let first_quarter = table.read(TABLE_SIZE as f64 / 4.0, Interpolation::Nearest);
        assert!((first_quarter - 1.0).abs() < EPS, "square first half should be 1.0");
        let third_quarter = table.read(3.0 * TABLE_SIZE as f64 / 4.0, Interpolation::Nearest);
        assert!((third_quarter - (-1.0)).abs() < EPS, "square second half should be -1.0");
    }

    #[test]
    fn test_triangle_table_peak() {
        let table = WaveTable::triangle(TABLE_SIZE);
        let mid = table.read(0.0, Interpolation::Nearest);
        // At phase 0, triangle = 4*|0 - 0.5| - 1 = 1.0
        assert!((mid - 1.0).abs() < 0.01, "triangle at phase 0 should be ~1, got {mid}");
    }

    #[test]
    fn test_linear_interpolation() {
        let table = WaveTable::from_samples(vec![0.0, 1.0, 0.0, -1.0]);
        let val = table.read(0.5, Interpolation::Linear);
        assert!((val - 0.5).abs() < EPS, "linear interp at 0.5 should be 0.5, got {val}");
    }

    #[test]
    fn test_cubic_interpolation_smooth() {
        let table = WaveTable::sine(TABLE_SIZE);
        // Cubic should be very close to actual sine at a non-sample point.
        let fractional_idx = TABLE_SIZE as f64 / 8.0 + 0.3;
        let cubic_val = table.read(fractional_idx, Interpolation::Cubic);
        let linear_val = table.read(fractional_idx, Interpolation::Linear);
        // Both should be close; cubic is generally more accurate.
        assert!((cubic_val - linear_val).abs() < 0.1, "cubic and linear should be similar for sine");
    }

    #[test]
    fn test_band_limited_sawtooth_harmonics() {
        let table = WaveTable::band_limited_sawtooth(TABLE_SIZE, 10);
        assert_eq!(table.len(), TABLE_SIZE);
        // Should be normalised to ~1.
        let peak = table.data.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!((peak - 1.0).abs() < 0.01, "should be normalised, peak={peak}");
    }

    #[test]
    fn test_band_limited_square_odd_harmonics() {
        let table = WaveTable::band_limited_square(TABLE_SIZE, 15);
        assert_eq!(table.len(), TABLE_SIZE);
        let peak = table.data.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!((peak - 1.0).abs() < 0.01, "should be normalised, peak={peak}");
    }

    #[test]
    fn test_band_limited_table_set_octaves() {
        let set = BandLimitedTableSet::sawtooth(TABLE_SIZE, SR);
        // Should have multiple octave tables.
        assert!(set.octave_count() > 5, "should cover many octaves, got {}", set.octave_count());
    }

    #[test]
    fn test_table_set_read_low_freq() {
        let set = BandLimitedTableSet::sawtooth(TABLE_SIZE, SR);
        let val = set.read(0.25, 100.0, Interpolation::Linear);
        assert!(val.is_finite(), "should produce finite sample");
    }

    #[test]
    fn test_table_set_read_high_freq() {
        let set = BandLimitedTableSet::sawtooth(TABLE_SIZE, SR);
        let val = set.read(0.25, 10000.0, Interpolation::Linear);
        assert!(val.is_finite(), "should produce finite sample at high freq");
    }

    #[test]
    fn test_wavetable_oscillator_generates_output() {
        let set = BandLimitedTableSet::sawtooth(TABLE_SIZE, SR);
        let mut osc = WavetableOscillator::new(440.0, SR);
        let mut buf: Vec<f64> = Vec::new();
        for _ in 0..1024 {
            buf.push(osc.next_sample(&set));
        }
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.01);
        assert!(any_nonzero, "oscillator should produce nonzero output");
    }

    #[test]
    fn test_wavetable_oscillator_phase_wraps() {
        let table = WaveTable::sine(TABLE_SIZE);
        let mut osc = WavetableOscillator::new(440.0, SR);
        for _ in 0..1000 {
            osc.next_sample_single(&table);
        }
        assert!(osc.phase() >= 0.0 && osc.phase() < 1.0);
    }

    #[test]
    fn test_morph_single_table() {
        let morph = WavetableMorph::new(vec![WaveTable::sine(TABLE_SIZE)]);
        let val = morph.read(0.25, Interpolation::Linear);
        let expected = (TAU * 0.25).sin();
        assert!((val - expected).abs() < 0.02, "single table morph should match sine");
    }

    #[test]
    fn test_morph_crossfade() {
        let tables = vec![
            WaveTable::from_samples(vec![1.0; 4]),
            WaveTable::from_samples(vec![-1.0; 4]),
        ];
        let mut morph = WavetableMorph::new(tables);
        morph.position = 0.5;
        let val = morph.read(0.0, Interpolation::Nearest);
        assert!((val - 0.0).abs() < EPS, "50% morph between +1 and -1 should be 0, got {val}");
    }

    #[test]
    fn test_morph_empty() {
        let morph = WavetableMorph::new(Vec::new());
        let val = morph.read(0.0, Interpolation::Linear);
        assert!((val - 0.0).abs() < EPS);
    }

    #[test]
    fn test_scanner_produces_range() {
        let mut scanner = WavetableScanner::new(1.0, SR);
        let mut min_pos = 1.0_f64;
        let mut max_pos = 0.0_f64;
        for _ in 0..(SR as usize) {
            let p = scanner.next_position();
            if p < min_pos { min_pos = p; }
            if p > max_pos { max_pos = p; }
        }
        assert!(max_pos - min_pos > 0.5, "scanner should sweep a range: min={min_pos} max={max_pos}");
    }

    #[test]
    fn test_scanner_reset() {
        let mut scanner = WavetableScanner::new(1.0, SR);
        for _ in 0..1000 {
            scanner.next_position();
        }
        scanner.reset();
        // Phase should be 0, first output near center.
        let pos = scanner.next_position();
        assert!((pos - 0.5).abs() < 0.1, "after reset, should start near center: {pos}");
    }

    #[test]
    fn test_wavetable_stack() {
        let table = WaveTable::sine(TABLE_SIZE);
        let mut stack = WavetableStack::new();
        stack.add(WavetableOscillator::new(440.0, SR), 0.5);
        stack.add(WavetableOscillator::new(880.0, SR), 0.5);
        assert_eq!(stack.count(), 2);
        let mut buf = Vec::new();
        for _ in 0..256 {
            buf.push(stack.next_sample(&table));
        }
        let any_nonzero = buf.iter().any(|s| s.abs() > 0.01);
        assert!(any_nonzero, "stack should produce nonzero output");
    }

    #[test]
    fn test_empty_table_read() {
        let table = WaveTable::from_samples(Vec::new());
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        let val = table.read(0.0, Interpolation::Linear);
        assert!((val - 0.0).abs() < EPS);
    }

    #[test]
    fn test_wrap_around_read() {
        let table = WaveTable::from_samples(vec![0.0, 0.5, 1.0, 0.5]);
        // Read past the end should wrap.
        let val = table.read(4.5, Interpolation::Linear);
        // index 4.5 mod 4 = 0.5 → interp between data[0]=0.0 and data[1]=0.5 → 0.25
        assert!((val - 0.25).abs() < EPS, "wrap-around should work, got {val}");
    }

    #[test]
    fn test_hermite_interp_straight_line() {
        // Hermite through a straight line y=x should be linear.
        let val = hermite_interp(0.0, 1.0, 2.0, 3.0, 0.5);
        assert!((val - 1.5).abs() < EPS, "hermite on line should be 1.5, got {val}");
    }

    #[test]
    fn test_oscillator_amplitude() {
        let table = WaveTable::sine(TABLE_SIZE);
        let mut osc = WavetableOscillator::new(440.0, SR);
        osc.amplitude = 0.25;
        let mut peak = 0.0_f64;
        for _ in 0..1000 {
            let s = osc.next_sample_single(&table).abs();
            if s > peak { peak = s; }
        }
        assert!(peak <= 0.26, "amplitude should limit peak to ~0.25, got {peak}");
    }
}
