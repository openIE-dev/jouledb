//! Arpeggiator engine.
//!
//! Takes held notes and plays them in configurable patterns — up, down,
//! up-down, random, played order. Supports octave range, rate, gate length,
//! latch mode, chord mode, tie, and rhythm patterns with accents and rests.
//! Pure Rust — no audio runtime dependency.

use std::collections::BTreeSet;

// ── Arpeggiator Types ────────────────────────────────────────────

/// Arpeggiator play direction / mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArpMode {
    Up,
    Down,
    UpDown,
    DownUp,
    Random,
    PlayedOrder,
}

/// Note rate / subdivision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArpRate {
    Quarter,
    Eighth,
    Sixteenth,
    DottedQuarter,
    DottedEighth,
    TripletQuarter,
    TripletEighth,
    TripletSixteenth,
}

impl ArpRate {
    /// Ticks per note at the given PPQ.
    pub fn ticks(&self, ppq: u32) -> u32 {
        match self {
            ArpRate::Quarter => ppq,
            ArpRate::Eighth => ppq / 2,
            ArpRate::Sixteenth => ppq / 4,
            ArpRate::DottedQuarter => ppq * 3 / 2,
            ArpRate::DottedEighth => ppq * 3 / 4,
            ArpRate::TripletQuarter => ppq * 2 / 3,
            ArpRate::TripletEighth => ppq / 3,
            ArpRate::TripletSixteenth => ppq / 6,
        }
    }
}

/// Rhythm pattern step: normal, accent, rest, or tie.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RhythmStep {
    Normal,
    Accent,
    Rest,
    Tie,
}

/// Output note from the arpeggiator.
#[derive(Debug, Clone, PartialEq)]
pub struct ArpNote {
    pub note: u8,
    pub velocity: u8,
    pub gate_ticks: u32,
    pub is_tie: bool,
}

/// Arpeggiator configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ArpConfig {
    pub mode: ArpMode,
    pub rate: ArpRate,
    pub octave_range: u8,    // 1-4
    pub gate_percent: f64,   // 0.0..=1.0
    pub latch: bool,
    pub chord_mode: bool,
    pub base_velocity: u8,
    pub accent_velocity: u8,
    pub rhythm_pattern: Vec<RhythmStep>,
}

impl Default for ArpConfig {
    fn default() -> Self {
        Self {
            mode: ArpMode::Up,
            rate: ArpRate::Eighth,
            octave_range: 1,
            gate_percent: 0.5,
            latch: false,
            chord_mode: false,
            base_velocity: 100,
            accent_velocity: 127,
            rhythm_pattern: Vec::new(),
        }
    }
}

// ── Arpeggiator State ────────────────────────────────────────────

/// Arpeggiator engine with held-note tracking and step position.
#[derive(Debug, Clone)]
pub struct Arpeggiator {
    pub config: ArpConfig,
    held_notes: BTreeSet<u8>,
    played_order: Vec<u8>,
    latched_notes: BTreeSet<u8>,
    step_index: usize,
    direction_ascending: bool,
    rng_state: u64,
}

impl Arpeggiator {
    pub fn new(config: ArpConfig) -> Self {
        Self {
            config,
            held_notes: BTreeSet::new(),
            played_order: Vec::new(),
            latched_notes: BTreeSet::new(),
            step_index: 0,
            direction_ascending: true,
            rng_state: 12345,
        }
    }

    /// Seed the internal PRNG for deterministic random mode.
    pub fn seed_random(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    fn next_random(&mut self) -> u64 {
        // xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    /// Press a note (add to held notes).
    pub fn note_on(&mut self, note: u8) {
        self.held_notes.insert(note);
        if !self.played_order.contains(&note) {
            self.played_order.push(note);
        }
        if self.config.latch {
            self.latched_notes.insert(note);
        }
    }

    /// Release a note.
    pub fn note_off(&mut self, note: u8) {
        self.held_notes.remove(&note);
        self.played_order.retain(|n| *n != note);
        if !self.config.latch {
            // If not latching, clear if no notes held
            if self.held_notes.is_empty() {
                self.reset();
            }
        }
    }

    /// Get the active note set (held or latched).
    pub fn active_notes(&self) -> Vec<u8> {
        if self.config.latch && self.held_notes.is_empty() && !self.latched_notes.is_empty() {
            self.latched_notes.iter().copied().collect()
        } else if self.held_notes.is_empty() {
            Vec::new()
        } else {
            self.held_notes.iter().copied().collect()
        }
    }

    /// Build the full note sequence for the current held notes + octave range.
    pub fn build_sequence(&self) -> Vec<u8> {
        let base_notes = self.active_notes();
        if base_notes.is_empty() {
            return Vec::new();
        }

        let ordered = match self.config.mode {
            ArpMode::PlayedOrder => {
                // Use played order, filter to active
                let active: BTreeSet<u8> = base_notes.iter().copied().collect();
                self.played_order
                    .iter()
                    .filter(|n| active.contains(n))
                    .copied()
                    .collect::<Vec<_>>()
            }
            _ => base_notes.clone(),
        };

        let mut sequence = Vec::new();
        for octave in 0..self.config.octave_range {
            for &note in &ordered {
                let shifted = note as u16 + octave as u16 * 12;
                if shifted <= 127 {
                    sequence.push(shifted as u8);
                }
            }
        }
        sequence
    }

    /// Get the next note(s) from the arpeggiator. Call once per step.
    pub fn next_step(&mut self, ppq: u32) -> Vec<ArpNote> {
        let sequence = self.build_sequence();
        if sequence.is_empty() {
            return Vec::new();
        }

        let step_ticks = self.config.rate.ticks(ppq);
        let gate_ticks = (step_ticks as f64 * self.config.gate_percent) as u32;

        // Check rhythm pattern
        let rhythm = if !self.config.rhythm_pattern.is_empty() {
            let idx = self.step_index % self.config.rhythm_pattern.len();
            self.config.rhythm_pattern[idx]
        } else {
            RhythmStep::Normal
        };

        let velocity = match rhythm {
            RhythmStep::Accent => self.config.accent_velocity,
            RhythmStep::Rest => {
                self.step_index += 1;
                return Vec::new();
            }
            RhythmStep::Tie | RhythmStep::Normal => self.config.base_velocity,
        };

        let is_tie = rhythm == RhythmStep::Tie;

        if self.config.chord_mode {
            // Play all notes simultaneously
            let notes: Vec<ArpNote> = sequence
                .iter()
                .map(|note| ArpNote { note: *note, velocity, gate_ticks, is_tie })
                .collect();
            self.step_index += 1;
            return notes;
        }

        let note_idx = self.compute_note_index(sequence.len());
        let note = sequence[note_idx];
        self.advance_step(sequence.len());

        vec![ArpNote { note, velocity, gate_ticks, is_tie }]
    }

    fn compute_note_index(&mut self, seq_len: usize) -> usize {
        match self.config.mode {
            ArpMode::Up | ArpMode::PlayedOrder => {
                self.step_index % seq_len
            }
            ArpMode::Down => {
                (seq_len - 1) - (self.step_index % seq_len)
            }
            ArpMode::UpDown => {
                if seq_len <= 1 {
                    return 0;
                }
                let cycle = 2 * (seq_len - 1);
                let pos = self.step_index % cycle;
                if pos < seq_len {
                    pos
                } else {
                    cycle - pos
                }
            }
            ArpMode::DownUp => {
                if seq_len <= 1 {
                    return 0;
                }
                let cycle = 2 * (seq_len - 1);
                let pos = self.step_index % cycle;
                if pos < seq_len {
                    (seq_len - 1) - pos
                } else {
                    pos - (seq_len - 1)
                }
            }
            ArpMode::Random => {
                let r = self.next_random();
                (r % seq_len as u64) as usize
            }
        }
    }

    fn advance_step(&mut self, _seq_len: usize) {
        self.step_index += 1;
    }

    /// Reset the arpeggiator state.
    pub fn reset(&mut self) {
        self.step_index = 0;
        self.direction_ascending = true;
        self.latched_notes.clear();
    }

    /// Clear latch.
    pub fn clear_latch(&mut self) {
        self.latched_notes.clear();
        if self.held_notes.is_empty() {
            self.reset();
        }
    }

    /// Number of held notes.
    pub fn held_count(&self) -> usize {
        self.held_notes.len()
    }

    /// Current step index.
    pub fn current_step(&self) -> usize {
        self.step_index
    }
}

/// Generate N steps of arpeggiator output for preview.
pub fn preview_arp(config: &ArpConfig, notes: &[u8], num_steps: usize, ppq: u32) -> Vec<Vec<ArpNote>> {
    let mut arp = Arpeggiator::new(config.clone());
    for &n in notes {
        arp.note_on(n);
    }
    let mut output = Vec::new();
    for _ in 0..num_steps {
        output.push(arp.next_step(ppq));
    }
    output
}

/// Calculate total ticks for N arp steps at given rate and PPQ.
pub fn arp_duration_ticks(rate: ArpRate, ppq: u32, num_steps: usize) -> u64 {
    rate.ticks(ppq) as u64 * num_steps as u64
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn c_major_triad() -> Vec<u8> {
        vec![60, 64, 67] // C4, E4, G4
    }

    #[test]
    fn test_arp_rate_ticks() {
        assert_eq!(ArpRate::Quarter.ticks(480), 480);
        assert_eq!(ArpRate::Eighth.ticks(480), 240);
        assert_eq!(ArpRate::Sixteenth.ticks(480), 120);
        assert_eq!(ArpRate::TripletEighth.ticks(480), 160);
    }

    #[test]
    fn test_note_on_off() {
        let mut arp = Arpeggiator::new(ArpConfig::default());
        arp.note_on(60);
        arp.note_on(64);
        assert_eq!(arp.held_count(), 2);
        arp.note_off(60);
        assert_eq!(arp.held_count(), 1);
    }

    #[test]
    fn test_up_mode() {
        let config = ArpConfig { mode: ArpMode::Up, ..Default::default() };
        let steps = preview_arp(&config, &c_major_triad(), 6, 480);
        let notes: Vec<u8> = steps.iter().map(|s| s[0].note).collect();
        assert_eq!(notes, vec![60, 64, 67, 60, 64, 67]);
    }

    #[test]
    fn test_down_mode() {
        let config = ArpConfig { mode: ArpMode::Down, ..Default::default() };
        let steps = preview_arp(&config, &c_major_triad(), 6, 480);
        let notes: Vec<u8> = steps.iter().map(|s| s[0].note).collect();
        assert_eq!(notes, vec![67, 64, 60, 67, 64, 60]);
    }

    #[test]
    fn test_up_down_mode() {
        let config = ArpConfig { mode: ArpMode::UpDown, ..Default::default() };
        let steps = preview_arp(&config, &c_major_triad(), 8, 480);
        let notes: Vec<u8> = steps.iter().map(|s| s[0].note).collect();
        // Up: 60, 64, 67 then Down: 64, 60 then Up: 64, 67 then Down: 64
        assert_eq!(notes, vec![60, 64, 67, 64, 60, 64, 67, 64]);
    }

    #[test]
    fn test_down_up_mode() {
        let config = ArpConfig { mode: ArpMode::DownUp, ..Default::default() };
        let steps = preview_arp(&config, &c_major_triad(), 6, 480);
        let notes: Vec<u8> = steps.iter().map(|s| s[0].note).collect();
        // Down: 67, 64, 60, Up: 64, 67, Down: 64
        assert_eq!(notes, vec![67, 64, 60, 64, 67, 64]);
    }

    #[test]
    fn test_random_mode_deterministic() {
        let config = ArpConfig { mode: ArpMode::Random, ..Default::default() };
        let steps1 = preview_arp(&config, &c_major_triad(), 10, 480);
        let steps2 = preview_arp(&config, &c_major_triad(), 10, 480);
        // Same seed -> same sequence
        let n1: Vec<u8> = steps1.iter().map(|s| s[0].note).collect();
        let n2: Vec<u8> = steps2.iter().map(|s| s[0].note).collect();
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_octave_range() {
        let config = ArpConfig { mode: ArpMode::Up, octave_range: 2, ..Default::default() };
        let steps = preview_arp(&config, &[60, 64, 67], 6, 480);
        let notes: Vec<u8> = steps.iter().map(|s| s[0].note).collect();
        assert_eq!(notes, vec![60, 64, 67, 72, 76, 79]);
    }

    #[test]
    fn test_gate_length() {
        let config = ArpConfig { gate_percent: 0.75, ..Default::default() };
        let steps = preview_arp(&config, &[60], 1, 480);
        // Eighth note = 240 ticks, gate 75% = 180
        assert_eq!(steps[0][0].gate_ticks, 180);
    }

    #[test]
    fn test_chord_mode() {
        let config = ArpConfig { chord_mode: true, ..Default::default() };
        let steps = preview_arp(&config, &c_major_triad(), 2, 480);
        assert_eq!(steps[0].len(), 3);
        assert_eq!(steps[1].len(), 3);
    }

    #[test]
    fn test_latch_mode() {
        let mut arp = Arpeggiator::new(ArpConfig { latch: true, ..Default::default() });
        arp.note_on(60);
        arp.note_on(64);
        arp.note_off(60);
        arp.note_off(64);
        // Notes should still be latched
        let notes = arp.active_notes();
        assert_eq!(notes.len(), 2);
    }

    #[test]
    fn test_clear_latch() {
        let mut arp = Arpeggiator::new(ArpConfig { latch: true, ..Default::default() });
        arp.note_on(60);
        arp.note_off(60);
        assert!(!arp.active_notes().is_empty());
        arp.clear_latch();
        assert!(arp.active_notes().is_empty());
    }

    #[test]
    fn test_rhythm_pattern_rest() {
        let config = ArpConfig {
            rhythm_pattern: vec![RhythmStep::Normal, RhythmStep::Rest, RhythmStep::Normal],
            ..Default::default()
        };
        let steps = preview_arp(&config, &[60], 3, 480);
        assert_eq!(steps[0].len(), 1);
        assert_eq!(steps[1].len(), 0); // rest
        assert_eq!(steps[2].len(), 1);
    }

    #[test]
    fn test_rhythm_pattern_accent() {
        let config = ArpConfig {
            rhythm_pattern: vec![RhythmStep::Accent, RhythmStep::Normal],
            base_velocity: 80,
            accent_velocity: 120,
            ..Default::default()
        };
        let steps = preview_arp(&config, &[60], 2, 480);
        assert_eq!(steps[0][0].velocity, 120);
        assert_eq!(steps[1][0].velocity, 80);
    }

    #[test]
    fn test_rhythm_pattern_tie() {
        let config = ArpConfig {
            rhythm_pattern: vec![RhythmStep::Normal, RhythmStep::Tie],
            ..Default::default()
        };
        let steps = preview_arp(&config, &[60], 2, 480);
        assert!(!steps[0][0].is_tie);
        assert!(steps[1][0].is_tie);
    }

    #[test]
    fn test_played_order_mode() {
        let mut arp = Arpeggiator::new(ArpConfig { mode: ArpMode::PlayedOrder, ..Default::default() });
        arp.note_on(67); // G first
        arp.note_on(60); // C second
        arp.note_on(64); // E third
        let seq = arp.build_sequence();
        assert_eq!(seq, vec![67, 60, 64]);
    }

    #[test]
    fn test_empty_arp_no_crash() {
        let mut arp = Arpeggiator::new(ArpConfig::default());
        let notes = arp.next_step(480);
        assert!(notes.is_empty());
    }

    #[test]
    fn test_arp_duration_ticks() {
        assert_eq!(arp_duration_ticks(ArpRate::Eighth, 480, 8), 1920);
    }

    #[test]
    fn test_single_note_up_down() {
        // With only one note, up-down should just repeat it
        let config = ArpConfig { mode: ArpMode::UpDown, ..Default::default() };
        let steps = preview_arp(&config, &[60], 4, 480);
        let notes: Vec<u8> = steps.iter().map(|s| s[0].note).collect();
        assert_eq!(notes, vec![60, 60, 60, 60]);
    }

    #[test]
    fn test_octave_range_clipping() {
        // High note + 3 octaves should clip at 127
        let config = ArpConfig { mode: ArpMode::Up, octave_range: 4, ..Default::default() };
        let steps = preview_arp(&config, &[120], 4, 480);
        for s in &steps {
            for n in s {
                assert!(n.note <= 127);
            }
        }
    }

    #[test]
    fn test_dotted_rate() {
        assert_eq!(ArpRate::DottedQuarter.ticks(480), 720);
        assert_eq!(ArpRate::DottedEighth.ticks(480), 360);
    }
}
