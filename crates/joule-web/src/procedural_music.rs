//! Procedural music generation.
//!
//! Markov chain melody generation, Euclidean rhythm generator, bass lines
//! from chord roots, accompaniment patterns, tension/energy curves,
//! section structure, drum pattern generation, and energy-based dynamic
//! mixing. Pure Rust.

use std::collections::HashMap;

// ── Markov Chain Melody Generator ────────────────────────────────

/// Transition table for a Markov chain: from state -> [(to_state, weight)].
#[derive(Debug, Clone)]
pub struct MarkovChain {
    transitions: HashMap<u8, Vec<(u8, f64)>>,
    rng_state: u64,
}

impl MarkovChain {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            rng_state: 42,
        }
    }

    /// Seed the PRNG for deterministic output.
    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    fn next_random(&mut self) -> f64 {
        // xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Add a transition from `from` to `to` with a weight.
    pub fn add_transition(&mut self, from: u8, to: u8, weight: f64) {
        let entry = self.transitions.entry(from).or_insert_with(Vec::new);
        if let Some(existing) = entry.iter_mut().find(|(s, _)| *s == to) {
            existing.1 += weight;
        } else {
            entry.push((to, weight));
        }
    }

    /// Build a default scale-degree transition table (scale degrees 1-7).
    pub fn build_diatonic_defaults(&mut self) {
        // Common melodic movements
        let moves: &[(u8, u8, f64)] = &[
            (1, 2, 3.0), (1, 3, 2.0), (1, 5, 2.0), (1, 7, 1.0),
            (2, 1, 2.0), (2, 3, 3.0), (2, 4, 1.0), (2, 5, 1.0),
            (3, 2, 2.0), (3, 4, 3.0), (3, 1, 2.0), (3, 5, 1.0),
            (4, 3, 2.0), (4, 5, 3.0), (4, 2, 1.0), (4, 6, 1.0),
            (5, 4, 2.0), (5, 6, 2.0), (5, 1, 3.0), (5, 3, 1.0),
            (6, 5, 2.0), (6, 7, 2.0), (6, 4, 1.0), (6, 1, 1.0),
            (7, 1, 4.0), (7, 6, 1.0), (7, 5, 1.0),
        ];
        for &(from, to, w) in moves {
            self.add_transition(from, to, w);
        }
    }

    /// Generate a sequence of scale degrees starting from `start`.
    pub fn generate(&mut self, start: u8, length: usize) -> Vec<u8> {
        let mut sequence = Vec::with_capacity(length);
        let mut current = start;
        sequence.push(current);

        for _ in 1..length {
            current = self.next_state(current);
            sequence.push(current);
        }
        sequence
    }

    fn next_state(&mut self, current: u8) -> u8 {
        let transitions = match self.transitions.get(&current) {
            Some(t) if !t.is_empty() => t.clone(),
            _ => return current,
        };

        let total: f64 = transitions.iter().map(|(_, w)| w).sum();
        if total < 1e-12 { return current; }

        let r = self.next_random() * total;
        let mut cumulative = 0.0;
        for (state, weight) in &transitions {
            cumulative += weight;
            if r < cumulative {
                return *state;
            }
        }
        transitions.last().map(|(s, _)| *s).unwrap_or(current)
    }
}

/// Convert scale degrees to MIDI notes given a root and scale intervals.
pub fn degrees_to_midi(degrees: &[u8], root: u8, octave: u8, scale_intervals: &[u8]) -> Vec<u8> {
    degrees
        .iter()
        .map(|deg| {
            if *deg == 0 || scale_intervals.is_empty() {
                return root + octave * 12;
            }
            let idx = ((deg - 1) as usize) % scale_intervals.len();
            let extra_oct = ((deg - 1) as usize) / scale_intervals.len();
            let midi = (octave as u16 + 1) * 12 + root as u16 + scale_intervals[idx] as u16 + extra_oct as u16 * 12;
            midi.min(127) as u8
        })
        .collect()
}

// ── Euclidean Rhythm Generator ───────────────────────────────────

/// Generate a Euclidean rhythm: distribute K onsets over N steps.
/// Returns a boolean vector where `true` = onset.
pub fn euclidean_rhythm(onsets: usize, steps: usize) -> Vec<bool> {
    if steps == 0 {
        return Vec::new();
    }
    if onsets >= steps {
        return vec![true; steps];
    }
    if onsets == 0 {
        return vec![false; steps];
    }

    // Bjorklund's algorithm
    let mut pattern: Vec<Vec<bool>> = Vec::new();
    let mut remainder: Vec<Vec<bool>> = Vec::new();

    for _ in 0..onsets {
        pattern.push(vec![true]);
    }
    for _ in 0..(steps - onsets) {
        remainder.push(vec![false]);
    }

    loop {
        let min_len = pattern.len().min(remainder.len());
        if remainder.len() <= 1 {
            break;
        }

        let mut new_pattern = Vec::new();
        for i in 0..min_len {
            let mut combined = pattern[i].clone();
            combined.extend_from_slice(&remainder[i]);
            new_pattern.push(combined);
        }

        let extra_pattern: Vec<Vec<bool>> = if pattern.len() > min_len {
            pattern[min_len..].to_vec()
        } else {
            Vec::new()
        };

        let extra_remainder: Vec<Vec<bool>> = if remainder.len() > min_len {
            remainder[min_len..].to_vec()
        } else {
            Vec::new()
        };

        pattern = new_pattern;
        remainder = if !extra_pattern.is_empty() {
            extra_pattern
        } else {
            extra_remainder
        };
    }

    let mut result = Vec::new();
    for p in &pattern {
        result.extend_from_slice(p);
    }
    for r in &remainder {
        result.extend_from_slice(r);
    }
    result.truncate(steps);
    result
}

/// Rotate a rhythm pattern by N steps.
pub fn rotate_rhythm(pattern: &[bool], rotation: usize) -> Vec<bool> {
    if pattern.is_empty() { return Vec::new(); }
    let r = rotation % pattern.len();
    let mut rotated = pattern[r..].to_vec();
    rotated.extend_from_slice(&pattern[..r]);
    rotated
}

// ── Bass Line Generator ──────────────────────────────────────────

/// Generate a bass line from chord roots (pitch classes) with a rhythm pattern.
pub fn bass_line_from_chords(
    chord_roots: &[u8],
    steps_per_chord: usize,
    octave: u8,
    rhythm: &[bool],
) -> Vec<Option<u8>> {
    let mut line = Vec::new();
    for &root in chord_roots {
        let base_note = (octave + 1) * 12 + root;
        for step in 0..steps_per_chord {
            let rhythm_idx = step % rhythm.len().max(1);
            if !rhythm.is_empty() && rhythm[rhythm_idx] {
                line.push(Some(base_note.min(127)));
            } else if rhythm.is_empty() {
                line.push(Some(base_note.min(127)));
            } else {
                line.push(None);
            }
        }
    }
    line
}

/// Generate a walking bass line (root, 5th, approach patterns).
pub fn walking_bass(chord_roots: &[u8], octave: u8) -> Vec<u8> {
    let mut line = Vec::new();
    for (i, &root) in chord_roots.iter().enumerate() {
        let base = (octave + 1) * 12 + root;
        line.push(base.min(127));
        line.push((base + 7).min(127)); // 5th
        line.push((base + 4).min(127)); // 3rd

        // Chromatic approach to next root
        let next_root = if i + 1 < chord_roots.len() {
            (octave + 1) * 12 + chord_roots[i + 1]
        } else {
            (octave + 1) * 12 + chord_roots[0]
        };
        let approach = if next_root > base {
            next_root.saturating_sub(1)
        } else {
            (next_root + 1).min(127)
        };
        line.push(approach);
    }
    line
}

// ── Accompaniment Patterns ───────────────────────────────────────

/// Accompaniment style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccompStyle {
    BlockChord,
    Arpeggiated,
    Strummed,
    Alberti,
}

/// A single accompaniment event.
#[derive(Debug, Clone, PartialEq)]
pub struct AccompEvent {
    pub step: usize,
    pub notes: Vec<u8>,
    pub velocity: u8,
}

/// Generate accompaniment for a chord (given as MIDI notes) in a given style.
pub fn generate_accompaniment(
    chord_notes: &[u8],
    steps: usize,
    style: AccompStyle,
    velocity: u8,
) -> Vec<AccompEvent> {
    let mut events = Vec::new();
    match style {
        AccompStyle::BlockChord => {
            events.push(AccompEvent {
                step: 0,
                notes: chord_notes.to_vec(),
                velocity,
            });
        }
        AccompStyle::Arpeggiated => {
            for (i, &note) in chord_notes.iter().enumerate() {
                if i < steps {
                    events.push(AccompEvent {
                        step: i,
                        notes: vec![note],
                        velocity,
                    });
                }
            }
        }
        AccompStyle::Strummed => {
            for (i, &note) in chord_notes.iter().enumerate() {
                events.push(AccompEvent {
                    step: i, // slight delay per note simulates strum
                    notes: vec![note],
                    velocity: velocity.saturating_sub(i as u8 * 5),
                });
            }
        }
        AccompStyle::Alberti => {
            // Alberti bass: C-E-G-E pattern (bottom, top, middle, top)
            if chord_notes.len() >= 3 {
                let alberti_pattern = [0usize, 2, 1, 2]; // indices into chord
                for step in 0..steps {
                    let idx = alberti_pattern[step % alberti_pattern.len()];
                    let note_idx = idx.min(chord_notes.len() - 1);
                    events.push(AccompEvent {
                        step,
                        notes: vec![chord_notes[note_idx]],
                        velocity,
                    });
                }
            }
        }
    }
    events
}

// ── Tension / Energy Curve ───────────────────────────────────────

/// Energy level for procedural music.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyState {
    pub intensity: f64,  // 0.0 (calm) to 1.0 (maximum)
    pub tension: f64,    // 0.0 (resolved) to 1.0 (maximum dissonance)
    pub density: f64,    // 0.0 (sparse) to 1.0 (dense)
}

impl Default for EnergyState {
    fn default() -> Self {
        Self { intensity: 0.5, tension: 0.3, density: 0.5 }
    }
}

/// Map energy state to musical parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MusicParams {
    pub tempo_bpm: f64,
    pub velocity: u8,
    pub note_density: f64,  // fraction of steps with notes
    pub octave_range: u8,
    pub consonance: f64,    // 0=dissonant, 1=consonant
}

/// Map an energy state to music parameters.
pub fn energy_to_params(energy: &EnergyState) -> MusicParams {
    let tempo = 80.0 + energy.intensity * 80.0; // 80-160 BPM
    let vel = (60.0 + energy.intensity * 67.0).min(127.0) as u8;
    let density = 0.2 + energy.density * 0.6;
    let octaves = 1 + (energy.intensity * 3.0) as u8;
    let consonance = 1.0 - energy.tension;

    MusicParams {
        tempo_bpm: tempo,
        velocity: vel,
        note_density: density,
        octave_range: octaves,
        consonance,
    }
}

// ── Section Structure ────────────────────────────────────────────

/// Musical section type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectionType {
    Intro,
    Verse,
    Chorus,
    Bridge,
    Breakdown,
    Buildup,
    Outro,
}

/// A section with duration and energy.
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub section_type: SectionType,
    pub bars: u32,
    pub energy: EnergyState,
}

/// Generate a standard song structure.
pub fn standard_structure() -> Vec<Section> {
    vec![
        Section { section_type: SectionType::Intro, bars: 4, energy: EnergyState { intensity: 0.3, tension: 0.2, density: 0.3 } },
        Section { section_type: SectionType::Verse, bars: 8, energy: EnergyState { intensity: 0.5, tension: 0.3, density: 0.5 } },
        Section { section_type: SectionType::Chorus, bars: 8, energy: EnergyState { intensity: 0.8, tension: 0.4, density: 0.7 } },
        Section { section_type: SectionType::Verse, bars: 8, energy: EnergyState { intensity: 0.5, tension: 0.3, density: 0.5 } },
        Section { section_type: SectionType::Chorus, bars: 8, energy: EnergyState { intensity: 0.9, tension: 0.5, density: 0.8 } },
        Section { section_type: SectionType::Bridge, bars: 4, energy: EnergyState { intensity: 0.6, tension: 0.6, density: 0.4 } },
        Section { section_type: SectionType::Chorus, bars: 8, energy: EnergyState { intensity: 1.0, tension: 0.4, density: 0.9 } },
        Section { section_type: SectionType::Outro, bars: 4, energy: EnergyState { intensity: 0.2, tension: 0.1, density: 0.2 } },
    ]
}

/// Generate a game-oriented adaptive structure.
pub fn adaptive_structure(energy_curve: &[f64]) -> Vec<Section> {
    energy_curve.iter().enumerate().map(|(i, &e)| {
        let st = if e < 0.2 {
            SectionType::Breakdown
        } else if e < 0.4 {
            SectionType::Verse
        } else if e < 0.7 {
            SectionType::Chorus
        } else {
            SectionType::Buildup
        };
        Section {
            section_type: st,
            bars: 4,
            energy: EnergyState { intensity: e, tension: e * 0.5, density: e * 0.8 },
        }
    }).collect()
}

// ── Drum Pattern Generator ───────────────────────────────────────

/// Drum instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrumPart {
    Kick,
    Snare,
    ClosedHat,
    OpenHat,
    Clap,
    Tom,
    Ride,
    Crash,
}

impl DrumPart {
    /// Standard General MIDI note for this drum.
    pub fn midi_note(&self) -> u8 {
        match self {
            DrumPart::Kick => 36,
            DrumPart::Snare => 38,
            DrumPart::ClosedHat => 42,
            DrumPart::OpenHat => 46,
            DrumPart::Clap => 39,
            DrumPart::Tom => 45,
            DrumPart::Ride => 51,
            DrumPart::Crash => 49,
        }
    }
}

/// A drum pattern: map from DrumPart to step pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct DrumPattern {
    pub steps: u8,
    pub parts: Vec<(DrumPart, Vec<bool>)>,
}

impl DrumPattern {
    pub fn new(steps: u8) -> Self {
        Self { steps, parts: Vec::new() }
    }

    pub fn add_part(&mut self, part: DrumPart, pattern: Vec<bool>) {
        let mut p = pattern;
        p.resize(self.steps as usize, false);
        self.parts.push((part, p));
    }

    /// Generate a basic rock/pop pattern (16 steps).
    pub fn basic_rock() -> Self {
        let mut dp = DrumPattern::new(16);
        // Kick on 1 and 9 (beats 1 and 3)
        dp.add_part(DrumPart::Kick, vec![
            true, false, false, false, false, false, false, false,
            true, false, false, false, false, false, false, false,
        ]);
        // Snare on 5 and 13 (beats 2 and 4)
        dp.add_part(DrumPart::Snare, vec![
            false, false, false, false, true, false, false, false,
            false, false, false, false, true, false, false, false,
        ]);
        // Hi-hat on every even step
        dp.add_part(DrumPart::ClosedHat, vec![
            true, false, true, false, true, false, true, false,
            true, false, true, false, true, false, true, false,
        ]);
        dp
    }

    /// Generate from Euclidean rhythms.
    pub fn euclidean(kick_onsets: usize, snare_onsets: usize, hat_onsets: usize, steps: usize) -> Self {
        let mut dp = DrumPattern::new(steps as u8);
        dp.add_part(DrumPart::Kick, euclidean_rhythm(kick_onsets, steps));
        dp.add_part(DrumPart::Snare, rotate_rhythm(&euclidean_rhythm(snare_onsets, steps), steps / 4));
        dp.add_part(DrumPart::ClosedHat, euclidean_rhythm(hat_onsets, steps));
        dp
    }
}

/// Energy-based drum pattern selection.
pub fn drum_pattern_for_energy(energy: f64, steps: usize) -> DrumPattern {
    if energy < 0.3 {
        // Sparse: minimal kick and hat
        let mut dp = DrumPattern::new(steps as u8);
        dp.add_part(DrumPart::Kick, euclidean_rhythm(2, steps));
        dp.add_part(DrumPart::ClosedHat, euclidean_rhythm(4, steps));
        dp
    } else if energy < 0.6 {
        // Medium: standard pattern
        DrumPattern::basic_rock()
    } else {
        // High: dense with fills
        let mut dp = DrumPattern::new(steps as u8);
        dp.add_part(DrumPart::Kick, euclidean_rhythm(6, steps));
        dp.add_part(DrumPart::Snare, euclidean_rhythm(4, steps));
        dp.add_part(DrumPart::ClosedHat, euclidean_rhythm(12, steps));
        dp.add_part(DrumPart::OpenHat, euclidean_rhythm(2, steps));
        dp
    }
}

// ── Dynamic Mixing ───────────────────────────────────────────────

/// Mix level for a track.
#[derive(Debug, Clone, PartialEq)]
pub struct MixLevel {
    pub name: String,
    pub volume: f64,   // 0.0 to 1.0
    pub active: bool,
}

/// Generate mix levels based on energy state.
pub fn energy_mix(energy: &EnergyState) -> Vec<MixLevel> {
    vec![
        MixLevel {
            name: "drums".into(),
            volume: 0.5 + energy.intensity * 0.5,
            active: energy.intensity > 0.1,
        },
        MixLevel {
            name: "bass".into(),
            volume: 0.6 + energy.intensity * 0.3,
            active: energy.intensity > 0.15,
        },
        MixLevel {
            name: "melody".into(),
            volume: 0.4 + energy.density * 0.4,
            active: energy.density > 0.2,
        },
        MixLevel {
            name: "pad".into(),
            volume: 0.3 + (1.0 - energy.intensity) * 0.4,
            active: true,
        },
        MixLevel {
            name: "fx".into(),
            volume: energy.tension * 0.6,
            active: energy.tension > 0.3,
        },
    ]
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_generate_deterministic() {
        let mut mc = MarkovChain::new();
        mc.build_diatonic_defaults();
        mc.seed(123);
        let seq1 = mc.generate(1, 8);

        let mut mc2 = MarkovChain::new();
        mc2.build_diatonic_defaults();
        mc2.seed(123);
        let seq2 = mc2.generate(1, 8);

        assert_eq!(seq1, seq2);
    }

    #[test]
    fn test_markov_stays_in_range() {
        let mut mc = MarkovChain::new();
        mc.build_diatonic_defaults();
        let seq = mc.generate(1, 100);
        for &deg in &seq {
            assert!(deg >= 1 && deg <= 7);
        }
    }

    #[test]
    fn test_euclidean_4_16() {
        let r = euclidean_rhythm(4, 16);
        assert_eq!(r.len(), 16);
        let onsets: usize = r.iter().filter(|&&b| b).count();
        assert_eq!(onsets, 4);
    }

    #[test]
    fn test_euclidean_3_8() {
        let r = euclidean_rhythm(3, 8);
        assert_eq!(r.len(), 8);
        let onsets: usize = r.iter().filter(|&&b| b).count();
        assert_eq!(onsets, 3);
        // Classic tresillo: [true, false, false, true, false, false, true, false]
        assert!(r[0]);
    }

    #[test]
    fn test_euclidean_0_8() {
        let r = euclidean_rhythm(0, 8);
        assert!(r.iter().all(|b| !b));
    }

    #[test]
    fn test_euclidean_8_8() {
        let r = euclidean_rhythm(8, 8);
        assert!(r.iter().all(|b| *b));
    }

    #[test]
    fn test_euclidean_empty() {
        let r = euclidean_rhythm(3, 0);
        assert!(r.is_empty());
    }

    #[test]
    fn test_rotate_rhythm() {
        let pattern = vec![true, false, false, true];
        let rotated = rotate_rhythm(&pattern, 1);
        assert_eq!(rotated, vec![false, false, true, true]);
    }

    #[test]
    fn test_bass_line_from_chords() {
        let roots = vec![0, 5, 7, 0]; // C, F, G, C
        let rhythm = vec![true, false, true, false];
        let line = bass_line_from_chords(&roots, 4, 2, &rhythm);
        assert_eq!(line.len(), 16);
        assert_eq!(line[0], Some(36)); // C2
        assert_eq!(line[1], None);     // rest
    }

    #[test]
    fn test_walking_bass() {
        let roots = vec![0, 5]; // C, F
        let line = walking_bass(&roots, 2);
        assert_eq!(line.len(), 8); // 4 notes per chord * 2 chords
        assert_eq!(line[0], 36); // C2 root
    }

    #[test]
    fn test_accomp_block_chord() {
        let chord = vec![60, 64, 67];
        let events = generate_accompaniment(&chord, 4, AccompStyle::BlockChord, 100);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].notes.len(), 3);
    }

    #[test]
    fn test_accomp_arpeggiated() {
        let chord = vec![60, 64, 67];
        let events = generate_accompaniment(&chord, 4, AccompStyle::Arpeggiated, 100);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].step, 0);
        assert_eq!(events[1].step, 1);
    }

    #[test]
    fn test_accomp_alberti() {
        let chord = vec![60, 64, 67]; // C, E, G
        let events = generate_accompaniment(&chord, 8, AccompStyle::Alberti, 100);
        assert_eq!(events.len(), 8);
        // Alberti: C, G, E, G, C, G, E, G
        assert_eq!(events[0].notes, vec![60]); // bottom (C)
        assert_eq!(events[1].notes, vec![67]); // top (G)
        assert_eq!(events[2].notes, vec![64]); // middle (E)
        assert_eq!(events[3].notes, vec![67]); // top (G)
    }

    #[test]
    fn test_energy_to_params() {
        let calm = EnergyState { intensity: 0.0, tension: 0.0, density: 0.0 };
        let p = energy_to_params(&calm);
        assert!((p.tempo_bpm - 80.0).abs() < 1e-6);
        assert!(p.velocity <= 70);
        assert!((p.consonance - 1.0).abs() < 1e-6);

        let intense = EnergyState { intensity: 1.0, tension: 1.0, density: 1.0 };
        let p2 = energy_to_params(&intense);
        assert!((p2.tempo_bpm - 160.0).abs() < 1e-6);
        assert!(p2.velocity >= 120);
    }

    #[test]
    fn test_standard_structure() {
        let structure = standard_structure();
        assert!(structure.len() >= 6);
        assert_eq!(structure[0].section_type, SectionType::Intro);
        assert_eq!(structure.last().unwrap().section_type, SectionType::Outro);
    }

    #[test]
    fn test_adaptive_structure() {
        let curve = vec![0.1, 0.3, 0.5, 0.8, 0.9, 0.5, 0.2];
        let structure = adaptive_structure(&curve);
        assert_eq!(structure.len(), 7);
        assert_eq!(structure[0].section_type, SectionType::Breakdown); // low energy
        assert_eq!(structure[3].section_type, SectionType::Buildup);   // high energy
    }

    #[test]
    fn test_drum_basic_rock() {
        let dp = DrumPattern::basic_rock();
        assert_eq!(dp.steps, 16);
        assert_eq!(dp.parts.len(), 3);
    }

    #[test]
    fn test_drum_euclidean() {
        let dp = DrumPattern::euclidean(4, 2, 8, 16);
        assert_eq!(dp.parts.len(), 3);
        for (_, pattern) in &dp.parts {
            assert_eq!(pattern.len(), 16);
        }
    }

    #[test]
    fn test_drum_pattern_for_energy() {
        let low = drum_pattern_for_energy(0.1, 16);
        let high = drum_pattern_for_energy(0.8, 16);
        assert!(low.parts.len() <= high.parts.len());
    }

    #[test]
    fn test_energy_mix() {
        let energy = EnergyState { intensity: 0.8, tension: 0.5, density: 0.7 };
        let mix = energy_mix(&energy);
        assert_eq!(mix.len(), 5);
        assert!(mix[0].active); // drums active at high energy
    }

    #[test]
    fn test_energy_mix_low() {
        let energy = EnergyState { intensity: 0.05, tension: 0.1, density: 0.1 };
        let mix = energy_mix(&energy);
        assert!(!mix[0].active); // drums inactive at very low energy
    }

    #[test]
    fn test_degrees_to_midi() {
        let major = [0u8, 2, 4, 5, 7, 9, 11];
        let midi = degrees_to_midi(&[1, 3, 5], 0, 4, &major);
        assert_eq!(midi, vec![60, 64, 67]); // C4, E4, G4
    }

    #[test]
    fn test_degrees_to_midi_wrapping() {
        let major = [0u8, 2, 4, 5, 7, 9, 11];
        let midi = degrees_to_midi(&[8], 0, 4, &major); // degree 8 = root+octave
        assert_eq!(midi, vec![72]); // C5
    }

    #[test]
    fn test_drum_part_midi() {
        assert_eq!(DrumPart::Kick.midi_note(), 36);
        assert_eq!(DrumPart::Snare.midi_note(), 38);
        assert_eq!(DrumPart::ClosedHat.midi_note(), 42);
    }

    #[test]
    fn test_euclidean_5_8() {
        let r = euclidean_rhythm(5, 8);
        let onsets: usize = r.iter().filter(|&&b| b).count();
        assert_eq!(onsets, 5);
    }
}
