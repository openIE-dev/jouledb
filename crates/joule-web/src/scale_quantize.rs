//! Musical scale definitions and note quantization.
//!
//! Covers major, minor, harmonic/melodic minor, pentatonic, blues, modes,
//! chromatic, whole tone, diminished. Quantize MIDI notes to nearest scale
//! degree, transpose within scale, detect key from note histogram, and
//! compute relative major/minor. Pure Rust.

// ── Scale Definitions ────────────────────────────────────────────

/// Named scale types with their interval patterns (semitones from root).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleType {
    Major,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
    PentatonicMajor,
    PentatonicMinor,
    Blues,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Locrian,
    Chromatic,
    WholeTone,
    Diminished,        // half-whole diminished
    DiminishedWhole,   // whole-half diminished
}

impl ScaleType {
    /// Semitone intervals for this scale relative to root.
    pub fn intervals(&self) -> &[u8] {
        match self {
            ScaleType::Major => &[0, 2, 4, 5, 7, 9, 11],
            ScaleType::NaturalMinor => &[0, 2, 3, 5, 7, 8, 10],
            ScaleType::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            ScaleType::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
            ScaleType::PentatonicMajor => &[0, 2, 4, 7, 9],
            ScaleType::PentatonicMinor => &[0, 3, 5, 7, 10],
            ScaleType::Blues => &[0, 3, 5, 6, 7, 10],
            ScaleType::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            ScaleType::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
            ScaleType::Lydian => &[0, 2, 4, 6, 7, 9, 11],
            ScaleType::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            ScaleType::Locrian => &[0, 1, 3, 5, 6, 8, 10],
            ScaleType::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ScaleType::WholeTone => &[0, 2, 4, 6, 8, 10],
            ScaleType::Diminished => &[0, 1, 3, 4, 6, 7, 9, 10],
            ScaleType::DiminishedWhole => &[0, 2, 3, 5, 6, 8, 9, 11],
        }
    }

    /// Number of scale degrees.
    pub fn degree_count(&self) -> usize {
        self.intervals().len()
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            ScaleType::Major => "Major",
            ScaleType::NaturalMinor => "Natural Minor",
            ScaleType::HarmonicMinor => "Harmonic Minor",
            ScaleType::MelodicMinor => "Melodic Minor",
            ScaleType::PentatonicMajor => "Pentatonic Major",
            ScaleType::PentatonicMinor => "Pentatonic Minor",
            ScaleType::Blues => "Blues",
            ScaleType::Dorian => "Dorian",
            ScaleType::Phrygian => "Phrygian",
            ScaleType::Lydian => "Lydian",
            ScaleType::Mixolydian => "Mixolydian",
            ScaleType::Locrian => "Locrian",
            ScaleType::Chromatic => "Chromatic",
            ScaleType::WholeTone => "Whole Tone",
            ScaleType::Diminished => "Diminished (H-W)",
            ScaleType::DiminishedWhole => "Diminished (W-H)",
        }
    }
}

// ── Scale Instance ───────────────────────────────────────────────

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// A concrete scale: root pitch class + scale type.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    pub root: u8,         // 0-11
    pub scale_type: ScaleType,
}

impl Scale {
    pub fn new(root: u8, scale_type: ScaleType) -> Self {
        Self { root: root % 12, scale_type }
    }

    /// Pitch classes in this scale.
    pub fn pitch_classes(&self) -> Vec<u8> {
        self.scale_type
            .intervals()
            .iter()
            .map(|i| (self.root + i) % 12)
            .collect()
    }

    /// Check if a pitch class belongs to this scale.
    pub fn contains(&self, pc: u8) -> bool {
        let relative = (pc % 12 + 12 - self.root) % 12;
        self.scale_type.intervals().contains(&relative)
    }

    /// Get the scale degree (1-based) of a pitch class, or None if not in scale.
    pub fn degree_of(&self, pc: u8) -> Option<usize> {
        let relative = (pc % 12 + 12 - self.root) % 12;
        self.scale_type
            .intervals()
            .iter()
            .position(|i| *i == relative)
            .map(|p| p + 1)
    }

    /// Get the pitch class at a given scale degree (1-based).
    pub fn note_at_degree(&self, degree: usize) -> Option<u8> {
        if degree == 0 { return None; }
        let intervals = self.scale_type.intervals();
        let idx = (degree - 1) % intervals.len();
        let octave_offset = ((degree - 1) / intervals.len()) as u8 * 12;
        Some((self.root + intervals[idx] + octave_offset) % 12)
    }

    /// Get MIDI note at a given degree and octave.
    pub fn midi_note_at_degree(&self, degree: usize, octave: u8) -> Option<u8> {
        if degree == 0 { return None; }
        let intervals = self.scale_type.intervals();
        let idx = (degree - 1) % intervals.len();
        let extra_octaves = ((degree - 1) / intervals.len()) as u8;
        let midi = (octave + 1 + extra_octaves) as u16 * 12 + self.root as u16 + intervals[idx] as u16;
        if midi > 127 { None } else { Some(midi as u8) }
    }

    /// Quantize a MIDI note to the nearest note in this scale.
    pub fn quantize(&self, midi_note: u8) -> u8 {
        let pc = midi_note % 12;
        let octave_base = midi_note - pc;

        if self.contains(pc) {
            return midi_note;
        }

        let intervals = self.scale_type.intervals();
        let mut best = midi_note;
        let mut best_dist = u8::MAX;

        for &interval in intervals {
            let scale_pc = (self.root + interval) % 12;
            // Try same octave
            for offset in [0i16, 12, -12] {
                let candidate = octave_base as i16 + scale_pc as i16 + offset;
                if candidate >= 0 && candidate <= 127 {
                    let dist = (candidate - midi_note as i16).unsigned_abs() as u8;
                    if dist < best_dist {
                        best_dist = dist;
                        best = candidate as u8;
                    }
                }
            }
        }
        best
    }

    /// Transpose a MIDI note by N scale degrees (positive = up, negative = down).
    pub fn transpose_by_degrees(&self, midi_note: u8, degrees: i32) -> u8 {
        let quantized = self.quantize(midi_note);
        let pc = quantized % 12;
        let octave = quantized / 12;
        let intervals = self.scale_type.intervals();

        // Find current degree
        let relative = (pc + 12 - self.root) % 12;
        let current_idx = intervals
            .iter()
            .position(|i| *i == relative)
            .unwrap_or(0) as i32;

        let len = intervals.len() as i32;
        let new_idx = current_idx + degrees;
        let scale_idx = new_idx.rem_euclid(len) as usize;
        let octave_shift = new_idx.div_euclid(len);

        let new_octave = octave as i32 + octave_shift;
        let new_midi = new_octave * 12 + self.root as i32 + intervals[scale_idx] as i32;
        new_midi.clamp(0, 127) as u8
    }

    /// Human-readable name (e.g., "C Major").
    pub fn name(&self) -> String {
        format!("{} {}", NOTE_NAMES[self.root as usize], self.scale_type.name())
    }
}

// ── Key Detection ────────────────────────────────────────────────

/// Detect the most likely major key from a histogram of pitch class counts.
/// histogram[0] = count for C, histogram[1] = count for C#, etc.
pub fn detect_key(histogram: &[u32; 12]) -> (u8, ScaleType) {
    // Krumhansl-Schmuckler key profiles (simplified)
    let major_profile: [f64; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    let minor_profile: [f64; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];

    let total: f64 = histogram.iter().map(|c| *c as f64).sum();
    if total < 1e-9 {
        return (0, ScaleType::Major);
    }

    let norm: Vec<f64> = histogram.iter().map(|c| *c as f64 / total).collect();

    let mut best_key = 0u8;
    let mut best_scale = ScaleType::Major;
    let mut best_corr = f64::NEG_INFINITY;

    for root in 0..12u8 {
        // Major correlation
        let maj_corr = correlation(&norm, &major_profile, root as usize);
        if maj_corr > best_corr {
            best_corr = maj_corr;
            best_key = root;
            best_scale = ScaleType::Major;
        }
        // Minor correlation
        let min_corr = correlation(&norm, &minor_profile, root as usize);
        if min_corr > best_corr {
            best_corr = min_corr;
            best_key = root;
            best_scale = ScaleType::NaturalMinor;
        }
    }

    (best_key, best_scale)
}

fn correlation(input: &[f64], profile: &[f64; 12], offset: usize) -> f64 {
    let mean_i: f64 = input.iter().sum::<f64>() / 12.0;
    let mean_p: f64 = profile.iter().sum::<f64>() / 12.0;

    let mut cov = 0.0;
    let mut var_i = 0.0;
    let mut var_p = 0.0;

    for i in 0..12 {
        let pi = (i + offset) % 12;
        let diff_i = input[pi] - mean_i;
        let diff_p = profile[i] - mean_p;
        cov += diff_i * diff_p;
        var_i += diff_i * diff_i;
        var_p += diff_p * diff_p;
    }

    let denom = (var_i * var_p).sqrt();
    if denom < 1e-12 { 0.0 } else { cov / denom }
}

// ── Relative/Parallel Keys ───────────────────────────────────────

/// Relative minor of a major key (e.g., C major -> A minor).
pub fn relative_minor(major_root: u8) -> u8 {
    (major_root + 9) % 12
}

/// Relative major of a minor key (e.g., A minor -> C major).
pub fn relative_major(minor_root: u8) -> u8 {
    (minor_root + 3) % 12
}

/// Parallel minor of a major key (same root, minor quality).
pub fn parallel_minor(root: u8) -> Scale {
    Scale::new(root, ScaleType::NaturalMinor)
}

/// Parallel major of a minor key (same root, major quality).
pub fn parallel_major(root: u8) -> Scale {
    Scale::new(root, ScaleType::Major)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_major_scale_intervals() {
        assert_eq!(ScaleType::Major.intervals(), &[0, 2, 4, 5, 7, 9, 11]);
        assert_eq!(ScaleType::Major.degree_count(), 7);
    }

    #[test]
    fn test_scale_pitch_classes_c_major() {
        let s = Scale::new(0, ScaleType::Major);
        assert_eq!(s.pitch_classes(), vec![0, 2, 4, 5, 7, 9, 11]);
    }

    #[test]
    fn test_scale_pitch_classes_g_major() {
        let s = Scale::new(7, ScaleType::Major);
        // G, A, B, C, D, E, F#
        assert_eq!(s.pitch_classes(), vec![7, 9, 11, 0, 2, 4, 6]);
    }

    #[test]
    fn test_contains() {
        let s = Scale::new(0, ScaleType::Major);
        assert!(s.contains(0)); // C
        assert!(s.contains(4)); // E
        assert!(!s.contains(1)); // C#
        assert!(!s.contains(6)); // F#
    }

    #[test]
    fn test_degree_of() {
        let s = Scale::new(0, ScaleType::Major);
        assert_eq!(s.degree_of(0), Some(1)); // C = degree 1
        assert_eq!(s.degree_of(4), Some(3)); // E = degree 3
        assert_eq!(s.degree_of(1), None);    // C# not in scale
    }

    #[test]
    fn test_note_at_degree() {
        let s = Scale::new(0, ScaleType::Major);
        assert_eq!(s.note_at_degree(1), Some(0)); // C
        assert_eq!(s.note_at_degree(3), Some(4)); // E
        assert_eq!(s.note_at_degree(5), Some(7)); // G
    }

    #[test]
    fn test_quantize_in_scale() {
        let s = Scale::new(0, ScaleType::Major);
        assert_eq!(s.quantize(60), 60); // C4 already in scale
    }

    #[test]
    fn test_quantize_out_of_scale() {
        let s = Scale::new(0, ScaleType::Major);
        // C#4 (61) -> nearest is C4 (60) or D4 (62), both 1 away. Should pick one.
        let q = s.quantize(61);
        assert!(q == 60 || q == 62);
    }

    #[test]
    fn test_quantize_f_sharp() {
        let s = Scale::new(0, ScaleType::Major);
        // F# (66) -> nearest F (65) or G (67)
        let q = s.quantize(66);
        assert!(q == 65 || q == 67);
    }

    #[test]
    fn test_transpose_up_one_degree() {
        let s = Scale::new(0, ScaleType::Major);
        let result = s.transpose_by_degrees(60, 1); // C4 + 1 degree = D4
        assert_eq!(result, 62);
    }

    #[test]
    fn test_transpose_down_one_degree() {
        let s = Scale::new(0, ScaleType::Major);
        let result = s.transpose_by_degrees(62, -1); // D4 - 1 degree = C4
        assert_eq!(result, 60);
    }

    #[test]
    fn test_transpose_across_octave() {
        let s = Scale::new(0, ScaleType::Major);
        let result = s.transpose_by_degrees(71, 1); // B4 + 1 = C5
        assert_eq!(result, 72);
    }

    #[test]
    fn test_pentatonic_scale() {
        let s = Scale::new(0, ScaleType::PentatonicMajor);
        assert_eq!(s.pitch_classes(), vec![0, 2, 4, 7, 9]);
        assert!(!s.contains(5)); // F not in C pentatonic
    }

    #[test]
    fn test_blues_scale() {
        let s = Scale::new(0, ScaleType::Blues);
        // C, Eb, F, F#, G, Bb
        assert_eq!(s.pitch_classes(), vec![0, 3, 5, 6, 7, 10]);
    }

    #[test]
    fn test_chromatic_contains_all() {
        let s = Scale::new(0, ScaleType::Chromatic);
        for pc in 0..12 {
            assert!(s.contains(pc));
        }
    }

    #[test]
    fn test_detect_key_c_major() {
        let mut hist = [0u32; 12];
        // Heavily weight C major notes
        for &pc in &[0, 2, 4, 5, 7, 9, 11] {
            hist[pc as usize] = 10;
        }
        hist[0] = 20; // extra C
        hist[7] = 15; // extra G
        let (key, scale) = detect_key(&hist);
        assert_eq!(key, 0);
        assert_eq!(scale, ScaleType::Major);
    }

    #[test]
    fn test_detect_key_empty() {
        let hist = [0u32; 12];
        let (key, _scale) = detect_key(&hist);
        assert_eq!(key, 0); // default
    }

    #[test]
    fn test_relative_minor() {
        assert_eq!(relative_minor(0), 9); // C -> Am
        assert_eq!(relative_minor(7), 4); // G -> Em
    }

    #[test]
    fn test_relative_major() {
        assert_eq!(relative_major(9), 0); // Am -> C
        assert_eq!(relative_major(4), 7); // Em -> G
    }

    #[test]
    fn test_parallel_minor() {
        let s = parallel_minor(0);
        assert_eq!(s.root, 0);
        assert_eq!(s.scale_type, ScaleType::NaturalMinor);
    }

    #[test]
    fn test_scale_name() {
        let s = Scale::new(0, ScaleType::Major);
        assert_eq!(s.name(), "C Major");
        let s2 = Scale::new(9, ScaleType::NaturalMinor);
        assert_eq!(s2.name(), "A Natural Minor");
    }

    #[test]
    fn test_midi_note_at_degree() {
        let s = Scale::new(0, ScaleType::Major);
        assert_eq!(s.midi_note_at_degree(1, 4), Some(60)); // C4
        assert_eq!(s.midi_note_at_degree(3, 4), Some(64)); // E4
    }

    #[test]
    fn test_diminished_scale() {
        let s = Scale::new(0, ScaleType::Diminished);
        assert_eq!(s.pitch_classes(), vec![0, 1, 3, 4, 6, 7, 9, 10]);
        assert_eq!(s.scale_type.degree_count(), 8);
    }

    #[test]
    fn test_whole_tone_scale() {
        let s = Scale::new(0, ScaleType::WholeTone);
        assert_eq!(s.pitch_classes(), vec![0, 2, 4, 6, 8, 10]);
    }
}
