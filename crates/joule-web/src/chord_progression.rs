//! Chord analysis and generation.
//!
//! Chord types (major, minor, dim, aug, 7th, sus, add9, power), chord
//! recognition, progression generation in key, Roman numeral analysis,
//! voice leading, inversions, and Nashville number system. Pure Rust.

use std::collections::HashMap;

// ── Note / Pitch Class ───────────────────────────────────────────

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Parse a note name to pitch class (0-11). C=0.
pub fn note_to_pc(name: &str) -> Option<u8> {
    let name = name.trim();
    if name.is_empty() { return None; }
    // Handle flats
    if name.len() >= 2 && name.as_bytes()[1] == b'b' {
        let sharp_equiv = match name.as_bytes()[0] {
            b'D' => "C#", b'E' => "D#", b'G' => "F#",
            b'A' => "G#", b'B' => "A#", b'C' => "B", b'F' => "E",
            _ => return None,
        };
        return note_to_pc(sharp_equiv);
    }
    let key = if name.len() >= 2 && name.as_bytes()[1] == b'#' {
        &name[..2]
    } else {
        &name[..1]
    };
    NOTE_NAMES.iter().position(|n| *n == key).map(|p| p as u8)
}

/// Pitch class to note name.
pub fn pc_to_note(pc: u8) -> &'static str {
    NOTE_NAMES[(pc % 12) as usize]
}

// ── Chord Types ──────────────────────────────────────────────────

/// Supported chord qualities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordType {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    Diminished7,
    HalfDiminished7,
    Sus2,
    Sus4,
    Add9,
    Power,
    MinorMajor7,
}

impl ChordType {
    /// Semitone intervals from root for this chord type.
    pub fn intervals(&self) -> &[u8] {
        match self {
            ChordType::Major => &[0, 4, 7],
            ChordType::Minor => &[0, 3, 7],
            ChordType::Diminished => &[0, 3, 6],
            ChordType::Augmented => &[0, 4, 8],
            ChordType::Dominant7 => &[0, 4, 7, 10],
            ChordType::Major7 => &[0, 4, 7, 11],
            ChordType::Minor7 => &[0, 3, 7, 10],
            ChordType::Diminished7 => &[0, 3, 6, 9],
            ChordType::HalfDiminished7 => &[0, 3, 6, 10],
            ChordType::Sus2 => &[0, 2, 7],
            ChordType::Sus4 => &[0, 5, 7],
            ChordType::Add9 => &[0, 2, 4, 7],
            ChordType::Power => &[0, 7],
            ChordType::MinorMajor7 => &[0, 3, 7, 11],
        }
    }

    /// Short display name.
    pub fn suffix(&self) -> &str {
        match self {
            ChordType::Major => "",
            ChordType::Minor => "m",
            ChordType::Diminished => "dim",
            ChordType::Augmented => "aug",
            ChordType::Dominant7 => "7",
            ChordType::Major7 => "maj7",
            ChordType::Minor7 => "m7",
            ChordType::Diminished7 => "dim7",
            ChordType::HalfDiminished7 => "m7b5",
            ChordType::Sus2 => "sus2",
            ChordType::Sus4 => "sus4",
            ChordType::Add9 => "add9",
            ChordType::Power => "5",
            ChordType::MinorMajor7 => "mMaj7",
        }
    }
}

// ── Chord ────────────────────────────────────────────────────────

/// A chord: root pitch class + quality + optional inversion.
#[derive(Debug, Clone, PartialEq)]
pub struct Chord {
    pub root: u8,       // pitch class 0-11
    pub chord_type: ChordType,
    pub inversion: u8,  // 0 = root, 1 = first, 2 = second, etc.
}

impl Chord {
    pub fn new(root: u8, chord_type: ChordType) -> Self {
        Self { root: root % 12, chord_type, inversion: 0 }
    }

    pub fn with_inversion(root: u8, chord_type: ChordType, inversion: u8) -> Self {
        let max_inv = chord_type.intervals().len().saturating_sub(1) as u8;
        Self { root: root % 12, chord_type, inversion: inversion.min(max_inv) }
    }

    /// Get the pitch classes of the chord notes.
    pub fn pitch_classes(&self) -> Vec<u8> {
        self.chord_type
            .intervals()
            .iter()
            .map(|i| (self.root + i) % 12)
            .collect()
    }

    /// Get MIDI note numbers for a given octave, with inversion applied.
    pub fn midi_notes(&self, octave: u8) -> Vec<u8> {
        let base = (octave + 1) * 12 + self.root;
        let intervals = self.chord_type.intervals();
        let mut notes: Vec<u8> = intervals.iter().map(|i| base + i).collect();

        // Apply inversion: move bottom N notes up an octave
        for i in 0..self.inversion as usize {
            if i < notes.len() {
                notes[i] += 12;
            }
        }
        notes.sort();
        notes
    }

    /// Display name (e.g., "Cm7").
    pub fn name(&self) -> String {
        let inv_str = match self.inversion {
            0 => String::new(),
            n => {
                let bass_pc = self.pitch_classes()[n as usize % self.pitch_classes().len()];
                format!("/{}", pc_to_note(bass_pc))
            }
        };
        format!("{}{}{}", pc_to_note(self.root), self.chord_type.suffix(), inv_str)
    }
}

// ── Chord Recognition ────────────────────────────────────────────

/// Recognize a chord from a set of pitch classes.
pub fn recognize_chord(pitch_classes: &[u8]) -> Option<(u8, ChordType)> {
    let mut pcs: Vec<u8> = pitch_classes.iter().map(|p| p % 12).collect();
    pcs.sort();
    pcs.dedup();

    if pcs.is_empty() {
        return None;
    }

    let all_types = [
        ChordType::Major, ChordType::Minor, ChordType::Diminished,
        ChordType::Augmented, ChordType::Dominant7, ChordType::Major7,
        ChordType::Minor7, ChordType::Diminished7, ChordType::HalfDiminished7,
        ChordType::Sus2, ChordType::Sus4, ChordType::Add9, ChordType::Power,
    ];

    // Try each pitch class as root
    for &root in &pcs {
        let intervals: Vec<u8> = pcs.iter().map(|p| (p + 12 - root) % 12).collect();
        let mut sorted_intervals = intervals.clone();
        sorted_intervals.sort();

        for &ct in &all_types {
            let mut chord_intervals: Vec<u8> = ct.intervals().to_vec();
            chord_intervals.sort();
            if sorted_intervals == chord_intervals {
                return Some((root, ct));
            }
        }
    }
    None
}

// ── Scale Degree Chords (Roman Numerals) ─────────────────────────

/// Major scale intervals.
const MAJOR_SCALE: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];

/// Natural minor scale intervals.
const MINOR_SCALE: [u8; 7] = [0, 2, 3, 5, 7, 8, 10];

/// Chord qualities for each degree of the major scale.
const MAJOR_SCALE_CHORDS: [ChordType; 7] = [
    ChordType::Major, ChordType::Minor, ChordType::Minor,
    ChordType::Major, ChordType::Major, ChordType::Minor, ChordType::Diminished,
];

/// Chord qualities for each degree of the natural minor scale.
const MINOR_SCALE_CHORDS: [ChordType; 7] = [
    ChordType::Minor, ChordType::Diminished, ChordType::Major,
    ChordType::Minor, ChordType::Minor, ChordType::Major, ChordType::Major,
];

/// Roman numeral symbols.
const ROMAN_UPPER: [&str; 7] = ["I", "II", "III", "IV", "V", "VI", "VII"];
const ROMAN_LOWER: [&str; 7] = ["i", "ii", "iii", "iv", "v", "vi", "vii"];

/// Get the diatonic chord at a scale degree in a major key.
pub fn major_key_chord(key_root: u8, degree: usize) -> Option<Chord> {
    if degree == 0 || degree > 7 { return None; }
    let idx = degree - 1;
    let root = (key_root + MAJOR_SCALE[idx]) % 12;
    Some(Chord::new(root, MAJOR_SCALE_CHORDS[idx]))
}

/// Get the diatonic chord at a scale degree in a minor key.
pub fn minor_key_chord(key_root: u8, degree: usize) -> Option<Chord> {
    if degree == 0 || degree > 7 { return None; }
    let idx = degree - 1;
    let root = (key_root + MINOR_SCALE[idx]) % 12;
    Some(Chord::new(root, MINOR_SCALE_CHORDS[idx]))
}

/// Roman numeral analysis of a chord in a major key.
pub fn roman_numeral(key_root: u8, chord: &Chord) -> Option<String> {
    let interval = (chord.root + 12 - key_root) % 12;
    let degree = MAJOR_SCALE.iter().position(|s| *s == interval)?;
    let expected = MAJOR_SCALE_CHORDS[degree];
    let numeral = if chord.chord_type == ChordType::Minor
        || chord.chord_type == ChordType::Diminished
        || chord.chord_type == ChordType::Minor7
        || chord.chord_type == ChordType::HalfDiminished7
    {
        ROMAN_LOWER[degree].to_string()
    } else {
        ROMAN_UPPER[degree].to_string()
    };

    let suffix = if chord.chord_type != expected {
        chord.chord_type.suffix().to_string()
    } else if chord.chord_type == ChordType::Diminished {
        "°".to_string()
    } else {
        String::new()
    };

    Some(format!("{}{}", numeral, suffix))
}

// ── Progressions ─────────────────────────────────────────────────

/// Common chord progressions as scale degree sequences.
pub fn common_progressions() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("I-IV-V-I", vec![1, 4, 5, 1]),
        ("I-V-vi-IV", vec![1, 5, 6, 4]),
        ("ii-V-I", vec![2, 5, 1]),
        ("I-vi-IV-V", vec![1, 6, 4, 5]),
        ("I-IV-vi-V", vec![1, 4, 6, 5]),
        ("vi-IV-I-V", vec![6, 4, 1, 5]),
        ("I-V-IV-V", vec![1, 5, 4, 5]),
        ("I-iii-IV-V", vec![1, 3, 4, 5]),
        ("ii-V-I-vi", vec![2, 5, 1, 6]),
        ("I-vi-ii-V", vec![1, 6, 2, 5]),
    ]
}

/// Generate a chord progression in a major key from degree sequence.
pub fn progression_in_major(key_root: u8, degrees: &[usize]) -> Vec<Chord> {
    degrees.iter().filter_map(|d| major_key_chord(key_root, *d)).collect()
}

/// Generate a chord progression in a minor key from degree sequence.
pub fn progression_in_minor(key_root: u8, degrees: &[usize]) -> Vec<Chord> {
    degrees.iter().filter_map(|d| minor_key_chord(key_root, *d)).collect()
}

// ── Voice Leading ────────────────────────────────────────────────

/// Voice lead between two chords, minimizing total movement.
/// Returns MIDI notes for the target chord voiced close to the source.
pub fn voice_lead(source_notes: &[u8], target_chord: &Chord) -> Vec<u8> {
    let target_pcs = target_chord.pitch_classes();
    if source_notes.is_empty() || target_pcs.is_empty() {
        return target_chord.midi_notes(4);
    }

    let mut result = Vec::new();
    for &src in source_notes {
        let mut best = 0u8;
        let mut best_dist = u8::MAX;
        for &pc in &target_pcs {
            // Find closest octave of this pitch class
            for oct_offset in [0i16, 12, -12, 24, -24] {
                let candidate = src as i16 + (pc as i16 - (src as i16 % 12)) + oct_offset;
                if candidate >= 0 && candidate <= 127 {
                    let dist = (candidate - src as i16).unsigned_abs() as u8;
                    if dist < best_dist {
                        best_dist = dist;
                        best = candidate as u8;
                    }
                }
            }
        }
        result.push(best);
    }
    result.sort();
    result
}

/// Total voice-leading distance (semitones) between two note sets.
pub fn voice_leading_distance(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len().min(b.len());
    let mut dist = 0u32;
    for i in 0..len {
        dist += (a[i] as i32 - b[i] as i32).unsigned_abs();
    }
    dist
}

// ── Nashville Numbers ────────────────────────────────────────────

/// Convert a chord to Nashville number notation in a given major key.
pub fn to_nashville(key_root: u8, chord: &Chord) -> Option<String> {
    let interval = (chord.root + 12 - key_root) % 12;
    let degree = MAJOR_SCALE.iter().position(|s| *s == interval)?;
    let num = degree + 1;
    let suffix = chord.chord_type.suffix();
    Some(format!("{}{}", num, suffix))
}

/// Parse Nashville number to chord in key.
pub fn from_nashville(key_root: u8, number: usize, chord_type: ChordType) -> Option<Chord> {
    if number == 0 || number > 7 { return None; }
    let root = (key_root + MAJOR_SCALE[number - 1]) % 12;
    Some(Chord::new(root, chord_type))
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_to_pc() {
        assert_eq!(note_to_pc("C"), Some(0));
        assert_eq!(note_to_pc("C#"), Some(1));
        assert_eq!(note_to_pc("A"), Some(9));
    }

    #[test]
    fn test_note_to_pc_flat() {
        assert_eq!(note_to_pc("Db"), note_to_pc("C#"));
        assert_eq!(note_to_pc("Bb"), note_to_pc("A#"));
    }

    #[test]
    fn test_chord_pitch_classes() {
        let c_major = Chord::new(0, ChordType::Major);
        assert_eq!(c_major.pitch_classes(), vec![0, 4, 7]); // C, E, G
    }

    #[test]
    fn test_chord_midi_notes() {
        let c_major = Chord::new(0, ChordType::Major);
        let notes = c_major.midi_notes(4); // C4, E4, G4
        assert_eq!(notes, vec![60, 64, 67]);
    }

    #[test]
    fn test_chord_inversion_first() {
        let c_major = Chord::with_inversion(0, ChordType::Major, 1);
        let notes = c_major.midi_notes(4);
        // C moves up octave: E4(64), G4(67), C5(72)
        assert_eq!(notes, vec![64, 67, 72]);
    }

    #[test]
    fn test_chord_inversion_second() {
        let c_major = Chord::with_inversion(0, ChordType::Major, 2);
        let notes = c_major.midi_notes(4);
        // C and E move up: G4(67), C5(72), E5(76)
        assert_eq!(notes, vec![67, 72, 76]);
    }

    #[test]
    fn test_chord_name() {
        let c = Chord::new(0, ChordType::Major);
        assert_eq!(c.name(), "C");
        let am = Chord::new(9, ChordType::Minor);
        assert_eq!(am.name(), "Am");
        let g7 = Chord::new(7, ChordType::Dominant7);
        assert_eq!(g7.name(), "G7");
    }

    #[test]
    fn test_recognize_major() {
        let result = recognize_chord(&[0, 4, 7]);
        assert_eq!(result, Some((0, ChordType::Major)));
    }

    #[test]
    fn test_recognize_minor() {
        let result = recognize_chord(&[9, 0, 4]); // A, C, E = Am
        assert_eq!(result, Some((9, ChordType::Minor)));
    }

    #[test]
    fn test_recognize_dominant7() {
        let result = recognize_chord(&[7, 11, 2, 5]); // G, B, D, F = G7
        assert_eq!(result, Some((7, ChordType::Dominant7)));
    }

    #[test]
    fn test_major_key_chords() {
        // C major: C, Dm, Em, F, G, Am, Bdim
        let chords: Vec<Chord> = (1..=7).filter_map(|d| major_key_chord(0, d)).collect();
        assert_eq!(chords.len(), 7);
        assert_eq!(chords[0].chord_type, ChordType::Major); // I
        assert_eq!(chords[1].chord_type, ChordType::Minor); // ii
        assert_eq!(chords[3].chord_type, ChordType::Major); // IV
        assert_eq!(chords[6].chord_type, ChordType::Diminished); // vii°
    }

    #[test]
    fn test_progression_1_4_5_1() {
        let prog = progression_in_major(0, &[1, 4, 5, 1]);
        assert_eq!(prog.len(), 4);
        assert_eq!(prog[0].root, 0);  // C
        assert_eq!(prog[1].root, 5);  // F
        assert_eq!(prog[2].root, 7);  // G
        assert_eq!(prog[3].root, 0);  // C
    }

    #[test]
    fn test_roman_numeral_analysis() {
        let c = Chord::new(0, ChordType::Major);
        assert_eq!(roman_numeral(0, &c), Some("I".into()));

        let dm = Chord::new(2, ChordType::Minor);
        assert_eq!(roman_numeral(0, &dm), Some("ii".into()));

        let bdim = Chord::new(11, ChordType::Diminished);
        assert_eq!(roman_numeral(0, &bdim), Some("vii°".into()));
    }

    #[test]
    fn test_voice_leading() {
        let c_notes = vec![60, 64, 67]; // C major
        let f_chord = Chord::new(5, ChordType::Major); // F major
        let voiced = voice_lead(&c_notes, &f_chord);
        // Should be close to source: F4(65), A4(69) or similar close voicing
        for (s, t) in c_notes.iter().zip(voiced.iter()) {
            assert!((*s as i16 - *t as i16).abs() <= 7);
        }
    }

    #[test]
    fn test_voice_leading_distance() {
        let a = vec![60, 64, 67];
        let b = vec![60, 65, 69];
        let dist = voice_leading_distance(&a, &b);
        assert_eq!(dist, 0 + 1 + 2); // 3
    }

    #[test]
    fn test_nashville_number() {
        let g = Chord::new(7, ChordType::Major);
        assert_eq!(to_nashville(0, &g), Some("5".into()));

        let am = Chord::new(9, ChordType::Minor);
        assert_eq!(to_nashville(0, &am), Some("6m".into()));
    }

    #[test]
    fn test_from_nashville() {
        let chord = from_nashville(0, 5, ChordType::Major).unwrap();
        assert_eq!(chord.root, 7); // G in C major
    }

    #[test]
    fn test_minor_key_chords() {
        // A minor: Am, Bdim, C, Dm, Em, F, G
        let chords: Vec<Chord> = (1..=7).filter_map(|d| minor_key_chord(9, d)).collect();
        assert_eq!(chords.len(), 7);
        assert_eq!(chords[0].chord_type, ChordType::Minor);      // i
        assert_eq!(chords[1].chord_type, ChordType::Diminished);  // ii°
        assert_eq!(chords[2].chord_type, ChordType::Major);       // III
    }

    #[test]
    fn test_common_progressions_exist() {
        let progs = common_progressions();
        assert!(progs.len() >= 5);
        for (_, degrees) in &progs {
            assert!(!degrees.is_empty());
            for &d in degrees {
                assert!(d >= 1 && d <= 7);
            }
        }
    }

    #[test]
    fn test_sus_chords() {
        let csus4 = Chord::new(0, ChordType::Sus4);
        assert_eq!(csus4.pitch_classes(), vec![0, 5, 7]);
        let csus2 = Chord::new(0, ChordType::Sus2);
        assert_eq!(csus2.pitch_classes(), vec![0, 2, 7]);
    }

    #[test]
    fn test_power_chord() {
        let c5 = Chord::new(0, ChordType::Power);
        assert_eq!(c5.pitch_classes(), vec![0, 7]);
        assert_eq!(c5.name(), "C5");
    }

    #[test]
    fn test_recognize_empty() {
        assert_eq!(recognize_chord(&[]), None);
    }

    #[test]
    fn test_progression_in_minor() {
        let prog = progression_in_minor(9, &[1, 4, 5, 1]); // Am, Dm, Em, Am
        assert_eq!(prog.len(), 4);
        assert_eq!(prog[0].root, 9);  // A
        assert_eq!(prog[1].root, 2);  // D
    }

    #[test]
    fn test_invalid_degree() {
        assert!(major_key_chord(0, 0).is_none());
        assert!(major_key_chord(0, 8).is_none());
    }
}
