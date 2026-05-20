//! Music theory utilities.
//!
//! Interval naming, note frequency (A4=440Hz equal temperament), cents,
//! circle of fifths, enharmonic equivalents, key signatures, transposition,
//! relative/parallel keys, tritone substitution, common-tone modulation
//! paths. Pure Rust.

use std::fmt;

// ── Note Names ───────────────────────────────────────────────────

const SHARP_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

const FLAT_NAMES: [&str; 12] = [
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
];

/// Pitch class (0-11, C=0).
pub type PitchClass = u8;

/// Parse a note name to pitch class.
pub fn parse_note(name: &str) -> Option<PitchClass> {
    let name = name.trim();
    if name.is_empty() { return None; }
    let (note, _rest) = if name.len() >= 2 && (name.as_bytes()[1] == b'#' || name.as_bytes()[1] == b'b') {
        (&name[..2], &name[2..])
    } else {
        (&name[..1], &name[1..])
    };

    // Try sharp names
    if let Some(pos) = SHARP_NAMES.iter().position(|n| *n == note) {
        return Some(pos as u8);
    }
    // Try flat names
    if let Some(pos) = FLAT_NAMES.iter().position(|n| *n == note) {
        return Some(pos as u8);
    }
    None
}

/// Pitch class to sharp name.
pub fn to_sharp_name(pc: PitchClass) -> &'static str {
    SHARP_NAMES[(pc % 12) as usize]
}

/// Pitch class to flat name.
pub fn to_flat_name(pc: PitchClass) -> &'static str {
    FLAT_NAMES[(pc % 12) as usize]
}

// ── Intervals ────────────────────────────────────────────────────

/// Musical interval quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalQuality {
    Perfect,
    Major,
    Minor,
    Augmented,
    Diminished,
    Tritone,
}

/// A named interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval {
    pub semitones: u8,
    pub quality: IntervalQuality,
    pub name: &'static str,
    pub short_name: &'static str,
}

/// All intervals within an octave.
pub fn intervals() -> Vec<Interval> {
    vec![
        Interval { semitones: 0,  quality: IntervalQuality::Perfect,    name: "Unison",          short_name: "P1"  },
        Interval { semitones: 1,  quality: IntervalQuality::Minor,      name: "Minor 2nd",       short_name: "m2"  },
        Interval { semitones: 2,  quality: IntervalQuality::Major,      name: "Major 2nd",       short_name: "M2"  },
        Interval { semitones: 3,  quality: IntervalQuality::Minor,      name: "Minor 3rd",       short_name: "m3"  },
        Interval { semitones: 4,  quality: IntervalQuality::Major,      name: "Major 3rd",       short_name: "M3"  },
        Interval { semitones: 5,  quality: IntervalQuality::Perfect,    name: "Perfect 4th",     short_name: "P4"  },
        Interval { semitones: 6,  quality: IntervalQuality::Tritone,    name: "Tritone",         short_name: "TT"  },
        Interval { semitones: 7,  quality: IntervalQuality::Perfect,    name: "Perfect 5th",     short_name: "P5"  },
        Interval { semitones: 8,  quality: IntervalQuality::Minor,      name: "Minor 6th",       short_name: "m6"  },
        Interval { semitones: 9,  quality: IntervalQuality::Major,      name: "Major 6th",       short_name: "M6"  },
        Interval { semitones: 10, quality: IntervalQuality::Minor,      name: "Minor 7th",       short_name: "m7"  },
        Interval { semitones: 11, quality: IntervalQuality::Major,      name: "Major 7th",       short_name: "M7"  },
        Interval { semitones: 12, quality: IntervalQuality::Perfect,    name: "Octave",          short_name: "P8"  },
    ]
}

/// Get interval name from semitone distance.
pub fn interval_from_semitones(semitones: u8) -> Option<Interval> {
    let s = semitones % 13; // fold into one octave + octave
    intervals().into_iter().find(|i| i.semitones == s)
}

/// Compute the semitone interval between two pitch classes (ascending).
pub fn interval_between(from: PitchClass, to: PitchClass) -> u8 {
    (to as i16 - from as i16).rem_euclid(12) as u8
}

// ── Frequency / Cents ────────────────────────────────────────────

/// A4 reference frequency in Hz.
pub const A4_FREQ: f64 = 440.0;

/// A4 MIDI note number.
pub const A4_MIDI: u8 = 69;

/// Convert MIDI note number to frequency (equal temperament, A4=440).
pub fn midi_to_freq(midi_note: u8) -> f64 {
    A4_FREQ * 2.0_f64.powf((midi_note as f64 - A4_MIDI as f64) / 12.0)
}

/// Convert frequency to the nearest MIDI note number.
pub fn freq_to_midi(freq: f64) -> u8 {
    if freq <= 0.0 { return 0; }
    let note = 12.0 * (freq / A4_FREQ).log2() + A4_MIDI as f64;
    note.round().clamp(0.0, 127.0) as u8
}

/// Convert frequency to MIDI note with cents deviation.
pub fn freq_to_midi_cents(freq: f64) -> (u8, f64) {
    if freq <= 0.0 { return (0, 0.0); }
    let exact = 12.0 * (freq / A4_FREQ).log2() + A4_MIDI as f64;
    let nearest = exact.round();
    let cents = (exact - nearest) * 100.0;
    (nearest.clamp(0.0, 127.0) as u8, cents)
}

/// Calculate cents between two frequencies.
pub fn cents_between(freq_a: f64, freq_b: f64) -> f64 {
    if freq_a <= 0.0 || freq_b <= 0.0 {
        return 0.0;
    }
    1200.0 * (freq_b / freq_a).log2()
}

/// Calculate cents between two MIDI notes (always an integer multiple of 100).
pub fn midi_cents_between(note_a: u8, note_b: u8) -> f64 {
    (note_b as f64 - note_a as f64) * 100.0
}

// ── Circle of Fifths ─────────────────────────────────────────────

/// Circle of fifths: returns pitch classes in order starting from C.
pub fn circle_of_fifths() -> [PitchClass; 12] {
    let mut circle = [0u8; 12];
    for i in 0..12 {
        circle[i] = ((i as u16 * 7) % 12) as u8;
    }
    circle
}

/// Position of a pitch class on the circle of fifths (0 = C).
pub fn circle_position(pc: PitchClass) -> usize {
    let circle = circle_of_fifths();
    circle.iter().position(|c| *c == pc % 12).unwrap_or(0)
}

/// Distance on circle of fifths between two keys (0-6, shortest path).
pub fn circle_distance(a: PitchClass, b: PitchClass) -> u8 {
    let pos_a = circle_position(a);
    let pos_b = circle_position(b);
    let d = (pos_a as i16 - pos_b as i16).unsigned_abs() as u8;
    d.min(12 - d)
}

// ── Key Signatures ───────────────────────────────────────────────

/// Key signature: number of sharps (positive) or flats (negative).
pub fn key_signature_accidentals(pc: PitchClass, is_minor: bool) -> i8 {
    let major_pc = if is_minor { (pc + 3) % 12 } else { pc };
    // Map from pitch class to sharps/flats on circle of fifths
    let circle_pos = circle_position(major_pc);
    if circle_pos <= 6 {
        circle_pos as i8
    } else {
        circle_pos as i8 - 12
    }
}

/// Get the sharps in a key signature (note names).
pub fn key_sharps(accidentals: i8) -> Vec<&'static str> {
    let sharp_order = ["F#", "C#", "G#", "D#", "A#", "E#", "B#"];
    if accidentals <= 0 { return Vec::new(); }
    sharp_order[..accidentals.min(7) as usize].to_vec()
}

/// Get the flats in a key signature (note names).
pub fn key_flats(accidentals: i8) -> Vec<&'static str> {
    let flat_order = ["Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"];
    if accidentals >= 0 { return Vec::new(); }
    flat_order[..(-accidentals).min(7) as usize].to_vec()
}

// ── Enharmonic Equivalents ───────────────────────────────────────

/// Get enharmonic equivalents for a note name.
pub fn enharmonic_equivalents(name: &str) -> Vec<String> {
    let pc = match parse_note(name) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut equivs = Vec::new();
    let sharp = SHARP_NAMES[pc as usize];
    let flat = FLAT_NAMES[pc as usize];

    if sharp != name { equivs.push(sharp.to_string()); }
    if flat != name && flat != sharp { equivs.push(flat.to_string()); }

    // Double sharps / flats for edge cases
    match pc {
        0 => { equivs.push("B#".to_string()); }  // C = B#
        4 => { equivs.push("Fb".to_string()); }   // E = Fb
        5 => { equivs.push("E#".to_string()); }   // F = E#
        11 => { equivs.push("Cb".to_string()); }  // B = Cb
        _ => {}
    }
    equivs.retain(|e| e != name);
    equivs.sort();
    equivs.dedup();
    equivs
}

// ── Transposition ────────────────────────────────────────────────

/// Transpose a MIDI note by semitones.
pub fn transpose(midi_note: u8, semitones: i8) -> u8 {
    let result = midi_note as i16 + semitones as i16;
    result.clamp(0, 127) as u8
}

/// Transpose a pitch class by semitones.
pub fn transpose_pc(pc: PitchClass, semitones: i8) -> PitchClass {
    ((pc as i16 + semitones as i16).rem_euclid(12)) as u8
}

// ── Related Keys ─────────────────────────────────────────────────

/// Relative minor of a major key.
pub fn relative_minor(major_root: PitchClass) -> PitchClass {
    (major_root + 9) % 12
}

/// Relative major of a minor key.
pub fn relative_major(minor_root: PitchClass) -> PitchClass {
    (minor_root + 3) % 12
}

/// Parallel minor (same root, minor quality).
pub fn parallel_minor(root: PitchClass) -> PitchClass {
    root // same root
}

/// Parallel major (same root, major quality).
pub fn parallel_major(root: PitchClass) -> PitchClass {
    root // same root
}

/// Tritone substitution: replace a dominant 7th chord root.
pub fn tritone_substitution(root: PitchClass) -> PitchClass {
    (root + 6) % 12
}

// ── Modulation Paths ─────────────────────────────────────────────

/// Find common tones between two scales (given as pitch class sets).
pub fn common_tones(scale_a: &[PitchClass], scale_b: &[PitchClass]) -> Vec<PitchClass> {
    scale_a.iter().filter(|&&pc| scale_b.contains(&pc)).copied().collect()
}

/// Find modulation path (intermediate keys) via circle of fifths.
/// Returns keys to pass through from `from` to `to` (exclusive of endpoints).
pub fn modulation_path(from: PitchClass, to: PitchClass) -> Vec<PitchClass> {
    if from == to { return Vec::new(); }

    let circle = circle_of_fifths();
    let pos_from = circle_position(from);
    let pos_to = circle_position(to);

    let clockwise_dist = (pos_to + 12 - pos_from) % 12;
    let counter_dist = (pos_from + 12 - pos_to) % 12;

    let mut path = Vec::new();
    if clockwise_dist <= counter_dist {
        // Go clockwise (sharps direction)
        for step in 1..clockwise_dist {
            path.push(circle[(pos_from + step) % 12]);
        }
    } else {
        // Go counterclockwise (flats direction)
        for step in 1..counter_dist {
            path.push(circle[(pos_from + 12 - step) % 12]);
        }
    }
    path
}

/// Suggest closely related keys for modulation from a given major key.
/// Returns: relative minor, dominant, subdominant, and their relative minors.
pub fn closely_related_keys(root: PitchClass) -> Vec<(PitchClass, &'static str)> {
    vec![
        (relative_minor(root), "Relative minor"),
        ((root + 7) % 12, "Dominant"),
        ((root + 5) % 12, "Subdominant"),
        (relative_minor((root + 7) % 12), "Dominant relative minor"),
        (relative_minor((root + 5) % 12), "Subdominant relative minor"),
    ]
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_note_sharp() {
        assert_eq!(parse_note("C"), Some(0));
        assert_eq!(parse_note("C#"), Some(1));
        assert_eq!(parse_note("A"), Some(9));
    }

    #[test]
    fn test_parse_note_flat() {
        assert_eq!(parse_note("Db"), Some(1));
        assert_eq!(parse_note("Bb"), Some(10));
    }

    #[test]
    fn test_interval_names() {
        let p5 = interval_from_semitones(7).unwrap();
        assert_eq!(p5.name, "Perfect 5th");
        assert_eq!(p5.quality, IntervalQuality::Perfect);

        let m3 = interval_from_semitones(3).unwrap();
        assert_eq!(m3.name, "Minor 3rd");
    }

    #[test]
    fn test_interval_between() {
        assert_eq!(interval_between(0, 7), 7); // C to G = P5
        assert_eq!(interval_between(7, 0), 5); // G to C = P4
        assert_eq!(interval_between(0, 0), 0); // unison
    }

    #[test]
    fn test_midi_to_freq_a4() {
        let f = midi_to_freq(69);
        assert!((f - 440.0).abs() < 1e-6);
    }

    #[test]
    fn test_midi_to_freq_a5() {
        let f = midi_to_freq(81);
        assert!((f - 880.0).abs() < 0.1);
    }

    #[test]
    fn test_midi_to_freq_c4() {
        let f = midi_to_freq(60);
        assert!((f - 261.626).abs() < 0.01);
    }

    #[test]
    fn test_freq_to_midi_roundtrip() {
        for note in 21..=108 {
            let freq = midi_to_freq(note);
            let back = freq_to_midi(freq);
            assert_eq!(back, note);
        }
    }

    #[test]
    fn test_freq_to_midi_cents() {
        let (note, cents) = freq_to_midi_cents(440.0);
        assert_eq!(note, 69);
        assert!(cents.abs() < 1e-6);
    }

    #[test]
    fn test_cents_between_octave() {
        let c = cents_between(440.0, 880.0);
        assert!((c - 1200.0).abs() < 1e-6);
    }

    #[test]
    fn test_cents_between_semitone() {
        let c = cents_between(midi_to_freq(60), midi_to_freq(61));
        assert!((c - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_circle_of_fifths() {
        let circle = circle_of_fifths();
        assert_eq!(circle[0], 0);  // C
        assert_eq!(circle[1], 7);  // G
        assert_eq!(circle[2], 2);  // D
        assert_eq!(circle[11], 5); // F
    }

    #[test]
    fn test_circle_distance() {
        assert_eq!(circle_distance(0, 7), 1); // C to G = 1 step
        assert_eq!(circle_distance(0, 5), 1); // C to F = 1 step (other direction)
        assert_eq!(circle_distance(0, 6), 6); // C to F# = 6 steps (tritone)
    }

    #[test]
    fn test_key_signature_c_major() {
        assert_eq!(key_signature_accidentals(0, false), 0); // C major: no accidentals
    }

    #[test]
    fn test_key_signature_g_major() {
        assert_eq!(key_signature_accidentals(7, false), 1); // G major: 1 sharp
    }

    #[test]
    fn test_key_signature_f_major() {
        assert_eq!(key_signature_accidentals(5, false), -1); // F major: 1 flat
    }

    #[test]
    fn test_key_sharps() {
        let sharps = key_sharps(2); // D major
        assert_eq!(sharps, vec!["F#", "C#"]);
    }

    #[test]
    fn test_key_flats() {
        let flats = key_flats(-3); // Eb major
        assert_eq!(flats, vec!["Bb", "Eb", "Ab"]);
    }

    #[test]
    fn test_enharmonic() {
        let equivs = enharmonic_equivalents("C#");
        assert!(equivs.contains(&"Db".to_string()));
    }

    #[test]
    fn test_transpose() {
        assert_eq!(transpose(60, 7), 67);  // C4 + P5 = G4
        assert_eq!(transpose(60, -5), 55); // C4 - P4 = G3
    }

    #[test]
    fn test_transpose_clamp() {
        assert_eq!(transpose(0, -1), 0);
        assert_eq!(transpose(127, 1), 127);
    }

    #[test]
    fn test_relative_keys() {
        assert_eq!(relative_minor(0), 9);  // C major -> A minor
        assert_eq!(relative_major(9), 0);  // A minor -> C major
    }

    #[test]
    fn test_tritone_sub() {
        assert_eq!(tritone_substitution(7), 1); // G7 -> Db7
        assert_eq!(tritone_substitution(0), 6); // C -> F#/Gb
    }

    #[test]
    fn test_common_tones() {
        let c_major = vec![0, 2, 4, 5, 7, 9, 11];
        let g_major = vec![7, 9, 11, 0, 2, 4, 6];
        let common = common_tones(&c_major, &g_major);
        assert_eq!(common.len(), 6); // all except F/F#
    }

    #[test]
    fn test_modulation_path() {
        let path = modulation_path(0, 2); // C to D = 2 steps on circle
        assert_eq!(path, vec![7]); // via G
    }

    #[test]
    fn test_modulation_path_same_key() {
        assert!(modulation_path(0, 0).is_empty());
    }

    #[test]
    fn test_closely_related_keys() {
        let keys = closely_related_keys(0); // C major
        assert_eq!(keys.len(), 5);
        // Should include Am, G, F, Em, Dm
        let pcs: Vec<u8> = keys.iter().map(|(pc, _)| *pc).collect();
        assert!(pcs.contains(&9));  // Am
        assert!(pcs.contains(&7));  // G
        assert!(pcs.contains(&5));  // F
    }

    #[test]
    fn test_midi_cents_between() {
        assert!((midi_cents_between(60, 72) - 1200.0).abs() < 1e-6);
    }
}
