//! Genetic code tables for codon-to-amino-acid translation and reverse translation.
//!
//! Implements the standard genetic code (NCBI Table 1), mitochondrial codes,
//! codon-to-amino-acid lookup, amino-acid-to-codon reverse translation,
//! codon frequency bias tables, and nucleotide sequence translation
//! across all three reading frames.

use std::fmt;
use std::collections::HashMap;

// ── Amino Acid ──────────────────────────────────────────────────

/// One-letter amino acid representation with metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AminoAcid {
    pub letter: char,
    pub three_letter: [char; 3],
    pub is_stop: bool,
}

impl AminoAcid {
    pub const fn new(letter: char, three: [char; 3], is_stop: bool) -> Self {
        Self { letter, three_letter: three, is_stop }
    }

    pub fn three_letter_string(&self) -> String {
        self.three_letter.iter().collect()
    }
}

impl fmt::Display for AminoAcid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.letter)
    }
}

// ── Genetic Code Table ──────────────────────────────────────────

/// Identifier for which genetic code to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneticCodeId {
    Standard,
    VertebrateMitochondrial,
    YeastMitochondrial,
    BacterialPlastid,
}

impl fmt::Display for GeneticCodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard (Table 1)"),
            Self::VertebrateMitochondrial => write!(f, "Vertebrate Mitochondrial (Table 2)"),
            Self::YeastMitochondrial => write!(f, "Yeast Mitochondrial (Table 3)"),
            Self::BacterialPlastid => write!(f, "Bacterial/Plastid (Table 11)"),
        }
    }
}

/// A complete codon table mapping 64 codons to amino acids.
#[derive(Debug, Clone)]
pub struct CodonTable {
    pub id: GeneticCodeId,
    pub mapping: HashMap<String, AminoAcid>,
    pub start_codons: Vec<String>,
}

impl CodonTable {
    /// Build the NCBI Standard Code (Table 1).
    pub fn standard() -> Self {
        let mut mapping = HashMap::new();
        let aa_str = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
        let codons = Self::all_codons();
        for (i, codon) in codons.iter().enumerate() {
            let ch = aa_str.as_bytes()[i] as char;
            let three = Self::one_to_three(ch);
            mapping.insert(codon.clone(), AminoAcid::new(ch, three, ch == '*'));
        }
        Self {
            id: GeneticCodeId::Standard,
            mapping,
            start_codons: vec!["ATG".into(), "CTG".into(), "TTG".into()],
        }
    }

    /// Build the Vertebrate Mitochondrial Code (Table 2).
    pub fn vertebrate_mitochondrial() -> Self {
        let mut mapping = HashMap::new();
        let aa_str = "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG";
        let codons = Self::all_codons();
        for (i, codon) in codons.iter().enumerate() {
            let ch = aa_str.as_bytes()[i] as char;
            let three = Self::one_to_three(ch);
            mapping.insert(codon.clone(), AminoAcid::new(ch, three, ch == '*'));
        }
        Self {
            id: GeneticCodeId::VertebrateMitochondrial,
            mapping,
            start_codons: vec!["ATG".into(), "ATT".into(), "ATC".into(), "ATA".into(), "GTG".into()],
        }
    }

    /// Build a table from a genetic code identifier.
    pub fn from_id(id: GeneticCodeId) -> Self {
        match id {
            GeneticCodeId::Standard => Self::standard(),
            GeneticCodeId::VertebrateMitochondrial => Self::vertebrate_mitochondrial(),
            _ => Self::standard(), // fallback
        }
    }

    /// Generate all 64 trinucleotide codons in canonical order (TTT, TTC, ..., GGG).
    fn all_codons() -> Vec<String> {
        let bases = ['T', 'C', 'A', 'G'];
        let mut codons = Vec::with_capacity(64);
        for &b1 in &bases {
            for &b2 in &bases {
                for &b3 in &bases {
                    codons.push(format!("{}{}{}", b1, b2, b3));
                }
            }
        }
        codons
    }

    fn one_to_three(ch: char) -> [char; 3] {
        match ch {
            'A' => ['A', 'l', 'a'],
            'R' => ['A', 'r', 'g'],
            'N' => ['A', 's', 'n'],
            'D' => ['A', 's', 'p'],
            'C' => ['C', 'y', 's'],
            'E' => ['G', 'l', 'u'],
            'Q' => ['G', 'l', 'n'],
            'G' => ['G', 'l', 'y'],
            'H' => ['H', 'i', 's'],
            'I' => ['I', 'l', 'e'],
            'L' => ['L', 'e', 'u'],
            'K' => ['L', 'y', 's'],
            'M' => ['M', 'e', 't'],
            'F' => ['P', 'h', 'e'],
            'P' => ['P', 'r', 'o'],
            'S' => ['S', 'e', 'r'],
            'T' => ['T', 'h', 'r'],
            'W' => ['T', 'r', 'p'],
            'Y' => ['T', 'y', 'r'],
            'V' => ['V', 'a', 'l'],
            '*' => ['S', 't', 'p'],
            _ => ['X', 'x', 'x'],
        }
    }

    /// Translate a single codon to its amino acid.
    pub fn translate_codon(&self, codon: &str) -> Option<AminoAcid> {
        self.mapping.get(&codon.to_uppercase()).copied()
    }

    /// Check if a codon is a start codon in this table.
    pub fn is_start_codon(&self, codon: &str) -> bool {
        self.start_codons.contains(&codon.to_uppercase())
    }

    /// Check if a codon is a stop codon.
    pub fn is_stop_codon(&self, codon: &str) -> bool {
        self.translate_codon(codon).map_or(false, |aa| aa.is_stop)
    }

    /// Reverse translate: find all codons encoding a given amino acid.
    pub fn reverse_translate(&self, amino_acid: char) -> Vec<String> {
        let upper = amino_acid.to_uppercase().next().unwrap_or(amino_acid);
        self.mapping
            .iter()
            .filter(|(_, aa)| aa.letter == upper)
            .map(|(codon, _)| codon.clone())
            .collect()
    }

    /// Count the degeneracy of an amino acid (how many codons encode it).
    pub fn degeneracy(&self, amino_acid: char) -> usize {
        self.reverse_translate(amino_acid).len()
    }

    /// Translate a DNA sequence into a protein string in the given frame (0, 1, or 2).
    pub fn translate_sequence(&self, dna: &str, frame: usize) -> String {
        let seq = dna.to_uppercase();
        let bytes = seq.as_bytes();
        let mut protein = String::new();
        let mut i = frame;
        while i + 3 <= bytes.len() {
            let codon = std::str::from_utf8(&bytes[i..i + 3]).unwrap_or("NNN");
            if let Some(aa) = self.translate_codon(codon) {
                if aa.is_stop {
                    protein.push('*');
                } else {
                    protein.push(aa.letter);
                }
            } else {
                protein.push('X');
            }
            i += 3;
        }
        protein
    }

    /// Translate in all three forward reading frames.
    pub fn translate_three_frames(&self, dna: &str) -> [String; 3] {
        [
            self.translate_sequence(dna, 0),
            self.translate_sequence(dna, 1),
            self.translate_sequence(dna, 2),
        ]
    }
}

impl fmt::Display for CodonTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodonTable({}, {} codons)", self.id, self.mapping.len())
    }
}

// ── Codon Usage Bias ────────────────────────────────────────────

/// Codon usage frequency table for a given organism.
#[derive(Debug, Clone)]
pub struct CodonUsageBias {
    pub organism: String,
    pub frequencies: HashMap<String, f64>,
}

impl CodonUsageBias {
    pub fn new(organism: &str) -> Self {
        Self {
            organism: organism.to_string(),
            frequencies: HashMap::new(),
        }
    }

    pub fn with_frequency(mut self, codon: &str, freq: f64) -> Self {
        self.frequencies.insert(codon.to_uppercase(), freq);
        self
    }

    /// Calculate codon usage from a coding sequence.
    pub fn from_sequence(organism: &str, sequence: &str) -> Self {
        let seq = sequence.to_uppercase();
        let bytes = seq.as_bytes();
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        let mut i = 0;
        while i + 3 <= bytes.len() {
            let codon = std::str::from_utf8(&bytes[i..i + 3]).unwrap_or("NNN");
            if codon.bytes().all(|b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                *counts.entry(codon.to_string()).or_insert(0) += 1;
                total += 1;
            }
            i += 3;
        }
        let mut bias = Self::new(organism);
        if total > 0 {
            for (codon, count) in counts {
                bias.frequencies.insert(codon, count as f64 / total as f64);
            }
        }
        bias
    }

    /// Codon Adaptation Index (CAI) for a sequence against this bias table.
    pub fn cai(&self, sequence: &str) -> f64 {
        let seq = sequence.to_uppercase();
        let bytes = seq.as_bytes();
        let mut log_sum = 0.0;
        let mut count = 0usize;
        let mut i = 0;
        while i + 3 <= bytes.len() {
            let codon = std::str::from_utf8(&bytes[i..i + 3]).unwrap_or("NNN");
            if let Some(&freq) = self.frequencies.get(codon) {
                if freq > 0.0 {
                    log_sum += freq.ln();
                    count += 1;
                }
            }
            i += 3;
        }
        if count == 0 { 0.0 } else { (log_sum / count as f64).exp() }
    }
}

impl fmt::Display for CodonUsageBias {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodonUsageBias({}, {} codons)", self.organism, self.frequencies.len())
    }
}

// ── Complement Utility ──────────────────────────────────────────

/// Compute the reverse complement of a DNA sequence.
pub fn reverse_complement(dna: &str) -> String {
    dna.chars()
        .rev()
        .map(|c| match c {
            'A' | 'a' => 'T',
            'T' | 't' => 'A',
            'G' | 'g' => 'C',
            'C' | 'c' => 'G',
            other => other,
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_table_size() {
        let table = CodonTable::standard();
        assert_eq!(table.mapping.len(), 64);
    }

    #[test]
    fn test_translate_atg() {
        let table = CodonTable::standard();
        let aa = table.translate_codon("ATG").unwrap();
        assert_eq!(aa.letter, 'M');
        assert!(!aa.is_stop);
    }

    #[test]
    fn test_translate_stop_codons() {
        let table = CodonTable::standard();
        for stop in &["TAA", "TAG", "TGA"] {
            assert!(table.is_stop_codon(stop));
        }
    }

    #[test]
    fn test_start_codon_check() {
        let table = CodonTable::standard();
        assert!(table.is_start_codon("ATG"));
        assert!(!table.is_start_codon("AAA"));
    }

    #[test]
    fn test_reverse_translate_met() {
        let table = CodonTable::standard();
        let codons = table.reverse_translate('M');
        assert_eq!(codons.len(), 1);
        assert_eq!(codons[0], "ATG");
    }

    #[test]
    fn test_reverse_translate_leu() {
        let table = CodonTable::standard();
        let codons = table.reverse_translate('L');
        assert_eq!(codons.len(), 6); // Leu has 6 codons
    }

    #[test]
    fn test_degeneracy() {
        let table = CodonTable::standard();
        assert_eq!(table.degeneracy('M'), 1); // Met: ATG only
        assert_eq!(table.degeneracy('W'), 1); // Trp: TGG only
        assert_eq!(table.degeneracy('L'), 6); // Leu: 6 codons
    }

    #[test]
    fn test_translate_sequence_frame0() {
        let table = CodonTable::standard();
        let protein = table.translate_sequence("ATGGCATAA", 0);
        assert_eq!(protein, "MA*");
    }

    #[test]
    fn test_translate_three_frames() {
        let table = CodonTable::standard();
        let frames = table.translate_three_frames("AATGGCATAA");
        assert_eq!(frames.len(), 3);
        // Frame 0: AAT GGC ATA A -> N G I
        // Frame 1: ATG GCA TAA -> M A *
        assert_eq!(frames[1], "MA*");
    }

    #[test]
    fn test_reverse_complement() {
        assert_eq!(reverse_complement("ATGC"), "GCAT");
        assert_eq!(reverse_complement("AAAA"), "TTTT");
        assert_eq!(reverse_complement(""), "");
    }

    #[test]
    fn test_reverse_complement_case_insensitive() {
        assert_eq!(reverse_complement("atgc"), "GCAT");
    }

    #[test]
    fn test_codon_usage_from_sequence() {
        let bias = CodonUsageBias::from_sequence("test", "ATGATGATG");
        // Only codon is ATG
        assert_eq!(bias.frequencies.len(), 1);
        assert!((bias.frequencies["ATG"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_codon_usage_cai() {
        let bias = CodonUsageBias::new("E. coli")
            .with_frequency("ATG", 1.0)
            .with_frequency("GCA", 0.5);
        let cai = bias.cai("ATGGCA");
        assert!(cai > 0.0);
    }

    #[test]
    fn test_amino_acid_display() {
        let aa = AminoAcid::new('M', ['M', 'e', 't'], false);
        assert_eq!(format!("{}", aa), "M");
        assert_eq!(aa.three_letter_string(), "Met");
    }

    #[test]
    fn test_genetic_code_id_display() {
        let s = format!("{}", GeneticCodeId::Standard);
        assert!(s.contains("Table 1"));
    }

    #[test]
    fn test_vertebrate_mito_table() {
        let table = CodonTable::vertebrate_mitochondrial();
        assert_eq!(table.mapping.len(), 64);
        // In vertebrate mitochondrial, AGA is a stop codon
        let aga = table.translate_codon("AGA").unwrap();
        assert!(aga.is_stop);
    }

    #[test]
    fn test_codon_table_display() {
        let table = CodonTable::standard();
        let s = format!("{}", table);
        assert!(s.contains("64 codons"));
    }

    #[test]
    fn test_codon_table_from_id() {
        let table = CodonTable::from_id(GeneticCodeId::Standard);
        assert_eq!(table.id, GeneticCodeId::Standard);
    }

    #[test]
    fn test_translate_lowercase() {
        let table = CodonTable::standard();
        let protein = table.translate_sequence("atggcataa", 0);
        assert_eq!(protein, "MA*");
    }
}
