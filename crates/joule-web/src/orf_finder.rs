//! Open reading frame detection with six-frame translation and longest ORF extraction.
//!
//! Provides comprehensive ORF scanning across all six reading frames
//! (three forward, three reverse-complement), configurable minimum
//! length filtering, ATG-only or alternative start codon modes,
//! and ranked extraction of the longest ORFs by frame and strand.

use std::fmt;

// ── Strand ──────────────────────────────────────────────────────

/// Strand designation for ORF results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrfStrand {
    Forward,
    Reverse,
}

impl fmt::Display for OrfStrand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "+"),
            Self::Reverse => write!(f, "-"),
        }
    }
}

// ── ORF ─────────────────────────────────────────────────────────

/// A single open reading frame identified in a sequence.
#[derive(Debug, Clone)]
pub struct Orf {
    pub start: usize,
    pub end: usize,
    pub strand: OrfStrand,
    pub frame: u8,
    pub protein: String,
    pub has_start_codon: bool,
    pub has_stop_codon: bool,
}

impl Orf {
    pub fn new(start: usize, end: usize, strand: OrfStrand, frame: u8) -> Self {
        Self {
            start,
            end,
            strand,
            frame,
            protein: String::new(),
            has_start_codon: false,
            has_stop_codon: false,
        }
    }

    pub fn with_protein(mut self, protein: &str) -> Self {
        self.protein = protein.to_string();
        self
    }

    pub fn with_start_codon(mut self, has: bool) -> Self {
        self.has_start_codon = has;
        self
    }

    pub fn with_stop_codon(mut self, has: bool) -> Self {
        self.has_stop_codon = has;
        self
    }

    /// Length in nucleotides.
    pub fn length_nt(&self) -> usize {
        if self.end >= self.start { self.end - self.start + 1 } else { 0 }
    }

    /// Length in amino acids.
    pub fn length_aa(&self) -> usize {
        self.protein.len()
    }

    /// Whether this ORF is complete (has both start and stop).
    pub fn is_complete(&self) -> bool {
        self.has_start_codon && self.has_stop_codon
    }
}

impl fmt::Display for Orf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ORF({}-{}, {}, frame={}, {}nt, {}aa, {})",
            self.start,
            self.end,
            self.strand,
            self.frame,
            self.length_nt(),
            self.length_aa(),
            if self.is_complete() { "complete" } else { "partial" },
        )
    }
}

// ── ORF Finder Config ───────────────────────────────────────────

/// Configuration for ORF detection.
#[derive(Debug, Clone)]
pub struct OrfFinderConfig {
    pub min_length_nt: usize,
    pub require_start_codon: bool,
    pub allow_alternative_starts: bool,
    pub scan_reverse: bool,
    pub max_orfs: usize,
}

impl OrfFinderConfig {
    pub fn new() -> Self {
        Self {
            min_length_nt: 90,
            require_start_codon: true,
            allow_alternative_starts: false,
            scan_reverse: true,
            max_orfs: 1000,
        }
    }

    pub fn with_min_length_nt(mut self, len: usize) -> Self {
        self.min_length_nt = len;
        self
    }

    pub fn with_require_start_codon(mut self, req: bool) -> Self {
        self.require_start_codon = req;
        self
    }

    pub fn with_allow_alternative_starts(mut self, allow: bool) -> Self {
        self.allow_alternative_starts = allow;
        self
    }

    pub fn with_scan_reverse(mut self, scan: bool) -> Self {
        self.scan_reverse = scan;
        self
    }

    pub fn with_max_orfs(mut self, max: usize) -> Self {
        self.max_orfs = max;
        self
    }
}

impl fmt::Display for OrfFinderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrfFinderConfig(min={}nt, start_req={}, alt_starts={}, reverse={})",
            self.min_length_nt, self.require_start_codon,
            self.allow_alternative_starts, self.scan_reverse,
        )
    }
}

// ── Codon Translation (minimal, self-contained) ─────────────────

fn translate_codon_simple(codon: &[u8]) -> char {
    if codon.len() < 3 {
        return 'X';
    }
    match (codon[0], codon[1], codon[2]) {
        (b'A', b'T', b'G') => 'M',
        (b'T', b'T', b'T') | (b'T', b'T', b'C') => 'F',
        (b'T', b'T', b'A') | (b'T', b'T', b'G') => 'L',
        (b'C', b'T', _) => 'L',
        (b'A', b'T', b'T') | (b'A', b'T', b'C') | (b'A', b'T', b'A') => 'I',
        (b'G', b'T', _) => 'V',
        (b'T', b'C', _) => 'S',
        (b'C', b'C', _) => 'P',
        (b'A', b'C', _) => 'T',
        (b'G', b'C', _) => 'A',
        (b'T', b'A', b'T') | (b'T', b'A', b'C') => 'Y',
        (b'T', b'A', b'A') | (b'T', b'A', b'G') => '*',
        (b'C', b'A', b'T') | (b'C', b'A', b'C') => 'H',
        (b'C', b'A', b'A') | (b'C', b'A', b'G') => 'Q',
        (b'A', b'A', b'T') | (b'A', b'A', b'C') => 'N',
        (b'A', b'A', b'A') | (b'A', b'A', b'G') => 'K',
        (b'G', b'A', b'T') | (b'G', b'A', b'C') => 'D',
        (b'G', b'A', b'A') | (b'G', b'A', b'G') => 'E',
        (b'T', b'G', b'T') | (b'T', b'G', b'C') => 'C',
        (b'T', b'G', b'A') => '*',
        (b'T', b'G', b'G') => 'W',
        (b'C', b'G', _) => 'R',
        (b'A', b'G', b'T') | (b'A', b'G', b'C') => 'S',
        (b'A', b'G', b'A') | (b'A', b'G', b'G') => 'R',
        (b'G', b'G', _) => 'G',
        _ => 'X',
    }
}

fn is_start_codon(codon: &[u8], allow_alt: bool) -> bool {
    if codon.len() < 3 {
        return false;
    }
    match (codon[0], codon[1], codon[2]) {
        (b'A', b'T', b'G') => true,
        (b'G', b'T', b'G') | (b'T', b'T', b'G') => allow_alt,
        _ => false,
    }
}

fn is_stop_codon(codon: &[u8]) -> bool {
    if codon.len() < 3 {
        return false;
    }
    matches!(
        (codon[0], codon[1], codon[2]),
        (b'T', b'A', b'A') | (b'T', b'A', b'G') | (b'T', b'G', b'A')
    )
}

fn reverse_complement(dna: &[u8]) -> Vec<u8> {
    dna.iter()
        .rev()
        .map(|b| match b {
            b'A' => b'T',
            b'T' => b'A',
            b'G' => b'C',
            b'C' => b'G',
            other => *other,
        })
        .collect()
}

// ── ORF Scanning ────────────────────────────────────────────────

/// Scan a single reading frame for ORFs.
fn scan_frame(
    seq: &[u8],
    frame: usize,
    strand: OrfStrand,
    config: &OrfFinderConfig,
) -> Vec<Orf> {
    let mut orfs = Vec::new();
    let mut i = frame;
    let mut current_start: Option<usize> = if !config.require_start_codon {
        Some(frame)
    } else {
        None
    };
    let mut protein = String::new();

    while i + 3 <= seq.len() {
        let codon = &seq[i..i + 3];

        if current_start.is_none()
            && is_start_codon(codon, config.allow_alternative_starts)
        {
            current_start = Some(i);
            protein.clear();
            protein.push(translate_codon_simple(codon));
        } else if current_start.is_some() && !is_stop_codon(codon) {
            protein.push(translate_codon_simple(codon));
        }

        if current_start.is_some() && is_stop_codon(codon) {
            let start = current_start.unwrap();
            let end = i + 2;
            let nt_len = end - start + 1;
            if nt_len >= config.min_length_nt {
                let orf = Orf::new(start, end, strand, frame as u8)
                    .with_protein(&protein)
                    .with_start_codon(is_start_codon(&seq[start..start + 3], true))
                    .with_stop_codon(true);
                orfs.push(orf);
            }
            current_start = None;
            protein.clear();
        }

        i += 3;
    }

    // Handle ORF extending to end of sequence (no stop codon found)
    if let Some(start) = current_start {
        let end = seq.len() - 1;
        let nt_len = end - start + 1;
        if nt_len >= config.min_length_nt {
            let has_start = start + 3 <= seq.len()
                && is_start_codon(&seq[start..start + 3], true);
            let orf = Orf::new(start, end, strand, frame as u8)
                .with_protein(&protein)
                .with_start_codon(has_start)
                .with_stop_codon(false);
            orfs.push(orf);
        }
    }

    orfs
}

// ── Public API ──────────────────────────────────────────────────

/// Find all ORFs in a DNA sequence across six frames.
pub fn find_orfs(sequence: &str, config: &OrfFinderConfig) -> Vec<Orf> {
    let fwd: Vec<u8> = sequence.to_uppercase().bytes().collect();
    let mut all_orfs = Vec::new();

    // Forward frames
    for frame in 0..3 {
        let mut orfs = scan_frame(&fwd, frame, OrfStrand::Forward, config);
        all_orfs.append(&mut orfs);
    }

    // Reverse frames
    if config.scan_reverse {
        let rev = reverse_complement(&fwd);
        for frame in 0..3 {
            let mut orfs = scan_frame(&rev, frame, OrfStrand::Reverse, config);
            all_orfs.append(&mut orfs);
        }
    }

    // Sort by length descending, then truncate
    all_orfs.sort_by(|a, b| b.length_nt().cmp(&a.length_nt()));
    all_orfs.truncate(config.max_orfs);
    all_orfs
}

/// Find the single longest ORF across all six frames.
pub fn longest_orf(sequence: &str, config: &OrfFinderConfig) -> Option<Orf> {
    let orfs = find_orfs(sequence, config);
    orfs.into_iter().next()
}

/// Six-frame translation: translate a DNA sequence in all six frames.
pub fn six_frame_translation(dna: &str) -> Vec<(OrfStrand, u8, String)> {
    let fwd: Vec<u8> = dna.to_uppercase().bytes().collect();
    let rev = reverse_complement(&fwd);
    let mut results = Vec::new();

    for frame in 0..3u8 {
        let mut protein = String::new();
        let mut i = frame as usize;
        while i + 3 <= fwd.len() {
            protein.push(translate_codon_simple(&fwd[i..i + 3]));
            i += 3;
        }
        results.push((OrfStrand::Forward, frame, protein));
    }

    for frame in 0..3u8 {
        let mut protein = String::new();
        let mut i = frame as usize;
        while i + 3 <= rev.len() {
            protein.push(translate_codon_simple(&rev[i..i + 3]));
            i += 3;
        }
        results.push((OrfStrand::Reverse, frame, protein));
    }

    results
}

/// Summarize ORFs by frame and strand.
pub fn orf_summary(orfs: &[Orf]) -> Vec<(OrfStrand, u8, usize, usize)> {
    let mut summary = Vec::new();
    for frame in 0..3u8 {
        for &strand in &[OrfStrand::Forward, OrfStrand::Reverse] {
            let matching: Vec<&Orf> = orfs.iter()
                .filter(|o| o.strand == strand && o.frame == frame)
                .collect();
            let max_len = matching.iter().map(|o| o.length_nt()).max().unwrap_or(0);
            summary.push((strand, frame, matching.len(), max_len));
        }
    }
    summary
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_orf_seq(codons: usize) -> String {
        let mut s = String::from("ATG");
        for _ in 0..codons.saturating_sub(2) {
            s.push_str("GCA"); // Ala codons
        }
        s.push_str("TAA");
        s
    }

    #[test]
    fn test_orf_length_nt() {
        let orf = Orf::new(0, 89, OrfStrand::Forward, 0);
        assert_eq!(orf.length_nt(), 90);
    }

    #[test]
    fn test_orf_is_complete() {
        let orf = Orf::new(0, 89, OrfStrand::Forward, 0)
            .with_start_codon(true)
            .with_stop_codon(true);
        assert!(orf.is_complete());
    }

    #[test]
    fn test_orf_is_partial() {
        let orf = Orf::new(0, 89, OrfStrand::Forward, 0)
            .with_start_codon(true)
            .with_stop_codon(false);
        assert!(!orf.is_complete());
    }

    #[test]
    fn test_orf_display() {
        let orf = Orf::new(100, 399, OrfStrand::Forward, 1)
            .with_protein("MAAA")
            .with_start_codon(true)
            .with_stop_codon(true);
        let s = format!("{}", orf);
        assert!(s.contains("ORF(100-399"));
        assert!(s.contains("complete"));
    }

    #[test]
    fn test_config_builders() {
        let config = OrfFinderConfig::new()
            .with_min_length_nt(150)
            .with_require_start_codon(false)
            .with_scan_reverse(false)
            .with_max_orfs(50);
        assert_eq!(config.min_length_nt, 150);
        assert!(!config.require_start_codon);
        assert!(!config.scan_reverse);
        assert_eq!(config.max_orfs, 50);
    }

    #[test]
    fn test_config_display() {
        let config = OrfFinderConfig::new();
        let s = format!("{}", config);
        assert!(s.contains("min=90nt"));
    }

    #[test]
    fn test_find_orfs_simple() {
        let seq = make_orf_seq(40); // 40 codons = 120nt
        let config = OrfFinderConfig::new()
            .with_min_length_nt(90)
            .with_scan_reverse(false);
        let orfs = find_orfs(&seq, &config);
        assert!(!orfs.is_empty());
        assert!(orfs[0].has_start_codon);
        assert!(orfs[0].has_stop_codon);
    }

    #[test]
    fn test_find_orfs_too_short() {
        let seq = "ATGGCATAA"; // 9nt
        let config = OrfFinderConfig::new()
            .with_min_length_nt(90)
            .with_scan_reverse(false);
        let orfs = find_orfs(seq, &config);
        assert!(orfs.is_empty());
    }

    #[test]
    fn test_longest_orf() {
        let short = make_orf_seq(32); // 96nt
        let long = make_orf_seq(50); // 150nt
        let seq = format!("{}NNNNNN{}", short, long);
        let config = OrfFinderConfig::new()
            .with_min_length_nt(90)
            .with_scan_reverse(false);
        let best = longest_orf(&seq, &config);
        assert!(best.is_some());
        let best = best.unwrap();
        assert!(best.length_nt() >= 150);
    }

    #[test]
    fn test_six_frame_translation_count() {
        let seq = "ATGATGATGATGATGATG";
        let frames = six_frame_translation(seq);
        assert_eq!(frames.len(), 6);
    }

    #[test]
    fn test_six_frame_forward_frame0() {
        let seq = "ATGGCATAA";
        let frames = six_frame_translation(seq);
        let (strand, frame, protein) = &frames[0];
        assert_eq!(*strand, OrfStrand::Forward);
        assert_eq!(*frame, 0);
        assert_eq!(protein, "MA*");
    }

    #[test]
    fn test_reverse_complement_internal() {
        let fwd = b"ATGC";
        let rev = reverse_complement(fwd);
        assert_eq!(rev, b"GCAT");
    }

    #[test]
    fn test_translate_codon_simple_met() {
        assert_eq!(translate_codon_simple(b"ATG"), 'M');
    }

    #[test]
    fn test_translate_codon_simple_stop() {
        assert_eq!(translate_codon_simple(b"TAA"), '*');
        assert_eq!(translate_codon_simple(b"TAG"), '*');
        assert_eq!(translate_codon_simple(b"TGA"), '*');
    }

    #[test]
    fn test_orf_summary() {
        let seq = make_orf_seq(40);
        let config = OrfFinderConfig::new()
            .with_min_length_nt(90)
            .with_scan_reverse(false);
        let orfs = find_orfs(&seq, &config);
        let summary = orf_summary(&orfs);
        assert_eq!(summary.len(), 6); // 3 frames x 2 strands
    }

    #[test]
    fn test_strand_display() {
        assert_eq!(format!("{}", OrfStrand::Forward), "+");
        assert_eq!(format!("{}", OrfStrand::Reverse), "-");
    }

    #[test]
    fn test_orf_length_aa() {
        let orf = Orf::new(0, 89, OrfStrand::Forward, 0)
            .with_protein("MAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        assert_eq!(orf.length_aa(), 30);
    }

    #[test]
    fn test_alternative_start_codons() {
        // GTG start
        let mut seq = String::from("GTG");
        for _ in 0..38 {
            seq.push_str("GCA");
        }
        seq.push_str("TAA");
        let config = OrfFinderConfig::new()
            .with_min_length_nt(90)
            .with_allow_alternative_starts(true)
            .with_scan_reverse(false);
        let orfs = find_orfs(&seq, &config);
        assert!(!orfs.is_empty());
    }

    #[test]
    fn test_no_start_codon_required() {
        // Sequence without ATG but with stop
        let mut seq = String::new();
        for _ in 0..35 {
            seq.push_str("GCA");
        }
        seq.push_str("TAA");
        let config = OrfFinderConfig::new()
            .with_min_length_nt(90)
            .with_require_start_codon(false)
            .with_scan_reverse(false);
        let orfs = find_orfs(&seq, &config);
        assert!(!orfs.is_empty());
        assert!(!orfs[0].has_start_codon);
    }
}
