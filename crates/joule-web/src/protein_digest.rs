//! In-silico protein digestion with trypsin/chymotrypsin rules.
//!
//! Simulates enzymatic digestion of protein sequences, supporting
//! configurable enzymes, missed cleavage counts, and peptide
//! length/mass filtering. Includes semi-specific and non-specific
//! digestion modes.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────

/// Water mass added for intact peptide (N + C terminus).
const WATER_MASS: f64 = 18.010_565;

// ── Enzyme ──────────────────────────────────────────────────────

/// Supported proteolytic enzymes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Enzyme {
    /// Cleaves after K/R, not before P.
    Trypsin,
    /// Cleaves after K/R, even before P.
    TrypsinP,
    /// Cleaves after F/W/Y/L, not before P.
    Chymotrypsin,
    /// Cleaves after D.
    AspN,
    /// Cleaves after D/E.
    GluC,
    /// Cleaves after K.
    LysC,
    /// No specificity — all subsequences.
    NonSpecific,
}

impl Enzyme {
    /// Returns true if the enzyme cleaves after residue `aa`
    /// given the following residue `next`.
    pub fn cleaves_after(&self, aa: char, next: Option<char>) -> bool {
        match self {
            Self::Trypsin => {
                (aa == 'K' || aa == 'R') && next != Some('P')
            }
            Self::TrypsinP => aa == 'K' || aa == 'R',
            Self::Chymotrypsin => {
                (aa == 'F' || aa == 'W' || aa == 'Y' || aa == 'L') && next != Some('P')
            }
            Self::AspN => aa == 'D',
            Self::GluC => aa == 'D' || aa == 'E',
            Self::LysC => aa == 'K',
            Self::NonSpecific => true,
        }
    }
}

impl fmt::Display for Enzyme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trypsin => write!(f, "trypsin"),
            Self::TrypsinP => write!(f, "trypsin/P"),
            Self::Chymotrypsin => write!(f, "chymotrypsin"),
            Self::AspN => write!(f, "Asp-N"),
            Self::GluC => write!(f, "Glu-C"),
            Self::LysC => write!(f, "Lys-C"),
            Self::NonSpecific => write!(f, "non-specific"),
        }
    }
}

// ── DigestionConfig ─────────────────────────────────────────────

/// Configuration for in-silico digestion.
#[derive(Debug, Clone)]
pub struct DigestionConfig {
    pub enzyme: Enzyme,
    pub max_missed_cleavages: usize,
    pub min_length: usize,
    pub max_length: usize,
    pub min_mass: f64,
    pub max_mass: f64,
    pub semi_specific: bool,
}

impl DigestionConfig {
    pub fn new(enzyme: Enzyme) -> Self {
        Self {
            enzyme,
            max_missed_cleavages: 2,
            min_length: 6,
            max_length: 50,
            min_mass: 400.0,
            max_mass: 6000.0,
            semi_specific: false,
        }
    }

    pub fn with_missed_cleavages(mut self, mc: usize) -> Self {
        self.max_missed_cleavages = mc;
        self
    }

    pub fn with_length_range(mut self, lo: usize, hi: usize) -> Self {
        self.min_length = lo;
        self.max_length = hi;
        self
    }

    pub fn with_mass_range(mut self, lo: f64, hi: f64) -> Self {
        self.min_mass = lo;
        self.max_mass = hi;
        self
    }

    pub fn with_semi_specific(mut self, yes: bool) -> Self {
        self.semi_specific = yes;
        self
    }
}

impl fmt::Display for DigestionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Digest({}, mc={}, len={}-{}, mass={:.0}-{:.0})",
            self.enzyme,
            self.max_missed_cleavages,
            self.min_length,
            self.max_length,
            self.min_mass,
            self.max_mass,
        )
    }
}

// ── DigestedPeptide ─────────────────────────────────────────────

/// A peptide produced by in-silico digestion.
#[derive(Debug, Clone, PartialEq)]
pub struct DigestedPeptide {
    pub sequence: String,
    pub start: usize,
    pub end: usize,
    pub missed_cleavages: usize,
    pub mass: f64,
}

impl DigestedPeptide {
    pub fn new(sequence: &str, start: usize, end: usize, missed: usize) -> Self {
        let mass = peptide_mass_simple(sequence);
        Self {
            sequence: sequence.to_string(),
            start,
            end,
            missed_cleavages: missed,
            mass,
        }
    }

    /// Peptide length in residues.
    pub fn length(&self) -> usize {
        self.sequence.len()
    }
}

impl fmt::Display for DigestedPeptide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}-{}] mc={} mass={:.2}",
            self.sequence, self.start, self.end, self.missed_cleavages, self.mass
        )
    }
}

// ── Residue mass helper ─────────────────────────────────────────

fn residue_mass_of(aa: char) -> f64 {
    match aa {
        'G' => 57.021_464,
        'A' => 71.037_114,
        'V' => 99.068_414,
        'L' => 113.084_064,
        'I' => 113.084_064,
        'P' => 97.052_764,
        'F' => 147.068_414,
        'W' => 186.079_313,
        'M' => 131.040_485,
        'S' => 87.032_028,
        'T' => 101.047_679,
        'C' => 103.009_185,
        'Y' => 163.063_329,
        'H' => 137.058_912,
        'D' => 115.026_943,
        'E' => 129.042_593,
        'N' => 114.042_927,
        'Q' => 128.058_578,
        'K' => 128.094_963,
        'R' => 156.101_111,
        _ => 0.0,
    }
}

fn peptide_mass_simple(seq: &str) -> f64 {
    seq.chars().map(residue_mass_of).sum::<f64>() + WATER_MASS
}

// ── Cleavage site detection ─────────────────────────────────────

/// Find all cleavage positions (0-based indices after which cleavage
/// occurs) in a protein sequence for the given enzyme.
pub fn find_cleavage_sites(protein: &str, enzyme: Enzyme) -> Vec<usize> {
    if enzyme == Enzyme::NonSpecific {
        return (0..protein.len()).collect();
    }
    let chars: Vec<char> = protein.chars().collect();
    let mut sites = Vec::new();
    for i in 0..chars.len() - 1 {
        if enzyme.cleaves_after(chars[i], Some(chars[i + 1])) {
            sites.push(i);
        }
    }
    // C-terminus is always a cleavage boundary.
    sites.push(chars.len() - 1);
    sites
}

// ── Core digestion ──────────────────────────────────────────────

/// Perform in-silico digestion, returning all valid peptides.
pub fn digest(protein: &str, config: &DigestionConfig) -> Vec<DigestedPeptide> {
    if protein.is_empty() {
        return Vec::new();
    }

    if config.enzyme == Enzyme::NonSpecific {
        return digest_nonspecific(protein, config);
    }

    let sites = find_cleavage_sites(protein, config.enzyme);
    let mut peptides = Vec::new();

    // Build segment boundaries: [0, site0+1, site1+1, ..., len].
    let mut boundaries = vec![0usize];
    for &s in &sites {
        let b = s + 1;
        if b <= protein.len() && (boundaries.is_empty() || *boundaries.last().unwrap() != b) {
            boundaries.push(b);
        }
    }
    if *boundaries.last().unwrap() != protein.len() {
        boundaries.push(protein.len());
    }

    // Enumerate peptides with up to max_missed_cleavages.
    for i in 0..boundaries.len() - 1 {
        for mc in 0..=config.max_missed_cleavages {
            let j = i + 1 + mc;
            if j >= boundaries.len() {
                break;
            }
            let start = boundaries[i];
            let end = boundaries[j];
            let seq = &protein[start..end];
            let len = seq.len();

            if len < config.min_length || len > config.max_length {
                continue;
            }
            let mass = peptide_mass_simple(seq);
            if mass < config.min_mass || mass > config.max_mass {
                continue;
            }

            peptides.push(DigestedPeptide::new(seq, start, end, mc));
        }
    }

    peptides
}

fn digest_nonspecific(protein: &str, config: &DigestionConfig) -> Vec<DigestedPeptide> {
    let mut peptides = Vec::new();
    let n = protein.len();
    for start in 0..n {
        for end in (start + config.min_length)..=(start + config.max_length).min(n) {
            let seq = &protein[start..end];
            let mass = peptide_mass_simple(seq);
            if mass >= config.min_mass && mass <= config.max_mass {
                peptides.push(DigestedPeptide::new(seq, start, end, 0));
            }
        }
    }
    peptides
}

// ── Coverage ────────────────────────────────────────────────────

/// Compute sequence coverage as a fraction of the protein covered
/// by the given peptides.
pub fn sequence_coverage(protein_len: usize, peptides: &[DigestedPeptide]) -> f64 {
    if protein_len == 0 {
        return 0.0;
    }
    let mut covered = vec![false; protein_len];
    for pep in peptides {
        for i in pep.start..pep.end.min(protein_len) {
            covered[i] = true;
        }
    }
    let n = covered.iter().filter(|&&c| c).count();
    n as f64 / protein_len as f64
}

/// Return unique peptide sequences (deduplicated).
pub fn unique_sequences(peptides: &[DigestedPeptide]) -> Vec<String> {
    let mut seen = Vec::new();
    for p in peptides {
        if !seen.contains(&p.sequence) {
            seen.push(p.sequence.clone());
        }
    }
    seen
}

/// Count internal cleavage sites within a peptide (missed cleavages).
pub fn count_internal_cleavages(peptide: &str, enzyme: Enzyme) -> usize {
    if peptide.len() < 2 {
        return 0;
    }
    let chars: Vec<char> = peptide.chars().collect();
    let mut count = 0;
    // Don't count the C-terminal cleavage site.
    for i in 0..chars.len() - 1 {
        let next = if i + 1 < chars.len() { Some(chars[i + 1]) } else { None };
        if enzyme.cleaves_after(chars[i], next) {
            count += 1;
        }
    }
    count
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const INSULIN_B: &str = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT";

    #[test]
    fn test_trypsin_cleavage_kr() {
        assert!(Enzyme::Trypsin.cleaves_after('K', Some('A')));
        assert!(Enzyme::Trypsin.cleaves_after('R', Some('G')));
    }

    #[test]
    fn test_trypsin_no_cleavage_before_p() {
        assert!(!Enzyme::Trypsin.cleaves_after('K', Some('P')));
        assert!(!Enzyme::Trypsin.cleaves_after('R', Some('P')));
    }

    #[test]
    fn test_trypsinp_cleaves_before_p() {
        assert!(Enzyme::TrypsinP.cleaves_after('K', Some('P')));
    }

    #[test]
    fn test_chymotrypsin_cleaves_fwyl() {
        assert!(Enzyme::Chymotrypsin.cleaves_after('F', Some('G')));
        assert!(Enzyme::Chymotrypsin.cleaves_after('W', Some('A')));
        assert!(Enzyme::Chymotrypsin.cleaves_after('Y', Some('A')));
        assert!(Enzyme::Chymotrypsin.cleaves_after('L', Some('A')));
    }

    #[test]
    fn test_chymotrypsin_no_before_p() {
        assert!(!Enzyme::Chymotrypsin.cleaves_after('F', Some('P')));
    }

    #[test]
    fn test_enzyme_display() {
        assert_eq!(format!("{}", Enzyme::Trypsin), "trypsin");
        assert_eq!(format!("{}", Enzyme::LysC), "Lys-C");
    }

    #[test]
    fn test_digest_simple() {
        // MAKER: trypsin cleaves after K → MA, KER and with mc=1 → MAKER
        let cfg = DigestionConfig::new(Enzyme::Trypsin)
            .with_length_range(2, 50)
            .with_mass_range(0.0, 10000.0);
        let peps = digest("MAKER", &cfg);
        assert!(!peps.is_empty());
    }

    #[test]
    fn test_digest_missed_cleavages() {
        let cfg = DigestionConfig::new(Enzyme::Trypsin)
            .with_missed_cleavages(0)
            .with_length_range(1, 50)
            .with_mass_range(0.0, 10000.0);
        let peps = digest("MAKER", &cfg);
        // All peptides should have 0 missed cleavages.
        assert!(peps.iter().all(|p| p.missed_cleavages == 0));
    }

    #[test]
    fn test_digest_length_filter() {
        let cfg = DigestionConfig::new(Enzyme::Trypsin)
            .with_length_range(7, 25)
            .with_mass_range(0.0, 10000.0);
        let peps = digest(INSULIN_B, &cfg);
        assert!(peps.iter().all(|p| p.length() >= 7 && p.length() <= 25));
    }

    #[test]
    fn test_digest_mass_filter() {
        let cfg = DigestionConfig::new(Enzyme::Trypsin)
            .with_length_range(1, 100)
            .with_mass_range(500.0, 3000.0);
        let peps = digest(INSULIN_B, &cfg);
        assert!(peps.iter().all(|p| p.mass >= 500.0 && p.mass <= 3000.0));
    }

    #[test]
    fn test_digest_empty_protein() {
        let cfg = DigestionConfig::new(Enzyme::Trypsin);
        let peps = digest("", &cfg);
        assert!(peps.is_empty());
    }

    #[test]
    fn test_sequence_coverage() {
        let protein = "ABCDEFGHIJ";
        let peps = vec![
            DigestedPeptide::new("ABCDEF", 0, 6, 0),
            DigestedPeptide::new("GHIJ", 6, 10, 0),
        ];
        let cov = sequence_coverage(protein.len(), &peps);
        assert!((cov - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_partial_coverage() {
        let peps = vec![DigestedPeptide::new("ABC", 0, 3, 0)];
        let cov = sequence_coverage(10, &peps);
        assert!((cov - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_unique_sequences() {
        let peps = vec![
            DigestedPeptide::new("PEPTIDE", 0, 7, 0),
            DigestedPeptide::new("PEPTIDE", 10, 17, 0),
            DigestedPeptide::new("OTHER", 20, 25, 0),
        ];
        let uniq = unique_sequences(&peps);
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn test_count_internal_cleavages() {
        // MAKER has K at position 2 → 1 internal site for trypsin.
        let c = count_internal_cleavages("MAKER", Enzyme::Trypsin);
        assert_eq!(c, 1);
    }

    #[test]
    fn test_peptide_mass_positive() {
        let mass = peptide_mass_simple("PEPTIDE");
        assert!(mass > 700.0);
    }

    #[test]
    fn test_digested_peptide_display() {
        let p = DigestedPeptide::new("MAKER", 0, 5, 1);
        let d = format!("{}", p);
        assert!(d.contains("MAKER"));
        assert!(d.contains("mc=1"));
    }

    #[test]
    fn test_config_display() {
        let cfg = DigestionConfig::new(Enzyme::Trypsin);
        let d = format!("{}", cfg);
        assert!(d.contains("trypsin"));
        assert!(d.contains("mc=2"));
    }

    #[test]
    fn test_nonspecific_digestion() {
        let cfg = DigestionConfig::new(Enzyme::NonSpecific)
            .with_length_range(3, 5)
            .with_mass_range(0.0, 100000.0);
        let peps = digest("ACDEFG", &cfg);
        assert!(peps.len() > 1);
        assert!(peps.iter().all(|p| p.length() >= 3 && p.length() <= 5));
    }
}
