//! Pairwise Sequence Alignment Scoring — substitution matrices (BLOSUM62,
//! BLOSUM50, PAM250, PAM120, identity), gap penalty models (linear, affine),
//! alignment score computation for nucleotide and protein sequences.
//!
//! Pure-Rust scoring engine for biological sequence alignment with
//! configurable substitution matrices and gap penalty schemes.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AlignScoreError {
    InvalidResidue(char),
    EmptySequence,
    MatrixMissing(char, char),
    InvalidGapPenalty(String),
}

impl fmt::Display for AlignScoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResidue(c) => write!(f, "invalid residue: '{c}'"),
            Self::EmptySequence => write!(f, "empty sequence"),
            Self::MatrixMissing(a, b) => write!(f, "no score for ({a}, {b})"),
            Self::InvalidGapPenalty(s) => write!(f, "invalid gap penalty: {s}"),
        }
    }
}

impl std::error::Error for AlignScoreError {}

// ── Substitution Matrix Type ────────────────────────────────────

/// Predefined substitution matrix identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixKind {
    Blosum62,
    Blosum50,
    Pam250,
    Pam120,
    Identity,
    DnaDefault,
}

impl fmt::Display for MatrixKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blosum62 => write!(f, "BLOSUM62"),
            Self::Blosum50 => write!(f, "BLOSUM50"),
            Self::Pam250 => write!(f, "PAM250"),
            Self::Pam120 => write!(f, "PAM120"),
            Self::Identity => write!(f, "Identity"),
            Self::DnaDefault => write!(f, "DNA-Default"),
        }
    }
}

// ── Substitution Matrix ─────────────────────────────────────────

/// A substitution scoring matrix for residue pairs.
#[derive(Debug, Clone)]
pub struct SubstitutionMatrix {
    kind: MatrixKind,
    scores: HashMap<(u8, u8), f64>,
}

impl SubstitutionMatrix {
    /// Create an empty matrix of the given kind.
    fn new(kind: MatrixKind) -> Self {
        Self { kind, scores: HashMap::new() }
    }

    /// Insert a symmetric score for residue pair `(a, b)`.
    fn insert(&mut self, a: u8, b: u8, score: f64) {
        let au = a.to_ascii_uppercase();
        let bu = b.to_ascii_uppercase();
        self.scores.insert((au, bu), score);
        self.scores.insert((bu, au), score);
    }

    /// Look up the score for two residues.
    pub fn score(&self, a: u8, b: u8) -> Result<f64, AlignScoreError> {
        let au = a.to_ascii_uppercase();
        let bu = b.to_ascii_uppercase();
        self.scores
            .get(&(au, bu))
            .copied()
            .ok_or(AlignScoreError::MatrixMissing(au as char, bu as char))
    }

    /// Matrix identifier.
    pub fn kind(&self) -> MatrixKind {
        self.kind
    }

    /// Build a BLOSUM62 matrix (20 standard amino acids, partial).
    pub fn blosum62() -> Self {
        let mut m = Self::new(MatrixKind::Blosum62);
        let aa = b"ARNDCQEGHILKMFPSTWYV";
        #[rustfmt::skip]
        let row_scores: &[&[f64]] = &[
            &[ 4.0,-1.0,-2.0,-2.0, 0.0,-1.0,-1.0, 0.0,-2.0,-1.0,-1.0,-1.0,-1.0,-2.0,-1.0, 1.0, 0.0,-3.0,-2.0, 0.0],
            &[-1.0, 5.0, 0.0,-2.0,-3.0, 1.0, 0.0,-2.0, 0.0,-3.0,-2.0, 2.0,-1.0,-3.0,-2.0,-1.0,-1.0,-3.0,-2.0,-3.0],
            &[-2.0, 0.0, 6.0, 1.0,-3.0, 0.0, 0.0, 0.0, 1.0,-3.0,-3.0, 0.0,-2.0,-3.0,-2.0, 1.0, 0.0,-4.0,-2.0,-3.0],
            &[-2.0,-2.0, 1.0, 6.0,-3.0, 0.0, 2.0,-1.0,-1.0,-3.0,-4.0,-1.0,-3.0,-3.0,-1.0, 0.0,-1.0,-4.0,-3.0,-3.0],
            &[ 0.0,-3.0,-3.0,-3.0, 9.0,-3.0,-4.0,-3.0,-3.0,-1.0,-1.0,-3.0,-1.0,-2.0,-3.0,-1.0,-1.0,-2.0,-2.0,-1.0],
            &[-1.0, 1.0, 0.0, 0.0,-3.0, 5.0, 2.0,-2.0, 0.0,-3.0,-2.0, 1.0, 0.0,-3.0,-1.0, 0.0,-1.0,-2.0,-1.0,-2.0],
            &[-1.0, 0.0, 0.0, 2.0,-4.0, 2.0, 5.0,-2.0, 0.0,-3.0,-3.0, 1.0,-2.0,-3.0,-1.0, 0.0,-1.0,-3.0,-2.0,-2.0],
            &[ 0.0,-2.0, 0.0,-1.0,-3.0,-2.0,-2.0, 6.0,-2.0,-4.0,-4.0,-2.0,-3.0,-3.0,-2.0, 0.0,-2.0,-2.0,-3.0,-3.0],
            &[-2.0, 0.0, 1.0,-1.0,-3.0, 0.0, 0.0,-2.0, 8.0,-3.0,-3.0,-1.0,-2.0,-1.0,-2.0,-1.0,-2.0,-2.0, 2.0,-3.0],
            &[-1.0,-3.0,-3.0,-3.0,-1.0,-3.0,-3.0,-4.0,-3.0, 4.0, 2.0,-3.0, 1.0, 0.0,-3.0,-2.0,-1.0,-3.0,-1.0, 3.0],
            &[-1.0,-2.0,-3.0,-4.0,-1.0,-2.0,-3.0,-4.0,-3.0, 2.0, 4.0,-2.0, 2.0, 0.0,-3.0,-2.0,-1.0,-2.0,-1.0, 1.0],
            &[-1.0, 2.0, 0.0,-1.0,-3.0, 1.0, 1.0,-2.0,-1.0,-3.0,-2.0, 5.0,-1.0,-3.0,-1.0, 0.0,-1.0,-3.0,-2.0,-2.0],
            &[-1.0,-1.0,-2.0,-3.0,-1.0, 0.0,-2.0,-3.0,-2.0, 1.0, 2.0,-1.0, 5.0, 0.0,-2.0,-1.0,-1.0,-1.0,-1.0, 1.0],
            &[-2.0,-3.0,-3.0,-3.0,-2.0,-3.0,-3.0,-3.0,-1.0, 0.0, 0.0,-3.0, 0.0, 6.0,-4.0,-2.0,-2.0, 1.0, 3.0,-1.0],
            &[-1.0,-2.0,-2.0,-1.0,-3.0,-1.0,-1.0,-2.0,-2.0,-3.0,-3.0,-1.0,-2.0,-4.0, 7.0,-1.0,-1.0,-4.0,-3.0,-2.0],
            &[ 1.0,-1.0, 1.0, 0.0,-1.0, 0.0, 0.0, 0.0,-1.0,-2.0,-2.0, 0.0,-1.0,-2.0,-1.0, 4.0, 1.0,-3.0,-2.0,-2.0],
            &[ 0.0,-1.0, 0.0,-1.0,-1.0,-1.0,-1.0,-2.0,-2.0,-1.0,-1.0,-1.0,-1.0,-2.0,-1.0, 1.0, 5.0,-2.0,-2.0, 0.0],
            &[-3.0,-3.0,-4.0,-4.0,-2.0,-2.0,-3.0,-2.0,-2.0,-3.0,-2.0,-3.0,-1.0, 1.0,-4.0,-3.0,-2.0,11.0, 2.0,-3.0],
            &[-2.0,-2.0,-2.0,-3.0,-2.0,-1.0,-2.0,-3.0, 2.0,-1.0,-1.0,-2.0,-1.0, 3.0,-3.0,-2.0,-2.0, 2.0, 7.0,-1.0],
            &[ 0.0,-3.0,-3.0,-3.0,-1.0,-2.0,-2.0,-3.0,-3.0, 3.0, 1.0,-2.0, 1.0,-1.0,-2.0,-2.0, 0.0,-3.0,-1.0, 4.0],
        ];
        for (i, &ai) in aa.iter().enumerate() {
            for (j, &aj) in aa.iter().enumerate() {
                m.insert(ai, aj, row_scores[i][j]);
            }
        }
        m
    }

    /// Build a BLOSUM50 matrix (simplified diagonal-heavy).
    pub fn blosum50() -> Self {
        let mut m = Self::new(MatrixKind::Blosum50);
        let aa = b"ARNDCQEGHILKMFPSTWYV";
        for (i, &ai) in aa.iter().enumerate() {
            for (j, &aj) in aa.iter().enumerate() {
                let score = if i == j { 5.0 } else { -2.0 };
                m.insert(ai, aj, score);
            }
        }
        m
    }

    /// Build a PAM250 matrix (simplified).
    pub fn pam250() -> Self {
        let mut m = Self::new(MatrixKind::Pam250);
        let aa = b"ARNDCQEGHILKMFPSTWYV";
        for (i, &ai) in aa.iter().enumerate() {
            for (j, &aj) in aa.iter().enumerate() {
                let score = if i == j { 2.0 } else { -1.0 };
                m.insert(ai, aj, score);
            }
        }
        m
    }

    /// Build a PAM120 matrix (simplified).
    pub fn pam120() -> Self {
        let mut m = Self::new(MatrixKind::Pam120);
        let aa = b"ARNDCQEGHILKMFPSTWYV";
        for (i, &ai) in aa.iter().enumerate() {
            for (j, &aj) in aa.iter().enumerate() {
                let score = if i == j { 4.0 } else { -3.0 };
                m.insert(ai, aj, score);
            }
        }
        m
    }

    /// Build a simple identity matrix (match=1, mismatch=0).
    pub fn identity() -> Self {
        let mut m = Self::new(MatrixKind::Identity);
        let all = b"ACDEFGHIKLMNPQRSTVWY";
        for &a in all {
            for &b in all {
                m.insert(a, b, if a == b { 1.0 } else { 0.0 });
            }
        }
        m
    }

    /// Build a default DNA scoring matrix (match=2, mismatch=-3).
    pub fn dna_default() -> Self {
        let mut m = Self::new(MatrixKind::DnaDefault);
        let bases = b"ACGT";
        for &a in bases {
            for &b in bases {
                m.insert(a, b, if a == b { 2.0 } else { -3.0 });
            }
        }
        m
    }
}

impl fmt::Display for SubstitutionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SubstitutionMatrix({}, {} entries)", self.kind, self.scores.len())
    }
}

// ── Gap Penalty Model ───────────────────────────────────────────

/// Gap penalty scheme for alignment scoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GapPenalty {
    /// Linear: cost = gap_open * length.
    Linear { gap_open: f64 },
    /// Affine: cost = gap_open + gap_extend * (length - 1).
    Affine { gap_open: f64, gap_extend: f64 },
}

impl GapPenalty {
    /// Create a linear gap penalty.
    pub fn linear(penalty: f64) -> Self {
        Self::Linear { gap_open: penalty }
    }

    /// Create an affine gap penalty.
    pub fn affine(open: f64, extend: f64) -> Self {
        Self::Affine { gap_open: open, gap_extend: extend }
    }

    /// Compute total penalty for a gap of `length` residues.
    pub fn cost(&self, length: usize) -> f64 {
        if length == 0 {
            return 0.0;
        }
        match self {
            Self::Linear { gap_open } => gap_open * length as f64,
            Self::Affine { gap_open, gap_extend } => {
                gap_open + gap_extend * (length as f64 - 1.0)
            }
        }
    }
}

impl fmt::Display for GapPenalty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear { gap_open } => write!(f, "Linear(gap={gap_open})"),
            Self::Affine { gap_open, gap_extend } => {
                write!(f, "Affine(open={gap_open}, ext={gap_extend})")
            }
        }
    }
}

// ── Alignment Score Config ──────────────────────────────────────

/// Configuration for pairwise alignment scoring.
#[derive(Debug, Clone)]
pub struct AlignScoreConfig {
    matrix: SubstitutionMatrix,
    gap_penalty: GapPenalty,
    case_sensitive: bool,
}

impl AlignScoreConfig {
    /// Create a config with the given substitution matrix.
    pub fn new(matrix: SubstitutionMatrix) -> Self {
        Self {
            matrix,
            gap_penalty: GapPenalty::linear(-1.0),
            case_sensitive: false,
        }
    }

    pub fn with_gap_penalty(mut self, gp: GapPenalty) -> Self {
        self.gap_penalty = gp;
        self
    }

    pub fn with_case_sensitive(mut self, cs: bool) -> Self {
        self.case_sensitive = cs;
        self
    }

    pub fn matrix(&self) -> &SubstitutionMatrix { &self.matrix }
    pub fn gap_penalty(&self) -> &GapPenalty { &self.gap_penalty }
}

impl fmt::Display for AlignScoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlignScoreConfig({}, {})", self.matrix, self.gap_penalty)
    }
}

// ── Scored Pair ─────────────────────────────────────────────────

/// Result of scoring a pair of aligned residues.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPair {
    pub residue_a: u8,
    pub residue_b: u8,
    pub score: f64,
}

impl fmt::Display for ScoredPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{}: {:.1}",
            self.residue_a as char, self.residue_b as char, self.score
        )
    }
}

// ── Alignment Scorer ────────────────────────────────────────────

/// Scores aligned sequence pairs using the configured matrix and gap model.
#[derive(Debug, Clone)]
pub struct AlignmentScorer {
    config: AlignScoreConfig,
}

impl AlignmentScorer {
    pub fn new(config: AlignScoreConfig) -> Self {
        Self { config }
    }

    /// Score a single residue pair.
    pub fn score_pair(&self, a: u8, b: u8) -> Result<ScoredPair, AlignScoreError> {
        let score = self.config.matrix.score(a, b)?;
        Ok(ScoredPair { residue_a: a, residue_b: b, score })
    }

    /// Score an aligned pair (with gaps represented by b'-').
    pub fn score_aligned(
        &self,
        aligned_a: &[u8],
        aligned_b: &[u8],
    ) -> Result<f64, AlignScoreError> {
        if aligned_a.is_empty() || aligned_b.is_empty() {
            return Err(AlignScoreError::EmptySequence);
        }
        let mut total = 0.0;
        let mut gap_len_a: usize = 0;
        let mut gap_len_b: usize = 0;

        for (&a, &b) in aligned_a.iter().zip(aligned_b.iter()) {
            let is_gap_a = a == b'-';
            let is_gap_b = b == b'-';

            if is_gap_a {
                if gap_len_b > 0 {
                    total += self.config.gap_penalty.cost(gap_len_b);
                    gap_len_b = 0;
                }
                gap_len_a += 1;
            } else if is_gap_b {
                if gap_len_a > 0 {
                    total += self.config.gap_penalty.cost(gap_len_a);
                    gap_len_a = 0;
                }
                gap_len_b += 1;
            } else {
                if gap_len_a > 0 {
                    total += self.config.gap_penalty.cost(gap_len_a);
                    gap_len_a = 0;
                }
                if gap_len_b > 0 {
                    total += self.config.gap_penalty.cost(gap_len_b);
                    gap_len_b = 0;
                }
                total += self.config.matrix.score(a, b)?;
            }
        }
        if gap_len_a > 0 {
            total += self.config.gap_penalty.cost(gap_len_a);
        }
        if gap_len_b > 0 {
            total += self.config.gap_penalty.cost(gap_len_b);
        }
        Ok(total)
    }

    /// Compute percent identity from aligned sequences.
    pub fn percent_identity(aligned_a: &[u8], aligned_b: &[u8]) -> f64 {
        if aligned_a.is_empty() {
            return 0.0;
        }
        let matches = aligned_a
            .iter()
            .zip(aligned_b.iter())
            .filter(|(a, b)| **a != b'-' && **b != b'-' && a.to_ascii_uppercase() == b.to_ascii_uppercase())
            .count();
        let cols = aligned_a.len().max(aligned_b.len());
        if cols == 0 { 0.0 } else { matches as f64 / cols as f64 * 100.0 }
    }

    /// Count gaps in an aligned sequence.
    pub fn gap_count(aligned: &[u8]) -> usize {
        aligned.iter().filter(|&&c| c == b'-').count()
    }

    /// Compute coverage: fraction of non-gap positions.
    pub fn coverage(aligned: &[u8]) -> f64 {
        if aligned.is_empty() {
            return 0.0;
        }
        let non_gap = aligned.iter().filter(|&&c| c != b'-').count();
        non_gap as f64 / aligned.len() as f64
    }
}

impl fmt::Display for AlignmentScorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlignmentScorer({})", self.config)
    }
}

// ── Sequence Validator ──────────────────────────────────────────

/// Validate that a sequence contains only valid residues.
pub fn validate_protein(seq: &[u8]) -> Result<(), AlignScoreError> {
    for &c in seq {
        let u = c.to_ascii_uppercase();
        if !b"ACDEFGHIKLMNPQRSTVWY".contains(&u) {
            return Err(AlignScoreError::InvalidResidue(c as char));
        }
    }
    Ok(())
}

/// Validate a DNA sequence (ACGT only).
pub fn validate_dna(seq: &[u8]) -> Result<(), AlignScoreError> {
    for &c in seq {
        let u = c.to_ascii_uppercase();
        if !b"ACGT".contains(&u) {
            return Err(AlignScoreError::InvalidResidue(c as char));
        }
    }
    Ok(())
}

/// Reverse complement of a DNA sequence.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|b| match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            other => other,
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blosum62_diagonal() {
        let m = SubstitutionMatrix::blosum62();
        assert!((m.score(b'A', b'A').unwrap() - 4.0).abs() < 1e-9);
        assert!((m.score(b'W', b'W').unwrap() - 11.0).abs() < 1e-9);
    }

    #[test]
    fn blosum62_symmetric() {
        let m = SubstitutionMatrix::blosum62();
        let ab = m.score(b'A', b'R').unwrap();
        let ba = m.score(b'R', b'A').unwrap();
        assert!((ab - ba).abs() < 1e-9);
    }

    #[test]
    fn dna_matrix_match() {
        let m = SubstitutionMatrix::dna_default();
        assert!((m.score(b'A', b'A').unwrap() - 2.0).abs() < 1e-9);
        assert!((m.score(b'A', b'C').unwrap() - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn gap_linear_cost() {
        let gp = GapPenalty::linear(-2.0);
        assert!((gp.cost(3) - (-6.0)).abs() < 1e-9);
        assert!((gp.cost(0)).abs() < 1e-9);
    }

    #[test]
    fn gap_affine_cost() {
        let gp = GapPenalty::affine(-10.0, -0.5);
        assert!((gp.cost(1) - (-10.0)).abs() < 1e-9);
        assert!((gp.cost(4) - (-11.5)).abs() < 1e-9);
    }

    #[test]
    fn score_identical_dna() {
        let cfg = AlignScoreConfig::new(SubstitutionMatrix::dna_default());
        let scorer = AlignmentScorer::new(cfg);
        let s = scorer.score_aligned(b"ACGT", b"ACGT").unwrap();
        assert!((s - 8.0).abs() < 1e-9);
    }

    #[test]
    fn score_with_gaps() {
        let cfg = AlignScoreConfig::new(SubstitutionMatrix::dna_default())
            .with_gap_penalty(GapPenalty::linear(-2.0));
        let scorer = AlignmentScorer::new(cfg);
        let s = scorer.score_aligned(b"AC-GT", b"ACAGT").unwrap();
        assert!(s < 8.0); // gap should reduce score
    }

    #[test]
    fn percent_identity_exact() {
        let pid = AlignmentScorer::percent_identity(b"ACGT", b"ACGT");
        assert!((pid - 100.0).abs() < 1e-9);
    }

    #[test]
    fn percent_identity_half() {
        let pid = AlignmentScorer::percent_identity(b"ACAT", b"ACGT");
        assert!((pid - 75.0).abs() < 1e-9);
    }

    #[test]
    fn gap_count_test() {
        assert_eq!(AlignmentScorer::gap_count(b"AC--GT"), 2);
        assert_eq!(AlignmentScorer::gap_count(b"ACGT"), 0);
    }

    #[test]
    fn coverage_test() {
        let c = AlignmentScorer::coverage(b"AC--GT");
        assert!((c - 4.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn validate_protein_ok() {
        assert!(validate_protein(b"ACDEFGHIK").is_ok());
    }

    #[test]
    fn validate_protein_bad() {
        assert!(validate_protein(b"ACXZ").is_err());
    }

    #[test]
    fn validate_dna_ok() {
        assert!(validate_dna(b"ACGTACGT").is_ok());
    }

    #[test]
    fn validate_dna_bad() {
        assert!(validate_dna(b"ACGX").is_err());
    }

    #[test]
    fn reverse_complement_test() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
        assert_eq!(reverse_complement(b"AAAC"), b"GTTT");
    }

    #[test]
    fn identity_matrix_scores() {
        let m = SubstitutionMatrix::identity();
        assert!((m.score(b'A', b'A').unwrap() - 1.0).abs() < 1e-9);
        assert!((m.score(b'A', b'C').unwrap()).abs() < 1e-9);
    }

    #[test]
    fn config_builder() {
        let cfg = AlignScoreConfig::new(SubstitutionMatrix::blosum62())
            .with_gap_penalty(GapPenalty::affine(-11.0, -1.0))
            .with_case_sensitive(true);
        assert!(format!("{cfg}").contains("BLOSUM62"));
    }

    #[test]
    fn score_pair_blosum() {
        let cfg = AlignScoreConfig::new(SubstitutionMatrix::blosum62());
        let scorer = AlignmentScorer::new(cfg);
        let sp = scorer.score_pair(b'A', b'A').unwrap();
        assert!((sp.score - 4.0).abs() < 1e-9);
        assert!(format!("{sp}").contains("A-A"));
    }

    #[test]
    fn empty_sequence_err() {
        let cfg = AlignScoreConfig::new(SubstitutionMatrix::dna_default());
        let scorer = AlignmentScorer::new(cfg);
        assert!(scorer.score_aligned(b"", b"ACGT").is_err());
    }
}
