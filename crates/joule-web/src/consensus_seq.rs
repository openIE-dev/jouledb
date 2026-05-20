//! Multiple Sequence Consensus — position weight matrix (PWM), position
//! frequency matrix (PFM), consensus sequence, information content,
//! sequence logo scoring, profile-based sequence comparison.
//!
//! Pure-Rust consensus engine for multiple sequence alignment profiles
//! with configurable alphabets, pseudocount smoothing, and bit-scaled
//! information content for motif analysis.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusError {
    EmptyAlignment,
    UnequalLengths,
    InvalidAlphabet(String),
    InvalidPseudocount(f64),
    PositionOutOfRange(usize),
}

impl fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAlignment => write!(f, "empty alignment"),
            Self::UnequalLengths => write!(f, "sequences have unequal lengths"),
            Self::InvalidAlphabet(s) => write!(f, "invalid alphabet: {s}"),
            Self::InvalidPseudocount(p) => write!(f, "invalid pseudocount: {p}"),
            Self::PositionOutOfRange(p) => write!(f, "position out of range: {p}"),
        }
    }
}

impl std::error::Error for ConsensusError {}

// ── Alphabet ────────────────────────────────────────────────────

/// Sequence alphabet type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Alphabet {
    Dna,
    Rna,
    Protein,
    Custom(Vec<u8>),
}

impl Alphabet {
    /// Characters in this alphabet.
    pub fn chars(&self) -> &[u8] {
        match self {
            Self::Dna => b"ACGT",
            Self::Rna => b"ACGU",
            Self::Protein => b"ACDEFGHIKLMNPQRSTVWY",
            Self::Custom(c) => c,
        }
    }

    /// Number of symbols.
    pub fn size(&self) -> usize {
        self.chars().len()
    }

    /// Background frequency (uniform).
    pub fn background(&self) -> f64 {
        1.0 / self.size() as f64
    }
}

impl fmt::Display for Alphabet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dna => write!(f, "DNA"),
            Self::Rna => write!(f, "RNA"),
            Self::Protein => write!(f, "Protein"),
            Self::Custom(c) => write!(f, "Custom({})", c.len()),
        }
    }
}

// ── Position Frequency Matrix (PFM) ────────────────────────────

/// Position Frequency Matrix: raw counts at each position.
#[derive(Debug, Clone)]
pub struct PosFreqMatrix {
    counts: Vec<HashMap<u8, usize>>,
    num_seqs: usize,
    width: usize,
    alphabet: Alphabet,
}

impl PosFreqMatrix {
    /// Build a PFM from aligned sequences.
    pub fn from_alignment(
        sequences: &[&[u8]],
        alphabet: Alphabet,
    ) -> Result<Self, ConsensusError> {
        if sequences.is_empty() {
            return Err(ConsensusError::EmptyAlignment);
        }
        let width = sequences[0].len();
        if width == 0 {
            return Err(ConsensusError::EmptyAlignment);
        }
        for seq in sequences {
            if seq.len() != width {
                return Err(ConsensusError::UnequalLengths);
            }
        }

        let mut counts = vec![HashMap::new(); width];
        for seq in sequences {
            for (pos, &base) in seq.iter().enumerate() {
                let b = base.to_ascii_uppercase();
                if b != b'-' {
                    *counts[pos].entry(b).or_insert(0) += 1;
                }
            }
        }

        Ok(Self {
            counts,
            num_seqs: sequences.len(),
            width,
            alphabet,
        })
    }

    /// Count of symbol `c` at position `pos`.
    pub fn count(&self, pos: usize, c: u8) -> usize {
        self.counts
            .get(pos)
            .and_then(|m| m.get(&c.to_ascii_uppercase()))
            .copied()
            .unwrap_or(0)
    }

    /// Width (number of positions).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of sequences.
    pub fn depth(&self) -> usize {
        self.num_seqs
    }

    /// Most frequent symbol at a position.
    pub fn consensus_at(&self, pos: usize) -> Option<u8> {
        self.counts.get(pos).and_then(|m| {
            m.iter().max_by_key(|&(_, &v)| v).map(|(&k, _)| k)
        })
    }

    /// Consensus sequence (majority rule).
    pub fn consensus(&self) -> Vec<u8> {
        (0..self.width)
            .map(|p| self.consensus_at(p).unwrap_or(b'N'))
            .collect()
    }
}

impl fmt::Display for PosFreqMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PFM(width={}, depth={}, alphabet={})",
            self.width, self.num_seqs, self.alphabet
        )
    }
}

// ── Position Weight Matrix (PWM) ────────────────────────────────

/// Position Weight Matrix: log-odds scores relative to background.
#[derive(Debug, Clone)]
pub struct PosWeightMatrix {
    weights: Vec<HashMap<u8, f64>>,
    width: usize,
    alphabet: Alphabet,
    pseudocount: f64,
}

impl PosWeightMatrix {
    /// Build a PWM from a PFM with pseudocount smoothing.
    pub fn from_pfm(pfm: &PosFreqMatrix, pseudocount: f64) -> Result<Self, ConsensusError> {
        if pseudocount < 0.0 {
            return Err(ConsensusError::InvalidPseudocount(pseudocount));
        }

        let bg = pfm.alphabet.background();
        let alpha_size = pfm.alphabet.size() as f64;
        let total_pseudo = pseudocount * alpha_size;

        let mut weights = Vec::with_capacity(pfm.width);
        for pos in 0..pfm.width {
            let mut col = HashMap::new();
            let col_total = pfm.counts[pos].values().sum::<usize>() as f64 + total_pseudo;

            for &c in pfm.alphabet.chars() {
                let count = pfm.count(pos, c) as f64 + pseudocount;
                let freq = count / col_total;
                let log_odds = (freq / bg).ln() / std::f64::consts::LN_2;
                col.insert(c, log_odds);
            }
            weights.push(col);
        }

        Ok(Self {
            weights,
            width: pfm.width,
            alphabet: pfm.alphabet.clone(),
            pseudocount,
        })
    }

    /// Score a sequence against the PWM.
    pub fn score(&self, seq: &[u8]) -> Result<f64, ConsensusError> {
        if seq.len() < self.width {
            return Ok(f64::NEG_INFINITY);
        }
        let mut best = f64::NEG_INFINITY;
        for start in 0..=seq.len() - self.width {
            let s: f64 = (0..self.width)
                .map(|i| {
                    let c = seq[start + i].to_ascii_uppercase();
                    self.weights[i].get(&c).copied().unwrap_or(0.0)
                })
                .sum();
            if s > best {
                best = s;
            }
        }
        Ok(best)
    }

    /// Score at a specific position in a sequence.
    pub fn score_at(&self, seq: &[u8], start: usize) -> Result<f64, ConsensusError> {
        if start + self.width > seq.len() {
            return Err(ConsensusError::PositionOutOfRange(start));
        }
        let s: f64 = (0..self.width)
            .map(|i| {
                let c = seq[start + i].to_ascii_uppercase();
                self.weights[i].get(&c).copied().unwrap_or(0.0)
            })
            .sum();
        Ok(s)
    }

    /// Weight for symbol `c` at position `pos`.
    pub fn weight(&self, pos: usize, c: u8) -> f64 {
        self.weights
            .get(pos)
            .and_then(|m| m.get(&c.to_ascii_uppercase()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Width of the motif.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Max possible score.
    pub fn max_score(&self) -> f64 {
        self.weights
            .iter()
            .map(|col| {
                col.values()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum()
    }

    /// Min possible score.
    pub fn min_score(&self) -> f64 {
        self.weights
            .iter()
            .map(|col| {
                col.values()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
            })
            .sum()
    }
}

impl fmt::Display for PosWeightMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PWM(width={}, alphabet={}, pseudo={:.2})",
            self.width, self.alphabet, self.pseudocount
        )
    }
}

// ── Information Content ─────────────────────────────────────────

/// Compute per-position information content (bits).
pub fn information_content(pfm: &PosFreqMatrix) -> Vec<f64> {
    let max_bits = (pfm.alphabet.size() as f64).log2();
    let mut ic = Vec::with_capacity(pfm.width());

    for pos in 0..pfm.width() {
        let total: usize = pfm.counts[pos].values().sum();
        if total == 0 {
            ic.push(0.0);
            continue;
        }
        let entropy: f64 = pfm.alphabet.chars().iter().map(|c| {
            let cnt = pfm.count(pos, *c) as f64;
            if cnt == 0.0 { return 0.0; }
            let p = cnt / total as f64;
            -p * p.log2()
        }).sum();
        ic.push(max_bits - entropy);
    }

    ic
}

/// Total information content across all positions.
pub fn total_information(pfm: &PosFreqMatrix) -> f64 {
    information_content(pfm).iter().sum()
}

// ── Sequence Logo Heights ───────────────────────────────────────

/// Compute letter heights for a sequence logo at a given position.
pub fn logo_heights(
    pfm: &PosFreqMatrix,
    pos: usize,
) -> Result<Vec<(u8, f64)>, ConsensusError> {
    if pos >= pfm.width() {
        return Err(ConsensusError::PositionOutOfRange(pos));
    }

    let ic = information_content(pfm);
    let pos_ic = ic[pos];

    let total: usize = pfm.counts[pos].values().sum();
    if total == 0 {
        return Ok(Vec::new());
    }

    let mut heights: Vec<(u8, f64)> = pfm
        .alphabet
        .chars()
        .iter()
        .map(|c| {
            let cnt = pfm.count(pos, *c) as f64;
            let freq = cnt / total as f64;
            (*c, freq * pos_ic)
        })
        .filter(|&(_, h)| h > 0.0)
        .collect();

    heights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(heights)
}

// ── Consensus Config ────────────────────────────────────────────

/// Configuration for consensus building.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    alphabet: Alphabet,
    pseudocount: f64,
    threshold: f64,
    include_gaps: bool,
}

impl ConsensusConfig {
    pub fn new(alphabet: Alphabet) -> Self {
        Self {
            alphabet,
            pseudocount: 0.1,
            threshold: 0.5,
            include_gaps: false,
        }
    }

    pub fn with_pseudocount(mut self, p: f64) -> Self { self.pseudocount = p; self }
    pub fn with_threshold(mut self, t: f64) -> Self { self.threshold = t; self }
    pub fn with_gaps(mut self, g: bool) -> Self { self.include_gaps = g; self }
}

impl fmt::Display for ConsensusConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConsensusConfig(alphabet={}, pseudo={:.2}, thresh={:.2})",
            self.alphabet, self.pseudocount, self.threshold
        )
    }
}

// ── Consensus Builder ───────────────────────────────────────────

/// Builds consensus sequences from multiple alignments.
#[derive(Debug, Clone)]
pub struct ConsensusBuilder {
    config: ConsensusConfig,
}

impl ConsensusBuilder {
    pub fn new(config: ConsensusConfig) -> Self {
        Self { config }
    }

    /// Build consensus from aligned sequences.
    pub fn build(&self, sequences: &[&[u8]]) -> Result<ConsensusResult, ConsensusError> {
        let pfm = PosFreqMatrix::from_alignment(sequences, self.config.alphabet.clone())?;
        let pwm = PosWeightMatrix::from_pfm(&pfm, self.config.pseudocount)?;
        let consensus = self.threshold_consensus(&pfm);
        let ic = information_content(&pfm);
        let total_ic = ic.iter().sum();

        Ok(ConsensusResult {
            consensus,
            pfm,
            pwm,
            information_content: ic,
            total_information: total_ic,
        })
    }

    /// Consensus with threshold: ambiguous positions get 'N'.
    fn threshold_consensus(&self, pfm: &PosFreqMatrix) -> Vec<u8> {
        (0..pfm.width())
            .map(|pos| {
                let total: usize = pfm.counts[pos].values().sum();
                if total == 0 {
                    return b'N';
                }
                if let Some((&best_c, &best_cnt)) =
                    pfm.counts[pos].iter().max_by_key(|&(_, &v)| v)
                {
                    let freq = best_cnt as f64 / total as f64;
                    if freq >= self.config.threshold {
                        best_c
                    } else {
                        b'N'
                    }
                } else {
                    b'N'
                }
            })
            .collect()
    }
}

impl fmt::Display for ConsensusBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConsensusBuilder({})", self.config)
    }
}

// ── Consensus Result ────────────────────────────────────────────

/// Result of consensus building.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub consensus: Vec<u8>,
    pub pfm: PosFreqMatrix,
    pub pwm: PosWeightMatrix,
    pub information_content: Vec<f64>,
    pub total_information: f64,
}

impl ConsensusResult {
    /// Consensus as a string.
    pub fn consensus_string(&self) -> String {
        String::from_utf8_lossy(&self.consensus).to_string()
    }

    /// Width of the motif.
    pub fn width(&self) -> usize {
        self.consensus.len()
    }
}

impl fmt::Display for ConsensusResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Consensus({}, width={}, IC={:.2} bits)",
            self.consensus_string(),
            self.width(),
            self.total_information
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dna_seqs() -> Vec<Vec<u8>> {
        vec![
            b"ACGT".to_vec(),
            b"ACGT".to_vec(),
            b"ACGT".to_vec(),
            b"ACGA".to_vec(),
        ]
    }

    #[test]
    fn pfm_from_alignment() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        assert_eq!(pfm.width(), 4);
        assert_eq!(pfm.depth(), 4);
    }

    #[test]
    fn pfm_counts() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        assert_eq!(pfm.count(0, b'A'), 4);
        assert_eq!(pfm.count(3, b'T'), 3);
        assert_eq!(pfm.count(3, b'A'), 1);
    }

    #[test]
    fn pfm_consensus() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let cons = pfm.consensus();
        assert_eq!(cons, b"ACGT");
    }

    #[test]
    fn pwm_from_pfm() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let pwm = PosWeightMatrix::from_pfm(&pfm, 0.1).unwrap();
        assert_eq!(pwm.width(), 4);
    }

    #[test]
    fn pwm_score_self() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let pwm = PosWeightMatrix::from_pfm(&pfm, 0.1).unwrap();
        let score = pwm.score(b"ACGT").unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn pwm_score_at() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let pwm = PosWeightMatrix::from_pfm(&pfm, 0.1).unwrap();
        let score = pwm.score_at(b"ACGT", 0).unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn pwm_max_min_score() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let pwm = PosWeightMatrix::from_pfm(&pfm, 0.1).unwrap();
        assert!(pwm.max_score() > pwm.min_score());
    }

    #[test]
    fn information_content_positive() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let ic = information_content(&pfm);
        assert!(ic.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn perfect_conservation_max_ic() {
        let seqs = vec![b"AAAA".to_vec(); 10];
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let ic = information_content(&pfm);
        // Max IC for DNA is 2.0 bits
        assert!((ic[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn logo_heights_sorted() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        let heights = logo_heights(&pfm, 0).unwrap();
        assert!(!heights.is_empty());
        // Should be sorted descending by height
        for w in heights.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn consensus_builder_basic() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let cb = ConsensusBuilder::new(ConsensusConfig::new(Alphabet::Dna));
        let result = cb.build(&refs).unwrap();
        assert_eq!(result.width(), 4);
        assert!(result.total_information > 0.0);
    }

    #[test]
    fn consensus_string() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let cb = ConsensusBuilder::new(ConsensusConfig::new(Alphabet::Dna));
        let result = cb.build(&refs).unwrap();
        assert_eq!(result.consensus_string(), "ACGT");
    }

    #[test]
    fn empty_alignment_err() {
        let cb = ConsensusBuilder::new(ConsensusConfig::new(Alphabet::Dna));
        assert!(cb.build(&[]).is_err());
    }

    #[test]
    fn unequal_lengths_err() {
        let cb = ConsensusBuilder::new(ConsensusConfig::new(Alphabet::Dna));
        assert!(cb.build(&[b"AC", b"ACG"]).is_err());
    }

    #[test]
    fn negative_pseudocount_err() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        assert!(PosWeightMatrix::from_pfm(&pfm, -1.0).is_err());
    }

    #[test]
    fn alphabet_display() {
        assert_eq!(format!("{}", Alphabet::Dna), "DNA");
        assert_eq!(format!("{}", Alphabet::Protein), "Protein");
    }

    #[test]
    fn config_builder() {
        let cfg = ConsensusConfig::new(Alphabet::Dna)
            .with_pseudocount(0.5)
            .with_threshold(0.7)
            .with_gaps(true);
        assert!(format!("{cfg}").contains("pseudo=0.50"));
    }

    #[test]
    fn result_display() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let cb = ConsensusBuilder::new(ConsensusConfig::new(Alphabet::Dna));
        let result = cb.build(&refs).unwrap();
        assert!(format!("{result}").contains("Consensus("));
    }

    #[test]
    fn builder_display() {
        let cb = ConsensusBuilder::new(ConsensusConfig::new(Alphabet::Rna));
        assert!(format!("{cb}").contains("ConsensusBuilder("));
    }

    #[test]
    fn total_information_positive() {
        let seqs = dna_seqs();
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        let pfm = PosFreqMatrix::from_alignment(&refs, Alphabet::Dna).unwrap();
        assert!(total_information(&pfm) > 0.0);
    }
}
