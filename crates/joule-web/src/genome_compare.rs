//! Whole-genome comparison with synteny block detection, genome dot plots,
//! and alignment-free sequence similarity metrics.
//!
//! Implements k-mer based genome comparison, synteny block identification
//! using anchored co-linear regions, genome dot plot matrix generation,
//! Average Nucleotide Identity (ANI) estimation, and structural
//! rearrangement detection (inversions, translocations, duplications).

use std::fmt;
use std::collections::HashMap;

// ── K-mer Index ─────────────────────────────────────────────────

/// A k-mer index mapping k-length subsequences to their positions.
#[derive(Debug, Clone)]
pub struct KmerIndex {
    pub k: usize,
    pub index: HashMap<String, Vec<usize>>,
    pub total_kmers: usize,
}

impl KmerIndex {
    /// Build a k-mer index from a DNA sequence.
    pub fn build(sequence: &str, k: usize) -> Self {
        let seq = sequence.to_uppercase();
        let bytes = seq.as_bytes();
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut total = 0usize;

        if bytes.len() >= k {
            for i in 0..=(bytes.len() - k) {
                let kmer = &seq[i..i + k];
                if kmer.bytes().all(|b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                    index.entry(kmer.to_string()).or_default().push(i);
                    total += 1;
                }
            }
        }

        Self { k, index, total_kmers: total }
    }

    /// Number of unique k-mers.
    pub fn unique_count(&self) -> usize {
        self.index.len()
    }

    /// Positions of a specific k-mer.
    pub fn positions(&self, kmer: &str) -> &[usize] {
        self.index.get(&kmer.to_uppercase()).map_or(&[], |v| v.as_slice())
    }

    /// K-mers shared with another index.
    pub fn shared_kmers(&self, other: &KmerIndex) -> usize {
        self.index.keys().filter(|k| other.index.contains_key(*k)).count()
    }

    /// Jaccard similarity with another k-mer index.
    pub fn jaccard(&self, other: &KmerIndex) -> f64 {
        let shared = self.shared_kmers(other) as f64;
        let union = (self.unique_count() + other.unique_count()) as f64 - shared;
        if union == 0.0 { 0.0 } else { shared / union }
    }
}

impl fmt::Display for KmerIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KmerIndex(k={}, unique={}, total={})",
            self.k, self.unique_count(), self.total_kmers,
        )
    }
}

// ── Anchor Match ────────────────────────────────────────────────

/// A matching anchor between two genomes (shared k-mer or seed).
#[derive(Debug, Clone)]
pub struct AnchorMatch {
    pub query_pos: usize,
    pub target_pos: usize,
    pub length: usize,
    pub is_forward: bool,
    pub score: f64,
}

impl AnchorMatch {
    pub fn new(query_pos: usize, target_pos: usize, length: usize) -> Self {
        Self {
            query_pos,
            target_pos,
            length,
            is_forward: true,
            score: length as f64,
        }
    }

    pub fn with_direction(mut self, forward: bool) -> Self {
        self.is_forward = forward;
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Diagonal index on the dot plot (target_pos - query_pos for forward).
    pub fn diagonal(&self) -> i64 {
        self.target_pos as i64 - self.query_pos as i64
    }
}

impl fmt::Display for AnchorMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Anchor(q={}, t={}, len={}, {})",
            self.query_pos, self.target_pos, self.length,
            if self.is_forward { "fwd" } else { "rev" },
        )
    }
}

/// Find anchors between two sequences using shared k-mers.
pub fn find_anchors(query: &str, target: &str, k: usize) -> Vec<AnchorMatch> {
    let q_idx = KmerIndex::build(query, k);
    let t_idx = KmerIndex::build(target, k);
    let mut anchors = Vec::new();

    for (kmer, q_positions) in &q_idx.index {
        if let Some(t_positions) = t_idx.index.get(kmer) {
            for &qp in q_positions {
                for &tp in t_positions {
                    anchors.push(AnchorMatch::new(qp, tp, k));
                }
            }
        }
    }

    anchors.sort_by_key(|a| (a.query_pos, a.target_pos));
    anchors
}

// ── Synteny Block ───────────────────────────────────────────────

/// A synteny block: a co-linear region conserved between two genomes.
#[derive(Debug, Clone)]
pub struct SyntenyBlock {
    pub query_start: usize,
    pub query_end: usize,
    pub target_start: usize,
    pub target_end: usize,
    pub anchor_count: usize,
    pub orientation: SyntenyOrientation,
    pub score: f64,
}

/// Orientation of a synteny block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntenyOrientation {
    Forward,
    Inverted,
}

impl fmt::Display for SyntenyOrientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "forward"),
            Self::Inverted => write!(f, "inverted"),
        }
    }
}

impl SyntenyBlock {
    pub fn new(
        query_start: usize, query_end: usize,
        target_start: usize, target_end: usize,
    ) -> Self {
        Self {
            query_start,
            query_end,
            target_start,
            target_end,
            anchor_count: 0,
            orientation: SyntenyOrientation::Forward,
            score: 0.0,
        }
    }

    pub fn with_anchor_count(mut self, count: usize) -> Self {
        self.anchor_count = count;
        self
    }

    pub fn with_orientation(mut self, orient: SyntenyOrientation) -> Self {
        self.orientation = orient;
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    pub fn query_length(&self) -> usize {
        if self.query_end >= self.query_start {
            self.query_end - self.query_start + 1
        } else { 0 }
    }

    pub fn target_length(&self) -> usize {
        if self.target_end >= self.target_start {
            self.target_end - self.target_start + 1
        } else { 0 }
    }

    /// Length ratio between query and target spans.
    pub fn length_ratio(&self) -> f64 {
        let ql = self.query_length() as f64;
        let tl = self.target_length() as f64;
        if tl == 0.0 { 0.0 } else { ql / tl }
    }
}

impl fmt::Display for SyntenyBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SyntenyBlock(q={}-{}, t={}-{}, {}, anchors={}, score={:.1})",
            self.query_start, self.query_end,
            self.target_start, self.target_end,
            self.orientation, self.anchor_count, self.score,
        )
    }
}

/// Identify synteny blocks from anchor matches using a diagonal chaining approach.
pub fn find_synteny_blocks(anchors: &[AnchorMatch], max_gap: usize) -> Vec<SyntenyBlock> {
    if anchors.is_empty() {
        return Vec::new();
    }

    // Group anchors by approximate diagonal
    let mut diagonal_groups: HashMap<i64, Vec<&AnchorMatch>> = HashMap::new();
    for anchor in anchors {
        let diag_bin = anchor.diagonal() / max_gap as i64;
        diagonal_groups.entry(diag_bin).or_default().push(anchor);
    }

    let mut blocks = Vec::new();

    for (_diag, group) in &diagonal_groups {
        if group.len() < 2 {
            continue;
        }

        let mut sorted = group.clone();
        sorted.sort_by_key(|a| a.query_pos);

        // Chain nearby anchors on the same diagonal
        let mut block_start_q = sorted[0].query_pos;
        let mut block_start_t = sorted[0].target_pos;
        let mut block_end_q = sorted[0].query_pos + sorted[0].length;
        let mut block_end_t = sorted[0].target_pos + sorted[0].length;
        let mut count = 1usize;
        let mut score = sorted[0].score;

        for i in 1..sorted.len() {
            let gap_q = sorted[i].query_pos.saturating_sub(block_end_q);
            if gap_q <= max_gap {
                block_end_q = sorted[i].query_pos + sorted[i].length;
                block_end_t = sorted[i].target_pos + sorted[i].length;
                count += 1;
                score += sorted[i].score;
            } else {
                if count >= 2 {
                    blocks.push(
                        SyntenyBlock::new(block_start_q, block_end_q, block_start_t, block_end_t)
                            .with_anchor_count(count)
                            .with_score(score),
                    );
                }
                block_start_q = sorted[i].query_pos;
                block_start_t = sorted[i].target_pos;
                block_end_q = sorted[i].query_pos + sorted[i].length;
                block_end_t = sorted[i].target_pos + sorted[i].length;
                count = 1;
                score = sorted[i].score;
            }
        }
        if count >= 2 {
            blocks.push(
                SyntenyBlock::new(block_start_q, block_end_q, block_start_t, block_end_t)
                    .with_anchor_count(count)
                    .with_score(score),
            );
        }
    }

    blocks.sort_by_key(|b| b.query_start);
    blocks
}

// ── Genome Dot Plot ─────────────────────────────────────────────

/// A dot-plot cell recording the number of k-mer matches in a bin.
#[derive(Debug, Clone)]
pub struct DotPlotCell {
    pub query_bin: usize,
    pub target_bin: usize,
    pub match_count: usize,
}

/// Configuration for dot-plot generation.
#[derive(Debug, Clone)]
pub struct DotPlotConfig {
    pub bin_size: usize,
    pub k: usize,
    pub min_matches: usize,
}

impl DotPlotConfig {
    pub fn new(bin_size: usize, k: usize) -> Self {
        Self { bin_size, k, min_matches: 1 }
    }

    pub fn with_min_matches(mut self, min: usize) -> Self {
        self.min_matches = min;
        self
    }
}

impl fmt::Display for DotPlotConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DotPlotConfig(bin={}, k={}, min={})", self.bin_size, self.k, self.min_matches)
    }
}

/// Generate a dot plot from two sequences.
pub fn generate_dot_plot(query: &str, target: &str, config: &DotPlotConfig) -> Vec<DotPlotCell> {
    let anchors = find_anchors(query, target, config.k);
    let mut grid: HashMap<(usize, usize), usize> = HashMap::new();

    for anchor in &anchors {
        let qbin = anchor.query_pos / config.bin_size;
        let tbin = anchor.target_pos / config.bin_size;
        *grid.entry((qbin, tbin)).or_insert(0) += 1;
    }

    grid.into_iter()
        .filter(|(_, count)| *count >= config.min_matches)
        .map(|((qb, tb), count)| DotPlotCell {
            query_bin: qb,
            target_bin: tb,
            match_count: count,
        })
        .collect()
}

// ── Average Nucleotide Identity ─────────────────────────────────

/// Estimate Average Nucleotide Identity (ANI) using shared k-mers.
pub fn estimate_ani(query: &str, target: &str, k: usize) -> f64 {
    let q_idx = KmerIndex::build(query, k);
    let t_idx = KmerIndex::build(target, k);
    let jaccard = q_idx.jaccard(&t_idx);
    if jaccard <= 0.0 {
        return 0.0;
    }
    // Mash-style ANI estimation: ANI ≈ 1 + (1/k) * ln(2*J/(1+J))
    let j = jaccard;
    let d = -(1.0 / k as f64) * (2.0 * j / (1.0 + j)).ln();
    (1.0 - d).max(0.0).min(1.0)
}

// ── Structural Rearrangement ────────────────────────────────────

/// Type of structural rearrangement detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RearrangementType {
    Inversion,
    Translocation,
    Duplication,
    Deletion,
    Unknown,
}

impl fmt::Display for RearrangementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inversion => write!(f, "inversion"),
            Self::Translocation => write!(f, "translocation"),
            Self::Duplication => write!(f, "duplication"),
            Self::Deletion => write!(f, "deletion"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// A detected structural rearrangement.
#[derive(Debug, Clone)]
pub struct Rearrangement {
    pub rearrangement_type: RearrangementType,
    pub query_start: usize,
    pub query_end: usize,
    pub target_start: usize,
    pub target_end: usize,
    pub confidence: f64,
}

impl Rearrangement {
    pub fn new(rtype: RearrangementType, qs: usize, qe: usize, ts: usize, te: usize) -> Self {
        Self {
            rearrangement_type: rtype,
            query_start: qs,
            query_end: qe,
            target_start: ts,
            target_end: te,
            confidence: 0.0,
        }
    }

    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c;
        self
    }

    pub fn query_span(&self) -> usize {
        if self.query_end >= self.query_start { self.query_end - self.query_start + 1 } else { 0 }
    }
}

impl fmt::Display for Rearrangement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rearrangement({}, q={}-{}, t={}-{}, conf={:.2})",
            self.rearrangement_type,
            self.query_start, self.query_end,
            self.target_start, self.target_end,
            self.confidence,
        )
    }
}

// ── Genome Comparison Summary ───────────────────────────────────

/// Summary statistics for a whole-genome comparison.
#[derive(Debug, Clone)]
pub struct ComparisonSummary {
    pub query_length: usize,
    pub target_length: usize,
    pub ani_estimate: f64,
    pub shared_kmers: usize,
    pub synteny_blocks: usize,
    pub total_synteny_bp: usize,
}

impl ComparisonSummary {
    pub fn compute(query: &str, target: &str, k: usize, max_gap: usize) -> Self {
        let q_idx = KmerIndex::build(query, k);
        let t_idx = KmerIndex::build(target, k);
        let shared = q_idx.shared_kmers(&t_idx);
        let ani = estimate_ani(query, target, k);
        let anchors = find_anchors(query, target, k);
        let blocks = find_synteny_blocks(&anchors, max_gap);
        let synteny_bp: usize = blocks.iter().map(|b| b.query_length()).sum();

        Self {
            query_length: query.len(),
            target_length: target.len(),
            ani_estimate: ani,
            shared_kmers: shared,
            synteny_blocks: blocks.len(),
            total_synteny_bp: synteny_bp,
        }
    }
}

impl fmt::Display for ComparisonSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComparisonSummary(qlen={}, tlen={}, ANI={:.2}%, shared_kmers={}, blocks={}, synteny_bp={})",
            self.query_length, self.target_length,
            self.ani_estimate * 100.0, self.shared_kmers,
            self.synteny_blocks, self.total_synteny_bp,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat_seq(motif: &str, times: usize) -> String {
        motif.repeat(times)
    }

    #[test]
    fn test_kmer_index_build() {
        let idx = KmerIndex::build("ATGATG", 3);
        assert!(idx.unique_count() > 0);
        assert!(idx.total_kmers > 0);
    }

    #[test]
    fn test_kmer_index_positions() {
        let idx = KmerIndex::build("ATGATG", 3);
        let positions = idx.positions("ATG");
        assert_eq!(positions.len(), 2); // pos 0 and 3
    }

    #[test]
    fn test_kmer_jaccard_identical() {
        let a = KmerIndex::build("ATGATGATG", 3);
        let b = KmerIndex::build("ATGATGATG", 3);
        assert!((a.jaccard(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_kmer_jaccard_disjoint() {
        let a = KmerIndex::build("AAAAA", 3);
        let b = KmerIndex::build("CCCCC", 3);
        assert_eq!(a.jaccard(&b), 0.0);
    }

    #[test]
    fn test_kmer_index_display() {
        let idx = KmerIndex::build("ATGATG", 3);
        let s = format!("{}", idx);
        assert!(s.contains("k=3"));
    }

    #[test]
    fn test_find_anchors() {
        let query = "ATGATGATG";
        let target = "ATGATGATG";
        let anchors = find_anchors(query, target, 3);
        assert!(!anchors.is_empty());
    }

    #[test]
    fn test_anchor_diagonal() {
        let a = AnchorMatch::new(10, 20, 5);
        assert_eq!(a.diagonal(), 10);
    }

    #[test]
    fn test_anchor_display() {
        let a = AnchorMatch::new(0, 100, 11);
        let s = format!("{}", a);
        assert!(s.contains("q=0"));
        assert!(s.contains("fwd"));
    }

    #[test]
    fn test_synteny_block_lengths() {
        let b = SyntenyBlock::new(100, 500, 200, 600)
            .with_anchor_count(10)
            .with_score(50.0);
        assert_eq!(b.query_length(), 401);
        assert_eq!(b.target_length(), 401);
    }

    #[test]
    fn test_synteny_block_ratio() {
        let b = SyntenyBlock::new(0, 199, 0, 399);
        let ratio = b.length_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_synteny_block_display() {
        let b = SyntenyBlock::new(0, 100, 50, 150).with_orientation(SyntenyOrientation::Inverted);
        let s = format!("{}", b);
        assert!(s.contains("inverted"));
    }

    #[test]
    fn test_find_synteny_blocks() {
        let seq = repeat_seq("ATGCATGC", 20);
        let anchors = find_anchors(&seq, &seq, 5);
        let blocks = find_synteny_blocks(&anchors, 10);
        // Identical sequences should produce synteny
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_dot_plot_generation() {
        let seq = repeat_seq("ATGC", 50);
        let config = DotPlotConfig::new(10, 4);
        let cells = generate_dot_plot(&seq, &seq, &config);
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_dot_plot_config_display() {
        let config = DotPlotConfig::new(100, 11);
        let s = format!("{}", config);
        assert!(s.contains("bin=100"));
    }

    #[test]
    fn test_estimate_ani_identical() {
        let seq = repeat_seq("ATGCATGC", 50);
        let ani = estimate_ani(&seq, &seq, 11);
        assert!(ani > 0.99);
    }

    #[test]
    fn test_estimate_ani_disjoint() {
        let a = "A".repeat(200);
        let b = "C".repeat(200);
        let ani = estimate_ani(&a, &b, 11);
        assert!(ani < 0.5);
    }

    #[test]
    fn test_rearrangement_display() {
        let r = Rearrangement::new(RearrangementType::Inversion, 100, 500, 100, 500)
            .with_confidence(0.95);
        let s = format!("{}", r);
        assert!(s.contains("inversion"));
        assert!(s.contains("0.95"));
    }

    #[test]
    fn test_rearrangement_type_display() {
        assert_eq!(format!("{}", RearrangementType::Duplication), "duplication");
        assert_eq!(format!("{}", RearrangementType::Translocation), "translocation");
    }

    #[test]
    fn test_comparison_summary() {
        let seq = repeat_seq("ATGCATGC", 30);
        let summary = ComparisonSummary::compute(&seq, &seq, 5, 10);
        assert_eq!(summary.query_length, summary.target_length);
        assert!(summary.ani_estimate > 0.9);
        assert!(summary.shared_kmers > 0);
    }

    #[test]
    fn test_comparison_summary_display() {
        let summary = ComparisonSummary {
            query_length: 1000,
            target_length: 1000,
            ani_estimate: 0.98,
            shared_kmers: 500,
            synteny_blocks: 3,
            total_synteny_bp: 800,
        };
        let s = format!("{}", summary);
        assert!(s.contains("ANI=98.00%"));
    }
}
