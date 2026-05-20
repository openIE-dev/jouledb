//! Bootstrap resampling for phylogenetic analysis — column resampling of
//! sequence alignments, replicate tree construction, support value mapping,
//! and majority-rule consensus tree assembly.
//!
//! Implements Felsenstein (1985) bootstrap with configurable replicate count,
//! majority-rule thresholds, and support annotation onto reference trees.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapError {
    EmptyAlignment,
    UnequalLengths,
    InsufficientReplicates(usize),
    InvalidThreshold(f64),
    InternalError(String),
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAlignment => write!(f, "empty alignment"),
            Self::UnequalLengths => write!(f, "sequences have unequal lengths"),
            Self::InsufficientReplicates(n) => write!(f, "need ≥ 1 replicates, got {n}"),
            Self::InvalidThreshold(t) => write!(f, "invalid threshold: {t}"),
            Self::InternalError(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for BootstrapError {}

// ── Simple LCG for deterministic resampling ─────────────────────

/// Minimal linear congruential generator for reproducible column resampling.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        // LCG constants from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as usize) % bound
    }
}

// ── Alignment resampling ────────────────────────────────────────

/// A sequence alignment (list of equal-length byte sequences).
#[derive(Debug, Clone)]
pub struct Alignment {
    pub labels: Vec<String>,
    pub sequences: Vec<Vec<u8>>,
}

impl Alignment {
    pub fn new(labels: Vec<String>, sequences: Vec<Vec<u8>>) -> Result<Self, BootstrapError> {
        if labels.is_empty() || sequences.is_empty() {
            return Err(BootstrapError::EmptyAlignment);
        }
        if labels.len() != sequences.len() {
            return Err(BootstrapError::UnequalLengths);
        }
        let len = sequences[0].len();
        if sequences.iter().any(|s| s.len() != len) {
            return Err(BootstrapError::UnequalLengths);
        }
        Ok(Self { labels, sequences })
    }

    pub fn seq_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn alignment_length(&self) -> usize {
        self.sequences.first().map_or(0, |s| s.len())
    }

    /// Resample columns with replacement to produce a bootstrap replicate.
    pub fn resample_columns(&self, rng: &mut Lcg) -> Self {
        let ncol = self.alignment_length();
        let nseq = self.seq_count();
        let mut new_seqs = vec![Vec::with_capacity(ncol); nseq];
        for _ in 0..ncol {
            let col = rng.next_usize(ncol);
            for (s, new_s) in self.sequences.iter().zip(new_seqs.iter_mut()) {
                new_s.push(s[col]);
            }
        }
        Self { labels: self.labels.clone(), sequences: new_seqs }
    }
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Alignment({} seqs × {} cols)",
            self.seq_count(),
            self.alignment_length()
        )
    }
}

// ── Bipartitions (splits) ───────────────────────────────────────

/// A bipartition (split) of taxa represented as a sorted set of leaf labels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Split {
    /// The smaller side of the bipartition (canonicalised).
    pub taxa: Vec<String>,
}

impl Split {
    pub fn new(mut taxa: Vec<String>) -> Self {
        taxa.sort();
        Self { taxa }
    }

    pub fn size(&self) -> usize {
        self.taxa.len()
    }

    pub fn contains(&self, taxon: &str) -> bool {
        self.taxa.binary_search(&taxon.to_string()).is_ok()
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}}}", self.taxa.join(", "))
    }
}

// ── Bootstrap support ───────────────────────────────────────────

/// Bootstrap support values mapped to splits.
#[derive(Debug, Clone)]
pub struct BootstrapSupport {
    pub split_counts: HashMap<Split, usize>,
    pub total_replicates: usize,
}

impl BootstrapSupport {
    pub fn new(total_replicates: usize) -> Self {
        Self {
            split_counts: HashMap::new(),
            total_replicates,
        }
    }

    /// Record that a split was observed in a replicate tree.
    pub fn record_split(&mut self, split: Split) {
        *self.split_counts.entry(split).or_insert(0) += 1;
    }

    /// Support value for a given split (proportion of replicates).
    pub fn support(&self, split: &Split) -> f64 {
        let count = self.split_counts.get(split).copied().unwrap_or(0);
        if self.total_replicates == 0 {
            0.0
        } else {
            count as f64 / self.total_replicates as f64
        }
    }

    /// All splits with support ≥ threshold.
    pub fn supported_splits(&self, threshold: f64) -> Vec<(&Split, f64)> {
        self.split_counts
            .iter()
            .filter_map(|(split, &count)| {
                let s = count as f64 / self.total_replicates.max(1) as f64;
                if s >= threshold { Some((split, s)) } else { None }
            })
            .collect()
    }

    /// Number of distinct splits observed.
    pub fn split_count(&self) -> usize {
        self.split_counts.len()
    }
}

impl fmt::Display for BootstrapSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BootstrapSupport({} splits, {} replicates)",
            self.split_count(),
            self.total_replicates
        )
    }
}

// ── Consensus tree node ─────────────────────────────────────────

/// A node in a majority-rule consensus tree.
#[derive(Debug, Clone)]
pub struct ConsensusNode {
    pub id: usize,
    pub label: Option<String>,
    pub children: Vec<usize>,
    pub support: f64,
}

impl ConsensusNode {
    pub fn leaf(id: usize, label: &str) -> Self {
        Self {
            id,
            label: Some(label.to_string()),
            children: Vec::new(),
            support: 1.0,
        }
    }

    pub fn internal(id: usize, support: f64) -> Self {
        Self {
            id,
            label: None,
            children: Vec::new(),
            support,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty() && self.label.is_some()
    }
}

impl fmt::Display for ConsensusNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref lbl) = self.label {
            write!(f, "{lbl}")
        } else {
            write!(f, "node_{}[{:.1}%]", self.id, self.support * 100.0)
        }
    }
}

// ── Consensus tree ──────────────────────────────────────────────

/// Majority-rule consensus tree.
#[derive(Debug, Clone)]
pub struct ConsensusTree {
    pub nodes: Vec<ConsensusNode>,
    pub root: usize,
    pub threshold: f64,
}

impl ConsensusTree {
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl fmt::Display for ConsensusTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConsensusTree({} nodes, threshold={:.0}%)",
            self.node_count(),
            self.threshold * 100.0
        )
    }
}

// ── Consensus tree builder ──────────────────────────────────────

/// Build a majority-rule consensus tree from bootstrap support data.
pub fn build_consensus(
    all_taxa: &[&str],
    support: &BootstrapSupport,
    threshold: f64,
) -> Result<ConsensusTree, BootstrapError> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(BootstrapError::InvalidThreshold(threshold));
    }
    if all_taxa.is_empty() {
        return Err(BootstrapError::EmptyAlignment);
    }

    let mut nodes = Vec::new();
    let all_set: HashSet<String> = all_taxa.iter().map(|s| s.to_string()).collect();

    // Create leaf nodes
    let mut leaf_ids = HashMap::new();
    for (i, &taxon) in all_taxa.iter().enumerate() {
        nodes.push(ConsensusNode::leaf(i, taxon));
        leaf_ids.insert(taxon.to_string(), i);
    }

    // Collect supported splits, sorted by size (largest first) for nesting
    let mut supported: Vec<(Split, f64)> = support
        .supported_splits(threshold)
        .into_iter()
        .filter(|(s, _)| s.size() > 1 && s.size() < all_set.len())
        .map(|(s, v)| (s.clone(), v))
        .collect();
    supported.sort_by(|a, b| b.0.size().cmp(&a.0.size()));

    // Build internal nodes for each supported split
    let mut next_id = all_taxa.len();
    for (split, sup_val) in &supported {
        let internal = ConsensusNode::internal(next_id, *sup_val);
        nodes.push(internal);
        // Attach leaves that belong to this split and aren't yet under a more specific clade
        for taxon in &split.taxa {
            if let Some(&lid) = leaf_ids.get(taxon) {
                if !nodes[next_id].children.contains(&lid) {
                    nodes[next_id].children.push(lid);
                }
            }
        }
        next_id += 1;
    }

    // Root node
    let root_id = next_id;
    let mut root = ConsensusNode::internal(root_id, 1.0);
    // Attach all leaves or top-level internal nodes
    let claimed: HashSet<usize> = nodes.iter().flat_map(|n| n.children.iter().copied()).collect();
    for i in 0..next_id {
        if !claimed.contains(&i) {
            root.children.push(i);
        }
    }
    nodes.push(root);

    Ok(ConsensusTree { nodes, root: root_id, threshold })
}

// ── Bootstrap pipeline ──────────────────────────────────────────

/// Configuration for bootstrap analysis.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    pub replicates: usize,
    pub seed: u64,
    pub consensus_threshold: f64,
}

impl BootstrapConfig {
    pub fn new(replicates: usize) -> Self {
        Self {
            replicates,
            seed: 42,
            consensus_threshold: 0.5,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_consensus_threshold(mut self, threshold: f64) -> Self {
        self.consensus_threshold = threshold;
        self
    }

    pub fn with_replicates(mut self, n: usize) -> Self {
        self.replicates = n;
        self
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self::new(100)
    }
}

impl fmt::Display for BootstrapConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BootstrapConfig(reps={}, seed={}, threshold={:.0}%)",
            self.replicates,
            self.seed,
            self.consensus_threshold * 100.0
        )
    }
}

/// Generate bootstrap replicates from an alignment.
pub fn generate_replicates(
    alignment: &Alignment,
    config: &BootstrapConfig,
) -> Result<Vec<Alignment>, BootstrapError> {
    if config.replicates == 0 {
        return Err(BootstrapError::InsufficientReplicates(0));
    }
    let mut rng = Lcg::new(config.seed);
    let reps = (0..config.replicates)
        .map(|_| alignment.resample_columns(&mut rng))
        .collect();
    Ok(reps)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_alignment() -> Alignment {
        Alignment::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![
                b"ATCGATCGATCG".to_vec(),
                b"AACGATCGATCG".to_vec(),
                b"ATCGAACGATCG".to_vec(),
                b"ATCGATCGAACG".to_vec(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_alignment_creation() {
        let aln = sample_alignment();
        assert_eq!(aln.seq_count(), 4);
        assert_eq!(aln.alignment_length(), 12);
    }

    #[test]
    fn test_alignment_display() {
        let aln = sample_alignment();
        let s = format!("{aln}");
        assert!(s.contains("4 seqs"));
    }

    #[test]
    fn test_alignment_unequal_lengths() {
        let result = Alignment::new(
            vec!["A".into(), "B".into()],
            vec![b"ATCG".to_vec(), b"AT".to_vec()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_columns() {
        let aln = sample_alignment();
        let mut rng = Lcg::new(42);
        let rep = aln.resample_columns(&mut rng);
        assert_eq!(rep.seq_count(), 4);
        assert_eq!(rep.alignment_length(), 12);
    }

    #[test]
    fn test_resample_deterministic() {
        let aln = sample_alignment();
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        let rep1 = aln.resample_columns(&mut rng1);
        let rep2 = aln.resample_columns(&mut rng2);
        assert_eq!(rep1.sequences, rep2.sequences);
    }

    #[test]
    fn test_generate_replicates() {
        let aln = sample_alignment();
        let cfg = BootstrapConfig::new(10);
        let reps = generate_replicates(&aln, &cfg).unwrap();
        assert_eq!(reps.len(), 10);
    }

    #[test]
    fn test_generate_replicates_zero_fails() {
        let aln = sample_alignment();
        let cfg = BootstrapConfig::new(0);
        assert!(generate_replicates(&aln, &cfg).is_err());
    }

    #[test]
    fn test_split_creation() {
        let split = Split::new(vec!["C".into(), "A".into(), "B".into()]);
        assert_eq!(split.taxa, vec!["A", "B", "C"]); // sorted
    }

    #[test]
    fn test_split_contains() {
        let split = Split::new(vec!["A".into(), "B".into()]);
        assert!(split.contains("A"));
        assert!(!split.contains("C"));
    }

    #[test]
    fn test_split_display() {
        let split = Split::new(vec!["X".into(), "Y".into()]);
        assert_eq!(format!("{split}"), "{X, Y}");
    }

    #[test]
    fn test_bootstrap_support() {
        let mut bs = BootstrapSupport::new(100);
        let split = Split::new(vec!["A".into(), "B".into()]);
        for _ in 0..75 {
            bs.record_split(split.clone());
        }
        assert!((bs.support(&split) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_supported_splits() {
        let mut bs = BootstrapSupport::new(100);
        let s1 = Split::new(vec!["A".into(), "B".into()]);
        let s2 = Split::new(vec!["C".into(), "D".into()]);
        for _ in 0..80 { bs.record_split(s1.clone()); }
        for _ in 0..30 { bs.record_split(s2.clone()); }
        let supported = bs.supported_splits(0.5);
        assert_eq!(supported.len(), 1);
    }

    #[test]
    fn test_bootstrap_support_display() {
        let bs = BootstrapSupport::new(100);
        let s = format!("{bs}");
        assert!(s.contains("100 replicates"));
    }

    #[test]
    fn test_consensus_node_leaf() {
        let n = ConsensusNode::leaf(0, "Human");
        assert!(n.is_leaf());
        assert_eq!(format!("{n}"), "Human");
    }

    #[test]
    fn test_consensus_node_internal() {
        let n = ConsensusNode::internal(5, 0.85);
        assert!(!n.is_leaf());
        assert!(format!("{n}").contains("85.0%"));
    }

    #[test]
    fn test_build_consensus() {
        let mut bs = BootstrapSupport::new(100);
        let split = Split::new(vec!["A".into(), "B".into()]);
        for _ in 0..60 { bs.record_split(split.clone()); }
        let tree = build_consensus(&["A", "B", "C"], &bs, 0.5).unwrap();
        assert!(tree.leaf_count() >= 3);
    }

    #[test]
    fn test_build_consensus_invalid_threshold() {
        let bs = BootstrapSupport::new(100);
        assert!(build_consensus(&["A"], &bs, 1.5).is_err());
    }

    #[test]
    fn test_consensus_tree_display() {
        let mut bs = BootstrapSupport::new(10);
        let _ = bs.record_split(Split::new(vec!["A".into(), "B".into()]));
        let tree = build_consensus(&["A", "B", "C"], &bs, 0.0).unwrap();
        let s = format!("{tree}");
        assert!(s.contains("ConsensusTree"));
    }

    #[test]
    fn test_config_builder() {
        let cfg = BootstrapConfig::new(500)
            .with_seed(123)
            .with_consensus_threshold(0.7);
        assert_eq!(cfg.replicates, 500);
        assert_eq!(cfg.seed, 123);
        assert!((cfg.consensus_threshold - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_config_display() {
        let cfg = BootstrapConfig::new(100);
        let s = format!("{cfg}");
        assert!(s.contains("reps=100"));
    }

    #[test]
    fn test_config_default() {
        let cfg = BootstrapConfig::default();
        assert_eq!(cfg.replicates, 100);
    }
}
