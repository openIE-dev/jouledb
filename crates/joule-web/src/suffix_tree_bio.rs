//! Generalized Suffix Tree — Ukkonen's online construction, multi-sequence
//! indexing, longest common substring, pattern matching across biological
//! sequences.
//!
//! Pure-Rust suffix tree using Ukkonen's algorithm with suffix links for
//! O(n) construction. Supports generalized trees over multiple sequences
//! with unique sentinels, exact pattern search, and LCS extraction.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SuffixTreeError {
    EmptySequence,
    InvalidPosition(String),
    TooManySequences(usize),
}

impl fmt::Display for SuffixTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "sequence must not be empty"),
            Self::InvalidPosition(s) => write!(f, "invalid position: {s}"),
            Self::TooManySequences(n) => {
                write!(f, "too many sequences: {n} (max 254)")
            }
        }
    }
}

impl std::error::Error for SuffixTreeError {}

// ── Node ────────────────────────────────────────────────────────

/// Internal / leaf node in the suffix tree.
#[derive(Debug, Clone)]
struct Node {
    children: HashMap<u8, usize>,
    suffix_link: usize,
    start: usize,
    end: Option<usize>,
    /// Which sequences pass through this node (bitset for up to 64 seqs).
    seq_mask: u64,
}

impl Node {
    fn new(start: usize, end: Option<usize>) -> Self {
        Self {
            children: HashMap::new(),
            suffix_link: 0,
            start,
            end,
            seq_mask: 0,
        }
    }

    fn edge_length(&self, global_end: usize) -> usize {
        self.end.unwrap_or(global_end) - self.start
    }
}

// ── Active point ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct ActivePoint {
    node: usize,
    edge: Option<u8>,
    length: usize,
}

// ── SuffixTree ──────────────────────────────────────────────────

/// Generalized suffix tree built via Ukkonen's algorithm.
#[derive(Debug, Clone)]
pub struct SuffixTree {
    nodes: Vec<Node>,
    text: Vec<u8>,
    seq_boundaries: Vec<usize>,
    global_end: usize,
    active: ActivePoint,
    remaining: usize,
    last_new_internal: usize,
}

impl SuffixTree {
    /// Create an empty suffix tree.
    pub fn new() -> Self {
        let root = Node::new(0, Some(0));
        Self {
            nodes: vec![root],
            text: Vec::new(),
            seq_boundaries: Vec::new(),
            global_end: 0,
            active: ActivePoint { node: 0, edge: None, length: 0 },
            remaining: 0,
            last_new_internal: 0,
        }
    }

    /// Add a sequence to the generalized suffix tree.
    pub fn add_sequence(&mut self, seq: &[u8]) -> Result<(), SuffixTreeError> {
        if seq.is_empty() {
            return Err(SuffixTreeError::EmptySequence);
        }
        let seq_idx = self.seq_boundaries.len();
        if seq_idx >= 254 {
            return Err(SuffixTreeError::TooManySequences(seq_idx + 1));
        }
        // Append sequence with unique sentinel.
        let sentinel = (seq_idx as u8).wrapping_add(1);
        for &b in seq {
            self.extend_char(b);
        }
        self.extend_char(sentinel);
        self.seq_boundaries.push(self.text.len());
        // Mark leaves with sequence membership.
        self.mark_leaves(seq_idx);
        Ok(())
    }

    fn extend_char(&mut self, c: u8) {
        self.text.push(c);
        self.global_end = self.text.len();
        self.remaining += 1;
        self.last_new_internal = 0;

        while self.remaining > 0 {
            if self.active.length == 0 {
                self.active.edge = Some(c);
            }

            let edge_char = self.active.edge.unwrap_or(c);
            if !self.nodes[self.active.node].children.contains_key(&edge_char) {
                // New leaf.
                let leaf = self.new_node(self.global_end - 1, None);
                self.nodes[self.active.node].children.insert(edge_char, leaf);
                self.add_suffix_link(self.active.node);
            } else {
                let next = self.nodes[self.active.node].children[&edge_char];
                let edge_len = self.nodes[next].edge_length(self.global_end);
                if self.active.length >= edge_len {
                    self.active.node = next;
                    self.active.length -= edge_len;
                    self.active.edge = if self.active.length > 0 {
                        Some(self.text[self.global_end - 1 - self.active.length
                            + edge_len])
                    } else {
                        Some(c)
                    };
                    continue;
                }
                let pos = self.nodes[next].start + self.active.length;
                if pos < self.text.len() && self.text[pos] == c {
                    self.active.length += 1;
                    self.add_suffix_link(self.active.node);
                    break;
                }
                // Split edge.
                let split = self.new_node(
                    self.nodes[next].start,
                    Some(self.nodes[next].start + self.active.length),
                );
                self.nodes[self.active.node].children.insert(edge_char, split);
                let leaf = self.new_node(self.global_end - 1, None);
                self.nodes[split].children.insert(c, leaf);
                self.nodes[next].start += self.active.length;
                let first_of_next = self.text[self.nodes[next].start];
                self.nodes[split].children.insert(first_of_next, next);
                self.add_suffix_link(split);
            }
            self.remaining -= 1;
            if self.active.node == 0 && self.active.length > 0 {
                self.active.length -= 1;
                self.active.edge = Some(
                    self.text[self.global_end - self.remaining],
                );
            } else {
                let sl = self.nodes[self.active.node].suffix_link;
                self.active.node = if sl != 0 { sl } else { 0 };
            }
        }
    }

    fn new_node(&mut self, start: usize, end: Option<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::new(start, end));
        id
    }

    fn add_suffix_link(&mut self, node: usize) {
        if self.last_new_internal != 0 {
            self.nodes[self.last_new_internal].suffix_link = node;
        }
        self.last_new_internal = node;
    }

    fn mark_leaves(&mut self, seq_idx: usize) {
        let mask = 1u64 << (seq_idx.min(63));
        for node in &mut self.nodes {
            if node.end.is_none() {
                node.seq_mask |= mask;
            }
        }
    }

    // ── Queries ─────────────────────────────────────────────────

    /// Check whether `pattern` occurs in any indexed sequence.
    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.find_node(pattern).is_some()
    }

    /// Count occurrences of `pattern` across all indexed sequences.
    pub fn count_occurrences(&self, pattern: &[u8]) -> usize {
        match self.find_node(pattern) {
            Some(node) => self.count_leaves(node),
            None => 0,
        }
    }

    /// Return all start positions (global) where `pattern` occurs.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        if let Some(node) = self.find_node(pattern) {
            self.collect_positions(node, &mut positions);
        }
        positions.sort();
        positions
    }

    /// Longest common substring between sequences at indices `a` and `b`.
    pub fn longest_common_substring(&mut self, a: usize, b: usize) -> Vec<u8> {
        let mask_a = 1u64 << a.min(63);
        let mask_b = 1u64 << b.min(63);
        let mut best_len = 0usize;
        let mut best_end = 0usize;
        self.lcs_walk(0, 0, mask_a, mask_b, &mut best_len, &mut best_end);
        if best_len == 0 {
            return Vec::new();
        }
        self.text[best_end - best_len..best_end].to_vec()
    }

    /// Number of sequences stored.
    pub fn sequence_count(&self) -> usize {
        self.seq_boundaries.len()
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ── Internal traversal ──────────────────────────────────────

    fn find_node(&self, pattern: &[u8]) -> Option<usize> {
        if pattern.is_empty() {
            return Some(0);
        }
        let mut cur = 0usize;
        let mut i = 0usize;
        while i < pattern.len() {
            let c = pattern[i];
            let child = *self.nodes[cur].children.get(&c)?;
            let edge_start = self.nodes[child].start;
            let edge_end = self.nodes[child].end.unwrap_or(self.global_end);
            let edge_len = edge_end - edge_start;
            let compare_len = edge_len.min(pattern.len() - i);
            for j in 0..compare_len {
                if self.text[edge_start + j] != pattern[i + j] {
                    return None;
                }
            }
            i += compare_len;
            cur = child;
        }
        Some(cur)
    }

    fn count_leaves(&self, node: usize) -> usize {
        if self.nodes[node].children.is_empty() {
            return 1;
        }
        let mut total = 0;
        for &child in self.nodes[node].children.values() {
            total += self.count_leaves(child);
        }
        total
    }

    fn collect_positions(&self, node: usize, out: &mut Vec<usize>) {
        if self.nodes[node].children.is_empty() {
            out.push(self.nodes[node].start);
            return;
        }
        for &child in self.nodes[node].children.values() {
            self.collect_positions(child, out);
        }
    }

    fn lcs_walk(
        &mut self,
        node: usize,
        depth: usize,
        mask_a: u64,
        mask_b: u64,
        best_len: &mut usize,
        best_end: &mut usize,
    ) -> u64 {
        let mut combined = self.nodes[node].seq_mask;
        let children: Vec<usize> = self.nodes[node].children.values().copied().collect();
        for child in children {
            let child_depth =
                depth + self.nodes[child].edge_length(self.global_end);
            let child_mask =
                self.lcs_walk(child, child_depth, mask_a, mask_b, best_len, best_end);
            combined |= child_mask;
        }
        if (combined & mask_a) != 0 && (combined & mask_b) != 0 && depth > *best_len {
            *best_len = depth;
            // Approximate edge end for extracting substring.
            let edge_end = self.nodes[node].end.unwrap_or(self.global_end);
            *best_end = edge_end;
        }
        self.nodes[node].seq_mask |= combined;
        combined
    }
}

impl Default for SuffixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SuffixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SuffixTree({} seqs, {} nodes, {} chars)",
            self.sequence_count(),
            self.node_count(),
            self.text.len()
        )
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Builder for constructing a generalized suffix tree.
#[derive(Debug, Clone)]
pub struct SuffixTreeBuilder {
    sequences: Vec<Vec<u8>>,
}

impl SuffixTreeBuilder {
    pub fn new() -> Self {
        Self { sequences: Vec::new() }
    }

    pub fn with_sequence(mut self, seq: &[u8]) -> Self {
        self.sequences.push(seq.to_vec());
        self
    }

    pub fn with_string(mut self, s: &str) -> Self {
        self.sequences.push(s.as_bytes().to_vec());
        self
    }

    pub fn build(self) -> Result<SuffixTree, SuffixTreeError> {
        let mut tree = SuffixTree::new();
        for seq in &self.sequences {
            tree.add_sequence(seq)?;
        }
        Ok(tree)
    }
}

impl Default for SuffixTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = SuffixTree::new();
        assert_eq!(tree.sequence_count(), 0);
        assert_eq!(tree.node_count(), 1); // root only
    }

    #[test]
    fn test_single_sequence() {
        let mut tree = SuffixTree::new();
        tree.add_sequence(b"BANANA").unwrap();
        assert!(tree.contains(b"BAN"));
        assert!(tree.contains(b"ANA"));
        assert!(tree.contains(b"NA"));
        assert!(!tree.contains(b"BANZ"));
    }

    #[test]
    fn test_empty_sequence_error() {
        let mut tree = SuffixTree::new();
        assert!(tree.add_sequence(b"").is_err());
    }

    #[test]
    fn test_count_occurrences() {
        let mut tree = SuffixTree::new();
        tree.add_sequence(b"ABCABC").unwrap();
        assert_eq!(tree.count_occurrences(b"ABC"), 2);
        assert_eq!(tree.count_occurrences(b"XYZ"), 0);
    }

    #[test]
    fn test_builder() {
        let tree = SuffixTreeBuilder::new()
            .with_string("GATTACA")
            .with_string("TACA")
            .build()
            .unwrap();
        assert_eq!(tree.sequence_count(), 2);
        assert!(tree.contains(b"TACA"));
    }

    #[test]
    fn test_display() {
        let tree = SuffixTreeBuilder::new()
            .with_string("ATCG")
            .build()
            .unwrap();
        let s = format!("{tree}");
        assert!(s.contains("1 seqs"));
    }

    #[test]
    fn test_locate_positions() {
        let mut tree = SuffixTree::new();
        tree.add_sequence(b"AAAA").unwrap();
        let pos = tree.locate(b"AA");
        assert!(pos.len() >= 2);
    }

    #[test]
    fn test_multiple_sequences() {
        let tree = SuffixTreeBuilder::new()
            .with_string("ACGT")
            .with_string("TGCA")
            .build()
            .unwrap();
        assert!(tree.contains(b"ACG"));
        assert!(tree.contains(b"GCA"));
    }

    #[test]
    fn test_lcs_basic() {
        let mut tree = SuffixTreeBuilder::new()
            .with_string("ABCXYZ")
            .with_string("XYZABC")
            .build()
            .unwrap();
        let lcs = tree.longest_common_substring(0, 1);
        assert!(lcs.len() >= 3); // "ABC" or "XYZ"
    }

    #[test]
    fn test_single_char_sequence() {
        let mut tree = SuffixTree::new();
        tree.add_sequence(b"A").unwrap();
        assert!(tree.contains(b"A"));
        assert!(!tree.contains(b"B"));
    }

    #[test]
    fn test_repeated_pattern() {
        let mut tree = SuffixTree::new();
        tree.add_sequence(b"ATATAT").unwrap();
        assert_eq!(tree.count_occurrences(b"AT"), 3);
        assert_eq!(tree.count_occurrences(b"TAT"), 2);
    }

    #[test]
    fn test_contains_empty_pattern() {
        let mut tree = SuffixTree::new();
        tree.add_sequence(b"XYZ").unwrap();
        assert!(tree.contains(b""));
    }

    #[test]
    fn test_node_count_grows() {
        let mut tree = SuffixTree::new();
        let n0 = tree.node_count();
        tree.add_sequence(b"ABCDEF").unwrap();
        assert!(tree.node_count() > n0);
    }

    #[test]
    fn test_long_sequence() {
        let seq: Vec<u8> = (0..200).map(|i| b"ACGT"[i % 4]).collect();
        let mut tree = SuffixTree::new();
        tree.add_sequence(&seq).unwrap();
        assert!(tree.contains(&seq[10..20]));
    }

    #[test]
    fn test_builder_default() {
        let builder = SuffixTreeBuilder::default();
        let tree = builder.build().unwrap();
        assert_eq!(tree.sequence_count(), 0);
    }

    #[test]
    fn test_error_display() {
        let e = SuffixTreeError::EmptySequence;
        assert_eq!(format!("{e}"), "sequence must not be empty");
        let e2 = SuffixTreeError::TooManySequences(300);
        assert!(format!("{e2}").contains("300"));
    }

    #[test]
    fn test_dna_motif_search() {
        let tree = SuffixTreeBuilder::new()
            .with_sequence(b"ATGCGATCGATCG")
            .build()
            .unwrap();
        assert!(tree.contains(b"GATCG"));
        assert!(!tree.contains(b"NNNN"));
    }

    #[test]
    fn test_suffix_tree_default() {
        let tree = SuffixTree::default();
        assert_eq!(tree.sequence_count(), 0);
    }

    #[test]
    fn test_with_sequence_builder() {
        let tree = SuffixTreeBuilder::new()
            .with_sequence(b"HELLO")
            .build()
            .unwrap();
        assert!(tree.contains(b"ELL"));
    }

    #[test]
    fn test_three_sequences() {
        let tree = SuffixTreeBuilder::new()
            .with_string("AAA")
            .with_string("BBB")
            .with_string("CCC")
            .build()
            .unwrap();
        assert_eq!(tree.sequence_count(), 3);
        assert!(tree.contains(b"AA"));
        assert!(tree.contains(b"BB"));
        assert!(tree.contains(b"CC"));
    }
}
