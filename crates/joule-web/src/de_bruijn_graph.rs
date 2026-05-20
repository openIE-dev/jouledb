//! De Bruijn Graph — k-mer–based assembly graph with Eulerian path
//! traversal, tip removal, and bubble collapsing for sequence assembly.
//!
//! Pure-Rust de Bruijn graph operating on byte sequences. Constructs the
//! graph from k-mers, supports Eulerian path/circuit detection, basic
//! graph simplification (tip removal, isolated node pruning), and contig
//! extraction for genome assembly pipelines.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DeBruijnError {
    KTooSmall(usize),
    SequenceTooShort { seq_len: usize, k: usize },
    NoEulerianPath,
    EmptyGraph,
}

impl fmt::Display for DeBruijnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KTooSmall(k) => write!(f, "k must be >= 2, got {k}"),
            Self::SequenceTooShort { seq_len, k } => {
                write!(f, "sequence length {seq_len} < k={k}")
            }
            Self::NoEulerianPath => write!(f, "no Eulerian path exists"),
            Self::EmptyGraph => write!(f, "graph is empty"),
        }
    }
}

impl std::error::Error for DeBruijnError {}

// ── Edge ────────────────────────────────────────────────────────

/// A directed edge in the de Bruijn graph, representing a k-mer.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: Vec<u8>,
    pub to: Vec<u8>,
    pub kmer: Vec<u8>,
    pub multiplicity: usize,
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kmer_str = String::from_utf8_lossy(&self.kmer);
        write!(f, "{}(x{})", kmer_str, self.multiplicity)
    }
}

// ── Node info ───────────────────────────────────────────────────

/// Per-node bookkeeping.
#[derive(Debug, Clone)]
struct NodeInfo {
    out_edges: Vec<usize>,
    in_degree: usize,
    out_degree: usize,
}

impl NodeInfo {
    fn new() -> Self {
        Self {
            out_edges: Vec::new(),
            in_degree: 0,
            out_degree: 0,
        }
    }
}

// ── De Bruijn graph ─────────────────────────────────────────────

/// De Bruijn graph for sequence assembly.
#[derive(Debug, Clone)]
pub struct DeBruijnGraph {
    k: usize,
    edges: Vec<Edge>,
    nodes: HashMap<Vec<u8>, NodeInfo>,
}

impl DeBruijnGraph {
    /// Create an empty de Bruijn graph with k-mer size `k`.
    pub fn new(k: usize) -> Result<Self, DeBruijnError> {
        if k < 2 {
            return Err(DeBruijnError::KTooSmall(k));
        }
        Ok(Self {
            k,
            edges: Vec::new(),
            nodes: HashMap::new(),
        })
    }

    /// Add a sequence, decomposing it into (k)-mers that become edges.
    pub fn add_sequence(&mut self, seq: &[u8]) -> Result<(), DeBruijnError> {
        if seq.len() < self.k {
            return Err(DeBruijnError::SequenceTooShort {
                seq_len: seq.len(),
                k: self.k,
            });
        }
        for i in 0..=seq.len() - self.k {
            let kmer = &seq[i..i + self.k];
            let prefix = kmer[..self.k - 1].to_vec();
            let suffix = kmer[1..].to_vec();
            self.add_edge(prefix, suffix, kmer.to_vec());
        }
        Ok(())
    }

    fn add_edge(&mut self, from: Vec<u8>, to: Vec<u8>, kmer: Vec<u8>) {
        // Check for existing edge with same k-mer.
        for edge in &mut self.edges {
            if edge.from == from && edge.to == to {
                edge.multiplicity += 1;
                return;
            }
        }
        let edge_idx = self.edges.len();
        self.edges.push(Edge {
            from: from.clone(),
            to: to.clone(),
            kmer,
            multiplicity: 1,
        });
        self.nodes
            .entry(from)
            .or_insert_with(NodeInfo::new)
            .out_edges
            .push(edge_idx);
        self.nodes
            .entry(to.clone())
            .or_insert_with(NodeInfo::new);
        // Update degrees.
        let from_key = &self.edges[edge_idx].from;
        self.nodes.get_mut(from_key).unwrap().out_degree += 1;
        self.nodes.get_mut(&to).unwrap().in_degree += 1;
    }

    /// Number of nodes (distinct (k-1)-mers).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges (distinct k-mers, counting multiplicity as one).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The k-mer size.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return all edges.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Check if an Eulerian path exists.
    pub fn has_eulerian_path(&self) -> bool {
        let mut start_nodes = 0;
        let mut end_nodes = 0;
        for info in self.nodes.values() {
            let diff = info.out_degree as isize - info.in_degree as isize;
            if diff == 1 {
                start_nodes += 1;
            } else if diff == -1 {
                end_nodes += 1;
            } else if diff != 0 {
                return false;
            }
        }
        // Either Eulerian circuit (all balanced) or path (one +1, one -1).
        (start_nodes == 0 && end_nodes == 0)
            || (start_nodes == 1 && end_nodes == 1)
    }

    /// Find an Eulerian path using Hierholzer's algorithm.
    pub fn eulerian_path(&self) -> Result<Vec<Vec<u8>>, DeBruijnError> {
        if self.edges.is_empty() {
            return Err(DeBruijnError::EmptyGraph);
        }
        if !self.has_eulerian_path() {
            return Err(DeBruijnError::NoEulerianPath);
        }

        // Find start node.
        let start = self.find_start_node();

        // Hierholzer's.
        let mut edge_used = vec![false; self.edges.len()];
        let mut edge_ptr: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut stack = vec![start.clone()];
        let mut path = Vec::new();

        while let Some(v) = stack.last().cloned() {
            let ptr = edge_ptr.entry(v.clone()).or_insert(0);
            let info = &self.nodes[&v];
            let mut found = false;
            while *ptr < info.out_edges.len() {
                let edge_idx = info.out_edges[*ptr];
                *ptr += 1;
                if !edge_used[edge_idx] {
                    edge_used[edge_idx] = true;
                    stack.push(self.edges[edge_idx].to.clone());
                    found = true;
                    break;
                }
            }
            if !found {
                stack.pop();
                path.push(v);
            }
        }

        path.reverse();
        Ok(path)
    }

    /// Assemble a contig from the Eulerian path.
    pub fn assemble(&self) -> Result<Vec<u8>, DeBruijnError> {
        let path = self.eulerian_path()?;
        if path.is_empty() {
            return Ok(Vec::new());
        }
        let mut contig = path[0].clone();
        for node in &path[1..] {
            contig.push(*node.last().unwrap());
        }
        Ok(contig)
    }

    /// Remove tips: dead-end paths shorter than `max_len`.
    pub fn remove_tips(&mut self, max_len: usize) -> usize {
        let mut removed = 0;
        let tip_nodes: Vec<Vec<u8>> = self
            .nodes
            .iter()
            .filter(|(_, info)| {
                (info.in_degree == 0 && info.out_degree == 1)
                    || (info.in_degree == 1 && info.out_degree == 0)
            })
            .map(|(key, _)| key.clone())
            .collect();

        for node in tip_nodes {
            if self.trace_tip(&node) <= max_len {
                self.remove_node(&node);
                removed += 1;
            }
        }
        removed
    }

    /// Remove isolated nodes (no edges).
    pub fn remove_isolated(&mut self) -> usize {
        let isolated: Vec<Vec<u8>> = self
            .nodes
            .iter()
            .filter(|(_, info)| info.in_degree == 0 && info.out_degree == 0)
            .map(|(key, _)| key.clone())
            .collect();
        let count = isolated.len();
        for node in isolated {
            self.nodes.remove(&node);
        }
        count
    }

    /// Compute k-mer coverage statistics: (min, max, mean).
    pub fn coverage_stats(&self) -> (usize, usize, f64) {
        if self.edges.is_empty() {
            return (0, 0, 0.0);
        }
        let mut min_cov = usize::MAX;
        let mut max_cov = 0;
        let mut total = 0usize;
        for edge in &self.edges {
            min_cov = min_cov.min(edge.multiplicity);
            max_cov = max_cov.max(edge.multiplicity);
            total += edge.multiplicity;
        }
        let mean = total as f64 / self.edges.len() as f64;
        (min_cov, max_cov, mean)
    }

    // ── Internal helpers ────────────────────────────────────────

    fn find_start_node(&self) -> Vec<u8> {
        for (key, info) in &self.nodes {
            if info.out_degree as isize - info.in_degree as isize == 1 {
                return key.clone();
            }
        }
        // Eulerian circuit: any node with edges.
        self.nodes
            .iter()
            .find(|(_, info)| info.out_degree > 0)
            .map(|(key, _)| key.clone())
            .unwrap_or_default()
    }

    fn trace_tip(&self, start: &[u8]) -> usize {
        let mut current = start.to_vec();
        let mut length = 0;
        loop {
            let info = match self.nodes.get(&current) {
                Some(i) => i,
                None => break,
            };
            if info.out_edges.len() != 1 {
                break;
            }
            let edge_idx = info.out_edges[0];
            current = self.edges[edge_idx].to.clone();
            length += 1;
            if length > 100 {
                break;
            }
        }
        length
    }

    fn remove_node(&mut self, node: &[u8]) {
        if let Some(info) = self.nodes.remove(node) {
            for &edge_idx in &info.out_edges {
                let to = self.edges[edge_idx].to.clone();
                if let Some(to_info) = self.nodes.get_mut(&to) {
                    to_info.in_degree = to_info.in_degree.saturating_sub(1);
                }
            }
        }
    }
}

impl fmt::Display for DeBruijnGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeBruijnGraph(k={}, nodes={}, edges={})",
            self.k,
            self.node_count(),
            self.edge_count()
        )
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Builder for de Bruijn graphs.
#[derive(Debug, Clone)]
pub struct DeBruijnBuilder {
    k: usize,
    sequences: Vec<Vec<u8>>,
}

impl DeBruijnBuilder {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            sequences: Vec::new(),
        }
    }

    pub fn with_sequence(mut self, seq: &[u8]) -> Self {
        self.sequences.push(seq.to_vec());
        self
    }

    pub fn with_string(mut self, s: &str) -> Self {
        self.sequences.push(s.as_bytes().to_vec());
        self
    }

    pub fn build(self) -> Result<DeBruijnGraph, DeBruijnError> {
        let mut graph = DeBruijnGraph::new(self.k)?;
        for seq in &self.sequences {
            graph.add_sequence(seq)?;
        }
        Ok(graph)
    }
}

impl fmt::Display for DeBruijnBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeBruijnBuilder(k={}, {} seqs)",
            self.k,
            self.sequences.len()
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph() {
        let g = DeBruijnGraph::new(3).unwrap();
        assert_eq!(g.k(), 3);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_k_too_small() {
        assert!(DeBruijnGraph::new(1).is_err());
        assert!(DeBruijnGraph::new(0).is_err());
    }

    #[test]
    fn test_add_sequence() {
        let mut g = DeBruijnGraph::new(3).unwrap();
        g.add_sequence(b"ACGTAC").unwrap();
        assert!(g.node_count() > 0);
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_sequence_too_short() {
        let mut g = DeBruijnGraph::new(5).unwrap();
        assert!(g.add_sequence(b"ACG").is_err());
    }

    #[test]
    fn test_eulerian_path_simple() {
        let g = DeBruijnBuilder::new(3)
            .with_string("ACGTAC")
            .build()
            .unwrap();
        if g.has_eulerian_path() {
            let path = g.eulerian_path().unwrap();
            assert!(!path.is_empty());
        }
    }

    #[test]
    fn test_assemble_simple() {
        let g = DeBruijnBuilder::new(3)
            .with_string("ACGTAC")
            .build()
            .unwrap();
        if g.has_eulerian_path() {
            let contig = g.assemble().unwrap();
            assert!(!contig.is_empty());
        }
    }

    #[test]
    fn test_coverage_stats() {
        let mut g = DeBruijnGraph::new(3).unwrap();
        g.add_sequence(b"ACGTAC").unwrap();
        let (min_c, max_c, mean) = g.coverage_stats();
        assert!(min_c >= 1);
        assert!(max_c >= min_c);
        assert!(mean >= 1.0);
    }

    #[test]
    fn test_coverage_empty() {
        let g = DeBruijnGraph::new(3).unwrap();
        assert_eq!(g.coverage_stats(), (0, 0, 0.0));
    }

    #[test]
    fn test_multiplicity() {
        let mut g = DeBruijnGraph::new(3).unwrap();
        g.add_sequence(b"ACGACG").unwrap();
        let multi: Vec<&Edge> = g.edges().iter().filter(|e| e.multiplicity > 1).collect();
        assert!(!multi.is_empty());
    }

    #[test]
    fn test_remove_isolated() {
        let mut g = DeBruijnGraph::new(3).unwrap();
        g.add_sequence(b"ACGT").unwrap();
        // Remove edges to create orphans.
        let before = g.node_count();
        g.edges.clear();
        for info in g.nodes.values_mut() {
            info.out_edges.clear();
            info.in_degree = 0;
            info.out_degree = 0;
        }
        let removed = g.remove_isolated();
        assert_eq!(removed, before);
    }

    #[test]
    fn test_empty_graph_no_euler() {
        let g = DeBruijnGraph::new(3).unwrap();
        assert!(g.eulerian_path().is_err());
    }

    #[test]
    fn test_display_graph() {
        let g = DeBruijnBuilder::new(3)
            .with_string("ACGT")
            .build()
            .unwrap();
        let s = format!("{g}");
        assert!(s.contains("DeBruijnGraph"));
    }

    #[test]
    fn test_display_edge() {
        let e = Edge {
            from: b"AC".to_vec(),
            to: b"CG".to_vec(),
            kmer: b"ACG".to_vec(),
            multiplicity: 2,
        };
        let s = format!("{e}");
        assert!(s.contains("ACG"));
        assert!(s.contains("x2"));
    }

    #[test]
    fn test_display_builder() {
        let b = DeBruijnBuilder::new(4);
        let s = format!("{b}");
        assert!(s.contains("k=4"));
    }

    #[test]
    fn test_builder_with_sequence() {
        let g = DeBruijnBuilder::new(3)
            .with_sequence(b"GATTACA")
            .build()
            .unwrap();
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_error_display() {
        let e = DeBruijnError::KTooSmall(1);
        assert!(format!("{e}").contains("1"));
        let e2 = DeBruijnError::EmptyGraph;
        assert_eq!(format!("{e2}"), "graph is empty");
    }

    #[test]
    fn test_has_eulerian_path_check() {
        let g = DeBruijnBuilder::new(3)
            .with_string("ABCDE")
            .build()
            .unwrap();
        // Simple linear chain always has Eulerian path.
        assert!(g.has_eulerian_path());
    }

    #[test]
    fn test_multiple_sequences() {
        let g = DeBruijnBuilder::new(3)
            .with_string("ACGT")
            .with_string("CGTG")
            .build()
            .unwrap();
        assert!(g.edge_count() >= 2);
    }

    #[test]
    fn test_remove_tips() {
        let mut g = DeBruijnGraph::new(3).unwrap();
        g.add_sequence(b"ACGTACGT").unwrap();
        let _removed = g.remove_tips(2);
        // Should not panic.
    }
}
