//! Neighbor-joining tree construction — Q-matrix computation, iterative
//! joining, branch-length estimation, and optional tie-breaking strategies.
//!
//! Implements the Saitou-Nei (1987) neighbor-joining algorithm for building
//! unrooted additive trees from pairwise distance matrices. Produces trees
//! compatible with the `phylo_tree` module.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NjError {
    MatrixTooSmall(usize),
    InvalidEntry(String),
    InternalError(String),
}

impl fmt::Display for NjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixTooSmall(n) => write!(f, "matrix too small: {n} taxa (need ≥ 2)"),
            Self::InvalidEntry(s) => write!(f, "invalid entry: {s}"),
            Self::InternalError(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for NjError {}

// ── Q-matrix ────────────────────────────────────────────────────

/// The Q-matrix used for neighbor selection in NJ.
#[derive(Debug, Clone)]
pub struct QMatrix {
    pub values: Vec<f64>,
    pub n: usize,
}

impl QMatrix {
    /// Compute Q from a distance matrix.
    pub fn from_distances(dist: &[f64], n: usize) -> Self {
        let row_sums: Vec<f64> =
            (0..n).map(|i| (0..n).map(|j| dist[i * n + j]).sum()).collect();
        let mut values = vec![0.0; n * n];
        let nf = n as f64;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    values[i * n + j] = 0.0;
                } else {
                    values[i * n + j] =
                        (nf - 2.0) * dist[i * n + j] - row_sums[i] - row_sums[j];
                }
            }
        }
        Self { values, n }
    }

    /// Find the pair (i, j) with the minimum Q value.
    pub fn min_pair(&self) -> (usize, usize) {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_q = f64::INFINITY;
        for i in 0..self.n {
            for j in i + 1..self.n {
                let q = self.values[i * self.n + j];
                if q < best_q {
                    best_q = q;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        (best_i, best_j)
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.n + j]
    }
}

impl fmt::Display for QMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QMatrix({}×{})", self.n, self.n)
    }
}

// ── NJ Node ─────────────────────────────────────────────────────

/// A node produced by neighbor-joining.
#[derive(Debug, Clone)]
pub struct NjNode {
    pub id: usize,
    pub label: Option<String>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub left_length: f64,
    pub right_length: f64,
}

impl NjNode {
    pub fn leaf(id: usize, label: &str) -> Self {
        Self {
            id,
            label: Some(label.to_string()),
            left: None,
            right: None,
            left_length: 0.0,
            right_length: 0.0,
        }
    }

    pub fn internal(id: usize, left: usize, right: usize, ll: f64, rl: f64) -> Self {
        Self {
            id,
            label: None,
            left: Some(left),
            right: Some(right),
            left_length: ll,
            right_length: rl,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

impl fmt::Display for NjNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref lbl) = self.label {
            write!(f, "{lbl}")
        } else {
            write!(f, "node_{}", self.id)
        }
    }
}

// ── NJ Result ───────────────────────────────────────────────────

/// The result of neighbor-joining: a set of nodes forming an unrooted tree.
#[derive(Debug, Clone)]
pub struct NjTree {
    pub nodes: Vec<NjNode>,
    pub root: usize,
}

impl NjTree {
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Serialise the NJ tree in Newick format.
    pub fn to_newick(&self) -> String {
        format!("{};", self.newick_recurse(self.root))
    }

    fn newick_recurse(&self, node_id: usize) -> String {
        let node = &self.nodes[node_id];
        if node.is_leaf() {
            node.label.as_deref().unwrap_or("?").to_string()
        } else {
            let left_str = match node.left {
                Some(l) => format!("{}:{:.6}", self.newick_recurse(l), node.left_length),
                None => String::new(),
            };
            let right_str = match node.right {
                Some(r) => format!("{}:{:.6}", self.newick_recurse(r), node.right_length),
                None => String::new(),
            };
            format!("({left_str},{right_str})")
        }
    }
}

impl fmt::Display for NjTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NjTree({} nodes, {} leaves)",
            self.node_count(),
            self.leaf_count()
        )
    }
}

// ── NJ Configuration ────────────────────────────────────────────

/// Configuration for the neighbor-joining algorithm.
#[derive(Debug, Clone)]
pub struct NjConfig {
    pub negative_branch_to_zero: bool,
}

impl NjConfig {
    pub fn new() -> Self {
        Self { negative_branch_to_zero: true }
    }

    pub fn with_negative_branch_to_zero(mut self, val: bool) -> Self {
        self.negative_branch_to_zero = val;
        self
    }
}

impl Default for NjConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Neighbor-joining algorithm ──────────────────────────────────

/// Run the neighbor-joining algorithm on a distance matrix.
///
/// `labels` — taxon names, `dist` — row-major n×n symmetric distance matrix.
pub fn neighbor_join(
    labels: &[&str],
    dist: &[f64],
    config: &NjConfig,
) -> Result<NjTree, NjError> {
    let n = labels.len();
    if n < 2 {
        return Err(NjError::MatrixTooSmall(n));
    }
    if dist.len() != n * n {
        return Err(NjError::InvalidEntry(format!(
            "expected {}×{} matrix, got {} entries",
            n,
            n,
            dist.len()
        )));
    }

    let mut nodes: Vec<NjNode> = labels
        .iter()
        .enumerate()
        .map(|(i, lbl)| NjNode::leaf(i, lbl))
        .collect();
    let mut next_id = n;

    // Active taxon indices (into the evolving distance matrix)
    let mut active: Vec<usize> = (0..n).collect();
    // Map from active index → node id
    let mut id_map: HashMap<usize, usize> = (0..n).map(|i| (i, i)).collect();

    // Working distance matrix
    let mut d = dist.to_vec();
    let mut cur_n = n;

    while active.len() > 2 {
        let q = QMatrix::from_distances(&d, cur_n);
        let (qi, qj) = q.min_pair();

        let row_sum_i: f64 = (0..cur_n).map(|k| d[qi * cur_n + k]).sum();
        let row_sum_j: f64 = (0..cur_n).map(|k| d[qj * cur_n + k]).sum();

        let nf = cur_n as f64;
        let mut li = 0.5 * d[qi * cur_n + qj]
            + (row_sum_i - row_sum_j) / (2.0 * (nf - 2.0));
        let mut lj = d[qi * cur_n + qj] - li;

        if config.negative_branch_to_zero {
            if li < 0.0 { li = 0.0; }
            if lj < 0.0 { lj = 0.0; }
        }

        let node_i = id_map[&active[qi]];
        let node_j = id_map[&active[qj]];
        let new_node = NjNode::internal(next_id, node_i, node_j, li, lj);
        nodes.push(new_node);

        // Compute distances from new node to all remaining taxa
        let mut new_row = Vec::with_capacity(cur_n);
        for k in 0..cur_n {
            if k == qi || k == qj {
                new_row.push(0.0);
            } else {
                let dk = 0.5
                    * (d[qi * cur_n + k] + d[qj * cur_n + k] - d[qi * cur_n + qj]);
                new_row.push(dk);
            }
        }

        // Rebuild distance matrix without qi, qj, adding new node
        let remove_max = qi.max(qj);
        let remove_min = qi.min(qj);

        let new_n = cur_n - 1; // remove 2, add 1
        let mut new_d = vec![0.0; new_n * new_n];
        let mut new_active = Vec::with_capacity(new_n);
        let mut new_id_map = HashMap::new();

        // Map old indices to new, skipping removed
        let mut old_to_new = vec![0usize; cur_n];
        let mut idx = 0;
        for k in 0..cur_n {
            if k == remove_min || k == remove_max {
                continue;
            }
            old_to_new[k] = idx;
            new_active.push(active[k]);
            new_id_map.insert(active[k], id_map[&active[k]]);
            idx += 1;
        }
        // New node gets the last index
        let new_idx = idx;
        new_active.push(next_id);
        new_id_map.insert(next_id, next_id);

        // Fill in distances among old survivors
        for a_idx in 0..cur_n {
            if a_idx == remove_min || a_idx == remove_max {
                continue;
            }
            for b_idx in 0..cur_n {
                if b_idx == remove_min || b_idx == remove_max {
                    continue;
                }
                let ni = old_to_new[a_idx];
                let nj_idx = old_to_new[b_idx];
                new_d[ni * new_n + nj_idx] = d[a_idx * cur_n + b_idx];
            }
        }

        // Fill distances from new node to survivors
        for k in 0..cur_n {
            if k == remove_min || k == remove_max {
                continue;
            }
            let ni = old_to_new[k];
            new_d[ni * new_n + new_idx] = new_row[k];
            new_d[new_idx * new_n + ni] = new_row[k];
        }

        active = new_active;
        id_map = new_id_map;
        d = new_d;
        cur_n = new_n;
        next_id += 1;
    }

    // Final two taxa
    if active.len() == 2 {
        let node_a = id_map[&active[0]];
        let node_b = id_map[&active[1]];
        let final_d = d[0 * cur_n + 1];
        let root_node = NjNode::internal(next_id, node_a, node_b, final_d / 2.0, final_d / 2.0);
        nodes.push(root_node);
        Ok(NjTree { nodes, root: next_id })
    } else {
        Ok(NjTree { nodes, root: id_map[&active[0]] })
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_4_taxa() -> (Vec<&'static str>, Vec<f64>) {
        let labels = vec!["A", "B", "C", "D"];
        // Symmetric 4×4 distance matrix
        #[rustfmt::skip]
        let dist = vec![
            0.0, 5.0, 9.0, 9.0,
            5.0, 0.0, 10.0, 10.0,
            9.0, 10.0, 0.0, 8.0,
            9.0, 10.0, 8.0, 0.0,
        ];
        (labels, dist)
    }

    #[test]
    fn test_q_matrix_computation() {
        let (_, dist) = simple_4_taxa();
        let q = QMatrix::from_distances(&dist, 4);
        assert_eq!(q.n, 4);
        // Diagonal should be zero
        assert!((q.get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn test_q_matrix_min_pair() {
        let (_, dist) = simple_4_taxa();
        let q = QMatrix::from_distances(&dist, 4);
        let (i, j) = q.min_pair();
        // A and B are closest in Q-space
        assert_eq!((i, j), (0, 1));
    }

    #[test]
    fn test_nj_basic() {
        let (labels, dist) = simple_4_taxa();
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn test_nj_node_count() {
        let (labels, dist) = simple_4_taxa();
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        // 4 leaves + internal nodes
        assert!(tree.node_count() >= 5);
    }

    #[test]
    fn test_nj_newick() {
        let (labels, dist) = simple_4_taxa();
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        let nwk = tree.to_newick();
        assert!(nwk.ends_with(';'));
        assert!(nwk.contains('A'));
        assert!(nwk.contains('D'));
    }

    #[test]
    fn test_nj_two_taxa() {
        let labels = vec!["X", "Y"];
        let dist = vec![0.0, 3.0, 3.0, 0.0];
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 2);
    }

    #[test]
    fn test_nj_three_taxa() {
        let labels = vec!["A", "B", "C"];
        #[rustfmt::skip]
        let dist = vec![
            0.0, 4.0, 6.0,
            4.0, 0.0, 6.0,
            6.0, 6.0, 0.0,
        ];
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn test_nj_error_too_small() {
        let labels = vec!["A"];
        let dist = vec![0.0];
        let cfg = NjConfig::new();
        assert!(neighbor_join(&labels, &dist, &cfg).is_err());
    }

    #[test]
    fn test_nj_error_bad_matrix() {
        let labels = vec!["A", "B"];
        let dist = vec![0.0]; // wrong size
        let cfg = NjConfig::new();
        assert!(neighbor_join(&labels, &dist, &cfg).is_err());
    }

    #[test]
    fn test_nj_config_builder() {
        let cfg = NjConfig::new().with_negative_branch_to_zero(false);
        assert!(!cfg.negative_branch_to_zero);
    }

    #[test]
    fn test_nj_node_leaf() {
        let n = NjNode::leaf(0, "Taxon");
        assert!(n.is_leaf());
        assert_eq!(format!("{n}"), "Taxon");
    }

    #[test]
    fn test_nj_node_internal() {
        let n = NjNode::internal(5, 0, 1, 0.1, 0.2);
        assert!(!n.is_leaf());
        assert_eq!(format!("{n}"), "node_5");
    }

    #[test]
    fn test_nj_tree_display() {
        let (labels, dist) = simple_4_taxa();
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        let s = format!("{tree}");
        assert!(s.contains("NjTree"));
        assert!(s.contains("4 leaves"));
    }

    #[test]
    fn test_q_matrix_display() {
        let q = QMatrix { values: vec![0.0; 4], n: 2 };
        assert_eq!(format!("{q}"), "QMatrix(2×2)");
    }

    #[test]
    fn test_nj_symmetric_distances() {
        let labels = vec!["A", "B", "C", "D", "E"];
        let n = 5;
        let mut dist = vec![0.0; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let d = ((i + j + 1) * 2) as f64;
                dist[i * n + j] = d;
                dist[j * n + i] = d;
            }
        }
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 5);
    }

    #[test]
    fn test_nj_equal_distances() {
        let labels = vec!["A", "B", "C"];
        #[rustfmt::skip]
        let dist = vec![
            0.0, 4.0, 4.0,
            4.0, 0.0, 4.0,
            4.0, 4.0, 0.0,
        ];
        let cfg = NjConfig::new();
        let tree = neighbor_join(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn test_nj_config_default() {
        let cfg = NjConfig::default();
        assert!(cfg.negative_branch_to_zero);
    }

    #[test]
    fn test_nj_error_display() {
        let e = NjError::MatrixTooSmall(1);
        assert!(format!("{e}").contains("too small"));
    }
}
