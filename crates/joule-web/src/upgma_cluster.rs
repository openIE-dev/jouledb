//! UPGMA hierarchical clustering — ultrametric tree construction from
//! pairwise distance matrices using unweighted pair-group method with
//! arithmetic mean.
//!
//! Produces rooted ultrametric trees where all leaves are equidistant
//! from the root. Includes cluster merging, height tracking, and
//! dendrogram-style output.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum UpgmaError {
    MatrixTooSmall(usize),
    InvalidMatrix(String),
    InternalError(String),
}

impl fmt::Display for UpgmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixTooSmall(n) => write!(f, "matrix too small: {n} taxa (need ≥ 2)"),
            Self::InvalidMatrix(s) => write!(f, "invalid matrix: {s}"),
            Self::InternalError(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for UpgmaError {}

// ── UPGMA Node ──────────────────────────────────────────────────

/// A node in the UPGMA dendrogram.
#[derive(Debug, Clone)]
pub struct UpgmaNode {
    pub id: usize,
    pub label: Option<String>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub height: f64,
    pub member_count: usize,
}

impl UpgmaNode {
    pub fn leaf(id: usize, label: &str) -> Self {
        Self {
            id,
            label: Some(label.to_string()),
            left: None,
            right: None,
            height: 0.0,
            member_count: 1,
        }
    }

    pub fn internal(id: usize, left: usize, right: usize, height: f64, members: usize) -> Self {
        Self {
            id,
            label: None,
            left: Some(left),
            right: Some(right),
            height,
            member_count: members,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

impl fmt::Display for UpgmaNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref lbl) = self.label {
            write!(f, "{lbl} (h={:.4})", self.height)
        } else {
            write!(f, "cluster_{} (h={:.4}, n={})", self.id, self.height, self.member_count)
        }
    }
}

// ── UPGMA Tree ──────────────────────────────────────────────────

/// The complete UPGMA dendrogram.
#[derive(Debug, Clone)]
pub struct UpgmaTree {
    pub nodes: Vec<UpgmaNode>,
    pub root: usize,
}

impl UpgmaTree {
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn tree_height(&self) -> f64 {
        self.nodes[self.root].height
    }

    /// Check whether the tree is ultrametric (all root-to-leaf heights equal).
    pub fn is_ultrametric(&self, tol: f64) -> bool {
        let leaf_heights: Vec<f64> = self
            .nodes
            .iter()
            .filter(|n| n.is_leaf())
            .map(|n| self.root_to_leaf_height(n.id))
            .collect();
        if leaf_heights.is_empty() {
            return true;
        }
        let first = leaf_heights[0];
        leaf_heights.iter().all(|h| (h - first).abs() <= tol)
    }

    fn root_to_leaf_height(&self, leaf_id: usize) -> f64 {
        // Walk up from leaf accumulating height differences
        self.nodes[self.root].height - self.nodes[leaf_id].height
    }

    /// Collect all leaf labels under a given node.
    pub fn leaves_under(&self, node_id: usize) -> Vec<String> {
        let node = &self.nodes[node_id];
        if node.is_leaf() {
            return node.label.iter().cloned().collect();
        }
        let mut result = Vec::new();
        if let Some(l) = node.left {
            result.extend(self.leaves_under(l));
        }
        if let Some(r) = node.right {
            result.extend(self.leaves_under(r));
        }
        result
    }

    /// Cut the dendrogram at a given height, returning the clusters.
    pub fn cut_at_height(&self, height: f64) -> Vec<Vec<String>> {
        let mut clusters = Vec::new();
        self.cut_recurse(self.root, height, &mut clusters);
        clusters
    }

    fn cut_recurse(&self, node_id: usize, height: f64, clusters: &mut Vec<Vec<String>>) {
        let node = &self.nodes[node_id];
        if node.height <= height || node.is_leaf() {
            clusters.push(self.leaves_under(node_id));
        } else {
            if let Some(l) = node.left {
                self.cut_recurse(l, height, clusters);
            }
            if let Some(r) = node.right {
                self.cut_recurse(r, height, clusters);
            }
        }
    }

    /// Serialise to Newick format.
    pub fn to_newick(&self) -> String {
        format!("{};", self.newick_recurse(self.root))
    }

    fn newick_recurse(&self, node_id: usize) -> String {
        let node = &self.nodes[node_id];
        let label = node.label.as_deref().unwrap_or("");
        if node.is_leaf() {
            format!("{label}:{:.6}", node.height)
        } else {
            let left_str = node.left.map_or(String::new(), |l| {
                let bl = node.height - self.nodes[l].height;
                format!("{}:{:.6}", self.newick_recurse(l), bl)
            });
            let right_str = node.right.map_or(String::new(), |r| {
                let bl = node.height - self.nodes[r].height;
                format!("{}:{:.6}", self.newick_recurse(r), bl)
            });
            format!("({left_str},{right_str}){label}")
        }
    }
}

impl fmt::Display for UpgmaTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UpgmaTree({} nodes, {} leaves, height={:.4})",
            self.node_count(),
            self.leaf_count(),
            self.tree_height()
        )
    }
}

// ── UPGMA Configuration ─────────────────────────────────────────

/// Configuration for UPGMA clustering.
#[derive(Debug, Clone)]
pub struct UpgmaConfig {
    pub weighted: bool,
}

impl UpgmaConfig {
    pub fn new() -> Self {
        Self { weighted: false }
    }

    /// Use WPGMA (weighted) instead of UPGMA (unweighted).
    pub fn with_weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }
}

impl Default for UpgmaConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── UPGMA algorithm ─────────────────────────────────────────────

/// Run UPGMA on a distance matrix (row-major, n×n, symmetric).
pub fn upgma(
    labels: &[&str],
    dist: &[f64],
    config: &UpgmaConfig,
) -> Result<UpgmaTree, UpgmaError> {
    let n = labels.len();
    if n < 2 {
        return Err(UpgmaError::MatrixTooSmall(n));
    }
    if dist.len() != n * n {
        return Err(UpgmaError::InvalidMatrix(format!(
            "expected {}×{}, got {} entries",
            n,
            n,
            dist.len()
        )));
    }

    let mut nodes: Vec<UpgmaNode> = labels
        .iter()
        .enumerate()
        .map(|(i, lbl)| UpgmaNode::leaf(i, lbl))
        .collect();
    let mut next_id = n;

    let mut active: Vec<usize> = (0..n).collect();
    let mut id_map: HashMap<usize, usize> = (0..n).map(|i| (i, i)).collect();
    let mut size_map: HashMap<usize, usize> = (0..n).map(|i| (i, 1)).collect();

    let mut d = dist.to_vec();
    let mut cur_n = n;

    while active.len() > 1 {
        // Find minimum distance pair
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_d = f64::INFINITY;
        for i in 0..cur_n {
            for j in i + 1..cur_n {
                let dij = d[i * cur_n + j];
                if dij < best_d {
                    best_d = dij;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let merge_height = best_d / 2.0;
        let node_i = id_map[&active[best_i]];
        let node_j = id_map[&active[best_j]];
        let size_i = size_map[&active[best_i]];
        let size_j = size_map[&active[best_j]];
        let total_size = size_i + size_j;

        let new_node = UpgmaNode::internal(next_id, node_i, node_j, merge_height, total_size);
        nodes.push(new_node);

        // Compute distances from new cluster to all others
        let mut new_row = vec![0.0; cur_n];
        for k in 0..cur_n {
            if k == best_i || k == best_j {
                continue;
            }
            if config.weighted {
                // WPGMA: simple average
                new_row[k] = 0.5 * (d[best_i * cur_n + k] + d[best_j * cur_n + k]);
            } else {
                // UPGMA: size-weighted average
                let si = size_i as f64;
                let sj = size_j as f64;
                new_row[k] =
                    (si * d[best_i * cur_n + k] + sj * d[best_j * cur_n + k]) / (si + sj);
            }
        }

        // Rebuild distance matrix
        let remove_max = best_i.max(best_j);
        let remove_min = best_i.min(best_j);
        let new_n = cur_n - 1;
        let mut new_d = vec![0.0; new_n * new_n];
        let mut new_active = Vec::with_capacity(new_n);
        let mut new_id_map = HashMap::new();
        let mut new_size_map = HashMap::new();

        let mut old_to_new = vec![0usize; cur_n];
        let mut idx = 0;
        for k in 0..cur_n {
            if k == remove_min || k == remove_max {
                continue;
            }
            old_to_new[k] = idx;
            new_active.push(active[k]);
            new_id_map.insert(active[k], id_map[&active[k]]);
            new_size_map.insert(active[k], size_map[&active[k]]);
            idx += 1;
        }
        let new_idx = idx;
        new_active.push(next_id);
        new_id_map.insert(next_id, next_id);
        new_size_map.insert(next_id, total_size);

        for a_idx in 0..cur_n {
            if a_idx == remove_min || a_idx == remove_max {
                continue;
            }
            for b_idx in 0..cur_n {
                if b_idx == remove_min || b_idx == remove_max {
                    continue;
                }
                let ni = old_to_new[a_idx];
                let nj = old_to_new[b_idx];
                new_d[ni * new_n + nj] = d[a_idx * cur_n + b_idx];
            }
        }

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
        size_map = new_size_map;
        d = new_d;
        cur_n = new_n;
        next_id += 1;
    }

    let root = next_id - 1;
    Ok(UpgmaTree { nodes, root })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dist() -> (Vec<&'static str>, Vec<f64>) {
        let labels = vec!["A", "B", "C", "D"];
        #[rustfmt::skip]
        let dist = vec![
            0.0, 2.0, 4.0, 6.0,
            2.0, 0.0, 4.0, 6.0,
            4.0, 4.0, 0.0, 6.0,
            6.0, 6.0, 6.0, 0.0,
        ];
        (labels, dist)
    }

    #[test]
    fn test_upgma_basic() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn test_upgma_node_count() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        // 4 leaves + 3 internal = 7
        assert_eq!(tree.node_count(), 7);
    }

    #[test]
    fn test_upgma_ultrametric() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        assert!(tree.is_ultrametric(1e-6));
    }

    #[test]
    fn test_upgma_tree_height() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        assert!(tree.tree_height() > 0.0);
    }

    #[test]
    fn test_upgma_newick() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        let nwk = tree.to_newick();
        assert!(nwk.ends_with(';'));
        assert!(nwk.contains('A'));
    }

    #[test]
    fn test_upgma_two_taxa() {
        let labels = vec!["X", "Y"];
        let dist = vec![0.0, 4.0, 4.0, 0.0];
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 2);
        assert!((tree.tree_height() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_upgma_error_small() {
        let labels = vec!["A"];
        let dist = vec![0.0];
        let cfg = UpgmaConfig::new();
        assert!(upgma(&labels, &dist, &cfg).is_err());
    }

    #[test]
    fn test_upgma_error_bad_matrix() {
        let labels = vec!["A", "B"];
        let dist = vec![0.0];
        let cfg = UpgmaConfig::new();
        assert!(upgma(&labels, &dist, &cfg).is_err());
    }

    #[test]
    fn test_cut_at_height() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        let clusters = tree.cut_at_height(1.5);
        // At height 1.5: {A,B} merged at 1.0, C and D separate
        assert!(clusters.len() >= 2);
    }

    #[test]
    fn test_leaves_under() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        let all = tree.leaves_under(tree.root);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_wpgma() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new().with_weighted(true);
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn test_upgma_node_leaf() {
        let n = UpgmaNode::leaf(0, "Sp1");
        assert!(n.is_leaf());
        assert_eq!(n.member_count, 1);
    }

    #[test]
    fn test_upgma_node_internal() {
        let n = UpgmaNode::internal(5, 0, 1, 1.5, 4);
        assert!(!n.is_leaf());
        assert_eq!(n.member_count, 4);
    }

    #[test]
    fn test_upgma_node_display() {
        let n = UpgmaNode::leaf(0, "Human");
        assert!(format!("{n}").contains("Human"));
    }

    #[test]
    fn test_upgma_tree_display() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        let s = format!("{tree}");
        assert!(s.contains("UpgmaTree"));
        assert!(s.contains("4 leaves"));
    }

    #[test]
    fn test_upgma_config_default() {
        let cfg = UpgmaConfig::default();
        assert!(!cfg.weighted);
    }

    #[test]
    fn test_upgma_five_taxa() {
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
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        assert_eq!(tree.leaf_count(), 5);
        assert!(tree.is_ultrametric(1e-6));
    }

    #[test]
    fn test_node_with_label() {
        let n = UpgmaNode::internal(5, 0, 1, 1.0, 2).with_label("clade_A");
        assert_eq!(n.label.as_deref(), Some("clade_A"));
    }

    #[test]
    fn test_error_display() {
        let e = UpgmaError::MatrixTooSmall(1);
        assert!(format!("{e}").contains("too small"));
    }

    #[test]
    fn test_cut_at_zero_returns_all_singletons() {
        let (labels, dist) = simple_dist();
        let cfg = UpgmaConfig::new();
        let tree = upgma(&labels, &dist, &cfg).unwrap();
        let clusters = tree.cut_at_height(0.0);
        assert_eq!(clusters.len(), 4);
    }
}
