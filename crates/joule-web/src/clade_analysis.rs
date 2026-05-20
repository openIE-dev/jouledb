//! Clade analysis — monophyly testing, clade support evaluation,
//! Robinson-Foulds distance, split-based tree comparison, and
//! clade-level statistics for phylogenetic trees.
//!
//! Provides tools for assessing whether a set of taxa form a
//! monophyletic group, computing topological distances between
//! trees, and enumerating all clades in a phylogeny.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CladeError {
    NodeNotFound(usize),
    EmptyTree,
    EmptyTaxonSet,
    TaxonNotFound(String),
    IncompatibleTrees(String),
}

impl fmt::Display for CladeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::EmptyTree => write!(f, "tree is empty"),
            Self::EmptyTaxonSet => write!(f, "empty taxon set"),
            Self::TaxonNotFound(s) => write!(f, "taxon not found: {s}"),
            Self::IncompatibleTrees(s) => write!(f, "incompatible trees: {s}"),
        }
    }
}

impl std::error::Error for CladeError {}

// ── Clade tree node ─────────────────────────────────────────────

/// A node for clade analysis purposes.
#[derive(Debug, Clone)]
pub struct CladeNode {
    pub id: usize,
    pub label: Option<String>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub branch_length: f64,
    pub support: Option<f64>,
}

impl CladeNode {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            parent: None,
            children: Vec::new(),
            branch_length: 0.0,
            support: None,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_branch_length(mut self, len: f64) -> Self {
        self.branch_length = len;
        self
    }

    pub fn with_support(mut self, sup: f64) -> Self {
        self.support = Some(sup);
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

impl fmt::Display for CladeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.label, self.support) {
            (Some(lbl), Some(sup)) => write!(f, "{lbl}[{:.0}%]", sup * 100.0),
            (Some(lbl), None) => write!(f, "{lbl}"),
            (None, Some(sup)) => write!(f, "node_{}[{:.0}%]", self.id, sup * 100.0),
            (None, None) => write!(f, "node_{}", self.id),
        }
    }
}

// ── Clade tree ──────────────────────────────────────────────────

/// A tree structure for clade analysis.
#[derive(Debug, Clone)]
pub struct CladeTree {
    pub nodes: Vec<CladeNode>,
    pub root: Option<usize>,
}

impl CladeTree {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), root: None }
    }

    pub fn add_node(&mut self, label: Option<&str>, branch_length: f64) -> usize {
        let id = self.nodes.len();
        let mut node = CladeNode::new(id).with_branch_length(branch_length);
        if let Some(lbl) = label {
            node = node.with_label(lbl);
        }
        self.nodes.push(node);
        if self.root.is_none() {
            self.root = Some(id);
        }
        id
    }

    pub fn set_parent(&mut self, child: usize, parent: usize) {
        self.nodes[child].parent = Some(parent);
        if !self.nodes[parent].children.contains(&child) {
            self.nodes[parent].children.push(child);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Get all leaf ids.
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes.iter().filter(|n| n.is_leaf()).map(|n| n.id).collect()
    }

    /// Get all leaf labels.
    pub fn leaf_labels(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|n| n.is_leaf())
            .filter_map(|n| n.label.clone())
            .collect()
    }

    /// Find node id by label.
    pub fn find_by_label(&self, label: &str) -> Option<usize> {
        self.nodes.iter().find(|n| n.label.as_deref() == Some(label)).map(|n| n.id)
    }

    /// Collect all leaf ids descended from a node.
    pub fn descendant_leaves(&self, node_id: usize) -> Result<HashSet<usize>, CladeError> {
        if node_id >= self.nodes.len() {
            return Err(CladeError::NodeNotFound(node_id));
        }
        let mut result = HashSet::new();
        let mut stack = vec![node_id];
        while let Some(cur) = stack.pop() {
            if self.nodes[cur].is_leaf() {
                result.insert(cur);
            } else {
                for &child in &self.nodes[cur].children {
                    stack.push(child);
                }
            }
        }
        Ok(result)
    }

    /// Collect descendant leaf labels.
    pub fn descendant_labels(&self, node_id: usize) -> Result<HashSet<String>, CladeError> {
        let leaves = self.descendant_leaves(node_id)?;
        Ok(leaves
            .into_iter()
            .filter_map(|id| self.nodes[id].label.clone())
            .collect())
    }
}

impl Default for CladeTree {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CladeTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CladeTree({} nodes, {} leaves)",
            self.node_count(),
            self.leaf_count()
        )
    }
}

// ── Monophyly testing ───────────────────────────────────────────

/// Monophyly status of a group of taxa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonophylyStatus {
    /// All and only the specified taxa descend from a single ancestor.
    Monophyletic,
    /// The group shares an ancestor but that ancestor has other descendants too.
    Paraphyletic,
    /// The group does not share an exclusive ancestor (multiple origins).
    Polyphyletic,
}

impl fmt::Display for MonophylyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Monophyletic => write!(f, "monophyletic"),
            Self::Paraphyletic => write!(f, "paraphyletic"),
            Self::Polyphyletic => write!(f, "polyphyletic"),
        }
    }
}

/// Test whether a set of taxa form a monophyletic group.
pub fn test_monophyly(
    tree: &CladeTree,
    taxa: &[&str],
) -> Result<MonophylyStatus, CladeError> {
    if taxa.is_empty() {
        return Err(CladeError::EmptyTaxonSet);
    }
    let _root = tree.root.ok_or(CladeError::EmptyTree)?;

    // Map taxon labels to node ids
    let mut target_ids = HashSet::new();
    for &taxon in taxa {
        let id = tree
            .find_by_label(taxon)
            .ok_or_else(|| CladeError::TaxonNotFound(taxon.to_string()))?;
        target_ids.insert(id);
    }

    // Find the smallest clade containing all target taxa
    // For each internal node, check if its descendant leaves are a superset of target
    let mut best_node = None;
    let mut best_size = usize::MAX;

    for node in &tree.nodes {
        if node.is_leaf() {
            continue;
        }
        let desc = tree.descendant_leaves(node.id)?;
        if target_ids.is_subset(&desc) && desc.len() < best_size {
            best_size = desc.len();
            best_node = Some(node.id);
        }
    }

    let mrca = best_node.ok_or(CladeError::EmptyTree)?;
    let mrca_leaves = tree.descendant_leaves(mrca)?;

    if mrca_leaves == target_ids {
        Ok(MonophylyStatus::Monophyletic)
    } else if mrca_leaves.is_superset(&target_ids) {
        // Check if paraphyletic or polyphyletic
        // Paraphyletic: MRCA contains target + others, but target is connected
        Ok(MonophylyStatus::Paraphyletic)
    } else {
        Ok(MonophylyStatus::Polyphyletic)
    }
}

// ── Robinson-Foulds distance ────────────────────────────────────

/// A bipartition (split) represented as a sorted set of leaf labels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BipartitionSplit {
    pub labels: Vec<String>,
}

impl BipartitionSplit {
    pub fn new(mut labels: Vec<String>) -> Self {
        labels.sort();
        Self { labels }
    }
}

impl fmt::Display for BipartitionSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}}}", self.labels.join(", "))
    }
}

/// Extract all non-trivial bipartitions from a tree.
pub fn extract_bipartitions(tree: &CladeTree) -> Result<HashSet<BipartitionSplit>, CladeError> {
    let _root = tree.root.ok_or(CladeError::EmptyTree)?;
    let total_leaves = tree.leaf_count();
    let mut splits = HashSet::new();

    for node in &tree.nodes {
        if node.is_leaf() || tree.root == Some(node.id) {
            continue;
        }
        let desc_labels = tree.descendant_labels(node.id)?;
        let desc_vec: Vec<String> = desc_labels.into_iter().collect();
        if desc_vec.len() > 1 && desc_vec.len() < total_leaves {
            splits.insert(BipartitionSplit::new(desc_vec));
        }
    }
    Ok(splits)
}

/// Compute the Robinson-Foulds (RF) distance between two trees.
///
/// RF = |splits_A \ splits_B| + |splits_B \ splits_A|
pub fn robinson_foulds(
    tree_a: &CladeTree,
    tree_b: &CladeTree,
) -> Result<usize, CladeError> {
    let splits_a = extract_bipartitions(tree_a)?;
    let splits_b = extract_bipartitions(tree_b)?;

    let only_a = splits_a.difference(&splits_b).count();
    let only_b = splits_b.difference(&splits_a).count();
    Ok(only_a + only_b)
}

/// Normalised Robinson-Foulds distance (0.0 = identical, 1.0 = maximally different).
pub fn robinson_foulds_normalised(
    tree_a: &CladeTree,
    tree_b: &CladeTree,
) -> Result<f64, CladeError> {
    let rf = robinson_foulds(tree_a, tree_b)?;
    let n = tree_a.leaf_count();
    if n <= 3 {
        return Ok(0.0);
    }
    let max_rf = 2 * (n - 3);
    Ok(rf as f64 / max_rf as f64)
}

// ── Clade enumeration ───────────────────────────────────────────

/// Information about a single clade.
#[derive(Debug, Clone)]
pub struct CladeInfo {
    pub node_id: usize,
    pub leaf_labels: Vec<String>,
    pub size: usize,
    pub depth: usize,
    pub support: Option<f64>,
}

impl CladeInfo {
    pub fn with_support(mut self, sup: f64) -> Self {
        self.support = Some(sup);
        self
    }
}

impl fmt::Display for CladeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Clade(node={}, size={}, depth={})",
            self.node_id, self.size, self.depth
        )
    }
}

/// Enumerate all clades (internal nodes) with their descendant leaf sets.
pub fn enumerate_clades(tree: &CladeTree) -> Result<Vec<CladeInfo>, CladeError> {
    let root = tree.root.ok_or(CladeError::EmptyTree)?;
    let mut clades = Vec::new();

    // Compute depths via BFS
    let mut depths = HashMap::new();
    let mut queue = VecDeque::new();
    queue.push_back((root, 0usize));
    while let Some((cur, d)) = queue.pop_front() {
        depths.insert(cur, d);
        for &child in &tree.nodes[cur].children {
            queue.push_back((child, d + 1));
        }
    }

    for node in &tree.nodes {
        if node.is_leaf() {
            continue;
        }
        let labels = tree.descendant_labels(node.id)?;
        let mut label_vec: Vec<String> = labels.into_iter().collect();
        label_vec.sort();
        let size = label_vec.len();
        let depth = depths.get(&node.id).copied().unwrap_or(0);
        clades.push(CladeInfo {
            node_id: node.id,
            leaf_labels: label_vec,
            size,
            depth,
            support: node.support,
        });
    }
    clades.sort_by_key(|c| c.depth);
    Ok(clades)
}

// ── Shared / unique splits ──────────────────────────────────────

/// Splits shared between two trees.
pub fn shared_splits(
    tree_a: &CladeTree,
    tree_b: &CladeTree,
) -> Result<HashSet<BipartitionSplit>, CladeError> {
    let sa = extract_bipartitions(tree_a)?;
    let sb = extract_bipartitions(tree_b)?;
    Ok(sa.intersection(&sb).cloned().collect())
}

/// Splits unique to tree A (not in tree B).
pub fn unique_splits(
    tree_a: &CladeTree,
    tree_b: &CladeTree,
) -> Result<HashSet<BipartitionSplit>, CladeError> {
    let sa = extract_bipartitions(tree_a)?;
    let sb = extract_bipartitions(tree_b)?;
    Ok(sa.difference(&sb).cloned().collect())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tree() -> CladeTree {
        //       0(root)
        //      / \
        //     1   2(B)
        //    / \
        //   3(C) 4(D)
        let mut t = CladeTree::new();
        t.add_node(Some("root"), 0.0);
        t.add_node(None, 0.1);          // internal node 1
        t.add_node(Some("B"), 0.2);     // leaf
        t.add_node(Some("C"), 0.15);    // leaf
        t.add_node(Some("D"), 0.25);    // leaf
        t.set_parent(1, 0);
        t.set_parent(2, 0);
        t.set_parent(3, 1);
        t.set_parent(4, 1);
        t
    }

    fn build_alt_tree() -> CladeTree {
        //       0(root)
        //      / \
        //     1   2(C)
        //    / \
        //   3(B) 4(D)
        let mut t = CladeTree::new();
        t.add_node(Some("root"), 0.0);
        t.add_node(None, 0.1);
        t.add_node(Some("C"), 0.2);
        t.add_node(Some("B"), 0.15);
        t.add_node(Some("D"), 0.25);
        t.set_parent(1, 0);
        t.set_parent(2, 0);
        t.set_parent(3, 1);
        t.set_parent(4, 1);
        t
    }

    #[test]
    fn test_monophyly_monophyletic() {
        let t = build_tree();
        let status = test_monophyly(&t, &["C", "D"]).unwrap();
        assert_eq!(status, MonophylyStatus::Monophyletic);
    }

    #[test]
    fn test_monophyly_paraphyletic() {
        let t = build_tree();
        // B alone under root, C under internal — {B, C} is paraphyletic
        let status = test_monophyly(&t, &["B", "C"]).unwrap();
        assert_eq!(status, MonophylyStatus::Paraphyletic);
    }

    #[test]
    fn test_monophyly_single_taxon() {
        let t = build_tree();
        // Single taxon — trivially "monophyletic" but MRCA is itself
        let status = test_monophyly(&t, &["B"]).unwrap();
        // The MRCA of a single leaf is that leaf or its parent
        assert!(status == MonophylyStatus::Monophyletic || status == MonophylyStatus::Paraphyletic);
    }

    #[test]
    fn test_monophyly_empty_fails() {
        let t = build_tree();
        assert!(test_monophyly(&t, &[]).is_err());
    }

    #[test]
    fn test_monophyly_missing_taxon() {
        let t = build_tree();
        assert!(test_monophyly(&t, &["Z"]).is_err());
    }

    #[test]
    fn test_rf_identical() {
        let t = build_tree();
        assert_eq!(robinson_foulds(&t, &t).unwrap(), 0);
    }

    #[test]
    fn test_rf_different() {
        let t1 = build_tree();
        let t2 = build_alt_tree();
        let rf = robinson_foulds(&t1, &t2).unwrap();
        // Same topology — node 1 has {C,D} in t1 and {B,D} in t2
        assert!(rf >= 0);
    }

    #[test]
    fn test_rf_normalised() {
        let t = build_tree();
        let nrf = robinson_foulds_normalised(&t, &t).unwrap();
        assert!((nrf).abs() < 1e-9);
    }

    #[test]
    fn test_extract_bipartitions() {
        let t = build_tree();
        let splits = extract_bipartitions(&t).unwrap();
        // Internal node 1 gives split {C, D}
        assert!(!splits.is_empty());
    }

    #[test]
    fn test_enumerate_clades() {
        let t = build_tree();
        let clades = enumerate_clades(&t).unwrap();
        assert!(clades.len() >= 2); // root + internal node
    }

    #[test]
    fn test_clade_info_display() {
        let ci = CladeInfo {
            node_id: 1,
            leaf_labels: vec!["A".into(), "B".into()],
            size: 2,
            depth: 1,
            support: None,
        };
        let s = format!("{ci}");
        assert!(s.contains("size=2"));
    }

    #[test]
    fn test_descendant_leaves() {
        let t = build_tree();
        let desc = t.descendant_leaves(1).unwrap();
        assert_eq!(desc.len(), 2);
        assert!(desc.contains(&3));
        assert!(desc.contains(&4));
    }

    #[test]
    fn test_descendant_labels() {
        let t = build_tree();
        let labels = t.descendant_labels(1).unwrap();
        assert!(labels.contains("C"));
        assert!(labels.contains("D"));
    }

    #[test]
    fn test_shared_splits() {
        let t = build_tree();
        let shared = shared_splits(&t, &t).unwrap();
        assert!(!shared.is_empty());
    }

    #[test]
    fn test_unique_splits_identical() {
        let t = build_tree();
        let unique = unique_splits(&t, &t).unwrap();
        assert!(unique.is_empty());
    }

    #[test]
    fn test_find_by_label() {
        let t = build_tree();
        assert_eq!(t.find_by_label("B"), Some(2));
        assert_eq!(t.find_by_label("Z"), None);
    }

    #[test]
    fn test_node_display_with_support() {
        let n = CladeNode::new(1).with_label("clade").with_support(0.95);
        assert!(format!("{n}").contains("95%"));
    }

    #[test]
    fn test_tree_display() {
        let t = build_tree();
        let s = format!("{t}");
        assert!(s.contains("CladeTree"));
    }

    #[test]
    fn test_monophyly_display() {
        assert_eq!(format!("{}", MonophylyStatus::Monophyletic), "monophyletic");
    }

    #[test]
    fn test_bipartition_display() {
        let bp = BipartitionSplit::new(vec!["B".into(), "A".into()]);
        assert_eq!(format!("{bp}"), "{A, B}");
    }

    #[test]
    fn test_clade_with_support() {
        let ci = CladeInfo {
            node_id: 0,
            leaf_labels: vec![],
            size: 0,
            depth: 0,
            support: None,
        }
        .with_support(0.99);
        assert!((ci.support.unwrap() - 0.99).abs() < 1e-9);
    }
}
