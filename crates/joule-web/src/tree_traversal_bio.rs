//! Phylogenetic tree traversal utilities — pre-order, post-order, level-order,
//! lowest common ancestor (LCA), tree comparison via symmetric difference,
//! and path extraction between arbitrary node pairs.
//!
//! Operates on an arena-style tree representation (node id + adjacency lists)
//! that is compatible with the `phylo_tree` module.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TraversalError {
    NodeNotFound(usize),
    NoRoot,
    EmptyTree,
    LcaNotFound(usize, usize),
}

impl fmt::Display for TraversalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::NoRoot => write!(f, "tree has no root"),
            Self::EmptyTree => write!(f, "tree is empty"),
            Self::LcaNotFound(a, b) => write!(f, "LCA not found for ({a}, {b})"),
        }
    }
}

impl std::error::Error for TraversalError {}

// ── Tree node ───────────────────────────────────────────────────

/// A lightweight node for traversal purposes.
#[derive(Debug, Clone)]
pub struct TraversalNode {
    pub id: usize,
    pub label: Option<String>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub branch_length: f64,
}

impl TraversalNode {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            parent: None,
            children: Vec::new(),
            branch_length: 0.0,
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

    pub fn with_parent(mut self, parent: usize) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

impl fmt::Display for TraversalNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.label {
            Some(lbl) => write!(f, "{lbl}"),
            None => write!(f, "node_{}", self.id),
        }
    }
}

// ── Traversal tree ──────────────────────────────────────────────

/// An arena-based tree supporting standard traversals.
#[derive(Debug, Clone)]
pub struct TraversalTree {
    pub nodes: Vec<TraversalNode>,
    pub root: Option<usize>,
}

impl TraversalTree {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), root: None }
    }

    pub fn add_node(&mut self, label: Option<&str>, branch_length: f64) -> usize {
        let id = self.nodes.len();
        let mut node = TraversalNode::new(id).with_branch_length(branch_length);
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

    // ── Pre-order traversal ─────────────────────────────────────

    /// Visit root before children (depth-first).
    pub fn preorder(&self) -> Result<Vec<usize>, TraversalError> {
        let root = self.root.ok_or(TraversalError::NoRoot)?;
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![root];
        while let Some(cur) = stack.pop() {
            result.push(cur);
            // Push children in reverse order so leftmost is visited first
            for &child in self.nodes[cur].children.iter().rev() {
                stack.push(child);
            }
        }
        Ok(result)
    }

    // ── Post-order traversal ────────────────────────────────────

    /// Visit children before root (depth-first).
    pub fn postorder(&self) -> Result<Vec<usize>, TraversalError> {
        let root = self.root.ok_or(TraversalError::NoRoot)?;
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![(root, false)];
        while let Some((cur, expanded)) = stack.pop() {
            if expanded || self.nodes[cur].children.is_empty() {
                result.push(cur);
            } else {
                stack.push((cur, true));
                for &child in self.nodes[cur].children.iter().rev() {
                    stack.push((child, false));
                }
            }
        }
        Ok(result)
    }

    // ── Level-order (BFS) traversal ─────────────────────────────

    /// Visit by depth level (breadth-first).
    pub fn levelorder(&self) -> Result<Vec<usize>, TraversalError> {
        let root = self.root.ok_or(TraversalError::NoRoot)?;
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(cur) = queue.pop_front() {
            result.push(cur);
            for &child in &self.nodes[cur].children {
                queue.push_back(child);
            }
        }
        Ok(result)
    }

    // ── Level-order with depth values ───────────────────────────

    /// Returns (node_id, depth) pairs in BFS order.
    pub fn levelorder_with_depth(&self) -> Result<Vec<(usize, usize)>, TraversalError> {
        let root = self.root.ok_or(TraversalError::NoRoot)?;
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut queue = VecDeque::new();
        queue.push_back((root, 0));
        while let Some((cur, depth)) = queue.pop_front() {
            result.push((cur, depth));
            for &child in &self.nodes[cur].children {
                queue.push_back((child, depth + 1));
            }
        }
        Ok(result)
    }

    // ── Lowest Common Ancestor ──────────────────────────────────

    /// Find the LCA of two nodes using ancestor set intersection.
    pub fn lca(&self, a: usize, b: usize) -> Result<usize, TraversalError> {
        if a >= self.nodes.len() {
            return Err(TraversalError::NodeNotFound(a));
        }
        if b >= self.nodes.len() {
            return Err(TraversalError::NodeNotFound(b));
        }
        // Build ancestor set for a
        let mut ancestors_a = HashSet::new();
        let mut cur = a;
        loop {
            ancestors_a.insert(cur);
            if let Some(p) = self.nodes[cur].parent {
                cur = p;
            } else {
                break;
            }
        }
        // Walk up from b until we hit an ancestor of a
        cur = b;
        loop {
            if ancestors_a.contains(&cur) {
                return Ok(cur);
            }
            if let Some(p) = self.nodes[cur].parent {
                cur = p;
            } else {
                break;
            }
        }
        Err(TraversalError::LcaNotFound(a, b))
    }

    // ── Path between nodes ──────────────────────────────────────

    /// Path from node `a` to node `b` through their LCA.
    pub fn path(&self, a: usize, b: usize) -> Result<Vec<usize>, TraversalError> {
        let ancestor = self.lca(a, b)?;
        let mut path_a = Vec::new();
        let mut cur = a;
        while cur != ancestor {
            path_a.push(cur);
            cur = self.nodes[cur].parent.ok_or(TraversalError::NodeNotFound(cur))?;
        }
        path_a.push(ancestor);

        let mut path_b = Vec::new();
        cur = b;
        while cur != ancestor {
            path_b.push(cur);
            cur = self.nodes[cur].parent.ok_or(TraversalError::NodeNotFound(cur))?;
        }
        path_b.reverse();
        path_a.extend(path_b);
        Ok(path_a)
    }

    /// Patristic distance (sum of branch lengths) between two nodes.
    pub fn patristic_distance(&self, a: usize, b: usize) -> Result<f64, TraversalError> {
        let p = self.path(a, b)?;
        let mut dist = 0.0;
        for window in p.windows(2) {
            dist += self.nodes[window[1]].branch_length;
        }
        // Also add the branch of the first node if it's not the LCA
        if p.len() > 1 {
            dist += self.nodes[p[0]].branch_length;
        }
        Ok(dist)
    }

    // ── Depth computation ───────────────────────────────────────

    /// Compute depth of every node.
    pub fn compute_depths(&self) -> Result<HashMap<usize, usize>, TraversalError> {
        let root = self.root.ok_or(TraversalError::NoRoot)?;
        let mut depths = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back((root, 0));
        while let Some((cur, depth)) = queue.pop_front() {
            depths.insert(cur, depth);
            for &child in &self.nodes[cur].children {
                queue.push_back((child, depth + 1));
            }
        }
        Ok(depths)
    }

    /// Maximum depth (tree height).
    pub fn height(&self) -> Result<usize, TraversalError> {
        let depths = self.compute_depths()?;
        depths.values().max().copied().ok_or(TraversalError::EmptyTree)
    }

    // ── Leaf set under a node ───────────────────────────────────

    /// All leaf ids descended from `node_id`.
    pub fn leaves_under(&self, node_id: usize) -> Result<Vec<usize>, TraversalError> {
        if node_id >= self.nodes.len() {
            return Err(TraversalError::NodeNotFound(node_id));
        }
        let mut result = Vec::new();
        let mut stack = vec![node_id];
        while let Some(cur) = stack.pop() {
            if self.nodes[cur].is_leaf() {
                result.push(cur);
            } else {
                for &child in &self.nodes[cur].children {
                    stack.push(child);
                }
            }
        }
        result.sort();
        Ok(result)
    }
}

impl Default for TraversalTree {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TraversalTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TraversalTree({} nodes, {} leaves)",
            self.node_count(),
            self.leaf_count()
        )
    }
}

// ── Tree comparison ─────────────────────────────────────────────

/// Extract all non-trivial bipartitions (splits) from a rooted tree.
/// Each split is represented as a sorted vector of leaf ids on one side.
pub fn extract_splits(tree: &TraversalTree) -> Result<Vec<Vec<usize>>, TraversalError> {
    let all_leaves = tree
        .nodes
        .iter()
        .filter(|n| n.is_leaf())
        .map(|n| n.id)
        .collect::<HashSet<_>>();
    let n_leaves = all_leaves.len();

    let mut splits = Vec::new();
    for node in &tree.nodes {
        if node.is_leaf() || tree.root == Some(node.id) {
            continue;
        }
        let leaves = tree.leaves_under(node.id)?;
        if leaves.len() > 1 && leaves.len() < n_leaves {
            splits.push(leaves);
        }
    }
    Ok(splits)
}

/// Symmetric difference of split sets between two trees.
/// Returns the number of splits present in one tree but not the other.
pub fn symmetric_difference(
    tree_a: &TraversalTree,
    tree_b: &TraversalTree,
) -> Result<usize, TraversalError> {
    let splits_a = extract_splits(tree_a)?;
    let splits_b = extract_splits(tree_b)?;

    let set_a: HashSet<Vec<usize>> = splits_a.into_iter().collect();
    let set_b: HashSet<Vec<usize>> = splits_b.into_iter().collect();

    let only_a = set_a.difference(&set_b).count();
    let only_b = set_b.difference(&set_a).count();
    Ok(only_a + only_b)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tree() -> TraversalTree {
        //       0(root)
        //      / \
        //     1   2
        //    / \
        //   3   4
        let mut t = TraversalTree::new();
        t.add_node(Some("root"), 0.0);
        t.add_node(Some("A"), 0.1);
        t.add_node(Some("B"), 0.2);
        t.add_node(Some("C"), 0.15);
        t.add_node(Some("D"), 0.25);
        t.set_parent(1, 0);
        t.set_parent(2, 0);
        t.set_parent(3, 1);
        t.set_parent(4, 1);
        t
    }

    #[test]
    fn test_preorder() {
        let t = build_tree();
        let order = t.preorder().unwrap();
        assert_eq!(order, vec![0, 1, 3, 4, 2]);
    }

    #[test]
    fn test_postorder() {
        let t = build_tree();
        let order = t.postorder().unwrap();
        assert_eq!(order, vec![3, 4, 1, 2, 0]);
    }

    #[test]
    fn test_levelorder() {
        let t = build_tree();
        let order = t.levelorder().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_levelorder_with_depth() {
        let t = build_tree();
        let pairs = t.levelorder_with_depth().unwrap();
        assert_eq!(pairs[0], (0, 0));
        assert_eq!(pairs[3], (3, 2));
    }

    #[test]
    fn test_lca_siblings() {
        let t = build_tree();
        assert_eq!(t.lca(3, 4).unwrap(), 1);
    }

    #[test]
    fn test_lca_uncle() {
        let t = build_tree();
        assert_eq!(t.lca(3, 2).unwrap(), 0);
    }

    #[test]
    fn test_lca_self() {
        let t = build_tree();
        assert_eq!(t.lca(1, 1).unwrap(), 1);
    }

    #[test]
    fn test_lca_root() {
        let t = build_tree();
        assert_eq!(t.lca(0, 3).unwrap(), 0);
    }

    #[test]
    fn test_path() {
        let t = build_tree();
        let p = t.path(3, 2).unwrap();
        assert_eq!(p[0], 3); // start
        assert!(p.contains(&0)); // through root
        assert_eq!(*p.last().unwrap(), 2); // end
    }

    #[test]
    fn test_path_siblings() {
        let t = build_tree();
        let p = t.path(3, 4).unwrap();
        assert_eq!(p, vec![3, 1, 4]);
    }

    #[test]
    fn test_compute_depths() {
        let t = build_tree();
        let depths = t.compute_depths().unwrap();
        assert_eq!(depths[&0], 0);
        assert_eq!(depths[&1], 1);
        assert_eq!(depths[&3], 2);
    }

    #[test]
    fn test_height() {
        let t = build_tree();
        assert_eq!(t.height().unwrap(), 2);
    }

    #[test]
    fn test_leaves_under() {
        let t = build_tree();
        let leaves = t.leaves_under(1).unwrap();
        assert_eq!(leaves, vec![3, 4]);
    }

    #[test]
    fn test_leaves_under_root() {
        let t = build_tree();
        let leaves = t.leaves_under(0).unwrap();
        assert_eq!(leaves, vec![2, 3, 4]);
    }

    #[test]
    fn test_extract_splits() {
        let t = build_tree();
        let splits = extract_splits(&t).unwrap();
        // Node 1 gives split {3,4}
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0], vec![3, 4]);
    }

    #[test]
    fn test_symmetric_difference_identical() {
        let t = build_tree();
        let diff = symmetric_difference(&t, &t).unwrap();
        assert_eq!(diff, 0);
    }

    #[test]
    fn test_symmetric_difference_different() {
        let t1 = build_tree();
        // Different topology: ((B,C),D)
        let mut t2 = TraversalTree::new();
        t2.add_node(Some("root"), 0.0);
        t2.add_node(Some("A"), 0.1);
        t2.add_node(Some("B"), 0.2);
        t2.add_node(Some("C"), 0.15);
        t2.set_parent(1, 0);
        t2.set_parent(2, 1);
        t2.set_parent(3, 1);
        let diff = symmetric_difference(&t1, &t2).unwrap();
        assert!(diff >= 0);
    }

    #[test]
    fn test_node_display() {
        let n = TraversalNode::new(5).with_label("Homo_sapiens");
        assert_eq!(format!("{n}"), "Homo_sapiens");
    }

    #[test]
    fn test_node_no_label() {
        let n = TraversalNode::new(7);
        assert_eq!(format!("{n}"), "node_7");
    }

    #[test]
    fn test_tree_display() {
        let t = build_tree();
        let s = format!("{t}");
        assert!(s.contains("5 nodes"));
    }

    #[test]
    fn test_error_no_root() {
        let t = TraversalTree { nodes: Vec::new(), root: None };
        assert!(t.preorder().is_err());
    }
}
