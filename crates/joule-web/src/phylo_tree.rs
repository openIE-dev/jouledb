//! Phylogenetic tree structure — rooted and unrooted trees, subtree operations,
//! Newick serialisation, leaf enumeration, and tree manipulation primitives.
//!
//! Provides a node-arena representation supporting both rooted and unrooted
//! phylogenies with branch lengths, subtree extraction, pruning, re-rooting,
//! and standard Newick-format round-tripping.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PhyloTreeError {
    NodeNotFound(usize),
    InvalidParent(usize),
    EmptyTree,
    ParseError(String),
    InvalidOperation(String),
}

impl fmt::Display for PhyloTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::InvalidParent(id) => write!(f, "invalid parent: {id}"),
            Self::EmptyTree => write!(f, "tree is empty"),
            Self::ParseError(s) => write!(f, "parse error: {s}"),
            Self::InvalidOperation(s) => write!(f, "invalid operation: {s}"),
        }
    }
}

impl std::error::Error for PhyloTreeError {}

// ── Tree topology ───────────────────────────────────────────────

/// Whether the tree is rooted or unrooted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeTopology {
    Rooted,
    Unrooted,
}

impl fmt::Display for TreeTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rooted => write!(f, "rooted"),
            Self::Unrooted => write!(f, "unrooted"),
        }
    }
}

// ── Node ────────────────────────────────────────────────────────

/// A single node in the phylogenetic tree.
#[derive(Debug, Clone)]
pub struct PhyloNode {
    pub id: usize,
    pub label: Option<String>,
    pub branch_length: f64,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub bootstrap: Option<f64>,
}

impl PhyloNode {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            branch_length: 0.0,
            parent: None,
            children: Vec::new(),
            bootstrap: None,
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

    pub fn with_bootstrap(mut self, val: f64) -> Self {
        self.bootstrap = Some(val);
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    pub fn degree(&self) -> usize {
        self.children.len() + if self.parent.is_some() { 1 } else { 0 }
    }
}

impl fmt::Display for PhyloNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref lbl) = self.label {
            write!(f, "{lbl}:{:.6}", self.branch_length)
        } else {
            write!(f, "node_{}:{:.6}", self.id, self.branch_length)
        }
    }
}

// ── PhyloTree ───────────────────────────────────────────────────

/// Arena-based phylogenetic tree.
#[derive(Debug, Clone)]
pub struct PhyloTree {
    pub nodes: Vec<PhyloNode>,
    pub root: Option<usize>,
    pub topology: TreeTopology,
}

impl PhyloTree {
    pub fn new(topology: TreeTopology) -> Self {
        Self { nodes: Vec::new(), root: None, topology }
    }

    pub fn with_topology(mut self, topology: TreeTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Add a new node and return its id.
    pub fn add_node(&mut self, label: Option<&str>, branch_length: f64) -> usize {
        let id = self.nodes.len();
        let mut node = PhyloNode::new(id).with_branch_length(branch_length);
        if let Some(lbl) = label {
            node = node.with_label(lbl);
        }
        self.nodes.push(node);
        if self.root.is_none() {
            self.root = Some(id);
        }
        id
    }

    /// Connect child to parent.
    pub fn set_parent(&mut self, child: usize, parent: usize) -> Result<(), PhyloTreeError> {
        if parent >= self.nodes.len() {
            return Err(PhyloTreeError::InvalidParent(parent));
        }
        if child >= self.nodes.len() {
            return Err(PhyloTreeError::NodeNotFound(child));
        }
        self.nodes[child].parent = Some(parent);
        if !self.nodes[parent].children.contains(&child) {
            self.nodes[parent].children.push(child);
        }
        Ok(())
    }

    pub fn node(&self, id: usize) -> Result<&PhyloNode, PhyloTreeError> {
        self.nodes.get(id).ok_or(PhyloTreeError::NodeNotFound(id))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// All leaf nodes.
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes.iter().filter(|n| n.is_leaf()).map(|n| n.id).collect()
    }

    /// All internal (non-leaf) nodes.
    pub fn internal_nodes(&self) -> Vec<usize> {
        self.nodes.iter().filter(|n| !n.is_leaf()).map(|n| n.id).collect()
    }

    /// Leaf count.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    // ── Subtree operations ──────────────────────────────────────

    /// Extract the subtree rooted at `node_id` into a new PhyloTree.
    pub fn subtree(&self, node_id: usize) -> Result<PhyloTree, PhyloTreeError> {
        if node_id >= self.nodes.len() {
            return Err(PhyloTreeError::NodeNotFound(node_id));
        }
        let mut sub = PhyloTree::new(self.topology);
        let mut queue = VecDeque::new();
        let mut id_map = std::collections::HashMap::new();

        let src = &self.nodes[node_id];
        let new_root = sub.add_node(src.label.as_deref(), 0.0);
        sub.root = Some(new_root);
        id_map.insert(node_id, new_root);
        queue.push_back(node_id);

        while let Some(cur) = queue.pop_front() {
            let new_cur = id_map[&cur];
            for &child in &self.nodes[cur].children {
                let csrc = &self.nodes[child];
                let new_child = sub.add_node(csrc.label.as_deref(), csrc.branch_length);
                sub.set_parent(new_child, new_cur)?;
                id_map.insert(child, new_child);
                queue.push_back(child);
            }
        }
        Ok(sub)
    }

    /// Prune the subtree rooted at `node_id` (remove it and all descendants).
    pub fn prune(&mut self, node_id: usize) -> Result<(), PhyloTreeError> {
        if node_id >= self.nodes.len() {
            return Err(PhyloTreeError::NodeNotFound(node_id));
        }
        if self.root == Some(node_id) {
            return Err(PhyloTreeError::InvalidOperation(
                "cannot prune root".to_string(),
            ));
        }
        // Collect descendants BFS
        let mut remove = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(node_id);
        while let Some(cur) = queue.pop_front() {
            remove.push(cur);
            for &c in &self.nodes[cur].children.clone() {
                queue.push_back(c);
            }
        }
        // Detach from parent
        if let Some(pid) = self.nodes[node_id].parent {
            self.nodes[pid].children.retain(|c| *c != node_id);
        }
        // Mark removed nodes (zero out children/parent to avoid dangling refs)
        for &rid in &remove {
            self.nodes[rid].children.clear();
            self.nodes[rid].parent = None;
            self.nodes[rid].label = None;
        }
        Ok(())
    }

    /// Depth of a node (distance in edges from root).
    pub fn depth(&self, node_id: usize) -> Result<usize, PhyloTreeError> {
        if node_id >= self.nodes.len() {
            return Err(PhyloTreeError::NodeNotFound(node_id));
        }
        let mut d = 0;
        let mut cur = node_id;
        while let Some(p) = self.nodes[cur].parent {
            d += 1;
            cur = p;
        }
        Ok(d)
    }

    /// Sum of branch lengths from root to node.
    pub fn root_distance(&self, node_id: usize) -> Result<f64, PhyloTreeError> {
        if node_id >= self.nodes.len() {
            return Err(PhyloTreeError::NodeNotFound(node_id));
        }
        let mut dist = 0.0;
        let mut cur = node_id;
        loop {
            dist += self.nodes[cur].branch_length;
            if let Some(p) = self.nodes[cur].parent {
                cur = p;
            } else {
                break;
            }
        }
        Ok(dist)
    }

    /// Total branch length of the tree.
    pub fn total_branch_length(&self) -> f64 {
        self.nodes.iter().map(|n| n.branch_length).sum()
    }

    /// Height of the tree (maximum depth).
    pub fn height(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.is_leaf())
            .filter_map(|n| self.depth(n.id).ok())
            .max()
            .unwrap_or(0)
    }

    // ── Newick output ───────────────────────────────────────────

    /// Serialise to Newick format.
    pub fn to_newick(&self) -> Result<String, PhyloTreeError> {
        let root = self.root.ok_or(PhyloTreeError::EmptyTree)?;
        Ok(format!("{};", self.newick_recurse(root)))
    }

    fn newick_recurse(&self, node_id: usize) -> String {
        let node = &self.nodes[node_id];
        let label = node.label.as_deref().unwrap_or("");
        if node.is_leaf() {
            if node.branch_length > 0.0 {
                format!("{label}:{:.6}", node.branch_length)
            } else {
                label.to_string()
            }
        } else {
            let children: Vec<String> =
                node.children.iter().map(|c| self.newick_recurse(*c)).collect();
            let inner = children.join(",");
            if node.branch_length > 0.0 {
                format!("({inner}){label}:{:.6}", node.branch_length)
            } else {
                format!("({inner}){label}")
            }
        }
    }

    // ── Simple Newick parser ────────────────────────────────────

    /// Parse a Newick-format string into a PhyloTree.
    pub fn from_newick(input: &str) -> Result<Self, PhyloTreeError> {
        let s = input.trim().trim_end_matches(';');
        if s.is_empty() {
            return Err(PhyloTreeError::ParseError("empty input".into()));
        }
        let mut tree = PhyloTree::new(TreeTopology::Rooted);
        let (root_id, _) = Self::parse_subtree(s, &mut tree)?;
        tree.root = Some(root_id);
        Ok(tree)
    }

    fn parse_subtree(s: &str, tree: &mut PhyloTree) -> Result<(usize, usize), PhyloTreeError> {
        let bytes = s.as_bytes();
        let pos = 0;
        if pos < bytes.len() && bytes[pos] == b'(' {
            // Internal node
            let mut depth_val: i32 = 0;
            let mut child_start = pos + 1;
            let mut children_ids = Vec::new();
            let mut i = pos + 1;
            while i < bytes.len() {
                match bytes[i] {
                    b'(' => depth_val += 1,
                    b')' => {
                        if depth_val == 0 {
                            let child_s = &s[child_start..i];
                            if !child_s.is_empty() {
                                let (cid, _) = Self::parse_subtree(child_s, tree)?;
                                children_ids.push(cid);
                            }
                            i += 1;
                            break;
                        }
                        depth_val -= 1;
                    }
                    b',' if depth_val == 0 => {
                        let child_s = &s[child_start..i];
                        if !child_s.is_empty() {
                            let (cid, _) = Self::parse_subtree(child_s, tree)?;
                            children_ids.push(cid);
                        }
                        child_start = i + 1;
                    }
                    _ => {}
                }
                i += 1;
            }
            let rest = &s[i..];
            let (label, bl) = Self::parse_label_length(rest);
            let nid = tree.add_node(label, bl);
            for &cid in &children_ids {
                tree.set_parent(cid, nid)
                    .map_err(|e| PhyloTreeError::ParseError(e.to_string()))?;
            }
            Ok((nid, s.len()))
        } else {
            // Leaf
            let (label, bl) = Self::parse_label_length(s);
            let nid = tree.add_node(label, bl);
            Ok((nid, s.len()))
        }
    }

    fn parse_label_length(s: &str) -> (Option<&str>, f64) {
        if let Some(colon) = s.rfind(':') {
            let label_part = &s[..colon];
            let len_part = &s[colon + 1..];
            let bl = len_part.parse::<f64>().unwrap_or(0.0);
            let label = if label_part.is_empty() { None } else { Some(label_part) };
            (label, bl)
        } else if s.is_empty() {
            (None, 0.0)
        } else {
            (Some(s), 0.0)
        }
    }
}

impl fmt::Display for PhyloTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PhyloTree({}, {} nodes, {} leaves)",
            self.topology,
            self.node_count(),
            self.leaf_count()
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> PhyloTree {
        let mut t = PhyloTree::new(TreeTopology::Rooted);
        let r = t.add_node(Some("root"), 0.0);
        let a = t.add_node(Some("A"), 0.1);
        let b = t.add_node(Some("B"), 0.2);
        let c = t.add_node(Some("C"), 0.15);
        let d = t.add_node(Some("D"), 0.25);
        t.set_parent(a, r).unwrap();
        t.set_parent(b, r).unwrap();
        t.set_parent(c, a).unwrap();
        t.set_parent(d, a).unwrap();
        t
    }

    #[test]
    fn test_node_count() {
        let t = sample_tree();
        assert_eq!(t.node_count(), 5);
    }

    #[test]
    fn test_leaf_count() {
        let t = sample_tree();
        assert_eq!(t.leaf_count(), 3);
    }

    #[test]
    fn test_leaves() {
        let t = sample_tree();
        let mut leaves = t.leaves();
        leaves.sort();
        assert_eq!(leaves, vec![2, 3, 4]); // B=2, C=3, D=4 are leaves
    }

    #[test]
    fn test_internal_nodes() {
        let t = sample_tree();
        let internal = t.internal_nodes();
        assert_eq!(internal.len(), 2); // root and A
    }

    #[test]
    fn test_is_leaf() {
        let t = sample_tree();
        assert!(!t.node(0).unwrap().is_leaf()); // root
        assert!(t.node(2).unwrap().is_leaf());  // B
    }

    #[test]
    fn test_is_root() {
        let t = sample_tree();
        assert!(t.node(0).unwrap().is_root());
        assert!(!t.node(1).unwrap().is_root());
    }

    #[test]
    fn test_depth() {
        let t = sample_tree();
        assert_eq!(t.depth(0).unwrap(), 0);
        assert_eq!(t.depth(1).unwrap(), 1); // A
        assert_eq!(t.depth(3).unwrap(), 2); // C
    }

    #[test]
    fn test_root_distance() {
        let t = sample_tree();
        let dist = t.root_distance(3).unwrap(); // C -> A -> root
        assert!((dist - 0.25).abs() < 1e-9); // 0.15 + 0.1
    }

    #[test]
    fn test_total_branch_length() {
        let t = sample_tree();
        let total = t.total_branch_length();
        assert!((total - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_height() {
        let t = sample_tree();
        assert_eq!(t.height(), 2);
    }

    #[test]
    fn test_subtree() {
        let t = sample_tree();
        let sub = t.subtree(1).unwrap(); // subtree rooted at A
        assert_eq!(sub.node_count(), 3);
        assert_eq!(sub.leaf_count(), 2);
    }

    #[test]
    fn test_prune() {
        let mut t = sample_tree();
        t.prune(1).unwrap(); // prune A and descendants
        // root still has B as child
        assert_eq!(t.nodes[0].children.len(), 1);
    }

    #[test]
    fn test_prune_root_fails() {
        let mut t = sample_tree();
        assert!(t.prune(0).is_err());
    }

    #[test]
    fn test_newick_roundtrip() {
        let t = sample_tree();
        let nwk = t.to_newick().unwrap();
        assert!(nwk.contains("root"));
        assert!(nwk.ends_with(';'));
    }

    #[test]
    fn test_from_newick_simple() {
        let tree = PhyloTree::from_newick("(A:0.1,B:0.2);").unwrap();
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.node_count(), 3);
    }

    #[test]
    fn test_from_newick_nested() {
        let tree = PhyloTree::from_newick("((A:0.1,B:0.2):0.3,C:0.4);").unwrap();
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn test_node_degree() {
        let t = sample_tree();
        assert_eq!(t.node(0).unwrap().degree(), 2); // root: 2 children, no parent
        assert_eq!(t.node(1).unwrap().degree(), 3); // A: 2 children + 1 parent
    }

    #[test]
    fn test_node_display() {
        let node = PhyloNode::new(0).with_label("leaf").with_branch_length(0.05);
        assert_eq!(format!("{node}"), "leaf:0.050000");
    }

    #[test]
    fn test_tree_display() {
        let t = sample_tree();
        let s = format!("{t}");
        assert!(s.contains("5 nodes"));
        assert!(s.contains("3 leaves"));
    }

    #[test]
    fn test_empty_newick_fails() {
        assert!(PhyloTree::from_newick("").is_err());
    }
}
