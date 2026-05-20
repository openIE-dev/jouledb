//! Newick tree format parser — branch lengths, bootstrap values, tree operations.
//!
//! Parses the Newick (New Hampshire) phylogenetic tree format with support
//! for labeled and unlabeled nodes, branch lengths, bootstrap/support
//! values, multiple trees, and tree traversal operations.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NewickError {
    EmptyInput,
    UnbalancedParentheses,
    MissingSemicolon,
    InvalidBranchLength(String),
    InvalidBootstrap(String),
    MalformedTree(String),
}

impl fmt::Display for NewickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty Newick input"),
            Self::UnbalancedParentheses => write!(f, "unbalanced parentheses"),
            Self::MissingSemicolon => write!(f, "missing terminal semicolon"),
            Self::InvalidBranchLength(s) => write!(f, "invalid branch length: {s}"),
            Self::InvalidBootstrap(s) => write!(f, "invalid bootstrap value: {s}"),
            Self::MalformedTree(s) => write!(f, "malformed tree: {s}"),
        }
    }
}

impl std::error::Error for NewickError {}

// ── Tree node ───────────────────────────────────────────────────

/// A node in a phylogenetic tree.
#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub struct TreeNode {
    pub label: Option<String>,
    pub branch_length: Option<f64>,
    pub bootstrap: Option<f64>,
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    pub fn leaf(label: &str) -> Self {
        Self {
            label: Some(label.to_string()),
            branch_length: None,
            bootstrap: None,
            children: Vec::new(),
        }
    }

    pub fn internal() -> Self {
        Self {
            label: None,
            branch_length: None,
            bootstrap: None,
            children: Vec::new(),
        }
    }

    pub fn with_branch_length(mut self, bl: f64) -> Self {
        self.branch_length = Some(bl);
        self
    }

    pub fn with_bootstrap(mut self, bs: f64) -> Self {
        self.bootstrap = Some(bs);
        self
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_child(mut self, child: TreeNode) -> Self {
        self.children.push(child);
        self
    }

    /// True if this is a leaf (no children).
    pub fn is_leaf(&self) -> bool { self.children.is_empty() }

    /// True if this is an internal node.
    pub fn is_internal(&self) -> bool { !self.children.is_empty() }

    /// Number of leaves in this subtree.
    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.children.iter().map(|c| c.leaf_count()).sum()
        }
    }

    /// Total number of nodes (internal + leaves).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Maximum depth of the tree.
    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            0
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Total branch length of the subtree.
    pub fn total_branch_length(&self) -> f64 {
        let own = self.branch_length.unwrap_or(0.0);
        own + self.children.iter().map(|c| c.total_branch_length()).sum::<f64>()
    }

    /// Collect all leaf labels.
    pub fn leaf_labels(&self) -> Vec<String> {
        if self.is_leaf() {
            if let Some(ref l) = self.label {
                return vec![l.clone()];
            }
            return Vec::new();
        }
        self.children.iter().flat_map(|c| c.leaf_labels()).collect()
    }

    /// Find a node by label (depth-first).
    pub fn find(&self, label: &str) -> Option<&TreeNode> {
        if self.label.as_deref() == Some(label) {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(label) {
                return Some(found);
            }
        }
        None
    }

    /// Serialize to Newick format.
    pub fn to_newick(&self) -> String {
        let mut s = String::new();
        if !self.children.is_empty() {
            s.push('(');
            for (i, child) in self.children.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&child.to_newick());
            }
            s.push(')');
        }
        if let Some(ref bs) = self.bootstrap {
            if self.is_internal() {
                let formatted = format_float(*bs);
                s.push_str(&formatted);
            }
        }
        if let Some(ref l) = self.label {
            s.push_str(l);
        }
        if let Some(bl) = self.branch_length {
            s.push(':');
            s.push_str(&format_float(bl));
        }
        s
    }
}

impl fmt::Display for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref l) = self.label {
            write!(f, "{l}")?;
        } else {
            write!(f, "(internal)")?;
        }
        if let Some(bl) = self.branch_length {
            write!(f, ":{}", format_float(bl))?;
        }
        if self.is_internal() {
            write!(f, " [{} children]", self.children.len())?;
        }
        Ok(())
    }
}

/// Format a float, stripping trailing zeros.
fn format_float(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        let s = format!("{:.6}", v);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// Newick format parser.
#[derive(Debug, Clone)]
pub struct NewickParser {
    support_as_label: bool,
}

impl NewickParser {
    pub fn new() -> Self {
        Self { support_as_label: false }
    }

    pub fn with_support_as_label(mut self, s: bool) -> Self {
        self.support_as_label = s;
        self
    }

    /// Parse a Newick string (may contain multiple trees separated by `;`).
    pub fn parse(&self, input: &str) -> Result<Vec<TreeNode>, NewickError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(NewickError::EmptyInput);
        }
        let mut trees = Vec::new();
        for part in trimmed.split(';') {
            let t = part.trim();
            if t.is_empty() { continue; }
            let tree = self.parse_tree(t)?;
            trees.push(tree);
        }
        if trees.is_empty() {
            return Err(NewickError::EmptyInput);
        }
        Ok(trees)
    }

    fn parse_tree(&self, input: &str) -> Result<TreeNode, NewickError> {
        let chars: Vec<char> = input.chars().collect();
        let (node, pos) = self.parse_node(&chars, 0)?;
        if pos != chars.len() {
            // Allow trailing whitespace
            let remaining: String = chars[pos..].iter().collect();
            if !remaining.trim().is_empty() {
                return Err(NewickError::MalformedTree(
                    format!("unexpected characters at position {pos}"),
                ));
            }
        }
        Ok(node)
    }

    fn parse_node(&self, chars: &[char], start: usize) -> Result<(TreeNode, usize), NewickError> {
        let mut pos = start;
        let mut node = TreeNode::internal();

        // Check for children
        if pos < chars.len() && chars[pos] == '(' {
            pos += 1; // skip '('
            loop {
                let (child, new_pos) = self.parse_node(chars, pos)?;
                node.children.push(child);
                pos = new_pos;
                if pos >= chars.len() {
                    return Err(NewickError::UnbalancedParentheses);
                }
                if chars[pos] == ',' {
                    pos += 1;
                } else if chars[pos] == ')' {
                    pos += 1;
                    break;
                } else {
                    return Err(NewickError::MalformedTree(
                        format!("expected ',' or ')' at position {pos}"),
                    ));
                }
            }
        }

        // Parse label (and possibly bootstrap for internal nodes)
        let label_start = pos;
        while pos < chars.len() && !matches!(chars[pos], ':' | ',' | ')' | ';' | '(') {
            pos += 1;
        }
        let label_str: String = chars[label_start..pos].iter().collect();
        let label_trimmed = label_str.trim();

        if !label_trimmed.is_empty() {
            if node.is_internal() && !self.support_as_label {
                // Try to parse as bootstrap value
                if let Ok(bs) = label_trimmed.parse::<f64>() {
                    node.bootstrap = Some(bs);
                } else {
                    node.label = Some(label_trimmed.to_string());
                }
            } else {
                node.label = Some(label_trimmed.to_string());
            }
        }

        // Parse branch length
        if pos < chars.len() && chars[pos] == ':' {
            pos += 1;
            let bl_start = pos;
            while pos < chars.len() && !matches!(chars[pos], ',' | ')' | ';' | '(') {
                pos += 1;
            }
            let bl_str: String = chars[bl_start..pos].iter().collect();
            let bl: f64 = bl_str.trim().parse()
                .map_err(|_| NewickError::InvalidBranchLength(bl_str))?;
            node.branch_length = Some(bl);
        }

        // If leaf with no label, keep it as internal with no children
        if node.is_leaf() && node.label.is_none() && start == pos {
            // Empty node — just leave it
        }

        Ok((node, pos))
    }
}

impl fmt::Display for NewickParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NewickParser(support_as_label={})", self.support_as_label)
    }
}

/// Quick parse with default settings.
pub fn parse_newick(input: &str) -> Result<Vec<TreeNode>, NewickError> {
    NewickParser::new().parse(input)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t01_simple_tree() {
        let trees = parse_newick("(A,B);").unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].leaf_count(), 2);
    }

    #[test]
    fn t02_branch_lengths() {
        let trees = parse_newick("(A:0.1,B:0.2):0.3;").unwrap();
        let root = &trees[0];
        assert!((root.branch_length.unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn t03_nested_tree() {
        let trees = parse_newick("((A,B),C);").unwrap();
        assert_eq!(trees[0].leaf_count(), 3);
        assert_eq!(trees[0].depth(), 2);
    }

    #[test]
    fn t04_bootstrap_values() {
        let trees = parse_newick("((A,B)95,(C,D)80);").unwrap();
        let root = &trees[0];
        assert_eq!(root.children[0].bootstrap, Some(95.0));
        assert_eq!(root.children[1].bootstrap, Some(80.0));
    }

    #[test]
    fn t05_leaf_labels() {
        let trees = parse_newick("((A,B),(C,D));").unwrap();
        let mut labels = trees[0].leaf_labels();
        labels.sort();
        assert_eq!(labels, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn t06_node_count() {
        let trees = parse_newick("((A,B),C);").unwrap();
        assert_eq!(trees[0].node_count(), 5); // root + internal + 3 leaves
    }

    #[test]
    fn t07_total_branch_length() {
        let trees = parse_newick("(A:1.0,B:2.0);").unwrap();
        assert!((trees[0].total_branch_length() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn t08_find_node() {
        let trees = parse_newick("((A,B),C);").unwrap();
        assert!(trees[0].find("A").is_some());
        assert!(trees[0].find("Z").is_none());
    }

    #[test]
    fn t09_is_leaf() {
        let leaf = TreeNode::leaf("X");
        assert!(leaf.is_leaf());
        assert!(!leaf.is_internal());
    }

    #[test]
    fn t10_serialization_roundtrip() {
        let input = "(A:0.1,B:0.2);";
        let trees = parse_newick(input).unwrap();
        let serialized = format!("{};", trees[0].to_newick());
        let reparsed = parse_newick(&serialized).unwrap();
        assert_eq!(reparsed[0].leaf_count(), 2);
    }

    #[test]
    fn t11_empty_input() {
        assert_eq!(parse_newick(""), Err(NewickError::EmptyInput));
    }

    #[test]
    fn t12_unbalanced_parens() {
        assert!(matches!(
            parse_newick("((A,B);"),
            Err(NewickError::UnbalancedParentheses)
        ));
    }

    #[test]
    fn t13_builder_api() {
        let tree = TreeNode::internal()
            .with_child(TreeNode::leaf("A").with_branch_length(0.5))
            .with_child(TreeNode::leaf("B").with_branch_length(0.3));
        assert_eq!(tree.leaf_count(), 2);
        assert!((tree.total_branch_length() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn t14_single_leaf() {
        let trees = parse_newick("A;").unwrap();
        assert_eq!(trees[0].label.as_deref(), Some("A"));
        assert!(trees[0].is_leaf());
    }

    #[test]
    fn t15_complex_branch_lengths() {
        let trees = parse_newick("((A:0.01,B:0.02):0.03,C:0.04);").unwrap();
        let tbl = trees[0].total_branch_length();
        assert!((tbl - 0.10).abs() < 1e-9);
    }

    #[test]
    fn t16_display_leaf() {
        let leaf = TreeNode::leaf("Human").with_branch_length(0.5);
        let s = format!("{leaf}");
        assert!(s.contains("Human"));
        assert!(s.contains("0.5"));
    }

    #[test]
    fn t17_display_internal() {
        let node = TreeNode::internal()
            .with_child(TreeNode::leaf("A"))
            .with_child(TreeNode::leaf("B"));
        let s = format!("{node}");
        assert!(s.contains("2 children"));
    }

    #[test]
    fn t18_multiple_trees() {
        let trees = parse_newick("(A,B);(C,D);").unwrap();
        assert_eq!(trees.len(), 2);
    }

    #[test]
    fn t19_depth_single() {
        let trees = parse_newick("A;").unwrap();
        assert_eq!(trees[0].depth(), 0);
    }

    #[test]
    fn t20_display_parser() {
        let p = NewickParser::new().with_support_as_label(true);
        let s = format!("{p}");
        assert!(s.contains("support_as_label=true"));
    }
}
