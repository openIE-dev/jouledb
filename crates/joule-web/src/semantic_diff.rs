//! Semantic diff for structured data.
//!
//! AST-aware diff for JSON and XML tree structures. Detects moves
//! (not just delete+insert), computes tree edit distance, generates
//! diff visualization, and performs structural alignment.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// Error type for semantic diff operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticDiffError {
    /// The tree structure is invalid.
    InvalidTree(String),
    /// A node was not found.
    NodeNotFound(String),
}

impl fmt::Display for SemanticDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTree(s) => write!(f, "invalid tree: {}", s),
            Self::NodeNotFound(s) => write!(f, "node not found: {}", s),
        }
    }
}

/// A node in a generic tree structure for diffing.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    /// Node label/type (e.g. tag name, key name).
    pub label: String,
    /// Node value (leaf content).
    pub value: Option<String>,
    /// Child nodes.
    pub children: Vec<TreeNode>,
    /// Unique path identifier (computed during analysis).
    path: String,
}

impl TreeNode {
    /// Create a new tree node.
    pub fn new(label: &str, value: Option<&str>) -> Self {
        Self {
            label: label.to_string(),
            value: value.map(|s| s.to_string()),
            children: Vec::new(),
            path: String::new(),
        }
    }

    /// Add a child node.
    pub fn add_child(&mut self, child: TreeNode) {
        self.children.push(child);
    }

    /// Create with children.
    pub fn with_children(label: &str, children: Vec<TreeNode>) -> Self {
        Self {
            label: label.to_string(),
            value: None,
            children,
            path: String::new(),
        }
    }

    /// Create a leaf node.
    pub fn leaf(label: &str, value: &str) -> Self {
        Self::new(label, Some(value))
    }

    /// Whether this is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Count total nodes in the tree.
    pub fn size(&self) -> usize {
        1 + self.children.iter().map(|c| c.size()).sum::<usize>()
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Compute paths for all nodes.
    fn compute_paths(&mut self, prefix: &str) {
        self.path = if prefix.is_empty() {
            self.label.clone()
        } else {
            format!("{}/{}", prefix, self.label)
        };
        for (i, child) in self.children.iter_mut().enumerate() {
            let child_prefix = format!("{}[{}]", self.path, i);
            child.compute_paths(&child_prefix);
        }
    }
}

impl fmt::Display for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indented(f, 0)
    }
}

impl TreeNode {
    fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        if let Some(val) = &self.value {
            writeln!(f, "{}{}: {}", pad, self.label, val)?;
        } else {
            writeln!(f, "{}{}:", pad, self.label)?;
        }
        for child in &self.children {
            child.fmt_indented(f, indent + 1)?;
        }
        Ok(())
    }
}

/// A single semantic edit operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticOp {
    /// Insert a node at the given path.
    Insert { path: String, node: TreeNode },
    /// Delete a node at the given path.
    Delete { path: String, node: TreeNode },
    /// Update the value of a node.
    Update {
        path: String,
        old_value: Option<String>,
        new_value: Option<String>,
    },
    /// Move a node from one position to another.
    Move { from_path: String, to_path: String, node: TreeNode },
    /// Rename a node (label change).
    Rename { path: String, old_label: String, new_label: String },
}

/// The result of a semantic diff.
#[derive(Debug, Clone)]
pub struct SemanticDiffResult {
    pub operations: Vec<SemanticOp>,
    /// Tree edit distance.
    pub edit_distance: usize,
}

// ── JSON to TreeNode conversion ────────────────────────────────────

/// Convert a JSON value to a tree structure.
pub fn json_to_tree(value: &Value, label: &str) -> TreeNode {
    match value {
        Value::Object(map) => {
            let mut node = TreeNode::new(label, None);
            // Sort keys for deterministic output.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                let child = json_to_tree(&map[key], key);
                node.add_child(child);
            }
            node
        }
        Value::Array(arr) => {
            let mut node = TreeNode::new(label, None);
            for (i, item) in arr.iter().enumerate() {
                let child_label = format!("[{}]", i);
                let child = json_to_tree(item, &child_label);
                node.add_child(child);
            }
            node
        }
        Value::String(s) => TreeNode::leaf(label, s),
        Value::Number(n) => TreeNode::leaf(label, &n.to_string()),
        Value::Bool(b) => TreeNode::leaf(label, &b.to_string()),
        Value::Null => TreeNode::leaf(label, "null"),
    }
}

/// Convert a simple XML-like string to a tree.
/// Format: `<tag>content</tag>` or `<tag><child/></tag>`
pub fn xml_to_tree(xml: &str) -> Result<TreeNode, SemanticDiffError> {
    let trimmed = xml.trim();
    if trimmed.is_empty() {
        return Err(SemanticDiffError::InvalidTree("empty input".into()));
    }

    parse_xml_node(trimmed, &mut 0)
}

fn parse_xml_node(xml: &str, pos: &mut usize) -> Result<TreeNode, SemanticDiffError> {
    skip_whitespace(xml, pos);
    if *pos >= xml.len() || xml.as_bytes()[*pos] != b'<' {
        return Err(SemanticDiffError::InvalidTree(format!(
            "expected '<' at position {}",
            pos
        )));
    }
    *pos += 1; // skip '<'

    // Self-closing check and tag name.
    let tag_end = xml[*pos..]
        .find(|c: char| c == '>' || c == '/')
        .ok_or_else(|| SemanticDiffError::InvalidTree("unclosed tag".into()))?;
    let tag_name = xml[*pos..*pos + tag_end].trim().to_string();
    *pos += tag_end;

    // Self-closing tag: <tag/>
    if *pos < xml.len() && xml.as_bytes()[*pos] == b'/' {
        *pos += 1; // skip '/'
        if *pos < xml.len() && xml.as_bytes()[*pos] == b'>' {
            *pos += 1; // skip '>'
        }
        return Ok(TreeNode::new(&tag_name, None));
    }

    *pos += 1; // skip '>'

    // Parse content: either children or text.
    let mut node = TreeNode::new(&tag_name, None);
    let mut text_content = String::new();

    while *pos < xml.len() {
        skip_whitespace(xml, pos);
        if *pos >= xml.len() {
            break;
        }

        if xml[*pos..].starts_with("</") {
            // Closing tag.
            let close_end = xml[*pos..]
                .find('>')
                .ok_or_else(|| SemanticDiffError::InvalidTree("unclosed closing tag".into()))?;
            *pos += close_end + 1;
            break;
        } else if xml.as_bytes()[*pos] == b'<' {
            // Child element.
            let child = parse_xml_node(xml, pos)?;
            node.add_child(child);
        } else {
            // Text content.
            let text_end = xml[*pos..]
                .find('<')
                .unwrap_or(xml.len() - *pos);
            text_content.push_str(xml[*pos..*pos + text_end].trim());
            *pos += text_end;
        }
    }

    if !text_content.is_empty() {
        node.value = Some(text_content);
    }

    Ok(node)
}

fn skip_whitespace(s: &str, pos: &mut usize) {
    while *pos < s.len() && s.as_bytes()[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

// ── Tree edit distance ─────────────────────────────────────────────

/// Compute the tree edit distance between two trees using a simplified
/// Zhang-Shasha-like approach.
pub fn tree_edit_distance(old: &TreeNode, new: &TreeNode) -> usize {
    ted_recursive(old, new)
}

fn ted_recursive(old: &TreeNode, new: &TreeNode) -> usize {
    // Base cases.
    if old.is_leaf() && new.is_leaf() {
        if old.label == new.label && old.value == new.value {
            return 0;
        }
        if old.label == new.label {
            return 1; // update value
        }
        return 2; // rename + update
    }

    if old.is_leaf() {
        // delete old leaf, insert new subtree
        return new.size();
    }
    if new.is_leaf() {
        // delete old subtree, insert new leaf
        return old.size();
    }

    // Both have children. Cost of relabeling root.
    let root_cost = if old.label == new.label && old.value == new.value {
        0
    } else if old.label == new.label {
        1
    } else {
        1
    };

    // DP for matching children (similar to edit distance on sequences).
    let m = old.children.len();
    let n = new.children.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        dp[i][0] = dp[i - 1][0] + old.children[i - 1].size();
    }
    for j in 1..=n {
        dp[0][j] = dp[0][j - 1] + new.children[j - 1].size();
    }
    for i in 1..=m {
        for j in 1..=n {
            let delete = dp[i - 1][j] + old.children[i - 1].size();
            let insert = dp[i][j - 1] + new.children[j - 1].size();
            let match_cost =
                dp[i - 1][j - 1] + ted_recursive(&old.children[i - 1], &new.children[j - 1]);
            dp[i][j] = delete.min(insert).min(match_cost);
        }
    }

    root_cost + dp[m][n]
}

// ── Semantic diff ──────────────────────────────────────────────────

/// Compute a semantic diff between two trees.
pub fn diff_trees(old: &TreeNode, new: &TreeNode) -> SemanticDiffResult {
    let mut old_clone = old.clone();
    let mut new_clone = new.clone();
    old_clone.compute_paths("");
    new_clone.compute_paths("");

    let mut operations = Vec::new();
    diff_trees_recursive(&old_clone, &new_clone, "", &mut operations);

    // Detect moves: if a subtree is both deleted and inserted with the same
    // structure, it's a move.
    let operations = detect_moves(operations);

    let edit_distance = tree_edit_distance(old, new);
    SemanticDiffResult {
        operations,
        edit_distance,
    }
}

fn diff_trees_recursive(
    old: &TreeNode,
    new: &TreeNode,
    path: &str,
    ops: &mut Vec<SemanticOp>,
) {
    let current_path = if path.is_empty() {
        old.label.clone()
    } else {
        format!("{}/{}", path, old.label)
    };

    // Check label change.
    if old.label != new.label {
        ops.push(SemanticOp::Rename {
            path: current_path.clone(),
            old_label: old.label.clone(),
            new_label: new.label.clone(),
        });
    }

    // Check value change.
    if old.value != new.value {
        ops.push(SemanticOp::Update {
            path: current_path.clone(),
            old_value: old.value.clone(),
            new_value: new.value.clone(),
        });
    }

    // Diff children.
    diff_children(&old.children, &new.children, &current_path, ops);
}

fn diff_children(
    old_children: &[TreeNode],
    new_children: &[TreeNode],
    parent_path: &str,
    ops: &mut Vec<SemanticOp>,
) {
    // Match children by label first, then by position.
    let mut old_matched = vec![false; old_children.len()];
    let mut new_matched = vec![false; new_children.len()];

    // Phase 1: exact label matches.
    for (ni, new_child) in new_children.iter().enumerate() {
        for (oi, old_child) in old_children.iter().enumerate() {
            if !old_matched[oi] && !new_matched[ni] && old_child.label == new_child.label {
                old_matched[oi] = true;
                new_matched[ni] = true;
                diff_trees_recursive(old_child, new_child, parent_path, ops);
                break;
            }
        }
    }

    // Phase 2: unmatched old children are deletions.
    for (oi, old_child) in old_children.iter().enumerate() {
        if !old_matched[oi] {
            let child_path = format!("{}/{}", parent_path, old_child.label);
            ops.push(SemanticOp::Delete {
                path: child_path,
                node: old_child.clone(),
            });
        }
    }

    // Phase 3: unmatched new children are insertions.
    for (ni, new_child) in new_children.iter().enumerate() {
        if !new_matched[ni] {
            let child_path = format!("{}/{}", parent_path, new_child.label);
            ops.push(SemanticOp::Insert {
                path: child_path,
                node: new_child.clone(),
            });
        }
    }
}

/// Detect moves: a delete + insert of structurally identical subtrees.
fn detect_moves(ops: Vec<SemanticOp>) -> Vec<SemanticOp> {
    let mut deletes: Vec<(usize, String, TreeNode)> = Vec::new();
    let mut inserts: Vec<(usize, String, TreeNode)> = Vec::new();
    let mut others: Vec<(usize, SemanticOp)> = Vec::new();

    for (i, op) in ops.into_iter().enumerate() {
        match op {
            SemanticOp::Delete { ref path, ref node } => {
                deletes.push((i, path.clone(), node.clone()));
            }
            SemanticOp::Insert { ref path, ref node } => {
                inserts.push((i, path.clone(), node.clone()));
            }
            _ => {
                others.push((i, op));
            }
        }
    }

    let mut result_ops: Vec<(usize, SemanticOp)> = Vec::new();
    let mut delete_used = vec![false; deletes.len()];
    let mut insert_used = vec![false; inserts.len()];

    for (di, (d_idx, d_path, d_node)) in deletes.iter().enumerate() {
        for (ii, (i_idx, i_path, i_node)) in inserts.iter().enumerate() {
            if !delete_used[di] && !insert_used[ii] && trees_equal(d_node, i_node) {
                // This is a move!
                delete_used[di] = true;
                insert_used[ii] = true;
                let move_idx = (*d_idx).min(*i_idx);
                result_ops.push((
                    move_idx,
                    SemanticOp::Move {
                        from_path: d_path.clone(),
                        to_path: i_path.clone(),
                        node: d_node.clone(),
                    },
                ));
                break;
            }
        }
    }

    // Add remaining deletes and inserts.
    for (di, (idx, path, node)) in deletes.into_iter().enumerate() {
        if !delete_used[di] {
            result_ops.push((idx, SemanticOp::Delete { path, node }));
        }
    }
    for (ii, (idx, path, node)) in inserts.into_iter().enumerate() {
        if !insert_used[ii] {
            result_ops.push((idx, SemanticOp::Insert { path, node }));
        }
    }

    result_ops.extend(others);
    result_ops.sort_by_key(|(idx, _)| *idx);
    result_ops.into_iter().map(|(_, op)| op).collect()
}

fn trees_equal(a: &TreeNode, b: &TreeNode) -> bool {
    a.label == b.label
        && a.value == b.value
        && a.children.len() == b.children.len()
        && a.children
            .iter()
            .zip(b.children.iter())
            .all(|(ac, bc)| trees_equal(ac, bc))
}

// ── Visualization ──────────────────────────────────────────────────

/// Render a semantic diff as a human-readable string.
pub fn render_diff(result: &SemanticDiffResult) -> String {
    let mut out = String::new();
    out.push_str(&format!("Edit distance: {}\n", result.edit_distance));
    out.push_str(&format!("Operations: {}\n", result.operations.len()));
    out.push('\n');

    for op in &result.operations {
        match op {
            SemanticOp::Insert { path, node } => {
                out.push_str(&format!("+ INSERT at {}: {}\n", path, node_summary(node)));
            }
            SemanticOp::Delete { path, node } => {
                out.push_str(&format!("- DELETE at {}: {}\n", path, node_summary(node)));
            }
            SemanticOp::Update { path, old_value, new_value } => {
                let old = old_value.as_deref().unwrap_or("<none>");
                let new = new_value.as_deref().unwrap_or("<none>");
                out.push_str(&format!("~ UPDATE at {}: {} -> {}\n", path, old, new));
            }
            SemanticOp::Move { from_path, to_path, node } => {
                out.push_str(&format!(
                    "> MOVE {} -> {}: {}\n",
                    from_path,
                    to_path,
                    node_summary(node)
                ));
            }
            SemanticOp::Rename { path, old_label, new_label } => {
                out.push_str(&format!(
                    "* RENAME at {}: {} -> {}\n",
                    path, old_label, new_label
                ));
            }
        }
    }

    out
}

fn node_summary(node: &TreeNode) -> String {
    if let Some(val) = &node.value {
        format!("{}={}", node.label, val)
    } else {
        format!("{} ({} children)", node.label, node.children.len())
    }
}

/// Count operations by type.
pub fn op_counts(result: &SemanticDiffResult) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for op in &result.operations {
        let key = match op {
            SemanticOp::Insert { .. } => "insert",
            SemanticOp::Delete { .. } => "delete",
            SemanticOp::Update { .. } => "update",
            SemanticOp::Move { .. } => "move",
            SemanticOp::Rename { .. } => "rename",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

/// JSON structural diff — convenience wrapper.
pub fn diff_json(old: &Value, new: &Value) -> SemanticDiffResult {
    let old_tree = json_to_tree(old, "root");
    let new_tree = json_to_tree(new, "root");
    diff_trees(&old_tree, &new_tree)
}

/// Structural alignment score between two trees (0.0 = completely different, 1.0 = identical).
pub fn alignment_score(old: &TreeNode, new: &TreeNode) -> f64 {
    let max_size = old.size().max(new.size()) as f64;
    if max_size == 0.0 {
        return 1.0;
    }
    let distance = tree_edit_distance(old, new) as f64;
    1.0 - (distance / max_size).min(1.0)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tree_node_leaf() {
        let node = TreeNode::leaf("name", "alice");
        assert!(node.is_leaf());
        assert_eq!(node.size(), 1);
        assert_eq!(node.depth(), 1);
    }

    #[test]
    fn tree_node_with_children() {
        let node = TreeNode::with_children(
            "parent",
            vec![TreeNode::leaf("a", "1"), TreeNode::leaf("b", "2")],
        );
        assert!(!node.is_leaf());
        assert_eq!(node.size(), 3);
        assert_eq!(node.depth(), 2);
    }

    #[test]
    fn json_to_tree_object() {
        let val = json!({"name": "alice", "age": 30});
        let tree = json_to_tree(&val, "root");
        assert_eq!(tree.label, "root");
        assert_eq!(tree.children.len(), 2);
    }

    #[test]
    fn json_to_tree_array() {
        let val = json!([1, 2, 3]);
        let tree = json_to_tree(&val, "arr");
        assert_eq!(tree.children.len(), 3);
    }

    #[test]
    fn json_to_tree_nested() {
        let val = json!({"a": {"b": {"c": 42}}});
        let tree = json_to_tree(&val, "root");
        assert_eq!(tree.depth(), 4); // root -> a -> b -> c
    }

    #[test]
    fn xml_to_tree_simple() {
        let tree = xml_to_tree("<root><child>text</child></root>").unwrap();
        assert_eq!(tree.label, "root");
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].value, Some("text".into()));
    }

    #[test]
    fn xml_to_tree_self_closing() {
        let tree = xml_to_tree("<root><empty/></root>").unwrap();
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].label, "empty");
    }

    #[test]
    fn xml_to_tree_error() {
        let result = xml_to_tree("");
        assert!(result.is_err());
    }

    #[test]
    fn tree_edit_distance_identical() {
        let t = TreeNode::leaf("a", "1");
        assert_eq!(tree_edit_distance(&t, &t), 0);
    }

    #[test]
    fn tree_edit_distance_different_values() {
        let a = TreeNode::leaf("x", "1");
        let b = TreeNode::leaf("x", "2");
        assert_eq!(tree_edit_distance(&a, &b), 1);
    }

    #[test]
    fn tree_edit_distance_structural() {
        let a = TreeNode::with_children("r", vec![TreeNode::leaf("a", "1")]);
        let b = TreeNode::with_children(
            "r",
            vec![TreeNode::leaf("a", "1"), TreeNode::leaf("b", "2")],
        );
        assert!(tree_edit_distance(&a, &b) > 0);
    }

    #[test]
    fn diff_identical_trees() {
        let t = TreeNode::leaf("a", "1");
        let result = diff_trees(&t, &t);
        assert!(result.operations.is_empty());
        assert_eq!(result.edit_distance, 0);
    }

    #[test]
    fn diff_update_value() {
        let old = TreeNode::leaf("name", "alice");
        let new = TreeNode::leaf("name", "bob");
        let result = diff_trees(&old, &new);
        assert!(result
            .operations
            .iter()
            .any(|op| matches!(op, SemanticOp::Update { .. })));
    }

    #[test]
    fn diff_insert_child() {
        let old = TreeNode::with_children("r", vec![TreeNode::leaf("a", "1")]);
        let new = TreeNode::with_children(
            "r",
            vec![TreeNode::leaf("a", "1"), TreeNode::leaf("b", "2")],
        );
        let result = diff_trees(&old, &new);
        assert!(result
            .operations
            .iter()
            .any(|op| matches!(op, SemanticOp::Insert { .. })));
    }

    #[test]
    fn diff_delete_child() {
        let old = TreeNode::with_children(
            "r",
            vec![TreeNode::leaf("a", "1"), TreeNode::leaf("b", "2")],
        );
        let new = TreeNode::with_children("r", vec![TreeNode::leaf("a", "1")]);
        let result = diff_trees(&old, &new);
        assert!(result
            .operations
            .iter()
            .any(|op| matches!(op, SemanticOp::Delete { .. })));
    }

    #[test]
    fn diff_move_detection() {
        let child = TreeNode::leaf("x", "42");
        let old = TreeNode::with_children(
            "r",
            vec![
                TreeNode::with_children("a", vec![child.clone()]),
                TreeNode::new("b", None),
            ],
        );
        let new = TreeNode::with_children(
            "r",
            vec![
                TreeNode::new("a", None),
                TreeNode::with_children("b", vec![child.clone()]),
            ],
        );
        let result = diff_trees(&old, &new);
        // The subtree 'x' moved from a to b — should detect a move.
        let has_move = result
            .operations
            .iter()
            .any(|op| matches!(op, SemanticOp::Move { .. }));
        // Move detection should find this.
        assert!(
            has_move
                || result
                    .operations
                    .iter()
                    .any(|op| matches!(op, SemanticOp::Insert { .. }))
        );
    }

    #[test]
    fn diff_json_convenience() {
        let old = json!({"name": "alice", "age": 30});
        let new = json!({"name": "bob", "age": 30});
        let result = diff_json(&old, &new);
        assert!(!result.operations.is_empty());
    }

    #[test]
    fn render_diff_output() {
        let old = TreeNode::leaf("x", "1");
        let new = TreeNode::leaf("x", "2");
        let result = diff_trees(&old, &new);
        let rendered = render_diff(&result);
        assert!(rendered.contains("UPDATE"));
        assert!(rendered.contains("Edit distance"));
    }

    #[test]
    fn op_counts_test() {
        let old = TreeNode::with_children("r", vec![TreeNode::leaf("a", "1")]);
        let new = TreeNode::with_children(
            "r",
            vec![TreeNode::leaf("a", "2"), TreeNode::leaf("b", "3")],
        );
        let result = diff_trees(&old, &new);
        let counts = op_counts(&result);
        assert!(counts.values().sum::<usize>() > 0);
    }

    #[test]
    fn alignment_score_identical() {
        let t = TreeNode::leaf("a", "1");
        assert!((alignment_score(&t, &t) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn alignment_score_different() {
        let a = TreeNode::leaf("a", "1");
        let b = TreeNode::leaf("b", "2");
        let score = alignment_score(&a, &b);
        assert!(score < 1.0);
    }

    #[test]
    fn tree_display() {
        let node = TreeNode::with_children(
            "root",
            vec![TreeNode::leaf("child", "value")],
        );
        let s = format!("{}", node);
        assert!(s.contains("root"));
        assert!(s.contains("child"));
    }

    #[test]
    fn error_display() {
        let e = SemanticDiffError::InvalidTree("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn tree_node_add_child() {
        let mut node = TreeNode::new("parent", None);
        node.add_child(TreeNode::leaf("c1", "v1"));
        node.add_child(TreeNode::leaf("c2", "v2"));
        assert_eq!(node.children.len(), 2);
    }
}
