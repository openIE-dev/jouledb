//! Mind map with radial layout, collapse/expand, and outline conversion.
//!
//! Replaces MindMeister / jsMind. Provides radial layout, auto-balance,
//! collapse/expand, depth-limited view, and outline text conversion.
//! Pure Rust — no browser dependency.

use std::collections::HashMap;
use std::f64::consts::PI;

// ── Data types ───────────────────────────────────────────────────

/// A node in the mind map.
#[derive(Debug, Clone)]
pub struct MindmapNode {
    pub id: String,
    pub text: String,
    pub children: Vec<MindmapNode>,
    pub collapsed: bool,
}

impl MindmapNode {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            children: Vec::new(),
            collapsed: false,
        }
    }

    pub fn add_child(&mut self, child: MindmapNode) {
        self.children.push(child);
    }

    pub fn remove_child(&mut self, id: &str) -> Option<MindmapNode> {
        if let Some(pos) = self.children.iter().position(|c| c.id == id) {
            Some(self.children.remove(pos))
        } else {
            None
        }
    }

    pub fn find(&self, id: &str) -> Option<&MindmapNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut MindmapNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_mut(id) {
                return Some(found);
            }
        }
        None
    }

    /// Find the parent of a given node id.
    pub fn find_parent(&self, id: &str) -> Option<&MindmapNode> {
        for child in &self.children {
            if child.id == id {
                return Some(self);
            }
            if let Some(parent) = child.find_parent(id) {
                return Some(parent);
            }
        }
        None
    }

    /// Total node count in this subtree.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Maximum depth of this subtree.
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self.children.iter().map(|c| c.max_depth()).max().unwrap_or(0)
        }
    }

    /// Collapse this node (hide children).
    pub fn collapse(&mut self) {
        self.collapsed = true;
    }

    /// Expand this node (show children).
    pub fn expand(&mut self) {
        self.collapsed = false;
    }

    /// Recursively collapse all descendants.
    pub fn collapse_all(&mut self) {
        self.collapsed = true;
        for child in &mut self.children {
            child.collapse_all();
        }
    }

    /// Recursively expand all descendants.
    pub fn expand_all(&mut self) {
        self.collapsed = false;
        for child in &mut self.children {
            child.expand_all();
        }
    }

    /// Reorder siblings: move child at `from` index to `to` index.
    pub fn reorder_child(&mut self, from: usize, to: usize) -> bool {
        if from >= self.children.len() || to >= self.children.len() {
            return false;
        }
        let child = self.children.remove(from);
        self.children.insert(to, child);
        true
    }

    /// Visible node count (respecting collapsed state).
    pub fn visible_count(&self) -> usize {
        if self.collapsed {
            1
        } else {
            1 + self.children.iter().map(|c| c.visible_count()).sum::<usize>()
        }
    }

    /// Get nodes up to a certain depth (0 = root only).
    pub fn depth_limited(&self, max_depth: usize) -> MindmapNode {
        let mut clone = MindmapNode::new(self.id.clone(), self.text.clone());
        clone.collapsed = self.collapsed;
        if max_depth > 0 && !self.collapsed {
            for child in &self.children {
                clone.children.push(child.depth_limited(max_depth - 1));
            }
        }
        clone
    }
}

// ── Mindmap ──────────────────────────────────────────────────────

/// Position computed by layout.
#[derive(Debug, Clone, Copy)]
pub struct MindmapPosition {
    pub x: f64,
    pub y: f64,
}

/// The mind map.
#[derive(Debug, Clone)]
pub struct Mindmap {
    pub root: MindmapNode,
    positions: HashMap<String, MindmapPosition>,
}

impl Mindmap {
    pub fn new(root: MindmapNode) -> Self {
        Self {
            root,
            positions: HashMap::new(),
        }
    }

    pub fn find(&self, id: &str) -> Option<&MindmapNode> {
        self.root.find(id)
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut MindmapNode> {
        self.root.find_mut(id)
    }

    /// Add a child to a node by id.
    pub fn add_child(&mut self, parent_id: &str, child: MindmapNode) -> bool {
        if let Some(parent) = self.root.find_mut(parent_id) {
            parent.add_child(child);
            true
        } else {
            false
        }
    }

    /// Remove a node by id (cannot remove root).
    pub fn remove_node(&mut self, id: &str) -> bool {
        if self.root.id == id {
            return false;
        }
        self.remove_recursive(&mut self.root.children.clone(), id)
    }

    fn remove_recursive(&mut self, _children: &[MindmapNode], id: &str) -> bool {
        // We need to find parent and remove from it.
        fn remove_from(node: &mut MindmapNode, target: &str) -> bool {
            if let Some(pos) = node.children.iter().position(|c| c.id == target) {
                node.children.remove(pos);
                return true;
            }
            for child in &mut node.children {
                if remove_from(child, target) {
                    return true;
                }
            }
            false
        }
        remove_from(&mut self.root, id)
    }

    /// Move a node to a new parent.
    pub fn move_node(&mut self, node_id: &str, new_parent_id: &str) -> bool {
        if node_id == self.root.id {
            return false;
        }
        // Check new parent is not a descendant of node.
        if let Some(node) = self.root.find(node_id) {
            if node.find(new_parent_id).is_some() {
                return false;
            }
        }

        // Remove from current location.
        fn extract(node: &mut MindmapNode, target: &str) -> Option<MindmapNode> {
            if let Some(pos) = node.children.iter().position(|c| c.id == target) {
                return Some(node.children.remove(pos));
            }
            for child in &mut node.children {
                if let Some(found) = extract(child, target) {
                    return Some(found);
                }
            }
            None
        }

        if let Some(extracted) = extract(&mut self.root, node_id) {
            if let Some(parent) = self.root.find_mut(new_parent_id) {
                parent.add_child(extracted);
                return true;
            }
        }
        false
    }

    /// Radial layout: children distributed in sectors around parent.
    pub fn radial_layout(&mut self, center_x: f64, center_y: f64, radius_step: f64) {
        self.positions.clear();
        self.positions.insert(
            self.root.id.clone(),
            MindmapPosition { x: center_x, y: center_y },
        );
        self.layout_children(&self.root.clone(), center_x, center_y, 0.0, 2.0 * PI, radius_step, 1);
    }

    fn layout_children(
        &mut self,
        node: &MindmapNode,
        parent_x: f64,
        parent_y: f64,
        start_angle: f64,
        sweep: f64,
        radius_step: f64,
        depth: usize,
    ) {
        if node.collapsed || node.children.is_empty() {
            return;
        }

        let n = node.children.len();
        let radius = radius_step * depth as f64;
        let angle_step = sweep / n as f64;

        for (i, child) in node.children.iter().enumerate() {
            let angle = start_angle + angle_step * (i as f64 + 0.5);
            let x = parent_x + radius * angle.cos();
            let y = parent_y + radius * angle.sin();
            self.positions.insert(
                child.id.clone(),
                MindmapPosition { x, y },
            );

            // Subdivide the sector for grandchildren.
            let child_start = start_angle + angle_step * i as f64;
            self.layout_children(child, x, y, child_start, angle_step, radius_step, depth + 1);
        }
    }

    pub fn position(&self, id: &str) -> Option<&MindmapPosition> {
        self.positions.get(id)
    }

    /// Auto-balance: distribute children of root evenly between left and right.
    pub fn auto_balance(&mut self) {
        let n = self.root.children.len();
        if n <= 1 {
            return;
        }
        // Already balanced if children are evenly split. Just sort by weight.
        self.root.children.sort_by(|a, b| {
            b.node_count().cmp(&a.node_count())
        });

        // Interleave large and small subtrees for visual balance.
        let mut balanced = Vec::with_capacity(n);
        let mut left = 0;
        let mut right = n - 1;
        let mut take_left = true;
        while left <= right {
            if take_left {
                balanced.push(self.root.children[left].clone());
                left += 1;
            } else {
                balanced.push(self.root.children[right].clone());
                if right == 0 {
                    break;
                }
                right -= 1;
            }
            take_left = !take_left;
        }
        self.root.children = balanced;
    }

    /// Convert to indented outline text.
    pub fn to_outline(&self) -> String {
        let mut output = String::new();
        self.outline_recursive(&self.root, 0, &mut output);
        output
    }

    fn outline_recursive(&self, node: &MindmapNode, depth: usize, output: &mut String) {
        for _ in 0..depth {
            output.push_str("  ");
        }
        output.push_str(&node.text);
        output.push('\n');
        if !node.collapsed {
            for child in &node.children {
                self.outline_recursive(child, depth + 1, output);
            }
        }
    }

    /// Parse from indented outline text.
    pub fn from_outline(text: &str) -> Option<Self> {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return None;
        }

        let mut id_counter = 0_usize;
        let mut next_id = || {
            let id = format!("n{id_counter}");
            id_counter += 1;
            id
        };

        // Parse indentation levels.
        let mut entries: Vec<(usize, String)> = Vec::new();
        for line in &lines {
            if line.trim().is_empty() {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            let level = indent / 2;
            entries.push((level, line.trim().to_string()));
        }

        if entries.is_empty() {
            return None;
        }

        let root_id = next_id();
        let mut root = MindmapNode::new(root_id, entries[0].1.clone());

        // Stack of (depth, node_ref path).
        fn build(
            parent: &mut MindmapNode,
            entries: &[(usize, String)],
            start: usize,
            parent_level: usize,
            id_counter: &mut usize,
        ) {
            let mut i = start;
            while i < entries.len() {
                let (level, text) = &entries[i];
                if *level <= parent_level {
                    break;
                }
                if *level == parent_level + 1 {
                    let id = format!("n{id_counter}");
                    *id_counter += 1;
                    let mut child = MindmapNode::new(id, text.clone());
                    build(&mut child, entries, i + 1, *level, id_counter);
                    parent.add_child(child);
                }
                i += 1;
            }
        }

        build(&mut root, &entries, 1, 0, &mut id_counter);
        Some(Mindmap::new(root))
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mindmap() -> Mindmap {
        let mut root = MindmapNode::new("root", "Central Topic");
        let mut branch1 = MindmapNode::new("b1", "Branch 1");
        branch1.add_child(MindmapNode::new("b1a", "Leaf 1a"));
        branch1.add_child(MindmapNode::new("b1b", "Leaf 1b"));
        let mut branch2 = MindmapNode::new("b2", "Branch 2");
        branch2.add_child(MindmapNode::new("b2a", "Leaf 2a"));
        root.add_child(branch1);
        root.add_child(branch2);
        root.add_child(MindmapNode::new("b3", "Branch 3"));
        Mindmap::new(root)
    }

    #[test]
    fn test_node_count() {
        let mm = sample_mindmap();
        assert_eq!(mm.root.node_count(), 7);
    }

    #[test]
    fn test_max_depth() {
        let mm = sample_mindmap();
        assert_eq!(mm.root.max_depth(), 2);
    }

    #[test]
    fn test_find() {
        let mm = sample_mindmap();
        assert!(mm.find("b1a").is_some());
        assert!(mm.find("nonexistent").is_none());
    }

    #[test]
    fn test_add_child() {
        let mut mm = sample_mindmap();
        assert!(mm.add_child("b3", MindmapNode::new("b3a", "Leaf 3a")));
        assert_eq!(mm.root.node_count(), 8);
    }

    #[test]
    fn test_remove_node() {
        let mut mm = sample_mindmap();
        assert!(mm.remove_node("b1a"));
        assert_eq!(mm.root.node_count(), 6);
        // Cannot remove root.
        assert!(!mm.remove_node("root"));
    }

    #[test]
    fn test_move_node() {
        let mut mm = sample_mindmap();
        assert!(mm.move_node("b1a", "b2"));
        let b2 = mm.find("b2").unwrap();
        assert_eq!(b2.children.len(), 2);
    }

    #[test]
    fn test_collapse_expand() {
        let mut mm = sample_mindmap();
        mm.root.collapse_all();
        assert_eq!(mm.root.visible_count(), 1);
        mm.root.expand_all();
        assert_eq!(mm.root.visible_count(), 7);
    }

    #[test]
    fn test_collapse_subtree() {
        let mut mm = sample_mindmap();
        mm.find_mut("b1").unwrap().collapse();
        // b1 visible but its children hidden.
        assert_eq!(mm.root.visible_count(), 5);
    }

    #[test]
    fn test_depth_limited() {
        let mm = sample_mindmap();
        let limited = mm.root.depth_limited(1);
        assert_eq!(limited.children.len(), 3);
        // Grandchildren should be empty.
        assert!(limited.children[0].children.is_empty());
    }

    #[test]
    fn test_reorder_child() {
        let mut mm = sample_mindmap();
        let first = mm.root.children[0].id.clone();
        mm.root.reorder_child(0, 2);
        assert_eq!(mm.root.children[2].id, first);
    }

    #[test]
    fn test_radial_layout() {
        let mut mm = sample_mindmap();
        mm.radial_layout(400.0, 300.0, 100.0);
        let root_pos = mm.position("root").unwrap();
        assert_eq!(root_pos.x, 400.0);
        assert_eq!(root_pos.y, 300.0);
        // Children should be positioned.
        assert!(mm.position("b1").is_some());
        assert!(mm.position("b2").is_some());
    }

    #[test]
    fn test_to_outline() {
        let mm = sample_mindmap();
        let outline = mm.to_outline();
        assert!(outline.starts_with("Central Topic\n"));
        assert!(outline.contains("  Branch 1\n"));
        assert!(outline.contains("    Leaf 1a\n"));
    }

    #[test]
    fn test_from_outline() {
        let text = "Central Topic\n  Branch A\n    Leaf A1\n  Branch B\n";
        let mm = Mindmap::from_outline(text).unwrap();
        assert_eq!(mm.root.text, "Central Topic");
        assert_eq!(mm.root.children.len(), 2);
        assert_eq!(mm.root.children[0].children.len(), 1);
    }

    #[test]
    fn test_roundtrip_outline() {
        let mm = sample_mindmap();
        let outline = mm.to_outline();
        let mm2 = Mindmap::from_outline(&outline).unwrap();
        assert_eq!(mm2.root.node_count(), mm.root.node_count());
    }

    #[test]
    fn test_auto_balance() {
        let mut mm = sample_mindmap();
        mm.auto_balance();
        // After balancing, root should still have 3 children.
        assert_eq!(mm.root.children.len(), 3);
    }
}
