//! Tree data structure for hierarchical UI components.
//!
//! Replaces rc-tree / react-arborist. Supports expand/collapse,
//! single/multi select, drag-move, and flattening for virtualised
//! rendering. Pure Rust — no browser dependency.

// ── TreeNode ───────────────────────────────────────────────────

/// A single node in the tree.
#[derive(Debug, Clone)]
pub struct TreeNode<T> {
    pub id: String,
    pub data: T,
    pub children: Vec<TreeNode<T>>,
    pub expanded: bool,
    pub selected: bool,
    pub disabled: bool,
}

impl<T> TreeNode<T> {
    pub fn new(id: impl Into<String>, data: T) -> Self {
        Self {
            id: id.into(),
            data,
            children: Vec::new(),
            expanded: false,
            selected: false,
            disabled: false,
        }
    }

    pub fn add_child(&mut self, node: TreeNode<T>) {
        self.children.push(node);
    }

    pub fn remove_child(&mut self, id: &str) -> Option<TreeNode<T>> {
        if let Some(pos) = self.children.iter().position(|c| c.id == id) {
            Some(self.children.remove(pos))
        } else {
            None
        }
    }

    pub fn find(&self, id: &str) -> Option<&TreeNode<T>> {
        if self.id == id { return Some(self); }
        for child in &self.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut TreeNode<T>> {
        if self.id == id { return Some(self); }
        for child in &mut self.children {
            if let Some(found) = child.find_mut(id) {
                return Some(found);
            }
        }
        None
    }

    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    pub fn is_leaf(&self) -> bool { self.children.is_empty() }

    pub fn child_count(&self) -> usize { self.children.len() }
}

// ── FlatNode ───────────────────────────────────────────────────

/// Flattened representation for rendering (e.g., in a virtualised list).
#[derive(Debug, Clone, PartialEq)]
pub struct FlatNode<'a> {
    pub id: &'a str,
    pub depth: usize,
    pub is_leaf: bool,
    pub expanded: bool,
    pub selected: bool,
    pub has_children: bool,
}

// ── Tree ───────────────────────────────────────────────────────

/// Top-level tree container with selection semantics.
#[derive(Debug, Clone)]
pub struct Tree<T> {
    pub root: Vec<TreeNode<T>>,
    pub multi_select: bool,
}

impl<T> Default for Tree<T> {
    fn default() -> Self { Self::new() }
}

impl<T> Tree<T> {
    pub fn new() -> Self {
        Self { root: Vec::new(), multi_select: false }
    }

    pub fn add_root(&mut self, node: TreeNode<T>) {
        self.root.push(node);
    }

    pub fn find(&self, id: &str) -> Option<&TreeNode<T>> {
        for r in &self.root {
            if let Some(n) = r.find(id) { return Some(n); }
        }
        None
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut TreeNode<T>> {
        for r in &mut self.root {
            if let Some(n) = r.find_mut(id) { return Some(n); }
        }
        None
    }

    pub fn toggle_expand(&mut self, id: &str) {
        if let Some(n) = self.find_mut(id) {
            n.expanded = !n.expanded;
        }
    }

    pub fn expand_all(&mut self) {
        fn expand_recursive<T>(node: &mut TreeNode<T>) {
            node.expanded = true;
            for c in &mut node.children { expand_recursive(c); }
        }
        for r in &mut self.root { expand_recursive(r); }
    }

    pub fn collapse_all(&mut self) {
        fn collapse_recursive<T>(node: &mut TreeNode<T>) {
            node.expanded = false;
            for c in &mut node.children { collapse_recursive(c); }
        }
        for r in &mut self.root { collapse_recursive(r); }
    }

    pub fn select(&mut self, id: &str) {
        if !self.multi_select {
            self.deselect_all();
        }
        if let Some(n) = self.find_mut(id) {
            n.selected = true;
        }
    }

    pub fn deselect(&mut self, id: &str) {
        if let Some(n) = self.find_mut(id) {
            n.selected = false;
        }
    }

    pub fn toggle_select(&mut self, id: &str) {
        if let Some(selected) = self.find(id).map(|n| n.selected) {
            if selected {
                self.deselect(id);
            } else {
                self.select(id);
            }
        }
    }

    pub fn selected_ids(&self) -> Vec<&str> {
        let mut ids = Vec::new();
        fn collect<'a, T>(node: &'a TreeNode<T>, ids: &mut Vec<&'a str>) {
            if node.selected { ids.push(&node.id); }
            for c in &node.children { collect(c, ids); }
        }
        for r in &self.root { collect(r, &mut ids); }
        ids
    }

    /// Flatten the visible tree (respecting expanded state) for rendering.
    pub fn flatten(&self) -> Vec<FlatNode<'_>> {
        let mut flat = Vec::new();
        fn recurse<'a, T>(node: &'a TreeNode<T>, depth: usize, flat: &mut Vec<FlatNode<'a>>) {
            flat.push(FlatNode {
                id: &node.id,
                depth,
                is_leaf: node.is_leaf(),
                expanded: node.expanded,
                selected: node.selected,
                has_children: !node.children.is_empty(),
            });
            if node.expanded {
                for c in &node.children {
                    recurse(c, depth + 1, flat);
                }
            }
        }
        for r in &self.root { recurse(r, 0, &mut flat); }
        flat
    }

    /// Move a node to a new parent (or root) at a given index.
    pub fn move_node(&mut self, node_id: &str, new_parent_id: Option<&str>, index: usize) -> bool {
        let removed = self.remove_from_anywhere(node_id);
        let Some(node) = removed else { return false };

        match new_parent_id {
            None => {
                let idx = index.min(self.root.len());
                self.root.insert(idx, node);
                true
            }
            Some(pid) => {
                if let Some(parent) = self.find_mut(pid) {
                    let idx = index.min(parent.children.len());
                    parent.children.insert(idx, node);
                    true
                } else {
                    self.root.push(node);
                    false
                }
            }
        }
    }

    /// Depth-first walk over all nodes (regardless of expanded state).
    pub fn walk(&self, mut f: impl FnMut(&TreeNode<T>, usize)) {
        fn recurse<T>(node: &TreeNode<T>, depth: usize, f: &mut impl FnMut(&TreeNode<T>, usize)) {
            f(node, depth);
            for c in &node.children { recurse(c, depth + 1, f); }
        }
        for r in &self.root { recurse(r, 0, &mut f); }
    }

    // ── private helpers ───────────────────────────────────────

    fn deselect_all(&mut self) {
        fn desel<T>(node: &mut TreeNode<T>) {
            node.selected = false;
            for c in &mut node.children { desel(c); }
        }
        for r in &mut self.root { desel(r); }
    }

    fn remove_from_anywhere(&mut self, id: &str) -> Option<TreeNode<T>> {
        if let Some(pos) = self.root.iter().position(|n| n.id == id) {
            return Some(self.root.remove(pos));
        }
        for r in &mut self.root {
            if let Some(n) = remove_recursive(r, id) {
                return Some(n);
            }
        }
        None
    }
}

fn remove_recursive<T>(node: &mut TreeNode<T>, id: &str) -> Option<TreeNode<T>> {
    if let Some(pos) = node.children.iter().position(|c| c.id == id) {
        return Some(node.children.remove(pos));
    }
    for child in &mut node.children {
        if let Some(n) = remove_recursive(child, id) {
            return Some(n);
        }
    }
    None
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> Tree<&'static str> {
        let mut tree = Tree::new();
        let mut root = TreeNode::new("r1", "Root");
        let mut child1 = TreeNode::new("c1", "Child 1");
        child1.add_child(TreeNode::new("gc1", "Grandchild 1"));
        child1.add_child(TreeNode::new("gc2", "Grandchild 2"));
        root.add_child(child1);
        root.add_child(TreeNode::new("c2", "Child 2"));
        tree.add_root(root);
        tree
    }

    #[test]
    fn add_and_find() {
        let tree = sample_tree();
        assert!(tree.find("r1").is_some());
        assert!(tree.find("gc1").is_some());
        assert!(tree.find("missing").is_none());
    }

    #[test]
    fn nested_children_depth() {
        let tree = sample_tree();
        let r = tree.find("r1").unwrap();
        assert_eq!(r.depth(), 2);
        assert_eq!(r.child_count(), 2);
    }

    #[test]
    fn flatten_respects_expanded() {
        let mut tree = sample_tree();
        let flat = tree.flatten();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].id, "r1");

        tree.toggle_expand("r1");
        let flat = tree.flatten();
        assert_eq!(flat.len(), 3); // r1, c1, c2
        assert_eq!(flat[1].depth, 1);

        tree.toggle_expand("c1");
        let flat = tree.flatten();
        assert_eq!(flat.len(), 5); // r1, c1, gc1, gc2, c2
        assert_eq!(flat[2].depth, 2);
    }

    #[test]
    fn expand_collapse_all() {
        let mut tree = sample_tree();
        tree.expand_all();
        let flat = tree.flatten();
        assert_eq!(flat.len(), 5);
        tree.collapse_all();
        let flat = tree.flatten();
        assert_eq!(flat.len(), 1);
    }

    #[test]
    fn single_select_deselects_previous() {
        let mut tree = sample_tree();
        tree.select("c1");
        assert_eq!(tree.selected_ids(), vec!["c1"]);
        tree.select("c2");
        assert_eq!(tree.selected_ids(), vec!["c2"]);
    }

    #[test]
    fn multi_select() {
        let mut tree = sample_tree();
        tree.multi_select = true;
        tree.select("c1");
        tree.select("gc1");
        let sel = tree.selected_ids();
        assert!(sel.contains(&"c1"));
        assert!(sel.contains(&"gc1"));
        assert_eq!(sel.len(), 2);
    }

    #[test]
    fn toggle_select() {
        let mut tree = sample_tree();
        tree.select("c1");
        assert_eq!(tree.selected_ids(), vec!["c1"]);
        tree.toggle_select("c1");
        assert!(tree.selected_ids().is_empty());
    }

    #[test]
    fn move_node_to_root() {
        let mut tree = sample_tree();
        assert!(tree.move_node("gc1", None, 0));
        assert!(tree.find("gc1").is_some());
        assert_eq!(tree.root.len(), 2);
        assert_eq!(tree.root[0].id, "gc1");
    }

    #[test]
    fn walk_visits_all() {
        let tree = sample_tree();
        let mut count = 0;
        tree.walk(|_, _| count += 1);
        assert_eq!(count, 5);
    }

    #[test]
    fn is_leaf() {
        let tree = sample_tree();
        assert!(!tree.find("r1").unwrap().is_leaf());
        assert!(tree.find("gc1").unwrap().is_leaf());
    }

    #[test]
    fn remove_child() {
        let mut tree = sample_tree();
        let r = tree.find_mut("c1").unwrap();
        let removed = r.remove_child("gc1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "gc1");
        assert_eq!(r.child_count(), 1);
    }
}
