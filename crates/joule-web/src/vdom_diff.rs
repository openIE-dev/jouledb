//! Virtual DOM diffing engine — keyed reconciliation with minimal patch sets.
//!
//! Replaces virtual-dom / snabbdom / inferno-vdom diffing with a pure-Rust
//! implementation. Produces a minimal `Vec<DiffPatch>` to transform one VNode
//! tree into another, supporting keyed children, attribute modifications,
//! reorder operations, and component-level replace.

use std::collections::HashMap;

// ── Node types ──────────────────────────────────────────────────────────

/// Virtual DOM node — the core tree element.
#[derive(Debug, Clone, PartialEq)]
pub enum VNode {
    /// An element node with tag, attributes, key, and children.
    Element(VElement),
    /// A text node.
    Text(String),
    /// A component placeholder referencing a named component with props.
    Component {
        name: String,
        props: HashMap<String, String>,
        key: Option<String>,
        children: Vec<VNode>,
    },
    /// An empty placeholder (renders nothing).
    Empty,
}

/// A virtual element with tag, attributes, optional key, and children.
#[derive(Debug, Clone, PartialEq)]
pub struct VElement {
    pub tag: String,
    pub attrs: HashMap<String, String>,
    pub key: Option<String>,
    pub children: Vec<VNode>,
}

impl VElement {
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            attrs: HashMap::new(),
            key: None,
            children: Vec::new(),
        }
    }

    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attrs.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_key(mut self, key: &str) -> Self {
        self.key = Some(key.to_string());
        self
    }

    pub fn with_child(mut self, child: VNode) -> Self {
        self.children.push(child);
        self
    }
}

impl VNode {
    /// Create a new element node.
    pub fn element(tag: &str) -> Self {
        VNode::Element(VElement::new(tag))
    }

    /// Create a text node.
    pub fn text(s: &str) -> Self {
        VNode::Text(s.to_string())
    }

    /// Create a component node.
    pub fn component(name: &str) -> Self {
        VNode::Component {
            name: name.to_string(),
            props: HashMap::new(),
            key: None,
            children: Vec::new(),
        }
    }

    /// Get the key if this node has one.
    pub fn key(&self) -> Option<&str> {
        match self {
            VNode::Element(el) => el.key.as_deref(),
            VNode::Component { key, .. } => key.as_deref(),
            _ => None,
        }
    }

    /// Get children of this node.
    pub fn children(&self) -> &[VNode] {
        match self {
            VNode::Element(el) => &el.children,
            VNode::Component { children, .. } => children,
            _ => &[],
        }
    }

    /// Check if this is the same "type" as another node (same tag or component name).
    pub fn is_same_type(&self, other: &VNode) -> bool {
        match (self, other) {
            (VNode::Element(a), VNode::Element(b)) => a.tag == b.tag,
            (VNode::Text(_), VNode::Text(_)) => true,
            (VNode::Component { name: a, .. }, VNode::Component { name: b, .. }) => a == b,
            (VNode::Empty, VNode::Empty) => true,
            _ => false,
        }
    }
}

// ── Patches ─────────────────────────────────────────────────────────────

/// A single diff patch to transform the old tree into the new tree.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffPatch {
    /// Replace the node at `path` with a new node.
    Replace {
        path: Vec<usize>,
        new_node: VNode,
    },
    /// Add a child at `path[..path.len()-1]` at index `path.last()`.
    AddChild {
        path: Vec<usize>,
        node: VNode,
    },
    /// Remove the child at `path`.
    RemoveChild {
        path: Vec<usize>,
    },
    /// Modify attributes on the element at `path`.
    ModifyAttrs {
        path: Vec<usize>,
        set: HashMap<String, String>,
        remove: Vec<String>,
    },
    /// Reorder children at `path` according to the moves list.
    Reorder {
        path: Vec<usize>,
        moves: Vec<ReorderMove>,
    },
    /// Update text content at `path`.
    UpdateText {
        path: Vec<usize>,
        new_text: String,
    },
}

/// A single child reorder operation.
#[derive(Debug, Clone, PartialEq)]
pub struct ReorderMove {
    pub from: usize,
    pub to: usize,
}

// ── Diff algorithm ──────────────────────────────────────────────────────

/// Diff two VNode trees and produce a minimal set of patches.
pub fn diff(old: &VNode, new: &VNode) -> Vec<DiffPatch> {
    let mut patches = Vec::new();
    diff_recursive(old, new, &[], &mut patches);
    patches
}

fn diff_recursive(
    old: &VNode,
    new: &VNode,
    path: &[usize],
    patches: &mut Vec<DiffPatch>,
) {
    if !old.is_same_type(new) {
        patches.push(DiffPatch::Replace {
            path: path.to_vec(),
            new_node: new.clone(),
        });
        return;
    }

    match (old, new) {
        (VNode::Text(old_text), VNode::Text(new_text)) => {
            if old_text != new_text {
                patches.push(DiffPatch::UpdateText {
                    path: path.to_vec(),
                    new_text: new_text.clone(),
                });
            }
        }
        (VNode::Element(old_el), VNode::Element(new_el)) => {
            // Diff attributes
            diff_attrs(old_el, new_el, path, patches);
            // Diff children
            diff_children(&old_el.children, &new_el.children, path, patches);
        }
        (
            VNode::Component { props: old_props, children: old_children, .. },
            VNode::Component { props: new_props, children: new_children, .. },
        ) => {
            // Diff props as attributes
            let old_el_stub = VElement {
                tag: String::new(),
                attrs: old_props.clone(),
                key: None,
                children: Vec::new(),
            };
            let new_el_stub = VElement {
                tag: String::new(),
                attrs: new_props.clone(),
                key: None,
                children: Vec::new(),
            };
            diff_attrs(&old_el_stub, &new_el_stub, path, patches);
            diff_children(old_children, new_children, path, patches);
        }
        (VNode::Empty, VNode::Empty) => {}
        _ => {}
    }
}

fn diff_attrs(
    old_el: &VElement,
    new_el: &VElement,
    path: &[usize],
    patches: &mut Vec<DiffPatch>,
) {
    let mut set = HashMap::new();
    let mut remove = Vec::new();

    // Find changed or new attributes
    for (k, v) in &new_el.attrs {
        match old_el.attrs.get(k) {
            Some(old_v) if old_v == v => {}
            _ => {
                set.insert(k.clone(), v.clone());
            }
        }
    }

    // Find removed attributes
    for k in old_el.attrs.keys() {
        if !new_el.attrs.contains_key(k) {
            remove.push(k.clone());
        }
    }

    if !set.is_empty() || !remove.is_empty() {
        patches.push(DiffPatch::ModifyAttrs {
            path: path.to_vec(),
            set,
            remove,
        });
    }
}

fn diff_children(
    old_children: &[VNode],
    new_children: &[VNode],
    parent_path: &[usize],
    patches: &mut Vec<DiffPatch>,
) {
    let old_has_keys = old_children.iter().any(|c| c.key().is_some());
    let new_has_keys = new_children.iter().any(|c| c.key().is_some());

    if old_has_keys && new_has_keys {
        diff_keyed_children(old_children, new_children, parent_path, patches);
    } else {
        diff_unkeyed_children(old_children, new_children, parent_path, patches);
    }
}

fn diff_unkeyed_children(
    old_children: &[VNode],
    new_children: &[VNode],
    parent_path: &[usize],
    patches: &mut Vec<DiffPatch>,
) {
    let common_len = old_children.len().min(new_children.len());

    for i in 0..common_len {
        let mut child_path = parent_path.to_vec();
        child_path.push(i);
        diff_recursive(&old_children[i], &new_children[i], &child_path, patches);
    }

    // Extra new children — add them
    for i in common_len..new_children.len() {
        let mut child_path = parent_path.to_vec();
        child_path.push(i);
        patches.push(DiffPatch::AddChild {
            path: child_path,
            node: new_children[i].clone(),
        });
    }

    // Extra old children — remove them (in reverse order for index stability)
    for i in (common_len..old_children.len()).rev() {
        let mut child_path = parent_path.to_vec();
        child_path.push(i);
        patches.push(DiffPatch::RemoveChild {
            path: child_path,
        });
    }
}

fn diff_keyed_children(
    old_children: &[VNode],
    new_children: &[VNode],
    parent_path: &[usize],
    patches: &mut Vec<DiffPatch>,
) {
    // Build key-to-index maps
    let mut old_key_map: HashMap<String, usize> = HashMap::new();
    for (i, child) in old_children.iter().enumerate() {
        if let Some(k) = child.key() {
            old_key_map.insert(k.to_string(), i);
        }
    }

    let mut new_key_map: HashMap<String, usize> = HashMap::new();
    for (i, child) in new_children.iter().enumerate() {
        if let Some(k) = child.key() {
            new_key_map.insert(k.to_string(), i);
        }
    }

    // Track reorder moves
    let mut moves = Vec::new();

    // Remove old children that are not in new list
    for (i, child) in old_children.iter().enumerate().rev() {
        if let Some(k) = child.key() {
            if !new_key_map.contains_key(k) {
                let mut child_path = parent_path.to_vec();
                child_path.push(i);
                patches.push(DiffPatch::RemoveChild { path: child_path });
            }
        }
    }

    // Add new children that are not in old list, diff those that are
    for (new_idx, new_child) in new_children.iter().enumerate() {
        if let Some(k) = new_child.key() {
            if let Some(&old_idx) = old_key_map.get(k) {
                // Exists in both: diff the nodes
                let mut child_path = parent_path.to_vec();
                child_path.push(old_idx);
                diff_recursive(&old_children[old_idx], new_child, &child_path, patches);

                // Track if position changed
                if old_idx != new_idx {
                    moves.push(ReorderMove {
                        from: old_idx,
                        to: new_idx,
                    });
                }
            } else {
                // New child
                let mut child_path = parent_path.to_vec();
                child_path.push(new_idx);
                patches.push(DiffPatch::AddChild {
                    path: child_path,
                    node: new_child.clone(),
                });
            }
        }
    }

    if !moves.is_empty() {
        patches.push(DiffPatch::Reorder {
            path: parent_path.to_vec(),
            moves,
        });
    }
}

// ── Patch application helper ────────────────────────────────────────────

/// Statistics from a diff operation.
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    pub replacements: usize,
    pub additions: usize,
    pub removals: usize,
    pub attr_modifications: usize,
    pub reorders: usize,
    pub text_updates: usize,
}

impl DiffStats {
    /// Compute statistics from a patch list.
    pub fn from_patches(patches: &[DiffPatch]) -> Self {
        let mut stats = Self::default();
        for p in patches {
            match p {
                DiffPatch::Replace { .. } => stats.replacements += 1,
                DiffPatch::AddChild { .. } => stats.additions += 1,
                DiffPatch::RemoveChild { .. } => stats.removals += 1,
                DiffPatch::ModifyAttrs { .. } => stats.attr_modifications += 1,
                DiffPatch::Reorder { .. } => stats.reorders += 1,
                DiffPatch::UpdateText { .. } => stats.text_updates += 1,
            }
        }
        stats
    }

    /// Total number of operations.
    pub fn total(&self) -> usize {
        self.replacements
            + self.additions
            + self.removals
            + self.attr_modifications
            + self.reorders
            + self.text_updates
    }
}

/// Count the total number of nodes in a VNode tree.
pub fn count_nodes(node: &VNode) -> usize {
    match node {
        VNode::Empty => 1,
        VNode::Text(_) => 1,
        VNode::Element(el) => {
            1 + el.children.iter().map(count_nodes).sum::<usize>()
        }
        VNode::Component { children, .. } => {
            1 + children.iter().map(count_nodes).sum::<usize>()
        }
    }
}

/// Collect all keys present in a VNode tree.
pub fn collect_keys(node: &VNode) -> Vec<String> {
    let mut keys = Vec::new();
    collect_keys_recursive(node, &mut keys);
    keys
}

fn collect_keys_recursive(node: &VNode, keys: &mut Vec<String>) {
    if let Some(k) = node.key() {
        keys.push(k.to_string());
    }
    for child in node.children() {
        collect_keys_recursive(child, keys);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn div() -> VElement {
        VElement::new("div")
    }

    fn span() -> VElement {
        VElement::new("span")
    }

    #[test]
    fn identical_trees_no_patches() {
        let old = VNode::Element(div().with_child(VNode::text("hi")));
        let new = VNode::Element(div().with_child(VNode::text("hi")));
        let patches = diff(&old, &new);
        assert!(patches.is_empty());
    }

    #[test]
    fn text_change_produces_update_text() {
        let old = VNode::text("hello");
        let new = VNode::text("world");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], DiffPatch::UpdateText { new_text, .. } if new_text == "world"));
    }

    #[test]
    fn different_tags_produce_replace() {
        let old = VNode::Element(div());
        let new = VNode::Element(span());
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], DiffPatch::Replace { .. }));
    }

    #[test]
    fn add_attribute() {
        let old = VNode::Element(div());
        let new = VNode::Element(div().with_attr("class", "red"));
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            DiffPatch::ModifyAttrs { set, remove, .. } => {
                assert_eq!(set.get("class").unwrap(), "red");
                assert!(remove.is_empty());
            }
            other => panic!("expected ModifyAttrs, got {:?}", other),
        }
    }

    #[test]
    fn remove_attribute() {
        let old = VNode::Element(div().with_attr("class", "red"));
        let new = VNode::Element(div());
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            DiffPatch::ModifyAttrs { set, remove, .. } => {
                assert!(set.is_empty());
                assert_eq!(remove, &["class"]);
            }
            other => panic!("expected ModifyAttrs, got {:?}", other),
        }
    }

    #[test]
    fn change_attribute_value() {
        let old = VNode::Element(div().with_attr("class", "red"));
        let new = VNode::Element(div().with_attr("class", "blue"));
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            DiffPatch::ModifyAttrs { set, .. } => {
                assert_eq!(set.get("class").unwrap(), "blue");
            }
            other => panic!("expected ModifyAttrs, got {:?}", other),
        }
    }

    #[test]
    fn add_child_to_empty_parent() {
        let old = VNode::Element(div());
        let new = VNode::Element(div().with_child(VNode::text("child")));
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], DiffPatch::AddChild { .. }));
    }

    #[test]
    fn remove_child() {
        let old = VNode::Element(div().with_child(VNode::text("child")));
        let new = VNode::Element(div());
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], DiffPatch::RemoveChild { .. }));
    }

    #[test]
    fn nested_change_produces_correct_path() {
        let old = VNode::Element(
            div().with_child(VNode::Element(span().with_child(VNode::text("old")))),
        );
        let new = VNode::Element(
            div().with_child(VNode::Element(span().with_child(VNode::text("new")))),
        );
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            DiffPatch::UpdateText { path, new_text } => {
                assert_eq!(path, &[0, 0]);
                assert_eq!(new_text, "new");
            }
            other => panic!("expected UpdateText, got {:?}", other),
        }
    }

    #[test]
    fn keyed_children_reorder() {
        let old = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("a")))
                .with_child(VNode::Element(span().with_key("b")))
                .with_child(VNode::Element(span().with_key("c"))),
        );
        let new = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("c")))
                .with_child(VNode::Element(span().with_key("a")))
                .with_child(VNode::Element(span().with_key("b"))),
        );
        let patches = diff(&old, &new);
        let stats = DiffStats::from_patches(&patches);
        assert!(stats.reorders > 0);
    }

    #[test]
    fn keyed_children_add_new_key() {
        let old = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("a")))
                .with_child(VNode::Element(span().with_key("b"))),
        );
        let new = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("a")))
                .with_child(VNode::Element(span().with_key("b")))
                .with_child(VNode::Element(span().with_key("c"))),
        );
        let patches = diff(&old, &new);
        let stats = DiffStats::from_patches(&patches);
        assert!(stats.additions > 0);
    }

    #[test]
    fn keyed_children_remove_key() {
        let old = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("a")))
                .with_child(VNode::Element(span().with_key("b")))
                .with_child(VNode::Element(span().with_key("c"))),
        );
        let new = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("a")))
                .with_child(VNode::Element(span().with_key("c"))),
        );
        let patches = diff(&old, &new);
        let stats = DiffStats::from_patches(&patches);
        assert!(stats.removals > 0);
    }

    #[test]
    fn component_node_diff_props() {
        let old = VNode::Component {
            name: "Button".to_string(),
            props: {
                let mut m = HashMap::new();
                m.insert("label".to_string(), "Click".to_string());
                m
            },
            key: None,
            children: Vec::new(),
        };
        let new = VNode::Component {
            name: "Button".to_string(),
            props: {
                let mut m = HashMap::new();
                m.insert("label".to_string(), "Submit".to_string());
                m
            },
            key: None,
            children: Vec::new(),
        };
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], DiffPatch::ModifyAttrs { .. }));
    }

    #[test]
    fn replace_text_with_element() {
        let old = VNode::text("hello");
        let new = VNode::Element(div());
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], DiffPatch::Replace { .. }));
    }

    #[test]
    fn diff_stats_counts() {
        let old = VNode::Element(
            div()
                .with_attr("class", "old")
                .with_child(VNode::text("keep"))
                .with_child(VNode::text("remove")),
        );
        let new = VNode::Element(
            div()
                .with_attr("class", "new")
                .with_child(VNode::text("keep")),
        );
        let patches = diff(&old, &new);
        let stats = DiffStats::from_patches(&patches);
        assert!(stats.total() > 0);
        assert!(stats.attr_modifications > 0);
    }

    #[test]
    fn count_nodes_tree() {
        let tree = VNode::Element(
            div()
                .with_child(VNode::text("a"))
                .with_child(VNode::Element(span().with_child(VNode::text("b")))),
        );
        assert_eq!(count_nodes(&tree), 4);
    }

    #[test]
    fn collect_keys_from_tree() {
        let tree = VNode::Element(
            div()
                .with_child(VNode::Element(span().with_key("x")))
                .with_child(VNode::Element(span().with_key("y"))),
        );
        let keys = collect_keys(&tree);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"x".to_string()));
        assert!(keys.contains(&"y".to_string()));
    }

    #[test]
    fn empty_nodes_no_patches() {
        let old = VNode::Empty;
        let new = VNode::Empty;
        let patches = diff(&old, &new);
        assert!(patches.is_empty());
    }

    #[test]
    fn multiple_children_changes() {
        let old = VNode::Element(
            div()
                .with_child(VNode::text("a"))
                .with_child(VNode::text("b"))
                .with_child(VNode::text("c")),
        );
        let new = VNode::Element(
            div()
                .with_child(VNode::text("a"))
                .with_child(VNode::text("x"))
                .with_child(VNode::text("c"))
                .with_child(VNode::text("d")),
        );
        let patches = diff(&old, &new);
        let stats = DiffStats::from_patches(&patches);
        // b -> x = text_update, d = addition
        assert!(stats.text_updates >= 1);
        assert!(stats.additions >= 1);
    }

    #[test]
    fn is_same_type_checks() {
        assert!(VNode::text("a").is_same_type(&VNode::text("b")));
        assert!(VNode::Element(div()).is_same_type(&VNode::Element(div())));
        assert!(!VNode::Element(div()).is_same_type(&VNode::text("x")));
        assert!(!VNode::Element(div()).is_same_type(&VNode::Element(span())));
        assert!(VNode::Empty.is_same_type(&VNode::Empty));
    }

    #[test]
    fn deeply_nested_diff() {
        let build_deep = |text: &str| -> VNode {
            VNode::Element(div().with_child(VNode::Element(
                div().with_child(VNode::Element(div().with_child(VNode::text(text)))),
            )))
        };
        let old = build_deep("deep-old");
        let new = build_deep("deep-new");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            DiffPatch::UpdateText { path, .. } => {
                assert_eq!(path, &[0, 0, 0]);
            }
            other => panic!("expected UpdateText, got {:?}", other),
        }
    }
}
