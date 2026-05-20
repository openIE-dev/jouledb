//! Virtual DOM with O(n) keyed diffing.
//!
//! Provides a `VNode` tree representation and a `diff` function that produces
//! a minimal `Vec<Patch>` to transform an old tree into a new one.

use std::collections::HashMap;

// ── Node types ──────────────────────────────────────────────────

/// Binding between a DOM event name and a handler id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventBinding {
    pub event_name: String,
    pub handler_id: u64,
}

/// A virtual DOM node.
#[derive(Debug, Clone, PartialEq)]
pub enum VNode {
    Element {
        tag: String,
        attrs: HashMap<String, String>,
        key: Option<String>,
        children: Vec<VNode>,
        event_handlers: Vec<EventBinding>,
    },
    Text(String),
    Fragment(Vec<VNode>),
    Empty,
}

impl VNode {
    // ── Constructors ────────────────────────────────────────────

    pub fn element(tag: &str) -> Self {
        VNode::Element {
            tag: tag.to_string(),
            attrs: HashMap::new(),
            key: None,
            children: Vec::new(),
            event_handlers: Vec::new(),
        }
    }

    pub fn text(s: &str) -> Self {
        VNode::Text(s.to_string())
    }

    pub fn fragment(children: Vec<VNode>) -> Self {
        VNode::Fragment(children)
    }

    // ── Builder methods (consume + return self) ─────────────────

    pub fn attr(mut self, key: &str, value: &str) -> Self {
        if let VNode::Element { ref mut attrs, .. } = self {
            attrs.insert(key.to_string(), value.to_string());
        }
        self
    }

    pub fn child(mut self, node: VNode) -> Self {
        if let VNode::Element { ref mut children, .. } = self {
            children.push(node);
        }
        self
    }

    pub fn key(mut self, k: &str) -> Self {
        if let VNode::Element { ref mut key, .. } = self {
            *key = Some(k.to_string());
        }
        self
    }

    pub fn on(mut self, event: &str, handler_id: u64) -> Self {
        if let VNode::Element { ref mut event_handlers, .. } = self {
            event_handlers.push(EventBinding {
                event_name: event.to_string(),
                handler_id,
            });
        }
        self
    }
}

// ── Patches ─────────────────────────────────────────────────────

/// A single mutation to apply to the real DOM.
#[derive(Debug, Clone, PartialEq)]
pub enum Patch {
    Replace {
        path: Vec<usize>,
        node: VNode,
    },
    InsertChild {
        path: Vec<usize>,
        index: usize,
        node: VNode,
    },
    RemoveChild {
        path: Vec<usize>,
        index: usize,
    },
    UpdateAttrs {
        path: Vec<usize>,
        set: HashMap<String, String>,
        remove: Vec<String>,
    },
    UpdateText {
        path: Vec<usize>,
        text: String,
    },
    ReorderChildren {
        path: Vec<usize>,
        moves: Vec<(usize, usize)>,
    },
}

// ── Diff algorithm ──────────────────────────────────────────────

/// Compute the minimal set of patches to transform `old` into `new`.
pub fn diff(old: &VNode, new: &VNode) -> Vec<Patch> {
    let mut patches = Vec::new();
    diff_recursive(old, new, &mut Vec::new(), &mut patches);
    patches
}

fn diff_recursive(
    old: &VNode,
    new: &VNode,
    path: &mut Vec<usize>,
    patches: &mut Vec<Patch>,
) {
    match (old, new) {
        // ── Both text ──
        (VNode::Text(a), VNode::Text(b)) => {
            if a != b {
                patches.push(Patch::UpdateText {
                    path: path.clone(),
                    text: b.clone(),
                });
            }
        }
        // ── Both empty ──
        (VNode::Empty, VNode::Empty) => {}
        // ── Both elements ──
        (
            VNode::Element {
                tag: tag_a,
                attrs: attrs_a,
                key: key_a,
                children: children_a,
                ..
            },
            VNode::Element {
                tag: tag_b,
                attrs: attrs_b,
                key: key_b,
                children: children_b,
                ..
            },
        ) => {
            // Different tag or key → full replace
            if tag_a != tag_b || key_a != key_b {
                patches.push(Patch::Replace {
                    path: path.clone(),
                    node: new.clone(),
                });
                return;
            }

            // Diff attributes
            diff_attrs(attrs_a, attrs_b, path, patches);

            // Diff children
            diff_children(children_a, children_b, path, patches);
        }
        // ── Both fragments ──
        (VNode::Fragment(a), VNode::Fragment(b)) => {
            diff_children(a, b, path, patches);
        }
        // ── Mismatched variants → replace ──
        _ => {
            patches.push(Patch::Replace {
                path: path.clone(),
                node: new.clone(),
            });
        }
    }
}

fn diff_attrs(
    old: &HashMap<String, String>,
    new: &HashMap<String, String>,
    path: &[usize],
    patches: &mut Vec<Patch>,
) {
    let mut set = HashMap::new();
    let mut remove = Vec::new();

    // Attrs added or changed
    for (k, v) in new {
        match old.get(k) {
            Some(old_v) if old_v == v => {}
            _ => {
                set.insert(k.clone(), v.clone());
            }
        }
    }
    // Attrs removed
    for k in old.keys() {
        if !new.contains_key(k) {
            remove.push(k.clone());
        }
    }

    if !set.is_empty() || !remove.is_empty() {
        patches.push(Patch::UpdateAttrs {
            path: path.to_vec(),
            set,
            remove,
        });
    }
}

/// Diff two child lists using keyed LIS-based diffing when keys are present,
/// falling back to index-based diffing otherwise.
fn diff_children(
    old: &[VNode],
    new: &[VNode],
    path: &mut Vec<usize>,
    patches: &mut Vec<Patch>,
) {
    let old_keys = extract_keys(old);
    let new_keys = extract_keys(new);

    let any_keys = old_keys.iter().any(|k| k.is_some()) || new_keys.iter().any(|k| k.is_some());

    if any_keys {
        diff_children_keyed(old, new, &old_keys, &new_keys, path, patches);
    } else {
        diff_children_unkeyed(old, new, path, patches);
    }
}

fn extract_keys(nodes: &[VNode]) -> Vec<Option<String>> {
    nodes
        .iter()
        .map(|n| match n {
            VNode::Element { key, .. } => key.clone(),
            _ => None,
        })
        .collect()
}

fn diff_children_unkeyed(
    old: &[VNode],
    new: &[VNode],
    path: &mut Vec<usize>,
    patches: &mut Vec<Patch>,
) {
    let min_len = old.len().min(new.len());

    // Diff common prefix
    for i in 0..min_len {
        path.push(i);
        diff_recursive(&old[i], &new[i], path, patches);
        path.pop();
    }

    // Extra new children → insert
    for i in min_len..new.len() {
        patches.push(Patch::InsertChild {
            path: path.clone(),
            index: i,
            node: new[i].clone(),
        });
    }

    // Extra old children → remove (from end to keep indices stable)
    for i in (min_len..old.len()).rev() {
        patches.push(Patch::RemoveChild {
            path: path.clone(),
            index: i,
        });
    }
}

fn diff_children_keyed(
    old: &[VNode],
    new: &[VNode],
    old_keys: &[Option<String>],
    new_keys: &[Option<String>],
    path: &mut Vec<usize>,
    patches: &mut Vec<Patch>,
) {
    // Build map: key -> old index
    let mut old_key_map: HashMap<String, usize> = HashMap::new();
    for (i, k) in old_keys.iter().enumerate() {
        if let Some(key) = k {
            old_key_map.insert(key.clone(), i);
        }
    }

    // For each new child, find its old index (if any).
    let mut new_to_old: Vec<Option<usize>> = Vec::with_capacity(new.len());
    for k in new_keys {
        if let Some(key) = k {
            new_to_old.push(old_key_map.get(key).copied());
        } else {
            new_to_old.push(None);
        }
    }

    // Track which old indices are reused
    let mut old_used: Vec<bool> = vec![false; old.len()];
    for idx in &new_to_old {
        if let Some(i) = idx {
            old_used[*i] = true;
        }
    }

    // Remove old children that are not in new (reverse order for index stability)
    for i in (0..old.len()).rev() {
        if !old_used[i] {
            patches.push(Patch::RemoveChild {
                path: path.clone(),
                index: i,
            });
        }
    }

    // Insert new children that have no old counterpart
    for (ni, old_idx) in new_to_old.iter().enumerate() {
        if old_idx.is_none() {
            patches.push(Patch::InsertChild {
                path: path.clone(),
                index: ni,
                node: new[ni].clone(),
            });
        }
    }

    // For matched nodes, compute moves via LIS on old indices sequence.
    let matched: Vec<(usize, usize)> = new_to_old
        .iter()
        .enumerate()
        .filter_map(|(ni, oi)| oi.map(|o| (ni, o)))
        .collect();

    if !matched.is_empty() {
        // old indices in new order
        let old_indices: Vec<usize> = matched.iter().map(|&(_, o)| o).collect();
        let lis = longest_increasing_subsequence(&old_indices);
        let lis_set: std::collections::HashSet<usize> =
            lis.into_iter().collect();

        let mut moves = Vec::new();
        for (seq_idx, &(new_idx, old_idx)) in matched.iter().enumerate() {
            if !lis_set.contains(&seq_idx) {
                moves.push((old_idx, new_idx));
            }
        }

        if !moves.is_empty() {
            patches.push(Patch::ReorderChildren {
                path: path.clone(),
                moves,
            });
        }

        // Diff content of matched pairs
        for &(new_idx, old_idx) in &matched {
            path.push(new_idx);
            diff_recursive(&old[old_idx], &new[new_idx], path, patches);
            path.pop();
        }
    }
}

/// Returns indices into `seq` that form the longest increasing subsequence.
fn longest_increasing_subsequence(seq: &[usize]) -> Vec<usize> {
    if seq.is_empty() {
        return Vec::new();
    }

    let n = seq.len();
    // tails[i] = index into seq of the smallest tail element for IS of length i+1
    let mut tails: Vec<usize> = Vec::new();
    // prev[i] = predecessor index in seq for element i
    let mut prev = vec![0usize; n];
    // tails_idx tracks the seq-index stored in tails
    let mut tails_idx: Vec<usize> = Vec::new();

    for i in 0..n {
        let val = seq[i];
        // Binary search for leftmost tail >= val
        let pos = tails.partition_point(|t| *t < val);

        if pos == tails.len() {
            tails.push(val);
            tails_idx.push(i);
        } else {
            tails[pos] = val;
            tails_idx[pos] = i;
        }

        prev[i] = if pos > 0 { tails_idx[pos - 1] } else { i };
    }

    // Reconstruct
    let mut result = Vec::with_capacity(tails.len());
    let mut idx = *tails_idx.last().unwrap_or(&0);
    for _ in 0..tails.len() {
        result.push(idx);
        idx = prev[idx];
    }
    result.reverse();
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn el(tag: &str) -> VNode {
        VNode::element(tag)
    }

    #[test]
    fn diff_identical_trees_produces_no_patches() {
        let tree = el("div").child(el("span").attr("class", "a"));
        let patches = diff(&tree, &tree);
        assert!(patches.is_empty());
    }

    #[test]
    fn diff_text_change() {
        let old = VNode::text("hello");
        let new = VNode::text("world");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], Patch::UpdateText { text, .. } if text == "world"));
    }

    #[test]
    fn diff_text_same() {
        let node = VNode::text("same");
        assert!(diff(&node, &node).is_empty());
    }

    #[test]
    fn diff_attr_add() {
        let old = el("div");
        let new = el("div").attr("class", "x");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            Patch::UpdateAttrs { set, remove, .. } => {
                assert_eq!(set.get("class").map(String::as_str), Some("x"));
                assert!(remove.is_empty());
            }
            _ => panic!("expected UpdateAttrs"),
        }
    }

    #[test]
    fn diff_attr_remove() {
        let old = el("div").attr("class", "x");
        let new = el("div");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            Patch::UpdateAttrs { remove, set, .. } => {
                assert!(remove.contains(&"class".to_string()));
                assert!(set.is_empty());
            }
            _ => panic!("expected UpdateAttrs"),
        }
    }

    #[test]
    fn diff_attr_change() {
        let old = el("div").attr("class", "a");
        let new = el("div").attr("class", "b");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        match &patches[0] {
            Patch::UpdateAttrs { set, .. } => {
                assert_eq!(set.get("class").map(String::as_str), Some("b"));
            }
            _ => panic!("expected UpdateAttrs"),
        }
    }

    #[test]
    fn diff_child_insert() {
        let old = el("ul");
        let new = el("ul").child(el("li"));
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], Patch::InsertChild { index: 0, .. }));
    }

    #[test]
    fn diff_child_remove() {
        let old = el("ul").child(el("li"));
        let new = el("ul");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], Patch::RemoveChild { index: 0, .. }));
    }

    #[test]
    fn diff_child_reorder_keyed() {
        let old = el("ul")
            .child(el("li").key("a").child(VNode::text("A")))
            .child(el("li").key("b").child(VNode::text("B")))
            .child(el("li").key("c").child(VNode::text("C")));
        let new = el("ul")
            .child(el("li").key("c").child(VNode::text("C")))
            .child(el("li").key("a").child(VNode::text("A")))
            .child(el("li").key("b").child(VNode::text("B")));
        let patches = diff(&old, &new);
        // Should produce a ReorderChildren, not 3 removes + 3 inserts
        let has_reorder = patches.iter().any(|p| matches!(p, Patch::ReorderChildren { .. }));
        assert!(has_reorder, "expected ReorderChildren patch, got: {patches:?}");
        // Should NOT have any Remove+Insert patches
        let removes = patches.iter().filter(|p| matches!(p, Patch::RemoveChild { .. })).count();
        let inserts = patches.iter().filter(|p| matches!(p, Patch::InsertChild { .. })).count();
        assert_eq!(removes, 0);
        assert_eq!(inserts, 0);
    }

    #[test]
    fn diff_replace_different_tag() {
        let old = el("div");
        let new = el("span");
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], Patch::Replace { .. }));
    }

    #[test]
    fn diff_fragment_handling() {
        let old = VNode::fragment(vec![VNode::text("a"), VNode::text("b")]);
        let new = VNode::fragment(vec![VNode::text("a"), VNode::text("c")]);
        let patches = diff(&old, &new);
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], Patch::UpdateText { text, .. } if text == "c"));
    }

    #[test]
    fn diff_empty_nodes() {
        assert!(diff(&VNode::Empty, &VNode::Empty).is_empty());
    }

    #[test]
    fn diff_empty_to_element() {
        let patches = diff(&VNode::Empty, &el("div"));
        assert_eq!(patches.len(), 1);
        assert!(matches!(&patches[0], Patch::Replace { .. }));
    }

    #[test]
    fn diff_event_bindings_preserved() {
        let old = el("button").on("click", 1);
        let new = el("button").on("click", 1);
        assert!(diff(&old, &new).is_empty());
    }

    #[test]
    fn lis_basic() {
        let result = longest_increasing_subsequence(&[3, 1, 2, 0, 4]);
        // LIS is [1, 2, 4] at indices [1, 2, 4]
        assert_eq!(result.len(), 3);
    }
}
