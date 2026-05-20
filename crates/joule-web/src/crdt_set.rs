//! CRDT sets — G-Set (grow-only), 2P-Set (add/remove), OR-Set (observed-remove),
//! LWW-Element-Set, state-based merge, concurrent add/remove resolution,
//! element query.

use std::collections::{HashMap, HashSet};

// ── G-Set (Grow-Only Set) ────────────────────────────────────────────────────

/// A state-based grow-only set. Elements can only be added, never removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GSet {
    elements: HashSet<String>,
}

impl GSet {
    /// Create a new empty G-Set.
    pub fn new() -> Self {
        Self {
            elements: HashSet::new(),
        }
    }

    /// Add an element.
    pub fn add(&mut self, element: &str) {
        self.elements.insert(element.to_string());
    }

    /// Check if an element is in the set.
    pub fn contains(&self, element: &str) -> bool {
        self.elements.contains(element)
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Merge with another G-Set (union).
    pub fn merge(&mut self, other: &GSet) {
        for elem in &other.elements {
            self.elements.insert(elem.clone());
        }
    }

    /// Create a merged set without mutating self.
    pub fn merged(&self, other: &GSet) -> GSet {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Get all elements as a sorted vector.
    pub fn elements(&self) -> Vec<&str> {
        let mut elems: Vec<&str> = self.elements.iter().map(|s| s.as_str()).collect();
        elems.sort();
        elems
    }

    /// Check if self is a subset of other.
    pub fn is_subset_of(&self, other: &GSet) -> bool {
        self.elements.is_subset(&other.elements)
    }
}

impl Default for GSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── 2P-Set (Two-Phase Set) ──────────────────────────────────────────────────

/// A two-phase set: elements can be added and removed, but once removed,
/// they cannot be re-added. Removal wins over concurrent add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoPSet {
    added: HashSet<String>,
    removed: HashSet<String>,
}

impl TwoPSet {
    /// Create a new empty 2P-Set.
    pub fn new() -> Self {
        Self {
            added: HashSet::new(),
            removed: HashSet::new(),
        }
    }

    /// Add an element. Has no effect if the element was previously removed.
    pub fn add(&mut self, element: &str) {
        if !self.removed.contains(element) {
            self.added.insert(element.to_string());
        }
    }

    /// Remove an element. The element must have been added first.
    /// Returns true if the element was removed.
    pub fn remove(&mut self, element: &str) -> bool {
        if self.added.contains(element) {
            self.removed.insert(element.to_string());
            true
        } else {
            false
        }
    }

    /// Check if an element is in the set (added and not removed).
    pub fn contains(&self, element: &str) -> bool {
        self.added.contains(element) && !self.removed.contains(element)
    }

    /// Get the number of live elements.
    pub fn len(&self) -> usize {
        self.added.difference(&self.removed).count()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Merge with another 2P-Set (union both added and removed sets).
    pub fn merge(&mut self, other: &TwoPSet) {
        for elem in &other.added {
            self.added.insert(elem.clone());
        }
        for elem in &other.removed {
            self.removed.insert(elem.clone());
        }
    }

    /// Create a merged set without mutating self.
    pub fn merged(&self, other: &TwoPSet) -> TwoPSet {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Get all live elements as a sorted vector.
    pub fn elements(&self) -> Vec<&str> {
        let mut elems: Vec<&str> = self.added
            .difference(&self.removed)
            .map(|s| s.as_str())
            .collect();
        elems.sort();
        elems
    }

    /// Check if an element has been removed (tombstoned).
    pub fn is_tombstoned(&self, element: &str) -> bool {
        self.removed.contains(element)
    }
}

impl Default for TwoPSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── OR-Set (Observed-Remove Set) ─────────────────────────────────────────────

/// A unique tag for tracking add operations in OR-Set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UniqueTag {
    /// Replica that created this tag.
    pub replica_id: String,
    /// Sequence number at the replica.
    pub seq: u64,
}

/// An observed-remove set. Elements can be added and removed concurrently.
/// Only the specific add operations that were observed at the time of removal
/// are negated; concurrent adds survive.
#[derive(Debug, Clone)]
pub struct ORSet {
    /// Replica id of this node.
    replica_id: String,
    /// Sequence counter for generating unique tags.
    seq: u64,
    /// Map from element to the set of unique tags tracking its additions.
    elements: HashMap<String, HashSet<UniqueTag>>,
}

impl ORSet {
    /// Create a new OR-Set for the given replica.
    pub fn new(replica_id: &str) -> Self {
        Self {
            replica_id: replica_id.to_string(),
            seq: 0,
            elements: HashMap::new(),
        }
    }

    /// Add an element, creating a new unique tag.
    pub fn add(&mut self, element: &str) {
        self.seq += 1;
        let tag = UniqueTag {
            replica_id: self.replica_id.clone(),
            seq: self.seq,
        };
        self.elements
            .entry(element.to_string())
            .or_default()
            .insert(tag);
    }

    /// Remove an element by clearing all its observed tags.
    /// Returns true if the element was present.
    pub fn remove(&mut self, element: &str) -> bool {
        match self.elements.get_mut(element) {
            Some(tags) if !tags.is_empty() => {
                tags.clear();
                true
            }
            _ => false,
        }
    }

    /// Check if an element is in the set (has at least one active tag).
    pub fn contains(&self, element: &str) -> bool {
        self.elements
            .get(element)
            .map_or(false, |tags| !tags.is_empty())
    }

    /// Get the number of live elements.
    pub fn len(&self) -> usize {
        self.elements
            .values()
            .filter(|tags| !tags.is_empty())
            .count()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Merge with another OR-Set.
    /// For each element, the union of tags from both replicas is taken.
    pub fn merge(&mut self, other: &ORSet) {
        for (elem, other_tags) in &other.elements {
            let my_tags = self.elements.entry(elem.clone()).or_default();
            for tag in other_tags {
                my_tags.insert(tag.clone());
            }
        }
    }

    /// Get all live elements as a sorted vector.
    pub fn elements(&self) -> Vec<&str> {
        let mut elems: Vec<&str> = self.elements
            .iter()
            .filter(|(_, tags)| !tags.is_empty())
            .map(|(elem, _)| elem.as_str())
            .collect();
        elems.sort();
        elems
    }

    /// Get the number of tags for a specific element.
    pub fn tag_count(&self, element: &str) -> usize {
        self.elements
            .get(element)
            .map_or(0, |tags| tags.len())
    }

    /// Get the replica id.
    pub fn replica_id(&self) -> &str {
        &self.replica_id
    }
}

// ── LWW-Element-Set (Last-Writer-Wins Element Set) ───────────────────────────

/// Timestamp entry for LWW-Element-Set.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LwwTimestamp {
    add_ts: f64,
    remove_ts: f64,
}

/// A last-writer-wins element set. Add and remove carry timestamps;
/// the operation with the highest timestamp wins. Ties broken in favor of add.
#[derive(Debug, Clone)]
pub struct LwwElementSet {
    /// Map from element to (add_timestamp, remove_timestamp).
    entries: HashMap<String, LwwTimestamp>,
    /// Bias: if true, add wins ties; if false, remove wins ties.
    add_bias: bool,
}

impl LwwElementSet {
    /// Create a new LWW-Element-Set with add-bias (add wins ties).
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            add_bias: true,
        }
    }

    /// Create a new LWW-Element-Set with remove-bias (remove wins ties).
    pub fn with_remove_bias() -> Self {
        Self {
            entries: HashMap::new(),
            add_bias: false,
        }
    }

    /// Add an element with the given timestamp.
    pub fn add(&mut self, element: &str, timestamp: f64) {
        let entry = self.entries.entry(element.to_string()).or_insert(LwwTimestamp {
            add_ts: f64::NEG_INFINITY,
            remove_ts: f64::NEG_INFINITY,
        });
        if timestamp > entry.add_ts {
            entry.add_ts = timestamp;
        }
    }

    /// Remove an element with the given timestamp.
    pub fn remove(&mut self, element: &str, timestamp: f64) {
        let entry = self.entries.entry(element.to_string()).or_insert(LwwTimestamp {
            add_ts: f64::NEG_INFINITY,
            remove_ts: f64::NEG_INFINITY,
        });
        if timestamp > entry.remove_ts {
            entry.remove_ts = timestamp;
        }
    }

    /// Check if an element is in the set.
    pub fn contains(&self, element: &str) -> bool {
        match self.entries.get(element) {
            Some(ts) => {
                if ts.add_ts > ts.remove_ts {
                    true
                } else if ts.add_ts < ts.remove_ts {
                    false
                } else {
                    // Tie.
                    self.add_bias
                }
            }
            None => false,
        }
    }

    /// Get the number of live elements.
    pub fn len(&self) -> usize {
        self.entries
            .keys()
            .filter(|k| self.contains(k))
            .count()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Merge with another LWW-Element-Set.
    pub fn merge(&mut self, other: &LwwElementSet) {
        for (elem, other_ts) in &other.entries {
            let entry = self.entries.entry(elem.clone()).or_insert(LwwTimestamp {
                add_ts: f64::NEG_INFINITY,
                remove_ts: f64::NEG_INFINITY,
            });
            if other_ts.add_ts > entry.add_ts {
                entry.add_ts = other_ts.add_ts;
            }
            if other_ts.remove_ts > entry.remove_ts {
                entry.remove_ts = other_ts.remove_ts;
            }
        }
    }

    /// Get all live elements as a sorted vector.
    pub fn elements(&self) -> Vec<&str> {
        let mut elems: Vec<&str> = self.entries
            .keys()
            .filter(|k| self.contains(k))
            .map(|s| s.as_str())
            .collect();
        elems.sort();
        elems
    }

    /// Get the add timestamp for an element.
    pub fn add_timestamp(&self, element: &str) -> Option<f64> {
        self.entries.get(element).map(|ts| ts.add_ts)
    }

    /// Get the remove timestamp for an element.
    pub fn remove_timestamp(&self, element: &str) -> Option<f64> {
        self.entries.get(element).map(|ts| ts.remove_ts)
    }
}

impl Default for LwwElementSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── G-Set tests ──

    #[test]
    fn gset_new_is_empty() {
        let s = GSet::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn gset_add_and_contains() {
        let mut s = GSet::new();
        s.add("a");
        s.add("b");
        assert!(s.contains("a"));
        assert!(s.contains("b"));
        assert!(!s.contains("c"));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn gset_merge() {
        let mut s1 = GSet::new();
        let mut s2 = GSet::new();
        s1.add("a");
        s1.add("b");
        s2.add("b");
        s2.add("c");
        s1.merge(&s2);
        assert_eq!(s1.len(), 3);
        assert!(s1.contains("a"));
        assert!(s1.contains("b"));
        assert!(s1.contains("c"));
    }

    #[test]
    fn gset_merge_is_idempotent() {
        let mut s1 = GSet::new();
        let s2 = GSet::new();
        s1.add("x");
        s1.merge(&s2);
        s1.merge(&s2);
        assert_eq!(s1.len(), 1);
    }

    #[test]
    fn gset_is_subset() {
        let mut s1 = GSet::new();
        let mut s2 = GSet::new();
        s1.add("a");
        s2.add("a");
        s2.add("b");
        assert!(s1.is_subset_of(&s2));
        assert!(!s2.is_subset_of(&s1));
    }

    // ── 2P-Set tests ──

    #[test]
    fn twopset_add_and_remove() {
        let mut s = TwoPSet::new();
        s.add("a");
        assert!(s.contains("a"));
        assert!(s.remove("a"));
        assert!(!s.contains("a"));
    }

    #[test]
    fn twopset_remove_is_permanent() {
        let mut s = TwoPSet::new();
        s.add("a");
        s.remove("a");
        s.add("a"); // Re-add should not work.
        assert!(!s.contains("a"));
    }

    #[test]
    fn twopset_cannot_remove_unadded() {
        let mut s = TwoPSet::new();
        assert!(!s.remove("x"));
    }

    #[test]
    fn twopset_merge() {
        let mut s1 = TwoPSet::new();
        let mut s2 = TwoPSet::new();
        s1.add("a");
        s1.add("b");
        s2.add("b");
        s2.add("c");
        s2.remove("b");
        s1.merge(&s2);
        assert!(s1.contains("a"));
        assert!(!s1.contains("b")); // Removed in s2, tombstone propagated.
        assert!(s1.contains("c"));
    }

    #[test]
    fn twopset_is_tombstoned() {
        let mut s = TwoPSet::new();
        s.add("a");
        s.remove("a");
        assert!(s.is_tombstoned("a"));
        assert!(!s.is_tombstoned("b"));
    }

    #[test]
    fn twopset_elements() {
        let mut s = TwoPSet::new();
        s.add("c");
        s.add("a");
        s.add("b");
        s.remove("b");
        let elems = s.elements();
        assert_eq!(elems, vec!["a", "c"]);
    }

    // ── OR-Set tests ──

    #[test]
    fn orset_add_and_contains() {
        let mut s = ORSet::new("r1");
        s.add("a");
        assert!(s.contains("a"));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn orset_remove() {
        let mut s = ORSet::new("r1");
        s.add("a");
        assert!(s.remove("a"));
        assert!(!s.contains("a"));
    }

    #[test]
    fn orset_concurrent_add_and_remove() {
        // Simulate concurrent add (r2) and remove (r1).
        let mut r1 = ORSet::new("r1");
        let mut r2 = ORSet::new("r2");
        // Both add "x".
        r1.add("x");
        r2.add("x");
        // r1 removes "x" (only sees its own tag).
        r1.remove("x");
        // Merge: r2's add should survive because r1 only removed tags it observed.
        r1.merge(&r2);
        assert!(r1.contains("x")); // r2's concurrent add survives.
    }

    #[test]
    fn orset_readd_after_remove() {
        let mut s = ORSet::new("r1");
        s.add("a");
        s.remove("a");
        s.add("a"); // Re-add creates a new tag.
        assert!(s.contains("a"));
    }

    #[test]
    fn orset_tag_count() {
        let mut s = ORSet::new("r1");
        s.add("a");
        s.add("a");
        assert_eq!(s.tag_count("a"), 2);
    }

    #[test]
    fn orset_merge_two_replicas() {
        let mut r1 = ORSet::new("r1");
        let mut r2 = ORSet::new("r2");
        r1.add("a");
        r2.add("b");
        r1.merge(&r2);
        assert!(r1.contains("a"));
        assert!(r1.contains("b"));
        assert_eq!(r1.len(), 2);
    }

    #[test]
    fn orset_elements_sorted() {
        let mut s = ORSet::new("r1");
        s.add("c");
        s.add("a");
        s.add("b");
        let elems = s.elements();
        assert_eq!(elems, vec!["a", "b", "c"]);
    }

    // ── LWW-Element-Set tests ──

    #[test]
    fn lww_add_and_contains() {
        let mut s = LwwElementSet::new();
        s.add("a", 1.0);
        assert!(s.contains("a"));
    }

    #[test]
    fn lww_remove_wins_later_timestamp() {
        let mut s = LwwElementSet::new();
        s.add("a", 1.0);
        s.remove("a", 2.0);
        assert!(!s.contains("a"));
    }

    #[test]
    fn lww_add_wins_later_timestamp() {
        let mut s = LwwElementSet::new();
        s.remove("a", 1.0);
        s.add("a", 2.0);
        assert!(s.contains("a"));
    }

    #[test]
    fn lww_add_bias_on_tie() {
        let mut s = LwwElementSet::new();
        s.add("a", 1.0);
        s.remove("a", 1.0);
        assert!(s.contains("a")); // Add bias: add wins ties.
    }

    #[test]
    fn lww_remove_bias_on_tie() {
        let mut s = LwwElementSet::with_remove_bias();
        s.add("a", 1.0);
        s.remove("a", 1.0);
        assert!(!s.contains("a")); // Remove bias: remove wins ties.
    }

    #[test]
    fn lww_merge() {
        let mut s1 = LwwElementSet::new();
        let mut s2 = LwwElementSet::new();
        s1.add("a", 1.0);
        s2.remove("a", 2.0);
        s1.merge(&s2);
        assert!(!s1.contains("a")); // Remove at t=2 wins over add at t=1.
    }

    #[test]
    fn lww_merge_add_wins_with_higher_ts() {
        let mut s1 = LwwElementSet::new();
        let mut s2 = LwwElementSet::new();
        s1.remove("a", 1.0);
        s2.add("a", 3.0);
        s1.merge(&s2);
        assert!(s1.contains("a"));
    }

    #[test]
    fn lww_elements_sorted() {
        let mut s = LwwElementSet::new();
        s.add("c", 1.0);
        s.add("a", 2.0);
        s.add("b", 3.0);
        let elems = s.elements();
        assert_eq!(elems, vec!["a", "b", "c"]);
    }

    #[test]
    fn lww_timestamps() {
        let mut s = LwwElementSet::new();
        s.add("x", 5.0);
        s.remove("x", 3.0);
        assert_eq!(s.add_timestamp("x"), Some(5.0));
        assert_eq!(s.remove_timestamp("x"), Some(3.0));
    }

    #[test]
    fn lww_len() {
        let mut s = LwwElementSet::new();
        s.add("a", 1.0);
        s.add("b", 2.0);
        s.remove("a", 3.0);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn gset_default() {
        let s = GSet::default();
        assert!(s.is_empty());
    }
}
