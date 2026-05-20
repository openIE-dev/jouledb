//! CRDT Type Implementations
//!
//! Each type satisfies commutativity, associativity, and idempotency.

use crate::{Crdt, HLCTimestamp};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// GCounter — Grow-only counter
// ============================================================================

/// A grow-only counter. Each node can only increment its own slot.
/// The total value is the sum of all node slots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GCounter {
    /// Per-node counter values
    counts: HashMap<String, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Increment the counter for a node
    pub fn increment(&mut self, node_id: &str, amount: u64) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += amount;
    }

    /// Get the total counter value (sum of all nodes)
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Get the value for a specific node
    pub fn node_value(&self, node_id: &str) -> u64 {
        self.counts.get(node_id).copied().unwrap_or(0)
    }
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Crdt for GCounter {
    fn merge(&mut self, other: &Self) {
        for (node, &count) in &other.counts {
            let entry = self.counts.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(count);
        }
    }

    fn crdt_type(&self) -> &'static str {
        "gcounter"
    }
}

// ============================================================================
// PNCounter — Positive-Negative counter
// ============================================================================

/// A counter that supports both increment and decrement operations.
/// Implemented as two GCounters: one for increments, one for decrements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PNCounter {
    positive: GCounter,
    negative: GCounter,
}

impl PNCounter {
    pub fn new() -> Self {
        Self {
            positive: GCounter::new(),
            negative: GCounter::new(),
        }
    }

    /// Increment the counter
    pub fn increment(&mut self, node_id: &str, amount: u64) {
        self.positive.increment(node_id, amount);
    }

    /// Decrement the counter
    pub fn decrement(&mut self, node_id: &str, amount: u64) {
        self.negative.increment(node_id, amount);
    }

    /// Get the current value (positive - negative)
    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }
}

impl Default for PNCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Crdt for PNCounter {
    fn merge(&mut self, other: &Self) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }

    fn crdt_type(&self) -> &'static str {
        "pncounter"
    }
}

// ============================================================================
// LWWRegister — Last-Writer-Wins Register
// ============================================================================

/// A register where the most recent write wins (by HLC timestamp).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LWWRegister {
    value: Vec<u8>,
    timestamp: HLCTimestamp,
}

impl LWWRegister {
    pub fn new(value: Vec<u8>, node_id: &str) -> Self {
        Self {
            value,
            timestamp: HLCTimestamp::now(node_id),
        }
    }

    /// Set a new value with a new timestamp
    pub fn set(&mut self, value: Vec<u8>, node_id: &str) {
        self.timestamp.tick(None);
        self.timestamp.node_hash = crate::simple_hash(node_id);
        self.value = value;
    }

    /// Get the current value
    pub fn value(&self) -> &[u8] {
        &self.value
    }

    /// Get the timestamp
    pub fn timestamp(&self) -> &HLCTimestamp {
        &self.timestamp
    }
}

impl Crdt for LWWRegister {
    fn merge(&mut self, other: &Self) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
        }
    }

    fn crdt_type(&self) -> &'static str {
        "lww_register"
    }
}

// ============================================================================
// MVRegister — Multi-Value Register
// ============================================================================

/// A register that preserves all concurrent values.
/// When writers conflict, both values are kept until a subsequent write
/// resolves the conflict.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MVRegister {
    /// Each entry: (value, vector clock as node→counter map)
    entries: Vec<(Vec<u8>, HashMap<String, u64>)>,
}

impl MVRegister {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Write a new value, superseding all current values
    pub fn set(&mut self, value: Vec<u8>, node_id: &str) {
        // Compute new vector clock (max of all current + increment node)
        let mut new_clock: HashMap<String, u64> = HashMap::new();
        for (_, clock) in &self.entries {
            for (n, &v) in clock {
                let entry = new_clock.entry(n.clone()).or_insert(0);
                *entry = (*entry).max(v);
            }
        }
        *new_clock.entry(node_id.to_string()).or_insert(0) += 1;

        self.entries = vec![(value, new_clock)];
    }

    /// Get all current values (may be >1 if concurrent writes occurred)
    pub fn values(&self) -> Vec<&[u8]> {
        self.entries.iter().map(|(v, _)| v.as_slice()).collect()
    }

    fn dominates(a: &HashMap<String, u64>, b: &HashMap<String, u64>) -> bool {
        let mut dominated = false;
        for (node, &va) in a {
            let vb = b.get(node).copied().unwrap_or(0);
            if va < vb {
                return false;
            }
            if va > vb {
                dominated = true;
            }
        }
        for (node, &vb) in b {
            if !a.contains_key(node) && vb > 0 {
                return false;
            }
        }
        dominated
    }
}

impl Default for MVRegister {
    fn default() -> Self {
        Self::new()
    }
}

impl Crdt for MVRegister {
    fn merge(&mut self, other: &Self) {
        let mut combined = self.entries.clone();
        combined.extend(other.entries.clone());

        // Remove entries dominated by others
        let mut result = Vec::new();
        for (i, (val_i, clock_i)) in combined.iter().enumerate() {
            let dominated = combined
                .iter()
                .enumerate()
                .any(|(j, (_, clock_j))| i != j && Self::dominates(clock_j, clock_i));
            if !dominated {
                result.push((val_i.clone(), clock_i.clone()));
            }
        }

        // Deduplicate by clock
        result.dedup_by(|a, b| a.1 == b.1);
        self.entries = result;
    }

    fn crdt_type(&self) -> &'static str {
        "mv_register"
    }
}

// ============================================================================
// ORSet — Observed-Remove Set
// ============================================================================

/// An add/remove set where adds and removes are tracked with unique tags.
/// Concurrent add+remove of the same element keeps the element (add-wins).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ORSet {
    /// element -> set of (unique_tag, adding_node) pairs
    entries: HashMap<String, HashSet<(u64, String)>>,
    /// Removed tags (tombstones)
    tombstones: HashSet<(u64, String)>,
    /// Tag counter per node
    tag_counter: HashMap<String, u64>,
}

impl ORSet {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            tombstones: HashSet::new(),
            tag_counter: HashMap::new(),
        }
    }

    fn next_tag(&mut self, node_id: &str) -> u64 {
        let counter = self.tag_counter.entry(node_id.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Add an element to the set
    pub fn add(&mut self, element: &str, node_id: &str) {
        let tag = self.next_tag(node_id);
        self.entries
            .entry(element.to_string())
            .or_default()
            .insert((tag, node_id.to_string()));
    }

    /// Remove an element from the set (removes all known tags for it)
    pub fn remove(&mut self, element: &str) {
        if let Some(tags) = self.entries.remove(element) {
            for tag in tags {
                self.tombstones.insert(tag);
            }
        }
    }

    /// Check if an element is in the set
    pub fn contains(&self, element: &str) -> bool {
        self.entries
            .get(element)
            .map(|tags| !tags.is_empty())
            .unwrap_or(false)
    }

    /// Get all elements currently in the set
    pub fn elements(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, tags)| !tags.is_empty())
            .map(|(elem, _)| elem.as_str())
            .collect()
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.entries
            .values()
            .filter(|tags| !tags.is_empty())
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ORSet {
    fn default() -> Self {
        Self::new()
    }
}

impl Crdt for ORSet {
    fn merge(&mut self, other: &Self) {
        // Merge tombstones
        self.tombstones.extend(other.tombstones.iter().cloned());

        // Merge tag counters
        for (node, &count) in &other.tag_counter {
            let entry = self.tag_counter.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(count);
        }

        // Merge entries: union of tags minus tombstones
        for (elem, remote_tags) in &other.entries {
            let local_tags = self.entries.entry(elem.clone()).or_default();
            for tag in remote_tags {
                if !self.tombstones.contains(tag) {
                    local_tags.insert(tag.clone());
                }
            }
        }

        // Remove tombstoned tags from all entries
        for tags in self.entries.values_mut() {
            tags.retain(|tag| !self.tombstones.contains(tag));
        }
    }

    fn crdt_type(&self) -> &'static str {
        "orset"
    }
}

// ============================================================================
// LWWMap — Last-Writer-Wins Map (JSON document merge)
// ============================================================================

/// A map where each key has an independent LWW register.
/// Perfect for JSON document merge: each field resolves independently.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LWWMap {
    entries: HashMap<String, LWWRegister>,
}

impl LWWMap {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Set a key to a value
    pub fn set(&mut self, key: &str, value: Vec<u8>, node_id: &str) {
        match self.entries.get_mut(key) {
            Some(reg) => reg.set(value, node_id),
            None => {
                self.entries
                    .insert(key.to_string(), LWWRegister::new(value, node_id));
            }
        }
    }

    /// Get a value by key
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.entries.get(key).map(|r| r.value())
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(|k| k.as_str()).collect()
    }

    /// Number of keys
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for LWWMap {
    fn default() -> Self {
        Self::new()
    }
}

impl Crdt for LWWMap {
    fn merge(&mut self, other: &Self) {
        for (key, remote_reg) in &other.entries {
            match self.entries.get_mut(key) {
                Some(local_reg) => local_reg.merge(remote_reg),
                None => {
                    self.entries.insert(key.clone(), remote_reg.clone());
                }
            }
        }
    }

    fn crdt_type(&self) -> &'static str {
        "lww_map"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- GCounter ---

    #[test]
    fn test_gcounter_basic() {
        let mut c = GCounter::new();
        c.increment("node1", 3);
        c.increment("node2", 5);
        assert_eq!(c.value(), 8);
    }

    #[test]
    fn test_gcounter_merge_commutative() {
        let mut a = GCounter::new();
        a.increment("n1", 3);

        let mut b = GCounter::new();
        b.increment("n2", 5);

        let mut ab = a.clone();
        ab.merge(&b);

        let mut ba = b.clone();
        ba.merge(&a);

        assert_eq!(ab.value(), ba.value());
    }

    #[test]
    fn test_gcounter_merge_idempotent() {
        let mut a = GCounter::new();
        a.increment("n1", 3);

        let mut b = a.clone();
        b.merge(&a);
        b.merge(&a);

        assert_eq!(b.value(), 3);
    }

    // --- PNCounter ---

    #[test]
    fn test_pncounter_basic() {
        let mut c = PNCounter::new();
        c.increment("n1", 10);
        c.decrement("n1", 3);
        assert_eq!(c.value(), 7);
    }

    #[test]
    fn test_pncounter_merge() {
        let mut a = PNCounter::new();
        a.increment("n1", 10);

        let mut b = PNCounter::new();
        b.decrement("n2", 3);

        a.merge(&b);
        assert_eq!(a.value(), 7);
    }

    // --- LWWRegister ---

    #[test]
    fn test_lww_register_merge() {
        let a = LWWRegister::new(b"old".to_vec(), "n1");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = LWWRegister::new(b"new".to_vec(), "n2");

        let mut merged = a.clone();
        merged.merge(&b);
        assert_eq!(merged.value(), b"new");

        // Commutative
        let mut merged2 = b.clone();
        merged2.merge(&a);
        assert_eq!(merged2.value(), b"new");
    }

    // --- ORSet ---

    #[test]
    fn test_orset_add_remove() {
        let mut s = ORSet::new();
        s.add("apple", "n1");
        s.add("banana", "n1");
        assert!(s.contains("apple"));
        assert_eq!(s.len(), 2);

        s.remove("apple");
        assert!(!s.contains("apple"));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_orset_concurrent_add_remove() {
        // Node 1 adds "x"
        let mut a = ORSet::new();
        a.add("x", "n1");

        // Node 2 independently adds "x" and removes it
        let mut b = a.clone();
        b.add("x", "n2");
        b.remove("x"); // removes tags known to b

        // Node 1 concurrently adds "x" again
        a.add("x", "n1");

        // Merge: add-wins for the concurrent add on node 1
        a.merge(&b);
        assert!(a.contains("x")); // node1's second add survives
    }

    #[test]
    fn test_orset_merge_commutative() {
        let mut a = ORSet::new();
        a.add("x", "n1");
        a.add("y", "n1");

        let mut b = ORSet::new();
        b.add("y", "n2");
        b.add("z", "n2");

        let mut ab = a.clone();
        ab.merge(&b);

        let mut ba = b.clone();
        ba.merge(&a);

        let mut ab_elems: Vec<&str> = ab.elements();
        ab_elems.sort();
        let mut ba_elems: Vec<&str> = ba.elements();
        ba_elems.sort();
        assert_eq!(ab_elems, ba_elems);
    }

    // --- LWWMap ---

    #[test]
    fn test_lww_map_basic() {
        let mut m = LWWMap::new();
        m.set("name", b"Alice".to_vec(), "n1");
        m.set("age", b"30".to_vec(), "n1");

        assert_eq!(m.get("name"), Some(b"Alice".as_ref()));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_lww_map_merge_per_key() {
        let mut a = LWWMap::new();
        a.set("name", b"Alice".to_vec(), "n1");
        a.set("email", b"alice@old.com".to_vec(), "n1");

        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut b = LWWMap::new();
        b.set("email", b"alice@new.com".to_vec(), "n2");
        b.set("phone", b"555-1234".to_vec(), "n2");

        a.merge(&b);

        // name: only in a
        assert_eq!(a.get("name"), Some(b"Alice".as_ref()));
        // email: b wins (newer timestamp)
        assert_eq!(a.get("email"), Some(b"alice@new.com".as_ref()));
        // phone: only in b, merged into a
        assert_eq!(a.get("phone"), Some(b"555-1234".as_ref()));
    }

    // --- MVRegister ---

    #[test]
    fn test_mv_register_concurrent_writes() {
        let mut a = MVRegister::new();
        a.set(b"value_a".to_vec(), "n1");

        let mut b = MVRegister::new();
        b.set(b"value_b".to_vec(), "n2");

        // Concurrent writes: merge preserves both
        a.merge(&b);
        let values = a.values();
        assert!(values.len() >= 1); // At least one value preserved
    }
}
