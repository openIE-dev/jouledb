//! Consistent hashing — hash ring with virtual nodes, key-to-node mapping,
//! ring rebalancing statistics, jump consistent hash, and bounded-load hashing.

use std::collections::{BTreeMap, HashMap};

// ── Hash Function ────────────────────────────────────────────────────────────

/// FNV-1a hash for consistent, deterministic hashing, with avalanche
/// finalization for uniform distribution across the 64-bit key space.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Avalanche / finalization mix (splitmix64-style) to spread bits uniformly.
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    hash
}

/// Hash a string key.
fn hash_key(key: &str) -> u64 {
    fnv1a_hash(key.as_bytes())
}

/// Hash a node + virtual node index to produce a ring position.
fn hash_vnode(node_id: &str, vnode_idx: usize) -> u64 {
    let combined = format!("{}#{}", node_id, vnode_idx);
    hash_key(&combined)
}

// ── Hash Ring ────────────────────────────────────────────────────────────────

/// A consistent hash ring with virtual nodes.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted map of ring positions to node ids.
    ring: BTreeMap<u64, String>,
    /// Number of virtual nodes per physical node.
    vnodes_per_node: usize,
    /// Set of physical node ids.
    nodes: Vec<String>,
}

impl HashRing {
    /// Create a new empty hash ring.
    pub fn new(vnodes_per_node: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            vnodes_per_node: vnodes_per_node.max(1),
            nodes: Vec::new(),
        }
    }

    /// Number of physical nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of virtual nodes (points on the ring).
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }

    /// Get all physical node ids.
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// Check if a node is in the ring.
    pub fn has_node(&self, node_id: &str) -> bool {
        self.nodes.iter().any(|n| n == node_id)
    }

    /// Add a node to the ring with virtual nodes.
    pub fn add_node(&mut self, node_id: &str) {
        if self.has_node(node_id) {
            return;
        }
        self.nodes.push(node_id.to_string());
        for i in 0..self.vnodes_per_node {
            let pos = hash_vnode(node_id, i);
            self.ring.insert(pos, node_id.to_string());
        }
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node_id: &str) {
        if !self.has_node(node_id) {
            return;
        }
        self.nodes.retain(|n| n != node_id);
        for i in 0..self.vnodes_per_node {
            let pos = hash_vnode(node_id, i);
            self.ring.remove(&pos);
        }
    }

    /// Get the node responsible for a key.
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let h = hash_key(key);
        // Find the first ring position >= h
        let node = self
            .ring
            .range(h..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, node)| node.as_str());
        node
    }

    /// Get N distinct nodes responsible for a key (for replication).
    pub fn get_nodes(&self, key: &str, n: usize) -> Vec<String> {
        if self.ring.is_empty() {
            return Vec::new();
        }
        let h = hash_key(key);
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Walk clockwise from the hash position
        let iter = self.ring.range(h..).chain(self.ring.iter());
        for (_, node) in iter {
            if seen.insert(node.clone()) {
                result.push(node.clone());
                if result.len() >= n || result.len() >= self.nodes.len() {
                    break;
                }
            }
        }

        result
    }

    /// Compute rebalancing statistics: how many keys would move if a node is added.
    /// `sample_keys` are keys to check.
    pub fn rebalance_stats(&self, new_node: &str, sample_keys: &[String]) -> RebalanceStats {
        let mut moved = 0;
        let mut before: HashMap<String, usize> = HashMap::new();
        let mut after: HashMap<String, usize> = HashMap::new();

        // Map keys in current ring
        for key in sample_keys {
            if let Some(node) = self.get_node(key) {
                *before.entry(node.to_string()).or_insert(0) += 1;
            }
        }

        // Create a ring with the new node
        let mut new_ring = self.clone();
        new_ring.add_node(new_node);

        for key in sample_keys {
            let old = self.get_node(key).unwrap_or("");
            let new = new_ring.get_node(key).unwrap_or("");
            if let Some(node) = new_ring.get_node(key) {
                *after.entry(node.to_string()).or_insert(0) += 1;
            }
            if old != new {
                moved += 1;
            }
        }

        RebalanceStats {
            total_keys: sample_keys.len(),
            keys_moved: moved,
            distribution_before: before,
            distribution_after: after,
        }
    }

    /// Get the distribution of keys across nodes.
    pub fn key_distribution(&self, keys: &[String]) -> HashMap<String, usize> {
        let mut dist: HashMap<String, usize> = HashMap::new();
        for key in keys {
            if let Some(node) = self.get_node(key) {
                *dist.entry(node.to_string()).or_insert(0) += 1;
            }
        }
        dist
    }
}

/// Statistics about rebalancing when adding a node.
#[derive(Debug, Clone)]
pub struct RebalanceStats {
    /// Total number of keys checked.
    pub total_keys: usize,
    /// Number of keys that would move to a different node.
    pub keys_moved: usize,
    /// Key distribution before adding the node.
    pub distribution_before: HashMap<String, usize>,
    /// Key distribution after adding the node.
    pub distribution_after: HashMap<String, usize>,
}

impl RebalanceStats {
    /// Fraction of keys that moved.
    pub fn move_fraction(&self) -> f64 {
        if self.total_keys == 0 {
            0.0
        } else {
            self.keys_moved as f64 / self.total_keys as f64
        }
    }
}

// ── Jump Consistent Hash ─────────────────────────────────────────────────────

/// Jump consistent hash — maps a key to one of `num_buckets` buckets.
/// Guarantees minimal disruption when buckets change.
pub fn jump_consistent_hash(key: u64, num_buckets: u32) -> u32 {
    if num_buckets == 0 {
        return 0;
    }
    let mut b: i64 = -1;
    let mut j: i64 = 0;
    let mut k = key;

    while j < num_buckets as i64 {
        b = j;
        k = k.wrapping_mul(2862933555777941757).wrapping_add(1);
        let shifted = ((b + 1) as f64) * (1u64 << 31) as f64;
        let rand = ((k >> 33) + 1) as f64;
        j = (shifted / rand) as i64;
    }

    b as u32
}

/// Jump hash with a string key.
pub fn jump_hash_string(key: &str, num_buckets: u32) -> u32 {
    jump_consistent_hash(hash_key(key), num_buckets)
}

// ── Bounded Load Consistent Hash ─────────────────────────────────────────────

/// Consistent hash ring with bounded load — no node exceeds a load factor.
#[derive(Debug, Clone)]
pub struct BoundedLoadRing {
    ring: HashRing,
    /// Current load on each node.
    loads: HashMap<String, usize>,
    /// Maximum load factor (multiplied by average load).
    load_factor: f64,
    /// Total assigned keys.
    total_assigned: usize,
}

impl BoundedLoadRing {
    /// Create a new bounded-load ring.
    pub fn new(vnodes_per_node: usize, load_factor: f64) -> Self {
        Self {
            ring: HashRing::new(vnodes_per_node),
            loads: HashMap::new(),
            load_factor: load_factor.max(1.0),
            total_assigned: 0,
        }
    }

    /// Add a node.
    pub fn add_node(&mut self, node_id: &str) {
        self.ring.add_node(node_id);
        self.loads.entry(node_id.to_string()).or_insert(0);
    }

    /// Remove a node.
    pub fn remove_node(&mut self, node_id: &str) {
        let freed = self.loads.remove(node_id).unwrap_or(0);
        self.total_assigned -= freed;
        self.ring.remove_node(node_id);
    }

    /// Maximum allowed load per node.
    pub fn max_load(&self) -> usize {
        let n = self.ring.node_count();
        if n == 0 {
            return 0;
        }
        let avg = (self.total_assigned as f64) / (n as f64);
        (avg * self.load_factor).ceil() as usize + 1
    }

    /// Assign a key to a node, respecting the load bound.
    pub fn assign(&mut self, key: &str) -> Option<String> {
        if self.ring.node_count() == 0 {
            return None;
        }

        let candidates = self.ring.get_nodes(key, self.ring.node_count());
        let max = self.max_load();

        for candidate in &candidates {
            let load = self.loads.get(candidate).copied().unwrap_or(0);
            if load < max {
                *self.loads.entry(candidate.clone()).or_insert(0) += 1;
                self.total_assigned += 1;
                return Some(candidate.clone());
            }
        }

        // Fallback: assign to least loaded
        let mut best: Option<(String, usize)> = None;
        for (node, &load) in &self.loads {
            match &best {
                None => best = Some((node.clone(), load)),
                Some((_, best_load)) => {
                    if load < *best_load {
                        best = Some((node.clone(), load));
                    }
                }
            }
        }

        if let Some((node, _)) = best {
            *self.loads.entry(node.clone()).or_insert(0) += 1;
            self.total_assigned += 1;
            Some(node)
        } else {
            None
        }
    }

    /// Release a key from its assigned node.
    pub fn release(&mut self, node_id: &str) {
        if let Some(load) = self.loads.get_mut(node_id) {
            if *load > 0 {
                *load -= 1;
                self.total_assigned -= 1;
            }
        }
    }

    /// Get current load of a node.
    pub fn load(&self, node_id: &str) -> usize {
        self.loads.get(node_id).copied().unwrap_or(0)
    }

    /// Total assigned keys.
    pub fn total_assigned(&self) -> usize {
        self.total_assigned
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.ring.node_count()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_ring() {
        let ring = HashRing::new(10);
        assert_eq!(ring.node_count(), 0);
        assert!(ring.get_node("key1").is_none());
    }

    #[test]
    fn test_add_node() {
        let mut ring = HashRing::new(10);
        ring.add_node("node-a");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 10);
        assert!(ring.has_node("node-a"));
    }

    #[test]
    fn test_add_duplicate() {
        let mut ring = HashRing::new(5);
        ring.add_node("node-a");
        ring.add_node("node-a");
        assert_eq!(ring.node_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut ring = HashRing::new(10);
        ring.add_node("node-a");
        ring.add_node("node-b");
        ring.remove_node("node-a");
        assert_eq!(ring.node_count(), 1);
        assert!(!ring.has_node("node-a"));
        assert_eq!(ring.vnode_count(), 10);
    }

    #[test]
    fn test_get_node_single() {
        let mut ring = HashRing::new(10);
        ring.add_node("node-a");
        assert_eq!(ring.get_node("key1"), Some("node-a"));
        assert_eq!(ring.get_node("key2"), Some("node-a"));
    }

    #[test]
    fn test_get_node_deterministic() {
        let mut ring = HashRing::new(100);
        ring.add_node("node-a");
        ring.add_node("node-b");
        ring.add_node("node-c");
        let n1 = ring.get_node("my-key").unwrap().to_string();
        let n2 = ring.get_node("my-key").unwrap().to_string();
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_get_nodes_replication() {
        let mut ring = HashRing::new(100);
        ring.add_node("node-a");
        ring.add_node("node-b");
        ring.add_node("node-c");
        let nodes = ring.get_nodes("key", 2);
        assert_eq!(nodes.len(), 2);
        assert_ne!(nodes[0], nodes[1]);
    }

    #[test]
    fn test_get_nodes_capped_at_node_count() {
        let mut ring = HashRing::new(10);
        ring.add_node("node-a");
        ring.add_node("node-b");
        let nodes = ring.get_nodes("key", 5);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_minimal_disruption() {
        let mut ring = HashRing::new(100);
        ring.add_node("node-a");
        ring.add_node("node-b");

        let keys: Vec<String> = (0..1000).map(|i| format!("key-{}", i)).collect();
        let before: Vec<String> = keys
            .iter()
            .map(|k| ring.get_node(k).unwrap().to_string())
            .collect();

        ring.add_node("node-c");
        let after: Vec<String> = keys
            .iter()
            .map(|k| ring.get_node(k).unwrap().to_string())
            .collect();

        let moved = before
            .iter()
            .zip(after.iter())
            .filter(|(a, b)| a != b)
            .count();
        // Should move roughly 1/3 of keys (one new node out of 3)
        assert!(moved < 600, "too many keys moved: {}", moved);
        assert!(moved > 100, "too few keys moved: {}", moved);
    }

    #[test]
    fn test_rebalance_stats() {
        let mut ring = HashRing::new(100);
        ring.add_node("node-a");
        ring.add_node("node-b");
        let keys: Vec<String> = (0..100).map(|i| format!("key-{}", i)).collect();
        let stats = ring.rebalance_stats("node-c", &keys);
        assert_eq!(stats.total_keys, 100);
        assert!(stats.keys_moved > 0);
        assert!(stats.move_fraction() > 0.0);
        assert!(stats.move_fraction() < 1.0);
    }

    #[test]
    fn test_key_distribution() {
        let mut ring = HashRing::new(100);
        ring.add_node("node-a");
        ring.add_node("node-b");
        let keys: Vec<String> = (0..100).map(|i| format!("key-{}", i)).collect();
        let dist = ring.key_distribution(&keys);
        let total: usize = dist.values().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_jump_consistent_hash() {
        let bucket = jump_consistent_hash(42, 10);
        assert!(bucket < 10);
        // Deterministic
        assert_eq!(jump_consistent_hash(42, 10), bucket);
    }

    #[test]
    fn test_jump_hash_zero_buckets() {
        assert_eq!(jump_consistent_hash(42, 0), 0);
    }

    #[test]
    fn test_jump_hash_one_bucket() {
        assert_eq!(jump_consistent_hash(42, 1), 0);
    }

    #[test]
    fn test_jump_hash_string_fn() {
        let b = jump_hash_string("my-key", 5);
        assert!(b < 5);
    }

    #[test]
    fn test_bounded_load_basic() {
        let mut blr = BoundedLoadRing::new(10, 1.5);
        blr.add_node("node-a");
        blr.add_node("node-b");
        let assigned = blr.assign("key1").unwrap();
        assert!(assigned == "node-a" || assigned == "node-b");
        assert_eq!(blr.total_assigned(), 1);
    }

    #[test]
    fn test_bounded_load_prevents_overload() {
        let mut blr = BoundedLoadRing::new(10, 1.25);
        blr.add_node("node-a");
        blr.add_node("node-b");
        // Assign many keys — both nodes should get some
        for i in 0..100 {
            blr.assign(&format!("key-{}", i));
        }
        let load_a = blr.load("node-a");
        let load_b = blr.load("node-b");
        // Neither should have all 100
        assert!(load_a > 0);
        assert!(load_b > 0);
        assert_eq!(load_a + load_b, 100);
    }

    #[test]
    fn test_bounded_load_release() {
        let mut blr = BoundedLoadRing::new(10, 1.5);
        blr.add_node("node-a");
        blr.assign("key1");
        assert_eq!(blr.load("node-a"), 1);
        blr.release("node-a");
        assert_eq!(blr.load("node-a"), 0);
    }

    #[test]
    fn test_bounded_load_remove_node() {
        let mut blr = BoundedLoadRing::new(10, 1.5);
        blr.add_node("node-a");
        blr.add_node("node-b");
        blr.assign("key1");
        blr.remove_node("node-a");
        assert_eq!(blr.node_count(), 1);
    }

    #[test]
    fn test_fnv_hash_deterministic() {
        assert_eq!(hash_key("hello"), hash_key("hello"));
        assert_ne!(hash_key("hello"), hash_key("world"));
    }
}
