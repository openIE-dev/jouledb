//! Data sharding — hash-based sharding, range-based sharding, shard assignment,
//! rebalancing (minimal data movement), shard routing, shard statistics,
//! virtual shards, shard splitting.

use std::collections::HashMap;

// ── Shard Info ───────────────────────────────────────────────────────────────

/// Metadata for a single shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardInfo {
    /// Shard identifier.
    pub shard_id: u32,
    /// Node(s) assigned to this shard.
    pub assigned_nodes: Vec<String>,
    /// Number of keys in this shard.
    pub key_count: usize,
    /// Approximate data size in bytes.
    pub data_bytes: usize,
    /// Whether the shard is active.
    pub active: bool,
}

// ── Shard Statistics ─────────────────────────────────────────────────────────

/// Statistics across all shards.
#[derive(Debug, Clone)]
pub struct ShardStats {
    /// Total number of shards.
    pub shard_count: usize,
    /// Total number of keys across all shards.
    pub total_keys: usize,
    /// Total data size in bytes.
    pub total_bytes: usize,
    /// Number of nodes.
    pub node_count: usize,
    /// Average keys per shard.
    pub avg_keys_per_shard: f64,
    /// Max keys in any shard.
    pub max_keys: usize,
    /// Min keys in any shard.
    pub min_keys: usize,
    /// Standard deviation of key counts.
    pub key_stddev: f64,
}

// ── Hash-Based Sharding ──────────────────────────────────────────────────────

/// A simple hash function (FNV-1a inspired) for deterministic key routing.
fn hash_key(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Hash-based shard router. Keys are assigned to shards via hash modulo.
#[derive(Debug, Clone)]
pub struct HashSharding {
    /// Number of virtual shards.
    num_shards: u32,
    /// Assignment: shard_id -> node_id.
    shard_to_node: HashMap<u32, String>,
    /// Per-shard key count tracking.
    shard_key_counts: HashMap<u32, usize>,
    /// Per-shard byte size tracking.
    shard_byte_sizes: HashMap<u32, usize>,
}

impl HashSharding {
    /// Create a new hash-based sharding scheme with the given number of virtual shards.
    pub fn new(num_shards: u32) -> Self {
        Self {
            num_shards,
            shard_to_node: HashMap::new(),
            shard_key_counts: HashMap::new(),
            shard_byte_sizes: HashMap::new(),
        }
    }

    /// Get the number of virtual shards.
    pub fn num_shards(&self) -> u32 {
        self.num_shards
    }

    /// Route a key to a shard id.
    pub fn route(&self, key: &str) -> u32 {
        (hash_key(key) % self.num_shards as u64) as u32
    }

    /// Route a key to a node. Returns None if the shard is unassigned.
    pub fn route_to_node(&self, key: &str) -> Option<&str> {
        let shard = self.route(key);
        self.shard_to_node.get(&shard).map(|s| s.as_str())
    }

    /// Assign a shard to a node.
    pub fn assign(&mut self, shard_id: u32, node_id: &str) {
        self.shard_to_node.insert(shard_id, node_id.to_string());
    }

    /// Assign all shards evenly across a set of nodes.
    pub fn assign_evenly(&mut self, nodes: &[&str]) {
        if nodes.is_empty() {
            return;
        }
        for shard_id in 0..self.num_shards {
            let node_idx = shard_id as usize % nodes.len();
            self.shard_to_node.insert(shard_id, nodes[node_idx].to_string());
        }
    }

    /// Record a key insertion for statistics tracking.
    pub fn record_key(&mut self, key: &str, size_bytes: usize) {
        let shard = self.route(key);
        *self.shard_key_counts.entry(shard).or_insert(0) += 1;
        *self.shard_byte_sizes.entry(shard).or_insert(0) += size_bytes;
    }

    /// Get the node assignment for a specific shard.
    pub fn node_for_shard(&self, shard_id: u32) -> Option<&str> {
        self.shard_to_node.get(&shard_id).map(|s| s.as_str())
    }

    /// Get shards assigned to a specific node (sorted).
    pub fn shards_for_node(&self, node_id: &str) -> Vec<u32> {
        let mut shards: Vec<u32> = self.shard_to_node
            .iter()
            .filter(|(_, n)| n.as_str() == node_id)
            .map(|(s, _)| *s)
            .collect();
        shards.sort();
        shards
    }

    /// Rebalance shards across nodes with minimal movement.
    /// Returns a list of (shard_id, old_node, new_node) moves.
    pub fn rebalance(&mut self, nodes: &[&str]) -> Vec<(u32, String, String)> {
        if nodes.is_empty() {
            return Vec::new();
        }
        let target_per_node = self.num_shards as usize / nodes.len();
        let remainder = self.num_shards as usize % nodes.len();

        // Build node -> desired count.
        let mut desired: HashMap<&str, usize> = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            let count = target_per_node + if i < remainder { 1 } else { 0 };
            desired.insert(node, count);
        }

        // Count current assignments per node.
        let mut current_counts: HashMap<&str, usize> = HashMap::new();
        for node in nodes {
            current_counts.insert(node, 0);
        }
        for (_, node) in &self.shard_to_node {
            if let Some(c) = current_counts.get_mut(node.as_str()) {
                *c += 1;
            }
        }

        // Find overloaded shards to move.
        let mut moves = Vec::new();
        let mut shards_to_move: Vec<(u32, String)> = Vec::new();

        // Collect shards from overloaded nodes or shards assigned to removed nodes.
        for shard_id in 0..self.num_shards {
            let current_node = self.shard_to_node.get(&shard_id).cloned();
            match &current_node {
                Some(node) if !nodes.contains(&node.as_str()) => {
                    // Node was removed — shard must move.
                    shards_to_move.push((shard_id, node.clone()));
                }
                _ => {}
            }
        }

        // For nodes with too many shards, shed extras.
        for node in nodes {
            let current = current_counts.get(node).copied().unwrap_or(0);
            let want = desired.get(node).copied().unwrap_or(0);
            if current > want {
                let excess = current - want;
                let mut count = 0;
                for shard_id in 0..self.num_shards {
                    if count >= excess {
                        break;
                    }
                    if let Some(n) = self.shard_to_node.get(&shard_id) {
                        if n.as_str() == *node {
                            let already_moving = shards_to_move.iter().any(|(s, _)| *s == shard_id);
                            if !already_moving {
                                shards_to_move.push((shard_id, node.to_string()));
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Assign shards_to_move to underloaded nodes.
        // Recount after marking moves.
        let mut assigned: HashMap<&str, usize> = HashMap::new();
        for node in nodes {
            assigned.insert(node, 0);
        }
        for shard_id in 0..self.num_shards {
            let is_moving = shards_to_move.iter().any(|(s, _)| *s == shard_id);
            if !is_moving {
                if let Some(n) = self.shard_to_node.get(&shard_id) {
                    if let Some(c) = assigned.get_mut(n.as_str()) {
                        *c += 1;
                    }
                }
            }
        }

        for (shard_id, old_node) in shards_to_move {
            // Find the node with the most remaining capacity.
            let mut best_node: Option<&str> = None;
            let mut best_gap: usize = 0;
            for node in nodes {
                let current = assigned.get(node).copied().unwrap_or(0);
                let want = desired.get(node).copied().unwrap_or(0);
                if current < want {
                    let gap = want - current;
                    if gap > best_gap {
                        best_gap = gap;
                        best_node = Some(node);
                    }
                }
            }
            if let Some(new_node) = best_node {
                moves.push((shard_id, old_node, new_node.to_string()));
                self.shard_to_node.insert(shard_id, new_node.to_string());
                *assigned.get_mut(new_node).unwrap() += 1;
            }
        }

        // Assign any unassigned shards.
        for shard_id in 0..self.num_shards {
            if !self.shard_to_node.contains_key(&shard_id) {
                let mut best_node: Option<&str> = None;
                let mut best_gap: usize = 0;
                for node in nodes {
                    let current = assigned.get(node).copied().unwrap_or(0);
                    let want = desired.get(node).copied().unwrap_or(0);
                    if current < want {
                        let gap = want - current;
                        if gap > best_gap {
                            best_gap = gap;
                            best_node = Some(node);
                        }
                    }
                }
                if let Some(new_node) = best_node {
                    self.shard_to_node.insert(shard_id, new_node.to_string());
                    *assigned.get_mut(new_node).unwrap() += 1;
                }
            }
        }

        moves
    }

    /// Get statistics across all shards.
    pub fn stats(&self) -> ShardStats {
        let mut total_keys = 0usize;
        let mut total_bytes = 0usize;
        let mut key_counts = Vec::new();
        for shard_id in 0..self.num_shards {
            let kc = self.shard_key_counts.get(&shard_id).copied().unwrap_or(0);
            total_keys += kc;
            total_bytes += self.shard_byte_sizes.get(&shard_id).copied().unwrap_or(0);
            key_counts.push(kc);
        }
        let shard_count = self.num_shards as usize;
        let avg = if shard_count > 0 { total_keys as f64 / shard_count as f64 } else { 0.0 };
        let max_keys = key_counts.iter().copied().max().unwrap_or(0);
        let min_keys = key_counts.iter().copied().min().unwrap_or(0);
        let variance = if shard_count > 0 {
            key_counts.iter().map(|k| {
                let diff = *k as f64 - avg;
                diff * diff
            }).sum::<f64>() / shard_count as f64
        } else {
            0.0
        };
        let nodes: std::collections::HashSet<&str> = self.shard_to_node.values().map(|s| s.as_str()).collect();
        ShardStats {
            shard_count,
            total_keys,
            total_bytes,
            node_count: nodes.len(),
            avg_keys_per_shard: avg,
            max_keys,
            min_keys,
            key_stddev: variance.sqrt(),
        }
    }
}

// ── Range-Based Sharding ─────────────────────────────────────────────────────

/// A range boundary for range-based sharding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeShard {
    /// Shard identifier.
    pub shard_id: u32,
    /// Inclusive lower bound of the key range.
    pub lower: String,
    /// Exclusive upper bound (empty string means unbounded).
    pub upper: String,
    /// Assigned node.
    pub node_id: String,
    /// Key count.
    pub key_count: usize,
}

/// Range-based shard router. Keys are routed by lexicographic range.
#[derive(Debug, Clone)]
pub struct RangeSharding {
    /// Sorted ranges.
    ranges: Vec<RangeShard>,
    /// Next shard id.
    next_shard_id: u32,
}

impl RangeSharding {
    /// Create a new range-based sharding scheme with a single shard covering all keys.
    pub fn new(node_id: &str) -> Self {
        Self {
            ranges: vec![RangeShard {
                shard_id: 0,
                lower: String::new(),
                upper: String::new(),
                node_id: node_id.to_string(),
                key_count: 0,
            }],
            next_shard_id: 1,
        }
    }

    /// Route a key to its shard id.
    pub fn route(&self, key: &str) -> u32 {
        for range in &self.ranges {
            let in_lower = key >= range.lower.as_str();
            let in_upper = range.upper.is_empty() || key < range.upper.as_str();
            if in_lower && in_upper {
                return range.shard_id;
            }
        }
        // Fallback to last shard.
        self.ranges.last().map(|r| r.shard_id).unwrap_or(0)
    }

    /// Route a key to its node.
    pub fn route_to_node(&self, key: &str) -> &str {
        for range in &self.ranges {
            let in_lower = key >= range.lower.as_str();
            let in_upper = range.upper.is_empty() || key < range.upper.as_str();
            if in_lower && in_upper {
                return &range.node_id;
            }
        }
        &self.ranges.last().unwrap().node_id
    }

    /// Record a key insertion.
    pub fn record_key(&mut self, key: &str) {
        let shard_id = self.route(key);
        if let Some(range) = self.ranges.iter_mut().find(|r| r.shard_id == shard_id) {
            range.key_count += 1;
        }
    }

    /// Split a shard at the given key boundary. The existing shard keeps
    /// [lower, split_key) and a new shard gets [split_key, upper).
    /// Returns the new shard id, or None if the split key is not in range.
    pub fn split(&mut self, shard_id: u32, split_key: &str, new_node: &str) -> Option<u32> {
        let idx = self.ranges.iter().position(|r| r.shard_id == shard_id)?;
        let range = &self.ranges[idx];
        let in_lower = split_key > range.lower.as_str();
        let in_upper = range.upper.is_empty() || split_key < range.upper.as_str();
        if !in_lower || !in_upper {
            return None;
        }
        let new_id = self.next_shard_id;
        self.next_shard_id += 1;
        let new_range = RangeShard {
            shard_id: new_id,
            lower: split_key.to_string(),
            upper: self.ranges[idx].upper.clone(),
            node_id: new_node.to_string(),
            key_count: 0,
        };
        self.ranges[idx].upper = split_key.to_string();
        // Approximate: split key count in half.
        let half = self.ranges[idx].key_count / 2;
        self.ranges[idx].key_count -= half;
        let insert_pos = idx + 1;
        let mut new_range_with_count = new_range;
        new_range_with_count.key_count = half;
        self.ranges.insert(insert_pos, new_range_with_count);
        Some(new_id)
    }

    /// Get the number of shards.
    pub fn shard_count(&self) -> usize {
        self.ranges.len()
    }

    /// Get all ranges.
    pub fn ranges(&self) -> &[RangeShard] {
        &self.ranges
    }

    /// Get a specific shard range.
    pub fn get_shard(&self, shard_id: u32) -> Option<&RangeShard> {
        self.ranges.iter().find(|r| r.shard_id == shard_id)
    }

    /// Reassign a shard to a different node.
    pub fn reassign(&mut self, shard_id: u32, new_node: &str) -> bool {
        if let Some(range) = self.ranges.iter_mut().find(|r| r.shard_id == shard_id) {
            range.node_id = new_node.to_string();
            true
        } else {
            false
        }
    }

    /// Get shards assigned to a node (sorted by shard_id).
    pub fn shards_for_node(&self, node_id: &str) -> Vec<u32> {
        let mut shards: Vec<u32> = self.ranges
            .iter()
            .filter(|r| r.node_id == node_id)
            .map(|r| r.shard_id)
            .collect();
        shards.sort();
        shards
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Hash sharding tests ──

    #[test]
    fn hash_sharding_route_deterministic() {
        let s = HashSharding::new(8);
        let shard1 = s.route("key_a");
        let shard2 = s.route("key_a");
        assert_eq!(shard1, shard2);
    }

    #[test]
    fn hash_sharding_route_range() {
        let s = HashSharding::new(8);
        let shard = s.route("test");
        assert!(shard < 8);
    }

    #[test]
    fn hash_sharding_assign_and_route() {
        let mut s = HashSharding::new(4);
        s.assign(0, "node_a");
        s.assign(1, "node_b");
        s.assign(2, "node_a");
        s.assign(3, "node_b");
        // Every key should route to a node.
        for key in &["k1", "k2", "k3", "k4"] {
            assert!(s.route_to_node(key).is_some());
        }
    }

    #[test]
    fn hash_sharding_assign_evenly() {
        let mut s = HashSharding::new(8);
        s.assign_evenly(&["n1", "n2"]);
        let n1_shards = s.shards_for_node("n1");
        let n2_shards = s.shards_for_node("n2");
        assert_eq!(n1_shards.len(), 4);
        assert_eq!(n2_shards.len(), 4);
    }

    #[test]
    fn hash_sharding_record_key() {
        let mut s = HashSharding::new(4);
        s.record_key("key1", 100);
        s.record_key("key2", 200);
        let stats = s.stats();
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.total_bytes, 300);
    }

    #[test]
    fn hash_sharding_rebalance_add_node() {
        let mut s = HashSharding::new(6);
        s.assign_evenly(&["n1", "n2"]);
        // n1 has 3, n2 has 3. Now add n3.
        let moves = s.rebalance(&["n1", "n2", "n3"]);
        assert!(!moves.is_empty());
        let n3_shards = s.shards_for_node("n3");
        assert_eq!(n3_shards.len(), 2); // 6/3 = 2 per node.
    }

    #[test]
    fn hash_sharding_rebalance_remove_node() {
        let mut s = HashSharding::new(6);
        s.assign_evenly(&["n1", "n2", "n3"]);
        // Remove n3.
        let moves = s.rebalance(&["n1", "n2"]);
        assert!(!moves.is_empty());
        let n3_shards = s.shards_for_node("n3");
        assert_eq!(n3_shards.len(), 0);
        let n1_shards = s.shards_for_node("n1");
        let n2_shards = s.shards_for_node("n2");
        assert_eq!(n1_shards.len() + n2_shards.len(), 6);
    }

    #[test]
    fn hash_sharding_stats_empty() {
        let s = HashSharding::new(4);
        let stats = s.stats();
        assert_eq!(stats.shard_count, 4);
        assert_eq!(stats.total_keys, 0);
        assert_eq!(stats.min_keys, 0);
    }

    #[test]
    fn hash_sharding_unassigned_route() {
        let s = HashSharding::new(4);
        assert!(s.route_to_node("key").is_none());
    }

    // ── Range sharding tests ──

    #[test]
    fn range_sharding_single_shard() {
        let rs = RangeSharding::new("n1");
        assert_eq!(rs.shard_count(), 1);
        assert_eq!(rs.route("anything"), 0);
        assert_eq!(rs.route_to_node("anything"), "n1");
    }

    #[test]
    fn range_sharding_split() {
        let mut rs = RangeSharding::new("n1");
        let new_id = rs.split(0, "m", "n2").unwrap();
        assert_eq!(rs.shard_count(), 2);
        // Keys before "m" go to shard 0, keys >= "m" go to new shard.
        assert_eq!(rs.route("abc"), 0);
        assert_eq!(rs.route_to_node("abc"), "n1");
        assert_eq!(rs.route("xyz"), new_id);
        assert_eq!(rs.route_to_node("xyz"), "n2");
    }

    #[test]
    fn range_sharding_multiple_splits() {
        let mut rs = RangeSharding::new("n1");
        rs.split(0, "g", "n2");
        // Shard 0: [, g), shard 1: [g, )
        // Split shard 1 at "m".
        rs.split(1, "m", "n3");
        assert_eq!(rs.shard_count(), 3);
        assert_eq!(rs.route_to_node("abc"), "n1");
        assert_eq!(rs.route_to_node("hello"), "n2");
        assert_eq!(rs.route_to_node("xyz"), "n3");
    }

    #[test]
    fn range_sharding_invalid_split() {
        let mut rs = RangeSharding::new("n1");
        rs.split(0, "m", "n2");
        // Try to split shard 0 at "z" which is beyond its upper bound "m".
        let result = rs.split(0, "z", "n3");
        assert!(result.is_none());
    }

    #[test]
    fn range_sharding_record_key() {
        let mut rs = RangeSharding::new("n1");
        rs.record_key("hello");
        rs.record_key("world");
        let shard = rs.get_shard(0).unwrap();
        assert_eq!(shard.key_count, 2);
    }

    #[test]
    fn range_sharding_reassign() {
        let mut rs = RangeSharding::new("n1");
        assert!(rs.reassign(0, "n2"));
        assert_eq!(rs.route_to_node("key"), "n2");
    }

    #[test]
    fn range_sharding_reassign_nonexistent() {
        let mut rs = RangeSharding::new("n1");
        assert!(!rs.reassign(99, "n2"));
    }

    #[test]
    fn range_sharding_shards_for_node() {
        let mut rs = RangeSharding::new("n1");
        rs.split(0, "m", "n2");
        let n1_shards = rs.shards_for_node("n1");
        let n2_shards = rs.shards_for_node("n2");
        assert_eq!(n1_shards.len(), 1);
        assert_eq!(n2_shards.len(), 1);
    }

    #[test]
    fn hash_key_different_for_different_inputs() {
        let h1 = hash_key("alpha");
        let h2 = hash_key("beta");
        assert_ne!(h1, h2);
    }

    #[test]
    fn range_sharding_boundary_key() {
        let mut rs = RangeSharding::new("n1");
        rs.split(0, "m", "n2");
        // "m" itself should go to the new shard [m, ).
        assert_eq!(rs.route_to_node("m"), "n2");
        // Key just before "m" goes to shard 0.
        assert_eq!(rs.route_to_node("lzzz"), "n1");
    }
}
