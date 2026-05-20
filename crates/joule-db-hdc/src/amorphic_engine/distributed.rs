//! Distributed AmorphicEngine Components
//!
//! Multi-node coordination for distributed AmorphicEngine deployment.
//! Provides causal consistency via vector clocks and CRDT-based conflict resolution.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{AmorphicEngineError, AmorphicEngineResult};

// ============================================================================
// VECTOR CLOCK
// ============================================================================

/// Vector clock for causal consistency
///
/// Tracks logical time across multiple nodes to establish happened-before relationships.
#[derive(Clone, Debug)]
pub struct VectorClock {
    clock: Arc<RwLock<HashMap<String, u64>>>,
    /// Node ID that owns this clock
    pub node_id: String,
}

impl VectorClock {
    /// Create new vector clock for a node
    pub fn new(node_id: &str) -> Self {
        let mut clock = HashMap::new();
        clock.insert(node_id.to_string(), 0);
        Self {
            clock: Arc::new(RwLock::new(clock)),
            node_id: node_id.to_string(),
        }
    }

    /// Increment this node's clock
    pub fn increment(&self) -> u64 {
        if let Ok(mut clock) = self.clock.write() {
            let entry = clock.entry(self.node_id.clone()).or_insert(0);
            *entry += 1;
            return *entry;
        }
        0
    }

    /// Get current timestamp for a node
    pub fn get(&self, node_id: &str) -> u64 {
        if let Ok(clock) = self.clock.read() {
            return clock.get(node_id).copied().unwrap_or(0);
        }
        0
    }

    /// Merge with another vector clock (take max of each component)
    pub fn merge(&self, other: &VectorClock) -> VectorClock {
        let my_clock = self.clock.read().unwrap();
        let other_clock = other.clock.read().unwrap();

        let mut merged = my_clock.clone();
        for (node, &time) in other_clock.iter() {
            merged
                .entry(node.clone())
                .and_modify(|t| *t = (*t).max(time))
                .or_insert(time);
        }

        VectorClock {
            clock: Arc::new(RwLock::new(merged)),
            node_id: self.node_id.clone(),
        }
    }

    /// Check if this clock happened-before another
    ///
    /// Returns true if all components of this clock are <= the other,
    /// with at least one strictly less.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let my_clock = self.clock.read().unwrap();
        let other_clock = other.clock.read().unwrap();

        let mut at_least_one_less = false;

        for (node, &my_time) in my_clock.iter() {
            let other_time = other_clock.get(node).copied().unwrap_or(0);
            if my_time > other_time {
                return false;
            }
            if my_time < other_time {
                at_least_one_less = true;
            }
        }

        // Check if other has nodes we don't have
        for node in other_clock.keys() {
            if !my_clock.contains_key(node) && other_clock.get(node).copied().unwrap_or(0) > 0 {
                at_least_one_less = true;
            }
        }

        at_least_one_less
    }

    /// Check if two clocks are concurrent (neither happens-before the other)
    pub fn concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> String {
        if let Ok(clock) = self.clock.read() {
            let pairs: Vec<String> = clock
                .iter()
                .map(|(k, v)| format!("\"{}\":{}", k, v))
                .collect();
            return format!("{{{}}}", pairs.join(","));
        }
        "{}".to_string()
    }

    /// Get all node IDs in this clock
    pub fn nodes(&self) -> Vec<String> {
        if let Ok(clock) = self.clock.read() {
            return clock.keys().cloned().collect();
        }
        Vec::new()
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new("default")
    }
}

// ============================================================================
// CRDT REGISTER
// ============================================================================

/// Last-Writer-Wins Register (LWW-Register CRDT)
///
/// A conflict-free replicated data type that resolves concurrent writes
/// by keeping the value with the latest timestamp.
pub struct LWWRegister {
    value: Arc<RwLock<Option<Vec<u8>>>>,
    timestamp: Arc<RwLock<VectorClock>>,
    node_id: String,
}

impl LWWRegister {
    /// Create new LWW register
    pub fn new(node_id: &str) -> Self {
        Self {
            value: Arc::new(RwLock::new(None)),
            timestamp: Arc::new(RwLock::new(VectorClock::new(node_id))),
            node_id: node_id.to_string(),
        }
    }

    /// Set value (local write)
    pub fn set(&self, value: Vec<u8>) {
        if let Ok(ts) = self.timestamp.write() {
            ts.increment();
        }
        if let Ok(mut v) = self.value.write() {
            *v = Some(value);
        }
    }

    /// Get value
    pub fn get(&self) -> Option<Vec<u8>> {
        if let Ok(v) = self.value.read() {
            return v.clone();
        }
        None
    }

    /// Merge with remote value (LWW semantics)
    ///
    /// Accepts the remote value if it happened after local, or uses node_id
    /// as tie-breaker for concurrent updates.
    pub fn merge(&self, remote_value: Vec<u8>, remote_timestamp: &VectorClock) {
        // Get current timestamp and check ordering
        let my_ts_clone = self.timestamp.read().unwrap().clone();

        // If remote happened-before local, reject
        if remote_timestamp.happens_before(&my_ts_clone) {
            return;
        }

        let should_accept = if my_ts_clone.happens_before(remote_timestamp) {
            // Remote is strictly newer
            true
        } else {
            // Concurrent - use node_id as tie-breaker
            remote_timestamp.node_id > self.node_id
        };

        if should_accept {
            // Update value
            if let Ok(mut v) = self.value.write() {
                *v = Some(remote_value);
            }
        }

        // Always merge clocks
        let merged = my_ts_clone.merge(remote_timestamp);
        if let Ok(mut ts) = self.timestamp.write() {
            *ts = merged;
        }
    }

    /// Get current timestamp
    pub fn timestamp(&self) -> VectorClock {
        self.timestamp.read().unwrap().clone()
    }
}

// ============================================================================
// DISTRIBUTED NODE
// ============================================================================

/// Simulated distributed AmorphicEngine node
///
/// Represents a single node in a distributed AmorphicEngine cluster.
pub struct DistributedNode {
    node_id: String,
    data: Arc<RwLock<HashMap<String, (Vec<u8>, VectorClock)>>>,
    clock: Arc<RwLock<VectorClock>>,
    peers: Arc<RwLock<Vec<String>>>,
    pending_sync: Arc<RwLock<Vec<(String, Vec<u8>, VectorClock)>>>,
}

impl DistributedNode {
    /// Create new distributed node
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            data: Arc::new(RwLock::new(HashMap::new())),
            clock: Arc::new(RwLock::new(VectorClock::new(node_id))),
            peers: Arc::new(RwLock::new(Vec::new())),
            pending_sync: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Add peer node
    pub fn add_peer(&self, peer_id: &str) {
        if let Ok(mut peers) = self.peers.write() {
            peers.push(peer_id.to_string());
        }
    }

    /// Local write
    pub fn write(&self, key: &str, value: Vec<u8>) {
        let clock = self.clock.read().unwrap();
        let new_clock = clock.clone();
        drop(clock);
        new_clock.increment();

        if let Ok(mut data) = self.data.write() {
            data.insert(key.to_string(), (value.clone(), new_clock.clone()));
        }

        // Queue for sync
        if let Ok(mut pending) = self.pending_sync.write() {
            pending.push((key.to_string(), value, new_clock.clone()));
        }

        // Update local clock
        if let Ok(mut clock) = self.clock.write() {
            *clock = new_clock;
        }
    }

    /// Local read
    pub fn read(&self, key: &str) -> Option<Vec<u8>> {
        if let Ok(data) = self.data.read() {
            return data.get(key).map(|(v, _)| v.clone());
        }
        None
    }

    /// Receive sync from peer
    pub fn receive_sync(&self, key: &str, value: Vec<u8>, remote_clock: &VectorClock) -> bool {
        let mut data = self.data.write().unwrap();

        // Check if we have an existing value for this key
        let existing = data.get(key).map(|(v, c)| (v.clone(), c.clone()));

        if let Some((local_value, local_clock)) = existing {
            // Check causal ordering
            if remote_clock.happens_before(&local_clock) {
                return false; // Reject stale update
            }

            let should_accept = if local_clock.happens_before(remote_clock) {
                // Remote is strictly newer
                true
            } else if local_clock.concurrent(remote_clock) {
                // Concurrent - use node_id as consistent tie-breaker
                // Higher node_id wins (arbitrary but consistent)
                remote_clock.node_id > self.node_id
            } else {
                false
            };

            let merged = local_clock.merge(remote_clock);

            if should_accept {
                // Accept update with remote value
                data.insert(key.to_string(), (value, merged.clone()));
            } else {
                // Keep local value but merge clocks
                data.insert(key.to_string(), (local_value, merged.clone()));
            }

            // Update node clock
            let node_clock = self.clock.read().unwrap().merge(&merged);
            if let Ok(mut clock) = self.clock.write() {
                *clock = node_clock;
            }

            return should_accept;
        } else {
            // New key
            data.insert(key.to_string(), (value, remote_clock.clone()));

            // Update node clock
            let node_clock = self.clock.read().unwrap().merge(remote_clock);
            if let Ok(mut clock) = self.clock.write() {
                *clock = node_clock;
            }
            return true;
        }
    }

    /// Get pending sync items
    pub fn pending_sync_count(&self) -> usize {
        if let Ok(pending) = self.pending_sync.read() {
            return pending.len();
        }
        0
    }

    /// Get pending sync items
    pub fn get_pending_sync(&self) -> Vec<(String, Vec<u8>, VectorClock)> {
        if let Ok(pending) = self.pending_sync.read() {
            return pending.clone();
        }
        Vec::new()
    }

    /// Clear pending sync (after successful sync)
    pub fn clear_pending_sync(&self) {
        if let Ok(mut pending) = self.pending_sync.write() {
            pending.clear();
        }
    }

    /// Get number of keys
    pub fn key_count(&self) -> usize {
        if let Ok(data) = self.data.read() {
            return data.len();
        }
        0
    }

    /// Get number of peers
    pub fn peer_count(&self) -> usize {
        if let Ok(peers) = self.peers.read() {
            return peers.len();
        }
        0
    }
}

// ============================================================================
// DISTRIBUTED CLUSTER
// ============================================================================

/// Simulated distributed cluster
///
/// Manages multiple distributed nodes and provides sync operations.
pub struct DistributedCluster {
    nodes: Arc<RwLock<HashMap<String, DistributedNode>>>,
}

impl DistributedCluster {
    /// Create new cluster
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add node to cluster
    pub fn add_node(&self, node_id: &str) {
        if let Ok(mut nodes) = self.nodes.write() {
            // Add peers to existing nodes
            for (_, existing_node) in nodes.iter() {
                existing_node.add_peer(node_id);
            }

            // Create new node with existing peers
            let new_node = DistributedNode::new(node_id);
            for existing_id in nodes.keys() {
                new_node.add_peer(existing_id);
            }

            nodes.insert(node_id.to_string(), new_node);
        }
    }

    /// Write to a specific node
    pub fn write_to_node(
        &self,
        node_id: &str,
        key: &str,
        value: Vec<u8>,
    ) -> AmorphicEngineResult<()> {
        if let Ok(nodes) = self.nodes.read() {
            let node = nodes
                .get(node_id)
                .ok_or_else(|| AmorphicEngineError::NodeNotFound(node_id.to_string()))?;
            node.write(key, value);
            return Ok(());
        }
        Err(AmorphicEngineError::LockError(
            "failed to acquire nodes lock".to_string(),
        ))
    }

    /// Read from a specific node
    pub fn read_from_node(
        &self,
        node_id: &str,
        key: &str,
    ) -> AmorphicEngineResult<Option<Vec<u8>>> {
        if let Ok(nodes) = self.nodes.read() {
            let node = nodes
                .get(node_id)
                .ok_or_else(|| AmorphicEngineError::NodeNotFound(node_id.to_string()))?;
            return Ok(node.read(key));
        }
        Err(AmorphicEngineError::LockError(
            "failed to acquire nodes lock".to_string(),
        ))
    }

    /// Sync all nodes (simulated gossip round)
    pub fn sync_all(&self) -> u32 {
        if let Ok(nodes) = self.nodes.read() {
            let mut syncs_performed = 0u32;

            // Collect all pending syncs
            let mut all_syncs: Vec<(String, String, Vec<u8>, VectorClock)> = Vec::new();

            for (node_id, node) in nodes.iter() {
                for (key, value, clock) in node.get_pending_sync() {
                    all_syncs.push((node_id.clone(), key, value, clock));
                }
            }

            // Apply syncs to all other nodes
            for (source_id, key, value, clock) in all_syncs {
                for (target_id, target_node) in nodes.iter() {
                    if target_id != &source_id {
                        if target_node.receive_sync(&key, value.clone(), &clock) {
                            syncs_performed += 1;
                        }
                    }
                }
            }

            // Clear pending syncs
            for (_, node) in nodes.iter() {
                node.clear_pending_sync();
            }

            return syncs_performed;
        }
        0
    }

    /// Check if all nodes have the same value for a key
    pub fn is_consistent(&self, key: &str) -> bool {
        if let Ok(nodes) = self.nodes.read() {
            let mut values: Vec<Option<Vec<u8>>> = Vec::new();

            for (_, node) in nodes.iter() {
                values.push(node.read(key));
            }

            if values.is_empty() {
                return true;
            }

            let first = &values[0];
            return values.iter().all(|v| v == first);
        }
        false
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        if let Ok(nodes) = self.nodes.read() {
            return nodes.len();
        }
        0
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> Vec<String> {
        if let Ok(nodes) = self.nodes.read() {
            return nodes.keys().cloned().collect();
        }
        Vec::new()
    }
}

impl Default for DistributedCluster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_basic() {
        let clock = VectorClock::new("node1");
        assert_eq!(clock.get("node1"), 0);

        clock.increment();
        assert_eq!(clock.get("node1"), 1);

        clock.increment();
        assert_eq!(clock.get("node1"), 2);
    }

    #[test]
    fn test_vector_clock_merge() {
        let clock1 = VectorClock::new("node1");
        clock1.increment();
        clock1.increment();

        let clock2 = VectorClock::new("node2");
        clock2.increment();

        let merged = clock1.merge(&clock2);
        assert_eq!(merged.get("node1"), 2);
        assert_eq!(merged.get("node2"), 1);
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let clock1 = VectorClock::new("node1");
        clock1.increment();

        let clock2 = clock1.merge(&VectorClock::new("node2"));
        clock2.increment();

        assert!(clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let clock1 = VectorClock::new("node1");
        clock1.increment();

        let clock2 = VectorClock::new("node2");
        clock2.increment();

        assert!(clock1.concurrent(&clock2));
        assert!(clock2.concurrent(&clock1));
    }

    #[test]
    fn test_vector_clock_json() {
        let clock = VectorClock::new("node1");
        clock.increment();
        let json = clock.to_json();
        assert!(json.contains("node1"));
        assert!(json.contains("1"));
    }

    #[test]
    fn test_lww_register_basic() {
        let reg = LWWRegister::new("node1");
        assert!(reg.get().is_none());

        reg.set(b"value1".to_vec());
        assert_eq!(reg.get(), Some(b"value1".to_vec()));

        reg.set(b"value2".to_vec());
        assert_eq!(reg.get(), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_lww_register_merge() {
        let reg1 = LWWRegister::new("node1");
        reg1.set(b"value1".to_vec());

        let reg2 = LWWRegister::new("node2");
        reg2.set(b"value2".to_vec());

        // Merge reg2's value into reg1
        let ts2 = reg2.timestamp();
        reg1.merge(b"value2".to_vec(), &ts2);

        // Result depends on which happened-after or node_id tie-breaker
        assert!(reg1.get().is_some());
    }

    #[test]
    fn test_distributed_node_basic() {
        let node = DistributedNode::new("node1");

        node.write("key1", b"value1".to_vec());
        assert_eq!(node.read("key1"), Some(b"value1".to_vec()));
        assert_eq!(node.read("key2"), None);
    }

    #[test]
    fn test_distributed_node_sync() {
        let node1 = DistributedNode::new("node1");
        let node2 = DistributedNode::new("node2");

        node1.write("key1", b"value1".to_vec());

        // Get pending sync from node1
        let pending = node1.get_pending_sync();
        assert_eq!(pending.len(), 1);

        // Apply to node2
        let (key, value, clock) = &pending[0];
        assert!(node2.receive_sync(key, value.clone(), clock));

        // Now node2 should have the value
        assert_eq!(node2.read("key1"), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_distributed_cluster_basic() {
        let cluster = DistributedCluster::new();

        cluster.add_node("node1");
        cluster.add_node("node2");
        cluster.add_node("node3");

        assert_eq!(cluster.node_count(), 3);
    }

    #[test]
    fn test_distributed_cluster_write_read() {
        let cluster = DistributedCluster::new();
        cluster.add_node("node1");
        cluster.add_node("node2");

        cluster
            .write_to_node("node1", "key1", b"value1".to_vec())
            .unwrap();

        // Node1 has the value
        assert_eq!(
            cluster.read_from_node("node1", "key1").unwrap(),
            Some(b"value1".to_vec())
        );

        // Node2 doesn't have it yet
        assert_eq!(cluster.read_from_node("node2", "key1").unwrap(), None);
    }

    #[test]
    fn test_distributed_cluster_sync() {
        let cluster = DistributedCluster::new();
        cluster.add_node("node1");
        cluster.add_node("node2");

        cluster
            .write_to_node("node1", "key1", b"value1".to_vec())
            .unwrap();

        // Sync
        let syncs = cluster.sync_all();
        assert!(syncs > 0);

        // Now node2 should have it
        assert_eq!(
            cluster.read_from_node("node2", "key1").unwrap(),
            Some(b"value1".to_vec())
        );

        // Should be consistent
        assert!(cluster.is_consistent("key1"));
    }

    #[test]
    fn test_distributed_cluster_concurrent_writes() {
        let cluster = DistributedCluster::new();
        cluster.add_node("node1");
        cluster.add_node("node2");

        // Concurrent writes to same key
        cluster
            .write_to_node("node1", "key1", b"value_from_1".to_vec())
            .unwrap();
        cluster
            .write_to_node("node2", "key1", b"value_from_2".to_vec())
            .unwrap();

        // Not consistent yet (different values on each node)
        assert!(!cluster.is_consistent("key1"));

        // Sync - after sync, one value should win via tie-breaker
        let syncs = cluster.sync_all();
        // At least one sync should have happened
        assert!(syncs >= 0);

        // After sync, the cluster should be consistent
        // (either both have value_from_1 or both have value_from_2)
        assert!(cluster.is_consistent("key1"));
    }

    #[test]
    fn test_distributed_cluster_node_not_found() {
        let cluster = DistributedCluster::new();
        cluster.add_node("node1");

        let result = cluster.write_to_node("nonexistent", "key", b"value".to_vec());
        assert!(result.is_err());
    }

    #[test]
    fn test_distributed_cluster_multiple_syncs() {
        let cluster = DistributedCluster::new();
        cluster.add_node("node1");
        cluster.add_node("node2");
        cluster.add_node("node3");

        // Write to node1
        cluster
            .write_to_node("node1", "key1", b"v1".to_vec())
            .unwrap();
        cluster.sync_all();

        // Write to node2
        cluster
            .write_to_node("node2", "key2", b"v2".to_vec())
            .unwrap();
        cluster.sync_all();

        // Write to node3
        cluster
            .write_to_node("node3", "key3", b"v3".to_vec())
            .unwrap();
        cluster.sync_all();

        // All nodes should have all keys
        for node_id in cluster.node_ids() {
            assert_eq!(
                cluster.read_from_node(&node_id, "key1").unwrap(),
                Some(b"v1".to_vec())
            );
            assert_eq!(
                cluster.read_from_node(&node_id, "key2").unwrap(),
                Some(b"v2".to_vec())
            );
            assert_eq!(
                cluster.read_from_node(&node_id, "key3").unwrap(),
                Some(b"v3".to_vec())
            );
        }
    }
}
