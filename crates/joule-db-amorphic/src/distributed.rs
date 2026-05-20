//! Distributed Mode with Consistent Hashing
//!
//! This module provides horizontal scaling capabilities through consistent
//! hashing for data distribution across nodes.

use crate::{
    AmorphicError, AmorphicResult, RecordId, ShardedAmorphicStore, ShardedStoreStats, Value,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Unique identifier for a node in the cluster
pub type NodeId = u64;

/// Configuration for a node in the cluster
#[derive(Clone, Debug)]
pub struct NodeConfig {
    /// Unique node identifier
    pub id: NodeId,
    /// Node hostname or IP address
    pub host: String,
    /// Node port
    pub port: u16,
    /// Weight for virtual node allocation (higher = more data)
    pub weight: u32,
    /// Whether this node is available
    pub available: bool,
}

impl NodeConfig {
    /// Create a new node configuration
    pub fn new(id: NodeId, host: &str, port: u16) -> Self {
        Self {
            id,
            host: host.to_string(),
            port,
            weight: 100,
            available: true,
        }
    }

    /// Builder pattern for weight
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Get the address string
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Consistent hash ring for node selection
///
/// Uses virtual nodes (vnodes) to improve distribution uniformity.
#[derive(Debug)]
pub struct ConsistentHashRing {
    /// Map from hash positions to node IDs
    ring: BTreeMap<u64, NodeId>,
    /// Number of virtual nodes per physical node (multiplied by weight)
    vnodes_per_node: usize,
    /// All nodes in the cluster
    nodes: HashMap<NodeId, NodeConfig>,
}

impl ConsistentHashRing {
    /// Default number of virtual nodes per physical node
    const DEFAULT_VNODES: usize = 150;

    /// Create a new consistent hash ring
    pub fn new() -> Self {
        Self {
            ring: BTreeMap::new(),
            vnodes_per_node: Self::DEFAULT_VNODES,
            nodes: HashMap::new(),
        }
    }

    /// Create with custom virtual node count
    pub fn with_vnodes(vnodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            vnodes_per_node: vnodes.max(1),
            nodes: HashMap::new(),
        }
    }

    /// Compute hash for a key
    fn hash_key(key: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Compute hash for a vnode
    fn hash_vnode(node_id: NodeId, vnode_idx: usize) -> u64 {
        Self::hash_key(&format!("{}:{}", node_id, vnode_idx))
    }

    /// Add a node to the ring
    pub fn add_node(&mut self, config: NodeConfig) -> AmorphicResult<()> {
        if self.nodes.contains_key(&config.id) {
            return Err(AmorphicError::InvalidQuery(format!(
                "Node {} already exists",
                config.id
            )));
        }

        // Calculate number of vnodes based on weight
        let vnode_count =
            (self.vnodes_per_node as f64 * config.weight as f64 / 100.0).ceil() as usize;

        // Add virtual nodes to the ring
        for i in 0..vnode_count {
            let hash = Self::hash_vnode(config.id, i);
            self.ring.insert(hash, config.id);
        }

        self.nodes.insert(config.id, config);
        Ok(())
    }

    /// Remove a node from the ring
    pub fn remove_node(&mut self, node_id: NodeId) -> AmorphicResult<()> {
        if !self.nodes.contains_key(&node_id) {
            return Err(AmorphicError::NotFound);
        }

        let config = self.nodes.remove(&node_id).unwrap();
        let vnode_count =
            (self.vnodes_per_node as f64 * config.weight as f64 / 100.0).ceil() as usize;

        // Remove all virtual nodes
        for i in 0..vnode_count {
            let hash = Self::hash_vnode(node_id, i);
            self.ring.remove(&hash);
        }

        Ok(())
    }

    /// Mark a node as unavailable (without removing it)
    pub fn set_node_available(&mut self, node_id: NodeId, available: bool) -> AmorphicResult<()> {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.available = available;
            Ok(())
        } else {
            Err(AmorphicError::NotFound)
        }
    }

    /// Get the primary node for a key
    pub fn get_node(&self, key: &str) -> Option<NodeId> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = Self::hash_key(key);

        // Find the first node with hash >= key hash
        let node_id = self
            .ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next()) // Wrap around
            .map(|(_, id)| id)?;

        // Skip unavailable nodes
        if let Some(node) = self.nodes.get(&node_id) {
            if node.available {
                return Some(*node_id);
            }
        }

        // Find next available node
        self.find_next_available(hash)
    }

    /// Find next available node from a position
    fn find_next_available(&self, from_hash: u64) -> Option<NodeId> {
        // Search forward
        for (_, &node_id) in self.ring.range(from_hash..) {
            if let Some(node) = self.nodes.get(&node_id) {
                if node.available {
                    return Some(node_id);
                }
            }
        }

        // Wrap around and search from beginning
        for (_, &node_id) in self.ring.iter() {
            if let Some(node) = self.nodes.get(&node_id) {
                if node.available {
                    return Some(node_id);
                }
            }
        }

        None
    }

    /// Get N replicas for a key (for replication)
    pub fn get_replicas(&self, key: &str, count: usize) -> Vec<NodeId> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = Self::hash_key(key);
        let mut replicas = Vec::with_capacity(count);
        let mut seen = std::collections::HashSet::new();

        // Search forward from hash position
        for (_, &node_id) in self.ring.range(hash..) {
            if seen.insert(node_id) {
                if let Some(node) = self.nodes.get(&node_id) {
                    if node.available {
                        replicas.push(node_id);
                        if replicas.len() >= count {
                            return replicas;
                        }
                    }
                }
            }
        }

        // Wrap around
        for (_, &node_id) in self.ring.iter() {
            if seen.insert(node_id) {
                if let Some(node) = self.nodes.get(&node_id) {
                    if node.available {
                        replicas.push(node_id);
                        if replicas.len() >= count {
                            return replicas;
                        }
                    }
                }
            }
        }

        replicas
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &NodeConfig> {
        self.nodes.values()
    }

    /// Get a specific node config
    pub fn get_node_config(&self, node_id: NodeId) -> Option<&NodeConfig> {
        self.nodes.get(&node_id)
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of available nodes
    pub fn available_node_count(&self) -> usize {
        self.nodes.values().filter(|n| n.available).count()
    }

    /// Get the distribution of keys across nodes (for diagnostics)
    pub fn get_distribution(&self, sample_keys: &[&str]) -> HashMap<NodeId, usize> {
        let mut dist = HashMap::new();
        for key in sample_keys {
            if let Some(node) = self.get_node(key) {
                *dist.entry(node).or_insert(0) += 1;
            }
        }
        dist
    }
}

impl Default for ConsistentHashRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from a distributed query
#[derive(Clone, Debug)]
pub struct DistributedQueryResult {
    /// Records from all nodes (id, fields map)
    pub records: Vec<(RecordId, HashMap<String, Value>)>,
    /// Nodes that participated
    pub participating_nodes: Vec<NodeId>,
    /// Nodes that failed
    pub failed_nodes: Vec<NodeId>,
    /// Total query time in milliseconds
    pub total_time_ms: u64,
}

impl DistributedQueryResult {
    /// Get record count
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

/// Local node store wrapper
struct LocalNodeStore {
    node_id: NodeId,
    store: ShardedAmorphicStore,
}

/// Distributed store for horizontal scaling
///
/// This is a simulation layer for distributed operations.
/// In a real deployment, this would use network transport.
pub struct DistributedStore {
    /// Local node ID
    local_node: NodeId,
    /// All node configurations
    nodes: HashMap<NodeId, NodeConfig>,
    /// Consistent hash ring for routing
    ring: RwLock<ConsistentHashRing>,
    /// Replication factor
    replication_factor: usize,
    /// Local stores (simulating remote nodes for testing)
    local_stores: RwLock<HashMap<NodeId, Arc<RwLock<ShardedAmorphicStore>>>>,
    /// Next record ID (global counter)
    next_id: AtomicU64,
}

impl DistributedStore {
    /// Create a new distributed store
    pub fn new(local_node_id: NodeId) -> Self {
        Self {
            local_node: local_node_id,
            nodes: HashMap::new(),
            ring: RwLock::new(ConsistentHashRing::new()),
            replication_factor: 1,
            local_stores: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create with custom replication factor
    pub fn with_replication(local_node_id: NodeId, replication_factor: usize) -> Self {
        Self {
            local_node: local_node_id,
            nodes: HashMap::new(),
            ring: RwLock::new(ConsistentHashRing::new()),
            replication_factor: replication_factor.max(1),
            local_stores: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Add a node to the cluster
    pub fn add_node(&mut self, config: NodeConfig) -> AmorphicResult<()> {
        let node_id = config.id;

        // Add to ring
        self.ring.write().unwrap().add_node(config.clone())?;

        // Add to nodes map
        self.nodes.insert(node_id, config);

        // Create local store simulation
        self.local_stores
            .write()
            .unwrap()
            .insert(node_id, Arc::new(RwLock::new(ShardedAmorphicStore::new())));

        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: NodeId) -> AmorphicResult<()> {
        // Remove from ring
        self.ring.write().unwrap().remove_node(node_id)?;

        // Remove from nodes map
        self.nodes.remove(&node_id);

        // Remove local store
        self.local_stores.write().unwrap().remove(&node_id);

        Ok(())
    }

    /// Mark a node as available or unavailable
    pub fn set_node_available(&self, node_id: NodeId, available: bool) -> AmorphicResult<()> {
        self.ring
            .write()
            .unwrap()
            .set_node_available(node_id, available)
    }

    /// Get the next global ID
    fn next_global_id(&self) -> RecordId {
        self.next_id.fetch_add(1, Ordering::SeqCst) as RecordId
    }

    /// Route a key to its primary node
    pub fn route(&self, key: &str) -> Option<NodeId> {
        self.ring.read().unwrap().get_node(key)
    }

    /// Route a key to all its replica nodes
    pub fn route_replicas(&self, key: &str) -> Vec<NodeId> {
        self.ring
            .read()
            .unwrap()
            .get_replicas(key, self.replication_factor)
    }

    /// Ingest a JSON document
    pub fn ingest_json(&self, json: &str) -> AmorphicResult<RecordId> {
        let nodes = self.route_replicas(json);

        if nodes.is_empty() {
            return Err(AmorphicError::QueryError("No available nodes".to_string()));
        }

        let global_id = self.next_global_id();

        // Write to all replica nodes
        let stores = self.local_stores.read().unwrap();
        for node_id in nodes {
            if let Some(store) = stores.get(&node_id) {
                store.write().unwrap().ingest_json(json)?;
            }
        }

        Ok(global_id)
    }

    /// Ingest a row
    pub fn ingest_row(&self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        let content = values.join("|");
        let nodes = self.route_replicas(&content);

        if nodes.is_empty() {
            return Err(AmorphicError::QueryError("No available nodes".to_string()));
        }

        let global_id = self.next_global_id();

        // Write to all replica nodes
        let stores = self.local_stores.read().unwrap();
        for node_id in nodes {
            if let Some(store) = stores.get(&node_id) {
                store.write().unwrap().ingest_row(columns, values)?;
            }
        }

        Ok(global_id)
    }

    /// Query a range across all nodes (scatter-gather)
    pub fn query_range(
        &self,
        field: &str,
        min: f64,
        max: f64,
    ) -> AmorphicResult<DistributedQueryResult> {
        use std::time::Instant;

        let start = Instant::now();
        let mut all_records = Vec::new();
        let mut participating = Vec::new();
        let mut failed = Vec::new();

        let stores = self.local_stores.read().unwrap();

        // Query all nodes in parallel (simulated)
        for (&node_id, store_arc) in stores.iter() {
            match store_arc.read() {
                Ok(store) => {
                    // Query this node's shard
                    match store.query_range(field, min, max) {
                        Ok(results) => {
                            // Convert QueryResult to records
                            for record in results.into_records() {
                                all_records.push((record.id, record.fields.clone()));
                            }
                            participating.push(node_id);
                        }
                        Err(_) => {
                            failed.push(node_id);
                        }
                    }
                }
                Err(_) => {
                    failed.push(node_id);
                }
            }
        }

        Ok(DistributedQueryResult {
            records: all_records,
            participating_nodes: participating,
            failed_nodes: failed,
            total_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Query for a specific value across all nodes
    pub fn query_equals(
        &self,
        field: &str,
        value: &Value,
    ) -> AmorphicResult<DistributedQueryResult> {
        use std::time::Instant;

        let start = Instant::now();
        let mut all_records = Vec::new();
        let mut participating = Vec::new();
        let mut failed = Vec::new();

        let stores = self.local_stores.read().unwrap();

        for (&node_id, store_arc) in stores.iter() {
            match store_arc.read() {
                Ok(store) => {
                    match store.query_equals(field, value) {
                        Ok(results) => {
                            // Convert QueryResult to records
                            for record in results.into_records() {
                                all_records.push((record.id, record.fields.clone()));
                            }
                            participating.push(node_id);
                        }
                        Err(_) => {
                            failed.push(node_id);
                        }
                    }
                }
                Err(_) => {
                    failed.push(node_id);
                }
            }
        }

        Ok(DistributedQueryResult {
            records: all_records,
            participating_nodes: participating,
            failed_nodes: failed,
            total_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Get statistics from all nodes
    pub fn stats(&self) -> HashMap<NodeId, ShardedStoreStats> {
        let mut all_stats = HashMap::new();

        let stores = self.local_stores.read().unwrap();
        for (&node_id, store_arc) in stores.iter() {
            if let Ok(store) = store_arc.read() {
                if let Ok(stats) = store.stats() {
                    all_stats.insert(node_id, stats);
                }
            }
        }

        all_stats
    }

    /// Get total record count across all nodes
    pub fn total_record_count(&self) -> usize {
        let stores = self.local_stores.read().unwrap();
        stores
            .values()
            .filter_map(|s| s.read().ok())
            .filter_map(|s| s.stats().ok())
            .map(|stats| stats.total_records)
            .sum()
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of available nodes
    pub fn available_node_count(&self) -> usize {
        self.ring.read().unwrap().available_node_count()
    }

    /// Get the local node ID
    pub fn local_node_id(&self) -> NodeId {
        self.local_node
    }

    /// Get replication factor
    pub fn replication_factor(&self) -> usize {
        self.replication_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_ring_new() {
        let ring = ConsistentHashRing::new();
        assert_eq!(ring.node_count(), 0);
        assert!(ring.get_node("test").is_none());
    }

    #[test]
    fn test_consistent_hash_ring_add_node() {
        let mut ring = ConsistentHashRing::new();

        ring.add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        assert_eq!(ring.node_count(), 1);

        ring.add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();
        assert_eq!(ring.node_count(), 2);
    }

    #[test]
    fn test_consistent_hash_ring_get_node() {
        let mut ring = ConsistentHashRing::new();

        ring.add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        ring.add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();
        ring.add_node(NodeConfig::new(3, "localhost", 8003))
            .unwrap();

        // Same key should always route to same node
        let node1 = ring.get_node("test_key_1").unwrap();
        let node1_again = ring.get_node("test_key_1").unwrap();
        assert_eq!(node1, node1_again);

        // Different keys may route to different nodes
        let node2 = ring.get_node("another_key").unwrap();
        assert!(node2 > 0); // Should have a valid node
    }

    #[test]
    fn test_consistent_hash_ring_remove_node() {
        let mut ring = ConsistentHashRing::new();

        ring.add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        ring.add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();
        assert_eq!(ring.node_count(), 2);

        ring.remove_node(1).unwrap();
        assert_eq!(ring.node_count(), 1);

        // Should error on removing non-existent node
        assert!(ring.remove_node(99).is_err());
    }

    #[test]
    fn test_consistent_hash_ring_unavailable() {
        let mut ring = ConsistentHashRing::new();

        ring.add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        ring.add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        let initial_node = ring.get_node("test_key").unwrap();

        // Mark node as unavailable
        ring.set_node_available(initial_node, false).unwrap();

        // Should now route to a different node
        let new_node = ring.get_node("test_key").unwrap();
        assert_ne!(new_node, initial_node);
    }

    #[test]
    fn test_consistent_hash_ring_replicas() {
        let mut ring = ConsistentHashRing::new();

        ring.add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        ring.add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();
        ring.add_node(NodeConfig::new(3, "localhost", 8003))
            .unwrap();

        let replicas = ring.get_replicas("test_key", 2);
        assert_eq!(replicas.len(), 2);

        // All replicas should be unique
        assert_ne!(replicas[0], replicas[1]);
    }

    #[test]
    fn test_consistent_hash_ring_weight() {
        let mut ring = ConsistentHashRing::with_vnodes(500);

        // Node 1 has double weight
        ring.add_node(NodeConfig::new(1, "localhost", 8001).with_weight(200))
            .unwrap();
        ring.add_node(NodeConfig::new(2, "localhost", 8002).with_weight(100))
            .unwrap();

        // Generate many keys and check distribution
        let keys: Vec<String> = (0..10000).map(|i| format!("key_{}", i)).collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let dist = ring.get_distribution(&key_refs);

        // Node 1 should have more keys than node 2 due to higher weight
        let count_1 = *dist.get(&1).unwrap_or(&0);
        let count_2 = *dist.get(&2).unwrap_or(&0);

        // With more vnodes and keys, the ratio should be closer to 2x
        // Allow for wide variance due to hash distribution
        let ratio = count_1 as f64 / count_2.max(1) as f64;
        assert!(
            ratio > 1.0,
            "Node 1 (weight 200) should have more keys than node 2 (weight 100), ratio was {}",
            ratio
        );
    }

    #[test]
    fn test_distributed_store_new() {
        let store = DistributedStore::new(1);
        assert_eq!(store.local_node_id(), 1);
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.replication_factor(), 1);
    }

    #[test]
    fn test_distributed_store_add_nodes() {
        let mut store = DistributedStore::new(1);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        assert_eq!(store.node_count(), 2);
        assert_eq!(store.available_node_count(), 2);
    }

    #[test]
    fn test_distributed_store_ingest_json() {
        let mut store = DistributedStore::new(1);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        assert!(id > 0);

        assert!(store.total_record_count() >= 1);
    }

    #[test]
    fn test_distributed_store_ingest_multiple() {
        let mut store = DistributedStore::new(1);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        for i in 0..100 {
            store
                .ingest_json(&format!(r#"{{"id": {}, "value": "test_{}"}} "#, i, i))
                .unwrap();
        }

        // Records should be distributed across nodes
        let stats = store.stats();
        let total: usize = stats.values().map(|s| s.total_records).sum();
        assert_eq!(total, 100);

        // Both nodes should have some records
        for (_, node_stats) in stats {
            assert!(node_stats.total_records > 0);
        }
    }

    #[test]
    fn test_distributed_store_query_range() {
        let mut store = DistributedStore::new(1);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        // Ingest some records with numeric values
        for i in 0..50 {
            store
                .ingest_json(&format!(r#"{{"value": {}}}"#, i))
                .unwrap();
        }

        // Query range
        let result = store.query_range("value", 10.0, 20.0).unwrap();

        // Should have found some records
        assert!(!result.records.is_empty());
        assert_eq!(result.participating_nodes.len(), 2);
        assert!(result.failed_nodes.is_empty());
    }

    #[test]
    fn test_distributed_store_replication() {
        let mut store = DistributedStore::with_replication(1, 2);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();
        store
            .add_node(NodeConfig::new(3, "localhost", 8003))
            .unwrap();

        // With replication factor 2, each record should be on 2 nodes
        store.ingest_json(r#"{"test": "data"}"#).unwrap();

        // Total count across all nodes should be 2 (replicated)
        let total = store.total_record_count();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_distributed_store_node_failure() {
        let mut store = DistributedStore::new(1);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        // Mark node 1 as unavailable
        store.set_node_available(1, false).unwrap();

        // Ingestion should still work (routes to node 2)
        let id = store.ingest_json(r#"{"key": "value"}"#).unwrap();
        assert!(id > 0);

        // Only node 2 should be used
        let stats = store.stats();
        let node1_count = stats.get(&1).map(|s| s.total_records).unwrap_or(0);
        let node2_count = stats.get(&2).map(|s| s.total_records).unwrap_or(0);

        assert_eq!(node1_count, 0);
        assert!(node2_count > 0);
    }

    #[test]
    fn test_distributed_store_routing() {
        let mut store = DistributedStore::new(1);

        store
            .add_node(NodeConfig::new(1, "localhost", 8001))
            .unwrap();
        store
            .add_node(NodeConfig::new(2, "localhost", 8002))
            .unwrap();

        // Same key should always route to same node
        let node1 = store.route("test_key").unwrap();
        let node1_again = store.route("test_key").unwrap();
        assert_eq!(node1, node1_again);
    }

    #[test]
    fn test_node_config() {
        let config = NodeConfig::new(1, "192.168.1.1", 8080);
        assert_eq!(config.id, 1);
        assert_eq!(config.host, "192.168.1.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.weight, 100);
        assert!(config.available);
        assert_eq!(config.address(), "192.168.1.1:8080");
    }

    #[test]
    fn test_node_config_weight() {
        let config = NodeConfig::new(1, "localhost", 8080).with_weight(200);
        assert_eq!(config.weight, 200);
    }

    #[test]
    fn test_distributed_query_result() {
        let result = DistributedQueryResult {
            records: vec![],
            participating_nodes: vec![1, 2],
            failed_nodes: vec![3],
            total_time_ms: 100,
        };

        assert_eq!(result.participating_nodes.len(), 2);
        assert_eq!(result.failed_nodes.len(), 1);
        assert_eq!(result.total_time_ms, 100);
    }
}
