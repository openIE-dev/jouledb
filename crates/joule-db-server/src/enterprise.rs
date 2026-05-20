//! Enterprise Features for JouleDB Server
//!
//! Provides high-availability and scalability features:
//! - Replication (leader-follower)
//! - Sharding
//! - Load balancing
//! - Failover
//! - Cluster management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// Types
// ============================================================================

/// Enterprise error
#[derive(Debug, Clone, PartialEq)]
pub enum EnterpriseError {
    NodeNotFound(String),
    NodeUnreachable(String),
    QuorumNotReached,
    ShardNotFound(String),
    ReplicationFailed(String),
    LeaderElectionFailed(String),
    ConfigurationError(String),
    ConnectionError(String),
}

impl std::fmt::Display for EnterpriseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            Self::NodeUnreachable(id) => write!(f, "Node unreachable: {}", id),
            Self::QuorumNotReached => write!(f, "Quorum not reached"),
            Self::ShardNotFound(id) => write!(f, "Shard not found: {}", id),
            Self::ReplicationFailed(msg) => write!(f, "Replication failed: {}", msg),
            Self::LeaderElectionFailed(msg) => write!(f, "Leader election failed: {}", msg),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            Self::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
        }
    }
}

impl std::error::Error for EnterpriseError {}

/// Node role in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Primary/leader node
    Leader,
    /// Secondary/follower node
    Follower,
    /// Candidate during election
    Candidate,
    /// Observer (read-only)
    Observer,
}

/// Node health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealth {
    /// Node is healthy
    Healthy,
    /// Node is degraded
    Degraded,
    /// Node is unhealthy
    Unhealthy,
    /// Node is unknown/unreachable
    Unknown,
}

/// Replication mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationMode {
    /// Synchronous replication (wait for all replicas)
    Synchronous,
    /// Semi-synchronous (wait for quorum)
    SemiSync,
    /// Asynchronous replication
    Async,
}

impl Default for ReplicationMode {
    fn default() -> Self {
        Self::SemiSync
    }
}

/// Sharding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding
    Hash,
    /// Range-based sharding
    Range,
    /// Directory-based sharding
    Directory,
    /// Geographic sharding
    Geographic,
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        Self::Hash
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Random
    Random,
    /// Weighted
    Weighted,
    /// Latency-based
    LatencyBased,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

// ============================================================================
// Node Information
// ============================================================================

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node ID
    pub id: String,
    /// Node address
    pub address: String,
    /// Node role
    pub role: NodeRole,
    /// Health status
    pub health: NodeHealth,
    /// Last heartbeat timestamp
    pub last_heartbeat: u64,
    /// Replication lag (in milliseconds)
    pub replication_lag_ms: u64,
    /// Node weight (for load balancing)
    pub weight: u32,
    /// Node region/zone
    pub region: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl NodeInfo {
    /// Create new node info
    pub fn new(id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            role: NodeRole::Follower,
            health: NodeHealth::Unknown,
            last_heartbeat: 0,
            replication_lag_ms: 0,
            weight: 1,
            region: None,
            metadata: HashMap::new(),
        }
    }

    /// Check if node is available for reads
    pub fn is_readable(&self) -> bool {
        matches!(self.health, NodeHealth::Healthy | NodeHealth::Degraded)
    }

    /// Check if node is available for writes
    pub fn is_writable(&self) -> bool {
        self.role == NodeRole::Leader && self.health == NodeHealth::Healthy
    }
}

// ============================================================================
// Shard Information
// ============================================================================

/// Shard information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard ID
    pub id: String,
    /// Shard range start (for range-based sharding)
    pub range_start: Option<Vec<u8>>,
    /// Shard range end (for range-based sharding)
    pub range_end: Option<Vec<u8>>,
    /// Primary node ID
    pub primary_node: String,
    /// Replica node IDs
    pub replica_nodes: Vec<String>,
    /// Shard status
    pub status: ShardStatus,
    /// Record count
    pub record_count: u64,
    /// Size in bytes
    pub size_bytes: u64,
}

/// Shard status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStatus {
    /// Shard is online
    Online,
    /// Shard is being migrated
    Migrating,
    /// Shard is being split
    Splitting,
    /// Shard is offline
    Offline,
}

// ============================================================================
// Cluster Configuration
// ============================================================================

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Cluster name
    pub cluster_name: String,
    /// Replication mode
    pub replication_mode: ReplicationMode,
    /// Minimum replication factor
    pub min_replication_factor: usize,
    /// Target replication factor
    pub target_replication_factor: usize,
    /// Sharding strategy
    pub sharding_strategy: ShardingStrategy,
    /// Number of shards
    pub num_shards: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Heartbeat interval (seconds)
    pub heartbeat_interval_secs: u64,
    /// Election timeout (seconds)
    pub election_timeout_secs: u64,
    /// Maximum replication lag before node is unhealthy (ms)
    pub max_replication_lag_ms: u64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            cluster_name: "joule-db-cluster".to_string(),
            replication_mode: ReplicationMode::SemiSync,
            min_replication_factor: 2,
            target_replication_factor: 3,
            sharding_strategy: ShardingStrategy::Hash,
            num_shards: 16,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            heartbeat_interval_secs: 5,
            election_timeout_secs: 10,
            max_replication_lag_ms: 1000,
        }
    }
}

// ============================================================================
// Replication Manager
// ============================================================================

/// Replication Manager
pub struct ReplicationManager {
    config: ClusterConfig,
    local_node_id: String,
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    current_leader: Arc<RwLock<Option<String>>>,
    current_term: Arc<RwLock<u64>>,
    replication_log: Arc<RwLock<Vec<ReplicationEntry>>>,
}

/// Replication log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationEntry {
    pub sequence: u64,
    pub term: u64,
    pub timestamp: u64,
    pub operation: ReplicationOperation,
}

/// Replication operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationOperation {
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        key: Vec<u8>,
    },
    Batch {
        operations: Vec<ReplicationOperation>,
    },
}

impl ReplicationManager {
    /// Create new replication manager
    pub fn new(local_node_id: impl Into<String>, config: ClusterConfig) -> Self {
        Self {
            config,
            local_node_id: local_node_id.into(),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            current_leader: Arc::new(RwLock::new(None)),
            current_term: Arc::new(RwLock::new(0)),
            replication_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Get local node ID
    pub fn local_node_id(&self) -> &str {
        &self.local_node_id
    }

    /// Register a node
    pub fn register_node(&self, node: NodeInfo) {
        crate::lock_util::write_lock(&self.nodes).insert(node.id.clone(), node);
    }

    /// Remove a node
    pub fn remove_node(&self, node_id: &str) {
        crate::lock_util::write_lock(&self.nodes).remove(node_id);
    }

    /// Get node info
    pub fn get_node(&self, node_id: &str) -> Option<NodeInfo> {
        crate::lock_util::read_lock(&self.nodes)
            .get(node_id)
            .cloned()
    }

    /// List all nodes
    pub fn list_nodes(&self) -> Vec<NodeInfo> {
        crate::lock_util::read_lock(&self.nodes)
            .values()
            .cloned()
            .collect()
    }

    /// Get healthy nodes
    pub fn healthy_nodes(&self) -> Vec<NodeInfo> {
        crate::lock_util::read_lock(&self.nodes)
            .values()
            .filter(|n| n.health == NodeHealth::Healthy)
            .cloned()
            .collect()
    }

    /// Get current leader
    pub fn get_leader(&self) -> Option<NodeInfo> {
        let leader_id = crate::lock_util::read_lock(&self.current_leader).clone()?;
        self.get_node(&leader_id)
    }

    /// Check if this node is leader
    pub fn is_leader(&self) -> bool {
        crate::lock_util::read_lock(&self.current_leader)
            .as_ref()
            .map(|id| id == &self.local_node_id)
            .unwrap_or(false)
    }

    /// Update heartbeat for a node
    pub fn update_heartbeat(&self, node_id: &str) {
        if let Some(node) = crate::lock_util::write_lock(&self.nodes).get_mut(node_id) {
            node.last_heartbeat = Self::current_timestamp();
            node.health = NodeHealth::Healthy;
        }
    }

    /// Check node health based on heartbeats
    pub fn check_health(&self) {
        let now = Self::current_timestamp();
        let timeout = self.config.heartbeat_interval_secs * 3;

        let mut nodes = crate::lock_util::write_lock(&self.nodes);
        for node in nodes.values_mut() {
            if node.last_heartbeat == 0 {
                node.health = NodeHealth::Unknown;
            } else if now - node.last_heartbeat > timeout * 2 {
                node.health = NodeHealth::Unhealthy;
            } else if now - node.last_heartbeat > timeout {
                node.health = NodeHealth::Degraded;
            } else {
                node.health = NodeHealth::Healthy;
            }
        }
    }

    /// Append operation to replication log
    pub fn append_operation(&self, operation: ReplicationOperation) -> u64 {
        let mut log = crate::lock_util::write_lock(&self.replication_log);
        let sequence = log.len() as u64 + 1;
        let term = *crate::lock_util::read_lock(&self.current_term);

        log.push(ReplicationEntry {
            sequence,
            term,
            timestamp: Self::current_timestamp(),
            operation,
        });

        sequence
    }

    /// Get log entries since a sequence number
    pub fn get_entries_since(&self, sequence: u64) -> Vec<ReplicationEntry> {
        crate::lock_util::read_lock(&self.replication_log)
            .iter()
            .filter(|e| e.sequence > sequence)
            .cloned()
            .collect()
    }

    /// Get last sequence number
    pub fn last_sequence(&self) -> u64 {
        crate::lock_util::read_lock(&self.replication_log)
            .last()
            .map(|e| e.sequence)
            .unwrap_or(0)
    }

    /// Start leader election
    pub fn start_election(&self) -> Result<bool, EnterpriseError> {
        // Increment term
        {
            let mut term = crate::lock_util::write_lock(&self.current_term);
            *term += 1;
        }

        // Vote for self
        let votes = 1;
        let total_nodes = crate::lock_util::read_lock(&self.nodes).len();

        // Simple majority (in real impl, would send vote requests to other nodes)
        let quorum = total_nodes / 2 + 1;

        if votes >= quorum {
            // Become leader
            let mut leader = crate::lock_util::write_lock(&self.current_leader);
            *leader = Some(self.local_node_id.clone());

            // Update local node role
            if let Some(node) =
                crate::lock_util::write_lock(&self.nodes).get_mut(&self.local_node_id)
            {
                node.role = NodeRole::Leader;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Step down as leader
    pub fn step_down(&self) {
        let mut leader = crate::lock_util::write_lock(&self.current_leader);
        if leader.as_ref() == Some(&self.local_node_id) {
            *leader = None;

            if let Some(node) =
                crate::lock_util::write_lock(&self.nodes).get_mut(&self.local_node_id)
            {
                node.role = NodeRole::Follower;
            }
        }
    }
}

// ============================================================================
// Shard Manager
// ============================================================================

/// Shard Manager
pub struct ShardManager {
    /// Cluster configuration for shard management
    config: ClusterConfig,
    shards: Arc<RwLock<HashMap<String, ShardInfo>>>,
}

impl ShardManager {
    /// Create new shard manager
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the cluster configuration
    pub fn config(&self) -> &ClusterConfig {
        &self.config
    }

    /// Initialize shards
    pub fn initialize_shards(&self, num_shards: usize) {
        let mut shards = crate::lock_util::write_lock(&self.shards);
        for i in 0..num_shards {
            let shard_id = format!("shard_{}", i);
            shards.insert(
                shard_id.clone(),
                ShardInfo {
                    id: shard_id,
                    range_start: None,
                    range_end: None,
                    primary_node: String::new(),
                    replica_nodes: Vec::new(),
                    status: ShardStatus::Online,
                    record_count: 0,
                    size_bytes: 0,
                },
            );
        }
    }

    /// Get shard for a key (hash-based)
    pub fn get_shard_for_key(&self, key: &[u8]) -> Option<ShardInfo> {
        let shards = crate::lock_util::read_lock(&self.shards);
        if shards.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);
        let shard_index = (hash as usize) % shards.len();
        let shard_id = format!("shard_{}", shard_index);
        shards.get(&shard_id).cloned()
    }

    fn hash_key(&self, key: &[u8]) -> u64 {
        // Simple FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in key {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Get shard by ID
    pub fn get_shard(&self, shard_id: &str) -> Option<ShardInfo> {
        crate::lock_util::read_lock(&self.shards)
            .get(shard_id)
            .cloned()
    }

    /// List all shards
    pub fn list_shards(&self) -> Vec<ShardInfo> {
        crate::lock_util::read_lock(&self.shards)
            .values()
            .cloned()
            .collect()
    }

    /// Update shard info
    pub fn update_shard(&self, shard: ShardInfo) {
        crate::lock_util::write_lock(&self.shards).insert(shard.id.clone(), shard);
    }

    /// Assign node to shard
    pub fn assign_node_to_shard(
        &self,
        shard_id: &str,
        node_id: &str,
        as_primary: bool,
    ) -> Result<(), EnterpriseError> {
        let mut shards = crate::lock_util::write_lock(&self.shards);
        let shard = shards
            .get_mut(shard_id)
            .ok_or_else(|| EnterpriseError::ShardNotFound(shard_id.to_string()))?;

        if as_primary {
            shard.primary_node = node_id.to_string();
        } else {
            if !shard.replica_nodes.contains(&node_id.to_string()) {
                shard.replica_nodes.push(node_id.to_string());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Load Balancer
// ============================================================================

/// Load Balancer
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    nodes: Arc<RwLock<Vec<NodeInfo>>>,
    round_robin_index: Arc<RwLock<usize>>,
    connection_counts: Arc<RwLock<HashMap<String, usize>>>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            nodes: Arc::new(RwLock::new(Vec::new())),
            round_robin_index: Arc::new(RwLock::new(0)),
            connection_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update node list
    pub fn update_nodes(&self, nodes: Vec<NodeInfo>) {
        *crate::lock_util::write_lock(&self.nodes) = nodes;
    }

    /// Get next node for a request
    pub fn get_node(&self) -> Option<NodeInfo> {
        let nodes = crate::lock_util::read_lock(&self.nodes);
        let healthy: Vec<&NodeInfo> = nodes.iter().filter(|n| n.is_readable()).collect();

        if healthy.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut index = crate::lock_util::write_lock(&self.round_robin_index);
                let node = healthy[*index % healthy.len()].clone();
                *index = (*index + 1) % healthy.len();
                Some(node)
            }
            LoadBalancingStrategy::LeastConnections => {
                let counts = crate::lock_util::read_lock(&self.connection_counts);
                healthy
                    .into_iter()
                    .min_by_key(|n| counts.get(&n.id).copied().unwrap_or(0))
                    .cloned()
            }
            LoadBalancingStrategy::Random => {
                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as usize;
                Some(healthy[seed % healthy.len()].clone())
            }
            LoadBalancingStrategy::Weighted => {
                let total_weight: u32 = healthy.iter().map(|n| n.weight).sum();
                if total_weight == 0 {
                    return healthy.first().cloned().cloned();
                }

                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u32;
                let target = seed % total_weight;

                let mut cumulative = 0u32;
                for node in &healthy {
                    cumulative += node.weight;
                    if cumulative > target {
                        return Some((*node).clone());
                    }
                }
                healthy.last().cloned().cloned()
            }
            LoadBalancingStrategy::LatencyBased => {
                // In real impl, would track latency per node
                // For now, use replication lag as proxy
                healthy
                    .into_iter()
                    .min_by_key(|n| n.replication_lag_ms)
                    .cloned()
            }
        }
    }

    /// Get node for writes (leader only)
    pub fn get_write_node(&self) -> Option<NodeInfo> {
        crate::lock_util::read_lock(&self.nodes)
            .iter()
            .find(|n| n.is_writable())
            .cloned()
    }

    /// Increment connection count
    pub fn increment_connections(&self, node_id: &str) {
        let mut counts = crate::lock_util::write_lock(&self.connection_counts);
        *counts.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Decrement connection count
    pub fn decrement_connections(&self, node_id: &str) {
        let mut counts = crate::lock_util::write_lock(&self.connection_counts);
        if let Some(count) = counts.get_mut(node_id) {
            *count = count.saturating_sub(1);
        }
    }
}

// ============================================================================
// Failover Manager
// ============================================================================

/// Failover Manager
pub struct FailoverManager {
    replication_manager: Arc<ReplicationManager>,
    failover_in_progress: Arc<RwLock<bool>>,
    last_failover: Arc<RwLock<Option<u64>>>,
}

impl FailoverManager {
    /// Create new failover manager
    pub fn new(replication_manager: Arc<ReplicationManager>) -> Self {
        Self {
            replication_manager,
            failover_in_progress: Arc::new(RwLock::new(false)),
            last_failover: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if failover is needed
    pub fn check_failover_needed(&self) -> bool {
        // Check if leader is healthy
        if let Some(leader) = self.replication_manager.get_leader() {
            if leader.health == NodeHealth::Unhealthy {
                return true;
            }
        } else {
            // No leader - failover needed
            return true;
        }
        false
    }

    /// Trigger failover
    pub fn trigger_failover(&self) -> Result<Option<NodeInfo>, EnterpriseError> {
        // Check if already in progress
        {
            let mut in_progress = crate::lock_util::write_lock(&self.failover_in_progress);
            if *in_progress {
                return Err(EnterpriseError::LeaderElectionFailed(
                    "Failover already in progress".to_string(),
                ));
            }
            *in_progress = true;
        }

        // Select best candidate
        let candidates = self.replication_manager.healthy_nodes();
        let best = candidates
            .into_iter()
            .filter(|n| n.role == NodeRole::Follower)
            .min_by_key(|n| n.replication_lag_ms);

        if let Some(new_leader) = best {
            // In real impl, would coordinate with other nodes
            // For now, just update local state
            *crate::lock_util::write_lock(&self.failover_in_progress) = false;
            *crate::lock_util::write_lock(&self.last_failover) =
                Some(ReplicationManager::current_timestamp());
            Ok(Some(new_leader))
        } else {
            *crate::lock_util::write_lock(&self.failover_in_progress) = false;
            Err(EnterpriseError::LeaderElectionFailed(
                "No healthy candidates".to_string(),
            ))
        }
    }

    /// Get time since last failover
    pub fn time_since_last_failover(&self) -> Option<Duration> {
        let last = crate::lock_util::read_lock(&self.last_failover);
        last.map(|ts| {
            let now = ReplicationManager::current_timestamp();
            Duration::from_secs(now - ts)
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_info() {
        let node = NodeInfo::new("node1", "127.0.0.1:8080");
        assert_eq!(node.id, "node1");
        assert_eq!(node.role, NodeRole::Follower);
        assert_eq!(node.health, NodeHealth::Unknown);
        assert!(!node.is_writable());
    }

    #[test]
    fn test_replication_manager() {
        let config = ClusterConfig::default();
        let manager = ReplicationManager::new("node1", config);

        // Register nodes
        let mut node1 = NodeInfo::new("node1", "127.0.0.1:8080");
        node1.role = NodeRole::Leader;
        node1.health = NodeHealth::Healthy;
        manager.register_node(node1);

        let mut node2 = NodeInfo::new("node2", "127.0.0.1:8081");
        node2.health = NodeHealth::Healthy;
        manager.register_node(node2);

        assert_eq!(manager.list_nodes().len(), 2);
        assert_eq!(manager.healthy_nodes().len(), 2);
    }

    #[test]
    fn test_append_operation() {
        let config = ClusterConfig::default();
        let manager = ReplicationManager::new("node1", config);

        let seq1 = manager.append_operation(ReplicationOperation::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        });

        let seq2 = manager.append_operation(ReplicationOperation::Delete {
            key: b"key2".to_vec(),
        });

        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
        assert_eq!(manager.last_sequence(), 2);
    }

    #[test]
    fn test_shard_manager() {
        let config = ClusterConfig::default();
        let manager = ShardManager::new(config);

        manager.initialize_shards(4);
        assert_eq!(manager.list_shards().len(), 4);

        // Test key routing
        let shard = manager.get_shard_for_key(b"test_key").unwrap();
        assert!(shard.id.starts_with("shard_"));
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let nodes = vec![
            NodeInfo {
                id: "node1".to_string(),
                address: "127.0.0.1:8080".to_string(),
                role: NodeRole::Follower,
                health: NodeHealth::Healthy,
                last_heartbeat: 0,
                replication_lag_ms: 0,
                weight: 1,
                region: None,
                metadata: HashMap::new(),
            },
            NodeInfo {
                id: "node2".to_string(),
                address: "127.0.0.1:8081".to_string(),
                role: NodeRole::Follower,
                health: NodeHealth::Healthy,
                last_heartbeat: 0,
                replication_lag_ms: 0,
                weight: 1,
                region: None,
                metadata: HashMap::new(),
            },
        ];

        lb.update_nodes(nodes);

        let first = lb.get_node().unwrap().id;
        let second = lb.get_node().unwrap().id;
        let third = lb.get_node().unwrap().id;

        // Should cycle through nodes
        assert_eq!(first, "node1");
        assert_eq!(second, "node2");
        assert_eq!(third, "node1");
    }

    #[test]
    fn test_failover_manager() {
        let config = ClusterConfig::default();
        let replication_manager = Arc::new(ReplicationManager::new("node1", config));
        let failover_manager = FailoverManager::new(replication_manager.clone());

        // No leader initially
        assert!(failover_manager.check_failover_needed());

        // Add healthy leader
        let mut leader = NodeInfo::new("node1", "127.0.0.1:8080");
        leader.role = NodeRole::Leader;
        leader.health = NodeHealth::Healthy;
        replication_manager.register_node(leader);
        *crate::lock_util::write_lock(&replication_manager.current_leader) =
            Some("node1".to_string());

        assert!(!failover_manager.check_failover_needed());
    }

    #[test]
    fn test_cluster_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.num_shards, 16);
        assert_eq!(config.target_replication_factor, 3);
        assert_eq!(config.replication_mode, ReplicationMode::SemiSync);
    }
}
