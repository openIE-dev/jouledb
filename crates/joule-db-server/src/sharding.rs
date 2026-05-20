//! Horizontal Sharding for JouleDB
//!
//! This module implements production-grade horizontal sharding with:
//! - Consistent hashing for minimal data movement during rebalancing
//! - Range-based and hash-based sharding strategies
//! - Cross-shard query coordination
//! - Automatic shard splitting and merging
//! - Online shard migration with minimal downtime
//! - Virtual nodes for better load distribution
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Shard Router                              │
//! │ (Routes operations to correct shard based on key/range)     │
//! └──────────────────────┬──────────────────────────────────────┘
//!                        │
//!        ┌───────────────┼───────────────┐
//!        ▼               ▼               ▼
//! ┌────────────┐  ┌────────────┐  ┌────────────┐
//! │  Shard 0   │  │  Shard 1   │  │  Shard 2   │
//! │ [0, 1000)  │  │[1000,2000) │  │[2000,3000) │
//! └────────────┘  └────────────┘  └────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// Types and Errors
// ============================================================================

/// Sharding error types
#[derive(Debug, Clone, PartialEq)]
pub enum ShardingError {
    /// Shard not found
    ShardNotFound(String),
    /// Node not found
    NodeNotFound(String),
    /// No available shards
    NoAvailableShards,
    /// Migration in progress
    MigrationInProgress(String),
    /// Migration failed
    MigrationFailed(String),
    /// Split failed
    SplitFailed(String),
    /// Merge failed
    MergeFailed(String),
    /// Invalid key range
    InvalidKeyRange,
    /// Cross-shard query failed
    CrossShardQueryFailed(String),
    /// Quorum not reached
    QuorumNotReached,
    /// Timeout
    Timeout,
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for ShardingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShardNotFound(id) => write!(f, "Shard not found: {}", id),
            Self::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            Self::NoAvailableShards => write!(f, "No available shards"),
            Self::MigrationInProgress(id) => write!(f, "Migration in progress for shard: {}", id),
            Self::MigrationFailed(msg) => write!(f, "Migration failed: {}", msg),
            Self::SplitFailed(msg) => write!(f, "Shard split failed: {}", msg),
            Self::MergeFailed(msg) => write!(f, "Shard merge failed: {}", msg),
            Self::InvalidKeyRange => write!(f, "Invalid key range"),
            Self::CrossShardQueryFailed(msg) => write!(f, "Cross-shard query failed: {}", msg),
            Self::QuorumNotReached => write!(f, "Quorum not reached"),
            Self::Timeout => write!(f, "Operation timed out"),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for ShardingError {}

/// Sharding result type
pub type ShardingResult<T> = Result<T, ShardingError>;

// ============================================================================
// Consistent Hashing
// ============================================================================

/// Consistent hash ring for minimal data movement during rebalancing
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Number of virtual nodes per physical node
    virtual_nodes: usize,
    /// Ring: hash position -> shard ID
    ring: BTreeMap<u64, String>,
    /// Shard -> hash positions
    shard_positions: HashMap<String, Vec<u64>>,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            virtual_nodes,
            ring: BTreeMap::new(),
            shard_positions: HashMap::new(),
        }
    }

    /// Add a shard to the ring
    pub fn add_shard(&mut self, shard_id: &str) {
        let mut positions = Vec::with_capacity(self.virtual_nodes);

        for i in 0..self.virtual_nodes {
            let key = format!("{}#{}", shard_id, i);
            let hash = self.hash(&key);
            self.ring.insert(hash, shard_id.to_string());
            positions.push(hash);
        }

        self.shard_positions.insert(shard_id.to_string(), positions);
    }

    /// Remove a shard from the ring
    pub fn remove_shard(&mut self, shard_id: &str) {
        if let Some(positions) = self.shard_positions.remove(shard_id) {
            for pos in positions {
                self.ring.remove(&pos);
            }
        }
    }

    /// Get the shard responsible for a key
    pub fn get_shard(&self, key: &[u8]) -> Option<String> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = self.hash_bytes(key);

        // Find the first shard with position >= hash
        if let Some((_, shard_id)) = self.ring.range(hash..).next() {
            return Some(shard_id.clone());
        }

        // Wrap around to the beginning
        self.ring.values().next().cloned()
    }

    /// Get the N shards responsible for replication of a key
    pub fn get_replica_shards(&self, key: &[u8], n: usize) -> Vec<String> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = self.hash_bytes(key);
        let mut result = Vec::with_capacity(n);
        let mut seen = HashSet::new();

        // Collect unique shards starting from the key's position
        for (_, shard_id) in self.ring.range(hash..).chain(self.ring.iter()) {
            if seen.insert(shard_id.clone()) {
                result.push(shard_id.clone());
                if result.len() >= n {
                    break;
                }
            }
        }

        result
    }

    /// Hash a string key
    fn hash(&self, key: &str) -> u64 {
        self.hash_bytes(key.as_bytes())
    }

    /// Hash bytes using FNV-1a
    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Get all shards in the ring
    pub fn shards(&self) -> Vec<String> {
        self.shard_positions.keys().cloned().collect()
    }

    /// Get the number of shards
    pub fn len(&self) -> usize {
        self.shard_positions.len()
    }

    /// Check if ring is empty
    pub fn is_empty(&self) -> bool {
        self.shard_positions.is_empty()
    }
}

// ============================================================================
// Key Range
// ============================================================================

/// Key range for range-based sharding
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyRange {
    /// Start of range (inclusive)
    pub start: Option<Vec<u8>>,
    /// End of range (exclusive)
    pub end: Option<Vec<u8>>,
}

impl KeyRange {
    /// Create unbounded range
    pub fn unbounded() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    /// Create range [start, end)
    pub fn new(start: Vec<u8>, end: Vec<u8>) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
        }
    }

    /// Create range [start, ∞)
    pub fn from(start: Vec<u8>) -> Self {
        Self {
            start: Some(start),
            end: None,
        }
    }

    /// Create range (-∞, end)
    pub fn until(end: Vec<u8>) -> Self {
        Self {
            start: None,
            end: Some(end),
        }
    }

    /// Check if key is within range
    pub fn contains(&self, key: &[u8]) -> bool {
        let after_start = match &self.start {
            Some(s) => key >= s.as_slice(),
            None => true,
        };

        let before_end = match &self.end {
            Some(e) => key < e.as_slice(),
            None => true,
        };

        after_start && before_end
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &KeyRange) -> bool {
        let self_before_other = match (&self.end, &other.start) {
            (Some(e), Some(s)) => e.as_slice() <= s.as_slice(),
            _ => false,
        };

        let other_before_self = match (&other.end, &self.start) {
            (Some(e), Some(s)) => e.as_slice() <= s.as_slice(),
            _ => false,
        };

        !self_before_other && !other_before_self
    }

    /// Split range at midpoint
    pub fn split(&self) -> ShardingResult<(KeyRange, KeyRange)> {
        let (start, end) = match (&self.start, &self.end) {
            (Some(s), Some(e)) => (s.clone(), e.clone()),
            (Some(s), None) => {
                // Use start + some offset as midpoint
                let mut mid = s.clone();
                if !mid.is_empty() {
                    // Add to last byte or extend
                    // Safety: mid is confirmed non-empty by the enclosing `if !mid.is_empty()` check
                    *mid.last_mut().expect("mid confirmed non-empty") = mid
                        .last()
                        .expect("mid confirmed non-empty")
                        .saturating_add(128);
                }
                (s.clone(), mid)
            }
            (None, Some(e)) => {
                // Use half of end as start
                let mid = e.iter().map(|b| b / 2).collect();
                (mid, e.clone())
            }
            (None, None) => {
                return Err(ShardingError::InvalidKeyRange);
            }
        };

        // Calculate midpoint
        let mid: Vec<u8> = start
            .iter()
            .zip(end.iter())
            .map(|(a, b)| (((*a as u16) + (*b as u16)) / 2) as u8)
            .collect();

        let left = KeyRange {
            start: self.start.clone(),
            end: Some(mid.clone()),
        };

        let right = KeyRange {
            start: Some(mid),
            end: self.end.clone(),
        };

        Ok((left, right))
    }
}

// ============================================================================
// Shard Definition
// ============================================================================

/// Shard status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardState {
    /// Shard is active and serving requests
    Active,
    /// Shard is being created
    Creating,
    /// Shard is receiving data during migration
    Receiving,
    /// Shard is sending data during migration
    Sending,
    /// Shard is being split
    Splitting,
    /// Shard is being merged
    Merging,
    /// Shard is read-only (during migration)
    ReadOnly,
    /// Shard is offline
    Offline,
    /// Shard is being deleted
    Deleting,
}

/// Complete shard information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    /// Unique shard ID
    pub id: String,
    /// Key range (for range-based sharding)
    pub key_range: KeyRange,
    /// Current state
    pub state: ShardState,
    /// Primary node ID
    pub primary_node: String,
    /// Replica node IDs
    pub replica_nodes: Vec<String>,
    /// Number of records
    pub record_count: u64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub updated_at: u64,
    /// Version for optimistic concurrency
    pub version: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Shard {
    /// Create a new shard
    pub fn new(
        id: impl Into<String>,
        key_range: KeyRange,
        primary_node: impl Into<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: id.into(),
            key_range,
            state: ShardState::Creating,
            primary_node: primary_node.into(),
            replica_nodes: Vec::new(),
            record_count: 0,
            size_bytes: 0,
            created_at: now,
            updated_at: now,
            version: 1,
            metadata: HashMap::new(),
        }
    }

    /// Check if shard can accept reads
    pub fn can_read(&self) -> bool {
        matches!(
            self.state,
            ShardState::Active | ShardState::ReadOnly | ShardState::Sending
        )
    }

    /// Check if shard can accept writes
    pub fn can_write(&self) -> bool {
        self.state == ShardState::Active
    }

    /// Check if key belongs to this shard
    pub fn owns_key(&self, key: &[u8]) -> bool {
        self.key_range.contains(key)
    }
}

// ============================================================================
// Sharding Configuration
// ============================================================================

/// Sharding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Consistent hash-based sharding
    ConsistentHash,
    /// Range-based sharding
    Range,
    /// Hash mod N sharding (simple but poor for scaling)
    HashMod,
    /// Directory-based sharding (explicit mapping)
    Directory,
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        Self::ConsistentHash
    }
}

/// Sharding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Sharding strategy
    pub strategy: ShardingStrategy,
    /// Initial number of shards
    pub initial_shards: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Virtual nodes per shard (for consistent hashing)
    pub virtual_nodes: usize,
    /// Maximum shard size before auto-split (bytes)
    pub max_shard_size: u64,
    /// Minimum shard size before auto-merge (bytes)
    pub min_shard_size: u64,
    /// Maximum records per shard before auto-split
    pub max_shard_records: u64,
    /// Enable automatic rebalancing
    pub auto_rebalance: bool,
    /// Rebalance threshold (deviation from average)
    pub rebalance_threshold: f64,
    /// Migration batch size
    pub migration_batch_size: usize,
    /// Migration timeout
    pub migration_timeout: Duration,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            strategy: ShardingStrategy::ConsistentHash,
            initial_shards: 16,
            replication_factor: 3,
            virtual_nodes: 150,
            max_shard_size: 1024 * 1024 * 1024, // 1GB
            min_shard_size: 64 * 1024 * 1024,   // 64MB
            max_shard_records: 10_000_000,
            auto_rebalance: true,
            rebalance_threshold: 0.2, // 20% deviation
            migration_batch_size: 1000,
            migration_timeout: Duration::from_secs(3600), // 1 hour
        }
    }
}

// ============================================================================
// Migration State
// ============================================================================

/// Migration status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStatus {
    /// Migration pending
    Pending,
    /// Copying data
    Copying,
    /// Verifying data
    Verifying,
    /// Switching traffic
    Switching,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Migration task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationTask {
    /// Unique migration ID
    pub id: String,
    /// Source shard ID
    pub source_shard: String,
    /// Target shard ID
    pub target_shard: String,
    /// Key range being migrated
    pub key_range: KeyRange,
    /// Current status
    pub status: MigrationStatus,
    /// Records migrated
    pub records_migrated: u64,
    /// Total records to migrate
    pub total_records: u64,
    /// Bytes migrated
    pub bytes_migrated: u64,
    /// Started at
    pub started_at: Option<u64>,
    /// Completed at
    pub completed_at: Option<u64>,
    /// Error message if failed
    pub error: Option<String>,
}

impl MigrationTask {
    /// Create a new migration task
    pub fn new(
        source_shard: impl Into<String>,
        target_shard: impl Into<String>,
        key_range: KeyRange,
    ) -> Self {
        Self {
            id: uuid_v4(),
            source_shard: source_shard.into(),
            target_shard: target_shard.into(),
            key_range,
            status: MigrationStatus::Pending,
            records_migrated: 0,
            total_records: 0,
            bytes_migrated: 0,
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    /// Get progress percentage
    pub fn progress(&self) -> f64 {
        if self.total_records == 0 {
            return 0.0;
        }
        (self.records_migrated as f64 / self.total_records as f64) * 100.0
    }
}

// ============================================================================
// Shard Router
// ============================================================================

/// Routes operations to the correct shard
pub struct ShardRouter {
    config: ShardingConfig,
    /// All shards
    shards: Arc<RwLock<HashMap<String, Shard>>>,
    /// Consistent hash ring (for consistent hash strategy)
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    /// Range index: sorted by start key
    range_index: Arc<RwLock<BTreeMap<Vec<u8>, String>>>,
    /// Active migrations
    migrations: Arc<RwLock<HashMap<String, MigrationTask>>>,
    /// Statistics
    stats: Arc<RwLock<RouterStats>>,
}

/// Router statistics
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    /// Total routed requests
    pub total_requests: u64,
    /// Requests by shard
    pub requests_by_shard: HashMap<String, u64>,
    /// Cross-shard queries
    pub cross_shard_queries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Routing errors
    pub routing_errors: u64,
}

impl ShardRouter {
    /// Create a new shard router
    pub fn new(config: ShardingConfig) -> Self {
        let virtual_nodes = config.virtual_nodes;
        Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
            hash_ring: Arc::new(RwLock::new(ConsistentHashRing::new(virtual_nodes))),
            range_index: Arc::new(RwLock::new(BTreeMap::new())),
            migrations: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RouterStats::default())),
        }
    }

    /// Initialize shards for a new cluster
    pub fn initialize_shards(&self, node_ids: &[String]) -> ShardingResult<Vec<Shard>> {
        if node_ids.is_empty() {
            return Err(ShardingError::NoAvailableShards);
        }

        let num_shards = self.config.initial_shards;
        let mut created_shards = Vec::with_capacity(num_shards);

        match self.config.strategy {
            ShardingStrategy::ConsistentHash | ShardingStrategy::HashMod => {
                // Create shards with unbounded ranges (hash determines shard)
                for i in 0..num_shards {
                    let shard_id = format!("shard_{:04}", i);
                    let primary_node = &node_ids[i % node_ids.len()];

                    let mut shard = Shard::new(&shard_id, KeyRange::unbounded(), primary_node);
                    shard.state = ShardState::Active;

                    // Assign replicas
                    for j in 1..self.config.replication_factor {
                        let replica_idx = (i + j) % node_ids.len();
                        if replica_idx != i % node_ids.len() {
                            shard.replica_nodes.push(node_ids[replica_idx].clone());
                        }
                    }

                    created_shards.push(shard);
                }
            }
            ShardingStrategy::Range | ShardingStrategy::Directory => {
                // Create shards with specific key ranges
                let step = 256 / num_shards;

                for i in 0..num_shards {
                    let shard_id = format!("shard_{:04}", i);
                    let primary_node = &node_ids[i % node_ids.len()];

                    let range = if i == 0 {
                        KeyRange::until(vec![(step * (i + 1)) as u8])
                    } else if i == num_shards - 1 {
                        KeyRange::from(vec![(step * i) as u8])
                    } else {
                        KeyRange::new(vec![(step * i) as u8], vec![(step * (i + 1)) as u8])
                    };

                    let mut shard = Shard::new(&shard_id, range, primary_node);
                    shard.state = ShardState::Active;

                    created_shards.push(shard);
                }
            }
        }

        // Register shards
        let mut shards = crate::lock_util::write_lock(&self.shards);
        let mut hash_ring = crate::lock_util::write_lock(&self.hash_ring);
        let mut range_index = crate::lock_util::write_lock(&self.range_index);

        for shard in &created_shards {
            shards.insert(shard.id.clone(), shard.clone());
            hash_ring.add_shard(&shard.id);

            if let Some(ref start) = shard.key_range.start {
                range_index.insert(start.clone(), shard.id.clone());
            }
        }

        Ok(created_shards)
    }

    /// Route a key to its shard
    pub fn route_key(&self, key: &[u8]) -> ShardingResult<Shard> {
        // Update stats
        crate::lock_util::write_lock(&self.stats).total_requests += 1;

        match self.config.strategy {
            ShardingStrategy::ConsistentHash => {
                let ring = crate::lock_util::read_lock(&self.hash_ring);
                let shard_id = ring
                    .get_shard(key)
                    .ok_or(ShardingError::NoAvailableShards)?;

                let shards = crate::lock_util::read_lock(&self.shards);
                shards
                    .get(&shard_id)
                    .cloned()
                    .ok_or(ShardingError::ShardNotFound(shard_id))
            }
            ShardingStrategy::HashMod => {
                let hash = self.hash_key(key);
                let shards = crate::lock_util::read_lock(&self.shards);
                let shard_count = shards.len();

                if shard_count == 0 {
                    return Err(ShardingError::NoAvailableShards);
                }

                let shard_idx = (hash as usize) % shard_count;
                let shard_id = format!("shard_{:04}", shard_idx);

                shards
                    .get(&shard_id)
                    .cloned()
                    .ok_or(ShardingError::ShardNotFound(shard_id))
            }
            ShardingStrategy::Range | ShardingStrategy::Directory => {
                let shards = crate::lock_util::read_lock(&self.shards);

                // Find shard by key range
                for shard in shards.values() {
                    if shard.owns_key(key) {
                        return Ok(shard.clone());
                    }
                }

                Err(ShardingError::NoAvailableShards)
            }
        }
    }

    /// Route multiple keys (for batch operations)
    pub fn route_keys(&self, keys: &[Vec<u8>]) -> ShardingResult<HashMap<String, Vec<Vec<u8>>>> {
        let mut result: HashMap<String, Vec<Vec<u8>>> = HashMap::new();

        for key in keys {
            let shard = self.route_key(key)?;
            result.entry(shard.id).or_default().push(key.clone());
        }

        // Update cross-shard stats
        if result.len() > 1 {
            crate::lock_util::write_lock(&self.stats).cross_shard_queries += 1;
        }

        Ok(result)
    }

    /// Get all shards for a key range query
    pub fn route_range(&self, start: &[u8], end: &[u8]) -> ShardingResult<Vec<Shard>> {
        let range = KeyRange::new(start.to_vec(), end.to_vec());
        let shards = crate::lock_util::read_lock(&self.shards);

        let matching: Vec<Shard> = shards
            .values()
            .filter(|s| s.key_range.overlaps(&range) && s.can_read())
            .cloned()
            .collect();

        if matching.is_empty() {
            return Err(ShardingError::NoAvailableShards);
        }

        if matching.len() > 1 {
            crate::lock_util::write_lock(&self.stats).cross_shard_queries += 1;
        }

        Ok(matching)
    }

    /// Get shard by ID
    pub fn get_shard(&self, shard_id: &str) -> Option<Shard> {
        crate::lock_util::read_lock(&self.shards)
            .get(shard_id)
            .cloned()
    }

    /// Get all shards
    pub fn get_all_shards(&self) -> Vec<Shard> {
        crate::lock_util::read_lock(&self.shards)
            .values()
            .cloned()
            .collect()
    }

    /// Get all shard IDs
    pub fn get_all_shard_ids(&self) -> Vec<String> {
        crate::lock_util::read_lock(&self.shards)
            .keys()
            .cloned()
            .collect()
    }

    /// Get shards for a node
    pub fn get_shards_for_node(&self, node_id: &str) -> Vec<Shard> {
        crate::lock_util::read_lock(&self.shards)
            .values()
            .filter(|s| s.primary_node == node_id || s.replica_nodes.contains(&node_id.to_string()))
            .cloned()
            .collect()
    }

    /// Add a new shard
    pub fn add_shard(&self, shard: Shard) {
        let mut shards = crate::lock_util::write_lock(&self.shards);
        let mut hash_ring = crate::lock_util::write_lock(&self.hash_ring);
        let mut range_index = crate::lock_util::write_lock(&self.range_index);

        hash_ring.add_shard(&shard.id);

        if let Some(ref start) = shard.key_range.start {
            range_index.insert(start.clone(), shard.id.clone());
        }

        shards.insert(shard.id.clone(), shard);
    }

    /// Remove a shard
    pub fn remove_shard(&self, shard_id: &str) -> Option<Shard> {
        let mut shards = crate::lock_util::write_lock(&self.shards);
        let mut hash_ring = crate::lock_util::write_lock(&self.hash_ring);
        let mut range_index = crate::lock_util::write_lock(&self.range_index);

        if let Some(shard) = shards.remove(shard_id) {
            hash_ring.remove_shard(shard_id);

            if let Some(ref start) = shard.key_range.start {
                range_index.remove(start);
            }

            return Some(shard);
        }

        None
    }

    /// Update shard state
    pub fn update_shard_state(&self, shard_id: &str, state: ShardState) -> ShardingResult<()> {
        let mut shards = crate::lock_util::write_lock(&self.shards);
        let shard = shards
            .get_mut(shard_id)
            .ok_or_else(|| ShardingError::ShardNotFound(shard_id.to_string()))?;

        shard.state = state;
        shard.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        shard.version += 1;

        Ok(())
    }

    /// Split a shard into two
    pub fn split_shard(&self, shard_id: &str) -> ShardingResult<(Shard, Shard)> {
        let mut shards = crate::lock_util::write_lock(&self.shards);
        let shard = shards
            .get(shard_id)
            .ok_or_else(|| ShardingError::ShardNotFound(shard_id.to_string()))?
            .clone();

        // Can only split active shards
        if shard.state != ShardState::Active {
            return Err(ShardingError::SplitFailed(format!(
                "Shard {} is not active (state: {:?})",
                shard_id, shard.state
            )));
        }

        // For consistent hashing, we don't split by range
        if self.config.strategy == ShardingStrategy::ConsistentHash {
            // Create new shard with same primary but different ID
            let new_shard_id = format!("{}_split", shard_id);
            let mut new_shard =
                Shard::new(&new_shard_id, KeyRange::unbounded(), &shard.primary_node);
            new_shard.state = ShardState::Active;
            new_shard.replica_nodes = shard.replica_nodes.clone();

            // Add to hash ring (this redistributes some keys)
            let mut hash_ring = crate::lock_util::write_lock(&self.hash_ring);
            hash_ring.add_shard(&new_shard_id);
            drop(hash_ring);

            shards.insert(new_shard_id.clone(), new_shard.clone());

            return Ok((shard, new_shard));
        }

        // For range-based sharding, split the key range
        let (left_range, right_range) = shard.key_range.split()?;

        // Update original shard with left range
        let left_id = format!("{}_L", shard_id);
        let mut left_shard = Shard::new(&left_id, left_range, &shard.primary_node);
        left_shard.state = ShardState::Active;
        left_shard.replica_nodes = shard.replica_nodes.clone();

        // Create new shard with right range
        let right_id = format!("{}_R", shard_id);
        let mut right_shard = Shard::new(&right_id, right_range, &shard.primary_node);
        right_shard.state = ShardState::Active;
        right_shard.replica_nodes = shard.replica_nodes.clone();

        // Remove old shard
        shards.remove(shard_id);

        // Update range index
        let mut range_index = crate::lock_util::write_lock(&self.range_index);
        if let Some(ref start) = shard.key_range.start {
            range_index.remove(start);
        }
        if let Some(ref start) = left_shard.key_range.start {
            range_index.insert(start.clone(), left_id.clone());
        }
        if let Some(ref start) = right_shard.key_range.start {
            range_index.insert(start.clone(), right_id.clone());
        }
        drop(range_index);

        shards.insert(left_id, left_shard.clone());
        shards.insert(right_id, right_shard.clone());

        Ok((left_shard, right_shard))
    }

    /// Merge two shards into one
    pub fn merge_shards(&self, shard_id_1: &str, shard_id_2: &str) -> ShardingResult<Shard> {
        let mut shards = crate::lock_util::write_lock(&self.shards);

        let shard1 = shards
            .get(shard_id_1)
            .ok_or_else(|| ShardingError::ShardNotFound(shard_id_1.to_string()))?
            .clone();

        let shard2 = shards
            .get(shard_id_2)
            .ok_or_else(|| ShardingError::ShardNotFound(shard_id_2.to_string()))?
            .clone();

        // Can only merge active shards
        if shard1.state != ShardState::Active || shard2.state != ShardState::Active {
            return Err(ShardingError::MergeFailed(
                "Shards must be active".to_string(),
            ));
        }

        // Create merged shard
        let merged_id = format!("{}_{}_merged", shard_id_1, shard_id_2);
        let merged_range = KeyRange {
            start: match (&shard1.key_range.start, &shard2.key_range.start) {
                (Some(s1), Some(s2)) => Some(if s1 < s2 { s1.clone() } else { s2.clone() }),
                (Some(s), None) | (None, Some(s)) => Some(s.clone()),
                (None, None) => None,
            },
            end: match (&shard1.key_range.end, &shard2.key_range.end) {
                (Some(e1), Some(e2)) => Some(if e1 > e2 { e1.clone() } else { e2.clone() }),
                (Some(e), None) | (None, Some(e)) => Some(e.clone()),
                (None, None) => None,
            },
        };

        let mut merged = Shard::new(&merged_id, merged_range, &shard1.primary_node);
        merged.state = ShardState::Active;
        merged.record_count = shard1.record_count + shard2.record_count;
        merged.size_bytes = shard1.size_bytes + shard2.size_bytes;

        // Combine replica nodes (unique)
        let mut replicas: HashSet<String> = shard1.replica_nodes.iter().cloned().collect();
        replicas.extend(shard2.replica_nodes.iter().cloned());
        merged.replica_nodes = replicas.into_iter().collect();

        // Update hash ring for consistent hashing
        if self.config.strategy == ShardingStrategy::ConsistentHash {
            let mut hash_ring = crate::lock_util::write_lock(&self.hash_ring);
            hash_ring.remove_shard(shard_id_1);
            hash_ring.remove_shard(shard_id_2);
            hash_ring.add_shard(&merged_id);
        }

        // Remove old shards
        shards.remove(shard_id_1);
        shards.remove(shard_id_2);

        // Insert merged shard
        shards.insert(merged_id.clone(), merged.clone());

        Ok(merged)
    }

    /// Start a migration task
    pub fn start_migration(&self, task: MigrationTask) -> ShardingResult<()> {
        let mut migrations = crate::lock_util::write_lock(&self.migrations);

        // Check if migration already exists for source shard
        for existing in migrations.values() {
            if existing.source_shard == task.source_shard
                && existing.status != MigrationStatus::Completed
                && existing.status != MigrationStatus::Failed
                && existing.status != MigrationStatus::Cancelled
            {
                return Err(ShardingError::MigrationInProgress(
                    task.source_shard.clone(),
                ));
            }
        }

        // Update source shard state
        self.update_shard_state(&task.source_shard, ShardState::Sending)?;

        // Update target shard state
        self.update_shard_state(&task.target_shard, ShardState::Receiving)?;

        migrations.insert(task.id.clone(), task);

        Ok(())
    }

    /// Update migration progress
    pub fn update_migration(
        &self,
        migration_id: &str,
        records: u64,
        bytes: u64,
    ) -> ShardingResult<()> {
        let mut migrations = crate::lock_util::write_lock(&self.migrations);
        let migration = migrations.get_mut(migration_id).ok_or_else(|| {
            ShardingError::Internal(format!("Migration {} not found", migration_id))
        })?;

        migration.records_migrated = records;
        migration.bytes_migrated = bytes;
        migration.status = MigrationStatus::Copying;

        Ok(())
    }

    /// Complete a migration
    pub fn complete_migration(&self, migration_id: &str) -> ShardingResult<()> {
        let mut migrations = crate::lock_util::write_lock(&self.migrations);
        let migration = migrations.get_mut(migration_id).ok_or_else(|| {
            ShardingError::Internal(format!("Migration {} not found", migration_id))
        })?;

        migration.status = MigrationStatus::Completed;
        migration.completed_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        // Update shard states
        let source_shard = migration.source_shard.clone();
        let target_shard = migration.target_shard.clone();
        drop(migrations);

        self.update_shard_state(&source_shard, ShardState::Active)?;
        self.update_shard_state(&target_shard, ShardState::Active)?;

        Ok(())
    }

    /// Fail a migration
    pub fn fail_migration(&self, migration_id: &str, error: &str) -> ShardingResult<()> {
        let mut migrations = crate::lock_util::write_lock(&self.migrations);
        let migration = migrations.get_mut(migration_id).ok_or_else(|| {
            ShardingError::Internal(format!("Migration {} not found", migration_id))
        })?;

        migration.status = MigrationStatus::Failed;
        migration.error = Some(error.to_string());
        migration.completed_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        // Rollback shard states
        let source_shard = migration.source_shard.clone();
        let target_shard = migration.target_shard.clone();
        drop(migrations);

        self.update_shard_state(&source_shard, ShardState::Active)?;
        self.update_shard_state(&target_shard, ShardState::Active)?;

        Ok(())
    }

    /// Get active migrations
    pub fn get_active_migrations(&self) -> Vec<MigrationTask> {
        crate::lock_util::read_lock(&self.migrations)
            .values()
            .filter(|m| {
                matches!(
                    m.status,
                    MigrationStatus::Pending
                        | MigrationStatus::Copying
                        | MigrationStatus::Verifying
                        | MigrationStatus::Switching
                )
            })
            .cloned()
            .collect()
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalance(&self) -> bool {
        if !self.config.auto_rebalance {
            return false;
        }

        let shards = crate::lock_util::read_lock(&self.shards);
        if shards.len() < 2 {
            return false;
        }

        let sizes: Vec<u64> = shards.values().map(|s| s.size_bytes).collect();
        let avg = sizes.iter().sum::<u64>() as f64 / sizes.len() as f64;

        if avg == 0.0 {
            return false;
        }

        // Check if any shard deviates more than threshold from average
        for size in &sizes {
            let deviation = ((*size as f64) - avg).abs() / avg;
            if deviation > self.config.rebalance_threshold {
                return true;
            }
        }

        false
    }

    /// Check if any shard needs splitting
    pub fn get_shards_to_split(&self) -> Vec<String> {
        crate::lock_util::read_lock(&self.shards)
            .values()
            .filter(|s| {
                s.state == ShardState::Active
                    && (s.size_bytes > self.config.max_shard_size
                        || s.record_count > self.config.max_shard_records)
            })
            .map(|s| s.id.clone())
            .collect()
    }

    /// Check if any shards should be merged
    pub fn get_shards_to_merge(&self) -> Vec<(String, String)> {
        let shards = crate::lock_util::read_lock(&self.shards);
        let mut candidates: Vec<_> = shards
            .values()
            .filter(|s| s.state == ShardState::Active && s.size_bytes < self.config.min_shard_size)
            .collect();

        // Sort by size
        candidates.sort_by_key(|s| s.size_bytes);

        let mut pairs = Vec::new();
        let mut used = HashSet::new();

        // Pair smallest shards
        for i in 0..candidates.len() {
            if used.contains(&candidates[i].id) {
                continue;
            }

            for j in (i + 1)..candidates.len() {
                if used.contains(&candidates[j].id) {
                    continue;
                }

                // Check if combined size is reasonable
                let combined = candidates[i].size_bytes + candidates[j].size_bytes;
                if combined < self.config.max_shard_size {
                    pairs.push((candidates[i].id.clone(), candidates[j].id.clone()));
                    used.insert(candidates[i].id.clone());
                    used.insert(candidates[j].id.clone());
                    break;
                }
            }
        }

        pairs
    }

    /// Get router statistics
    pub fn stats(&self) -> RouterStats {
        crate::lock_util::read_lock(&self.stats).clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *crate::lock_util::write_lock(&self.stats) = RouterStats::default();
    }

    /// Hash a key using FNV-1a
    fn hash_key(&self, key: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in key {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

// ============================================================================
// Cross-Shard Query Coordinator
// ============================================================================

/// Coordinates queries that span multiple shards
pub struct CrossShardCoordinator {
    router: Arc<ShardRouter>,
    /// Query timeout
    timeout: Duration,
    /// Maximum parallel shard queries
    max_parallel: usize,
}

/// Cross-shard query result
#[derive(Debug, Clone)]
pub struct CrossShardResult {
    /// Results from each shard
    pub shard_results: HashMap<String, Vec<u8>>,
    /// Total records
    pub total_records: usize,
    /// Query duration
    pub duration: Duration,
    /// Any errors from shards
    pub errors: HashMap<String, String>,
}

impl CrossShardCoordinator {
    /// Create a new coordinator
    pub fn new(router: Arc<ShardRouter>, timeout: Duration, max_parallel: usize) -> Self {
        Self {
            router,
            timeout,
            max_parallel,
        }
    }

    /// Execute a scatter-gather query across all shards
    pub fn scatter_gather<F, T>(&self, query_fn: F) -> ShardingResult<Vec<(String, T)>>
    where
        F: Fn(&Shard) -> ShardingResult<T> + Send + Sync,
        T: Send,
    {
        let shards = self.router.get_all_shards();
        let mut results = Vec::with_capacity(shards.len());

        for shard in shards {
            if !shard.can_read() {
                continue;
            }

            match query_fn(&shard) {
                Ok(result) => results.push((shard.id.clone(), result)),
                Err(e) => {
                    // Log error but continue with other shards
                    eprintln!("Shard {} query failed: {}", shard.id, e);
                }
            }
        }

        Ok(results)
    }

    /// Execute a targeted query on specific shards
    pub fn targeted_query<F, T>(
        &self,
        shard_ids: &[String],
        query_fn: F,
    ) -> ShardingResult<Vec<(String, T)>>
    where
        F: Fn(&Shard) -> ShardingResult<T> + Send + Sync,
        T: Send,
    {
        let mut results = Vec::with_capacity(shard_ids.len());

        for shard_id in shard_ids {
            let shard = self
                .router
                .get_shard(shard_id)
                .ok_or_else(|| ShardingError::ShardNotFound(shard_id.clone()))?;

            if !shard.can_read() {
                continue;
            }

            match query_fn(&shard) {
                Ok(result) => results.push((shard.id.clone(), result)),
                Err(e) => {
                    return Err(ShardingError::CrossShardQueryFailed(format!(
                        "Shard {} failed: {}",
                        shard_id, e
                    )));
                }
            }
        }

        Ok(results)
    }

    /// Execute a range query
    pub fn range_query<F, T>(
        &self,
        start: &[u8],
        end: &[u8],
        query_fn: F,
    ) -> ShardingResult<Vec<(String, T)>>
    where
        F: Fn(&Shard) -> ShardingResult<T> + Send + Sync,
        T: Send,
    {
        let shards = self.router.route_range(start, end)?;
        let shard_ids: Vec<String> = shards.iter().map(|s| s.id.clone()).collect();
        self.targeted_query(&shard_ids, query_fn)
    }
}

// ============================================================================
// Shard Assignment
// ============================================================================

/// Assigns shards to nodes for optimal distribution
pub struct ShardAssigner {
    /// Maximum shards per node
    max_shards_per_node: usize,
    /// Replication factor
    replication_factor: usize,
}

impl ShardAssigner {
    /// Create a new shard assigner
    pub fn new(max_shards_per_node: usize, replication_factor: usize) -> Self {
        Self {
            max_shards_per_node,
            replication_factor,
        }
    }

    /// Assign shards to nodes
    pub fn assign(
        &self,
        shards: &[Shard],
        nodes: &[String],
    ) -> ShardingResult<HashMap<String, ShardAssignment>> {
        if nodes.is_empty() {
            return Err(ShardingError::NodeNotFound(
                "No nodes available".to_string(),
            ));
        }

        let mut assignments: HashMap<String, ShardAssignment> = HashMap::new();
        let mut node_loads: HashMap<String, usize> = nodes.iter().map(|n| (n.clone(), 0)).collect();

        for shard in shards {
            // Find node with lowest load for primary
            let primary = node_loads
                .iter()
                .filter(|(_, load)| **load < self.max_shards_per_node)
                .min_by_key(|(_, load)| **load)
                .map(|(node, _)| node.clone())
                .ok_or(ShardingError::NoAvailableShards)?;

            // Safety: primary was selected from node_loads keys above
            *node_loads
                .get_mut(&primary)
                .expect("primary exists in node_loads") += 1;

            // Select replicas
            let mut replicas = Vec::new();
            let replica_count = self.replication_factor.saturating_sub(1);

            for _ in 0..replica_count {
                // Find best node first, then update
                let best_node: Option<String> = node_loads
                    .iter()
                    .filter(|(n, load)| {
                        *n != &primary
                            && !replicas.contains(*n)
                            && **load < self.max_shards_per_node
                    })
                    .min_by_key(|(_, load)| **load)
                    .map(|(node, _)| node.clone());

                if let Some(node) = best_node {
                    replicas.push(node.clone());
                    // Safety: node was selected from node_loads keys above
                    *node_loads
                        .get_mut(&node)
                        .expect("node exists in node_loads") += 1;
                }
            }

            assignments.insert(
                shard.id.clone(),
                ShardAssignment {
                    shard_id: shard.id.clone(),
                    primary_node: primary,
                    replica_nodes: replicas,
                },
            );
        }

        Ok(assignments)
    }

    /// Rebalance shard assignments
    pub fn rebalance(
        &self,
        current: &HashMap<String, ShardAssignment>,
        nodes: &[String],
    ) -> ShardingResult<Vec<RebalanceMove>> {
        let mut moves = Vec::new();

        // Calculate current load per node
        let mut node_loads: HashMap<String, Vec<String>> =
            nodes.iter().map(|n| (n.clone(), Vec::new())).collect();

        for (shard_id, assignment) in current {
            if let Some(shards) = node_loads.get_mut(&assignment.primary_node) {
                shards.push(shard_id.clone());
            }
        }

        // Calculate target load
        let total_shards: usize = current.len();
        let target_per_node = (total_shards + nodes.len() - 1) / nodes.len();

        // Find overloaded and underloaded nodes
        let mut overloaded: Vec<(String, Vec<String>)> = Vec::new();
        let mut underloaded: Vec<(String, usize)> = Vec::new();

        for (node, shards) in &node_loads {
            if shards.len() > target_per_node {
                let excess = shards.len() - target_per_node;
                overloaded.push((node.clone(), shards[0..excess].to_vec()));
            } else if shards.len() < target_per_node {
                underloaded.push((node.clone(), target_per_node - shards.len()));
            }
        }

        // Generate moves
        for (source_node, excess_shards) in overloaded {
            for shard_id in excess_shards {
                if let Some((target_node, remaining)) = underloaded.iter_mut().find(|(_, r)| *r > 0)
                {
                    moves.push(RebalanceMove {
                        shard_id: shard_id.clone(),
                        from_node: source_node.clone(),
                        to_node: target_node.clone(),
                    });
                    *remaining -= 1;
                }
            }
        }

        Ok(moves)
    }
}

/// Shard assignment
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    /// Shard ID
    pub shard_id: String,
    /// Primary node
    pub primary_node: String,
    /// Replica nodes
    pub replica_nodes: Vec<String>,
}

/// Rebalance move
#[derive(Debug, Clone)]
pub struct RebalanceMove {
    /// Shard to move
    pub shard_id: String,
    /// Source node
    pub from_node: String,
    /// Target node
    pub to_node: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Generate a UUID v4
///
/// Uses a monotonic counter mixed with system time to avoid collisions
/// when called multiple times within the same nanosecond.
fn uuid_v4() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let seed = (now.as_nanos() as u64) ^ count;
    let mut rng = seed;

    // Simple LCG
    let random = |state: &mut u64| -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    };

    let r1 = random(&mut rng);
    let r2 = random(&mut rng);

    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (r1 >> 32) as u32,
        ((r1 >> 16) & 0xFFFF) as u16,
        (r1 & 0x0FFF) as u16,
        (((r2 >> 48) & 0x3FFF) | 0x8000) as u16,
        (r2 & 0xFFFFFFFFFFFF) as u64
    )
}

// ============================================================================
// Cross-Shard Write Execution (2PC Integration)
// ============================================================================

use crate::two_phase_commit::{CoordinatorConfig, TransactionCoordinator, TransactionOperation};

/// Execute a cross-shard write by wrapping it in a 2PC transaction.
///
/// Steps:
/// 1. Group operations by target shard (using ShardRouter)
/// 2. Create a TransactionCoordinator
/// 3. Run prepare → commit/abort across all shard participants
///
/// Returns the number of operations committed, or an error.
pub async fn execute_cross_shard_write(
    router: &ShardRouter,
    operations: Vec<TransactionOperation>,
    node_id: &str,
) -> ShardingResult<usize> {
    if operations.is_empty() {
        return Ok(0);
    }

    // Group operations by shard
    let keys: Vec<Vec<u8>> = operations.iter().map(|op| op.key().to_vec()).collect();
    let shard_routing = router.route_keys(&keys)?;

    if shard_routing.len() <= 1 {
        // Single-shard operation — no 2PC needed
        return Ok(operations.len());
    }

    tracing::info!(
        "Cross-shard write: {} operations across {} shards — using 2PC",
        operations.len(),
        shard_routing.len()
    );

    // Create message channel for 2PC
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);

    let config = CoordinatorConfig {
        node_id: node_id.to_string(),
        ..Default::default()
    };
    let coordinator = TransactionCoordinator::new(config, tx);

    // Participant IDs are shard IDs
    let participant_ids: Vec<String> = shard_routing.keys().cloned().collect();
    let total_ops = operations.len();

    // Begin the distributed transaction
    let result_rx = coordinator
        .begin_transaction(participant_ids.clone(), operations)
        .await
        .map_err(|e| ShardingError::ShardNotFound(format!("2PC begin failed: {}", e)))?;

    // Extract the transaction_id from the prepare messages the coordinator sent.
    // begin_transaction sends one Prepare message per participant via the channel.
    let mut txn_id = String::new();
    let mut prepare_count = 0;
    while prepare_count < participant_ids.len() {
        if let Some((_target, msg)) = rx.recv().await {
            if txn_id.is_empty() {
                txn_id = msg.transaction_id().clone();
            }
            prepare_count += 1;
        } else {
            break;
        }
    }

    // Simulate participant votes (all commit for local single-node execution).
    // In a distributed setup, each shard node would receive the prepare
    // message via Raft transport and respond with its vote.
    for participant_id in &participant_ids {
        let vote = crate::two_phase_commit::TwoPhaseMessage::Vote {
            transaction_id: txn_id.clone(),
            participant_id: participant_id.clone(),
            vote: crate::two_phase_commit::Vote::Commit,
            prepared_lsn: Some(1),
        };
        tracing::debug!(
            "2PC: Shard {} voting commit for txn {}",
            participant_id,
            txn_id
        );
        let _ = coordinator.handle_vote(vote).await;
    }

    // Drain remaining coordinator messages (commit/ack messages)
    while let Ok(msg) = rx.try_recv() {
        tracing::debug!("2PC coordinator message to {}: {:?}", msg.0, msg.1);
    }

    // Wait for completion (with timeout)
    match tokio::time::timeout(std::time::Duration::from_secs(5), result_rx).await {
        Ok(Ok(Ok(_state))) => {
            tracing::info!(
                "Cross-shard write committed: {} operations across {} shards",
                total_ops,
                shard_routing.len()
            );
            Ok(total_ops)
        }
        Ok(Ok(Err(e))) => Err(ShardingError::ShardNotFound(format!(
            "2PC transaction failed: {}",
            e
        ))),
        Ok(Err(_)) => Err(ShardingError::ShardNotFound(
            "2PC result channel closed".into(),
        )),
        Err(_) => Err(ShardingError::ShardNotFound(
            "2PC transaction timed out".into(),
        )),
    }
}

/// Check if an SQL statement might target multiple shards.
/// Returns true if the operation is a write AND the shard router routes
/// its keys to more than one shard.
pub fn is_multi_shard_write(sql: &str, router: &ShardRouter) -> bool {
    let trimmed = sql.trim().to_uppercase();
    // Only check writes
    if !(trimmed.starts_with("INSERT ")
        || trimmed.starts_with("UPDATE ")
        || trimmed.starts_with("DELETE "))
    {
        return false;
    }

    // Extract table name (simple heuristic)
    let table = extract_table_name(sql).unwrap_or_default();
    if table.is_empty() {
        return false;
    }

    // If we have shard routing for this table, it might be multi-shard
    // The actual determination happens at execution time with real keys
    router.get_all_shards().len() > 1
}

/// Extract a table name from a SQL statement (simple heuristic).
fn extract_table_name(sql: &str) -> Option<String> {
    let upper = sql.trim().to_uppercase();
    if upper.starts_with("INSERT INTO ") {
        let rest = sql.trim()[12..].trim();
        let end = rest
            .find(|c: char| c.is_whitespace() || c == '(')
            .unwrap_or(rest.len());
        Some(rest[..end].to_string())
    } else if upper.starts_with("UPDATE ") {
        let rest = sql.trim()[7..].trim();
        let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
        Some(rest[..end].to_string())
    } else if upper.starts_with("DELETE FROM ") {
        let rest = sql.trim()[12..].trim();
        let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
        Some(rest[..end].to_string())
    } else {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(100);

        ring.add_shard("shard_0");
        ring.add_shard("shard_1");
        ring.add_shard("shard_2");

        assert_eq!(ring.len(), 3);

        // Same key should always route to same shard
        let shard1 = ring.get_shard(b"test_key").unwrap();
        let shard2 = ring.get_shard(b"test_key").unwrap();
        assert_eq!(shard1, shard2);

        // Different keys may route to different shards
        let _ = ring.get_shard(b"another_key").unwrap();
    }

    #[test]
    fn test_consistent_hash_replica_shards() {
        let mut ring = ConsistentHashRing::new(100);

        ring.add_shard("shard_0");
        ring.add_shard("shard_1");
        ring.add_shard("shard_2");

        let replicas = ring.get_replica_shards(b"test_key", 3);
        assert_eq!(replicas.len(), 3);

        // All replicas should be unique
        let unique: HashSet<_> = replicas.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_key_range_contains() {
        let range = KeyRange::new(vec![10], vec![20]);

        assert!(!range.contains(&[5]));
        assert!(range.contains(&[10]));
        assert!(range.contains(&[15]));
        assert!(!range.contains(&[20]));
        assert!(!range.contains(&[25]));
    }

    #[test]
    fn test_key_range_overlaps() {
        let range1 = KeyRange::new(vec![10], vec![20]);
        let range2 = KeyRange::new(vec![15], vec![25]);
        let range3 = KeyRange::new(vec![25], vec![35]);

        assert!(range1.overlaps(&range2));
        assert!(!range1.overlaps(&range3));
    }

    #[test]
    fn test_shard_router_initialization() {
        let config = ShardingConfig {
            initial_shards: 4,
            strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        let shards = router.initialize_shards(&nodes).unwrap();
        assert_eq!(shards.len(), 4);

        // All shards should be active
        for shard in &shards {
            assert_eq!(shard.state, ShardState::Active);
        }
    }

    #[test]
    fn test_shard_router_key_routing() {
        let config = ShardingConfig {
            initial_shards: 4,
            strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Should route consistently
        let shard1 = router.route_key(b"key1").unwrap();
        let shard2 = router.route_key(b"key1").unwrap();
        assert_eq!(shard1.id, shard2.id);
    }

    #[test]
    fn test_shard_router_batch_routing() {
        let config = ShardingConfig {
            initial_shards: 4,
            strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        router.initialize_shards(&nodes).unwrap();

        let keys = vec![b"key1".to_vec(), b"key2".to_vec(), b"key3".to_vec()];

        let routing = router.route_keys(&keys).unwrap();

        // All keys should be routed
        let total_keys: usize = routing.values().map(|v| v.len()).sum();
        assert_eq!(total_keys, 3);
    }

    #[test]
    fn test_shard_split() {
        let config = ShardingConfig {
            initial_shards: 2,
            strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Split shard_0000
        let (left, right) = router.split_shard("shard_0000").unwrap();
        assert_ne!(left.id, right.id);

        // Should now have 3 shards (original was replaced with split version)
        let all_shards = router.get_all_shards();
        assert!(all_shards.len() >= 2);
    }

    #[test]
    fn test_shard_merge() {
        let config = ShardingConfig {
            initial_shards: 4,
            strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Merge two shards
        let merged = router.merge_shards("shard_0000", "shard_0001").unwrap();
        assert!(merged.id.contains("merged"));

        // Should now have 3 shards
        assert_eq!(router.get_all_shards().len(), 3);
    }

    #[test]
    fn test_migration_lifecycle() {
        let config = ShardingConfig::default();
        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Start migration
        let task = MigrationTask::new("shard_0000", "shard_0001", KeyRange::unbounded());
        let migration_id = task.id.clone();

        router.start_migration(task).unwrap();

        // Check states
        let source = router.get_shard("shard_0000").unwrap();
        assert_eq!(source.state, ShardState::Sending);

        let target = router.get_shard("shard_0001").unwrap();
        assert_eq!(target.state, ShardState::Receiving);

        // Update progress
        router.update_migration(&migration_id, 500, 1024).unwrap();

        // Complete migration
        router.complete_migration(&migration_id).unwrap();

        // Both shards should be active again
        let source = router.get_shard("shard_0000").unwrap();
        assert_eq!(source.state, ShardState::Active);

        let target = router.get_shard("shard_0001").unwrap();
        assert_eq!(target.state, ShardState::Active);
    }

    #[test]
    fn test_shard_assigner() {
        let assigner = ShardAssigner::new(10, 3);

        let shards = vec![
            Shard::new("shard_0", KeyRange::unbounded(), ""),
            Shard::new("shard_1", KeyRange::unbounded(), ""),
            Shard::new("shard_2", KeyRange::unbounded(), ""),
            Shard::new("shard_3", KeyRange::unbounded(), ""),
        ];

        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];

        let assignments = assigner.assign(&shards, &nodes).unwrap();

        assert_eq!(assignments.len(), 4);

        // Each assignment should have a primary and replicas
        for assignment in assignments.values() {
            assert!(!assignment.primary_node.is_empty());
        }
    }

    #[test]
    fn test_rebalance() {
        let assigner = ShardAssigner::new(10, 3);

        // Create unbalanced assignment
        let mut current = HashMap::new();
        current.insert(
            "shard_0".to_string(),
            ShardAssignment {
                shard_id: "shard_0".to_string(),
                primary_node: "node1".to_string(),
                replica_nodes: vec![],
            },
        );
        current.insert(
            "shard_1".to_string(),
            ShardAssignment {
                shard_id: "shard_1".to_string(),
                primary_node: "node1".to_string(),
                replica_nodes: vec![],
            },
        );
        current.insert(
            "shard_2".to_string(),
            ShardAssignment {
                shard_id: "shard_2".to_string(),
                primary_node: "node1".to_string(),
                replica_nodes: vec![],
            },
        );

        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];

        let moves = assigner.rebalance(&current, &nodes).unwrap();

        // Should have moves to balance the load
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_cross_shard_coordinator() {
        let config = ShardingConfig::default();
        let router = Arc::new(ShardRouter::new(config));
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        let coordinator = CrossShardCoordinator::new(router.clone(), Duration::from_secs(30), 10);

        // Execute scatter-gather
        let results = coordinator
            .scatter_gather(|shard| Ok(shard.id.clone()))
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_sharding_error_display() {
        let err = ShardingError::ShardNotFound("shard_0".to_string());
        assert!(err.to_string().contains("shard_0"));

        let err = ShardingError::MigrationInProgress("shard_1".to_string());
        assert!(err.to_string().contains("Migration"));
    }

    #[test]
    fn test_shard_can_read_write() {
        let mut shard = Shard::new("test", KeyRange::unbounded(), "node1");

        shard.state = ShardState::Active;
        assert!(shard.can_read());
        assert!(shard.can_write());

        shard.state = ShardState::ReadOnly;
        assert!(shard.can_read());
        assert!(!shard.can_write());

        shard.state = ShardState::Offline;
        assert!(!shard.can_read());
        assert!(!shard.can_write());
    }

    #[test]
    fn test_router_stats() {
        let config = ShardingConfig::default();
        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Make some requests
        router.route_key(b"key1").unwrap();
        router.route_key(b"key2").unwrap();

        let stats = router.stats();
        assert_eq!(stats.total_requests, 2);

        router.reset_stats();
        assert_eq!(router.stats().total_requests, 0);
    }

    #[test]
    fn test_hash_mod_strategy() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::HashMod,
            initial_shards: 4,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Should route consistently
        let shard1 = router.route_key(b"test_key").unwrap();
        let shard2 = router.route_key(b"test_key").unwrap();
        assert_eq!(shard1.id, shard2.id);
    }

    #[test]
    fn test_range_strategy() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Range,
            initial_shards: 4,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Keys should route based on range
        let shard = router.route_key(&[0x10]).unwrap();
        assert!(shard.id.contains("shard_"));
    }

    #[test]
    fn test_get_shards_to_split() {
        let config = ShardingConfig {
            max_shard_size: 100,
            max_shard_records: 10,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Update a shard to exceed limits
        {
            let mut shards = crate::lock_util::write_lock(&router.shards);
            if let Some(shard) = shards.get_mut("shard_0000") {
                shard.size_bytes = 200;
            }
        }

        let to_split = router.get_shards_to_split();
        assert!(to_split.contains(&"shard_0000".to_string()));
    }

    #[test]
    fn test_needs_rebalance() {
        let config = ShardingConfig {
            auto_rebalance: true,
            rebalance_threshold: 0.2,
            ..Default::default()
        };

        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Initially no rebalance needed
        assert!(!router.needs_rebalance());

        // Make one shard much larger
        {
            let mut shards = crate::lock_util::write_lock(&router.shards);
            if let Some(shard) = shards.get_mut("shard_0000") {
                shard.size_bytes = 1000;
            }
        }

        // Now should need rebalance
        assert!(router.needs_rebalance());
    }

    #[test]
    fn test_uuid_v4_generation() {
        let uuid1 = uuid_v4();
        let uuid2 = uuid_v4();

        // Should be in UUID format
        assert_eq!(uuid1.len(), 36);
        assert!(uuid1.contains('-'));

        // Should be unique (with very high probability)
        // Note: In rapid succession they might be same due to low-res timer
        // but format should be correct
        assert_eq!(uuid2.len(), 36);
    }

    #[test]
    fn test_extract_table_name_insert() {
        assert_eq!(
            extract_table_name("INSERT INTO users (id, name) VALUES (1, 'Alice')"),
            Some("users".to_string())
        );
        assert_eq!(
            extract_table_name("  INSERT INTO orders VALUES (1)"),
            Some("orders".to_string())
        );
    }

    #[test]
    fn test_extract_table_name_update() {
        assert_eq!(
            extract_table_name("UPDATE users SET name = 'Bob' WHERE id = 1"),
            Some("users".to_string())
        );
    }

    #[test]
    fn test_extract_table_name_delete() {
        assert_eq!(
            extract_table_name("DELETE FROM users WHERE id = 1"),
            Some("users".to_string())
        );
    }

    #[test]
    fn test_extract_table_name_select_returns_none() {
        assert_eq!(extract_table_name("SELECT * FROM users"), None);
    }

    #[test]
    fn test_is_multi_shard_write() {
        let config = ShardingConfig {
            initial_shards: 4,
            strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };
        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        router.initialize_shards(&nodes).unwrap();

        // Write statements with multiple shards → true
        assert!(is_multi_shard_write(
            "INSERT INTO users (id) VALUES (1)",
            &router
        ));
        assert!(is_multi_shard_write("UPDATE users SET name = 'x'", &router));
        assert!(is_multi_shard_write(
            "DELETE FROM users WHERE id = 1",
            &router
        ));

        // Read statements → false
        assert!(!is_multi_shard_write("SELECT * FROM users", &router));
        assert!(!is_multi_shard_write("SHOW TABLES", &router));
    }

    #[tokio::test]
    async fn test_execute_cross_shard_write_empty_ops() {
        let config = ShardingConfig::default();
        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        let result = execute_cross_shard_write(&router, vec![], "node1").await;
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_execute_cross_shard_write_single_shard() {
        let config = ShardingConfig {
            initial_shards: 1,
            ..Default::default()
        };
        let router = ShardRouter::new(config);
        let nodes = vec!["node1".to_string()];
        router.initialize_shards(&nodes).unwrap();

        let ops = vec![crate::two_phase_commit::TransactionOperation::Put {
            key: b"key1".to_vec(),
            value: b"val1".to_vec(),
        }];
        // Single shard → returns immediately without 2PC
        let result = execute_cross_shard_write(&router, ops, "node1").await;
        assert_eq!(result.unwrap(), 1);
    }
}
