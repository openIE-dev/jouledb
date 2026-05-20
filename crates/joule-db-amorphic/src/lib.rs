//! # Amorphic Database
//!
//! A database without fixed structure. Data exists in a hyperdimensional hologram
//! and materializes into the structure you need at query time.
//!
//! ## The Core Insight
//!
//! Traditional databases force you to choose a structure upfront:
//! - Relational: Tables with fixed schemas
//! - Document: JSON-like nested structures
//! - Graph: Nodes and edges
//! - Vector: Embeddings for similarity
//! - Time-series: Timestamped sequences
//!
//! **Amorphic** stores data as hyperdimensional holograms that contain ALL potential
//! structures simultaneously. Your query determines what form the data takes.
//!
//! ```text
//!                     ┌─────────────────┐
//!                     │  HDC Hologram   │
//!                     │  (Amorphic)     │
//!                     └────────┬────────┘
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          │                   │                   │
//!          ▼                   ▼                   ▼
//!    ┌──────────┐       ┌──────────┐       ┌──────────┐
//!    │Relational│       │  Graph   │       │  Vector  │
//!    │   View   │       │   View   │       │   View   │
//!    └──────────┘       └──────────┘       └──────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use joule_db_amorphic::{AmorphicStore, Query};
//!
//! let mut store = AmorphicStore::new();
//!
//! // Ingest data - any format
//! store.ingest_json(r#"{"name": "Alice", "age": 30, "friends": ["Bob"]}"#)?;
//! store.ingest_row(&["name", "age"], &["Bob", "25"])?;
//! store.ingest_edge("Alice", "KNOWS", "Bob")?;
//!
//! // Query as relational
//! let rows = store.query_sql("SELECT name WHERE age > 25")?;
//!
//! // Query as graph (same data!)
//! let paths = store.query_graph("Alice", "KNOWS", 2)?;
//!
//! // Query as vector (same data!)
//! let similar = store.query_similar("Alice", 5)?;
//! ```

use joule_db_hdc::{BinaryHV, BundleAccumulator, DistanceMetric, HNSWIndex};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

// GPU acceleration module
pub mod gpu;

// Batch feature serving for content recommendation
pub mod batch;
pub use batch::{BatchFeatureRequest, BatchFeatureResponse, BatchGetRequest, BatchGetResponse};

// Content event schema and user profile management
pub mod events;
pub use events::{ContentEvent, EventProcessor, UserProfile, UserStats};

// Trending/hot-content index with time-decayed scoring
pub mod trending;
pub use trending::{TrendWindow, TrendingIndex, TrendingItem};

// Content moderation workflow (trust & safety)
pub mod moderation;
pub use moderation::{
    FlaggedItem, ModerationAction, ModerationMatch, ModerationPolicy, ModerationQueue,
    ReviewStatus,
};

// ContentID: live fingerprint matching for copyright detection
pub mod content_id;
pub use content_id::{ContentIdIndex, ContentMatch, ContentReference, MatchPolicy};

// Temporal field validity for rights/licensing
pub mod temporal_fields;
pub use temporal_fields::{TemporalField, TemporalStore};

// JouleDB CDN — broadcast and streaming protocol adapters
pub mod cdn;

// JouleDB AI — tiered inference runtime (holographic → embedded → local → frontier)
pub mod ai;
pub use ai::{JouleDbAi, AiReceipt, InferenceTier, TierConstraints};

// Authentication & authorization — token-based auth with RBAC
pub mod auth;
pub use auth::{AuthStore, Identity, Operation, Permission, Role};

// Multi-tenancy — tenant isolation at storage layer
pub mod tenant;
pub use tenant::{MultiTenantStore, TenantConfig, TenantStats, TenantStatus};

// Module persistence — checkpoint/restore for all content infrastructure modules
pub mod persistence;
pub use persistence::{CheckpointManager, Persistable};

// Hologram delta sync for edge replication + client cache
pub mod hologram_sync;
pub use hologram_sync::{CatalogEntry, CatalogSync, HologramDelta};

// VoIP: voiceprint store, call quality analytics, CDR, deepfake detection
pub mod voiceprint;
pub use voiceprint::{
    CallDetailRecord, CallDirection, CallQualityMetrics, SyntheticRisk, VerificationResult,
    VoiceFeatures, Voiceprint, VoiceprintStore,
};

// Ad tech targeting engine (programmatic ad decisioning)
pub mod ad_targeting;
pub use ad_targeting::{AdCampaign, AdDecision, AdTargetingEngine, FrequencyCap, TargetingCriteria};

// Semantic hybrid search (vector + keyword + metadata fusion)
pub mod hybrid_search;
pub use hybrid_search::{HybridQuery, HybridResult, hybrid_search};

// Royalty calculation engine (usage-based revenue distribution)
pub mod royalty;
pub use royalty::{
    DistributionModel, RevenuePool, RightsHolder, RightsRole, RoyaltyCalculator, RoyaltyPayment,
    UsageEvent,
};

// Accessibility infrastructure (captions, scoring, epilepsy detection)
pub mod accessibility;
pub use accessibility::{
    AccessibilityGrade, AccessibilityScore, Caption, CaptionSource, CaptionStore, CaptionTrack,
    EpilepsyRisk,
};

// Distribution manifest system (multi-platform syndication)
pub mod distribution;
pub use distribution::{
    DistributionManager, DistributionManifest, DistributionStatus, PlatformDistribution,
};

// Selective replication + edge query routing
pub mod selective_replication;
pub use selective_replication::{
    ContentFilter, EdgeRouter, QueryRoute, ReplicationPolicy,
};

pub use gpu::{GpuContext, GpuVectorStore};

// Tiered storage module (memory -> mmap -> disk)
pub mod tiered;
pub use tiered::{StorageTier, TieredConfig, TieredStats, TieredStore};

// Entropy-based auto-partitioning module
pub mod partition;
pub use partition::{
    BIT_ENTROPY_THRESHOLD,
    HologramHealth,
    MAX_PARTITIONS_WARNING,
    Partition,
    PartitionConfig,
    PartitionEvent,
    PartitionEventType,
    PartitionHealthStatus,
    PartitionManager,
    PartitionManagerStats,
    PartitionedAmorphicStore,
    RETRIEVAL_PROBABILITY_THRESHOLD,
    RoutingStrategy,
    SNR_THRESHOLD,
    ShardConfig,
    // Fine-grained concurrency
    ShardedAmorphicStore,
    ShardedStoreStats,
    TARGET_PARTITION_SIZE,
};

// Platform detection and optimization module
pub mod platform;
pub use platform::{
    CpuArch, ParallelConfig, PlatformCapabilities, SimdCapabilities, SimdLevel,
    hamming_distance_optimized, hamming_distances_batch_optimized, hamming_top_k_optimized,
    parallel_map, parallel_reduce, platform,
};

// Columnar projections for OLAP analytics
pub mod columnar;
pub use columnar::{
    Column, ColumnStats, ColumnarStats, ColumnarStore, GroupAggregates, JoinResult,
};

// Query optimizer for cost-based decisions
pub mod optimizer;
pub use optimizer::{
    AggregateFunc,
    ColumnCostStats,
    JoinType,
    // Logical planning
    LogicalPlan,
    LogicalPlanBuilder,
    OptimizerStats,
    // Physical planning
    PhysicalPlan,
    PlanStep,
    Predicate,
    QueryOptimizer,
    QueryPlan,
    // Rule-based optimizer
    RuleBasedOptimizer,
    SortOrder,
};

// Memory manager with eviction policies
pub mod memory;
pub use memory::{
    AutoEvictionConfig,
    EvictionCandidate,
    EvictionPolicy,
    EvictionResult,
    ManagedStore,
    ManagedStoreBuilder,
    ManagedStoreStats,
    MemoryEstimator,
    // Auto-eviction integration
    MemoryManagedStore,
    MemoryManager,
    MemoryStats,
    StorePriority,
};

// SQL parser and executor
pub mod sql;
pub use sql::{
    CompareOp, FromClause, JoinClause, OrderByItem, SelectColumn, SelectStatement, SqlExecutor,
    SqlParser, SqlResult, SqlStatement, SqlValue, WhereClause,
};

// WAL-based durable storage module (requires "durable" feature)
#[cfg(feature = "durable")]
pub mod durable;
#[cfg(feature = "durable")]
pub use durable::DurableAmorphicStore;

// MVCC transaction module (requires "durable" feature for full functionality)
#[cfg(feature = "durable")]
pub mod tx;
#[cfg(feature = "durable")]
pub use tx::{AmorphicTransaction, AmorphicTransactionManager};

// Materialized views for pre-computed aggregations
pub mod materialized;
pub use materialized::{MaterializedView, MaterializedViewManager, RefreshPolicy};

// Space-filling curves for spatial queries
pub mod spatial;
pub use spatial::{CurveType, SpatialIndex, SpatialIndexExt};

// Streaming ingestion with backpressure
pub mod streaming;
pub use streaming::{IngestItem, IngestStatus, StreamConfig, StreamMetrics, StreamingIngester};

// Knowledge Core: compressed structural skeleton of human knowledge
pub mod knowledge;

// Distributed mode with consistent hashing
pub mod distributed;
pub use distributed::{
    ConsistentHashRing, DistributedQueryResult, DistributedStore, NodeConfig, NodeId,
};

/// Ordered key for B-tree index (handles NaN properly)
#[derive(Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Dimension for holographic storage
pub const DIMENSION: usize = 10000;

// =============================================================================
// ENGINEERING GUARDRAILS - Capacity Limits
// =============================================================================

/// Maximum fields per record before similarity degrades significantly
/// Based on capacity theorem: P(correct) ≈ 1 - N²/D, at N=316, P≈90%
pub const MAX_FIELDS_WARNING: usize = 200;
pub const MAX_FIELDS_ERROR: usize = 316;

/// Chunk size for hierarchical bundling of high-field-count records
/// Each chunk stays well under the capacity limit for good similarity
pub const HIERARCHICAL_CHUNK_SIZE: usize = 100;

/// Maximum records before global hologram becomes unusable for similarity
/// After this, use LSH index instead of global hologram
pub const GLOBAL_HOLOGRAM_SATURATION_WARNING: usize = 200;
pub const GLOBAL_HOLOGRAM_SATURATION_LIMIT: usize = 316;

/// Health status of the store
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// All systems nominal
    Healthy,
    /// Approaching limits, consider mitigation
    Warning(Vec<String>),
    /// Exceeding limits, some features degraded
    Degraded(Vec<String>),
}

/// Errors from the amorphic store
#[derive(Error, Debug)]
pub enum AmorphicError {
    #[error("Ingestion failed: {0}")]
    IngestionError(String),

    #[error("Query failed: {0}")]
    QueryError(String),

    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("No results found")]
    NotFound,

    #[error("Invalid query syntax: {0}")]
    InvalidQuery(String),

    // NEW: Capacity guardrails
    #[error(
        "Record has {0} fields, exceeds safe limit of {1}. Similarity matching will be degraded."
    )]
    FieldCountExceeded(usize, usize),

    #[error(
        "Store has {0} records, global hologram saturated (limit: {1}). Use LSH for similarity queries."
    )]
    HologramSaturated(usize, usize),

    #[error("Record {0} not found")]
    RecordNotFound(RecordId),

    #[error("Delete operation failed: {0}")]
    DeleteError(String),

    #[error("Concurrent access violation: {0}")]
    ConcurrencyError(String),
}

pub type AmorphicResult<T> = Result<T, AmorphicError>;

/// A record ID in the amorphic store
pub type RecordId = u64;

/// Primitive value types supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    /// Convert from JSON value
    pub fn from_json(json: &JsonValue) -> Self {
        match json {
            JsonValue::Null => Value::Null,
            JsonValue::Bool(b) => Value::Bool(*b),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::Null
                }
            }
            JsonValue::String(s) => Value::String(s.clone()),
            JsonValue::Array(arr) => Value::Array(arr.iter().map(Value::from_json).collect()),
            JsonValue::Object(obj) => {
                let map: HashMap<String, Value> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from_json(v)))
                    .collect();
                Value::Object(map)
            }
        }
    }

    /// Get as string if possible
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as i64 if possible
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Get as f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
}

/// An amorphic record - data in its encoded form
#[derive(Clone)]
pub struct AmorphicRecord {
    /// Unique ID
    pub id: RecordId,
    /// The holographic encoding (amorphic form)
    pub hologram: BinaryHV,
    /// Field-value pairs for materialization
    fields: HashMap<String, Value>,
    /// Graph edges from this record
    edges: Vec<(String, RecordId)>, // (relation, target)
    /// Timestamp if applicable
    timestamp: Option<u64>,
}

impl AmorphicRecord {
    /// Get a field value
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.fields.get(field)
    }

    /// Get all field names
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(|s| s.as_str())
    }

    /// Iterate over field name-value pairs
    pub fn fields_iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get edges
    pub fn edges(&self) -> &[(String, RecordId)] {
        &self.edges
    }
}

/// Query result that can be viewed in multiple ways
pub struct QueryResult {
    records: Vec<AmorphicRecord>,
}

impl QueryResult {
    /// View as rows (relational)
    pub fn as_rows(&self, columns: &[&str]) -> Vec<Vec<Option<Value>>> {
        self.records
            .iter()
            .map(|r| columns.iter().map(|c| r.fields.get(*c).cloned()).collect())
            .collect()
    }

    /// View as JSON documents
    pub fn as_documents(&self) -> Vec<HashMap<String, Value>> {
        self.records.iter().map(|r| r.fields.clone()).collect()
    }

    /// View as graph edges
    pub fn as_edges(&self) -> Vec<(RecordId, String, RecordId)> {
        self.records
            .iter()
            .flat_map(|r| {
                r.edges
                    .iter()
                    .map(|(rel, target)| (r.id, rel.clone(), *target))
            })
            .collect()
    }

    /// Get record count
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get records
    pub fn records(&self) -> &[AmorphicRecord] {
        &self.records
    }

    /// Consume and return records (for aggregation)
    pub fn into_records(self) -> Vec<AmorphicRecord> {
        self.records
    }
}

/// Result of a holographic join operation.
///
/// Contains the XOR-joined hologram (for further similarity queries)
/// and merged materialized fields (for human-readable output).
#[derive(Debug, Clone)]
pub struct HolographicJoinResult {
    /// The joined hologram: left.hologram ⊗ right.hologram.
    /// Can be used for further similarity searches or joins.
    pub joined_hologram: BinaryHV,
    /// Merged fields from both records (right fields prefixed with "right.").
    pub merged_fields: HashMap<String, Value>,
    /// Quality of the join (similarity of join field values, 0.0-1.0).
    pub join_quality: f32,
    /// ID of the left record.
    pub left_id: RecordId,
    /// ID of the right record.
    pub right_id: RecordId,
}

/// Number of LSH hash tables for approximate similarity search
const LSH_NUM_TABLES: usize = 32;
/// Number of bits sampled per LSH hash table (fewer bits = more collisions = more candidates)
const LSH_BITS_PER_TABLE: usize = 12;
/// Minimum number of tables a candidate must appear in (higher = more selective)
const LSH_MIN_TABLES: usize = 4;

/// HNSW default parameters
const HNSW_MAX_CONNECTIONS: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_EF_SEARCH: usize = 50;

/// Similarity index strategy for approximate nearest neighbor queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexStrategy {
    /// Locality-Sensitive Hashing only (O(1) lookup, lower recall).
    Lsh,
    /// Hierarchical Navigable Small World graph only (O(log n), higher recall).
    Hnsw,
    /// LSH for fast candidate generation, HNSW for accurate re-ranking.
    Hybrid,
}

impl Default for IndexStrategy {
    fn default() -> Self {
        IndexStrategy::Hybrid
    }
}

/// The Amorphic Store - a database without fixed structure
pub struct AmorphicStore {
    /// The holographic superposition of all records
    hologram: BundleAccumulator,

    /// Individual record storage (for materialization)
    pub(crate) records: HashMap<RecordId, AmorphicRecord>,

    /// Index: field name -> records containing that field
    field_index: HashMap<String, Vec<RecordId>>,

    /// Index: value hash -> records with that value
    value_index: HashMap<u64, Vec<RecordId>>,

    /// Graph index: source -> [(relation, target)]
    graph_index: HashMap<RecordId, Vec<(String, RecordId)>>,

    /// Reverse graph index: target -> [(relation, source)]
    reverse_graph_index: HashMap<RecordId, Vec<(String, RecordId)>>,

    /// Name-to-ID mapping for graph queries
    pub(crate) name_to_id: HashMap<String, RecordId>,

    /// B-tree index for numeric range queries: field -> (value -> record_ids)
    /// Enables O(log N) range queries instead of O(N)
    numeric_index: HashMap<String, BTreeMap<OrderedFloat, Vec<RecordId>>>,

    /// LSH hash tables for O(1) approximate similarity search
    /// Each table maps a hash (from sampled bits) to record IDs
    lsh_tables: Vec<HashMap<u64, Vec<RecordId>>>,

    /// Bit positions sampled for each LSH table
    lsh_bit_positions: Vec<Vec<usize>>,

    /// Contiguous hologram storage for cache-efficient similarity search
    /// holograms[i] corresponds to record with ID (i+1)
    hologram_array: Vec<BinaryHV>,

    /// Field vectors for encoding
    field_vectors: HashMap<String, BinaryHV>,

    /// Type vectors
    type_vectors: HashMap<String, BinaryHV>,

    /// Scalar base for numeric encoding
    scalar_base: BinaryHV,

    /// Next record ID
    next_id: RecordId,

    /// RNG for vector generation
    rng: StdRng,

    /// Columnar projections for fast OLAP queries
    columnar: ColumnarStore,

    /// HNSW index for O(log n) approximate nearest neighbor search.
    /// Populated alongside LSH during ingest when strategy includes HNSW.
    hnsw_index: Option<HNSWIndex>,

    /// Active similarity index strategy
    index_strategy: IndexStrategy,
}

impl AmorphicStore {
    /// Create a new amorphic store
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xA0F0_0F1C);

        let mut type_vectors = HashMap::new();
        type_vectors.insert(
            "null".to_string(),
            BinaryHV::from_hash(b"TYPE_NULL", DIMENSION),
        );
        type_vectors.insert(
            "bool".to_string(),
            BinaryHV::from_hash(b"TYPE_BOOL", DIMENSION),
        );
        type_vectors.insert(
            "int".to_string(),
            BinaryHV::from_hash(b"TYPE_INT", DIMENSION),
        );
        type_vectors.insert(
            "float".to_string(),
            BinaryHV::from_hash(b"TYPE_FLOAT", DIMENSION),
        );
        type_vectors.insert(
            "string".to_string(),
            BinaryHV::from_hash(b"TYPE_STRING", DIMENSION),
        );
        type_vectors.insert(
            "array".to_string(),
            BinaryHV::from_hash(b"TYPE_ARRAY", DIMENSION),
        );
        type_vectors.insert(
            "object".to_string(),
            BinaryHV::from_hash(b"TYPE_OBJECT", DIMENSION),
        );
        type_vectors.insert(
            "edge".to_string(),
            BinaryHV::from_hash(b"TYPE_EDGE", DIMENSION),
        );

        let scalar_base = BinaryHV::random(DIMENSION, rng.random());

        // Initialize LSH tables with random bit positions
        let mut lsh_tables = Vec::with_capacity(LSH_NUM_TABLES);
        let mut lsh_bit_positions = Vec::with_capacity(LSH_NUM_TABLES);

        for _ in 0..LSH_NUM_TABLES {
            lsh_tables.push(HashMap::new());

            // Select random bit positions for this table
            let mut positions: Vec<usize> = (0..DIMENSION).collect();
            // Fisher-Yates shuffle to pick random positions
            for i in (1..DIMENSION).rev() {
                let j = rng.random_range(0..=i);
                positions.swap(i, j);
            }
            positions.truncate(LSH_BITS_PER_TABLE);
            positions.sort_unstable(); // Sort for cache-friendly access
            lsh_bit_positions.push(positions);
        }

        Self {
            hologram: BundleAccumulator::new(DIMENSION),
            records: HashMap::new(),
            field_index: HashMap::new(),
            value_index: HashMap::new(),
            graph_index: HashMap::new(),
            reverse_graph_index: HashMap::new(),
            name_to_id: HashMap::new(),
            numeric_index: HashMap::new(),
            lsh_tables,
            lsh_bit_positions,
            hologram_array: Vec::new(),
            field_vectors: HashMap::new(),
            type_vectors,
            scalar_base,
            next_id: 1,
            rng,
            columnar: ColumnarStore::new(),
            hnsw_index: None,
            index_strategy: IndexStrategy::default(),
        }
    }

    /// Create a new amorphic store with a specific index strategy.
    pub fn with_index_strategy(strategy: IndexStrategy) -> Self {
        let mut store = Self::new();
        store.index_strategy = strategy;
        // HNSW dimension = 2 * number of u64 words (each u64 split into two u32 packed as f32)
        let hnsw_dim = (DIMENSION + 63) / 64 * 2;
        if matches!(strategy, IndexStrategy::Hnsw | IndexStrategy::Hybrid) {
            store.hnsw_index = Some(HNSWIndex::with_metric(
                hnsw_dim,
                HNSW_MAX_CONNECTIONS,
                HNSW_EF_CONSTRUCTION,
                DistanceMetric::Hamming,
            ));
        }
        store
    }

    /// Enable HNSW indexing on an existing store (builds index from existing records).
    pub fn enable_hnsw(&mut self) {
        let hnsw_dim = (DIMENSION + 63) / 64 * 2;
        let mut hnsw = HNSWIndex::with_metric(
            hnsw_dim,
            HNSW_MAX_CONNECTIONS,
            HNSW_EF_CONSTRUCTION,
            DistanceMetric::Hamming,
        );
        // Backfill existing records
        for (&id, record) in &self.records {
            let packed = record.hologram.to_f32_packed();
            let _ = hnsw.insert(id.to_string(), packed);
        }
        self.hnsw_index = Some(hnsw);
        if self.index_strategy == IndexStrategy::Lsh {
            self.index_strategy = IndexStrategy::Hybrid;
        }
    }

    /// Compute LSH hash for a binary hypervector for a specific table
    fn lsh_hash(&self, hv: &BinaryHV, table_idx: usize) -> u64 {
        let positions = &self.lsh_bit_positions[table_idx];
        let words = hv.as_words();
        let mut hash: u64 = 0;

        for (i, &pos) in positions.iter().enumerate() {
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            if word_idx < words.len() && (words[word_idx] >> bit_idx) & 1 == 1 {
                hash |= 1 << (i % 64);
            }
        }
        hash
    }

    /// Add a record to all LSH tables
    fn lsh_insert(&mut self, id: RecordId, hologram: &BinaryHV) {
        for table_idx in 0..LSH_NUM_TABLES {
            let hash = self.lsh_hash(hologram, table_idx);
            self.lsh_tables[table_idx].entry(hash).or_default().push(id);
        }
    }

    /// Get LSH candidates (records that hash to same bucket in multiple tables)
    fn lsh_candidates(&self, probe: &BinaryHV) -> Vec<RecordId> {
        // Count how many tables each candidate appears in
        let mut candidate_counts: HashMap<RecordId, usize> = HashMap::new();

        for table_idx in 0..LSH_NUM_TABLES {
            let hash = self.lsh_hash(probe, table_idx);
            if let Some(ids) = self.lsh_tables[table_idx].get(&hash) {
                for &id in ids {
                    *candidate_counts.entry(id).or_insert(0) += 1;
                }
            }
        }

        // Only return candidates that appear in at least LSH_MIN_TABLES tables
        candidate_counts
            .into_iter()
            .filter(|(_, count)| *count >= LSH_MIN_TABLES)
            .map(|(id, _)| id)
            .collect()
    }

    /// Get or create field vector
    fn get_field_vector(&mut self, field: &str) -> BinaryHV {
        if let Some(v) = self.field_vectors.get(field) {
            return v.clone();
        }
        let v = BinaryHV::from_hash(field.as_bytes(), DIMENSION);
        self.field_vectors.insert(field.to_string(), v.clone());
        v
    }

    /// Encode a value to hypervector
    fn encode_value(&mut self, value: &Value) -> BinaryHV {
        match value {
            Value::Null => self.type_vectors["null"].clone(),
            Value::Bool(b) => {
                let bool_vec = if *b {
                    BinaryHV::from_hash(b"TRUE", DIMENSION)
                } else {
                    BinaryHV::from_hash(b"FALSE", DIMENSION)
                };
                self.type_vectors["bool"].bind(&bool_vec)
            }
            Value::Int(i) => {
                // Permutation encoding for integers
                let shift = ((*i as i128 + i64::MAX as i128) % 10000) as usize;
                let int_vec = self.scalar_base.permute_words(shift);
                self.type_vectors["int"].bind(&int_vec)
            }
            Value::Float(f) => {
                let bytes = f.to_le_bytes();
                let float_vec = BinaryHV::from_hash(&bytes, DIMENSION);
                self.type_vectors["float"].bind(&float_vec)
            }
            Value::String(s) => {
                let str_vec = BinaryHV::from_hash(s.as_bytes(), DIMENSION);
                self.type_vectors["string"].bind(&str_vec)
            }
            Value::Array(arr) => {
                // Bundle array elements with position encoding
                let mut acc = BundleAccumulator::new(DIMENSION);
                for (i, v) in arr.iter().enumerate() {
                    let elem_vec = self.encode_value(v);
                    let pos_vec = self.scalar_base.permute_words(i);
                    acc.add(&elem_vec.bind(&pos_vec));
                }
                self.type_vectors["array"].bind(&acc.threshold())
            }
            Value::Object(obj) => {
                // Bundle field-value pairs
                let mut acc = BundleAccumulator::new(DIMENSION);
                for (k, v) in obj {
                    let field_vec = self.get_field_vector(k);
                    let val_vec = self.encode_value(v);
                    acc.add(&field_vec.bind(&val_vec));
                }
                self.type_vectors["object"].bind(&acc.threshold())
            }
        }
    }

    /// Encode a complete record
    fn encode_record(&mut self, fields: &HashMap<String, Value>) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        for (field, value) in fields {
            let field_vec = self.get_field_vector(field);
            let value_vec = self.encode_value(value);
            acc.add(&field_vec.bind(&value_vec));
        }

        acc.threshold()
    }

    /// Encode a record using hierarchical bundling for high field counts.
    ///
    /// When a record has more fields than the bundle capacity (~316), we split
    /// the fields into chunks of HIERARCHICAL_CHUNK_SIZE (100) and encode each
    /// chunk separately. The chunk holograms are then bundled with position
    /// encoding to distinguish between chunks.
    ///
    /// This allows encoding records with thousands of fields while maintaining
    /// good similarity matching within each chunk.
    fn encode_record_hierarchical(&mut self, fields: &HashMap<String, Value>) -> BinaryHV {
        let field_count = fields.len();

        // Use standard encoding for small records
        if field_count <= MAX_FIELDS_WARNING {
            return self.encode_record(fields);
        }

        // Collect fields into a vector for chunking
        let field_vec: Vec<_> = fields.iter().collect();

        // Encode each chunk separately
        let mut chunk_holograms = Vec::new();
        let mut chunk_start = 0;

        while chunk_start < field_count {
            let chunk_end = (chunk_start + HIERARCHICAL_CHUNK_SIZE).min(field_count);
            let chunk_fields: HashMap<String, Value> = field_vec[chunk_start..chunk_end]
                .iter()
                .map(|(k, v)| ((*k).clone(), (*v).clone()))
                .collect();

            let chunk_hv = self.encode_record(&chunk_fields);
            chunk_holograms.push(chunk_hv);
            chunk_start = chunk_end;
        }

        // Bundle chunk holograms with position encoding
        // This allows distinguishing which chunk a field came from
        let mut acc = BundleAccumulator::new(DIMENSION);
        for (i, chunk_hv) in chunk_holograms.iter().enumerate() {
            // Create a position vector by permuting the scalar base
            // Different permutation amounts create orthogonal position markers
            let pos_hv = self.scalar_base.permute_words(i * HIERARCHICAL_CHUNK_SIZE);
            acc.add(&chunk_hv.bind(&pos_hv));
        }

        acc.threshold()
    }

    /// Hash a value for indexing
    fn hash_value(value: &Value) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        match value {
            Value::Null => 0u8.hash(&mut hasher),
            Value::Bool(b) => b.hash(&mut hasher),
            Value::Int(i) => i.hash(&mut hasher),
            Value::Float(f) => f.to_bits().hash(&mut hasher),
            Value::String(s) => s.hash(&mut hasher),
            Value::Array(_) => 2u8.hash(&mut hasher), // Arrays indexed by content separately
            Value::Object(_) => 3u8.hash(&mut hasher),
        }
        hasher.finish()
    }

    /// Ingest JSON document
    pub fn ingest_json(&mut self, json: &str) -> AmorphicResult<RecordId> {
        let parsed: JsonValue = serde_json::from_str(json)?;
        let value = Value::from_json(&parsed);

        match value {
            Value::Object(fields) => self.ingest_fields(fields),
            _ => Err(AmorphicError::IngestionError(
                "Top-level JSON must be an object".to_string(),
            )),
        }
    }

    /// Ingest a row (column names + values)
    pub fn ingest_row(&mut self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        if columns.len() != values.len() {
            return Err(AmorphicError::IngestionError(
                "Column count must match value count".to_string(),
            ));
        }

        let fields: HashMap<String, Value> = columns
            .iter()
            .zip(values.iter())
            .map(|(c, v)| {
                // Try to parse as number, fall back to string
                let value = if let Ok(i) = v.parse::<i64>() {
                    Value::Int(i)
                } else if let Ok(f) = v.parse::<f64>() {
                    Value::Float(f)
                } else if *v == "true" {
                    Value::Bool(true)
                } else if *v == "false" {
                    Value::Bool(false)
                } else if *v == "null" {
                    Value::Null
                } else {
                    Value::String(v.to_string())
                };
                (c.to_string(), value)
            })
            .collect();

        self.ingest_fields(fields)
    }

    /// Ingest a graph edge
    pub fn ingest_edge(
        &mut self,
        source: &str,
        relation: &str,
        target: &str,
    ) -> AmorphicResult<RecordId> {
        // Ensure source and target exist as records
        let source_id = self.ensure_entity(source)?;
        let target_id = self.ensure_entity(target)?;

        // Create edge record
        let mut fields = HashMap::new();
        fields.insert("_type".to_string(), Value::String("edge".to_string()));
        fields.insert("_source".to_string(), Value::String(source.to_string()));
        fields.insert("_relation".to_string(), Value::String(relation.to_string()));
        fields.insert("_target".to_string(), Value::String(target.to_string()));

        let id = self.next_id;
        self.next_id += 1;

        let hologram = self.encode_record(&fields);

        // Add edge to graph indices
        self.graph_index
            .entry(source_id)
            .or_default()
            .push((relation.to_string(), target_id));

        self.reverse_graph_index
            .entry(target_id)
            .or_default()
            .push((relation.to_string(), source_id));

        // Update source record's edges
        if let Some(source_record) = self.records.get_mut(&source_id) {
            source_record.edges.push((relation.to_string(), target_id));
        }

        let record = AmorphicRecord {
            id,
            hologram: hologram.clone(),
            fields,
            edges: vec![],
            timestamp: None,
        };

        // Add to global hologram
        self.hologram.add(&hologram);
        self.records.insert(id, record);

        Ok(id)
    }

    /// Ingest with timestamp (time-series)
    pub fn ingest_timestamped(
        &mut self,
        fields: HashMap<String, Value>,
        timestamp: u64,
    ) -> AmorphicResult<RecordId> {
        let mut fields = fields;
        fields.insert("_timestamp".to_string(), Value::Int(timestamp as i64));

        let id = self.ingest_fields(fields)?;

        if let Some(record) = self.records.get_mut(&id) {
            record.timestamp = Some(timestamp);
        }

        Ok(id)
    }

    /// Core ingestion of field map
    fn ingest_fields(&mut self, fields: HashMap<String, Value>) -> AmorphicResult<RecordId> {
        // GUARDRAIL: Check global hologram saturation
        let record_count = self.records.len();
        if record_count >= GLOBAL_HOLOGRAM_SATURATION_LIMIT {
            eprintln!(
                "⚠️  WARNING: Store has {} records. Global hologram saturated. Use query_similar_to() with LSH instead of direct hologram queries.",
                record_count
            );
        }

        let id = self.next_id;
        self.next_id += 1;

        // Use hierarchical encoding for high-field-count records
        let field_count = fields.len();
        let hologram = if field_count > MAX_FIELDS_WARNING {
            // Hierarchical bundling handles unlimited fields
            self.encode_record_hierarchical(&fields)
        } else {
            self.encode_record(&fields)
        };

        // Update indices
        for (field, value) in &fields {
            self.field_index.entry(field.clone()).or_default().push(id);

            let value_hash = Self::hash_value(value);
            self.value_index.entry(value_hash).or_default().push(id);

            // Track name for graph queries
            if field == "name" || field == "_name" {
                if let Value::String(name) = value {
                    self.name_to_id.insert(name.clone(), id);
                }
            }

            // Update B-tree index for numeric values (enables O(log N) range queries)
            if let Some(num) = value.as_f64() {
                self.numeric_index
                    .entry(field.clone())
                    .or_default()
                    .entry(OrderedFloat(num))
                    .or_default()
                    .push(id);
            }

            // Record in columnar storage for fast OLAP queries
            self.columnar.record_value(field, id, value);
        }

        let record = AmorphicRecord {
            id,
            hologram: hologram.clone(),
            fields,
            edges: vec![],
            timestamp: None,
        };

        // Add to global hologram
        self.hologram.add(&hologram);

        // Add to LSH index for fast similarity search
        self.lsh_insert(id, &hologram);

        // Add to HNSW index if enabled
        if let Some(ref mut hnsw) = self.hnsw_index {
            let packed = hologram.to_f32_packed();
            let _ = hnsw.insert(id.to_string(), packed);
        }

        // Add to contiguous hologram array for cache-efficient similarity search
        // Ensure array is big enough (IDs are 1-based, array is 0-based)
        while self.hologram_array.len() < id as usize {
            self.hologram_array.push(BinaryHV::zeros(DIMENSION));
        }
        self.hologram_array[(id - 1) as usize] = hologram.clone();

        self.records.insert(id, record);

        Ok(id)
    }

    // ==================== PREPARE/APPLY METHODS (for WAL durability) ====================

    /// Prepare a JSON record for ingestion (parse, allocate ID, validate)
    ///
    /// Returns (id, fields) that can be logged to WAL before applying.
    /// Use with `apply_prepared_ingest()` to complete the operation.
    #[cfg(feature = "durable")]
    pub fn prepare_json_ingest(
        &mut self,
        json: &str,
    ) -> AmorphicResult<(RecordId, HashMap<String, Value>)> {
        let parsed: serde_json::Value = serde_json::from_str(json)?;
        let value = Value::from_json(&parsed);

        match value {
            Value::Object(fields) => {
                // Validate field count
                let field_count = fields.len();
                if field_count > MAX_FIELDS_ERROR {
                    return Err(AmorphicError::FieldCountExceeded(
                        field_count,
                        MAX_FIELDS_ERROR,
                    ));
                }

                // Allocate ID
                let id = self.next_id;
                self.next_id += 1;

                Ok((id, fields))
            }
            _ => Err(AmorphicError::IngestionError(
                "Top-level JSON must be an object".to_string(),
            )),
        }
    }

    /// Prepare a row record for ingestion (parse, allocate ID, validate)
    #[cfg(feature = "durable")]
    pub fn prepare_row_ingest(
        &mut self,
        columns: &[&str],
        values: &[&str],
    ) -> AmorphicResult<(RecordId, HashMap<String, Value>)> {
        if columns.len() != values.len() {
            return Err(AmorphicError::IngestionError(
                "Column count must match value count".to_string(),
            ));
        }

        let fields: HashMap<String, Value> = columns
            .iter()
            .zip(values.iter())
            .map(|(col, val)| (col.to_string(), Value::String(val.to_string())))
            .collect();

        // Validate field count
        let field_count = fields.len();
        if field_count > MAX_FIELDS_ERROR {
            return Err(AmorphicError::FieldCountExceeded(
                field_count,
                MAX_FIELDS_ERROR,
            ));
        }

        // Allocate ID
        let id = self.next_id;
        self.next_id += 1;

        Ok((id, fields))
    }

    /// Apply a prepared ingest operation (after WAL logging)
    #[cfg(feature = "durable")]
    pub fn apply_prepared_ingest(
        &mut self,
        id: RecordId,
        fields: HashMap<String, Value>,
    ) -> AmorphicResult<()> {
        // Warnings for large records
        let field_count = fields.len();
        if field_count > MAX_FIELDS_WARNING {
            eprintln!(
                "⚠️  WARNING: Record has {} fields (warning threshold: {}). Similarity matching may be degraded.",
                field_count, MAX_FIELDS_WARNING
            );
        }

        // Check global hologram saturation
        let record_count = self.records.len();
        if record_count >= GLOBAL_HOLOGRAM_SATURATION_LIMIT {
            eprintln!(
                "⚠️  WARNING: Store has {} records. Global hologram saturated.",
                record_count
            );
        }

        let hologram = self.encode_record(&fields);

        // Update indices
        for (field, value) in &fields {
            self.field_index.entry(field.clone()).or_default().push(id);

            let value_hash = Self::hash_value(value);
            self.value_index.entry(value_hash).or_default().push(id);

            // Track name for graph queries
            if field == "name" || field == "_name" {
                if let Value::String(name) = value {
                    self.name_to_id.insert(name.clone(), id);
                }
            }

            // Update B-tree index for numeric values
            if let Some(num) = value.as_f64() {
                self.numeric_index
                    .entry(field.clone())
                    .or_default()
                    .entry(OrderedFloat(num))
                    .or_default()
                    .push(id);
            }

            // Record in columnar storage for fast OLAP queries
            self.columnar.record_value(field, id, value);
        }

        let record = AmorphicRecord {
            id,
            hologram: hologram.clone(),
            fields,
            edges: vec![],
            timestamp: None,
        };

        // Add to global hologram
        self.hologram.add(&hologram);

        // Add to LSH index
        self.lsh_insert(id, &hologram);

        // Add to HNSW index if enabled
        if let Some(ref mut hnsw) = self.hnsw_index {
            let packed = hologram.to_f32_packed();
            let _ = hnsw.insert(id.to_string(), packed);
        }

        // Add to contiguous hologram array
        while self.hologram_array.len() < id as usize {
            self.hologram_array.push(BinaryHV::zeros(DIMENSION));
        }
        self.hologram_array[(id - 1) as usize] = hologram.clone();

        self.records.insert(id, record);

        Ok(())
    }

    /// Apply a recovered record from WAL replay
    ///
    /// Similar to apply_prepared_ingest but allows setting edges and timestamp.
    #[cfg(feature = "durable")]
    pub fn apply_recovered_record(
        &mut self,
        id: RecordId,
        fields: HashMap<String, Value>,
        edges: Vec<(String, RecordId)>,
        timestamp: Option<u64>,
    ) -> AmorphicResult<()> {
        // Ensure next_id is beyond this recovered record
        if id >= self.next_id {
            self.next_id = id + 1;
        }

        let hologram = self.encode_record(&fields);

        // Update indices
        for (field, value) in &fields {
            self.field_index.entry(field.clone()).or_default().push(id);

            let value_hash = Self::hash_value(value);
            self.value_index.entry(value_hash).or_default().push(id);

            if field == "name" || field == "_name" {
                if let Value::String(name) = value {
                    self.name_to_id.insert(name.clone(), id);
                }
            }

            if let Some(num) = value.as_f64() {
                self.numeric_index
                    .entry(field.clone())
                    .or_default()
                    .entry(OrderedFloat(num))
                    .or_default()
                    .push(id);
            }
        }

        // Update graph indices if edges present
        if !edges.is_empty() {
            self.graph_index.insert(id, edges.clone());
            for (relation, target) in &edges {
                self.reverse_graph_index
                    .entry(*target)
                    .or_default()
                    .push((relation.clone(), id));
            }
        }

        let record = AmorphicRecord {
            id,
            hologram: hologram.clone(),
            fields,
            edges,
            timestamp,
        };

        // Add to global hologram
        self.hologram.add(&hologram);

        // Add to LSH index
        self.lsh_insert(id, &hologram);

        // Add to HNSW index if enabled
        if let Some(ref mut hnsw) = self.hnsw_index {
            let packed = hologram.to_f32_packed();
            let _ = hnsw.insert(id.to_string(), packed);
        }

        // Add to contiguous hologram array
        while self.hologram_array.len() < id as usize {
            self.hologram_array.push(BinaryHV::zeros(DIMENSION));
        }
        self.hologram_array[(id - 1) as usize] = hologram.clone();

        self.records.insert(id, record);

        Ok(())
    }

    /// Ensure an entity exists, create if not
    fn ensure_entity(&mut self, name: &str) -> AmorphicResult<RecordId> {
        if let Some(id) = self.name_to_id.get(name) {
            return Ok(*id);
        }

        let mut fields = HashMap::new();
        fields.insert("name".to_string(), Value::String(name.to_string()));
        self.ingest_fields(fields)
    }

    // ==================== QUERY METHODS ====================

    /// Query by field existence (returns records that have this field)
    pub fn query_has_field(&self, field: &str) -> QueryResult {
        let ids = self.field_index.get(field).cloned().unwrap_or_default();
        let records: Vec<AmorphicRecord> = ids
            .iter()
            .filter_map(|id| self.records.get(id).cloned())
            .collect();
        QueryResult { records }
    }

    /// Query by exact field-value match
    pub fn query_equals(&self, field: &str, value: &Value) -> QueryResult {
        let value_hash = Self::hash_value(value);
        let candidate_ids = self
            .value_index
            .get(&value_hash)
            .cloned()
            .unwrap_or_default();

        let records: Vec<AmorphicRecord> = candidate_ids
            .iter()
            .filter_map(|id| {
                let record = self.records.get(id)?;
                if record.fields.get(field) == Some(value) {
                    Some(record.clone())
                } else {
                    None
                }
            })
            .collect();

        QueryResult { records }
    }

    /// Query by numeric range
    /// Query by numeric range using B-tree index (O(log N) + O(k) where k = result size)
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> QueryResult {
        // Use B-tree index if available (fast path)
        if let Some(btree) = self.numeric_index.get(field) {
            let min_key = OrderedFloat(min);
            let max_key = OrderedFloat(max);

            // O(log N) range query on B-tree
            let mut record_ids: Vec<RecordId> = btree
                .range(min_key..=max_key)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect();

            // Deduplicate (in case same record has multiple values)
            record_ids.sort_unstable();
            record_ids.dedup();

            let records: Vec<AmorphicRecord> = record_ids
                .iter()
                .filter_map(|id| self.records.get(id).cloned())
                .collect();

            return QueryResult { records };
        }

        // Fallback to linear scan if no B-tree index (shouldn't happen for numeric fields)
        let candidate_ids = self.field_index.get(field).cloned().unwrap_or_default();

        let records: Vec<AmorphicRecord> = candidate_ids
            .iter()
            .filter_map(|id| {
                let record = self.records.get(id)?;
                let value = record.fields.get(field)?;
                let num = value.as_f64()?;
                if num >= min && num <= max {
                    Some(record.clone())
                } else {
                    None
                }
            })
            .collect();

        QueryResult { records }
    }

    /// Query by holographic similarity (uses SIMD when available)
    pub fn query_similar(&self, probe: &BinaryHV, threshold: f32) -> QueryResult {
        let records: Vec<AmorphicRecord> = self
            .records
            .values()
            .filter(|r| r.hologram.similarity_simd(probe) >= threshold)
            .cloned()
            .collect();

        QueryResult { records }
    }

    /// Query similar to a named entity.
    ///
    /// Dispatches to HNSW (O(log n), high recall) or LSH (O(1), lower recall)
    /// based on the configured `IndexStrategy`.
    pub fn query_similar_to(&self, name: &str, k: usize) -> QueryResult {
        let source_id = match self.name_to_id.get(name) {
            Some(id) => *id,
            None => return QueryResult { records: vec![] },
        };

        // Get source hologram from contiguous array (better cache locality)
        let source_idx = (source_id - 1) as usize;
        if source_idx >= self.hologram_array.len() {
            return QueryResult { records: vec![] };
        }
        let source_hologram = &self.hologram_array[source_idx];

        // HNSW path: O(log n) with high recall
        if let Some(ref hnsw) = self.hnsw_index {
            if matches!(
                self.index_strategy,
                IndexStrategy::Hnsw | IndexStrategy::Hybrid
            ) {
                let packed = source_hologram.to_f32_packed();
                // Request k+1 since we need to exclude self
                let results = hnsw.query_with_ef(&packed, k + 1, HNSW_EF_SEARCH);

                let records: Vec<AmorphicRecord> = results
                    .into_iter()
                    .filter_map(|r| {
                        let rid: RecordId = r.id.parse().ok()?;
                        if rid == source_id {
                            return None;
                        }
                        self.records.get(&rid).cloned()
                    })
                    .take(k)
                    .collect();

                return QueryResult { records };
            }
        }

        // LSH path: O(1) candidate generation + brute-force rescore
        let candidates = self.lsh_candidates(source_hologram);

        let scored: Vec<(f32, RecordId)> = if candidates.len() >= k * 2 {
            // Fast path: only score LSH candidates
            candidates
                .into_iter()
                .filter(|&id| id != source_id && (id as usize) <= self.hologram_array.len())
                .map(|id| {
                    let idx = (id - 1) as usize;
                    let sim = self.hologram_array[idx].similarity_simd(source_hologram);
                    (sim, id)
                })
                .collect()
        } else {
            // Fallback: score all records using contiguous array
            self.hologram_array
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != source_idx)
                .map(|(idx, hv)| {
                    let sim = hv.similarity_simd(source_hologram);
                    (sim, (idx + 1) as RecordId)
                })
                .collect()
        };

        let mut scored = scored;
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let records: Vec<AmorphicRecord> = scored
            .into_iter()
            .take(k)
            .filter_map(|(_, id)| self.records.get(&id).cloned())
            .collect();

        QueryResult { records }
    }

    /// Query similar to a probe BinaryHV using HNSW (if available) or brute-force.
    pub fn query_similar_hnsw(&self, probe: &BinaryHV, k: usize) -> QueryResult {
        if let Some(ref hnsw) = self.hnsw_index {
            let packed = probe.to_f32_packed();
            let results = hnsw.query_with_ef(&packed, k, HNSW_EF_SEARCH);

            let records: Vec<AmorphicRecord> = results
                .into_iter()
                .filter_map(|r| {
                    let rid: RecordId = r.id.parse().ok()?;
                    self.records.get(&rid).cloned()
                })
                .collect();

            return QueryResult { records };
        }

        // Fallback to threshold-based brute force
        self.query_similar(probe, 0.5)
    }

    /// Content-based media retrieval: find records similar to a media hologram.
    ///
    /// Takes a BinaryHV produced by the media domain encoder (from frequency
    /// coefficients) and finds the k most similar records in the store.
    /// Uses HNSW when available, falls back to LSH + brute-force.
    ///
    /// This is the retrieval endpoint for MediaQL queries like:
    /// `FROM amorphic.images WHERE similar_to(media('query.jpg'), threshold: 0.8)`
    pub fn query_media_similar(&self, media_hologram: &BinaryHV, k: usize, threshold: f32) -> QueryResult {
        // Use HNSW path if available (higher recall)
        if let Some(ref hnsw) = self.hnsw_index {
            let packed = media_hologram.to_f32_packed();
            let results = hnsw.query_with_ef(&packed, k, HNSW_EF_SEARCH);

            let records: Vec<AmorphicRecord> = results
                .into_iter()
                .filter_map(|r| {
                    let rid: RecordId = r.id.parse().ok()?;
                    let record = self.records.get(&rid)?;
                    // Apply threshold filter
                    let sim = record.hologram.similarity(media_hologram);
                    if sim >= threshold {
                        Some(record.clone())
                    } else {
                        None
                    }
                })
                .collect();

            return QueryResult { records };
        }

        // Fallback: threshold-based scan
        self.query_similar(media_hologram, threshold)
    }

    /// Ingest a media hologram directly (from the media ingest pipeline).
    ///
    /// Stores the hologram with associated metadata fields. The hologram
    /// was produced by `FrequencyDomainEncoder` from DCT/FFT coefficients.
    pub fn ingest_media_hologram(
        &mut self,
        hologram: BinaryHV,
        metadata: HashMap<String, Value>,
    ) -> AmorphicResult<RecordId> {
        let id = self.next_id;
        self.next_id += 1;

        // Update standard indices from metadata
        for (field, value) in &metadata {
            self.field_index.entry(field.clone()).or_default().push(id);

            let value_hash = Self::hash_value(value);
            self.value_index.entry(value_hash).or_default().push(id);

            if field == "name" || field == "_name" {
                if let Value::String(name) = value {
                    self.name_to_id.insert(name.clone(), id);
                }
            }

            if let Some(num) = value.as_f64() {
                self.numeric_index
                    .entry(field.clone())
                    .or_default()
                    .entry(OrderedFloat(num))
                    .or_default()
                    .push(id);
            }

            self.columnar.record_value(field, id, value);
        }

        let record = AmorphicRecord {
            id,
            hologram: hologram.clone(),
            fields: metadata,
            edges: vec![],
            timestamp: None,
        };

        self.hologram.add(&hologram);
        self.lsh_insert(id, &hologram);

        if let Some(ref mut hnsw) = self.hnsw_index {
            let packed = hologram.to_f32_packed();
            let _ = hnsw.insert(id.to_string(), packed);
        }

        while self.hologram_array.len() < id as usize {
            self.hologram_array.push(BinaryHV::zeros(DIMENSION));
        }
        self.hologram_array[(id - 1) as usize] = hologram;

        self.records.insert(id, record);
        Ok(id)
    }

    /// Ingest a CLIP/DINOv2/SigLIP embedding directly.
    ///
    /// Converts a float embedding vector to a BinaryHV hologram via random hyperplane
    /// projection (Johnson-Lindenstrauss), which preserves cosine similarity structure.
    /// Hamming distance on the binary vector approximates angular distance on the original.
    ///
    /// Also stores the original f32 vector in metadata for precise re-ranking.
    ///
    /// This is the primary ingest path for content providers using learned embeddings
    /// from foundation models.
    pub fn ingest_embedding(
        &mut self,
        embedding: &[f32],
        mut metadata: HashMap<String, Value>,
    ) -> AmorphicResult<RecordId> {
        // Random hyperplane projection: preserves cosine similarity as Hamming distance.
        // Seed is fixed so the same projection matrix is used for all embeddings
        // (required for similarity to be meaningful across records).
        const PROJECTION_SEED: u64 = 0xDB_E0BE_D9B0_0001;
        let hologram = BinaryHV::from_embedding(embedding, DIMENSION, PROJECTION_SEED);

        // Store original embedding dimensions in metadata for re-ranking
        metadata.insert(
            "_embedding_dim".to_string(),
            Value::Int(embedding.len() as i64),
        );

        self.ingest_media_hologram(hologram, metadata)
    }

    /// GPU-accelerated similarity search
    ///
    /// Uses WebGPU to compute all similarities in parallel, achieving
    /// ~10x speedup for large datasets (10K+ vectors).
    pub fn query_similar_to_gpu(&self, name: &str, k: usize, gpu: &gpu::GpuContext) -> QueryResult {
        let source_id = match self.name_to_id.get(name) {
            Some(id) => *id,
            None => return QueryResult { records: vec![] },
        };

        let source_idx = (source_id - 1) as usize;
        if source_idx >= self.hologram_array.len() {
            return QueryResult { records: vec![] };
        }
        let source_hologram = &self.hologram_array[source_idx];

        // Compute all similarities on GPU in parallel
        let similarities = match gpu.compute_similarities(source_hologram, &self.hologram_array) {
            Ok(sims) => sims,
            Err(_) => {
                // Fall back to CPU on GPU error
                return self.query_similar_to(name, k);
            }
        };

        // Convert to scored pairs (skip self)
        let mut scored: Vec<(u32, RecordId)> = similarities
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| *idx != source_idx)
            .map(|(idx, sim)| (sim, (idx + 1) as RecordId))
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        // Return top k records
        let records: Vec<AmorphicRecord> = scored
            .into_iter()
            .take(k)
            .filter_map(|(_, id)| self.records.get(&id).cloned())
            .collect();

        QueryResult { records }
    }

    /// Upload vectors to GPU for fast repeated queries
    ///
    /// Call this once, then use `query_similar_to_gpu_fast` for fast queries.
    /// This avoids re-uploading 10K vectors on each query.
    pub fn create_gpu_store(&self, gpu: &gpu::GpuContext) -> AmorphicResult<gpu::GpuVectorStore> {
        gpu::GpuVectorStore::upload(gpu, &self.hologram_array)
    }

    /// Fast GPU similarity search using pre-uploaded vectors
    ///
    /// ~5-10x faster than `query_similar_to_gpu` because vectors don't need re-uploading.
    /// First call `create_gpu_store()` once, then use this for repeated queries.
    pub fn query_similar_to_gpu_fast(
        &self,
        name: &str,
        k: usize,
        gpu: &gpu::GpuContext,
        store: &gpu::GpuVectorStore,
    ) -> QueryResult {
        let source_id = match self.name_to_id.get(name) {
            Some(id) => *id,
            None => return QueryResult { records: vec![] },
        };

        let source_idx = (source_id - 1) as usize;
        if source_idx >= self.hologram_array.len() {
            return QueryResult { records: vec![] };
        }
        let source_hologram = &self.hologram_array[source_idx];

        // Compute all similarities on GPU using pre-uploaded vectors
        let similarities = match gpu.compute_similarities_fast(source_hologram, store) {
            Ok(sims) => sims,
            Err(_) => {
                // Fall back to CPU on GPU error
                return self.query_similar_to(name, k);
            }
        };

        // Convert to scored pairs (skip self)
        let mut scored: Vec<(u32, RecordId)> = similarities
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| *idx != source_idx)
            .map(|(idx, sim)| (sim, (idx + 1) as RecordId))
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        // Return top k records
        let records: Vec<AmorphicRecord> = scored
            .into_iter()
            .take(k)
            .filter_map(|(_, id)| self.records.get(&id).cloned())
            .collect();

        QueryResult { records }
    }

    /// Graph traversal query
    pub fn query_graph(&self, start: &str, relation: &str, depth: usize) -> QueryResult {
        let start_id = match self.name_to_id.get(start) {
            Some(id) => *id,
            None => return QueryResult { records: vec![] },
        };

        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![start_id];
        let mut results = Vec::new();

        for _ in 0..depth {
            let mut next_frontier = Vec::new();

            for id in frontier {
                if visited.contains(&id) {
                    continue;
                }
                visited.insert(id);

                if let Some(record) = self.records.get(&id) {
                    results.push(record.clone());
                }

                if let Some(edges) = self.graph_index.get(&id) {
                    for (rel, target) in edges {
                        if rel == relation || relation == "*" {
                            if !visited.contains(target) {
                                next_frontier.push(*target);
                            }
                        }
                    }
                }
            }

            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        QueryResult { records: results }
    }

    /// Time range query
    pub fn query_time_range(&self, start: u64, end: u64) -> QueryResult {
        let records: Vec<AmorphicRecord> = self
            .records
            .values()
            .filter(|r| {
                if let Some(ts) = r.timestamp {
                    ts >= start && ts <= end
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        QueryResult { records }
    }

    // ==================== HOLOGRAPHIC JOINS (SOTA Gap 4) ====================

    /// Holographic join: BIND-based join without materialization.
    ///
    /// Given two records encoded as holograms, a holographic join XORs them
    /// to produce a result hologram that directly encodes the joined attributes.
    ///
    /// Example: If `person = name ⊗ role` and `company = name ⊗ industry`,
    /// then `person ⊗ company = role ⊗ industry` — the join on "name" is
    /// a single XOR operation, with no intermediate table.
    ///
    /// # Arguments
    /// * `left_name` - Name of the left record
    /// * `right_name` - Name of the right record
    /// * `join_field` - The field to join on (used for verification, not computation)
    ///
    /// # Returns
    /// The joined hologram and the similarity score (higher = better join quality).
    pub fn holographic_join(
        &self,
        left_name: &str,
        right_name: &str,
        join_field: &str,
    ) -> AmorphicResult<HolographicJoinResult> {
        let left_id = self.name_to_id.get(left_name).ok_or_else(|| {
            AmorphicError::QueryError(format!("Record '{}' not found", left_name))
        })?;
        let right_id = self.name_to_id.get(right_name).ok_or_else(|| {
            AmorphicError::QueryError(format!("Record '{}' not found", right_name))
        })?;

        let left = self
            .records
            .get(left_id)
            .ok_or_else(|| AmorphicError::QueryError("Left record missing".into()))?;
        let right = self
            .records
            .get(right_id)
            .ok_or_else(|| AmorphicError::QueryError("Right record missing".into()))?;

        // The holographic join: XOR the two holograms.
        // Since each hologram = BUNDLE(field_i ⊗ value_i), the XOR "cancels out"
        // shared bindings (the join field) and produces the cross-product of
        // the remaining attributes.
        let joined_hologram = left.hologram.bind(&right.hologram);

        // Verify the join by checking that the join field exists in both records
        let left_has_field = left.fields.contains_key(join_field);
        let right_has_field = right.fields.contains_key(join_field);

        // Compute join quality: how well the join field values match
        let join_quality = if left_has_field && right_has_field {
            let left_val = &left.fields[join_field];
            let right_val = &right.fields[join_field];
            let left_hv = BinaryHV::from_bytes(
                format!("{:?}", left_val).as_bytes(),
                DIMENSION,
            );
            let right_hv = BinaryHV::from_bytes(
                format!("{:?}", right_val).as_bytes(),
                DIMENSION,
            );
            left_hv.similarity(&right_hv)
        } else {
            0.0
        };

        // Merge materialized fields (for result readability)
        let mut merged_fields = left.fields.clone();
        for (k, v) in &right.fields {
            if k != join_field {
                merged_fields
                    .entry(format!("right.{}", k))
                    .or_insert_with(|| v.clone());
            }
        }

        Ok(HolographicJoinResult {
            joined_hologram,
            merged_fields,
            join_quality,
            left_id: *left_id,
            right_id: *right_id,
        })
    }

    /// Batch holographic join: join all records from one set against another
    /// where the join field values match (above similarity threshold).
    pub fn holographic_join_batch(
        &self,
        left_ids: &[RecordId],
        right_ids: &[RecordId],
        join_field: &str,
        similarity_threshold: f32,
    ) -> Vec<HolographicJoinResult> {
        let mut results = Vec::new();

        // Encode the join field values for all records
        let left_vecs: Vec<(RecordId, Option<BinaryHV>)> = left_ids
            .iter()
            .map(|&id| {
                let hv = self.records.get(&id).and_then(|r| {
                    r.fields.get(join_field).map(|v| {
                        BinaryHV::from_bytes(format!("{:?}", v).as_bytes(), DIMENSION)
                    })
                });
                (id, hv)
            })
            .collect();

        let right_vecs: Vec<(RecordId, Option<BinaryHV>)> = right_ids
            .iter()
            .map(|&id| {
                let hv = self.records.get(&id).and_then(|r| {
                    r.fields.get(join_field).map(|v| {
                        BinaryHV::from_bytes(format!("{:?}", v).as_bytes(), DIMENSION)
                    })
                });
                (id, hv)
            })
            .collect();

        // Cross-match using similarity
        for (left_id, left_hv) in &left_vecs {
            let Some(lhv) = left_hv else { continue };
            for (right_id, right_hv) in &right_vecs {
                let Some(rhv) = right_hv else { continue };
                let sim = lhv.similarity(rhv);
                if sim >= similarity_threshold {
                    let left = &self.records[left_id];
                    let right = &self.records[right_id];
                    let joined_hologram = left.hologram.bind(&right.hologram);

                    let mut merged_fields = left.fields.clone();
                    for (k, v) in &right.fields {
                        if k != join_field {
                            merged_fields
                                .entry(format!("right.{}", k))
                                .or_insert_with(|| v.clone());
                        }
                    }

                    results.push(HolographicJoinResult {
                        joined_hologram,
                        merged_fields,
                        join_quality: sim,
                        left_id: *left_id,
                        right_id: *right_id,
                    });
                }
            }
        }

        results
    }

    /// Simple SQL-like query (SELECT fields WHERE field op value)
    pub fn query_sql(&self, query: &str) -> AmorphicResult<QueryResult> {
        // Very simple parser: "SELECT field1,field2 WHERE field op value"
        let query = query.trim().to_uppercase();

        // Handle SELECT * without WHERE
        if !query.contains("WHERE") {
            let records: Vec<AmorphicRecord> = self.records.values().cloned().collect();
            return Ok(QueryResult { records });
        }

        // Parse WHERE clause
        let parts: Vec<&str> = query.split("WHERE").collect();
        if parts.len() != 2 {
            return Err(AmorphicError::InvalidQuery(
                "Expected WHERE clause".to_string(),
            ));
        }

        let condition = parts[1].trim();

        // Parse condition (field op value)
        if let Some(pos) = condition.find('=') {
            let field = condition[..pos].trim().to_lowercase();
            let value_str = condition[pos + 1..]
                .trim()
                .trim_matches('\'')
                .trim_matches('"');

            // Try to parse as number or use as string
            let value = if let Ok(i) = value_str.parse::<i64>() {
                Value::Int(i)
            } else if let Ok(f) = value_str.parse::<f64>() {
                Value::Float(f)
            } else {
                Value::String(value_str.to_lowercase())
            };

            return Ok(self.query_equals(&field, &value));
        }

        if let Some(pos) = condition.find('>') {
            let field = condition[..pos].trim().to_lowercase();
            let value_str = condition[pos + 1..].trim();
            let min: f64 = value_str
                .parse()
                .map_err(|_| AmorphicError::InvalidQuery("Expected numeric value".to_string()))?;
            return Ok(self.query_range(&field, min, f64::MAX));
        }

        if let Some(pos) = condition.find('<') {
            let field = condition[..pos].trim().to_lowercase();
            let value_str = condition[pos + 1..].trim();
            let max: f64 = value_str
                .parse()
                .map_err(|_| AmorphicError::InvalidQuery("Expected numeric value".to_string()))?;
            return Ok(self.query_range(&field, f64::MIN, max));
        }

        Err(AmorphicError::InvalidQuery(
            "Unsupported query operator".to_string(),
        ))
    }

    /// Get global hologram (the amorphic state)
    pub fn hologram(&self) -> BinaryHV {
        self.hologram.threshold()
    }

    /// Get record count
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get a record by ID
    pub fn get(&self, id: RecordId) -> Option<&AmorphicRecord> {
        self.records.get(&id)
    }

    /// Get by name
    pub fn get_by_name(&self, name: &str) -> Option<&AmorphicRecord> {
        let id = self.name_to_id.get(name)?;
        self.records.get(id)
    }

    /// Get reference to the global hologram accumulator
    ///
    /// Used by the partition module to measure hologram health metrics.
    pub fn global_hologram(&self) -> &BundleAccumulator {
        &self.hologram
    }

    // ==================== ENGINEERING GUARDRAILS ====================

    /// Check health status of the store
    ///
    /// Returns warnings about capacity limits and degradation risks.
    pub fn health_check(&self) -> HealthStatus {
        let mut warnings = Vec::new();
        let mut degraded = Vec::new();

        // Check record count vs global hologram capacity
        let record_count = self.records.len();
        if record_count >= GLOBAL_HOLOGRAM_SATURATION_LIMIT {
            degraded.push(format!(
                "Global hologram saturated: {} records (limit: {}). Direct hologram similarity queries will return noise. Use query_similar_to() with LSH instead.",
                record_count, GLOBAL_HOLOGRAM_SATURATION_LIMIT
            ));
        } else if record_count >= GLOBAL_HOLOGRAM_SATURATION_WARNING {
            warnings.push(format!(
                "Approaching hologram saturation: {} records (warning: {}, limit: {})",
                record_count, GLOBAL_HOLOGRAM_SATURATION_WARNING, GLOBAL_HOLOGRAM_SATURATION_LIMIT
            ));
        }

        // Check for any large records (by sampling)
        let large_records: Vec<RecordId> = self
            .records
            .iter()
            .filter(|(_, r)| r.fields.len() > MAX_FIELDS_WARNING)
            .map(|(id, _)| *id)
            .take(10)
            .collect();

        if !large_records.is_empty() {
            warnings.push(format!(
                "Found {} record(s) with >{}  fields (e.g., IDs: {:?}). Similarity matching degraded for these records.",
                large_records.len(), MAX_FIELDS_WARNING, large_records
            ));
        }

        if !degraded.is_empty() {
            HealthStatus::Degraded(degraded)
        } else if !warnings.is_empty() {
            HealthStatus::Warning(warnings)
        } else {
            HealthStatus::Healthy
        }
    }

    /// Get detailed statistics about the store
    /// Number of records in the store.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    pub fn stats(&self) -> StoreStats {
        let total_fields: usize = self.records.values().map(|r| r.fields.len()).sum();
        let max_fields = self
            .records
            .values()
            .map(|r| r.fields.len())
            .max()
            .unwrap_or(0);
        let avg_fields = if self.records.is_empty() {
            0.0
        } else {
            total_fields as f64 / self.records.len() as f64
        };

        StoreStats {
            record_count: self.records.len(),
            total_fields,
            max_fields_per_record: max_fields,
            avg_fields_per_record: avg_fields,
            lsh_table_count: self.lsh_tables.len(),
            unique_field_names: self.field_index.len(),
            graph_edges: self.graph_index.values().map(|v| v.len()).sum(),
            hologram_saturation_pct: (self.records.len() as f64
                / GLOBAL_HOLOGRAM_SATURATION_LIMIT as f64
                * 100.0)
                .min(100.0),
        }
    }

    /// Delete a record by ID
    ///
    /// Note: This removes from indices but cannot "subtract" from the global hologram.
    /// The global hologram retains traces of deleted records until rebuild.
    pub fn delete(&mut self, id: RecordId) -> AmorphicResult<()> {
        let record = self
            .records
            .remove(&id)
            .ok_or(AmorphicError::RecordNotFound(id))?;

        // Remove from field index
        for field in record.fields.keys() {
            if let Some(ids) = self.field_index.get_mut(field) {
                ids.retain(|&x| x != id);
            }
        }

        // Remove from value index
        for value in record.fields.values() {
            let hash = Self::hash_value(value);
            if let Some(ids) = self.value_index.get_mut(&hash) {
                ids.retain(|&x| x != id);
            }
        }

        // Remove from numeric index
        for (field, value) in &record.fields {
            if let Some(num) = value.as_f64() {
                if let Some(btree) = self.numeric_index.get_mut(field) {
                    if let Some(ids) = btree.get_mut(&OrderedFloat(num)) {
                        ids.retain(|&x| x != id);
                    }
                }
            }
        }

        // Remove from LSH tables
        for table in &mut self.lsh_tables {
            for ids in table.values_mut() {
                ids.retain(|&x| x != id);
            }
        }

        // Remove from graph indices
        self.graph_index.remove(&id);
        for edges in self.graph_index.values_mut() {
            edges.retain(|(_, target)| *target != id);
        }
        self.reverse_graph_index.remove(&id);
        for edges in self.reverse_graph_index.values_mut() {
            edges.retain(|(_, source)| *source != id);
        }

        // Remove from name index
        if let Some(Value::String(name)) = record.fields.get("name") {
            self.name_to_id.remove(name);
        }
        if let Some(Value::String(name)) = record.fields.get("_name") {
            self.name_to_id.remove(name);
        }

        // Clear from hologram array (set to zeros)
        if (id as usize) <= self.hologram_array.len() {
            self.hologram_array[(id - 1) as usize] = BinaryHV::zeros(DIMENSION);
        }

        // NOTE: Cannot remove from global hologram - it's a superposition
        // The global hologram will retain traces until rebuild
        eprintln!(
            "⚠️  Record {} deleted from indices. Global hologram retains traces until rebuild().",
            id
        );

        Ok(())
    }

    /// Rebuild the global hologram from scratch
    ///
    /// Use this after many deletes to clean up the global hologram.
    /// This is expensive: O(N * D) where N = records, D = dimensions.
    pub fn rebuild_hologram(&mut self) {
        self.hologram = BundleAccumulator::new(DIMENSION);
        for record in self.records.values() {
            self.hologram.add(&record.hologram);
        }
        eprintln!(
            "✓ Global hologram rebuilt from {} records",
            self.records.len()
        );
    }

    /// Check if the global hologram is usable for direct similarity queries
    pub fn is_hologram_usable(&self) -> bool {
        self.records.len() < GLOBAL_HOLOGRAM_SATURATION_LIMIT
    }

    // ==================== INCREMENTAL UPDATES ====================

    /// Update specific fields of an existing record without full re-encoding.
    ///
    /// This is more efficient than delete + re-insert because:
    /// 1. Only the affected indices are updated
    /// 2. The record hologram is incrementally updated
    /// 3. The columnar store is updated in-place
    ///
    /// # Arguments
    /// * `id` - The record ID to update
    /// * `updates` - Map of field names to new values
    ///
    /// # Example
    /// ```ignore
    /// let mut updates = HashMap::new();
    /// updates.insert("age".to_string(), Value::Int(31));
    /// store.update_fields(id, updates)?;
    /// ```
    pub fn update_fields(
        &mut self,
        id: RecordId,
        updates: HashMap<String, Value>,
    ) -> AmorphicResult<()> {
        // First, extract what we need from the record to avoid borrow conflicts
        let (old_values, updated_fields, old_hologram) = {
            let record = self
                .records
                .get_mut(&id)
                .ok_or(AmorphicError::RecordNotFound(id))?;

            // Track old values for index cleanup
            let mut old_values: HashMap<String, Value> = HashMap::new();
            for (field, new_value) in &updates {
                if let Some(old_value) = record.fields.get(field) {
                    old_values.insert(field.clone(), old_value.clone());
                }
                // Update the materialized field
                record.fields.insert(field.clone(), new_value.clone());
            }

            let updated_fields = record.fields.clone();
            let old_hologram = record.hologram.clone();

            (old_values, updated_fields, old_hologram)
        };

        // Re-encode the record hologram with updated fields
        // Note: We re-encode the entire record because majority voting
        // doesn't support true incremental updates. This is still faster
        // than delete + re-insert because we don't need to update all indices.
        let new_hologram = self.encode_record(&updated_fields);

        // Update the record's hologram
        if let Some(record) = self.records.get_mut(&id) {
            record.hologram = new_hologram.clone();
        }

        // Update hologram array
        if (id as usize) <= self.hologram_array.len() {
            self.hologram_array[(id - 1) as usize] = new_hologram.clone();
        }

        // Update global hologram: subtract old, add new
        self.hologram.subtract(&old_hologram);
        self.hologram.add(&new_hologram);

        // Update LSH tables - compute hashes first to avoid borrow conflicts
        let lsh_hashes_old: Vec<u64> = (0..self.lsh_bit_positions.len())
            .map(|i| self.lsh_hash_for_table(&old_hologram, i))
            .collect();
        let lsh_hashes_new: Vec<u64> = (0..self.lsh_bit_positions.len())
            .map(|i| self.lsh_hash_for_table(&new_hologram, i))
            .collect();

        for (table_idx, table) in self.lsh_tables.iter_mut().enumerate() {
            // Remove from old hash bucket
            if let Some(ids) = table.get_mut(&lsh_hashes_old[table_idx]) {
                ids.retain(|&x| x != id);
            }
            // Add to new hash bucket
            table.entry(lsh_hashes_new[table_idx]).or_default().push(id);
        }

        // Update indices for changed fields only
        for (field, new_value) in &updates {
            // Remove old value from indices
            if let Some(old_value) = old_values.get(field) {
                // Remove from value index
                let old_hash = Self::hash_value(old_value);
                if let Some(ids) = self.value_index.get_mut(&old_hash) {
                    ids.retain(|&x| x != id);
                }

                // Remove from numeric index
                if let Some(old_num) = old_value.as_f64() {
                    if let Some(btree) = self.numeric_index.get_mut(field) {
                        if let Some(ids) = btree.get_mut(&OrderedFloat(old_num)) {
                            ids.retain(|&x| x != id);
                        }
                    }
                }
            }

            // Add new value to indices
            let new_hash = Self::hash_value(new_value);
            self.value_index.entry(new_hash).or_default().push(id);

            if let Some(new_num) = new_value.as_f64() {
                self.numeric_index
                    .entry(field.clone())
                    .or_default()
                    .entry(OrderedFloat(new_num))
                    .or_default()
                    .push(id);
            }

            // Update name index if applicable
            if field == "name" || field == "_name" {
                // Remove old name
                if let Some(Value::String(old_name)) = old_values.get(field) {
                    self.name_to_id.remove(old_name);
                }
                // Add new name
                if let Value::String(new_name) = new_value {
                    self.name_to_id.insert(new_name.clone(), id);
                }
            }
        }

        // Update columnar store
        self.columnar.update_values(id, &updates);

        Ok(())
    }

    /// Remove a record from LSH tables
    /// Compute LSH hash for a specific table
    fn lsh_hash_for_table(&self, hv: &BinaryHV, table_idx: usize) -> u64 {
        let positions = &self.lsh_bit_positions[table_idx];
        let words = hv.as_words();
        let mut hash: u64 = 0;
        for (i, &pos) in positions.iter().enumerate() {
            if pos < DIMENSION {
                let word_idx = pos / 64;
                let bit_idx = pos % 64;
                if word_idx < words.len() && (words[word_idx] >> bit_idx) & 1 == 1 {
                    hash |= 1u64 << i;
                }
            }
        }
        hash
    }

    // ==================== COLUMNAR ANALYTICS (OLAP) ====================

    /// Get reference to columnar store for direct aggregate operations
    pub fn columnar(&self) -> &ColumnarStore {
        &self.columnar
    }

    /// SUM of a numeric column (O(1) - pre-computed)
    pub fn sum(&self, field: &str) -> Option<f64> {
        self.columnar.sum(field)
    }

    /// COUNT of a numeric column (O(1))
    pub fn count(&self, field: &str) -> Option<usize> {
        self.columnar.count(field)
    }

    /// AVG of a numeric column (O(1) - pre-computed)
    pub fn avg(&self, field: &str) -> Option<f64> {
        self.columnar.avg(field)
    }

    /// MIN of a numeric column (O(1) - pre-computed)
    pub fn min(&self, field: &str) -> Option<f64> {
        self.columnar.min(field)
    }

    /// MAX of a numeric column (O(1) - pre-computed)
    pub fn max(&self, field: &str) -> Option<f64> {
        self.columnar.max(field)
    }

    /// SUM with range filter (vectorizable scan)
    /// e.g., SUM(price) WHERE date >= 19940101 AND date < 19950101
    pub fn sum_where_range(
        &self,
        sum_field: &str,
        filter_field: &str,
        min: f64,
        max: f64,
    ) -> Option<f64> {
        self.columnar
            .sum_where_range(sum_field, filter_field, min, max)
    }

    /// COUNT with range filter
    pub fn count_where_range(&self, filter_field: &str, min: f64, max: f64) -> Option<usize> {
        self.columnar.count_where_range(filter_field, min, max)
    }

    /// Hash join between two columns
    /// Returns matched (build_record_id, probe_record_id) pairs
    pub fn hash_join(&self, build_field: &str, probe_field: &str) -> Option<JoinResult> {
        self.columnar.hash_join(build_field, probe_field)
    }

    /// Hash join with SUM aggregation on the probe side
    /// e.g., SUM(l_extendedprice) WHERE l_orderkey = o_orderkey
    pub fn hash_join_sum(
        &self,
        build_field: &str,
        probe_field: &str,
        sum_field: &str,
    ) -> Option<f64> {
        self.columnar
            .hash_join_sum(build_field, probe_field, sum_field)
    }

    /// Get columnar statistics
    pub fn columnar_stats(&self) -> ColumnarStats {
        self.columnar.stats()
    }

    /// Create a query optimizer for cost-based decisions
    pub fn optimizer(&self) -> QueryOptimizer<'_> {
        QueryOptimizer::new(&self.columnar)
    }

    /// Execute an optimized hash join (automatically selects build side)
    pub fn optimized_hash_join(&self, field_a: &str, field_b: &str) -> Option<JoinResult> {
        let optimizer = self.optimizer();
        let (build, probe) = optimizer.optimize_join(field_a, field_b);
        self.columnar.hash_join(&build, &probe)
    }

    /// Execute an optimized hash join with SUM aggregation
    pub fn optimized_hash_join_sum(
        &self,
        field_a: &str,
        field_b: &str,
        sum_field: &str,
    ) -> Option<f64> {
        let optimizer = self.optimizer();
        let (build, probe) = optimizer.optimize_join(field_a, field_b);
        self.columnar.hash_join_sum(&build, &probe, sum_field)
    }
}

/// Statistics about the store
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub record_count: usize,
    pub total_fields: usize,
    pub max_fields_per_record: usize,
    pub avg_fields_per_record: f64,
    pub lsh_table_count: usize,
    pub unique_field_names: usize,
    pub graph_edges: usize,
    pub hologram_saturation_pct: f64,
}

// =============================================================================
// THREAD-SAFE WRAPPER
// =============================================================================

use std::sync::RwLock;

/// Thread-safe wrapper for AmorphicStore
///
/// Provides coarse-grained locking for safe concurrent access.
/// - Multiple readers can query simultaneously
/// - Writers get exclusive access
///
/// # Example
/// ```ignore
/// let store = ConcurrentAmorphicStore::new();
///
/// // Multiple threads can read
/// let result = store.query_equals("name", &Value::String("Alice".into()));
///
/// // Writes are exclusive
/// store.ingest_json(r#"{"name": "Bob"}"#)?;
/// ```
pub struct ConcurrentAmorphicStore {
    inner: RwLock<AmorphicStore>,
}

impl ConcurrentAmorphicStore {
    /// Create a new thread-safe store
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(AmorphicStore::new()),
        }
    }

    /// Wrap an existing store
    pub fn from_store(store: AmorphicStore) -> Self {
        Self {
            inner: RwLock::new(store),
        }
    }

    // ==================== WRITE OPERATIONS (exclusive lock) ====================

    /// Ingest JSON document (requires write lock)
    pub fn ingest_json(&self, json: &str) -> AmorphicResult<RecordId> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .ingest_json(json)
    }

    /// Ingest row (requires write lock)
    pub fn ingest_row(&self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .ingest_row(columns, values)
    }

    /// Ingest edge (requires write lock)
    pub fn ingest_edge(
        &self,
        source: &str,
        relation: &str,
        target: &str,
    ) -> AmorphicResult<RecordId> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .ingest_edge(source, relation, target)
    }

    /// Delete record (requires write lock)
    pub fn delete(&self, id: RecordId) -> AmorphicResult<()> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .delete(id)
    }

    /// Rebuild hologram (requires write lock)
    pub fn rebuild_hologram(&self) -> AmorphicResult<()> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .rebuild_hologram();
        Ok(())
    }

    /// Update specific fields of a record (requires write lock)
    pub fn update_fields(
        &self,
        id: RecordId,
        updates: HashMap<String, Value>,
    ) -> AmorphicResult<()> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .update_fields(id, updates)
    }

    // ==================== READ OPERATIONS (shared lock) ====================

    /// Query by exact field-value match (read lock)
    pub fn query_equals(&self, field: &str, value: &Value) -> AmorphicResult<QueryResult> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.query_equals(field, value))
    }

    /// Query by range (read lock)
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> AmorphicResult<QueryResult> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.query_range(field, min, max))
    }

    /// Query similar records (read lock)
    pub fn query_similar_to(&self, name: &str, k: usize) -> AmorphicResult<QueryResult> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.query_similar_to(name, k))
    }

    /// Graph traversal (read lock)
    pub fn query_graph(
        &self,
        start: &str,
        relation: &str,
        depth: usize,
    ) -> AmorphicResult<QueryResult> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.query_graph(start, relation, depth))
    }

    /// SQL-like query (read lock)
    pub fn query_sql(&self, query: &str) -> AmorphicResult<QueryResult> {
        self.inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .query_sql(query)
    }

    /// Get record by ID (read lock)
    pub fn get(&self, id: RecordId) -> AmorphicResult<Option<AmorphicRecord>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.get(id).cloned())
    }

    /// Health check (read lock)
    pub fn health_check(&self) -> AmorphicResult<HealthStatus> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.health_check())
    }

    /// Get statistics (read lock)
    pub fn stats(&self) -> AmorphicResult<StoreStats> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.stats())
    }

    /// Get record count (read lock)
    pub fn len(&self) -> AmorphicResult<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.len())
    }

    /// Check if empty (read lock)
    pub fn is_empty(&self) -> AmorphicResult<bool> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.is_empty())
    }
}

impl Default for ConcurrentAmorphicStore {
    fn default() -> Self {
        Self::new()
    }
}

// Make ConcurrentAmorphicStore Send + Sync
unsafe impl Send for ConcurrentAmorphicStore {}
unsafe impl Sync for ConcurrentAmorphicStore {}

impl Default for AmorphicStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_ingestion() {
        let mut store = AmorphicStore::new();

        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        assert_eq!(id, 1);

        let record = store.get(id).unwrap();
        assert_eq!(
            record.get("name"),
            Some(&Value::String("Alice".to_string()))
        );
        assert_eq!(record.get("age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_row_ingestion() {
        let mut store = AmorphicStore::new();

        let id = store
            .ingest_row(&["name", "score"], &["Bob", "95"])
            .unwrap();

        let record = store.get(id).unwrap();
        assert_eq!(record.get("name"), Some(&Value::String("Bob".to_string())));
        assert_eq!(record.get("score"), Some(&Value::Int(95)));
    }

    #[test]
    fn test_edge_ingestion() {
        let mut store = AmorphicStore::new();

        // Create entities first
        store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
        store.ingest_json(r#"{"name": "Bob"}"#).unwrap();

        // Create edge
        store.ingest_edge("Alice", "KNOWS", "Bob").unwrap();

        // Verify edge exists
        let alice = store.get_by_name("Alice").unwrap();
        assert!(alice.edges().iter().any(|(rel, _)| rel == "KNOWS"));
    }

    #[test]
    fn test_multi_view_query() {
        let mut store = AmorphicStore::new();

        // Ingest same data
        store
            .ingest_json(r#"{"name": "Alice", "age": 30, "city": "NYC"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Bob", "age": 25, "city": "LA"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Carol", "age": 35, "city": "NYC"}"#)
            .unwrap();
        store.ingest_edge("Alice", "KNOWS", "Bob").unwrap();
        store.ingest_edge("Bob", "KNOWS", "Carol").unwrap();

        // Query as relational
        let result = store.query_range("age", 26.0, 40.0);
        assert_eq!(result.len(), 2); // Alice (30) and Carol (35)

        let rows = result.as_rows(&["name", "age"]);
        assert_eq!(rows.len(), 2);

        // Query as graph (same data!)
        let result = store.query_graph("Alice", "KNOWS", 2);
        assert!(result.len() >= 2); // Alice -> Bob -> Carol

        // Query as documents
        let docs = result.as_documents();
        assert!(!docs.is_empty());

        // Query by similarity (same data!)
        let alice = store.get_by_name("Alice").unwrap();
        let similar = store.query_similar(&alice.hologram, 0.5);
        assert!(!similar.is_empty());
    }

    #[test]
    fn test_sql_query() {
        let mut store = AmorphicStore::new();

        store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        store.ingest_json(r#"{"name": "Bob", "age": 25}"#).unwrap();

        // Query with WHERE
        let result = store.query_sql("SELECT * WHERE age > 27").unwrap();
        assert_eq!(result.len(), 1);

        let record = &result.records()[0];
        assert_eq!(
            record.get("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_time_series() {
        let mut store = AmorphicStore::new();

        let mut fields1 = HashMap::new();
        fields1.insert("sensor".to_string(), Value::String("temp".to_string()));
        fields1.insert("value".to_string(), Value::Float(22.5));
        store.ingest_timestamped(fields1, 1000).unwrap();

        let mut fields2 = HashMap::new();
        fields2.insert("sensor".to_string(), Value::String("temp".to_string()));
        fields2.insert("value".to_string(), Value::Float(23.1));
        store.ingest_timestamped(fields2, 2000).unwrap();

        let mut fields3 = HashMap::new();
        fields3.insert("sensor".to_string(), Value::String("temp".to_string()));
        fields3.insert("value".to_string(), Value::Float(21.8));
        store.ingest_timestamped(fields3, 3000).unwrap();

        // Query time range
        let result = store.query_time_range(1500, 2500);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_similarity_search() {
        let mut store = AmorphicStore::new();

        // Similar entities
        store
            .ingest_json(r#"{"name": "Apple", "type": "fruit", "color": "red"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Cherry", "type": "fruit", "color": "red"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Banana", "type": "fruit", "color": "yellow"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Carrot", "type": "vegetable", "color": "orange"}"#)
            .unwrap();

        // Find similar to Apple
        let result = store.query_similar_to("Apple", 2);
        assert_eq!(result.len(), 2);

        // Cherry should be most similar (same type + color)
        let names: Vec<&str> = result
            .records()
            .iter()
            .filter_map(|r| r.get("name")?.as_str())
            .collect();

        assert!(
            names.contains(&"Cherry"),
            "Cherry should be similar to Apple"
        );
    }

    #[test]
    fn test_global_hologram() {
        let mut store = AmorphicStore::new();

        store.ingest_json(r#"{"name": "A", "value": 1}"#).unwrap();
        store.ingest_json(r#"{"name": "B", "value": 2}"#).unwrap();

        // Global hologram is superposition of all records
        let hologram = store.hologram();

        // Should have non-trivial similarity to individual records
        let a = store.get_by_name("A").unwrap();
        let b = store.get_by_name("B").unwrap();

        let sim_a = hologram.similarity(&a.hologram);
        let sim_b = hologram.similarity(&b.hologram);

        // Both should contribute to the hologram
        assert!(sim_a > 0.5, "A should be in hologram");
        assert!(sim_b > 0.5, "B should be in hologram");
    }

    // ==================== GUARDRAIL TESTS ====================

    #[test]
    fn test_health_check_healthy() {
        let mut store = AmorphicStore::new();
        store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        store.ingest_json(r#"{"name": "Bob", "age": 25}"#).unwrap();

        assert_eq!(store.health_check(), HealthStatus::Healthy);
    }

    #[test]
    fn test_stats() {
        let mut store = AmorphicStore::new();
        store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Bob", "age": 25, "city": "NYC"}"#)
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.record_count, 2);
        assert_eq!(stats.max_fields_per_record, 3);
        assert!(stats.avg_fields_per_record > 2.0);
    }

    #[test]
    fn test_delete_record() {
        let mut store = AmorphicStore::new();
        let id1 = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        let id2 = store.ingest_json(r#"{"name": "Bob", "age": 25}"#).unwrap();

        assert_eq!(store.len(), 2);

        // Delete Alice
        store.delete(id1).unwrap();
        assert_eq!(store.len(), 1);

        // Alice should be gone
        assert!(store.get(id1).is_none());
        assert!(store.get_by_name("Alice").is_none());

        // Bob should still exist
        assert!(store.get(id2).is_some());
        assert!(store.get_by_name("Bob").is_some());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut store = AmorphicStore::new();
        store.ingest_json(r#"{"name": "Alice"}"#).unwrap();

        let result = store.delete(999);
        assert!(matches!(result, Err(AmorphicError::RecordNotFound(999))));
    }

    #[test]
    fn test_rebuild_hologram() {
        let mut store = AmorphicStore::new();
        store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
        let id2 = store.ingest_json(r#"{"name": "Bob"}"#).unwrap();
        store.ingest_json(r#"{"name": "Carol"}"#).unwrap();

        // Delete Bob
        store.delete(id2).unwrap();

        // Rebuild should work
        store.rebuild_hologram();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_is_hologram_usable() {
        let store = AmorphicStore::new();
        assert!(store.is_hologram_usable());
    }

    #[test]
    fn test_hierarchical_bundling() {
        let mut store = AmorphicStore::new();

        // Create a record with many fields (above MAX_FIELDS_WARNING)
        // Hierarchical bundling should handle this without error
        let mut fields = HashMap::new();
        for i in 0..400 {
            fields.insert(format!("field{}", i), Value::Int(i as i64));
        }

        // This should succeed with hierarchical bundling
        let result = store.ingest_fields(fields);
        assert!(result.is_ok());
        let id = result.unwrap();

        // Verify the record was stored correctly
        let record = store.get(id).unwrap();
        assert_eq!(record.get("field0"), Some(&Value::Int(0)));
        assert_eq!(record.get("field99"), Some(&Value::Int(99)));
        assert_eq!(record.get("field399"), Some(&Value::Int(399)));
    }

    #[test]
    fn test_hierarchical_bundling_very_large() {
        let mut store = AmorphicStore::new();

        // Test with 1000 fields - demonstrates unlimited field support
        let mut fields = HashMap::new();
        for i in 0..1000 {
            fields.insert(format!("f{}", i), Value::Int(i as i64));
        }

        let result = store.ingest_fields(fields);
        assert!(result.is_ok());
        let id = result.unwrap();

        // Verify random fields were stored
        let record = store.get(id).unwrap();
        assert_eq!(record.get("f0"), Some(&Value::Int(0)));
        assert_eq!(record.get("f500"), Some(&Value::Int(500)));
        assert_eq!(record.get("f999"), Some(&Value::Int(999)));
    }

    #[test]
    fn test_hierarchical_bundling_similarity() {
        let mut store = AmorphicStore::new();

        // Create two similar high-field records
        let mut fields1 = HashMap::new();
        let mut fields2 = HashMap::new();
        for i in 0..300 {
            fields1.insert(format!("field{}", i), Value::Int(i as i64));
            // Second record has same structure but different values
            fields2.insert(format!("field{}", i), Value::Int((i + 1) as i64));
        }

        let id1 = store.ingest_fields(fields1).unwrap();
        let id2 = store.ingest_fields(fields2).unwrap();

        // Both records should have holograms
        let rec1 = store.get(id1).unwrap();
        let rec2 = store.get(id2).unwrap();

        // They should be somewhat similar (same field structure)
        let similarity = rec1.hologram.similarity(&rec2.hologram);
        assert!(
            similarity > 0.3,
            "Similar high-field records should have some similarity: {}",
            similarity
        );
    }

    #[test]
    fn test_concurrent_store_basic() {
        let store = ConcurrentAmorphicStore::new();

        // Write
        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        // Read
        let record = store.get(id).unwrap().unwrap();
        assert_eq!(
            record.get("name"),
            Some(&Value::String("Alice".to_string()))
        );

        // Query
        let result = store
            .query_equals("name", &Value::String("Alice".to_string()))
            .unwrap();
        assert_eq!(result.len(), 1);

        // Stats
        let stats = store.stats().unwrap();
        assert_eq!(stats.record_count, 1);
    }

    // ==================== INCREMENTAL ENCODING TESTS ====================

    #[test]
    fn test_update_fields_basic() {
        let mut store = AmorphicStore::new();
        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        // Update age
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        store.update_fields(id, updates).unwrap();

        // Verify update
        let record = store.get(id).unwrap();
        assert_eq!(
            record.get("name"),
            Some(&Value::String("Alice".to_string()))
        );
        assert_eq!(record.get("age"), Some(&Value::Int(31)));
    }

    #[test]
    fn test_update_fields_index_update() {
        let mut store = AmorphicStore::new();
        store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        let id2 = store.ingest_json(r#"{"name": "Bob", "age": 25}"#).unwrap();

        // Update Bob's age
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(35));
        store.update_fields(id2, updates).unwrap();

        // Query should reflect update
        let result = store.query_range("age", 33.0, 40.0);
        assert_eq!(result.len(), 1);
        let record = &result.records()[0];
        assert_eq!(record.get("name"), Some(&Value::String("Bob".to_string())));
    }

    #[test]
    fn test_update_fields_name_index() {
        let mut store = AmorphicStore::new();
        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        // Old name should be findable
        assert!(store.get_by_name("Alice").is_some());

        // Update name
        let mut updates = HashMap::new();
        updates.insert("name".to_string(), Value::String("Alicia".to_string()));
        store.update_fields(id, updates).unwrap();

        // Old name should not be findable
        assert!(store.get_by_name("Alice").is_none());

        // New name should be findable
        let record = store.get_by_name("Alicia");
        assert!(record.is_some());
        assert_eq!(record.unwrap().get("age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_update_fields_hologram_update() {
        let mut store = AmorphicStore::new();
        let id1 = store
            .ingest_json(r#"{"name": "Alice", "city": "NYC"}"#)
            .unwrap();
        let _id2 = store
            .ingest_json(r#"{"name": "Bob", "city": "LA"}"#)
            .unwrap();

        // Get original hologram
        let orig_hologram = store.get(id1).unwrap().hologram.clone();

        // Update city
        let mut updates = HashMap::new();
        updates.insert("city".to_string(), Value::String("SF".to_string()));
        store.update_fields(id1, updates).unwrap();

        // Hologram should have changed
        let new_hologram = &store.get(id1).unwrap().hologram;
        let similarity = orig_hologram.similarity(new_hologram);
        // Should be similar but not identical (one field changed)
        assert!(
            similarity > 0.5 && similarity < 1.0,
            "Hologram should change after update: similarity = {}",
            similarity
        );
    }

    #[test]
    fn test_update_fields_nonexistent() {
        let mut store = AmorphicStore::new();
        store.ingest_json(r#"{"name": "Alice"}"#).unwrap();

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));

        let result = store.update_fields(999, updates);
        assert!(matches!(result, Err(AmorphicError::RecordNotFound(999))));
    }

    #[test]
    fn test_update_fields_add_new_field() {
        let mut store = AmorphicStore::new();
        let id = store.ingest_json(r#"{"name": "Alice"}"#).unwrap();

        // Add a new field (age didn't exist before)
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));
        store.update_fields(id, updates).unwrap();

        // Verify new field was added
        let record = store.get(id).unwrap();
        assert_eq!(record.get("age"), Some(&Value::Int(30)));

        // Field should be queryable
        let result = store.query_range("age", 25.0, 35.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_update_fields_columnar() {
        let mut store = AmorphicStore::new();
        let id = store
            .ingest_json(r#"{"name": "Alice", "score": 100}"#)
            .unwrap();

        // Check initial columnar value
        assert_eq!(store.columnar().sum("score"), Some(100.0));

        // Update score
        let mut updates = HashMap::new();
        updates.insert("score".to_string(), Value::Int(150));
        store.update_fields(id, updates).unwrap();

        // Columnar should reflect update
        assert_eq!(store.columnar().sum("score"), Some(150.0));
    }

    #[test]
    fn test_concurrent_update_fields() {
        let store = ConcurrentAmorphicStore::new();
        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        // Update via concurrent store
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        store.update_fields(id, updates).unwrap();

        // Verify
        let record = store.get(id).unwrap().unwrap();
        assert_eq!(record.get("age"), Some(&Value::Int(31)));
    }
}
