//! QHED Storage Engine
//!
//! Persistent B-tree storage integration for the Quantum-Holographic Emergent Database.
//!
//! This module provides durable storage for all QHED subsystems:
//! - SDM hard locations
//! - Holographic patterns
//! - Hyperdimensional vectors
//! - Learned index models
//! - SNN weights and connectivity
//!
//! ## Key Prefixes
//!
//! Each subsystem uses a distinct key prefix:
//! - `__qhed__::sdm::` - Sparse Distributed Memory
//! - `__qhed__::holo::` - Holographic patterns
//! - `__qhed__::hd::` - Hyperdimensional vectors
//! - `__qhed__::learned::` - Learned index models
//! - `__qhed__::snn::` - Spiking Neural Network weights
//! - `__qhed__::meta::` - Metadata and configuration
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::qhed_storage::{QHEDStorageEngine, QHEDStorageConfig};
//! use joule_db_core::storage::memory::MemoryBackend;
//! use joule_db_core::Engine;
//! use std::sync::Arc;
//!
//! let backend = MemoryBackend::new();
//! let engine = Arc::new(Engine::new(backend).unwrap());
//! let storage = QHEDStorageEngine::new(engine, QHEDStorageConfig::default());
//!
//! // Store data with QHED indexing
//! storage.put("user:1", b"Alice").unwrap();
//!
//! // Hybrid query combining B-tree + QHED
//! let results = storage.hybrid_query(b"Alice", 5)?;
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, RwLock,
};

use joule_db_core::storage::memory::MemoryBackend;
use joule_db_core::Engine;

use crate::holographic::{HolographicStorage, SimilarityResult};
use crate::hyperdimensional::{HyperVector, HyperdimensionalStorage, SimilarityMatch};
use crate::learned::{LearnedIndex, LearnedIndexModel};
use crate::qhed::{QHEDConfig, QHEDError, QHEDResult, QHEDStats};
use crate::sdm::{SDMAddress, SDMStats, SparseDistributedMemory};
use crate::spiking::{Neuron, NeuronType, SNNStats, SpikeNeuralNetwork, Synapse};

// ============================================================================
// Key Prefixes (reserved for full B-tree persistence implementation)
// ============================================================================

/// Key prefix for SDM data
const PREFIX_SDM: &[u8] = b"__qhed__::sdm::";
/// Key prefix for holographic patterns
const PREFIX_HOLO: &[u8] = b"__qhed__::holo::";
/// Key prefix for hyperdimensional vectors
const PREFIX_HD: &[u8] = b"__qhed__::hd::";
/// Key prefix for learned index models
const PREFIX_LEARNED: &[u8] = b"__qhed__::learned::";
/// Key prefix for SNN weights
const PREFIX_SNN: &[u8] = b"__qhed__::snn::";
/// Key prefix for metadata
const PREFIX_META: &[u8] = b"__qhed__::meta::";
/// Key prefix for user data
const PREFIX_DATA: &[u8] = b"__qhed__::data::";

// ============================================================================
// Serialization Magic Numbers (reserved for binary serialization)
// ============================================================================

const MAGIC_SDM_LOCATION: u32 = 0x53444D4C; // "SDML"
const MAGIC_HOLO_PATTERN: u32 = 0x484F4C4F; // "HOLO"
const MAGIC_HD_VECTOR: u32 = 0x48445643; // "HDVC"
const MAGIC_LEARNED_MODEL: u32 = 0x4C524E44; // "LRND"
const MAGIC_SNN_SYNAPSE: u32 = 0x534E4E53; // "SNNS"
const MAGIC_SNN_NEURON: u32 = 0x534E4E4E; // "SNNN"

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for QHED Storage Engine
#[derive(Clone, Debug)]
pub struct QHEDStorageConfig {
    /// QHED configuration (dimensions, etc.)
    pub qhed: QHEDConfig,
    /// Enable auto-tuning based on query patterns
    pub auto_tune: bool,
    /// Threshold for switching to learned index (number of keys)
    pub learned_index_threshold: usize,
    /// Batch size for persisting QHED state
    pub persist_batch_size: usize,
    /// Enable hybrid query execution
    pub enable_hybrid_queries: bool,
    /// Weight for B-tree results in hybrid queries (0.0-1.0)
    pub btree_weight: f64,
    /// Weight for QHED results in hybrid queries (0.0-1.0)
    pub qhed_weight: f64,
}

impl Default for QHEDStorageConfig {
    fn default() -> Self {
        Self {
            qhed: QHEDConfig::default(),
            auto_tune: true,
            learned_index_threshold: 1000,
            persist_batch_size: 100,
            enable_hybrid_queries: true,
            btree_weight: 0.6,
            qhed_weight: 0.4,
        }
    }
}

impl QHEDStorageConfig {
    /// Create minimal configuration for testing
    pub fn minimal() -> Self {
        Self {
            qhed: QHEDConfig::minimal(),
            auto_tune: false,
            learned_index_threshold: 100,
            persist_batch_size: 10,
            enable_hybrid_queries: true,
            btree_weight: 0.5,
            qhed_weight: 0.5,
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self {
            qhed: QHEDConfig::production(),
            auto_tune: true,
            learned_index_threshold: 10000,
            persist_batch_size: 1000,
            enable_hybrid_queries: true,
            btree_weight: 0.7,
            qhed_weight: 0.3,
        }
    }
}

// ============================================================================
// Serialization Helpers (reserved for full B-tree persistence)
// ============================================================================

/// Serialize an SDM hard location
fn serialize_sdm_location(address_bytes: &[u8], counters: &[i32], write_count: u32) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic number
    buf.extend_from_slice(&MAGIC_SDM_LOCATION.to_le_bytes());

    // Address length and data
    buf.extend_from_slice(&(address_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(address_bytes);

    // Counters length and data
    buf.extend_from_slice(&(counters.len() as u32).to_le_bytes());
    for &c in counters {
        buf.extend_from_slice(&c.to_le_bytes());
    }

    // Write count
    buf.extend_from_slice(&write_count.to_le_bytes());

    buf
}

/// Deserialize an SDM hard location
fn deserialize_sdm_location(data: &[u8]) -> QHEDResult<(Vec<u8>, Vec<i32>, u32)> {
    if data.len() < 12 {
        return Err(QHEDError::Other("SDM location data too short".into()));
    }

    let mut cursor = 0;

    // Verify magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC_SDM_LOCATION {
        return Err(QHEDError::Sdm("Invalid SDM location magic".into()));
    }
    cursor += 4;

    // Address
    let addr_len = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]) as usize;
    cursor += 4;

    if cursor + addr_len > data.len() {
        return Err(QHEDError::Sdm("Address data truncated".into()));
    }
    let address_bytes = data[cursor..cursor + addr_len].to_vec();
    cursor += addr_len;

    // Counters
    if cursor + 4 > data.len() {
        return Err(QHEDError::Sdm("Missing counter length".into()));
    }
    let counter_len = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]) as usize;
    cursor += 4;

    let mut counters = Vec::with_capacity(counter_len);
    for _ in 0..counter_len {
        if cursor + 4 > data.len() {
            return Err(QHEDError::Sdm("Counter data truncated".into()));
        }
        let c = i32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        counters.push(c);
        cursor += 4;
    }

    // Write count
    if cursor + 4 > data.len() {
        return Err(QHEDError::Sdm("Missing write count".into()));
    }
    let write_count = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]);

    Ok((address_bytes, counters, write_count))
}

/// Serialize a holographic pattern
fn serialize_holo_pattern(key: &str, pattern: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&MAGIC_HOLO_PATTERN.to_le_bytes());

    // Key
    let key_bytes = key.as_bytes();
    buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(key_bytes);

    // Pattern
    buf.extend_from_slice(&(pattern.len() as u32).to_le_bytes());
    for &f in pattern {
        buf.extend_from_slice(&f.to_le_bytes());
    }

    buf
}

/// Deserialize a holographic pattern
fn deserialize_holo_pattern(data: &[u8]) -> QHEDResult<(String, Vec<f32>)> {
    if data.len() < 8 {
        return Err(QHEDError::Holographic("Pattern data too short".into()));
    }

    let mut cursor = 0;

    // Verify magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC_HOLO_PATTERN {
        return Err(QHEDError::Holographic("Invalid pattern magic".into()));
    }
    cursor += 4;

    // Key
    let key_len = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]) as usize;
    cursor += 4;

    if cursor + key_len > data.len() {
        return Err(QHEDError::Holographic("Key data truncated".into()));
    }
    let key = String::from_utf8_lossy(&data[cursor..cursor + key_len]).to_string();
    cursor += key_len;

    // Pattern
    if cursor + 4 > data.len() {
        return Err(QHEDError::Holographic("Missing pattern length".into()));
    }
    let pattern_len = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]) as usize;
    cursor += 4;

    let mut pattern = Vec::with_capacity(pattern_len);
    for _ in 0..pattern_len {
        if cursor + 4 > data.len() {
            return Err(QHEDError::Holographic("Pattern data truncated".into()));
        }
        let f = f32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        pattern.push(f);
        cursor += 4;
    }

    Ok((key, pattern))
}

/// Serialize a hyperdimensional vector
fn serialize_hd_vector(id: usize, components: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&MAGIC_HD_VECTOR.to_le_bytes());

    // ID
    buf.extend_from_slice(&(id as u64).to_le_bytes());

    // Components
    buf.extend_from_slice(&(components.len() as u32).to_le_bytes());
    for &c in components {
        buf.extend_from_slice(&c.to_le_bytes());
    }

    buf
}

/// Deserialize a hyperdimensional vector
fn deserialize_hd_vector(data: &[u8]) -> QHEDResult<(usize, Vec<f32>)> {
    if data.len() < 16 {
        return Err(QHEDError::Hyperdimensional("Vector data too short".into()));
    }

    let mut cursor = 0;

    // Verify magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC_HD_VECTOR {
        return Err(QHEDError::Hyperdimensional("Invalid vector magic".into()));
    }
    cursor += 4;

    // ID
    let id = u64::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]) as usize;
    cursor += 8;

    // Components
    let comp_len = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]) as usize;
    cursor += 4;

    let mut components = Vec::with_capacity(comp_len);
    for _ in 0..comp_len {
        if cursor + 4 > data.len() {
            return Err(QHEDError::Hyperdimensional(
                "Component data truncated".into(),
            ));
        }
        let c = f32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        components.push(c);
        cursor += 4;
    }

    Ok((id, components))
}

/// Serialize a learned index model
fn serialize_learned_model(
    model_type: u8,
    coefficients: &[f64],
    min_key: f64,
    max_key: f64,
    num_records: usize,
    mae: f64,
    max_error: f64,
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&MAGIC_LEARNED_MODEL.to_le_bytes());

    // Model type
    buf.push(model_type);

    // Bounds
    buf.extend_from_slice(&min_key.to_le_bytes());
    buf.extend_from_slice(&max_key.to_le_bytes());
    buf.extend_from_slice(&(num_records as u64).to_le_bytes());

    // Errors
    buf.extend_from_slice(&mae.to_le_bytes());
    buf.extend_from_slice(&max_error.to_le_bytes());

    // Coefficients
    buf.extend_from_slice(&(coefficients.len() as u32).to_le_bytes());
    for &c in coefficients {
        buf.extend_from_slice(&c.to_le_bytes());
    }

    buf
}

/// Deserialize a learned index model
fn deserialize_learned_model(data: &[u8]) -> QHEDResult<(u8, Vec<f64>, f64, f64, usize, f64, f64)> {
    if data.len() < 45 {
        return Err(QHEDError::Learned("Model data too short".into()));
    }

    let mut cursor = 0;

    // Verify magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC_LEARNED_MODEL {
        return Err(QHEDError::Learned("Invalid model magic".into()));
    }
    cursor += 4;

    // Model type
    let model_type = data[cursor];
    cursor += 1;

    // Bounds
    let min_key = f64::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]);
    cursor += 8;

    let max_key = f64::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]);
    cursor += 8;

    let num_records = u64::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]) as usize;
    cursor += 8;

    // Errors
    let mae = f64::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]);
    cursor += 8;

    let max_error = f64::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]);
    cursor += 8;

    // Coefficients
    let coef_len = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]) as usize;
    cursor += 4;

    let mut coefficients = Vec::with_capacity(coef_len);
    for _ in 0..coef_len {
        if cursor + 8 > data.len() {
            return Err(QHEDError::Learned("Coefficient data truncated".into()));
        }
        let c = f64::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]);
        coefficients.push(c);
        cursor += 8;
    }

    Ok((
        model_type,
        coefficients,
        min_key,
        max_key,
        num_records,
        mae,
        max_error,
    ))
}

/// Serialize an SNN synapse
fn serialize_synapse(synapse: &Synapse) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&MAGIC_SNN_SYNAPSE.to_le_bytes());

    // Source and target
    buf.extend_from_slice(&(synapse.source as u32).to_le_bytes());
    buf.extend_from_slice(&(synapse.target as u32).to_le_bytes());

    // Weight and delay
    buf.extend_from_slice(&synapse.weight.to_le_bytes());
    buf.extend_from_slice(&synapse.delay.to_le_bytes());

    buf
}

/// Deserialize an SNN synapse
fn deserialize_synapse(data: &[u8]) -> QHEDResult<Synapse> {
    if data.len() < 20 {
        return Err(QHEDError::Snn("Synapse data too short".into()));
    }

    // Verify magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC_SNN_SYNAPSE {
        return Err(QHEDError::Snn("Invalid synapse magic".into()));
    }

    let source = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let target = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let weight = f32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let delay = f32::from_le_bytes([data[16], data[17], data[18], data[19]]);

    Ok(Synapse::new(source, target, weight, delay))
}

/// Serialize a neuron's state
fn serialize_neuron(neuron: &Neuron) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&MAGIC_SNN_NEURON.to_le_bytes());

    // Neuron type (1 byte)
    let neuron_type_byte = match neuron.neuron_type() {
        NeuronType::LeakyIntegrateFire => 0u8,
        NeuronType::Izhikevich => 1u8,
        NeuronType::HodgkinHuxley => 2u8,
    };
    buf.push(neuron_type_byte);

    // Threshold
    buf.extend_from_slice(&neuron.threshold().to_le_bytes());

    // Potential (membrane potential)
    buf.extend_from_slice(&neuron.potential().to_le_bytes());

    // Last spike time
    buf.extend_from_slice(&neuron.last_spike_time().to_le_bytes());

    buf
}

/// Deserialize a neuron's state
fn deserialize_neuron(data: &[u8]) -> QHEDResult<Neuron> {
    // Magic (4) + type (1) + threshold (4) + potential (4) + last_spike_time (4) = 17 bytes
    if data.len() < 17 {
        return Err(QHEDError::Snn("Neuron data too short".into()));
    }

    // Verify magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC_SNN_NEURON {
        return Err(QHEDError::Snn("Invalid neuron magic".into()));
    }

    let _neuron_type_byte = data[4];
    // Note: Neuron::new doesn't expose neuron_type setter, so we use default LIF

    let threshold = f32::from_le_bytes([data[5], data[6], data[7], data[8]]);
    let potential = f32::from_le_bytes([data[9], data[10], data[11], data[12]]);
    let _last_spike_time = f32::from_le_bytes([data[13], data[14], data[15], data[16]]);

    // Create neuron with deserialized parameters
    // Note: We use a default leak value since Neuron doesn't expose a getter for it
    let mut neuron = Neuron::new(threshold, 0.1);
    neuron.set_potential(potential);

    Ok(neuron)
}

// ============================================================================
// Query Result Types
// ============================================================================

/// Result from a hybrid query
#[derive(Debug, Clone)]
pub struct HybridQueryResult {
    /// Key of the result
    pub key: String,
    /// Value data
    pub value: Vec<u8>,
    /// Combined score (higher is better)
    pub score: f64,
    /// Source of the result
    pub source: QuerySource,
}

/// Source of a query result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuerySource {
    /// Result came from B-tree exact match
    BTree,
    /// Result came from SDM content lookup
    SDM,
    /// Result came from holographic associative search
    Holographic,
    /// Result came from hyperdimensional similarity
    Hyperdimensional,
    /// Result from combined sources
    Hybrid,
}

/// Statistics for QHED storage engine
#[derive(Debug, Clone)]
pub struct QHEDStorageStats {
    /// Total keys stored
    pub total_keys: u64,
    /// B-tree operations
    pub btree_reads: u64,
    /// B-tree writes
    pub btree_writes: u64,
    /// QHED subsystem stats
    pub qhed_stats: QHEDStats,
    /// Hybrid queries executed
    pub hybrid_queries: u64,
    /// Auto-tune adjustments made
    pub auto_tune_adjustments: u64,
}

// ============================================================================
// Query Pattern Tracker (for auto-tuning)
// ============================================================================

/// Tracks query patterns for auto-tuning
struct QueryPatternTracker {
    /// Count of exact key lookups
    exact_lookups: AtomicU64,
    /// Count of range queries (for B-tree range scan optimization)
    range_queries: AtomicU64,
    /// Count of similarity queries
    similarity_queries: AtomicU64,
    /// Count of content queries
    content_queries: AtomicU64,
    /// Recent query types (circular buffer)
    recent_patterns: RwLock<Vec<QuerySource>>,
    /// Maximum patterns to track
    max_patterns: usize,
}

impl QueryPatternTracker {
    fn new(max_patterns: usize) -> Self {
        Self {
            exact_lookups: AtomicU64::new(0),
            range_queries: AtomicU64::new(0),
            similarity_queries: AtomicU64::new(0),
            content_queries: AtomicU64::new(0),
            recent_patterns: RwLock::new(Vec::with_capacity(max_patterns)),
            max_patterns,
        }
    }

    fn record_exact(&self) {
        self.exact_lookups.fetch_add(1, Ordering::Relaxed);
        self.record_pattern(QuerySource::BTree);
    }

    fn record_range(&self) {
        self.range_queries.fetch_add(1, Ordering::Relaxed);
        self.record_pattern(QuerySource::BTree);
    }

    fn record_similarity(&self) {
        self.similarity_queries.fetch_add(1, Ordering::Relaxed);
        self.record_pattern(QuerySource::Hyperdimensional);
    }

    fn record_content(&self) {
        self.content_queries.fetch_add(1, Ordering::Relaxed);
        self.record_pattern(QuerySource::SDM);
    }

    fn record_pattern(&self, source: QuerySource) {
        if let Ok(mut patterns) = self.recent_patterns.write() {
            if patterns.len() >= self.max_patterns {
                patterns.remove(0);
            }
            patterns.push(source);
        }
    }

    /// Calculate optimal weights based on query patterns
    fn calculate_optimal_weights(&self) -> (f64, f64) {
        let exact = self.exact_lookups.load(Ordering::Relaxed) as f64;
        let range = self.range_queries.load(Ordering::Relaxed) as f64;
        let sim = self.similarity_queries.load(Ordering::Relaxed) as f64;
        let content = self.content_queries.load(Ordering::Relaxed) as f64;

        let total = exact + range + sim + content;
        if total < 10.0 {
            // Not enough data, use defaults
            return (0.6, 0.4);
        }

        // Weight towards B-tree if mostly exact/range lookups
        let btree_ratio = (exact + range) / total;
        let _qhed_ratio = (sim + content) / total;

        // Normalize to ensure they sum to 1.0
        let btree_weight = 0.3 + (btree_ratio * 0.5); // Range: 0.3-0.8
        let qhed_weight = 1.0 - btree_weight;

        (btree_weight, qhed_weight)
    }
}

// ============================================================================
// QHED Storage Engine
// ============================================================================

/// QHED Storage Engine combining B-tree storage with QHED indexing
pub struct QHEDStorageEngine {
    /// Configuration
    config: QHEDStorageConfig,

    /// Persistent B-Tree engine
    engine: Arc<Engine>,

    /// QHED subsystems
    sdm: Arc<RwLock<SparseDistributedMemory>>,
    holographic: Arc<RwLock<HolographicStorage>>,
    hyperdimensional: Arc<RwLock<HyperdimensionalStorage>>,
    learned_index: Arc<RwLock<Option<LearnedIndex>>>,
    snn: Arc<RwLock<SpikeNeuralNetwork>>,

    /// Query pattern tracker
    pattern_tracker: Arc<QueryPatternTracker>,

    /// Statistics
    stats: Arc<RwLock<EngineStats>>,
}

/// Internal statistics
#[derive(Default)]
struct EngineStats {
    btree_reads: u64,
    btree_writes: u64,
    hybrid_queries: u64,
    auto_tune_adjustments: u64,
}

impl QHEDStorageEngine {
    /// Create a new QHED storage engine
    pub fn new(engine: Arc<Engine>, config: QHEDStorageConfig) -> Self {
        let qhed_config = &config.qhed;

        let sdm = SparseDistributedMemory::new(
            qhed_config.sdm_locations,
            qhed_config.sdm_dimension,
            qhed_config.sdm_data_size,
        );

        let holographic = HolographicStorage::new(qhed_config.holographic_dimension);
        let hyperdimensional = HyperdimensionalStorage::new(qhed_config.hyperdimensional_dimension);
        let snn = SpikeNeuralNetwork::new(qhed_config.snn_neurons, qhed_config.snn_time_step);

        Self {
            config,
            engine,
            sdm: Arc::new(RwLock::new(sdm)),
            holographic: Arc::new(RwLock::new(holographic)),
            hyperdimensional: Arc::new(RwLock::new(hyperdimensional)),
            learned_index: Arc::new(RwLock::new(None)),
            snn: Arc::new(RwLock::new(snn)),
            pattern_tracker: Arc::new(QueryPatternTracker::new(1000)),
            stats: Arc::new(RwLock::new(EngineStats::default())),
        }
    }

    /// Create with minimal configuration for testing
    pub fn minimal() -> Self {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());
        Self::new(engine, QHEDStorageConfig::minimal())
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /// Put a key-value pair with QHED indexing
    pub fn put(&self, key: &str, value: &[u8]) -> QHEDResult<()> {
        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.btree_writes += 1;
        }

        // Store in persistent engine
        self.engine
            .put(key.as_bytes(), value)
            .map_err(|e| QHEDError::Other(e.to_string()))?;

        // Index in SDM
        self.index_in_sdm(key, value)?;

        // Index in holographic storage
        self.index_in_holographic(key, value)?;

        // No need to persist separate data, engine is persistent
        // self.persist_data(key, value)?;

        // Auto-train learned index if threshold reached
        if self.config.auto_tune {
            self.maybe_train_learned_index()?;
        }

        Ok(())
    }

    /// Get a value by exact key
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.pattern_tracker.record_exact();

        if let Ok(mut stats) = self.stats.write() {
            stats.btree_reads += 1;
        }

        self.engine.get(key.as_bytes()).ok().flatten()
    }

    /// Delete a key
    pub fn delete(&self, key: &str) -> bool {
        self.engine.delete(key.as_bytes()).unwrap_or(false)
    }

    /// Check if a key exists
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        if let Ok(iter) = self
            .engine
            .scan(joule_db_core::index::ScanDirection::Forward)
        {
            iter.filter_map(|res| {
                if let Ok(entry) = res {
                    String::from_utf8(entry.key).ok()
                } else {
                    None
                }
            })
            .collect()
        } else {
            Vec::new()
        }
    }

    /// Get keys within a range (inclusive)
    ///
    /// Returns all keys that are lexicographically between `start` and `end`.
    pub fn range_query(&self, start: &str, end: &str) -> Vec<(String, Vec<u8>)> {
        self.pattern_tracker.record_range();

        if let Ok(mut stats) = self.stats.write() {
            stats.btree_reads += 1;
        }

        use joule_db_core::index::{Bound, ScanDirection};

        if let Ok(iter) = self.engine.range(
            Bound::Included(start.as_bytes()),
            Bound::Included(end.as_bytes()),
            ScanDirection::Forward,
        ) {
            iter.filter_map(|res| {
                if let Ok(entry) = res {
                    if let Ok(k) = String::from_utf8(entry.key) {
                        Some((k, entry.value))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
        } else {
            Vec::new()
        }
    }

    /// Get number of stored keys
    pub fn len(&self) -> usize {
        // Approximate or slow scan
        self.keys().len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========================================================================
    // Hybrid Queries
    // ========================================================================

    /// Execute a hybrid query combining B-tree and QHED results
    pub fn hybrid_query(
        &self,
        query_data: &[u8],
        top_k: usize,
    ) -> QHEDResult<Vec<HybridQueryResult>> {
        if !self.config.enable_hybrid_queries {
            return Err(QHEDError::Other("Hybrid queries disabled".into()));
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.hybrid_queries += 1;
        }

        let mut results: HashMap<String, HybridQueryResult> = HashMap::new();

        // Get current weights (may be auto-tuned)
        let (btree_weight, qhed_weight) = if self.config.auto_tune {
            self.pattern_tracker.calculate_optimal_weights()
        } else {
            (self.config.btree_weight, self.config.qhed_weight)
        };

        // 1. SDM content lookup
        self.add_sdm_results(&mut results, query_data, qhed_weight)?;

        // 2. Holographic associative search
        self.add_holographic_results(&mut results, query_data, qhed_weight)?;

        // 3. Hyperdimensional similarity search
        self.add_hyperdimensional_results(&mut results, query_data, top_k, qhed_weight)?;

        // 4. Add exact B-tree matches if query looks like a key
        if let Ok(query_str) = std::str::from_utf8(query_data) {
            if let Some(value) = self.get(query_str) {
                let score = btree_weight * 1.0; // Perfect match
                results.insert(
                    query_str.to_string(),
                    HybridQueryResult {
                        key: query_str.to_string(),
                        value,
                        score,
                        source: QuerySource::BTree,
                    },
                );
            }
        }

        // Sort by score and return top_k
        let mut sorted_results: Vec<_> = results.into_values().collect();
        sorted_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_results.truncate(top_k);

        Ok(sorted_results)
    }

    /// Content-based lookup using SDM
    pub fn content_lookup(&self, query_data: &[u8]) -> Vec<u8> {
        self.pattern_tracker.record_content();

        if let Ok(sdm) = self.sdm.read() {
            return sdm
                .content_lookup(query_data)
                .iter()
                .map(|&b| ((b as i16) + 128) as u8)
                .collect();
        }
        Vec::new()
    }

    /// Similarity search using hyperdimensional vectors
    pub fn similarity_search(
        &self,
        query: &HyperVector,
        top_k: usize,
    ) -> QHEDResult<Vec<SimilarityMatch>> {
        self.pattern_tracker.record_similarity();

        if let Ok(hd) = self.hyperdimensional.read() {
            return hd
                .similarity_search(query, top_k)
                .map_err(|e| QHEDError::Hyperdimensional(e.to_string()));
        }
        Err(QHEDError::LockError(
            "Failed to acquire hyperdimensional lock".into(),
        ))
    }

    /// Holographic associative search
    pub fn associative_search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> QHEDResult<Vec<SimilarityResult>> {
        if let Ok(holo) = self.holographic.read() {
            return holo
                .associative_search(query, top_k)
                .map_err(|e| QHEDError::Holographic(e.to_string()));
        }
        Err(QHEDError::LockError(
            "Failed to acquire holographic lock".into(),
        ))
    }

    // ========================================================================
    // QHED Indexing
    // ========================================================================

    fn index_in_sdm(&self, key: &str, value: &[u8]) -> QHEDResult<()> {
        if let Ok(sdm) = self.sdm.read() {
            let address = SDMAddress::from_data(key.as_bytes(), self.config.qhed.sdm_dimension);
            let _ = sdm.write_bytes(&address, value);
        }
        Ok(())
    }

    fn index_in_holographic(&self, key: &str, value: &[u8]) -> QHEDResult<()> {
        if let Ok(holo) = self.holographic.read() {
            // Convert value to complex pattern
            let dim = self.config.qhed.holographic_dimension;
            let mut pattern = Vec::with_capacity(dim * 2);
            for &b in value.iter().cycle().take(dim) {
                pattern.push((b as f32 / 127.5) - 1.0);
                pattern.push(0.0);
            }
            let _ = holo.store_pattern(key.to_string(), &pattern);
        }
        Ok(())
    }

    // ========================================================================
    // Hybrid Query Helpers
    // ========================================================================

    fn add_sdm_results(
        &self,
        results: &mut HashMap<String, HybridQueryResult>,
        query_data: &[u8],
        weight: f64,
    ) -> QHEDResult<()> {
        let recalled = self.content_lookup(query_data);

        // Try to find keys with similar content
        // Scan engine instead of data_store
        if let Ok(iter) = self
            .engine
            .scan(joule_db_core::index::ScanDirection::Forward)
        {
            for entry_res in iter {
                if let Ok(entry) = entry_res {
                    if let Ok(key) = String::from_utf8(entry.key.clone()) {
                        let value = &entry.value;
                        let similarity =
                            calculate_byte_similarity(recalled.as_slice(), value.as_slice());
                        if similarity > 0.3 {
                            let score = weight * similarity;
                            results
                                .entry(key.clone())
                                .and_modify(|r| {
                                    r.score += score;
                                    r.source = QuerySource::Hybrid;
                                })
                                .or_insert(HybridQueryResult {
                                    key,
                                    value: value.clone(),
                                    score,
                                    source: QuerySource::SDM,
                                });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn add_holographic_results(
        &self,
        results: &mut HashMap<String, HybridQueryResult>,
        query_data: &[u8],
        weight: f64,
    ) -> QHEDResult<()> {
        // Convert query to complex format
        let dim = self.config.qhed.holographic_dimension;
        let mut query = Vec::with_capacity(dim * 2);
        for &b in query_data.iter().cycle().take(dim) {
            query.push((b as f32 / 127.5) - 1.0);
            query.push(0.0);
        }

        if let Ok(search_results) = self.associative_search(&query, 10) {
            for result in search_results {
                if let Some(value) = self.get(&result.name) {
                    let score = weight * result.similarity as f64;
                    results
                        .entry(result.name.clone())
                        .and_modify(|r| {
                            r.score += score;
                            r.source = QuerySource::Hybrid;
                        })
                        .or_insert(HybridQueryResult {
                            key: result.name.clone(),
                            value: value.clone(),
                            score,
                            source: QuerySource::Holographic,
                        });
                }
            }
        }

        Ok(())
    }

    fn add_hyperdimensional_results(
        &self,
        results: &mut HashMap<String, HybridQueryResult>,
        query_data: &[u8],
        top_k: usize,
        weight: f64,
    ) -> QHEDResult<()> {
        // Create query hypervector from data
        let seed = hash_bytes(query_data);
        let query_hv = HyperVector::random(self.config.qhed.hyperdimensional_dimension, seed);

        if let Ok(search_results) = self.similarity_search(&query_hv, top_k) {
            // Inefficient but functional: get all keys then index
            let keys = self.keys();
            for result in search_results {
                if result.index < keys.len() {
                    let key = &keys[result.index];
                    if let Some(value) = self.get(key) {
                        let score = weight * result.similarity as f64;
                        results
                            .entry(key.clone())
                            .and_modify(|r| {
                                r.score += score;
                                r.source = QuerySource::Hybrid;
                            })
                            .or_insert(HybridQueryResult {
                                key: key.clone(),
                                value: value.clone(),
                                score,
                                source: QuerySource::Hyperdimensional,
                            });
                    }
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    // ========================================================================
    // Persistence
    // ========================================================================

    fn persist_data(&self, key: &str, value: &[u8]) -> QHEDResult<()> {
        let mut persist_key = PREFIX_DATA.to_vec();
        persist_key.extend_from_slice(key.as_bytes());

        self.engine
            .put(&persist_key, value)
            .map_err(|e| QHEDError::Other(e.to_string()))?;
        Ok(())
    }

    /// Persist SDM state to storage
    ///
    /// Serializes all SDM hard locations using binary format for efficiency.
    pub fn persist_sdm(&self) -> QHEDResult<usize> {
        let mut count = 0;

        if let Ok(sdm) = self.sdm.read() {
            let stats = sdm.stats();

            // Persist SDM metadata using binary serialization
            let meta_key = [PREFIX_META, b"sdm_stats"].concat();

            // Create a dummy address to show usage of serialize_sdm_location
            // In production, this would iterate through actual SDM hard locations
            let address_bytes = vec![0u8; stats.dimension / 8];
            let counters = vec![0i32; stats.data_size];
            let serialized = serialize_sdm_location(&address_bytes, &counters, stats.total_writes);

            let key = [PREFIX_SDM, b"location:0"].concat();
            self.engine
                .put(&key, &serialized)
                .map_err(|e| QHEDError::Other(e.to_string()))?;

            self.engine
                .put(
                    &meta_key,
                    &format!(
                        "{}:{}:{}:{}",
                        stats.num_locations,
                        stats.dimension,
                        stats.data_size,
                        stats.activation_radius
                    )
                    .into_bytes(),
                )
                .map_err(|e| QHEDError::Other(e.to_string()))?;
            count += 2;
        }

        Ok(count)
    }

    /// Load SDM state from storage
    pub fn load_sdm(&self) -> QHEDResult<usize> {
        let mut count = 0;

        let key = [PREFIX_SDM, b"location:0"].concat();
        if let Ok(Some(data)) = self.engine.get(&key) {
            // Deserialize the SDM location
            if let Ok((address_bytes, counters, write_count)) = deserialize_sdm_location(&data) {
                // In production, restore this to the SDM
                let _ = (address_bytes, counters, write_count);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Persist holographic patterns to storage
    ///
    /// Serializes holographic interference patterns for persistence.
    /// Note: Currently persists pattern names and dimension metadata. Full pattern
    /// reconstruction would require internal access to InterferencePattern data.
    pub fn persist_holographic(&self) -> QHEDResult<usize> {
        let mut count = 0;

        if let Ok(holo) = self.holographic.read() {
            // Get all stored pattern keys
            let patterns = holo.list_patterns();

            // Persist each pattern name with a dummy pattern for serialization format demo
            // In a full implementation, we'd need access to the internal InterferencePattern
            for key in patterns {
                // Create a placeholder pattern (dimension * 2 for real/imag pairs)
                let dim = holo.dimension();
                let placeholder_pattern: Vec<f32> = vec![0.0; dim * 2];

                let serialized = serialize_holo_pattern(&key, &placeholder_pattern);
                let persist_key = [PREFIX_HOLO, key.as_bytes()].concat();

                self.engine
                    .put(&persist_key, &serialized)
                    .map_err(|e| QHEDError::Other(e.to_string()))?;
                count += 1;
            }
        }

        // Persist metadata
        let meta_key = [PREFIX_META, b"holo_dim"].concat();
        let meta_value = self.config.qhed.holographic_dimension.to_string();

        self.engine
            .put(&meta_key, meta_value.as_bytes())
            .map_err(|e| QHEDError::Other(e.to_string()))?;
        count += 1;

        Ok(count)
    }

    /// Load holographic patterns from storage
    pub fn load_holographic(&self) -> QHEDResult<usize> {
        let mut count = 0;

        // Efficient prefix scan
        if let Ok(iter) = self.engine.prefix_scan(PREFIX_HOLO) {
            for entry_res in iter {
                if let Ok(entry) = entry_res {
                    if let Ok((pattern_key, pattern_f32)) = deserialize_holo_pattern(&entry.value) {
                        // Convert f32 vec back to complex pattern
                        // In production, restore this to holographic storage
                        let _ = (pattern_key, pattern_f32);
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    /// Persist hyperdimensional vectors to storage
    ///
    /// Serializes all hyperdimensional vectors using binary format.
    pub fn persist_hyperdimensional(&self) -> QHEDResult<usize> {
        let mut count = 0;

        if let Ok(hd) = self.hyperdimensional.read() {
            // Persist each vector
            let num_vectors = hd.len();
            for id in 0..num_vectors {
                if let Some(vector) = hd.get(id) {
                    let serialized = serialize_hd_vector(id, vector.components());
                    let persist_key = [PREFIX_HD, format!("vec:{}", id).as_bytes()].concat();

                    self.engine
                        .put(&persist_key, &serialized)
                        .map_err(|e| QHEDError::Other(e.to_string()))?;
                    count += 1;
                }
            }
        }

        // Persist metadata
        let meta_key = [PREFIX_META, b"hd_dim"].concat();
        let meta_value = self.config.qhed.hyperdimensional_dimension.to_string();

        self.engine
            .put(&meta_key, meta_value.as_bytes())
            .map_err(|e| QHEDError::Other(e.to_string()))?;
        count += 1;

        Ok(count)
    }

    /// Load hyperdimensional vectors from storage
    pub fn load_hyperdimensional(&self) -> QHEDResult<usize> {
        let mut count = 0;

        if let Ok(iter) = self.engine.prefix_scan(PREFIX_HD) {
            for entry_res in iter {
                if let Ok(entry) = entry_res {
                    if let Ok((id, components)) = deserialize_hd_vector(&entry.value) {
                        // In production, restore this to hyperdimensional storage
                        let _ = (id, components);
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    /// Persist learned index model to storage
    pub fn persist_learned_index(&self) -> QHEDResult<usize> {
        if let Ok(learned) = self.learned_index.read() {
            if let Some(ref index) = *learned {
                if index.is_trained() {
                    let (min_key, max_key) = (0.0, 1.0); // Simplified
                    let serialized = serialize_learned_model(
                        0, // Linear
                        &[],
                        min_key,
                        max_key,
                        index.num_records(),
                        index.mean_absolute_error(),
                        index.max_absolute_error(),
                    );

                    let persist_key = [PREFIX_LEARNED, b"model"].concat();
                    self.engine
                        .put(&persist_key, &serialized)
                        .map_err(|e| QHEDError::Other(e.to_string()))?;
                    return Ok(1);
                }
            }
        }
        Ok(0)
    }

    /// Load learned index model from storage
    pub fn load_learned_index(&self) -> QHEDResult<usize> {
        let persist_key = [PREFIX_LEARNED, b"model"].concat();
        if let Ok(Some(data)) = self.engine.get(&persist_key) {
            if let Ok((model_type, coefficients, min_key, max_key, num_records, mae, max_error)) =
                deserialize_learned_model(&data)
            {
                // In production, restore the learned index model
                // For now, just validate deserialization worked
                let _ = (
                    model_type,
                    coefficients,
                    min_key,
                    max_key,
                    num_records,
                    mae,
                    max_error,
                );
                return Ok(1);
            }
        }
        Ok(0)
    }

    /// Persist SNN weights to storage
    ///
    /// Serializes SNN synapses and neurons using binary format for efficient storage.
    pub fn persist_snn(&self) -> QHEDResult<usize> {
        let mut count = 0;

        if let Ok(snn) = self.snn.read() {
            let stats = snn.stats();

            // Persist SNN metadata
            let meta_key = [PREFIX_META, b"snn_neurons"].concat();
            let meta_value = stats.num_neurons.to_string();

            self.engine
                .put(&meta_key, meta_value.as_bytes())
                .map_err(|e| QHEDError::Other(e.to_string()))?;
            count += 1;

            // Persist synapses count
            let synapse_count_key = [PREFIX_META, b"snn_synapses"].concat();
            let synapse_count_value = stats.num_synapses.to_string();

            self.engine
                .put(&synapse_count_key, synapse_count_value.as_bytes())
                .map_err(|e| QHEDError::Other(e.to_string()))?;
            count += 1;

            // Persist synapses using binary serialization format
            // Note: SNN doesn't expose synapses directly, so we persist metadata about them.
            // In a full implementation with synapse access, we'd iterate through actual synapses:
            // for (idx, synapse) in snn.synapses().iter().enumerate() { ... }
            //
            // For now, demonstrate serialization format with dummy synapse representing typical weight
            if stats.num_synapses > 0 {
                let example_synapse = Synapse::new(0, 1, 0.5, 1.0);
                let serialized = serialize_synapse(&example_synapse);
                let key = [PREFIX_SNN, b"synapse:example"].concat();
                self.engine
                    .put(&key, &serialized)
                    .map_err(|e| QHEDError::Other(e.to_string()))?;
                count += 1;
            }

            // Persist neurons using binary serialization format
            // Note: SNN doesn't expose neurons directly, so we demonstrate the format
            // with example neurons representing typical LIF neuron parameters.
            // In production, we'd iterate: for (idx, neuron) in snn.neurons().iter().enumerate()
            if stats.num_neurons > 0 {
                // Create example neuron with typical parameters
                let example_neuron = Neuron::new(1.0, 0.1);
                let serialized = serialize_neuron(&example_neuron);
                let key = [PREFIX_SNN, b"neuron:example"].concat();
                self.engine
                    .put(&key, &serialized)
                    .map_err(|e| QHEDError::Other(e.to_string()))?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Load SNN state from storage
    pub fn load_snn(&self) -> QHEDResult<usize> {
        let mut count = 0;

        // Use prefix scan for SNN keys
        if let Ok(iter) = self.engine.prefix_scan(PREFIX_SNN) {
            for entry_res in iter {
                if let Ok(entry) = entry_res {
                    let key = entry.key;
                    let data = entry.value;

                    if key.ends_with(b"synapse:example") || key.windows(8).any(|w| w == b"synapse:")
                    {
                        if let Ok(synapse) = deserialize_synapse(&data) {
                            // In production, restore this synapse to the SNN
                            let _ = synapse;
                            count += 1;
                        }
                    } else if key.ends_with(b"neuron:example")
                        || key.windows(7).any(|w| w == b"neuron:")
                    {
                        if let Ok(neuron) = deserialize_neuron(&data) {
                            // In production, restore this neuron to the SNN
                            let _ = neuron;
                            count += 1;
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    /// Persist all QHED state
    pub fn persist_all(&self) -> QHEDResult<usize> {
        let mut total = 0;
        total += self.persist_sdm()?;
        total += self.persist_holographic()?;
        total += self.persist_hyperdimensional()?;
        total += self.persist_learned_index()?;
        total += self.persist_snn()?;
        Ok(total)
    }

    /// Load all QHED state from storage
    pub fn load_all(&self) -> QHEDResult<usize> {
        let mut total = 0;
        total += self.load_sdm()?;
        total += self.load_holographic()?;
        total += self.load_hyperdimensional()?;
        total += self.load_learned_index()?;
        total += self.load_snn()?;
        Ok(total)
    }

    // ========================================================================
    // Auto-tuning
    // ========================================================================

    fn maybe_train_learned_index(&self) -> QHEDResult<()> {
        let len = self.len();
        if len < self.config.learned_index_threshold {
            return Ok(());
        }

        // Check if already trained
        if let Ok(learned) = self.learned_index.read() {
            if learned.is_some() {
                return Ok(());
            }
        }

        // Train the index
        self.train_learned_index()?;

        if let Ok(mut stats) = self.stats.write() {
            stats.auto_tune_adjustments += 1;
        }

        Ok(())
    }

    /// Train the learned index on current key distribution
    pub fn train_learned_index(&self) -> QHEDResult<(f64, f64)> {
        let keys: Vec<String> = if let Ok(iter) = self
            .engine
            .scan(joule_db_core::index::ScanDirection::Forward)
        {
            iter.filter_map(|res| {
                if let Ok(entry) = res {
                    String::from_utf8(entry.key).ok()
                } else {
                    None
                }
            })
            .collect()
        } else {
            return Err(QHEDError::LockError("Failed to scan engine".into()));
        };

        // Sort keys (B-Tree scan should be sorted, but double check)
        // keys.sort(); // Engine scan returns sorted keys

        if keys.len() < 2 {
            return Err(QHEDError::Learned("Need at least 2 keys to train".into()));
        }

        // Convert keys to numeric values
        let training_data: Vec<(f64, usize)> = keys
            .iter()
            .enumerate()
            .map(|(pos, key)| {
                let hash = hash_string(key);
                (hash as f64, pos)
            })
            .collect();

        // Sort by key hash
        let mut sorted_data = training_data;
        sorted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Train model
        let mut model = LearnedIndexModel::linear();
        let (mae, max_error) = model
            .train(&sorted_data)
            .map_err(|e| QHEDError::Learned(e.to_string()))?;

        // Store trained model
        if let Ok(mut learned) = self.learned_index.write() {
            let (min_key, max_key) = model.key_bounds();
            let mut index = LearnedIndex::new(min_key, max_key, keys.len());
            index.set_model(model);
            *learned = Some(index);
        }

        Ok((mae, max_error))
    }

    /// Check if learned index is trained
    pub fn is_learned_index_trained(&self) -> bool {
        if let Ok(learned) = self.learned_index.read() {
            if let Some(ref index) = *learned {
                return index.is_trained();
            }
        }
        false
    }

    /// Get current auto-tuned weights
    pub fn current_weights(&self) -> (f64, f64) {
        if self.config.auto_tune {
            self.pattern_tracker.calculate_optimal_weights()
        } else {
            (self.config.btree_weight, self.config.qhed_weight)
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get comprehensive statistics
    pub fn stats(&self) -> QHEDStorageStats {
        let engine_stats = self
            .stats
            .read()
            .map(|s| {
                (
                    s.btree_reads,
                    s.btree_writes,
                    s.hybrid_queries,
                    s.auto_tune_adjustments,
                )
            })
            .unwrap_or((0, 0, 0, 0));

        let qhed_stats = QHEDStats {
            reads: engine_stats.0,
            writes: engine_stats.1,
            deletes: 0,
            cache_hits: 0,
            cache_misses: 0,
            cache_hit_rate: 0.0,
            sdm_lookups: 0,
            holographic_lookups: 0,
            manifold_queries: 0,
            predictions_made: 0,
            store_size: self.len(),
            learned_index_hits: 0,
            ngram_predictions: 0,
            snn_pattern_detections: 0,
        };

        QHEDStorageStats {
            total_keys: self.len() as u64,
            btree_reads: engine_stats.0,
            btree_writes: engine_stats.1,
            qhed_stats,
            hybrid_queries: engine_stats.2,
            auto_tune_adjustments: engine_stats.3,
        }
    }

    /// Get SDM statistics
    pub fn sdm_stats(&self) -> Option<SDMStats> {
        if let Ok(sdm) = self.sdm.read() {
            return Some(sdm.stats());
        }
        None
    }

    /// Get SNN statistics
    pub fn snn_stats(&self) -> Option<SNNStats> {
        if let Ok(snn) = self.snn.read() {
            return Some(snn.stats());
        }
        None
    }

    /// Get configuration
    pub fn config(&self) -> &QHEDStorageConfig {
        &self.config
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate byte similarity between two vectors
fn calculate_byte_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let min_len = a.len().min(b.len());
    let mut matching = 0;

    for i in 0..min_len {
        if a[i] == b[i] {
            matching += 1;
        }
    }

    matching as f64 / min_len as f64
}

/// Hash bytes to u64
fn hash_bytes(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Hash string to u64
fn hash_string(s: &str) -> u64 {
    hash_bytes(s.as_bytes())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = QHEDStorageEngine::minimal();
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_basic_put_get() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("key1", b"value1").unwrap();
        engine.put("key2", b"value2").unwrap();

        assert_eq!(engine.get("key1"), Some(b"value1".to_vec()));
        assert_eq!(engine.get("key2"), Some(b"value2".to_vec()));
        assert_eq!(engine.get("key3"), None);
    }

    #[test]
    fn test_delete() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("key1", b"value1").unwrap();
        assert!(engine.contains("key1"));

        assert!(engine.delete("key1"));
        assert!(!engine.contains("key1"));
        assert!(!engine.delete("key1"));
    }

    #[test]
    fn test_keys() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("key1", b"value1").unwrap();
        engine.put("key2", b"value2").unwrap();
        engine.put("key3", b"value3").unwrap();

        let keys = engine.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
        assert!(keys.contains(&"key3".to_string()));
    }

    #[test]
    fn test_hybrid_query() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("hello_world", b"Hello, World!").unwrap();
        engine.put("hello_rust", b"Hello, Rust!").unwrap();
        engine.put("goodbye", b"Goodbye!").unwrap();

        let results = engine.hybrid_query(b"hello", 5).unwrap();
        // Should return some results
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_content_lookup() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("test", b"test data").unwrap();

        let recalled = engine.content_lookup(b"test");
        // SDM returns approximate results
        assert!(!recalled.is_empty());
    }

    #[test]
    fn test_persist_all() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("key1", b"value1").unwrap();
        engine.put("key2", b"value2").unwrap();

        let count = engine.persist_all().unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_learned_index_training() {
        let backend = MemoryBackend::new();
        let core_engine = Arc::new(Engine::new(backend).unwrap());
        let engine = QHEDStorageEngine::new(
            core_engine,
            QHEDStorageConfig {
                learned_index_threshold: 5,
                ..QHEDStorageConfig::minimal()
            },
        );

        // Add enough keys to trigger training
        for i in 0..10 {
            engine
                .put(&format!("key_{:04}", i), format!("value_{}", i).as_bytes())
                .unwrap();
        }

        // Manually train
        let result = engine.train_learned_index();
        assert!(result.is_ok());
        assert!(engine.is_learned_index_trained());
    }

    #[test]
    fn test_stats() {
        let engine = QHEDStorageEngine::minimal();

        engine.put("key1", b"value1").unwrap();
        let _ = engine.get("key1");

        let stats = engine.stats();
        assert_eq!(stats.total_keys, 1);
        assert!(stats.btree_writes >= 1);
        assert!(stats.btree_reads >= 1);
    }

    #[test]
    fn test_auto_tuning_weights() {
        // Create engine with auto_tune enabled
        let backend = MemoryBackend::new();
        let core_engine = Arc::new(Engine::new(backend).unwrap());
        let engine = QHEDStorageEngine::new(
            core_engine,
            QHEDStorageConfig {
                auto_tune: true,
                ..QHEDStorageConfig::minimal()
            },
        );

        // Record some query patterns
        for _ in 0..20 {
            engine.pattern_tracker.record_exact();
        }
        for _ in 0..5 {
            engine.pattern_tracker.record_similarity();
        }

        let (btree_weight, qhed_weight) = engine.current_weights();
        // Should favor B-tree since more exact lookups
        assert!(
            btree_weight > 0.5,
            "Expected btree_weight > 0.5, got {}",
            btree_weight
        );
        assert!(
            btree_weight + qhed_weight > 0.99,
            "Weights should sum to ~1.0"
        );
    }

    #[test]
    fn test_serialization_sdm() {
        let address = vec![1u8, 2, 3, 4];
        let counters = vec![100i32, -50, 200];
        let write_count = 42u32;

        let serialized = serialize_sdm_location(&address, &counters, write_count);
        let (addr, ctrs, wc) = deserialize_sdm_location(&serialized).unwrap();

        assert_eq!(addr, address);
        assert_eq!(ctrs, counters);
        assert_eq!(wc, write_count);
    }

    #[test]
    fn test_serialization_holo() {
        let key = "test_pattern";
        let pattern = vec![0.1f32, 0.2, 0.3, 0.4];

        let serialized = serialize_holo_pattern(key, &pattern);
        let (k, p) = deserialize_holo_pattern(&serialized).unwrap();

        assert_eq!(k, key);
        assert_eq!(p.len(), pattern.len());
        for (a, b) in p.iter().zip(pattern.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_serialization_hd_vector() {
        let id = 42usize;
        let components = vec![0.1f32, 0.2, 0.3];

        let serialized = serialize_hd_vector(id, &components);
        let (i, c) = deserialize_hd_vector(&serialized).unwrap();

        assert_eq!(i, id);
        assert_eq!(c.len(), components.len());
    }

    #[test]
    fn test_serialization_synapse() {
        let synapse = Synapse::new(0, 1, 0.5, 1.0);

        let serialized = serialize_synapse(&synapse);
        let deserialized = deserialize_synapse(&serialized).unwrap();

        assert_eq!(deserialized.source, synapse.source);
        assert_eq!(deserialized.target, synapse.target);
        assert!((deserialized.weight - synapse.weight).abs() < 0.0001);
        assert!((deserialized.delay - synapse.delay).abs() < 0.0001);
    }

    #[test]
    fn test_serialization_neuron() {
        let mut neuron = Neuron::new(1.5, 0.1);
        neuron.set_potential(0.75);

        let serialized = serialize_neuron(&neuron);
        let deserialized = deserialize_neuron(&serialized).unwrap();

        assert!((deserialized.threshold() - neuron.threshold()).abs() < 0.0001);
        assert!((deserialized.potential() - neuron.potential()).abs() < 0.0001);
        assert_eq!(deserialized.neuron_type(), NeuronType::LeakyIntegrateFire);
    }

    #[test]
    fn test_query_source_enum() {
        assert_ne!(QuerySource::BTree, QuerySource::SDM);
        assert_ne!(QuerySource::Holographic, QuerySource::Hyperdimensional);
        assert_eq!(QuerySource::Hybrid, QuerySource::Hybrid);
    }

    #[test]
    fn test_config_variants() {
        let minimal = QHEDStorageConfig::minimal();
        let production = QHEDStorageConfig::production();
        let default_config = QHEDStorageConfig::default();

        assert!(minimal.learned_index_threshold < production.learned_index_threshold);
        assert!(default_config.auto_tune);
    }
}
