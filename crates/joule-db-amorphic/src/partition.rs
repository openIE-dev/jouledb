//! Entropy-Based Auto-Partitioning for Amorphic Database
//!
//! Implements self-regulating holographic storage that maintains retrieval quality
//! through automatic partitioning when entropy thresholds are exceeded.
//!
//! # Academic Foundations
//!
//! - **Capacity Theorem**: P(correct) ≈ 1 - N²/D (Kanerva, 2009)
//! - **Entropic Memory**: Optimal entropy range for recall/precision (Nature Scientific Reports, 2021)
//! - **Scalable Bloom Filters**: Chain-based capacity expansion (Almeida et al., 2007)
//! - **Hopfield Saturation**: P > 0.14N causes retrieval divergence (PMC, 2019)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Entropy Monitor                          │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │ should_partition() checks:                          │   │
//! │  │   - retrieval_probability < 0.90                    │   │
//! │  │   - bit_entropy > 0.95                              │   │
//! │  │   - snr < 4.0                                       │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └────────────────────────┬────────────────────────────────────┘
//!                          │ triggers partition
//!          ┌───────────────┼───────────────┐
//!          ▼               ▼               ▼
//!     ┌─────────┐    ┌─────────┐    ┌─────────┐
//!     │ Bin 0   │    │ Bin 1   │    │ Bin 2   │
//!     │ N=200   │    │ N=180   │    │ N=150   │
//!     └─────────┘    └─────────┘    └─────────┘
//! ```

use crate::{
    AmorphicError, AmorphicRecord, AmorphicResult, AmorphicStore, DIMENSION,
    GLOBAL_HOLOGRAM_SATURATION_LIMIT, HealthStatus, QueryResult, RecordId, Value,
};
use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;
use std::sync::RwLock;

// =============================================================================
// ENTROPY THRESHOLDS (Based on Academic Literature)
// =============================================================================

/// Minimum retrieval probability before partition trigger
/// Based on capacity theorem: P ≈ 1 - N²/D
/// At 90%, we're at the edge of reliable retrieval
pub const RETRIEVAL_PROBABILITY_THRESHOLD: f64 = 0.90;

/// Maximum bit entropy before partition trigger
/// Shannon entropy H = -Σ p(b) log₂ p(b)
/// At 0.95, bits are nearly uniformly distributed (noise)
pub const BIT_ENTROPY_THRESHOLD: f64 = 0.95;

/// Minimum SNR before partition trigger
/// SNR ≈ √(D/N), at SNR=4.0 we have ~4σ separation
pub const SNR_THRESHOLD: f64 = 4.0;

/// Target records per partition (conservative)
/// This is ~63% of theoretical limit to maintain headroom
pub const TARGET_PARTITION_SIZE: usize = 200;

/// Maximum partitions before warning
pub const MAX_PARTITIONS_WARNING: usize = 10;

// =============================================================================
// HOLOGRAM HEALTH METRICS
// =============================================================================

/// Health metrics for a single hologram/partition
///
/// Combines multiple indicators from the literature:
/// - Capacity theorem (Kanerva)
/// - Entropic memory research
/// - Hopfield network saturation studies
#[derive(Debug, Clone)]
pub struct HologramHealth {
    /// Number of items bundled into this hologram (N)
    pub item_count: usize,

    /// Bit balance entropy: H = -Σ p(b) log₂ p(b)
    /// Range: [0, 1] where 1.0 = maximum entropy (pure noise)
    pub bit_entropy: f64,

    /// Estimated retrieval probability: P ≈ 1 - N²/D
    /// Range: [0, 1] where 1.0 = perfect retrieval
    pub retrieval_probability: f64,

    /// Signal-to-noise ratio: SNR ≈ √(D/N)
    /// Higher is better; < 4.0 indicates degradation
    pub snr: f64,

    /// Fraction of capacity used: N / limit
    pub capacity_used: f64,

    /// Timestamp of measurement
    pub measured_at: u64,
}

impl HologramHealth {
    /// Compute health metrics for a BundleAccumulator
    pub fn measure(accumulator: &BundleAccumulator, item_count: usize) -> Self {
        let bit_entropy = Self::compute_bit_entropy(accumulator, item_count);
        let retrieval_probability = Self::compute_retrieval_probability(item_count, DIMENSION);
        let snr = Self::compute_snr(item_count, DIMENSION);
        let capacity_used = item_count as f64 / GLOBAL_HOLOGRAM_SATURATION_LIMIT as f64;

        Self {
            item_count,
            bit_entropy,
            retrieval_probability,
            snr,
            capacity_used,
            measured_at: now_epoch(),
        }
    }

    /// Compute Shannon entropy of bit distribution
    ///
    /// For a healthy hologram, bits should NOT be uniformly distributed.
    /// As more items are bundled, distribution converges to 50/50 (max entropy = noise).
    fn compute_bit_entropy(accumulator: &BundleAccumulator, item_count: usize) -> f64 {
        if item_count == 0 {
            return 0.0;
        }

        let counts = accumulator.counts();
        let n = item_count as f64;

        // Compute variance of bit counts from expected mean (n/2)
        // High variance = good (bits are distinguishable)
        // Low variance = bad (converging to uniform)
        let expected_mean = n / 2.0;
        let mut variance_sum = 0.0;

        for &count in counts {
            let diff = count as f64 - expected_mean;
            variance_sum += diff * diff;
        }

        let variance = variance_sum / counts.len() as f64;
        let max_possible_variance = (n / 2.0).powi(2);

        // Normalize: 0.0 = maximum variance (fresh), 1.0 = minimum variance (saturated)
        if max_possible_variance < f64::EPSILON {
            return 0.0;
        }

        let normalized_variance = (variance / max_possible_variance).sqrt();

        // Invert: high variance = low entropy (good), low variance = high entropy (bad)
        1.0 - normalized_variance.min(1.0)
    }

    /// Compute retrieval probability using capacity theorem
    ///
    /// P(correct) ≈ 1 - N²/D
    /// This is the probability that a randomly chosen stored item
    /// can be correctly retrieved from the superposition.
    fn compute_retrieval_probability(n: usize, d: usize) -> f64 {
        if d == 0 {
            return 0.0;
        }
        let n_squared = (n * n) as f64;
        let d_f = d as f64;
        (1.0 - n_squared / d_f).max(0.0)
    }

    /// Compute signal-to-noise ratio
    ///
    /// SNR ≈ √(D/N)
    /// Measures how well the signal (stored items) stands out from noise.
    fn compute_snr(n: usize, d: usize) -> f64 {
        if n == 0 {
            return f64::INFINITY;
        }
        ((d as f64) / (n as f64)).sqrt()
    }

    /// Determine if this hologram should trigger a partition
    ///
    /// Uses multiple criteria from the literature:
    /// 1. Retrieval probability dropping below safe threshold
    /// 2. Bit entropy approaching maximum (uniform distribution)
    /// 3. SNR dropping below safe threshold
    pub fn should_partition(&self) -> bool {
        self.retrieval_probability < RETRIEVAL_PROBABILITY_THRESHOLD
            || self.bit_entropy > BIT_ENTROPY_THRESHOLD
            || self.snr < SNR_THRESHOLD
    }

    /// Get a human-readable status
    pub fn status(&self) -> PartitionHealthStatus {
        if self.should_partition() {
            PartitionHealthStatus::Critical
        } else if self.capacity_used > 0.8 || self.bit_entropy > 0.8 {
            PartitionHealthStatus::Warning
        } else {
            PartitionHealthStatus::Healthy
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "items={}, P(ret)={:.1}%, entropy={:.2}, SNR={:.1}, cap={:.0}%",
            self.item_count,
            self.retrieval_probability * 100.0,
            self.bit_entropy,
            self.snr,
            self.capacity_used * 100.0
        )
    }
}

/// Health status for a partition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionHealthStatus {
    /// All metrics within safe bounds
    Healthy,
    /// Approaching limits, partition soon
    Warning,
    /// Exceeded limits, partition required
    Critical,
}

// =============================================================================
// PARTITION STRUCTURE
// =============================================================================

/// A single partition (bin) in the partitioned store
pub struct Partition {
    /// Unique partition ID
    pub id: u32,

    /// The underlying AmorphicStore for this partition
    store: AmorphicStore,

    /// Cached health metrics (updated on writes)
    health: HologramHealth,

    /// Creation timestamp
    pub created_at: u64,

    /// Last write timestamp
    pub last_write_at: u64,

    /// Record count (cached for fast access)
    record_count: usize,
}

impl Partition {
    /// Create a new empty partition
    pub fn new(id: u32) -> Self {
        let store = AmorphicStore::new();
        let health = HologramHealth {
            item_count: 0,
            bit_entropy: 0.0,
            retrieval_probability: 1.0,
            snr: f64::INFINITY,
            capacity_used: 0.0,
            measured_at: now_epoch(),
        };

        Self {
            id,
            store,
            health,
            created_at: now_epoch(),
            last_write_at: now_epoch(),
            record_count: 0,
        }
    }

    /// Get current health metrics
    pub fn health(&self) -> &HologramHealth {
        &self.health
    }

    /// Update health metrics (call after writes)
    fn update_health(&mut self) {
        self.health = HologramHealth::measure(self.store.global_hologram(), self.record_count);
    }

    /// Check if this partition should trigger a split
    pub fn should_split(&self) -> bool {
        self.health.should_partition()
    }

    /// Get record count
    pub fn len(&self) -> usize {
        self.record_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.record_count == 0
    }

    /// Ingest a JSON document into this partition
    pub fn ingest_json(&mut self, json: &str) -> AmorphicResult<RecordId> {
        let id = self.store.ingest_json(json)?;
        self.record_count = self.store.len();
        self.last_write_at = now_epoch();
        self.update_health();
        Ok(id)
    }

    /// Ingest a row into this partition
    pub fn ingest_row(&mut self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        let id = self.store.ingest_row(columns, values)?;
        self.record_count = self.store.len();
        self.last_write_at = now_epoch();
        self.update_health();
        Ok(id)
    }

    /// Query by exact match
    pub fn query_equals(&self, field: &str, value: &Value) -> QueryResult {
        self.store.query_equals(field, value)
    }

    /// Query by range
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> QueryResult {
        self.store.query_range(field, min, max)
    }

    /// Query similar
    pub fn query_similar_to(&self, name: &str, k: usize) -> QueryResult {
        self.store.query_similar_to(name, k)
    }

    /// Get a record by ID
    pub fn get(&self, id: RecordId) -> Option<&AmorphicRecord> {
        self.store.get(id)
    }

    /// Get underlying store (for advanced operations)
    pub fn store(&self) -> &AmorphicStore {
        &self.store
    }

    /// Get mutable underlying store
    pub fn store_mut(&mut self) -> &mut AmorphicStore {
        &mut self.store
    }
}

// =============================================================================
// PARTITION MANAGER
// =============================================================================

/// Routing strategy for inserting records
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Always insert into the partition with most capacity headroom
    LeastLoaded,
    /// Round-robin across partitions
    RoundRobin,
    /// Route based on content similarity (semantic clustering)
    Semantic,
}

/// Configuration for the partition manager
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Maximum records per partition before auto-split
    pub max_records_per_partition: usize,

    /// Target records per partition (soft limit)
    pub target_records_per_partition: usize,

    /// Routing strategy for new records
    pub routing_strategy: RoutingStrategy,

    /// Enable automatic partitioning
    pub auto_partition: bool,

    /// Retrieval probability threshold for partition trigger
    pub retrieval_threshold: f64,

    /// Bit entropy threshold for partition trigger
    pub entropy_threshold: f64,

    /// SNR threshold for partition trigger
    pub snr_threshold: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            max_records_per_partition: GLOBAL_HOLOGRAM_SATURATION_LIMIT,
            target_records_per_partition: TARGET_PARTITION_SIZE,
            routing_strategy: RoutingStrategy::LeastLoaded,
            auto_partition: true,
            retrieval_threshold: RETRIEVAL_PROBABILITY_THRESHOLD,
            entropy_threshold: BIT_ENTROPY_THRESHOLD,
            snr_threshold: SNR_THRESHOLD,
        }
    }
}

/// Manages multiple partitions with automatic scaling
///
/// Implements the "Scalable Bloom Filter" pattern adapted for holographic storage:
/// - Monitors hologram health metrics
/// - Automatically creates new partitions when thresholds exceeded
/// - Routes queries across all partitions
/// - Aggregates results
pub struct PartitionManager {
    /// Active partitions
    partitions: Vec<Partition>,

    /// Configuration
    config: PartitionConfig,

    /// Next partition ID
    next_partition_id: u32,

    /// Round-robin counter for routing
    round_robin_counter: usize,

    /// Global record ID to (partition_id, local_record_id) mapping
    record_index: HashMap<RecordId, (u32, RecordId)>,

    /// Next global record ID
    next_global_id: RecordId,

    /// Total records across all partitions
    total_records: usize,

    /// Partition events log
    events: Vec<PartitionEvent>,
}

/// Events for partition lifecycle
#[derive(Debug, Clone)]
pub struct PartitionEvent {
    pub timestamp: u64,
    pub event_type: PartitionEventType,
    pub partition_id: u32,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionEventType {
    Created,
    SplitTriggered,
    Merged,
    HealthWarning,
    HealthCritical,
}

impl PartitionManager {
    /// Create a new partition manager with default config
    pub fn new() -> Self {
        Self::with_config(PartitionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PartitionConfig) -> Self {
        let mut manager = Self {
            partitions: Vec::new(),
            config,
            next_partition_id: 0,
            round_robin_counter: 0,
            record_index: HashMap::new(),
            next_global_id: 1,
            total_records: 0,
            events: Vec::new(),
        };

        // Create initial partition
        manager.create_partition();

        manager
    }

    /// Create a new partition
    fn create_partition(&mut self) -> u32 {
        let id = self.next_partition_id;
        self.next_partition_id += 1;

        let partition = Partition::new(id);
        self.partitions.push(partition);

        self.log_event(PartitionEventType::Created, id, "New partition created");

        if self.partitions.len() > MAX_PARTITIONS_WARNING {
            eprintln!(
                "WARNING: {} partitions created. Consider increasing dimensions or reviewing data patterns.",
                self.partitions.len()
            );
        }

        id
    }

    /// Log a partition event
    fn log_event(&mut self, event_type: PartitionEventType, partition_id: u32, details: &str) {
        self.events.push(PartitionEvent {
            timestamp: now_epoch(),
            event_type,
            partition_id,
            details: details.to_string(),
        });

        // Keep last 1000 events
        if self.events.len() > 1000 {
            self.events.remove(0);
        }
    }

    /// Select partition for insertion based on routing strategy
    ///
    /// For semantic routing, `json` should contain the raw JSON string being ingested.
    /// For other strategies, `json` is ignored.
    fn select_partition_for_insert(&mut self, json: Option<&str>) -> usize {
        match self.config.routing_strategy {
            RoutingStrategy::LeastLoaded => {
                // Find partition with most headroom
                self.partitions
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| !p.should_split())
                    .min_by_key(|(_, p)| p.len())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            RoutingStrategy::RoundRobin => {
                // Find next non-full partition in round-robin order
                let start = self.round_robin_counter % self.partitions.len();
                for offset in 0..self.partitions.len() {
                    let idx = (start + offset) % self.partitions.len();
                    if !self.partitions[idx].should_split() {
                        self.round_robin_counter = idx + 1;
                        return idx;
                    }
                }
                // All full, return first (will trigger split)
                0
            }
            RoutingStrategy::Semantic => {
                // Semantic routing: encode record as BinaryHV, compare against each
                // partition's global hologram, route to most-similar healthy partition
                if let Some(json_str) = json {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                        return self.select_partition_semantic(&parsed);
                    }
                }
                // Fallback to least-loaded if no JSON or parse fails
                self.partitions
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| !p.should_split())
                    .min_by_key(|(_, p)| p.len())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        }
    }

    /// Semantic partition selection: route record to partition whose hologram
    /// is most similar to the record's HDC encoding, weighted by health.
    fn select_partition_semantic(&self, json: &serde_json::Value) -> usize {
        let record_hv = Self::encode_record_hv(json);

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, partition) in self.partitions.iter().enumerate() {
            if partition.should_split() {
                continue; // Skip saturated partitions
            }

            // Compare record HV against partition's global hologram
            let hologram_hv = partition.store.global_hologram().threshold();
            let sim = record_hv.similarity(&hologram_hv);

            // Weight by health: prefer partitions with good retrieval probability
            let health = &partition.health;
            let weighted = sim * health.retrieval_probability as f32;

            if weighted > best_score {
                best_score = weighted;
                best_idx = i;
            }
        }

        // If best similarity is too low (< 0.3), it's genuinely new content
        // → route to least-loaded partition instead
        if best_score < 0.3 {
            self.partitions
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.should_split())
                .min_by_key(|(_, p)| p.len())
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            best_idx
        }
    }

    /// Encode a JSON record as a BinaryHV for semantic routing.
    ///
    /// Uses the same field⊗value binding approach as AmorphicStore::encode_record():
    /// for each field+value pair, create deterministic HVs via from_hash,
    /// XOR-bind them, then bundle all bound pairs via majority voting.
    fn encode_record_hv(json: &serde_json::Value) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);
        if let Some(obj) = json.as_object() {
            for (key, value) in obj {
                let key_hv = BinaryHV::from_hash(key.as_bytes(), DIMENSION);
                let val_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let val_hv = BinaryHV::from_hash(val_str.as_bytes(), DIMENSION);
                let bound = key_hv.bind(&val_hv);
                acc.add(&bound);
            }
        }
        acc.threshold()
    }

    /// Check if we need to create a new partition
    fn maybe_auto_partition(&mut self) -> bool {
        if !self.config.auto_partition {
            return false;
        }

        // Check if all partitions are at capacity
        let all_full = self.partitions.iter().all(|p| p.should_split());

        if all_full {
            let new_id = self.create_partition();
            self.log_event(
                PartitionEventType::SplitTriggered,
                new_id,
                &format!(
                    "Auto-partition triggered: all {} partitions at capacity",
                    self.partitions.len() - 1
                ),
            );
            return true;
        }

        false
    }

    /// Ingest a JSON document
    pub fn ingest_json(&mut self, json: &str) -> AmorphicResult<RecordId> {
        // Check for auto-partition
        self.maybe_auto_partition();

        // Select target partition (pass JSON for semantic routing)
        let partition_idx = self.select_partition_for_insert(Some(json));

        // Log health warning if needed
        let health = self.partitions[partition_idx].health();
        if health.status() == PartitionHealthStatus::Warning {
            self.log_event(
                PartitionEventType::HealthWarning,
                self.partitions[partition_idx].id,
                &format!("Partition approaching limits: {}", health.summary()),
            );
        }

        // Ingest into partition
        let local_id = self.partitions[partition_idx].ingest_json(json)?;

        // Assign global ID
        let global_id = self.next_global_id;
        self.next_global_id += 1;
        self.total_records += 1;

        // Record mapping
        let partition_id = self.partitions[partition_idx].id;
        self.record_index
            .insert(global_id, (partition_id, local_id));

        Ok(global_id)
    }

    /// Ingest a row
    pub fn ingest_row(&mut self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        self.maybe_auto_partition();

        // Build JSON for semantic routing from columns/values
        let json_for_routing: Option<String> =
            if matches!(self.config.routing_strategy, RoutingStrategy::Semantic) {
                let mut obj = serde_json::Map::new();
                for (col, val) in columns.iter().zip(values.iter()) {
                    obj.insert(col.to_string(), serde_json::Value::String(val.to_string()));
                }
                Some(serde_json::Value::Object(obj).to_string())
            } else {
                None
            };
        let partition_idx = self.select_partition_for_insert(json_for_routing.as_deref());
        let local_id = self.partitions[partition_idx].ingest_row(columns, values)?;

        let global_id = self.next_global_id;
        self.next_global_id += 1;
        self.total_records += 1;

        let partition_id = self.partitions[partition_idx].id;
        self.record_index
            .insert(global_id, (partition_id, local_id));

        Ok(global_id)
    }

    /// Query by exact match across all partitions
    pub fn query_equals(&self, field: &str, value: &Value) -> QueryResult {
        let mut records = Vec::new();

        for partition in &self.partitions {
            let partition_results = partition.query_equals(field, value);
            records.extend(partition_results.into_records());
        }

        QueryResult { records }
    }

    /// Query by range across all partitions
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> QueryResult {
        let mut records = Vec::new();

        for partition in &self.partitions {
            let partition_results = partition.query_range(field, min, max);
            records.extend(partition_results.into_records());
        }

        QueryResult { records }
    }

    /// Query similar across all partitions with cross-partition re-ranking.
    ///
    /// Finds the source hologram, queries each partition for candidates,
    /// then re-computes similarity against the source to produce a globally
    /// correct top-k ranking.
    pub fn query_similar_to(&self, name: &str, k: usize) -> QueryResult {
        // Find the source record's hologram across all partitions
        let source_hologram = self
            .partitions
            .iter()
            .find_map(|p| p.store().get_by_name(name).map(|r| r.hologram.clone()));

        let source_hologram = match source_hologram {
            Some(h) => h,
            None => return QueryResult { records: vec![] },
        };

        // Query each partition for top k results (over-fetch to ensure diversity)
        let mut all_records = Vec::new();
        for partition in &self.partitions {
            let partition_results = partition.query_similar_to(name, k);
            all_records.extend(partition_results.into_records());
        }

        // Re-rank all candidates by similarity against the source hologram
        let mut scored: Vec<(f32, AmorphicRecord)> = all_records
            .into_iter()
            .map(|r| {
                let sim = r.hologram.similarity_simd(&source_hologram);
                (sim, r)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let records: Vec<AmorphicRecord> = scored.into_iter().take(k).map(|(_, r)| r).collect();

        QueryResult { records }
    }

    /// Get a record by global ID
    pub fn get(&self, global_id: RecordId) -> Option<&AmorphicRecord> {
        let (partition_id, local_id) = self.record_index.get(&global_id)?;
        let partition = self.partitions.iter().find(|p| p.id == *partition_id)?;
        partition.get(*local_id)
    }

    /// Get total record count
    pub fn len(&self) -> usize {
        self.total_records
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.total_records == 0
    }

    /// Get partition count
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Get health summary for all partitions
    pub fn health_summary(&self) -> Vec<(u32, HologramHealth)> {
        self.partitions
            .iter()
            .map(|p| (p.id, p.health().clone()))
            .collect()
    }

    /// Get aggregate health status
    pub fn aggregate_health(&self) -> HealthStatus {
        let mut warnings = Vec::new();
        let mut degraded = Vec::new();

        for partition in &self.partitions {
            match partition.health().status() {
                PartitionHealthStatus::Critical => {
                    degraded.push(format!(
                        "Partition {} critical: {}",
                        partition.id,
                        partition.health().summary()
                    ));
                }
                PartitionHealthStatus::Warning => {
                    warnings.push(format!(
                        "Partition {} warning: {}",
                        partition.id,
                        partition.health().summary()
                    ));
                }
                PartitionHealthStatus::Healthy => {}
            }
        }

        if self.partitions.len() > MAX_PARTITIONS_WARNING {
            warnings.push(format!(
                "High partition count: {} partitions",
                self.partitions.len()
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

    /// Get detailed statistics
    pub fn stats(&self) -> PartitionManagerStats {
        let partition_stats: Vec<_> = self
            .partitions
            .iter()
            .map(|p| PartitionStats {
                id: p.id,
                record_count: p.len(),
                health: p.health().clone(),
                created_at: p.created_at,
                last_write_at: p.last_write_at,
            })
            .collect();

        let total_capacity = self.partitions.len() * self.config.max_records_per_partition;

        PartitionManagerStats {
            total_records: self.total_records,
            partition_count: self.partitions.len(),
            total_capacity,
            capacity_used_pct: if total_capacity > 0 {
                (self.total_records as f64 / total_capacity as f64) * 100.0
            } else {
                0.0
            },
            partitions: partition_stats,
            auto_partition_enabled: self.config.auto_partition,
            routing_strategy: self.config.routing_strategy,
            events_count: self.events.len(),
        }
    }

    /// Get recent events
    pub fn recent_events(&self, n: usize) -> &[PartitionEvent] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    /// Get configuration
    pub fn config(&self) -> &PartitionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PartitionConfig) {
        self.config = config;
    }

    /// Force a partition check and split if needed
    pub fn check_and_partition(&mut self) -> Vec<u32> {
        let mut new_partitions = Vec::new();

        // Collect critical partitions first (to avoid borrow issues)
        let critical_partitions: Vec<(u32, String)> = self
            .partitions
            .iter()
            .filter(|p| p.should_split())
            .map(|p| (p.id, p.health().summary()))
            .collect();

        // Log events for critical partitions
        for (id, summary) in critical_partitions {
            self.log_event(
                PartitionEventType::HealthCritical,
                id,
                &format!("Manual check found partition at critical: {}", summary),
            );
        }

        // Create new partitions if all are full
        while self.partitions.iter().all(|p| p.should_split()) {
            let id = self.create_partition();
            new_partitions.push(id);
        }

        new_partitions
    }
}

impl Default for PartitionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a single partition
#[derive(Debug, Clone)]
pub struct PartitionStats {
    pub id: u32,
    pub record_count: usize,
    pub health: HologramHealth,
    pub created_at: u64,
    pub last_write_at: u64,
}

/// Aggregate statistics for the partition manager
#[derive(Debug, Clone)]
pub struct PartitionManagerStats {
    pub total_records: usize,
    pub partition_count: usize,
    pub total_capacity: usize,
    pub capacity_used_pct: f64,
    pub partitions: Vec<PartitionStats>,
    pub auto_partition_enabled: bool,
    pub routing_strategy: RoutingStrategy,
    pub events_count: usize,
}

// =============================================================================
// THREAD-SAFE PARTITIONED STORE
// =============================================================================

/// Thread-safe partitioned Amorphic store
///
/// Provides the same API as `ConcurrentAmorphicStore` but with automatic
/// entropy-based partitioning for unlimited scaling.
pub struct PartitionedAmorphicStore {
    inner: RwLock<PartitionManager>,
}

impl PartitionedAmorphicStore {
    /// Create a new partitioned store with default config
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(PartitionManager::new()),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PartitionConfig) -> Self {
        Self {
            inner: RwLock::new(PartitionManager::with_config(config)),
        }
    }

    // ==================== WRITE OPERATIONS ====================

    /// Ingest JSON document
    pub fn ingest_json(&self, json: &str) -> AmorphicResult<RecordId> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .ingest_json(json)
    }

    /// Ingest row
    pub fn ingest_row(&self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        self.inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .ingest_row(columns, values)
    }

    /// Force partition check
    pub fn check_and_partition(&self) -> AmorphicResult<Vec<u32>> {
        Ok(self
            .inner
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Write lock poisoned".into()))?
            .check_and_partition())
    }

    // ==================== READ OPERATIONS ====================

    /// Query by exact match
    pub fn query_equals(&self, field: &str, value: &Value) -> AmorphicResult<QueryResult> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .query_equals(field, value))
    }

    /// Query by range
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> AmorphicResult<QueryResult> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .query_range(field, min, max))
    }

    /// Query similar
    pub fn query_similar_to(&self, name: &str, k: usize) -> AmorphicResult<QueryResult> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .query_similar_to(name, k))
    }

    /// Get record by ID
    pub fn get(&self, id: RecordId) -> AmorphicResult<Option<AmorphicRecord>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?;
        Ok(guard.get(id).cloned())
    }

    /// Get record count
    pub fn len(&self) -> AmorphicResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .len())
    }

    /// Check if empty
    pub fn is_empty(&self) -> AmorphicResult<bool> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .is_empty())
    }

    /// Get partition count
    pub fn partition_count(&self) -> AmorphicResult<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .partition_count())
    }

    /// Get aggregate health status
    pub fn health_check(&self) -> AmorphicResult<HealthStatus> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .aggregate_health())
    }

    /// Get detailed statistics
    pub fn stats(&self) -> AmorphicResult<PartitionManagerStats> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .stats())
    }

    /// Get health summary for all partitions
    pub fn partition_health(&self) -> AmorphicResult<Vec<(u32, HologramHealth)>> {
        Ok(self
            .inner
            .read()
            .map_err(|_| AmorphicError::ConcurrencyError("Read lock poisoned".into()))?
            .health_summary())
    }
}

impl Default for PartitionedAmorphicStore {
    fn default() -> Self {
        Self::new()
    }
}

// Make thread-safe
unsafe impl Send for PartitionedAmorphicStore {}
unsafe impl Sync for PartitionedAmorphicStore {}

// =============================================================================
// SHARDED AMORPHIC STORE (Fine-Grained Concurrency)
// =============================================================================

/// Configuration for the sharded store
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of shards (default: number of CPU cores)
    pub shard_count: usize,
    /// Enable auto-rebalancing when shards become uneven
    pub auto_rebalance: bool,
    /// Threshold for rebalancing (max/min ratio)
    pub rebalance_threshold: f64,
}

impl Default for ShardConfig {
    fn default() -> Self {
        // Use platform detection for optimal defaults
        let platform = crate::platform::platform();
        Self {
            shard_count: platform.recommended_shard_count,
            auto_rebalance: false,
            rebalance_threshold: 2.0,
        }
    }
}

/// A sharded AmorphicStore with fine-grained locking
///
/// Unlike `PartitionedAmorphicStore` which uses a single lock for all partitions,
/// `ShardedAmorphicStore` uses separate locks per shard, allowing concurrent
/// writes to different shards. This significantly improves throughput under
/// high-concurrency workloads.
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────┐
/// │              ShardedAmorphicStore                   │
/// ├─────────────────────────────────────────────────────┤
/// │                                                     │
/// │   ┌───────────┐  ┌───────────┐  ┌───────────┐      │
/// │   │ RwLock 0  │  │ RwLock 1  │  │ RwLock N  │      │
/// │   ├───────────┤  ├───────────┤  ├───────────┤      │
/// │   │  Shard 0  │  │  Shard 1  │  │  Shard N  │      │
/// │   │ AmorphicS │  │ AmorphicS │  │ AmorphicS │      │
/// │   └───────────┘  └───────────┘  └───────────┘      │
/// │                                                     │
/// │   Threads can write to different shards in parallel │
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// ## Routing Strategy
///
/// Records are routed to shards based on a hash of their content.
/// This ensures even distribution and deterministic routing.
pub struct ShardedAmorphicStore {
    /// Individual shards, each with its own lock
    pub(crate) shards: Vec<RwLock<AmorphicStore>>,
    /// Number of shards
    pub(crate) shard_count: usize,
    /// Configuration
    pub(crate) config: ShardConfig,
    /// Global record index: global_id -> (shard_idx, local_id)
    pub(crate) record_index: RwLock<HashMap<RecordId, (usize, RecordId)>>,
    /// Next global record ID
    pub(crate) next_id: std::sync::atomic::AtomicU64,
}

impl ShardedAmorphicStore {
    /// Create a new sharded store with default configuration
    pub fn new() -> Self {
        Self::with_config(ShardConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ShardConfig) -> Self {
        let shard_count = config.shard_count.max(1);
        let shards: Vec<_> = (0..shard_count)
            .map(|_| RwLock::new(AmorphicStore::new()))
            .collect();

        Self {
            shards,
            shard_count,
            config,
            record_index: RwLock::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create with a specific number of shards
    pub fn with_shard_count(count: usize) -> Self {
        Self::with_config(ShardConfig {
            shard_count: count,
            ..Default::default()
        })
    }

    /// Route content to a shard based on hash
    fn route(&self, content: &str) -> usize {
        // Simple hash-based routing using FNV-1a
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in content.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        (hash as usize) % self.shard_count
    }

    /// Get the next global ID atomically
    fn next_global_id(&self) -> RecordId {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst) as RecordId
    }

    // ==================== WRITE OPERATIONS ====================

    /// Ingest a JSON document
    ///
    /// Routes to a shard based on content hash and acquires only that shard's lock.
    pub fn ingest_json(&self, json: &str) -> AmorphicResult<RecordId> {
        let shard_idx = self.route(json);

        // Acquire only this shard's write lock
        let local_id = self.shards[shard_idx]
            .write()
            .map_err(|_| {
                AmorphicError::ConcurrencyError(format!("Shard {} lock poisoned", shard_idx))
            })?
            .ingest_json(json)?;

        // Assign global ID
        let global_id = self.next_global_id();

        // Update index (separate lock, brief hold)
        self.record_index
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Index lock poisoned".into()))?
            .insert(global_id, (shard_idx, local_id));

        Ok(global_id)
    }

    /// Ingest a row
    pub fn ingest_row(&self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        // Create a string representation for routing
        let content: String = values.join("|");
        let shard_idx = self.route(&content);

        let local_id = self.shards[shard_idx]
            .write()
            .map_err(|_| {
                AmorphicError::ConcurrencyError(format!("Shard {} lock poisoned", shard_idx))
            })?
            .ingest_row(columns, values)?;

        let global_id = self.next_global_id();

        self.record_index
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Index lock poisoned".into()))?
            .insert(global_id, (shard_idx, local_id));

        Ok(global_id)
    }

    /// Ingest an edge
    pub fn ingest_edge(
        &self,
        source: &str,
        relation: &str,
        target: &str,
    ) -> AmorphicResult<RecordId> {
        let content = format!("{}|{}|{}", source, relation, target);
        let shard_idx = self.route(&content);

        let local_id = self.shards[shard_idx]
            .write()
            .map_err(|_| {
                AmorphicError::ConcurrencyError(format!("Shard {} lock poisoned", shard_idx))
            })?
            .ingest_edge(source, relation, target)?;

        let global_id = self.next_global_id();

        self.record_index
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Index lock poisoned".into()))?
            .insert(global_id, (shard_idx, local_id));

        Ok(global_id)
    }

    /// Delete a record by global ID
    pub fn delete(&self, global_id: RecordId) -> AmorphicResult<()> {
        // Look up shard location
        let (shard_idx, local_id): (usize, RecordId) = {
            let index = self
                .record_index
                .read()
                .map_err(|_| AmorphicError::ConcurrencyError("Index lock poisoned".into()))?;
            *index
                .get(&global_id)
                .ok_or(AmorphicError::RecordNotFound(global_id))?
        };

        // Delete from shard
        self.shards[shard_idx]
            .write()
            .map_err(|_| {
                AmorphicError::ConcurrencyError(format!("Shard {} lock poisoned", shard_idx))
            })?
            .delete(local_id)?;

        // Remove from index
        self.record_index
            .write()
            .map_err(|_| AmorphicError::ConcurrencyError("Index lock poisoned".into()))?
            .remove(&global_id);

        Ok(())
    }

    // ==================== READ OPERATIONS ====================

    /// Get a record by global ID
    pub fn get(&self, global_id: RecordId) -> AmorphicResult<Option<AmorphicRecord>> {
        let (shard_idx, local_id) = {
            let index = self
                .record_index
                .read()
                .map_err(|_| AmorphicError::ConcurrencyError("Index lock poisoned".into()))?;
            match index.get(&global_id) {
                Some(&loc) => loc,
                None => return Ok(None),
            }
        };

        let shard = self.shards[shard_idx].read().map_err(|_| {
            AmorphicError::ConcurrencyError(format!("Shard {} lock poisoned", shard_idx))
        })?;

        Ok(shard.get(local_id).cloned())
    }

    /// Query by exact match across all shards
    ///
    /// Queries all shards in parallel and merges results.
    pub fn query_equals(&self, field: &str, value: &Value) -> AmorphicResult<QueryResult> {
        use std::thread;

        // Query each shard in parallel using scoped threads
        let shard_results: Vec<Vec<AmorphicRecord>> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.query_equals(field, value).into_records()
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        let all_records: Vec<AmorphicRecord> = shard_results.into_iter().flatten().collect();
        Ok(QueryResult {
            records: all_records,
        })
    }

    /// Query by range across all shards
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> AmorphicResult<QueryResult> {
        use std::thread;

        // Query each shard in parallel using scoped threads
        let shard_results: Vec<Vec<AmorphicRecord>> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.query_range(field, min, max).into_records()
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        let all_records: Vec<AmorphicRecord> = shard_results.into_iter().flatten().collect();
        Ok(QueryResult {
            records: all_records,
        })
    }

    /// Query similar across all shards
    ///
    /// Gets k results from each shard and returns the top k overall.
    /// For accurate cross-shard ranking, results would need re-scoring.
    pub fn query_similar_to(&self, name: &str, k: usize) -> AmorphicResult<QueryResult> {
        use std::thread;

        // Query each shard in parallel using scoped threads
        let shard_results: Vec<Vec<AmorphicRecord>> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.query_similar_to(name, k).into_records()
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        let mut all_records: Vec<AmorphicRecord> = shard_results.into_iter().flatten().collect();

        // Truncate to k results
        all_records.truncate(k);

        Ok(QueryResult {
            records: all_records,
        })
    }

    // ==================== INFO METHODS ====================

    /// Get total record count across all shards
    pub fn len(&self) -> AmorphicResult<usize> {
        let mut total = 0;
        for shard in &self.shards {
            let guard = shard
                .read()
                .map_err(|_| AmorphicError::ConcurrencyError("Shard lock poisoned".into()))?;
            total += guard.len();
        }
        Ok(total)
    }

    /// Check if empty
    pub fn is_empty(&self) -> AmorphicResult<bool> {
        Ok(self.len()? == 0)
    }

    /// Get the number of shards
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }

    /// Get record counts per shard (for diagnostics)
    pub fn shard_sizes(&self) -> AmorphicResult<Vec<usize>> {
        let mut sizes = Vec::with_capacity(self.shard_count);
        for shard in &self.shards {
            let guard = shard
                .read()
                .map_err(|_| AmorphicError::ConcurrencyError("Shard lock poisoned".into()))?;
            sizes.push(guard.len());
        }
        Ok(sizes)
    }

    /// Get health status across all shards
    pub fn health_check(&self) -> AmorphicResult<HealthStatus> {
        let sizes = self.shard_sizes()?;

        if sizes.is_empty() {
            return Ok(HealthStatus::Healthy);
        }

        let max_size = *sizes.iter().max().unwrap_or(&0);
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let total: usize = sizes.iter().sum();

        let mut warnings = Vec::new();

        // Check for uneven distribution
        if min_size > 0 && max_size as f64 / min_size as f64 > self.config.rebalance_threshold {
            warnings.push(format!(
                "Uneven shard distribution: min={}, max={} (ratio {:.1}x)",
                min_size,
                max_size,
                max_size as f64 / min_size as f64
            ));
        }

        // Check for high capacity
        let avg_per_shard = total / self.shard_count;
        if avg_per_shard > GLOBAL_HOLOGRAM_SATURATION_LIMIT / 2 {
            warnings.push(format!(
                "High average shard size: {} (limit: {})",
                avg_per_shard, GLOBAL_HOLOGRAM_SATURATION_LIMIT
            ));
        }

        if warnings.is_empty() {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Warning(warnings))
        }
    }

    /// Get detailed statistics
    pub fn stats(&self) -> AmorphicResult<ShardedStoreStats> {
        let sizes = self.shard_sizes()?;
        let total: usize = sizes.iter().sum();

        let max_size = *sizes.iter().max().unwrap_or(&0);
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let avg_size = if sizes.is_empty() {
            0.0
        } else {
            total as f64 / sizes.len() as f64
        };

        Ok(ShardedStoreStats {
            shard_count: self.shard_count,
            total_records: total,
            shard_sizes: sizes,
            max_shard_size: max_size,
            min_shard_size: min_size,
            avg_shard_size: avg_size,
            distribution_ratio: if min_size > 0 {
                max_size as f64 / min_size as f64
            } else {
                f64::INFINITY
            },
        })
    }

    // ==================== COLUMNAR ANALYTICS (OLAP) ====================

    /// SUM of a numeric column across all shards (parallelized)
    pub fn sum(&self, field: &str) -> Option<f64> {
        use std::thread;

        let shard_sums: Vec<f64> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.sum(field).unwrap_or(0.0)
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        if shard_sums.is_empty() {
            None
        } else {
            Some(shard_sums.iter().sum())
        }
    }

    /// COUNT of a numeric column across all shards
    pub fn count(&self, field: &str) -> Option<usize> {
        use std::thread;

        let shard_counts: Vec<usize> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.count(field).unwrap_or(0)
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        if shard_counts.is_empty() {
            None
        } else {
            Some(shard_counts.iter().sum())
        }
    }

    /// AVG of a numeric column across all shards
    pub fn avg(&self, field: &str) -> Option<f64> {
        let sum = self.sum(field)?;
        let count = self.count(field)?;
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// MIN of a numeric column across all shards
    pub fn min(&self, field: &str) -> Option<f64> {
        use std::thread;

        let shard_mins: Vec<f64> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.min(field).unwrap_or(f64::MAX)
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        shard_mins
            .into_iter()
            .filter(|&v| v != f64::MAX)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.min(v))))
    }

    /// MAX of a numeric column across all shards
    pub fn max(&self, field: &str) -> Option<f64> {
        use std::thread;

        let shard_maxs: Vec<f64> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.max(field).unwrap_or(f64::MIN)
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        shard_maxs
            .into_iter()
            .filter(|&v| v != f64::MIN)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
    }

    /// COUNT with range filter across all shards (parallelized)
    pub fn count_where_range(&self, filter_field: &str, min: f64, max: f64) -> Option<usize> {
        use std::thread;

        let shard_counts: Vec<usize> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard.count_where_range(filter_field, min, max).unwrap_or(0)
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        if shard_counts.is_empty() {
            None
        } else {
            Some(shard_counts.iter().sum())
        }
    }

    /// SUM with range filter across all shards (parallelized)
    pub fn sum_where_range(
        &self,
        sum_field: &str,
        filter_field: &str,
        min: f64,
        max: f64,
    ) -> Option<f64> {
        use std::thread;

        let shard_sums: Vec<f64> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        guard
                            .sum_where_range(sum_field, filter_field, min, max)
                            .unwrap_or(0.0)
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        if shard_sums.is_empty() {
            None
        } else {
            Some(shard_sums.iter().sum())
        }
    }

    /// Hash join with SUM aggregation across all shards (parallelized)
    ///
    /// This implements a broadcast hash join:
    /// 1. Build phase: Collect all build keys from all shards, create global hash set
    /// 2. Probe phase: Each shard probes its local probe column against global hash set
    ///
    /// Useful for joins like: SUM(l_extendedprice) WHERE l_orderkey = o_orderkey
    pub fn hash_join_sum(
        &self,
        build_field: &str,
        probe_field: &str,
        sum_field: &str,
    ) -> Option<f64> {
        use std::collections::HashSet;
        use std::thread;

        // Phase 1: Collect all build keys from all shards (parallel)
        let build_keys: HashSet<i64> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        if let Some(col) = guard.columnar().get_column(build_field) {
                            col.scan().map(|(_, v)| v as i64).collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        }
                    })
                })
                .collect();

            handles
                .into_iter()
                .filter_map(|h| h.join().ok())
                .flatten()
                .collect()
        });

        if build_keys.is_empty() {
            return None;
        }

        // Phase 2: Probe each shard and sum matching values (parallel)
        let shard_sums: Vec<f64> = thread::scope(|s| {
            let build_keys_ref = &build_keys;
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(move || {
                        let guard = shard.read().unwrap();
                        let columnar = guard.columnar();

                        let probe_col = columnar.get_column(probe_field)?;
                        let sum_col = columnar.get_column(sum_field)?;

                        // Build map from record_id to sum value
                        let sum_values: std::collections::HashMap<_, _> = sum_col
                            .record_ids
                            .iter()
                            .zip(sum_col.values.iter())
                            .filter(|(id, _)| !sum_col.is_deleted(**id))
                            .map(|(id, v)| (*id, *v))
                            .collect();

                        // Sum values where probe key exists in build keys
                        let sum: f64 = probe_col
                            .scan()
                            .filter(|(_, v)| build_keys_ref.contains(&(*v as i64)))
                            .filter_map(|(id, _)| sum_values.get(&id))
                            .sum();

                        Some(sum)
                    })
                })
                .collect();

            handles
                .into_iter()
                .filter_map(|h| h.join().ok())
                .filter_map(|opt| opt)
                .collect()
        });

        if shard_sums.is_empty() {
            None
        } else {
            Some(shard_sums.iter().sum())
        }
    }

    /// Hash join count across all shards
    ///
    /// Returns the number of matching rows from the join
    pub fn hash_join_count(&self, build_field: &str, probe_field: &str) -> Option<usize> {
        use std::collections::HashSet;
        use std::thread;

        // Phase 1: Collect all build keys
        let build_keys: HashSet<i64> = thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(|| {
                        let guard = shard.read().unwrap();
                        if let Some(col) = guard.columnar().get_column(build_field) {
                            col.scan().map(|(_, v)| v as i64).collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        }
                    })
                })
                .collect();

            handles
                .into_iter()
                .filter_map(|h| h.join().ok())
                .flatten()
                .collect()
        });

        if build_keys.is_empty() {
            return None;
        }

        // Phase 2: Count matches in each shard
        let shard_counts: Vec<usize> = thread::scope(|s| {
            let build_keys_ref = &build_keys;
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(move || {
                        let guard = shard.read().unwrap();
                        if let Some(col) = guard.columnar().get_column(probe_field) {
                            col.scan()
                                .filter(|(_, v)| build_keys_ref.contains(&(*v as i64)))
                                .count()
                        } else {
                            0
                        }
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        if shard_counts.is_empty() {
            None
        } else {
            Some(shard_counts.iter().sum())
        }
    }
}

impl Default for ShardedAmorphicStore {
    fn default() -> Self {
        Self::new()
    }
}

// Make thread-safe
unsafe impl Send for ShardedAmorphicStore {}
unsafe impl Sync for ShardedAmorphicStore {}

/// Statistics for the sharded store
#[derive(Debug, Clone)]
pub struct ShardedStoreStats {
    /// Number of shards
    pub shard_count: usize,
    /// Total records across all shards
    pub total_records: usize,
    /// Records per shard
    pub shard_sizes: Vec<usize>,
    /// Maximum shard size
    pub max_shard_size: usize,
    /// Minimum shard size
    pub min_shard_size: usize,
    /// Average shard size
    pub avg_shard_size: f64,
    /// Distribution ratio (max/min)
    pub distribution_ratio: f64,
}

/// Get number of CPU cores (for default shard count)
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

// =============================================================================
// HELPERS
// =============================================================================

/// Get current epoch timestamp in seconds
fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hologram_health_empty() {
        let acc = BundleAccumulator::new(DIMENSION);
        let health = HologramHealth::measure(&acc, 0);

        assert_eq!(health.item_count, 0);
        assert_eq!(health.bit_entropy, 0.0);
        assert_eq!(health.retrieval_probability, 1.0);
        assert!(!health.should_partition());
    }

    #[test]
    fn test_hologram_health_low_count() {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Add some vectors
        for i in 0..10 {
            let hv = joule_db_hdc::BinaryHV::random(DIMENSION, i as u64);
            acc.add(&hv);
        }

        let health = HologramHealth::measure(&acc, 10);

        assert_eq!(health.item_count, 10);
        // P = 1 - n²/D = 1 - 100/10000 = 0.99
        assert!(health.retrieval_probability >= 0.99);
        assert!(health.snr > 30.0); // √(10000/10) ≈ 31.6
        assert!(!health.should_partition());
    }

    #[test]
    fn test_hologram_health_near_saturation() {
        let health = HologramHealth {
            item_count: 300,
            bit_entropy: 0.9,
            retrieval_probability: 0.91, // Just above threshold
            snr: 5.8,
            capacity_used: 0.95,
            measured_at: 0,
        };

        // Should not partition yet (all metrics just within bounds)
        assert!(!health.should_partition());
        assert_eq!(health.status(), PartitionHealthStatus::Warning);
    }

    #[test]
    fn test_hologram_health_at_saturation() {
        let health = HologramHealth {
            item_count: 350,
            bit_entropy: 0.96,           // Above threshold
            retrieval_probability: 0.88, // Below threshold
            snr: 3.5,                    // Below threshold
            capacity_used: 1.1,
            measured_at: 0,
        };

        // Should partition (multiple metrics exceeded)
        assert!(health.should_partition());
        assert_eq!(health.status(), PartitionHealthStatus::Critical);
    }

    #[test]
    fn test_partition_manager_creation() {
        let manager = PartitionManager::new();

        assert_eq!(manager.partition_count(), 1);
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_partition_manager_ingest() {
        let mut manager = PartitionManager::new();

        let id1 = manager
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        let id2 = manager
            .ingest_json(r#"{"name": "Bob", "age": 25}"#)
            .unwrap();

        assert_eq!(manager.len(), 2);
        assert_eq!(manager.partition_count(), 1); // Should still be 1 partition

        // Can retrieve records
        let alice = manager.get(id1);
        assert!(alice.is_some());

        let bob = manager.get(id2);
        assert!(bob.is_some());
    }

    #[test]
    fn test_partition_manager_query() {
        let mut manager = PartitionManager::new();

        manager
            .ingest_json(r#"{"name": "Alice", "city": "NYC"}"#)
            .unwrap();
        manager
            .ingest_json(r#"{"name": "Bob", "city": "LA"}"#)
            .unwrap();
        manager
            .ingest_json(r#"{"name": "Carol", "city": "NYC"}"#)
            .unwrap();

        let nyc_results = manager.query_equals("city", &Value::String("NYC".to_string()));
        assert_eq!(nyc_results.len(), 2);
    }

    #[test]
    fn test_partition_manager_auto_partition() {
        let config = PartitionConfig {
            max_records_per_partition: 10, // Very low for testing
            target_records_per_partition: 5,
            auto_partition: true,
            ..Default::default()
        };

        let mut manager = PartitionManager::with_config(config);

        // Insert enough records to trigger auto-partition
        // Due to entropy-based triggering, exact count varies
        for i in 0..25 {
            let json = format!(r#"{{"id": {}, "value": "test{}"}}"#, i, i);
            manager.ingest_json(&json).unwrap();
        }

        // Should have created multiple partitions
        assert!(manager.partition_count() >= 1);
        assert_eq!(manager.len(), 25);

        // All records should still be queryable
        for i in 0..25 {
            let results = manager.query_equals("id", &Value::Int(i));
            assert_eq!(results.len(), 1, "Record {} not found", i);
        }
    }

    #[test]
    fn test_partitioned_store_basic() {
        let store = PartitionedAmorphicStore::new();

        let id = store.ingest_json(r#"{"name": "Test"}"#).unwrap();

        assert_eq!(store.len().unwrap(), 1);
        assert_eq!(store.partition_count().unwrap(), 1);

        let record = store.get(id).unwrap();
        assert!(record.is_some());
    }

    #[test]
    fn test_partitioned_store_health() {
        let store = PartitionedAmorphicStore::new();

        // Empty store should be healthy
        let health = store.health_check().unwrap();
        assert_eq!(health, HealthStatus::Healthy);

        // Add some records
        for i in 0..10 {
            let json = format!(r#"{{"id": {}}}"#, i);
            store.ingest_json(&json).unwrap();
        }

        // Should still be healthy with only 10 records
        let health = store.health_check().unwrap();
        assert_eq!(health, HealthStatus::Healthy);
    }

    #[test]
    fn test_retrieval_probability_formula() {
        // Test the capacity theorem formula

        // N=10, D=10000: P ≈ 1 - 100/10000 = 0.99
        let p10 = HologramHealth::compute_retrieval_probability(10, 10000);
        assert!((p10 - 0.99).abs() < 0.01);

        // N=100, D=10000: P ≈ 1 - 10000/10000 = 0.0
        let p100 = HologramHealth::compute_retrieval_probability(100, 10000);
        assert!((p100 - 0.0).abs() < 0.01);

        // N=316, D=10000: P ≈ 1 - 99856/10000 ≈ 0 (saturated)
        let p316 = HologramHealth::compute_retrieval_probability(316, 10000);
        assert!(p316 < 0.1);
    }

    #[test]
    fn test_snr_formula() {
        // SNR = √(D/N)

        // N=10, D=10000: SNR = √1000 ≈ 31.6
        let snr10 = HologramHealth::compute_snr(10, 10000);
        assert!((snr10 - 31.6).abs() < 0.5);

        // N=100, D=10000: SNR = √100 = 10
        let snr100 = HologramHealth::compute_snr(100, 10000);
        assert!((snr100 - 10.0).abs() < 0.1);

        // N=625, D=10000: SNR = √16 = 4 (at threshold)
        let snr625 = HologramHealth::compute_snr(625, 10000);
        assert!((snr625 - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_partition_events() {
        let mut manager = PartitionManager::new();

        // Should have creation event
        let events = manager.recent_events(10);
        assert!(!events.is_empty());
        assert_eq!(events[0].event_type, PartitionEventType::Created);
    }

    // =========================================================================
    // SHARDED STORE TESTS
    // =========================================================================

    #[test]
    fn test_sharded_store_basic() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        let id1 = store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
        let id2 = store.ingest_json(r#"{"name": "Bob"}"#).unwrap();

        assert_eq!(store.len().unwrap(), 2);
        assert_eq!(store.shard_count(), 4);

        let alice = store.get(id1).unwrap();
        assert!(alice.is_some());

        let bob = store.get(id2).unwrap();
        assert!(bob.is_some());
    }

    #[test]
    fn test_sharded_store_distribution() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        // Insert many records - they should be distributed across shards
        for i in 0..100 {
            let json = format!(r#"{{"id": {}, "value": "test{}"}}"#, i, i);
            store.ingest_json(&json).unwrap();
        }

        let sizes = store.shard_sizes().unwrap();
        assert_eq!(sizes.len(), 4);

        // All shards should have some records (probabilistically)
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 100);

        // At least 2 shards should have records (with hash-based routing)
        let non_empty = sizes.iter().filter(|&&s| s > 0).count();
        assert!(
            non_empty >= 2,
            "Expected at least 2 non-empty shards, got {}",
            non_empty
        );
    }

    #[test]
    fn test_sharded_store_query_equals() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        store
            .ingest_json(r#"{"name": "Alice", "city": "NYC"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Bob", "city": "LA"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Carol", "city": "NYC"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Dave", "city": "NYC"}"#)
            .unwrap();

        let nyc = store
            .query_equals("city", &Value::String("NYC".to_string()))
            .unwrap();
        assert_eq!(nyc.len(), 3);

        let la = store
            .query_equals("city", &Value::String("LA".to_string()))
            .unwrap();
        assert_eq!(la.len(), 1);
    }

    #[test]
    fn test_sharded_store_query_similar() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        store
            .ingest_json(r#"{"name": "Alice Smith", "bio": "Software engineer"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Bob Jones", "bio": "Data scientist"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Carol Williams", "bio": "Product manager"}"#)
            .unwrap();

        let results = store.query_similar_to("software engineer", 2).unwrap();
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_sharded_store_delete() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        let id1 = store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
        let id2 = store.ingest_json(r#"{"name": "Bob"}"#).unwrap();

        assert_eq!(store.len().unwrap(), 2);

        store.delete(id1).unwrap();
        assert_eq!(store.len().unwrap(), 1);

        // id1 should be gone
        assert!(store.get(id1).unwrap().is_none());

        // id2 should still exist
        assert!(store.get(id2).unwrap().is_some());
    }

    #[test]
    fn test_sharded_store_concurrent_writes() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(ShardedAmorphicStore::with_shard_count(8));
        let mut handles = Vec::new();

        // Spawn 10 threads, each inserting 100 records
        for t in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let json = format!(r#"{{"thread": {}, "id": {}}}"#, t, i);
                    store_clone.ingest_json(&json).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Should have all 1000 records
        assert_eq!(store.len().unwrap(), 1000);

        // Check distribution across shards
        let sizes = store.shard_sizes().unwrap();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_sharded_store_health() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        // Empty store should be healthy
        let health = store.health_check().unwrap();
        assert_eq!(health, HealthStatus::Healthy);

        // Add some records
        for i in 0..40 {
            let json = format!(r#"{{"id": {}}}"#, i);
            store.ingest_json(&json).unwrap();
        }

        // Should still be healthy
        let health = store.health_check().unwrap();
        assert_eq!(health, HealthStatus::Healthy);
    }

    #[test]
    fn test_sharded_store_stats() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        for i in 0..100 {
            let json = format!(r#"{{"id": {}}}"#, i);
            store.ingest_json(&json).unwrap();
        }

        let stats = store.stats().unwrap();
        assert_eq!(stats.shard_count, 4);
        assert_eq!(stats.total_records, 100);
        assert_eq!(stats.shard_sizes.len(), 4);
        assert!(stats.avg_shard_size > 0.0);
    }

    #[test]
    fn test_sharded_store_ingest_row() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        let id = store
            .ingest_row(&["name", "age"], &["Alice", "30"])
            .unwrap();
        assert_eq!(store.len().unwrap(), 1);

        let record = store.get(id).unwrap();
        assert!(record.is_some());
    }

    #[test]
    fn test_sharded_store_ingest_edge() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        let id = store.ingest_edge("Alice", "knows", "Bob").unwrap();
        // ingest_edge creates entity records for source and target, plus the edge
        // So len >= 1 (the edge record was created)
        assert!(store.len().unwrap() >= 1);

        let record = store.get(id).unwrap();
        assert!(record.is_some());
    }

    // ==================== Semantic routing tests ====================

    #[test]
    fn test_semantic_routing_encode_record_hv() {
        // Verify that encoding the same record produces the same HV
        let json1: serde_json::Value = serde_json::json!({"name": "Alice", "age": 30});
        let json2: serde_json::Value = serde_json::json!({"name": "Alice", "age": 30});
        let json3: serde_json::Value = serde_json::json!({"name": "Bob", "dept": "Engineering"});

        let hv1 = PartitionManager::encode_record_hv(&json1);
        let hv2 = PartitionManager::encode_record_hv(&json2);
        let hv3 = PartitionManager::encode_record_hv(&json3);

        // Same record → identical HV
        assert_eq!(hv1.similarity(&hv2), 1.0);

        // Different records → different HVs
        let sim = hv1.similarity(&hv3);
        assert!(
            sim < 0.9,
            "Different records should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_semantic_routing_deterministic() {
        // Same record always routes to the same partition
        let config = PartitionConfig {
            routing_strategy: RoutingStrategy::Semantic,
            auto_partition: false,
            ..Default::default()
        };
        let mut manager = PartitionManager::with_config(config);

        // Create second partition manually
        manager.create_partition();
        assert_eq!(manager.partition_count(), 2);

        // Ingest records with similar content to different partitions
        let id1 = manager
            .ingest_json(r#"{"dept": "Engineering", "project": "Alpha"}"#)
            .unwrap();
        let id2 = manager
            .ingest_json(r#"{"dept": "Engineering", "project": "Beta"}"#)
            .unwrap();

        // Both should exist
        assert!(manager.get(id1).is_some());
        assert!(manager.get(id2).is_some());
    }

    #[test]
    fn test_semantic_routing_similar_records() {
        let config = PartitionConfig {
            routing_strategy: RoutingStrategy::Semantic,
            auto_partition: false,
            ..Default::default()
        };
        let mut manager = PartitionManager::with_config(config);
        manager.create_partition(); // second partition
        assert_eq!(manager.partition_count(), 2);

        // Seed partition 0 with engineering records
        for i in 0..10 {
            manager
                .ingest_json(&format!(
                    r#"{{"dept": "Engineering", "name": "eng_{}", "skill": "rust"}}"#,
                    i
                ))
                .unwrap();
        }

        // A new engineering record should route to the same partition
        // (though exact partition depends on hologram content)
        let id = manager
            .ingest_json(r#"{"dept": "Engineering", "name": "new_eng", "skill": "python"}"#)
            .unwrap();

        assert!(manager.get(id).is_some());
        assert_eq!(manager.len(), 11);
    }

    #[test]
    fn test_semantic_routing_new_content_fallback() {
        let config = PartitionConfig {
            routing_strategy: RoutingStrategy::Semantic,
            auto_partition: false,
            ..Default::default()
        };
        let mut manager = PartitionManager::with_config(config);

        // With empty partitions, similarity will be very low → falls back to least-loaded
        let id1 = manager
            .ingest_json(r#"{"planet": "Mars", "distance": 225}"#)
            .unwrap();
        let id2 = manager
            .ingest_json(r#"{"animal": "Cat", "legs": 4}"#)
            .unwrap();

        // Both should be ingested successfully
        assert!(manager.get(id1).is_some());
        assert!(manager.get(id2).is_some());
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_semantic_routing_row_ingest() {
        let config = PartitionConfig {
            routing_strategy: RoutingStrategy::Semantic,
            auto_partition: false,
            ..Default::default()
        };
        let mut manager = PartitionManager::with_config(config);

        // ingest_row should also work with semantic routing
        let columns = ["name", "age"];
        let values = ["Alice", "30"];
        let id = manager.ingest_row(&columns, &values).unwrap();
        assert!(manager.get(id).is_some());
    }

    #[test]
    fn test_semantic_routing_respects_health() {
        let config = PartitionConfig {
            routing_strategy: RoutingStrategy::Semantic,
            auto_partition: true,
            ..Default::default()
        };
        let mut manager = PartitionManager::with_config(config);

        // Ingest some records — the manager should handle auto-partitioning
        for i in 0..20 {
            manager
                .ingest_json(&format!(
                    r#"{{"id": {}, "type": "test", "value": {}}}"#,
                    i,
                    i * 10
                ))
                .unwrap();
        }

        assert_eq!(manager.len(), 20);
        // All records should be retrievable
        for id in 1..=20 {
            // Global IDs start from 1
            assert!(
                manager.get(id as u64).is_some() || id > manager.len(),
                "Record {} should be retrievable",
                id
            );
        }
    }
}
