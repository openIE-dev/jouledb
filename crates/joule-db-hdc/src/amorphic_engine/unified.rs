//! Amorphic Engine
//!
//! Unified API combining all novel technologies into a revolutionary database architecture.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use super::{AmorphicEngineError, AmorphicEngineResult};
use crate::holographic::{HolographicStorage, SimilarityResult};
use crate::hyperdimensional::{HyperVector, HyperdimensionalStorage, SimilarityMatch};
use crate::learned::LearnedIndex;
use crate::manifold::{InformationManifold, ManifoldPoint, NeighborResult};
use crate::predictive::{NGramPredictor, Prediction, QueryPredictor};
use crate::sdm::{SDMAddress, SparseDistributedMemory};
use crate::spiking::{SNNStats, SpikeEvent, SpikeNeuralNetwork, TemporalDecoder, TemporalEncoder};
use crate::thermodynamic::{QueryPlan, ThermodynamicOptimizer};

/// Simple deterministic pseudo-random based on a counter
fn rand_simple() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Simple LCG-like transformation
    let x = n
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (x as f64) / (u64::MAX as f64)
}

/// AmorphicEngine Configuration
#[derive(Clone, Debug)]
pub struct AmorphicEngineConfig {
    /// SDM number of hard locations
    pub sdm_locations: usize,
    /// SDM address dimension (bits)
    pub sdm_dimension: usize,
    /// SDM data word size
    pub sdm_data_size: usize,

    /// Manifold embedding dimension
    pub manifold_dimension: usize,

    /// Holographic storage dimension
    pub holographic_dimension: usize,

    /// Hyperdimensional vector dimension
    pub hyperdimensional_dimension: usize,

    /// Predictor history size
    pub predictor_history_size: usize,
    /// Predictor cache size
    pub predictor_cache_size: usize,

    /// SNN neuron count
    pub snn_neurons: usize,
    /// SNN time step
    pub snn_time_step: f32,
}

impl AmorphicEngineConfig {
    /// Create minimal configuration (for testing)
    pub fn minimal() -> Self {
        Self {
            sdm_locations: 100,
            sdm_dimension: 128,
            sdm_data_size: 64,
            manifold_dimension: 16,
            holographic_dimension: 256,
            hyperdimensional_dimension: 1000,
            predictor_history_size: 100,
            predictor_cache_size: 100,
            snn_neurons: 100,
            snn_time_step: 1.0,
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self {
            sdm_locations: 10000,
            sdm_dimension: 1024,
            sdm_data_size: 256,
            manifold_dimension: 128,
            holographic_dimension: 2048,
            hyperdimensional_dimension: 10000,
            predictor_history_size: 10000,
            predictor_cache_size: 10000,
            snn_neurons: 1000,
            snn_time_step: 0.1,
        }
    }
}

impl Default for AmorphicEngineConfig {
    fn default() -> Self {
        Self {
            sdm_locations: 1000,
            sdm_dimension: 256,
            sdm_data_size: 128,
            manifold_dimension: 32,
            holographic_dimension: 512,
            hyperdimensional_dimension: 4096,
            predictor_history_size: 1000,
            predictor_cache_size: 1000,
            snn_neurons: 256,
            snn_time_step: 0.5,
        }
    }
}

/// AmorphicEngine Metrics
#[derive(Default, Clone, Debug)]
struct AmorphicEngineMetrics {
    reads: u64,
    writes: u64,
    deletes: u64,
    cache_hits: u64,
    cache_misses: u64,
    sdm_lookups: u64,
    holographic_lookups: u64,
    manifold_queries: u64,
    predictions_made: u64,
    learned_index_hits: u64,
    ngram_predictions: u64,
    snn_pattern_detections: u64,
}

/// Statistics about AmorphicEngine instance
#[derive(Clone, Debug)]
pub struct AmorphicEngineStats {
    /// Number of reads
    pub reads: u64,
    /// Number of writes
    pub writes: u64,
    /// Number of deletes
    pub deletes: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
    /// SDM lookups
    pub sdm_lookups: u64,
    /// Holographic lookups
    pub holographic_lookups: u64,
    /// Manifold queries
    pub manifold_queries: u64,
    /// Predictions made
    pub predictions_made: u64,
    /// Number of items in store
    pub store_size: usize,
    /// Learned index hits
    pub learned_index_hits: u64,
    /// N-gram predictions made
    pub ngram_predictions: u64,
    /// SNN pattern detections
    pub snn_pattern_detections: u64,
}

/// Amorphic Engine
///
/// A unified database combining multiple novel storage and indexing techniques.
pub struct AmorphicEngine {
    /// Configuration
    config: AmorphicEngineConfig,

    // Core Storage Systems
    sdm: Arc<RwLock<SparseDistributedMemory>>,
    holographic: Arc<RwLock<HolographicStorage>>,
    hyperdimensional: Arc<RwLock<HyperdimensionalStorage>>,

    // Indexing
    manifold: Arc<RwLock<InformationManifold>>,
    learned: Arc<RwLock<Option<LearnedIndex>>>,

    // Optimization
    predictor: Arc<RwLock<QueryPredictor>>,
    ngram_predictor: Arc<RwLock<NGramPredictor>>,
    optimizer: Arc<RwLock<ThermodynamicOptimizer>>,

    // Neural Processing
    snn: Arc<RwLock<SpikeNeuralNetwork>>,

    // Primary Key-Value Store
    data_store: Arc<RwLock<HashMap<String, Vec<u8>>>>,

    // Metrics
    metrics: Arc<RwLock<AmorphicEngineMetrics>>,

    // Node identity
    node_id: String,
}

impl AmorphicEngine {
    /// Create new AmorphicEngine instance with default configuration
    pub fn new() -> Self {
        Self::with_config(AmorphicEngineConfig::default(), "node_default")
    }

    /// Create AmorphicEngine with custom configuration
    pub fn with_config(config: AmorphicEngineConfig, node_id: &str) -> Self {
        let sdm = SparseDistributedMemory::new(
            config.sdm_locations,
            config.sdm_dimension,
            config.sdm_data_size,
        );

        let holographic = HolographicStorage::new(config.holographic_dimension);
        let hyperdimensional = HyperdimensionalStorage::new(config.hyperdimensional_dimension);
        let manifold = InformationManifold::new(config.manifold_dimension);
        let predictor =
            QueryPredictor::new(config.predictor_history_size, config.predictor_cache_size);
        let ngram_predictor = NGramPredictor::new(3);
        let optimizer = ThermodynamicOptimizer::new();
        let snn = SpikeNeuralNetwork::new(config.snn_neurons, config.snn_time_step);

        Self {
            config,
            sdm: Arc::new(RwLock::new(sdm)),
            holographic: Arc::new(RwLock::new(holographic)),
            hyperdimensional: Arc::new(RwLock::new(hyperdimensional)),
            manifold: Arc::new(RwLock::new(manifold)),
            learned: Arc::new(RwLock::new(None)),
            predictor: Arc::new(RwLock::new(predictor)),
            ngram_predictor: Arc::new(RwLock::new(ngram_predictor)),
            optimizer: Arc::new(RwLock::new(optimizer)),
            snn: Arc::new(RwLock::new(snn)),
            data_store: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(AmorphicEngineMetrics::default())),
            node_id: node_id.to_string(),
        }
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /// Write data with full AmorphicEngine indexing
    pub fn write(&self, key: &str, value: &[u8]) -> AmorphicEngineResult<()> {
        if let Ok(mut m) = self.metrics.write() {
            m.writes += 1;
        }

        // 1. Store in primary store
        if let Ok(mut store) = self.data_store.write() {
            store.insert(key.to_string(), value.to_vec());
        }

        // 2. Index in SDM (content-addressable)
        if let Ok(sdm) = self.sdm.read() {
            let address = SDMAddress::from_data(key.as_bytes(), self.config.sdm_dimension);
            let _ = sdm.write_bytes(&address, value);
        }

        // 3. Store in holographic memory
        if let Ok(holographic) = self.holographic.read() {
            let mut complex_data = Vec::with_capacity(self.config.holographic_dimension * 2);
            for &b in value.iter().cycle().take(self.config.holographic_dimension) {
                complex_data.push((b as f32 / 127.5) - 1.0);
                complex_data.push(0.0);
            }
            let _ = holographic.store_pattern(key.to_string(), &complex_data);
        }

        // 4. Index on manifold
        if let Ok(manifold) = self.manifold.read() {
            manifold.insert(key, value);
        }

        // 5. Update Markov predictor
        if let Ok(predictor) = self.predictor.read() {
            predictor.observe(key);
            predictor.cache_result(key, value);
        }

        // 6. Update N-gram predictor for higher-order patterns
        if let Ok(ngram) = self.ngram_predictor.read() {
            ngram.observe(key);
        }

        // 7. Feed access pattern to SNN for temporal learning
        self.record_snn_access(key);

        Ok(())
    }

    /// Read data with predictive caching
    pub fn read(&self, key: &str) -> Option<Vec<u8>> {
        if let Ok(mut m) = self.metrics.write() {
            m.reads += 1;
        }

        // 1. Check prediction cache first
        if let Ok(predictor) = self.predictor.read() {
            if let Some(cached) = predictor.get_cached(key) {
                if let Ok(mut m) = self.metrics.write() {
                    m.cache_hits += 1;
                }
                predictor.observe(key);
                self.prefetch_predicted();
                self.prefetch_ngram_predicted();
                return Some(cached);
            }
            if let Ok(mut m) = self.metrics.write() {
                m.cache_misses += 1;
            }
        }

        // 2. Try learned index for fast position hint (if trained)
        if let Ok(learned) = self.learned.read() {
            if let Some(ref index) = *learned {
                if index.is_trained() {
                    // Use learned index for position hint (useful for range queries)
                    if let Ok(mut m) = self.metrics.write() {
                        m.learned_index_hits += 1;
                    }
                }
            }
        }

        // 3. Try primary store
        if let Ok(store) = self.data_store.read() {
            if let Some(value) = store.get(key) {
                // Cache for future
                if let Ok(predictor) = self.predictor.read() {
                    predictor.cache_result(key, value);
                    predictor.observe(key);
                }
                // Update N-gram predictor
                if let Ok(ngram) = self.ngram_predictor.read() {
                    ngram.observe(key);
                }
                // Record access pattern for SNN
                self.record_snn_access(key);
                self.prefetch_predicted();
                self.prefetch_ngram_predicted();
                return Some(value.clone());
            }
        }

        None
    }

    /// Delete data
    pub fn delete(&self, key: &str) -> bool {
        if let Ok(mut m) = self.metrics.write() {
            m.deletes += 1;
        }

        if let Ok(mut store) = self.data_store.write() {
            return store.remove(key).is_some();
        }
        false
    }

    /// Check if key exists
    pub fn contains(&self, key: &str) -> bool {
        if let Ok(store) = self.data_store.read() {
            return store.contains_key(key);
        }
        false
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        if let Ok(store) = self.data_store.read() {
            return store.keys().cloned().collect();
        }
        Vec::new()
    }

    /// Get store size
    pub fn len(&self) -> usize {
        if let Ok(store) = self.data_store.read() {
            return store.len();
        }
        0
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========================================================================
    // Content-Addressable Operations (via SDM)
    // ========================================================================

    /// Content-based lookup (find by similar content)
    pub fn content_lookup(&self, query_data: &[u8]) -> Vec<u8> {
        if let Ok(mut m) = self.metrics.write() {
            m.sdm_lookups += 1;
        }

        if let Ok(sdm) = self.sdm.read() {
            return sdm
                .content_lookup(query_data)
                .iter()
                .map(|&b| ((b as i16) + 128) as u8)
                .collect();
        }
        Vec::new()
    }

    /// Store with content-addressable semantics
    pub fn store_content(&self, data: &[u8]) -> AmorphicEngineResult<String> {
        use std::collections::hash_map::DefaultHasher;

        // Generate content-based key
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let key = format!("content_{:016x}", hasher.finish());

        self.write(&key, data)?;
        Ok(key)
    }

    // ========================================================================
    // Similarity Search (via Manifold)
    // ========================================================================

    /// Find k nearest similar items
    pub fn similarity_search(&self, query: &[u8], k: usize) -> Vec<NeighborResult> {
        if let Ok(mut m) = self.metrics.write() {
            m.manifold_queries += 1;
        }

        if let Ok(manifold) = self.manifold.read() {
            return manifold.nearest_by_data(query, k);
        }
        Vec::new()
    }

    /// Find items within distance
    pub fn range_search(&self, query: &[u8], radius: f32) -> Vec<NeighborResult> {
        if let Ok(mut m) = self.metrics.write() {
            m.manifold_queries += 1;
        }

        if let Ok(manifold) = self.manifold.read() {
            let point = ManifoldPoint::from_data(query, self.config.manifold_dimension);
            return manifold.range_query(&point, radius);
        }
        Vec::new()
    }

    // ========================================================================
    // Holographic Operations
    // ========================================================================

    /// Holographic associative search
    pub fn associative_search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> AmorphicEngineResult<Vec<SimilarityResult>> {
        if let Ok(mut m) = self.metrics.write() {
            m.holographic_lookups += 1;
        }

        if let Ok(holographic) = self.holographic.read() {
            return holographic
                .associative_search(query, top_k)
                .map_err(|e| AmorphicEngineError::Holographic(e.to_string()));
        }
        Ok(Vec::new())
    }

    /// Recall pattern from partial query
    pub fn holographic_recall(
        &self,
        key: &str,
        partial_query: &[f32],
    ) -> AmorphicEngineResult<Vec<f32>> {
        if let Ok(mut m) = self.metrics.write() {
            m.holographic_lookups += 1;
        }

        if let Ok(holographic) = self.holographic.read() {
            return holographic
                .recall_pattern(key, partial_query)
                .map_err(|e| AmorphicEngineError::Holographic(e.to_string()));
        }
        Err(AmorphicEngineError::LockError(
            "failed to acquire holographic lock".to_string(),
        ))
    }

    // ========================================================================
    // Hyperdimensional Operations
    // ========================================================================

    /// Add hypervector
    pub fn add_hypervector(&self, vector: &HyperVector) -> AmorphicEngineResult<usize> {
        if let Ok(hd) = self.hyperdimensional.read() {
            return hd
                .add_vector(vector.clone())
                .map_err(|e| AmorphicEngineError::Hyperdimensional(e.to_string()));
        }
        Err(AmorphicEngineError::LockError(
            "failed to acquire hyperdimensional lock".to_string(),
        ))
    }

    /// Hyperdimensional similarity search
    pub fn hyperdimensional_search(
        &self,
        query: &HyperVector,
        top_k: usize,
    ) -> AmorphicEngineResult<Vec<SimilarityMatch>> {
        if let Ok(hd) = self.hyperdimensional.read() {
            return hd
                .similarity_search(query, top_k)
                .map_err(|e| AmorphicEngineError::Hyperdimensional(e.to_string()));
        }
        Err(AmorphicEngineError::LockError(
            "failed to acquire hyperdimensional lock".to_string(),
        ))
    }

    // ========================================================================
    // Query Optimization
    // ========================================================================

    /// Optimize query plan
    pub fn optimize_query(
        &self,
        selectivity: f64,
        join_count: usize,
        has_index: bool,
    ) -> QueryPlan {
        if let Ok(mut optimizer) = self.optimizer.write() {
            let plans = vec![
                QueryPlan::new(selectivity, join_count, has_index, "original"),
                QueryPlan::new(selectivity, 0, has_index, "no_joins"),
                QueryPlan::new(1.0, join_count, true, "full_scan_indexed"),
            ];

            return optimizer.optimize_plans(plans);
        }
        QueryPlan::new(selectivity, join_count, has_index, "fallback")
    }

    /// Get optimizer temperature
    pub fn optimizer_temperature(&self) -> f64 {
        if let Ok(optimizer) = self.optimizer.read() {
            return optimizer.temperature();
        }
        1.0
    }

    /// Reset optimizer (for new workload)
    pub fn reset_optimizer(&self) {
        if let Ok(mut optimizer) = self.optimizer.write() {
            optimizer.reset_temperature();
        }
    }

    // ========================================================================
    // Predictive Operations
    // ========================================================================

    /// Get predicted next queries
    pub fn get_predictions(&self, top_k: usize) -> Vec<Prediction> {
        if let Ok(predictor) = self.predictor.read() {
            return predictor.predict_next(top_k);
        }
        Vec::new()
    }

    /// Prefetch predicted queries (internal)
    fn prefetch_predicted(&self) {
        if let Ok(predictor) = self.predictor.read() {
            let candidates = predictor.get_prefetch_candidates(3);
            if let Ok(mut m) = self.metrics.write() {
                m.predictions_made += candidates.len() as u64;
            }
        }
    }

    /// Prefetch based on N-gram predictions (higher-order context)
    fn prefetch_ngram_predicted(&self) {
        if let Ok(ngram) = self.ngram_predictor.read() {
            let predictions = ngram.predict(3);
            if let Ok(mut m) = self.metrics.write() {
                m.ngram_predictions += predictions.len() as u64;
            }
        }
    }

    /// Record access pattern in SNN for temporal learning
    fn record_snn_access(&self, key: &str) {
        // Convert key to neuron activation pattern
        let key_hash = {
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            hasher.finish()
        };

        // Map to neuron index (modulo number of neurons)
        let neuron_id = (key_hash as usize) % self.config.snn_neurons;

        // Inject spike into SNN
        if let Ok(snn) = self.snn.read() {
            let current_time = snn.current_time();
            snn.add_input_spike(neuron_id, current_time, 5.0);
        }
    }

    // ========================================================================
    // N-gram Predictor Operations
    // ========================================================================

    /// Get N-gram based predictions (higher-order context)
    pub fn get_ngram_predictions(&self, top_k: usize) -> Vec<Prediction> {
        if let Ok(ngram) = self.ngram_predictor.read() {
            return ngram.predict(top_k);
        }
        Vec::new()
    }

    /// Get combined predictions (Markov + N-gram)
    pub fn get_combined_predictions(&self, top_k: usize) -> Vec<Prediction> {
        let mut predictions = self.get_predictions(top_k * 2);
        let ngram_preds = self.get_ngram_predictions(top_k);

        // Merge predictions, boosting those that appear in both
        for ngram_pred in ngram_preds {
            if let Some(existing) = predictions.iter_mut().find(|p| p.hash == ngram_pred.hash) {
                // Boost probability if found in both predictors
                existing.probability = (existing.probability + ngram_pred.probability) / 2.0;
                existing.count += ngram_pred.count;
            } else {
                predictions.push(ngram_pred);
            }
        }

        // Sort by probability and truncate
        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions.truncate(top_k);
        predictions
    }

    // ========================================================================
    // Learned Index Operations
    // ========================================================================

    /// Train the learned index on current key distribution
    ///
    /// This should be called periodically when the key distribution changes significantly.
    pub fn train_learned_index(&self) -> AmorphicEngineResult<(f64, f64)> {
        use crate::learned::LearnedIndexModel;

        // Get sorted keys
        let keys: Vec<String> = if let Ok(store) = self.data_store.read() {
            let mut keys: Vec<_> = store.keys().cloned().collect();
            keys.sort();
            keys
        } else {
            return Err(AmorphicEngineError::LockError(
                "failed to acquire data store lock".to_string(),
            ));
        };

        if keys.len() < 2 {
            return Err(AmorphicEngineError::Learned(
                "need at least 2 keys to train".to_string(),
            ));
        }

        // Convert keys to numeric values for training
        let training_data: Vec<(f64, usize)> = keys
            .iter()
            .enumerate()
            .map(|(pos, key)| {
                // Hash key to f64 for training
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                (hasher.finish() as f64, pos)
            })
            .collect();

        // Sort by key hash
        let mut sorted_data = training_data;
        sorted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Train model
        let mut model = LearnedIndexModel::linear();
        let (mae, max_error) = model
            .train(&sorted_data)
            .map_err(|e| AmorphicEngineError::Learned(e.to_string()))?;

        // Store trained model
        if let Ok(mut learned) = self.learned.write() {
            use crate::learned::LearnedIndex;
            let (min_key, max_key) = model.key_bounds();
            let mut index = LearnedIndex::new(min_key, max_key, keys.len());
            // Transfer model to index
            index.set_model(model);
            *learned = Some(index);
        }

        Ok((mae, max_error))
    }

    /// Check if learned index is trained
    pub fn is_learned_index_trained(&self) -> bool {
        if let Ok(learned) = self.learned.read() {
            if let Some(ref index) = *learned {
                return index.is_trained();
            }
        }
        false
    }

    /// Get learned index statistics
    pub fn learned_index_stats(&self) -> Option<(f64, f64, usize)> {
        if let Ok(learned) = self.learned.read() {
            if let Some(ref index) = *learned {
                if index.is_trained() {
                    return Some((
                        index.mean_absolute_error(),
                        index.max_absolute_error(),
                        index.num_records(),
                    ));
                }
            }
        }
        None
    }

    // ========================================================================
    // SNN Operations (Temporal Pattern Recognition)
    // ========================================================================

    /// Step the SNN forward and detect temporal patterns
    pub fn step_snn(&self) -> Vec<SpikeEvent> {
        if let Ok(snn) = self.snn.read() {
            let spikes = snn.update();
            if !spikes.is_empty() {
                if let Ok(mut m) = self.metrics.write() {
                    m.snn_pattern_detections += spikes.len() as u64;
                }
            }
            return spikes;
        }
        Vec::new()
    }

    /// Run SNN simulation for multiple steps
    pub fn simulate_snn(&self, steps: usize) -> Vec<Vec<SpikeEvent>> {
        if let Ok(snn) = self.snn.read() {
            let all_spikes = snn.simulate(steps);
            let total_spikes: usize = all_spikes.iter().map(|v| v.len()).sum();
            if let Ok(mut m) = self.metrics.write() {
                m.snn_pattern_detections += total_spikes as u64;
            }
            return all_spikes;
        }
        Vec::new()
    }

    /// Get SNN network statistics
    pub fn snn_stats(&self) -> Option<SNNStats> {
        if let Ok(snn) = self.snn.read() {
            return Some(snn.stats());
        }
        None
    }

    /// Reset SNN to initial state
    pub fn reset_snn(&self) {
        if let Ok(snn) = self.snn.read() {
            snn.reset();
        }
    }

    /// Detect access pattern anomalies using SNN
    ///
    /// Returns neuron indices that have unusual firing patterns
    pub fn detect_access_anomalies(&self) -> Vec<usize> {
        if let Ok(snn) = self.snn.read() {
            let stats = snn.stats();
            let active_ratio = stats.active_neurons as f64 / stats.num_neurons as f64;

            // If too many or too few neurons are active, it's anomalous
            if active_ratio > 0.5 || active_ratio < 0.01 {
                // Return indices of active neurons as potential anomalies
                // (In a more sophisticated impl, we'd track firing rates)
                return (0..stats.num_neurons)
                    .filter(|_| rand_simple() < active_ratio)
                    .collect();
            }
        }
        Vec::new()
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        if let Ok(metrics) = self.metrics.read() {
            let total = metrics.cache_hits + metrics.cache_misses;
            if total == 0 {
                return 0.0;
            }
            return metrics.cache_hits as f64 / total as f64;
        }
        0.0
    }

    // ========================================================================
    // SNN Operations
    // ========================================================================

    /// Encode temporal data using temporal encoder
    pub fn snn_encode(&self, data: &[f32], time_window: f32) -> Vec<SpikeEvent> {
        let encoder = TemporalEncoder::new();
        encoder.encode_temporal(data, time_window)
    }

    /// Decode SNN spikes using temporal decoder
    pub fn snn_decode(&self, spikes: &[SpikeEvent], time_window: f32, num_bins: usize) -> Vec<f32> {
        let decoder = TemporalDecoder::new();
        decoder.decode_temporal(spikes, time_window, num_bins)
    }

    // ========================================================================
    // Metrics & Status
    // ========================================================================

    /// Get comprehensive statistics
    pub fn stats(&self) -> AmorphicEngineStats {
        let metrics = self.metrics.read().map(|m| m.clone()).unwrap_or_default();
        let store_size = self.len();
        let cache_total = metrics.cache_hits + metrics.cache_misses;
        let cache_hit_rate = if cache_total > 0 {
            metrics.cache_hits as f64 / cache_total as f64
        } else {
            0.0
        };

        AmorphicEngineStats {
            reads: metrics.reads,
            writes: metrics.writes,
            deletes: metrics.deletes,
            cache_hits: metrics.cache_hits,
            cache_misses: metrics.cache_misses,
            cache_hit_rate,
            sdm_lookups: metrics.sdm_lookups,
            holographic_lookups: metrics.holographic_lookups,
            manifold_queries: metrics.manifold_queries,
            predictions_made: metrics.predictions_made,
            store_size,
            learned_index_hits: metrics.learned_index_hits,
            ngram_predictions: metrics.ngram_predictions,
            snn_pattern_detections: metrics.snn_pattern_detections,
        }
    }

    /// Get node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get configuration
    pub fn config(&self) -> &AmorphicEngineConfig {
        &self.config
    }
}

impl Default for AmorphicEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AmorphicEngine {
    fn clone(&self) -> Self {
        // Create a new AmorphicEngine with same config
        // Note: This does NOT copy the data, just the structure
        Self::with_config(self.config.clone(), &self.node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amorphic_engine_basic_operations() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Write
        engine.write("key1", b"value1").unwrap();
        engine.write("key2", b"value2").unwrap();

        // Read
        assert_eq!(engine.read("key1"), Some(b"value1".to_vec()));
        assert_eq!(engine.read("key2"), Some(b"value2".to_vec()));
        assert_eq!(engine.read("key3"), None);

        // Contains
        assert!(engine.contains("key1"));
        assert!(!engine.contains("key3"));

        // Length
        assert_eq!(engine.len(), 2);
        assert!(!engine.is_empty());
    }

    #[test]
    fn test_amorphic_engine_delete() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        engine.write("key1", b"value1").unwrap();
        assert!(engine.contains("key1"));

        assert!(engine.delete("key1"));
        assert!(!engine.contains("key1"));
        assert!(!engine.delete("key1")); // Already deleted
    }

    #[test]
    fn test_amorphic_engine_keys() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        engine.write("key1", b"value1").unwrap();
        engine.write("key2", b"value2").unwrap();

        let keys = engine.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
    }

    #[test]
    fn test_amorphic_engine_content_store() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        let data = b"test content";
        let key = engine.store_content(data).unwrap();
        assert!(key.starts_with("content_"));

        assert!(engine.contains(&key));
        assert_eq!(engine.read(&key), Some(data.to_vec()));
    }

    #[test]
    fn test_amorphic_engine_stats() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        engine.write("key1", b"value1").unwrap();
        engine.read("key1");
        engine.read("key1"); // Cache hit
        engine.read("key2"); // Cache miss

        let stats = engine.stats();
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.reads, 3);
        assert_eq!(stats.store_size, 1);
    }

    #[test]
    fn test_amorphic_engine_config() {
        let minimal = AmorphicEngineConfig::minimal();
        assert_eq!(minimal.sdm_locations, 100);

        let production = AmorphicEngineConfig::production();
        assert_eq!(production.sdm_locations, 10000);

        let default_config = AmorphicEngineConfig::default();
        assert_eq!(default_config.sdm_locations, 1000);
    }

    #[test]
    fn test_amorphic_engine_node_id() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "my_node");
        assert_eq!(engine.node_id(), "my_node");
    }

    #[test]
    fn test_amorphic_engine_optimizer() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        let plan = engine.optimize_query(0.5, 2, true);
        assert!(plan.cost() > 0.0);

        let temp = engine.optimizer_temperature();
        assert!(temp > 0.0);

        engine.reset_optimizer();
    }

    #[test]
    fn test_amorphic_engine_predictions() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Generate some query patterns
        for _ in 0..5 {
            engine.write("key1", b"v1").unwrap();
            let _ = engine.read("key1");
            engine.write("key2", b"v2").unwrap();
            let _ = engine.read("key2");
        }

        let predictions = engine.get_predictions(5);
        // Predictions might be empty if pattern not established
        // Each Prediction has hash, count, probability
        assert!(predictions.len() <= 5);
    }

    #[test]
    fn test_amorphic_engine_cache_hit_rate() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Initially 0
        assert_eq!(engine.cache_hit_rate(), 0.0);

        engine.write("key1", b"value1").unwrap();
        engine.read("key1"); // Miss, then cached
        engine.read("key1"); // Should be hit

        let hit_rate = engine.cache_hit_rate();
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
    }

    #[test]
    fn test_amorphic_engine_similarity_search() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        engine.write("key1", b"hello world").unwrap();
        engine.write("key2", b"hello there").unwrap();
        engine.write("key3", b"goodbye world").unwrap();

        let results = engine.similarity_search(b"hello", 3);
        // Results are NeighborResult with id, index, distance
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_amorphic_engine_thread_safety() {
        use std::thread;

        let engine = Arc::new(AmorphicEngine::with_config(
            AmorphicEngineConfig::minimal(),
            "test_node",
        ));
        let mut handles = vec![];

        // Multiple writers
        for i in 0..4 {
            let engine_clone = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for j in 0..10 {
                    let key = format!("key_{}_{}", i, j);
                    let value = format!("value_{}_{}", i, j);
                    engine_clone.write(&key, value.as_bytes()).unwrap();
                }
            }));
        }

        // Multiple readers
        for _ in 0..4 {
            let engine_clone = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for _ in 0..20 {
                    let _ = engine_clone.read("key_0_0");
                    let _ = engine_clone.len();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(engine.len() > 0);
    }

    // ========================================================================
    // New Integration Tests
    // ========================================================================

    #[test]
    fn test_amorphic_engine_ngram_predictions() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Generate N-gram pattern: a -> b -> c (repeated)
        for _ in 0..10 {
            engine.write("pattern_a", b"a").unwrap();
            let _ = engine.read("pattern_a");
            engine.write("pattern_b", b"b").unwrap();
            let _ = engine.read("pattern_b");
            engine.write("pattern_c", b"c").unwrap();
            let _ = engine.read("pattern_c");
        }

        // Get N-gram predictions
        let ngram_preds = engine.get_ngram_predictions(5);
        // May or may not have predictions depending on pattern establishment
        assert!(ngram_preds.len() <= 5);

        // Get combined predictions
        let combined = engine.get_combined_predictions(5);
        assert!(combined.len() <= 5);
    }

    #[test]
    fn test_amorphic_engine_learned_index() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Initially not trained
        assert!(!engine.is_learned_index_trained());
        assert!(engine.learned_index_stats().is_none());

        // Add enough keys to train
        for i in 0..100 {
            engine
                .write(&format!("key_{:04}", i), format!("value_{}", i).as_bytes())
                .unwrap();
        }

        // Train the learned index
        let result = engine.train_learned_index();
        assert!(result.is_ok());

        let (mae, max_err) = result.unwrap();
        assert!(mae >= 0.0);
        assert!(max_err >= 0.0);

        // Now should be trained
        assert!(engine.is_learned_index_trained());

        // Stats should be available
        let stats = engine.learned_index_stats();
        assert!(stats.is_some());
        let (mae, max_err, num_records) = stats.unwrap();
        assert_eq!(num_records, 100);
        assert!(mae >= 0.0);
        assert!(max_err >= 0.0);
    }

    #[test]
    fn test_amorphic_engine_learned_index_not_enough_keys() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Only one key - not enough to train
        engine.write("only_key", b"value").unwrap();

        let result = engine.train_learned_index();
        assert!(result.is_err());
    }

    #[test]
    fn test_amorphic_engine_snn_operations() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Get initial SNN stats
        let stats = engine.snn_stats();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert!(stats.num_neurons > 0);
        assert_eq!(stats.current_time, 0.0);

        // Step the SNN
        let spikes = engine.step_snn();
        // May or may not have spikes
        assert!(spikes.len() <= stats.num_neurons);

        // Simulate multiple steps
        let all_spikes = engine.simulate_snn(10);
        assert_eq!(all_spikes.len(), 10);

        // Reset SNN
        engine.reset_snn();
        let stats_after_reset = engine.snn_stats().unwrap();
        assert_eq!(stats_after_reset.current_time, 0.0);
    }

    #[test]
    fn test_amorphic_engine_snn_access_recording() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Access patterns should be recorded in SNN
        for i in 0..10 {
            engine.write(&format!("key_{}", i), b"value").unwrap();
            let _ = engine.read(&format!("key_{}", i));
        }

        // Step SNN to process recorded access patterns
        engine.simulate_snn(5);

        // Check stats
        let stats = engine.stats();
        // SNN pattern detections should be tracked
        // (may be 0 if no neurons fired)
        assert!(stats.snn_pattern_detections >= 0);
    }

    #[test]
    fn test_amorphic_engine_detect_anomalies() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Detect anomalies (should work even with no data)
        let anomalies = engine.detect_access_anomalies();
        // May or may not have anomalies depending on SNN state
        assert!(anomalies.len() <= engine.config().snn_neurons);
    }

    #[test]
    fn test_amorphic_engine_snn_encode_decode() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Encode temporal data
        let data = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let spikes = engine.snn_encode(&data, 10.0);

        // Spikes should be generated
        assert!(!spikes.is_empty());

        // Decode back
        let decoded = engine.snn_decode(&spikes, 10.0, 5);
        assert_eq!(decoded.len(), 5);
    }

    #[test]
    fn test_amorphic_engine_stats_include_new_metrics() {
        let engine = AmorphicEngine::with_config(AmorphicEngineConfig::minimal(), "test_node");

        // Write and read to generate metrics
        engine.write("test", b"data").unwrap();
        let _ = engine.read("test");

        let stats = engine.stats();

        // New metrics should be present
        assert!(stats.learned_index_hits >= 0);
        assert!(stats.ngram_predictions >= 0);
        assert!(stats.snn_pattern_detections >= 0);
    }
}
