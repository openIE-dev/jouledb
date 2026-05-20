//! # Vector Similarity Search
//!
//! Provides vector indexing and approximate nearest neighbor (ANN) search.
//!
//! ## Features
//!
//! - Multiple similarity metrics (Cosine, Euclidean, Dot Product)
//! - HNSW-inspired approximate nearest neighbor search
//! - Configurable index parameters
//! - Batch insert and search operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_features::vector::{VectorIndex, VectorConfig, SimilarityMetric};
//!
//! let config = VectorConfig::new(128)
//!     .with_metric(SimilarityMetric::Cosine)
//!     .with_ef_construction(200)
//!     .with_m(16);
//!
//! let mut index = VectorIndex::new(config);
//! index.insert("doc1", vec![0.1, 0.2, 0.3, /* ... */]);
//!
//! let results = index.search(&query_vector, 10);
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Configuration for vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Dimensionality of vectors.
    pub dimensions: usize,
    /// Similarity metric to use.
    pub metric: SimilarityMetric,
    /// Number of bi-directional links per node (HNSW M parameter).
    pub m: usize,
    /// Size of dynamic candidate list during construction (HNSW ef_construction).
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (HNSW ef).
    pub ef_search: usize,
    /// Maximum number of layers in the graph.
    pub max_layers: usize,
}

impl VectorConfig {
    /// Create a new vector config with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            metric: SimilarityMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_layers: 16,
        }
    }

    /// Set the similarity metric.
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the M parameter (bi-directional links per node).
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the ef_construction parameter.
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the ef_search parameter.
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self::new(128)
    }
}

/// Similarity metric for vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity (normalized dot product).
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Dot product (inner product).
    DotProduct,
    /// Manhattan (L1) distance.
    Manhattan,
}

impl SimilarityMetric {
    /// Calculate similarity/distance between two vectors (SIMD-optimized).
    ///
    /// For Cosine and DotProduct, higher is better.
    /// For Euclidean and Manhattan, lower is better.
    ///
    /// This method automatically uses SIMD instructions when available:
    /// - x86_64: AVX2 (8 floats) or SSE (4 floats)
    /// - aarch64: NEON (4 floats)
    /// - Others: Auto-vectorized scalar fallback
    #[inline]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        match self {
            SimilarityMetric::Cosine => crate::simd::cosine_similarity(a, b),
            SimilarityMetric::Euclidean => crate::simd::euclidean_distance(a, b),
            SimilarityMetric::DotProduct => crate::simd::dot_product(a, b),
            SimilarityMetric::Manhattan => crate::simd::manhattan_distance(a, b),
        }
    }

    /// Calculate similarity/distance using scalar implementation (for comparison/testing).
    #[inline]
    pub fn calculate_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        match self {
            SimilarityMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot / (norm_a * norm_b)
                }
            }
            SimilarityMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            SimilarityMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            SimilarityMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        }
    }

    /// Returns true if higher values indicate more similarity.
    pub fn higher_is_better(&self) -> bool {
        matches!(
            self,
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct
        )
    }
}

/// A search result from vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The ID of the matched vector.
    pub id: String,
    /// The similarity/distance score.
    pub score: f32,
    /// Optional metadata associated with the vector.
    pub metadata: Option<serde_json::Value>,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(id: String, score: f32) -> Self {
        Self {
            id,
            score,
            metadata: None,
        }
    }

    /// Create a search result with metadata.
    pub fn with_metadata(id: String, score: f32, metadata: serde_json::Value) -> Self {
        Self {
            id,
            score,
            metadata: Some(metadata),
        }
    }
}

/// Internal node for priority queue operations.
#[derive(Debug, Clone)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// A stored vector with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredVector {
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
    /// Neighbors at each layer (HNSW structure).
    neighbors: Vec<Vec<usize>>,
    /// Layer this node was inserted at.
    layer: usize,
}

/// Vector index for similarity search.
///
/// Uses an HNSW-inspired algorithm for approximate nearest neighbor search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndex {
    config: VectorConfig,
    vectors: Vec<StoredVector>,
    id_to_index: HashMap<String, usize>,
    /// Entry point for search (highest layer node).
    entry_point: Option<usize>,
    /// Current maximum layer.
    max_layer: usize,
}

impl VectorIndex {
    /// Create a new vector index with the given configuration.
    pub fn new(config: VectorConfig) -> Self {
        Self {
            config,
            vectors: Vec::new(),
            id_to_index: HashMap::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Create a vector index with default configuration.
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self::new(VectorConfig::new(dimensions))
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the configuration.
    pub fn config(&self) -> &VectorConfig {
        &self.config
    }

    /// Calculate the layer for a new node (probabilistic).
    fn random_layer(&self) -> usize {
        let mut layer = 0;
        let ml = 1.0 / (self.config.m as f32).ln();
        while rand_float() < (-rand_float().ln() * ml).exp() && layer < self.config.max_layers - 1 {
            layer += 1;
        }
        layer.min(self.config.max_layers - 1)
    }

    /// Insert a vector with the given ID.
    pub fn insert(&mut self, id: impl Into<String>, vector: Vec<f32>) -> Result<(), VectorError> {
        self.insert_with_metadata(id, vector, None)
    }

    /// Insert a vector with metadata.
    pub fn insert_with_metadata(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), VectorError> {
        let id = id.into();

        // Validate dimensions
        if vector.len() != self.config.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Check for duplicate
        if self.id_to_index.contains_key(&id) {
            return Err(VectorError::DuplicateId(id));
        }

        let node_index = self.vectors.len();
        let node_layer = if self.vectors.is_empty() {
            0
        } else {
            self.random_layer()
        };

        // Initialize neighbors for each layer
        let neighbors = vec![Vec::new(); node_layer + 1];

        let stored = StoredVector {
            id: id.clone(),
            vector,
            metadata,
            neighbors,
            layer: node_layer,
        };

        self.vectors.push(stored);
        self.id_to_index.insert(id, node_index);

        // Update entry point if needed
        if self.entry_point.is_none() || node_layer > self.max_layer {
            self.entry_point = Some(node_index);
            self.max_layer = node_layer;
        }

        // Connect the new node to its neighbors
        if self.vectors.len() > 1 {
            self.connect_node(node_index)?;
        }

        Ok(())
    }

    /// Connect a newly inserted node to its neighbors.
    fn connect_node(&mut self, node_index: usize) -> Result<(), VectorError> {
        let node_layer = self.vectors[node_index].layer;
        let node_vector = self.vectors[node_index].vector.clone();

        // Find entry point and traverse down to the node's layer
        let mut current = self.entry_point.unwrap();

        // Traverse from top layer down to node_layer + 1
        for layer in (node_layer + 1..=self.max_layer).rev() {
            current = self
                .search_layer(&node_vector, current, 1, layer)
                .pop()
                .map(|c| c.id)
                .unwrap_or(current);
        }

        // For each layer from node_layer down to 0, find and connect neighbors
        for layer in (0..=node_layer).rev() {
            let candidates =
                self.search_layer(&node_vector, current, self.config.ef_construction, layer);

            // Select M best neighbors
            let neighbors: Vec<usize> = candidates
                .into_iter()
                .take(self.config.m)
                .map(|c| c.id)
                .collect();

            // Update the node's neighbors at this layer
            if layer < self.vectors[node_index].neighbors.len() {
                self.vectors[node_index].neighbors[layer] = neighbors.clone();
            }

            // Add bidirectional connections
            for &neighbor_idx in &neighbors {
                if neighbor_idx < self.vectors.len()
                    && layer < self.vectors[neighbor_idx].neighbors.len()
                {
                    let neighbor_neighbors = &mut self.vectors[neighbor_idx].neighbors[layer];
                    if !neighbor_neighbors.contains(&node_index) {
                        neighbor_neighbors.push(node_index);
                        // Prune if necessary
                        if neighbor_neighbors.len() > self.config.m * 2 {
                            self.prune_neighbors(neighbor_idx, layer);
                        }
                    }
                }
            }

            if let Some(first) = neighbors.first() {
                current = *first;
            }
        }

        Ok(())
    }

    /// Prune neighbors to keep only the best M*2 connections.
    fn prune_neighbors(&mut self, node_index: usize, layer: usize) {
        let node_vector = self.vectors[node_index].vector.clone();
        let neighbors = &self.vectors[node_index].neighbors[layer];

        let mut scored: Vec<(usize, f32)> = neighbors
            .iter()
            .filter_map(|&idx| {
                if idx < self.vectors.len() {
                    let dist = self
                        .config
                        .metric
                        .calculate(&node_vector, &self.vectors[idx].vector);
                    Some((idx, dist))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance/similarity
        if self.config.metric.higher_is_better() {
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        } else {
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        }

        self.vectors[node_index].neighbors[layer] = scored
            .into_iter()
            .take(self.config.m * 2)
            .map(|(idx, _)| idx)
            .collect();
    }

    /// Search a single layer for nearest neighbors.
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry_dist = self
            .config
            .metric
            .calculate(query, &self.vectors[entry_point].vector);

        visited[entry_point] = true;
        candidates.push(Candidate {
            id: entry_point,
            distance: if self.config.metric.higher_is_better() {
                -entry_dist
            } else {
                entry_dist
            },
        });
        results.push(Candidate {
            id: entry_point,
            distance: if self.config.metric.higher_is_better() {
                entry_dist
            } else {
                -entry_dist
            },
        });

        while let Some(current) = candidates.pop() {
            let worst_result = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);

            // For distance metrics, current.distance is positive (lower is better)
            // For similarity metrics, current.distance is negative (higher original is better)
            let current_dist = if self.config.metric.higher_is_better() {
                -current.distance
            } else {
                current.distance
            };

            let worst_dist = if self.config.metric.higher_is_better() {
                worst_result // This is the similarity value (higher is better)
            } else {
                -worst_result // This is the distance value (lower is better)
            };

            // Stop if current candidate is worse than the worst result
            if self.config.metric.higher_is_better() {
                if current_dist < worst_dist && results.len() >= ef {
                    break;
                }
            } else {
                if current_dist > worst_dist && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors at this layer
            if layer < self.vectors[current.id].neighbors.len() {
                for &neighbor_idx in &self.vectors[current.id].neighbors[layer] {
                    if neighbor_idx < self.vectors.len() && !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;
                        let neighbor_dist = self
                            .config
                            .metric
                            .calculate(query, &self.vectors[neighbor_idx].vector);

                        let should_add = if self.config.metric.higher_is_better() {
                            results.len() < ef || neighbor_dist > worst_dist
                        } else {
                            results.len() < ef || neighbor_dist < worst_dist
                        };

                        if should_add {
                            candidates.push(Candidate {
                                id: neighbor_idx,
                                distance: if self.config.metric.higher_is_better() {
                                    -neighbor_dist
                                } else {
                                    neighbor_dist
                                },
                            });
                            results.push(Candidate {
                                id: neighbor_idx,
                                distance: if self.config.metric.higher_is_better() {
                                    neighbor_dist
                                } else {
                                    -neighbor_dist
                                },
                            });

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert results to vector sorted by score
        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|c| Candidate {
                id: c.id,
                distance: if self.config.metric.higher_is_better() {
                    c.distance
                } else {
                    -c.distance
                },
            })
            .collect();

        // Sort by distance/similarity
        if self.config.metric.higher_is_better() {
            result_vec.sort_by(|a, b| {
                b.distance
                    .partial_cmp(&a.distance)
                    .unwrap_or(Ordering::Equal)
            });
        } else {
            result_vec.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
        }

        result_vec
    }

    /// Brute-force search (exact, used for small indexes).
    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.id_to_index.values().any(|&i| i == *idx))
            .map(|(idx, v)| {
                let score = self.config.metric.calculate(query, &v.vector);
                (idx, score)
            })
            .collect();

        // Sort by score
        if self.config.metric.higher_is_better() {
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        } else {
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        }

        scored
            .into_iter()
            .take(k)
            .map(|(idx, score)| {
                let stored = &self.vectors[idx];
                SearchResult {
                    id: stored.id.clone(),
                    score,
                    metadata: stored.metadata.clone(),
                }
            })
            .collect()
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorError> {
        if query.len() != self.config.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Use brute force for small indexes (more accurate and fast enough)
        if self.vectors.len() <= 100 {
            return Ok(self.brute_force_search(query, k));
        }

        let entry_point = self.entry_point.unwrap();
        let mut current = entry_point;

        // Traverse from top layer down to layer 1
        for layer in (1..=self.max_layer).rev() {
            let candidates = self.search_layer(query, current, 1, layer);
            if let Some(best) = candidates.first() {
                current = best.id;
            }
        }

        // Search layer 0 with ef_search
        let candidates = self.search_layer(query, current, self.config.ef_search.max(k), 0);

        // Return top k results
        let results: Vec<SearchResult> = candidates
            .into_iter()
            .take(k)
            .map(|c| {
                let stored = &self.vectors[c.id];
                SearchResult {
                    id: stored.id.clone(),
                    score: c.distance,
                    metadata: stored.metadata.clone(),
                }
            })
            .collect();

        Ok(results)
    }

    /// Search with a filter function.
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        filter: F,
    ) -> Result<Vec<SearchResult>, VectorError>
    where
        F: Fn(&str, Option<&serde_json::Value>) -> bool,
    {
        // For filtered search, we search for more candidates and then filter
        let candidates = self.search(query, k * 10)?;

        let filtered: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|r| filter(&r.id, r.metadata.as_ref()))
            .take(k)
            .collect();

        Ok(filtered)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<&[f32]> {
        self.id_to_index
            .get(id)
            .and_then(|&idx| self.vectors.get(idx))
            .map(|v| v.vector.as_slice())
    }

    /// Get vector with metadata by ID.
    pub fn get_with_metadata(&self, id: &str) -> Option<(&[f32], Option<&serde_json::Value>)> {
        self.id_to_index
            .get(id)
            .and_then(|&idx| self.vectors.get(idx))
            .map(|v| (v.vector.as_slice(), v.metadata.as_ref()))
    }

    /// Check if a vector exists.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.contains_key(id)
    }

    /// Remove a vector by ID.
    /// Note: This is a soft delete that doesn't actually remove the vector
    /// to avoid index corruption. Use `rebuild()` to compact the index.
    pub fn remove(&mut self, id: &str) -> bool {
        // For simplicity, we just remove from the ID map
        // A full implementation would need to update neighbor links
        self.id_to_index.remove(id).is_some()
    }

    /// Get all vector IDs.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.id_to_index.keys().map(|s| s.as_str())
    }

    /// Batch insert multiple vectors.
    pub fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<usize, VectorError> {
        let mut inserted = 0;
        for (id, vector, metadata) in vectors {
            self.insert_with_metadata(id, vector, metadata)?;
            inserted += 1;
        }
        Ok(inserted)
    }

    /// Clear all vectors from the index.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.id_to_index.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }
}

/// Errors that can occur during vector operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Duplicate vector ID: {0}")]
    DuplicateId(String),

    #[error("Vector not found: {0}")]
    NotFound(String),

    #[error("Index error: {0}")]
    IndexError(String),
}

/// Simple random float generator (0.0 to 1.0).
/// In production, you'd use a proper RNG.
fn rand_float() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f32 / u32::MAX as f32).fract()
}

/// Normalize a vector to unit length.
pub fn normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

/// Calculate the centroid of a set of vectors.
pub fn centroid(vectors: &[&[f32]]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }

    let dim = vectors[0].len();
    let mut result = vec![0.0; dim];

    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            result[i] += val;
        }
    }

    let n = vectors.len() as f32;
    for val in result.iter_mut() {
        *val /= n;
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn test_vector_config() {
        let config = VectorConfig::new(128)
            .with_metric(SimilarityMetric::Euclidean)
            .with_m(32)
            .with_ef_construction(100);

        assert_eq!(config.dimensions, 128);
        assert_eq!(config.metric, SimilarityMetric::Euclidean);
        assert_eq!(config.m, 32);
        assert_eq!(config.ef_construction, 100);
    }

    #[test]
    fn test_similarity_metrics() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        // Cosine: orthogonal = 0, same direction = 1
        assert!((SimilarityMetric::Cosine.calculate(&a, &b) - 0.0).abs() < 0.001);
        assert!((SimilarityMetric::Cosine.calculate(&a, &c) - 1.0).abs() < 0.001);

        // Euclidean
        let euclidean = SimilarityMetric::Euclidean.calculate(&a, &b);
        assert!((euclidean - 2.0_f32.sqrt()).abs() < 0.001);

        // Dot product
        assert_eq!(SimilarityMetric::DotProduct.calculate(&a, &b), 0.0);
        assert_eq!(SimilarityMetric::DotProduct.calculate(&a, &c), 1.0);

        // Manhattan
        assert_eq!(SimilarityMetric::Manhattan.calculate(&a, &b), 2.0);
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = VectorIndex::with_dimensions(4);

        // Insert some vectors
        index.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b", vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        index.insert("c", vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert("d", vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.len(), 4);

        // Search for nearest to "a"
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a"); // Exact match should be first
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = VectorIndex::with_dimensions(4);

        // Wrong dimension should fail
        let result = index.insert("test", vec![1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_duplicate_id() {
        let mut index = VectorIndex::with_dimensions(4);

        index.insert("test", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = index.insert("test", vec![0.0, 1.0, 0.0, 0.0]);
        assert!(matches!(result, Err(VectorError::DuplicateId(_))));
    }

    #[test]
    fn test_metadata() {
        let mut index = VectorIndex::with_dimensions(4);

        let metadata = serde_json::json!({
            "category": "test",
            "score": 0.95
        });

        index
            .insert_with_metadata("doc1", vec![1.0, 0.0, 0.0, 0.0], Some(metadata.clone()))
            .unwrap();

        let (_, stored_meta) = index.get_with_metadata("doc1").unwrap();
        assert_eq!(stored_meta, Some(&metadata));
    }

    #[test]
    fn test_search_with_filter() {
        let mut index = VectorIndex::with_dimensions(4);

        index
            .insert_with_metadata(
                "doc1",
                vec![1.0, 0.0, 0.0, 0.0],
                Some(serde_json::json!({"type": "article"})),
            )
            .unwrap();

        index
            .insert_with_metadata(
                "doc2",
                vec![0.9, 0.1, 0.0, 0.0],
                Some(serde_json::json!({"type": "book"})),
            )
            .unwrap();

        // Filter for only articles
        let results = index
            .search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, |_id, meta| {
                meta.and_then(|m| m.get("type"))
                    .and_then(|t| t.as_str())
                    .map(|t| t == "article")
                    .unwrap_or(false)
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_batch_insert() {
        let mut index = VectorIndex::with_dimensions(4);

        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0, 0.0], None),
            ("b".to_string(), vec![0.0, 1.0, 0.0, 0.0], None),
            ("c".to_string(), vec![0.0, 0.0, 1.0, 0.0], None),
        ];

        let inserted = index.insert_batch(vectors).unwrap();
        assert_eq!(inserted, 3);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_euclidean_search() {
        let config = VectorConfig::new(4).with_metric(SimilarityMetric::Euclidean);
        let mut index = VectorIndex::new(config);

        index.insert("a", vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("c", vec![2.0, 0.0, 0.0, 0.0]).unwrap();

        // Search from origin - closest should be "a"
        let results = index.search(&[0.1, 0.0, 0.0, 0.0], 3).unwrap();

        assert!(!results.is_empty());
        // "a" at origin should be closest to query at 0.1
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_centroid() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![2.0, 2.0];
        let v3 = vec![4.0, 4.0];

        let c = centroid(&[&v1[..], &v2[..], &v3[..]]).unwrap();
        assert!((c[0] - 2.0).abs() < 0.0001);
        assert!((c[1] - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_contains_and_get() {
        let mut index = VectorIndex::with_dimensions(4);

        index.insert("test", vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(index.contains("test"));
        assert!(!index.contains("nonexistent"));

        let vec = index.get("test").unwrap();
        assert_eq!(vec, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_clear() {
        let mut index = VectorIndex::with_dimensions(4);

        index.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b", vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);

        index.clear();

        assert!(index.is_empty());
        assert!(!index.contains("a"));
    }

    #[test]
    fn test_larger_index() {
        let mut index = VectorIndex::with_dimensions(16);

        // Insert 100 vectors
        for i in 0..100 {
            let vec = create_test_vector(16, i as f32);
            index.insert(format!("vec_{}", i), vec).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search should return results
        let query = create_test_vector(16, 50.5);
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        // Verify SIMD implementations produce same results as scalar
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.2).cos()).collect();

        let epsilon = 1e-4;

        // Test all metrics
        for metric in [
            SimilarityMetric::Cosine,
            SimilarityMetric::Euclidean,
            SimilarityMetric::DotProduct,
            SimilarityMetric::Manhattan,
        ] {
            let simd_result = metric.calculate(&a, &b);
            let scalar_result = metric.calculate_scalar(&a, &b);

            assert!(
                (simd_result - scalar_result).abs() < epsilon,
                "{:?}: SIMD={} vs Scalar={}, diff={}",
                metric,
                simd_result,
                scalar_result,
                (simd_result - scalar_result).abs()
            );
        }
    }

    #[test]
    fn test_simd_large_vectors() {
        // Test with large vectors to ensure SIMD paths work correctly
        let sizes = [64, 128, 256, 512, 1024];

        for size in sizes {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

            // These should not panic
            let _ = SimilarityMetric::Cosine.calculate(&a, &b);
            let _ = SimilarityMetric::Euclidean.calculate(&a, &b);
            let _ = SimilarityMetric::DotProduct.calculate(&a, &b);
            let _ = SimilarityMetric::Manhattan.calculate(&a, &b);
        }
    }
}
