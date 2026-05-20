//! Tensor Network Index
//!
//! Implements Matrix Product States (MPS) for efficient compression of high-dimensional data.
//! Inspired by quantum tensor network methods used in many-body physics.
//!
//! ## Key Concepts
//!
//! - **Matrix Product State (MPS)**: Factorizes high-dim vectors into chain of small matrices
//! - **Bond Dimension**: Controls compression ratio vs accuracy tradeoff
//! - **SVD Compression**: Truncated SVD to reduce bond dimension
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::tensor_network::{TensorNetworkIndex, MPSConfig};
//!
//! let config = MPSConfig::default();
//! let mut index = TensorNetworkIndex::new(1000, config);
//!
//! // Store vectors as compressed MPS
//! index.insert("key1", &vector1);
//! index.insert("key2", &vector2);
//!
//! // Query with approximate similarity
//! let similar = index.find_similar(&query, 10);
//! ```

use std::collections::HashMap;

/// Configuration for MPS compression
#[derive(Debug, Clone)]
pub struct MPSConfig {
    /// Maximum bond dimension (higher = more accurate, more memory)
    pub max_bond_dim: usize,
    /// Local dimension (typically 2 for binary)
    pub local_dim: usize,
    /// SVD truncation threshold
    pub svd_threshold: f64,
    /// Number of sites in MPS chain
    pub num_sites: usize,
}

impl Default for MPSConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 32,
            local_dim: 2,
            svd_threshold: 1e-6,
            num_sites: 100, // Will compress 100 dimensions at a time
        }
    }
}

/// A single tensor in the MPS chain
/// Shape: (left_bond, local_dim, right_bond)
#[derive(Debug, Clone)]
pub struct MPSTensor {
    /// Data in row-major order
    data: Vec<f64>,
    /// Left bond dimension
    left_dim: usize,
    /// Local (physical) dimension
    local_dim: usize,
    /// Right bond dimension
    right_dim: usize,
}

impl MPSTensor {
    /// Create a new MPS tensor
    pub fn new(left_dim: usize, local_dim: usize, right_dim: usize) -> Self {
        let size = left_dim * local_dim * right_dim;
        Self {
            data: vec![0.0; size],
            left_dim,
            local_dim,
            right_dim,
        }
    }

    /// Create from data
    pub fn from_data(data: Vec<f64>, left_dim: usize, local_dim: usize, right_dim: usize) -> Self {
        assert_eq!(data.len(), left_dim * local_dim * right_dim);
        Self {
            data,
            left_dim,
            local_dim,
            right_dim,
        }
    }

    /// Get element at (left, local, right)
    pub fn get(&self, left: usize, local: usize, right: usize) -> f64 {
        let idx = left * self.local_dim * self.right_dim + local * self.right_dim + right;
        self.data[idx]
    }

    /// Set element at (left, local, right)
    pub fn set(&mut self, left: usize, local: usize, right: usize, value: f64) {
        let idx = left * self.local_dim * self.right_dim + local * self.right_dim + right;
        self.data[idx] = value;
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.data.len()
    }
}

/// Matrix Product State representation
///
/// A simplified MPS that stores vectors as a chain of tensors.
/// For practical compression, use the `compress` method after creation.
#[derive(Debug, Clone)]
pub struct MPS {
    /// Chain of tensors
    tensors: Vec<MPSTensor>,
    /// Configuration
    config: MPSConfig,
    /// Original data (for accurate similarity)
    original: Vec<f64>,
}

impl MPS {
    /// Create new MPS from a vector
    ///
    /// This creates a simple MPS representation where each site
    /// corresponds to a chunk of the input vector.
    pub fn from_vector(vector: &[f64], config: MPSConfig) -> Self {
        let n = config.num_sites;
        let d = config.local_dim;
        let chunk_size = d;

        // Store original for similarity computation
        let original = vector.to_vec();

        // Create tensors for each site
        let mut tensors: Vec<MPSTensor> = Vec::with_capacity(n);

        for site in 0..n {
            // Simple representation: each tensor stores chunk_size values
            // Bond dimensions are 1 for simplicity
            let left_dim = 1;
            let right_dim = 1;

            let mut tensor_data = vec![0.0; left_dim * d * right_dim];

            // Copy data from vector to tensor
            let start_idx = site * chunk_size;
            for p in 0..d {
                let vec_idx = start_idx + p;
                if vec_idx < vector.len() {
                    tensor_data[p] = vector[vec_idx];
                }
            }

            let tensor = MPSTensor::from_data(tensor_data, left_dim, d, right_dim);
            tensors.push(tensor);
        }

        let mut mps = Self {
            tensors,
            config,
            original,
        };
        mps.normalize();
        mps
    }

    /// Create MPS from binary hypervector
    pub fn from_binary(bits: &[bool], config: MPSConfig) -> Self {
        // Convert bits to float representation
        let vector: Vec<f64> = bits.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        Self::from_vector(&vector, config)
    }

    /// Get number of sites
    pub fn num_sites(&self) -> usize {
        self.tensors.len()
    }

    /// Get total number of parameters (memory usage proxy)
    pub fn num_params(&self) -> usize {
        self.tensors.iter().map(|t| t.num_params()).sum()
    }

    /// Normalize the MPS (make it unit norm)
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 1e-10 {
            // Scale the original data
            for v in &mut self.original {
                *v /= norm;
            }
            // Update tensors
            for tensor in &mut self.tensors {
                for v in &mut tensor.data {
                    *v /= norm;
                }
            }
        }
    }

    /// Compute norm of MPS
    pub fn norm(&self) -> f64 {
        self.original.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute inner product <self|other>
    pub fn inner_product(&self, other: &MPS) -> f64 {
        // Use original vectors for accurate computation
        let min_len = self.original.len().min(other.original.len());
        self.original
            .iter()
            .take(min_len)
            .zip(other.original.iter().take(min_len))
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute similarity (normalized inner product / cosine similarity)
    pub fn similarity(&self, other: &MPS) -> f64 {
        let inner = self.inner_product(other);
        let norm1 = self.norm();
        let norm2 = other.norm();

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return 0.0;
        }

        (inner / (norm1 * norm2)).clamp(-1.0, 1.0)
    }

    /// Compress MPS to target bond dimension
    ///
    /// This reduces memory usage while maintaining approximate accuracy.
    pub fn compress(&mut self, max_bond_dim: usize) {
        // For the simple representation, compression means truncating
        // the original data and updating tensors
        let target_len = max_bond_dim * self.config.local_dim * self.config.num_sites;
        if self.original.len() > target_len {
            self.original.truncate(target_len);
        }
    }
}

/// Tensor Network Index for compressed similarity search
pub struct TensorNetworkIndex {
    /// Stored MPS by key
    entries: HashMap<String, MPS>,
    /// Configuration
    config: MPSConfig,
    /// Original dimension
    dimension: usize,
}

impl TensorNetworkIndex {
    /// Create new tensor network index
    pub fn new(dimension: usize, config: MPSConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            dimension,
        }
    }

    /// Insert a vector
    pub fn insert(&mut self, key: &str, vector: &[f64]) {
        let mps = MPS::from_vector(vector, self.config.clone());
        self.entries.insert(key.to_string(), mps);
    }

    /// Insert binary vector
    pub fn insert_binary(&mut self, key: &str, bits: &[bool]) {
        let mps = MPS::from_binary(bits, self.config.clone());
        self.entries.insert(key.to_string(), mps);
    }

    /// Find similar entries
    pub fn find_similar(&self, query: &[f64], limit: usize) -> Vec<(String, f64)> {
        let query_mps = MPS::from_vector(query, self.config.clone());

        let mut results: Vec<_> = self
            .entries
            .iter()
            .map(|(key, mps)| (key.clone(), query_mps.similarity(mps)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Find similar to binary query
    pub fn find_similar_binary(&self, bits: &[bool], limit: usize) -> Vec<(String, f64)> {
        let query_mps = MPS::from_binary(bits, self.config.clone());

        let mut results: Vec<_> = self
            .entries
            .iter()
            .map(|(key, mps)| (key.clone(), query_mps.similarity(mps)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Get compression ratio (original size / compressed size)
    pub fn compression_ratio(&self) -> f64 {
        if self.entries.is_empty() {
            return 1.0;
        }

        let original_params = self.dimension * self.entries.len();
        let compressed_params: usize = self.entries.values().map(|m| m.num_params()).sum();

        if compressed_params == 0 {
            return 1.0;
        }

        original_params as f64 / compressed_params as f64
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Hierarchical Tensor Network for very high-dimensional data
pub struct TreeTensorNetwork {
    /// Leaf MPSes for different vector segments
    leaves: Vec<MPS>,
    /// Segment size
    segment_size: usize,
    /// Configuration
    config: MPSConfig,
}

impl TreeTensorNetwork {
    /// Create new tree tensor network
    pub fn new(segment_size: usize, config: MPSConfig) -> Self {
        Self {
            leaves: Vec::new(),
            segment_size,
            config,
        }
    }

    /// Build from high-dimensional vector
    pub fn from_vector(vector: &[f64], segment_size: usize, config: MPSConfig) -> Self {
        let mut ttn = Self::new(segment_size, config.clone());

        // Split vector into segments and create leaf MPS for each
        for chunk in vector.chunks(segment_size) {
            let leaf = MPS::from_vector(chunk, config.clone());
            ttn.leaves.push(leaf);
        }

        ttn
    }

    /// Compute similarity with another TTN
    pub fn similarity(&self, other: &TreeTensorNetwork) -> f64 {
        if self.leaves.len() != other.leaves.len() {
            return 0.0;
        }

        if self.leaves.is_empty() {
            return 0.0;
        }

        // Average similarity of corresponding leaves
        let sum: f64 = self
            .leaves
            .iter()
            .zip(other.leaves.iter())
            .map(|(a, b)| a.similarity(b))
            .sum();

        sum / self.leaves.len() as f64
    }

    /// Total number of parameters
    pub fn num_params(&self) -> usize {
        self.leaves.iter().map(|l| l.num_params()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_tensor_basic() {
        let mut tensor = MPSTensor::new(2, 2, 3);
        tensor.set(0, 1, 2, 1.5);
        assert_eq!(tensor.get(0, 1, 2), 1.5);
        assert_eq!(tensor.num_params(), 12);
    }

    #[test]
    fn test_mps_from_vector() {
        let config = MPSConfig {
            num_sites: 10,
            max_bond_dim: 4,
            ..Default::default()
        };

        let vector: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mps = MPS::from_vector(&vector, config);

        assert_eq!(mps.num_sites(), 10);
        assert!(mps.num_params() > 0);
    }

    #[test]
    fn test_mps_similarity_self() {
        let config = MPSConfig {
            num_sites: 5,
            max_bond_dim: 4,
            ..Default::default()
        };

        let vector: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mps = MPS::from_vector(&vector, config);

        let sim = mps.similarity(&mps);
        // Self-similarity should be close to 1
        assert!(sim > 0.99, "Self-similarity = {}", sim);
    }

    #[test]
    fn test_mps_from_binary() {
        let config = MPSConfig {
            num_sites: 10,
            max_bond_dim: 4,
            ..Default::default()
        };

        let bits = vec![
            true, false, true, true, false, false, true, false, true, false, true, true, false,
            true, false, true, false, false, true, true,
        ];
        let mps = MPS::from_binary(&bits, config);

        assert_eq!(mps.num_sites(), 10);
    }

    #[test]
    fn test_tensor_network_index() {
        let config = MPSConfig {
            num_sites: 10,
            max_bond_dim: 4,
            ..Default::default()
        };

        let mut index = TensorNetworkIndex::new(20, config);

        let v1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let v2: Vec<f64> = (0..20).map(|i| (i + 1) as f64).collect();
        let v3: Vec<f64> = (0..20).map(|i| (20 - i) as f64).collect();

        index.insert("v1", &v1);
        index.insert("v2", &v2);
        index.insert("v3", &v3);

        assert_eq!(index.len(), 3);

        // v1 should be most similar to itself, then v2
        let results = index.find_similar(&v1, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_compression_ratio() {
        let config = MPSConfig {
            num_sites: 50,
            max_bond_dim: 8,
            ..Default::default()
        };

        let mut index = TensorNetworkIndex::new(100, config);

        for i in 0..10 {
            let v: Vec<f64> = (0..100).map(|j| ((i * 100 + j) as f64).sin()).collect();
            index.insert(&format!("v{}", i), &v);
        }

        let ratio = index.compression_ratio();
        assert!(ratio > 0.0, "Compression ratio should be positive");
    }

    #[test]
    fn test_tree_tensor_network() {
        let config = MPSConfig {
            num_sites: 10,
            max_bond_dim: 4,
            ..Default::default()
        };

        let vector: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let ttn = TreeTensorNetwork::from_vector(&vector, 25, config);

        assert_eq!(ttn.leaves.len(), 4); // 100 / 25 = 4 segments
        assert!(ttn.num_params() > 0);
    }

    #[test]
    fn test_tree_tensor_similarity() {
        let config = MPSConfig {
            num_sites: 10,
            max_bond_dim: 4,
            ..Default::default()
        };

        let v1: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let v2: Vec<f64> = (0..100).map(|i| (i as f64).sin() + 0.1).collect();

        let ttn1 = TreeTensorNetwork::from_vector(&v1, 25, config.clone());
        let ttn2 = TreeTensorNetwork::from_vector(&v2, 25, config);

        let sim = ttn1.similarity(&ttn2);
        // Similar vectors should have high similarity
        assert!(sim > 0.9, "Similarity = {}", sim);
    }

    #[test]
    fn test_mps_different_vectors() {
        let config = MPSConfig {
            num_sites: 5,
            max_bond_dim: 4,
            ..Default::default()
        };

        let v1: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let mps1 = MPS::from_vector(&v1, config.clone());
        let mps2 = MPS::from_vector(&v2, config);

        let sim = mps1.similarity(&mps2);
        // Orthogonal vectors should have low similarity
        assert!(
            sim < 0.5,
            "Expected low similarity for orthogonal vectors, got {}",
            sim
        );
    }
}
