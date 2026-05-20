//! Holographic Associative Memory Storage
//!
//! Brain-inspired storage using holographic principles.

use super::complex::Complex;
use super::interference::{InterferenceError, InterferencePattern};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Holographic storage errors
#[derive(Error, Debug, Clone)]
pub enum HolographicError {
    /// Data dimension mismatch
    #[error("Data dimension mismatch: expected {expected} (real/imag pairs), got {actual}")]
    DimensionMismatch {
        /// Expected dimension (number of complex pairs)
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// Pattern not found
    #[error("Pattern '{0}' not found")]
    PatternNotFound(String),

    /// Reference not found
    #[error("Reference '{0}' not found")]
    ReferenceNotFound(String),

    /// Interference error
    #[error("Interference error: {0}")]
    Interference(#[from] InterferenceError),

    /// Lock poisoned
    #[error("Lock poisoned")]
    LockPoisoned,
}

/// Statistics about holographic storage
#[derive(Debug, Clone)]
pub struct HolographicStats {
    /// Number of stored patterns
    pub num_patterns: usize,
    /// Number of reference patterns
    pub num_references: usize,
    /// Storage dimension
    pub dimension: usize,
    /// Query cache size
    pub cache_size: usize,
}

/// Result of similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Pattern name
    pub name: String,
    /// Similarity score
    pub similarity: f32,
}

/// Holographic Associative Memory Storage
pub struct HolographicStorage {
    patterns: Arc<RwLock<HashMap<String, InterferencePattern>>>,
    reference_patterns: Arc<RwLock<HashMap<String, Vec<Complex>>>>,
    dimension: usize,
    query_cache: Arc<RwLock<HashMap<String, Vec<Complex>>>>,
    seed: u64,
}

impl HolographicStorage {
    /// Create new holographic storage
    pub fn new(dimension: usize) -> Self {
        Self::with_seed(dimension, 12345)
    }

    /// Create new holographic storage with custom seed
    pub fn with_seed(dimension: usize, seed: u64) -> Self {
        Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
            reference_patterns: Arc::new(RwLock::new(HashMap::new())),
            dimension,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            seed,
        }
    }

    /// Generate random reference pattern (deterministic based on name)
    pub fn generate_reference(&self, name: &str) -> Vec<f32> {
        // Generate deterministic seed from name
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&name, &mut hasher);
        std::hash::Hash::hash(&self.seed, &mut hasher);
        let seed = std::hash::Hasher::finish(&hasher);

        let mut reference = Vec::with_capacity(self.dimension);
        let mut rng = seed;

        for _ in 0..self.dimension {
            // LCG for deterministic random
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Random angle on unit circle
            let angle = (rng as f64 / u64::MAX as f64) as f32 * 2.0 * PI;
            reference.push(Complex::unit(angle));
        }

        // Store reference
        {
            let mut refs = self.reference_patterns.write().unwrap();
            refs.insert(name.to_string(), reference.clone());
        }

        // Convert to flat array
        let mut result = Vec::with_capacity(self.dimension * 2);
        for c in reference {
            result.push(c.real);
            result.push(c.imag);
        }
        result
    }

    /// Store a pattern holographically
    ///
    /// Data should be a flat array of real/imag pairs (length = dimension * 2)
    pub fn store_pattern(&self, name: String, data: &[f32]) -> Result<(), HolographicError> {
        if data.len() != self.dimension * 2 {
            return Err(HolographicError::DimensionMismatch {
                expected: self.dimension * 2,
                actual: data.len(),
            });
        }

        // Convert data to complex vector
        let mut object = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            object.push(Complex::new(data[i * 2], data[i * 2 + 1]));
        }

        // Get or create reference pattern
        let reference = {
            let refs = self
                .reference_patterns
                .read()
                .map_err(|_| HolographicError::LockPoisoned)?;
            if let Some(ref_pattern) = refs.get(&name) {
                ref_pattern.clone()
            } else {
                // Generate new reference
                drop(refs);
                let ref_flat = self.generate_reference(&name);
                let mut ref_pattern = Vec::new();
                for i in 0..self.dimension {
                    ref_pattern.push(Complex::new(ref_flat[i * 2], ref_flat[i * 2 + 1]));
                }
                ref_pattern
            }
        };

        // Create interference pattern
        let interference = InterferencePattern::from_beams(&reference, &object)?;

        // Store
        {
            let mut patterns = self
                .patterns
                .write()
                .map_err(|_| HolographicError::LockPoisoned)?;
            patterns.insert(name, interference);
        }

        Ok(())
    }

    /// Recall pattern from partial query
    pub fn recall_pattern(
        &self,
        name: &str,
        partial_query: &[f32],
    ) -> Result<Vec<f32>, HolographicError> {
        // Get stored pattern
        let pattern = {
            let patterns = self
                .patterns
                .read()
                .map_err(|_| HolographicError::LockPoisoned)?;
            patterns
                .get(name)
                .cloned()
                .ok_or_else(|| HolographicError::PatternNotFound(name.to_string()))?
        };

        // Get reference
        let reference = {
            let refs = self
                .reference_patterns
                .read()
                .map_err(|_| HolographicError::LockPoisoned)?;
            refs.get(name)
                .cloned()
                .ok_or_else(|| HolographicError::ReferenceNotFound(name.to_string()))?
        };

        // Convert partial query to complex
        let mut query_complex = Vec::new();
        let query_len = (partial_query.len() / 2).min(self.dimension);
        for i in 0..query_len {
            query_complex.push(Complex::new(partial_query[i * 2], partial_query[i * 2 + 1]));
        }
        // Pad with zeros if needed
        while query_complex.len() < self.dimension {
            query_complex.push(Complex::zero());
        }

        // Reconstruct
        let reconstructed = pattern.reconstruct(&reference)?;

        // Convert to flat array
        let mut result = Vec::with_capacity(self.dimension * 2);
        for c in reconstructed {
            result.push(c.real);
            result.push(c.imag);
        }

        Ok(result)
    }

    /// Associative search (find similar patterns)
    pub fn associative_search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, HolographicError> {
        if query.len() != self.dimension * 2 {
            return Err(HolographicError::DimensionMismatch {
                expected: self.dimension * 2,
                actual: query.len(),
            });
        }

        // Convert query to complex
        let mut query_complex = Vec::new();
        for i in 0..self.dimension {
            query_complex.push(Complex::new(query[i * 2], query[i * 2 + 1]));
        }

        // Compute similarities
        let patterns = self
            .patterns
            .read()
            .map_err(|_| HolographicError::LockPoisoned)?;
        let refs = self
            .reference_patterns
            .read()
            .map_err(|_| HolographicError::LockPoisoned)?;
        let mut similarities: Vec<SimilarityResult> = Vec::new();

        for (name, pattern) in patterns.iter() {
            if let Some(reference) = refs.get(name) {
                // Reconstruct pattern
                if let Ok(reconstructed) = pattern.reconstruct(reference) {
                    // Compute similarity (dot product in complex space)
                    let mut similarity = 0.0f32;
                    for i in 0..self.dimension {
                        let dot = reconstructed[i] * query_complex[i].conjugate();
                        similarity += dot.real;
                    }
                    similarities.push(SimilarityResult {
                        name: name.clone(),
                        similarity,
                    });
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top-k
        Ok(similarities.into_iter().take(top_k).collect())
    }

    /// Get storage statistics
    pub fn stats(&self) -> HolographicStats {
        let patterns = self.patterns.read().unwrap();
        let refs = self.reference_patterns.read().unwrap();
        let cache = self.query_cache.read().unwrap();

        HolographicStats {
            num_patterns: patterns.len(),
            num_references: refs.len(),
            dimension: self.dimension,
            cache_size: cache.len(),
        }
    }

    /// Clear query cache
    pub fn clear_cache(&self) {
        self.query_cache.write().unwrap().clear();
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if pattern exists
    pub fn contains(&self, name: &str) -> bool {
        self.patterns.read().unwrap().contains_key(name)
    }

    /// Remove a pattern
    pub fn remove(&self, name: &str) -> bool {
        let mut patterns = self.patterns.write().unwrap();
        let mut refs = self.reference_patterns.write().unwrap();
        let p = patterns.remove(name).is_some();
        refs.remove(name);
        p
    }

    /// List all pattern names
    pub fn list_patterns(&self) -> Vec<String> {
        self.patterns.read().unwrap().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let storage = HolographicStorage::new(256);
        assert_eq!(storage.dimension(), 256);
        let stats = storage.stats();
        assert_eq!(stats.num_patterns, 0);
    }

    #[test]
    fn test_generate_reference() {
        let storage = HolographicStorage::new(100);
        let ref1 = storage.generate_reference("test");
        let ref2 = storage.generate_reference("test");

        // Same name should give same reference (after stored)
        assert_eq!(ref1.len(), 200); // 100 complex = 200 floats
    }

    #[test]
    fn test_store_pattern() {
        let storage = HolographicStorage::new(50);
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        storage.store_pattern("pattern1".into(), &data).unwrap();

        assert!(storage.contains("pattern1"));
        let stats = storage.stats();
        assert_eq!(stats.num_patterns, 1);
    }

    #[test]
    fn test_store_dimension_mismatch() {
        let storage = HolographicStorage::new(50);
        let data: Vec<f32> = vec![1.0; 50]; // Wrong size, should be 100

        let result = storage.store_pattern("test".into(), &data);
        assert!(matches!(
            result,
            Err(HolographicError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_recall_pattern() {
        let storage = HolographicStorage::new(50);
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        storage.store_pattern("pattern1".into(), &data).unwrap();

        // Recall with partial query
        let partial = &data[..50];
        let recalled = storage.recall_pattern("pattern1", partial).unwrap();

        assert_eq!(recalled.len(), 100);
    }

    #[test]
    fn test_recall_not_found() {
        let storage = HolographicStorage::new(50);
        let result = storage.recall_pattern("nonexistent", &[]);
        assert!(matches!(result, Err(HolographicError::PatternNotFound(_))));
    }

    #[test]
    fn test_associative_search() {
        let storage = HolographicStorage::new(50);

        // Store a few patterns
        for i in 0..5 {
            let data: Vec<f32> = (0..100).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
            storage
                .store_pattern(format!("pattern{}", i), &data)
                .unwrap();
        }

        // Search with query
        let query: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let results = storage.associative_search(&query, 3).unwrap();

        assert!(results.len() <= 3);
        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(results[i].similarity <= results[i - 1].similarity);
        }
    }

    #[test]
    fn test_remove_pattern() {
        let storage = HolographicStorage::new(50);
        let data: Vec<f32> = vec![1.0; 100];

        storage.store_pattern("test".into(), &data).unwrap();
        assert!(storage.contains("test"));

        assert!(storage.remove("test"));
        assert!(!storage.contains("test"));

        assert!(!storage.remove("nonexistent"));
    }

    #[test]
    fn test_list_patterns() {
        let storage = HolographicStorage::new(50);
        let data: Vec<f32> = vec![1.0; 100];

        storage.store_pattern("a".into(), &data).unwrap();
        storage.store_pattern("b".into(), &data).unwrap();

        let names = storage.list_patterns();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }
}
