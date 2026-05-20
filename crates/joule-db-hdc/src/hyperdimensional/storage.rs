//! Hyperdimensional Storage for VSA vectors

use super::vector::{HDError, HyperVector};
use std::sync::{Arc, RwLock};

/// Result of similarity search
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    /// Index in storage
    pub index: usize,
    /// Cosine similarity score
    pub similarity: f32,
}

/// Storage for hyperdimensional vectors with similarity search
pub struct HyperdimensionalStorage {
    vectors: Arc<RwLock<Vec<HyperVector>>>,
    dimension: usize,
}

impl HyperdimensionalStorage {
    /// Create new storage with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Arc::new(RwLock::new(Vec::new())),
            dimension,
        }
    }

    /// Add a vector to storage
    pub fn add_vector(&self, vector: HyperVector) -> Result<usize, HDError> {
        if vector.dimension() != self.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.dimension(),
            });
        }

        let mut vectors = self.vectors.write().unwrap();
        let index = vectors.len();
        vectors.push(vector);
        Ok(index)
    }

    /// Add multiple vectors
    pub fn add_vectors(&self, new_vectors: Vec<HyperVector>) -> Result<Vec<usize>, HDError> {
        for v in &new_vectors {
            if v.dimension() != self.dimension {
                return Err(HDError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.dimension(),
                });
            }
        }

        let mut vectors = self.vectors.write().unwrap();
        let start_index = vectors.len();
        let count = new_vectors.len();
        vectors.extend(new_vectors);
        Ok((start_index..start_index + count).collect())
    }

    /// Get vector by index
    pub fn get(&self, index: usize) -> Option<HyperVector> {
        self.vectors.read().unwrap().get(index).cloned()
    }

    /// Search for top-k similar vectors
    pub fn similarity_search(
        &self,
        query: &HyperVector,
        top_k: usize,
    ) -> Result<Vec<SimilarityMatch>, HDError> {
        if query.dimension() != self.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }

        let vectors = self.vectors.read().unwrap();
        let mut results: Vec<SimilarityMatch> = Vec::with_capacity(vectors.len());

        for (index, v) in vectors.iter().enumerate() {
            let similarity = query.similarity(v)?;
            results.push(SimilarityMatch { index, similarity });
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top-k
        results.truncate(top_k);
        Ok(results)
    }

    /// Find vectors above similarity threshold
    pub fn threshold_search(
        &self,
        query: &HyperVector,
        threshold: f32,
    ) -> Result<Vec<SimilarityMatch>, HDError> {
        if query.dimension() != self.dimension {
            return Err(HDError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }

        let vectors = self.vectors.read().unwrap();
        let mut results: Vec<SimilarityMatch> = Vec::new();

        for (index, v) in vectors.iter().enumerate() {
            let similarity = query.similarity(v)?;
            if similarity >= threshold {
                results.push(SimilarityMatch { index, similarity });
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Get number of stored vectors
    pub fn len(&self) -> usize {
        self.vectors.read().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Clear all vectors
    pub fn clear(&self) {
        self.vectors.write().unwrap().clear();
    }

    /// Remove vector by index (expensive - shifts subsequent indices)
    pub fn remove(&self, index: usize) -> Option<HyperVector> {
        let mut vectors = self.vectors.write().unwrap();
        if index < vectors.len() {
            Some(vectors.remove(index))
        } else {
            None
        }
    }

    /// Get statistics
    pub fn stats(&self) -> StorageStats {
        StorageStats {
            count: self.len(),
            dimension: self.dimension,
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of vectors
    pub count: usize,
    /// Vector dimension
    pub dimension: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let storage = HyperdimensionalStorage::new(1000);
        assert_eq!(storage.dimension(), 1000);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_add_vector() {
        let storage = HyperdimensionalStorage::new(1000);
        let v = HyperVector::random(1000, 42);

        let idx = storage.add_vector(v).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn test_add_wrong_dimension() {
        let storage = HyperdimensionalStorage::new(1000);
        let v = HyperVector::random(500, 42);

        assert!(matches!(
            storage.add_vector(v),
            Err(HDError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_similarity_search() {
        let storage = HyperdimensionalStorage::new(1000);

        // Add some vectors
        for i in 0..10 {
            let v = HyperVector::random(1000, i);
            storage.add_vector(v).unwrap();
        }

        // Search for similar to first vector
        let query = HyperVector::random(1000, 0); // Same seed as first
        let results = storage.similarity_search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be most similar (exact match)
        assert!(results[0].similarity > 0.99);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_threshold_search() {
        let storage = HyperdimensionalStorage::new(1000);

        // Add vectors
        for i in 0..5 {
            let v = HyperVector::random(1000, i);
            storage.add_vector(v).unwrap();
        }

        let query = HyperVector::random(1000, 0);
        let results = storage.threshold_search(&query, 0.9).unwrap();

        // Should find at least the exact match
        assert!(!results.is_empty());
        assert!(results[0].similarity >= 0.9);
    }

    #[test]
    fn test_get_vector() {
        let storage = HyperdimensionalStorage::new(1000);
        let v = HyperVector::random(1000, 42);
        storage.add_vector(v.clone()).unwrap();

        let retrieved = storage.get(0).unwrap();
        assert!(v.similarity(&retrieved).unwrap() > 0.99);

        assert!(storage.get(999).is_none());
    }

    #[test]
    fn test_remove() {
        let storage = HyperdimensionalStorage::new(1000);

        for i in 0..3 {
            storage.add_vector(HyperVector::random(1000, i)).unwrap();
        }

        assert_eq!(storage.len(), 3);
        storage.remove(1);
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn test_clear() {
        let storage = HyperdimensionalStorage::new(1000);

        for i in 0..5 {
            storage.add_vector(HyperVector::random(1000, i)).unwrap();
        }

        assert_eq!(storage.len(), 5);
        storage.clear();
        assert!(storage.is_empty());
    }
}
