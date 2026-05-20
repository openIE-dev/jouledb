//! Neural Layer
//!
//! Pattern recognition and similarity search using vector embeddings.
//! Pure Rust implementation - no WebNN dependency.

use super::{NeurosymbolicError, NeurosymbolicResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Result of pattern matching
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Pattern name/label
    pub name: String,
    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,
    /// Pattern index
    pub index: usize,
}

/// Neural Layer for pattern recognition
///
/// Uses normalized vectors and cosine similarity for matching.
pub struct NeuralLayer {
    /// Stored patterns: name -> (embedding, index)
    patterns: Arc<RwLock<HashMap<String, (Vec<f32>, usize)>>>,
    /// All embeddings for fast search
    embeddings: Arc<RwLock<Vec<Vec<f32>>>>,
    /// Names by index
    names: Arc<RwLock<Vec<String>>>,
    /// Vector dimension
    dimension: usize,
}

impl NeuralLayer {
    /// Create new neural layer with given dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
            embeddings: Arc::new(RwLock::new(Vec::new())),
            names: Arc::new(RwLock::new(Vec::new())),
            dimension,
        }
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get number of patterns
    pub fn len(&self) -> usize {
        self.embeddings.read().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a pattern with a name and embedding
    pub fn add_pattern(&self, name: &str, embedding: &[f32]) -> NeurosymbolicResult<usize> {
        if embedding.len() != self.dimension {
            return Err(NeurosymbolicError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        // Normalize embedding
        let normalized = Self::normalize(embedding);

        let mut patterns = self.patterns.write().unwrap();
        let mut embeddings = self.embeddings.write().unwrap();
        let mut names = self.names.write().unwrap();

        let index = embeddings.len();
        embeddings.push(normalized.clone());
        names.push(name.to_string());
        patterns.insert(name.to_string(), (normalized, index));

        Ok(index)
    }

    /// Get pattern by name
    pub fn get_pattern(&self, name: &str) -> Option<Vec<f32>> {
        let patterns = self.patterns.read().unwrap();
        patterns.get(name).map(|(emb, _)| emb.clone())
    }

    /// Match a query against stored patterns
    pub fn match_pattern(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> NeurosymbolicResult<Vec<PatternMatch>> {
        if query.len() != self.dimension {
            return Err(NeurosymbolicError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let normalized_query = Self::normalize(query);
        let embeddings = self.embeddings.read().unwrap();
        let names = self.names.read().unwrap();

        // Compute similarities
        let mut matches: Vec<PatternMatch> = embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| PatternMatch {
                name: names[idx].clone(),
                similarity: Self::cosine_similarity(&normalized_query, emb),
                index: idx,
            })
            .collect();

        // Sort by similarity descending
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        matches.truncate(top_k);

        Ok(matches)
    }

    /// Find patterns above similarity threshold
    pub fn match_threshold(
        &self,
        query: &[f32],
        threshold: f32,
    ) -> NeurosymbolicResult<Vec<PatternMatch>> {
        if query.len() != self.dimension {
            return Err(NeurosymbolicError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let normalized_query = Self::normalize(query);
        let embeddings = self.embeddings.read().unwrap();
        let names = self.names.read().unwrap();

        let matches: Vec<PatternMatch> = embeddings
            .iter()
            .enumerate()
            .filter_map(|(idx, emb)| {
                let sim = Self::cosine_similarity(&normalized_query, emb);
                if sim >= threshold {
                    Some(PatternMatch {
                        name: names[idx].clone(),
                        similarity: sim,
                        index: idx,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(matches)
    }

    /// Generate embedding from data (simple hash-based approach)
    pub fn generate_embedding(&self, data: &[u8]) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];

        // Simple hash-based embedding
        for (i, &byte) in data.iter().enumerate() {
            let idx = (i * 31 + byte as usize) % self.dimension;
            embedding[idx] += (byte as f32 - 128.0) / 128.0;
        }

        Self::normalize(&embedding)
    }

    /// Normalize vector to unit length
    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    /// Compute cosine similarity between two normalized vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

impl Clone for NeuralLayer {
    fn clone(&self) -> Self {
        Self {
            patterns: Arc::new(RwLock::new(self.patterns.read().unwrap().clone())),
            embeddings: Arc::new(RwLock::new(self.embeddings.read().unwrap().clone())),
            names: Arc::new(RwLock::new(self.names.read().unwrap().clone())),
            dimension: self.dimension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_layer_creation() {
        let layer = NeuralLayer::new(100);
        assert_eq!(layer.dimension(), 100);
        assert!(layer.is_empty());
    }

    #[test]
    fn test_add_pattern() {
        let layer = NeuralLayer::new(4);
        let embedding = vec![1.0, 0.0, 0.0, 0.0];

        let idx = layer.add_pattern("test", &embedding).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(layer.len(), 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let layer = NeuralLayer::new(4);
        let embedding = vec![1.0, 0.0, 0.0]; // Wrong dimension

        let result = layer.add_pattern("test", &embedding);
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_matching() {
        let layer = NeuralLayer::new(4);

        layer.add_pattern("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        layer.add_pattern("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        layer.add_pattern("c", &[0.9, 0.1, 0.0, 0.0]).unwrap(); // Similar to a

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let matches = layer.match_pattern(&query, 3).unwrap();

        assert_eq!(matches.len(), 3);
        // "a" should be first (exact match)
        assert_eq!(matches[0].name, "a");
        assert!(matches[0].similarity > 0.99);
        // "c" should be second (similar)
        assert_eq!(matches[1].name, "c");
    }

    #[test]
    fn test_threshold_matching() {
        let layer = NeuralLayer::new(4);

        layer.add_pattern("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        layer.add_pattern("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let matches = layer.match_threshold(&query, 0.5).unwrap();

        // Only "a" should match above 0.5
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "a");
    }

    #[test]
    fn test_generate_embedding() {
        let layer = NeuralLayer::new(10);

        let emb1 = layer.generate_embedding(b"hello");
        let emb2 = layer.generate_embedding(b"hello");
        let emb3 = layer.generate_embedding(b"world");

        // Same input should give same embedding
        assert_eq!(emb1, emb2);
        // Different input should give different embedding
        assert_ne!(emb1, emb3);
        // Should be normalized
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_get_pattern() {
        let layer = NeuralLayer::new(4);
        layer.add_pattern("test", &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let pattern = layer.get_pattern("test");
        assert!(pattern.is_some());

        let pattern = layer.get_pattern("nonexistent");
        assert!(pattern.is_none());
    }
}
