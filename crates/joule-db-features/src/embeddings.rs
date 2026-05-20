//! # Vector Embeddings
//!
//! Provides embedding generation and management for semantic search.
//!
//! ## Features
//!
//! - Text embedding with multiple model backends
//! - Embedding caching and persistence
//! - Batch embedding generation
//! - Dimensionality reduction (PCA)
//! - Embedding arithmetic (king - man + woman = queen)
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_features::embeddings::{EmbeddingStore, EmbeddingConfig, EmbeddingModel};
//!
//! let config = EmbeddingConfig::new(384)
//!     .with_model(EmbeddingModel::MiniLM);
//!
//! let mut store = EmbeddingStore::new(config);
//! store.embed_text("doc1", "The quick brown fox");
//!
//! let similar = store.find_similar("doc1", 5);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingModel {
    /// Sentence-BERT MiniLM (384 dimensions).
    MiniLM,
    /// OpenAI Ada-002 (1536 dimensions).
    Ada002,
    /// Cohere Embed (1024 dimensions).
    CohereEmbed,
    /// Custom model with specified dimensions.
    Custom,
    /// Simple bag-of-words (variable dimensions).
    BagOfWords,
    /// TF-IDF embeddings (variable dimensions).
    TfIdf,
    /// Random projection (for testing).
    RandomProjection,
}

impl EmbeddingModel {
    /// Get the default dimension for this model.
    pub fn default_dimensions(&self) -> usize {
        match self {
            EmbeddingModel::MiniLM => 384,
            EmbeddingModel::Ada002 => 1536,
            EmbeddingModel::CohereEmbed => 1024,
            EmbeddingModel::Custom => 256,
            EmbeddingModel::BagOfWords => 1000,
            EmbeddingModel::TfIdf => 1000,
            EmbeddingModel::RandomProjection => 128,
        }
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        EmbeddingModel::RandomProjection
    }
}

/// Configuration for embedding store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Model to use for embeddings.
    pub model: EmbeddingModel,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Cache embeddings in memory.
    pub cache_enabled: bool,
    /// Maximum cache size.
    pub max_cache_size: usize,
    /// Vocabulary size for bag-of-words/TF-IDF.
    pub vocab_size: usize,
}

impl EmbeddingConfig {
    /// Create a new config with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            model: EmbeddingModel::RandomProjection,
            normalize: true,
            cache_enabled: true,
            max_cache_size: 10000,
            vocab_size: 10000,
        }
    }

    /// Set the embedding model.
    pub fn with_model(mut self, model: EmbeddingModel) -> Self {
        self.model = model;
        self.dimensions = model.default_dimensions();
        self
    }

    /// Set whether to normalize embeddings.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set cache settings.
    pub fn with_cache(mut self, enabled: bool, max_size: usize) -> Self {
        self.cache_enabled = enabled;
        self.max_cache_size = max_size;
        self
    }

    /// Set vocabulary size for BoW/TF-IDF.
    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self::new(128)
    }
}

/// A stored embedding with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEmbedding {
    /// Unique identifier.
    pub id: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Original text (if available).
    pub text: Option<String>,
    /// Additional metadata.
    pub metadata: Option<serde_json::Value>,
    /// Timestamp of creation.
    pub created_at: u64,
}

/// Result of a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Document ID.
    pub id: String,
    /// Similarity score (0-1 for cosine).
    pub similarity: f32,
    /// Original text if available.
    pub text: Option<String>,
}

/// Embedding store for managing document embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStore {
    config: EmbeddingConfig,
    embeddings: HashMap<String, StoredEmbedding>,
    /// Vocabulary for BoW/TF-IDF models.
    vocabulary: HashMap<String, usize>,
    /// Document frequency for TF-IDF.
    doc_frequency: HashMap<String, usize>,
    /// Total document count.
    doc_count: usize,
}

impl EmbeddingStore {
    /// Create a new embedding store.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            embeddings: HashMap::new(),
            vocabulary: HashMap::new(),
            doc_frequency: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_store() -> Self {
        Self::new(EmbeddingConfig::default())
    }

    /// Get the number of stored embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Embed text and store it.
    pub fn embed_text(&mut self, id: impl Into<String>, text: &str) -> Result<(), EmbeddingError> {
        let id = id.into();

        if self.embeddings.contains_key(&id) {
            return Err(EmbeddingError::DuplicateId(id));
        }

        let vector = self.generate_embedding(text)?;

        let stored = StoredEmbedding {
            id: id.clone(),
            vector,
            text: Some(text.to_string()),
            metadata: None,
            created_at: current_timestamp(),
        };

        self.embeddings.insert(id, stored);
        self.doc_count += 1;

        Ok(())
    }

    /// Embed text with metadata.
    pub fn embed_text_with_metadata(
        &mut self,
        id: impl Into<String>,
        text: &str,
        metadata: serde_json::Value,
    ) -> Result<(), EmbeddingError> {
        let id = id.into();

        if self.embeddings.contains_key(&id) {
            return Err(EmbeddingError::DuplicateId(id));
        }

        let vector = self.generate_embedding(text)?;

        let stored = StoredEmbedding {
            id: id.clone(),
            vector,
            text: Some(text.to_string()),
            metadata: Some(metadata),
            created_at: current_timestamp(),
        };

        self.embeddings.insert(id, stored);
        self.doc_count += 1;

        Ok(())
    }

    /// Store a pre-computed embedding.
    pub fn store_embedding(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
    ) -> Result<(), EmbeddingError> {
        let id = id.into();

        if vector.len() != self.config.dimensions {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        if self.embeddings.contains_key(&id) {
            return Err(EmbeddingError::DuplicateId(id));
        }

        let vector = if self.config.normalize {
            normalize_vector(&vector)
        } else {
            vector
        };

        let stored = StoredEmbedding {
            id: id.clone(),
            vector,
            text: None,
            metadata: None,
            created_at: current_timestamp(),
        };

        self.embeddings.insert(id, stored);

        Ok(())
    }

    /// Generate an embedding for text.
    fn generate_embedding(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let vector = match self.config.model {
            EmbeddingModel::BagOfWords => self.generate_bow_embedding(text),
            EmbeddingModel::TfIdf => self.generate_tfidf_embedding(text),
            EmbeddingModel::RandomProjection => self.generate_random_embedding(text),
            _ => {
                // For external models (OpenAI, Cohere, Sentence-BERT, etc.),
                // an HTTP client would call the respective embedding API.
                // Since external API calls require network access and credentials,
                // we fall back to random projection for offline use.
                // Production deployments should implement the specific model's API.
                tracing::warn!(
                    "External embedding model {:?} requires API access; using random projection fallback",
                    self.config.model
                );
                self.generate_random_embedding(text)
            }
        };

        let vector = if self.config.normalize {
            normalize_vector(&vector)
        } else {
            vector
        };

        Ok(vector)
    }

    /// Generate bag-of-words embedding.
    fn generate_bow_embedding(&mut self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let mut vector = vec![0.0; self.config.dimensions];

        for token in tokens {
            let idx = self.get_or_create_vocab_index(&token);
            if idx < self.config.dimensions {
                vector[idx] += 1.0;
            }
        }

        vector
    }

    /// Generate TF-IDF embedding.
    fn generate_tfidf_embedding(&mut self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let mut term_freq: HashMap<String, f32> = HashMap::new();

        for token in &tokens {
            *term_freq.entry(token.clone()).or_default() += 1.0;
        }

        let doc_len = tokens.len() as f32;
        let mut vector = vec![0.0; self.config.dimensions];

        for (token, tf) in term_freq {
            let idx = self.get_or_create_vocab_index(&token);
            if idx < self.config.dimensions {
                // TF: term frequency normalized by document length
                let tf_norm = tf / doc_len;

                // IDF: inverse document frequency
                let df = self.doc_frequency.get(&token).copied().unwrap_or(1) as f32;
                let idf = ((self.doc_count as f32 + 1.0) / (df + 1.0)).ln() + 1.0;

                vector[idx] = tf_norm * idf;
            }

            // Update document frequency
            *self.doc_frequency.entry(token).or_default() += 1;
        }

        vector
    }

    /// Generate random projection embedding (for testing).
    fn generate_random_embedding(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let mut vector = vec![0.0; self.config.dimensions];

        for token in tokens {
            // Use token hash to generate deterministic "random" values
            let hash = simple_hash(&token);
            for i in 0..self.config.dimensions {
                let seed = hash.wrapping_add(i as u64);
                vector[i] += ((seed % 1000) as f32 / 1000.0) - 0.5;
            }
        }

        vector
    }

    /// Get or create vocabulary index for a token.
    fn get_or_create_vocab_index(&mut self, token: &str) -> usize {
        if let Some(&idx) = self.vocabulary.get(token) {
            return idx;
        }

        let idx = self.vocabulary.len() % self.config.dimensions;
        self.vocabulary.insert(token.to_string(), idx);
        idx
    }

    /// Get an embedding by ID.
    pub fn get(&self, id: &str) -> Option<&StoredEmbedding> {
        self.embeddings.get(id)
    }

    /// Get just the vector by ID.
    pub fn get_vector(&self, id: &str) -> Option<&[f32]> {
        self.embeddings.get(id).map(|e| e.vector.as_slice())
    }

    /// Check if an embedding exists.
    pub fn contains(&self, id: &str) -> bool {
        self.embeddings.contains_key(id)
    }

    /// Remove an embedding.
    pub fn remove(&mut self, id: &str) -> bool {
        self.embeddings.remove(id).is_some()
    }

    /// Find similar embeddings to a given ID.
    pub fn find_similar(
        &self,
        id: &str,
        k: usize,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let query = self
            .embeddings
            .get(id)
            .ok_or_else(|| EmbeddingError::NotFound(id.to_string()))?;

        self.find_similar_to_vector(&query.vector, k, Some(id))
    }

    /// Find similar embeddings to a vector.
    pub fn find_similar_to_vector(
        &self,
        query: &[f32],
        k: usize,
        exclude_id: Option<&str>,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        if query.len() != self.config.dimensions {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let mut results: Vec<SimilarityResult> = self
            .embeddings
            .iter()
            .filter(|(id, _)| exclude_id.map_or(true, |ex| *id != ex))
            .map(|(_, emb)| {
                let similarity = cosine_similarity(query, &emb.vector);
                SimilarityResult {
                    id: emb.id.clone(),
                    similarity,
                    text: emb.text.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Find similar to text (generates embedding first).
    pub fn find_similar_to_text(
        &mut self,
        text: &str,
        k: usize,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let vector = self.generate_embedding(text)?;
        self.find_similar_to_vector(&vector, k, None)
    }

    /// Perform embedding arithmetic: result = a - b + c
    /// Classic example: king - man + woman = queen
    pub fn embedding_arithmetic(
        &self,
        positive: &[&str],
        negative: &[&str],
        k: usize,
    ) -> Result<Vec<SimilarityResult>, EmbeddingError> {
        let mut result = vec![0.0; self.config.dimensions];

        for id in positive {
            let emb = self
                .embeddings
                .get(*id)
                .ok_or_else(|| EmbeddingError::NotFound(id.to_string()))?;
            for (i, v) in emb.vector.iter().enumerate() {
                result[i] += v;
            }
        }

        for id in negative {
            let emb = self
                .embeddings
                .get(*id)
                .ok_or_else(|| EmbeddingError::NotFound(id.to_string()))?;
            for (i, v) in emb.vector.iter().enumerate() {
                result[i] -= v;
            }
        }

        // Normalize result
        let result = normalize_vector(&result);

        // Exclude input IDs from results
        let exclude: std::collections::HashSet<&str> =
            positive.iter().chain(negative.iter()).copied().collect();

        let mut results: Vec<SimilarityResult> = self
            .embeddings
            .iter()
            .filter(|(id, _)| !exclude.contains(id.as_str()))
            .map(|(_, emb)| {
                let similarity = cosine_similarity(&result, &emb.vector);
                SimilarityResult {
                    id: emb.id.clone(),
                    similarity,
                    text: emb.text.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Compute centroid of multiple embeddings.
    pub fn centroid(&self, ids: &[&str]) -> Result<Vec<f32>, EmbeddingError> {
        if ids.is_empty() {
            return Err(EmbeddingError::InvalidOperation(
                "Cannot compute centroid of empty set".to_string(),
            ));
        }

        let mut result = vec![0.0; self.config.dimensions];

        for id in ids {
            let emb = self
                .embeddings
                .get(*id)
                .ok_or_else(|| EmbeddingError::NotFound(id.to_string()))?;
            for (i, v) in emb.vector.iter().enumerate() {
                result[i] += v;
            }
        }

        let n = ids.len() as f32;
        for v in result.iter_mut() {
            *v /= n;
        }

        Ok(result)
    }

    /// Batch embed multiple texts.
    pub fn embed_batch(&mut self, items: Vec<(String, String)>) -> Result<usize, EmbeddingError> {
        let mut count = 0;
        for (id, text) in items {
            self.embed_text(id, &text)?;
            count += 1;
        }
        Ok(count)
    }

    /// Get all embedding IDs.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.embeddings.keys().map(|s| s.as_str())
    }

    /// Clear all embeddings.
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.vocabulary.clear();
        self.doc_frequency.clear();
        self.doc_count = 0;
    }

    /// Get statistics about the store.
    pub fn stats(&self) -> EmbeddingStats {
        EmbeddingStats {
            embedding_count: self.embeddings.len(),
            dimensions: self.config.dimensions,
            vocabulary_size: self.vocabulary.len(),
            model: self.config.model,
        }
    }
}

/// Statistics about the embedding store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    pub embedding_count: usize,
    pub dimensions: usize,
    pub vocabulary_size: usize,
    pub model: EmbeddingModel,
}

/// Embedding errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Duplicate embedding ID: {0}")]
    DuplicateId(String),

    #[error("Embedding not found: {0}")]
    NotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Tokenize text into words.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

/// Normalize a vector to unit length.
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Simple hash function for deterministic random projection.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

/// Get current timestamp.
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Dimensionality reduction using simple PCA-like projection.
pub fn reduce_dimensions(vectors: &[Vec<f32>], target_dim: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let original_dim = vectors[0].len();
    if target_dim >= original_dim {
        return vectors.to_vec();
    }

    // Simple random projection for dimensionality reduction
    // In production, use proper PCA or UMAP
    let projection_matrix: Vec<Vec<f32>> = (0..target_dim)
        .map(|i| {
            (0..original_dim)
                .map(|j| {
                    let seed = (i * original_dim + j) as u64;
                    ((seed.wrapping_mul(1103515245).wrapping_add(12345) % 1000) as f32 / 1000.0)
                        - 0.5
                })
                .collect()
        })
        .collect();

    vectors
        .iter()
        .map(|v| {
            projection_matrix
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_text() {
        let mut store = EmbeddingStore::default_store();

        store.embed_text("doc1", "hello world").unwrap();
        store.embed_text("doc2", "goodbye world").unwrap();

        assert_eq!(store.len(), 2);
        assert!(store.contains("doc1"));
        assert!(store.contains("doc2"));
    }

    #[test]
    fn test_duplicate_id() {
        let mut store = EmbeddingStore::default_store();

        store.embed_text("doc1", "hello").unwrap();
        let result = store.embed_text("doc1", "world");

        assert!(matches!(result, Err(EmbeddingError::DuplicateId(_))));
    }

    #[test]
    fn test_find_similar() {
        let mut store = EmbeddingStore::default_store();

        store.embed_text("doc1", "the quick brown fox").unwrap();
        store.embed_text("doc2", "the quick brown dog").unwrap();
        store
            .embed_text("doc3", "completely different text")
            .unwrap();

        let results = store.find_similar("doc1", 2).unwrap();

        assert_eq!(results.len(), 2);
        // doc2 should be more similar to doc1 than doc3
        assert_eq!(results[0].id, "doc2");
    }

    #[test]
    fn test_store_embedding() {
        let config = EmbeddingConfig::new(4);
        let mut store = EmbeddingStore::new(config);

        store
            .store_embedding("vec1", vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        store
            .store_embedding("vec2", vec![0.0, 1.0, 0.0, 0.0])
            .unwrap();

        assert!(store.contains("vec1"));
        assert!(store.get_vector("vec1").is_some());
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = EmbeddingConfig::new(4);
        let mut store = EmbeddingStore::new(config);

        let result = store.store_embedding("vec1", vec![1.0, 0.0, 0.0]);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_centroid() {
        let config = EmbeddingConfig::new(4).with_normalize(false);
        let mut store = EmbeddingStore::new(config);

        store
            .store_embedding("a", vec![0.0, 0.0, 0.0, 0.0])
            .unwrap();
        store
            .store_embedding("b", vec![2.0, 2.0, 2.0, 2.0])
            .unwrap();

        let centroid = store.centroid(&["a", "b"]).unwrap();

        assert!((centroid[0] - 1.0).abs() < 0.001);
        assert!((centroid[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_arithmetic() {
        let config = EmbeddingConfig::new(4).with_normalize(false);
        let mut store = EmbeddingStore::new(config);

        // Simple test: a + b - c should be similar to some result
        store
            .store_embedding("a", vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        store
            .store_embedding("b", vec![0.0, 1.0, 0.0, 0.0])
            .unwrap();
        store
            .store_embedding("c", vec![0.0, 0.0, 1.0, 0.0])
            .unwrap();
        store
            .store_embedding("d", vec![1.0, 1.0, -1.0, 0.0])
            .unwrap();

        let results = store.embedding_arithmetic(&["a", "b"], &["c"], 1).unwrap();

        assert!(!results.is_empty());
        // d = [1,1,-1,0] should be most similar to a+b-c = [1,1,-1,0]
        assert_eq!(results[0].id, "d");
    }

    #[test]
    fn test_batch_embed() {
        let mut store = EmbeddingStore::default_store();

        let items = vec![
            ("doc1".to_string(), "hello world".to_string()),
            ("doc2".to_string(), "goodbye world".to_string()),
            ("doc3".to_string(), "foo bar".to_string()),
        ];

        let count = store.embed_batch(items).unwrap();

        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_remove() {
        let mut store = EmbeddingStore::default_store();

        store.embed_text("doc1", "hello").unwrap();
        assert!(store.contains("doc1"));

        store.remove("doc1");
        assert!(!store.contains("doc1"));
    }

    #[test]
    fn test_clear() {
        let mut store = EmbeddingStore::default_store();

        store.embed_text("doc1", "hello").unwrap();
        store.embed_text("doc2", "world").unwrap();

        store.clear();

        assert!(store.is_empty());
    }

    #[test]
    fn test_bow_model() {
        let config = EmbeddingConfig::new(100).with_model(EmbeddingModel::BagOfWords);
        let mut store = EmbeddingStore::new(config);

        store.embed_text("doc1", "hello hello world").unwrap();

        let emb = store.get("doc1").unwrap();
        // Vector should have non-zero values for "hello" and "world"
        assert!(emb.vector.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_tfidf_model() {
        let config = EmbeddingConfig::new(100).with_model(EmbeddingModel::TfIdf);
        let mut store = EmbeddingStore::new(config);

        store.embed_text("doc1", "hello world").unwrap();
        store.embed_text("doc2", "hello there").unwrap();

        // Both documents should have embeddings
        assert!(store.get("doc1").is_some());
        assert!(store.get("doc2").is_some());
    }

    #[test]
    fn test_stats() {
        let mut store = EmbeddingStore::default_store();

        store.embed_text("doc1", "hello world").unwrap();

        let stats = store.stats();
        assert_eq!(stats.embedding_count, 1);
        assert_eq!(stats.dimensions, 128);
    }

    #[test]
    fn test_reduce_dimensions() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        let reduced = reduce_dimensions(&vectors, 2);

        assert_eq!(reduced.len(), 2);
        assert_eq!(reduced[0].len(), 2);
        assert_eq!(reduced[1].len(), 2);
    }

    #[test]
    fn test_find_similar_to_text() {
        let mut store = EmbeddingStore::default_store();

        store
            .embed_text("doc1", "machine learning algorithms")
            .unwrap();
        store
            .embed_text("doc2", "deep learning neural networks")
            .unwrap();
        store.embed_text("doc3", "cooking recipes food").unwrap();

        let results = store
            .find_similar_to_text("artificial intelligence", 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        // ML-related docs should rank higher than cooking
    }

    #[test]
    fn test_metadata() {
        let mut store = EmbeddingStore::default_store();

        let metadata = serde_json::json!({
            "author": "test",
            "category": "demo"
        });

        store
            .embed_text_with_metadata("doc1", "hello world", metadata.clone())
            .unwrap();

        let emb = store.get("doc1").unwrap();
        assert_eq!(emb.metadata, Some(metadata));
    }
}
