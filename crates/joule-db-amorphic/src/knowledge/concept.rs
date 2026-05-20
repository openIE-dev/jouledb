//! Concept encoding: text → BinaryHV.
//!
//! Every concept in the knowledge core is a 10,000-dimensional binary vector.
//! Encoding uses character n-gram binding: each trigram gets a random base vector,
//! positionally permuted, then all trigrams are bundled via majority vote.
//!
//! This means:
//! - Similar words produce similar vectors ("cat" ~ "cats" ~ "catalog")
//! - The encoding is deterministic (same word → same vector, always)
//! - No training required — one-shot encoding

use crate::BinaryHV;
use joule_db_hdc::BundleAccumulator;
use std::collections::HashMap;

/// Default dimension for knowledge core vectors.
pub const KNOWLEDGE_DIM: usize = 10_000;

/// A concept that has been encoded into the holographic algebra.
#[derive(Clone, Debug)]
pub struct EncodedConcept {
    /// The original text label (e.g., "dog", "is_a", "run")
    pub label: String,
    /// The holographic encoding
    pub vector: BinaryHV,
}

impl EncodedConcept {
    /// Similarity to another encoded concept.
    pub fn similarity(&self, other: &EncodedConcept) -> f32 {
        self.vector.similarity(&other.vector)
    }
}

/// Encodes text concepts into BinaryHV using character n-gram binding.
pub struct ConceptEncoder {
    /// Cached encodings for concepts we've seen.
    cache: HashMap<String, BinaryHV>,
    /// Dimension of output vectors.
    dim: usize,
    /// Base seed for deterministic encoding.
    seed: u64,
    /// Use hash encoding instead of trigrams.
    /// Hash encoding gives maximum separation (every concept is near-orthogonal).
    /// Trigram encoding gives similarity preservation (similar words → similar vectors).
    /// Use hash for structured IDs (synsets, URIs). Use trigrams for natural language.
    pub hash_mode: bool,
}

impl ConceptEncoder {
    pub fn new(dim: usize) -> Self {
        Self {
            cache: HashMap::new(),
            dim,
            seed: 0xC0_4CE9_7000_1ED6, // "CONCEPT_KNOWLEDGE"
            hash_mode: false,
        }
    }

    pub fn with_default_dim() -> Self {
        Self::new(KNOWLEDGE_DIM)
    }

    /// Encode a concept string into a BinaryHV.
    /// Uses character trigram binding with positional permutation.
    /// Deterministic: same string → same vector.
    pub fn encode(&mut self, label: &str) -> EncodedConcept {
        let normalized = Self::normalize(label);

        if let Some(cached) = self.cache.get(&normalized) {
            return EncodedConcept {
                label: normalized,
                vector: cached.clone(),
            };
        }

        let vector = if self.hash_mode {
            BinaryHV::from_data(normalized.as_bytes(), self.dim)
        } else {
            self.encode_text(&normalized)
        };
        self.cache.insert(normalized.clone(), vector.clone());

        EncodedConcept {
            label: normalized,
            vector,
        }
    }

    /// Encode without caching (for one-off lookups).
    pub fn encode_ephemeral(&self, label: &str) -> EncodedConcept {
        let normalized = Self::normalize(label);
        let vector = if self.hash_mode {
            BinaryHV::from_data(normalized.as_bytes(), self.dim)
        } else {
            self.encode_text(&normalized)
        };
        EncodedConcept {
            label: normalized,
            vector,
        }
    }

    /// Number of cached concepts.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get a cached encoding if it exists.
    pub fn get_cached(&self, label: &str) -> Option<&BinaryHV> {
        self.cache.get(&Self::normalize(label))
    }

    /// Normalize a concept label: lowercase, trim, collapse whitespace.
    fn normalize(label: &str) -> String {
        label
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("_")
    }

    /// Core encoding: character trigram binding with positional permutation.
    fn encode_text(&self, text: &str) -> BinaryHV {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return BinaryHV::zeros(self.dim);
        }

        let mut acc = BundleAccumulator::new(self.dim);

        // Generate character-level base vectors (deterministic from seed + char)
        // Then bind trigrams with positional permutation
        if chars.len() < 3 {
            // Short text: just hash it
            let hv = BinaryHV::from_data(text.as_bytes(), self.dim);
            return hv;
        }

        for i in 0..chars.len().saturating_sub(2) {
            let c0 = chars[i] as u64;
            let c1 = chars[i + 1] as u64;
            let c2 = chars[i + 2] as u64;

            // Each character gets a deterministic base vector
            let v0 = BinaryHV::random(self.dim, self.seed.wrapping_add(c0));
            let v1 = BinaryHV::random(self.dim, self.seed.wrapping_add(c1 + 256));
            let v2 = BinaryHV::random(self.dim, self.seed.wrapping_add(c2 + 512));

            // Bind the trigram with positional permutation
            let trigram = v0.bind(&v1.permute(1)).bind(&v2.permute(2));

            // Positionally permute the trigram within the text
            let positioned = trigram.permute(i);

            acc.add(&positioned);
        }

        acc.threshold()
    }
}

impl Default for ConceptEncoder {
    fn default() -> Self {
        Self::with_default_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_encoding() {
        let mut enc = ConceptEncoder::with_default_dim();
        let a = enc.encode("dog");
        let b = enc.encode("dog");
        assert_eq!(a.vector.hamming_distance(&b.vector), 0);
    }

    #[test]
    fn test_similar_words_similar_vectors() {
        let mut enc = ConceptEncoder::with_default_dim();
        let cat = enc.encode("cat");
        let cats = enc.encode("cats");
        let dog = enc.encode("dog");

        let sim_cat_cats = cat.similarity(&cats);
        let sim_cat_dog = cat.similarity(&dog);

        assert!(
            sim_cat_cats > sim_cat_dog,
            "cat~cats ({sim_cat_cats}) should be more similar than cat~dog ({sim_cat_dog})"
        );
    }

    #[test]
    fn test_different_words_near_orthogonal() {
        let mut enc = ConceptEncoder::with_default_dim();
        let apple = enc.encode("apple");
        let quantum = enc.encode("quantum_mechanics");
        let sim = apple.similarity(&quantum);
        assert!(sim < 0.6, "unrelated concepts should be near-orthogonal: {sim}");
    }

    #[test]
    fn test_normalization() {
        let mut enc = ConceptEncoder::with_default_dim();
        let a = enc.encode("Hello World");
        let b = enc.encode("hello   world");
        assert_eq!(a.vector.hamming_distance(&b.vector), 0);
    }

    #[test]
    fn test_cache() {
        let mut enc = ConceptEncoder::with_default_dim();
        enc.encode("test");
        assert_eq!(enc.cache_size(), 1);
        enc.encode("test");
        assert_eq!(enc.cache_size(), 1); // No duplicate
        enc.encode("other");
        assert_eq!(enc.cache_size(), 2);
    }
}
