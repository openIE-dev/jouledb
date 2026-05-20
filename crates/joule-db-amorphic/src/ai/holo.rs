//! Tier 1: Built-in Holographic Inference — zero deps, works everywhere.
//!
//! Uses joule-db-hdc primitives that are always available:
//! - BinaryHV similarity for nearest-neighbor
//! - Character n-gram binding for text encoding
//! - Hamming distance for classification

use joule_db_hdc::{BinaryHV, BundleAccumulator};

use super::traits::HolographicInference;
use crate::{AmorphicStore, RecordId, DIMENSION};

/// Holographic inference engine — Tier 1 implementation.
///
/// Sub-microsecond inference using only HDC operations.
/// No models, no weights, no GPU required.
pub struct HoloEngine {
    dimension: usize,
}

impl HoloEngine {
    pub fn new() -> Self {
        Self {
            dimension: DIMENSION,
        }
    }

    /// Encode text as BinaryHV using character 3-gram binding.
    ///
    /// Each 3-gram is encoded as: hash(char1) ⊗ permute(hash(char2)) ⊗ permute²(hash(char3))
    /// All 3-grams are bundled via majority vote.
    /// This preserves word-level similarity: "movie" and "movies" will be close.
    pub fn text_to_hologram(&self, text: &str) -> BinaryHV {
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();

        if chars.len() < 3 {
            return BinaryHV::from_hash(text.as_bytes(), self.dimension);
        }

        let mut acc = BundleAccumulator::new(self.dimension);

        // Character-level 3-gram encoding
        for window in chars.windows(3) {
            let c0 = BinaryHV::from_hash(
                &[window[0] as u8],
                self.dimension,
            );
            let c1 = BinaryHV::from_hash(
                &[window[1] as u8],
                self.dimension,
            );
            let c2 = BinaryHV::from_hash(
                &[window[2] as u8],
                self.dimension,
            );

            // Positional binding: c0 ⊗ perm(c1) ⊗ perm²(c2)
            let p1 = c1.permute_words(1);
            let p2 = c2.permute_words(2);
            let trigram = c0.bind(&p1).bind(&p2);
            acc.add(&trigram);
        }

        // Also encode whole words for exact matching boost
        for word in lower.split_whitespace() {
            let word_hv = BinaryHV::from_hash(word.as_bytes(), self.dimension);
            acc.add(&word_hv);
        }

        acc.threshold()
    }
}

impl Default for HoloEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HolographicInference for HoloEngine {
    fn similarity(&self, query: &BinaryHV, k: usize) -> Vec<(RecordId, f32)> {
        // This is a standalone query — the store reference is passed per-call
        // via the facade, not held here. Return empty for the trait default.
        // The facade wraps this with store access.
        Vec::new()
    }

    fn attention_read(&self, query: &BinaryHV) -> Vec<f64> {
        // Requires SDM instance — delegated by facade
        Vec::new()
    }

    fn classify_holo(&self, input: &BinaryHV, categories: &[(String, BinaryHV)]) -> (String, f32) {
        categories
            .iter()
            .map(|(label, proto)| (label.clone(), input.similarity(proto)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| ("unknown".to_string(), 0.0))
    }

    fn encode_text(&self, text: &str) -> BinaryHV {
        self.text_to_hologram(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoding_similarity() {
        let engine = HoloEngine::new();

        let movie = engine.text_to_hologram("action movie");
        let film = engine.text_to_hologram("action film");
        let recipe = engine.text_to_hologram("chocolate cake recipe");

        // "action movie" and "action film" share the word "action"
        let sim_related = movie.similarity(&film);
        let sim_unrelated = movie.similarity(&recipe);

        assert!(
            sim_related > sim_unrelated,
            "Related texts should be more similar: {} vs {}",
            sim_related,
            sim_unrelated
        );
    }

    #[test]
    fn test_holographic_classification() {
        let engine = HoloEngine::new();

        let categories = vec![
            ("sports".to_string(), engine.text_to_hologram("football soccer basketball game")),
            ("tech".to_string(), engine.text_to_hologram("computer software programming code")),
            ("food".to_string(), engine.text_to_hologram("cooking recipe kitchen chef")),
        ];

        let input = engine.text_to_hologram("the basketball game was exciting");
        let (label, confidence) = engine.classify_holo(&input, &categories);

        assert_eq!(label, "sports");
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_short_text_encoding() {
        let engine = HoloEngine::new();

        // Short texts (< 3 chars) fall back to hash
        let short = engine.text_to_hologram("hi");
        assert_eq!(short.dimension(), DIMENSION);
    }

    #[test]
    fn test_empty_classification() {
        let engine = HoloEngine::new();
        let input = BinaryHV::from_hash(b"test", DIMENSION);
        let (label, _) = engine.classify_holo(&input, &[]);
        assert_eq!(label, "unknown");
    }
}
