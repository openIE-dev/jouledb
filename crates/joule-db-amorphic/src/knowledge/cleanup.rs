//! Cleanup Memory: denoise recovered vectors after holographic unbinding.
//!
//! After unbinding a triple from the knowledge core, the recovered vector
//! is noisy — it contains interference from all other triples in the bundle.
//! The cleanup step projects the noisy vector onto the nearest clean vector
//! in the concept index, sharpening "loyal, speak, fish, abstract_entity"
//! down to just "loyal."
//!
//! Based on: resonator network cleanup (Frady et al.) and
//! "Optimal Hyperdimensional Representation" (Frontiers 2026) which shows
//! maximizing separation improves decoding from 85% → 100%.

use crate::BinaryHV;
use std::collections::HashMap;

/// A cleanup memory: maps noisy recovered vectors to their nearest clean concept.
pub struct CleanupMemory {
    /// The codebook: label → clean vector.
    codebook: HashMap<String, BinaryHV>,
    /// Similarity threshold: below this, recovery is "failed" (too noisy).
    pub threshold: f32,
    /// Statistics
    pub cleanups: u64,
    pub successes: u64,
    pub failures: u64,
}

/// Result of a cleanup operation.
#[derive(Clone, Debug)]
pub struct CleanupResult {
    /// The nearest clean concept (if found above threshold).
    pub concept: Option<String>,
    /// Similarity to the nearest clean concept.
    pub similarity: f32,
    /// Top K candidates considered.
    pub candidates: Vec<(String, f32)>,
    /// Whether cleanup succeeded (similarity > threshold).
    pub success: bool,
}

impl CleanupMemory {
    pub fn new(threshold: f32) -> Self {
        Self {
            codebook: HashMap::new(),
            threshold,
            cleanups: 0,
            successes: 0,
            failures: 0,
        }
    }

    /// Default threshold: 0.52 (above chance for 10K-dim binary vectors).
    pub fn with_default_threshold() -> Self {
        Self::new(0.52)
    }

    /// Register a clean concept vector in the codebook.
    pub fn register(&mut self, label: &str, vector: BinaryHV) {
        self.codebook.insert(label.to_lowercase(), vector);
    }

    /// Register multiple concepts from a HashMap.
    pub fn register_all(&mut self, concepts: &HashMap<String, BinaryHV>) {
        for (label, vector) in concepts {
            self.codebook.insert(label.to_lowercase(), vector.clone());
        }
    }

    /// Codebook size.
    pub fn size(&self) -> usize {
        self.codebook.len()
    }

    /// Clean up a noisy vector: find the nearest concept in the codebook.
    pub fn cleanup(&mut self, noisy: &BinaryHV) -> CleanupResult {
        self.cleanups += 1;

        let mut candidates: Vec<(String, f32)> = self
            .codebook
            .iter()
            .map(|(label, clean)| (label.clone(), clean.similarity(noisy)))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(5);

        let (concept, similarity, success) = if let Some((label, sim)) = candidates.first() {
            if *sim >= self.threshold {
                self.successes += 1;
                (Some(label.clone()), *sim, true)
            } else {
                self.failures += 1;
                (None, *sim, false)
            }
        } else {
            self.failures += 1;
            (None, 0.0, false)
        };

        CleanupResult {
            concept,
            similarity,
            candidates,
            success,
        }
    }

    /// Clean up and return only the top N concepts above threshold.
    /// Filters out noise — this is the Inhibit primitive applied to cleanup.
    pub fn cleanup_top_n(&mut self, noisy: &BinaryHV, n: usize) -> Vec<(String, f32)> {
        self.cleanups += 1;

        let mut candidates: Vec<(String, f32)> = self
            .codebook
            .iter()
            .map(|(label, clean)| (label.clone(), clean.similarity(noisy)))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let result: Vec<(String, f32)> = candidates
            .into_iter()
            .filter(|(_, sim)| *sim >= self.threshold)
            .take(n)
            .collect();

        if result.is_empty() {
            self.failures += 1;
        } else {
            self.successes += 1;
        }

        result
    }

    /// Iterative cleanup: apply cleanup, then re-query using the clean vector.
    /// Each iteration sharpens the result. Converges in 2-3 rounds typically.
    pub fn iterative_cleanup(
        &mut self,
        noisy: &BinaryHV,
        max_rounds: usize,
    ) -> CleanupResult {
        let mut current = noisy.clone();
        let mut best_result = self.cleanup(&current);

        for _ in 1..max_rounds {
            if let Some(ref concept) = best_result.concept {
                let clean = match self.codebook.get(concept) {
                    Some(v) => v.clone(),
                    None => break,
                };
                let new_result = self.cleanup(&clean);
                if new_result.concept == best_result.concept {
                    break; // Converged
                }
                if new_result.similarity > best_result.similarity {
                    best_result = new_result;
                } else {
                    break; // No improvement
                }
            } else {
                break; // First round failed
            }
        }

        best_result
    }

    /// Cleanup success rate.
    pub fn success_rate(&self) -> f64 {
        if self.cleanups == 0 {
            return 0.0;
        }
        self.successes as f64 / self.cleanups as f64
    }
}

impl Default for CleanupMemory {
    fn default() -> Self {
        Self::with_default_threshold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_exact_match() {
        let mut mem = CleanupMemory::with_default_threshold();
        let dog = BinaryHV::random(10000, 1);
        let cat = BinaryHV::random(10000, 2);
        mem.register("dog", dog.clone());
        mem.register("cat", cat.clone());

        // Clean vector should match itself
        let result = mem.cleanup(&dog);
        assert!(result.success);
        assert_eq!(result.concept.as_deref(), Some("dog"));
        assert!((result.similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cleanup_noisy_vector() {
        let mut mem = CleanupMemory::with_default_threshold();
        let dog = BinaryHV::random(10000, 1);
        mem.register("dog", dog.clone());
        mem.register("cat", BinaryHV::random(10000, 2));
        mem.register("bird", BinaryHV::random(10000, 3));

        // Add noise: flip ~10% of bits
        let noise = BinaryHV::random(10000, 99);
        // XOR with a sparse mask (AND of two random = ~25% set, XOR = ~25% flipped)
        let noise2 = BinaryHV::random(10000, 100);
        let mask = BinaryHV::from_words(
            noise
                .as_words()
                .iter()
                .zip(noise2.as_words().iter())
                .map(|(a, b)| a & b) // ~25% bits set
                .collect(),
            10000,
        );
        let noisy_dog = dog.bind(&mask); // Flip ~25% of bits

        let result = mem.cleanup(&noisy_dog);
        // Should still recover "dog" despite noise
        assert_eq!(
            result.candidates[0].0, "dog",
            "noisy dog should still be closest to dog: {:?}",
            result.candidates
        );
    }

    #[test]
    fn test_cleanup_random_fails() {
        let mut mem = CleanupMemory::with_default_threshold();
        mem.register("dog", BinaryHV::random(10000, 1));
        mem.register("cat", BinaryHV::random(10000, 2));

        // Totally random vector should fail cleanup
        let random = BinaryHV::random(10000, 999);
        let result = mem.cleanup(&random);
        // Similarity should be near 0.5 (random chance)
        assert!(
            result.similarity < 0.55,
            "random vector should not match anything well: {}",
            result.similarity
        );
    }

    #[test]
    fn test_cleanup_top_n() {
        let mut mem = CleanupMemory::new(0.45); // Lower threshold
        mem.register("dog", BinaryHV::random(10000, 1));
        mem.register("cat", BinaryHV::random(10000, 2));
        mem.register("bird", BinaryHV::random(10000, 3));

        let dog = mem.codebook.get("dog").unwrap().clone();
        let results = mem.cleanup_top_n(&dog, 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "dog");
    }

    #[test]
    fn test_success_rate() {
        let mut mem = CleanupMemory::with_default_threshold();
        let dog = BinaryHV::random(10000, 1);
        mem.register("dog", dog.clone());

        mem.cleanup(&dog); // success
        mem.cleanup(&dog); // success
        mem.cleanup(&BinaryHV::random(10000, 999)); // likely fail

        assert!(mem.success_rate() > 0.5);
    }
}
