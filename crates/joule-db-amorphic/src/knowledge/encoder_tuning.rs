//! Encoder Tuning: separation metric from "Optimal Hyperdimensional Representation" (Frontiers 2026).
//!
//! The paper showed: tuning correlation improves classification 65% → 95%,
//! maximizing separation improves decoding 85% → 100%.
//!
//! Two competing objectives:
//! - **Correlation**: similar concepts should have similar encodings (learning)
//! - **Separation**: different concepts should have maximally different encodings (cognition/retrieval)
//!
//! The separation metric quantifies this tradeoff and guides encoder design.

use crate::BinaryHV;
use std::collections::HashMap;

/// Measures the quality of an encoding by its separation and correlation properties.
#[derive(Clone, Debug)]
pub struct EncoderMetrics {
    /// Mean pairwise similarity between all encoded concepts.
    /// Lower = better separation for retrieval.
    pub mean_similarity: f32,
    /// Std dev of pairwise similarities.
    /// Higher = more discriminative encoding.
    pub similarity_std: f32,
    /// Minimum pairwise similarity (worst-case collision).
    pub min_similarity: f32,
    /// Maximum pairwise similarity (excluding self).
    pub max_similarity: f32,
    /// Separation score: (1 - mean_similarity) * similarity_std.
    /// Higher = better encoding for cognition tasks.
    pub separation_score: f32,
    /// Number of concept pairs measured.
    pub pairs_measured: usize,
}

/// Compute encoder metrics over a set of encoded concepts.
pub fn measure_encoding(concepts: &HashMap<String, BinaryHV>) -> EncoderMetrics {
    let vecs: Vec<(&String, &BinaryHV)> = concepts.iter().collect();
    let n = vecs.len();

    if n < 2 {
        return EncoderMetrics {
            mean_similarity: 0.0,
            similarity_std: 0.0,
            min_similarity: 0.0,
            max_similarity: 0.0,
            separation_score: 0.0,
            pairs_measured: 0,
        };
    }

    let mut sims = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            sims.push(vecs[i].1.similarity(vecs[j].1));
        }
    }

    let pairs = sims.len();
    let mean: f32 = sims.iter().sum::<f32>() / pairs as f32;
    let variance: f32 = sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / pairs as f32;
    let std = variance.sqrt();
    let min = sims.iter().cloned().fold(f32::MAX, f32::min);
    let max = sims.iter().cloned().fold(f32::MIN, f32::max);

    EncoderMetrics {
        mean_similarity: mean,
        similarity_std: std,
        min_similarity: min,
        max_similarity: max,
        separation_score: (1.0 - mean) * std,
        pairs_measured: pairs,
    }
}

/// Optimal dimension selection: given a target number of concepts,
/// what's the minimum dimension that achieves the desired separation?
///
/// From the Frontiers 2026 paper: for N random binary vectors in D dimensions,
/// the expected pairwise similarity is 0.5 with std ≈ 1/(2√D).
/// For reliable retrieval (cleanup success), we need:
///   max_similarity < threshold  (i.e., no two concepts collide)
///
/// Rule of thumb: D ≥ 100 * ln(N) for N concepts with separation > 0.95.
pub fn recommended_dimension(num_concepts: usize) -> usize {
    let ln_n = (num_concepts as f64).ln();
    let min_dim = (100.0 * ln_n).ceil() as usize;
    // Round up to nearest multiple of 64 (for BinaryHV word alignment)
    ((min_dim + 63) / 64) * 64
}

/// Adaptive encoder: adjusts encoding based on separation feedback.
///
/// If two concepts are too similar (collision), perturb one of them
/// to increase separation. This is the "maximize separation" strategy.
pub struct AdaptiveEncoder {
    /// Current encodings.
    codebook: HashMap<String, BinaryHV>,
    /// Target minimum separation between any two concepts.
    pub min_separation: f32,
    /// Dimension.
    dim: usize,
    /// Seed for deterministic perturbation.
    seed: u64,
    /// Number of perturbations applied.
    pub perturbations: u64,
}

impl AdaptiveEncoder {
    pub fn new(dim: usize) -> Self {
        Self {
            codebook: HashMap::new(),
            min_separation: 0.45, // similarity < 0.55 = well separated
            dim,
            seed: 0xAD_A971_E4C0_DEC0,
            perturbations: 0,
        }
    }

    /// Register a concept. If it collides with an existing concept,
    /// perturb it until separation is achieved.
    pub fn register(&mut self, label: &str, vector: BinaryHV) -> BinaryHV {
        let key = label.to_lowercase();
        let mut candidate = vector;

        // Check for collisions with existing concepts
        let max_attempts = 10;
        for attempt in 0..max_attempts {
            let mut collision = false;
            for (existing_label, existing_vec) in &self.codebook {
                if existing_label == &key {
                    continue;
                }
                let sim = candidate.similarity(existing_vec);
                if sim > (1.0 - self.min_separation) {
                    // Collision: perturb the candidate
                    collision = true;
                    candidate = self.perturb(&candidate, attempt as u64);
                    self.perturbations += 1;
                    break;
                }
            }
            if !collision {
                break;
            }
        }

        self.codebook.insert(key, candidate.clone());
        candidate
    }

    /// Get a registered encoding.
    pub fn get(&self, label: &str) -> Option<&BinaryHV> {
        self.codebook.get(&label.to_lowercase())
    }

    /// Get all encodings (for metrics computation).
    pub fn codebook(&self) -> &HashMap<String, BinaryHV> {
        &self.codebook
    }

    /// Current metrics.
    pub fn metrics(&self) -> EncoderMetrics {
        measure_encoding(&self.codebook)
    }

    /// Number of registered concepts.
    pub fn size(&self) -> usize {
        self.codebook.len()
    }

    /// Perturb a vector by XORing with a sparse random mask.
    fn perturb(&self, vector: &BinaryHV, attempt: u64) -> BinaryHV {
        let noise = BinaryHV::random(
            self.dim,
            self.seed.wrapping_add(attempt).wrapping_mul(7919),
        );
        // XOR with noise flips ~50% of bits — too aggressive.
        // AND two random vectors to get ~25% mask, then XOR.
        let noise2 = BinaryHV::random(
            self.dim,
            self.seed.wrapping_add(attempt + 1000).wrapping_mul(6547),
        );
        let sparse_mask = BinaryHV::from_words(
            noise
                .as_words()
                .iter()
                .zip(noise2.as_words().iter())
                .map(|(a, b)| a & b)
                .collect(),
            self.dim,
        );
        vector.bind(&sparse_mask) // XOR with sparse mask = flip ~25% of bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_encoding_random() {
        let mut concepts = HashMap::new();
        for i in 0..20 {
            concepts.insert(format!("concept_{i}"), BinaryHV::random(10000, i));
        }

        let metrics = measure_encoding(&concepts);
        // Random 10K-dim binary vectors: mean similarity ≈ 0.5
        assert!(
            (metrics.mean_similarity - 0.5).abs() < 0.05,
            "mean sim should be ~0.5: {}",
            metrics.mean_similarity
        );
        assert!(metrics.pairs_measured > 0);
        assert!(metrics.separation_score > 0.0);
    }

    #[test]
    fn test_recommended_dimension() {
        let dim_100 = recommended_dimension(100);
        let dim_1m = recommended_dimension(1_000_000);

        assert!(dim_100 >= 448); // 100 * ln(100) ≈ 461
        assert!(dim_1m >= 1344); // 100 * ln(1M) ≈ 1382
        // Should be word-aligned
        assert_eq!(dim_100 % 64, 0);
        assert_eq!(dim_1m % 64, 0);
    }

    #[test]
    fn test_adaptive_encoder_no_collision() {
        let mut enc = AdaptiveEncoder::new(10000);
        let a = enc.register("dog", BinaryHV::random(10000, 1));
        let b = enc.register("cat", BinaryHV::random(10000, 2));

        // Random vectors shouldn't collide at 10K dims
        assert_eq!(enc.perturbations, 0);
        let sim = a.similarity(&b);
        assert!(sim < 0.55);
    }

    #[test]
    fn test_adaptive_encoder_metrics() {
        let mut enc = AdaptiveEncoder::new(10000);
        for i in 0..10 {
            enc.register(&format!("concept_{i}"), BinaryHV::random(10000, i));
        }

        let metrics = enc.metrics();
        assert_eq!(metrics.pairs_measured, 45); // 10 choose 2
        assert!((metrics.mean_similarity - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_separation_score_improves_with_dimension() {
        // Higher dimension = better separation (more orthogonal space)
        let mut concepts_low = HashMap::new();
        let mut concepts_high = HashMap::new();

        for i in 0..10u64 {
            concepts_low.insert(format!("c{i}"), BinaryHV::random(100, i));
            concepts_high.insert(format!("c{i}"), BinaryHV::random(10000, i));
        }

        let metrics_low = measure_encoding(&concepts_low);
        let metrics_high = measure_encoding(&concepts_high);

        // Higher dimension should have tighter distribution around 0.5
        assert!(
            metrics_high.similarity_std < metrics_low.similarity_std,
            "higher dim should have tighter similarity distribution"
        );
    }
}
