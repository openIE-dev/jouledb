//! Primitive 7: Forget — natural decoherence / decay.
//!
//! Information fades without active maintenance. This is not deletion —
//! it's entropy increase. A forgotten vector drifts toward the maximally
//! mixed state (random noise), losing its learned structure.
//!
//! Biological analogy: synaptic decay, memory consolidation failure.
//! Physical analogy: decoherence in quantum systems, thermal noise.

use crate::turbo_holographic::BinaryHV;

/// Rate of forgetting. Controls how fast a vector decoheres.
#[derive(Clone, Copy, Debug)]
pub struct DecayRate {
    /// Half-life in arbitrary time units. After this many units,
    /// half the bits will have flipped toward random.
    pub half_life: f64,
}

impl DecayRate {
    /// Create a decay rate from half-life.
    pub fn new(half_life: f64) -> Self {
        Self { half_life }
    }

    /// Very slow decay — long-term memory.
    pub fn slow() -> Self {
        Self { half_life: 10000.0 }
    }

    /// Medium decay — working memory.
    pub fn medium() -> Self {
        Self { half_life: 100.0 }
    }

    /// Fast decay — sensory buffer.
    pub fn fast() -> Self {
        Self { half_life: 10.0 }
    }

    /// Compute the probability of each bit flipping after `elapsed` time units.
    /// Returns a value in [0.0, 0.5] where 0.0 = no decay, 0.5 = fully random.
    pub fn flip_probability(&self, elapsed: f64) -> f64 {
        if self.half_life <= 0.0 {
            return 0.5;
        }
        0.5 * (1.0 - (-elapsed * std::f64::consts::LN_2 / self.half_life).exp())
    }
}

/// Result of a decay operation.
#[derive(Clone, Debug)]
pub struct Decay {
    /// The decayed vector.
    pub vector: BinaryHV,
    /// Number of bits that flipped due to decay.
    pub bits_flipped: u32,
    /// Remaining fidelity: similarity to the original (1.0 = no decay, 0.5 = fully random).
    pub fidelity: f32,
}

/// Trait for anything that can forget.
pub trait Forgettable {
    /// Apply decay: flip bits with probability proportional to elapsed time.
    /// Uses deterministic noise (seeded PRNG) so decay is reproducible.
    fn decay(&self, rate: &DecayRate, elapsed: f64, seed: u64) -> Decay;

    /// Check if a vector has decayed beyond usefulness.
    /// Returns true if fidelity drops below threshold (default: 0.55).
    fn is_forgotten(&self, rate: &DecayRate, elapsed: f64, threshold: f32) -> bool;
}

impl Forgettable for BinaryHV {
    fn decay(&self, rate: &DecayRate, elapsed: f64, seed: u64) -> Decay {
        let flip_prob = rate.flip_probability(elapsed);
        let dim = self.dimension();

        // Generate deterministic noise mask
        let noise = BinaryHV::random(dim, seed);
        let noise_words = noise.as_words();
        let orig_words = self.as_words();
        let num_words = orig_words.len();

        // For each bit: flip with probability `flip_prob`.
        // We threshold the noise vector to get the flip mask.
        // Bits where noise AND a probability-derived mask are both 1 get flipped.
        //
        // To get approximately `flip_prob` fraction of bits flipped:
        // Use multiple noise vectors XOR'd together to approximate the probability.
        // For simplicity: if flip_prob > 0.25, use noise directly (flips ~50% × coverage).
        // For fine-grained control, generate a second noise vector and AND them.
        let mut result_words = orig_words.to_vec();
        let mut bits_flipped = 0u32;

        if flip_prob < 0.001 {
            // Negligible decay
            return Decay {
                vector: self.clone(),
                bits_flipped: 0,
                fidelity: 1.0,
            };
        }

        if flip_prob >= 0.49 {
            // Fully random — replace with noise
            return Decay {
                vector: noise,
                bits_flipped: self.hamming_distance(&BinaryHV::random(dim, seed)) ,
                fidelity: 0.5,
            };
        }

        // Generate a mask where approximately `flip_prob` fraction of bits are set.
        // Use AND of two independent random vectors: P(bit=1) = 0.5 * 0.5 = 0.25
        // Use OR of two ANDs for P ≈ 0.4375, etc.
        // For arbitrary probability: use threshold on a counter.
        //
        // Simple approach: for each word, generate random u64 and mask based on
        // population threshold.
        let noise2 = BinaryHV::random(dim, seed.wrapping_add(1));
        let noise2_words = noise2.as_words();

        for i in 0..num_words {
            // AND of two random u64s: ~25% of bits set
            // XOR with a third: still ~25% but different pattern
            let mask = if flip_prob < 0.13 {
                // ~6.25%: AND three random vectors
                let noise3 = if i == 0 {
                    BinaryHV::random(dim, seed.wrapping_add(2))
                } else {
                    // Reuse shifted seed for other words
                    BinaryHV::random(dim, seed.wrapping_add(2 + i as u64))
                };
                noise_words[i] & noise2_words[i] & noise3.as_words()[i.min(noise3.as_words().len() - 1)]
            } else if flip_prob < 0.26 {
                // ~25%: AND two random vectors
                noise_words[i] & noise2_words[i]
            } else {
                // ~50%: single random vector
                noise_words[i]
            };

            // XOR with mask to flip selected bits
            result_words[i] ^= mask;
            bits_flipped += mask.count_ones();
        }

        let result = BinaryHV::from_words(result_words, dim);
        let fidelity = self.similarity(&result);

        Decay {
            vector: result,
            bits_flipped,
            fidelity,
        }
    }

    fn is_forgotten(&self, rate: &DecayRate, elapsed: f64, threshold: f32) -> bool {
        let expected_fidelity = 1.0 - rate.flip_probability(elapsed) as f32;
        expected_fidelity < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decay() {
        let hv = BinaryHV::random(10000, 42);
        let result = hv.decay(&DecayRate::slow(), 0.0, 99);
        assert_eq!(result.bits_flipped, 0);
        assert!((result.fidelity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_partial_decay() {
        let hv = BinaryHV::random(10000, 42);
        let result = hv.decay(&DecayRate::medium(), 50.0, 99);
        assert!(result.bits_flipped > 0);
        assert!(result.fidelity > 0.5);
        assert!(result.fidelity < 1.0);
    }

    #[test]
    fn test_full_decay() {
        let hv = BinaryHV::random(10000, 42);
        let result = hv.decay(&DecayRate::fast(), 100000.0, 99);
        // After very long time, should be near random
        assert!(result.fidelity < 0.55);
    }

    #[test]
    fn test_is_forgotten() {
        let hv = BinaryHV::random(10000, 42);
        assert!(!hv.is_forgotten(&DecayRate::slow(), 1.0, 0.55));
        assert!(hv.is_forgotten(&DecayRate::fast(), 100000.0, 0.55));
    }

    #[test]
    fn test_decay_is_deterministic() {
        let hv = BinaryHV::random(10000, 42);
        let d1 = hv.decay(&DecayRate::medium(), 50.0, 99);
        let d2 = hv.decay(&DecayRate::medium(), 50.0, 99);
        assert_eq!(d1.vector.hamming_distance(&d2.vector), 0);
    }
}
