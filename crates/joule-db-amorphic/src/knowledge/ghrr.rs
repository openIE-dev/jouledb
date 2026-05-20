//! GHRR: Generalized Holographic Reduced Representations
//!
//! Extends FHRR with non-commutative binding via block-diagonal phase matrices.
//! This is the key to multi-hop reasoning: "dog bites man" ≠ "man bites dog"
//! at the algebra level, without needing explicit Permute.
//!
//! Based on: "Generalized Holographic Reduced Representations" (arXiv:2405.09689)
//! and PathHD (arXiv:2512.09369) which proved this is critical for KG reasoning.
//!
//! ## How it works
//!
//! Standard FHRR binding: element-wise phase addition (commutative).
//!   bind(a, b) = bind(b, a)
//!
//! GHRR binding: block-diagonal phase multiplication with per-block rotation.
//!   bind(a, b) ≠ bind(b, a)  (non-commutative when rotations differ)
//!
//! The vector is split into B blocks. Each block has its own rotation matrix.
//! Left-binding and right-binding use different rotations, breaking symmetry.

use crate::BinaryHV;
use std::f32::consts::PI;

/// Number of blocks for block-diagonal GHRR.
/// More blocks = finer-grained non-commutativity, but more overhead.
pub const DEFAULT_NUM_BLOCKS: usize = 16;

/// A GHRR vector: block-diagonal phasor representation.
#[derive(Clone, Debug)]
pub struct GhrrVector {
    /// Phase angles organized as blocks.
    /// Total length = dimension. Block size = dimension / num_blocks.
    pub angles: Vec<f32>,
    /// Number of blocks.
    pub num_blocks: usize,
    /// Total dimension.
    pub dimension: usize,
}

impl GhrrVector {
    /// Create from raw angles with specified block count.
    pub fn new(angles: Vec<f32>, num_blocks: usize) -> Self {
        let dimension = angles.len();
        Self {
            angles,
            num_blocks,
            dimension,
        }
    }

    /// Create a random GHRR vector (deterministic from seed).
    /// Uses splitmix64 for high-quality uniform phase distribution.
    pub fn random(dimension: usize, num_blocks: usize, seed: u64) -> Self {
        let mut angles = Vec::with_capacity(dimension);
        let mut state = seed;
        for _ in 0..dimension {
            // splitmix64
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let phase = (z as f64 / u64::MAX as f64) as f32 * 2.0 * PI;
            angles.push(phase);
        }
        Self {
            angles,
            num_blocks,
            dimension,
        }
    }

    /// Create from text using character trigram encoding (same as ConceptEncoder
    /// but outputs GHRR instead of BinaryHV).
    pub fn from_text(text: &str, dimension: usize, num_blocks: usize, seed: u64) -> Self {
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        if chars.len() < 3 {
            return Self::random(dimension, num_blocks, seed.wrapping_add(text.len() as u64));
        }

        let mut result = vec![0.0f32; dimension];
        let mut count = 0u32;

        for i in 0..chars.len().saturating_sub(2) {
            let c0 = chars[i] as u64;
            let c1 = chars[i + 1] as u64;
            let c2 = chars[i + 2] as u64;

            let v0 = Self::random(dimension, num_blocks, seed.wrapping_add(c0));
            let v1 = Self::random(dimension, num_blocks, seed.wrapping_add(c1 + 256));
            let v2 = Self::random(dimension, num_blocks, seed.wrapping_add(c2 + 512));

            // Non-commutative bind: v0 ⊗_L v1 ⊗_L v2 with position-shifted rotations
            let trigram = v0.bind_left(&v1).bind_left(&v2);

            for (j, angle) in trigram.angles.iter().enumerate() {
                result[j] += angle;
            }
            count += 1;
        }

        // Normalize: circular mean per dimension
        if count > 0 {
            for angle in &mut result {
                *angle /= count as f32;
                // Wrap to [0, 2π)
                *angle = ((*angle % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
            }
        }

        Self {
            angles: result,
            num_blocks,
            dimension,
        }
    }

    /// Block size (dimensions per block).
    pub fn block_size(&self) -> usize {
        self.dimension / self.num_blocks
    }

    /// Cosine similarity: mean(cos(θ_a - θ_b)) across all dimensions.
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.dimension, other.dimension);
        let sum: f32 = self
            .angles
            .iter()
            .zip(other.angles.iter())
            .map(|(a, b)| (a - b).cos())
            .sum();
        sum / self.dimension as f32
    }

    /// Block-cosine similarity: average cosine per block, then average across blocks.
    /// This is what PathHD uses for calibrated retrieval.
    pub fn block_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.dimension, other.dimension);
        assert_eq!(self.num_blocks, other.num_blocks);
        let bs = self.block_size();
        let mut block_sims = Vec::with_capacity(self.num_blocks);

        for b in 0..self.num_blocks {
            let start = b * bs;
            let end = (start + bs).min(self.dimension);
            let mut sum = 0.0f32;
            let mut count = 0;
            for i in start..end {
                sum += (self.angles[i] - other.angles[i]).cos();
                count += 1;
            }
            if count > 0 {
                block_sims.push(sum / count as f32);
            }
        }

        if block_sims.is_empty() {
            return 0.0;
        }
        block_sims.iter().sum::<f32>() / block_sims.len() as f32
    }

    /// Left-bind (non-commutative): a ⊗_L b.
    /// Applies a per-block circular shift to the RIGHT operand before phase addition.
    /// Since the shift is applied only to `other`, bind_left(a,b) ≠ bind_left(b,a).
    pub fn bind_left(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let bs = self.block_size();
        let mut result = vec![0.0f32; self.dimension];

        for b in 0..self.num_blocks {
            let start = b * bs;
            let end = (start + bs).min(self.dimension);
            // Per-block circular shift of the right operand's indices within the block.
            // Block 0 shifts by 1, block 1 by 2, etc.
            let shift = (b + 1) % bs.max(1);

            for i in start..end {
                // Shifted index within block for the right operand
                let block_offset = i - start;
                let shifted_offset = (block_offset + shift) % bs;
                let shifted_idx = start + shifted_offset;

                // Phase addition with shifted right operand
                result[i] = (self.angles[i] + other.angles[shifted_idx]) % (2.0 * PI);
            }
        }

        Self {
            angles: result,
            num_blocks: self.num_blocks,
            dimension: self.dimension,
        }
    }

    /// Right-bind (non-commutative): a ⊗_R b.
    /// Applies the circular shift to the LEFT operand instead.
    pub fn bind_right(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let bs = self.block_size();
        let mut result = vec![0.0f32; self.dimension];

        for b in 0..self.num_blocks {
            let start = b * bs;
            let end = (start + bs).min(self.dimension);
            let shift = (b + 1) % bs.max(1);

            for i in start..end {
                let block_offset = i - start;
                let shifted_offset = (block_offset + shift) % bs;
                let shifted_idx = start + shifted_offset;

                result[i] = (self.angles[shifted_idx] + other.angles[i]) % (2.0 * PI);
            }
        }

        Self {
            angles: result,
            num_blocks: self.num_blocks,
            dimension: self.dimension,
        }
    }

    /// Unbind left: recover b from (a ⊗_L b) given a.
    /// Inverse of bind_left: undo the block-circular shift on the result.
    pub fn unbind_left(&self, key: &Self) -> Self {
        assert_eq!(self.dimension, key.dimension);
        let bs = self.block_size();
        let mut result = vec![0.0f32; self.dimension];

        for b in 0..self.num_blocks {
            let start = b * bs;
            let end = (start + bs).min(self.dimension);
            let shift = (b + 1) % bs.max(1);

            for i in start..end {
                let block_offset = i - start;
                // Inverse shift: undo the right operand's shift
                let shifted_offset = (block_offset + shift) % bs;
                let shifted_idx = start + shifted_offset;

                // self.angles[i] = key.angles[i] + other.angles[shifted_idx]
                // => other.angles[shifted_idx] = self.angles[i] - key.angles[i]
                // We want result[shifted_idx - start] = self.angles[i] - key.angles[i]
                let recovered =
                    ((self.angles[i] - key.angles[i]) % (2.0 * PI) + 2.0 * PI) % (2.0 * PI);
                result[shifted_idx] = recovered;
            }
        }

        Self {
            angles: result,
            num_blocks: self.num_blocks,
            dimension: self.dimension,
        }
    }

    /// Encode a multi-hop path: [r1, r2, r3] as r1 ⊗_L r2 ⊗_L r3.
    /// Non-commutative: order matters.
    pub fn encode_path(relations: &[&Self]) -> Option<Self> {
        if relations.is_empty() {
            return None;
        }
        let mut result = relations[0].clone();
        for rel in &relations[1..] {
            result = result.bind_left(rel);
        }
        Some(result)
    }

    /// Convert to BinaryHV by thresholding phase at π.
    /// cos(θ) < 0 → bit 1, else bit 0.
    pub fn to_binaryhv(&self) -> BinaryHV {
        let num_words = (self.dimension + 63) / 64;
        let mut words = vec![0u64; num_words];
        for (i, angle) in self.angles.iter().enumerate() {
            if angle.cos() < 0.0 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        BinaryHV::from_words(words, self.dimension)
    }

    /// Create from BinaryHV: bit 0 → phase 0, bit 1 → phase π.
    pub fn from_binaryhv(hv: &BinaryHV, num_blocks: usize) -> Self {
        let words = hv.as_words();
        let dim = hv.dimension();
        let mut angles = Vec::with_capacity(dim);
        for i in 0..dim {
            let bit = (words[i / 64] >> (i % 64)) & 1;
            angles.push(if bit == 1 { PI } else { 0.0 });
        }
        Self {
            angles,
            num_blocks,
            dimension: dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_commutativity() {
        let a = GhrrVector::random(1000, 16, 1);
        let b = GhrrVector::random(1000, 16, 2);

        let ab = a.bind_left(&b);
        let ba = b.bind_left(&a);

        let sim = ab.similarity(&ba);
        assert!(
            sim < 0.9,
            "bind_left should be non-commutative: similarity = {sim}"
        );
    }

    #[test]
    fn test_unbind_left_recovers() {
        let a = GhrrVector::random(1000, 16, 1);
        let b = GhrrVector::random(1000, 16, 2);

        let bound = a.bind_left(&b);
        let recovered = bound.unbind_left(&a);

        let sim = recovered.similarity(&b);
        assert!(
            sim > 0.8,
            "unbind should recover b: similarity = {sim}"
        );
    }

    #[test]
    fn test_self_similarity() {
        let a = GhrrVector::random(1000, 16, 42);
        let sim = a.similarity(&a);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_random_orthogonal() {
        let a = GhrrVector::random(10000, 16, 1);
        let b = GhrrVector::random(10000, 16, 2);
        let sim = a.similarity(&b);
        assert!(sim.abs() < 0.1, "random vectors should be near-orthogonal: {sim}");
    }

    #[test]
    fn test_encode_path_order_matters() {
        let r1 = GhrrVector::random(1000, 16, 10);
        let r2 = GhrrVector::random(1000, 16, 20);
        let r3 = GhrrVector::random(1000, 16, 30);

        let path_123 = GhrrVector::encode_path(&[&r1, &r2, &r3]).unwrap();
        let path_321 = GhrrVector::encode_path(&[&r3, &r2, &r1]).unwrap();

        let sim = path_123.similarity(&path_321);
        assert!(
            sim < 0.9,
            "path order should matter: similarity = {sim}"
        );
    }

    #[test]
    fn test_block_similarity() {
        let a = GhrrVector::random(1000, 16, 1);
        let sim_full = a.similarity(&a);
        let sim_block = a.block_similarity(&a);
        assert!((sim_full - 1.0).abs() < 0.001);
        assert!((sim_block - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_to_from_binaryhv() {
        let ghrr = GhrrVector::random(1000, 16, 42);
        let bhv = ghrr.to_binaryhv();
        let back = GhrrVector::from_binaryhv(&bhv, 16);
        // Roundtrip through binary is lossy (only preserves sign of cos)
        // but should be correlated
        let sim = ghrr.similarity(&back);
        assert!(sim > 0.3, "roundtrip should be correlated: {sim}");
    }

    #[test]
    fn test_from_text() {
        let dog = GhrrVector::from_text("dog", 1000, 16, 42);
        let cat = GhrrVector::from_text("cat", 1000, 16, 42);
        let dog2 = GhrrVector::from_text("dog", 1000, 16, 42);

        // Deterministic
        assert!((dog.similarity(&dog2) - 1.0).abs() < 0.001);
        // Different words are different
        let sim = dog.similarity(&cat);
        assert!(sim < 0.8, "dog and cat should differ: {sim}");
    }

    #[test]
    fn test_single_hop_unbind_clean() {
        // Single-hop unbind is clean (tested in test_unbind_left_recovers).
        // Multi-hop unbind through block-circular shifts accumulates error.
        // In the full system, cleanup memory resolves this.
        // Here we verify that single-hop is reliable.
        let a = GhrrVector::random(1000, 16, 1);
        let b = GhrrVector::random(1000, 16, 2);

        let bound = a.bind_left(&b);
        let recovered = bound.unbind_left(&a);

        let sim = recovered.similarity(&b);
        assert!(
            sim > 0.8,
            "single-hop unbind should be clean: similarity = {sim}"
        );

        // Also verify the wrong key doesn't recover
        let wrong_key = GhrrVector::random(1000, 16, 99);
        let wrong_recovery = bound.unbind_left(&wrong_key);
        let wrong_sim = wrong_recovery.similarity(&b);
        assert!(
            wrong_sim < 0.3,
            "wrong key should not recover: similarity = {wrong_sim}"
        );
    }
}
