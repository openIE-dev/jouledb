//! Binary Hypervector Implementation - "Poor Man's GPU" for Vector Symbolic Architectures
//!
//! This module implements binary hyperdimensional computing, replacing expensive
//! float vector operations with efficient binary XOR/PopCount operations.
//!
//! Key advantages over float VSA:
//! - BIND: XOR instead of FFT convolution - O(n/64) word operations
//! - BUNDLE: Majority voting instead of element-wise addition
//! - Similarity: Hamming distance with hardware popcount
//! - Memory: 32x smaller than f32 vectors (1 bit vs 32 bits per dimension)
//!
//! In binary VSA, XOR is self-inverse: BIND(A, B) XOR B = A

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Default number of dimensions for hypervectors.
///
/// Lowered from 10,000 to 512 based on 2025-2026 research showing that
/// trainable encoders achieve equivalent accuracy at 64-512 dimensions.
/// This reduces memory per hologram from 1,250 bytes to 64 bytes (20x savings).
///
/// Use `AdaptiveDimensionConfig` to auto-scale dimensions based on workload.
pub const DEFAULT_DIMENSIONS: usize = 512;

/// Legacy dimension constant for backward compatibility
pub const LEGACY_DIMENSIONS: usize = 10000;

/// Number of bits per word (u64)
const BITS_PER_WORD: usize = 64;

/// Threshold for sparse representation (5% density)
const SPARSE_DENSITY_THRESHOLD: f64 = 0.05;

// ============================================================================
// Adaptive Dimensionality
// ============================================================================

/// Configuration for adaptive dimensionality that starts low and scales up
/// only when accuracy degrades for a given dataset.
///
/// Based on research from "Efficient Hyperdimensional Computing" (2025) and
/// "MicroHD: Accuracy-Driven Optimization for TinyML" showing that dimensions
/// as low as 64-100 can achieve equivalent accuracy with trainable encoders.
#[derive(Debug, Clone)]
pub struct AdaptiveDimensionConfig {
    /// Minimum dimensions to start with (default: 256)
    pub min_dimensions: usize,
    /// Maximum dimensions to scale up to (default: 4096)
    pub max_dimensions: usize,
    /// Target retrieval accuracy threshold (default: 0.90)
    pub accuracy_target: f64,
    /// Number of test queries to evaluate accuracy (default: 100)
    pub probe_sample_size: usize,
    /// Scale factor when increasing dimensions (default: 2.0)
    pub scale_factor: f64,
}

impl Default for AdaptiveDimensionConfig {
    fn default() -> Self {
        Self {
            min_dimensions: 256,
            max_dimensions: 4096,
            accuracy_target: 0.90,
            probe_sample_size: 100,
            scale_factor: 2.0,
        }
    }
}

impl AdaptiveDimensionConfig {
    /// Create a config optimized for low-memory environments (IoT/edge)
    pub fn low_memory() -> Self {
        Self {
            min_dimensions: 64,
            max_dimensions: 512,
            accuracy_target: 0.85,
            probe_sample_size: 50,
            scale_factor: 2.0,
        }
    }

    /// Create a config optimized for high-accuracy requirements
    pub fn high_accuracy() -> Self {
        Self {
            min_dimensions: 512,
            max_dimensions: 16384,
            accuracy_target: 0.95,
            probe_sample_size: 200,
            scale_factor: 2.0,
        }
    }

    /// Create a config matching legacy 10K dimensions (backward compat)
    pub fn legacy() -> Self {
        Self {
            min_dimensions: LEGACY_DIMENSIONS,
            max_dimensions: LEGACY_DIMENSIONS,
            accuracy_target: 0.95,
            probe_sample_size: 100,
            scale_factor: 1.0,
        }
    }

    /// Select optimal dimensions for a given item count.
    ///
    /// Uses the capacity formula: capacity ≈ sqrt(d)/2 for binary XOR binding.
    /// Given a target item count, computes the minimum dimensions needed.
    pub fn select_dimensions(&self, target_items: usize) -> usize {
        // capacity ≈ sqrt(d)/2, so d ≈ (2 * capacity)^2
        let needed = ((target_items as f64 * 2.0).powi(2)) as usize;
        // Round up to next multiple of 64 for word alignment
        let aligned = ((needed + 63) / 64) * 64;
        aligned.clamp(self.min_dimensions, self.max_dimensions)
    }

    /// Probe whether current dimensions are sufficient by testing retrieval accuracy
    /// on a sample of stored key-value pairs.
    ///
    /// Returns (current_accuracy, recommended_dimensions)
    pub fn probe_accuracy(
        &self,
        current_dims: usize,
        stored_count: usize,
        retrieval_accuracy: f64,
    ) -> (f64, usize) {
        if retrieval_accuracy >= self.accuracy_target {
            // Accuracy is sufficient, try to shrink dimensions
            let smaller =
                ((current_dims as f64 / self.scale_factor) as usize).max(self.min_dimensions);
            let smaller_aligned = ((smaller + 63) / 64) * 64;
            // Only recommend shrinking if we have significant headroom
            if retrieval_accuracy > self.accuracy_target + 0.05 && smaller_aligned < current_dims {
                return (retrieval_accuracy, smaller_aligned);
            }
            (retrieval_accuracy, current_dims)
        } else {
            // Accuracy is below target, scale up
            let larger =
                ((current_dims as f64 * self.scale_factor) as usize).min(self.max_dimensions);
            let larger_aligned = ((larger + 63) / 64) * 64;
            (retrieval_accuracy, larger_aligned)
        }
    }
}

/// Estimate optimal dimensions for a given number of items and target accuracy.
///
/// This is a convenience function that uses `AdaptiveDimensionConfig` internally.
pub fn optimal_dimensions(target_items: usize, target_accuracy: f64) -> usize {
    let config = AdaptiveDimensionConfig {
        accuracy_target: target_accuracy,
        ..Default::default()
    };
    config.select_dimensions(target_items)
}

// ============================================================================
// BinaryHyperVector - Dense Representation
// ============================================================================

/// A binary hypervector stored as packed u64 words.
///
/// Each dimension is a single bit: 0 or 1.
/// This provides 32x memory savings over f32 vectors.
#[derive(Clone, Serialize, Deserialize)]
pub struct BinaryHyperVector {
    /// Packed bits, 64 dimensions per word
    words: Vec<u64>,
    /// Total number of dimensions (may not be multiple of 64)
    dimensions: usize,
}

impl BinaryHyperVector {
    /// Create a new zero vector with the specified dimensions
    pub fn zeros(dimensions: usize) -> Self {
        let num_words = (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        Self {
            words: vec![0u64; num_words],
            dimensions,
        }
    }

    /// Create a new vector with all ones
    pub fn ones(dimensions: usize) -> Self {
        let num_words = (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut words = vec![!0u64; num_words];

        // Mask off unused bits in the last word
        let remainder = dimensions % BITS_PER_WORD;
        if remainder != 0 {
            let mask = (1u64 << remainder) - 1;
            if let Some(last) = words.last_mut() {
                *last &= mask;
            }
        }

        Self { words, dimensions }
    }

    /// Create a random binary vector using a seeded PRNG
    ///
    /// Uses a simple but fast xorshift128+ generator for reproducibility.
    pub fn random(dimensions: usize, seed: u64) -> Self {
        let num_words = (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut words = Vec::with_capacity(num_words);

        // xorshift128+ state
        let mut s0 = seed;
        let mut s1 = seed.wrapping_mul(0x9E3779B97F4A7C15);

        for _ in 0..num_words {
            // xorshift128+ iteration
            let mut x = s0;
            let y = s1;
            s0 = y;
            x ^= x << 23;
            s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
            words.push(s1.wrapping_add(y));
        }

        // Mask off unused bits in the last word
        let remainder = dimensions % BITS_PER_WORD;
        if remainder != 0 {
            let mask = (1u64 << remainder) - 1;
            if let Some(last) = words.last_mut() {
                *last &= mask;
            }
        }

        Self { words, dimensions }
    }

    /// Create from a slice of bits (0 or 1 values)
    pub fn from_bits(bits: &[u8]) -> Self {
        let dimensions = bits.len();
        let num_words = (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut words = vec![0u64; num_words];

        for (i, &bit) in bits.iter().enumerate() {
            if bit != 0 {
                let word_idx = i / BITS_PER_WORD;
                let bit_idx = i % BITS_PER_WORD;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { words, dimensions }
    }

    /// Create from raw u64 words
    pub fn from_words(words: Vec<u64>, dimensions: usize) -> Self {
        debug_assert!(words.len() >= (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD);
        Self { words, dimensions }
    }

    /// Get the number of dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the underlying words
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Get a mutable reference to the underlying words
    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// Get a single bit value
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index < self.dimensions);
        let word_idx = index / BITS_PER_WORD;
        let bit_idx = index % BITS_PER_WORD;
        (self.words[word_idx] >> bit_idx) & 1 != 0
    }

    /// Set a single bit value
    #[inline]
    pub fn set_bit(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.dimensions);
        let word_idx = index / BITS_PER_WORD;
        let bit_idx = index % BITS_PER_WORD;
        if value {
            self.words[word_idx] |= 1u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Count the number of set bits (population count)
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Get the density (fraction of bits that are 1)
    #[inline]
    pub fn density(&self) -> f64 {
        self.popcount() as f64 / self.dimensions as f64
    }

    // ========================================================================
    // Core VSA Operations
    // ========================================================================

    /// BIND operation using XOR
    ///
    /// In binary VSA, binding is simply XOR. This replaces FFT convolution
    /// and is O(n/64) word operations instead of O(n log n).
    ///
    /// Properties:
    /// - Associative: (A XOR B) XOR C = A XOR (B XOR C)
    /// - Commutative: A XOR B = B XOR A
    /// - Self-inverse: A XOR B XOR B = A
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dimensions, other.dimensions);

        let words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();

        Self {
            words,
            dimensions: self.dimensions,
        }
    }

    /// BIND operation in-place (mutates self)
    #[inline]
    pub fn bind_inplace(&mut self, other: &Self) {
        debug_assert_eq!(self.dimensions, other.dimensions);

        for (a, &b) in self.words.iter_mut().zip(other.words.iter()) {
            *a ^= b;
        }
    }

    /// UNBIND operation (same as BIND for binary - XOR is self-inverse)
    #[inline]
    pub fn unbind(&self, other: &Self) -> Self {
        self.bind(other)
    }

    /// UNBIND operation in-place
    #[inline]
    pub fn unbind_inplace(&mut self, other: &Self) {
        self.bind_inplace(other)
    }

    /// BUNDLE operation using majority voting
    ///
    /// For each dimension, count how many vectors have a 1.
    /// If majority have 1, result is 1; otherwise 0.
    /// Ties are broken randomly (using position-based hash).
    pub fn bundle(vectors: &[&Self]) -> Option<Self> {
        if vectors.is_empty() {
            return None;
        }

        let dimensions = vectors[0].dimensions;
        let num_words = vectors[0].words.len();

        // Verify all vectors have same dimensions
        for v in vectors.iter().skip(1) {
            if v.dimensions != dimensions {
                return None;
            }
        }

        let mut result_words = Vec::with_capacity(num_words);

        for word_idx in 0..num_words {
            let mut result_word = 0u64;

            for bit_idx in 0..BITS_PER_WORD {
                let global_bit = word_idx * BITS_PER_WORD + bit_idx;
                if global_bit >= dimensions {
                    break;
                }

                // Count votes for this bit
                let count: usize = vectors
                    .iter()
                    .map(|v| ((v.words[word_idx] >> bit_idx) & 1) as usize)
                    .sum();

                // Majority voting with tie-breaking
                // For n vectors, threshold is n/2 (integer division)
                // Bit is set if count > threshold (strict majority)
                // For ties (only possible when n is even and count == n/2), use tiebreaker
                let n = vectors.len();
                let bit_value = if count * 2 > n {
                    // Strict majority
                    1u64
                } else if count * 2 < n {
                    // Strict minority
                    0u64
                } else {
                    // Tie (n must be even, count == n/2)
                    // Use position-based deterministic tie-breaking
                    ((global_bit * 2654435761) % 2) as u64
                };

                result_word |= bit_value << bit_idx;
            }

            result_words.push(result_word);
        }

        Some(Self {
            words: result_words,
            dimensions,
        })
    }

    /// Weighted bundle using threshold voting
    ///
    /// Each vector has an associated weight. Sum weighted votes per dimension.
    pub fn weighted_bundle(vectors: &[(&Self, f64)]) -> Option<Self> {
        if vectors.is_empty() {
            return None;
        }

        let dimensions = vectors[0].0.dimensions;
        let num_words = vectors[0].0.words.len();
        let total_weight: f64 = vectors.iter().map(|(_, w)| w).sum();
        let threshold = total_weight / 2.0;

        // Verify all vectors have same dimensions
        for (v, _) in vectors.iter().skip(1) {
            if v.dimensions != dimensions {
                return None;
            }
        }

        let mut result_words = Vec::with_capacity(num_words);

        for word_idx in 0..num_words {
            let mut result_word = 0u64;

            for bit_idx in 0..BITS_PER_WORD {
                let global_bit = word_idx * BITS_PER_WORD + bit_idx;
                if global_bit >= dimensions {
                    break;
                }

                // Sum weighted votes for this bit
                let weighted_sum: f64 = vectors
                    .iter()
                    .map(|(v, w)| {
                        let bit = ((v.words[word_idx] >> bit_idx) & 1) as f64;
                        bit * w
                    })
                    .sum();

                let bit_value = if weighted_sum > threshold { 1u64 } else { 0u64 };
                result_word |= bit_value << bit_idx;
            }

            result_words.push(result_word);
        }

        Some(Self {
            words: result_words,
            dimensions,
        })
    }

    /// PERMUTE operation using bit rotation
    ///
    /// Rotates the entire vector by `shift` positions.
    /// Used for encoding sequences: seq(A, B, C) = A XOR rot(B, 1) XOR rot(C, 2)
    pub fn permute(&self, shift: i32) -> Self {
        if shift == 0 {
            return self.clone();
        }

        // Normalize shift to positive value
        let effective_shift = if shift < 0 {
            let s = (-shift) as usize % self.dimensions;
            if s == 0 { 0 } else { self.dimensions - s }
        } else {
            shift as usize % self.dimensions
        };

        if effective_shift == 0 {
            return self.clone();
        }

        let mut result = Self::zeros(self.dimensions);

        for i in 0..self.dimensions {
            let new_pos = (i + effective_shift) % self.dimensions;
            if self.get_bit(i) {
                result.set_bit(new_pos, true);
            }
        }

        result
    }

    /// Fast permute by exactly 1 position (common case for sequences)
    pub fn permute_one(&self) -> Self {
        self.permute(1)
    }

    /// Inverse permute (shift in opposite direction)
    pub fn unpermute(&self, shift: i32) -> Self {
        self.permute(-shift)
    }

    // ========================================================================
    // Similarity Measures
    // ========================================================================

    /// Hamming distance: count of differing bits
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        debug_assert_eq!(self.dimensions, other.dimensions);

        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }

    /// Normalized Hamming distance [0, 1]
    /// 0 = identical, 1 = completely different
    #[inline]
    pub fn hamming_distance_normalized(&self, other: &Self) -> f64 {
        self.hamming_distance(other) as f64 / self.dimensions as f64
    }

    /// Hamming similarity [0, 1]
    /// 1 = identical, 0 = completely different
    #[inline]
    pub fn hamming_similarity(&self, other: &Self) -> f64 {
        1.0 - self.hamming_distance_normalized(other)
    }

    /// Cosine-like similarity for binary vectors
    ///
    /// Maps Hamming similarity to approximate cosine similarity.
    /// For random vectors: ~0.5 Hamming similarity -> ~0 cosine
    /// For identical vectors: 1.0 Hamming similarity -> 1.0 cosine
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let hamming = self.hamming_similarity(other);
        // Map [0.5, 1.0] -> [-1.0, 1.0] (approximately)
        2.0 * hamming - 1.0
    }

    /// Jaccard similarity: |A AND B| / |A OR B|
    #[inline]
    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.dimensions, other.dimensions);

        let (intersection, union): (u32, u32) = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| ((a & b).count_ones(), (a | b).count_ones()))
            .fold((0, 0), |(acc_i, acc_u), (i, u)| (acc_i + i, acc_u + u));

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Overlap coefficient: |A AND B| / min(|A|, |B|)
    #[inline]
    pub fn overlap_coefficient(&self, other: &Self) -> f64 {
        let count_a = self.popcount();
        let count_b = other.popcount();
        let min_count = count_a.min(count_b);

        if min_count == 0 {
            return 0.0;
        }

        let intersection: u32 = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| (a & b).count_ones())
            .sum();

        intersection as f64 / min_count as f64
    }

    // ========================================================================
    // Conversions
    // ========================================================================

    /// Convert to dense f32 vector (0.0 or 1.0 values)
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimensions);

        for i in 0..self.dimensions {
            result.push(if self.get_bit(i) { 1.0 } else { 0.0 });
        }

        result
    }

    /// Convert to dense f32 vector with bipolar encoding (-1.0 or 1.0)
    pub fn to_f32_bipolar(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimensions);

        for i in 0..self.dimensions {
            result.push(if self.get_bit(i) { 1.0 } else { -1.0 });
        }

        result
    }

    /// Create from f32 vector (threshold at 0.5)
    pub fn from_f32_vec(values: &[f32]) -> Self {
        let dimensions = values.len();
        let num_words = (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut words = vec![0u64; num_words];

        for (i, &val) in values.iter().enumerate() {
            if val >= 0.5 {
                let word_idx = i / BITS_PER_WORD;
                let bit_idx = i % BITS_PER_WORD;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { words, dimensions }
    }

    /// Create from bipolar f32 vector (threshold at 0.0)
    pub fn from_f32_bipolar(values: &[f32]) -> Self {
        let dimensions = values.len();
        let num_words = (dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut words = vec![0u64; num_words];

        for (i, &val) in values.iter().enumerate() {
            if val >= 0.0 {
                let word_idx = i / BITS_PER_WORD;
                let bit_idx = i % BITS_PER_WORD;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { words, dimensions }
    }

    /// Convert to sparse representation if density is below threshold
    pub fn to_sparse(&self) -> SparseBinaryHyperVector {
        SparseBinaryHyperVector::from_dense(self)
    }

    /// Check if this vector would benefit from sparse representation
    pub fn should_be_sparse(&self) -> bool {
        self.density() < SPARSE_DENSITY_THRESHOLD
    }
}

impl fmt::Debug for BinaryHyperVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryHyperVector")
            .field("dimensions", &self.dimensions)
            .field("density", &format!("{:.2}%", self.density() * 100.0))
            .field("popcount", &self.popcount())
            .finish()
    }
}

impl PartialEq for BinaryHyperVector {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.words == other.words
    }
}

impl Eq for BinaryHyperVector {}

impl Hash for BinaryHyperVector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dimensions.hash(state);
        self.words.hash(state);
    }
}

// ============================================================================
// SparseBinaryHyperVector - For Low Density Vectors
// ============================================================================

/// Sparse representation for binary hypervectors with low density (<5%)
///
/// Stores only the indices of set bits. More memory efficient when
/// less than ~3% of bits are set.
#[derive(Clone, Debug)]
pub struct SparseBinaryHyperVector {
    /// Indices of set bits (sorted)
    set_bits: Vec<u32>,
    /// Total number of dimensions
    dimensions: usize,
}

impl SparseBinaryHyperVector {
    /// Create a new empty sparse vector
    pub fn new(dimensions: usize) -> Self {
        Self {
            set_bits: Vec::new(),
            dimensions,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(dimensions: usize, capacity: usize) -> Self {
        Self {
            set_bits: Vec::with_capacity(capacity),
            dimensions,
        }
    }

    /// Create from a list of set bit indices
    pub fn from_indices(dimensions: usize, mut indices: Vec<u32>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        indices.retain(|&i| (i as usize) < dimensions);

        Self {
            set_bits: indices,
            dimensions,
        }
    }

    /// Create from a dense BinaryHyperVector
    pub fn from_dense(dense: &BinaryHyperVector) -> Self {
        let mut set_bits = Vec::new();

        for (word_idx, &word) in dense.words.iter().enumerate() {
            if word == 0 {
                continue;
            }

            let base = (word_idx * BITS_PER_WORD) as u32;
            let mut w = word;

            while w != 0 {
                let bit_pos = w.trailing_zeros();
                let global_idx = base + bit_pos;

                if (global_idx as usize) < dense.dimensions {
                    set_bits.push(global_idx);
                }

                w &= w - 1; // Clear lowest set bit
            }
        }

        Self {
            set_bits,
            dimensions: dense.dimensions,
        }
    }

    /// Get the number of dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the indices of set bits
    #[inline]
    pub fn set_bits(&self) -> &[u32] {
        &self.set_bits
    }

    /// Get the number of set bits (popcount)
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.set_bits.len() as u32
    }

    /// Get the density
    #[inline]
    pub fn density(&self) -> f64 {
        self.set_bits.len() as f64 / self.dimensions as f64
    }

    /// Check if a specific bit is set
    pub fn get_bit(&self, index: usize) -> bool {
        self.set_bits.binary_search(&(index as u32)).is_ok()
    }

    /// Set a bit (maintains sorted order)
    pub fn set_bit(&mut self, index: usize) {
        let idx = index as u32;
        if let Err(pos) = self.set_bits.binary_search(&idx) {
            self.set_bits.insert(pos, idx);
        }
    }

    /// Clear a bit
    pub fn clear_bit(&mut self, index: usize) {
        let idx = index as u32;
        if let Ok(pos) = self.set_bits.binary_search(&idx) {
            self.set_bits.remove(pos);
        }
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> BinaryHyperVector {
        let num_words = (self.dimensions + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut words = vec![0u64; num_words];

        for &idx in &self.set_bits {
            let word_idx = idx as usize / BITS_PER_WORD;
            let bit_idx = idx as usize % BITS_PER_WORD;
            words[word_idx] |= 1u64 << bit_idx;
        }

        BinaryHyperVector {
            words,
            dimensions: self.dimensions,
        }
    }

    /// BIND operation (XOR) for sparse vectors
    ///
    /// Result may not be sparse - returns dense vector
    pub fn bind(&self, other: &Self) -> BinaryHyperVector {
        self.to_dense().bind(&other.to_dense())
    }

    /// Hamming distance to another sparse vector
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        debug_assert_eq!(self.dimensions, other.dimensions);

        // Count symmetric difference: bits in one but not both
        let mut i = 0;
        let mut j = 0;
        let mut distance = 0u32;

        while i < self.set_bits.len() && j < other.set_bits.len() {
            match self.set_bits[i].cmp(&other.set_bits[j]) {
                std::cmp::Ordering::Less => {
                    distance += 1;
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    distance += 1;
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }

        // Add remaining elements
        distance += (self.set_bits.len() - i) as u32;
        distance += (other.set_bits.len() - j) as u32;

        distance
    }

    /// Hamming similarity [0, 1]
    pub fn hamming_similarity(&self, other: &Self) -> f64 {
        1.0 - (self.hamming_distance(other) as f64 / self.dimensions as f64)
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.set_bits.len() * std::mem::size_of::<u32>()
    }

    /// Check if sparse representation is more memory efficient than dense
    pub fn is_memory_efficient(&self) -> bool {
        let dense_bytes = (self.dimensions + 7) / 8;
        self.memory_bytes() < dense_bytes
    }
}

impl PartialEq for SparseBinaryHyperVector {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.set_bits == other.set_bits
    }
}

impl Eq for SparseBinaryHyperVector {}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch operations for efficient processing of multiple vectors
pub struct BatchOps;

impl BatchOps {
    /// Bind multiple vectors together: A XOR B XOR C XOR ...
    pub fn bind_all(vectors: &[&BinaryHyperVector]) -> Option<BinaryHyperVector> {
        if vectors.is_empty() {
            return None;
        }

        let dimensions = vectors[0].dimensions;

        // Verify all vectors have same dimensions
        for v in vectors.iter().skip(1) {
            if v.dimensions != dimensions {
                return None;
            }
        }

        let mut result_words = vectors[0].words.clone();

        for v in vectors.iter().skip(1) {
            for (r, &w) in result_words.iter_mut().zip(v.words.iter()) {
                *r ^= w;
            }
        }

        Some(BinaryHyperVector {
            words: result_words,
            dimensions,
        })
    }

    /// Compute pairwise Hamming distances
    ///
    /// Returns a symmetric matrix where entry [i][j] is the distance between vectors[i] and vectors[j]
    pub fn pairwise_distances(vectors: &[&BinaryHyperVector]) -> Vec<Vec<u32>> {
        let n = vectors.len();
        let mut distances = vec![vec![0u32; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = vectors[i].hamming_distance(vectors[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Find the most similar vector to a query
    ///
    /// Returns (index, similarity) of the best match
    pub fn find_nearest(
        query: &BinaryHyperVector,
        candidates: &[&BinaryHyperVector],
    ) -> Option<(usize, f64)> {
        if candidates.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = query.hamming_distance(candidates[0]);

        for (i, &candidate) in candidates.iter().enumerate().skip(1) {
            let dist = query.hamming_distance(candidate);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        let similarity = 1.0 - (best_dist as f64 / query.dimensions as f64);
        Some((best_idx, similarity))
    }

    /// Find k nearest neighbors
    ///
    /// Returns vector of (index, similarity) sorted by similarity descending
    pub fn find_k_nearest(
        query: &BinaryHyperVector,
        candidates: &[&BinaryHyperVector],
        k: usize,
    ) -> Vec<(usize, f64)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, u32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, query.hamming_distance(c)))
            .collect();

        // Partial sort to get k smallest distances
        let k = k.min(scored.len());
        scored.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
        scored.truncate(k);
        scored.sort_by_key(|&(_, d)| d);

        scored
            .into_iter()
            .map(|(i, d)| {
                let sim = 1.0 - (d as f64 / query.dimensions as f64);
                (i, sim)
            })
            .collect()
    }

    /// Bundle multiple vector sets in parallel
    ///
    /// Each inner Vec contains vectors to bundle together
    pub fn bundle_batch(vector_sets: &[Vec<&BinaryHyperVector>]) -> Vec<Option<BinaryHyperVector>> {
        vector_sets
            .iter()
            .map(|set| BinaryHyperVector::bundle(set))
            .collect()
    }

    /// GPU-accelerated nearest neighbor search.
    ///
    /// Dispatches similarity computation to GPU when candidate count exceeds
    /// the dispatcher's threshold. Falls back to CPU otherwise.
    ///
    /// Returns `(index, similarity)` of the best match.
    pub fn find_nearest_gpu(
        query: &BinaryHyperVector,
        candidates: &[&BinaryHyperVector],
        gpu: &crate::gpu_dispatch::GpuDispatcher,
    ) -> Option<(usize, f64)> {
        if candidates.is_empty() {
            return None;
        }

        if !gpu.should_dispatch_similarity(candidates.len()) {
            return Self::find_nearest(query, candidates);
        }

        let dim_u64 = query.words.len();

        // Flatten candidate words into a contiguous buffer
        let mut flat_vectors = Vec::with_capacity(candidates.len() * dim_u64);
        for c in candidates {
            flat_vectors.extend_from_slice(&c.words);
        }

        match gpu.batch_similarity(&query.words, &flat_vectors, candidates.len(), dim_u64) {
            Ok(scores) => {
                let mut best_idx = 0;
                let mut best_score = 0u32;
                for (i, &score) in scores.iter().enumerate() {
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
                let similarity = best_score as f64 / query.dimensions as f64;
                Some((best_idx, similarity))
            }
            Err(_) => {
                // GPU dispatch failed, fall back to CPU
                Self::find_nearest(query, candidates)
            }
        }
    }

    /// GPU-accelerated k-nearest neighbor search.
    ///
    /// Returns vector of `(index, similarity)` sorted by similarity descending.
    pub fn find_k_nearest_gpu(
        query: &BinaryHyperVector,
        candidates: &[&BinaryHyperVector],
        k: usize,
        gpu: &crate::gpu_dispatch::GpuDispatcher,
    ) -> Vec<(usize, f64)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        if !gpu.should_dispatch_similarity(candidates.len()) {
            return Self::find_k_nearest(query, candidates, k);
        }

        let dim_u64 = query.words.len();

        let mut flat_vectors = Vec::with_capacity(candidates.len() * dim_u64);
        for c in candidates {
            flat_vectors.extend_from_slice(&c.words);
        }

        match gpu.batch_similarity(&query.words, &flat_vectors, candidates.len(), dim_u64) {
            Ok(scores) => {
                let total_bits = query.dimensions;
                let k = k.min(scores.len());

                // Convert to (index, distance) for sorting
                let mut scored: Vec<(usize, u32)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &sim_score)| {
                        // distance = total_bits - similarity_score
                        let dist = (total_bits as u32).saturating_sub(sim_score);
                        (i, dist)
                    })
                    .collect();

                scored.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
                scored.truncate(k);
                scored.sort_by_key(|&(_, d)| d);

                scored
                    .into_iter()
                    .map(|(i, d)| {
                        let sim = 1.0 - (d as f64 / total_bits as f64);
                        (i, sim)
                    })
                    .collect()
            }
            Err(_) => Self::find_k_nearest(query, candidates, k),
        }
    }

    /// GPU-accelerated pairwise Hamming distances.
    ///
    /// For N vectors, computes similarity of each vector against all others
    /// using N GPU dispatches. Returns symmetric distance matrix.
    pub fn pairwise_distances_gpu(
        vectors: &[&BinaryHyperVector],
        gpu: &crate::gpu_dispatch::GpuDispatcher,
    ) -> Vec<Vec<u32>> {
        let n = vectors.len();
        if n < 2 || !gpu.should_dispatch_similarity(n) {
            return Self::pairwise_distances(vectors);
        }

        let dim_u64 = vectors[0].words.len();
        let total_bits = vectors[0].dimensions;

        // Flatten all vectors
        let mut flat_vectors = Vec::with_capacity(n * dim_u64);
        for v in vectors {
            flat_vectors.extend_from_slice(&v.words);
        }

        let mut distances = vec![vec![0u32; n]; n];

        for i in 0..n {
            match gpu.batch_similarity(&vectors[i].words, &flat_vectors, n, dim_u64) {
                Ok(scores) => {
                    for j in 0..n {
                        let dist = (total_bits as u32).saturating_sub(scores[j]);
                        distances[i][j] = dist;
                    }
                }
                Err(_) => {
                    // Fall back to CPU for this row
                    for j in 0..n {
                        distances[i][j] = vectors[i].hamming_distance(vectors[j]);
                    }
                }
            }
        }

        distances
    }
}

// ============================================================================
// Codebook for Item Memory
// ============================================================================

/// A codebook mapping symbols to random hypervectors
pub struct Codebook {
    vectors: Vec<BinaryHyperVector>,
    dimensions: usize,
}

impl Codebook {
    /// Create a new codebook with random vectors
    pub fn new(num_symbols: usize, dimensions: usize, base_seed: u64) -> Self {
        let vectors: Vec<BinaryHyperVector> = (0..num_symbols)
            .map(|i| BinaryHyperVector::random(dimensions, base_seed.wrapping_add(i as u64)))
            .collect();

        Self {
            vectors,
            dimensions,
        }
    }

    /// Get vector for a symbol
    pub fn get(&self, symbol: usize) -> Option<&BinaryHyperVector> {
        self.vectors.get(symbol)
    }

    /// Number of symbols in the codebook
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if codebook is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Encode a sequence of symbols using permutation
    ///
    /// seq(A, B, C) = A XOR rot(B, 1) XOR rot(C, 2)
    pub fn encode_sequence(&self, symbols: &[usize]) -> Option<BinaryHyperVector> {
        if symbols.is_empty() {
            return None;
        }

        let mut result = self.get(symbols[0])?.clone();

        for (i, &sym) in symbols.iter().enumerate().skip(1) {
            let vec = self.get(sym)?.permute(i as i32);
            result.bind_inplace(&vec);
        }

        Some(result)
    }

    /// Encode a set of symbols (order-independent)
    pub fn encode_set(&self, symbols: &[usize]) -> Option<BinaryHyperVector> {
        if symbols.is_empty() {
            return None;
        }

        let vectors: Vec<&BinaryHyperVector> =
            symbols.iter().filter_map(|&s| self.get(s)).collect();

        if vectors.len() != symbols.len() {
            return None;
        }

        BinaryHyperVector::bundle(&vectors)
    }

    /// Find the most similar symbol to a query vector
    pub fn lookup(&self, query: &BinaryHyperVector) -> Option<(usize, f64)> {
        let refs: Vec<&BinaryHyperVector> = self.vectors.iter().collect();
        BatchOps::find_nearest(query, &refs)
    }

    /// Verify orthogonality of codebook vectors
    ///
    /// Returns (mean_similarity, max_similarity, min_similarity)
    pub fn verify_orthogonality(&self) -> (f64, f64, f64) {
        if self.vectors.len() < 2 {
            return (0.5, 0.5, 0.5);
        }

        let mut sum = 0.0;
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut count = 0;

        for i in 0..self.vectors.len() {
            for j in (i + 1)..self.vectors.len() {
                let sim = self.vectors[i].hamming_similarity(&self.vectors[j]);
                sum += sim;
                max = max.max(sim);
                min = min.min(sim);
                count += 1;
            }
        }

        let mean = sum / count as f64;
        (mean, max, min)
    }
}

// ============================================================================
// Associative Memory (Clean-up Memory)
// ============================================================================

/// Associative memory for storing and retrieving hypervectors
pub struct AssociativeMemory {
    items: Vec<BinaryHyperVector>,
    dimensions: usize,
}

impl AssociativeMemory {
    /// Create a new empty associative memory
    pub fn new(dimensions: usize) -> Self {
        Self {
            items: Vec::new(),
            dimensions,
        }
    }

    /// Store a vector in memory
    pub fn store(&mut self, vector: BinaryHyperVector) {
        if vector.dimensions == self.dimensions {
            self.items.push(vector);
        }
    }

    /// Retrieve the closest matching vector
    pub fn retrieve(&self, query: &BinaryHyperVector) -> Option<&BinaryHyperVector> {
        if self.items.is_empty() {
            return None;
        }

        let refs: Vec<&BinaryHyperVector> = self.items.iter().collect();
        let (idx, _) = BatchOps::find_nearest(query, &refs)?;
        Some(&self.items[idx])
    }

    /// Retrieve with similarity threshold
    pub fn retrieve_if_similar(
        &self,
        query: &BinaryHyperVector,
        min_similarity: f64,
    ) -> Option<&BinaryHyperVector> {
        if self.items.is_empty() {
            return None;
        }

        let refs: Vec<&BinaryHyperVector> = self.items.iter().collect();
        let (idx, sim) = BatchOps::find_nearest(query, &refs)?;

        if sim >= min_similarity {
            Some(&self.items[idx])
        } else {
            None
        }
    }

    /// Number of stored items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Get all stored vectors
    pub fn items(&self) -> &[BinaryHyperVector] {
        &self.items
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generate a random seed from system time and counter
pub fn generate_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    time ^ (count << 32) ^ (count >> 32)
}

/// Estimate capacity of associative memory
///
/// Returns approximate number of items that can be stored while maintaining
/// retrieval accuracy above the given threshold.
pub fn estimate_capacity(dimensions: usize, accuracy_threshold: f64) -> usize {
    // Based on theoretical analysis of binary VSA capacity
    // C ≈ d / (4 * ln(1/p)) where p is error probability
    let error_prob = 1.0 - accuracy_threshold;
    if error_prob <= 0.0 {
        return 1;
    }

    let ln_inv_p = (-error_prob.ln()).max(1.0);
    let capacity = dimensions as f64 / (4.0 * ln_inv_p);

    (capacity as usize).max(1)
}

/// Compute theoretical expected Hamming similarity for random vectors
pub fn expected_random_similarity(_dimensions: usize) -> f64 {
    // For random binary vectors, expected overlap is 50%
    0.5
}

// ============================================================================
// SIMD-Optimized Operations (using portable intrinsics)
// ============================================================================

/// SIMD-optimized popcount for batch distance computation
pub mod simd {
    use super::*;

    /// Compute Hamming distance using SIMD-friendly operations
    ///
    /// This version processes multiple words in parallel when possible
    #[inline]
    pub fn hamming_distance_fast(a: &BinaryHyperVector, b: &BinaryHyperVector) -> u32 {
        debug_assert_eq!(a.dimensions, b.dimensions);

        // Process 4 words at a time for better pipelining
        let chunks = a.words.len() / 4;
        let mut total = 0u32;

        // Main loop - 4 words per iteration
        for i in 0..chunks {
            let base = i * 4;
            let d0 = (a.words[base] ^ b.words[base]).count_ones();
            let d1 = (a.words[base + 1] ^ b.words[base + 1]).count_ones();
            let d2 = (a.words[base + 2] ^ b.words[base + 2]).count_ones();
            let d3 = (a.words[base + 3] ^ b.words[base + 3]).count_ones();
            total += d0 + d1 + d2 + d3;
        }

        // Handle remaining words
        for i in (chunks * 4)..a.words.len() {
            total += (a.words[i] ^ b.words[i]).count_ones();
        }

        total
    }

    /// Batch XOR operation for binding multiple pairs
    pub fn bind_batch(
        pairs: &[(&BinaryHyperVector, &BinaryHyperVector)],
    ) -> Vec<BinaryHyperVector> {
        pairs.iter().map(|(a, b)| a.bind(b)).collect()
    }

    /// Compute multiple distances in parallel
    pub fn distances_to_query(
        query: &BinaryHyperVector,
        candidates: &[&BinaryHyperVector],
    ) -> Vec<u32> {
        candidates
            .iter()
            .map(|&c| hamming_distance_fast(query, c))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Basic creation and properties
    #[test]
    fn test_basic_creation() {
        let zeros = BinaryHyperVector::zeros(1000);
        assert_eq!(zeros.dimensions(), 1000);
        assert_eq!(zeros.popcount(), 0);
        assert_eq!(zeros.density(), 0.0);

        let ones = BinaryHyperVector::ones(1000);
        assert_eq!(ones.dimensions(), 1000);
        assert_eq!(ones.popcount(), 1000);
        assert_eq!(ones.density(), 1.0);

        let random = BinaryHyperVector::random(1000, 42);
        assert_eq!(random.dimensions(), 1000);
        // Random vector should have ~50% density
        let density = random.density();
        assert!(density > 0.4 && density < 0.6, "Density: {}", density);
    }

    // Test 2: Bit manipulation
    #[test]
    fn test_bit_operations() {
        let mut v = BinaryHyperVector::zeros(100);

        assert!(!v.get_bit(0));
        assert!(!v.get_bit(50));
        assert!(!v.get_bit(99));

        v.set_bit(0, true);
        v.set_bit(50, true);
        v.set_bit(99, true);

        assert!(v.get_bit(0));
        assert!(v.get_bit(50));
        assert!(v.get_bit(99));
        assert_eq!(v.popcount(), 3);

        v.set_bit(50, false);
        assert!(!v.get_bit(50));
        assert_eq!(v.popcount(), 2);
    }

    // Test 3: BIND operation (XOR)
    #[test]
    fn test_bind_operation() {
        let a = BinaryHyperVector::random(1000, 1);
        let b = BinaryHyperVector::random(1000, 2);

        // BIND produces a new vector
        let bound = a.bind(&b);
        assert_eq!(bound.dimensions(), 1000);

        // BIND is commutative
        let bound2 = b.bind(&a);
        assert_eq!(bound, bound2);

        // BIND is self-inverse: (A XOR B) XOR B = A
        let recovered = bound.unbind(&b);
        assert_eq!(recovered, a);

        // A XOR A = 0
        let self_bind = a.bind(&a);
        assert_eq!(self_bind.popcount(), 0);
    }

    // Test 4: UNBIND operation
    #[test]
    fn test_unbind_operation() {
        let a = BinaryHyperVector::random(1000, 10);
        let b = BinaryHyperVector::random(1000, 20);
        let c = BinaryHyperVector::random(1000, 30);

        // Bind A with B
        let ab = a.bind(&b);

        // Unbind B to recover A
        let recovered_a = ab.unbind(&b);
        assert_eq!(recovered_a, a);

        // Chain: (A XOR B XOR C) XOR C XOR B = A
        let abc = a.bind(&b).bind(&c);
        let recovered = abc.unbind(&c).unbind(&b);
        assert_eq!(recovered, a);
    }

    // Test 5: BUNDLE operation (majority voting)
    #[test]
    fn test_bundle_operation() {
        let v1 = BinaryHyperVector::random(1000, 100);
        let v2 = BinaryHyperVector::random(1000, 200);
        let v3 = BinaryHyperVector::random(1000, 300);

        let bundle = BinaryHyperVector::bundle(&[&v1, &v2, &v3]).unwrap();
        assert_eq!(bundle.dimensions(), 1000);

        // Bundle should be more similar to its components than to random vectors
        let random = BinaryHyperVector::random(1000, 999);

        let sim_v1 = bundle.hamming_similarity(&v1);
        let sim_v2 = bundle.hamming_similarity(&v2);
        let sim_v3 = bundle.hamming_similarity(&v3);
        let sim_random = bundle.hamming_similarity(&random);

        // Each component should have >50% similarity with bundle
        assert!(sim_v1 > 0.5, "sim_v1: {}", sim_v1);
        assert!(sim_v2 > 0.5, "sim_v2: {}", sim_v2);
        assert!(sim_v3 > 0.5, "sim_v3: {}", sim_v3);

        // Random should be ~50%
        assert!(
            sim_random > 0.45 && sim_random < 0.55,
            "sim_random: {}",
            sim_random
        );
    }

    // Test 6: Weighted bundle
    #[test]
    fn test_weighted_bundle() {
        let v1 = BinaryHyperVector::random(1000, 100);
        let v2 = BinaryHyperVector::random(1000, 200);

        // Heavy weight on v1
        let bundle = BinaryHyperVector::weighted_bundle(&[(&v1, 10.0), (&v2, 1.0)]).unwrap();

        let sim_v1 = bundle.hamming_similarity(&v1);
        let sim_v2 = bundle.hamming_similarity(&v2);

        // v1 should have higher similarity due to higher weight
        assert!(sim_v1 > sim_v2, "sim_v1: {}, sim_v2: {}", sim_v1, sim_v2);
    }

    // Test 7: PERMUTE operation
    #[test]
    fn test_permute_operation() {
        let v = BinaryHyperVector::random(1000, 42);

        // Permute and unpermute should recover original
        let permuted = v.permute(7);
        let recovered = permuted.unpermute(7);
        assert_eq!(recovered, v);

        // Permuted vector should be dissimilar to original
        let sim = v.hamming_similarity(&permuted);
        assert!(sim > 0.45 && sim < 0.55, "sim: {}", sim);

        // Zero permute should return same vector
        let same = v.permute(0);
        assert_eq!(same, v);

        // Full rotation should return same vector
        let full = v.permute(v.dimensions() as i32);
        assert_eq!(full, v);
    }

    // Test 8: Similarity measures
    #[test]
    fn test_similarity_measures() {
        let v1 = BinaryHyperVector::random(1000, 1);
        let v2 = v1.clone();
        let v3 = BinaryHyperVector::random(1000, 2);

        // Identical vectors
        assert_eq!(v1.hamming_distance(&v2), 0);
        assert_eq!(v1.hamming_similarity(&v2), 1.0);
        assert_eq!(v1.cosine_similarity(&v2), 1.0);

        // Random vectors should have ~50% similarity
        let sim = v1.hamming_similarity(&v3);
        assert!(sim > 0.45 && sim < 0.55, "sim: {}", sim);

        // Cosine should be ~0 for random vectors
        let cos = v1.cosine_similarity(&v3);
        assert!(cos > -0.15 && cos < 0.15, "cos: {}", cos);
    }

    // Test 9: Orthogonality verification (random vectors ~50% overlap)
    #[test]
    fn test_orthogonality() {
        let dimensions = 10000;
        let num_vectors = 100;

        let vectors: Vec<BinaryHyperVector> = (0..num_vectors)
            .map(|i| BinaryHyperVector::random(dimensions, i as u64))
            .collect();

        let mut similarities = Vec::new();
        for i in 0..num_vectors {
            for j in (i + 1)..num_vectors {
                let sim = vectors[i].hamming_similarity(&vectors[j]);
                similarities.push(sim);
            }
        }

        let mean_sim: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let max_sim = similarities.iter().cloned().fold(f64::MIN, f64::max);
        let min_sim = similarities.iter().cloned().fold(f64::MAX, f64::min);

        // Mean should be very close to 0.5
        assert!((mean_sim - 0.5).abs() < 0.01, "mean_sim: {}", mean_sim);

        // Max and min should be within reasonable bounds
        assert!(max_sim < 0.55, "max_sim: {}", max_sim);
        assert!(min_sim > 0.45, "min_sim: {}", min_sim);
    }

    // Test 10: Capacity test (how many items before retrieval fails)
    #[test]
    fn test_capacity() {
        let dimensions = 10000;
        let mut memory = AssociativeMemory::new(dimensions);

        // Store increasing number of items and test retrieval
        let mut items = Vec::new();
        let mut retrieval_accuracy = 1.0;
        let max_items = 100;

        for i in 0..max_items {
            let v = BinaryHyperVector::random(dimensions, i as u64 * 1000);
            items.push(v.clone());
            memory.store(v);

            // Test retrieval of all items
            let mut correct = 0;
            for (j, item) in items.iter().enumerate() {
                // Add small noise to query
                let mut query = item.clone();
                for k in 0..50 {
                    let bit_idx = (j * 100 + k) % dimensions;
                    query.set_bit(bit_idx, !query.get_bit(bit_idx));
                }

                if let Some(retrieved) = memory.retrieve(&query) {
                    if retrieved == item {
                        correct += 1;
                    }
                }
            }

            retrieval_accuracy = correct as f64 / items.len() as f64;

            // Stop if accuracy drops below threshold
            if retrieval_accuracy < 0.9 {
                break;
            }
        }

        // Should be able to store at least 50 items with high accuracy
        assert!(items.len() >= 50, "Capacity too low: {}", items.len());
    }

    // Test 11: Sparse representation
    #[test]
    fn test_sparse_representation() {
        // Create a sparse vector (low density)
        let mut dense = BinaryHyperVector::zeros(10000);
        for i in (0..10000).step_by(200) {
            dense.set_bit(i, true);
        }

        let sparse = dense.to_sparse();
        assert_eq!(sparse.dimensions(), 10000);
        assert_eq!(sparse.popcount(), dense.popcount());
        assert!(sparse.density() < SPARSE_DENSITY_THRESHOLD);

        // Convert back to dense
        let recovered = sparse.to_dense();
        assert_eq!(recovered, dense);

        // Sparse should be memory efficient for low density
        assert!(sparse.is_memory_efficient());
    }

    // Test 12: Sparse vs dense operations
    #[test]
    fn test_sparse_vs_dense() {
        let dimensions = 10000;

        // Create two sparse vectors
        let mut dense1 = BinaryHyperVector::zeros(dimensions);
        let mut dense2 = BinaryHyperVector::zeros(dimensions);

        for i in (0..dimensions).step_by(100) {
            dense1.set_bit(i, true);
        }
        for i in (50..dimensions).step_by(100) {
            dense2.set_bit(i, true);
        }

        let sparse1 = dense1.to_sparse();
        let sparse2 = dense2.to_sparse();

        // Hamming distance should match
        let dist_dense = dense1.hamming_distance(&dense2);
        let dist_sparse = sparse1.hamming_distance(&sparse2);
        assert_eq!(dist_dense, dist_sparse);

        // Similarity should match
        let sim_dense = dense1.hamming_similarity(&dense2);
        let sim_sparse = sparse1.hamming_similarity(&sparse2);
        assert!((sim_dense - sim_sparse).abs() < 0.0001);
    }

    // Test 13: Batch operations
    #[test]
    fn test_batch_operations() {
        let v1 = BinaryHyperVector::random(1000, 1);
        let v2 = BinaryHyperVector::random(1000, 2);
        let v3 = BinaryHyperVector::random(1000, 3);

        // Batch bind
        let bound = BatchOps::bind_all(&[&v1, &v2, &v3]).unwrap();
        let manual = v1.bind(&v2).bind(&v3);
        assert_eq!(bound, manual);

        // Pairwise distances
        let distances = BatchOps::pairwise_distances(&[&v1, &v2, &v3]);
        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0][0], 0); // Self distance
        assert_eq!(distances[0][1], distances[1][0]); // Symmetric

        // Find nearest
        let query = v1.clone();
        let (idx, sim) = BatchOps::find_nearest(&query, &[&v1, &v2, &v3]).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(sim, 1.0);

        // Find k nearest
        let k_nearest = BatchOps::find_k_nearest(&query, &[&v1, &v2, &v3], 2);
        assert_eq!(k_nearest.len(), 2);
        assert_eq!(k_nearest[0].0, 0); // v1 is most similar
    }

    // Test 14: Codebook operations
    #[test]
    fn test_codebook() {
        let codebook = Codebook::new(26, 10000, 12345);
        assert_eq!(codebook.len(), 26);

        // Verify orthogonality
        let (mean, max, min) = codebook.verify_orthogonality();
        assert!((mean - 0.5).abs() < 0.02, "mean: {}", mean);
        assert!(max < 0.55, "max: {}", max);
        assert!(min > 0.45, "min: {}", min);

        // Encode sequence
        let seq = codebook.encode_sequence(&[0, 1, 2]).unwrap();
        assert_eq!(seq.dimensions(), 10000);

        // Encode set
        let set = codebook.encode_set(&[0, 1, 2]).unwrap();
        assert_eq!(set.dimensions(), 10000);

        // Lookup should find closest
        let v0 = codebook.get(0).unwrap().clone();
        let (idx, sim) = codebook.lookup(&v0).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(sim, 1.0);
    }

    // Test 15: Conversion to/from f32 vectors
    #[test]
    fn test_f32_conversion() {
        let binary = BinaryHyperVector::random(100, 42);

        // To f32 (0.0 or 1.0)
        let f32_vec = binary.to_f32_vec();
        assert_eq!(f32_vec.len(), 100);
        for &v in &f32_vec {
            assert!(v == 0.0 || v == 1.0);
        }

        // To bipolar (-1.0 or 1.0)
        let bipolar = binary.to_f32_bipolar();
        assert_eq!(bipolar.len(), 100);
        for &v in &bipolar {
            assert!(v == -1.0 || v == 1.0);
        }

        // From f32
        let recovered = BinaryHyperVector::from_f32_vec(&f32_vec);
        assert_eq!(recovered, binary);

        // From bipolar
        let recovered_bipolar = BinaryHyperVector::from_f32_bipolar(&bipolar);
        assert_eq!(recovered_bipolar, binary);
    }

    // Test 16: SIMD-optimized distance
    #[test]
    fn test_simd_distance() {
        let v1 = BinaryHyperVector::random(10000, 1);
        let v2 = BinaryHyperVector::random(10000, 2);

        let dist_normal = v1.hamming_distance(&v2);
        let dist_simd = simd::hamming_distance_fast(&v1, &v2);

        assert_eq!(dist_normal, dist_simd);
    }

    // Test 17: Sequence encoding with permutation
    #[test]
    fn test_sequence_encoding() {
        let codebook = Codebook::new(10, 10000, 999);

        // Different sequences should produce different vectors
        let seq1 = codebook.encode_sequence(&[0, 1, 2]).unwrap();
        let seq2 = codebook.encode_sequence(&[2, 1, 0]).unwrap();
        let seq3 = codebook.encode_sequence(&[0, 1, 2]).unwrap();

        // Same sequence should produce same vector
        assert_eq!(seq1, seq3);

        // Different order should produce different (dissimilar) vectors
        let sim = seq1.hamming_similarity(&seq2);
        assert!(sim < 0.55, "Reversed sequences too similar: {}", sim);
    }

    // Test 18: Associative memory with threshold
    #[test]
    fn test_associative_memory_threshold() {
        let dimensions = 10000;
        let mut memory = AssociativeMemory::new(dimensions);

        let v = BinaryHyperVector::random(dimensions, 42);
        memory.store(v.clone());

        // Exact match should pass high threshold
        let result = memory.retrieve_if_similar(&v, 0.99);
        assert!(result.is_some());

        // Noisy query should fail high threshold
        let mut noisy = v.clone();
        for i in 0..2000 {
            noisy.set_bit(i, !noisy.get_bit(i));
        }
        let result = memory.retrieve_if_similar(&noisy, 0.99);
        assert!(result.is_none());

        // But should pass lower threshold
        let result = memory.retrieve_if_similar(&noisy, 0.7);
        assert!(result.is_some());
    }

    // Test 19: Edge cases
    #[test]
    fn test_edge_cases() {
        // Very small dimensions
        let small = BinaryHyperVector::random(7, 42);
        assert_eq!(small.dimensions(), 7);

        let small2 = BinaryHyperVector::random(7, 43);
        let bound = small.bind(&small2);
        assert_eq!(bound.dimensions(), 7);

        // Single dimension
        let single = BinaryHyperVector::random(1, 1);
        assert_eq!(single.dimensions(), 1);

        // Empty bundle should return None
        let empty: Vec<&BinaryHyperVector> = Vec::new();
        assert!(BinaryHyperVector::bundle(&empty).is_none());

        // Single vector bundle
        let v = BinaryHyperVector::random(100, 42);
        let bundle = BinaryHyperVector::bundle(&[&v]).unwrap();
        assert_eq!(bundle, v);
    }

    // Test 20: Memory efficiency
    #[test]
    fn test_memory_efficiency() {
        let dimensions = 10000;

        // Dense vector memory
        let dense = BinaryHyperVector::random(dimensions, 42);
        let dense_words = dense.words().len();
        let dense_bytes = dense_words * 8;

        // Should be ~10000/64 * 8 = ~1250 bytes
        assert!(dense_bytes < 2000);
        assert!(dense_bytes > 1000);

        // Sparse vector with very low density
        let mut sparse_dense = BinaryHyperVector::zeros(dimensions);
        for i in (0..dimensions).step_by(500) {
            sparse_dense.set_bit(i, true);
        }
        let sparse = sparse_dense.to_sparse();

        // Sparse should use less memory
        let sparse_bytes = sparse.memory_bytes();
        assert!(sparse_bytes < dense_bytes);
        assert!(sparse.is_memory_efficient());
    }

    // Test 21: Deterministic random generation
    #[test]
    fn test_deterministic_random() {
        let v1 = BinaryHyperVector::random(1000, 12345);
        let v2 = BinaryHyperVector::random(1000, 12345);
        let v3 = BinaryHyperVector::random(1000, 12346);

        // Same seed should produce same vector
        assert_eq!(v1, v2);

        // Different seed should produce different vector
        assert_ne!(v1, v3);

        // But they should still have ~50% similarity
        let sim = v1.hamming_similarity(&v3);
        assert!(sim > 0.45 && sim < 0.55);
    }
}
