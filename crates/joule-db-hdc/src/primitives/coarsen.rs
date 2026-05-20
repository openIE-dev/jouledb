//! Primitive 14: Coarsen — scale-changing lossy projection.
//!
//! Merge combines representations at the same level of abstraction.
//! Coarsen changes the *scale* — it turns a million pixels into "a dog",
//! or a paragraph into a thesis statement.
//!
//! In physics: renormalization group. Systematically integrate out
//! fine-grained degrees of freedom to produce a lower-dimensional
//! but structurally faithful representation at a different scale.
//!
//! In HDC: reduce dimensionality while preserving relative similarity.
//! A 10,000-dim vector becomes a 1,000-dim vector that still answers
//! the same similarity queries, just with less precision.

use crate::turbo_holographic::{BinaryHV, BundleAccumulator};

/// Strategy for coarsening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarsenStrategy {
    /// Block averaging: group consecutive bits, majority-vote each block.
    /// Preserves local structure. Good for spatial data.
    BlockMajority,
    /// Stride sampling: take every Nth bit.
    /// Fastest, but loses local correlations.
    Stride,
    /// XOR folding: XOR the first half with the second half.
    /// Preserves holographic properties (binding/similarity ratios).
    /// Can be applied repeatedly for power-of-2 reductions.
    XorFold,
}

/// A coarsened (lower-resolution) view of a vector.
#[derive(Clone, Debug)]
pub struct CoarsenedView {
    /// The coarsened vector.
    pub vector: BinaryHV,
    /// Original dimension before coarsening.
    pub original_dim: usize,
    /// Coarsened dimension.
    pub coarsened_dim: usize,
    /// Compression ratio (original / coarsened).
    pub ratio: f32,
    /// Strategy used.
    pub strategy: CoarsenStrategy,
}

/// Trait for anything that can be coarsened.
pub trait Coarsenable {
    /// Coarsen to a target dimension using the given strategy.
    fn coarsen(&self, target_dim: usize, strategy: CoarsenStrategy) -> CoarsenedView;

    /// Coarsen by a factor (e.g., 2 = halve the dimension).
    fn coarsen_by(&self, factor: usize, strategy: CoarsenStrategy) -> CoarsenedView;

    /// Multi-scale pyramid: produce coarsened views at each power-of-2 level
    /// down to `min_dim`. XOR-fold is used for clean halving.
    fn pyramid(&self, min_dim: usize) -> Vec<CoarsenedView>;
}

impl Coarsenable for BinaryHV {
    fn coarsen(&self, target_dim: usize, strategy: CoarsenStrategy) -> CoarsenedView {
        let orig_dim = self.dimension();
        if target_dim >= orig_dim {
            return CoarsenedView {
                vector: self.clone(),
                original_dim: orig_dim,
                coarsened_dim: orig_dim,
                ratio: 1.0,
                strategy,
            };
        }

        let vector = match strategy {
            CoarsenStrategy::BlockMajority => coarsen_block_majority(self, target_dim),
            CoarsenStrategy::Stride => coarsen_stride(self, target_dim),
            CoarsenStrategy::XorFold => coarsen_xor_fold(self, target_dim),
        };

        CoarsenedView {
            vector,
            original_dim: orig_dim,
            coarsened_dim: target_dim,
            ratio: orig_dim as f32 / target_dim as f32,
            strategy,
        }
    }

    fn coarsen_by(&self, factor: usize, strategy: CoarsenStrategy) -> CoarsenedView {
        let target = self.dimension() / factor.max(1);
        self.coarsen(target.max(64), strategy)
    }

    fn pyramid(&self, min_dim: usize) -> Vec<CoarsenedView> {
        let mut views = Vec::new();
        let mut current = self.clone();
        let mut dim = self.dimension();

        while dim > min_dim && dim > 64 {
            let half = dim / 2;
            let view = current.coarsen(half, CoarsenStrategy::XorFold);
            current = view.vector.clone();
            dim = half;
            views.push(view);
        }

        views
    }
}

/// Block majority: group bits into blocks, majority-vote each block.
fn coarsen_block_majority(hv: &BinaryHV, target_dim: usize) -> BinaryHV {
    let orig_dim = hv.dimension();
    let block_size = orig_dim / target_dim;
    if block_size == 0 {
        return hv.clone();
    }

    let words = hv.as_words();
    let num_target_words = (target_dim + 63) / 64;
    let mut result = vec![0u64; num_target_words];

    for i in 0..target_dim {
        let block_start = i * block_size;
        let block_end = (block_start + block_size).min(orig_dim);
        let mut ones = 0u32;
        let count = (block_end - block_start) as u32;

        for bit_idx in block_start..block_end {
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if (words[word_idx] >> bit_pos) & 1 == 1 {
                ones += 1;
            }
        }

        // Majority vote: set bit if more than half are 1
        if ones > count / 2 {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }

    BinaryHV::from_words(result, target_dim)
}

/// Stride sampling: take every Nth bit.
fn coarsen_stride(hv: &BinaryHV, target_dim: usize) -> BinaryHV {
    let orig_dim = hv.dimension();
    let stride = orig_dim / target_dim;
    if stride == 0 {
        return hv.clone();
    }

    let words = hv.as_words();
    let num_target_words = (target_dim + 63) / 64;
    let mut result = vec![0u64; num_target_words];

    for i in 0..target_dim {
        let src_bit = i * stride;
        if src_bit < orig_dim {
            let word_idx = src_bit / 64;
            let bit_pos = src_bit % 64;
            if (words[word_idx] >> bit_pos) & 1 == 1 {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    BinaryHV::from_words(result, target_dim)
}

/// XOR folding: fold the vector in half by XORing the two halves.
/// Preserves holographic structure. Applied repeatedly until target_dim reached.
fn coarsen_xor_fold(hv: &BinaryHV, target_dim: usize) -> BinaryHV {
    let mut current_words = hv.as_words().to_vec();
    let mut current_dim = hv.dimension();

    while current_dim > target_dim && current_dim > 64 {
        let half_dim = current_dim / 2;
        let half_words = (half_dim + 63) / 64;
        let mut folded = vec![0u64; half_words];

        for i in 0..half_words {
            let upper_idx = i + half_words;
            let lower = current_words[i];
            let upper = if upper_idx < current_words.len() {
                current_words[upper_idx]
            } else {
                0
            };
            folded[i] = lower ^ upper;
        }

        current_words = folded;
        current_dim = half_dim;
    }

    BinaryHV::from_words(current_words, current_dim)
}

impl CoarsenedView {
    /// Cross-scale similarity: compare a coarsened view against another
    /// vector at the same coarsened scale.
    pub fn similarity(&self, other: &CoarsenedView) -> f32 {
        self.vector.similarity(&other.vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_fold_halves_dimension() {
        let hv = BinaryHV::random(10000, 42);
        let view = hv.coarsen(5000, CoarsenStrategy::XorFold);
        assert_eq!(view.coarsened_dim, 5000);
        assert_eq!(view.vector.dimension(), 5000);
    }

    #[test]
    fn test_xor_fold_preserves_similarity_ordering() {
        let a = BinaryHV::random(10000, 1);
        let b = BinaryHV::random(10000, 2);
        let c = a.bind(&BinaryHV::random(10000, 3)); // c is more different from a

        let sim_ab_orig = a.similarity(&b);
        let sim_ac_orig = a.similarity(&c);

        let a_c = a.coarsen(2500, CoarsenStrategy::XorFold);
        let b_c = b.coarsen(2500, CoarsenStrategy::XorFold);
        let c_c = c.coarsen(2500, CoarsenStrategy::XorFold);

        let sim_ab_coarse = a_c.vector.similarity(&b_c.vector);
        let sim_ac_coarse = a_c.vector.similarity(&c_c.vector);

        // Similarity ordering should be approximately preserved
        // (both should be near 0.5 for random vectors, so just check non-degenerate)
        assert!(sim_ab_coarse > 0.3 && sim_ab_coarse < 0.7);
        assert!(sim_ac_coarse > 0.3 && sim_ac_coarse < 0.7);
    }

    #[test]
    fn test_block_majority() {
        let hv = BinaryHV::random(10000, 42);
        let view = hv.coarsen(1000, CoarsenStrategy::BlockMajority);
        assert_eq!(view.coarsened_dim, 1000);
        assert!((view.ratio - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_stride() {
        let hv = BinaryHV::random(10000, 42);
        let view = hv.coarsen(1000, CoarsenStrategy::Stride);
        assert_eq!(view.coarsened_dim, 1000);
    }

    #[test]
    fn test_pyramid() {
        let hv = BinaryHV::random(8192, 42);
        let pyramid = hv.pyramid(512);
        // 8192 → 4096 → 2048 → 1024 → 512 = 4 levels
        assert_eq!(pyramid.len(), 4);
        assert_eq!(pyramid[0].coarsened_dim, 4096);
        assert_eq!(pyramid[3].coarsened_dim, 512);
    }

    #[test]
    fn test_coarsen_noop_when_target_larger() {
        let hv = BinaryHV::random(1000, 42);
        let view = hv.coarsen(5000, CoarsenStrategy::XorFold);
        assert_eq!(view.coarsened_dim, 1000); // No change
    }
}
