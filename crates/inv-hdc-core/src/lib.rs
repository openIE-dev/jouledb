//! Shared hyperdimensional computing types for cross-layer HDC pipeline.
//!
//! **ask-davidc (Phasor) → joule-db (BinaryHV) → NeuralFS (semantic search)**

/// Trait for hypervector operations across different HDC implementations.
pub trait Hypervector: Sized {
    /// Dimensionality of the hypervector.
    fn dimension(&self) -> usize;
    /// Cosine similarity between two hypervectors.
    fn similarity(&self, other: &Self) -> f64;
    /// XOR-based binding (element-wise multiplication in bipolar).
    fn bind(&self, other: &Self) -> Self;
    /// Majority-rule bundling (element-wise addition + threshold).
    fn bundle(vectors: &[Self]) -> Self;
    /// Circular permutation (shift).
    fn permute(&self, amount: i32) -> Self;
}

/// A packed binary hypervector using u64 words.
///
/// Each bit represents one dimension. XOR = bind, popcount = similarity.
#[derive(Debug, Clone)]
pub struct BinaryHV {
    pub data: Vec<u64>,
    pub dim: usize,
}

impl BinaryHV {
    /// Create a BinaryHV from pre-packed word data.
    pub fn from_words(data: Vec<u64>, dim: usize) -> Self {
        Self { data, dim }
    }

    /// Create a deterministic pseudo-random BinaryHV from a seed.
    pub fn from_seed(dim: usize, seed: u64) -> Self {
        let words = (dim + 63) / 64;
        let mut data = Vec::with_capacity(words);
        let mut state = seed;
        for _ in 0..words {
            // Simple splitmix64
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            data.push(z);
        }
        Self { data, dim }
    }
}

impl Hypervector for BinaryHV {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn similarity(&self, other: &Self) -> f64 {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        let matching: u32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f64 / self.dim as f64
    }

    fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        let data: Vec<u64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        Self {
            data,
            dim: self.dim,
        }
    }

    fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self {
                data: vec![],
                dim: 0,
            };
        }
        let dim = vectors[0].dim;
        let words = vectors[0].data.len();
        let threshold = vectors.len() / 2;

        let mut result = vec![0u64; words];
        for w in 0..words {
            for bit in 0..64 {
                let count: usize = vectors
                    .iter()
                    .filter(|v| (v.data[w] >> bit) & 1 == 1)
                    .count();
                if count > threshold {
                    result[w] |= 1u64 << bit;
                }
            }
        }
        Self { data: result, dim }
    }

    fn permute(&self, amount: i32) -> Self {
        let n = self.dim;
        if n == 0 {
            return self.clone();
        }
        let shift = ((amount % n as i32) + n as i32) as usize % n;
        let mut bits: Vec<bool> = (0..n)
            .map(|i| {
                let word = i / 64;
                let bit = i % 64;
                (self.data[word] >> bit) & 1 == 1
            })
            .collect();
        bits.rotate_right(shift);
        let words = (n + 63) / 64;
        let mut data = vec![0u64; words];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                data[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self { data, dim: n }
    }
}
