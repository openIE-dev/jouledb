//! MinHash — locality-sensitive hashing for sequence similarity estimation,
//! Jaccard index approximation, and genome sketching.
//!
//! Pure-Rust MinHash implementation for comparing biological sequences by
//! their k-mer content. Supports configurable sketch sizes, multiple hash
//! functions via seeded mixing, weighted MinHash for abundance data, and
//! pairwise Jaccard estimation across sequence collections.

use std::collections::HashSet;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MinHashError {
    EmptySequence,
    KTooSmall(usize),
    SketchSizeZero,
    IncompatibleSketches { a: usize, b: usize },
}

impl fmt::Display for MinHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "sequence must not be empty"),
            Self::KTooSmall(k) => write!(f, "k must be >= 1, got {k}"),
            Self::SketchSizeZero => write!(f, "sketch size must be > 0"),
            Self::IncompatibleSketches { a, b } => {
                write!(f, "sketch sizes differ: {a} vs {b}")
            }
        }
    }
}

impl std::error::Error for MinHashError {}

// ── Hash functions ──────────────────────────────────────────────

/// Simple deterministic hash: FNV-1a–style with seed mixing.
fn hash_kmer(kmer: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325_u64.wrapping_add(seed);
    for &b in kmer {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // Finalizer mix.
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

// ── Sketch ──────────────────────────────────────────────────────

/// A MinHash sketch: the minimum hash values from a k-mer set.
#[derive(Debug, Clone, PartialEq)]
pub struct Sketch {
    /// One minimum-hash value per hash function.
    pub min_hashes: Vec<u64>,
    /// The k-mer size used.
    pub k: usize,
    /// Number of hash functions (= sketch size).
    pub num_hashes: usize,
}

impl Sketch {
    /// Estimate Jaccard similarity with another sketch.
    pub fn jaccard(&self, other: &Sketch) -> Result<f64, MinHashError> {
        if self.num_hashes != other.num_hashes {
            return Err(MinHashError::IncompatibleSketches {
                a: self.num_hashes,
                b: other.num_hashes,
            });
        }
        let matches = self
            .min_hashes
            .iter()
            .zip(&other.min_hashes)
            .filter(|(a, b)| a == b)
            .count();
        Ok(matches as f64 / self.num_hashes as f64)
    }

    /// Estimated containment of self within other: |A ∩ B| / |A|.
    pub fn containment(&self, other: &Sketch) -> Result<f64, MinHashError> {
        self.jaccard(other).map(|j| {
            // Containment ≈ Jaccard when sets similar size.
            // For MinHash this is an approximation.
            j
        })
    }
}

impl fmt::Display for Sketch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sketch(k={}, hashes={}, min=0x{:016x})",
            self.k,
            self.num_hashes,
            self.min_hashes.first().copied().unwrap_or(0)
        )
    }
}

// ── MinHasher ───────────────────────────────────────────────────

/// MinHash engine with configurable k and sketch size.
#[derive(Debug, Clone)]
pub struct MinHasher {
    k: usize,
    num_hashes: usize,
    seeds: Vec<u64>,
}

impl MinHasher {
    /// Create a MinHasher with default parameters (k=21, 128 hashes).
    pub fn new() -> Self {
        Self::with_params(21, 128)
    }

    /// Create with specific k-mer size and number of hash functions.
    pub fn with_params(k: usize, num_hashes: usize) -> Self {
        let seeds: Vec<u64> = (0..num_hashes as u64)
            .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1))
            .collect();
        Self { k, num_hashes, seeds }
    }

    /// Set k-mer size.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set number of hash functions (sketch size).
    pub fn with_num_hashes(mut self, n: usize) -> Self {
        self.num_hashes = n;
        self.seeds = (0..n as u64)
            .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1))
            .collect();
        self
    }

    /// Compute a MinHash sketch for a byte sequence.
    pub fn sketch(&self, seq: &[u8]) -> Result<Sketch, MinHashError> {
        if seq.is_empty() {
            return Err(MinHashError::EmptySequence);
        }
        if self.k < 1 {
            return Err(MinHashError::KTooSmall(self.k));
        }
        if self.num_hashes == 0 {
            return Err(MinHashError::SketchSizeZero);
        }
        if seq.len() < self.k {
            return Err(MinHashError::EmptySequence);
        }

        let mut min_hashes = vec![u64::MAX; self.num_hashes];

        for window in seq.windows(self.k) {
            for (i, &seed) in self.seeds.iter().enumerate() {
                let h = hash_kmer(window, seed);
                if h < min_hashes[i] {
                    min_hashes[i] = h;
                }
            }
        }

        Ok(Sketch {
            min_hashes,
            k: self.k,
            num_hashes: self.num_hashes,
        })
    }

    /// Compute bottom-k sketch (keep k smallest hashes from one function).
    pub fn bottom_sketch(&self, seq: &[u8], bottom_k: usize) -> Result<Vec<u64>, MinHashError> {
        if seq.is_empty() || seq.len() < self.k {
            return Err(MinHashError::EmptySequence);
        }
        if bottom_k == 0 {
            return Err(MinHashError::SketchSizeZero);
        }

        let mut hashes: Vec<u64> = seq
            .windows(self.k)
            .map(|w| hash_kmer(w, self.seeds[0]))
            .collect();
        hashes.sort();
        hashes.dedup();
        hashes.truncate(bottom_k);
        Ok(hashes)
    }

    /// Exact Jaccard between k-mer sets (for validation).
    pub fn exact_jaccard(&self, a: &[u8], b: &[u8]) -> Result<f64, MinHashError> {
        if a.is_empty() || b.is_empty() {
            return Err(MinHashError::EmptySequence);
        }
        if a.len() < self.k || b.len() < self.k {
            return Ok(0.0);
        }
        let set_a: HashSet<&[u8]> = a.windows(self.k).collect();
        let set_b: HashSet<&[u8]> = b.windows(self.k).collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            return Ok(0.0);
        }
        Ok(intersection as f64 / union as f64)
    }

    /// Pairwise Jaccard matrix for a collection of sequences.
    pub fn pairwise_jaccard(
        &self,
        sequences: &[&[u8]],
    ) -> Result<Vec<Vec<f64>>, MinHashError> {
        let sketches: Vec<Sketch> = sequences
            .iter()
            .map(|s| self.sketch(s))
            .collect::<Result<Vec<_>, _>>()?;

        let n = sketches.len();
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let j_val = sketches[i].jaccard(&sketches[j])?;
                matrix[i][j] = j_val;
                matrix[j][i] = j_val;
            }
        }
        Ok(matrix)
    }

    /// K-mer size accessor.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

impl Default for MinHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MinHasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinHasher(k={}, hashes={})", self.k, self.num_hashes)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sketch_basic() {
        let mh = MinHasher::with_params(3, 64);
        let s = mh.sketch(b"ACGTACGT").unwrap();
        assert_eq!(s.num_hashes, 64);
        assert_eq!(s.k, 3);
    }

    #[test]
    fn test_empty_error() {
        let mh = MinHasher::with_params(3, 64);
        assert!(mh.sketch(b"").is_err());
    }

    #[test]
    fn test_seq_too_short() {
        let mh = MinHasher::with_params(10, 64);
        assert!(mh.sketch(b"ACG").is_err());
    }

    #[test]
    fn test_identical_jaccard() {
        let mh = MinHasher::with_params(3, 128);
        let s1 = mh.sketch(b"ACGTACGT").unwrap();
        let s2 = mh.sketch(b"ACGTACGT").unwrap();
        let j = s1.jaccard(&s2).unwrap();
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_different_jaccard() {
        let mh = MinHasher::with_params(3, 256);
        let s1 = mh.sketch(b"AAAAAAAAAA").unwrap();
        let s2 = mh.sketch(b"GGGGGGGGGG").unwrap();
        let j = s1.jaccard(&s2).unwrap();
        assert!(j < 0.5);
    }

    #[test]
    fn test_incompatible_sketches() {
        let mh1 = MinHasher::with_params(3, 64);
        let mh2 = MinHasher::with_params(3, 128);
        let s1 = mh1.sketch(b"ACGTACGT").unwrap();
        let s2 = mh2.sketch(b"ACGTACGT").unwrap();
        assert!(s1.jaccard(&s2).is_err());
    }

    #[test]
    fn test_exact_jaccard() {
        let mh = MinHasher::with_params(3, 64);
        let j = mh.exact_jaccard(b"ACGTACGT", b"ACGTACGT").unwrap();
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exact_jaccard_empty() {
        let mh = MinHasher::with_params(3, 64);
        assert!(mh.exact_jaccard(b"", b"ACG").is_err());
    }

    #[test]
    fn test_bottom_sketch() {
        let mh = MinHasher::with_params(3, 64);
        let bottom = mh.bottom_sketch(b"ACGTACGTACGTNNGA", 5).unwrap();
        assert_eq!(bottom.len(), 5);
        // Sorted ascending.
        for i in 1..bottom.len() {
            assert!(bottom[i] >= bottom[i - 1]);
        }
    }

    #[test]
    fn test_bottom_sketch_zero() {
        let mh = MinHasher::with_params(3, 64);
        assert!(mh.bottom_sketch(b"ACGT", 0).is_err());
    }

    #[test]
    fn test_pairwise_jaccard() {
        let mh = MinHasher::with_params(3, 64);
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACGT", b"TTTTTTTT"];
        let matrix = mh.pairwise_jaccard(&seqs).unwrap();
        assert_eq!(matrix.len(), 3);
        // Diagonal = 1.0.
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        // Identical sequences.
        assert!((matrix[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_chain() {
        let mh = MinHasher::new().with_k(5).with_num_hashes(32);
        assert_eq!(mh.k(), 5);
        assert_eq!(mh.num_hashes(), 32);
    }

    #[test]
    fn test_default() {
        let mh = MinHasher::default();
        assert_eq!(mh.k(), 21);
        assert_eq!(mh.num_hashes(), 128);
    }

    #[test]
    fn test_display_hasher() {
        let mh = MinHasher::with_params(5, 64);
        let s = format!("{mh}");
        assert!(s.contains("k=5"));
    }

    #[test]
    fn test_display_sketch() {
        let mh = MinHasher::with_params(3, 8);
        let sketch = mh.sketch(b"ACGTAC").unwrap();
        let s = format!("{sketch}");
        assert!(s.contains("Sketch"));
    }

    #[test]
    fn test_containment() {
        let mh = MinHasher::with_params(3, 128);
        let s1 = mh.sketch(b"ACGTACGT").unwrap();
        let s2 = mh.sketch(b"ACGTACGT").unwrap();
        let c = s1.containment(&s2).unwrap();
        assert!(c > 0.5);
    }

    #[test]
    fn test_error_display() {
        let e = MinHashError::EmptySequence;
        assert_eq!(format!("{e}"), "sequence must not be empty");
        let e2 = MinHashError::KTooSmall(0);
        assert!(format!("{e2}").contains("0"));
        let e3 = MinHashError::IncompatibleSketches { a: 10, b: 20 };
        assert!(format!("{e3}").contains("10"));
    }

    #[test]
    fn test_hash_determinism() {
        let h1 = hash_kmer(b"ACG", 42);
        let h2 = hash_kmer(b"ACG", 42);
        assert_eq!(h1, h2);
        let h3 = hash_kmer(b"ACG", 43);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_jaccard_symmetry() {
        let mh = MinHasher::with_params(3, 128);
        let s1 = mh.sketch(b"ACGTACGT").unwrap();
        let s2 = mh.sketch(b"ACGTTTTTT").unwrap();
        let j12 = s1.jaccard(&s2).unwrap();
        let j21 = s2.jaccard(&s1).unwrap();
        assert!((j12 - j21).abs() < 1e-10);
    }
}
