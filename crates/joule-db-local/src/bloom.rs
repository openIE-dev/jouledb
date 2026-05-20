//! Bloom filter for probabilistic key existence checks.
//!
//! A space-efficient probabilistic data structure that can determine:
//! - Definitely NOT in set (guaranteed correct)
//! - POSSIBLY in set (may be false positive)
//!
//! Used to short-circuit B-tree lookups for non-existent keys.

use std::hash::{Hash, Hasher};

/// A Bloom filter for probabilistic set membership testing.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array stored as u64 words
    bits: Vec<u64>,
    /// Number of bits in the filter
    num_bits: usize,
    /// Number of hash functions to use
    num_hashes: u32,
    /// Number of items inserted
    count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter sized for `expected_items` with the given
    /// false-positive rate (e.g., 0.01 for 1%).
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let expected_items = expected_items.max(1);
        let fp = false_positive_rate.clamp(0.0001, 0.5);

        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let num_bits =
            (-(expected_items as f64) * fp.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_bits = num_bits.max(64); // minimum 64 bits

        // Optimal number of hashes: k = (m/n) * ln(2)
        let num_hashes = ((num_bits as f64 / expected_items as f64) * 2.0_f64.ln()).ceil() as u32;
        let num_hashes = num_hashes.clamp(1, 30);

        let num_words = (num_bits + 63) / 64;

        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            count: 0,
        }
    }

    /// Create a Bloom filter with explicit parameters.
    pub fn with_params(num_bits: usize, num_hashes: u32) -> Self {
        let num_bits = num_bits.max(64);
        let num_words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes: num_hashes.max(1),
            count: 0,
        }
    }

    /// Insert a key into the filter.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit = self.bit_index(h1, h2, i);
            let word = bit / 64;
            let offset = bit % 64;
            self.bits[word] |= 1u64 << offset;
        }
        self.count += 1;
    }

    /// Check if a key might be in the filter.
    /// Returns `false` if the key is definitely not present (no false negatives).
    /// Returns `true` if the key is possibly present (may be a false positive).
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit = self.bit_index(h1, h2, i);
            let word = bit / 64;
            let offset = bit % 64;
            if self.bits[word] & (1u64 << offset) == 0 {
                return false;
            }
        }
        true
    }

    /// Number of items inserted.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Estimated false positive rate at current fill level.
    pub fn estimated_fp_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        let n = self.count as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Size in bytes of the filter's bit array.
    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// Clear the filter (remove all entries).
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.count = 0;
    }

    /// Serialize the bloom filter to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.bits.len() * 8);
        buf.extend_from_slice(&(self.num_bits as u64).to_le_bytes());
        buf.extend_from_slice(&(self.num_hashes as u32).to_le_bytes());
        buf.extend_from_slice(&(self.count as u32).to_le_bytes());
        for word in &self.bits {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }

    /// Deserialize a bloom filter from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }
        let num_bits = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
        let num_hashes = u32::from_le_bytes(data[8..12].try_into().ok()?);
        let count = u32::from_le_bytes(data[12..16].try_into().ok()?) as usize;
        let num_words = (num_bits + 63) / 64;
        if data.len() < 16 + num_words * 8 {
            return None;
        }
        let mut bits = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let offset = 16 + i * 8;
            bits.push(u64::from_le_bytes(
                data[offset..offset + 8].try_into().ok()?,
            ));
        }
        Some(Self {
            bits,
            num_bits,
            num_hashes,
            count,
        })
    }

    // --- Private helpers ---

    /// Compute two independent hash values using FNV-1a variant.
    fn hash_pair(&self, key: &[u8]) -> (u64, u64) {
        // Hash 1: FNV-1a
        let mut h1 = FnvHasher::new();
        key.hash(&mut h1);
        let hash1 = h1.finish();

        // Hash 2: FNV-1a with different seed
        let mut h2 = FnvHasher::with_seed(0x517cc1b727220a95);
        key.hash(&mut h2);
        let hash2 = h2.finish();

        (hash1, hash2)
    }

    /// Compute the bit index for the i-th hash function using double hashing.
    fn bit_index(&self, h1: u64, h2: u64, i: u32) -> usize {
        (h1.wrapping_add((i as u64).wrapping_mul(h2))) as usize % self.num_bits
    }
}

/// Simple FNV-1a hasher for bloom filter hashing.
struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    fn new() -> Self {
        Self {
            state: 0xcbf29ce484222325,
        }
    }

    fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_basic() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"hello");
        bf.insert(b"world");

        assert!(bf.may_contain(b"hello"));
        assert!(bf.may_contain(b"world"));
        assert!(!bf.may_contain(b"missing"));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let keys: Vec<Vec<u8>> = (0..1000)
            .map(|i| format!("key_{}", i).into_bytes())
            .collect();

        for key in &keys {
            bf.insert(key);
        }

        // Every inserted key must be found (no false negatives)
        for key in &keys {
            assert!(bf.may_contain(key), "False negative for key: {:?}", key);
        }
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        let n = 10000;
        let target_fp = 0.01;
        let mut bf = BloomFilter::new(n, target_fp);

        for i in 0..n {
            bf.insert(format!("exists_{}", i).as_bytes());
        }

        // Check false positive rate on non-existent keys
        let test_count = 10000;
        let mut false_positives = 0;
        for i in 0..test_count {
            if bf.may_contain(format!("noexist_{}", i).as_bytes()) {
                false_positives += 1;
            }
        }

        let actual_fp = false_positives as f64 / test_count as f64;
        // Allow 3x the target rate (statistical variation)
        assert!(
            actual_fp < target_fp * 3.0,
            "FP rate too high: {:.4} (target: {:.4})",
            actual_fp,
            target_fp
        );
    }

    #[test]
    fn test_bloom_serialization() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"alpha");
        bf.insert(b"beta");
        bf.insert(b"gamma");

        let bytes = bf.to_bytes();
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();

        assert!(bf2.may_contain(b"alpha"));
        assert!(bf2.may_contain(b"beta"));
        assert!(bf2.may_contain(b"gamma"));
        assert!(!bf2.may_contain(b"delta"));
        assert_eq!(bf2.count(), 3);
        assert_eq!(bf2.num_bits(), bf.num_bits());
    }

    #[test]
    fn test_bloom_clear() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"key1");
        assert!(bf.may_contain(b"key1"));

        bf.clear();
        assert!(!bf.may_contain(b"key1"));
        assert_eq!(bf.count(), 0);
    }

    #[test]
    fn test_bloom_estimated_fp_rate() {
        let bf_empty = BloomFilter::new(100, 0.01);
        assert_eq!(bf_empty.estimated_fp_rate(), 0.0);

        let mut bf = BloomFilter::new(100, 0.01);
        for i in 0..100 {
            bf.insert(format!("key_{}", i).as_bytes());
        }
        let rate = bf.estimated_fp_rate();
        assert!(rate > 0.0 && rate < 0.05);
    }

    #[test]
    fn test_bloom_with_params() {
        let mut bf = BloomFilter::with_params(1024, 7);
        bf.insert(b"test");
        assert!(bf.may_contain(b"test"));
        assert_eq!(bf.num_bits(), 1024);
        assert_eq!(bf.num_hashes(), 7);
    }

    #[test]
    fn test_bloom_size() {
        let bf = BloomFilter::new(10000, 0.01);
        // ~96 KB for 10K items at 1% FP rate
        assert!(bf.size_bytes() > 0);
        assert!(bf.size_bytes() < 200_000);
    }
}
