//! Bloom filter — probabilistic set membership with configurable false positive rate.
//!
//! Supports standard and counting variants, union/intersection, serialization,
//! and optimal parameter calculation.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Optimal parameters ──────────────────────────────────────────────────────

/// Calculate optimal bit array size for given capacity and false positive rate.
pub fn optimal_bit_count(capacity: usize, fp_rate: f64) -> usize {
    let ln2 = std::f64::consts::LN_2;
    let m = -(capacity as f64 * fp_rate.ln()) / (ln2 * ln2);
    m.ceil() as usize
}

/// Calculate optimal number of hash functions.
pub fn optimal_hash_count(bit_count: usize, capacity: usize) -> usize {
    let k = (bit_count as f64 / capacity as f64) * std::f64::consts::LN_2;
    k.ceil().max(1.0) as usize
}

// ── BloomFilter ─────────────────────────────────────────────────────────────

/// Standard Bloom filter with configurable capacity and false-positive rate.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<bool>,
    hash_count: usize,
    item_count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with the given capacity and desired false positive rate.
    pub fn with_rate(capacity: usize, fp_rate: f64) -> Self {
        let bit_count = optimal_bit_count(capacity, fp_rate).max(8);
        let hash_count = optimal_hash_count(bit_count, capacity);
        Self {
            bits: vec![false; bit_count],
            hash_count,
            item_count: 0,
        }
    }

    /// Create a Bloom filter with explicit bit count and hash count.
    pub fn new(bit_count: usize, hash_count: usize) -> Self {
        Self {
            bits: vec![false; bit_count.max(1)],
            hash_count: hash_count.max(1),
            item_count: 0,
        }
    }

    /// Number of bits in the filter.
    pub fn bit_count(&self) -> usize {
        self.bits.len()
    }

    /// Number of hash functions used.
    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    /// Number of items inserted.
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Estimated false positive rate given current fill.
    pub fn estimated_fp_rate(&self) -> f64 {
        let m = self.bits.len() as f64;
        let k = self.hash_count as f64;
        let n = self.item_count as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    fn hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        // Double-hashing scheme: h(i) = h1 + i * h2
        let mut h1_hasher = DefaultHasher::new();
        item.hash(&mut h1_hasher);
        let h1 = h1_hasher.finish();

        let mut h2_hasher = {
            let mut s = DefaultHasher::new();
            s.write_u64(0xDEAD_BEEF);
            s
        };
        item.hash(&mut h2_hasher);
        let h2 = h2_hasher.finish();

        let m = self.bits.len() as u64;
        (0..self.hash_count)
            .map(|i| {
                let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
                (combined % m) as usize
            })
            .collect()
    }

    /// Insert an item into the filter.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for idx in self.hashes(item) {
            self.bits[idx] = true;
        }
        self.item_count += 1;
    }

    /// Check whether an item *might* be in the filter (may return false positives).
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        self.hashes(item).iter().all(|idx| self.bits[*idx])
    }

    /// Clear the filter.
    pub fn clear(&mut self) {
        self.bits.fill(false);
        self.item_count = 0;
    }

    /// Union of two filters (same size and hash count required).
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.bits.len() != other.bits.len() || self.hash_count != other.hash_count {
            return None;
        }
        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| *a || *b)
            .collect();
        Some(Self {
            bits,
            hash_count: self.hash_count,
            item_count: self.item_count + other.item_count,
        })
    }

    /// Intersection of two filters (same size and hash count required).
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.bits.len() != other.bits.len() || self.hash_count != other.hash_count {
            return None;
        }
        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| *a && *b)
            .collect();
        Some(Self {
            bits,
            hash_count: self.hash_count,
            item_count: 0, // unknown after intersection
        })
    }

    /// Serialize to a compact byte representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // header: bit_count (8 bytes), hash_count (8 bytes), item_count (8 bytes)
        out.extend_from_slice(&(self.bits.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.hash_count as u64).to_le_bytes());
        out.extend_from_slice(&(self.item_count as u64).to_le_bytes());
        // pack bits into bytes
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                if b {
                    byte |= 1 << i;
                }
            }
            out.push(byte);
        }
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 24 {
            return None;
        }
        let bit_count = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
        let hash_count = u64::from_le_bytes(data[8..16].try_into().ok()?) as usize;
        let item_count = u64::from_le_bytes(data[16..24].try_into().ok()?) as usize;
        let bit_bytes = &data[24..];
        let expected_bytes = (bit_count + 7) / 8;
        if bit_bytes.len() < expected_bytes {
            return None;
        }
        let mut bits = Vec::with_capacity(bit_count);
        for i in 0..bit_count {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bits.push((bit_bytes[byte_idx] >> bit_idx) & 1 == 1);
        }
        Some(Self {
            bits,
            hash_count,
            item_count,
        })
    }
}

// ── CountingBloomFilter ─────────────────────────────────────────────────────

/// Counting Bloom filter — supports removal via counters instead of bits.
#[derive(Debug, Clone)]
pub struct CountingBloomFilter {
    counters: Vec<u8>,
    hash_count: usize,
    item_count: usize,
}

impl CountingBloomFilter {
    pub fn with_rate(capacity: usize, fp_rate: f64) -> Self {
        let size = optimal_bit_count(capacity, fp_rate).max(8);
        let hash_count = optimal_hash_count(size, capacity);
        Self {
            counters: vec![0; size],
            hash_count,
            item_count: 0,
        }
    }

    pub fn new(size: usize, hash_count: usize) -> Self {
        Self {
            counters: vec![0; size.max(1)],
            hash_count: hash_count.max(1),
            item_count: 0,
        }
    }

    fn hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut h1_hasher = DefaultHasher::new();
        item.hash(&mut h1_hasher);
        let h1 = h1_hasher.finish();

        let mut h2_hasher = {
            let mut s = DefaultHasher::new();
            s.write_u64(0xDEAD_BEEF);
            s
        };
        item.hash(&mut h2_hasher);
        let h2 = h2_hasher.finish();

        let m = self.counters.len() as u64;
        (0..self.hash_count)
            .map(|i| {
                let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
                (combined % m) as usize
            })
            .collect()
    }

    pub fn insert<T: Hash>(&mut self, item: &T) {
        for idx in self.hashes(item) {
            self.counters[idx] = self.counters[idx].saturating_add(1);
        }
        self.item_count += 1;
    }

    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        self.hashes(item).iter().all(|idx| self.counters[*idx] > 0)
    }

    pub fn remove<T: Hash>(&mut self, item: &T) -> bool {
        if !self.contains(item) {
            return false;
        }
        for idx in self.hashes(item) {
            self.counters[idx] = self.counters[idx].saturating_sub(1);
        }
        self.item_count = self.item_count.saturating_sub(1);
        true
    }

    pub fn item_count(&self) -> usize {
        self.item_count
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut bf = BloomFilter::with_rate(100, 0.01);
        bf.insert(&"hello");
        bf.insert(&"world");
        assert!(bf.contains(&"hello"));
        assert!(bf.contains(&"world"));
        assert_eq!(bf.item_count(), 2);
    }

    #[test]
    fn test_not_contains() {
        let mut bf = BloomFilter::with_rate(1000, 0.001);
        bf.insert(&"alpha");
        // Very unlikely to be a false positive with low rate and few items
        assert!(!bf.contains(&"beta"));
        assert!(!bf.contains(&"gamma"));
    }

    #[test]
    fn test_clear() {
        let mut bf = BloomFilter::with_rate(100, 0.01);
        bf.insert(&42);
        bf.insert(&99);
        bf.clear();
        assert!(!bf.contains(&42));
        assert_eq!(bf.item_count(), 0);
    }

    #[test]
    fn test_optimal_parameters() {
        let bits = optimal_bit_count(1000, 0.01);
        let hashes = optimal_hash_count(bits, 1000);
        assert!(bits > 1000);
        assert!(hashes >= 1);
        assert!(hashes < 20);
    }

    #[test]
    fn test_union() {
        let mut a = BloomFilter::new(256, 3);
        let mut b = BloomFilter::new(256, 3);
        a.insert(&"hello");
        b.insert(&"world");
        let u = a.union(&b).unwrap();
        assert!(u.contains(&"hello"));
        assert!(u.contains(&"world"));
    }

    #[test]
    fn test_union_mismatched_returns_none() {
        let a = BloomFilter::new(256, 3);
        let b = BloomFilter::new(512, 3);
        assert!(a.union(&b).is_none());
    }

    #[test]
    fn test_intersection() {
        let mut a = BloomFilter::new(256, 3);
        let mut b = BloomFilter::new(256, 3);
        a.insert(&"shared");
        a.insert(&"only_a");
        b.insert(&"shared");
        b.insert(&"only_b");
        let inter = a.intersection(&b).unwrap();
        assert!(inter.contains(&"shared"));
        // "only_a" and "only_b" should likely not be in intersection
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut bf = BloomFilter::with_rate(100, 0.01);
        bf.insert(&"test1");
        bf.insert(&"test2");
        let bytes = bf.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).unwrap();
        assert!(restored.contains(&"test1"));
        assert!(restored.contains(&"test2"));
        assert_eq!(restored.bit_count(), bf.bit_count());
        assert_eq!(restored.hash_count(), bf.hash_count());
        assert_eq!(restored.item_count(), bf.item_count());
    }

    #[test]
    fn test_from_bytes_invalid() {
        assert!(BloomFilter::from_bytes(&[0; 10]).is_none());
    }

    #[test]
    fn test_estimated_fp_rate() {
        let mut bf = BloomFilter::with_rate(1000, 0.01);
        // Empty filter should have 0 FP rate
        assert_eq!(bf.estimated_fp_rate(), 0.0);
        for i in 0..100 {
            bf.insert(&i);
        }
        // After some inserts, FP rate should be small but positive
        let rate = bf.estimated_fp_rate();
        assert!(rate > 0.0);
        assert!(rate < 0.1);
    }

    #[test]
    fn test_counting_insert_remove() {
        let mut cbf = CountingBloomFilter::with_rate(100, 0.01);
        cbf.insert(&"hello");
        assert!(cbf.contains(&"hello"));
        assert_eq!(cbf.item_count(), 1);
        assert!(cbf.remove(&"hello"));
        assert!(!cbf.contains(&"hello"));
        assert_eq!(cbf.item_count(), 0);
    }

    #[test]
    fn test_counting_remove_nonexistent() {
        let mut cbf = CountingBloomFilter::new(256, 3);
        assert!(!cbf.remove(&"ghost"));
    }

    #[test]
    fn test_counting_multiple_inserts() {
        let mut cbf = CountingBloomFilter::new(256, 3);
        cbf.insert(&"x");
        cbf.insert(&"x");
        assert!(cbf.contains(&"x"));
        cbf.remove(&"x");
        // Still present because inserted twice
        assert!(cbf.contains(&"x"));
        cbf.remove(&"x");
        assert!(!cbf.contains(&"x"));
    }

    #[test]
    fn test_integer_keys() {
        let mut bf = BloomFilter::with_rate(500, 0.01);
        for i in 0..100 {
            bf.insert(&i);
        }
        for i in 0..100 {
            assert!(bf.contains(&i));
        }
    }
}
