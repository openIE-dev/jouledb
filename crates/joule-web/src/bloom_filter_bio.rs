//! Bloom Filter — probabilistic k-mer membership testing with configurable
//! false positive rate, counting variant, and union/intersection operations.
//!
//! Pure-Rust Bloom filter for biological k-mer sets. Supports standard and
//! counting Bloom filters, optimal sizing from expected element count and
//! desired false positive rate, bulk insertion from sequences, and set-like
//! operations (union, intersection check).

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BloomError {
    InvalidSize(String),
    InvalidFpRate(f64),
    KmerTooLong { kmer_len: usize, seq_len: usize },
    CounterOverflow,
}

impl fmt::Display for BloomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize(s) => write!(f, "invalid size: {s}"),
            Self::InvalidFpRate(r) => {
                write!(f, "false positive rate must be in (0,1), got {r}")
            }
            Self::KmerTooLong { kmer_len, seq_len } => {
                write!(f, "k-mer length {kmer_len} > sequence length {seq_len}")
            }
            Self::CounterOverflow => write!(f, "counter overflow in counting filter"),
        }
    }
}

impl std::error::Error for BloomError {}

// ── Hash helpers ────────────────────────────────────────────────

/// Double-hashing scheme: h(i) = h1 + i*h2.
fn hash_pair(data: &[u8]) -> (u64, u64) {
    // FNV-1a for h1.
    let mut h1: u64 = 0xcbf29ce484222325;
    for &b in data {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
    }
    // Murmur-style finalizer for h2.
    let mut h2 = h1;
    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xff51afd7ed558ccd);
    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xc4ceb9fe1a85ec53);
    h2 ^= h2 >> 33;
    (h1, h2)
}

fn nth_hash(h1: u64, h2: u64, i: usize, size: usize) -> usize {
    let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
    (h % size as u64) as usize
}

// ── Optimal sizing ──────────────────────────────────────────────

/// Compute optimal bit array size for `n` elements and target FP rate `p`.
pub fn optimal_bits(n: usize, p: f64) -> usize {
    if n == 0 || p <= 0.0 || p >= 1.0 {
        return 64;
    }
    let m = -(n as f64 * p.ln()) / (2.0_f64.ln().powi(2));
    (m.ceil() as usize).max(64)
}

/// Compute optimal number of hash functions.
pub fn optimal_hash_count(bits: usize, n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let k = (bits as f64 / n as f64) * 2.0_f64.ln();
    (k.ceil() as usize).max(1)
}

// ── Standard Bloom filter ───────────────────────────────────────

/// Standard Bloom filter for membership testing.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
    count: usize,
}

impl BloomFilter {
    /// Create a Bloom filter with explicit size and hash count.
    pub fn new(num_bits: usize, num_hashes: usize) -> Result<Self, BloomError> {
        if num_bits == 0 {
            return Err(BloomError::InvalidSize("num_bits must be > 0".into()));
        }
        if num_hashes == 0 {
            return Err(BloomError::InvalidSize("num_hashes must be > 0".into()));
        }
        let words = (num_bits + 63) / 64;
        Ok(Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
            count: 0,
        })
    }

    /// Create from expected element count and false positive rate.
    pub fn with_fp_rate(expected_elements: usize, fp_rate: f64) -> Result<Self, BloomError> {
        if fp_rate <= 0.0 || fp_rate >= 1.0 {
            return Err(BloomError::InvalidFpRate(fp_rate));
        }
        let bits = optimal_bits(expected_elements, fp_rate);
        let hashes = optimal_hash_count(bits, expected_elements);
        Self::new(bits, hashes)
    }

    /// Insert a key.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = nth_hash(h1, h2, i, self.num_bits);
            self.bits[idx / 64] |= 1u64 << (idx % 64);
        }
        self.count += 1;
    }

    /// Check membership (may return false positive).
    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = nth_hash(h1, h2, i, self.num_bits);
            if self.bits[idx / 64] & (1u64 << (idx % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Insert all k-mers from a sequence.
    pub fn insert_kmers(&mut self, seq: &[u8], k: usize) -> Result<usize, BloomError> {
        if k > seq.len() {
            return Err(BloomError::KmerTooLong {
                kmer_len: k,
                seq_len: seq.len(),
            });
        }
        let mut inserted = 0;
        for window in seq.windows(k) {
            self.insert(window);
            inserted += 1;
        }
        Ok(inserted)
    }

    /// Number of items inserted.
    pub fn insertion_count(&self) -> usize {
        self.count
    }

    /// Number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    /// Estimated false positive rate given current fill.
    pub fn estimated_fp_rate(&self) -> f64 {
        let ones = self.popcount() as f64;
        let fill = ones / self.num_bits as f64;
        fill.powi(self.num_hashes as i32)
    }

    /// Number of set bits.
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Union with another filter (bitwise OR).
    pub fn union(&self, other: &BloomFilter) -> Result<BloomFilter, BloomError> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(BloomError::InvalidSize("filters must have same parameters".into()));
        }
        let mut result = self.clone();
        for (a, b) in result.bits.iter_mut().zip(&other.bits) {
            *a |= b;
        }
        result.count = self.count + other.count;
        Ok(result)
    }

    /// Clear all bits.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.count = 0;
    }

    /// Memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

impl fmt::Display for BloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BloomFilter(bits={}, hashes={}, items={}, fp≈{:.6})",
            self.num_bits,
            self.num_hashes,
            self.count,
            self.estimated_fp_rate()
        )
    }
}

// ── Counting Bloom filter ───────────────────────────────────────

/// Counting Bloom filter: supports deletion via 8-bit counters.
#[derive(Debug, Clone)]
pub struct CountingBloomFilter {
    counters: Vec<u8>,
    num_slots: usize,
    num_hashes: usize,
    count: usize,
}

impl CountingBloomFilter {
    /// Create a counting Bloom filter.
    pub fn new(num_slots: usize, num_hashes: usize) -> Result<Self, BloomError> {
        if num_slots == 0 {
            return Err(BloomError::InvalidSize("num_slots must be > 0".into()));
        }
        if num_hashes == 0 {
            return Err(BloomError::InvalidSize("num_hashes must be > 0".into()));
        }
        Ok(Self {
            counters: vec![0u8; num_slots],
            num_slots,
            num_hashes,
            count: 0,
        })
    }

    /// Insert a key.
    pub fn insert(&mut self, key: &[u8]) -> Result<(), BloomError> {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = nth_hash(h1, h2, i, self.num_slots);
            if self.counters[idx] == 255 {
                return Err(BloomError::CounterOverflow);
            }
            self.counters[idx] += 1;
        }
        self.count += 1;
        Ok(())
    }

    /// Delete a key (must have been inserted).
    pub fn delete(&mut self, key: &[u8]) {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = nth_hash(h1, h2, i, self.num_slots);
            self.counters[idx] = self.counters[idx].saturating_sub(1);
        }
        self.count = self.count.saturating_sub(1);
    }

    /// Check membership.
    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = nth_hash(h1, h2, i, self.num_slots);
            if self.counters[idx] == 0 {
                return false;
            }
        }
        true
    }

    /// Number of items inserted.
    pub fn insertion_count(&self) -> usize {
        self.count
    }

    /// Memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.counters.len()
    }
}

impl fmt::Display for CountingBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CountingBloomFilter(slots={}, hashes={}, items={})",
            self.num_slots, self.num_hashes, self.count
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_filter() {
        let bf = BloomFilter::new(1024, 7).unwrap();
        assert_eq!(bf.num_bits(), 1024);
        assert_eq!(bf.num_hashes(), 7);
        assert_eq!(bf.insertion_count(), 0);
    }

    #[test]
    fn test_invalid_size() {
        assert!(BloomFilter::new(0, 3).is_err());
        assert!(BloomFilter::new(100, 0).is_err());
    }

    #[test]
    fn test_insert_contains() {
        let mut bf = BloomFilter::new(1024, 5).unwrap();
        bf.insert(b"ACGT");
        assert!(bf.contains(b"ACGT"));
    }

    #[test]
    fn test_not_contains() {
        let bf = BloomFilter::new(10000, 5).unwrap();
        assert!(!bf.contains(b"NEVER_INSERTED"));
    }

    #[test]
    fn test_from_fp_rate() {
        let bf = BloomFilter::with_fp_rate(1000, 0.01).unwrap();
        assert!(bf.num_bits() > 0);
        assert!(bf.num_hashes() > 0);
    }

    #[test]
    fn test_invalid_fp_rate() {
        assert!(BloomFilter::with_fp_rate(100, 0.0).is_err());
        assert!(BloomFilter::with_fp_rate(100, 1.0).is_err());
        assert!(BloomFilter::with_fp_rate(100, -0.5).is_err());
    }

    #[test]
    fn test_insert_kmers() {
        let mut bf = BloomFilter::new(10000, 5).unwrap();
        let n = bf.insert_kmers(b"ACGTACGT", 3).unwrap();
        assert_eq!(n, 6);
        assert!(bf.contains(b"ACG"));
        assert!(bf.contains(b"CGT"));
    }

    #[test]
    fn test_insert_kmers_too_long() {
        let mut bf = BloomFilter::new(1024, 5).unwrap();
        assert!(bf.insert_kmers(b"AC", 5).is_err());
    }

    #[test]
    fn test_estimated_fp_rate() {
        let mut bf = BloomFilter::new(10000, 5).unwrap();
        let fp0 = bf.estimated_fp_rate();
        assert!(fp0 < 1e-10);
        for i in 0..100u32 {
            bf.insert(&i.to_le_bytes());
        }
        let fp1 = bf.estimated_fp_rate();
        assert!(fp1 > fp0);
    }

    #[test]
    fn test_popcount() {
        let mut bf = BloomFilter::new(1024, 3).unwrap();
        assert_eq!(bf.popcount(), 0);
        bf.insert(b"TEST");
        assert!(bf.popcount() > 0);
    }

    #[test]
    fn test_union() {
        let mut bf1 = BloomFilter::new(1024, 3).unwrap();
        let mut bf2 = BloomFilter::new(1024, 3).unwrap();
        bf1.insert(b"AAA");
        bf2.insert(b"BBB");
        let u = bf1.union(&bf2).unwrap();
        assert!(u.contains(b"AAA"));
        assert!(u.contains(b"BBB"));
    }

    #[test]
    fn test_union_incompatible() {
        let bf1 = BloomFilter::new(1024, 3).unwrap();
        let bf2 = BloomFilter::new(2048, 3).unwrap();
        assert!(bf1.union(&bf2).is_err());
    }

    #[test]
    fn test_clear() {
        let mut bf = BloomFilter::new(1024, 3).unwrap();
        bf.insert(b"TEST");
        bf.clear();
        assert!(!bf.contains(b"TEST"));
        assert_eq!(bf.insertion_count(), 0);
    }

    #[test]
    fn test_display() {
        let bf = BloomFilter::new(1024, 5).unwrap();
        let s = format!("{bf}");
        assert!(s.contains("BloomFilter"));
    }

    #[test]
    fn test_counting_insert_delete() {
        let mut cbf = CountingBloomFilter::new(1024, 5).unwrap();
        cbf.insert(b"ACGT").unwrap();
        assert!(cbf.contains(b"ACGT"));
        cbf.delete(b"ACGT");
        assert!(!cbf.contains(b"ACGT"));
    }

    #[test]
    fn test_counting_display() {
        let cbf = CountingBloomFilter::new(512, 3).unwrap();
        let s = format!("{cbf}");
        assert!(s.contains("CountingBloomFilter"));
    }

    #[test]
    fn test_counting_invalid() {
        assert!(CountingBloomFilter::new(0, 3).is_err());
    }

    #[test]
    fn test_optimal_bits() {
        let bits = optimal_bits(1000, 0.01);
        assert!(bits > 1000);
    }

    #[test]
    fn test_optimal_hash_count() {
        let hashes = optimal_hash_count(10000, 1000);
        assert!(hashes >= 1);
    }

    #[test]
    fn test_size_bytes() {
        let bf = BloomFilter::new(1024, 3).unwrap();
        assert!(bf.size_bytes() > 0);
        let cbf = CountingBloomFilter::new(1024, 3).unwrap();
        assert!(cbf.size_bytes() > 0);
    }

    #[test]
    fn test_error_display() {
        let e = BloomError::InvalidFpRate(2.0);
        assert!(format!("{e}").contains("2"));
        let e2 = BloomError::CounterOverflow;
        assert_eq!(format!("{e2}"), "counter overflow in counting filter");
    }
}
