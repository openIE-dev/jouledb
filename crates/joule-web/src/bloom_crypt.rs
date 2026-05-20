//! Cryptographic bloom filter — keyed hash functions for privacy-preserving
//! membership testing, configurable false positive rate, union, serialization,
//! estimated element count, and optimal parameter calculation.

use serde::{Deserialize, Serialize};

// ── Inline SHA-256 ──────────────────────────────────────────────────────────

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_process_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g; g = f; f = e; e = d.wrapping_add(t1);
        d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a); state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c); state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e); state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g); state[7] = state[7].wrapping_add(h);
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let total_len = data.len() as u64;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 { buf.push(0x00); }
    buf.extend_from_slice(&(total_len * 8).to_be_bytes());
    for chunk in buf.chunks_exact(64) {
        let block: [u8; 64] = chunk.try_into().unwrap();
        sha256_process_block(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Keyed hash: HMAC-SHA256(key, data) — used for privacy.
fn keyed_hash(key: &[u8], data: &[u8]) -> [u8; 32] {
    let k_hash;
    let k = if key.len() > 64 {
        k_hash = sha256(key);
        &k_hash[..]
    } else {
        key
    };
    let mut k_padded = [0u8; 64];
    k_padded[..k.len()].copy_from_slice(k);

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_padded[i];
        opad[i] ^= k_padded[i];
    }

    let mut inner_data = Vec::with_capacity(64 + data.len());
    inner_data.extend_from_slice(&ipad);
    inner_data.extend_from_slice(data);
    let inner_digest = sha256(&inner_data);

    let mut outer_data = Vec::with_capacity(64 + 32);
    outer_data.extend_from_slice(&opad);
    outer_data.extend_from_slice(&inner_digest);
    sha256(&outer_data)
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Cryptographic bloom filter errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptBloomError {
    /// Invalid false positive rate (must be 0 < rate < 1).
    InvalidFpRate,
    /// Zero capacity requested.
    ZeroCapacity,
    /// Incompatible filters (different key or parameters).
    IncompatibleFilters(String),
    /// Invalid serialized data.
    InvalidData(String),
}

impl std::fmt::Display for CryptBloomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFpRate => write!(f, "false positive rate must be 0 < rate < 1"),
            Self::ZeroCapacity => write!(f, "capacity must be > 0"),
            Self::IncompatibleFilters(s) => write!(f, "incompatible filters: {s}"),
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
        }
    }
}

impl std::error::Error for CryptBloomError {}

// ── Optimal parameters ──────────────────────────────────────────────────────

/// Calculate optimal bit count for a given capacity and false positive rate.
pub fn optimal_bit_count(capacity: usize, fp_rate: f64) -> usize {
    let ln2 = std::f64::consts::LN_2;
    let m = -(capacity as f64 * fp_rate.ln()) / (ln2 * ln2);
    m.ceil().max(8.0) as usize
}

/// Calculate optimal number of hash functions.
pub fn optimal_hash_count(bit_count: usize, capacity: usize) -> usize {
    if capacity == 0 {
        return 1;
    }
    let k = (bit_count as f64 / capacity as f64) * std::f64::consts::LN_2;
    k.ceil().max(1.0) as usize
}

/// Parameters summary for a given capacity and FP rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomParameters {
    pub bit_count: usize,
    pub hash_count: usize,
    pub byte_count: usize,
    pub capacity: usize,
    pub fp_rate: f64,
}

/// Calculate optimal parameters for given capacity and desired FP rate.
pub fn calculate_parameters(capacity: usize, fp_rate: f64) -> Result<BloomParameters, CryptBloomError> {
    if capacity == 0 {
        return Err(CryptBloomError::ZeroCapacity);
    }
    if fp_rate <= 0.0 || fp_rate >= 1.0 {
        return Err(CryptBloomError::InvalidFpRate);
    }
    let bit_count = optimal_bit_count(capacity, fp_rate);
    let hash_count = optimal_hash_count(bit_count, capacity);
    Ok(BloomParameters {
        bit_count,
        hash_count,
        byte_count: (bit_count + 7) / 8,
        capacity,
        fp_rate,
    })
}

// ── CryptBloomFilter ────────────────────────────────────────────────────────

/// A cryptographic bloom filter using keyed hashes for privacy.
///
/// Unlike a standard bloom filter that uses general-purpose hashes,
/// this filter uses HMAC-SHA256 with a secret key so that the filter
/// contents are only meaningful to holders of the key.
#[derive(Debug, Clone)]
pub struct CryptBloomFilter {
    /// Bit array stored as bytes.
    bits: Vec<u8>,
    /// Number of logical bits.
    bit_count: usize,
    /// Number of keyed hash functions.
    hash_count: usize,
    /// Secret key for keyed hashing.
    key: Vec<u8>,
    /// Number of items inserted.
    item_count: usize,
}

impl CryptBloomFilter {
    /// Create a new cryptographic bloom filter with a secret key,
    /// capacity, and desired false positive rate.
    pub fn new(key: &[u8], capacity: usize, fp_rate: f64) -> Result<Self, CryptBloomError> {
        if capacity == 0 {
            return Err(CryptBloomError::ZeroCapacity);
        }
        if fp_rate <= 0.0 || fp_rate >= 1.0 {
            return Err(CryptBloomError::InvalidFpRate);
        }
        let bit_count = optimal_bit_count(capacity, fp_rate);
        let hash_count = optimal_hash_count(bit_count, capacity);
        let byte_count = (bit_count + 7) / 8;
        Ok(Self {
            bits: vec![0u8; byte_count],
            bit_count,
            hash_count,
            key: key.to_vec(),
            item_count: 0,
        })
    }

    /// Create with explicit parameters.
    pub fn with_params(key: &[u8], bit_count: usize, hash_count: usize) -> Self {
        let bc = bit_count.max(8);
        let byte_count = (bc + 7) / 8;
        Self {
            bits: vec![0u8; byte_count],
            bit_count: bc,
            hash_count: hash_count.max(1),
            key: key.to_vec(),
            item_count: 0,
        }
    }

    /// Bit count.
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// Hash count.
    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    /// Number of items inserted.
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Number of bits set to 1.
    pub fn bits_set(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Compute keyed hash positions for an item.
    fn positions(&self, item: &[u8]) -> Vec<usize> {
        // Use keyed hash with index prefix for each hash function.
        // h_i(item) = HMAC(key, i || item) mod bit_count
        (0..self.hash_count)
            .map(|i| {
                let mut input = Vec::with_capacity(4 + item.len());
                input.extend_from_slice(&(i as u32).to_le_bytes());
                input.extend_from_slice(item);
                let h = keyed_hash(&self.key, &input);
                // Use first 8 bytes as u64, mod bit_count.
                let val = u64::from_le_bytes(h[..8].try_into().unwrap());
                (val % self.bit_count as u64) as usize
            })
            .collect()
    }

    fn set_bit(&mut self, pos: usize) {
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        if byte_idx < self.bits.len() {
            self.bits[byte_idx] |= 1 << bit_idx;
        }
    }

    fn get_bit(&self, pos: usize) -> bool {
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        if byte_idx < self.bits.len() {
            (self.bits[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        let positions = self.positions(item);
        for pos in positions {
            self.set_bit(pos);
        }
        self.item_count += 1;
    }

    /// Check whether an item may be in the filter.
    /// Returns true if possibly present (may be false positive),
    /// false if definitely absent.
    pub fn query(&self, item: &[u8]) -> bool {
        let positions = self.positions(item);
        positions.iter().all(|pos| self.get_bit(*pos))
    }

    /// Estimated false positive rate given current fill.
    pub fn estimated_fp_rate(&self) -> f64 {
        let m = self.bit_count as f64;
        let k = self.hash_count as f64;
        let n = self.item_count as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Estimate the number of elements in the filter based on fill ratio.
    pub fn estimated_count(&self) -> f64 {
        let m = self.bit_count as f64;
        let k = self.hash_count as f64;
        let x = self.bits_set() as f64;
        if x >= m {
            return f64::INFINITY;
        }
        -(m / k) * (1.0 - x / m).ln()
    }

    /// Union of two filters (OR of bit arrays). Both must have the same
    /// key, bit count, and hash count.
    pub fn union(&self, other: &CryptBloomFilter) -> Result<CryptBloomFilter, CryptBloomError> {
        if self.bit_count != other.bit_count || self.hash_count != other.hash_count {
            return Err(CryptBloomError::IncompatibleFilters(
                "different parameters".to_string(),
            ));
        }
        if self.key != other.key {
            return Err(CryptBloomError::IncompatibleFilters(
                "different keys".to_string(),
            ));
        }
        let mut result = self.clone();
        for (i, byte) in other.bits.iter().enumerate() {
            if i < result.bits.len() {
                result.bits[i] |= byte;
            }
        }
        result.item_count = self.item_count + other.item_count;
        Ok(result)
    }

    /// Clear the filter.
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.item_count = 0;
    }

    /// Serialize to a compact representation.
    pub fn to_serializable(&self) -> SerializableCryptBloom {
        SerializableCryptBloom {
            bits: self.bits.clone(),
            bit_count: self.bit_count,
            hash_count: self.hash_count,
            item_count: self.item_count,
        }
    }

    /// Restore from serialized data (requires the original key).
    pub fn from_serializable(
        key: &[u8],
        s: &SerializableCryptBloom,
    ) -> Result<Self, CryptBloomError> {
        let expected_bytes = (s.bit_count + 7) / 8;
        if s.bits.len() != expected_bytes {
            return Err(CryptBloomError::InvalidData(format!(
                "expected {} bytes, got {}",
                expected_bytes,
                s.bits.len()
            )));
        }
        Ok(Self {
            bits: s.bits.clone(),
            bit_count: s.bit_count,
            hash_count: s.hash_count,
            key: key.to_vec(),
            item_count: s.item_count,
        })
    }

    /// Fill ratio: fraction of bits set.
    pub fn fill_ratio(&self) -> f64 {
        self.bits_set() as f64 / self.bit_count as f64
    }
}

/// Serializable form of a cryptographic bloom filter (excludes key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCryptBloom {
    pub bits: Vec<u8>,
    pub bit_count: usize,
    pub hash_count: usize,
    pub item_count: usize,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_filter() {
        let bf = CryptBloomFilter::new(b"secret", 100, 0.01).unwrap();
        assert!(bf.bit_count() > 0);
        assert!(bf.hash_count() > 0);
        assert_eq!(bf.item_count(), 0);
    }

    #[test]
    fn test_create_invalid_fp_rate() {
        assert!(CryptBloomFilter::new(b"key", 100, 0.0).is_err());
        assert!(CryptBloomFilter::new(b"key", 100, 1.0).is_err());
        assert!(CryptBloomFilter::new(b"key", 100, -0.5).is_err());
    }

    #[test]
    fn test_create_zero_capacity() {
        assert!(CryptBloomFilter::new(b"key", 0, 0.01).is_err());
    }

    #[test]
    fn test_insert_and_query() {
        let mut bf = CryptBloomFilter::new(b"key", 1000, 0.01).unwrap();
        bf.insert(b"hello");
        bf.insert(b"world");
        assert!(bf.query(b"hello"));
        assert!(bf.query(b"world"));
        assert_eq!(bf.item_count(), 2);
    }

    #[test]
    fn test_query_absent_item() {
        let mut bf = CryptBloomFilter::new(b"key", 1000, 0.001).unwrap();
        bf.insert(b"present");
        // An absent item *should* usually return false (not guaranteed, but with
        // low FP rate and few items it should).
        assert!(!bf.query(b"definitely-not-here-xyz"));
    }

    #[test]
    fn test_different_keys_different_results() {
        let mut bf1 = CryptBloomFilter::with_params(b"key-A", 1024, 7);
        let mut bf2 = CryptBloomFilter::with_params(b"key-B", 1024, 7);
        bf1.insert(b"item");
        bf2.insert(b"item");
        // Same item under different keys produces different bit patterns.
        assert_ne!(bf1.bits, bf2.bits);
    }

    #[test]
    fn test_keyed_privacy() {
        // An item inserted with key A cannot be found with key B
        // (even with same parameters).
        let mut bf = CryptBloomFilter::with_params(b"secret-key", 2048, 10);
        bf.insert(b"private-data");

        // Query with a filter that has different key but same bits would
        // use different hash positions. We simulate by checking that the
        // positions differ.
        let bf2 = CryptBloomFilter::with_params(b"wrong-key", 2048, 10);
        let pos1 = bf.positions(b"private-data");
        let pos2 = bf2.positions(b"private-data");
        assert_ne!(pos1, pos2);
    }

    #[test]
    fn test_bits_set() {
        let mut bf = CryptBloomFilter::with_params(b"k", 256, 3);
        assert_eq!(bf.bits_set(), 0);
        bf.insert(b"item");
        assert!(bf.bits_set() > 0);
        assert!(bf.bits_set() <= 3); // At most 3 bits for one item
    }

    #[test]
    fn test_estimated_fp_rate() {
        let mut bf = CryptBloomFilter::new(b"key", 1000, 0.01).unwrap();
        let rate_empty = bf.estimated_fp_rate();
        assert!(rate_empty < 0.001); // Empty filter has ~0 FP rate
        for i in 0..100u32 {
            bf.insert(&i.to_le_bytes());
        }
        let rate_filled = bf.estimated_fp_rate();
        assert!(rate_filled > rate_empty);
    }

    #[test]
    fn test_estimated_count() {
        let mut bf = CryptBloomFilter::new(b"key", 10000, 0.01).unwrap();
        for i in 0..500u32 {
            bf.insert(&i.to_le_bytes());
        }
        let est = bf.estimated_count();
        // Should be in the ballpark of 500.
        assert!(est > 300.0 && est < 700.0, "estimated count = {est}");
    }

    #[test]
    fn test_union() {
        let mut bf1 = CryptBloomFilter::with_params(b"key", 1024, 5);
        let mut bf2 = CryptBloomFilter::with_params(b"key", 1024, 5);
        bf1.insert(b"alpha");
        bf2.insert(b"beta");
        let merged = bf1.union(&bf2).unwrap();
        assert!(merged.query(b"alpha"));
        assert!(merged.query(b"beta"));
    }

    #[test]
    fn test_union_incompatible() {
        let bf1 = CryptBloomFilter::with_params(b"key", 1024, 5);
        let bf2 = CryptBloomFilter::with_params(b"key", 2048, 5);
        assert!(bf1.union(&bf2).is_err());
    }

    #[test]
    fn test_union_different_keys() {
        let bf1 = CryptBloomFilter::with_params(b"key-A", 1024, 5);
        let bf2 = CryptBloomFilter::with_params(b"key-B", 1024, 5);
        assert!(bf1.union(&bf2).is_err());
    }

    #[test]
    fn test_clear() {
        let mut bf = CryptBloomFilter::new(b"key", 100, 0.01).unwrap();
        bf.insert(b"data");
        assert!(bf.query(b"data"));
        bf.clear();
        assert!(!bf.query(b"data"));
        assert_eq!(bf.item_count(), 0);
        assert_eq!(bf.bits_set(), 0);
    }

    #[test]
    fn test_fill_ratio() {
        let mut bf = CryptBloomFilter::with_params(b"key", 256, 3);
        assert_eq!(bf.fill_ratio(), 0.0);
        bf.insert(b"item");
        assert!(bf.fill_ratio() > 0.0);
        assert!(bf.fill_ratio() <= 1.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut bf = CryptBloomFilter::new(b"secret", 500, 0.01).unwrap();
        bf.insert(b"foo");
        bf.insert(b"bar");

        let s = bf.to_serializable();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SerializableCryptBloom = serde_json::from_str(&json).unwrap();
        let bf2 = CryptBloomFilter::from_serializable(b"secret", &s2).unwrap();

        assert!(bf2.query(b"foo"));
        assert!(bf2.query(b"bar"));
        assert_eq!(bf2.item_count(), 2);
    }

    #[test]
    fn test_serialization_invalid() {
        let s = SerializableCryptBloom {
            bits: vec![0; 10],
            bit_count: 1000, // Expects 125 bytes, not 10
            hash_count: 5,
            item_count: 0,
        };
        assert!(CryptBloomFilter::from_serializable(b"key", &s).is_err());
    }

    #[test]
    fn test_calculate_parameters() {
        let p = calculate_parameters(10000, 0.001).unwrap();
        assert!(p.bit_count > 10000);
        assert!(p.hash_count >= 7);
        assert_eq!(p.byte_count, (p.bit_count + 7) / 8);
    }

    #[test]
    fn test_calculate_parameters_invalid() {
        assert!(calculate_parameters(0, 0.01).is_err());
        assert!(calculate_parameters(100, 0.0).is_err());
        assert!(calculate_parameters(100, 1.0).is_err());
    }

    #[test]
    fn test_with_params() {
        let bf = CryptBloomFilter::with_params(b"key", 512, 4);
        assert_eq!(bf.bit_count(), 512);
        assert_eq!(bf.hash_count(), 4);
    }

    #[test]
    fn test_many_inserts_no_false_negatives() {
        let mut bf = CryptBloomFilter::new(b"key", 10000, 0.001).unwrap();
        let items: Vec<Vec<u8>> = (0u32..1000).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            bf.insert(item);
        }
        // No false negatives: every inserted item must be found.
        for item in &items {
            assert!(bf.query(item), "false negative for item");
        }
    }
}
