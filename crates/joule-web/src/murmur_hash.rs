//! MurmurHash3 implementation.
//!
//! MurmurHash3 32-bit, MurmurHash3 128-bit (x86 and x64 variants),
//! seed support, streaming interface, bulk hashing, and
//! hash-to-bucket distribution analysis.

// ── Helpers ──────────────────────────────────────────────────────

#[inline]
fn rotl32(x: u32, r: u32) -> u32 {
    (x << r) | (x >> (32 - r))
}

#[inline]
fn rotl64(x: u64, r: u32) -> u64 {
    (x << r) | (x >> (64 - r))
}

#[inline]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

#[inline]
fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

fn get_u32_le(data: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
}

fn get_u64_le(data: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        data[i], data[i + 1], data[i + 2], data[i + 3],
        data[i + 4], data[i + 5], data[i + 6], data[i + 7],
    ])
}

// ── MurmurHash3 32-bit ──────────────────────────────────────────

const C1_32: u32 = 0xcc9e2d51;
const C2_32: u32 = 0x1b873593;

/// Compute MurmurHash3 32-bit hash of data with a seed.
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    let len = data.len();
    let nblocks = len / 4;
    let mut h1 = seed;

    // Body
    for i in 0..nblocks {
        let mut k1 = get_u32_le(data, i * 4);
        k1 = k1.wrapping_mul(C1_32);
        k1 = rotl32(k1, 15);
        k1 = k1.wrapping_mul(C2_32);

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    }

    // Tail
    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;

    if tail.len() >= 3 { k1 ^= (tail[2] as u32) << 16; }
    if tail.len() >= 2 { k1 ^= (tail[1] as u32) << 8; }
    if !tail.is_empty() {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(C1_32);
        k1 = rotl32(k1, 15);
        k1 = k1.wrapping_mul(C2_32);
        h1 ^= k1;
    }

    // Finalization
    h1 ^= len as u32;
    fmix32(h1)
}

// ── MurmurHash3 128-bit x86 ─────────────────────────────────────

/// MurmurHash3 128-bit hash result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash128 {
    pub h1: u64,
    pub h2: u64,
}

impl Hash128 {
    pub fn as_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.h1.to_le_bytes());
        out[8..].copy_from_slice(&self.h2.to_le_bytes());
        out
    }
}

impl std::fmt::Display for Hash128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}{:016x}", self.h1, self.h2)
    }
}

/// Compute MurmurHash3 128-bit (x86 variant, four 32-bit lanes).
pub fn murmur3_128_x86(data: &[u8], seed: u32) -> Hash128 {
    let len = data.len();
    let nblocks = len / 16;

    let mut h1 = seed as u32;
    let mut h2 = seed as u32;
    let mut h3 = seed as u32;
    let mut h4 = seed as u32;

    let c1: u32 = 0x239b961b;
    let c2: u32 = 0xab0e9789;
    let c3: u32 = 0x38b34ae5;
    let c4: u32 = 0xa1e38b93;

    for i in 0..nblocks {
        let mut k1 = get_u32_le(data, i * 16);
        let mut k2 = get_u32_le(data, i * 16 + 4);
        let mut k3 = get_u32_le(data, i * 16 + 8);
        let mut k4 = get_u32_le(data, i * 16 + 12);

        k1 = k1.wrapping_mul(c1); k1 = rotl32(k1, 15); k1 = k1.wrapping_mul(c2); h1 ^= k1;
        h1 = rotl32(h1, 19); h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x561ccd1b);

        k2 = k2.wrapping_mul(c2); k2 = rotl32(k2, 16); k2 = k2.wrapping_mul(c3); h2 ^= k2;
        h2 = rotl32(h2, 17); h2 = h2.wrapping_add(h3);
        h2 = h2.wrapping_mul(5).wrapping_add(0x0bcaa747);

        k3 = k3.wrapping_mul(c3); k3 = rotl32(k3, 17); k3 = k3.wrapping_mul(c4); h3 ^= k3;
        h3 = rotl32(h3, 15); h3 = h3.wrapping_add(h4);
        h3 = h3.wrapping_mul(5).wrapping_add(0x96cd1c35);

        k4 = k4.wrapping_mul(c4); k4 = rotl32(k4, 18); k4 = k4.wrapping_mul(c1); h4 ^= k4;
        h4 = rotl32(h4, 13); h4 = h4.wrapping_add(h1);
        h4 = h4.wrapping_mul(5).wrapping_add(0x32ac3b17);
    }

    // Tail
    let tail = &data[nblocks * 16..];
    let mut k1: u32 = 0;
    let mut k2: u32 = 0;
    let mut k3: u32 = 0;
    let mut k4: u32 = 0;

    let tlen = tail.len();
    if tlen >= 15 { k4 ^= (tail[14] as u32) << 16; }
    if tlen >= 14 { k4 ^= (tail[13] as u32) << 8; }
    if tlen >= 13 { k4 ^= tail[12] as u32; k4 = k4.wrapping_mul(c4); k4 = rotl32(k4, 18); k4 = k4.wrapping_mul(c1); h4 ^= k4; }
    if tlen >= 12 { k3 ^= (tail[11] as u32) << 24; }
    if tlen >= 11 { k3 ^= (tail[10] as u32) << 16; }
    if tlen >= 10 { k3 ^= (tail[9] as u32) << 8; }
    if tlen >= 9 { k3 ^= tail[8] as u32; k3 = k3.wrapping_mul(c3); k3 = rotl32(k3, 17); k3 = k3.wrapping_mul(c4); h3 ^= k3; }
    if tlen >= 8 { k2 ^= (tail[7] as u32) << 24; }
    if tlen >= 7 { k2 ^= (tail[6] as u32) << 16; }
    if tlen >= 6 { k2 ^= (tail[5] as u32) << 8; }
    if tlen >= 5 { k2 ^= tail[4] as u32; k2 = k2.wrapping_mul(c2); k2 = rotl32(k2, 16); k2 = k2.wrapping_mul(c3); h2 ^= k2; }
    if tlen >= 4 { k1 ^= (tail[3] as u32) << 24; }
    if tlen >= 3 { k1 ^= (tail[2] as u32) << 16; }
    if tlen >= 2 { k1 ^= (tail[1] as u32) << 8; }
    if tlen >= 1 { k1 ^= tail[0] as u32; k1 = k1.wrapping_mul(c1); k1 = rotl32(k1, 15); k1 = k1.wrapping_mul(c2); h1 ^= k1; }

    // Finalization
    let l = len as u32;
    h1 ^= l; h2 ^= l; h3 ^= l; h4 ^= l;
    h1 = h1.wrapping_add(h2).wrapping_add(h3).wrapping_add(h4);
    h2 = h2.wrapping_add(h1);
    h3 = h3.wrapping_add(h1);
    h4 = h4.wrapping_add(h1);
    h1 = fmix32(h1);
    h2 = fmix32(h2);
    h3 = fmix32(h3);
    h4 = fmix32(h4);
    h1 = h1.wrapping_add(h2).wrapping_add(h3).wrapping_add(h4);
    h2 = h2.wrapping_add(h1);
    h3 = h3.wrapping_add(h1);
    h4 = h4.wrapping_add(h1);

    Hash128 {
        h1: (h1 as u64) | ((h2 as u64) << 32),
        h2: (h3 as u64) | ((h4 as u64) << 32),
    }
}

// ── MurmurHash3 128-bit x64 ─────────────────────────────────────

/// Compute MurmurHash3 128-bit (x64 variant, two 64-bit lanes).
pub fn murmur3_128_x64(data: &[u8], seed: u64) -> Hash128 {
    let len = data.len();
    let nblocks = len / 16;

    let mut h1 = seed;
    let mut h2 = seed;

    let c1: u64 = 0x87c37b91114253d5;
    let c2: u64 = 0x4cf5ad432745937f;

    for i in 0..nblocks {
        let mut k1 = get_u64_le(data, i * 16);
        let mut k2 = get_u64_le(data, i * 16 + 8);

        k1 = k1.wrapping_mul(c1); k1 = rotl64(k1, 31); k1 = k1.wrapping_mul(c2); h1 ^= k1;
        h1 = rotl64(h1, 27); h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x52dce729);

        k2 = k2.wrapping_mul(c2); k2 = rotl64(k2, 33); k2 = k2.wrapping_mul(c1); h2 ^= k2;
        h2 = rotl64(h2, 31); h2 = h2.wrapping_add(h1);
        h2 = h2.wrapping_mul(5).wrapping_add(0x38495ab5);
    }

    // Tail
    let tail = &data[nblocks * 16..];
    let mut k1: u64 = 0;
    let mut k2: u64 = 0;
    let tlen = tail.len();

    if tlen >= 15 { k2 ^= (tail[14] as u64) << 48; }
    if tlen >= 14 { k2 ^= (tail[13] as u64) << 40; }
    if tlen >= 13 { k2 ^= (tail[12] as u64) << 32; }
    if tlen >= 12 { k2 ^= (tail[11] as u64) << 24; }
    if tlen >= 11 { k2 ^= (tail[10] as u64) << 16; }
    if tlen >= 10 { k2 ^= (tail[9] as u64) << 8; }
    if tlen >= 9 {
        k2 ^= tail[8] as u64;
        k2 = k2.wrapping_mul(c2); k2 = rotl64(k2, 33); k2 = k2.wrapping_mul(c1); h2 ^= k2;
    }
    if tlen >= 8 { k1 ^= (tail[7] as u64) << 56; }
    if tlen >= 7 { k1 ^= (tail[6] as u64) << 48; }
    if tlen >= 6 { k1 ^= (tail[5] as u64) << 40; }
    if tlen >= 5 { k1 ^= (tail[4] as u64) << 32; }
    if tlen >= 4 { k1 ^= (tail[3] as u64) << 24; }
    if tlen >= 3 { k1 ^= (tail[2] as u64) << 16; }
    if tlen >= 2 { k1 ^= (tail[1] as u64) << 8; }
    if tlen >= 1 {
        k1 ^= tail[0] as u64;
        k1 = k1.wrapping_mul(c1); k1 = rotl64(k1, 31); k1 = k1.wrapping_mul(c2); h1 ^= k1;
    }

    // Finalization
    h1 ^= len as u64;
    h2 ^= len as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    Hash128 { h1, h2 }
}

// ── Streaming MurmurHash3 32-bit ─────────────────────────────────

/// Streaming MurmurHash3 32-bit hasher.
pub struct Murmur3Hasher32 {
    seed: u32,
    buffer: Vec<u8>,
}

impl Murmur3Hasher32 {
    pub fn new(seed: u32) -> Self {
        Self { seed, buffer: Vec::new() }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    pub fn finalize(&self) -> u32 {
        murmur3_32(&self.buffer, self.seed)
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Streaming MurmurHash3 128-bit (x64) hasher.
pub struct Murmur3Hasher128 {
    seed: u64,
    buffer: Vec<u8>,
}

impl Murmur3Hasher128 {
    pub fn new(seed: u64) -> Self {
        Self { seed, buffer: Vec::new() }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    pub fn finalize(&self) -> Hash128 {
        murmur3_128_x64(&self.buffer, self.seed)
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ── Bulk Hashing ─────────────────────────────────────────────────

/// Hash multiple items and return their hashes.
pub fn bulk_hash_32(items: &[&[u8]], seed: u32) -> Vec<u32> {
    items.iter().map(|item| murmur3_32(item, seed)).collect()
}

/// Hash multiple items and return 128-bit hashes.
pub fn bulk_hash_128(items: &[&[u8]], seed: u64) -> Vec<Hash128> {
    items.iter().map(|item| murmur3_128_x64(item, seed)).collect()
}

// ── Distribution Analysis ────────────────────────────────────────

/// Analyze hash distribution across buckets. Returns (min_count, max_count, std_dev).
pub fn analyze_distribution(hashes: &[u32], num_buckets: u32) -> (usize, usize, f64) {
    let mut buckets = vec![0usize; num_buckets as usize];
    for &h in hashes {
        let idx = (h % num_buckets) as usize;
        buckets[idx] += 1;
    }
    let min = buckets.iter().copied().min().unwrap_or(0);
    let max = buckets.iter().copied().max().unwrap_or(0);
    let mean = hashes.len() as f64 / num_buckets as f64;
    let variance = buckets.iter()
        .map(|c| { let d = *c as f64 - mean; d * d })
        .sum::<f64>() / num_buckets as f64;
    (min, max, variance.sqrt())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murmur3_32_empty() {
        let h = murmur3_32(b"", 0);
        assert_eq!(h, 0); // Known: murmur3_32("", 0) = 0
    }

    #[test]
    fn test_murmur3_32_known() {
        // With seed 0, "hello" should produce a deterministic hash.
        let h = murmur3_32(b"hello", 0);
        assert_ne!(h, 0);
        assert_eq!(murmur3_32(b"hello", 0), h); // Deterministic
    }

    #[test]
    fn test_murmur3_32_seed_varies() {
        let h1 = murmur3_32(b"hello", 0);
        let h2 = murmur3_32(b"hello", 42);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_murmur3_32_different_inputs() {
        let h1 = murmur3_32(b"hello", 0);
        let h2 = murmur3_32(b"world", 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_murmur3_128_x64_deterministic() {
        let h1 = murmur3_128_x64(b"hello world", 0);
        let h2 = murmur3_128_x64(b"hello world", 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_murmur3_128_x64_different_seeds() {
        let h1 = murmur3_128_x64(b"hello", 0);
        let h2 = murmur3_128_x64(b"hello", 1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_murmur3_128_x86() {
        let h = murmur3_128_x86(b"hello", 0);
        assert_ne!(h.h1, 0);
    }

    #[test]
    fn test_streaming_32() {
        let mut h = Murmur3Hasher32::new(42);
        h.update(b"hello ");
        h.update(b"world");
        assert_eq!(h.finalize(), murmur3_32(b"hello world", 42));
    }

    #[test]
    fn test_streaming_128() {
        let mut h = Murmur3Hasher128::new(42);
        h.update(b"hello ");
        h.update(b"world");
        assert_eq!(h.finalize(), murmur3_128_x64(b"hello world", 42));
    }

    #[test]
    fn test_bulk_hash_32() {
        let items: Vec<&[u8]> = vec![b"a", b"b", b"c"];
        let hashes = bulk_hash_32(&items, 0);
        assert_eq!(hashes.len(), 3);
        // All different
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn test_bulk_hash_128() {
        let items: Vec<&[u8]> = vec![b"x", b"y", b"z"];
        let hashes = bulk_hash_128(&items, 0);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_distribution() {
        // Hash 1000 sequential strings into 10 buckets
        let hashes: Vec<u32> = (0..1000u32).map(|i| murmur3_32(&i.to_le_bytes(), 0)).collect();
        let (min, max, std_dev) = analyze_distribution(&hashes, 10);
        // Expect roughly uniform: each bucket ~100
        assert!(min > 50, "min={} too low", min);
        assert!(max < 200, "max={} too high", max);
        assert!(std_dev < 30.0, "std_dev={} too high", std_dev);
    }

    #[test]
    fn test_hash128_display() {
        let h = murmur3_128_x64(b"test", 0);
        let s = format!("{}", h);
        assert_eq!(s.len(), 32); // 16 hex digits per half
    }

    #[test]
    fn test_hash128_as_bytes() {
        let h = murmur3_128_x64(b"test", 0);
        let bytes = h.as_bytes();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_streaming_reset() {
        let mut h = Murmur3Hasher32::new(0);
        h.update(b"garbage");
        h.reset();
        h.update(b"hello");
        assert_eq!(h.finalize(), murmur3_32(b"hello", 0));
    }
}
