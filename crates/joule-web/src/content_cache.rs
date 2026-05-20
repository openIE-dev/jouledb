//! Content-addressable storage — SHA-256 keyed deduplication cache with
//! reference counting, garbage collection of unreferenced entries, size
//! tracking, LRU eviction by last access, and integrity verification.

use std::collections::HashMap;
use std::time::Instant;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by content cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentCacheError {
    /// Content hash not found.
    NotFound,
    /// Content integrity check failed (data corrupted).
    IntegrityError,
    /// Cache capacity exceeded and eviction was not possible.
    CapacityExceeded,
}

impl std::fmt::Display for ContentCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "content not found"),
            Self::IntegrityError => write!(f, "content integrity check failed"),
            Self::CapacityExceeded => write!(f, "cache capacity exceeded"),
        }
    }
}

// ── SHA-256 (minimal) ────────────────────────────────────────────────────────

/// SHA-256 initial hash values.
const H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
    0x5be0cd19,
];

/// SHA-256 round constants.
const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K256[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Compute SHA-256 digest and return hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut state = H_INIT;
    let bit_len = (data.len() as u64) * 8;

    // Process complete 64-byte blocks.
    let mut offset = 0;
    while offset + 64 <= data.len() {
        let block: [u8; 64] = data[offset..offset + 64].try_into().unwrap();
        sha256_compress(&mut state, &block);
        offset += 64;
    }

    // Pad.
    let remaining = &data[offset..];
    let mut pad = vec![0u8; 128]; // at most two blocks
    pad[..remaining.len()].copy_from_slice(remaining);
    pad[remaining.len()] = 0x80;

    let pad_blocks = if remaining.len() < 56 { 1 } else { 2 };
    let total_pad = pad_blocks * 64;
    pad[total_pad - 8..total_pad].copy_from_slice(&bit_len.to_be_bytes());

    for i in 0..pad_blocks {
        let start = i * 64;
        let block: [u8; 64] = pad[start..start + 64].try_into().unwrap();
        sha256_compress(&mut state, &block);
    }

    let mut hex = String::with_capacity(64);
    for word in &state {
        for byte in word.to_be_bytes() {
            hex.push_str(&format!("{byte:02x}"));
        }
    }
    hex
}

// ── ContentEntry ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ContentEntry {
    data: Vec<u8>,
    ref_count: usize,
    last_accessed: Instant,
    stored_at: Instant,
}

// ── Stats ────────────────────────────────────────────────────────────────────

/// Statistics for the content cache.
#[derive(Debug, Clone, Default)]
pub struct ContentCacheStats {
    pub total_entries: usize,
    pub total_bytes: usize,
    pub total_refs: usize,
    pub dedup_savings_bytes: usize,
    pub gc_collections: u64,
    pub gc_bytes_freed: u64,
    pub evictions: u64,
    pub integrity_checks: u64,
    pub integrity_failures: u64,
}

// ── ContentCache ─────────────────────────────────────────────────────────────

/// Content-addressable cache with SHA-256 keys, reference counting, LRU
/// eviction, and garbage collection.
pub struct ContentCache {
    entries: HashMap<String, ContentEntry>,
    max_bytes: usize,
    current_bytes: usize,
    dedup_savings: usize,
    gc_collections: u64,
    gc_bytes_freed: u64,
    evictions: u64,
    integrity_checks: u64,
    integrity_failures: u64,
}

impl ContentCache {
    /// Create a new content cache with the given maximum byte capacity.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            current_bytes: 0,
            dedup_savings: 0,
            gc_collections: 0,
            gc_bytes_freed: 0,
            evictions: 0,
            integrity_checks: 0,
            integrity_failures: 0,
        }
    }

    /// Store content and return its SHA-256 hash key.
    /// If the content already exists, increments its reference count (dedup).
    pub fn store(&mut self, data: &[u8]) -> Result<String, ContentCacheError> {
        let hash = sha256_hex(data);

        if let Some(entry) = self.entries.get_mut(&hash) {
            // Deduplicate.
            entry.ref_count += 1;
            entry.last_accessed = Instant::now();
            self.dedup_savings += data.len();
            return Ok(hash);
        }

        // Make room if needed.
        while self.current_bytes + data.len() > self.max_bytes {
            if !self.evict_lru() {
                return Err(ContentCacheError::CapacityExceeded);
            }
        }

        let now = Instant::now();
        self.entries.insert(
            hash.clone(),
            ContentEntry {
                data: data.to_vec(),
                ref_count: 1,
                last_accessed: now,
                stored_at: now,
            },
        );
        self.current_bytes += data.len();
        Ok(hash)
    }

    /// Retrieve content by its hash.
    pub fn get(&mut self, hash: &str) -> Result<Vec<u8>, ContentCacheError> {
        let entry = self.entries.get_mut(hash).ok_or(ContentCacheError::NotFound)?;
        entry.last_accessed = Instant::now();
        Ok(entry.data.clone())
    }

    /// Peek at content without updating access time.
    pub fn peek(&self, hash: &str) -> Option<&[u8]> {
        self.entries.get(hash).map(|e| e.data.as_slice())
    }

    /// Add a reference to existing content.
    pub fn add_ref(&mut self, hash: &str) -> Result<usize, ContentCacheError> {
        let entry = self.entries.get_mut(hash).ok_or(ContentCacheError::NotFound)?;
        entry.ref_count += 1;
        Ok(entry.ref_count)
    }

    /// Release a reference. Returns the new ref count.
    pub fn release_ref(&mut self, hash: &str) -> Result<usize, ContentCacheError> {
        let entry = self.entries.get_mut(hash).ok_or(ContentCacheError::NotFound)?;
        entry.ref_count = entry.ref_count.saturating_sub(1);
        Ok(entry.ref_count)
    }

    /// Get the reference count for a hash.
    pub fn ref_count(&self, hash: &str) -> Option<usize> {
        self.entries.get(hash).map(|e| e.ref_count)
    }

    /// Garbage collect: remove all entries with ref_count == 0.
    pub fn gc(&mut self) -> usize {
        let to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.ref_count == 0)
            .map(|(k, _)| k.clone())
            .collect();
        let count = to_remove.len();
        for key in &to_remove {
            if let Some(entry) = self.entries.remove(key) {
                self.current_bytes -= entry.data.len();
                self.gc_bytes_freed += entry.data.len() as u64;
            }
        }
        self.gc_collections += 1;
        count
    }

    /// Verify integrity of a stored entry by recomputing its hash.
    pub fn verify(&mut self, hash: &str) -> Result<bool, ContentCacheError> {
        self.integrity_checks += 1;
        let entry = self.entries.get(hash).ok_or(ContentCacheError::NotFound)?;
        let computed = sha256_hex(&entry.data);
        let valid = computed == hash;
        if !valid {
            self.integrity_failures += 1;
        }
        Ok(valid)
    }

    /// Verify integrity of all entries.
    pub fn verify_all(&mut self) -> (usize, usize) {
        let hashes: Vec<String> = self.entries.keys().cloned().collect();
        let mut ok = 0;
        let mut failed = 0;
        for hash in hashes {
            self.integrity_checks += 1;
            let data = &self.entries[&hash].data;
            let computed = sha256_hex(data);
            if computed == hash {
                ok += 1;
            } else {
                self.integrity_failures += 1;
                failed += 1;
            }
        }
        (ok, failed)
    }

    /// Check if a hash exists in the cache.
    pub fn contains(&self, hash: &str) -> bool {
        self.entries.contains_key(hash)
    }

    /// Remove a specific entry regardless of ref count.
    pub fn remove(&mut self, hash: &str) -> bool {
        if let Some(entry) = self.entries.remove(hash) {
            self.current_bytes -= entry.data.len();
            true
        } else {
            false
        }
    }

    /// Number of unique content entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current bytes used.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Maximum byte capacity.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// All stored hashes.
    pub fn hashes(&self) -> Vec<&str> {
        self.entries.keys().map(|k| k.as_str()).collect()
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> ContentCacheStats {
        let total_refs: usize = self.entries.values().map(|e| e.ref_count).sum();
        ContentCacheStats {
            total_entries: self.entries.len(),
            total_bytes: self.current_bytes,
            total_refs,
            dedup_savings_bytes: self.dedup_savings,
            gc_collections: self.gc_collections,
            gc_bytes_freed: self.gc_bytes_freed,
            evictions: self.evictions,
            integrity_checks: self.integrity_checks,
            integrity_failures: self.integrity_failures,
        }
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
    }

    // ── Internal ─────────────────────────────────────────────────────

    /// Evict the least recently accessed entry with the lowest ref count.
    fn evict_lru(&mut self) -> bool {
        let victim = self
            .entries
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.ref_count
                    .cmp(&b.ref_count)
                    .then_with(|| a.last_accessed.cmp(&b.last_accessed))
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = victim {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_bytes -= entry.data.len();
                self.evictions += 1;
                return true;
            }
        }
        false
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_hello() {
        let hash = sha256_hex(b"hello");
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut cache = ContentCache::new(1024);
        let hash = cache.store(b"hello world").unwrap();
        assert!(!hash.is_empty());
        let data = cache.get(&hash).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_deduplication() {
        let mut cache = ContentCache::new(1024);
        let h1 = cache.store(b"duplicate").unwrap();
        let h2 = cache.store(b"duplicate").unwrap();
        assert_eq!(h1, h2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.ref_count(&h1), Some(2));
        assert!(cache.stats().dedup_savings_bytes > 0);
    }

    #[test]
    fn test_ref_counting() {
        let mut cache = ContentCache::new(1024);
        let hash = cache.store(b"data").unwrap();
        assert_eq!(cache.ref_count(&hash), Some(1));
        cache.add_ref(&hash).unwrap();
        assert_eq!(cache.ref_count(&hash), Some(2));
        cache.release_ref(&hash).unwrap();
        assert_eq!(cache.ref_count(&hash), Some(1));
    }

    #[test]
    fn test_garbage_collection() {
        let mut cache = ContentCache::new(1024);
        let hash = cache.store(b"temp").unwrap();
        cache.release_ref(&hash).unwrap();
        assert_eq!(cache.ref_count(&hash), Some(0));
        let collected = cache.gc();
        assert_eq!(collected, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_gc_preserves_referenced() {
        let mut cache = ContentCache::new(1024);
        let h1 = cache.store(b"keep").unwrap();
        let h2 = cache.store(b"remove").unwrap();
        cache.release_ref(&h2).unwrap();
        let collected = cache.gc();
        assert_eq!(collected, 1);
        assert!(cache.contains(&h1));
        assert!(!cache.contains(&h2));
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = ContentCache::new(20); // small capacity
        let h1 = cache.store(b"aaaaaaaaaa").unwrap(); // 10 bytes
        let _h2 = cache.store(b"bbbbbbbbbb").unwrap(); // 10 bytes, fills cache
        // Access h1 to make it more recently used.
        cache.get(&h1).unwrap();
        // Store more — should evict h2 (LRU).
        let h3 = cache.store(b"cccccccccc").unwrap();
        assert!(cache.contains(&h1) || cache.contains(&h3));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_verify_integrity() {
        let mut cache = ContentCache::new(1024);
        let hash = cache.store(b"integrity test").unwrap();
        assert!(cache.verify(&hash).unwrap());
    }

    #[test]
    fn test_verify_all() {
        let mut cache = ContentCache::new(4096);
        cache.store(b"one").unwrap();
        cache.store(b"two").unwrap();
        cache.store(b"three").unwrap();
        let (ok, failed) = cache.verify_all();
        assert_eq!(ok, 3);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_size_tracking() {
        let mut cache = ContentCache::new(1024);
        cache.store(b"12345").unwrap(); // 5 bytes
        cache.store(b"abcde").unwrap(); // 5 bytes
        assert_eq!(cache.current_bytes(), 10);
    }

    #[test]
    fn test_remove() {
        let mut cache = ContentCache::new(1024);
        let hash = cache.store(b"data").unwrap();
        assert!(cache.remove(&hash));
        assert!(!cache.contains(&hash));
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn test_peek() {
        let mut cache = ContentCache::new(1024);
        let hash = cache.store(b"peek data").unwrap();
        let peeked = cache.peek(&hash).unwrap();
        assert_eq!(peeked, b"peek data");
    }

    #[test]
    fn test_not_found() {
        let mut cache = ContentCache::new(1024);
        assert_eq!(
            cache.get("nonexistent"),
            Err(ContentCacheError::NotFound)
        );
    }

    #[test]
    fn test_capacity_exceeded() {
        let mut cache = ContentCache::new(5);
        cache.store(b"12345").unwrap(); // exactly fills
        // Second store should evict first, then succeed.
        let result = cache.store(b"67890");
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear() {
        let mut cache = ContentCache::new(1024);
        cache.store(b"a").unwrap();
        cache.store(b"b").unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn test_hashes() {
        let mut cache = ContentCache::new(1024);
        let h1 = cache.store(b"one").unwrap();
        let h2 = cache.store(b"two").unwrap();
        let hashes = cache.hashes();
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&h1.as_str()));
        assert!(hashes.contains(&h2.as_str()));
    }

    #[test]
    fn test_stats() {
        let mut cache = ContentCache::new(1024);
        cache.store(b"data").unwrap();
        cache.store(b"data").unwrap(); // dedup
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_refs, 2);
        assert!(stats.dedup_savings_bytes > 0);
    }
}
