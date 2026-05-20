// content_store.rs — Content-addressable store: FNV hash keying,
// deduplication, reference counting, garbage collection, content
// verification, chunked storage with reassembly.

use std::collections::HashMap;

/// FNV-1a 64-bit hash.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001B3);
    }
    h
}

/// A content hash (FNV-1a 64-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub u64);

impl ContentHash {
    pub fn of(data: &[u8]) -> Self {
        Self(fnv1a_hash(data))
    }

    pub fn as_hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// A stored blob with reference count.
#[derive(Debug, Clone)]
struct StoredBlob {
    data: Vec<u8>,
    ref_count: u64,
}

/// Content-addressable blob store with deduplication and reference counting.
#[derive(Debug, Clone, Default)]
pub struct ContentStore {
    blobs: HashMap<ContentHash, StoredBlob>,
}

impl ContentStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store content and return its hash.  If the content already exists,
    /// its reference count is incremented.
    pub fn put(&mut self, data: &[u8]) -> ContentHash {
        let hash = ContentHash::of(data);
        match self.blobs.get_mut(&hash) {
            Some(blob) => {
                blob.ref_count += 1;
            }
            None => {
                self.blobs.insert(
                    hash,
                    StoredBlob {
                        data: data.to_vec(),
                        ref_count: 1,
                    },
                );
            }
        }
        hash
    }

    /// Retrieve content by hash.
    pub fn get(&self, hash: &ContentHash) -> Option<&[u8]> {
        self.blobs.get(hash).map(|b| b.data.as_slice())
    }

    /// Check if the store contains the given hash.
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.blobs.contains_key(hash)
    }

    /// Get the reference count for a hash.
    pub fn ref_count(&self, hash: &ContentHash) -> u64 {
        self.blobs.get(hash).map(|b| b.ref_count).unwrap_or(0)
    }

    /// Increment the reference count (without re-storing data).
    pub fn add_ref(&mut self, hash: &ContentHash) -> bool {
        match self.blobs.get_mut(hash) {
            Some(blob) => {
                blob.ref_count += 1;
                true
            }
            None => false,
        }
    }

    /// Decrement the reference count.  Does NOT remove the blob;
    /// call `gc()` to collect zero-ref blobs.
    pub fn release(&mut self, hash: &ContentHash) -> bool {
        match self.blobs.get_mut(hash) {
            Some(blob) => {
                blob.ref_count = blob.ref_count.saturating_sub(1);
                true
            }
            None => false,
        }
    }

    /// Garbage-collect blobs with zero references.  Returns the number removed.
    pub fn gc(&mut self) -> usize {
        let before = self.blobs.len();
        self.blobs.retain(|_, v| v.ref_count > 0);
        before - self.blobs.len()
    }

    /// Verify that stored data matches its hash (detects corruption).
    pub fn verify(&self, hash: &ContentHash) -> VerifyResult {
        match self.blobs.get(hash) {
            None => VerifyResult::NotFound,
            Some(blob) => {
                let computed = ContentHash::of(&blob.data);
                if computed == *hash {
                    VerifyResult::Ok
                } else {
                    VerifyResult::Corrupted {
                        expected: *hash,
                        actual: computed,
                    }
                }
            }
        }
    }

    /// Number of unique blobs stored.
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }

    /// Total bytes stored (sum of all blob sizes).
    pub fn total_bytes(&self) -> usize {
        self.blobs.values().map(|b| b.data.len()).sum()
    }

    /// Total reference count across all blobs.
    pub fn total_refs(&self) -> u64 {
        self.blobs.values().map(|b| b.ref_count).sum()
    }
}

/// Result of a content verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    Ok,
    NotFound,
    Corrupted {
        expected: ContentHash,
        actual: ContentHash,
    },
}

// ---------------------------------------------------------------------------
// Chunked storage: split large content into fixed-size chunks,
// store each chunk in the content store, reassemble on read.
// ---------------------------------------------------------------------------

/// Manifest for a chunked object: an ordered list of chunk hashes.
#[derive(Debug, Clone)]
pub struct ChunkManifest {
    pub chunk_size: usize,
    pub total_size: usize,
    pub chunk_hashes: Vec<ContentHash>,
}

impl ChunkManifest {
    pub fn chunk_count(&self) -> usize {
        self.chunk_hashes.len()
    }
}

/// Split data into fixed-size chunks, store each, return a manifest.
pub fn store_chunked(store: &mut ContentStore, data: &[u8], chunk_size: usize) -> ChunkManifest {
    let cs = if chunk_size == 0 { 1 } else { chunk_size };
    let mut hashes = Vec::new();
    for chunk in data.chunks(cs) {
        let hash = store.put(chunk);
        hashes.push(hash);
    }
    ChunkManifest {
        chunk_size: cs,
        total_size: data.len(),
        chunk_hashes: hashes,
    }
}

/// Reassemble a chunked object from its manifest.
pub fn load_chunked(store: &ContentStore, manifest: &ChunkManifest) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(manifest.total_size);
    for hash in &manifest.chunk_hashes {
        let chunk = store.get(hash)?;
        out.extend_from_slice(chunk);
    }
    Some(out)
}

/// Release all chunk references for a manifest.
pub fn release_chunked(store: &mut ContentStore, manifest: &ChunkManifest) {
    for hash in &manifest.chunk_hashes {
        store.release(hash);
    }
}

/// Verify all chunks in a manifest.
pub fn verify_chunked(store: &ContentStore, manifest: &ChunkManifest) -> Vec<(usize, VerifyResult)> {
    manifest
        .chunk_hashes
        .iter()
        .enumerate()
        .map(|(i, hash)| (i, store.verify(hash)))
        .collect()
}

// ---------------------------------------------------------------------------
// Deduplication stats
// ---------------------------------------------------------------------------

/// Statistics about deduplication efficiency.
#[derive(Debug, Clone)]
pub struct DedupStats {
    pub unique_blobs: usize,
    pub total_refs: u64,
    pub stored_bytes: usize,
    /// Logical bytes = sum of (blob_size * ref_count).
    pub logical_bytes: u64,
}

impl DedupStats {
    pub fn dedup_ratio(&self) -> f64 {
        if self.logical_bytes == 0 {
            return 1.0;
        }
        self.stored_bytes as f64 / self.logical_bytes as f64
    }

    pub fn savings_bytes(&self) -> u64 {
        self.logical_bytes.saturating_sub(self.stored_bytes as u64)
    }
}

pub fn compute_dedup_stats(store: &ContentStore) -> DedupStats {
    let unique_blobs = store.blob_count();
    let total_refs = store.total_refs();
    let stored_bytes = store.total_bytes();
    let logical_bytes: u64 = store
        .blobs
        .values()
        .map(|b| b.data.len() as u64 * b.ref_count)
        .sum();
    DedupStats {
        unique_blobs,
        total_refs,
        stored_bytes,
        logical_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = ContentHash::of(b"hello");
        let h2 = ContentHash::of(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_different() {
        let h1 = ContentHash::of(b"hello");
        let h2 = ContentHash::of(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_display() {
        let h = ContentHash::of(b"test");
        let s = h.to_string();
        assert_eq!(s.len(), 16);
        assert_eq!(s, h.as_hex());
    }

    #[test]
    fn test_put_and_get() {
        let mut store = ContentStore::new();
        let hash = store.put(b"hello world");
        let data = store.get(&hash).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_deduplication() {
        let mut store = ContentStore::new();
        let h1 = store.put(b"same data");
        let h2 = store.put(b"same data");
        assert_eq!(h1, h2);
        assert_eq!(store.blob_count(), 1);
        assert_eq!(store.ref_count(&h1), 2);
    }

    #[test]
    fn test_contains() {
        let mut store = ContentStore::new();
        let hash = store.put(b"x");
        assert!(store.contains(&hash));
        assert!(!store.contains(&ContentHash(999)));
    }

    #[test]
    fn test_add_ref() {
        let mut store = ContentStore::new();
        let hash = store.put(b"data");
        assert_eq!(store.ref_count(&hash), 1);
        assert!(store.add_ref(&hash));
        assert_eq!(store.ref_count(&hash), 2);
    }

    #[test]
    fn test_add_ref_missing() {
        let mut store = ContentStore::new();
        assert!(!store.add_ref(&ContentHash(42)));
    }

    #[test]
    fn test_release() {
        let mut store = ContentStore::new();
        let hash = store.put(b"data");
        store.put(b"data"); // ref_count = 2
        assert!(store.release(&hash));
        assert_eq!(store.ref_count(&hash), 1);
    }

    #[test]
    fn test_release_saturates() {
        let mut store = ContentStore::new();
        let hash = store.put(b"data");
        store.release(&hash);
        store.release(&hash); // already 0, should not underflow
        assert_eq!(store.ref_count(&hash), 0);
    }

    #[test]
    fn test_gc() {
        let mut store = ContentStore::new();
        let h1 = store.put(b"keep");
        let h2 = store.put(b"discard");
        store.release(&h2);
        let removed = store.gc();
        assert_eq!(removed, 1);
        assert!(store.contains(&h1));
        assert!(!store.contains(&h2));
    }

    #[test]
    fn test_gc_no_removal() {
        let mut store = ContentStore::new();
        store.put(b"a");
        store.put(b"b");
        let removed = store.gc();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_verify_ok() {
        let mut store = ContentStore::new();
        let hash = store.put(b"verified");
        assert_eq!(store.verify(&hash), VerifyResult::Ok);
    }

    #[test]
    fn test_verify_not_found() {
        let store = ContentStore::new();
        assert_eq!(store.verify(&ContentHash(123)), VerifyResult::NotFound);
    }

    #[test]
    fn test_total_bytes() {
        let mut store = ContentStore::new();
        store.put(b"aaa");
        store.put(b"bb");
        assert_eq!(store.total_bytes(), 5);
    }

    #[test]
    fn test_total_refs() {
        let mut store = ContentStore::new();
        store.put(b"x");
        store.put(b"x");
        store.put(b"y");
        assert_eq!(store.total_refs(), 3); // x:2 + y:1
    }

    // ---- Chunked storage ----

    #[test]
    fn test_chunked_roundtrip() {
        let mut store = ContentStore::new();
        let data = b"Hello, this is a test of chunked storage!";
        let manifest = store_chunked(&mut store, data, 10);
        assert_eq!(manifest.total_size, data.len());
        assert_eq!(manifest.chunk_count(), 5); // 41 bytes / 10 = 5 chunks

        let reassembled = load_chunked(&store, &manifest).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_chunked_single_chunk() {
        let mut store = ContentStore::new();
        let data = b"small";
        let manifest = store_chunked(&mut store, data, 1000);
        assert_eq!(manifest.chunk_count(), 1);
        let reassembled = load_chunked(&store, &manifest).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_chunked_empty() {
        let mut store = ContentStore::new();
        let manifest = store_chunked(&mut store, b"", 10);
        assert_eq!(manifest.chunk_count(), 0);
        let reassembled = load_chunked(&store, &manifest).unwrap();
        assert!(reassembled.is_empty());
    }

    #[test]
    fn test_chunked_dedup() {
        let mut store = ContentStore::new();
        // Repeated data: all chunks are identical.
        let data = vec![0xAA; 30];
        let manifest = store_chunked(&mut store, &data, 10);
        assert_eq!(manifest.chunk_count(), 3);
        // But only 1 unique blob stored (all chunks are the same bytes).
        assert_eq!(store.blob_count(), 1);
        assert_eq!(store.ref_count(&manifest.chunk_hashes[0]), 3);
    }

    #[test]
    fn test_release_chunked() {
        let mut store = ContentStore::new();
        let data = b"abcdefghij";
        let manifest = store_chunked(&mut store, data, 5);
        release_chunked(&mut store, &manifest);
        let removed = store.gc();
        assert_eq!(removed, 2);
    }

    #[test]
    fn test_verify_chunked() {
        let mut store = ContentStore::new();
        let data = b"verify each chunk";
        let manifest = store_chunked(&mut store, data, 6);
        let results = verify_chunked(&store, &manifest);
        for (_, result) in &results {
            assert_eq!(*result, VerifyResult::Ok);
        }
    }

    #[test]
    fn test_chunked_missing_chunk() {
        let mut store = ContentStore::new();
        let manifest = ChunkManifest {
            chunk_size: 10,
            total_size: 10,
            chunk_hashes: vec![ContentHash(999999)],
        };
        assert!(load_chunked(&store, &manifest).is_none());
    }

    // ---- Dedup stats ----

    #[test]
    fn test_dedup_stats() {
        let mut store = ContentStore::new();
        store.put(b"data"); // 4 bytes, ref=1
        store.put(b"data"); // same, ref=2
        store.put(b"other"); // 5 bytes, ref=1
        let stats = compute_dedup_stats(&store);
        assert_eq!(stats.unique_blobs, 2);
        assert_eq!(stats.total_refs, 3);
        assert_eq!(stats.stored_bytes, 9); // 4 + 5
        assert_eq!(stats.logical_bytes, 13); // 4*2 + 5*1
        assert!(stats.dedup_ratio() < 1.0);
        assert_eq!(stats.savings_bytes(), 4);
    }

    #[test]
    fn test_dedup_stats_empty() {
        let store = ContentStore::new();
        let stats = compute_dedup_stats(&store);
        assert_eq!(stats.unique_blobs, 0);
        assert!((stats.dedup_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chunk_size_zero_clamped() {
        let mut store = ContentStore::new();
        let data = b"abc";
        let manifest = store_chunked(&mut store, data, 0);
        assert_eq!(manifest.chunk_size, 1);
        assert_eq!(manifest.chunk_count(), 3);
    }
}
