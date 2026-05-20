//! MinHash-LSH Index
//!
//! Locality-Sensitive Hashing using MinHash signatures for near-duplicate
//! detection and set similarity (Jaccard) queries.
//!
//! # How it works
//!
//! 1. Each document/set is represented as a MinHash signature:
//!    k hash functions applied to the set's elements, taking the minimum.
//! 2. Signatures are split into b bands of r rows each (k = b * r).
//! 3. Two items are candidate pairs if ANY band hashes to the same bucket.
//! 4. Candidates are verified by computing exact Jaccard similarity.
//!
//! # Tuning
//!
//! - More bands (b↑, r↓): higher recall, more false positives.
//! - More rows (b↓, r↑): higher precision, more false negatives.
//! - Threshold ≈ (1/b)^(1/r).

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use super::traits::{
    Bound, Index, IndexEntry, IndexIterator, ScanDirection, SimilarityIndex,
};
use crate::error::IndexError;

/// MinHash-LSH index configuration.
#[derive(Debug, Clone)]
pub struct MinHashLshConfig {
    /// Number of hash functions (signature length). k = bands * rows.
    pub num_hashes: usize,
    /// Number of bands for LSH.
    pub bands: usize,
    /// Number of rows per band. Must satisfy bands * rows == num_hashes.
    pub rows: usize,
}

impl Default for MinHashLshConfig {
    fn default() -> Self {
        // Default: 128 hashes, 16 bands of 8 rows.
        // Threshold ≈ (1/16)^(1/8) ≈ 0.66 Jaccard similarity.
        Self {
            num_hashes: 128,
            bands: 16,
            rows: 8,
        }
    }
}

impl MinHashLshConfig {
    /// Approximate Jaccard similarity threshold for this configuration.
    pub fn threshold(&self) -> f64 {
        (1.0 / self.bands as f64).powf(1.0 / self.rows as f64)
    }
}

/// A MinHash signature: k minimum hash values.
type Signature = Vec<u64>;

/// Internal entry: key, value, signature, and tokenized set.
#[derive(Debug, Clone)]
struct LshEntry {
    key: Vec<u8>,
    value: Vec<u8>,
    signature: Signature,
    tokens: HashSet<u64>,
}

/// MinHash-LSH index for near-duplicate detection and Jaccard similarity.
pub struct MinHashLshIndex {
    config: MinHashLshConfig,
    /// All entries.
    entries: RwLock<Vec<LshEntry>>,
    /// Band buckets: band_index → (band_hash → entry_indices).
    buckets: RwLock<Vec<HashMap<u64, Vec<usize>>>>,
    /// Hash seeds for MinHash (one per hash function).
    hash_seeds: Vec<(u64, u64)>, // (a, b) for h(x) = (a*x + b) mod p
}

/// Large prime for hash functions.
const PRIME: u64 = 0xFFFFFFFFFFFFFFC5; // 2^64 - 59

impl MinHashLshIndex {
    /// Create a new MinHash-LSH index with the given configuration.
    pub fn new(config: MinHashLshConfig) -> Self {
        assert_eq!(
            config.bands * config.rows,
            config.num_hashes,
            "bands * rows must equal num_hashes"
        );

        // Generate deterministic hash seeds.
        let mut seeds = Vec::with_capacity(config.num_hashes);
        let mut rng: u64 = 0xDEADBEEFCAFEBABE;
        for _ in 0..config.num_hashes {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = rng | 1; // Ensure odd (coprime with 2^64).
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = rng;
            seeds.push((a, b));
        }

        let buckets = (0..config.bands).map(|_| HashMap::new()).collect();

        Self {
            config,
            entries: RwLock::new(Vec::new()),
            buckets: RwLock::new(buckets),
            hash_seeds: seeds,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MinHashLshConfig::default())
    }

    /// Compute MinHash signature for a set of token hashes.
    fn compute_signature(&self, tokens: &HashSet<u64>) -> Signature {
        self.hash_seeds
            .iter()
            .map(|&(a, b)| {
                tokens
                    .iter()
                    .map(|&token| a.wrapping_mul(token).wrapping_add(b))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect()
    }

    /// Hash a band (slice of signature) to a single bucket key.
    fn hash_band(&self, band: &[u64]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for &val in band {
            h ^= val;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        h
    }

    /// Tokenize a byte slice into a set of shingle hashes.
    /// Uses 3-byte shingles (trigrams).
    pub fn tokenize(data: &[u8]) -> HashSet<u64> {
        let mut tokens = HashSet::new();
        if data.len() < 3 {
            // For very short data, use individual bytes.
            for (i, &b) in data.iter().enumerate() {
                tokens.insert(b as u64 | ((i as u64) << 32));
            }
            return tokens;
        }
        for window in data.windows(3) {
            let h = (window[0] as u64)
                | ((window[1] as u64) << 8)
                | ((window[2] as u64) << 16);
            tokens.insert(h);
        }
        tokens
    }

    /// Compute exact Jaccard similarity between two token sets.
    pub fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
        let intersection = a.intersection(b).count() as f64;
        let union = a.union(b).count() as f64;
        if union == 0.0 { 0.0 } else { intersection / union }
    }

    /// Insert an entry with pre-computed tokens.
    fn insert_with_tokens(
        &self,
        key: &[u8],
        value: &[u8],
        tokens: HashSet<u64>,
    ) -> Result<(), IndexError> {
        let signature = self.compute_signature(&tokens);

        let mut entries = self.entries.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        let mut buckets = self.buckets.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        let idx = entries.len();

        // Insert into band buckets.
        for band_idx in 0..self.config.bands {
            let start = band_idx * self.config.rows;
            let end = start + self.config.rows;
            let band_hash = self.hash_band(&signature[start..end]);
            buckets[band_idx]
                .entry(band_hash)
                .or_default()
                .push(idx);
        }

        entries.push(LshEntry {
            key: key.to_vec(),
            value: value.to_vec(),
            signature,
            tokens,
        });

        Ok(())
    }

    /// Find candidate near-duplicates for a query, returning (entry, jaccard_similarity).
    pub fn find_similar(
        &self,
        query_data: &[u8],
        limit: usize,
    ) -> Result<Vec<(IndexEntry, f32)>, IndexError> {
        let query_tokens = Self::tokenize(query_data);
        let query_sig = self.compute_signature(&query_tokens);

        let entries = self.entries.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        let buckets = self.buckets.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        // Collect candidate indices from all bands.
        let mut candidates: HashSet<usize> = HashSet::new();
        for band_idx in 0..self.config.bands {
            let start = band_idx * self.config.rows;
            let end = start + self.config.rows;
            let band_hash = self.hash_band(&query_sig[start..end]);
            if let Some(indices) = buckets[band_idx].get(&band_hash) {
                for &idx in indices {
                    candidates.insert(idx);
                }
            }
        }

        // Compute exact Jaccard for candidates.
        let mut results: Vec<(IndexEntry, f32)> = candidates
            .iter()
            .filter_map(|&idx| {
                let entry = &entries[idx];
                let sim = Self::jaccard(&query_tokens, &entry.tokens) as f32;
                if sim > 0.0 {
                    Some((
                        IndexEntry::new(entry.key.clone(), entry.value.clone()),
                        sim,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending.
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

/// VecIterator (same pattern as btree.rs / art.rs).
struct VecIterator {
    entries: Vec<IndexEntry>,
    position: usize,
}

impl VecIterator {
    fn new(entries: Vec<IndexEntry>) -> Self {
        Self { entries, position: 0 }
    }
}

impl Iterator for VecIterator {
    type Item = Result<IndexEntry, IndexError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.entries.len() { return None; }
        let entry = self.entries[self.position].clone();
        self.position += 1;
        Some(Ok(entry))
    }
}

impl IndexIterator for VecIterator {}

impl Index for MinHashLshIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        let entries = self.entries.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        for entry in entries.iter() {
            if entry.key == key {
                return Ok(Some(entry.value.clone()));
            }
        }
        Ok(None)
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let tokens = Self::tokenize(key);
        self.insert_with_tokens(key, value, tokens)
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        // MinHash-LSH doesn't support efficient deletion.
        // Mark-and-skip approach: find and remove from entries.
        let mut entries = self.entries.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        let before = entries.len();
        entries.retain(|e| e.key != key);
        // Note: bucket indices become stale. Full rebuild needed for correctness.
        // For production, use tombstone + periodic rebuild.
        Ok(entries.len() < before)
    }

    fn range(
        &self,
        _start: Bound<&[u8]>,
        _end: Bound<&[u8]>,
        _direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        // LSH doesn't support range queries — return all entries.
        let entries = self.entries.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        let all: Vec<IndexEntry> = entries
            .iter()
            .map(|e| IndexEntry::new(e.key.clone(), e.value.clone()))
            .collect();
        Ok(Box::new(VecIterator::new(all)))
    }

    fn count(&self) -> Result<usize, IndexError> {
        Ok(self.len())
    }
}

impl SimilarityIndex for MinHashLshIndex {
    fn search(&self, query_key: &[u8], limit: usize) -> Result<Vec<(IndexEntry, f32)>, IndexError> {
        self.find_similar(query_key, limit)
    }

    fn snr(&self) -> f32 {
        // Approximate SNR based on index density.
        let n = self.len() as f32;
        if n <= 1.0 { return f32::INFINITY; }
        // Higher threshold config → better SNR.
        self.config.threshold() as f32 * (1.0 / n.sqrt())
    }

    fn estimated_capacity(&self) -> usize {
        // LSH scales well; practical limit is memory.
        usize::MAX / 2
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_threshold() {
        let config = MinHashLshConfig::default();
        let t = config.threshold();
        // 16 bands, 8 rows → threshold ≈ 0.66
        assert!(t > 0.5 && t < 0.8, "threshold was {}", t);
    }

    #[test]
    fn test_tokenize() {
        let tokens = MinHashLshIndex::tokenize(b"hello world");
        assert!(!tokens.is_empty());
        // "hello world" has 9 trigrams.
        assert_eq!(tokens.len(), 9);
    }

    #[test]
    fn test_tokenize_short() {
        let tokens = MinHashLshIndex::tokenize(b"hi");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_jaccard_identical() {
        let a: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let b: HashSet<u64> = [1, 2, 3].into_iter().collect();
        assert_eq!(MinHashLshIndex::jaccard(&a, &b), 1.0);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let b: HashSet<u64> = [4, 5, 6].into_iter().collect();
        assert_eq!(MinHashLshIndex::jaccard(&a, &b), 0.0);
    }

    #[test]
    fn test_jaccard_partial() {
        let a: HashSet<u64> = [1, 2, 3, 4].into_iter().collect();
        let b: HashSet<u64> = [3, 4, 5, 6].into_iter().collect();
        // intersection={3,4}=2, union={1,2,3,4,5,6}=6 → 2/6 ≈ 0.333
        let j = MinHashLshIndex::jaccard(&a, &b);
        assert!((j - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_insert_and_get() {
        let mut idx = MinHashLshIndex::with_defaults();
        idx.insert(b"hello world", b"doc1").unwrap();
        idx.insert(b"hello earth", b"doc2").unwrap();

        assert_eq!(idx.get(b"hello world").unwrap(), Some(b"doc1".to_vec()));
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_find_similar_exact() {
        let mut idx = MinHashLshIndex::with_defaults();
        idx.insert(b"the quick brown fox jumps over the lazy dog", b"doc1").unwrap();
        idx.insert(b"the quick brown fox leaps over the lazy dog", b"doc2").unwrap();
        idx.insert(b"completely different unrelated text here now", b"doc3").unwrap();

        let results = idx.find_similar(b"the quick brown fox jumps over the lazy dog", 10).unwrap();

        // Should find doc1 (exact match, sim=1.0) and doc2 (near-duplicate).
        assert!(!results.is_empty());
        // First result should be the exact match.
        assert_eq!(results[0].0.value, b"doc1");
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_find_similar_near_duplicate() {
        let mut idx = MinHashLshIndex::with_defaults();
        idx.insert(b"the quick brown fox jumps over the lazy dog", b"doc1").unwrap();
        idx.insert(b"the quick brown fox jumps over the lazy cat", b"doc2").unwrap();

        let results = idx.find_similar(b"the quick brown fox jumps over the lazy dog", 10).unwrap();

        // Both should be candidates.
        assert!(results.len() >= 2);
        // doc2 should have high similarity (only "dog"→"cat" differs).
        let doc2_sim = results.iter().find(|(e, _)| e.value == b"doc2").map(|(_, s)| *s);
        assert!(doc2_sim.unwrap_or(0.0) > 0.7);
    }

    #[test]
    fn test_find_similar_dissimilar() {
        let mut idx = MinHashLshIndex::with_defaults();
        idx.insert(b"aaaaaaaaaa", b"doc1").unwrap();
        idx.insert(b"zzzzzzzzzz", b"doc2").unwrap();

        let results = idx.find_similar(b"aaaaaaaaaa", 10).unwrap();

        // doc1 should match, doc2 should not (dissimilar).
        let has_doc1 = results.iter().any(|(e, _)| e.value == b"doc1");
        let has_doc2 = results.iter().any(|(e, s)| e.value == b"doc2" && *s > 0.5);
        assert!(has_doc1);
        assert!(!has_doc2);
    }

    #[test]
    fn test_similarity_index_trait() {
        let mut idx = MinHashLshIndex::with_defaults();
        idx.insert(b"hello world foo bar baz qux", b"v1").unwrap();
        idx.insert(b"hello world foo bar baz quux", b"v2").unwrap();

        let results = idx.search(b"hello world foo bar baz qux", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0.value, b"v1");
    }

    #[test]
    fn test_delete() {
        let mut idx = MinHashLshIndex::with_defaults();
        idx.insert(b"abc", b"1").unwrap();
        idx.insert(b"def", b"2").unwrap();
        assert_eq!(idx.len(), 2);

        assert!(idx.delete(b"abc").unwrap());
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.get(b"abc").unwrap(), None);
    }

    #[test]
    fn test_count() {
        let mut idx = MinHashLshIndex::with_defaults();
        assert_eq!(idx.count().unwrap(), 0);
        idx.insert(b"a", b"1").unwrap();
        idx.insert(b"b", b"2").unwrap();
        assert_eq!(idx.count().unwrap(), 2);
    }

    #[test]
    fn test_custom_config() {
        let config = MinHashLshConfig {
            num_hashes: 64,
            bands: 8,
            rows: 8,
        };
        let t = config.threshold();
        // threshold ≈ (1/8)^(1/8) ≈ 0.74
        assert!(t > 0.7 && t < 0.8);

        let mut idx = MinHashLshIndex::new(config);
        idx.insert(b"test data here", b"v1").unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_snr() {
        let idx = MinHashLshIndex::with_defaults();
        let snr = idx.snr();
        assert!(snr > 0.0);
    }

    #[test]
    fn test_estimated_capacity() {
        let idx = MinHashLshIndex::with_defaults();
        assert!(idx.estimated_capacity() > 1_000_000);
    }
}
