//! Hash-based index — extendible hashing with directory/buckets,
//! insert/search/delete, bucket splitting, directory doubling,
//! load factor tracking, statistics, rehashing.
//!
//! Replaces ad-hoc HashMap-based indices with an extendible hashing
//! implementation that exposes internal structure for database use cases.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by hash index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashIndexError {
    /// Key not found.
    KeyNotFound(String),
    /// Duplicate key on a unique index.
    DuplicateKey(String),
    /// Bucket overflow (should not happen with extendible hashing).
    BucketOverflow,
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for HashIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::DuplicateKey(k) => write!(f, "duplicate key: {k}"),
            Self::BucketOverflow => write!(f, "bucket overflow"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for HashIndexError {}

// ── Key type ─────────────────────────────────────────────────────

/// A hashable key for the index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HashKey {
    Int(i64),
    Text(String),
    Bytes(Vec<u8>),
}

impl fmt::Display for HashKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
            Self::Bytes(v) => write!(f, "{v:?}"),
        }
    }
}

impl HashKey {
    /// Compute a hash of this key using FNV-1a.
    fn hash_value(&self) -> u64 {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let bytes: Vec<u8> = match self {
            Self::Int(v) => v.to_le_bytes().to_vec(),
            Self::Text(s) => s.as_bytes().to_vec(),
            Self::Bytes(b) => b.clone(),
        };
        let mut hash = FNV_OFFSET;
        for byte in &bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

/// A row ID payload.
pub type RowId = u64;

// ── Bucket ───────────────────────────────────────────────────────

/// A bucket holding key-value pairs.
#[derive(Debug, Clone)]
struct Bucket {
    /// Local depth of this bucket.
    local_depth: u32,
    /// Entries stored in this bucket.
    entries: Vec<(HashKey, RowId)>,
    /// Maximum entries per bucket before splitting.
    capacity: usize,
}

impl Bucket {
    fn new(local_depth: u32, capacity: usize) -> Self {
        Self {
            local_depth,
            entries: Vec::new(),
            capacity,
        }
    }

    fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    fn find(&self, key: &HashKey) -> Option<RowId> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    fn remove(&mut self, key: &HashKey) -> Option<RowId> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            Some(self.entries.remove(pos).1)
        } else {
            None
        }
    }
}

// ── Statistics ───────────────────────────────────────────────────

/// Hash index statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct HashIndexStats {
    pub global_depth: u32,
    pub directory_size: usize,
    pub bucket_count: usize,
    pub total_entries: usize,
    pub bucket_capacity: usize,
    pub load_factor: f64,
    pub max_bucket_entries: usize,
    pub min_bucket_entries: usize,
    pub avg_bucket_entries: f64,
}

// ── HashIndex ────────────────────────────────────────────────────

/// An extendible hash index.
pub struct HashIndex {
    /// Global depth — number of bits used from the hash to index the directory.
    global_depth: u32,
    /// Directory mapping hash prefix -> bucket index.
    directory: Vec<usize>,
    /// All buckets.
    buckets: Vec<Bucket>,
    /// Bucket capacity.
    bucket_capacity: usize,
    /// Total entry count.
    entry_count: usize,
    /// Unique constraint.
    unique: bool,
    /// Number of bucket splits performed.
    split_count: u64,
    /// Number of directory doublings performed.
    double_count: u64,
}

impl HashIndex {
    /// Create a new hash index.
    /// `bucket_capacity` must be >= 1.
    pub fn new(bucket_capacity: usize, unique: bool) -> Result<Self, HashIndexError> {
        if bucket_capacity == 0 {
            return Err(HashIndexError::InvalidConfig(
                "bucket_capacity must be >= 1".into(),
            ));
        }
        let bucket = Bucket::new(0, bucket_capacity);
        Ok(Self {
            global_depth: 0,
            directory: vec![0],
            buckets: vec![bucket],
            bucket_capacity,
            entry_count: 0,
            unique,
            split_count: 0,
            double_count: 0,
        })
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Global depth.
    pub fn global_depth(&self) -> u32 {
        self.global_depth
    }

    /// Directory size.
    pub fn directory_size(&self) -> usize {
        self.directory.len()
    }

    /// Number of distinct buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    fn hash_to_dir_index(&self, hash: u64) -> usize {
        if self.global_depth == 0 {
            0
        } else {
            (hash as usize) & ((1 << self.global_depth) - 1)
        }
    }

    /// Search for a key.
    pub fn search(&self, key: &HashKey) -> Option<RowId> {
        let hash = key.hash_value();
        let dir_idx = self.hash_to_dir_index(hash);
        let bucket_idx = self.directory[dir_idx];
        self.buckets[bucket_idx].find(key)
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: HashKey, row_id: RowId) -> Result<(), HashIndexError> {
        if self.unique && self.search(&key).is_some() {
            return Err(HashIndexError::DuplicateKey(key.to_string()));
        }

        let hash = key.hash_value();
        let dir_idx = self.hash_to_dir_index(hash);
        let bucket_idx = self.directory[dir_idx];

        if !self.buckets[bucket_idx].is_full() {
            self.buckets[bucket_idx].entries.push((key, row_id));
            self.entry_count += 1;
            return Ok(());
        }

        // Bucket is full — need to split.
        self.split_bucket(bucket_idx);
        self.split_count += 1;

        // Retry insert after split.
        let new_dir_idx = self.hash_to_dir_index(hash);
        let new_bucket_idx = self.directory[new_dir_idx];
        if self.buckets[new_bucket_idx].is_full() {
            // Recursive split (rare, happens with many collisions).
            return self.insert(key, row_id);
        }
        self.buckets[new_bucket_idx].entries.push((key, row_id));
        self.entry_count += 1;
        Ok(())
    }

    fn split_bucket(&mut self, bucket_idx: usize) {
        let old_depth = self.buckets[bucket_idx].local_depth;
        let new_depth = old_depth + 1;

        // If local depth equals global depth, double the directory.
        if old_depth == self.global_depth {
            self.double_directory();
        }

        // Create a new sibling bucket.
        let new_bucket_idx = self.buckets.len();
        self.buckets
            .push(Bucket::new(new_depth, self.bucket_capacity));
        self.buckets[bucket_idx].local_depth = new_depth;

        // Redistribute entries.
        let entries: Vec<(HashKey, RowId)> = self.buckets[bucket_idx].entries.drain(..).collect();
        for (key, row_id) in entries {
            let hash = key.hash_value();
            let bit = (hash >> old_depth) & 1;
            if bit == 0 {
                self.buckets[bucket_idx].entries.push((key, row_id));
            } else {
                self.buckets[new_bucket_idx].entries.push((key, row_id));
            }
        }

        // Update directory pointers.
        for i in 0..self.directory.len() {
            if self.directory[i] == bucket_idx {
                let bit = (i >> old_depth) & 1;
                if bit == 1 {
                    self.directory[i] = new_bucket_idx;
                }
            }
        }
    }

    fn double_directory(&mut self) {
        let old_size = self.directory.len();
        self.directory.reserve(old_size);
        for i in 0..old_size {
            self.directory.push(self.directory[i]);
        }
        self.global_depth += 1;
        self.double_count += 1;
    }

    /// Delete a key. Returns the row ID if found.
    pub fn delete(&mut self, key: &HashKey) -> Result<RowId, HashIndexError> {
        let hash = key.hash_value();
        let dir_idx = self.hash_to_dir_index(hash);
        let bucket_idx = self.directory[dir_idx];
        match self.buckets[bucket_idx].remove(key) {
            Some(val) => {
                self.entry_count -= 1;
                Ok(val)
            }
            None => Err(HashIndexError::KeyNotFound(key.to_string())),
        }
    }

    /// Return the load factor (entries / total capacity).
    pub fn load_factor(&self) -> f64 {
        let total_cap = self.buckets.len() * self.bucket_capacity;
        if total_cap == 0 {
            return 0.0;
        }
        self.entry_count as f64 / total_cap as f64
    }

    /// Collect all entries (unordered).
    pub fn scan_all(&self) -> Vec<(HashKey, RowId)> {
        let mut seen_buckets = Vec::new();
        let mut results = Vec::new();
        for &bucket_idx in &self.directory {
            if !seen_buckets.contains(&bucket_idx) {
                seen_buckets.push(bucket_idx);
                for entry in &self.buckets[bucket_idx].entries {
                    results.push(entry.clone());
                }
            }
        }
        results
    }

    /// Get statistics.
    pub fn stats(&self) -> HashIndexStats {
        let mut seen = Vec::new();
        let mut max_entries = 0usize;
        let mut min_entries = usize::MAX;
        let mut total_entries_in_buckets = 0usize;
        let mut distinct_count = 0usize;

        for &bucket_idx in &self.directory {
            if !seen.contains(&bucket_idx) {
                seen.push(bucket_idx);
                distinct_count += 1;
                let count = self.buckets[bucket_idx].entries.len();
                total_entries_in_buckets += count;
                if count > max_entries {
                    max_entries = count;
                }
                if count < min_entries {
                    min_entries = count;
                }
            }
        }

        if distinct_count == 0 {
            min_entries = 0;
        }

        let avg = if distinct_count > 0 {
            total_entries_in_buckets as f64 / distinct_count as f64
        } else {
            0.0
        };

        HashIndexStats {
            global_depth: self.global_depth,
            directory_size: self.directory.len(),
            bucket_count: distinct_count,
            total_entries: self.entry_count,
            bucket_capacity: self.bucket_capacity,
            load_factor: self.load_factor(),
            max_bucket_entries: max_entries,
            min_bucket_entries: min_entries,
            avg_bucket_entries: avg,
        }
    }

    /// Rehash the entire index with a new bucket capacity.
    pub fn rehash(&mut self, new_capacity: usize) -> Result<(), HashIndexError> {
        if new_capacity == 0 {
            return Err(HashIndexError::InvalidConfig(
                "bucket_capacity must be >= 1".into(),
            ));
        }
        let all = self.scan_all();
        self.global_depth = 0;
        self.directory = vec![0];
        self.buckets = vec![Bucket::new(0, new_capacity)];
        self.bucket_capacity = new_capacity;
        self.entry_count = 0;
        for (key, row_id) in all {
            self.insert(key, row_id)?;
        }
        Ok(())
    }

    /// Number of splits performed.
    pub fn split_count(&self) -> u64 {
        self.split_count
    }

    /// Number of directory doublings performed.
    pub fn double_count(&self) -> u64 {
        self.double_count
    }

    /// Serialize statistics to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let stats = self.stats();
        serde_json::json!({
            "global_depth": stats.global_depth,
            "directory_size": stats.directory_size,
            "bucket_count": stats.bucket_count,
            "total_entries": stats.total_entries,
            "load_factor": stats.load_factor,
            "bucket_capacity": stats.bucket_capacity,
        })
    }
}

impl fmt::Debug for HashIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashIndex")
            .field("global_depth", &self.global_depth)
            .field("entries", &self.entry_count)
            .field("buckets", &self.buckets.len())
            .field("directory_size", &self.directory.len())
            .finish()
    }
}

// ── Per-bucket iterator ──────────────────────────────────────────

/// Iterate over entries in a specific bucket.
pub struct BucketIter<'a> {
    entries: &'a [(HashKey, RowId)],
    pos: usize,
}

impl<'a> Iterator for BucketIter<'a> {
    type Item = &'a (HashKey, RowId);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let item = &self.entries[self.pos];
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl HashIndex {
    /// Iterate over all entries by visiting each distinct bucket.
    pub fn iter(&self) -> impl Iterator<Item = (HashKey, RowId)> + '_ {
        let mut seen: HashMap<usize, bool> = HashMap::new();
        self.directory.iter().flat_map(move |bucket_idx| {
            if seen.contains_key(&bucket_idx) {
                Vec::new()
            } else {
                seen.insert(*bucket_idx, true);
                self.buckets[*bucket_idx].entries.clone()
            }
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn int_key(v: i64) -> HashKey {
        HashKey::Int(v)
    }

    #[test]
    fn create_empty() {
        let idx = HashIndex::new(4, false).unwrap();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.global_depth(), 0);
    }

    #[test]
    fn invalid_capacity_rejected() {
        let err = HashIndex::new(0, false).unwrap_err();
        assert!(matches!(err, HashIndexError::InvalidConfig(_)));
    }

    #[test]
    fn insert_and_search() {
        let mut idx = HashIndex::new(4, false).unwrap();
        idx.insert(int_key(42), 100).unwrap();
        assert_eq!(idx.search(&int_key(42)), Some(100));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn insert_many() {
        let mut idx = HashIndex::new(4, false).unwrap();
        for i in 0..100 {
            idx.insert(int_key(i), i as u64 * 10).unwrap();
        }
        assert_eq!(idx.len(), 100);
        for i in 0..100 {
            assert_eq!(idx.search(&int_key(i)), Some(i as u64 * 10));
        }
    }

    #[test]
    fn unique_rejects_duplicate() {
        let mut idx = HashIndex::new(4, true).unwrap();
        idx.insert(int_key(1), 10).unwrap();
        let err = idx.insert(int_key(1), 20).unwrap_err();
        assert!(matches!(err, HashIndexError::DuplicateKey(_)));
    }

    #[test]
    fn non_unique_allows_duplicates() {
        let mut idx = HashIndex::new(4, false).unwrap();
        idx.insert(int_key(1), 10).unwrap();
        idx.insert(int_key(1), 20).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn delete_existing() {
        let mut idx = HashIndex::new(4, true).unwrap();
        idx.insert(int_key(5), 50).unwrap();
        let val = idx.delete(&int_key(5)).unwrap();
        assert_eq!(val, 50);
        assert!(idx.is_empty());
    }

    #[test]
    fn delete_missing_errors() {
        let mut idx = HashIndex::new(4, false).unwrap();
        let err = idx.delete(&int_key(99)).unwrap_err();
        assert!(matches!(err, HashIndexError::KeyNotFound(_)));
    }

    #[test]
    fn bucket_splitting_occurs() {
        let mut idx = HashIndex::new(2, false).unwrap();
        for i in 0..20 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        assert!(idx.split_count() > 0);
        assert!(idx.bucket_count() > 1);
    }

    #[test]
    fn directory_doubling_occurs() {
        let mut idx = HashIndex::new(2, false).unwrap();
        for i in 0..30 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        assert!(idx.double_count() > 0);
        assert!(idx.global_depth() > 0);
    }

    #[test]
    fn load_factor_reasonable() {
        let mut idx = HashIndex::new(4, false).unwrap();
        for i in 0..20 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let lf = idx.load_factor();
        assert!(lf > 0.0);
        assert!(lf <= 1.0);
    }

    #[test]
    fn scan_all_contains_all() {
        let mut idx = HashIndex::new(4, true).unwrap();
        for i in 0..15 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let all = idx.scan_all();
        assert_eq!(all.len(), 15);
        for i in 0..15 {
            assert!(all.iter().any(|(k, _)| *k == int_key(i)));
        }
    }

    #[test]
    fn stats_populated() {
        let mut idx = HashIndex::new(3, false).unwrap();
        for i in 0..10 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let stats = idx.stats();
        assert_eq!(stats.total_entries, 10);
        assert_eq!(stats.bucket_capacity, 3);
        assert!(stats.bucket_count >= 1);
        assert!(stats.avg_bucket_entries > 0.0);
    }

    #[test]
    fn rehash_changes_capacity() {
        let mut idx = HashIndex::new(2, true).unwrap();
        for i in 0..10 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        idx.rehash(8).unwrap();
        assert_eq!(idx.len(), 10);
        for i in 0..10 {
            assert_eq!(idx.search(&int_key(i)), Some(i as u64));
        }
        assert_eq!(idx.stats().bucket_capacity, 8);
    }

    #[test]
    fn rehash_zero_errors() {
        let mut idx = HashIndex::new(4, false).unwrap();
        let err = idx.rehash(0).unwrap_err();
        assert!(matches!(err, HashIndexError::InvalidConfig(_)));
    }

    #[test]
    fn text_keys_work() {
        let mut idx = HashIndex::new(4, true).unwrap();
        idx.insert(HashKey::Text("hello".into()), 1).unwrap();
        idx.insert(HashKey::Text("world".into()), 2).unwrap();
        assert_eq!(idx.search(&HashKey::Text("hello".into())), Some(1));
        assert_eq!(idx.search(&HashKey::Text("world".into())), Some(2));
    }

    #[test]
    fn to_json_structure() {
        let mut idx = HashIndex::new(4, false).unwrap();
        idx.insert(int_key(1), 10).unwrap();
        let json = idx.to_json();
        assert!(json["global_depth"].is_number());
        assert_eq!(json["total_entries"], 1);
    }

    #[test]
    fn iterator_visits_all() {
        let mut idx = HashIndex::new(4, true).unwrap();
        for i in 0..10 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let collected: Vec<(HashKey, RowId)> = idx.iter().collect();
        assert_eq!(collected.len(), 10);
    }

    #[test]
    fn small_capacity_stress() {
        let mut idx = HashIndex::new(1, true).unwrap();
        for i in 0..20 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        assert_eq!(idx.len(), 20);
        for i in 0..20 {
            assert!(idx.search(&int_key(i)).is_some());
        }
    }

    #[test]
    fn delete_then_reinsert() {
        let mut idx = HashIndex::new(4, true).unwrap();
        idx.insert(int_key(5), 50).unwrap();
        idx.delete(&int_key(5)).unwrap();
        idx.insert(int_key(5), 500).unwrap();
        assert_eq!(idx.search(&int_key(5)), Some(500));
    }
}
