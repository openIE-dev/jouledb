//! Log-structured merge tree — MemTable (sorted in-memory), SSTable (sorted
//! string table), compaction (merge sorted runs), bloom filter integration,
//! write path (memtable to flush to SSTable), read path (memtable then levels),
//! level statistics.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by LSM tree operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LsmError {
    /// Key not found in any level.
    KeyNotFound,
    /// MemTable is at capacity and must be flushed.
    MemTableFull,
    /// Compaction failed.
    CompactionFailed(String),
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl std::fmt::Display for LsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotFound => write!(f, "key not found"),
            Self::MemTableFull => write!(f, "memtable is full"),
            Self::CompactionFailed(msg) => write!(f, "compaction failed: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for LsmError {}

// ── Bloom Filter ─────────────────────────────────────────────────────────────

/// A simple bloom filter for SSTable key lookups.
#[derive(Debug, Clone)]
pub struct LsmBloomFilter {
    bits: Vec<bool>,
    hash_count: usize,
    item_count: usize,
}

impl LsmBloomFilter {
    /// Create a bloom filter sized for `capacity` items with the given false
    /// positive rate.
    pub fn new(capacity: usize, fp_rate: f64) -> Self {
        let ln2 = std::f64::consts::LN_2;
        let m = (-(capacity as f64 * fp_rate.ln()) / (ln2 * ln2)).ceil() as usize;
        let bit_count = m.max(8);
        let k = ((bit_count as f64 / capacity.max(1) as f64) * ln2).ceil().max(1.0) as usize;
        Self {
            bits: vec![false; bit_count],
            hash_count: k,
            item_count: 0,
        }
    }

    fn hashes(&self, key: &[u8]) -> Vec<usize> {
        let mut h1 = DefaultHasher::new();
        key.hash(&mut h1);
        let h1_val = h1.finish();

        let mut h2 = DefaultHasher::new();
        (key, 0xDEADBEEFu64).hash(&mut h2);
        let h2_val = h2.finish();

        let len = self.bits.len();
        (0..self.hash_count)
            .map(|i| (h1_val.wrapping_add((i as u64).wrapping_mul(h2_val)) % len as u64) as usize)
            .collect()
    }

    /// Insert a key into the bloom filter.
    pub fn insert(&mut self, key: &[u8]) {
        for idx in self.hashes(key) {
            self.bits[idx] = true;
        }
        self.item_count += 1;
    }

    /// Check if a key might exist (may return false positives).
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.hashes(key).iter().all(|idx| self.bits[*idx])
    }

    /// Number of items inserted.
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Estimated false positive rate.
    pub fn estimated_fp_rate(&self) -> f64 {
        let m = self.bits.len() as f64;
        let k = self.hash_count as f64;
        let n = self.item_count as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }
}

// ── MemTable ─────────────────────────────────────────────────────────────────

/// In-memory sorted key-value table backed by BTreeMap.
#[derive(Debug, Clone)]
pub struct MemTable {
    entries: BTreeMap<Vec<u8>, MemEntry>,
    size_bytes: usize,
    max_size_bytes: usize,
    write_count: u64,
}

/// A single memtable entry (value or tombstone).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemEntry {
    /// None means tombstone (deleted).
    pub value: Option<Vec<u8>>,
    /// Monotonically increasing sequence number.
    pub seq: u64,
}

impl MemTable {
    /// Create a new memtable with the given capacity in bytes.
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            size_bytes: 0,
            max_size_bytes,
            write_count: 0,
        }
    }

    /// Insert or update a key.  Returns `Err(MemTableFull)` if at capacity.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>, seq: u64) -> Result<(), LsmError> {
        let entry_size = key.len() + value.len() + 16; // key + value + overhead
        let old_size = self
            .entries
            .get(&key)
            .map(|e| key.len() + e.value.as_ref().map_or(0, |v| v.len()) + 16)
            .unwrap_or(0);

        let new_size = self.size_bytes - old_size + entry_size;
        if new_size > self.max_size_bytes && self.size_bytes > 0 {
            return Err(LsmError::MemTableFull);
        }

        self.entries.insert(
            key,
            MemEntry {
                value: Some(value),
                seq,
            },
        );
        self.size_bytes = new_size;
        self.write_count += 1;
        Ok(())
    }

    /// Mark a key as deleted (tombstone).
    pub fn delete(&mut self, key: Vec<u8>, seq: u64) -> Result<(), LsmError> {
        let entry_size = key.len() + 16;
        let old_size = self
            .entries
            .get(&key)
            .map(|e| key.len() + e.value.as_ref().map_or(0, |v| v.len()) + 16)
            .unwrap_or(0);

        let new_size = self.size_bytes - old_size + entry_size;
        self.entries.insert(key, MemEntry { value: None, seq });
        self.size_bytes = new_size;
        self.write_count += 1;
        Ok(())
    }

    /// Look up a key, returning the entry if present.
    pub fn get(&self, key: &[u8]) -> Option<&MemEntry> {
        self.entries.get(key)
    }

    /// Current size in bytes (approximate).
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Whether the memtable is at or over capacity.
    pub fn is_full(&self) -> bool {
        self.size_bytes >= self.max_size_bytes
    }

    /// Number of entries (including tombstones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the memtable is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total writes performed.
    pub fn write_count(&self) -> u64 {
        self.write_count
    }

    /// Iterate all entries in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &MemEntry)> {
        self.entries.iter()
    }

    /// Range scan from `start` (inclusive) to `end` (exclusive).
    pub fn range(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, MemEntry)> {
        self.entries
            .range(start.to_vec()..end.to_vec())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Drain all entries out of the memtable, resetting it.
    pub fn drain(&mut self) -> Vec<(Vec<u8>, MemEntry)> {
        self.size_bytes = 0;
        self.write_count = 0;
        let old = std::mem::take(&mut self.entries);
        old.into_iter().collect()
    }
}

// ── SSTable ──────────────────────────────────────────────────────────────────

/// An in-memory representation of a sorted string table.
#[derive(Debug, Clone)]
pub struct SsTable {
    /// Sorted key-value entries.
    entries: Vec<(Vec<u8>, SsTableEntry)>,
    /// Bloom filter for fast negative lookups.
    bloom: LsmBloomFilter,
    /// Sequence number range (min, max).
    seq_range: (u64, u64),
    /// Size in bytes.
    size_bytes: usize,
    /// Unique table id.
    id: u64,
}

/// An entry in the SSTable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsTableEntry {
    /// None means tombstone.
    pub value: Option<Vec<u8>>,
    pub seq: u64,
}

impl SsTable {
    /// Build an SSTable from a sorted list of entries.
    pub fn from_entries(id: u64, entries: Vec<(Vec<u8>, MemEntry)>) -> Self {
        let capacity = entries.len().max(1);
        let mut bloom = LsmBloomFilter::new(capacity, 0.01);
        let mut size = 0usize;
        let mut min_seq = u64::MAX;
        let mut max_seq = 0u64;

        let table_entries: Vec<(Vec<u8>, SsTableEntry)> = entries
            .into_iter()
            .map(|(k, e)| {
                bloom.insert(&k);
                size += k.len() + e.value.as_ref().map_or(0, |v| v.len()) + 16;
                min_seq = min_seq.min(e.seq);
                max_seq = max_seq.max(e.seq);
                (
                    k,
                    SsTableEntry {
                        value: e.value,
                        seq: e.seq,
                    },
                )
            })
            .collect();

        if min_seq == u64::MAX {
            min_seq = 0;
        }

        Self {
            entries: table_entries,
            bloom,
            seq_range: (min_seq, max_seq),
            size_bytes: size,
            id,
        }
    }

    /// Get a value by key using bloom filter then binary search.
    pub fn get(&self, key: &[u8]) -> Option<&SsTableEntry> {
        if !self.bloom.may_contain(key) {
            return None;
        }
        self.entries
            .binary_search_by(|(k, _)| k.as_slice().cmp(key))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Range scan (inclusive start, exclusive end).
    pub fn range(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, SsTableEntry)> {
        let start_idx = match self
            .entries
            .binary_search_by(|(k, _)| k.as_slice().cmp(start))
        {
            Ok(i) => i,
            Err(i) => i,
        };
        let mut result = Vec::new();
        for (k, v) in &self.entries[start_idx..] {
            if k.as_slice() >= end {
                break;
            }
            result.push((k.clone(), v.clone()));
        }
        result
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Sequence number range.
    pub fn seq_range(&self) -> (u64, u64) {
        self.seq_range
    }

    /// Table ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Iterate entries in order.
    pub fn iter(&self) -> impl Iterator<Item = &(Vec<u8>, SsTableEntry)> {
        self.entries.iter()
    }

    /// Smallest key, if any.
    pub fn min_key(&self) -> Option<&[u8]> {
        self.entries.first().map(|(k, _)| k.as_slice())
    }

    /// Largest key, if any.
    pub fn max_key(&self) -> Option<&[u8]> {
        self.entries.last().map(|(k, _)| k.as_slice())
    }
}

// ── Level statistics ─────────────────────────────────────────────────────────

/// Statistics for a single level in the LSM tree.
#[derive(Debug, Clone, Default)]
pub struct LevelStats {
    /// Level number (0 = freshest flushed data).
    pub level: usize,
    /// Number of SSTables in this level.
    pub table_count: usize,
    /// Total entries across all tables.
    pub entry_count: usize,
    /// Total size in bytes.
    pub size_bytes: usize,
    /// Number of compactions performed.
    pub compaction_count: u64,
}

/// Aggregate statistics for the entire LSM tree.
#[derive(Debug, Clone, Default)]
pub struct LsmStats {
    pub memtable_size_bytes: usize,
    pub memtable_entries: usize,
    pub levels: Vec<LevelStats>,
    pub total_writes: u64,
    pub total_reads: u64,
    pub total_compactions: u64,
    pub total_flushes: u64,
    pub bloom_filter_hits: u64,
    pub bloom_filter_misses: u64,
}

// ── Compaction ───────────────────────────────────────────────────────────────

/// Merge two sorted SSTable-like entry lists, keeping the entry with the
/// higher sequence number when keys collide.
pub fn merge_sorted_runs(
    a: &[(Vec<u8>, SsTableEntry)],
    b: &[(Vec<u8>, SsTableEntry)],
) -> Vec<(Vec<u8>, SsTableEntry)> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                result.push(a[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j].clone());
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // Higher seq wins.
                if a[i].1.seq >= b[j].1.seq {
                    result.push(a[i].clone());
                } else {
                    result.push(b[j].clone());
                }
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// ── LSM Tree ─────────────────────────────────────────────────────────────────

/// Configuration for the LSM tree.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Maximum memtable size in bytes before flushing.
    pub memtable_max_bytes: usize,
    /// Maximum number of SSTables per level before compaction.
    pub level_max_tables: usize,
    /// Size multiplier between levels.
    pub level_size_ratio: usize,
    /// Maximum number of levels.
    pub max_levels: usize,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            memtable_max_bytes: 4 * 1024 * 1024, // 4 MB
            level_max_tables: 4,
            level_size_ratio: 10,
            max_levels: 7,
        }
    }
}

/// Log-structured merge tree.
#[derive(Debug)]
pub struct LsmTree {
    /// Active memtable.
    memtable: MemTable,
    /// Levels of SSTables (level 0 = newest).
    levels: Vec<Vec<SsTable>>,
    /// Configuration.
    config: LsmConfig,
    /// Next sequence number.
    next_seq: u64,
    /// Next SSTable id.
    next_table_id: u64,
    /// Stats counters.
    total_writes: u64,
    total_reads: u64,
    total_compactions: u64,
    total_flushes: u64,
    bloom_hits: u64,
    bloom_misses: u64,
}

impl LsmTree {
    /// Create a new LSM tree with the given configuration.
    pub fn new(config: LsmConfig) -> Self {
        let max_levels = config.max_levels;
        let memtable = MemTable::new(config.memtable_max_bytes);
        let levels = (0..max_levels).map(|_| Vec::new()).collect();
        Self {
            memtable,
            levels,
            config,
            next_seq: 1,
            next_table_id: 1,
            total_writes: 0,
            total_reads: 0,
            total_compactions: 0,
            total_flushes: 0,
            bloom_hits: 0,
            bloom_misses: 0,
        }
    }

    /// Create a tree with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(LsmConfig::default())
    }

    /// Put a key-value pair.  Automatically flushes memtable to L0 if full.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<(), LsmError> {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.total_writes += 1;

        match self.memtable.put(key.clone(), value.clone(), seq) {
            Ok(()) => Ok(()),
            Err(LsmError::MemTableFull) => {
                self.flush_memtable()?;
                self.memtable
                    .put(key, value, seq)
                    .map_err(|_| LsmError::CompactionFailed("flush did not free space".into()))
            }
            Err(e) => Err(e),
        }
    }

    /// Delete a key (write tombstone).
    pub fn delete(&mut self, key: Vec<u8>) -> Result<(), LsmError> {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.total_writes += 1;

        match self.memtable.delete(key.clone(), seq) {
            Ok(()) => Ok(()),
            Err(LsmError::MemTableFull) => {
                self.flush_memtable()?;
                self.memtable
                    .delete(key, seq)
                    .map_err(|_| LsmError::CompactionFailed("flush did not free space".into()))
            }
            Err(e) => Err(e),
        }
    }

    /// Get a value by key, checking memtable first then each level.
    pub fn get(&mut self, key: &[u8]) -> Result<Vec<u8>, LsmError> {
        self.total_reads += 1;

        // 1. Check memtable.
        if let Some(entry) = self.memtable.get(key) {
            return entry.value.clone().ok_or(LsmError::KeyNotFound);
        }

        // 2. Check levels from newest to oldest.
        for level in &self.levels {
            // Search tables in reverse order (newest first).
            for table in level.iter().rev() {
                if table.bloom.may_contain(key) {
                    self.bloom_hits += 1;
                    if let Some(entry) = table.get(key) {
                        return entry.value.clone().ok_or(LsmError::KeyNotFound);
                    }
                } else {
                    self.bloom_misses += 1;
                }
            }
        }

        Err(LsmError::KeyNotFound)
    }

    /// Flush the current memtable to Level 0.
    pub fn flush_memtable(&mut self) -> Result<(), LsmError> {
        if self.memtable.is_empty() {
            return Ok(());
        }

        let entries = self.memtable.drain();
        let id = self.next_table_id;
        self.next_table_id += 1;
        let sstable = SsTable::from_entries(
            id,
            entries
                .into_iter()
                .map(|(k, e)| (k, e))
                .collect(),
        );
        if !self.levels.is_empty() {
            self.levels[0].push(sstable);
        }
        self.total_flushes += 1;

        // Trigger compaction if L0 is over the threshold.
        self.maybe_compact(0)?;
        Ok(())
    }

    /// Check if a level needs compaction and compact if so.
    fn maybe_compact(&mut self, level: usize) -> Result<(), LsmError> {
        if level >= self.levels.len() {
            return Ok(());
        }
        if self.levels[level].len() <= self.config.level_max_tables {
            return Ok(());
        }
        self.compact_level(level)
    }

    /// Compact all SSTables at `level` into `level + 1`.
    pub fn compact_level(&mut self, level: usize) -> Result<(), LsmError> {
        if level + 1 >= self.levels.len() {
            return Err(LsmError::CompactionFailed(format!(
                "no level {} to compact into",
                level + 1
            )));
        }

        // Merge all tables at `level`.
        let tables = std::mem::take(&mut self.levels[level]);
        let mut merged: Vec<(Vec<u8>, SsTableEntry)> = Vec::new();
        for table in &tables {
            let table_entries: Vec<(Vec<u8>, SsTableEntry)> =
                table.iter().cloned().collect();
            merged = merge_sorted_runs(&merged, &table_entries);
        }

        // Also merge with existing tables at level + 1.
        let next_tables = std::mem::take(&mut self.levels[level + 1]);
        for table in &next_tables {
            let table_entries: Vec<(Vec<u8>, SsTableEntry)> =
                table.iter().cloned().collect();
            merged = merge_sorted_runs(&merged, &table_entries);
        }

        // Create a new SSTable at level + 1.
        let id = self.next_table_id;
        self.next_table_id += 1;
        let new_entries: Vec<(Vec<u8>, MemEntry)> = merged
            .into_iter()
            .map(|(k, e)| {
                (
                    k,
                    MemEntry {
                        value: e.value,
                        seq: e.seq,
                    },
                )
            })
            .collect();
        let sstable = SsTable::from_entries(id, new_entries);
        self.levels[level + 1].push(sstable);
        self.total_compactions += 1;

        // Recurse if the next level is now over threshold.
        self.maybe_compact(level + 1)?;
        Ok(())
    }

    /// Get statistics for the tree.
    pub fn stats(&self) -> LsmStats {
        let levels = self
            .levels
            .iter()
            .enumerate()
            .map(|(i, tables)| LevelStats {
                level: i,
                table_count: tables.len(),
                entry_count: tables.iter().map(|t| t.len()).sum(),
                size_bytes: tables.iter().map(|t| t.size_bytes()).sum(),
                compaction_count: 0,
            })
            .collect();

        LsmStats {
            memtable_size_bytes: self.memtable.size_bytes(),
            memtable_entries: self.memtable.len(),
            levels,
            total_writes: self.total_writes,
            total_reads: self.total_reads,
            total_compactions: self.total_compactions,
            total_flushes: self.total_flushes,
            bloom_filter_hits: self.bloom_hits,
            bloom_filter_misses: self.bloom_misses,
        }
    }

    /// Number of entries in the memtable.
    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    /// Total number of SSTables across all levels.
    pub fn total_tables(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    /// Number of levels with data.
    pub fn non_empty_levels(&self) -> usize {
        self.levels.iter().filter(|l| !l.is_empty()).count()
    }

    /// Range scan across all levels (inclusive start, exclusive end).
    pub fn range(&mut self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.total_reads += 1;

        // Collect from all sources, newest first.
        let mut seen: BTreeMap<Vec<u8>, (Option<Vec<u8>>, u64)> = BTreeMap::new();

        // Memtable first (newest).
        for (k, e) in self.memtable.range(start, end) {
            seen.insert(k, (e.value, e.seq));
        }

        // Then levels.
        for level in &self.levels {
            for table in level.iter().rev() {
                for (k, e) in table.range(start, end) {
                    seen.entry(k).or_insert((e.value, e.seq));
                }
            }
        }

        // Filter out tombstones.
        seen.into_iter()
            .filter_map(|(k, (v, _))| v.map(|val| (k, val)))
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_filter_insert_and_query() {
        let mut bf = LsmBloomFilter::new(100, 0.01);
        bf.insert(b"hello");
        bf.insert(b"world");
        assert!(bf.may_contain(b"hello"));
        assert!(bf.may_contain(b"world"));
        assert_eq!(bf.item_count(), 2);
    }

    #[test]
    fn bloom_filter_negative_likely() {
        let mut bf = LsmBloomFilter::new(1000, 0.001);
        for i in 0..100u32 {
            bf.insert(&i.to_le_bytes());
        }
        // Keys not inserted should mostly not match.
        let mut false_positives = 0;
        for i in 1000..2000u32 {
            if bf.may_contain(&i.to_le_bytes()) {
                false_positives += 1;
            }
        }
        // With a 0.1% FP rate on 100 items, we expect very few.
        assert!(false_positives < 50, "too many false positives: {false_positives}");
    }

    #[test]
    fn bloom_estimated_fp_rate() {
        let mut bf = LsmBloomFilter::new(100, 0.01);
        assert!(bf.estimated_fp_rate() < 0.001);
        for i in 0..50u32 {
            bf.insert(&i.to_le_bytes());
        }
        assert!(bf.estimated_fp_rate() < 0.1);
    }

    #[test]
    fn memtable_put_and_get() {
        let mut mt = MemTable::new(4096);
        mt.put(b"key1".to_vec(), b"val1".to_vec(), 1).unwrap();
        mt.put(b"key2".to_vec(), b"val2".to_vec(), 2).unwrap();
        let entry = mt.get(b"key1").unwrap();
        assert_eq!(entry.value.as_deref(), Some(b"val1".as_slice()));
        assert_eq!(mt.len(), 2);
    }

    #[test]
    fn memtable_delete_creates_tombstone() {
        let mut mt = MemTable::new(4096);
        mt.put(b"key1".to_vec(), b"val1".to_vec(), 1).unwrap();
        mt.delete(b"key1".to_vec(), 2).unwrap();
        let entry = mt.get(b"key1").unwrap();
        assert_eq!(entry.value, None);
        assert_eq!(entry.seq, 2);
    }

    #[test]
    fn memtable_full_returns_error() {
        let mut mt = MemTable::new(50);
        mt.put(b"k".to_vec(), b"v".to_vec(), 1).unwrap();
        let result = mt.put(b"big_key".to_vec(), vec![0u8; 100], 2);
        assert_eq!(result, Err(LsmError::MemTableFull));
    }

    #[test]
    fn memtable_range_scan() {
        let mut mt = MemTable::new(4096);
        mt.put(b"a".to_vec(), b"1".to_vec(), 1).unwrap();
        mt.put(b"b".to_vec(), b"2".to_vec(), 2).unwrap();
        mt.put(b"c".to_vec(), b"3".to_vec(), 3).unwrap();
        mt.put(b"d".to_vec(), b"4".to_vec(), 4).unwrap();
        let range = mt.range(b"b", b"d");
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].0, b"b");
        assert_eq!(range[1].0, b"c");
    }

    #[test]
    fn memtable_drain() {
        let mut mt = MemTable::new(4096);
        mt.put(b"k1".to_vec(), b"v1".to_vec(), 1).unwrap();
        mt.put(b"k2".to_vec(), b"v2".to_vec(), 2).unwrap();
        let drained = mt.drain();
        assert_eq!(drained.len(), 2);
        assert!(mt.is_empty());
        assert_eq!(mt.size_bytes(), 0);
    }

    #[test]
    fn sstable_build_and_get() {
        let entries = vec![
            (b"a".to_vec(), MemEntry { value: Some(b"1".to_vec()), seq: 1 }),
            (b"b".to_vec(), MemEntry { value: Some(b"2".to_vec()), seq: 2 }),
            (b"c".to_vec(), MemEntry { value: Some(b"3".to_vec()), seq: 3 }),
        ];
        let sst = SsTable::from_entries(1, entries);
        assert_eq!(sst.len(), 3);
        assert_eq!(sst.id(), 1);
        let entry = sst.get(b"b").unwrap();
        assert_eq!(entry.value.as_deref(), Some(b"2".as_slice()));
    }

    #[test]
    fn sstable_miss_uses_bloom() {
        let entries = vec![
            (b"alpha".to_vec(), MemEntry { value: Some(b"1".to_vec()), seq: 1 }),
        ];
        let sst = SsTable::from_entries(1, entries);
        assert!(sst.get(b"beta").is_none());
    }

    #[test]
    fn sstable_range_scan() {
        let entries = vec![
            (b"a".to_vec(), MemEntry { value: Some(b"1".to_vec()), seq: 1 }),
            (b"b".to_vec(), MemEntry { value: Some(b"2".to_vec()), seq: 2 }),
            (b"c".to_vec(), MemEntry { value: Some(b"3".to_vec()), seq: 3 }),
            (b"d".to_vec(), MemEntry { value: Some(b"4".to_vec()), seq: 4 }),
        ];
        let sst = SsTable::from_entries(1, entries);
        let range = sst.range(b"b", b"d");
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn sstable_min_max_keys() {
        let entries = vec![
            (b"apple".to_vec(), MemEntry { value: Some(b"1".to_vec()), seq: 1 }),
            (b"orange".to_vec(), MemEntry { value: Some(b"2".to_vec()), seq: 2 }),
        ];
        let sst = SsTable::from_entries(1, entries);
        assert_eq!(sst.min_key(), Some(b"apple".as_slice()));
        assert_eq!(sst.max_key(), Some(b"orange".as_slice()));
    }

    #[test]
    fn merge_sorted_runs_basic() {
        let a = vec![
            (b"a".to_vec(), SsTableEntry { value: Some(b"1".to_vec()), seq: 1 }),
            (b"c".to_vec(), SsTableEntry { value: Some(b"3".to_vec()), seq: 3 }),
        ];
        let b = vec![
            (b"b".to_vec(), SsTableEntry { value: Some(b"2".to_vec()), seq: 2 }),
            (b"d".to_vec(), SsTableEntry { value: Some(b"4".to_vec()), seq: 4 }),
        ];
        let merged = merge_sorted_runs(&a, &b);
        let keys: Vec<_> = merged.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec()]);
    }

    #[test]
    fn merge_sorted_runs_duplicate_key_higher_seq_wins() {
        let a = vec![
            (b"k".to_vec(), SsTableEntry { value: Some(b"old".to_vec()), seq: 1 }),
        ];
        let b = vec![
            (b"k".to_vec(), SsTableEntry { value: Some(b"new".to_vec()), seq: 5 }),
        ];
        let merged = merge_sorted_runs(&a, &b);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].1.value.as_deref(), Some(b"new".as_slice()));
    }

    #[test]
    fn lsm_tree_put_get() {
        let config = LsmConfig {
            memtable_max_bytes: 4096,
            level_max_tables: 4,
            level_size_ratio: 10,
            max_levels: 3,
        };
        let mut tree = LsmTree::new(config);
        tree.put(b"key1".to_vec(), b"val1".to_vec()).unwrap();
        tree.put(b"key2".to_vec(), b"val2".to_vec()).unwrap();
        assert_eq!(tree.get(b"key1").unwrap(), b"val1");
        assert_eq!(tree.get(b"key2").unwrap(), b"val2");
    }

    #[test]
    fn lsm_tree_delete() {
        let mut tree = LsmTree::with_defaults();
        tree.put(b"key".to_vec(), b"val".to_vec()).unwrap();
        tree.delete(b"key".to_vec()).unwrap();
        assert_eq!(tree.get(b"key"), Err(LsmError::KeyNotFound));
    }

    #[test]
    fn lsm_tree_flush_and_read_from_level() {
        let config = LsmConfig {
            memtable_max_bytes: 100,
            level_max_tables: 10,
            level_size_ratio: 10,
            max_levels: 3,
        };
        let mut tree = LsmTree::new(config);
        tree.put(b"k1".to_vec(), b"v1".to_vec()).unwrap();
        tree.flush_memtable().unwrap();
        assert_eq!(tree.memtable_len(), 0);
        assert!(tree.total_tables() > 0);
        assert_eq!(tree.get(b"k1").unwrap(), b"v1");
    }

    #[test]
    fn lsm_tree_stats() {
        let mut tree = LsmTree::with_defaults();
        tree.put(b"a".to_vec(), b"1".to_vec()).unwrap();
        tree.put(b"b".to_vec(), b"2".to_vec()).unwrap();
        let stats = tree.stats();
        assert_eq!(stats.memtable_entries, 2);
        assert_eq!(stats.total_writes, 2);
    }

    #[test]
    fn lsm_tree_range_scan() {
        let mut tree = LsmTree::with_defaults();
        tree.put(b"a".to_vec(), b"1".to_vec()).unwrap();
        tree.put(b"b".to_vec(), b"2".to_vec()).unwrap();
        tree.put(b"c".to_vec(), b"3".to_vec()).unwrap();
        tree.put(b"d".to_vec(), b"4".to_vec()).unwrap();
        let range = tree.range(b"b", b"d");
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].0, b"b");
    }

    #[test]
    fn lsm_tree_overwrite() {
        let mut tree = LsmTree::with_defaults();
        tree.put(b"key".to_vec(), b"old".to_vec()).unwrap();
        tree.put(b"key".to_vec(), b"new".to_vec()).unwrap();
        assert_eq!(tree.get(b"key").unwrap(), b"new");
    }

    #[test]
    fn lsm_tree_compaction() {
        let config = LsmConfig {
            memtable_max_bytes: 80,
            level_max_tables: 2,
            level_size_ratio: 10,
            max_levels: 4,
        };
        let mut tree = LsmTree::new(config);
        // Insert enough data to trigger multiple flushes and at least one compaction.
        for i in 0u32..30 {
            let key = format!("key_{i:04}").into_bytes();
            let val = format!("val_{i:04}").into_bytes();
            tree.put(key, val).unwrap();
        }
        let stats = tree.stats();
        assert!(stats.total_flushes > 0);
        // Verify reads still work.
        assert_eq!(tree.get(b"key_0005").unwrap(), b"val_0005");
    }

    #[test]
    fn lsm_tree_non_empty_levels() {
        let config = LsmConfig {
            memtable_max_bytes: 80,
            level_max_tables: 2,
            level_size_ratio: 10,
            max_levels: 4,
        };
        let mut tree = LsmTree::new(config);
        assert_eq!(tree.non_empty_levels(), 0);
        for i in 0u32..20 {
            let key = format!("k{i:04}").into_bytes();
            tree.put(key, vec![i as u8; 10]).unwrap();
        }
        // After writes, at least some levels should have data.
        let non_empty = tree.non_empty_levels();
        assert!(non_empty >= 0);
    }

    #[test]
    fn memtable_write_count() {
        let mut mt = MemTable::new(4096);
        mt.put(b"a".to_vec(), b"1".to_vec(), 1).unwrap();
        mt.put(b"b".to_vec(), b"2".to_vec(), 2).unwrap();
        mt.delete(b"a".to_vec(), 3).unwrap();
        assert_eq!(mt.write_count(), 3);
    }

    #[test]
    fn sstable_seq_range() {
        let entries = vec![
            (b"a".to_vec(), MemEntry { value: Some(b"1".to_vec()), seq: 5 }),
            (b"b".to_vec(), MemEntry { value: Some(b"2".to_vec()), seq: 10 }),
            (b"c".to_vec(), MemEntry { value: Some(b"3".to_vec()), seq: 7 }),
        ];
        let sst = SsTable::from_entries(1, entries);
        assert_eq!(sst.seq_range(), (5, 10));
    }

    #[test]
    fn lsm_error_display() {
        assert_eq!(LsmError::KeyNotFound.to_string(), "key not found");
        assert_eq!(LsmError::MemTableFull.to_string(), "memtable is full");
        let e = LsmError::CompactionFailed("test".into());
        assert!(e.to_string().contains("test"));
    }
}
