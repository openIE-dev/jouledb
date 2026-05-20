//! LSM-Tree (Log-Structured Merge-Tree) storage engine.
//!
//! Write-optimized storage engine alternative to B-tree. Ideal for
//! write-heavy workloads like time-series, logging, and IoT telemetry.
//!
//! **Read path**: MemTable → Immutable MemTables → L0 SSTables → L1 → L2 ... (bloom filter skip)
//! **Write path**: MemTable → (flush at threshold) → SSTable L0 → (compact) → L1/L2

pub mod compaction;
pub mod manifest;
pub mod memtable;
pub mod sstable;

use compaction::{CompactionConfig, CompactionResult};
use manifest::Manifest;
use memtable::MemTable;
use sstable::{SSTableReader, SSTableWriter};

use std::fs;
use std::io;
use std::ops::Bound;
use std::path::{Path, PathBuf};

#[cfg(feature = "adaptive")]
use std::sync::Arc;

/// Configuration for the LSM engine.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Flush memtable when it exceeds this size in bytes.
    pub memtable_size_threshold: usize,
    /// Number of L0 SSTables that triggers compaction.
    pub l0_compaction_trigger: usize,
    /// Size multiplier between levels.
    pub level_size_multiplier: usize,
    /// Maximum number of levels.
    pub max_levels: usize,
    /// Target SSTable size in bytes.
    pub target_sst_size: u64,
    /// Target L1 size in bytes.
    pub l1_target_size: u64,
    /// Bloom filter false-positive rate.
    pub bloom_fp_rate: f64,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            memtable_size_threshold: 4 * 1024 * 1024, // 4MB
            l0_compaction_trigger: 4,
            level_size_multiplier: 10,
            max_levels: 7,
            target_sst_size: 4 * 1024 * 1024, // 4MB
            l1_target_size: 64 * 1024 * 1024, // 64MB
            bloom_fp_rate: 0.01,
        }
    }
}

/// LSM-Tree storage engine.
pub struct LsmEngine {
    dir: PathBuf,
    active: MemTable,
    immutable: Vec<MemTable>,
    manifest: Manifest,
    config: LsmConfig,
    /// Optional adaptive controller for energy-aware parameter tuning.
    #[cfg(feature = "adaptive")]
    adaptive: Option<Arc<joule_db_energy::AdaptiveController>>,
}

impl LsmEngine {
    /// Open or create an LSM engine at the given directory.
    pub fn open(dir: &Path, config: LsmConfig) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let manifest = Manifest::load_or_create(dir, config.max_levels)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            active: MemTable::new(),
            immutable: Vec::new(),
            manifest,
            config,
            #[cfg(feature = "adaptive")]
            adaptive: None,
        })
    }

    /// Open with an adaptive controller for energy-aware behavior modulation.
    #[cfg(feature = "adaptive")]
    pub fn open_adaptive(
        dir: &Path,
        config: LsmConfig,
        controller: Arc<joule_db_energy::AdaptiveController>,
    ) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let manifest = Manifest::load_or_create(dir, config.max_levels)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            active: MemTable::new(),
            immutable: Vec::new(),
            manifest,
            config,
            adaptive: Some(controller),
        })
    }

    /// Get current adaptive parameters (if adaptive mode is enabled).
    #[cfg(feature = "adaptive")]
    pub fn adaptive_params(&self) -> Option<joule_db_energy::AdaptiveParams> {
        self.adaptive.as_ref().map(|c| c.current_params())
    }

    /// Get a value by key.
    ///
    /// Searches: active memtable → immutable memtables → L0 (newest first) → L1 → ...
    pub fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        // 1. Check active memtable
        if let Some(entry) = self.active.get(key) {
            return Ok(entry.clone()); // Some(value) or None (tombstone)
        }

        // 2. Check immutable memtables (newest first)
        for imm in self.immutable.iter().rev() {
            if let Some(entry) = imm.get(key) {
                return Ok(entry.clone());
            }
        }

        // 3. Check SSTables level by level
        for level_tables in &self.manifest.levels {
            // Search newest first within each level (by ID descending)
            let mut tables: Vec<&sstable::SSTableMeta> = level_tables.iter().collect();
            tables.sort_by(|a, b| b.id.cmp(&a.id));

            for meta in tables {
                // Quick key range check
                if key < meta.first_key.as_slice() || key > meta.last_key.as_slice() {
                    continue;
                }

                let reader = SSTableReader::open(&meta.path)?;

                // Bloom filter check
                if !reader.may_contain(key) {
                    continue;
                }

                if let Some(value_opt) = reader.get(key)? {
                    return Ok(value_opt); // Some(value) or None (tombstone)
                }
            }
        }

        Ok(None)
    }

    /// Insert a key-value pair.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> io::Result<()> {
        self.active.put(key, value);
        self.maybe_flush()?;
        Ok(())
    }

    /// Delete a key (write a tombstone).
    pub fn delete(&mut self, key: Vec<u8>) -> io::Result<()> {
        self.active.delete(key);
        self.maybe_flush()?;
        Ok(())
    }

    /// Range scan over keys in [start, end] (inclusive).
    /// Returns sorted, deduplicated results with tombstones resolved.
    pub fn range(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Collect entries from all sources, tagged with recency (lower = newer)
        let mut all_entries: Vec<(Vec<u8>, Option<Vec<u8>>, usize)> = Vec::new();
        let mut priority = 0;

        // Active memtable (newest)
        for (k, v) in self
            .active
            .range(Bound::Included(start), Bound::Included(end))
        {
            all_entries.push((k.clone(), v.clone(), priority));
        }
        priority += 1;

        // Immutable memtables (newer first)
        for imm in self.immutable.iter().rev() {
            for (k, v) in imm.range(Bound::Included(start), Bound::Included(end)) {
                all_entries.push((k.clone(), v.clone(), priority));
            }
            priority += 1;
        }

        // SSTables level by level
        for level_tables in &self.manifest.levels {
            let mut tables: Vec<&sstable::SSTableMeta> = level_tables.iter().collect();
            tables.sort_by(|a, b| b.id.cmp(&a.id));

            for meta in tables {
                // Quick range overlap check
                if meta.last_key.as_slice() < start || meta.first_key.as_slice() > end {
                    continue;
                }

                let reader = SSTableReader::open(&meta.path)?;
                for (k, v) in reader.range(start, end)? {
                    all_entries.push((k, v, priority));
                }
                priority += 1;
            }
        }

        // Sort by key, then by priority (lower = newer)
        all_entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));

        // Deduplicate: keep first occurrence of each key (newest), skip tombstones
        let mut result = Vec::new();
        let mut last_key: Option<&[u8]> = None;
        for (key, value, _) in &all_entries {
            if last_key == Some(key.as_slice()) {
                continue; // Duplicate key, skip older version
            }
            last_key = Some(key.as_slice());
            if let Some(val) = value {
                result.push((key.clone(), val.clone()));
            }
            // Tombstone (None): key is deleted, don't include
        }

        Ok(result)
    }

    /// Scan all entries (range over entire keyspace).
    pub fn scan(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.range(&[], &[0xFF; 32])
    }

    /// Force flush the active memtable to an L0 SSTable.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.active.is_empty() {
            return Ok(());
        }

        let entries = self.active.drain();
        self.write_memtable_to_sst(entries)?;
        Ok(())
    }

    /// Force compaction. Returns true if compaction was performed.
    pub fn compact(&mut self) -> io::Result<bool> {
        let compaction_config = self.compaction_config();
        if let Some(level) = compaction::needs_compaction(&self.manifest, &compaction_config) {
            compaction::compact_level(&mut self.manifest, level, &compaction_config)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Sync: flush memtable and run compaction if needed.
    pub fn sync(&mut self) -> io::Result<()> {
        self.flush()?;
        // Flush any immutable memtables
        self.flush_immutables()?;
        // Run compaction until stable
        let compaction_config = self.compaction_config();
        while let Some(level) = compaction::needs_compaction(&self.manifest, &compaction_config) {
            compaction::compact_level(&mut self.manifest, level, &compaction_config)?;
        }
        Ok(())
    }

    /// Number of entries in the active memtable.
    pub fn memtable_size(&self) -> usize {
        self.active.len()
    }

    /// Total number of SSTables across all levels.
    pub fn total_sstables(&self) -> usize {
        self.manifest.total_sstables()
    }

    /// Get SSTable count per level.
    pub fn level_stats(&self) -> Vec<(usize, usize, u64)> {
        (0..self.config.max_levels)
            .map(|l| (l, self.manifest.level_count(l), self.manifest.level_size(l)))
            .collect()
    }

    // --- Private helpers ---

    fn maybe_flush(&mut self) -> io::Result<()> {
        let threshold = self.effective_memtable_threshold();
        if self.active.size_bytes() >= threshold {
            // Freeze active memtable and create a new one
            let frozen = std::mem::replace(&mut self.active, MemTable::new());
            self.immutable.push(frozen);
            self.flush_immutables()?;

            // Check if compaction is needed
            let compaction_config = self.compaction_config();
            if let Some(level) = compaction::needs_compaction(&self.manifest, &compaction_config) {
                compaction::compact_level(&mut self.manifest, level, &compaction_config)?;
            }
        }
        Ok(())
    }

    /// Get the effective memtable threshold (adaptive or static).
    fn effective_memtable_threshold(&self) -> usize {
        #[cfg(feature = "adaptive")]
        if let Some(ref ctrl) = self.adaptive {
            return ctrl.current_params().memtable_threshold;
        }
        self.config.memtable_size_threshold
    }

    fn flush_immutables(&mut self) -> io::Result<()> {
        while let Some(imm) = self.immutable.pop() {
            let entries = imm.drain_owned();
            self.write_memtable_to_sst(entries)?;
        }
        Ok(())
    }

    fn write_memtable_to_sst(
        &mut self,
        entries: std::collections::BTreeMap<Vec<u8>, memtable::MemEntry>,
    ) -> io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let (id, path) = self.manifest.allocate_sst();
        let mut writer = SSTableWriter::new(&path, entries.len())?;

        for (key, value) in &entries {
            writer.add(key, value.as_deref())?;
        }

        let meta = writer.finish(id, 0)?; // Always flush to L0
        self.manifest.add_sstable(meta);
        self.manifest.save()?;

        Ok(())
    }

    fn compaction_config(&self) -> CompactionConfig {
        let l0_trigger = self.effective_l0_trigger();
        CompactionConfig {
            l0_compaction_trigger: l0_trigger,
            level_size_multiplier: self.config.level_size_multiplier,
            max_levels: self.config.max_levels,
            l1_target_size: self.config.l1_target_size,
            target_sst_size: self.config.target_sst_size,
        }
    }

    /// Get the effective L0 compaction trigger (adaptive or static).
    fn effective_l0_trigger(&self) -> usize {
        #[cfg(feature = "adaptive")]
        if let Some(ref ctrl) = self.adaptive {
            return ctrl.current_params().l0_compaction_trigger;
        }
        self.config.l0_compaction_trigger
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config() -> LsmConfig {
        LsmConfig {
            memtable_size_threshold: 1024, // 1KB for fast flush in tests
            l0_compaction_trigger: 4,
            level_size_multiplier: 10,
            max_levels: 4,
            target_sst_size: 4096,
            l1_target_size: 16384,
            bloom_fp_rate: 0.01,
        }
    }

    #[test]
    fn test_lsm_put_get() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        engine.put(b"key1".to_vec(), b"val1".to_vec()).unwrap();
        engine.put(b"key2".to_vec(), b"val2".to_vec()).unwrap();

        assert_eq!(engine.get(b"key1").unwrap(), Some(b"val1".to_vec()));
        assert_eq!(engine.get(b"key2").unwrap(), Some(b"val2".to_vec()));
        assert_eq!(engine.get(b"key3").unwrap(), None);
    }

    #[test]
    fn test_lsm_overwrite() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        engine.put(b"key".to_vec(), b"old".to_vec()).unwrap();
        engine.put(b"key".to_vec(), b"new".to_vec()).unwrap();

        assert_eq!(engine.get(b"key").unwrap(), Some(b"new".to_vec()));
    }

    #[test]
    fn test_lsm_delete() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        engine.put(b"key".to_vec(), b"val".to_vec()).unwrap();
        assert_eq!(engine.get(b"key").unwrap(), Some(b"val".to_vec()));

        engine.delete(b"key".to_vec()).unwrap();
        assert_eq!(engine.get(b"key").unwrap(), None);
    }

    #[test]
    fn test_lsm_range_scan() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        for i in 0u8..10 {
            engine.put(vec![i], vec![i * 10]).unwrap();
        }

        let results = engine.range(&[3], &[7]).unwrap();
        assert_eq!(results.len(), 5); // 3, 4, 5, 6, 7
        assert_eq!(results[0], (vec![3], vec![30]));
        assert_eq!(results[4], (vec![7], vec![70]));
    }

    #[test]
    fn test_lsm_flush_to_sstable() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        engine.put(b"a".to_vec(), b"1".to_vec()).unwrap();
        engine.put(b"b".to_vec(), b"2".to_vec()).unwrap();
        engine.flush().unwrap();

        assert_eq!(engine.memtable_size(), 0);
        assert_eq!(engine.total_sstables(), 1);

        // Data should still be readable from SSTable
        assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), Some(b"2".to_vec()));
    }

    #[test]
    fn test_lsm_persistence_across_reopen() {
        let dir = TempDir::new().unwrap();

        // Write data and flush
        {
            let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();
            engine
                .put(b"persist_key".to_vec(), b"persist_val".to_vec())
                .unwrap();
            engine.flush().unwrap();
        }

        // Reopen and verify
        {
            let engine = LsmEngine::open(dir.path(), test_config()).unwrap();
            assert_eq!(
                engine.get(b"persist_key").unwrap(),
                Some(b"persist_val".to_vec())
            );
        }
    }

    #[test]
    fn test_lsm_auto_flush_on_threshold() {
        let dir = TempDir::new().unwrap();
        let config = LsmConfig {
            memtable_size_threshold: 100, // Very low threshold
            ..test_config()
        };
        let mut engine = LsmEngine::open(dir.path(), config).unwrap();

        // Write enough data to trigger auto-flush
        for i in 0..20u32 {
            let key = format!("key_{:04}", i);
            let val = format!("value_{:04}", i);
            engine.put(key.into_bytes(), val.into_bytes()).unwrap();
        }

        // Should have flushed at least once
        assert!(engine.total_sstables() > 0);
    }

    #[test]
    fn test_lsm_compaction_cycle() {
        let dir = TempDir::new().unwrap();
        let config = LsmConfig {
            memtable_size_threshold: 64, // Very low for quick flush
            l0_compaction_trigger: 4,
            max_levels: 4,
            target_sst_size: 4096,
            l1_target_size: 16384,
            ..test_config()
        };
        let mut engine = LsmEngine::open(dir.path(), config).unwrap();

        // Write enough to trigger multiple flushes and compaction
        for i in 0..100u32 {
            let key = format!("key_{:06}", i);
            let val = format!("value_{:06}", i);
            engine.put(key.into_bytes(), val.into_bytes()).unwrap();
        }

        engine.sync().unwrap();

        // Verify all data is still accessible after compaction
        for i in 0..100u32 {
            let key = format!("key_{:06}", i);
            let val = format!("value_{:06}", i);
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(val.into_bytes()),
                "Failed to get {}",
                key
            );
        }
    }

    #[test]
    fn test_lsm_delete_after_flush() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        engine.put(b"key".to_vec(), b"val".to_vec()).unwrap();
        engine.flush().unwrap();

        // Delete after flush — tombstone is in memtable, data in SSTable
        engine.delete(b"key".to_vec()).unwrap();
        assert_eq!(engine.get(b"key").unwrap(), None);

        // Flush the tombstone too
        engine.flush().unwrap();
        assert_eq!(engine.get(b"key").unwrap(), None);
    }

    #[test]
    fn test_lsm_range_with_deletes() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        for i in 0u8..10 {
            engine.put(vec![i], vec![i]).unwrap();
        }
        engine.flush().unwrap();

        // Delete some keys
        engine.delete(vec![3]).unwrap();
        engine.delete(vec![5]).unwrap();
        engine.delete(vec![7]).unwrap();

        let results = engine.range(&[0], &[9]).unwrap();
        assert_eq!(results.len(), 7); // 10 - 3 deleted = 7
        assert!(!results.iter().any(|(k, _)| k == &[3]));
        assert!(!results.iter().any(|(k, _)| k == &[5]));
        assert!(!results.iter().any(|(k, _)| k == &[7]));
    }

    #[test]
    fn test_lsm_large_dataset() {
        let dir = TempDir::new().unwrap();
        let config = LsmConfig {
            memtable_size_threshold: 256,
            l0_compaction_trigger: 3,
            max_levels: 4,
            target_sst_size: 2048,
            l1_target_size: 8192,
            ..test_config()
        };
        let mut engine = LsmEngine::open(dir.path(), config).unwrap();

        // Insert 500 keys
        for i in 0..500u32 {
            let key = format!("k{:06}", i);
            let val = format!("v{:06}", i);
            engine.put(key.into_bytes(), val.into_bytes()).unwrap();
        }

        engine.sync().unwrap();

        // Verify all 500 keys
        for i in 0..500u32 {
            let key = format!("k{:06}", i);
            let val = format!("v{:06}", i);
            assert_eq!(engine.get(key.as_bytes()).unwrap(), Some(val.into_bytes()));
        }

        // Range scan middle portion
        let start = format!("k{:06}", 200);
        let end = format!("k{:06}", 299);
        let results = engine.range(start.as_bytes(), end.as_bytes()).unwrap();
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_lsm_level_stats() {
        let dir = TempDir::new().unwrap();
        let mut engine = LsmEngine::open(dir.path(), test_config()).unwrap();

        let stats = engine.level_stats();
        assert_eq!(stats.len(), 4); // max_levels = 4 in test_config
        assert_eq!(stats[0].1, 0); // L0 count = 0

        engine.put(b"a".to_vec(), b"1".to_vec()).unwrap();
        engine.flush().unwrap();

        let stats = engine.level_stats();
        assert_eq!(stats[0].1, 1); // L0 count = 1
        assert!(stats[0].2 > 0); // L0 size > 0
    }
}
