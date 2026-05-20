//! Leveled compaction for the LSM-Tree engine.
//!
//! L0: Up to `l0_compaction_trigger` SSTables (may have overlapping keys).
//! L1+: Non-overlapping key ranges, 10x size ratio per level.
//!
//! Compaction merges SSTables from one level into the next, producing
//! new non-overlapping SSTables and discarding obsolete files.

use super::manifest::Manifest;
use super::sstable::{SSTableMeta, SSTableReader, SSTableWriter};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

/// Configuration for the compaction strategy.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Number of L0 SSTables that triggers compaction.
    pub l0_compaction_trigger: usize,
    /// Size multiplier between levels (e.g., 10 means L1 is 10x L0).
    pub level_size_multiplier: usize,
    /// Maximum number of levels.
    pub max_levels: usize,
    /// Target size for L1 in bytes.
    pub l1_target_size: u64,
    /// Target SSTable size in bytes for output.
    pub target_sst_size: u64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            l0_compaction_trigger: 4,
            level_size_multiplier: 10,
            max_levels: 7,
            l1_target_size: 64 * 1024 * 1024, // 64MB
            target_sst_size: 4 * 1024 * 1024, // 4MB
        }
    }
}

/// Determine if compaction is needed and which level to compact.
pub fn needs_compaction(manifest: &Manifest, config: &CompactionConfig) -> Option<usize> {
    // Check L0 first — trigger when too many SSTables
    if manifest.level_count(0) >= config.l0_compaction_trigger {
        return Some(0);
    }

    // Check L1+ — trigger when level exceeds size target
    let mut target_size = config.l1_target_size;
    for level in 1..config.max_levels.saturating_sub(1) {
        if manifest.level_size(level) > target_size {
            return Some(level);
        }
        target_size *= config.level_size_multiplier as u64;
    }

    None
}

/// Run compaction from `level` into `level + 1`.
///
/// For L0→L1: merges ALL L0 SSTables with overlapping L1 SSTables.
/// For Ln→Ln+1: picks one SSTable from Ln, merges with overlapping Ln+1 SSTables.
///
/// Returns the IDs of removed SSTables and newly created SSTableMetas.
pub fn compact_level(
    manifest: &mut Manifest,
    level: usize,
    config: &CompactionConfig,
) -> io::Result<CompactionResult> {
    let target_level = level + 1;

    if level >= config.max_levels.saturating_sub(1) {
        return Ok(CompactionResult::default());
    }

    // Select input SSTables from the source level
    let (source_ids, key_range) = if level == 0 {
        // L0: take ALL SSTables
        let l0 = &manifest.levels[0];
        if l0.is_empty() {
            return Ok(CompactionResult::default());
        }
        let ids: Vec<u64> = l0.iter().map(|m| m.id).collect();
        let min_key = l0
            .iter()
            .map(|m| m.first_key.as_slice())
            .min()
            .unwrap_or_default()
            .to_vec();
        let max_key = l0
            .iter()
            .map(|m| m.last_key.as_slice())
            .max()
            .unwrap_or_default()
            .to_vec();
        (ids, (min_key, max_key))
    } else {
        // Ln: pick the SSTable with the smallest first_key (round-robin-ish)
        let ln = &manifest.levels[level];
        if ln.is_empty() {
            return Ok(CompactionResult::default());
        }
        // We checked is_empty() above so min_by is guaranteed Some,
        // but we express the fallback explicitly so the impl never
        // panics — the substrate's panic-free invariant must hold
        // even on impossible branches.
        let pick = match ln.iter().min_by(|a, b| a.first_key.cmp(&b.first_key)) {
            Some(p) => p,
            None => return Ok(CompactionResult::default()),
        };
        let ids = vec![pick.id];
        let range = (pick.first_key.clone(), pick.last_key.clone());
        (ids, range)
    };

    // Find overlapping SSTables in the target level
    let target_ids: Vec<u64> = manifest
        .levels
        .get(target_level)
        .map(|tl| {
            tl.iter()
                .filter(|m| ranges_overlap(&m.first_key, &m.last_key, &key_range.0, &key_range.1))
                .map(|m| m.id)
                .collect()
        })
        .unwrap_or_default();

    // Collect all input SSTable paths, sorted by ID descending (newest first = highest priority)
    let mut input_metas: Vec<&SSTableMeta> = Vec::new();
    for id in &source_ids {
        if let Some(meta) = find_meta(manifest, *id) {
            input_metas.push(meta);
        }
    }
    for id in &target_ids {
        if let Some(meta) = find_meta(manifest, *id) {
            input_metas.push(meta);
        }
    }
    // Sort by ID descending so newest SSTable (highest ID) gets index 0 (highest priority)
    input_metas.sort_by(|a, b| b.id.cmp(&a.id));

    if input_metas.is_empty() {
        return Ok(CompactionResult::default());
    }

    // Multi-way merge: read all entries, deduplicate (latest wins)
    // Source SSTables are "newer" (take priority), listed first
    let merged = merge_sstables(&input_metas, target_level >= config.max_levels - 1)?;

    // Write output SSTables
    let dir = manifest.dir().to_path_buf();
    let new_metas = write_merged_entries(&dir, manifest, &merged, target_level, config)?;

    // Collect all removed IDs
    let mut removed_ids: Vec<u64> = source_ids;
    removed_ids.extend(&target_ids);

    // Delete old SSTable files
    for id in &removed_ids {
        if let Some(meta) = find_meta_cloned(manifest, *id) {
            let _ = fs::remove_file(&meta.path);
        }
    }

    // Update manifest
    manifest.remove_sstables(&removed_ids);
    for meta in &new_metas {
        manifest.add_sstable(meta.clone());
    }
    manifest.save()?;

    Ok(CompactionResult {
        removed_ids,
        new_sstables: new_metas,
    })
}

/// Result of a compaction operation.
#[derive(Debug, Default)]
pub struct CompactionResult {
    pub removed_ids: Vec<u64>,
    pub new_sstables: Vec<SSTableMeta>,
}

/// Check if two key ranges overlap.
fn ranges_overlap(a_first: &[u8], a_last: &[u8], b_first: &[u8], b_last: &[u8]) -> bool {
    a_first <= b_last && b_first <= a_last
}

/// Find an SSTableMeta by id across all levels.
fn find_meta(manifest: &Manifest, id: u64) -> Option<&SSTableMeta> {
    for level in &manifest.levels {
        for meta in level {
            if meta.id == id {
                return Some(meta);
            }
        }
    }
    None
}

/// Find and clone an SSTableMeta by id.
fn find_meta_cloned(manifest: &Manifest, id: u64) -> Option<SSTableMeta> {
    find_meta(manifest, id).cloned()
}

/// Multi-way merge of SSTables. Earlier entries in `metas` take priority (newer).
/// If `drop_tombstones` is true, tombstones are removed (only safe at max level).
fn merge_sstables(
    metas: &[&SSTableMeta],
    drop_tombstones: bool,
) -> io::Result<Vec<(Vec<u8>, Option<Vec<u8>>)>> {
    // Read all entries, tag with SSTable index for priority (lower index = newer)
    let mut all_entries: Vec<(Vec<u8>, Option<Vec<u8>>, usize)> = Vec::new();

    for (idx, meta) in metas.iter().enumerate() {
        let reader = SSTableReader::open(&meta.path)?;
        let entries = reader.iter()?;
        for (key, value) in entries {
            all_entries.push((key, value, idx));
        }
    }

    // Sort by key, then by SSTable index (lower = newer = higher priority)
    all_entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));

    // Deduplicate: keep first occurrence of each key (newest)
    let mut merged: BTreeMap<Vec<u8>, Option<Vec<u8>>> = BTreeMap::new();
    for (key, value, _idx) in all_entries {
        merged.entry(key).or_insert(value);
    }

    // Convert to sorted vec, optionally dropping tombstones
    let result: Vec<(Vec<u8>, Option<Vec<u8>>)> = merged
        .into_iter()
        .filter(|(_, v)| !drop_tombstones || v.is_some())
        .collect();

    Ok(result)
}

/// Write merged entries into one or more SSTables at the target level.
fn write_merged_entries(
    dir: &Path,
    manifest: &mut Manifest,
    entries: &[(Vec<u8>, Option<Vec<u8>>)],
    target_level: usize,
    config: &CompactionConfig,
) -> io::Result<Vec<SSTableMeta>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let mut new_metas = Vec::new();
    let entries_per_sst = estimate_entries_per_sst(entries, config.target_sst_size);
    let mut chunks = entries.chunks(entries_per_sst.max(1));

    while let Some(chunk) = chunks.next() {
        let (id, path) = manifest.allocate_sst();
        let mut writer = SSTableWriter::new(&path, chunk.len())?;
        for (key, value) in chunk {
            writer.add(key, value.as_deref())?;
        }
        let meta = writer.finish(id, target_level)?;
        new_metas.push(meta);
    }

    Ok(new_metas)
}

/// Estimate how many entries fit in one SSTable of target_sst_size.
fn estimate_entries_per_sst(entries: &[(Vec<u8>, Option<Vec<u8>>)], target_size: u64) -> usize {
    if entries.is_empty() {
        return 1;
    }
    // Approximate average entry size
    let total_size: usize = entries
        .iter()
        .take(100) // Sample first 100 entries
        .map(|(k, v)| k.len() + v.as_ref().map_or(1, |v| v.len() + 5) + 5)
        .sum();
    let sample_count = entries.len().min(100);
    let avg_entry_size = total_size / sample_count.max(1);
    (target_size as usize / avg_entry_size.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_sst(
        dir: &Path,
        manifest: &mut Manifest,
        level: usize,
        entries: &[(&[u8], Option<&[u8]>)],
    ) -> SSTableMeta {
        let (id, path) = manifest.allocate_sst();
        let mut writer = SSTableWriter::new(&path, entries.len()).unwrap();
        for (key, value) in entries {
            writer.add(key, *value).unwrap();
        }
        let meta = writer.finish(id, level).unwrap();
        manifest.add_sstable(meta.clone());
        meta
    }

    #[test]
    fn test_needs_compaction_l0_trigger() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 7);
        let config = CompactionConfig {
            l0_compaction_trigger: 4,
            ..Default::default()
        };

        // No compaction needed yet
        assert_eq!(needs_compaction(&m, &config), None);

        // Add 4 L0 SSTables
        for i in 0..4 {
            make_sst(dir.path(), &mut m, 0, &[(&[i], Some(&[i]))]);
        }

        assert_eq!(needs_compaction(&m, &config), Some(0));
    }

    #[test]
    fn test_compact_l0_to_l1() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 7);
        let config = CompactionConfig::default();

        // Create 4 L0 SSTables with overlapping keys
        make_sst(
            dir.path(),
            &mut m,
            0,
            &[(b"a", Some(b"1")), (b"c", Some(b"3"))],
        );
        make_sst(
            dir.path(),
            &mut m,
            0,
            &[(b"b", Some(b"2")), (b"d", Some(b"4"))],
        );
        make_sst(
            dir.path(),
            &mut m,
            0,
            &[(b"a", Some(b"10")), (b"e", Some(b"5"))],
        );
        make_sst(dir.path(), &mut m, 0, &[(b"c", Some(b"30"))]);

        assert_eq!(m.level_count(0), 4);
        assert_eq!(m.level_count(1), 0);

        m.save().unwrap();
        let result = compact_level(&mut m, 0, &config).unwrap();

        // All L0 SSTables should be removed
        assert_eq!(result.removed_ids.len(), 4);
        assert_eq!(m.level_count(0), 0);
        // L1 should have at least 1 SSTable
        assert!(m.level_count(1) > 0);

        // Verify merged data: latest writes win
        // SSTable 3 wrote a=10, SSTable 4 wrote c=30
        let mut all_entries = Vec::new();
        for meta in &m.levels[1] {
            let reader = SSTableReader::open(&meta.path).unwrap();
            all_entries.extend(reader.iter().unwrap());
        }
        all_entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Should have: a=10, b=2, c=30, d=4, e=5
        assert_eq!(all_entries.len(), 5);
        let map: std::collections::HashMap<Vec<u8>, Vec<u8>> = all_entries
            .into_iter()
            .filter_map(|(k, v)| v.map(|val| (k, val)))
            .collect();
        assert_eq!(map[b"a".as_slice()], b"10");
        assert_eq!(map[b"b".as_slice()], b"2");
        assert_eq!(map[b"c".as_slice()], b"30");
        assert_eq!(map[b"d".as_slice()], b"4");
        assert_eq!(map[b"e".as_slice()], b"5");
    }

    #[test]
    fn test_compact_tombstone_merge() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 7);
        let config = CompactionConfig::default();

        // L0: one SSTable with data, another with tombstone for same key
        make_sst(
            dir.path(),
            &mut m,
            0,
            &[(b"key1", Some(b"val1")), (b"key2", Some(b"val2"))],
        );
        make_sst(dir.path(), &mut m, 0, &[(b"key1", None)]); // delete key1
        make_sst(dir.path(), &mut m, 0, &[(b"key3", Some(b"val3"))]);
        make_sst(dir.path(), &mut m, 0, &[(b"key4", Some(b"val4"))]);

        m.save().unwrap();
        compact_level(&mut m, 0, &config).unwrap();

        let mut all_entries = Vec::new();
        for meta in &m.levels[1] {
            let reader = SSTableReader::open(&meta.path).unwrap();
            all_entries.extend(reader.iter().unwrap());
        }
        all_entries.sort_by(|a, b| a.0.cmp(&b.0));

        // key1 should be a tombstone (None), not dropped (not at max level)
        let key1 = all_entries.iter().find(|(k, _)| k == b"key1").unwrap();
        assert_eq!(key1.1, None);
        // key2, key3, key4 should have data
        let key2 = all_entries.iter().find(|(k, _)| k == b"key2").unwrap();
        assert_eq!(key2.1, Some(b"val2".to_vec()));
    }

    #[test]
    fn test_compact_drops_tombstones_at_max_level() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 3); // max_levels=3 → max level index 2
        let config = CompactionConfig {
            max_levels: 3,
            l0_compaction_trigger: 2,
            ..Default::default()
        };

        // Put one SSTable at level 1 with data + tombstone
        make_sst(
            dir.path(),
            &mut m,
            1,
            &[(b"a", Some(b"1")), (b"b", None), (b"c", Some(b"3"))],
        );

        m.save().unwrap();
        // Compact L1→L2 (L2 is max level, so tombstones should be dropped)
        compact_level(&mut m, 1, &config).unwrap();

        let mut all_entries = Vec::new();
        for meta in &m.levels[2] {
            let reader = SSTableReader::open(&meta.path).unwrap();
            all_entries.extend(reader.iter().unwrap());
        }

        // Tombstone for "b" should be dropped at max level
        assert!(all_entries.iter().all(|(k, _)| k != b"b"));
        assert_eq!(all_entries.len(), 2); // a, c
    }

    #[test]
    fn test_ranges_overlap() {
        assert!(ranges_overlap(b"a", b"d", b"c", b"f")); // partial overlap
        assert!(ranges_overlap(b"a", b"f", b"c", b"d")); // containment
        assert!(!ranges_overlap(b"a", b"b", b"c", b"d")); // no overlap
        assert!(ranges_overlap(b"a", b"c", b"c", b"d")); // edge touch
    }
}
