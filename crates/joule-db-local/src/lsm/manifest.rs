//! Manifest for the LSM-Tree engine.
//!
//! Tracks which SSTable files exist at which levels.
//! Persisted as a JSON file with atomic rename for crash safety.

use super::sstable::SSTableMeta;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Manifest tracking all SSTables organized by level.
#[derive(Debug)]
pub struct Manifest {
    dir: PathBuf,
    pub levels: Vec<Vec<SSTableMeta>>,
    pub next_sst_id: u64,
}

/// Serializable manifest entry for JSON persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct ManifestEntry {
    id: u64,
    level: usize,
    filename: String,
    first_key: Vec<u8>,
    last_key: Vec<u8>,
    entry_count: usize,
    size_bytes: u64,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ManifestData {
    next_sst_id: u64,
    entries: Vec<ManifestEntry>,
}

impl Manifest {
    /// Create a new empty manifest.
    pub fn new(dir: &Path, max_levels: usize) -> Self {
        Self {
            dir: dir.to_path_buf(),
            levels: (0..max_levels).map(|_| Vec::new()).collect(),
            next_sst_id: 1,
        }
    }

    /// Load manifest from disk, or create a new one if it doesn't exist.
    pub fn load_or_create(dir: &Path, max_levels: usize) -> io::Result<Self> {
        let manifest_path = dir.join("manifest.json");
        if manifest_path.exists() {
            Self::load(dir, max_levels)
        } else {
            Ok(Self::new(dir, max_levels))
        }
    }

    /// Load manifest from disk.
    pub fn load(dir: &Path, max_levels: usize) -> io::Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let data = fs::read_to_string(&manifest_path)?;
        let manifest_data: ManifestData = serde_json::from_str(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut levels: Vec<Vec<SSTableMeta>> = (0..max_levels).map(|_| Vec::new()).collect();

        for entry in manifest_data.entries {
            let level = entry.level;
            if level >= levels.len() {
                levels.resize_with(level + 1, Vec::new);
            }
            levels[level].push(SSTableMeta {
                id: entry.id,
                level: entry.level,
                path: dir.join(&entry.filename),
                first_key: entry.first_key,
                last_key: entry.last_key,
                entry_count: entry.entry_count,
                size_bytes: entry.size_bytes,
            });
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            levels,
            next_sst_id: manifest_data.next_sst_id,
        })
    }

    /// Save manifest to disk with atomic rename.
    pub fn save(&self) -> io::Result<()> {
        let mut entries = Vec::new();
        for level_tables in &self.levels {
            for meta in level_tables {
                entries.push(ManifestEntry {
                    id: meta.id,
                    level: meta.level,
                    filename: meta
                        .path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                    first_key: meta.first_key.clone(),
                    last_key: meta.last_key.clone(),
                    entry_count: meta.entry_count,
                    size_bytes: meta.size_bytes,
                });
            }
        }

        let manifest_data = ManifestData {
            next_sst_id: self.next_sst_id,
            entries,
        };

        let json = serde_json::to_string_pretty(&manifest_data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Atomic write: write to temp file, then rename
        let temp_path = self.dir.join("manifest.json.tmp");
        let manifest_path = self.dir.join("manifest.json");
        fs::write(&temp_path, json)?;
        fs::rename(&temp_path, &manifest_path)?;

        Ok(())
    }

    /// Allocate a new SSTable ID and return (id, file_path).
    pub fn allocate_sst(&mut self) -> (u64, PathBuf) {
        let id = self.next_sst_id;
        self.next_sst_id += 1;
        let path = self.dir.join(format!("{:06}.sst", id));
        (id, path)
    }

    /// Add an SSTable to a level.
    pub fn add_sstable(&mut self, meta: SSTableMeta) {
        let level = meta.level;
        if level >= self.levels.len() {
            self.levels.resize_with(level + 1, Vec::new);
        }
        self.levels[level].push(meta);
    }

    /// Remove SSTables by their IDs.
    pub fn remove_sstables(&mut self, ids: &[u64]) {
        for level in &mut self.levels {
            level.retain(|m| !ids.contains(&m.id));
        }
    }

    /// Get total number of SSTables across all levels.
    pub fn total_sstables(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    /// Get total size of a level in bytes.
    pub fn level_size(&self, level: usize) -> u64 {
        self.levels
            .get(level)
            .map_or(0, |l| l.iter().map(|m| m.size_bytes).sum())
    }

    /// Get the number of SSTables in a given level.
    pub fn level_count(&self, level: usize) -> usize {
        self.levels.get(level).map_or(0, |l| l.len())
    }

    /// Get the directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_manifest_new() {
        let dir = TempDir::new().unwrap();
        let m = Manifest::new(dir.path(), 7);
        assert_eq!(m.levels.len(), 7);
        assert_eq!(m.next_sst_id, 1);
        assert_eq!(m.total_sstables(), 0);
    }

    #[test]
    fn test_manifest_allocate_sst() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 7);
        let (id1, path1) = m.allocate_sst();
        let (id2, _) = m.allocate_sst();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert!(path1.to_string_lossy().contains("000001.sst"));
    }

    #[test]
    fn test_manifest_save_load() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 7);

        let (id, path) = m.allocate_sst();
        m.add_sstable(SSTableMeta {
            id,
            level: 0,
            path: path.clone(),
            first_key: b"aaa".to_vec(),
            last_key: b"zzz".to_vec(),
            entry_count: 100,
            size_bytes: 4096,
        });

        m.save().unwrap();

        let loaded = Manifest::load(dir.path(), 7).unwrap();
        assert_eq!(loaded.next_sst_id, 2);
        assert_eq!(loaded.levels[0].len(), 1);
        assert_eq!(loaded.levels[0][0].first_key, b"aaa");
        assert_eq!(loaded.levels[0][0].entry_count, 100);
    }

    #[test]
    fn test_manifest_remove_sstables() {
        let dir = TempDir::new().unwrap();
        let mut m = Manifest::new(dir.path(), 7);

        for _ in 0..3 {
            let (id, path) = m.allocate_sst();
            m.add_sstable(SSTableMeta {
                id,
                level: 0,
                path,
                first_key: vec![],
                last_key: vec![],
                entry_count: 0,
                size_bytes: 0,
            });
        }
        assert_eq!(m.level_count(0), 3);

        m.remove_sstables(&[1, 3]);
        assert_eq!(m.level_count(0), 1);
        assert_eq!(m.levels[0][0].id, 2);
    }
}
