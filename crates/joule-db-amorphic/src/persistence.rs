//! Module Persistence — checkpoint/restore for all content infrastructure modules.
//!
//! The core AmorphicStore has WAL-based durability via `DurableAmorphicStore`.
//! This module adds checkpoint/restore for the satellite modules:
//! - TrendingIndex
//! - EventProcessor (user profiles)
//! - VoiceprintStore
//! - ModerationPolicy + ModerationQueue
//! - ContentIdIndex
//! - TemporalStore
//! - CaptionStore
//! - AdTargetingEngine
//! - DistributionManager
//! - RoyaltyCalculator
//!
//! Strategy: periodic snapshots to disk (not WAL — these modules are either
//! reconstructible from events or small enough to snapshot entirely).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{AmorphicError, AmorphicResult, RecordId, Value};

/// Trait for modules that can checkpoint/restore their state.
pub trait Persistable {
    /// Serialize state to bytes.
    fn checkpoint(&self) -> AmorphicResult<Vec<u8>>;
    /// Restore state from bytes.
    fn restore(&mut self, data: &[u8]) -> AmorphicResult<()>;
    /// Module name (used as filename prefix).
    fn module_name(&self) -> &str;
}

/// Manages checkpoints for all persistent modules.
pub struct CheckpointManager {
    /// Directory for checkpoint files
    dir: PathBuf,
}

impl CheckpointManager {
    /// Open or create checkpoint directory.
    pub fn open(dir: impl AsRef<Path>) -> AmorphicResult<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| {
            AmorphicError::IngestionError(format!("Failed to create checkpoint dir: {}", e))
        })?;
        Ok(Self { dir })
    }

    /// Save a module's state to disk.
    pub fn save(&self, module: &dyn Persistable) -> AmorphicResult<()> {
        let data = module.checkpoint()?;
        let path = self.dir.join(format!("{}.ckpt", module.module_name()));

        // Write to temp file then atomic rename (crash-safe)
        let tmp_path = self.dir.join(format!("{}.ckpt.tmp", module.module_name()));
        std::fs::write(&tmp_path, &data).map_err(|e| {
            AmorphicError::IngestionError(format!("Checkpoint write failed: {}", e))
        })?;
        std::fs::rename(&tmp_path, &path).map_err(|e| {
            AmorphicError::IngestionError(format!("Checkpoint rename failed: {}", e))
        })?;

        Ok(())
    }

    /// Restore a module's state from disk (if checkpoint exists).
    pub fn restore(&self, module: &mut dyn Persistable) -> AmorphicResult<bool> {
        let path = self.dir.join(format!("{}.ckpt", module.module_name()));

        if !path.exists() {
            return Ok(false);
        }

        let data = std::fs::read(&path).map_err(|e| {
            AmorphicError::IngestionError(format!("Checkpoint read failed: {}", e))
        })?;

        module.restore(&data)?;
        Ok(true)
    }

    /// List all available checkpoints.
    pub fn list_checkpoints(&self) -> Vec<String> {
        std::fs::read_dir(&self.dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.ends_with(".ckpt") {
                            Some(name.trim_end_matches(".ckpt").to_string())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Delete a checkpoint.
    pub fn delete(&self, module_name: &str) -> AmorphicResult<()> {
        let path = self.dir.join(format!("{}.ckpt", module_name));
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                AmorphicError::IngestionError(format!("Checkpoint delete failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Save all modules in a collection.
    pub fn save_all(&self, modules: &[&dyn Persistable]) -> AmorphicResult<usize> {
        let mut saved = 0;
        for module in modules {
            self.save(*module)?;
            saved += 1;
        }
        Ok(saved)
    }
}

// ============================================================================
// Persistable implementations for content infrastructure modules
// ============================================================================

/// Serializable snapshot of TemporalStore.
#[derive(Serialize, Deserialize)]
struct TemporalSnapshot {
    fields: Vec<((RecordId, String), Vec<crate::temporal_fields::TemporalField>)>,
}

impl Persistable for crate::temporal_fields::TemporalStore {
    fn checkpoint(&self) -> AmorphicResult<Vec<u8>> {
        // TemporalStore fields are pub(crate), we can access from same crate
        let snapshot = TemporalSnapshot {
            fields: self
                .fields
                .iter()
                .map(|((rid, field), versions)| ((*rid, field.clone()), versions.clone()))
                .collect(),
        };
        serde_json::to_vec(&snapshot).map_err(|e| {
            AmorphicError::IngestionError(format!("Temporal checkpoint failed: {}", e))
        })
    }

    fn restore(&mut self, data: &[u8]) -> AmorphicResult<()> {
        let snapshot: TemporalSnapshot = serde_json::from_slice(data).map_err(|e| {
            AmorphicError::IngestionError(format!("Temporal restore failed: {}", e))
        })?;
        self.fields.clear();
        for ((rid, field), versions) in snapshot.fields {
            self.fields.insert((rid, field), versions);
        }
        Ok(())
    }

    fn module_name(&self) -> &str {
        "temporal_fields"
    }
}

/// Serializable snapshot for CaptionStore.
#[derive(Serialize, Deserialize)]
struct CaptionSnapshot {
    tracks: Vec<(String, Vec<crate::accessibility::CaptionTrack>)>,
}

impl Persistable for crate::accessibility::CaptionStore {
    fn checkpoint(&self) -> AmorphicResult<Vec<u8>> {
        let snapshot = CaptionSnapshot {
            tracks: self
                .tracks
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        serde_json::to_vec(&snapshot).map_err(|e| {
            AmorphicError::IngestionError(format!("Caption checkpoint failed: {}", e))
        })
    }

    fn restore(&mut self, data: &[u8]) -> AmorphicResult<()> {
        let snapshot: CaptionSnapshot = serde_json::from_slice(data).map_err(|e| {
            AmorphicError::IngestionError(format!("Caption restore failed: {}", e))
        })?;
        self.tracks.clear();
        for (k, v) in snapshot.tracks {
            self.tracks.insert(k, v);
        }
        Ok(())
    }

    fn module_name(&self) -> &str {
        "captions"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_fields::{TemporalField, TemporalStore};
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_manager_save_restore() {
        let dir = tempdir().unwrap();
        let mgr = CheckpointManager::open(dir.path()).unwrap();

        // Create and populate a TemporalStore
        let mut store = TemporalStore::new();
        store.set(
            1,
            "streaming_rights",
            TemporalField::global(Value::Bool(true)),
        );
        store.set(
            2,
            "price",
            TemporalField::bounded(Value::Float(9.99), 0, u64::MAX),
        );

        // Save
        mgr.save(&store).unwrap();

        // Verify file exists
        let checkpoints = mgr.list_checkpoints();
        assert!(checkpoints.contains(&"temporal_fields".to_string()));

        // Restore into a new store
        let mut restored = TemporalStore::new();
        let found = mgr.restore(&mut restored).unwrap();
        assert!(found);

        // Verify data survived
        assert!(restored.can_stream(1, "US", 1000));
        let price = restored.query_valid_at(2, "price", 1000, None);
        assert_eq!(price, Some(&Value::Float(9.99)));
    }

    #[test]
    fn test_checkpoint_missing() {
        let dir = tempdir().unwrap();
        let mgr = CheckpointManager::open(dir.path()).unwrap();

        let mut store = TemporalStore::new();
        let found = mgr.restore(&mut store).unwrap();
        assert!(!found); // No checkpoint file → returns false
    }

    #[test]
    fn test_atomic_write() {
        let dir = tempdir().unwrap();
        let mgr = CheckpointManager::open(dir.path()).unwrap();

        let mut store = TemporalStore::new();
        store.set(1, "test", TemporalField::global(Value::Bool(true)));

        // Save twice — should overwrite atomically
        mgr.save(&store).unwrap();
        mgr.save(&store).unwrap();

        let checkpoints = mgr.list_checkpoints();
        assert_eq!(checkpoints.len(), 1); // Only one checkpoint file
    }

    #[test]
    fn test_caption_store_persistence() {
        let dir = tempdir().unwrap();
        let mgr = CheckpointManager::open(dir.path()).unwrap();

        let mut store = crate::accessibility::CaptionStore::new();
        store.add_track(crate::accessibility::CaptionTrack {
            content_id: "movie_1".into(),
            language: "en".into(),
            captions: vec![crate::accessibility::Caption {
                start_ms: 0,
                end_ms: 5000,
                text: "Hello world".into(),
                speaker: None,
                language: "en".into(),
            }],
            source: crate::accessibility::CaptionSource::Human,
        });

        mgr.save(&store).unwrap();

        let mut restored = crate::accessibility::CaptionStore::new();
        mgr.restore(&mut restored).unwrap();

        let results = restored.search("hello");
        assert_eq!(results.len(), 1);
    }
}
