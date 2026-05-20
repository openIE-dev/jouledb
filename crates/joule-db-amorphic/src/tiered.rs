//! Tiered Storage for Amorphic Database
//!
//! Implements a three-tier storage hierarchy:
//! - **Hot**: In-memory for fastest access (< 1µs)
//! - **Warm**: Memory-mapped files for medium access (< 10µs)
//! - **Cold**: Disk storage for archival (< 1ms)
//!
//! ## Memory Management
//!
//! The system automatically promotes/demotes data based on access patterns:
//! - Frequently accessed data moves to hot tier
//! - Idle data gradually cools to warm, then cold
//! - Memory budget controls hot tier size
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_amorphic::tiered::{TieredStore, TieredConfig};
//!
//! let config = TieredConfig {
//!     hot_budget_bytes: 100 * 1024 * 1024,  // 100MB hot tier
//!     warm_budget_bytes: 1024 * 1024 * 1024, // 1GB warm tier
//!     storage_path: PathBuf::from("./data"),
//!     ..Default::default()
//! };
//!
//! let store = TieredStore::new(config)?;
//! ```

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{AmorphicError, AmorphicResult, DIMENSION};
use joule_db_hdc::BinaryHV;

/// Storage tier for a hologram
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    /// In-memory, fastest access
    Hot,
    /// Memory-mapped, medium access
    Warm,
    /// On-disk, slowest access
    Cold,
}

/// Configuration for tiered storage
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Maximum bytes for hot tier (in-memory)
    pub hot_budget_bytes: usize,
    /// Maximum bytes for warm tier (mmap)
    pub warm_budget_bytes: usize,
    /// Path for persistent storage
    pub storage_path: PathBuf,
    /// Access count threshold to promote from cold to warm
    pub cold_to_warm_threshold: u32,
    /// Access count threshold to promote from warm to hot
    pub warm_to_hot_threshold: u32,
    /// Seconds of inactivity before demotion consideration
    pub demotion_idle_seconds: u64,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            hot_budget_bytes: 100 * 1024 * 1024,   // 100MB
            warm_budget_bytes: 1024 * 1024 * 1024, // 1GB
            storage_path: PathBuf::from("./amorphic_data"),
            cold_to_warm_threshold: 3,
            warm_to_hot_threshold: 10,
            demotion_idle_seconds: 300, // 5 minutes
        }
    }
}

/// Metadata for a stored hologram
#[derive(Debug, Clone)]
struct HologramMeta {
    /// Current storage tier
    tier: StorageTier,
    /// Number of times accessed
    access_count: u32,
    /// Last access timestamp (unix seconds)
    last_access: u64,
    /// Size in bytes
    size_bytes: usize,
    /// Offset in cold storage file (if cold)
    cold_offset: Option<u64>,
}

/// Size of a single hologram in bytes (10000 bits = 1250 bytes, rounded to u64 = 157 * 8 = 1256)
const HOLOGRAM_SIZE_BYTES: usize = ((DIMENSION + 63) / 64) * 8;

/// Tiered storage manager for holograms
pub struct TieredStore {
    config: TieredConfig,

    /// Hot tier: in-memory holograms
    hot: HashMap<u64, BinaryHV>,

    /// Warm tier: memory-mapped region
    warm_file: Option<File>,
    warm_map: Option<memmap2::MmapMut>,
    warm_index: HashMap<u64, usize>, // id -> offset in mmap
    warm_next_slot: usize,

    /// Cold tier: disk file with index
    cold_file: Option<File>,
    cold_index: HashMap<u64, u64>, // id -> offset in file
    cold_next_offset: u64,

    /// Metadata for all holograms
    meta: HashMap<u64, HologramMeta>,

    /// Current memory usage
    hot_bytes: usize,
    warm_bytes: usize,

    /// Statistics
    stats: TieredStats,
}

/// Statistics for tiered storage operations
#[derive(Debug, Default)]
pub struct TieredStats {
    pub hot_hits: AtomicU64,
    pub warm_hits: AtomicU64,
    pub cold_hits: AtomicU64,
    pub promotions: AtomicU64,
    pub demotions: AtomicU64,
}

impl TieredStore {
    /// Create a new tiered store
    pub fn new(config: TieredConfig) -> AmorphicResult<Self> {
        // Ensure storage directory exists
        fs::create_dir_all(&config.storage_path).map_err(|e| {
            AmorphicError::IngestionError(format!("Failed to create storage dir: {}", e))
        })?;

        // Initialize warm tier file
        let warm_path = config.storage_path.join("warm.dat");
        let warm_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&warm_path)
            .map_err(|e| {
                AmorphicError::IngestionError(format!("Failed to open warm file: {}", e))
            })?;

        // Pre-allocate warm file to budget size
        warm_file
            .set_len(config.warm_budget_bytes as u64)
            .map_err(|e| {
                AmorphicError::IngestionError(format!("Failed to allocate warm file: {}", e))
            })?;

        // Memory-map the warm file
        let warm_map = unsafe {
            memmap2::MmapMut::map_mut(&warm_file).map_err(|e| {
                AmorphicError::IngestionError(format!("Failed to mmap warm file: {}", e))
            })?
        };

        // Initialize cold tier file
        let cold_path = config.storage_path.join("cold.dat");
        let cold_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(true)
            .open(&cold_path)
            .map_err(|e| {
                AmorphicError::IngestionError(format!("Failed to open cold file: {}", e))
            })?;

        Ok(Self {
            config,
            hot: HashMap::new(),
            warm_file: Some(warm_file),
            warm_map: Some(warm_map),
            warm_index: HashMap::new(),
            warm_next_slot: 0,
            cold_file: Some(cold_file),
            cold_index: HashMap::new(),
            cold_next_offset: 0,
            meta: HashMap::new(),
            hot_bytes: 0,
            warm_bytes: 0,
            stats: TieredStats::default(),
        })
    }

    /// Create an in-memory only store (no persistence)
    pub fn in_memory() -> Self {
        Self {
            config: TieredConfig::default(),
            hot: HashMap::new(),
            warm_file: None,
            warm_map: None,
            warm_index: HashMap::new(),
            warm_next_slot: 0,
            cold_file: None,
            cold_index: HashMap::new(),
            cold_next_offset: 0,
            meta: HashMap::new(),
            hot_bytes: 0,
            warm_bytes: 0,
            stats: TieredStats::default(),
        }
    }

    /// Store a hologram (starts in hot tier)
    pub fn put(&mut self, id: u64, hologram: BinaryHV) -> AmorphicResult<()> {
        let size = HOLOGRAM_SIZE_BYTES;

        // Check if we need to evict from hot tier
        while self.hot_bytes + size > self.config.hot_budget_bytes && !self.hot.is_empty() {
            self.demote_coldest_hot()?;
        }

        // Insert into hot tier
        self.hot.insert(id, hologram);
        self.hot_bytes += size;

        // Update metadata
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.meta.insert(
            id,
            HologramMeta {
                tier: StorageTier::Hot,
                access_count: 0,
                last_access: now,
                size_bytes: size,
                cold_offset: None,
            },
        );

        Ok(())
    }

    /// Get a hologram (may promote from lower tiers)
    pub fn get(&mut self, id: u64) -> AmorphicResult<Option<&BinaryHV>> {
        // Update access metadata
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(meta) = self.meta.get_mut(&id) {
            meta.access_count += 1;
            meta.last_access = now;

            match meta.tier {
                StorageTier::Hot => {
                    self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(self.hot.get(&id));
                }
                StorageTier::Warm => {
                    self.stats.warm_hits.fetch_add(1, Ordering::Relaxed);

                    // Check if should promote to hot
                    if meta.access_count >= self.config.warm_to_hot_threshold {
                        self.promote_to_hot(id)?;
                        return Ok(self.hot.get(&id));
                    }

                    // Read from warm tier
                    if let Some(offset) = self.warm_index.get(&id) {
                        if let Some(ref map) = self.warm_map {
                            let start = *offset;
                            let end = start + HOLOGRAM_SIZE_BYTES;
                            if end <= map.len() {
                                let hv = deserialize_hologram(&map[start..end])?;
                                // Temporarily promote to hot for return
                                self.hot.insert(id, hv);
                                return Ok(self.hot.get(&id));
                            }
                        }
                    }
                }
                StorageTier::Cold => {
                    self.stats.cold_hits.fetch_add(1, Ordering::Relaxed);

                    // Check if should promote
                    if meta.access_count >= self.config.cold_to_warm_threshold {
                        self.promote_to_warm(id)?;
                        // Recursively get from warm
                        return self.get(id);
                    }

                    // Read from cold storage
                    if let Some(offset) = meta.cold_offset {
                        let hv = self.read_cold(offset)?;
                        // Temporarily put in hot for return
                        self.hot.insert(id, hv);
                        return Ok(self.hot.get(&id));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get a hologram without promotion (read-only)
    pub fn peek(&self, id: u64) -> AmorphicResult<Option<BinaryHV>> {
        if let Some(meta) = self.meta.get(&id) {
            match meta.tier {
                StorageTier::Hot => {
                    return Ok(self.hot.get(&id).cloned());
                }
                StorageTier::Warm => {
                    if let Some(offset) = self.warm_index.get(&id) {
                        if let Some(ref map) = self.warm_map {
                            let start = *offset;
                            let end = start + HOLOGRAM_SIZE_BYTES;
                            if end <= map.len() {
                                return Ok(Some(deserialize_hologram(&map[start..end])?));
                            }
                        }
                    }
                }
                StorageTier::Cold => {
                    if let Some(offset) = meta.cold_offset {
                        return Ok(Some(self.read_cold(offset)?));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Get current tier for an id
    pub fn tier(&self, id: u64) -> Option<StorageTier> {
        self.meta.get(&id).map(|m| m.tier)
    }

    /// Get statistics
    pub fn stats(&self) -> &TieredStats {
        &self.stats
    }

    /// Get memory usage by tier
    pub fn memory_usage(&self) -> (usize, usize) {
        (self.hot_bytes, self.warm_bytes)
    }

    /// Number of items in each tier
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let hot = self.hot.len();
        let warm = self.warm_index.len();
        let cold = self.cold_index.len();
        (hot, warm, cold)
    }

    /// Demote the coldest (least recently accessed) item from hot tier
    fn demote_coldest_hot(&mut self) -> AmorphicResult<()> {
        // Find least recently accessed hot item
        let coldest_id = self
            .meta
            .iter()
            .filter(|(_, m)| m.tier == StorageTier::Hot)
            .min_by_key(|(_, m)| m.last_access)
            .map(|(id, _)| *id);

        if let Some(id) = coldest_id {
            self.demote_to_warm(id)?;
        }

        Ok(())
    }

    /// Demote from hot to warm tier
    fn demote_to_warm(&mut self, id: u64) -> AmorphicResult<()> {
        if let Some(hv) = self.hot.remove(&id) {
            self.hot_bytes -= HOLOGRAM_SIZE_BYTES;

            // Check warm budget
            if self.warm_bytes + HOLOGRAM_SIZE_BYTES > self.config.warm_budget_bytes {
                // Need to demote something from warm to cold first
                self.demote_coldest_warm()?;
            }

            // Write to warm tier
            if let Some(ref mut map) = self.warm_map {
                let offset = self.warm_next_slot * HOLOGRAM_SIZE_BYTES;
                if offset + HOLOGRAM_SIZE_BYTES <= map.len() {
                    serialize_hologram(&hv, &mut map[offset..offset + HOLOGRAM_SIZE_BYTES]);
                    self.warm_index.insert(id, offset);
                    self.warm_next_slot += 1;
                    self.warm_bytes += HOLOGRAM_SIZE_BYTES;

                    if let Some(meta) = self.meta.get_mut(&id) {
                        meta.tier = StorageTier::Warm;
                    }

                    self.stats.demotions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Demote the coldest item from warm to cold tier
    fn demote_coldest_warm(&mut self) -> AmorphicResult<()> {
        let coldest_id = self
            .meta
            .iter()
            .filter(|(_, m)| m.tier == StorageTier::Warm)
            .min_by_key(|(_, m)| m.last_access)
            .map(|(id, _)| *id);

        if let Some(id) = coldest_id {
            self.demote_to_cold(id)?;
        }

        Ok(())
    }

    /// Demote from warm to cold tier
    fn demote_to_cold(&mut self, id: u64) -> AmorphicResult<()> {
        if let Some(offset) = self.warm_index.remove(&id) {
            self.warm_bytes -= HOLOGRAM_SIZE_BYTES;

            // Read from warm
            if let Some(ref map) = self.warm_map {
                let hv = deserialize_hologram(&map[offset..offset + HOLOGRAM_SIZE_BYTES])?;

                // Write to cold
                let cold_offset = self.write_cold(&hv)?;
                self.cold_index.insert(id, cold_offset);

                if let Some(meta) = self.meta.get_mut(&id) {
                    meta.tier = StorageTier::Cold;
                    meta.cold_offset = Some(cold_offset);
                }

                self.stats.demotions.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Promote from cold to warm tier
    fn promote_to_warm(&mut self, id: u64) -> AmorphicResult<()> {
        if let Some(meta) = self.meta.get(&id) {
            if let Some(offset) = meta.cold_offset {
                let hv = self.read_cold(offset)?;

                // Ensure warm has space
                if self.warm_bytes + HOLOGRAM_SIZE_BYTES > self.config.warm_budget_bytes {
                    self.demote_coldest_warm()?;
                }

                // Write to warm
                if let Some(ref mut map) = self.warm_map {
                    let warm_offset = self.warm_next_slot * HOLOGRAM_SIZE_BYTES;
                    if warm_offset + HOLOGRAM_SIZE_BYTES <= map.len() {
                        serialize_hologram(
                            &hv,
                            &mut map[warm_offset..warm_offset + HOLOGRAM_SIZE_BYTES],
                        );
                        self.warm_index.insert(id, warm_offset);
                        self.warm_next_slot += 1;
                        self.warm_bytes += HOLOGRAM_SIZE_BYTES;

                        // Remove from cold index (data stays on disk but won't be used)
                        self.cold_index.remove(&id);

                        if let Some(meta) = self.meta.get_mut(&id) {
                            meta.tier = StorageTier::Warm;
                            meta.cold_offset = None;
                        }

                        self.stats.promotions.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        Ok(())
    }

    /// Promote from warm to hot tier
    fn promote_to_hot(&mut self, id: u64) -> AmorphicResult<()> {
        if let Some(offset) = self.warm_index.get(&id) {
            if let Some(ref map) = self.warm_map {
                let hv = deserialize_hologram(&map[*offset..*offset + HOLOGRAM_SIZE_BYTES])?;

                // Ensure hot has space
                while self.hot_bytes + HOLOGRAM_SIZE_BYTES > self.config.hot_budget_bytes
                    && !self.hot.is_empty()
                {
                    self.demote_coldest_hot()?;
                }

                // Move to hot
                self.hot.insert(id, hv);
                self.hot_bytes += HOLOGRAM_SIZE_BYTES;

                // Remove from warm index (data stays in mmap but won't be used)
                self.warm_index.remove(&id);
                self.warm_bytes -= HOLOGRAM_SIZE_BYTES;

                if let Some(meta) = self.meta.get_mut(&id) {
                    meta.tier = StorageTier::Hot;
                }

                self.stats.promotions.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Write hologram to cold storage, return offset
    fn write_cold(&mut self, hv: &BinaryHV) -> AmorphicResult<u64> {
        if let Some(ref mut file) = self.cold_file {
            let offset = self.cold_next_offset;

            let mut buffer = vec![0u8; HOLOGRAM_SIZE_BYTES];
            serialize_hologram(hv, &mut buffer);

            file.write_all(&buffer)
                .map_err(|e| AmorphicError::IngestionError(format!("Cold write failed: {}", e)))?;

            self.cold_next_offset += HOLOGRAM_SIZE_BYTES as u64;

            Ok(offset)
        } else {
            Err(AmorphicError::IngestionError(
                "Cold storage not available".to_string(),
            ))
        }
    }

    /// Read hologram from cold storage at offset
    fn read_cold(&self, offset: u64) -> AmorphicResult<BinaryHV> {
        if let Some(ref file) = self.cold_file {
            let mut file = file.try_clone().map_err(|e| {
                AmorphicError::QueryError(format!("Failed to clone cold file: {}", e))
            })?;

            file.seek(SeekFrom::Start(offset))
                .map_err(|e| AmorphicError::QueryError(format!("Cold seek failed: {}", e)))?;

            let mut buffer = vec![0u8; HOLOGRAM_SIZE_BYTES];
            file.read_exact(&mut buffer)
                .map_err(|e| AmorphicError::QueryError(format!("Cold read failed: {}", e)))?;

            deserialize_hologram(&buffer)
        } else {
            Err(AmorphicError::QueryError(
                "Cold storage not available".to_string(),
            ))
        }
    }

    /// Flush warm tier to disk
    pub fn flush(&self) -> AmorphicResult<()> {
        if let Some(ref map) = self.warm_map {
            map.flush()
                .map_err(|e| AmorphicError::IngestionError(format!("Flush failed: {}", e)))?;
        }
        if let Some(ref file) = self.cold_file {
            file.sync_all()
                .map_err(|e| AmorphicError::IngestionError(format!("Cold sync failed: {}", e)))?;
        }
        Ok(())
    }
}

/// Serialize hologram to bytes (standalone to avoid borrow issues)
fn serialize_hologram(hv: &BinaryHV, buffer: &mut [u8]) {
    let words = hv.as_words();
    for (i, &word) in words.iter().enumerate() {
        let offset = i * 8;
        if offset + 8 <= buffer.len() {
            buffer[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// Deserialize hologram from bytes (standalone)
fn deserialize_hologram(buffer: &[u8]) -> AmorphicResult<BinaryHV> {
    // Use from_bytes which expects raw byte data
    Ok(BinaryHV::from_bytes(buffer, DIMENSION))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_hot_tier_basic() {
        let mut store = TieredStore::in_memory();
        let hv = BinaryHV::random(DIMENSION, 42);

        store.put(1, hv.clone()).unwrap();

        assert_eq!(store.tier(1), Some(StorageTier::Hot));

        let retrieved = store.get(1).unwrap().unwrap();
        assert_eq!(retrieved.as_words(), hv.as_words());
    }

    #[test]
    fn test_tiered_storage_with_disk() {
        let dir = tempdir().unwrap();
        let config = TieredConfig {
            hot_budget_bytes: HOLOGRAM_SIZE_BYTES * 2, // Only 2 items in hot
            warm_budget_bytes: HOLOGRAM_SIZE_BYTES * 4,
            storage_path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut store = TieredStore::new(config).unwrap();

        // Insert 5 items - should overflow hot tier
        for i in 0..5 {
            let hv = BinaryHV::random(DIMENSION, i);
            store.put(i as u64, hv).unwrap();
        }

        // Check tier distribution
        let (hot, warm, cold) = store.tier_counts();
        assert!(hot <= 2, "Hot tier should have at most 2 items");
        assert!(warm > 0 || cold > 0, "Some items should be in lower tiers");
    }

    #[test]
    fn test_promotion_on_access() {
        let dir = tempdir().unwrap();
        let config = TieredConfig {
            hot_budget_bytes: HOLOGRAM_SIZE_BYTES * 2,
            warm_budget_bytes: HOLOGRAM_SIZE_BYTES * 4,
            storage_path: dir.path().to_path_buf(),
            warm_to_hot_threshold: 3, // Promote after 3 accesses
            ..Default::default()
        };

        let mut store = TieredStore::new(config).unwrap();

        // Insert 3 items to fill hot tier
        for i in 0..3 {
            store.put(i as u64, BinaryHV::random(DIMENSION, i)).unwrap();
        }

        // Item 0 should have been demoted
        let initial_tier = store.tier(0);

        // Access item 0 multiple times to promote it
        for _ in 0..5 {
            store.get(0).unwrap();
        }

        // Should be back in hot tier after enough accesses
        let final_tier = store.tier(0);
        assert_eq!(final_tier, Some(StorageTier::Hot));
    }
}
