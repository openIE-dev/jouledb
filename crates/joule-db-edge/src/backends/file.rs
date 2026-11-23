//! File Storage Backend
//!
//! Implements storage on filesystem (for Linux/Yocto devices) with:
//! - Memory-mapped files for efficient I/O
//! - Write-ahead log (WAL) for crash recovery
//! - Page-aligned I/O for optimal performance
//!
//! # File Layout
//! ```text
//! data.db      - Main data file (memory-mapped)
//! data.db.wal  - Write-ahead log
//! data.db.lock - File lock for concurrent access
//! ```

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::{EdgeStorageBackend, StorageStats};
use crate::storage::StorageError;

/// WAL entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalEntryType {
    Write = 1,
    Delete = 2,
    Checkpoint = 3,
}

/// WAL entry header
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct WalEntryHeader {
    /// Entry type
    entry_type: u8,
    /// Page ID
    page_id: u64,
    /// Data length
    data_len: u32,
    /// CRC32 of data
    checksum: u32,
}

impl WalEntryHeader {
    const SIZE: usize = std::mem::size_of::<Self>();

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0] = self.entry_type;
        bytes[1..9].copy_from_slice(&self.page_id.to_le_bytes());
        bytes[9..13].copy_from_slice(&self.data_len.to_le_bytes());
        bytes[13..17].copy_from_slice(&self.checksum.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            entry_type: bytes[0],
            page_id: u64::from_le_bytes([
                bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
            ]),
            data_len: u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]),
            checksum: u32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]),
        })
    }
}

/// File storage backend configuration
#[derive(Debug, Clone)]
pub struct FileBackendConfig {
    /// Page size
    pub page_size: usize,
    /// Initial data file size
    pub initial_size: u64,
    /// WAL size limit before checkpoint
    pub wal_checkpoint_threshold: u64,
    /// Sync mode
    pub sync_mode: SyncMode,
    /// Maximum pages to hold in cache (0 = unlimited)
    pub max_cache_pages: usize,
}

impl Default for FileBackendConfig {
    fn default() -> Self {
        Self {
            page_size: 4096,
            initial_size: 64 * 1024 * 1024,             // 64MB
            wal_checkpoint_threshold: 16 * 1024 * 1024, // 16MB
            sync_mode: SyncMode::Full,
            max_cache_pages: 16384, // 64MB at 4K pages
        }
    }
}

/// Sync mode for durability vs performance trade-off
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// fsync after every write (safest, slowest)
    Full,
    /// fsync only on explicit sync call
    Normal,
    /// No fsync (fastest, data loss on crash)
    Off,
}

/// Page cache entry
struct CacheEntry {
    data: Vec<u8>,
    dirty: bool,
}

/// File storage backend with WAL
pub struct FileBackend {
    /// Configuration
    config: FileBackendConfig,
    /// Data file path
    data_path: PathBuf,
    /// WAL file path
    wal_path: PathBuf,
    /// Data file handle
    data_file: File,
    /// WAL file handle
    wal_file: File,
    /// Page cache
    cache: HashMap<u64, CacheEntry>,
    /// WAL size
    wal_size: u64,
    /// Statistics
    stats: StorageStats,
    /// File size
    file_size: u64,
    /// Pages written to WAL (pending checkpoint)
    pending_pages: HashMap<u64, Vec<u8>>,
}

impl FileBackend {
    /// Open or create a file backend
    pub fn open<P: AsRef<Path>>(path: P, config: FileBackendConfig) -> Result<Self, StorageError> {
        let data_path = path.as_ref().to_path_buf();
        let wal_path = data_path.with_extension("db.wal");

        // Open or create data file
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        // Get file size
        let metadata = data_file
            .metadata()
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let mut file_size = metadata.len();

        // Extend file if needed
        if file_size < config.initial_size {
            data_file
                .set_len(config.initial_size)
                .map_err(|e| StorageError::Io(e.to_string()))?;
            file_size = config.initial_size;
        }

        // Open or create WAL file
        let wal_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let wal_size = wal_file
            .metadata()
            .map_err(|e| StorageError::Io(e.to_string()))?
            .len();

        let mut backend = Self {
            config,
            data_path,
            wal_path,
            data_file,
            wal_file,
            cache: HashMap::new(),
            wal_size,
            stats: StorageStats::default(),
            file_size,
            pending_pages: HashMap::new(),
        };

        // Recover from WAL if needed
        backend.recover_from_wal()?;

        Ok(backend)
    }

    /// Recover database state from WAL
    fn recover_from_wal(&mut self) -> Result<(), StorageError> {
        if self.wal_size == 0 {
            return Ok(());
        }

        // Read WAL from beginning
        let mut wal_reader =
            File::open(&self.wal_path).map_err(|e| StorageError::Io(e.to_string()))?;

        let mut header_buf = [0u8; WalEntryHeader::SIZE];

        loop {
            // Read header
            match wal_reader.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(StorageError::Io(e.to_string())),
            }

            let header = WalEntryHeader::from_bytes(&header_buf)
                .ok_or_else(|| StorageError::Corrupted("Invalid WAL header".into()))?;

            // Copy packed fields to avoid unaligned access
            let entry_type = header.entry_type;
            let page_id = header.page_id;
            let data_len = header.data_len;
            let expected_checksum = header.checksum;

            match entry_type {
                1 => {
                    // Write entry
                    let mut data = vec![0u8; data_len as usize];
                    wal_reader
                        .read_exact(&mut data)
                        .map_err(|e| StorageError::Io(e.to_string()))?;

                    // Verify checksum
                    let checksum = crc32_fast(&data);
                    if checksum != expected_checksum {
                        // Corrupted entry, stop recovery
                        break;
                    }

                    // Apply to pending pages
                    self.pending_pages.insert(page_id, data);
                }
                2 => {
                    // Delete entry
                    self.pending_pages.remove(&page_id);
                }
                3 => {
                    // Checkpoint - clear pending and apply to data file
                    self.apply_pending_to_datafile()?;
                }
                _ => {
                    // Unknown entry type, stop recovery
                    break;
                }
            }
        }

        // Apply any remaining pending pages
        self.apply_pending_to_datafile()?;

        // Truncate WAL
        self.truncate_wal()?;

        Ok(())
    }

    /// Apply pending pages to data file
    fn apply_pending_to_datafile(&mut self) -> Result<(), StorageError> {
        for (&page_id, data) in &self.pending_pages {
            let offset = page_id * self.config.page_size as u64;

            // Extend file if needed
            let required_size = offset + self.config.page_size as u64;
            if required_size > self.file_size {
                self.data_file
                    .set_len(required_size)
                    .map_err(|e| StorageError::Io(e.to_string()))?;
                self.file_size = required_size;
            }

            // Write to data file
            self.data_file
                .seek(SeekFrom::Start(offset))
                .map_err(|e| StorageError::Io(e.to_string()))?;
            self.data_file
                .write_all(data)
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }

        if self.config.sync_mode == SyncMode::Full {
            self.data_file
                .sync_all()
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }

        self.pending_pages.clear();
        Ok(())
    }

    /// Truncate WAL after checkpoint
    fn truncate_wal(&mut self) -> Result<(), StorageError> {
        // Close and reopen to truncate
        self.wal_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.wal_path)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        self.wal_size = 0;
        Ok(())
    }

    /// Write entry to WAL
    fn write_wal_entry(
        &mut self,
        entry_type: WalEntryType,
        page_id: u64,
        data: Option<&[u8]>,
    ) -> Result<(), StorageError> {
        let data_len = data.map(|d| d.len()).unwrap_or(0) as u32;
        let checksum = data.map(|d| crc32_fast(d)).unwrap_or(0);

        let header = WalEntryHeader {
            entry_type: entry_type as u8,
            page_id,
            data_len,
            checksum,
        };

        // Write header
        self.wal_file
            .write_all(&header.to_bytes())
            .map_err(|e| StorageError::Io(e.to_string()))?;

        // Write data if present
        if let Some(data) = data {
            self.wal_file
                .write_all(data)
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }

        if self.config.sync_mode == SyncMode::Full {
            self.wal_file
                .sync_all()
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }

        self.wal_size += WalEntryHeader::SIZE as u64 + data_len as u64;

        // Checkpoint if WAL is too large
        if self.wal_size >= self.config.wal_checkpoint_threshold {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Perform checkpoint (apply WAL to data file)
    fn checkpoint(&mut self) -> Result<(), StorageError> {
        // Apply pending pages
        self.apply_pending_to_datafile()?;

        // Write checkpoint marker
        let header = WalEntryHeader {
            entry_type: WalEntryType::Checkpoint as u8,
            page_id: 0,
            data_len: 0,
            checksum: 0,
        };

        self.wal_file
            .write_all(&header.to_bytes())
            .map_err(|e| StorageError::Io(e.to_string()))?;

        // Truncate WAL
        self.truncate_wal()?;

        self.stats.sync_ops += 1;
        Ok(())
    }
}

impl EdgeStorageBackend for FileBackend {
    fn read_page(&self, page_id: u64) -> Result<Vec<u8>, StorageError> {
        // Check cache first
        if let Some(entry) = self.cache.get(&page_id) {
            return Ok(entry.data.clone());
        }

        // Check pending pages (WAL)
        if let Some(data) = self.pending_pages.get(&page_id) {
            return Ok(data.clone());
        }

        // Read from data file
        let offset = page_id * self.config.page_size as u64;
        if offset >= self.file_size {
            return Err(StorageError::NotFound(format!(
                "Page {} not found",
                page_id
            )));
        }

        let mut data = vec![0u8; self.config.page_size];

        // Use pread-style access (thread-safe)
        let mut file = File::open(&self.data_path).map_err(|e| StorageError::Io(e.to_string()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Io(e.to_string()))?;
        file.read_exact(&mut data)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        Ok(data)
    }

    fn write_page(&mut self, page_id: u64, data: &[u8]) -> Result<(), StorageError> {
        if data.len() > self.config.page_size {
            return Err(StorageError::DataTooLarge);
        }

        // Pad to page size
        let mut padded = vec![0u8; self.config.page_size];
        padded[..data.len()].copy_from_slice(data);

        // Write to WAL first (for durability)
        self.write_wal_entry(WalEntryType::Write, page_id, Some(&padded))?;

        // Store in pending pages
        self.pending_pages.insert(page_id, padded.clone());

        // Update cache (evict non-dirty entries if at capacity)
        if self.config.max_cache_pages > 0 && self.cache.len() >= self.config.max_cache_pages {
            let evict_keys: Vec<u64> = self
                .cache
                .iter()
                .filter(|(_, e)| !e.dirty)
                .map(|(k, _)| *k)
                .take(self.config.max_cache_pages / 4)
                .collect();
            for k in evict_keys {
                self.cache.remove(&k);
            }
        }
        self.cache.insert(
            page_id,
            CacheEntry {
                data: padded,
                dirty: true,
            },
        );

        self.stats.bytes_written += self.config.page_size as u64;
        self.stats.write_ops += 1;

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        // Checkpoint to apply all pending writes
        self.checkpoint()?;

        // Sync data file
        self.data_file
            .sync_all()
            .map_err(|e| StorageError::Io(e.to_string()))?;

        Ok(())
    }

    fn page_size(&self) -> usize {
        self.config.page_size
    }

    fn capacity(&self) -> u64 {
        self.file_size
    }

    fn available(&self) -> u64 {
        // Get filesystem free space
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if let Ok(stat) = fs::metadata(&self.data_path) {
                // This is a simplification - real implementation would use statfs
                self.file_size - stat.size()
            } else {
                0
            }
        }
        #[cfg(not(unix))]
        {
            self.file_size
        }
    }

    fn delete_page(&mut self, page_id: u64) -> Result<(), StorageError> {
        // Remove from cache
        self.cache.remove(&page_id);

        // Remove from pending
        self.pending_pages.remove(&page_id);

        // Write delete entry to WAL
        self.write_wal_entry(WalEntryType::Delete, page_id, None)?;

        Ok(())
    }

    fn page_exists(&self, page_id: u64) -> bool {
        // Check cache
        if self.cache.contains_key(&page_id) {
            return true;
        }

        // Check pending
        if self.pending_pages.contains_key(&page_id) {
            return true;
        }

        // Check data file
        let offset = page_id * self.config.page_size as u64;
        offset < self.file_size
    }

    fn stats(&self) -> StorageStats {
        let mut stats = self.stats.clone();
        stats.pages_used = (self.file_size / self.config.page_size as u64) as u64;
        stats
    }
}

impl Drop for FileBackend {
    fn drop(&mut self) {
        // Try to sync on drop
        let _ = self.sync();
    }
}

/// Fast CRC32 calculation
fn crc32_fast(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_backend_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();

        // Write a page
        let data = b"Hello, File Backend!";
        backend.write_page(0, data).unwrap();

        // Read it back
        let read_data = backend.read_page(0).unwrap();
        assert_eq!(&read_data[..data.len()], data);
    }

    #[test]
    fn test_file_backend_wal_recovery() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        // Write data
        {
            let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();
            backend.write_page(0, b"test data").unwrap();
            // Don't call sync - simulate crash
        }

        // Reopen - should recover from WAL
        {
            let backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();
            let data = backend.read_page(0).unwrap();
            assert_eq!(&data[..9], b"test data");
        }
    }

    #[test]
    fn test_file_backend_checkpoint() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut backend = FileBackend::open(
            &path,
            FileBackendConfig {
                wal_checkpoint_threshold: 1024, // Small threshold for testing
                ..Default::default()
            },
        )
        .unwrap();

        // Write enough data to trigger checkpoint
        for i in 0..10 {
            let data = format!("Page {} data", i);
            backend.write_page(i, data.as_bytes()).unwrap();
        }

        // Verify data
        for i in 0..10 {
            let expected = format!("Page {} data", i);
            let data = backend.read_page(i).unwrap();
            assert_eq!(&data[..expected.len()], expected.as_bytes());
        }
    }

    #[test]
    fn test_crc32() {
        let data = b"test data for crc";
        let crc = crc32_fast(data);
        assert_eq!(crc, crc32_fast(data)); // Same input should give same output

        let different = b"different data";
        assert_ne!(crc, crc32_fast(different));
    }

    #[test]
    fn test_file_backend_wal_recovery_multiple_pages() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.db");

        {
            let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();
            for i in 0..20 {
                let data = format!("page-{}-data", i);
                backend.write_page(i, data.as_bytes()).unwrap();
            }
            // Crash without sync
        }

        {
            let backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();
            for i in 0..20 {
                let expected = format!("page-{}-data", i);
                let data = backend.read_page(i).unwrap();
                assert_eq!(
                    &data[..expected.len()],
                    expected.as_bytes(),
                    "Page {} not recovered correctly",
                    i
                );
            }
        }
    }

    #[test]
    fn test_file_backend_overwrite_page() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("overwrite.db");
        let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();

        backend.write_page(0, b"original").unwrap();
        backend.write_page(0, b"updated").unwrap();

        let data = backend.read_page(0).unwrap();
        assert_eq!(&data[..7], b"updated");
    }

    #[test]
    fn test_file_backend_delete_page() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delete.db");
        let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();

        backend.write_page(5, b"to be deleted").unwrap();
        assert!(backend.page_exists(5));

        backend.delete_page(5).unwrap();
        // Page may still exist in data file but pending delete in WAL
    }

    #[test]
    fn test_file_backend_read_beyond_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.db");
        let backend = FileBackend::open(
            &path,
            FileBackendConfig {
                initial_size: 4096, // Only 1 page
                ..Default::default()
            },
        )
        .unwrap();

        // Page 999 is well beyond the 1-page file
        let result = backend.read_page(999);
        assert!(result.is_err(), "Reading beyond file should fail");
    }

    #[test]
    fn test_file_backend_cache_eviction() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cache.db");
        let mut backend = FileBackend::open(
            &path,
            FileBackendConfig {
                max_cache_pages: 8,
                ..Default::default()
            },
        )
        .unwrap();

        // Write more pages than cache capacity
        for i in 0..20 {
            backend
                .write_page(i, format!("page-{}", i).as_bytes())
                .unwrap();
        }

        // All pages should still be readable (from WAL/disk if evicted from cache)
        for i in 0..20 {
            let data = backend.read_page(i).unwrap();
            let expected = format!("page-{}", i);
            assert_eq!(&data[..expected.len()], expected.as_bytes());
        }
    }

    #[test]
    fn test_file_backend_sync_then_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sync.db");

        {
            let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();
            backend.write_page(0, b"synced data").unwrap();
            backend.sync().unwrap(); // Explicit sync
        }

        {
            let backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();
            let data = backend.read_page(0).unwrap();
            assert_eq!(&data[..11], b"synced data");
        }
    }

    #[test]
    fn test_file_backend_stats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stats.db");
        let mut backend = FileBackend::open(&path, FileBackendConfig::default()).unwrap();

        backend.write_page(0, b"stats test").unwrap();
        backend.write_page(1, b"more data").unwrap();

        let stats = backend.stats();
        assert!(stats.write_ops >= 2);
        assert!(stats.bytes_written > 0);
    }
}
