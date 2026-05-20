//! File-based storage backend with WAL support
//!
//! Implements persistent storage using the local filesystem with
//! write-ahead logging for crash recovery and durability.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

use joule_db_core::error::StorageError;
use joule_db_core::storage::page::DEFAULT_PAGE_SIZE;
use joule_db_core::storage::{CommittedMeta, Page, PageId, StorageBackend, StorageStats};

use super::wal::{RecoveryManager, WalManager};

const DATA_FILE: &str = "data.wdb";
const META_FILE: &str = "meta.wdb";

/// Magic prefix for the v1 `meta.wdb` format. Bytes "JDBM" interpreted
/// as little-endian — distinguishes v1 records from the legacy v0
/// layout (`[u64 next_page_id][u64 free_pages...]` with no prefix).
///
/// CoW MVCC Phase 2: the v1 format adds an atomically-committed root
/// pointer + version counter so the writer's commit appears atomic to
/// snapshot-holding readers. See `docs/joule-db/cow-mvcc-design.md`.
const META_V1_MAGIC: [u8; 4] = *b"JDBM";

/// Current `meta.wdb` format version. Bumped whenever the layout
/// changes; readers fall back to legacy parsing for older versions.
const META_V1_FORMAT_VERSION: u32 = 1;

/// Header size for a v1 record (everything before the variable-length
/// `free_pages` array): magic(4) + format_version(4) +
/// committed_version(8) + committed_root(8) + next_page_id(8) +
/// free_pages_count(4) = 36 bytes.
const META_V1_HEADER_SIZE: usize = 36;

/// Size of the trailing CRC32 checksum.
const META_V1_CRC_SIZE: usize = 4;

/// IEEE 802.3 CRC-32 over `data`. Inlined here to avoid a workspace
/// dependency on `crc32fast` for a single 30-line use site. Same
/// polynomial as `joule_db_core::storage::page::crc32`, kept private.
fn crc32_meta(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// File-based storage backend with WAL support
pub struct FileBackend {
    /// Database directory path
    path: PathBuf,
    /// Data file handle
    data_file: RwLock<File>,
    /// Write-ahead log manager
    wal: Mutex<WalManager>,
    /// Page cache
    cache: RwLock<HashMap<PageId, Page>>,
    /// Next page ID
    next_page_id: AtomicU64,
    /// Free page list
    free_pages: RwLock<Vec<PageId>>,
    /// Page size
    page_size: usize,
    /// Statistics
    stats: RwLock<FileStats>,
    /// Current transaction ID (0 = no active transaction)
    current_tx_id: AtomicU64,
    /// Next transaction ID
    next_tx_id: AtomicU64,
    /// Dirty pages in current transaction
    dirty_pages: RwLock<HashMap<PageId, Page>>,
    /// Monotonic counter used to build unique `meta.wdb.tmp.<N>` names.
    /// Together with a per-call mutex, this eliminates the "Failed to
    /// rename metadata: No such file or directory" race observed on
    /// 2026-04-22 when scholar-ingestd's shutdown-sync overlapped with
    /// an in-flight checkpoint: two calls wrote to the same fixed
    /// `meta.wdb.tmp` path, and the second rename found its tmp gone.
    save_metadata_seq: AtomicU64,
    /// Serialises writes to `meta.wdb`. Keeps tmp → rename atomic with
    /// respect to other metadata savers.
    save_metadata_lock: Mutex<()>,
    /// **CoW MVCC Phase 2.** Most-recently-committed `(committed_root,
    /// committed_version)` mirroring the `meta.wdb` v1 record on disk.
    /// `None` until the first `write_committed_meta` call (a freshly
    /// migrated v0 database has no committed_meta yet — Engine fills
    /// this in on its first commit).
    ///
    /// Writes go through `write_committed_meta`, which serialises the
    /// full v1 record (committed_meta + allocator state) atomically
    /// via tmp + fsync + rename. See `docs/joule-db/cow-mvcc-design.md`.
    committed_meta: RwLock<Option<CommittedMeta>>,
}

#[derive(Debug, Default)]
struct FileStats {
    pages_read: u64,
    pages_written: u64,
    wal_writes: u64,
    checkpoints: u64,
}

impl FileBackend {
    /// Open or create a database at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        fs::create_dir_all(&path)
            .map_err(|e| StorageError::Backend(format!("Failed to create directory: {}", e)))?;

        // Open WAL first (for recovery)
        let wal = WalManager::open(&path)?;

        // Open or create data file
        let data_path = path.join(DATA_FILE);
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .map_err(|e| StorageError::Backend(format!("Failed to open data file: {}", e)))?;

        // Load metadata — v1 format (with committed_meta) is preferred;
        // v0 (legacy: just next_page_id + free_pages) is accepted and
        // upgraded on the next save.
        let meta_path = path.join(META_FILE);
        let (committed_meta, next_page_id, free_pages) = if meta_path.exists() {
            Self::load_metadata(&meta_path)?
        } else {
            (None, 1, Vec::new())
        };

        let mut backend = Self {
            path,
            data_file: RwLock::new(data_file),
            wal: Mutex::new(wal),
            cache: RwLock::new(HashMap::new()),
            next_page_id: AtomicU64::new(next_page_id),
            free_pages: RwLock::new(free_pages),
            page_size: DEFAULT_PAGE_SIZE,
            stats: RwLock::new(FileStats::default()),
            current_tx_id: AtomicU64::new(0),
            next_tx_id: AtomicU64::new(1),
            dirty_pages: RwLock::new(HashMap::new()),
            save_metadata_seq: AtomicU64::new(0),
            save_metadata_lock: Mutex::new(()),
            committed_meta: RwLock::new(committed_meta),
        };

        // Run crash recovery
        backend.recover()?;

        Ok(backend)
    }

    /// Run crash recovery from WAL
    fn recover(&mut self) -> Result<(), StorageError> {
        let wal = self.wal.lock();
        let result = RecoveryManager::recover(&wal)?;
        drop(wal);

        if result.pages_to_apply.is_empty() && result.uncommitted_transactions.is_empty() {
            return Ok(());
        }

        log::info!(
            "WAL recovery: applying {} pages, {} committed txs, {} uncommitted txs",
            result.pages_to_apply.len(),
            result.committed_transactions.len(),
            result.uncommitted_transactions.len()
        );

        // Apply committed pages directly to data file
        for (page_id, data) in result.pages_to_apply {
            self.write_page_direct(page_id, &data)?;
        }

        // Sync data file
        {
            let file = self.data_file.read();
            file.sync_all()
                .map_err(|e| StorageError::Backend(format!("Sync error: {}", e)))?;
        }

        // Log rollback for uncommitted transactions
        {
            let mut wal = self.wal.lock();
            for tx_id in result.uncommitted_transactions {
                wal.log_rollback(tx_id)?;
            }
        }

        // Checkpoint after recovery
        self.checkpoint()?;

        Ok(())
    }

    /// Write page directly to data file (bypassing WAL, for recovery)
    fn write_page_direct(&mut self, page_id: PageId, data: &[u8]) -> Result<(), StorageError> {
        let mut file = self.data_file.write();
        let offset = self.page_offset(page_id);

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Backend(format!("Seek error: {}", e)))?;

        file.write_all(data)
            .map_err(|e| StorageError::Backend(format!("Write error: {}", e)))?;

        // Invalidate cache
        let mut cache = self.cache.write();
        cache.remove(&page_id);

        Ok(())
    }

    /// Load metadata from file.
    ///
    /// Recognises both the v1 layout
    /// (`magic|format_version|committed_version|committed_root|next_page_id|free_pages_count|free_pages...|crc32`)
    /// and the legacy v0 layout (`next_page_id|free_pages...`, no
    /// magic prefix). v0 returns `committed_meta = None`; the caller's
    /// next `save_metadata` call upgrades the file to v1.
    fn load_metadata(
        path: &Path,
    ) -> Result<(Option<CommittedMeta>, PageId, Vec<PageId>), StorageError> {
        let mut file = File::open(path)
            .map_err(|e| StorageError::Backend(format!("Failed to open metadata: {}", e)))?;

        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| StorageError::Backend(format!("Failed to read metadata: {}", e)))?;

        if buf.len() < 8 {
            return Ok((None, 1, Vec::new()));
        }

        // v1 detection: the first 4 bytes must match META_V1_MAGIC AND
        // the file must be at least header + crc bytes long.
        if buf.len() >= META_V1_HEADER_SIZE + META_V1_CRC_SIZE
            && buf[0..4] == META_V1_MAGIC
        {
            return Self::load_metadata_v1(&buf);
        }

        // v0 fallback: legacy layout. Caller will upgrade on next save.
        let next_page_id = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);

        let mut free_pages = Vec::new();
        let mut offset = 8;
        while offset + 8 <= buf.len() {
            let page_id = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            free_pages.push(page_id);
            offset += 8;
        }

        Ok((None, next_page_id, free_pages))
    }

    /// Parse a v1 meta record. Verifies CRC32 over all bytes preceding
    /// the trailing 4-byte checksum; rejects the record on mismatch
    /// rather than silently regressing to defaults.
    fn load_metadata_v1(
        buf: &[u8],
    ) -> Result<(Option<CommittedMeta>, PageId, Vec<PageId>), StorageError> {
        // Validate CRC32 over `buf[..len-4]`.
        let crc_offset = buf.len() - META_V1_CRC_SIZE;
        let stored_crc = u32::from_le_bytes([
            buf[crc_offset],
            buf[crc_offset + 1],
            buf[crc_offset + 2],
            buf[crc_offset + 3],
        ]);
        let actual_crc = crc32_meta(&buf[..crc_offset]);
        if stored_crc != actual_crc {
            return Err(StorageError::Backend(format!(
                "meta.wdb v1 CRC mismatch: stored {:08x} computed {:08x}",
                stored_crc, actual_crc
            )));
        }

        // Skip magic (4) + format_version (4) — both already checked
        // by the caller. Read the four u64 fields and the count.
        let format_version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        if format_version != META_V1_FORMAT_VERSION {
            return Err(StorageError::Backend(format!(
                "meta.wdb v1 unsupported format_version {}, expected {}",
                format_version, META_V1_FORMAT_VERSION
            )));
        }

        let committed_version = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let committed_root = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let next_page_id = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let free_pages_count = u32::from_le_bytes(buf[32..36].try_into().unwrap()) as usize;

        // Bounds-check: header + count*8 + crc must fit.
        let expected_len = META_V1_HEADER_SIZE + free_pages_count * 8 + META_V1_CRC_SIZE;
        if buf.len() != expected_len {
            return Err(StorageError::Backend(format!(
                "meta.wdb v1 length mismatch: got {} expected {} (free_pages_count={})",
                buf.len(),
                expected_len,
                free_pages_count
            )));
        }

        let mut free_pages = Vec::with_capacity(free_pages_count);
        let base = META_V1_HEADER_SIZE;
        for i in 0..free_pages_count {
            let off = base + i * 8;
            let page_id = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            free_pages.push(page_id);
        }

        Ok((
            Some(CommittedMeta {
                format_version,
                committed_version,
                committed_root,
            }),
            next_page_id,
            free_pages,
        ))
    }

    /// Save metadata to file atomically (write to temp, fsync, rename).
    ///
    /// Two protections against the 2026-04-22 rename race:
    ///
    /// 1. **Unique tmp filename** — `meta.wdb.tmp.<pid>.<seq>` means two
    ///    concurrent callers never share a tmp path, even if the
    ///    serialising lock below somehow fails.
    /// 2. **Serialising mutex** — only one metadata save proceeds at a
    ///    time, so `std::fs::rename` can't race another caller's
    ///    create / sync / rename sequence on the same `meta.wdb`.
    ///
    /// Best-effort: on success the tmp file is gone (rename consumed
    /// it); on any earlier error we attempt to clean the tmp so future
    /// saves don't see stale `*.tmp.<pid>.<N>` debris.
    fn save_metadata(&self) -> Result<(), StorageError> {
        // parking_lot Mutex never poisons — direct lock acquisition.
        let _guard = self.save_metadata_lock.lock();

        let meta_path = self.path.join(META_FILE);
        let seq = self.save_metadata_seq.fetch_add(1, Ordering::SeqCst);
        let tmp_name = format!(
            "{}.tmp.{pid}.{seq}",
            META_FILE,
            pid = std::process::id(),
            seq = seq
        );
        let tmp_path = self.path.join(&tmp_name);

        // Wrap all work through `finish_or_cleanup` so an early error
        // still removes the partial tmp. We intentionally ignore
        // `remove_file` failures — the file might never have been
        // created, which is fine.
        let finish_or_cleanup = |res: Result<(), StorageError>| -> Result<(), StorageError> {
            if res.is_err() {
                let _ = std::fs::remove_file(&tmp_path);
            }
            res
        };

        let write_result = (|| -> Result<(), StorageError> {
            let mut file = File::create(&tmp_path).map_err(|e| {
                StorageError::Backend(format!(
                    "Failed to create metadata tmp {}: {}",
                    tmp_path.display(),
                    e
                ))
            })?;

            // Build the full v1 record in memory so we can checksum it
            // before any I/O hits the tmp file. Layout matches
            // `load_metadata_v1`.
            let next_page_id = self.next_page_id.load(Ordering::SeqCst);
            let free_pages = self.free_pages.read();
            let committed = self.committed_meta.read();

            // If write_committed_meta hasn't been called yet (v0
            // database that hasn't done its first commit), persist a
            // sentinel CommittedMeta with version 0 / root 0. Engine's
            // open path treats `committed_root == 0` as "fall back to
            // legacy page-1 root pointer" so this is safe.
            let cm = committed.unwrap_or(CommittedMeta {
                format_version: META_V1_FORMAT_VERSION,
                committed_version: 0,
                committed_root: 0,
            });

            let mut buf = Vec::with_capacity(
                META_V1_HEADER_SIZE + free_pages.len() * 8 + META_V1_CRC_SIZE,
            );
            buf.extend_from_slice(&META_V1_MAGIC);
            buf.extend_from_slice(&cm.format_version.to_le_bytes());
            buf.extend_from_slice(&cm.committed_version.to_le_bytes());
            buf.extend_from_slice(&cm.committed_root.to_le_bytes());
            buf.extend_from_slice(&next_page_id.to_le_bytes());
            buf.extend_from_slice(&(free_pages.len() as u32).to_le_bytes());
            for &page_id in free_pages.iter() {
                buf.extend_from_slice(&page_id.to_le_bytes());
            }

            // Trailing CRC32 over everything written so far.
            let crc = crc32_meta(&buf);
            buf.extend_from_slice(&crc.to_le_bytes());

            file.write_all(&buf)
                .map_err(|e| StorageError::Backend(format!("Failed to write metadata: {}", e)))?;

            file.sync_all()
                .map_err(|e| StorageError::Backend(format!("Failed to sync metadata: {}", e)))?;
            Ok(())
        })();
        finish_or_cleanup(write_result)?;

        std::fs::rename(&tmp_path, &meta_path).map_err(|e| {
            // Best-effort cleanup: the rename failed, tmp may still be present.
            let _ = std::fs::remove_file(&tmp_path);
            StorageError::Backend(format!(
                "Failed to rename metadata {} -> {}: {}",
                tmp_path.display(),
                meta_path.display(),
                e
            ))
        })?;

        Ok(())
    }

    /// Get file offset for a page
    fn page_offset(&self, page_id: PageId) -> u64 {
        (page_id - 1) * self.page_size as u64
    }

    /// Begin a new transaction.
    ///
    /// Returns an error if a transaction is already active — callers must
    /// commit or rollback before starting a new one.
    pub fn begin_transaction(&self) -> Result<u64, StorageError> {
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::SeqCst);
        match self
            .current_tx_id
            .compare_exchange(0, tx_id, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Ok(tx_id),
            Err(active) => Err(StorageError::Backend(format!(
                "Transaction {} already active — commit or rollback before starting a new one",
                active
            ))),
        }
    }

    /// Commit current transaction
    pub fn commit_transaction(&mut self) -> Result<(), StorageError> {
        let tx_id = self.current_tx_id.load(Ordering::SeqCst);
        if tx_id == 0 {
            return Err(StorageError::Backend("No active transaction".to_string()));
        }

        // Get dirty pages
        let dirty_pages: Vec<(PageId, Page)> = {
            let mut dirty = self.dirty_pages.write();
            dirty.drain().collect()
        };

        // Write dirty pages to data file
        for (page_id, page) in &dirty_pages {
            let encoded = page.encode(self.page_size)?;

            let mut file = self.data_file.write();
            let offset = self.page_offset(*page_id);

            file.seek(SeekFrom::Start(offset))
                .map_err(|e| StorageError::Backend(format!("Seek error: {}", e)))?;

            file.write_all(&encoded)
                .map_err(|e| StorageError::Backend(format!("Write error: {}", e)))?;
        }

        // Log commit to WAL
        {
            let mut wal = self.wal.lock();
            wal.log_commit(tx_id)?;
        }

        // Update cache with committed pages
        {
            let mut cache = self.cache.write();
            for (page_id, page) in dirty_pages {
                cache.insert(page_id, page);
            }
        }

        // Clear current transaction
        self.current_tx_id.store(0, Ordering::SeqCst);

        // **No per-commit auto-checkpoint.** Previously this fired
        // `self.checkpoint()` whenever WAL exceeded `MAX_WAL_SIZE`,
        // which caused an F_FULLFSYNC storm during buffer-pool batch
        // flushes (e.g. 50K-page `flush_all` calls 50K auto-commits;
        // once WAL crossed 64 MB at the ~1024th page, every
        // subsequent commit fired a `data_file.sync_all` +
        // `save_metadata` pair = 2 F_FULLFSYNC syscalls each).
        //
        // The truncate logic now lives in `sync()` below, which the
        // engine calls at every commit boundary. WAL grows
        // unbounded between sync points, which is the price of
        // batched durability — for a single-writer workload that
        // calls sync per batch, this is fine.

        Ok(())
    }

    /// Rollback current transaction
    pub fn rollback_transaction(&mut self) -> Result<(), StorageError> {
        let tx_id = self.current_tx_id.load(Ordering::SeqCst);
        if tx_id == 0 {
            return Err(StorageError::Backend("No active transaction".to_string()));
        }

        // Clear dirty pages (discard changes)
        {
            let mut dirty = self.dirty_pages.write();
            dirty.clear();
        }

        // Log rollback to WAL
        {
            let mut wal = self.wal.lock();
            wal.log_rollback(tx_id)?;
        }

        // Clear current transaction
        self.current_tx_id.store(0, Ordering::SeqCst);

        Ok(())
    }

    /// Create a checkpoint
    pub fn checkpoint(&mut self) -> Result<(), StorageError> {
        // Sync data file first
        {
            let file = self.data_file.read();
            file.sync_all()
                .map_err(|e| StorageError::Backend(format!("Sync error: {}", e)))?;
        }

        // Save metadata
        self.save_metadata()?;

        // Log checkpoint and truncate WAL
        {
            let mut wal = self.wal.lock();
            wal.log_checkpoint()?;
            wal.truncate()?;
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.checkpoints += 1;
        }

        Ok(())
    }

    /// Get WAL statistics
    pub fn wal_stats(&self) -> (u64, u64) {
        let wal = self.wal.lock();
        (wal.current_lsn(), wal.last_checkpoint_lsn())
    }
}

impl StorageBackend for FileBackend {
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        // Check dirty pages first (uncommitted writes)
        {
            let dirty = self.dirty_pages.read();
            if let Some(page) = dirty.get(&page_id) {
                return Ok(Some(page.clone()));
            }
        }

        // Check cache
        {
            let cache = self.cache.read();
            if let Some(page) = cache.get(&page_id) {
                return Ok(Some(page.clone()));
            }
        }

        // Read from file
        let mut file = self.data_file.write();
        let offset = self.page_offset(page_id);

        // Check if page exists
        let file_len = file
            .seek(SeekFrom::End(0))
            .map_err(|e| StorageError::Backend(format!("Seek error: {}", e)))?;

        if offset >= file_len {
            return Ok(None);
        }

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Backend(format!("Seek error: {}", e)))?;

        let mut buf = vec![0u8; self.page_size];
        let bytes_read = file
            .read(&mut buf)
            .map_err(|e| StorageError::Backend(format!("Read error: {}", e)))?;

        if bytes_read == 0 {
            return Ok(None);
        }

        // Decode page
        let page = Page::decode(&buf).map_err(|e| StorageError::Corrupted {
            page_id,
            reason: e.to_string(),
        })?;

        // Cache it
        {
            let mut cache = self.cache.write();
            cache.insert(page_id, page.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.pages_read += 1;
        }

        Ok(Some(page))
    }

    fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
        let page_id = page.id;
        let encoded = page.encode(self.page_size)?;

        // Get or create transaction
        let (tx_id, auto_commit) = {
            let current = self.current_tx_id.load(Ordering::SeqCst);
            if current == 0 {
                // Auto-transaction for single operations
                (self.begin_transaction()?, true)
            } else {
                (current, false)
            }
        };

        // Write to WAL first
        {
            let mut wal = self.wal.lock();
            wal.log_page_write(tx_id, page_id, &encoded)?;
        }

        // Add to dirty pages
        {
            let mut dirty = self.dirty_pages.write();
            dirty.insert(page_id, page);
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.pages_written += 1;
            stats.wal_writes += 1;
        }

        // Auto-commit if this was an implicit transaction
        if auto_commit {
            self.commit_transaction()?;
        }

        Ok(())
    }

    fn allocate_page(&mut self) -> Result<PageId, StorageError> {
        // Try to reuse a free page
        {
            let mut free_pages = self.free_pages.write();
            if let Some(page_id) = free_pages.pop() {
                return Ok(page_id);
            }
        }

        // Allocate new page
        let page_id = self.next_page_id.fetch_add(1, Ordering::SeqCst);
        Ok(page_id)
    }

    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
        // Remove from cache
        {
            let mut cache = self.cache.write();
            cache.remove(&page_id);
        }

        // Remove from dirty pages
        {
            let mut dirty = self.dirty_pages.write();
            dirty.remove(&page_id);
        }

        // Add to free list
        {
            let mut free_pages = self.free_pages.write();
            free_pages.push(page_id);
        }

        Ok(())
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        // Commit any pending transaction
        let tx_id = self.current_tx_id.load(Ordering::SeqCst);
        if tx_id != 0 {
            self.commit_transaction()?;
        }

        // Sync WAL
        {
            let mut wal = self.wal.lock();
            wal.sync()?;
        }

        // Sync data file
        {
            let file = self.data_file.read();
            file.sync_all()
                .map_err(|e| StorageError::Backend(format!("Sync error: {}", e)))?;
        }

        // Save metadata
        self.save_metadata()?;

        // **Deferred WAL truncate.** Previously the per-commit
        // auto-checkpoint did this; that triggered F_FULLFSYNC
        // storms in batch flushes (see `commit_transaction`
        // above). Now we truncate WAL at sync time only — at most
        // once per engine commit boundary, regardless of how many
        // pages were written in between.
        let needs_truncate = {
            let wal = self.wal.lock();
            wal.needs_checkpoint()
        };
        if needs_truncate {
            // Data file + meta are already synced above (the WAL
            // truncate must follow durable data, otherwise crash
            // recovery would lose the truncated entries).
            let mut wal = self.wal.lock();
            wal.log_checkpoint()?;
            wal.truncate()?;
        }

        Ok(())
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn stats(&self) -> StorageStats {
        let _cache = self.cache.read();
        let free_pages = self.free_pages.read();
        let stats = self.stats.read();

        StorageStats {
            total_pages: self.next_page_id.load(Ordering::SeqCst) - 1,
            free_pages: free_pages.len() as u64,
            pages_read: stats.pages_read,
            pages_written: stats.pages_written,
            page_size: self.page_size,
        }
    }

    fn read_committed_meta(&self) -> Result<Option<CommittedMeta>, StorageError> {
        // **Production bug fix (2026-04-28).** Originally this
        // returned the in-memory mirror, which was populated only
        // at `Database::open` and at our own `write_committed_meta`
        // calls. That worked for the single-process happy path but
        // BROKE the whole point of `Engine::refresh_from_backend`:
        // a peer process's commits never updated our in-memory
        // mirror, so scholar-server's refresher tick read the
        // mirror, saw no change, and silently never advanced.
        //
        // Re-read meta.wdb from disk on every call. The file is
        // ~40 bytes for an empty free list and a handful of KB even
        // with thousands of free entries — cheap. If the on-disk
        // record fails to load (concurrent rename, partial write,
        // missing file), we fall back to the in-memory mirror so
        // the engine doesn't lose its existing view.
        let meta_path = self.path.join(META_FILE);
        if meta_path.exists() {
            if let Ok((on_disk, _next, _free)) = Self::load_metadata(&meta_path) {
                // Sync the mirror so subsequent saves pick up from
                // the latest committed version (otherwise the
                // monotonic-version sanity rail in
                // `write_committed_meta` could refuse a legitimate
                // peer-advanced commit).
                if let Some(m) = on_disk {
                    *self.committed_meta.write() = Some(m);
                }
                return Ok(on_disk
                    .filter(|m| m.committed_root != 0 || m.committed_version != 0));
            }
        }
        // Fallback: in-memory mirror (filtered as before).
        Ok(self
            .committed_meta
            .read()
            .filter(|m| m.committed_root != 0 || m.committed_version != 0))
    }

    fn any_external_snapshots_live(&self) -> bool {
        // CoW MVCC Phase 4: scan <db>/snapshots/ for peer-process
        // lockfiles. Stale lockfiles (process died holding one) are
        // GC'd inline by the registry scan.
        crate::snapshot_registry::any_external_live(&self.path)
    }

    fn write_committed_meta(&mut self, meta: &CommittedMeta) -> Result<(), StorageError> {
        // Refuse to retire to an older committed_version. This is a
        // sanity rail against an Engine bug; legitimate commits always
        // monotonically increase. Note: a v0→v1 migration writes
        // `committed_version = 1` against a previously-stored sentinel
        // 0, which is fine.
        {
            let prev = self.committed_meta.read();
            if let Some(prev) = *prev {
                if meta.committed_version < prev.committed_version {
                    return Err(StorageError::Backend(format!(
                        "write_committed_meta refused: new version {} < previous {}",
                        meta.committed_version, prev.committed_version
                    )));
                }
            }
        }

        // CoW MVCC commit invariant: the new root's pages must already
        // be durable on disk before the meta record is renamed into
        // place. Otherwise a crash between data write and meta swap
        // could expose readers to a root pointing at half-written
        // pages. Sync the data file first.
        {
            let file = self.data_file.read();
            file.sync_all()
                .map_err(|e| StorageError::Backend(format!("Sync error: {}", e)))?;
        }

        // Stage the in-memory mirror, then atomically write the
        // combined v1 record (committed_meta + allocator state) via
        // the existing tmp + fsync + rename machinery.
        *self.committed_meta.write() = Some(*meta);

        self.save_metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_core::storage::PageType;
    use tempfile::TempDir;

    #[test]
    fn test_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut backend = FileBackend::open(temp_dir.path()).unwrap();

        // Allocate page
        let page_id = backend.allocate_page().unwrap();

        // Write page
        let page = Page::with_data(page_id, PageType::BTreeLeaf, b"test data".to_vec());
        backend.write_page(page).unwrap();

        // Read page
        let read = backend.read_page(page_id).unwrap().unwrap();
        assert_eq!(read.data, b"test data");

        // Sync
        backend.sync().unwrap();
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Create and write
        {
            let mut backend = FileBackend::open(&path).unwrap();
            let page_id = backend.allocate_page().unwrap();
            let page = Page::with_data(page_id, PageType::BTreeLeaf, b"persistent".to_vec());
            backend.write_page(page).unwrap();
            backend.sync().unwrap();
        }

        // Reopen and read
        {
            let backend = FileBackend::open(&path).unwrap();
            let read = backend.read_page(1).unwrap().unwrap();
            assert_eq!(read.data, b"persistent");
        }
    }

    /// Regression for the 2026-04-22 "Failed to rename metadata: No such
    /// file or directory" race. Pre-fix, two concurrent `save_metadata`
    /// calls both wrote to `meta.wdb.tmp`; the first `rename` consumed
    /// the file and the second bailed with ENOENT. Post-fix: each call
    /// uses a unique `meta.wdb.tmp.<pid>.<seq>` filename behind a
    /// serialising mutex, so N concurrent saves never collide.
    #[test]
    fn concurrent_save_metadata_does_not_race() {
        use std::sync::Arc;
        use std::thread;

        let temp_dir = TempDir::new().unwrap();
        let backend = Arc::new(FileBackend::open(temp_dir.path()).unwrap());

        // Pre-populate some free pages so save_metadata has work to do.
        {
            let mut free = backend.free_pages.write();
            for i in 0..64 {
                free.push(i + 1);
            }
        }

        let mut handles = Vec::new();
        for _ in 0..16 {
            let b = Arc::clone(&backend);
            handles.push(thread::spawn(move || -> Result<(), StorageError> {
                for _ in 0..25 {
                    b.save_metadata()?;
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().unwrap().expect("no rename race");
        }

        // meta.wdb should exist and be readable.
        let meta = temp_dir.path().join(META_FILE);
        assert!(meta.exists(), "meta.wdb survived");
        let (committed_meta, next_page_id, free_pages) =
            FileBackend::load_metadata(&meta).unwrap();
        assert_eq!(next_page_id, 1); // nothing allocated yet
        assert_eq!(free_pages.len(), 64);
        // No commits ran in this test, so the v1 record stores a
        // sentinel CommittedMeta with version 0 / root 0.
        assert_eq!(
            committed_meta.expect("v1 meta record present").committed_version,
            0
        );

        // No `meta.wdb.tmp.*` debris should remain after clean saves.
        let mut debris = 0;
        for e in std::fs::read_dir(temp_dir.path()).unwrap() {
            let name = e.unwrap().file_name().to_string_lossy().into_owned();
            if name.starts_with("meta.wdb.tmp") {
                debris += 1;
            }
        }
        assert_eq!(debris, 0, "tmp files left behind after successful saves");
    }

    #[test]
    fn test_wal_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write with WAL
        {
            let mut backend = FileBackend::open(&path).unwrap();

            // Transaction 1
            let _tx1 = backend.begin_transaction().unwrap();
            let page1 = Page::with_data(1, PageType::BTreeLeaf, b"tx1 data".to_vec());
            backend.write_page(page1).unwrap();
            backend.commit_transaction().unwrap();

            backend.sync().unwrap();
        }

        // Reopen (triggers recovery)
        {
            let backend = FileBackend::open(&path).unwrap();
            let read = backend.read_page(1).unwrap().unwrap();
            assert_eq!(read.data, b"tx1 data");
        }
    }

    #[test]
    fn test_transaction_rollback() {
        let temp_dir = TempDir::new().unwrap();
        let mut backend = FileBackend::open(temp_dir.path()).unwrap();

        // Write committed data
        let page1 = Page::with_data(1, PageType::BTreeLeaf, b"committed".to_vec());
        backend.write_page(page1).unwrap();

        // Start transaction with uncommitted write
        let _tx = backend.begin_transaction().unwrap();

        // Manually add to dirty pages (simulating write without auto-commit)
        {
            let mut dirty = backend.dirty_pages.write();
            let page2 = Page::with_data(1, PageType::BTreeLeaf, b"uncommitted".to_vec());
            dirty.insert(1, page2);
        }

        // Rollback
        backend.rollback_transaction().unwrap();

        // Read should return committed data (from cache/file, not dirty)
        // Note: In real scenario, we'd need to re-read from file
        backend.sync().unwrap();
    }

    #[test]
    fn test_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let mut backend = FileBackend::open(temp_dir.path()).unwrap();

        // Write some data
        for i in 1..=5 {
            let page = Page::with_data(i, PageType::BTreeLeaf, format!("page {}", i).into_bytes());
            backend.write_page(page).unwrap();
        }

        // Checkpoint
        backend.checkpoint().unwrap();

        // WAL should be truncated
        let (_current_lsn, checkpoint_lsn) = backend.wal_stats();
        assert!(checkpoint_lsn > 0);
    }

    #[test]
    fn test_multiple_transactions() {
        let temp_dir = TempDir::new().unwrap();
        let mut backend = FileBackend::open(temp_dir.path()).unwrap();

        // Transaction 1
        let page1 = Page::with_data(1, PageType::BTreeLeaf, b"first".to_vec());
        backend.write_page(page1).unwrap();

        // Transaction 2
        let page2 = Page::with_data(2, PageType::BTreeLeaf, b"second".to_vec());
        backend.write_page(page2).unwrap();

        // Both should be readable
        let read1 = backend.read_page(1).unwrap().unwrap();
        let read2 = backend.read_page(2).unwrap().unwrap();

        assert_eq!(read1.data, b"first");
        assert_eq!(read2.data, b"second");
    }
}
