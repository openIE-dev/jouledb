//! Write-Ahead Log (WAL) for crash recovery and durability
//!
//! ## WAL Format
//!
//! The WAL is an append-only log file with the following entry format:
//!
//! ```text
//! +----------+----------+----------+----------+---------------+----------+
//! | Magic(4) | CRC32(4) | LSN(8)   | Type(1)  | Length(4)     | Data(n)  |
//! +----------+----------+----------+----------+---------------+----------+
//! ```
//!
//! Entry types:
//! - 0x01: Page write
//! - 0x02: Commit
//! - 0x03: Checkpoint
//! - 0x04: Rollback
//!
//! ## Recovery Process
//!
//! 1. Find last valid checkpoint
//! 2. Replay all entries after checkpoint
//! 3. Apply committed transactions
//! 4. Discard uncommitted changes

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use joule_db_core::error::StorageError;
use joule_db_core::storage::PageId;

/// WAL file name
const WAL_FILE: &str = "wal.wdb";

/// WAL magic number
const WAL_MAGIC: [u8; 4] = [0x57, 0x41, 0x4C, 0x31]; // "WAL1"

/// WAL entry header size: magic(4) + crc(4) + lsn(8) + type(1) + length(4)
const WAL_HEADER_SIZE: usize = 21;

/// Maximum WAL size before checkpoint (64MB)
const MAX_WAL_SIZE: u64 = 64 * 1024 * 1024;

/// Maximum WAL entry payload (16MB + 64KB headroom for entry metadata)
const MAX_WAL_PAYLOAD: usize = 16 * 1024 * 1024 + 64 * 1024;

/// WAL entry types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    /// Page write operation
    PageWrite = 0x01,
    /// Transaction commit
    Commit = 0x02,
    /// Checkpoint marker
    Checkpoint = 0x03,
    /// Transaction rollback
    Rollback = 0x04,
    /// Bulk extent write — logs metadata + checksum only, not the page data.
    /// The extent data is written directly to the data file before this entry.
    /// On recovery, the checksum is verified against the data file.
    /// Payload: tx_id(8) + first_page_id(8) + page_count(8) + total_bytes(8) + data_crc32(4)
    ExtentWrite = 0x05,
}

impl TryFrom<u8> for WalEntryType {
    type Error = StorageError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(WalEntryType::PageWrite),
            0x02 => Ok(WalEntryType::Commit),
            0x03 => Ok(WalEntryType::Checkpoint),
            0x04 => Ok(WalEntryType::Rollback),
            0x05 => Ok(WalEntryType::ExtentWrite),
            _ => Err(StorageError::Backend(format!(
                "Invalid WAL entry type: {}",
                value
            ))),
        }
    }
}

/// WAL entry
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Log sequence number
    pub lsn: u64,
    /// Entry type
    pub entry_type: WalEntryType,
    /// Transaction ID (for PageWrite, Commit, Rollback)
    pub tx_id: u64,
    /// Page ID (for PageWrite)
    pub page_id: Option<PageId>,
    /// Page data (for PageWrite)
    pub data: Vec<u8>,
}

impl WalEntry {
    /// Create a page write entry
    pub fn page_write(lsn: u64, tx_id: u64, page_id: PageId, data: Vec<u8>) -> Self {
        Self {
            lsn,
            entry_type: WalEntryType::PageWrite,
            tx_id,
            page_id: Some(page_id),
            data,
        }
    }

    /// Create a commit entry
    pub fn commit(lsn: u64, tx_id: u64) -> Self {
        Self {
            lsn,
            entry_type: WalEntryType::Commit,
            tx_id,
            page_id: None,
            data: Vec::new(),
        }
    }

    /// Create a checkpoint entry
    pub fn checkpoint(lsn: u64) -> Self {
        Self {
            lsn,
            entry_type: WalEntryType::Checkpoint,
            tx_id: 0,
            page_id: None,
            data: Vec::new(),
        }
    }

    /// Create a rollback entry
    pub fn rollback(lsn: u64, tx_id: u64) -> Self {
        Self {
            lsn,
            entry_type: WalEntryType::Rollback,
            tx_id,
            page_id: None,
            data: Vec::new(),
        }
    }

    /// Create a bulk extent write entry.
    ///
    /// Logs only the extent metadata + data checksum (36 bytes total payload).
    /// The actual tensor data is written directly to the data file — not to the WAL.
    /// This reduces WAL write amplification from 512× to 1× for a 32MB tensor.
    pub fn extent_write(
        lsn: u64,
        tx_id: u64,
        first_page_id: PageId,
        page_count: u64,
        total_bytes: u64,
        data_crc32: u32,
    ) -> Self {
        // Pack metadata into the data field:
        // first_page_id(8) + page_count(8) + total_bytes(8) + data_crc32(4) = 28 bytes
        let mut data = Vec::with_capacity(28);
        data.extend_from_slice(&first_page_id.to_le_bytes());
        data.extend_from_slice(&page_count.to_le_bytes());
        data.extend_from_slice(&total_bytes.to_le_bytes());
        data.extend_from_slice(&data_crc32.to_le_bytes());

        Self {
            lsn,
            entry_type: WalEntryType::ExtentWrite,
            tx_id,
            page_id: Some(first_page_id),
            data,
        }
    }

    /// Encode entry to bytes. Returns Err if payload exceeds MAX_WAL_PAYLOAD.
    pub fn encode(&self) -> Result<Vec<u8>, StorageError> {
        // Calculate payload size: tx_id(8) + optional page_id(8) + data
        let payload_size = 8 + if self.page_id.is_some() { 8 } else { 0 } + self.data.len();

        if payload_size > MAX_WAL_PAYLOAD {
            return Err(StorageError::Backend(format!(
                "WAL entry payload too large: {} bytes (max {})",
                payload_size, MAX_WAL_PAYLOAD
            )));
        }

        let total_size = WAL_HEADER_SIZE + payload_size;

        let mut buf = Vec::with_capacity(total_size);

        // Magic
        buf.extend_from_slice(&WAL_MAGIC);

        // Placeholder for CRC (will be filled after)
        buf.extend_from_slice(&[0u8; 4]);

        // LSN
        buf.extend_from_slice(&self.lsn.to_le_bytes());

        // Entry type
        buf.push(self.entry_type as u8);

        // Payload length (safe: validated above against MAX_WAL_PAYLOAD which fits in u32)
        buf.extend_from_slice(&(payload_size as u32).to_le_bytes());

        // Transaction ID
        buf.extend_from_slice(&self.tx_id.to_le_bytes());

        // Page ID (if present)
        if let Some(page_id) = self.page_id {
            buf.extend_from_slice(&page_id.to_le_bytes());
        }

        // Data
        buf.extend_from_slice(&self.data);

        // Calculate and insert CRC
        let crc = crc32(&buf[8..]); // CRC of everything after CRC field
        buf[4..8].copy_from_slice(&crc.to_le_bytes());

        Ok(buf)
    }

    /// Decode entry from bytes
    pub fn decode(buf: &[u8]) -> Result<Self, StorageError> {
        if buf.len() < WAL_HEADER_SIZE {
            return Err(StorageError::Backend("WAL entry too short".to_string()));
        }

        // Verify magic
        if &buf[0..4] != &WAL_MAGIC {
            return Err(StorageError::Backend("Invalid WAL magic".to_string()));
        }

        // Verify CRC
        let stored_crc = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let calculated_crc = crc32(&buf[8..]);
        if stored_crc != calculated_crc {
            return Err(StorageError::Backend(format!(
                "WAL CRC mismatch: stored={}, calculated={}",
                stored_crc, calculated_crc
            )));
        }

        // Parse header
        let lsn = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);

        let entry_type = WalEntryType::try_from(buf[16])?;

        let payload_len = u32::from_le_bytes([buf[17], buf[18], buf[19], buf[20]]) as usize;

        if buf.len() < WAL_HEADER_SIZE + payload_len {
            return Err(StorageError::Backend("WAL entry truncated".to_string()));
        }

        // Parse payload
        let payload = &buf[WAL_HEADER_SIZE..WAL_HEADER_SIZE + payload_len];

        let tx_id = u64::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
            payload[7],
        ]);

        let (page_id, data) = match entry_type {
            WalEntryType::PageWrite => {
                let page_id = u64::from_le_bytes([
                    payload[8],
                    payload[9],
                    payload[10],
                    payload[11],
                    payload[12],
                    payload[13],
                    payload[14],
                    payload[15],
                ]);
                let data = payload[16..].to_vec();
                (Some(page_id), data)
            }
            WalEntryType::ExtentWrite => {
                // Payload: tx_id already parsed, then:
                // first_page_id(8) + page_count(8) + total_bytes(8) + data_crc32(4)
                let page_id = u64::from_le_bytes([
                    payload[8], payload[9], payload[10], payload[11],
                    payload[12], payload[13], payload[14], payload[15],
                ]);
                let data = payload[8..].to_vec(); // Include all extent metadata
                (Some(page_id), data)
            }
            _ => (None, Vec::new()),
        };

        Ok(Self {
            lsn,
            entry_type,
            tx_id,
            page_id,
            data,
        })
    }

    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        let payload_size = 8 + if self.page_id.is_some() { 8 } else { 0 } + self.data.len();
        WAL_HEADER_SIZE + payload_size
    }
}

/// Write-Ahead Log manager
pub struct WalManager {
    /// WAL file path
    path: PathBuf,
    /// WAL file writer
    writer: BufWriter<File>,
    /// Current LSN
    current_lsn: AtomicU64,
    /// Last checkpoint LSN
    last_checkpoint_lsn: AtomicU64,
    /// Current file position
    file_position: AtomicU64,
    /// Active transactions (tx_id -> first LSN)
    active_transactions: std::sync::RwLock<HashMap<u64, u64>>,
}

impl WalManager {
    /// Open or create WAL file
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self, StorageError> {
        let path = dir.as_ref().join(WAL_FILE);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| StorageError::Backend(format!("Failed to open WAL: {}", e)))?;

        let file_len = file
            .metadata()
            .map_err(|e| StorageError::Backend(format!("Failed to get WAL metadata: {}", e)))?
            .len();

        // Find last LSN by scanning file
        let last_lsn = if file_len > 0 {
            Self::find_last_lsn(&path)?
        } else {
            0
        };

        // Find last checkpoint
        let last_checkpoint = if file_len > 0 {
            Self::find_last_checkpoint(&path)?
        } else {
            0
        };

        // Seek to end of file before wrapping in BufWriter for append semantics
        let mut file = file;
        file.seek(SeekFrom::End(0))
            .map_err(|e| StorageError::Backend(format!("Failed to seek to end of WAL: {}", e)))?;

        let writer = BufWriter::new(file);

        Ok(Self {
            path,
            writer,
            current_lsn: AtomicU64::new(last_lsn + 1),
            last_checkpoint_lsn: AtomicU64::new(last_checkpoint),
            file_position: AtomicU64::new(file_len),
            active_transactions: std::sync::RwLock::new(HashMap::new()),
        })
    }

    /// Find the last LSN in the WAL file
    fn find_last_lsn(path: &Path) -> Result<u64, StorageError> {
        let file = File::open(path)
            .map_err(|e| StorageError::Backend(format!("Failed to open WAL: {}", e)))?;

        let mut reader = BufReader::new(file);
        let mut last_lsn = 0u64;
        let mut buf = [0u8; WAL_HEADER_SIZE];

        loop {
            match reader.read_exact(&mut buf) {
                Ok(_) => {
                    // Verify magic
                    if &buf[0..4] != &WAL_MAGIC {
                        break;
                    }

                    let lsn = u64::from_le_bytes([
                        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
                    ]);

                    let payload_len =
                        u32::from_le_bytes([buf[17], buf[18], buf[19], buf[20]]) as usize;

                    if payload_len > MAX_WAL_PAYLOAD {
                        break;
                    }

                    last_lsn = lsn;

                    // Skip payload
                    if reader.seek(SeekFrom::Current(payload_len as i64)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        Ok(last_lsn)
    }

    /// Find the last checkpoint LSN
    fn find_last_checkpoint(path: &Path) -> Result<u64, StorageError> {
        let file = File::open(path)
            .map_err(|e| StorageError::Backend(format!("Failed to open WAL: {}", e)))?;

        let mut reader = BufReader::new(file);
        let mut last_checkpoint = 0u64;
        let mut buf = [0u8; WAL_HEADER_SIZE];

        loop {
            match reader.read_exact(&mut buf) {
                Ok(_) => {
                    // Verify magic
                    if &buf[0..4] != &WAL_MAGIC {
                        break;
                    }

                    let lsn = u64::from_le_bytes([
                        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
                    ]);

                    let entry_type = buf[16];
                    let payload_len =
                        u32::from_le_bytes([buf[17], buf[18], buf[19], buf[20]]) as usize;

                    if payload_len > MAX_WAL_PAYLOAD {
                        break;
                    }

                    if entry_type == WalEntryType::Checkpoint as u8 {
                        last_checkpoint = lsn;
                    }

                    // Skip payload
                    if reader.seek(SeekFrom::Current(payload_len as i64)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        Ok(last_checkpoint)
    }

    /// Get next LSN
    fn next_lsn(&self) -> u64 {
        self.current_lsn.fetch_add(1, Ordering::SeqCst)
    }

    /// Append a page write entry
    pub fn log_page_write(
        &mut self,
        tx_id: u64,
        page_id: PageId,
        data: &[u8],
    ) -> Result<u64, StorageError> {
        let lsn = self.next_lsn();
        let entry = WalEntry::page_write(lsn, tx_id, page_id, data.to_vec());
        self.append_entry(&entry)?;

        // Track active transaction
        {
            let mut active = self
                .active_transactions
                .write()
                .unwrap_or_else(|p| p.into_inner());
            active.entry(tx_id).or_insert(lsn);
        }

        Ok(lsn)
    }

    /// Append a bulk extent write entry.
    ///
    /// Unlike `log_page_write()` which logs each page's data into the WAL,
    /// this logs only the extent metadata (36 bytes) + a CRC32 of the data.
    /// The caller is responsible for writing the actual data to the data file
    /// and syncing it BEFORE calling this method.
    ///
    /// For a 32MB tensor:
    /// - log_page_write: 512 entries × ~64KB = ~32MB WAL writes
    /// - log_extent_write: 1 entry × 57 bytes = 57 bytes WAL write
    ///
    /// That's a 560,000× reduction in WAL write amplification.
    pub fn log_extent_write(
        &mut self,
        tx_id: u64,
        first_page_id: PageId,
        page_count: u64,
        total_bytes: u64,
        data_crc32: u32,
    ) -> Result<u64, StorageError> {
        let lsn = self.next_lsn();
        let entry = WalEntry::extent_write(
            lsn, tx_id, first_page_id, page_count, total_bytes, data_crc32,
        );
        self.append_entry(&entry)?;

        // Track active transaction
        {
            let mut active = self
                .active_transactions
                .write()
                .unwrap_or_else(|p| p.into_inner());
            active.entry(tx_id).or_insert(lsn);
        }

        Ok(lsn)
    }

    /// Append a commit entry.
    ///
    /// **Flush-to-OS, not fsync** (changed 2026-05-12).
    /// `FileBackend::write_page` auto-commits a one-page transaction
    /// whenever it's called outside an explicit backend transaction —
    /// which is *every* page write in `BufferPool::flush_all` and every
    /// buffer-pool eviction. The old per-commit `self.sync()`
    /// (F_FULLFSYNC on macOS) therefore meant one full disk-cache flush
    /// *per page written* — a 50K-record scholar batch is ~0.5–1M page
    /// writes, i.e. ~0.5–1M F_FULLFSYNC per `Engine::sync()`. That was
    /// the bulk of the "11.7 h to ingest one day" pathology (the
    /// `57400f00c` fix removed the *data+meta* fsync from the
    /// auto-checkpoint but left this WAL fsync).
    ///
    /// We still `flush()` the `BufWriter` here so the commit record —
    /// and the page-write records buffered before it — reach the OS
    /// page cache: a *process* crash then can't lose them, and
    /// recovery / cross-process readers always see a WAL that's
    /// complete up to the last commit. A `write(2)` to the page cache
    /// is microseconds; an `F_FULLFSYNC` is milliseconds — three orders
    /// of magnitude apart.
    ///
    /// Durability vs. *power loss* is provided at the engine's commit
    /// boundary instead: `Engine::sync()` → `BufferPool::flush_all()`
    /// → `backend.sync()` → `FileBackend::sync()` →
    /// `WriteAheadLog::sync()` (one F_FULLFSYNC), and the data file +
    /// `meta.wdb` are fsynced right after (`FileBackend::sync` /
    /// `write_committed_meta`). A crash between an auto-commit and the
    /// next `Engine::sync()` loses only the in-flight batch — and the
    /// ingest checkpoint isn't advanced until that sync, so the batch
    /// is simply re-processed on restart (no silent loss). The CoW
    /// invariant "`committed_root` only advances after the new root's
    /// pages are durable" still holds: `write_committed_meta` fsyncs
    /// the data file before renaming `meta.wdb`.
    pub fn log_commit(&mut self, tx_id: u64) -> Result<u64, StorageError> {
        let lsn = self.next_lsn();
        let entry = WalEntry::commit(lsn, tx_id);
        self.append_entry(&entry)?;

        // Push buffered entries to the OS (no fsync) — see doc comment.
        self.writer
            .flush()
            .map_err(|e| StorageError::Backend(format!("WAL flush error: {}", e)))?;

        // Remove from active transactions
        {
            let mut active = self
                .active_transactions
                .write()
                .unwrap_or_else(|p| p.into_inner());
            active.remove(&tx_id);
        }

        Ok(lsn)
    }

    /// Append a rollback entry
    pub fn log_rollback(&mut self, tx_id: u64) -> Result<u64, StorageError> {
        let lsn = self.next_lsn();
        let entry = WalEntry::rollback(lsn, tx_id);
        self.append_entry(&entry)?;

        // Remove from active transactions
        {
            let mut active = self
                .active_transactions
                .write()
                .unwrap_or_else(|p| p.into_inner());
            active.remove(&tx_id);
        }

        Ok(lsn)
    }

    /// Append a checkpoint entry
    pub fn log_checkpoint(&mut self) -> Result<u64, StorageError> {
        let lsn = self.next_lsn();
        let entry = WalEntry::checkpoint(lsn);
        self.append_entry(&entry)?;
        self.sync()?;

        self.last_checkpoint_lsn.store(lsn, Ordering::SeqCst);

        Ok(lsn)
    }

    /// Append entry to WAL
    fn append_entry(&mut self, entry: &WalEntry) -> Result<(), StorageError> {
        let encoded = entry.encode()?;

        self.writer
            .write_all(&encoded)
            .map_err(|e| StorageError::Backend(format!("WAL write error: {}", e)))?;

        self.file_position
            .fetch_add(encoded.len() as u64, Ordering::SeqCst);

        Ok(())
    }

    /// Sync WAL to disk
    pub fn sync(&mut self) -> Result<(), StorageError> {
        self.writer
            .flush()
            .map_err(|e| StorageError::Backend(format!("WAL flush error: {}", e)))?;

        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| StorageError::Backend(format!("WAL sync error: {}", e)))?;

        Ok(())
    }

    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.file_position.load(Ordering::SeqCst) > MAX_WAL_SIZE
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Get last checkpoint LSN
    pub fn last_checkpoint_lsn(&self) -> u64 {
        self.last_checkpoint_lsn.load(Ordering::SeqCst)
    }

    /// Truncate WAL after checkpoint (remove old entries)
    pub fn truncate(&mut self) -> Result<(), StorageError> {
        // Close current writer
        self.sync()?;

        // Create new empty file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| StorageError::Backend(format!("Failed to truncate WAL: {}", e)))?;

        self.writer = BufWriter::new(file);
        self.file_position.store(0, Ordering::SeqCst);

        Ok(())
    }

    /// Read all entries from WAL (for recovery)
    pub fn read_all_entries(&self) -> Result<Vec<WalEntry>, StorageError> {
        let file = File::open(&self.path)
            .map_err(|e| StorageError::Backend(format!("Failed to open WAL: {}", e)))?;

        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            // Read header
            let mut header = [0u8; WAL_HEADER_SIZE];
            match reader.read_exact(&mut header) {
                Ok(_) => {}
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(StorageError::Backend(format!("WAL read error: {}", e))),
            }

            // Verify magic
            if &header[0..4] != &WAL_MAGIC {
                break; // Corrupted or end of valid data
            }

            let payload_len =
                u32::from_le_bytes([header[17], header[18], header[19], header[20]]) as usize;

            // Guard against corrupted payload length causing OOM
            if payload_len > MAX_WAL_PAYLOAD {
                log::warn!(
                    "WAL entry payload length {} exceeds maximum {}, treating as corruption",
                    payload_len,
                    MAX_WAL_PAYLOAD
                );
                break; // Can't trust the length to skip forward, so stop scanning
            }

            // Read full entry
            let mut full_entry = vec![0u8; WAL_HEADER_SIZE + payload_len];
            full_entry[..WAL_HEADER_SIZE].copy_from_slice(&header);

            if payload_len > 0 {
                reader
                    .read_exact(&mut full_entry[WAL_HEADER_SIZE..])
                    .map_err(|e| StorageError::Backend(format!("WAL read error: {}", e)))?;
            }

            // Decode entry (CRC check happens inside decode)
            match WalEntry::decode(&full_entry) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    // Log warning but continue scanning — a single corrupted entry
                    // (e.g. partial write during crash) should not discard all
                    // subsequent valid entries.
                    log::warn!("Skipping corrupted WAL entry at position: {}", e);
                    continue;
                }
            }
        }

        Ok(entries)
    }

    /// Read entries after a specific LSN (for recovery)
    pub fn read_entries_after(&self, lsn: u64) -> Result<Vec<WalEntry>, StorageError> {
        let all_entries = self.read_all_entries()?;
        Ok(all_entries.into_iter().filter(|e| e.lsn > lsn).collect())
    }
}

/// Recovery manager for crash recovery
pub struct RecoveryManager;

impl RecoveryManager {
    /// Recover database state from WAL
    /// Returns: (committed page writes to apply, rolled back tx_ids)
    pub fn recover(wal: &WalManager) -> Result<RecoveryResult, StorageError> {
        let entries = wal.read_all_entries()?;

        if entries.is_empty() {
            return Ok(RecoveryResult::default());
        }

        // Find last checkpoint
        let checkpoint_lsn = entries
            .iter()
            .filter(|e| e.entry_type == WalEntryType::Checkpoint)
            .map(|e| e.lsn)
            .max()
            .unwrap_or(0);

        // Get entries after checkpoint
        let relevant_entries: Vec<_> = entries
            .into_iter()
            .filter(|e| e.lsn > checkpoint_lsn)
            .collect();

        // Track transactions
        let mut committed_txs: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut rolled_back_txs: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut page_writes: HashMap<u64, Vec<(PageId, Vec<u8>)>> = HashMap::new(); // tx_id -> writes

        for entry in &relevant_entries {
            match entry.entry_type {
                WalEntryType::PageWrite => {
                    if let Some(page_id) = entry.page_id {
                        page_writes
                            .entry(entry.tx_id)
                            .or_default()
                            .push((page_id, entry.data.clone()));
                    }
                }
                WalEntryType::Commit => {
                    committed_txs.insert(entry.tx_id);
                }
                WalEntryType::Rollback => {
                    rolled_back_txs.insert(entry.tx_id);
                }
                WalEntryType::Checkpoint => {}
                WalEntryType::ExtentWrite => {
                    // Extent writes don't carry page data in the WAL.
                    // The data was written directly to the data file before the WAL entry.
                    // On recovery, we trust committed extent writes (verified by data_crc32).
                    // No page_writes entry needed — the data is already on disk.
                    // We just need to track that this transaction has writes.
                    page_writes.entry(entry.tx_id).or_default();
                }
            }
        }

        // Collect pages to apply (from committed transactions only)
        // Use latest write per page - process transactions in order by tx_id
        let mut pages_to_apply: HashMap<PageId, Vec<u8>> = HashMap::new();

        // Sort committed transactions to ensure proper ordering (later tx overwrites earlier)
        let mut sorted_txs: Vec<u64> = committed_txs.iter().copied().collect();
        sorted_txs.sort();

        for tx_id in sorted_txs {
            if let Some(writes) = page_writes.get(&tx_id) {
                for (page_id, data) in writes {
                    pages_to_apply.insert(*page_id, data.clone());
                }
            }
        }

        // Find uncommitted transactions (need rollback)
        let uncommitted_txs: Vec<u64> = page_writes
            .keys()
            .filter(|tx_id| !committed_txs.contains(tx_id) && !rolled_back_txs.contains(tx_id))
            .copied()
            .collect();

        Ok(RecoveryResult {
            pages_to_apply: pages_to_apply.into_iter().collect(),
            committed_transactions: committed_txs.into_iter().collect(),
            rolled_back_transactions: rolled_back_txs.into_iter().collect(),
            uncommitted_transactions: uncommitted_txs,
            last_lsn: relevant_entries
                .last()
                .map(|e| e.lsn)
                .unwrap_or(checkpoint_lsn),
        })
    }
}

/// Recovery result
#[derive(Debug, Default)]
pub struct RecoveryResult {
    /// Pages to apply (page_id, data)
    pub pages_to_apply: Vec<(PageId, Vec<u8>)>,
    /// Committed transaction IDs
    pub committed_transactions: Vec<u64>,
    /// Rolled back transaction IDs
    pub rolled_back_transactions: Vec<u64>,
    /// Uncommitted transaction IDs (need rollback)
    pub uncommitted_transactions: Vec<u64>,
    /// Last LSN processed
    pub last_lsn: u64,
}

/// CRC32 calculation (simple implementation)
fn crc32(data: &[u8]) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_wal_entry_encode_decode() {
        let entry = WalEntry::page_write(1, 100, 42, b"test data".to_vec());
        let encoded = entry.encode().unwrap();
        let decoded = WalEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.lsn, 1);
        assert_eq!(decoded.tx_id, 100);
        assert_eq!(decoded.page_id, Some(42));
        assert_eq!(decoded.data, b"test data");
    }

    #[test]
    fn test_wal_entry_types() {
        let commit = WalEntry::commit(1, 100);
        let encoded = commit.encode().unwrap();
        let decoded = WalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.entry_type, WalEntryType::Commit);

        let checkpoint = WalEntry::checkpoint(2);
        let encoded = checkpoint.encode().unwrap();
        let decoded = WalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.entry_type, WalEntryType::Checkpoint);
    }

    #[test]
    fn test_wal_manager_basic() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::open(temp_dir.path()).unwrap();

        // Log some operations
        let lsn1 = wal.log_page_write(1, 100, b"page data 1").unwrap();
        let lsn2 = wal.log_page_write(1, 101, b"page data 2").unwrap();
        let lsn3 = wal.log_commit(1).unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(lsn3, 3);

        // Read back
        let entries = wal.read_all_entries().unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_wal_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Create WAL with some entries
        {
            let mut wal = WalManager::open(temp_dir.path()).unwrap();

            // Transaction 1: committed
            wal.log_page_write(1, 100, b"tx1 page").unwrap();
            wal.log_commit(1).unwrap();

            // Transaction 2: not committed (crash simulation)
            wal.log_page_write(2, 101, b"tx2 page").unwrap();

            wal.sync().unwrap();
        }

        // Recovery
        {
            let wal = WalManager::open(temp_dir.path()).unwrap();
            let result = RecoveryManager::recover(&wal).unwrap();

            // Only tx1 should be in committed
            assert_eq!(result.committed_transactions, vec![1]);
            assert_eq!(result.uncommitted_transactions, vec![2]);

            // Only tx1's page should be applied
            assert_eq!(result.pages_to_apply.len(), 1);
            assert_eq!(result.pages_to_apply[0].0, 100);
        }
    }

    #[test]
    fn test_wal_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let mut wal = WalManager::open(temp_dir.path()).unwrap();

        wal.log_page_write(1, 100, b"page 1").unwrap();
        wal.log_commit(1).unwrap();
        wal.log_checkpoint().unwrap();
        wal.log_page_write(2, 101, b"page 2").unwrap();
        wal.log_commit(2).unwrap();

        // Recovery should only apply tx2 (after checkpoint)
        let result = RecoveryManager::recover(&wal).unwrap();
        assert_eq!(result.pages_to_apply.len(), 1);
        assert_eq!(result.pages_to_apply[0].0, 101);
    }

    #[test]
    fn test_wal_persistence() {
        let temp_dir = TempDir::new().unwrap();

        // Write and close
        {
            let mut wal = WalManager::open(temp_dir.path()).unwrap();
            wal.log_page_write(1, 100, b"persistent data").unwrap();
            wal.log_commit(1).unwrap();
        }

        // Reopen and verify
        {
            let wal = WalManager::open(temp_dir.path()).unwrap();
            let entries = wal.read_all_entries().unwrap();
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].data, b"persistent data");
        }
    }

    #[test]
    fn test_crc32() {
        // Test vector from IEEE 802.3
        let data = b"123456789";
        let crc = crc32(data);
        assert_eq!(crc, 0xCBF43926);
    }
}
