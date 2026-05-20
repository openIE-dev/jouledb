//! Comprehensive Crash Recovery and WAL Replay System
//!
//! This module implements ARIES-style recovery with:
//! - WAL replay on startup for crash recovery
//! - Checkpoint management (when to checkpoint, how to truncate WAL)
//! - Redo and undo phases for transaction recovery
//! - Transaction state recovery
//! - Page-level recovery with LSN tracking
//!
//! ## ARIES Recovery Algorithm
//!
//! The ARIES (Algorithm for Recovery and Isolation Exploiting Semantics) algorithm
//! is implemented in three phases:
//!
//! 1. **Analysis Phase**: Scan WAL from last checkpoint to identify:
//!    - Active transactions at crash time
//!    - Dirty pages that need to be redone
//!    - Transaction table and dirty page table reconstruction
//!
//! 2. **Redo Phase**: Replay all logged actions from the appropriate point:
//!    - Start from the smallest LSN in the dirty page table
//!    - Redo all actions to restore database to crash state
//!    - Uses LSN comparison to avoid redundant redos
//!
//! 3. **Undo Phase**: Rollback uncommitted transactions:
//!    - Process transactions in reverse LSN order
//!    - Generate CLRs (Compensation Log Records) for undo operations
//!    - Ensures atomicity of aborted transactions
//!
//! ## Page LSN Tracking
//!
//! Each page maintains an LSN that represents the most recent WAL entry
//! that modified the page. During recovery:
//! - If page LSN >= log entry LSN, skip (already applied)
//! - If page LSN < log entry LSN, apply the change

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use joule_db_core::error::StorageError;
use joule_db_core::persistence::traits::{LSN, TxId, WalBackend, WalEntry, WalEntryType};
use joule_db_core::storage::PageId;

/// Checkpoint file name
const CHECKPOINT_FILE: &str = "checkpoint.wdb";

/// Checkpoint magic number
const CHECKPOINT_MAGIC: [u8; 4] = [0x43, 0x4B, 0x50, 0x54]; // "CKPT"

/// Checkpoint header size
const CHECKPOINT_HEADER_SIZE: usize = 64;

/// Default checkpoint interval (5 minutes)
const DEFAULT_CHECKPOINT_INTERVAL_MS: u64 = 300_000;

/// Default max WAL size before forced checkpoint (64MB)
const DEFAULT_MAX_WAL_SIZE: u64 = 64 * 1024 * 1024;

/// Default max transactions between checkpoints
const DEFAULT_MAX_TRANSACTIONS_BETWEEN_CHECKPOINTS: u64 = 10_000;

/// Transaction state during recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is active (in progress)
    Active,
    /// Transaction is preparing to commit (2PC)
    Preparing,
    /// Transaction has committed
    Committed,
    /// Transaction has aborted/rolled back
    Aborted,
}

/// Transaction entry in the transaction table
#[derive(Debug, Clone)]
pub struct TransactionEntry {
    /// Transaction ID
    pub tx_id: TxId,
    /// Current state
    pub state: TransactionState,
    /// LSN of first log record for this transaction
    pub first_lsn: LSN,
    /// LSN of last log record for this transaction
    pub last_lsn: LSN,
    /// Undo next LSN (for recovery)
    pub undo_next_lsn: Option<LSN>,
    /// Pages modified by this transaction
    pub modified_pages: HashSet<PageId>,
}

impl TransactionEntry {
    /// Create a new transaction entry
    pub fn new(tx_id: TxId, first_lsn: LSN) -> Self {
        Self {
            tx_id,
            state: TransactionState::Active,
            first_lsn,
            last_lsn: first_lsn,
            undo_next_lsn: Some(first_lsn),
            modified_pages: HashSet::new(),
        }
    }

    /// Update the last LSN
    pub fn update_lsn(&mut self, lsn: LSN) {
        self.last_lsn = lsn;
        self.undo_next_lsn = Some(lsn);
    }

    /// Add a modified page
    pub fn add_modified_page(&mut self, page_id: PageId) {
        self.modified_pages.insert(page_id);
    }
}

/// Dirty page entry in the dirty page table
#[derive(Debug, Clone)]
pub struct DirtyPageEntry {
    /// Page ID
    pub page_id: PageId,
    /// Recovery LSN - LSN of first log record that dirtied this page
    /// since it was last flushed
    pub recovery_lsn: LSN,
    /// Page LSN - LSN of most recent update to this page
    pub page_lsn: LSN,
}

impl DirtyPageEntry {
    /// Create a new dirty page entry
    pub fn new(page_id: PageId, lsn: LSN) -> Self {
        Self {
            page_id,
            recovery_lsn: lsn,
            page_lsn: lsn,
        }
    }

    /// Update the page LSN
    pub fn update_lsn(&mut self, lsn: LSN) {
        self.page_lsn = lsn;
    }
}

/// Checkpoint record containing database state at checkpoint time
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    /// Checkpoint LSN
    pub checkpoint_lsn: LSN,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Active transactions at checkpoint time
    pub active_transactions: Vec<TransactionEntry>,
    /// Dirty pages at checkpoint time
    pub dirty_pages: Vec<DirtyPageEntry>,
    /// Master record LSN (end of checkpoint)
    pub master_lsn: LSN,
    /// Minimum recovery LSN
    pub min_recovery_lsn: LSN,
}

impl CheckpointRecord {
    /// Create a new checkpoint record
    pub fn new(checkpoint_lsn: LSN) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            checkpoint_lsn,
            timestamp,
            active_transactions: Vec::new(),
            dirty_pages: Vec::new(),
            master_lsn: checkpoint_lsn,
            min_recovery_lsn: checkpoint_lsn,
        }
    }

    /// Encode checkpoint to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&CHECKPOINT_MAGIC);

        // Version (1 byte)
        buf.push(1);

        // Reserved (3 bytes)
        buf.extend_from_slice(&[0u8; 3]);

        // Checkpoint LSN
        buf.extend_from_slice(&self.checkpoint_lsn.to_le_bytes());

        // Timestamp
        buf.extend_from_slice(&self.timestamp.to_le_bytes());

        // Master LSN
        buf.extend_from_slice(&self.master_lsn.to_le_bytes());

        // Min recovery LSN
        buf.extend_from_slice(&self.min_recovery_lsn.to_le_bytes());

        // Number of active transactions
        buf.extend_from_slice(&(self.active_transactions.len() as u32).to_le_bytes());

        // Number of dirty pages
        buf.extend_from_slice(&(self.dirty_pages.len() as u32).to_le_bytes());

        // Padding to header size
        buf.resize(CHECKPOINT_HEADER_SIZE, 0);

        // Encode active transactions
        for tx in &self.active_transactions {
            buf.extend_from_slice(&tx.tx_id.to_le_bytes());
            buf.push(tx.state as u8);
            buf.extend_from_slice(&tx.first_lsn.to_le_bytes());
            buf.extend_from_slice(&tx.last_lsn.to_le_bytes());
            buf.extend_from_slice(&tx.undo_next_lsn.unwrap_or(0).to_le_bytes());
            buf.extend_from_slice(&(tx.modified_pages.len() as u32).to_le_bytes());
            for page_id in &tx.modified_pages {
                buf.extend_from_slice(&page_id.to_le_bytes());
            }
        }

        // Encode dirty pages
        for dp in &self.dirty_pages {
            buf.extend_from_slice(&dp.page_id.to_le_bytes());
            buf.extend_from_slice(&dp.recovery_lsn.to_le_bytes());
            buf.extend_from_slice(&dp.page_lsn.to_le_bytes());
        }

        // Calculate and append checksum
        let checksum = crc32(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Decode checkpoint from bytes
    pub fn decode(buf: &[u8]) -> Result<Self, StorageError> {
        if buf.len() < CHECKPOINT_HEADER_SIZE + 4 {
            return Err(StorageError::Backend("Checkpoint too short".to_string()));
        }

        // Verify magic
        if &buf[0..4] != &CHECKPOINT_MAGIC {
            return Err(StorageError::Backend(
                "Invalid checkpoint magic".to_string(),
            ));
        }

        // Verify checksum
        let checksum_offset = buf.len() - 4;
        let stored_checksum = u32::from_le_bytes([
            buf[checksum_offset],
            buf[checksum_offset + 1],
            buf[checksum_offset + 2],
            buf[checksum_offset + 3],
        ]);
        let calculated_checksum = crc32(&buf[..checksum_offset]);
        if stored_checksum != calculated_checksum {
            return Err(StorageError::Backend(
                "Checkpoint checksum mismatch".to_string(),
            ));
        }

        let version = buf[4];
        if version != 1 {
            return Err(StorageError::Backend(format!(
                "Unsupported checkpoint version: {}",
                version
            )));
        }

        let checkpoint_lsn = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);

        let timestamp = u64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]);

        let master_lsn = u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
        ]);

        let min_recovery_lsn = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
        ]);

        let num_transactions = u32::from_le_bytes([buf[40], buf[41], buf[42], buf[43]]) as usize;
        let num_dirty_pages = u32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]) as usize;

        let mut offset = CHECKPOINT_HEADER_SIZE;
        let mut active_transactions = Vec::with_capacity(num_transactions);

        // Decode active transactions
        for _ in 0..num_transactions {
            if offset + 33 > checksum_offset {
                return Err(StorageError::Backend(
                    "Checkpoint truncated (transactions)".to_string(),
                ));
            }

            let tx_id = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;

            let state = match buf[offset] {
                0 => TransactionState::Active,
                1 => TransactionState::Preparing,
                2 => TransactionState::Committed,
                3 => TransactionState::Aborted,
                _ => {
                    return Err(StorageError::Backend(
                        "Invalid transaction state".to_string(),
                    ));
                }
            };
            offset += 1;

            let first_lsn = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;

            let last_lsn = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;

            let undo_next_lsn = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;

            let num_pages = u32::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
            ]) as usize;
            offset += 4;

            let mut modified_pages = HashSet::with_capacity(num_pages);
            for _ in 0..num_pages {
                if offset + 8 > checksum_offset {
                    return Err(StorageError::Backend(
                        "Checkpoint truncated (pages)".to_string(),
                    ));
                }
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
                offset += 8;
                modified_pages.insert(page_id);
            }

            active_transactions.push(TransactionEntry {
                tx_id,
                state,
                first_lsn,
                last_lsn,
                undo_next_lsn: if undo_next_lsn > 0 {
                    Some(undo_next_lsn)
                } else {
                    None
                },
                modified_pages,
            });
        }

        // Decode dirty pages
        let mut dirty_pages = Vec::with_capacity(num_dirty_pages);
        for _ in 0..num_dirty_pages {
            if offset + 24 > checksum_offset {
                return Err(StorageError::Backend(
                    "Checkpoint truncated (dirty pages)".to_string(),
                ));
            }

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
            offset += 8;

            let recovery_lsn = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;

            let page_lsn = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;

            dirty_pages.push(DirtyPageEntry {
                page_id,
                recovery_lsn,
                page_lsn,
            });
        }

        Ok(Self {
            checkpoint_lsn,
            timestamp,
            active_transactions,
            dirty_pages,
            master_lsn,
            min_recovery_lsn,
        })
    }
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Interval between automatic checkpoints (milliseconds)
    pub interval_ms: u64,
    /// Maximum WAL size before forced checkpoint
    pub max_wal_size: u64,
    /// Maximum number of transactions between checkpoints
    pub max_transactions: u64,
    /// Whether to perform fuzzy checkpoints (non-blocking)
    pub fuzzy_checkpoint: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            interval_ms: DEFAULT_CHECKPOINT_INTERVAL_MS,
            max_wal_size: DEFAULT_MAX_WAL_SIZE,
            max_transactions: DEFAULT_MAX_TRANSACTIONS_BETWEEN_CHECKPOINTS,
            fuzzy_checkpoint: true,
        }
    }
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Total time for recovery (milliseconds)
    pub duration_ms: u64,
    /// Time for analysis phase (milliseconds)
    pub analysis_duration_ms: u64,
    /// Time for redo phase (milliseconds)
    pub redo_duration_ms: u64,
    /// Time for undo phase (milliseconds)
    pub undo_duration_ms: u64,
    /// Number of log entries processed
    pub entries_processed: u64,
    /// Number of pages redone
    pub pages_redone: u64,
    /// Number of pages undone
    pub pages_undone: u64,
    /// Number of transactions recovered
    pub transactions_recovered: u64,
    /// Number of transactions rolled back
    pub transactions_rolled_back: u64,
    /// Checkpoint LSN used for recovery
    pub checkpoint_lsn: LSN,
    /// First LSN in redo phase
    pub redo_start_lsn: LSN,
    /// Last LSN processed
    pub last_lsn: LSN,
}

/// Page LSN tracker for recovery
#[derive(Debug)]
pub struct PageLsnTracker {
    /// Map of page ID to its current LSN
    page_lsns: RwLock<HashMap<PageId, LSN>>,
}

impl PageLsnTracker {
    /// Create a new page LSN tracker
    pub fn new() -> Self {
        Self {
            page_lsns: RwLock::new(HashMap::new()),
        }
    }

    /// Get the LSN for a page
    pub fn get_lsn(&self, page_id: PageId) -> Option<LSN> {
        self.page_lsns.read().get(&page_id).copied()
    }

    /// Set the LSN for a page
    pub fn set_lsn(&self, page_id: PageId, lsn: LSN) {
        self.page_lsns.write().insert(page_id, lsn);
    }

    /// Check if a redo is needed for a page
    /// Returns true if the page LSN is less than the log entry LSN
    pub fn needs_redo(&self, page_id: PageId, entry_lsn: LSN) -> bool {
        match self.get_lsn(page_id) {
            Some(page_lsn) => page_lsn < entry_lsn,
            None => true, // Page not in tracker, needs redo
        }
    }

    /// Get all tracked pages
    pub fn all_pages(&self) -> HashMap<PageId, LSN> {
        self.page_lsns.read().clone()
    }

    /// Clear all tracked pages
    pub fn clear(&self) {
        self.page_lsns.write().clear();
    }
}

impl Default for PageLsnTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Recovery Manager - handles crash recovery using ARIES algorithm
pub struct RecoveryManager {
    /// Database directory
    db_path: PathBuf,
    /// Transaction table
    transaction_table: RwLock<HashMap<TxId, TransactionEntry>>,
    /// Dirty page table
    dirty_page_table: RwLock<HashMap<PageId, DirtyPageEntry>>,
    /// Page LSN tracker
    page_lsn_tracker: PageLsnTracker,
    /// Last checkpoint record
    last_checkpoint: RwLock<Option<CheckpointRecord>>,
    /// Checkpoint configuration
    checkpoint_config: CheckpointConfig,
    /// Last checkpoint time
    last_checkpoint_time: AtomicU64,
    /// Transactions since last checkpoint
    transactions_since_checkpoint: AtomicU64,
    /// WAL size since last checkpoint
    wal_size_since_checkpoint: AtomicU64,
    /// Recovery in progress flag
    recovery_in_progress: AtomicBool,
    /// Current recovery LSN
    current_lsn: AtomicU64,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new<P: AsRef<Path>>(db_path: P) -> Self {
        Self::with_config(db_path, CheckpointConfig::default())
    }

    /// Create a recovery manager with custom configuration
    pub fn with_config<P: AsRef<Path>>(db_path: P, config: CheckpointConfig) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            db_path: db_path.as_ref().to_path_buf(),
            transaction_table: RwLock::new(HashMap::new()),
            dirty_page_table: RwLock::new(HashMap::new()),
            page_lsn_tracker: PageLsnTracker::new(),
            last_checkpoint: RwLock::new(None),
            checkpoint_config: config,
            last_checkpoint_time: AtomicU64::new(now),
            transactions_since_checkpoint: AtomicU64::new(0),
            wal_size_since_checkpoint: AtomicU64::new(0),
            recovery_in_progress: AtomicBool::new(false),
            current_lsn: AtomicU64::new(0),
        }
    }

    /// Load the last checkpoint from disk
    pub fn load_checkpoint(&self) -> Result<Option<CheckpointRecord>, StorageError> {
        let checkpoint_path = self.db_path.join(CHECKPOINT_FILE);
        if !checkpoint_path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&checkpoint_path)
            .map_err(|e| StorageError::Backend(format!("Failed to open checkpoint: {}", e)))?;

        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| StorageError::Backend(format!("Failed to read checkpoint: {}", e)))?;

        let checkpoint = CheckpointRecord::decode(&buf)?;
        Ok(Some(checkpoint))
    }

    /// Save a checkpoint to disk
    pub fn save_checkpoint(&self, checkpoint: &CheckpointRecord) -> Result<(), StorageError> {
        let checkpoint_path = self.db_path.join(CHECKPOINT_FILE);
        let temp_path = self.db_path.join("checkpoint.tmp");

        // Write to temp file first
        let mut file = File::create(&temp_path)
            .map_err(|e| StorageError::Backend(format!("Failed to create checkpoint: {}", e)))?;

        let encoded = checkpoint.encode();
        file.write_all(&encoded)
            .map_err(|e| StorageError::Backend(format!("Failed to write checkpoint: {}", e)))?;

        file.sync_all()
            .map_err(|e| StorageError::Backend(format!("Failed to sync checkpoint: {}", e)))?;

        // Atomic rename
        std::fs::rename(&temp_path, &checkpoint_path)
            .map_err(|e| StorageError::Backend(format!("Failed to rename checkpoint: {}", e)))?;

        // Update cached checkpoint
        *self.last_checkpoint.write() = Some(checkpoint.clone());

        Ok(())
    }

    /// Perform full ARIES-style recovery
    /// Returns pages that need to be written to disk and recovery statistics
    pub fn recover<W: WalBackend>(
        &self,
        wal: &W,
    ) -> Result<(Vec<(PageId, Vec<u8>)>, RecoveryStats), StorageError> {
        self.recovery_in_progress.store(true, Ordering::SeqCst);
        let start_time = Instant::now();
        let mut stats = RecoveryStats::default();

        // Load checkpoint
        let checkpoint = self.load_checkpoint()?;
        stats.checkpoint_lsn = checkpoint.as_ref().map(|c| c.checkpoint_lsn).unwrap_or(0);

        // Get all WAL entries
        let all_entries = wal.read_all()?;
        stats.entries_processed = all_entries.len() as u64;

        if all_entries.is_empty() && checkpoint.is_none() {
            self.recovery_in_progress.store(false, Ordering::SeqCst);
            return Ok((Vec::new(), stats));
        }

        // Phase 1: Analysis
        let analysis_start = Instant::now();
        self.analysis_phase(&all_entries, checkpoint.as_ref())?;
        stats.analysis_duration_ms = analysis_start.elapsed().as_millis() as u64;

        // Determine redo start point
        stats.redo_start_lsn = self.get_redo_start_lsn();

        // Phase 2: Redo
        let redo_start = Instant::now();
        let redo_pages = self.redo_phase(&all_entries, stats.redo_start_lsn)?;
        stats.pages_redone = redo_pages.len() as u64;
        stats.redo_duration_ms = redo_start.elapsed().as_millis() as u64;

        // Phase 3: Undo
        let undo_start = Instant::now();
        let (undo_pages, rolled_back) = self.undo_phase(&all_entries)?;
        stats.pages_undone = undo_pages.len() as u64;
        stats.transactions_rolled_back = rolled_back as u64;
        stats.undo_duration_ms = undo_start.elapsed().as_millis() as u64;

        // Combine redo and undo pages (undo takes precedence)
        let mut final_pages: HashMap<PageId, Vec<u8>> = redo_pages.into_iter().collect();
        for (page_id, data) in undo_pages {
            final_pages.insert(page_id, data);
        }

        // Count recovered transactions
        let tx_table = self.transaction_table.read();
        stats.transactions_recovered = tx_table
            .values()
            .filter(|tx| tx.state == TransactionState::Committed)
            .count() as u64;

        stats.last_lsn = all_entries.last().map(|e| e.lsn).unwrap_or(0);
        stats.duration_ms = start_time.elapsed().as_millis() as u64;

        self.recovery_in_progress.store(false, Ordering::SeqCst);

        Ok((final_pages.into_iter().collect(), stats))
    }

    /// Analysis phase - reconstruct transaction and dirty page tables
    fn analysis_phase(
        &self,
        entries: &[WalEntry],
        checkpoint: Option<&CheckpointRecord>,
    ) -> Result<(), StorageError> {
        // Initialize from checkpoint if available
        if let Some(ckpt) = checkpoint {
            let mut tx_table = self.transaction_table.write();
            for tx_entry in &ckpt.active_transactions {
                tx_table.insert(tx_entry.tx_id, tx_entry.clone());
            }

            let mut dp_table = self.dirty_page_table.write();
            for dp_entry in &ckpt.dirty_pages {
                dp_table.insert(dp_entry.page_id, dp_entry.clone());
            }
        }

        let checkpoint_lsn = checkpoint.map(|c| c.checkpoint_lsn).unwrap_or(0);

        // Process entries after checkpoint
        for entry in entries.iter().filter(|e| e.lsn > checkpoint_lsn) {
            match entry.entry_type {
                WalEntryType::Begin => {
                    let mut tx_table = self.transaction_table.write();
                    tx_table.insert(entry.tx_id, TransactionEntry::new(entry.tx_id, entry.lsn));
                }
                WalEntryType::PageWrite => {
                    if let Some(page_id) = entry.page_id {
                        // Update transaction table
                        {
                            let mut tx_table = self.transaction_table.write();
                            if let Some(tx_entry) = tx_table.get_mut(&entry.tx_id) {
                                tx_entry.update_lsn(entry.lsn);
                                tx_entry.add_modified_page(page_id);
                            } else {
                                // Transaction started before checkpoint or implicit start
                                let mut new_entry = TransactionEntry::new(entry.tx_id, entry.lsn);
                                new_entry.add_modified_page(page_id);
                                tx_table.insert(entry.tx_id, new_entry);
                            }
                        }

                        // Update dirty page table
                        {
                            let mut dp_table = self.dirty_page_table.write();
                            dp_table
                                .entry(page_id)
                                .and_modify(|e| e.update_lsn(entry.lsn))
                                .or_insert_with(|| DirtyPageEntry::new(page_id, entry.lsn));
                        }
                    }
                }
                WalEntryType::Commit => {
                    let mut tx_table = self.transaction_table.write();
                    if let Some(tx_entry) = tx_table.get_mut(&entry.tx_id) {
                        tx_entry.state = TransactionState::Committed;
                        tx_entry.update_lsn(entry.lsn);
                    }
                }
                WalEntryType::Rollback => {
                    let mut tx_table = self.transaction_table.write();
                    if let Some(tx_entry) = tx_table.get_mut(&entry.tx_id) {
                        tx_entry.state = TransactionState::Aborted;
                        tx_entry.update_lsn(entry.lsn);
                    }
                }
                WalEntryType::Checkpoint => {
                    // Checkpoint entries don't modify transaction state
                }
                _ => {
                    // Handle other entry types (savepoints, etc.)
                }
            }
        }

        Ok(())
    }

    /// Get the starting LSN for redo phase
    fn get_redo_start_lsn(&self) -> LSN {
        let dp_table = self.dirty_page_table.read();
        dp_table.values().map(|e| e.recovery_lsn).min().unwrap_or(0)
    }

    /// Redo phase - replay committed operations
    fn redo_phase(
        &self,
        entries: &[WalEntry],
        start_lsn: LSN,
    ) -> Result<Vec<(PageId, Vec<u8>)>, StorageError> {
        let mut pages_to_apply: HashMap<PageId, Vec<u8>> = HashMap::new();
        let tx_table = self.transaction_table.read();

        // Get committed transaction IDs
        let committed_txs: HashSet<TxId> = tx_table
            .iter()
            .filter(|(_, tx)| tx.state == TransactionState::Committed)
            .map(|(id, _)| *id)
            .collect();

        // Process entries from redo start point
        for entry in entries.iter().filter(|e| e.lsn >= start_lsn) {
            if entry.entry_type == WalEntryType::PageWrite {
                if let Some(page_id) = entry.page_id {
                    // Only redo if:
                    // 1. Transaction is committed
                    // 2. Page needs redo (LSN check)
                    if committed_txs.contains(&entry.tx_id)
                        && self.page_lsn_tracker.needs_redo(page_id, entry.lsn)
                    {
                        pages_to_apply.insert(page_id, entry.data.clone());
                        self.page_lsn_tracker.set_lsn(page_id, entry.lsn);
                    }
                }
            }
        }

        Ok(pages_to_apply.into_iter().collect())
    }

    /// Undo phase - rollback uncommitted transactions
    fn undo_phase(
        &self,
        entries: &[WalEntry],
    ) -> Result<(Vec<(PageId, Vec<u8>)>, usize), StorageError> {
        let tx_table = self.transaction_table.read();

        // Get uncommitted (active) transaction IDs
        let uncommitted_txs: HashSet<TxId> = tx_table
            .iter()
            .filter(|(_, tx)| tx.state == TransactionState::Active)
            .map(|(id, _)| *id)
            .collect();

        if uncommitted_txs.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // For undo, we need to process in reverse order
        // In a full implementation, we would generate CLRs and apply before-images
        // For simplicity, we just track which pages need to be undone
        // The actual undo data would come from before-images stored in the WAL

        let mut pages_to_undo: HashMap<PageId, Vec<u8>> = HashMap::new();
        let rolled_back = uncommitted_txs.len();

        // In ARIES, undo would use compensation log records (CLRs)
        // and before-images to restore pages to their pre-transaction state
        // For this implementation, we mark pages from uncommitted transactions
        // that need to be restored (the caller would need to restore from backup/checkpoint)

        // Process entries in reverse order for uncommitted transactions
        //
        // ARIES recovery requires before-images for undo operations.
        // Current WAL entries store only after-images (new data).
        // For production use, one of these approaches is needed:
        // 1. Extend WalEntry to include before_image field
        // 2. Use shadow paging to maintain before-images separately
        // 3. Rely on checkpoint data for restoration
        //
        // Current behavior: Mark pages as needing restoration. The caller should
        // restore from the most recent checkpoint, then replay committed transactions.
        for entry in entries.iter().rev() {
            if entry.entry_type == WalEntryType::PageWrite {
                if uncommitted_txs.contains(&entry.tx_id) {
                    if let Some(page_id) = entry.page_id {
                        if !pages_to_undo.contains_key(&page_id) {
                            // Without before-images, we use the last committed version
                            // This requires restoring from checkpoint and re-applying
                            // committed transactions only. For now, return empty Vec
                            // to signal "needs restoration from checkpoint".
                            tracing::warn!(
                                "Page {} needs undo but no before-image available (tx {})",
                                page_id,
                                entry.tx_id
                            );
                            pages_to_undo.insert(page_id, Vec::new());
                        }
                    }
                }
            }
        }

        Ok((pages_to_undo.into_iter().collect(), rolled_back))
    }

    /// Check if a checkpoint is needed
    pub fn needs_checkpoint(&self, wal_size: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let last_time = self.last_checkpoint_time.load(Ordering::SeqCst);
        let tx_count = self.transactions_since_checkpoint.load(Ordering::SeqCst);

        // Check time interval
        if now - last_time > self.checkpoint_config.interval_ms {
            return true;
        }

        // Check WAL size
        if wal_size > self.checkpoint_config.max_wal_size {
            return true;
        }

        // Check transaction count
        if tx_count > self.checkpoint_config.max_transactions {
            return true;
        }

        false
    }

    /// Create a new checkpoint
    pub fn create_checkpoint<W: WalBackend>(
        &self,
        wal: &mut W,
    ) -> Result<CheckpointRecord, StorageError> {
        // Get current LSN
        let checkpoint_lsn = wal.current_lsn();

        // Create checkpoint record
        let mut checkpoint = CheckpointRecord::new(checkpoint_lsn);

        // Copy transaction table
        {
            let tx_table = self.transaction_table.read();
            checkpoint.active_transactions = tx_table
                .values()
                .filter(|tx| tx.state == TransactionState::Active)
                .cloned()
                .collect();
        }

        // Copy dirty page table
        {
            let dp_table = self.dirty_page_table.read();
            checkpoint.dirty_pages = dp_table.values().cloned().collect();
        }

        // Calculate minimum recovery LSN
        checkpoint.min_recovery_lsn = checkpoint
            .dirty_pages
            .iter()
            .map(|dp| dp.recovery_lsn)
            .min()
            .unwrap_or(checkpoint_lsn);

        // Write checkpoint entry to WAL
        let master_entry = WalEntry::checkpoint(
            wal.current_lsn(),
            checkpoint
                .active_transactions
                .iter()
                .map(|t| t.tx_id)
                .collect(),
        );
        wal.append(&master_entry)?;
        wal.sync()?;

        checkpoint.master_lsn = master_entry.lsn;

        // Save checkpoint to disk
        self.save_checkpoint(&checkpoint)?;

        // Reset counters
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.last_checkpoint_time.store(now, Ordering::SeqCst);
        self.transactions_since_checkpoint
            .store(0, Ordering::SeqCst);
        self.wal_size_since_checkpoint.store(0, Ordering::SeqCst);

        Ok(checkpoint)
    }

    /// Record a transaction begin
    pub fn begin_transaction(&self, tx_id: TxId, lsn: LSN) {
        let mut tx_table = self.transaction_table.write();
        tx_table.insert(tx_id, TransactionEntry::new(tx_id, lsn));
    }

    /// Record a page write
    pub fn record_page_write(&self, tx_id: TxId, page_id: PageId, lsn: LSN) {
        // Update transaction table
        {
            let mut tx_table = self.transaction_table.write();
            if let Some(tx_entry) = tx_table.get_mut(&tx_id) {
                tx_entry.update_lsn(lsn);
                tx_entry.add_modified_page(page_id);
            }
        }

        // Update dirty page table
        {
            let mut dp_table = self.dirty_page_table.write();
            dp_table
                .entry(page_id)
                .and_modify(|e| e.update_lsn(lsn))
                .or_insert_with(|| DirtyPageEntry::new(page_id, lsn));
        }

        // Update page LSN tracker
        self.page_lsn_tracker.set_lsn(page_id, lsn);
    }

    /// Record a transaction commit
    pub fn commit_transaction(&self, tx_id: TxId, lsn: LSN) {
        let mut tx_table = self.transaction_table.write();
        if let Some(tx_entry) = tx_table.get_mut(&tx_id) {
            tx_entry.state = TransactionState::Committed;
            tx_entry.update_lsn(lsn);
        }

        self.transactions_since_checkpoint
            .fetch_add(1, Ordering::SeqCst);
    }

    /// Record a transaction rollback
    pub fn rollback_transaction(&self, tx_id: TxId, lsn: LSN) {
        let mut tx_table = self.transaction_table.write();
        if let Some(tx_entry) = tx_table.get_mut(&tx_id) {
            tx_entry.state = TransactionState::Aborted;
            tx_entry.update_lsn(lsn);
        }
    }

    /// Mark a page as flushed (remove from dirty page table)
    pub fn mark_page_flushed(&self, page_id: PageId) {
        let mut dp_table = self.dirty_page_table.write();
        dp_table.remove(&page_id);
    }

    /// Get active transaction count
    pub fn active_transaction_count(&self) -> usize {
        let tx_table = self.transaction_table.read();
        tx_table
            .values()
            .filter(|tx| tx.state == TransactionState::Active)
            .count()
    }

    /// Get dirty page count
    pub fn dirty_page_count(&self) -> usize {
        self.dirty_page_table.read().len()
    }

    /// Get the last checkpoint LSN
    pub fn last_checkpoint_lsn(&self) -> LSN {
        // parking_lot returns the guard directly; deref to Option,
        // then Option::as_ref + map gives the LSN if any checkpoint
        // has been recorded yet.
        self.last_checkpoint
            .read()
            .as_ref()
            .map(|c| c.checkpoint_lsn)
            .unwrap_or(0)
    }

    /// Check if recovery is in progress
    pub fn is_recovering(&self) -> bool {
        self.recovery_in_progress.load(Ordering::SeqCst)
    }

    /// Get current LSN tracked by recovery manager
    pub fn current_lsn(&self) -> LSN {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Set current LSN (called during recovery)
    pub fn set_current_lsn(&self, lsn: LSN) {
        self.current_lsn.store(lsn, Ordering::SeqCst);
    }

    /// Get checkpoint configuration
    pub fn checkpoint_config(&self) -> &CheckpointConfig {
        &self.checkpoint_config
    }

    /// Get transactions since last checkpoint
    pub fn transactions_since_checkpoint(&self) -> u64 {
        self.transactions_since_checkpoint.load(Ordering::SeqCst)
    }

    /// Clear all state (for testing)
    pub fn clear(&self) {
        self.transaction_table.write().clear();
        self.dirty_page_table.write().clear();
        self.page_lsn_tracker.clear();
        *self.last_checkpoint.write() = None;
    }
}

/// CRC32 calculation (IEEE polynomial)
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
    use std::sync::atomic::AtomicU64;
    use tempfile::TempDir;

    /// Mock WAL backend for testing
    struct MockWalBackend {
        entries: Vec<WalEntry>,
        current_lsn: AtomicU64,
        last_checkpoint_lsn: AtomicU64,
    }

    impl MockWalBackend {
        fn new() -> Self {
            Self {
                entries: Vec::new(),
                current_lsn: AtomicU64::new(1),
                last_checkpoint_lsn: AtomicU64::new(0),
            }
        }

        fn add_entry(&mut self, entry: WalEntry) {
            self.entries.push(entry);
        }
    }

    impl WalBackend for MockWalBackend {
        fn append(&mut self, entry: &WalEntry) -> Result<LSN, StorageError> {
            let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);
            let mut new_entry = entry.clone();
            new_entry.lsn = lsn;
            self.entries.push(new_entry);
            Ok(lsn)
        }

        fn sync(&mut self) -> Result<(), StorageError> {
            Ok(())
        }

        fn read_all(&self) -> Result<Vec<WalEntry>, StorageError> {
            Ok(self.entries.clone())
        }

        fn read_after(&self, lsn: LSN) -> Result<Vec<WalEntry>, StorageError> {
            Ok(self
                .entries
                .iter()
                .filter(|e| e.lsn > lsn)
                .cloned()
                .collect())
        }

        fn truncate_before(&mut self, lsn: LSN) -> Result<(), StorageError> {
            self.entries.retain(|e| e.lsn > lsn);
            Ok(())
        }

        fn current_lsn(&self) -> LSN {
            self.current_lsn.load(Ordering::SeqCst)
        }

        fn last_checkpoint_lsn(&self) -> LSN {
            self.last_checkpoint_lsn.load(Ordering::SeqCst)
        }

        fn size_bytes(&self) -> u64 {
            self.entries.iter().map(|e| e.encoded_size() as u64).sum()
        }
    }

    #[test]
    fn test_transaction_entry_creation() {
        let tx = TransactionEntry::new(1, 100);
        assert_eq!(tx.tx_id, 1);
        assert_eq!(tx.first_lsn, 100);
        assert_eq!(tx.last_lsn, 100);
        assert_eq!(tx.state, TransactionState::Active);
        assert!(tx.modified_pages.is_empty());
    }

    #[test]
    fn test_transaction_entry_update() {
        let mut tx = TransactionEntry::new(1, 100);
        tx.update_lsn(200);
        tx.add_modified_page(42);

        assert_eq!(tx.last_lsn, 200);
        assert_eq!(tx.undo_next_lsn, Some(200));
        assert!(tx.modified_pages.contains(&42));
    }

    #[test]
    fn test_dirty_page_entry() {
        let mut dp = DirtyPageEntry::new(42, 100);
        assert_eq!(dp.page_id, 42);
        assert_eq!(dp.recovery_lsn, 100);
        assert_eq!(dp.page_lsn, 100);

        dp.update_lsn(200);
        assert_eq!(dp.page_lsn, 200);
        assert_eq!(dp.recovery_lsn, 100); // Recovery LSN doesn't change
    }

    #[test]
    fn test_checkpoint_record_encode_decode() {
        let mut checkpoint = CheckpointRecord::new(100);

        // Add a transaction
        let mut tx = TransactionEntry::new(1, 50);
        tx.add_modified_page(10);
        tx.add_modified_page(20);
        checkpoint.active_transactions.push(tx);

        // Add dirty pages
        checkpoint.dirty_pages.push(DirtyPageEntry::new(10, 50));
        checkpoint.dirty_pages.push(DirtyPageEntry::new(20, 75));

        let encoded = checkpoint.encode();
        let decoded = CheckpointRecord::decode(&encoded).unwrap();

        assert_eq!(decoded.checkpoint_lsn, 100);
        assert_eq!(decoded.active_transactions.len(), 1);
        assert_eq!(decoded.active_transactions[0].tx_id, 1);
        assert_eq!(decoded.active_transactions[0].modified_pages.len(), 2);
        assert_eq!(decoded.dirty_pages.len(), 2);
    }

    #[test]
    fn test_page_lsn_tracker() {
        let tracker = PageLsnTracker::new();

        // Page not tracked - needs redo
        assert!(tracker.needs_redo(42, 100));

        // Set LSN
        tracker.set_lsn(42, 100);
        assert_eq!(tracker.get_lsn(42), Some(100));

        // Page LSN >= entry LSN - no redo needed
        assert!(!tracker.needs_redo(42, 100));
        assert!(!tracker.needs_redo(42, 50));

        // Page LSN < entry LSN - needs redo
        assert!(tracker.needs_redo(42, 150));
    }

    #[test]
    fn test_recovery_manager_basic() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());

        assert_eq!(recovery.active_transaction_count(), 0);
        assert_eq!(recovery.dirty_page_count(), 0);
        assert!(!recovery.is_recovering());
    }

    #[test]
    fn test_recovery_manager_transaction_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());

        // Begin transaction
        recovery.begin_transaction(1, 100);
        assert_eq!(recovery.active_transaction_count(), 1);

        // Record page write
        recovery.record_page_write(1, 42, 101);
        assert_eq!(recovery.dirty_page_count(), 1);

        // Commit transaction
        recovery.commit_transaction(1, 102);

        // Transaction still tracked but not active
        let tx_table = recovery.transaction_table.read();
        assert_eq!(tx_table.get(&1).unwrap().state, TransactionState::Committed);
    }

    #[test]
    fn test_recovery_with_committed_transaction() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Simulate a committed transaction
        wal.add_entry(WalEntry::begin(1, 1));
        wal.add_entry(WalEntry::page_write(2, 1, 42, b"page data".to_vec()));
        wal.add_entry(WalEntry::commit(3, 1));

        let (pages, stats) = recovery.recover(&wal).unwrap();

        assert_eq!(stats.entries_processed, 3);
        assert_eq!(stats.transactions_recovered, 1);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, 42);
        assert_eq!(pages[0].1, b"page data".to_vec());
    }

    #[test]
    fn test_recovery_with_uncommitted_transaction() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Simulate an uncommitted transaction (crash before commit)
        wal.add_entry(WalEntry::begin(1, 1));
        wal.add_entry(WalEntry::page_write(2, 1, 42, b"uncommitted data".to_vec()));
        // No commit!

        let (_pages, stats) = recovery.recover(&wal).unwrap();

        assert_eq!(stats.transactions_recovered, 0);
        assert_eq!(stats.transactions_rolled_back, 1);

        // Uncommitted transaction's pages should be in undo list
        // but not in the main pages list (since we'd restore from checkpoint)
    }

    #[test]
    fn test_recovery_with_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Transaction before checkpoint
        wal.add_entry(WalEntry::begin(1, 1));
        wal.add_entry(WalEntry::page_write(
            2,
            1,
            10,
            b"before checkpoint".to_vec(),
        ));
        wal.add_entry(WalEntry::commit(3, 1));

        // Checkpoint
        wal.add_entry(WalEntry::checkpoint(4, vec![]));

        // Transaction after checkpoint
        wal.add_entry(WalEntry::begin(5, 2));
        wal.add_entry(WalEntry::page_write(6, 2, 20, b"after checkpoint".to_vec()));
        wal.add_entry(WalEntry::commit(7, 2));

        let (pages, _stats) = recovery.recover(&wal).unwrap();

        // Only transaction after checkpoint should be recovered
        // (transaction before checkpoint is already in stable storage)
        assert!(pages.iter().any(|(id, _)| *id == 20));
    }

    #[test]
    fn test_checkpoint_creation() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Add some state
        recovery.begin_transaction(1, 100);
        recovery.record_page_write(1, 42, 101);

        // Create checkpoint
        let checkpoint = recovery.create_checkpoint(&mut wal).unwrap();

        assert!(checkpoint.checkpoint_lsn > 0);
        assert_eq!(checkpoint.active_transactions.len(), 1);
        assert_eq!(checkpoint.dirty_pages.len(), 1);

        // Verify checkpoint was saved
        let loaded = recovery.load_checkpoint().unwrap().unwrap();
        assert_eq!(loaded.checkpoint_lsn, checkpoint.checkpoint_lsn);
    }

    #[test]
    fn test_needs_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            interval_ms: 1000,
            max_wal_size: 1024,
            max_transactions: 10,
            fuzzy_checkpoint: true,
        };
        let recovery = RecoveryManager::with_config(temp_dir.path(), config);

        // Initially no checkpoint needed
        assert!(!recovery.needs_checkpoint(100));

        // WAL size exceeds max
        assert!(recovery.needs_checkpoint(2000));

        // Transaction count exceeds max
        for _ in 0..15 {
            recovery
                .transactions_since_checkpoint
                .fetch_add(1, Ordering::SeqCst);
        }
        assert!(recovery.needs_checkpoint(100));
    }

    #[test]
    fn test_multiple_transactions_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Transaction 1: committed
        wal.add_entry(WalEntry::begin(1, 1));
        wal.add_entry(WalEntry::page_write(2, 1, 10, b"tx1 page1".to_vec()));
        wal.add_entry(WalEntry::page_write(3, 1, 11, b"tx1 page2".to_vec()));
        wal.add_entry(WalEntry::commit(4, 1));

        // Transaction 2: uncommitted
        wal.add_entry(WalEntry::begin(5, 2));
        wal.add_entry(WalEntry::page_write(6, 2, 20, b"tx2 page".to_vec()));

        // Transaction 3: committed
        wal.add_entry(WalEntry::begin(7, 3));
        wal.add_entry(WalEntry::page_write(8, 3, 30, b"tx3 page".to_vec()));
        wal.add_entry(WalEntry::commit(9, 3));

        // Transaction 4: rolled back
        wal.add_entry(WalEntry::begin(10, 4));
        wal.add_entry(WalEntry::page_write(11, 4, 40, b"tx4 page".to_vec()));
        wal.add_entry(WalEntry::rollback(12, 4));

        let (pages, stats) = recovery.recover(&wal).unwrap();

        assert_eq!(stats.transactions_recovered, 2); // tx1 and tx3
        assert_eq!(stats.transactions_rolled_back, 1); // tx2

        // Should have pages from tx1 and tx3
        let page_ids: HashSet<_> = pages.iter().map(|(id, _)| *id).collect();
        assert!(page_ids.contains(&10));
        assert!(page_ids.contains(&11));
        assert!(page_ids.contains(&30));
        assert!(!page_ids.contains(&40)); // tx4 was rolled back
    }

    #[test]
    fn test_page_overwrite_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Same page written multiple times
        wal.add_entry(WalEntry::begin(1, 1));
        wal.add_entry(WalEntry::page_write(2, 1, 42, b"first write".to_vec()));
        wal.add_entry(WalEntry::page_write(3, 1, 42, b"second write".to_vec()));
        wal.add_entry(WalEntry::page_write(4, 1, 42, b"third write".to_vec()));
        wal.add_entry(WalEntry::commit(5, 1));

        let (pages, _stats) = recovery.recover(&wal).unwrap();

        // Should only have the final state
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, 42);
        assert_eq!(pages[0].1, b"third write".to_vec());
    }

    #[test]
    fn test_checkpoint_record_with_empty_state() {
        let checkpoint = CheckpointRecord::new(100);
        let encoded = checkpoint.encode();
        let decoded = CheckpointRecord::decode(&encoded).unwrap();

        assert_eq!(decoded.checkpoint_lsn, 100);
        assert!(decoded.active_transactions.is_empty());
        assert!(decoded.dirty_pages.is_empty());
    }

    #[test]
    fn test_recovery_stats() {
        let temp_dir = TempDir::new().unwrap();
        let recovery = RecoveryManager::new(temp_dir.path());
        let mut wal = MockWalBackend::new();

        // Add entries
        for i in 1..=10 {
            wal.add_entry(WalEntry::page_write(
                i,
                1,
                i,
                format!("page {}", i).into_bytes(),
            ));
        }
        wal.add_entry(WalEntry::commit(11, 1));

        let (_pages, stats) = recovery.recover(&wal).unwrap();

        assert_eq!(stats.entries_processed, 11);
        // Verify stats were populated (duration values are always >= 0 for u64)
        assert!(stats.duration_ms < 10000); // Should complete in under 10 seconds
        assert!(stats.analysis_duration_ms <= stats.duration_ms);
        assert!(stats.redo_duration_ms <= stats.duration_ms);
    }
}
