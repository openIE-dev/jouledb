//! Journaling filesystem concepts — journal entries, write-ahead journaling,
//! checkpoint, crash recovery simulation, ordered vs writeback journaling
//! modes, journal replay.

use std::collections::HashMap;

// ── Journal Mode ────────────────────────────────────────────────────────────

/// Journaling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalMode {
    /// Journal only metadata (inode updates, block allocations).
    Ordered,
    /// Journal both metadata and data.
    Writeback,
    /// Full data journaling — both metadata and data written to journal first.
    Full,
}

// ── Operation Kind ──────────────────────────────────────────────────────────

/// Type of filesystem operation being journaled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    CreateFile { path: String, size: u64 },
    WriteData { path: String, offset: u64, data: Vec<u8> },
    DeleteFile { path: String },
    Rename { old_path: String, new_path: String },
    Mkdir { path: String },
    UpdateMetadata { path: String, new_size: u64 },
    AllocateBlock { block_id: u64 },
    FreeBlock { block_id: u64 },
}

impl OpKind {
    fn is_metadata(&self) -> bool {
        matches!(
            self,
            OpKind::CreateFile { .. }
                | OpKind::DeleteFile { .. }
                | OpKind::Rename { .. }
                | OpKind::Mkdir { .. }
                | OpKind::UpdateMetadata { .. }
                | OpKind::AllocateBlock { .. }
                | OpKind::FreeBlock { .. }
        )
    }

    fn is_data(&self) -> bool {
        matches!(self, OpKind::WriteData { .. })
    }
}

// ── Journal Entry ───────────────────────────────────────────────────────────

/// A single journal entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    pub sequence: u64,
    pub transaction_id: u64,
    pub operation: OpKind,
    pub committed: bool,
}

// ── Transaction State ───────────────────────────────────────────────────────

/// State of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    Active,
    Committed,
    Aborted,
    CheckedPointed,
}

// ── Transaction ─────────────────────────────────────────────────────────────

/// A transaction groups journal entries atomically.
#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: u64,
    pub state: TxState,
    pub entries: Vec<u64>,
    pub created_at: u64,
}

// ── Error ───────────────────────────────────────────────────────────────────

/// Journal errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalError {
    TransactionNotFound(u64),
    TransactionNotActive(u64),
    JournalFull,
    CorruptEntry(u64),
    ReplayFailed(String),
    InvalidState(String),
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::TransactionNotFound(id) => write!(f, "tx {id} not found"),
            JournalError::TransactionNotActive(id) => write!(f, "tx {id} not active"),
            JournalError::JournalFull => write!(f, "journal full"),
            JournalError::CorruptEntry(seq) => write!(f, "corrupt entry at seq {seq}"),
            JournalError::ReplayFailed(msg) => write!(f, "replay failed: {msg}"),
            JournalError::InvalidState(msg) => write!(f, "invalid state: {msg}"),
        }
    }
}

// ── Simulated Disk State ────────────────────────────────────────────────────

/// Represents the on-disk filesystem state that the journal protects.
#[derive(Debug, Clone)]
pub struct DiskState {
    pub files: HashMap<String, Vec<u8>>,
    pub directories: Vec<String>,
    pub allocated_blocks: Vec<u64>,
    pub write_count: u64,
}

impl DiskState {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            directories: vec!["/".to_string()],
            allocated_blocks: Vec::new(),
            write_count: 0,
        }
    }

    fn apply_operation(&mut self, op: &OpKind) {
        self.write_count += 1;
        match op {
            OpKind::CreateFile { path, size } => {
                self.files.insert(path.clone(), vec![0; *size as usize]);
            }
            OpKind::WriteData { path, offset, data } => {
                if let Some(file) = self.files.get_mut(path) {
                    let start = *offset as usize;
                    let end = start + data.len();
                    if end > file.len() {
                        file.resize(end, 0);
                    }
                    file[start..end].copy_from_slice(data);
                }
            }
            OpKind::DeleteFile { path } => {
                self.files.remove(path);
            }
            OpKind::Rename {
                old_path,
                new_path,
            } => {
                if let Some(content) = self.files.remove(old_path) {
                    self.files.insert(new_path.clone(), content);
                }
            }
            OpKind::Mkdir { path } => {
                if !self.directories.contains(path) {
                    self.directories.push(path.clone());
                }
            }
            OpKind::UpdateMetadata { path, new_size } => {
                if let Some(file) = self.files.get_mut(path) {
                    file.resize(*new_size as usize, 0);
                }
            }
            OpKind::AllocateBlock { block_id } => {
                if !self.allocated_blocks.contains(block_id) {
                    self.allocated_blocks.push(*block_id);
                }
            }
            OpKind::FreeBlock { block_id } => {
                self.allocated_blocks.retain(|b| b != block_id);
            }
        }
    }
}

// ── Journal ─────────────────────────────────────────────────────────────────

/// Filesystem journal managing write-ahead logging.
#[derive(Debug)]
pub struct FsJournal {
    mode: JournalMode,
    entries: Vec<JournalEntry>,
    transactions: HashMap<u64, Transaction>,
    next_sequence: u64,
    next_tx_id: u64,
    max_entries: usize,
    /// The "real" on-disk state.
    disk: DiskState,
    /// Checkpoint marker: entries before this sequence are persisted.
    checkpoint_sequence: u64,
    clock: u64,
    /// Count of crash-recovery replays performed.
    replay_count: u64,
}

impl FsJournal {
    /// Create a new filesystem journal.
    pub fn new(mode: JournalMode, max_entries: usize) -> Self {
        Self {
            mode,
            entries: Vec::new(),
            transactions: HashMap::new(),
            next_sequence: 1,
            next_tx_id: 1,
            max_entries,
            disk: DiskState::new(),
            checkpoint_sequence: 0,
            clock: 1,
            replay_count: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        let t = self.clock;
        self.clock += 1;
        t
    }

    /// Current journal mode.
    pub fn mode(&self) -> JournalMode {
        self.mode
    }

    /// Begin a new transaction.
    pub fn begin_transaction(&mut self) -> u64 {
        let id = self.next_tx_id;
        self.next_tx_id += 1;
        let now = self.tick();
        self.transactions.insert(
            id,
            Transaction {
                id,
                state: TxState::Active,
                entries: Vec::new(),
                created_at: now,
            },
        );
        id
    }

    /// Add an operation to a transaction.
    pub fn add_operation(
        &mut self,
        tx_id: u64,
        operation: OpKind,
    ) -> Result<u64, JournalError> {
        let tx = self
            .transactions
            .get(&tx_id)
            .ok_or(JournalError::TransactionNotFound(tx_id))?;
        if tx.state != TxState::Active {
            return Err(JournalError::TransactionNotActive(tx_id));
        }
        if self.entries.len() >= self.max_entries {
            return Err(JournalError::JournalFull);
        }

        // Check if this operation should be journaled based on mode
        let should_journal = match self.mode {
            JournalMode::Ordered => operation.is_metadata(),
            JournalMode::Writeback => operation.is_metadata(),
            JournalMode::Full => true, // journal everything
        };

        let seq = self.next_sequence;
        self.next_sequence += 1;

        if should_journal {
            self.entries.push(JournalEntry {
                sequence: seq,
                transaction_id: tx_id,
                operation,
                committed: false,
            });
        } else {
            // In ordered/writeback modes, data writes go directly to disk
            // but we still track them in the journal for the commit protocol
            self.entries.push(JournalEntry {
                sequence: seq,
                transaction_id: tx_id,
                operation,
                committed: false,
            });
        }

        // Add entry sequence to transaction
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.entries.push(seq);
        }

        Ok(seq)
    }

    /// Commit a transaction — apply all its operations to disk.
    pub fn commit(&mut self, tx_id: u64) -> Result<(), JournalError> {
        let tx = self
            .transactions
            .get(&tx_id)
            .ok_or(JournalError::TransactionNotFound(tx_id))?;
        if tx.state != TxState::Active {
            return Err(JournalError::TransactionNotActive(tx_id));
        }

        let entry_seqs: Vec<u64> = tx.entries.clone();

        // Mark all entries as committed
        for entry in &mut self.entries {
            if entry_seqs.contains(&entry.sequence) {
                entry.committed = true;
            }
        }

        // Apply operations to disk in order
        let operations: Vec<OpKind> = self
            .entries
            .iter()
            .filter(|e| entry_seqs.contains(&e.sequence))
            .map(|e| e.operation.clone())
            .collect();

        for op in &operations {
            self.disk.apply_operation(op);
        }

        // Mark transaction as committed
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.state = TxState::Committed;
        }

        Ok(())
    }

    /// Abort a transaction — discard its journal entries.
    pub fn abort(&mut self, tx_id: u64) -> Result<(), JournalError> {
        let tx = self
            .transactions
            .get(&tx_id)
            .ok_or(JournalError::TransactionNotFound(tx_id))?;
        if tx.state != TxState::Active {
            return Err(JournalError::TransactionNotActive(tx_id));
        }

        let entry_seqs: Vec<u64> = tx.entries.clone();

        // Remove entries for this transaction
        self.entries
            .retain(|e| !entry_seqs.contains(&e.sequence));

        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.state = TxState::Aborted;
        }

        Ok(())
    }

    /// Checkpoint — persist journal to disk and truncate.
    /// All committed entries up to this point are considered durable.
    pub fn checkpoint(&mut self) -> u64 {
        let max_committed_seq = self
            .entries
            .iter()
            .filter(|e| e.committed)
            .map(|e| e.sequence)
            .max()
            .unwrap_or(self.checkpoint_sequence);

        self.checkpoint_sequence = max_committed_seq;

        // Mark committed transactions as checkpointed
        let committed_tx_ids: Vec<u64> = self
            .transactions
            .values()
            .filter(|tx| tx.state == TxState::Committed)
            .map(|tx| tx.id)
            .collect();

        for tx_id in committed_tx_ids {
            if let Some(tx) = self.transactions.get_mut(&tx_id) {
                tx.state = TxState::CheckedPointed;
            }
        }

        // Remove checkpointed entries from the journal
        self.entries
            .retain(|e| e.sequence > self.checkpoint_sequence || !e.committed);

        self.checkpoint_sequence
    }

    /// Simulate a crash — lose all uncommitted work.
    /// Returns the state after crash (only committed data survives).
    pub fn simulate_crash(&mut self) -> CrashReport {
        // Count lost entries (uncommitted)
        let lost_entries: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| !e.committed)
            .map(|e| e.sequence)
            .collect();

        let lost_tx_ids: Vec<u64> = self
            .transactions
            .values()
            .filter(|tx| tx.state == TxState::Active)
            .map(|tx| tx.id)
            .collect();

        // Remove uncommitted entries
        self.entries.retain(|e| e.committed);

        // Abort active transactions
        for tx_id in &lost_tx_ids {
            if let Some(tx) = self.transactions.get_mut(tx_id) {
                tx.state = TxState::Aborted;
            }
        }

        CrashReport {
            lost_entries: lost_entries.len(),
            lost_transactions: lost_tx_ids.len(),
            surviving_entries: self.entries.len(),
        }
    }

    /// Replay the journal after a crash — re-apply committed but
    /// not-yet-checkpointed entries.
    pub fn replay(&mut self) -> Result<ReplayReport, JournalError> {
        let committed_entries: Vec<OpKind> = self
            .entries
            .iter()
            .filter(|e| e.committed && e.sequence > self.checkpoint_sequence)
            .map(|e| e.operation.clone())
            .collect();

        let replayed = committed_entries.len();
        for op in &committed_entries {
            self.disk.apply_operation(op);
        }

        self.replay_count += 1;

        Ok(ReplayReport {
            replayed_entries: replayed,
            replay_number: self.replay_count,
        })
    }

    /// Get the current disk state.
    pub fn disk_state(&self) -> &DiskState {
        &self.disk
    }

    /// Number of journal entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of active transactions.
    pub fn active_transactions(&self) -> usize {
        self.transactions
            .values()
            .filter(|tx| tx.state == TxState::Active)
            .count()
    }

    /// Number of committed transactions.
    pub fn committed_transactions(&self) -> usize {
        self.transactions
            .values()
            .filter(|tx| tx.state == TxState::Committed)
            .count()
    }

    /// Total transactions ever created.
    pub fn total_transactions(&self) -> usize {
        self.transactions.len()
    }

    /// Get the checkpoint sequence number.
    pub fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence
    }

    /// Get a transaction by ID.
    pub fn get_transaction(&self, tx_id: u64) -> Option<&Transaction> {
        self.transactions.get(&tx_id)
    }

    /// Journal utilization (entries / max_entries).
    pub fn utilization(&self) -> f64 {
        self.entries.len() as f64 / self.max_entries as f64
    }
}

// ── Crash Report ────────────────────────────────────────────────────────────

/// Report from a simulated crash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrashReport {
    pub lost_entries: usize,
    pub lost_transactions: usize,
    pub surviving_entries: usize,
}

/// Report from a journal replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    pub replayed_entries: usize,
    pub replay_number: u64,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_and_commit_transaction() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/test.txt".into(),
                    size: 100,
                },
            )
            .unwrap();
        journal.commit(tx).unwrap();
        assert!(journal.disk_state().files.contains_key("/test.txt"));
    }

    #[test]
    fn test_abort_transaction() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/test.txt".into(),
                    size: 100,
                },
            )
            .unwrap();
        journal.abort(tx).unwrap();
        assert!(!journal.disk_state().files.contains_key("/test.txt"));
    }

    #[test]
    fn test_write_data_operation() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/f.txt".into(),
                    size: 10,
                },
            )
            .unwrap();
        journal
            .add_operation(
                tx,
                OpKind::WriteData {
                    path: "/f.txt".into(),
                    offset: 0,
                    data: b"hello".to_vec(),
                },
            )
            .unwrap();
        journal.commit(tx).unwrap();
        let file_data = &journal.disk_state().files["/f.txt"];
        assert_eq!(&file_data[..5], b"hello");
    }

    #[test]
    fn test_delete_file_operation() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx1 = journal.begin_transaction();
        journal
            .add_operation(
                tx1,
                OpKind::CreateFile {
                    path: "/del.txt".into(),
                    size: 5,
                },
            )
            .unwrap();
        journal.commit(tx1).unwrap();

        let tx2 = journal.begin_transaction();
        journal
            .add_operation(
                tx2,
                OpKind::DeleteFile {
                    path: "/del.txt".into(),
                },
            )
            .unwrap();
        journal.commit(tx2).unwrap();
        assert!(!journal.disk_state().files.contains_key("/del.txt"));
    }

    #[test]
    fn test_rename_operation() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx1 = journal.begin_transaction();
        journal
            .add_operation(
                tx1,
                OpKind::CreateFile {
                    path: "/old.txt".into(),
                    size: 5,
                },
            )
            .unwrap();
        journal.commit(tx1).unwrap();

        let tx2 = journal.begin_transaction();
        journal
            .add_operation(
                tx2,
                OpKind::Rename {
                    old_path: "/old.txt".into(),
                    new_path: "/new.txt".into(),
                },
            )
            .unwrap();
        journal.commit(tx2).unwrap();
        assert!(!journal.disk_state().files.contains_key("/old.txt"));
        assert!(journal.disk_state().files.contains_key("/new.txt"));
    }

    #[test]
    fn test_mkdir_operation() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::Mkdir {
                    path: "/data".into(),
                },
            )
            .unwrap();
        journal.commit(tx).unwrap();
        assert!(journal.disk_state().directories.contains(&"/data".to_string()));
    }

    #[test]
    fn test_crash_loses_uncommitted() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);

        // Committed transaction
        let tx1 = journal.begin_transaction();
        journal
            .add_operation(
                tx1,
                OpKind::CreateFile {
                    path: "/safe.txt".into(),
                    size: 10,
                },
            )
            .unwrap();
        journal.commit(tx1).unwrap();

        // Uncommitted transaction
        let tx2 = journal.begin_transaction();
        journal
            .add_operation(
                tx2,
                OpKind::CreateFile {
                    path: "/lost.txt".into(),
                    size: 10,
                },
            )
            .unwrap();

        let report = journal.simulate_crash();
        assert_eq!(report.lost_entries, 1);
        assert_eq!(report.lost_transactions, 1);
        assert!(journal.disk_state().files.contains_key("/safe.txt"));
        assert!(!journal.disk_state().files.contains_key("/lost.txt"));
    }

    #[test]
    fn test_checkpoint() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/a.txt".into(),
                    size: 5,
                },
            )
            .unwrap();
        journal.commit(tx).unwrap();
        let seq = journal.checkpoint();
        assert!(seq > 0);
        assert_eq!(journal.entry_count(), 0); // Entries truncated after checkpoint
    }

    #[test]
    fn test_replay_after_crash() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/replay.txt".into(),
                    size: 10,
                },
            )
            .unwrap();
        journal.commit(tx).unwrap();
        // Replay committed-but-not-checkpointed entries
        let report = journal.replay().unwrap();
        assert_eq!(report.replayed_entries, 1);
    }

    #[test]
    fn test_journal_full() {
        let mut journal = FsJournal::new(JournalMode::Full, 2);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/a.txt".into(),
                    size: 1,
                },
            )
            .unwrap();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/b.txt".into(),
                    size: 1,
                },
            )
            .unwrap();
        let result = journal.add_operation(
            tx,
            OpKind::CreateFile {
                path: "/c.txt".into(),
                size: 1,
            },
        );
        assert!(matches!(result, Err(JournalError::JournalFull)));
    }

    #[test]
    fn test_transaction_not_found() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let result = journal.commit(999);
        assert!(matches!(
            result,
            Err(JournalError::TransactionNotFound(999))
        ));
    }

    #[test]
    fn test_double_commit() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal.commit(tx).unwrap();
        let result = journal.commit(tx);
        assert!(matches!(result, Err(JournalError::TransactionNotActive(_))));
    }

    #[test]
    fn test_active_transaction_count() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx1 = journal.begin_transaction();
        let _tx2 = journal.begin_transaction();
        assert_eq!(journal.active_transactions(), 2);
        journal.commit(tx1).unwrap();
        assert_eq!(journal.active_transactions(), 1);
    }

    #[test]
    fn test_journal_mode_ordered() {
        let journal = FsJournal::new(JournalMode::Ordered, 100);
        assert_eq!(journal.mode(), JournalMode::Ordered);
    }

    #[test]
    fn test_utilization() {
        let mut journal = FsJournal::new(JournalMode::Full, 10);
        let tx = journal.begin_transaction();
        journal
            .add_operation(
                tx,
                OpKind::CreateFile {
                    path: "/a".into(),
                    size: 1,
                },
            )
            .unwrap();
        assert!((journal.utilization() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_block_alloc_free() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        journal
            .add_operation(tx, OpKind::AllocateBlock { block_id: 42 })
            .unwrap();
        journal.commit(tx).unwrap();
        assert!(journal.disk_state().allocated_blocks.contains(&42));

        let tx2 = journal.begin_transaction();
        journal
            .add_operation(tx2, OpKind::FreeBlock { block_id: 42 })
            .unwrap();
        journal.commit(tx2).unwrap();
        assert!(!journal.disk_state().allocated_blocks.contains(&42));
    }

    #[test]
    fn test_multiple_transactions_isolation() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx1 = journal.begin_transaction();
        let tx2 = journal.begin_transaction();

        journal
            .add_operation(
                tx1,
                OpKind::CreateFile {
                    path: "/tx1.txt".into(),
                    size: 5,
                },
            )
            .unwrap();
        journal
            .add_operation(
                tx2,
                OpKind::CreateFile {
                    path: "/tx2.txt".into(),
                    size: 5,
                },
            )
            .unwrap();

        journal.commit(tx1).unwrap();
        // tx2 not committed yet
        assert!(journal.disk_state().files.contains_key("/tx1.txt"));
        assert!(!journal.disk_state().files.contains_key("/tx2.txt"));
    }

    #[test]
    fn test_get_transaction() {
        let mut journal = FsJournal::new(JournalMode::Full, 100);
        let tx = journal.begin_transaction();
        let tx_obj = journal.get_transaction(tx).unwrap();
        assert_eq!(tx_obj.state, TxState::Active);
    }
}
