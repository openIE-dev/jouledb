//! Transaction log (ARIES-style concepts) — WAL records (begin/update/commit/abort),
//! log sequence numbers, undo/redo operations, checkpoint records, log truncation,
//! crash recovery simulation.
//!
//! Replaces external WAL libraries with a pure Rust transaction log that models
//! ARIES recovery protocol concepts.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by transaction log operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxLogError {
    /// Transaction not found.
    TxNotFound(u64),
    /// Transaction already committed or aborted.
    TxFinished(u64),
    /// Log sequence number not found.
    LsnNotFound(u64),
    /// Invalid operation.
    InvalidOp(String),
}

impl fmt::Display for TxLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TxNotFound(id) => write!(f, "transaction {id} not found"),
            Self::TxFinished(id) => write!(f, "transaction {id} already finished"),
            Self::LsnNotFound(lsn) => write!(f, "LSN {lsn} not found"),
            Self::InvalidOp(msg) => write!(f, "invalid operation: {msg}"),
        }
    }
}

impl std::error::Error for TxLogError {}

// ── Log record types ─────────────────────────────────────────────

/// Type of WAL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogRecordType {
    /// Transaction begin.
    Begin,
    /// Data update: page_id, offset, old_value, new_value.
    Update {
        page_id: u64,
        offset: u32,
        old_value: Vec<u8>,
        new_value: Vec<u8>,
    },
    /// Compensation log record (CLR) for undo.
    Compensate {
        page_id: u64,
        offset: u32,
        undo_value: Vec<u8>,
        undo_next_lsn: u64,
    },
    /// Transaction commit.
    Commit,
    /// Transaction abort.
    Abort,
    /// Checkpoint record.
    Checkpoint {
        active_txns: Vec<(u64, u64)>, // (tx_id, last_lsn)
        dirty_pages: Vec<(u64, u64)>, // (page_id, recovery_lsn)
    },
    /// End record (after commit or abort processing complete).
    End,
}

/// A single WAL log record.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// Log sequence number.
    pub lsn: u64,
    /// Transaction ID (0 for checkpoint records).
    pub tx_id: u64,
    /// Previous LSN for this transaction (0 if none).
    pub prev_lsn: u64,
    /// The record type and payload.
    pub record_type: LogRecordType,
}

// ── Transaction state ────────────────────────────────────────────

/// State of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    Active,
    Committed,
    Aborted,
}

/// Transaction metadata.
#[derive(Debug, Clone)]
pub struct Transaction {
    pub tx_id: u64,
    pub state: TxState,
    pub first_lsn: u64,
    pub last_lsn: u64,
}

// ── Page state for recovery ──────────────────────────────────────

/// Simulated page state for crash recovery.
#[derive(Debug, Clone)]
pub struct PageState {
    pub page_id: u64,
    pub data: HashMap<u32, Vec<u8>>,
    pub page_lsn: u64,
}

impl PageState {
    fn new(page_id: u64) -> Self {
        Self {
            page_id,
            data: HashMap::new(),
            page_lsn: 0,
        }
    }
}

// ── Recovery result ──────────────────────────────────────────────

/// Result of crash recovery.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Number of records replayed in the redo phase.
    pub redo_count: usize,
    /// Number of records undone in the undo phase.
    pub undo_count: usize,
    /// Transactions that were rolled back.
    pub rolled_back: Vec<u64>,
    /// Final page states after recovery.
    pub page_states: HashMap<u64, PageState>,
}

// ── Log statistics ───────────────────────────────────────────────

/// Transaction log statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogStats {
    pub total_records: usize,
    pub begin_count: usize,
    pub update_count: usize,
    pub commit_count: usize,
    pub abort_count: usize,
    pub checkpoint_count: usize,
    pub compensate_count: usize,
    pub end_count: usize,
    pub active_tx_count: usize,
    pub next_lsn: u64,
    pub next_tx_id: u64,
    pub truncated_below: u64,
}

// ── TransactionLog ───────────────────────────────────────────────

/// An in-memory transaction log with ARIES-style WAL records.
pub struct TransactionLog {
    records: Vec<LogRecord>,
    transactions: HashMap<u64, Transaction>,
    next_lsn: u64,
    next_tx_id: u64,
    /// LSN below which records have been truncated.
    truncated_below: u64,
}

impl TransactionLog {
    /// Create a new empty transaction log.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            transactions: HashMap::new(),
            next_lsn: 1,
            next_tx_id: 1,
            truncated_below: 0,
        }
    }

    /// Begin a new transaction. Returns the transaction ID.
    pub fn begin_tx(&mut self) -> u64 {
        let tx_id = self.next_tx_id;
        self.next_tx_id += 1;
        let lsn = self.append_record(tx_id, LogRecordType::Begin);
        self.transactions.insert(
            tx_id,
            Transaction {
                tx_id,
                state: TxState::Active,
                first_lsn: lsn,
                last_lsn: lsn,
            },
        );
        tx_id
    }

    /// Write an update record for a transaction.
    pub fn write_update(
        &mut self,
        tx_id: u64,
        page_id: u64,
        offset: u32,
        old_value: Vec<u8>,
        new_value: Vec<u8>,
    ) -> Result<u64, TxLogError> {
        self.check_active(tx_id)?;
        let lsn = self.append_record(
            tx_id,
            LogRecordType::Update {
                page_id,
                offset,
                old_value,
                new_value,
            },
        );
        self.transactions.get_mut(&tx_id).unwrap().last_lsn = lsn;
        Ok(lsn)
    }

    /// Commit a transaction.
    pub fn commit_tx(&mut self, tx_id: u64) -> Result<u64, TxLogError> {
        self.check_active(tx_id)?;
        let lsn = self.append_record(tx_id, LogRecordType::Commit);
        let tx = self.transactions.get_mut(&tx_id).unwrap();
        tx.state = TxState::Committed;
        tx.last_lsn = lsn;
        // Write end record.
        let end_lsn = self.append_record(tx_id, LogRecordType::End);
        self.transactions.get_mut(&tx_id).unwrap().last_lsn = end_lsn;
        Ok(lsn)
    }

    /// Abort a transaction, generating CLR records for undo.
    pub fn abort_tx(&mut self, tx_id: u64) -> Result<Vec<u64>, TxLogError> {
        self.check_active(tx_id)?;
        let abort_lsn = self.append_record(tx_id, LogRecordType::Abort);
        self.transactions.get_mut(&tx_id).unwrap().last_lsn = abort_lsn;

        // Generate CLR records by scanning backwards.
        let mut clr_lsns = Vec::new();
        let update_records: Vec<LogRecord> = self
            .records
            .iter()
            .filter(|r| r.tx_id == tx_id)
            .filter(|r| matches!(r.record_type, LogRecordType::Update { .. }))
            .cloned()
            .collect();

        for rec in update_records.iter().rev() {
            if let LogRecordType::Update {
                page_id,
                offset,
                old_value,
                ..
            } = &rec.record_type
            {
                let undo_next = rec.prev_lsn;
                let clr_lsn = self.append_record(
                    tx_id,
                    LogRecordType::Compensate {
                        page_id: *page_id,
                        offset: *offset,
                        undo_value: old_value.clone(),
                        undo_next_lsn: undo_next,
                    },
                );
                self.transactions.get_mut(&tx_id).unwrap().last_lsn = clr_lsn;
                clr_lsns.push(clr_lsn);
            }
        }

        let tx = self.transactions.get_mut(&tx_id).unwrap();
        tx.state = TxState::Aborted;

        // Write end record.
        let end_lsn = self.append_record(tx_id, LogRecordType::End);
        self.transactions.get_mut(&tx_id).unwrap().last_lsn = end_lsn;

        Ok(clr_lsns)
    }

    /// Write a checkpoint record.
    pub fn write_checkpoint(&mut self) -> u64 {
        let active_txns: Vec<(u64, u64)> = self
            .transactions
            .values()
            .filter(|tx| tx.state == TxState::Active)
            .map(|tx| (tx.tx_id, tx.last_lsn))
            .collect();
        // Dirty pages would normally come from the buffer pool; we simulate empty.
        self.append_record(
            0,
            LogRecordType::Checkpoint {
                active_txns,
                dirty_pages: Vec::new(),
            },
        )
    }

    /// Truncate the log, removing records with LSN < min_lsn.
    pub fn truncate(&mut self, min_lsn: u64) {
        self.records.retain(|r| r.lsn >= min_lsn);
        if min_lsn > self.truncated_below {
            self.truncated_below = min_lsn;
        }
    }

    /// Get a record by LSN.
    pub fn get_record(&self, lsn: u64) -> Option<&LogRecord> {
        self.records.iter().find(|r| r.lsn == lsn)
    }

    /// Get a transaction by ID.
    pub fn get_transaction(&self, tx_id: u64) -> Option<&Transaction> {
        self.transactions.get(&tx_id)
    }

    /// Get all records for a transaction.
    pub fn tx_records(&self, tx_id: u64) -> Vec<&LogRecord> {
        self.records.iter().filter(|r| r.tx_id == tx_id).collect()
    }

    /// All records in the log.
    pub fn all_records(&self) -> &[LogRecord] {
        &self.records
    }

    /// Simulate crash recovery using ARIES-style redo/undo.
    pub fn recover(&self) -> RecoveryResult {
        let mut pages: HashMap<u64, PageState> = HashMap::new();
        let mut redo_count = 0;
        let mut undo_count = 0;

        // Find the latest checkpoint.
        let checkpoint = self
            .records
            .iter()
            .rev()
            .find(|r| matches!(r.record_type, LogRecordType::Checkpoint { .. }));

        let start_lsn = checkpoint.map_or(0, |c| c.lsn);

        // ── Redo phase ──
        // Replay all updates and CLRs from the checkpoint forward.
        for rec in &self.records {
            if rec.lsn < start_lsn {
                continue;
            }
            match &rec.record_type {
                LogRecordType::Update {
                    page_id,
                    offset,
                    new_value,
                    ..
                } => {
                    let page = pages
                        .entry(*page_id)
                        .or_insert_with(|| PageState::new(*page_id));
                    page.data.insert(*offset, new_value.clone());
                    page.page_lsn = rec.lsn;
                    redo_count += 1;
                }
                LogRecordType::Compensate {
                    page_id,
                    offset,
                    undo_value,
                    ..
                } => {
                    let page = pages
                        .entry(*page_id)
                        .or_insert_with(|| PageState::new(*page_id));
                    page.data.insert(*offset, undo_value.clone());
                    page.page_lsn = rec.lsn;
                    redo_count += 1;
                }
                _ => {}
            }
        }

        // ── Undo phase ──
        // Find transactions that are active (not committed, not aborted with end).
        let committed: HashSet<u64> = self
            .records
            .iter()
            .filter(|r| matches!(r.record_type, LogRecordType::Commit))
            .map(|r| r.tx_id)
            .collect();

        let ended: HashSet<u64> = self
            .records
            .iter()
            .filter(|r| matches!(r.record_type, LogRecordType::End))
            .map(|r| r.tx_id)
            .collect();

        let mut to_undo: Vec<u64> = self
            .transactions
            .values()
            .filter(|tx| !committed.contains(&tx.tx_id) && !ended.contains(&tx.tx_id))
            .map(|tx| tx.tx_id)
            .collect();
        to_undo.sort();

        let mut rolled_back = Vec::new();

        for tx_id in &to_undo {
            // Undo updates in reverse order.
            let updates: Vec<LogRecord> = self
                .records
                .iter()
                .filter(|r| r.tx_id == *tx_id)
                .filter(|r| matches!(r.record_type, LogRecordType::Update { .. }))
                .cloned()
                .collect();

            for rec in updates.iter().rev() {
                if let LogRecordType::Update {
                    page_id,
                    offset,
                    old_value,
                    ..
                } = &rec.record_type
                {
                    let page = pages
                        .entry(*page_id)
                        .or_insert_with(|| PageState::new(*page_id));
                    page.data.insert(*offset, old_value.clone());
                    undo_count += 1;
                }
            }
            rolled_back.push(*tx_id);
        }

        RecoveryResult {
            redo_count,
            undo_count,
            rolled_back,
            page_states: pages,
        }
    }

    /// Get log statistics.
    pub fn stats(&self) -> LogStats {
        let mut begin_count = 0;
        let mut update_count = 0;
        let mut commit_count = 0;
        let mut abort_count = 0;
        let mut checkpoint_count = 0;
        let mut compensate_count = 0;
        let mut end_count = 0;

        for rec in &self.records {
            match &rec.record_type {
                LogRecordType::Begin => begin_count += 1,
                LogRecordType::Update { .. } => update_count += 1,
                LogRecordType::Commit => commit_count += 1,
                LogRecordType::Abort => abort_count += 1,
                LogRecordType::Checkpoint { .. } => checkpoint_count += 1,
                LogRecordType::Compensate { .. } => compensate_count += 1,
                LogRecordType::End => end_count += 1,
            }
        }

        let active_tx_count = self
            .transactions
            .values()
            .filter(|tx| tx.state == TxState::Active)
            .count();

        LogStats {
            total_records: self.records.len(),
            begin_count,
            update_count,
            commit_count,
            abort_count,
            checkpoint_count,
            compensate_count,
            end_count,
            active_tx_count,
            next_lsn: self.next_lsn,
            next_tx_id: self.next_tx_id,
            truncated_below: self.truncated_below,
        }
    }

    // ── Private helpers ──

    fn append_record(&mut self, tx_id: u64, record_type: LogRecordType) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        let prev_lsn = if tx_id > 0 {
            self.transactions
                .get(&tx_id)
                .map_or(0, |tx| tx.last_lsn)
        } else {
            0
        };
        self.records.push(LogRecord {
            lsn,
            tx_id,
            prev_lsn,
            record_type,
        });
        lsn
    }

    fn check_active(&self, tx_id: u64) -> Result<(), TxLogError> {
        match self.transactions.get(&tx_id) {
            None => Err(TxLogError::TxNotFound(tx_id)),
            Some(tx) if tx.state != TxState::Active => Err(TxLogError::TxFinished(tx_id)),
            _ => Ok(()),
        }
    }
}

impl Default for TransactionLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_log_empty() {
        let log = TransactionLog::new();
        let stats = log.stats();
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.active_tx_count, 0);
    }

    #[test]
    fn begin_creates_record() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        assert_eq!(tx, 1);
        let stats = log.stats();
        assert_eq!(stats.begin_count, 1);
        assert_eq!(stats.active_tx_count, 1);
    }

    #[test]
    fn write_update() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        let lsn = log
            .write_update(tx, 100, 0, vec![0, 0], vec![1, 2])
            .unwrap();
        assert!(lsn > 0);
        assert_eq!(log.stats().update_count, 1);
    }

    #[test]
    fn commit_tx() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.write_update(tx, 1, 0, vec![0], vec![1]).unwrap();
        let lsn = log.commit_tx(tx).unwrap();
        assert!(lsn > 0);
        let state = log.get_transaction(tx).unwrap();
        assert_eq!(state.state, TxState::Committed);
    }

    #[test]
    fn commit_writes_end() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.commit_tx(tx).unwrap();
        assert_eq!(log.stats().end_count, 1);
    }

    #[test]
    fn abort_generates_clrs() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.write_update(tx, 1, 0, vec![10], vec![20]).unwrap();
        log.write_update(tx, 2, 4, vec![30], vec![40]).unwrap();
        let clrs = log.abort_tx(tx).unwrap();
        assert_eq!(clrs.len(), 2);
        assert_eq!(log.stats().compensate_count, 2);
        assert_eq!(log.stats().abort_count, 1);
    }

    #[test]
    fn abort_sets_state() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.abort_tx(tx).unwrap();
        let state = log.get_transaction(tx).unwrap();
        assert_eq!(state.state, TxState::Aborted);
    }

    #[test]
    fn double_commit_fails() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.commit_tx(tx).unwrap();
        let err = log.commit_tx(tx).unwrap_err();
        assert_eq!(err, TxLogError::TxFinished(tx));
    }

    #[test]
    fn update_on_missing_tx_fails() {
        let mut log = TransactionLog::new();
        let err = log.write_update(999, 1, 0, vec![], vec![]).unwrap_err();
        assert_eq!(err, TxLogError::TxNotFound(999));
    }

    #[test]
    fn checkpoint_records_active() {
        let mut log = TransactionLog::new();
        let tx1 = log.begin_tx();
        let _tx2 = log.begin_tx();
        log.commit_tx(tx1).unwrap();
        let ckpt_lsn = log.write_checkpoint();
        let rec = log.get_record(ckpt_lsn).unwrap();
        if let LogRecordType::Checkpoint { active_txns, .. } = &rec.record_type {
            // Only tx2 should be active.
            assert_eq!(active_txns.len(), 1);
        } else {
            panic!("expected checkpoint record");
        }
    }

    #[test]
    fn truncate_removes_old_records() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.write_update(tx, 1, 0, vec![0], vec![1]).unwrap();
        log.write_update(tx, 2, 0, vec![0], vec![2]).unwrap();
        let total_before = log.all_records().len();
        // Truncate records below LSN 3.
        log.truncate(3);
        assert!(log.all_records().len() < total_before);
        assert_eq!(log.stats().truncated_below, 3);
    }

    #[test]
    fn tx_records_returns_only_for_tx() {
        let mut log = TransactionLog::new();
        let tx1 = log.begin_tx();
        let tx2 = log.begin_tx();
        log.write_update(tx1, 1, 0, vec![0], vec![1]).unwrap();
        log.write_update(tx2, 2, 0, vec![0], vec![2]).unwrap();
        let recs1 = log.tx_records(tx1);
        assert!(recs1.iter().all(|r| r.tx_id == tx1));
    }

    #[test]
    fn recovery_redo_committed() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.write_update(tx, 1, 0, vec![0], vec![42]).unwrap();
        log.commit_tx(tx).unwrap();
        let result = log.recover();
        assert!(result.redo_count > 0);
        assert_eq!(result.undo_count, 0);
        let page = result.page_states.get(&1).unwrap();
        assert_eq!(page.data.get(&0), Some(&vec![42]));
    }

    #[test]
    fn recovery_undoes_uncommitted() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.write_update(tx, 1, 0, vec![0], vec![42]).unwrap();
        // Don't commit — leave active.
        let result = log.recover();
        assert!(result.undo_count > 0);
        assert!(result.rolled_back.contains(&tx));
        let page = result.page_states.get(&1).unwrap();
        // After undo, should have old_value.
        assert_eq!(page.data.get(&0), Some(&vec![0]));
    }

    #[test]
    fn recovery_with_checkpoint() {
        let mut log = TransactionLog::new();
        let tx1 = log.begin_tx();
        log.write_update(tx1, 1, 0, vec![0], vec![1]).unwrap();
        log.commit_tx(tx1).unwrap();
        log.write_checkpoint();
        let tx2 = log.begin_tx();
        log.write_update(tx2, 2, 0, vec![0], vec![2]).unwrap();
        log.commit_tx(tx2).unwrap();
        let result = log.recover();
        assert!(result.page_states.contains_key(&2));
    }

    #[test]
    fn lsn_increases_monotonically() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        let lsn1 = log.write_update(tx, 1, 0, vec![], vec![]).unwrap();
        let lsn2 = log.write_update(tx, 2, 0, vec![], vec![]).unwrap();
        assert!(lsn2 > lsn1);
    }

    #[test]
    fn multiple_txns_interleaved() {
        let mut log = TransactionLog::new();
        let tx1 = log.begin_tx();
        let tx2 = log.begin_tx();
        log.write_update(tx1, 1, 0, vec![0], vec![1]).unwrap();
        log.write_update(tx2, 2, 0, vec![0], vec![2]).unwrap();
        log.commit_tx(tx1).unwrap();
        log.abort_tx(tx2).unwrap();

        let stats = log.stats();
        assert_eq!(stats.commit_count, 1);
        assert_eq!(stats.abort_count, 1);
    }

    #[test]
    fn stats_comprehensive() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        log.write_update(tx, 1, 0, vec![], vec![]).unwrap();
        log.commit_tx(tx).unwrap();
        log.write_checkpoint();

        let stats = log.stats();
        assert_eq!(stats.begin_count, 1);
        assert_eq!(stats.update_count, 1);
        assert_eq!(stats.commit_count, 1);
        assert_eq!(stats.checkpoint_count, 1);
        assert_eq!(stats.end_count, 1);
        assert_eq!(stats.active_tx_count, 0);
    }

    #[test]
    fn prev_lsn_chain() {
        let mut log = TransactionLog::new();
        let tx = log.begin_tx();
        let lsn1 = log.write_update(tx, 1, 0, vec![], vec![]).unwrap();
        let lsn2 = log.write_update(tx, 2, 0, vec![], vec![]).unwrap();
        let rec2 = log.get_record(lsn2).unwrap();
        assert_eq!(rec2.prev_lsn, lsn1);
    }
}
