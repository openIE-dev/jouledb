//! # Two-Phase Commit (2PC) Protocol Implementation
//!
//! A comprehensive implementation of the Two-Phase Commit protocol for distributed
//! transactions across multiple database nodes.
//!
//! ## Overview
//!
//! Two-Phase Commit ensures atomic commitment of distributed transactions:
//! - **Phase 1 (Prepare)**: Coordinator asks all participants to prepare
//! - **Phase 2 (Commit/Abort)**: Coordinator tells all participants to commit or abort
//!
//! ## Features
//!
//! - Transaction coordinator managing distributed transactions
//! - Transaction participant responding to prepare/commit/abort
//! - Recovery log for coordinator and participant failures
//! - Timeout handling with presumed abort semantics
//! - Persistent transaction state for crash recovery
//!
//! ## Safety Guarantees
//!
//! - **Atomicity**: All participants commit or all abort
//! - **Durability**: Decisions are logged before acknowledgment
//! - **Recovery**: Transactions can be recovered after crashes

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc, oneshot};
// TableStorage trait import needed for AmorphicParticipantStorage (insert/delete methods)
use joule_db_query::executor::TableStorage as _;

// ============================================================================
// Constants
// ============================================================================

/// Default prepare timeout (waiting for votes)
const DEFAULT_PREPARE_TIMEOUT_MS: u64 = 5000;

/// Default commit timeout (waiting for acks)
const DEFAULT_COMMIT_TIMEOUT_MS: u64 = 10000;

/// Default participant timeout (waiting for decision)
const DEFAULT_PARTICIPANT_TIMEOUT_MS: u64 = 30000;

/// Maximum retries for commit/abort messages
const MAX_COMMIT_RETRIES: u32 = 10;

/// Retry interval for commit/abort messages
const COMMIT_RETRY_INTERVAL_MS: u64 = 1000;

// ============================================================================
// Types and Identifiers
// ============================================================================

/// Unique identifier for a transaction
pub type TransactionId = String;

/// Unique identifier for a participant node
pub type ParticipantId = String;

/// Sequence number for log entries
pub type LogSequenceNumber = u64;

// ============================================================================
// Transaction State Machine
// ============================================================================

/// Transaction state following the 2PC protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    /// Transaction has been initiated but not yet prepared
    Started,
    /// Coordinator is collecting votes from participants
    Preparing,
    /// All participants have voted to commit (participant-side prepared state)
    Prepared,
    /// Coordinator has decided to commit, sending commit messages
    Committing,
    /// Transaction has been successfully committed
    Committed,
    /// Coordinator has decided to abort, sending abort messages
    Aborting,
    /// Transaction has been aborted
    Aborted,
    /// Unknown state (for recovery scenarios)
    Unknown,
}

impl TransactionState {
    /// Check if the transaction is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Committed | Self::Aborted)
    }

    /// Check if the transaction can be safely aborted
    pub fn can_abort(&self) -> bool {
        matches!(
            self,
            Self::Started | Self::Preparing | Self::Prepared | Self::Aborting
        )
    }

    /// Check if the transaction has been decided (commit or abort)
    pub fn is_decided(&self) -> bool {
        matches!(
            self,
            Self::Committing | Self::Committed | Self::Aborting | Self::Aborted
        )
    }
}

impl std::fmt::Display for TransactionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Started => write!(f, "Started"),
            Self::Preparing => write!(f, "Preparing"),
            Self::Prepared => write!(f, "Prepared"),
            Self::Committing => write!(f, "Committing"),
            Self::Committed => write!(f, "Committed"),
            Self::Aborting => write!(f, "Aborting"),
            Self::Aborted => write!(f, "Aborted"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ============================================================================
// Vote Types
// ============================================================================

/// Vote from a participant in response to Prepare
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vote {
    /// Participant is ready to commit
    Commit,
    /// Participant cannot commit (must abort)
    Abort,
}

// ============================================================================
// 2PC Messages
// ============================================================================

/// Messages exchanged in the 2PC protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TwoPhaseMessage {
    /// Phase 1: Coordinator asks participant to prepare
    Prepare {
        transaction_id: TransactionId,
        coordinator_id: ParticipantId,
        participants: Vec<ParticipantId>,
        operations: Vec<TransactionOperation>,
        timestamp: u64,
    },
    /// Phase 1: Participant's vote in response to Prepare
    Vote {
        transaction_id: TransactionId,
        participant_id: ParticipantId,
        vote: Vote,
        prepared_lsn: Option<LogSequenceNumber>,
    },
    /// Phase 2: Coordinator tells participant to commit
    Commit {
        transaction_id: TransactionId,
        coordinator_id: ParticipantId,
        commit_timestamp: u64,
    },
    /// Phase 2: Coordinator tells participant to abort
    Abort {
        transaction_id: TransactionId,
        coordinator_id: ParticipantId,
        reason: String,
    },
    /// Acknowledgment from participant after commit/abort
    Ack {
        transaction_id: TransactionId,
        participant_id: ParticipantId,
        success: bool,
        error: Option<String>,
    },
    /// Query transaction status (for recovery)
    QueryStatus {
        transaction_id: TransactionId,
        requester_id: ParticipantId,
    },
    /// Response to status query
    StatusResponse {
        transaction_id: TransactionId,
        state: TransactionState,
        responder_id: ParticipantId,
    },
}

impl TwoPhaseMessage {
    /// Get the transaction ID from any message
    pub fn transaction_id(&self) -> &TransactionId {
        match self {
            Self::Prepare { transaction_id, .. } => transaction_id,
            Self::Vote { transaction_id, .. } => transaction_id,
            Self::Commit { transaction_id, .. } => transaction_id,
            Self::Abort { transaction_id, .. } => transaction_id,
            Self::Ack { transaction_id, .. } => transaction_id,
            Self::QueryStatus { transaction_id, .. } => transaction_id,
            Self::StatusResponse { transaction_id, .. } => transaction_id,
        }
    }
}

// ============================================================================
// Transaction Operations
// ============================================================================

/// Operations within a distributed transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOperation {
    /// Put a key-value pair
    Put { key: Vec<u8>, value: Vec<u8> },
    /// Delete a key
    Delete { key: Vec<u8> },
    /// Conditional put (CAS-like)
    ConditionalPut {
        key: Vec<u8>,
        expected_value: Option<Vec<u8>>,
        new_value: Vec<u8>,
    },
}

impl TransactionOperation {
    /// Get the key affected by this operation
    pub fn key(&self) -> &[u8] {
        match self {
            Self::Put { key, .. } => key,
            Self::Delete { key } => key,
            Self::ConditionalPut { key, .. } => key,
        }
    }
}

// ============================================================================
// 2PC Errors
// ============================================================================

/// Errors that can occur during 2PC operations
#[derive(Debug, Clone)]
pub enum TwoPhaseError {
    /// Transaction not found
    TransactionNotFound(TransactionId),
    /// Transaction already exists
    TransactionExists(TransactionId),
    /// Invalid state transition
    InvalidStateTransition {
        transaction_id: TransactionId,
        from: TransactionState,
        to: TransactionState,
    },
    /// Timeout waiting for votes or acks
    Timeout {
        transaction_id: TransactionId,
        phase: String,
    },
    /// Participant voted to abort
    ParticipantAborted {
        transaction_id: TransactionId,
        participant_id: ParticipantId,
        reason: String,
    },
    /// Communication failure
    CommunicationFailure {
        participant_id: ParticipantId,
        error: String,
    },
    /// Recovery failure
    RecoveryFailure(String),
    /// Log write failure
    LogWriteFailure(String),
    /// Coordinator not found (for participants)
    CoordinatorNotFound(TransactionId),
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for TwoPhaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TransactionNotFound(id) => write!(f, "Transaction not found: {}", id),
            Self::TransactionExists(id) => write!(f, "Transaction already exists: {}", id),
            Self::InvalidStateTransition {
                transaction_id,
                from,
                to,
            } => {
                write!(
                    f,
                    "Invalid state transition for {}: {} -> {}",
                    transaction_id, from, to
                )
            }
            Self::Timeout {
                transaction_id,
                phase,
            } => {
                write!(
                    f,
                    "Timeout in {} phase for transaction {}",
                    phase, transaction_id
                )
            }
            Self::ParticipantAborted {
                transaction_id,
                participant_id,
                reason,
            } => {
                write!(
                    f,
                    "Participant {} aborted transaction {}: {}",
                    participant_id, transaction_id, reason
                )
            }
            Self::CommunicationFailure {
                participant_id,
                error,
            } => {
                write!(
                    f,
                    "Communication failure with {}: {}",
                    participant_id, error
                )
            }
            Self::RecoveryFailure(msg) => write!(f, "Recovery failure: {}", msg),
            Self::LogWriteFailure(msg) => write!(f, "Log write failure: {}", msg),
            Self::CoordinatorNotFound(id) => {
                write!(f, "Coordinator not found for transaction {}", id)
            }
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for TwoPhaseError {}

/// Result type for 2PC operations
pub type TwoPhaseResult<T> = Result<T, TwoPhaseError>;

// ============================================================================
// Recovery Log
// ============================================================================

/// Entry in the recovery log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryLogEntry {
    /// Log sequence number
    pub lsn: LogSequenceNumber,
    /// Timestamp of the entry
    pub timestamp: u64,
    /// Transaction ID
    pub transaction_id: TransactionId,
    /// Type of log record
    pub record_type: RecoveryLogRecordType,
}

/// Types of recovery log records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryLogRecordType {
    /// Transaction started (coordinator)
    TransactionStart {
        participants: Vec<ParticipantId>,
        operations: Vec<TransactionOperation>,
    },
    /// Prepare sent to all participants (coordinator)
    PrepareSent,
    /// Vote received from participant (coordinator)
    VoteReceived {
        participant_id: ParticipantId,
        vote: Vote,
    },
    /// Decision made (coordinator) - CRITICAL: must be durable before sending commit/abort
    Decision { commit: bool },
    /// Commit/Abort sent to participant (coordinator)
    DecisionSent { participant_id: ParticipantId },
    /// Ack received from participant (coordinator)
    AckReceived { participant_id: ParticipantId },
    /// Transaction complete (coordinator)
    TransactionComplete,
    /// Prepare received (participant)
    PrepareReceived { coordinator_id: ParticipantId },
    /// Vote sent (participant) - CRITICAL: participant must write before voting commit
    VoteSent { vote: Vote },
    /// Decision received (participant)
    DecisionReceived { commit: bool },
    /// Applied to local state (participant)
    Applied,
}

/// Recovery log for persistent transaction state.
///
/// When created with [`RecoveryLog::new_durable`], entries are persisted to a
/// JSON-lines file. Each line is one [`RecoveryLogEntry`] serialized as JSON.
/// `sync()` calls `fsync()` on the underlying file to guarantee durability.
///
/// When created with [`RecoveryLog::new`], entries are kept in memory only
/// (suitable for tests or ephemeral coordinators).
pub struct RecoveryLog {
    /// In-memory log entries (also serves as read cache for durable mode)
    entries: RwLock<Vec<RecoveryLogEntry>>,
    /// Next LSN to assign
    next_lsn: AtomicU64,
    /// Synced to disk LSN
    synced_lsn: AtomicU64,
    /// Durable file handle (None = in-memory only)
    log_file: Option<std::sync::Mutex<std::fs::File>>,
    /// Path to log file (for rewrite on truncate)
    log_path: Option<std::path::PathBuf>,
}

impl RecoveryLog {
    /// Create a new in-memory-only recovery log (for tests).
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            next_lsn: AtomicU64::new(1),
            synced_lsn: AtomicU64::new(0),
            log_file: None,
            log_path: None,
        }
    }

    /// Create a durable recovery log backed by a file.
    ///
    /// If the file already exists, existing entries are loaded and the LSN
    /// counter is advanced past the highest existing LSN. New entries are
    /// appended to the file.
    pub fn new_durable(path: &std::path::Path) -> Result<Self, String> {
        use std::io::{BufRead, Write};

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create log directory: {e}"))?;
        }

        // Load existing entries
        let mut entries = Vec::new();
        let mut max_lsn: u64 = 0;
        if path.exists() {
            let file = std::fs::File::open(path)
                .map_err(|e| format!("Failed to open recovery log: {e}"))?;
            let reader = std::io::BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read log line: {e}"))?;
                if line.trim().is_empty() {
                    continue;
                }
                let entry: RecoveryLogEntry = serde_json::from_str(&line)
                    .map_err(|e| format!("Failed to parse log entry: {e}"))?;
                if entry.lsn > max_lsn {
                    max_lsn = entry.lsn;
                }
                entries.push(entry);
            }
        }

        // Open file in append mode for new writes
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("Failed to open recovery log for append: {e}"))?;

        Ok(Self {
            entries: RwLock::new(entries),
            next_lsn: AtomicU64::new(max_lsn + 1),
            synced_lsn: AtomicU64::new(max_lsn),
            log_file: Some(std::sync::Mutex::new(file)),
            log_path: Some(path.to_path_buf()),
        })
    }

    /// Persist a single entry to the durable log file (if configured).
    fn persist_entry(&self, entry: &RecoveryLogEntry) -> TwoPhaseResult<()> {
        if let Some(ref file_mutex) = self.log_file {
            use std::io::Write;
            let mut file = file_mutex.lock().map_err(|e| {
                TwoPhaseError::LogWriteFailure(format!("Log file lock poisoned: {e}"))
            })?;
            let json = serde_json::to_string(entry).map_err(|e| {
                TwoPhaseError::LogWriteFailure(format!("Failed to serialize log entry: {e}"))
            })?;
            writeln!(file, "{}", json).map_err(|e| {
                TwoPhaseError::LogWriteFailure(format!("Failed to write log entry: {e}"))
            })?;
        }
        Ok(())
    }

    /// Append an entry to the log
    pub async fn append(
        &self,
        transaction_id: TransactionId,
        record_type: RecoveryLogRecordType,
    ) -> TwoPhaseResult<LogSequenceNumber> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let entry = RecoveryLogEntry {
            lsn,
            timestamp,
            transaction_id,
            record_type,
        };

        self.persist_entry(&entry)?;

        let mut entries = self.entries.write().await;
        entries.push(entry);

        Ok(lsn)
    }

    /// Force sync to disk (fsync).
    pub async fn sync(&self) -> TwoPhaseResult<()> {
        if let Some(ref file_mutex) = self.log_file {
            let file = file_mutex.lock().map_err(|e| {
                TwoPhaseError::LogWriteFailure(format!("Log file lock poisoned: {e}"))
            })?;
            file.sync_all()
                .map_err(|e| TwoPhaseError::LogWriteFailure(format!("fsync failed: {e}")))?;
        }
        let current_lsn = self.next_lsn.load(Ordering::SeqCst).saturating_sub(1);
        self.synced_lsn.store(current_lsn, Ordering::SeqCst);
        Ok(())
    }

    /// Append and sync (for critical records like Decision)
    pub async fn append_sync(
        &self,
        transaction_id: TransactionId,
        record_type: RecoveryLogRecordType,
    ) -> TwoPhaseResult<LogSequenceNumber> {
        let lsn = self.append(transaction_id, record_type).await?;
        self.sync().await?;
        Ok(lsn)
    }

    /// Get all entries for a transaction
    pub async fn get_transaction_entries(
        &self,
        transaction_id: &TransactionId,
    ) -> Vec<RecoveryLogEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| &e.transaction_id == transaction_id)
            .cloned()
            .collect()
    }

    /// Get all incomplete transactions (for recovery)
    pub async fn get_incomplete_transactions(&self) -> Vec<(TransactionId, Vec<RecoveryLogEntry>)> {
        let entries = self.entries.read().await;
        let mut txn_entries: HashMap<TransactionId, Vec<RecoveryLogEntry>> = HashMap::new();

        for entry in entries.iter() {
            txn_entries
                .entry(entry.transaction_id.clone())
                .or_default()
                .push(entry.clone());
        }

        // Filter out completed transactions
        txn_entries
            .into_iter()
            .filter(|(_, entries)| {
                !entries
                    .iter()
                    .any(|e| matches!(e.record_type, RecoveryLogRecordType::TransactionComplete))
            })
            .collect()
    }

    /// Truncate log entries before the given LSN (garbage collection).
    ///
    /// In durable mode this rewrites the log file with only retained entries
    /// using atomic write-temp-then-rename.
    pub async fn truncate_before(&self, lsn: LogSequenceNumber) {
        let mut entries = self.entries.write().await;
        entries.retain(|e| e.lsn >= lsn);

        // Rewrite the log file if durable
        if let (Some(file_mutex), Some(log_path)) = (&self.log_file, &self.log_path) {
            use std::io::Write;
            let tmp_path = log_path.with_extension("tmp");
            if let Ok(mut tmp_file) = std::fs::File::create(&tmp_path) {
                for entry in entries.iter() {
                    if let Ok(json) = serde_json::to_string(entry) {
                        let _ = writeln!(tmp_file, "{}", json);
                    }
                }
                let _ = tmp_file.sync_all();
                // Atomic rename
                if std::fs::rename(&tmp_path, log_path).is_ok() {
                    // Reopen the file for future appends
                    if let Ok(new_file) = std::fs::OpenOptions::new().append(true).open(log_path) {
                        if let Ok(mut guard) = file_mutex.lock() {
                            *guard = new_file;
                        }
                    }
                }
            }
        }
    }

    /// Get the current synced LSN
    pub fn synced_lsn(&self) -> LogSequenceNumber {
        self.synced_lsn.load(Ordering::SeqCst)
    }
}

impl Default for RecoveryLog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Transaction Coordinator State
// ============================================================================

/// State tracked by coordinator for each transaction
#[derive(Debug)]
struct CoordinatorTransaction {
    /// Transaction ID
    id: TransactionId,
    /// Current state
    state: TransactionState,
    /// Participating nodes
    participants: Vec<ParticipantId>,
    /// Operations in this transaction
    operations: Vec<TransactionOperation>,
    /// Votes received from participants
    votes: HashMap<ParticipantId, Vote>,
    /// Acks received from participants
    acks: HashSet<ParticipantId>,
    /// Creation time
    created_at: Instant,
    /// Last activity time
    last_activity: Instant,
    /// Number of commit/abort retries
    retries: u32,
    /// Completion notifier
    completion_tx: Option<oneshot::Sender<TwoPhaseResult<TransactionState>>>,
}

impl CoordinatorTransaction {
    fn new(
        id: TransactionId,
        participants: Vec<ParticipantId>,
        operations: Vec<TransactionOperation>,
        completion_tx: Option<oneshot::Sender<TwoPhaseResult<TransactionState>>>,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            state: TransactionState::Started,
            participants,
            operations,
            votes: HashMap::new(),
            acks: HashSet::new(),
            created_at: now,
            last_activity: now,
            retries: 0,
            completion_tx,
        }
    }

    fn all_votes_received(&self) -> bool {
        self.votes.len() == self.participants.len()
    }

    fn all_acks_received(&self) -> bool {
        self.acks.len() == self.participants.len()
    }

    fn should_commit(&self) -> bool {
        self.all_votes_received() && self.votes.values().all(|v| *v == Vote::Commit)
    }
}

// ============================================================================
// Transaction Coordinator
// ============================================================================

/// Configuration for the coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Node ID of this coordinator
    pub node_id: ParticipantId,
    /// Timeout for prepare phase
    pub prepare_timeout: Duration,
    /// Timeout for commit phase
    pub commit_timeout: Duration,
    /// Maximum retries for commit/abort
    pub max_retries: u32,
    /// Retry interval for commit/abort
    pub retry_interval: Duration,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            prepare_timeout: Duration::from_millis(DEFAULT_PREPARE_TIMEOUT_MS),
            commit_timeout: Duration::from_millis(DEFAULT_COMMIT_TIMEOUT_MS),
            max_retries: MAX_COMMIT_RETRIES,
            retry_interval: Duration::from_millis(COMMIT_RETRY_INTERVAL_MS),
        }
    }
}

/// Transaction Coordinator for managing distributed transactions
pub struct TransactionCoordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Active transactions
    transactions: RwLock<HashMap<TransactionId, CoordinatorTransaction>>,
    /// Recovery log
    recovery_log: Arc<RecoveryLog>,
    /// Message sender for outgoing messages
    message_tx: mpsc::Sender<(ParticipantId, TwoPhaseMessage)>,
    /// Transaction ID counter
    txn_counter: AtomicU64,
    /// Statistics
    stats: CoordinatorStats,
}

/// Coordinator statistics
#[derive(Debug, Default)]
pub struct CoordinatorStats {
    pub transactions_started: AtomicU64,
    pub transactions_committed: AtomicU64,
    pub transactions_aborted: AtomicU64,
    pub prepare_timeouts: AtomicU64,
    pub commit_timeouts: AtomicU64,
    pub retries_total: AtomicU64,
}

impl TransactionCoordinator {
    /// Create a new transaction coordinator
    pub fn new(
        config: CoordinatorConfig,
        message_tx: mpsc::Sender<(ParticipantId, TwoPhaseMessage)>,
    ) -> Self {
        Self {
            config,
            transactions: RwLock::new(HashMap::new()),
            recovery_log: Arc::new(RecoveryLog::new()),
            message_tx,
            txn_counter: AtomicU64::new(1),
            stats: CoordinatorStats::default(),
        }
    }

    /// Generate a new transaction ID
    fn generate_txn_id(&self) -> TransactionId {
        let counter = self.txn_counter.fetch_add(1, Ordering::SeqCst);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        format!("{}:{}:{}", self.config.node_id, timestamp, counter)
    }

    /// Begin a new distributed transaction
    pub async fn begin_transaction(
        &self,
        participants: Vec<ParticipantId>,
        operations: Vec<TransactionOperation>,
    ) -> TwoPhaseResult<oneshot::Receiver<TwoPhaseResult<TransactionState>>> {
        let txn_id = self.generate_txn_id();
        let (completion_tx, completion_rx) = oneshot::channel();

        // Log transaction start
        self.recovery_log
            .append_sync(
                txn_id.clone(),
                RecoveryLogRecordType::TransactionStart {
                    participants: participants.clone(),
                    operations: operations.clone(),
                },
            )
            .await?;

        let txn = CoordinatorTransaction::new(
            txn_id.clone(),
            participants.clone(),
            operations.clone(),
            Some(completion_tx),
        );

        {
            let mut transactions = self.transactions.write().await;
            if transactions.contains_key(&txn_id) {
                return Err(TwoPhaseError::TransactionExists(txn_id));
            }
            transactions.insert(txn_id.clone(), txn);
        }

        self.stats
            .transactions_started
            .fetch_add(1, Ordering::Relaxed);

        // Start prepare phase
        self.start_prepare_phase(&txn_id).await?;

        Ok(completion_rx)
    }

    /// Start the prepare phase for a transaction
    async fn start_prepare_phase(&self, txn_id: &TransactionId) -> TwoPhaseResult<()> {
        let (participants, operations) = {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(txn_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(txn_id.clone()))?;

            txn.state = TransactionState::Preparing;
            txn.last_activity = Instant::now();

            (txn.participants.clone(), txn.operations.clone())
        };

        // Log prepare sent
        self.recovery_log
            .append(txn_id.clone(), RecoveryLogRecordType::PrepareSent)
            .await?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Send prepare to all participants
        for participant_id in &participants {
            let message = TwoPhaseMessage::Prepare {
                transaction_id: txn_id.clone(),
                coordinator_id: self.config.node_id.clone(),
                participants: participants.clone(),
                operations: operations.clone(),
                timestamp,
            };

            if let Err(e) = self
                .message_tx
                .send((participant_id.clone(), message))
                .await
            {
                tracing::error!("Failed to send prepare to {}: {}", participant_id, e);
            }
        }

        Ok(())
    }

    /// Handle a vote from a participant
    pub async fn handle_vote(&self, message: TwoPhaseMessage) -> TwoPhaseResult<()> {
        let TwoPhaseMessage::Vote {
            transaction_id,
            participant_id,
            vote,
            prepared_lsn: _,
        } = message
        else {
            return Err(TwoPhaseError::Internal("Expected Vote message".into()));
        };

        // Log vote received
        self.recovery_log
            .append(
                transaction_id.clone(),
                RecoveryLogRecordType::VoteReceived {
                    participant_id: participant_id.clone(),
                    vote,
                },
            )
            .await?;

        let should_decide = {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(&transaction_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(transaction_id.clone()))?;

            if txn.state != TransactionState::Preparing {
                return Ok(()); // Duplicate or late vote
            }

            txn.votes.insert(participant_id.clone(), vote);
            txn.last_activity = Instant::now();

            tracing::debug!(
                "Received vote {:?} from {} for transaction {} ({}/{})",
                vote,
                participant_id,
                transaction_id,
                txn.votes.len(),
                txn.participants.len()
            );

            txn.all_votes_received()
        };

        if should_decide {
            self.make_decision(&transaction_id).await?;
        }

        Ok(())
    }

    /// Make commit/abort decision based on votes
    async fn make_decision(&self, txn_id: &TransactionId) -> TwoPhaseResult<()> {
        let (should_commit, participants) = {
            let transactions = self.transactions.read().await;
            let txn = transactions
                .get(txn_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(txn_id.clone()))?;

            (txn.should_commit(), txn.participants.clone())
        };

        // CRITICAL: Log decision before sending to participants
        self.recovery_log
            .append_sync(
                txn_id.clone(),
                RecoveryLogRecordType::Decision {
                    commit: should_commit,
                },
            )
            .await?;

        // Update state
        {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(txn_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(txn_id.clone()))?;

            txn.state = if should_commit {
                TransactionState::Committing
            } else {
                TransactionState::Aborting
            };
            txn.last_activity = Instant::now();
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Send decision to all participants
        for participant_id in &participants {
            let message = if should_commit {
                TwoPhaseMessage::Commit {
                    transaction_id: txn_id.clone(),
                    coordinator_id: self.config.node_id.clone(),
                    commit_timestamp: timestamp,
                }
            } else {
                TwoPhaseMessage::Abort {
                    transaction_id: txn_id.clone(),
                    coordinator_id: self.config.node_id.clone(),
                    reason: "One or more participants voted to abort".into(),
                }
            };

            // Log decision sent
            self.recovery_log
                .append(
                    txn_id.clone(),
                    RecoveryLogRecordType::DecisionSent {
                        participant_id: participant_id.clone(),
                    },
                )
                .await?;

            if let Err(e) = self
                .message_tx
                .send((participant_id.clone(), message))
                .await
            {
                tracing::error!("Failed to send decision to {}: {}", participant_id, e);
            }
        }

        tracing::info!(
            "Transaction {} decided to {}",
            txn_id,
            if should_commit { "commit" } else { "abort" }
        );

        Ok(())
    }

    /// Handle an acknowledgment from a participant
    pub async fn handle_ack(&self, message: TwoPhaseMessage) -> TwoPhaseResult<()> {
        let TwoPhaseMessage::Ack {
            transaction_id,
            participant_id,
            success,
            error,
        } = message
        else {
            return Err(TwoPhaseError::Internal("Expected Ack message".into()));
        };

        if !success {
            tracing::error!(
                "Participant {} failed to complete transaction {}: {:?}",
                participant_id,
                transaction_id,
                error
            );
        }

        // Log ack received
        self.recovery_log
            .append(
                transaction_id.clone(),
                RecoveryLogRecordType::AckReceived {
                    participant_id: participant_id.clone(),
                },
            )
            .await?;

        let (should_complete, final_state) = {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(&transaction_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(transaction_id.clone()))?;

            if !matches!(
                txn.state,
                TransactionState::Committing | TransactionState::Aborting
            ) {
                return Ok(()); // Duplicate or late ack
            }

            txn.acks.insert(participant_id.clone());
            txn.last_activity = Instant::now();

            let all_acked = txn.all_acks_received();
            let final_state = if txn.state == TransactionState::Committing {
                TransactionState::Committed
            } else {
                TransactionState::Aborted
            };

            (all_acked, final_state)
        };

        if should_complete {
            self.complete_transaction(&transaction_id, final_state)
                .await?;
        }

        Ok(())
    }

    /// Complete a transaction
    async fn complete_transaction(
        &self,
        txn_id: &TransactionId,
        final_state: TransactionState,
    ) -> TwoPhaseResult<()> {
        // Log completion
        self.recovery_log
            .append_sync(txn_id.clone(), RecoveryLogRecordType::TransactionComplete)
            .await?;

        let completion_tx = {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(txn_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(txn_id.clone()))?;

            txn.state = final_state;
            txn.completion_tx.take()
        };

        // Update stats
        if final_state == TransactionState::Committed {
            self.stats
                .transactions_committed
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats
                .transactions_aborted
                .fetch_add(1, Ordering::Relaxed);
        }

        // Notify waiters
        if let Some(tx) = completion_tx {
            let _ = tx.send(Ok(final_state));
        }

        tracing::info!(
            "Transaction {} completed with state {}",
            txn_id,
            final_state
        );

        Ok(())
    }

    /// Handle status query from a recovering participant
    pub async fn handle_status_query(
        &self,
        message: TwoPhaseMessage,
    ) -> TwoPhaseResult<TwoPhaseMessage> {
        let TwoPhaseMessage::QueryStatus {
            transaction_id,
            requester_id: _,
        } = message
        else {
            return Err(TwoPhaseError::Internal(
                "Expected QueryStatus message".into(),
            ));
        };

        let state = {
            let transactions = self.transactions.read().await;
            transactions
                .get(&transaction_id)
                .map(|txn| txn.state)
                .unwrap_or(TransactionState::Unknown)
        };

        Ok(TwoPhaseMessage::StatusResponse {
            transaction_id,
            state,
            responder_id: self.config.node_id.clone(),
        })
    }

    /// Check for timed out transactions (call periodically)
    pub async fn check_timeouts(&self) -> Vec<TransactionId> {
        let mut timed_out = Vec::new();
        let now = Instant::now();

        let mut transactions = self.transactions.write().await;

        for (txn_id, txn) in transactions.iter_mut() {
            match txn.state {
                TransactionState::Preparing => {
                    if now.duration_since(txn.last_activity) > self.config.prepare_timeout {
                        tracing::warn!("Transaction {} timed out in prepare phase", txn_id);
                        timed_out.push(txn_id.clone());
                        self.stats.prepare_timeouts.fetch_add(1, Ordering::Relaxed);
                    }
                }
                TransactionState::Committing | TransactionState::Aborting => {
                    if now.duration_since(txn.last_activity) > self.config.commit_timeout {
                        if txn.retries < self.config.max_retries {
                            txn.retries += 1;
                            txn.last_activity = now;
                            self.stats.retries_total.fetch_add(1, Ordering::Relaxed);
                            // Will retry sending decision
                        } else {
                            tracing::warn!(
                                "Transaction {} timed out in commit/abort phase after {} retries",
                                txn_id,
                                txn.retries
                            );
                            self.stats.commit_timeouts.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                other => {
                    tracing::debug!(
                        "Transaction {} in state {:?} during timeout check, skipping",
                        txn_id,
                        other
                    );
                }
            }
        }

        drop(transactions);

        // Abort timed out transactions in prepare phase (presumed abort)
        for txn_id in &timed_out {
            let _ = self
                .abort_transaction(txn_id, "Prepare phase timeout".into())
                .await;
        }

        timed_out
    }

    /// Abort a transaction (can be called externally or due to timeout)
    pub async fn abort_transaction(
        &self,
        txn_id: &TransactionId,
        reason: String,
    ) -> TwoPhaseResult<()> {
        let participants = {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(txn_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(txn_id.clone()))?;

            if !txn.state.can_abort() {
                return Err(TwoPhaseError::InvalidStateTransition {
                    transaction_id: txn_id.clone(),
                    from: txn.state,
                    to: TransactionState::Aborting,
                });
            }

            txn.state = TransactionState::Aborting;
            txn.last_activity = Instant::now();
            txn.participants.clone()
        };

        // Log decision
        self.recovery_log
            .append_sync(
                txn_id.clone(),
                RecoveryLogRecordType::Decision { commit: false },
            )
            .await?;

        // Send abort to all participants
        for participant_id in &participants {
            let message = TwoPhaseMessage::Abort {
                transaction_id: txn_id.clone(),
                coordinator_id: self.config.node_id.clone(),
                reason: reason.clone(),
            };

            if let Err(e) = self
                .message_tx
                .send((participant_id.clone(), message))
                .await
            {
                tracing::error!("Failed to send abort to {}: {}", participant_id, e);
            }
        }

        Ok(())
    }

    /// Recover incomplete transactions after coordinator restart
    pub async fn recover(&self) -> TwoPhaseResult<Vec<TransactionId>> {
        let incomplete = self.recovery_log.get_incomplete_transactions().await;
        let mut recovered = Vec::new();

        for (txn_id, entries) in incomplete {
            let mut decision: Option<bool> = None;
            let mut participants = Vec::new();
            let mut operations = Vec::new();
            let mut acks = HashSet::new();

            for entry in &entries {
                match &entry.record_type {
                    RecoveryLogRecordType::TransactionStart {
                        participants: p,
                        operations: o,
                    } => {
                        participants = p.clone();
                        operations = o.clone();
                    }
                    RecoveryLogRecordType::Decision { commit } => {
                        decision = Some(*commit);
                    }
                    RecoveryLogRecordType::AckReceived { participant_id } => {
                        acks.insert(participant_id.clone());
                    }
                    other => {
                        tracing::warn!(
                            "Unexpected recovery log record {:?} for txn {} during coordinator recovery",
                            other,
                            txn_id
                        );
                    }
                }
            }

            match decision {
                Some(true) => {
                    // Decision was to commit - resend commit to participants that haven't acked
                    let pending: Vec<_> = participants
                        .iter()
                        .filter(|p| !acks.contains(*p))
                        .cloned()
                        .collect();

                    if pending.is_empty() {
                        // All acked, just complete
                        self.recovery_log
                            .append_sync(txn_id.clone(), RecoveryLogRecordType::TransactionComplete)
                            .await?;
                    } else {
                        // Recreate transaction and resend commits
                        let (completion_tx, _) = oneshot::channel();
                        let mut txn = CoordinatorTransaction::new(
                            txn_id.clone(),
                            participants.clone(),
                            operations,
                            Some(completion_tx),
                        );
                        txn.state = TransactionState::Committing;
                        txn.acks = acks;

                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;

                        for participant_id in &pending {
                            let message = TwoPhaseMessage::Commit {
                                transaction_id: txn_id.clone(),
                                coordinator_id: self.config.node_id.clone(),
                                commit_timestamp: timestamp,
                            };
                            let _ = self
                                .message_tx
                                .send((participant_id.clone(), message))
                                .await;
                        }

                        self.transactions.write().await.insert(txn_id.clone(), txn);
                    }
                }
                Some(false) => {
                    // Decision was to abort - resend abort to participants that haven't acked
                    let pending: Vec<_> = participants
                        .iter()
                        .filter(|p| !acks.contains(*p))
                        .cloned()
                        .collect();

                    if pending.is_empty() {
                        self.recovery_log
                            .append_sync(txn_id.clone(), RecoveryLogRecordType::TransactionComplete)
                            .await?;
                    } else {
                        let (completion_tx, _) = oneshot::channel();
                        let mut txn = CoordinatorTransaction::new(
                            txn_id.clone(),
                            participants.clone(),
                            operations,
                            Some(completion_tx),
                        );
                        txn.state = TransactionState::Aborting;
                        txn.acks = acks;

                        for participant_id in &pending {
                            let message = TwoPhaseMessage::Abort {
                                transaction_id: txn_id.clone(),
                                coordinator_id: self.config.node_id.clone(),
                                reason: "Recovered after coordinator restart".into(),
                            };
                            let _ = self
                                .message_tx
                                .send((participant_id.clone(), message))
                                .await;
                        }

                        self.transactions.write().await.insert(txn_id.clone(), txn);
                    }
                }
                None => {
                    // No decision recorded - presumed abort
                    self.recovery_log
                        .append_sync(
                            txn_id.clone(),
                            RecoveryLogRecordType::Decision { commit: false },
                        )
                        .await?;

                    for participant_id in &participants {
                        let message = TwoPhaseMessage::Abort {
                            transaction_id: txn_id.clone(),
                            coordinator_id: self.config.node_id.clone(),
                            reason: "Presumed abort after coordinator recovery".into(),
                        };
                        let _ = self
                            .message_tx
                            .send((participant_id.clone(), message))
                            .await;
                    }
                }
            }

            recovered.push(txn_id);
        }

        Ok(recovered)
    }

    /// Get transaction state
    pub async fn get_transaction_state(&self, txn_id: &TransactionId) -> Option<TransactionState> {
        let transactions = self.transactions.read().await;
        transactions.get(txn_id).map(|txn| txn.state)
    }

    /// Get coordinator statistics snapshot
    pub fn stats(&self) -> CoordinatorStatsSnapshot {
        CoordinatorStatsSnapshot {
            transactions_started: self.stats.transactions_started.load(Ordering::Relaxed),
            transactions_committed: self.stats.transactions_committed.load(Ordering::Relaxed),
            transactions_aborted: self.stats.transactions_aborted.load(Ordering::Relaxed),
            prepare_timeouts: self.stats.prepare_timeouts.load(Ordering::Relaxed),
            commit_timeouts: self.stats.commit_timeouts.load(Ordering::Relaxed),
            retries_total: self.stats.retries_total.load(Ordering::Relaxed),
        }
    }

    /// Clean up completed transactions (garbage collection)
    pub async fn cleanup_completed(&self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut transactions = self.transactions.write().await;
        let before = transactions.len();

        transactions.retain(|_, txn| {
            if txn.state.is_terminal() {
                now.duration_since(txn.last_activity) < max_age
            } else {
                true
            }
        });

        before - transactions.len()
    }
}

/// Snapshot of coordinator statistics
#[derive(Debug, Clone)]
pub struct CoordinatorStatsSnapshot {
    pub transactions_started: u64,
    pub transactions_committed: u64,
    pub transactions_aborted: u64,
    pub prepare_timeouts: u64,
    pub commit_timeouts: u64,
    pub retries_total: u64,
}

// ============================================================================
// Transaction Participant State
// ============================================================================

/// State tracked by participant for each transaction
#[derive(Debug)]
struct ParticipantTransaction {
    /// Transaction ID
    id: TransactionId,
    /// Current state
    state: TransactionState,
    /// Coordinator ID
    coordinator_id: ParticipantId,
    /// Operations to execute
    operations: Vec<TransactionOperation>,
    /// Prepared LSN (for rollback)
    prepared_lsn: Option<LogSequenceNumber>,
    /// Creation time
    created_at: Instant,
    /// Last activity time
    last_activity: Instant,
    /// Our vote
    vote: Option<Vote>,
}

impl ParticipantTransaction {
    fn new(
        id: TransactionId,
        coordinator_id: ParticipantId,
        operations: Vec<TransactionOperation>,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            state: TransactionState::Started,
            coordinator_id,
            operations,
            prepared_lsn: None,
            created_at: now,
            last_activity: now,
            vote: None,
        }
    }
}

// ============================================================================
// Transaction Participant
// ============================================================================

/// Configuration for the participant
#[derive(Debug, Clone)]
pub struct ParticipantConfig {
    /// Node ID of this participant
    pub node_id: ParticipantId,
    /// Timeout waiting for coordinator decision
    pub decision_timeout: Duration,
}

impl Default for ParticipantConfig {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            decision_timeout: Duration::from_millis(DEFAULT_PARTICIPANT_TIMEOUT_MS),
        }
    }
}

/// Callback trait for participant to interact with local storage
pub trait ParticipantStorage: Send + Sync {
    /// Prepare operations (check constraints, acquire locks)
    fn prepare(&self, operations: &[TransactionOperation]) -> TwoPhaseResult<LogSequenceNumber>;

    /// Commit prepared operations
    fn commit(&self, lsn: LogSequenceNumber) -> TwoPhaseResult<()>;

    /// Rollback/abort prepared operations
    fn rollback(&self, lsn: LogSequenceNumber) -> TwoPhaseResult<()>;
}

/// In-memory storage implementation for testing
pub struct InMemoryStorage {
    data: std::sync::Mutex<HashMap<Vec<u8>, Vec<u8>>>,
    prepared: std::sync::Mutex<HashMap<LogSequenceNumber, Vec<TransactionOperation>>>,
    next_lsn: AtomicU64,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            data: std::sync::Mutex::new(HashMap::new()),
            prepared: std::sync::Mutex::new(HashMap::new()),
            next_lsn: AtomicU64::new(1),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let data = crate::lock_util::mutex_lock(&self.data);
        data.get(key).cloned()
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticipantStorage for InMemoryStorage {
    fn prepare(&self, operations: &[TransactionOperation]) -> TwoPhaseResult<LogSequenceNumber> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);

        // In a real implementation, this would:
        // 1. Acquire locks on affected keys
        // 2. Validate constraints
        // 3. Write to WAL

        let mut prepared = crate::lock_util::mutex_lock(&self.prepared);
        prepared.insert(lsn, operations.to_vec());

        Ok(lsn)
    }

    fn commit(&self, lsn: LogSequenceNumber) -> TwoPhaseResult<()> {
        let mut prepared = crate::lock_util::mutex_lock(&self.prepared);
        if let Some(ops) = prepared.remove(&lsn) {
            let mut data = crate::lock_util::mutex_lock(&self.data);
            for op in ops {
                match op {
                    TransactionOperation::Put { key, value } => {
                        data.insert(key, value);
                    }
                    TransactionOperation::Delete { key } => {
                        data.remove(&key);
                    }
                    TransactionOperation::ConditionalPut { key, new_value, .. } => {
                        data.insert(key, new_value);
                    }
                }
            }
        }

        Ok(())
    }

    fn rollback(&self, lsn: LogSequenceNumber) -> TwoPhaseResult<()> {
        let mut prepared = crate::lock_util::mutex_lock(&self.prepared);
        prepared.remove(&lsn);

        Ok(())
    }
}

/// Transaction Participant for responding to coordinator messages
pub struct TransactionParticipant<S: ParticipantStorage> {
    /// Configuration
    config: ParticipantConfig,
    /// Active transactions
    transactions: RwLock<HashMap<TransactionId, ParticipantTransaction>>,
    /// Recovery log
    recovery_log: Arc<RecoveryLog>,
    /// Local storage
    storage: Arc<S>,
    /// Message sender for outgoing messages
    message_tx: mpsc::Sender<(ParticipantId, TwoPhaseMessage)>,
    /// Statistics
    stats: ParticipantStats,
}

/// Participant statistics
#[derive(Debug, Default)]
pub struct ParticipantStats {
    pub prepares_received: AtomicU64,
    pub votes_commit: AtomicU64,
    pub votes_abort: AtomicU64,
    pub commits_received: AtomicU64,
    pub aborts_received: AtomicU64,
    pub decision_timeouts: AtomicU64,
}

impl<S: ParticipantStorage + 'static> TransactionParticipant<S> {
    /// Create a new transaction participant
    pub fn new(
        config: ParticipantConfig,
        storage: Arc<S>,
        message_tx: mpsc::Sender<(ParticipantId, TwoPhaseMessage)>,
    ) -> Self {
        Self {
            config,
            transactions: RwLock::new(HashMap::new()),
            recovery_log: Arc::new(RecoveryLog::new()),
            storage,
            message_tx,
            stats: ParticipantStats::default(),
        }
    }

    /// Handle a prepare message from coordinator
    pub async fn handle_prepare(&self, message: TwoPhaseMessage) -> TwoPhaseResult<()> {
        let TwoPhaseMessage::Prepare {
            transaction_id,
            coordinator_id,
            participants: _,
            operations,
            timestamp: _,
        } = message
        else {
            return Err(TwoPhaseError::Internal("Expected Prepare message".into()));
        };

        self.stats.prepares_received.fetch_add(1, Ordering::Relaxed);

        // Log prepare received
        self.recovery_log
            .append(
                transaction_id.clone(),
                RecoveryLogRecordType::PrepareReceived {
                    coordinator_id: coordinator_id.clone(),
                },
            )
            .await?;

        // Create transaction record
        let txn = ParticipantTransaction::new(
            transaction_id.clone(),
            coordinator_id.clone(),
            operations.clone(),
        );

        {
            let mut transactions = self.transactions.write().await;
            transactions.insert(transaction_id.clone(), txn);
        }

        // Try to prepare locally
        let (vote, prepared_lsn) = match self.storage.prepare(&operations) {
            Ok(lsn) => {
                self.stats.votes_commit.fetch_add(1, Ordering::Relaxed);
                (Vote::Commit, Some(lsn))
            }
            Err(e) => {
                tracing::warn!("Failed to prepare transaction {}: {}", transaction_id, e);
                self.stats.votes_abort.fetch_add(1, Ordering::Relaxed);
                (Vote::Abort, None)
            }
        };

        // Update transaction state
        {
            let mut transactions = self.transactions.write().await;
            if let Some(txn) = transactions.get_mut(&transaction_id) {
                txn.state = TransactionState::Prepared;
                txn.vote = Some(vote);
                txn.prepared_lsn = prepared_lsn;
                txn.last_activity = Instant::now();
            }
        }

        // CRITICAL: Log vote before sending (must be durable if voting commit)
        self.recovery_log
            .append_sync(
                transaction_id.clone(),
                RecoveryLogRecordType::VoteSent { vote },
            )
            .await?;

        // Send vote to coordinator
        let vote_message = TwoPhaseMessage::Vote {
            transaction_id,
            participant_id: self.config.node_id.clone(),
            vote,
            prepared_lsn,
        };

        self.message_tx
            .send((coordinator_id, vote_message))
            .await
            .map_err(|e| TwoPhaseError::CommunicationFailure {
                participant_id: "coordinator".into(),
                error: e.to_string(),
            })?;

        Ok(())
    }

    /// Handle a commit message from coordinator
    pub async fn handle_commit(&self, message: TwoPhaseMessage) -> TwoPhaseResult<()> {
        let TwoPhaseMessage::Commit {
            transaction_id,
            coordinator_id,
            commit_timestamp: _,
        } = message
        else {
            return Err(TwoPhaseError::Internal("Expected Commit message".into()));
        };

        self.stats.commits_received.fetch_add(1, Ordering::Relaxed);

        // Log decision received
        self.recovery_log
            .append(
                transaction_id.clone(),
                RecoveryLogRecordType::DecisionReceived { commit: true },
            )
            .await?;

        let prepared_lsn = {
            let mut transactions = self.transactions.write().await;
            let txn = transactions
                .get_mut(&transaction_id)
                .ok_or_else(|| TwoPhaseError::TransactionNotFound(transaction_id.clone()))?;

            txn.state = TransactionState::Committing;
            txn.last_activity = Instant::now();
            txn.prepared_lsn
        };

        // Apply to local storage
        let success = if let Some(lsn) = prepared_lsn {
            match self.storage.commit(lsn) {
                Ok(()) => {
                    self.recovery_log
                        .append(transaction_id.clone(), RecoveryLogRecordType::Applied)
                        .await?;
                    true
                }
                Err(e) => {
                    tracing::error!("Failed to commit transaction {}: {}", transaction_id, e);
                    false
                }
            }
        } else {
            false
        };

        // Update state
        {
            let mut transactions = self.transactions.write().await;
            if let Some(txn) = transactions.get_mut(&transaction_id) {
                txn.state = TransactionState::Committed;
                txn.last_activity = Instant::now();
            }
        }

        // Send ack to coordinator
        let ack_message = TwoPhaseMessage::Ack {
            transaction_id,
            participant_id: self.config.node_id.clone(),
            success,
            error: if success {
                None
            } else {
                Some("Commit failed".into())
            },
        };

        self.message_tx
            .send((coordinator_id, ack_message))
            .await
            .map_err(|e| TwoPhaseError::CommunicationFailure {
                participant_id: "coordinator".into(),
                error: e.to_string(),
            })?;

        Ok(())
    }

    /// Handle an abort message from coordinator
    pub async fn handle_abort(&self, message: TwoPhaseMessage) -> TwoPhaseResult<()> {
        let TwoPhaseMessage::Abort {
            transaction_id,
            coordinator_id,
            reason,
        } = message
        else {
            return Err(TwoPhaseError::Internal("Expected Abort message".into()));
        };

        self.stats.aborts_received.fetch_add(1, Ordering::Relaxed);

        tracing::debug!("Aborting transaction {}: {}", transaction_id, reason);

        // Log decision received
        self.recovery_log
            .append(
                transaction_id.clone(),
                RecoveryLogRecordType::DecisionReceived { commit: false },
            )
            .await?;

        let prepared_lsn = {
            let mut transactions = self.transactions.write().await;
            if let Some(txn) = transactions.get_mut(&transaction_id) {
                txn.state = TransactionState::Aborting;
                txn.last_activity = Instant::now();
                txn.prepared_lsn
            } else {
                // Transaction might not exist if abort came before prepare
                None
            }
        };

        // Rollback local storage
        let success = if let Some(lsn) = prepared_lsn {
            match self.storage.rollback(lsn) {
                Ok(()) => true,
                Err(e) => {
                    tracing::error!("Failed to rollback transaction {}: {}", transaction_id, e);
                    false
                }
            }
        } else {
            true
        };

        // Update state
        {
            let mut transactions = self.transactions.write().await;
            if let Some(txn) = transactions.get_mut(&transaction_id) {
                txn.state = TransactionState::Aborted;
                txn.last_activity = Instant::now();
            }
        }

        // Send ack to coordinator
        let ack_message = TwoPhaseMessage::Ack {
            transaction_id,
            participant_id: self.config.node_id.clone(),
            success,
            error: if success {
                None
            } else {
                Some("Rollback failed".into())
            },
        };

        self.message_tx
            .send((coordinator_id, ack_message))
            .await
            .map_err(|e| TwoPhaseError::CommunicationFailure {
                participant_id: "coordinator".into(),
                error: e.to_string(),
            })?;

        Ok(())
    }

    /// Check for timed out transactions and query coordinator
    pub async fn check_timeouts(&self) -> Vec<TransactionId> {
        let mut uncertain = Vec::new();
        let now = Instant::now();

        let transactions = self.transactions.read().await;

        for (txn_id, txn) in transactions.iter() {
            if txn.state == TransactionState::Prepared
                && now.duration_since(txn.last_activity) > self.config.decision_timeout
            {
                uncertain.push((txn_id.clone(), txn.coordinator_id.clone()));
                self.stats.decision_timeouts.fetch_add(1, Ordering::Relaxed);
            }
        }

        drop(transactions);

        // Query coordinator for uncertain transactions
        let mut queried = Vec::new();
        for (txn_id, coordinator_id) in uncertain {
            let query = TwoPhaseMessage::QueryStatus {
                transaction_id: txn_id.clone(),
                requester_id: self.config.node_id.clone(),
            };

            if self.message_tx.send((coordinator_id, query)).await.is_ok() {
                queried.push(txn_id);
            }
        }

        queried
    }

    /// Handle status response from coordinator (during recovery)
    pub async fn handle_status_response(&self, message: TwoPhaseMessage) -> TwoPhaseResult<()> {
        let TwoPhaseMessage::StatusResponse {
            transaction_id,
            state,
            responder_id,
        } = message
        else {
            return Err(TwoPhaseError::Internal(
                "Expected StatusResponse message".into(),
            ));
        };

        match state {
            TransactionState::Committed => {
                // Commit locally
                let commit_msg = TwoPhaseMessage::Commit {
                    transaction_id: transaction_id.clone(),
                    coordinator_id: responder_id,
                    commit_timestamp: 0,
                };
                self.handle_commit(commit_msg).await?;
            }
            TransactionState::Aborted => {
                // Abort locally
                let abort_msg = TwoPhaseMessage::Abort {
                    transaction_id: transaction_id.clone(),
                    coordinator_id: responder_id,
                    reason: "Recovered abort from coordinator".into(),
                };
                self.handle_abort(abort_msg).await?;
            }
            TransactionState::Unknown => {
                // Coordinator doesn't know - presumed abort
                let abort_msg = TwoPhaseMessage::Abort {
                    transaction_id: transaction_id.clone(),
                    coordinator_id: responder_id,
                    reason: "Presumed abort - coordinator unknown".into(),
                };
                self.handle_abort(abort_msg).await?;
            }
            _ => {
                // Still in progress, keep waiting
                tracing::debug!(
                    "Transaction {} still in progress at coordinator (state: {})",
                    transaction_id,
                    state
                );
            }
        }

        Ok(())
    }

    /// Recover incomplete transactions after participant restart
    pub async fn recover(&self) -> TwoPhaseResult<Vec<TransactionId>> {
        let incomplete = self.recovery_log.get_incomplete_transactions().await;
        let mut recovered = Vec::new();

        for (txn_id, entries) in incomplete {
            let mut coordinator_id: Option<ParticipantId> = None;
            let mut vote_sent: Option<Vote> = None;
            let mut decision_received: Option<bool> = None;

            for entry in &entries {
                match &entry.record_type {
                    RecoveryLogRecordType::PrepareReceived {
                        coordinator_id: cid,
                    } => {
                        coordinator_id = Some(cid.clone());
                    }
                    RecoveryLogRecordType::VoteSent { vote } => {
                        vote_sent = Some(*vote);
                    }
                    RecoveryLogRecordType::DecisionReceived { commit } => {
                        decision_received = Some(*commit);
                    }
                    other => {
                        tracing::warn!(
                            "Unexpected recovery log record {:?} for txn {} during participant recovery",
                            other,
                            txn_id
                        );
                    }
                }
            }

            if let Some(cid) = coordinator_id {
                if decision_received.is_none() && vote_sent == Some(Vote::Commit) {
                    // We voted commit but never got decision - query coordinator
                    let query = TwoPhaseMessage::QueryStatus {
                        transaction_id: txn_id.clone(),
                        requester_id: self.config.node_id.clone(),
                    };
                    let _ = self.message_tx.send((cid, query)).await;
                } else if vote_sent.is_none() || vote_sent == Some(Vote::Abort) {
                    // Never voted commit - safe to abort
                    // Log and clean up
                    self.recovery_log
                        .append_sync(
                            txn_id.clone(),
                            RecoveryLogRecordType::DecisionReceived { commit: false },
                        )
                        .await?;
                }
            }

            recovered.push(txn_id);
        }

        Ok(recovered)
    }

    /// Get transaction state
    pub async fn get_transaction_state(&self, txn_id: &TransactionId) -> Option<TransactionState> {
        let transactions = self.transactions.read().await;
        transactions.get(txn_id).map(|txn| txn.state)
    }

    /// Get participant statistics snapshot
    pub fn stats(&self) -> ParticipantStatsSnapshot {
        ParticipantStatsSnapshot {
            prepares_received: self.stats.prepares_received.load(Ordering::Relaxed),
            votes_commit: self.stats.votes_commit.load(Ordering::Relaxed),
            votes_abort: self.stats.votes_abort.load(Ordering::Relaxed),
            commits_received: self.stats.commits_received.load(Ordering::Relaxed),
            aborts_received: self.stats.aborts_received.load(Ordering::Relaxed),
            decision_timeouts: self.stats.decision_timeouts.load(Ordering::Relaxed),
        }
    }

    /// Clean up completed transactions (garbage collection)
    pub async fn cleanup_completed(&self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut transactions = self.transactions.write().await;
        let before = transactions.len();

        transactions.retain(|_, txn| {
            if txn.state.is_terminal() {
                now.duration_since(txn.last_activity) < max_age
            } else {
                true
            }
        });

        before - transactions.len()
    }
}

/// Snapshot of participant statistics
#[derive(Debug, Clone)]
pub struct ParticipantStatsSnapshot {
    pub prepares_received: u64,
    pub votes_commit: u64,
    pub votes_abort: u64,
    pub commits_received: u64,
    pub aborts_received: u64,
    pub decision_timeouts: u64,
}

// ============================================================================
// Amorphic Storage Participant (bridges 2PC ↔ AmorphicTableStorage)
// ============================================================================

/// ParticipantStorage backed by JouleDB's amorphic storage.
/// Translates 2PC operations (key-value Put/Delete) to table-level operations.
///
/// Key format: `table_name:row_key` — the table is extracted from the key prefix.
/// Value format: JSON-serialized row data.
pub struct AmorphicParticipantStorage {
    amorphic: std::sync::Arc<crate::amorphic_adapter::AmorphicTableStorage>,
    prepared: std::sync::Mutex<HashMap<LogSequenceNumber, Vec<TransactionOperation>>>,
    next_lsn: AtomicU64,
}

impl AmorphicParticipantStorage {
    pub fn new(amorphic: std::sync::Arc<crate::amorphic_adapter::AmorphicTableStorage>) -> Self {
        Self {
            amorphic,
            prepared: std::sync::Mutex::new(HashMap::new()),
            next_lsn: AtomicU64::new(1),
        }
    }

    /// Parse a 2PC key into (table_name, row_key).
    fn parse_key(key: &[u8]) -> Option<(String, String)> {
        let s = std::str::from_utf8(key).ok()?;
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() == 2 {
            Some((parts[0].to_string(), parts[1].to_string()))
        } else {
            None
        }
    }
}

impl ParticipantStorage for AmorphicParticipantStorage {
    fn prepare(&self, operations: &[TransactionOperation]) -> TwoPhaseResult<LogSequenceNumber> {
        // Validate that all referenced tables exist
        let tables = self.amorphic.list_tables();
        for op in operations {
            if let Some((table, _)) = Self::parse_key(op.key()) {
                if !tables.contains(&table) {
                    return Err(TwoPhaseError::Internal(format!(
                        "Table '{}' not found",
                        table
                    )));
                }
            }
        }

        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let mut prepared = crate::lock_util::mutex_lock(&self.prepared);
        prepared.insert(lsn, operations.to_vec());
        Ok(lsn)
    }

    fn commit(&self, lsn: LogSequenceNumber) -> TwoPhaseResult<()> {
        let mut prepared = crate::lock_util::mutex_lock(&self.prepared);
        if let Some(ops) = prepared.remove(&lsn) {
            for op in ops {
                match op {
                    TransactionOperation::Put { key, value } => {
                        if let Some((table, _row_key)) = Self::parse_key(&key) {
                            // Parse value as JSON row data and insert
                            if let Ok(row_json) =
                                serde_json::from_slice::<serde_json::Value>(&value)
                            {
                                if let Some(obj) = row_json.as_object() {
                                    let columns: Vec<String> = obj.keys().cloned().collect();
                                    let values: Vec<joule_db_query::ast::Value> = obj
                                        .values()
                                        .map(|v| match v {
                                            serde_json::Value::String(s) => {
                                                joule_db_query::ast::Value::String(s.clone())
                                            }
                                            serde_json::Value::Number(n) => {
                                                if let Some(i) = n.as_i64() {
                                                    joule_db_query::ast::Value::Int(i)
                                                } else {
                                                    joule_db_query::ast::Value::Float(
                                                        n.as_f64().unwrap_or(0.0),
                                                    )
                                                }
                                            }
                                            serde_json::Value::Bool(b) => {
                                                joule_db_query::ast::Value::Bool(*b)
                                            }
                                            serde_json::Value::Null => {
                                                joule_db_query::ast::Value::Null
                                            }
                                            _ => joule_db_query::ast::Value::String(v.to_string()),
                                        })
                                        .collect();
                                    let row =
                                        joule_db_query::executor::RowData::new(columns, values);
                                    let _ = self.amorphic.insert(&table, &row);
                                }
                            }
                        }
                    }
                    TransactionOperation::Delete { key } => {
                        if let Some((table, row_key)) = Self::parse_key(&key) {
                            let filter = joule_db_query::ast::Expression::eq(
                                joule_db_query::ast::Expression::Column("id".into()),
                                joule_db_query::ast::Expression::Literal(
                                    joule_db_query::ast::Value::String(row_key),
                                ),
                            );
                            let _ = self.amorphic.delete(&table, Some(&filter));
                        }
                    }
                    TransactionOperation::ConditionalPut { key, new_value, .. } => {
                        // Same as Put for now (CAS validation was done in prepare)
                        if let Some((table, _row_key)) = Self::parse_key(&key) {
                            if let Ok(row_json) =
                                serde_json::from_slice::<serde_json::Value>(&new_value)
                            {
                                if let Some(obj) = row_json.as_object() {
                                    let columns: Vec<String> = obj.keys().cloned().collect();
                                    let values: Vec<joule_db_query::ast::Value> = obj
                                        .values()
                                        .map(|v| match v {
                                            serde_json::Value::String(s) => {
                                                joule_db_query::ast::Value::String(s.clone())
                                            }
                                            serde_json::Value::Number(n) => {
                                                if let Some(i) = n.as_i64() {
                                                    joule_db_query::ast::Value::Int(i)
                                                } else {
                                                    joule_db_query::ast::Value::Float(
                                                        n.as_f64().unwrap_or(0.0),
                                                    )
                                                }
                                            }
                                            serde_json::Value::Bool(b) => {
                                                joule_db_query::ast::Value::Bool(*b)
                                            }
                                            serde_json::Value::Null => {
                                                joule_db_query::ast::Value::Null
                                            }
                                            _ => joule_db_query::ast::Value::String(v.to_string()),
                                        })
                                        .collect();
                                    let row =
                                        joule_db_query::executor::RowData::new(columns, values);
                                    let _ = self.amorphic.insert(&table, &row);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn rollback(&self, lsn: LogSequenceNumber) -> TwoPhaseResult<()> {
        let mut prepared = crate::lock_util::mutex_lock(&self.prepared);
        prepared.remove(&lsn);
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn create_test_operations() -> Vec<TransactionOperation> {
        vec![
            TransactionOperation::Put {
                key: b"key1".to_vec(),
                value: b"value1".to_vec(),
            },
            TransactionOperation::Put {
                key: b"key2".to_vec(),
                value: b"value2".to_vec(),
            },
        ]
    }

    #[test]
    fn test_transaction_state_is_terminal() {
        assert!(!TransactionState::Started.is_terminal());
        assert!(!TransactionState::Preparing.is_terminal());
        assert!(!TransactionState::Prepared.is_terminal());
        assert!(!TransactionState::Committing.is_terminal());
        assert!(TransactionState::Committed.is_terminal());
        assert!(!TransactionState::Aborting.is_terminal());
        assert!(TransactionState::Aborted.is_terminal());
    }

    #[test]
    fn test_transaction_state_can_abort() {
        assert!(TransactionState::Started.can_abort());
        assert!(TransactionState::Preparing.can_abort());
        assert!(TransactionState::Prepared.can_abort());
        assert!(!TransactionState::Committing.can_abort());
        assert!(!TransactionState::Committed.can_abort());
        assert!(TransactionState::Aborting.can_abort());
        assert!(!TransactionState::Aborted.can_abort());
    }

    #[test]
    fn test_two_phase_message_transaction_id() {
        let prepare = TwoPhaseMessage::Prepare {
            transaction_id: "txn1".into(),
            coordinator_id: "coord".into(),
            participants: vec![],
            operations: vec![],
            timestamp: 0,
        };
        assert_eq!(prepare.transaction_id(), "txn1");

        let vote = TwoPhaseMessage::Vote {
            transaction_id: "txn2".into(),
            participant_id: "p1".into(),
            vote: Vote::Commit,
            prepared_lsn: None,
        };
        assert_eq!(vote.transaction_id(), "txn2");
    }

    #[test]
    fn test_transaction_operation_key() {
        let put = TransactionOperation::Put {
            key: b"mykey".to_vec(),
            value: b"myvalue".to_vec(),
        };
        assert_eq!(put.key(), b"mykey");

        let delete = TransactionOperation::Delete {
            key: b"otherkey".to_vec(),
        };
        assert_eq!(delete.key(), b"otherkey");
    }

    #[tokio::test]
    async fn test_recovery_log_append_and_read() {
        let log = RecoveryLog::new();

        let lsn1 = log
            .append(
                "txn1".into(),
                RecoveryLogRecordType::TransactionStart {
                    participants: vec!["p1".into(), "p2".into()],
                    operations: create_test_operations(),
                },
            )
            .await
            .unwrap();

        let lsn2 = log
            .append("txn1".into(), RecoveryLogRecordType::PrepareSent)
            .await
            .unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);

        let entries = log.get_transaction_entries(&"txn1".into()).await;
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn test_recovery_log_incomplete_transactions() {
        let log = RecoveryLog::new();

        // Complete transaction
        log.append(
            "txn1".into(),
            RecoveryLogRecordType::TransactionStart {
                participants: vec![],
                operations: vec![],
            },
        )
        .await
        .unwrap();
        log.append("txn1".into(), RecoveryLogRecordType::TransactionComplete)
            .await
            .unwrap();

        // Incomplete transaction
        log.append(
            "txn2".into(),
            RecoveryLogRecordType::TransactionStart {
                participants: vec![],
                operations: vec![],
            },
        )
        .await
        .unwrap();

        let incomplete = log.get_incomplete_transactions().await;
        assert_eq!(incomplete.len(), 1);
        assert_eq!(incomplete[0].0, "txn2");
    }

    #[tokio::test]
    async fn test_coordinator_begin_transaction() {
        let (tx, _rx) = mpsc::channel(100);
        let config = CoordinatorConfig {
            node_id: "coord1".into(),
            ..Default::default()
        };

        let coordinator = TransactionCoordinator::new(config, tx);

        let result = coordinator
            .begin_transaction(vec!["p1".into(), "p2".into()], create_test_operations())
            .await;

        assert!(result.is_ok());

        let stats = coordinator.stats();
        assert_eq!(stats.transactions_started, 1);
    }

    #[tokio::test]
    async fn test_coordinator_handle_votes_commit() {
        let (tx, mut rx) = mpsc::channel(100);
        let config = CoordinatorConfig {
            node_id: "coord1".into(),
            ..Default::default()
        };

        let coordinator = TransactionCoordinator::new(config, tx);

        // Begin transaction
        let _completion_rx = coordinator
            .begin_transaction(vec!["p1".into(), "p2".into()], create_test_operations())
            .await
            .unwrap();

        // Get the transaction ID from the prepare messages
        let (_, msg1) = rx.recv().await.unwrap();
        let txn_id = msg1.transaction_id().clone();

        // Drain remaining prepare message
        let _ = rx.recv().await;

        // Send commit votes from both participants
        let vote1 = TwoPhaseMessage::Vote {
            transaction_id: txn_id.clone(),
            participant_id: "p1".into(),
            vote: Vote::Commit,
            prepared_lsn: Some(1),
        };
        coordinator.handle_vote(vote1).await.unwrap();

        let vote2 = TwoPhaseMessage::Vote {
            transaction_id: txn_id.clone(),
            participant_id: "p2".into(),
            vote: Vote::Commit,
            prepared_lsn: Some(1),
        };
        coordinator.handle_vote(vote2).await.unwrap();

        // Should have sent commit messages
        let state = coordinator.get_transaction_state(&txn_id).await.unwrap();
        assert_eq!(state, TransactionState::Committing);
    }

    #[tokio::test]
    async fn test_coordinator_handle_votes_abort() {
        let (tx, mut rx) = mpsc::channel(100);
        let config = CoordinatorConfig {
            node_id: "coord1".into(),
            ..Default::default()
        };

        let coordinator = TransactionCoordinator::new(config, tx);

        // Begin transaction
        let _completion_rx = coordinator
            .begin_transaction(vec!["p1".into(), "p2".into()], create_test_operations())
            .await
            .unwrap();

        // Get the transaction ID
        let (_, msg1) = rx.recv().await.unwrap();
        let txn_id = msg1.transaction_id().clone();
        let _ = rx.recv().await;

        // One participant votes abort
        let vote1 = TwoPhaseMessage::Vote {
            transaction_id: txn_id.clone(),
            participant_id: "p1".into(),
            vote: Vote::Commit,
            prepared_lsn: Some(1),
        };
        coordinator.handle_vote(vote1).await.unwrap();

        let vote2 = TwoPhaseMessage::Vote {
            transaction_id: txn_id.clone(),
            participant_id: "p2".into(),
            vote: Vote::Abort,
            prepared_lsn: None,
        };
        coordinator.handle_vote(vote2).await.unwrap();

        // Should be aborting
        let state = coordinator.get_transaction_state(&txn_id).await.unwrap();
        assert_eq!(state, TransactionState::Aborting);
    }

    #[tokio::test]
    async fn test_participant_handle_prepare_commit() {
        let (tx, mut rx) = mpsc::channel(100);
        let config = ParticipantConfig {
            node_id: "p1".into(),
            ..Default::default()
        };

        let storage = Arc::new(InMemoryStorage::new());
        let participant = TransactionParticipant::new(config, storage, tx);

        let prepare = TwoPhaseMessage::Prepare {
            transaction_id: "txn1".into(),
            coordinator_id: "coord".into(),
            participants: vec!["p1".into()],
            operations: create_test_operations(),
            timestamp: 0,
        };

        participant.handle_prepare(prepare).await.unwrap();

        // Should have sent a vote
        let (target, msg) = rx.recv().await.unwrap();
        assert_eq!(target, "coord");

        if let TwoPhaseMessage::Vote { vote, .. } = msg {
            assert_eq!(vote, Vote::Commit);
        } else {
            panic!("Expected Vote message");
        }

        let stats = participant.stats();
        assert_eq!(stats.prepares_received, 1);
        assert_eq!(stats.votes_commit, 1);
    }

    #[tokio::test]
    async fn test_participant_handle_commit() {
        let (tx, mut rx) = mpsc::channel(100);
        let config = ParticipantConfig {
            node_id: "p1".into(),
            ..Default::default()
        };

        let storage = Arc::new(InMemoryStorage::new());
        let participant = TransactionParticipant::new(config, storage.clone(), tx);

        // First prepare
        let prepare = TwoPhaseMessage::Prepare {
            transaction_id: "txn1".into(),
            coordinator_id: "coord".into(),
            participants: vec!["p1".into()],
            operations: create_test_operations(),
            timestamp: 0,
        };
        participant.handle_prepare(prepare).await.unwrap();
        let _ = rx.recv().await; // drain vote

        // Then commit
        let commit = TwoPhaseMessage::Commit {
            transaction_id: "txn1".into(),
            coordinator_id: "coord".into(),
            commit_timestamp: 0,
        };
        participant.handle_commit(commit).await.unwrap();

        // Check ack was sent
        let (target, msg) = rx.recv().await.unwrap();
        assert_eq!(target, "coord");

        if let TwoPhaseMessage::Ack { success, .. } = msg {
            assert!(success);
        } else {
            panic!("Expected Ack message");
        }

        // Verify data was committed
        let value = storage.get(b"key1");
        assert_eq!(value, Some(b"value1".to_vec()));
    }

    #[tokio::test]
    async fn test_participant_handle_abort() {
        let (tx, mut rx) = mpsc::channel(100);
        let config = ParticipantConfig {
            node_id: "p1".into(),
            ..Default::default()
        };

        let storage = Arc::new(InMemoryStorage::new());
        let participant = TransactionParticipant::new(config, storage.clone(), tx);

        // First prepare
        let prepare = TwoPhaseMessage::Prepare {
            transaction_id: "txn1".into(),
            coordinator_id: "coord".into(),
            participants: vec!["p1".into()],
            operations: create_test_operations(),
            timestamp: 0,
        };
        participant.handle_prepare(prepare).await.unwrap();
        let _ = rx.recv().await; // drain vote

        // Then abort
        let abort = TwoPhaseMessage::Abort {
            transaction_id: "txn1".into(),
            coordinator_id: "coord".into(),
            reason: "Test abort".into(),
        };
        participant.handle_abort(abort).await.unwrap();

        // Check ack was sent
        let (_, msg) = rx.recv().await.unwrap();
        if let TwoPhaseMessage::Ack { success, .. } = msg {
            assert!(success);
        } else {
            panic!("Expected Ack message");
        }

        // Verify data was NOT committed
        let value = storage.get(b"key1");
        assert_eq!(value, None);
    }

    #[test]
    fn test_in_memory_storage() {
        let storage = InMemoryStorage::new();

        let ops = vec![
            TransactionOperation::Put {
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            TransactionOperation::Put {
                key: b"k2".to_vec(),
                value: b"v2".to_vec(),
            },
        ];

        let lsn = storage.prepare(&ops).unwrap();
        assert!(lsn > 0);

        // Before commit, data shouldn't be visible
        assert_eq!(storage.get(b"k1"), None);

        storage.commit(lsn).unwrap();

        // After commit, data should be visible
        assert_eq!(storage.get(b"k1"), Some(b"v1".to_vec()));
        assert_eq!(storage.get(b"k2"), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_in_memory_storage_rollback() {
        let storage = InMemoryStorage::new();

        let ops = vec![TransactionOperation::Put {
            key: b"k1".to_vec(),
            value: b"v1".to_vec(),
        }];

        let lsn = storage.prepare(&ops).unwrap();
        storage.rollback(lsn).unwrap();

        // Data should not be visible after rollback
        assert_eq!(storage.get(b"k1"), None);
    }

    #[tokio::test]
    async fn test_coordinator_cleanup() {
        let (tx, _rx) = mpsc::channel(100);
        let config = CoordinatorConfig {
            node_id: "coord1".into(),
            ..Default::default()
        };

        let coordinator = TransactionCoordinator::new(config, tx);

        // This test verifies cleanup doesn't panic on empty state
        let cleaned = coordinator.cleanup_completed(Duration::from_secs(0)).await;
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_coordinator_stats_default() {
        let stats = CoordinatorStats::default();
        assert_eq!(stats.transactions_started.load(Ordering::Relaxed), 0);
        assert_eq!(stats.transactions_committed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_participant_stats_default() {
        let stats = ParticipantStats::default();
        assert_eq!(stats.prepares_received.load(Ordering::Relaxed), 0);
        assert_eq!(stats.votes_commit.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_durable_recovery_log_persist_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("2pc.log");

        // Write some entries to a durable log
        {
            let log = RecoveryLog::new_durable(&log_path).unwrap();
            log.append_sync(
                "txn1".into(),
                RecoveryLogRecordType::TransactionStart {
                    participants: vec!["p1".into(), "p2".into()],
                    operations: create_test_operations(),
                },
            )
            .await
            .unwrap();

            log.append(
                "txn1".into(),
                RecoveryLogRecordType::Decision { commit: true },
            )
            .await
            .unwrap();
            log.sync().await.unwrap();
        }
        // Log dropped here — file should be flushed

        // Reload from the same file
        let log2 = RecoveryLog::new_durable(&log_path).unwrap();
        let entries = log2.get_transaction_entries(&"txn1".into()).await;
        assert_eq!(entries.len(), 2);
        assert!(matches!(
            entries[0].record_type,
            RecoveryLogRecordType::TransactionStart { .. }
        ));
        assert!(matches!(
            entries[1].record_type,
            RecoveryLogRecordType::Decision { commit: true }
        ));

        // Next LSN should continue from where we left off
        let lsn = log2
            .append("txn2".into(), RecoveryLogRecordType::PrepareSent)
            .await
            .unwrap();
        assert_eq!(lsn, 3);
    }

    #[tokio::test]
    async fn test_durable_log_incomplete_transactions_after_reload() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("2pc.log");

        {
            let log = RecoveryLog::new_durable(&log_path).unwrap();

            // Complete txn1
            log.append(
                "txn1".into(),
                RecoveryLogRecordType::TransactionStart {
                    participants: vec!["p1".into()],
                    operations: vec![],
                },
            )
            .await
            .unwrap();
            log.append("txn1".into(), RecoveryLogRecordType::TransactionComplete)
                .await
                .unwrap();

            // Incomplete txn2 (decision made but not completed)
            log.append(
                "txn2".into(),
                RecoveryLogRecordType::TransactionStart {
                    participants: vec!["p1".into()],
                    operations: create_test_operations(),
                },
            )
            .await
            .unwrap();
            log.append(
                "txn2".into(),
                RecoveryLogRecordType::Decision { commit: true },
            )
            .await
            .unwrap();
            log.sync().await.unwrap();
        }

        // Reload and check recovery
        let log2 = RecoveryLog::new_durable(&log_path).unwrap();
        let incomplete = log2.get_incomplete_transactions().await;
        assert_eq!(incomplete.len(), 1);
        assert_eq!(incomplete[0].0, "txn2");
    }

    #[tokio::test]
    async fn test_durable_log_truncate_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("2pc.log");

        let log = RecoveryLog::new_durable(&log_path).unwrap();
        let lsn1 = log
            .append("txn1".into(), RecoveryLogRecordType::PrepareSent)
            .await
            .unwrap();
        let _lsn2 = log
            .append("txn2".into(), RecoveryLogRecordType::PrepareSent)
            .await
            .unwrap();
        let lsn3 = log
            .append("txn3".into(), RecoveryLogRecordType::PrepareSent)
            .await
            .unwrap();
        log.sync().await.unwrap();

        // Truncate entries before lsn2 (keep lsn2 and lsn3)
        log.truncate_before(lsn1 + 1).await;
        drop(log);

        // Reload — should only have lsn2 and lsn3
        let log2 = RecoveryLog::new_durable(&log_path).unwrap();
        let txn1_entries = log2.get_transaction_entries(&"txn1".into()).await;
        assert_eq!(txn1_entries.len(), 0); // truncated

        let txn3_entries = log2.get_transaction_entries(&"txn3".into()).await;
        assert_eq!(txn3_entries.len(), 1);
        assert_eq!(txn3_entries[0].lsn, lsn3);
    }

    fn create_test_amorphic() -> std::sync::Arc<crate::amorphic_adapter::AmorphicTableStorage> {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        // Leak the tempdir so it stays alive for the test
        std::mem::forget(dir);
        std::sync::Arc::new(crate::amorphic_adapter::AmorphicTableStorage::new(store))
    }

    #[test]
    fn test_amorphic_participant_parse_key() {
        assert_eq!(
            AmorphicParticipantStorage::parse_key(b"users:42"),
            Some(("users".to_string(), "42".to_string()))
        );
        assert_eq!(
            AmorphicParticipantStorage::parse_key(b"orders:abc:def"),
            Some(("orders".to_string(), "abc:def".to_string()))
        );
        assert_eq!(AmorphicParticipantStorage::parse_key(b"nocolon"), None);
    }

    #[test]
    fn test_amorphic_participant_prepare_validates_tables() {
        let amorphic = create_test_amorphic();
        let storage = AmorphicParticipantStorage::new(amorphic);

        // Keys reference "users" table which doesn't exist → prepare fails
        let ops = vec![TransactionOperation::Put {
            key: b"users:1".to_vec(),
            value: b"{\"id\": 1}".to_vec(),
        }];
        let result = storage.prepare(&ops);
        assert!(result.is_err());
    }

    #[test]
    fn test_amorphic_participant_rollback() {
        let amorphic = create_test_amorphic();
        let storage = AmorphicParticipantStorage::new(amorphic);

        // Prepare with key that has no colon (skips table check) — succeeds
        let ops = vec![TransactionOperation::Put {
            key: b"raw_key".to_vec(),
            value: b"value".to_vec(),
        }];
        let lsn = storage.prepare(&ops).unwrap();

        // Rollback should clear the prepared operations
        storage.rollback(lsn).unwrap();

        // Commit after rollback should be a no-op (no prepared ops for that lsn)
        let result = storage.commit(lsn);
        assert!(result.is_ok());
    }
}
