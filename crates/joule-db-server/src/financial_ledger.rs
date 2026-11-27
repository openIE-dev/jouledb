//! SQL-Native Double-Entry Core Banking Ledger (`TigerBeetle` Invariants + SQL Queryability)
//!
//! Eliminates the "PostgreSQL + TigerBeetle" split-brain architecture by embedding an
//! exact `128-bit` minor-unit double-entry accounting state machine directly inside JouleDB:
//! - **Conservation of Money ($\sum \text{debits} \equiv \sum \text{credits}$):** Enforced structurally
//!   on both 2-account transfers and N-leg atomic journal entries.
//! - **Hard Overdraft Ceilings:** `debits_must_not_exceed_credits` (`posted_debits + pending_debits <= posted_credits`)
//!   and `credits_must_not_exceed_debits` (`posted_credits + pending_credits <= posted_debits`).
//! - **Two-Phase Pending Holds (`HOLD` -> `POST` / `VOID`):** Full or partial capture with automatic
//!   release of uncaptured hold remainder.
//! - **End-to-End Native Idempotency Keys:** Retrying a timed-out request with the same
//!   `idempotency_key` and identical payload returns `idempotent_replay = true` without moving funds
//!   twice; reusing an `idempotency_key` with a mutated payload returns `IdempotencyConflict`.
//! - **Tamper-Evident SHA-256 Hash-Chained Audit Log + `fsync` WAL:** Persisted to
//!   `financial_ledger.jwal`, verifiable in one SQL call via `LEDGER VERIFY`, and queryable via
//!   virtual SQL tables `jouledb_ledger_accounts`, `jouledb_ledger_transfers`, and `jouledb_ledger_legs`.
//! - **v0.14.0 Binary `JWL1` Frames + Single-`fdatasync` Batch & FIFO Group Commit:** Compact
//!   CRC-32 framed binary WAL (`0x89 'J' 'W' 'L' 0x01`, backward-compatible with JSONL `.jwal`),
//!   explicit `submit_transfer_batch` (`atomic` or per-item clearing), and zero-starvation
//!   Leader-Follower group-commit coalescing on concurrent `submit_transfer` callers.

#![deny(clippy::unwrap_used)]

use crate::decimal128::Decimal128;
use crate::query::{QueryErrorResponse, QueryResponse};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const LEDGER_WAL_FILENAME: &str = "financial_ledger.jwal";
pub const JWL1_MAGIC: [u8; 5] = [0x89, b'J', b'W', b'L', 0x01];
pub const MAX_LEDGER_BATCH_SIZE: usize = 32_768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AccountFlags {
    /// Liability / customer wallet floor: `posted_debits + pending_debits <= posted_credits`.
    #[serde(default)]
    pub debits_must_not_exceed_credits: bool,
    /// Asset / reserve floor: `posted_credits + pending_credits <= posted_debits`.
    #[serde(default)]
    pub credits_must_not_exceed_debits: bool,
    /// Compliance / AML freeze: blocks new debits and credits while `true`.
    #[serde(default)]
    pub frozen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerAccount {
    pub account_id: String,
    pub ledger_id: u32,
    pub code: u16,
    pub scale: u8,
    pub flags: AccountFlags,
    pub posted_debits: i128,
    pub posted_credits: i128,
    pub pending_debits: i128,
    pub pending_credits: i128,
    pub created_at_ns: u64,
    pub version: u64,
}

impl LedgerAccount {
    /// Net credit balance (`posted_credits - posted_debits`), standard for customer wallets/liabilities.
    #[inline]
    pub fn net_credits_minor(&self) -> i128 {
        self.posted_credits.saturating_sub(self.posted_debits)
    }

    /// Available spendable credit balance (`posted_credits - posted_debits - pending_debits`).
    #[inline]
    pub fn available_credits_minor(&self) -> i128 {
        self.posted_credits
            .saturating_sub(self.posted_debits)
            .saturating_sub(self.pending_debits)
    }

    /// Net debit balance (`posted_debits - posted_credits`), standard for asset/vault accounts.
    #[inline]
    pub fn net_debits_minor(&self) -> i128 {
        self.posted_debits.saturating_sub(self.posted_credits)
    }

    pub fn format_net_credits(&self) -> String {
        Decimal128::new(self.net_credits_minor(), self.scale)
            .map(|d| d.format_fixed())
            .unwrap_or_else(|| self.net_credits_minor().to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegDirection {
    Debit,
    Credit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerLeg {
    pub account_id: String,
    pub direction: LegDirection,
    pub amount_minor: i128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferKind {
    Standard,
    MultiLeg,
    Hold,
    PostHold,
    VoidHold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferStatus {
    Posted,
    Pending,
    Captured,
    Voided,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerTransferRecord {
    pub seq: u64,
    pub transfer_id: String,
    pub idempotency_key: String,
    pub kind: TransferKind,
    pub status: TransferStatus,
    pub ledger_id: u32,
    pub scale: u8,
    pub legs: Vec<LedgerLeg>,
    pub total_amount_minor: i128,
    pub captured_amount_minor: Option<i128>,
    pub parent_hold_id: Option<String>,
    pub memo: Option<String>,
    pub payload_hash: String,
    pub prev_hash: String,
    pub entry_hash: String,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferReceipt {
    pub record: LedgerTransferRecord,
    pub idempotent_replay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainVerificationReport {
    pub valid: bool,
    pub accounts_count: usize,
    pub journal_entries_verified: usize,
    pub total_posted_debits: i128,
    pub total_posted_credits: i128,
    pub total_pending_debits: i128,
    pub total_pending_credits: i128,
    pub head_hash: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    AccountAlreadyExists(String),
    AccountNotFound(String),
    AccountFrozen(String),
    InvalidAmount(String),
    SameAccountTransfer(String),
    LedgerCurrencyMismatch {
        expected: u32,
        found: u32,
        account: String,
    },
    ScaleMismatch {
        expected: u8,
        found: u8,
        account: String,
    },
    UnbalancedLegs {
        debits: i128,
        credits: i128,
    },
    InsufficientCreditBalance {
        account: String,
        available: i128,
        required: i128,
    },
    InsufficientDebitBalance {
        account: String,
        available: i128,
        required: i128,
    },
    HoldNotFound(String),
    HoldAlreadyResolved {
        hold_id: String,
        status: TransferStatus,
    },
    CaptureExceedsHold {
        hold_id: String,
        hold_amount: i128,
        capture_amount: i128,
    },
    IdempotencyConflict {
        idempotency_key: String,
    },
    Persistence(String),
    Syntax(String),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccountAlreadyExists(a) => write!(f, "ledger account '{a}' already exists"),
            Self::AccountNotFound(a) => write!(f, "ledger account '{a}' not found"),
            Self::AccountFrozen(a) => write!(f, "ledger account '{a}' is frozen"),
            Self::InvalidAmount(m) => write!(f, "invalid ledger amount: {m}"),
            Self::SameAccountTransfer(a) => {
                write!(f, "debit and credit account '{a}' cannot be identical")
            }
            Self::LedgerCurrencyMismatch {
                expected,
                found,
                account,
            } => write!(
                f,
                "ledger mismatch on account '{account}': expected ledger {expected}, found {found}"
            ),
            Self::ScaleMismatch {
                expected,
                found,
                account,
            } => write!(
                f,
                "scale mismatch on account '{account}': expected scale {expected}, found {found}"
            ),
            Self::UnbalancedLegs { debits, credits } => write!(
                f,
                "double-entry conservation violated: sum(debits)={debits} != sum(credits)={credits}"
            ),
            Self::InsufficientCreditBalance {
                account,
                available,
                required,
            } => write!(
                f,
                "overdraft blocked on account '{account}' (debits_must_not_exceed_credits): available={available}, required={required}"
            ),
            Self::InsufficientDebitBalance {
                account,
                available,
                required,
            } => write!(
                f,
                "overdraft blocked on account '{account}' (credits_must_not_exceed_debits): available={available}, required={required}"
            ),
            Self::HoldNotFound(h) => write!(f, "pending hold '{h}' not found"),
            Self::HoldAlreadyResolved { hold_id, status } => {
                write!(f, "hold '{hold_id}' is already in status {status:?}")
            }
            Self::CaptureExceedsHold {
                hold_id,
                hold_amount,
                capture_amount,
            } => write!(
                f,
                "capture amount {capture_amount} exceeds pending hold '{hold_id}' amount {hold_amount}"
            ),
            Self::IdempotencyConflict { idempotency_key } => write!(
                f,
                "idempotency conflict: key '{idempotency_key}' was already committed with different transfer parameters"
            ),
            Self::Persistence(e) => write!(f, "ledger WAL persistence error: {e}"),
            Self::Syntax(e) => write!(f, "ledger SQL syntax error: {e}"),
        }
    }
}

impl LedgerError {
    pub fn sqlstate(&self) -> &'static str {
        match self {
            Self::AccountAlreadyExists(_) | Self::IdempotencyConflict { .. } => "23505",
            Self::UnbalancedLegs { .. }
            | Self::InsufficientCreditBalance { .. }
            | Self::InsufficientDebitBalance { .. }
            | Self::SameAccountTransfer(_)
            | Self::CaptureExceedsHold { .. }
            | Self::InvalidAmount(_) => "23514",
            Self::Syntax(_) => "42601",
            _ => "P0001",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WalEvent {
    CreateAccount {
        seq: u64,
        account: LedgerAccount,
        prev_hash: String,
        entry_hash: String,
    },
    Transfer {
        record: LedgerTransferRecord,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateLedgerAccountReq {
    pub account_id: String,
    #[serde(default = "default_ledger_id")]
    pub ledger_id: u32,
    #[serde(default = "default_code")]
    pub code: u16,
    #[serde(default = "default_scale")]
    pub scale: u8,
    #[serde(default)]
    pub flags: AccountFlags,
}

fn default_ledger_id() -> u32 {
    1
}
fn default_code() -> u16 {
    1000
}
fn default_scale() -> u8 {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateTransferReq {
    pub idempotency_key: String,
    #[serde(default)]
    pub transfer_id: Option<String>,
    #[serde(default)]
    pub debit_account_id: Option<String>,
    #[serde(default)]
    pub credit_account_id: Option<String>,
    #[serde(default)]
    pub amount_minor: Option<i128>,
    #[serde(default)]
    pub legs: Vec<LedgerLeg>,
    #[serde(default)]
    pub hold: bool,
    #[serde(default)]
    pub memo: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResolveHoldReq {
    pub idempotency_key: String,
    #[serde(default)]
    pub capture_amount_minor: Option<i128>,
    #[serde(default)]
    pub memo: Option<String>,
}

fn default_atomic_true() -> bool {
    true
}

fn default_include_receipts_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct BatchTransferReq {
    pub transfers: Vec<CreateTransferReq>,
    #[serde(default = "default_atomic_true")]
    pub atomic: bool,
    #[serde(default = "default_include_receipts_true")]
    pub include_receipts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTransferItemOutcome {
    pub index: usize,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<TransferReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sqlstate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTransferReport {
    pub atomic: bool,
    pub submitted: usize,
    pub committed: usize,
    pub idempotent_replays: usize,
    pub rejected: usize,
    pub fsync_calls: usize,
    pub wal_bytes_written: usize,
    pub head_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<BatchTransferItemOutcome>,
}

enum WaiterState {
    Waiting(Option<CreateTransferReq>),
    PromotedToLeader(CreateTransferReq),
    Done(Result<TransferReceipt, LedgerError>),
}

type GroupWaiterSlot = Arc<(Mutex<WaiterState>, Condvar)>;

struct GroupCommitQueue {
    leader_active: bool,
    target_wave: usize,
    pending: VecDeque<GroupWaiterSlot>,
}

impl Default for GroupCommitQueue {
    fn default() -> Self {
        Self {
            leader_active: false,
            target_wave: 1,
            pending: VecDeque::new(),
        }
    }
}

struct LedgerState {
    accounts: Vec<LedgerAccount>,
    account_index: FxHashMap<String, usize>,
    transfers: Vec<LedgerTransferRecord>,
    holds_by_id: FxHashMap<String, usize>,
    idempotency_index: FxHashMap<String, usize>,
    wal_events: Vec<WalEvent>,
    seq: u64,
    head_hash: String,
    wal_file: Option<File>,
    wal_write_pos: u64,
    wal_alloc_len: u64,
    wal_path: Option<PathBuf>,
}

/// Thread-safe, strictly serialized Double-Entry Financial Ledger Engine
/// with pre-allocated binary `JWL1` CRC-32 frames and adaptive FIFO group-commit coalescing.
pub struct FinancialLedger {
    inner: Mutex<LedgerState>,
    group_commit: (Mutex<GroupCommitQueue>, Condvar),
    persistent: bool,
}

impl Default for FinancialLedger {
    fn default() -> Self {
        Self::new_in_memory()
    }
}

impl FinancialLedger {
    pub fn new_in_memory() -> Self {
        Self {
            inner: Mutex::new(LedgerState {
                accounts: Vec::new(),
                account_index: FxHashMap::default(),
                transfers: Vec::new(),
                holds_by_id: FxHashMap::default(),
                idempotency_index: FxHashMap::default(),
                wal_events: Vec::new(),
                seq: 0,
                head_hash: "0".repeat(64),
                wal_file: None,
                wal_write_pos: 0,
                wal_alloc_len: 0,
                wal_path: None,
            }),
            group_commit: (Mutex::new(GroupCommitQueue::default()), Condvar::new()),
            persistent: false,
        }
    }

    #[inline]
    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    pub fn open(dir_or_file: impl AsRef<Path>) -> Result<Self, LedgerError> {
        let p = dir_or_file.as_ref();
        let wal_path = if p.extension().is_some_and(|e| e == "jwal") {
            p.to_path_buf()
        } else {
            std::fs::create_dir_all(p).map_err(|e| LedgerError::Persistence(e.to_string()))?;
            p.join(LEDGER_WAL_FILENAME)
        };
        if let Some(parent) = wal_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| LedgerError::Persistence(e.to_string()))?;
        }

        let mut state = LedgerState {
            accounts: Vec::new(),
            account_index: FxHashMap::default(),
            transfers: Vec::new(),
            holds_by_id: FxHashMap::default(),
            idempotency_index: FxHashMap::default(),
            wal_events: Vec::new(),
            seq: 0,
            head_hash: "0".repeat(64),
            wal_file: None,
            wal_write_pos: 0,
            wal_alloc_len: 0,
            wal_path: Some(wal_path.clone()),
        };

        let mut initial_write_pos = 0u64;
        let mut initial_alloc_len = 0u64;
        if wal_path.exists() {
            let mut raw = Vec::new();
            let mut f =
                File::open(&wal_path).map_err(|e| LedgerError::Persistence(e.to_string()))?;
            f.read_to_end(&mut raw)
                .map_err(|e| LedgerError::Persistence(e.to_string()))?;
            initial_alloc_len = raw.len() as u64;
            let (events, valid_end_pos) = decode_wal_stream(&raw)?;
            initial_write_pos = valid_end_pos as u64;
            for ev in events {
                replay_wal_event(&mut state, ev)?;
            }
        }

        // Open WITHOUT O_APPEND so writes overwrite pre-initialized WAL extents in-place
        // without changing `i_size` on every commit (avoiding Linux ext4 jbd2 journal stalls).
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&wal_path)
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        file.seek(SeekFrom::Start(initial_write_pos))
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        state.wal_write_pos = initial_write_pos;
        state.wal_alloc_len = initial_alloc_len;
        state.wal_file = Some(file);
        Ok(Self {
            inner: Mutex::new(state),
            group_commit: (Mutex::new(GroupCommitQueue::default()), Condvar::new()),
            persistent: true,
        })
    }

    pub fn create_account(
        &self,
        req: CreateLedgerAccountReq,
    ) -> Result<LedgerAccount, LedgerError> {
        let id = req.account_id.trim().to_string();
        if id.is_empty() {
            return Err(LedgerError::Syntax("account_id cannot be empty".into()));
        }
        if req.scale > 18 {
            return Err(LedgerError::InvalidAmount("scale must be <= 18".into()));
        }
        let mut st = self
            .inner
            .lock()
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        if st.account_index.contains_key(&id) {
            return Err(LedgerError::AccountAlreadyExists(id));
        }

        let next_seq = st.seq + 1;
        let account = LedgerAccount {
            account_id: id.clone(),
            ledger_id: req.ledger_id,
            code: req.code,
            scale: req.scale,
            flags: req.flags,
            posted_debits: 0,
            posted_credits: 0,
            pending_debits: 0,
            pending_credits: 0,
            created_at_ns: now_ns(),
            version: 1,
        };
        let prev_hash = st.head_hash.clone();
        let entry_hash = compute_account_event_hash(&prev_hash, next_seq, &account);
        let ev = WalEvent::CreateAccount {
            seq: next_seq,
            account: account.clone(),
            prev_hash,
            entry_hash: entry_hash.clone(),
        };
        persist_wal_event(&mut st, &ev)?;
        st.seq = next_seq;
        st.head_hash = entry_hash;
        let acct_idx = st.accounts.len();
        st.accounts.push(account.clone());
        st.account_index.insert(id, acct_idx);
        st.wal_events.push(ev);
        Ok(account)
    }

    pub fn list_accounts(&self) -> Vec<LedgerAccount> {
        self.inner
            .lock()
            .map(|st| {
                let mut out = st.accounts.clone();
                out.sort_by(|a, b| a.account_id.cmp(&b.account_id));
                out
            })
            .unwrap_or_default()
    }

    pub fn list_transfers(&self) -> Vec<LedgerTransferRecord> {
        self.inner
            .lock()
            .map(|st| st.transfers.clone())
            .unwrap_or_default()
    }

    pub fn submit_transfer(&self, req: CreateTransferReq) -> Result<TransferReceipt, LedgerError> {
        if !self.persistent {
            let mut st = self
                .inner
                .lock()
                .map_err(|e| LedgerError::Persistence(e.to_string()))?;
            let (mut outcomes, _, _) = execute_transfer_wave_locked(&mut st, vec![req], true)?;
            return match outcomes.pop() {
                Some(Ok(r)) => Ok(r),
                Some(Err(e)) => Err(e),
                None => Err(LedgerError::Persistence(
                    "empty single-transfer wave".into(),
                )),
            };
        }

        // Persistent WAL path: FIFO Single-Wave Leader-Follower Group Commit.
        // At concurrency = 1, leader_active is false -> 0 allocations, 0 wait, 1 fdatasync.
        // Under concurrency > 1, followers arriving during NVMe fdatasync coalesce into the
        // next single-fdatasync wave with strict FIFO leadership handoff (zero starvation).
        let mut leader_req = {
            let mut q = self
                .group_commit
                .0
                .lock()
                .map_err(|e| LedgerError::Persistence(e.to_string()))?;
            if !q.leader_active {
                q.leader_active = true;
                Some(req)
            } else {
                let slot: GroupWaiterSlot =
                    Arc::new((Mutex::new(WaiterState::Waiting(Some(req))), Condvar::new()));
                q.pending.push_back(slot.clone());
                let ready_len = q.pending.len().saturating_add(1);
                let target = q.target_wave;
                drop(q);
                if ready_len >= target {
                    self.group_commit.1.notify_one();
                }

                let (lock, cvar) = &*slot;
                let mut state = lock
                    .lock()
                    .map_err(|e| LedgerError::Persistence(e.to_string()))?;
                loop {
                    match &mut *state {
                        WaiterState::Waiting(_) => {
                            state = cvar
                                .wait(state)
                                .map_err(|e| LedgerError::Persistence(e.to_string()))?;
                        }
                        WaiterState::Done(_) => {
                            let prev = std::mem::replace(&mut *state, WaiterState::Waiting(None));
                            return match prev {
                                WaiterState::Done(res) => res,
                                _ => Err(LedgerError::Persistence(
                                    "invalid waiter terminal state".into(),
                                )),
                            };
                        }
                        WaiterState::PromotedToLeader(_) => {
                            let prev = std::mem::replace(&mut *state, WaiterState::Waiting(None));
                            break match prev {
                                WaiterState::PromotedToLeader(r) => Some(r),
                                _ => None,
                            };
                        }
                    }
                }
            }
        };

        let my_req = match leader_req.take() {
            Some(r) => r,
            None => {
                self.handoff_group_leader();
                return Err(LedgerError::Persistence(
                    "missing promoted leader request".into(),
                ));
            }
        };

        // Adaptive wave re-coalescing: at c=1 (`target_wave == 1`), waits 0 ns.
        // Under active concurrency (`target_wave >= 2`), prevents convoy splitting across two
        // NVMe fdatasync calls by waking immediately via `group_commit.1` once returning peers enqueue.
        let mut follower_slots: Vec<GroupWaiterSlot> = Vec::new();
        let mut wave_reqs: Vec<CreateTransferReq> = Vec::new();
        wave_reqs.push(my_req);
        if let Ok(mut q) = self.group_commit.0.lock() {
            if q.target_wave >= 2 && q.pending.len().saturating_add(1) < q.target_wave {
                let deadline = Instant::now() + std::time::Duration::from_micros(1200);
                while q.pending.len().saturating_add(1) < q.target_wave {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    match self.group_commit.1.wait_timeout(q, deadline - now) {
                        Ok((guard, timeout_res)) => {
                            q = guard;
                            if timeout_res.timed_out() {
                                break;
                            }
                        }
                        Err(_) => {
                            return Err(LedgerError::Persistence(
                                "group commit condvar poisoned".into(),
                            ));
                        }
                    }
                }
            }
            while wave_reqs.len() < MAX_LEDGER_BATCH_SIZE {
                let Some(slot) = q.pending.pop_front() else {
                    break;
                };
                let extracted = slot.0.lock().ok().and_then(|mut g| match &mut *g {
                    WaiterState::Waiting(opt) => opt.take(),
                    _ => None,
                });
                if let Some(freq) = extracted {
                    wave_reqs.push(freq);
                    follower_slots.push(slot);
                } else {
                    if let Ok(mut g) = slot.0.lock() {
                        *g = WaiterState::Done(Err(LedgerError::Persistence(
                            "corrupt follower slot".into(),
                        )));
                    }
                    slot.1.notify_one();
                }
            }
        }

        let wave_len = wave_reqs.len();
        let wave_res = match self.inner.lock() {
            Ok(mut st) => execute_transfer_wave_locked(&mut st, wave_reqs, false),
            Err(e) => Err(LedgerError::Persistence(e.to_string())),
        };

        if let Ok(mut q) = self.group_commit.0.lock() {
            q.target_wave = wave_len
                .saturating_add(q.pending.len())
                .clamp(1, MAX_LEDGER_BATCH_SIZE);
        }

        let my_result = match wave_res {
            Ok((outcomes, _, _)) => {
                let mut iter = outcomes.into_iter();
                let leader_out = iter.next().unwrap_or_else(|| {
                    Err(LedgerError::Persistence("missing leader outcome".into()))
                });
                for (slot, out) in follower_slots.into_iter().zip(iter) {
                    if let Ok(mut g) = slot.0.lock() {
                        *g = WaiterState::Done(out);
                    }
                    slot.1.notify_one();
                }
                leader_out
            }
            Err(batch_err) => {
                for slot in follower_slots {
                    if let Ok(mut g) = slot.0.lock() {
                        *g = WaiterState::Done(Err(batch_err.clone()));
                    }
                    slot.1.notify_one();
                }
                Err(batch_err)
            }
        };

        self.handoff_group_leader();
        my_result
    }

    fn handoff_group_leader(&self) {
        if let Ok(mut q) = self.group_commit.0.lock() {
            while let Some(next_slot) = q.pending.pop_front() {
                let promoted = next_slot.0.lock().ok().and_then(|mut g| {
                    let req_opt = match &mut *g {
                        WaiterState::Waiting(opt) => opt.take(),
                        _ => None,
                    };
                    if let Some(r) = req_opt {
                        *g = WaiterState::PromotedToLeader(r);
                        Some(())
                    } else {
                        None
                    }
                });
                if promoted.is_some() {
                    next_slot.1.notify_one();
                    return;
                }
                if let Ok(mut g) = next_slot.0.lock() {
                    *g = WaiterState::Done(Err(LedgerError::Persistence(
                        "dropped empty waiter".into(),
                    )));
                }
                next_slot.1.notify_one();
            }
            q.leader_active = false;
        }
    }

    /// Execute up to `32,768` double-entry transfers in a single lock acquisition and
    /// a **single** `JWL1` binary WAL `write_all` + `fdatasync`.
    ///
    /// - `atomic = true`: All-or-nothing ACID journal batch; if any item fails validation
    ///   (e.g. `NO_OVERDRAFT` or `IdempotencyConflict`), zero state changes or WAL bytes commit.
    /// - `atomic = false`: High-throughput per-item core-banking clearing batch; valid items
    ///   commit together in 1 `fdatasync` while rejected items return item-level `sqlstate` errors.
    pub fn submit_transfer_batch(
        &self,
        reqs: Vec<CreateTransferReq>,
        atomic: bool,
    ) -> Result<BatchTransferReport, LedgerError> {
        self.submit_transfer_batch_with_options(reqs, atomic, true)
    }

    pub fn submit_transfer_batch_with_options(
        &self,
        reqs: Vec<CreateTransferReq>,
        atomic: bool,
        include_receipts: bool,
    ) -> Result<BatchTransferReport, LedgerError> {
        if reqs.is_empty() {
            return Err(LedgerError::Syntax(
                "transfers batch cannot be empty".into(),
            ));
        }
        if reqs.len() > MAX_LEDGER_BATCH_SIZE {
            return Err(LedgerError::Syntax(format!(
                "batch size {} exceeds MAX_LEDGER_BATCH_SIZE ({MAX_LEDGER_BATCH_SIZE})",
                reqs.len()
            )));
        }
        let submitted = reqs.len();
        let mut st = self
            .inner
            .lock()
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        let (outcomes, fsync_calls, wal_bytes_written) =
            execute_transfer_wave_locked(&mut st, reqs, atomic)?;
        let head_hash = st.head_hash.clone();
        drop(st);

        let mut committed = 0usize;
        let mut idempotent_replays = 0usize;
        let mut rejected = 0usize;
        let mut results = if include_receipts {
            Vec::with_capacity(submitted)
        } else {
            Vec::new()
        };

        for (index, out) in outcomes.into_iter().enumerate() {
            match out {
                Ok(receipt) => {
                    if receipt.idempotent_replay {
                        idempotent_replays += 1;
                    } else {
                        committed += 1;
                    }
                    if include_receipts {
                        results.push(BatchTransferItemOutcome {
                            index,
                            ok: true,
                            receipt: Some(receipt),
                            error: None,
                            sqlstate: None,
                        });
                    }
                }
                Err(err) => {
                    if atomic {
                        return Err(err);
                    }
                    rejected += 1;
                    if include_receipts {
                        results.push(BatchTransferItemOutcome {
                            index,
                            ok: false,
                            receipt: None,
                            error: Some(err.to_string()),
                            sqlstate: Some(err.sqlstate().to_string()),
                        });
                    }
                }
            }
        }

        Ok(BatchTransferReport {
            atomic,
            submitted,
            committed,
            idempotent_replays,
            rejected,
            fsync_calls,
            wal_bytes_written,
            head_hash,
            results,
        })
    }

    pub fn post_hold(
        &self,
        hold_id: &str,
        req: ResolveHoldReq,
    ) -> Result<TransferReceipt, LedgerError> {
        self.resolve_hold_inner(hold_id, req, true)
    }

    pub fn void_hold(
        &self,
        hold_id: &str,
        req: ResolveHoldReq,
    ) -> Result<TransferReceipt, LedgerError> {
        self.resolve_hold_inner(hold_id, req, false)
    }

    fn resolve_hold_inner(
        &self,
        hold_id: &str,
        req: ResolveHoldReq,
        capture: bool,
    ) -> Result<TransferReceipt, LedgerError> {
        let idem = req.idempotency_key.trim().to_string();
        if idem.is_empty() {
            return Err(LedgerError::Syntax("idempotency_key is mandatory".into()));
        }
        let kind = if capture {
            TransferKind::PostHold
        } else {
            TransferKind::VoidHold
        };
        let payload_hash = compute_transfer_payload_hash(
            kind,
            &[],
            Some(hold_id),
            if capture {
                req.capture_amount_minor
            } else {
                None
            },
        );

        let mut st = self
            .inner
            .lock()
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;

        if let Some(&existing_idx) = st.idempotency_index.get(&idem)
            && let Some(existing) = st.transfers.get(existing_idx)
        {
            if existing.payload_hash == payload_hash {
                return Ok(TransferReceipt {
                    record: existing.clone(),
                    idempotent_replay: true,
                });
            }
            return Err(LedgerError::IdempotencyConflict {
                idempotency_key: idem,
            });
        }

        let hold_idx = st
            .holds_by_id
            .get(hold_id)
            .copied()
            .ok_or_else(|| LedgerError::HoldNotFound(hold_id.to_string()))?;
        let hold_rec = st
            .transfers
            .get(hold_idx)
            .cloned()
            .ok_or_else(|| LedgerError::HoldNotFound(hold_id.to_string()))?;

        if hold_rec.status != TransferStatus::Pending {
            return Err(LedgerError::HoldAlreadyResolved {
                hold_id: hold_id.to_string(),
                status: hold_rec.status,
            });
        }

        let capture_amt = if capture {
            let amt = req
                .capture_amount_minor
                .unwrap_or(hold_rec.total_amount_minor);
            if amt < 0 {
                return Err(LedgerError::InvalidAmount(
                    "capture_amount_minor cannot be negative".into(),
                ));
            }
            if amt > hold_rec.total_amount_minor {
                return Err(LedgerError::CaptureExceedsHold {
                    hold_id: hold_id.to_string(),
                    hold_amount: hold_rec.total_amount_minor,
                    capture_amount: amt,
                });
            }
            Some(amt)
        } else {
            None
        };

        let next_seq = st.seq + 1;
        let prev_hash = st.head_hash.clone();
        let mut record = LedgerTransferRecord {
            seq: next_seq,
            transfer_id: format!("tx_{next_seq:08}"),
            idempotency_key: idem.clone(),
            kind,
            status: if capture {
                TransferStatus::Captured
            } else {
                TransferStatus::Voided
            },
            ledger_id: hold_rec.ledger_id,
            scale: hold_rec.scale,
            legs: hold_rec.legs.clone(),
            total_amount_minor: hold_rec.total_amount_minor,
            captured_amount_minor: capture_amt,
            parent_hold_id: Some(hold_id.to_string()),
            memo: req.memo,
            payload_hash,
            prev_hash,
            entry_hash: String::new(),
            timestamp_ns: now_ns(),
        };
        record.entry_hash = compute_transfer_entry_hash(&record);

        let ev = WalEvent::Transfer {
            record: record.clone(),
        };
        persist_wal_event(&mut st, &ev)?;

        apply_transfer_to_accounts(&mut st, &record)?;
        let idx = st.transfers.len();
        st.seq = next_seq;
        st.head_hash = record.entry_hash.clone();
        st.idempotency_index.insert(idem, idx);
        st.transfers.push(record.clone());
        st.wal_events.push(ev);

        Ok(TransferReceipt {
            record,
            idempotent_replay: false,
        })
    }

    /// Recomputes every SHA-256 chain link from genesis and verifies global double-entry conservation.
    pub fn verify_chain(&self) -> ChainVerificationReport {
        let Ok(st) = self.inner.lock() else {
            return ChainVerificationReport {
                valid: false,
                accounts_count: 0,
                journal_entries_verified: 0,
                total_posted_debits: 0,
                total_posted_credits: 0,
                total_pending_debits: 0,
                total_pending_credits: 0,
                head_hash: String::new(),
                error: Some("ledger mutex poisoned".into()),
            };
        };

        let mut expected_prev = "0".repeat(64);
        let mut expected_seq = 0u64;
        for ev in &st.wal_events {
            expected_seq += 1;
            match ev {
                WalEvent::CreateAccount {
                    seq,
                    account,
                    prev_hash,
                    entry_hash,
                } => {
                    if *seq != expected_seq || *prev_hash != expected_prev {
                        return ChainVerificationReport {
                            valid: false,
                            accounts_count: st.accounts.len(),
                            journal_entries_verified: (expected_seq - 1) as usize,
                            total_posted_debits: 0,
                            total_posted_credits: 0,
                            total_pending_debits: 0,
                            total_pending_credits: 0,
                            head_hash: st.head_hash.clone(),
                            error: Some(format!("chain discontinuity at seq {seq}")),
                        };
                    }
                    let recomputed = compute_account_event_hash(prev_hash, *seq, account);
                    if &recomputed != entry_hash {
                        return ChainVerificationReport {
                            valid: false,
                            accounts_count: st.accounts.len(),
                            journal_entries_verified: (expected_seq - 1) as usize,
                            total_posted_debits: 0,
                            total_posted_credits: 0,
                            total_pending_debits: 0,
                            total_pending_credits: 0,
                            head_hash: st.head_hash.clone(),
                            error: Some(format!("hash tamper detected at seq {seq}")),
                        };
                    }
                    expected_prev = entry_hash.clone();
                }
                WalEvent::Transfer { record } => {
                    if record.seq != expected_seq || record.prev_hash != expected_prev {
                        return ChainVerificationReport {
                            valid: false,
                            accounts_count: st.accounts.len(),
                            journal_entries_verified: (expected_seq - 1) as usize,
                            total_posted_debits: 0,
                            total_posted_credits: 0,
                            total_pending_debits: 0,
                            total_pending_credits: 0,
                            head_hash: st.head_hash.clone(),
                            error: Some(format!("chain discontinuity at seq {}", record.seq)),
                        };
                    }
                    let recomputed = compute_transfer_entry_hash(record);
                    if recomputed != record.entry_hash {
                        return ChainVerificationReport {
                            valid: false,
                            accounts_count: st.accounts.len(),
                            journal_entries_verified: (expected_seq - 1) as usize,
                            total_posted_debits: 0,
                            total_posted_credits: 0,
                            total_pending_debits: 0,
                            total_pending_credits: 0,
                            head_hash: st.head_hash.clone(),
                            error: Some(format!("hash tamper detected at seq {}", record.seq)),
                        };
                    }
                    expected_prev = record.entry_hash.clone();
                }
            }
        }

        let mut total_pd = 0i128;
        let mut total_pc = 0i128;
        let mut total_pend_d = 0i128;
        let mut total_pend_c = 0i128;
        for acct in &st.accounts {
            total_pd = total_pd.saturating_add(acct.posted_debits);
            total_pc = total_pc.saturating_add(acct.posted_credits);
            total_pend_d = total_pend_d.saturating_add(acct.pending_debits);
            total_pend_c = total_pend_c.saturating_add(acct.pending_credits);
        }

        let conserved = total_pd == total_pc && total_pend_d == total_pend_c;
        ChainVerificationReport {
            valid: conserved && expected_prev == st.head_hash,
            accounts_count: st.accounts.len(),
            journal_entries_verified: st.wal_events.len(),
            total_posted_debits: total_pd,
            total_posted_credits: total_pc,
            total_pending_debits: total_pend_d,
            total_pending_credits: total_pend_c,
            head_hash: st.head_hash.clone(),
            error: (!conserved).then(|| {
                format!(
                    "global conservation mismatch: posted({total_pd} vs {total_pc}), pending({total_pend_d} vs {total_pend_c})"
                )
            }),
        }
    }

    /// Virtual SQL table projection (`SELECT * FROM jouledb_ledger_accounts` / `jouledb_ledger_transfers` / `jouledb_ledger_legs`).
    pub fn try_virtual_table(
        &self,
        table_name: &str,
    ) -> Option<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
        let lower = table_name.to_ascii_lowercase();
        let name = lower.strip_prefix("public.").unwrap_or(&lower);
        match name {
            "jouledb_ledger_accounts" => {
                let cols = vec![
                    "account_id".into(),
                    "ledger_id".into(),
                    "code".into(),
                    "scale".into(),
                    "posted_debits".into(),
                    "posted_credits".into(),
                    "pending_debits".into(),
                    "pending_credits".into(),
                    "net_credits_minor".into(),
                    "available_credits_minor".into(),
                    "formatted_balance".into(),
                    "no_overdraft_credit".into(),
                    "no_overdraft_debit".into(),
                    "frozen".into(),
                ];
                let rows = self
                    .list_accounts()
                    .into_iter()
                    .map(|a| {
                        let fmt = a.format_net_credits();
                        let net_c = a.net_credits_minor();
                        let avail_c = a.available_credits_minor();
                        vec![
                            serde_json::json!(a.account_id),
                            serde_json::json!(a.ledger_id),
                            serde_json::json!(a.code),
                            serde_json::json!(a.scale),
                            i128_json(a.posted_debits),
                            i128_json(a.posted_credits),
                            i128_json(a.pending_debits),
                            i128_json(a.pending_credits),
                            i128_json(net_c),
                            i128_json(avail_c),
                            serde_json::json!(fmt),
                            serde_json::json!(a.flags.debits_must_not_exceed_credits),
                            serde_json::json!(a.flags.credits_must_not_exceed_debits),
                            serde_json::json!(a.flags.frozen),
                        ]
                    })
                    .collect();
                Some((cols, rows))
            }
            "jouledb_ledger_transfers" => {
                let cols = vec![
                    "seq".into(),
                    "transfer_id".into(),
                    "idempotency_key".into(),
                    "kind".into(),
                    "status".into(),
                    "ledger_id".into(),
                    "debit_account".into(),
                    "credit_account".into(),
                    "amount_minor".into(),
                    "captured_amount_minor".into(),
                    "parent_hold_id".into(),
                    "memo".into(),
                    "prev_hash".into(),
                    "entry_hash".into(),
                ];
                let rows = self
                    .list_transfers()
                    .into_iter()
                    .map(|t| {
                        let first_debit = t
                            .legs
                            .iter()
                            .find(|l| l.direction == LegDirection::Debit)
                            .map(|l| l.account_id.clone())
                            .unwrap_or_default();
                        let first_credit = t
                            .legs
                            .iter()
                            .find(|l| l.direction == LegDirection::Credit)
                            .map(|l| l.account_id.clone())
                            .unwrap_or_default();
                        vec![
                            serde_json::json!(t.seq),
                            serde_json::json!(t.transfer_id),
                            serde_json::json!(t.idempotency_key),
                            serde_json::json!(format!("{:?}", t.kind).to_ascii_lowercase()),
                            serde_json::json!(format!("{:?}", t.status).to_ascii_lowercase()),
                            serde_json::json!(t.ledger_id),
                            serde_json::json!(first_debit),
                            serde_json::json!(first_credit),
                            i128_json(t.total_amount_minor),
                            t.captured_amount_minor
                                .map(i128_json)
                                .unwrap_or(serde_json::Value::Null),
                            serde_json::json!(t.parent_hold_id),
                            serde_json::json!(t.memo),
                            serde_json::json!(t.prev_hash),
                            serde_json::json!(t.entry_hash),
                        ]
                    })
                    .collect();
                Some((cols, rows))
            }
            "jouledb_ledger_legs" => {
                let cols = vec![
                    "seq".into(),
                    "transfer_id".into(),
                    "leg_index".into(),
                    "account_id".into(),
                    "direction".into(),
                    "amount_minor".into(),
                    "status".into(),
                ];
                let mut rows = Vec::new();
                for t in self.list_transfers() {
                    let st_str = format!("{:?}", t.status).to_ascii_lowercase();
                    for (idx, leg) in t.legs.into_iter().enumerate() {
                        rows.push(vec![
                            serde_json::json!(t.seq),
                            serde_json::json!(t.transfer_id),
                            serde_json::json!(idx as i64),
                            serde_json::json!(leg.account_id),
                            serde_json::json!(match leg.direction {
                                LegDirection::Debit => "debit",
                                LegDirection::Credit => "credit",
                            }),
                            i128_json(leg.amount_minor),
                            serde_json::json!(st_str),
                        ]);
                    }
                }
                Some((cols, rows))
            }
            _ => None,
        }
    }
}

pub fn is_ledger_sql(sql: &str) -> bool {
    let t = sql.trim_start();
    t.len() >= 7
        && t.as_bytes()
            .get(..6)
            .is_some_and(|p| p.eq_ignore_ascii_case(b"LEDGER"))
        && t.as_bytes().get(6).is_some_and(|b| b.is_ascii_whitespace())
}

/// Execute a native `LEDGER ...` SQL statement.
pub fn execute_ledger_sql(
    ledger: &FinancialLedger,
    sql: &str,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    let rest = trimmed
        .get(6..)
        .map(str::trim)
        .ok_or_else(|| QueryErrorResponse::syntax_error("incomplete LEDGER command", 1, 1))?;
    let tokens = tokenize_ledger_cmd(rest);
    if tokens.is_empty() {
        return Err(QueryErrorResponse::syntax_error(
            "expected LEDGER subcommand",
            1,
            1,
        ));
    }

    let sub0 = tokens[0].to_ascii_uppercase();
    match sub0.as_str() {
        "VERIFY" => {
            let rep = ledger.verify_chain();
            Ok(QueryResponse {
                columns: vec![
                    "valid".into(),
                    "accounts_count".into(),
                    "entries_verified".into(),
                    "total_posted_debits".into(),
                    "total_posted_credits".into(),
                    "total_pending_debits".into(),
                    "total_pending_credits".into(),
                    "head_hash".into(),
                ],
                rows: vec![vec![
                    serde_json::json!(rep.valid),
                    serde_json::json!(rep.accounts_count as i64),
                    serde_json::json!(rep.journal_entries_verified as i64),
                    i128_json(rep.total_posted_debits),
                    i128_json(rep.total_posted_credits),
                    i128_json(rep.total_pending_debits),
                    i128_json(rep.total_pending_credits),
                    serde_json::json!(rep.head_hash),
                ]],
                affected_rows: None,
                execution_time_ms: start.elapsed().as_millis() as u64,
                truncated: false,
                warnings: rep.error.into_iter().collect(),
                energy_joules: None,
                power_watts: None,
                device_target: None,
                algorithm_type: Some("double-entry-sha256-verify".into()),
                session_id: None,
                viz_hint: None,
                energy_receipt: None,
            })
        }
        "CREATE" => {
            // LEDGER CREATE ACCOUNT <id> [LEDGER <u32>] [CODE <u16>] [SCALE <u8>] [NO_OVERDRAFT | DEBITS_MUST_NOT_EXCEED_CREDITS | CREDITS_MUST_NOT_EXCEED_DEBITS]
            if tokens
                .get(1)
                .is_none_or(|t| !t.eq_ignore_ascii_case("ACCOUNT"))
            {
                return Err(QueryErrorResponse::syntax_error(
                    "expected LEDGER CREATE ACCOUNT <id> ...",
                    1,
                    1,
                ));
            }
            let acct_id = tokens.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("missing account_id after CREATE ACCOUNT", 1, 1)
            })?;
            let mut ledger_id = 1u32;
            let mut code = 1000u16;
            let mut scale = 2u8;
            let mut flags = AccountFlags::default();
            let mut i = 3usize;
            while i < tokens.len() {
                match tokens[i].to_ascii_uppercase().as_str() {
                    "LEDGER" => {
                        i += 1;
                        ledger_id =
                            tokens.get(i).and_then(|s| s.parse().ok()).ok_or_else(|| {
                                QueryErrorResponse::syntax_error("invalid LEDGER u32", 1, 1)
                            })?;
                    }
                    "CODE" => {
                        i += 1;
                        code = tokens.get(i).and_then(|s| s.parse().ok()).ok_or_else(|| {
                            QueryErrorResponse::syntax_error("invalid CODE u16", 1, 1)
                        })?;
                    }
                    "SCALE" => {
                        i += 1;
                        scale = tokens.get(i).and_then(|s| s.parse().ok()).ok_or_else(|| {
                            QueryErrorResponse::syntax_error("invalid SCALE u8", 1, 1)
                        })?;
                    }
                    "NO_OVERDRAFT" | "NO_OVERDRAFT_CREDIT" | "DEBITS_MUST_NOT_EXCEED_CREDITS" => {
                        flags.debits_must_not_exceed_credits = true;
                    }
                    "NO_OVERDRAFT_DEBIT" | "CREDITS_MUST_NOT_EXCEED_DEBITS" => {
                        flags.credits_must_not_exceed_debits = true;
                    }
                    "FROZEN" => {
                        flags.frozen = true;
                    }
                    other => {
                        return Err(QueryErrorResponse::syntax_error(
                            &format!("unexpected option '{other}' in LEDGER CREATE ACCOUNT"),
                            1,
                            1,
                        ));
                    }
                }
                i += 1;
            }
            let acct = ledger
                .create_account(CreateLedgerAccountReq {
                    account_id: acct_id,
                    ledger_id,
                    code,
                    scale,
                    flags,
                })
                .map_err(ledger_err_to_query_err)?;
            Ok(QueryResponse {
                columns: vec![
                    "account_id".into(),
                    "ledger_id".into(),
                    "code".into(),
                    "scale".into(),
                ],
                rows: vec![vec![
                    serde_json::json!(acct.account_id),
                    serde_json::json!(acct.ledger_id),
                    serde_json::json!(acct.code),
                    serde_json::json!(acct.scale),
                ]],
                affected_rows: Some(1),
                execution_time_ms: start.elapsed().as_millis() as u64,
                truncated: false,
                warnings: Vec::new(),
                energy_joules: None,
                power_watts: None,
                device_target: None,
                algorithm_type: Some("double-entry-ledger".into()),
                session_id: None,
                viz_hint: None,
                energy_receipt: None,
            })
        }
        "TRANSFER" | "HOLD" => {
            // LEDGER TRANSFER|HOLD <amount_minor> FROM <debit> TO <credit> IDEMPOTENCY [KEY] '<key>' [HOLD_ID '<hid>'] [MEMO '<memo>']
            let is_hold = sub0 == "HOLD";
            let amt_minor: i128 = tokens
                .get(1)
                .and_then(|s| s.replace('_', "").parse::<i128>().ok())
                .ok_or_else(|| {
                    QueryErrorResponse::syntax_error(
                        "expected integer minor-unit amount after TRANSFER/HOLD",
                        1,
                        1,
                    )
                })?;
            let mut from_acct = None;
            let mut to_acct = None;
            let mut idem_key = None;
            let mut hold_id = None;
            let mut memo = None;
            let mut i = 2usize;
            while i < tokens.len() {
                match tokens[i].to_ascii_uppercase().as_str() {
                    "FROM" => {
                        i += 1;
                        from_acct = tokens.get(i).cloned();
                    }
                    "TO" => {
                        i += 1;
                        to_acct = tokens.get(i).cloned();
                    }
                    "IDEMPOTENCY" => {
                        i += 1;
                        if tokens.get(i).is_some_and(|t| t.eq_ignore_ascii_case("KEY")) {
                            i += 1;
                        }
                        idem_key = tokens.get(i).cloned();
                    }
                    "HOLD_ID" | "TRANSFER_ID" => {
                        i += 1;
                        hold_id = tokens.get(i).cloned();
                    }
                    "MEMO" => {
                        i += 1;
                        memo = tokens.get(i).cloned();
                    }
                    other => {
                        return Err(QueryErrorResponse::syntax_error(
                            &format!("unexpected token '{other}' in LEDGER {sub0}"),
                            1,
                            1,
                        ));
                    }
                }
                i += 1;
            }
            let receipt = ledger
                .submit_transfer(CreateTransferReq {
                    idempotency_key: idem_key.unwrap_or_default(),
                    transfer_id: hold_id,
                    debit_account_id: from_acct,
                    credit_account_id: to_acct,
                    amount_minor: Some(amt_minor),
                    legs: Vec::new(),
                    hold: is_hold,
                    memo,
                })
                .map_err(ledger_err_to_query_err)?;
            Ok(receipt_to_query_response(receipt, start))
        }
        "MULTILEG" => {
            // LEDGER MULTILEG DEBIT acct1 100 DEBIT acct2 50 CREDIT acct3 150 IDEMPOTENCY KEY 'k1' [MEMO 'm']
            let mut legs = Vec::new();
            let mut idem_key = None;
            let mut memo = None;
            let mut i = 1usize;
            while i < tokens.len() {
                match tokens[i].to_ascii_uppercase().as_str() {
                    "DEBIT" | "CREDIT" => {
                        let dir = if tokens[i].eq_ignore_ascii_case("DEBIT") {
                            LegDirection::Debit
                        } else {
                            LegDirection::Credit
                        };
                        let acct = tokens.get(i + 1).cloned().ok_or_else(|| {
                            QueryErrorResponse::syntax_error("missing account in MULTILEG", 1, 1)
                        })?;
                        let amt = tokens
                            .get(i + 2)
                            .and_then(|s| s.replace('_', "").parse::<i128>().ok())
                            .ok_or_else(|| {
                                QueryErrorResponse::syntax_error("missing amount in MULTILEG", 1, 1)
                            })?;
                        legs.push(LedgerLeg {
                            account_id: acct,
                            direction: dir,
                            amount_minor: amt,
                        });
                        i += 2;
                    }
                    "IDEMPOTENCY" => {
                        i += 1;
                        if tokens.get(i).is_some_and(|t| t.eq_ignore_ascii_case("KEY")) {
                            i += 1;
                        }
                        idem_key = tokens.get(i).cloned();
                    }
                    "MEMO" => {
                        i += 1;
                        memo = tokens.get(i).cloned();
                    }
                    other => {
                        return Err(QueryErrorResponse::syntax_error(
                            &format!("unexpected token '{other}' in LEDGER MULTILEG"),
                            1,
                            1,
                        ));
                    }
                }
                i += 1;
            }
            let receipt = ledger
                .submit_transfer(CreateTransferReq {
                    idempotency_key: idem_key.unwrap_or_default(),
                    transfer_id: None,
                    debit_account_id: None,
                    credit_account_id: None,
                    amount_minor: None,
                    legs,
                    hold: false,
                    memo,
                })
                .map_err(ledger_err_to_query_err)?;
            Ok(receipt_to_query_response(receipt, start))
        }
        "POST" | "VOID" => {
            // LEDGER POST HOLD '<hid>' [AMOUNT <amt>] IDEMPOTENCY KEY '<key>'
            // LEDGER VOID HOLD '<hid>' IDEMPOTENCY KEY '<key>'
            let is_post = sub0 == "POST";
            if tokens
                .get(1)
                .is_none_or(|t| !t.eq_ignore_ascii_case("HOLD"))
            {
                return Err(QueryErrorResponse::syntax_error(
                    &format!("expected LEDGER {sub0} HOLD <hold_id> ..."),
                    1,
                    1,
                ));
            }
            let hid = tokens
                .get(2)
                .cloned()
                .ok_or_else(|| QueryErrorResponse::syntax_error("missing hold_id", 1, 1))?;
            let mut capture_amt = None;
            let mut idem_key = None;
            let mut memo = None;
            let mut i = 3usize;
            while i < tokens.len() {
                match tokens[i].to_ascii_uppercase().as_str() {
                    "AMOUNT" => {
                        i += 1;
                        capture_amt = Some(
                            tokens
                                .get(i)
                                .and_then(|s| s.replace('_', "").parse::<i128>().ok())
                                .ok_or_else(|| {
                                    QueryErrorResponse::syntax_error("invalid capture AMOUNT", 1, 1)
                                })?,
                        );
                    }
                    "IDEMPOTENCY" => {
                        i += 1;
                        if tokens.get(i).is_some_and(|t| t.eq_ignore_ascii_case("KEY")) {
                            i += 1;
                        }
                        idem_key = tokens.get(i).cloned();
                    }
                    "MEMO" => {
                        i += 1;
                        memo = tokens.get(i).cloned();
                    }
                    other => {
                        return Err(QueryErrorResponse::syntax_error(
                            &format!("unexpected token '{other}' in LEDGER {sub0} HOLD"),
                            1,
                            1,
                        ));
                    }
                }
                i += 1;
            }
            let req = ResolveHoldReq {
                idempotency_key: idem_key.unwrap_or_default(),
                capture_amount_minor: capture_amt,
                memo,
            };
            let receipt = if is_post {
                ledger.post_hold(&hid, req)
            } else {
                ledger.void_hold(&hid, req)
            }
            .map_err(ledger_err_to_query_err)?;
            Ok(receipt_to_query_response(receipt, start))
        }
        "BATCH" => {
            // LEDGER BATCH [NON_ATOMIC] TRANSFER <amt> FROM <d> TO <c> IDEMPOTENCY [KEY] '<k>' [MEMO '<m>'] [TRANSFER ...]
            let mut idx = 1usize;
            let mut atomic = true;
            if let Some(tok) = tokens.get(idx) {
                if tok.eq_ignore_ascii_case("NON_ATOMIC") || tok.eq_ignore_ascii_case("CLEARING") {
                    atomic = false;
                    idx += 1;
                } else if tok.eq_ignore_ascii_case("ATOMIC") {
                    atomic = true;
                    idx += 1;
                }
            }
            let mut reqs = Vec::new();
            while idx < tokens.len() {
                let verb = tokens[idx].to_ascii_uppercase();
                if verb != "TRANSFER" && verb != "HOLD" {
                    return Err(QueryErrorResponse::syntax_error(
                        &format!("expected TRANSFER or HOLD inside LEDGER BATCH, got '{verb}'"),
                        1,
                        1,
                    ));
                }
                let is_hold = verb == "HOLD";
                let amt_minor: i128 = tokens
                    .get(idx + 1)
                    .and_then(|s| s.replace('_', "").parse::<i128>().ok())
                    .ok_or_else(|| {
                        QueryErrorResponse::syntax_error(
                            "expected integer minor-unit amount in LEDGER BATCH item",
                            1,
                            1,
                        )
                    })?;
                idx += 2;
                let mut from_acct = None;
                let mut to_acct = None;
                let mut idem_key = None;
                let mut hold_id = None;
                let mut memo = None;
                while idx < tokens.len() {
                    let kw = tokens[idx].to_ascii_uppercase();
                    if kw == "TRANSFER" || kw == "HOLD" {
                        break;
                    }
                    match kw.as_str() {
                        "FROM" => {
                            idx += 1;
                            from_acct = tokens.get(idx).cloned();
                        }
                        "TO" => {
                            idx += 1;
                            to_acct = tokens.get(idx).cloned();
                        }
                        "IDEMPOTENCY" => {
                            idx += 1;
                            if tokens
                                .get(idx)
                                .is_some_and(|t| t.eq_ignore_ascii_case("KEY"))
                            {
                                idx += 1;
                            }
                            idem_key = tokens.get(idx).cloned();
                        }
                        "HOLD_ID" | "TRANSFER_ID" => {
                            idx += 1;
                            hold_id = tokens.get(idx).cloned();
                        }
                        "MEMO" => {
                            idx += 1;
                            memo = tokens.get(idx).cloned();
                        }
                        other => {
                            return Err(QueryErrorResponse::syntax_error(
                                &format!("unexpected token '{other}' in LEDGER BATCH item"),
                                1,
                                1,
                            ));
                        }
                    }
                    idx += 1;
                }
                reqs.push(CreateTransferReq {
                    idempotency_key: idem_key.unwrap_or_default(),
                    transfer_id: hold_id,
                    debit_account_id: from_acct,
                    credit_account_id: to_acct,
                    amount_minor: Some(amt_minor),
                    legs: Vec::new(),
                    hold: is_hold,
                    memo,
                });
            }
            let rep = ledger
                .submit_transfer_batch(reqs, atomic)
                .map_err(ledger_err_to_query_err)?;
            Ok(QueryResponse {
                columns: vec![
                    "submitted".into(),
                    "committed".into(),
                    "idempotent_replays".into(),
                    "rejected".into(),
                    "fsync_calls".into(),
                    "wal_bytes_written".into(),
                    "head_hash".into(),
                ],
                rows: vec![vec![
                    serde_json::json!(rep.submitted as i64),
                    serde_json::json!(rep.committed as i64),
                    serde_json::json!(rep.idempotent_replays as i64),
                    serde_json::json!(rep.rejected as i64),
                    serde_json::json!(rep.fsync_calls as i64),
                    serde_json::json!(rep.wal_bytes_written as i64),
                    serde_json::json!(rep.head_hash),
                ]],
                affected_rows: Some(rep.committed),
                execution_time_ms: start.elapsed().as_millis() as u64,
                truncated: false,
                warnings: Vec::new(),
                energy_joules: None,
                power_watts: None,
                device_target: None,
                algorithm_type: Some("double-entry-jwl1-batch".into()),
                session_id: None,
                viz_hint: None,
                energy_receipt: None,
            })
        }
        other => Err(QueryErrorResponse::syntax_error(
            &format!("unknown LEDGER subcommand '{other}'"),
            1,
            1,
        )),
    }
}

fn ledger_err_to_query_err(err: LedgerError) -> QueryErrorResponse {
    QueryErrorResponse {
        code: err.sqlstate().to_string(),
        message: err.to_string(),
        line: None,
        column: None,
    }
}

fn receipt_to_query_response(receipt: TransferReceipt, start: Instant) -> QueryResponse {
    let r = receipt.record;
    QueryResponse {
        columns: vec![
            "seq".into(),
            "transfer_id".into(),
            "status".into(),
            "amount_minor".into(),
            "idempotent_replay".into(),
            "entry_hash".into(),
        ],
        rows: vec![vec![
            serde_json::json!(r.seq),
            serde_json::json!(r.transfer_id),
            serde_json::json!(format!("{:?}", r.status).to_ascii_lowercase()),
            i128_json(r.total_amount_minor),
            serde_json::json!(receipt.idempotent_replay),
            serde_json::json!(r.entry_hash),
        ]],
        affected_rows: Some(usize::from(!receipt.idempotent_replay)),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("double-entry-ledger".into()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    }
}

fn i128_json(v: i128) -> serde_json::Value {
    if let Ok(i) = i64::try_from(v) {
        serde_json::Value::Number(i.into())
    } else {
        serde_json::Value::String(v.to_string())
    }
}

const ZERO_PREALLOC_BUF: [u8; 65_536] = [0u8; 65_536];

fn write_and_fdatasync_wal(
    file: &mut File,
    wal_write_pos: &mut u64,
    wal_alloc_len: &mut u64,
    data: &[u8],
) -> std::io::Result<()> {
    let needed = wal_write_pos.saturating_add(data.len() as u64);
    if needed > *wal_alloc_len {
        let step: u64 = if *wal_alloc_len < 256 * 1024 {
            256 * 1024
        } else {
            32 * 1024 * 1024
        };
        let target = needed.max(wal_alloc_len.saturating_add(step));
        file.seek(SeekFrom::Start(*wal_alloc_len))?;
        let mut cur = *wal_alloc_len;
        while cur < target {
            let chunk = ((target - cur) as usize).min(ZERO_PREALLOC_BUF.len());
            file.write_all(&ZERO_PREALLOC_BUF[..chunk])?;
            cur += chunk as u64;
        }
        file.sync_all()?;
        *wal_alloc_len = target;
        file.seek(SeekFrom::Start(*wal_write_pos))?;
    }
    file.write_all(data)?;
    *wal_write_pos = needed;
    file.sync_data()
}

fn persist_wal_event(st: &mut LedgerState, ev: &WalEvent) -> Result<(), LedgerError> {
    if st.wal_file.is_some() {
        let mut buf = Vec::with_capacity(256);
        encode_jwl1_frame(ev, &mut buf)?;
        let mut pos = st.wal_write_pos;
        let mut alloc = st.wal_alloc_len;
        if let Some(ref mut file) = st.wal_file {
            write_and_fdatasync_wal(file, &mut pos, &mut alloc, &buf)
                .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        }
        st.wal_write_pos = pos;
        st.wal_alloc_len = alloc;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AccountUndo {
    idx: usize,
    posted_debits: i128,
    posted_credits: i128,
    pending_debits: i128,
    pending_credits: i128,
    version: u64,
}

struct WaveUndoSnapshot {
    orig_seq: u64,
    orig_head_hash: String,
    orig_transfers_len: usize,
    orig_wal_events_len: usize,
    saved_accounts: Vec<AccountUndo>,
    saved_account_set: FxHashSet<usize>,
    added_holds: Vec<String>,
    added_idempotency: Vec<String>,
}

impl WaveUndoSnapshot {
    fn capture(st: &LedgerState) -> Self {
        Self {
            orig_seq: st.seq,
            orig_head_hash: st.head_hash.clone(),
            orig_transfers_len: st.transfers.len(),
            orig_wal_events_len: st.wal_events.len(),
            saved_accounts: Vec::new(),
            saved_account_set: FxHashSet::default(),
            added_holds: Vec::new(),
            added_idempotency: Vec::new(),
        }
    }

    #[inline]
    fn save_account_idx_before_mutate(&mut self, st: &LedgerState, idx: usize) {
        let already = if self.saved_accounts.len() < 8 {
            self.saved_accounts.iter().any(|u| u.idx == idx)
        } else {
            !self.saved_account_set.insert(idx)
        };
        if !already && let Some(acct) = st.accounts.get(idx) {
            if self.saved_accounts.len() == 7 {
                for u in &self.saved_accounts {
                    self.saved_account_set.insert(u.idx);
                }
                self.saved_account_set.insert(idx);
            }
            self.saved_accounts.push(AccountUndo {
                idx,
                posted_debits: acct.posted_debits,
                posted_credits: acct.posted_credits,
                pending_debits: acct.pending_debits,
                pending_credits: acct.pending_credits,
                version: acct.version,
            });
        }
    }

    fn rollback(self, st: &mut LedgerState) {
        st.seq = self.orig_seq;
        st.head_hash = self.orig_head_hash;
        st.transfers.truncate(self.orig_transfers_len);
        st.wal_events.truncate(self.orig_wal_events_len);
        for hid in self.added_holds {
            st.holds_by_id.remove(&hid);
        }
        for ik in self.added_idempotency {
            st.idempotency_index.remove(&ik);
        }
        for u in self.saved_accounts {
            if let Some(acct) = st.accounts.get_mut(u.idx) {
                acct.posted_debits = u.posted_debits;
                acct.posted_credits = u.posted_credits;
                acct.pending_debits = u.pending_debits;
                acct.pending_credits = u.pending_credits;
                acct.version = u.version;
            }
        }
    }
}

/// Validates and stages `reqs` under `st`, coalescing all newly committed `WalEvent`s into a
/// single `JWL1` binary buffer and executing at most **one** `file.sync_data()` call into
/// pre-initialized WAL extents.
fn execute_transfer_wave_locked(
    st: &mut LedgerState,
    reqs: Vec<CreateTransferReq>,
    atomic: bool,
) -> Result<(Vec<Result<TransferReceipt, LedgerError>>, usize, usize), LedgerError> {
    let mut undo = WaveUndoSnapshot::capture(st);
    let mut outcomes: Vec<Result<TransferReceipt, LedgerError>> = Vec::with_capacity(reqs.len());
    let mut wal_buf: Vec<u8> = if st.wal_file.is_some() {
        Vec::with_capacity(reqs.len().saturating_mul(208))
    } else {
        Vec::new()
    };
    let mut staged_commits = 0usize;

    for req in reqs {
        match stage_one_transfer_in_wave(st, req, &mut undo, &mut wal_buf) {
            Ok(receipt) => {
                if !receipt.idempotent_replay {
                    staged_commits += 1;
                }
                outcomes.push(Ok(receipt));
            }
            Err(err) => {
                if atomic {
                    undo.rollback(st);
                    return Err(err);
                }
                outcomes.push(Err(err));
            }
        }
    }

    let mut fsync_calls = 0usize;
    let wal_bytes_written = wal_buf.len();
    if staged_commits > 0 && !wal_buf.is_empty() && st.wal_file.is_some() {
        let mut pos = st.wal_write_pos;
        let mut alloc = st.wal_alloc_len;
        let io_res = if let Some(ref mut file) = st.wal_file {
            write_and_fdatasync_wal(file, &mut pos, &mut alloc, &wal_buf)
        } else {
            Ok(())
        };
        if let Err(io_err) = io_res {
            if let Some(ref mut file) = st.wal_file {
                let _ = file.seek(SeekFrom::Start(st.wal_write_pos));
            }
            undo.rollback(st);
            return Err(LedgerError::Persistence(io_err.to_string()));
        }
        st.wal_write_pos = pos;
        st.wal_alloc_len = alloc;
        fsync_calls = 1;
    }

    Ok((outcomes, fsync_calls, wal_bytes_written))
}

fn stage_one_transfer_in_wave(
    st: &mut LedgerState,
    req: CreateTransferReq,
    undo: &mut WaveUndoSnapshot,
    wal_buf: &mut Vec<u8>,
) -> Result<TransferReceipt, LedgerError> {
    let idem = req.idempotency_key.trim().to_string();
    if idem.is_empty() {
        return Err(LedgerError::Syntax(
            "idempotency_key is mandatory for financial transfers".into(),
        ));
    }

    let legs = if !req.legs.is_empty() {
        req.legs
    } else {
        let debit = req
            .debit_account_id
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| LedgerError::Syntax("missing debit_account_id".into()))?;
        let credit = req
            .credit_account_id
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| LedgerError::Syntax("missing credit_account_id".into()))?;
        if debit == credit {
            return Err(LedgerError::SameAccountTransfer(debit));
        }
        let amt = req
            .amount_minor
            .ok_or_else(|| LedgerError::InvalidAmount("missing amount_minor".into()))?;
        vec![
            LedgerLeg {
                account_id: debit,
                direction: LegDirection::Debit,
                amount_minor: amt,
            },
            LedgerLeg {
                account_id: credit,
                direction: LegDirection::Credit,
                amount_minor: amt,
            },
        ]
    };

    if legs.len() < 2 {
        return Err(LedgerError::Syntax(
            "transfer requires at least 2 legs (debit + credit)".into(),
        ));
    }
    if req.hold && legs.len() != 2 {
        return Err(LedgerError::Syntax(
            "two-phase HOLD requires exactly 1 debit leg and 1 credit leg".into(),
        ));
    }

    let kind = if req.hold {
        TransferKind::Hold
    } else if legs.len() == 2 {
        TransferKind::Standard
    } else {
        TransferKind::MultiLeg
    };

    let payload_hash = compute_transfer_payload_hash(kind, &legs, None, None);

    // 1. Native End-to-End Idempotency Check
    if let Some(&existing_idx) = st.idempotency_index.get(&idem)
        && let Some(existing) = st.transfers.get(existing_idx)
    {
        if existing.payload_hash == payload_hash {
            return Ok(TransferReceipt {
                record: existing.clone(),
                idempotent_replay: true,
            });
        }
        return Err(LedgerError::IdempotencyConflict {
            idempotency_key: idem,
        });
    }

    // 2. Conservation of Money + Account Validation
    let (sum_debits, expected_ledger, expected_scale, leg_acct_indices): (
        i128,
        u32,
        u8,
        Vec<usize>,
    ) = if legs.len() == 2
        && legs[0].direction == LegDirection::Debit
        && legs[1].direction == LegDirection::Credit
    {
        // Zero-HashMap stack fast path for standard 2-leg core-banking transfers & holds
        let amt0 = legs[0].amount_minor;
        let amt1 = legs[1].amount_minor;
        if amt0 <= 0 || amt1 <= 0 {
            return Err(LedgerError::InvalidAmount(
                "every leg amount must be strictly positive (> 0)".into(),
            ));
        }
        if amt0 != amt1 {
            return Err(LedgerError::UnbalancedLegs {
                debits: amt0,
                credits: amt1,
            });
        }
        let d_idx = st
            .account_index
            .get(&legs[0].account_id)
            .copied()
            .ok_or_else(|| LedgerError::AccountNotFound(legs[0].account_id.clone()))?;
        let c_idx = st
            .account_index
            .get(&legs[1].account_id)
            .copied()
            .ok_or_else(|| LedgerError::AccountNotFound(legs[1].account_id.clone()))?;
        if d_idx == c_idx {
            return Err(LedgerError::SameAccountTransfer(legs[0].account_id.clone()));
        }
        let d_acct = &st.accounts[d_idx];
        let c_acct = &st.accounts[c_idx];
        if d_acct.flags.frozen {
            return Err(LedgerError::AccountFrozen(d_acct.account_id.clone()));
        }
        if c_acct.flags.frozen {
            return Err(LedgerError::AccountFrozen(c_acct.account_id.clone()));
        }
        if c_acct.ledger_id != d_acct.ledger_id {
            return Err(LedgerError::LedgerCurrencyMismatch {
                expected: d_acct.ledger_id,
                found: c_acct.ledger_id,
                account: c_acct.account_id.clone(),
            });
        }
        if c_acct.scale != d_acct.scale {
            return Err(LedgerError::ScaleMismatch {
                expected: d_acct.scale,
                found: c_acct.scale,
                account: c_acct.account_id.clone(),
            });
        }
        if d_acct.flags.debits_must_not_exceed_credits {
            let current_reserved = d_acct.posted_debits.saturating_add(d_acct.pending_debits);
            let required_after = current_reserved.saturating_add(amt0);
            if required_after > d_acct.posted_credits {
                return Err(LedgerError::InsufficientCreditBalance {
                    account: d_acct.account_id.clone(),
                    available: d_acct.posted_credits.saturating_sub(current_reserved),
                    required: amt0,
                });
            }
        }
        if c_acct.flags.credits_must_not_exceed_debits {
            let current_reserved = c_acct.posted_credits.saturating_add(c_acct.pending_credits);
            let required_after = current_reserved.saturating_add(amt1);
            if required_after > c_acct.posted_debits {
                return Err(LedgerError::InsufficientDebitBalance {
                    account: c_acct.account_id.clone(),
                    available: c_acct.posted_debits.saturating_sub(current_reserved),
                    required: amt1,
                });
            }
        }
        (amt0, d_acct.ledger_id, d_acct.scale, vec![d_idx, c_idx])
    } else {
        let mut sum_debits: i128 = 0;
        let mut sum_credits: i128 = 0;
        let mut expected_ledger: Option<u32> = None;
        let mut expected_scale: Option<u8> = None;
        let mut delta_debits: FxHashMap<usize, i128> = FxHashMap::default();
        let mut delta_credits: FxHashMap<usize, i128> = FxHashMap::default();
        let mut idxs = Vec::with_capacity(legs.len());

        for leg in &legs {
            if leg.amount_minor <= 0 {
                return Err(LedgerError::InvalidAmount(
                    "every leg amount must be strictly positive (> 0)".into(),
                ));
            }
            let acct_idx = st
                .account_index
                .get(&leg.account_id)
                .copied()
                .ok_or_else(|| LedgerError::AccountNotFound(leg.account_id.clone()))?;
            idxs.push(acct_idx);
            let acct = &st.accounts[acct_idx];
            if acct.flags.frozen {
                return Err(LedgerError::AccountFrozen(acct.account_id.clone()));
            }
            if let Some(exp_l) = expected_ledger {
                if acct.ledger_id != exp_l {
                    return Err(LedgerError::LedgerCurrencyMismatch {
                        expected: exp_l,
                        found: acct.ledger_id,
                        account: acct.account_id.clone(),
                    });
                }
            } else {
                expected_ledger = Some(acct.ledger_id);
            }
            if let Some(exp_s) = expected_scale {
                if acct.scale != exp_s {
                    return Err(LedgerError::ScaleMismatch {
                        expected: exp_s,
                        found: acct.scale,
                        account: acct.account_id.clone(),
                    });
                }
            } else {
                expected_scale = Some(acct.scale);
            }

            match leg.direction {
                LegDirection::Debit => {
                    sum_debits = sum_debits.checked_add(leg.amount_minor).ok_or_else(|| {
                        LedgerError::InvalidAmount("debit sum i128 overflow".into())
                    })?;
                    let entry = delta_debits.entry(acct_idx).or_insert(0);
                    *entry = entry.checked_add(leg.amount_minor).ok_or_else(|| {
                        LedgerError::InvalidAmount("account debit overflow".into())
                    })?;
                }
                LegDirection::Credit => {
                    sum_credits = sum_credits.checked_add(leg.amount_minor).ok_or_else(|| {
                        LedgerError::InvalidAmount("credit sum i128 overflow".into())
                    })?;
                    let entry = delta_credits.entry(acct_idx).or_insert(0);
                    *entry = entry.checked_add(leg.amount_minor).ok_or_else(|| {
                        LedgerError::InvalidAmount("account credit overflow".into())
                    })?;
                }
            }
        }

        if sum_debits != sum_credits {
            return Err(LedgerError::UnbalancedLegs {
                debits: sum_debits,
                credits: sum_credits,
            });
        }

        if delta_debits.len() == 1
            && delta_credits.len() == 1
            && delta_debits.keys().next() == delta_credits.keys().next()
        {
            let same_idx = delta_debits.keys().next().copied().unwrap_or(0);
            let same_id = st
                .accounts
                .get(same_idx)
                .map(|a| a.account_id.clone())
                .unwrap_or_default();
            return Err(LedgerError::SameAccountTransfer(same_id));
        }

        for (&acct_idx, &add_d) in &delta_debits {
            let acct = &st.accounts[acct_idx];
            if acct.flags.debits_must_not_exceed_credits {
                let add_c = delta_credits.get(&acct_idx).copied().unwrap_or(0);
                let effective_credits = if req.hold {
                    acct.posted_credits
                } else {
                    acct.posted_credits.saturating_add(add_c)
                };
                let current_reserved = acct.posted_debits.saturating_add(acct.pending_debits);
                let required_after = current_reserved.saturating_add(add_d);
                if required_after > effective_credits {
                    return Err(LedgerError::InsufficientCreditBalance {
                        account: acct.account_id.clone(),
                        available: effective_credits.saturating_sub(current_reserved),
                        required: add_d,
                    });
                }
            }
        }
        for (&acct_idx, &add_c) in &delta_credits {
            let acct = &st.accounts[acct_idx];
            if acct.flags.credits_must_not_exceed_debits {
                let add_d = delta_debits.get(&acct_idx).copied().unwrap_or(0);
                let effective_debits = if req.hold {
                    acct.posted_debits
                } else {
                    acct.posted_debits.saturating_add(add_d)
                };
                let current_reserved = acct.posted_credits.saturating_add(acct.pending_credits);
                let required_after = current_reserved.saturating_add(add_c);
                if required_after > effective_debits {
                    return Err(LedgerError::InsufficientDebitBalance {
                        account: acct.account_id.clone(),
                        available: effective_debits.saturating_sub(current_reserved),
                        required: add_c,
                    });
                }
            }
        }

        (
            sum_debits,
            expected_ledger.unwrap_or(1),
            expected_scale.unwrap_or(2),
            idxs,
        )
    };

    let next_seq = st.seq + 1;
    let transfer_id = req
        .transfer_id
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("tx_{next_seq:08}"));
    if req.hold && st.holds_by_id.contains_key(&transfer_id) {
        return Err(LedgerError::Syntax(format!(
            "hold_id '{transfer_id}' already exists"
        )));
    }
    let ts = now_ns();
    let prev_hash = st.head_hash.clone();
    let status = if req.hold {
        TransferStatus::Pending
    } else {
        TransferStatus::Posted
    };
    let mut record = LedgerTransferRecord {
        seq: next_seq,
        transfer_id: transfer_id.clone(),
        idempotency_key: idem.clone(),
        kind,
        status,
        ledger_id: expected_ledger,
        scale: expected_scale,
        legs,
        total_amount_minor: sum_debits,
        captured_amount_minor: None,
        parent_hold_id: None,
        memo: req.memo,
        payload_hash,
        prev_hash,
        entry_hash: String::new(),
        timestamp_ns: ts,
    };
    record.entry_hash = compute_transfer_entry_hash(&record);

    let ev = WalEvent::Transfer {
        record: record.clone(),
    };
    if st.wal_file.is_some() {
        encode_jwl1_frame(&ev, wal_buf)?;
    }

    for &acct_idx in &leg_acct_indices {
        undo.save_account_idx_before_mutate(st, acct_idx);
    }
    for (leg, &acct_idx) in record.legs.iter().zip(leg_acct_indices.iter()) {
        let acct = &mut st.accounts[acct_idx];
        if req.hold {
            match leg.direction {
                LegDirection::Debit => {
                    acct.pending_debits = acct.pending_debits.saturating_add(leg.amount_minor);
                }
                LegDirection::Credit => {
                    acct.pending_credits = acct.pending_credits.saturating_add(leg.amount_minor);
                }
            }
        } else {
            match leg.direction {
                LegDirection::Debit => {
                    acct.posted_debits = acct.posted_debits.saturating_add(leg.amount_minor);
                }
                LegDirection::Credit => {
                    acct.posted_credits = acct.posted_credits.saturating_add(leg.amount_minor);
                }
            }
        }
        acct.version += 1;
    }

    let idx = st.transfers.len();
    st.seq = next_seq;
    st.head_hash = record.entry_hash.clone();
    if req.hold {
        st.holds_by_id.insert(transfer_id.clone(), idx);
        undo.added_holds.push(transfer_id);
    }
    st.idempotency_index.insert(idem.clone(), idx);
    undo.added_idempotency.push(idem);
    st.transfers.push(record.clone());
    st.wal_events.push(ev);

    Ok(TransferReceipt {
        record,
        idempotent_replay: false,
    })
}

// ============================================================================
// Binary `JWL1` Frame Codec + IEEE CRC-32 + Hybrid JSONL Backward Reader
// ============================================================================

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (c & 1) != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

const CRC32_TABLE: [u32; 256] = build_crc32_table();

#[inline]
fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        let idx = ((crc ^ (b as u32)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

fn write_len_str(out: &mut Vec<u8>, s: &str) -> Result<(), LedgerError> {
    let bytes = s.as_bytes();
    let len = u16::try_from(bytes.len()).map_err(|_| {
        LedgerError::Persistence(format!(
            "JWL1 string length {} exceeds u16::MAX",
            bytes.len()
        ))
    })?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn read_len_str(buf: &[u8], pos: &mut usize) -> Result<String, LedgerError> {
    let end_len = pos
        .checked_add(2)
        .ok_or_else(|| LedgerError::Persistence("corrupt JWL1 str len overflow".into()))?;
    let hdr = buf
        .get(*pos..end_len)
        .ok_or_else(|| LedgerError::Persistence("truncated JWL1 str length".into()))?;
    let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
    *pos = end_len;
    let end_str = pos
        .checked_add(len)
        .ok_or_else(|| LedgerError::Persistence("corrupt JWL1 str body overflow".into()))?;
    let slice = buf
        .get(*pos..end_str)
        .ok_or_else(|| LedgerError::Persistence("truncated JWL1 str body".into()))?;
    *pos = end_str;
    std::str::from_utf8(slice)
        .map(str::to_string)
        .map_err(|e| LedgerError::Persistence(format!("invalid UTF-8 in JWL1 frame: {e}")))
}

fn write_hash_hex(out: &mut Vec<u8>, hex_str: &str) -> Result<(), LedgerError> {
    if let Some(raw32) = hex_decode_32(hex_str) {
        out.push(1);
        out.extend_from_slice(&raw32);
        Ok(())
    } else {
        out.push(0);
        write_len_str(out, hex_str)
    }
}

fn read_hash_hex(buf: &[u8], pos: &mut usize) -> Result<String, LedgerError> {
    let mode = *buf
        .get(*pos)
        .ok_or_else(|| LedgerError::Persistence("truncated JWL1 hash tag".into()))?;
    *pos += 1;
    if mode == 1 {
        let end = pos
            .checked_add(32)
            .ok_or_else(|| LedgerError::Persistence("corrupt JWL1 hash32 overflow".into()))?;
        let raw = buf
            .get(*pos..end)
            .ok_or_else(|| LedgerError::Persistence("truncated JWL1 hash32".into()))?;
        *pos = end;
        Ok(hex_encode(raw))
    } else {
        read_len_str(buf, pos)
    }
}

fn read_exact_bytes<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], LedgerError> {
    let end = pos
        .checked_add(n)
        .ok_or_else(|| LedgerError::Persistence("corrupt JWL1 offset overflow".into()))?;
    let s = buf
        .get(*pos..end)
        .ok_or_else(|| LedgerError::Persistence("truncated JWL1 payload".into()))?;
    *pos = end;
    Ok(s)
}

fn encode_jwl1_frame(ev: &WalEvent, out: &mut Vec<u8>) -> Result<(), LedgerError> {
    out.extend_from_slice(&JWL1_MAGIC);
    let len_offset = out.len();
    out.extend_from_slice(&[0u8; 8]); // placeholder: u32 payload_len + u32 crc32
    let payload_start = out.len();

    match ev {
        WalEvent::CreateAccount {
            seq,
            account,
            prev_hash,
            entry_hash,
        } => {
            out.push(1);
            out.extend_from_slice(&seq.to_le_bytes());
            write_len_str(out, &account.account_id)?;
            out.extend_from_slice(&account.ledger_id.to_le_bytes());
            out.extend_from_slice(&account.code.to_le_bytes());
            out.push(account.scale);
            let mut flags_byte = 0u8;
            if account.flags.debits_must_not_exceed_credits {
                flags_byte |= 0x01;
            }
            if account.flags.credits_must_not_exceed_debits {
                flags_byte |= 0x02;
            }
            if account.flags.frozen {
                flags_byte |= 0x04;
            }
            out.push(flags_byte);
            out.extend_from_slice(&account.created_at_ns.to_le_bytes());
            write_hash_hex(out, prev_hash)?;
            write_hash_hex(out, entry_hash)?;
        }
        WalEvent::Transfer { record } => {
            out.push(2);
            out.extend_from_slice(&record.seq.to_le_bytes());
            write_len_str(out, &record.transfer_id)?;
            write_len_str(out, &record.idempotency_key)?;
            out.push(match record.kind {
                TransferKind::Standard => 0,
                TransferKind::MultiLeg => 1,
                TransferKind::Hold => 2,
                TransferKind::PostHold => 3,
                TransferKind::VoidHold => 4,
            });
            out.push(match record.status {
                TransferStatus::Posted => 0,
                TransferStatus::Pending => 1,
                TransferStatus::Captured => 2,
                TransferStatus::Voided => 3,
            });
            out.extend_from_slice(&record.ledger_id.to_le_bytes());
            out.push(record.scale);
            let legs_len = u16::try_from(record.legs.len())
                .map_err(|_| LedgerError::Persistence("too many legs for JWL1 u16".into()))?;
            out.extend_from_slice(&legs_len.to_le_bytes());
            for leg in &record.legs {
                write_len_str(out, &leg.account_id)?;
                out.push(match leg.direction {
                    LegDirection::Debit => 0,
                    LegDirection::Credit => 1,
                });
                out.extend_from_slice(&leg.amount_minor.to_le_bytes());
            }
            out.extend_from_slice(&record.total_amount_minor.to_le_bytes());
            if let Some(cap) = record.captured_amount_minor {
                out.push(1);
                out.extend_from_slice(&cap.to_le_bytes());
            } else {
                out.push(0);
            }
            if let Some(ref ph) = record.parent_hold_id {
                out.push(1);
                write_len_str(out, ph)?;
            } else {
                out.push(0);
            }
            if let Some(ref m) = record.memo {
                out.push(1);
                write_len_str(out, m)?;
            } else {
                out.push(0);
            }
            write_hash_hex(out, &record.payload_hash)?;
            write_hash_hex(out, &record.prev_hash)?;
            write_hash_hex(out, &record.entry_hash)?;
            out.extend_from_slice(&record.timestamp_ns.to_le_bytes());
        }
    }

    let payload_len = out.len() - payload_start;
    let payload_len_u32 = u32::try_from(payload_len)
        .map_err(|_| LedgerError::Persistence("JWL1 frame exceeds u32::MAX".into()))?;
    let crc = crc32_ieee(&out[payload_start..]);
    out[len_offset..len_offset + 4].copy_from_slice(&payload_len_u32.to_le_bytes());
    out[len_offset + 4..len_offset + 8].copy_from_slice(&crc.to_le_bytes());
    Ok(())
}

fn decode_jwl1_payload(payload: &[u8]) -> Result<WalEvent, LedgerError> {
    let mut pos = 0usize;
    let tag = *read_exact_bytes(payload, &mut pos, 1)?
        .first()
        .ok_or_else(|| LedgerError::Persistence("empty JWL1 payload".into()))?;
    match tag {
        1 => {
            let s8 = read_exact_bytes(payload, &mut pos, 8)?;
            let seq = u64::from_le_bytes([s8[0], s8[1], s8[2], s8[3], s8[4], s8[5], s8[6], s8[7]]);
            let account_id = read_len_str(payload, &mut pos)?;
            let meta = read_exact_bytes(payload, &mut pos, 16)?;
            let ledger_id = u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]]);
            let code = u16::from_le_bytes([meta[4], meta[5]]);
            let scale = meta[6];
            let fb = meta[7];
            let created_at_ns = u64::from_le_bytes([
                meta[8], meta[9], meta[10], meta[11], meta[12], meta[13], meta[14], meta[15],
            ]);
            let prev_hash = read_hash_hex(payload, &mut pos)?;
            let entry_hash = read_hash_hex(payload, &mut pos)?;
            Ok(WalEvent::CreateAccount {
                seq,
                account: LedgerAccount {
                    account_id,
                    ledger_id,
                    code,
                    scale,
                    flags: AccountFlags {
                        debits_must_not_exceed_credits: (fb & 0x01) != 0,
                        credits_must_not_exceed_debits: (fb & 0x02) != 0,
                        frozen: (fb & 0x04) != 0,
                    },
                    posted_debits: 0,
                    posted_credits: 0,
                    pending_debits: 0,
                    pending_credits: 0,
                    created_at_ns,
                    version: 1,
                },
                prev_hash,
                entry_hash,
            })
        }
        2 => {
            let s8 = read_exact_bytes(payload, &mut pos, 8)?;
            let seq = u64::from_le_bytes([s8[0], s8[1], s8[2], s8[3], s8[4], s8[5], s8[6], s8[7]]);
            let transfer_id = read_len_str(payload, &mut pos)?;
            let idempotency_key = read_len_str(payload, &mut pos)?;
            let hdr = read_exact_bytes(payload, &mut pos, 9)?;
            let kind = match hdr[0] {
                0 => TransferKind::Standard,
                1 => TransferKind::MultiLeg,
                2 => TransferKind::Hold,
                3 => TransferKind::PostHold,
                4 => TransferKind::VoidHold,
                other => {
                    return Err(LedgerError::Persistence(format!(
                        "unknown JWL1 transfer kind {other}"
                    )));
                }
            };
            let status = match hdr[1] {
                0 => TransferStatus::Posted,
                1 => TransferStatus::Pending,
                2 => TransferStatus::Captured,
                3 => TransferStatus::Voided,
                other => {
                    return Err(LedgerError::Persistence(format!(
                        "unknown JWL1 transfer status {other}"
                    )));
                }
            };
            let ledger_id = u32::from_le_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]);
            let scale = hdr[6];
            let legs_count = u16::from_le_bytes([hdr[7], hdr[8]]) as usize;
            let mut legs = Vec::with_capacity(legs_count.min(256));
            for _ in 0..legs_count {
                let account_id = read_len_str(payload, &mut pos)?;
                let lb = read_exact_bytes(payload, &mut pos, 17)?;
                let direction = match lb[0] {
                    0 => LegDirection::Debit,
                    1 => LegDirection::Credit,
                    other => {
                        return Err(LedgerError::Persistence(format!(
                            "unknown JWL1 leg direction {other}"
                        )));
                    }
                };
                let mut amt_arr = [0u8; 16];
                amt_arr.copy_from_slice(&lb[1..17]);
                legs.push(LedgerLeg {
                    account_id,
                    direction,
                    amount_minor: i128::from_le_bytes(amt_arr),
                });
            }
            let tot_b = read_exact_bytes(payload, &mut pos, 16)?;
            let mut tot_arr = [0u8; 16];
            tot_arr.copy_from_slice(tot_b);
            let total_amount_minor = i128::from_le_bytes(tot_arr);

            let has_cap = read_exact_bytes(payload, &mut pos, 1)?[0];
            let captured_amount_minor = if has_cap != 0 {
                let cb = read_exact_bytes(payload, &mut pos, 16)?;
                let mut carr = [0u8; 16];
                carr.copy_from_slice(cb);
                Some(i128::from_le_bytes(carr))
            } else {
                None
            };
            let has_ph = read_exact_bytes(payload, &mut pos, 1)?[0];
            let parent_hold_id = if has_ph != 0 {
                Some(read_len_str(payload, &mut pos)?)
            } else {
                None
            };
            let has_memo = read_exact_bytes(payload, &mut pos, 1)?[0];
            let memo = if has_memo != 0 {
                Some(read_len_str(payload, &mut pos)?)
            } else {
                None
            };
            let payload_hash = read_hash_hex(payload, &mut pos)?;
            let prev_hash = read_hash_hex(payload, &mut pos)?;
            let entry_hash = read_hash_hex(payload, &mut pos)?;
            let tsb = read_exact_bytes(payload, &mut pos, 8)?;
            let timestamp_ns = u64::from_le_bytes([
                tsb[0], tsb[1], tsb[2], tsb[3], tsb[4], tsb[5], tsb[6], tsb[7],
            ]);
            Ok(WalEvent::Transfer {
                record: LedgerTransferRecord {
                    seq,
                    transfer_id,
                    idempotency_key,
                    kind,
                    status,
                    ledger_id,
                    scale,
                    legs,
                    total_amount_minor,
                    captured_amount_minor,
                    parent_hold_id,
                    memo,
                    payload_hash,
                    prev_hash,
                    entry_hash,
                    timestamp_ns,
                },
            })
        }
        other => Err(LedgerError::Persistence(format!(
            "unknown JWL1 event tag {other}"
        ))),
    }
}

/// Decodes a `.jwal` byte stream containing `JWL1` binary CRC-32 frames,
/// legacy `{...}\n` JSONL lines, or any mix of both, stopping cleanly at
/// zero-initialized pre-allocated WAL tail padding (`0x00`).
fn decode_wal_stream(raw: &[u8]) -> Result<(Vec<WalEvent>, usize), LedgerError> {
    let mut events = Vec::new();
    let mut pos = 0usize;
    let mut valid_end = 0usize;
    while pos < raw.len() {
        while pos < raw.len() && matches!(raw[pos], b' ' | b'\t' | b'\r' | b'\n') {
            pos += 1;
        }
        if pos >= raw.len() || raw[pos] == 0 {
            break;
        }
        if raw[pos] == JWL1_MAGIC[0] {
            let hdr_end = pos
                .checked_add(13)
                .ok_or_else(|| LedgerError::Persistence("corrupt JWL1 header offset".into()))?;
            let hdr = raw
                .get(pos..hdr_end)
                .ok_or_else(|| LedgerError::Persistence("truncated JWL1 frame header".into()))?;
            if hdr[..5] != JWL1_MAGIC {
                return Err(LedgerError::Persistence("invalid JWL1 frame magic".into()));
            }
            let payload_len = u32::from_le_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) as usize;
            let expected_crc = u32::from_le_bytes([hdr[9], hdr[10], hdr[11], hdr[12]]);
            let body_end = hdr_end
                .checked_add(payload_len)
                .ok_or_else(|| LedgerError::Persistence("corrupt JWL1 frame length".into()))?;
            let payload = raw
                .get(hdr_end..body_end)
                .ok_or_else(|| LedgerError::Persistence("truncated JWL1 frame payload".into()))?;
            let actual_crc = crc32_ieee(payload);
            if actual_crc != expected_crc {
                return Err(LedgerError::Persistence(format!(
                    "JWL1 CRC-32 mismatch at byte {pos}: expected {expected_crc:#010x}, got {actual_crc:#010x}"
                )));
            }
            events.push(decode_jwl1_payload(payload)?);
            pos = body_end;
            valid_end = pos;
        } else {
            let line_end = raw[pos..]
                .iter()
                .position(|&b| b == b'\n' || b == 0)
                .map_or(raw.len(), |rel| pos + rel);
            let line = &raw[pos..line_end];
            let ev: WalEvent = serde_json::from_slice(line)
                .map_err(|e| LedgerError::Persistence(format!("corrupt ledger WAL JSONL: {e}")))?;
            events.push(ev);
            pos = if line_end < raw.len() && raw[line_end] == b'\n' {
                line_end + 1
            } else {
                line_end
            };
            valid_end = pos;
        }
    }
    Ok((events, valid_end))
}

fn replay_wal_event(st: &mut LedgerState, ev: WalEvent) -> Result<(), LedgerError> {
    match &ev {
        WalEvent::CreateAccount {
            seq,
            account,
            entry_hash,
            ..
        } => {
            st.seq = *seq;
            st.head_hash = entry_hash.clone();
            if let Some(&idx) = st.account_index.get(&account.account_id) {
                st.accounts[idx] = account.clone();
            } else {
                let idx = st.accounts.len();
                st.account_index.insert(account.account_id.clone(), idx);
                st.accounts.push(account.clone());
            }
        }
        WalEvent::Transfer { record } => {
            apply_transfer_to_accounts(st, record)?;
            let idx = st.transfers.len();
            st.seq = record.seq;
            st.head_hash = record.entry_hash.clone();
            if record.kind == TransferKind::Hold {
                st.holds_by_id.insert(record.transfer_id.clone(), idx);
            }
            st.idempotency_index
                .insert(record.idempotency_key.clone(), idx);
            st.transfers.push(record.clone());
        }
    }
    st.wal_events.push(ev);
    Ok(())
}

fn get_account_mut<'a>(
    st: &'a mut LedgerState,
    account_id: &str,
) -> Result<&'a mut LedgerAccount, LedgerError> {
    let idx = st
        .account_index
        .get(account_id)
        .copied()
        .ok_or_else(|| LedgerError::AccountNotFound(account_id.to_string()))?;
    st.accounts
        .get_mut(idx)
        .ok_or_else(|| LedgerError::AccountNotFound(account_id.to_string()))
}

fn apply_transfer_to_accounts(
    st: &mut LedgerState,
    record: &LedgerTransferRecord,
) -> Result<(), LedgerError> {
    match record.kind {
        TransferKind::Standard | TransferKind::MultiLeg => {
            for leg in &record.legs {
                let acct = get_account_mut(st, &leg.account_id)?;
                match leg.direction {
                    LegDirection::Debit => {
                        acct.posted_debits = acct.posted_debits.saturating_add(leg.amount_minor);
                    }
                    LegDirection::Credit => {
                        acct.posted_credits = acct.posted_credits.saturating_add(leg.amount_minor);
                    }
                }
                acct.version += 1;
            }
        }
        TransferKind::Hold => {
            for leg in &record.legs {
                let acct = get_account_mut(st, &leg.account_id)?;
                match leg.direction {
                    LegDirection::Debit => {
                        acct.pending_debits = acct.pending_debits.saturating_add(leg.amount_minor);
                    }
                    LegDirection::Credit => {
                        acct.pending_credits =
                            acct.pending_credits.saturating_add(leg.amount_minor);
                    }
                }
                acct.version += 1;
            }
        }
        TransferKind::PostHold => {
            let hold_id = record
                .parent_hold_id
                .as_deref()
                .ok_or_else(|| LedgerError::HoldNotFound(String::new()))?;
            let hold_idx = st
                .holds_by_id
                .get(hold_id)
                .copied()
                .ok_or_else(|| LedgerError::HoldNotFound(hold_id.to_string()))?;
            let captured = record
                .captured_amount_minor
                .unwrap_or(record.total_amount_minor);
            for leg in &record.legs {
                let acct = get_account_mut(st, &leg.account_id)?;
                match leg.direction {
                    LegDirection::Debit => {
                        acct.pending_debits = acct.pending_debits.saturating_sub(leg.amount_minor);
                        acct.posted_debits = acct.posted_debits.saturating_add(captured);
                    }
                    LegDirection::Credit => {
                        acct.pending_credits =
                            acct.pending_credits.saturating_sub(leg.amount_minor);
                        acct.posted_credits = acct.posted_credits.saturating_add(captured);
                    }
                }
                acct.version += 1;
            }
            if let Some(hold_entry) = st.transfers.get_mut(hold_idx) {
                hold_entry.status = TransferStatus::Captured;
                hold_entry.captured_amount_minor = Some(captured);
            }
        }
        TransferKind::VoidHold => {
            let hold_id = record
                .parent_hold_id
                .as_deref()
                .ok_or_else(|| LedgerError::HoldNotFound(String::new()))?;
            let hold_idx = st
                .holds_by_id
                .get(hold_id)
                .copied()
                .ok_or_else(|| LedgerError::HoldNotFound(hold_id.to_string()))?;
            for leg in &record.legs {
                let acct = get_account_mut(st, &leg.account_id)?;
                match leg.direction {
                    LegDirection::Debit => {
                        acct.pending_debits = acct.pending_debits.saturating_sub(leg.amount_minor);
                    }
                    LegDirection::Credit => {
                        acct.pending_credits =
                            acct.pending_credits.saturating_sub(leg.amount_minor);
                    }
                }
                acct.version += 1;
            }
            if let Some(hold_entry) = st.transfers.get_mut(hold_idx) {
                hold_entry.status = TransferStatus::Voided;
                hold_entry.captured_amount_minor = Some(0);
            }
        }
    }
    Ok(())
}

#[inline]
fn update_u128_dec(h: &mut Sha256, mut v: u128) {
    let mut buf = [0u8; 40];
    let mut i = buf.len();
    if v == 0 {
        h.update(b"0");
        return;
    }
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    h.update(&buf[i..]);
}

#[inline]
fn update_i128_dec(h: &mut Sha256, v: i128) {
    if v < 0 {
        h.update(b"-");
        update_u128_dec(h, v.unsigned_abs());
    } else {
        update_u128_dec(h, v as u128);
    }
}

#[inline]
fn kind_debug_bytes(k: TransferKind) -> &'static [u8] {
    match k {
        TransferKind::Standard => b"Standard",
        TransferKind::MultiLeg => b"MultiLeg",
        TransferKind::Hold => b"Hold",
        TransferKind::PostHold => b"PostHold",
        TransferKind::VoidHold => b"VoidHold",
    }
}

fn compute_transfer_payload_hash(
    kind: TransferKind,
    legs: &[LedgerLeg],
    parent_hold_id: Option<&str>,
    capture_amount_minor: Option<i128>,
) -> String {
    let mut h = Sha256::new();
    h.update(kind_debug_bytes(kind));
    h.update(b"|");
    for leg in legs {
        h.update(leg.account_id.as_bytes());
        match leg.direction {
            LegDirection::Debit => h.update(b":Debit:"),
            LegDirection::Credit => h.update(b":Credit:"),
        }
        update_i128_dec(&mut h, leg.amount_minor);
        h.update(b";");
    }
    if let Some(hid) = parent_hold_id {
        h.update(b"hold=");
        h.update(hid.as_bytes());
    }
    if let Some(cap) = capture_amount_minor {
        h.update(b"|cap=");
        update_i128_dec(&mut h, cap);
    }
    hex_encode(&h.finalize())
}

fn compute_account_event_hash(prev_hash: &str, seq: u64, a: &LedgerAccount) -> String {
    let mut h = Sha256::new();
    h.update(prev_hash.as_bytes());
    h.update(b"|acct|");
    update_u128_dec(&mut h, seq as u128);
    h.update(b"|");
    h.update(a.account_id.as_bytes());
    h.update(b"|");
    update_u128_dec(&mut h, a.ledger_id as u128);
    h.update(b"|");
    update_u128_dec(&mut h, a.code as u128);
    h.update(b"|");
    update_u128_dec(&mut h, a.scale as u128);
    h.update(if a.flags.debits_must_not_exceed_credits {
        &b"|true|"[..]
    } else {
        &b"|false|"[..]
    });
    h.update(if a.flags.credits_must_not_exceed_debits {
        &b"true|"[..]
    } else {
        &b"false|"[..]
    });
    h.update(if a.flags.frozen {
        &b"true|"[..]
    } else {
        &b"false|"[..]
    });
    update_u128_dec(&mut h, a.created_at_ns as u128);
    hex_encode(&h.finalize())
}

fn compute_transfer_entry_hash(r: &LedgerTransferRecord) -> String {
    let mut h = Sha256::new();
    h.update(r.prev_hash.as_bytes());
    h.update(b"|tx|");
    update_u128_dec(&mut h, r.seq as u128);
    h.update(b"|");
    h.update(r.transfer_id.as_bytes());
    h.update(b"|");
    h.update(r.idempotency_key.as_bytes());
    h.update(b"|");
    h.update(kind_debug_bytes(r.kind));
    h.update(b"|");
    update_u128_dec(&mut h, r.ledger_id as u128);
    h.update(b"|");
    update_u128_dec(&mut h, r.scale as u128);
    h.update(b"|");
    update_i128_dec(&mut h, r.total_amount_minor);
    h.update(b"|");
    h.update(r.payload_hash.as_bytes());
    h.update(b"|");
    match r.captured_amount_minor {
        Some(c) => {
            h.update(b"Some(");
            update_i128_dec(&mut h, c);
            h.update(b")|");
        }
        None => h.update(b"None|"),
    }
    match r.parent_hold_id {
        Some(ref ph) => {
            h.update(&format!("Some({ph:?})|").into_bytes());
        }
        None => h.update(b"None|"),
    }
    update_u128_dec(&mut h, r.timestamp_ns as u128);
    hex_encode(&h.finalize())
}

const HEX_LUT: &[u8; 16] = b"0123456789abcdef";

#[inline]
fn hex_encode(bytes: &[u8]) -> String {
    let mut out = vec![0u8; bytes.len() * 2];
    for (i, &b) in bytes.iter().enumerate() {
        out[i * 2] = HEX_LUT[(b >> 4) as usize];
        out[i * 2 + 1] = HEX_LUT[(b & 0x0F) as usize];
    }
    String::from_utf8(out).unwrap_or_default()
}

#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[inline]
fn hex_decode_32(hex: &str) -> Option<[u8; 32]> {
    let b = hex.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_nibble(b[i * 2])?;
        let lo = hex_nibble(b[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn tokenize_ledger_cmd(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() || c == ',' || c == '(' || c == ')' {
            chars.next();
            continue;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            chars.next();
            let mut s = String::new();
            while let Some(ch) = chars.next() {
                if ch == quote {
                    break;
                }
                s.push(ch);
            }
            out.push(s);
        } else {
            let mut s = String::new();
            while let Some(&ch) = chars.peek() {
                if ch.is_whitespace() || ch == ',' || ch == '(' || ch == ')' || ch == ';' {
                    break;
                }
                s.push(ch);
                chars.next();
            }
            out.push(s);
        }
    }
    out
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// ============================================================================
// HTTP REST Router (`/api/v1/ledger/*`)
// ============================================================================

pub fn financial_ledger_routes(ledger: std::sync::Arc<FinancialLedger>) -> axum::Router {
    use axum::routing::{get, post};
    axum::Router::new()
        .route(
            "/api/v1/ledger/accounts",
            post(http_create_account).get(http_list_accounts),
        )
        .route(
            "/api/v1/ledger/transfers",
            post(http_submit_transfer).get(http_list_transfers),
        )
        .route(
            "/api/v1/ledger/transfers/batch",
            post(http_submit_transfer_batch),
        )
        .route("/api/v1/ledger/holds/{id}/post", post(http_post_hold))
        .route("/api/v1/ledger/holds/{id}/void", post(http_void_hold))
        .route("/api/v1/ledger/verify", get(http_verify_chain))
        .with_state(ledger)
}

fn ledger_http_status(err: &LedgerError) -> axum::http::StatusCode {
    use axum::http::StatusCode;
    match err {
        LedgerError::AccountAlreadyExists(_) | LedgerError::IdempotencyConflict { .. } => {
            StatusCode::CONFLICT
        }
        LedgerError::AccountNotFound(_) | LedgerError::HoldNotFound(_) => StatusCode::NOT_FOUND,
        LedgerError::InsufficientCreditBalance { .. }
        | LedgerError::InsufficientDebitBalance { .. }
        | LedgerError::AccountFrozen(_)
        | LedgerError::HoldAlreadyResolved { .. }
        | LedgerError::CaptureExceedsHold { .. }
        | LedgerError::UnbalancedLegs { .. }
        | LedgerError::SameAccountTransfer(_)
        | LedgerError::LedgerCurrencyMismatch { .. }
        | LedgerError::ScaleMismatch { .. }
        | LedgerError::InvalidAmount(_) => StatusCode::UNPROCESSABLE_ENTITY,
        LedgerError::Syntax(_) => StatusCode::BAD_REQUEST,
        LedgerError::Persistence(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn http_create_account(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
    axum::Json(req): axum::Json<CreateLedgerAccountReq>,
) -> Result<
    (axum::http::StatusCode, axum::Json<serde_json::Value>),
    (axum::http::StatusCode, axum::Json<serde_json::Value>),
> {
    match ledger.create_account(req) {
        Ok(acct) => Ok((
            axum::http::StatusCode::CREATED,
            axum::Json(serde_json::json!({ "account": acct })),
        )),
        Err(e) => Err((
            ledger_http_status(&e),
            axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": e.sqlstate() })),
        )),
    }
}

async fn http_list_accounts(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "accounts": ledger.list_accounts() }))
}

async fn http_submit_transfer(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
    axum::Json(req): axum::Json<CreateTransferReq>,
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    let res = if ledger.is_persistent() {
        tokio::task::spawn_blocking(move || ledger.submit_transfer(req))
            .await
            .map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": "P0001" })),
                )
            })?
    } else {
        ledger.submit_transfer(req)
    };
    match res {
        Ok(receipt) => Ok(axum::Json(serde_json::json!(receipt))),
        Err(e) => Err((
            ledger_http_status(&e),
            axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": e.sqlstate() })),
        )),
    }
}

async fn http_submit_transfer_batch(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
    axum::Json(req): axum::Json<BatchTransferReq>,
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    let res = if ledger.is_persistent() {
        tokio::task::spawn_blocking(move || {
            ledger.submit_transfer_batch_with_options(
                req.transfers,
                req.atomic,
                req.include_receipts,
            )
        })
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": "P0001" })),
            )
        })?
    } else {
        ledger.submit_transfer_batch_with_options(req.transfers, req.atomic, req.include_receipts)
    };
    match res {
        Ok(report) => Ok(axum::Json(serde_json::json!(report))),
        Err(e) => Err((
            ledger_http_status(&e),
            axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": e.sqlstate() })),
        )),
    }
}

/// Fast-path native `pgwire` / SQL function dispatcher for core banking:
/// - `SELECT ledger_transfer(idem, debit, credit, amount_minor)` -> returns `seq` (`BIGINT`)
/// - `SELECT ledger_transfer_batch(idems, debits, credits, amounts [, atomic])` -> returns `committed` (`BIGINT`)
pub fn try_execute_ledger_function_sql(
    ledger: &FinancialLedger,
    sql: &str,
    args: &[serde_json::Value],
    start: Instant,
) -> Option<Result<QueryResponse, QueryErrorResponse>> {
    let t = sql.trim().trim_end_matches(';').trim();
    if t.len() < 22 || !t.as_bytes()[..6].eq_ignore_ascii_case(b"SELECT") {
        return None;
    }
    let after_sel = t[6..].trim_start();
    let (is_batch, fn_rest) = if after_sel
        .as_bytes()
        .get(..21)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"LEDGER_TRANSFER_BATCH"))
    {
        (true, after_sel[21..].trim_start())
    } else if after_sel
        .as_bytes()
        .get(..15)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"LEDGER_TRANSFER"))
    {
        (false, after_sel[15..].trim_start())
    } else {
        return None;
    };
    if !fn_rest.starts_with('(') {
        return None;
    }
    let close_idx = find_matching_rparen(fn_rest)?;
    let inside = &fn_rest[1..close_idx];
    let tail = fn_rest[close_idx + 1..].trim();
    // Allow optional `AS alias` tail only; let complex `FROM` / `WHERE` queries fall through
    let col_alias = if tail.is_empty() {
        if is_batch {
            "ledger_transfer_batch".to_string()
        } else {
            "ledger_transfer".to_string()
        }
    } else if tail
        .as_bytes()
        .get(..3)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"AS "))
    {
        let al = tail[3..].trim().trim_matches('"');
        if al.contains(' ') {
            return None;
        }
        al.to_string()
    } else {
        return None;
    };

    let raw_args = split_top_level_sql_args(inside);
    if !is_batch {
        if raw_args.len() != 4 {
            return Some(Err(QueryErrorResponse::syntax_error(
                "LEDGER_TRANSFER expects 4 arguments: (idempotency_key, debit_account, credit_account, amount_minor)",
                1,
                1,
            )));
        }
        let idem = match resolve_scalar_str_arg(&raw_args[0], args) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let debit = match resolve_scalar_str_arg(&raw_args[1], args) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let credit = match resolve_scalar_str_arg(&raw_args[2], args) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let amt = match resolve_scalar_i128_arg(&raw_args[3], args) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let res = ledger
            .submit_transfer(CreateTransferReq {
                idempotency_key: idem,
                transfer_id: None,
                debit_account_id: Some(debit),
                credit_account_id: Some(credit),
                amount_minor: Some(amt),
                legs: Vec::new(),
                hold: false,
                memo: None,
            })
            .map_err(ledger_err_to_query_err);
        return Some(res.map(|receipt| QueryResponse {
            columns: vec![col_alias],
            rows: vec![vec![serde_json::json!(receipt.record.seq as i64)]],
            affected_rows: Some(usize::from(!receipt.idempotent_replay)),
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated: false,
            warnings: Vec::new(),
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: Some("double-entry-jwl1-pgwire".into()),
            session_id: None,
            viz_hint: None,
            energy_receipt: None,
        }));
    }

    if raw_args.len() != 4 && raw_args.len() != 5 {
        return Some(Err(QueryErrorResponse::syntax_error(
            "LEDGER_TRANSFER_BATCH expects 4 or 5 arguments: (idems, debits, credits, amounts [, atomic])",
            1,
            1,
        )));
    }
    let idems = match resolve_str_vec_arg(&raw_args[0], args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let debits = match resolve_str_vec_arg(&raw_args[1], args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let credits = match resolve_str_vec_arg(&raw_args[2], args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let amounts = match resolve_i128_vec_arg(&raw_args[3], args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let atomic = if raw_args.len() == 5 {
        match resolve_bool_arg(&raw_args[4], args) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        }
    } else {
        false
    };
    if idems.len() != debits.len() || idems.len() != credits.len() || idems.len() != amounts.len() {
        return Some(Err(QueryErrorResponse::syntax_error(
            "LEDGER_TRANSFER_BATCH array arguments must all have identical length",
            1,
            1,
        )));
    }
    let mut reqs = Vec::with_capacity(idems.len());
    for (((ik, d), c), amt) in idems.into_iter().zip(debits).zip(credits).zip(amounts) {
        reqs.push(CreateTransferReq {
            idempotency_key: ik,
            transfer_id: None,
            debit_account_id: Some(d),
            credit_account_id: Some(c),
            amount_minor: Some(amt),
            legs: Vec::new(),
            hold: false,
            memo: None,
        });
    }
    let res = ledger
        .submit_transfer_batch_with_options(reqs, atomic, false)
        .map_err(ledger_err_to_query_err);
    Some(res.map(|rep| QueryResponse {
        columns: vec![col_alias],
        rows: vec![vec![serde_json::json!(rep.committed as i64)]],
        affected_rows: Some(rep.committed),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("double-entry-jwl1-batch-pgwire".into()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    }))
}

fn find_matching_rparen(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_sq = false;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if in_sq {
            if b == b'\'' {
                if bytes.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                in_sq = false;
            }
        } else {
            match b {
                b'\'' => in_sq = true,
                b'(' => depth += 1,
                b')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn split_top_level_sql_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut bracket = 0usize;
    let mut in_sq = false;
    let bytes = s.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if in_sq {
            if b == b'\'' {
                if bytes.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                in_sq = false;
            }
        } else {
            match b {
                b'\'' => in_sq = true,
                b'(' => depth += 1,
                b')' => depth = depth.saturating_sub(1),
                b'[' => bracket += 1,
                b']' => bracket = bracket.saturating_sub(1),
                b',' if depth == 0 && bracket == 0 => {
                    out.push(s[start..i].trim().to_string());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    let rem = s[start..].trim();
    if !rem.is_empty() {
        out.push(rem.to_string());
    }
    out
}

fn strip_pg_typecast(raw: &str) -> &str {
    let t = raw.trim();
    if let Some(pos) = t.rfind("::") {
        let after = &t[pos + 2..];
        if !after.contains('\'') && !after.contains(',') {
            return t[..pos].trim();
        }
    }
    t
}

fn resolve_param_value<'a>(
    raw: &str,
    args: &'a [serde_json::Value],
) -> Option<&'a serde_json::Value> {
    let clean = strip_pg_typecast(raw);
    let idx_str = clean.strip_prefix('$')?;
    let idx_1: usize = idx_str.parse().ok()?;
    idx_1.checked_sub(1).and_then(|i| args.get(i))
}

fn unquote_sql_str(raw: &str) -> Option<String> {
    let clean = strip_pg_typecast(raw);
    if clean.len() >= 2 && clean.starts_with('\'') && clean.ends_with('\'') {
        Some(clean[1..clean.len() - 1].replace("''", "'"))
    } else {
        None
    }
}

fn resolve_scalar_str_arg(
    raw: &str,
    args: &[serde_json::Value],
) -> Result<String, QueryErrorResponse> {
    if let Some(v) = resolve_param_value(raw, args) {
        return match v {
            serde_json::Value::String(s) => Ok(s.clone()),
            other => Ok(other.to_string()),
        };
    }
    unquote_sql_str(raw)
        .ok_or_else(|| QueryErrorResponse::syntax_error("expected string argument", 1, 1))
}

fn resolve_scalar_i128_arg(
    raw: &str,
    args: &[serde_json::Value],
) -> Result<i128, QueryErrorResponse> {
    if let Some(v) = resolve_param_value(raw, args) {
        return match v {
            serde_json::Value::Number(n) => n
                .as_i64()
                .map(i128::from)
                .or_else(|| n.as_u64().map(i128::from))
                .ok_or_else(|| QueryErrorResponse::syntax_error("invalid integer parameter", 1, 1)),
            serde_json::Value::String(s) => s
                .trim()
                .parse::<i128>()
                .map_err(|_| QueryErrorResponse::syntax_error("invalid integer string", 1, 1)),
            _ => Err(QueryErrorResponse::syntax_error(
                "expected integer parameter",
                1,
                1,
            )),
        };
    }
    let clean = strip_pg_typecast(raw);
    let unq = unquote_sql_str(clean).unwrap_or_else(|| clean.to_string());
    unq.replace('_', "")
        .parse::<i128>()
        .map_err(|_| QueryErrorResponse::syntax_error("invalid integer literal", 1, 1))
}

fn resolve_bool_arg(raw: &str, args: &[serde_json::Value]) -> Result<bool, QueryErrorResponse> {
    if let Some(v) = resolve_param_value(raw, args) {
        return match v {
            serde_json::Value::Bool(b) => Ok(*b),
            serde_json::Value::String(s) => Ok(matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "true" | "t" | "1"
            )),
            serde_json::Value::Number(n) => Ok(n.as_i64().unwrap_or(0) != 0),
            _ => Err(QueryErrorResponse::syntax_error(
                "invalid bool parameter",
                1,
                1,
            )),
        };
    }
    let clean = strip_pg_typecast(raw);
    let s = unquote_sql_str(clean).unwrap_or_else(|| clean.to_string());
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "t" | "1" => Ok(true),
        "false" | "f" | "0" => Ok(false),
        _ => Err(QueryErrorResponse::syntax_error(
            "invalid boolean literal",
            1,
            1,
        )),
    }
}

fn parse_pg_or_json_str_list(s: &str) -> Result<Vec<String>, QueryErrorResponse> {
    let t = s.trim();
    if t.is_empty() || t == "{}" || t == "[]" {
        return Ok(Vec::new());
    }
    if t.starts_with('[')
        && let Ok(v) = serde_json::from_str::<Vec<String>>(t)
    {
        return Ok(v);
    }
    let inner = if t.starts_with('{') && t.ends_with('}') {
        &t[1..t.len() - 1]
    } else {
        t
    };
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_q {
            if ch == '\\' {
                if let Some(nx) = chars.next() {
                    cur.push(nx);
                }
            } else if ch == '"' {
                in_q = false;
            } else {
                cur.push(ch);
            }
        } else {
            match ch {
                '"' => in_q = true,
                ',' => {
                    out.push(cur.trim().trim_matches('\'').to_string());
                    cur.clear();
                }
                _ => cur.push(ch),
            }
        }
    }
    out.push(cur.trim().trim_matches('\'').to_string());
    Ok(out)
}

fn resolve_str_vec_arg(
    raw: &str,
    args: &[serde_json::Value],
) -> Result<Vec<String>, QueryErrorResponse> {
    if let Some(v) = resolve_param_value(raw, args) {
        return match v {
            serde_json::Value::Array(arr) => Ok(arr
                .iter()
                .map(|x| match x {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect()),
            serde_json::Value::String(s) => parse_pg_or_json_str_list(s),
            _ => Err(QueryErrorResponse::syntax_error(
                "expected array or string list parameter",
                1,
                1,
            )),
        };
    }
    let clean = strip_pg_typecast(raw);
    if let Some(s) = unquote_sql_str(clean) {
        return parse_pg_or_json_str_list(&s);
    }
    if clean
        .as_bytes()
        .get(..6)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"ARRAY["))
        && clean.ends_with(']')
    {
        let inner = &clean[6..clean.len() - 1];
        return split_top_level_sql_args(inner)
            .into_iter()
            .map(|elem| resolve_scalar_str_arg(&elem, args))
            .collect();
    }
    Err(QueryErrorResponse::syntax_error(
        "expected array literal in LEDGER_TRANSFER_BATCH",
        1,
        1,
    ))
}

fn resolve_i128_vec_arg(
    raw: &str,
    args: &[serde_json::Value],
) -> Result<Vec<i128>, QueryErrorResponse> {
    if let Some(v) = resolve_param_value(raw, args) {
        return match v {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|x| match x {
                    serde_json::Value::Number(n) => n
                        .as_i64()
                        .map(i128::from)
                        .or_else(|| n.as_u64().map(i128::from))
                        .ok_or_else(|| {
                            QueryErrorResponse::syntax_error("invalid integer in array", 1, 1)
                        }),
                    serde_json::Value::String(s) => s.trim().parse::<i128>().map_err(|_| {
                        QueryErrorResponse::syntax_error("invalid integer in array", 1, 1)
                    }),
                    _ => Err(QueryErrorResponse::syntax_error(
                        "invalid element in integer array",
                        1,
                        1,
                    )),
                })
                .collect(),
            serde_json::Value::String(s) => {
                let strs = parse_pg_or_json_str_list(s)?;
                strs.into_iter()
                    .map(|item| {
                        item.replace('_', "").parse::<i128>().map_err(|_| {
                            QueryErrorResponse::syntax_error("invalid integer in pg array", 1, 1)
                        })
                    })
                    .collect()
            }
            _ => Err(QueryErrorResponse::syntax_error(
                "expected integer array parameter",
                1,
                1,
            )),
        };
    }
    let clean = strip_pg_typecast(raw);
    if let Some(s) = unquote_sql_str(clean) {
        let strs = parse_pg_or_json_str_list(&s)?;
        return strs
            .into_iter()
            .map(|item| {
                item.replace('_', "").parse::<i128>().map_err(|_| {
                    QueryErrorResponse::syntax_error("invalid integer in pg array", 1, 1)
                })
            })
            .collect();
    }
    if clean
        .as_bytes()
        .get(..6)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"ARRAY["))
        && clean.ends_with(']')
    {
        let inner = &clean[6..clean.len() - 1];
        return split_top_level_sql_args(inner)
            .into_iter()
            .map(|elem| resolve_scalar_i128_arg(&elem, args))
            .collect();
    }
    Err(QueryErrorResponse::syntax_error(
        "expected integer array literal in LEDGER_TRANSFER_BATCH",
        1,
        1,
    ))
}

async fn http_list_transfers(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "transfers": ledger.list_transfers() }))
}

async fn http_post_hold(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
    axum::extract::Path(hold_id): axum::extract::Path<String>,
    axum::Json(req): axum::Json<ResolveHoldReq>,
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    match ledger.post_hold(&hold_id, req) {
        Ok(receipt) => Ok(axum::Json(serde_json::json!(receipt))),
        Err(e) => Err((
            ledger_http_status(&e),
            axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": e.sqlstate() })),
        )),
    }
}

async fn http_void_hold(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
    axum::extract::Path(hold_id): axum::extract::Path<String>,
    axum::Json(req): axum::Json<ResolveHoldReq>,
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    match ledger.void_hold(&hold_id, req) {
        Ok(receipt) => Ok(axum::Json(serde_json::json!(receipt))),
        Err(e) => Err((
            ledger_http_status(&e),
            axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": e.sqlstate() })),
        )),
    }
}

async fn http_verify_chain(
    axum::extract::State(ledger): axum::extract::State<std::sync::Arc<FinancialLedger>>,
) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!(ledger.verify_chain()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn double_entry_conservation_idempotency_two_phase_hold_and_crash_replay() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = FinancialLedger::open(dir.path()).unwrap();

        // Treasury (unconstrained funding source) + Alice wallet (cannot overdraft!) + Merchant
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "treasury_usd".into(),
                ledger_id: 1,
                code: 1000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "alice_usd".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags {
                    debits_must_not_exceed_credits: true,
                    credits_must_not_exceed_debits: false,
                    frozen: false,
                },
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "merchant_usd".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();

        // Fund Alice with $100.00 (10_000 cents)
        let r1 = ledger
            .submit_transfer(CreateTransferReq {
                idempotency_key: "fund-alice-001".into(),
                transfer_id: None,
                debit_account_id: Some("treasury_usd".into()),
                credit_account_id: Some("alice_usd".into()),
                amount_minor: Some(10_000),
                legs: vec![],
                hold: false,
                memo: Some("Initial deposit".into()),
            })
            .unwrap();
        assert!(!r1.idempotent_replay);

        // Retry exact same idempotency_key -> idempotent replay, NO second deposit!
        let r1_retry = ledger
            .submit_transfer(CreateTransferReq {
                idempotency_key: "fund-alice-001".into(),
                transfer_id: None,
                debit_account_id: Some("treasury_usd".into()),
                credit_account_id: Some("alice_usd".into()),
                amount_minor: Some(10_000),
                legs: vec![],
                hold: false,
                memo: Some("Initial deposit".into()),
            })
            .unwrap();
        assert!(r1_retry.idempotent_replay);
        assert_eq!(r1_retry.record.seq, r1.record.seq);

        // Mutating amount under same idempotency_key -> hard IdempotencyConflict!
        let conflict = ledger.submit_transfer(CreateTransferReq {
            idempotency_key: "fund-alice-001".into(),
            transfer_id: None,
            debit_account_id: Some("treasury_usd".into()),
            credit_account_id: Some("alice_usd".into()),
            amount_minor: Some(99_999),
            legs: vec![],
            hold: false,
            memo: None,
        });
        assert!(matches!(
            conflict,
            Err(LedgerError::IdempotencyConflict { .. })
        ));

        // Two-phase card authorization hold for $60.00 (6_000 cents)
        ledger
            .submit_transfer(CreateTransferReq {
                idempotency_key: "auth-hotel-001".into(),
                transfer_id: Some("hold_hotel_1".into()),
                debit_account_id: Some("alice_usd".into()),
                credit_account_id: Some("merchant_usd".into()),
                amount_minor: Some(6_000),
                legs: vec![],
                hold: true,
                memo: Some("Hotel check-in auth".into()),
            })
            .unwrap();

        // Alice only has $40.00 available now -> trying to spend $50.00 MUST fail overdraft check!
        let overdraft = ledger.submit_transfer(CreateTransferReq {
            idempotency_key: "overdraft-attempt".into(),
            transfer_id: None,
            debit_account_id: Some("alice_usd".into()),
            credit_account_id: Some("merchant_usd".into()),
            amount_minor: Some(5_000),
            legs: vec![],
            hold: false,
            memo: None,
        });
        assert!(matches!(
            overdraft,
            Err(LedgerError::InsufficientCreditBalance { .. })
        ));

        // Partial capture of $45.00 (4_500 cents) -> releases remaining $15.00 hold automatically!
        ledger
            .post_hold(
                "hold_hotel_1",
                ResolveHoldReq {
                    idempotency_key: "capture-hotel-001".into(),
                    capture_amount_minor: Some(4_500),
                    memo: Some("Checkout final bill".into()),
                },
            )
            .unwrap();

        // Now Alice has $55.00 (5_500 cents) available -> spending $50.00 succeeds!
        ledger
            .submit_transfer(CreateTransferReq {
                idempotency_key: "dinner-001".into(),
                transfer_id: None,
                debit_account_id: Some("alice_usd".into()),
                credit_account_id: Some("merchant_usd".into()),
                amount_minor: Some(5_000),
                legs: vec![],
                hold: false,
                memo: None,
            })
            .unwrap();

        let rep = ledger.verify_chain();
        assert!(rep.valid, "{rep:?}");
        assert_eq!(rep.total_posted_debits, 19_500);
        assert_eq!(rep.total_posted_credits, 19_500);
        assert_eq!(rep.total_pending_debits, 0);
        assert_eq!(rep.total_pending_credits, 0);

        // Reopen from disk WAL and verify identical balances + chain proof
        drop(ledger);
        let reopened = FinancialLedger::open(dir.path()).unwrap();
        let rep2 = reopened.verify_chain();
        assert!(rep2.valid);
        assert_eq!(rep2.head_hash, rep.head_hash);
        let accts = reopened.list_accounts();
        let alice = accts.iter().find(|a| a.account_id == "alice_usd").unwrap();
        assert_eq!(alice.net_credits_minor(), 500); // $5.00 remaining
    }

    #[test]
    fn concurrent_64_thread_hot_wallet_never_overdrafts_or_loses_a_penny() {
        let ledger = Arc::new(FinancialLedger::new_in_memory());
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "mint".into(),
                ledger_id: 1,
                code: 1000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "hot_wallet".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags {
                    debits_must_not_exceed_credits: true,
                    credits_must_not_exceed_debits: false,
                    frozen: false,
                },
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "sink".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();

        // Fund hot_wallet with exactly 500 units; 64 threads each attempt 20 debits of 1 unit (1,280 total attempts)
        ledger
            .submit_transfer(CreateTransferReq {
                idempotency_key: "seed".into(),
                transfer_id: None,
                debit_account_id: Some("mint".into()),
                credit_account_id: Some("hot_wallet".into()),
                amount_minor: Some(500),
                legs: vec![],
                hold: false,
                memo: None,
            })
            .unwrap();

        let mut handles = Vec::new();
        for tid in 0..64 {
            let l = ledger.clone();
            handles.push(std::thread::spawn(move || {
                let mut ok = 0usize;
                for j in 0..20 {
                    if l.submit_transfer(CreateTransferReq {
                        idempotency_key: format!("t-{tid}-{j}"),
                        transfer_id: None,
                        debit_account_id: Some("hot_wallet".into()),
                        credit_account_id: Some("sink".into()),
                        amount_minor: Some(1),
                        legs: vec![],
                        hold: false,
                        memo: None,
                    })
                    .is_ok()
                    {
                        ok += 1;
                    }
                }
                ok
            }));
        }
        let succeeded: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(succeeded, 500);
        let rep = ledger.verify_chain();
        assert!(rep.valid);
        let hot = ledger
            .list_accounts()
            .into_iter()
            .find(|a| a.account_id == "hot_wallet")
            .unwrap();
        assert_eq!(hot.net_credits_minor(), 0);
    }

    #[test]
    fn jwl1_batch_atomic_rollback_non_atomic_clearing_and_disk_group_commit() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Arc::new(FinancialLedger::open(dir.path()).unwrap());
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "treasury".into(),
                ledger_id: 1,
                code: 1000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "wallet_a".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags {
                    debits_must_not_exceed_credits: true,
                    credits_must_not_exceed_debits: false,
                    frozen: false,
                },
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "wallet_b".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();

        // 1. Atomic batch where item #2 overdrafts -> entire batch rolls back cleanly!
        let bad_batch = ledger.submit_transfer_batch(
            vec![
                CreateTransferReq {
                    idempotency_key: "b-seed".into(),
                    transfer_id: None,
                    debit_account_id: Some("treasury".into()),
                    credit_account_id: Some("wallet_a".into()),
                    amount_minor: Some(1_000),
                    legs: vec![],
                    hold: false,
                    memo: None,
                },
                CreateTransferReq {
                    idempotency_key: "b-ok".into(),
                    transfer_id: None,
                    debit_account_id: Some("wallet_a".into()),
                    credit_account_id: Some("wallet_b".into()),
                    amount_minor: Some(400),
                    legs: vec![],
                    hold: false,
                    memo: None,
                },
                CreateTransferReq {
                    idempotency_key: "b-overdraft".into(),
                    transfer_id: None,
                    debit_account_id: Some("wallet_a".into()),
                    credit_account_id: Some("wallet_b".into()),
                    amount_minor: Some(900),
                    legs: vec![],
                    hold: false,
                    memo: None,
                },
            ],
            true,
        );
        assert!(matches!(
            bad_batch,
            Err(LedgerError::InsufficientCreditBalance { .. })
        ));
        assert_eq!(ledger.list_transfers().len(), 0);

        // 2. Non-atomic clearing batch: items #0 and #1 commit in 1 single fdatasync, item #2 rejects!
        let rep = ledger
            .submit_transfer_batch(
                vec![
                    CreateTransferReq {
                        idempotency_key: "c-seed".into(),
                        transfer_id: None,
                        debit_account_id: Some("treasury".into()),
                        credit_account_id: Some("wallet_a".into()),
                        amount_minor: Some(1_000),
                        legs: vec![],
                        hold: false,
                        memo: None,
                    },
                    CreateTransferReq {
                        idempotency_key: "c-ok".into(),
                        transfer_id: None,
                        debit_account_id: Some("wallet_a".into()),
                        credit_account_id: Some("wallet_b".into()),
                        amount_minor: Some(400),
                        legs: vec![],
                        hold: false,
                        memo: None,
                    },
                    CreateTransferReq {
                        idempotency_key: "c-overdraft".into(),
                        transfer_id: None,
                        debit_account_id: Some("wallet_a".into()),
                        credit_account_id: Some("wallet_b".into()),
                        amount_minor: Some(900),
                        legs: vec![],
                        hold: false,
                        memo: None,
                    },
                ],
                false,
            )
            .unwrap();
        assert_eq!(rep.submitted, 3);
        assert_eq!(rep.committed, 2);
        assert_eq!(rep.rejected, 1);
        assert_eq!(rep.fsync_calls, 1);

        // 3. 16-thread concurrent persistent `submit_transfer` exercising FIFO group-commit coalescer
        // wallet_a currently has 600 remaining; 16 threads x 50 transfers of 10 = 8,000 attempted -> exact 60 succeed!
        let mut handles = Vec::new();
        for tid in 0..16 {
            let l = ledger.clone();
            handles.push(std::thread::spawn(move || {
                let mut ok = 0usize;
                for j in 0..50 {
                    if l.submit_transfer(CreateTransferReq {
                        idempotency_key: format!("gc-{tid}-{j}"),
                        transfer_id: None,
                        debit_account_id: Some("wallet_a".into()),
                        credit_account_id: Some("wallet_b".into()),
                        amount_minor: Some(10),
                        legs: vec![],
                        hold: false,
                        memo: None,
                    })
                    .is_ok()
                    {
                        ok += 1;
                    }
                }
                ok
            }));
        }
        let gc_ok: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(gc_ok, 60);

        let verify1 = ledger.verify_chain();
        assert!(verify1.valid, "{verify1:?}");
        drop(ledger);

        // 4. Reopen JWL1 binary WAL from disk and verify bit-for-bit identical chain & balances
        let reopened = FinancialLedger::open(dir.path()).unwrap();
        let verify2 = reopened.verify_chain();
        assert!(verify2.valid, "{verify2:?}");
        assert_eq!(verify2.head_hash, verify1.head_hash);
        assert_eq!(
            verify2.journal_entries_verified,
            verify1.journal_entries_verified
        );
        let wa = reopened
            .list_accounts()
            .into_iter()
            .find(|a| a.account_id == "wallet_a")
            .unwrap();
        let wb = reopened
            .list_accounts()
            .into_iter()
            .find(|a| a.account_id == "wallet_b")
            .unwrap();
        assert_eq!(wa.net_credits_minor(), 0);
        assert_eq!(wb.net_credits_minor(), 1_000);

        // 5. Corrupt a payload byte inside the first JWL1 frame and verify CRC-32 catches it on open!
        drop(reopened);
        let wal_file_path = dir.path().join(LEDGER_WAL_FILENAME);
        let mut wal_bytes = std::fs::read(&wal_file_path).unwrap();
        assert_eq!(&wal_bytes[..5], &JWL1_MAGIC);
        // Pre-allocated WAL has >= 256 KiB capacity with zero-initialized tail
        assert!(wal_bytes.len() >= 256 * 1024);
        wal_bytes[20] ^= 0xFF;
        std::fs::write(&wal_file_path, &wal_bytes).unwrap();
        let err = FinancialLedger::open(dir.path());
        assert!(
            matches!(err, Err(LedgerError::Persistence(ref m)) if m.contains("CRC-32")),
            "expected CRC-32 error, got {}",
            err.err().map_or_else(|| "Ok".into(), |e| e.to_string())
        );
    }

    #[test]
    fn pgwire_ledger_sql_functions_single_and_batch_parity() {
        let ledger = FinancialLedger::new_in_memory();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "vault".into(),
                ledger_id: 1,
                code: 1000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "cust_1".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags {
                    debits_must_not_exceed_credits: true,
                    credits_must_not_exceed_debits: false,
                    frozen: false,
                },
            })
            .unwrap();
        ledger
            .create_account(CreateLedgerAccountReq {
                account_id: "cust_2".into(),
                ledger_id: 1,
                code: 2000,
                scale: 2,
                flags: AccountFlags::default(),
            })
            .unwrap();

        let t0 = Instant::now();
        let r1 = try_execute_ledger_function_sql(
            &ledger,
            "SELECT ledger_transfer($1, $2, $3, $4)",
            &[
                serde_json::json!("fn-seed-1"),
                serde_json::json!("vault"),
                serde_json::json!("cust_1"),
                serde_json::json!(1_000),
            ],
            t0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(r1.rows[0][0], serde_json::json!(4));

        // PostgreSQL wire text-array format "{a,b,c}" as sent by psycopg3 for text[] / bigint[]
        let r_batch = try_execute_ledger_function_sql(
            &ledger,
            "SELECT ledger_transfer_batch($1::text[], $2::text[], $3::text[], $4::bigint[])",
            &[
                serde_json::json!("{fn-b1,fn-b2,fn-over}"),
                serde_json::json!("{cust_1,cust_1,cust_1}"),
                serde_json::json!("{cust_2,cust_2,cust_2}"),
                serde_json::json!("{300,400,500}"),
            ],
            t0,
        )
        .unwrap()
        .unwrap();
        // 300 + 400 commit, 500 rejects due to NO_OVERDRAFT (300 remaining < 500)
        assert_eq!(r_batch.rows[0][0], serde_json::json!(2));
        assert!(ledger.verify_chain().valid);
    }
}
