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

use crate::decimal128::Decimal128;
use crate::query::{QueryErrorResponse, QueryResponse};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const LEDGER_WAL_FILENAME: &str = "financial_ledger.jwal";

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

struct LedgerState {
    accounts: BTreeMap<String, LedgerAccount>,
    transfers: Vec<LedgerTransferRecord>,
    holds_by_id: HashMap<String, usize>,
    idempotency_index: HashMap<String, usize>,
    wal_events: Vec<WalEvent>,
    seq: u64,
    head_hash: String,
    wal_file: Option<File>,
    wal_path: Option<PathBuf>,
}

/// Thread-safe, strictly serialized Double-Entry Financial Ledger Engine.
pub struct FinancialLedger {
    inner: Mutex<LedgerState>,
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
                accounts: BTreeMap::new(),
                transfers: Vec::new(),
                holds_by_id: HashMap::new(),
                idempotency_index: HashMap::new(),
                wal_events: Vec::new(),
                seq: 0,
                head_hash: "0".repeat(64),
                wal_file: None,
                wal_path: None,
            }),
        }
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
            accounts: BTreeMap::new(),
            transfers: Vec::new(),
            holds_by_id: HashMap::new(),
            idempotency_index: HashMap::new(),
            wal_events: Vec::new(),
            seq: 0,
            head_hash: "0".repeat(64),
            wal_file: None,
            wal_path: Some(wal_path.clone()),
        };

        if wal_path.exists() {
            let f = File::open(&wal_path).map_err(|e| LedgerError::Persistence(e.to_string()))?;
            let reader = BufReader::new(f);
            for line_res in reader.lines() {
                let line = line_res.map_err(|e| LedgerError::Persistence(e.to_string()))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let ev: WalEvent = serde_json::from_str(trimmed)
                    .map_err(|e| LedgerError::Persistence(format!("corrupt ledger WAL: {e}")))?;
                replay_wal_event(&mut state, ev)?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        state.wal_file = Some(file);
        Ok(Self {
            inner: Mutex::new(state),
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
        if st.accounts.contains_key(&id) {
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
        st.accounts.insert(id, account.clone());
        st.wal_events.push(ev);
        Ok(account)
    }

    pub fn list_accounts(&self) -> Vec<LedgerAccount> {
        self.inner
            .lock()
            .map(|st| st.accounts.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn list_transfers(&self) -> Vec<LedgerTransferRecord> {
        self.inner
            .lock()
            .map(|st| st.transfers.clone())
            .unwrap_or_default()
    }

    pub fn submit_transfer(&self, req: CreateTransferReq) -> Result<TransferReceipt, LedgerError> {
        let idem = req.idempotency_key.trim().to_string();
        if idem.is_empty() {
            return Err(LedgerError::Syntax(
                "idempotency_key is mandatory for financial transfers".into(),
            ));
        }

        let legs = if !req.legs.is_empty() {
            req.legs.clone()
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

        let mut st = self
            .inner
            .lock()
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;

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
        let mut sum_debits: i128 = 0;
        let mut sum_credits: i128 = 0;
        let mut expected_ledger: Option<u32> = None;
        let mut expected_scale: Option<u8> = None;
        let mut delta_debits: HashMap<String, i128> = HashMap::new();
        let mut delta_credits: HashMap<String, i128> = HashMap::new();

        for leg in &legs {
            if leg.amount_minor <= 0 {
                return Err(LedgerError::InvalidAmount(
                    "every leg amount must be strictly positive (> 0)".into(),
                ));
            }
            let acct = st
                .accounts
                .get(&leg.account_id)
                .ok_or_else(|| LedgerError::AccountNotFound(leg.account_id.clone()))?;
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
                    let entry = delta_debits.entry(leg.account_id.clone()).or_insert(0);
                    *entry = entry
                        .checked_add(leg.amount_minor)
                        .ok_or_else(|| LedgerError::InvalidAmount("account debit overflow".into()))?;
                }
                LegDirection::Credit => {
                    sum_credits = sum_credits.checked_add(leg.amount_minor).ok_or_else(|| {
                        LedgerError::InvalidAmount("credit sum i128 overflow".into())
                    })?;
                    let entry = delta_credits.entry(leg.account_id.clone()).or_insert(0);
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

        // Reject degenerate self-cancel where all legs hit one single account
        if delta_debits.len() == 1
            && delta_credits.len() == 1
            && delta_debits.keys().next() == delta_credits.keys().next()
        {
            let same = delta_debits.keys().next().cloned().unwrap_or_default();
            return Err(LedgerError::SameAccountTransfer(same));
        }

        // 3. Enforce Overdraft / Floor Invariants BEFORE mutating any account
        for (acct_id, add_d) in &delta_debits {
            if let Some(acct) = st.accounts.get(acct_id)
                && acct.flags.debits_must_not_exceed_credits
            {
                let add_c = delta_credits.get(acct_id).copied().unwrap_or(0);
                // Pending credits do NOT count as spendable funds until posted;
                // pending debits DO reserve funds immediately.
                let effective_credits = if req.hold {
                    acct.posted_credits
                } else {
                    acct.posted_credits.saturating_add(add_c)
                };
                let current_reserved = acct.posted_debits.saturating_add(acct.pending_debits);
                let required_after = current_reserved.saturating_add(*add_d);
                if required_after > effective_credits {
                    return Err(LedgerError::InsufficientCreditBalance {
                        account: acct_id.clone(),
                        available: effective_credits.saturating_sub(current_reserved),
                        required: *add_d,
                    });
                }
            }
        }
        for (acct_id, add_c) in &delta_credits {
            if let Some(acct) = st.accounts.get(acct_id)
                && acct.flags.credits_must_not_exceed_debits
            {
                let add_d = delta_debits.get(acct_id).copied().unwrap_or(0);
                let effective_debits = if req.hold {
                    acct.posted_debits
                } else {
                    acct.posted_debits.saturating_add(add_d)
                };
                let current_reserved = acct.posted_credits.saturating_add(acct.pending_credits);
                let required_after = current_reserved.saturating_add(*add_c);
                if required_after > effective_debits {
                    return Err(LedgerError::InsufficientDebitBalance {
                        account: acct_id.clone(),
                        available: effective_debits.saturating_sub(current_reserved),
                        required: *add_c,
                    });
                }
            }
        }

        // 4. Construct hash-chained journal entry & fsync WAL BEFORE in-memory commit
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
            ledger_id: expected_ledger.unwrap_or(1),
            scale: expected_scale.unwrap_or(2),
            legs: legs.clone(),
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
        persist_wal_event(&mut st, &ev)?;

        // 5. Apply atomic balance updates
        apply_transfer_to_accounts(&mut st, &record)?;
        let idx = st.transfers.len();
        st.seq = next_seq;
        st.head_hash = record.entry_hash.clone();
        if req.hold {
            st.holds_by_id.insert(transfer_id, idx);
        }
        st.idempotency_index.insert(idem, idx);
        st.transfers.push(record.clone());
        st.wal_events.push(ev);

        Ok(TransferReceipt {
            record,
            idempotent_replay: false,
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
        for acct in st.accounts.values() {
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
                        ledger_id = tokens
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| {
                                QueryErrorResponse::syntax_error("invalid LEDGER u32", 1, 1)
                            })?;
                    }
                    "CODE" => {
                        i += 1;
                        code = tokens
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| {
                                QueryErrorResponse::syntax_error("invalid CODE u16", 1, 1)
                            })?;
                    }
                    "SCALE" => {
                        i += 1;
                        scale = tokens
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| {
                                QueryErrorResponse::syntax_error("invalid SCALE u8", 1, 1)
                            })?;
                    }
                    "NO_OVERDRAFT"
                    | "NO_OVERDRAFT_CREDIT"
                    | "DEBITS_MUST_NOT_EXCEED_CREDITS" => {
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
            if tokens.get(1).is_none_or(|t| !t.eq_ignore_ascii_case("HOLD")) {
                return Err(QueryErrorResponse::syntax_error(
                    &format!("expected LEDGER {sub0} HOLD <hold_id> ..."),
                    1,
                    1,
                ));
            }
            let hid = tokens.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("missing hold_id", 1, 1)
            })?;
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
                                    QueryErrorResponse::syntax_error(
                                        "invalid capture AMOUNT",
                                        1,
                                        1,
                                    )
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

fn persist_wal_event(st: &mut LedgerState, ev: &WalEvent) -> Result<(), LedgerError> {
    if let Some(ref mut file) = st.wal_file {
        let mut line =
            serde_json::to_vec(ev).map_err(|e| LedgerError::Persistence(e.to_string()))?;
        line.push(b'\n');
        file.write_all(&line)
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
        file.sync_data()
            .map_err(|e| LedgerError::Persistence(e.to_string()))?;
    }
    Ok(())
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
            st.accounts.insert(account.account_id.clone(), account.clone());
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

fn apply_transfer_to_accounts(
    st: &mut LedgerState,
    record: &LedgerTransferRecord,
) -> Result<(), LedgerError> {
    match record.kind {
        TransferKind::Standard | TransferKind::MultiLeg => {
            for leg in &record.legs {
                let acct = st
                    .accounts
                    .get_mut(&leg.account_id)
                    .ok_or_else(|| LedgerError::AccountNotFound(leg.account_id.clone()))?;
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
                let acct = st
                    .accounts
                    .get_mut(&leg.account_id)
                    .ok_or_else(|| LedgerError::AccountNotFound(leg.account_id.clone()))?;
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
                let acct = st
                    .accounts
                    .get_mut(&leg.account_id)
                    .ok_or_else(|| LedgerError::AccountNotFound(leg.account_id.clone()))?;
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
                let acct = st
                    .accounts
                    .get_mut(&leg.account_id)
                    .ok_or_else(|| LedgerError::AccountNotFound(leg.account_id.clone()))?;
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

fn compute_transfer_payload_hash(
    kind: TransferKind,
    legs: &[LedgerLeg],
    parent_hold_id: Option<&str>,
    capture_amount_minor: Option<i128>,
) -> String {
    let mut h = Sha256::new();
    h.update(format!("{kind:?}|").as_bytes());
    for leg in legs {
        h.update(
            format!(
                "{}:{:?}:{};",
                leg.account_id, leg.direction, leg.amount_minor
            )
            .as_bytes(),
        );
    }
    if let Some(hid) = parent_hold_id {
        h.update(b"hold=");
        h.update(hid.as_bytes());
    }
    if let Some(cap) = capture_amount_minor {
        h.update(format!("|cap={cap}").as_bytes());
    }
    hex_encode(&h.finalize())
}

fn compute_account_event_hash(prev_hash: &str, seq: u64, a: &LedgerAccount) -> String {
    let mut h = Sha256::new();
    h.update(prev_hash.as_bytes());
    h.update(
        format!(
            "|acct|{seq}|{}|{}|{}|{}|{}|{}|{}|{}",
            a.account_id,
            a.ledger_id,
            a.code,
            a.scale,
            a.flags.debits_must_not_exceed_credits,
            a.flags.credits_must_not_exceed_debits,
            a.flags.frozen,
            a.created_at_ns
        )
        .as_bytes(),
    );
    hex_encode(&h.finalize())
}

fn compute_transfer_entry_hash(r: &LedgerTransferRecord) -> String {
    let mut h = Sha256::new();
    h.update(r.prev_hash.as_bytes());
    h.update(
        format!(
            "|tx|{}|{}|{}|{:?}|{}|{}|{}|{}|{:?}|{:?}|{}",
            r.seq,
            r.transfer_id,
            r.idempotency_key,
            r.kind,
            r.ledger_id,
            r.scale,
            r.total_amount_minor,
            r.payload_hash,
            r.captured_amount_minor,
            r.parent_hold_id,
            r.timestamp_ns
        )
        .as_bytes(),
    );
    hex_encode(&h.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
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
) -> Result<(axum::http::StatusCode, axum::Json<serde_json::Value>), (axum::http::StatusCode, axum::Json<serde_json::Value>)> {
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
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)> {
    match ledger.submit_transfer(req) {
        Ok(receipt) => Ok(axum::Json(serde_json::json!(receipt))),
        Err(e) => Err((
            ledger_http_status(&e),
            axum::Json(serde_json::json!({ "error": e.to_string(), "sqlstate": e.sqlstate() })),
        )),
    }
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
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)> {
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
) -> Result<axum::Json<serde_json::Value>, (axum::http::StatusCode, axum::Json<serde_json::Value>)> {
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
        assert!(matches!(conflict, Err(LedgerError::IdempotencyConflict { .. })));

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
}
