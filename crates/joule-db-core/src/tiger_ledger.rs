//! `TigerBeetle`-Class Strictly Serializable Double-Entry Financial Ledger & `VOPR` Fault Simulator.
//!
//! Absorbs Family 7 (`TigerBeetle`) and Standard `C7/C8` (`VOPR` Deterministic Simulation Testing)
//! into safe Rust (`#![forbid(unsafe_code)]`, `0` `.unwrap()`s):
//! 1. **Fixed-Point `i128` Double-Entry Accounts (`LedgerAccount`):** Enforces `DebitsMustNotExceedCredits`
//!    (asset/escrow accounts) and `CreditsMustNotExceedDebits` (liability/wallet accounts) across
//!    both `pending` and `posted` balances.
//! 2. **Two-Phase Pending/Posted/Voided & Atomic `Linked` Chains (`LedgerTransfer`):**
//!    - `Pending`: reserves funds in `debits_pending` / `credits_pending` without moving posted funds.
//!    - `PostPending`: commits a prior pending transfer (optionally for a partial settlement amount!).
//!    - `VoidPending`: releases reserved pending funds cleanly on timeout/cancellation.
//!    - `Linked`: chains arbitrary multi-leg transfers into an atomic all-or-nothing batch — if the
//!      last leg violates a balance cap, every preceding leg in the chain rolls back atomically!
//! 3. **Global Conservation Invariant:** At every commit boundary,
//!    $\sum \text{debits\_posted} == \sum \text{credits\_posted}$ and
//!    $\sum \text{debits\_pending} == \sum \text{credits\_pending}$.
//! 4. **`VoprDeterministicSim`:** Seeded SplitMix64 fault-injected binary journal simulator injecting
//!    torn tail writes and 1-bit NAND flips across `N` seeds and proving 100% conservation after recovery.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Account invariant bitflags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountFlags {
    /// Rejects any transfer where `debits_posted + debits_pending > credits_posted`.
    pub debits_must_not_exceed_credits: bool,
    /// Rejects any transfer where `credits_posted + credits_pending > debits_posted`.
    pub credits_must_not_exceed_debits: bool,
    /// Rejects all new transfers touching this account.
    pub closed: bool,
}

/// Immutable-history double-entry account state (`128`-bit integer minor units).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerAccount {
    pub id: u128,
    pub ledger: u32,
    pub code: u16,
    pub flags: AccountFlags,
    pub debits_pending: i128,
    pub debits_posted: i128,
    pub credits_pending: i128,
    pub credits_posted: i128,
}

impl LedgerAccount {
    pub fn new(id: u128, ledger: u32, code: u16, flags: AccountFlags) -> Self {
        Self {
            id,
            ledger,
            code,
            flags,
            debits_pending: 0,
            debits_posted: 0,
            credits_pending: 0,
            credits_posted: 0,
        }
    }

    fn check_invariants(&self) -> Result<(), TransferError> {
        if self.debits_pending < 0
            || self.debits_posted < 0
            || self.credits_pending < 0
            || self.credits_posted < 0
        {
            return Err(TransferError::NegativeBalanceOverflow);
        }
        if self.flags.debits_must_not_exceed_credits
            && self.debits_posted.saturating_add(self.debits_pending) > self.credits_posted
        {
            return Err(TransferError::ExceedsCredits {
                account_id: self.id,
            });
        }
        if self.flags.credits_must_not_exceed_debits
            && self.credits_posted.saturating_add(self.credits_pending) > self.debits_posted
        {
            return Err(TransferError::ExceedsDebits {
                account_id: self.id,
            });
        }
        Ok(())
    }
}

/// Transfer lifecycle kind (`TigerBeetle` two-phase & linked semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferKind {
    /// Immediate single-phase posted transfer.
    DirectPost,
    /// Phase 1: reserve `amount` in `debits_pending` / `credits_pending`.
    PendingReserve,
    /// Phase 2a: settle a prior `PendingReserve` (`pending_id`) up to its reserved amount.
    PostPending { pending_id: u128 },
    /// Phase 2b: void/release a prior `PendingReserve` (`pending_id`) without posting.
    VoidPending { pending_id: u128 },
}

/// Input transfer command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerTransfer {
    pub id: u128,
    pub debit_account_id: u128,
    pub credit_account_id: u128,
    pub amount: i128,
    pub ledger: u32,
    pub code: u16,
    pub kind: TransferKind,
    /// When `true`, links this transfer atomically with the immediately following transfer(s)
    /// until a transfer with `linked == false` closes the chain.
    pub linked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PendingState {
    Open,
    Posted,
    Voided,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PendingRecord {
    debit_account_id: u128,
    credit_account_id: u128,
    reserved_amount: i128,
    ledger: u32,
    state: PendingState,
}

/// Deterministic rejection reason (`TigerBeetle`-style exhaustive error taxonomy).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferError {
    IdAlreadyExists(u128),
    SameDebitAndCreditAccount(u128),
    NonPositiveAmount,
    AccountNotFound(u128),
    AccountClosed(u128),
    LedgerCurrencyMismatch {
        debit_ledger: u32,
        credit_ledger: u32,
    },
    ExceedsCredits {
        account_id: u128,
    },
    ExceedsDebits {
        account_id: u128,
    },
    PendingTransferNotFound(u128),
    PendingTransferAlreadyResolved(u128),
    SettlementExceedsPendingAmount,
    UnclosedLinkedChain,
    LinkedChainAborted {
        failed_id: u128,
    },
    NegativeBalanceOverflow,
}

/// Batch execution receipt with strict conservation proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferBatchReceipt {
    pub submitted: usize,
    pub committed: usize,
    pub rejected: usize,
    pub errors: Vec<(u128, TransferError)>,
    pub total_debits_posted: i128,
    pub total_credits_posted: i128,
    pub total_debits_pending: i128,
    pub total_credits_pending: i128,
    pub zero_sum_conserved: bool,
}

/// Strictly serializable in-memory + journal-recoverable `TigerBeetle`-class state machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TigerDoubleEntryEngine {
    accounts: BTreeMap<u128, LedgerAccount>,
    seen_ids: BTreeMap<u128, bool>,
    pending_transfers: BTreeMap<u128, PendingRecord>,
    committed_log: Vec<LedgerTransfer>,
}

impl TigerDoubleEntryEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_account(&mut self, account: LedgerAccount) -> Result<(), String> {
        if self.accounts.contains_key(&account.id) {
            return Err(format!("account {} already exists", account.id));
        }
        self.accounts.insert(account.id, account);
        Ok(())
    }

    pub fn account(&self, id: u128) -> Option<&LedgerAccount> {
        self.accounts.get(&id)
    }

    pub fn verify_zero_sum_conservation(&self) -> (i128, i128, i128, i128, bool) {
        let mut dp = 0i128;
        let mut cp = 0i128;
        let mut dpend = 0i128;
        let mut cpend = 0i128;
        for acct in self.accounts.values() {
            dp = dp.saturating_add(acct.debits_posted);
            cp = cp.saturating_add(acct.credits_posted);
            dpend = dpend.saturating_add(acct.debits_pending);
            cpend = cpend.saturating_add(acct.credits_pending);
        }
        (dp, cp, dpend, cpend, dp == cp && dpend == cpend)
    }

    fn apply_single(&mut self, tf: &LedgerTransfer) -> Result<(), TransferError> {
        if self.seen_ids.contains_key(&tf.id) {
            return Err(TransferError::IdAlreadyExists(tf.id));
        }
        match tf.kind {
            TransferKind::DirectPost | TransferKind::PendingReserve => {
                if tf.amount <= 0 {
                    return Err(TransferError::NonPositiveAmount);
                }
                if tf.debit_account_id == tf.credit_account_id {
                    return Err(TransferError::SameDebitAndCreditAccount(
                        tf.debit_account_id,
                    ));
                }
                let d_orig = self
                    .accounts
                    .get(&tf.debit_account_id)
                    .cloned()
                    .ok_or(TransferError::AccountNotFound(tf.debit_account_id))?;
                let c_orig = self
                    .accounts
                    .get(&tf.credit_account_id)
                    .cloned()
                    .ok_or(TransferError::AccountNotFound(tf.credit_account_id))?;
                if d_orig.flags.closed {
                    return Err(TransferError::AccountClosed(d_orig.id));
                }
                if c_orig.flags.closed {
                    return Err(TransferError::AccountClosed(c_orig.id));
                }
                if d_orig.ledger != tf.ledger || c_orig.ledger != tf.ledger {
                    return Err(TransferError::LedgerCurrencyMismatch {
                        debit_ledger: d_orig.ledger,
                        credit_ledger: c_orig.ledger,
                    });
                }

                let mut d_next = d_orig;
                let mut c_next = c_orig;
                if tf.kind == TransferKind::DirectPost {
                    d_next.debits_posted = d_next
                        .debits_posted
                        .checked_add(tf.amount)
                        .ok_or(TransferError::NegativeBalanceOverflow)?;
                    c_next.credits_posted = c_next
                        .credits_posted
                        .checked_add(tf.amount)
                        .ok_or(TransferError::NegativeBalanceOverflow)?;
                } else {
                    d_next.debits_pending = d_next
                        .debits_pending
                        .checked_add(tf.amount)
                        .ok_or(TransferError::NegativeBalanceOverflow)?;
                    c_next.credits_pending = c_next
                        .credits_pending
                        .checked_add(tf.amount)
                        .ok_or(TransferError::NegativeBalanceOverflow)?;
                }
                d_next.check_invariants()?;
                c_next.check_invariants()?;

                self.accounts.insert(d_next.id, d_next);
                self.accounts.insert(c_next.id, c_next);
                if tf.kind == TransferKind::PendingReserve {
                    self.pending_transfers.insert(
                        tf.id,
                        PendingRecord {
                            debit_account_id: tf.debit_account_id,
                            credit_account_id: tf.credit_account_id,
                            reserved_amount: tf.amount,
                            ledger: tf.ledger,
                            state: PendingState::Open,
                        },
                    );
                }
                self.seen_ids.insert(tf.id, true);
                Ok(())
            }
            TransferKind::PostPending { pending_id } => {
                let prec = self
                    .pending_transfers
                    .get(&pending_id)
                    .cloned()
                    .ok_or(TransferError::PendingTransferNotFound(pending_id))?;
                if prec.state != PendingState::Open {
                    return Err(TransferError::PendingTransferAlreadyResolved(pending_id));
                }
                let settle_amt = if tf.amount == 0 {
                    prec.reserved_amount
                } else {
                    tf.amount
                };
                if settle_amt <= 0 || settle_amt > prec.reserved_amount {
                    return Err(TransferError::SettlementExceedsPendingAmount);
                }
                let mut d_next = self
                    .accounts
                    .get(&prec.debit_account_id)
                    .cloned()
                    .ok_or(TransferError::AccountNotFound(prec.debit_account_id))?;
                let mut c_next = self
                    .accounts
                    .get(&prec.credit_account_id)
                    .cloned()
                    .ok_or(TransferError::AccountNotFound(prec.credit_account_id))?;

                d_next.debits_pending -= prec.reserved_amount;
                c_next.credits_pending -= prec.reserved_amount;
                d_next.debits_posted += settle_amt;
                c_next.credits_posted += settle_amt;
                d_next.check_invariants()?;
                c_next.check_invariants()?;

                self.accounts.insert(d_next.id, d_next);
                self.accounts.insert(c_next.id, c_next);
                if let Some(p) = self.pending_transfers.get_mut(&pending_id) {
                    p.state = PendingState::Posted;
                }
                self.seen_ids.insert(tf.id, true);
                Ok(())
            }
            TransferKind::VoidPending { pending_id } => {
                let prec = self
                    .pending_transfers
                    .get(&pending_id)
                    .cloned()
                    .ok_or(TransferError::PendingTransferNotFound(pending_id))?;
                if prec.state != PendingState::Open {
                    return Err(TransferError::PendingTransferAlreadyResolved(pending_id));
                }
                let mut d_next = self
                    .accounts
                    .get(&prec.debit_account_id)
                    .cloned()
                    .ok_or(TransferError::AccountNotFound(prec.debit_account_id))?;
                let mut c_next = self
                    .accounts
                    .get(&prec.credit_account_id)
                    .cloned()
                    .ok_or(TransferError::AccountNotFound(prec.credit_account_id))?;

                d_next.debits_pending -= prec.reserved_amount;
                c_next.credits_pending -= prec.reserved_amount;
                d_next.check_invariants()?;
                c_next.check_invariants()?;

                self.accounts.insert(d_next.id, d_next);
                self.accounts.insert(c_next.id, c_next);
                if let Some(p) = self.pending_transfers.get_mut(&pending_id) {
                    p.state = PendingState::Voided;
                }
                self.seen_ids.insert(tf.id, true);
                Ok(())
            }
        }
    }

    /// Submit a batch of transfers (`DirectPost`, `PendingReserve`, `PostPending`, `VoidPending`,
    /// and atomic `linked` chains) with strict savepoint rollback on linked-chain failures.
    pub fn submit_transfers(&mut self, transfers: &[LedgerTransfer]) -> TransferBatchReceipt {
        let mut committed = 0usize;
        let mut errors: Vec<(u128, TransferError)> = Vec::new();
        let mut idx = 0usize;

        while idx < transfers.len() {
            if !transfers[idx].linked {
                match self.apply_single(&transfers[idx]) {
                    Ok(()) => {
                        self.committed_log.push(transfers[idx].clone());
                        committed += 1;
                    }
                    Err(e) => errors.push((transfers[idx].id, e)),
                }
                idx += 1;
            } else {
                // Gather full `linked` chain `[idx ..= chain_end]`
                let chain_start = idx;
                while idx < transfers.len() && transfers[idx].linked {
                    idx += 1;
                }
                if idx >= transfers.len() {
                    // Unclosed linked chain at end of batch!
                    for t in &transfers[chain_start..] {
                        errors.push((t.id, TransferError::UnclosedLinkedChain));
                    }
                    break;
                }
                let chain_end = idx;
                idx += 1;

                // Atomic savepoint before executing the linked chain!
                let snap_accounts = self.accounts.clone();
                let snap_seen = self.seen_ids.clone();
                let snap_pending = self.pending_transfers.clone();

                let mut failed: Option<(usize, TransferError)> = None;
                for (offset, t) in transfers[chain_start..=chain_end].iter().enumerate() {
                    if let Err(e) = self.apply_single(t) {
                        failed = Some((chain_start + offset, e));
                        break;
                    }
                }

                if let Some((fail_pos, root_err)) = failed {
                    self.accounts = snap_accounts;
                    self.seen_ids = snap_seen;
                    self.pending_transfers = snap_pending;
                    let fail_id = transfers[fail_pos].id;
                    for (pos, t) in transfers[chain_start..=chain_end].iter().enumerate() {
                        if chain_start + pos == fail_pos {
                            errors.push((t.id, root_err.clone()));
                        } else {
                            errors.push((
                                t.id,
                                TransferError::LinkedChainAborted { failed_id: fail_id },
                            ));
                        }
                    }
                } else {
                    for t in &transfers[chain_start..=chain_end] {
                        self.committed_log.push(t.clone());
                        committed += 1;
                    }
                }
            }
        }

        let (dp, cp, dpend, cpend, conserved) = self.verify_zero_sum_conservation();
        TransferBatchReceipt {
            submitted: transfers.len(),
            committed,
            rejected: transfers.len().saturating_sub(committed),
            errors,
            total_debits_posted: dp,
            total_credits_posted: cp,
            total_debits_pending: dpend,
            total_credits_pending: cpend,
            zero_sum_conserved: conserved,
        }
    }

    /// Encode the committed transfer log as CRC32-protected `VOPR` WAL frames (`[magic:4][len:4][crc:4][json]`).
    pub fn encode_crc_wal(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for tf in &self.committed_log {
            if let Ok(payload) = serde_json::to_vec(tf) {
                let crc = crc32_ieee(&payload);
                out.extend_from_slice(b"JTB1");
                out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                out.extend_from_slice(&crc.to_le_bytes());
                out.extend_from_slice(&payload);
            }
        }
        out
    }

    /// Replay a potentially fault-injected (torn-tail / bit-rotted) `VOPR` WAL byte stream onto
    /// `initial_accounts`, halting cleanly at the first corrupted or torn frame.
    pub fn recover_from_faulty_wal(
        initial_accounts: &[LedgerAccount],
        wal_bytes: &[u8],
    ) -> (Self, usize, usize) {
        let mut eng = Self::new();
        for a in initial_accounts {
            let _ = eng.create_account(a.clone());
        }
        let mut cur = 0usize;
        let mut replayed = 0usize;
        let mut corrupted_frames_halted = 0usize;
        while cur + 12 <= wal_bytes.len() {
            if &wal_bytes[cur..cur + 4] != b"JTB1" {
                corrupted_frames_halted += 1;
                break;
            }
            let mut lb = [0u8; 4];
            lb.copy_from_slice(&wal_bytes[cur + 4..cur + 8]);
            let plen = u32::from_le_bytes(lb) as usize;
            let mut cb = [0u8; 4];
            cb.copy_from_slice(&wal_bytes[cur + 8..cur + 12]);
            let expected_crc = u32::from_le_bytes(cb);
            if cur + 12 + plen > wal_bytes.len() {
                // Torn tail write from simulated power loss!
                corrupted_frames_halted += 1;
                break;
            }
            let payload = &wal_bytes[cur + 12..cur + 12 + plen];
            if crc32_ieee(payload) != expected_crc {
                // 1-bit NAND bit-rot caught by CRC-32!
                corrupted_frames_halted += 1;
                break;
            }
            if let Ok(mut tf) = serde_json::from_slice::<LedgerTransfer>(payload) {
                tf.linked = false;
                if eng.apply_single(&tf).is_ok() {
                    eng.committed_log.push(tf);
                    replayed += 1;
                }
            }
            cur += 12 + plen;
        }
        (eng, replayed, corrupted_frames_halted)
    }
}

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Receipt from running `VOPR`-class multi-seed deterministic storage fault simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoprSimulationReceipt {
    pub seeds_tested: usize,
    pub total_transfers_simulated: usize,
    pub torn_tails_detected: usize,
    pub bit_flips_detected: usize,
    pub conservation_failures: usize,
    pub invariant_clean: bool,
}

/// Run `num_seeds` deterministic `VOPR` fault-injection campaigns (random linked chains,
/// two-phase pending settlements/voids, torn WAL tails, and 1-bit NAND flips) and verify
/// 100% double-entry conservation across every single recovered state.
pub fn run_vopr_deterministic_fault_campaign(
    num_seeds: usize,
    transfers_per_seed: usize,
) -> VoprSimulationReceipt {
    let seeds = num_seeds.clamp(1, 500);
    let t_per_seed = transfers_per_seed.clamp(8, 2_000);
    let mut torn_tails_detected = 0usize;
    let mut bit_flips_detected = 0usize;
    let mut conservation_failures = 0usize;

    for s in 0..seeds {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((s as u64) + 1) ^ 0xBF58_476D_1CE4_E5B9;
        let mut next_u64 = || {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        let mut init_accounts = Vec::new();
        // Account 1 = central treasury (`no cap`); Accounts 2..=8 = customer wallets (`debits <= credits`)
        init_accounts.push(LedgerAccount::new(1, 840, 1, AccountFlags::default()));
        for aid in 2..=8u128 {
            init_accounts.push(LedgerAccount::new(
                aid,
                840,
                10,
                AccountFlags {
                    debits_must_not_exceed_credits: true,
                    credits_must_not_exceed_debits: false,
                    closed: false,
                },
            ));
        }

        let mut eng = TigerDoubleEntryEngine::new();
        for a in &init_accounts {
            let _ = eng.create_account(a.clone());
        }

        // Seed initial treasury -> wallet funding
        let mut fund_batch = Vec::new();
        for aid in 2..=8u128 {
            fund_batch.push(LedgerTransfer {
                id: aid * 100_000 + (s as u128),
                debit_account_id: 1,
                credit_account_id: aid,
                amount: 10_000,
                ledger: 840,
                code: 1,
                kind: TransferKind::DirectPost,
                linked: false,
            });
        }
        let _ = eng.submit_transfers(&fund_batch);

        let mut batch = Vec::with_capacity(t_per_seed);
        let mut last_pending_id: Option<u128> = None;
        for step in 0..t_per_seed {
            let tid = ((s as u128) + 1) * 10_000_000 + (step as u128) + 1;
            let r = next_u64();
            let from = 2 + (r % 7) as u128;
            let mut to = 2 + ((r >> 8) % 7) as u128;
            if to == from {
                to = if from == 8 { 2 } else { from + 1 };
            }
            let amt = ((r >> 16) % 2_500 + 1) as i128;
            let mode = (r >> 32) % 5;

            let (kind, linked) = if mode == 0 {
                last_pending_id = Some(tid);
                (TransferKind::PendingReserve, false)
            } else if mode == 1
                && let Some(pid) = last_pending_id.take()
            {
                (TransferKind::PostPending { pending_id: pid }, false)
            } else if mode == 2
                && let Some(pid) = last_pending_id.take()
            {
                (TransferKind::VoidPending { pending_id: pid }, false)
            } else {
                let link_it = (mode == 3) && (step + 1 < t_per_seed);
                (TransferKind::DirectPost, link_it)
            };

            batch.push(LedgerTransfer {
                id: tid,
                debit_account_id: from,
                credit_account_id: to,
                amount: amt,
                ledger: 840,
                code: 20,
                kind,
                linked,
            });
        }
        let rec = eng.submit_transfers(&batch);
        if !rec.zero_sum_conserved {
            conservation_failures += 1;
        }

        // Inject deterministic `VOPR` fault into serialized WAL and verify recovered state!
        let mut wal = eng.encode_crc_wal();
        if wal.len() > 24 {
            if s % 2 == 0 {
                // Fault A: Torn tail truncation mid-frame!
                let cut = wal.len() - ((next_u64() as usize % 11) + 1);
                wal.truncate(cut);
                let (rec_eng, _, halted) =
                    TigerDoubleEntryEngine::recover_from_faulty_wal(&init_accounts, &wal);
                torn_tails_detected += halted;
                if !rec_eng.verify_zero_sum_conservation().4 {
                    conservation_failures += 1;
                }
            } else {
                // Fault B: 1-bit silent NAND bit-rot mid-stream!
                let pos = (next_u64() as usize) % wal.len();
                wal[pos] ^= 1u8 << ((next_u64() % 8) as u8);
                let (rec_eng, _, halted) =
                    TigerDoubleEntryEngine::recover_from_faulty_wal(&init_accounts, &wal);
                bit_flips_detected += halted;
                if !rec_eng.verify_zero_sum_conservation().4 {
                    conservation_failures += 1;
                }
            }
        }
    }

    VoprSimulationReceipt {
        seeds_tested: seeds,
        total_transfers_simulated: seeds * t_per_seed,
        torn_tails_detected,
        bit_flips_detected,
        conservation_failures,
        invariant_clean: conservation_failures == 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiger_linked_chain_rollback_two_phase_pending_and_vopr_conservation() {
        let sim = run_vopr_deterministic_fault_campaign(32, 64);
        assert!(sim.invariant_clean);
        assert_eq!(sim.conservation_failures, 0);
        assert!(sim.torn_tails_detected > 0);
        assert!(sim.bit_flips_detected > 0);
    }
}
