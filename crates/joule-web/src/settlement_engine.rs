//! Settlement Engine — T+1/T+2 cycle management, settlement instruction
//! matching, partial settlement, settlement netting, fail management,
//! and SettlementConfig builder.
//!
//! Pure-Rust settlement pipeline that processes trade instructions through
//! matching, netting, and settlement cycles with configurable fail policies.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SettlementError {
    InvalidInstruction(String),
    InsufficientPosition(String),
    SettlementFailed(String),
    CycleViolation(String),
    NettingError(String),
    ConfigError(String),
}

impl fmt::Display for SettlementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInstruction(s) => write!(f, "invalid instruction: {s}"),
            Self::InsufficientPosition(s) => write!(f, "insufficient position: {s}"),
            Self::SettlementFailed(s) => write!(f, "settlement failed: {s}"),
            Self::CycleViolation(s) => write!(f, "cycle violation: {s}"),
            Self::NettingError(s) => write!(f, "netting error: {s}"),
            Self::ConfigError(s) => write!(f, "config error: {s}"),
        }
    }
}

impl std::error::Error for SettlementError {}

// ── Settlement cycle ────────────────────────────────────────────

/// Settlement cycle type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SettlementCycle {
    /// Trade date + 0 (same day).
    T0,
    /// Trade date + 1 business day.
    T1,
    /// Trade date + 2 business days.
    T2,
    /// Trade date + 3 business days (legacy).
    T3,
}

impl SettlementCycle {
    pub fn days(self) -> u32 {
        match self {
            Self::T0 => 0,
            Self::T1 => 1,
            Self::T2 => 2,
            Self::T3 => 3,
        }
    }
}

impl fmt::Display for SettlementCycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::T0 => write!(f, "T+0"),
            Self::T1 => write!(f, "T+1"),
            Self::T2 => write!(f, "T+2"),
            Self::T3 => write!(f, "T+3"),
        }
    }
}

// ── Instruction side & status ───────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionSide {
    Deliver,
    Receive,
}

impl fmt::Display for InstructionSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deliver => write!(f, "DELIVER"),
            Self::Receive => write!(f, "RECEIVE"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionStatus {
    Pending,
    Matched,
    PartiallySettled,
    Settled,
    Failed,
    Cancelled,
}

impl fmt::Display for InstructionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Pending => "PENDING",
            Self::Matched => "MATCHED",
            Self::PartiallySettled => "PARTIAL",
            Self::Settled => "SETTLED",
            Self::Failed => "FAILED",
            Self::Cancelled => "CANCELLED",
        };
        write!(f, "{label}")
    }
}

// ── Fail policy ─────────────────────────────────────────────────

/// Policy applied when settlement fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailPolicy {
    /// Retry next cycle automatically.
    AutoRetry,
    /// Apply a penalty and retry.
    PenaltyRetry,
    /// Cancel after max retries.
    CancelAfterRetries,
    /// Partial settlement allowed.
    PartialFill,
}

impl fmt::Display for FailPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AutoRetry => write!(f, "AutoRetry"),
            Self::PenaltyRetry => write!(f, "PenaltyRetry"),
            Self::CancelAfterRetries => write!(f, "CancelAfterRetries"),
            Self::PartialFill => write!(f, "PartialFill"),
        }
    }
}

// ── Settlement instruction ──────────────────────────────────────

/// A single settlement instruction.
#[derive(Debug, Clone)]
pub struct SettlementInstruction {
    pub id: u64,
    pub trade_id: u64,
    pub security_id: String,
    pub side: InstructionSide,
    pub quantity: f64,
    pub settled_quantity: f64,
    pub price: f64,
    pub counterparty: String,
    pub account: String,
    pub trade_date: u64,
    pub settlement_date: u64,
    pub status: InstructionStatus,
    pub retry_count: u32,
}

impl SettlementInstruction {
    pub fn new(
        id: u64,
        trade_id: u64,
        security_id: &str,
        side: InstructionSide,
        quantity: f64,
        price: f64,
        counterparty: &str,
        account: &str,
        trade_date: u64,
        settlement_date: u64,
    ) -> Self {
        Self {
            id,
            trade_id,
            security_id: security_id.to_string(),
            side,
            quantity,
            settled_quantity: 0.0,
            price,
            counterparty: counterparty.to_string(),
            account: account.to_string(),
            trade_date,
            settlement_date,
            status: InstructionStatus::Pending,
            retry_count: 0,
        }
    }

    /// Cash amount for this instruction.
    pub fn cash_amount(&self) -> f64 {
        self.quantity * self.price
    }

    /// Outstanding quantity still to settle.
    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.settled_quantity
    }

    /// Whether the instruction is fully settled.
    pub fn is_fully_settled(&self) -> bool {
        (self.remaining_quantity()).abs() < 1e-9
    }
}

impl fmt::Display for SettlementInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SI-{}: {} {} x {:.2} @ {:.4} [{}]",
            self.id, self.side, self.security_id, self.quantity, self.price, self.status
        )
    }
}

// ── Match result ────────────────────────────────────────────────

/// Result of matching two instructions.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub deliver_id: u64,
    pub receive_id: u64,
    pub security_id: String,
    pub matched_quantity: f64,
    pub matched_price: f64,
    pub is_exact: bool,
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_exact { "exact" } else { "partial" };
        write!(
            f,
            "Match({kind}): D={} R={} qty={:.2}",
            self.deliver_id, self.receive_id, self.matched_quantity
        )
    }
}

// ── Netting result ──────────────────────────────────────────────

/// Net obligation after netting a set of instructions.
#[derive(Debug, Clone)]
pub struct NetObligation {
    pub security_id: String,
    pub counterparty: String,
    pub net_quantity: f64,
    pub net_cash: f64,
    pub instruction_count: usize,
}

impl fmt::Display for NetObligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Net({}, {}): qty={:.2} cash={:.2} from {} instructions",
            self.security_id,
            self.counterparty,
            self.net_quantity,
            self.net_cash,
            self.instruction_count,
        )
    }
}

// ── Settlement config builder ───────────────────────────────────

/// Configuration for the settlement engine.
#[derive(Debug, Clone)]
pub struct SettlementConfig {
    pub default_cycle: SettlementCycle,
    pub fail_policy: FailPolicy,
    pub max_retries: u32,
    pub penalty_rate_bps: f64,
    pub partial_settlement_enabled: bool,
    pub netting_enabled: bool,
    pub price_tolerance: f64,
    pub quantity_tolerance: f64,
}

impl Default for SettlementConfig {
    fn default() -> Self {
        Self {
            default_cycle: SettlementCycle::T2,
            fail_policy: FailPolicy::AutoRetry,
            max_retries: 3,
            penalty_rate_bps: 5.0,
            partial_settlement_enabled: false,
            netting_enabled: true,
            price_tolerance: 0.0001,
            quantity_tolerance: 0.01,
        }
    }
}

impl SettlementConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cycle(mut self, cycle: SettlementCycle) -> Self {
        self.default_cycle = cycle;
        self
    }

    pub fn with_fail_policy(mut self, policy: FailPolicy) -> Self {
        self.fail_policy = policy;
        self
    }

    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    pub fn with_penalty_rate_bps(mut self, bps: f64) -> Self {
        self.penalty_rate_bps = bps;
        self
    }

    pub fn with_partial_settlement(mut self, enabled: bool) -> Self {
        self.partial_settlement_enabled = enabled;
        self
    }

    pub fn with_netting(mut self, enabled: bool) -> Self {
        self.netting_enabled = enabled;
        self
    }

    pub fn with_price_tolerance(mut self, tol: f64) -> Self {
        self.price_tolerance = tol;
        self
    }

    pub fn with_quantity_tolerance(mut self, tol: f64) -> Self {
        self.quantity_tolerance = tol;
        self
    }
}

impl fmt::Display for SettlementConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SettlementConfig(cycle={}, fail={}, retries={}, netting={})",
            self.default_cycle, self.fail_policy, self.max_retries, self.netting_enabled
        )
    }
}

// ── Settlement engine ───────────────────────────────────────────

/// Core settlement engine.
pub struct SettlementEngine {
    config: SettlementConfig,
    instructions: Vec<SettlementInstruction>,
    matches: Vec<MatchResult>,
    net_obligations: Vec<NetObligation>,
    penalties_accrued: f64,
    current_date: u64,
}

impl SettlementEngine {
    pub fn new(config: SettlementConfig) -> Self {
        Self {
            config,
            instructions: Vec::new(),
            matches: Vec::new(),
            net_obligations: Vec::new(),
            penalties_accrued: 0.0,
            current_date: 0,
        }
    }

    pub fn set_current_date(&mut self, date: u64) {
        self.current_date = date;
    }

    /// Submit a new settlement instruction.
    pub fn submit_instruction(
        &mut self,
        instr: SettlementInstruction,
    ) -> Result<u64, SettlementError> {
        if instr.quantity <= 0.0 {
            return Err(SettlementError::InvalidInstruction(
                "quantity must be positive".into(),
            ));
        }
        if instr.price < 0.0 {
            return Err(SettlementError::InvalidInstruction(
                "price must be non-negative".into(),
            ));
        }
        let id = instr.id;
        self.instructions.push(instr);
        Ok(id)
    }

    /// Compute settlement date from trade date using configured cycle.
    pub fn compute_settlement_date(&self, trade_date: u64) -> u64 {
        trade_date + self.config.default_cycle.days() as u64
    }

    /// Run the matching engine across pending instructions.
    pub fn run_matching(&mut self) -> Vec<MatchResult> {
        let mut new_matches = Vec::new();
        let len = self.instructions.len();

        for i in 0..len {
            if self.instructions[i].status != InstructionStatus::Pending {
                continue;
            }
            if self.instructions[i].side != InstructionSide::Deliver {
                continue;
            }

            for j in 0..len {
                if i == j {
                    continue;
                }
                if self.instructions[j].status != InstructionStatus::Pending {
                    continue;
                }
                if self.instructions[j].side != InstructionSide::Receive {
                    continue;
                }
                if self.instructions[i].security_id != self.instructions[j].security_id {
                    continue;
                }

                let price_diff =
                    (self.instructions[i].price - self.instructions[j].price).abs();
                if price_diff > self.config.price_tolerance {
                    continue;
                }

                let qty_diff = (self.instructions[i].remaining_quantity()
                    - self.instructions[j].remaining_quantity())
                .abs();
                let is_exact = qty_diff <= self.config.quantity_tolerance;
                let matched_qty = self.instructions[i]
                    .remaining_quantity()
                    .min(self.instructions[j].remaining_quantity());

                if matched_qty <= 0.0 {
                    continue;
                }

                let m = MatchResult {
                    deliver_id: self.instructions[i].id,
                    receive_id: self.instructions[j].id,
                    security_id: self.instructions[i].security_id.clone(),
                    matched_quantity: matched_qty,
                    matched_price: self.instructions[i].price,
                    is_exact,
                };
                new_matches.push(m);

                self.instructions[i].status = InstructionStatus::Matched;
                self.instructions[j].status = InstructionStatus::Matched;
                break;
            }
        }

        self.matches.extend(new_matches.clone());
        new_matches
    }

    /// Settle matched instructions. Returns number of settled instructions.
    pub fn settle_matched(&mut self) -> Result<usize, SettlementError> {
        let mut settled_count = 0usize;

        for m in &self.matches.clone() {
            let deliver = self
                .instructions
                .iter_mut()
                .find(|i| i.id == m.deliver_id);
            if let Some(d) = deliver {
                if d.status == InstructionStatus::Matched
                    || d.status == InstructionStatus::PartiallySettled
                {
                    d.settled_quantity += m.matched_quantity;
                    d.status = if d.is_fully_settled() {
                        InstructionStatus::Settled
                    } else {
                        InstructionStatus::PartiallySettled
                    };
                    if d.is_fully_settled() {
                        settled_count += 1;
                    }
                }
            }

            let receive = self
                .instructions
                .iter_mut()
                .find(|i| i.id == m.receive_id);
            if let Some(r) = receive {
                if r.status == InstructionStatus::Matched
                    || r.status == InstructionStatus::PartiallySettled
                {
                    r.settled_quantity += m.matched_quantity;
                    r.status = if r.is_fully_settled() {
                        InstructionStatus::Settled
                    } else {
                        InstructionStatus::PartiallySettled
                    };
                    if r.is_fully_settled() {
                        settled_count += 1;
                    }
                }
            }
        }

        Ok(settled_count)
    }

    /// Process fails: apply penalties and retry or cancel.
    pub fn process_fails(&mut self) -> Vec<u64> {
        let mut failed_ids = Vec::new();

        for instr in &mut self.instructions {
            let is_overdue = instr.settlement_date <= self.current_date
                && instr.status != InstructionStatus::Settled
                && instr.status != InstructionStatus::Cancelled;

            if !is_overdue {
                continue;
            }

            instr.retry_count += 1;

            match self.config.fail_policy {
                FailPolicy::AutoRetry => {
                    if instr.retry_count > self.config.max_retries {
                        instr.status = InstructionStatus::Failed;
                        failed_ids.push(instr.id);
                    } else {
                        instr.status = InstructionStatus::Pending;
                    }
                }
                FailPolicy::PenaltyRetry => {
                    let penalty =
                        instr.cash_amount() * self.config.penalty_rate_bps / 10_000.0;
                    self.penalties_accrued += penalty;
                    if instr.retry_count > self.config.max_retries {
                        instr.status = InstructionStatus::Failed;
                        failed_ids.push(instr.id);
                    } else {
                        instr.status = InstructionStatus::Pending;
                    }
                }
                FailPolicy::CancelAfterRetries => {
                    if instr.retry_count > self.config.max_retries {
                        instr.status = InstructionStatus::Cancelled;
                        failed_ids.push(instr.id);
                    } else {
                        instr.status = InstructionStatus::Pending;
                    }
                }
                FailPolicy::PartialFill => {
                    if instr.settled_quantity > 0.0 {
                        instr.status = InstructionStatus::PartiallySettled;
                    } else {
                        instr.status = InstructionStatus::Failed;
                        failed_ids.push(instr.id);
                    }
                }
            }
        }

        failed_ids
    }

    /// Compute net obligations by security and counterparty.
    pub fn compute_netting(&mut self) -> Vec<NetObligation> {
        let mut nets: HashMap<(String, String), (f64, f64, usize)> = HashMap::new();

        for instr in &self.instructions {
            if instr.status == InstructionStatus::Cancelled
                || instr.status == InstructionStatus::Settled
            {
                continue;
            }
            let key = (instr.security_id.clone(), instr.counterparty.clone());
            let entry = nets.entry(key).or_insert((0.0, 0.0, 0));
            let sign = match instr.side {
                InstructionSide::Deliver => -1.0,
                InstructionSide::Receive => 1.0,
            };
            entry.0 += sign * instr.remaining_quantity();
            entry.1 += sign * instr.remaining_quantity() * instr.price;
            entry.2 += 1;
        }

        self.net_obligations = nets
            .into_iter()
            .map(|((sec, cpty), (qty, cash, cnt))| NetObligation {
                security_id: sec,
                counterparty: cpty,
                net_quantity: qty,
                net_cash: cash,
                instruction_count: cnt,
            })
            .collect();

        self.net_obligations.clone()
    }

    pub fn instructions(&self) -> &[SettlementInstruction] {
        &self.instructions
    }

    pub fn matches(&self) -> &[MatchResult] {
        &self.matches
    }

    pub fn penalties_accrued(&self) -> f64 {
        self.penalties_accrued
    }

    pub fn config(&self) -> &SettlementConfig {
        &self.config
    }
}

impl fmt::Display for SettlementEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SettlementEngine(instructions={}, matches={}, penalties={:.2})",
            self.instructions.len(),
            self.matches.len(),
            self.penalties_accrued,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> SettlementEngine {
        SettlementEngine::new(SettlementConfig::default())
    }

    fn deliver(id: u64, sec: &str, qty: f64, price: f64) -> SettlementInstruction {
        SettlementInstruction::new(id, id, sec, InstructionSide::Deliver, qty, price, "CP1", "ACC1", 100, 102)
    }

    fn receive(id: u64, sec: &str, qty: f64, price: f64) -> SettlementInstruction {
        SettlementInstruction::new(id, id, sec, InstructionSide::Receive, qty, price, "CP1", "ACC2", 100, 102)
    }

    #[test]
    fn test_config_builder() {
        let cfg = SettlementConfig::new()
            .with_cycle(SettlementCycle::T1)
            .with_fail_policy(FailPolicy::PenaltyRetry)
            .with_max_retries(5)
            .with_partial_settlement(true);
        assert_eq!(cfg.default_cycle, SettlementCycle::T1);
        assert_eq!(cfg.fail_policy, FailPolicy::PenaltyRetry);
        assert_eq!(cfg.max_retries, 5);
        assert!(cfg.partial_settlement_enabled);
    }

    #[test]
    fn test_settlement_cycle_days() {
        assert_eq!(SettlementCycle::T0.days(), 0);
        assert_eq!(SettlementCycle::T1.days(), 1);
        assert_eq!(SettlementCycle::T2.days(), 2);
        assert_eq!(SettlementCycle::T3.days(), 3);
    }

    #[test]
    fn test_compute_settlement_date() {
        let eng = make_engine();
        assert_eq!(eng.compute_settlement_date(100), 102);
    }

    #[test]
    fn test_submit_valid_instruction() {
        let mut eng = make_engine();
        let r = eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0));
        assert!(r.is_ok());
        assert_eq!(eng.instructions().len(), 1);
    }

    #[test]
    fn test_submit_negative_quantity_fails() {
        let mut eng = make_engine();
        let r = eng.submit_instruction(deliver(1, "AAPL", -10.0, 150.0));
        assert!(r.is_err());
    }

    #[test]
    fn test_submit_negative_price_fails() {
        let mut eng = make_engine();
        let r = eng.submit_instruction(deliver(1, "AAPL", 10.0, -1.0));
        assert!(r.is_err());
    }

    #[test]
    fn test_exact_matching() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.submit_instruction(receive(2, "AAPL", 100.0, 150.0)).unwrap();
        let matches = eng.run_matching();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].is_exact);
    }

    #[test]
    fn test_no_match_different_security() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.submit_instruction(receive(2, "GOOG", 100.0, 150.0)).unwrap();
        let matches = eng.run_matching();
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_no_match_price_mismatch() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.submit_instruction(receive(2, "AAPL", 100.0, 200.0)).unwrap();
        let matches = eng.run_matching();
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_settle_matched() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.submit_instruction(receive(2, "AAPL", 100.0, 150.0)).unwrap();
        eng.run_matching();
        let settled = eng.settle_matched().unwrap();
        assert_eq!(settled, 2);
    }

    #[test]
    fn test_instruction_cash_amount() {
        let instr = deliver(1, "AAPL", 100.0, 150.0);
        assert!((instr.cash_amount() - 15000.0).abs() < 1e-9);
    }

    #[test]
    fn test_remaining_quantity() {
        let mut instr = deliver(1, "AAPL", 100.0, 150.0);
        instr.settled_quantity = 40.0;
        assert!((instr.remaining_quantity() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_netting_computation() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.submit_instruction(receive(2, "AAPL", 60.0, 150.0)).unwrap();
        let nets = eng.compute_netting();
        assert!(!nets.is_empty());
    }

    #[test]
    fn test_fail_auto_retry() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.set_current_date(103);
        let failed = eng.process_fails();
        assert!(failed.is_empty());
        assert_eq!(eng.instructions()[0].status, InstructionStatus::Pending);
    }

    #[test]
    fn test_fail_exceeds_max_retries() {
        let mut eng = SettlementEngine::new(
            SettlementConfig::new().with_max_retries(1),
        );
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.set_current_date(103);
        eng.process_fails(); // retry 1
        eng.process_fails(); // retry 2 → exceeds
        assert_eq!(eng.instructions()[0].status, InstructionStatus::Failed);
    }

    #[test]
    fn test_penalty_accrual() {
        let mut eng = SettlementEngine::new(
            SettlementConfig::new()
                .with_fail_policy(FailPolicy::PenaltyRetry)
                .with_penalty_rate_bps(10.0),
        );
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.set_current_date(103);
        eng.process_fails();
        assert!(eng.penalties_accrued() > 0.0);
    }

    #[test]
    fn test_cancel_after_retries() {
        let mut eng = SettlementEngine::new(
            SettlementConfig::new()
                .with_fail_policy(FailPolicy::CancelAfterRetries)
                .with_max_retries(0),
        );
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.set_current_date(103);
        eng.process_fails();
        assert_eq!(eng.instructions()[0].status, InstructionStatus::Cancelled);
    }

    #[test]
    fn test_display_impls() {
        let cfg = SettlementConfig::default();
        let s = format!("{cfg}");
        assert!(s.contains("SettlementConfig"));

        let instr = deliver(1, "AAPL", 100.0, 150.0);
        let s = format!("{instr}");
        assert!(s.contains("DELIVER"));

        assert_eq!(format!("{}", SettlementCycle::T2), "T+2");
    }

    #[test]
    fn test_engine_display() {
        let eng = make_engine();
        let s = format!("{eng}");
        assert!(s.contains("SettlementEngine"));
    }

    #[test]
    fn test_partial_match_quantity() {
        let mut eng = make_engine();
        eng.submit_instruction(deliver(1, "AAPL", 100.0, 150.0)).unwrap();
        eng.submit_instruction(receive(2, "AAPL", 60.0, 150.0)).unwrap();
        let matches = eng.run_matching();
        assert_eq!(matches.len(), 1);
        assert!(!matches[0].is_exact);
        assert!((matches[0].matched_quantity - 60.0).abs() < 1e-9);
    }
}
