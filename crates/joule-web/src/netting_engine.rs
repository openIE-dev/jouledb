//! Netting Engine — bilateral netting, multilateral netting, close-out
//! netting, payment netting, novation netting, and netting set optimization.
//!
//! Pure-Rust netting engine that reduces gross obligations to net positions
//! across bilateral and multilateral participant sets.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NettingError {
    InvalidObligation(String),
    ParticipantNotFound(String),
    NettingSetEmpty(String),
    OptimizationFailed(String),
}

impl fmt::Display for NettingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidObligation(s) => write!(f, "invalid obligation: {s}"),
            Self::ParticipantNotFound(s) => write!(f, "participant not found: {s}"),
            Self::NettingSetEmpty(s) => write!(f, "netting set empty: {s}"),
            Self::OptimizationFailed(s) => write!(f, "optimization failed: {s}"),
        }
    }
}

impl std::error::Error for NettingError {}

// ── Netting mode ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NettingMode {
    /// Net between two parties.
    Bilateral,
    /// Net across all parties through a central node.
    Multilateral,
    /// Net obligations upon default/termination.
    CloseOut,
    /// Net cash payment obligations.
    Payment,
    /// Replace multiple obligations with one (novation).
    Novation,
}

impl fmt::Display for NettingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bilateral => write!(f, "Bilateral"),
            Self::Multilateral => write!(f, "Multilateral"),
            Self::CloseOut => write!(f, "CloseOut"),
            Self::Payment => write!(f, "Payment"),
            Self::Novation => write!(f, "Novation"),
        }
    }
}

// ── Obligation ──────────────────────────────────────────────────

/// A directed obligation from one party to another.
#[derive(Debug, Clone)]
pub struct Obligation {
    pub id: u64,
    pub from: String,
    pub to: String,
    pub currency: String,
    pub amount: f64,
    pub security_id: Option<String>,
    pub quantity: Option<f64>,
    pub is_active: bool,
}

impl Obligation {
    /// Cash obligation.
    pub fn cash(id: u64, from: &str, to: &str, currency: &str, amount: f64) -> Self {
        Self {
            id,
            from: from.to_string(),
            to: to.to_string(),
            currency: currency.to_string(),
            amount,
            security_id: None,
            quantity: None,
            is_active: true,
        }
    }

    /// Securities obligation.
    pub fn securities(
        id: u64,
        from: &str,
        to: &str,
        security_id: &str,
        quantity: f64,
        cash: f64,
        currency: &str,
    ) -> Self {
        Self {
            id,
            from: from.to_string(),
            to: to.to_string(),
            currency: currency.to_string(),
            amount: cash,
            security_id: Some(security_id.to_string()),
            quantity: Some(quantity),
            is_active: true,
        }
    }
}

impl fmt::Display for Obligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref sec) = self.security_id {
            write!(
                f,
                "Oblig-{}: {}→{} {} qty={:.2} cash={:.2}{}",
                self.id,
                self.from,
                self.to,
                sec,
                self.quantity.unwrap_or(0.0),
                self.amount,
                self.currency,
            )
        } else {
            write!(
                f,
                "Oblig-{}: {}→{} {:.2}{}",
                self.id, self.from, self.to, self.amount, self.currency,
            )
        }
    }
}

// ── Net position ────────────────────────────────────────────────

/// Net position after netting.
#[derive(Debug, Clone)]
pub struct NetPosition {
    pub party_a: String,
    pub party_b: String,
    pub currency: String,
    pub net_cash: f64,
    pub security_nets: HashMap<String, f64>,
    pub obligation_count: usize,
    pub gross_exposure: f64,
    pub net_exposure: f64,
}

impl NetPosition {
    /// Netting efficiency as a percentage reduction.
    pub fn netting_efficiency(&self) -> f64 {
        if self.gross_exposure == 0.0 {
            return 0.0;
        }
        (1.0 - self.net_exposure / self.gross_exposure) * 100.0
    }

    /// Direction of the net cash flow (who pays whom).
    pub fn payer(&self) -> &str {
        if self.net_cash >= 0.0 {
            &self.party_a
        } else {
            &self.party_b
        }
    }

    pub fn receiver(&self) -> &str {
        if self.net_cash >= 0.0 {
            &self.party_b
        } else {
            &self.party_a
        }
    }
}

impl fmt::Display for NetPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Net({}<->{}): cash={:.2}{} efficiency={:.1}%",
            self.party_a,
            self.party_b,
            self.net_cash.abs(),
            self.currency,
            self.netting_efficiency(),
        )
    }
}

// ── Multilateral net result ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MultilateralNetResult {
    pub participant: String,
    pub currency: String,
    pub net_cash: f64,
    pub security_nets: HashMap<String, f64>,
    pub contribution_count: usize,
}

impl MultilateralNetResult {
    pub fn is_net_payer(&self) -> bool {
        self.net_cash < 0.0
    }

    pub fn is_net_receiver(&self) -> bool {
        self.net_cash > 0.0
    }
}

impl fmt::Display for MultilateralNetResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dir = if self.net_cash >= 0.0 { "receives" } else { "pays" };
        write!(
            f,
            "{} {} {:.2}{}",
            self.participant,
            dir,
            self.net_cash.abs(),
            self.currency,
        )
    }
}

// ── Close-out result ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CloseOutResult {
    pub defaulter: String,
    pub counterparty: String,
    pub terminated_count: usize,
    pub net_closeout_amount: f64,
    pub currency: String,
}

impl fmt::Display for CloseOutResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CloseOut({}->{}: terminated={}, net={:.2}{})",
            self.defaulter,
            self.counterparty,
            self.terminated_count,
            self.net_closeout_amount,
            self.currency,
        )
    }
}

// ── Netting set statistics ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NettingSetStats {
    pub mode: NettingMode,
    pub obligation_count: usize,
    pub participant_count: usize,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub netting_ratio: f64,
}

impl NettingSetStats {
    pub fn reduction_pct(&self) -> f64 {
        if self.gross_exposure == 0.0 {
            return 0.0;
        }
        (1.0 - self.netting_ratio) * 100.0
    }
}

impl fmt::Display for NettingSetStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NettingStats(mode={}, obligs={}, gross={:.2}, net={:.2}, reduction={:.1}%)",
            self.mode,
            self.obligation_count,
            self.gross_exposure,
            self.net_exposure,
            self.reduction_pct(),
        )
    }
}

// ── Netting engine ──────────────────────────────────────────────

pub struct NettingEngine {
    obligations: Vec<Obligation>,
    next_id: u64,
}

impl NettingEngine {
    pub fn new() -> Self {
        Self {
            obligations: Vec::new(),
            next_id: 1,
        }
    }

    pub fn add_obligation(&mut self, mut oblig: Obligation) -> u64 {
        oblig.id = self.next_id;
        self.next_id += 1;
        let id = oblig.id;
        self.obligations.push(oblig);
        id
    }

    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }

    /// Bilateral netting: net all obligations between two specific parties.
    pub fn bilateral_net(
        &self,
        party_a: &str,
        party_b: &str,
        currency: &str,
    ) -> Result<NetPosition, NettingError> {
        let relevant: Vec<&Obligation> = self
            .obligations
            .iter()
            .filter(|o| {
                o.is_active
                    && o.currency == currency
                    && ((o.from == party_a && o.to == party_b)
                        || (o.from == party_b && o.to == party_a))
            })
            .collect();

        if relevant.is_empty() {
            return Err(NettingError::NettingSetEmpty(
                format!("{party_a} <-> {party_b}"),
            ));
        }

        let mut net_cash = 0.0f64;
        let mut sec_nets: HashMap<String, f64> = HashMap::new();
        let mut gross = 0.0f64;

        for o in &relevant {
            let sign = if o.from == party_a { 1.0 } else { -1.0 };
            net_cash += sign * o.amount;
            gross += o.amount.abs();

            if let (Some(sec), Some(qty)) = (&o.security_id, o.quantity) {
                let entry = sec_nets.entry(sec.clone()).or_insert(0.0);
                *entry += sign * qty;
            }
        }

        let net_exp = net_cash.abs();
        Ok(NetPosition {
            party_a: party_a.to_string(),
            party_b: party_b.to_string(),
            currency: currency.to_string(),
            net_cash,
            security_nets: sec_nets,
            obligation_count: relevant.len(),
            gross_exposure: gross,
            net_exposure: net_exp,
        })
    }

    /// Multilateral netting: compute net position for each participant.
    pub fn multilateral_net(
        &self,
        currency: &str,
    ) -> Vec<MultilateralNetResult> {
        let active: Vec<&Obligation> = self
            .obligations
            .iter()
            .filter(|o| o.is_active && o.currency == currency)
            .collect();

        let mut cash_map: HashMap<String, f64> = HashMap::new();
        let mut sec_map: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let mut count_map: HashMap<String, usize> = HashMap::new();

        for o in &active {
            *cash_map.entry(o.from.clone()).or_insert(0.0) -= o.amount;
            *cash_map.entry(o.to.clone()).or_insert(0.0) += o.amount;
            *count_map.entry(o.from.clone()).or_insert(0) += 1;
            *count_map.entry(o.to.clone()).or_insert(0) += 1;

            if let (Some(sec), Some(qty)) = (&o.security_id, o.quantity) {
                let from_entry = sec_map.entry(o.from.clone()).or_default();
                *from_entry.entry(sec.clone()).or_insert(0.0) -= qty;
                let to_entry = sec_map.entry(o.to.clone()).or_default();
                *to_entry.entry(sec.clone()).or_insert(0.0) += qty;
            }
        }

        cash_map
            .into_iter()
            .map(|(participant, net_cash)| {
                let security_nets = sec_map.remove(&participant).unwrap_or_default();
                let contribution_count = count_map.get(&participant).copied().unwrap_or(0);
                MultilateralNetResult {
                    participant,
                    currency: currency.to_string(),
                    net_cash,
                    security_nets,
                    contribution_count,
                }
            })
            .collect()
    }

    /// Close-out netting: terminate all obligations with a defaulter.
    pub fn close_out_net(
        &mut self,
        defaulter: &str,
        currency: &str,
    ) -> Vec<CloseOutResult> {
        let mut results_map: HashMap<String, (usize, f64)> = HashMap::new();

        for o in &mut self.obligations {
            if !o.is_active || o.currency != currency {
                continue;
            }
            if o.from != defaulter && o.to != defaulter {
                continue;
            }

            let counterparty = if o.from == defaulter {
                o.to.clone()
            } else {
                o.from.clone()
            };
            let sign = if o.from == defaulter { -1.0 } else { 1.0 };

            let entry = results_map.entry(counterparty).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += sign * o.amount;
            o.is_active = false;
        }

        results_map
            .into_iter()
            .map(|(cpty, (count, net))| CloseOutResult {
                defaulter: defaulter.to_string(),
                counterparty: cpty,
                terminated_count: count,
                net_closeout_amount: net,
                currency: currency.to_string(),
            })
            .collect()
    }

    /// Payment netting: net cash flows for a specific settlement date bucket.
    pub fn payment_net(
        &self,
        currency: &str,
    ) -> Vec<NetPosition> {
        // Collect unique bilateral pairs
        let mut pairs: Vec<(String, String)> = Vec::new();
        for o in &self.obligations {
            if !o.is_active || o.currency != currency || o.security_id.is_some() {
                continue;
            }
            let pair = if o.from < o.to {
                (o.from.clone(), o.to.clone())
            } else {
                (o.to.clone(), o.from.clone())
            };
            if !pairs.contains(&pair) {
                pairs.push(pair);
            }
        }

        pairs
            .iter()
            .filter_map(|(a, b)| self.bilateral_net(a, b, currency).ok())
            .collect()
    }

    /// Novation netting: replace obligations between two parties with a single net.
    pub fn novation_net(
        &mut self,
        party_a: &str,
        party_b: &str,
        currency: &str,
    ) -> Result<Obligation, NettingError> {
        let net = self.bilateral_net(party_a, party_b, currency)?;

        // Deactivate original obligations
        for o in &mut self.obligations {
            if !o.is_active || o.currency != currency {
                continue;
            }
            if (o.from == party_a && o.to == party_b)
                || (o.from == party_b && o.to == party_a)
            {
                o.is_active = false;
            }
        }

        // Create single net obligation
        let (from, to, amt) = if net.net_cash >= 0.0 {
            (party_a.to_string(), party_b.to_string(), net.net_cash)
        } else {
            (party_b.to_string(), party_a.to_string(), net.net_cash.abs())
        };

        let novated = Obligation::cash(0, &from, &to, currency, amt);
        let id = self.add_obligation(novated);

        Ok(self.obligations.iter().find(|o| o.id == id).unwrap().clone())
    }

    /// Compute overall netting set statistics.
    pub fn compute_stats(&self, mode: NettingMode, currency: &str) -> NettingSetStats {
        let active: Vec<&Obligation> = self
            .obligations
            .iter()
            .filter(|o| o.is_active && o.currency == currency)
            .collect();

        let mut participants = std::collections::HashSet::new();
        let mut gross = 0.0f64;

        for o in &active {
            participants.insert(o.from.clone());
            participants.insert(o.to.clone());
            gross += o.amount.abs();
        }

        let multilateral = self.multilateral_net(currency);
        let net: f64 = multilateral.iter().map(|m| m.net_cash.abs()).sum::<f64>() / 2.0;
        let ratio = if gross > 0.0 { net / gross } else { 0.0 };

        NettingSetStats {
            mode,
            obligation_count: active.len(),
            participant_count: participants.len(),
            gross_exposure: gross,
            net_exposure: net,
            netting_ratio: ratio,
        }
    }
}

impl fmt::Display for NettingEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let active = self.obligations.iter().filter(|o| o.is_active).count();
        write!(
            f,
            "NettingEngine(obligations={}, active={})",
            self.obligations.len(),
            active,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_engine() -> NettingEngine {
        let mut ne = NettingEngine::new();
        ne.add_obligation(Obligation::cash(0, "A", "B", "USD", 1000.0));
        ne.add_obligation(Obligation::cash(0, "B", "A", "USD", 600.0));
        ne.add_obligation(Obligation::cash(0, "A", "C", "USD", 500.0));
        ne.add_obligation(Obligation::cash(0, "C", "A", "USD", 300.0));
        ne
    }

    #[test]
    fn test_add_obligation() {
        let mut ne = NettingEngine::new();
        let id = ne.add_obligation(Obligation::cash(0, "A", "B", "USD", 100.0));
        assert_eq!(id, 1);
        assert_eq!(ne.obligations().len(), 1);
    }

    #[test]
    fn test_bilateral_net_simple() {
        let ne = setup_engine();
        let net = ne.bilateral_net("A", "B", "USD").unwrap();
        assert!((net.net_cash - 400.0).abs() < 1e-9);
        assert_eq!(net.obligation_count, 2);
    }

    #[test]
    fn test_bilateral_net_efficiency() {
        let ne = setup_engine();
        let net = ne.bilateral_net("A", "B", "USD").unwrap();
        assert!(net.netting_efficiency() > 0.0);
    }

    #[test]
    fn test_bilateral_net_empty() {
        let ne = setup_engine();
        let r = ne.bilateral_net("A", "B", "EUR");
        assert!(r.is_err());
    }

    #[test]
    fn test_multilateral_net() {
        let ne = setup_engine();
        let results = ne.multilateral_net("USD");
        assert!(!results.is_empty());
        let total: f64 = results.iter().map(|r| r.net_cash).sum();
        assert!(total.abs() < 1e-9); // Must balance to zero
    }

    #[test]
    fn test_multilateral_net_payer_receiver() {
        let ne = setup_engine();
        let results = ne.multilateral_net("USD");
        let payers: Vec<_> = results.iter().filter(|r| r.is_net_payer()).collect();
        let receivers: Vec<_> = results.iter().filter(|r| r.is_net_receiver()).collect();
        assert!(!payers.is_empty() || !receivers.is_empty());
    }

    #[test]
    fn test_close_out_netting() {
        let mut ne = setup_engine();
        let results = ne.close_out_net("A", "USD");
        assert!(!results.is_empty());
        // All A obligations should be deactivated
        let active_a = ne.obligations().iter().filter(|o| {
            o.is_active && (o.from == "A" || o.to == "A")
        }).count();
        assert_eq!(active_a, 0);
    }

    #[test]
    fn test_payment_netting() {
        let ne = setup_engine();
        let nets = ne.payment_net("USD");
        assert!(!nets.is_empty());
    }

    #[test]
    fn test_novation_netting() {
        let mut ne = setup_engine();
        let novated = ne.novation_net("A", "B", "USD").unwrap();
        assert!((novated.amount - 400.0).abs() < 1e-9);
        // Original obligations deactivated
        let originals: Vec<_> = ne.obligations().iter()
            .filter(|o| o.is_active && ((o.from == "A" && o.to == "B") || (o.from == "B" && o.to == "A")))
            .collect();
        // Only the novated one should remain active
        assert_eq!(originals.len(), 1);
    }

    #[test]
    fn test_netting_stats() {
        let ne = setup_engine();
        let stats = ne.compute_stats(NettingMode::Multilateral, "USD");
        assert_eq!(stats.obligation_count, 4);
        assert_eq!(stats.participant_count, 3);
        assert!(stats.gross_exposure > 0.0);
    }

    #[test]
    fn test_netting_stats_reduction() {
        let ne = setup_engine();
        let stats = ne.compute_stats(NettingMode::Multilateral, "USD");
        assert!(stats.reduction_pct() >= 0.0);
    }

    #[test]
    fn test_securities_obligation() {
        let mut ne = NettingEngine::new();
        ne.add_obligation(Obligation::securities(0, "A", "B", "AAPL", 100.0, 15000.0, "USD"));
        ne.add_obligation(Obligation::securities(0, "B", "A", "AAPL", 60.0, 9000.0, "USD"));
        let net = ne.bilateral_net("A", "B", "USD").unwrap();
        let aapl_net = net.security_nets.get("AAPL").copied().unwrap_or(0.0);
        assert!((aapl_net - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_net_position_payer_receiver() {
        let ne = setup_engine();
        let net = ne.bilateral_net("A", "B", "USD").unwrap();
        assert_eq!(net.payer(), "A");
        assert_eq!(net.receiver(), "B");
    }

    #[test]
    fn test_close_out_result_display() {
        let r = CloseOutResult {
            defaulter: "A".into(),
            counterparty: "B".into(),
            terminated_count: 5,
            net_closeout_amount: 1000.0,
            currency: "USD".into(),
        };
        let s = format!("{r}");
        assert!(s.contains("CloseOut"));
    }

    #[test]
    fn test_engine_display() {
        let ne = setup_engine();
        let s = format!("{ne}");
        assert!(s.contains("NettingEngine"));
    }

    #[test]
    fn test_netting_mode_display() {
        assert_eq!(format!("{}", NettingMode::Bilateral), "Bilateral");
        assert_eq!(format!("{}", NettingMode::CloseOut), "CloseOut");
    }

    #[test]
    fn test_zero_amount_bilateral() {
        let mut ne = NettingEngine::new();
        ne.add_obligation(Obligation::cash(0, "A", "B", "USD", 500.0));
        ne.add_obligation(Obligation::cash(0, "B", "A", "USD", 500.0));
        let net = ne.bilateral_net("A", "B", "USD").unwrap();
        assert!(net.net_cash.abs() < 1e-9);
    }
}
