//! Trade Surveillance — pattern detection (layering, spoofing, wash trades),
//! alert generation, alert scoring/prioritisation, investigation workflow,
//! and case management.
//!
//! Pure-Rust surveillance engine that analyses order/trade streams for
//! manipulative patterns and drives an investigation lifecycle from alert
//! through case closure.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SurveillanceError {
    InvalidOrder(String),
    DuplicateAlert(u64),
    CaseNotFound(u64),
    InvalidTransition(String),
}

impl fmt::Display for SurveillanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOrder(s) => write!(f, "invalid order: {s}"),
            Self::DuplicateAlert(id) => write!(f, "duplicate alert: {id}"),
            Self::CaseNotFound(id) => write!(f, "case not found: {id}"),
            Self::InvalidTransition(s) => write!(f, "invalid transition: {s}"),
        }
    }
}

impl std::error::Error for SurveillanceError {}

// ── Side / Order type ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side { Buy, Sell }

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Buy => write!(f, "BUY"), Self::Sell => write!(f, "SELL") }
    }
}

// ── Pattern type ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    Layering,
    Spoofing,
    WashTrade,
    FrontRunning,
    Ramping,
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layering => write!(f, "LAYERING"),
            Self::Spoofing => write!(f, "SPOOFING"),
            Self::WashTrade => write!(f, "WASH_TRADE"),
            Self::FrontRunning => write!(f, "FRONT_RUNNING"),
            Self::Ramping => write!(f, "RAMPING"),
        }
    }
}

// ── Alert severity ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity { Low, Medium, High, Critical }

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ── Case status ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseStatus {
    Open,
    UnderInvestigation,
    Escalated,
    Closed,
    FalsePositive,
}

impl fmt::Display for CaseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "OPEN"),
            Self::UnderInvestigation => write!(f, "UNDER_INVESTIGATION"),
            Self::Escalated => write!(f, "ESCALATED"),
            Self::Closed => write!(f, "CLOSED"),
            Self::FalsePositive => write!(f, "FALSE_POSITIVE"),
        }
    }
}

// ── Order ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    pub account_id: String,
    pub instrument: String,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ms: u64,
    pub cancelled: bool,
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Order({} {} {} {:.2}x{:.0} @{}ms{})",
            self.order_id, self.instrument, self.side,
            self.price, self.quantity, self.timestamp_ms,
            if self.cancelled { " CANCELLED" } else { "" },
        )
    }
}

// ── Alert ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_id: u64,
    pub pattern: PatternType,
    pub severity: Severity,
    pub score: f64,
    pub instrument: String,
    pub account_id: String,
    pub order_ids: Vec<u64>,
    pub description: String,
    pub timestamp_ms: u64,
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Alert({} {} {} score={:.2} {})",
            self.alert_id, self.pattern, self.severity,
            self.score, self.instrument,
        )
    }
}

// ── Case ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Case {
    pub case_id: u64,
    pub status: CaseStatus,
    pub alert_ids: Vec<u64>,
    pub assignee: String,
    pub notes: Vec<String>,
    pub created_ms: u64,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Case({} {} alerts={} assignee={})",
            self.case_id, self.status, self.alert_ids.len(), self.assignee,
        )
    }
}

// ── Pattern detectors ───────────────────────────────────────────

/// Detect layering: many same-side orders at different price levels,
/// large fraction cancelled shortly after placement.
pub fn detect_layering(orders: &[Order], cancel_ratio_threshold: f64, min_levels: usize) -> Vec<Vec<usize>> {
    let mut by_instrument: HashMap<&str, Vec<(usize, &Order)>> = HashMap::new();
    for (i, o) in orders.iter().enumerate() {
        by_instrument.entry(&o.instrument).or_default().push((i, o));
    }
    let mut groups = Vec::new();
    for (_inst, ords) in &by_instrument {
        for side in [Side::Buy, Side::Sell] {
            let side_ords: Vec<_> = ords.iter().filter(|(_, o)| o.side == side).collect();
            if side_ords.len() < min_levels { continue; }
            let mut prices = std::collections::HashSet::new();
            let mut cancelled_count = 0usize;
            let mut indices = Vec::new();
            for &(idx, o) in &side_ords {
                prices.insert(o.price.to_bits());
                if o.cancelled { cancelled_count += 1; }
                indices.push(*idx);
            }
            if prices.len() >= min_levels {
                let ratio = cancelled_count as f64 / side_ords.len() as f64;
                if ratio >= cancel_ratio_threshold {
                    groups.push(indices);
                }
            }
        }
    }
    groups
}

/// Detect spoofing: large order placed then cancelled before fill.
pub fn detect_spoofing(orders: &[Order], size_threshold: f64) -> Vec<usize> {
    orders.iter().enumerate()
        .filter(|(_, o)| o.cancelled && o.quantity >= size_threshold)
        .map(|(i, _)| i)
        .collect()
}

/// Detect wash trades: same account on both sides within a time window.
pub fn detect_wash_trades(orders: &[Order], window_ms: u64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..orders.len() {
        for j in (i + 1)..orders.len() {
            let (a, b) = (&orders[i], &orders[j]);
            if a.account_id == b.account_id
                && a.instrument == b.instrument
                && a.side != b.side
                && b.timestamp_ms.saturating_sub(a.timestamp_ms) <= window_ms
            {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

// ── Alert scoring ───────────────────────────────────────────────

pub fn score_alert(pattern: PatternType, order_count: usize, total_quantity: f64) -> f64 {
    let base = match pattern {
        PatternType::Layering => 40.0,
        PatternType::Spoofing => 50.0,
        PatternType::WashTrade => 60.0,
        PatternType::FrontRunning => 70.0,
        PatternType::Ramping => 45.0,
    };
    let qty_factor = (total_quantity.ln().max(1.0)) * 2.0;
    let count_factor = (order_count as f64).sqrt() * 5.0;
    (base + qty_factor + count_factor).min(100.0)
}

pub fn severity_from_score(score: f64) -> Severity {
    if score >= 80.0 { Severity::Critical }
    else if score >= 60.0 { Severity::High }
    else if score >= 40.0 { Severity::Medium }
    else { Severity::Low }
}

// ── Surveillance engine ─────────────────────────────────────────

pub struct SurveillanceEngine {
    next_alert_id: u64,
    next_case_id: u64,
    alerts: Vec<Alert>,
    cases: Vec<Case>,
}

impl SurveillanceEngine {
    pub fn new() -> Self {
        Self { next_alert_id: 1, next_case_id: 1, alerts: Vec::new(), cases: Vec::new() }
    }

    pub fn alerts(&self) -> &[Alert] { &self.alerts }
    pub fn cases(&self) -> &[Case] { &self.cases }

    pub fn generate_alert(
        &mut self, pattern: PatternType, instrument: &str,
        account_id: &str, order_ids: Vec<u64>, description: &str,
        timestamp_ms: u64, total_quantity: f64,
    ) -> &Alert {
        let score = score_alert(pattern, order_ids.len(), total_quantity);
        let severity = severity_from_score(score);
        let alert = Alert {
            alert_id: self.next_alert_id,
            pattern, severity, score,
            instrument: instrument.into(),
            account_id: account_id.into(),
            order_ids, description: description.into(),
            timestamp_ms,
        };
        self.next_alert_id += 1;
        self.alerts.push(alert);
        self.alerts.last().unwrap()
    }

    pub fn create_case(&mut self, alert_ids: Vec<u64>, assignee: &str, created_ms: u64) -> &Case {
        let case = Case {
            case_id: self.next_case_id,
            status: CaseStatus::Open,
            alert_ids,
            assignee: assignee.into(),
            notes: Vec::new(),
            created_ms,
        };
        self.next_case_id += 1;
        self.cases.push(case);
        self.cases.last().unwrap()
    }

    pub fn transition_case(&mut self, case_id: u64, new_status: CaseStatus) -> Result<(), SurveillanceError> {
        let case = self.cases.iter_mut().find(|c| c.case_id == case_id)
            .ok_or(SurveillanceError::CaseNotFound(case_id))?;
        case.status = new_status;
        Ok(())
    }

    pub fn add_case_note(&mut self, case_id: u64, note: &str) -> Result<(), SurveillanceError> {
        let case = self.cases.iter_mut().find(|c| c.case_id == case_id)
            .ok_or(SurveillanceError::CaseNotFound(case_id))?;
        case.notes.push(note.into());
        Ok(())
    }

    /// Prioritised alerts sorted by score descending.
    pub fn prioritised_alerts(&self) -> Vec<&Alert> {
        let mut sorted: Vec<&Alert> = self.alerts.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }
}

impl fmt::Display for SurveillanceEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurveillanceEngine(alerts={} cases={})", self.alerts.len(), self.cases.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn buy_order(id: u64, acct: &str, inst: &str, price: f64, qty: f64, ts: u64, cancelled: bool) -> Order {
        Order { order_id: id, account_id: acct.into(), instrument: inst.into(), side: Side::Buy, price, quantity: qty, timestamp_ms: ts, cancelled }
    }
    fn sell_order(id: u64, acct: &str, inst: &str, price: f64, qty: f64, ts: u64, cancelled: bool) -> Order {
        Order { order_id: id, account_id: acct.into(), instrument: inst.into(), side: Side::Sell, price, quantity: qty, timestamp_ms: ts, cancelled }
    }

    #[test]
    fn test_detect_spoofing() {
        let orders = vec![
            buy_order(1, "A", "AAPL", 150.0, 10000.0, 100, true),
            buy_order(2, "A", "AAPL", 151.0, 50.0, 200, false),
        ];
        let spoof = detect_spoofing(&orders, 5000.0);
        assert_eq!(spoof, vec![0]);
    }

    #[test]
    fn test_detect_wash_trades() {
        let orders = vec![
            buy_order(1, "ACCT1", "AAPL", 150.0, 100.0, 1000, false),
            sell_order(2, "ACCT1", "AAPL", 150.0, 100.0, 1050, false),
        ];
        let pairs = detect_wash_trades(&orders, 100);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1));
    }

    #[test]
    fn test_no_wash_trade_different_account() {
        let orders = vec![
            buy_order(1, "A", "AAPL", 150.0, 100.0, 1000, false),
            sell_order(2, "B", "AAPL", 150.0, 100.0, 1050, false),
        ];
        let pairs = detect_wash_trades(&orders, 100);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_detect_layering() {
        let orders = vec![
            buy_order(1, "A", "AAPL", 149.0, 100.0, 100, true),
            buy_order(2, "A", "AAPL", 148.0, 100.0, 110, true),
            buy_order(3, "A", "AAPL", 147.0, 100.0, 120, true),
        ];
        let groups = detect_layering(&orders, 0.9, 3);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_no_layering_few_levels() {
        let orders = vec![
            buy_order(1, "A", "AAPL", 149.0, 100.0, 100, true),
            buy_order(2, "A", "AAPL", 149.0, 100.0, 110, true),
        ];
        let groups = detect_layering(&orders, 0.9, 3);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_score_alert() {
        let s = score_alert(PatternType::WashTrade, 2, 1000.0);
        assert!(s > 60.0);
    }

    #[test]
    fn test_severity_from_score() {
        assert_eq!(severity_from_score(90.0), Severity::Critical);
        assert_eq!(severity_from_score(65.0), Severity::High);
        assert_eq!(severity_from_score(45.0), Severity::Medium);
        assert_eq!(severity_from_score(20.0), Severity::Low);
    }

    #[test]
    fn test_generate_alert() {
        let mut engine = SurveillanceEngine::new();
        let alert = engine.generate_alert(
            PatternType::Spoofing, "AAPL", "ACCT1",
            vec![1, 2], "spoofing detected", 5000, 10000.0,
        );
        assert_eq!(alert.alert_id, 1);
        assert_eq!(alert.pattern, PatternType::Spoofing);
    }

    #[test]
    fn test_create_case() {
        let mut engine = SurveillanceEngine::new();
        engine.generate_alert(PatternType::Spoofing, "AAPL", "A", vec![1], "x", 100, 100.0);
        let case = engine.create_case(vec![1], "analyst1", 200);
        assert_eq!(case.case_id, 1);
        assert_eq!(case.status, CaseStatus::Open);
    }

    #[test]
    fn test_transition_case() {
        let mut engine = SurveillanceEngine::new();
        engine.create_case(vec![], "analyst1", 100);
        engine.transition_case(1, CaseStatus::UnderInvestigation).unwrap();
        assert_eq!(engine.cases()[0].status, CaseStatus::UnderInvestigation);
    }

    #[test]
    fn test_transition_case_not_found() {
        let mut engine = SurveillanceEngine::new();
        let err = engine.transition_case(99, CaseStatus::Closed).unwrap_err();
        assert_eq!(err, SurveillanceError::CaseNotFound(99));
    }

    #[test]
    fn test_add_case_note() {
        let mut engine = SurveillanceEngine::new();
        engine.create_case(vec![], "analyst1", 100);
        engine.add_case_note(1, "investigated").unwrap();
        assert_eq!(engine.cases()[0].notes.len(), 1);
    }

    #[test]
    fn test_prioritised_alerts() {
        let mut engine = SurveillanceEngine::new();
        engine.generate_alert(PatternType::Layering, "X", "A", vec![1], "low", 100, 10.0);
        engine.generate_alert(PatternType::WashTrade, "X", "A", vec![1, 2, 3, 4, 5], "high", 200, 50000.0);
        let pri = engine.prioritised_alerts();
        assert!(pri[0].score >= pri[1].score);
    }

    #[test]
    fn test_display_order() {
        let o = buy_order(1, "A", "AAPL", 150.0, 100.0, 1000, false);
        assert!(format!("{o}").contains("AAPL"));
    }

    #[test]
    fn test_display_alert() {
        let a = Alert {
            alert_id: 1, pattern: PatternType::Spoofing, severity: Severity::High,
            score: 75.0, instrument: "AAPL".into(), account_id: "A".into(),
            order_ids: vec![1], description: "test".into(), timestamp_ms: 100,
        };
        assert!(format!("{a}").contains("SPOOFING"));
    }

    #[test]
    fn test_display_case() {
        let c = Case {
            case_id: 1, status: CaseStatus::Open, alert_ids: vec![1],
            assignee: "bob".into(), notes: vec![], created_ms: 100,
        };
        assert!(format!("{c}").contains("OPEN"));
    }

    #[test]
    fn test_engine_display() {
        let engine = SurveillanceEngine::new();
        assert!(format!("{engine}").contains("alerts=0"));
    }

    #[test]
    fn test_wash_trade_outside_window() {
        let orders = vec![
            buy_order(1, "A", "AAPL", 150.0, 100.0, 1000, false),
            sell_order(2, "A", "AAPL", 150.0, 100.0, 5000, false),
        ];
        let pairs = detect_wash_trades(&orders, 100);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Buy), "BUY");
        assert_eq!(format!("{}", Side::Sell), "SELL");
    }
}
