//! Wash Trade Detection — self-trade detection (same beneficial owner),
//! pre-arranged trade patterns, timing-based detection, circular trading
//! paths, and WashTradeConfig builder.
//!
//! Pure-Rust wash-trade detector that analyses trade streams for
//! manipulative self-dealing patterns using ownership graphs, temporal
//! clustering, and circular-path detection.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum WashTradeError {
    InvalidTrade(String),
    OwnerNotFound(String),
    ConfigError(String),
}

impl fmt::Display for WashTradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTrade(s) => write!(f, "invalid trade: {s}"),
            Self::OwnerNotFound(s) => write!(f, "owner not found: {s}"),
            Self::ConfigError(s) => write!(f, "config error: {s}"),
        }
    }
}

impl std::error::Error for WashTradeError {}

// ── Trade side ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide { Buy, Sell }

impl fmt::Display for TradeSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Buy => write!(f, "BUY"), Self::Sell => write!(f, "SELL") }
    }
}

// ── Detection reason ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionReason {
    SameBeneficialOwner,
    PreArrangedPattern,
    TimingAnomaly,
    CircularPath,
}

impl fmt::Display for DetectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SameBeneficialOwner => write!(f, "SAME_BENEFICIAL_OWNER"),
            Self::PreArrangedPattern => write!(f, "PRE_ARRANGED"),
            Self::TimingAnomaly => write!(f, "TIMING_ANOMALY"),
            Self::CircularPath => write!(f, "CIRCULAR_PATH"),
        }
    }
}

// ── Trade record ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub trade_id: u64,
    pub buyer_account: String,
    pub seller_account: String,
    pub instrument: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ms: u64,
}

impl fmt::Display for TradeRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Trade({} {} {} {:.2}x{:.0} @{}ms)",
            self.trade_id, self.instrument,
            self.buyer_account, self.price, self.quantity, self.timestamp_ms,
        )
    }
}

// ── Wash trade alert ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WashTradeAlert {
    pub trade_ids: Vec<u64>,
    pub reason: DetectionReason,
    pub confidence: f64,
    pub instrument: String,
    pub accounts: Vec<String>,
    pub total_quantity: f64,
    pub description: String,
}

impl fmt::Display for WashTradeAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "WashAlert({} conf={:.2} trades={} {})",
            self.reason, self.confidence,
            self.trade_ids.len(), self.instrument,
        )
    }
}

// ── Ownership mapping ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OwnershipMap {
    /// account_id -> beneficial_owner_id
    account_to_owner: HashMap<String, String>,
}

impl OwnershipMap {
    pub fn new() -> Self { Self { account_to_owner: HashMap::new() } }

    pub fn register(&mut self, account_id: &str, owner_id: &str) {
        self.account_to_owner.insert(account_id.into(), owner_id.into());
    }

    pub fn owner_of(&self, account_id: &str) -> Option<&str> {
        self.account_to_owner.get(account_id).map(|s| s.as_str())
    }

    pub fn same_owner(&self, acct_a: &str, acct_b: &str) -> bool {
        match (self.owner_of(acct_a), self.owner_of(acct_b)) {
            (Some(a), Some(b)) => a == b,
            _ => acct_a == acct_b,
        }
    }

    pub fn len(&self) -> usize { self.account_to_owner.len() }
    pub fn is_empty(&self) -> bool { self.account_to_owner.is_empty() }
}

impl fmt::Display for OwnershipMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OwnershipMap(accounts={})", self.account_to_owner.len())
    }
}

// ── WashTradeConfig builder ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WashTradeConfig {
    pub timing_window_ms: u64,
    pub price_tolerance_pct: f64,
    pub quantity_tolerance_pct: f64,
    pub min_circular_depth: usize,
    pub max_circular_depth: usize,
    pub pre_arranged_window_ms: u64,
    pub min_confidence: f64,
}

impl Default for WashTradeConfig {
    fn default() -> Self {
        Self {
            timing_window_ms: 5000,
            price_tolerance_pct: 0.001,
            quantity_tolerance_pct: 0.05,
            min_circular_depth: 3,
            max_circular_depth: 6,
            pre_arranged_window_ms: 1000,
            min_confidence: 0.70,
        }
    }
}

impl WashTradeConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_timing_window_ms(mut self, ms: u64) -> Self { self.timing_window_ms = ms; self }
    pub fn with_price_tolerance_pct(mut self, p: f64) -> Self { self.price_tolerance_pct = p; self }
    pub fn with_quantity_tolerance_pct(mut self, p: f64) -> Self { self.quantity_tolerance_pct = p; self }
    pub fn with_min_circular_depth(mut self, d: usize) -> Self { self.min_circular_depth = d; self }
    pub fn with_max_circular_depth(mut self, d: usize) -> Self { self.max_circular_depth = d; self }
    pub fn with_pre_arranged_window_ms(mut self, ms: u64) -> Self { self.pre_arranged_window_ms = ms; self }
    pub fn with_min_confidence(mut self, c: f64) -> Self { self.min_confidence = c; self }
}

impl fmt::Display for WashTradeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "WashTradeConfig(window={}ms price_tol={:.4} qty_tol={:.4})",
            self.timing_window_ms, self.price_tolerance_pct, self.quantity_tolerance_pct,
        )
    }
}

// ── Self-trade detection ────────────────────────────────────────

pub fn detect_self_trades(
    trades: &[TradeRecord],
    ownership: &OwnershipMap,
) -> Vec<WashTradeAlert> {
    let mut alerts = Vec::new();
    for trade in trades {
        if ownership.same_owner(&trade.buyer_account, &trade.seller_account) {
            alerts.push(WashTradeAlert {
                trade_ids: vec![trade.trade_id],
                reason: DetectionReason::SameBeneficialOwner,
                confidence: 1.0,
                instrument: trade.instrument.clone(),
                accounts: vec![trade.buyer_account.clone(), trade.seller_account.clone()],
                total_quantity: trade.quantity,
                description: format!(
                    "self-trade: {} buys from {} on {}",
                    trade.buyer_account, trade.seller_account, trade.instrument,
                ),
            });
        }
    }
    alerts
}

// ── Pre-arranged trade detection ────────────────────────────────

pub fn detect_pre_arranged(
    trades: &[TradeRecord],
    config: &WashTradeConfig,
) -> Vec<WashTradeAlert> {
    let mut alerts = Vec::new();
    for i in 0..trades.len() {
        for j in (i + 1)..trades.len() {
            let (a, b) = (&trades[i], &trades[j]);
            if a.instrument != b.instrument { continue; }
            let time_diff = b.timestamp_ms.saturating_sub(a.timestamp_ms);
            if time_diff > config.pre_arranged_window_ms { continue; }
            // Matching price and quantity (within tolerance).
            let price_diff = (a.price - b.price).abs() / a.price.max(1e-12);
            let qty_diff = (a.quantity - b.quantity).abs() / a.quantity.max(1e-12);
            if price_diff <= config.price_tolerance_pct && qty_diff <= config.quantity_tolerance_pct {
                // Opposite sides (buyer of one is seller of other).
                if a.buyer_account == b.seller_account || a.seller_account == b.buyer_account {
                    let conf = 1.0 - (time_diff as f64 / config.pre_arranged_window_ms as f64).min(1.0);
                    if conf >= config.min_confidence {
                        alerts.push(WashTradeAlert {
                            trade_ids: vec![a.trade_id, b.trade_id],
                            reason: DetectionReason::PreArrangedPattern,
                            confidence: conf,
                            instrument: a.instrument.clone(),
                            accounts: vec![a.buyer_account.clone(), a.seller_account.clone(), b.buyer_account.clone(), b.seller_account.clone()],
                            total_quantity: a.quantity + b.quantity,
                            description: format!("pre-arranged trades {} and {}", a.trade_id, b.trade_id),
                        });
                    }
                }
            }
        }
    }
    alerts
}

// ── Timing-based detection ──────────────────────────────────────

pub fn detect_timing_anomalies(
    trades: &[TradeRecord],
    config: &WashTradeConfig,
) -> Vec<WashTradeAlert> {
    let mut by_account_pair: HashMap<(String, String), Vec<&TradeRecord>> = HashMap::new();
    for t in trades {
        let key = if t.buyer_account <= t.seller_account {
            (t.buyer_account.clone(), t.seller_account.clone())
        } else {
            (t.seller_account.clone(), t.buyer_account.clone())
        };
        by_account_pair.entry(key).or_default().push(t);
    }

    let mut alerts = Vec::new();
    for ((_a, _b), group) in &by_account_pair {
        if group.len() < 2 { continue; }
        let mut intervals = Vec::new();
        for i in 1..group.len() {
            let dt = group[i].timestamp_ms.saturating_sub(group[i - 1].timestamp_ms);
            intervals.push(dt);
        }
        // Check for suspiciously regular intervals.
        if intervals.len() >= 2 {
            let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
            let variance = intervals.iter()
                .map(|dt| { let d = *dt as f64 - mean; d * d })
                .sum::<f64>() / intervals.len() as f64;
            let cv = if mean > 0.0 { variance.sqrt() / mean } else { 0.0 };
            // Low coefficient of variation = suspiciously regular timing.
            if cv < 0.1 && mean < config.timing_window_ms as f64 {
                let ids: Vec<u64> = group.iter().map(|t| t.trade_id).collect();
                let total_qty: f64 = group.iter().map(|t| t.quantity).sum();
                alerts.push(WashTradeAlert {
                    trade_ids: ids,
                    reason: DetectionReason::TimingAnomaly,
                    confidence: 1.0 - cv,
                    instrument: group[0].instrument.clone(),
                    accounts: vec![group[0].buyer_account.clone(), group[0].seller_account.clone()],
                    total_quantity: total_qty,
                    description: format!("regular timing (cv={cv:.4}, mean={mean:.0}ms)"),
                });
            }
        }
    }
    alerts
}

// ── Circular trading path detection ─────────────────────────────

pub fn detect_circular_paths(
    trades: &[TradeRecord],
    config: &WashTradeConfig,
) -> Vec<WashTradeAlert> {
    // Build directed graph: buyer -> seller edges.
    let mut graph: HashMap<&str, Vec<(usize, &str)>> = HashMap::new();
    for (i, t) in trades.iter().enumerate() {
        graph.entry(t.buyer_account.as_str()).or_default().push((i, t.seller_account.as_str()));
    }

    let mut alerts = Vec::new();
    let accounts: HashSet<&str> = trades.iter()
        .flat_map(|t| [t.buyer_account.as_str(), t.seller_account.as_str()])
        .collect();

    for &start in &accounts {
        // BFS to find cycles back to start.
        let mut queue: VecDeque<(Vec<usize>, &str)> = VecDeque::new();
        if let Some(neighbors) = graph.get(start) {
            for &(idx, next) in neighbors {
                queue.push_back((vec![idx], next));
            }
        }
        while let Some((path, current)) = queue.pop_front() {
            if path.len() > config.max_circular_depth { continue; }
            if current == start && path.len() >= config.min_circular_depth {
                let ids: Vec<u64> = path.iter().map(|i| trades[*i].trade_id).collect();
                let total_qty: f64 = path.iter().map(|i| trades[*i].quantity).sum();
                alerts.push(WashTradeAlert {
                    trade_ids: ids,
                    reason: DetectionReason::CircularPath,
                    confidence: 0.85,
                    instrument: trades[path[0]].instrument.clone(),
                    accounts: vec![start.into()],
                    total_quantity: total_qty,
                    description: format!("circular path of length {}", path.len()),
                });
                continue;
            }
            if let Some(neighbors) = graph.get(current) {
                for &(idx, next) in neighbors {
                    if !path.contains(&idx) {
                        let mut new_path = path.clone();
                        new_path.push(idx);
                        queue.push_back((new_path, next));
                    }
                }
            }
        }
    }
    alerts
}

// ── Detector engine ─────────────────────────────────────────────

pub struct WashTradeDetector {
    config: WashTradeConfig,
    ownership: OwnershipMap,
}

impl WashTradeDetector {
    pub fn new(config: WashTradeConfig, ownership: OwnershipMap) -> Self {
        Self { config, ownership }
    }

    pub fn config(&self) -> &WashTradeConfig { &self.config }

    pub fn detect_all(&self, trades: &[TradeRecord]) -> Vec<WashTradeAlert> {
        let mut all = Vec::new();
        all.extend(detect_self_trades(trades, &self.ownership));
        all.extend(detect_pre_arranged(trades, &self.config));
        all.extend(detect_timing_anomalies(trades, &self.config));
        all.extend(detect_circular_paths(trades, &self.config));
        all
    }
}

impl fmt::Display for WashTradeDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WashTradeDetector({} {})", self.config, self.ownership)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(id: u64, buyer: &str, seller: &str, inst: &str, price: f64, qty: f64, ts: u64) -> TradeRecord {
        TradeRecord {
            trade_id: id, buyer_account: buyer.into(), seller_account: seller.into(),
            instrument: inst.into(), price, quantity: qty, timestamp_ms: ts,
        }
    }

    fn sample_ownership() -> OwnershipMap {
        let mut om = OwnershipMap::new();
        om.register("ACCT1", "OWNER_A");
        om.register("ACCT2", "OWNER_A");
        om.register("ACCT3", "OWNER_B");
        om
    }

    #[test]
    fn test_self_trade_same_owner() {
        let om = sample_ownership();
        let trades = vec![make_trade(1, "ACCT1", "ACCT2", "AAPL", 150.0, 100.0, 1000)];
        let alerts = detect_self_trades(&trades, &om);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].reason, DetectionReason::SameBeneficialOwner);
    }

    #[test]
    fn test_no_self_trade_different_owner() {
        let om = sample_ownership();
        let trades = vec![make_trade(1, "ACCT1", "ACCT3", "AAPL", 150.0, 100.0, 1000)];
        let alerts = detect_self_trades(&trades, &om);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_pre_arranged_detected() {
        let cfg = WashTradeConfig::new().with_min_confidence(0.5);
        let trades = vec![
            make_trade(1, "A", "B", "AAPL", 150.0, 100.0, 1000),
            make_trade(2, "B", "A", "AAPL", 150.0, 100.0, 1100),
        ];
        let alerts = detect_pre_arranged(&trades, &cfg);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_pre_arranged_outside_window() {
        let cfg = WashTradeConfig::new().with_pre_arranged_window_ms(100);
        let trades = vec![
            make_trade(1, "A", "B", "AAPL", 150.0, 100.0, 1000),
            make_trade(2, "B", "A", "AAPL", 150.0, 100.0, 5000),
        ];
        let alerts = detect_pre_arranged(&trades, &cfg);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_timing_anomaly() {
        let cfg = WashTradeConfig::new().with_timing_window_ms(10000);
        let trades = vec![
            make_trade(1, "A", "B", "AAPL", 150.0, 100.0, 1000),
            make_trade(2, "A", "B", "AAPL", 150.0, 100.0, 2000),
            make_trade(3, "A", "B", "AAPL", 150.0, 100.0, 3000),
            make_trade(4, "A", "B", "AAPL", 150.0, 100.0, 4000),
        ];
        let alerts = detect_timing_anomalies(&trades, &cfg);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].reason, DetectionReason::TimingAnomaly);
    }

    #[test]
    fn test_circular_path() {
        let cfg = WashTradeConfig::new()
            .with_min_circular_depth(3)
            .with_max_circular_depth(6);
        let trades = vec![
            make_trade(1, "A", "B", "AAPL", 150.0, 100.0, 1000),
            make_trade(2, "B", "C", "AAPL", 150.0, 100.0, 2000),
            make_trade(3, "C", "A", "AAPL", 150.0, 100.0, 3000),
        ];
        let alerts = detect_circular_paths(&trades, &cfg);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_no_circular_short_path() {
        let cfg = WashTradeConfig::new().with_min_circular_depth(4);
        let trades = vec![
            make_trade(1, "A", "B", "AAPL", 150.0, 100.0, 1000),
            make_trade(2, "B", "A", "AAPL", 150.0, 100.0, 2000),
        ];
        let alerts = detect_circular_paths(&trades, &cfg);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_ownership_map_same_owner() {
        let om = sample_ownership();
        assert!(om.same_owner("ACCT1", "ACCT2"));
        assert!(!om.same_owner("ACCT1", "ACCT3"));
    }

    #[test]
    fn test_ownership_map_unknown_account() {
        let om = sample_ownership();
        assert!(!om.same_owner("ACCT1", "UNKNOWN"));
    }

    #[test]
    fn test_config_builder() {
        let cfg = WashTradeConfig::new()
            .with_timing_window_ms(10000)
            .with_price_tolerance_pct(0.01);
        assert_eq!(cfg.timing_window_ms, 10000);
        assert!((cfg.price_tolerance_pct - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_detect_all() {
        let om = sample_ownership();
        let detector = WashTradeDetector::new(WashTradeConfig::new(), om);
        let trades = vec![make_trade(1, "ACCT1", "ACCT2", "AAPL", 150.0, 100.0, 1000)];
        let alerts = detector.detect_all(&trades);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_trade_display() {
        let t = make_trade(1, "A", "B", "AAPL", 150.0, 100.0, 1000);
        assert!(format!("{t}").contains("AAPL"));
    }

    #[test]
    fn test_alert_display() {
        let a = WashTradeAlert {
            trade_ids: vec![1], reason: DetectionReason::SameBeneficialOwner,
            confidence: 1.0, instrument: "AAPL".into(), accounts: vec!["A".into()],
            total_quantity: 100.0, description: "test".into(),
        };
        assert!(format!("{a}").contains("SAME_BENEFICIAL_OWNER"));
    }

    #[test]
    fn test_detector_display() {
        let detector = WashTradeDetector::new(WashTradeConfig::new(), OwnershipMap::new());
        assert!(format!("{detector}").contains("WashTradeDetector"));
    }

    #[test]
    fn test_ownership_len() {
        let om = sample_ownership();
        assert_eq!(om.len(), 3);
        assert!(!om.is_empty());
    }

    #[test]
    fn test_ownership_new_empty() {
        let om = OwnershipMap::new();
        assert!(om.is_empty());
    }

    #[test]
    fn test_config_display() {
        let cfg = WashTradeConfig::new();
        assert!(format!("{cfg}").contains("WashTradeConfig"));
    }

    #[test]
    fn test_detection_reason_display() {
        assert_eq!(format!("{}", DetectionReason::CircularPath), "CIRCULAR_PATH");
        assert_eq!(format!("{}", DetectionReason::TimingAnomaly), "TIMING_ANOMALY");
    }

    #[test]
    fn test_same_account_self_trade() {
        let om = OwnershipMap::new();
        let trades = vec![make_trade(1, "A", "A", "AAPL", 150.0, 100.0, 1000)];
        let alerts = detect_self_trades(&trades, &om);
        assert_eq!(alerts.len(), 1);
    }
}
