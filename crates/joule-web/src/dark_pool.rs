//! # Dark Pool Matching
//!
//! Implements a dark pool venue with midpoint crossing, minimum quantity
//! thresholds, conditional indications of interest (IOI), information
//! leakage metrics, and anti-gaming protections. Orders rest invisibly
//! and match at the midpoint of the national best bid and offer (NBBO).

use std::fmt;
use std::collections::VecDeque;

// ── Core Types ──

/// Side of a dark pool order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DarkSide {
    Buy,
    Sell,
}

impl fmt::Display for DarkSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DarkSide::Buy => write!(f, "BUY"),
            DarkSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order condition type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderCondition {
    /// Unconditional — always eligible to match.
    Firm,
    /// Only match if minimum quantity met.
    MinQuantity,
    /// Conditional IOI — requires confirmation.
    Conditional,
}

impl fmt::Display for OrderCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OrderCondition::Firm => "FIRM",
            OrderCondition::MinQuantity => "MIN_QTY",
            OrderCondition::Conditional => "CONDITIONAL",
        };
        write!(f, "{s}")
    }
}

// ── NBBO Reference ──

/// National Best Bid and Offer reference prices.
#[derive(Clone, Copy, Debug)]
pub struct NbboRef {
    pub bid: f64,
    pub ask: f64,
    pub timestamp_ns: u64,
}

impl NbboRef {
    pub fn new(bid: f64, ask: f64, timestamp_ns: u64) -> Self {
        Self { bid, ask, timestamp_ns }
    }

    pub fn midpoint(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.midpoint();
        if mid > 1e-12 { (self.spread() / mid) * 10_000.0 } else { 0.0 }
    }

    pub fn is_valid(&self) -> bool {
        self.bid > 0.0 && self.ask > self.bid
    }
}

impl fmt::Display for NbboRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NBBO({:.4}/{:.4} mid={:.4} spread={:.1}bps)",
            self.bid, self.ask, self.midpoint(), self.spread_bps())
    }
}

// ── Dark Order ──

/// An order resting in the dark pool.
#[derive(Clone, Debug)]
pub struct DarkOrder {
    pub id: u64,
    pub symbol: String,
    pub side: DarkSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub min_quantity: f64,
    pub condition: OrderCondition,
    pub peg_to_midpoint: bool,
    pub limit_price: f64,
    pub timestamp_ns: u64,
    pub is_active: bool,
    pub participant_id: String,
}

impl DarkOrder {
    pub fn new(id: u64, symbol: &str, side: DarkSide, quantity: f64) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            quantity,
            filled_quantity: 0.0,
            min_quantity: 0.0,
            condition: OrderCondition::Firm,
            peg_to_midpoint: true,
            limit_price: 0.0,
            timestamp_ns: 0,
            is_active: true,
            participant_id: String::new(),
        }
    }

    pub fn with_min_quantity(mut self, min: f64) -> Self {
        self.min_quantity = min;
        self.condition = OrderCondition::MinQuantity;
        self
    }

    pub fn with_limit(mut self, price: f64) -> Self {
        self.limit_price = price;
        self.peg_to_midpoint = false;
        self
    }

    pub fn with_condition(mut self, cond: OrderCondition) -> Self {
        self.condition = cond;
        self
    }

    pub fn with_participant(mut self, pid: &str) -> Self {
        self.participant_id = pid.to_string();
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ns = ts;
        self
    }

    pub fn remaining(&self) -> f64 {
        (self.quantity - self.filled_quantity).max(0.0)
    }

    /// Whether this order can match a given contra quantity at the midpoint.
    pub fn can_match(&self, contra_qty: f64, midpoint: f64) -> bool {
        if !self.is_active { return false; }
        if self.remaining() < 1e-12 { return false; }

        // Check minimum quantity.
        let matchable = contra_qty.min(self.remaining());
        if matchable < self.min_quantity && self.min_quantity > 0.0 { return false; }

        // Check limit price constraint.
        if !self.peg_to_midpoint && self.limit_price > 0.0 {
            match self.side {
                DarkSide::Buy if midpoint > self.limit_price => return false,
                DarkSide::Sell if midpoint < self.limit_price => return false,
                _ => {}
            }
        }

        true
    }

    pub fn apply_fill(&mut self, qty: f64) {
        self.filled_quantity += qty;
        if self.remaining() < 1e-12 {
            self.is_active = false;
        }
    }
}

impl fmt::Display for DarkOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DarkOrder(#{} {} {} qty={:.2} rem={:.2} cond={} mid_peg={})",
            self.id, self.side, self.symbol, self.quantity,
            self.remaining(), self.condition, self.peg_to_midpoint,
        )
    }
}

// ── Dark Match ──

/// A match produced by the dark pool.
#[derive(Clone, Debug)]
pub struct DarkMatch {
    pub match_id: u64,
    pub symbol: String,
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub price_improvement_bps: f64,
}

impl DarkMatch {
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

impl fmt::Display for DarkMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DarkMatch(#{} {} {:.4}@{:.4} improve={:.1}bps)",
            self.match_id, self.symbol, self.quantity,
            self.price, self.price_improvement_bps,
        )
    }
}

// ── Leakage Metrics ──

/// Information leakage metrics for a participant.
#[derive(Clone, Debug)]
pub struct LeakageMetrics {
    pub participant_id: String,
    pub orders_submitted: u32,
    pub orders_filled: u32,
    pub orders_cancelled: u32,
    pub avg_resting_time_ns: u64,
    pub adverse_selection_bps: f64,
    pub mark_out_5s_bps: f64,
}

impl LeakageMetrics {
    pub fn new(pid: &str) -> Self {
        Self {
            participant_id: pid.to_string(),
            orders_submitted: 0,
            orders_filled: 0,
            orders_cancelled: 0,
            avg_resting_time_ns: 0,
            adverse_selection_bps: 0.0,
            mark_out_5s_bps: 0.0,
        }
    }

    pub fn fill_rate(&self) -> f64 {
        if self.orders_submitted > 0 {
            self.orders_filled as f64 / self.orders_submitted as f64
        } else {
            0.0
        }
    }

    pub fn cancel_rate(&self) -> f64 {
        if self.orders_submitted > 0 {
            self.orders_cancelled as f64 / self.orders_submitted as f64
        } else {
            0.0
        }
    }

    /// Toxicity score: higher means more informed/toxic flow.
    pub fn toxicity_score(&self) -> f64 {
        self.adverse_selection_bps.abs() + self.mark_out_5s_bps.abs()
    }
}

impl fmt::Display for LeakageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Leakage({} fill={:.1}% cancel={:.1}% toxicity={:.1}bps)",
            self.participant_id, self.fill_rate() * 100.0,
            self.cancel_rate() * 100.0, self.toxicity_score(),
        )
    }
}

// ── Dark Pool ──

/// The dark pool matching engine.
#[derive(Clone, Debug)]
pub struct DarkPool {
    pub symbol: String,
    buys: VecDeque<DarkOrder>,
    sells: VecDeque<DarkOrder>,
    nbbo: Option<NbboRef>,
    next_match_id: u64,
    pub matches: Vec<DarkMatch>,
    pub total_volume: f64,
    pub match_count: u64,
    pub min_order_size: f64,
    pub anti_gaming_enabled: bool,
}

impl DarkPool {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            buys: VecDeque::new(),
            sells: VecDeque::new(),
            nbbo: None,
            next_match_id: 1,
            matches: Vec::new(),
            total_volume: 0.0,
            match_count: 0,
            min_order_size: 0.0,
            anti_gaming_enabled: true,
        }
    }

    pub fn with_min_order_size(mut self, size: f64) -> Self {
        self.min_order_size = size;
        self
    }

    pub fn with_anti_gaming(mut self, enabled: bool) -> Self {
        self.anti_gaming_enabled = enabled;
        self
    }

    /// Update the NBBO reference.
    pub fn update_nbbo(&mut self, bid: f64, ask: f64, timestamp_ns: u64) {
        self.nbbo = Some(NbboRef::new(bid, ask, timestamp_ns));
    }

    /// Submit an order and attempt midpoint crossing.
    pub fn submit(&mut self, order: DarkOrder) -> Vec<DarkMatch> {
        if order.quantity < self.min_order_size {
            return Vec::new();
        }

        let mut incoming = order;
        let mut result = Vec::new();

        let midpoint = match &self.nbbo {
            Some(n) if n.is_valid() => n.midpoint(),
            _ => return { self.rest_order(incoming); Vec::new() },
        };

        let contra = match incoming.side {
            DarkSide::Buy => &mut self.sells,
            DarkSide::Sell => &mut self.buys,
        };

        let mut i = 0;
        while i < contra.len() && incoming.remaining() > 1e-12 {
            let resting = &contra[i];
            if !resting.is_active {
                i += 1;
                continue;
            }

            let matchable = incoming.remaining().min(resting.remaining());
            if !resting.can_match(matchable, midpoint) || !incoming.can_match(matchable, midpoint) {
                i += 1;
                continue;
            }

            // Anti-gaming: skip if same participant.
            if self.anti_gaming_enabled
                && !incoming.participant_id.is_empty()
                && incoming.participant_id == resting.participant_id
            {
                i += 1;
                continue;
            }

            let fill_qty = matchable;
            let (buy_id, sell_id) = match incoming.side {
                DarkSide::Buy => (incoming.id, resting.id),
                DarkSide::Sell => (resting.id, incoming.id),
            };

            // Calculate price improvement.
            let improvement = match &self.nbbo {
                Some(n) => {
                    let half_spread = n.spread() / 2.0;
                    if n.midpoint() > 1e-12 {
                        (half_spread / n.midpoint()) * 10_000.0
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };

            let dm = DarkMatch {
                match_id: self.next_match_id,
                symbol: self.symbol.clone(),
                buy_order_id: buy_id,
                sell_order_id: sell_id,
                price: midpoint,
                quantity: fill_qty,
                timestamp_ns: incoming.timestamp_ns,
                price_improvement_bps: improvement,
            };

            self.next_match_id += 1;
            self.total_volume += fill_qty;
            self.match_count += 1;

            incoming.apply_fill(fill_qty);
            contra[i].apply_fill(fill_qty);
            result.push(dm);
            i += 1;
        }

        // Remove fully filled contra orders.
        match incoming.side {
            DarkSide::Buy => self.sells.retain(|o| o.is_active),
            DarkSide::Sell => self.buys.retain(|o| o.is_active),
        };

        // Rest remainder.
        if incoming.remaining() > 1e-12 && incoming.is_active {
            self.rest_order(incoming);
        }

        self.matches.extend(result.clone());
        result
    }

    fn rest_order(&mut self, order: DarkOrder) {
        match order.side {
            DarkSide::Buy => self.buys.push_back(order),
            DarkSide::Sell => self.sells.push_back(order),
        }
    }

    /// Cancel an order.
    pub fn cancel(&mut self, order_id: u64) -> bool {
        for o in self.buys.iter_mut().chain(self.sells.iter_mut()) {
            if o.id == order_id && o.is_active {
                o.is_active = false;
                return true;
            }
        }
        false
    }

    pub fn buy_depth(&self) -> f64 {
        self.buys.iter().filter(|o| o.is_active).map(|o| o.remaining()).sum()
    }

    pub fn sell_depth(&self) -> f64 {
        self.sells.iter().filter(|o| o.is_active).map(|o| o.remaining()).sum()
    }

    pub fn active_order_count(&self) -> usize {
        self.buys.iter().chain(self.sells.iter()).filter(|o| o.is_active).count()
    }

    pub fn midpoint(&self) -> Option<f64> {
        self.nbbo.as_ref().filter(|n| n.is_valid()).map(|n| n.midpoint())
    }

    /// Average price improvement across all matches in basis points.
    pub fn avg_price_improvement_bps(&self) -> f64 {
        if self.matches.is_empty() { return 0.0; }
        let total: f64 = self.matches.iter().map(|m| m.price_improvement_bps * m.quantity).sum();
        let vol: f64 = self.matches.iter().map(|m| m.quantity).sum();
        if vol > 1e-12 { total / vol } else { 0.0 }
    }
}

impl fmt::Display for DarkPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DarkPool({} buys={:.0} sells={:.0} matches={} vol={:.0})",
            self.symbol, self.buy_depth(), self.sell_depth(),
            self.match_count, self.total_volume,
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> DarkPool {
        let mut p = DarkPool::new("AAPL");
        p.update_nbbo(149.0, 151.0, 0);
        p
    }

    #[test]
    fn test_nbbo_midpoint() {
        let n = NbboRef::new(99.0, 101.0, 0);
        assert!((n.midpoint() - 100.0).abs() < 1e-9);
        assert!((n.spread() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_nbbo_spread_bps() {
        let n = NbboRef::new(99.0, 101.0, 0);
        let expected = (2.0 / 100.0) * 10_000.0;
        assert!((n.spread_bps() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dark_order_creation() {
        let o = DarkOrder::new(1, "AAPL", DarkSide::Buy, 500.0);
        assert!((o.remaining() - 500.0).abs() < 1e-9);
        assert!(o.peg_to_midpoint);
    }

    #[test]
    fn test_order_min_quantity() {
        let o = DarkOrder::new(1, "AAPL", DarkSide::Buy, 500.0)
            .with_min_quantity(100.0);
        assert!(!o.can_match(50.0, 150.0)); // Below min.
        assert!(o.can_match(200.0, 150.0)); // Above min.
    }

    #[test]
    fn test_order_limit_constraint() {
        let o = DarkOrder::new(1, "AAPL", DarkSide::Buy, 500.0)
            .with_limit(150.0);
        assert!(o.can_match(100.0, 150.0));
        assert!(!o.can_match(100.0, 151.0)); // Above limit.
    }

    #[test]
    fn test_midpoint_cross() {
        let mut p = pool();
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 100.0));
        let matches = p.submit(DarkOrder::new(2, "AAPL", DarkSide::Sell, 100.0));
        assert_eq!(matches.len(), 1);
        assert!((matches[0].price - 150.0).abs() < 1e-9);
        assert!((matches[0].quantity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_partial_cross() {
        let mut p = pool();
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 200.0));
        let matches = p.submit(DarkOrder::new(2, "AAPL", DarkSide::Sell, 80.0));
        assert_eq!(matches.len(), 1);
        assert!((matches[0].quantity - 80.0).abs() < 1e-9);
        assert!((p.buy_depth() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_match_without_nbbo() {
        let mut p = DarkPool::new("AAPL");
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 100.0));
        let matches = p.submit(DarkOrder::new(2, "AAPL", DarkSide::Sell, 100.0));
        assert!(matches.is_empty());
    }

    #[test]
    fn test_anti_gaming_same_participant() {
        let mut p = pool();
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 100.0).with_participant("FIRM_A"));
        let matches = p.submit(
            DarkOrder::new(2, "AAPL", DarkSide::Sell, 100.0).with_participant("FIRM_A")
        );
        assert!(matches.is_empty());
    }

    #[test]
    fn test_cancel() {
        let mut p = pool();
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 100.0));
        assert!(p.cancel(1));
        assert_eq!(p.active_order_count(), 0);
    }

    #[test]
    fn test_min_order_size() {
        let mut p = pool().with_min_order_size(50.0);
        let matches = p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 10.0));
        assert!(matches.is_empty());
        assert_eq!(p.active_order_count(), 0);
    }

    #[test]
    fn test_price_improvement() {
        let mut p = pool(); // NBBO 149/151, mid 150, spread 2.
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 100.0));
        let matches = p.submit(DarkOrder::new(2, "AAPL", DarkSide::Sell, 100.0));
        assert!(matches[0].price_improvement_bps > 0.0);
    }

    #[test]
    fn test_leakage_metrics() {
        let mut lm = LeakageMetrics::new("FIRM_A");
        lm.orders_submitted = 100;
        lm.orders_filled = 40;
        lm.orders_cancelled = 30;
        lm.adverse_selection_bps = 5.0;
        lm.mark_out_5s_bps = 3.0;
        assert!((lm.fill_rate() - 0.40).abs() < 1e-9);
        assert!((lm.cancel_rate() - 0.30).abs() < 1e-9);
        assert!((lm.toxicity_score() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_pool_display() {
        let p = pool();
        let s = format!("{p}");
        assert!(s.contains("DarkPool"));
        assert!(s.contains("AAPL"));
    }

    #[test]
    fn test_dark_match_notional() {
        let dm = DarkMatch {
            match_id: 1, symbol: "AAPL".into(),
            buy_order_id: 1, sell_order_id: 2,
            price: 150.0, quantity: 100.0,
            timestamp_ns: 0, price_improvement_bps: 5.0,
        };
        assert!((dm.notional() - 15000.0).abs() < 1e-9);
    }

    #[test]
    fn test_nbbo_display() {
        let n = NbboRef::new(99.0, 101.0, 0);
        let s = format!("{n}");
        assert!(s.contains("NBBO"));
    }

    #[test]
    fn test_leakage_display() {
        let lm = LeakageMetrics::new("X");
        let s = format!("{lm}");
        assert!(s.contains("Leakage"));
    }

    #[test]
    fn test_avg_price_improvement() {
        let mut p = pool();
        p.submit(DarkOrder::new(1, "AAPL", DarkSide::Buy, 100.0));
        p.submit(DarkOrder::new(2, "AAPL", DarkSide::Sell, 100.0));
        assert!(p.avg_price_improvement_bps() > 0.0);
    }
}
