//! # Crossing Network Engine
//!
//! Implements a periodic crossing network that accumulates orders and
//! matches them in discrete batch auctions. Supports multiple price
//! determination methods (midpoint, reference, volume-maximising),
//! pro-rata and time-priority allocation, and crossing session lifecycle
//! management.

use std::fmt;
use std::collections::VecDeque;

// ── Core Types ──

/// Side of a crossing order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CrossSide {
    Buy,
    Sell,
}

impl fmt::Display for CrossSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrossSide::Buy => write!(f, "BUY"),
            CrossSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Price determination method for the crossing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriceMethod {
    /// Use the midpoint of external NBBO.
    Midpoint,
    /// Use a fixed reference price.
    Reference,
    /// Maximise matched volume (uncross).
    VolumeMaximising,
    /// Use the last trade price.
    LastTrade,
}

impl fmt::Display for PriceMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PriceMethod::Midpoint => "MIDPOINT",
            PriceMethod::Reference => "REFERENCE",
            PriceMethod::VolumeMaximising => "VOL_MAX",
            PriceMethod::LastTrade => "LAST_TRADE",
        };
        write!(f, "{s}")
    }
}

/// Allocation method when demand exceeds supply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationMethod {
    /// First-come first-served.
    TimePriority,
    /// Pro-rata based on order size.
    ProRata,
}

impl fmt::Display for AllocationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationMethod::TimePriority => write!(f, "TIME"),
            AllocationMethod::ProRata => write!(f, "PRO_RATA"),
        }
    }
}

/// Session state of the crossing engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    /// Accepting orders.
    Open,
    /// Locked — no new orders, awaiting cross.
    Locked,
    /// Cross is in progress.
    Crossing,
    /// Session complete.
    Closed,
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SessionState::Open => "OPEN",
            SessionState::Locked => "LOCKED",
            SessionState::Crossing => "CROSSING",
            SessionState::Closed => "CLOSED",
        };
        write!(f, "{s}")
    }
}

// ── Crossing Order ──

/// An order submitted to the crossing network.
#[derive(Clone, Debug)]
pub struct CrossOrder {
    pub id: u64,
    pub symbol: String,
    pub side: CrossSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub limit_price: f64,
    pub timestamp_ns: u64,
    pub participant_id: String,
    pub is_active: bool,
}

impl CrossOrder {
    pub fn new(id: u64, symbol: &str, side: CrossSide, quantity: f64) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            quantity,
            filled_quantity: 0.0,
            limit_price: 0.0,
            timestamp_ns: 0,
            participant_id: String::new(),
            is_active: true,
        }
    }

    pub fn with_limit(mut self, price: f64) -> Self {
        self.limit_price = price;
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ns = ts;
        self
    }

    pub fn with_participant(mut self, pid: &str) -> Self {
        self.participant_id = pid.to_string();
        self
    }

    pub fn remaining(&self) -> f64 {
        (self.quantity - self.filled_quantity).max(0.0)
    }

    /// Whether this order is eligible at a given cross price.
    pub fn eligible_at(&self, cross_price: f64) -> bool {
        if !self.is_active || self.remaining() < 1e-12 { return false; }
        if self.limit_price < 1e-12 { return true; } // Market-type.
        match self.side {
            CrossSide::Buy => cross_price <= self.limit_price,
            CrossSide::Sell => cross_price >= self.limit_price,
        }
    }
}

impl fmt::Display for CrossOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CrossOrder(#{} {} {} qty={:.2} rem={:.2} limit={:.2})",
            self.id, self.side, self.symbol, self.quantity,
            self.remaining(), self.limit_price,
        )
    }
}

// ── Cross Result ──

/// A single fill from a crossing session.
#[derive(Clone, Debug)]
pub struct CrossFill {
    pub order_id: u64,
    pub side: CrossSide,
    pub price: f64,
    pub quantity: f64,
    pub participant_id: String,
}

impl CrossFill {
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

impl fmt::Display for CrossFill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CrossFill(#{} {} {:.4}@{:.4})",
            self.order_id, self.side, self.quantity, self.price,
        )
    }
}

/// Result of a crossing session.
#[derive(Clone, Debug)]
pub struct CrossResult {
    pub session_id: u64,
    pub cross_price: f64,
    pub total_volume: f64,
    pub fills: Vec<CrossFill>,
    pub buy_surplus: f64,
    pub sell_surplus: f64,
    pub timestamp_ns: u64,
}

impl CrossResult {
    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }

    pub fn total_notional(&self) -> f64 {
        self.cross_price * self.total_volume
    }

    pub fn imbalance(&self) -> f64 {
        self.buy_surplus - self.sell_surplus
    }

    pub fn imbalance_ratio(&self) -> f64 {
        let total = self.buy_surplus + self.sell_surplus + self.total_volume * 2.0;
        if total > 1e-12 { self.imbalance() / total } else { 0.0 }
    }
}

impl fmt::Display for CrossResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CrossResult(session={} price={:.4} vol={:.2} fills={} imbal={:.2})",
            self.session_id, self.cross_price, self.total_volume,
            self.fills.len(), self.imbalance(),
        )
    }
}

// ── Crossing Engine ──

/// A periodic crossing network.
#[derive(Clone, Debug)]
pub struct CrossingEngine {
    pub symbol: String,
    pub price_method: PriceMethod,
    pub allocation: AllocationMethod,
    pub state: SessionState,
    pub reference_price: f64,
    buys: VecDeque<CrossOrder>,
    sells: VecDeque<CrossOrder>,
    next_session_id: u64,
    pub results: Vec<CrossResult>,
    pub total_crossed_volume: f64,
    pub session_count: u64,
}

impl CrossingEngine {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            price_method: PriceMethod::Midpoint,
            allocation: AllocationMethod::ProRata,
            state: SessionState::Open,
            reference_price: 0.0,
            buys: VecDeque::new(),
            sells: VecDeque::new(),
            next_session_id: 1,
            results: Vec::new(),
            total_crossed_volume: 0.0,
            session_count: 0,
        }
    }

    pub fn with_price_method(mut self, method: PriceMethod) -> Self {
        self.price_method = method;
        self
    }

    pub fn with_allocation(mut self, alloc: AllocationMethod) -> Self {
        self.allocation = alloc;
        self
    }

    pub fn with_reference_price(mut self, price: f64) -> Self {
        self.reference_price = price;
        self
    }

    /// Submit an order if the session is open.
    pub fn submit(&mut self, order: CrossOrder) -> bool {
        if self.state != SessionState::Open { return false; }
        match order.side {
            CrossSide::Buy => self.buys.push_back(order),
            CrossSide::Sell => self.sells.push_back(order),
        }
        true
    }

    /// Cancel an order.
    pub fn cancel(&mut self, order_id: u64) -> bool {
        if self.state != SessionState::Open { return false; }
        for o in self.buys.iter_mut().chain(self.sells.iter_mut()) {
            if o.id == order_id && o.is_active {
                o.is_active = false;
                return true;
            }
        }
        false
    }

    /// Lock the session — no more orders accepted.
    pub fn lock(&mut self) {
        if self.state == SessionState::Open {
            self.state = SessionState::Locked;
        }
    }

    /// Execute the cross.
    pub fn cross(&mut self, nbbo_bid: f64, nbbo_ask: f64, timestamp_ns: u64) -> Option<CrossResult> {
        if self.state != SessionState::Open && self.state != SessionState::Locked {
            return None;
        }
        self.state = SessionState::Crossing;

        let cross_price = self.determine_price(nbbo_bid, nbbo_ask);
        if cross_price < 1e-12 {
            self.state = SessionState::Closed;
            return None;
        }

        // Capture allocation method before mutable borrows.
        let allocation = self.allocation;

        // Gather eligible orders.
        let mut eligible_buys: Vec<&mut CrossOrder> = self.buys.iter_mut()
            .filter(|o| o.eligible_at(cross_price))
            .collect();
        let mut eligible_sells: Vec<&mut CrossOrder> = self.sells.iter_mut()
            .filter(|o| o.eligible_at(cross_price))
            .collect();

        let buy_qty: f64 = eligible_buys.iter().map(|o| o.remaining()).sum();
        let sell_qty: f64 = eligible_sells.iter().map(|o| o.remaining()).sum();
        let matched_volume = buy_qty.min(sell_qty);

        if matched_volume < 1e-12 {
            self.state = SessionState::Closed;
            return None;
        }

        let mut fills = Vec::new();

        // Allocate to buy side.
        Self::allocate_fills_static(allocation, &mut eligible_buys, matched_volume, cross_price, &mut fills);
        // Allocate to sell side.
        Self::allocate_fills_static(allocation, &mut eligible_sells, matched_volume, cross_price, &mut fills);

        let result = CrossResult {
            session_id: self.next_session_id,
            cross_price,
            total_volume: matched_volume,
            fills,
            buy_surplus: (buy_qty - matched_volume).max(0.0),
            sell_surplus: (sell_qty - matched_volume).max(0.0),
            timestamp_ns,
        };

        self.next_session_id += 1;
        self.total_crossed_volume += matched_volume;
        self.session_count += 1;
        self.state = SessionState::Closed;
        self.results.push(result.clone());

        Some(result)
    }

    fn determine_price(&self, nbbo_bid: f64, nbbo_ask: f64) -> f64 {
        match self.price_method {
            PriceMethod::Midpoint => {
                if nbbo_bid > 0.0 && nbbo_ask > nbbo_bid {
                    (nbbo_bid + nbbo_ask) / 2.0
                } else {
                    0.0
                }
            }
            PriceMethod::Reference => self.reference_price,
            PriceMethod::VolumeMaximising => {
                // Simplified: use midpoint as proxy.
                if nbbo_bid > 0.0 && nbbo_ask > nbbo_bid {
                    (nbbo_bid + nbbo_ask) / 2.0
                } else {
                    self.reference_price
                }
            }
            PriceMethod::LastTrade => self.reference_price,
        }
    }

    fn allocate_fills_static(
        allocation: AllocationMethod,
        orders: &mut [&mut CrossOrder],
        total: f64,
        price: f64,
        fills: &mut Vec<CrossFill>,
    ) {
        let total_eligible: f64 = orders.iter().map(|o| o.remaining()).sum();
        if total_eligible < 1e-12 { return; }

        let mut remaining = total;

        match allocation {
            AllocationMethod::TimePriority => {
                for order in orders.iter_mut() {
                    if remaining < 1e-12 { break; }
                    let alloc = order.remaining().min(remaining);
                    order.filled_quantity += alloc;
                    remaining -= alloc;
                    fills.push(CrossFill {
                        order_id: order.id,
                        side: order.side,
                        price,
                        quantity: alloc,
                        participant_id: order.participant_id.clone(),
                    });
                }
            }
            AllocationMethod::ProRata => {
                for order in orders.iter_mut() {
                    let share = order.remaining() / total_eligible;
                    let alloc = (total * share).min(order.remaining()).min(remaining);
                    if alloc < 1e-12 { continue; }
                    order.filled_quantity += alloc;
                    remaining -= alloc;
                    fills.push(CrossFill {
                        order_id: order.id,
                        side: order.side,
                        price,
                        quantity: alloc,
                        participant_id: order.participant_id.clone(),
                    });
                }
            }
        }
    }

    /// Reset engine for a new session.
    pub fn reset(&mut self) {
        self.buys.clear();
        self.sells.clear();
        self.state = SessionState::Open;
    }

    pub fn buy_interest(&self) -> f64 {
        self.buys.iter().filter(|o| o.is_active).map(|o| o.remaining()).sum()
    }

    pub fn sell_interest(&self) -> f64 {
        self.sells.iter().filter(|o| o.is_active).map(|o| o.remaining()).sum()
    }

    pub fn order_count(&self) -> usize {
        self.buys.iter().chain(self.sells.iter()).filter(|o| o.is_active).count()
    }

    /// Pre-cross indication of matching potential.
    pub fn indicative_volume(&self, price: f64) -> f64 {
        let buy_qty: f64 = self.buys.iter()
            .filter(|o| o.eligible_at(price))
            .map(|o| o.remaining())
            .sum();
        let sell_qty: f64 = self.sells.iter()
            .filter(|o| o.eligible_at(price))
            .map(|o| o.remaining())
            .sum();
        buy_qty.min(sell_qty)
    }
}

impl fmt::Display for CrossingEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CrossingEngine({} state={} buys={:.0} sells={:.0} method={} sessions={})",
            self.symbol, self.state, self.buy_interest(), self.sell_interest(),
            self.price_method, self.session_count,
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> CrossingEngine {
        CrossingEngine::new("AAPL")
    }

    #[test]
    fn test_submit_order() {
        let mut e = eng();
        assert!(e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0)));
        assert_eq!(e.order_count(), 1);
        assert!((e.buy_interest() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_cross() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 100.0));
        let result = e.cross(149.0, 151.0, 0).unwrap();
        assert!((result.cross_price - 150.0).abs() < 1e-9);
        assert!((result.total_volume - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalanced_cross() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 200.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 100.0));
        let result = e.cross(149.0, 151.0, 0).unwrap();
        assert!((result.total_volume - 100.0).abs() < 1e-9);
        assert!((result.buy_surplus - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_pro_rata_allocation() {
        let mut e = eng().with_allocation(AllocationMethod::ProRata);
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Buy, 300.0));
        e.submit(CrossOrder::new(3, "AAPL", CrossSide::Sell, 200.0));
        let result = e.cross(149.0, 151.0, 0).unwrap();
        assert!((result.total_volume - 200.0).abs() < 1e-9);
        // Buy fills should be pro-rata: 50 and 150.
        let buy_fills: Vec<&CrossFill> = result.fills.iter()
            .filter(|f| f.side == CrossSide::Buy)
            .collect();
        assert_eq!(buy_fills.len(), 2);
    }

    #[test]
    fn test_time_priority_allocation() {
        let mut e = eng().with_allocation(AllocationMethod::TimePriority);
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0).with_timestamp(1));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Buy, 100.0).with_timestamp(2));
        e.submit(CrossOrder::new(3, "AAPL", CrossSide::Sell, 150.0));
        let result = e.cross(149.0, 151.0, 0).unwrap();
        let buy_fills: Vec<&CrossFill> = result.fills.iter()
            .filter(|f| f.side == CrossSide::Buy)
            .collect();
        assert!((buy_fills[0].quantity - 100.0).abs() < 1e-9); // First order filled fully.
    }

    #[test]
    fn test_reference_price() {
        let mut e = eng()
            .with_price_method(PriceMethod::Reference)
            .with_reference_price(155.0);
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 100.0));
        let result = e.cross(0.0, 0.0, 0).unwrap();
        assert!((result.cross_price - 155.0).abs() < 1e-9);
    }

    #[test]
    fn test_limit_order_filtering() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0).with_limit(149.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 100.0));
        // Midpoint is 150, but buyer limit is 149 → not eligible.
        let result = e.cross(149.0, 151.0, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_cancel() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        assert!(e.cancel(1));
        assert_eq!(e.order_count(), 0);
    }

    #[test]
    fn test_locked_rejects_submit() {
        let mut e = eng();
        e.lock();
        assert!(!e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0)));
    }

    #[test]
    fn test_indicative_volume() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 200.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 150.0));
        assert!((e.indicative_volume(150.0) - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_reset() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        e.cross(149.0, 151.0, 0);
        e.reset();
        assert_eq!(e.state, SessionState::Open);
        assert_eq!(e.order_count(), 0);
    }

    #[test]
    fn test_session_count() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 100.0));
        e.cross(149.0, 151.0, 0);
        assert_eq!(e.session_count, 1);
    }

    #[test]
    fn test_cross_result_display() {
        let r = CrossResult {
            session_id: 1, cross_price: 150.0, total_volume: 100.0,
            fills: Vec::new(), buy_surplus: 10.0, sell_surplus: 0.0, timestamp_ns: 0,
        };
        let s = format!("{r}");
        assert!(s.contains("CrossResult"));
    }

    #[test]
    fn test_cross_fill_notional() {
        let f = CrossFill {
            order_id: 1, side: CrossSide::Buy, price: 150.0,
            quantity: 100.0, participant_id: String::new(),
        };
        assert!((f.notional() - 15000.0).abs() < 1e-9);
    }

    #[test]
    fn test_engine_display() {
        let e = eng();
        let s = format!("{e}");
        assert!(s.contains("CrossingEngine"));
        assert!(s.contains("AAPL"));
    }

    #[test]
    fn test_imbalance_ratio() {
        let r = CrossResult {
            session_id: 1, cross_price: 100.0, total_volume: 100.0,
            fills: Vec::new(), buy_surplus: 50.0, sell_surplus: 0.0, timestamp_ns: 0,
        };
        assert!(r.imbalance_ratio() > 0.0);
    }

    #[test]
    fn test_no_cross_without_interest() {
        let mut e = eng();
        let result = e.cross(149.0, 151.0, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_order_eligible_market() {
        let o = CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0);
        assert!(o.eligible_at(200.0)); // No limit → always eligible.
    }

    #[test]
    fn test_total_crossed_volume() {
        let mut e = eng();
        e.submit(CrossOrder::new(1, "AAPL", CrossSide::Buy, 100.0));
        e.submit(CrossOrder::new(2, "AAPL", CrossSide::Sell, 100.0));
        e.cross(149.0, 151.0, 0);
        assert!((e.total_crossed_volume - 100.0).abs() < 1e-9);
    }
}
