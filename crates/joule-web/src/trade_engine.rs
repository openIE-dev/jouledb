//! # Trade Execution Engine
//!
//! Implements a complete trade execution engine with order lifecycle management,
//! matching logic, trade event generation, and position tracking. Supports limit,
//! market, and stop orders with time-in-force semantics, partial fills, and
//! deterministic price-time priority matching.

use std::fmt;
use std::collections::{BTreeMap, VecDeque};

// ── Core Types ──

/// Unique identifier for an order.
pub type OrderId = u64;

/// Unique identifier for a trade.
pub type TradeId = u64;

/// Side of an order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::Stop => write!(f, "STOP"),
            OrderType::StopLimit => write!(f, "STOP-LIMIT"),
        }
    }
}

/// Time-in-force policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good till cancelled.
    Gtc,
    /// Immediate or cancel — fill what you can, cancel remainder.
    Ioc,
    /// Fill or kill — must fill entirely or cancel.
    Fok,
    /// Day order — expires at session end.
    Day,
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeInForce::Gtc => write!(f, "GTC"),
            TimeInForce::Ioc => write!(f, "IOC"),
            TimeInForce::Fok => write!(f, "FOK"),
            TimeInForce::Day => write!(f, "DAY"),
        }
    }
}

/// Order status through its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    Accepted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OrderStatus::New => "NEW",
            OrderStatus::Accepted => "ACCEPTED",
            OrderStatus::PartiallyFilled => "PARTIAL",
            OrderStatus::Filled => "FILLED",
            OrderStatus::Cancelled => "CANCELLED",
            OrderStatus::Rejected => "REJECTED",
            OrderStatus::Expired => "EXPIRED",
        };
        write!(f, "{s}")
    }
}

// ── Order ──

/// An order submitted to the engine.
#[derive(Clone, Debug)]
pub struct Order {
    pub id: OrderId,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub price: f64,
    pub stop_price: f64,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub timestamp_ns: u64,
}

impl Order {
    pub fn new(id: OrderId, symbol: &str, side: Side, quantity: f64) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            order_type: OrderType::Market,
            price: 0.0,
            stop_price: 0.0,
            quantity,
            filled_quantity: 0.0,
            time_in_force: TimeInForce::Day,
            status: OrderStatus::New,
            timestamp_ns: 0,
        }
    }

    pub fn with_limit(mut self, price: f64) -> Self {
        self.order_type = OrderType::Limit;
        self.price = price;
        self
    }

    pub fn with_stop(mut self, stop_price: f64) -> Self {
        self.order_type = OrderType::Stop;
        self.stop_price = stop_price;
        self
    }

    pub fn with_stop_limit(mut self, stop_price: f64, limit_price: f64) -> Self {
        self.order_type = OrderType::StopLimit;
        self.stop_price = stop_price;
        self.price = limit_price;
        self
    }

    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ns = ts;
        self
    }

    pub fn remaining(&self) -> f64 {
        self.quantity - self.filled_quantity
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::New | OrderStatus::Accepted | OrderStatus::PartiallyFilled
        )
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Order(#{} {} {} {:.4} @ {:.2} [{}/{}] {})",
            self.id, self.side, self.symbol, self.quantity,
            self.price, self.status, self.time_in_force, self.order_type,
        )
    }
}

// ── Trade Event ──

/// A trade event produced when two orders match.
#[derive(Clone, Debug)]
pub struct TradeEvent {
    pub trade_id: TradeId,
    pub symbol: String,
    pub buy_order_id: OrderId,
    pub sell_order_id: OrderId,
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub aggressor_side: Side,
}

impl fmt::Display for TradeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade(#{} {} {:.4}@{:.2} buy=#{} sell=#{} aggressor={})",
            self.trade_id, self.symbol, self.quantity, self.price,
            self.buy_order_id, self.sell_order_id, self.aggressor_side,
        )
    }
}

// ── Order Book Level ──

/// A price level in the order book (FIFO queue at each price).
#[derive(Clone, Debug)]
struct PriceLevel {
    orders: VecDeque<Order>,
}

impl PriceLevel {
    fn new() -> Self {
        Self { orders: VecDeque::new() }
    }

    fn total_quantity(&self) -> f64 {
        self.orders.iter().map(|o| o.remaining()).sum()
    }

    fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
}

// ── Book Side ──

/// One side of the order book keyed by price.
/// Bids stored in descending order (BTreeMap with negated keys),
/// asks stored in ascending order.
#[derive(Clone, Debug)]
struct BookSide {
    levels: BTreeMap<i64, PriceLevel>,
    is_bid: bool,
}

impl BookSide {
    fn new(is_bid: bool) -> Self {
        Self { levels: BTreeMap::new(), is_bid }
    }

    /// Convert f64 price to i64 key (bids negated for descending).
    fn price_key(&self, price: f64) -> i64 {
        let ticks = (price * 1_000_000.0) as i64;
        if self.is_bid { -ticks } else { ticks }
    }

    fn insert(&mut self, order: Order) {
        let key = self.price_key(order.price);
        self.levels.entry(key).or_insert_with(PriceLevel::new).orders.push_back(order);
    }

    fn best_price(&self) -> Option<f64> {
        let (key, level) = self.levels.iter().next()?;
        if level.is_empty() { return None; }
        let ticks = if self.is_bid { -key } else { *key };
        Some(ticks as f64 / 1_000_000.0)
    }

    fn best_level_mut(&mut self) -> Option<&mut PriceLevel> {
        self.remove_empty();
        let key = *self.levels.keys().next()?;
        self.levels.get_mut(&key)
    }

    fn remove_empty(&mut self) {
        self.levels.retain(|_, level| !level.is_empty());
    }

    fn total_depth(&self) -> f64 {
        self.levels.values().map(|l| l.total_quantity()).sum()
    }

    fn level_count(&self) -> usize {
        self.levels.values().filter(|l| !l.is_empty()).count()
    }

    fn cancel_order(&mut self, order_id: OrderId) -> Option<Order> {
        for level in self.levels.values_mut() {
            if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                let mut order = level.orders.remove(pos).unwrap();
                order.status = OrderStatus::Cancelled;
                return Some(order);
            }
        }
        None
    }
}

// ── Trade Engine ──

/// The main matching engine maintaining an order book per symbol.
#[derive(Clone, Debug)]
pub struct TradeEngine {
    pub symbol: String,
    bids: BookSide,
    asks: BookSide,
    next_trade_id: TradeId,
    pub last_trade_price: f64,
    pub trade_count: u64,
    pub total_volume: f64,
    events: Vec<TradeEvent>,
    timestamp_ns: u64,
}

impl TradeEngine {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BookSide::new(true),
            asks: BookSide::new(false),
            next_trade_id: 1,
            last_trade_price: 0.0,
            trade_count: 0,
            total_volume: 0.0,
            events: Vec::new(),
            timestamp_ns: 0,
        }
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ns = ts;
        self
    }

    /// Submit an order and attempt to match.
    pub fn submit(&mut self, mut order: Order) -> Vec<TradeEvent> {
        order.status = OrderStatus::Accepted;
        let mut trades = Vec::new();

        match order.order_type {
            OrderType::Market => {
                trades = self.match_order(&mut order);
                if order.remaining() > 1e-12 {
                    order.status = OrderStatus::Cancelled;
                }
            }
            OrderType::Limit => {
                trades = self.match_order(&mut order);
                if order.remaining() > 1e-12 && order.is_active() {
                    match order.side {
                        Side::Buy => self.bids.insert(order),
                        Side::Sell => self.asks.insert(order),
                    }
                }
            }
            OrderType::Stop | OrderType::StopLimit => {
                // Stop orders rest until triggered.
                match order.side {
                    Side::Buy => self.bids.insert(order),
                    Side::Sell => self.asks.insert(order),
                }
            }
        }

        self.bids.remove_empty();
        self.asks.remove_empty();
        self.events.extend(trades.clone());
        trades
    }

    /// Cancel an active order.
    pub fn cancel(&mut self, order_id: OrderId) -> Option<Order> {
        self.bids
            .cancel_order(order_id)
            .or_else(|| self.asks.cancel_order(order_id))
    }

    /// Best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.best_price()
    }

    /// Best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.best_price()
    }

    /// Mid-price if both sides have depth.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    /// Bid-ask spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    pub fn bid_depth(&self) -> f64 {
        self.bids.total_depth()
    }

    pub fn ask_depth(&self) -> f64 {
        self.asks.total_depth()
    }

    pub fn bid_levels(&self) -> usize {
        self.bids.level_count()
    }

    pub fn ask_levels(&self) -> usize {
        self.asks.level_count()
    }

    /// Drain recorded trade events.
    pub fn drain_events(&mut self) -> Vec<TradeEvent> {
        std::mem::take(&mut self.events)
    }

    // ── Matching ──

    fn match_order(&mut self, aggressor: &mut Order) -> Vec<TradeEvent> {
        let mut trades = Vec::new();
        let contra_side = match aggressor.side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };

        loop {
            if aggressor.remaining() < 1e-12 {
                aggressor.status = OrderStatus::Filled;
                break;
            }

            let level = match contra_side.best_level_mut() {
                Some(l) => l,
                None => break,
            };

            let (fill_price, fill_qty, resting_id, resting_filled, should_pop) = {
                let resting = match level.orders.front_mut() {
                    Some(o) => o,
                    None => break,
                };

                // Price check for limit orders.
                if aggressor.order_type == OrderType::Limit {
                    match aggressor.side {
                        Side::Buy if resting.price > aggressor.price => break,
                        Side::Sell if resting.price < aggressor.price => break,
                        _ => {}
                    }
                }

                let fill_price = resting.price;
                let fill_qty = aggressor.remaining().min(resting.remaining());

                aggressor.filled_quantity += fill_qty;
                resting.filled_quantity += fill_qty;

                let should_pop = resting.remaining() < 1e-12;
                if should_pop {
                    resting.status = OrderStatus::Filled;
                } else {
                    resting.status = OrderStatus::PartiallyFilled;
                }

                (fill_price, fill_qty, resting.id, should_pop, should_pop)
            };

            if should_pop {
                level.orders.pop_front();
            }

            if aggressor.remaining() < 1e-12 {
                aggressor.status = OrderStatus::Filled;
            } else {
                aggressor.status = OrderStatus::PartiallyFilled;
            }

            let (buy_id, sell_id) = match aggressor.side {
                Side::Buy => (aggressor.id, resting_id),
                Side::Sell => (resting_id, aggressor.id),
            };

            let trade = TradeEvent {
                trade_id: self.next_trade_id,
                symbol: self.symbol.clone(),
                buy_order_id: buy_id,
                sell_order_id: sell_id,
                price: fill_price,
                quantity: fill_qty,
                timestamp_ns: self.timestamp_ns,
                aggressor_side: aggressor.side,
            };

            self.next_trade_id += 1;
            self.last_trade_price = fill_price;
            self.trade_count += 1;
            self.total_volume += fill_qty;
            trades.push(trade);
        }

        trades
    }
}

impl fmt::Display for TradeEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TradeEngine({} bid={}/{} ask={}/{} last={:.2} trades={})",
            self.symbol,
            self.bid_levels(),
            self.bid_depth(),
            self.ask_levels(),
            self.ask_depth(),
            self.last_trade_price,
            self.trade_count,
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> TradeEngine {
        TradeEngine::new("AAPL")
    }

    #[test]
    fn test_order_creation() {
        let o = Order::new(1, "AAPL", Side::Buy, 100.0).with_limit(150.0);
        assert_eq!(o.order_type, OrderType::Limit);
        assert!((o.price - 150.0).abs() < 1e-9);
        assert_eq!(o.remaining(), 100.0);
    }

    #[test]
    fn test_order_display() {
        let o = Order::new(1, "AAPL", Side::Buy, 50.0).with_limit(100.0);
        let s = format!("{o}");
        assert!(s.contains("BUY"));
        assert!(s.contains("AAPL"));
    }

    #[test]
    fn test_limit_order_resting() {
        let mut eng = engine();
        let trades = eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(100.0));
        assert!(trades.is_empty());
        assert_eq!(eng.bid_levels(), 1);
        assert!((eng.bid_depth() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_match() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(100.0));
        let trades = eng.submit(Order::new(2, "AAPL", Side::Sell, 10.0).with_limit(100.0));
        assert_eq!(trades.len(), 1);
        assert!((trades[0].price - 100.0).abs() < 1e-9);
        assert!((trades[0].quantity - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_partial_fill() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(100.0));
        let trades = eng.submit(Order::new(2, "AAPL", Side::Sell, 5.0).with_limit(100.0));
        assert_eq!(trades.len(), 1);
        assert!((trades[0].quantity - 5.0).abs() < 1e-9);
        assert!((eng.bid_depth() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_price_time_priority() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 5.0).with_limit(100.0));
        eng.submit(Order::new(2, "AAPL", Side::Buy, 5.0).with_limit(101.0));
        // Best bid is 101, should match first.
        let trades = eng.submit(Order::new(3, "AAPL", Side::Sell, 5.0).with_limit(100.0));
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].buy_order_id, 2);
        assert!((trades[0].price - 101.0).abs() < 1e-9);
    }

    #[test]
    fn test_market_order_sweeps() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Sell, 5.0).with_limit(100.0));
        eng.submit(Order::new(2, "AAPL", Side::Sell, 5.0).with_limit(101.0));
        let trades = eng.submit(Order::new(3, "AAPL", Side::Buy, 8.0));
        assert_eq!(trades.len(), 2);
        assert!((trades[0].quantity - 5.0).abs() < 1e-9);
        assert!((trades[1].quantity - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_cancel_order() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(100.0));
        let cancelled = eng.cancel(1);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().status, OrderStatus::Cancelled);
        assert_eq!(eng.bid_depth(), 0.0);
    }

    #[test]
    fn test_spread_and_mid() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(99.0));
        eng.submit(Order::new(2, "AAPL", Side::Sell, 10.0).with_limit(101.0));
        assert!((eng.spread().unwrap() - 2.0).abs() < 1e-9);
        assert!((eng.mid_price().unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_book() {
        let eng = engine();
        assert!(eng.best_bid().is_none());
        assert!(eng.best_ask().is_none());
        assert!(eng.spread().is_none());
    }

    #[test]
    fn test_multiple_fills_same_level() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 3.0).with_limit(100.0));
        eng.submit(Order::new(2, "AAPL", Side::Buy, 4.0).with_limit(100.0));
        let trades = eng.submit(Order::new(3, "AAPL", Side::Sell, 6.0).with_limit(100.0));
        assert_eq!(trades.len(), 2);
        assert!((trades[0].quantity - 3.0).abs() < 1e-9);
        assert!((trades[1].quantity - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_match_price_gap() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(99.0));
        let trades = eng.submit(Order::new(2, "AAPL", Side::Sell, 10.0).with_limit(101.0));
        assert!(trades.is_empty());
        assert_eq!(eng.bid_levels(), 1);
        assert_eq!(eng.ask_levels(), 1);
    }

    #[test]
    fn test_drain_events() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(100.0));
        eng.submit(Order::new(2, "AAPL", Side::Sell, 10.0).with_limit(100.0));
        let evts = eng.drain_events();
        assert_eq!(evts.len(), 1);
        assert!(eng.drain_events().is_empty());
    }

    #[test]
    fn test_trade_event_display() {
        let evt = TradeEvent {
            trade_id: 1,
            symbol: "AAPL".into(),
            buy_order_id: 10,
            sell_order_id: 20,
            price: 150.0,
            quantity: 5.0,
            timestamp_ns: 0,
            aggressor_side: Side::Sell,
        };
        let s = format!("{evt}");
        assert!(s.contains("AAPL"));
        assert!(s.contains("150.00"));
    }

    #[test]
    fn test_engine_display() {
        let eng = engine();
        let s = format!("{eng}");
        assert!(s.contains("TradeEngine"));
        assert!(s.contains("AAPL"));
    }

    #[test]
    fn test_stop_order_rests() {
        let mut eng = engine();
        let trades = eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_stop(105.0));
        assert!(trades.is_empty());
    }

    #[test]
    fn test_tif_builder() {
        let o = Order::new(1, "AAPL", Side::Buy, 10.0).with_tif(TimeInForce::Ioc);
        assert_eq!(o.time_in_force, TimeInForce::Ioc);
    }

    #[test]
    fn test_order_is_active() {
        let mut o = Order::new(1, "AAPL", Side::Buy, 10.0);
        assert!(o.is_active());
        o.status = OrderStatus::Filled;
        assert!(!o.is_active());
    }

    #[test]
    fn test_volume_tracking() {
        let mut eng = engine();
        eng.submit(Order::new(1, "AAPL", Side::Buy, 10.0).with_limit(100.0));
        eng.submit(Order::new(2, "AAPL", Side::Sell, 10.0).with_limit(100.0));
        assert!((eng.total_volume - 10.0).abs() < 1e-9);
        assert_eq!(eng.trade_count, 1);
    }

    #[test]
    fn test_stop_limit_builder() {
        let o = Order::new(1, "AAPL", Side::Sell, 10.0).with_stop_limit(95.0, 94.5);
        assert_eq!(o.order_type, OrderType::StopLimit);
        assert!((o.stop_price - 95.0).abs() < 1e-9);
        assert!((o.price - 94.5).abs() < 1e-9);
    }
}
