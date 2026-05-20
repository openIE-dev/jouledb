//! Limit Order Book — bid/ask sides, price-time priority, best bid/ask
//! tracking, spread calculation, and full book snapshot support.
//!
//! Pure-Rust limit order book with separate bid and ask half-books,
//! price-time FIFO ordering, O(1) best-bid/ask access, and volume
//! aggregation at each price level.

use std::collections::BTreeMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum OrderBookError {
    InvalidPrice(String),
    InvalidQuantity(String),
    OrderNotFound(u64),
    EmptyBook(String),
    DuplicateOrder(u64),
}

impl fmt::Display for OrderBookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
            Self::InvalidQuantity(s) => write!(f, "invalid quantity: {s}"),
            Self::OrderNotFound(id) => write!(f, "order not found: {id}"),
            Self::EmptyBook(s) => write!(f, "empty book: {s}"),
            Self::DuplicateOrder(id) => write!(f, "duplicate order id: {id}"),
        }
    }
}

impl std::error::Error for OrderBookError {}

// ── Side ────────────────────────────────────────────────────────

/// Which side of the book an order belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bid => write!(f, "BID"),
            Self::Ask => write!(f, "ASK"),
        }
    }
}

// ── BookEntry ───────────────────────────────────────────────────

/// A single resting order on the book.
#[derive(Debug, Clone, PartialEq)]
pub struct BookEntry {
    pub order_id: u64,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    pub remaining: f64,
    pub timestamp_ns: u64,
}

impl BookEntry {
    pub fn new(order_id: u64, side: Side, price: f64, quantity: f64, timestamp_ns: u64) -> Self {
        Self { order_id, side, price, quantity, remaining: quantity, timestamp_ns }
    }

    /// Fraction already filled.
    pub fn fill_ratio(&self) -> f64 {
        if self.quantity <= 0.0 { return 0.0; }
        1.0 - self.remaining / self.quantity
    }

    /// Whether the order is fully filled.
    pub fn is_filled(&self) -> bool {
        self.remaining <= 1e-12
    }
}

impl fmt::Display for BookEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entry(id={}, {} {:.4}@{:.2}, rem={:.4})",
            self.order_id, self.side, self.quantity, self.price, self.remaining)
    }
}

// ── PriceLevel ──────────────────────────────────────────────────

/// All orders resting at a single price, in time-priority order.
#[derive(Debug, Clone)]
struct PriceLevel {
    orders: Vec<BookEntry>,
}

impl PriceLevel {
    fn new() -> Self { Self { orders: Vec::new() } }

    fn total_volume(&self) -> f64 {
        self.orders.iter().map(|o| o.remaining).sum()
    }

    fn order_count(&self) -> usize { self.orders.len() }

    fn add(&mut self, entry: BookEntry) {
        self.orders.push(entry);
    }

    fn remove(&mut self, order_id: u64) -> Option<BookEntry> {
        if let Some(pos) = self.orders.iter().position(|o| o.order_id == order_id) {
            Some(self.orders.remove(pos))
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool { self.orders.is_empty() }
}

// ── BookSnapshot ────────────────────────────────────────────────

/// A point-in-time snapshot of the order book.
#[derive(Debug, Clone)]
pub struct BookSnapshot {
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub spread: Option<f64>,
    pub spread_bps: Option<f64>,
    pub bid_depth: usize,
    pub ask_depth: usize,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
    pub bid_levels: Vec<(f64, f64, usize)>,
    pub ask_levels: Vec<(f64, f64, usize)>,
}

impl fmt::Display for BookSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bid_str = self.best_bid.map_or("--".into(), |v| format!("{v:.2}"));
        let ask_str = self.best_ask.map_or("--".into(), |v| format!("{v:.2}"));
        let spread_str = self.spread.map_or("--".into(), |v| format!("{v:.4}"));
        write!(f, "Book(bid={bid_str}, ask={ask_str}, spread={spread_str}, \
            bid_levels={}, ask_levels={})", self.bid_depth, self.ask_depth)
    }
}

// ── OrderBook ───────────────────────────────────────────────────

/// A limit order book with bid/ask sides and price-time priority.
#[derive(Debug, Clone)]
pub struct OrderBook {
    symbol: String,
    bids: BTreeMap<i64, PriceLevel>,
    asks: BTreeMap<i64, PriceLevel>,
    tick_size: f64,
    order_count: u64,
}

/// Convert f64 price to integer ticks for BTreeMap ordering.
fn price_to_ticks(price: f64, tick_size: f64) -> i64 {
    (price / tick_size).round() as i64
}

fn ticks_to_price(ticks: i64, tick_size: f64) -> f64 {
    ticks as f64 * tick_size
}

impl OrderBook {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            tick_size: 0.01,
            order_count: 0,
        }
    }

    pub fn with_tick_size(mut self, tick_size: f64) -> Self {
        self.tick_size = tick_size;
        self
    }

    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = symbol.to_string();
        self
    }

    /// Add a resting order to the book.
    pub fn add_order(&mut self, entry: BookEntry) -> Result<(), OrderBookError> {
        if entry.price <= 0.0 {
            return Err(OrderBookError::InvalidPrice(format!("{}", entry.price)));
        }
        if entry.remaining <= 0.0 {
            return Err(OrderBookError::InvalidQuantity(format!("{}", entry.remaining)));
        }
        let ticks = price_to_ticks(entry.price, self.tick_size);
        let book = match entry.side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        book.entry(ticks).or_insert_with(PriceLevel::new).add(entry);
        self.order_count += 1;
        Ok(())
    }

    /// Remove an order by id from a given side.
    pub fn remove_order(&mut self, order_id: u64, side: Side) -> Result<BookEntry, OrderBookError> {
        let book = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        let mut found_ticks = None;
        let mut found_entry = None;
        for (&ticks, level) in book.iter_mut() {
            if let Some(e) = level.remove(order_id) {
                found_ticks = Some(ticks);
                found_entry = Some(e);
                break;
            }
        }
        if let (Some(ticks), Some(entry)) = (found_ticks, found_entry) {
            if book.get(&ticks).map_or(false, |l| l.is_empty()) {
                book.remove(&ticks);
            }
            self.order_count -= 1;
            Ok(entry)
        } else {
            Err(OrderBookError::OrderNotFound(order_id))
        }
    }

    /// Best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|t| ticks_to_price(*t, self.tick_size))
    }

    /// Best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|t| ticks_to_price(*t, self.tick_size))
    }

    /// Bid-ask spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        }
    }

    /// Spread in basis points relative to midpoint.
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(a), Some(b)) => {
                let mid = (a + b) / 2.0;
                if mid.abs() < 1e-12 { None } else { Some((a - b) / mid * 10_000.0) }
            }
            _ => None,
        }
    }

    /// Midpoint price.
    pub fn midpoint(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(a), Some(b)) => Some((a + b) / 2.0),
            _ => None,
        }
    }

    /// Total volume on the bid side.
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.values().map(|l| l.total_volume()).sum()
    }

    /// Total volume on the ask side.
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.values().map(|l| l.total_volume()).sum()
    }

    /// Number of distinct price levels on each side.
    pub fn depth(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }

    /// Capture a full snapshot.
    pub fn snapshot(&self) -> BookSnapshot {
        let bid_levels: Vec<(f64, f64, usize)> = self.bids.iter().rev()
            .map(|(&t, l)| (ticks_to_price(t, self.tick_size), l.total_volume(), l.order_count()))
            .collect();
        let ask_levels: Vec<(f64, f64, usize)> = self.asks.iter()
            .map(|(&t, l)| (ticks_to_price(t, self.tick_size), l.total_volume(), l.order_count()))
            .collect();
        BookSnapshot {
            best_bid: self.best_bid(),
            best_ask: self.best_ask(),
            spread: self.spread(),
            spread_bps: self.spread_bps(),
            bid_depth: self.bids.len(),
            ask_depth: self.asks.len(),
            total_bid_volume: self.total_bid_volume(),
            total_ask_volume: self.total_ask_volume(),
            bid_levels,
            ask_levels,
        }
    }

    /// Volume-weighted average price across top N levels on a side.
    pub fn vwap(&self, side: Side, levels: usize) -> Option<f64> {
        let book = match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        let iter: Vec<_> = match side {
            Side::Bid => book.iter().rev().take(levels).collect(),
            Side::Ask => book.iter().take(levels).collect(),
        };
        let mut total_value = 0.0;
        let mut total_vol = 0.0;
        for (&t, level) in iter {
            let px = ticks_to_price(t, self.tick_size);
            let vol = level.total_volume();
            total_value += px * vol;
            total_vol += vol;
        }
        if total_vol > 1e-12 { Some(total_value / total_vol) } else { None }
    }

    /// Symbol accessor.
    pub fn symbol(&self) -> &str { &self.symbol }

    /// Total order count.
    pub fn order_count(&self) -> u64 { self.order_count }

    /// Whether the book is empty.
    pub fn is_empty(&self) -> bool { self.bids.is_empty() && self.asks.is_empty() }
}

impl fmt::Display for OrderBook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (bd, ad) = self.depth();
        write!(f, "OrderBook({}, bid_levels={bd}, ask_levels={ad}, orders={})",
            self.symbol, self.order_count)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bid(id: u64, price: f64, qty: f64) -> BookEntry {
        BookEntry::new(id, Side::Bid, price, qty, id * 1000)
    }

    fn make_ask(id: u64, price: f64, qty: f64) -> BookEntry {
        BookEntry::new(id, Side::Ask, price, qty, id * 1000)
    }

    #[test]
    fn test_empty_book() {
        let book = OrderBook::new("AAPL");
        assert!(book.is_empty());
        assert_eq!(book.best_bid(), None);
        assert_eq!(book.best_ask(), None);
        assert_eq!(book.spread(), None);
    }

    #[test]
    fn test_add_bid() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        assert_eq!(book.best_bid(), Some(150.00));
        assert_eq!(book.order_count(), 1);
    }

    #[test]
    fn test_add_ask() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_ask(1, 151.00, 50.0)).unwrap();
        assert_eq!(book.best_ask(), Some(151.00));
    }

    #[test]
    fn test_spread() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        book.add_order(make_ask(2, 150.10, 100.0)).unwrap();
        let spread = book.spread().unwrap();
        assert!((spread - 0.10).abs() < 1e-6);
    }

    #[test]
    fn test_spread_bps() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 100.00, 100.0)).unwrap();
        book.add_order(make_ask(2, 100.10, 100.0)).unwrap();
        let bps = book.spread_bps().unwrap();
        assert!((bps - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_midpoint() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 100.00, 50.0)).unwrap();
        book.add_order(make_ask(2, 102.00, 50.0)).unwrap();
        assert!((book.midpoint().unwrap() - 101.0).abs() < 1e-6);
    }

    #[test]
    fn test_best_bid_multiple_levels() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 149.00, 100.0)).unwrap();
        book.add_order(make_bid(2, 150.00, 100.0)).unwrap();
        book.add_order(make_bid(3, 148.00, 100.0)).unwrap();
        assert_eq!(book.best_bid(), Some(150.00));
    }

    #[test]
    fn test_best_ask_multiple_levels() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_ask(1, 153.00, 100.0)).unwrap();
        book.add_order(make_ask(2, 151.00, 100.0)).unwrap();
        book.add_order(make_ask(3, 152.00, 100.0)).unwrap();
        assert_eq!(book.best_ask(), Some(151.00));
    }

    #[test]
    fn test_remove_order() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        let removed = book.remove_order(1, Side::Bid).unwrap();
        assert_eq!(removed.order_id, 1);
        assert!(book.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut book = OrderBook::new("AAPL");
        assert!(book.remove_order(999, Side::Bid).is_err());
    }

    #[test]
    fn test_invalid_price() {
        let mut book = OrderBook::new("AAPL");
        let entry = BookEntry::new(1, Side::Bid, -5.0, 100.0, 1000);
        assert!(book.add_order(entry).is_err());
    }

    #[test]
    fn test_invalid_quantity() {
        let mut book = OrderBook::new("AAPL");
        let entry = BookEntry::new(1, Side::Bid, 100.0, 0.0, 1000);
        assert!(book.add_order(entry).is_err());
    }

    #[test]
    fn test_volume() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        book.add_order(make_bid(2, 150.00, 200.0)).unwrap();
        book.add_order(make_ask(3, 151.00, 50.0)).unwrap();
        assert!((book.total_bid_volume() - 300.0).abs() < 1e-6);
        assert!((book.total_ask_volume() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_depth() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        book.add_order(make_bid(2, 149.00, 100.0)).unwrap();
        book.add_order(make_ask(3, 151.00, 100.0)).unwrap();
        assert_eq!(book.depth(), (2, 1));
    }

    #[test]
    fn test_snapshot() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        book.add_order(make_ask(2, 151.00, 100.0)).unwrap();
        let snap = book.snapshot();
        assert_eq!(snap.bid_depth, 1);
        assert_eq!(snap.ask_depth, 1);
        assert_eq!(snap.bid_levels.len(), 1);
        assert_eq!(snap.ask_levels.len(), 1);
    }

    #[test]
    fn test_vwap() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        book.add_order(make_bid(2, 149.00, 100.0)).unwrap();
        let vwap = book.vwap(Side::Bid, 2).unwrap();
        assert!((vwap - 149.50).abs() < 1e-6);
    }

    #[test]
    fn test_tick_size() {
        let book = OrderBook::new("BTC").with_tick_size(0.50);
        assert_eq!(book.symbol(), "BTC");
        let s = format!("{book}");
        assert!(s.contains("BTC"));
    }

    #[test]
    fn test_entry_fill_ratio() {
        let mut e = make_bid(1, 100.0, 200.0);
        e.remaining = 50.0;
        assert!((e.fill_ratio() - 0.75).abs() < 1e-6);
        assert!(!e.is_filled());
    }

    #[test]
    fn test_entry_display() {
        let e = make_bid(42, 150.50, 100.0);
        let s = format!("{e}");
        assert!(s.contains("42"));
        assert!(s.contains("BID"));
    }

    #[test]
    fn test_snapshot_display() {
        let mut book = OrderBook::new("AAPL");
        book.add_order(make_bid(1, 150.00, 100.0)).unwrap();
        book.add_order(make_ask(2, 151.00, 100.0)).unwrap();
        let snap = book.snapshot();
        let s = format!("{snap}");
        assert!(s.contains("150.00"));
        assert!(s.contains("151.00"));
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Bid), "BID");
        assert_eq!(format!("{}", Side::Ask), "ASK");
    }
}
