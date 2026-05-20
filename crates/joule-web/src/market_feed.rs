//! Market data feed — Level 1/2 data structures, quote/trade tick processing,
//! NBBO calculation, feed handler state machine, conflation (throttling),
//! stale data detection.
//!
//! Pure-Rust market data infrastructure for consuming and normalizing
//! real-time price feeds from multiple venues:
//!
//! - [`Quote`] / [`Trade`] — canonical Level 1 tick types
//! - [`Level2Entry`] / [`Level2Book`] — depth-of-book representation
//! - [`NbboCalculator`] — best bid/offer across venues
//! - [`FeedHandler`] — state machine managing feed lifecycle
//! - [`Conflator`] — throttle tick rates while preserving last-value semantics
//! - [`StaleDetector`] — flag symbols with no updates beyond a threshold

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FeedError {
    InvalidPrice(String),
    InvalidSize(String),
    StaleData(String),
    UnknownVenue(String),
    InvalidState(String),
}

impl fmt::Display for FeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
            Self::InvalidSize(s) => write!(f, "invalid size: {s}"),
            Self::StaleData(s) => write!(f, "stale data: {s}"),
            Self::UnknownVenue(s) => write!(f, "unknown venue: {s}"),
            Self::InvalidState(s) => write!(f, "invalid state: {s}"),
        }
    }
}

impl std::error::Error for FeedError {}

// ── Side ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bid => write!(f, "Bid"),
            Self::Ask => write!(f, "Ask"),
        }
    }
}

// ── Quote ───────────────────────────────────────────────────────

/// Level 1 quote (best bid/ask from a single venue).
#[derive(Debug, Clone, PartialEq)]
pub struct Quote {
    pub symbol: String,
    pub venue: String,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
    pub timestamp_us: u64,
    pub sequence: u64,
}

impl Quote {
    pub fn new(symbol: &str, venue: &str, bid_price: f64, bid_size: f64,
               ask_price: f64, ask_size: f64, timestamp_us: u64, sequence: u64) -> Self {
        Self {
            symbol: symbol.to_string(),
            venue: venue.to_string(),
            bid_price, bid_size, ask_price, ask_size, timestamp_us, sequence,
        }
    }

    pub fn spread(&self) -> f64 { self.ask_price - self.bid_price }

    pub fn mid_price(&self) -> f64 { (self.bid_price + self.ask_price) / 2.0 }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid_price();
        if mid == 0.0 { return 0.0; }
        (self.spread() / mid) * 10_000.0
    }

    pub fn is_crossed(&self) -> bool { self.bid_price > self.ask_price }

    pub fn is_locked(&self) -> bool { (self.bid_price - self.ask_price).abs() < 1e-12 }
}

impl fmt::Display for Quote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{} {:.4}/{:.4} [{:.0}x{:.0}]",
               self.symbol, self.venue, self.bid_price, self.ask_price,
               self.bid_size, self.ask_size)
    }
}

// ── Trade ───────────────────────────────────────────────────────

/// A single trade execution.
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
    pub symbol: String,
    pub venue: String,
    pub price: f64,
    pub size: f64,
    pub aggressor: Side,
    pub timestamp_us: u64,
    pub sequence: u64,
}

impl Trade {
    pub fn new(symbol: &str, venue: &str, price: f64, size: f64,
               aggressor: Side, timestamp_us: u64, sequence: u64) -> Self {
        Self {
            symbol: symbol.to_string(),
            venue: venue.to_string(),
            price, size, aggressor, timestamp_us, sequence,
        }
    }

    pub fn notional(&self) -> f64 { self.price * self.size }
}

impl fmt::Display for Trade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{} {:.4}x{:.0} {} t={}",
               self.symbol, self.venue, self.price, self.size,
               self.aggressor, self.timestamp_us)
    }
}

// ── Level 2 ─────────────────────────────────────────────────────

/// Single price level in the order book.
#[derive(Debug, Clone, PartialEq)]
pub struct Level2Entry {
    pub price: f64,
    pub size: f64,
    pub order_count: u32,
    pub venue: String,
}

impl Level2Entry {
    pub fn new(price: f64, size: f64, order_count: u32, venue: &str) -> Self {
        Self { price, size, order_count, venue: venue.to_string() }
    }
}

impl fmt::Display for Level2Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} x {:.0} ({} orders, {})", self.price, self.size,
               self.order_count, self.venue)
    }
}

/// Depth-of-book snapshot for a single symbol.
#[derive(Debug, Clone)]
pub struct Level2Book {
    pub symbol: String,
    pub bids: Vec<Level2Entry>,
    pub asks: Vec<Level2Entry>,
    pub timestamp_us: u64,
}

impl Level2Book {
    pub fn new(symbol: &str, timestamp_us: u64) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp_us,
        }
    }

    pub fn add_bid(&mut self, entry: Level2Entry) {
        let pos = self.bids.iter()
            .position(|b| b.price <= entry.price)
            .unwrap_or(self.bids.len());
        self.bids.insert(pos, entry);
    }

    pub fn add_ask(&mut self, entry: Level2Entry) {
        let pos = self.asks.iter()
            .position(|a| a.price >= entry.price)
            .unwrap_or(self.asks.len());
        self.asks.insert(pos, entry);
    }

    pub fn best_bid(&self) -> Option<&Level2Entry> { self.bids.first() }

    pub fn best_ask(&self) -> Option<&Level2Entry> { self.asks.first() }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a.price - b.price),
            _ => None,
        }
    }

    pub fn total_bid_depth(&self) -> f64 { self.bids.iter().map(|e| e.size).sum() }

    pub fn total_ask_depth(&self) -> f64 { self.asks.iter().map(|e| e.size).sum() }

    pub fn depth_levels(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }
}

impl fmt::Display for Level2Book {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L2[{}] bids={} asks={} ts={}",
               self.symbol, self.bids.len(), self.asks.len(), self.timestamp_us)
    }
}

// ── NBBO Calculator ─────────────────────────────────────────────

/// National Best Bid and Offer across venues.
#[derive(Debug, Clone, PartialEq)]
pub struct Nbbo {
    pub symbol: String,
    pub best_bid: f64,
    pub best_bid_size: f64,
    pub best_bid_venue: String,
    pub best_ask: f64,
    pub best_ask_size: f64,
    pub best_ask_venue: String,
    pub timestamp_us: u64,
}

impl Nbbo {
    pub fn spread(&self) -> f64 { self.best_ask - self.best_bid }

    pub fn mid_price(&self) -> f64 { (self.best_bid + self.best_ask) / 2.0 }

    pub fn is_crossed(&self) -> bool { self.best_bid > self.best_ask }
}

impl fmt::Display for Nbbo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NBBO[{}] {:.4}({})/{:.4}({})",
               self.symbol, self.best_bid, self.best_bid_venue,
               self.best_ask, self.best_ask_venue)
    }
}

/// Computes NBBO across multiple venue quotes.
pub struct NbboCalculator {
    /// symbol -> venue -> latest quote
    venue_quotes: HashMap<String, HashMap<String, Quote>>,
}

impl NbboCalculator {
    pub fn new() -> Self { Self { venue_quotes: HashMap::new() } }

    pub fn update(&mut self, quote: Quote) {
        self.venue_quotes
            .entry(quote.symbol.clone())
            .or_default()
            .insert(quote.venue.clone(), quote);
    }

    pub fn nbbo(&self, symbol: &str) -> Option<Nbbo> {
        let venues = self.venue_quotes.get(symbol)?;
        let mut best_bid = f64::MIN;
        let mut best_bid_size = 0.0;
        let mut best_bid_venue = String::new();
        let mut best_ask = f64::MAX;
        let mut best_ask_size = 0.0;
        let mut best_ask_venue = String::new();

        for q in venues.values() {
            if q.bid_price > best_bid {
                best_bid = q.bid_price;
                best_bid_size = q.bid_size;
                best_bid_venue = q.venue.clone();
            }
            if q.ask_price < best_ask {
                best_ask = q.ask_price;
                best_ask_size = q.ask_size;
                best_ask_venue = q.venue.clone();
            }
        }
        if best_bid == f64::MIN || best_ask == f64::MAX { return None; }

        Some(Nbbo {
            symbol: symbol.to_string(),
            best_bid, best_bid_size, best_bid_venue,
            best_ask, best_ask_size, best_ask_venue,
            timestamp_us: venues.values().map(|q| q.timestamp_us).max().unwrap_or(0),
        })
    }

    pub fn symbols(&self) -> Vec<String> {
        self.venue_quotes.keys().cloned().collect()
    }
}

impl fmt::Display for NbboCalculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NbboCalculator[{} symbols]", self.venue_quotes.len())
    }
}

// ── Feed Handler State Machine ──────────────────────────────────

/// Feed lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedState {
    Disconnected,
    Connecting,
    Connected,
    Subscribing,
    Streaming,
    Recovering,
    Error,
}

impl fmt::Display for FeedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disconnected => write!(f, "Disconnected"),
            Self::Connecting => write!(f, "Connecting"),
            Self::Connected => write!(f, "Connected"),
            Self::Subscribing => write!(f, "Subscribing"),
            Self::Streaming => write!(f, "Streaming"),
            Self::Recovering => write!(f, "Recovering"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Feed handler state machine with transition validation.
pub struct FeedHandler {
    state: FeedState,
    venue: String,
    subscriptions: Vec<String>,
    quotes_received: u64,
    trades_received: u64,
    errors: u64,
    last_activity_us: u64,
}

impl FeedHandler {
    pub fn new(venue: &str) -> Self {
        Self {
            state: FeedState::Disconnected,
            venue: venue.to_string(),
            subscriptions: Vec::new(),
            quotes_received: 0,
            trades_received: 0,
            errors: 0,
            last_activity_us: 0,
        }
    }

    pub fn state(&self) -> FeedState { self.state }
    pub fn venue(&self) -> &str { &self.venue }
    pub fn quotes_received(&self) -> u64 { self.quotes_received }
    pub fn trades_received(&self) -> u64 { self.trades_received }

    pub fn connect(&mut self) -> Result<(), FeedError> {
        match self.state {
            FeedState::Disconnected | FeedState::Error => {
                self.state = FeedState::Connecting;
                Ok(())
            }
            _ => Err(FeedError::InvalidState(
                format!("cannot connect from {}", self.state))),
        }
    }

    pub fn on_connected(&mut self) -> Result<(), FeedError> {
        if self.state != FeedState::Connecting {
            return Err(FeedError::InvalidState(
                format!("expected Connecting, got {}", self.state)));
        }
        self.state = FeedState::Connected;
        Ok(())
    }

    pub fn subscribe(&mut self, symbol: &str) -> Result<(), FeedError> {
        match self.state {
            FeedState::Connected | FeedState::Streaming => {
                if !self.subscriptions.contains(&symbol.to_string()) {
                    self.subscriptions.push(symbol.to_string());
                }
                self.state = FeedState::Subscribing;
                Ok(())
            }
            _ => Err(FeedError::InvalidState(
                format!("cannot subscribe in state {}", self.state))),
        }
    }

    pub fn on_subscribed(&mut self) -> Result<(), FeedError> {
        if self.state != FeedState::Subscribing {
            return Err(FeedError::InvalidState(
                format!("expected Subscribing, got {}", self.state)));
        }
        self.state = FeedState::Streaming;
        Ok(())
    }

    pub fn on_quote(&mut self, _quote: &Quote) {
        self.quotes_received += 1;
        self.last_activity_us = _quote.timestamp_us;
    }

    pub fn on_trade(&mut self, trade: &Trade) {
        self.trades_received += 1;
        self.last_activity_us = trade.timestamp_us;
    }

    pub fn on_error(&mut self) {
        self.errors += 1;
        self.state = FeedState::Error;
    }

    pub fn recover(&mut self) -> Result<(), FeedError> {
        if self.state != FeedState::Error {
            return Err(FeedError::InvalidState(
                format!("cannot recover from {}", self.state)));
        }
        self.state = FeedState::Recovering;
        Ok(())
    }

    pub fn on_recovered(&mut self) -> Result<(), FeedError> {
        if self.state != FeedState::Recovering {
            return Err(FeedError::InvalidState(
                format!("expected Recovering, got {}", self.state)));
        }
        self.state = FeedState::Streaming;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.state = FeedState::Disconnected;
        self.subscriptions.clear();
    }
}

impl fmt::Display for FeedHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeedHandler[{} state={} subs={} q={} t={}]",
               self.venue, self.state, self.subscriptions.len(),
               self.quotes_received, self.trades_received)
    }
}

// ── Conflator ───────────────────────────────────────────────────

/// Conflation (throttling) buffer — keeps last-value semantics per symbol.
pub struct Conflator {
    interval_us: u64,
    last_flush_us: HashMap<String, u64>,
    pending_quotes: HashMap<String, Quote>,
    pending_trades: HashMap<String, Trade>,
    total_conflated: u64,
    total_emitted: u64,
}

impl Conflator {
    pub fn new(interval_us: u64) -> Self {
        Self {
            interval_us,
            last_flush_us: HashMap::new(),
            pending_quotes: HashMap::new(),
            pending_trades: HashMap::new(),
            total_conflated: 0,
            total_emitted: 0,
        }
    }

    pub fn with_interval_us(mut self, interval_us: u64) -> Self {
        self.interval_us = interval_us;
        self
    }

    /// Buffer a quote. Returns `Some` if the conflation window has elapsed.
    pub fn on_quote(&mut self, quote: Quote) -> Option<Quote> {
        let sym = quote.symbol.clone();
        let ts = quote.timestamp_us;
        let first = !self.last_flush_us.contains_key(&sym);
        let last = self.last_flush_us.get(&sym).copied().unwrap_or(0);

        self.pending_quotes.insert(sym.clone(), quote);

        if first || ts.saturating_sub(last) >= self.interval_us {
            self.last_flush_us.insert(sym.clone(), ts);
            self.total_emitted += 1;
            self.pending_quotes.remove(&sym)
        } else {
            self.total_conflated += 1;
            None
        }
    }

    /// Buffer a trade. Returns `Some` if the conflation window has elapsed.
    pub fn on_trade(&mut self, trade: Trade) -> Option<Trade> {
        let sym = trade.symbol.clone();
        let ts = trade.timestamp_us;
        let first = !self.last_flush_us.contains_key(&sym);
        let last = self.last_flush_us.get(&sym).copied().unwrap_or(0);

        self.pending_trades.insert(sym.clone(), trade);

        if first || ts.saturating_sub(last) >= self.interval_us {
            self.last_flush_us.insert(sym.clone(), ts);
            self.total_emitted += 1;
            self.pending_trades.remove(&sym)
        } else {
            self.total_conflated += 1;
            None
        }
    }

    /// Flush all pending data regardless of time.
    pub fn flush(&mut self) -> (Vec<Quote>, Vec<Trade>) {
        let quotes: Vec<Quote> = self.pending_quotes.drain().map(|(_, v)| v).collect();
        let trades: Vec<Trade> = self.pending_trades.drain().map(|(_, v)| v).collect();
        self.total_emitted += (quotes.len() + trades.len()) as u64;
        (quotes, trades)
    }

    pub fn total_conflated(&self) -> u64 { self.total_conflated }
    pub fn total_emitted(&self) -> u64 { self.total_emitted }
}

impl fmt::Display for Conflator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Conflator[{}us conflated={} emitted={}]",
               self.interval_us, self.total_conflated, self.total_emitted)
    }
}

// ── Stale Data Detector ─────────────────────────────────────────

/// Detects symbols that have not received updates within a threshold.
pub struct StaleDetector {
    threshold_us: u64,
    last_update: HashMap<String, u64>,
}

impl StaleDetector {
    pub fn new(threshold_us: u64) -> Self {
        Self { threshold_us, last_update: HashMap::new() }
    }

    pub fn with_threshold_us(mut self, threshold_us: u64) -> Self {
        self.threshold_us = threshold_us;
        self
    }

    pub fn record_update(&mut self, symbol: &str, timestamp_us: u64) {
        self.last_update.insert(symbol.to_string(), timestamp_us);
    }

    pub fn is_stale(&self, symbol: &str, now_us: u64) -> bool {
        match self.last_update.get(symbol) {
            Some(&last) => now_us.saturating_sub(last) > self.threshold_us,
            None => true,
        }
    }

    pub fn stale_symbols(&self, now_us: u64) -> Vec<String> {
        self.last_update.iter()
            .filter(|&(_, &last)| now_us.saturating_sub(last) > self.threshold_us)
            .map(|(sym, _)| sym.clone())
            .collect()
    }

    pub fn staleness_us(&self, symbol: &str, now_us: u64) -> Option<u64> {
        self.last_update.get(symbol).map(|last| now_us.saturating_sub(*last))
    }

    pub fn tracked_count(&self) -> usize { self.last_update.len() }
}

impl fmt::Display for StaleDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StaleDetector[threshold={}us tracking={}]",
               self.threshold_us, self.last_update.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_spread() {
        let q = Quote::new("AAPL", "NYSE", 150.00, 100.0, 150.10, 200.0, 1000, 1);
        assert!((q.spread() - 0.10).abs() < 1e-9);
        assert!((q.mid_price() - 150.05).abs() < 1e-9);
    }

    #[test]
    fn quote_spread_bps() {
        let q = Quote::new("AAPL", "NYSE", 100.00, 100.0, 100.02, 100.0, 1000, 1);
        let bps = q.spread_bps();
        assert!((bps - 2.0).abs() < 0.01);
    }

    #[test]
    fn quote_crossed_locked() {
        let crossed = Quote::new("X", "V", 10.05, 1.0, 10.00, 1.0, 0, 0);
        assert!(crossed.is_crossed());
        let locked = Quote::new("X", "V", 10.00, 1.0, 10.00, 1.0, 0, 0);
        assert!(locked.is_locked());
    }

    #[test]
    fn trade_notional() {
        let t = Trade::new("AAPL", "NYSE", 150.0, 100.0, Side::Bid, 1000, 1);
        assert!((t.notional() - 15000.0).abs() < 1e-9);
    }

    #[test]
    fn level2_book_ordering() {
        let mut book = Level2Book::new("AAPL", 1000);
        book.add_bid(Level2Entry::new(150.00, 100.0, 5, "NYSE"));
        book.add_bid(Level2Entry::new(150.05, 200.0, 3, "ARCA"));
        book.add_ask(Level2Entry::new(150.15, 50.0, 2, "NYSE"));
        book.add_ask(Level2Entry::new(150.10, 150.0, 4, "BATS"));
        assert!((book.best_bid().unwrap().price - 150.05).abs() < 1e-9);
        assert!((book.best_ask().unwrap().price - 150.10).abs() < 1e-9);
    }

    #[test]
    fn level2_depth() {
        let mut book = Level2Book::new("AAPL", 1000);
        book.add_bid(Level2Entry::new(150.00, 100.0, 5, "NYSE"));
        book.add_bid(Level2Entry::new(149.95, 200.0, 3, "ARCA"));
        assert!((book.total_bid_depth() - 300.0).abs() < 1e-9);
        assert_eq!(book.depth_levels(), (2, 0));
    }

    #[test]
    fn nbbo_multi_venue() {
        let mut calc = NbboCalculator::new();
        calc.update(Quote::new("AAPL", "NYSE", 150.00, 100.0, 150.10, 100.0, 1000, 1));
        calc.update(Quote::new("AAPL", "ARCA", 150.02, 200.0, 150.08, 150.0, 1001, 2));
        let nbbo = calc.nbbo("AAPL").unwrap();
        assert!((nbbo.best_bid - 150.02).abs() < 1e-9);
        assert_eq!(nbbo.best_bid_venue, "ARCA");
        assert!((nbbo.best_ask - 150.08).abs() < 1e-9);
        assert_eq!(nbbo.best_ask_venue, "ARCA");
    }

    #[test]
    fn nbbo_missing_symbol() {
        let calc = NbboCalculator::new();
        assert!(calc.nbbo("AAPL").is_none());
    }

    #[test]
    fn feed_handler_lifecycle() {
        let mut fh = FeedHandler::new("NYSE");
        assert_eq!(fh.state(), FeedState::Disconnected);
        fh.connect().unwrap();
        assert_eq!(fh.state(), FeedState::Connecting);
        fh.on_connected().unwrap();
        assert_eq!(fh.state(), FeedState::Connected);
        fh.subscribe("AAPL").unwrap();
        assert_eq!(fh.state(), FeedState::Subscribing);
        fh.on_subscribed().unwrap();
        assert_eq!(fh.state(), FeedState::Streaming);
    }

    #[test]
    fn feed_handler_invalid_transition() {
        let mut fh = FeedHandler::new("NYSE");
        assert!(fh.on_connected().is_err());
    }

    #[test]
    fn feed_handler_error_recovery() {
        let mut fh = FeedHandler::new("NYSE");
        fh.connect().unwrap();
        fh.on_connected().unwrap();
        fh.on_error();
        assert_eq!(fh.state(), FeedState::Error);
        fh.recover().unwrap();
        assert_eq!(fh.state(), FeedState::Recovering);
        fh.on_recovered().unwrap();
        assert_eq!(fh.state(), FeedState::Streaming);
    }

    #[test]
    fn feed_handler_counters() {
        let mut fh = FeedHandler::new("NYSE");
        fh.connect().unwrap();
        fh.on_connected().unwrap();
        let q = Quote::new("AAPL", "NYSE", 150.0, 100.0, 150.1, 100.0, 1000, 1);
        fh.on_quote(&q);
        fh.on_quote(&q);
        let t = Trade::new("AAPL", "NYSE", 150.05, 50.0, Side::Bid, 1001, 2);
        fh.on_trade(&t);
        assert_eq!(fh.quotes_received(), 2);
        assert_eq!(fh.trades_received(), 1);
    }

    #[test]
    fn conflator_throttles() {
        let mut conf = Conflator::new(1000); // 1ms
        let q1 = Quote::new("AAPL", "NYSE", 150.0, 100.0, 150.1, 100.0, 0, 1);
        let r1 = conf.on_quote(q1);
        assert!(r1.is_some()); // first always goes through
        let q2 = Quote::new("AAPL", "NYSE", 150.01, 100.0, 150.11, 100.0, 500, 2);
        let r2 = conf.on_quote(q2);
        assert!(r2.is_none()); // within window
        let q3 = Quote::new("AAPL", "NYSE", 150.02, 100.0, 150.12, 100.0, 1001, 3);
        let r3 = conf.on_quote(q3);
        assert!(r3.is_some()); // past window
    }

    #[test]
    fn conflator_flush() {
        let mut conf = Conflator::new(100_000);
        let q = Quote::new("AAPL", "NYSE", 150.0, 100.0, 150.1, 100.0, 0, 1);
        conf.on_quote(q);
        let q2 = Quote::new("AAPL", "NYSE", 150.01, 100.0, 150.11, 100.0, 50, 2);
        conf.on_quote(q2);
        let (quotes, _trades) = conf.flush();
        assert_eq!(quotes.len(), 1); // last-value per symbol
    }

    #[test]
    fn stale_detector_fresh() {
        let mut det = StaleDetector::new(5_000_000); // 5 seconds
        det.record_update("AAPL", 10_000_000);
        assert!(!det.is_stale("AAPL", 12_000_000));
    }

    #[test]
    fn stale_detector_stale() {
        let mut det = StaleDetector::new(5_000_000);
        det.record_update("AAPL", 10_000_000);
        assert!(det.is_stale("AAPL", 20_000_000));
    }

    #[test]
    fn stale_detector_unknown_is_stale() {
        let det = StaleDetector::new(5_000_000);
        assert!(det.is_stale("AAPL", 0));
    }

    #[test]
    fn stale_detector_list() {
        let mut det = StaleDetector::new(1000);
        det.record_update("AAPL", 100);
        det.record_update("GOOG", 5000);
        let stale = det.stale_symbols(5500);
        assert!(stale.contains(&"AAPL".to_string()));
        assert!(!stale.contains(&"GOOG".to_string()));
    }

    #[test]
    fn display_impls() {
        let q = Quote::new("AAPL", "NYSE", 150.0, 100.0, 150.1, 100.0, 1000, 1);
        assert!(format!("{q}").contains("AAPL"));
        let t = Trade::new("AAPL", "NYSE", 150.05, 50.0, Side::Bid, 1000, 1);
        assert!(format!("{t}").contains("150.05"));
        let calc = NbboCalculator::new();
        assert!(format!("{calc}").contains("0 symbols"));
    }
}
