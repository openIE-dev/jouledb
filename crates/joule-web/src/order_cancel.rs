//! Order Cancellation — order cancellation, cancel-replace, mass cancel,
//! cancel rate tracking, and cancellation audit trail.
//!
//! Pure-Rust cancellation engine supporting single-order cancel,
//! cancel-replace (atomic amend), mass cancel by criteria, cancel rate
//! monitoring, and a complete audit log of cancellation events.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CancelError {
    OrderNotFound(u64),
    AlreadyCancelled(u64),
    AlreadyFilled(u64),
    InvalidReplacement(String),
    RateLimitExceeded(String),
    MassCancelFailed(String),
}

impl fmt::Display for CancelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrderNotFound(id) => write!(f, "order not found: {id}"),
            Self::AlreadyCancelled(id) => write!(f, "order already cancelled: {id}"),
            Self::AlreadyFilled(id) => write!(f, "order already filled: {id}"),
            Self::InvalidReplacement(s) => write!(f, "invalid replacement: {s}"),
            Self::RateLimitExceeded(s) => write!(f, "cancel rate limit exceeded: {s}"),
            Self::MassCancelFailed(s) => write!(f, "mass cancel failed: {s}"),
        }
    }
}

impl std::error::Error for CancelError {}

// ── CancelReason ────────────────────────────────────────────────

/// Reason for cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CancelReason {
    UserRequested,
    TimeInForceExpiry,
    SelfTradePrevention,
    RiskLimit,
    MassCancel,
    SystemCancel,
    CancelReplace,
    SessionEnd,
}

impl fmt::Display for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserRequested => write!(f, "USER"),
            Self::TimeInForceExpiry => write!(f, "TIF_EXPIRY"),
            Self::SelfTradePrevention => write!(f, "STP"),
            Self::RiskLimit => write!(f, "RISK"),
            Self::MassCancel => write!(f, "MASS"),
            Self::SystemCancel => write!(f, "SYSTEM"),
            Self::CancelReplace => write!(f, "REPLACE"),
            Self::SessionEnd => write!(f, "SESSION_END"),
        }
    }
}

// ── CancelledOrderState ─────────────────────────────────────────

/// State of an order relevant to cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderState {
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
}

impl OrderState {
    pub fn is_cancellable(self) -> bool {
        matches!(self, Self::Open | Self::PartiallyFilled)
    }
}

impl fmt::Display for OrderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "OPEN"),
            Self::PartiallyFilled => write!(f, "PARTIAL"),
            Self::Filled => write!(f, "FILLED"),
            Self::Cancelled => write!(f, "CANCELLED"),
        }
    }
}

// ── CancelRequest ───────────────────────────────────────────────

/// A request to cancel a single order.
#[derive(Debug, Clone)]
pub struct CancelRequest {
    pub order_id: u64,
    pub reason: CancelReason,
    pub timestamp_ns: u64,
    pub client_cancel_id: Option<String>,
}

impl CancelRequest {
    pub fn new(order_id: u64, reason: CancelReason, timestamp_ns: u64) -> Self {
        Self { order_id, reason, timestamp_ns, client_cancel_id: None }
    }

    pub fn with_client_cancel_id(mut self, id: &str) -> Self {
        self.client_cancel_id = Some(id.to_string());
        self
    }
}

impl fmt::Display for CancelRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CancelReq(order={}, reason={})", self.order_id, self.reason)
    }
}

// ── CancelReplaceRequest ────────────────────────────────────────

/// A request to cancel an order and replace it atomically.
#[derive(Debug, Clone)]
pub struct CancelReplaceRequest {
    pub original_order_id: u64,
    pub new_price: Option<f64>,
    pub new_quantity: Option<f64>,
    pub timestamp_ns: u64,
    pub client_replace_id: Option<String>,
}

impl CancelReplaceRequest {
    pub fn new(original_id: u64, timestamp_ns: u64) -> Self {
        Self {
            original_order_id: original_id,
            new_price: None,
            new_quantity: None,
            timestamp_ns,
            client_replace_id: None,
        }
    }

    pub fn with_new_price(mut self, price: f64) -> Self {
        self.new_price = Some(price);
        self
    }

    pub fn with_new_quantity(mut self, qty: f64) -> Self {
        self.new_quantity = Some(qty);
        self
    }

    pub fn with_client_replace_id(mut self, id: &str) -> Self {
        self.client_replace_id = Some(id.to_string());
        self
    }

    /// Validate the replacement parameters.
    pub fn validate(&self) -> Result<(), CancelError> {
        if let Some(p) = self.new_price {
            if p <= 0.0 {
                return Err(CancelError::InvalidReplacement(format!("price {p}")));
            }
        }
        if let Some(q) = self.new_quantity {
            if q <= 0.0 {
                return Err(CancelError::InvalidReplacement(format!("quantity {q}")));
            }
        }
        if self.new_price.is_none() && self.new_quantity.is_none() {
            return Err(CancelError::InvalidReplacement("no changes specified".into()));
        }
        Ok(())
    }
}

impl fmt::Display for CancelReplaceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let px = self.new_price.map_or("--".into(), |p| format!("{p:.2}"));
        let qty = self.new_quantity.map_or("--".into(), |q| format!("{q:.4}"));
        write!(f, "CancelReplace(order={}, px={px}, qty={qty})", self.original_order_id)
    }
}

// ── MassCancelCriteria ──────────────────────────────────────────

/// Criteria for a mass cancellation.
#[derive(Debug, Clone)]
pub struct MassCancelCriteria {
    pub side: Option<CancelSide>,
    pub min_price: Option<f64>,
    pub max_price: Option<f64>,
    pub symbol: Option<String>,
    pub reason: CancelReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CancelSide {
    Buy,
    Sell,
}

impl fmt::Display for CancelSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

impl MassCancelCriteria {
    pub fn all(reason: CancelReason) -> Self {
        Self { side: None, min_price: None, max_price: None, symbol: None, reason }
    }

    pub fn with_side(mut self, side: CancelSide) -> Self {
        self.side = Some(side);
        self
    }

    pub fn with_price_range(mut self, min: f64, max: f64) -> Self {
        self.min_price = Some(min);
        self.max_price = Some(max);
        self
    }

    pub fn with_symbol(mut self, sym: &str) -> Self {
        self.symbol = Some(sym.to_string());
        self
    }

    /// Check if an order matches these criteria.
    pub fn matches(&self, side: CancelSide, price: f64, symbol: &str) -> bool {
        if let Some(s) = self.side {
            if s != side { return false; }
        }
        if let Some(min) = self.min_price {
            if price < min { return false; }
        }
        if let Some(max) = self.max_price {
            if price > max { return false; }
        }
        if let Some(ref sym) = self.symbol {
            if sym != symbol { return false; }
        }
        true
    }
}

impl fmt::Display for MassCancelCriteria {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side_str = self.side.map_or("ALL".into(), |s| format!("{s}"));
        write!(f, "MassCancel(side={side_str}, reason={})", self.reason)
    }
}

// ── CancelEvent ─────────────────────────────────────────────────

/// An audit event for a cancellation.
#[derive(Debug, Clone)]
pub struct CancelEvent {
    pub event_id: u64,
    pub order_id: u64,
    pub reason: CancelReason,
    pub timestamp_ns: u64,
    pub remaining_at_cancel: f64,
    pub original_quantity: f64,
}

impl CancelEvent {
    /// Fraction that was unfilled at cancellation.
    pub fn cancel_ratio(&self) -> f64 {
        if self.original_quantity <= 0.0 { 0.0 }
        else { self.remaining_at_cancel / self.original_quantity }
    }
}

impl fmt::Display for CancelEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CancelEvent(id={}, order={}, reason={}, rem={:.4})",
            self.event_id, self.order_id, self.reason, self.remaining_at_cancel)
    }
}

// ── CancelRateTracker ───────────────────────────────────────────

/// Tracks cancel-to-order rates over a sliding window.
#[derive(Debug, Clone)]
pub struct CancelRateTracker {
    window_ns: u64,
    orders_submitted: Vec<u64>,
    cancels: Vec<u64>,
    max_rate: f64,
}

impl CancelRateTracker {
    pub fn new(window_ns: u64) -> Self {
        Self { window_ns, orders_submitted: Vec::new(), cancels: Vec::new(), max_rate: 1.0 }
    }

    pub fn with_max_rate(mut self, rate: f64) -> Self {
        self.max_rate = rate;
        self
    }

    /// Record an order submission.
    pub fn record_order(&mut self, timestamp_ns: u64) {
        self.orders_submitted.push(timestamp_ns);
        self.prune(timestamp_ns);
    }

    /// Record a cancellation.
    pub fn record_cancel(&mut self, timestamp_ns: u64) {
        self.cancels.push(timestamp_ns);
        self.prune(timestamp_ns);
    }

    fn prune(&mut self, now_ns: u64) {
        let cutoff = now_ns.saturating_sub(self.window_ns);
        self.orders_submitted.retain(|t| *t >= cutoff);
        self.cancels.retain(|t| *t >= cutoff);
    }

    /// Current cancel-to-order ratio in the window.
    pub fn cancel_rate(&self) -> f64 {
        if self.orders_submitted.is_empty() { return 0.0; }
        self.cancels.len() as f64 / self.orders_submitted.len() as f64
    }

    /// Whether the rate exceeds the maximum.
    pub fn is_exceeded(&self) -> bool {
        self.cancel_rate() > self.max_rate
    }

    /// Orders in window.
    pub fn order_count(&self) -> usize { self.orders_submitted.len() }

    /// Cancels in window.
    pub fn cancel_count(&self) -> usize { self.cancels.len() }
}

impl fmt::Display for CancelRateTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CancelRate(rate={:.3}, orders={}, cancels={}, max={:.3})",
            self.cancel_rate(), self.order_count(), self.cancel_count(), self.max_rate)
    }
}

// ── CancelEngine ────────────────────────────────────────────────

/// Order cancellation engine with audit trail and rate tracking.
#[derive(Debug, Clone)]
pub struct CancelEngine {
    events: Vec<CancelEvent>,
    next_event_id: u64,
    rate_tracker: CancelRateTracker,
}

impl CancelEngine {
    pub fn new(window_ns: u64) -> Self {
        Self {
            events: Vec::new(),
            next_event_id: 1,
            rate_tracker: CancelRateTracker::new(window_ns),
        }
    }

    pub fn with_max_cancel_rate(mut self, rate: f64) -> Self {
        self.rate_tracker = self.rate_tracker.with_max_rate(rate);
        self
    }

    /// Record an order submission (for rate tracking).
    pub fn record_order(&mut self, timestamp_ns: u64) {
        self.rate_tracker.record_order(timestamp_ns);
    }

    /// Process a cancel request.
    pub fn cancel(&mut self, state: OrderState, original_qty: f64, remaining: f64,
                  req: &CancelRequest) -> Result<CancelEvent, CancelError>
    {
        if !state.is_cancellable() {
            if state == OrderState::Cancelled {
                return Err(CancelError::AlreadyCancelled(req.order_id));
            }
            return Err(CancelError::AlreadyFilled(req.order_id));
        }
        if self.rate_tracker.is_exceeded() {
            return Err(CancelError::RateLimitExceeded(
                format!("rate={:.3}", self.rate_tracker.cancel_rate())));
        }
        let event = CancelEvent {
            event_id: self.next_event_id,
            order_id: req.order_id,
            reason: req.reason.clone(),
            timestamp_ns: req.timestamp_ns,
            remaining_at_cancel: remaining,
            original_quantity: original_qty,
        };
        self.next_event_id += 1;
        self.rate_tracker.record_cancel(req.timestamp_ns);
        self.events.push(event.clone());
        Ok(event)
    }

    /// Process a cancel-replace request.
    pub fn cancel_replace(&mut self, state: OrderState, original_qty: f64, remaining: f64,
                          req: &CancelReplaceRequest) -> Result<CancelEvent, CancelError>
    {
        req.validate()?;
        let cancel_req = CancelRequest::new(
            req.original_order_id, CancelReason::CancelReplace, req.timestamp_ns);
        self.cancel(state, original_qty, remaining, &cancel_req)
    }

    /// Process a mass cancel. Returns count of orders that would be cancelled.
    pub fn mass_cancel_count(&self, orders: &[(u64, CancelSide, f64, &str)],
                             criteria: &MassCancelCriteria) -> usize
    {
        orders.iter()
            .filter(|(_, side, price, sym)| criteria.matches(*side, *price, sym))
            .count()
    }

    /// Cancel events accessor.
    pub fn events(&self) -> &[CancelEvent] { &self.events }

    /// Total cancellation count.
    pub fn total_cancels(&self) -> usize { self.events.len() }

    /// Rate tracker accessor.
    pub fn rate_tracker(&self) -> &CancelRateTracker { &self.rate_tracker }

    /// Cancellation events for a specific order.
    pub fn events_for_order(&self, order_id: u64) -> Vec<&CancelEvent> {
        self.events.iter().filter(|e| e.order_id == order_id).collect()
    }

    /// Average cancel ratio across all events.
    pub fn avg_cancel_ratio(&self) -> f64 {
        if self.events.is_empty() { return 0.0; }
        let sum: f64 = self.events.iter().map(|e| e.cancel_ratio()).sum();
        sum / self.events.len() as f64
    }
}

impl fmt::Display for CancelEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CancelEngine(events={}, rate={:.3})",
            self.events.len(), self.rate_tracker.cancel_rate())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_open_order() {
        let mut eng = CancelEngine::new(1_000_000_000);
        eng.record_order(1000);
        let req = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        let event = eng.cancel(OrderState::Open, 100.0, 100.0, &req).unwrap();
        assert_eq!(event.order_id, 1);
        assert!((event.cancel_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cancel_partial_order() {
        let mut eng = CancelEngine::new(1_000_000_000);
        eng.record_order(1000);
        let req = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        let event = eng.cancel(OrderState::PartiallyFilled, 100.0, 40.0, &req).unwrap();
        assert!((event.cancel_ratio() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_cancel_filled() {
        let mut eng = CancelEngine::new(1_000_000_000);
        let req = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        assert!(eng.cancel(OrderState::Filled, 100.0, 0.0, &req).is_err());
    }

    #[test]
    fn test_cancel_already_cancelled() {
        let mut eng = CancelEngine::new(1_000_000_000);
        let req = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        assert!(eng.cancel(OrderState::Cancelled, 100.0, 100.0, &req).is_err());
    }

    #[test]
    fn test_cancel_replace() {
        let mut eng = CancelEngine::new(1_000_000_000);
        eng.record_order(1000);
        let req = CancelReplaceRequest::new(1, 2000).with_new_price(101.0);
        let event = eng.cancel_replace(OrderState::Open, 100.0, 100.0, &req).unwrap();
        assert_eq!(event.reason, CancelReason::CancelReplace);
    }

    #[test]
    fn test_cancel_replace_no_changes() {
        let mut eng = CancelEngine::new(1_000_000_000);
        let req = CancelReplaceRequest::new(1, 2000);
        assert!(eng.cancel_replace(OrderState::Open, 100.0, 100.0, &req).is_err());
    }

    #[test]
    fn test_cancel_replace_invalid_price() {
        let mut eng = CancelEngine::new(1_000_000_000);
        let req = CancelReplaceRequest::new(1, 2000).with_new_price(-5.0);
        assert!(eng.cancel_replace(OrderState::Open, 100.0, 100.0, &req).is_err());
    }

    #[test]
    fn test_rate_tracker_basic() {
        let mut rt = CancelRateTracker::new(10_000);
        rt.record_order(1000);
        rt.record_order(2000);
        rt.record_cancel(3000);
        assert!((rt.cancel_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rate_tracker_window() {
        let mut rt = CancelRateTracker::new(5000);
        rt.record_order(1000);
        rt.record_cancel(2000);
        // After window expires
        rt.record_order(8000); // prunes old
        assert_eq!(rt.order_count(), 1);
        assert_eq!(rt.cancel_count(), 0);
    }

    #[test]
    fn test_rate_limit_exceeded() {
        let mut eng = CancelEngine::new(1_000_000_000).with_max_cancel_rate(0.5);
        eng.record_order(1000);
        eng.record_order(1000);
        eng.record_order(1000);
        // First cancel succeeds: rate before = 0/3 = 0.0
        let req1 = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        assert!(eng.cancel(OrderState::Open, 100.0, 100.0, &req1).is_ok());
        // Second cancel succeeds: rate before = 1/3 = 0.33
        let req2 = CancelRequest::new(2, CancelReason::UserRequested, 3000);
        assert!(eng.cancel(OrderState::Open, 100.0, 100.0, &req2).is_ok());
        // Third cancel fails: rate before = 2/3 = 0.67 > 0.5
        let req3 = CancelRequest::new(3, CancelReason::UserRequested, 4000);
        assert!(eng.cancel(OrderState::Open, 100.0, 100.0, &req3).is_err());
    }

    #[test]
    fn test_mass_cancel_criteria() {
        let criteria = MassCancelCriteria::all(CancelReason::MassCancel)
            .with_side(CancelSide::Buy)
            .with_price_range(95.0, 105.0);
        assert!(criteria.matches(CancelSide::Buy, 100.0, "AAPL"));
        assert!(!criteria.matches(CancelSide::Sell, 100.0, "AAPL"));
        assert!(!criteria.matches(CancelSide::Buy, 110.0, "AAPL"));
    }

    #[test]
    fn test_mass_cancel_count() {
        let eng = CancelEngine::new(1_000_000_000);
        let orders = vec![
            (1u64, CancelSide::Buy, 100.0, "AAPL"),
            (2, CancelSide::Sell, 101.0, "AAPL"),
            (3, CancelSide::Buy, 99.0, "GOOG"),
        ];
        let criteria = MassCancelCriteria::all(CancelReason::MassCancel)
            .with_side(CancelSide::Buy);
        assert_eq!(eng.mass_cancel_count(&orders, &criteria), 2);
    }

    #[test]
    fn test_events_for_order() {
        let mut eng = CancelEngine::new(1_000_000_000);
        eng.record_order(1000);
        eng.record_order(1000);
        let r1 = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        eng.cancel(OrderState::Open, 100.0, 100.0, &r1).unwrap();
        let r2 = CancelRequest::new(2, CancelReason::UserRequested, 3000);
        eng.cancel(OrderState::Open, 50.0, 50.0, &r2).unwrap();
        assert_eq!(eng.events_for_order(1).len(), 1);
    }

    #[test]
    fn test_avg_cancel_ratio() {
        let mut eng = CancelEngine::new(1_000_000_000);
        eng.record_order(1000);
        eng.record_order(1000);
        let r1 = CancelRequest::new(1, CancelReason::UserRequested, 2000);
        eng.cancel(OrderState::Open, 100.0, 100.0, &r1).unwrap();  // ratio=1.0
        let r2 = CancelRequest::new(2, CancelReason::UserRequested, 3000);
        eng.cancel(OrderState::PartiallyFilled, 100.0, 50.0, &r2).unwrap(); // ratio=0.5
        assert!((eng.avg_cancel_ratio() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_display_engine() {
        let eng = CancelEngine::new(1_000_000_000);
        let s = format!("{eng}");
        assert!(s.contains("events=0"));
    }

    #[test]
    fn test_display_event() {
        let e = CancelEvent {
            event_id: 1, order_id: 42, reason: CancelReason::UserRequested,
            timestamp_ns: 1000, remaining_at_cancel: 50.0, original_quantity: 100.0,
        };
        let s = format!("{e}");
        assert!(s.contains("42"));
        assert!(s.contains("USER"));
    }

    #[test]
    fn test_cancel_request_display() {
        let r = CancelRequest::new(7, CancelReason::SessionEnd, 5000)
            .with_client_cancel_id("cc-1");
        let s = format!("{r}");
        assert!(s.contains("7"));
        assert!(s.contains("SESSION_END"));
    }

    #[test]
    fn test_cancel_replace_display() {
        let r = CancelReplaceRequest::new(3, 1000)
            .with_new_price(105.0)
            .with_new_quantity(200.0)
            .with_client_replace_id("cr-1");
        let s = format!("{r}");
        assert!(s.contains("105.00"));
    }
}
