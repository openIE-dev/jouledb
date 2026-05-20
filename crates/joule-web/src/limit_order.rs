//! Limit Order — fill-or-kill, good-till-cancel, immediate-or-cancel
//! time-in-force policies, order lifecycle, and expiration tracking.
//!
//! Pure-Rust limit order representation with configurable time-in-force,
//! partial fill tracking, builder-pattern construction, and lifecycle
//! state management from new through terminal states.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LimitOrderError {
    InvalidPrice(String),
    InvalidQuantity(String),
    AlreadyTerminal(u64),
    InvalidTransition(String),
    InsufficientFill(String),
    Expired(u64),
}

impl fmt::Display for LimitOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
            Self::InvalidQuantity(s) => write!(f, "invalid quantity: {s}"),
            Self::AlreadyTerminal(id) => write!(f, "order {id} is already terminal"),
            Self::InvalidTransition(s) => write!(f, "invalid transition: {s}"),
            Self::InsufficientFill(s) => write!(f, "insufficient fill: {s}"),
            Self::Expired(id) => write!(f, "order {id} has expired"),
        }
    }
}

impl std::error::Error for LimitOrderError {}

// ── Side ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

// ── TimeInForce ─────────────────────────────────────────────────

/// Time-in-force policy for a limit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeInForce {
    /// Good-till-cancel: rests on book until filled or explicitly cancelled.
    GoodTillCancel,
    /// Fill-or-kill: must fill entirely on entry or be cancelled.
    FillOrKill,
    /// Immediate-or-cancel: fill what you can, cancel the rest.
    ImmediateOrCancel,
    /// Good-till-date: rests until a given expiry timestamp.
    GoodTillDate { expiry_ns: u64 },
    /// Day order: cancelled at end of session.
    Day,
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GoodTillCancel => write!(f, "GTC"),
            Self::FillOrKill => write!(f, "FOK"),
            Self::ImmediateOrCancel => write!(f, "IOC"),
            Self::GoodTillDate { expiry_ns } => write!(f, "GTD({expiry_ns})"),
            Self::Day => write!(f, "DAY"),
        }
    }
}

// ── OrderStatus ─────────────────────────────────────────────────

/// Lifecycle state of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderStatus {
    /// Newly created, not yet submitted.
    New,
    /// Accepted and resting on the book.
    Open,
    /// Partially filled, still resting.
    PartiallyFilled,
    /// Completely filled.
    Filled,
    /// Cancelled by user or system.
    Cancelled,
    /// Rejected at submission.
    Rejected,
    /// Expired (GTD or DAY).
    Expired,
}

impl OrderStatus {
    /// Whether this is a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Filled | Self::Cancelled | Self::Rejected | Self::Expired)
    }
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::New => write!(f, "NEW"),
            Self::Open => write!(f, "OPEN"),
            Self::PartiallyFilled => write!(f, "PARTIAL"),
            Self::Filled => write!(f, "FILLED"),
            Self::Cancelled => write!(f, "CANCELLED"),
            Self::Rejected => write!(f, "REJECTED"),
            Self::Expired => write!(f, "EXPIRED"),
        }
    }
}

// ── Fill ────────────────────────────────────────────────────────

/// Record of a partial or full fill against an order.
#[derive(Debug, Clone, PartialEq)]
pub struct Fill {
    pub fill_id: u64,
    pub quantity: f64,
    pub price: f64,
    pub timestamp_ns: u64,
    pub counterparty_order_id: u64,
}

impl Fill {
    pub fn new(fill_id: u64, quantity: f64, price: f64, timestamp_ns: u64, counter_id: u64) -> Self {
        Self { fill_id, quantity, price, timestamp_ns, counterparty_order_id: counter_id }
    }

    /// Notional value of the fill.
    pub fn notional(&self) -> f64 { self.quantity * self.price }
}

impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fill(id={}, {:.4}@{:.2}, notional={:.2})",
            self.fill_id, self.quantity, self.price, self.notional())
    }
}

// ── LimitOrder ──────────────────────────────────────────────────

/// A limit order with price, quantity, time-in-force, and fill tracking.
#[derive(Debug, Clone)]
pub struct LimitOrder {
    pub order_id: u64,
    pub side: OrderSide,
    pub price: f64,
    pub quantity: f64,
    pub remaining: f64,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub created_ns: u64,
    pub updated_ns: u64,
    pub fills: Vec<Fill>,
    pub client_order_id: Option<String>,
    pub min_quantity: f64,
    pub display_quantity: Option<f64>,
    pub post_only: bool,
}

impl LimitOrder {
    pub fn new(order_id: u64, side: OrderSide, price: f64, quantity: f64) -> Self {
        Self {
            order_id,
            side,
            price,
            quantity,
            remaining: quantity,
            time_in_force: TimeInForce::GoodTillCancel,
            status: OrderStatus::New,
            created_ns: 0,
            updated_ns: 0,
            fills: Vec::new(),
            client_order_id: None,
            min_quantity: 0.0,
            display_quantity: None,
            post_only: false,
        }
    }

    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    pub fn with_created_ns(mut self, ts: u64) -> Self {
        self.created_ns = ts;
        self.updated_ns = ts;
        self
    }

    pub fn with_client_order_id(mut self, id: &str) -> Self {
        self.client_order_id = Some(id.to_string());
        self
    }

    pub fn with_min_quantity(mut self, mq: f64) -> Self {
        self.min_quantity = mq;
        self
    }

    pub fn with_display_quantity(mut self, dq: f64) -> Self {
        self.display_quantity = Some(dq);
        self
    }

    pub fn with_post_only(mut self, po: bool) -> Self {
        self.post_only = po;
        self
    }

    /// Validate the order parameters.
    pub fn validate(&self) -> Result<(), LimitOrderError> {
        if self.price <= 0.0 {
            return Err(LimitOrderError::InvalidPrice(format!("{}", self.price)));
        }
        if self.quantity <= 0.0 {
            return Err(LimitOrderError::InvalidQuantity(format!("{}", self.quantity)));
        }
        if self.min_quantity > self.quantity {
            return Err(LimitOrderError::InvalidQuantity(
                format!("min_quantity {} > quantity {}", self.min_quantity, self.quantity)));
        }
        Ok(())
    }

    /// Mark as open (accepted on book).
    pub fn accept(&mut self, timestamp_ns: u64) -> Result<(), LimitOrderError> {
        if self.status.is_terminal() {
            return Err(LimitOrderError::AlreadyTerminal(self.order_id));
        }
        self.status = OrderStatus::Open;
        self.updated_ns = timestamp_ns;
        Ok(())
    }

    /// Apply a fill to this order.
    pub fn apply_fill(&mut self, fill: Fill) -> Result<(), LimitOrderError> {
        if self.status.is_terminal() {
            return Err(LimitOrderError::AlreadyTerminal(self.order_id));
        }
        if fill.quantity > self.remaining + 1e-12 {
            return Err(LimitOrderError::InsufficientFill(
                format!("fill {} > remaining {}", fill.quantity, self.remaining)));
        }
        self.remaining -= fill.quantity;
        if self.remaining < 1e-12 {
            self.remaining = 0.0;
        }
        self.updated_ns = fill.timestamp_ns;
        self.fills.push(fill);
        self.status = if self.remaining <= 1e-12 {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
        Ok(())
    }

    /// Cancel the order.
    pub fn cancel(&mut self, timestamp_ns: u64) -> Result<(), LimitOrderError> {
        if self.status.is_terminal() {
            return Err(LimitOrderError::AlreadyTerminal(self.order_id));
        }
        self.status = OrderStatus::Cancelled;
        self.updated_ns = timestamp_ns;
        Ok(())
    }

    /// Expire the order (GTD / DAY).
    pub fn expire(&mut self, timestamp_ns: u64) -> Result<(), LimitOrderError> {
        if self.status.is_terminal() {
            return Err(LimitOrderError::AlreadyTerminal(self.order_id));
        }
        self.status = OrderStatus::Expired;
        self.updated_ns = timestamp_ns;
        Ok(())
    }

    /// Check if the order has expired at the given time.
    pub fn is_expired_at(&self, now_ns: u64) -> bool {
        match self.time_in_force {
            TimeInForce::GoodTillDate { expiry_ns } => now_ns >= expiry_ns,
            _ => false,
        }
    }

    /// Filled quantity.
    pub fn filled_quantity(&self) -> f64 {
        self.quantity - self.remaining
    }

    /// Fill ratio as fraction.
    pub fn fill_ratio(&self) -> f64 {
        if self.quantity <= 0.0 { 0.0 } else { self.filled_quantity() / self.quantity }
    }

    /// Average fill price.
    pub fn avg_fill_price(&self) -> Option<f64> {
        let total_notional: f64 = self.fills.iter().map(|f| f.notional()).sum();
        let total_qty: f64 = self.fills.iter().map(|f| f.quantity).sum();
        if total_qty > 1e-12 { Some(total_notional / total_qty) } else { None }
    }

    /// Total notional value of all fills.
    pub fn total_notional(&self) -> f64 {
        self.fills.iter().map(|f| f.notional()).sum()
    }

    /// Whether the order would cross an opposite price.
    pub fn would_cross(&self, opposite_price: f64) -> bool {
        match self.side {
            OrderSide::Buy => self.price >= opposite_price,
            OrderSide::Sell => self.price <= opposite_price,
        }
    }

    /// Whether TIF allows resting on book.
    pub fn can_rest(&self) -> bool {
        matches!(self.time_in_force,
            TimeInForce::GoodTillCancel | TimeInForce::GoodTillDate { .. } | TimeInForce::Day)
    }
}

impl fmt::Display for LimitOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LimitOrder(id={}, {} {:.4}@{:.2}, tif={}, status={}, filled={:.4})",
            self.order_id, self.side, self.quantity, self.price,
            self.time_in_force, self.status, self.filled_quantity())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_order() {
        let o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        assert_eq!(o.status, OrderStatus::New);
        assert_eq!(o.remaining, 50.0);
        assert!(!o.status.is_terminal());
    }

    #[test]
    fn test_validate_good() {
        let o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        assert!(o.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_price() {
        let o = LimitOrder::new(1, OrderSide::Buy, -1.0, 50.0);
        assert!(o.validate().is_err());
    }

    #[test]
    fn test_validate_bad_quantity() {
        let o = LimitOrder::new(1, OrderSide::Buy, 100.0, 0.0);
        assert!(o.validate().is_err());
    }

    #[test]
    fn test_accept() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        o.accept(1000).unwrap();
        assert_eq!(o.status, OrderStatus::Open);
    }

    #[test]
    fn test_fill_partial() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 100.0);
        o.accept(1000).unwrap();
        o.apply_fill(Fill::new(1, 30.0, 100.0, 2000, 99)).unwrap();
        assert_eq!(o.status, OrderStatus::PartiallyFilled);
        assert!((o.remaining - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_fill_complete() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 100.0);
        o.accept(1000).unwrap();
        o.apply_fill(Fill::new(1, 100.0, 100.0, 2000, 99)).unwrap();
        assert_eq!(o.status, OrderStatus::Filled);
        assert!(o.status.is_terminal());
    }

    #[test]
    fn test_fill_too_large() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        o.accept(1000).unwrap();
        assert!(o.apply_fill(Fill::new(1, 60.0, 100.0, 2000, 99)).is_err());
    }

    #[test]
    fn test_cancel() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        o.accept(1000).unwrap();
        o.cancel(2000).unwrap();
        assert_eq!(o.status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_cancel_terminal() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        o.accept(1000).unwrap();
        o.cancel(2000).unwrap();
        assert!(o.cancel(3000).is_err());
    }

    #[test]
    fn test_expire() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0)
            .with_time_in_force(TimeInForce::GoodTillDate { expiry_ns: 5000 });
        o.accept(1000).unwrap();
        assert!(!o.is_expired_at(4000));
        assert!(o.is_expired_at(5000));
        o.expire(5000).unwrap();
        assert_eq!(o.status, OrderStatus::Expired);
    }

    #[test]
    fn test_avg_fill_price() {
        let mut o = LimitOrder::new(1, OrderSide::Buy, 100.0, 100.0);
        o.accept(1000).unwrap();
        o.apply_fill(Fill::new(1, 50.0, 99.0, 2000, 10)).unwrap();
        o.apply_fill(Fill::new(2, 50.0, 101.0, 3000, 11)).unwrap();
        let avg = o.avg_fill_price().unwrap();
        assert!((avg - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_fill_ratio() {
        let mut o = LimitOrder::new(1, OrderSide::Sell, 200.0, 100.0);
        o.accept(1000).unwrap();
        o.apply_fill(Fill::new(1, 25.0, 200.0, 2000, 10)).unwrap();
        assert!((o.fill_ratio() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_would_cross_buy() {
        let o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        assert!(o.would_cross(100.0));
        assert!(o.would_cross(99.0));
        assert!(!o.would_cross(101.0));
    }

    #[test]
    fn test_would_cross_sell() {
        let o = LimitOrder::new(1, OrderSide::Sell, 100.0, 50.0);
        assert!(o.would_cross(100.0));
        assert!(o.would_cross(101.0));
        assert!(!o.would_cross(99.0));
    }

    #[test]
    fn test_can_rest() {
        let gtc = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0);
        assert!(gtc.can_rest());
        let fok = LimitOrder::new(2, OrderSide::Buy, 100.0, 50.0)
            .with_time_in_force(TimeInForce::FillOrKill);
        assert!(!fok.can_rest());
    }

    #[test]
    fn test_builder_chain() {
        let o = LimitOrder::new(1, OrderSide::Buy, 100.0, 50.0)
            .with_time_in_force(TimeInForce::ImmediateOrCancel)
            .with_created_ns(5000)
            .with_client_order_id("client-42")
            .with_min_quantity(10.0)
            .with_post_only(true);
        assert_eq!(o.time_in_force, TimeInForce::ImmediateOrCancel);
        assert_eq!(o.client_order_id.as_deref(), Some("client-42"));
        assert!(o.post_only);
    }

    #[test]
    fn test_display() {
        let o = LimitOrder::new(42, OrderSide::Sell, 150.50, 200.0)
            .with_time_in_force(TimeInForce::FillOrKill);
        let s = format!("{o}");
        assert!(s.contains("42"));
        assert!(s.contains("SELL"));
        assert!(s.contains("FOK"));
    }

    #[test]
    fn test_fill_display() {
        let fill = Fill::new(7, 50.0, 100.0, 1000, 99);
        let s = format!("{fill}");
        assert!(s.contains("5000.00")); // notional
    }

    #[test]
    fn test_tif_display() {
        assert_eq!(format!("{}", TimeInForce::GoodTillCancel), "GTC");
        assert_eq!(format!("{}", TimeInForce::FillOrKill), "FOK");
        assert_eq!(format!("{}", TimeInForce::ImmediateOrCancel), "IOC");
        assert_eq!(format!("{}", TimeInForce::Day), "DAY");
    }
}
