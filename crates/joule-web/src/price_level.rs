//! Price Level — price level aggregation, volume at price, order queue
//! management, and level statistics for order book analytics.
//!
//! Pure-Rust price level with FIFO order queue, volume tracking,
//! order count, and statistical accessors for market microstructure
//! analysis.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PriceLevelError {
    InvalidPrice(String),
    InvalidQuantity(String),
    OrderNotFound(u64),
    DuplicateOrder(u64),
    EmptyLevel,
}

impl fmt::Display for PriceLevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
            Self::InvalidQuantity(s) => write!(f, "invalid quantity: {s}"),
            Self::OrderNotFound(id) => write!(f, "order not found at level: {id}"),
            Self::DuplicateOrder(id) => write!(f, "duplicate order at level: {id}"),
            Self::EmptyLevel => write!(f, "price level is empty"),
        }
    }
}

impl std::error::Error for PriceLevelError {}

// ── LevelSide ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LevelSide {
    Bid,
    Ask,
}

impl fmt::Display for LevelSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bid => write!(f, "BID"),
            Self::Ask => write!(f, "ASK"),
        }
    }
}

// ── QueuedOrder ─────────────────────────────────────────────────

/// An order queued at a price level.
#[derive(Debug, Clone, PartialEq)]
pub struct QueuedOrder {
    pub order_id: u64,
    pub quantity: f64,
    pub remaining: f64,
    pub timestamp_ns: u64,
    pub is_hidden: bool,
}

impl QueuedOrder {
    pub fn new(order_id: u64, quantity: f64, timestamp_ns: u64) -> Self {
        Self { order_id, quantity, remaining: quantity, timestamp_ns, is_hidden: false }
    }

    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.is_hidden = hidden;
        self
    }

    pub fn is_filled(&self) -> bool { self.remaining <= 1e-12 }

    pub fn fill_ratio(&self) -> f64 {
        if self.quantity <= 0.0 { 0.0 } else { 1.0 - self.remaining / self.quantity }
    }

    /// Visible quantity (zero if hidden).
    pub fn visible_quantity(&self) -> f64 {
        if self.is_hidden { 0.0 } else { self.remaining }
    }
}

impl fmt::Display for QueuedOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hidden = if self.is_hidden { " [hidden]" } else { "" };
        write!(f, "Queued(id={}, rem={:.4}/{:.4}{hidden})",
            self.order_id, self.remaining, self.quantity)
    }
}

// ── LevelStats ──────────────────────────────────────────────────

/// Statistics for a price level.
#[derive(Debug, Clone)]
pub struct LevelStats {
    pub price: f64,
    pub side: LevelSide,
    pub order_count: usize,
    pub total_volume: f64,
    pub visible_volume: f64,
    pub hidden_volume: f64,
    pub avg_order_size: f64,
    pub max_order_size: f64,
    pub min_order_size: f64,
    pub notional: f64,
}

impl fmt::Display for LevelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LevelStats({} {:.2}: vol={:.4}, orders={}, notional={:.2})",
            self.side, self.price, self.total_volume, self.order_count, self.notional)
    }
}

// ── PriceLevel ──────────────────────────────────────────────────

/// A price level in the order book with a FIFO queue of orders.
#[derive(Debug, Clone)]
pub struct PriceLevel {
    price: f64,
    side: LevelSide,
    orders: Vec<QueuedOrder>,
    total_volume: f64,
    visible_volume: f64,
}

impl PriceLevel {
    pub fn new(price: f64, side: LevelSide) -> Result<Self, PriceLevelError> {
        if price <= 0.0 {
            return Err(PriceLevelError::InvalidPrice(format!("{price}")));
        }
        Ok(Self { price, side, orders: Vec::new(), total_volume: 0.0, visible_volume: 0.0 })
    }

    pub fn with_price(mut self, price: f64) -> Self {
        self.price = price;
        self
    }

    pub fn with_side(mut self, side: LevelSide) -> Self {
        self.side = side;
        self
    }

    /// Add an order to the end of the queue (time priority).
    pub fn add_order(&mut self, order: QueuedOrder) -> Result<(), PriceLevelError> {
        if order.remaining <= 0.0 {
            return Err(PriceLevelError::InvalidQuantity(format!("{}", order.remaining)));
        }
        if self.orders.iter().any(|o| o.order_id == order.order_id) {
            return Err(PriceLevelError::DuplicateOrder(order.order_id));
        }
        self.total_volume += order.remaining;
        self.visible_volume += order.visible_quantity();
        self.orders.push(order);
        Ok(())
    }

    /// Remove an order by id.
    pub fn remove_order(&mut self, order_id: u64) -> Result<QueuedOrder, PriceLevelError> {
        let pos = self.orders.iter().position(|o| o.order_id == order_id)
            .ok_or(PriceLevelError::OrderNotFound(order_id))?;
        let order = self.orders.remove(pos);
        self.total_volume -= order.remaining;
        self.visible_volume -= order.visible_quantity();
        Ok(order)
    }

    /// Reduce the remaining quantity of the front-of-queue order.
    pub fn fill_front(&mut self, qty: f64) -> Result<(u64, f64), PriceLevelError> {
        if self.orders.is_empty() {
            return Err(PriceLevelError::EmptyLevel);
        }
        let front = &mut self.orders[0];
        let actual = qty.min(front.remaining);
        front.remaining -= actual;
        self.total_volume -= actual;
        if !front.is_hidden {
            self.visible_volume -= actual;
        }
        let id = front.order_id;
        if front.is_filled() {
            self.orders.remove(0);
        }
        Ok((id, actual))
    }

    /// Reduce quantity of a specific order.
    pub fn fill_order(&mut self, order_id: u64, qty: f64) -> Result<f64, PriceLevelError> {
        let order = self.orders.iter_mut().find(|o| o.order_id == order_id)
            .ok_or(PriceLevelError::OrderNotFound(order_id))?;
        let actual = qty.min(order.remaining);
        order.remaining -= actual;
        self.total_volume -= actual;
        if !order.is_hidden {
            self.visible_volume -= actual;
        }
        if order.is_filled() {
            self.orders.retain(|o| !o.is_filled());
        }
        Ok(actual)
    }

    /// Modify the quantity of an existing order.
    pub fn modify_order(&mut self, order_id: u64, new_remaining: f64)
        -> Result<f64, PriceLevelError>
    {
        if new_remaining <= 0.0 {
            return Err(PriceLevelError::InvalidQuantity(format!("{new_remaining}")));
        }
        let order = self.orders.iter_mut().find(|o| o.order_id == order_id)
            .ok_or(PriceLevelError::OrderNotFound(order_id))?;
        let old = order.remaining;
        let delta = new_remaining - old;
        order.remaining = new_remaining;
        self.total_volume += delta;
        if !order.is_hidden {
            self.visible_volume += delta;
        }
        Ok(old)
    }

    /// Price accessor.
    pub fn price(&self) -> f64 { self.price }

    /// Side accessor.
    pub fn side(&self) -> LevelSide { self.side }

    /// Total volume at this level.
    pub fn total_volume(&self) -> f64 { self.total_volume }

    /// Visible volume (excludes hidden orders).
    pub fn visible_volume(&self) -> f64 { self.visible_volume }

    /// Hidden volume.
    pub fn hidden_volume(&self) -> f64 { self.total_volume - self.visible_volume }

    /// Number of orders at this level.
    pub fn order_count(&self) -> usize { self.orders.len() }

    /// Whether the level is empty.
    pub fn is_empty(&self) -> bool { self.orders.is_empty() }

    /// Notional value (price * total volume).
    pub fn notional(&self) -> f64 { self.price * self.total_volume }

    /// Front-of-queue order id.
    pub fn front_order_id(&self) -> Option<u64> {
        self.orders.first().map(|o| o.order_id)
    }

    /// Queue position of a given order (0-indexed).
    pub fn queue_position(&self, order_id: u64) -> Option<usize> {
        self.orders.iter().position(|o| o.order_id == order_id)
    }

    /// Volume ahead of a given order.
    pub fn volume_ahead(&self, order_id: u64) -> Option<f64> {
        let pos = self.queue_position(order_id)?;
        Some(self.orders[..pos].iter().map(|o| o.remaining).sum())
    }

    /// Compute level statistics.
    pub fn stats(&self) -> LevelStats {
        let sizes: Vec<f64> = self.orders.iter().map(|o| o.remaining).collect();
        LevelStats {
            price: self.price,
            side: self.side,
            order_count: self.orders.len(),
            total_volume: self.total_volume,
            visible_volume: self.visible_volume,
            hidden_volume: self.hidden_volume(),
            avg_order_size: if sizes.is_empty() { 0.0 }
                else { sizes.iter().sum::<f64>() / sizes.len() as f64 },
            max_order_size: sizes.iter().cloned().fold(0.0_f64, f64::max),
            min_order_size: sizes.iter().cloned().fold(f64::INFINITY, f64::min),
            notional: self.notional(),
        }
    }

    /// Iterate over orders in queue order.
    pub fn orders(&self) -> &[QueuedOrder] { &self.orders }
}

impl fmt::Display for PriceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PriceLevel({} {:.2}: vol={:.4}, orders={})",
            self.side, self.price, self.total_volume, self.orders.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_level() -> PriceLevel {
        let mut lvl = PriceLevel::new(100.0, LevelSide::Bid).unwrap();
        lvl.add_order(QueuedOrder::new(1, 50.0, 1000)).unwrap();
        lvl.add_order(QueuedOrder::new(2, 30.0, 2000)).unwrap();
        lvl.add_order(QueuedOrder::new(3, 20.0, 3000)).unwrap();
        lvl
    }

    #[test]
    fn test_new_level() {
        let lvl = PriceLevel::new(100.0, LevelSide::Ask).unwrap();
        assert!(lvl.is_empty());
        assert!((lvl.total_volume() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_price() {
        assert!(PriceLevel::new(-1.0, LevelSide::Bid).is_err());
        assert!(PriceLevel::new(0.0, LevelSide::Bid).is_err());
    }

    #[test]
    fn test_add_order() {
        let lvl = make_level();
        assert_eq!(lvl.order_count(), 3);
        assert!((lvl.total_volume() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_duplicate() {
        let mut lvl = make_level();
        let result = lvl.add_order(QueuedOrder::new(1, 10.0, 9000));
        assert!(result.is_err());
    }

    #[test]
    fn test_add_zero_qty() {
        let mut lvl = PriceLevel::new(100.0, LevelSide::Bid).unwrap();
        assert!(lvl.add_order(QueuedOrder::new(1, 0.0, 1000)).is_err());
    }

    #[test]
    fn test_remove_order() {
        let mut lvl = make_level();
        let removed = lvl.remove_order(2).unwrap();
        assert_eq!(removed.order_id, 2);
        assert_eq!(lvl.order_count(), 2);
        assert!((lvl.total_volume() - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_remove_not_found() {
        let mut lvl = make_level();
        assert!(lvl.remove_order(999).is_err());
    }

    #[test]
    fn test_fill_front() {
        let mut lvl = make_level();
        let (id, filled) = lvl.fill_front(30.0).unwrap();
        assert_eq!(id, 1);
        assert!((filled - 30.0).abs() < 1e-6);
        assert!((lvl.total_volume() - 70.0).abs() < 1e-6);
        // front still has 20 remaining
        assert_eq!(lvl.front_order_id(), Some(1));
    }

    #[test]
    fn test_fill_front_removes() {
        let mut lvl = make_level();
        let (id, _) = lvl.fill_front(50.0).unwrap();
        assert_eq!(id, 1);
        assert_eq!(lvl.front_order_id(), Some(2));
        assert_eq!(lvl.order_count(), 2);
    }

    #[test]
    fn test_fill_order_specific() {
        let mut lvl = make_level();
        let filled = lvl.fill_order(2, 10.0).unwrap();
        assert!((filled - 10.0).abs() < 1e-6);
        assert!((lvl.total_volume() - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_modify_order() {
        let mut lvl = make_level();
        let old = lvl.modify_order(2, 50.0).unwrap();
        assert!((old - 30.0).abs() < 1e-6);
        assert!((lvl.total_volume() - 120.0).abs() < 1e-6);
    }

    #[test]
    fn test_modify_invalid() {
        let mut lvl = make_level();
        assert!(lvl.modify_order(2, 0.0).is_err());
    }

    #[test]
    fn test_queue_position() {
        let lvl = make_level();
        assert_eq!(lvl.queue_position(1), Some(0));
        assert_eq!(lvl.queue_position(2), Some(1));
        assert_eq!(lvl.queue_position(3), Some(2));
        assert_eq!(lvl.queue_position(99), None);
    }

    #[test]
    fn test_volume_ahead() {
        let lvl = make_level();
        assert!((lvl.volume_ahead(1).unwrap() - 0.0).abs() < 1e-6);
        assert!((lvl.volume_ahead(2).unwrap() - 50.0).abs() < 1e-6);
        assert!((lvl.volume_ahead(3).unwrap() - 80.0).abs() < 1e-6);
    }

    #[test]
    fn test_hidden_volume() {
        let mut lvl = PriceLevel::new(100.0, LevelSide::Ask).unwrap();
        lvl.add_order(QueuedOrder::new(1, 50.0, 1000)).unwrap();
        lvl.add_order(QueuedOrder::new(2, 30.0, 2000).with_hidden(true)).unwrap();
        assert!((lvl.visible_volume() - 50.0).abs() < 1e-6);
        assert!((lvl.hidden_volume() - 30.0).abs() < 1e-6);
        assert!((lvl.total_volume() - 80.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats() {
        let lvl = make_level();
        let s = lvl.stats();
        assert_eq!(s.order_count, 3);
        assert!((s.total_volume - 100.0).abs() < 1e-6);
        assert!((s.max_order_size - 50.0).abs() < 1e-6);
        assert!((s.min_order_size - 20.0).abs() < 1e-6);
        assert!((s.notional - 10000.0).abs() < 1e-2);
    }

    #[test]
    fn test_notional() {
        let lvl = make_level();
        assert!((lvl.notional() - 10000.0).abs() < 1e-2);
    }

    #[test]
    fn test_display() {
        let lvl = make_level();
        let s = format!("{lvl}");
        assert!(s.contains("BID"));
        assert!(s.contains("100.00"));
    }

    #[test]
    fn test_queued_order_display() {
        let o = QueuedOrder::new(5, 100.0, 1000).with_hidden(true);
        let s = format!("{o}");
        assert!(s.contains("hidden"));
    }

    #[test]
    fn test_stats_display() {
        let lvl = make_level();
        let s = format!("{}", lvl.stats());
        assert!(s.contains("BID"));
        assert!(s.contains("100.00"));
    }

    #[test]
    fn test_fill_ratio() {
        let mut o = QueuedOrder::new(1, 100.0, 1000);
        o.remaining = 25.0;
        assert!((o.fill_ratio() - 0.75).abs() < 1e-6);
    }
}
