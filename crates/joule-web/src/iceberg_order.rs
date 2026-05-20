//! # Iceberg / Hidden Orders
//!
//! Implements iceberg order management with configurable display quantity,
//! automatic replenishment logic, randomised display sizing, and fill
//! tracking. Supports both fixed and proportional replenishment strategies
//! with information leakage minimisation through display jitter.

use std::fmt;

// ── Core Types ──

/// Replenishment strategy for iceberg display quantity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplenishStrategy {
    /// Fixed display quantity on every refill.
    Fixed,
    /// Random display quantity within a range.
    Randomised,
    /// Proportional to remaining hidden quantity.
    Proportional,
    /// Decaying display size as order fills.
    Decaying,
}

impl fmt::Display for ReplenishStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ReplenishStrategy::Fixed => "FIXED",
            ReplenishStrategy::Randomised => "RANDOM",
            ReplenishStrategy::Proportional => "PROPORTIONAL",
            ReplenishStrategy::Decaying => "DECAYING",
        };
        write!(f, "{s}")
    }
}

/// Side of the iceberg order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcebergSide {
    Buy,
    Sell,
}

impl fmt::Display for IcebergSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IcebergSide::Buy => write!(f, "BUY"),
            IcebergSide::Sell => write!(f, "SELL"),
        }
    }
}

// ── Iceberg Fill ──

/// Record of a single fill against the iceberg.
#[derive(Clone, Debug)]
pub struct IcebergFill {
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub display_was: f64,
    pub replenish_seq: u32,
}

impl IcebergFill {
    pub fn new(price: f64, quantity: f64, timestamp_ns: u64) -> Self {
        Self {
            price,
            quantity,
            timestamp_ns,
            display_was: 0.0,
            replenish_seq: 0,
        }
    }

    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

impl fmt::Display for IcebergFill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IcebergFill({:.4}@{:.2} display_was={:.2} seq={})",
            self.quantity, self.price, self.display_was, self.replenish_seq,
        )
    }
}

// ── Display Configuration ──

/// Configuration for display quantity behaviour.
#[derive(Clone, Debug)]
pub struct DisplayConfig {
    pub base_display_qty: f64,
    pub min_display_qty: f64,
    pub max_display_qty: f64,
    pub strategy: ReplenishStrategy,
    pub decay_rate: f64,
    pub jitter_seed: u64,
}

impl DisplayConfig {
    pub fn new(display_qty: f64) -> Self {
        Self {
            base_display_qty: display_qty,
            min_display_qty: display_qty,
            max_display_qty: display_qty,
            strategy: ReplenishStrategy::Fixed,
            decay_rate: 0.95,
            jitter_seed: 42,
        }
    }

    pub fn with_strategy(mut self, strategy: ReplenishStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_display_qty = min;
        self.max_display_qty = max;
        self
    }

    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_jitter_seed(mut self, seed: u64) -> Self {
        self.jitter_seed = seed;
        self
    }

    /// Compute next display quantity.
    pub fn next_display(&self, remaining: f64, replenish_count: u32) -> f64 {
        let raw = match self.strategy {
            ReplenishStrategy::Fixed => self.base_display_qty,
            ReplenishStrategy::Randomised => {
                let hash = self.simple_hash(replenish_count);
                let range = self.max_display_qty - self.min_display_qty;
                self.min_display_qty + range * hash
            }
            ReplenishStrategy::Proportional => {
                remaining * (self.base_display_qty / (remaining + self.base_display_qty))
            }
            ReplenishStrategy::Decaying => {
                self.base_display_qty * self.decay_rate.powi(replenish_count as i32)
            }
        };
        raw.min(remaining).max(0.0)
    }

    /// Deterministic pseudo-random in [0, 1) from seed + count.
    fn simple_hash(&self, count: u32) -> f64 {
        let mut x = self.jitter_seed.wrapping_add(count as u64);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl fmt::Display for DisplayConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DisplayConfig(base={:.2} range=[{:.2},{:.2}] strategy={})",
            self.base_display_qty, self.min_display_qty,
            self.max_display_qty, self.strategy,
        )
    }
}

// ── Iceberg Order ──

/// A complete iceberg order with hidden and displayed components.
#[derive(Clone, Debug)]
pub struct IcebergOrder {
    pub id: u64,
    pub symbol: String,
    pub side: IcebergSide,
    pub total_quantity: f64,
    pub filled_quantity: f64,
    pub display_quantity: f64,
    pub limit_price: f64,
    pub display_config: DisplayConfig,
    pub replenish_count: u32,
    pub fills: Vec<IcebergFill>,
    pub is_active: bool,
    pub created_ns: u64,
}

impl IcebergOrder {
    pub fn new(id: u64, symbol: &str, side: IcebergSide,
               total_qty: f64, display_qty: f64, limit_price: f64) -> Self {
        let config = DisplayConfig::new(display_qty);
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            total_quantity: total_qty,
            filled_quantity: 0.0,
            display_quantity: display_qty.min(total_qty),
            limit_price,
            display_config: config,
            replenish_count: 0,
            fills: Vec::new(),
            is_active: true,
            created_ns: 0,
        }
    }

    pub fn with_display_config(mut self, config: DisplayConfig) -> Self {
        self.display_config = config;
        self.display_quantity = self.display_config.next_display(
            self.hidden_quantity() + self.display_quantity,
            0,
        );
        self
    }

    pub fn with_created(mut self, ts: u64) -> Self {
        self.created_ns = ts;
        self
    }

    /// Hidden (non-displayed) quantity.
    pub fn hidden_quantity(&self) -> f64 {
        (self.total_quantity - self.filled_quantity - self.display_quantity).max(0.0)
    }

    /// Total remaining quantity (display + hidden).
    pub fn remaining(&self) -> f64 {
        (self.total_quantity - self.filled_quantity).max(0.0)
    }

    /// Completion percentage.
    pub fn completion_pct(&self) -> f64 {
        if self.total_quantity > 1e-12 {
            (self.filled_quantity / self.total_quantity) * 100.0
        } else {
            0.0
        }
    }

    /// Average fill price.
    pub fn avg_fill_price(&self) -> f64 {
        if self.filled_quantity < 1e-12 { return 0.0; }
        let notional: f64 = self.fills.iter().map(|f| f.notional()).sum();
        notional / self.filled_quantity
    }

    /// Process an incoming fill against the displayed quantity.
    pub fn process_fill(&mut self, price: f64, qty: f64, timestamp_ns: u64) -> bool {
        if !self.is_active || qty < 1e-12 { return false; }

        let fill_qty = qty.min(self.display_quantity);
        if fill_qty < 1e-12 { return false; }

        let mut fill = IcebergFill::new(price, fill_qty, timestamp_ns);
        fill.display_was = self.display_quantity;
        fill.replenish_seq = self.replenish_count;

        self.filled_quantity += fill_qty;
        self.display_quantity -= fill_qty;
        self.fills.push(fill);

        // Check if fully filled.
        if self.remaining() < 1e-12 {
            self.is_active = false;
            self.display_quantity = 0.0;
            return true;
        }

        // Replenish display if exhausted.
        if self.display_quantity < 1e-12 && self.hidden_quantity() > 1e-12 {
            self.replenish();
        }

        true
    }

    /// Replenish the display quantity from the hidden reserve.
    fn replenish(&mut self) {
        self.replenish_count += 1;
        let new_display = self.display_config.next_display(
            self.remaining(),
            self.replenish_count,
        );
        self.display_quantity = new_display.min(self.remaining());
    }

    /// Cancel the iceberg order.
    pub fn cancel(&mut self) {
        self.is_active = false;
    }

    /// Iceberg ratio: total / display.
    pub fn iceberg_ratio(&self) -> f64 {
        if self.display_config.base_display_qty > 1e-12 {
            self.total_quantity / self.display_config.base_display_qty
        } else {
            0.0
        }
    }

    /// Information leakage score: higher means more detectable.
    pub fn leakage_score(&self) -> f64 {
        if self.fills.len() < 2 { return 0.0; }
        let display_sizes: Vec<f64> = self.fills.iter().map(|f| f.display_was).collect();
        let mean = display_sizes.iter().sum::<f64>() / display_sizes.len() as f64;
        let variance = display_sizes.iter()
            .map(|d| { let diff = d - mean; diff * diff })
            .sum::<f64>() / display_sizes.len() as f64;
        let cv = if mean > 1e-12 { variance.sqrt() / mean } else { 0.0 };
        // Low CV means predictable display sizes → higher leakage.
        1.0 - cv.min(1.0)
    }
}

impl fmt::Display for IcebergOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Iceberg(#{} {} {} total={:.2} display={:.2} hidden={:.2} filled={:.1}% replenish={})",
            self.id, self.side, self.symbol, self.total_quantity,
            self.display_quantity, self.hidden_quantity(),
            self.completion_pct(), self.replenish_count,
        )
    }
}

// ── Iceberg Manager ──

/// Manages multiple iceberg orders.
#[derive(Clone, Debug)]
pub struct IcebergManager {
    pub orders: Vec<IcebergOrder>,
    next_id: u64,
}

impl IcebergManager {
    pub fn new() -> Self {
        Self { orders: Vec::new(), next_id: 1 }
    }

    pub fn submit(&mut self, mut order: IcebergOrder) -> u64 {
        let id = self.next_id;
        order.id = id;
        self.next_id += 1;
        self.orders.push(order);
        id
    }

    pub fn get(&self, id: u64) -> Option<&IcebergOrder> {
        self.orders.iter().find(|o| o.id == id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut IcebergOrder> {
        self.orders.iter_mut().find(|o| o.id == id)
    }

    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(o) = self.get_mut(id) {
            o.cancel();
            true
        } else {
            false
        }
    }

    pub fn active_count(&self) -> usize {
        self.orders.iter().filter(|o| o.is_active).count()
    }

    pub fn total_displayed(&self) -> f64 {
        self.orders.iter()
            .filter(|o| o.is_active)
            .map(|o| o.display_quantity)
            .sum()
    }

    pub fn total_hidden(&self) -> f64 {
        self.orders.iter()
            .filter(|o| o.is_active)
            .map(|o| o.hidden_quantity())
            .sum()
    }
}

impl fmt::Display for IcebergManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IcebergMgr(active={} displayed={:.2} hidden={:.2})",
            self.active_count(), self.total_displayed(), self.total_hidden(),
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_order() -> IcebergOrder {
        IcebergOrder::new(1, "AAPL", IcebergSide::Buy, 1000.0, 100.0, 150.0)
    }

    #[test]
    fn test_basic_creation() {
        let o = make_order();
        assert!((o.total_quantity - 1000.0).abs() < 1e-9);
        assert!((o.display_quantity - 100.0).abs() < 1e-9);
        assert!((o.hidden_quantity() - 900.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_fill() {
        let mut o = make_order();
        assert!(o.process_fill(150.0, 50.0, 100));
        assert!((o.filled_quantity - 50.0).abs() < 1e-9);
        assert!((o.display_quantity - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_replenish() {
        let mut o = make_order();
        // Fill entire display.
        o.process_fill(150.0, 100.0, 100);
        // Should have replenished.
        assert!(o.display_quantity > 0.0);
        assert_eq!(o.replenish_count, 1);
    }

    #[test]
    fn test_full_fill() {
        let mut o = IcebergOrder::new(1, "AAPL", IcebergSide::Buy, 200.0, 100.0, 150.0);
        o.process_fill(150.0, 100.0, 100);
        o.process_fill(150.0, 100.0, 200);
        assert!(!o.is_active);
        assert!((o.completion_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_fill_price() {
        let mut o = make_order();
        o.process_fill(150.0, 100.0, 100);
        o.process_fill(151.0, 100.0, 200);
        let expected = (150.0 * 100.0 + 151.0 * 100.0) / 200.0;
        assert!((o.avg_fill_price() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_iceberg_ratio() {
        let o = make_order();
        assert!((o.iceberg_ratio() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_cancel() {
        let mut o = make_order();
        o.cancel();
        assert!(!o.is_active);
    }

    #[test]
    fn test_display_config_fixed() {
        let cfg = DisplayConfig::new(100.0);
        assert!((cfg.next_display(500.0, 0) - 100.0).abs() < 1e-9);
        assert!((cfg.next_display(500.0, 5) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_config_decaying() {
        let cfg = DisplayConfig::new(100.0)
            .with_strategy(ReplenishStrategy::Decaying)
            .with_decay_rate(0.50);
        let d0 = cfg.next_display(1000.0, 0);
        let d1 = cfg.next_display(1000.0, 1);
        let d2 = cfg.next_display(1000.0, 2);
        assert!((d0 - 100.0).abs() < 1e-9);
        assert!((d1 - 50.0).abs() < 1e-9);
        assert!((d2 - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_config_randomised() {
        let cfg = DisplayConfig::new(100.0)
            .with_strategy(ReplenishStrategy::Randomised)
            .with_range(50.0, 150.0)
            .with_jitter_seed(12345);
        let d1 = cfg.next_display(1000.0, 0);
        let d2 = cfg.next_display(1000.0, 1);
        assert!(d1 >= 50.0 && d1 <= 150.0);
        assert!(d2 >= 50.0 && d2 <= 150.0);
    }

    #[test]
    fn test_display_capped_by_remaining() {
        let cfg = DisplayConfig::new(100.0);
        let d = cfg.next_display(30.0, 0);
        assert!((d - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_manager_submit() {
        let mut mgr = IcebergManager::new();
        let o = IcebergOrder::new(0, "AAPL", IcebergSide::Buy, 500.0, 50.0, 150.0);
        let id = mgr.submit(o);
        assert_eq!(id, 1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_manager_cancel() {
        let mut mgr = IcebergManager::new();
        let o = IcebergOrder::new(0, "AAPL", IcebergSide::Buy, 500.0, 50.0, 150.0);
        let id = mgr.submit(o);
        assert!(mgr.cancel(id));
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_manager_totals() {
        let mut mgr = IcebergManager::new();
        mgr.submit(IcebergOrder::new(0, "AAPL", IcebergSide::Buy, 500.0, 50.0, 150.0));
        mgr.submit(IcebergOrder::new(0, "MSFT", IcebergSide::Sell, 300.0, 30.0, 200.0));
        assert!((mgr.total_displayed() - 80.0).abs() < 1e-9);
        assert!((mgr.total_hidden() - 720.0).abs() < 1e-9);
    }

    #[test]
    fn test_leakage_score_fixed() {
        let mut o = IcebergOrder::new(1, "AAPL", IcebergSide::Buy, 1000.0, 100.0, 150.0);
        // Fill repeatedly to get multiple replenishments.
        for i in 0..5 {
            o.process_fill(150.0, 100.0, (i + 1) * 100);
        }
        // Fixed strategy → all display sizes same → high leakage.
        let score = o.leakage_score();
        assert!(score > 0.5);
    }

    #[test]
    fn test_order_display() {
        let o = make_order();
        let s = format!("{o}");
        assert!(s.contains("Iceberg"));
        assert!(s.contains("BUY"));
        assert!(s.contains("AAPL"));
    }

    #[test]
    fn test_fill_display() {
        let f = IcebergFill::new(150.0, 50.0, 100);
        let s = format!("{f}");
        assert!(s.contains("IcebergFill"));
    }

    #[test]
    fn test_config_display() {
        let cfg = DisplayConfig::new(100.0);
        let s = format!("{cfg}");
        assert!(s.contains("DisplayConfig"));
    }

    #[test]
    fn test_manager_display() {
        let mgr = IcebergManager::new();
        let s = format!("{mgr}");
        assert!(s.contains("IcebergMgr"));
    }

    #[test]
    fn test_no_fill_when_inactive() {
        let mut o = make_order();
        o.cancel();
        assert!(!o.process_fill(150.0, 50.0, 100));
    }
}
