//! Order Amendment — price/quantity modification, amendment validation,
//! queue priority rules, and amendment audit trail.
//!
//! Pure-Rust order amendment engine supporting price and quantity
//! modifications with configurable priority rules (lose-priority on
//! price change, keep-priority on quantity decrease), validation, and
//! a full audit log.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AmendError {
    OrderNotFound(u64),
    NotAmendable(String),
    InvalidPrice(String),
    InvalidQuantity(String),
    NoChange(u64),
    QuantityBelowFilled(String),
    AmendLimitExceeded(String),
}

impl fmt::Display for AmendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrderNotFound(id) => write!(f, "order not found: {id}"),
            Self::NotAmendable(s) => write!(f, "not amendable: {s}"),
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
            Self::InvalidQuantity(s) => write!(f, "invalid quantity: {s}"),
            Self::NoChange(id) => write!(f, "no change for order: {id}"),
            Self::QuantityBelowFilled(s) => write!(f, "quantity below filled: {s}"),
            Self::AmendLimitExceeded(s) => write!(f, "amend limit exceeded: {s}"),
        }
    }
}

impl std::error::Error for AmendError {}

// ── AmendableState ──────────────────────────────────────────────

/// Order states that allow amendment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmendableState {
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
}

impl AmendableState {
    pub fn can_amend(self) -> bool {
        matches!(self, Self::Open | Self::PartiallyFilled)
    }
}

impl fmt::Display for AmendableState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "OPEN"),
            Self::PartiallyFilled => write!(f, "PARTIAL"),
            Self::Filled => write!(f, "FILLED"),
            Self::Cancelled => write!(f, "CANCELLED"),
        }
    }
}

// ── PriorityAction ──────────────────────────────────────────────

/// What happens to queue priority after an amendment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityAction {
    /// Order keeps its position in the queue.
    Keep,
    /// Order loses priority and goes to back of queue.
    Lose,
}

impl fmt::Display for PriorityAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keep => write!(f, "KEEP"),
            Self::Lose => write!(f, "LOSE"),
        }
    }
}

// ── PriorityRules ───────────────────────────────────────────────

/// Configurable rules for when priority is lost.
#[derive(Debug, Clone)]
pub struct PriorityRules {
    pub lose_on_price_change: bool,
    pub lose_on_quantity_increase: bool,
    pub lose_on_quantity_decrease: bool,
}

impl PriorityRules {
    /// Default rules: lose on price change and qty increase, keep on qty decrease.
    pub fn default_rules() -> Self {
        Self {
            lose_on_price_change: true,
            lose_on_quantity_increase: true,
            lose_on_quantity_decrease: false,
        }
    }

    pub fn with_lose_on_price_change(mut self, v: bool) -> Self {
        self.lose_on_price_change = v;
        self
    }

    pub fn with_lose_on_quantity_increase(mut self, v: bool) -> Self {
        self.lose_on_quantity_increase = v;
        self
    }

    pub fn with_lose_on_quantity_decrease(mut self, v: bool) -> Self {
        self.lose_on_quantity_decrease = v;
        self
    }

    /// Determine priority action for a given amendment.
    pub fn evaluate(&self, old_price: f64, new_price: f64,
                    old_qty: f64, new_qty: f64) -> PriorityAction
    {
        if self.lose_on_price_change && (old_price - new_price).abs() > 1e-12 {
            return PriorityAction::Lose;
        }
        if self.lose_on_quantity_increase && new_qty > old_qty + 1e-12 {
            return PriorityAction::Lose;
        }
        if self.lose_on_quantity_decrease && new_qty < old_qty - 1e-12 {
            return PriorityAction::Lose;
        }
        PriorityAction::Keep
    }
}

impl fmt::Display for PriorityRules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PriorityRules(price={}, qty_up={}, qty_down={})",
            self.lose_on_price_change, self.lose_on_quantity_increase,
            self.lose_on_quantity_decrease)
    }
}

// ── AmendRequest ────────────────────────────────────────────────

/// A request to amend an order's price and/or quantity.
#[derive(Debug, Clone)]
pub struct AmendRequest {
    pub order_id: u64,
    pub new_price: Option<f64>,
    pub new_quantity: Option<f64>,
    pub timestamp_ns: u64,
    pub client_amend_id: Option<String>,
}

impl AmendRequest {
    pub fn new(order_id: u64, timestamp_ns: u64) -> Self {
        Self { order_id, new_price: None, new_quantity: None, timestamp_ns, client_amend_id: None }
    }

    pub fn with_new_price(mut self, price: f64) -> Self {
        self.new_price = Some(price);
        self
    }

    pub fn with_new_quantity(mut self, qty: f64) -> Self {
        self.new_quantity = Some(qty);
        self
    }

    pub fn with_client_amend_id(mut self, id: &str) -> Self {
        self.client_amend_id = Some(id.to_string());
        self
    }

    /// Whether this request changes the price.
    pub fn changes_price(&self) -> bool { self.new_price.is_some() }

    /// Whether this request changes the quantity.
    pub fn changes_quantity(&self) -> bool { self.new_quantity.is_some() }
}

impl fmt::Display for AmendRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let px = self.new_price.map_or("--".into(), |p| format!("{p:.2}"));
        let qty = self.new_quantity.map_or("--".into(), |q| format!("{q:.4}"));
        write!(f, "AmendReq(order={}, px={px}, qty={qty})", self.order_id)
    }
}

// ── AmendResult ─────────────────────────────────────────────────

/// The outcome of a successful amendment.
#[derive(Debug, Clone)]
pub struct AmendResult {
    pub amend_id: u64,
    pub order_id: u64,
    pub old_price: f64,
    pub new_price: f64,
    pub old_quantity: f64,
    pub new_quantity: f64,
    pub priority_action: PriorityAction,
    pub timestamp_ns: u64,
}

impl AmendResult {
    /// Price changed?
    pub fn price_changed(&self) -> bool {
        (self.old_price - self.new_price).abs() > 1e-12
    }

    /// Quantity changed?
    pub fn quantity_changed(&self) -> bool {
        (self.old_quantity - self.new_quantity).abs() > 1e-12
    }

    /// Quantity delta (positive = increase).
    pub fn quantity_delta(&self) -> f64 {
        self.new_quantity - self.old_quantity
    }

    /// Price delta.
    pub fn price_delta(&self) -> f64 {
        self.new_price - self.old_price
    }
}

impl fmt::Display for AmendResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AmendResult(id={}, order={}, px={:.2}->{:.2}, qty={:.4}->{:.4}, priority={})",
            self.amend_id, self.order_id,
            self.old_price, self.new_price,
            self.old_quantity, self.new_quantity,
            self.priority_action)
    }
}

// ── OrderAmendEngine ────────────────────────────────────────────

/// Order amendment engine with validation, priority rules, and audit trail.
#[derive(Debug, Clone)]
pub struct OrderAmendEngine {
    rules: PriorityRules,
    events: Vec<AmendResult>,
    next_amend_id: u64,
    max_amends_per_order: usize,
}

impl OrderAmendEngine {
    pub fn new() -> Self {
        Self {
            rules: PriorityRules::default_rules(),
            events: Vec::new(),
            next_amend_id: 1,
            max_amends_per_order: 100,
        }
    }

    pub fn with_rules(mut self, rules: PriorityRules) -> Self {
        self.rules = rules;
        self
    }

    pub fn with_max_amends_per_order(mut self, n: usize) -> Self {
        self.max_amends_per_order = n;
        self
    }

    /// Validate an amendment request against order state.
    pub fn validate(&self, req: &AmendRequest, state: AmendableState,
                    current_price: f64, current_qty: f64, filled_qty: f64)
        -> Result<(), AmendError>
    {
        if !state.can_amend() {
            return Err(AmendError::NotAmendable(format!("state={state}")));
        }
        if req.new_price.is_none() && req.new_quantity.is_none() {
            return Err(AmendError::NoChange(req.order_id));
        }
        if let Some(p) = req.new_price {
            if p <= 0.0 {
                return Err(AmendError::InvalidPrice(format!("{p}")));
            }
        }
        if let Some(q) = req.new_quantity {
            if q <= 0.0 {
                return Err(AmendError::InvalidQuantity(format!("{q}")));
            }
            if q < filled_qty {
                return Err(AmendError::QuantityBelowFilled(
                    format!("new_qty={q} < filled={filled_qty}")));
            }
        }
        // Check if anything actually changes
        let same_price = req.new_price.map_or(true, |p| (p - current_price).abs() < 1e-12);
        let same_qty = req.new_quantity.map_or(true, |q| (q - current_qty).abs() < 1e-12);
        if same_price && same_qty {
            return Err(AmendError::NoChange(req.order_id));
        }
        // Check amend count limit
        let count = self.amend_count_for(req.order_id);
        if count >= self.max_amends_per_order {
            return Err(AmendError::AmendLimitExceeded(
                format!("order {} has {count} amends", req.order_id)));
        }
        Ok(())
    }

    /// Apply an amendment.
    pub fn amend(&mut self, req: &AmendRequest, state: AmendableState,
                 current_price: f64, current_qty: f64, filled_qty: f64)
        -> Result<AmendResult, AmendError>
    {
        self.validate(req, state, current_price, current_qty, filled_qty)?;

        let new_price = req.new_price.unwrap_or(current_price);
        let new_qty = req.new_quantity.unwrap_or(current_qty);
        let priority = self.rules.evaluate(current_price, new_price, current_qty, new_qty);

        let result = AmendResult {
            amend_id: self.next_amend_id,
            order_id: req.order_id,
            old_price: current_price,
            new_price,
            old_quantity: current_qty,
            new_quantity: new_qty,
            priority_action: priority,
            timestamp_ns: req.timestamp_ns,
        };
        self.next_amend_id += 1;
        self.events.push(result.clone());
        Ok(result)
    }

    /// Number of amendments for a specific order.
    pub fn amend_count_for(&self, order_id: u64) -> usize {
        self.events.iter().filter(|e| e.order_id == order_id).count()
    }

    /// All amendment events.
    pub fn events(&self) -> &[AmendResult] { &self.events }

    /// Total amendment count.
    pub fn total_amends(&self) -> usize { self.events.len() }

    /// Events for a specific order.
    pub fn events_for_order(&self, order_id: u64) -> Vec<&AmendResult> {
        self.events.iter().filter(|e| e.order_id == order_id).collect()
    }

    /// Count of amendments that caused priority loss.
    pub fn priority_loss_count(&self) -> usize {
        self.events.iter().filter(|e| e.priority_action == PriorityAction::Lose).count()
    }

    /// Rules accessor.
    pub fn rules(&self) -> &PriorityRules { &self.rules }
}

impl fmt::Display for OrderAmendEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AmendEngine(amends={}, priority_losses={})",
            self.events.len(), self.priority_loss_count())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> OrderAmendEngine {
        OrderAmendEngine::new()
    }

    #[test]
    fn test_amend_price() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_price(105.0);
        let result = eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert!(result.price_changed());
        assert!(!result.quantity_changed());
        assert_eq!(result.priority_action, PriorityAction::Lose);
    }

    #[test]
    fn test_amend_quantity_decrease() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_quantity(30.0);
        let result = eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert!(!result.price_changed());
        assert!(result.quantity_changed());
        assert_eq!(result.priority_action, PriorityAction::Keep);
    }

    #[test]
    fn test_amend_quantity_increase() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_quantity(80.0);
        let result = eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert_eq!(result.priority_action, PriorityAction::Lose);
    }

    #[test]
    fn test_amend_both() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000)
            .with_new_price(101.0)
            .with_new_quantity(60.0);
        let result = eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert!(result.price_changed());
        assert!(result.quantity_changed());
    }

    #[test]
    fn test_amend_filled_state() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_price(105.0);
        assert!(eng.amend(&req, AmendableState::Filled, 100.0, 50.0, 50.0).is_err());
    }

    #[test]
    fn test_amend_cancelled_state() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_price(105.0);
        assert!(eng.amend(&req, AmendableState::Cancelled, 100.0, 50.0, 0.0).is_err());
    }

    #[test]
    fn test_amend_no_change() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_price(100.0);
        assert!(eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).is_err());
    }

    #[test]
    fn test_amend_invalid_price() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_price(-5.0);
        assert!(eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).is_err());
    }

    #[test]
    fn test_amend_invalid_quantity() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_quantity(0.0);
        assert!(eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).is_err());
    }

    #[test]
    fn test_amend_below_filled() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_quantity(20.0);
        assert!(eng.amend(&req, AmendableState::PartiallyFilled, 100.0, 50.0, 30.0).is_err());
    }

    #[test]
    fn test_amend_partial_fill_state() {
        let mut eng = engine();
        let req = AmendRequest::new(1, 1000).with_new_quantity(40.0);
        let result = eng.amend(&req, AmendableState::PartiallyFilled, 100.0, 50.0, 30.0).unwrap();
        assert!((result.quantity_delta() - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_amend_limit() {
        let mut eng = engine().with_max_amends_per_order(2);
        let req1 = AmendRequest::new(1, 1000).with_new_price(101.0);
        eng.amend(&req1, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        let req2 = AmendRequest::new(1, 2000).with_new_price(102.0);
        eng.amend(&req2, AmendableState::Open, 101.0, 50.0, 0.0).unwrap();
        let req3 = AmendRequest::new(1, 3000).with_new_price(103.0);
        assert!(eng.amend(&req3, AmendableState::Open, 102.0, 50.0, 0.0).is_err());
    }

    #[test]
    fn test_events_for_order() {
        let mut eng = engine();
        let req1 = AmendRequest::new(1, 1000).with_new_price(101.0);
        eng.amend(&req1, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        let req2 = AmendRequest::new(2, 2000).with_new_price(99.0);
        eng.amend(&req2, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert_eq!(eng.events_for_order(1).len(), 1);
        assert_eq!(eng.events_for_order(2).len(), 1);
    }

    #[test]
    fn test_priority_loss_count() {
        let mut eng = engine();
        // Price change => lose
        let r1 = AmendRequest::new(1, 1000).with_new_price(101.0);
        eng.amend(&r1, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        // Qty decrease => keep
        let r2 = AmendRequest::new(2, 2000).with_new_quantity(30.0);
        eng.amend(&r2, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert_eq!(eng.priority_loss_count(), 1);
    }

    #[test]
    fn test_custom_rules() {
        let rules = PriorityRules::default_rules()
            .with_lose_on_quantity_decrease(true);
        let mut eng = engine().with_rules(rules);
        let req = AmendRequest::new(1, 1000).with_new_quantity(30.0);
        let result = eng.amend(&req, AmendableState::Open, 100.0, 50.0, 0.0).unwrap();
        assert_eq!(result.priority_action, PriorityAction::Lose);
    }

    #[test]
    fn test_display_engine() {
        let eng = engine();
        let s = format!("{eng}");
        assert!(s.contains("amends=0"));
    }

    #[test]
    fn test_display_result() {
        let r = AmendResult {
            amend_id: 1, order_id: 42,
            old_price: 100.0, new_price: 101.0,
            old_quantity: 50.0, new_quantity: 60.0,
            priority_action: PriorityAction::Lose,
            timestamp_ns: 1000,
        };
        let s = format!("{r}");
        assert!(s.contains("42"));
        assert!(s.contains("LOSE"));
    }

    #[test]
    fn test_display_request() {
        let req = AmendRequest::new(7, 1000)
            .with_new_price(105.0)
            .with_client_amend_id("am-1");
        let s = format!("{req}");
        assert!(s.contains("105.00"));
    }

    #[test]
    fn test_priority_rules_display() {
        let r = PriorityRules::default_rules();
        let s = format!("{r}");
        assert!(s.contains("price=true"));
    }
}
