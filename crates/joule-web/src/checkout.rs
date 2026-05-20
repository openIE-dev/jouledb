//! Checkout flow state machine — step transitions, address validation, order summary.
//!
//! Models a multi-step checkout process with strict forward/backward navigation,
//! address validation, and order confirmation. Pure domain logic, no HTTP.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Checkout domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckoutError {
    /// Invalid step transition.
    InvalidTransition { from: CheckoutStep, to: CheckoutStep },
    /// Missing required address field.
    MissingField(String),
    /// Cart is empty.
    EmptyCart,
    /// Session already complete.
    AlreadyComplete,
    /// Payment failed.
    PaymentFailed(String),
}

impl std::fmt::Display for CheckoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from:?} to {to:?}")
            }
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::EmptyCart => write!(f, "cart is empty"),
            Self::AlreadyComplete => write!(f, "checkout already complete"),
            Self::PaymentFailed(msg) => write!(f, "payment failed: {msg}"),
        }
    }
}

impl std::error::Error for CheckoutError {}

// ── Checkout Step ───────────────────────────────────────────────

/// Steps in the checkout flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckoutStep {
    CartReview,
    ShippingInfo,
    ShippingMethod,
    PaymentInfo,
    Review,
    Processing,
    Complete,
    Failed,
}

impl CheckoutStep {
    /// Ordinal position for ordering steps.
    fn ordinal(&self) -> u8 {
        match self {
            Self::CartReview => 0,
            Self::ShippingInfo => 1,
            Self::ShippingMethod => 2,
            Self::PaymentInfo => 3,
            Self::Review => 4,
            Self::Processing => 5,
            Self::Complete => 6,
            Self::Failed => 7,
        }
    }

    /// Valid next steps from this step.
    fn valid_next(&self) -> &[CheckoutStep] {
        match self {
            Self::CartReview => &[Self::ShippingInfo],
            Self::ShippingInfo => &[Self::CartReview, Self::ShippingMethod],
            Self::ShippingMethod => &[Self::ShippingInfo, Self::PaymentInfo],
            Self::PaymentInfo => &[Self::ShippingMethod, Self::Review],
            Self::Review => &[Self::PaymentInfo, Self::Processing],
            Self::Processing => &[Self::Complete, Self::Failed],
            Self::Complete => &[],
            Self::Failed => &[Self::PaymentInfo, Self::CartReview],
        }
    }

    /// Whether this is a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete)
    }

    /// Whether the user can go back from this step.
    pub fn can_go_back(&self) -> bool {
        !matches!(self, Self::CartReview | Self::Processing | Self::Complete)
    }
}

// ── Address ─────────────────────────────────────────────────────

/// Shipping or billing address.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Address {
    pub name: String,
    pub line1: String,
    pub line2: Option<String>,
    pub city: String,
    pub state: String,
    pub postal_code: String,
    pub country: String,
}

impl Address {
    /// Validate required fields are non-empty.
    pub fn validate(&self) -> Result<(), CheckoutError> {
        let checks = [
            ("name", &self.name),
            ("line1", &self.line1),
            ("city", &self.city),
            ("state", &self.state),
            ("postal_code", &self.postal_code),
            ("country", &self.country),
        ];
        for (field, value) in checks {
            if value.trim().is_empty() {
                return Err(CheckoutError::MissingField(field.to_string()));
            }
        }
        Ok(())
    }
}

// ── Checkout Item ───────────────────────────────────────────────

/// An item in the checkout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckoutItem {
    pub product_id: String,
    pub name: String,
    pub price_cents: i64,
    pub quantity: u32,
}

impl CheckoutItem {
    pub fn line_total(&self) -> i64 {
        self.price_cents * self.quantity as i64
    }
}

// ── Shipping Method ─────────────────────────────────────────────

/// Available shipping methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShippingOption {
    pub id: String,
    pub name: String,
    pub cost_cents: i64,
    pub estimated_days: u32,
}

// ── Order Summary ───────────────────────────────────────────────

/// Computed order summary with all line items and totals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSummary {
    pub subtotal_cents: i64,
    pub shipping_cents: i64,
    pub tax_cents: i64,
    pub discount_cents: i64,
    pub total_cents: i64,
    pub item_count: u32,
}

// ── Order Confirmation ──────────────────────────────────────────

/// Order confirmation after successful checkout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderConfirmation {
    pub order_id: String,
    pub summary: OrderSummary,
    pub shipping_address: Address,
    pub shipping_method: String,
}

// ── Checkout Session ────────────────────────────────────────────

/// A checkout session managing the flow from cart to order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckoutSession {
    pub id: String,
    pub step: CheckoutStep,
    pub items: Vec<CheckoutItem>,
    pub shipping_address: Option<Address>,
    pub billing_address: Option<Address>,
    pub selected_shipping: Option<ShippingOption>,
    pub discount_cents: i64,
    pub tax_rate: f64,
    pub metadata: HashMap<String, String>,
    pub order_confirmation: Option<OrderConfirmation>,
    next_order_seq: u64,
}

impl CheckoutSession {
    /// Create a new checkout session.
    pub fn new(id: impl Into<String>, items: Vec<CheckoutItem>) -> Self {
        Self {
            id: id.into(),
            step: CheckoutStep::CartReview,
            items,
            shipping_address: None,
            billing_address: None,
            selected_shipping: None,
            discount_cents: 0,
            tax_rate: 0.0,
            metadata: HashMap::new(),
            order_confirmation: None,
            next_order_seq: 1000,
        }
    }

    /// Move to the next (or previous) step.
    pub fn go_to(&mut self, target: CheckoutStep) -> Result<(), CheckoutError> {
        if self.step == CheckoutStep::Complete {
            return Err(CheckoutError::AlreadyComplete);
        }
        if self.step.valid_next().contains(&target) {
            self.step = target;
            Ok(())
        } else {
            Err(CheckoutError::InvalidTransition {
                from: self.step,
                to: target,
            })
        }
    }

    /// Advance one step forward (convenience).
    pub fn advance(&mut self) -> Result<(), CheckoutError> {
        let next = match self.step {
            CheckoutStep::CartReview => CheckoutStep::ShippingInfo,
            CheckoutStep::ShippingInfo => CheckoutStep::ShippingMethod,
            CheckoutStep::ShippingMethod => CheckoutStep::PaymentInfo,
            CheckoutStep::PaymentInfo => CheckoutStep::Review,
            CheckoutStep::Review => CheckoutStep::Processing,
            CheckoutStep::Processing => CheckoutStep::Complete,
            CheckoutStep::Complete => return Err(CheckoutError::AlreadyComplete),
            CheckoutStep::Failed => CheckoutStep::PaymentInfo,
        };
        self.go_to(next)
    }

    /// Go back one step.
    pub fn go_back(&mut self) -> Result<(), CheckoutError> {
        let prev = match self.step {
            CheckoutStep::ShippingInfo => CheckoutStep::CartReview,
            CheckoutStep::ShippingMethod => CheckoutStep::ShippingInfo,
            CheckoutStep::PaymentInfo => CheckoutStep::ShippingMethod,
            CheckoutStep::Review => CheckoutStep::PaymentInfo,
            CheckoutStep::Failed => CheckoutStep::PaymentInfo,
            _ => {
                return Err(CheckoutError::InvalidTransition {
                    from: self.step,
                    to: self.step,
                })
            }
        };
        self.go_to(prev)
    }

    /// Set the shipping address (validates required fields).
    pub fn set_shipping_address(&mut self, addr: Address) -> Result<(), CheckoutError> {
        addr.validate()?;
        self.shipping_address = Some(addr);
        Ok(())
    }

    /// Set the billing address.
    pub fn set_billing_address(&mut self, addr: Address) -> Result<(), CheckoutError> {
        addr.validate()?;
        self.billing_address = Some(addr);
        Ok(())
    }

    /// Select a shipping method.
    pub fn set_shipping_method(&mut self, method: ShippingOption) {
        self.selected_shipping = Some(method);
    }

    /// Apply a discount.
    pub fn apply_discount(&mut self, cents: i64) {
        self.discount_cents = cents;
    }

    /// Set the tax rate (e.g., 0.08 for 8%).
    pub fn set_tax_rate(&mut self, rate: f64) {
        self.tax_rate = rate;
    }

    /// Compute the order summary.
    pub fn summary(&self) -> OrderSummary {
        let subtotal: i64 = self.items.iter().map(|i| i.line_total()).sum();
        let shipping = self
            .selected_shipping
            .as_ref()
            .map_or(0, |s| s.cost_cents);
        let taxable = (subtotal - self.discount_cents).max(0);
        let tax = (taxable as f64 * self.tax_rate).round() as i64;
        let total = (subtotal + shipping + tax - self.discount_cents).max(0);
        let item_count: u32 = self.items.iter().map(|i| i.quantity).sum();
        OrderSummary {
            subtotal_cents: subtotal,
            shipping_cents: shipping,
            tax_cents: tax,
            discount_cents: self.discount_cents,
            total_cents: total,
            item_count,
        }
    }

    /// Complete the checkout and generate an order confirmation.
    pub fn complete(&mut self) -> Result<OrderConfirmation, CheckoutError> {
        if self.items.is_empty() {
            return Err(CheckoutError::EmptyCart);
        }
        if self.step != CheckoutStep::Processing {
            return Err(CheckoutError::InvalidTransition {
                from: self.step,
                to: CheckoutStep::Complete,
            });
        }

        let order_id = format!("ORD-{:06}", self.next_order_seq);
        self.next_order_seq += 1;
        let summary = self.summary();
        let shipping_address = self
            .shipping_address
            .clone()
            .unwrap_or_default();
        let shipping_method = self
            .selected_shipping
            .as_ref()
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "Standard".to_string());

        let confirmation = OrderConfirmation {
            order_id,
            summary,
            shipping_address,
            shipping_method,
        };

        self.order_confirmation = Some(confirmation.clone());
        self.step = CheckoutStep::Complete;
        Ok(confirmation)
    }

    /// Mark checkout as failed.
    pub fn fail(&mut self, reason: &str) -> Result<(), CheckoutError> {
        if self.step != CheckoutStep::Processing {
            return Err(CheckoutError::InvalidTransition {
                from: self.step,
                to: CheckoutStep::Failed,
            });
        }
        self.step = CheckoutStep::Failed;
        self.metadata
            .insert("failure_reason".into(), reason.into());
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_items() -> Vec<CheckoutItem> {
        vec![
            CheckoutItem {
                product_id: "p1".into(),
                name: "Widget".into(),
                price_cents: 1500,
                quantity: 2,
            },
            CheckoutItem {
                product_id: "p2".into(),
                name: "Gadget".into(),
                price_cents: 2500,
                quantity: 1,
            },
        ]
    }

    fn valid_address() -> Address {
        Address {
            name: "Alice Smith".into(),
            line1: "123 Main St".into(),
            line2: None,
            city: "Springfield".into(),
            state: "IL".into(),
            postal_code: "62701".into(),
            country: "US".into(),
        }
    }

    fn sample_shipping() -> ShippingOption {
        ShippingOption {
            id: "std".into(),
            name: "Standard".into(),
            cost_cents: 599,
            estimated_days: 5,
        }
    }

    #[test]
    fn initial_step_is_cart_review() {
        let session = CheckoutSession::new("s1", sample_items());
        assert_eq!(session.step, CheckoutStep::CartReview);
    }

    #[test]
    fn advance_through_steps() {
        let mut s = CheckoutSession::new("s2", sample_items());
        s.advance().unwrap(); // -> ShippingInfo
        assert_eq!(s.step, CheckoutStep::ShippingInfo);
        s.advance().unwrap(); // -> ShippingMethod
        assert_eq!(s.step, CheckoutStep::ShippingMethod);
        s.advance().unwrap(); // -> PaymentInfo
        assert_eq!(s.step, CheckoutStep::PaymentInfo);
        s.advance().unwrap(); // -> Review
        assert_eq!(s.step, CheckoutStep::Review);
    }

    #[test]
    fn go_back() {
        let mut s = CheckoutSession::new("s3", sample_items());
        s.advance().unwrap(); // ShippingInfo
        s.advance().unwrap(); // ShippingMethod
        s.go_back().unwrap(); // ShippingInfo
        assert_eq!(s.step, CheckoutStep::ShippingInfo);
    }

    #[test]
    fn invalid_skip() {
        let mut s = CheckoutSession::new("s4", sample_items());
        let result = s.go_to(CheckoutStep::PaymentInfo);
        assert!(matches!(
            result,
            Err(CheckoutError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn address_validation_ok() {
        let addr = valid_address();
        assert!(addr.validate().is_ok());
    }

    #[test]
    fn address_validation_missing_city() {
        let mut addr = valid_address();
        addr.city = "".into();
        let result = addr.validate();
        assert!(matches!(result, Err(CheckoutError::MissingField(f)) if f == "city"));
    }

    #[test]
    fn order_summary() {
        let mut s = CheckoutSession::new("s5", sample_items());
        s.set_shipping_method(sample_shipping());
        s.set_tax_rate(0.08);
        s.apply_discount(500);
        let summary = s.summary();
        // subtotal: 1500*2 + 2500 = 5500
        assert_eq!(summary.subtotal_cents, 5500);
        assert_eq!(summary.shipping_cents, 599);
        assert_eq!(summary.discount_cents, 500);
        // tax on (5500-500) = 5000 * 0.08 = 400
        assert_eq!(summary.tax_cents, 400);
        // total: 5500 + 599 + 400 - 500 = 5999
        assert_eq!(summary.total_cents, 5999);
        assert_eq!(summary.item_count, 3);
    }

    #[test]
    fn complete_checkout() {
        let mut s = CheckoutSession::new("s6", sample_items());
        s.set_shipping_address(valid_address()).unwrap();
        s.set_shipping_method(sample_shipping());
        // Walk to Processing
        s.advance().unwrap(); // ShippingInfo
        s.advance().unwrap(); // ShippingMethod
        s.advance().unwrap(); // PaymentInfo
        s.advance().unwrap(); // Review
        s.advance().unwrap(); // Processing
        assert_eq!(s.step, CheckoutStep::Processing);

        let confirmation = s.complete().unwrap();
        assert!(confirmation.order_id.starts_with("ORD-"));
        assert_eq!(s.step, CheckoutStep::Complete);
    }

    #[test]
    fn cannot_complete_empty_cart() {
        let mut s = CheckoutSession::new("s7", vec![]);
        s.advance().unwrap();
        s.advance().unwrap();
        s.advance().unwrap();
        s.advance().unwrap();
        s.advance().unwrap();
        let result = s.complete();
        assert!(matches!(result, Err(CheckoutError::EmptyCart)));
    }

    #[test]
    fn fail_and_retry() {
        let mut s = CheckoutSession::new("s8", sample_items());
        // Walk to Processing
        for _ in 0..5 {
            s.advance().unwrap();
        }
        s.fail("card declined").unwrap();
        assert_eq!(s.step, CheckoutStep::Failed);
        assert_eq!(s.metadata.get("failure_reason").unwrap(), "card declined");
        // Retry from Failed -> PaymentInfo
        s.advance().unwrap();
        assert_eq!(s.step, CheckoutStep::PaymentInfo);
    }

    #[test]
    fn already_complete() {
        let mut s = CheckoutSession::new("s9", sample_items());
        for _ in 0..5 {
            s.advance().unwrap();
        }
        s.complete().unwrap();
        let result = s.advance();
        assert!(matches!(result, Err(CheckoutError::AlreadyComplete)));
    }

    #[test]
    fn line_total() {
        let item = CheckoutItem {
            product_id: "x".into(),
            name: "X".into(),
            price_cents: 1000,
            quantity: 3,
        };
        assert_eq!(item.line_total(), 3000);
    }

    #[test]
    fn step_ordinals_ordered() {
        assert!(CheckoutStep::CartReview.ordinal() < CheckoutStep::ShippingInfo.ordinal());
        assert!(CheckoutStep::Review.ordinal() < CheckoutStep::Processing.ordinal());
        assert!(CheckoutStep::Processing.ordinal() < CheckoutStep::Complete.ordinal());
    }
}
