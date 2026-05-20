//! Payment processing model — intents, methods, refunds, and idempotency.
//!
//! Replaces Stripe.js / PayPal SDK with a pure-Rust payment domain model.
//! No HTTP calls — only constructs payment objects and validates state transitions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Payment domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaymentError {
    /// Invalid state transition.
    InvalidTransition { from: PaymentStatus, to: PaymentStatus },
    /// Refund exceeds captured amount.
    RefundExceedsCaptured { captured: u64, requested: u64 },
    /// Duplicate idempotency key.
    DuplicateIdempotencyKey(String),
    /// Payment not found.
    NotFound(String),
    /// Zero or negative amount.
    InvalidAmount,
    /// Refund on non-succeeded payment.
    CannotRefund(PaymentStatus),
}

impl std::fmt::Display for PaymentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from:?} to {to:?}")
            }
            Self::RefundExceedsCaptured { captured, requested } => {
                write!(f, "refund {requested} exceeds captured {captured}")
            }
            Self::DuplicateIdempotencyKey(k) => write!(f, "duplicate idempotency key: {k}"),
            Self::NotFound(id) => write!(f, "payment not found: {id}"),
            Self::InvalidAmount => write!(f, "amount must be positive"),
            Self::CannotRefund(s) => write!(f, "cannot refund payment in status {s:?}"),
        }
    }
}

impl std::error::Error for PaymentError {}

// ── Payment Method ──────────────────────────────────────────────

/// Supported payment methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentMethod {
    /// Credit or debit card.
    Card {
        last4: String,
        brand: String,
        exp_month: u8,
        exp_year: u16,
    },
    /// Bank transfer / ACH.
    BankTransfer,
    /// Digital wallet (e.g. "apple_pay", "google_pay").
    Wallet(String),
    /// Cryptocurrency.
    Crypto,
}

// ── Payment Status ──────────────────────────────────────────────

/// Lifecycle states for a payment intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PaymentStatus {
    Created,
    Processing,
    Succeeded,
    Failed,
    Cancelled,
}

impl PaymentStatus {
    /// Returns the set of valid next states.
    pub fn valid_transitions(self) -> &'static [PaymentStatus] {
        match self {
            Self::Created => &[Self::Processing, Self::Cancelled],
            Self::Processing => &[Self::Succeeded, Self::Failed],
            Self::Succeeded => &[], // terminal (refunds are separate)
            Self::Failed => &[Self::Created], // retry
            Self::Cancelled => &[],
        }
    }

    /// Check whether transitioning to `next` is legal.
    pub fn can_transition_to(self, next: Self) -> bool {
        self.valid_transitions().contains(&next)
    }
}

// ── Payment Intent ──────────────────────────────────────────────

/// A payment intent represents a single attempt to collect money.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentIntent {
    pub id: String,
    pub amount_cents: u64,
    pub currency: String,
    pub status: PaymentStatus,
    pub payment_method: Option<PaymentMethod>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub idempotency_key: Option<String>,
}

impl PaymentIntent {
    /// Create a new payment intent.
    pub fn new(id: impl Into<String>, amount_cents: u64, currency: impl Into<String>) -> Result<Self, PaymentError> {
        if amount_cents == 0 {
            return Err(PaymentError::InvalidAmount);
        }
        let now = Utc::now();
        Ok(Self {
            id: id.into(),
            amount_cents,
            currency: currency.into().to_uppercase(),
            status: PaymentStatus::Created,
            payment_method: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            idempotency_key: None,
        })
    }

    /// Attach a payment method.
    pub fn attach_method(&mut self, method: PaymentMethod) {
        self.payment_method = Some(method);
        self.updated_at = Utc::now();
    }

    /// Transition to a new status.
    pub fn transition(&mut self, next: PaymentStatus) -> Result<(), PaymentError> {
        if !self.status.can_transition_to(next) {
            return Err(PaymentError::InvalidTransition {
                from: self.status,
                to: next,
            });
        }
        self.status = next;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Set the idempotency key.
    pub fn set_idempotency_key(&mut self, key: impl Into<String>) {
        self.idempotency_key = Some(key.into());
    }

    /// Add metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

// ── Refund ──────────────────────────────────────────────────────

/// Type of refund.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefundType {
    Full,
    Partial,
}

/// A refund against a succeeded payment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Refund {
    pub id: String,
    pub payment_id: String,
    pub amount_cents: u64,
    pub refund_type: RefundType,
    pub reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl Refund {
    /// Create a refund. `total_captured` is the original payment amount,
    /// `already_refunded` is the sum of prior refunds.
    pub fn new(
        id: impl Into<String>,
        payment: &PaymentIntent,
        amount_cents: u64,
        already_refunded: u64,
        reason: Option<String>,
    ) -> Result<Self, PaymentError> {
        if payment.status != PaymentStatus::Succeeded {
            return Err(PaymentError::CannotRefund(payment.status));
        }
        if amount_cents == 0 {
            return Err(PaymentError::InvalidAmount);
        }
        let available = payment.amount_cents.saturating_sub(already_refunded);
        if amount_cents > available {
            return Err(PaymentError::RefundExceedsCaptured {
                captured: available,
                requested: amount_cents,
            });
        }
        let refund_type = if amount_cents == available {
            RefundType::Full
        } else {
            RefundType::Partial
        };
        Ok(Self {
            id: id.into(),
            payment_id: payment.id.clone(),
            amount_cents,
            refund_type,
            reason,
            created_at: Utc::now(),
        })
    }
}

// ── Payment Processor ───────────────────────────────────────────

/// Trait for payment processing backends.
pub trait PaymentProcessor {
    /// Create a payment intent on the backend.
    fn create_intent(&mut self, intent: &PaymentIntent) -> Result<String, PaymentError>;
    /// Confirm (charge) a payment intent.
    fn confirm(&mut self, payment_id: &str) -> Result<PaymentStatus, PaymentError>;
    /// Issue a refund.
    fn refund(&mut self, payment_id: &str, amount_cents: u64) -> Result<Refund, PaymentError>;
}

// ── Idempotency Registry ────────────────────────────────────────

/// Tracks idempotency keys to prevent duplicate payments.
#[derive(Debug, Default)]
pub struct IdempotencyRegistry {
    keys: HashMap<String, String>, // key -> payment_id
}

impl IdempotencyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a key. Returns error if already used.
    pub fn register(&mut self, key: &str, payment_id: &str) -> Result<(), PaymentError> {
        if self.keys.contains_key(key) {
            return Err(PaymentError::DuplicateIdempotencyKey(key.to_string()));
        }
        self.keys.insert(key.to_string(), payment_id.to_string());
        Ok(())
    }

    /// Look up the payment_id for an idempotency key.
    pub fn lookup(&self, key: &str) -> Option<&str> {
        self.keys.get(key).map(|s| s.as_str())
    }

    /// Number of tracked keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

// ── In-Memory Processor (for testing / demo) ────────────────────

/// Simple in-memory payment processor.
#[derive(Debug, Default)]
pub struct InMemoryProcessor {
    intents: HashMap<String, PaymentIntent>,
    refunds: Vec<Refund>,
    idempotency: IdempotencyRegistry,
    next_refund_id: u64,
}

impl InMemoryProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a stored intent by ID.
    pub fn get_intent(&self, id: &str) -> Option<&PaymentIntent> {
        self.intents.get(id)
    }

    /// Total refunded amount for a payment.
    pub fn total_refunded(&self, payment_id: &str) -> u64 {
        self.refunds
            .iter()
            .filter(|r| r.payment_id == payment_id)
            .map(|r| r.amount_cents)
            .sum()
    }
}

impl PaymentProcessor for InMemoryProcessor {
    fn create_intent(&mut self, intent: &PaymentIntent) -> Result<String, PaymentError> {
        if let Some(key) = &intent.idempotency_key {
            self.idempotency.register(key, &intent.id)?;
        }
        self.intents.insert(intent.id.clone(), intent.clone());
        Ok(intent.id.clone())
    }

    fn confirm(&mut self, payment_id: &str) -> Result<PaymentStatus, PaymentError> {
        let intent = self
            .intents
            .get_mut(payment_id)
            .ok_or_else(|| PaymentError::NotFound(payment_id.to_string()))?;
        intent.transition(PaymentStatus::Processing)?;
        intent.transition(PaymentStatus::Succeeded)?;
        Ok(PaymentStatus::Succeeded)
    }

    fn refund(&mut self, payment_id: &str, amount_cents: u64) -> Result<Refund, PaymentError> {
        let intent = self
            .intents
            .get(payment_id)
            .ok_or_else(|| PaymentError::NotFound(payment_id.to_string()))?
            .clone();
        let already = self.total_refunded(payment_id);
        self.next_refund_id += 1;
        let refund = Refund::new(
            format!("re_{}", self.next_refund_id),
            &intent,
            amount_cents,
            already,
            None,
        )?;
        self.refunds.push(refund.clone());
        Ok(refund)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_payment_intent() {
        let pi = PaymentIntent::new("pi_1", 5000, "usd").unwrap();
        assert_eq!(pi.amount_cents, 5000);
        assert_eq!(pi.currency, "USD");
        assert_eq!(pi.status, PaymentStatus::Created);
    }

    #[test]
    fn zero_amount_rejected() {
        assert!(PaymentIntent::new("pi_0", 0, "usd").is_err());
    }

    #[test]
    fn valid_transitions() {
        let mut pi = PaymentIntent::new("pi_2", 1000, "eur").unwrap();
        assert!(pi.transition(PaymentStatus::Processing).is_ok());
        assert!(pi.transition(PaymentStatus::Succeeded).is_ok());
    }

    #[test]
    fn invalid_transition_rejected() {
        let mut pi = PaymentIntent::new("pi_3", 1000, "gbp").unwrap();
        assert!(pi.transition(PaymentStatus::Succeeded).is_err());
    }

    #[test]
    fn attach_payment_method() {
        let mut pi = PaymentIntent::new("pi_4", 2500, "usd").unwrap();
        pi.attach_method(PaymentMethod::Card {
            last4: "4242".into(),
            brand: "visa".into(),
            exp_month: 12,
            exp_year: 2028,
        });
        assert!(pi.payment_method.is_some());
    }

    #[test]
    fn full_refund() {
        let mut pi = PaymentIntent::new("pi_5", 3000, "usd").unwrap();
        pi.transition(PaymentStatus::Processing).unwrap();
        pi.transition(PaymentStatus::Succeeded).unwrap();
        let refund = Refund::new("re_1", &pi, 3000, 0, None).unwrap();
        assert_eq!(refund.refund_type, RefundType::Full);
        assert_eq!(refund.amount_cents, 3000);
    }

    #[test]
    fn partial_refund() {
        let mut pi = PaymentIntent::new("pi_6", 5000, "usd").unwrap();
        pi.transition(PaymentStatus::Processing).unwrap();
        pi.transition(PaymentStatus::Succeeded).unwrap();
        let refund = Refund::new("re_2", &pi, 2000, 0, None).unwrap();
        assert_eq!(refund.refund_type, RefundType::Partial);
    }

    #[test]
    fn refund_exceeds_captured() {
        let mut pi = PaymentIntent::new("pi_7", 1000, "usd").unwrap();
        pi.transition(PaymentStatus::Processing).unwrap();
        pi.transition(PaymentStatus::Succeeded).unwrap();
        let err = Refund::new("re_3", &pi, 1500, 0, None).unwrap_err();
        assert!(matches!(err, PaymentError::RefundExceedsCaptured { .. }));
    }

    #[test]
    fn cannot_refund_non_succeeded() {
        let pi = PaymentIntent::new("pi_8", 1000, "usd").unwrap();
        let err = Refund::new("re_4", &pi, 1000, 0, None).unwrap_err();
        assert!(matches!(err, PaymentError::CannotRefund(PaymentStatus::Created)));
    }

    #[test]
    fn idempotency_prevents_duplicates() {
        let mut reg = IdempotencyRegistry::new();
        assert!(reg.register("key_1", "pi_a").is_ok());
        assert!(reg.register("key_1", "pi_b").is_err());
        assert_eq!(reg.lookup("key_1"), Some("pi_a"));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn in_memory_processor_flow() {
        let mut proc = InMemoryProcessor::new();
        let pi = PaymentIntent::new("pi_9", 10000, "usd").unwrap();
        proc.create_intent(&pi).unwrap();
        proc.confirm("pi_9").unwrap();
        let intent = proc.get_intent("pi_9").unwrap();
        assert_eq!(intent.status, PaymentStatus::Succeeded);

        let r1 = proc.refund("pi_9", 4000).unwrap();
        assert_eq!(r1.refund_type, RefundType::Partial);
        let r2 = proc.refund("pi_9", 6000).unwrap();
        assert_eq!(r2.refund_type, RefundType::Full);

        assert!(proc.refund("pi_9", 1).is_err());
    }

    #[test]
    fn metadata_round_trip() {
        let mut pi = PaymentIntent::new("pi_10", 500, "jpy").unwrap();
        pi.set_metadata("order_id", "ord_123");
        assert_eq!(pi.metadata.get("order_id").unwrap(), "ord_123");
    }

    #[test]
    fn serialization_round_trip() {
        let pi = PaymentIntent::new("pi_11", 9900, "usd").unwrap();
        let json = serde_json::to_string(&pi).unwrap();
        let restored: PaymentIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, "pi_11");
        assert_eq!(restored.amount_cents, 9900);
    }

    #[test]
    fn failed_payment_can_retry() {
        let mut pi = PaymentIntent::new("pi_12", 1000, "usd").unwrap();
        pi.transition(PaymentStatus::Processing).unwrap();
        pi.transition(PaymentStatus::Failed).unwrap();
        assert!(pi.transition(PaymentStatus::Created).is_ok());
    }

    #[test]
    fn cancelled_is_terminal() {
        let mut pi = PaymentIntent::new("pi_13", 1000, "usd").unwrap();
        pi.transition(PaymentStatus::Cancelled).unwrap();
        assert!(pi.transition(PaymentStatus::Processing).is_err());
        assert!(pi.transition(PaymentStatus::Created).is_err());
    }
}
