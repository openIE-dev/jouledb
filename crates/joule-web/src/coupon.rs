//! Coupon and discount codes — validation, redemption, stacking, and bulk generation.
//!
//! Replaces coupon/promo-code libraries with a pure-Rust discount model
//! that supports percentage and fixed-amount discounts, stacking rules,
//! usage limits, and expiration.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Coupon domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CouponError {
    /// Coupon not found.
    NotFound(String),
    /// Coupon has expired.
    Expired(String),
    /// Coupon has reached max uses.
    MaxUsesReached(String),
    /// Purchase does not meet minimum.
    MinPurchaseNotMet { code: String, min_cents: i64, actual_cents: i64 },
    /// Coupon not applicable to any items.
    NotApplicable(String),
    /// Cannot stack exclusive coupons.
    CannotStack(String),
    /// Duplicate coupon code.
    DuplicateCode(String),
}

impl std::fmt::Display for CouponError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(c) => write!(f, "coupon not found: {c}"),
            Self::Expired(c) => write!(f, "coupon expired: {c}"),
            Self::MaxUsesReached(c) => write!(f, "coupon max uses reached: {c}"),
            Self::MinPurchaseNotMet { code, min_cents, actual_cents } => {
                write!(f, "coupon {code} requires min {min_cents}, got {actual_cents}")
            }
            Self::NotApplicable(c) => write!(f, "coupon not applicable: {c}"),
            Self::CannotStack(c) => write!(f, "cannot stack exclusive coupon: {c}"),
            Self::DuplicateCode(c) => write!(f, "duplicate coupon code: {c}"),
        }
    }
}

impl std::error::Error for CouponError {}

// ── Discount Type ───────────────────────────────────────────────

/// Type of discount a coupon provides.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DiscountType {
    /// Percentage off (0.0 to 100.0).
    Percentage(f64),
    /// Fixed amount off in cents.
    FixedAmount(i64),
}

// ── Stacking Rule ───────────────────────────────────────────────

/// Whether a coupon can be combined with others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StackingRule {
    /// Can be combined with other stackable coupons.
    Stackable,
    /// Cannot be combined with any other coupon.
    Exclusive,
}

// ── Coupon ───────────────────────────────────────────────────────

/// A discount coupon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coupon {
    pub code: String,
    pub discount_type: DiscountType,
    pub min_purchase_cents: Option<i64>,
    pub max_uses: Option<u32>,
    pub used_count: u32,
    pub expires_at: Option<DateTime<Utc>>,
    /// If non-empty, coupon only applies to these product IDs.
    pub applicable_product_ids: Vec<String>,
    pub stacking_rule: StackingRule,
    pub description: Option<String>,
}

impl Coupon {
    /// Create a new coupon.
    pub fn new(code: impl Into<String>, discount_type: DiscountType) -> Self {
        Self {
            code: code.into(),
            discount_type,
            min_purchase_cents: None,
            max_uses: None,
            used_count: 0,
            expires_at: None,
            applicable_product_ids: Vec::new(),
            stacking_rule: StackingRule::Stackable,
            description: None,
        }
    }

    pub fn with_min_purchase(mut self, cents: i64) -> Self {
        self.min_purchase_cents = Some(cents);
        self
    }

    pub fn with_max_uses(mut self, max: u32) -> Self {
        self.max_uses = Some(max);
        self
    }

    pub fn with_expiry(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    pub fn with_products(mut self, ids: Vec<String>) -> Self {
        self.applicable_product_ids = ids;
        self
    }

    pub fn exclusive(mut self) -> Self {
        self.stacking_rule = StackingRule::Exclusive;
        self
    }

    /// Calculate the discount amount in cents for a given subtotal.
    pub fn discount_amount(&self, subtotal_cents: i64) -> i64 {
        match self.discount_type {
            DiscountType::Percentage(pct) => {
                ((subtotal_cents as f64) * pct / 100.0).round() as i64
            }
            DiscountType::FixedAmount(amount) => amount.min(subtotal_cents),
        }
    }

    /// Whether the coupon is still valid (not expired, not maxed out).
    pub fn is_valid(&self, now: DateTime<Utc>) -> bool {
        if let Some(exp) = self.expires_at {
            if now >= exp {
                return false;
            }
        }
        if let Some(max) = self.max_uses {
            if self.used_count >= max {
                return false;
            }
        }
        true
    }

    /// Whether the coupon applies to a given product.
    pub fn applies_to_product(&self, product_id: &str) -> bool {
        if self.applicable_product_ids.is_empty() {
            return true;
        }
        self.applicable_product_ids.iter().any(|id| id == product_id)
    }
}

// ── Code Generation ─────────────────────────────────────────────

/// Generate a random alphanumeric coupon code of given length.
/// Uses a simple deterministic PRNG seeded from the given seed value.
pub fn generate_code(length: usize, seed: u64) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut state = seed;
    let mut code = String::with_capacity(length);
    for _ in 0..length {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let idx = (state as usize) % CHARS.len();
        code.push(CHARS[idx] as char);
    }
    code
}

/// Generate a batch of unique coupon codes.
pub fn generate_codes_bulk(count: usize, length: usize, base_seed: u64) -> Vec<String> {
    let mut codes = Vec::with_capacity(count);
    let mut seen = std::collections::HashSet::new();
    for i in 0..count {
        let seed = base_seed.wrapping_add(i as u64).wrapping_mul(6364136223846793005);
        let code = generate_code(length, seed);
        if seen.insert(code.clone()) {
            codes.push(code);
        }
    }
    codes
}

// ── Coupon Store ────────────────────────────────────────────────

/// In-memory coupon store with validation and redemption.
#[derive(Debug, Default)]
pub struct CouponStore {
    coupons: HashMap<String, Coupon>,
}

impl CouponStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a coupon to the store.
    pub fn add(&mut self, coupon: Coupon) -> Result<(), CouponError> {
        if self.coupons.contains_key(&coupon.code) {
            return Err(CouponError::DuplicateCode(coupon.code));
        }
        self.coupons.insert(coupon.code.clone(), coupon);
        Ok(())
    }

    /// Get a coupon by code.
    pub fn get(&self, code: &str) -> Option<&Coupon> {
        self.coupons.get(code)
    }

    /// Validate a coupon against a cart total and product list.
    pub fn validate(
        &self,
        code: &str,
        cart_total_cents: i64,
        product_ids: &[&str],
        now: DateTime<Utc>,
    ) -> Result<&Coupon, CouponError> {
        let coupon = self
            .coupons
            .get(code)
            .ok_or_else(|| CouponError::NotFound(code.to_string()))?;

        if !coupon.is_valid(now) {
            if coupon.expires_at.is_some_and(|exp| now >= exp) {
                return Err(CouponError::Expired(code.to_string()));
            }
            return Err(CouponError::MaxUsesReached(code.to_string()));
        }

        if let Some(min) = coupon.min_purchase_cents {
            if cart_total_cents < min {
                return Err(CouponError::MinPurchaseNotMet {
                    code: code.to_string(),
                    min_cents: min,
                    actual_cents: cart_total_cents,
                });
            }
        }

        if !coupon.applicable_product_ids.is_empty()
            && !product_ids
                .iter()
                .any(|pid| coupon.applies_to_product(pid))
        {
            return Err(CouponError::NotApplicable(code.to_string()));
        }

        Ok(coupon)
    }

    /// Redeem a coupon (increment used_count).
    pub fn redeem(&mut self, code: &str, now: DateTime<Utc>) -> Result<(), CouponError> {
        let coupon = self
            .coupons
            .get_mut(code)
            .ok_or_else(|| CouponError::NotFound(code.to_string()))?;
        if !coupon.is_valid(now) {
            return Err(CouponError::MaxUsesReached(code.to_string()));
        }
        coupon.used_count += 1;
        Ok(())
    }
}

// ── Apply Coupons ───────────────────────────────────────────────

/// Apply multiple coupons to a cart total, respecting stacking rules.
/// Returns the total discount in cents.
pub fn apply_coupons(
    store: &CouponStore,
    codes: &[&str],
    cart_total_cents: i64,
    product_ids: &[&str],
    now: DateTime<Utc>,
) -> Result<i64, CouponError> {
    if codes.is_empty() {
        return Ok(0);
    }

    let mut coupons = Vec::new();
    for code in codes {
        let coupon = store.validate(code, cart_total_cents, product_ids, now)?;
        coupons.push(coupon);
    }

    // Check stacking rules
    if coupons.len() > 1 {
        for c in &coupons {
            if c.stacking_rule == StackingRule::Exclusive {
                return Err(CouponError::CannotStack(c.code.clone()));
            }
        }
    }

    let mut total_discount: i64 = 0;
    let mut remaining = cart_total_cents;
    for coupon in &coupons {
        let discount = coupon.discount_amount(remaining);
        total_discount += discount;
        remaining -= discount;
        if remaining <= 0 {
            break;
        }
    }

    // Never discount below zero
    Ok(total_discount.min(cart_total_cents))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn percentage_discount() {
        let coupon = Coupon::new("SAVE20", DiscountType::Percentage(20.0));
        assert_eq!(coupon.discount_amount(10000), 2000);
    }

    #[test]
    fn fixed_discount() {
        let coupon = Coupon::new("FLAT500", DiscountType::FixedAmount(500));
        assert_eq!(coupon.discount_amount(10000), 500);
    }

    #[test]
    fn fixed_discount_capped_at_subtotal() {
        let coupon = Coupon::new("FLAT5000", DiscountType::FixedAmount(5000));
        assert_eq!(coupon.discount_amount(3000), 3000);
    }

    #[test]
    fn coupon_expiry() {
        let exp = now() - Duration::days(1);
        let coupon = Coupon::new("OLD", DiscountType::Percentage(10.0)).with_expiry(exp);
        assert!(!coupon.is_valid(now()));
    }

    #[test]
    fn coupon_max_uses() {
        let mut coupon =
            Coupon::new("LIMITED", DiscountType::Percentage(10.0)).with_max_uses(2);
        coupon.used_count = 2;
        assert!(!coupon.is_valid(now()));
    }

    #[test]
    fn coupon_product_filter() {
        let coupon = Coupon::new("PROD", DiscountType::Percentage(10.0))
            .with_products(vec!["prod-1".into(), "prod-2".into()]);
        assert!(coupon.applies_to_product("prod-1"));
        assert!(!coupon.applies_to_product("prod-3"));
    }

    #[test]
    fn store_validate_and_redeem() {
        let mut store = CouponStore::new();
        store
            .add(Coupon::new("WELCOME", DiscountType::Percentage(15.0)).with_max_uses(3))
            .unwrap();
        store
            .validate("WELCOME", 5000, &["any"], now())
            .unwrap();
        store.redeem("WELCOME", now()).unwrap();
        let coupon = store.get("WELCOME").unwrap();
        assert_eq!(coupon.used_count, 1);
    }

    #[test]
    fn store_min_purchase() {
        let mut store = CouponStore::new();
        store
            .add(
                Coupon::new("BIG", DiscountType::FixedAmount(1000))
                    .with_min_purchase(5000),
            )
            .unwrap();
        let result = store.validate("BIG", 3000, &["any"], now());
        assert!(matches!(result, Err(CouponError::MinPurchaseNotMet { .. })));
    }

    #[test]
    fn exclusive_cannot_stack() {
        let mut store = CouponStore::new();
        store
            .add(Coupon::new("A", DiscountType::Percentage(10.0)).exclusive())
            .unwrap();
        store
            .add(Coupon::new("B", DiscountType::Percentage(5.0)))
            .unwrap();
        let result = apply_coupons(&store, &["A", "B"], 10000, &["any"], now());
        assert!(matches!(result, Err(CouponError::CannotStack(_))));
    }

    #[test]
    fn stackable_coupons() {
        let mut store = CouponStore::new();
        store
            .add(Coupon::new("S1", DiscountType::FixedAmount(500)))
            .unwrap();
        store
            .add(Coupon::new("S2", DiscountType::FixedAmount(300)))
            .unwrap();
        let discount = apply_coupons(&store, &["S1", "S2"], 10000, &["any"], now()).unwrap();
        assert_eq!(discount, 800);
    }

    #[test]
    fn generate_code_deterministic() {
        let a = generate_code(8, 42);
        let b = generate_code(8, 42);
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn bulk_generate_unique() {
        let codes = generate_codes_bulk(10, 10, 12345);
        assert_eq!(codes.len(), 10);
        let set: std::collections::HashSet<_> = codes.iter().collect();
        assert_eq!(set.len(), 10); // all unique
    }

    #[test]
    fn not_found() {
        let store = CouponStore::new();
        let result = store.validate("NOPE", 1000, &["any"], now());
        assert!(matches!(result, Err(CouponError::NotFound(_))));
    }

    #[test]
    fn duplicate_code_rejected() {
        let mut store = CouponStore::new();
        store
            .add(Coupon::new("DUP", DiscountType::Percentage(10.0)))
            .unwrap();
        let result = store.add(Coupon::new("DUP", DiscountType::Percentage(20.0)));
        assert!(matches!(result, Err(CouponError::DuplicateCode(_))));
    }
}
