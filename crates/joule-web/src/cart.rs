//! Shopping cart — add, remove, update, discount, merge, TTL, and serialization.
//!
//! Replaces Redux cart slices and JS shopping cart libraries with a pure-Rust
//! cart model that tracks items, quantities, discounts, and expiration.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Cart domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CartError {
    /// Item not found in cart.
    ItemNotFound(String),
    /// Quantity exceeds maximum allowed per item.
    MaxQuantityExceeded { product_id: String, max: u32 },
    /// Cart has expired.
    CartExpired,
    /// Zero quantity.
    ZeroQuantity,
}

impl std::fmt::Display for CartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ItemNotFound(id) => write!(f, "item not found: {id}"),
            Self::MaxQuantityExceeded { product_id, max } => {
                write!(f, "max quantity {max} exceeded for {product_id}")
            }
            Self::CartExpired => write!(f, "cart has expired"),
            Self::ZeroQuantity => write!(f, "quantity must be at least 1"),
        }
    }
}

impl std::error::Error for CartError {}

// ── Cart Item ───────────────────────────────────────────────────

/// A single item in the shopping cart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CartItem {
    pub product_id: String,
    pub name: String,
    pub price_cents: i64,
    pub quantity: u32,
    pub variant: Option<String>,
}

impl CartItem {
    pub fn new(
        product_id: impl Into<String>,
        name: impl Into<String>,
        price_cents: i64,
        quantity: u32,
        variant: Option<String>,
    ) -> Self {
        Self {
            product_id: product_id.into(),
            name: name.into(),
            price_cents,
            quantity,
            variant,
        }
    }

    /// Line total in cents.
    pub fn line_total(&self) -> i64 {
        self.price_cents * self.quantity as i64
    }
}

// ── Discount ────────────────────────────────────────────────────

/// A discount applied to the cart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Discount {
    /// Percentage off the subtotal (0.0 - 100.0).
    Percentage(f64),
    /// Fixed amount off in cents.
    FixedAmount(i64),
}

impl Discount {
    /// Compute the discount amount in cents given a subtotal.
    pub fn compute(&self, subtotal_cents: i64) -> i64 {
        match self {
            Self::Percentage(pct) => {
                let raw = subtotal_cents as f64 * (pct / 100.0);
                let amount = raw.round() as i64;
                amount.min(subtotal_cents).max(0)
            }
            Self::FixedAmount(amount) => (*amount).min(subtotal_cents).max(0),
        }
    }
}

// ── Cart ────────────────────────────────────────────────────────

/// A shopping cart with items, discount, expiration, and quantity limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cart {
    pub id: String,
    items: HashMap<String, CartItem>,
    pub discount: Option<Discount>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub max_quantity_per_item: u32,
}

impl Cart {
    /// Create a new cart. `ttl_minutes` of 0 means no expiration.
    pub fn new(id: impl Into<String>, ttl_minutes: u32, max_quantity_per_item: u32) -> Self {
        let now = Utc::now();
        let expires_at = if ttl_minutes > 0 {
            Some(now + Duration::minutes(ttl_minutes as i64))
        } else {
            None
        };
        Self {
            id: id.into(),
            items: HashMap::new(),
            discount: None,
            created_at: now,
            expires_at,
            max_quantity_per_item,
        }
    }

    /// Check if the cart has expired at the given time.
    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.is_some_and(|exp| now >= exp)
    }

    /// Check expiration against the current time.
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(Utc::now())
    }

    fn check_expired(&self) -> Result<(), CartError> {
        if self.is_expired() {
            Err(CartError::CartExpired)
        } else {
            Ok(())
        }
    }

    /// Add an item or increment quantity if already present (same product_id + variant).
    pub fn add_item(&mut self, item: CartItem) -> Result<(), CartError> {
        self.check_expired()?;
        if item.quantity == 0 {
            return Err(CartError::ZeroQuantity);
        }
        let key = cart_key(&item.product_id, item.variant.as_deref());
        if let Some(existing) = self.items.get_mut(&key) {
            let new_qty = existing.quantity + item.quantity;
            if new_qty > self.max_quantity_per_item {
                return Err(CartError::MaxQuantityExceeded {
                    product_id: item.product_id,
                    max: self.max_quantity_per_item,
                });
            }
            existing.quantity = new_qty;
        } else {
            if item.quantity > self.max_quantity_per_item {
                return Err(CartError::MaxQuantityExceeded {
                    product_id: item.product_id,
                    max: self.max_quantity_per_item,
                });
            }
            self.items.insert(key, item);
        }
        Ok(())
    }

    /// Remove an item entirely.
    pub fn remove_item(&mut self, product_id: &str, variant: Option<&str>) -> Result<CartItem, CartError> {
        self.check_expired()?;
        let key = cart_key(product_id, variant);
        self.items
            .remove(&key)
            .ok_or_else(|| CartError::ItemNotFound(product_id.to_string()))
    }

    /// Update the quantity of an existing item.
    pub fn update_quantity(
        &mut self,
        product_id: &str,
        variant: Option<&str>,
        quantity: u32,
    ) -> Result<(), CartError> {
        self.check_expired()?;
        if quantity == 0 {
            self.remove_item(product_id, variant)?;
            return Ok(());
        }
        if quantity > self.max_quantity_per_item {
            return Err(CartError::MaxQuantityExceeded {
                product_id: product_id.to_string(),
                max: self.max_quantity_per_item,
            });
        }
        let key = cart_key(product_id, variant);
        let item = self
            .items
            .get_mut(&key)
            .ok_or_else(|| CartError::ItemNotFound(product_id.to_string()))?;
        item.quantity = quantity;
        Ok(())
    }

    /// Get an item reference.
    pub fn get_item(&self, product_id: &str, variant: Option<&str>) -> Option<&CartItem> {
        let key = cart_key(product_id, variant);
        self.items.get(&key)
    }

    /// All items in the cart.
    pub fn items(&self) -> Vec<&CartItem> {
        self.items.values().collect()
    }

    /// Number of distinct line items.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Total number of units across all items.
    pub fn total_units(&self) -> u32 {
        self.items.values().map(|i| i.quantity).sum()
    }

    /// Subtotal before discount.
    pub fn subtotal_cents(&self) -> i64 {
        self.items.values().map(|i| i.line_total()).sum()
    }

    /// Discount amount in cents.
    pub fn discount_cents(&self) -> i64 {
        self.discount
            .as_ref()
            .map(|d| d.compute(self.subtotal_cents()))
            .unwrap_or(0)
    }

    /// Total after discount.
    pub fn total_cents(&self) -> i64 {
        self.subtotal_cents() - self.discount_cents()
    }

    /// Apply a discount.
    pub fn apply_discount(&mut self, discount: Discount) {
        self.discount = Some(discount);
    }

    /// Remove the current discount.
    pub fn remove_discount(&mut self) {
        self.discount = None;
    }

    /// Merge another cart (guest) into this cart (authenticated).
    /// Items from `other` are added; if the same product+variant exists, quantities sum.
    pub fn merge(&mut self, other: &Cart) -> Result<(), CartError> {
        self.check_expired()?;
        for item in other.items.values() {
            let key = cart_key(&item.product_id, item.variant.as_deref());
            if let Some(existing) = self.items.get_mut(&key) {
                let new_qty = (existing.quantity + item.quantity).min(self.max_quantity_per_item);
                existing.quantity = new_qty;
            } else {
                let mut cloned = item.clone();
                cloned.quantity = cloned.quantity.min(self.max_quantity_per_item);
                self.items.insert(key, cloned);
            }
        }
        Ok(())
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
        self.discount = None;
    }

    /// Whether the cart is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Serialize the cart to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Restore a cart from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Build a composite key for product+variant.
fn cart_key(product_id: &str, variant: Option<&str>) -> String {
    match variant {
        Some(v) => format!("{product_id}::{v}"),
        None => product_id.to_string(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, price: i64, qty: u32) -> CartItem {
        CartItem::new(id, format!("Product {id}"), price, qty, None)
    }

    #[test]
    fn add_and_subtotal() {
        let mut cart = Cart::new("c1", 0, 99);
        cart.add_item(item("a", 1000, 2)).unwrap();
        cart.add_item(item("b", 500, 1)).unwrap();
        assert_eq!(cart.subtotal_cents(), 2500);
        assert_eq!(cart.item_count(), 2);
        assert_eq!(cart.total_units(), 3);
    }

    #[test]
    fn add_increments_quantity() {
        let mut cart = Cart::new("c2", 0, 10);
        cart.add_item(item("a", 100, 2)).unwrap();
        cart.add_item(item("a", 100, 3)).unwrap();
        assert_eq!(cart.get_item("a", None).unwrap().quantity, 5);
        assert_eq!(cart.item_count(), 1);
    }

    #[test]
    fn max_quantity_enforced() {
        let mut cart = Cart::new("c3", 0, 5);
        assert!(cart.add_item(item("a", 100, 6)).is_err());
        cart.add_item(item("a", 100, 3)).unwrap();
        assert!(cart.add_item(item("a", 100, 3)).is_err());
    }

    #[test]
    fn remove_item() {
        let mut cart = Cart::new("c4", 0, 99);
        cart.add_item(item("a", 100, 1)).unwrap();
        let removed = cart.remove_item("a", None).unwrap();
        assert_eq!(removed.product_id, "a");
        assert!(cart.is_empty());
    }

    #[test]
    fn update_quantity() {
        let mut cart = Cart::new("c5", 0, 99);
        cart.add_item(item("a", 100, 1)).unwrap();
        cart.update_quantity("a", None, 5).unwrap();
        assert_eq!(cart.get_item("a", None).unwrap().quantity, 5);
    }

    #[test]
    fn update_to_zero_removes() {
        let mut cart = Cart::new("c6", 0, 99);
        cart.add_item(item("a", 100, 3)).unwrap();
        cart.update_quantity("a", None, 0).unwrap();
        assert!(cart.is_empty());
    }

    #[test]
    fn percentage_discount() {
        let mut cart = Cart::new("c7", 0, 99);
        cart.add_item(item("a", 10000, 1)).unwrap();
        cart.apply_discount(Discount::Percentage(15.0));
        assert_eq!(cart.discount_cents(), 1500);
        assert_eq!(cart.total_cents(), 8500);
    }

    #[test]
    fn fixed_discount() {
        let mut cart = Cart::new("c8", 0, 99);
        cart.add_item(item("a", 5000, 1)).unwrap();
        cart.apply_discount(Discount::FixedAmount(2000));
        assert_eq!(cart.total_cents(), 3000);
    }

    #[test]
    fn discount_capped_at_subtotal() {
        let mut cart = Cart::new("c9", 0, 99);
        cart.add_item(item("a", 1000, 1)).unwrap();
        cart.apply_discount(Discount::FixedAmount(5000));
        assert_eq!(cart.total_cents(), 0);
    }

    #[test]
    fn merge_carts() {
        let mut auth = Cart::new("auth", 0, 99);
        auth.add_item(item("a", 100, 2)).unwrap();

        let mut guest = Cart::new("guest", 0, 99);
        guest.add_item(item("a", 100, 1)).unwrap();
        guest.add_item(item("b", 200, 3)).unwrap();

        auth.merge(&guest).unwrap();
        assert_eq!(auth.get_item("a", None).unwrap().quantity, 3);
        assert_eq!(auth.get_item("b", None).unwrap().quantity, 3);
        assert_eq!(auth.item_count(), 2);
    }

    #[test]
    fn cart_expiration() {
        let mut cart = Cart::new("exp", 30, 99);
        let future = Utc::now() + Duration::minutes(60);
        assert!(cart.is_expired_at(future));
        assert!(!cart.is_expired());

        // Adding to an expired cart should fail (simulate by setting expires_at to past)
        cart.expires_at = Some(Utc::now() - Duration::minutes(1));
        assert!(cart.add_item(item("a", 100, 1)).is_err());
    }

    #[test]
    fn variant_separation() {
        let mut cart = Cart::new("v1", 0, 99);
        cart.add_item(CartItem::new("shoe", "Shoe", 5000, 1, Some("red".into())))
            .unwrap();
        cart.add_item(CartItem::new("shoe", "Shoe", 5000, 1, Some("blue".into())))
            .unwrap();
        assert_eq!(cart.item_count(), 2);
        assert_eq!(cart.total_units(), 2);
    }

    #[test]
    fn json_round_trip() {
        let mut cart = Cart::new("json1", 0, 99);
        cart.add_item(item("x", 999, 2)).unwrap();
        cart.apply_discount(Discount::Percentage(10.0));
        let json = cart.to_json().unwrap();
        let restored = Cart::from_json(&json).unwrap();
        assert_eq!(restored.subtotal_cents(), cart.subtotal_cents());
        assert_eq!(restored.item_count(), 1);
    }

    #[test]
    fn zero_quantity_rejected() {
        let mut cart = Cart::new("zq", 0, 99);
        assert!(cart.add_item(item("a", 100, 0)).is_err());
    }

    #[test]
    fn clear_cart() {
        let mut cart = Cart::new("cl", 0, 99);
        cart.add_item(item("a", 100, 1)).unwrap();
        cart.apply_discount(Discount::FixedAmount(50));
        cart.clear();
        assert!(cart.is_empty());
        assert!(cart.discount.is_none());
    }
}
