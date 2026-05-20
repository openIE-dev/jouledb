//! HDC-powered E-commerce and Retail module
//!
//! Provides holographic encoding for:
//! - Product similarity and recommendations
//! - Customer segmentation and behavior analysis
//! - Inventory optimization
//! - Fraud detection in transactions

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProductCategory {
    Electronics,
    Clothing,
    Home,
    Beauty,
    Sports,
    Food,
    Books,
    Toys,
    Automotive,
    Garden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CustomerSegment {
    NewCustomer,
    Occasional,
    Regular,
    Loyal,
    VIP,
    Churned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Completed,
    Refunded,
    Cancelled,
    Disputed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Product {
    pub id: String,
    pub name: String,
    pub category: ProductCategory,
    pub price: f64,
    pub brand: String,
    pub attributes: HashMap<String, String>,
    pub stock_level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Customer {
    pub id: String,
    pub segment: CustomerSegment,
    pub total_orders: u32,
    pub total_spent: f64,
    pub avg_order_value: f64,
    pub days_since_last_order: u32,
    pub preferred_categories: Vec<ProductCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub customer_id: String,
    pub items: Vec<(String, u32, f64)>,
    pub total: f64,
    pub status: TransactionStatus,
    pub timestamp: u64,
    pub payment_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartSession {
    pub session_id: String,
    pub customer_id: Option<String>,
    pub items: Vec<String>,
    pub page_views: Vec<String>,
    pub duration_secs: u32,
    pub converted: bool,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for retail domain data
    pub struct RetailLink {
        seed: 0x8E7A_0001,
        dimension: 10000,
        fields: ["product", "customer", "transaction", "cart", "category", "brand", "item"],
        scalars: ["price", "quantity", "orders", "spent", "days", "stock"],
        enums: {
            category_vectors: ProductCategory => [ProductCategory::Electronics, ProductCategory::Clothing, ProductCategory::Home, ProductCategory::Beauty, ProductCategory::Sports, ProductCategory::Food, ProductCategory::Books, ProductCategory::Toys, ProductCategory::Automotive, ProductCategory::Garden],
            segment_vectors: CustomerSegment => [CustomerSegment::NewCustomer, CustomerSegment::Occasional, CustomerSegment::Regular, CustomerSegment::Loyal, CustomerSegment::VIP, CustomerSegment::Churned],
            status_vectors: TransactionStatus => [TransactionStatus::Pending, TransactionStatus::Completed, TransactionStatus::Refunded, TransactionStatus::Cancelled, TransactionStatus::Disputed]
        },
        dynamic: {
            brand_vectors: "brand"
        },
    }
}

impl RetailLink {
    pub fn encode_product(&mut self, product: &Product) -> BinaryHV {
        let category_hv =
            self.field_vectors["category"].bind(&self.category_vectors[&product.category]);
        let price_hv = self.encode_scalar("price", (product.price * 100.0) as u32, 100000);
        let brand_vec = self.brand_vectors(&product.brand);
        let brand_hv = self.field_vectors["brand"].bind(&brand_vec);
        let name_hv = BinaryHV::from_hash(product.name.as_bytes(), DIMENSION);
        let stock_hv = self.encode_scalar("stock", product.stock_level.min(10000), 10000);
        let mut components = vec![category_hv, price_hv, brand_hv, name_hv, stock_hv];
        for (attr_name, attr_val) in &product.attributes {
            let attr_hv =
                BinaryHV::from_hash(format!("{}:{}", attr_name, attr_val).as_bytes(), DIMENSION);
            components.push(attr_hv);
        }
        self.bundle(&components)
    }

    pub fn encode_customer(&self, customer: &Customer) -> BinaryHV {
        let segment_hv =
            self.field_vectors["customer"].bind(&self.segment_vectors[&customer.segment]);
        let orders_hv = self.encode_scalar("orders", customer.total_orders.min(1000), 1000);
        let spent_hv = self.encode_scalar("spent", (customer.total_spent / 10.0) as u32, 100000);
        let recency_hv = self.encode_scalar("days", customer.days_since_last_order.min(365), 365);
        let mut components = vec![segment_hv, orders_hv, spent_hv, recency_hv];
        for cat in &customer.preferred_categories {
            components.push(self.field_vectors["category"].bind(&self.category_vectors[cat]));
        }
        self.bundle(&components)
    }

    pub fn encode_transaction(&self, txn: &Transaction) -> BinaryHV {
        let status_hv = self.field_vectors["transaction"].bind(&self.status_vectors[&txn.status]);
        let total_hv = self.encode_scalar("spent", (txn.total * 100.0) as u32, 100000);
        let items_count_hv = self.encode_scalar("quantity", txn.items.len() as u32, 100);
        let mut components = vec![status_hv, total_hv, items_count_hv];
        for (item_id, _, _) in &txn.items {
            components.push(
                self.field_vectors["item"]
                    .bind(&BinaryHV::from_hash(item_id.as_bytes(), DIMENSION)),
            );
        }
        self.bundle(&components)
    }

    pub fn encode_cart_session(&self, cart: &CartSession) -> BinaryHV {
        let duration_hv = self.encode_scalar("days", cart.duration_secs / 60, 120);
        let items_hv = self.encode_scalar("quantity", cart.items.len() as u32, 50);
        let mut components = vec![duration_hv, items_hv];
        for item in &cart.items {
            components.push(
                self.field_vectors["item"].bind(&BinaryHV::from_hash(item.as_bytes(), DIMENSION)),
            );
        }
        self.bundle(&components)
    }
}

pub struct ProductCatalog {
    encoder: RetailLink,
    product_vectors: HashMap<String, BinaryHV>,
    products: HashMap<String, Product>,
}

impl ProductCatalog {
    pub fn new() -> Self {
        Self {
            encoder: RetailLink::new(),
            product_vectors: HashMap::new(),
            products: HashMap::new(),
        }
    }

    pub fn add_product(&mut self, product: Product) {
        let hv = self.encoder.encode_product(&product);
        self.product_vectors.insert(product.id.clone(), hv);
        self.products.insert(product.id.clone(), product);
    }

    pub fn find_similar(&self, product_id: &str, limit: usize) -> Vec<(String, f32)> {
        let query = match self.product_vectors.get(product_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .product_vectors
            .iter()
            .filter(|(id, _)| *id != product_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn product_count(&self) -> usize {
        self.products.len()
    }
}

impl Default for ProductCatalog {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RecommendationEngine {
    encoder: RetailLink,
    customer_vectors: HashMap<String, BinaryHV>,
    purchase_history: HashMap<String, BundleAccumulator>,
    product_catalog: ProductCatalog,
}

impl RecommendationEngine {
    pub fn new() -> Self {
        Self {
            encoder: RetailLink::new(),
            customer_vectors: HashMap::new(),
            purchase_history: HashMap::new(),
            product_catalog: ProductCatalog::new(),
        }
    }

    pub fn register_customer(&mut self, customer: &Customer) {
        let hv = self.encoder.encode_customer(customer);
        self.customer_vectors.insert(customer.id.clone(), hv);
        if !self.purchase_history.contains_key(&customer.id) {
            self.purchase_history
                .insert(customer.id.clone(), BundleAccumulator::new(DIMENSION));
        }
    }

    pub fn add_product(&mut self, product: Product) {
        self.product_catalog.add_product(product);
    }

    pub fn record_purchase(&mut self, customer_id: &str, product: &Product) {
        let hv = self.encoder.encode_product(product);
        if let Some(acc) = self.purchase_history.get_mut(customer_id) {
            acc.add(&hv);
        }
    }

    pub fn recommend(&self, customer_id: &str, limit: usize) -> Vec<(String, f32)> {
        let history = match self.purchase_history.get(customer_id) {
            Some(acc) => acc.threshold(),
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .product_catalog
            .product_vectors
            .iter()
            .map(|(id, hv)| (id.clone(), history.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}

impl Default for RecommendationEngine {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TransactionFraudDetector {
    encoder: RetailLink,
    normal_patterns: BundleAccumulator,
    fraud_patterns: BundleAccumulator,
    threshold: f32,
}

#[derive(Debug, Clone)]
pub struct FraudAlert {
    pub transaction_id: String,
    pub fraud_score: f32,
    pub indicators: Vec<String>,
}

impl TransactionFraudDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: RetailLink::new(),
            normal_patterns: BundleAccumulator::new(DIMENSION),
            fraud_patterns: BundleAccumulator::new(DIMENSION),
            threshold,
        }
    }

    pub fn learn_normal(&mut self, txn: &Transaction) {
        self.normal_patterns
            .add(&self.encoder.encode_transaction(txn));
    }
    pub fn learn_fraud(&mut self, txn: &Transaction) {
        self.fraud_patterns
            .add(&self.encoder.encode_transaction(txn));
    }

    pub fn detect(&self, txn: &Transaction) -> Option<FraudAlert> {
        let hv = self.encoder.encode_transaction(txn);
        let fraud_sim = hv.similarity(&self.fraud_patterns.threshold());
        let normal_sim = hv.similarity(&self.normal_patterns.threshold());
        let score = fraud_sim - normal_sim;
        if score > self.threshold {
            Some(FraudAlert {
                transaction_id: txn.id.clone(),
                fraud_score: score,
                indicators: vec!["pattern_match".to_string()],
            })
        } else {
            None
        }
    }
}

impl Default for TransactionFraudDetector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_encoding() {
        let mut encoder = RetailLink::new();
        let product = Product {
            id: "P1".to_string(),
            name: "Laptop".to_string(),
            category: ProductCategory::Electronics,
            price: 999.99,
            brand: "TechBrand".to_string(),
            attributes: HashMap::new(),
            stock_level: 50,
        };
        assert_eq!(encoder.encode_product(&product).dimension(), DIMENSION);
    }

    #[test]
    fn test_customer_encoding() {
        let encoder = RetailLink::new();
        let customer = Customer {
            id: "C1".to_string(),
            segment: CustomerSegment::Loyal,
            total_orders: 25,
            total_spent: 2500.0,
            avg_order_value: 100.0,
            days_since_last_order: 7,
            preferred_categories: vec![ProductCategory::Electronics],
        };
        assert_eq!(encoder.encode_customer(&customer).dimension(), DIMENSION);
    }

    #[test]
    fn test_product_catalog() {
        let mut catalog = ProductCatalog::new();
        catalog.add_product(Product {
            id: "P1".to_string(),
            name: "Laptop".to_string(),
            category: ProductCategory::Electronics,
            price: 999.99,
            brand: "TechBrand".to_string(),
            attributes: HashMap::new(),
            stock_level: 50,
        });
        assert_eq!(catalog.product_count(), 1);
    }

    #[test]
    fn test_recommendation_engine() {
        let mut engine = RecommendationEngine::new();
        let customer = Customer {
            id: "C1".to_string(),
            segment: CustomerSegment::Regular,
            total_orders: 10,
            total_spent: 1000.0,
            avg_order_value: 100.0,
            days_since_last_order: 14,
            preferred_categories: vec![],
        };
        engine.register_customer(&customer);
        assert!(engine.purchase_history.contains_key("C1"));
    }

    #[test]
    fn test_fraud_detection() {
        let mut detector = TransactionFraudDetector::new(0.3);
        let txn = Transaction {
            id: "T1".to_string(),
            customer_id: "C1".to_string(),
            items: vec![("P1".to_string(), 1, 100.0)],
            total: 100.0,
            status: TransactionStatus::Completed,
            timestamp: 0,
            payment_method: "card".to_string(),
        };
        detector.learn_normal(&txn);
        assert!(detector.detect(&txn).is_none());
    }
}
