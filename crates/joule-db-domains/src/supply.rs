//! HDC-powered Logistics and Supply Chain module
//!
//! Provides holographic encoding for:
//! - Shipment tracking and similarity
//! - Route optimization
//! - Demand forecasting
//! - Supplier matching and risk assessment

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShipmentStatus {
    Created,
    PickedUp,
    InTransit,
    OutForDelivery,
    Delivered,
    Delayed,
    Lost,
    Returned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportMode {
    Air,
    Sea,
    Rail,
    Road,
    Multimodal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupplierTier {
    Strategic,
    Preferred,
    Approved,
    Probationary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProductClassification {
    Perishable,
    Hazardous,
    Fragile,
    Standard,
    HighValue,
    Bulk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shipment {
    pub id: String,
    pub origin: String,
    pub destination: String,
    pub status: ShipmentStatus,
    pub transport_mode: TransportMode,
    pub weight_kg: f64,
    pub volume_m3: f64,
    pub classification: ProductClassification,
    pub estimated_delivery: u64,
    pub actual_delivery: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Supplier {
    pub id: String,
    pub name: String,
    pub tier: SupplierTier,
    pub location: String,
    pub lead_time_days: u32,
    pub on_time_delivery_rate: f32,
    pub quality_score: f32,
    pub products: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Warehouse {
    pub id: String,
    pub location: String,
    pub capacity_m3: f64,
    pub utilization: f32,
    pub throughput_daily: u32,
    pub supported_modes: Vec<TransportMode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    pub origin: String,
    pub destination: String,
    pub waypoints: Vec<String>,
    pub distance_km: f64,
    pub avg_transit_hours: f32,
    pub cost_per_kg: f64,
    pub transport_mode: TransportMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandForecast {
    pub product_id: String,
    pub location: String,
    pub period: u64,
    pub quantity: u32,
    pub confidence: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for supply chain domain data
    pub struct SupplyLink {
        seed: 0x50B1_0001,
        dimension: 10000,
        fields: ["shipment", "supplier", "warehouse", "route", "location", "product", "forecast"],
        scalars: ["weight", "volume", "distance", "time", "cost", "quantity", "rate"],
        enums: {
            status_vectors: ShipmentStatus => [ShipmentStatus::Created, ShipmentStatus::PickedUp, ShipmentStatus::InTransit, ShipmentStatus::OutForDelivery, ShipmentStatus::Delivered, ShipmentStatus::Delayed, ShipmentStatus::Lost, ShipmentStatus::Returned],
            mode_vectors: TransportMode => [TransportMode::Air, TransportMode::Sea, TransportMode::Rail, TransportMode::Road, TransportMode::Multimodal],
            tier_vectors: SupplierTier => [SupplierTier::Strategic, SupplierTier::Preferred, SupplierTier::Approved, SupplierTier::Probationary],
            class_vectors: ProductClassification => [ProductClassification::Perishable, ProductClassification::Hazardous, ProductClassification::Fragile, ProductClassification::Standard, ProductClassification::HighValue, ProductClassification::Bulk]
        },
        dynamic: {
            location_vectors: "location"
        },
    }
}

impl SupplyLink {
    pub fn encode_shipment(&mut self, shipment: &Shipment) -> BinaryHV {
        let status_hv = self.field_vectors["shipment"].bind(&self.status_vectors[&shipment.status]);
        let mode_hv = self.mode_vectors[&shipment.transport_mode].clone();
        let class_hv = self.class_vectors[&shipment.classification].clone();
        let origin_vec = self.location_vectors(&shipment.origin);
        let origin_hv = self.field_vectors["location"].bind(&origin_vec);
        let dest_vec = self.location_vectors(&shipment.destination);
        let dest_hv = self.field_vectors["location"].bind(&dest_vec).permute(1);
        let weight_hv = self.encode_scalar("weight", (shipment.weight_kg * 10.0) as u32, 100000);
        let volume_hv = self.encode_scalar("volume", (shipment.volume_m3 * 100.0) as u32, 10000);
        self.bundle(&[
            status_hv, mode_hv, class_hv, origin_hv, dest_hv, weight_hv, volume_hv,
        ])
    }

    pub fn encode_supplier(&mut self, supplier: &Supplier) -> BinaryHV {
        let tier_hv = self.field_vectors["supplier"].bind(&self.tier_vectors[&supplier.tier]);
        let location_vec = self.location_vectors(&supplier.location);
        let location_hv = self.field_vectors["location"].bind(&location_vec);
        let lead_time_hv = self.encode_scalar("time", supplier.lead_time_days.min(365), 365);
        let otd_hv =
            self.encode_scalar("rate", (supplier.on_time_delivery_rate * 100.0) as u32, 100);
        let quality_hv = self.encode_scalar("rate", (supplier.quality_score * 100.0) as u32, 100);
        let mut components = vec![tier_hv, location_hv, lead_time_hv, otd_hv, quality_hv];
        for product in &supplier.products {
            components.push(
                self.field_vectors["product"]
                    .bind(&BinaryHV::from_hash(product.as_bytes(), DIMENSION)),
            );
        }
        self.bundle(&components)
    }

    pub fn encode_warehouse(&mut self, warehouse: &Warehouse) -> BinaryHV {
        let location_vec = self.location_vectors(&warehouse.location);
        let location_hv = self.field_vectors["location"].bind(&location_vec);
        let capacity_hv =
            self.encode_scalar("volume", (warehouse.capacity_m3 / 100.0) as u32, 10000);
        let util_hv = self.encode_scalar("rate", (warehouse.utilization * 100.0) as u32, 100);
        let throughput_hv =
            self.encode_scalar("quantity", warehouse.throughput_daily.min(10000), 10000);
        let mut components = vec![location_hv, capacity_hv, util_hv, throughput_hv];
        for mode in &warehouse.supported_modes {
            components.push(self.mode_vectors[mode].clone());
        }
        self.bundle(&components)
    }

    pub fn encode_route(&mut self, route: &Route) -> BinaryHV {
        let origin_vec = self.location_vectors(&route.origin);
        let origin_hv = self.field_vectors["location"].bind(&origin_vec);
        let dest_vec = self.location_vectors(&route.destination);
        let dest_hv = self.field_vectors["location"].bind(&dest_vec).permute(1);
        let mode_hv = self.mode_vectors[&route.transport_mode].clone();
        let distance_hv = self.encode_scalar("distance", route.distance_km as u32, 20000);
        let time_hv = self.encode_scalar("time", route.avg_transit_hours as u32, 720);
        let cost_hv = self.encode_scalar("cost", (route.cost_per_kg * 100.0) as u32, 10000);
        self.bundle(&[origin_hv, dest_hv, mode_hv, distance_hv, time_hv, cost_hv])
    }
}

pub struct ShipmentTracker {
    encoder: SupplyLink,
    shipment_vectors: HashMap<String, BinaryHV>,
    shipments: HashMap<String, Shipment>,
    delayed_patterns: BundleAccumulator,
}

impl ShipmentTracker {
    pub fn new() -> Self {
        Self {
            encoder: SupplyLink::new(),
            shipment_vectors: HashMap::new(),
            shipments: HashMap::new(),
            delayed_patterns: BundleAccumulator::new(DIMENSION),
        }
    }

    pub fn track(&mut self, shipment: Shipment) {
        let hv = self.encoder.encode_shipment(&shipment);
        if shipment.status == ShipmentStatus::Delayed {
            self.delayed_patterns.add(&hv);
        }
        self.shipment_vectors.insert(shipment.id.clone(), hv);
        self.shipments.insert(shipment.id.clone(), shipment);
    }

    pub fn find_similar(&self, shipment_id: &str, limit: usize) -> Vec<(String, f32)> {
        let query = match self.shipment_vectors.get(shipment_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .shipment_vectors
            .iter()
            .filter(|(id, _)| *id != shipment_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn predict_delay(&mut self, shipment: &Shipment) -> f32 {
        let hv = self.encoder.encode_shipment(shipment);
        hv.similarity(&self.delayed_patterns.threshold())
    }

    pub fn shipment_count(&self) -> usize {
        self.shipments.len()
    }
}

impl Default for ShipmentTracker {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SupplierMatcher {
    encoder: SupplyLink,
    supplier_vectors: HashMap<String, BinaryHV>,
    suppliers: HashMap<String, Supplier>,
}

impl SupplierMatcher {
    pub fn new() -> Self {
        Self {
            encoder: SupplyLink::new(),
            supplier_vectors: HashMap::new(),
            suppliers: HashMap::new(),
        }
    }

    pub fn add_supplier(&mut self, supplier: Supplier) {
        let hv = self.encoder.encode_supplier(&supplier);
        self.supplier_vectors.insert(supplier.id.clone(), hv);
        self.suppliers.insert(supplier.id.clone(), supplier);
    }

    pub fn find_matches(&mut self, requirements: &Supplier, limit: usize) -> Vec<(String, f32)> {
        let query = self.encoder.encode_supplier(requirements);
        let mut results: Vec<_> = self
            .supplier_vectors
            .iter()
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn supplier_count(&self) -> usize {
        self.suppliers.len()
    }
}

impl Default for SupplierMatcher {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RouteOptimizer {
    encoder: SupplyLink,
    route_vectors: HashMap<String, BinaryHV>,
    routes: HashMap<String, Route>,
}

impl RouteOptimizer {
    pub fn new() -> Self {
        Self {
            encoder: SupplyLink::new(),
            route_vectors: HashMap::new(),
            routes: HashMap::new(),
        }
    }

    pub fn add_route(&mut self, route: Route) {
        let hv = self.encoder.encode_route(&route);
        self.route_vectors.insert(route.id.clone(), hv);
        self.routes.insert(route.id.clone(), route);
    }

    pub fn find_similar_routes(&self, route_id: &str, limit: usize) -> Vec<(String, f32)> {
        let query = match self.route_vectors.get(route_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .route_vectors
            .iter()
            .filter(|(id, _)| *id != route_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

impl Default for RouteOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shipment_encoding() {
        let mut encoder = SupplyLink::new();
        let shipment = Shipment {
            id: "S1".to_string(),
            origin: "NYC".to_string(),
            destination: "LAX".to_string(),
            status: ShipmentStatus::InTransit,
            transport_mode: TransportMode::Air,
            weight_kg: 100.0,
            volume_m3: 0.5,
            classification: ProductClassification::Standard,
            estimated_delivery: 1000,
            actual_delivery: None,
        };
        assert_eq!(encoder.encode_shipment(&shipment).dimension(), DIMENSION);
    }

    #[test]
    fn test_supplier_encoding() {
        let mut encoder = SupplyLink::new();
        let supplier = Supplier {
            id: "SUP1".to_string(),
            name: "Acme".to_string(),
            tier: SupplierTier::Preferred,
            location: "Shanghai".to_string(),
            lead_time_days: 14,
            on_time_delivery_rate: 0.95,
            quality_score: 0.92,
            products: vec!["widgets".to_string()],
        };
        assert_eq!(encoder.encode_supplier(&supplier).dimension(), DIMENSION);
    }

    #[test]
    fn test_shipment_tracker() {
        let mut tracker = ShipmentTracker::new();
        tracker.track(Shipment {
            id: "S1".to_string(),
            origin: "NYC".to_string(),
            destination: "LAX".to_string(),
            status: ShipmentStatus::Delivered,
            transport_mode: TransportMode::Road,
            weight_kg: 50.0,
            volume_m3: 0.2,
            classification: ProductClassification::Standard,
            estimated_delivery: 1000,
            actual_delivery: Some(950),
        });
        assert_eq!(tracker.shipment_count(), 1);
    }

    #[test]
    fn test_supplier_matcher() {
        let mut matcher = SupplierMatcher::new();
        matcher.add_supplier(Supplier {
            id: "SUP1".to_string(),
            name: "Acme".to_string(),
            tier: SupplierTier::Strategic,
            location: "Shanghai".to_string(),
            lead_time_days: 7,
            on_time_delivery_rate: 0.98,
            quality_score: 0.95,
            products: vec![],
        });
        assert_eq!(matcher.supplier_count(), 1);
    }

    #[test]
    fn test_route_optimizer() {
        let mut optimizer = RouteOptimizer::new();
        optimizer.add_route(Route {
            id: "R1".to_string(),
            origin: "NYC".to_string(),
            destination: "CHI".to_string(),
            waypoints: vec![],
            distance_km: 1200.0,
            avg_transit_hours: 18.0,
            cost_per_kg: 0.5,
            transport_mode: TransportMode::Road,
        });
        assert_eq!(optimizer.route_count(), 1);
    }
}
