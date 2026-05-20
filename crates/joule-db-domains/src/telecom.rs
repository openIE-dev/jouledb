//! HDC-powered Telecom and Networks module
//!
//! Provides holographic encoding for:
//! - Network traffic analysis and anomaly detection
//! - Customer churn prediction
//! - Service quality monitoring
//! - Cell tower coverage optimization

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkType {
    LTE4G,
    NR5G,
    WiFi,
    Fiber,
    DSL,
    Cable,
    Satellite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceType {
    Voice,
    Data,
    SMS,
    Video,
    IoT,
    Enterprise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CustomerSegment {
    Consumer,
    SMB,
    Enterprise,
    Government,
    Wholesale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlarmSeverity {
    Critical,
    Major,
    Minor,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellTower {
    pub id: String,
    pub location: String,
    pub lat: f64,
    pub lon: f64,
    pub network_type: NetworkType,
    pub capacity_mbps: u32,
    pub current_load_mbps: u32,
    pub connected_users: u32,
    pub coverage_radius_km: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSession {
    pub id: String,
    pub subscriber_id: String,
    pub tower_id: String,
    pub service_type: ServiceType,
    pub duration_secs: u32,
    pub data_uploaded_mb: f64,
    pub data_downloaded_mb: f64,
    pub avg_latency_ms: f32,
    pub packet_loss_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscriber {
    pub id: String,
    pub segment: CustomerSegment,
    pub plan_type: String,
    pub tenure_months: u32,
    pub monthly_usage_gb: f64,
    pub avg_monthly_bill: f64,
    pub support_tickets: u32,
    pub services: Vec<ServiceType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAlarm {
    pub id: String,
    pub source_id: String,
    pub severity: AlarmSeverity,
    pub alarm_type: String,
    pub timestamp: u64,
    pub description: String,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub tower_id: String,
    pub timestamp: u64,
    pub signal_strength_dbm: f32,
    pub throughput_mbps: f32,
    pub latency_ms: f32,
    pub jitter_ms: f32,
    pub call_drop_rate: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for telecom domain data
    pub struct TelecomLink {
        seed: 0x7E1E_0001,
        dimension: 10000,
        fields: ["tower", "session", "subscriber", "alarm", "quality", "location", "service"],
        scalars: ["bandwidth", "latency", "usage", "duration", "count", "signal", "rate"],
        enums: {
            network_type_vectors: NetworkType => [NetworkType::LTE4G, NetworkType::NR5G, NetworkType::WiFi, NetworkType::Fiber, NetworkType::DSL, NetworkType::Cable, NetworkType::Satellite],
            service_type_vectors: ServiceType => [ServiceType::Voice, ServiceType::Data, ServiceType::SMS, ServiceType::Video, ServiceType::IoT, ServiceType::Enterprise],
            segment_vectors: CustomerSegment => [CustomerSegment::Consumer, CustomerSegment::SMB, CustomerSegment::Enterprise, CustomerSegment::Government, CustomerSegment::Wholesale],
            severity_vectors: AlarmSeverity => [AlarmSeverity::Critical, AlarmSeverity::Major, AlarmSeverity::Minor, AlarmSeverity::Warning, AlarmSeverity::Info]
        },
        dynamic: {
            location_vectors: "location"
        },
    }
}

impl TelecomLink {
    pub fn encode_cell_tower(&mut self, tower: &CellTower) -> BinaryHV {
        let network_hv =
            self.field_vectors["tower"].bind(&self.network_type_vectors[&tower.network_type]);
        let location_vec = self.location_vectors(&tower.location);
        let location_hv = self.field_vectors["location"].bind(&location_vec);
        let capacity_hv = self.encode_scalar("bandwidth", tower.capacity_mbps, 10000);
        let load_hv = self
            .encode_scalar("bandwidth", tower.current_load_mbps, 10000)
            .permute(1);
        let users_hv = self.encode_scalar("count", tower.connected_users.min(10000), 10000);
        let coverage_hv =
            self.encode_scalar("signal", (tower.coverage_radius_km * 10.0) as u32, 500);
        self.bundle(&[
            network_hv,
            location_hv,
            capacity_hv,
            load_hv,
            users_hv,
            coverage_hv,
        ])
    }

    pub fn encode_network_session(&self, session: &NetworkSession) -> BinaryHV {
        let service_hv =
            self.field_vectors["service"].bind(&self.service_type_vectors[&session.service_type]);
        let duration_hv = self.encode_scalar("duration", session.duration_secs.min(86400), 86400);
        let download_hv =
            self.encode_scalar("usage", (session.data_downloaded_mb * 10.0) as u32, 100000);
        let upload_hv = self
            .encode_scalar("usage", (session.data_uploaded_mb * 10.0) as u32, 100000)
            .permute(1);
        let latency_hv = self.encode_scalar("latency", session.avg_latency_ms as u32, 1000);
        let loss_hv =
            self.encode_scalar("rate", (session.packet_loss_rate * 10000.0) as u32, 10000);
        self.bundle(&[
            service_hv,
            duration_hv,
            download_hv,
            upload_hv,
            latency_hv,
            loss_hv,
        ])
    }

    pub fn encode_subscriber(&self, subscriber: &Subscriber) -> BinaryHV {
        let segment_hv =
            self.field_vectors["subscriber"].bind(&self.segment_vectors[&subscriber.segment]);
        let tenure_hv = self.encode_scalar("duration", subscriber.tenure_months.min(240), 240);
        let usage_hv =
            self.encode_scalar("usage", (subscriber.monthly_usage_gb * 10.0) as u32, 10000);
        let bill_hv = self.encode_scalar("rate", subscriber.avg_monthly_bill as u32, 1000);
        let tickets_hv = self.encode_scalar("count", subscriber.support_tickets.min(100), 100);
        let mut components = vec![segment_hv, tenure_hv, usage_hv, bill_hv, tickets_hv];
        for service in &subscriber.services {
            components
                .push(self.field_vectors["service"].bind(&self.service_type_vectors[service]));
        }
        self.bundle(&components)
    }

    pub fn encode_alarm(&self, alarm: &NetworkAlarm) -> BinaryHV {
        let severity_hv = self.field_vectors["alarm"].bind(&self.severity_vectors[&alarm.severity]);
        let type_hv = BinaryHV::from_hash(alarm.alarm_type.as_bytes(), DIMENSION);
        let desc_hv = BinaryHV::from_hash(alarm.description.as_bytes(), DIMENSION);
        self.bundle(&[severity_hv, type_hv, desc_hv])
    }

    pub fn encode_quality_metrics(&self, metrics: &QualityMetrics) -> BinaryHV {
        let signal_hv =
            self.encode_scalar("signal", (metrics.signal_strength_dbm + 120.0) as u32, 120);
        let throughput_hv = self.encode_scalar("bandwidth", metrics.throughput_mbps as u32, 1000);
        let latency_hv = self.encode_scalar("latency", metrics.latency_ms as u32, 500);
        let jitter_hv = self.encode_scalar("latency", metrics.jitter_ms as u32, 100);
        let drop_hv = self.encode_scalar("rate", (metrics.call_drop_rate * 10000.0) as u32, 10000);
        self.bundle(&[signal_hv, throughput_hv, latency_hv, jitter_hv, drop_hv])
    }
}

pub struct NetworkMonitor {
    encoder: TelecomLink,
    tower_vectors: HashMap<String, BinaryHV>,
    towers: HashMap<String, CellTower>,
    anomaly_patterns: BundleAccumulator,
    normal_patterns: BundleAccumulator,
}

impl NetworkMonitor {
    pub fn new() -> Self {
        Self {
            encoder: TelecomLink::new(),
            tower_vectors: HashMap::new(),
            towers: HashMap::new(),
            anomaly_patterns: BundleAccumulator::new(DIMENSION),
            normal_patterns: BundleAccumulator::new(DIMENSION),
        }
    }

    pub fn register_tower(&mut self, tower: CellTower) {
        let hv = self.encoder.encode_cell_tower(&tower);
        self.tower_vectors.insert(tower.id.clone(), hv);
        self.towers.insert(tower.id.clone(), tower);
    }

    pub fn learn_normal_quality(&mut self, metrics: &QualityMetrics) {
        self.normal_patterns
            .add(&self.encoder.encode_quality_metrics(metrics));
    }

    pub fn learn_anomaly_quality(&mut self, metrics: &QualityMetrics) {
        self.anomaly_patterns
            .add(&self.encoder.encode_quality_metrics(metrics));
    }

    pub fn detect_anomaly(&self, metrics: &QualityMetrics) -> Option<f32> {
        let hv = self.encoder.encode_quality_metrics(metrics);
        let anomaly_sim = hv.similarity(&self.anomaly_patterns.threshold());
        let normal_sim = hv.similarity(&self.normal_patterns.threshold());
        let score = anomaly_sim - normal_sim;
        if score > 0.2 { Some(score) } else { None }
    }

    pub fn tower_count(&self) -> usize {
        self.towers.len()
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ChurnPredictor {
    encoder: TelecomLink,
    churned_patterns: BundleAccumulator,
    retained_patterns: BundleAccumulator,
    threshold: f32,
}

#[derive(Debug, Clone)]
pub struct ChurnRisk {
    pub subscriber_id: String,
    pub churn_score: f32,
    pub risk_factors: Vec<String>,
}

impl ChurnPredictor {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: TelecomLink::new(),
            churned_patterns: BundleAccumulator::new(DIMENSION),
            retained_patterns: BundleAccumulator::new(DIMENSION),
            threshold,
        }
    }

    pub fn learn_churned(&mut self, subscriber: &Subscriber) {
        self.churned_patterns
            .add(&self.encoder.encode_subscriber(subscriber));
    }
    pub fn learn_retained(&mut self, subscriber: &Subscriber) {
        self.retained_patterns
            .add(&self.encoder.encode_subscriber(subscriber));
    }

    pub fn predict(&self, subscriber: &Subscriber) -> Option<ChurnRisk> {
        let hv = self.encoder.encode_subscriber(subscriber);
        let churn_sim = hv.similarity(&self.churned_patterns.threshold());
        let retain_sim = hv.similarity(&self.retained_patterns.threshold());
        let score = churn_sim - retain_sim;
        if score > self.threshold {
            Some(ChurnRisk {
                subscriber_id: subscriber.id.clone(),
                churn_score: score,
                risk_factors: vec!["pattern_match".to_string()],
            })
        } else {
            None
        }
    }
}

impl Default for ChurnPredictor {
    fn default() -> Self {
        Self::new(0.3)
    }
}

pub struct CoverageOptimizer {
    encoder: TelecomLink,
    tower_vectors: HashMap<String, BinaryHV>,
    towers: HashMap<String, CellTower>,
}

impl CoverageOptimizer {
    pub fn new() -> Self {
        Self {
            encoder: TelecomLink::new(),
            tower_vectors: HashMap::new(),
            towers: HashMap::new(),
        }
    }

    pub fn add_tower(&mut self, tower: CellTower) {
        let hv = self.encoder.encode_cell_tower(&tower);
        self.tower_vectors.insert(tower.id.clone(), hv);
        self.towers.insert(tower.id.clone(), tower);
    }

    pub fn find_similar_towers(&self, tower_id: &str, limit: usize) -> Vec<(String, f32)> {
        let query = match self.tower_vectors.get(tower_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .tower_vectors
            .iter()
            .filter(|(id, _)| *id != tower_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn get_overloaded_towers(&self, threshold: f32) -> Vec<String> {
        self.towers
            .iter()
            .filter(|(_, t)| {
                let load_ratio = t.current_load_mbps as f32 / t.capacity_mbps as f32;
                load_ratio > threshold
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn tower_count(&self) -> usize {
        self.towers.len()
    }
}

impl Default for CoverageOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_tower_encoding() {
        let mut encoder = TelecomLink::new();
        let tower = CellTower {
            id: "T1".to_string(),
            location: "NYC".to_string(),
            lat: 40.7128,
            lon: -74.0060,
            network_type: NetworkType::NR5G,
            capacity_mbps: 5000,
            current_load_mbps: 3000,
            connected_users: 500,
            coverage_radius_km: 2.5,
        };
        assert_eq!(encoder.encode_cell_tower(&tower).dimension(), DIMENSION);
    }

    #[test]
    fn test_subscriber_encoding() {
        let encoder = TelecomLink::new();
        let subscriber = Subscriber {
            id: "S1".to_string(),
            segment: CustomerSegment::Consumer,
            plan_type: "Unlimited".to_string(),
            tenure_months: 24,
            monthly_usage_gb: 50.0,
            avg_monthly_bill: 80.0,
            support_tickets: 2,
            services: vec![ServiceType::Voice, ServiceType::Data],
        };
        assert_eq!(
            encoder.encode_subscriber(&subscriber).dimension(),
            DIMENSION
        );
    }

    #[test]
    fn test_network_monitor() {
        let mut monitor = NetworkMonitor::new();
        monitor.register_tower(CellTower {
            id: "T1".to_string(),
            location: "NYC".to_string(),
            lat: 40.7128,
            lon: -74.0060,
            network_type: NetworkType::LTE4G,
            capacity_mbps: 1000,
            current_load_mbps: 500,
            connected_users: 200,
            coverage_radius_km: 5.0,
        });
        assert_eq!(monitor.tower_count(), 1);
    }

    #[test]
    fn test_churn_predictor() {
        let mut predictor = ChurnPredictor::new(0.3);
        let retained = Subscriber {
            id: "S1".to_string(),
            segment: CustomerSegment::Consumer,
            plan_type: "Basic".to_string(),
            tenure_months: 36,
            monthly_usage_gb: 30.0,
            avg_monthly_bill: 50.0,
            support_tickets: 0,
            services: vec![ServiceType::Data],
        };
        predictor.learn_retained(&retained);
        assert!(predictor.predict(&retained).is_none());
    }

    #[test]
    fn test_coverage_optimizer() {
        let mut optimizer = CoverageOptimizer::new();
        optimizer.add_tower(CellTower {
            id: "T1".to_string(),
            location: "LAX".to_string(),
            lat: 34.0522,
            lon: -118.2437,
            network_type: NetworkType::NR5G,
            capacity_mbps: 8000,
            current_load_mbps: 7500,
            connected_users: 1000,
            coverage_radius_km: 1.5,
        });
        let overloaded = optimizer.get_overloaded_towers(0.9);
        assert_eq!(overloaded.len(), 1);
    }
}
