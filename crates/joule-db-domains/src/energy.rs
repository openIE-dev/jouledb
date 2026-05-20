//! HDC-powered Energy and Utilities module
//!
//! Provides holographic encoding for:
//! - Load forecasting and demand prediction
//! - Grid anomaly detection
//! - Renewable energy optimization
//! - Smart meter pattern analysis

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergySource {
    Solar,
    Wind,
    Hydro,
    Nuclear,
    NaturalGas,
    Coal,
    Geothermal,
    Battery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GridStatus {
    Normal,
    HighDemand,
    LowDemand,
    Critical,
    Maintenance,
    Outage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsumerType {
    Residential,
    Commercial,
    Industrial,
    Agricultural,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeOfUse {
    Peak,
    OffPeak,
    MidPeak,
    SuperOffPeak,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPlant {
    pub id: String,
    pub name: String,
    pub source: EnergySource,
    pub capacity_mw: f64,
    pub current_output_mw: f64,
    pub efficiency: f32,
    pub location: String,
    pub carbon_intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridNode {
    pub id: String,
    pub node_type: String,
    pub voltage_kv: f64,
    pub load_mw: f64,
    pub status: GridStatus,
    pub connected_nodes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartMeter {
    pub id: String,
    pub consumer_id: String,
    pub consumer_type: ConsumerType,
    pub location: String,
    pub readings: Vec<MeterReading>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeterReading {
    pub timestamp: u64,
    pub consumption_kwh: f64,
    pub voltage: f32,
    pub power_factor: f32,
    pub time_of_use: TimeOfUse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadForecast {
    pub region: String,
    pub timestamp: u64,
    pub predicted_load_mw: f64,
    pub confidence: f32,
    pub weather_factor: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for energy domain data
    pub struct EnergyLink {
        seed: 0xE0E1_0001,
        dimension: 10000,
        fields: ["plant", "grid", "meter", "reading", "location", "source", "forecast"],
        scalars: ["power", "voltage", "efficiency", "carbon", "load", "consumption"],
        enums: {
            source_vectors: EnergySource => [EnergySource::Solar, EnergySource::Wind, EnergySource::Hydro, EnergySource::Nuclear, EnergySource::NaturalGas, EnergySource::Coal, EnergySource::Geothermal, EnergySource::Battery],
            status_vectors: GridStatus => [GridStatus::Normal, GridStatus::HighDemand, GridStatus::LowDemand, GridStatus::Critical, GridStatus::Maintenance, GridStatus::Outage],
            consumer_vectors: ConsumerType => [ConsumerType::Residential, ConsumerType::Commercial, ConsumerType::Industrial, ConsumerType::Agricultural],
            tou_vectors: TimeOfUse => [TimeOfUse::Peak, TimeOfUse::OffPeak, TimeOfUse::MidPeak, TimeOfUse::SuperOffPeak]
        },
        dynamic: {
            location_vectors: "location"
        },
    }
}

impl EnergyLink {
    pub fn encode_power_plant(&mut self, plant: &PowerPlant) -> BinaryHV {
        let source_hv = self.field_vectors["source"].bind(&self.source_vectors[&plant.source]);
        let capacity_hv = self.encode_scalar("power", plant.capacity_mw as u32, 5000);
        let output_hv = self
            .encode_scalar("power", plant.current_output_mw as u32, 5000)
            .permute(1);
        let efficiency_hv =
            self.encode_scalar("efficiency", (plant.efficiency * 100.0) as u32, 100);
        let carbon_hv =
            self.encode_scalar("carbon", (plant.carbon_intensity * 1000.0) as u32, 1000);
        let location_vec = self.location_vectors(&plant.location);
        let location_hv = self.field_vectors["location"].bind(&location_vec);
        self.bundle(&[
            source_hv,
            capacity_hv,
            output_hv,
            efficiency_hv,
            carbon_hv,
            location_hv,
        ])
    }

    pub fn encode_grid_node(&mut self, node: &GridNode) -> BinaryHV {
        let status_hv = self.field_vectors["grid"].bind(&self.status_vectors[&node.status]);
        let voltage_hv = self.encode_scalar("voltage", node.voltage_kv as u32, 1000);
        let load_hv = self.encode_scalar("load", node.load_mw as u32, 10000);
        let type_hv = BinaryHV::from_hash(node.node_type.as_bytes(), DIMENSION);
        self.bundle(&[status_hv, voltage_hv, load_hv, type_hv])
    }

    pub fn encode_meter_reading(
        &self,
        reading: &MeterReading,
        consumer_type: ConsumerType,
    ) -> BinaryHV {
        let consumer_hv = self.consumer_vectors[&consumer_type].clone();
        let tou_hv = self.tou_vectors[&reading.time_of_use].clone();
        let consumption_hv = self.encode_scalar(
            "consumption",
            (reading.consumption_kwh * 10.0) as u32,
            10000,
        );
        let voltage_hv = self.encode_scalar("voltage", reading.voltage as u32, 500);
        let pf_hv = self.encode_scalar("efficiency", (reading.power_factor * 100.0) as u32, 100);
        self.bundle(&[consumer_hv, tou_hv, consumption_hv, voltage_hv, pf_hv])
    }

    pub fn encode_smart_meter(&mut self, meter: &SmartMeter) -> BinaryHV {
        let consumer_hv =
            self.field_vectors["meter"].bind(&self.consumer_vectors[&meter.consumer_type]);
        let location_vec = self.location_vectors(&meter.location);
        let location_hv = self.field_vectors["location"].bind(&location_vec);
        let mut components = vec![consumer_hv, location_hv];
        for reading in meter.readings.iter().take(10) {
            components.push(self.encode_meter_reading(reading, meter.consumer_type));
        }
        self.bundle(&components)
    }
}

pub struct GridMonitor {
    encoder: EnergyLink,
    node_vectors: HashMap<String, BinaryHV>,
    nodes: HashMap<String, GridNode>,
    anomaly_patterns: BundleAccumulator,
    normal_patterns: BundleAccumulator,
}

impl GridMonitor {
    pub fn new() -> Self {
        Self {
            encoder: EnergyLink::new(),
            node_vectors: HashMap::new(),
            nodes: HashMap::new(),
            anomaly_patterns: BundleAccumulator::new(DIMENSION),
            normal_patterns: BundleAccumulator::new(DIMENSION),
        }
    }

    pub fn register_node(&mut self, node: GridNode) {
        let hv = self.encoder.encode_grid_node(&node);
        match node.status {
            GridStatus::Normal | GridStatus::LowDemand => self.normal_patterns.add(&hv),
            GridStatus::Critical | GridStatus::Outage => self.anomaly_patterns.add(&hv),
            _ => {}
        }
        self.node_vectors.insert(node.id.clone(), hv);
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn detect_anomaly(&mut self, node: &GridNode) -> Option<f32> {
        let hv = self.encoder.encode_grid_node(node);
        let anomaly_sim = hv.similarity(&self.anomaly_patterns.threshold());
        let normal_sim = hv.similarity(&self.normal_patterns.threshold());
        let score = anomaly_sim - normal_sim;
        if score > 0.2 { Some(score) } else { None }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for GridMonitor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LoadForecaster {
    encoder: EnergyLink,
    high_load_patterns: BundleAccumulator,
    low_load_patterns: BundleAccumulator,
    reading_history: HashMap<ConsumerType, BundleAccumulator>,
}

impl LoadForecaster {
    pub fn new() -> Self {
        let mut reading_history = HashMap::new();
        for ct in [
            ConsumerType::Residential,
            ConsumerType::Commercial,
            ConsumerType::Industrial,
            ConsumerType::Agricultural,
        ] {
            reading_history.insert(ct, BundleAccumulator::new(DIMENSION));
        }
        Self {
            encoder: EnergyLink::new(),
            high_load_patterns: BundleAccumulator::new(DIMENSION),
            low_load_patterns: BundleAccumulator::new(DIMENSION),
            reading_history,
        }
    }

    pub fn learn_high_load(&mut self, reading: &MeterReading, consumer_type: ConsumerType) {
        let hv = self.encoder.encode_meter_reading(reading, consumer_type);
        self.high_load_patterns.add(&hv);
        self.reading_history
            .get_mut(&consumer_type)
            .unwrap()
            .add(&hv);
    }

    pub fn learn_low_load(&mut self, reading: &MeterReading, consumer_type: ConsumerType) {
        let hv = self.encoder.encode_meter_reading(reading, consumer_type);
        self.low_load_patterns.add(&hv);
        self.reading_history
            .get_mut(&consumer_type)
            .unwrap()
            .add(&hv);
    }

    pub fn predict_load(&self, reading: &MeterReading, consumer_type: ConsumerType) -> f32 {
        let hv = self.encoder.encode_meter_reading(reading, consumer_type);
        let high_sim = hv.similarity(&self.high_load_patterns.threshold());
        let low_sim = hv.similarity(&self.low_load_patterns.threshold());
        (high_sim - low_sim + 1.0) / 2.0
    }
}

impl Default for LoadForecaster {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RenewableOptimizer {
    encoder: EnergyLink,
    plant_vectors: HashMap<String, BinaryHV>,
    plants: HashMap<String, PowerPlant>,
    renewable_sources: Vec<EnergySource>,
}

impl RenewableOptimizer {
    pub fn new() -> Self {
        Self {
            encoder: EnergyLink::new(),
            plant_vectors: HashMap::new(),
            plants: HashMap::new(),
            renewable_sources: vec![
                EnergySource::Solar,
                EnergySource::Wind,
                EnergySource::Hydro,
                EnergySource::Geothermal,
            ],
        }
    }

    pub fn add_plant(&mut self, plant: PowerPlant) {
        let hv = self.encoder.encode_power_plant(&plant);
        self.plant_vectors.insert(plant.id.clone(), hv);
        self.plants.insert(plant.id.clone(), plant);
    }

    pub fn find_similar_plants(&self, plant_id: &str, limit: usize) -> Vec<(String, f32)> {
        let query = match self.plant_vectors.get(plant_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .plant_vectors
            .iter()
            .filter(|(id, _)| *id != plant_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn get_renewable_capacity(&self) -> f64 {
        self.plants
            .values()
            .filter(|p| self.renewable_sources.contains(&p.source))
            .map(|p| p.capacity_mw)
            .sum()
    }

    pub fn plant_count(&self) -> usize {
        self.plants.len()
    }
}

impl Default for RenewableOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_plant_encoding() {
        let mut encoder = EnergyLink::new();
        let plant = PowerPlant {
            id: "PP1".to_string(),
            name: "Solar Farm".to_string(),
            source: EnergySource::Solar,
            capacity_mw: 100.0,
            current_output_mw: 75.0,
            efficiency: 0.22,
            location: "Arizona".to_string(),
            carbon_intensity: 0.0,
        };
        assert_eq!(encoder.encode_power_plant(&plant).dimension(), DIMENSION);
    }

    #[test]
    fn test_grid_node_encoding() {
        let mut encoder = EnergyLink::new();
        let node = GridNode {
            id: "N1".to_string(),
            node_type: "substation".to_string(),
            voltage_kv: 230.0,
            load_mw: 500.0,
            status: GridStatus::Normal,
            connected_nodes: vec![],
        };
        assert_eq!(encoder.encode_grid_node(&node).dimension(), DIMENSION);
    }

    #[test]
    fn test_grid_monitor() {
        let mut monitor = GridMonitor::new();
        monitor.register_node(GridNode {
            id: "N1".to_string(),
            node_type: "substation".to_string(),
            voltage_kv: 230.0,
            load_mw: 500.0,
            status: GridStatus::Normal,
            connected_nodes: vec![],
        });
        assert_eq!(monitor.node_count(), 1);
    }

    #[test]
    fn test_load_forecaster() {
        let mut forecaster = LoadForecaster::new();
        let reading = MeterReading {
            timestamp: 0,
            consumption_kwh: 50.0,
            voltage: 240.0,
            power_factor: 0.95,
            time_of_use: TimeOfUse::Peak,
        };
        forecaster.learn_high_load(&reading, ConsumerType::Residential);
        let prediction = forecaster.predict_load(&reading, ConsumerType::Residential);
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }

    #[test]
    fn test_renewable_optimizer() {
        let mut optimizer = RenewableOptimizer::new();
        optimizer.add_plant(PowerPlant {
            id: "PP1".to_string(),
            name: "Wind Farm".to_string(),
            source: EnergySource::Wind,
            capacity_mw: 200.0,
            current_output_mw: 150.0,
            efficiency: 0.35,
            location: "Texas".to_string(),
            carbon_intensity: 0.0,
        });
        assert_eq!(optimizer.get_renewable_capacity(), 200.0);
    }
}
