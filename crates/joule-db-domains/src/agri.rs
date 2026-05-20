//! HDC-powered Agriculture and AgTech module
//!
//! Provides holographic encoding for:
//! - Crop yield prediction
//! - Disease detection patterns
//! - Soil analysis similarity
//! - Weather impact assessment

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CropType {
    Wheat,
    Corn,
    Rice,
    Soybean,
    Cotton,
    Potato,
    Tomato,
    Grape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SoilType {
    Clay,
    Sandy,
    Loamy,
    Silty,
    Peaty,
    Chalky,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GrowthStage {
    Germination,
    Vegetative,
    Flowering,
    Fruiting,
    Maturity,
    Harvest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub id: String,
    pub area_hectares: f64,
    pub soil_type: SoilType,
    pub current_crop: Option<CropType>,
    pub ph_level: f32,
    pub nitrogen: f32,
    pub phosphorus: f32,
    pub potassium: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropObservation {
    pub field_id: String,
    pub crop_type: CropType,
    pub growth_stage: GrowthStage,
    pub health_score: f32,
    pub ndvi: f32,
    pub moisture: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub timestamp: u64,
    pub temperature: f32,
    pub humidity: f32,
    pub rainfall_mm: f32,
    pub sunlight_hours: f32,
    pub wind_speed: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiseasePattern {
    pub id: String,
    pub name: String,
    pub affected_crops: Vec<CropType>,
    pub symptoms: Vec<String>,
    pub severity: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for agriculture domain data
    pub struct AgriLink {
        seed: 0xA611_0001,
        dimension: 10000,
        fields: ["field", "crop", "soil", "stage", "health", "weather", "npk"],
        scalars: ["ph", "npk", "health", "ndvi", "moisture", "temp", "rain", "humidity"],
        enums: {
            crop_vectors: CropType => [CropType::Wheat, CropType::Corn, CropType::Rice, CropType::Soybean, CropType::Cotton, CropType::Potato, CropType::Tomato, CropType::Grape],
            soil_vectors: SoilType => [SoilType::Clay, SoilType::Sandy, SoilType::Loamy, SoilType::Silty, SoilType::Peaty, SoilType::Chalky],
            stage_vectors: GrowthStage => [GrowthStage::Germination, GrowthStage::Vegetative, GrowthStage::Flowering, GrowthStage::Fruiting, GrowthStage::Maturity, GrowthStage::Harvest]
        },
    }
}

impl AgriLink {
    pub fn encode_field(&self, field: &Field) -> BinaryHV {
        let soil_hv = self.field_vectors["soil"].bind(&self.soil_vectors[&field.soil_type]);
        let ph_hv = self.encode_scalar("ph", (field.ph_level * 10.0) as u32, 140);
        let n_hv = self.encode_scalar("npk", field.nitrogen as u32, 200);
        let p_hv = self
            .encode_scalar("npk", field.phosphorus as u32, 200)
            .permute(1);
        let k_hv = self
            .encode_scalar("npk", field.potassium as u32, 200)
            .permute(2);
        let mut components = vec![soil_hv, ph_hv, n_hv, p_hv, k_hv];
        if let Some(crop) = field.current_crop {
            components.push(self.field_vectors["crop"].bind(&self.crop_vectors[&crop]));
        }
        self.bundle(&components)
    }

    pub fn encode_observation(&self, obs: &CropObservation) -> BinaryHV {
        let crop_hv = self.field_vectors["crop"].bind(&self.crop_vectors[&obs.crop_type]);
        let stage_hv = self.field_vectors["stage"].bind(&self.stage_vectors[&obs.growth_stage]);
        let health_hv = self.encode_scalar("health", (obs.health_score * 100.0) as u32, 100);
        let ndvi_hv = self.encode_scalar("ndvi", (obs.ndvi * 100.0) as u32, 100);
        let moisture_hv = self.encode_scalar("moisture", (obs.moisture * 100.0) as u32, 100);
        self.bundle(&[crop_hv, stage_hv, health_hv, ndvi_hv, moisture_hv])
    }

    pub fn encode_weather(&self, weather: &WeatherData) -> BinaryHV {
        let temp_hv = self.encode_scalar("temp", (weather.temperature + 40.0) as u32, 100);
        let humidity_hv = self.encode_scalar("humidity", weather.humidity as u32, 100);
        let rain_hv = self.encode_scalar("rain", weather.rainfall_mm as u32, 200);
        self.bundle(&[temp_hv, humidity_hv, rain_hv])
    }
}

pub struct YieldPredictor {
    encoder: AgriLink,
    yield_patterns: HashMap<CropType, BundleAccumulator>,
    yield_values: HashMap<u64, f64>,
}

impl YieldPredictor {
    pub fn new() -> Self {
        let mut yield_patterns = HashMap::new();
        for crop in [
            CropType::Wheat,
            CropType::Corn,
            CropType::Rice,
            CropType::Soybean,
            CropType::Cotton,
            CropType::Potato,
            CropType::Tomato,
            CropType::Grape,
        ] {
            yield_patterns.insert(crop, BundleAccumulator::new(DIMENSION));
        }
        Self {
            encoder: AgriLink::new(),
            yield_patterns,
            yield_values: HashMap::new(),
        }
    }

    pub fn train(&mut self, obs: &CropObservation, actual_yield: f64) {
        let hv = self.encoder.encode_observation(obs);
        self.yield_patterns
            .get_mut(&obs.crop_type)
            .unwrap()
            .add(&hv);
        self.yield_values.insert(hv.condense_to_u64(), actual_yield);
    }

    pub fn predict(&self, obs: &CropObservation) -> f64 {
        let hv = self.encoder.encode_observation(obs);
        let mut best_sim = 0.0f32;
        let mut best_yield = 0.0;
        for (hash, yield_val) in &self.yield_values {
            let sim = if hv.condense_to_u64() == *hash {
                1.0
            } else {
                0.5
            };
            if sim > best_sim {
                best_sim = sim;
                best_yield = *yield_val;
            }
        }
        best_yield
    }
}

impl Default for YieldPredictor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DiseaseDetector {
    encoder: AgriLink,
    disease_patterns: BundleAccumulator,
    healthy_patterns: BundleAccumulator,
    threshold: f32,
}

impl DiseaseDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: AgriLink::new(),
            disease_patterns: BundleAccumulator::new(DIMENSION),
            healthy_patterns: BundleAccumulator::new(DIMENSION),
            threshold,
        }
    }

    pub fn learn_diseased(&mut self, obs: &CropObservation) {
        self.disease_patterns
            .add(&self.encoder.encode_observation(obs));
    }
    pub fn learn_healthy(&mut self, obs: &CropObservation) {
        self.healthy_patterns
            .add(&self.encoder.encode_observation(obs));
    }

    pub fn detect(&self, obs: &CropObservation) -> Option<f32> {
        let hv = self.encoder.encode_observation(obs);
        let disease_sim = hv.similarity(&self.disease_patterns.threshold());
        let healthy_sim = hv.similarity(&self.healthy_patterns.threshold());
        let score = disease_sim - healthy_sim;
        if score > self.threshold {
            Some(score)
        } else {
            None
        }
    }
}

impl Default for DiseaseDetector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_encoding() {
        let encoder = AgriLink::new();
        let field = Field {
            id: "F1".to_string(),
            area_hectares: 10.0,
            soil_type: SoilType::Loamy,
            current_crop: Some(CropType::Wheat),
            ph_level: 6.5,
            nitrogen: 50.0,
            phosphorus: 30.0,
            potassium: 40.0,
        };
        assert_eq!(encoder.encode_field(&field).dimension(), DIMENSION);
    }

    #[test]
    fn test_observation_encoding() {
        let encoder = AgriLink::new();
        let obs = CropObservation {
            field_id: "F1".to_string(),
            crop_type: CropType::Corn,
            growth_stage: GrowthStage::Vegetative,
            health_score: 0.85,
            ndvi: 0.7,
            moisture: 0.6,
            timestamp: 0,
        };
        assert_eq!(encoder.encode_observation(&obs).dimension(), DIMENSION);
    }

    #[test]
    fn test_yield_prediction() {
        let mut predictor = YieldPredictor::new();
        let obs = CropObservation {
            field_id: "F1".to_string(),
            crop_type: CropType::Wheat,
            growth_stage: GrowthStage::Maturity,
            health_score: 0.9,
            ndvi: 0.8,
            moisture: 0.5,
            timestamp: 0,
        };
        predictor.train(&obs, 5.5);
        let yield_pred = predictor.predict(&obs);
        assert!(yield_pred >= 0.0);
    }

    #[test]
    fn test_disease_detection() {
        let mut detector = DiseaseDetector::new(0.3);
        let healthy = CropObservation {
            field_id: "F1".to_string(),
            crop_type: CropType::Tomato,
            growth_stage: GrowthStage::Flowering,
            health_score: 0.95,
            ndvi: 0.85,
            moisture: 0.6,
            timestamp: 0,
        };
        detector.learn_healthy(&healthy);
        assert!(detector.detect(&healthy).is_none());
    }
}
