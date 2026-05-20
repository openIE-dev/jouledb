//! JouleDB IoT Link
//!
//! HDC-powered IoT and Industrial Sensor Fusion module.
//! Provides real-time sensor aggregation, anomaly detection, and predictive maintenance.

pub use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::{HashMap, VecDeque};

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

/// Sensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SensorType {
    Temperature,
    Pressure,
    Humidity,
    Vibration,
    Current,
    Voltage,
    Flow,
    Level,
    Speed,
    Position,
    Proximity,
    Light,
    Sound,
    Gas,
    Accelerometer,
    Gyroscope,
    Strain,
    Force,
    Torque,
    Custom,
}

/// A sensor reading
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SensorReading {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub value: f64,
    pub unit: String,
    pub quality: DataQuality,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DataQuality {
    Good,
    Uncertain,
    Bad,
    Stale,
}

/// Sensor metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SensorMetadata {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub location: Location,
    pub asset_id: String,
    pub min_value: f64,
    pub max_value: f64,
    pub unit: String,
    pub sample_rate_hz: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Location {
    pub zone: String,
    pub building: Option<String>,
    pub floor: Option<String>,
    pub room: Option<String>,
    pub coordinates: Option<(f64, f64, f64)>,
}

/// An industrial asset (machine, equipment)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Asset {
    pub asset_id: String,
    pub name: String,
    pub asset_type: String,
    pub location: Location,
    pub sensors: Vec<String>,
    pub status: AssetStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AssetStatus {
    Running,
    Idle,
    Maintenance,
    Fault,
    Offline,
}

/// A maintenance event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaintenanceEvent {
    pub event_id: String,
    pub asset_id: String,
    pub event_type: MaintenanceType,
    pub description: String,
    pub sensor_readings_before: Vec<SensorReading>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MaintenanceType {
    Scheduled,
    Corrective,
    Predictive,
    Emergency,
}

/// An alert/alarm
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Alert {
    pub alert_id: String,
    pub sensor_id: String,
    pub asset_id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub value: f64,
    pub threshold: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

// ============================================================================
// IoT Link Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder for IoT and industrial sensor data
    pub struct IotLink {
        seed: 0x1007_5E50,
        dimension: 10000,
        fields: ["sensor_id", "sensor_type", "value", "quality", "location",
                 "asset", "status", "severity", "threshold"],
        scalars: ["value", "threshold", "rate"],
        enums: {
            sensor_type_vectors: SensorType => [SensorType::Temperature, SensorType::Pressure, SensorType::Humidity, SensorType::Vibration, SensorType::Current, SensorType::Voltage, SensorType::Flow, SensorType::Level, SensorType::Speed, SensorType::Position, SensorType::Proximity, SensorType::Light, SensorType::Sound, SensorType::Gas, SensorType::Accelerometer, SensorType::Gyroscope, SensorType::Strain, SensorType::Force, SensorType::Torque, SensorType::Custom],
            quality_vectors: DataQuality => [DataQuality::Good, DataQuality::Uncertain, DataQuality::Bad, DataQuality::Stale],
            status_vectors: AssetStatus => [AssetStatus::Running, AssetStatus::Idle, AssetStatus::Maintenance, AssetStatus::Fault, AssetStatus::Offline],
            severity_vectors: AlertSeverity => [AlertSeverity::Info, AlertSeverity::Warning, AlertSeverity::Critical, AlertSeverity::Emergency],
            maintenance_vectors: MaintenanceType => [MaintenanceType::Scheduled, MaintenanceType::Corrective, MaintenanceType::Predictive, MaintenanceType::Emergency]
        },
    }
}

impl IotLink {
    /// Encode a sensor reading
    pub fn encode_reading(&self, reading: &SensorReading) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Sensor ID
        let sensor_hv = BinaryHV::from_hash(reading.sensor_id.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["sensor_id"].bind(&sensor_hv));

        // Sensor type
        acc.add(
            &self.field_vectors["sensor_type"]
                .bind(&self.sensor_type_vectors[&reading.sensor_type]),
        );

        // Value: integer part directly as shift to preserve monotonicity for
        // typical sensor ranges (0-156). Values beyond 156 wrap, which is
        // acceptable for coarse similarity.
        let value_shift = (reading.value.abs()) as usize % 157;
        let value_vec = self.scalar_bases["value"].permute_words(value_shift);
        acc.add(&self.field_vectors["value"].bind(&value_vec));

        // Quality
        acc.add(&self.field_vectors["quality"].bind(&self.quality_vectors[&reading.quality]));

        acc.threshold()
    }

    /// Encode multiple sensor readings into a fused state vector
    pub fn encode_sensor_fusion(&self, readings: &[SensorReading]) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        for reading in readings {
            acc.add(&self.encode_reading(reading));
        }

        acc.threshold()
    }

    /// Encode an asset state
    pub fn encode_asset(&self, asset: &Asset) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Asset ID
        let asset_hv = BinaryHV::from_hash(asset.asset_id.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["asset"].bind(&asset_hv));

        // Status
        acc.add(&self.field_vectors["status"].bind(&self.status_vectors[&asset.status]));

        // Location
        let loc_hv = BinaryHV::from_hash(asset.location.zone.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["location"].bind(&loc_hv));

        acc.threshold()
    }

    /// Encode an alert
    pub fn encode_alert(&self, alert: &Alert) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Sensor
        let sensor_hv = BinaryHV::from_hash(alert.sensor_id.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["sensor_id"].bind(&sensor_hv));

        // Severity
        acc.add(&self.field_vectors["severity"].bind(&self.severity_vectors[&alert.severity]));

        // Value deviation from threshold
        let deviation = (alert.value - alert.threshold).abs() / alert.threshold.abs().max(1.0);
        let dev_shift = (deviation * 100.0) as usize % 157;
        let dev_vec = self.scalar_bases["threshold"].permute_words(dev_shift);
        acc.add(&self.field_vectors["threshold"].bind(&dev_vec));

        acc.threshold()
    }

    /// Encode a maintenance event (for failure pattern learning)
    pub fn encode_maintenance(&self, event: &MaintenanceEvent) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Asset
        let asset_hv = BinaryHV::from_hash(event.asset_id.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["asset"].bind(&asset_hv));

        // Maintenance type
        acc.add(&self.maintenance_vectors[&event.event_type]);

        // Pre-failure sensor state
        if !event.sensor_readings_before.is_empty() {
            let fusion_hv = self.encode_sensor_fusion(&event.sensor_readings_before);
            acc.add(&fusion_hv);
        }

        acc.threshold()
    }
}

// ============================================================================
// Sensor Fusion Engine
// ============================================================================

/// Real-time sensor fusion with holographic state
pub struct SensorFusion {
    /// Current fused state per asset
    asset_states: HashMap<String, BundleAccumulator>,
    /// Latest readings per sensor
    latest_readings: HashMap<String, SensorReading>,
    /// Encoder
    encoder: IotLink,
}

impl SensorFusion {
    pub fn new() -> Self {
        Self {
            asset_states: HashMap::new(),
            latest_readings: HashMap::new(),
            encoder: IotLink::new(),
        }
    }

    /// Update with new sensor reading
    pub fn update(&mut self, reading: SensorReading, asset_id: &str) {
        let hv = self.encoder.encode_reading(&reading);

        // Update asset state
        let state = self
            .asset_states
            .entry(asset_id.to_string())
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        state.add(&hv);

        // Store latest reading
        self.latest_readings
            .insert(reading.sensor_id.clone(), reading);
    }

    /// Get fused state for an asset
    pub fn get_state(&self, asset_id: &str) -> Option<BinaryHV> {
        self.asset_states.get(asset_id).map(|s| s.threshold())
    }

    /// Compare two asset states
    pub fn compare_states(&self, asset1: &str, asset2: &str) -> Option<f32> {
        let s1 = self.asset_states.get(asset1)?;
        let s2 = self.asset_states.get(asset2)?;
        Some(s1.threshold().similarity(&s2.threshold()))
    }

    /// Get latest reading for a sensor
    pub fn get_reading(&self, sensor_id: &str) -> Option<&SensorReading> {
        self.latest_readings.get(sensor_id)
    }
}

impl Default for SensorFusion {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Predictive Maintenance
// ============================================================================

/// Predictive maintenance using holographic failure patterns
pub struct PredictiveMaintenance {
    /// Failure patterns per asset type
    failure_patterns: HashMap<String, BundleAccumulator>,
    /// Normal operation patterns per asset type
    normal_patterns: HashMap<String, BundleAccumulator>,
    /// Failure count per asset type
    failure_counts: HashMap<String, usize>,
    /// Normal count per asset type
    normal_counts: HashMap<String, usize>,
    /// Encoder
    encoder: IotLink,
}

impl PredictiveMaintenance {
    pub fn new() -> Self {
        Self {
            failure_patterns: HashMap::new(),
            normal_patterns: HashMap::new(),
            failure_counts: HashMap::new(),
            normal_counts: HashMap::new(),
            encoder: IotLink::new(),
        }
    }

    /// Learn from a failure event
    pub fn learn_failure(&mut self, event: &MaintenanceEvent, asset_type: &str) {
        // Use the same encoding as predict_failure (sensor fusion only)
        // so that similarity comparisons are in the same vector space.
        let hv = self
            .encoder
            .encode_sensor_fusion(&event.sensor_readings_before);

        let bundle = self
            .failure_patterns
            .entry(asset_type.to_string())
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        *self
            .failure_counts
            .entry(asset_type.to_string())
            .or_insert(0) += 1;
    }

    /// Learn normal operation pattern
    pub fn learn_normal(&mut self, readings: &[SensorReading], asset_type: &str) {
        let hv = self.encoder.encode_sensor_fusion(readings);

        let bundle = self
            .normal_patterns
            .entry(asset_type.to_string())
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        *self
            .normal_counts
            .entry(asset_type.to_string())
            .or_insert(0) += 1;
    }

    /// Predict failure probability based on current sensor state
    pub fn predict_failure(
        &self,
        readings: &[SensorReading],
        asset_type: &str,
    ) -> FailurePrediction {
        let current_hv = self.encoder.encode_sensor_fusion(readings);

        let failure_sim = self
            .failure_patterns
            .get(asset_type)
            .map(|p| current_hv.similarity(&p.threshold()))
            .unwrap_or(0.5);

        let normal_sim = self
            .normal_patterns
            .get(asset_type)
            .map(|p| current_hv.similarity(&p.threshold()))
            .unwrap_or(0.5);

        // Higher failure similarity relative to normal = higher risk
        let risk_score = failure_sim / (failure_sim + normal_sim);

        let recommendation = if risk_score > 0.7 {
            MaintenanceRecommendation::Immediate
        } else if risk_score > 0.5 {
            MaintenanceRecommendation::Soon
        } else if risk_score > 0.3 {
            MaintenanceRecommendation::Scheduled
        } else {
            MaintenanceRecommendation::None
        };

        FailurePrediction {
            risk_score,
            failure_similarity: failure_sim,
            normal_similarity: normal_sim,
            recommendation,
        }
    }
}

impl Default for PredictiveMaintenance {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FailurePrediction {
    pub risk_score: f32,
    pub failure_similarity: f32,
    pub normal_similarity: f32,
    pub recommendation: MaintenanceRecommendation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceRecommendation {
    None,
    Scheduled,
    Soon,
    Immediate,
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// IoT anomaly detector
pub struct IotAnomalyDetector {
    /// Baseline per sensor
    baselines: HashMap<String, SensorBaseline>,
    /// Encoder
    encoder: IotLink,
}

struct SensorBaseline {
    bundle: BundleAccumulator,
    count: usize,
    min_value: f64,
    max_value: f64,
    sum: f64,
}

impl IotAnomalyDetector {
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            encoder: IotLink::new(),
        }
    }

    /// Learn normal behavior for a sensor
    pub fn learn(&mut self, reading: &SensorReading) {
        let hv = self.encoder.encode_reading(reading);

        let baseline = self
            .baselines
            .entry(reading.sensor_id.clone())
            .or_insert_with(|| SensorBaseline {
                bundle: BundleAccumulator::new(DIMENSION),
                count: 0,
                min_value: f64::MAX,
                max_value: f64::MIN,
                sum: 0.0,
            });

        baseline.bundle.add(&hv);
        baseline.count += 1;
        baseline.min_value = baseline.min_value.min(reading.value);
        baseline.max_value = baseline.max_value.max(reading.value);
        baseline.sum += reading.value;
    }

    /// Check if a reading is anomalous
    pub fn check(&self, reading: &SensorReading) -> AnomalyResult {
        let Some(baseline) = self.baselines.get(&reading.sensor_id) else {
            return AnomalyResult {
                is_anomaly: false,
                holographic_deviation: 0.5,
                statistical_deviation: 0.0,
                reason: "No baseline".to_string(),
            };
        };

        if baseline.count < 50 {
            return AnomalyResult {
                is_anomaly: false,
                holographic_deviation: 0.5,
                statistical_deviation: 0.0,
                reason: "Insufficient baseline data".to_string(),
            };
        }

        let hv = self.encoder.encode_reading(reading);
        let baseline_hv = baseline.bundle.threshold();
        let holo_sim = hv.similarity(&baseline_hv);

        // Statistical check
        let mean = baseline.sum / baseline.count as f64;
        let range = baseline.max_value - baseline.min_value;
        let stat_dev = if range > 0.0 {
            (reading.value - mean).abs() / range
        } else {
            0.0
        };

        let is_anomaly = holo_sim < 0.4 || stat_dev > 2.0;

        let reason = if holo_sim < 0.4 && stat_dev > 2.0 {
            "Pattern and statistical anomaly"
        } else if holo_sim < 0.4 {
            "Pattern anomaly"
        } else if stat_dev > 2.0 {
            "Statistical anomaly"
        } else {
            "Normal"
        };

        AnomalyResult {
            is_anomaly,
            holographic_deviation: 1.0 - holo_sim,
            statistical_deviation: stat_dev as f32,
            reason: reason.to_string(),
        }
    }
}

impl Default for IotAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub is_anomaly: bool,
    pub holographic_deviation: f32,
    pub statistical_deviation: f32,
    pub reason: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reading(sensor_id: &str, sensor_type: SensorType, value: f64) -> SensorReading {
        SensorReading {
            sensor_id: sensor_id.to_string(),
            sensor_type,
            value,
            unit: "unit".to_string(),
            quality: DataQuality::Good,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_sensor_encoding() {
        let link = IotLink::new();

        let reading = make_reading("temp_001", SensorType::Temperature, 25.5);
        let hv = link.encode_reading(&reading);

        // Same reading should encode consistently
        let hv2 = link.encode_reading(&reading);
        assert!((hv.similarity(&hv2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sensor_fusion() {
        let mut fusion = SensorFusion::new();

        // Add multiple sensor readings for an asset
        fusion.update(
            make_reading("temp_001", SensorType::Temperature, 25.0),
            "machine_1",
        );
        fusion.update(
            make_reading("vib_001", SensorType::Vibration, 0.5),
            "machine_1",
        );
        fusion.update(
            make_reading("curr_001", SensorType::Current, 15.0),
            "machine_1",
        );

        let state = fusion.get_state("machine_1");
        assert!(state.is_some());
    }

    #[test]
    fn test_predictive_maintenance() {
        let mut pm = PredictiveMaintenance::new();

        // Learn normal patterns
        for i in 0..100 {
            let readings = vec![
                make_reading("temp", SensorType::Temperature, 25.0 + (i as f64 * 0.1)),
                make_reading("vib", SensorType::Vibration, 0.5),
            ];
            pm.learn_normal(&readings, "pump");
        }

        // Learn failure pattern (high temp, high vibration)
        for _ in 0..20 {
            let event = MaintenanceEvent {
                event_id: "fail_001".to_string(),
                asset_id: "pump_001".to_string(),
                event_type: MaintenanceType::Corrective,
                description: "Bearing failure".to_string(),
                sensor_readings_before: vec![
                    make_reading("temp", SensorType::Temperature, 85.0),
                    make_reading("vib", SensorType::Vibration, 5.0),
                ],
                timestamp: 1000,
            };
            pm.learn_failure(&event, "pump");
        }

        // Predict on normal state
        let normal_pred = pm.predict_failure(
            &vec![
                make_reading("temp", SensorType::Temperature, 26.0),
                make_reading("vib", SensorType::Vibration, 0.6),
            ],
            "pump",
        );
        println!("Normal prediction: {:?}", normal_pred);

        // Predict on concerning state
        let risky_pred = pm.predict_failure(
            &vec![
                make_reading("temp", SensorType::Temperature, 75.0),
                make_reading("vib", SensorType::Vibration, 3.5),
            ],
            "pump",
        );
        println!("Risky prediction: {:?}", risky_pred);

        assert!(risky_pred.risk_score > normal_pred.risk_score);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = IotAnomalyDetector::new();

        // Learn normal temperature range
        for i in 0..100 {
            detector.learn(&make_reading(
                "temp_001",
                SensorType::Temperature,
                20.0 + (i as f64 % 10.0),
            ));
        }

        // Check normal reading
        let normal_result =
            detector.check(&make_reading("temp_001", SensorType::Temperature, 25.0));
        println!("Normal check: {:?}", normal_result);

        // Check anomalous reading
        let anomaly_result =
            detector.check(&make_reading("temp_001", SensorType::Temperature, 100.0));
        println!("Anomaly check: {:?}", anomaly_result);

        assert!(anomaly_result.statistical_deviation > normal_result.statistical_deviation);
    }
}
