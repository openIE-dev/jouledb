//! Sensor data fusion: Kalman filter (1D), complementary filter, moving average,
//! sensor calibration, multi-sensor weighted fusion, outlier rejection,
//! sensor health monitoring, and fusion confidence scoring.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use uuid::Uuid;

// ── Types ──

/// Kind of sensor producing data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorKind {
    Accelerometer,
    Gyroscope,
    Magnetometer,
    Temperature,
    Pressure,
    Humidity,
    Light,
    Proximity,
    Custom,
}

impl SensorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Accelerometer => "accelerometer",
            Self::Gyroscope => "gyroscope",
            Self::Magnetometer => "magnetometer",
            Self::Temperature => "temperature",
            Self::Pressure => "pressure",
            Self::Humidity => "humidity",
            Self::Light => "light",
            Self::Proximity => "proximity",
            Self::Custom => "custom",
        }
    }
}

/// Health status of a sensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensorHealth {
    Healthy,
    Degraded,
    Faulty,
    Offline,
}

impl SensorHealth {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Faulty => "faulty",
            Self::Offline => "offline",
        }
    }
}

/// A single reading from a sensor.
#[derive(Debug, Clone)]
pub struct SensorReading {
    pub sensor_id: String,
    pub kind: SensorKind,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub unit: String,
}

impl SensorReading {
    pub fn new(sensor_id: &str, kind: SensorKind, value: f64, unit: &str) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            kind,
            value,
            timestamp: Utc::now(),
            unit: unit.to_string(),
        }
    }

    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }
}

/// Calibration parameters for a sensor (offset + scale).
#[derive(Debug, Clone)]
pub struct Calibration {
    pub offset: f64,
    pub scale: f64,
}

impl Calibration {
    pub fn new(offset: f64, scale: f64) -> Self {
        Self { offset, scale }
    }

    pub fn identity() -> Self {
        Self { offset: 0.0, scale: 1.0 }
    }

    /// Apply calibration: result = (raw - offset) * scale
    pub fn apply(&self, raw: f64) -> f64 {
        (raw - self.offset) * self.scale
    }
}

/// Result of a fusion operation.
#[derive(Debug, Clone)]
pub struct FusionResult {
    pub id: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub sources: Vec<String>,
}

// ── 1D Kalman Filter ──

/// Simple one-dimensional Kalman filter for sensor fusion.
#[derive(Debug, Clone)]
pub struct KalmanFilter1D {
    /// State estimate.
    pub estimate: f64,
    /// Estimate uncertainty (covariance).
    pub error_covariance: f64,
    /// Process noise.
    process_noise: f64,
    /// Measurement noise.
    measurement_noise: f64,
    /// Kalman gain (last computed).
    pub kalman_gain: f64,
    /// Number of updates applied.
    pub update_count: u64,
}

impl KalmanFilter1D {
    pub fn new(initial_estimate: f64, initial_error: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            estimate: initial_estimate,
            error_covariance: initial_error,
            process_noise,
            measurement_noise,
            kalman_gain: 0.0,
            update_count: 0,
        }
    }

    /// Predict step: propagate state forward.
    pub fn predict(&mut self) {
        // In 1D with no control input, the estimate stays the same.
        // Only the covariance grows.
        self.error_covariance += self.process_noise;
    }

    /// Update step: incorporate a new measurement.
    pub fn update(&mut self, measurement: f64) {
        let denom = self.error_covariance + self.measurement_noise;
        if denom.abs() < 1e-15 {
            return;
        }
        self.kalman_gain = self.error_covariance / denom;
        self.estimate += self.kalman_gain * (measurement - self.estimate);
        self.error_covariance *= 1.0 - self.kalman_gain;
        self.update_count += 1;
    }

    /// Combined predict + update in one call.
    pub fn step(&mut self, measurement: f64) -> f64 {
        self.predict();
        self.update(measurement);
        self.estimate
    }

    /// Reset the filter to new initial conditions.
    pub fn reset(&mut self, initial_estimate: f64, initial_error: f64) {
        self.estimate = initial_estimate;
        self.error_covariance = initial_error;
        self.kalman_gain = 0.0;
        self.update_count = 0;
    }
}

// ── Complementary Filter ──

/// Complementary filter that blends a fast (high-frequency) and slow (low-frequency) sensor.
#[derive(Debug, Clone)]
pub struct ComplementaryFilter {
    /// Blending factor: 0.0 = all slow, 1.0 = all fast.
    alpha: f64,
    /// Current fused output.
    pub output: f64,
    initialized: bool,
}

impl ComplementaryFilter {
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            alpha,
            output: 0.0,
            initialized: false,
        }
    }

    /// Update with a fast reading (e.g., gyro rate integrated) and a slow reading (e.g., accel angle).
    pub fn update(&mut self, fast: f64, slow: f64) -> f64 {
        if !self.initialized {
            self.output = slow;
            self.initialized = true;
        }
        self.output = self.alpha * (self.output + fast) + (1.0 - self.alpha) * slow;
        self.output
    }

    pub fn reset(&mut self) {
        self.output = 0.0;
        self.initialized = false;
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }
}

// ── Moving Average Filter ──

/// Simple moving average over a sliding window.
#[derive(Debug, Clone)]
pub struct MovingAverageFilter {
    window_size: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl MovingAverageFilter {
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.max(1);
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    /// Add a new sample and return the current moving average.
    pub fn add(&mut self, value: f64) -> f64 {
        self.buffer.push_back(value);
        self.sum += value;

        if self.buffer.len() > self.window_size {
            if let Some(old) = self.buffer.pop_front() {
                self.sum -= old;
            }
        }

        self.average()
    }

    pub fn average(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        self.sum / self.buffer.len() as f64
    }

    pub fn count(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.window_size
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Current variance of the window.
    pub fn variance(&self) -> f64 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        let mean = self.average();
        let sum_sq: f64 = self.buffer.iter().map(|v| (v - mean).powi(2)).sum();
        sum_sq / self.buffer.len() as f64
    }
}

// ── Outlier Rejection ──

/// Outlier rejection using z-score on a sliding window.
#[derive(Debug, Clone)]
pub struct OutlierRejector {
    threshold: f64,
    window: VecDeque<f64>,
    max_window: usize,
    rejected_count: u64,
    total_count: u64,
}

impl OutlierRejector {
    pub fn new(threshold: f64, max_window: usize) -> Self {
        Self {
            threshold: threshold.abs(),
            window: VecDeque::with_capacity(max_window),
            max_window: max_window.max(3),
            rejected_count: 0,
            total_count: 0,
        }
    }

    fn stats(&self) -> (f64, f64) {
        if self.window.is_empty() {
            return (0.0, 1.0);
        }
        let n = self.window.len() as f64;
        let mean: f64 = self.window.iter().sum::<f64>() / n;
        if self.window.len() < 2 {
            return (mean, 1.0);
        }
        let var: f64 = self.window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt().max(1e-12);
        (mean, std)
    }

    /// Feed a value. Returns `Some(value)` if accepted, `None` if rejected as outlier.
    pub fn feed(&mut self, value: f64) -> Option<f64> {
        self.total_count += 1;

        if self.window.len() < 3 {
            // Not enough data — accept unconditionally.
            self.window.push_back(value);
            if self.window.len() > self.max_window {
                self.window.pop_front();
            }
            return Some(value);
        }

        let (mean, std) = self.stats();
        let z = (value - mean).abs() / std;

        if z > self.threshold {
            self.rejected_count += 1;
            None
        } else {
            self.window.push_back(value);
            if self.window.len() > self.max_window {
                self.window.pop_front();
            }
            Some(value)
        }
    }

    pub fn rejected_count(&self) -> u64 {
        self.rejected_count
    }

    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    pub fn rejection_rate(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.rejected_count as f64 / self.total_count as f64
    }
}

// ── Sensor Health Monitor ──

/// Monitors sensor health based on update frequency and value range.
#[derive(Debug, Clone)]
pub struct SensorHealthMonitor {
    sensors: HashMap<String, SensorState>,
    /// Maximum age (seconds) before a sensor is considered offline.
    max_age_secs: i64,
    /// How many consecutive out-of-range readings before marking degraded.
    degraded_threshold: u32,
    /// How many consecutive out-of-range readings before marking faulty.
    faulty_threshold: u32,
}

#[derive(Debug, Clone)]
struct SensorState {
    last_seen: DateTime<Utc>,
    health: SensorHealth,
    min_expected: f64,
    max_expected: f64,
    consecutive_oor: u32,
    total_readings: u64,
}

impl SensorHealthMonitor {
    pub fn new(max_age_secs: i64, degraded_threshold: u32, faulty_threshold: u32) -> Self {
        Self {
            sensors: HashMap::new(),
            max_age_secs,
            degraded_threshold,
            faulty_threshold,
        }
    }

    /// Register a sensor with its expected value range.
    pub fn register(&mut self, sensor_id: &str, min_expected: f64, max_expected: f64) {
        self.sensors.insert(sensor_id.to_string(), SensorState {
            last_seen: Utc::now(),
            health: SensorHealth::Healthy,
            min_expected,
            max_expected,
            consecutive_oor: 0,
            total_readings: 0,
        });
    }

    /// Record a reading from a sensor and update its health.
    pub fn record(&mut self, sensor_id: &str, value: f64, now: DateTime<Utc>) {
        if let Some(state) = self.sensors.get_mut(sensor_id) {
            state.last_seen = now;
            state.total_readings += 1;

            if value < state.min_expected || value > state.max_expected {
                state.consecutive_oor += 1;
            } else {
                state.consecutive_oor = 0;
            }

            if state.consecutive_oor >= self.faulty_threshold {
                state.health = SensorHealth::Faulty;
            } else if state.consecutive_oor >= self.degraded_threshold {
                state.health = SensorHealth::Degraded;
            } else {
                state.health = SensorHealth::Healthy;
            }
        }
    }

    /// Check for sensors that have gone offline (not reported within `max_age_secs`).
    pub fn check_timeouts(&mut self, now: DateTime<Utc>) {
        for state in self.sensors.values_mut() {
            let age = now.signed_duration_since(state.last_seen).num_seconds();
            if age > self.max_age_secs {
                state.health = SensorHealth::Offline;
            }
        }
    }

    pub fn health(&self, sensor_id: &str) -> Option<SensorHealth> {
        self.sensors.get(sensor_id).map(|s| s.health)
    }

    pub fn all_healthy(&self) -> bool {
        self.sensors.values().all(|s| s.health == SensorHealth::Healthy)
    }

    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    /// Return ids of sensors with a specific health status.
    pub fn sensors_with_health(&self, health: SensorHealth) -> Vec<String> {
        let mut result: Vec<String> = self.sensors.iter()
            .filter(|(_, s)| s.health == health)
            .map(|(id, _)| id.clone())
            .collect();
        result.sort();
        result
    }
}

// ── Multi-Sensor Weighted Fusion ──

/// Fuses readings from multiple sensors with configurable weights.
#[derive(Debug, Clone)]
pub struct WeightedFusion {
    weights: HashMap<String, f64>,
    calibrations: HashMap<String, Calibration>,
    latest: HashMap<String, f64>,
}

impl WeightedFusion {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            calibrations: HashMap::new(),
            latest: HashMap::new(),
        }
    }

    /// Add a sensor with its relative weight.
    pub fn add_sensor(&mut self, sensor_id: &str, weight: f64) {
        self.weights.insert(sensor_id.to_string(), weight.abs());
    }

    /// Set calibration for a sensor.
    pub fn set_calibration(&mut self, sensor_id: &str, cal: Calibration) {
        self.calibrations.insert(sensor_id.to_string(), cal);
    }

    /// Feed a new reading from a specific sensor.
    pub fn feed(&mut self, sensor_id: &str, raw_value: f64) {
        let calibrated = if let Some(cal) = self.calibrations.get(sensor_id) {
            cal.apply(raw_value)
        } else {
            raw_value
        };
        self.latest.insert(sensor_id.to_string(), calibrated);
    }

    /// Compute the weighted fusion of all latest readings.
    pub fn fuse(&self) -> Option<FusionResult> {
        if self.latest.is_empty() {
            return None;
        }

        let mut total_weight = 0.0_f64;
        let mut weighted_sum = 0.0_f64;
        let mut sources = Vec::new();

        for (id, value) in &self.latest {
            let w = self.weights.get(id).copied().unwrap_or(1.0);
            weighted_sum += w * value;
            total_weight += w;
            sources.push(id.clone());
        }

        if total_weight.abs() < 1e-15 {
            return None;
        }

        let fused = weighted_sum / total_weight;

        // Confidence: based on how many registered sensors contributed.
        let expected = self.weights.len().max(1) as f64;
        let actual = self.latest.len() as f64;
        let confidence = (actual / expected).min(1.0);

        sources.sort();

        Some(FusionResult {
            id: Uuid::new_v4().to_string(),
            value: fused,
            confidence,
            timestamp: Utc::now(),
            sources,
        })
    }

    /// Reset all latest readings.
    pub fn clear_readings(&mut self) {
        self.latest.clear();
    }

    pub fn sensor_count(&self) -> usize {
        self.weights.len()
    }
}

impl Default for WeightedFusion {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fusion Confidence Scorer ──

/// Computes a confidence score for fused data based on multiple quality indicators.
#[derive(Debug, Clone)]
pub struct FusionConfidence {
    /// Weight for agreement factor (how close sensors agree).
    agreement_weight: f64,
    /// Weight for freshness factor (how recent the data is).
    freshness_weight: f64,
    /// Weight for coverage factor (how many sensors contributed).
    coverage_weight: f64,
    /// Maximum acceptable age in seconds for freshness scoring.
    max_age_secs: f64,
}

impl FusionConfidence {
    pub fn new(agreement_weight: f64, freshness_weight: f64, coverage_weight: f64, max_age_secs: f64) -> Self {
        Self {
            agreement_weight,
            freshness_weight,
            coverage_weight,
            max_age_secs: max_age_secs.max(1.0),
        }
    }

    /// Compute confidence given a set of values, their timestamps, and the expected sensor count.
    pub fn score(&self, values: &[f64], timestamps: &[DateTime<Utc>], expected_count: usize, now: DateTime<Utc>) -> f64 {
        let agreement = self.agreement_score(values);
        let freshness = self.freshness_score(timestamps, now);
        let coverage = self.coverage_score(values.len(), expected_count);

        let total_weight = self.agreement_weight + self.freshness_weight + self.coverage_weight;
        if total_weight.abs() < 1e-15 {
            return 0.0;
        }

        let raw = (self.agreement_weight * agreement
            + self.freshness_weight * freshness
            + self.coverage_weight * coverage) / total_weight;

        raw.clamp(0.0, 1.0)
    }

    fn agreement_score(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let n = values.len() as f64;
        let mean: f64 = values.iter().sum::<f64>() / n;
        if mean.abs() < 1e-15 {
            // All near zero — check absolute variance.
            let var: f64 = values.iter().map(|v| v.powi(2)).sum::<f64>() / n;
            return if var < 1e-10 { 1.0 } else { (1.0 / (1.0 + var)).max(0.0) };
        }
        let cv = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt() / mean.abs();
        (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
    }

    fn freshness_score(&self, timestamps: &[DateTime<Utc>], now: DateTime<Utc>) -> f64 {
        if timestamps.is_empty() {
            return 0.0;
        }
        let ages: Vec<f64> = timestamps.iter().map(|t| {
            now.signed_duration_since(*t).num_milliseconds().max(0) as f64 / 1000.0
        }).collect();
        let avg_age: f64 = ages.iter().sum::<f64>() / ages.len() as f64;
        (1.0 - avg_age / self.max_age_secs).clamp(0.0, 1.0)
    }

    fn coverage_score(&self, actual: usize, expected: usize) -> f64 {
        if expected == 0 {
            return 0.0;
        }
        (actual as f64 / expected as f64).min(1.0)
    }
}

impl Default for FusionConfidence {
    fn default() -> Self {
        Self::new(0.4, 0.3, 0.3, 10.0)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn kalman_converges() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
        for _ in 0..100 {
            kf.step(10.0);
        }
        assert!((kf.estimate - 10.0).abs() < 0.1);
    }

    #[test]
    fn kalman_gain_decreases() {
        let mut kf = KalmanFilter1D::new(0.0, 10.0, 0.01, 1.0);
        kf.step(5.0);
        let g1 = kf.kalman_gain;
        kf.step(5.0);
        let g2 = kf.kalman_gain;
        // Gain should decrease as confidence grows.
        assert!(g2 < g1);
    }

    #[test]
    fn kalman_reset() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
        kf.step(10.0);
        kf.reset(0.0, 1.0);
        assert_eq!(kf.update_count, 0);
        assert!((kf.estimate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn complementary_filter_blends() {
        let mut cf = ComplementaryFilter::new(0.98);
        // First update initializes to slow value.
        let v = cf.update(0.0, 45.0);
        assert!((v - 45.0).abs() < 1e-10);

        // Subsequent updates blend.
        let v2 = cf.update(0.1, 44.9);
        // Should be close to 45.0.
        assert!((v2 - 45.0).abs() < 1.0);
    }

    #[test]
    fn complementary_filter_alpha_clamp() {
        let cf = ComplementaryFilter::new(1.5);
        assert!((cf.alpha() - 1.0).abs() < 1e-10);
        let cf2 = ComplementaryFilter::new(-0.5);
        assert!((cf2.alpha() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn complementary_filter_reset() {
        let mut cf = ComplementaryFilter::new(0.5);
        cf.update(1.0, 2.0);
        cf.reset();
        assert!((cf.output - 0.0).abs() < 1e-10);
    }

    #[test]
    fn moving_average_basic() {
        let mut ma = MovingAverageFilter::new(3);
        assert!((ma.add(3.0) - 3.0).abs() < 1e-10);
        assert!((ma.add(6.0) - 4.5).abs() < 1e-10);
        assert!((ma.add(9.0) - 6.0).abs() < 1e-10);
        // Window is full, old values drop off.
        assert!((ma.add(12.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn moving_average_variance() {
        let mut ma = MovingAverageFilter::new(4);
        ma.add(2.0);
        ma.add(4.0);
        ma.add(6.0);
        ma.add(8.0);
        // mean = 5, var = mean of squares of deviations = (9+1+1+9)/4 = 5
        assert!((ma.variance() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn moving_average_clear() {
        let mut ma = MovingAverageFilter::new(5);
        ma.add(10.0);
        ma.add(20.0);
        ma.clear();
        assert_eq!(ma.count(), 0);
        assert!((ma.average() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn moving_average_window_size_at_least_one() {
        let ma = MovingAverageFilter::new(0);
        assert_eq!(ma.window_size(), 1);
    }

    #[test]
    fn outlier_rejection_basic() {
        let mut or = OutlierRejector::new(2.0, 20);
        // Feed stable values.
        for _ in 0..10 {
            assert!(or.feed(10.0).is_some());
        }
        // An extreme outlier should be rejected.
        assert!(or.feed(1000.0).is_none());
        assert_eq!(or.rejected_count(), 1);
    }

    #[test]
    fn outlier_rejection_rate() {
        let mut or = OutlierRejector::new(2.0, 50);
        for _ in 0..20 {
            or.feed(5.0);
        }
        or.feed(500.0); // outlier
        assert!(or.rejection_rate() > 0.0);
        assert!(or.rejection_rate() < 0.1);
    }

    #[test]
    fn calibration_apply() {
        let cal = Calibration::new(1.0, 2.0);
        // (5.0 - 1.0) * 2.0 = 8.0
        assert!((cal.apply(5.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn calibration_identity() {
        let cal = Calibration::identity();
        assert!((cal.apply(42.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn sensor_health_monitor_basic() {
        let mut mon = SensorHealthMonitor::new(60, 3, 5);
        mon.register("s1", 0.0, 100.0);
        let now = Utc::now();
        mon.record("s1", 50.0, now);
        assert_eq!(mon.health("s1"), Some(SensorHealth::Healthy));
    }

    #[test]
    fn sensor_health_monitor_degraded() {
        let mut mon = SensorHealthMonitor::new(60, 2, 5);
        mon.register("s1", 0.0, 100.0);
        let now = Utc::now();
        mon.record("s1", 200.0, now); // oor #1
        mon.record("s1", 200.0, now); // oor #2 >= degraded_threshold
        assert_eq!(mon.health("s1"), Some(SensorHealth::Degraded));
    }

    #[test]
    fn sensor_health_monitor_faulty() {
        let mut mon = SensorHealthMonitor::new(60, 2, 3);
        mon.register("s1", 0.0, 100.0);
        let now = Utc::now();
        for _ in 0..3 {
            mon.record("s1", -50.0, now);
        }
        assert_eq!(mon.health("s1"), Some(SensorHealth::Faulty));
    }

    #[test]
    fn sensor_health_monitor_offline() {
        let mut mon = SensorHealthMonitor::new(10, 3, 5);
        mon.register("s1", 0.0, 100.0);
        let now = Utc::now();
        mon.record("s1", 50.0, now);
        let later = now + Duration::seconds(30);
        mon.check_timeouts(later);
        assert_eq!(mon.health("s1"), Some(SensorHealth::Offline));
    }

    #[test]
    fn sensor_health_monitor_recovery() {
        let mut mon = SensorHealthMonitor::new(60, 2, 5);
        mon.register("s1", 0.0, 100.0);
        let now = Utc::now();
        mon.record("s1", 200.0, now); // oor
        mon.record("s1", 200.0, now); // oor -> degraded
        assert_eq!(mon.health("s1"), Some(SensorHealth::Degraded));
        mon.record("s1", 50.0, now);  // in range -> resets
        assert_eq!(mon.health("s1"), Some(SensorHealth::Healthy));
    }

    #[test]
    fn weighted_fusion_basic() {
        let mut wf = WeightedFusion::new();
        wf.add_sensor("a", 2.0);
        wf.add_sensor("b", 1.0);
        wf.feed("a", 10.0);
        wf.feed("b", 7.0);
        let result = wf.fuse().unwrap();
        // (2*10 + 1*7) / 3 = 9.0
        assert!((result.value - 9.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_fusion_with_calibration() {
        let mut wf = WeightedFusion::new();
        wf.add_sensor("a", 1.0);
        wf.set_calibration("a", Calibration::new(2.0, 0.5));
        wf.feed("a", 10.0); // (10 - 2) * 0.5 = 4.0
        let result = wf.fuse().unwrap();
        assert!((result.value - 4.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_fusion_confidence() {
        let mut wf = WeightedFusion::new();
        wf.add_sensor("a", 1.0);
        wf.add_sensor("b", 1.0);
        wf.add_sensor("c", 1.0);
        // Only feed 2 of 3.
        wf.feed("a", 10.0);
        wf.feed("b", 10.0);
        let result = wf.fuse().unwrap();
        assert!((result.confidence - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_fusion_empty() {
        let wf = WeightedFusion::new();
        assert!(wf.fuse().is_none());
    }

    #[test]
    fn fusion_confidence_perfect() {
        let fc = FusionConfidence::default();
        let now = Utc::now();
        let vals = vec![10.0, 10.0, 10.0];
        let stamps = vec![now, now, now];
        let score = fc.score(&vals, &stamps, 3, now);
        assert!(score > 0.9);
    }

    #[test]
    fn fusion_confidence_low_coverage() {
        let fc = FusionConfidence::default();
        let now = Utc::now();
        let vals = vec![10.0];
        let stamps = vec![now];
        let score = fc.score(&vals, &stamps, 5, now);
        assert!(score < 0.7); // Only 1 of 5 sensors contributed.
    }

    #[test]
    fn fusion_confidence_stale_data() {
        let fc = FusionConfidence::new(0.0, 1.0, 0.0, 10.0);
        let now = Utc::now();
        let old = now - Duration::seconds(20);
        let vals = vec![10.0];
        let stamps = vec![old];
        let score = fc.score(&vals, &stamps, 1, now);
        assert!((score - 0.0).abs() < 1e-10); // Way past max_age.
    }

    #[test]
    fn sensor_reading_builder() {
        let ts = Utc::now();
        let r = SensorReading::new("s1", SensorKind::Temperature, 22.5, "C")
            .with_timestamp(ts);
        assert_eq!(r.sensor_id, "s1");
        assert!((r.value - 22.5).abs() < 1e-10);
        assert_eq!(r.timestamp, ts);
    }

    #[test]
    fn sensor_kind_as_str() {
        assert_eq!(SensorKind::Accelerometer.as_str(), "accelerometer");
        assert_eq!(SensorKind::Custom.as_str(), "custom");
    }

    #[test]
    fn sensors_with_health_filter() {
        let mut mon = SensorHealthMonitor::new(60, 2, 5);
        mon.register("a", 0.0, 100.0);
        mon.register("b", 0.0, 100.0);
        let now = Utc::now();
        mon.record("a", 50.0, now);
        mon.record("b", 50.0, now);
        let healthy = mon.sensors_with_health(SensorHealth::Healthy);
        assert_eq!(healthy.len(), 2);
    }
}
