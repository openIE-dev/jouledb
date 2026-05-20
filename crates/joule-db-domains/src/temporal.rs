//! HDC-powered Time-Series Analysis and Forecasting module
//!
//! Provides holographic encoding for:
//! - Temporal pattern recognition
//! - Anomaly detection in time series
//! - Similarity-based forecasting
//! - Seasonality detection

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePoint {
    pub timestamp: u64,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub id: String,
    pub name: String,
    pub points: Vec<TimePoint>,
    pub granularity: Granularity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Granularity {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrendDirection {
    Up,
    Down,
    Flat,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub period: u64,
    pub strength: f64,
    pub phase: f64,
}

// ============================================================================
// Temporal Encoder (macro-generated core + wrapper)
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder core for temporal domain data
    pub struct TemporalLinkCore {
        seed: 0x7E00_0001,
        dimension: 10000,
        fields: ["timestamp", "value", "trend", "delta", "position", "series"],
        scalars: ["value", "delta", "position", "hour", "day", "month"],
        enums: {
            granularity_vectors: Granularity => [Granularity::Second, Granularity::Minute,
                                                  Granularity::Hour, Granularity::Day,
                                                  Granularity::Week, Granularity::Month,
                                                  Granularity::Year],
            trend_vectors: TrendDirection => [TrendDirection::Up, TrendDirection::Down,
                                              TrendDirection::Flat, TrendDirection::Volatile]
        },
    }
}

pub struct TemporalLink {
    core: TemporalLinkCore,
    pub window_size: usize,
}

impl TemporalLink {
    pub fn new() -> Self {
        Self::with_window_size(24) // Default 24-point window
    }

    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            core: TemporalLinkCore::new(),
            window_size,
        }
    }

    pub fn encode_window(&self, points: &[TimePoint]) -> BinaryHV {
        if points.is_empty() {
            return BinaryHV::zeros(DIMENSION);
        }

        let mut components = Vec::new();

        // Encode each point with positional binding
        for (i, point) in points.iter().enumerate() {
            let val_scaled = ((point.value + 1000.0) / 2000.0 * 100.0).clamp(0.0, 100.0) as u32;
            let val_hv = self.core.encode_scalar("value", val_scaled, 100);
            let pos_hv = self
                .core
                .encode_scalar("position", i as u32, self.window_size as u32);
            components.push(val_hv.bind(&pos_hv));
        }

        // Encode deltas (changes between consecutive points)
        for i in 1..points.len() {
            let delta = points[i].value - points[i - 1].value;
            let delta_sign = if delta > 0.0 {
                1
            } else if delta < 0.0 {
                2
            } else {
                0
            };
            let delta_hv = self.core.field_vectors["delta"]
                .bind(&self.core.encode_scalar("delta", delta_sign, 2));
            components.push(delta_hv.permute(i));
        }

        self.core.bundle(&components)
    }

    pub fn encode_trend(&self, points: &[TimePoint]) -> TrendDirection {
        if points.len() < 2 {
            return TrendDirection::Flat;
        }

        let first = points.first().unwrap().value;
        let last = points.last().unwrap().value;
        let diff = last - first;
        let threshold = (first.abs() * 0.05).max(0.01);

        // Calculate volatility
        let mut changes = 0;
        for i in 1..points.len() {
            if (points[i].value - points[i - 1].value).abs() > threshold {
                changes += 1;
            }
        }
        let volatility = changes as f64 / points.len() as f64;

        if volatility > 0.5 {
            TrendDirection::Volatile
        } else if diff > threshold {
            TrendDirection::Up
        } else if diff < -threshold {
            TrendDirection::Down
        } else {
            TrendDirection::Flat
        }
    }
}

impl Default for TemporalLink {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Time Series Database
// ============================================================================

pub struct TimeSeriesDb {
    encoder: TemporalLink,
    pattern_hologram: BundleAccumulator,
    series_vectors: HashMap<String, BinaryHV>,
    series_data: HashMap<String, TimeSeries>,
    window_cache: HashMap<String, VecDeque<BinaryHV>>,
}

impl TimeSeriesDb {
    pub fn new() -> Self {
        Self {
            encoder: TemporalLink::new(),
            pattern_hologram: BundleAccumulator::new(DIMENSION),
            series_vectors: HashMap::new(),
            series_data: HashMap::new(),
            window_cache: HashMap::new(),
        }
    }

    pub fn add_series(&mut self, series: TimeSeries) {
        if series.points.len() >= self.encoder.window_size {
            let window = &series.points[series.points.len() - self.encoder.window_size..];
            let hv = self.encoder.encode_window(window);
            self.pattern_hologram.add(&hv);
            self.series_vectors.insert(series.id.clone(), hv);
        }
        self.series_data.insert(series.id.clone(), series);
    }

    pub fn append_point(&mut self, series_id: &str, point: TimePoint) {
        if let Some(series) = self.series_data.get_mut(series_id) {
            series.points.push(point);

            // Update window cache
            if series.points.len() >= self.encoder.window_size {
                let start = series.points.len() - self.encoder.window_size;
                let window = &series.points[start..];
                let hv = self.encoder.encode_window(window);
                self.series_vectors
                    .insert(series_id.to_string(), hv.clone());

                let cache = self
                    .window_cache
                    .entry(series_id.to_string())
                    .or_insert_with(VecDeque::new);
                cache.push_back(hv);
                if cache.len() > 100 {
                    cache.pop_front();
                }
            }
        }
    }

    pub fn find_similar_patterns(
        &self,
        series_id: &str,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query_hv = match self.series_vectors.get(series_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut results: Vec<(String, f32)> = self
            .series_vectors
            .iter()
            .filter(|(id, _)| *id != series_id)
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .filter(|(_, sim)| *sim >= min_sim)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn get_trend(&self, series_id: &str) -> Option<TrendDirection> {
        self.series_data
            .get(series_id)
            .map(|s| self.encoder.encode_trend(&s.points))
    }

    pub fn series_count(&self) -> usize {
        self.series_data.len()
    }
}

impl Default for TimeSeriesDb {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Anomaly Detector
// ============================================================================

pub struct TemporalAnomalyDetector {
    encoder: TemporalLink,
    normal_patterns: BundleAccumulator,
    anomaly_patterns: BundleAccumulator,
    threshold: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalAnomaly {
    pub timestamp: u64,
    pub anomaly_score: f32,
    pub expected_trend: TrendDirection,
    pub actual_trend: TrendDirection,
}

impl TemporalAnomalyDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: TemporalLink::new(),
            normal_patterns: BundleAccumulator::new(DIMENSION),
            anomaly_patterns: BundleAccumulator::new(DIMENSION),
            threshold,
        }
    }

    pub fn learn_normal(&mut self, points: &[TimePoint]) {
        let hv = self.encoder.encode_window(points);
        self.normal_patterns.add(&hv);
    }

    pub fn learn_anomaly(&mut self, points: &[TimePoint]) {
        let hv = self.encoder.encode_window(points);
        self.anomaly_patterns.add(&hv);
    }

    pub fn detect(&self, points: &[TimePoint]) -> Option<TemporalAnomaly> {
        let window_hv = self.encoder.encode_window(points);
        let normal_sim = window_hv.similarity(&self.normal_patterns.threshold());
        let anomaly_sim = window_hv.similarity(&self.anomaly_patterns.threshold());

        let anomaly_score = anomaly_sim - normal_sim;

        if anomaly_score > self.threshold {
            Some(TemporalAnomaly {
                timestamp: points.last().map(|p| p.timestamp).unwrap_or(0),
                anomaly_score,
                expected_trend: TrendDirection::Flat,
                actual_trend: self.encoder.encode_trend(points),
            })
        } else {
            None
        }
    }
}

impl Default for TemporalAnomalyDetector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

// ============================================================================
// Forecaster
// ============================================================================

pub struct PatternForecaster {
    encoder: TemporalLink,
    pattern_outcomes: HashMap<u64, (BinaryHV, f64)>, // pattern hash -> (pattern, avg next value)
    pattern_hologram: BundleAccumulator,
}

impl PatternForecaster {
    pub fn new() -> Self {
        Self {
            encoder: TemporalLink::new(),
            pattern_outcomes: HashMap::new(),
            pattern_hologram: BundleAccumulator::new(DIMENSION),
        }
    }

    pub fn train(&mut self, history: &[TimePoint], next_value: f64) {
        let pattern_hv = self.encoder.encode_window(history);
        let hash = pattern_hv.condense_to_u64();

        self.pattern_hologram.add(&pattern_hv);
        self.pattern_outcomes.insert(hash, (pattern_hv, next_value));
    }

    pub fn forecast(&self, current: &[TimePoint]) -> f64 {
        let query_hv = self.encoder.encode_window(current);

        // Find most similar historical pattern
        let mut best_sim = 0.0f32;
        let mut best_value = current.last().map(|p| p.value).unwrap_or(0.0);

        for (_, (pattern_hv, next_val)) in &self.pattern_outcomes {
            let sim = query_hv.similarity(pattern_hv);
            if sim > best_sim {
                best_sim = sim;
                best_value = *next_val;
            }
        }

        best_value
    }

    pub fn pattern_count(&self) -> usize {
        self.pattern_outcomes.len()
    }
}

impl Default for PatternForecaster {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points(values: &[f64]) -> Vec<TimePoint> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| TimePoint {
                timestamp: i as u64 * 1000,
                value: v,
                labels: HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_window_encoding() {
        let encoder = TemporalLink::new();
        let points = create_test_points(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let hv = encoder.encode_window(&points);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_trend_detection() {
        let encoder = TemporalLink::new();

        // Use values with gradual changes below the volatility threshold
        let up_points = create_test_points(&[100.0, 101.0, 102.0, 103.0, 110.0]);
        assert_eq!(encoder.encode_trend(&up_points), TrendDirection::Up);

        let down_points = create_test_points(&[100.0, 99.0, 98.0, 97.0, 90.0]);
        assert_eq!(encoder.encode_trend(&down_points), TrendDirection::Down);
    }

    #[test]
    fn test_time_series_db() {
        let mut db = TimeSeriesDb::new();

        let series = TimeSeries {
            id: "cpu_usage".to_string(),
            name: "CPU Usage".to_string(),
            points: create_test_points(
                &(0..30)
                    .map(|i| (i as f64).sin() * 50.0 + 50.0)
                    .collect::<Vec<_>>(),
            ),
            granularity: Granularity::Minute,
        };

        db.add_series(series);
        assert_eq!(db.series_count(), 1);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = TemporalAnomalyDetector::new(0.3);

        // Learn normal pattern
        let normal = create_test_points(&[50.0, 51.0, 52.0, 51.0, 50.0]);
        detector.learn_normal(&normal);

        // Learn anomaly pattern
        let anomaly = create_test_points(&[50.0, 100.0, 150.0, 200.0, 250.0]);
        detector.learn_anomaly(&anomaly);

        // Test detection
        let test_normal = create_test_points(&[49.0, 50.0, 51.0, 50.0, 49.0]);
        assert!(detector.detect(&test_normal).is_none());
    }

    #[test]
    fn test_forecaster() {
        let mut forecaster = PatternForecaster::new();

        let pattern = create_test_points(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        forecaster.train(&pattern, 6.0);

        assert_eq!(forecaster.pattern_count(), 1);
    }
}
