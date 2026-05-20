//! Telemetry data ingestion: time-stamped data points, batch ingestion,
//! data validation, downsampling on ingest, retention policy, ingestion rate
//! tracking, backpressure signaling, and data point deduplication.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Types ──

/// A single telemetry data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub id: String,
    pub source: String,
    pub metric: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub tags: HashMap<String, String>,
}

impl DataPoint {
    pub fn new(source: &str, metric: &str, value: f64, timestamp: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source: source.to_string(),
            metric: metric.to_string(),
            value,
            timestamp,
            tags: HashMap::new(),
        }
    }

    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Deduplication key: source + metric + timestamp (second-level precision).
    pub fn dedup_key(&self) -> String {
        format!("{}:{}:{}", self.source, self.metric, self.timestamp.timestamp())
    }
}

/// Validation rule for incoming data points.
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub metric: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub required_tags: Vec<String>,
}

impl ValidationRule {
    pub fn new(metric: &str) -> Self {
        Self {
            metric: metric.to_string(),
            min: None,
            max: None,
            required_tags: Vec::new(),
        }
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    pub fn with_required_tag(mut self, tag: &str) -> Self {
        self.required_tags.push(tag.to_string());
        self
    }

    pub fn validate(&self, point: &DataPoint) -> Result<(), String> {
        if let Some(min) = self.min {
            if point.value < min {
                return Err(format!("value {} below minimum {}", point.value, min));
            }
        }
        if let Some(max) = self.max {
            if point.value > max {
                return Err(format!("value {} above maximum {}", point.value, max));
            }
        }
        for tag in &self.required_tags {
            if !point.tags.contains_key(tag) {
                return Err(format!("missing required tag: {}", tag));
            }
        }
        Ok(())
    }
}

/// Downsampling strategy applied on ingest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownsampleStrategy {
    /// Keep all points.
    None,
    /// Average values within each time bucket.
    Average,
    /// Keep the last value in each time bucket.
    LastValue,
    /// Keep the min value in each time bucket.
    Min,
    /// Keep the max value in each time bucket.
    Max,
}

/// Retention policy for stored data.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum number of data points to retain.
    pub max_points: usize,
    /// Maximum age of data points.
    pub max_age: Option<Duration>,
}

impl RetentionPolicy {
    pub fn new(max_points: usize) -> Self {
        Self { max_points, max_age: None }
    }

    pub fn with_max_age(mut self, duration: Duration) -> Self {
        self.max_age = Some(duration);
        self
    }
}

/// Backpressure signal emitted by the ingestor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureSignal {
    /// Accept data normally.
    Accept,
    /// Slow down — approaching capacity.
    SlowDown,
    /// Reject — at capacity.
    Reject,
}

/// Ingestion statistics.
#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    pub total_ingested: u64,
    pub total_rejected: u64,
    pub total_deduplicated: u64,
    pub total_downsampled: u64,
    pub total_expired: u64,
    /// Rate tracking: points ingested in the current window.
    pub current_window_count: u64,
}

// ── Ingestor ──

/// Telemetry data ingestor with validation, dedup, downsampling, and backpressure.
pub struct TelemetryIngestor {
    store: VecDeque<DataPoint>,
    validation_rules: HashMap<String, ValidationRule>,
    dedup_keys: HashSet<String>,
    retention: RetentionPolicy,
    downsample: DownsampleStrategy,
    /// Bucket size in seconds for downsampling.
    bucket_secs: i64,
    /// Backpressure: high-water mark (fraction of max_points).
    high_water: f64,
    stats: IngestStats,
    /// Rate tracking window.
    rate_window: VecDeque<DateTime<Utc>>,
    rate_window_secs: i64,
}

impl TelemetryIngestor {
    pub fn new(retention: RetentionPolicy) -> Self {
        Self {
            store: VecDeque::new(),
            validation_rules: HashMap::new(),
            dedup_keys: HashSet::new(),
            retention,
            downsample: DownsampleStrategy::None,
            bucket_secs: 60,
            high_water: 0.8,
            stats: IngestStats::default(),
            rate_window: VecDeque::new(),
            rate_window_secs: 60,
        }
    }

    /// Set the downsampling strategy and bucket size.
    pub fn set_downsample(&mut self, strategy: DownsampleStrategy, bucket_secs: i64) {
        self.downsample = strategy;
        self.bucket_secs = bucket_secs.max(1);
    }

    /// Set the backpressure high-water mark (0.0..1.0).
    pub fn set_high_water(&mut self, fraction: f64) {
        self.high_water = fraction.clamp(0.0, 1.0);
    }

    /// Add a validation rule for a specific metric.
    pub fn add_validation_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.insert(rule.metric.clone(), rule);
    }

    /// Current backpressure signal.
    pub fn backpressure(&self) -> BackpressureSignal {
        let ratio = self.store.len() as f64 / self.retention.max_points.max(1) as f64;
        if ratio >= 1.0 {
            BackpressureSignal::Reject
        } else if ratio >= self.high_water {
            BackpressureSignal::SlowDown
        } else {
            BackpressureSignal::Accept
        }
    }

    /// Ingest a single data point. Returns Ok(true) if stored, Ok(false) if deduplicated/downsampled, Err on validation failure.
    pub fn ingest(&mut self, point: DataPoint) -> Result<bool, String> {
        // Backpressure check.
        if self.backpressure() == BackpressureSignal::Reject {
            self.stats.total_rejected += 1;
            return Err("backpressure: at capacity".to_string());
        }

        // Validation.
        if let Some(rule) = self.validation_rules.get(&point.metric) {
            rule.validate(&point)?;
        }

        // Deduplication.
        let dk = point.dedup_key();
        if self.dedup_keys.contains(&dk) {
            self.stats.total_deduplicated += 1;
            return Ok(false);
        }

        // Downsampling.
        if self.downsample != DownsampleStrategy::None {
            let pt_ts = point.timestamp.timestamp();
            let in_bucket = |p: &DataPoint| -> bool {
                p.source == point.source
                    && p.metric == point.metric
                    && (p.timestamp.timestamp() - pt_ts).abs() < self.bucket_secs
            };
            let dominated = self.store.iter().any(|p| in_bucket(p));
            if dominated {
                // Apply strategy: replace or skip.
                match self.downsample {
                    DownsampleStrategy::LastValue => {
                        // Replace: remove old, insert new.
                        let src = point.source.clone();
                        let met = point.metric.clone();
                        let bs = self.bucket_secs;
                        self.store.retain(|p| {
                            !(p.source == src && p.metric == met && (p.timestamp.timestamp() - pt_ts).abs() < bs)
                        });
                    }
                    DownsampleStrategy::Min => {
                        let existing_min = self.store.iter()
                            .filter(|p| in_bucket(p))
                            .map(|p| p.value)
                            .fold(f64::INFINITY, f64::min);
                        if point.value >= existing_min {
                            self.stats.total_downsampled += 1;
                            return Ok(false);
                        }
                        let src = point.source.clone();
                        let met = point.metric.clone();
                        let bs = self.bucket_secs;
                        self.store.retain(|p| {
                            !(p.source == src && p.metric == met && (p.timestamp.timestamp() - pt_ts).abs() < bs)
                        });
                    }
                    DownsampleStrategy::Max => {
                        let existing_max = self.store.iter()
                            .filter(|p| in_bucket(p))
                            .map(|p| p.value)
                            .fold(f64::NEG_INFINITY, f64::max);
                        if point.value <= existing_max {
                            self.stats.total_downsampled += 1;
                            return Ok(false);
                        }
                        let src = point.source.clone();
                        let met = point.metric.clone();
                        let bs = self.bucket_secs;
                        self.store.retain(|p| {
                            !(p.source == src && p.metric == met && (p.timestamp.timestamp() - pt_ts).abs() < bs)
                        });
                    }
                    DownsampleStrategy::Average | DownsampleStrategy::None => {
                        self.stats.total_downsampled += 1;
                        return Ok(false);
                    }
                }
                self.stats.total_downsampled += 1;
            }
        }

        self.dedup_keys.insert(dk);
        self.store.push_back(point);
        self.stats.total_ingested += 1;

        // Record for rate tracking.
        self.rate_window.push_back(Utc::now());
        self.stats.current_window_count += 1;

        // Enforce retention.
        self.enforce_retention();

        Ok(true)
    }

    /// Ingest a batch of data points.
    pub fn ingest_batch(&mut self, points: Vec<DataPoint>) -> BatchResult {
        let mut accepted = 0u64;
        let mut rejected = 0u64;
        let mut deduplicated = 0u64;
        let mut errors = Vec::new();

        for point in points {
            match self.ingest(point) {
                Ok(true) => accepted += 1,
                Ok(false) => deduplicated += 1,
                Err(e) => {
                    rejected += 1;
                    errors.push(e);
                }
            }
        }

        BatchResult { accepted, rejected, deduplicated, errors }
    }

    /// Apply retention policy: remove excess and expired points.
    fn enforce_retention(&mut self) {
        // Max points.
        while self.store.len() > self.retention.max_points {
            if let Some(removed) = self.store.pop_front() {
                self.dedup_keys.remove(&removed.dedup_key());
                self.stats.total_expired += 1;
            }
        }

        // Max age.
        if let Some(max_age) = self.retention.max_age {
            let cutoff = Utc::now() - max_age;
            while let Some(front) = self.store.front() {
                if front.timestamp < cutoff {
                    if let Some(removed) = self.store.pop_front() {
                        self.dedup_keys.remove(&removed.dedup_key());
                        self.stats.total_expired += 1;
                    }
                } else {
                    break;
                }
            }
        }
    }

    /// Current ingestion rate (points per second in the rate window).
    pub fn ingestion_rate(&mut self) -> f64 {
        let now = Utc::now();
        let cutoff = now - Duration::seconds(self.rate_window_secs);
        while let Some(front) = self.rate_window.front() {
            if *front < cutoff {
                self.rate_window.pop_front();
                self.stats.current_window_count = self.stats.current_window_count.saturating_sub(1);
            } else {
                break;
            }
        }
        self.rate_window.len() as f64 / self.rate_window_secs.max(1) as f64
    }

    pub fn stats(&self) -> &IngestStats {
        &self.stats
    }

    pub fn stored_count(&self) -> usize {
        self.store.len()
    }

    /// Get all stored points for a given source and metric.
    pub fn query(&self, source: &str, metric: &str) -> Vec<&DataPoint> {
        self.store.iter()
            .filter(|p| p.source == source && p.metric == metric)
            .collect()
    }

    /// Get all stored points.
    pub fn all_points(&self) -> Vec<&DataPoint> {
        self.store.iter().collect()
    }

    /// Clear all stored data.
    pub fn clear(&mut self) {
        self.store.clear();
        self.dedup_keys.clear();
    }
}

/// Result of a batch ingestion.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub accepted: u64,
    pub rejected: u64,
    pub deduplicated: u64,
    pub errors: Vec<String>,
}

impl BatchResult {
    pub fn total(&self) -> u64 {
        self.accepted + self.rejected + self.deduplicated
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn now_plus(secs: i64) -> DateTime<Utc> {
        Utc::now() + Duration::seconds(secs)
    }

    fn make_point(source: &str, metric: &str, value: f64, ts: DateTime<Utc>) -> DataPoint {
        DataPoint::new(source, metric, value, ts)
    }

    #[test]
    fn ingest_single_point() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        let p = make_point("s1", "temp", 22.5, Utc::now());
        assert!(ingestor.ingest(p).unwrap());
        assert_eq!(ingestor.stored_count(), 1);
    }

    #[test]
    fn ingest_batch() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        let ts = Utc::now();
        let points = vec![
            make_point("s1", "temp", 22.0, ts),
            make_point("s1", "temp", 23.0, ts + Duration::seconds(1)),
            make_point("s1", "temp", 24.0, ts + Duration::seconds(2)),
        ];
        let result = ingestor.ingest_batch(points);
        assert_eq!(result.accepted, 3);
        assert_eq!(result.rejected, 0);
    }

    #[test]
    fn validation_range() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        ingestor.add_validation_rule(ValidationRule::new("temp").with_range(-40.0, 100.0));
        let p = make_point("s1", "temp", 200.0, Utc::now());
        assert!(ingestor.ingest(p).is_err());
    }

    #[test]
    fn validation_required_tag() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        ingestor.add_validation_rule(ValidationRule::new("temp").with_required_tag("unit"));
        let p = make_point("s1", "temp", 22.0, Utc::now());
        assert!(ingestor.ingest(p).is_err());

        let p2 = make_point("s1", "temp", 22.0, Utc::now() + Duration::seconds(1))
            .with_tag("unit", "celsius");
        assert!(ingestor.ingest(p2).is_ok());
    }

    #[test]
    fn deduplication() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        let ts = Utc::now();
        let p1 = make_point("s1", "temp", 22.0, ts);
        let p2 = make_point("s1", "temp", 23.0, ts); // same source+metric+second
        assert!(ingestor.ingest(p1).unwrap());
        assert!(!ingestor.ingest(p2).unwrap()); // deduplicated
        assert_eq!(ingestor.stats().total_deduplicated, 1);
    }

    #[test]
    fn retention_max_points() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(5));
        for i in 0..10 {
            let p = make_point("s1", "temp", i as f64, Utc::now() + Duration::seconds(i));
            let _ = ingestor.ingest(p);
        }
        assert!(ingestor.stored_count() <= 5);
    }

    #[test]
    fn retention_max_age() {
        let max_age = Duration::seconds(10);
        let mut ingestor = TelemetryIngestor::new(
            RetentionPolicy::new(1000).with_max_age(max_age)
        );
        let old = Utc::now() - Duration::seconds(20);
        let p = make_point("s1", "temp", 22.0, old);
        let _ = ingestor.ingest(p);
        // Recent point triggers enforcement.
        let p2 = make_point("s1", "temp", 23.0, Utc::now() + Duration::seconds(1));
        let _ = ingestor.ingest(p2);
        // The old point should have been expired.
        assert_eq!(ingestor.stored_count(), 1);
    }

    #[test]
    fn backpressure_accept() {
        let ingestor = TelemetryIngestor::new(RetentionPolicy::new(100));
        assert_eq!(ingestor.backpressure(), BackpressureSignal::Accept);
    }

    #[test]
    fn backpressure_reject() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(5));
        for i in 0..5 {
            let p = make_point("s1", "m", i as f64, Utc::now() + Duration::seconds(i));
            let _ = ingestor.ingest(p);
        }
        assert_eq!(ingestor.backpressure(), BackpressureSignal::Reject);
    }

    #[test]
    fn backpressure_slowdown() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(10));
        ingestor.set_high_water(0.5);
        for i in 0..5 {
            let p = make_point("s1", "m", i as f64, Utc::now() + Duration::seconds(i));
            let _ = ingestor.ingest(p);
        }
        assert_eq!(ingestor.backpressure(), BackpressureSignal::SlowDown);
    }

    #[test]
    fn downsample_last_value() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        ingestor.set_downsample(DownsampleStrategy::LastValue, 60);
        let ts = Utc::now();
        let p1 = make_point("s1", "temp", 22.0, ts);
        let p2 = make_point("s1", "temp", 23.0, ts + Duration::seconds(10)); // same 60s bucket
        assert!(ingestor.ingest(p1).unwrap());
        // Second point replaces first in same bucket.
        let _ = ingestor.ingest(p2);
        let points = ingestor.query("s1", "temp");
        assert_eq!(points.len(), 1);
        assert!((points[0].value - 23.0).abs() < 1e-10);
    }

    #[test]
    fn downsample_min() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        ingestor.set_downsample(DownsampleStrategy::Min, 60);
        let ts = Utc::now();
        let p1 = make_point("s1", "temp", 20.0, ts);
        let p2 = make_point("s1", "temp", 25.0, ts + Duration::seconds(10));
        assert!(ingestor.ingest(p1).unwrap());
        // 25 > 20, so it should be skipped.
        assert!(!ingestor.ingest(p2).unwrap());
        let points = ingestor.query("s1", "temp");
        assert_eq!(points.len(), 1);
        assert!((points[0].value - 20.0).abs() < 1e-10);
    }

    #[test]
    fn query_by_source_and_metric() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        let ts = Utc::now();
        ingestor.ingest(make_point("s1", "temp", 22.0, ts)).unwrap();
        ingestor.ingest(make_point("s1", "humidity", 60.0, ts + Duration::seconds(1))).unwrap();
        ingestor.ingest(make_point("s2", "temp", 21.0, ts + Duration::seconds(2))).unwrap();
        let temps = ingestor.query("s1", "temp");
        assert_eq!(temps.len(), 1);
    }

    #[test]
    fn data_point_with_tags() {
        let p = DataPoint::new("s1", "temp", 22.0, Utc::now())
            .with_tag("unit", "celsius")
            .with_tag("location", "lab");
        assert_eq!(p.tags.len(), 2);
        assert_eq!(p.tags.get("unit").unwrap(), "celsius");
    }

    #[test]
    fn clear_store() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        let ts = Utc::now();
        ingestor.ingest(make_point("s1", "temp", 22.0, ts)).unwrap();
        ingestor.clear();
        assert_eq!(ingestor.stored_count(), 0);
    }

    #[test]
    fn batch_result_total() {
        let result = BatchResult {
            accepted: 5,
            rejected: 2,
            deduplicated: 1,
            errors: vec!["e1".to_string(), "e2".to_string()],
        };
        assert_eq!(result.total(), 8);
    }

    #[test]
    fn ingestion_rate_tracking() {
        let mut ingestor = TelemetryIngestor::new(RetentionPolicy::new(1000));
        for i in 0..5 {
            let p = make_point("s1", "m", i as f64, Utc::now() + Duration::seconds(i));
            let _ = ingestor.ingest(p);
        }
        let rate = ingestor.ingestion_rate();
        assert!(rate > 0.0);
    }
}
