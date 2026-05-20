//! Time Series Storage and Queries
//!
//! Provides time-partitioned storage with:
//! - Automatic partitioning by time
//! - Efficient time-range queries
//! - Downsampling/rollups
//! - Retention policies
//! - Optimized batch writes (1M+ writes/second)

pub mod optimized_write;
pub use optimized_write::{OptimizedTimeSeriesWriter, WriteBuffer};

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp in nanoseconds since epoch
    pub timestamp: i64,
    /// Value
    pub value: f64,
    /// Tags/labels
    pub tags: HashMap<String, String>,
}

impl DataPoint {
    /// Create new data point with current timestamp
    pub fn now(value: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        Self {
            timestamp,
            value,
            tags: HashMap::new(),
        }
    }

    /// Create data point with specific timestamp
    pub fn at(timestamp: i64, value: f64) -> Self {
        Self {
            timestamp,
            value,
            tags: HashMap::new(),
        }
    }

    /// Add tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }
}

/// Aggregation function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregation {
    Sum,
    Count,
    Mean,
    Min,
    Max,
    First,
    Last,
    Stddev,
    Variance,
    Percentile(u8),
}

/// Downsample policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsamplePolicy {
    /// Source retention (how long to keep raw data)
    pub source_retention: Duration,
    /// Target interval for downsampled data
    pub target_interval: Duration,
    /// Aggregation function
    pub aggregation: Aggregation,
    /// Target retention (how long to keep downsampled data)
    pub target_retention: Duration,
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Name
    pub name: String,
    /// Duration to keep data
    pub duration: Duration,
    /// Shard duration (time per partition)
    pub shard_duration: Duration,
    /// Replication factor
    pub replication_factor: u32,
    /// Is default policy
    pub is_default: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            duration: Duration::from_secs(7 * 24 * 3600), // 7 days
            shard_duration: Duration::from_secs(24 * 3600), // 1 day
            replication_factor: 1,
            is_default: true,
        }
    }
}

/// Time series configuration
#[derive(Debug, Clone)]
pub struct TimeSeriesConfig {
    /// Partition interval
    pub partition_interval: Duration,
    /// Default retention
    pub default_retention: Duration,
    /// Enable compression
    pub compression: bool,
    /// Downsample policies
    pub downsample_policies: Vec<DownsamplePolicy>,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            partition_interval: Duration::from_secs(3600), // 1 hour
            default_retention: Duration::from_secs(7 * 24 * 3600), // 7 days
            compression: true,
            downsample_policies: Vec::new(),
        }
    }
}

/// Time partition
#[derive(Debug)]
struct Partition {
    start_time: i64,
    end_time: i64,
    data: BTreeMap<i64, Vec<DataPoint>>,
}

impl Partition {
    fn new(start_time: i64, end_time: i64) -> Self {
        Self {
            start_time,
            end_time,
            data: BTreeMap::new(),
        }
    }

    fn insert(&mut self, point: DataPoint) {
        self.data
            .entry(point.timestamp)
            .or_insert_with(Vec::new)
            .push(point);
    }

    fn query(&self, start: i64, end: i64) -> Vec<DataPoint> {
        self.data
            .range(start..=end)
            .flat_map(|(_, points)| points.iter().cloned())
            .collect()
    }

    fn count(&self) -> usize {
        self.data.values().map(|v| v.len()).sum()
    }
}

/// Time series store
pub struct TimeSeriesStore {
    config: TimeSeriesConfig,
    /// Metrics: name -> partitions
    metrics: Arc<RwLock<HashMap<String, Vec<Partition>>>>,
    /// Retention policies
    retention_policies: Arc<RwLock<HashMap<String, RetentionPolicy>>>,
}

impl TimeSeriesStore {
    /// Create new time series store
    pub fn new(config: TimeSeriesConfig) -> Self {
        let mut policies = HashMap::new();
        policies.insert("default".to_string(), RetentionPolicy::default());

        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            retention_policies: Arc::new(RwLock::new(policies)),
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(TimeSeriesConfig::default())
    }

    /// Write a data point
    pub fn write(&self, metric: &str, point: DataPoint) {
        let mut metrics = self.metrics.write().unwrap();
        let partitions = metrics.entry(metric.to_string()).or_insert_with(Vec::new);

        // Find or create partition
        let partition_duration = self.config.partition_interval.as_nanos() as i64;
        let partition_start = (point.timestamp / partition_duration) * partition_duration;
        let partition_end = partition_start + partition_duration;

        let partition = partitions
            .iter_mut()
            .find(|p| p.start_time == partition_start);

        match partition {
            Some(p) => p.insert(point),
            None => {
                let mut p = Partition::new(partition_start, partition_end);
                p.insert(point);
                partitions.push(p);
            }
        }
    }

    /// Write multiple data points
    pub fn write_batch(&self, metric: &str, points: Vec<DataPoint>) {
        for point in points {
            self.write(metric, point);
        }
    }

    /// Optimized batch write (groups by partition, sorts, then writes)
    pub fn write_batch_optimized(&self, metric: &str, mut points: Vec<DataPoint>) {
        if points.is_empty() {
            return;
        }

        // Sort by timestamp for better compression and partition grouping
        points.sort_by_key(|p| p.timestamp);

        let mut metrics = self.metrics.write().unwrap();
        let partitions = metrics.entry(metric.to_string()).or_insert_with(Vec::new);

        let partition_duration = self.config.partition_interval.as_nanos() as i64;

        // Group points by partition
        let mut partition_map: std::collections::BTreeMap<i64, Vec<DataPoint>> =
            std::collections::BTreeMap::new();

        for point in points {
            let partition_start = (point.timestamp / partition_duration) * partition_duration;
            partition_map
                .entry(partition_start)
                .or_insert_with(Vec::new)
                .push(point);
        }

        // Write each partition's points in batch
        for (partition_start, partition_points) in partition_map {
            let partition_end = partition_start + partition_duration;

            // Find or create partition
            let partition = partitions
                .iter_mut()
                .find(|p| p.start_time == partition_start);

            match partition {
                Some(p) => {
                    // Batch insert into existing partition
                    for point in partition_points {
                        p.insert(point);
                    }
                }
                None => {
                    // Create new partition and batch insert
                    let mut p = Partition::new(partition_start, partition_end);
                    for point in partition_points {
                        p.insert(point);
                    }
                    partitions.push(p);
                }
            }
        }
    }

    /// Query time range
    pub fn query(&self, metric: &str, start: i64, end: i64) -> Vec<DataPoint> {
        let metrics = self.metrics.read().unwrap();
        let partitions = match metrics.get(metric) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut results = Vec::new();
        for partition in partitions {
            if partition.end_time >= start && partition.start_time <= end {
                results.extend(partition.query(start, end));
            }
        }

        results.sort_by_key(|p| p.timestamp);
        results
    }

    /// Query with aggregation
    pub fn query_aggregate(
        &self,
        metric: &str,
        start: i64,
        end: i64,
        interval: Duration,
        aggregation: Aggregation,
    ) -> Vec<DataPoint> {
        let points = self.query(metric, start, end);
        if points.is_empty() {
            return Vec::new();
        }

        let interval_ns = interval.as_nanos() as i64;
        let mut buckets: BTreeMap<i64, Vec<f64>> = BTreeMap::new();

        for point in points {
            let bucket_time = (point.timestamp / interval_ns) * interval_ns;
            buckets
                .entry(bucket_time)
                .or_insert_with(Vec::new)
                .push(point.value);
        }

        buckets
            .into_iter()
            .map(|(time, values)| {
                let value = Self::aggregate(&values, aggregation);
                DataPoint::at(time, value)
            })
            .collect()
    }

    fn aggregate(values: &[f64], aggregation: Aggregation) -> f64 {
        match aggregation {
            Aggregation::Sum => values.iter().sum(),
            Aggregation::Count => values.len() as f64,
            Aggregation::Mean => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            Aggregation::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
            Aggregation::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            Aggregation::First => *values.first().unwrap_or(&0.0),
            Aggregation::Last => *values.last().unwrap_or(&0.0),
            Aggregation::Stddev => {
                if values.len() < 2 {
                    return 0.0;
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                variance.sqrt()
            }
            Aggregation::Variance => {
                if values.len() < 2 {
                    return 0.0;
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
            }
            Aggregation::Percentile(p) => {
                if values.is_empty() {
                    return 0.0;
                }
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((p as f64 / 100.0) * (sorted.len() - 1) as f64) as usize;
                sorted[idx]
            }
        }
    }

    /// Get metric names
    pub fn list_metrics(&self) -> Vec<String> {
        self.metrics.read().unwrap().keys().cloned().collect()
    }

    /// Get metric count
    pub fn metric_count(&self, metric: &str) -> usize {
        self.metrics
            .read()
            .unwrap()
            .get(metric)
            .map(|partitions| partitions.iter().map(|p| p.count()).sum())
            .unwrap_or(0)
    }

    /// Delete metric
    pub fn delete_metric(&self, metric: &str) -> bool {
        self.metrics.write().unwrap().remove(metric).is_some()
    }

    /// Apply retention policy (remove old data)
    pub fn apply_retention(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;

        let retention_ns = self.config.default_retention.as_nanos() as i64;
        let cutoff = now - retention_ns;

        let mut metrics = self.metrics.write().unwrap();
        for partitions in metrics.values_mut() {
            partitions.retain(|p| p.end_time > cutoff);
        }
    }

    /// Add retention policy
    pub fn add_retention_policy(&self, policy: RetentionPolicy) {
        self.retention_policies
            .write()
            .unwrap()
            .insert(policy.name.clone(), policy);
    }

    /// Get retention policies
    pub fn get_retention_policies(&self) -> Vec<RetentionPolicy> {
        self.retention_policies
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_creation() {
        let point = DataPoint::at(1000, 42.0).with_tag("host", "server1");
        assert_eq!(point.timestamp, 1000);
        assert_eq!(point.value, 42.0);
        assert_eq!(point.tags.get("host"), Some(&"server1".to_string()));
    }

    #[test]
    fn test_write_and_query() {
        let store = TimeSeriesStore::with_defaults();

        store.write("cpu", DataPoint::at(1000, 50.0));
        store.write("cpu", DataPoint::at(2000, 60.0));
        store.write("cpu", DataPoint::at(3000, 55.0));

        let results = store.query("cpu", 0, 5000);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_range() {
        let store = TimeSeriesStore::with_defaults();

        store.write("cpu", DataPoint::at(1000, 50.0));
        store.write("cpu", DataPoint::at(2000, 60.0));
        store.write("cpu", DataPoint::at(3000, 55.0));

        let results = store.query("cpu", 1500, 2500);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, 60.0);
    }

    #[test]
    fn test_aggregation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::Sum), 15.0);
        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::Count), 5.0);
        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::Mean), 3.0);
        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::Min), 1.0);
        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::Max), 5.0);
        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::First), 1.0);
        assert_eq!(TimeSeriesStore::aggregate(&values, Aggregation::Last), 5.0);
    }

    #[test]
    fn test_query_aggregate() {
        let store = TimeSeriesStore::with_defaults();

        // Write points across multiple intervals
        for i in 0..10 {
            store.write("cpu", DataPoint::at(i * 1000, i as f64 * 10.0));
        }

        let results = store.query_aggregate(
            "cpu",
            0,
            10000,
            Duration::from_nanos(5000),
            Aggregation::Mean,
        );

        assert!(!results.is_empty());
    }

    #[test]
    fn test_list_metrics() {
        let store = TimeSeriesStore::with_defaults();

        store.write("cpu", DataPoint::at(1000, 50.0));
        store.write("memory", DataPoint::at(1000, 1024.0));

        let metrics = store.list_metrics();
        assert!(metrics.contains(&"cpu".to_string()));
        assert!(metrics.contains(&"memory".to_string()));
    }

    #[test]
    fn test_delete_metric() {
        let store = TimeSeriesStore::with_defaults();

        store.write("cpu", DataPoint::at(1000, 50.0));
        assert!(store.delete_metric("cpu"));
        assert!(!store.delete_metric("cpu"));
    }

    #[test]
    fn test_retention_policy() {
        let store = TimeSeriesStore::with_defaults();

        let policy = RetentionPolicy {
            name: "weekly".to_string(),
            duration: Duration::from_secs(7 * 24 * 3600),
            shard_duration: Duration::from_secs(24 * 3600),
            replication_factor: 1,
            is_default: false,
        };

        store.add_retention_policy(policy);
        let policies = store.get_retention_policies();
        assert!(policies.iter().any(|p| p.name == "weekly"));
    }
}
