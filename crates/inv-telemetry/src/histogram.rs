use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Serialize;

/// Default latency-oriented buckets (in seconds).
pub const DEFAULT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Energy-oriented buckets (in microjoules).
pub fn energy_buckets() -> Vec<f64> {
    vec![
        10.0,
        50.0,
        100.0,
        500.0,
        1_000.0,
        5_000.0,
        10_000.0,
        50_000.0,
        100_000.0,
        500_000.0,
        1_000_000.0,
    ]
}

/// Internal state protected by a lock.
#[derive(Debug, Clone)]
struct HistogramInner {
    /// Upper bound for each bucket.
    bounds: Vec<f64>,
    /// Count of observations that fell into each bucket (cumulative).
    counts: Vec<u64>,
    /// Sum of all observed values.
    sum: f64,
    /// Total number of observations.
    total_count: u64,
    /// Minimum observed value (f64::MAX if no observations).
    min: f64,
    /// Maximum observed value (f64::MIN if no observations).
    max: f64,
}

/// A histogram metric that tracks the distribution of observed values.
///
/// Observations are bucketed into configurable upper-bound bins.
/// Supports computing percentiles, mean, min, max from the distribution.
/// Thread-safe via `Arc<RwLock<>>`.
#[derive(Debug, Clone)]
pub struct Histogram {
    name: String,
    description: String,
    labels: HashMap<String, String>,
    inner: Arc<RwLock<HistogramInner>>,
}

/// Point-in-time snapshot of a histogram's distribution.
#[derive(Debug, Clone, Serialize)]
pub struct HistogramSnapshot {
    pub name: String,
    pub description: String,
    pub labels: HashMap<String, String>,
    pub buckets: Vec<HistogramBucket>,
    pub sum: f64,
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// A single bucket in a histogram snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct HistogramBucket {
    pub upper_bound: f64,
    pub count: u64,
}

impl Histogram {
    /// Create a new histogram with the default latency-oriented buckets.
    pub fn new(name: &str, description: &str) -> Self {
        Self::with_buckets(name, description, DEFAULT_BUCKETS)
    }

    /// Create a histogram with custom bucket boundaries.
    ///
    /// Boundaries should be sorted in ascending order. A `+Inf` bucket
    /// is always appended automatically.
    pub fn with_buckets(name: &str, description: &str, bounds: &[f64]) -> Self {
        let mut sorted = bounds.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let counts = vec![0u64; sorted.len() + 1]; // +1 for +Inf

        Self {
            name: name.to_string(),
            description: description.to_string(),
            labels: HashMap::new(),
            inner: Arc::new(RwLock::new(HistogramInner {
                bounds: sorted,
                counts,
                sum: 0.0,
                total_count: 0,
                min: f64::MAX,
                max: f64::MIN,
            })),
        }
    }

    /// Add a single label key-value pair.
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Create a histogram with pre-defined labels.
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Record a single observation.
    pub fn observe(&self, value: f64) {
        let mut inner = self.inner.write().unwrap();
        inner.sum += value;
        inner.total_count += 1;

        if value < inner.min {
            inner.min = value;
        }
        if value > inner.max {
            inner.max = value;
        }

        // Find the first bucket whose upper bound >= value.
        let mut placed = false;
        for (i, bound) in inner.bounds.iter().enumerate() {
            if value <= *bound {
                inner.counts[i] += 1;
                placed = true;
                break;
            }
        }
        // If value exceeds all bounds, it goes into the +Inf bucket.
        if !placed {
            let last = inner.counts.len() - 1;
            inner.counts[last] += 1;
        }
    }

    /// Reset all observations. Primarily for testing.
    pub fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        for c in inner.counts.iter_mut() {
            *c = 0;
        }
        inner.sum = 0.0;
        inner.total_count = 0;
        inner.min = f64::MAX;
        inner.max = f64::MIN;
    }

    /// The total number of observations.
    pub fn count(&self) -> u64 {
        self.inner.read().unwrap().total_count
    }

    /// The sum of all observed values.
    pub fn sum(&self) -> f64 {
        self.inner.read().unwrap().sum
    }

    /// The histogram's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The histogram's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The histogram's labels.
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    /// Take a point-in-time snapshot with computed percentiles.
    pub fn snapshot(&self) -> HistogramSnapshot {
        let inner = self.inner.read().unwrap();

        let mut buckets = Vec::with_capacity(inner.bounds.len() + 1);
        for (i, bound) in inner.bounds.iter().enumerate() {
            buckets.push(HistogramBucket {
                upper_bound: *bound,
                count: inner.counts[i],
            });
        }
        // +Inf bucket
        buckets.push(HistogramBucket {
            upper_bound: f64::INFINITY,
            count: *inner.counts.last().unwrap_or(&0),
        });

        let mean = if inner.total_count > 0 {
            inner.sum / inner.total_count as f64
        } else {
            0.0
        };

        let min = if inner.total_count > 0 {
            inner.min
        } else {
            0.0
        };
        let max = if inner.total_count > 0 {
            inner.max
        } else {
            0.0
        };

        // Estimate percentiles from the cumulative bucket distribution.
        let p50 = Self::estimate_percentile(&inner.bounds, &inner.counts, inner.total_count, 0.50);
        let p95 = Self::estimate_percentile(&inner.bounds, &inner.counts, inner.total_count, 0.95);
        let p99 = Self::estimate_percentile(&inner.bounds, &inner.counts, inner.total_count, 0.99);

        HistogramSnapshot {
            name: self.name.clone(),
            description: self.description.clone(),
            labels: self.labels.clone(),
            buckets,
            sum: inner.sum,
            count: inner.total_count,
            min,
            max,
            mean,
            p50,
            p95,
            p99,
        }
    }

    /// Estimate a percentile from the bucket distribution using linear
    /// interpolation within the target bucket.
    fn estimate_percentile(bounds: &[f64], counts: &[u64], total: u64, quantile: f64) -> f64 {
        if total == 0 {
            return 0.0;
        }

        let target = (quantile * total as f64).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, count) in counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                if i < bounds.len() {
                    // Linear interpolation within the bucket.
                    let prev_bound = if i == 0 { 0.0 } else { bounds[i - 1] };
                    let bucket_count = *count;
                    if bucket_count == 0 {
                        return bounds[i];
                    }
                    let within_bucket = target - (cumulative - bucket_count);
                    let fraction = within_bucket as f64 / bucket_count as f64;
                    return prev_bound + fraction * (bounds[i] - prev_bound);
                } else {
                    // +Inf bucket — return the last finite bound as best estimate.
                    return bounds.last().copied().unwrap_or(0.0);
                }
            }
        }

        bounds.last().copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_histogram_empty() {
        let h = Histogram::new("latency", "Request latency");
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.name(), "latency");
    }

    #[test]
    fn observe_increments_count_and_sum() {
        let h = Histogram::new("lat", "");
        h.observe(0.1);
        h.observe(0.2);
        h.observe(0.3);
        assert_eq!(h.count(), 3);
        assert!((h.sum() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn observations_land_in_correct_buckets() {
        let h = Histogram::with_buckets("x", "", &[1.0, 5.0, 10.0]);
        h.observe(0.5); // bucket 1.0
        h.observe(3.0); // bucket 5.0
        h.observe(7.0); // bucket 10.0
        h.observe(15.0); // +Inf bucket

        let snap = h.snapshot();
        assert_eq!(snap.buckets[0].count, 1); // ≤1.0
        assert_eq!(snap.buckets[1].count, 1); // ≤5.0
        assert_eq!(snap.buckets[2].count, 1); // ≤10.0
        assert_eq!(snap.buckets[3].count, 1); // +Inf
    }

    #[test]
    fn min_max_tracked() {
        let h = Histogram::new("x", "");
        h.observe(5.0);
        h.observe(1.0);
        h.observe(10.0);

        let snap = h.snapshot();
        assert!((snap.min - 1.0).abs() < 1e-9);
        assert!((snap.max - 10.0).abs() < 1e-9);
    }

    #[test]
    fn mean_computed_correctly() {
        let h = Histogram::new("x", "");
        h.observe(2.0);
        h.observe(4.0);
        h.observe(6.0);

        let snap = h.snapshot();
        assert!((snap.mean - 4.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_estimation() {
        let h = Histogram::with_buckets("x", "", &[10.0, 20.0, 50.0, 100.0]);
        // 100 observations spread across buckets
        for _ in 0..50 {
            h.observe(5.0); // bucket ≤10
        }
        for _ in 0..30 {
            h.observe(15.0); // bucket ≤20
        }
        for _ in 0..15 {
            h.observe(30.0); // bucket ≤50
        }
        for _ in 0..5 {
            h.observe(75.0); // bucket ≤100
        }

        let snap = h.snapshot();
        assert_eq!(snap.count, 100);
        // p50 should be in the ≤10 bucket (first 50 observations)
        assert!(snap.p50 <= 10.0, "p50={}", snap.p50);
        // p95 should be in the ≤50 bucket
        assert!(snap.p95 > 10.0 && snap.p95 <= 50.0, "p95={}", snap.p95);
        // p99 should be in the ≤100 bucket
        assert!(snap.p99 > 20.0 && snap.p99 <= 100.0, "p99={}", snap.p99);
    }

    #[test]
    fn empty_histogram_snapshot_zeros() {
        let h = Histogram::new("x", "");
        let snap = h.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.mean, 0.0);
        assert_eq!(snap.min, 0.0);
        assert_eq!(snap.max, 0.0);
        assert_eq!(snap.p50, 0.0);
    }

    #[test]
    fn reset_clears_all() {
        let h = Histogram::new("x", "");
        h.observe(1.0);
        h.observe(2.0);
        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn custom_energy_buckets() {
        let buckets = energy_buckets();
        let h = Histogram::with_buckets("energy", "Energy per op", &buckets);
        h.observe(100.0);
        h.observe(5000.0);
        assert_eq!(h.count(), 2);
    }

    #[test]
    fn labels_on_histogram() {
        let h = Histogram::new("lat", "Latency")
            .with_label("service", "api")
            .with_label("endpoint", "/health");
        assert_eq!(h.labels().get("service").unwrap(), "api");
        assert_eq!(h.labels().len(), 2);
    }

    #[test]
    fn clone_shares_inner_state() {
        let h1 = Histogram::new("x", "");
        let h2 = h1.clone();
        h1.observe(1.0);
        assert_eq!(h2.count(), 1);
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let h = Histogram::with_buckets("test", "desc", &[1.0, 5.0]);
        h.observe(0.5);
        h.observe(3.0);
        let snap = h.snapshot();
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["name"], "test");
        assert_eq!(json["count"], 2);
        assert!(json["buckets"].as_array().unwrap().len() == 3); // 2 + Inf
    }

    #[test]
    fn default_buckets_are_sorted() {
        for i in 1..DEFAULT_BUCKETS.len() {
            assert!(
                DEFAULT_BUCKETS[i] > DEFAULT_BUCKETS[i - 1],
                "Buckets must be sorted"
            );
        }
    }
}
