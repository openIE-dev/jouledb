//! # Metrics and Monitoring System
//!
//! Comprehensive metrics collection and monitoring for JouleDB Server.
//!
//! ## Features
//!
//! - Counter, Gauge, and Histogram metric types
//! - Dimensional labels for metric filtering
//! - Prometheus-compatible exposition format
//! - Built-in database metrics (queries, storage, connections)
//! - Thread-safe metric collection

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Metric type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Monotonically increasing counter
    Counter,
    /// Value that can go up or down
    Gauge,
    /// Distribution of values with configurable buckets
    Histogram,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::Counter => write!(f, "counter"),
            MetricType::Gauge => write!(f, "gauge"),
            MetricType::Histogram => write!(f, "histogram"),
        }
    }
}

/// Labels for dimensional metric data
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Labels {
    labels: Vec<(String, String)>,
}

impl Labels {
    /// Create empty labels
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }

    /// Create labels from key-value pairs
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut labels: Vec<_> = pairs
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        labels.sort_by(|a, b| a.0.cmp(&b.0));
        Self { labels }
    }

    /// Add a label
    pub fn add<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.labels.push((key.into(), value.into()));
        self.labels.sort_by(|a, b| a.0.cmp(&b.0));
    }

    /// Get a label value
    pub fn get(&self, key: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Check if labels are empty
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Get the number of labels
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Iterate over labels
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.labels.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Format labels for Prometheus output
    pub fn to_prometheus_string(&self) -> String {
        if self.labels.is_empty() {
            return String::new();
        }
        let pairs: Vec<String> = self
            .labels
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
            .collect();
        format!("{{{}}}", pairs.join(","))
    }
}

/// Escape special characters in label values for Prometheus format
fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Counter metric - monotonically increasing value
#[derive(Debug)]
pub struct Counter {
    value: AtomicU64,
    name: String,
    help: String,
    labels: Labels,
}

impl Counter {
    /// Create a new counter
    pub fn new<N: Into<String>, H: Into<String>>(name: N, help: H, labels: Labels) -> Self {
        Self {
            value: AtomicU64::new(0),
            name: name.into(),
            help: help.into(),
            labels,
        }
    }

    /// Increment the counter by 1
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by a given amount
    pub fn inc_by(&self, amount: u64) {
        self.value.fetch_add(amount, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to 0
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }

    /// Get the metric name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the help text
    pub fn help(&self) -> &str {
        &self.help
    }

    /// Get the labels
    pub fn labels(&self) -> &Labels {
        &self.labels
    }
}

/// Gauge metric - value that can increase or decrease
#[derive(Debug)]
pub struct Gauge {
    value: AtomicI64,
    name: String,
    help: String,
    labels: Labels,
}

impl Gauge {
    /// Create a new gauge
    pub fn new<N: Into<String>, H: Into<String>>(name: N, help: H, labels: Labels) -> Self {
        Self {
            value: AtomicI64::new(0),
            name: name.into(),
            help: help.into(),
            labels,
        }
    }

    /// Set the gauge to a specific value
    pub fn set(&self, value: i64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the gauge by 1
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the gauge by 1
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }

    /// Add a value to the gauge
    pub fn add(&self, amount: i64) {
        self.value.fetch_add(amount, Ordering::Relaxed);
    }

    /// Subtract a value from the gauge
    pub fn sub(&self, amount: i64) {
        self.value.fetch_sub(amount, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Get the metric name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the help text
    pub fn help(&self) -> &str {
        &self.help
    }

    /// Get the labels
    pub fn labels(&self) -> &Labels {
        &self.labels
    }
}

/// Histogram bucket
#[derive(Debug)]
struct HistogramBucket {
    upper_bound: f64,
    count: AtomicU64,
}

/// Histogram metric - distribution of values
#[derive(Debug)]
pub struct Histogram {
    buckets: Vec<HistogramBucket>,
    sum: AtomicU64, // Stores f64 bits, uses CAS for atomic add
    count: AtomicU64,
    name: String,
    help: String,
    labels: Labels,
}

impl Histogram {
    /// Default bucket boundaries
    pub const DEFAULT_BUCKETS: [f64; 11] = [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ];

    /// Create a new histogram with default buckets
    pub fn new<N: Into<String>, H: Into<String>>(name: N, help: H, labels: Labels) -> Self {
        Self::with_buckets(name, help, labels, &Self::DEFAULT_BUCKETS)
    }

    /// Create a new histogram with custom buckets
    pub fn with_buckets<N: Into<String>, H: Into<String>>(
        name: N,
        help: H,
        labels: Labels,
        buckets: &[f64],
    ) -> Self {
        let mut sorted_buckets: Vec<f64> = buckets.to_vec();
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_buckets.push(f64::INFINITY);

        let histogram_buckets = sorted_buckets
            .into_iter()
            .map(|ub| HistogramBucket {
                upper_bound: ub,
                count: AtomicU64::new(0),
            })
            .collect();

        Self {
            buckets: histogram_buckets,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
            name: name.into(),
            help: help.into(),
            labels,
        }
    }

    /// Create buckets for latency measurements (in seconds)
    pub fn latency_buckets() -> Vec<f64> {
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ]
    }

    /// Create buckets for size measurements (in bytes)
    pub fn size_buckets() -> Vec<f64> {
        vec![
            100.0,
            1000.0,
            10000.0,
            100000.0,
            1000000.0,
            10000000.0,
            100000000.0,
        ]
    }

    /// Observe a value
    pub fn observe(&self, value: f64) {
        for bucket in &self.buckets {
            if value <= bucket.upper_bound {
                bucket.count.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Use compare-and-swap loop for atomic float addition
        loop {
            let current = self.sum.load(Ordering::Relaxed);
            let current_f64 = f64::from_bits(current);
            let new_f64 = current_f64 + value;
            let new_bits = new_f64.to_bits();
            match self.sum.compare_exchange_weak(
                current,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Observe a duration in seconds
    pub fn observe_duration(&self, duration: Duration) {
        self.observe(duration.as_secs_f64());
    }

    /// Start a timer that observes on drop
    pub fn start_timer(&self) -> HistogramTimer<'_> {
        HistogramTimer {
            histogram: self,
            start: Instant::now(),
        }
    }

    /// Get the count of observations
    pub fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the sum of all observations
    pub fn get_sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Get bucket counts
    pub fn get_buckets(&self) -> Vec<(f64, u64)> {
        self.buckets
            .iter()
            .map(|b| (b.upper_bound, b.count.load(Ordering::Relaxed)))
            .collect()
    }

    /// Get the metric name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the help text
    pub fn help(&self) -> &str {
        &self.help
    }

    /// Get the labels
    pub fn labels(&self) -> &Labels {
        &self.labels
    }

    /// Reset the histogram
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.count.store(0, Ordering::Relaxed);
        }
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

/// Timer for histogram observations
pub struct HistogramTimer<'a> {
    histogram: &'a Histogram,
    start: Instant,
}

impl<'a> Drop for HistogramTimer<'a> {
    fn drop(&mut self) {
        self.histogram.observe_duration(self.start.elapsed());
    }
}

/// Generic metric wrapper
#[derive(Debug, Clone)]
pub enum Metric {
    Counter(Arc<Counter>),
    Gauge(Arc<Gauge>),
    Histogram(Arc<Histogram>),
}

impl Metric {
    /// Get the metric name
    pub fn name(&self) -> &str {
        match self {
            Metric::Counter(c) => c.name(),
            Metric::Gauge(g) => g.name(),
            Metric::Histogram(h) => h.name(),
        }
    }

    /// Get the metric help text
    pub fn help(&self) -> &str {
        match self {
            Metric::Counter(c) => c.help(),
            Metric::Gauge(g) => g.help(),
            Metric::Histogram(h) => h.help(),
        }
    }

    /// Get the metric labels
    pub fn labels(&self) -> &Labels {
        match self {
            Metric::Counter(c) => c.labels(),
            Metric::Gauge(g) => g.labels(),
            Metric::Histogram(h) => h.labels(),
        }
    }

    /// Get the metric type
    pub fn metric_type(&self) -> MetricType {
        match self {
            Metric::Counter(_) => MetricType::Counter,
            Metric::Gauge(_) => MetricType::Gauge,
            Metric::Histogram(_) => MetricType::Histogram,
        }
    }
}

/// Registry for managing metrics
#[derive(Debug)]
pub struct MetricsRegistry {
    metrics: RwLock<HashMap<String, Metric>>,
    prefix: String,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            metrics: RwLock::new(HashMap::new()),
            prefix: String::new(),
        }
    }

    /// Create a new metrics registry with a prefix
    pub fn with_prefix<P: Into<String>>(prefix: P) -> Self {
        Self {
            metrics: RwLock::new(HashMap::new()),
            prefix: prefix.into(),
        }
    }

    /// Get the full metric name with prefix
    fn full_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}_{}", self.prefix, name)
        }
    }

    /// Generate a unique key for a metric with labels
    fn metric_key(&self, name: &str, labels: &Labels) -> String {
        let label_str = labels.to_prometheus_string();
        format!("{}{}", self.full_name(name), label_str)
    }

    /// Register a counter metric
    pub fn register_counter<N: Into<String>, H: Into<String>>(
        &self,
        name: N,
        help: H,
        labels: Labels,
    ) -> Arc<Counter> {
        let name = name.into();
        let full_name = self.full_name(&name);
        let key = self.metric_key(&name, &labels);
        let counter = Arc::new(Counter::new(full_name, help, labels));

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        metrics.insert(key, Metric::Counter(counter.clone()));
        counter
    }

    /// Register a gauge metric
    pub fn register_gauge<N: Into<String>, H: Into<String>>(
        &self,
        name: N,
        help: H,
        labels: Labels,
    ) -> Arc<Gauge> {
        let name = name.into();
        let full_name = self.full_name(&name);
        let key = self.metric_key(&name, &labels);
        let gauge = Arc::new(Gauge::new(full_name, help, labels));

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        metrics.insert(key, Metric::Gauge(gauge.clone()));
        gauge
    }

    /// Register a histogram metric
    pub fn register_histogram<N: Into<String>, H: Into<String>>(
        &self,
        name: N,
        help: H,
        labels: Labels,
    ) -> Arc<Histogram> {
        let name = name.into();
        let full_name = self.full_name(&name);
        let key = self.metric_key(&name, &labels);
        let histogram = Arc::new(Histogram::new(full_name, help, labels));

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        metrics.insert(key, Metric::Histogram(histogram.clone()));
        histogram
    }

    /// Register a histogram metric with custom buckets
    pub fn register_histogram_with_buckets<N: Into<String>, H: Into<String>>(
        &self,
        name: N,
        help: H,
        labels: Labels,
        buckets: &[f64],
    ) -> Arc<Histogram> {
        let name = name.into();
        let full_name = self.full_name(&name);
        let key = self.metric_key(&name, &labels);
        let histogram = Arc::new(Histogram::with_buckets(full_name, help, labels, buckets));

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        metrics.insert(key, Metric::Histogram(histogram.clone()));
        histogram
    }

    /// Get a metric by name and labels
    pub fn get_metric(&self, name: &str, labels: &Labels) -> Option<Metric> {
        let key = self.metric_key(name, labels);
        let metrics = crate::lock_util::read_lock(&self.metrics);
        metrics.get(&key).cloned()
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> Vec<Metric> {
        let metrics = crate::lock_util::read_lock(&self.metrics);
        metrics.values().cloned().collect()
    }

    /// Unregister a metric
    pub fn unregister(&self, name: &str, labels: &Labels) -> bool {
        let key = self.metric_key(name, labels);
        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        metrics.remove(&key).is_some()
    }

    /// Clear all metrics
    pub fn clear(&self) {
        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        metrics.clear();
    }

    /// Get the count of registered metrics
    pub fn metric_count(&self) -> usize {
        let metrics = crate::lock_util::read_lock(&self.metrics);
        metrics.len()
    }
}

/// Prometheus exporter for metrics
#[derive(Debug)]
pub struct PrometheusExporter {
    registry: Arc<MetricsRegistry>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(registry: Arc<MetricsRegistry>) -> Self {
        Self { registry }
    }

    /// Export all metrics in Prometheus text format
    pub fn export(&self) -> String {
        let metrics = self.registry.get_all_metrics();
        let mut output = String::new();
        let mut exported_help: std::collections::HashSet<String> = std::collections::HashSet::new();

        for metric in metrics {
            let name = metric.name();
            let help = metric.help();
            let labels = metric.labels();

            // Export HELP and TYPE only once per metric name
            if !exported_help.contains(name) {
                output.push_str(&format!("# HELP {} {}\n", name, help));
                output.push_str(&format!("# TYPE {} {}\n", name, metric.metric_type()));
                exported_help.insert(name.to_string());
            }

            match &metric {
                Metric::Counter(counter) => {
                    let label_str = labels.to_prometheus_string();
                    output.push_str(&format!("{}{} {}\n", name, label_str, counter.get()));
                }
                Metric::Gauge(gauge) => {
                    let label_str = labels.to_prometheus_string();
                    output.push_str(&format!("{}{} {}\n", name, label_str, gauge.get()));
                }
                Metric::Histogram(histogram) => {
                    self.export_histogram(&mut output, histogram);
                }
            }
        }

        output
    }

    /// Export a histogram in Prometheus format
    fn export_histogram(&self, output: &mut String, histogram: &Histogram) {
        let name = histogram.name();
        let base_labels = histogram.labels();

        for (upper_bound, count) in histogram.get_buckets() {
            let mut bucket_labels = base_labels.clone();
            if upper_bound.is_infinite() {
                bucket_labels.add("le", "+Inf");
            } else {
                bucket_labels.add("le", format!("{}", upper_bound));
            }
            output.push_str(&format!(
                "{}_bucket{} {}\n",
                name,
                bucket_labels.to_prometheus_string(),
                count
            ));
        }

        let label_str = base_labels.to_prometheus_string();
        output.push_str(&format!(
            "{}_sum{} {}\n",
            name,
            label_str,
            histogram.get_sum()
        ));
        output.push_str(&format!(
            "{}_count{} {}\n",
            name,
            label_str,
            histogram.get_count()
        ));
    }

    /// Export metrics as JSON
    pub fn export_json(&self) -> String {
        let metrics = self.registry.get_all_metrics();
        let mut json_metrics: Vec<serde_json::Value> = Vec::new();

        for metric in metrics {
            let labels_map: HashMap<String, String> = metric
                .labels()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();

            let value = match &metric {
                Metric::Counter(c) => serde_json::json!({
                    "name": metric.name(),
                    "type": "counter",
                    "help": metric.help(),
                    "labels": labels_map,
                    "value": c.get()
                }),
                Metric::Gauge(g) => serde_json::json!({
                    "name": metric.name(),
                    "type": "gauge",
                    "help": metric.help(),
                    "labels": labels_map,
                    "value": g.get()
                }),
                Metric::Histogram(h) => {
                    let buckets: Vec<serde_json::Value> = h
                        .get_buckets()
                        .iter()
                        .map(|(ub, count)| {
                            serde_json::json!({
                                "upper_bound": if ub.is_infinite() { "+Inf".to_string() } else { ub.to_string() },
                                "count": count
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "name": metric.name(),
                        "type": "histogram",
                        "help": metric.help(),
                        "labels": labels_map,
                        "buckets": buckets,
                        "sum": h.get_sum(),
                        "count": h.get_count()
                    })
                }
            };
            json_metrics.push(value);
        }

        serde_json::to_string_pretty(&json_metrics).unwrap_or_else(|_| "[]".to_string())
    }
}

/// Built-in database metrics collection
#[derive(Debug)]
pub struct DatabaseMetrics {
    registry: Arc<MetricsRegistry>,

    // Query metrics
    pub query_total: Arc<Counter>,
    pub query_errors: Arc<Counter>,
    pub query_latency: Arc<Histogram>,
    pub query_rows_returned: Arc<Histogram>,

    // Storage metrics
    pub page_reads: Arc<Counter>,
    pub page_writes: Arc<Counter>,
    pub cache_hits: Arc<Counter>,
    pub cache_misses: Arc<Counter>,
    pub cache_size: Arc<Gauge>,
    pub storage_bytes: Arc<Gauge>,

    // Connection metrics
    pub connections_active: Arc<Gauge>,
    pub connections_idle: Arc<Gauge>,
    pub connections_total: Arc<Counter>,
    pub connection_wait_time: Arc<Histogram>,
    pub connection_duration: Arc<Histogram>,

    // Transaction metrics
    pub transactions_total: Arc<Counter>,
    pub transactions_committed: Arc<Counter>,
    pub transactions_rolled_back: Arc<Counter>,
    pub transaction_duration: Arc<Histogram>,

    // Replication metrics
    pub replication_lag_seconds: Arc<Gauge>,
    pub replication_bytes_sent: Arc<Counter>,
    pub replication_bytes_received: Arc<Counter>,
}

impl DatabaseMetrics {
    /// Create a new database metrics collection
    pub fn new(registry: Arc<MetricsRegistry>) -> Self {
        let latency_buckets = Histogram::latency_buckets();
        let _size_buckets = Histogram::size_buckets();

        Self {
            // Query metrics
            query_total: registry.register_counter(
                "query_total",
                "Total number of queries executed",
                Labels::new(),
            ),
            query_errors: registry.register_counter(
                "query_errors_total",
                "Total number of query errors",
                Labels::new(),
            ),
            query_latency: registry.register_histogram_with_buckets(
                "query_latency_seconds",
                "Query latency in seconds",
                Labels::new(),
                &latency_buckets,
            ),
            query_rows_returned: registry.register_histogram_with_buckets(
                "query_rows_returned",
                "Number of rows returned per query",
                Labels::new(),
                &[1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
            ),

            // Storage metrics
            page_reads: registry.register_counter(
                "page_reads_total",
                "Total number of page reads",
                Labels::new(),
            ),
            page_writes: registry.register_counter(
                "page_writes_total",
                "Total number of page writes",
                Labels::new(),
            ),
            cache_hits: registry.register_counter(
                "cache_hits_total",
                "Total number of cache hits",
                Labels::new(),
            ),
            cache_misses: registry.register_counter(
                "cache_misses_total",
                "Total number of cache misses",
                Labels::new(),
            ),
            cache_size: registry.register_gauge(
                "cache_size_bytes",
                "Current cache size in bytes",
                Labels::new(),
            ),
            storage_bytes: registry.register_gauge(
                "storage_bytes",
                "Total storage size in bytes",
                Labels::new(),
            ),

            // Connection metrics
            connections_active: registry.register_gauge(
                "connections_active",
                "Number of active connections",
                Labels::new(),
            ),
            connections_idle: registry.register_gauge(
                "connections_idle",
                "Number of idle connections",
                Labels::new(),
            ),
            connections_total: registry.register_counter(
                "connections_total",
                "Total number of connections created",
                Labels::new(),
            ),
            connection_wait_time: registry.register_histogram_with_buckets(
                "connection_wait_seconds",
                "Time spent waiting for a connection",
                Labels::new(),
                &latency_buckets,
            ),
            connection_duration: registry.register_histogram_with_buckets(
                "connection_duration_seconds",
                "Duration of connection usage",
                Labels::new(),
                &[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
            ),

            // Transaction metrics
            transactions_total: registry.register_counter(
                "transactions_total",
                "Total number of transactions",
                Labels::new(),
            ),
            transactions_committed: registry.register_counter(
                "transactions_committed_total",
                "Total number of committed transactions",
                Labels::new(),
            ),
            transactions_rolled_back: registry.register_counter(
                "transactions_rolled_back_total",
                "Total number of rolled back transactions",
                Labels::new(),
            ),
            transaction_duration: registry.register_histogram_with_buckets(
                "transaction_duration_seconds",
                "Transaction duration in seconds",
                Labels::new(),
                &latency_buckets,
            ),

            // Replication metrics
            replication_lag_seconds: registry.register_gauge(
                "replication_lag_seconds",
                "Replication lag in seconds",
                Labels::new(),
            ),
            replication_bytes_sent: registry.register_counter(
                "replication_bytes_sent_total",
                "Total bytes sent for replication",
                Labels::new(),
            ),
            replication_bytes_received: registry.register_counter(
                "replication_bytes_received_total",
                "Total bytes received for replication",
                Labels::new(),
            ),

            registry,
        }
    }

    /// Record a successful query
    pub fn record_query(&self, latency: Duration, rows_returned: u64) {
        self.query_total.inc();
        self.query_latency.observe_duration(latency);
        self.query_rows_returned.observe(rows_returned as f64);
    }

    /// Record a query error
    pub fn record_query_error(&self) {
        self.query_total.inc();
        self.query_errors.inc();
    }

    /// Record a page read
    pub fn record_page_read(&self, cache_hit: bool) {
        self.page_reads.inc();
        if cache_hit {
            self.cache_hits.inc();
        } else {
            self.cache_misses.inc();
        }
    }

    /// Record a page write
    pub fn record_page_write(&self) {
        self.page_writes.inc();
    }

    /// Update cache size
    pub fn update_cache_size(&self, size_bytes: i64) {
        self.cache_size.set(size_bytes);
    }

    /// Update storage size
    pub fn update_storage_size(&self, size_bytes: i64) {
        self.storage_bytes.set(size_bytes);
    }

    /// Record a new connection
    pub fn record_connection_acquired(&self, wait_time: Duration) {
        self.connections_total.inc();
        self.connections_active.inc();
        self.connection_wait_time.observe_duration(wait_time);
    }

    /// Record a connection release
    pub fn record_connection_released(&self, duration: Duration) {
        self.connections_active.dec();
        self.connections_idle.inc();
        self.connection_duration.observe_duration(duration);
    }

    /// Record connection becoming active from idle
    pub fn record_connection_activated(&self) {
        self.connections_idle.dec();
        self.connections_active.inc();
    }

    /// Record a transaction start
    pub fn record_transaction_start(&self) {
        self.transactions_total.inc();
    }

    /// Record a transaction commit
    pub fn record_transaction_commit(&self, duration: Duration) {
        self.transactions_committed.inc();
        self.transaction_duration.observe_duration(duration);
    }

    /// Record a transaction rollback
    pub fn record_transaction_rollback(&self, duration: Duration) {
        self.transactions_rolled_back.inc();
        self.transaction_duration.observe_duration(duration);
    }

    /// Update replication lag
    pub fn update_replication_lag(&self, lag_seconds: f64) {
        self.replication_lag_seconds.set(lag_seconds as i64);
    }

    /// Record replication bytes sent
    pub fn record_replication_sent(&self, bytes: u64) {
        self.replication_bytes_sent.inc_by(bytes);
    }

    /// Record replication bytes received
    pub fn record_replication_received(&self, bytes: u64) {
        self.replication_bytes_received.inc_by(bytes);
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> DatabaseMetricsSnapshot {
        DatabaseMetricsSnapshot {
            query_total: self.query_total.get(),
            query_errors: self.query_errors.get(),
            query_latency_count: self.query_latency.get_count(),
            query_latency_sum: self.query_latency.get_sum(),
            page_reads: self.page_reads.get(),
            page_writes: self.page_writes.get(),
            cache_hits: self.cache_hits.get(),
            cache_misses: self.cache_misses.get(),
            cache_size_bytes: self.cache_size.get(),
            storage_bytes: self.storage_bytes.get(),
            connections_active: self.connections_active.get(),
            connections_idle: self.connections_idle.get(),
            connections_total: self.connections_total.get(),
            transactions_total: self.transactions_total.get(),
            transactions_committed: self.transactions_committed.get(),
            transactions_rolled_back: self.transactions_rolled_back.get(),
            replication_lag_seconds: self.replication_lag_seconds.get() as f64,
        }
    }

    /// Get the registry
    pub fn registry(&self) -> &Arc<MetricsRegistry> {
        &self.registry
    }
}

/// Snapshot of database metrics
#[derive(Debug, Clone)]
pub struct DatabaseMetricsSnapshot {
    pub query_total: u64,
    pub query_errors: u64,
    pub query_latency_count: u64,
    pub query_latency_sum: f64,
    pub page_reads: u64,
    pub page_writes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_size_bytes: i64,
    pub storage_bytes: i64,
    pub connections_active: i64,
    pub connections_idle: i64,
    pub connections_total: u64,
    pub transactions_total: u64,
    pub transactions_committed: u64,
    pub transactions_rolled_back: u64,
    pub replication_lag_seconds: f64,
}

impl DatabaseMetricsSnapshot {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Calculate query error rate
    pub fn query_error_rate(&self) -> f64 {
        if self.query_total == 0 {
            0.0
        } else {
            self.query_errors as f64 / self.query_total as f64
        }
    }

    /// Calculate average query latency
    pub fn avg_query_latency(&self) -> f64 {
        if self.query_latency_count == 0 {
            0.0
        } else {
            self.query_latency_sum / self.query_latency_count as f64
        }
    }

    /// Calculate transaction commit ratio
    pub fn transaction_commit_ratio(&self) -> f64 {
        if self.transactions_total == 0 {
            0.0
        } else {
            self.transactions_committed as f64 / self.transactions_total as f64
        }
    }
}

/// Helper to create labeled metrics
pub struct LabeledMetricFamily<T> {
    name: String,
    help: String,
    registry: Arc<MetricsRegistry>,
    metrics: RwLock<HashMap<Labels, Arc<T>>>,
    create_fn: Box<dyn Fn(&str, &str, Labels, &MetricsRegistry) -> Arc<T> + Send + Sync>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for LabeledMetricFamily<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LabeledMetricFamily")
            .field("name", &self.name)
            .field("help", &self.help)
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl LabeledMetricFamily<Counter> {
    /// Create a new labeled counter family
    pub fn new_counter<N: Into<String>, H: Into<String>>(
        name: N,
        help: H,
        registry: Arc<MetricsRegistry>,
    ) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            registry,
            metrics: RwLock::new(HashMap::new()),
            create_fn: Box::new(|name, help, labels, registry| {
                registry.register_counter(name, help, labels)
            }),
        }
    }

    /// Get or create a counter with the given labels
    pub fn with_labels(&self, labels: Labels) -> Arc<Counter> {
        {
            let metrics = crate::lock_util::read_lock(&self.metrics);
            if let Some(metric) = metrics.get(&labels) {
                return metric.clone();
            }
        }

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        if let Some(metric) = metrics.get(&labels) {
            return metric.clone();
        }

        let counter = (self.create_fn)(&self.name, &self.help, labels.clone(), &self.registry);
        metrics.insert(labels, counter.clone());
        counter
    }
}

impl LabeledMetricFamily<Gauge> {
    /// Create a new labeled gauge family
    pub fn new_gauge<N: Into<String>, H: Into<String>>(
        name: N,
        help: H,
        registry: Arc<MetricsRegistry>,
    ) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            registry,
            metrics: RwLock::new(HashMap::new()),
            create_fn: Box::new(|name, help, labels, registry| {
                registry.register_gauge(name, help, labels)
            }),
        }
    }

    /// Get or create a gauge with the given labels
    pub fn with_labels(&self, labels: Labels) -> Arc<Gauge> {
        {
            let metrics = crate::lock_util::read_lock(&self.metrics);
            if let Some(metric) = metrics.get(&labels) {
                return metric.clone();
            }
        }

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        if let Some(metric) = metrics.get(&labels) {
            return metric.clone();
        }

        let gauge = (self.create_fn)(&self.name, &self.help, labels.clone(), &self.registry);
        metrics.insert(labels, gauge.clone());
        gauge
    }
}

impl LabeledMetricFamily<Histogram> {
    /// Create a new labeled histogram family
    pub fn new_histogram<N: Into<String>, H: Into<String>>(
        name: N,
        help: H,
        registry: Arc<MetricsRegistry>,
    ) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            registry,
            metrics: RwLock::new(HashMap::new()),
            create_fn: Box::new(|name, help, labels, registry| {
                registry.register_histogram(name, help, labels)
            }),
        }
    }

    /// Get or create a histogram with the given labels
    pub fn with_labels(&self, labels: Labels) -> Arc<Histogram> {
        {
            let metrics = crate::lock_util::read_lock(&self.metrics);
            if let Some(metric) = metrics.get(&labels) {
                return metric.clone();
            }
        }

        let mut metrics = crate::lock_util::write_lock(&self.metrics);
        if let Some(metric) = metrics.get(&labels) {
            return metric.clone();
        }

        let histogram = (self.create_fn)(&self.name, &self.help, labels.clone(), &self.registry);
        metrics.insert(labels, histogram.clone());
        histogram
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_operations() {
        let counter = Counter::new("test_counter", "A test counter", Labels::new());

        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.inc_by(5);
        assert_eq!(counter.get(), 6);

        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_gauge_operations() {
        let gauge = Gauge::new("test_gauge", "A test gauge", Labels::new());

        assert_eq!(gauge.get(), 0);

        gauge.set(42);
        assert_eq!(gauge.get(), 42);

        gauge.inc();
        assert_eq!(gauge.get(), 43);

        gauge.dec();
        assert_eq!(gauge.get(), 42);

        gauge.add(10);
        assert_eq!(gauge.get(), 52);

        gauge.sub(20);
        assert_eq!(gauge.get(), 32);
    }

    #[test]
    fn test_histogram_operations() {
        let histogram = Histogram::new("test_histogram", "A test histogram", Labels::new());

        histogram.observe(0.1);
        histogram.observe(0.5);
        histogram.observe(1.0);

        assert_eq!(histogram.get_count(), 3);
        assert!((histogram.get_sum() - 1.6).abs() < 0.001);

        let buckets = histogram.get_buckets();
        assert!(!buckets.is_empty());

        // Verify bucket counts
        for (upper_bound, count) in &buckets {
            if *upper_bound >= 1.0 {
                assert!(*count >= 3, "All 3 observations should be <= 1.0");
            }
        }
    }

    #[test]
    fn test_histogram_timer() {
        let histogram = Histogram::new("timer_test", "Timer test", Labels::new());

        {
            let _timer = histogram.start_timer();
            std::thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(histogram.get_count(), 1);
        assert!(histogram.get_sum() >= 0.01);
    }

    #[test]
    fn test_labels() {
        let mut labels = Labels::new();
        assert!(labels.is_empty());

        labels.add("method", "GET");
        labels.add("status", "200");

        assert_eq!(labels.len(), 2);
        assert_eq!(labels.get("method"), Some("GET"));
        assert_eq!(labels.get("status"), Some("200"));
        assert_eq!(labels.get("nonexistent"), None);

        let prometheus_str = labels.to_prometheus_string();
        assert!(prometheus_str.contains("method=\"GET\""));
        assert!(prometheus_str.contains("status=\"200\""));
    }

    #[test]
    fn test_labels_from_pairs() {
        let labels = Labels::from_pairs([("method", "POST"), ("endpoint", "/api/v1")]);

        assert_eq!(labels.len(), 2);
        assert_eq!(labels.get("method"), Some("POST"));
        assert_eq!(labels.get("endpoint"), Some("/api/v1"));
    }

    #[test]
    fn test_labels_escape() {
        let mut labels = Labels::new();
        labels.add("message", "Hello \"World\"\nNew line");

        let prometheus_str = labels.to_prometheus_string();
        assert!(prometheus_str.contains("\\\""));
        assert!(prometheus_str.contains("\\n"));
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();

        let counter = registry.register_counter("requests", "Total requests", Labels::new());
        let gauge = registry.register_gauge("connections", "Active connections", Labels::new());
        let histogram = registry.register_histogram("latency", "Request latency", Labels::new());

        counter.inc();
        gauge.set(5);
        histogram.observe(0.1);

        assert_eq!(registry.metric_count(), 3);

        let metrics = registry.get_all_metrics();
        assert_eq!(metrics.len(), 3);
    }

    #[test]
    fn test_registry_with_prefix() {
        let registry = MetricsRegistry::with_prefix("joule_db");

        let counter = registry.register_counter("requests", "Total requests", Labels::new());

        assert_eq!(counter.name(), "joule_db_requests");
    }

    #[test]
    fn test_prometheus_exporter() {
        let registry = Arc::new(MetricsRegistry::new());

        let counter = registry.register_counter(
            "http_requests_total",
            "Total HTTP requests",
            Labels::from_pairs([("method", "GET")]),
        );
        counter.inc_by(100);

        let gauge =
            registry.register_gauge("active_connections", "Active connections", Labels::new());
        gauge.set(42);

        let histogram = registry.register_histogram(
            "request_duration_seconds",
            "Request duration",
            Labels::new(),
        );
        histogram.observe(0.1);
        histogram.observe(0.2);

        let exporter = PrometheusExporter::new(registry);
        let output = exporter.export();

        assert!(output.contains("# HELP http_requests_total"));
        assert!(output.contains("# TYPE http_requests_total counter"));
        assert!(output.contains("http_requests_total{method=\"GET\"} 100"));

        assert!(output.contains("# HELP active_connections"));
        assert!(output.contains("# TYPE active_connections gauge"));
        assert!(output.contains("active_connections 42"));

        assert!(output.contains("# HELP request_duration_seconds"));
        assert!(output.contains("# TYPE request_duration_seconds histogram"));
        assert!(output.contains("request_duration_seconds_bucket"));
        assert!(output.contains("request_duration_seconds_sum"));
        assert!(output.contains("request_duration_seconds_count 2"));
    }

    #[test]
    fn test_prometheus_exporter_json() {
        let registry = Arc::new(MetricsRegistry::new());

        let counter = registry.register_counter("test_counter", "A test counter", Labels::new());
        counter.inc_by(10);

        let exporter = PrometheusExporter::new(registry);
        let json = exporter.export_json();

        assert!(json.contains("test_counter"));
        assert!(json.contains("\"type\": \"counter\""));
        assert!(json.contains("\"value\": 10"));
    }

    #[test]
    fn test_database_metrics() {
        let registry = Arc::new(MetricsRegistry::with_prefix("joule_db"));
        let metrics = DatabaseMetrics::new(registry);

        // Test query recording
        metrics.record_query(Duration::from_millis(50), 100);
        metrics.record_query(Duration::from_millis(100), 50);
        metrics.record_query_error();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.query_total, 3);
        assert_eq!(snapshot.query_errors, 1);

        // Test cache metrics
        metrics.record_page_read(true);
        metrics.record_page_read(true);
        metrics.record_page_read(false);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.page_reads, 3);
        assert_eq!(snapshot.cache_hits, 2);
        assert_eq!(snapshot.cache_misses, 1);
        assert!((snapshot.cache_hit_ratio() - 0.666).abs() < 0.01);

        // Test connection metrics
        metrics.record_connection_acquired(Duration::from_millis(5));
        metrics.record_connection_activated();
        metrics.record_connection_released(Duration::from_secs(10));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.connections_total, 1);
    }

    #[test]
    fn test_database_metrics_snapshot_calculations() {
        let snapshot = DatabaseMetricsSnapshot {
            query_total: 100,
            query_errors: 5,
            query_latency_count: 95,
            query_latency_sum: 9.5,
            page_reads: 1000,
            page_writes: 100,
            cache_hits: 800,
            cache_misses: 200,
            cache_size_bytes: 1024 * 1024,
            storage_bytes: 10 * 1024 * 1024,
            connections_active: 10,
            connections_idle: 5,
            connections_total: 100,
            transactions_total: 50,
            transactions_committed: 45,
            transactions_rolled_back: 5,
            replication_lag_seconds: 0.5,
        };

        assert!((snapshot.cache_hit_ratio() - 0.8).abs() < 0.001);
        assert!((snapshot.query_error_rate() - 0.05).abs() < 0.001);
        assert!((snapshot.avg_query_latency() - 0.1).abs() < 0.001);
        assert!((snapshot.transaction_commit_ratio() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_labeled_metric_family_counter() {
        let registry = Arc::new(MetricsRegistry::new());
        let family =
            LabeledMetricFamily::new_counter("http_requests", "HTTP requests", registry.clone());

        let get_counter = family.with_labels(Labels::from_pairs([("method", "GET")]));
        let post_counter = family.with_labels(Labels::from_pairs([("method", "POST")]));

        get_counter.inc_by(100);
        post_counter.inc_by(50);

        assert_eq!(get_counter.get(), 100);
        assert_eq!(post_counter.get(), 50);

        // Same labels should return same counter
        let get_counter2 = family.with_labels(Labels::from_pairs([("method", "GET")]));
        assert_eq!(get_counter2.get(), 100);
    }

    #[test]
    fn test_labeled_metric_family_gauge() {
        let registry = Arc::new(MetricsRegistry::new());
        let family =
            LabeledMetricFamily::new_gauge("pool_size", "Connection pool size", registry.clone());

        let primary = family.with_labels(Labels::from_pairs([("pool", "primary")]));
        let replica = family.with_labels(Labels::from_pairs([("pool", "replica")]));

        primary.set(10);
        replica.set(5);

        assert_eq!(primary.get(), 10);
        assert_eq!(replica.get(), 5);
    }

    #[test]
    fn test_labeled_metric_family_histogram() {
        let registry = Arc::new(MetricsRegistry::new());
        let family = LabeledMetricFamily::new_histogram(
            "request_latency",
            "Request latency",
            registry.clone(),
        );

        let api_v1 = family.with_labels(Labels::from_pairs([("endpoint", "/api/v1")]));
        let api_v2 = family.with_labels(Labels::from_pairs([("endpoint", "/api/v2")]));

        api_v1.observe(0.1);
        api_v1.observe(0.2);
        api_v2.observe(0.05);

        assert_eq!(api_v1.get_count(), 2);
        assert_eq!(api_v2.get_count(), 1);
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(format!("{}", MetricType::Counter), "counter");
        assert_eq!(format!("{}", MetricType::Gauge), "gauge");
        assert_eq!(format!("{}", MetricType::Histogram), "histogram");
    }

    #[test]
    fn test_histogram_custom_buckets() {
        let buckets = vec![1.0, 5.0, 10.0, 50.0, 100.0];
        let histogram = Histogram::with_buckets(
            "custom_histogram",
            "Custom buckets",
            Labels::new(),
            &buckets,
        );

        histogram.observe(2.0);
        histogram.observe(7.0);
        histogram.observe(75.0);

        let bucket_data = histogram.get_buckets();
        // Buckets: 1.0, 5.0, 10.0, 50.0, 100.0, +Inf
        assert_eq!(bucket_data.len(), 6);

        // Value 2.0 is in buckets [5.0, 10.0, 50.0, 100.0, +Inf]
        // Value 7.0 is in buckets [10.0, 50.0, 100.0, +Inf]
        // Value 75.0 is in buckets [100.0, +Inf]
        assert_eq!(bucket_data[0].1, 0); // <= 1.0: 0
        assert_eq!(bucket_data[1].1, 1); // <= 5.0: 1 (2.0)
        assert_eq!(bucket_data[2].1, 2); // <= 10.0: 2 (2.0, 7.0)
        assert_eq!(bucket_data[3].1, 2); // <= 50.0: 2 (2.0, 7.0)
        assert_eq!(bucket_data[4].1, 3); // <= 100.0: 3 (2.0, 7.0, 75.0)
        assert_eq!(bucket_data[5].1, 3); // <= +Inf: 3
    }

    #[test]
    fn test_registry_unregister() {
        let registry = MetricsRegistry::new();

        let labels = Labels::from_pairs([("type", "test")]);
        let _counter = registry.register_counter("test_metric", "Test", labels.clone());

        assert_eq!(registry.metric_count(), 1);

        let removed = registry.unregister("test_metric", &labels);
        assert!(removed);
        assert_eq!(registry.metric_count(), 0);

        let removed_again = registry.unregister("test_metric", &labels);
        assert!(!removed_again);
    }

    #[test]
    fn test_registry_clear() {
        let registry = MetricsRegistry::new();

        registry.register_counter("counter1", "Counter 1", Labels::new());
        registry.register_gauge("gauge1", "Gauge 1", Labels::new());
        registry.register_histogram("histogram1", "Histogram 1", Labels::new());

        assert_eq!(registry.metric_count(), 3);

        registry.clear();
        assert_eq!(registry.metric_count(), 0);
    }

    #[test]
    fn test_transaction_metrics() {
        let registry = Arc::new(MetricsRegistry::new());
        let metrics = DatabaseMetrics::new(registry);

        metrics.record_transaction_start();
        metrics.record_transaction_start();
        metrics.record_transaction_start();

        metrics.record_transaction_commit(Duration::from_millis(100));
        metrics.record_transaction_commit(Duration::from_millis(150));
        metrics.record_transaction_rollback(Duration::from_millis(50));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.transactions_total, 3);
        assert_eq!(snapshot.transactions_committed, 2);
        assert_eq!(snapshot.transactions_rolled_back, 1);
    }

    #[test]
    fn test_replication_metrics() {
        let registry = Arc::new(MetricsRegistry::new());
        let metrics = DatabaseMetrics::new(registry);

        metrics.update_replication_lag(1.5);
        metrics.record_replication_sent(1024);
        metrics.record_replication_received(2048);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.replication_lag_seconds, 1.0); // Truncated to i64
        assert_eq!(metrics.replication_bytes_sent.get(), 1024);
        assert_eq!(metrics.replication_bytes_received.get(), 2048);
    }

    #[test]
    fn test_storage_metrics() {
        let registry = Arc::new(MetricsRegistry::new());
        let metrics = DatabaseMetrics::new(registry);

        metrics.update_cache_size(1024 * 1024); // 1MB
        metrics.update_storage_size(10 * 1024 * 1024); // 10MB

        for _ in 0..5 {
            metrics.record_page_write();
        }

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.cache_size_bytes, 1024 * 1024);
        assert_eq!(snapshot.storage_bytes, 10 * 1024 * 1024);
        assert_eq!(snapshot.page_writes, 5);
    }
}
