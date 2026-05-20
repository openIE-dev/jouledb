use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;

use crate::counter::{Counter, CounterSnapshot};
use crate::gauge::{Gauge, GaugeSnapshot};
use crate::histogram::{Histogram, HistogramSnapshot};

/// A unified snapshot of all metrics from the registry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub counters: Vec<CounterSnapshot>,
    pub gauges: Vec<GaugeSnapshot>,
    pub histograms: Vec<HistogramSnapshot>,
}

/// A thread-safe registry for managing all metrics in a subsystem or node.
///
/// Metrics are registered by name. Requesting the same name twice returns
/// the existing metric (deduplication). The registry owns the canonical
/// instances; callers hold clones that share the underlying atomic state.
///
/// The registry itself is cheaply cloneable (Arc-wrapped) and safe to
/// share across threads.
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    counters: Arc<DashMap<String, Counter>>,
    gauges: Arc<DashMap<String, Gauge>>,
    histograms: Arc<DashMap<String, Histogram>>,
}

impl MetricsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            counters: Arc::new(DashMap::new()),
            gauges: Arc::new(DashMap::new()),
            histograms: Arc::new(DashMap::new()),
        }
    }

    // ---- counters -----------------------------------------------------------

    /// Get or create a counter with the given name and description.
    ///
    /// If a counter with this name already exists, the existing one is
    /// returned (description and labels are not updated).
    pub fn counter(&self, name: &str, description: &str) -> Counter {
        self.counters
            .entry(name.to_string())
            .or_insert_with(|| Counter::new(name, description))
            .clone()
    }

    /// Get or create a counter with labels.
    pub fn counter_with_labels(
        &self,
        name: &str,
        description: &str,
        labels: HashMap<String, String>,
    ) -> Counter {
        let key = Self::labeled_key(name, &labels);
        self.counters
            .entry(key)
            .or_insert_with(|| Counter::new(name, description).with_labels(labels))
            .clone()
    }

    /// Retrieve an existing counter by name.
    pub fn get_counter(&self, name: &str) -> Option<Counter> {
        self.counters.get(name).map(|c| c.clone())
    }

    // ---- gauges -------------------------------------------------------------

    /// Get or create a gauge with the given name and description.
    pub fn gauge(&self, name: &str, description: &str) -> Gauge {
        self.gauges
            .entry(name.to_string())
            .or_insert_with(|| Gauge::new(name, description))
            .clone()
    }

    /// Get or create a gauge with labels.
    pub fn gauge_with_labels(
        &self,
        name: &str,
        description: &str,
        labels: HashMap<String, String>,
    ) -> Gauge {
        let key = Self::labeled_key(name, &labels);
        self.gauges
            .entry(key)
            .or_insert_with(|| Gauge::new(name, description).with_labels(labels))
            .clone()
    }

    /// Retrieve an existing gauge by name.
    pub fn get_gauge(&self, name: &str) -> Option<Gauge> {
        self.gauges.get(name).map(|g| g.clone())
    }

    // ---- histograms ---------------------------------------------------------

    /// Get or create a histogram with the given name, description, and
    /// default latency buckets.
    pub fn histogram(&self, name: &str, description: &str) -> Histogram {
        self.histograms
            .entry(name.to_string())
            .or_insert_with(|| Histogram::new(name, description))
            .clone()
    }

    /// Get or create a histogram with custom buckets.
    pub fn histogram_with_buckets(
        &self,
        name: &str,
        description: &str,
        buckets: &[f64],
    ) -> Histogram {
        self.histograms
            .entry(name.to_string())
            .or_insert_with(|| Histogram::with_buckets(name, description, buckets))
            .clone()
    }

    /// Get or create a histogram with labels.
    pub fn histogram_with_labels(
        &self,
        name: &str,
        description: &str,
        labels: HashMap<String, String>,
    ) -> Histogram {
        let key = Self::labeled_key(name, &labels);
        self.histograms
            .entry(key)
            .or_insert_with(|| Histogram::new(name, description).with_labels(labels))
            .clone()
    }

    /// Retrieve an existing histogram by name.
    pub fn get_histogram(&self, name: &str) -> Option<Histogram> {
        self.histograms.get(name).map(|h| h.clone())
    }

    // ---- snapshot -----------------------------------------------------------

    /// Take a snapshot of all registered metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters: Vec<CounterSnapshot> =
            self.counters.iter().map(|c| c.value().snapshot()).collect();

        let gauges: Vec<GaugeSnapshot> = self.gauges.iter().map(|g| g.value().snapshot()).collect();

        let histograms: Vec<HistogramSnapshot> = self
            .histograms
            .iter()
            .map(|h| h.value().snapshot())
            .collect();

        MetricsSnapshot {
            counters,
            gauges,
            histograms,
        }
    }

    /// Total number of registered metrics across all types.
    pub fn metric_count(&self) -> usize {
        self.counters.len() + self.gauges.len() + self.histograms.len()
    }

    /// Number of registered counters.
    pub fn counter_count(&self) -> usize {
        self.counters.len()
    }

    /// Number of registered gauges.
    pub fn gauge_count(&self) -> usize {
        self.gauges.len()
    }

    /// Number of registered histograms.
    pub fn histogram_count(&self) -> usize {
        self.histograms.len()
    }

    // ---- helpers ------------------------------------------------------------

    /// Build a map key that incorporates label values for uniqueness.
    fn labeled_key(name: &str, labels: &HashMap<String, String>) -> String {
        if labels.is_empty() {
            return name.to_string();
        }
        let mut pairs: Vec<_> = labels.iter().collect();
        pairs.sort_by_key(|(k, _)| (*k).clone());
        let suffix: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{}_{{{}}}", name, suffix.join(","))
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry() {
        let reg = MetricsRegistry::new();
        assert_eq!(reg.metric_count(), 0);
    }

    #[test]
    fn counter_get_or_create() {
        let reg = MetricsRegistry::new();
        let c1 = reg.counter("requests", "Total requests");
        let c2 = reg.counter("requests", "ignored description");
        c1.inc();
        assert_eq!(c2.get(), 1, "Same counter returned");
        assert_eq!(reg.counter_count(), 1);
    }

    #[test]
    fn gauge_get_or_create() {
        let reg = MetricsRegistry::new();
        let g1 = reg.gauge("cpu", "CPU usage");
        let g2 = reg.gauge("cpu", "ignored");
        g1.set(42.0);
        assert!((g2.get() - 42.0).abs() < 0.01);
        assert_eq!(reg.gauge_count(), 1);
    }

    #[test]
    fn histogram_get_or_create() {
        let reg = MetricsRegistry::new();
        let h1 = reg.histogram("latency", "Request latency");
        let h2 = reg.histogram("latency", "ignored");
        h1.observe(0.5);
        assert_eq!(h2.count(), 1);
        assert_eq!(reg.histogram_count(), 1);
    }

    #[test]
    fn get_returns_existing() {
        let reg = MetricsRegistry::new();
        assert!(reg.get_counter("x").is_none());
        reg.counter("x", "");
        assert!(reg.get_counter("x").is_some());
    }

    #[test]
    fn labeled_counters_are_distinct() {
        let reg = MetricsRegistry::new();
        let mut labels_get = HashMap::new();
        labels_get.insert("method".into(), "GET".into());
        let mut labels_post = HashMap::new();
        labels_post.insert("method".into(), "POST".into());

        let c_get = reg.counter_with_labels("http_requests", "", labels_get);
        let c_post = reg.counter_with_labels("http_requests", "", labels_post);

        c_get.inc_by(10);
        c_post.inc_by(5);

        assert_eq!(c_get.get(), 10);
        assert_eq!(c_post.get(), 5);
        assert_eq!(reg.counter_count(), 2);
    }

    #[test]
    fn labeled_gauges_are_distinct() {
        let reg = MetricsRegistry::new();
        let mut l1 = HashMap::new();
        l1.insert("node".into(), "a".into());
        let mut l2 = HashMap::new();
        l2.insert("node".into(), "b".into());

        let g1 = reg.gauge_with_labels("mem", "", l1);
        let g2 = reg.gauge_with_labels("mem", "", l2);
        g1.set(100.0);
        g2.set(200.0);
        assert!((g1.get() - 100.0).abs() < 0.01);
        assert!((g2.get() - 200.0).abs() < 0.01);
    }

    #[test]
    fn metric_count_tracks_all_types() {
        let reg = MetricsRegistry::new();
        reg.counter("c1", "");
        reg.counter("c2", "");
        reg.gauge("g1", "");
        reg.histogram("h1", "");
        assert_eq!(reg.metric_count(), 4);
    }

    #[test]
    fn snapshot_captures_all() {
        let reg = MetricsRegistry::new();
        let c = reg.counter("req", "requests");
        let g = reg.gauge("cpu", "CPU");
        let h = reg.histogram("lat", "latency");

        c.inc_by(100);
        g.set(72.5);
        h.observe(0.05);

        let snap = reg.snapshot();
        assert_eq!(snap.counters.len(), 1);
        assert_eq!(snap.gauges.len(), 1);
        assert_eq!(snap.histograms.len(), 1);
        assert_eq!(snap.counters[0].value, 100);
        assert!((snap.gauges[0].value - 72.5).abs() < 0.01);
        assert_eq!(snap.histograms[0].count, 1);
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let reg = MetricsRegistry::new();
        reg.counter("x", "desc");
        let snap = reg.snapshot();
        let json = serde_json::to_value(&snap).unwrap();
        assert!(json["counters"].is_array());
        assert!(json["gauges"].is_array());
        assert!(json["histograms"].is_array());
    }

    #[test]
    fn histogram_with_custom_buckets() {
        let reg = MetricsRegistry::new();
        let h = reg.histogram_with_buckets("energy", "Energy", &[10.0, 100.0, 1000.0]);
        h.observe(50.0);
        let snap = h.snapshot();
        assert_eq!(snap.buckets.len(), 4); // 3 + Inf
    }

    #[test]
    fn labeled_key_deterministic() {
        let mut labels = HashMap::new();
        labels.insert("b".into(), "2".into());
        labels.insert("a".into(), "1".into());
        let key = MetricsRegistry::labeled_key("metric", &labels);
        assert_eq!(key, "metric_{a=1,b=2}");
    }

    #[test]
    fn default_creates_empty() {
        let reg = MetricsRegistry::default();
        assert_eq!(reg.metric_count(), 0);
    }

    #[test]
    fn concurrent_registry_access() {
        let reg = MetricsRegistry::new();
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let r = reg.clone();
                std::thread::spawn(move || {
                    let c = r.counter(&format!("counter_{i}"), "");
                    c.inc_by(100);
                    let g = r.gauge(&format!("gauge_{i}"), "");
                    g.set(i as f64);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(reg.counter_count(), 10);
        assert_eq!(reg.gauge_count(), 10);
    }
}
