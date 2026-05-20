//! Metric registry — register counters, gauges, histograms, and timers with
//! label dimensions, metric lookup, enumeration, scoped registries, and snapshots.
//!
//! Replaces `metrics`, `prometheus`, and `opentelemetry` registry layers with a
//! pure-Rust metric registry supporting dimensional labels, prefix scoping,
//! merge, removal, and snapshot export.

use std::collections::HashMap;
use std::fmt;

// ── Metric Value ────────────────────────────────────────────

/// The value held by a metric.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    /// Monotonically increasing counter.
    Counter(f64),
    /// Gauge that can go up and down.
    Gauge(f64),
    /// Histogram: (bucket_bounds, bucket_counts, sum, count).
    Histogram {
        bounds: Vec<f64>,
        counts: Vec<u64>,
        sum: f64,
        count: u64,
    },
    /// Timer: durations in nanoseconds (sum, count, min, max).
    Timer {
        sum_ns: u64,
        count: u64,
        min_ns: u64,
        max_ns: u64,
    },
}

impl MetricValue {
    pub fn kind_str(&self) -> &'static str {
        match self {
            MetricValue::Counter(_) => "counter",
            MetricValue::Gauge(_) => "gauge",
            MetricValue::Histogram { .. } => "histogram",
            MetricValue::Timer { .. } => "timer",
        }
    }
}

// ── Metric Entry ────────────────────────────────────────────

/// A registered metric with its name, labels, description, and value.
#[derive(Debug, Clone)]
pub struct MetricEntry {
    /// Metric name (fully qualified with any prefix).
    pub name: String,
    /// Description / help text.
    pub description: Option<String>,
    /// Label dimensions.
    pub labels: HashMap<String, String>,
    /// The metric value.
    pub value: MetricValue,
}

impl MetricEntry {
    pub fn kind(&self) -> &'static str {
        self.value.kind_str()
    }
}

// ── Metric Key ──────────────────────────────────────────────

/// A key uniquely identifying a metric (name + sorted labels).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetricKey {
    pub name: String,
    /// Labels sorted by key for deterministic hashing.
    pub labels: Vec<(String, String)>,
}

impl MetricKey {
    pub fn new(name: &str, labels: &HashMap<String, String>) -> Self {
        let mut sorted: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            name: name.to_string(),
            labels: sorted,
        }
    }

    pub fn simple(name: &str) -> Self {
        Self {
            name: name.to_string(),
            labels: Vec::new(),
        }
    }
}

impl fmt::Display for MetricKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.labels.is_empty() {
            write!(f, "{}", self.name)
        } else {
            let parts: Vec<String> = self.labels.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            write!(f, "{}[{}]", self.name, parts.join(","))
        }
    }
}

// ── Registry Snapshot ───────────────────────────────────────

/// An immutable snapshot of all metrics in a registry.
#[derive(Debug, Clone)]
pub struct RegistrySnapshot {
    pub entries: Vec<MetricEntry>,
}

impl RegistrySnapshot {
    /// Number of metrics.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Filter entries by kind.
    pub fn counters(&self) -> Vec<&MetricEntry> {
        self.entries.iter().filter(|e| matches!(e.value, MetricValue::Counter(_))).collect()
    }

    pub fn gauges(&self) -> Vec<&MetricEntry> {
        self.entries.iter().filter(|e| matches!(e.value, MetricValue::Gauge(_))).collect()
    }

    pub fn histograms(&self) -> Vec<&MetricEntry> {
        self.entries.iter().filter(|e| matches!(e.value, MetricValue::Histogram { .. })).collect()
    }

    pub fn timers(&self) -> Vec<&MetricEntry> {
        self.entries.iter().filter(|e| matches!(e.value, MetricValue::Timer { .. })).collect()
    }

    /// Find by name.
    pub fn get(&self, name: &str) -> Option<&MetricEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}

// ── Metric Registry ─────────────────────────────────────────

/// A registry of named metrics with label dimensions.
#[derive(Debug, Clone)]
pub struct MetricRegistry {
    /// Prefix prepended to all metric names.
    prefix: Option<String>,
    /// Metrics indexed by key.
    metrics: HashMap<MetricKey, MetricEntry>,
}

impl MetricRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            prefix: None,
            metrics: HashMap::new(),
        }
    }

    /// Create a scoped registry with a prefix.
    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            prefix: Some(prefix.to_string()),
            metrics: HashMap::new(),
        }
    }

    /// Apply the prefix to a name.
    fn prefixed(&self, name: &str) -> String {
        match &self.prefix {
            Some(p) => format!("{}.{}", p, name),
            None => name.to_string(),
        }
    }

    /// Get the prefix, if any.
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    // ── Registration ────────────────────────────────────────

    /// Register a counter with initial value 0.
    pub fn register_counter(&mut self, name: &str, description: Option<&str>) {
        self.register_counter_with_labels(name, description, HashMap::new());
    }

    /// Register a counter with labels.
    pub fn register_counter_with_labels(
        &mut self,
        name: &str,
        description: Option<&str>,
        labels: HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, &labels);
        let entry = MetricEntry {
            name: full_name,
            description: description.map(|s| s.to_string()),
            labels,
            value: MetricValue::Counter(0.0),
        };
        self.metrics.insert(key, entry);
    }

    /// Register a gauge with initial value 0.
    pub fn register_gauge(&mut self, name: &str, description: Option<&str>) {
        self.register_gauge_with_labels(name, description, HashMap::new());
    }

    /// Register a gauge with labels.
    pub fn register_gauge_with_labels(
        &mut self,
        name: &str,
        description: Option<&str>,
        labels: HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, &labels);
        let entry = MetricEntry {
            name: full_name,
            description: description.map(|s| s.to_string()),
            labels,
            value: MetricValue::Gauge(0.0),
        };
        self.metrics.insert(key, entry);
    }

    /// Register a histogram with explicit bounds.
    pub fn register_histogram(
        &mut self,
        name: &str,
        description: Option<&str>,
        bounds: &[f64],
    ) {
        self.register_histogram_with_labels(name, description, bounds, HashMap::new());
    }

    /// Register a histogram with labels.
    pub fn register_histogram_with_labels(
        &mut self,
        name: &str,
        description: Option<&str>,
        bounds: &[f64],
        labels: HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, &labels);
        let sorted_bounds = {
            let mut b = bounds.to_vec();
            b.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
            b
        };
        let n = sorted_bounds.len();
        let entry = MetricEntry {
            name: full_name,
            description: description.map(|s| s.to_string()),
            labels,
            value: MetricValue::Histogram {
                bounds: sorted_bounds,
                counts: vec![0; n + 1],
                sum: 0.0,
                count: 0,
            },
        };
        self.metrics.insert(key, entry);
    }

    /// Register a timer.
    pub fn register_timer(&mut self, name: &str, description: Option<&str>) {
        self.register_timer_with_labels(name, description, HashMap::new());
    }

    /// Register a timer with labels.
    pub fn register_timer_with_labels(
        &mut self,
        name: &str,
        description: Option<&str>,
        labels: HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, &labels);
        let entry = MetricEntry {
            name: full_name,
            description: description.map(|s| s.to_string()),
            labels,
            value: MetricValue::Timer {
                sum_ns: 0,
                count: 0,
                min_ns: u64::MAX,
                max_ns: 0,
            },
        };
        self.metrics.insert(key, entry);
    }

    // ── Mutation ────────────────────────────────────────────

    /// Increment a counter.
    pub fn counter_inc(&mut self, name: &str, value: f64) {
        self.counter_inc_with_labels(name, value, &HashMap::new());
    }

    /// Increment a counter with labels.
    pub fn counter_inc_with_labels(
        &mut self,
        name: &str,
        value: f64,
        labels: &HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, labels);
        if let Some(entry) = self.metrics.get_mut(&key) {
            if let MetricValue::Counter(ref mut v) = entry.value {
                if value >= 0.0 {
                    *v += value;
                }
            }
        }
    }

    /// Set a gauge value.
    pub fn gauge_set(&mut self, name: &str, value: f64) {
        self.gauge_set_with_labels(name, value, &HashMap::new());
    }

    /// Set a gauge with labels.
    pub fn gauge_set_with_labels(
        &mut self,
        name: &str,
        value: f64,
        labels: &HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, labels);
        if let Some(entry) = self.metrics.get_mut(&key) {
            if let MetricValue::Gauge(ref mut v) = entry.value {
                *v = value;
            }
        }
    }

    /// Observe a histogram value.
    pub fn histogram_observe(&mut self, name: &str, value: f64) {
        self.histogram_observe_with_labels(name, value, &HashMap::new());
    }

    /// Observe a histogram with labels.
    pub fn histogram_observe_with_labels(
        &mut self,
        name: &str,
        value: f64,
        labels: &HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, labels);
        if let Some(entry) = self.metrics.get_mut(&key) {
            if let MetricValue::Histogram {
                ref bounds,
                ref mut counts,
                ref mut sum,
                ref mut count,
            } = entry.value
            {
                *sum += value;
                *count += 1;
                let mut placed = false;
                for (i, bound) in bounds.iter().enumerate() {
                    if value <= *bound {
                        counts[i] += 1;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    let last = counts.len() - 1;
                    counts[last] += 1;
                }
            }
        }
    }

    /// Record a timer duration in nanoseconds.
    pub fn timer_record(&mut self, name: &str, duration_ns: u64) {
        self.timer_record_with_labels(name, duration_ns, &HashMap::new());
    }

    /// Record a timer duration with labels.
    pub fn timer_record_with_labels(
        &mut self,
        name: &str,
        duration_ns: u64,
        labels: &HashMap<String, String>,
    ) {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, labels);
        if let Some(entry) = self.metrics.get_mut(&key) {
            if let MetricValue::Timer {
                ref mut sum_ns,
                ref mut count,
                ref mut min_ns,
                ref mut max_ns,
            } = entry.value
            {
                *sum_ns += duration_ns;
                *count += 1;
                if duration_ns < *min_ns {
                    *min_ns = duration_ns;
                }
                if duration_ns > *max_ns {
                    *max_ns = duration_ns;
                }
            }
        }
    }

    // ── Lookup ──────────────────────────────────────────────

    /// Look up a metric by name (no labels).
    pub fn get(&self, name: &str) -> Option<&MetricEntry> {
        let full_name = self.prefixed(name);
        let key = MetricKey::simple(&full_name);
        self.metrics.get(&key)
    }

    /// Look up a metric by name and labels.
    pub fn get_with_labels(&self, name: &str, labels: &HashMap<String, String>) -> Option<&MetricEntry> {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, labels);
        self.metrics.get(&key)
    }

    // ── Enumeration ─────────────────────────────────────────

    /// Number of registered metrics.
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// All metric names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.metrics.keys().map(|k| k.name.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

    /// Get all entries as a Vec (sorted by name).
    pub fn entries(&self) -> Vec<&MetricEntry> {
        let mut entries: Vec<&MetricEntry> = self.metrics.values().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    // ── Removal ─────────────────────────────────────────────

    /// Remove a metric by name (no labels).
    pub fn remove(&mut self, name: &str) -> bool {
        let full_name = self.prefixed(name);
        let key = MetricKey::simple(&full_name);
        self.metrics.remove(&key).is_some()
    }

    /// Remove a metric by name and labels.
    pub fn remove_with_labels(&mut self, name: &str, labels: &HashMap<String, String>) -> bool {
        let full_name = self.prefixed(name);
        let key = MetricKey::new(&full_name, labels);
        self.metrics.remove(&key).is_some()
    }

    /// Remove all metrics whose names start with a prefix.
    pub fn remove_by_prefix(&mut self, prefix: &str) {
        let full_prefix = self.prefixed(prefix);
        self.metrics.retain(|k, _| !k.name.starts_with(&full_prefix));
    }

    // ── Snapshot & Merge ────────────────────────────────────

    /// Take a snapshot of all metrics.
    pub fn snapshot(&self) -> RegistrySnapshot {
        let mut entries: Vec<MetricEntry> = self.metrics.values().cloned().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        RegistrySnapshot { entries }
    }

    /// Merge another registry into this one (overwrites on conflict).
    pub fn merge(&mut self, other: &MetricRegistry) {
        for (key, entry) in &other.metrics {
            self.metrics.insert(key.clone(), entry.clone());
        }
    }

    /// Create a sub-scope (child registry with extended prefix).
    pub fn scope(&self, sub_prefix: &str) -> MetricRegistry {
        let new_prefix = match &self.prefix {
            Some(p) => format!("{}.{}", p, sub_prefix),
            None => sub_prefix.to_string(),
        };
        MetricRegistry::with_prefix(&new_prefix)
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_counter() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("requests", Some("Total requests"));
        let entry = reg.get("requests").unwrap();
        assert_eq!(entry.value, MetricValue::Counter(0.0));
        assert_eq!(entry.description.as_deref(), Some("Total requests"));
    }

    #[test]
    fn test_counter_inc() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("hits", None);
        reg.counter_inc("hits", 5.0);
        reg.counter_inc("hits", 3.0);
        let entry = reg.get("hits").unwrap();
        assert_eq!(entry.value, MetricValue::Counter(8.0));
    }

    #[test]
    fn test_counter_negative_ignored() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("c", None);
        reg.counter_inc("c", 10.0);
        reg.counter_inc("c", -5.0);
        let entry = reg.get("c").unwrap();
        assert_eq!(entry.value, MetricValue::Counter(10.0));
    }

    #[test]
    fn test_register_gauge() {
        let mut reg = MetricRegistry::new();
        reg.register_gauge("temp", Some("Temperature"));
        reg.gauge_set("temp", 72.5);
        let entry = reg.get("temp").unwrap();
        assert_eq!(entry.value, MetricValue::Gauge(72.5));
    }

    #[test]
    fn test_gauge_overwrite() {
        let mut reg = MetricRegistry::new();
        reg.register_gauge("g", None);
        reg.gauge_set("g", 10.0);
        reg.gauge_set("g", 20.0);
        let entry = reg.get("g").unwrap();
        assert_eq!(entry.value, MetricValue::Gauge(20.0));
    }

    #[test]
    fn test_register_histogram() {
        let mut reg = MetricRegistry::new();
        reg.register_histogram("latency", Some("Request latency"), &[0.1, 0.5, 1.0]);
        reg.histogram_observe("latency", 0.3);
        reg.histogram_observe("latency", 0.8);
        let entry = reg.get("latency").unwrap();
        if let MetricValue::Histogram { count, sum, .. } = &entry.value {
            assert_eq!(*count, 2);
            assert!((*sum - 1.1).abs() < 1e-9);
        } else {
            panic!("Expected histogram");
        }
    }

    #[test]
    fn test_register_timer() {
        let mut reg = MetricRegistry::new();
        reg.register_timer("handler_time", None);
        reg.timer_record("handler_time", 1_000_000);
        reg.timer_record("handler_time", 2_000_000);
        let entry = reg.get("handler_time").unwrap();
        if let MetricValue::Timer { sum_ns, count, min_ns, max_ns } = &entry.value {
            assert_eq!(*sum_ns, 3_000_000);
            assert_eq!(*count, 2);
            assert_eq!(*min_ns, 1_000_000);
            assert_eq!(*max_ns, 2_000_000);
        } else {
            panic!("Expected timer");
        }
    }

    #[test]
    fn test_prefix_scoped() {
        let mut reg = MetricRegistry::with_prefix("myapp");
        reg.register_counter("requests", None);
        // Internal name includes prefix
        let entry = reg.get("requests").unwrap();
        assert_eq!(entry.name, "myapp.requests");
    }

    #[test]
    fn test_sub_scope() {
        let parent = MetricRegistry::with_prefix("app");
        let child = parent.scope("http");
        assert_eq!(child.prefix(), Some("app.http"));
    }

    #[test]
    fn test_labels() {
        let mut reg = MetricRegistry::new();
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        reg.register_counter_with_labels("requests", None, labels.clone());
        reg.counter_inc_with_labels("requests", 1.0, &labels);
        let entry = reg.get_with_labels("requests", &labels).unwrap();
        assert_eq!(entry.value, MetricValue::Counter(1.0));
    }

    #[test]
    fn test_different_labels_different_metrics() {
        let mut reg = MetricRegistry::new();
        let mut labels_get = HashMap::new();
        labels_get.insert("method".to_string(), "GET".to_string());
        let mut labels_post = HashMap::new();
        labels_post.insert("method".to_string(), "POST".to_string());
        reg.register_counter_with_labels("requests", None, labels_get.clone());
        reg.register_counter_with_labels("requests", None, labels_post.clone());
        reg.counter_inc_with_labels("requests", 5.0, &labels_get);
        reg.counter_inc_with_labels("requests", 3.0, &labels_post);
        let get_entry = reg.get_with_labels("requests", &labels_get).unwrap();
        let post_entry = reg.get_with_labels("requests", &labels_post).unwrap();
        assert_eq!(get_entry.value, MetricValue::Counter(5.0));
        assert_eq!(post_entry.value, MetricValue::Counter(3.0));
    }

    #[test]
    fn test_remove() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("a", None);
        reg.register_counter("b", None);
        assert_eq!(reg.len(), 2);
        assert!(reg.remove("a"));
        assert_eq!(reg.len(), 1);
        assert!(!reg.remove("a")); // already removed
    }

    #[test]
    fn test_remove_by_prefix() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("http.requests", None);
        reg.register_counter("http.errors", None);
        reg.register_counter("db.queries", None);
        reg.remove_by_prefix("http");
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_snapshot() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("a", None);
        reg.register_gauge("b", None);
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap.counters().len(), 1);
        assert_eq!(snap.gauges().len(), 1);
    }

    #[test]
    fn test_merge() {
        let mut r1 = MetricRegistry::new();
        r1.register_counter("a", None);
        let mut r2 = MetricRegistry::new();
        r2.register_gauge("b", None);
        r1.merge(&r2);
        assert_eq!(r1.len(), 2);
    }

    #[test]
    fn test_names_sorted() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("z", None);
        reg.register_counter("a", None);
        reg.register_counter("m", None);
        let names = reg.names();
        assert_eq!(names, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_metric_key_display() {
        let key = MetricKey::simple("requests");
        assert_eq!(format!("{}", key), "requests");

        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        let key2 = MetricKey::new("requests", &labels);
        assert_eq!(format!("{}", key2), "requests[method=GET]");
    }

    #[test]
    fn test_entries_sorted() {
        let mut reg = MetricRegistry::new();
        reg.register_counter("z_metric", None);
        reg.register_counter("a_metric", None);
        let entries = reg.entries();
        assert_eq!(entries[0].name, "a_metric");
        assert_eq!(entries[1].name, "z_metric");
    }

    #[test]
    fn test_empty_registry() {
        let reg = MetricRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.names().is_empty());
    }

    #[test]
    fn test_metric_value_kind() {
        assert_eq!(MetricValue::Counter(0.0).kind_str(), "counter");
        assert_eq!(MetricValue::Gauge(0.0).kind_str(), "gauge");
        let h = MetricValue::Histogram {
            bounds: vec![],
            counts: vec![],
            sum: 0.0,
            count: 0,
        };
        assert_eq!(h.kind_str(), "histogram");
        let t = MetricValue::Timer {
            sum_ns: 0,
            count: 0,
            min_ns: 0,
            max_ns: 0,
        };
        assert_eq!(t.kind_str(), "timer");
    }
}
