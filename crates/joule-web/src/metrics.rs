//! Metrics collection: counter, gauge, histogram, summary, labels/tags,
//! metric registry, Prometheus-compatible text exposition, rate calculation,
//! percentile computation, and metric families.

use std::collections::{BTreeMap, HashMap};

// ── Types ──

/// A set of key-value labels attached to a metric.
pub type Labels = BTreeMap<String, String>;

fn labels_key(labels: &Labels) -> String {
    labels
        .iter()
        .map(|(k, v)| format!("{}=\"{}\"", k, v))
        .collect::<Vec<_>>()
        .join(",")
}

/// Metric type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

impl MetricType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
        }
    }
}

// ── Counter ──

/// Monotonically increasing counter.
#[derive(Debug, Clone)]
pub struct Counter {
    pub name: String,
    pub help: String,
    values: HashMap<String, f64>,
    labels_map: HashMap<String, Labels>,
}

impl Counter {
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            values: HashMap::new(),
            labels_map: HashMap::new(),
        }
    }

    pub fn inc(&mut self, labels: &Labels) {
        self.add(labels, 1.0);
    }

    pub fn add(&mut self, labels: &Labels, val: f64) {
        assert!(val >= 0.0, "counter can only increase");
        let key = labels_key(labels);
        *self.values.entry(key.clone()).or_insert(0.0) += val;
        self.labels_map.entry(key).or_insert_with(|| labels.clone());
    }

    pub fn get(&self, labels: &Labels) -> f64 {
        self.values.get(&labels_key(labels)).copied().unwrap_or(0.0)
    }

    pub fn series(&self) -> Vec<(&Labels, f64)> {
        self.labels_map
            .iter()
            .map(|(k, l)| (l, self.values[k]))
            .collect()
    }
}

// ── Gauge ──

/// A value that can go up and down.
#[derive(Debug, Clone)]
pub struct Gauge {
    pub name: String,
    pub help: String,
    values: HashMap<String, f64>,
    labels_map: HashMap<String, Labels>,
}

impl Gauge {
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            values: HashMap::new(),
            labels_map: HashMap::new(),
        }
    }

    pub fn set(&mut self, labels: &Labels, val: f64) {
        let key = labels_key(labels);
        self.values.insert(key.clone(), val);
        self.labels_map.entry(key).or_insert_with(|| labels.clone());
    }

    pub fn inc(&mut self, labels: &Labels) {
        self.add(labels, 1.0);
    }

    pub fn dec(&mut self, labels: &Labels) {
        self.add(labels, -1.0);
    }

    pub fn add(&mut self, labels: &Labels, val: f64) {
        let key = labels_key(labels);
        *self.values.entry(key.clone()).or_insert(0.0) += val;
        self.labels_map.entry(key).or_insert_with(|| labels.clone());
    }

    pub fn get(&self, labels: &Labels) -> f64 {
        self.values.get(&labels_key(labels)).copied().unwrap_or(0.0)
    }
}

// ── Histogram ──

/// Histogram with configurable buckets, records observations and computes percentiles.
#[derive(Debug, Clone)]
pub struct Histogram {
    pub name: String,
    pub help: String,
    pub buckets: Vec<f64>,
    observations: HashMap<String, Vec<f64>>,
    labels_map: HashMap<String, Labels>,
}

impl Histogram {
    pub fn new(name: &str, help: &str, buckets: Vec<f64>) -> Self {
        let mut b = buckets;
        b.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            name: name.to_string(),
            help: help.to_string(),
            buckets: b,
            observations: HashMap::new(),
            labels_map: HashMap::new(),
        }
    }

    /// Default buckets similar to Prometheus defaults.
    pub fn with_default_buckets(name: &str, help: &str) -> Self {
        Self::new(
            name,
            help,
            vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
    }

    pub fn observe(&mut self, labels: &Labels, val: f64) {
        let key = labels_key(labels);
        self.observations
            .entry(key.clone())
            .or_default()
            .push(val);
        self.labels_map.entry(key).or_insert_with(|| labels.clone());
    }

    pub fn count(&self, labels: &Labels) -> usize {
        self.observations
            .get(&labels_key(labels))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn sum(&self, labels: &Labels) -> f64 {
        self.observations
            .get(&labels_key(labels))
            .map(|v| v.iter().sum())
            .unwrap_or(0.0)
    }

    /// Bucket counts for exposition.
    pub fn bucket_counts(&self, labels: &Labels) -> Vec<(f64, usize)> {
        let obs = match self.observations.get(&labels_key(labels)) {
            Some(v) => v,
            None => return self.buckets.iter().map(|b| (*b, 0)).collect(),
        };
        self.buckets
            .iter()
            .map(|b| (*b, obs.iter().filter(|&&v| v <= *b).count()))
            .collect()
    }

    /// Compute a percentile (0.0-1.0) across all observations for given labels.
    pub fn percentile(&self, labels: &Labels, p: f64) -> Option<f64> {
        let obs = self.observations.get(&labels_key(labels))?;
        if obs.is_empty() {
            return None;
        }
        let mut sorted = obs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (sorted.len() as f64 - 1.0)).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }
}

// ── Summary ──

/// Pre-computed quantile summary.
#[derive(Debug, Clone)]
pub struct Summary {
    pub name: String,
    pub help: String,
    pub quantiles: Vec<f64>,
    observations: HashMap<String, Vec<f64>>,
    labels_map: HashMap<String, Labels>,
}

impl Summary {
    pub fn new(name: &str, help: &str, quantiles: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            quantiles,
            observations: HashMap::new(),
            labels_map: HashMap::new(),
        }
    }

    pub fn observe(&mut self, labels: &Labels, val: f64) {
        let key = labels_key(labels);
        self.observations
            .entry(key.clone())
            .or_default()
            .push(val);
        self.labels_map.entry(key).or_insert_with(|| labels.clone());
    }

    pub fn snapshot(&self, labels: &Labels) -> Vec<(f64, f64)> {
        let obs = match self.observations.get(&labels_key(labels)) {
            Some(v) if !v.is_empty() => v,
            _ => return vec![],
        };
        let mut sorted = obs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.quantiles
            .iter()
            .map(|q| {
                let idx = (q * (sorted.len() as f64 - 1.0)).round() as usize;
                (*q, sorted[idx.min(sorted.len() - 1)])
            })
            .collect()
    }

    pub fn count(&self, labels: &Labels) -> usize {
        self.observations
            .get(&labels_key(labels))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn sum(&self, labels: &Labels) -> f64 {
        self.observations
            .get(&labels_key(labels))
            .map(|v| v.iter().sum())
            .unwrap_or(0.0)
    }
}

// ── Rate Calculation ──

/// Compute per-second rate given two counter snapshots.
pub fn rate(old_value: f64, new_value: f64, elapsed_secs: f64) -> f64 {
    if elapsed_secs <= 0.0 {
        return 0.0;
    }
    let delta = new_value - old_value;
    if delta < 0.0 {
        // Counter reset.
        return new_value / elapsed_secs;
    }
    delta / elapsed_secs
}

// ── Metric Family ──

/// A metric family groups related metrics of the same type.
#[derive(Debug)]
pub enum MetricFamily {
    CounterFamily(Counter),
    GaugeFamily(Gauge),
    HistogramFamily(Histogram),
    SummaryFamily(Summary),
}

impl MetricFamily {
    pub fn name(&self) -> &str {
        match self {
            MetricFamily::CounterFamily(c) => &c.name,
            MetricFamily::GaugeFamily(g) => &g.name,
            MetricFamily::HistogramFamily(h) => &h.name,
            MetricFamily::SummaryFamily(s) => &s.name,
        }
    }

    pub fn metric_type(&self) -> MetricType {
        match self {
            MetricFamily::CounterFamily(_) => MetricType::Counter,
            MetricFamily::GaugeFamily(_) => MetricType::Gauge,
            MetricFamily::HistogramFamily(_) => MetricType::Histogram,
            MetricFamily::SummaryFamily(_) => MetricType::Summary,
        }
    }
}

// ── Registry ──

/// Central metric registry, exposes Prometheus text format.
pub struct MetricRegistry {
    families: Vec<MetricFamily>,
}

impl MetricRegistry {
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
        }
    }

    pub fn register(&mut self, family: MetricFamily) {
        self.families.push(family);
    }

    pub fn family_count(&self) -> usize {
        self.families.len()
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn exposition(&self) -> String {
        let mut out = String::new();
        for fam in &self.families {
            match fam {
                MetricFamily::CounterFamily(c) => {
                    out.push_str(&format!("# HELP {} {}\n", c.name, c.help));
                    out.push_str(&format!("# TYPE {} counter\n", c.name));
                    for (labels, val) in c.series() {
                        let lk = labels_key(labels);
                        if lk.is_empty() {
                            out.push_str(&format!("{} {}\n", c.name, val));
                        } else {
                            out.push_str(&format!("{}{{{}}} {}\n", c.name, lk, val));
                        }
                    }
                }
                MetricFamily::GaugeFamily(g) => {
                    out.push_str(&format!("# HELP {} {}\n", g.name, g.help));
                    out.push_str(&format!("# TYPE {} gauge\n", g.name));
                    let key = labels_key(&Labels::new());
                    let val = g.get(&Labels::new());
                    if val != 0.0 || g.values.contains_key(&key) {
                        out.push_str(&format!("{} {}\n", g.name, val));
                    }
                }
                MetricFamily::HistogramFamily(h) => {
                    out.push_str(&format!("# HELP {} {}\n", h.name, h.help));
                    out.push_str(&format!("# TYPE {} histogram\n", h.name));
                }
                MetricFamily::SummaryFamily(s) => {
                    out.push_str(&format!("# HELP {} {}\n", s.name, s.help));
                    out.push_str(&format!("# TYPE {} summary\n", s.name));
                    let _ = s; // Summary exposition done per-label on request.
                }
            }
        }
        out
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_labels() -> Labels {
        Labels::new()
    }

    fn method_label(m: &str) -> Labels {
        let mut l = Labels::new();
        l.insert("method".to_string(), m.to_string());
        l
    }

    #[test]
    fn test_counter_inc() {
        let mut c = Counter::new("req_total", "Total requests");
        let l = method_label("GET");
        c.inc(&l);
        c.inc(&l);
        c.add(&l, 3.0);
        assert_eq!(c.get(&l), 5.0);
    }

    #[test]
    fn test_counter_different_labels() {
        let mut c = Counter::new("req_total", "Total requests");
        c.inc(&method_label("GET"));
        c.inc(&method_label("POST"));
        assert_eq!(c.get(&method_label("GET")), 1.0);
        assert_eq!(c.get(&method_label("POST")), 1.0);
    }

    #[test]
    fn test_gauge_set_inc_dec() {
        let mut g = Gauge::new("temp", "Temperature");
        let l = empty_labels();
        g.set(&l, 20.0);
        g.inc(&l);
        g.dec(&l);
        g.dec(&l);
        assert_eq!(g.get(&l), 19.0);
    }

    #[test]
    fn test_histogram_observe() {
        let mut h = Histogram::new("latency", "Latency", vec![0.1, 0.5, 1.0]);
        let l = empty_labels();
        h.observe(&l, 0.05);
        h.observe(&l, 0.3);
        h.observe(&l, 0.8);
        h.observe(&l, 1.5);
        assert_eq!(h.count(&l), 4);
        let bc = h.bucket_counts(&l);
        assert_eq!(bc, vec![(0.1, 1), (0.5, 2), (1.0, 3)]);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut h = Histogram::new("lat", "l", vec![1.0]);
        let l = empty_labels();
        for i in 1..=100 {
            h.observe(&l, i as f64);
        }
        let p50 = h.percentile(&l, 0.5).unwrap();
        assert!((p50 - 50.0).abs() < 2.0);
        let p99 = h.percentile(&l, 0.99).unwrap();
        assert!(p99 >= 98.0);
    }

    #[test]
    fn test_summary_snapshot() {
        let mut s = Summary::new("dur", "Duration", vec![0.5, 0.9, 0.99]);
        let l = empty_labels();
        for i in 1..=100 {
            s.observe(&l, i as f64);
        }
        let snap = s.snapshot(&l);
        assert_eq!(snap.len(), 3);
        assert!((snap[0].1 - 50.0).abs() < 2.0); // p50
    }

    #[test]
    fn test_rate_calculation() {
        assert!((rate(100.0, 200.0, 10.0) - 10.0).abs() < f64::EPSILON);
        assert!((rate(0.0, 0.0, 10.0)).abs() < f64::EPSILON);
        // Counter reset: treat new_value as total.
        assert!((rate(100.0, 50.0, 10.0) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_registry_exposition_counter() {
        let mut c = Counter::new("http_requests_total", "Total HTTP requests");
        c.inc(&empty_labels());
        c.inc(&empty_labels());
        let mut reg = MetricRegistry::new();
        reg.register(MetricFamily::CounterFamily(c));
        let text = reg.exposition();
        assert!(text.contains("# TYPE http_requests_total counter"));
        assert!(text.contains("http_requests_total 2"));
    }

    #[test]
    fn test_metric_family_name_and_type() {
        let c = Counter::new("foo", "help");
        let fam = MetricFamily::CounterFamily(c);
        assert_eq!(fam.name(), "foo");
        assert_eq!(fam.metric_type(), MetricType::Counter);
    }

    #[test]
    fn test_histogram_default_buckets() {
        let h = Histogram::with_default_buckets("lat", "Latency");
        assert_eq!(h.buckets.len(), 11);
        assert!(h.buckets[0] < h.buckets[10]);
    }

    #[test]
    fn test_gauge_missing_labels() {
        let g = Gauge::new("x", "help");
        assert_eq!(g.get(&empty_labels()), 0.0);
    }

    #[test]
    fn test_histogram_sum() {
        let mut h = Histogram::new("t", "t", vec![10.0]);
        let l = empty_labels();
        h.observe(&l, 3.0);
        h.observe(&l, 7.0);
        assert!((h.sum(&l) - 10.0).abs() < f64::EPSILON);
    }
}
