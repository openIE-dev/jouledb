//! Prometheus exposition format — metric types, label pairs, families, and text output.
//!
//! Replaces `prometheus` and `prometheus-client` crates with a pure-Rust
//! implementation of the Prometheus exposition format (text/plain 0.0.4).
//! Supports counter, gauge, histogram, and summary metric types, label pairs,
//! metric families, HELP/TYPE lines, optional timestamps, and metric registration.

use std::collections::HashMap;
use std::fmt;

// ── Label Pair ──────────────────────────────────────────────

/// A single label name=value pair attached to a metric.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelPair {
    pub name: String,
    pub value: String,
}

impl LabelPair {
    pub fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    /// Format as Prometheus label: `name="value"` with escaping.
    pub fn format(&self) -> String {
        let escaped = self
            .value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        format!("{}=\"{}\"", self.name, escaped)
    }
}

// ── Metric Type ─────────────────────────────────────────────

/// Prometheus metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Untyped,
}

impl MetricType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
            MetricType::Untyped => "untyped",
        }
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Metric Sample ───────────────────────────────────────────

/// A single sample (one line in exposition format).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSample {
    /// Metric name (may include suffix like `_total`, `_bucket`, `_sum`, `_count`).
    pub name: String,
    /// Labels for this sample.
    pub labels: Vec<LabelPair>,
    /// The value.
    pub value: f64,
    /// Optional timestamp in milliseconds since epoch.
    pub timestamp_ms: Option<i64>,
}

impl MetricSample {
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            labels: Vec::new(),
            value,
            timestamp_ms: None,
        }
    }

    pub fn with_label(mut self, name: &str, value: &str) -> Self {
        self.labels.push(LabelPair::new(name, value));
        self
    }

    pub fn with_labels(mut self, labels: Vec<LabelPair>) -> Self {
        self.labels = labels;
        self
    }

    pub fn with_timestamp(mut self, ts_ms: i64) -> Self {
        self.timestamp_ms = Some(ts_ms);
        self
    }

    /// Format as a single exposition line.
    pub fn format_line(&self) -> String {
        let labels_str = if self.labels.is_empty() {
            String::new()
        } else {
            let parts: Vec<String> = self.labels.iter().map(|l| l.format()).collect();
            format!("{{{}}}", parts.join(","))
        };

        let value_str = format_value(self.value);

        match self.timestamp_ms {
            Some(ts) => format!("{}{} {} {}", self.name, labels_str, value_str, ts),
            None => format!("{}{} {}", self.name, labels_str, value_str),
        }
    }
}

/// Format a float for Prometheus: +Inf, -Inf, NaN, or plain number.
fn format_value(v: f64) -> String {
    if v.is_infinite() && v.is_sign_positive() {
        "+Inf".to_string()
    } else if v.is_infinite() && v.is_sign_negative() {
        "-Inf".to_string()
    } else if v.is_nan() {
        "NaN".to_string()
    } else if v == v.floor() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

// ── Metric Family ───────────────────────────────────────────

/// A family of metrics sharing the same name, HELP text, TYPE, and samples.
#[derive(Debug, Clone)]
pub struct MetricFamily {
    pub name: String,
    pub help: Option<String>,
    pub metric_type: MetricType,
    pub samples: Vec<MetricSample>,
}

impl MetricFamily {
    pub fn new(name: &str, metric_type: MetricType) -> Self {
        Self {
            name: name.to_string(),
            help: None,
            metric_type,
            samples: Vec::new(),
        }
    }

    pub fn with_help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    pub fn add_sample(&mut self, sample: MetricSample) {
        self.samples.push(sample);
    }

    /// Format as Prometheus exposition text block.
    pub fn format(&self) -> String {
        let mut out = String::new();
        if let Some(help) = &self.help {
            let escaped = help.replace('\\', "\\\\").replace('\n', "\\n");
            out.push_str(&format!("# HELP {} {}\n", self.name, escaped));
        }
        out.push_str(&format!("# TYPE {} {}\n", self.name, self.metric_type));
        for sample in &self.samples {
            out.push_str(&sample.format_line());
            out.push('\n');
        }
        out
    }
}

// ── Counter ─────────────────────────────────────────────────

/// A monotonically increasing counter.
#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    help: Option<String>,
    labels: Vec<LabelPair>,
    value: f64,
    timestamp_ms: Option<i64>,
}

impl Counter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            help: None,
            labels: Vec::new(),
            value: 0.0,
            timestamp_ms: None,
        }
    }

    pub fn with_help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    pub fn with_label(mut self, name: &str, value: &str) -> Self {
        self.labels.push(LabelPair::new(name, value));
        self
    }

    pub fn with_timestamp(mut self, ts_ms: i64) -> Self {
        self.timestamp_ms = Some(ts_ms);
        self
    }

    pub fn inc(&mut self) {
        self.value += 1.0;
    }

    pub fn inc_by(&mut self, v: f64) {
        if v >= 0.0 {
            self.value += v;
        }
    }

    pub fn get(&self) -> f64 {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    pub fn to_family(&self) -> MetricFamily {
        let mut family =
            MetricFamily::new(&format!("{}_total", self.name), MetricType::Counter);
        family.help = self.help.clone();
        let mut sample =
            MetricSample::new(&format!("{}_total", self.name), self.value)
                .with_labels(self.labels.clone());
        sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(sample);
        family
    }
}

// ── Gauge ───────────────────────────────────────────────────

/// A gauge that can go up and down.
#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    help: Option<String>,
    labels: Vec<LabelPair>,
    value: f64,
    timestamp_ms: Option<i64>,
}

impl Gauge {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            help: None,
            labels: Vec::new(),
            value: 0.0,
            timestamp_ms: None,
        }
    }

    pub fn with_help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    pub fn with_label(mut self, name: &str, value: &str) -> Self {
        self.labels.push(LabelPair::new(name, value));
        self
    }

    pub fn with_timestamp(mut self, ts_ms: i64) -> Self {
        self.timestamp_ms = Some(ts_ms);
        self
    }

    pub fn set(&mut self, v: f64) {
        self.value = v;
    }

    pub fn inc(&mut self) {
        self.value += 1.0;
    }

    pub fn dec(&mut self) {
        self.value -= 1.0;
    }

    pub fn add(&mut self, v: f64) {
        self.value += v;
    }

    pub fn sub(&mut self, v: f64) {
        self.value -= v;
    }

    pub fn get(&self) -> f64 {
        self.value
    }

    pub fn to_family(&self) -> MetricFamily {
        let mut family = MetricFamily::new(&self.name, MetricType::Gauge);
        family.help = self.help.clone();
        let mut sample =
            MetricSample::new(&self.name, self.value).with_labels(self.labels.clone());
        sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(sample);
        family
    }
}

// ── Histogram Metric ────────────────────────────────────────

/// A Prometheus histogram with configurable buckets.
#[derive(Debug, Clone)]
pub struct HistogramMetric {
    name: String,
    help: Option<String>,
    labels: Vec<LabelPair>,
    /// Upper bounds for each bucket.
    buckets: Vec<f64>,
    /// Count per bucket.
    bucket_counts: Vec<u64>,
    /// Sum of all observed values.
    sum: f64,
    /// Total observations.
    count: u64,
    timestamp_ms: Option<i64>,
}

impl HistogramMetric {
    /// Create with the given upper-bound buckets. `+Inf` is added automatically.
    pub fn new(name: &str, buckets: &[f64]) -> Self {
        let mut sorted_buckets: Vec<f64> = buckets.to_vec();
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_buckets.dedup();
        let len = sorted_buckets.len();
        Self {
            name: name.to_string(),
            help: None,
            labels: Vec::new(),
            buckets: sorted_buckets,
            bucket_counts: vec![0; len],
            sum: 0.0,
            count: 0,
            timestamp_ms: None,
        }
    }

    /// Default Prometheus buckets.
    pub fn with_default_buckets(name: &str) -> Self {
        Self::new(
            name,
            &[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
    }

    pub fn with_help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    pub fn with_label(mut self, name: &str, value: &str) -> Self {
        self.labels.push(LabelPair::new(name, value));
        self
    }

    pub fn with_timestamp(mut self, ts_ms: i64) -> Self {
        self.timestamp_ms = Some(ts_ms);
        self
    }

    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        for (i, bound) in self.buckets.iter().enumerate() {
            if value <= *bound {
                self.bucket_counts[i] += 1;
            }
        }
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn to_family(&self) -> MetricFamily {
        let mut family = MetricFamily::new(&self.name, MetricType::Histogram);
        family.help = self.help.clone();

        // Bucket counts (already cumulative from observe())
        for (i, bound) in self.buckets.iter().enumerate() {
            let mut labels = self.labels.clone();
            labels.push(LabelPair::new("le", &format_value(*bound)));
            let mut sample = MetricSample::new(&format!("{}_bucket", self.name), self.bucket_counts[i] as f64)
                .with_labels(labels);
            sample.timestamp_ms = self.timestamp_ms;
            family.add_sample(sample);
        }
        // +Inf bucket
        let mut inf_labels = self.labels.clone();
        inf_labels.push(LabelPair::new("le", "+Inf"));
        let mut inf_sample = MetricSample::new(
            &format!("{}_bucket", self.name),
            self.count as f64,
        )
        .with_labels(inf_labels);
        inf_sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(inf_sample);

        // _sum
        let mut sum_sample =
            MetricSample::new(&format!("{}_sum", self.name), self.sum)
                .with_labels(self.labels.clone());
        sum_sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(sum_sample);

        // _count
        let mut count_sample =
            MetricSample::new(&format!("{}_count", self.name), self.count as f64)
                .with_labels(self.labels.clone());
        count_sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(count_sample);

        family
    }
}

// ── Summary Metric ──────────────────────────────────────────

/// A Prometheus summary metric with pre-computed quantiles.
#[derive(Debug, Clone)]
pub struct SummaryMetric {
    name: String,
    help: Option<String>,
    labels: Vec<LabelPair>,
    /// Quantile-value pairs (e.g., (0.5, 120.0) means median is 120).
    quantiles: Vec<(f64, f64)>,
    sum: f64,
    count: u64,
    timestamp_ms: Option<i64>,
}

impl SummaryMetric {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            help: None,
            labels: Vec::new(),
            quantiles: Vec::new(),
            sum: 0.0,
            count: 0,
            timestamp_ms: None,
        }
    }

    pub fn with_help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    pub fn with_label(mut self, name: &str, value: &str) -> Self {
        self.labels.push(LabelPair::new(name, value));
        self
    }

    pub fn with_timestamp(mut self, ts_ms: i64) -> Self {
        self.timestamp_ms = Some(ts_ms);
        self
    }

    pub fn set_quantile(&mut self, quantile: f64, value: f64) {
        self.quantiles.push((quantile, value));
    }

    pub fn set_sum(&mut self, sum: f64) {
        self.sum = sum;
    }

    pub fn set_count(&mut self, count: u64) {
        self.count = count;
    }

    pub fn to_family(&self) -> MetricFamily {
        let mut family = MetricFamily::new(&self.name, MetricType::Summary);
        family.help = self.help.clone();

        for (q, v) in &self.quantiles {
            let mut labels = self.labels.clone();
            labels.push(LabelPair::new("quantile", &format!("{}", q)));
            let mut sample =
                MetricSample::new(&self.name, *v).with_labels(labels);
            sample.timestamp_ms = self.timestamp_ms;
            family.add_sample(sample);
        }

        let mut sum_sample =
            MetricSample::new(&format!("{}_sum", self.name), self.sum)
                .with_labels(self.labels.clone());
        sum_sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(sum_sample);

        let mut count_sample =
            MetricSample::new(&format!("{}_count", self.name), self.count as f64)
                .with_labels(self.labels.clone());
        count_sample.timestamp_ms = self.timestamp_ms;
        family.add_sample(count_sample);

        family
    }
}

// ── Metric Registry ─────────────────────────────────────────

/// Registry that holds metric families and renders them as exposition text.
#[derive(Debug, Clone)]
pub struct PrometheusRegistry {
    families: Vec<MetricFamily>,
}

impl PrometheusRegistry {
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
        }
    }

    pub fn register(&mut self, family: MetricFamily) {
        self.families.push(family);
    }

    pub fn register_counter(&mut self, counter: &Counter) {
        self.families.push(counter.to_family());
    }

    pub fn register_gauge(&mut self, gauge: &Gauge) {
        self.families.push(gauge.to_family());
    }

    pub fn register_histogram(&mut self, histogram: &HistogramMetric) {
        self.families.push(histogram.to_family());
    }

    pub fn register_summary(&mut self, summary: &SummaryMetric) {
        self.families.push(summary.to_family());
    }

    /// Remove all families with the given name.
    pub fn unregister(&mut self, name: &str) {
        self.families.retain(|f| f.name != name);
    }

    /// Get a family by name.
    pub fn get(&self, name: &str) -> Option<&MetricFamily> {
        self.families.iter().find(|f| f.name == name)
    }

    /// Number of registered families.
    pub fn len(&self) -> usize {
        self.families.len()
    }

    pub fn is_empty(&self) -> bool {
        self.families.is_empty()
    }

    /// Render all families in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (i, family) in self.families.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&family.format());
        }
        out
    }

    /// Content-Type header for Prometheus text format.
    pub fn content_type() -> &'static str {
        "text/plain; version=0.0.4; charset=utf-8"
    }

    /// Return an iterator over all families.
    pub fn families(&self) -> &[MetricFamily] {
        &self.families
    }

    /// Merge another registry into this one.
    pub fn merge(&mut self, other: &PrometheusRegistry) {
        for family in &other.families {
            self.families.push(family.clone());
        }
    }
}

impl Default for PrometheusRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Validate metric name ────────────────────────────────────

/// Check that a metric name matches [a-zA-Z_:][a-zA-Z0-9_:]*.
pub fn is_valid_metric_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_' || first == ':') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

/// Check that a label name matches [a-zA-Z_][a-zA-Z0-9_]*.
pub fn is_valid_label_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_pair_format() {
        let lp = LabelPair::new("method", "GET");
        assert_eq!(lp.format(), "method=\"GET\"");
    }

    #[test]
    fn test_label_pair_escaping() {
        let lp = LabelPair::new("path", "/a\"b\\c\nd");
        assert_eq!(lp.format(), "path=\"/a\\\"b\\\\c\\nd\"");
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::Counter.to_string(), "counter");
        assert_eq!(MetricType::Gauge.to_string(), "gauge");
        assert_eq!(MetricType::Histogram.to_string(), "histogram");
        assert_eq!(MetricType::Summary.to_string(), "summary");
        assert_eq!(MetricType::Untyped.to_string(), "untyped");
    }

    #[test]
    fn test_sample_no_labels() {
        let s = MetricSample::new("my_counter", 42.0);
        assert_eq!(s.format_line(), "my_counter 42");
    }

    #[test]
    fn test_sample_with_labels() {
        let s = MetricSample::new("http_requests_total", 1027.0)
            .with_label("method", "GET")
            .with_label("code", "200");
        assert_eq!(
            s.format_line(),
            "http_requests_total{method=\"GET\",code=\"200\"} 1027"
        );
    }

    #[test]
    fn test_sample_with_timestamp() {
        let s = MetricSample::new("cpu_temp", 65.3).with_timestamp(1395066363000);
        assert_eq!(s.format_line(), "cpu_temp 65.3 1395066363000");
    }

    #[test]
    fn test_format_value_special() {
        assert_eq!(format_value(f64::INFINITY), "+Inf");
        assert_eq!(format_value(f64::NEG_INFINITY), "-Inf");
        assert_eq!(format_value(f64::NAN), "NaN");
        assert_eq!(format_value(0.0), "0");
        assert_eq!(format_value(1.5), "1.5");
    }

    #[test]
    fn test_counter_basic() {
        let mut c = Counter::new("requests");
        assert_eq!(c.get(), 0.0);
        c.inc();
        assert_eq!(c.get(), 1.0);
        c.inc_by(4.5);
        assert_eq!(c.get(), 5.5);
        c.inc_by(-1.0); // negative ignored
        assert_eq!(c.get(), 5.5);
    }

    #[test]
    fn test_counter_reset() {
        let mut c = Counter::new("x");
        c.inc_by(100.0);
        c.reset();
        assert_eq!(c.get(), 0.0);
    }

    #[test]
    fn test_counter_to_family() {
        let mut c = Counter::new("http_requests")
            .with_help("Total HTTP requests")
            .with_label("method", "POST");
        c.inc_by(5.0);
        let family = c.to_family();
        assert_eq!(family.metric_type, MetricType::Counter);
        let text = family.format();
        assert!(text.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(text.contains("# TYPE http_requests_total counter"));
        assert!(text.contains("http_requests_total{method=\"POST\"} 5"));
    }

    #[test]
    fn test_gauge_operations() {
        let mut g = Gauge::new("temperature");
        g.set(20.0);
        assert_eq!(g.get(), 20.0);
        g.inc();
        assert_eq!(g.get(), 21.0);
        g.dec();
        assert_eq!(g.get(), 20.0);
        g.add(5.0);
        assert_eq!(g.get(), 25.0);
        g.sub(3.0);
        assert_eq!(g.get(), 22.0);
    }

    #[test]
    fn test_gauge_to_family() {
        let mut g = Gauge::new("cpu_usage").with_help("CPU usage percentage");
        g.set(72.5);
        let text = g.to_family().format();
        assert!(text.contains("# TYPE cpu_usage gauge"));
        assert!(text.contains("cpu_usage 72.5"));
    }

    #[test]
    fn test_histogram_observe() {
        let mut h = HistogramMetric::new("request_duration", &[0.1, 0.5, 1.0]);
        h.observe(0.05);
        h.observe(0.3);
        h.observe(0.8);
        h.observe(2.0);
        assert_eq!(h.count(), 4);
        let expected_sum = 0.05 + 0.3 + 0.8 + 2.0;
        assert!((h.sum() - expected_sum).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_to_family() {
        let mut h = HistogramMetric::new("latency", &[0.1, 0.5])
            .with_help("Request latency");
        h.observe(0.05);
        h.observe(0.3);
        let text = h.to_family().format();
        assert!(text.contains("# TYPE latency histogram"));
        assert!(text.contains("latency_bucket{le=\"0.1\"} 1"));
        assert!(text.contains("latency_bucket{le=\"0.5\"} 2"));
        assert!(text.contains("latency_bucket{le=\"+Inf\"} 2"));
        assert!(text.contains("latency_sum 0.35"));
        assert!(text.contains("latency_count 2"));
    }

    #[test]
    fn test_histogram_default_buckets() {
        let h = HistogramMetric::with_default_buckets("dur");
        let family = h.to_family();
        // 11 default buckets + 1 +Inf + _sum + _count = 14 samples
        assert_eq!(family.samples.len(), 14);
    }

    #[test]
    fn test_summary_metric() {
        let mut s = SummaryMetric::new("rpc_duration")
            .with_help("RPC duration summary");
        s.set_quantile(0.5, 120.0);
        s.set_quantile(0.9, 250.0);
        s.set_quantile(0.99, 480.0);
        s.set_sum(5000.0);
        s.set_count(100);
        let text = s.to_family().format();
        assert!(text.contains("# TYPE rpc_duration summary"));
        assert!(text.contains("rpc_duration{quantile=\"0.5\"} 120"));
        assert!(text.contains("rpc_duration{quantile=\"0.99\"} 480"));
        assert!(text.contains("rpc_duration_sum 5000"));
        assert!(text.contains("rpc_duration_count 100"));
    }

    #[test]
    fn test_registry_render() {
        let mut reg = PrometheusRegistry::new();

        let mut c = Counter::new("http_requests");
        c.inc_by(10.0);
        reg.register_counter(&c);

        let mut g = Gauge::new("temp");
        g.set(22.5);
        reg.register_gauge(&g);

        let text = reg.render();
        assert!(text.contains("http_requests_total"));
        assert!(text.contains("temp 22.5"));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = PrometheusRegistry::new();
        reg.register(MetricFamily::new("a", MetricType::Counter));
        reg.register(MetricFamily::new("b", MetricType::Gauge));
        assert_eq!(reg.len(), 2);
        reg.unregister("a");
        assert_eq!(reg.len(), 1);
        assert!(reg.get("b").is_some());
    }

    #[test]
    fn test_registry_merge() {
        let mut r1 = PrometheusRegistry::new();
        r1.register(MetricFamily::new("a", MetricType::Counter));
        let mut r2 = PrometheusRegistry::new();
        r2.register(MetricFamily::new("b", MetricType::Gauge));
        r1.merge(&r2);
        assert_eq!(r1.len(), 2);
    }

    #[test]
    fn test_registry_content_type() {
        assert!(PrometheusRegistry::content_type().contains("text/plain"));
        assert!(PrometheusRegistry::content_type().contains("0.0.4"));
    }

    #[test]
    fn test_valid_metric_name() {
        assert!(is_valid_metric_name("http_requests_total"));
        assert!(is_valid_metric_name("_private_metric"));
        assert!(is_valid_metric_name("my:scoped:metric"));
        assert!(!is_valid_metric_name(""));
        assert!(!is_valid_metric_name("1bad"));
        assert!(!is_valid_metric_name("has space"));
    }

    #[test]
    fn test_valid_label_name() {
        assert!(is_valid_label_name("method"));
        assert!(is_valid_label_name("_internal"));
        assert!(!is_valid_label_name(""));
        assert!(!is_valid_label_name("1abc"));
        assert!(!is_valid_label_name("has:colon"));
    }

    #[test]
    fn test_metric_family_format() {
        let mut family = MetricFamily::new("up", MetricType::Gauge).with_help("Server up");
        family.add_sample(MetricSample::new("up", 1.0));
        let text = family.format();
        assert!(text.contains("# HELP up Server up\n"));
        assert!(text.contains("# TYPE up gauge\n"));
        assert!(text.contains("up 1\n"));
    }

    #[test]
    fn test_counter_with_timestamp() {
        let mut c = Counter::new("events").with_timestamp(1700000000000);
        c.inc_by(42.0);
        let text = c.to_family().format();
        assert!(text.contains("1700000000000"));
    }

    #[test]
    fn test_histogram_with_labels() {
        let mut h = HistogramMetric::new("dur", &[1.0])
            .with_label("service", "web");
        h.observe(0.5);
        let text = h.to_family().format();
        assert!(text.contains("service=\"web\""));
    }

    #[test]
    fn test_empty_registry() {
        let reg = PrometheusRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.render(), "");
    }
}
