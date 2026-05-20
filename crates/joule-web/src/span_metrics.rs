//! Span-based metrics — request duration tracking, span attributes, links,
//! status, baggage propagation, metric extraction from spans, SLA compliance
//! checking, and latency distribution.
//!
//! Replaces OpenTelemetry span processors and metric extractors with a
//! pure-Rust span model that tracks durations, attributes, parent/child
//! relationships, and computes SLA compliance from collected spans.

use std::collections::HashMap;
use std::fmt;

// ── Span Status ─────────────────────────────────────────────

/// Status of a completed span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    /// Operation completed without error.
    Ok,
    /// Operation had an error.
    Error,
    /// Status not set.
    Unset,
}

impl SpanStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpanStatus::Ok => "OK",
            SpanStatus::Error => "ERROR",
            SpanStatus::Unset => "UNSET",
        }
    }

    pub fn is_error(&self) -> bool {
        matches!(self, SpanStatus::Error)
    }
}

impl fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Span Link ───────────────────────────────────────────────

/// A link to another span (e.g., a triggering span from a different trace).
#[derive(Debug, Clone, PartialEq)]
pub struct SpanLink {
    pub trace_id: String,
    pub span_id: String,
    pub attributes: HashMap<String, String>,
}

impl SpanLink {
    pub fn new(trace_id: &str, span_id: &str) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            span_id: span_id.to_string(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }
}

// ── Baggage ─────────────────────────────────────────────────

/// Baggage: key-value pairs propagated across span boundaries.
#[derive(Debug, Clone, Default)]
pub struct Baggage {
    items: HashMap<String, String>,
}

impl Baggage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.items.insert(key.to_string(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.items.get(key).map(|s| s.as_str())
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.items.remove(key)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn items(&self) -> &HashMap<String, String> {
        &self.items
    }

    /// Merge another baggage into this one (other takes precedence).
    pub fn merge(&mut self, other: &Baggage) {
        for (k, v) in &other.items {
            self.items.insert(k.clone(), v.clone());
        }
    }
}

// ── Span ────────────────────────────────────────────────────

/// A recorded span representing an operation.
#[derive(Debug, Clone)]
pub struct Span {
    /// Unique span identifier.
    pub span_id: String,
    /// Trace identifier.
    pub trace_id: String,
    /// Parent span identifier (if any).
    pub parent_id: Option<String>,
    /// Operation name.
    pub operation: String,
    /// Service name.
    pub service: String,
    /// Start time in microseconds since epoch.
    pub start_us: u64,
    /// End time in microseconds since epoch (0 if not finished).
    pub end_us: u64,
    /// Status.
    pub status: SpanStatus,
    /// Attributes (key-value pairs).
    pub attributes: HashMap<String, String>,
    /// Links to other spans.
    pub links: Vec<SpanLink>,
    /// Baggage.
    pub baggage: Baggage,
    /// Error message (if status == Error).
    pub error_message: Option<String>,
}

impl Span {
    /// Create a new span.
    pub fn new(span_id: &str, trace_id: &str, operation: &str, service: &str, start_us: u64) -> Self {
        Self {
            span_id: span_id.to_string(),
            trace_id: trace_id.to_string(),
            parent_id: None,
            operation: operation.to_string(),
            service: service.to_string(),
            start_us,
            end_us: 0,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            links: Vec::new(),
            baggage: Baggage::new(),
            error_message: None,
        }
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_link(mut self, link: SpanLink) -> Self {
        self.links.push(link);
        self
    }

    pub fn with_baggage_item(mut self, key: &str, value: &str) -> Self {
        self.baggage.set(key, value);
        self
    }

    /// Finish the span with given end time and status.
    pub fn finish(&mut self, end_us: u64, status: SpanStatus) {
        self.end_us = end_us;
        self.status = status;
    }

    /// Finish with error.
    pub fn finish_error(&mut self, end_us: u64, message: &str) {
        self.end_us = end_us;
        self.status = SpanStatus::Error;
        self.error_message = Some(message.to_string());
    }

    /// Duration in microseconds (0 if not finished).
    pub fn duration_us(&self) -> u64 {
        if self.end_us > self.start_us {
            self.end_us - self.start_us
        } else {
            0
        }
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.duration_us() as f64 / 1000.0
    }

    /// Whether this span has finished.
    pub fn is_finished(&self) -> bool {
        self.end_us > 0
    }

    /// Whether this is a root span (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }
}

// ── Span Metrics Extractor ──────────────────────────────────

/// Extracted metrics from a collection of spans.
#[derive(Debug, Clone)]
pub struct SpanMetrics {
    /// Total spans processed.
    pub total_spans: usize,
    /// Spans with error status.
    pub error_spans: usize,
    /// Per-operation: count.
    pub operation_counts: HashMap<String, u64>,
    /// Per-operation: total duration in microseconds.
    pub operation_durations_us: HashMap<String, u64>,
    /// Per-service: count.
    pub service_counts: HashMap<String, u64>,
    /// All durations in microseconds (for distribution).
    pub durations_us: Vec<u64>,
}

impl SpanMetrics {
    /// Extract metrics from a slice of spans.
    pub fn from_spans(spans: &[Span]) -> Self {
        let mut operation_counts: HashMap<String, u64> = HashMap::new();
        let mut operation_durations_us: HashMap<String, u64> = HashMap::new();
        let mut service_counts: HashMap<String, u64> = HashMap::new();
        let mut durations_us = Vec::new();
        let mut error_spans = 0;

        for span in spans {
            if !span.is_finished() {
                continue;
            }
            *operation_counts.entry(span.operation.clone()).or_insert(0) += 1;
            *operation_durations_us
                .entry(span.operation.clone())
                .or_insert(0) += span.duration_us();
            *service_counts.entry(span.service.clone()).or_insert(0) += 1;
            durations_us.push(span.duration_us());
            if span.status.is_error() {
                error_spans += 1;
            }
        }

        let total_spans = spans.iter().filter(|s| s.is_finished()).count();

        Self {
            total_spans,
            error_spans,
            operation_counts,
            operation_durations_us,
            service_counts,
            durations_us,
        }
    }

    /// Error rate (0.0 to 1.0).
    pub fn error_rate(&self) -> f64 {
        if self.total_spans == 0 {
            return 0.0;
        }
        self.error_spans as f64 / self.total_spans as f64
    }

    /// Average duration across all spans in microseconds.
    pub fn avg_duration_us(&self) -> f64 {
        if self.durations_us.is_empty() {
            return 0.0;
        }
        self.durations_us.iter().sum::<u64>() as f64 / self.durations_us.len() as f64
    }

    /// Percentile of durations in microseconds.
    pub fn percentile_us(&self, pct: f64) -> u64 {
        if self.durations_us.is_empty() {
            return 0;
        }
        let mut sorted = self.durations_us.clone();
        sorted.sort();
        let rank = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        let rank = rank.min(sorted.len() - 1);
        sorted[rank]
    }

    /// Average duration for a specific operation (microseconds).
    pub fn avg_operation_duration_us(&self, operation: &str) -> f64 {
        let count = self.operation_counts.get(operation).copied().unwrap_or(0);
        let dur = self
            .operation_durations_us
            .get(operation)
            .copied()
            .unwrap_or(0);
        if count == 0 {
            0.0
        } else {
            dur as f64 / count as f64
        }
    }
}

// ── SLA Compliance ──────────────────────────────────────────

/// SLA definition for a service or operation.
#[derive(Debug, Clone)]
pub struct SlaDefinition {
    /// Name of the SLA.
    pub name: String,
    /// Target latency in microseconds (e.g., p99 < 500ms = 500_000us).
    pub target_latency_us: u64,
    /// Percentile at which latency is measured (e.g., 99.0).
    pub latency_percentile: f64,
    /// Target error rate (e.g., 0.001 = 0.1%).
    pub target_error_rate: f64,
    /// Target availability (e.g., 0.999 = 99.9%).
    pub target_availability: f64,
}

impl SlaDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            target_latency_us: 500_000, // 500ms
            latency_percentile: 99.0,
            target_error_rate: 0.001,
            target_availability: 0.999,
        }
    }

    pub fn with_latency(mut self, target_us: u64, percentile: f64) -> Self {
        self.target_latency_us = target_us;
        self.latency_percentile = percentile;
        self
    }

    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.target_error_rate = rate;
        self
    }

    pub fn with_availability(mut self, avail: f64) -> Self {
        self.target_availability = avail;
        self
    }
}

/// Result of an SLA compliance check.
#[derive(Debug, Clone)]
pub struct SlaResult {
    pub sla_name: String,
    /// Whether latency is within SLA.
    pub latency_compliant: bool,
    /// Actual latency at the target percentile.
    pub actual_latency_us: u64,
    /// Whether error rate is within SLA.
    pub error_rate_compliant: bool,
    /// Actual error rate.
    pub actual_error_rate: f64,
    /// Whether availability is within SLA.
    pub availability_compliant: bool,
    /// Actual availability.
    pub actual_availability: f64,
    /// Overall pass/fail.
    pub overall_compliant: bool,
}

/// Check SLA compliance from span metrics.
pub fn check_sla(metrics: &SpanMetrics, sla: &SlaDefinition) -> SlaResult {
    let actual_latency = metrics.percentile_us(sla.latency_percentile);
    let actual_error_rate = metrics.error_rate();
    let actual_availability = 1.0 - actual_error_rate;

    let latency_ok = actual_latency <= sla.target_latency_us;
    let error_ok = actual_error_rate <= sla.target_error_rate;
    let avail_ok = actual_availability >= sla.target_availability;

    SlaResult {
        sla_name: sla.name.clone(),
        latency_compliant: latency_ok,
        actual_latency_us: actual_latency,
        error_rate_compliant: error_ok,
        actual_error_rate,
        availability_compliant: avail_ok,
        actual_availability,
        overall_compliant: latency_ok && error_ok && avail_ok,
    }
}

// ── Latency Distribution ────────────────────────────────────

/// Latency distribution buckets.
#[derive(Debug, Clone)]
pub struct LatencyDistribution {
    /// Bucket upper bounds in microseconds.
    pub bounds_us: Vec<u64>,
    /// Count per bucket.
    pub counts: Vec<u64>,
    /// Overflow count (values above all bounds).
    pub overflow: u64,
    /// Total count.
    pub total: u64,
}

impl LatencyDistribution {
    /// Create with given upper bounds in microseconds.
    pub fn new(bounds_us: &[u64]) -> Self {
        let mut sorted = bounds_us.to_vec();
        sorted.sort();
        sorted.dedup();
        let len = sorted.len();
        Self {
            bounds_us: sorted,
            counts: vec![0; len],
            overflow: 0,
            total: 0,
        }
    }

    /// Default latency buckets (in microseconds).
    pub fn default_buckets() -> Self {
        Self::new(&[
            1_000,     // 1ms
            5_000,     // 5ms
            10_000,    // 10ms
            25_000,    // 25ms
            50_000,    // 50ms
            100_000,   // 100ms
            250_000,   // 250ms
            500_000,   // 500ms
            1_000_000, // 1s
            5_000_000, // 5s
        ])
    }

    /// Record a latency value.
    pub fn record(&mut self, duration_us: u64) {
        self.total += 1;
        let mut placed = false;
        for (i, bound) in self.bounds_us.iter().enumerate() {
            if duration_us <= *bound {
                self.counts[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            self.overflow += 1;
        }
    }

    /// Build from a slice of span durations.
    pub fn from_spans(spans: &[Span]) -> Self {
        let mut dist = Self::default_buckets();
        for span in spans {
            if span.is_finished() {
                dist.record(span.duration_us());
            }
        }
        dist
    }

    /// Fraction of requests under a given latency.
    pub fn fraction_under(&self, threshold_us: u64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut count = 0u64;
        for (i, bound) in self.bounds_us.iter().enumerate() {
            if *bound <= threshold_us {
                count += self.counts[i];
            }
        }
        count as f64 / self.total as f64
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span(id: &str, op: &str, svc: &str, start: u64, end: u64, status: SpanStatus) -> Span {
        let mut s = Span::new(id, "trace-1", op, svc, start);
        s.finish(end, status);
        s
    }

    #[test]
    fn test_span_creation() {
        let s = Span::new("s1", "t1", "GET /api", "web", 1000);
        assert_eq!(s.span_id, "s1");
        assert_eq!(s.operation, "GET /api");
        assert!(!s.is_finished());
        assert!(s.is_root());
    }

    #[test]
    fn test_span_with_parent() {
        let s = Span::new("s2", "t1", "db_query", "db", 2000).with_parent("s1");
        assert!(!s.is_root());
        assert_eq!(s.parent_id.as_deref(), Some("s1"));
    }

    #[test]
    fn test_span_finish() {
        let mut s = Span::new("s1", "t1", "op", "svc", 1000);
        s.finish(2000, SpanStatus::Ok);
        assert!(s.is_finished());
        assert_eq!(s.duration_us(), 1000);
        assert!((s.duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_span_finish_error() {
        let mut s = Span::new("s1", "t1", "op", "svc", 1000);
        s.finish_error(3000, "timeout");
        assert_eq!(s.status, SpanStatus::Error);
        assert_eq!(s.error_message.as_deref(), Some("timeout"));
        assert_eq!(s.duration_us(), 2000);
    }

    #[test]
    fn test_span_status() {
        assert_eq!(SpanStatus::Ok.as_str(), "OK");
        assert_eq!(SpanStatus::Error.as_str(), "ERROR");
        assert!(!SpanStatus::Ok.is_error());
        assert!(SpanStatus::Error.is_error());
        assert_eq!(format!("{}", SpanStatus::Unset), "UNSET");
    }

    #[test]
    fn test_span_link() {
        let link = SpanLink::new("trace-2", "span-5").with_attribute("reason", "retried");
        assert_eq!(link.trace_id, "trace-2");
        assert_eq!(link.attributes.get("reason").unwrap(), "retried");
    }

    #[test]
    fn test_span_with_link() {
        let link = SpanLink::new("t2", "s5");
        let s = Span::new("s1", "t1", "op", "svc", 0).with_link(link);
        assert_eq!(s.links.len(), 1);
    }

    #[test]
    fn test_baggage() {
        let mut b = Baggage::new();
        b.set("user_id", "abc");
        assert_eq!(b.get("user_id"), Some("abc"));
        assert_eq!(b.len(), 1);
        b.remove("user_id");
        assert!(b.is_empty());
    }

    #[test]
    fn test_baggage_merge() {
        let mut b1 = Baggage::new();
        b1.set("a", "1");
        b1.set("b", "2");
        let mut b2 = Baggage::new();
        b2.set("b", "override");
        b2.set("c", "3");
        b1.merge(&b2);
        assert_eq!(b1.get("a"), Some("1"));
        assert_eq!(b1.get("b"), Some("override"));
        assert_eq!(b1.get("c"), Some("3"));
    }

    #[test]
    fn test_span_metrics_extraction() {
        let spans = vec![
            make_span("s1", "GET", "web", 0, 1000, SpanStatus::Ok),
            make_span("s2", "GET", "web", 0, 2000, SpanStatus::Ok),
            make_span("s3", "POST", "web", 0, 3000, SpanStatus::Error),
            make_span("s4", "query", "db", 0, 500, SpanStatus::Ok),
        ];
        let m = SpanMetrics::from_spans(&spans);
        assert_eq!(m.total_spans, 4);
        assert_eq!(m.error_spans, 1);
        assert_eq!(m.operation_counts["GET"], 2);
        assert_eq!(m.operation_counts["POST"], 1);
        assert_eq!(m.service_counts["web"], 3);
        assert_eq!(m.service_counts["db"], 1);
    }

    #[test]
    fn test_span_metrics_error_rate() {
        let spans = vec![
            make_span("s1", "op", "svc", 0, 100, SpanStatus::Ok),
            make_span("s2", "op", "svc", 0, 100, SpanStatus::Error),
        ];
        let m = SpanMetrics::from_spans(&spans);
        assert!((m.error_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_span_metrics_avg_duration() {
        let spans = vec![
            make_span("s1", "op", "svc", 0, 1000, SpanStatus::Ok),
            make_span("s2", "op", "svc", 0, 3000, SpanStatus::Ok),
        ];
        let m = SpanMetrics::from_spans(&spans);
        assert!((m.avg_duration_us() - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_span_metrics_percentile() {
        let spans: Vec<Span> = (1..=100)
            .map(|i| make_span(&format!("s{}", i), "op", "svc", 0, i * 100, SpanStatus::Ok))
            .collect();
        let m = SpanMetrics::from_spans(&spans);
        let p50 = m.percentile_us(50.0);
        assert!(p50 >= 4000 && p50 <= 6000, "p50={}", p50);
    }

    #[test]
    fn test_span_metrics_avg_operation() {
        let spans = vec![
            make_span("s1", "GET", "web", 0, 1000, SpanStatus::Ok),
            make_span("s2", "GET", "web", 0, 3000, SpanStatus::Ok),
        ];
        let m = SpanMetrics::from_spans(&spans);
        assert!((m.avg_operation_duration_us("GET") - 2000.0).abs() < 0.01);
        assert_eq!(m.avg_operation_duration_us("POST"), 0.0);
    }

    #[test]
    fn test_sla_compliant() {
        let spans: Vec<Span> = (0..100)
            .map(|i| make_span(&format!("s{}", i), "op", "svc", 0, 100, SpanStatus::Ok))
            .collect();
        let m = SpanMetrics::from_spans(&spans);
        let sla = SlaDefinition::new("test")
            .with_latency(1000, 99.0)
            .with_error_rate(0.01)
            .with_availability(0.99);
        let result = check_sla(&m, &sla);
        assert!(result.overall_compliant);
        assert!(result.latency_compliant);
        assert!(result.error_rate_compliant);
    }

    #[test]
    fn test_sla_latency_violation() {
        let spans: Vec<Span> = (0..100)
            .map(|i| make_span(&format!("s{}", i), "op", "svc", 0, 1_000_000, SpanStatus::Ok))
            .collect();
        let m = SpanMetrics::from_spans(&spans);
        let sla = SlaDefinition::new("test").with_latency(500_000, 99.0);
        let result = check_sla(&m, &sla);
        assert!(!result.latency_compliant);
    }

    #[test]
    fn test_sla_error_rate_violation() {
        let mut spans = Vec::new();
        for i in 0..90 {
            spans.push(make_span(&format!("s{}", i), "op", "svc", 0, 100, SpanStatus::Ok));
        }
        for i in 90..100 {
            spans.push(make_span(&format!("s{}", i), "op", "svc", 0, 100, SpanStatus::Error));
        }
        let m = SpanMetrics::from_spans(&spans);
        let sla = SlaDefinition::new("test").with_error_rate(0.05);
        let result = check_sla(&m, &sla);
        assert!(!result.error_rate_compliant);
    }

    #[test]
    fn test_latency_distribution() {
        let mut dist = LatencyDistribution::new(&[100, 500, 1000]);
        dist.record(50);
        dist.record(200);
        dist.record(800);
        dist.record(2000);
        assert_eq!(dist.total, 4);
        assert_eq!(dist.counts[0], 1); // <=100
        assert_eq!(dist.counts[1], 1); // <=500
        assert_eq!(dist.counts[2], 1); // <=1000
        assert_eq!(dist.overflow, 1);
    }

    #[test]
    fn test_latency_distribution_from_spans() {
        let spans = vec![
            make_span("s1", "op", "svc", 0, 500, SpanStatus::Ok),
            make_span("s2", "op", "svc", 0, 5000, SpanStatus::Ok),
            make_span("s3", "op", "svc", 0, 50000, SpanStatus::Ok),
        ];
        let dist = LatencyDistribution::from_spans(&spans);
        assert_eq!(dist.total, 3);
    }

    #[test]
    fn test_latency_fraction_under() {
        let mut dist = LatencyDistribution::new(&[100, 500, 1000]);
        for _ in 0..80 {
            dist.record(50);
        }
        for _ in 0..20 {
            dist.record(800);
        }
        let frac = dist.fraction_under(500);
        assert!((frac - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_span_attributes() {
        let s = Span::new("s1", "t1", "op", "svc", 0)
            .with_attribute("http.method", "GET")
            .with_attribute("http.status_code", "200");
        assert_eq!(s.attributes.get("http.method").unwrap(), "GET");
        assert_eq!(s.attributes.len(), 2);
    }

    #[test]
    fn test_span_baggage_propagation() {
        let s = Span::new("s1", "t1", "op", "svc", 0)
            .with_baggage_item("tenant", "acme");
        assert_eq!(s.baggage.get("tenant"), Some("acme"));
    }

    #[test]
    fn test_unfinished_spans_excluded() {
        let spans = vec![
            Span::new("s1", "t1", "op", "svc", 0), // not finished
            make_span("s2", "op", "svc", 0, 100, SpanStatus::Ok),
        ];
        let m = SpanMetrics::from_spans(&spans);
        assert_eq!(m.total_spans, 1);
    }
}
