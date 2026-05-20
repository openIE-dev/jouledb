//! Metric export formats — JSON, OpenMetrics text, StatsD, Graphite line protocol,
//! OTLP-like format, batch export, metric filtering, and configurable export intervals.
//!
//! Replaces metric export libraries (`statsd-client`, `graphite`, `opentelemetry-stdout`)
//! with a pure-Rust multi-format metric exporter supporting JSON, OpenMetrics,
//! StatsD, Graphite, and OTLP-like serialization.

use std::collections::HashMap;
use std::fmt;

// ── Metric Point ────────────────────────────────────────────

/// A single metric data point for export.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricPoint {
    /// Metric name.
    pub name: String,
    /// Value.
    pub value: f64,
    /// Metric kind (counter, gauge, histogram, etc.).
    pub kind: MetricKind,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Labels/tags.
    pub labels: HashMap<String, String>,
    /// Unit (optional).
    pub unit: Option<String>,
    /// Description (optional).
    pub description: Option<String>,
}

impl MetricPoint {
    pub fn counter(name: &str, value: f64, timestamp_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            value,
            kind: MetricKind::Counter,
            timestamp_ms,
            labels: HashMap::new(),
            unit: None,
            description: None,
        }
    }

    pub fn gauge(name: &str, value: f64, timestamp_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            value,
            kind: MetricKind::Gauge,
            timestamp_ms,
            labels: HashMap::new(),
            unit: None,
            description: None,
        }
    }

    pub fn histogram_sum(name: &str, sum: f64, timestamp_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            value: sum,
            kind: MetricKind::HistogramSum,
            timestamp_ms,
            labels: HashMap::new(),
            unit: None,
            description: None,
        }
    }

    pub fn histogram_count(name: &str, count: f64, timestamp_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            value: count,
            kind: MetricKind::HistogramCount,
            timestamp_ms,
            labels: HashMap::new(),
            unit: None,
            description: None,
        }
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.to_string());
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// Kind of metric for export formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    Counter,
    Gauge,
    HistogramSum,
    HistogramCount,
    HistogramBucket,
    Summary,
    Untyped,
}

impl MetricKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricKind::Counter => "counter",
            MetricKind::Gauge => "gauge",
            MetricKind::HistogramSum => "histogram_sum",
            MetricKind::HistogramCount => "histogram_count",
            MetricKind::HistogramBucket => "histogram_bucket",
            MetricKind::Summary => "summary",
            MetricKind::Untyped => "untyped",
        }
    }
}

impl fmt::Display for MetricKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Metric Filter ───────────────────────────────────────────

/// Filter for selecting which metrics to export.
#[derive(Debug, Clone)]
pub struct MetricFilter {
    /// Include only metrics whose names match these prefixes (empty = include all).
    pub include_prefixes: Vec<String>,
    /// Exclude metrics whose names match these prefixes.
    pub exclude_prefixes: Vec<String>,
    /// Include only these metric kinds (empty = include all).
    pub include_kinds: Vec<MetricKind>,
    /// Required labels (metric must have all of these).
    pub required_labels: HashMap<String, String>,
}

impl MetricFilter {
    pub fn new() -> Self {
        Self {
            include_prefixes: Vec::new(),
            exclude_prefixes: Vec::new(),
            include_kinds: Vec::new(),
            required_labels: HashMap::new(),
        }
    }

    pub fn include_prefix(mut self, prefix: &str) -> Self {
        self.include_prefixes.push(prefix.to_string());
        self
    }

    pub fn exclude_prefix(mut self, prefix: &str) -> Self {
        self.exclude_prefixes.push(prefix.to_string());
        self
    }

    pub fn include_kind(mut self, kind: MetricKind) -> Self {
        self.include_kinds.push(kind);
        self
    }

    pub fn require_label(mut self, key: &str, value: &str) -> Self {
        self.required_labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if a metric point passes this filter.
    pub fn matches(&self, point: &MetricPoint) -> bool {
        // Check include prefixes
        if !self.include_prefixes.is_empty()
            && !self
                .include_prefixes
                .iter()
                .any(|p| point.name.starts_with(p))
        {
            return false;
        }

        // Check exclude prefixes
        if self
            .exclude_prefixes
            .iter()
            .any(|p| point.name.starts_with(p))
        {
            return false;
        }

        // Check kinds
        if !self.include_kinds.is_empty() && !self.include_kinds.contains(&point.kind) {
            return false;
        }

        // Check required labels
        for (k, v) in &self.required_labels {
            match point.labels.get(k) {
                Some(val) if val == v => {}
                _ => return false,
            }
        }

        true
    }

    /// Filter a batch of points.
    pub fn filter<'a>(&self, points: &'a [MetricPoint]) -> Vec<&'a MetricPoint> {
        points.iter().filter(|p| self.matches(p)).collect()
    }
}

impl Default for MetricFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Export Formats ───────────────────────────────────────────

/// Export a batch of metric points as JSON.
pub fn export_json(points: &[MetricPoint]) -> String {
    let mut out = String::from("[\n");
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str("  {");
        out.push_str(&format!("\"name\":\"{}\",", escape_json(&p.name)));
        out.push_str(&format!("\"value\":{},", format_json_number(p.value)));
        out.push_str(&format!("\"kind\":\"{}\",", p.kind.as_str()));
        out.push_str(&format!("\"timestamp_ms\":{}", p.timestamp_ms));
        if !p.labels.is_empty() {
            out.push_str(",\"labels\":{");
            let mut sorted_labels: Vec<(&String, &String)> = p.labels.iter().collect();
            sorted_labels.sort_by_key(|(k, _)| k.as_str());
            for (j, (k, v)) in sorted_labels.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)));
            }
            out.push('}');
        }
        if let Some(unit) = &p.unit {
            out.push_str(&format!(",\"unit\":\"{}\"", escape_json(unit)));
        }
        if let Some(desc) = &p.description {
            out.push_str(&format!(",\"description\":\"{}\"", escape_json(desc)));
        }
        out.push('}');
    }
    out.push_str("\n]");
    out
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn format_json_number(v: f64) -> String {
    if v.is_nan() {
        "null".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "\"Infinity\"".to_string()
        } else {
            "\"-Infinity\"".to_string()
        }
    } else if v == v.floor() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Export in OpenMetrics text format.
pub fn export_openmetrics(points: &[MetricPoint]) -> String {
    let mut out = String::new();

    // Group by base metric name
    let mut families: Vec<String> = Vec::new();
    let mut seen_names: Vec<String> = Vec::new();
    for p in points {
        let base = base_metric_name(&p.name);
        if !seen_names.contains(&base) {
            seen_names.push(base.clone());
        }
    }

    for base in &seen_names {
        // Find all points for this base name
        let family_points: Vec<&MetricPoint> = points
            .iter()
            .filter(|p| base_metric_name(&p.name) == *base)
            .collect();
        if family_points.is_empty() {
            continue;
        }

        let first = family_points[0];
        let type_str = match first.kind {
            MetricKind::Counter => "counter",
            MetricKind::Gauge => "gauge",
            MetricKind::HistogramSum | MetricKind::HistogramCount | MetricKind::HistogramBucket => {
                "histogram"
            }
            MetricKind::Summary => "summary",
            MetricKind::Untyped => "unknown",
        };

        if let Some(desc) = &first.description {
            families.push(format!(
                "# HELP {} {}",
                base,
                desc.replace('\\', "\\\\").replace('\n', "\\n")
            ));
        }
        families.push(format!("# TYPE {} {}", base, type_str));

        for p in &family_points {
            let label_str = format_openmetrics_labels(&p.labels);
            let value_str = format_openmetrics_value(p.value);
            families.push(format!(
                "{}{} {} {}",
                p.name, label_str, value_str, p.timestamp_ms
            ));
        }
    }

    out.push_str(&families.join("\n"));
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str("# EOF\n");
    out
}

fn base_metric_name(name: &str) -> String {
    let suffixes = ["_total", "_sum", "_count", "_bucket"];
    for suffix in &suffixes {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    name.to_string()
}

fn format_openmetrics_labels(labels: &HashMap<String, String>) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let mut sorted: Vec<(&String, &String)> = labels.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());
    let parts: Vec<String> = sorted
        .iter()
        .map(|(k, v)| {
            format!(
                "{}=\"{}\"",
                k,
                v.replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
            )
        })
        .collect();
    format!("{{{}}}", parts.join(","))
}

fn format_openmetrics_value(v: f64) -> String {
    if v.is_infinite() && v.is_sign_positive() {
        "+Inf".to_string()
    } else if v.is_infinite() && v.is_sign_negative() {
        "-Inf".to_string()
    } else if v.is_nan() {
        "NaN".to_string()
    } else if v == v.floor() && v.abs() < 1e15 {
        format!("{}.0", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Export in StatsD format (one line per metric).
///
/// Format: `<metric_name>:<value>|<type>[|#tag1:val1,tag2:val2]`
pub fn export_statsd(points: &[MetricPoint]) -> String {
    let mut lines = Vec::new();
    for p in points {
        let type_char = match p.kind {
            MetricKind::Counter => "c",
            MetricKind::Gauge => "g",
            MetricKind::HistogramSum | MetricKind::HistogramCount | MetricKind::HistogramBucket => {
                "ms"
            }
            MetricKind::Summary => "ms",
            MetricKind::Untyped => "g",
        };

        let value_str = if p.value == p.value.floor() && p.value.abs() < 1e15 {
            format!("{}", p.value as i64)
        } else {
            format!("{}", p.value)
        };

        let mut line = format!("{}:{}|{}", p.name, value_str, type_char);

        if !p.labels.is_empty() {
            let mut sorted: Vec<(&String, &String)> = p.labels.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            let tags: Vec<String> = sorted
                .iter()
                .map(|(k, v)| format!("{}:{}", k, v))
                .collect();
            line.push_str(&format!("|#{}", tags.join(",")));
        }

        lines.push(line);
    }
    lines.join("\n")
}

/// Export in Graphite line protocol.
///
/// Format: `<metric_path> <value> <timestamp_secs>`
pub fn export_graphite(points: &[MetricPoint]) -> String {
    let mut lines = Vec::new();
    for p in points {
        // Labels become part of the path
        let mut path = p.name.clone();
        if !p.labels.is_empty() {
            let mut sorted: Vec<(&String, &String)> = p.labels.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in &sorted {
                path.push_str(&format!(".{}.{}", k, v));
            }
        }
        // Replace spaces with underscores in path
        let path = path.replace(' ', "_");
        let ts_secs = p.timestamp_ms / 1000;
        let value_str = if p.value == p.value.floor() && p.value.abs() < 1e15 {
            format!("{}", p.value as i64)
        } else {
            format!("{}", p.value)
        };
        lines.push(format!("{} {} {}", path, value_str, ts_secs));
    }
    lines.join("\n")
}

/// Export in OTLP-like (OpenTelemetry Protocol) text format.
///
/// This is a simplified text representation of the OTLP data model.
pub fn export_otlp(points: &[MetricPoint]) -> String {
    let mut out = String::new();
    out.push_str("ResourceMetrics {\n");
    out.push_str("  ScopeMetrics {\n");

    for p in points {
        out.push_str("    Metric {\n");
        out.push_str(&format!("      name: \"{}\"\n", p.name));
        if let Some(desc) = &p.description {
            out.push_str(&format!("      description: \"{}\"\n", desc));
        }
        if let Some(unit) = &p.unit {
            out.push_str(&format!("      unit: \"{}\"\n", unit));
        }

        let data_type = match p.kind {
            MetricKind::Counter => "Sum",
            MetricKind::Gauge => "Gauge",
            MetricKind::HistogramSum
            | MetricKind::HistogramCount
            | MetricKind::HistogramBucket => "Histogram",
            MetricKind::Summary => "Summary",
            MetricKind::Untyped => "Gauge",
        };
        out.push_str(&format!("      data_type: {}\n", data_type));

        out.push_str("      DataPoint {\n");
        let value_str = if p.value == p.value.floor() && p.value.abs() < 1e15 {
            format!("{}", p.value as i64)
        } else {
            format!("{}", p.value)
        };
        out.push_str(&format!("        value: {}\n", value_str));
        out.push_str(&format!("        time_unix_nano: {}\n", p.timestamp_ms as u64 * 1_000_000));

        if !p.labels.is_empty() {
            out.push_str("        attributes: [");
            let mut sorted: Vec<(&String, &String)> = p.labels.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            let attrs: Vec<String> = sorted
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            out.push_str(&attrs.join(", "));
            out.push_str("]\n");
        }
        out.push_str("      }\n");
        out.push_str("    }\n");
    }

    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

// ── Export Config ────────────────────────────────────────────

/// Format to export metrics in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    OpenMetrics,
    StatsD,
    Graphite,
    Otlp,
}

impl ExportFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::OpenMetrics => "openmetrics",
            ExportFormat::StatsD => "statsd",
            ExportFormat::Graphite => "graphite",
            ExportFormat::Otlp => "otlp",
        }
    }

    /// Content-Type for this format.
    pub fn content_type(&self) -> &'static str {
        match self {
            ExportFormat::Json => "application/json",
            ExportFormat::OpenMetrics => "application/openmetrics-text; version=1.0.0; charset=utf-8",
            ExportFormat::StatsD => "text/plain",
            ExportFormat::Graphite => "text/plain",
            ExportFormat::Otlp => "text/plain",
        }
    }
}

impl fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for a metric exporter.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    /// Export interval in milliseconds.
    pub interval_ms: u64,
    /// Maximum batch size.
    pub batch_size: usize,
    /// Filter to apply before export.
    pub filter: MetricFilter,
    /// Prefix to prepend to all metric names on export.
    pub prefix: Option<String>,
}

impl ExportConfig {
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            interval_ms: 10_000,
            batch_size: 1000,
            filter: MetricFilter::new(),
            prefix: None,
        }
    }

    pub fn with_interval(mut self, interval_ms: u64) -> Self {
        self.interval_ms = interval_ms;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_filter(mut self, filter: MetricFilter) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = Some(prefix.to_string());
        self
    }
}

// ── Batch Exporter ──────────────────────────────────────────

/// Batch metric exporter with buffering, filtering, and multi-format output.
#[derive(Debug, Clone)]
pub struct BatchExporter {
    config: ExportConfig,
    buffer: Vec<MetricPoint>,
    /// Total points exported.
    total_exported: u64,
    /// Total export batches.
    total_batches: u64,
}

impl BatchExporter {
    pub fn new(config: ExportConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            total_exported: 0,
            total_batches: 0,
        }
    }

    /// Add a metric point to the buffer.
    pub fn add(&mut self, point: MetricPoint) {
        if self.config.filter.matches(&point) {
            let point = match &self.config.prefix {
                Some(prefix) => MetricPoint {
                    name: format!("{}.{}", prefix, point.name),
                    ..point
                },
                None => point,
            };
            self.buffer.push(point);
        }
    }

    /// Add multiple points.
    pub fn add_batch(&mut self, points: &[MetricPoint]) {
        for p in points {
            self.add(p.clone());
        }
    }

    /// Number of buffered points.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is at or above the batch size.
    pub fn should_flush(&self) -> bool {
        self.buffer.len() >= self.config.batch_size
    }

    /// Flush the buffer and return the serialized output.
    pub fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }
        let points = std::mem::take(&mut self.buffer);
        self.total_exported += points.len() as u64;
        self.total_batches += 1;
        self.export_points(&points)
    }

    /// Export specific points without buffering.
    pub fn export_points(&self, points: &[MetricPoint]) -> String {
        match self.config.format {
            ExportFormat::Json => export_json(points),
            ExportFormat::OpenMetrics => export_openmetrics(points),
            ExportFormat::StatsD => export_statsd(points),
            ExportFormat::Graphite => export_graphite(points),
            ExportFormat::Otlp => export_otlp(points),
        }
    }

    pub fn total_exported(&self) -> u64 {
        self.total_exported
    }

    pub fn total_batches(&self) -> u64 {
        self.total_batches
    }

    pub fn format(&self) -> ExportFormat {
        self.config.format
    }

    /// Content type for the configured format.
    pub fn content_type(&self) -> &'static str {
        self.config.format.content_type()
    }
}

/// Export points in the specified format.
pub fn export(points: &[MetricPoint], format: ExportFormat) -> String {
    match format {
        ExportFormat::Json => export_json(points),
        ExportFormat::OpenMetrics => export_openmetrics(points),
        ExportFormat::StatsD => export_statsd(points),
        ExportFormat::Graphite => export_graphite(points),
        ExportFormat::Otlp => export_otlp(points),
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<MetricPoint> {
        vec![
            MetricPoint::counter("http_requests_total", 1027.0, 1700000000000)
                .with_label("method", "GET")
                .with_label("code", "200"),
            MetricPoint::gauge("temperature", 22.5, 1700000000000)
                .with_description("Current temp"),
            MetricPoint::counter("http_requests_total", 3.0, 1700000000000)
                .with_label("method", "POST")
                .with_label("code", "201"),
        ]
    }

    #[test]
    fn test_metric_point_counter() {
        let p = MetricPoint::counter("req", 5.0, 1000);
        assert_eq!(p.kind, MetricKind::Counter);
        assert_eq!(p.value, 5.0);
    }

    #[test]
    fn test_metric_point_gauge() {
        let p = MetricPoint::gauge("temp", 22.0, 1000);
        assert_eq!(p.kind, MetricKind::Gauge);
    }

    #[test]
    fn test_metric_point_with_labels() {
        let p = MetricPoint::counter("req", 1.0, 0)
            .with_label("method", "GET")
            .with_label("path", "/api");
        assert_eq!(p.labels.len(), 2);
    }

    #[test]
    fn test_metric_kind_display() {
        assert_eq!(MetricKind::Counter.to_string(), "counter");
        assert_eq!(MetricKind::Gauge.to_string(), "gauge");
    }

    #[test]
    fn test_export_json() {
        let points = vec![
            MetricPoint::counter("hits", 42.0, 1000),
        ];
        let json = export_json(&points);
        assert!(json.contains("\"name\":\"hits\""));
        assert!(json.contains("\"value\":42"));
        assert!(json.contains("\"kind\":\"counter\""));
        assert!(json.contains("\"timestamp_ms\":1000"));
    }

    #[test]
    fn test_export_json_with_labels() {
        let points = vec![
            MetricPoint::counter("req", 5.0, 1000)
                .with_label("method", "GET"),
        ];
        let json = export_json(&points);
        assert!(json.contains("\"labels\":{"));
        assert!(json.contains("\"method\":\"GET\""));
    }

    #[test]
    fn test_export_json_special_values() {
        let points = vec![
            MetricPoint::gauge("nan_val", f64::NAN, 0),
            MetricPoint::gauge("inf_val", f64::INFINITY, 0),
        ];
        let json = export_json(&points);
        assert!(json.contains("null")); // NaN -> null
        assert!(json.contains("\"Infinity\"")); // +Inf -> "Infinity"
    }

    #[test]
    fn test_export_openmetrics() {
        let points = vec![
            MetricPoint::counter("requests_total", 100.0, 1000)
                .with_description("Total requests"),
        ];
        let text = export_openmetrics(&points);
        assert!(text.contains("# HELP requests Total requests"));
        assert!(text.contains("# TYPE requests counter"));
        assert!(text.contains("requests_total 100.0 1000"));
        assert!(text.contains("# EOF"));
    }

    #[test]
    fn test_export_statsd() {
        let points = vec![
            MetricPoint::counter("hits", 10.0, 0),
            MetricPoint::gauge("temp", 22.5, 0)
                .with_label("host", "web01"),
        ];
        let text = export_statsd(&points);
        assert!(text.contains("hits:10|c"));
        assert!(text.contains("temp:22.5|g|#host:web01"));
    }

    #[test]
    fn test_export_graphite() {
        let points = vec![
            MetricPoint::counter("app.requests", 42.0, 1700000000000),
        ];
        let text = export_graphite(&points);
        assert!(text.contains("app.requests 42 1700000000"));
    }

    #[test]
    fn test_export_graphite_with_labels() {
        let points = vec![
            MetricPoint::counter("requests", 5.0, 1000000)
                .with_label("host", "web01"),
        ];
        let text = export_graphite(&points);
        assert!(text.contains("requests.host.web01 5 1000"));
    }

    #[test]
    fn test_export_otlp() {
        let points = vec![
            MetricPoint::gauge("cpu_usage", 72.5, 1000)
                .with_unit("percent")
                .with_description("CPU usage"),
        ];
        let text = export_otlp(&points);
        assert!(text.contains("ResourceMetrics"));
        assert!(text.contains("name: \"cpu_usage\""));
        assert!(text.contains("data_type: Gauge"));
        assert!(text.contains("value: 72"));
        assert!(text.contains("unit: \"percent\""));
    }

    #[test]
    fn test_metric_filter_include_prefix() {
        let filter = MetricFilter::new().include_prefix("http");
        let p1 = MetricPoint::counter("http_requests", 1.0, 0);
        let p2 = MetricPoint::counter("db_queries", 1.0, 0);
        assert!(filter.matches(&p1));
        assert!(!filter.matches(&p2));
    }

    #[test]
    fn test_metric_filter_exclude_prefix() {
        let filter = MetricFilter::new().exclude_prefix("internal");
        let p1 = MetricPoint::counter("internal_debug", 1.0, 0);
        let p2 = MetricPoint::counter("http_requests", 1.0, 0);
        assert!(!filter.matches(&p1));
        assert!(filter.matches(&p2));
    }

    #[test]
    fn test_metric_filter_kind() {
        let filter = MetricFilter::new().include_kind(MetricKind::Counter);
        let p1 = MetricPoint::counter("c", 1.0, 0);
        let p2 = MetricPoint::gauge("g", 1.0, 0);
        assert!(filter.matches(&p1));
        assert!(!filter.matches(&p2));
    }

    #[test]
    fn test_metric_filter_required_label() {
        let filter = MetricFilter::new().require_label("env", "prod");
        let p1 = MetricPoint::counter("c", 1.0, 0).with_label("env", "prod");
        let p2 = MetricPoint::counter("c", 1.0, 0).with_label("env", "dev");
        let p3 = MetricPoint::counter("c", 1.0, 0);
        assert!(filter.matches(&p1));
        assert!(!filter.matches(&p2));
        assert!(!filter.matches(&p3));
    }

    #[test]
    fn test_metric_filter_batch() {
        let filter = MetricFilter::new().include_prefix("http");
        let points = vec![
            MetricPoint::counter("http_req", 1.0, 0),
            MetricPoint::counter("db_query", 1.0, 0),
            MetricPoint::counter("http_err", 1.0, 0),
        ];
        let filtered = filter.filter(&points);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_batch_exporter() {
        let config = ExportConfig::new(ExportFormat::Json)
            .with_batch_size(2)
            .with_interval(5000);
        let mut exporter = BatchExporter::new(config);
        exporter.add(MetricPoint::counter("a", 1.0, 0));
        assert_eq!(exporter.buffered(), 1);
        assert!(!exporter.should_flush());
        exporter.add(MetricPoint::counter("b", 2.0, 0));
        assert!(exporter.should_flush());
        let output = exporter.flush();
        assert!(output.contains("\"name\":\"a\""));
        assert!(output.contains("\"name\":\"b\""));
        assert_eq!(exporter.buffered(), 0);
        assert_eq!(exporter.total_exported(), 2);
        assert_eq!(exporter.total_batches(), 1);
    }

    #[test]
    fn test_batch_exporter_with_prefix() {
        let config = ExportConfig::new(ExportFormat::StatsD)
            .with_prefix("myapp");
        let mut exporter = BatchExporter::new(config);
        exporter.add(MetricPoint::counter("req", 1.0, 0));
        let output = exporter.flush();
        assert!(output.contains("myapp.req"));
    }

    #[test]
    fn test_batch_exporter_with_filter() {
        let filter = MetricFilter::new().include_prefix("http");
        let config = ExportConfig::new(ExportFormat::Json).with_filter(filter);
        let mut exporter = BatchExporter::new(config);
        exporter.add(MetricPoint::counter("http_req", 1.0, 0));
        exporter.add(MetricPoint::counter("db_query", 1.0, 0));
        assert_eq!(exporter.buffered(), 1); // db_query filtered out
    }

    #[test]
    fn test_batch_exporter_flush_empty() {
        let config = ExportConfig::new(ExportFormat::Json);
        let mut exporter = BatchExporter::new(config);
        assert_eq!(exporter.flush(), "");
    }

    #[test]
    fn test_export_format_content_type() {
        assert_eq!(ExportFormat::Json.content_type(), "application/json");
        assert!(ExportFormat::OpenMetrics.content_type().contains("openmetrics"));
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(format!("{}", ExportFormat::StatsD), "statsd");
        assert_eq!(format!("{}", ExportFormat::Graphite), "graphite");
    }

    #[test]
    fn test_export_function() {
        let points = sample_points();
        let json = export(&points, ExportFormat::Json);
        assert!(json.contains("http_requests_total"));
        let statsd = export(&points, ExportFormat::StatsD);
        assert!(statsd.contains("|c"));
    }

    #[test]
    fn test_batch_exporter_add_batch() {
        let config = ExportConfig::new(ExportFormat::Json);
        let mut exporter = BatchExporter::new(config);
        let points = vec![
            MetricPoint::counter("a", 1.0, 0),
            MetricPoint::counter("b", 2.0, 0),
        ];
        exporter.add_batch(&points);
        assert_eq!(exporter.buffered(), 2);
    }

    #[test]
    fn test_export_config_defaults() {
        let config = ExportConfig::new(ExportFormat::Json);
        assert_eq!(config.interval_ms, 10_000);
        assert_eq!(config.batch_size, 1000);
        assert!(config.prefix.is_none());
    }

    #[test]
    fn test_export_json_with_unit_and_desc() {
        let points = vec![
            MetricPoint::gauge("temp", 22.0, 0)
                .with_unit("celsius")
                .with_description("Temperature reading"),
        ];
        let json = export_json(&points);
        assert!(json.contains("\"unit\":\"celsius\""));
        assert!(json.contains("\"description\":\"Temperature reading\""));
    }
}
