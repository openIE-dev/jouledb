use crate::registry::MetricsSnapshot;

/// Exports metrics in Prometheus text exposition format.
///
/// Produces output compatible with Prometheus scrapers:
/// ```text
/// # HELP metric_name description
/// # TYPE metric_name counter|gauge|histogram
/// metric_name{label="value"} 42
/// ```
pub struct PrometheusExporter;

impl PrometheusExporter {
    /// Export a `MetricsSnapshot` to Prometheus text format.
    pub fn export(snapshot: &MetricsSnapshot) -> String {
        let mut out = String::new();

        // Counters
        for c in &snapshot.counters {
            out.push_str(&format!("# HELP {} {}\n", c.name, c.description));
            out.push_str(&format!("# TYPE {} counter\n", c.name));
            let labels = Self::format_labels(&c.labels);
            out.push_str(&format!("{}{} {}\n", c.name, labels, c.value));
        }

        // Gauges
        for g in &snapshot.gauges {
            out.push_str(&format!("# HELP {} {}\n", g.name, g.description));
            out.push_str(&format!("# TYPE {} gauge\n", g.name));
            let labels = Self::format_labels(&g.labels);
            out.push_str(&format!("{}{} {}\n", g.name, labels, g.value));
        }

        // Histograms
        for h in &snapshot.histograms {
            out.push_str(&format!("# HELP {} {}\n", h.name, h.description));
            out.push_str(&format!("# TYPE {} histogram\n", h.name));
            let labels = Self::format_labels(&h.labels);

            let mut cumulative = 0u64;
            for bucket in &h.buckets {
                cumulative += bucket.count;
                let le = if bucket.upper_bound.is_infinite() {
                    "+Inf".to_string()
                } else {
                    format!("{}", bucket.upper_bound)
                };
                if labels.is_empty() {
                    out.push_str(&format!(
                        "{}_bucket{{le=\"{}\"}} {}\n",
                        h.name, le, cumulative
                    ));
                } else {
                    // Strip the surrounding braces and merge with le label.
                    let inner = &labels[1..labels.len() - 1];
                    out.push_str(&format!(
                        "{}_bucket{{{},le=\"{}\"}} {}\n",
                        h.name, inner, le, cumulative
                    ));
                }
            }
            out.push_str(&format!("{}_sum{} {}\n", h.name, labels, h.sum));
            out.push_str(&format!("{}_count{} {}\n", h.name, labels, h.count));
        }

        out
    }

    /// Format labels as `{key="value",key2="value2"}` or empty string.
    fn format_labels(labels: &std::collections::HashMap<String, String>) -> String {
        if labels.is_empty() {
            return String::new();
        }
        let mut pairs: Vec<_> = labels.iter().collect();
        pairs.sort_by_key(|(k, _)| (*k).clone());
        let inner: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
        format!("{{{}}}", inner.join(","))
    }
}

/// Exports metrics as a JSON value.
///
/// Uses the serde-derived serialization from `MetricsSnapshot`.
pub struct JsonExporter;

impl JsonExporter {
    /// Export a `MetricsSnapshot` to a JSON value.
    pub fn export(snapshot: &MetricsSnapshot) -> serde_json::Value {
        serde_json::to_value(snapshot)
            .unwrap_or(serde_json::json!({"error": "serialization failed"}))
    }

    /// Export a `MetricsSnapshot` to a pretty-printed JSON string.
    pub fn export_string(snapshot: &MetricsSnapshot) -> String {
        serde_json::to_string_pretty(snapshot).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::MetricsRegistry;

    fn sample_registry() -> MetricsRegistry {
        let reg = MetricsRegistry::new();
        let c = reg.counter("http_requests_total", "Total HTTP requests");
        c.inc_by(42);

        let g = reg.gauge("cpu_usage", "CPU utilization");
        g.set(72.5);

        let h = reg.histogram_with_buckets(
            "request_duration_seconds",
            "Request duration",
            &[0.1, 0.5, 1.0],
        );
        h.observe(0.05);
        h.observe(0.3);
        h.observe(0.8);

        reg
    }

    #[test]
    fn prometheus_counter_format() {
        let reg = MetricsRegistry::new();
        reg.counter("test_total", "A test counter").inc_by(10);
        let snap = reg.snapshot();
        let output = PrometheusExporter::export(&snap);

        assert!(output.contains("# HELP test_total A test counter"));
        assert!(output.contains("# TYPE test_total counter"));
        assert!(output.contains("test_total 10"));
    }

    #[test]
    fn prometheus_gauge_format() {
        let reg = MetricsRegistry::new();
        reg.gauge("temperature", "Current temp").set(23.5);
        let snap = reg.snapshot();
        let output = PrometheusExporter::export(&snap);

        assert!(output.contains("# TYPE temperature gauge"));
        assert!(output.contains("temperature 23.5"));
    }

    #[test]
    fn prometheus_histogram_format() {
        let reg = MetricsRegistry::new();
        let h = reg.histogram_with_buckets("dur", "Duration", &[0.1, 1.0]);
        h.observe(0.05);
        h.observe(0.5);

        let snap = reg.snapshot();
        let output = PrometheusExporter::export(&snap);

        assert!(output.contains("# TYPE dur histogram"));
        assert!(output.contains("dur_bucket{le=\"0.1\"} 1"));
        assert!(output.contains("dur_bucket{le=\"1\"} 2"));
        assert!(output.contains("dur_bucket{le=\"+Inf\"} 2"));
        assert!(output.contains("dur_sum"));
        assert!(output.contains("dur_count 2"));
    }

    #[test]
    fn prometheus_with_labels() {
        let reg = MetricsRegistry::new();
        let mut labels = std::collections::HashMap::new();
        labels.insert("method".into(), "GET".into());
        reg.counter_with_labels("http_requests", "Requests", labels)
            .inc_by(5);

        let snap = reg.snapshot();
        let output = PrometheusExporter::export(&snap);
        assert!(output.contains("http_requests{method=\"GET\"} 5"));
    }

    #[test]
    fn prometheus_cumulative_buckets() {
        let reg = MetricsRegistry::new();
        let h = reg.histogram_with_buckets("x", "", &[1.0, 5.0, 10.0]);
        h.observe(0.5); // bucket ≤1.0
        h.observe(3.0); // bucket ≤5.0
        h.observe(7.0); // bucket ≤10.0

        let snap = reg.snapshot();
        let output = PrometheusExporter::export(&snap);

        // Cumulative: ≤1.0 → 1, ≤5.0 → 2, ≤10.0 → 3, +Inf → 3
        assert!(output.contains("x_bucket{le=\"1\"} 1"));
        assert!(output.contains("x_bucket{le=\"5\"} 2"));
        assert!(output.contains("x_bucket{le=\"10\"} 3"));
        assert!(output.contains("x_bucket{le=\"+Inf\"} 3"));
    }

    #[test]
    fn json_export_structure() {
        let reg = sample_registry();
        let snap = reg.snapshot();
        let json = JsonExporter::export(&snap);

        assert!(json["counters"].is_array());
        assert!(json["gauges"].is_array());
        assert!(json["histograms"].is_array());
    }

    #[test]
    fn json_export_string() {
        let reg = MetricsRegistry::new();
        reg.counter("x", "desc");
        let snap = reg.snapshot();
        let s = JsonExporter::export_string(&snap);
        assert!(s.contains("\"counters\""));
    }

    #[test]
    fn empty_snapshot_exports_cleanly() {
        let reg = MetricsRegistry::new();
        let snap = reg.snapshot();
        let prom = PrometheusExporter::export(&snap);
        assert!(prom.is_empty());
        let json = JsonExporter::export(&snap);
        assert!(json["counters"].as_array().unwrap().is_empty());
    }

    #[test]
    fn labels_sorted_deterministically() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("z".into(), "3".into());
        labels.insert("a".into(), "1".into());
        labels.insert("m".into(), "2".into());

        let formatted = PrometheusExporter::format_labels(&labels);
        assert_eq!(formatted, "{a=\"1\",m=\"2\",z=\"3\"}");
    }

    #[test]
    fn full_pipeline() {
        let reg = sample_registry();
        let snap = reg.snapshot();

        // Prometheus
        let prom = PrometheusExporter::export(&snap);
        assert!(!prom.is_empty());
        assert!(prom.contains("# HELP"));
        assert!(prom.contains("# TYPE"));

        // JSON
        let json = JsonExporter::export(&snap);
        assert!(!json["counters"].as_array().unwrap().is_empty());
    }
}
