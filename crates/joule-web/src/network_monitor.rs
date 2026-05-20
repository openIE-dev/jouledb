//! Network request monitor with filtering, waterfall timing, and HAR-like export.
//!
//! Models browser-style network inspection: each request records method, URL,
//! headers, body, status, timing breakdowns (DNS, connect, TLS, TTFB, content),
//! and can be filtered, aggregated, or exported.

use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// Waterfall timing breakdown for a network request.
#[derive(Debug, Clone, Default)]
pub struct WaterfallTiming {
    pub dns_ms: f64,
    pub connect_ms: f64,
    pub tls_ms: f64,
    pub ttfb_ms: f64,
    pub content_ms: f64,
}

impl WaterfallTiming {
    pub fn total_ms(&self) -> f64 {
        self.dns_ms + self.connect_ms + self.tls_ms + self.ttfb_ms + self.content_ms
    }
}

/// A single network request/response entry.
#[derive(Debug, Clone)]
pub struct RequestEntry {
    pub id: String,
    pub method: String,
    pub url: String,
    pub request_headers: HashMap<String, String>,
    pub request_body: Option<String>,
    pub response_status: Option<u16>,
    pub response_headers: HashMap<String, String>,
    pub response_body: Option<String>,
    pub start_time: f64,
    pub duration_ms: Option<f64>,
    pub size_bytes: Option<u64>,
    pub initiator: Option<String>,
    pub content_type: Option<String>,
    pub waterfall: Option<WaterfallTiming>,
}

impl RequestEntry {
    pub fn new(method: &str, url: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            method: method.to_string(),
            url: url.to_string(),
            request_headers: HashMap::new(),
            request_body: None,
            response_status: None,
            response_headers: HashMap::new(),
            response_body: None,
            start_time: 0.0,
            duration_ms: None,
            size_bytes: None,
            initiator: None,
            content_type: None,
            waterfall: None,
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.response_status = Some(status);
        self
    }

    pub fn with_duration(mut self, ms: f64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    pub fn with_size(mut self, bytes: u64) -> Self {
        self.size_bytes = Some(bytes);
        self
    }

    pub fn with_start_time(mut self, t: f64) -> Self {
        self.start_time = t;
        self
    }

    pub fn with_initiator(mut self, initiator: &str) -> Self {
        self.initiator = Some(initiator.to_string());
        self
    }

    pub fn with_content_type(mut self, ct: &str) -> Self {
        self.content_type = Some(ct.to_string());
        self
    }

    pub fn with_request_header(mut self, key: &str, value: &str) -> Self {
        self.request_headers
            .insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_response_header(mut self, key: &str, value: &str) -> Self {
        self.response_headers
            .insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_waterfall(mut self, wf: WaterfallTiming) -> Self {
        self.waterfall = Some(wf);
        self
    }

    /// Check if status indicates success (2xx).
    pub fn is_success(&self) -> bool {
        self.response_status
            .map(|s| (200..300).contains(&s))
            .unwrap_or(false)
    }

    /// Check if status indicates an error (4xx or 5xx).
    pub fn is_error(&self) -> bool {
        self.response_status
            .map(|s| s >= 400)
            .unwrap_or(false)
    }
}

/// Aggregate network stats.
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub total_requests: usize,
    pub total_size_bytes: u64,
    pub total_duration_ms: f64,
    pub error_count: usize,
    pub average_duration_ms: f64,
}

// ── RequestLog ──

/// A log of network requests with filtering and export.
pub struct RequestLog {
    entries: Vec<RequestEntry>,
}

impl RequestLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, entry: RequestEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[RequestEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Filter by HTTP method.
    pub fn filter_by_method(&self, method: &str) -> Vec<&RequestEntry> {
        let m = method.to_uppercase();
        self.entries.iter().filter(|e| e.method == m).collect()
    }

    /// Filter by status code range (inclusive).
    pub fn filter_by_status_range(&self, min: u16, max: u16) -> Vec<&RequestEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.response_status
                    .map(|s| s >= min && s <= max)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Filter by URL substring pattern.
    pub fn filter_by_url(&self, pattern: &str) -> Vec<&RequestEntry> {
        self.entries
            .iter()
            .filter(|e| e.url.contains(pattern))
            .collect()
    }

    /// Filter by content type substring.
    pub fn filter_by_content_type(&self, ct: &str) -> Vec<&RequestEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.content_type
                    .as_ref()
                    .map(|t| t.contains(ct))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Compute aggregate statistics.
    pub fn stats(&self) -> NetworkStats {
        let total_requests = self.entries.len();
        let total_size_bytes: u64 = self
            .entries
            .iter()
            .filter_map(|e| e.size_bytes)
            .sum();
        let total_duration_ms: f64 = self
            .entries
            .iter()
            .filter_map(|e| e.duration_ms)
            .sum();
        let error_count = self.entries.iter().filter(|e| e.is_error()).count();
        let entries_with_dur = self
            .entries
            .iter()
            .filter(|e| e.duration_ms.is_some())
            .count();
        let average_duration_ms = if entries_with_dur > 0 {
            total_duration_ms / entries_with_dur as f64
        } else {
            0.0
        };

        NetworkStats {
            total_requests,
            total_size_bytes,
            total_duration_ms,
            error_count,
            average_duration_ms,
        }
    }

    /// Export entries in a HAR-like JSON format.
    pub fn to_har_json(&self) -> Value {
        let entries: Vec<Value> = self
            .entries
            .iter()
            .map(|e| {
                let mut entry = json!({
                    "request": {
                        "method": e.method,
                        "url": e.url,
                        "headers": header_list(&e.request_headers),
                    },
                    "response": {
                        "status": e.response_status.unwrap_or(0),
                        "headers": header_list(&e.response_headers),
                        "content": {
                            "size": e.size_bytes.unwrap_or(0),
                            "mimeType": e.content_type.as_deref().unwrap_or(""),
                        },
                    },
                    "startedDateTime": e.start_time,
                    "time": e.duration_ms.unwrap_or(0.0),
                });

                if let Some(wf) = &e.waterfall {
                    entry["timings"] = json!({
                        "dns": wf.dns_ms,
                        "connect": wf.connect_ms,
                        "ssl": wf.tls_ms,
                        "wait": wf.ttfb_ms,
                        "receive": wf.content_ms,
                    });
                }

                entry
            })
            .collect();

        json!({
            "log": {
                "version": "1.2",
                "entries": entries,
            }
        })
    }
}

impl Default for RequestLog {
    fn default() -> Self {
        Self::new()
    }
}

fn header_list(headers: &HashMap<String, String>) -> Vec<Value> {
    headers
        .iter()
        .map(|(k, v)| json!({"name": k, "value": v}))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_log() -> RequestLog {
        let mut log = RequestLog::new();
        log.add(
            RequestEntry::new("GET", "https://api.example.com/users")
                .with_status(200)
                .with_duration(120.0)
                .with_size(4096)
                .with_content_type("application/json"),
        );
        log.add(
            RequestEntry::new("POST", "https://api.example.com/users")
                .with_status(201)
                .with_duration(200.0)
                .with_size(512),
        );
        log.add(
            RequestEntry::new("GET", "https://cdn.example.com/style.css")
                .with_status(404)
                .with_duration(50.0)
                .with_size(0)
                .with_content_type("text/css"),
        );
        log.add(
            RequestEntry::new("GET", "https://api.example.com/health")
                .with_status(500)
                .with_duration(30.0)
                .with_size(128),
        );
        log
    }

    #[test]
    fn test_entry_creation() {
        let entry = RequestEntry::new("GET", "https://example.com");
        assert_eq!(entry.method, "GET");
        assert_eq!(entry.url, "https://example.com");
        assert!(entry.response_status.is_none());
    }

    #[test]
    fn test_is_success_and_error() {
        let ok = RequestEntry::new("GET", "/").with_status(200);
        assert!(ok.is_success());
        assert!(!ok.is_error());

        let err = RequestEntry::new("GET", "/").with_status(500);
        assert!(!err.is_success());
        assert!(err.is_error());
    }

    #[test]
    fn test_filter_by_method() {
        let log = sample_log();
        let gets = log.filter_by_method("GET");
        assert_eq!(gets.len(), 3);
        let posts = log.filter_by_method("POST");
        assert_eq!(posts.len(), 1);
    }

    #[test]
    fn test_filter_by_status_range() {
        let log = sample_log();
        let success = log.filter_by_status_range(200, 299);
        assert_eq!(success.len(), 2);
        let errors = log.filter_by_status_range(400, 599);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_filter_by_url() {
        let log = sample_log();
        let api = log.filter_by_url("api.example.com");
        assert_eq!(api.len(), 3);
        let cdn = log.filter_by_url("cdn.");
        assert_eq!(cdn.len(), 1);
    }

    #[test]
    fn test_filter_by_content_type() {
        let log = sample_log();
        let json = log.filter_by_content_type("json");
        assert_eq!(json.len(), 1);
    }

    #[test]
    fn test_stats() {
        let log = sample_log();
        let stats = log.stats();
        assert_eq!(stats.total_requests, 4);
        assert_eq!(stats.total_size_bytes, 4736);
        assert!((stats.total_duration_ms - 400.0).abs() < f64::EPSILON);
        assert_eq!(stats.error_count, 2);
        assert!((stats.average_duration_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_waterfall_timing() {
        let wf = WaterfallTiming {
            dns_ms: 10.0,
            connect_ms: 20.0,
            tls_ms: 15.0,
            ttfb_ms: 50.0,
            content_ms: 100.0,
        };
        assert!((wf.total_ms() - 195.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_har_export() {
        let mut log = RequestLog::new();
        log.add(
            RequestEntry::new("GET", "https://example.com")
                .with_status(200)
                .with_waterfall(WaterfallTiming {
                    dns_ms: 5.0,
                    connect_ms: 10.0,
                    tls_ms: 8.0,
                    ttfb_ms: 30.0,
                    content_ms: 50.0,
                }),
        );
        let har = log.to_har_json();
        let entries = har["log"]["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["request"]["method"], "GET");
        assert_eq!(entries[0]["timings"]["dns"], 5.0);
    }

    #[test]
    fn test_clear() {
        let mut log = sample_log();
        assert!(!log.is_empty());
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_request_headers() {
        let entry = RequestEntry::new("GET", "/")
            .with_request_header("Accept", "application/json")
            .with_response_header("Content-Type", "application/json");
        assert_eq!(entry.request_headers.get("Accept").unwrap(), "application/json");
        assert_eq!(
            entry.response_headers.get("Content-Type").unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_initiator() {
        let entry = RequestEntry::new("GET", "/api/data")
            .with_initiator("fetch@app.js:42");
        assert_eq!(entry.initiator.as_deref(), Some("fetch@app.js:42"));
    }
}
