use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

/// Status of a completed span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpanStatus {
    Ok,
    Error(String),
}

/// A completed span ready for collection/export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishedSpan {
    /// Hex-encoded trace ID.
    pub trace_id: String,
    /// Hex-encoded span ID.
    pub span_id: String,
    /// Hex-encoded parent span ID.
    pub parent_span_id: Option<String>,
    /// Human-readable span name, e.g., "HTTP GET /api/v1/nodes".
    pub name: String,
    /// Originating service name, e.g., "invisible-api".
    pub service: String,
    /// Span start time in microseconds since the Unix epoch.
    pub start_epoch_us: u64,
    /// Span duration in microseconds.
    pub duration_us: u64,
    /// Terminal status of the span.
    pub status: SpanStatus,
    /// Arbitrary key-value attributes attached to the span.
    pub attributes: HashMap<String, String>,
}

/// Query parameters for filtering spans.
#[derive(Debug, Clone, Default)]
pub struct SpanQuery {
    pub trace_id: Option<String>,
    pub service: Option<String>,
    pub min_duration_us: Option<u64>,
    pub since_epoch_us: Option<u64>,
    pub limit: Option<usize>,
}

/// A bounded, thread-safe collector for finished spans.
///
/// Spans are stored in a ring buffer. When full, oldest spans are evicted.
pub struct SpanCollector {
    spans: RwLock<Vec<FinishedSpan>>,
    max_spans: usize,
}

impl SpanCollector {
    /// Create a new collector with the given maximum capacity.
    pub fn new(max_spans: usize) -> Self {
        Self {
            spans: RwLock::new(Vec::with_capacity(max_spans)),
            max_spans,
        }
    }

    /// Record a finished span. If the collector is at capacity the oldest span
    /// is evicted first.
    pub fn record(&self, span: FinishedSpan) {
        let mut spans = self.spans.write().unwrap();
        if spans.len() >= self.max_spans {
            spans.remove(0);
        }
        spans.push(span);
    }

    /// Drain all spans from the collector, returning them and leaving the
    /// buffer empty.
    pub fn drain(&self) -> Vec<FinishedSpan> {
        let mut spans = self.spans.write().unwrap();
        std::mem::take(&mut *spans)
    }

    /// Return the most recent `limit` spans, ordered oldest-first (newest
    /// last).
    pub fn recent(&self, limit: usize) -> Vec<FinishedSpan> {
        let spans = self.spans.read().unwrap();
        let start = spans.len().saturating_sub(limit);
        spans[start..].to_vec()
    }

    /// Filter spans according to the supplied query parameters.
    ///
    /// All non-`None` fields act as conjunctive (AND) filters. The `limit`
    /// field is applied last, after all other filters.
    pub fn query(&self, q: &SpanQuery) -> Vec<FinishedSpan> {
        let spans = self.spans.read().unwrap();
        let mut results: Vec<FinishedSpan> = spans
            .iter()
            .filter(|s| {
                if let Some(ref tid) = q.trace_id
                    && s.trace_id != *tid
                {
                    return false;
                }
                if let Some(ref svc) = q.service
                    && s.service != *svc
                {
                    return false;
                }
                if let Some(min_dur) = q.min_duration_us
                    && s.duration_us < min_dur
                {
                    return false;
                }
                if let Some(since) = q.since_epoch_us
                    && s.start_epoch_us < since
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        if let Some(limit) = q.limit {
            results.truncate(limit);
        }

        results
    }

    /// Return the number of spans currently stored.
    pub fn count(&self) -> usize {
        self.spans.read().unwrap().len()
    }

    /// Remove all stored spans.
    pub fn clear(&self) {
        self.spans.write().unwrap().clear();
    }

    /// Return the maximum number of spans this collector can hold.
    pub fn capacity(&self) -> usize {
        self.max_spans
    }
}

impl Default for SpanCollector {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_span(trace_id: &str, service: &str, duration_us: u64) -> FinishedSpan {
        FinishedSpan {
            trace_id: trace_id.to_string(),
            span_id: format!("span-{}", rand::random::<u32>()),
            parent_span_id: None,
            name: "test-span".to_string(),
            service: service.to_string(),
            start_epoch_us: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            duration_us,
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn new_collector_is_empty() {
        let collector = SpanCollector::new(100);
        assert_eq!(collector.count(), 0);
        assert_eq!(collector.capacity(), 100);
    }

    #[test]
    fn record_and_count() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("trace-1", "svc-a", 500));
        assert_eq!(collector.count(), 1);
        collector.record(test_span("trace-2", "svc-b", 600));
        assert_eq!(collector.count(), 2);
    }

    #[test]
    fn bounded_eviction() {
        let collector = SpanCollector::new(3);
        for i in 0..5 {
            collector.record(test_span(&format!("trace-{i}"), "svc", 100));
        }
        assert_eq!(collector.count(), 3);

        // The first two spans (trace-0, trace-1) should have been evicted.
        let all = collector.drain();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].trace_id, "trace-2");
        assert_eq!(all[1].trace_id, "trace-3");
        assert_eq!(all[2].trace_id, "trace-4");
    }

    #[test]
    fn drain_returns_all_and_clears() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("t1", "svc", 100));
        collector.record(test_span("t2", "svc", 200));
        let drained = collector.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn recent_returns_newest() {
        let collector = SpanCollector::new(10);
        for i in 0..5 {
            collector.record(test_span(&format!("trace-{i}"), "svc", 100));
        }
        let recent = collector.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].trace_id, "trace-3");
        assert_eq!(recent[1].trace_id, "trace-4");
    }

    #[test]
    fn recent_with_limit_larger_than_count() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("t1", "svc", 100));
        collector.record(test_span("t2", "svc", 200));
        let recent = collector.recent(50);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn query_by_trace_id() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("abc", "svc", 100));
        collector.record(test_span("def", "svc", 200));
        collector.record(test_span("abc", "svc", 300));

        let results = collector.query(&SpanQuery {
            trace_id: Some("abc".to_string()),
            ..Default::default()
        });
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|s| s.trace_id == "abc"));
    }

    #[test]
    fn query_by_service() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("t1", "api", 100));
        collector.record(test_span("t2", "scheduler", 200));
        collector.record(test_span("t3", "api", 300));

        let results = collector.query(&SpanQuery {
            service: Some("api".to_string()),
            ..Default::default()
        });
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|s| s.service == "api"));
    }

    #[test]
    fn query_by_min_duration() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("t1", "svc", 50));
        collector.record(test_span("t2", "svc", 150));
        collector.record(test_span("t3", "svc", 300));

        let results = collector.query(&SpanQuery {
            min_duration_us: Some(100),
            ..Default::default()
        });
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|s| s.duration_us >= 100));
    }

    #[test]
    fn query_by_since() {
        let collector = SpanCollector::new(10);

        let mut old_span = test_span("t1", "svc", 100);
        old_span.start_epoch_us = 1_000_000;

        let mut new_span = test_span("t2", "svc", 200);
        new_span.start_epoch_us = 5_000_000;

        collector.record(old_span);
        collector.record(new_span);

        let results = collector.query(&SpanQuery {
            since_epoch_us: Some(3_000_000),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].trace_id, "t2");
    }

    #[test]
    fn query_with_limit() {
        let collector = SpanCollector::new(10);
        for i in 0..5 {
            collector.record(test_span(&format!("t{i}"), "svc", 100));
        }

        let results = collector.query(&SpanQuery {
            limit: Some(3),
            ..Default::default()
        });
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn query_combining_filters() {
        let collector = SpanCollector::new(20);
        collector.record(test_span("abc", "api", 500));
        collector.record(test_span("abc", "api", 50));
        collector.record(test_span("abc", "scheduler", 500));
        collector.record(test_span("def", "api", 500));

        let results = collector.query(&SpanQuery {
            trace_id: Some("abc".to_string()),
            service: Some("api".to_string()),
            min_duration_us: Some(100),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].trace_id, "abc");
        assert_eq!(results[0].service, "api");
        assert!(results[0].duration_us >= 100);
    }

    #[test]
    fn query_empty_collector() {
        let collector = SpanCollector::new(10);
        let results = collector.query(&SpanQuery {
            trace_id: Some("nope".to_string()),
            ..Default::default()
        });
        assert!(results.is_empty());
    }

    #[test]
    fn clear_empties_collector() {
        let collector = SpanCollector::new(10);
        collector.record(test_span("t1", "svc", 100));
        collector.record(test_span("t2", "svc", 200));
        assert_eq!(collector.count(), 2);
        collector.clear();
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn span_status_variants() {
        let ok = SpanStatus::Ok;
        let err = SpanStatus::Error("timeout".to_string());

        assert_eq!(ok, SpanStatus::Ok);
        assert_eq!(err, SpanStatus::Error("timeout".to_string()));
        assert_ne!(ok, err);
    }

    #[test]
    fn finished_span_serialization() {
        let span = test_span("trace-ser", "svc-ser", 42);
        let json = serde_json::to_string(&span).expect("serialize");
        let deserialized: FinishedSpan = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.trace_id, "trace-ser");
        assert_eq!(deserialized.service, "svc-ser");
        assert_eq!(deserialized.duration_us, 42);
        assert_eq!(deserialized.status, SpanStatus::Ok);
    }

    #[test]
    fn default_collector() {
        let collector = SpanCollector::default();
        assert_eq!(collector.capacity(), 10_000);
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn capacity_returns_max() {
        let collector = SpanCollector::new(512);
        assert_eq!(collector.capacity(), 512);
    }
}
