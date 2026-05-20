use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

/// A monotonically increasing counter metric.
///
/// Counters track cumulative values that only go up: request counts,
/// bytes transmitted, errors observed, etc. Thread-safe via `AtomicU64`.
#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    description: String,
    labels: HashMap<String, String>,
    value: Arc<AtomicU64>,
}

/// Snapshot of a counter's state at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct CounterSnapshot {
    pub name: String,
    pub description: String,
    pub labels: HashMap<String, String>,
    pub value: u64,
}

impl Counter {
    /// Create a new counter with the given name and description.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            labels: HashMap::new(),
            value: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a counter with pre-defined labels.
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Add a single label key-value pair.
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Increment the counter by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by the given amount.
    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Read the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero. Primarily for testing.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }

    /// The counter's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The counter's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The counter's labels.
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    /// Take a point-in-time snapshot.
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            name: self.name.clone(),
            description: self.description.clone(),
            labels: self.labels.clone(),
            value: self.get(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_counter_starts_at_zero() {
        let c = Counter::new("requests_total", "Total requests");
        assert_eq!(c.get(), 0);
        assert_eq!(c.name(), "requests_total");
        assert_eq!(c.description(), "Total requests");
    }

    #[test]
    fn inc_increments_by_one() {
        let c = Counter::new("x", "");
        c.inc();
        c.inc();
        c.inc();
        assert_eq!(c.get(), 3);
    }

    #[test]
    fn inc_by_adds_arbitrary_amount() {
        let c = Counter::new("bytes_sent", "");
        c.inc_by(1024);
        c.inc_by(512);
        assert_eq!(c.get(), 1536);
    }

    #[test]
    fn reset_zeros_value() {
        let c = Counter::new("x", "");
        c.inc_by(100);
        c.reset();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn labels_attached_correctly() {
        let c = Counter::new("http_requests", "HTTP requests")
            .with_label("method", "GET")
            .with_label("status", "200");
        assert_eq!(c.labels().get("method").unwrap(), "GET");
        assert_eq!(c.labels().get("status").unwrap(), "200");
    }

    #[test]
    fn with_labels_replaces_all() {
        let mut labels = HashMap::new();
        labels.insert("env".into(), "prod".into());
        let c = Counter::new("x", "").with_labels(labels);
        assert_eq!(c.labels().len(), 1);
        assert_eq!(c.labels().get("env").unwrap(), "prod");
    }

    #[test]
    fn snapshot_captures_current_state() {
        let c = Counter::new("ops", "operations").with_label("region", "us-east");
        c.inc_by(42);
        let snap = c.snapshot();
        assert_eq!(snap.name, "ops");
        assert_eq!(snap.value, 42);
        assert_eq!(snap.labels.get("region").unwrap(), "us-east");
    }

    #[test]
    fn clone_shares_atomic_value() {
        let c1 = Counter::new("shared", "");
        let c2 = c1.clone();
        c1.inc_by(10);
        assert_eq!(c2.get(), 10);
        c2.inc_by(5);
        assert_eq!(c1.get(), 15);
    }

    #[test]
    fn concurrent_increments() {
        let c = Counter::new("concurrent", "");
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let counter = c.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        counter.inc();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(c.get(), 10_000);
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let c = Counter::new("test", "desc");
        c.inc_by(7);
        let snap = c.snapshot();
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["name"], "test");
        assert_eq!(json["value"], 7);
    }
}
