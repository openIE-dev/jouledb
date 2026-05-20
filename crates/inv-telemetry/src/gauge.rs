use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use serde::Serialize;

/// A gauge metric that can go up and down.
///
/// Gauges represent a single numerical value that can increase or decrease:
/// CPU utilization, memory usage, temperature, active connections, etc.
///
/// Internally stores values as `i64` with milli-precision (value × 1000)
/// to avoid floating-point atomics while supporting fractional values.
#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    description: String,
    labels: HashMap<String, String>,
    /// Stored as value * 1000 for milli-precision without AtomicF64.
    value_millis: Arc<AtomicI64>,
}

/// Snapshot of a gauge's state at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct GaugeSnapshot {
    pub name: String,
    pub description: String,
    pub labels: HashMap<String, String>,
    pub value: f64,
}

impl Gauge {
    /// Create a new gauge with the given name and description.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            labels: HashMap::new(),
            value_millis: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Create a gauge with pre-defined labels.
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Add a single label key-value pair.
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Set the gauge to an exact floating-point value.
    pub fn set(&self, value: f64) {
        let millis = (value * 1000.0) as i64;
        self.value_millis.store(millis, Ordering::Relaxed);
    }

    /// Set the gauge to an exact integer value.
    pub fn set_i64(&self, value: i64) {
        self.value_millis.store(value * 1000, Ordering::Relaxed);
    }

    /// Increment the gauge by the given floating-point amount.
    pub fn inc(&self, delta: f64) {
        let millis = (delta * 1000.0) as i64;
        self.value_millis.fetch_add(millis, Ordering::Relaxed);
    }

    /// Decrement the gauge by the given floating-point amount.
    pub fn dec(&self, delta: f64) {
        let millis = (delta * 1000.0) as i64;
        self.value_millis.fetch_sub(millis, Ordering::Relaxed);
    }

    /// Increment the gauge by 1.
    pub fn inc_one(&self) {
        self.value_millis.fetch_add(1000, Ordering::Relaxed);
    }

    /// Decrement the gauge by 1.
    pub fn dec_one(&self) {
        self.value_millis.fetch_sub(1000, Ordering::Relaxed);
    }

    /// Read the current value as a float.
    pub fn get(&self) -> f64 {
        self.value_millis.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Read the raw milli-precision integer value.
    pub fn get_raw(&self) -> i64 {
        self.value_millis.load(Ordering::Relaxed)
    }

    /// Reset the gauge to zero.
    pub fn reset(&self) {
        self.value_millis.store(0, Ordering::Relaxed);
    }

    /// The gauge's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The gauge's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The gauge's labels.
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    /// Take a point-in-time snapshot.
    pub fn snapshot(&self) -> GaugeSnapshot {
        GaugeSnapshot {
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
    fn new_gauge_starts_at_zero() {
        let g = Gauge::new("cpu_usage", "CPU utilization");
        assert_eq!(g.get(), 0.0);
        assert_eq!(g.name(), "cpu_usage");
    }

    #[test]
    fn set_and_get_float() {
        let g = Gauge::new("temp", "Temperature");
        g.set(72.5);
        assert!((g.get() - 72.5).abs() < 0.01);
    }

    #[test]
    fn set_i64_and_get() {
        let g = Gauge::new("connections", "Active connections");
        g.set_i64(42);
        assert!((g.get() - 42.0).abs() < 0.01);
    }

    #[test]
    fn inc_and_dec_float() {
        let g = Gauge::new("x", "");
        g.set(10.0);
        g.inc(2.5);
        assert!((g.get() - 12.5).abs() < 0.01);
        g.dec(3.0);
        assert!((g.get() - 9.5).abs() < 0.01);
    }

    #[test]
    fn inc_one_dec_one() {
        let g = Gauge::new("x", "");
        g.inc_one();
        g.inc_one();
        g.inc_one();
        assert!((g.get() - 3.0).abs() < 0.01);
        g.dec_one();
        assert!((g.get() - 2.0).abs() < 0.01);
    }

    #[test]
    fn negative_values() {
        let g = Gauge::new("x", "");
        g.set(-5.5);
        assert!((g.get() - (-5.5)).abs() < 0.01);
    }

    #[test]
    fn reset_zeros_gauge() {
        let g = Gauge::new("x", "");
        g.set(100.0);
        g.reset();
        assert_eq!(g.get(), 0.0);
    }

    #[test]
    fn labels_attached_correctly() {
        let g = Gauge::new("mem_bytes", "Memory")
            .with_label("node", "node-1")
            .with_label("region", "us-east");
        assert_eq!(g.labels().get("node").unwrap(), "node-1");
        assert_eq!(g.labels().len(), 2);
    }

    #[test]
    fn clone_shares_atomic() {
        let g1 = Gauge::new("shared", "");
        let g2 = g1.clone();
        g1.set(42.0);
        assert!((g2.get() - 42.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_captures_state() {
        let g = Gauge::new("watts", "Power draw").with_label("source", "pdu");
        g.set(120.5);
        let snap = g.snapshot();
        assert_eq!(snap.name, "watts");
        assert!((snap.value - 120.5).abs() < 0.01);
        assert_eq!(snap.labels.get("source").unwrap(), "pdu");
    }

    #[test]
    fn concurrent_updates() {
        let g = Gauge::new("concurrent", "");
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let gauge = g.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        gauge.inc_one();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert!((g.get() - 10_000.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let g = Gauge::new("test", "desc");
        g.set(2.72);
        let snap = g.snapshot();
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["name"], "test");
        assert!((json["value"].as_f64().unwrap() - 2.72).abs() < 0.01);
    }
}
