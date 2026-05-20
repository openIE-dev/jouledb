//! Performance profiler with events, aggregation, Chrome Trace export, and hot-path detection.
//!
//! Records `ProfileEvent` spans (with optional nesting via `parent_id`), computes
//! aggregate statistics (count, total, mean, min, max, p50, p95, p99), and
//! exports to Chrome Trace Event JSON format.

use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// A single profiling event (span).
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub id: String,
    pub name: String,
    pub category: String,
    pub start_us: u64,
    pub duration_us: u64,
    pub parent_id: Option<String>,
    pub args: HashMap<String, String>,
}

impl ProfileEvent {
    pub fn new(name: &str, category: &str, start_us: u64, duration_us: u64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            category: category.to_string(),
            start_us,
            duration_us,
            parent_id: None,
            args: HashMap::new(),
        }
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    pub fn with_arg(mut self, key: &str, value: &str) -> Self {
        self.args.insert(key.to_string(), value.to_string());
        self
    }

    pub fn end_us(&self) -> u64 {
        self.start_us + self.duration_us
    }
}

/// Aggregate statistics for events sharing a name.
#[derive(Debug, Clone)]
pub struct AggregateStats {
    pub name: String,
    pub count: usize,
    pub total_us: u64,
    pub mean_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

/// A node in the hot-path tree.
#[derive(Debug, Clone)]
pub struct HotPathNode {
    pub name: String,
    pub self_time_us: u64,
    pub total_time_us: u64,
    pub children: Vec<HotPathNode>,
}

// ── ProfileSession ──

/// A profiling session that records events and computes statistics.
pub struct ProfileSession {
    pub events: Vec<ProfileEvent>,
    recording: bool,
}

impl ProfileSession {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            recording: false,
        }
    }

    pub fn start_recording(&mut self) {
        self.recording = true;
        self.events.clear();
    }

    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Record an event. Only accepted while recording.
    pub fn record(&mut self, event: ProfileEvent) {
        if self.recording {
            self.events.push(event);
        }
    }

    /// Compute aggregate stats per event name.
    pub fn aggregate(&self) -> Vec<AggregateStats> {
        let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
        for e in &self.events {
            groups
                .entry(e.name.clone())
                .or_default()
                .push(e.duration_us);
        }

        let mut result: Vec<AggregateStats> = groups
            .into_iter()
            .map(|(name, mut durations)| {
                durations.sort_unstable();
                let count = durations.len();
                let total: u64 = durations.iter().sum();
                let mean = total as f64 / count as f64;
                let min = durations[0];
                let max = durations[count - 1];
                let p50 = percentile(&durations, 50.0);
                let p95 = percentile(&durations, 95.0);
                let p99 = percentile(&durations, 99.0);
                AggregateStats {
                    name,
                    count,
                    total_us: total,
                    mean_us: mean,
                    min_us: min,
                    max_us: max,
                    p50_us: p50,
                    p95_us: p95,
                    p99_us: p99,
                }
            })
            .collect();

        result.sort_by(|a, b| b.total_us.cmp(&a.total_us));
        result
    }

    /// Compute self-time for an event (total_time minus time of direct children).
    pub fn self_time_us(&self, event_id: &str) -> u64 {
        let event = match self.events.iter().find(|e| e.id == event_id) {
            Some(e) => e,
            None => return 0,
        };

        let children_time: u64 = self
            .events
            .iter()
            .filter(|e| e.parent_id.as_deref() == Some(event_id))
            .map(|e| e.duration_us)
            .sum();

        event.duration_us.saturating_sub(children_time)
    }

    /// Export events to Chrome Trace Event format JSON.
    pub fn to_chrome_trace_json(&self) -> Value {
        let trace_events: Vec<Value> = self
            .events
            .iter()
            .map(|e| {
                let mut obj = json!({
                    "name": e.name,
                    "cat": e.category,
                    "ph": "X",
                    "ts": e.start_us,
                    "dur": e.duration_us,
                    "pid": 1,
                    "tid": 1,
                });
                if !e.args.is_empty() {
                    let args_val: serde_json::Map<String, Value> = e
                        .args
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                        .collect();
                    obj["args"] = Value::Object(args_val);
                }
                obj
            })
            .collect();

        json!({ "traceEvents": trace_events })
    }

    /// Detect the hot path: the chain of nested events with the longest cumulative time.
    pub fn hot_path(&self) -> Vec<String> {
        // Build tree of root events, walk to find longest path
        let roots: Vec<&ProfileEvent> = self
            .events
            .iter()
            .filter(|e| e.parent_id.is_none())
            .collect();

        let mut best_path: Vec<String> = Vec::new();
        let mut best_time: u64 = 0;

        for root in &roots {
            let mut path = Vec::new();
            self.walk_hot_path(&root.id, &mut path, &mut best_path, &mut best_time);
        }

        best_path
    }

    fn walk_hot_path(
        &self,
        event_id: &str,
        current_path: &mut Vec<String>,
        best_path: &mut Vec<String>,
        best_time: &mut u64,
    ) {
        let event = match self.events.iter().find(|e| e.id == event_id) {
            Some(e) => e,
            None => return,
        };

        current_path.push(event.name.clone());

        let children: Vec<&ProfileEvent> = self
            .events
            .iter()
            .filter(|e| e.parent_id.as_deref() == Some(event_id))
            .collect();

        if children.is_empty() {
            // Leaf: compute total path time
            let total: u64 = current_path
                .iter()
                .filter_map(|name| {
                    self.events
                        .iter()
                        .find(|e| e.name == *name)
                        .map(|e| e.duration_us)
                })
                .sum();
            if total > *best_time {
                *best_time = total;
                *best_path = current_path.clone();
            }
        } else {
            // Recurse into the heaviest child
            let heaviest = children
                .iter()
                .max_by_key(|c| c.duration_us)
                .unwrap();
            self.walk_hot_path(&heaviest.id, current_path, best_path, best_time);
        }

        current_path.pop();
    }
}

impl Default for ProfileSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a percentile from a sorted slice.
fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((pct / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let e = ProfileEvent::new("render", "ui", 1000, 500);
        assert_eq!(e.name, "render");
        assert_eq!(e.end_us(), 1500);
    }

    #[test]
    fn test_event_with_parent() {
        let parent = ProfileEvent::new("render", "ui", 0, 1000);
        let child = ProfileEvent::new("layout", "ui", 100, 200).with_parent(&parent.id);
        assert_eq!(child.parent_id.as_deref(), Some(parent.id.as_str()));
    }

    #[test]
    fn test_recording_gate() {
        let mut sess = ProfileSession::new();
        sess.record(ProfileEvent::new("a", "x", 0, 100));
        assert_eq!(sess.events.len(), 0); // not recording

        sess.start_recording();
        sess.record(ProfileEvent::new("b", "x", 0, 100));
        assert_eq!(sess.events.len(), 1);

        sess.stop_recording();
        sess.record(ProfileEvent::new("c", "x", 0, 100));
        assert_eq!(sess.events.len(), 1);
    }

    #[test]
    fn test_aggregate_stats() {
        let mut sess = ProfileSession::new();
        sess.start_recording();
        for i in 0..10 {
            sess.record(ProfileEvent::new("op", "cat", i * 100, (i + 1) * 10));
        }
        let stats = sess.aggregate();
        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.count, 10);
        assert_eq!(s.min_us, 10);
        assert_eq!(s.max_us, 100);
    }

    #[test]
    fn test_percentile_computation() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&data, 50.0), 6); // idx = round(0.5 * 9) = 5 -> data[5] = 6
        assert_eq!(percentile(&data, 0.0), 1);
        assert_eq!(percentile(&data, 100.0), 10);
    }

    #[test]
    fn test_self_time() {
        let mut sess = ProfileSession::new();
        sess.start_recording();
        let parent = ProfileEvent::new("parent", "x", 0, 1000);
        let pid = parent.id.clone();
        let child = ProfileEvent::new("child", "x", 100, 300).with_parent(&pid);
        sess.record(parent);
        sess.record(child);

        assert_eq!(sess.self_time_us(&pid), 700);
    }

    #[test]
    fn test_chrome_trace_export() {
        let mut sess = ProfileSession::new();
        sess.start_recording();
        sess.record(
            ProfileEvent::new("paint", "render", 0, 500).with_arg("layer", "main"),
        );
        let trace = sess.to_chrome_trace_json();
        let events = trace["traceEvents"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["name"], "paint");
        assert_eq!(events[0]["ph"], "X");
        assert_eq!(events[0]["ts"], 0);
        assert_eq!(events[0]["dur"], 500);
        assert_eq!(events[0]["args"]["layer"], "main");
    }

    #[test]
    fn test_hot_path() {
        let mut sess = ProfileSession::new();
        sess.start_recording();
        let root = ProfileEvent::new("main", "x", 0, 5000);
        let rid = root.id.clone();
        let fast_child = ProfileEvent::new("fast", "x", 0, 100).with_parent(&rid);
        let slow_child = ProfileEvent::new("slow", "x", 100, 4000).with_parent(&rid);
        sess.record(root);
        sess.record(fast_child);
        sess.record(slow_child);

        let path = sess.hot_path();
        assert_eq!(path, vec!["main", "slow"]);
    }

    #[test]
    fn test_multiple_aggregates() {
        let mut sess = ProfileSession::new();
        sess.start_recording();
        sess.record(ProfileEvent::new("alpha", "x", 0, 100));
        sess.record(ProfileEvent::new("beta", "x", 100, 200));
        sess.record(ProfileEvent::new("alpha", "x", 300, 150));
        let stats = sess.aggregate();
        assert_eq!(stats.len(), 2);
        // Sorted by total descending: alpha=250, beta=200
        let alpha = stats.iter().find(|s| s.name == "alpha").unwrap();
        assert_eq!(alpha.count, 2);
        assert_eq!(alpha.total_us, 250);
    }

    #[test]
    fn test_event_args() {
        let e = ProfileEvent::new("db", "io", 0, 50)
            .with_arg("query", "SELECT *")
            .with_arg("rows", "42");
        assert_eq!(e.args.len(), 2);
        assert_eq!(e.args.get("query").unwrap(), "SELECT *");
    }

    #[test]
    fn test_start_clears_events() {
        let mut sess = ProfileSession::new();
        sess.start_recording();
        sess.record(ProfileEvent::new("a", "x", 0, 1));
        assert_eq!(sess.events.len(), 1);
        sess.start_recording();
        assert_eq!(sess.events.len(), 0);
    }

    #[test]
    fn test_empty_aggregate() {
        let sess = ProfileSession::new();
        assert!(sess.aggregate().is_empty());
    }
}
