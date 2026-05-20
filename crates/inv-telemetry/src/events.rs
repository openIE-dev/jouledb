use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Classification of platform events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    WorkloadDeployed,
    WorkloadDestroyed,
    WorkloadMigrated,
    WorkloadScaled,
    NodeJoined,
    NodeLeft,
    NodeHealthChanged,
    EnergyBudgetExceeded,
    CarbonShiftTriggered,
    SchedulingDecision,
    AutoScaleAction,
    SecurityAlert,
    PolicyViolation,
    SecretRotated,
    RateLimitExceeded,
    AuditEvent,
    BackpressureTriggered,
    CrdtSyncCompleted,
    Custom,
}

/// Severity level for events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// A structured platform event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub kind: EventKind,
    pub severity: Severity,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub node_id: Option<String>,
    pub workload_id: Option<String>,
    pub message: String,
    pub metadata: HashMap<String, String>,
}

impl Event {
    /// Create a new event with the given kind, severity, source, and message.
    pub fn new(kind: EventKind, severity: Severity, source: &str, message: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kind,
            severity,
            timestamp: Utc::now(),
            source: source.to_string(),
            node_id: None,
            workload_id: None,
            message: message.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a node ID to the event.
    pub fn with_node(mut self, node_id: &str) -> Self {
        self.node_id = Some(node_id.to_string());
        self
    }

    /// Attach a workload ID to the event.
    pub fn with_workload(mut self, workload_id: &str) -> Self {
        self.workload_id = Some(workload_id.to_string());
        self
    }

    /// Add a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// A bounded, thread-safe stream of structured events.
///
/// Events are stored in a ring buffer with a configurable maximum size.
/// When the buffer is full, the oldest events are evicted.
#[derive(Debug, Clone)]
pub struct EventStream {
    events: Arc<RwLock<Vec<Event>>>,
    max_events: usize,
}

impl EventStream {
    /// Create a new event stream with the given capacity.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::with_capacity(max_events))),
            max_events,
        }
    }

    /// Emit (record) an event into the stream.
    pub fn emit(&self, event: Event) {
        let mut events = self.events.write().unwrap();
        if events.len() >= self.max_events {
            events.remove(0);
        }
        events.push(event);
    }

    /// Retrieve the most recent `limit` events (newest last).
    pub fn recent(&self, limit: usize) -> Vec<Event> {
        let events = self.events.read().unwrap();
        let start = events.len().saturating_sub(limit);
        events[start..].to_vec()
    }

    /// Retrieve all events.
    pub fn all(&self) -> Vec<Event> {
        self.events.read().unwrap().clone()
    }

    /// Filter events by kind.
    pub fn filter_by_kind(&self, kind: EventKind) -> Vec<Event> {
        self.events
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.kind == kind)
            .cloned()
            .collect()
    }

    /// Filter events by severity (at or above the given level).
    pub fn filter_by_severity(&self, min_severity: Severity) -> Vec<Event> {
        self.events
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.severity >= min_severity)
            .cloned()
            .collect()
    }

    /// Filter events by source.
    pub fn filter_by_source(&self, source: &str) -> Vec<Event> {
        self.events
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.source == source)
            .cloned()
            .collect()
    }

    /// Total number of events currently in the stream.
    pub fn count(&self) -> usize {
        self.events.read().unwrap().len()
    }

    /// Clear all events from the stream.
    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }

    /// Maximum capacity of the stream.
    pub fn capacity(&self) -> usize {
        self.max_events
    }
}

impl Default for EventStream {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_event(kind: EventKind, severity: Severity, source: &str) -> Event {
        Event::new(kind, severity, source, "test message")
    }

    #[test]
    fn event_creation() {
        let e = Event::new(
            EventKind::WorkloadDeployed,
            Severity::Info,
            "scheduler",
            "Deployed workload web-1",
        );
        assert_eq!(e.kind, EventKind::WorkloadDeployed);
        assert_eq!(e.severity, Severity::Info);
        assert_eq!(e.source, "scheduler");
        assert!(!e.id.is_empty());
    }

    #[test]
    fn event_builder_methods() {
        let e = Event::new(EventKind::NodeJoined, Severity::Info, "mesh", "Node joined")
            .with_node("node-1")
            .with_workload("wl-1")
            .with_metadata("region", "us-east");

        assert_eq!(e.node_id.as_deref(), Some("node-1"));
        assert_eq!(e.workload_id.as_deref(), Some("wl-1"));
        assert_eq!(e.metadata.get("region").unwrap(), "us-east");
    }

    #[test]
    fn emit_and_retrieve() {
        let stream = EventStream::new(100);
        stream.emit(test_event(
            EventKind::WorkloadDeployed,
            Severity::Info,
            "sched",
        ));
        stream.emit(test_event(EventKind::NodeJoined, Severity::Info, "mesh"));
        assert_eq!(stream.count(), 2);
    }

    #[test]
    fn recent_returns_latest() {
        let stream = EventStream::new(100);
        for i in 0..10 {
            stream.emit(Event::new(
                EventKind::Custom,
                Severity::Info,
                "test",
                &format!("msg-{i}"),
            ));
        }
        let recent = stream.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].message, "msg-7");
        assert_eq!(recent[2].message, "msg-9");
    }

    #[test]
    fn eviction_when_full() {
        let stream = EventStream::new(3);
        for i in 0..5 {
            stream.emit(Event::new(
                EventKind::Custom,
                Severity::Info,
                "test",
                &format!("msg-{i}"),
            ));
        }
        assert_eq!(stream.count(), 3);
        let all = stream.all();
        assert_eq!(all[0].message, "msg-2");
        assert_eq!(all[2].message, "msg-4");
    }

    #[test]
    fn filter_by_kind() {
        let stream = EventStream::new(100);
        stream.emit(test_event(EventKind::NodeJoined, Severity::Info, "mesh"));
        stream.emit(test_event(
            EventKind::WorkloadDeployed,
            Severity::Info,
            "sched",
        ));
        stream.emit(test_event(EventKind::NodeJoined, Severity::Info, "mesh"));

        let joins = stream.filter_by_kind(EventKind::NodeJoined);
        assert_eq!(joins.len(), 2);
    }

    #[test]
    fn filter_by_severity() {
        let stream = EventStream::new(100);
        stream.emit(test_event(EventKind::Custom, Severity::Debug, "a"));
        stream.emit(test_event(EventKind::Custom, Severity::Info, "b"));
        stream.emit(test_event(EventKind::Custom, Severity::Warning, "c"));
        stream.emit(test_event(EventKind::Custom, Severity::Error, "d"));
        stream.emit(test_event(EventKind::Custom, Severity::Critical, "e"));

        let warnings_up = stream.filter_by_severity(Severity::Warning);
        assert_eq!(warnings_up.len(), 3); // Warning, Error, Critical
    }

    #[test]
    fn filter_by_source() {
        let stream = EventStream::new(100);
        stream.emit(test_event(EventKind::Custom, Severity::Info, "scheduler"));
        stream.emit(test_event(EventKind::Custom, Severity::Info, "runtime"));
        stream.emit(test_event(EventKind::Custom, Severity::Info, "scheduler"));

        let sched = stream.filter_by_source("scheduler");
        assert_eq!(sched.len(), 2);
    }

    #[test]
    fn clear_empties_stream() {
        let stream = EventStream::new(100);
        stream.emit(test_event(EventKind::Custom, Severity::Info, "a"));
        stream.emit(test_event(EventKind::Custom, Severity::Info, "b"));
        stream.clear();
        assert_eq!(stream.count(), 0);
    }

    #[test]
    fn default_capacity() {
        let stream = EventStream::default();
        assert_eq!(stream.capacity(), 10_000);
    }

    #[test]
    fn event_serializes_to_json() {
        let e = Event::new(
            EventKind::EnergyBudgetExceeded,
            Severity::Warning,
            "energy",
            "Budget exceeded",
        )
        .with_node("node-1");

        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["kind"], "energy_budget_exceeded");
        assert_eq!(json["severity"], "warning");
        assert_eq!(json["node_id"], "node-1");
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Debug < Severity::Info);
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }
}
