//! CQRS / Event Sourcing — command handling, event generation, append-only
//! event store, projection building, aggregate roots, snapshots, replay,
//! and version tracking.
//!
//! Replaces JS event-sourcing libraries (EventStoreDB client, Axon Framework)
//! with a pure-Rust CQRS engine that tracks every domain event.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// CQRS domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CqrsError {
    /// Aggregate not found.
    AggregateNotFound(String),
    /// Version conflict (optimistic concurrency).
    VersionConflict { aggregate_id: String, expected: u64, actual: u64 },
    /// Command rejected by aggregate.
    CommandRejected { aggregate_id: String, reason: String },
    /// Event store error.
    StoreError(String),
    /// Snapshot not found.
    SnapshotNotFound(String),
    /// Stream not found.
    StreamNotFound(String),
    /// Invalid command.
    InvalidCommand(String),
}

impl std::fmt::Display for CqrsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AggregateNotFound(id) => write!(f, "aggregate not found: {id}"),
            Self::VersionConflict { aggregate_id, expected, actual } => {
                write!(f, "version conflict for {aggregate_id}: expected {expected}, got {actual}")
            }
            Self::CommandRejected { aggregate_id, reason } => {
                write!(f, "command rejected for {aggregate_id}: {reason}")
            }
            Self::StoreError(msg) => write!(f, "store error: {msg}"),
            Self::SnapshotNotFound(id) => write!(f, "snapshot not found: {id}"),
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
            Self::InvalidCommand(msg) => write!(f, "invalid command: {msg}"),
        }
    }
}

impl std::error::Error for CqrsError {}

// ── Event Envelope ──────────────────────────────────────────────

/// An event envelope wrapping domain event data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub event_id: String,
    pub stream_id: String,
    pub event_type: String,
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

impl EventEnvelope {
    pub fn new(
        stream_id: impl Into<String>,
        event_type: impl Into<String>,
        version: u64,
        data: HashMap<String, String>,
    ) -> Self {
        Self {
            event_id: format!("{}-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0), version),
            stream_id: stream_id.into(),
            event_type: event_type.into(),
            version,
            timestamp: Utc::now(),
            data,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ── Command ─────────────────────────────────────────────────────

/// A command targeting an aggregate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Command {
    pub command_id: String,
    pub aggregate_id: String,
    pub command_type: String,
    pub payload: HashMap<String, String>,
    pub expected_version: Option<u64>,
    pub timestamp: DateTime<Utc>,
}

impl Command {
    pub fn new(
        aggregate_id: impl Into<String>,
        command_type: impl Into<String>,
        payload: HashMap<String, String>,
    ) -> Self {
        let ts = Utc::now();
        Self {
            command_id: format!("cmd-{}", ts.timestamp_nanos_opt().unwrap_or(0)),
            aggregate_id: aggregate_id.into(),
            command_type: command_type.into(),
            payload,
            expected_version: None,
            timestamp: ts,
        }
    }

    pub fn with_expected_version(mut self, v: u64) -> Self {
        self.expected_version = Some(v);
        self
    }
}

// ── Snapshot ────────────────────────────────────────────────────

/// A snapshot of aggregate state at a specific version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub aggregate_id: String,
    pub version: u64,
    pub state: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

// ── Event Store ─────────────────────────────────────────────────

/// Append-only event store.
#[derive(Debug, Default, Clone)]
pub struct EventStore {
    /// Events indexed by stream_id.
    streams: HashMap<String, Vec<EventEnvelope>>,
    /// Global event log (all events in order).
    global_log: Vec<EventEnvelope>,
    /// Snapshots indexed by aggregate_id.
    snapshots: HashMap<String, Vec<Snapshot>>,
}

impl EventStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append events to a stream with optimistic concurrency.
    pub fn append(
        &mut self,
        stream_id: &str,
        expected_version: Option<u64>,
        events: Vec<EventEnvelope>,
    ) -> Result<u64, CqrsError> {
        let stream = self.streams.entry(stream_id.to_string()).or_default();
        let current_version = stream.last().map_or(0, |e| e.version);

        if let Some(expected) = expected_version {
            if expected != current_version {
                return Err(CqrsError::VersionConflict {
                    aggregate_id: stream_id.to_string(),
                    expected,
                    actual: current_version,
                });
            }
        }

        let mut new_version = current_version;
        for mut event in events {
            new_version += 1;
            event.version = new_version;
            event.stream_id = stream_id.to_string();
            self.global_log.push(event.clone());
            stream.push(event);
        }
        Ok(new_version)
    }

    /// Read all events for a stream.
    pub fn read_stream(&self, stream_id: &str) -> Result<&[EventEnvelope], CqrsError> {
        self.streams
            .get(stream_id)
            .map(|v| v.as_slice())
            .ok_or_else(|| CqrsError::StreamNotFound(stream_id.to_string()))
    }

    /// Read events from a specific version onward.
    pub fn read_from_version(
        &self,
        stream_id: &str,
        from_version: u64,
    ) -> Result<Vec<&EventEnvelope>, CqrsError> {
        let stream = self.streams.get(stream_id)
            .ok_or_else(|| CqrsError::StreamNotFound(stream_id.to_string()))?;
        Ok(stream.iter().filter(|e| e.version > from_version).collect())
    }

    /// Get the current version of a stream.
    pub fn stream_version(&self, stream_id: &str) -> u64 {
        self.streams.get(stream_id)
            .and_then(|s| s.last())
            .map_or(0, |e| e.version)
    }

    /// Save a snapshot.
    pub fn save_snapshot(&mut self, snapshot: Snapshot) {
        self.snapshots
            .entry(snapshot.aggregate_id.clone())
            .or_default()
            .push(snapshot);
    }

    /// Get the latest snapshot for an aggregate.
    pub fn latest_snapshot(&self, aggregate_id: &str) -> Option<&Snapshot> {
        self.snapshots.get(aggregate_id)
            .and_then(|snaps| snaps.last())
    }

    /// Get the global event count.
    pub fn event_count(&self) -> usize {
        self.global_log.len()
    }

    /// Get all events globally (for projections).
    pub fn all_events(&self) -> &[EventEnvelope] {
        &self.global_log
    }

    /// Get all stream IDs.
    pub fn stream_ids(&self) -> Vec<&str> {
        self.streams.keys().map(|s| s.as_str()).collect()
    }
}

// ── Aggregate Root ──────────────────────────────────────────────

/// A simple aggregate root that tracks state via event replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateRoot {
    pub id: String,
    pub aggregate_type: String,
    pub version: u64,
    pub state: HashMap<String, String>,
    pub uncommitted_events: Vec<EventEnvelope>,
}

impl AggregateRoot {
    pub fn new(id: impl Into<String>, aggregate_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            aggregate_type: aggregate_type.into(),
            version: 0,
            state: HashMap::new(),
            uncommitted_events: Vec::new(),
        }
    }

    /// Apply an event to the aggregate state.
    pub fn apply(&mut self, event: &EventEnvelope) {
        // Merge event data into state.
        for (k, v) in &event.data {
            self.state.insert(k.clone(), v.clone());
        }
        self.version = event.version;
    }

    /// Emit a new event (not yet committed to store).
    pub fn emit(&mut self, event_type: impl Into<String>, data: HashMap<String, String>) {
        let event = EventEnvelope::new(
            self.id.clone(),
            event_type,
            self.version + 1,
            data,
        );
        self.apply(&event);
        self.uncommitted_events.push(event);
    }

    /// Take all uncommitted events (after committing to store).
    pub fn take_uncommitted(&mut self) -> Vec<EventEnvelope> {
        std::mem::take(&mut self.uncommitted_events)
    }

    /// Restore from a snapshot and events.
    pub fn restore(
        id: impl Into<String>,
        aggregate_type: impl Into<String>,
        snapshot: Option<&Snapshot>,
        events: &[EventEnvelope],
    ) -> Self {
        let mut agg = Self::new(id, aggregate_type);
        if let Some(snap) = snapshot {
            agg.state = snap.state.clone();
            agg.version = snap.version;
        }
        for event in events {
            agg.apply(event);
        }
        agg
    }

    /// Create a snapshot of current state.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            aggregate_id: self.id.clone(),
            version: self.version,
            state: self.state.clone(),
            timestamp: Utc::now(),
        }
    }
}

// ── Projection ──────────────────────────────────────────────────

/// A read-model projection built from events.
#[derive(Debug, Clone, Default)]
pub struct Projection {
    pub name: String,
    pub data: HashMap<String, HashMap<String, String>>,
    pub last_processed_version: HashMap<String, u64>,
}

impl Projection {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: HashMap::new(),
            last_processed_version: HashMap::new(),
        }
    }

    /// Process an event, updating the projection.
    pub fn process(&mut self, event: &EventEnvelope) {
        let entry = self.data.entry(event.stream_id.clone()).or_default();
        for (k, v) in &event.data {
            entry.insert(k.clone(), v.clone());
        }
        entry.insert("_last_event".to_string(), event.event_type.clone());
        entry.insert("_version".to_string(), event.version.to_string());
        self.last_processed_version.insert(event.stream_id.clone(), event.version);
    }

    /// Replay all events from the store.
    pub fn replay(&mut self, store: &EventStore) {
        self.data.clear();
        self.last_processed_version.clear();
        for event in store.all_events() {
            self.process(event);
        }
    }

    /// Query projection data for a stream.
    pub fn query(&self, stream_id: &str) -> Option<&HashMap<String, String>> {
        self.data.get(stream_id)
    }

    /// Get all projected records.
    pub fn all_records(&self) -> &HashMap<String, HashMap<String, String>> {
        &self.data
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(stream: &str, etype: &str, key: &str, val: &str) -> EventEnvelope {
        let mut data = HashMap::new();
        data.insert(key.to_string(), val.to_string());
        EventEnvelope::new(stream, etype, 0, data)
    }

    #[test]
    fn test_event_store_append_and_read() {
        let mut store = EventStore::new();
        let events = vec![make_event("order-1", "OrderCreated", "status", "created")];
        let version = store.append("order-1", None, events).unwrap();
        assert_eq!(version, 1);

        let stream = store.read_stream("order-1").unwrap();
        assert_eq!(stream.len(), 1);
        assert_eq!(stream[0].event_type, "OrderCreated");
    }

    #[test]
    fn test_version_conflict() {
        let mut store = EventStore::new();
        store.append("s1", None, vec![make_event("s1", "E1", "k", "v")]).unwrap();
        let err = store.append("s1", Some(0), vec![make_event("s1", "E2", "k", "v")]).unwrap_err();
        assert!(matches!(err, CqrsError::VersionConflict { .. }));
    }

    #[test]
    fn test_optimistic_concurrency() {
        let mut store = EventStore::new();
        store.append("s1", Some(0), vec![make_event("s1", "E1", "k", "v1")]).unwrap();
        store.append("s1", Some(1), vec![make_event("s1", "E2", "k", "v2")]).unwrap();
        assert_eq!(store.stream_version("s1"), 2);
    }

    #[test]
    fn test_read_from_version() {
        let mut store = EventStore::new();
        store.append("s1", None, vec![
            make_event("s1", "E1", "k", "v1"),
            make_event("s1", "E2", "k", "v2"),
            make_event("s1", "E3", "k", "v3"),
        ]).unwrap();
        let events = store.read_from_version("s1", 1).unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_aggregate_root() {
        let mut agg = AggregateRoot::new("order-1", "Order");
        let mut data = HashMap::new();
        data.insert("status".to_string(), "created".to_string());
        agg.emit("OrderCreated", data);
        assert_eq!(agg.state.get("status"), Some(&"created".to_string()));
        assert_eq!(agg.version, 1);
        assert_eq!(agg.uncommitted_events.len(), 1);
    }

    #[test]
    fn test_aggregate_commit() {
        let mut store = EventStore::new();
        let mut agg = AggregateRoot::new("order-1", "Order");

        let mut data = HashMap::new();
        data.insert("status".to_string(), "created".to_string());
        agg.emit("OrderCreated", data);

        let events = agg.take_uncommitted();
        assert!(agg.uncommitted_events.is_empty());
        store.append("order-1", Some(0), events).unwrap();
        assert_eq!(store.stream_version("order-1"), 1);
    }

    #[test]
    fn test_snapshot_and_restore() {
        let mut agg = AggregateRoot::new("order-1", "Order");
        let mut d1 = HashMap::new();
        d1.insert("status".to_string(), "created".to_string());
        agg.emit("OrderCreated", d1);

        let snap = agg.snapshot();
        assert_eq!(snap.version, 1);

        // Restore from snapshot.
        let restored = AggregateRoot::restore("order-1", "Order", Some(&snap), &[]);
        assert_eq!(restored.version, 1);
        assert_eq!(restored.state.get("status"), Some(&"created".to_string()));
    }

    #[test]
    fn test_snapshot_store() {
        let mut store = EventStore::new();
        store.append("s1", None, vec![make_event("s1", "E1", "k", "v")]).unwrap();
        store.save_snapshot(Snapshot {
            aggregate_id: "s1".into(),
            version: 1,
            state: {
                let mut m = HashMap::new();
                m.insert("k".into(), "v".into());
                m
            },
            timestamp: Utc::now(),
        });
        let snap = store.latest_snapshot("s1").unwrap();
        assert_eq!(snap.version, 1);
    }

    #[test]
    fn test_projection() {
        let mut store = EventStore::new();
        store.append("order-1", None, vec![
            make_event("order-1", "OrderCreated", "status", "created"),
        ]).unwrap();
        store.append("order-1", Some(1), vec![
            make_event("order-1", "OrderShipped", "status", "shipped"),
        ]).unwrap();

        let mut proj = Projection::new("orders");
        proj.replay(&store);

        let data = proj.query("order-1").unwrap();
        assert_eq!(data.get("status"), Some(&"shipped".to_string()));
        assert_eq!(data.get("_last_event"), Some(&"OrderShipped".to_string()));
    }

    #[test]
    fn test_projection_multiple_streams() {
        let mut store = EventStore::new();
        store.append("s1", None, vec![make_event("s1", "E1", "x", "1")]).unwrap();
        store.append("s2", None, vec![make_event("s2", "E1", "y", "2")]).unwrap();

        let mut proj = Projection::new("all");
        proj.replay(&store);
        assert_eq!(proj.all_records().len(), 2);
    }

    #[test]
    fn test_global_event_count() {
        let mut store = EventStore::new();
        store.append("s1", None, vec![make_event("s1", "E1", "k", "v")]).unwrap();
        store.append("s2", None, vec![
            make_event("s2", "E1", "k", "v"),
            make_event("s2", "E2", "k", "v"),
        ]).unwrap();
        assert_eq!(store.event_count(), 3);
    }

    #[test]
    fn test_stream_not_found() {
        let store = EventStore::new();
        assert!(matches!(store.read_stream("nope"), Err(CqrsError::StreamNotFound(_))));
    }

    #[test]
    fn test_command_creation() {
        let mut payload = HashMap::new();
        payload.insert("item".to_string(), "widget".to_string());
        let cmd = Command::new("order-1", "CreateOrder", payload).with_expected_version(0);
        assert_eq!(cmd.aggregate_id, "order-1");
        assert_eq!(cmd.expected_version, Some(0));
    }
}
