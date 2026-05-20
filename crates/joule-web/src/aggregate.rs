//! DDD Aggregate Root — consistency boundary enforcement, invariant validation,
//! domain event publishing, command handling, version-checked apply, aggregate
//! reconstruction from events, and snapshot support.
//!
//! Replaces ad-hoc aggregate patterns in JS/TS with a pure-Rust aggregate
//! root that enforces consistency boundaries and tracks domain events.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Aggregate domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateError {
    /// Invariant violation.
    InvariantViolation { aggregate_id: String, rule: String },
    /// Version conflict.
    VersionConflict { aggregate_id: String, expected: u64, actual: u64 },
    /// Command rejected.
    CommandRejected { aggregate_id: String, reason: String },
    /// Unknown event type during replay.
    UnknownEventType { aggregate_id: String, event_type: String },
    /// Aggregate not found.
    NotFound(String),
    /// Aggregate already exists.
    AlreadyExists(String),
    /// Invalid state for operation.
    InvalidState { aggregate_id: String, reason: String },
    /// Snapshot error.
    SnapshotError(String),
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvariantViolation { aggregate_id, rule } => {
                write!(f, "invariant violation on {aggregate_id}: {rule}")
            }
            Self::VersionConflict { aggregate_id, expected, actual } => {
                write!(f, "version conflict for {aggregate_id}: expected {expected}, got {actual}")
            }
            Self::CommandRejected { aggregate_id, reason } => {
                write!(f, "command rejected for {aggregate_id}: {reason}")
            }
            Self::UnknownEventType { aggregate_id, event_type } => {
                write!(f, "unknown event type for {aggregate_id}: {event_type}")
            }
            Self::NotFound(id) => write!(f, "aggregate not found: {id}"),
            Self::AlreadyExists(id) => write!(f, "aggregate already exists: {id}"),
            Self::InvalidState { aggregate_id, reason } => {
                write!(f, "invalid state for {aggregate_id}: {reason}")
            }
            Self::SnapshotError(msg) => write!(f, "snapshot error: {msg}"),
        }
    }
}

impl std::error::Error for AggregateError {}

// ── AggregateEvent ──────────────────────────────────────────────

/// A domain event produced by an aggregate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateEvent {
    pub event_id: String,
    pub aggregate_id: String,
    pub event_type: String,
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub data: HashMap<String, String>,
}

impl AggregateEvent {
    pub fn new(
        aggregate_id: impl Into<String>,
        event_type: impl Into<String>,
        version: u64,
        data: HashMap<String, String>,
    ) -> Self {
        let ts = Utc::now();
        let agg_id = aggregate_id.into();
        Self {
            event_id: format!("{}-{}-{}", agg_id, version, ts.timestamp_nanos_opt().unwrap_or(0)),
            aggregate_id: agg_id,
            event_type: event_type.into(),
            version,
            timestamp: ts,
            data,
        }
    }
}

// ── InvariantRule ────────────────────────────────────────────────

/// An invariant rule that can be checked against aggregate state.
#[derive(Debug, Clone)]
pub struct InvariantRule {
    pub name: String,
    pub description: String,
    check_fn: fn(&HashMap<String, String>) -> bool,
}

impl InvariantRule {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        check_fn: fn(&HashMap<String, String>) -> bool,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            check_fn,
        }
    }

    /// Evaluate the rule against the given state.
    pub fn check(&self, state: &HashMap<String, String>) -> bool {
        (self.check_fn)(state)
    }
}

// ── Snapshot ────────────────────────────────────────────────────

/// A snapshot of aggregate state at a given version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateSnapshot {
    pub aggregate_id: String,
    pub version: u64,
    pub state: HashMap<String, String>,
    pub taken_at: DateTime<Utc>,
}

// ── Command ─────────────────────────────────────────────────────

/// A command to be handled by an aggregate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Command {
    pub command_type: String,
    pub aggregate_id: String,
    pub data: HashMap<String, String>,
    pub expected_version: Option<u64>,
}

impl Command {
    pub fn new(
        command_type: impl Into<String>,
        aggregate_id: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        Self {
            command_type: command_type.into(),
            aggregate_id: aggregate_id.into(),
            data,
            expected_version: None,
        }
    }

    pub fn with_version(mut self, version: u64) -> Self {
        self.expected_version = Some(version);
        self
    }
}

// ── CommandHandler ──────────────────────────────────────────────

/// A registered command handler that maps a command type to event production.
#[derive(Clone)]
pub struct CommandHandler {
    pub command_type: String,
    handler_fn: fn(&HashMap<String, String>, &HashMap<String, String>)
        -> Result<Vec<(String, HashMap<String, String>)>, String>,
}

impl CommandHandler {
    pub fn new(
        command_type: impl Into<String>,
        handler_fn: fn(&HashMap<String, String>, &HashMap<String, String>)
            -> Result<Vec<(String, HashMap<String, String>)>, String>,
    ) -> Self {
        Self {
            command_type: command_type.into(),
            handler_fn,
        }
    }

    pub fn handle(
        &self,
        state: &HashMap<String, String>,
        data: &HashMap<String, String>,
    ) -> Result<Vec<(String, HashMap<String, String>)>, String> {
        (self.handler_fn)(state, data)
    }
}

impl std::fmt::Debug for CommandHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandHandler")
            .field("command_type", &self.command_type)
            .finish()
    }
}

// ── EventApplier ────────────────────────────────────────────────

/// An event applier that mutates aggregate state.
#[derive(Clone)]
pub struct EventApplier {
    pub event_type: String,
    apply_fn: fn(&mut HashMap<String, String>, &HashMap<String, String>),
}

impl EventApplier {
    pub fn new(
        event_type: impl Into<String>,
        apply_fn: fn(&mut HashMap<String, String>, &HashMap<String, String>),
    ) -> Self {
        Self {
            event_type: event_type.into(),
            apply_fn,
        }
    }

    pub fn apply(&self, state: &mut HashMap<String, String>, data: &HashMap<String, String>) {
        (self.apply_fn)(state, data);
    }
}

impl std::fmt::Debug for EventApplier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventApplier")
            .field("event_type", &self.event_type)
            .finish()
    }
}

// ── AggregateRoot ───────────────────────────────────────────────

/// An aggregate root with consistency boundary, event sourcing, and snapshots.
#[derive(Debug, Clone)]
pub struct AggregateRoot {
    id: String,
    version: u64,
    state: HashMap<String, String>,
    pending_events: Vec<AggregateEvent>,
    invariants: Vec<InvariantRule>,
    command_handlers: Vec<CommandHandler>,
    event_appliers: Vec<EventApplier>,
    created_at: DateTime<Utc>,
}

impl AggregateRoot {
    /// Create a new aggregate root.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: 0,
            state: HashMap::new(),
            pending_events: Vec::new(),
            invariants: Vec::new(),
            command_handlers: Vec::new(),
            event_appliers: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Restore from a snapshot.
    pub fn from_snapshot(snapshot: &AggregateSnapshot) -> Self {
        Self {
            id: snapshot.aggregate_id.clone(),
            version: snapshot.version,
            state: snapshot.state.clone(),
            pending_events: Vec::new(),
            invariants: Vec::new(),
            command_handlers: Vec::new(),
            event_appliers: Vec::new(),
            created_at: snapshot.taken_at,
        }
    }

    pub fn id(&self) -> &str { &self.id }
    pub fn version(&self) -> u64 { self.version }
    pub fn state(&self) -> &HashMap<String, String> { &self.state }
    pub fn created_at(&self) -> DateTime<Utc> { self.created_at }

    /// Register an invariant rule.
    pub fn add_invariant(&mut self, rule: InvariantRule) {
        self.invariants.push(rule);
    }

    /// Register a command handler.
    pub fn add_command_handler(&mut self, handler: CommandHandler) {
        self.command_handlers.push(handler);
    }

    /// Register an event applier.
    pub fn add_event_applier(&mut self, applier: EventApplier) {
        self.event_appliers.push(applier);
    }

    /// Validate all invariants against current state.
    pub fn validate_invariants(&self) -> Result<(), AggregateError> {
        for rule in &self.invariants {
            if !rule.check(&self.state) {
                return Err(AggregateError::InvariantViolation {
                    aggregate_id: self.id.clone(),
                    rule: rule.name.clone(),
                });
            }
        }
        Ok(())
    }

    /// Handle a command, producing events.
    pub fn handle_command(&mut self, cmd: &Command) -> Result<Vec<AggregateEvent>, AggregateError> {
        // Version check if requested.
        if let Some(expected) = cmd.expected_version {
            if expected != self.version {
                return Err(AggregateError::VersionConflict {
                    aggregate_id: self.id.clone(),
                    expected,
                    actual: self.version,
                });
            }
        }

        // Find handler.
        let handler = self.command_handlers
            .iter()
            .find(|h| h.command_type == cmd.command_type)
            .cloned()
            .ok_or_else(|| AggregateError::CommandRejected {
                aggregate_id: self.id.clone(),
                reason: format!("no handler for command type: {}", cmd.command_type),
            })?;

        // Execute handler.
        let event_specs = handler.handle(&self.state, &cmd.data).map_err(|reason| {
            AggregateError::CommandRejected {
                aggregate_id: self.id.clone(),
                reason,
            }
        })?;

        // Build and apply events.
        let mut produced = Vec::new();
        for (event_type, data) in event_specs {
            self.version += 1;
            let event = AggregateEvent::new(&self.id, &event_type, self.version, data);

            // Apply via applier.
            let applier = self.event_appliers.iter().find(|a| a.event_type == event_type).cloned();
            if let Some(ap) = applier {
                ap.apply(&mut self.state, &event.data);
            }

            self.pending_events.push(event.clone());
            produced.push(event);
        }

        // Validate invariants after state change.
        self.validate_invariants()?;

        Ok(produced)
    }

    /// Apply a single event directly (for replay / event sourcing).
    pub fn apply_event(&mut self, event: &AggregateEvent) -> Result<(), AggregateError> {
        let et = event.event_type.clone();
        let applier = self.event_appliers.iter().find(|a| a.event_type == et).cloned();
        match applier {
            Some(ap) => {
                ap.apply(&mut self.state, &event.data);
                self.version = event.version;
                Ok(())
            }
            None => Err(AggregateError::UnknownEventType {
                aggregate_id: self.id.clone(),
                event_type: et,
            }),
        }
    }

    /// Replay a sequence of events.
    pub fn replay(&mut self, events: &[AggregateEvent]) -> Result<(), AggregateError> {
        for event in events {
            self.apply_event(event)?;
        }
        Ok(())
    }

    /// Take a snapshot of the current state.
    pub fn take_snapshot(&self) -> AggregateSnapshot {
        AggregateSnapshot {
            aggregate_id: self.id.clone(),
            version: self.version,
            state: self.state.clone(),
            taken_at: Utc::now(),
        }
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<AggregateEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Number of pending events.
    pub fn pending_event_count(&self) -> usize {
        self.pending_events.len()
    }

    /// Set a state value directly (for testing / reconstruction).
    pub fn set_state(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.state.insert(key.into(), value.into());
    }

    /// Get a state value.
    pub fn get_state(&self, key: &str) -> Option<&str> {
        self.state.get(key).map(|s| s.as_str())
    }

    /// Remove a state value.
    pub fn remove_state(&mut self, key: &str) -> Option<String> {
        self.state.remove(key)
    }
}

// ── AggregateStore ──────────────────────────────────────────────

/// In-memory store for aggregates with snapshot support.
#[derive(Debug)]
pub struct AggregateStore {
    snapshots: HashMap<String, Vec<AggregateSnapshot>>,
    events: HashMap<String, Vec<AggregateEvent>>,
}

impl AggregateStore {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            events: HashMap::new(),
        }
    }

    /// Save events for an aggregate.
    pub fn save_events(&mut self, agg_id: &str, events: Vec<AggregateEvent>) {
        self.events.entry(agg_id.to_string()).or_default().extend(events);
    }

    /// Get all events for an aggregate.
    pub fn get_events(&self, agg_id: &str) -> Vec<AggregateEvent> {
        self.events.get(agg_id).cloned().unwrap_or_default()
    }

    /// Save a snapshot.
    pub fn save_snapshot(&mut self, snapshot: AggregateSnapshot) {
        self.snapshots.entry(snapshot.aggregate_id.clone()).or_default().push(snapshot);
    }

    /// Get the latest snapshot.
    pub fn latest_snapshot(&self, agg_id: &str) -> Option<&AggregateSnapshot> {
        self.snapshots.get(agg_id).and_then(|snaps| snaps.last())
    }

    /// Get events after a specific version.
    pub fn events_after_version(&self, agg_id: &str, version: u64) -> Vec<AggregateEvent> {
        self.events.get(agg_id)
            .map(|evts| evts.iter().filter(|e| e.version > version).cloned().collect())
            .unwrap_or_default()
    }

    /// Total event count across all aggregates.
    pub fn total_event_count(&self) -> usize {
        self.events.values().map(|v| v.len()).sum()
    }

    /// Total snapshot count.
    pub fn total_snapshot_count(&self) -> usize {
        self.snapshots.values().map(|v| v.len()).sum()
    }
}

impl Default for AggregateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aggregate() -> AggregateRoot {
        let mut agg = AggregateRoot::new("order-1");

        // Event applier: "item_added" sets item in state.
        agg.add_event_applier(EventApplier::new("item_added", |state, data| {
            if let Some(item) = data.get("item") {
                let count: u32 = state.get("item_count").and_then(|c| c.parse().ok()).unwrap_or(0);
                state.insert("item_count".to_string(), (count + 1).to_string());
                state.insert(format!("item_{count}"), item.clone());
            }
        }));

        // Event applier: "order_placed".
        agg.add_event_applier(EventApplier::new("order_placed", |state, _data| {
            state.insert("status".to_string(), "placed".to_string());
        }));

        // Command handler: "add_item".
        agg.add_command_handler(CommandHandler::new(
            "add_item",
            |_state, data| {
                if data.get("item").map(|i| i.is_empty()).unwrap_or(true) {
                    return Err("item name required".to_string());
                }
                Ok(vec![("item_added".to_string(), data.clone())])
            },
        ));

        // Command handler: "place_order".
        agg.add_command_handler(CommandHandler::new(
            "place_order",
            |state, data| {
                let count: u32 = state.get("item_count").and_then(|c| c.parse().ok()).unwrap_or(0);
                if count == 0 {
                    return Err("cannot place empty order".to_string());
                }
                Ok(vec![("order_placed".to_string(), data.clone())])
            },
        ));

        agg
    }

    #[test]
    fn test_new_aggregate() {
        let agg = AggregateRoot::new("agg-1");
        assert_eq!(agg.id(), "agg-1");
        assert_eq!(agg.version(), 0);
    }

    #[test]
    fn test_handle_command_success() {
        let mut agg = make_aggregate();
        let mut data = HashMap::new();
        data.insert("item".to_string(), "widget".to_string());
        let cmd = Command::new("add_item", "order-1", data);
        let events = agg.handle_command(&cmd).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "item_added");
        assert_eq!(agg.version(), 1);
        assert_eq!(agg.get_state("item_count"), Some("1"));
    }

    #[test]
    fn test_handle_command_rejected() {
        let mut agg = make_aggregate();
        let data = HashMap::new();
        let cmd = Command::new("add_item", "order-1", data);
        let result = agg.handle_command(&cmd);
        assert!(matches!(result, Err(AggregateError::CommandRejected { .. })));
    }

    #[test]
    fn test_version_conflict() {
        let mut agg = make_aggregate();
        let data = HashMap::new();
        let cmd = Command::new("add_item", "order-1", data).with_version(5);
        let result = agg.handle_command(&cmd);
        assert!(matches!(result, Err(AggregateError::VersionConflict { .. })));
    }

    #[test]
    fn test_unknown_command() {
        let mut agg = make_aggregate();
        let cmd = Command::new("unknown", "order-1", HashMap::new());
        let result = agg.handle_command(&cmd);
        assert!(matches!(result, Err(AggregateError::CommandRejected { .. })));
    }

    #[test]
    fn test_multiple_commands() {
        let mut agg = make_aggregate();
        let mut d1 = HashMap::new();
        d1.insert("item".to_string(), "widget".to_string());
        agg.handle_command(&Command::new("add_item", "order-1", d1)).unwrap();

        let mut d2 = HashMap::new();
        d2.insert("item".to_string(), "gadget".to_string());
        agg.handle_command(&Command::new("add_item", "order-1", d2)).unwrap();

        assert_eq!(agg.version(), 2);
        assert_eq!(agg.get_state("item_count"), Some("2"));
    }

    #[test]
    fn test_place_order_requires_items() {
        let mut agg = make_aggregate();
        let cmd = Command::new("place_order", "order-1", HashMap::new());
        let result = agg.handle_command(&cmd);
        assert!(matches!(result, Err(AggregateError::CommandRejected { .. })));
    }

    #[test]
    fn test_place_order_success() {
        let mut agg = make_aggregate();
        let mut d1 = HashMap::new();
        d1.insert("item".to_string(), "widget".to_string());
        agg.handle_command(&Command::new("add_item", "order-1", d1)).unwrap();
        agg.handle_command(&Command::new("place_order", "order-1", HashMap::new())).unwrap();
        assert_eq!(agg.get_state("status"), Some("placed"));
    }

    #[test]
    fn test_snapshot_and_restore() {
        let mut agg = make_aggregate();
        let mut d = HashMap::new();
        d.insert("item".to_string(), "widget".to_string());
        agg.handle_command(&Command::new("add_item", "order-1", d)).unwrap();
        let snap = agg.take_snapshot();
        assert_eq!(snap.version, 1);
        assert_eq!(snap.aggregate_id, "order-1");

        let restored = AggregateRoot::from_snapshot(&snap);
        assert_eq!(restored.version(), 1);
        assert_eq!(restored.get_state("item_count"), Some("1"));
    }

    #[test]
    fn test_event_replay() {
        let mut agg = make_aggregate();
        let mut d = HashMap::new();
        d.insert("item".to_string(), "widget".to_string());
        agg.handle_command(&Command::new("add_item", "order-1", d)).unwrap();
        let events = agg.drain_events();

        // Replay on a fresh aggregate.
        let mut agg2 = make_aggregate();
        agg2.replay(&events).unwrap();
        assert_eq!(agg2.version(), 1);
        assert_eq!(agg2.get_state("item_count"), Some("1"));
    }

    #[test]
    fn test_invariant_violation() {
        let mut agg = AggregateRoot::new("inv-1");
        agg.add_invariant(InvariantRule::new(
            "max_items",
            "cannot have more than 2 items",
            |state| {
                let count: u32 = state.get("item_count").and_then(|c| c.parse().ok()).unwrap_or(0);
                count <= 2
            },
        ));
        agg.add_event_applier(EventApplier::new("item_added", |state, _data| {
            let count: u32 = state.get("item_count").and_then(|c| c.parse().ok()).unwrap_or(0);
            state.insert("item_count".to_string(), (count + 1).to_string());
        }));
        agg.add_command_handler(CommandHandler::new(
            "add_item",
            |_state, data| Ok(vec![("item_added".to_string(), data.clone())]),
        ));

        agg.handle_command(&Command::new("add_item", "inv-1", HashMap::new())).unwrap();
        agg.handle_command(&Command::new("add_item", "inv-1", HashMap::new())).unwrap();
        let result = agg.handle_command(&Command::new("add_item", "inv-1", HashMap::new()));
        assert!(matches!(result, Err(AggregateError::InvariantViolation { .. })));
    }

    #[test]
    fn test_drain_events() {
        let mut agg = make_aggregate();
        let mut d = HashMap::new();
        d.insert("item".to_string(), "widget".to_string());
        agg.handle_command(&Command::new("add_item", "order-1", d)).unwrap();
        assert_eq!(agg.pending_event_count(), 1);
        let events = agg.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(agg.pending_event_count(), 0);
    }

    #[test]
    fn test_aggregate_store() {
        let mut store = AggregateStore::new();
        let event = AggregateEvent::new("agg-1", "created", 1, HashMap::new());
        store.save_events("agg-1", vec![event.clone()]);
        assert_eq!(store.get_events("agg-1").len(), 1);
        assert_eq!(store.total_event_count(), 1);
    }

    #[test]
    fn test_aggregate_store_snapshots() {
        let mut store = AggregateStore::new();
        let snap = AggregateSnapshot {
            aggregate_id: "agg-1".to_string(),
            version: 3,
            state: HashMap::new(),
            taken_at: Utc::now(),
        };
        store.save_snapshot(snap.clone());
        let latest = store.latest_snapshot("agg-1").unwrap();
        assert_eq!(latest.version, 3);
        assert_eq!(store.total_snapshot_count(), 1);
    }

    #[test]
    fn test_events_after_version() {
        let mut store = AggregateStore::new();
        let e1 = AggregateEvent::new("agg-1", "ev1", 1, HashMap::new());
        let e2 = AggregateEvent::new("agg-1", "ev2", 2, HashMap::new());
        let e3 = AggregateEvent::new("agg-1", "ev3", 3, HashMap::new());
        store.save_events("agg-1", vec![e1, e2, e3]);
        let after = store.events_after_version("agg-1", 1);
        assert_eq!(after.len(), 2);
    }

    #[test]
    fn test_set_and_remove_state() {
        let mut agg = AggregateRoot::new("agg-1");
        agg.set_state("key", "value");
        assert_eq!(agg.get_state("key"), Some("value"));
        agg.remove_state("key");
        assert!(agg.get_state("key").is_none());
    }

    #[test]
    fn test_command_with_version() {
        let cmd = Command::new("test", "agg-1", HashMap::new()).with_version(5);
        assert_eq!(cmd.expected_version, Some(5));
    }
}
