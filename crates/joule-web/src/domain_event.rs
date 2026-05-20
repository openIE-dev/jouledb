//! Domain events — event base (id, timestamp, aggregate_id, version), event
//! store (append-only), event replay, event handlers, event bus (publish/
//! subscribe), event sourcing helpers, and event upcasting (version migration).
//!
//! Replaces JS event sourcing libraries (EventStoreDB client, Node EventEmitter)
//! with a pure-Rust domain event system for DDD-style event-driven architectures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Domain event errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainEventError {
    /// Stream not found.
    StreamNotFound(String),
    /// Event not found.
    EventNotFound(String),
    /// Handler error.
    HandlerError { handler_id: String, reason: String },
    /// Version conflict.
    VersionConflict { stream_id: String, expected: u64, actual: u64 },
    /// Upcast error.
    UpcastError { event_type: String, from_version: u32, reason: String },
    /// Duplicate subscription.
    DuplicateSubscription(String),
    /// Serialization error.
    SerializationError(String),
}

impl std::fmt::Display for DomainEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
            Self::EventNotFound(id) => write!(f, "event not found: {id}"),
            Self::HandlerError { handler_id, reason } => {
                write!(f, "handler {handler_id} error: {reason}")
            }
            Self::VersionConflict { stream_id, expected, actual } => {
                write!(f, "version conflict on {stream_id}: expected {expected}, got {actual}")
            }
            Self::UpcastError { event_type, from_version, reason } => {
                write!(f, "upcast error for {event_type} v{from_version}: {reason}")
            }
            Self::DuplicateSubscription(id) => write!(f, "duplicate subscription: {id}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl std::error::Error for DomainEventError {}

// ── DomainEvent ─────────────────────────────────────────────────

/// A domain event with identity, versioning, and metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainEvent {
    pub event_id: String,
    pub event_type: String,
    pub aggregate_id: String,
    pub aggregate_type: String,
    pub version: u64,
    pub schema_version: u32,
    pub timestamp: DateTime<Utc>,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

impl DomainEvent {
    pub fn new(
        event_type: impl Into<String>,
        aggregate_id: impl Into<String>,
        aggregate_type: impl Into<String>,
        version: u64,
        data: HashMap<String, String>,
    ) -> Self {
        let ts = Utc::now();
        let agg_id = aggregate_id.into();
        Self {
            event_id: format!("{}-{}-{}", agg_id, version, ts.timestamp_nanos_opt().unwrap_or(0)),
            event_type: event_type.into(),
            aggregate_id: agg_id,
            aggregate_type: aggregate_type.into(),
            version,
            schema_version: 1,
            timestamp: ts,
            data,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this event.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set schema version.
    pub fn with_schema_version(mut self, version: u32) -> Self {
        self.schema_version = version;
        self
    }
}

// ── EventStream ─────────────────────────────────────────────────

/// An ordered stream of events for a single aggregate.
#[derive(Debug, Clone)]
pub struct EventStream {
    pub stream_id: String,
    pub aggregate_type: String,
    events: Vec<DomainEvent>,
}

impl EventStream {
    pub fn new(stream_id: impl Into<String>, aggregate_type: impl Into<String>) -> Self {
        Self {
            stream_id: stream_id.into(),
            aggregate_type: aggregate_type.into(),
            events: Vec::new(),
        }
    }

    /// Append an event with version check.
    pub fn append(&mut self, event: DomainEvent) -> Result<(), DomainEventError> {
        let expected = self.events.last().map(|e| e.version + 1).unwrap_or(1);
        if event.version != expected {
            return Err(DomainEventError::VersionConflict {
                stream_id: self.stream_id.clone(),
                expected,
                actual: event.version,
            });
        }
        self.events.push(event);
        Ok(())
    }

    /// All events in order.
    pub fn events(&self) -> &[DomainEvent] {
        &self.events
    }

    /// Events after a given version.
    pub fn events_after(&self, version: u64) -> Vec<&DomainEvent> {
        self.events.iter().filter(|e| e.version > version).collect()
    }

    /// Latest version in the stream.
    pub fn current_version(&self) -> u64 {
        self.events.last().map(|e| e.version).unwrap_or(0)
    }

    /// Event count.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ── EventStore ──────────────────────────────────────────────────

/// Append-only event store.
#[derive(Debug)]
pub struct EventStore {
    streams: HashMap<String, EventStream>,
    global_log: Vec<DomainEvent>,
}

impl EventStore {
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            global_log: Vec::new(),
        }
    }

    /// Append events to a stream. Creates the stream if it doesn't exist.
    pub fn append(
        &mut self,
        stream_id: &str,
        aggregate_type: &str,
        events: Vec<DomainEvent>,
    ) -> Result<(), DomainEventError> {
        let stream = self.streams
            .entry(stream_id.to_string())
            .or_insert_with(|| EventStream::new(stream_id, aggregate_type));
        for event in events {
            stream.append(event.clone())?;
            self.global_log.push(event);
        }
        Ok(())
    }

    /// Get a stream by id.
    pub fn get_stream(&self, stream_id: &str) -> Option<&EventStream> {
        self.streams.get(stream_id)
    }

    /// Get all events for a stream.
    pub fn get_events(&self, stream_id: &str) -> Result<Vec<&DomainEvent>, DomainEventError> {
        self.streams.get(stream_id)
            .map(|s| s.events().iter().collect())
            .ok_or_else(|| DomainEventError::StreamNotFound(stream_id.to_string()))
    }

    /// Get events after a version.
    pub fn get_events_after(
        &self,
        stream_id: &str,
        version: u64,
    ) -> Result<Vec<&DomainEvent>, DomainEventError> {
        self.streams.get(stream_id)
            .map(|s| s.events_after(version))
            .ok_or_else(|| DomainEventError::StreamNotFound(stream_id.to_string()))
    }

    /// Total event count across all streams.
    pub fn total_event_count(&self) -> usize {
        self.global_log.len()
    }

    /// Stream count.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Global log in order.
    pub fn global_log(&self) -> &[DomainEvent] {
        &self.global_log
    }

    /// All events of a given type across all streams.
    pub fn events_by_type(&self, event_type: &str) -> Vec<&DomainEvent> {
        self.global_log.iter().filter(|e| e.event_type == event_type).collect()
    }
}

impl Default for EventStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── EventHandler ────────────────────────────────────────────────

/// A handler result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerResult {
    Ok,
    Error(String),
    Skip,
}

/// An event handler that processes domain events.
#[derive(Clone)]
pub struct EventHandler {
    pub handler_id: String,
    pub event_types: Vec<String>,
    handle_fn: fn(&DomainEvent) -> HandlerResult,
}

impl EventHandler {
    pub fn new(
        handler_id: impl Into<String>,
        event_types: Vec<String>,
        handle_fn: fn(&DomainEvent) -> HandlerResult,
    ) -> Self {
        Self {
            handler_id: handler_id.into(),
            event_types,
            handle_fn,
        }
    }

    /// Whether this handler handles the given event type.
    pub fn handles(&self, event_type: &str) -> bool {
        self.event_types.iter().any(|t| t == event_type)
    }

    /// Handle an event.
    pub fn handle(&self, event: &DomainEvent) -> HandlerResult {
        (self.handle_fn)(event)
    }
}

impl std::fmt::Debug for EventHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventHandler")
            .field("handler_id", &self.handler_id)
            .field("event_types", &self.event_types)
            .finish()
    }
}

// ── EventBus ────────────────────────────────────────────────────

/// Publish/subscribe event bus for domain events.
#[derive(Debug)]
pub struct DomainEventBus {
    handlers: Vec<EventHandler>,
    published_count: u64,
    error_count: u64,
}

impl DomainEventBus {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
            published_count: 0,
            error_count: 0,
        }
    }

    /// Subscribe a handler.
    pub fn subscribe(&mut self, handler: EventHandler) -> Result<(), DomainEventError> {
        if self.handlers.iter().any(|h| h.handler_id == handler.handler_id) {
            return Err(DomainEventError::DuplicateSubscription(handler.handler_id));
        }
        self.handlers.push(handler);
        Ok(())
    }

    /// Unsubscribe a handler by id.
    pub fn unsubscribe(&mut self, handler_id: &str) -> bool {
        let before = self.handlers.len();
        self.handlers.retain(|h| h.handler_id != handler_id);
        self.handlers.len() < before
    }

    /// Publish a single event to all matching handlers.
    pub fn publish(&mut self, event: &DomainEvent) -> Vec<(String, HandlerResult)> {
        self.published_count += 1;
        let mut results = Vec::new();
        let handlers: Vec<EventHandler> = self.handlers.clone();
        for handler in &handlers {
            if handler.handles(&event.event_type) {
                let result = handler.handle(event);
                if matches!(result, HandlerResult::Error(_)) {
                    self.error_count += 1;
                }
                results.push((handler.handler_id.clone(), result));
            }
        }
        results
    }

    /// Publish multiple events.
    pub fn publish_all(&mut self, events: &[DomainEvent]) -> Vec<Vec<(String, HandlerResult)>> {
        events.iter().map(|e| self.publish(e)).collect()
    }

    /// Handler count.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Published event count.
    pub fn published_count(&self) -> u64 {
        self.published_count
    }

    /// Error count.
    pub fn error_count(&self) -> u64 {
        self.error_count
    }
}

impl Default for DomainEventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ── EventUpcaster ───────────────────────────────────────────────

/// An upcaster that migrates events from one schema version to another.
#[derive(Clone)]
pub struct EventUpcaster {
    pub event_type: String,
    pub from_version: u32,
    pub to_version: u32,
    upcast_fn: fn(&mut DomainEvent),
}

impl EventUpcaster {
    pub fn new(
        event_type: impl Into<String>,
        from_version: u32,
        to_version: u32,
        upcast_fn: fn(&mut DomainEvent),
    ) -> Self {
        Self {
            event_type: event_type.into(),
            from_version,
            to_version,
            upcast_fn,
        }
    }

    /// Apply the upcast to a mutable event.
    pub fn upcast(&self, event: &mut DomainEvent) {
        (self.upcast_fn)(event);
        event.schema_version = self.to_version;
    }
}

impl std::fmt::Debug for EventUpcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventUpcaster")
            .field("event_type", &self.event_type)
            .field("from_version", &self.from_version)
            .field("to_version", &self.to_version)
            .finish()
    }
}

/// Registry of upcasters for migrating event schemas.
#[derive(Debug, Default)]
pub struct UpcasterRegistry {
    upcasters: Vec<EventUpcaster>,
}

impl UpcasterRegistry {
    pub fn new() -> Self {
        Self { upcasters: Vec::new() }
    }

    /// Register an upcaster.
    pub fn register(&mut self, upcaster: EventUpcaster) {
        self.upcasters.push(upcaster);
    }

    /// Upcast an event through all applicable version migrations.
    pub fn upcast(&self, event: &mut DomainEvent) {
        loop {
            let found = self.upcasters.iter().find(|u| {
                u.event_type == event.event_type && u.from_version == event.schema_version
            }).cloned();
            match found {
                Some(upcaster) => upcaster.upcast(event),
                None => break,
            }
        }
    }

    /// Upcast a batch of events.
    pub fn upcast_all(&self, events: &mut [DomainEvent]) {
        for event in events.iter_mut() {
            self.upcast(event);
        }
    }
}

// ── EventSourcingHelper ─────────────────────────────────────────

/// Replays events to reconstruct state.
pub fn replay_events(
    events: &[DomainEvent],
    mut apply: impl FnMut(&mut HashMap<String, String>, &DomainEvent),
) -> HashMap<String, String> {
    let mut state = HashMap::new();
    for event in events {
        apply(&mut state, event);
    }
    state
}

/// Replays events from a given version.
pub fn replay_from_version(
    events: &[DomainEvent],
    from_version: u64,
    mut apply: impl FnMut(&mut HashMap<String, String>, &DomainEvent),
    initial_state: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut state = initial_state;
    for event in events.iter().filter(|e| e.version > from_version) {
        apply(&mut state, event);
    }
    state
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(event_type: &str, agg_id: &str, version: u64) -> DomainEvent {
        DomainEvent::new(event_type, agg_id, "Order", version, HashMap::new())
    }

    fn make_event_with_data(
        event_type: &str,
        agg_id: &str,
        version: u64,
        data: HashMap<String, String>,
    ) -> DomainEvent {
        DomainEvent::new(event_type, agg_id, "Order", version, data)
    }

    #[test]
    fn test_domain_event_creation() {
        let event = make_event("order_created", "order-1", 1);
        assert_eq!(event.event_type, "order_created");
        assert_eq!(event.aggregate_id, "order-1");
        assert_eq!(event.version, 1);
        assert_eq!(event.schema_version, 1);
    }

    #[test]
    fn test_event_with_metadata() {
        let event = make_event("order_created", "order-1", 1)
            .with_metadata("user_id", "user-42");
        assert_eq!(event.metadata.get("user_id").unwrap(), "user-42");
    }

    #[test]
    fn test_event_stream_append() {
        let mut stream = EventStream::new("order-1", "Order");
        stream.append(make_event("created", "order-1", 1)).unwrap();
        stream.append(make_event("updated", "order-1", 2)).unwrap();
        assert_eq!(stream.len(), 2);
        assert_eq!(stream.current_version(), 2);
    }

    #[test]
    fn test_event_stream_version_conflict() {
        let mut stream = EventStream::new("order-1", "Order");
        stream.append(make_event("created", "order-1", 1)).unwrap();
        let result = stream.append(make_event("updated", "order-1", 5));
        assert!(matches!(result, Err(DomainEventError::VersionConflict { .. })));
    }

    #[test]
    fn test_event_stream_events_after() {
        let mut stream = EventStream::new("order-1", "Order");
        stream.append(make_event("ev1", "order-1", 1)).unwrap();
        stream.append(make_event("ev2", "order-1", 2)).unwrap();
        stream.append(make_event("ev3", "order-1", 3)).unwrap();
        let after = stream.events_after(1);
        assert_eq!(after.len(), 2);
    }

    #[test]
    fn test_event_store_append_and_get() {
        let mut store = EventStore::new();
        let events = vec![
            make_event("created", "order-1", 1),
            make_event("updated", "order-1", 2),
        ];
        store.append("order-1", "Order", events).unwrap();
        let retrieved = store.get_events("order-1").unwrap();
        assert_eq!(retrieved.len(), 2);
    }

    #[test]
    fn test_event_store_stream_not_found() {
        let store = EventStore::new();
        let result = store.get_events("missing");
        assert!(matches!(result, Err(DomainEventError::StreamNotFound(_))));
    }

    #[test]
    fn test_event_store_global_log() {
        let mut store = EventStore::new();
        store.append("s1", "A", vec![make_event("ev1", "s1", 1)]).unwrap();
        store.append("s2", "B", vec![make_event("ev2", "s2", 1)]).unwrap();
        assert_eq!(store.total_event_count(), 2);
        assert_eq!(store.stream_count(), 2);
    }

    #[test]
    fn test_event_store_events_by_type() {
        let mut store = EventStore::new();
        store.append("s1", "A", vec![
            make_event("created", "s1", 1),
            make_event("updated", "s1", 2),
        ]).unwrap();
        store.append("s2", "B", vec![
            make_event("created", "s2", 1),
        ]).unwrap();
        assert_eq!(store.events_by_type("created").len(), 2);
        assert_eq!(store.events_by_type("updated").len(), 1);
    }

    #[test]
    fn test_event_bus_publish_subscribe() {
        let mut bus = DomainEventBus::new();
        bus.subscribe(EventHandler::new(
            "h1",
            vec!["order_created".to_string()],
            |_event| HandlerResult::Ok,
        )).unwrap();
        let event = make_event("order_created", "order-1", 1);
        let results = bus.publish(&event);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, HandlerResult::Ok);
        assert_eq!(bus.published_count(), 1);
    }

    #[test]
    fn test_event_bus_unsubscribe() {
        let mut bus = DomainEventBus::new();
        bus.subscribe(EventHandler::new("h1", vec!["ev".to_string()], |_| HandlerResult::Ok)).unwrap();
        assert!(bus.unsubscribe("h1"));
        assert_eq!(bus.handler_count(), 0);
    }

    #[test]
    fn test_event_bus_duplicate_subscription() {
        let mut bus = DomainEventBus::new();
        bus.subscribe(EventHandler::new("h1", vec!["ev".to_string()], |_| HandlerResult::Ok)).unwrap();
        let result = bus.subscribe(EventHandler::new("h1", vec!["ev".to_string()], |_| HandlerResult::Ok));
        assert!(matches!(result, Err(DomainEventError::DuplicateSubscription(_))));
    }

    #[test]
    fn test_event_bus_error_tracking() {
        let mut bus = DomainEventBus::new();
        bus.subscribe(EventHandler::new(
            "h1",
            vec!["ev".to_string()],
            |_| HandlerResult::Error("fail".to_string()),
        )).unwrap();
        let event = make_event("ev", "agg-1", 1);
        bus.publish(&event);
        assert_eq!(bus.error_count(), 1);
    }

    #[test]
    fn test_event_upcaster() {
        let mut event = make_event("order_created", "order-1", 1);
        event.schema_version = 1;

        let upcaster = EventUpcaster::new("order_created", 1, 2, |event| {
            event.data.insert("new_field".to_string(), "default".to_string());
        });
        upcaster.upcast(&mut event);
        assert_eq!(event.schema_version, 2);
        assert_eq!(event.data.get("new_field").unwrap(), "default");
    }

    #[test]
    fn test_upcaster_registry_chain() {
        let mut registry = UpcasterRegistry::new();
        registry.register(EventUpcaster::new("ev", 1, 2, |event| {
            event.data.insert("v2_field".to_string(), "added".to_string());
        }));
        registry.register(EventUpcaster::new("ev", 2, 3, |event| {
            event.data.insert("v3_field".to_string(), "also_added".to_string());
        }));

        let mut event = make_event("ev", "agg-1", 1);
        event.schema_version = 1;
        registry.upcast(&mut event);
        assert_eq!(event.schema_version, 3);
        assert_eq!(event.data.get("v2_field").unwrap(), "added");
        assert_eq!(event.data.get("v3_field").unwrap(), "also_added");
    }

    #[test]
    fn test_replay_events() {
        let mut d1 = HashMap::new();
        d1.insert("item".to_string(), "widget".to_string());
        let mut d2 = HashMap::new();
        d2.insert("item".to_string(), "gadget".to_string());
        let events = vec![
            make_event_with_data("item_added", "order-1", 1, d1),
            make_event_with_data("item_added", "order-1", 2, d2),
        ];
        let state = replay_events(&events, |state, event| {
            let count: u32 = state.get("count").and_then(|c| c.parse().ok()).unwrap_or(0);
            state.insert("count".to_string(), (count + 1).to_string());
            if let Some(item) = event.data.get("item") {
                state.insert(format!("item_{count}"), item.clone());
            }
        });
        assert_eq!(state.get("count").unwrap(), "2");
    }

    #[test]
    fn test_replay_from_version() {
        let events = vec![
            make_event("ev1", "agg-1", 1),
            make_event("ev2", "agg-1", 2),
            make_event("ev3", "agg-1", 3),
        ];
        let mut initial = HashMap::new();
        initial.insert("count".to_string(), "1".to_string());
        let state = replay_from_version(&events, 1, |state, _event| {
            let c: u32 = state.get("count").and_then(|c| c.parse().ok()).unwrap_or(0);
            state.insert("count".to_string(), (c + 1).to_string());
        }, initial);
        assert_eq!(state.get("count").unwrap(), "3");
    }

    #[test]
    fn test_publish_all() {
        let mut bus = DomainEventBus::new();
        bus.subscribe(EventHandler::new("h1", vec!["ev".to_string()], |_| HandlerResult::Ok)).unwrap();
        let events = vec![make_event("ev", "a", 1), make_event("ev", "a", 2)];
        let results = bus.publish_all(&events);
        assert_eq!(results.len(), 2);
        assert_eq!(bus.published_count(), 2);
    }

    #[test]
    fn test_handler_skip() {
        let mut bus = DomainEventBus::new();
        bus.subscribe(EventHandler::new(
            "h1",
            vec!["ev".to_string()],
            |_| HandlerResult::Skip,
        )).unwrap();
        let event = make_event("ev", "agg-1", 1);
        let results = bus.publish(&event);
        assert_eq!(results[0].1, HandlerResult::Skip);
        // Skip is not an error.
        assert_eq!(bus.error_count(), 0);
    }
}
