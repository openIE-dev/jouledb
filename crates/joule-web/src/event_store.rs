//! Event store — append-only stream, optimistic concurrency (expected version),
//! stream read (forward/backward), all-events global read, event metadata,
//! catch-up subscription, and soft stream deletion.
//!
//! Replaces JS event-store clients (EventStoreDB, Marten) with a pure-Rust
//! append-only event store that tracks every domain event with energy awareness.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Event store errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventStoreError {
    /// Stream not found.
    StreamNotFound(String),
    /// Optimistic concurrency conflict.
    VersionConflict { stream_id: String, expected: u64, actual: u64 },
    /// Stream has been soft-deleted.
    StreamDeleted(String),
    /// Event not found.
    EventNotFound { stream_id: String, position: u64 },
    /// Empty append — no events provided.
    EmptyAppend(String),
    /// Subscription not found.
    SubscriptionNotFound(String),
    /// Invalid read direction or bounds.
    InvalidRead(String),
}

impl std::fmt::Display for EventStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
            Self::VersionConflict { stream_id, expected, actual } => {
                write!(f, "version conflict on {stream_id}: expected {expected}, got {actual}")
            }
            Self::StreamDeleted(id) => write!(f, "stream deleted: {id}"),
            Self::EventNotFound { stream_id, position } => {
                write!(f, "event not found at position {position} in {stream_id}")
            }
            Self::EmptyAppend(id) => write!(f, "empty append to {id}"),
            Self::SubscriptionNotFound(id) => write!(f, "subscription not found: {id}"),
            Self::InvalidRead(msg) => write!(f, "invalid read: {msg}"),
        }
    }
}

impl std::error::Error for EventStoreError {}

// ── Read Direction ──────────────────────────────────────────────

/// Direction for reading events from a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReadDirection {
    Forward,
    Backward,
}

// ── Event Metadata ──────────────────────────────────────────────

/// Metadata attached to a stored event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventMetadata {
    pub correlation_id: Option<String>,
    pub causation_id: Option<String>,
    pub user_id: Option<String>,
    pub custom: HashMap<String, String>,
}

impl EventMetadata {
    pub fn new() -> Self {
        Self {
            correlation_id: None,
            causation_id: None,
            user_id: None,
            custom: HashMap::new(),
        }
    }

    pub fn with_correlation(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    pub fn with_causation(mut self, id: impl Into<String>) -> Self {
        self.causation_id = Some(id.into());
        self
    }

    pub fn with_user(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// ── Stored Event ────────────────────────────────────────────────

/// A persisted event in the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredEvent {
    /// Global sequential position across all streams.
    pub global_position: u64,
    /// Position within its stream.
    pub stream_position: u64,
    /// The stream this event belongs to.
    pub stream_id: String,
    /// Event type discriminator.
    pub event_type: String,
    /// Event data payload.
    pub data: HashMap<String, String>,
    /// Event metadata.
    pub metadata: EventMetadata,
    /// Timestamp of when the event was stored.
    pub timestamp: DateTime<Utc>,
}

// ── Stream Info ─────────────────────────────────────────────────

/// Information about a stored stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamInfo {
    pub stream_id: String,
    pub current_version: u64,
    pub event_count: u64,
    pub created_at: DateTime<Utc>,
    pub last_event_at: Option<DateTime<Utc>>,
    pub is_deleted: bool,
    pub deleted_at: Option<DateTime<Utc>>,
}

// ── Expected Version ────────────────────────────────────────────

/// Concurrency control on appends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedVersion {
    /// The stream must not exist yet.
    NoStream,
    /// The stream must be at exactly this version.
    Exact(u64),
    /// No concurrency check — always append.
    Any,
}

// ── Catch-Up Subscription ───────────────────────────────────────

/// A catch-up subscription that tracks its position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatchUpSubscription {
    pub subscription_id: String,
    /// Which stream to subscribe to, or None for all-stream.
    pub stream_id: Option<String>,
    /// Last processed global position.
    pub last_position: u64,
    /// Whether the subscription is active.
    pub active: bool,
    pub created_at: DateTime<Utc>,
}

impl CatchUpSubscription {
    pub fn new(subscription_id: impl Into<String>, stream_id: Option<String>) -> Self {
        Self {
            subscription_id: subscription_id.into(),
            stream_id,
            last_position: 0,
            active: true,
            created_at: Utc::now(),
        }
    }

    /// Advance the subscription position.
    pub fn advance(&mut self, position: u64) {
        if position > self.last_position {
            self.last_position = position;
        }
    }

    /// Pause the subscription.
    pub fn pause(&mut self) {
        self.active = false;
    }

    /// Resume the subscription.
    pub fn resume(&mut self) {
        self.active = true;
    }
}

// ── Append Result ───────────────────────────────────────────────

/// Result of a successful append operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendResult {
    pub stream_id: String,
    pub first_position: u64,
    pub last_position: u64,
    pub event_count: usize,
    pub next_expected_version: u64,
}

// ── Internal Stream ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Stream {
    info: StreamInfo,
    events: Vec<StoredEvent>,
}

// ── EventStore ──────────────────────────────────────────────────

/// Append-only event store with optimistic concurrency and subscriptions.
#[derive(Debug)]
pub struct EventStore {
    streams: HashMap<String, Stream>,
    /// Global ordered log of all events across all streams.
    global_log: Vec<StoredEvent>,
    /// Next global position counter.
    next_global_position: u64,
    /// Catch-up subscriptions.
    subscriptions: HashMap<String, CatchUpSubscription>,
}

impl EventStore {
    /// Create a new, empty event store.
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            global_log: Vec::new(),
            next_global_position: 0,
            subscriptions: HashMap::new(),
        }
    }

    /// Append events to a stream with optimistic concurrency.
    pub fn append(
        &mut self,
        stream_id: &str,
        expected: ExpectedVersion,
        events: Vec<(String, HashMap<String, String>, EventMetadata)>,
    ) -> Result<AppendResult, EventStoreError> {
        if events.is_empty() {
            return Err(EventStoreError::EmptyAppend(stream_id.to_string()));
        }

        // Check if stream is deleted.
        if let Some(stream) = self.streams.get(stream_id) {
            if stream.info.is_deleted {
                return Err(EventStoreError::StreamDeleted(stream_id.to_string()));
            }
        }

        // Optimistic concurrency check.
        let current_version = self
            .streams
            .get(stream_id)
            .map(|s| s.info.current_version)
            .unwrap_or(0);

        match expected {
            ExpectedVersion::NoStream => {
                if self.streams.contains_key(stream_id) {
                    return Err(EventStoreError::VersionConflict {
                        stream_id: stream_id.to_string(),
                        expected: 0,
                        actual: current_version,
                    });
                }
            }
            ExpectedVersion::Exact(v) => {
                if current_version != v {
                    return Err(EventStoreError::VersionConflict {
                        stream_id: stream_id.to_string(),
                        expected: v,
                        actual: current_version,
                    });
                }
            }
            ExpectedVersion::Any => {}
        }

        let now = Utc::now();
        let first_global = self.next_global_position;
        let event_count = events.len();

        // Ensure stream exists.
        if !self.streams.contains_key(stream_id) {
            self.streams.insert(
                stream_id.to_string(),
                Stream {
                    info: StreamInfo {
                        stream_id: stream_id.to_string(),
                        current_version: 0,
                        event_count: 0,
                        created_at: now,
                        last_event_at: None,
                        is_deleted: false,
                        deleted_at: None,
                    },
                    events: Vec::new(),
                },
            );
        }

        let stream = self.streams.get_mut(stream_id).unwrap();

        for (event_type, data, metadata) in events {
            stream.info.current_version += 1;
            stream.info.event_count += 1;
            stream.info.last_event_at = Some(now);

            let stored = StoredEvent {
                global_position: self.next_global_position,
                stream_position: stream.info.current_version,
                stream_id: stream_id.to_string(),
                event_type,
                data,
                metadata,
                timestamp: now,
            };

            stream.events.push(stored.clone());
            self.global_log.push(stored);
            self.next_global_position += 1;
        }

        let new_version = stream.info.current_version;

        Ok(AppendResult {
            stream_id: stream_id.to_string(),
            first_position: first_global,
            last_position: self.next_global_position - 1,
            event_count,
            next_expected_version: new_version,
        })
    }

    /// Read events from a stream.
    pub fn read_stream(
        &self,
        stream_id: &str,
        direction: ReadDirection,
        from_position: u64,
        count: usize,
    ) -> Result<Vec<StoredEvent>, EventStoreError> {
        let stream = self
            .streams
            .get(stream_id)
            .ok_or_else(|| EventStoreError::StreamNotFound(stream_id.to_string()))?;

        if stream.info.is_deleted {
            return Err(EventStoreError::StreamDeleted(stream_id.to_string()));
        }

        match direction {
            ReadDirection::Forward => {
                let events: Vec<StoredEvent> = stream
                    .events
                    .iter()
                    .filter(|e| e.stream_position >= from_position)
                    .take(count)
                    .cloned()
                    .collect();
                Ok(events)
            }
            ReadDirection::Backward => {
                let mut events: Vec<StoredEvent> = stream
                    .events
                    .iter()
                    .filter(|e| e.stream_position <= from_position)
                    .cloned()
                    .collect();
                events.reverse();
                events.truncate(count);
                Ok(events)
            }
        }
    }

    /// Read all events globally (across all streams) from a position.
    pub fn read_all(
        &self,
        from_position: u64,
        count: usize,
    ) -> Vec<StoredEvent> {
        self.global_log
            .iter()
            .filter(|e| e.global_position >= from_position)
            .take(count)
            .cloned()
            .collect()
    }

    /// Get stream information.
    pub fn stream_info(&self, stream_id: &str) -> Option<&StreamInfo> {
        self.streams.get(stream_id).map(|s| &s.info)
    }

    /// List all stream IDs.
    pub fn stream_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.streams.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Soft-delete a stream (events remain but stream is marked deleted).
    pub fn delete_stream(&mut self, stream_id: &str) -> Result<(), EventStoreError> {
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| EventStoreError::StreamNotFound(stream_id.to_string()))?;

        if stream.info.is_deleted {
            return Err(EventStoreError::StreamDeleted(stream_id.to_string()));
        }

        stream.info.is_deleted = true;
        stream.info.deleted_at = Some(Utc::now());
        Ok(())
    }

    /// Restore a soft-deleted stream.
    pub fn restore_stream(&mut self, stream_id: &str) -> Result<(), EventStoreError> {
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| EventStoreError::StreamNotFound(stream_id.to_string()))?;

        if !stream.info.is_deleted {
            return Ok(());
        }

        stream.info.is_deleted = false;
        stream.info.deleted_at = None;
        Ok(())
    }

    /// Total number of events across all streams.
    pub fn total_event_count(&self) -> u64 {
        self.next_global_position
    }

    /// Get a single event by stream and position.
    pub fn get_event(
        &self,
        stream_id: &str,
        position: u64,
    ) -> Result<StoredEvent, EventStoreError> {
        let stream = self
            .streams
            .get(stream_id)
            .ok_or_else(|| EventStoreError::StreamNotFound(stream_id.to_string()))?;

        stream
            .events
            .iter()
            .find(|e| e.stream_position == position)
            .cloned()
            .ok_or(EventStoreError::EventNotFound {
                stream_id: stream_id.to_string(),
                position,
            })
    }

    // ── Subscriptions ───────────────────────────────────────────

    /// Create a catch-up subscription.
    pub fn create_subscription(
        &mut self,
        subscription_id: impl Into<String>,
        stream_id: Option<String>,
    ) -> CatchUpSubscription {
        let sub = CatchUpSubscription::new(subscription_id, stream_id);
        self.subscriptions
            .insert(sub.subscription_id.clone(), sub.clone());
        sub
    }

    /// Poll a subscription for new events since its last position.
    pub fn poll_subscription(
        &mut self,
        subscription_id: &str,
        max_count: usize,
    ) -> Result<Vec<StoredEvent>, EventStoreError> {
        let sub = self
            .subscriptions
            .get(subscription_id)
            .ok_or_else(|| {
                EventStoreError::SubscriptionNotFound(subscription_id.to_string())
            })?;

        if !sub.active {
            return Ok(Vec::new());
        }

        let from = sub.last_position;
        let stream_filter = sub.stream_id.clone();

        let events: Vec<StoredEvent> = self
            .global_log
            .iter()
            .filter(|e| e.global_position >= from)
            .filter(|e| {
                stream_filter
                    .as_ref()
                    .map(|sid| e.stream_id == *sid)
                    .unwrap_or(true)
            })
            .take(max_count)
            .cloned()
            .collect();

        // Advance subscription position.
        if let Some(last) = events.last() {
            let new_pos = last.global_position + 1;
            if let Some(sub) = self.subscriptions.get_mut(subscription_id) {
                sub.advance(new_pos);
            }
        }

        Ok(events)
    }

    /// Get subscription state.
    pub fn subscription(&self, subscription_id: &str) -> Option<&CatchUpSubscription> {
        self.subscriptions.get(subscription_id)
    }

    /// Pause a subscription.
    pub fn pause_subscription(
        &mut self,
        subscription_id: &str,
    ) -> Result<(), EventStoreError> {
        let sub = self
            .subscriptions
            .get_mut(subscription_id)
            .ok_or_else(|| {
                EventStoreError::SubscriptionNotFound(subscription_id.to_string())
            })?;
        sub.pause();
        Ok(())
    }

    /// Resume a subscription.
    pub fn resume_subscription(
        &mut self,
        subscription_id: &str,
    ) -> Result<(), EventStoreError> {
        let sub = self
            .subscriptions
            .get_mut(subscription_id)
            .ok_or_else(|| {
                EventStoreError::SubscriptionNotFound(subscription_id.to_string())
            })?;
        sub.resume();
        Ok(())
    }

    /// Remove a subscription.
    pub fn remove_subscription(
        &mut self,
        subscription_id: &str,
    ) -> Result<(), EventStoreError> {
        self.subscriptions
            .remove(subscription_id)
            .map(|_| ())
            .ok_or_else(|| {
                EventStoreError::SubscriptionNotFound(subscription_id.to_string())
            })
    }

    /// Count of active streams (non-deleted).
    pub fn active_stream_count(&self) -> usize {
        self.streams.values().filter(|s| !s.info.is_deleted).count()
    }

    /// Read events of a specific type from the global log.
    pub fn read_by_event_type(
        &self,
        event_type: &str,
        from_position: u64,
        count: usize,
    ) -> Vec<StoredEvent> {
        self.global_log
            .iter()
            .filter(|e| e.global_position >= from_position && e.event_type == event_type)
            .take(count)
            .cloned()
            .collect()
    }
}

impl Default for EventStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(
        event_type: &str,
        key: &str,
        val: &str,
    ) -> (String, HashMap<String, String>, EventMetadata) {
        let mut data = HashMap::new();
        data.insert(key.to_string(), val.to_string());
        (event_type.to_string(), data, EventMetadata::new())
    }

    #[test]
    fn test_append_creates_stream() {
        let mut store = EventStore::new();
        let result = store
            .append("stream-1", ExpectedVersion::NoStream, vec![make_event("Created", "name", "alice")])
            .unwrap();
        assert_eq!(result.stream_id, "stream-1");
        assert_eq!(result.event_count, 1);
        assert_eq!(result.next_expected_version, 1);
    }

    #[test]
    fn test_append_multiple_events() {
        let mut store = EventStore::new();
        let events = vec![
            make_event("Created", "name", "alice"),
            make_event("Updated", "name", "bob"),
            make_event("Deleted", "reason", "inactive"),
        ];
        let result = store.append("s1", ExpectedVersion::NoStream, events).unwrap();
        assert_eq!(result.event_count, 3);
        assert_eq!(result.first_position, 0);
        assert_eq!(result.last_position, 2);
        assert_eq!(result.next_expected_version, 3);
    }

    #[test]
    fn test_optimistic_concurrency_exact() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::NoStream, vec![make_event("E1", "k", "v")]).unwrap();
        // Version is now 1, append expecting 1 should succeed.
        store.append("s1", ExpectedVersion::Exact(1), vec![make_event("E2", "k", "v")]).unwrap();
        // Expecting 1 again should fail (version is now 2).
        let err = store
            .append("s1", ExpectedVersion::Exact(1), vec![make_event("E3", "k", "v")])
            .unwrap_err();
        assert!(matches!(err, EventStoreError::VersionConflict { .. }));
    }

    #[test]
    fn test_no_stream_conflict() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::NoStream, vec![make_event("E1", "k", "v")]).unwrap();
        let err = store
            .append("s1", ExpectedVersion::NoStream, vec![make_event("E2", "k", "v")])
            .unwrap_err();
        assert!(matches!(err, EventStoreError::VersionConflict { .. }));
    }

    #[test]
    fn test_any_version_always_succeeds() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "v")]).unwrap();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E2", "k", "v")]).unwrap();
        assert_eq!(store.stream_info("s1").unwrap().current_version, 2);
    }

    #[test]
    fn test_empty_append_error() {
        let mut store = EventStore::new();
        let err = store.append("s1", ExpectedVersion::Any, vec![]).unwrap_err();
        assert!(matches!(err, EventStoreError::EmptyAppend(_)));
    }

    #[test]
    fn test_read_stream_forward() {
        let mut store = EventStore::new();
        let events = vec![
            make_event("E1", "k", "1"),
            make_event("E2", "k", "2"),
            make_event("E3", "k", "3"),
        ];
        store.append("s1", ExpectedVersion::NoStream, events).unwrap();
        let read = store.read_stream("s1", ReadDirection::Forward, 1, 10).unwrap();
        assert_eq!(read.len(), 3);
        assert_eq!(read[0].event_type, "E1");
        assert_eq!(read[2].event_type, "E3");
    }

    #[test]
    fn test_read_stream_forward_with_offset() {
        let mut store = EventStore::new();
        let events = vec![
            make_event("E1", "k", "1"),
            make_event("E2", "k", "2"),
            make_event("E3", "k", "3"),
        ];
        store.append("s1", ExpectedVersion::NoStream, events).unwrap();
        let read = store.read_stream("s1", ReadDirection::Forward, 2, 10).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].event_type, "E2");
    }

    #[test]
    fn test_read_stream_backward() {
        let mut store = EventStore::new();
        let events = vec![
            make_event("E1", "k", "1"),
            make_event("E2", "k", "2"),
            make_event("E3", "k", "3"),
        ];
        store.append("s1", ExpectedVersion::NoStream, events).unwrap();
        let read = store.read_stream("s1", ReadDirection::Backward, 3, 2).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].event_type, "E3");
        assert_eq!(read[1].event_type, "E2");
    }

    #[test]
    fn test_read_stream_not_found() {
        let store = EventStore::new();
        let err = store.read_stream("missing", ReadDirection::Forward, 0, 10).unwrap_err();
        assert!(matches!(err, EventStoreError::StreamNotFound(_)));
    }

    #[test]
    fn test_read_all_global() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "1")]).unwrap();
        store.append("s2", ExpectedVersion::Any, vec![make_event("E2", "k", "2")]).unwrap();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E3", "k", "3")]).unwrap();

        let all = store.read_all(0, 100);
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].stream_id, "s1");
        assert_eq!(all[1].stream_id, "s2");
        assert_eq!(all[2].stream_id, "s1");
    }

    #[test]
    fn test_soft_delete_stream() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "v")]).unwrap();
        store.delete_stream("s1").unwrap();

        let info = store.stream_info("s1").unwrap();
        assert!(info.is_deleted);
        assert!(info.deleted_at.is_some());

        // Read from deleted stream fails.
        let err = store.read_stream("s1", ReadDirection::Forward, 0, 10).unwrap_err();
        assert!(matches!(err, EventStoreError::StreamDeleted(_)));

        // Append to deleted stream fails.
        let err = store.append("s1", ExpectedVersion::Any, vec![make_event("E2", "k", "v")]).unwrap_err();
        assert!(matches!(err, EventStoreError::StreamDeleted(_)));
    }

    #[test]
    fn test_restore_deleted_stream() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "v")]).unwrap();
        store.delete_stream("s1").unwrap();
        store.restore_stream("s1").unwrap();

        let info = store.stream_info("s1").unwrap();
        assert!(!info.is_deleted);

        // Can read again.
        let read = store.read_stream("s1", ReadDirection::Forward, 0, 10).unwrap();
        assert_eq!(read.len(), 1);
    }

    #[test]
    fn test_delete_nonexistent_stream() {
        let mut store = EventStore::new();
        let err = store.delete_stream("missing").unwrap_err();
        assert!(matches!(err, EventStoreError::StreamNotFound(_)));
    }

    #[test]
    fn test_event_metadata() {
        let mut store = EventStore::new();
        let meta = EventMetadata::new()
            .with_correlation("corr-1")
            .with_causation("cause-1")
            .with_user("user-42")
            .with_custom("source", "test");
        let mut data = HashMap::new();
        data.insert("k".to_string(), "v".to_string());
        store
            .append("s1", ExpectedVersion::Any, vec![("E1".to_string(), data, meta)])
            .unwrap();

        let event = store.get_event("s1", 1).unwrap();
        assert_eq!(event.metadata.correlation_id.as_deref(), Some("corr-1"));
        assert_eq!(event.metadata.causation_id.as_deref(), Some("cause-1"));
        assert_eq!(event.metadata.user_id.as_deref(), Some("user-42"));
        assert_eq!(event.metadata.custom.get("source").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn test_catch_up_subscription() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "1")]).unwrap();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E2", "k", "2")]).unwrap();

        store.create_subscription("sub-1", Some("s1".to_string()));

        // Poll gets existing events.
        let events = store.poll_subscription("sub-1", 10).unwrap();
        assert_eq!(events.len(), 2);

        // No new events.
        let events = store.poll_subscription("sub-1", 10).unwrap();
        assert_eq!(events.len(), 0);

        // Append new event, then poll.
        store.append("s1", ExpectedVersion::Any, vec![make_event("E3", "k", "3")]).unwrap();
        let events = store.poll_subscription("sub-1", 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "E3");
    }

    #[test]
    fn test_subscription_stream_filter() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "1")]).unwrap();
        store.append("s2", ExpectedVersion::Any, vec![make_event("E2", "k", "2")]).unwrap();

        store.create_subscription("sub-s1", Some("s1".to_string()));
        let events = store.poll_subscription("sub-s1", 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].stream_id, "s1");
    }

    #[test]
    fn test_global_subscription() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "1")]).unwrap();
        store.append("s2", ExpectedVersion::Any, vec![make_event("E2", "k", "2")]).unwrap();

        store.create_subscription("sub-all", None);
        let events = store.poll_subscription("sub-all", 10).unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_pause_resume_subscription() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "1")]).unwrap();
        store.create_subscription("sub-1", None);

        store.pause_subscription("sub-1").unwrap();
        let events = store.poll_subscription("sub-1", 10).unwrap();
        assert!(events.is_empty()); // Paused returns nothing.

        store.resume_subscription("sub-1").unwrap();
        let events = store.poll_subscription("sub-1", 10).unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_remove_subscription() {
        let mut store = EventStore::new();
        store.create_subscription("sub-1", None);
        store.remove_subscription("sub-1").unwrap();
        let err = store.poll_subscription("sub-1", 10).unwrap_err();
        assert!(matches!(err, EventStoreError::SubscriptionNotFound(_)));
    }

    #[test]
    fn test_get_event_not_found() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "v")]).unwrap();
        let err = store.get_event("s1", 999).unwrap_err();
        assert!(matches!(err, EventStoreError::EventNotFound { .. }));
    }

    #[test]
    fn test_total_event_count() {
        let mut store = EventStore::new();
        assert_eq!(store.total_event_count(), 0);
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "v")]).unwrap();
        store.append("s2", ExpectedVersion::Any, vec![
            make_event("E2", "k", "v"),
            make_event("E3", "k", "v"),
        ]).unwrap();
        assert_eq!(store.total_event_count(), 3);
    }

    #[test]
    fn test_read_by_event_type() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![
            make_event("Created", "k", "1"),
            make_event("Updated", "k", "2"),
            make_event("Created", "k", "3"),
        ]).unwrap();

        let created = store.read_by_event_type("Created", 0, 100);
        assert_eq!(created.len(), 2);
        for e in &created {
            assert_eq!(e.event_type, "Created");
        }
    }

    #[test]
    fn test_stream_ids_sorted() {
        let mut store = EventStore::new();
        store.append("zulu", ExpectedVersion::Any, vec![make_event("E", "k", "v")]).unwrap();
        store.append("alpha", ExpectedVersion::Any, vec![make_event("E", "k", "v")]).unwrap();
        store.append("mike", ExpectedVersion::Any, vec![make_event("E", "k", "v")]).unwrap();
        let ids = store.stream_ids();
        assert_eq!(ids, vec!["alpha", "mike", "zulu"]);
    }

    #[test]
    fn test_active_stream_count() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E", "k", "v")]).unwrap();
        store.append("s2", ExpectedVersion::Any, vec![make_event("E", "k", "v")]).unwrap();
        assert_eq!(store.active_stream_count(), 2);
        store.delete_stream("s1").unwrap();
        assert_eq!(store.active_stream_count(), 1);
    }

    #[test]
    fn test_global_position_monotonic() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E1", "k", "v")]).unwrap();
        store.append("s2", ExpectedVersion::Any, vec![make_event("E2", "k", "v")]).unwrap();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E3", "k", "v")]).unwrap();
        let all = store.read_all(0, 100);
        for (i, event) in all.iter().enumerate() {
            assert_eq!(event.global_position, i as u64);
        }
    }

    #[test]
    fn test_read_count_limit() {
        let mut store = EventStore::new();
        for i in 0..10 {
            let label = format!("E{i}");
            store.append("s1", ExpectedVersion::Any, vec![make_event(&label, "k", "v")]).unwrap();
        }
        let read = store.read_stream("s1", ReadDirection::Forward, 1, 3).unwrap();
        assert_eq!(read.len(), 3);

        let all = store.read_all(0, 5);
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_subscription_not_found() {
        let mut store = EventStore::new();
        let err = store.pause_subscription("ghost").unwrap_err();
        assert!(matches!(err, EventStoreError::SubscriptionNotFound(_)));
    }

    #[test]
    fn test_double_delete_error() {
        let mut store = EventStore::new();
        store.append("s1", ExpectedVersion::Any, vec![make_event("E", "k", "v")]).unwrap();
        store.delete_stream("s1").unwrap();
        let err = store.delete_stream("s1").unwrap_err();
        assert!(matches!(err, EventStoreError::StreamDeleted(_)));
    }

    #[test]
    fn test_default_metadata() {
        let meta = EventMetadata::default();
        assert!(meta.correlation_id.is_none());
        assert!(meta.causation_id.is_none());
        assert!(meta.user_id.is_none());
        assert!(meta.custom.is_empty());
    }
}
