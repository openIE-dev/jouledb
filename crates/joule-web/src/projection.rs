//! Event projections — projection handler, state rebuilding from events,
//! incremental projection, position tracking (last processed event), projection
//! reset, multiple projections per stream, and projection status.
//!
//! Replaces JS projection libraries (EventStoreDB projections, Axon projections)
//! with a pure-Rust projection engine that builds read-side state from events.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Projection errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    /// Projection not found.
    NotFound(String),
    /// Projection already exists.
    AlreadyExists(String),
    /// Handler error during projection.
    HandlerError { projection_id: String, event_type: String, reason: String },
    /// Projection is not running.
    NotRunning(String),
    /// Projection is already running.
    AlreadyRunning(String),
    /// No handler registered for event type.
    NoHandler { projection_id: String, event_type: String },
    /// Invalid state.
    InvalidState(String),
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "projection not found: {id}"),
            Self::AlreadyExists(id) => write!(f, "projection already exists: {id}"),
            Self::HandlerError { projection_id, event_type, reason } => {
                write!(f, "handler error in {projection_id} for {event_type}: {reason}")
            }
            Self::NotRunning(id) => write!(f, "projection {id} is not running"),
            Self::AlreadyRunning(id) => write!(f, "projection {id} is already running"),
            Self::NoHandler { projection_id, event_type } => {
                write!(f, "no handler in {projection_id} for {event_type}")
            }
            Self::InvalidState(msg) => write!(f, "invalid projection state: {msg}"),
        }
    }
}

impl std::error::Error for ProjectionError {}

// ── Projection Status ───────────────────────────────────────────

/// Status of a projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectionStatus {
    /// Created but not yet started.
    Created,
    /// Actively processing events.
    Running,
    /// Paused — retains position.
    Paused,
    /// Stopped — retains position.
    Stopped,
    /// Faulted — a handler returned an error.
    Faulted,
}

// ── Projection Event ────────────────────────────────────────────

/// An event to be processed by a projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionEvent {
    pub global_position: u64,
    pub stream_id: String,
    pub event_type: String,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl ProjectionEvent {
    pub fn new(
        global_position: u64,
        stream_id: impl Into<String>,
        event_type: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        Self {
            global_position,
            stream_id: stream_id.into(),
            event_type: event_type.into(),
            data,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

// ── Projection Handler ──────────────────────────────────────────

/// A handler function for a specific event type.
#[derive(Clone)]
pub struct ProjectionHandler {
    pub event_type: String,
    handler_fn: fn(&mut HashMap<String, serde_json::Value>, &ProjectionEvent) -> Result<(), String>,
}

impl ProjectionHandler {
    pub fn new(
        event_type: impl Into<String>,
        handler_fn: fn(&mut HashMap<String, serde_json::Value>, &ProjectionEvent) -> Result<(), String>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            handler_fn,
        }
    }

    pub fn handle(
        &self,
        state: &mut HashMap<String, serde_json::Value>,
        event: &ProjectionEvent,
    ) -> Result<(), String> {
        (self.handler_fn)(state, event)
    }
}

impl std::fmt::Debug for ProjectionHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectionHandler")
            .field("event_type", &self.event_type)
            .finish()
    }
}

// ── Projection Info ─────────────────────────────────────────────

/// Information about a projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionInfo {
    pub projection_id: String,
    pub status: ProjectionStatus,
    pub last_position: u64,
    pub events_processed: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error_message: Option<String>,
}

// ── Single Projection ───────────────────────────────────────────

/// A single projection that maintains state from events.
#[derive(Debug)]
pub struct Projection {
    pub id: String,
    pub status: ProjectionStatus,
    /// Last global position processed.
    pub last_position: u64,
    /// Count of events processed.
    pub events_processed: u64,
    /// The projected state.
    state: HashMap<String, serde_json::Value>,
    /// Handlers keyed by event type.
    handlers: Vec<ProjectionHandler>,
    /// Optional stream filter.
    stream_filter: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error_message: Option<String>,
}

impl Projection {
    pub fn new(id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            status: ProjectionStatus::Created,
            last_position: 0,
            events_processed: 0,
            state: HashMap::new(),
            handlers: Vec::new(),
            stream_filter: None,
            created_at: now,
            updated_at: now,
            error_message: None,
        }
    }

    /// Set a stream filter (only process events from this stream).
    pub fn with_stream_filter(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_filter = Some(stream_id.into());
        self
    }

    /// Register a handler.
    pub fn add_handler(&mut self, handler: ProjectionHandler) {
        self.handlers.push(handler);
    }

    /// Start the projection.
    pub fn start(&mut self) -> Result<(), ProjectionError> {
        match self.status {
            ProjectionStatus::Running => {
                return Err(ProjectionError::AlreadyRunning(self.id.clone()));
            }
            _ => {
                self.status = ProjectionStatus::Running;
                self.error_message = None;
                self.updated_at = Utc::now();
                Ok(())
            }
        }
    }

    /// Stop the projection.
    pub fn stop(&mut self) {
        self.status = ProjectionStatus::Stopped;
        self.updated_at = Utc::now();
    }

    /// Pause the projection.
    pub fn pause(&mut self) {
        self.status = ProjectionStatus::Paused;
        self.updated_at = Utc::now();
    }

    /// Process a single event.
    pub fn process_event(&mut self, event: &ProjectionEvent) -> Result<(), ProjectionError> {
        if self.status != ProjectionStatus::Running {
            return Err(ProjectionError::NotRunning(self.id.clone()));
        }

        // Stream filter check.
        if let Some(filter) = &self.stream_filter {
            if event.stream_id != *filter {
                return Ok(());
            }
        }

        // Skip if already processed.
        if event.global_position < self.last_position {
            return Ok(());
        }

        let event_type = event.event_type.clone();
        let projection_id = self.id.clone();

        // Find handler.
        let handler = self
            .handlers
            .iter()
            .find(|h| h.event_type == event_type);

        if let Some(handler) = handler {
            let result = handler.handle(&mut self.state, event);
            match result {
                Ok(()) => {
                    self.last_position = event.global_position + 1;
                    self.events_processed += 1;
                    self.updated_at = Utc::now();
                    Ok(())
                }
                Err(reason) => {
                    self.status = ProjectionStatus::Faulted;
                    self.error_message = Some(reason.clone());
                    self.updated_at = Utc::now();
                    Err(ProjectionError::HandlerError {
                        projection_id,
                        event_type,
                        reason,
                    })
                }
            }
        } else {
            // No handler — skip the event silently and advance position.
            self.last_position = event.global_position + 1;
            Ok(())
        }
    }

    /// Process a batch of events.
    pub fn process_events(&mut self, events: &[ProjectionEvent]) -> Result<u64, ProjectionError> {
        let mut count = 0;
        for event in events {
            self.process_event(event)?;
            count += 1;
        }
        Ok(count)
    }

    /// Reset projection to initial state.
    pub fn reset(&mut self) {
        self.state.clear();
        self.last_position = 0;
        self.events_processed = 0;
        self.status = ProjectionStatus::Created;
        self.error_message = None;
        self.updated_at = Utc::now();
    }

    /// Get the projected state.
    pub fn state(&self) -> &HashMap<String, serde_json::Value> {
        &self.state
    }

    /// Get a specific value from state.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.state.get(key)
    }

    /// Get projection info.
    pub fn info(&self) -> ProjectionInfo {
        ProjectionInfo {
            projection_id: self.id.clone(),
            status: self.status,
            last_position: self.last_position,
            events_processed: self.events_processed,
            created_at: self.created_at,
            updated_at: self.updated_at,
            error_message: self.error_message.clone(),
        }
    }

    /// Check if a handler is registered for a given event type.
    pub fn has_handler(&self, event_type: &str) -> bool {
        self.handlers.iter().any(|h| h.event_type == event_type)
    }

    /// Number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }
}

// ── Projection Manager ──────────────────────────────────────────

/// Manages multiple projections.
#[derive(Debug)]
pub struct ProjectionManager {
    projections: HashMap<String, Projection>,
}

impl ProjectionManager {
    pub fn new() -> Self {
        Self {
            projections: HashMap::new(),
        }
    }

    /// Register a new projection.
    pub fn register(&mut self, projection: Projection) -> Result<(), ProjectionError> {
        if self.projections.contains_key(&projection.id) {
            return Err(ProjectionError::AlreadyExists(projection.id.clone()));
        }
        self.projections.insert(projection.id.clone(), projection);
        Ok(())
    }

    /// Get a projection by ID.
    pub fn get(&self, id: &str) -> Option<&Projection> {
        self.projections.get(id)
    }

    /// Get a mutable reference to a projection.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Projection> {
        self.projections.get_mut(id)
    }

    /// Remove a projection.
    pub fn remove(&mut self, id: &str) -> Result<Projection, ProjectionError> {
        self.projections
            .remove(id)
            .ok_or_else(|| ProjectionError::NotFound(id.to_string()))
    }

    /// Dispatch an event to all running projections.
    pub fn dispatch(&mut self, event: &ProjectionEvent) -> Vec<(String, Result<(), ProjectionError>)> {
        let ids: Vec<String> = self.projections.keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            if let Some(proj) = self.projections.get_mut(&id) {
                if proj.status == ProjectionStatus::Running {
                    let result = proj.process_event(event);
                    results.push((id, result));
                }
            }
        }
        results
    }

    /// List all projection IDs.
    pub fn projection_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.projections.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Get info for all projections (sorted by id).
    pub fn all_info(&self) -> Vec<ProjectionInfo> {
        let mut ids: Vec<&String> = self.projections.keys().collect();
        ids.sort();
        ids.iter()
            .filter_map(|id| self.projections.get(*id).map(|p| p.info()))
            .collect()
    }

    /// Count projections by status.
    pub fn count_by_status(&self, status: ProjectionStatus) -> usize {
        self.projections.values().filter(|p| p.status == status).count()
    }

    /// Start all projections.
    pub fn start_all(&mut self) {
        for proj in self.projections.values_mut() {
            let _ = proj.start();
        }
    }

    /// Stop all projections.
    pub fn stop_all(&mut self) {
        for proj in self.projections.values_mut() {
            proj.stop();
        }
    }

    /// Reset all projections.
    pub fn reset_all(&mut self) {
        for proj in self.projections.values_mut() {
            proj.reset();
        }
    }

    /// Total projections count.
    pub fn count(&self) -> usize {
        self.projections.len()
    }
}

impl Default for ProjectionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn count_handler(
        state: &mut HashMap<String, serde_json::Value>,
        _event: &ProjectionEvent,
    ) -> Result<(), String> {
        let count = state
            .get("count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        state.insert("count".to_string(), serde_json::json!(count + 1));
        Ok(())
    }

    fn sum_handler(
        state: &mut HashMap<String, serde_json::Value>,
        event: &ProjectionEvent,
    ) -> Result<(), String> {
        let amount: i64 = event
            .data
            .get("amount")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let total = state
            .get("total")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        state.insert("total".to_string(), serde_json::json!(total + amount));
        Ok(())
    }

    fn fail_handler(
        _state: &mut HashMap<String, serde_json::Value>,
        _event: &ProjectionEvent,
    ) -> Result<(), String> {
        Err("deliberate failure".to_string())
    }

    fn make_event(pos: u64, stream: &str, event_type: &str) -> ProjectionEvent {
        ProjectionEvent::new(pos, stream, event_type, HashMap::new())
    }

    fn make_event_with_data(pos: u64, stream: &str, event_type: &str, key: &str, val: &str) -> ProjectionEvent {
        let mut data = HashMap::new();
        data.insert(key.to_string(), val.to_string());
        ProjectionEvent::new(pos, stream, event_type, data)
    }

    #[test]
    fn test_projection_basic_processing() {
        let mut proj = Projection::new("counter");
        proj.add_handler(ProjectionHandler::new("ItemAdded", count_handler));
        proj.start().unwrap();

        proj.process_event(&make_event(0, "s1", "ItemAdded")).unwrap();
        proj.process_event(&make_event(1, "s1", "ItemAdded")).unwrap();

        assert_eq!(proj.state().get("count").unwrap(), &serde_json::json!(2));
        assert_eq!(proj.events_processed, 2);
        assert_eq!(proj.last_position, 2);
    }

    #[test]
    fn test_projection_must_be_running() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));

        let err = proj.process_event(&make_event(0, "s1", "E")).unwrap_err();
        assert!(matches!(err, ProjectionError::NotRunning(_)));
    }

    #[test]
    fn test_projection_start_stop_pause() {
        let mut proj = Projection::new("p1");

        proj.start().unwrap();
        assert_eq!(proj.status, ProjectionStatus::Running);

        proj.pause();
        assert_eq!(proj.status, ProjectionStatus::Paused);

        proj.start().unwrap();
        assert_eq!(proj.status, ProjectionStatus::Running);

        proj.stop();
        assert_eq!(proj.status, ProjectionStatus::Stopped);
    }

    #[test]
    fn test_projection_double_start_error() {
        let mut proj = Projection::new("p1");
        proj.start().unwrap();
        let err = proj.start().unwrap_err();
        assert!(matches!(err, ProjectionError::AlreadyRunning(_)));
    }

    #[test]
    fn test_projection_fault_on_handler_error() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("Bad", fail_handler));
        proj.start().unwrap();

        let err = proj.process_event(&make_event(0, "s1", "Bad")).unwrap_err();
        assert!(matches!(err, ProjectionError::HandlerError { .. }));
        assert_eq!(proj.status, ProjectionStatus::Faulted);
        assert!(proj.error_message.is_some());
    }

    #[test]
    fn test_projection_reset() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));
        proj.start().unwrap();
        proj.process_event(&make_event(0, "s1", "E")).unwrap();
        proj.process_event(&make_event(1, "s1", "E")).unwrap();

        proj.reset();
        assert_eq!(proj.status, ProjectionStatus::Created);
        assert_eq!(proj.last_position, 0);
        assert_eq!(proj.events_processed, 0);
        assert!(proj.state().is_empty());
    }

    #[test]
    fn test_projection_stream_filter() {
        let mut proj = Projection::new("p1").with_stream_filter("s1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));
        proj.start().unwrap();

        proj.process_event(&make_event(0, "s1", "E")).unwrap();
        proj.process_event(&make_event(1, "s2", "E")).unwrap(); // Filtered out.
        proj.process_event(&make_event(2, "s1", "E")).unwrap();

        assert_eq!(proj.state().get("count").unwrap(), &serde_json::json!(2));
    }

    #[test]
    fn test_projection_skips_already_processed() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));
        proj.start().unwrap();

        proj.process_event(&make_event(0, "s1", "E")).unwrap();
        proj.process_event(&make_event(1, "s1", "E")).unwrap();

        // Re-process old events — should be skipped.
        proj.process_event(&make_event(0, "s1", "E")).unwrap();
        proj.process_event(&make_event(1, "s1", "E")).unwrap();

        assert_eq!(proj.events_processed, 2);
    }

    #[test]
    fn test_projection_unhandled_event_skipped() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("Known", count_handler));
        proj.start().unwrap();

        proj.process_event(&make_event(0, "s1", "Unknown")).unwrap(); // No handler, skip.
        assert_eq!(proj.events_processed, 0);
        assert_eq!(proj.last_position, 1); // Position still advances.
    }

    #[test]
    fn test_projection_process_batch() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));
        proj.start().unwrap();

        let events = vec![
            make_event(0, "s1", "E"),
            make_event(1, "s1", "E"),
            make_event(2, "s1", "E"),
        ];
        let count = proj.process_events(&events).unwrap();
        assert_eq!(count, 3);
        assert_eq!(proj.events_processed, 3);
    }

    #[test]
    fn test_projection_sum_handler() {
        let mut proj = Projection::new("totals");
        proj.add_handler(ProjectionHandler::new("Deposit", sum_handler));
        proj.start().unwrap();

        proj.process_event(&make_event_with_data(0, "s1", "Deposit", "amount", "100")).unwrap();
        proj.process_event(&make_event_with_data(1, "s1", "Deposit", "amount", "50")).unwrap();

        assert_eq!(proj.state().get("total").unwrap(), &serde_json::json!(150));
    }

    #[test]
    fn test_projection_info() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));
        proj.start().unwrap();
        proj.process_event(&make_event(0, "s1", "E")).unwrap();

        let info = proj.info();
        assert_eq!(info.projection_id, "p1");
        assert_eq!(info.status, ProjectionStatus::Running);
        assert_eq!(info.events_processed, 1);
        assert_eq!(info.last_position, 1);
    }

    #[test]
    fn test_projection_has_handler() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E1", count_handler));
        assert!(proj.has_handler("E1"));
        assert!(!proj.has_handler("E2"));
        assert_eq!(proj.handler_count(), 1);
    }

    #[test]
    fn test_manager_register_and_get() {
        let mut mgr = ProjectionManager::new();
        let proj = Projection::new("p1");
        mgr.register(proj).unwrap();
        assert!(mgr.get("p1").is_some());
        assert!(mgr.get("p2").is_none());
    }

    #[test]
    fn test_manager_duplicate_register() {
        let mut mgr = ProjectionManager::new();
        mgr.register(Projection::new("p1")).unwrap();
        let err = mgr.register(Projection::new("p1")).unwrap_err();
        assert!(matches!(err, ProjectionError::AlreadyExists(_)));
    }

    #[test]
    fn test_manager_dispatch_to_running() {
        let mut mgr = ProjectionManager::new();

        let mut p1 = Projection::new("p1");
        p1.add_handler(ProjectionHandler::new("E", count_handler));
        p1.start().unwrap();
        mgr.register(p1).unwrap();

        let mut p2 = Projection::new("p2");
        p2.add_handler(ProjectionHandler::new("E", count_handler));
        // p2 not started.
        mgr.register(p2).unwrap();

        let event = make_event(0, "s1", "E");
        let results = mgr.dispatch(&event);

        // Only p1 should have been dispatched to.
        assert_eq!(results.len(), 1);
        assert!(results.iter().all(|(_, r)| r.is_ok()));
    }

    #[test]
    fn test_manager_start_stop_all() {
        let mut mgr = ProjectionManager::new();
        mgr.register(Projection::new("p1")).unwrap();
        mgr.register(Projection::new("p2")).unwrap();

        mgr.start_all();
        assert_eq!(mgr.count_by_status(ProjectionStatus::Running), 2);

        mgr.stop_all();
        assert_eq!(mgr.count_by_status(ProjectionStatus::Stopped), 2);
    }

    #[test]
    fn test_manager_reset_all() {
        let mut mgr = ProjectionManager::new();
        let mut p = Projection::new("p1");
        p.add_handler(ProjectionHandler::new("E", count_handler));
        p.start().unwrap();
        p.process_event(&make_event(0, "s1", "E")).unwrap();
        mgr.register(p).unwrap();

        mgr.reset_all();
        let p = mgr.get("p1").unwrap();
        assert_eq!(p.events_processed, 0);
        assert_eq!(p.status, ProjectionStatus::Created);
    }

    #[test]
    fn test_manager_remove() {
        let mut mgr = ProjectionManager::new();
        mgr.register(Projection::new("p1")).unwrap();
        let removed = mgr.remove("p1").unwrap();
        assert_eq!(removed.id, "p1");
        assert!(mgr.get("p1").is_none());
    }

    #[test]
    fn test_manager_remove_not_found() {
        let mut mgr = ProjectionManager::new();
        let err = mgr.remove("ghost").unwrap_err();
        assert!(matches!(err, ProjectionError::NotFound(_)));
    }

    #[test]
    fn test_manager_projection_ids_sorted() {
        let mut mgr = ProjectionManager::new();
        mgr.register(Projection::new("zulu")).unwrap();
        mgr.register(Projection::new("alpha")).unwrap();
        mgr.register(Projection::new("mike")).unwrap();
        assert_eq!(mgr.projection_ids(), vec!["alpha", "mike", "zulu"]);
    }

    #[test]
    fn test_manager_all_info() {
        let mut mgr = ProjectionManager::new();
        mgr.register(Projection::new("b")).unwrap();
        mgr.register(Projection::new("a")).unwrap();

        let infos = mgr.all_info();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].projection_id, "a");
        assert_eq!(infos[1].projection_id, "b");
    }

    #[test]
    fn test_event_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        let event = ProjectionEvent::new(0, "s1", "E", HashMap::new()).with_metadata(meta);
        assert_eq!(event.metadata.get("source").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn test_manager_count() {
        let mut mgr = ProjectionManager::new();
        assert_eq!(mgr.count(), 0);
        mgr.register(Projection::new("a")).unwrap();
        mgr.register(Projection::new("b")).unwrap();
        assert_eq!(mgr.count(), 2);
    }

    #[test]
    fn test_projection_get_value() {
        let mut proj = Projection::new("p1");
        proj.add_handler(ProjectionHandler::new("E", count_handler));
        proj.start().unwrap();
        proj.process_event(&make_event(0, "s1", "E")).unwrap();
        assert_eq!(proj.get("count"), Some(&serde_json::json!(1)));
        assert_eq!(proj.get("missing"), None);
    }
}
