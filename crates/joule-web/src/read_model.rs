//! Read model builder — denormalized view construction, event handler mapping,
//! view query API, view consistency (eventual), materialized view refresh,
//! view versioning, and multi-view from same events.
//!
//! Replaces JS read-model libraries (EventStoreDB projections, Sequelize views)
//! with a pure-Rust read model builder for CQRS query-side views.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Read model errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadModelError {
    /// View not found.
    ViewNotFound(String),
    /// View already exists.
    ViewAlreadyExists(String),
    /// Handler error.
    HandlerError { view_id: String, event_type: String, reason: String },
    /// View not ready (still catching up).
    ViewNotReady(String),
    /// Query error.
    QueryError(String),
    /// View version mismatch.
    VersionMismatch { view_id: String, expected: u64, actual: u64 },
}

impl std::fmt::Display for ReadModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ViewNotFound(id) => write!(f, "view not found: {id}"),
            Self::ViewAlreadyExists(id) => write!(f, "view already exists: {id}"),
            Self::HandlerError { view_id, event_type, reason } => {
                write!(f, "handler error in view {view_id} for {event_type}: {reason}")
            }
            Self::ViewNotReady(id) => write!(f, "view {id} not ready"),
            Self::QueryError(msg) => write!(f, "query error: {msg}"),
            Self::VersionMismatch { view_id, expected, actual } => {
                write!(f, "version mismatch for {view_id}: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for ReadModelError {}

// ── View Status ─────────────────────────────────────────────────

/// Status of a materialized view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewStatus {
    /// View created, not yet populated.
    Created,
    /// View is catching up with events.
    Building,
    /// View is up-to-date and ready for queries.
    Ready,
    /// View needs refresh (stale).
    Stale,
    /// View encountered an error.
    Faulted,
}

// ── Read Model Event ────────────────────────────────────────────

/// An event consumed by the read model builder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadModelEvent {
    pub global_position: u64,
    pub stream_id: String,
    pub event_type: String,
    pub data: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl ReadModelEvent {
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
            timestamp: Utc::now(),
        }
    }
}

// ── View Row ────────────────────────────────────────────────────

/// A single row/record in a materialized view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewRow {
    pub id: String,
    pub fields: HashMap<String, serde_json::Value>,
    pub updated_at: DateTime<Utc>,
}

impl ViewRow {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: HashMap::new(),
            updated_at: Utc::now(),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.fields.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.fields.get(key)
    }
}

// ── View Handler ────────────────────────────────────────────────

/// A handler that transforms events into view rows.
#[derive(Clone)]
pub struct ViewHandler {
    pub event_type: String,
    handler_fn: fn(&mut ViewData, &ReadModelEvent) -> Result<(), String>,
}

impl ViewHandler {
    pub fn new(
        event_type: impl Into<String>,
        handler_fn: fn(&mut ViewData, &ReadModelEvent) -> Result<(), String>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            handler_fn,
        }
    }

    pub fn handle(&self, data: &mut ViewData, event: &ReadModelEvent) -> Result<(), String> {
        (self.handler_fn)(data, event)
    }
}

impl std::fmt::Debug for ViewHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewHandler")
            .field("event_type", &self.event_type)
            .finish()
    }
}

// ── View Data ───────────────────────────────────────────────────

/// The underlying data store for a view (rows keyed by ID).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ViewData {
    rows: HashMap<String, ViewRow>,
}

impl ViewData {
    pub fn new() -> Self {
        Self {
            rows: HashMap::new(),
        }
    }

    /// Insert or update a row.
    pub fn upsert(&mut self, row: ViewRow) {
        self.rows.insert(row.id.clone(), row);
    }

    /// Get a row by ID.
    pub fn get(&self, id: &str) -> Option<&ViewRow> {
        self.rows.get(id)
    }

    /// Get a mutable row by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ViewRow> {
        self.rows.get_mut(id)
    }

    /// Remove a row.
    pub fn remove(&mut self, id: &str) -> Option<ViewRow> {
        self.rows.remove(id)
    }

    /// Count rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get all row IDs (sorted).
    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.rows.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Get all rows (sorted by ID for deterministic output).
    pub fn all_rows(&self) -> Vec<&ViewRow> {
        let mut ids: Vec<&String> = self.rows.keys().collect();
        ids.sort();
        ids.iter().filter_map(|id| self.rows.get(*id)).collect()
    }

    /// Query rows matching a predicate.
    pub fn query<F>(&self, predicate: F) -> Vec<&ViewRow>
    where
        F: Fn(&ViewRow) -> bool,
    {
        let mut ids: Vec<&String> = self.rows.keys().collect();
        ids.sort();
        ids.iter()
            .filter_map(|id| self.rows.get(*id))
            .filter(|row| predicate(row))
            .collect()
    }

    /// Clear all rows.
    pub fn clear(&mut self) {
        self.rows.clear();
    }
}

// ── Materialized View ───────────────────────────────────────────

/// A materialized view built from events.
#[derive(Debug)]
pub struct MaterializedView {
    pub view_id: String,
    pub status: ViewStatus,
    pub version: u64,
    pub last_position: u64,
    pub events_processed: u64,
    pub data: ViewData,
    handlers: Vec<ViewHandler>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error_message: Option<String>,
}

impl MaterializedView {
    pub fn new(view_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            view_id: view_id.into(),
            status: ViewStatus::Created,
            version: 1,
            last_position: 0,
            events_processed: 0,
            data: ViewData::new(),
            handlers: Vec::new(),
            created_at: now,
            updated_at: now,
            error_message: None,
        }
    }

    /// Register an event handler.
    pub fn add_handler(&mut self, handler: ViewHandler) {
        self.handlers.push(handler);
    }

    /// Start building (mark as Building).
    pub fn start_building(&mut self) {
        self.status = ViewStatus::Building;
        self.updated_at = Utc::now();
    }

    /// Mark as ready.
    pub fn mark_ready(&mut self) {
        self.status = ViewStatus::Ready;
        self.updated_at = Utc::now();
    }

    /// Mark as stale (needing refresh).
    pub fn mark_stale(&mut self) {
        self.status = ViewStatus::Stale;
        self.updated_at = Utc::now();
    }

    /// Apply a single event.
    pub fn apply_event(&mut self, event: &ReadModelEvent) -> Result<(), ReadModelError> {
        // Skip already-processed events.
        if event.global_position < self.last_position {
            return Ok(());
        }

        let event_type = event.event_type.clone();

        // Find handler.
        let handler = self
            .handlers
            .iter()
            .find(|h| h.event_type == event_type);

        if let Some(handler) = handler {
            let result = handler.handle(&mut self.data, event);
            match result {
                Ok(()) => {
                    self.last_position = event.global_position + 1;
                    self.events_processed += 1;
                    self.updated_at = Utc::now();
                    Ok(())
                }
                Err(reason) => {
                    self.status = ViewStatus::Faulted;
                    self.error_message = Some(reason.clone());
                    self.updated_at = Utc::now();
                    Err(ReadModelError::HandlerError {
                        view_id: self.view_id.clone(),
                        event_type,
                        reason,
                    })
                }
            }
        } else {
            // No handler — skip and advance.
            self.last_position = event.global_position + 1;
            Ok(())
        }
    }

    /// Apply a batch of events.
    pub fn apply_events(&mut self, events: &[ReadModelEvent]) -> Result<u64, ReadModelError> {
        let mut count = 0;
        for event in events {
            self.apply_event(event)?;
            count += 1;
        }
        Ok(count)
    }

    /// Refresh: clear data and mark as stale for re-processing.
    pub fn refresh(&mut self) {
        self.data.clear();
        self.last_position = 0;
        self.events_processed = 0;
        self.version += 1;
        self.status = ViewStatus::Stale;
        self.error_message = None;
        self.updated_at = Utc::now();
    }

    /// Query the view (only if Ready).
    pub fn query_rows<F>(&self, predicate: F) -> Result<Vec<&ViewRow>, ReadModelError>
    where
        F: Fn(&ViewRow) -> bool,
    {
        if self.status == ViewStatus::Faulted {
            return Err(ReadModelError::ViewNotReady(self.view_id.clone()));
        }
        Ok(self.data.query(predicate))
    }

    /// Get a specific row by ID.
    pub fn get_row(&self, id: &str) -> Option<&ViewRow> {
        self.data.get(id)
    }

    /// Row count.
    pub fn row_count(&self) -> usize {
        self.data.len()
    }

    /// Check if a handler is registered.
    pub fn has_handler(&self, event_type: &str) -> bool {
        self.handlers.iter().any(|h| h.event_type == event_type)
    }
}

// ── Read Model Builder ──────────────────────────────────────────

/// Manages multiple materialized views.
#[derive(Debug)]
pub struct ReadModelBuilder {
    views: HashMap<String, MaterializedView>,
}

impl ReadModelBuilder {
    pub fn new() -> Self {
        Self {
            views: HashMap::new(),
        }
    }

    /// Register a view.
    pub fn register_view(&mut self, view: MaterializedView) -> Result<(), ReadModelError> {
        if self.views.contains_key(&view.view_id) {
            return Err(ReadModelError::ViewAlreadyExists(view.view_id.clone()));
        }
        self.views.insert(view.view_id.clone(), view);
        Ok(())
    }

    /// Get a view.
    pub fn get_view(&self, view_id: &str) -> Option<&MaterializedView> {
        self.views.get(view_id)
    }

    /// Get a mutable view.
    pub fn get_view_mut(&mut self, view_id: &str) -> Option<&mut MaterializedView> {
        self.views.get_mut(view_id)
    }

    /// Remove a view.
    pub fn remove_view(&mut self, view_id: &str) -> Result<MaterializedView, ReadModelError> {
        self.views
            .remove(view_id)
            .ok_or_else(|| ReadModelError::ViewNotFound(view_id.to_string()))
    }

    /// Dispatch an event to all views.
    pub fn dispatch(&mut self, event: &ReadModelEvent) -> Vec<(String, Result<(), ReadModelError>)> {
        let ids: Vec<String> = self.views.keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            if let Some(view) = self.views.get_mut(&id) {
                let result = view.apply_event(event);
                results.push((id, result));
            }
        }
        results
    }

    /// Dispatch a batch of events to all views.
    pub fn dispatch_batch(&mut self, events: &[ReadModelEvent]) -> Vec<(String, Result<u64, ReadModelError>)> {
        let ids: Vec<String> = self.views.keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            if let Some(view) = self.views.get_mut(&id) {
                let result = view.apply_events(events);
                results.push((id, result));
            }
        }
        results
    }

    /// List all view IDs (sorted).
    pub fn view_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.views.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Count views.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Refresh all views.
    pub fn refresh_all(&mut self) {
        for view in self.views.values_mut() {
            view.refresh();
        }
    }

    /// Count views by status.
    pub fn count_by_status(&self, status: ViewStatus) -> usize {
        self.views.values().filter(|v| v.status == status).count()
    }
}

impl Default for ReadModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn user_created_handler(data: &mut ViewData, event: &ReadModelEvent) -> Result<(), String> {
        let user_id = event.data.get("user_id").cloned().unwrap_or_default();
        let name = event.data.get("name").cloned().unwrap_or_default();
        let mut row = ViewRow::new(&user_id);
        row.set("name", serde_json::json!(name));
        row.set("active", serde_json::json!(true));
        data.upsert(row);
        Ok(())
    }

    fn user_updated_handler(data: &mut ViewData, event: &ReadModelEvent) -> Result<(), String> {
        let user_id = event.data.get("user_id").cloned().unwrap_or_default();
        if let Some(row) = data.get_mut(&user_id) {
            if let Some(name) = event.data.get("name") {
                row.set("name", serde_json::json!(name));
            }
            Ok(())
        } else {
            Err(format!("user not found: {user_id}"))
        }
    }

    fn user_deleted_handler(data: &mut ViewData, event: &ReadModelEvent) -> Result<(), String> {
        let user_id = event.data.get("user_id").cloned().unwrap_or_default();
        data.remove(&user_id);
        Ok(())
    }

    fn fail_handler(_data: &mut ViewData, _event: &ReadModelEvent) -> Result<(), String> {
        Err("deliberate failure".to_string())
    }

    fn make_event(pos: u64, event_type: &str, kv: &[(&str, &str)]) -> ReadModelEvent {
        let mut data = HashMap::new();
        for (k, v) in kv {
            data.insert(k.to_string(), v.to_string());
        }
        ReadModelEvent::new(pos, "stream-1", event_type, data)
    }

    fn make_user_view() -> MaterializedView {
        let mut view = MaterializedView::new("users");
        view.add_handler(ViewHandler::new("UserCreated", user_created_handler));
        view.add_handler(ViewHandler::new("UserUpdated", user_updated_handler));
        view.add_handler(ViewHandler::new("UserDeleted", user_deleted_handler));
        view
    }

    #[test]
    fn test_view_build_and_query() {
        let mut view = make_user_view();
        view.start_building();

        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")])).unwrap();
        view.apply_event(&make_event(1, "UserCreated", &[("user_id", "u2"), ("name", "Bob")])).unwrap();
        view.mark_ready();

        assert_eq!(view.row_count(), 2);
        let row = view.get_row("u1").unwrap();
        assert_eq!(row.get("name").unwrap(), &serde_json::json!("Alice"));
    }

    #[test]
    fn test_view_update_event() {
        let mut view = make_user_view();
        view.start_building();

        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")])).unwrap();
        view.apply_event(&make_event(1, "UserUpdated", &[("user_id", "u1"), ("name", "Alicia")])).unwrap();

        let row = view.get_row("u1").unwrap();
        assert_eq!(row.get("name").unwrap(), &serde_json::json!("Alicia"));
    }

    #[test]
    fn test_view_delete_event() {
        let mut view = make_user_view();
        view.start_building();

        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")])).unwrap();
        view.apply_event(&make_event(1, "UserDeleted", &[("user_id", "u1")])).unwrap();

        assert_eq!(view.row_count(), 0);
    }

    #[test]
    fn test_view_skips_old_events() {
        let mut view = make_user_view();
        view.start_building();

        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")])).unwrap();
        view.apply_event(&make_event(1, "UserCreated", &[("user_id", "u2"), ("name", "Bob")])).unwrap();

        // Re-apply old event.
        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u3"), ("name", "Charlie")])).unwrap();
        assert_eq!(view.row_count(), 2); // u3 was skipped.
    }

    #[test]
    fn test_view_handler_error_faults_view() {
        let mut view = MaterializedView::new("bad-view");
        view.add_handler(ViewHandler::new("Bad", fail_handler));
        view.start_building();

        let err = view.apply_event(&make_event(0, "Bad", &[])).unwrap_err();
        assert!(matches!(err, ReadModelError::HandlerError { .. }));
        assert_eq!(view.status, ViewStatus::Faulted);
    }

    #[test]
    fn test_view_no_handler_skips_event() {
        let mut view = make_user_view();
        view.start_building();

        view.apply_event(&make_event(0, "UnknownEvent", &[])).unwrap();
        assert_eq!(view.events_processed, 0);
        assert_eq!(view.last_position, 1);
    }

    #[test]
    fn test_view_refresh() {
        let mut view = make_user_view();
        view.start_building();
        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")])).unwrap();
        view.mark_ready();

        view.refresh();
        assert_eq!(view.status, ViewStatus::Stale);
        assert_eq!(view.row_count(), 0);
        assert_eq!(view.last_position, 0);
        assert_eq!(view.version, 2);
    }

    #[test]
    fn test_view_query_predicate() {
        let mut view = make_user_view();
        view.start_building();
        view.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")])).unwrap();
        view.apply_event(&make_event(1, "UserCreated", &[("user_id", "u2"), ("name", "Bob")])).unwrap();
        view.mark_ready();

        let results = view
            .query_rows(|row| {
                row.get("name")
                    .and_then(|v| v.as_str())
                    .map(|n| n.starts_with('A'))
                    .unwrap_or(false)
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "u1");
    }

    #[test]
    fn test_view_query_faulted_returns_error() {
        let mut view = MaterializedView::new("v1");
        view.add_handler(ViewHandler::new("Bad", fail_handler));
        view.start_building();
        let _ = view.apply_event(&make_event(0, "Bad", &[]));

        let err = view.query_rows(|_| true).unwrap_err();
        assert!(matches!(err, ReadModelError::ViewNotReady(_)));
    }

    #[test]
    fn test_view_batch_apply() {
        let mut view = make_user_view();
        view.start_building();

        let events = vec![
            make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")]),
            make_event(1, "UserCreated", &[("user_id", "u2"), ("name", "Bob")]),
        ];
        let count = view.apply_events(&events).unwrap();
        assert_eq!(count, 2);
        assert_eq!(view.row_count(), 2);
    }

    #[test]
    fn test_view_has_handler() {
        let view = make_user_view();
        assert!(view.has_handler("UserCreated"));
        assert!(!view.has_handler("OrderPlaced"));
    }

    #[test]
    fn test_builder_register_and_get() {
        let mut builder = ReadModelBuilder::new();
        builder.register_view(make_user_view()).unwrap();
        assert!(builder.get_view("users").is_some());
    }

    #[test]
    fn test_builder_duplicate_register() {
        let mut builder = ReadModelBuilder::new();
        builder.register_view(make_user_view()).unwrap();
        let err = builder.register_view(make_user_view()).unwrap_err();
        assert!(matches!(err, ReadModelError::ViewAlreadyExists(_)));
    }

    #[test]
    fn test_builder_dispatch_to_multiple_views() {
        let mut builder = ReadModelBuilder::new();

        let mut v1 = make_user_view();
        v1.start_building();
        builder.register_view(v1).unwrap();

        let mut v2 = MaterializedView::new("user-count");
        fn count_handler(data: &mut ViewData, _event: &ReadModelEvent) -> Result<(), String> {
            let count = data
                .get("_count")
                .and_then(|r| r.get("value"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let mut row = ViewRow::new("_count");
            row.set("value", serde_json::json!(count + 1));
            data.upsert(row);
            Ok(())
        }
        v2.add_handler(ViewHandler::new("UserCreated", count_handler));
        v2.start_building();
        builder.register_view(v2).unwrap();

        let event = make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "Alice")]);
        let results = builder.dispatch(&event);
        // Both views should have received the event.
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, r)| r.is_ok()));
    }

    #[test]
    fn test_builder_remove_view() {
        let mut builder = ReadModelBuilder::new();
        builder.register_view(make_user_view()).unwrap();
        let removed = builder.remove_view("users").unwrap();
        assert_eq!(removed.view_id, "users");
        assert!(builder.get_view("users").is_none());
    }

    #[test]
    fn test_builder_remove_not_found() {
        let mut builder = ReadModelBuilder::new();
        let err = builder.remove_view("ghost").unwrap_err();
        assert!(matches!(err, ReadModelError::ViewNotFound(_)));
    }

    #[test]
    fn test_builder_view_ids_sorted() {
        let mut builder = ReadModelBuilder::new();
        builder.register_view(MaterializedView::new("zulu")).unwrap();
        builder.register_view(MaterializedView::new("alpha")).unwrap();
        assert_eq!(builder.view_ids(), vec!["alpha", "zulu"]);
    }

    #[test]
    fn test_builder_refresh_all() {
        let mut builder = ReadModelBuilder::new();
        let mut v = make_user_view();
        v.start_building();
        v.apply_event(&make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "A")])).unwrap();
        builder.register_view(v).unwrap();

        builder.refresh_all();
        let view = builder.get_view("users").unwrap();
        assert_eq!(view.status, ViewStatus::Stale);
        assert_eq!(view.row_count(), 0);
    }

    #[test]
    fn test_builder_count_by_status() {
        let mut builder = ReadModelBuilder::new();
        builder.register_view(MaterializedView::new("v1")).unwrap();
        builder.register_view(MaterializedView::new("v2")).unwrap();
        assert_eq!(builder.count_by_status(ViewStatus::Created), 2);
    }

    #[test]
    fn test_view_row_operations() {
        let mut row = ViewRow::new("r1");
        row.set("name", serde_json::json!("Alice"));
        row.set("age", serde_json::json!(30));

        assert_eq!(row.get("name").unwrap(), &serde_json::json!("Alice"));
        assert_eq!(row.get("age").unwrap(), &serde_json::json!(30));
        assert!(row.get("missing").is_none());
    }

    #[test]
    fn test_view_data_query() {
        let mut data = ViewData::new();
        let mut r1 = ViewRow::new("r1");
        r1.set("score", serde_json::json!(10));
        let mut r2 = ViewRow::new("r2");
        r2.set("score", serde_json::json!(20));
        let mut r3 = ViewRow::new("r3");
        r3.set("score", serde_json::json!(5));

        data.upsert(r1);
        data.upsert(r2);
        data.upsert(r3);

        let high_scores = data.query(|row| {
            row.get("score")
                .and_then(|v| v.as_i64())
                .map(|s| s >= 10)
                .unwrap_or(false)
        });
        assert_eq!(high_scores.len(), 2);
    }

    #[test]
    fn test_view_data_all_rows_sorted() {
        let mut data = ViewData::new();
        data.upsert(ViewRow::new("c"));
        data.upsert(ViewRow::new("a"));
        data.upsert(ViewRow::new("b"));

        let rows = data.all_rows();
        let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_builder_dispatch_batch() {
        let mut builder = ReadModelBuilder::new();
        let mut v = make_user_view();
        v.start_building();
        builder.register_view(v).unwrap();

        let events = vec![
            make_event(0, "UserCreated", &[("user_id", "u1"), ("name", "A")]),
            make_event(1, "UserCreated", &[("user_id", "u2"), ("name", "B")]),
        ];
        let results = builder.dispatch_batch(&events);
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_ok());

        let view = builder.get_view("users").unwrap();
        assert_eq!(view.row_count(), 2);
    }
}
