//! Change Data Capture (CDC) — change event types (insert/update/delete),
//! before/after images, schema evolution, event ordering, and consumer
//! checkpoints for tracking replay position.
//!
//! Replaces JS CDC libraries (Debezium connectors, change-streams) with a
//! pure-Rust CDC engine that tracks energy per captured change.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// CDC errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdcError {
    /// Source not found.
    SourceNotFound(String),
    /// Consumer not found.
    ConsumerNotFound(String),
    /// Out-of-order event.
    OutOfOrder { expected_seq: u64, got_seq: u64 },
    /// Schema version not found.
    SchemaVersionNotFound { source: String, version: u32 },
    /// Checkpoint not found.
    CheckpointNotFound(String),
    /// Duplicate event id.
    DuplicateEvent(String),
    /// Invalid schema migration.
    InvalidSchemaMigration(String),
}

impl std::fmt::Display for CdcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceNotFound(id) => write!(f, "source not found: {id}"),
            Self::ConsumerNotFound(id) => write!(f, "consumer not found: {id}"),
            Self::OutOfOrder { expected_seq, got_seq } => {
                write!(f, "out of order: expected seq {expected_seq}, got {got_seq}")
            }
            Self::SchemaVersionNotFound { source, version } => {
                write!(f, "schema version {version} not found for source {source}")
            }
            Self::CheckpointNotFound(id) => write!(f, "checkpoint not found: {id}"),
            Self::DuplicateEvent(id) => write!(f, "duplicate event: {id}"),
            Self::InvalidSchemaMigration(msg) => write!(f, "invalid schema migration: {msg}"),
        }
    }
}

impl std::error::Error for CdcError {}

// ── Event Types ─────────────────────────────────────────────────

/// The type of change captured.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    Insert,
    Update,
    Delete,
    /// Schema change (DDL).
    SchemaChange,
    /// Snapshot (initial load).
    Snapshot,
}

/// A single change event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    pub event_id: String,
    pub source_id: String,
    pub change_type: ChangeType,
    pub sequence: u64,
    /// Before image (None for inserts).
    pub before: Option<HashMap<String, serde_json::Value>>,
    /// After image (None for deletes).
    pub after: Option<HashMap<String, serde_json::Value>>,
    /// Primary key fields.
    pub key: HashMap<String, serde_json::Value>,
    pub schema_version: u32,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Schema definition for a CDC source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcSchema {
    pub source_id: String,
    pub version: u32,
    pub fields: Vec<FieldDefinition>,
    pub created_at: DateTime<Utc>,
}

/// A field in a CDC schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: FieldType,
    pub nullable: bool,
    pub default_value: Option<serde_json::Value>,
}

/// Supported field types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int64,
    Float64,
    Bool,
    Timestamp,
    Json,
    Binary,
}

// ── Consumer Checkpoint ─────────────────────────────────────────

/// A consumer's position in the change stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerCheckpoint {
    pub consumer_id: String,
    pub source_id: String,
    pub last_sequence: u64,
    pub last_event_id: String,
    pub updated_at: DateTime<Utc>,
}

// ── CDC Source ───────────────────────────────────────────────────

/// A registered CDC source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcSource {
    pub source_id: String,
    pub description: String,
    pub schemas: Vec<CdcSchema>,
    pub current_schema_version: u32,
    pub created_at: DateTime<Utc>,
}

// ── CDC Engine ──────────────────────────────────────────────────

/// The main CDC engine managing sources, events, and consumers.
#[derive(Debug, Clone)]
pub struct CdcEngine {
    sources: HashMap<String, CdcSource>,
    events: Vec<ChangeEvent>,
    event_ids: HashMap<String, usize>,
    /// source_id -> next expected sequence.
    source_sequences: HashMap<String, u64>,
    checkpoints: HashMap<String, ConsumerCheckpoint>,
    total_energy_uj: u64,
}

impl CdcEngine {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            events: Vec::new(),
            event_ids: HashMap::new(),
            source_sequences: HashMap::new(),
            checkpoints: HashMap::new(),
            total_energy_uj: 0,
        }
    }

    /// Register a CDC source with an initial schema.
    pub fn register_source(
        &mut self,
        source_id: &str,
        description: &str,
        initial_fields: Vec<FieldDefinition>,
    ) -> CdcSource {
        let schema = CdcSchema {
            source_id: source_id.to_string(),
            version: 1,
            fields: initial_fields,
            created_at: Utc::now(),
        };
        let source = CdcSource {
            source_id: source_id.to_string(),
            description: description.to_string(),
            schemas: vec![schema],
            current_schema_version: 1,
            created_at: Utc::now(),
        };
        self.sources.insert(source_id.to_string(), source.clone());
        self.source_sequences.insert(source_id.to_string(), 1);
        self.total_energy_uj += 10;
        source
    }

    /// Evolve a source's schema by adding/removing fields.
    pub fn evolve_schema(
        &mut self,
        source_id: &str,
        new_fields: Vec<FieldDefinition>,
    ) -> Result<CdcSchema, CdcError> {
        let source = self
            .sources
            .get_mut(source_id)
            .ok_or_else(|| CdcError::SourceNotFound(source_id.to_string()))?;

        let new_version = source.current_schema_version + 1;

        // Validate: new fields must be a superset of required (non-nullable) old fields
        // that don't have defaults, or the removed fields must be nullable.
        let old_schema = source.schemas.last().unwrap();
        for old_field in &old_schema.fields {
            let still_exists = new_fields.iter().any(|f| f.name == old_field.name);
            if !still_exists && !old_field.nullable && old_field.default_value.is_none() {
                return Err(CdcError::InvalidSchemaMigration(format!(
                    "cannot remove non-nullable field '{}' without default",
                    old_field.name
                )));
            }
        }

        let schema = CdcSchema {
            source_id: source_id.to_string(),
            version: new_version,
            fields: new_fields,
            created_at: Utc::now(),
        };
        source.schemas.push(schema.clone());
        source.current_schema_version = new_version;
        self.total_energy_uj += 15;
        Ok(schema)
    }

    /// Capture a change event.
    pub fn capture(&mut self, event: ChangeEvent) -> Result<u64, CdcError> {
        // Validate source exists.
        if !self.sources.contains_key(&event.source_id) {
            return Err(CdcError::SourceNotFound(event.source_id.clone()));
        }
        // Duplicate check.
        if self.event_ids.contains_key(&event.event_id) {
            return Err(CdcError::DuplicateEvent(event.event_id.clone()));
        }
        // Ordering check.
        let expected = self.source_sequences.get(&event.source_id).copied().unwrap_or(1);
        if event.sequence != expected {
            return Err(CdcError::OutOfOrder {
                expected_seq: expected,
                got_seq: event.sequence,
            });
        }

        let idx = self.events.len();
        self.event_ids.insert(event.event_id.clone(), idx);
        self.source_sequences.insert(event.source_id.clone(), expected + 1);
        self.events.push(event);
        self.total_energy_uj += 8;
        Ok(idx as u64)
    }

    /// Read events for a source from a given sequence (inclusive).
    pub fn read_events(
        &self,
        source_id: &str,
        from_sequence: u64,
        limit: usize,
    ) -> Result<Vec<&ChangeEvent>, CdcError> {
        if !self.sources.contains_key(source_id) {
            return Err(CdcError::SourceNotFound(source_id.to_string()));
        }
        let result: Vec<&ChangeEvent> = self
            .events
            .iter()
            .filter(|e| e.source_id == source_id && e.sequence >= from_sequence)
            .take(limit)
            .collect();
        Ok(result)
    }

    /// Read events for a consumer, starting from their checkpoint.
    pub fn read_for_consumer(
        &self,
        consumer_id: &str,
        source_id: &str,
        limit: usize,
    ) -> Result<Vec<&ChangeEvent>, CdcError> {
        if !self.sources.contains_key(source_id) {
            return Err(CdcError::SourceNotFound(source_id.to_string()));
        }
        let from_seq = self
            .checkpoints
            .get(consumer_id)
            .filter(|cp| cp.source_id == source_id)
            .map(|cp| cp.last_sequence + 1)
            .unwrap_or(1);
        self.read_events(source_id, from_seq, limit)
    }

    /// Commit a consumer checkpoint.
    pub fn commit_checkpoint(
        &mut self,
        consumer_id: &str,
        source_id: &str,
        last_sequence: u64,
        last_event_id: &str,
    ) -> Result<(), CdcError> {
        if !self.sources.contains_key(source_id) {
            return Err(CdcError::SourceNotFound(source_id.to_string()));
        }
        let cp = ConsumerCheckpoint {
            consumer_id: consumer_id.to_string(),
            source_id: source_id.to_string(),
            last_sequence,
            last_event_id: last_event_id.to_string(),
            updated_at: Utc::now(),
        };
        self.checkpoints.insert(consumer_id.to_string(), cp);
        self.total_energy_uj += 5;
        Ok(())
    }

    /// Get a consumer's checkpoint.
    pub fn get_checkpoint(&self, consumer_id: &str) -> Option<&ConsumerCheckpoint> {
        self.checkpoints.get(consumer_id)
    }

    /// Get a source.
    pub fn get_source(&self, source_id: &str) -> Option<&CdcSource> {
        self.sources.get(source_id)
    }

    /// Get a specific schema version for a source.
    pub fn get_schema(
        &self,
        source_id: &str,
        version: u32,
    ) -> Result<&CdcSchema, CdcError> {
        let source = self
            .sources
            .get(source_id)
            .ok_or_else(|| CdcError::SourceNotFound(source_id.to_string()))?;
        source
            .schemas
            .iter()
            .find(|s| s.version == version)
            .ok_or_else(|| CdcError::SchemaVersionNotFound {
                source: source_id.to_string(),
                version,
            })
    }

    /// Total events captured.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get all events (for testing/inspection).
    pub fn all_events(&self) -> &[ChangeEvent] {
        &self.events
    }

    /// Total energy consumed.
    pub fn total_energy_uj(&self) -> u64 {
        self.total_energy_uj
    }

    /// Compute change summary for a source.
    pub fn change_summary(&self, source_id: &str) -> HashMap<ChangeType, u64> {
        let mut counts = HashMap::new();
        for event in &self.events {
            if event.source_id == source_id {
                *counts.entry(event.change_type.clone()).or_insert(0) += 1;
            }
        }
        counts
    }
}

impl Default for CdcEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper ──────────────────────────────────────────────────────

/// Build a change event conveniently.
pub fn make_change_event(
    event_id: &str,
    source_id: &str,
    change_type: ChangeType,
    sequence: u64,
    key: HashMap<String, serde_json::Value>,
    before: Option<HashMap<String, serde_json::Value>>,
    after: Option<HashMap<String, serde_json::Value>>,
) -> ChangeEvent {
    ChangeEvent {
        event_id: event_id.to_string(),
        source_id: source_id.to_string(),
        change_type,
        sequence,
        before,
        after,
        key,
        schema_version: 1,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fields() -> Vec<FieldDefinition> {
        vec![
            FieldDefinition {
                name: "id".into(),
                field_type: FieldType::Int64,
                nullable: false,
                default_value: None,
            },
            FieldDefinition {
                name: "name".into(),
                field_type: FieldType::String,
                nullable: false,
                default_value: None,
            },
            FieldDefinition {
                name: "email".into(),
                field_type: FieldType::String,
                nullable: true,
                default_value: None,
            },
        ]
    }

    fn pk(id: i64) -> HashMap<String, serde_json::Value> {
        let mut m = HashMap::new();
        m.insert("id".into(), serde_json::json!(id));
        m
    }

    fn row(id: i64, name: &str) -> HashMap<String, serde_json::Value> {
        let mut m = HashMap::new();
        m.insert("id".into(), serde_json::json!(id));
        m.insert("name".into(), serde_json::json!(name));
        m
    }

    #[test]
    fn test_register_source() {
        let mut engine = CdcEngine::new();
        let src = engine.register_source("users", "User table", sample_fields());
        assert_eq!(src.source_id, "users");
        assert_eq!(src.current_schema_version, 1);
        assert_eq!(src.schemas.len(), 1);
        assert_eq!(src.schemas[0].fields.len(), 3);
    }

    #[test]
    fn test_capture_insert() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "User table", sample_fields());

        let event = make_change_event(
            "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "alice")),
        );
        let idx = engine.capture(event).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(engine.event_count(), 1);
    }

    #[test]
    fn test_capture_update_with_before_after() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "User table", sample_fields());

        let e1 = make_change_event(
            "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "alice")),
        );
        engine.capture(e1).unwrap();

        let e2 = make_change_event(
            "e2",
            "users",
            ChangeType::Update,
            2,
            pk(1),
            Some(row(1, "alice")),
            Some(row(1, "alice_updated")),
        );
        engine.capture(e2).unwrap();
        assert_eq!(engine.event_count(), 2);

        let events = engine.read_events("users", 2, 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].change_type, ChangeType::Update);
        assert!(events[0].before.is_some());
        assert!(events[0].after.is_some());
    }

    #[test]
    fn test_capture_delete() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "User table", sample_fields());

        let e1 = make_change_event(
            "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "alice")),
        );
        engine.capture(e1).unwrap();

        let e2 = make_change_event(
            "e2", "users", ChangeType::Delete, 2, pk(1), Some(row(1, "alice")), None,
        );
        engine.capture(e2).unwrap();

        let events = engine.read_events("users", 2, 10).unwrap();
        assert_eq!(events[0].change_type, ChangeType::Delete);
        assert!(events[0].after.is_none());
    }

    #[test]
    fn test_duplicate_event_rejected() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        let e = make_change_event(
            "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "a")),
        );
        engine.capture(e.clone()).unwrap();
        assert_eq!(
            engine.capture(e),
            Err(CdcError::DuplicateEvent("e1".into()))
        );
    }

    #[test]
    fn test_out_of_order() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        let e = make_change_event(
            "e1", "users", ChangeType::Insert, 5, pk(1), None, Some(row(1, "a")),
        );
        assert!(matches!(
            engine.capture(e),
            Err(CdcError::OutOfOrder { expected_seq: 1, got_seq: 5 })
        ));
    }

    #[test]
    fn test_source_not_found() {
        let mut engine = CdcEngine::new();
        let e = make_change_event(
            "e1", "missing", ChangeType::Insert, 1, pk(1), None, Some(row(1, "a")),
        );
        assert_eq!(
            engine.capture(e),
            Err(CdcError::SourceNotFound("missing".into()))
        );
    }

    #[test]
    fn test_consumer_checkpoint() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        for i in 1..=5 {
            let e = make_change_event(
                &format!("e{i}"),
                "users",
                ChangeType::Insert,
                i,
                pk(i as i64),
                None,
                Some(row(i as i64, &format!("user{i}"))),
            );
            engine.capture(e).unwrap();
        }

        // Consumer reads first 3.
        let batch = engine.read_for_consumer("c1", "users", 3).unwrap();
        assert_eq!(batch.len(), 3);

        // Commit checkpoint at seq 3.
        engine.commit_checkpoint("c1", "users", 3, "e3").unwrap();
        let cp = engine.get_checkpoint("c1").unwrap();
        assert_eq!(cp.last_sequence, 3);

        // Next read starts at 4.
        let batch2 = engine.read_for_consumer("c1", "users", 10).unwrap();
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2[0].sequence, 4);
    }

    #[test]
    fn test_schema_evolution_add_field() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        let mut new_fields = sample_fields();
        new_fields.push(FieldDefinition {
            name: "age".into(),
            field_type: FieldType::Int64,
            nullable: true,
            default_value: None,
        });

        let schema = engine.evolve_schema("users", new_fields).unwrap();
        assert_eq!(schema.version, 2);
        assert_eq!(schema.fields.len(), 4);

        let src = engine.get_source("users").unwrap();
        assert_eq!(src.current_schema_version, 2);
        assert_eq!(src.schemas.len(), 2);
    }

    #[test]
    fn test_schema_evolution_remove_nullable_field() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        // Remove 'email' (nullable) — should succeed.
        let new_fields = vec![
            FieldDefinition {
                name: "id".into(),
                field_type: FieldType::Int64,
                nullable: false,
                default_value: None,
            },
            FieldDefinition {
                name: "name".into(),
                field_type: FieldType::String,
                nullable: false,
                default_value: None,
            },
        ];
        assert!(engine.evolve_schema("users", new_fields).is_ok());
    }

    #[test]
    fn test_schema_evolution_remove_required_field_fails() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        // Remove 'name' (non-nullable, no default) — should fail.
        let new_fields = vec![FieldDefinition {
            name: "id".into(),
            field_type: FieldType::Int64,
            nullable: false,
            default_value: None,
        }];
        assert!(matches!(
            engine.evolve_schema("users", new_fields),
            Err(CdcError::InvalidSchemaMigration(_))
        ));
    }

    #[test]
    fn test_get_schema_version() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        let s = engine.get_schema("users", 1).unwrap();
        assert_eq!(s.version, 1);

        assert!(matches!(
            engine.get_schema("users", 99),
            Err(CdcError::SchemaVersionNotFound { .. })
        ));
    }

    #[test]
    fn test_change_summary() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());

        engine
            .capture(make_change_event(
                "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "a")),
            ))
            .unwrap();
        engine
            .capture(make_change_event(
                "e2", "users", ChangeType::Insert, 2, pk(2), None, Some(row(2, "b")),
            ))
            .unwrap();
        engine
            .capture(make_change_event(
                "e3",
                "users",
                ChangeType::Update,
                3,
                pk(1),
                Some(row(1, "a")),
                Some(row(1, "a2")),
            ))
            .unwrap();

        let summary = engine.change_summary("users");
        assert_eq!(summary.get(&ChangeType::Insert), Some(&2));
        assert_eq!(summary.get(&ChangeType::Update), Some(&1));
        assert_eq!(summary.get(&ChangeType::Delete), None);
    }

    #[test]
    fn test_energy_tracking() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());
        assert!(engine.total_energy_uj() > 0);

        engine
            .capture(make_change_event(
                "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "a")),
            ))
            .unwrap();
        assert!(engine.total_energy_uj() > 10);
    }

    #[test]
    fn test_read_events_source_not_found() {
        let engine = CdcEngine::new();
        assert!(matches!(
            engine.read_events("nope", 1, 10),
            Err(CdcError::SourceNotFound(_))
        ));
    }

    #[test]
    fn test_commit_checkpoint_bad_source() {
        let mut engine = CdcEngine::new();
        assert!(matches!(
            engine.commit_checkpoint("c1", "nope", 1, "e1"),
            Err(CdcError::SourceNotFound(_))
        ));
    }

    #[test]
    fn test_change_type_serde() {
        let ct = ChangeType::SchemaChange;
        let json = serde_json::to_string(&ct).unwrap();
        let parsed: ChangeType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ChangeType::SchemaChange);
    }

    #[test]
    fn test_field_type_serde() {
        let ft = FieldType::Timestamp;
        let json = serde_json::to_string(&ft).unwrap();
        let parsed: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, FieldType::Timestamp);
    }

    #[test]
    fn test_default_engine() {
        let engine = CdcEngine::default();
        assert_eq!(engine.event_count(), 0);
    }

    #[test]
    fn test_error_display() {
        let e = CdcError::OutOfOrder {
            expected_seq: 5,
            got_seq: 3,
        };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_evolve_schema_source_not_found() {
        let mut engine = CdcEngine::new();
        assert!(matches!(
            engine.evolve_schema("nope", vec![]),
            Err(CdcError::SourceNotFound(_))
        ));
    }

    #[test]
    fn test_multiple_sources() {
        let mut engine = CdcEngine::new();
        engine.register_source("users", "Users", sample_fields());
        engine.register_source("orders", "Orders", vec![
            FieldDefinition {
                name: "order_id".into(),
                field_type: FieldType::Int64,
                nullable: false,
                default_value: None,
            },
        ]);

        engine
            .capture(make_change_event(
                "e1", "users", ChangeType::Insert, 1, pk(1), None, Some(row(1, "a")),
            ))
            .unwrap();
        let mut opk = HashMap::new();
        opk.insert("order_id".into(), serde_json::json!(100));
        engine
            .capture(make_change_event(
                "e2", "orders", ChangeType::Insert, 1, opk, None, None,
            ))
            .unwrap();

        assert_eq!(engine.event_count(), 2);
        assert_eq!(engine.read_events("users", 1, 10).unwrap().len(), 1);
        assert_eq!(engine.read_events("orders", 1, 10).unwrap().len(), 1);
    }
}
