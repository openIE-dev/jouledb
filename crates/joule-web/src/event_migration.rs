//! Event schema migration — event version tracking, upcaster chain (v1->v2->v3),
//! downcaster, migration runner, event envelope with version, breaking change
//! detection, and migration dry-run.
//!
//! Replaces JS event migration libraries (EventStoreDB projections, custom
//! upcasters) with a pure-Rust event schema migration engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Event migration errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationError {
    /// Upcaster not found for event type / version pair.
    UpcasterNotFound { event_type: String, from_version: u32 },
    /// Downcaster not found.
    DowncasterNotFound { event_type: String, from_version: u32 },
    /// Upcaster chain broken (gap in versions).
    ChainBroken { event_type: String, from: u32, to: u32 },
    /// Migration failed during transformation.
    TransformError { event_type: String, version: u32, reason: String },
    /// Breaking change detected.
    BreakingChange { event_type: String, field: String, description: String },
    /// Duplicate upcaster registration.
    DuplicateUpcaster { event_type: String, from_version: u32 },
    /// Invalid version.
    InvalidVersion { event_type: String, version: u32, reason: String },
    /// Dry-run failure.
    DryRunFailure { event_type: String, failures: Vec<String> },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UpcasterNotFound { event_type, from_version } => {
                write!(f, "no upcaster for {event_type} v{from_version}")
            }
            Self::DowncasterNotFound { event_type, from_version } => {
                write!(f, "no downcaster for {event_type} v{from_version}")
            }
            Self::ChainBroken { event_type, from, to } => {
                write!(f, "upcaster chain broken for {event_type}: v{from} -> v{to}")
            }
            Self::TransformError { event_type, version, reason } => {
                write!(f, "transform error for {event_type} v{version}: {reason}")
            }
            Self::BreakingChange { event_type, field, description } => {
                write!(f, "breaking change in {event_type}.{field}: {description}")
            }
            Self::DuplicateUpcaster { event_type, from_version } => {
                write!(f, "duplicate upcaster for {event_type} v{from_version}")
            }
            Self::InvalidVersion { event_type, version, reason } => {
                write!(f, "invalid version for {event_type} v{version}: {reason}")
            }
            Self::DryRunFailure { event_type, failures } => {
                write!(f, "dry-run failed for {event_type}: {} issues", failures.len())
            }
        }
    }
}

impl std::error::Error for MigrationError {}

// ── Versioned Event Envelope ────────────────────────────────────

/// An event envelope with explicit schema version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionedEvent {
    pub event_id: String,
    pub event_type: String,
    pub schema_version: u32,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl VersionedEvent {
    pub fn new(
        event_type: impl Into<String>,
        schema_version: u32,
        data: HashMap<String, String>,
    ) -> Self {
        let et = event_type.into();
        Self {
            event_id: format!("{}-v{}-{}", et, schema_version, Utc::now().timestamp_nanos_opt().unwrap_or(0)),
            event_type: et,
            schema_version,
            data,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Create a new version of this event with updated data.
    pub fn upgrade_to(&self, new_version: u32, new_data: HashMap<String, String>) -> Self {
        Self {
            event_id: self.event_id.clone(),
            event_type: self.event_type.clone(),
            schema_version: new_version,
            data: new_data,
            metadata: self.metadata.clone(),
            timestamp: self.timestamp,
        }
    }
}

// ── Schema Change ───────────────────────────────────────────────

/// Describes a schema change between versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaChange {
    /// A field was added (with a default value).
    FieldAdded { field: String, default_value: String },
    /// A field was removed.
    FieldRemoved { field: String },
    /// A field was renamed.
    FieldRenamed { old_name: String, new_name: String },
    /// A field type changed.
    FieldTypeChanged { field: String, old_type: String, new_type: String },
    /// A field value was transformed.
    FieldTransformed { field: String, description: String },
}

impl SchemaChange {
    /// Check if this is a breaking change.
    pub fn is_breaking(&self) -> bool {
        matches!(
            self,
            SchemaChange::FieldRemoved { .. }
                | SchemaChange::FieldTypeChanged { .. }
        )
    }
}

// ── Upcaster ────────────────────────────────────────────────────

/// An upcaster that transforms events from one version to the next.
#[derive(Clone)]
pub struct Upcaster {
    pub event_type: String,
    pub from_version: u32,
    pub to_version: u32,
    pub changes: Vec<SchemaChange>,
    transform_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
}

impl Upcaster {
    pub fn new(
        event_type: impl Into<String>,
        from_version: u32,
        to_version: u32,
        changes: Vec<SchemaChange>,
        transform_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            from_version,
            to_version,
            changes,
            transform_fn,
        }
    }

    pub fn transform(&self, data: &HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        (self.transform_fn)(data)
    }
}

impl std::fmt::Debug for Upcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upcaster")
            .field("event_type", &self.event_type)
            .field("from_version", &self.from_version)
            .field("to_version", &self.to_version)
            .field("changes", &self.changes)
            .finish()
    }
}

// ── Downcaster ──────────────────────────────────────────────────

/// A downcaster that transforms events from a newer version to an older one.
#[derive(Clone)]
pub struct Downcaster {
    pub event_type: String,
    pub from_version: u32,
    pub to_version: u32,
    transform_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
}

impl Downcaster {
    pub fn new(
        event_type: impl Into<String>,
        from_version: u32,
        to_version: u32,
        transform_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            from_version,
            to_version,
            transform_fn,
        }
    }

    pub fn transform(&self, data: &HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        (self.transform_fn)(data)
    }
}

impl std::fmt::Debug for Downcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Downcaster")
            .field("event_type", &self.event_type)
            .field("from_version", &self.from_version)
            .field("to_version", &self.to_version)
            .finish()
    }
}

// ── Migration Result ────────────────────────────────────────────

/// Result of a migration run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationResult {
    pub events_processed: usize,
    pub events_migrated: usize,
    pub events_unchanged: usize,
    pub errors: Vec<String>,
}

impl MigrationResult {
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }
}

// ── Dry Run Result ──────────────────────────────────────────────

/// Result of a migration dry-run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DryRunResult {
    pub events_checked: usize,
    pub would_migrate: usize,
    pub would_skip: usize,
    pub breaking_changes: Vec<(String, Vec<SchemaChange>)>,
    pub errors: Vec<String>,
}

impl DryRunResult {
    pub fn is_safe(&self) -> bool {
        self.breaking_changes.is_empty() && self.errors.is_empty()
    }
}

// ── Event Migration Registry ────────────────────────────────────

/// Registry for event schema migrations.
#[derive(Debug)]
pub struct EventMigrationRegistry {
    /// Upcasters keyed by (event_type, from_version).
    upcasters: HashMap<(String, u32), Upcaster>,
    /// Downcasters keyed by (event_type, from_version).
    downcasters: HashMap<(String, u32), Downcaster>,
    /// Current schema versions keyed by event_type.
    current_versions: HashMap<String, u32>,
}

impl EventMigrationRegistry {
    pub fn new() -> Self {
        Self {
            upcasters: HashMap::new(),
            downcasters: HashMap::new(),
            current_versions: HashMap::new(),
        }
    }

    /// Register the current schema version for an event type.
    pub fn register_version(&mut self, event_type: impl Into<String>, version: u32) {
        self.current_versions.insert(event_type.into(), version);
    }

    /// Get the current schema version.
    pub fn current_version(&self, event_type: &str) -> Option<u32> {
        self.current_versions.get(event_type).copied()
    }

    /// Register an upcaster.
    pub fn register_upcaster(&mut self, upcaster: Upcaster) -> Result<(), MigrationError> {
        let key = (upcaster.event_type.clone(), upcaster.from_version);
        if self.upcasters.contains_key(&key) {
            return Err(MigrationError::DuplicateUpcaster {
                event_type: upcaster.event_type.clone(),
                from_version: upcaster.from_version,
            });
        }
        self.upcasters.insert(key, upcaster);
        Ok(())
    }

    /// Register a downcaster.
    pub fn register_downcaster(&mut self, downcaster: Downcaster) {
        let key = (downcaster.event_type.clone(), downcaster.from_version);
        self.downcasters.insert(key, downcaster);
    }

    /// Upcast a single event to the target version.
    pub fn upcast(
        &self,
        event: &VersionedEvent,
        target_version: u32,
    ) -> Result<VersionedEvent, MigrationError> {
        if event.schema_version >= target_version {
            return Ok(event.clone());
        }

        let mut current = event.clone();

        while current.schema_version < target_version {
            let key = (current.event_type.clone(), current.schema_version);
            let upcaster = self
                .upcasters
                .get(&key)
                .ok_or_else(|| MigrationError::UpcasterNotFound {
                    event_type: current.event_type.clone(),
                    from_version: current.schema_version,
                })?;

            // Validate chain continuity.
            if upcaster.to_version != current.schema_version + 1 {
                return Err(MigrationError::ChainBroken {
                    event_type: current.event_type.clone(),
                    from: current.schema_version,
                    to: upcaster.to_version,
                });
            }

            let new_data = upcaster.transform(&current.data).map_err(|reason| {
                MigrationError::TransformError {
                    event_type: current.event_type.clone(),
                    version: current.schema_version,
                    reason,
                }
            })?;

            current = current.upgrade_to(upcaster.to_version, new_data);
        }

        Ok(current)
    }

    /// Upcast to the current version.
    pub fn upcast_to_current(
        &self,
        event: &VersionedEvent,
    ) -> Result<VersionedEvent, MigrationError> {
        let target = self
            .current_versions
            .get(&event.event_type)
            .copied()
            .ok_or_else(|| MigrationError::InvalidVersion {
                event_type: event.event_type.clone(),
                version: event.schema_version,
                reason: "no current version registered".to_string(),
            })?;
        self.upcast(event, target)
    }

    /// Downcast a single event to a target version.
    pub fn downcast(
        &self,
        event: &VersionedEvent,
        target_version: u32,
    ) -> Result<VersionedEvent, MigrationError> {
        if event.schema_version <= target_version {
            return Ok(event.clone());
        }

        let mut current = event.clone();

        while current.schema_version > target_version {
            let key = (current.event_type.clone(), current.schema_version);
            let downcaster = self
                .downcasters
                .get(&key)
                .ok_or_else(|| MigrationError::DowncasterNotFound {
                    event_type: current.event_type.clone(),
                    from_version: current.schema_version,
                })?;

            let new_data = downcaster.transform(&current.data).map_err(|reason| {
                MigrationError::TransformError {
                    event_type: current.event_type.clone(),
                    version: current.schema_version,
                    reason,
                }
            })?;

            current = current.upgrade_to(downcaster.to_version, new_data);
        }

        Ok(current)
    }

    /// Detect breaking changes in the upcaster chain for an event type.
    pub fn detect_breaking_changes(&self, event_type: &str) -> Vec<SchemaChange> {
        let mut breaking = Vec::new();
        let mut version = 1u32;

        loop {
            let key = (event_type.to_string(), version);
            if let Some(upcaster) = self.upcasters.get(&key) {
                for change in &upcaster.changes {
                    if change.is_breaking() {
                        breaking.push(change.clone());
                    }
                }
                version = upcaster.to_version;
            } else {
                break;
            }
        }

        breaking
    }

    /// Run migration on a batch of events.
    pub fn migrate_batch(
        &self,
        events: &[VersionedEvent],
        target_version: u32,
    ) -> MigrationResult {
        let mut result = MigrationResult {
            events_processed: events.len(),
            events_migrated: 0,
            events_unchanged: 0,
            errors: Vec::new(),
        };

        for event in events {
            if event.schema_version == target_version {
                result.events_unchanged += 1;
                continue;
            }

            match self.upcast(event, target_version) {
                Ok(_) => {
                    result.events_migrated += 1;
                }
                Err(e) => {
                    result.errors.push(e.to_string());
                }
            }
        }

        result
    }

    /// Dry-run a migration (check without applying).
    pub fn dry_run(
        &self,
        events: &[VersionedEvent],
        target_version: u32,
    ) -> DryRunResult {
        let mut result = DryRunResult {
            events_checked: events.len(),
            would_migrate: 0,
            would_skip: 0,
            breaking_changes: Vec::new(),
            errors: Vec::new(),
        };

        // Check breaking changes for event types in the batch.
        let mut checked_types: Vec<String> = Vec::new();
        for event in events {
            if !checked_types.contains(&event.event_type) {
                let breaks = self.detect_breaking_changes(&event.event_type);
                if !breaks.is_empty() {
                    result
                        .breaking_changes
                        .push((event.event_type.clone(), breaks));
                }
                checked_types.push(event.event_type.clone());
            }
        }

        for event in events {
            if event.schema_version == target_version {
                result.would_skip += 1;
                continue;
            }

            match self.upcast(event, target_version) {
                Ok(_) => {
                    result.would_migrate += 1;
                }
                Err(e) => {
                    result.errors.push(e.to_string());
                }
            }
        }

        result
    }

    /// Check if an upcaster chain exists from a given version to a target.
    pub fn has_chain(&self, event_type: &str, from: u32, to: u32) -> bool {
        let mut current = from;
        while current < to {
            let key = (event_type.to_string(), current);
            if let Some(upcaster) = self.upcasters.get(&key) {
                current = upcaster.to_version;
            } else {
                return false;
            }
        }
        current == to
    }

    /// Count registered upcasters.
    pub fn upcaster_count(&self) -> usize {
        self.upcasters.len()
    }

    /// Count registered downcasters.
    pub fn downcaster_count(&self) -> usize {
        self.downcasters.len()
    }
}

impl Default for EventMigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_v1_event(key: &str, val: &str) -> VersionedEvent {
        let mut data = HashMap::new();
        data.insert(key.to_string(), val.to_string());
        VersionedEvent::new("UserCreated", 1, data)
    }

    fn v1_to_v2(data: &HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        let mut new_data = data.clone();
        // v2 adds "email" field with default.
        new_data.entry("email".to_string()).or_insert_with(|| "unknown@example.com".to_string());
        Ok(new_data)
    }

    fn v2_to_v3(data: &HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        let mut new_data = data.clone();
        // v3 renames "name" to "full_name".
        if let Some(name) = new_data.remove("name") {
            new_data.insert("full_name".to_string(), name);
        }
        Ok(new_data)
    }

    fn v3_to_v2_down(data: &HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        let mut new_data = data.clone();
        if let Some(full_name) = new_data.remove("full_name") {
            new_data.insert("name".to_string(), full_name);
        }
        Ok(new_data)
    }

    fn v2_to_v1_down(data: &HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        let mut new_data = data.clone();
        new_data.remove("email");
        Ok(new_data)
    }

    fn build_registry() -> EventMigrationRegistry {
        let mut reg = EventMigrationRegistry::new();
        reg.register_version("UserCreated", 3);

        reg.register_upcaster(Upcaster::new(
            "UserCreated",
            1,
            2,
            vec![SchemaChange::FieldAdded {
                field: "email".to_string(),
                default_value: "unknown@example.com".to_string(),
            }],
            v1_to_v2,
        ))
        .unwrap();

        reg.register_upcaster(Upcaster::new(
            "UserCreated",
            2,
            3,
            vec![SchemaChange::FieldRenamed {
                old_name: "name".to_string(),
                new_name: "full_name".to_string(),
            }],
            v2_to_v3,
        ))
        .unwrap();

        reg.register_downcaster(Downcaster::new("UserCreated", 3, 2, v3_to_v2_down));
        reg.register_downcaster(Downcaster::new("UserCreated", 2, 1, v2_to_v1_down));

        reg
    }

    #[test]
    fn test_upcast_v1_to_v2() {
        let reg = build_registry();
        let event = make_v1_event("name", "Alice");
        let upcasted = reg.upcast(&event, 2).unwrap();

        assert_eq!(upcasted.schema_version, 2);
        assert_eq!(upcasted.data.get("name").map(|s| s.as_str()), Some("Alice"));
        assert_eq!(upcasted.data.get("email").map(|s| s.as_str()), Some("unknown@example.com"));
    }

    #[test]
    fn test_upcast_v1_to_v3() {
        let reg = build_registry();
        let event = make_v1_event("name", "Alice");
        let upcasted = reg.upcast(&event, 3).unwrap();

        assert_eq!(upcasted.schema_version, 3);
        assert_eq!(upcasted.data.get("full_name").map(|s| s.as_str()), Some("Alice"));
        assert_eq!(upcasted.data.get("email").map(|s| s.as_str()), Some("unknown@example.com"));
        assert!(upcasted.data.get("name").is_none()); // Renamed.
    }

    #[test]
    fn test_upcast_to_current() {
        let reg = build_registry();
        let event = make_v1_event("name", "Bob");
        let upcasted = reg.upcast_to_current(&event).unwrap();
        assert_eq!(upcasted.schema_version, 3);
    }

    #[test]
    fn test_upcast_already_at_target() {
        let reg = build_registry();
        let mut data = HashMap::new();
        data.insert("full_name".to_string(), "Alice".to_string());
        let event = VersionedEvent::new("UserCreated", 3, data);
        let upcasted = reg.upcast(&event, 3).unwrap();
        assert_eq!(upcasted.schema_version, 3);
    }

    #[test]
    fn test_upcast_missing_upcaster() {
        let reg = EventMigrationRegistry::new();
        let event = make_v1_event("name", "Alice");
        let err = reg.upcast(&event, 2).unwrap_err();
        assert!(matches!(err, MigrationError::UpcasterNotFound { .. }));
    }

    #[test]
    fn test_downcast_v3_to_v1() {
        let reg = build_registry();
        let mut data = HashMap::new();
        data.insert("full_name".to_string(), "Alice".to_string());
        data.insert("email".to_string(), "alice@example.com".to_string());
        let event = VersionedEvent::new("UserCreated", 3, data);

        let downcasted = reg.downcast(&event, 1).unwrap();
        assert_eq!(downcasted.schema_version, 1);
        assert_eq!(downcasted.data.get("name").map(|s| s.as_str()), Some("Alice"));
        assert!(downcasted.data.get("email").is_none());
        assert!(downcasted.data.get("full_name").is_none());
    }

    #[test]
    fn test_downcast_already_at_target() {
        let reg = build_registry();
        let event = make_v1_event("name", "Alice");
        let downcasted = reg.downcast(&event, 1).unwrap();
        assert_eq!(downcasted.schema_version, 1);
    }

    #[test]
    fn test_downcast_missing_downcaster() {
        let reg = EventMigrationRegistry::new();
        let mut data = HashMap::new();
        data.insert("name".to_string(), "A".to_string());
        let event = VersionedEvent::new("Evt", 3, data);
        let err = reg.downcast(&event, 1).unwrap_err();
        assert!(matches!(err, MigrationError::DowncasterNotFound { .. }));
    }

    #[test]
    fn test_duplicate_upcaster() {
        let mut reg = build_registry();
        let err = reg
            .register_upcaster(Upcaster::new("UserCreated", 1, 2, vec![], v1_to_v2))
            .unwrap_err();
        assert!(matches!(err, MigrationError::DuplicateUpcaster { .. }));
    }

    #[test]
    fn test_detect_breaking_changes() {
        let reg = build_registry();
        let breaks = reg.detect_breaking_changes("UserCreated");
        // FieldAdded and FieldRenamed are not breaking; only FieldRemoved and FieldTypeChanged are.
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_detect_breaking_changes_with_removal() {
        let mut reg = EventMigrationRegistry::new();
        reg.register_upcaster(Upcaster::new(
            "OrderPlaced",
            1,
            2,
            vec![SchemaChange::FieldRemoved { field: "legacy_id".to_string() }],
            |data| {
                let mut d = data.clone();
                d.remove("legacy_id");
                Ok(d)
            },
        ))
        .unwrap();

        let breaks = reg.detect_breaking_changes("OrderPlaced");
        assert_eq!(breaks.len(), 1);
        assert!(matches!(&breaks[0], SchemaChange::FieldRemoved { field } if field == "legacy_id"));
    }

    #[test]
    fn test_schema_change_is_breaking() {
        assert!(!SchemaChange::FieldAdded { field: "x".into(), default_value: "".into() }.is_breaking());
        assert!(!SchemaChange::FieldRenamed { old_name: "a".into(), new_name: "b".into() }.is_breaking());
        assert!(SchemaChange::FieldRemoved { field: "x".into() }.is_breaking());
        assert!(SchemaChange::FieldTypeChanged {
            field: "x".into(),
            old_type: "str".into(),
            new_type: "int".into(),
        }
        .is_breaking());
    }

    #[test]
    fn test_migrate_batch() {
        let reg = build_registry();
        let events = vec![
            make_v1_event("name", "Alice"),
            make_v1_event("name", "Bob"),
            {
                let mut d = HashMap::new();
                d.insert("full_name".to_string(), "Charlie".to_string());
                VersionedEvent::new("UserCreated", 3, d)
            },
        ];

        let result = reg.migrate_batch(&events, 3);
        assert!(result.is_success());
        assert_eq!(result.events_processed, 3);
        assert_eq!(result.events_migrated, 2);
        assert_eq!(result.events_unchanged, 1);
    }

    #[test]
    fn test_dry_run_safe() {
        let reg = build_registry();
        let events = vec![
            make_v1_event("name", "Alice"),
            make_v1_event("name", "Bob"),
        ];

        let result = reg.dry_run(&events, 3);
        assert!(result.is_safe());
        assert_eq!(result.events_checked, 2);
        assert_eq!(result.would_migrate, 2);
        assert_eq!(result.would_skip, 0);
    }

    #[test]
    fn test_dry_run_with_breaking_changes() {
        let mut reg = EventMigrationRegistry::new();
        reg.register_upcaster(Upcaster::new(
            "Order",
            1,
            2,
            vec![SchemaChange::FieldRemoved { field: "old".into() }],
            |data| Ok(data.clone()),
        ))
        .unwrap();

        let events = vec![VersionedEvent::new("Order", 1, HashMap::new())];
        let result = reg.dry_run(&events, 2);
        assert!(!result.is_safe());
        assert_eq!(result.breaking_changes.len(), 1);
    }

    #[test]
    fn test_has_chain() {
        let reg = build_registry();
        assert!(reg.has_chain("UserCreated", 1, 3));
        assert!(reg.has_chain("UserCreated", 1, 2));
        assert!(reg.has_chain("UserCreated", 2, 3));
        assert!(!reg.has_chain("UserCreated", 1, 4)); // No v3->v4 upcaster.
    }

    #[test]
    fn test_current_version() {
        let reg = build_registry();
        assert_eq!(reg.current_version("UserCreated"), Some(3));
        assert_eq!(reg.current_version("Unknown"), None);
    }

    #[test]
    fn test_upcaster_downcaster_counts() {
        let reg = build_registry();
        assert_eq!(reg.upcaster_count(), 2);
        assert_eq!(reg.downcaster_count(), 2);
    }

    #[test]
    fn test_versioned_event_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        let event = make_v1_event("name", "Alice").with_metadata(meta);
        assert_eq!(event.metadata.get("source").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn test_transform_error() {
        let mut reg = EventMigrationRegistry::new();
        reg.register_upcaster(Upcaster::new(
            "Bad",
            1,
            2,
            vec![],
            |_| Err("bad transform".to_string()),
        ))
        .unwrap();

        let event = VersionedEvent::new("Bad", 1, HashMap::new());
        let err = reg.upcast(&event, 2).unwrap_err();
        assert!(matches!(err, MigrationError::TransformError { .. }));
    }

    #[test]
    fn test_upcast_preserves_event_id() {
        let reg = build_registry();
        let event = make_v1_event("name", "Alice");
        let original_id = event.event_id.clone();
        let upcasted = reg.upcast(&event, 3).unwrap();
        assert_eq!(upcasted.event_id, original_id);
    }

    #[test]
    fn test_upcast_preserves_timestamp() {
        let reg = build_registry();
        let event = make_v1_event("name", "Alice");
        let original_ts = event.timestamp;
        let upcasted = reg.upcast(&event, 3).unwrap();
        assert_eq!(upcasted.timestamp, original_ts);
    }

    #[test]
    fn test_migrate_batch_with_errors() {
        let reg = EventMigrationRegistry::new(); // No upcasters registered.
        let events = vec![make_v1_event("name", "Alice")];
        let result = reg.migrate_batch(&events, 3);
        assert!(!result.is_success());
        assert_eq!(result.errors.len(), 1);
    }
}
