//! DDD Entity base — identity-based equality, version tracking for optimistic
//! concurrency, domain events collection, entity lifecycle management
//! (new/persisted/modified/deleted), and audit fields.
//!
//! Replaces ad-hoc entity patterns in JS/TS (TypeORM entities, MikroORM
//! base entity) with a pure-Rust entity base that enforces DDD invariants.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Entity domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityError {
    /// Entity already deleted.
    AlreadyDeleted(String),
    /// Version conflict for optimistic concurrency.
    VersionConflict { id: String, expected: u64, actual: u64 },
    /// Invalid state transition.
    InvalidTransition { id: String, from: EntityLifecycle, to: EntityLifecycle },
    /// Validation failed.
    ValidationFailed { id: String, reason: String },
    /// Entity not found.
    NotFound(String),
    /// Duplicate entity.
    Duplicate(String),
}

impl std::fmt::Display for EntityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyDeleted(id) => write!(f, "entity already deleted: {id}"),
            Self::VersionConflict { id, expected, actual } => {
                write!(f, "version conflict for {id}: expected {expected}, got {actual}")
            }
            Self::InvalidTransition { id, from, to } => {
                write!(f, "invalid transition for {id}: {from:?} -> {to:?}")
            }
            Self::ValidationFailed { id, reason } => {
                write!(f, "validation failed for {id}: {reason}")
            }
            Self::NotFound(id) => write!(f, "entity not found: {id}"),
            Self::Duplicate(id) => write!(f, "duplicate entity: {id}"),
        }
    }
}

impl std::error::Error for EntityError {}

// ── Lifecycle ───────────────────────────────────────────────────

/// Entity lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityLifecycle {
    /// Newly created, not yet persisted.
    New,
    /// Persisted and clean.
    Persisted,
    /// Modified since last persistence.
    Modified,
    /// Marked for deletion.
    Deleted,
}

impl EntityLifecycle {
    /// Whether transition from `self` to `target` is valid.
    pub fn can_transition_to(self, target: Self) -> bool {
        matches!(
            (self, target),
            (Self::New, Self::Persisted)
                | (Self::Persisted, Self::Modified)
                | (Self::Persisted, Self::Deleted)
                | (Self::Modified, Self::Persisted)
                | (Self::Modified, Self::Deleted)
        )
    }
}

// ── Domain Event Record ─────────────────────────────────────────

/// A domain event collected by the entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainEventRecord {
    pub event_type: String,
    pub payload: HashMap<String, String>,
    pub occurred_at: DateTime<Utc>,
}

impl DomainEventRecord {
    pub fn new(event_type: impl Into<String>, payload: HashMap<String, String>) -> Self {
        Self {
            event_type: event_type.into(),
            payload,
            occurred_at: Utc::now(),
        }
    }

    pub fn simple(event_type: impl Into<String>) -> Self {
        Self::new(event_type, HashMap::new())
    }
}

// ── Audit Info ──────────────────────────────────────────────────

/// Audit tracking fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditInfo {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: String,
    pub updated_by: Option<String>,
}

impl AuditInfo {
    pub fn new(created_by: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            updated_at: now,
            created_by: created_by.into(),
            updated_by: None,
        }
    }

    pub fn mark_updated(&mut self, by: impl Into<String>) {
        self.updated_at = Utc::now();
        self.updated_by = Some(by.into());
    }
}

// ── Entity ──────────────────────────────────────────────────────

/// Base entity with identity, versioning, lifecycle, and domain events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    id: String,
    version: u64,
    lifecycle: EntityLifecycle,
    audit: AuditInfo,
    events: Vec<DomainEventRecord>,
    attributes: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity.
    pub fn new(id: impl Into<String>, created_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: 0,
            lifecycle: EntityLifecycle::New,
            audit: AuditInfo::new(created_by),
            events: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Reconstruct a persisted entity from storage.
    pub fn from_persisted(
        id: impl Into<String>,
        version: u64,
        audit: AuditInfo,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self {
            id: id.into(),
            version,
            lifecycle: EntityLifecycle::Persisted,
            audit,
            events: Vec::new(),
            attributes,
        }
    }

    /// The entity's unique identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Current version (for optimistic concurrency).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Current lifecycle state.
    pub fn lifecycle(&self) -> EntityLifecycle {
        self.lifecycle
    }

    /// Audit info.
    pub fn audit(&self) -> &AuditInfo {
        &self.audit
    }

    /// Entity attributes.
    pub fn attributes(&self) -> &HashMap<String, String> {
        &self.attributes
    }

    /// Set an attribute, transitioning to Modified if currently Persisted.
    pub fn set_attribute(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
        modified_by: impl Into<String>,
    ) -> Result<(), EntityError> {
        if self.lifecycle == EntityLifecycle::Deleted {
            return Err(EntityError::AlreadyDeleted(self.id.clone()));
        }
        let key = key.into();
        let value = value.into();
        self.attributes.insert(key.clone(), value.clone());
        if self.lifecycle == EntityLifecycle::Persisted {
            self.lifecycle = EntityLifecycle::Modified;
        }
        let by = modified_by.into();
        self.audit.mark_updated(&by);
        let mut payload = HashMap::new();
        payload.insert("key".to_string(), key);
        payload.insert("value".to_string(), value);
        payload.insert("by".to_string(), by);
        self.events.push(DomainEventRecord::new("attribute_changed", payload));
        Ok(())
    }

    /// Remove an attribute.
    pub fn remove_attribute(
        &mut self,
        key: &str,
        modified_by: impl Into<String>,
    ) -> Result<Option<String>, EntityError> {
        if self.lifecycle == EntityLifecycle::Deleted {
            return Err(EntityError::AlreadyDeleted(self.id.clone()));
        }
        let removed = self.attributes.remove(key);
        if removed.is_some() && self.lifecycle == EntityLifecycle::Persisted {
            self.lifecycle = EntityLifecycle::Modified;
        }
        if removed.is_some() {
            let by = modified_by.into();
            self.audit.mark_updated(&by);
            let mut payload = HashMap::new();
            payload.insert("key".to_string(), key.to_string());
            payload.insert("by".to_string(), by);
            self.events.push(DomainEventRecord::new("attribute_removed", payload));
        }
        Ok(removed)
    }

    /// Get an attribute value.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// Transition lifecycle state.
    pub fn transition_to(&mut self, target: EntityLifecycle) -> Result<(), EntityError> {
        if !self.lifecycle.can_transition_to(target) {
            return Err(EntityError::InvalidTransition {
                id: self.id.clone(),
                from: self.lifecycle,
                to: target,
            });
        }
        if target == EntityLifecycle::Persisted {
            self.version += 1;
        }
        self.lifecycle = target;
        Ok(())
    }

    /// Mark as persisted, bumping version.
    pub fn mark_persisted(&mut self) -> Result<(), EntityError> {
        self.transition_to(EntityLifecycle::Persisted)
    }

    /// Mark for deletion.
    pub fn mark_deleted(&mut self) -> Result<(), EntityError> {
        self.transition_to(EntityLifecycle::Deleted)
    }

    /// Apply a version-checked update.
    pub fn apply_versioned(
        &mut self,
        expected_version: u64,
        modified_by: impl Into<String>,
        f: impl FnOnce(&mut HashMap<String, String>),
    ) -> Result<(), EntityError> {
        if self.lifecycle == EntityLifecycle::Deleted {
            return Err(EntityError::AlreadyDeleted(self.id.clone()));
        }
        if self.version != expected_version {
            return Err(EntityError::VersionConflict {
                id: self.id.clone(),
                expected: expected_version,
                actual: self.version,
            });
        }
        f(&mut self.attributes);
        if self.lifecycle == EntityLifecycle::Persisted {
            self.lifecycle = EntityLifecycle::Modified;
        }
        self.audit.mark_updated(modified_by);
        Ok(())
    }

    /// Register a domain event on this entity.
    pub fn record_event(&mut self, event: DomainEventRecord) {
        self.events.push(event);
    }

    /// Drain all pending domain events.
    pub fn drain_events(&mut self) -> Vec<DomainEventRecord> {
        std::mem::take(&mut self.events)
    }

    /// Pending event count.
    pub fn pending_event_count(&self) -> usize {
        self.events.len()
    }

    /// Whether the entity has pending events.
    pub fn has_pending_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Whether the entity is dirty (new or modified).
    pub fn is_dirty(&self) -> bool {
        matches!(self.lifecycle, EntityLifecycle::New | EntityLifecycle::Modified)
    }

    /// Whether the entity is deleted.
    pub fn is_deleted(&self) -> bool {
        self.lifecycle == EntityLifecycle::Deleted
    }
}

/// Equality is identity-based: two entities are equal iff they have the same id.
impl PartialEq for Entity {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Entity {}

impl std::hash::Hash for Entity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// ── Entity Collection ───────────────────────────────────────────

/// A typed collection of entities with identity-based deduplication.
#[derive(Debug, Clone)]
pub struct EntityCollection {
    entities: Vec<Entity>,
}

impl EntityCollection {
    pub fn new() -> Self {
        Self { entities: Vec::new() }
    }

    /// Add an entity. Returns error if duplicate id.
    pub fn add(&mut self, entity: Entity) -> Result<(), EntityError> {
        if self.entities.iter().any(|e| e.id() == entity.id()) {
            return Err(EntityError::Duplicate(entity.id().to_string()));
        }
        self.entities.push(entity);
        Ok(())
    }

    /// Get by id.
    pub fn get(&self, id: &str) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id() == id)
    }

    /// Get mutable by id.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id() == id)
    }

    /// Remove by id.
    pub fn remove(&mut self, id: &str) -> Option<Entity> {
        if let Some(pos) = self.entities.iter().position(|e| e.id() == id) {
            Some(self.entities.remove(pos))
        } else {
            None
        }
    }

    /// Count.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Iterate.
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter()
    }

    /// All dirty entities.
    pub fn dirty(&self) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.is_dirty()).collect()
    }

    /// All deleted entities.
    pub fn deleted(&self) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.is_deleted()).collect()
    }

    /// Drain all events from all entities.
    pub fn drain_all_events(&mut self) -> Vec<(String, Vec<DomainEventRecord>)> {
        self.entities
            .iter_mut()
            .filter(|e| e.has_pending_events())
            .map(|e| {
                let id = e.id().to_string();
                let events = e.drain_events();
                (id, events)
            })
            .collect()
    }
}

impl Default for EntityCollection {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_entity_lifecycle() {
        let e = Entity::new("e1", "alice");
        assert_eq!(e.lifecycle(), EntityLifecycle::New);
        assert_eq!(e.version(), 0);
        assert!(e.is_dirty());
    }

    #[test]
    fn test_identity_based_equality() {
        let e1 = Entity::new("e1", "alice");
        let e2 = Entity::new("e1", "bob");
        let e3 = Entity::new("e2", "alice");
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_mark_persisted() {
        let mut e = Entity::new("e1", "alice");
        assert!(e.mark_persisted().is_ok());
        assert_eq!(e.lifecycle(), EntityLifecycle::Persisted);
        assert_eq!(e.version(), 1);
        assert!(!e.is_dirty());
    }

    #[test]
    fn test_modify_after_persisted() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        e.set_attribute("name", "test", "bob").unwrap();
        assert_eq!(e.lifecycle(), EntityLifecycle::Modified);
        assert!(e.is_dirty());
    }

    #[test]
    fn test_version_bump_on_persist() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        assert_eq!(e.version(), 1);
        e.set_attribute("k", "v", "alice").unwrap();
        e.mark_persisted().unwrap();
        assert_eq!(e.version(), 2);
    }

    #[test]
    fn test_delete_from_persisted() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        assert!(e.mark_deleted().is_ok());
        assert!(e.is_deleted());
    }

    #[test]
    fn test_delete_from_modified() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        e.set_attribute("k", "v", "alice").unwrap();
        assert!(e.mark_deleted().is_ok());
        assert!(e.is_deleted());
    }

    #[test]
    fn test_cannot_modify_deleted() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        e.mark_deleted().unwrap();
        let result = e.set_attribute("k", "v", "alice");
        assert!(matches!(result, Err(EntityError::AlreadyDeleted(_))));
    }

    #[test]
    fn test_invalid_transition_new_to_deleted() {
        let mut e = Entity::new("e1", "alice");
        let result = e.mark_deleted();
        assert!(matches!(result, Err(EntityError::InvalidTransition { .. })));
    }

    #[test]
    fn test_invalid_transition_new_to_modified() {
        let mut e = Entity::new("e1", "alice");
        let result = e.transition_to(EntityLifecycle::Modified);
        assert!(matches!(result, Err(EntityError::InvalidTransition { .. })));
    }

    #[test]
    fn test_domain_events_collected() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        e.set_attribute("name", "widget", "bob").unwrap();
        e.set_attribute("price", "100", "bob").unwrap();
        assert_eq!(e.pending_event_count(), 2);
        assert!(e.has_pending_events());
    }

    #[test]
    fn test_drain_events() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        e.set_attribute("name", "widget", "bob").unwrap();
        let events = e.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "attribute_changed");
        assert!(!e.has_pending_events());
    }

    #[test]
    fn test_record_custom_event() {
        let mut e = Entity::new("e1", "alice");
        e.record_event(DomainEventRecord::simple("custom_event"));
        assert_eq!(e.pending_event_count(), 1);
        let events = e.drain_events();
        assert_eq!(events[0].event_type, "custom_event");
    }

    #[test]
    fn test_audit_info() {
        let mut e = Entity::new("e1", "alice");
        assert_eq!(e.audit().created_by, "alice");
        assert!(e.audit().updated_by.is_none());
        e.mark_persisted().unwrap();
        e.set_attribute("k", "v", "bob").unwrap();
        assert_eq!(e.audit().updated_by.as_deref(), Some("bob"));
    }

    #[test]
    fn test_version_checked_apply_success() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        let result = e.apply_versioned(1, "bob", |attrs| {
            attrs.insert("key".to_string(), "val".to_string());
        });
        assert!(result.is_ok());
        assert_eq!(e.get_attribute("key"), Some("val"));
        assert_eq!(e.lifecycle(), EntityLifecycle::Modified);
    }

    #[test]
    fn test_version_checked_apply_conflict() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        let result = e.apply_versioned(0, "bob", |_| {});
        assert!(matches!(result, Err(EntityError::VersionConflict { .. })));
    }

    #[test]
    fn test_from_persisted() {
        let audit = AuditInfo::new("sys");
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), "v".to_string());
        let e = Entity::from_persisted("e1", 5, audit, attrs);
        assert_eq!(e.lifecycle(), EntityLifecycle::Persisted);
        assert_eq!(e.version(), 5);
        assert_eq!(e.get_attribute("k"), Some("v"));
    }

    #[test]
    fn test_remove_attribute() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        e.set_attribute("k", "v", "alice").unwrap();
        e.mark_persisted().unwrap();
        let removed = e.remove_attribute("k", "bob").unwrap();
        assert_eq!(removed.as_deref(), Some("v"));
        assert_eq!(e.lifecycle(), EntityLifecycle::Modified);
    }

    #[test]
    fn test_remove_attribute_not_found() {
        let mut e = Entity::new("e1", "alice");
        e.mark_persisted().unwrap();
        let removed = e.remove_attribute("missing", "bob").unwrap();
        assert!(removed.is_none());
        // Should stay Persisted since nothing changed.
        assert_eq!(e.lifecycle(), EntityLifecycle::Persisted);
    }

    #[test]
    fn test_entity_collection_add_and_get() {
        let mut coll = EntityCollection::new();
        coll.add(Entity::new("e1", "alice")).unwrap();
        coll.add(Entity::new("e2", "bob")).unwrap();
        assert_eq!(coll.len(), 2);
        assert!(coll.get("e1").is_some());
        assert!(coll.get("e3").is_none());
    }

    #[test]
    fn test_entity_collection_duplicate() {
        let mut coll = EntityCollection::new();
        coll.add(Entity::new("e1", "alice")).unwrap();
        let result = coll.add(Entity::new("e1", "bob"));
        assert!(matches!(result, Err(EntityError::Duplicate(_))));
    }

    #[test]
    fn test_entity_collection_remove() {
        let mut coll = EntityCollection::new();
        coll.add(Entity::new("e1", "alice")).unwrap();
        let removed = coll.remove("e1");
        assert!(removed.is_some());
        assert!(coll.is_empty());
    }

    #[test]
    fn test_entity_collection_dirty() {
        let mut coll = EntityCollection::new();
        coll.add(Entity::new("e1", "alice")).unwrap();
        let mut e2 = Entity::new("e2", "bob");
        e2.mark_persisted().unwrap();
        coll.add(e2).unwrap();
        assert_eq!(coll.dirty().len(), 1);
    }

    #[test]
    fn test_entity_collection_drain_all_events() {
        let mut coll = EntityCollection::new();
        let mut e1 = Entity::new("e1", "alice");
        e1.record_event(DomainEventRecord::simple("ev1"));
        coll.add(e1).unwrap();
        let mut e2 = Entity::new("e2", "bob");
        e2.record_event(DomainEventRecord::simple("ev2"));
        coll.add(e2).unwrap();
        let all = coll.drain_all_events();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_lifecycle_transitions_table() {
        use EntityLifecycle::*;
        assert!(New.can_transition_to(Persisted));
        assert!(!New.can_transition_to(Deleted));
        assert!(!New.can_transition_to(Modified));
        assert!(Persisted.can_transition_to(Modified));
        assert!(Persisted.can_transition_to(Deleted));
        assert!(!Persisted.can_transition_to(New));
        assert!(Modified.can_transition_to(Persisted));
        assert!(Modified.can_transition_to(Deleted));
        assert!(!Modified.can_transition_to(New));
        assert!(!Deleted.can_transition_to(New));
        assert!(!Deleted.can_transition_to(Persisted));
    }

    #[test]
    fn test_hash_by_identity() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Entity::new("e1", "a"));
        set.insert(Entity::new("e1", "b"));
        assert_eq!(set.len(), 1);
        set.insert(Entity::new("e2", "c"));
        assert_eq!(set.len(), 2);
    }
}
