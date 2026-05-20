//! Repository pattern — in-memory repository, CRUD operations, query by
//! specification, pagination support, unit of work (change tracking),
//! optimistic concurrency (version check), and repository statistics.
//!
//! Replaces ad-hoc data access in JS/TS (TypeORM Repository, Prisma client)
//! with a pure-Rust repository that enforces domain boundaries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Repository domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepositoryError {
    /// Entity not found.
    NotFound(String),
    /// Entity already exists.
    AlreadyExists(String),
    /// Version conflict.
    VersionConflict { id: String, expected: u64, actual: u64 },
    /// Query error.
    QueryError(String),
    /// Unit of work error.
    UnitOfWorkError(String),
    /// Specification error.
    SpecificationError(String),
}

impl std::fmt::Display for RepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "entity not found: {id}"),
            Self::AlreadyExists(id) => write!(f, "entity already exists: {id}"),
            Self::VersionConflict { id, expected, actual } => {
                write!(f, "version conflict for {id}: expected {expected}, got {actual}")
            }
            Self::QueryError(msg) => write!(f, "query error: {msg}"),
            Self::UnitOfWorkError(msg) => write!(f, "unit of work error: {msg}"),
            Self::SpecificationError(msg) => write!(f, "specification error: {msg}"),
        }
    }
}

impl std::error::Error for RepositoryError {}

// ── Storable Entity ─────────────────────────────────────────────

/// An entity record stored in the repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorableEntity {
    pub id: String,
    pub version: u64,
    pub entity_type: String,
    pub data: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl StorableEntity {
    pub fn new(
        id: impl Into<String>,
        entity_type: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            version: 1,
            entity_type: entity_type.into(),
            data,
            created_at: now,
            updated_at: now,
        }
    }
}

// ── Specification ───────────────────────────────────────────────

/// A predicate for querying entities.
#[derive(Debug, Clone)]
pub enum Specification {
    /// Field equals value.
    FieldEquals { field: String, value: String },
    /// Field contains substring.
    FieldContains { field: String, substring: String },
    /// Entity type equals.
    TypeEquals(String),
    /// All of these specs must match.
    And(Vec<Specification>),
    /// At least one must match.
    Or(Vec<Specification>),
    /// Negation.
    Not(Box<Specification>),
    /// Always true.
    All,
}

impl Specification {
    pub fn field_equals(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::FieldEquals { field: field.into(), value: value.into() }
    }

    pub fn field_contains(field: impl Into<String>, substring: impl Into<String>) -> Self {
        Self::FieldContains { field: field.into(), substring: substring.into() }
    }

    pub fn type_equals(entity_type: impl Into<String>) -> Self {
        Self::TypeEquals(entity_type.into())
    }

    pub fn and(specs: Vec<Specification>) -> Self {
        Self::And(specs)
    }

    pub fn or(specs: Vec<Specification>) -> Self {
        Self::Or(specs)
    }

    pub fn not(spec: Specification) -> Self {
        Self::Not(Box::new(spec))
    }

    /// Evaluate this specification against an entity.
    pub fn matches(&self, entity: &StorableEntity) -> bool {
        match self {
            Self::FieldEquals { field, value } => {
                entity.data.get(field).map(|v| v == value).unwrap_or(false)
            }
            Self::FieldContains { field, substring } => {
                entity.data.get(field).map(|v| v.contains(substring.as_str())).unwrap_or(false)
            }
            Self::TypeEquals(t) => entity.entity_type == *t,
            Self::And(specs) => specs.iter().all(|s| s.matches(entity)),
            Self::Or(specs) => specs.iter().any(|s| s.matches(entity)),
            Self::Not(spec) => !spec.matches(entity),
            Self::All => true,
        }
    }
}

// ── Sort Order ──────────────────────────────────────────────────

/// Sort order for queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Sort by a specific field.
#[derive(Debug, Clone)]
pub struct SortBy {
    pub field: String,
    pub order: SortOrder,
}

impl SortBy {
    pub fn asc(field: impl Into<String>) -> Self {
        Self { field: field.into(), order: SortOrder::Ascending }
    }

    pub fn desc(field: impl Into<String>) -> Self {
        Self { field: field.into(), order: SortOrder::Descending }
    }
}

// ── Page ────────────────────────────────────────────────────────

/// A paginated result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub page_number: usize,
    pub page_size: usize,
    pub total_items: usize,
}

impl<T> Page<T> {
    pub fn total_pages(&self) -> usize {
        if self.page_size == 0 { return 0; }
        (self.total_items + self.page_size - 1) / self.page_size
    }

    pub fn has_next(&self) -> bool {
        self.page_number < self.total_pages()
    }

    pub fn has_previous(&self) -> bool {
        self.page_number > 1
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ── UnitOfWork ──────────────────────────────────────────────────

/// Change tracking status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeStatus {
    Added,
    Modified,
    Deleted,
}

/// A tracked change in the unit of work.
#[derive(Debug, Clone)]
pub struct TrackedChange {
    pub entity_id: String,
    pub status: ChangeStatus,
    pub entity: Option<StorableEntity>,
}

/// Unit of work — tracks changes and commits them atomically.
#[derive(Debug)]
pub struct UnitOfWork {
    changes: Vec<TrackedChange>,
    committed: bool,
}

impl UnitOfWork {
    pub fn new() -> Self {
        Self { changes: Vec::new(), committed: false }
    }

    /// Track an addition.
    pub fn track_add(&mut self, entity: StorableEntity) {
        self.changes.push(TrackedChange {
            entity_id: entity.id.clone(),
            status: ChangeStatus::Added,
            entity: Some(entity),
        });
    }

    /// Track a modification.
    pub fn track_modify(&mut self, entity: StorableEntity) {
        self.changes.push(TrackedChange {
            entity_id: entity.id.clone(),
            status: ChangeStatus::Modified,
            entity: Some(entity),
        });
    }

    /// Track a deletion.
    pub fn track_delete(&mut self, entity_id: impl Into<String>) {
        let entity_id = entity_id.into();
        self.changes.push(TrackedChange {
            entity_id,
            status: ChangeStatus::Deleted,
            entity: None,
        });
    }

    /// Number of pending changes.
    pub fn pending_count(&self) -> usize {
        self.changes.len()
    }

    /// Whether already committed.
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Drain changes.
    pub fn drain_changes(&mut self) -> Vec<TrackedChange> {
        self.committed = true;
        std::mem::take(&mut self.changes)
    }

    /// Clear without committing.
    pub fn rollback(&mut self) {
        self.changes.clear();
    }
}

impl Default for UnitOfWork {
    fn default() -> Self {
        Self::new()
    }
}

// ── Repository Statistics ───────────────────────────────────────

/// Repository usage statistics.
#[derive(Debug, Clone, Default)]
pub struct RepoStats {
    pub total_entities: usize,
    pub entity_types: HashMap<String, usize>,
    pub total_reads: u64,
    pub total_writes: u64,
    pub total_deletes: u64,
    pub total_queries: u64,
}

// ── InMemoryRepository ──────────────────────────────────────────

/// An in-memory repository with CRUD, query, pagination, and versioning.
#[derive(Debug)]
pub struct InMemoryRepository {
    entities: HashMap<String, StorableEntity>,
    stats: RepoStats,
}

impl InMemoryRepository {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            stats: RepoStats::default(),
        }
    }

    /// Add a new entity.
    pub fn add(&mut self, entity: StorableEntity) -> Result<(), RepositoryError> {
        if self.entities.contains_key(&entity.id) {
            return Err(RepositoryError::AlreadyExists(entity.id.clone()));
        }
        let et = entity.entity_type.clone();
        self.entities.insert(entity.id.clone(), entity);
        *self.stats.entity_types.entry(et).or_insert(0) += 1;
        self.stats.total_entities += 1;
        self.stats.total_writes += 1;
        Ok(())
    }

    /// Get by id.
    pub fn get(&mut self, id: &str) -> Result<&StorableEntity, RepositoryError> {
        self.stats.total_reads += 1;
        self.entities.get(id).ok_or_else(|| RepositoryError::NotFound(id.to_string()))
    }

    /// Get by id (immutable borrow, no stats update).
    pub fn peek(&self, id: &str) -> Option<&StorableEntity> {
        self.entities.get(id)
    }

    /// Update with version check.
    pub fn update(&mut self, entity: StorableEntity) -> Result<(), RepositoryError> {
        let existing = self.entities.get(&entity.id)
            .ok_or_else(|| RepositoryError::NotFound(entity.id.clone()))?;
        if existing.version != entity.version.saturating_sub(1) {
            return Err(RepositoryError::VersionConflict {
                id: entity.id.clone(),
                expected: existing.version + 1,
                actual: entity.version,
            });
        }
        self.entities.insert(entity.id.clone(), entity);
        self.stats.total_writes += 1;
        Ok(())
    }

    /// Delete by id.
    pub fn delete(&mut self, id: &str) -> Result<StorableEntity, RepositoryError> {
        let entity = self.entities.remove(id)
            .ok_or_else(|| RepositoryError::NotFound(id.to_string()))?;
        if let Some(count) = self.stats.entity_types.get_mut(&entity.entity_type) {
            *count = count.saturating_sub(1);
        }
        self.stats.total_entities = self.stats.total_entities.saturating_sub(1);
        self.stats.total_deletes += 1;
        Ok(entity)
    }

    /// Query by specification.
    pub fn query(&mut self, spec: &Specification) -> Vec<&StorableEntity> {
        self.stats.total_queries += 1;
        self.entities.values().filter(|e| spec.matches(e)).collect()
    }

    /// Query with pagination.
    pub fn query_paged(
        &mut self,
        spec: &Specification,
        page: usize,
        page_size: usize,
    ) -> Page<StorableEntity> {
        self.stats.total_queries += 1;
        let mut matching: Vec<StorableEntity> = self.entities.values()
            .filter(|e| spec.matches(e))
            .cloned()
            .collect();
        // Sort by id for deterministic pagination.
        matching.sort_by(|a, b| a.id.cmp(&b.id));
        let total = matching.len();
        let start = (page.saturating_sub(1)) * page_size;
        let items: Vec<StorableEntity> = matching.into_iter().skip(start).take(page_size).collect();
        Page {
            items,
            page_number: page,
            page_size,
            total_items: total,
        }
    }

    /// Count entities matching a specification.
    pub fn count(&self, spec: &Specification) -> usize {
        self.entities.values().filter(|e| spec.matches(e)).count()
    }

    /// Whether an entity exists by id.
    pub fn exists(&self, id: &str) -> bool {
        self.entities.contains_key(id)
    }

    /// Total entity count.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the repository is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Apply a unit of work.
    pub fn apply_unit_of_work(&mut self, uow: &mut UnitOfWork) -> Result<(), RepositoryError> {
        let changes = uow.drain_changes();
        for change in changes {
            match change.status {
                ChangeStatus::Added => {
                    if let Some(entity) = change.entity {
                        self.add(entity)?;
                    }
                }
                ChangeStatus::Modified => {
                    if let Some(entity) = change.entity {
                        self.update(entity)?;
                    }
                }
                ChangeStatus::Deleted => {
                    self.delete(&change.entity_id)?;
                }
            }
        }
        Ok(())
    }

    /// Get repository statistics.
    pub fn stats(&self) -> &RepoStats {
        &self.stats
    }

    /// Clear all entities.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.stats.total_entities = 0;
        self.stats.entity_types.clear();
    }
}

impl Default for InMemoryRepository {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, etype: &str) -> StorableEntity {
        let mut data = HashMap::new();
        data.insert("name".to_string(), id.to_string());
        StorableEntity::new(id, etype, data)
    }

    #[test]
    fn test_add_and_get() {
        let mut repo = InMemoryRepository::new();
        let e = make_entity("e1", "user");
        repo.add(e).unwrap();
        let found = repo.get("e1").unwrap();
        assert_eq!(found.id, "e1");
    }

    #[test]
    fn test_add_duplicate() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        let result = repo.add(make_entity("e1", "user"));
        assert!(matches!(result, Err(RepositoryError::AlreadyExists(_))));
    }

    #[test]
    fn test_get_not_found() {
        let mut repo = InMemoryRepository::new();
        let result = repo.get("missing");
        assert!(matches!(result, Err(RepositoryError::NotFound(_))));
    }

    #[test]
    fn test_update_with_version() {
        let mut repo = InMemoryRepository::new();
        let e = make_entity("e1", "user");
        repo.add(e).unwrap();
        let mut updated = make_entity("e1", "user");
        updated.version = 2;
        updated.data.insert("name".to_string(), "updated".to_string());
        repo.update(updated).unwrap();
        let found = repo.get("e1").unwrap();
        assert_eq!(found.version, 2);
    }

    #[test]
    fn test_update_version_conflict() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        let mut updated = make_entity("e1", "user");
        updated.version = 5; // wrong version
        let result = repo.update(updated);
        assert!(matches!(result, Err(RepositoryError::VersionConflict { .. })));
    }

    #[test]
    fn test_delete() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        let deleted = repo.delete("e1").unwrap();
        assert_eq!(deleted.id, "e1");
        assert!(!repo.exists("e1"));
    }

    #[test]
    fn test_delete_not_found() {
        let mut repo = InMemoryRepository::new();
        let result = repo.delete("missing");
        assert!(matches!(result, Err(RepositoryError::NotFound(_))));
    }

    #[test]
    fn test_query_field_equals() {
        let mut repo = InMemoryRepository::new();
        let mut e = make_entity("e1", "user");
        e.data.insert("status".to_string(), "active".to_string());
        repo.add(e).unwrap();
        let mut e2 = make_entity("e2", "user");
        e2.data.insert("status".to_string(), "inactive".to_string());
        repo.add(e2).unwrap();
        let spec = Specification::field_equals("status", "active");
        let results = repo.query(&spec);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "e1");
    }

    #[test]
    fn test_query_type_equals() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        repo.add(make_entity("e2", "order")).unwrap();
        let spec = Specification::type_equals("user");
        let results = repo.query(&spec);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_and() {
        let mut repo = InMemoryRepository::new();
        let mut e = make_entity("e1", "user");
        e.data.insert("status".to_string(), "active".to_string());
        repo.add(e).unwrap();
        let spec = Specification::and(vec![
            Specification::type_equals("user"),
            Specification::field_equals("status", "active"),
        ]);
        let results = repo.query(&spec);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_or() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        repo.add(make_entity("e2", "order")).unwrap();
        repo.add(make_entity("e3", "product")).unwrap();
        let spec = Specification::or(vec![
            Specification::type_equals("user"),
            Specification::type_equals("order"),
        ]);
        let results = repo.query(&spec);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_not() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        repo.add(make_entity("e2", "order")).unwrap();
        let spec = Specification::not(Specification::type_equals("user"));
        let results = repo.query(&spec);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_type, "order");
    }

    #[test]
    fn test_query_paged() {
        let mut repo = InMemoryRepository::new();
        for i in 0..10 {
            repo.add(make_entity(&format!("e{i:02}"), "user")).unwrap();
        }
        let page1 = repo.query_paged(&Specification::All, 1, 3);
        assert_eq!(page1.items.len(), 3);
        assert_eq!(page1.total_items, 10);
        assert_eq!(page1.total_pages(), 4);
        assert!(page1.has_next());
        assert!(!page1.has_previous());

        let page4 = repo.query_paged(&Specification::All, 4, 3);
        assert_eq!(page4.items.len(), 1);
        assert!(!page4.has_next());
        assert!(page4.has_previous());
    }

    #[test]
    fn test_count() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        repo.add(make_entity("e2", "user")).unwrap();
        repo.add(make_entity("e3", "order")).unwrap();
        assert_eq!(repo.count(&Specification::type_equals("user")), 2);
    }

    #[test]
    fn test_unit_of_work_add() {
        let mut repo = InMemoryRepository::new();
        let mut uow = UnitOfWork::new();
        uow.track_add(make_entity("e1", "user"));
        uow.track_add(make_entity("e2", "order"));
        assert_eq!(uow.pending_count(), 2);
        repo.apply_unit_of_work(&mut uow).unwrap();
        assert!(uow.is_committed());
        assert_eq!(repo.len(), 2);
    }

    #[test]
    fn test_unit_of_work_delete() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        let mut uow = UnitOfWork::new();
        uow.track_delete("e1");
        repo.apply_unit_of_work(&mut uow).unwrap();
        assert!(!repo.exists("e1"));
    }

    #[test]
    fn test_unit_of_work_rollback() {
        let mut uow = UnitOfWork::new();
        uow.track_add(make_entity("e1", "user"));
        uow.rollback();
        assert_eq!(uow.pending_count(), 0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        repo.add(make_entity("e2", "user")).unwrap();
        let _ = repo.get("e1");
        let _ = repo.query(&Specification::All);
        let stats = repo.stats();
        assert_eq!(stats.total_entities, 2);
        assert_eq!(stats.total_writes, 2);
        assert_eq!(stats.total_reads, 1);
        assert_eq!(stats.total_queries, 1);
    }

    #[test]
    fn test_field_contains() {
        let mut repo = InMemoryRepository::new();
        let mut e = make_entity("e1", "user");
        e.data.insert("bio".to_string(), "hello world".to_string());
        repo.add(e).unwrap();
        let spec = Specification::field_contains("bio", "world");
        let results = repo.query(&spec);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        repo.clear();
        assert!(repo.is_empty());
        assert_eq!(repo.stats().total_entities, 0);
    }

    #[test]
    fn test_peek_no_stats() {
        let mut repo = InMemoryRepository::new();
        repo.add(make_entity("e1", "user")).unwrap();
        assert!(repo.peek("e1").is_some());
        assert!(repo.peek("missing").is_none());
        assert_eq!(repo.stats().total_reads, 0);
    }
}
