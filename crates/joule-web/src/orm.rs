//! Simple ORM model — entity definitions, in-memory repository, dirty tracking.
//!
//! Replaces Prisma, TypeORM, Sequelize with pure Rust entity management.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// ORM errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrmError {
    /// Record not found by the given ID.
    NotFound(String),
    /// Duplicate primary key.
    DuplicateKey(String),
    /// Validation failed.
    ValidationError(String),
    /// Type mismatch on a field.
    TypeMismatch { field: String, expected: String },
}

impl fmt::Display for OrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "record not found: {id}"),
            Self::DuplicateKey(id) => write!(f, "duplicate key: {id}"),
            Self::ValidationError(msg) => write!(f, "validation error: {msg}"),
            Self::TypeMismatch { field, expected } => {
                write!(f, "type mismatch on `{field}`: expected {expected}")
            }
        }
    }
}

impl std::error::Error for OrmError {}

// ── Field Types ─────────────────────────────────────────────────

/// Supported column types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    Text,
    Integer,
    Float,
    Bool,
    Timestamp,
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "TEXT"),
            Self::Integer => write!(f, "INTEGER"),
            Self::Float => write!(f, "FLOAT"),
            Self::Bool => write!(f, "BOOLEAN"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
        }
    }
}

// ── Field ───────────────────────────────────────────────────────

/// A single column / field definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub field_type: FieldType,
    pub nullable: bool,
    pub default: Option<FieldValue>,
}

/// A concrete value stored in a field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    Text(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Timestamp(i64),
    Null,
}

impl FieldValue {
    /// Return the type of this value.
    pub fn field_type(&self) -> Option<FieldType> {
        match self {
            Self::Text(_) => Some(FieldType::Text),
            Self::Integer(_) => Some(FieldType::Integer),
            Self::Float(_) => Some(FieldType::Float),
            Self::Bool(_) => Some(FieldType::Bool),
            Self::Timestamp(_) => Some(FieldType::Timestamp),
            Self::Null => None,
        }
    }
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "{s}"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Timestamp(ts) => write!(f, "{ts}"),
            Self::Null => write!(f, "NULL"),
        }
    }
}

// ── Entity Trait ────────────────────────────────────────────────

/// Trait for types that map to a database table.
pub trait Entity: Clone + fmt::Debug {
    /// The table name.
    fn table_name() -> &'static str;
    /// The primary key column name.
    fn primary_key() -> &'static str;
    /// Get the primary key value for this instance.
    fn pk_value(&self) -> String;
    /// Convert to a field map.
    fn to_fields(&self) -> HashMap<String, FieldValue>;
    /// Reconstruct from a field map.
    fn from_fields(fields: &HashMap<String, FieldValue>) -> Result<Self, OrmError>;
}

// ── Schema ──────────────────────────────────────────────────────

/// Describes the columns of an entity.
#[derive(Debug, Clone)]
pub struct Schema {
    pub entity_name: String,
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn new(entity_name: impl Into<String>) -> Self {
        Self {
            entity_name: entity_name.into(),
            fields: Vec::new(),
        }
    }

    pub fn add_field(&mut self, field: Field) -> &mut Self {
        self.fields.push(field);
        self
    }

    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }
}

// ── Where Clauses ───────────────────────────────────────────────

/// Comparison operators for queries.
#[derive(Debug, Clone)]
pub enum WhereOp {
    Eq(FieldValue),
    Ne(FieldValue),
    Gt(FieldValue),
    Lt(FieldValue),
    Like(String),
}

/// A single where clause.
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub field: String,
    pub op: WhereOp,
}

/// Query builder for filtering repository results.
#[derive(Debug, Clone, Default)]
pub struct Query {
    pub clauses: Vec<WhereClause>,
}

impl Query {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn eq(mut self, field: impl Into<String>, value: FieldValue) -> Self {
        self.clauses.push(WhereClause {
            field: field.into(),
            op: WhereOp::Eq(value),
        });
        self
    }

    pub fn ne(mut self, field: impl Into<String>, value: FieldValue) -> Self {
        self.clauses.push(WhereClause {
            field: field.into(),
            op: WhereOp::Ne(value),
        });
        self
    }

    pub fn gt(mut self, field: impl Into<String>, value: FieldValue) -> Self {
        self.clauses.push(WhereClause {
            field: field.into(),
            op: WhereOp::Gt(value),
        });
        self
    }

    pub fn lt(mut self, field: impl Into<String>, value: FieldValue) -> Self {
        self.clauses.push(WhereClause {
            field: field.into(),
            op: WhereOp::Lt(value),
        });
        self
    }

    pub fn like(mut self, field: impl Into<String>, pattern: impl Into<String>) -> Self {
        self.clauses.push(WhereClause {
            field: field.into(),
            op: WhereOp::Like(pattern.into()),
        });
        self
    }

    /// Test whether a record's fields match all clauses.
    pub fn matches(&self, fields: &HashMap<String, FieldValue>) -> bool {
        self.clauses.iter().all(|clause| {
            let Some(val) = fields.get(&clause.field) else {
                return false;
            };
            match &clause.op {
                WhereOp::Eq(expected) => val == expected,
                WhereOp::Ne(expected) => val != expected,
                WhereOp::Gt(threshold) => cmp_field_values(val, threshold) == Some(std::cmp::Ordering::Greater),
                WhereOp::Lt(threshold) => cmp_field_values(val, threshold) == Some(std::cmp::Ordering::Less),
                WhereOp::Like(pattern) => {
                    if let FieldValue::Text(s) = val {
                        simple_like_match(s, pattern)
                    } else {
                        false
                    }
                }
            }
        })
    }
}

fn cmp_field_values(a: &FieldValue, b: &FieldValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (FieldValue::Integer(x), FieldValue::Integer(y)) => Some(x.cmp(y)),
        (FieldValue::Float(x), FieldValue::Float(y)) => x.partial_cmp(y),
        (FieldValue::Text(x), FieldValue::Text(y)) => Some(x.cmp(y)),
        (FieldValue::Timestamp(x), FieldValue::Timestamp(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Simple LIKE matching: `%` matches any substring.
fn simple_like_match(text: &str, pattern: &str) -> bool {
    let lower_text = text.to_lowercase();
    let lower_pattern = pattern.to_lowercase();
    let parts: Vec<&str> = lower_pattern.split('%').collect();
    if parts.len() == 1 {
        return lower_text == lower_pattern;
    }
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if let Some(found) = lower_text[pos..].find(part) {
            if i == 0 && found != 0 {
                return false; // must match at start
            }
            pos += found + part.len();
        } else {
            return false;
        }
    }
    // If pattern doesn't end with %, text must end at pos
    if !lower_pattern.ends_with('%') {
        return pos == lower_text.len();
    }
    true
}

// ── Dirty Tracking ──────────────────────────────────────────────

/// Tracks which fields have been modified since last save.
#[derive(Debug, Clone)]
pub struct DirtyTracker {
    original: HashMap<String, FieldValue>,
    current: HashMap<String, FieldValue>,
}

impl DirtyTracker {
    pub fn new(fields: HashMap<String, FieldValue>) -> Self {
        Self {
            original: fields.clone(),
            current: fields,
        }
    }

    pub fn set(&mut self, field: impl Into<String>, value: FieldValue) {
        self.current.insert(field.into(), value);
    }

    pub fn changed_fields(&self) -> HashSet<String> {
        let mut changed = HashSet::new();
        for (key, val) in &self.current {
            if self.original.get(key) != Some(val) {
                changed.insert(key.clone());
            }
        }
        changed
    }

    pub fn is_dirty(&self) -> bool {
        !self.changed_fields().is_empty()
    }

    pub fn mark_clean(&mut self) {
        self.original = self.current.clone();
    }

    pub fn current(&self) -> &HashMap<String, FieldValue> {
        &self.current
    }
}

// ── Repository ──────────────────────────────────────────────────

/// In-memory repository for entities.
#[derive(Debug)]
pub struct Repository<T: Entity> {
    records: HashMap<String, T>,
}

impl<T: Entity> Default for Repository<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Entity> Repository<T> {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    pub fn insert(&mut self, entity: T) -> Result<(), OrmError> {
        let pk = entity.pk_value();
        if self.records.contains_key(&pk) {
            return Err(OrmError::DuplicateKey(pk));
        }
        self.records.insert(pk, entity);
        Ok(())
    }

    pub fn find_by_id(&self, id: &str) -> Result<&T, OrmError> {
        self.records
            .get(id)
            .ok_or_else(|| OrmError::NotFound(id.to_string()))
    }

    pub fn find_all(&self) -> Vec<&T> {
        self.records.values().collect()
    }

    pub fn update(&mut self, entity: T) -> Result<(), OrmError> {
        let pk = entity.pk_value();
        if !self.records.contains_key(&pk) {
            return Err(OrmError::NotFound(pk));
        }
        self.records.insert(pk, entity);
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<T, OrmError> {
        self.records
            .remove(id)
            .ok_or_else(|| OrmError::NotFound(id.to_string()))
    }

    pub fn find_where(&self, query: &Query) -> Vec<&T> {
        self.records
            .values()
            .filter(|entity| {
                let fields = entity.to_fields();
                query.matches(&fields)
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.records.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test entity: a simple "User" row.
    #[derive(Debug, Clone)]
    struct User {
        id: String,
        name: String,
        age: i64,
        active: bool,
    }

    impl Entity for User {
        fn table_name() -> &'static str {
            "users"
        }
        fn primary_key() -> &'static str {
            "id"
        }
        fn pk_value(&self) -> String {
            self.id.clone()
        }
        fn to_fields(&self) -> HashMap<String, FieldValue> {
            let mut m = HashMap::new();
            m.insert("id".into(), FieldValue::Text(self.id.clone()));
            m.insert("name".into(), FieldValue::Text(self.name.clone()));
            m.insert("age".into(), FieldValue::Integer(self.age));
            m.insert("active".into(), FieldValue::Bool(self.active));
            m
        }
        fn from_fields(fields: &HashMap<String, FieldValue>) -> Result<Self, OrmError> {
            let id = match fields.get("id") {
                Some(FieldValue::Text(s)) => s.clone(),
                _ => return Err(OrmError::TypeMismatch { field: "id".into(), expected: "Text".into() }),
            };
            let name = match fields.get("name") {
                Some(FieldValue::Text(s)) => s.clone(),
                _ => return Err(OrmError::TypeMismatch { field: "name".into(), expected: "Text".into() }),
            };
            let age = match fields.get("age") {
                Some(FieldValue::Integer(n)) => *n,
                _ => return Err(OrmError::TypeMismatch { field: "age".into(), expected: "Integer".into() }),
            };
            let active = match fields.get("active") {
                Some(FieldValue::Bool(b)) => *b,
                _ => return Err(OrmError::TypeMismatch { field: "active".into(), expected: "Bool".into() }),
            };
            Ok(Self { id, name, age, active })
        }
    }

    fn make_user(id: &str, name: &str, age: i64) -> User {
        User { id: id.into(), name: name.into(), age, active: true }
    }

    #[test]
    fn insert_and_find() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        let found = repo.find_by_id("1").unwrap();
        assert_eq!(found.name, "Alice");
    }

    #[test]
    fn duplicate_key_rejected() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        let err = repo.insert(make_user("1", "Bob", 25)).unwrap_err();
        assert_eq!(err, OrmError::DuplicateKey("1".into()));
    }

    #[test]
    fn find_all_returns_all() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        repo.insert(make_user("2", "Bob", 25)).unwrap();
        assert_eq!(repo.find_all().len(), 2);
    }

    #[test]
    fn update_existing() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        repo.update(make_user("1", "Alice Updated", 31)).unwrap();
        let found = repo.find_by_id("1").unwrap();
        assert_eq!(found.name, "Alice Updated");
        assert_eq!(found.age, 31);
    }

    #[test]
    fn update_missing_fails() {
        let mut repo = Repository::<User>::new();
        let err = repo.update(make_user("99", "Ghost", 0)).unwrap_err();
        assert_eq!(err, OrmError::NotFound("99".into()));
    }

    #[test]
    fn delete_returns_entity() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        let removed = repo.delete("1").unwrap();
        assert_eq!(removed.name, "Alice");
        assert_eq!(repo.count(), 0);
    }

    #[test]
    fn delete_missing_fails() {
        let mut repo = Repository::<User>::new();
        let err = repo.delete("99").unwrap_err();
        assert_eq!(err, OrmError::NotFound("99".into()));
    }

    #[test]
    fn query_eq_filter() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        repo.insert(make_user("2", "Bob", 25)).unwrap();
        let q = Query::new().eq("name", FieldValue::Text("Bob".into()));
        let results = repo.find_where(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Bob");
    }

    #[test]
    fn query_gt_lt() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        repo.insert(make_user("2", "Bob", 25)).unwrap();
        repo.insert(make_user("3", "Charlie", 35)).unwrap();
        let q = Query::new()
            .gt("age", FieldValue::Integer(26))
            .lt("age", FieldValue::Integer(34));
        let results = repo.find_where(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");
    }

    #[test]
    fn query_like_pattern() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice Smith", 30)).unwrap();
        repo.insert(make_user("2", "Bob Jones", 25)).unwrap();
        repo.insert(make_user("3", "Alice Johnson", 28)).unwrap();
        let q = Query::new().like("name", "Alice%");
        let results = repo.find_where(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn dirty_tracker_detects_changes() {
        let fields: HashMap<String, FieldValue> = [
            ("name".into(), FieldValue::Text("Alice".into())),
            ("age".into(), FieldValue::Integer(30)),
        ]
        .into_iter()
        .collect();
        let mut tracker = DirtyTracker::new(fields);
        assert!(!tracker.is_dirty());

        tracker.set("age", FieldValue::Integer(31));
        assert!(tracker.is_dirty());
        let changed = tracker.changed_fields();
        assert!(changed.contains("age"));
        assert!(!changed.contains("name"));

        tracker.mark_clean();
        assert!(!tracker.is_dirty());
    }

    #[test]
    fn schema_field_names() {
        let mut schema = Schema::new("users");
        schema.add_field(Field {
            name: "id".into(),
            field_type: FieldType::Text,
            nullable: false,
            default: None,
        });
        schema.add_field(Field {
            name: "name".into(),
            field_type: FieldType::Text,
            nullable: false,
            default: None,
        });
        assert_eq!(schema.field_names(), vec!["id", "name"]);
    }

    #[test]
    fn entity_roundtrip() {
        let user = make_user("42", "Test", 99);
        let fields = user.to_fields();
        let restored = User::from_fields(&fields).unwrap();
        assert_eq!(restored.id, "42");
        assert_eq!(restored.name, "Test");
        assert_eq!(restored.age, 99);
    }

    #[test]
    fn query_ne_filter() {
        let mut repo = Repository::<User>::new();
        repo.insert(make_user("1", "Alice", 30)).unwrap();
        repo.insert(make_user("2", "Bob", 25)).unwrap();
        let q = Query::new().ne("name", FieldValue::Text("Alice".into()));
        let results = repo.find_where(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Bob");
    }

    #[test]
    fn like_middle_wildcard() {
        assert!(simple_like_match("hello world", "%lo wo%"));
        assert!(!simple_like_match("hello world", "%xyz%"));
    }

    #[test]
    fn field_value_display() {
        assert_eq!(FieldValue::Integer(42).to_string(), "42");
        assert_eq!(FieldValue::Null.to_string(), "NULL");
        assert_eq!(FieldValue::Bool(true).to_string(), "true");
    }
}
