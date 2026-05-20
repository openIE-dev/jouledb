//! Change tracking.
//!
//! Track field-level changes to objects via change sets with old/new values.
//! Supports merging change sets, conflict detection, applying/rolling back
//! changes, change notification callbacks, and dirty tracking.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// Error type for change set operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeError {
    /// A field conflict was detected during merge.
    Conflict { field: String, message: String },
    /// The field was not found.
    FieldNotFound(String),
    /// The expected old value did not match.
    StaleValue { field: String, expected: String, actual: String },
    /// A notification callback failed.
    NotifyFailed(String),
}

impl fmt::Display for ChangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict { field, message } => {
                write!(f, "conflict on '{}': {}", field, message)
            }
            Self::FieldNotFound(s) => write!(f, "field not found: {}", s),
            Self::StaleValue { field, expected, actual } => {
                write!(
                    f,
                    "stale value for '{}': expected {}, got {}",
                    field, expected, actual
                )
            }
            Self::NotifyFailed(s) => write!(f, "notify failed: {}", s),
        }
    }
}

/// A single field change.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    /// The field path (e.g. "name", "address.city").
    pub field: String,
    /// The old value (None if the field was newly added).
    pub old_value: Option<Value>,
    /// The new value (None if the field was removed).
    pub new_value: Option<Value>,
    /// Timestamp of the change (milliseconds since epoch).
    pub timestamp_ms: i64,
}

impl fmt::Display for FieldChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let old = match &self.old_value {
            Some(v) => v.to_string(),
            None => "<none>".into(),
        };
        let new = match &self.new_value {
            Some(v) => v.to_string(),
            None => "<none>".into(),
        };
        write!(f, "{}: {} -> {}", self.field, old, new)
    }
}

/// A set of changes to an object.
#[derive(Debug, Clone)]
pub struct ChangeSet {
    /// The changes, keyed by field name.
    changes: HashMap<String, FieldChange>,
    /// Optional identifier for the object being tracked.
    pub object_id: Option<String>,
    /// Description of this change set.
    pub description: Option<String>,
}

impl ChangeSet {
    /// Create a new empty change set.
    pub fn new() -> Self {
        Self {
            changes: HashMap::new(),
            object_id: None,
            description: None,
        }
    }

    /// Create a change set for a specific object.
    pub fn for_object(object_id: &str) -> Self {
        Self {
            changes: HashMap::new(),
            object_id: Some(object_id.to_string()),
            description: None,
        }
    }

    /// Record a field change.
    pub fn record(
        &mut self,
        field: &str,
        old_value: Option<Value>,
        new_value: Option<Value>,
        timestamp_ms: i64,
    ) {
        let change = FieldChange {
            field: field.to_string(),
            old_value,
            new_value,
            timestamp_ms,
        };
        self.changes.insert(field.to_string(), change);
    }

    /// Record a value change (convenience for non-optional values).
    pub fn set(&mut self, field: &str, old: Value, new: Value, timestamp_ms: i64) {
        self.record(field, Some(old), Some(new), timestamp_ms);
    }

    /// Record a field addition.
    pub fn add(&mut self, field: &str, value: Value, timestamp_ms: i64) {
        self.record(field, None, Some(value), timestamp_ms);
    }

    /// Record a field removal.
    pub fn remove(&mut self, field: &str, old_value: Value, timestamp_ms: i64) {
        self.record(field, Some(old_value), None, timestamp_ms);
    }

    /// Get a specific field change.
    pub fn get(&self, field: &str) -> Option<&FieldChange> {
        self.changes.get(field)
    }

    /// Check if a field has been changed.
    pub fn has_change(&self, field: &str) -> bool {
        self.changes.contains_key(field)
    }

    /// Get all changed field names.
    pub fn changed_fields(&self) -> Vec<String> {
        let mut fields: Vec<String> = self.changes.keys().cloned().collect();
        fields.sort();
        fields
    }

    /// Number of changes.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Whether there are no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Iterate over all changes.
    pub fn iter(&self) -> impl Iterator<Item = &FieldChange> {
        self.changes.values()
    }

    /// Clear all recorded changes.
    pub fn clear(&mut self) {
        self.changes.clear();
    }
}

// ── Merge ──────────────────────────────────────────────────────────

/// A conflict found during merge.
#[derive(Debug, Clone)]
pub struct MergeConflict {
    pub field: String,
    pub ours: FieldChange,
    pub theirs: FieldChange,
}

/// Result of merging two change sets.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged change set.
    pub merged: ChangeSet,
    /// Any conflicts that were found.
    pub conflicts: Vec<MergeConflict>,
}

/// Merge two change sets. Non-overlapping changes are combined.
/// Overlapping changes produce conflicts.
pub fn merge(a: &ChangeSet, b: &ChangeSet) -> MergeResult {
    let mut merged = ChangeSet::new();
    let mut conflicts = Vec::new();

    // Add all changes from a.
    for (field, change) in &a.changes {
        merged.changes.insert(field.clone(), change.clone());
    }

    // Add changes from b, detecting conflicts.
    for (field, b_change) in &b.changes {
        if let Some(a_change) = a.changes.get(field) {
            // Both modified the same field.
            if a_change.new_value == b_change.new_value {
                // Same new value — no conflict. Keep either (same result).
            } else {
                // Conflict: different new values.
                conflicts.push(MergeConflict {
                    field: field.clone(),
                    ours: a_change.clone(),
                    theirs: b_change.clone(),
                });
            }
        } else {
            merged.changes.insert(field.clone(), b_change.clone());
        }
    }

    MergeResult { merged, conflicts }
}

// ── Apply / Rollback ───────────────────────────────────────────────

/// Apply a change set to a JSON object.
pub fn apply_changes(doc: &mut Value, changes: &ChangeSet) -> Result<(), ChangeError> {
    let obj = doc
        .as_object_mut()
        .ok_or_else(|| ChangeError::FieldNotFound("document is not an object".into()))?;

    // Collect changes into a vec to avoid borrow issues.
    let change_list: Vec<(String, Option<Value>, Option<Value>)> = changes
        .iter()
        .map(|c| (c.field.clone(), c.old_value.clone(), c.new_value.clone()))
        .collect();

    for (field, old_value, new_value) in change_list {
        // Verify the current value matches the expected old value.
        if let Some(expected_old) = &old_value {
            let current = obj.get(&field);
            if current != Some(expected_old) {
                let actual_str = current
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "<missing>".into());
                return Err(ChangeError::StaleValue {
                    field: field.clone(),
                    expected: expected_old.to_string(),
                    actual: actual_str,
                });
            }
        }

        match new_value {
            Some(v) => {
                obj.insert(field, v);
            }
            None => {
                obj.remove(&field);
            }
        }
    }

    Ok(())
}

/// Rollback a change set (restore old values).
pub fn rollback_changes(doc: &mut Value, changes: &ChangeSet) -> Result<(), ChangeError> {
    let obj = doc
        .as_object_mut()
        .ok_or_else(|| ChangeError::FieldNotFound("document is not an object".into()))?;

    let change_list: Vec<(String, Option<Value>)> = changes
        .iter()
        .map(|c| (c.field.clone(), c.old_value.clone()))
        .collect();

    for (field, old_value) in change_list {
        match old_value {
            Some(v) => {
                obj.insert(field, v);
            }
            None => {
                obj.remove(&field);
            }
        }
    }

    Ok(())
}

// ── Conflict detection ─────────────────────────────────────────────

/// Check if two change sets have conflicting changes.
pub fn has_conflicts(a: &ChangeSet, b: &ChangeSet) -> bool {
    for (field, a_change) in &a.changes {
        if let Some(b_change) = b.changes.get(field) {
            if a_change.new_value != b_change.new_value {
                return true;
            }
        }
    }
    false
}

/// Find all conflicting fields between two change sets.
pub fn find_conflicts(a: &ChangeSet, b: &ChangeSet) -> Vec<String> {
    let mut conflicts = Vec::new();
    for (field, a_change) in &a.changes {
        if let Some(b_change) = b.changes.get(field) {
            if a_change.new_value != b_change.new_value {
                conflicts.push(field.clone());
            }
        }
    }
    conflicts.sort();
    conflicts
}

// ── Dirty tracking ─────────────────────────────────────────────────

/// A dirty tracker that monitors which fields have been modified.
#[derive(Debug, Clone)]
pub struct DirtyTracker {
    /// The original values (snapshot).
    original: HashMap<String, Value>,
    /// The current values.
    current: HashMap<String, Value>,
}

impl DirtyTracker {
    /// Create a tracker from a JSON object.
    pub fn from_value(value: &Value) -> Self {
        let mut original = HashMap::new();
        if let Some(obj) = value.as_object() {
            for (k, v) in obj {
                original.insert(k.clone(), v.clone());
            }
        }
        Self {
            current: original.clone(),
            original,
        }
    }

    /// Update a field value.
    pub fn set_field(&mut self, field: &str, value: Value) {
        self.current.insert(field.to_string(), value);
    }

    /// Remove a field.
    pub fn remove_field(&mut self, field: &str) {
        self.current.remove(field);
    }

    /// Check if any field has been modified.
    pub fn is_dirty(&self) -> bool {
        self.original != self.current
    }

    /// Check if a specific field is dirty.
    pub fn is_field_dirty(&self, field: &str) -> bool {
        self.original.get(field) != self.current.get(field)
    }

    /// Get all dirty field names.
    pub fn dirty_fields(&self) -> Vec<String> {
        let mut fields = Vec::new();
        // Check modified/removed fields.
        for (k, v) in &self.original {
            match self.current.get(k) {
                Some(cv) if cv != v => fields.push(k.clone()),
                None => fields.push(k.clone()),
                _ => {}
            }
        }
        // Check added fields.
        for k in self.current.keys() {
            if !self.original.contains_key(k) {
                fields.push(k.clone());
            }
        }
        fields.sort();
        fields
    }

    /// Generate a ChangeSet from the current dirty state.
    pub fn to_change_set(&self, timestamp_ms: i64) -> ChangeSet {
        let mut cs = ChangeSet::new();
        for (k, v) in &self.original {
            match self.current.get(k) {
                Some(cv) if cv != v => {
                    cs.set(k, v.clone(), cv.clone(), timestamp_ms);
                }
                None => {
                    cs.remove(k, v.clone(), timestamp_ms);
                }
                _ => {}
            }
        }
        for (k, v) in &self.current {
            if !self.original.contains_key(k) {
                cs.add(k, v.clone(), timestamp_ms);
            }
        }
        cs
    }

    /// Reset the tracker to the current state (clear dirty flags).
    pub fn commit(&mut self) {
        self.original = self.current.clone();
    }

    /// Revert to original values.
    pub fn revert(&mut self) {
        self.current = self.original.clone();
    }
}

// ── Change notification ────────────────────────────────────────────

/// A type for change notification callbacks.
pub type ChangeCallback = Box<dyn Fn(&FieldChange) -> Result<(), String>>;

/// A change notifier that invokes callbacks when changes are recorded.
pub struct ChangeNotifier {
    change_set: ChangeSet,
    callbacks: Vec<ChangeCallback>,
}

impl ChangeNotifier {
    /// Create a new notifier.
    pub fn new() -> Self {
        Self {
            change_set: ChangeSet::new(),
            callbacks: Vec::new(),
        }
    }

    /// Register a callback.
    pub fn on_change(&mut self, callback: ChangeCallback) {
        self.callbacks.push(callback);
    }

    /// Record a change and notify callbacks.
    pub fn record(
        &mut self,
        field: &str,
        old_value: Option<Value>,
        new_value: Option<Value>,
        timestamp_ms: i64,
    ) -> Result<(), ChangeError> {
        let change = FieldChange {
            field: field.to_string(),
            old_value: old_value.clone(),
            new_value: new_value.clone(),
            timestamp_ms,
        };

        for cb in &self.callbacks {
            cb(&change).map_err(|e| ChangeError::NotifyFailed(e))?;
        }

        self.change_set
            .record(field, old_value, new_value, timestamp_ms);
        Ok(())
    }

    /// Get the underlying change set.
    pub fn change_set(&self) -> &ChangeSet {
        &self.change_set
    }

    /// Number of registered callbacks.
    pub fn callback_count(&self) -> usize {
        self.callbacks.len()
    }
}

// ── Diff between JSON values ───────────────────────────────────────

/// Generate a ChangeSet by diffing two JSON objects.
pub fn diff_objects(old: &Value, new: &Value, timestamp_ms: i64) -> ChangeSet {
    let mut cs = ChangeSet::new();

    let old_obj = old.as_object();
    let new_obj = new.as_object();

    match (old_obj, new_obj) {
        (Some(o), Some(n)) => {
            // Removed fields.
            for (k, v) in o {
                if !n.contains_key(k) {
                    cs.remove(k, v.clone(), timestamp_ms);
                }
            }
            // Changed or added fields.
            for (k, v) in n {
                match o.get(k) {
                    Some(ov) if ov != v => {
                        cs.set(k, ov.clone(), v.clone(), timestamp_ms);
                    }
                    None => {
                        cs.add(k, v.clone(), timestamp_ms);
                    }
                    _ => {}
                }
            }
        }
        _ => {
            // Non-object: treat as a single root change.
            if old != new {
                cs.record("", Some(old.clone()), Some(new.clone()), timestamp_ms);
            }
        }
    }

    cs
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_change_set() {
        let cs = ChangeSet::new();
        assert!(cs.is_empty());
        assert_eq!(cs.len(), 0);
    }

    #[test]
    fn record_and_get() {
        let mut cs = ChangeSet::new();
        cs.set("name", json!("alice"), json!("bob"), 1000);
        assert!(cs.has_change("name"));
        let change = cs.get("name").unwrap();
        assert_eq!(change.old_value, Some(json!("alice")));
        assert_eq!(change.new_value, Some(json!("bob")));
    }

    #[test]
    fn add_field() {
        let mut cs = ChangeSet::new();
        cs.add("email", json!("a@b.com"), 1000);
        let change = cs.get("email").unwrap();
        assert!(change.old_value.is_none());
        assert_eq!(change.new_value, Some(json!("a@b.com")));
    }

    #[test]
    fn remove_field() {
        let mut cs = ChangeSet::new();
        cs.remove("legacy", json!(true), 1000);
        let change = cs.get("legacy").unwrap();
        assert_eq!(change.old_value, Some(json!(true)));
        assert!(change.new_value.is_none());
    }

    #[test]
    fn changed_fields_sorted() {
        let mut cs = ChangeSet::new();
        cs.set("z", json!(1), json!(2), 1000);
        cs.set("a", json!(3), json!(4), 1000);
        let fields = cs.changed_fields();
        assert_eq!(fields, vec!["a", "z"]);
    }

    #[test]
    fn merge_non_overlapping() {
        let mut a = ChangeSet::new();
        a.set("name", json!("old"), json!("new"), 1000);
        let mut b = ChangeSet::new();
        b.set("age", json!(30), json!(31), 1000);

        let result = merge(&a, &b);
        assert!(result.conflicts.is_empty());
        assert_eq!(result.merged.len(), 2);
    }

    #[test]
    fn merge_with_conflict() {
        let mut a = ChangeSet::new();
        a.set("name", json!("base"), json!("alice"), 1000);
        let mut b = ChangeSet::new();
        b.set("name", json!("base"), json!("bob"), 1000);

        let result = merge(&a, &b);
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(result.conflicts[0].field, "name");
    }

    #[test]
    fn merge_same_value_no_conflict() {
        let mut a = ChangeSet::new();
        a.set("name", json!("base"), json!("same"), 1000);
        let mut b = ChangeSet::new();
        b.set("name", json!("base"), json!("same"), 1000);

        let result = merge(&a, &b);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn apply_changes_success() {
        let mut doc = json!({"name": "alice", "age": 30});
        let mut cs = ChangeSet::new();
        cs.set("name", json!("alice"), json!("bob"), 1000);
        apply_changes(&mut doc, &cs).unwrap();
        assert_eq!(doc["name"], json!("bob"));
    }

    #[test]
    fn apply_changes_stale() {
        let mut doc = json!({"name": "charlie"});
        let mut cs = ChangeSet::new();
        cs.set("name", json!("alice"), json!("bob"), 1000);
        let result = apply_changes(&mut doc, &cs);
        assert!(result.is_err());
    }

    #[test]
    fn rollback_changes_restores() {
        let mut doc = json!({"name": "bob", "age": 30});
        let mut cs = ChangeSet::new();
        cs.set("name", json!("alice"), json!("bob"), 1000);
        rollback_changes(&mut doc, &cs).unwrap();
        assert_eq!(doc["name"], json!("alice"));
    }

    #[test]
    fn has_conflicts_detection() {
        let mut a = ChangeSet::new();
        a.set("x", json!(1), json!(2), 1000);
        let mut b = ChangeSet::new();
        b.set("x", json!(1), json!(3), 1000);
        assert!(has_conflicts(&a, &b));
    }

    #[test]
    fn find_conflicts_fields() {
        let mut a = ChangeSet::new();
        a.set("x", json!(1), json!(2), 1000);
        a.set("y", json!(1), json!(2), 1000);
        let mut b = ChangeSet::new();
        b.set("x", json!(1), json!(3), 1000);
        let conflicts = find_conflicts(&a, &b);
        assert_eq!(conflicts, vec!["x"]);
    }

    #[test]
    fn dirty_tracker_basic() {
        let doc = json!({"name": "alice", "age": 30});
        let mut tracker = DirtyTracker::from_value(&doc);
        assert!(!tracker.is_dirty());

        tracker.set_field("name", json!("bob"));
        assert!(tracker.is_dirty());
        assert!(tracker.is_field_dirty("name"));
        assert!(!tracker.is_field_dirty("age"));
    }

    #[test]
    fn dirty_tracker_commit() {
        let doc = json!({"x": 1});
        let mut tracker = DirtyTracker::from_value(&doc);
        tracker.set_field("x", json!(2));
        assert!(tracker.is_dirty());
        tracker.commit();
        assert!(!tracker.is_dirty());
    }

    #[test]
    fn dirty_tracker_revert() {
        let doc = json!({"x": 1});
        let mut tracker = DirtyTracker::from_value(&doc);
        tracker.set_field("x", json!(2));
        tracker.revert();
        assert!(!tracker.is_dirty());
    }

    #[test]
    fn dirty_tracker_to_change_set() {
        let doc = json!({"name": "alice"});
        let mut tracker = DirtyTracker::from_value(&doc);
        tracker.set_field("name", json!("bob"));
        tracker.set_field("email", json!("b@c.com"));
        let cs = tracker.to_change_set(1000);
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn diff_objects_basic() {
        let old = json!({"a": 1, "b": 2});
        let new = json!({"a": 1, "b": 3, "c": 4});
        let cs = diff_objects(&old, &new, 1000);
        assert!(cs.has_change("b"));
        assert!(cs.has_change("c"));
        assert!(!cs.has_change("a"));
    }

    #[test]
    fn change_notifier() {
        use std::sync::{Arc, Mutex};
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();
        let mut notifier = ChangeNotifier::new();
        notifier.on_change(Box::new(move |change| {
            log_clone.lock().unwrap().push(change.field.clone());
            Ok(())
        }));
        notifier
            .record("name", Some(json!("a")), Some(json!("b")), 1000)
            .unwrap();
        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(notifier.callback_count(), 1);
    }

    #[test]
    fn field_change_display() {
        let change = FieldChange {
            field: "name".into(),
            old_value: Some(json!("alice")),
            new_value: Some(json!("bob")),
            timestamp_ms: 1000,
        };
        let s = change.to_string();
        assert!(s.contains("name"));
    }

    #[test]
    fn error_display() {
        let e = ChangeError::FieldNotFound("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn for_object_constructor() {
        let cs = ChangeSet::for_object("user-123");
        assert_eq!(cs.object_id, Some("user-123".into()));
        assert!(cs.is_empty());
    }

    #[test]
    fn clear_change_set() {
        let mut cs = ChangeSet::new();
        cs.set("x", json!(1), json!(2), 1000);
        assert!(!cs.is_empty());
        cs.clear();
        assert!(cs.is_empty());
    }
}
