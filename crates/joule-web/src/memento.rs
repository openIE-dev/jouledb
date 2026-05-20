//! Memento pattern — state capture, undo/redo, labelled history.
//!
//! The `Originator` captures its state into `Memento` snapshots.
//! A `Caretaker` stores mementos in a bounded history with undo/redo
//! support, labels, and diff between snapshots.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Memento ────────────────────────────────────────────────────────

/// An opaque snapshot of an originator's state.
#[derive(Debug, Clone)]
pub struct Memento {
    id: u64,
    label: String,
    state: HashMap<String, Value>,
    timestamp: u64,
}

impl Memento {
    /// Unique ID of this memento.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Timestamp (monotonic sequence from the originator).
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Read-only access to the captured state.
    pub fn state(&self) -> &HashMap<String, Value> {
        &self.state
    }

    /// Number of keys in the captured state.
    pub fn field_count(&self) -> usize {
        self.state.len()
    }
}

// ── Diff ───────────────────────────────────────────────────────────

/// A single field-level change between two mementos.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldChange {
    Added { key: String, value: Value },
    Removed { key: String, value: Value },
    Modified { key: String, old: Value, new: Value },
}

/// Diff between two mementos.
#[derive(Debug, Clone)]
pub struct MementoDiff {
    pub from_id: u64,
    pub to_id: u64,
    pub changes: Vec<FieldChange>,
}

impl MementoDiff {
    /// Number of changed fields.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Whether the two mementos are identical.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

/// Compute a diff between two mementos.
pub fn diff_mementos(a: &Memento, b: &Memento) -> MementoDiff {
    let mut changes = Vec::new();

    // Collect all keys from both sides (sorted for determinism).
    let mut all_keys: Vec<String> = a
        .state
        .keys()
        .chain(b.state.keys())
        .cloned()
        .collect();
    all_keys.sort();
    all_keys.dedup();

    for key in all_keys {
        match (a.state.get(&key), b.state.get(&key)) {
            (Some(old), Some(new)) => {
                if old != new {
                    changes.push(FieldChange::Modified {
                        key,
                        old: old.clone(),
                        new: new.clone(),
                    });
                }
            }
            (Some(old), None) => {
                changes.push(FieldChange::Removed {
                    key,
                    value: old.clone(),
                });
            }
            (None, Some(new)) => {
                changes.push(FieldChange::Added {
                    key,
                    value: new.clone(),
                });
            }
            (None, None) => {}
        }
    }

    MementoDiff {
        from_id: a.id,
        to_id: b.id,
        changes,
    }
}

// ── Originator ─────────────────────────────────────────────────────

/// The object whose state is being captured and restored.
pub struct Originator {
    state: HashMap<String, Value>,
    next_id: u64,
}

impl Originator {
    /// Create with empty state.
    pub fn new() -> Self {
        Self {
            state: HashMap::new(),
            next_id: 0,
        }
    }

    /// Set a field.
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }

    /// Get a field.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Remove a field.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.state.remove(key)
    }

    /// Number of fields.
    pub fn field_count(&self) -> usize {
        self.state.len()
    }

    /// Read-only access to the full state.
    pub fn state(&self) -> &HashMap<String, Value> {
        &self.state
    }

    /// Capture the current state as a memento.
    pub fn save(&mut self, label: impl Into<String>) -> Memento {
        self.next_id += 1;
        Memento {
            id: self.next_id,
            label: label.into(),
            state: self.state.clone(),
            timestamp: self.next_id,
        }
    }

    /// Restore state from a memento.
    pub fn restore(&mut self, memento: &Memento) {
        self.state = memento.state.clone();
    }
}

impl Default for Originator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Originator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Originator")
            .field("field_count", &self.state.len())
            .field("next_id", &self.next_id)
            .finish()
    }
}

// ── Caretaker ──────────────────────────────────────────────────────

/// Manages a bounded history of mementos with undo/redo.
pub struct Caretaker {
    history: Vec<Memento>,
    /// Points to the current position in history.
    /// -1 means "before any memento".
    cursor: i64,
    max_history: usize,
}

impl Caretaker {
    /// Create with a maximum history size.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            cursor: -1,
            max_history: max_history.max(1),
        }
    }

    /// Push a new memento. Truncates any redo history beyond the cursor.
    pub fn push(&mut self, memento: Memento) {
        // Discard redo tail.
        let new_len = if self.cursor >= 0 {
            (self.cursor as usize) + 1
        } else {
            0
        };
        self.history.truncate(new_len);
        self.history.push(memento);
        self.cursor = (self.history.len() as i64) - 1;

        // Enforce max size.
        while self.history.len() > self.max_history {
            self.history.remove(0);
            self.cursor -= 1;
        }
    }

    /// Undo: move cursor back and return the memento at the new position.
    /// Returns `None` if already at the beginning.
    pub fn undo(&mut self) -> Option<&Memento> {
        if self.cursor > 0 {
            self.cursor -= 1;
            Some(&self.history[self.cursor as usize])
        } else {
            None
        }
    }

    /// Redo: move cursor forward and return the memento at the new position.
    /// Returns `None` if already at the end.
    pub fn redo(&mut self) -> Option<&Memento> {
        let max = (self.history.len() as i64) - 1;
        if self.cursor < max {
            self.cursor += 1;
            Some(&self.history[self.cursor as usize])
        } else {
            None
        }
    }

    /// Whether undo is possible.
    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    /// Whether redo is possible.
    pub fn can_redo(&self) -> bool {
        self.cursor < (self.history.len() as i64) - 1
    }

    /// The current memento (at the cursor).
    pub fn current(&self) -> Option<&Memento> {
        if self.cursor >= 0 && (self.cursor as usize) < self.history.len() {
            Some(&self.history[self.cursor as usize])
        } else {
            None
        }
    }

    /// Number of mementos in history.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// All labels in order.
    pub fn labels(&self) -> Vec<String> {
        self.history.iter().map(|m| m.label.clone()).collect()
    }

    /// Get a memento by index.
    pub fn get(&self, index: usize) -> Option<&Memento> {
        self.history.get(index)
    }

    /// Get the cursor position.
    pub fn cursor_position(&self) -> i64 {
        self.cursor
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.cursor = -1;
    }

    /// Diff between the current memento and the previous one.
    pub fn diff_with_previous(&self) -> Option<MementoDiff> {
        if self.cursor <= 0 {
            return None;
        }
        let prev = &self.history[(self.cursor - 1) as usize];
        let curr = &self.history[self.cursor as usize];
        Some(diff_mementos(prev, curr))
    }
}

impl Default for Caretaker {
    fn default() -> Self {
        Self::new(100)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_originator() -> Originator {
        let mut o = Originator::new();
        o.set("name", Value::String("Alice".to_string()));
        o.set("score", Value::Number(100.into()));
        o
    }

    #[test]
    fn originator_set_get() {
        let o = setup_originator();
        assert_eq!(o.get("name"), Some(&Value::String("Alice".to_string())));
        assert_eq!(o.field_count(), 2);
    }

    #[test]
    fn originator_remove() {
        let mut o = setup_originator();
        assert_eq!(o.remove("name"), Some(Value::String("Alice".to_string())));
        assert_eq!(o.field_count(), 1);
    }

    #[test]
    fn save_and_restore() {
        let mut o = setup_originator();
        let memento = o.save("initial");

        o.set("name", Value::String("Bob".to_string()));
        assert_eq!(o.get("name"), Some(&Value::String("Bob".to_string())));

        o.restore(&memento);
        assert_eq!(o.get("name"), Some(&Value::String("Alice".to_string())));
    }

    #[test]
    fn memento_metadata() {
        let mut o = setup_originator();
        let m = o.save("snapshot1");
        assert_eq!(m.id(), 1);
        assert_eq!(m.label(), "snapshot1");
        assert_eq!(m.field_count(), 2);
        assert!(m.timestamp() > 0);
    }

    #[test]
    fn memento_ids_increment() {
        let mut o = Originator::new();
        let m1 = o.save("a");
        let m2 = o.save("b");
        assert!(m2.id() > m1.id());
    }

    #[test]
    fn caretaker_push_and_current() {
        let mut o = setup_originator();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("v1"));
        assert_eq!(ct.len(), 1);
        assert_eq!(ct.current().unwrap().label(), "v1");
    }

    #[test]
    fn undo_redo_cycle() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);

        o.set("x", Value::Number(1.into()));
        ct.push(o.save("step1"));

        o.set("x", Value::Number(2.into()));
        ct.push(o.save("step2"));

        o.set("x", Value::Number(3.into()));
        ct.push(o.save("step3"));

        // Undo to step2.
        assert!(ct.can_undo());
        let m = ct.undo().unwrap();
        o.restore(m);
        assert_eq!(o.get("x"), Some(&Value::Number(2.into())));

        // Redo to step3.
        assert!(ct.can_redo());
        let m = ct.redo().unwrap();
        o.restore(m);
        assert_eq!(o.get("x"), Some(&Value::Number(3.into())));
    }

    #[test]
    fn undo_at_beginning() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("only"));
        assert!(!ct.can_undo());
        assert!(ct.undo().is_none());
    }

    #[test]
    fn redo_at_end() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("only"));
        assert!(!ct.can_redo());
        assert!(ct.redo().is_none());
    }

    #[test]
    fn push_after_undo_truncates_redo() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);

        o.set("v", Value::Number(1.into()));
        ct.push(o.save("s1"));

        o.set("v", Value::Number(2.into()));
        ct.push(o.save("s2"));

        o.set("v", Value::Number(3.into()));
        ct.push(o.save("s3"));

        // Undo to s2.
        ct.undo();

        // Push new state — s3 should be gone.
        o.set("v", Value::Number(99.into()));
        ct.push(o.save("s4"));

        assert_eq!(ct.len(), 3); // s1, s2, s4
        assert_eq!(ct.labels(), vec!["s1", "s2", "s4"]);
        assert!(!ct.can_redo());
    }

    #[test]
    fn bounded_history() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(3);

        for i in 0..5 {
            o.set("i", Value::Number(i.into()));
            ct.push(o.save(format!("step{i}")));
        }

        assert_eq!(ct.len(), 3);
        // Oldest entries trimmed.
        let labels = ct.labels();
        assert_eq!(labels, vec!["step2", "step3", "step4"]);
    }

    #[test]
    fn caretaker_labels() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("alpha"));
        ct.push(o.save("beta"));
        ct.push(o.save("gamma"));
        assert_eq!(ct.labels(), vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn caretaker_get_by_index() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("first"));
        ct.push(o.save("second"));
        assert_eq!(ct.get(0).unwrap().label(), "first");
        assert_eq!(ct.get(1).unwrap().label(), "second");
        assert!(ct.get(5).is_none());
    }

    #[test]
    fn caretaker_clear() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("x"));
        ct.clear();
        assert!(ct.is_empty());
        assert!(ct.current().is_none());
    }

    #[test]
    fn diff_no_changes() {
        let mut o = Originator::new();
        o.set("k", Value::Number(1.into()));
        let m1 = o.save("a");
        let m2 = o.save("b");
        let d = diff_mementos(&m1, &m2);
        assert!(d.is_empty());
    }

    #[test]
    fn diff_modified() {
        let mut o = Originator::new();
        o.set("k", Value::Number(1.into()));
        let m1 = o.save("a");
        o.set("k", Value::Number(2.into()));
        let m2 = o.save("b");
        let d = diff_mementos(&m1, &m2);
        assert_eq!(d.change_count(), 1);
        match &d.changes[0] {
            FieldChange::Modified { key, old, new } => {
                assert_eq!(key, "k");
                assert_eq!(old, &Value::Number(1.into()));
                assert_eq!(new, &Value::Number(2.into()));
            }
            _ => panic!("expected Modified"),
        }
    }

    #[test]
    fn diff_added_and_removed() {
        let mut o = Originator::new();
        o.set("a", Value::Number(1.into()));
        let m1 = o.save("s1");
        o.remove("a");
        o.set("b", Value::Number(2.into()));
        let m2 = o.save("s2");

        let d = diff_mementos(&m1, &m2);
        assert_eq!(d.change_count(), 2);
        assert!(d.changes.iter().any(|c| matches!(c, FieldChange::Removed { key, .. } if key == "a")));
        assert!(d.changes.iter().any(|c| matches!(c, FieldChange::Added { key, .. } if key == "b")));
    }

    #[test]
    fn caretaker_diff_with_previous() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);

        o.set("x", Value::Number(1.into()));
        ct.push(o.save("s1"));

        o.set("x", Value::Number(2.into()));
        ct.push(o.save("s2"));

        let diff = ct.diff_with_previous().unwrap();
        assert_eq!(diff.change_count(), 1);
    }

    #[test]
    fn caretaker_diff_at_start() {
        let mut o = Originator::new();
        let mut ct = Caretaker::new(10);
        ct.push(o.save("only"));
        assert!(ct.diff_with_previous().is_none());
    }

    #[test]
    fn empty_caretaker() {
        let ct = Caretaker::new(10);
        assert!(ct.is_empty());
        assert!(ct.current().is_none());
        assert_eq!(ct.cursor_position(), -1);
    }
}
