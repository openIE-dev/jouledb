//! State debugging tools with action recording, time travel, filtering,
//! diffing, and JSON import/export.

use serde::{Deserialize, Serialize};

// ── ActionRecord ──

/// A recorded action with before/after state snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub action_type: String,
    pub payload_json: Option<String>,
    pub timestamp_ms: u64,
    pub state_before_json: String,
    pub state_after_json: String,
}

impl ActionRecord {
    pub fn new(
        action_type: impl Into<String>,
        payload_json: Option<String>,
        timestamp_ms: u64,
        state_before_json: impl Into<String>,
        state_after_json: impl Into<String>,
    ) -> Self {
        Self {
            action_type: action_type.into(),
            payload_json,
            timestamp_ms,
            state_before_json: state_before_json.into(),
            state_after_json: state_after_json.into(),
        }
    }
}

// ── StateDiff ──

/// A diff entry between two JSON state strings.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffEntry {
    pub path: String,
    pub old_value: String,
    pub new_value: String,
}

// ── DevToolsStore ──

/// DevTools store that records actions and supports time-travel debugging.
#[derive(Clone, Serialize, Deserialize)]
pub struct DevToolsStore {
    history: Vec<ActionRecord>,
    max_history: usize,
    current_index: Option<usize>,
}

impl Default for DevToolsStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DevToolsStore {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 1000,
            current_index: None,
        }
    }

    /// Set the maximum number of records to keep.
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Record an action with before/after state.
    pub fn record(&mut self, record: ActionRecord) {
        self.history.push(record);
        self.current_index = Some(self.history.len() - 1);

        // Prune if over limit
        if self.history.len() > self.max_history {
            let excess = self.history.len() - self.max_history;
            self.history.drain(0..excess);
            self.current_index = Some(self.history.len() - 1);
        }
    }

    /// Number of recorded actions.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get a specific record by index.
    pub fn get(&self, index: usize) -> Option<&ActionRecord> {
        self.history.get(index)
    }

    /// Get all records.
    pub fn all(&self) -> &[ActionRecord] {
        &self.history
    }

    /// Current time-travel index.
    pub fn current_index(&self) -> Option<usize> {
        self.current_index
    }

    // ── Time Travel ──

    /// Jump to a specific index in history. Returns the state_after_json at
    /// that point, or None if out of bounds.
    pub fn jump_to(&mut self, index: usize) -> Option<&str> {
        if index < self.history.len() {
            self.current_index = Some(index);
            Some(&self.history[index].state_after_json)
        } else {
            None
        }
    }

    /// Replay from the start up to `count` actions. Returns the final
    /// state_after_json or the initial state_before_json if count is 0.
    pub fn replay(&mut self, count: usize) -> Option<&str> {
        if self.history.is_empty() {
            return None;
        }
        if count == 0 {
            self.current_index = None;
            return Some(&self.history[0].state_before_json);
        }
        let idx = count.min(self.history.len()) - 1;
        self.jump_to(idx)
    }

    /// Get the state_before_json of the first record (initial state).
    pub fn initial_state(&self) -> Option<&str> {
        self.history.first().map(|r| r.state_before_json.as_str())
    }

    /// Get the state_after_json of the last record (latest state).
    pub fn latest_state(&self) -> Option<&str> {
        self.history.last().map(|r| r.state_after_json.as_str())
    }

    // ── Filtering ──

    /// Filter records by action type.
    pub fn filter_by_type(&self, action_type: &str) -> Vec<&ActionRecord> {
        self.history
            .iter()
            .filter(|r| r.action_type == action_type)
            .collect()
    }

    /// Filter records by action type prefix.
    pub fn filter_by_prefix(&self, prefix: &str) -> Vec<&ActionRecord> {
        self.history
            .iter()
            .filter(|r| r.action_type.starts_with(prefix))
            .collect()
    }

    // ── Export/Import ──

    /// Export the entire history as a JSON string.
    pub fn export_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.history).map_err(|e| e.to_string())
    }

    /// Import history from a JSON string, replacing current history.
    pub fn import_json(&mut self, json: &str) -> Result<usize, String> {
        let records: Vec<ActionRecord> =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        let count = records.len();
        self.history = records;
        self.current_index = if self.history.is_empty() {
            None
        } else {
            Some(self.history.len() - 1)
        };
        // Prune if needed
        if self.history.len() > self.max_history {
            let excess = self.history.len() - self.max_history;
            self.history.drain(0..excess);
        }
        Ok(count)
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.current_index = None;
    }

    // ── Diff ──

    /// Compute a simple diff between two JSON state strings.
    /// Compares top-level keys of JSON objects.
    pub fn diff_states(before: &str, after: &str) -> Vec<DiffEntry> {
        let mut diffs = Vec::new();

        let before_map: Result<serde_json::Map<String, serde_json::Value>, _> =
            serde_json::from_str(before);
        let after_map: Result<serde_json::Map<String, serde_json::Value>, _> =
            serde_json::from_str(after);

        match (before_map, after_map) {
            (Ok(b), Ok(a)) => {
                // Check changed and removed keys
                for (key, bval) in &b {
                    match a.get(key) {
                        Some(aval) if aval != bval => {
                            diffs.push(DiffEntry {
                                path: key.clone(),
                                old_value: bval.to_string(),
                                new_value: aval.to_string(),
                            });
                        }
                        None => {
                            diffs.push(DiffEntry {
                                path: key.clone(),
                                old_value: bval.to_string(),
                                new_value: "null".to_string(),
                            });
                        }
                        _ => {}
                    }
                }
                // Check added keys
                for (key, aval) in &a {
                    if !b.contains_key(key) {
                        diffs.push(DiffEntry {
                            path: key.clone(),
                            old_value: "null".to_string(),
                            new_value: aval.to_string(),
                        });
                    }
                }
            }
            _ => {
                // Non-object JSON — compare as raw strings
                if before != after {
                    diffs.push(DiffEntry {
                        path: ".".to_string(),
                        old_value: before.to_string(),
                        new_value: after.to_string(),
                    });
                }
            }
        }

        diffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(action: &str, before: &str, after: &str, ts: u64) -> ActionRecord {
        ActionRecord::new(action, None, ts, before, after)
    }

    #[test]
    fn record_and_retrieve() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("INC", r#"{"count":0}"#, r#"{"count":1}"#, 100));
        assert_eq!(dt.len(), 1);
        assert_eq!(dt.get(0).unwrap().action_type, "INC");
    }

    #[test]
    fn time_travel_jump() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("A", r#"{"v":0}"#, r#"{"v":1}"#, 1));
        dt.record(make_record("B", r#"{"v":1}"#, r#"{"v":2}"#, 2));
        dt.record(make_record("C", r#"{"v":2}"#, r#"{"v":3}"#, 3));

        let state = dt.jump_to(1).unwrap();
        assert_eq!(state, r#"{"v":2}"#);
        assert_eq!(dt.current_index(), Some(1));

        assert!(dt.jump_to(100).is_none());
    }

    #[test]
    fn replay_from_start() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("A", r#"{"v":0}"#, r#"{"v":1}"#, 1));
        dt.record(make_record("B", r#"{"v":1}"#, r#"{"v":2}"#, 2));

        // Replay 0 actions → initial state
        assert_eq!(dt.replay(0), Some(r#"{"v":0}"#));

        // Replay 1 action → after first
        assert_eq!(dt.replay(1), Some(r#"{"v":1}"#));

        // Replay all
        assert_eq!(dt.replay(2), Some(r#"{"v":2}"#));

        // Replay beyond length
        assert_eq!(dt.replay(100), Some(r#"{"v":2}"#));
    }

    #[test]
    fn filter_by_type() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("INC", "{}", "{}", 1));
        dt.record(make_record("DEC", "{}", "{}", 2));
        dt.record(make_record("INC", "{}", "{}", 3));

        let filtered = dt.filter_by_type("INC");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_by_prefix() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("user/login", "{}", "{}", 1));
        dt.record(make_record("user/logout", "{}", "{}", 2));
        dt.record(make_record("cart/add", "{}", "{}", 3));

        let filtered = dt.filter_by_prefix("user/");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn export_import_json() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("A", r#"{"x":1}"#, r#"{"x":2}"#, 1));
        dt.record(make_record("B", r#"{"x":2}"#, r#"{"x":3}"#, 2));

        let json = dt.export_json().unwrap();

        let mut dt2 = DevToolsStore::new();
        let count = dt2.import_json(&json).unwrap();
        assert_eq!(count, 2);
        assert_eq!(dt2.len(), 2);
        assert_eq!(dt2.get(0).unwrap().action_type, "A");
    }

    #[test]
    fn max_history_pruning() {
        let mut dt = DevToolsStore::new().with_max_history(3);
        for i in 0..5 {
            dt.record(make_record(&format!("A{i}"), "{}", "{}", i as u64));
        }
        assert_eq!(dt.len(), 3);
        // Oldest two should be pruned
        assert_eq!(dt.get(0).unwrap().action_type, "A2");
    }

    #[test]
    fn diff_states_changed_key() {
        let before = r#"{"count":1,"name":"a"}"#;
        let after = r#"{"count":2,"name":"a"}"#;
        let diffs = DevToolsStore::diff_states(before, after);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].path, "count");
        assert_eq!(diffs[0].old_value, "1");
        assert_eq!(diffs[0].new_value, "2");
    }

    #[test]
    fn diff_states_added_key() {
        let before = r#"{"a":1}"#;
        let after = r#"{"a":1,"b":2}"#;
        let diffs = DevToolsStore::diff_states(before, after);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].path, "b");
        assert_eq!(diffs[0].old_value, "null");
    }

    #[test]
    fn diff_states_removed_key() {
        let before = r#"{"a":1,"b":2}"#;
        let after = r#"{"a":1}"#;
        let diffs = DevToolsStore::diff_states(before, after);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].path, "b");
        assert_eq!(diffs[0].new_value, "null");
    }

    #[test]
    fn diff_states_no_change() {
        let state = r#"{"x":1}"#;
        let diffs = DevToolsStore::diff_states(state, state);
        assert!(diffs.is_empty());
    }

    #[test]
    fn initial_and_latest_state() {
        let mut dt = DevToolsStore::new();
        assert!(dt.initial_state().is_none());
        assert!(dt.latest_state().is_none());

        dt.record(make_record("A", r#"{"v":0}"#, r#"{"v":1}"#, 1));
        dt.record(make_record("B", r#"{"v":1}"#, r#"{"v":2}"#, 2));

        assert_eq!(dt.initial_state(), Some(r#"{"v":0}"#));
        assert_eq!(dt.latest_state(), Some(r#"{"v":2}"#));
    }

    #[test]
    fn clear_history() {
        let mut dt = DevToolsStore::new();
        dt.record(make_record("A", "{}", "{}", 1));
        dt.clear();
        assert!(dt.is_empty());
        assert_eq!(dt.current_index(), None);
    }

    #[test]
    fn record_with_payload() {
        let mut dt = DevToolsStore::new();
        let rec = ActionRecord::new(
            "ADD",
            Some(r#"{"amount":5}"#.to_string()),
            100,
            r#"{"total":0}"#,
            r#"{"total":5}"#,
        );
        dt.record(rec);
        assert_eq!(
            dt.get(0).unwrap().payload_json.as_deref(),
            Some(r#"{"amount":5}"#)
        );
    }
}
