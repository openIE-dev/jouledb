//! Deep configuration merging for nested JSON objects with pluggable
//! array merge strategies, conflict resolution policies, merge path
//! tracking, and diff computation.

use serde_json::Value;
use std::collections::HashMap;

// ── Types ───────────────────────────────────────────────────────

/// How to merge arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayStrategy {
    /// The source array completely replaces the target.
    Replace,
    /// Append source elements to the target array.
    Append,
    /// Merge element-by-element by index position.
    MergeByIndex,
}

/// How to resolve value conflicts on the same key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictPolicy {
    /// Last (source) value wins — the default.
    LastWins,
    /// First (target) value wins.
    FirstWins,
    /// Return an error on any conflict.
    Error,
}

/// A record of a single merge operation at a path.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeEntry {
    pub path: String,
    pub action: MergeAction,
}

/// What happened at a merge point.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeAction {
    /// A new key was added.
    Added,
    /// An existing value was replaced.
    Replaced,
    /// An array was appended to.
    Appended,
    /// Elements were merged by index.
    MergedByIndex,
    /// Kept the original value (FirstWins).
    Kept,
}

/// A diff entry between two configs.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffEntry {
    pub path: String,
    pub kind: DiffKind,
    pub left: Option<Value>,
    pub right: Option<Value>,
}

/// The type of difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    /// Present only in the left config.
    OnlyLeft,
    /// Present only in the right config.
    OnlyRight,
    /// Present in both but different values.
    Changed,
}

/// Error during merge.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeError {
    pub path: String,
    pub message: String,
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "merge error at '{}': {}", self.path, self.message)
    }
}

// ── Merge Options ───────────────────────────────────────────────

/// Configuration for the merge operation.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    pub array_strategy: ArrayStrategy,
    pub conflict_policy: ConflictPolicy,
    /// Per-path overrides for array strategy.
    pub path_array_overrides: HashMap<String, ArrayStrategy>,
}

impl MergeOptions {
    pub fn new() -> Self {
        Self {
            array_strategy: ArrayStrategy::Replace,
            conflict_policy: ConflictPolicy::LastWins,
            path_array_overrides: HashMap::new(),
        }
    }

    pub fn with_array_strategy(mut self, strategy: ArrayStrategy) -> Self {
        self.array_strategy = strategy;
        self
    }

    pub fn with_conflict_policy(mut self, policy: ConflictPolicy) -> Self {
        self.conflict_policy = policy;
        self
    }

    pub fn with_path_override(mut self, path: impl Into<String>, strategy: ArrayStrategy) -> Self {
        self.path_array_overrides.insert(path.into(), strategy);
        self
    }

    fn array_strategy_for(&self, path: &str) -> ArrayStrategy {
        self.path_array_overrides
            .get(path)
            .copied()
            .unwrap_or(self.array_strategy)
    }
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ── Merge ───────────────────────────────────────────────────────

/// Result of a merge: the merged value plus a log of what happened.
#[derive(Debug, Clone)]
pub struct MergeResult {
    pub value: Value,
    pub log: Vec<MergeEntry>,
}

/// Deep-merge `source` into `target`, respecting `options`.
pub fn merge(
    target: &Value,
    source: &Value,
    options: &MergeOptions,
) -> Result<MergeResult, MergeError> {
    let mut log = Vec::new();
    let value = merge_inner(target, source, options, "$", &mut log)?;
    Ok(MergeResult { value, log })
}

fn merge_inner(
    target: &Value,
    source: &Value,
    options: &MergeOptions,
    path: &str,
    log: &mut Vec<MergeEntry>,
) -> Result<Value, MergeError> {
    match (target, source) {
        // Both objects → recurse.
        (Value::Object(tgt_map), Value::Object(src_map)) => {
            let mut result = tgt_map.clone();
            for (key, src_val) in src_map {
                let child_path = if path == "$" {
                    format!("$.{}", key)
                } else {
                    format!("{}.{}", path, key)
                };
                if let Some(tgt_val) = tgt_map.get(key) {
                    let merged = merge_inner(tgt_val, src_val, options, &child_path, log)?;
                    result.insert(key.clone(), merged);
                } else {
                    result.insert(key.clone(), src_val.clone());
                    log.push(MergeEntry { path: child_path, action: MergeAction::Added });
                }
            }
            Ok(Value::Object(result))
        }
        // Both arrays → strategy-dependent.
        (Value::Array(tgt_arr), Value::Array(src_arr)) => {
            let strategy = options.array_strategy_for(path);
            match strategy {
                ArrayStrategy::Replace => {
                    log.push(MergeEntry { path: path.to_string(), action: MergeAction::Replaced });
                    Ok(source.clone())
                }
                ArrayStrategy::Append => {
                    let mut combined = tgt_arr.clone();
                    combined.extend(src_arr.iter().cloned());
                    log.push(MergeEntry { path: path.to_string(), action: MergeAction::Appended });
                    Ok(Value::Array(combined))
                }
                ArrayStrategy::MergeByIndex => {
                    let max_len = tgt_arr.len().max(src_arr.len());
                    let mut merged = Vec::with_capacity(max_len);
                    for i in 0..max_len {
                        let child_path = format!("{}[{}]", path, i);
                        match (tgt_arr.get(i), src_arr.get(i)) {
                            (Some(t), Some(s)) => {
                                let m = merge_inner(t, s, options, &child_path, log)?;
                                merged.push(m);
                            }
                            (Some(t), None) => merged.push(t.clone()),
                            (None, Some(s)) => {
                                merged.push(s.clone());
                                log.push(MergeEntry { path: child_path, action: MergeAction::Added });
                            }
                            (None, None) => break,
                        }
                    }
                    log.push(MergeEntry { path: path.to_string(), action: MergeAction::MergedByIndex });
                    Ok(Value::Array(merged))
                }
            }
        }
        // Scalar conflict.
        _ => {
            if target == source {
                Ok(target.clone())
            } else {
                match options.conflict_policy {
                    ConflictPolicy::LastWins => {
                        log.push(MergeEntry { path: path.to_string(), action: MergeAction::Replaced });
                        Ok(source.clone())
                    }
                    ConflictPolicy::FirstWins => {
                        log.push(MergeEntry { path: path.to_string(), action: MergeAction::Kept });
                        Ok(target.clone())
                    }
                    ConflictPolicy::Error => {
                        Err(MergeError {
                            path: path.to_string(),
                            message: format!(
                                "conflict: target={}, source={}",
                                target, source
                            ),
                        })
                    }
                }
            }
        }
    }
}

// ── Diff ────────────────────────────────────────────────────────

/// Compute the diff between two JSON values.
pub fn diff(left: &Value, right: &Value) -> Vec<DiffEntry> {
    let mut entries = Vec::new();
    diff_inner(left, right, "$", &mut entries);
    entries
}

fn diff_inner(left: &Value, right: &Value, path: &str, out: &mut Vec<DiffEntry>) {
    match (left, right) {
        (Value::Object(lm), Value::Object(rm)) => {
            // Keys only in left.
            for (k, lv) in lm {
                let child = if path == "$" {
                    format!("$.{}", k)
                } else {
                    format!("{}.{}", path, k)
                };
                if let Some(rv) = rm.get(k) {
                    diff_inner(lv, rv, &child, out);
                } else {
                    out.push(DiffEntry {
                        path: child,
                        kind: DiffKind::OnlyLeft,
                        left: Some(lv.clone()),
                        right: None,
                    });
                }
            }
            // Keys only in right.
            for (k, rv) in rm {
                if !lm.contains_key(k) {
                    let child = if path == "$" {
                        format!("$.{}", k)
                    } else {
                        format!("{}.{}", path, k)
                    };
                    out.push(DiffEntry {
                        path: child,
                        kind: DiffKind::OnlyRight,
                        left: None,
                        right: Some(rv.clone()),
                    });
                }
            }
        }
        (Value::Array(la), Value::Array(ra)) => {
            let max_len = la.len().max(ra.len());
            for i in 0..max_len {
                let child = format!("{}[{}]", path, i);
                match (la.get(i), ra.get(i)) {
                    (Some(l), Some(r)) => diff_inner(l, r, &child, out),
                    (Some(l), None) => {
                        out.push(DiffEntry {
                            path: child,
                            kind: DiffKind::OnlyLeft,
                            left: Some(l.clone()),
                            right: None,
                        });
                    }
                    (None, Some(r)) => {
                        out.push(DiffEntry {
                            path: child,
                            kind: DiffKind::OnlyRight,
                            left: None,
                            right: Some(r.clone()),
                        });
                    }
                    (None, None) => break,
                }
            }
        }
        _ => {
            if left != right {
                out.push(DiffEntry {
                    path: path.to_string(),
                    kind: DiffKind::Changed,
                    left: Some(left.clone()),
                    right: Some(right.clone()),
                });
            }
        }
    }
}

// ── Multi-merge ─────────────────────────────────────────────────

/// Merge multiple configs in order (first is base, each subsequent overlays).
pub fn merge_many(
    configs: &[Value],
    options: &MergeOptions,
) -> Result<MergeResult, MergeError> {
    if configs.is_empty() {
        return Ok(MergeResult { value: Value::Null, log: Vec::new() });
    }
    let mut result = configs[0].clone();
    let mut all_log = Vec::new();
    for source in &configs[1..] {
        let mr = merge(&result, source, options)?;
        result = mr.value;
        all_log.extend(mr.log);
    }
    Ok(MergeResult { value: result, log: all_log })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merge_flat_objects() {
        let a = json!({"x": 1});
        let b = json!({"y": 2});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert_eq!(r.value, json!({"x": 1, "y": 2}));
    }

    #[test]
    fn merge_nested_objects() {
        let a = json!({"server": {"host": "localhost", "port": 3000}});
        let b = json!({"server": {"port": 8080}});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert_eq!(r.value, json!({"server": {"host": "localhost", "port": 8080}}));
    }

    #[test]
    fn merge_last_wins_default() {
        let a = json!({"x": 1});
        let b = json!({"x": 2});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert_eq!(r.value, json!({"x": 2}));
    }

    #[test]
    fn merge_first_wins() {
        let opts = MergeOptions::new().with_conflict_policy(ConflictPolicy::FirstWins);
        let a = json!({"x": 1});
        let b = json!({"x": 2});
        let r = merge(&a, &b, &opts).unwrap();
        assert_eq!(r.value, json!({"x": 1}));
    }

    #[test]
    fn merge_error_on_conflict() {
        let opts = MergeOptions::new().with_conflict_policy(ConflictPolicy::Error);
        let a = json!({"x": 1});
        let b = json!({"x": 2});
        let r = merge(&a, &b, &opts);
        assert!(r.is_err());
        let err = r.unwrap_err();
        assert!(err.path.contains("x"));
    }

    #[test]
    fn array_replace_strategy() {
        let opts = MergeOptions::new().with_array_strategy(ArrayStrategy::Replace);
        let a = json!({"tags": [1, 2]});
        let b = json!({"tags": [3, 4]});
        let r = merge(&a, &b, &opts).unwrap();
        assert_eq!(r.value, json!({"tags": [3, 4]}));
    }

    #[test]
    fn array_append_strategy() {
        let opts = MergeOptions::new().with_array_strategy(ArrayStrategy::Append);
        let a = json!({"tags": [1, 2]});
        let b = json!({"tags": [3, 4]});
        let r = merge(&a, &b, &opts).unwrap();
        assert_eq!(r.value, json!({"tags": [1, 2, 3, 4]}));
    }

    #[test]
    fn array_merge_by_index() {
        let opts = MergeOptions::new().with_array_strategy(ArrayStrategy::MergeByIndex);
        let a = json!({"items": [{"a": 1}, {"b": 2}]});
        let b = json!({"items": [{"a": 10}, {"c": 3}]});
        let r = merge(&a, &b, &opts).unwrap();
        assert_eq!(r.value["items"][0]["a"], json!(10));
        assert_eq!(r.value["items"][1]["b"], json!(2));
        assert_eq!(r.value["items"][1]["c"], json!(3));
    }

    #[test]
    fn path_override_for_arrays() {
        let opts = MergeOptions::new()
            .with_array_strategy(ArrayStrategy::Replace)
            .with_path_override("$.tags", ArrayStrategy::Append);
        let a = json!({"tags": [1], "ids": [10]});
        let b = json!({"tags": [2], "ids": [20]});
        let r = merge(&a, &b, &opts).unwrap();
        assert_eq!(r.value["tags"], json!([1, 2]));
        assert_eq!(r.value["ids"], json!([20])); // Replace (default)
    }

    #[test]
    fn merge_log_tracks_added() {
        let a = json!({});
        let b = json!({"new_key": "value"});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert!(r.log.iter().any(|e| e.action == MergeAction::Added));
    }

    #[test]
    fn merge_log_tracks_replaced() {
        let a = json!({"x": 1});
        let b = json!({"x": 2});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert!(r.log.iter().any(|e| e.action == MergeAction::Replaced));
    }

    #[test]
    fn merge_identical_no_log() {
        let a = json!({"x": 1});
        let b = json!({"x": 1});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert!(r.log.is_empty());
    }

    #[test]
    fn diff_identical() {
        let a = json!({"x": 1});
        assert!(diff(&a, &a).is_empty());
    }

    #[test]
    fn diff_changed_value() {
        let a = json!({"x": 1});
        let b = json!({"x": 2});
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].kind, DiffKind::Changed);
    }

    #[test]
    fn diff_only_left() {
        let a = json!({"x": 1, "y": 2});
        let b = json!({"x": 1});
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].kind, DiffKind::OnlyLeft);
    }

    #[test]
    fn diff_only_right() {
        let a = json!({"x": 1});
        let b = json!({"x": 1, "z": 3});
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].kind, DiffKind::OnlyRight);
    }

    #[test]
    fn diff_nested() {
        let a = json!({"s": {"a": 1}});
        let b = json!({"s": {"a": 2}});
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert!(d[0].path.contains("a"));
    }

    #[test]
    fn diff_array_length_mismatch() {
        let a = json!({"arr": [1, 2, 3]});
        let b = json!({"arr": [1, 2]});
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].kind, DiffKind::OnlyLeft);
    }

    #[test]
    fn merge_many_three_configs() {
        let configs = vec![
            json!({"a": 1}),
            json!({"b": 2}),
            json!({"c": 3}),
        ];
        let r = merge_many(&configs, &MergeOptions::new()).unwrap();
        assert_eq!(r.value, json!({"a": 1, "b": 2, "c": 3}));
    }

    #[test]
    fn merge_many_empty() {
        let r = merge_many(&[], &MergeOptions::new()).unwrap();
        assert_eq!(r.value, Value::Null);
    }

    #[test]
    fn merge_deep_three_levels() {
        let a = json!({"l1": {"l2": {"l3": "a"}}});
        let b = json!({"l1": {"l2": {"l3": "b", "extra": true}}});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert_eq!(r.value["l1"]["l2"]["l3"], json!("b"));
        assert_eq!(r.value["l1"]["l2"]["extra"], json!(true));
    }

    #[test]
    fn merge_source_adds_new_subtree() {
        let a = json!({"x": 1});
        let b = json!({"y": {"z": 2}});
        let r = merge(&a, &b, &MergeOptions::new()).unwrap();
        assert_eq!(r.value["y"]["z"], json!(2));
    }

    #[test]
    fn merge_error_display() {
        let e = MergeError { path: "$.x".into(), message: "conflict".into() };
        let s = format!("{}", e);
        assert!(s.contains("$.x"));
        assert!(s.contains("conflict"));
    }
}
