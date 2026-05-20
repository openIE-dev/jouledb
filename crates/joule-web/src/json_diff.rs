//! JSON structural diff.
//!
//! Deep comparison of JSON values producing a list of operations (add, remove,
//! replace, move). Outputs JSON Patch (RFC 6902) format. Supports applying
//! patches, generating reverse patches, and diff minimization.

use serde_json::Value;
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// Error type for JSON diff/patch operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonDiffError {
    /// A path was not found in the document.
    PathNotFound(String),
    /// An array index was out of bounds.
    IndexOutOfBounds { index: usize, length: usize },
    /// A test operation failed.
    TestFailed { path: String, expected: String, actual: String },
    /// Invalid JSON Pointer syntax.
    InvalidPointer(String),
    /// Generic patch application error.
    PatchFailed(String),
}

impl fmt::Display for JsonDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathNotFound(p) => write!(f, "path not found: {}", p),
            Self::IndexOutOfBounds { index, length } => {
                write!(f, "index {} out of bounds (length {})", index, length)
            }
            Self::TestFailed { path, expected, actual } => {
                write!(f, "test at '{}': expected {}, got {}", path, expected, actual)
            }
            Self::InvalidPointer(s) => write!(f, "invalid pointer: {}", s),
            Self::PatchFailed(s) => write!(f, "patch failed: {}", s),
        }
    }
}

/// A single diff operation.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffOp {
    /// Add a value at a path.
    Add { path: String, value: Value },
    /// Remove a value at a path.
    Remove { path: String, old_value: Value },
    /// Replace a value at a path.
    Replace { path: String, old_value: Value, new_value: Value },
    /// Move a value from one path to another.
    Move { from: String, path: String },
}

/// RFC 6902 JSON Patch operation for serialization.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchOp {
    Add { path: String, value: Value },
    Remove { path: String },
    Replace { path: String, value: Value },
    Move { from: String, path: String },
    Copy { from: String, path: String },
    Test { path: String, value: Value },
}

/// Result of a JSON diff.
#[derive(Debug, Clone)]
pub struct JsonDiffResult {
    pub operations: Vec<DiffOp>,
}

// ── JSON Pointer helpers ───────────────────────────────────────────

fn escape_pointer(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn pointer_join(base: &str, segment: &str) -> String {
    format!("{}/{}", base, escape_pointer(segment))
}

fn pointer_join_index(base: &str, index: usize) -> String {
    format!("{}/{}", base, index)
}

/// Resolve a JSON pointer to a reference in a value.
fn resolve<'a>(doc: &'a Value, pointer: &str) -> Option<&'a Value> {
    if pointer.is_empty() {
        return Some(doc);
    }
    if !pointer.starts_with('/') {
        return None;
    }
    let segments: Vec<&str> = pointer[1..].split('/').collect();
    let mut current = doc;
    for seg in segments {
        let decoded = seg.replace("~1", "/").replace("~0", "~");
        match current {
            Value::Object(map) => {
                current = map.get(&decoded)?;
            }
            Value::Array(arr) => {
                let idx: usize = decoded.parse().ok()?;
                current = arr.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Resolve a JSON pointer to a mutable reference in a value.
fn resolve_mut<'a>(doc: &'a mut Value, pointer: &str) -> Option<&'a mut Value> {
    if pointer.is_empty() {
        return Some(doc);
    }
    if !pointer.starts_with('/') {
        return None;
    }
    let segments: Vec<String> = pointer[1..]
        .split('/')
        .map(|s| s.replace("~1", "/").replace("~0", "~"))
        .collect();
    let mut current = doc;
    for seg in &segments {
        match current {
            Value::Object(map) => {
                current = map.get_mut(seg.as_str())?;
            }
            Value::Array(arr) => {
                let idx: usize = seg.parse().ok()?;
                current = arr.get_mut(idx)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn split_pointer(path: &str) -> Result<(String, String), JsonDiffError> {
    if path.is_empty() {
        return Err(JsonDiffError::InvalidPointer("empty path".into()));
    }
    let segments: Vec<&str> = path[1..].split('/').collect();
    if segments.is_empty() {
        return Err(JsonDiffError::InvalidPointer(path.into()));
    }
    let last = segments.last().unwrap().replace("~1", "/").replace("~0", "~");
    if segments.len() == 1 {
        Ok((String::new(), last))
    } else {
        let parent = format!("/{}", segments[..segments.len() - 1].join("/"));
        Ok((parent, last))
    }
}

// ── Diff engine ────────────────────────────────────────────────────

/// Compute a structural diff between two JSON values.
pub fn diff(old: &Value, new: &Value) -> JsonDiffResult {
    let mut operations = Vec::new();
    diff_recursive(old, new, String::new(), &mut operations);
    JsonDiffResult { operations }
}

fn diff_recursive(old: &Value, new: &Value, path: String, ops: &mut Vec<DiffOp>) {
    if old == new {
        return;
    }

    match (old, new) {
        (Value::Object(old_map), Value::Object(new_map)) => {
            // Removed keys.
            for (key, old_val) in old_map {
                let child_path = pointer_join(&path, key);
                if !new_map.contains_key(key) {
                    ops.push(DiffOp::Remove {
                        path: child_path,
                        old_value: old_val.clone(),
                    });
                }
            }
            // Added or changed keys.
            for (key, new_val) in new_map {
                let child_path = pointer_join(&path, key);
                match old_map.get(key) {
                    Some(old_val) => diff_recursive(old_val, new_val, child_path, ops),
                    None => {
                        ops.push(DiffOp::Add {
                            path: child_path,
                            value: new_val.clone(),
                        });
                    }
                }
            }
        }
        (Value::Array(old_arr), Value::Array(new_arr)) => {
            diff_arrays(old_arr, new_arr, &path, ops);
        }
        _ => {
            ops.push(DiffOp::Replace {
                path,
                old_value: old.clone(),
                new_value: new.clone(),
            });
        }
    }
}

fn diff_arrays(old: &[Value], new: &[Value], path: &str, ops: &mut Vec<DiffOp>) {
    let old_len = old.len();
    let new_len = new.len();
    let common = old_len.min(new_len);

    for i in 0..common {
        let child_path = pointer_join_index(path, i);
        diff_recursive(&old[i], &new[i], child_path, ops);
    }

    // Elements removed from the end.
    if old_len > new_len {
        // Remove from the end backwards to keep indices stable.
        for i in (new_len..old_len).rev() {
            ops.push(DiffOp::Remove {
                path: pointer_join_index(path, i),
                old_value: old[i].clone(),
            });
        }
    }

    // Elements added at the end.
    if new_len > old_len {
        for i in old_len..new_len {
            ops.push(DiffOp::Add {
                path: pointer_join_index(path, i),
                value: new[i].clone(),
            });
        }
    }
}

/// Convert diff operations to RFC 6902 JSON Patch operations.
pub fn to_patch(result: &JsonDiffResult) -> Vec<PatchOp> {
    result
        .operations
        .iter()
        .map(|op| match op {
            DiffOp::Add { path, value } => PatchOp::Add {
                path: path.clone(),
                value: value.clone(),
            },
            DiffOp::Remove { path, .. } => PatchOp::Remove { path: path.clone() },
            DiffOp::Replace { path, new_value, .. } => PatchOp::Replace {
                path: path.clone(),
                value: new_value.clone(),
            },
            DiffOp::Move { from, path } => PatchOp::Move {
                from: from.clone(),
                path: path.clone(),
            },
        })
        .collect()
}

/// Serialize patch operations to a JSON array (RFC 6902 format).
pub fn patch_to_json(ops: &[PatchOp]) -> Value {
    let arr: Vec<Value> = ops
        .iter()
        .map(|op| match op {
            PatchOp::Add { path, value } => {
                serde_json::json!({"op": "add", "path": path, "value": value})
            }
            PatchOp::Remove { path } => {
                serde_json::json!({"op": "remove", "path": path})
            }
            PatchOp::Replace { path, value } => {
                serde_json::json!({"op": "replace", "path": path, "value": value})
            }
            PatchOp::Move { from, path } => {
                serde_json::json!({"op": "move", "from": from, "path": path})
            }
            PatchOp::Copy { from, path } => {
                serde_json::json!({"op": "copy", "from": from, "path": path})
            }
            PatchOp::Test { path, value } => {
                serde_json::json!({"op": "test", "path": path, "value": value})
            }
        })
        .collect();
    Value::Array(arr)
}

// ── Patch application ──────────────────────────────────────────────

/// Apply a JSON Patch to a document.
pub fn apply_patch(doc: &mut Value, ops: &[PatchOp]) -> Result<(), JsonDiffError> {
    let backup = doc.clone();
    for op in ops {
        if let Err(e) = apply_single(doc, op) {
            *doc = backup;
            return Err(e);
        }
    }
    Ok(())
}

fn apply_single(doc: &mut Value, op: &PatchOp) -> Result<(), JsonDiffError> {
    match op {
        PatchOp::Add { path, value } => add_value(doc, path, value.clone()),
        PatchOp::Remove { path } => {
            remove_value(doc, path)?;
            Ok(())
        }
        PatchOp::Replace { path, value } => {
            if path.is_empty() {
                *doc = value.clone();
                return Ok(());
            }
            let target = resolve_mut(doc, path)
                .ok_or_else(|| JsonDiffError::PathNotFound(path.clone()))?;
            *target = value.clone();
            Ok(())
        }
        PatchOp::Move { from, path } => {
            let val = remove_value(doc, from)?;
            add_value(doc, path, val)
        }
        PatchOp::Copy { from, path } => {
            let val = resolve(doc, from)
                .ok_or_else(|| JsonDiffError::PathNotFound(from.clone()))?
                .clone();
            add_value(doc, path, val)
        }
        PatchOp::Test { path, value } => {
            let actual = resolve(doc, path)
                .ok_or_else(|| JsonDiffError::PathNotFound(path.clone()))?;
            if actual != value {
                return Err(JsonDiffError::TestFailed {
                    path: path.clone(),
                    expected: value.to_string(),
                    actual: actual.to_string(),
                });
            }
            Ok(())
        }
    }
}

fn add_value(doc: &mut Value, path: &str, value: Value) -> Result<(), JsonDiffError> {
    if path.is_empty() {
        *doc = value;
        return Ok(());
    }
    let (parent_ptr, key) = split_pointer(path)?;
    let parent = if parent_ptr.is_empty() {
        doc
    } else {
        resolve_mut(doc, &parent_ptr)
            .ok_or_else(|| JsonDiffError::PathNotFound(parent_ptr.clone()))?
    };
    match parent {
        Value::Object(map) => {
            map.insert(key, value);
            Ok(())
        }
        Value::Array(arr) => {
            if key == "-" {
                arr.push(value);
                Ok(())
            } else {
                let idx: usize = key.parse().map_err(|_| {
                    JsonDiffError::InvalidPointer(format!("not an index: {}", key))
                })?;
                if idx > arr.len() {
                    return Err(JsonDiffError::IndexOutOfBounds {
                        index: idx,
                        length: arr.len(),
                    });
                }
                arr.insert(idx, value);
                Ok(())
            }
        }
        _ => Err(JsonDiffError::PathNotFound(path.into())),
    }
}

fn remove_value(doc: &mut Value, path: &str) -> Result<Value, JsonDiffError> {
    let (parent_ptr, key) = split_pointer(path)?;
    let parent = if parent_ptr.is_empty() {
        doc
    } else {
        resolve_mut(doc, &parent_ptr)
            .ok_or_else(|| JsonDiffError::PathNotFound(parent_ptr.clone()))?
    };
    match parent {
        Value::Object(map) => map
            .remove(&key)
            .ok_or_else(|| JsonDiffError::PathNotFound(path.into())),
        Value::Array(arr) => {
            let idx: usize = key.parse().map_err(|_| {
                JsonDiffError::InvalidPointer(format!("not an index: {}", key))
            })?;
            if idx >= arr.len() {
                return Err(JsonDiffError::IndexOutOfBounds {
                    index: idx,
                    length: arr.len(),
                });
            }
            Ok(arr.remove(idx))
        }
        _ => Err(JsonDiffError::PathNotFound(path.into())),
    }
}

// ── Reverse patch ──────────────────────────────────────────────────

/// Generate a reverse patch from a diff result (undoes the diff).
pub fn reverse_patch(result: &JsonDiffResult) -> Vec<PatchOp> {
    let mut reversed = Vec::new();
    for op in result.operations.iter().rev() {
        match op {
            DiffOp::Add { path, .. } => {
                reversed.push(PatchOp::Remove { path: path.clone() });
            }
            DiffOp::Remove { path, old_value } => {
                reversed.push(PatchOp::Add {
                    path: path.clone(),
                    value: old_value.clone(),
                });
            }
            DiffOp::Replace { path, old_value, .. } => {
                reversed.push(PatchOp::Replace {
                    path: path.clone(),
                    value: old_value.clone(),
                });
            }
            DiffOp::Move { from, path } => {
                reversed.push(PatchOp::Move {
                    from: path.clone(),
                    path: from.clone(),
                });
            }
        }
    }
    reversed
}

// ── Diff minimization ──────────────────────────────────────────────

/// Minimize a diff by coalescing related operations.
/// For example, if an entire object is replaced, collapse its child ops
/// into a single replace.
pub fn minimize(result: &JsonDiffResult) -> JsonDiffResult {
    let ops = &result.operations;
    if ops.is_empty() {
        return JsonDiffResult {
            operations: Vec::new(),
        };
    }

    // Group operations by common parent path.
    let mut minimized: Vec<DiffOp> = Vec::new();
    let mut skip_children_of: Option<String> = None;

    for op in ops {
        let op_path = match op {
            DiffOp::Add { path, .. } => path.as_str(),
            DiffOp::Remove { path, .. } => path.as_str(),
            DiffOp::Replace { path, .. } => path.as_str(),
            DiffOp::Move { path, .. } => path.as_str(),
        };

        // If this operation is a child of a path we're already replacing, skip it.
        if let Some(parent) = &skip_children_of {
            if op_path.starts_with(parent.as_str())
                && op_path.len() > parent.len()
                && op_path.as_bytes().get(parent.len()) == Some(&b'/')
            {
                continue;
            } else {
                skip_children_of = None;
            }
        }

        // If this is a replace that covers an entire subtree, we can
        // skip child operations.
        if let DiffOp::Replace { path, .. } = op {
            skip_children_of = Some(path.clone());
        }

        minimized.push(op.clone());
    }

    JsonDiffResult {
        operations: minimized,
    }
}

/// Count the number of operations of each type.
pub fn diff_stats(result: &JsonDiffResult) -> (usize, usize, usize, usize) {
    let mut adds = 0;
    let mut removes = 0;
    let mut replaces = 0;
    let mut moves = 0;
    for op in &result.operations {
        match op {
            DiffOp::Add { .. } => adds += 1,
            DiffOp::Remove { .. } => removes += 1,
            DiffOp::Replace { .. } => replaces += 1,
            DiffOp::Move { .. } => moves += 1,
        }
    }
    (adds, removes, replaces, moves)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn diff_identical() {
        let v = json!({"a": 1, "b": [1, 2]});
        let result = diff(&v, &v);
        assert!(result.operations.is_empty());
    }

    #[test]
    fn diff_add_key() {
        let old = json!({"a": 1});
        let new = json!({"a": 1, "b": 2});
        let result = diff(&old, &new);
        assert_eq!(result.operations.len(), 1);
        assert!(matches!(&result.operations[0], DiffOp::Add { path, .. } if path == "/b"));
    }

    #[test]
    fn diff_remove_key() {
        let old = json!({"a": 1, "b": 2});
        let new = json!({"a": 1});
        let result = diff(&old, &new);
        assert_eq!(result.operations.len(), 1);
        assert!(matches!(&result.operations[0], DiffOp::Remove { path, .. } if path == "/b"));
    }

    #[test]
    fn diff_replace_value() {
        let old = json!({"a": 1});
        let new = json!({"a": 2});
        let result = diff(&old, &new);
        assert_eq!(result.operations.len(), 1);
        assert!(matches!(&result.operations[0], DiffOp::Replace { path, .. } if path == "/a"));
    }

    #[test]
    fn diff_nested_objects() {
        let old = json!({"a": {"b": 1, "c": 2}});
        let new = json!({"a": {"b": 1, "c": 3, "d": 4}});
        let result = diff(&old, &new);
        let (adds, _, replaces, _) = diff_stats(&result);
        assert_eq!(adds, 1);
        assert_eq!(replaces, 1);
    }

    #[test]
    fn diff_arrays() {
        let old = json!([1, 2, 3]);
        let new = json!([1, 4, 3, 5]);
        let result = diff(&old, &new);
        assert!(!result.operations.is_empty());
    }

    #[test]
    fn diff_array_shrink() {
        let old = json!([1, 2, 3]);
        let new = json!([1]);
        let result = diff(&old, &new);
        let (_, removes, _, _) = diff_stats(&result);
        assert_eq!(removes, 2);
    }

    #[test]
    fn diff_type_change() {
        let old = json!({"a": 1});
        let new = json!({"a": "hello"});
        let result = diff(&old, &new);
        assert_eq!(result.operations.len(), 1);
        assert!(matches!(&result.operations[0], DiffOp::Replace { .. }));
    }

    #[test]
    fn to_rfc6902_patch() {
        let old = json!({"a": 1});
        let new = json!({"a": 2, "b": 3});
        let result = diff(&old, &new);
        let patch = to_patch(&result);
        assert!(patch.len() >= 2);
    }

    #[test]
    fn patch_to_json_format() {
        let ops = vec![
            PatchOp::Add {
                path: "/a".into(),
                value: json!(1),
            },
            PatchOp::Remove { path: "/b".into() },
        ];
        let j = patch_to_json(&ops);
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 2);
    }

    #[test]
    fn apply_patch_add() {
        let mut doc = json!({"a": 1});
        apply_patch(
            &mut doc,
            &[PatchOp::Add {
                path: "/b".into(),
                value: json!(2),
            }],
        )
        .unwrap();
        assert_eq!(doc, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn apply_patch_remove() {
        let mut doc = json!({"a": 1, "b": 2});
        apply_patch(&mut doc, &[PatchOp::Remove { path: "/b".into() }]).unwrap();
        assert_eq!(doc, json!({"a": 1}));
    }

    #[test]
    fn apply_patch_replace() {
        let mut doc = json!({"a": 1});
        apply_patch(
            &mut doc,
            &[PatchOp::Replace {
                path: "/a".into(),
                value: json!(99),
            }],
        )
        .unwrap();
        assert_eq!(doc, json!({"a": 99}));
    }

    #[test]
    fn apply_patch_rollback_on_failure() {
        let mut doc = json!({"a": 1});
        let original = doc.clone();
        let result = apply_patch(
            &mut doc,
            &[
                PatchOp::Replace {
                    path: "/a".into(),
                    value: json!(2),
                },
                PatchOp::Remove {
                    path: "/nonexistent".into(),
                },
            ],
        );
        assert!(result.is_err());
        assert_eq!(doc, original);
    }

    #[test]
    fn diff_and_apply_roundtrip() {
        let old = json!({"name": "alice", "age": 30, "tags": ["a", "b"]});
        let new = json!({"name": "alice", "age": 31, "tags": ["a", "c"], "active": true});
        let result = diff(&old, &new);
        let patch = to_patch(&result);
        let mut doc = old.clone();
        apply_patch(&mut doc, &patch).unwrap();
        assert_eq!(doc, new);
    }

    #[test]
    fn reverse_patch_undoes_diff() {
        let old = json!({"a": 1, "b": 2});
        let new = json!({"a": 1, "b": 3, "c": 4});
        let result = diff(&old, &new);
        let patch = to_patch(&result);
        let mut doc = old.clone();
        apply_patch(&mut doc, &patch).unwrap();
        assert_eq!(doc, new);

        let rev = reverse_patch(&result);
        apply_patch(&mut doc, &rev).unwrap();
        assert_eq!(doc, old);
    }

    #[test]
    fn minimize_collapses_children() {
        let result = JsonDiffResult {
            operations: vec![
                DiffOp::Replace {
                    path: "/a".into(),
                    old_value: json!({"x": 1}),
                    new_value: json!({"y": 2}),
                },
                DiffOp::Remove {
                    path: "/a/x".into(),
                    old_value: json!(1),
                },
                DiffOp::Add {
                    path: "/a/y".into(),
                    value: json!(2),
                },
            ],
        };
        let min = minimize(&result);
        assert_eq!(min.operations.len(), 1);
    }

    #[test]
    fn diff_stats_counts() {
        let result = JsonDiffResult {
            operations: vec![
                DiffOp::Add {
                    path: "/a".into(),
                    value: json!(1),
                },
                DiffOp::Remove {
                    path: "/b".into(),
                    old_value: json!(2),
                },
                DiffOp::Replace {
                    path: "/c".into(),
                    old_value: json!(3),
                    new_value: json!(4),
                },
            ],
        };
        let (adds, removes, replaces, moves) = diff_stats(&result);
        assert_eq!(adds, 1);
        assert_eq!(removes, 1);
        assert_eq!(replaces, 1);
        assert_eq!(moves, 0);
    }

    #[test]
    fn error_display() {
        let e = JsonDiffError::PathNotFound("/x".into());
        assert!(e.to_string().contains("/x"));
    }

    #[test]
    fn diff_empty_objects() {
        let old = json!({});
        let new = json!({});
        let result = diff(&old, &new);
        assert!(result.operations.is_empty());
    }

    #[test]
    fn diff_null_values() {
        let old = json!(null);
        let new = json!(42);
        let result = diff(&old, &new);
        assert_eq!(result.operations.len(), 1);
    }

    #[test]
    fn apply_test_op_success() {
        let mut doc = json!({"a": 1});
        let result = apply_patch(
            &mut doc,
            &[PatchOp::Test {
                path: "/a".into(),
                value: json!(1),
            }],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn apply_test_op_failure() {
        let mut doc = json!({"a": 1});
        let result = apply_patch(
            &mut doc,
            &[PatchOp::Test {
                path: "/a".into(),
                value: json!(999),
            }],
        );
        assert!(result.is_err());
    }
}
