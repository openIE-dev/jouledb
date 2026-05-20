// SPDX-License-Identifier: MIT
//! JSON Patch -- RFC 6902 implementation.
//!
//! Operations: add, remove, replace, move, copy, test.
//! Atomic apply (all-or-nothing), diff generation, path parsing with
//! RFC 6901 JSON Pointer escaping.

use serde_json::{Map, Value};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PatchError {
    InvalidPath(String),
    PathNotFound(String),
    TestFailed { path: String, expected: Value, actual: Value },
    IndexOutOfBounds { path: String, index: usize, len: usize },
    InvalidOperation(String),
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(p) => write!(f, "invalid path: {p}"),
            Self::PathNotFound(p) => write!(f, "path not found: {p}"),
            Self::TestFailed { path, expected, actual } => {
                write!(f, "test failed at {path}: expected {expected}, got {actual}")
            }
            Self::IndexOutOfBounds { path, index, len } => {
                write!(f, "index {index} out of bounds (len {len}) at {path}")
            }
            Self::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
        }
    }
}

// ── Patch Operation ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PatchOp {
    Add { path: String, value: Value },
    Remove { path: String },
    Replace { path: String, value: Value },
    Move { from: String, path: String },
    Copy { from: String, path: String },
    Test { path: String, value: Value },
}

impl PatchOp {
    /// Parse a JSON object into a PatchOp.
    pub fn from_value(v: &Value) -> Result<Self, PatchError> {
        let obj = v
            .as_object()
            .ok_or_else(|| PatchError::InvalidOperation("expected object".into()))?;
        let op = obj
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PatchError::InvalidOperation("missing 'op'".into()))?;
        let path = obj
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        match op {
            "add" => {
                let value = obj
                    .get("value")
                    .cloned()
                    .ok_or_else(|| PatchError::InvalidOperation("add: missing 'value'".into()))?;
                Ok(PatchOp::Add { path, value })
            }
            "remove" => Ok(PatchOp::Remove { path }),
            "replace" => {
                let value = obj
                    .get("value")
                    .cloned()
                    .ok_or_else(|| PatchError::InvalidOperation("replace: missing 'value'".into()))?;
                Ok(PatchOp::Replace { path, value })
            }
            "move" => {
                let from = obj
                    .get("from")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| PatchError::InvalidOperation("move: missing 'from'".into()))?
                    .to_string();
                Ok(PatchOp::Move { from, path })
            }
            "copy" => {
                let from = obj
                    .get("from")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| PatchError::InvalidOperation("copy: missing 'from'".into()))?
                    .to_string();
                Ok(PatchOp::Copy { from, path })
            }
            "test" => {
                let value = obj
                    .get("value")
                    .cloned()
                    .ok_or_else(|| PatchError::InvalidOperation("test: missing 'value'".into()))?;
                Ok(PatchOp::Test { path, value })
            }
            other => Err(PatchError::InvalidOperation(format!("unknown op: {other}"))),
        }
    }

    /// Serialize back to JSON.
    pub fn to_value(&self) -> Value {
        match self {
            PatchOp::Add { path, value } => {
                Value::Object(make_op_obj("add", path, Some(value.clone()), None))
            }
            PatchOp::Remove { path } => {
                Value::Object(make_op_obj("remove", path, None, None))
            }
            PatchOp::Replace { path, value } => {
                Value::Object(make_op_obj("replace", path, Some(value.clone()), None))
            }
            PatchOp::Move { from, path } => {
                Value::Object(make_op_obj("move", path, None, Some(from)))
            }
            PatchOp::Copy { from, path } => {
                Value::Object(make_op_obj("copy", path, None, Some(from)))
            }
            PatchOp::Test { path, value } => {
                Value::Object(make_op_obj("test", path, Some(value.clone()), None))
            }
        }
    }
}

fn make_op_obj(op: &str, path: &str, value: Option<Value>, from: Option<&str>) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("op".into(), Value::String(op.into()));
    if let Some(from_str) = from {
        m.insert("from".into(), Value::String(from_str.into()));
    }
    m.insert("path".into(), Value::String(path.into()));
    if let Some(v) = value {
        m.insert("value".into(), v);
    }
    m
}

// ── Path helpers ────────────────────────────────────────────────────────────

fn parse_pointer(path: &str) -> Vec<String> {
    if path.is_empty() {
        return vec![];
    }
    let stripped = path.strip_prefix('/').unwrap_or(path);
    stripped
        .split('/')
        .map(|seg| seg.replace("~1", "/").replace("~0", "~"))
        .collect()
}

fn escape_segment(seg: &str) -> String {
    seg.replace('~', "~0").replace('/', "~1")
}

fn pointer_get<'a>(doc: &'a Value, path: &str) -> Result<&'a Value, PatchError> {
    let segs = parse_pointer(path);
    let mut cur = doc;
    for seg in &segs {
        match cur {
            Value::Object(map) => {
                cur = map
                    .get(seg.as_str())
                    .ok_or_else(|| PatchError::PathNotFound(path.into()))?;
            }
            Value::Array(arr) => {
                let idx: usize = seg
                    .parse()
                    .map_err(|_| PatchError::InvalidPath(path.into()))?;
                cur = arr
                    .get(idx)
                    .ok_or_else(|| PatchError::IndexOutOfBounds {
                        path: path.into(),
                        index: idx,
                        len: arr.len(),
                    })?;
            }
            _ => return Err(PatchError::PathNotFound(path.into())),
        }
    }
    Ok(cur)
}

fn pointer_add(doc: &mut Value, path: &str, value: Value) -> Result<(), PatchError> {
    if path.is_empty() {
        *doc = value;
        return Ok(());
    }
    let segs = parse_pointer(path);
    let (parent_segs, last) = segs.split_at(segs.len() - 1);

    let mut cur = doc;
    for seg in parent_segs {
        cur = descend_mut(cur, seg, path)?;
    }

    let key = &last[0];
    match cur {
        Value::Object(map) => {
            map.insert(key.clone(), value);
            Ok(())
        }
        Value::Array(arr) => {
            if key == "-" {
                arr.push(value);
                Ok(())
            } else {
                let idx: usize = key
                    .parse()
                    .map_err(|_| PatchError::InvalidPath(path.into()))?;
                if idx > arr.len() {
                    return Err(PatchError::IndexOutOfBounds {
                        path: path.into(),
                        index: idx,
                        len: arr.len(),
                    });
                }
                arr.insert(idx, value);
                Ok(())
            }
        }
        _ => Err(PatchError::PathNotFound(path.into())),
    }
}

fn pointer_remove(doc: &mut Value, path: &str) -> Result<Value, PatchError> {
    if path.is_empty() {
        return Err(PatchError::InvalidPath("cannot remove root".into()));
    }
    let segs = parse_pointer(path);
    let (parent_segs, last) = segs.split_at(segs.len() - 1);

    let mut cur = doc;
    for seg in parent_segs {
        cur = descend_mut(cur, seg, path)?;
    }

    let key = &last[0];
    match cur {
        Value::Object(map) => map
            .remove(key.as_str())
            .ok_or_else(|| PatchError::PathNotFound(path.into())),
        Value::Array(arr) => {
            let idx: usize = key
                .parse()
                .map_err(|_| PatchError::InvalidPath(path.into()))?;
            if idx >= arr.len() {
                return Err(PatchError::IndexOutOfBounds {
                    path: path.into(),
                    index: idx,
                    len: arr.len(),
                });
            }
            Ok(arr.remove(idx))
        }
        _ => Err(PatchError::PathNotFound(path.into())),
    }
}

fn descend_mut<'a>(
    val: &'a mut Value,
    seg: &str,
    full_path: &str,
) -> Result<&'a mut Value, PatchError> {
    match val {
        Value::Object(map) => map
            .get_mut(seg)
            .ok_or_else(|| PatchError::PathNotFound(full_path.into())),
        Value::Array(arr) => {
            let idx: usize = seg
                .parse()
                .map_err(|_| PatchError::InvalidPath(full_path.into()))?;
            let len = arr.len();
            arr.get_mut(idx)
                .ok_or_else(|| PatchError::IndexOutOfBounds {
                    path: full_path.into(),
                    index: idx,
                    len,
                })
        }
        _ => Err(PatchError::PathNotFound(full_path.into())),
    }
}

// ── Apply ───────────────────────────────────────────────────────────────────

/// Apply a sequence of patch operations atomically.
/// On error, the original document is unchanged.
pub fn apply(doc: &mut Value, ops: &[PatchOp]) -> Result<(), PatchError> {
    let backup = doc.clone();
    for op in ops {
        if let Err(e) = apply_one(doc, op) {
            *doc = backup;
            return Err(e);
        }
    }
    Ok(())
}

fn apply_one(doc: &mut Value, op: &PatchOp) -> Result<(), PatchError> {
    match op {
        PatchOp::Add { path, value } => pointer_add(doc, path, value.clone()),
        PatchOp::Remove { path } => {
            pointer_remove(doc, path)?;
            Ok(())
        }
        PatchOp::Replace { path, value } => {
            if path.is_empty() {
                *doc = value.clone();
                return Ok(());
            }
            pointer_remove(doc, path)?;
            pointer_add(doc, path, value.clone())
        }
        PatchOp::Move { from, path } => {
            let val = pointer_remove(doc, from)?;
            pointer_add(doc, path, val)
        }
        PatchOp::Copy { from, path } => {
            let val = pointer_get(doc, from)?.clone();
            pointer_add(doc, path, val)
        }
        PatchOp::Test { path, value } => {
            let actual = pointer_get(doc, path)?;
            if actual != value {
                Err(PatchError::TestFailed {
                    path: path.clone(),
                    expected: value.clone(),
                    actual: actual.clone(),
                })
            } else {
                Ok(())
            }
        }
    }
}

/// Parse a JSON array of patch operations.
pub fn parse_patch(patch: &Value) -> Result<Vec<PatchOp>, PatchError> {
    let arr = patch
        .as_array()
        .ok_or_else(|| PatchError::InvalidOperation("patch must be array".into()))?;
    arr.iter().map(PatchOp::from_value).collect()
}

// ── Diff ────────────────────────────────────────────────────────────────────

/// Generate a JSON Patch that transforms `left` into `right`.
pub fn diff(left: &Value, right: &Value) -> Vec<PatchOp> {
    let mut ops = Vec::new();
    diff_inner(left, right, String::new(), &mut ops);
    ops
}

fn diff_inner(left: &Value, right: &Value, path: String, ops: &mut Vec<PatchOp>) {
    if left == right {
        return;
    }
    match (left, right) {
        (Value::Object(lm), Value::Object(rm)) => {
            // Removed keys
            for key in lm.keys() {
                if !rm.contains_key(key) {
                    ops.push(PatchOp::Remove {
                        path: format!("{}/{}", path, escape_segment(key)),
                    });
                }
            }
            // Added or changed keys
            let mut keys: Vec<&String> = rm.keys().collect();
            keys.sort();
            for key in keys {
                let child_path = format!("{}/{}", path, escape_segment(key));
                match lm.get(key) {
                    Some(lv) => diff_inner(lv, &rm[key], child_path, ops),
                    None => ops.push(PatchOp::Add {
                        path: child_path,
                        value: rm[key].clone(),
                    }),
                }
            }
        }
        (Value::Array(la), Value::Array(ra)) => {
            // Simple strategy: replace differing elements, add/remove tail
            let min_len = la.len().min(ra.len());
            for i in 0..min_len {
                diff_inner(&la[i], &ra[i], format!("{path}/{i}"), ops);
            }
            // Extra in right -> add
            for i in min_len..ra.len() {
                ops.push(PatchOp::Add {
                    path: format!("{path}/-"),
                    value: ra[i].clone(),
                });
            }
            // Extra in left -> remove from end
            for i in (min_len..la.len()).rev() {
                ops.push(PatchOp::Remove {
                    path: format!("{path}/{i}"),
                });
            }
        }
        _ => {
            ops.push(PatchOp::Replace {
                path,
                value: right.clone(),
            });
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn add_to_object() {
        let mut doc = json!({"a": 1});
        apply(&mut doc, &[PatchOp::Add { path: "/b".into(), value: json!(2) }]).unwrap();
        assert_eq!(doc, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn add_to_array() {
        let mut doc = json!([1, 2, 3]);
        apply(&mut doc, &[PatchOp::Add { path: "/1".into(), value: json!(99) }]).unwrap();
        assert_eq!(doc, json!([1, 99, 2, 3]));
    }

    #[test]
    fn add_array_end() {
        let mut doc = json!([1]);
        apply(&mut doc, &[PatchOp::Add { path: "/-".into(), value: json!(2) }]).unwrap();
        assert_eq!(doc, json!([1, 2]));
    }

    #[test]
    fn remove_object_key() {
        let mut doc = json!({"a": 1, "b": 2});
        apply(&mut doc, &[PatchOp::Remove { path: "/b".into() }]).unwrap();
        assert_eq!(doc, json!({"a": 1}));
    }

    #[test]
    fn remove_array_element() {
        let mut doc = json!([1, 2, 3]);
        apply(&mut doc, &[PatchOp::Remove { path: "/1".into() }]).unwrap();
        assert_eq!(doc, json!([1, 3]));
    }

    #[test]
    fn replace_value() {
        let mut doc = json!({"a": 1});
        apply(&mut doc, &[PatchOp::Replace { path: "/a".into(), value: json!(99) }]).unwrap();
        assert_eq!(doc, json!({"a": 99}));
    }

    #[test]
    fn replace_root() {
        let mut doc = json!(1);
        apply(&mut doc, &[PatchOp::Replace { path: String::new(), value: json!(2) }]).unwrap();
        assert_eq!(doc, json!(2));
    }

    #[test]
    fn move_op() {
        let mut doc = json!({"a": 1, "b": {"c": 2}});
        apply(&mut doc, &[PatchOp::Move { from: "/b/c".into(), path: "/d".into() }]).unwrap();
        assert_eq!(doc["d"], json!(2));
        assert_eq!(doc["b"], json!({}));
    }

    #[test]
    fn copy_op() {
        let mut doc = json!({"a": [1, 2]});
        apply(&mut doc, &[PatchOp::Copy { from: "/a".into(), path: "/b".into() }]).unwrap();
        assert_eq!(doc["b"], json!([1, 2]));
        assert_eq!(doc["a"], json!([1, 2])); // original untouched
    }

    #[test]
    fn test_op_pass() {
        let mut doc = json!({"a": 42});
        apply(&mut doc, &[PatchOp::Test { path: "/a".into(), value: json!(42) }]).unwrap();
    }

    #[test]
    fn test_op_fail() {
        let mut doc = json!({"a": 42});
        let r = apply(&mut doc, &[PatchOp::Test { path: "/a".into(), value: json!(99) }]);
        assert!(matches!(r, Err(PatchError::TestFailed { .. })));
    }

    #[test]
    fn atomic_rollback() {
        let mut doc = json!({"a": 1, "b": 2});
        let original = doc.clone();
        let ops = vec![
            PatchOp::Replace { path: "/a".into(), value: json!(99) },
            PatchOp::Remove { path: "/nonexistent".into() }, // will fail
        ];
        let r = apply(&mut doc, &ops);
        assert!(r.is_err());
        assert_eq!(doc, original); // rolled back
    }

    #[test]
    fn parse_patch_from_json() {
        let patch = json!([
            {"op": "add", "path": "/x", "value": 1},
            {"op": "remove", "path": "/y"},
            {"op": "replace", "path": "/z", "value": 2},
            {"op": "move", "from": "/a", "path": "/b"},
            {"op": "copy", "from": "/c", "path": "/d"},
            {"op": "test", "path": "/e", "value": 3}
        ]);
        let ops = parse_patch(&patch).unwrap();
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn parse_patch_invalid() {
        assert!(parse_patch(&json!("not array")).is_err());
        assert!(parse_patch(&json!([{"op": "unknown", "path": "/x"}])).is_err());
        assert!(parse_patch(&json!([{"path": "/x"}])).is_err()); // missing op
    }

    #[test]
    fn escape_sequences() {
        let mut doc = json!({"a/b": 1, "c~d": 2});
        apply(&mut doc, &[PatchOp::Replace {
            path: "/a~1b".into(),
            value: json!(10),
        }])
        .unwrap();
        assert_eq!(doc["a/b"], json!(10));
        apply(&mut doc, &[PatchOp::Replace {
            path: "/c~0d".into(),
            value: json!(20),
        }])
        .unwrap();
        assert_eq!(doc["c~d"], json!(20));
    }

    #[test]
    fn diff_objects() {
        let left = json!({"a": 1, "b": 2, "c": 3});
        let right = json!({"a": 1, "b": 99, "d": 4});
        let ops = diff(&left, &right);
        let mut doc = left.clone();
        apply(&mut doc, &ops).unwrap();
        assert_eq!(doc, right);
    }

    #[test]
    fn diff_arrays() {
        let left = json!([1, 2, 3]);
        let right = json!([1, 99, 3, 4]);
        let ops = diff(&left, &right);
        let mut doc = left.clone();
        apply(&mut doc, &ops).unwrap();
        assert_eq!(doc, right);
    }

    #[test]
    fn diff_nested() {
        let left = json!({"x": {"y": 1}});
        let right = json!({"x": {"y": 2, "z": 3}});
        let ops = diff(&left, &right);
        let mut doc = left.clone();
        apply(&mut doc, &ops).unwrap();
        assert_eq!(doc, right);
    }

    #[test]
    fn diff_scalars() {
        let ops = diff(&json!(1), &json!(2));
        assert_eq!(ops.len(), 1);
        let mut doc = json!(1);
        apply(&mut doc, &ops).unwrap();
        assert_eq!(doc, json!(2));
    }

    #[test]
    fn diff_identical() {
        assert!(diff(&json!({"a": 1}), &json!({"a": 1})).is_empty());
    }

    #[test]
    fn roundtrip_to_value() {
        let ops = vec![
            PatchOp::Add { path: "/a".into(), value: json!(1) },
            PatchOp::Remove { path: "/b".into() },
            PatchOp::Move { from: "/c".into(), path: "/d".into() },
        ];
        let arr: Vec<Value> = ops.iter().map(|o| o.to_value()).collect();
        let parsed = parse_patch(&Value::Array(arr)).unwrap();
        assert_eq!(parsed, ops);
    }

    #[test]
    fn nested_array_operations() {
        let mut doc = json!({"items": [{"id": 1}, {"id": 2}]});
        apply(&mut doc, &[PatchOp::Add {
            path: "/items/1/name".into(),
            value: json!("test"),
        }])
        .unwrap();
        assert_eq!(doc["items"][1]["name"], json!("test"));
    }

    #[test]
    fn index_out_of_bounds() {
        let mut doc = json!([1, 2]);
        let r = apply(&mut doc, &[PatchOp::Add { path: "/5".into(), value: json!(3) }]);
        assert!(matches!(r, Err(PatchError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn error_display() {
        let e = PatchError::PathNotFound("/a/b".into());
        assert_eq!(format!("{e}"), "path not found: /a/b");
    }

    #[test]
    fn add_to_root() {
        let mut doc = json!(null);
        apply(&mut doc, &[PatchOp::Add { path: String::new(), value: json!({"new": "root"}) }]).unwrap();
        assert_eq!(doc, json!({"new": "root"}));
    }

    #[test]
    fn remove_root_fails() {
        let mut doc = json!(42);
        let r = apply(&mut doc, &[PatchOp::Remove { path: String::new() }]);
        assert!(r.is_err());
    }

    #[test]
    fn multiple_sequential_ops() {
        let mut doc = json!({"list": []});
        let ops = vec![
            PatchOp::Add { path: "/list/-".into(), value: json!(1) },
            PatchOp::Add { path: "/list/-".into(), value: json!(2) },
            PatchOp::Add { path: "/list/-".into(), value: json!(3) },
            PatchOp::Remove { path: "/list/1".into() },
        ];
        apply(&mut doc, &ops).unwrap();
        assert_eq!(doc, json!({"list": [1, 3]}));
    }
}
