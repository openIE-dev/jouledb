//! Data transformation operations.
//!
//! Replaces `lodash`, `Ramda`, `dataweave`, and similar transform libraries with
//! pure-Rust operations: field rename, type casting, computed columns, conditional
//! transforms, flatten/unflatten nested JSON, string transforms, and null handling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by data transformation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformError {
    /// Field not found in the record.
    FieldNotFound(String),
    /// Type cast failed.
    CastFailed { field: String, target: String, reason: String },
    /// Invalid transform configuration.
    InvalidConfig(String),
    /// Flatten/unflatten failed.
    StructureError(String),
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldNotFound(field) => write!(f, "field not found: {field}"),
            Self::CastFailed { field, target, reason } => {
                write!(f, "cast {field} to {target} failed: {reason}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::StructureError(msg) => write!(f, "structure error: {msg}"),
        }
    }
}

impl std::error::Error for TransformError {}

// ── Row type alias ───────────────────────────────────────────────

/// A single data row represented as key-value pairs.
pub type Row = HashMap<String, serde_json::Value>;

// ── Transform operations ─────────────────────────────────────────

/// A single transformation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOp {
    /// Rename a field.
    Rename { from: String, to: String },
    /// Remove a field.
    Remove(String),
    /// Cast a field to a different type.
    Cast { field: String, target: CastTarget },
    /// Add a field with a constant value.
    AddConstant { field: String, value: serde_json::Value },
    /// Copy one field to another.
    Copy { from: String, to: String },
    /// String transform on a field.
    StringTransform { field: String, op: StringOp },
    /// Replace null with a default value.
    DefaultValue { field: String, default: serde_json::Value },
    /// Coalesce: use the first non-null value from a list of fields.
    Coalesce { fields: Vec<String>, output: String },
    /// Conditional transform: if condition field matches value, apply transform.
    Conditional {
        condition_field: String,
        condition_value: serde_json::Value,
        then_ops: Vec<TransformOp>,
    },
    /// Flatten a nested object into dot-separated keys.
    Flatten { field: String, separator: String },
    /// Unflatten dot-separated keys back into nested objects.
    Unflatten { separator: String },
}

/// Target type for cast operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CastTarget {
    String,
    Integer,
    Float,
    Boolean,
}

impl fmt::Display for CastTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Integer => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::Boolean => write!(f, "boolean"),
        }
    }
}

/// String transformation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StringOp {
    /// Trim whitespace.
    Trim,
    /// Convert to uppercase.
    Upper,
    /// Convert to lowercase.
    Lower,
    /// Replace occurrences.
    Replace { from: String, to: String },
    /// Take a substring.
    Substring { start: usize, length: usize },
    /// Prepend a prefix.
    Prepend(String),
    /// Append a suffix.
    Append(String),
    /// Pad left to a minimum length.
    PadLeft { min_length: usize, pad_char: char },
    /// Pad right to a minimum length.
    PadRight { min_length: usize, pad_char: char },
}

// ── Transform engine ─────────────────────────────────────────────

/// The transform engine that applies operations to data rows.
#[derive(Debug, Clone)]
pub struct TransformEngine {
    /// Ordered list of operations.
    ops: Vec<TransformOp>,
    /// Whether to skip errors and continue (true) or fail (false).
    lenient: bool,
}

impl TransformEngine {
    /// Create a new transform engine.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            lenient: false,
        }
    }

    /// Set lenient mode (skip errors instead of failing).
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
    }

    /// Add an operation.
    pub fn add_op(&mut self, op: TransformOp) {
        self.ops.push(op);
    }

    /// Number of operations.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Transform a single row.
    pub fn transform_row(&self, row: &Row) -> Result<Row, TransformError> {
        let mut result = row.clone();
        for op in &self.ops {
            match self.apply_op(&mut result, op) {
                Ok(()) => {}
                Err(e) if self.lenient => {
                    // In lenient mode, skip the failed operation.
                    let _ = e;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(result)
    }

    /// Transform a dataset.
    pub fn transform(&self, data: &[Row]) -> Result<Vec<Row>, TransformError> {
        data.iter().map(|row| self.transform_row(row)).collect()
    }

    /// Apply a single operation to a row.
    fn apply_op(&self, row: &mut Row, op: &TransformOp) -> Result<(), TransformError> {
        match op {
            TransformOp::Rename { from, to } => {
                if let Some(value) = row.remove(from) {
                    row.insert(to.clone(), value);
                }
                Ok(())
            }
            TransformOp::Remove(field) => {
                row.remove(field);
                Ok(())
            }
            TransformOp::Cast { field, target } => {
                if let Some(value) = row.get(field).cloned() {
                    let casted = cast_value(&value, *target).map_err(|reason| {
                        TransformError::CastFailed {
                            field: field.clone(),
                            target: target.to_string(),
                            reason,
                        }
                    })?;
                    row.insert(field.clone(), casted);
                }
                Ok(())
            }
            TransformOp::AddConstant { field, value } => {
                row.insert(field.clone(), value.clone());
                Ok(())
            }
            TransformOp::Copy { from, to } => {
                if let Some(value) = row.get(from).cloned() {
                    row.insert(to.clone(), value);
                }
                Ok(())
            }
            TransformOp::StringTransform { field, op: string_op } => {
                if let Some(serde_json::Value::String(s)) = row.get(field) {
                    let transformed = apply_string_op(s, string_op);
                    row.insert(field.clone(), serde_json::Value::String(transformed));
                }
                Ok(())
            }
            TransformOp::DefaultValue { field, default } => {
                let needs_default = match row.get(field) {
                    None => true,
                    Some(v) if v.is_null() => true,
                    _ => false,
                };
                if needs_default {
                    row.insert(field.clone(), default.clone());
                }
                Ok(())
            }
            TransformOp::Coalesce { fields, output } => {
                let mut found = None;
                for f in fields {
                    if let Some(v) = row.get(f) {
                        if !v.is_null() {
                            found = Some(v.clone());
                            break;
                        }
                    }
                }
                if let Some(v) = found {
                    row.insert(output.clone(), v);
                }
                Ok(())
            }
            TransformOp::Conditional {
                condition_field,
                condition_value,
                then_ops,
            } => {
                let matches = row.get(condition_field) == Some(condition_value);
                if matches {
                    for sub_op in then_ops {
                        self.apply_op(row, sub_op)?;
                    }
                }
                Ok(())
            }
            TransformOp::Flatten { field, separator } => {
                if let Some(serde_json::Value::Object(obj)) = row.remove(field) {
                    let flattened = flatten_object(&serde_json::Value::Object(obj), field, separator);
                    for (k, v) in flattened {
                        row.insert(k, v);
                    }
                }
                Ok(())
            }
            TransformOp::Unflatten { separator } => {
                let keys: Vec<String> = row.keys().cloned().collect();
                let mut nested_keys = Vec::new();
                for key in &keys {
                    if key.contains(separator.as_str()) {
                        nested_keys.push(key.clone());
                    }
                }
                for key in &nested_keys {
                    let value = row.remove(key).unwrap();
                    unflatten_key(row, key, separator, value);
                }
                Ok(())
            }
        }
    }
}

impl Default for TransformEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper functions ─────────────────────────────────────────────

/// Cast a JSON value to the target type.
fn cast_value(
    value: &serde_json::Value,
    target: CastTarget,
) -> Result<serde_json::Value, String> {
    match target {
        CastTarget::String => match value {
            serde_json::Value::String(_) => Ok(value.clone()),
            serde_json::Value::Number(n) => Ok(serde_json::json!(n.to_string())),
            serde_json::Value::Bool(b) => Ok(serde_json::json!(b.to_string())),
            serde_json::Value::Null => Ok(serde_json::json!("")),
            _ => Err(format!("cannot cast {value} to string")),
        },
        CastTarget::Integer => match value {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(serde_json::json!(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(serde_json::json!(f as i64))
                } else {
                    Err("number out of i64 range".into())
                }
            }
            serde_json::Value::String(s) => {
                let i: i64 = s.parse().map_err(|e| format!("parse int: {e}"))?;
                Ok(serde_json::json!(i))
            }
            serde_json::Value::Bool(b) => Ok(serde_json::json!(if *b { 1 } else { 0 })),
            _ => Err(format!("cannot cast to integer")),
        },
        CastTarget::Float => match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    Ok(serde_json::json!(f))
                } else {
                    Err("number not convertible to f64".into())
                }
            }
            serde_json::Value::String(s) => {
                let f: f64 = s.parse().map_err(|e| format!("parse float: {e}"))?;
                Ok(serde_json::json!(f))
            }
            serde_json::Value::Bool(b) => Ok(serde_json::json!(if *b { 1.0 } else { 0.0 })),
            _ => Err(format!("cannot cast to float")),
        },
        CastTarget::Boolean => match value {
            serde_json::Value::Bool(_) => Ok(value.clone()),
            serde_json::Value::Number(n) => Ok(serde_json::json!(n.as_f64().unwrap_or(0.0) != 0.0)),
            serde_json::Value::String(s) => {
                let b = matches!(s.as_str(), "true" | "1" | "yes" | "on");
                Ok(serde_json::json!(b))
            }
            serde_json::Value::Null => Ok(serde_json::json!(false)),
            _ => Err(format!("cannot cast to boolean")),
        },
    }
}

/// Apply a string operation.
fn apply_string_op(s: &str, op: &StringOp) -> String {
    match op {
        StringOp::Trim => s.trim().to_string(),
        StringOp::Upper => s.to_uppercase(),
        StringOp::Lower => s.to_lowercase(),
        StringOp::Replace { from, to } => s.replace(from.as_str(), to.as_str()),
        StringOp::Substring { start, length } => {
            let chars: Vec<char> = s.chars().collect();
            let end = (*start + *length).min(chars.len());
            let actual_start = (*start).min(chars.len());
            chars[actual_start..end].iter().collect()
        }
        StringOp::Prepend(prefix) => format!("{prefix}{s}"),
        StringOp::Append(suffix) => format!("{s}{suffix}"),
        StringOp::PadLeft { min_length, pad_char } => {
            if s.len() >= *min_length {
                s.to_string()
            } else {
                let padding: String = std::iter::repeat(*pad_char).take(min_length - s.len()).collect();
                format!("{padding}{s}")
            }
        }
        StringOp::PadRight { min_length, pad_char } => {
            if s.len() >= *min_length {
                s.to_string()
            } else {
                let padding: String = std::iter::repeat(*pad_char).take(min_length - s.len()).collect();
                format!("{s}{padding}")
            }
        }
    }
}

/// Flatten a nested JSON object into dot-separated keys.
fn flatten_object(
    value: &serde_json::Value,
    prefix: &str,
    separator: &str,
) -> Vec<(String, serde_json::Value)> {
    let mut result = Vec::new();
    if let serde_json::Value::Object(map) = value {
        for (key, val) in map {
            let full_key = format!("{prefix}{separator}{key}");
            if val.is_object() {
                result.extend(flatten_object(val, &full_key, separator));
            } else {
                result.push((full_key, val.clone()));
            }
        }
    } else {
        result.push((prefix.to_string(), value.clone()));
    }
    result
}

/// Unflatten a dot-separated key into nested objects in the row.
fn unflatten_key(
    row: &mut Row,
    key: &str,
    separator: &str,
    value: serde_json::Value,
) {
    let parts: Vec<&str> = key.splitn(2, separator).collect();
    if parts.len() == 1 {
        row.insert(key.to_string(), value);
        return;
    }

    let first = parts[0];
    let rest = parts[1];

    let entry = row
        .entry(first.to_string())
        .or_insert_with(|| serde_json::json!({}));

    if let serde_json::Value::Object(map) = entry {
        let mut sub_row: Row = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        unflatten_key(&mut sub_row, rest, separator, value);
        *entry = serde_json::Value::Object(
            sub_row.into_iter().collect(),
        );
    }
}

// ── Convenience constructors ─────────────────────────────────────

/// Create a rename operation.
pub fn rename(from: impl Into<String>, to: impl Into<String>) -> TransformOp {
    TransformOp::Rename {
        from: from.into(),
        to: to.into(),
    }
}

/// Create a cast operation.
pub fn cast(field: impl Into<String>, target: CastTarget) -> TransformOp {
    TransformOp::Cast {
        field: field.into(),
        target,
    }
}

/// Create a default value operation.
pub fn default_value(field: impl Into<String>, default: serde_json::Value) -> TransformOp {
    TransformOp::DefaultValue {
        field: field.into(),
        default,
    }
}

/// Create a string transform operation.
pub fn string_transform(field: impl Into<String>, op: StringOp) -> TransformOp {
    TransformOp::StringTransform {
        field: field.into(),
        op,
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn rename_field() {
        let mut engine = TransformEngine::new();
        engine.add_op(rename("first_name", "name"));

        let input = row(&[("first_name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("Alice")));
        assert!(!output.contains_key("first_name"));
    }

    #[test]
    fn rename_missing_field_is_noop() {
        let mut engine = TransformEngine::new();
        engine.add_op(rename("nonexistent", "target"));

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert!(!output.contains_key("target"));
        assert!(output.contains_key("name"));
    }

    #[test]
    fn remove_field() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Remove("temp".into()));

        let input = row(&[
            ("name", serde_json::json!("Alice")),
            ("temp", serde_json::json!(42)),
        ]);
        let output = engine.transform_row(&input).unwrap();
        assert!(!output.contains_key("temp"));
        assert!(output.contains_key("name"));
    }

    #[test]
    fn cast_number_to_string() {
        let mut engine = TransformEngine::new();
        engine.add_op(cast("age", CastTarget::String));

        let input = row(&[("age", serde_json::json!(42))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("age"), Some(&serde_json::json!("42")));
    }

    #[test]
    fn cast_string_to_integer() {
        let mut engine = TransformEngine::new();
        engine.add_op(cast("count", CastTarget::Integer));

        let input = row(&[("count", serde_json::json!("100"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("count"), Some(&serde_json::json!(100)));
    }

    #[test]
    fn cast_string_to_float() {
        let mut engine = TransformEngine::new();
        engine.add_op(cast("price", CastTarget::Float));

        let input = row(&[("price", serde_json::json!("3.14"))]);
        let output = engine.transform_row(&input).unwrap();
        let val = output.get("price").unwrap().as_f64().unwrap();
        assert!((val - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn cast_to_boolean() {
        let mut engine = TransformEngine::new();
        engine.add_op(cast("active", CastTarget::Boolean));

        let input = row(&[("active", serde_json::json!("true"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("active"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn cast_invalid_fails() {
        let mut engine = TransformEngine::new();
        engine.add_op(cast("name", CastTarget::Integer));

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let result = engine.transform_row(&input);
        assert!(result.is_err());
    }

    #[test]
    fn add_constant() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::AddConstant {
            field: "source".into(),
            value: serde_json::json!("csv"),
        });

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("source"), Some(&serde_json::json!("csv")));
    }

    #[test]
    fn copy_field() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Copy {
            from: "name".into(),
            to: "name_backup".into(),
        });

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("Alice")));
        assert_eq!(output.get("name_backup"), Some(&serde_json::json!("Alice")));
    }

    #[test]
    fn string_trim() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform("name", StringOp::Trim));

        let input = row(&[("name", serde_json::json!("  Alice  "))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("Alice")));
    }

    #[test]
    fn string_upper() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform("name", StringOp::Upper));

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("ALICE")));
    }

    #[test]
    fn string_lower() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform("name", StringOp::Lower));

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("alice")));
    }

    #[test]
    fn string_replace() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform(
            "text",
            StringOp::Replace {
                from: "hello".into(),
                to: "world".into(),
            },
        ));

        let input = row(&[("text", serde_json::json!("say hello"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("text"), Some(&serde_json::json!("say world")));
    }

    #[test]
    fn string_substring() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform(
            "code",
            StringOp::Substring { start: 0, length: 3 },
        ));

        let input = row(&[("code", serde_json::json!("ABCDEF"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("code"), Some(&serde_json::json!("ABC")));
    }

    #[test]
    fn string_pad_left() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform(
            "id",
            StringOp::PadLeft { min_length: 5, pad_char: '0' },
        ));

        let input = row(&[("id", serde_json::json!("42"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("id"), Some(&serde_json::json!("00042")));
    }

    #[test]
    fn default_value_for_null() {
        let mut engine = TransformEngine::new();
        engine.add_op(default_value("status", serde_json::json!("unknown")));

        let input = row(&[("status", serde_json::Value::Null)]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("status"), Some(&serde_json::json!("unknown")));
    }

    #[test]
    fn default_value_for_missing() {
        let mut engine = TransformEngine::new();
        engine.add_op(default_value("status", serde_json::json!("unknown")));

        let input = row(&[("name", serde_json::json!("Alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("status"), Some(&serde_json::json!("unknown")));
    }

    #[test]
    fn default_value_does_not_overwrite() {
        let mut engine = TransformEngine::new();
        engine.add_op(default_value("status", serde_json::json!("unknown")));

        let input = row(&[("status", serde_json::json!("active"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("status"), Some(&serde_json::json!("active")));
    }

    #[test]
    fn coalesce_picks_first_non_null() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Coalesce {
            fields: vec!["a".into(), "b".into(), "c".into()],
            output: "result".into(),
        });

        let input = row(&[
            ("a", serde_json::Value::Null),
            ("b", serde_json::json!(42)),
            ("c", serde_json::json!(99)),
        ]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("result"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn conditional_applies_when_matches() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Conditional {
            condition_field: "type".into(),
            condition_value: serde_json::json!("premium"),
            then_ops: vec![TransformOp::AddConstant {
                field: "discount".into(),
                value: serde_json::json!(0.2),
            }],
        });

        let input = row(&[("type", serde_json::json!("premium"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("discount"), Some(&serde_json::json!(0.2)));
    }

    #[test]
    fn conditional_does_not_apply_when_no_match() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Conditional {
            condition_field: "type".into(),
            condition_value: serde_json::json!("premium"),
            then_ops: vec![TransformOp::AddConstant {
                field: "discount".into(),
                value: serde_json::json!(0.2),
            }],
        });

        let input = row(&[("type", serde_json::json!("basic"))]);
        let output = engine.transform_row(&input).unwrap();
        assert!(!output.contains_key("discount"));
    }

    #[test]
    fn flatten_nested_object() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Flatten {
            field: "address".into(),
            separator: ".".into(),
        });

        let input = row(&[(
            "address",
            serde_json::json!({
                "city": "NYC",
                "zip": "10001"
            }),
        )]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("address.city"), Some(&serde_json::json!("NYC")));
        assert_eq!(output.get("address.zip"), Some(&serde_json::json!("10001")));
    }

    #[test]
    fn unflatten_keys() {
        let mut engine = TransformEngine::new();
        engine.add_op(TransformOp::Unflatten {
            separator: ".".into(),
        });

        let input = row(&[
            ("address.city", serde_json::json!("NYC")),
            ("address.zip", serde_json::json!("10001")),
            ("name", serde_json::json!("Alice")),
        ]);
        let output = engine.transform_row(&input).unwrap();
        assert!(output.contains_key("address"));
        assert!(!output.contains_key("address.city"));
        let addr = output.get("address").unwrap();
        assert_eq!(addr.get("city"), Some(&serde_json::json!("NYC")));
    }

    #[test]
    fn transform_dataset() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform("name", StringOp::Upper));

        let data = vec![
            row(&[("name", serde_json::json!("alice"))]),
            row(&[("name", serde_json::json!("bob"))]),
        ];
        let output = engine.transform(&data).unwrap();
        assert_eq!(output[0].get("name"), Some(&serde_json::json!("ALICE")));
        assert_eq!(output[1].get("name"), Some(&serde_json::json!("BOB")));
    }

    #[test]
    fn lenient_mode_skips_errors() {
        let mut engine = TransformEngine::new();
        engine.set_lenient(true);
        engine.add_op(cast("name", CastTarget::Integer)); // will fail
        engine.add_op(string_transform("name", StringOp::Upper)); // should still run

        let input = row(&[("name", serde_json::json!("alice"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("ALICE")));
    }

    #[test]
    fn chained_transforms() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform("name", StringOp::Trim));
        engine.add_op(string_transform("name", StringOp::Upper));
        engine.add_op(rename("name", "full_name"));

        let input = row(&[("name", serde_json::json!("  alice  "))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("full_name"), Some(&serde_json::json!("ALICE")));
    }

    #[test]
    fn string_prepend_append() {
        let mut engine = TransformEngine::new();
        engine.add_op(string_transform("name", StringOp::Prepend("Mr. ".into())));
        engine.add_op(string_transform("name", StringOp::Append(" Jr.".into())));

        let input = row(&[("name", serde_json::json!("Smith"))]);
        let output = engine.transform_row(&input).unwrap();
        assert_eq!(output.get("name"), Some(&serde_json::json!("Mr. Smith Jr.")));
    }

    #[test]
    fn error_display() {
        let e = TransformError::FieldNotFound("x".into());
        assert!(format!("{e}").contains("field not found"));
        let e2 = TransformError::CastFailed {
            field: "a".into(),
            target: "int".into(),
            reason: "bad".into(),
        };
        assert!(format!("{e2}").contains("cast"));
    }

    #[test]
    fn cast_bool_to_int() {
        let result = cast_value(&serde_json::json!(true), CastTarget::Integer).unwrap();
        assert_eq!(result, serde_json::json!(1));
    }

    #[test]
    fn cast_null_to_string() {
        let result = cast_value(&serde_json::Value::Null, CastTarget::String).unwrap();
        assert_eq!(result, serde_json::json!(""));
    }
}
