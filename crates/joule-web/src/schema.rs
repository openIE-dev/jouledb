//! Runtime validation: type-safe schema definition and data validation.
//!
//! Replaces Zod, Yup, Joi with a pure-Rust builder API that validates
//! `serde_json::Value` trees and collects dotted-path errors.

use std::fmt;
use serde_json::Value;

// ── Errors ──────────────────────────────────────────────────────

/// A single validation error anchored to a JSON path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    pub path: String,
    pub message: String,
    pub rule: String,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

// ── Schema types ────────────────────────────────────────────────

/// Format constraints for string schemas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringFormat {
    Email,
    Url,
    Uuid,
    Date,
    DateTime,
    Ip,
}

/// A single field inside an `Object` schema.
#[derive(Debug, Clone)]
pub struct FieldSchema {
    pub name: String,
    pub schema: Schema,
    pub required: bool,
}

/// Recursive schema definition.
#[derive(Debug, Clone)]
pub enum Schema {
    String_ {
        min_len: Option<usize>,
        max_len: Option<usize>,
        pattern: Option<String>,
        format: Option<StringFormat>,
    },
    Number {
        min: Option<f64>,
        max: Option<f64>,
        integer: bool,
    },
    Boolean,
    Array {
        item: Box<Schema>,
        min_items: Option<usize>,
        max_items: Option<usize>,
    },
    Object {
        fields: Vec<FieldSchema>,
        allow_extra: bool,
    },
    Optional(Box<Schema>),
    OneOf(Vec<Schema>),
    Enum_(Vec<String>),
    Null,
    Any,
}

// ── Validation ──────────────────────────────────────────────────

/// Validate a JSON value against a schema, collecting all errors.
pub fn validate(schema: &Schema, value: &Value) -> Result<(), Vec<SchemaError>> {
    let mut errors = Vec::new();
    validate_inner(schema, value, "", &mut errors);
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn validate_inner(schema: &Schema, value: &Value, path: &str, errors: &mut Vec<SchemaError>) {
    match schema {
        Schema::Any => {}
        Schema::Null => {
            if !value.is_null() {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected null".into(),
                    rule: "null".into(),
                });
            }
        }
        Schema::Boolean => {
            if !value.is_boolean() {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected boolean".into(),
                    rule: "type".into(),
                });
            }
        }
        Schema::String_ { min_len, max_len, pattern, format } => {
            if let Some(s) = value.as_str() {
                if let Some(min) = min_len {
                    if s.len() < *min {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("string length {} < minimum {}", s.len(), min),
                            rule: "min_length".into(),
                        });
                    }
                }
                if let Some(max) = max_len {
                    if s.len() > *max {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("string length {} > maximum {}", s.len(), max),
                            rule: "max_length".into(),
                        });
                    }
                }
                if let Some(pat) = pattern {
                    if !simple_pattern_match(pat, s) {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("does not match pattern {pat}"),
                            rule: "pattern".into(),
                        });
                    }
                }
                if let Some(fmt) = format {
                    if !check_format(fmt, s) {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("invalid format: {fmt:?}"),
                            rule: "format".into(),
                        });
                    }
                }
            } else {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected string".into(),
                    rule: "type".into(),
                });
            }
        }
        Schema::Number { min, max, integer } => {
            if let Some(n) = value.as_f64() {
                if *integer && n.fract() != 0.0 {
                    errors.push(SchemaError {
                        path: path.to_string(),
                        message: "expected integer".into(),
                        rule: "integer".into(),
                    });
                }
                if let Some(lo) = min {
                    if n < *lo {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("{n} < minimum {lo}"),
                            rule: "min".into(),
                        });
                    }
                }
                if let Some(hi) = max {
                    if n > *hi {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("{n} > maximum {hi}"),
                            rule: "max".into(),
                        });
                    }
                }
            } else {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected number".into(),
                    rule: "type".into(),
                });
            }
        }
        Schema::Array { item, min_items, max_items } => {
            if let Some(arr) = value.as_array() {
                if let Some(min) = min_items {
                    if arr.len() < *min {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("array length {} < minimum {}", arr.len(), min),
                            rule: "min_items".into(),
                        });
                    }
                }
                if let Some(max) = max_items {
                    if arr.len() > *max {
                        errors.push(SchemaError {
                            path: path.to_string(),
                            message: format!("array length {} > maximum {}", arr.len(), max),
                            rule: "max_items".into(),
                        });
                    }
                }
                for (i, v) in arr.iter().enumerate() {
                    let child = if path.is_empty() {
                        format!("[{i}]")
                    } else {
                        format!("{path}[{i}]")
                    };
                    validate_inner(item, v, &child, errors);
                }
            } else {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected array".into(),
                    rule: "type".into(),
                });
            }
        }
        Schema::Object { fields, allow_extra } => {
            if let Some(obj) = value.as_object() {
                for field in fields {
                    let child = if path.is_empty() {
                        field.name.clone()
                    } else {
                        format!("{path}.{}", field.name)
                    };
                    match obj.get(&field.name) {
                        Some(v) => validate_inner(&field.schema, v, &child, errors),
                        None if field.required => {
                            errors.push(SchemaError {
                                path: child,
                                message: "required field missing".into(),
                                rule: "required".into(),
                            });
                        }
                        None => {}
                    }
                }
                if !allow_extra {
                    let known: std::collections::HashSet<&str> =
                        fields.iter().map(|f| f.name.as_str()).collect();
                    for key in obj.keys() {
                        if !known.contains(key.as_str()) {
                            let child = if path.is_empty() {
                                key.clone()
                            } else {
                                format!("{path}.{key}")
                            };
                            errors.push(SchemaError {
                                path: child,
                                message: format!("unexpected field '{key}'"),
                                rule: "no_extra".into(),
                            });
                        }
                    }
                }
            } else {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected object".into(),
                    rule: "type".into(),
                });
            }
        }
        Schema::Optional(inner) => {
            if !value.is_null() {
                validate_inner(inner, value, path, errors);
            }
        }
        Schema::OneOf(variants) => {
            let matched = variants.iter().any(|v| {
                let mut tmp = Vec::new();
                validate_inner(v, value, path, &mut tmp);
                tmp.is_empty()
            });
            if !matched {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "does not match any variant".into(),
                    rule: "one_of".into(),
                });
            }
        }
        Schema::Enum_(variants) => {
            if let Some(s) = value.as_str() {
                if !variants.iter().any(|v| v == s) {
                    errors.push(SchemaError {
                        path: path.to_string(),
                        message: format!("'{s}' not in enum"),
                        rule: "enum".into(),
                    });
                }
            } else {
                errors.push(SchemaError {
                    path: path.to_string(),
                    message: "expected string for enum".into(),
                    rule: "type".into(),
                });
            }
        }
    }
}

/// Minimal pattern matching (prefix/suffix wildcards only, no regex dep).
fn simple_pattern_match(pattern: &str, value: &str) -> bool {
    if pattern.starts_with('^') && pattern.ends_with('$') {
        value == &pattern[1..pattern.len() - 1]
    } else if pattern.starts_with('^') {
        value.starts_with(&pattern[1..])
    } else if pattern.ends_with('$') {
        value.ends_with(&pattern[..pattern.len() - 1])
    } else {
        value.contains(pattern)
    }
}

fn check_format(fmt: &StringFormat, s: &str) -> bool {
    match fmt {
        StringFormat::Email => s.contains('@') && s.contains('.') && s.len() >= 5,
        StringFormat::Url => s.starts_with("http://") || s.starts_with("https://"),
        StringFormat::Uuid => {
            s.len() == 36
                && s.chars()
                    .enumerate()
                    .all(|(i, c)| match i {
                        8 | 13 | 18 | 23 => c == '-',
                        _ => c.is_ascii_hexdigit(),
                    })
        }
        StringFormat::Date => {
            // YYYY-MM-DD
            s.len() == 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-'
                && s[0..4].parse::<u16>().is_ok()
                && s[5..7].parse::<u8>().is_ok()
                && s[8..10].parse::<u8>().is_ok()
        }
        StringFormat::DateTime => {
            s.contains('T') && s.len() >= 19
        }
        StringFormat::Ip => {
            let parts: Vec<&str> = s.split('.').collect();
            parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok())
        }
    }
}

// ── Coercion ────────────────────────────────────────────────────

/// Try to coerce a value to match the expected schema type.
pub fn coerce(schema: &Schema, value: &Value) -> Value {
    match schema {
        Schema::Number { .. } => {
            if let Some(s) = value.as_str() {
                if let Ok(n) = s.parse::<f64>() {
                    return Value::from(n);
                }
            }
            value.clone()
        }
        Schema::Boolean => {
            if let Some(s) = value.as_str() {
                return match s {
                    "true" | "1" | "yes" => Value::Bool(true),
                    "false" | "0" | "no" => Value::Bool(false),
                    _ => value.clone(),
                };
            }
            value.clone()
        }
        Schema::String_ { .. } => {
            match value {
                Value::Number(n) => Value::String(n.to_string()),
                Value::Bool(b) => Value::String(b.to_string()),
                _ => value.clone(),
            }
        }
        Schema::Array { item, .. } => {
            if let Some(arr) = value.as_array() {
                Value::Array(arr.iter().map(|v| coerce(item, v)).collect())
            } else {
                value.clone()
            }
        }
        Schema::Optional(inner) => {
            if value.is_null() { value.clone() } else { coerce(inner, value) }
        }
        _ => value.clone(),
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Fluent builder for `Schema` values.
pub struct SchemaBuilder {
    schema: Schema,
}

impl Schema {
    pub fn string() -> SchemaBuilder {
        SchemaBuilder {
            schema: Schema::String_ {
                min_len: None,
                max_len: None,
                pattern: None,
                format: None,
            },
        }
    }

    pub fn number() -> SchemaBuilder {
        SchemaBuilder {
            schema: Schema::Number { min: None, max: None, integer: false },
        }
    }

    pub fn bool() -> SchemaBuilder {
        SchemaBuilder { schema: Schema::Boolean }
    }

    pub fn array(item: Schema) -> SchemaBuilder {
        SchemaBuilder {
            schema: Schema::Array {
                item: Box::new(item),
                min_items: None,
                max_items: None,
            },
        }
    }

    pub fn object() -> SchemaBuilder {
        SchemaBuilder {
            schema: Schema::Object {
                fields: Vec::new(),
                allow_extra: false,
            },
        }
    }
}

impl SchemaBuilder {
    pub fn min(mut self, n: f64) -> Self {
        if let Schema::Number { ref mut min, .. } = self.schema { *min = Some(n); }
        self
    }

    pub fn max(mut self, n: f64) -> Self {
        if let Schema::Number { ref mut max, .. } = self.schema { *max = Some(n); }
        self
    }

    pub fn min_length(mut self, n: usize) -> Self {
        if let Schema::String_ { ref mut min_len, .. } = self.schema { *min_len = Some(n); }
        self
    }

    pub fn max_length(mut self, n: usize) -> Self {
        if let Schema::String_ { ref mut max_len, .. } = self.schema { *max_len = Some(n); }
        self
    }

    pub fn pattern(mut self, regex: &str) -> Self {
        if let Schema::String_ { ref mut pattern, .. } = self.schema {
            *pattern = Some(regex.to_string());
        }
        self
    }

    pub fn format(mut self, fmt: StringFormat) -> Self {
        if let Schema::String_ { ref mut format, .. } = self.schema {
            *format = Some(fmt);
        }
        self
    }

    pub fn integer(mut self) -> Self {
        if let Schema::Number { ref mut integer, .. } = self.schema { *integer = true; }
        self
    }

    pub fn optional(self) -> Schema {
        Schema::Optional(Box::new(self.schema))
    }

    pub fn min_items(mut self, n: usize) -> Self {
        if let Schema::Array { ref mut min_items, .. } = self.schema { *min_items = Some(n); }
        self
    }

    pub fn max_items(mut self, n: usize) -> Self {
        if let Schema::Array { ref mut max_items, .. } = self.schema { *max_items = Some(n); }
        self
    }

    pub fn field(mut self, name: &str, schema: Schema, required: bool) -> Self {
        if let Schema::Object { ref mut fields, .. } = self.schema {
            fields.push(FieldSchema {
                name: name.to_string(),
                schema,
                required,
            });
        }
        self
    }

    pub fn allow_extra(mut self) -> Self {
        if let Schema::Object { ref mut allow_extra, .. } = self.schema { *allow_extra = true; }
        self
    }

    pub fn build(self) -> Schema {
        self.schema
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn string_min_length() {
        let s = Schema::string().min_length(3).build();
        assert!(validate(&s, &json!("abc")).is_ok());
        assert!(validate(&s, &json!("ab")).is_err());
    }

    #[test]
    fn string_max_length() {
        let s = Schema::string().max_length(5).build();
        assert!(validate(&s, &json!("hello")).is_ok());
        assert!(validate(&s, &json!("toolong")).is_err());
    }

    #[test]
    fn email_format() {
        let s = Schema::string().format(StringFormat::Email).build();
        assert!(validate(&s, &json!("user@example.com")).is_ok());
        assert!(validate(&s, &json!("nope")).is_err());
    }

    #[test]
    fn number_range() {
        let s = Schema::number().min(0.0).max(100.0).build();
        assert!(validate(&s, &json!(50)).is_ok());
        assert!(validate(&s, &json!(101)).is_err());
        assert!(validate(&s, &json!(-1)).is_err());
    }

    #[test]
    fn integer_rejects_float() {
        let s = Schema::number().integer().build();
        assert!(validate(&s, &json!(42)).is_ok());
        assert!(validate(&s, &json!(3.14)).is_err());
    }

    #[test]
    fn boolean_validation() {
        let s = Schema::bool().build();
        assert!(validate(&s, &json!(true)).is_ok());
        assert!(validate(&s, &json!("true")).is_err());
    }

    #[test]
    fn array_min_max_items() {
        let s = Schema::array(Schema::number().build()).min_items(1).max_items(3).build();
        assert!(validate(&s, &json!([1, 2])).is_ok());
        assert!(validate(&s, &json!([])).is_err());
        assert!(validate(&s, &json!([1, 2, 3, 4])).is_err());
    }

    #[test]
    fn object_required_field_missing() {
        let s = Schema::object()
            .field("name", Schema::string().build(), true)
            .build();
        assert!(validate(&s, &json!({})).is_err());
        assert!(validate(&s, &json!({"name": "ok"})).is_ok());
    }

    #[test]
    fn object_extra_field_rejected() {
        let s = Schema::object()
            .field("name", Schema::string().build(), true)
            .build();
        let errs = validate(&s, &json!({"name": "ok", "extra": 1})).unwrap_err();
        assert!(errs.iter().any(|e| e.rule == "no_extra"));
    }

    #[test]
    fn optional_accepts_null() {
        let s = Schema::string().optional();
        assert!(validate(&s, &json!(null)).is_ok());
        assert!(validate(&s, &json!("hi")).is_ok());
        assert!(validate(&s, &json!(42)).is_err());
    }

    #[test]
    fn one_of_matches() {
        let s = Schema::OneOf(vec![
            Schema::string().build(),
            Schema::number().build(),
        ]);
        assert!(validate(&s, &json!("hi")).is_ok());
        assert!(validate(&s, &json!(42)).is_ok());
        assert!(validate(&s, &json!(true)).is_err());
    }

    #[test]
    fn enum_validation() {
        let s = Schema::Enum_(vec!["red".into(), "green".into(), "blue".into()]);
        assert!(validate(&s, &json!("red")).is_ok());
        assert!(validate(&s, &json!("yellow")).is_err());
    }

    #[test]
    fn nested_object_path_errors() {
        let addr = Schema::object()
            .field("city", Schema::string().build(), true)
            .build();
        let s = Schema::object()
            .field("address", addr, true)
            .build();
        let errs = validate(&s, &json!({"address": {}})).unwrap_err();
        assert!(errs.iter().any(|e| e.path == "address.city"));
    }

    #[test]
    fn coerce_string_to_number() {
        let s = Schema::number().build();
        let v = coerce(&s, &json!("42"));
        assert_eq!(v, json!(42.0));
    }

    #[test]
    fn coerce_string_to_boolean() {
        let s = Schema::Boolean;
        assert_eq!(coerce(&s, &json!("true")), json!(true));
        assert_eq!(coerce(&s, &json!("false")), json!(false));
    }

    #[test]
    fn complex_nested_schema() {
        let item = Schema::object()
            .field("id", Schema::number().integer().build(), true)
            .field("tags", Schema::array(Schema::string().build()).min_items(1).build(), true)
            .build();
        let s = Schema::array(item).build();
        let good = json!([{"id": 1, "tags": ["a"]}, {"id": 2, "tags": ["b", "c"]}]);
        assert!(validate(&s, &good).is_ok());
        let bad = json!([{"id": 1.5, "tags": []}]);
        let errs = validate(&s, &bad).unwrap_err();
        assert!(errs.len() >= 2); // integer + min_items
    }
}
