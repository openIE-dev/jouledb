//! JSON Schema validation engine (subset of Draft 7+): validate `serde_json::Value`
//! against a schema definition with type checks, required fields, min/max,
//! minLength/maxLength, pattern, enum, nested objects, arrays with items,
//! allOf / anyOf / oneOf combinators, and custom validators.

use serde_json::Value;
use std::collections::HashMap;

// ── Schema Types ────────────────────────────────────────────────

/// Supported JSON Schema types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

// ── Custom Validator ────────────────────────────────────────────

/// A named custom validator: `fn(&Value) -> Option<String>` where
/// `None` means valid and `Some(msg)` means invalid.
pub struct CustomValidator {
    pub name: String,
    pub validate_fn: Box<dyn Fn(&Value) -> Option<String>>,
}

impl std::fmt::Debug for CustomValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomValidator").field("name", &self.name).finish()
    }
}

impl Clone for CustomValidator {
    fn clone(&self) -> Self {
        // Custom validators are not cloneable — produce a no-op placeholder.
        Self {
            name: self.name.clone(),
            validate_fn: Box::new(|_| None),
        }
    }
}

// ── Schema Definition ───────────────────────────────────────────

/// A JSON Schema definition (subset of Draft 7+).
#[derive(Debug, Clone)]
pub struct Schema {
    pub schema_type: Option<SchemaType>,
    pub required: Vec<String>,
    pub properties: HashMap<String, Schema>,
    pub items: Option<Box<Schema>>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub exclusive_minimum: Option<f64>,
    pub exclusive_maximum: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub enum_values: Vec<Value>,
    pub min_items: Option<usize>,
    pub max_items: Option<usize>,
    pub unique_items: bool,
    /// allOf: value must satisfy ALL schemas.
    pub all_of: Vec<Schema>,
    /// anyOf: value must satisfy AT LEAST ONE schema.
    pub any_of: Vec<Schema>,
    /// oneOf: value must satisfy EXACTLY ONE schema.
    pub one_of: Vec<Schema>,
    /// not: value must NOT satisfy this schema.
    pub not: Option<Box<Schema>>,
    /// Custom validators.
    pub custom_validators: Vec<CustomValidator>,
    /// Additional properties allowed?
    pub additional_properties: Option<bool>,
    /// Description (documentation only).
    pub description: Option<String>,
}

impl Schema {
    pub fn new() -> Self {
        Self {
            schema_type: None,
            required: Vec::new(),
            properties: HashMap::new(),
            items: None,
            minimum: None,
            maximum: None,
            exclusive_minimum: None,
            exclusive_maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            enum_values: Vec::new(),
            min_items: None,
            max_items: None,
            unique_items: false,
            all_of: Vec::new(),
            any_of: Vec::new(),
            one_of: Vec::new(),
            not: None,
            custom_validators: Vec::new(),
            additional_properties: None,
            description: None,
        }
    }

    pub fn typed(t: SchemaType) -> Self {
        let mut s = Self::new();
        s.schema_type = Some(t);
        s
    }

    pub fn string() -> Self { Self::typed(SchemaType::String) }
    pub fn number() -> Self { Self::typed(SchemaType::Number) }
    pub fn integer() -> Self { Self::typed(SchemaType::Integer) }
    pub fn boolean() -> Self { Self::typed(SchemaType::Boolean) }
    pub fn null() -> Self { Self::typed(SchemaType::Null) }

    pub fn array(items: Schema) -> Self {
        let mut s = Self::typed(SchemaType::Array);
        s.items = Some(Box::new(items));
        s
    }

    pub fn object() -> Self { Self::typed(SchemaType::Object) }

    // ── Builders ────────────────────────────────────────────────

    pub fn with_required(mut self, fields: &[&str]) -> Self {
        self.required = fields.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_property(mut self, name: &str, schema: Schema) -> Self {
        self.properties.insert(name.to_string(), schema);
        self
    }

    pub fn with_minimum(mut self, min: f64) -> Self {
        self.minimum = Some(min);
        self
    }

    pub fn with_maximum(mut self, max: f64) -> Self {
        self.maximum = Some(max);
        self
    }

    pub fn with_exclusive_minimum(mut self, min: f64) -> Self {
        self.exclusive_minimum = Some(min);
        self
    }

    pub fn with_exclusive_maximum(mut self, max: f64) -> Self {
        self.exclusive_maximum = Some(max);
        self
    }

    pub fn with_min_length(mut self, min: usize) -> Self {
        self.min_length = Some(min);
        self
    }

    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    pub fn with_pattern(mut self, pat: &str) -> Self {
        self.pattern = Some(pat.to_string());
        self
    }

    pub fn with_enum(mut self, values: Vec<Value>) -> Self {
        self.enum_values = values;
        self
    }

    pub fn with_min_items(mut self, min: usize) -> Self {
        self.min_items = Some(min);
        self
    }

    pub fn with_max_items(mut self, max: usize) -> Self {
        self.max_items = Some(max);
        self
    }

    pub fn with_unique_items(mut self) -> Self {
        self.unique_items = true;
        self
    }

    pub fn with_all_of(mut self, schemas: Vec<Schema>) -> Self {
        self.all_of = schemas;
        self
    }

    pub fn with_any_of(mut self, schemas: Vec<Schema>) -> Self {
        self.any_of = schemas;
        self
    }

    pub fn with_one_of(mut self, schemas: Vec<Schema>) -> Self {
        self.one_of = schemas;
        self
    }

    pub fn with_not(mut self, schema: Schema) -> Self {
        self.not = Some(Box::new(schema));
        self
    }

    pub fn with_custom(mut self, name: impl Into<String>, f: impl Fn(&Value) -> Option<String> + 'static) -> Self {
        self.custom_validators.push(CustomValidator {
            name: name.into(),
            validate_fn: Box::new(f),
        });
        self
    }

    pub fn with_no_additional_properties(mut self) -> Self {
        self.additional_properties = Some(false);
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

impl Default for Schema {
    fn default() -> Self { Self::new() }
}

// ── Validation Error ────────────────────────────────────────────

/// A single validation error at a specific JSON path.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl ValidationError {
    fn new(path: &str, message: impl Into<String>) -> Self {
        Self {
            path: path.to_string(),
            message: message.into(),
        }
    }
}

// ── Validation Result ───────────────────────────────────────────

/// Result of validating a value against a schema.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ── Validate ────────────────────────────────────────────────────

/// Validate a `serde_json::Value` against a `Schema`, returning all errors.
pub fn validate(value: &Value, schema: &Schema) -> ValidationResult {
    let mut errors = Vec::new();
    validate_inner(value, schema, "$", &mut errors);
    ValidationResult { errors }
}

fn validate_inner(value: &Value, schema: &Schema, path: &str, errors: &mut Vec<ValidationError>) {
    // Enum constraint.
    if !schema.enum_values.is_empty() && !schema.enum_values.contains(value) {
        errors.push(ValidationError::new(
            path,
            format!("value not in enum: {:?}", schema.enum_values),
        ));
    }

    // Type check.
    if let Some(expected_type) = &schema.schema_type {
        let type_ok = match expected_type {
            SchemaType::String => value.is_string(),
            SchemaType::Number => value.is_number(),
            SchemaType::Integer => value.is_i64() || value.is_u64(),
            SchemaType::Boolean => value.is_boolean(),
            SchemaType::Array => value.is_array(),
            SchemaType::Object => value.is_object(),
            SchemaType::Null => value.is_null(),
        };
        if !type_ok {
            errors.push(ValidationError::new(
                path,
                format!("expected type {:?}, got {:?}", expected_type, value_type_name(value)),
            ));
            return;
        }
    }

    // String constraints.
    if let Some(s) = value.as_str() {
        let char_count = s.chars().count();
        if let Some(min) = schema.min_length {
            if char_count < min {
                errors.push(ValidationError::new(
                    path,
                    format!("string length {} < minimum {}", char_count, min),
                ));
            }
        }
        if let Some(max) = schema.max_length {
            if char_count > max {
                errors.push(ValidationError::new(
                    path,
                    format!("string length {} > maximum {}", char_count, max),
                ));
            }
        }
        if let Some(pat) = &schema.pattern {
            if !simple_pattern_match(pat, s) {
                errors.push(ValidationError::new(
                    path,
                    format!("string does not match pattern '{}'", pat),
                ));
            }
        }
    }

    // Number constraints.
    if let Some(n) = value.as_f64() {
        if let Some(min) = schema.minimum {
            if n < min {
                errors.push(ValidationError::new(path, format!("value {} < minimum {}", n, min)));
            }
        }
        if let Some(max) = schema.maximum {
            if n > max {
                errors.push(ValidationError::new(path, format!("value {} > maximum {}", n, max)));
            }
        }
        if let Some(emin) = schema.exclusive_minimum {
            if n <= emin {
                errors.push(ValidationError::new(
                    path,
                    format!("value {} <= exclusive minimum {}", n, emin),
                ));
            }
        }
        if let Some(emax) = schema.exclusive_maximum {
            if n >= emax {
                errors.push(ValidationError::new(
                    path,
                    format!("value {} >= exclusive maximum {}", n, emax),
                ));
            }
        }
    }

    // Array constraints.
    if let Some(arr) = value.as_array() {
        if let Some(min) = schema.min_items {
            if arr.len() < min {
                errors.push(ValidationError::new(
                    path,
                    format!("array length {} < minItems {}", arr.len(), min),
                ));
            }
        }
        if let Some(max) = schema.max_items {
            if arr.len() > max {
                errors.push(ValidationError::new(
                    path,
                    format!("array length {} > maxItems {}", arr.len(), max),
                ));
            }
        }
        if schema.unique_items {
            let mut seen = Vec::new();
            for (i, item) in arr.iter().enumerate() {
                if seen.contains(item) {
                    errors.push(ValidationError::new(
                        path,
                        format!("duplicate item at index {}", i),
                    ));
                } else {
                    seen.push(item.clone());
                }
            }
        }
        if let Some(items_schema) = &schema.items {
            for (i, item) in arr.iter().enumerate() {
                let item_path = format!("{}[{}]", path, i);
                validate_inner(item, items_schema, &item_path, errors);
            }
        }
    }

    // Object constraints.
    if let Some(obj) = value.as_object() {
        for req in &schema.required {
            if !obj.contains_key(req) {
                errors.push(ValidationError::new(
                    path,
                    format!("missing required field '{}'", req),
                ));
            }
        }
        for (key, prop_schema) in &schema.properties {
            if let Some(prop_value) = obj.get(key) {
                let prop_path = format!("{}.{}", path, key);
                validate_inner(prop_value, prop_schema, &prop_path, errors);
            }
        }
        if schema.additional_properties == Some(false) {
            for key in obj.keys() {
                if !schema.properties.contains_key(key) {
                    errors.push(ValidationError::new(
                        path,
                        format!("additional property '{}' not allowed", key),
                    ));
                }
            }
        }
    }

    // allOf.
    for (i, sub) in schema.all_of.iter().enumerate() {
        let mut sub_errors = Vec::new();
        validate_inner(value, sub, path, &mut sub_errors);
        if !sub_errors.is_empty() {
            errors.push(ValidationError::new(
                path,
                format!("allOf[{}] failed: {}", i, sub_errors[0].message),
            ));
        }
    }

    // anyOf.
    if !schema.any_of.is_empty() {
        let any_match = schema.any_of.iter().any(|sub| {
            let mut sub_errors = Vec::new();
            validate_inner(value, sub, path, &mut sub_errors);
            sub_errors.is_empty()
        });
        if !any_match {
            errors.push(ValidationError::new(
                path,
                "value does not match any of the anyOf schemas",
            ));
        }
    }

    // oneOf.
    if !schema.one_of.is_empty() {
        let match_count = schema.one_of.iter().filter(|sub| {
            let mut sub_errors = Vec::new();
            validate_inner(value, sub, path, &mut sub_errors);
            sub_errors.is_empty()
        }).count();
        if match_count != 1 {
            errors.push(ValidationError::new(
                path,
                format!("value matches {} of oneOf schemas, expected exactly 1", match_count),
            ));
        }
    }

    // not.
    if let Some(not_schema) = &schema.not {
        let mut sub_errors = Vec::new();
        validate_inner(value, not_schema, path, &mut sub_errors);
        if sub_errors.is_empty() {
            errors.push(ValidationError::new(path, "value must NOT match the 'not' schema"));
        }
    }

    // Custom validators.
    for cv in &schema.custom_validators {
        if let Some(msg) = (cv.validate_fn)(value) {
            errors.push(ValidationError::new(
                path,
                format!("custom '{}': {}", cv.name, msg),
            ));
        }
    }
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Simple pattern matching: `^...$` anchored literal and `.*` wildcard.
fn simple_pattern_match(pattern: &str, value: &str) -> bool {
    let pat = pattern.strip_prefix('^').unwrap_or(pattern);
    let (pat, anchored_end) = if let Some(p) = pat.strip_suffix('$') {
        (p, true)
    } else {
        (pat, false)
    };

    if pat.contains(".*") {
        let parts: Vec<&str> = pat.split(".*").collect();
        let mut remaining = value;
        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() { continue; }
            if i == 0 && pattern.starts_with('^') {
                if !remaining.starts_with(part) { return false; }
                remaining = &remaining[part.len()..];
            } else if let Some(pos) = remaining.find(part) {
                remaining = &remaining[pos + part.len()..];
            } else {
                return false;
            }
        }
        if anchored_end {
            if let Some(last) = parts.last() {
                if !last.is_empty() { return value.ends_with(last); }
            }
        }
        true
    } else if pattern.starts_with('^') && anchored_end {
        value == pat
    } else if pattern.starts_with('^') {
        value.starts_with(pat)
    } else if anchored_end {
        value.ends_with(pat)
    } else {
        value.contains(pat)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_valid_string() {
        let schema = Schema::string();
        assert!(validate(&json!("hello"), &schema).is_valid());
    }

    #[test]
    fn test_wrong_type() {
        let schema = Schema::string();
        let result = validate(&json!(42), &schema);
        assert!(!result.is_valid());
        assert!(result.errors[0].message.contains("expected type"));
    }

    #[test]
    fn test_number_minimum() {
        let schema = Schema::number().with_minimum(10.0);
        assert!(!validate(&json!(5), &schema).is_valid());
    }

    #[test]
    fn test_number_maximum() {
        let schema = Schema::number().with_maximum(100.0);
        assert!(!validate(&json!(200), &schema).is_valid());
    }

    #[test]
    fn test_exclusive_minimum() {
        let schema = Schema::number().with_exclusive_minimum(10.0);
        assert!(!validate(&json!(10), &schema).is_valid());
        assert!(validate(&json!(11), &schema).is_valid());
    }

    #[test]
    fn test_exclusive_maximum() {
        let schema = Schema::number().with_exclusive_maximum(10.0);
        assert!(!validate(&json!(10), &schema).is_valid());
        assert!(validate(&json!(9), &schema).is_valid());
    }

    #[test]
    fn test_string_length() {
        let schema = Schema::string().with_min_length(3).with_max_length(5);
        assert!(validate(&json!("abc"), &schema).is_valid());
        assert!(!validate(&json!("ab"), &schema).is_valid());
        assert!(!validate(&json!("abcdef"), &schema).is_valid());
    }

    #[test]
    fn test_string_pattern() {
        let schema = Schema::string().with_pattern("^hello.*world$");
        assert!(validate(&json!("hello beautiful world"), &schema).is_valid());
        assert!(!validate(&json!("hi world"), &schema).is_valid());
    }

    #[test]
    fn test_enum_values() {
        let schema = Schema::string().with_enum(vec![json!("red"), json!("green"), json!("blue")]);
        assert!(validate(&json!("red"), &schema).is_valid());
        assert!(!validate(&json!("yellow"), &schema).is_valid());
    }

    #[test]
    fn test_object_required() {
        let schema = Schema::object()
            .with_required(&["name", "age"])
            .with_property("name", Schema::string())
            .with_property("age", Schema::integer());
        assert!(validate(&json!({"name": "Alice", "age": 30}), &schema).is_valid());
        let result = validate(&json!({"name": "Alice"}), &schema);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_nested_object() {
        let address = Schema::object()
            .with_required(&["street"])
            .with_property("street", Schema::string());
        let schema = Schema::object()
            .with_required(&["address"])
            .with_property("address", address);
        assert!(validate(&json!({"address": {"street": "123 Main"}}), &schema).is_valid());
        assert!(!validate(&json!({"address": {"zip": "12345"}}), &schema).is_valid());
    }

    #[test]
    fn test_array_items() {
        let schema = Schema::array(Schema::integer());
        assert!(validate(&json!([1, 2, 3]), &schema).is_valid());
        assert!(!validate(&json!([1, "two", 3]), &schema).is_valid());
    }

    #[test]
    fn test_array_min_max_items() {
        let schema = Schema::array(Schema::number()).with_min_items(2).with_max_items(4);
        assert!(validate(&json!([1, 2, 3]), &schema).is_valid());
        assert!(!validate(&json!([1]), &schema).is_valid());
        assert!(!validate(&json!([1, 2, 3, 4, 5]), &schema).is_valid());
    }

    #[test]
    fn test_unique_items() {
        let schema = Schema::array(Schema::integer()).with_unique_items();
        assert!(validate(&json!([1, 2, 3]), &schema).is_valid());
        assert!(!validate(&json!([1, 2, 2]), &schema).is_valid());
    }

    #[test]
    fn test_all_of() {
        let schema = Schema::new().with_all_of(vec![
            Schema::number().with_minimum(0.0),
            Schema::number().with_maximum(100.0),
        ]);
        assert!(validate(&json!(50), &schema).is_valid());
        assert!(!validate(&json!(150), &schema).is_valid());
    }

    #[test]
    fn test_any_of() {
        let schema = Schema::new().with_any_of(vec![
            Schema::string(),
            Schema::integer(),
        ]);
        assert!(validate(&json!("hello"), &schema).is_valid());
        assert!(validate(&json!(42), &schema).is_valid());
        assert!(!validate(&json!(true), &schema).is_valid());
    }

    #[test]
    fn test_one_of() {
        let schema = Schema::new().with_one_of(vec![
            Schema::string().with_min_length(5),
            Schema::string().with_max_length(3),
        ]);
        assert!(validate(&json!("hello world"), &schema).is_valid()); // matches first only
        assert!(validate(&json!("hi"), &schema).is_valid()); // matches second only
        assert!(!validate(&json!("abcd"), &schema).is_valid()); // matches neither
    }

    #[test]
    fn test_not() {
        let schema = Schema::new().with_not(Schema::string());
        assert!(validate(&json!(42), &schema).is_valid());
        assert!(!validate(&json!("hello"), &schema).is_valid());
    }

    #[test]
    fn test_custom_validator() {
        let schema = Schema::integer().with_custom("even", |v| {
            if let Some(n) = v.as_i64() {
                if n % 2 != 0 {
                    return Some("must be even".into());
                }
            }
            None
        });
        assert!(validate(&json!(4), &schema).is_valid());
        assert!(!validate(&json!(3), &schema).is_valid());
    }

    #[test]
    fn test_no_additional_properties() {
        let schema = Schema::object()
            .with_property("name", Schema::string())
            .with_no_additional_properties();
        assert!(validate(&json!({"name": "Alice"}), &schema).is_valid());
        assert!(!validate(&json!({"name": "Alice", "age": 30}), &schema).is_valid());
    }

    #[test]
    fn test_null_type() {
        let schema = Schema::null();
        assert!(validate(&json!(null), &schema).is_valid());
        assert!(!validate(&json!("not null"), &schema).is_valid());
    }

    #[test]
    fn test_boolean_type() {
        let schema = Schema::boolean();
        assert!(validate(&json!(true), &schema).is_valid());
        assert!(!validate(&json!(1), &schema).is_valid());
    }

    #[test]
    fn test_integer_rejects_float() {
        let schema = Schema::integer();
        assert!(validate(&json!(42), &schema).is_valid());
        assert!(!validate(&json!(3.14), &schema).is_valid());
    }

    #[test]
    fn test_multiple_errors_collected() {
        let schema = Schema::object().with_required(&["a", "b", "c"]);
        let result = validate(&json!({}), &schema);
        assert_eq!(result.error_count(), 3);
    }

    #[test]
    fn test_no_schema_type_accepts_anything() {
        let schema = Schema::new();
        assert!(validate(&json!("hello"), &schema).is_valid());
        assert!(validate(&json!(42), &schema).is_valid());
        assert!(validate(&json!(null), &schema).is_valid());
    }

    #[test]
    fn test_error_count() {
        let schema = Schema::object().with_required(&["x", "y"]);
        let result = validate(&json!({}), &schema);
        assert_eq!(result.error_count(), 2);
    }
}
