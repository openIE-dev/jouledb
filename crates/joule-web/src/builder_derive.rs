//! Builder pattern generator (runtime) — field definitions, optional/required
//! fields, default values, validation on build, fluent API, nested builders,
//! builder from existing instance, and builder error reporting.
//!
//! Replaces TypeScript builder patterns, Java's Lombok @Builder, and
//! derive_builder with a pure-Rust runtime builder system.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from builder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuilderError {
    /// A required field was not set.
    MissingField(String),
    /// A field value failed validation.
    ValidationFailed { field: String, reason: String },
    /// A field was set more than once.
    DuplicateField(String),
    /// Unknown field name.
    UnknownField(String),
    /// Type mismatch for a field.
    TypeMismatch { field: String, expected: String, got: String },
    /// Multiple errors accumulated.
    Multiple(Vec<BuilderError>),
}

impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => write!(f, "required field not set: {name}"),
            Self::ValidationFailed { field, reason } => {
                write!(f, "validation failed for {field}: {reason}")
            }
            Self::DuplicateField(name) => write!(f, "field set more than once: {name}"),
            Self::UnknownField(name) => write!(f, "unknown field: {name}"),
            Self::TypeMismatch { field, expected, got } => {
                write!(f, "type mismatch for {field}: expected {expected}, got {got}")
            }
            Self::Multiple(errs) => {
                write!(f, "{} builder errors:", errs.len())?;
                for e in errs {
                    write!(f, "\n  - {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for BuilderError {}

// ── Field Definition ────────────────────────────────────────────

/// Expected type of a field value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    /// Any JSON value.
    Any,
    /// Must be a string.
    Str,
    /// Must be a number.
    Number,
    /// Must be an integer.
    Int,
    /// Must be a bool.
    Bool,
    /// Must be an array.
    Array,
    /// Must be an object.
    Object,
}

impl FieldType {
    /// Validate that a JSON value matches this type.
    pub fn validate(&self, value: &Value) -> bool {
        match self {
            Self::Any => true,
            Self::Str => value.is_string(),
            Self::Number => value.is_number(),
            Self::Int => value.is_i64() || value.is_u64(),
            Self::Bool => value.is_boolean(),
            Self::Array => value.is_array(),
            Self::Object => value.is_object(),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            Self::Any => "any",
            Self::Str => "string",
            Self::Number => "number",
            Self::Int => "integer",
            Self::Bool => "boolean",
            Self::Array => "array",
            Self::Object => "object",
        }
    }
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Defines a field in a builder.
#[derive(Debug, Clone)]
pub struct FieldDef {
    /// Field name.
    pub name: String,
    /// Expected type.
    pub field_type: FieldType,
    /// Whether the field is required.
    pub required: bool,
    /// Default value (if any).
    pub default: Option<Value>,
    /// Description.
    pub description: Option<String>,
    /// Custom validator function pointer.
    validator: Option<fn(&Value) -> Result<(), String>>,
}

impl FieldDef {
    /// Create a required field definition.
    pub fn required(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: true,
            default: None,
            description: None,
            validator: None,
        }
    }

    /// Create an optional field definition.
    pub fn optional(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: false,
            default: None,
            description: None,
            validator: None,
        }
    }

    /// Set a default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self.required = false;
        self
    }

    /// Set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set a custom validator.
    pub fn with_validator(mut self, validator: fn(&Value) -> Result<(), String>) -> Self {
        self.validator = Some(validator);
        self
    }
}

// ── Builder Schema ──────────────────────────────────────────────

/// Schema for a builder: defines what fields it has and how to build.
#[derive(Debug, Clone)]
pub struct BuilderSchema {
    /// Name of the type being built.
    pub name: String,
    /// Field definitions, in order.
    pub fields: Vec<FieldDef>,
    /// Whether to reject unknown fields.
    pub strict: bool,
    /// Global validator run after all fields are set.
    global_validator: Option<fn(&HashMap<String, Value>) -> Result<(), String>>,
}

impl BuilderSchema {
    /// Create a new builder schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            strict: true,
            global_validator: None,
        }
    }

    /// Add a field definition.
    pub fn with_field(mut self, field: FieldDef) -> Self {
        self.fields.push(field);
        self
    }

    /// Allow unknown fields (not strict).
    pub fn allow_unknown(mut self) -> Self {
        self.strict = false;
        self
    }

    /// Set a global validator.
    pub fn with_global_validator(
        mut self,
        validator: fn(&HashMap<String, Value>) -> Result<(), String>,
    ) -> Self {
        self.global_validator = Some(validator);
        self
    }

    /// Get a field definition by name.
    pub fn get_field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Required field names.
    pub fn required_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.required)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Create a builder instance from this schema.
    pub fn builder(&self) -> Builder<'_> {
        Builder::new(self)
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// A runtime builder instance. Set fields, then call `build()`.
#[derive(Debug)]
pub struct Builder<'a> {
    schema: &'a BuilderSchema,
    values: HashMap<String, Value>,
    errors: Vec<BuilderError>,
}

impl<'a> Builder<'a> {
    /// Create a builder from a schema.
    pub fn new(schema: &'a BuilderSchema) -> Self {
        Self {
            schema,
            values: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Create a builder pre-populated from an existing map (for "builder from instance").
    pub fn from_values(schema: &'a BuilderSchema, values: HashMap<String, Value>) -> Self {
        Self {
            schema,
            values,
            errors: Vec::new(),
        }
    }

    /// Set a field value (fluent API).
    pub fn set(mut self, field: impl Into<String>, value: Value) -> Self {
        let name = field.into();
        self.set_mut(&name, value);
        self
    }

    /// Set a string field (convenience).
    pub fn set_str(self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(field, Value::String(value.into()))
    }

    /// Set an integer field (convenience).
    pub fn set_int(self, field: impl Into<String>, value: i64) -> Self {
        self.set(field, Value::Number(value.into()))
    }

    /// Set a bool field (convenience).
    pub fn set_bool(self, field: impl Into<String>, value: bool) -> Self {
        self.set(field, Value::Bool(value))
    }

    /// Mutably set a field value.
    fn set_mut(&mut self, name: &str, value: Value) {
        // Check if field is known.
        if let Some(def) = self.schema.get_field(name) {
            // Type check.
            if !def.field_type.validate(&value) {
                self.errors.push(BuilderError::TypeMismatch {
                    field: name.to_string(),
                    expected: def.field_type.name().to_string(),
                    got: json_type_name(&value),
                });
                return;
            }
            // Custom validator.
            if let Some(validator) = def.validator {
                if let Err(reason) = validator(&value) {
                    self.errors.push(BuilderError::ValidationFailed {
                        field: name.to_string(),
                        reason,
                    });
                    return;
                }
            }
        } else if self.schema.strict {
            self.errors.push(BuilderError::UnknownField(name.to_string()));
            return;
        }
        self.values.insert(name.to_string(), value);
    }

    /// Check if a field has been set.
    pub fn has_field(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Get a field value.
    pub fn get_field(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    /// Clear a previously set field.
    pub fn clear_field(mut self, name: impl Into<String>) -> Self {
        self.values.remove(&name.into());
        self
    }

    /// Whether there are accumulated errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Build the final value map.
    pub fn build(mut self) -> Result<HashMap<String, Value>, BuilderError> {
        // Apply defaults for unset fields.
        for def in &self.schema.fields {
            if !self.values.contains_key(&def.name) {
                if let Some(default) = &def.default {
                    self.values.insert(def.name.clone(), default.clone());
                }
            }
        }

        // Check required fields.
        for def in &self.schema.fields {
            if def.required && !self.values.contains_key(&def.name) {
                self.errors.push(BuilderError::MissingField(def.name.clone()));
            }
        }

        // Run global validator.
        if let Some(validator) = self.schema.global_validator {
            if let Err(reason) = validator(&self.values) {
                self.errors.push(BuilderError::ValidationFailed {
                    field: "<global>".to_string(),
                    reason,
                });
            }
        }

        if self.errors.is_empty() {
            Ok(self.values)
        } else if self.errors.len() == 1 {
            Err(self.errors.into_iter().next().unwrap())
        } else {
            Err(BuilderError::Multiple(self.errors))
        }
    }

    /// Build as a JSON Value (object).
    pub fn build_json(self) -> Result<Value, BuilderError> {
        let map = self.build()?;
        let obj: serde_json::Map<String, Value> = map.into_iter().collect();
        Ok(Value::Object(obj))
    }
}

fn json_type_name(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "boolean".to_string(),
        Value::Number(_) => "number".to_string(),
        Value::String(_) => "string".to_string(),
        Value::Array(_) => "array".to_string(),
        Value::Object(_) => "object".to_string(),
    }
}

// ── Nested Builder ──────────────────────────────────────────────

/// A nested builder helper: build a sub-object, then inject into a parent builder.
pub struct NestedBuilder<'a> {
    parent_field: String,
    child_schema: &'a BuilderSchema,
}

impl<'a> NestedBuilder<'a> {
    /// Create a nested builder.
    pub fn new(parent_field: impl Into<String>, child_schema: &'a BuilderSchema) -> Self {
        Self {
            parent_field: parent_field.into(),
            child_schema,
        }
    }

    /// Build the nested object and return as (field_name, Value) pair.
    pub fn build(
        &self,
        configure: impl FnOnce(Builder<'_>) -> Builder<'_>,
    ) -> Result<(String, Value), BuilderError> {
        let builder = Builder::new(self.child_schema);
        let configured = configure(builder);
        let json = configured.build_json()?;
        Ok((self.parent_field.clone(), json))
    }
}

// ── Quick Builder ───────────────────────────────────────────────

/// A quick builder for simple cases: no schema, just collect key-value pairs.
#[derive(Debug, Default)]
pub struct QuickBuilder {
    values: HashMap<String, Value>,
}

impl QuickBuilder {
    /// Create a new quick builder.
    pub fn new() -> Self {
        Self { values: HashMap::new() }
    }

    /// Set a value.
    pub fn set(mut self, key: impl Into<String>, value: Value) -> Self {
        self.values.insert(key.into(), value);
        self
    }

    /// Set a string value.
    pub fn set_str(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, Value::String(value.into()))
    }

    /// Set an integer value.
    pub fn set_int(self, key: impl Into<String>, value: i64) -> Self {
        self.set(key, Value::Number(value.into()))
    }

    /// Set a bool value.
    pub fn set_bool(self, key: impl Into<String>, value: bool) -> Self {
        self.set(key, Value::Bool(value))
    }

    /// Build into a map.
    pub fn build(self) -> HashMap<String, Value> {
        self.values
    }

    /// Build into a JSON object.
    pub fn build_json(self) -> Value {
        let obj: serde_json::Map<String, Value> = self.values.into_iter().collect();
        Value::Object(obj)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_schema() -> BuilderSchema {
        BuilderSchema::new("User")
            .with_field(FieldDef::required("name", FieldType::Str))
            .with_field(
                FieldDef::optional("age", FieldType::Int)
                    .with_default(json!(0)),
            )
            .with_field(FieldDef::optional("email", FieldType::Str))
            .with_field(
                FieldDef::optional("active", FieldType::Bool)
                    .with_default(json!(true)),
            )
    }

    #[test]
    fn test_build_ok() {
        let schema = user_schema();
        let result = schema
            .builder()
            .set_str("name", "Alice")
            .set_int("age", 30)
            .build();
        let map = result.unwrap();
        assert_eq!(map["name"], json!("Alice"));
        assert_eq!(map["age"], json!(30));
        assert_eq!(map["active"], json!(true)); // default
    }

    #[test]
    fn test_build_missing_required() {
        let schema = user_schema();
        let result = schema.builder().set_int("age", 30).build();
        assert!(result.is_err());
        match result.unwrap_err() {
            BuilderError::MissingField(f) => assert_eq!(f, "name"),
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn test_build_type_mismatch() {
        let schema = user_schema();
        let result = schema.builder().set("name", json!(42)).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_unknown_field_strict() {
        let schema = user_schema();
        let result = schema
            .builder()
            .set_str("name", "Alice")
            .set_str("unknown", "val")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_unknown_field_lenient() {
        let schema = user_schema().allow_unknown();
        let result = schema
            .builder()
            .set_str("name", "Alice")
            .set_str("unknown", "val")
            .build();
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["unknown"], json!("val"));
    }

    #[test]
    fn test_build_defaults_applied() {
        let schema = user_schema();
        let result = schema.builder().set_str("name", "Alice").build().unwrap();
        assert_eq!(result["age"], json!(0));
        assert_eq!(result["active"], json!(true));
    }

    #[test]
    fn test_build_json() {
        let schema = user_schema();
        let result = schema
            .builder()
            .set_str("name", "Alice")
            .build_json()
            .unwrap();
        assert!(result.is_object());
        assert_eq!(result["name"], "Alice");
    }

    #[test]
    fn test_custom_validator() {
        fn validate_age(v: &Value) -> Result<(), String> {
            if let Some(n) = v.as_i64() {
                if n >= 0 && n <= 150 {
                    return Ok(());
                }
            }
            Err("age must be 0-150".to_string())
        }

        let schema = BuilderSchema::new("User")
            .with_field(FieldDef::required("name", FieldType::Str))
            .with_field(
                FieldDef::optional("age", FieldType::Int).with_validator(validate_age),
            );

        let result = schema
            .builder()
            .set_str("name", "Alice")
            .set_int("age", 200)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_global_validator() {
        fn check_consistency(map: &HashMap<String, Value>) -> Result<(), String> {
            if let (Some(min), Some(max)) = (map.get("min"), map.get("max")) {
                if let (Some(mn), Some(mx)) = (min.as_i64(), max.as_i64()) {
                    if mn > mx {
                        return Err("min must be <= max".to_string());
                    }
                }
            }
            Ok(())
        }

        let schema = BuilderSchema::new("Range")
            .with_field(FieldDef::required("min", FieldType::Int))
            .with_field(FieldDef::required("max", FieldType::Int))
            .with_global_validator(check_consistency);

        let result = schema.builder().set_int("min", 10).set_int("max", 5).build();
        assert!(result.is_err());

        let result = schema.builder().set_int("min", 1).set_int("max", 10).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_from_values() {
        let schema = user_schema();
        let mut existing = HashMap::new();
        existing.insert("name".to_string(), json!("Bob"));
        existing.insert("age".to_string(), json!(25));

        let result = Builder::from_values(&schema, existing)
            .set_str("email", "bob@example.com")
            .build()
            .unwrap();
        assert_eq!(result["name"], json!("Bob"));
        assert_eq!(result["email"], json!("bob@example.com"));
    }

    #[test]
    fn test_builder_clear_field() {
        let schema = user_schema();
        let result = schema
            .builder()
            .set_str("name", "Alice")
            .set_int("age", 30)
            .clear_field("age")
            .build()
            .unwrap();
        // age should use default (0) since it was cleared.
        assert_eq!(result["age"], json!(0));
    }

    #[test]
    fn test_builder_has_field() {
        let schema = user_schema();
        let builder = schema.builder().set_str("name", "Alice");
        assert!(builder.has_field("name"));
        assert!(!builder.has_field("age"));
    }

    #[test]
    fn test_builder_get_field() {
        let schema = user_schema();
        let builder = schema.builder().set_str("name", "Alice");
        assert_eq!(builder.get_field("name"), Some(&json!("Alice")));
        assert_eq!(builder.get_field("age"), None);
    }

    #[test]
    fn test_nested_builder() {
        let addr_schema = BuilderSchema::new("Address")
            .with_field(FieldDef::required("street", FieldType::Str))
            .with_field(FieldDef::required("city", FieldType::Str));

        let nested = NestedBuilder::new("address", &addr_schema);
        let (field, value) = nested
            .build(|b| b.set_str("street", "123 Main").set_str("city", "NYC"))
            .unwrap();
        assert_eq!(field, "address");
        assert_eq!(value["street"], "123 Main");
        assert_eq!(value["city"], "NYC");
    }

    #[test]
    fn test_quick_builder() {
        let result = QuickBuilder::new()
            .set_str("name", "Alice")
            .set_int("age", 30)
            .set_bool("active", true)
            .build_json();
        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[test]
    fn test_multiple_errors() {
        let schema = BuilderSchema::new("Strict")
            .with_field(FieldDef::required("a", FieldType::Str))
            .with_field(FieldDef::required("b", FieldType::Int));

        let result = schema.builder().build();
        match result.unwrap_err() {
            BuilderError::Multiple(errs) => assert_eq!(errs.len(), 2),
            other => panic!("expected Multiple, got {other}"),
        }
    }

    #[test]
    fn test_required_fields() {
        let schema = user_schema();
        assert_eq!(schema.required_fields(), vec!["name"]);
    }

    #[test]
    fn test_field_type_validation() {
        assert!(FieldType::Str.validate(&json!("hello")));
        assert!(!FieldType::Str.validate(&json!(42)));
        assert!(FieldType::Int.validate(&json!(42)));
        assert!(!FieldType::Int.validate(&json!(3.14)));
        assert!(FieldType::Bool.validate(&json!(true)));
        assert!(FieldType::Any.validate(&json!(null)));
    }

    #[test]
    fn test_field_description() {
        let f = FieldDef::required("name", FieldType::Str)
            .with_description("User's full name");
        assert_eq!(f.description.as_deref(), Some("User's full name"));
    }

    #[test]
    fn test_builder_error_display() {
        let e = BuilderError::MissingField("name".to_string());
        assert_eq!(e.to_string(), "required field not set: name");
    }

    #[test]
    fn test_quick_builder_build_map() {
        let map = QuickBuilder::new().set_str("x", "y").build();
        assert_eq!(map["x"], json!("y"));
    }
}
