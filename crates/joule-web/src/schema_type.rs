//! Schema type system — primitive types, struct/enum/array/map/optional types,
//! type validation, schema compatibility checking, schema evolution rules,
//! schema-to-JSON-Schema conversion, and type coercion rules.
//!
//! Replaces JSON Schema generators, Zod, Yup, io-ts, and similar TypeScript
//! type validation libraries with a pure-Rust schema type system.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from schema type operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaTypeError {
    /// Value does not conform to schema.
    ValidationFailed { path: String, expected: String, got: String },
    /// Schema compatibility check failed.
    IncompatibleSchema { reason: String },
    /// Unknown type reference.
    UnknownType(String),
    /// Coercion not possible.
    CoercionFailed { from: String, to: String },
    /// Invalid schema definition.
    InvalidSchema(String),
}

impl fmt::Display for SchemaTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValidationFailed { path, expected, got } => {
                write!(f, "validation failed at {path}: expected {expected}, got {got}")
            }
            Self::IncompatibleSchema { reason } => write!(f, "incompatible schema: {reason}"),
            Self::UnknownType(name) => write!(f, "unknown type: {name}"),
            Self::CoercionFailed { from, to } => write!(f, "cannot coerce {from} to {to}"),
            Self::InvalidSchema(msg) => write!(f, "invalid schema: {msg}"),
        }
    }
}

impl std::error::Error for SchemaTypeError {}

// ── Schema Type ─────────────────────────────────────────────────

/// A schema type node in the type system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaType {
    /// Null type.
    Null,
    /// Boolean type.
    Bool,
    /// Integer type (i64).
    Int,
    /// Floating point type (f64).
    Float,
    /// String type.
    Str,
    /// An array of a given element type.
    Array(Box<SchemaType>),
    /// A map from string keys to a given value type.
    Map(Box<SchemaType>),
    /// An optional (nullable) type.
    Optional(Box<SchemaType>),
    /// A struct with named, typed fields.
    Struct(StructSchema),
    /// An enum with named variants, each optionally carrying data.
    Enum(EnumSchema),
    /// A reference to a named type (for recursive/shared schemas).
    Ref(String),
    /// A union of types (any of).
    Union(Vec<SchemaType>),
    /// Constant literal value.
    Const(Value),
    /// Any type (accepts everything).
    Any,
}

impl SchemaType {
    /// Human-readable type name.
    pub fn type_name(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool => "bool".to_string(),
            Self::Int => "int".to_string(),
            Self::Float => "float".to_string(),
            Self::Str => "string".to_string(),
            Self::Array(inner) => format!("array<{}>", inner.type_name()),
            Self::Map(inner) => format!("map<string, {}>", inner.type_name()),
            Self::Optional(inner) => format!("optional<{}>", inner.type_name()),
            Self::Struct(s) => s.name.clone(),
            Self::Enum(e) => e.name.clone(),
            Self::Ref(name) => format!("ref({name})"),
            Self::Union(types) => {
                let names: Vec<String> = types.iter().map(|t| t.type_name()).collect();
                format!("union<{}>", names.join(" | "))
            }
            Self::Const(v) => format!("const({v})"),
            Self::Any => "any".to_string(),
        }
    }

    /// Whether this is a primitive type.
    pub fn is_primitive(&self) -> bool {
        matches!(self, Self::Null | Self::Bool | Self::Int | Self::Float | Self::Str)
    }

    /// Whether this type allows null values.
    pub fn is_nullable(&self) -> bool {
        matches!(self, Self::Null | Self::Optional(_) | Self::Any)
    }
}

// ── Struct Schema ───────────────────────────────────────────────

/// Schema for a struct type with named fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructSchema {
    /// Type name.
    pub name: String,
    /// Fields in order.
    pub fields: Vec<SchemaField>,
    /// Whether additional properties are allowed.
    pub additional_properties: bool,
}

impl StructSchema {
    /// Create a new struct schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            additional_properties: false,
        }
    }

    /// Add a field.
    pub fn with_field(mut self, field: SchemaField) -> Self {
        self.fields.push(field);
        self
    }

    /// Allow additional properties.
    pub fn allow_additional(mut self) -> Self {
        self.additional_properties = true;
        self
    }

    /// Get a field by name.
    pub fn get_field(&self, name: &str) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Required field names.
    pub fn required_fields(&self) -> Vec<&str> {
        self.fields.iter().filter(|f| f.required).map(|f| f.name.as_str()).collect()
    }
}

/// A field in a struct schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaField {
    /// Field name.
    pub name: String,
    /// Field type.
    pub schema: SchemaType,
    /// Whether the field is required.
    pub required: bool,
    /// Default value.
    pub default: Option<Value>,
    /// Description.
    pub description: Option<String>,
}

impl SchemaField {
    /// Create a required field.
    pub fn required(name: impl Into<String>, schema: SchemaType) -> Self {
        Self {
            name: name.into(),
            schema,
            required: true,
            default: None,
            description: None,
        }
    }

    /// Create an optional field.
    pub fn optional(name: impl Into<String>, schema: SchemaType) -> Self {
        Self {
            name: name.into(),
            schema,
            required: false,
            default: None,
            description: None,
        }
    }

    /// Set default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// ── Enum Schema ─────────────────────────────────────────────────

/// Schema for an enum type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumSchema {
    /// Type name.
    pub name: String,
    /// Variants.
    pub variants: Vec<EnumVariant>,
}

impl EnumSchema {
    /// Create a new enum schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variants: Vec::new(),
        }
    }

    /// Add a variant.
    pub fn with_variant(mut self, variant: EnumVariant) -> Self {
        self.variants.push(variant);
        self
    }

    /// Get variant by name.
    pub fn get_variant(&self, name: &str) -> Option<&EnumVariant> {
        self.variants.iter().find(|v| v.name == name)
    }

    /// Variant names.
    pub fn variant_names(&self) -> Vec<&str> {
        self.variants.iter().map(|v| v.name.as_str()).collect()
    }
}

/// A variant in an enum schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumVariant {
    /// Variant name.
    pub name: String,
    /// Optional payload type.
    pub payload: Option<SchemaType>,
    /// Description.
    pub description: Option<String>,
}

impl EnumVariant {
    /// A unit variant (no payload).
    pub fn unit(name: impl Into<String>) -> Self {
        Self { name: name.into(), payload: None, description: None }
    }

    /// A variant with payload.
    pub fn with_payload(name: impl Into<String>, schema: SchemaType) -> Self {
        Self { name: name.into(), payload: Some(schema), description: None }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// ── Validation ──────────────────────────────────────────────────

/// Validate a JSON value against a schema type.
pub fn validate(schema: &SchemaType, value: &Value) -> Result<(), Vec<SchemaTypeError>> {
    let mut errors = Vec::new();
    validate_inner(schema, value, "$", &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_inner(schema: &SchemaType, value: &Value, path: &str, errors: &mut Vec<SchemaTypeError>) {
    match schema {
        SchemaType::Any => {}
        SchemaType::Null => {
            if !value.is_null() {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "null".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Bool => {
            if !value.is_boolean() {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "bool".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Int => {
            if !value.is_i64() && !value.is_u64() {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "int".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Float => {
            if !value.is_number() {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "float".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Str => {
            if !value.is_string() {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "string".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Array(inner) => {
            if let Some(arr) = value.as_array() {
                for (i, item) in arr.iter().enumerate() {
                    let item_path = format!("{path}[{i}]");
                    validate_inner(inner, item, &item_path, errors);
                }
            } else {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "array".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Map(val_type) => {
            if let Some(obj) = value.as_object() {
                for (k, v) in obj {
                    let kpath = format!("{path}.{k}");
                    validate_inner(val_type, v, &kpath, errors);
                }
            } else {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: "object".to_string(),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Optional(inner) => {
            if !value.is_null() {
                validate_inner(inner, value, path, errors);
            }
        }
        SchemaType::Struct(s) => {
            if let Some(obj) = value.as_object() {
                for field in &s.fields {
                    if let Some(val) = obj.get(&field.name) {
                        let fp = format!("{path}.{}", field.name);
                        validate_inner(&field.schema, val, &fp, errors);
                    } else if field.required {
                        errors.push(SchemaTypeError::ValidationFailed {
                            path: format!("{path}.{}", field.name),
                            expected: field.schema.type_name(),
                            got: "missing".to_string(),
                        });
                    }
                }
                if !s.additional_properties {
                    for key in obj.keys() {
                        if !s.fields.iter().any(|f| f.name == *key) {
                            errors.push(SchemaTypeError::ValidationFailed {
                                path: format!("{path}.{key}"),
                                expected: "no additional properties".to_string(),
                                got: "extra field".to_string(),
                            });
                        }
                    }
                }
            } else {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: format!("object ({})", s.name),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Enum(e) => {
            if let Some(s) = value.as_str() {
                if !e.variants.iter().any(|v| v.name == s && v.payload.is_none()) {
                    errors.push(SchemaTypeError::ValidationFailed {
                        path: path.to_string(),
                        expected: format!("one of {:?}", e.variant_names()),
                        got: s.to_string(),
                    });
                }
            } else if let Some(obj) = value.as_object() {
                if obj.len() != 1 {
                    errors.push(SchemaTypeError::ValidationFailed {
                        path: path.to_string(),
                        expected: "enum object with single key".to_string(),
                        got: format!("object with {} keys", obj.len()),
                    });
                } else {
                    let (key, val) = obj.iter().next().unwrap();
                    if let Some(variant) = e.get_variant(key) {
                        if let Some(payload_schema) = &variant.payload {
                            let vp = format!("{path}.{key}");
                            validate_inner(payload_schema, val, &vp, errors);
                        }
                    } else {
                        errors.push(SchemaTypeError::ValidationFailed {
                            path: path.to_string(),
                            expected: format!("variant of {}", e.name),
                            got: key.clone(),
                        });
                    }
                }
            } else {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: format!("string or object for enum {}", e.name),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Union(types) => {
            let any_valid = types.iter().any(|t| {
                let mut sub = Vec::new();
                validate_inner(t, value, path, &mut sub);
                sub.is_empty()
            });
            if !any_valid {
                let names: Vec<String> = types.iter().map(|t| t.type_name()).collect();
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: format!("one of [{}]", names.join(", ")),
                    got: json_type_name(value),
                });
            }
        }
        SchemaType::Const(expected) => {
            if value != expected {
                errors.push(SchemaTypeError::ValidationFailed {
                    path: path.to_string(),
                    expected: format!("const({expected})"),
                    got: value.to_string(),
                });
            }
        }
        SchemaType::Ref(_name) => {
            // Ref resolution is done by callers who own a type environment.
            // Standalone validation treats Ref as Any.
        }
    }
}

fn json_type_name(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "bool".to_string(),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int".to_string()
            } else {
                "float".to_string()
            }
        }
        Value::String(_) => "string".to_string(),
        Value::Array(_) => "array".to_string(),
        Value::Object(_) => "object".to_string(),
    }
}

// ── Compatibility ───────────────────────────────────────────────

/// Check whether a `writer` schema is compatible with a `reader` schema.
/// Uses Avro-style compatibility: the reader can read data written by the writer.
pub fn is_compatible(reader: &SchemaType, writer: &SchemaType) -> bool {
    match (reader, writer) {
        (SchemaType::Any, _) | (_, SchemaType::Any) => true,
        (SchemaType::Null, SchemaType::Null) => true,
        (SchemaType::Bool, SchemaType::Bool) => true,
        (SchemaType::Int, SchemaType::Int) => true,
        (SchemaType::Float, SchemaType::Float) => true,
        (SchemaType::Float, SchemaType::Int) => true, // int -> float promotion
        (SchemaType::Str, SchemaType::Str) => true,
        (SchemaType::Array(r), SchemaType::Array(w)) => is_compatible(r, w),
        (SchemaType::Map(r), SchemaType::Map(w)) => is_compatible(r, w),
        (SchemaType::Optional(r), SchemaType::Optional(w)) => is_compatible(r, w),
        (SchemaType::Optional(r), w) => is_compatible(r, w), // non-nullable -> optional OK
        (SchemaType::Struct(r), SchemaType::Struct(w)) => {
            // Every required reader field must be present in writer.
            for rf in &r.fields {
                if rf.required {
                    match w.get_field(&rf.name) {
                        Some(wf) => {
                            if !is_compatible(&rf.schema, &wf.schema) {
                                return false;
                            }
                        }
                        None => {
                            if rf.default.is_none() {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        }
        (SchemaType::Enum(r), SchemaType::Enum(w)) => {
            // Writer variants must be a subset of reader variants.
            w.variants.iter().all(|wv| r.variants.iter().any(|rv| rv.name == wv.name))
        }
        (SchemaType::Union(r_types), w) => {
            // Writer must be compatible with at least one union branch.
            r_types.iter().any(|rt| is_compatible(rt, w))
        }
        (r, SchemaType::Union(w_types)) => {
            // Every writer branch must be compatible with reader.
            w_types.iter().all(|wt| is_compatible(r, wt))
        }
        _ => false,
    }
}

// ── Coercion ────────────────────────────────────────────────────

/// Attempt to coerce a value to match a target schema type.
pub fn coerce(value: &Value, target: &SchemaType) -> Result<Value, SchemaTypeError> {
    match target {
        SchemaType::Any => Ok(value.clone()),
        SchemaType::Null => {
            if value.is_null() {
                Ok(Value::Null)
            } else {
                Err(SchemaTypeError::CoercionFailed {
                    from: json_type_name(value),
                    to: "null".to_string(),
                })
            }
        }
        SchemaType::Bool => match value {
            Value::Bool(_) => Ok(value.clone()),
            Value::String(s) => match s.as_str() {
                "true" | "1" | "yes" => Ok(Value::Bool(true)),
                "false" | "0" | "no" => Ok(Value::Bool(false)),
                _ => Err(SchemaTypeError::CoercionFailed {
                    from: "string".to_string(),
                    to: "bool".to_string(),
                }),
            },
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Value::Bool(i != 0))
                } else {
                    Err(SchemaTypeError::CoercionFailed {
                        from: "number".to_string(),
                        to: "bool".to_string(),
                    })
                }
            }
            _ => Err(SchemaTypeError::CoercionFailed {
                from: json_type_name(value),
                to: "bool".to_string(),
            }),
        },
        SchemaType::Int => match value {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Value::Number(i.into()))
                } else if let Some(f) = n.as_f64() {
                    Ok(Value::Number((f as i64).into()))
                } else {
                    Err(SchemaTypeError::CoercionFailed {
                        from: "number".to_string(),
                        to: "int".to_string(),
                    })
                }
            }
            Value::String(s) => s
                .parse::<i64>()
                .map(|i| Value::Number(i.into()))
                .map_err(|_| SchemaTypeError::CoercionFailed {
                    from: "string".to_string(),
                    to: "int".to_string(),
                }),
            Value::Bool(b) => Ok(Value::Number(if *b { 1 } else { 0 }.into())),
            _ => Err(SchemaTypeError::CoercionFailed {
                from: json_type_name(value),
                to: "int".to_string(),
            }),
        },
        SchemaType::Float => match value {
            Value::Number(_) => Ok(value.clone()),
            Value::String(s) => s
                .parse::<f64>()
                .ok()
                .and_then(|f| serde_json::Number::from_f64(f).map(Value::Number))
                .ok_or_else(|| SchemaTypeError::CoercionFailed {
                    from: "string".to_string(),
                    to: "float".to_string(),
                }),
            _ => Err(SchemaTypeError::CoercionFailed {
                from: json_type_name(value),
                to: "float".to_string(),
            }),
        },
        SchemaType::Str => match value {
            Value::String(_) => Ok(value.clone()),
            Value::Number(n) => Ok(Value::String(n.to_string())),
            Value::Bool(b) => Ok(Value::String(b.to_string())),
            Value::Null => Ok(Value::String("null".to_string())),
            _ => Err(SchemaTypeError::CoercionFailed {
                from: json_type_name(value),
                to: "string".to_string(),
            }),
        },
        SchemaType::Optional(inner) => {
            if value.is_null() {
                Ok(Value::Null)
            } else {
                coerce(value, inner)
            }
        }
        _ => {
            // For complex types, coercion is identity-or-fail.
            let mut errs = Vec::new();
            validate_inner(target, value, "$", &mut errs);
            if errs.is_empty() {
                Ok(value.clone())
            } else {
                Err(SchemaTypeError::CoercionFailed {
                    from: json_type_name(value),
                    to: target.type_name(),
                })
            }
        }
    }
}

// ── JSON Schema conversion ──────────────────────────────────────

/// Convert a `SchemaType` to a JSON Schema (draft-07 style) as `serde_json::Value`.
pub fn to_json_schema(schema: &SchemaType) -> Value {
    match schema {
        SchemaType::Null => serde_json::json!({"type": "null"}),
        SchemaType::Bool => serde_json::json!({"type": "boolean"}),
        SchemaType::Int => serde_json::json!({"type": "integer"}),
        SchemaType::Float => serde_json::json!({"type": "number"}),
        SchemaType::Str => serde_json::json!({"type": "string"}),
        SchemaType::Any => serde_json::json!({}),
        SchemaType::Array(inner) => {
            serde_json::json!({
                "type": "array",
                "items": to_json_schema(inner)
            })
        }
        SchemaType::Map(val) => {
            serde_json::json!({
                "type": "object",
                "additionalProperties": to_json_schema(val)
            })
        }
        SchemaType::Optional(inner) => {
            let inner_js = to_json_schema(inner);
            serde_json::json!({
                "oneOf": [inner_js, {"type": "null"}]
            })
        }
        SchemaType::Struct(s) => {
            let mut props = serde_json::Map::new();
            let mut required = Vec::new();
            for field in &s.fields {
                props.insert(field.name.clone(), to_json_schema(&field.schema));
                if field.required {
                    required.push(Value::String(field.name.clone()));
                }
            }
            let mut obj = serde_json::Map::new();
            obj.insert("type".to_string(), Value::String("object".to_string()));
            obj.insert("properties".to_string(), Value::Object(props));
            if !required.is_empty() {
                obj.insert("required".to_string(), Value::Array(required));
            }
            obj.insert(
                "additionalProperties".to_string(),
                Value::Bool(s.additional_properties),
            );
            Value::Object(obj)
        }
        SchemaType::Enum(e) => {
            let values: Vec<Value> = e
                .variants
                .iter()
                .filter(|v| v.payload.is_none())
                .map(|v| Value::String(v.name.clone()))
                .collect();
            serde_json::json!({"enum": values})
        }
        SchemaType::Union(types) => {
            let schemas: Vec<Value> = types.iter().map(to_json_schema).collect();
            serde_json::json!({"oneOf": schemas})
        }
        SchemaType::Const(v) => serde_json::json!({"const": v}),
        SchemaType::Ref(name) => serde_json::json!({"$ref": format!("#/definitions/{name}")}),
    }
}

// ── Schema Evolution Rules ──────────────────────────────────────

/// Describes an allowed schema evolution change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvolutionRule {
    /// Adding optional fields is always safe.
    AddOptionalField,
    /// Removing optional fields is safe (reader ignores).
    RemoveOptionalField,
    /// Widening numeric types (int -> float) is safe.
    WidenNumeric,
    /// Making a required field optional is safe (with default).
    MakeOptional,
    /// Adding enum variants is safe for readers.
    AddEnumVariant,
}

/// Check which evolution rules apply when migrating from `old` to `new`.
pub fn check_evolution(old: &SchemaType, new: &SchemaType) -> Vec<EvolutionRule> {
    let mut rules = Vec::new();
    match (old, new) {
        (SchemaType::Int, SchemaType::Float) => {
            rules.push(EvolutionRule::WidenNumeric);
        }
        (SchemaType::Struct(old_s), SchemaType::Struct(new_s)) => {
            // Check for added optional fields.
            for nf in &new_s.fields {
                if !old_s.fields.iter().any(|of| of.name == nf.name) && !nf.required {
                    rules.push(EvolutionRule::AddOptionalField);
                }
            }
            // Check for removed optional fields.
            for of in &old_s.fields {
                if !new_s.fields.iter().any(|nf| nf.name == of.name) && !of.required {
                    rules.push(EvolutionRule::RemoveOptionalField);
                }
            }
            // Check required->optional transitions.
            for of in &old_s.fields {
                if let Some(nf) = new_s.fields.iter().find(|nf| nf.name == of.name) {
                    if of.required && !nf.required {
                        rules.push(EvolutionRule::MakeOptional);
                    }
                }
            }
        }
        (SchemaType::Enum(old_e), SchemaType::Enum(new_e)) => {
            for nv in &new_e.variants {
                if !old_e.variants.iter().any(|ov| ov.name == nv.name) {
                    rules.push(EvolutionRule::AddEnumVariant);
                }
            }
        }
        _ => {}
    }
    rules
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_primitives() {
        assert!(validate(&SchemaType::Bool, &json!(true)).is_ok());
        assert!(validate(&SchemaType::Int, &json!(42)).is_ok());
        assert!(validate(&SchemaType::Float, &json!(3.14)).is_ok());
        assert!(validate(&SchemaType::Str, &json!("hello")).is_ok());
        assert!(validate(&SchemaType::Null, &json!(null)).is_ok());
    }

    #[test]
    fn test_validate_primitive_mismatch() {
        let errs = validate(&SchemaType::Bool, &json!("true")).unwrap_err();
        assert_eq!(errs.len(), 1);
        match &errs[0] {
            SchemaTypeError::ValidationFailed { expected, got, .. } => {
                assert_eq!(expected, "bool");
                assert_eq!(got, "string");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_validate_array() {
        let schema = SchemaType::Array(Box::new(SchemaType::Int));
        assert!(validate(&schema, &json!([1, 2, 3])).is_ok());
        assert!(validate(&schema, &json!([1, "two"])).is_err());
        assert!(validate(&schema, &json!("not array")).is_err());
    }

    #[test]
    fn test_validate_optional() {
        let schema = SchemaType::Optional(Box::new(SchemaType::Int));
        assert!(validate(&schema, &json!(null)).is_ok());
        assert!(validate(&schema, &json!(42)).is_ok());
        assert!(validate(&schema, &json!("nope")).is_err());
    }

    #[test]
    fn test_validate_struct() {
        let schema = SchemaType::Struct(
            StructSchema::new("User")
                .with_field(SchemaField::required("name", SchemaType::Str))
                .with_field(SchemaField::optional("age", SchemaType::Int)),
        );
        assert!(validate(&schema, &json!({"name": "Alice"})).is_ok());
        assert!(validate(&schema, &json!({"name": "Alice", "age": 30})).is_ok());
        // Missing required field.
        assert!(validate(&schema, &json!({"age": 30})).is_err());
    }

    #[test]
    fn test_validate_struct_no_extra() {
        let schema = SchemaType::Struct(
            StructSchema::new("Strict")
                .with_field(SchemaField::required("x", SchemaType::Int)),
        );
        let errs = validate(&schema, &json!({"x": 1, "y": 2})).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, SchemaTypeError::ValidationFailed { got, .. } if got == "extra field")));
    }

    #[test]
    fn test_validate_struct_allow_extra() {
        let schema = SchemaType::Struct(
            StructSchema::new("Loose")
                .with_field(SchemaField::required("x", SchemaType::Int))
                .allow_additional(),
        );
        assert!(validate(&schema, &json!({"x": 1, "y": 2})).is_ok());
    }

    #[test]
    fn test_validate_enum_unit() {
        let schema = SchemaType::Enum(
            EnumSchema::new("Color")
                .with_variant(EnumVariant::unit("Red"))
                .with_variant(EnumVariant::unit("Green")),
        );
        assert!(validate(&schema, &json!("Red")).is_ok());
        assert!(validate(&schema, &json!("Blue")).is_err());
    }

    #[test]
    fn test_validate_enum_with_payload() {
        let schema = SchemaType::Enum(
            EnumSchema::new("Shape")
                .with_variant(EnumVariant::with_payload("Circle", SchemaType::Float))
                .with_variant(EnumVariant::unit("Square")),
        );
        assert!(validate(&schema, &json!({"Circle": 3.0})).is_ok());
        assert!(validate(&schema, &json!({"Circle": "bad"})).is_err());
        assert!(validate(&schema, &json!("Square")).is_ok());
    }

    #[test]
    fn test_validate_union() {
        let schema = SchemaType::Union(vec![SchemaType::Int, SchemaType::Str]);
        assert!(validate(&schema, &json!(42)).is_ok());
        assert!(validate(&schema, &json!("hi")).is_ok());
        assert!(validate(&schema, &json!(true)).is_err());
    }

    #[test]
    fn test_validate_map() {
        let schema = SchemaType::Map(Box::new(SchemaType::Int));
        assert!(validate(&schema, &json!({"a": 1, "b": 2})).is_ok());
        assert!(validate(&schema, &json!({"a": "x"})).is_err());
    }

    #[test]
    fn test_validate_const() {
        let schema = SchemaType::Const(json!(42));
        assert!(validate(&schema, &json!(42)).is_ok());
        assert!(validate(&schema, &json!(43)).is_err());
    }

    #[test]
    fn test_validate_any() {
        assert!(validate(&SchemaType::Any, &json!(null)).is_ok());
        assert!(validate(&SchemaType::Any, &json!([1, 2, 3])).is_ok());
    }

    #[test]
    fn test_compatibility_basic() {
        assert!(is_compatible(&SchemaType::Int, &SchemaType::Int));
        assert!(is_compatible(&SchemaType::Float, &SchemaType::Int));
        assert!(!is_compatible(&SchemaType::Int, &SchemaType::Float));
        assert!(!is_compatible(&SchemaType::Str, &SchemaType::Int));
    }

    #[test]
    fn test_compatibility_optional() {
        assert!(is_compatible(
            &SchemaType::Optional(Box::new(SchemaType::Int)),
            &SchemaType::Int
        ));
    }

    #[test]
    fn test_coerce_string_to_int() {
        let result = coerce(&json!("42"), &SchemaType::Int).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_coerce_int_to_string() {
        let result = coerce(&json!(42), &SchemaType::Str).unwrap();
        assert_eq!(result, json!("42"));
    }

    #[test]
    fn test_coerce_string_to_bool() {
        assert_eq!(coerce(&json!("true"), &SchemaType::Bool).unwrap(), json!(true));
        assert_eq!(coerce(&json!("0"), &SchemaType::Bool).unwrap(), json!(false));
    }

    #[test]
    fn test_coerce_null_to_optional() {
        let result = coerce(&json!(null), &SchemaType::Optional(Box::new(SchemaType::Int))).unwrap();
        assert_eq!(result, json!(null));
    }

    #[test]
    fn test_to_json_schema_struct() {
        let schema = SchemaType::Struct(
            StructSchema::new("User")
                .with_field(SchemaField::required("name", SchemaType::Str))
                .with_field(SchemaField::optional("age", SchemaType::Int)),
        );
        let js = to_json_schema(&schema);
        assert_eq!(js["type"], "object");
        assert_eq!(js["properties"]["name"]["type"], "string");
        assert_eq!(js["required"], json!(["name"]));
    }

    #[test]
    fn test_to_json_schema_array() {
        let schema = SchemaType::Array(Box::new(SchemaType::Str));
        let js = to_json_schema(&schema);
        assert_eq!(js["type"], "array");
        assert_eq!(js["items"]["type"], "string");
    }

    #[test]
    fn test_to_json_schema_enum() {
        let schema = SchemaType::Enum(
            EnumSchema::new("Status")
                .with_variant(EnumVariant::unit("Active"))
                .with_variant(EnumVariant::unit("Inactive")),
        );
        let js = to_json_schema(&schema);
        assert_eq!(js["enum"], json!(["Active", "Inactive"]));
    }

    #[test]
    fn test_evolution_add_optional_field() {
        let old = SchemaType::Struct(
            StructSchema::new("V1").with_field(SchemaField::required("x", SchemaType::Int)),
        );
        let new = SchemaType::Struct(
            StructSchema::new("V2")
                .with_field(SchemaField::required("x", SchemaType::Int))
                .with_field(SchemaField::optional("y", SchemaType::Str)),
        );
        let rules = check_evolution(&old, &new);
        assert!(rules.contains(&EvolutionRule::AddOptionalField));
    }

    #[test]
    fn test_evolution_widen_numeric() {
        let rules = check_evolution(&SchemaType::Int, &SchemaType::Float);
        assert!(rules.contains(&EvolutionRule::WidenNumeric));
    }

    #[test]
    fn test_type_name() {
        assert_eq!(SchemaType::Int.type_name(), "int");
        let arr = SchemaType::Array(Box::new(SchemaType::Str));
        assert_eq!(arr.type_name(), "array<string>");
        let opt = SchemaType::Optional(Box::new(SchemaType::Bool));
        assert_eq!(opt.type_name(), "optional<bool>");
    }

    #[test]
    fn test_is_primitive() {
        assert!(SchemaType::Int.is_primitive());
        assert!(SchemaType::Str.is_primitive());
        assert!(!SchemaType::Array(Box::new(SchemaType::Int)).is_primitive());
    }

    #[test]
    fn test_struct_required_fields() {
        let s = StructSchema::new("T")
            .with_field(SchemaField::required("a", SchemaType::Int))
            .with_field(SchemaField::optional("b", SchemaType::Str))
            .with_field(SchemaField::required("c", SchemaType::Bool));
        assert_eq!(s.required_fields(), vec!["a", "c"]);
    }

    #[test]
    fn test_enum_variant_names() {
        let e = EnumSchema::new("Dir")
            .with_variant(EnumVariant::unit("N"))
            .with_variant(EnumVariant::unit("S"));
        assert_eq!(e.variant_names(), vec!["N", "S"]);
    }
}
