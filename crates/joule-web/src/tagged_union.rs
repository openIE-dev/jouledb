//! Tagged union / algebraic data type — variant definitions, pattern matching
//! dispatch, visitor pattern, fold/map over variants, serialization
//! (internally/externally tagged), and exhaustive matching checks.
//!
//! Replaces TypeScript discriminated unions, Haskell ADTs, and various
//! sum-type libraries with a pure-Rust runtime tagged union system.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from tagged union operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaggedUnionError {
    /// Unknown variant name.
    UnknownVariant(String),
    /// Variant already defined.
    DuplicateVariant(String),
    /// Not all variants are covered by a match.
    NonExhaustiveMatch { missing: Vec<String> },
    /// Deserialization failure.
    DeserializeFailed(String),
    /// Payload type mismatch.
    PayloadMismatch { variant: String, expected: String, got: String },
    /// Tag field not found.
    TagFieldMissing(String),
}

impl fmt::Display for TaggedUnionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariant(v) => write!(f, "unknown variant: {v}"),
            Self::DuplicateVariant(v) => write!(f, "duplicate variant: {v}"),
            Self::NonExhaustiveMatch { missing } => {
                write!(f, "non-exhaustive match, missing: {}", missing.join(", "))
            }
            Self::DeserializeFailed(msg) => write!(f, "deserialize failed: {msg}"),
            Self::PayloadMismatch { variant, expected, got } => {
                write!(f, "payload mismatch for {variant}: expected {expected}, got {got}")
            }
            Self::TagFieldMissing(field) => write!(f, "tag field missing: {field}"),
        }
    }
}

impl std::error::Error for TaggedUnionError {}

// ── Payload ─────────────────────────────────────────────────────

/// Payload kind for a variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PayloadKind {
    /// No payload (unit variant).
    Unit,
    /// A single value.
    Single(Value),
    /// Named fields (struct variant).
    Struct(HashMap<String, Value>),
    /// Positional fields (tuple variant).
    Tuple(Vec<Value>),
}

impl PayloadKind {
    /// Whether this is a unit variant.
    pub fn is_unit(&self) -> bool {
        matches!(self, Self::Unit)
    }

    /// Try to get as a single value.
    pub fn as_single(&self) -> Option<&Value> {
        match self {
            Self::Single(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as struct fields.
    pub fn as_struct(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Self::Struct(m) => Some(m),
            _ => None,
        }
    }

    /// Try to get as tuple fields.
    pub fn as_tuple(&self) -> Option<&[Value]> {
        match self {
            Self::Tuple(v) => Some(v),
            _ => None,
        }
    }
}

// ── Variant Definition ──────────────────────────────────────────

/// Defines what kinds of payloads a variant can carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariantShape {
    /// No data.
    Unit,
    /// Single anonymous value.
    Single,
    /// Named fields with type labels (for documentation).
    Struct(Vec<String>),
    /// Positional fields count.
    Tuple(usize),
}

/// A variant definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantDef {
    /// Variant name.
    pub name: String,
    /// Shape of the variant.
    pub shape: VariantShape,
    /// Optional description.
    pub description: Option<String>,
}

impl VariantDef {
    /// Create a unit variant definition.
    pub fn unit(name: impl Into<String>) -> Self {
        Self { name: name.into(), shape: VariantShape::Unit, description: None }
    }

    /// Create a single-value variant definition.
    pub fn single(name: impl Into<String>) -> Self {
        Self { name: name.into(), shape: VariantShape::Single, description: None }
    }

    /// Create a struct variant definition with field names.
    pub fn with_fields(name: impl Into<String>, fields: Vec<String>) -> Self {
        Self { name: name.into(), shape: VariantShape::Struct(fields), description: None }
    }

    /// Create a tuple variant definition.
    pub fn tuple(name: impl Into<String>, arity: usize) -> Self {
        Self { name: name.into(), shape: VariantShape::Tuple(arity), description: None }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// ── Tagged Union Value ──────────────────────────────────────────

/// A concrete tagged union value: a variant name plus payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaggedValue {
    /// The active variant name.
    pub variant: String,
    /// The payload.
    pub payload: PayloadKind,
}

impl TaggedValue {
    /// Create a unit tagged value.
    pub fn unit(variant: impl Into<String>) -> Self {
        Self { variant: variant.into(), payload: PayloadKind::Unit }
    }

    /// Create a single-value tagged value.
    pub fn single(variant: impl Into<String>, value: Value) -> Self {
        Self { variant: variant.into(), payload: PayloadKind::Single(value) }
    }

    /// Create a struct tagged value.
    pub fn with_fields(variant: impl Into<String>, fields: HashMap<String, Value>) -> Self {
        Self { variant: variant.into(), payload: PayloadKind::Struct(fields) }
    }

    /// Create a tuple tagged value.
    pub fn tuple(variant: impl Into<String>, values: Vec<Value>) -> Self {
        Self { variant: variant.into(), payload: PayloadKind::Tuple(values) }
    }

    /// Map the payload (if single) through a function.
    pub fn map_single(self, f: impl FnOnce(Value) -> Value) -> Self {
        match self.payload {
            PayloadKind::Single(v) => Self {
                variant: self.variant,
                payload: PayloadKind::Single(f(v)),
            },
            other => Self { variant: self.variant, payload: other },
        }
    }
}

impl fmt::Display for TaggedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.payload {
            PayloadKind::Unit => write!(f, "{}", self.variant),
            PayloadKind::Single(v) => write!(f, "{}({})", self.variant, v),
            PayloadKind::Struct(fields) => {
                let entries: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect();
                write!(f, "{} {{ {} }}", self.variant, entries.join(", "))
            }
            PayloadKind::Tuple(vals) => {
                let entries: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                write!(f, "{}({})", self.variant, entries.join(", "))
            }
        }
    }
}

// ── Tagged Union Definition ─────────────────────────────────────

/// Tagging strategy for serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaggingStrategy {
    /// Externally tagged: `{"VariantName": payload}`.
    External,
    /// Internally tagged: `{"type": "VariantName", ...fields}`.
    Internal { tag_field: String },
    /// Adjacently tagged: `{"type": "VariantName", "value": payload}`.
    Adjacent { tag_field: String, content_field: String },
    /// Untagged: tries each variant shape in order.
    Untagged,
}

impl Default for TaggingStrategy {
    fn default() -> Self {
        Self::External
    }
}

/// A tagged union definition: the set of allowed variants and tagging strategy.
#[derive(Debug, Clone)]
pub struct TaggedUnionDef {
    /// Union name.
    pub name: String,
    /// Variant definitions in order.
    pub variants: Vec<VariantDef>,
    /// Serialization tagging strategy.
    pub strategy: TaggingStrategy,
}

impl TaggedUnionDef {
    /// Create a new tagged union definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variants: Vec::new(),
            strategy: TaggingStrategy::default(),
        }
    }

    /// Set the tagging strategy.
    pub fn with_strategy(mut self, strategy: TaggingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add a variant definition.
    pub fn with_variant(mut self, variant: VariantDef) -> Self {
        self.variants.push(variant);
        self
    }

    /// Get variant names.
    pub fn variant_names(&self) -> Vec<&str> {
        self.variants.iter().map(|v| v.name.as_str()).collect()
    }

    /// Get variant def by name.
    pub fn get_variant(&self, name: &str) -> Option<&VariantDef> {
        self.variants.iter().find(|v| v.name == name)
    }

    /// Validate a tagged value against this definition.
    pub fn validate(&self, value: &TaggedValue) -> Result<(), TaggedUnionError> {
        let def = self
            .get_variant(&value.variant)
            .ok_or_else(|| TaggedUnionError::UnknownVariant(value.variant.clone()))?;

        match (&def.shape, &value.payload) {
            (VariantShape::Unit, PayloadKind::Unit) => Ok(()),
            (VariantShape::Single, PayloadKind::Single(_)) => Ok(()),
            (VariantShape::Struct(fields), PayloadKind::Struct(map)) => {
                for field in fields {
                    if !map.contains_key(field) {
                        return Err(TaggedUnionError::PayloadMismatch {
                            variant: value.variant.clone(),
                            expected: format!("field {field}"),
                            got: "missing".to_string(),
                        });
                    }
                }
                Ok(())
            }
            (VariantShape::Tuple(n), PayloadKind::Tuple(vals)) => {
                if vals.len() != *n {
                    return Err(TaggedUnionError::PayloadMismatch {
                        variant: value.variant.clone(),
                        expected: format!("{n} elements"),
                        got: format!("{} elements", vals.len()),
                    });
                }
                Ok(())
            }
            (expected_shape, _) => Err(TaggedUnionError::PayloadMismatch {
                variant: value.variant.clone(),
                expected: format!("{expected_shape:?}"),
                got: format!("{:?}", value.payload),
            }),
        }
    }

    /// Serialize a tagged value to JSON according to the tagging strategy.
    pub fn to_json(&self, value: &TaggedValue) -> Result<Value, TaggedUnionError> {
        self.validate(value)?;
        match &self.strategy {
            TaggingStrategy::External => {
                let payload_json = payload_to_json(&value.payload);
                let mut map = serde_json::Map::new();
                map.insert(value.variant.clone(), payload_json);
                Ok(Value::Object(map))
            }
            TaggingStrategy::Internal { tag_field } => {
                let mut map = match &value.payload {
                    PayloadKind::Unit => serde_json::Map::new(),
                    PayloadKind::Struct(fields) => {
                        let mut m = serde_json::Map::new();
                        for (k, v) in fields {
                            m.insert(k.clone(), v.clone());
                        }
                        m
                    }
                    PayloadKind::Single(v) => {
                        let mut m = serde_json::Map::new();
                        m.insert("value".to_string(), v.clone());
                        m
                    }
                    PayloadKind::Tuple(vals) => {
                        let mut m = serde_json::Map::new();
                        m.insert("values".to_string(), Value::Array(vals.clone()));
                        m
                    }
                };
                map.insert(tag_field.clone(), Value::String(value.variant.clone()));
                Ok(Value::Object(map))
            }
            TaggingStrategy::Adjacent { tag_field, content_field } => {
                let payload_json = payload_to_json(&value.payload);
                let mut map = serde_json::Map::new();
                map.insert(tag_field.clone(), Value::String(value.variant.clone()));
                if !value.payload.is_unit() {
                    map.insert(content_field.clone(), payload_json);
                }
                Ok(Value::Object(map))
            }
            TaggingStrategy::Untagged => Ok(payload_to_json(&value.payload)),
        }
    }

    /// Deserialize a JSON value into a tagged value.
    pub fn from_json(&self, json: &Value) -> Result<TaggedValue, TaggedUnionError> {
        match &self.strategy {
            TaggingStrategy::External => {
                let obj = json.as_object().ok_or_else(|| {
                    TaggedUnionError::DeserializeFailed("expected object".to_string())
                })?;
                if obj.len() != 1 {
                    return Err(TaggedUnionError::DeserializeFailed(
                        "externally tagged union must have exactly one key".to_string(),
                    ));
                }
                let (variant_name, payload_json) = obj.iter().next().unwrap();
                let def = self.get_variant(variant_name).ok_or_else(|| {
                    TaggedUnionError::UnknownVariant(variant_name.clone())
                })?;
                let payload = json_to_payload(payload_json, &def.shape);
                Ok(TaggedValue { variant: variant_name.clone(), payload })
            }
            TaggingStrategy::Internal { tag_field } => {
                let obj = json.as_object().ok_or_else(|| {
                    TaggedUnionError::DeserializeFailed("expected object".to_string())
                })?;
                let variant_name = obj
                    .get(tag_field)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| TaggedUnionError::TagFieldMissing(tag_field.clone()))?;
                let def = self.get_variant(variant_name).ok_or_else(|| {
                    TaggedUnionError::UnknownVariant(variant_name.to_string())
                })?;
                let payload = match &def.shape {
                    VariantShape::Unit => PayloadKind::Unit,
                    VariantShape::Single => {
                        let val = obj.get("value").cloned().unwrap_or(Value::Null);
                        PayloadKind::Single(val)
                    }
                    VariantShape::Struct(fields) => {
                        let mut map = HashMap::new();
                        for field in fields {
                            if let Some(val) = obj.get(field) {
                                map.insert(field.clone(), val.clone());
                            }
                        }
                        PayloadKind::Struct(map)
                    }
                    VariantShape::Tuple(_) => {
                        let vals = obj
                            .get("values")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        PayloadKind::Tuple(vals)
                    }
                };
                Ok(TaggedValue { variant: variant_name.to_string(), payload })
            }
            TaggingStrategy::Adjacent { tag_field, content_field } => {
                let obj = json.as_object().ok_or_else(|| {
                    TaggedUnionError::DeserializeFailed("expected object".to_string())
                })?;
                let variant_name = obj
                    .get(tag_field)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| TaggedUnionError::TagFieldMissing(tag_field.clone()))?;
                let def = self.get_variant(variant_name).ok_or_else(|| {
                    TaggedUnionError::UnknownVariant(variant_name.to_string())
                })?;
                let content = obj.get(content_field);
                let payload = match content {
                    Some(v) => json_to_payload(v, &def.shape),
                    None => PayloadKind::Unit,
                };
                Ok(TaggedValue { variant: variant_name.to_string(), payload })
            }
            TaggingStrategy::Untagged => {
                // Try each variant in order.
                for def in &self.variants {
                    let payload = json_to_payload(json, &def.shape);
                    let val = TaggedValue { variant: def.name.clone(), payload };
                    if self.validate(&val).is_ok() {
                        return Ok(val);
                    }
                }
                Err(TaggedUnionError::DeserializeFailed(
                    "no variant matched for untagged union".to_string(),
                ))
            }
        }
    }
}

fn payload_to_json(payload: &PayloadKind) -> Value {
    match payload {
        PayloadKind::Unit => Value::Null,
        PayloadKind::Single(v) => v.clone(),
        PayloadKind::Struct(map) => {
            let obj: serde_json::Map<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Value::Object(obj)
        }
        PayloadKind::Tuple(vals) => Value::Array(vals.clone()),
    }
}

fn json_to_payload(json: &Value, shape: &VariantShape) -> PayloadKind {
    match shape {
        VariantShape::Unit => PayloadKind::Unit,
        VariantShape::Single => PayloadKind::Single(json.clone()),
        VariantShape::Struct(fields) => {
            if let Some(obj) = json.as_object() {
                let mut map = HashMap::new();
                for field in fields {
                    if let Some(val) = obj.get(field) {
                        map.insert(field.clone(), val.clone());
                    }
                }
                PayloadKind::Struct(map)
            } else {
                PayloadKind::Struct(HashMap::new())
            }
        }
        VariantShape::Tuple(_) => {
            if let Some(arr) = json.as_array() {
                PayloadKind::Tuple(arr.clone())
            } else {
                PayloadKind::Tuple(Vec::new())
            }
        }
    }
}

// ── Match Arm & Dispatch ────────────────────────────────────────

/// An arm in a pattern match over a tagged union.
pub struct MatchArm<T> {
    /// Variant name to match.
    pub variant: String,
    /// Handler function.
    handler: Box<dyn Fn(&PayloadKind) -> T>,
}

/// Builder for pattern matching dispatch over tagged union values.
pub struct MatchBuilder<T> {
    arms: Vec<MatchArm<T>>,
    default: Option<Box<dyn Fn(&TaggedValue) -> T>>,
}

impl<T> MatchBuilder<T> {
    /// Create a new match builder.
    pub fn new() -> Self {
        Self { arms: Vec::new(), default: None }
    }

    /// Add a match arm for a variant.
    pub fn on(mut self, variant: impl Into<String>, handler: impl Fn(&PayloadKind) -> T + 'static) -> Self {
        self.arms.push(MatchArm {
            variant: variant.into(),
            handler: Box::new(handler),
        });
        self
    }

    /// Set a default handler for unmatched variants.
    pub fn default_handler(mut self, handler: impl Fn(&TaggedValue) -> T + 'static) -> Self {
        self.default = Some(Box::new(handler));
        self
    }

    /// Check exhaustiveness against a union definition.
    pub fn check_exhaustive(&self, def: &TaggedUnionDef) -> Result<(), TaggedUnionError> {
        let covered: Vec<&str> = self.arms.iter().map(|a| a.variant.as_str()).collect();
        let missing: Vec<String> = def
            .variants
            .iter()
            .filter(|v| !covered.contains(&v.name.as_str()))
            .map(|v| v.name.clone())
            .collect();
        if missing.is_empty() || self.default.is_some() {
            Ok(())
        } else {
            Err(TaggedUnionError::NonExhaustiveMatch { missing })
        }
    }

    /// Execute the match on a tagged value.
    pub fn dispatch(&self, value: &TaggedValue) -> Result<T, TaggedUnionError> {
        for arm in &self.arms {
            if arm.variant == value.variant {
                return Ok((arm.handler)(&value.payload));
            }
        }
        if let Some(default) = &self.default {
            Ok(default(value))
        } else {
            Err(TaggedUnionError::UnknownVariant(value.variant.clone()))
        }
    }
}

impl<T> Default for MatchBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Visitor ─────────────────────────────────────────────────────

/// A visitor over tagged union values.
pub trait TaggedVisitor {
    /// Output type.
    type Output;

    /// Visit a unit variant.
    fn visit_unit(&self, variant: &str) -> Self::Output;

    /// Visit a single-value variant.
    fn visit_single(&self, variant: &str, value: &Value) -> Self::Output;

    /// Visit a struct variant.
    fn visit_struct(&self, variant: &str, fields: &HashMap<String, Value>) -> Self::Output;

    /// Visit a tuple variant.
    fn visit_tuple(&self, variant: &str, values: &[Value]) -> Self::Output;

    /// Dispatch a tagged value through this visitor.
    fn visit(&self, value: &TaggedValue) -> Self::Output {
        match &value.payload {
            PayloadKind::Unit => self.visit_unit(&value.variant),
            PayloadKind::Single(v) => self.visit_single(&value.variant, v),
            PayloadKind::Struct(m) => self.visit_struct(&value.variant, m),
            PayloadKind::Tuple(vs) => self.visit_tuple(&value.variant, vs),
        }
    }
}

// ── Fold / Map ──────────────────────────────────────────────────

/// Map a function over the single-value payloads in a collection of tagged values.
pub fn map_payloads(
    values: &[TaggedValue],
    f: impl Fn(&str, &Value) -> Value,
) -> Vec<TaggedValue> {
    values
        .iter()
        .map(|tv| match &tv.payload {
            PayloadKind::Single(v) => TaggedValue {
                variant: tv.variant.clone(),
                payload: PayloadKind::Single(f(&tv.variant, v)),
            },
            _ => tv.clone(),
        })
        .collect()
}

/// Fold over a collection of tagged values, accumulating a result.
pub fn fold_payloads<A>(
    values: &[TaggedValue],
    init: A,
    f: impl Fn(A, &str, &PayloadKind) -> A,
) -> A {
    values.iter().fold(init, |acc, tv| f(acc, &tv.variant, &tv.payload))
}

/// Filter tagged values by variant name.
pub fn filter_variant<'a>(values: &'a [TaggedValue], variant: &str) -> Vec<&'a TaggedValue> {
    values.iter().filter(|tv| tv.variant == variant).collect()
}

/// Group tagged values by variant name.
pub fn group_by_variant(values: &[TaggedValue]) -> HashMap<String, Vec<&TaggedValue>> {
    let mut groups: HashMap<String, Vec<&TaggedValue>> = HashMap::new();
    for tv in values {
        groups.entry(tv.variant.clone()).or_default().push(tv);
    }
    groups
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shape_def() -> TaggedUnionDef {
        TaggedUnionDef::new("Shape")
            .with_variant(VariantDef::unit("Circle"))
            .with_variant(VariantDef::single("Square"))
            .with_variant(VariantDef::with_fields(
                "Rectangle",
                vec!["width".to_string(), "height".to_string()],
            ))
            .with_variant(VariantDef::tuple("Point", 2))
    }

    #[test]
    fn test_validate_unit_variant() {
        let def = shape_def();
        let val = TaggedValue::unit("Circle");
        assert!(def.validate(&val).is_ok());
    }

    #[test]
    fn test_validate_single_variant() {
        let def = shape_def();
        let val = TaggedValue::single("Square", json!(5));
        assert!(def.validate(&val).is_ok());
    }

    #[test]
    fn test_validate_struct_variant() {
        let def = shape_def();
        let mut fields = HashMap::new();
        fields.insert("width".to_string(), json!(10));
        fields.insert("height".to_string(), json!(20));
        let val = TaggedValue::with_fields("Rectangle", fields);
        assert!(def.validate(&val).is_ok());
    }

    #[test]
    fn test_validate_tuple_variant() {
        let def = shape_def();
        let val = TaggedValue::tuple("Point", vec![json!(1), json!(2)]);
        assert!(def.validate(&val).is_ok());
    }

    #[test]
    fn test_validate_unknown_variant() {
        let def = shape_def();
        let val = TaggedValue::unit("Triangle");
        let err = def.validate(&val).unwrap_err();
        assert_eq!(err, TaggedUnionError::UnknownVariant("Triangle".to_string()));
    }

    #[test]
    fn test_validate_payload_mismatch() {
        let def = shape_def();
        let val = TaggedValue::single("Circle", json!(5)); // Circle is unit
        assert!(def.validate(&val).is_err());
    }

    #[test]
    fn test_validate_tuple_arity_mismatch() {
        let def = shape_def();
        let val = TaggedValue::tuple("Point", vec![json!(1)]);
        assert!(def.validate(&val).is_err());
    }

    #[test]
    fn test_external_tagging_roundtrip() {
        let def = shape_def();
        let val = TaggedValue::single("Square", json!(5));
        let json = def.to_json(&val).unwrap();
        assert_eq!(json, json!({"Square": 5}));
        let back = def.from_json(&json).unwrap();
        assert_eq!(back.variant, "Square");
    }

    #[test]
    fn test_internal_tagging_roundtrip() {
        let def = TaggedUnionDef::new("Event")
            .with_strategy(TaggingStrategy::Internal { tag_field: "type".to_string() })
            .with_variant(VariantDef::unit("Click"))
            .with_variant(VariantDef::with_fields(
                "Move",
                vec!["x".to_string(), "y".to_string()],
            ));

        let val = TaggedValue::unit("Click");
        let json = def.to_json(&val).unwrap();
        assert_eq!(json["type"], "Click");
        let back = def.from_json(&json).unwrap();
        assert_eq!(back.variant, "Click");
    }

    #[test]
    fn test_adjacent_tagging_roundtrip() {
        let def = TaggedUnionDef::new("Msg")
            .with_strategy(TaggingStrategy::Adjacent {
                tag_field: "t".to_string(),
                content_field: "c".to_string(),
            })
            .with_variant(VariantDef::single("Text"));

        let val = TaggedValue::single("Text", json!("hello"));
        let json = def.to_json(&val).unwrap();
        assert_eq!(json["t"], "Text");
        assert_eq!(json["c"], "hello");
        let back = def.from_json(&json).unwrap();
        assert_eq!(back.variant, "Text");
    }

    #[test]
    fn test_match_dispatch() {
        let matcher = MatchBuilder::<String>::new()
            .on("Circle", |_| "circle".to_string())
            .on("Square", |p| {
                if let PayloadKind::Single(v) = p {
                    format!("square({})", v)
                } else {
                    "square".to_string()
                }
            });

        let result = matcher.dispatch(&TaggedValue::unit("Circle")).unwrap();
        assert_eq!(result, "circle");

        let result = matcher.dispatch(&TaggedValue::single("Square", json!(5))).unwrap();
        assert_eq!(result, "square(5)");
    }

    #[test]
    fn test_match_default_handler() {
        let matcher = MatchBuilder::<String>::new()
            .on("A", |_| "a".to_string())
            .default_handler(|tv| format!("default:{}", tv.variant));

        let result = matcher.dispatch(&TaggedValue::unit("B")).unwrap();
        assert_eq!(result, "default:B");
    }

    #[test]
    fn test_match_no_handler() {
        let matcher = MatchBuilder::<String>::new()
            .on("A", |_| "a".to_string());
        let err = matcher.dispatch(&TaggedValue::unit("B")).unwrap_err();
        assert_eq!(err, TaggedUnionError::UnknownVariant("B".to_string()));
    }

    #[test]
    fn test_exhaustive_check_pass() {
        let def = TaggedUnionDef::new("AB")
            .with_variant(VariantDef::unit("A"))
            .with_variant(VariantDef::unit("B"));
        let matcher = MatchBuilder::<()>::new()
            .on("A", |_| ())
            .on("B", |_| ());
        assert!(matcher.check_exhaustive(&def).is_ok());
    }

    #[test]
    fn test_exhaustive_check_fail() {
        let def = TaggedUnionDef::new("AB")
            .with_variant(VariantDef::unit("A"))
            .with_variant(VariantDef::unit("B"));
        let matcher = MatchBuilder::<()>::new()
            .on("A", |_| ());
        let err = matcher.check_exhaustive(&def).unwrap_err();
        assert!(matches!(err, TaggedUnionError::NonExhaustiveMatch { missing } if missing == vec!["B"]));
    }

    #[test]
    fn test_exhaustive_check_with_default() {
        let def = TaggedUnionDef::new("AB")
            .with_variant(VariantDef::unit("A"))
            .with_variant(VariantDef::unit("B"));
        let matcher = MatchBuilder::<()>::new()
            .on("A", |_| ())
            .default_handler(|_| ());
        assert!(matcher.check_exhaustive(&def).is_ok());
    }

    #[test]
    fn test_visitor() {
        struct CountVisitor;
        impl TaggedVisitor for CountVisitor {
            type Output = usize;
            fn visit_unit(&self, _variant: &str) -> usize { 0 }
            fn visit_single(&self, _variant: &str, _value: &Value) -> usize { 1 }
            fn visit_struct(&self, _variant: &str, fields: &HashMap<String, Value>) -> usize {
                fields.len()
            }
            fn visit_tuple(&self, _variant: &str, values: &[Value]) -> usize { values.len() }
        }
        let v = CountVisitor;
        assert_eq!(v.visit(&TaggedValue::unit("A")), 0);
        assert_eq!(v.visit(&TaggedValue::single("B", json!(1))), 1);
        let mut m = HashMap::new();
        m.insert("x".to_string(), json!(1));
        m.insert("y".to_string(), json!(2));
        assert_eq!(v.visit(&TaggedValue::with_fields("C", m)), 2);
    }

    #[test]
    fn test_map_payloads() {
        let values = vec![
            TaggedValue::single("A", json!(1)),
            TaggedValue::single("B", json!(2)),
            TaggedValue::unit("C"),
        ];
        let mapped = map_payloads(&values, |_variant, v| {
            json!(v.as_i64().unwrap_or(0) * 10)
        });
        assert_eq!(mapped[0].payload.as_single(), Some(&json!(10)));
        assert_eq!(mapped[1].payload.as_single(), Some(&json!(20)));
        assert!(mapped[2].payload.is_unit());
    }

    #[test]
    fn test_fold_payloads() {
        let values = vec![
            TaggedValue::single("A", json!(1)),
            TaggedValue::single("A", json!(2)),
            TaggedValue::single("B", json!(3)),
        ];
        let sum = fold_payloads(&values, 0i64, |acc, _variant, payload| {
            if let PayloadKind::Single(v) = payload {
                acc + v.as_i64().unwrap_or(0)
            } else {
                acc
            }
        });
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_filter_variant() {
        let values = vec![
            TaggedValue::unit("A"),
            TaggedValue::unit("B"),
            TaggedValue::unit("A"),
        ];
        let filtered = filter_variant(&values, "A");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_group_by_variant() {
        let values = vec![
            TaggedValue::unit("A"),
            TaggedValue::unit("B"),
            TaggedValue::unit("A"),
        ];
        let groups = group_by_variant(&values);
        assert_eq!(groups["A"].len(), 2);
        assert_eq!(groups["B"].len(), 1);
    }

    #[test]
    fn test_tagged_value_display() {
        assert_eq!(TaggedValue::unit("None").to_string(), "None");
        assert_eq!(TaggedValue::single("Some", json!(42)).to_string(), "Some(42)");
    }

    #[test]
    fn test_tagged_value_map_single() {
        let val = TaggedValue::single("X", json!(5));
        let mapped = val.map_single(|v| json!(v.as_i64().unwrap_or(0) + 1));
        assert_eq!(mapped.payload.as_single(), Some(&json!(6)));
    }

    #[test]
    fn test_variant_names() {
        let def = shape_def();
        assert_eq!(def.variant_names(), vec!["Circle", "Square", "Rectangle", "Point"]);
    }

    #[test]
    fn test_variant_description() {
        let v = VariantDef::unit("X").with_description("The X variant");
        assert_eq!(v.description.as_deref(), Some("The X variant"));
    }

    #[test]
    fn test_struct_variant_missing_field() {
        let def = shape_def();
        let mut fields = HashMap::new();
        fields.insert("width".to_string(), json!(10));
        // Missing "height"
        let val = TaggedValue::with_fields("Rectangle", fields);
        let err = def.validate(&val).unwrap_err();
        assert!(matches!(err, TaggedUnionError::PayloadMismatch { .. }));
    }
}
