//! Protobuf reflection — runtime message descriptors, field accessors, dynamic messages.
//!
//! Pure-Rust protobuf reflection API. Supports runtime message descriptors,
//! field accessors (get/set by name or number), dynamic messages, message diff,
//! message merge, default value resolution, and a descriptor registry for
//! looking up types at runtime.

use std::collections::HashMap;
use std::fmt;

// ── Field Kind ───────────────────────────────────────────────

/// Protobuf field kind for reflection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Int32,
    Int64,
    Uint32,
    Uint64,
    Sint32,
    Sint64,
    Float,
    Double,
    Bool,
    String,
    Bytes,
    Message(String),
    Enum(String),
}

impl fmt::Display for FieldKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int32 => f.write_str("int32"),
            Self::Int64 => f.write_str("int64"),
            Self::Uint32 => f.write_str("uint32"),
            Self::Uint64 => f.write_str("uint64"),
            Self::Sint32 => f.write_str("sint32"),
            Self::Sint64 => f.write_str("sint64"),
            Self::Float => f.write_str("float"),
            Self::Double => f.write_str("double"),
            Self::Bool => f.write_str("bool"),
            Self::String => f.write_str("string"),
            Self::Bytes => f.write_str("bytes"),
            Self::Message(name) => write!(f, "message({name})"),
            Self::Enum(name) => write!(f, "enum({name})"),
        }
    }
}

// ── Field Cardinality ────────────────────────────────────────

/// Field cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    Singular,
    Repeated,
    Map,
}

// ── Dynamic Value ────────────────────────────────────────────

/// A dynamic protobuf value.
#[derive(Debug, Clone, PartialEq)]
pub enum DynValue {
    Null,
    Int32(i32),
    Int64(i64),
    Uint32(u32),
    Uint64(u64),
    Float(f32),
    Double(f64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Enum(i32),
    Message(Box<DynMessage>),
    List(Vec<DynValue>),
    Map(Vec<(DynValue, DynValue)>),
}

impl DynValue {
    /// Get as i32.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Double(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as bytes.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Get as enum value.
    pub fn as_enum(&self) -> Option<i32> {
        match self {
            Self::Enum(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as sub-message.
    pub fn as_message(&self) -> Option<&DynMessage> {
        match self {
            Self::Message(m) => Some(m),
            _ => None,
        }
    }

    /// Get as list.
    pub fn as_list(&self) -> Option<&[DynValue]> {
        match self {
            Self::List(l) => Some(l),
            _ => None,
        }
    }

    /// Whether this is the default value for its kind.
    pub fn is_default(&self) -> bool {
        match self {
            Self::Null => true,
            Self::Int32(v) => *v == 0,
            Self::Int64(v) => *v == 0,
            Self::Uint32(v) => *v == 0,
            Self::Uint64(v) => *v == 0,
            Self::Float(v) => *v == 0.0,
            Self::Double(v) => *v == 0.0,
            Self::Bool(v) => !*v,
            Self::String(s) => s.is_empty(),
            Self::Bytes(b) => b.is_empty(),
            Self::Enum(v) => *v == 0,
            Self::Message(_) => false,
            Self::List(l) => l.is_empty(),
            Self::Map(m) => m.is_empty(),
        }
    }

    /// Get the default value for a field kind.
    pub fn default_for(kind: &FieldKind) -> Self {
        match kind {
            FieldKind::Int32 | FieldKind::Sint32 => Self::Int32(0),
            FieldKind::Int64 | FieldKind::Sint64 => Self::Int64(0),
            FieldKind::Uint32 => Self::Uint32(0),
            FieldKind::Uint64 => Self::Uint64(0),
            FieldKind::Float => Self::Float(0.0),
            FieldKind::Double => Self::Double(0.0),
            FieldKind::Bool => Self::Bool(false),
            FieldKind::String => Self::String(String::new()),
            FieldKind::Bytes => Self::Bytes(Vec::new()),
            FieldKind::Enum(_) => Self::Enum(0),
            FieldKind::Message(_) => Self::Null,
        }
    }
}

impl fmt::Display for DynValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null"),
            Self::Int32(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Uint32(v) => write!(f, "{v}"),
            Self::Uint64(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Double(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            Self::Enum(v) => write!(f, "enum({v})"),
            Self::Message(m) => write!(f, "message({})", m.type_name),
            Self::List(l) => write!(f, "[{} items]", l.len()),
            Self::Map(m) => write!(f, "{{{} entries}}", m.len()),
        }
    }
}

// ── Field Descriptor ─────────────────────────────────────────

/// Runtime field descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDescriptor {
    /// Field name.
    pub name: String,
    /// Field number.
    pub number: u32,
    /// Field kind.
    pub kind: FieldKind,
    /// Cardinality.
    pub cardinality: Cardinality,
    /// Map key kind (only for Map cardinality).
    pub map_key_kind: Option<FieldKind>,
    /// JSON name.
    pub json_name: String,
    /// Whether deprecated.
    pub deprecated: bool,
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, number: u32, kind: FieldKind) -> Self {
        let name = name.into();
        let json_name = name.clone();
        Self {
            name,
            number,
            kind,
            cardinality: Cardinality::Singular,
            map_key_kind: None,
            json_name,
            deprecated: false,
        }
    }

    /// Set cardinality.
    pub fn with_cardinality(mut self, cardinality: Cardinality) -> Self {
        self.cardinality = cardinality;
        self
    }

    /// Set JSON name.
    pub fn with_json_name(mut self, name: impl Into<String>) -> Self {
        self.json_name = name.into();
        self
    }

    /// Set deprecated.
    pub fn with_deprecated(mut self, deprecated: bool) -> Self {
        self.deprecated = deprecated;
        self
    }

    /// Set map key kind.
    pub fn with_map_key(mut self, key_kind: FieldKind) -> Self {
        self.cardinality = Cardinality::Map;
        self.map_key_kind = Some(key_kind);
        self
    }

    /// Default value for this field.
    pub fn default_value(&self) -> DynValue {
        match self.cardinality {
            Cardinality::Singular => DynValue::default_for(&self.kind),
            Cardinality::Repeated => DynValue::List(Vec::new()),
            Cardinality::Map => DynValue::Map(Vec::new()),
        }
    }
}

// ── Message Descriptor ───────────────────────────────────────

/// Runtime message descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDescriptor {
    /// Fully-qualified type name.
    pub full_name: String,
    /// Fields.
    pub fields: Vec<FieldDescriptor>,
    /// Nested message type names.
    pub nested_types: Vec<String>,
    /// Nested enum type names.
    pub nested_enums: Vec<String>,
}

impl MessageDescriptor {
    pub fn new(full_name: impl Into<String>) -> Self {
        Self {
            full_name: full_name.into(),
            fields: Vec::new(),
            nested_types: Vec::new(),
            nested_enums: Vec::new(),
        }
    }

    /// Add a field.
    pub fn add_field(&mut self, field: FieldDescriptor) {
        self.fields.push(field);
    }

    /// Find field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Find field by number.
    pub fn field_by_number(&self, number: u32) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.number == number)
    }

    /// Find field by JSON name.
    pub fn field_by_json_name(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.json_name == name)
    }

    /// Number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// All field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Create a new DynMessage with all defaults.
    pub fn new_message(&self) -> DynMessage {
        let mut msg = DynMessage::new(self.full_name.clone());
        for field in &self.fields {
            msg.set(&field.name, field.default_value());
        }
        msg
    }
}

// ── Dynamic Message ──────────────────────────────────────────

/// A dynamic protobuf message that stores fields by name.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMessage {
    /// Type name.
    pub type_name: String,
    /// Fields by name.
    fields: HashMap<String, DynValue>,
}

impl DynMessage {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            fields: HashMap::new(),
        }
    }

    /// Set a field value.
    pub fn set(&mut self, name: &str, value: DynValue) {
        self.fields.insert(name.to_string(), value);
    }

    /// Get a field value.
    pub fn get(&self, name: &str) -> Option<&DynValue> {
        self.fields.get(name)
    }

    /// Get a field value, returning default if not set.
    pub fn get_or_default(&self, name: &str, desc: &MessageDescriptor) -> DynValue {
        if let Some(val) = self.fields.get(name) {
            val.clone()
        } else if let Some(fd) = desc.field_by_name(name) {
            fd.default_value()
        } else {
            DynValue::Null
        }
    }

    /// Remove a field.
    pub fn remove(&mut self, name: &str) -> Option<DynValue> {
        self.fields.remove(name)
    }

    /// Whether a field is set (and not default).
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.get(name).map(|v| !v.is_default()).unwrap_or(false)
    }

    /// Clear all fields.
    pub fn clear(&mut self) {
        self.fields.clear();
    }

    /// Number of set fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// All field names that are set (sorted).
    pub fn field_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.fields.keys().cloned().collect();
        names.sort();
        names
    }

    /// Merge another message into this one. For singular fields, `other`
    /// overwrites `self`. For repeated fields, values are appended.
    /// For sub-messages, merge recursively.
    pub fn merge(&mut self, other: &DynMessage) {
        for (name, value) in &other.fields {
            match (self.fields.get(name), value) {
                // Merge sub-messages recursively.
                (Some(DynValue::Message(existing)), DynValue::Message(incoming)) => {
                    let mut merged = (**existing).clone();
                    merged.merge(incoming);
                    self.fields.insert(name.clone(), DynValue::Message(Box::new(merged)));
                }
                // Append lists.
                (Some(DynValue::List(existing)), DynValue::List(incoming)) => {
                    let mut combined = existing.clone();
                    combined.extend(incoming.iter().cloned());
                    self.fields.insert(name.clone(), DynValue::List(combined));
                }
                // Overwrite everything else.
                _ => {
                    if !value.is_default() {
                        self.fields.insert(name.clone(), value.clone());
                    }
                }
            }
        }
    }

    /// Compute the diff between this message and another. Returns fields
    /// that differ (name, self_value, other_value).
    pub fn diff(&self, other: &DynMessage) -> Vec<FieldDiff> {
        let mut diffs = Vec::new();
        let mut all_keys: Vec<String> = self.fields.keys()
            .chain(other.fields.keys())
            .cloned()
            .collect();
        all_keys.sort();
        all_keys.dedup();

        for key in all_keys {
            let a = self.fields.get(&key);
            let b = other.fields.get(&key);
            match (a, b) {
                (Some(va), Some(vb)) if va != vb => {
                    diffs.push(FieldDiff {
                        field_name: key,
                        left: Some(va.clone()),
                        right: Some(vb.clone()),
                    });
                }
                (Some(va), None) => {
                    diffs.push(FieldDiff {
                        field_name: key,
                        left: Some(va.clone()),
                        right: None,
                    });
                }
                (None, Some(vb)) => {
                    diffs.push(FieldDiff {
                        field_name: key,
                        left: None,
                        right: Some(vb.clone()),
                    });
                }
                _ => {} // equal or both None
            }
        }
        diffs
    }

    /// Check equality with another message (field-by-field).
    pub fn equals(&self, other: &DynMessage) -> bool {
        self.type_name == other.type_name && self.fields == other.fields
    }
}

// ── Field Diff ───────────────────────────────────────────────

/// A difference in a single field between two messages.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDiff {
    pub field_name: String,
    pub left: Option<DynValue>,
    pub right: Option<DynValue>,
}

impl FieldDiff {
    /// Whether the field was added (only in right).
    pub fn is_added(&self) -> bool {
        self.left.is_none() && self.right.is_some()
    }

    /// Whether the field was removed (only in left).
    pub fn is_removed(&self) -> bool {
        self.left.is_some() && self.right.is_none()
    }

    /// Whether the field was modified (in both but different).
    pub fn is_modified(&self) -> bool {
        self.left.is_some() && self.right.is_some()
    }
}

impl fmt::Display for FieldDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => write!(f, "{}: {l} -> {r}", self.field_name),
            (Some(l), None) => write!(f, "{}: {l} -> <removed>", self.field_name),
            (None, Some(r)) => write!(f, "{}: <added> -> {r}", self.field_name),
            (None, None) => write!(f, "{}: <none>", self.field_name),
        }
    }
}

// ── Enum Descriptor ──────────────────────────────────────────

/// Runtime enum descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDescriptor {
    /// Fully-qualified name.
    pub full_name: String,
    /// Values: (name, number).
    pub values: Vec<(String, i32)>,
}

impl EnumDescriptor {
    pub fn new(full_name: impl Into<String>) -> Self {
        Self {
            full_name: full_name.into(),
            values: Vec::new(),
        }
    }

    /// Add a value.
    pub fn add_value(&mut self, name: impl Into<String>, number: i32) {
        self.values.push((name.into(), number));
    }

    /// Find name by number.
    pub fn name_by_number(&self, number: i32) -> Option<&str> {
        self.values.iter()
            .find(|(_, n)| *n == number)
            .map(|(name, _)| name.as_str())
    }

    /// Find number by name.
    pub fn number_by_name(&self, name: &str) -> Option<i32> {
        self.values.iter()
            .find(|(n, _)| n == name)
            .map(|(_, num)| *num)
    }
}

// ── Descriptor Registry ──────────────────────────────────────

/// Registry for looking up descriptors at runtime.
#[derive(Debug, Clone, Default)]
pub struct DescriptorRegistry {
    messages: HashMap<String, MessageDescriptor>,
    enums: HashMap<String, EnumDescriptor>,
}

impl DescriptorRegistry {
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
            enums: HashMap::new(),
        }
    }

    /// Register a message descriptor.
    pub fn register_message(&mut self, desc: MessageDescriptor) {
        self.messages.insert(desc.full_name.clone(), desc);
    }

    /// Register an enum descriptor.
    pub fn register_enum(&mut self, desc: EnumDescriptor) {
        self.enums.insert(desc.full_name.clone(), desc);
    }

    /// Look up a message descriptor.
    pub fn get_message(&self, name: &str) -> Option<&MessageDescriptor> {
        self.messages.get(name)
    }

    /// Look up an enum descriptor.
    pub fn get_enum(&self, name: &str) -> Option<&EnumDescriptor> {
        self.enums.get(name)
    }

    /// All registered message type names (sorted).
    pub fn message_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.messages.keys().cloned().collect();
        names.sort();
        names
    }

    /// All registered enum type names (sorted).
    pub fn enum_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.enums.keys().cloned().collect();
        names.sort();
        names
    }

    /// Create a new DynMessage from a registered descriptor.
    pub fn new_message(&self, type_name: &str) -> Option<DynMessage> {
        self.messages.get(type_name).map(|desc| desc.new_message())
    }

    /// Total registered types.
    pub fn type_count(&self) -> usize {
        self.messages.len() + self.enums.len()
    }

    /// Validate that all message field references resolve.
    pub fn validate_references(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (msg_name, desc) in &self.messages {
            for field in &desc.fields {
                match &field.kind {
                    FieldKind::Message(ref_name) => {
                        if !self.messages.contains_key(ref_name) {
                            errors.push(format!(
                                "{msg_name}.{}: unresolved message type '{ref_name}'",
                                field.name
                            ));
                        }
                    }
                    FieldKind::Enum(ref_name) => {
                        if !self.enums.contains_key(ref_name) {
                            errors.push(format!(
                                "{msg_name}.{}: unresolved enum type '{ref_name}'",
                                field.name
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
        errors.sort();
        errors
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn person_descriptor() -> MessageDescriptor {
        let mut desc = MessageDescriptor::new("example.Person");
        desc.add_field(FieldDescriptor::new("name", 1, FieldKind::String));
        desc.add_field(FieldDescriptor::new("age", 2, FieldKind::Int32));
        desc.add_field(FieldDescriptor::new("email", 3, FieldKind::String));
        desc.add_field(
            FieldDescriptor::new("tags", 4, FieldKind::String)
                .with_cardinality(Cardinality::Repeated),
        );
        desc
    }

    #[test]
    fn field_kind_display() {
        assert_eq!(FieldKind::Int32.to_string(), "int32");
        assert_eq!(FieldKind::String.to_string(), "string");
        assert_eq!(FieldKind::Message("Foo".into()).to_string(), "message(Foo)");
        assert_eq!(FieldKind::Enum("Bar".into()).to_string(), "enum(Bar)");
    }

    #[test]
    fn dyn_value_defaults() {
        assert!(DynValue::Int32(0).is_default());
        assert!(DynValue::String(String::new()).is_default());
        assert!(DynValue::Bool(false).is_default());
        assert!(DynValue::Null.is_default());
        assert!(!DynValue::Int32(1).is_default());
        assert!(!DynValue::String("hi".into()).is_default());
    }

    #[test]
    fn dyn_value_default_for_kind() {
        assert_eq!(DynValue::default_for(&FieldKind::Int32), DynValue::Int32(0));
        assert_eq!(DynValue::default_for(&FieldKind::String), DynValue::String(String::new()));
        assert_eq!(DynValue::default_for(&FieldKind::Bool), DynValue::Bool(false));
        assert_eq!(DynValue::default_for(&FieldKind::Float), DynValue::Float(0.0));
    }

    #[test]
    fn dyn_value_accessors() {
        assert_eq!(DynValue::Int32(42).as_i32(), Some(42));
        assert_eq!(DynValue::Int64(100).as_i64(), Some(100));
        assert_eq!(DynValue::Uint32(5).as_u32(), Some(5));
        assert_eq!(DynValue::Uint64(9).as_u64(), Some(9));
        assert_eq!(DynValue::Bool(true).as_bool(), Some(true));
        assert_eq!(DynValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(DynValue::Bytes(vec![1]).as_bytes(), Some([1].as_slice()));
        assert_eq!(DynValue::Enum(2).as_enum(), Some(2));
    }

    #[test]
    fn dyn_value_type_mismatch() {
        assert_eq!(DynValue::Int32(42).as_str(), None);
        assert_eq!(DynValue::String("hi".into()).as_i32(), None);
        assert_eq!(DynValue::Bool(true).as_f64(), None);
    }

    #[test]
    fn dyn_value_display() {
        assert_eq!(DynValue::Null.to_string(), "null");
        assert_eq!(DynValue::Int32(42).to_string(), "42");
        assert_eq!(DynValue::Bool(true).to_string(), "true");
    }

    #[test]
    fn field_descriptor_default() {
        let fd = FieldDescriptor::new("count", 1, FieldKind::Int32);
        assert_eq!(fd.default_value(), DynValue::Int32(0));

        let fd = FieldDescriptor::new("tags", 2, FieldKind::String)
            .with_cardinality(Cardinality::Repeated);
        assert_eq!(fd.default_value(), DynValue::List(Vec::new()));
    }

    #[test]
    fn field_descriptor_builder() {
        let fd = FieldDescriptor::new("old_field", 5, FieldKind::String)
            .with_json_name("oldField")
            .with_deprecated(true);
        assert_eq!(fd.json_name, "oldField");
        assert!(fd.deprecated);
    }

    #[test]
    fn field_descriptor_map() {
        let fd = FieldDescriptor::new("attrs", 6, FieldKind::String)
            .with_map_key(FieldKind::String);
        assert_eq!(fd.cardinality, Cardinality::Map);
        assert_eq!(fd.map_key_kind, Some(FieldKind::String));
    }

    #[test]
    fn message_descriptor_lookup() {
        let desc = person_descriptor();
        assert_eq!(desc.field_count(), 4);
        assert!(desc.field_by_name("name").is_some());
        assert!(desc.field_by_number(2).is_some());
        assert_eq!(desc.field_by_number(2).unwrap().name, "age");
        assert!(desc.field_by_name("missing").is_none());
    }

    #[test]
    fn message_descriptor_field_names() {
        let desc = person_descriptor();
        let names = desc.field_names();
        assert_eq!(names, vec!["name", "age", "email", "tags"]);
    }

    #[test]
    fn message_descriptor_new_message() {
        let desc = person_descriptor();
        let msg = desc.new_message();
        assert_eq!(msg.type_name, "example.Person");
        assert_eq!(msg.field_count(), 4);
    }

    #[test]
    fn dyn_message_set_get() {
        let mut msg = DynMessage::new("test");
        msg.set("name", DynValue::String("Alice".into()));
        msg.set("age", DynValue::Int32(30));

        assert_eq!(msg.get("name").unwrap().as_str(), Some("Alice"));
        assert_eq!(msg.get("age").unwrap().as_i32(), Some(30));
        assert!(msg.get("missing").is_none());
    }

    #[test]
    fn dyn_message_has_field() {
        let mut msg = DynMessage::new("test");
        msg.set("name", DynValue::String("Alice".into()));
        msg.set("empty", DynValue::String(String::new()));

        assert!(msg.has_field("name"));
        assert!(!msg.has_field("empty")); // default string is ""
        assert!(!msg.has_field("missing"));
    }

    #[test]
    fn dyn_message_remove() {
        let mut msg = DynMessage::new("test");
        msg.set("x", DynValue::Int32(1));
        assert!(msg.remove("x").is_some());
        assert!(msg.get("x").is_none());
        assert!(msg.remove("x").is_none());
    }

    #[test]
    fn dyn_message_clear() {
        let mut msg = DynMessage::new("test");
        msg.set("a", DynValue::Int32(1));
        msg.set("b", DynValue::Int32(2));
        msg.clear();
        assert_eq!(msg.field_count(), 0);
    }

    #[test]
    fn dyn_message_field_names_sorted() {
        let mut msg = DynMessage::new("test");
        msg.set("zebra", DynValue::Int32(1));
        msg.set("alpha", DynValue::Int32(2));
        let names = msg.field_names();
        assert_eq!(names, vec!["alpha", "zebra"]);
    }

    #[test]
    fn dyn_message_get_or_default() {
        let desc = person_descriptor();
        let msg = DynMessage::new("example.Person");
        let val = msg.get_or_default("age", &desc);
        assert_eq!(val, DynValue::Int32(0));
    }

    #[test]
    fn dyn_message_merge_scalar() {
        let mut a = DynMessage::new("test");
        a.set("name", DynValue::String("Alice".into()));
        a.set("age", DynValue::Int32(30));

        let mut b = DynMessage::new("test");
        b.set("name", DynValue::String("Bob".into()));

        a.merge(&b);
        assert_eq!(a.get("name").unwrap().as_str(), Some("Bob"));
        assert_eq!(a.get("age").unwrap().as_i32(), Some(30));
    }

    #[test]
    fn dyn_message_merge_list() {
        let mut a = DynMessage::new("test");
        a.set("tags", DynValue::List(vec![DynValue::String("x".into())]));

        let mut b = DynMessage::new("test");
        b.set("tags", DynValue::List(vec![DynValue::String("y".into())]));

        a.merge(&b);
        let tags = a.get("tags").unwrap().as_list().unwrap();
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn dyn_message_merge_submessage() {
        let mut inner_a = DynMessage::new("Inner");
        inner_a.set("x", DynValue::Int32(1));
        inner_a.set("y", DynValue::Int32(2));

        let mut inner_b = DynMessage::new("Inner");
        inner_b.set("y", DynValue::Int32(99));
        inner_b.set("z", DynValue::Int32(3));

        let mut a = DynMessage::new("Outer");
        a.set("inner", DynValue::Message(Box::new(inner_a)));

        let mut b = DynMessage::new("Outer");
        b.set("inner", DynValue::Message(Box::new(inner_b)));

        a.merge(&b);
        let inner = a.get("inner").unwrap().as_message().unwrap();
        assert_eq!(inner.get("x").unwrap().as_i32(), Some(1));
        assert_eq!(inner.get("y").unwrap().as_i32(), Some(99));
        assert_eq!(inner.get("z").unwrap().as_i32(), Some(3));
    }

    #[test]
    fn dyn_message_diff_equal() {
        let mut a = DynMessage::new("test");
        a.set("x", DynValue::Int32(1));
        let b = a.clone();
        let diffs = a.diff(&b);
        assert!(diffs.is_empty());
    }

    #[test]
    fn dyn_message_diff_modified() {
        let mut a = DynMessage::new("test");
        a.set("x", DynValue::Int32(1));
        let mut b = DynMessage::new("test");
        b.set("x", DynValue::Int32(2));

        let diffs = a.diff(&b);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].is_modified());
        assert_eq!(diffs[0].field_name, "x");
    }

    #[test]
    fn dyn_message_diff_added_removed() {
        let mut a = DynMessage::new("test");
        a.set("x", DynValue::Int32(1));
        let mut b = DynMessage::new("test");
        b.set("y", DynValue::Int32(2));

        let diffs = a.diff(&b);
        assert_eq!(diffs.len(), 2);
        let removed = diffs.iter().find(|d| d.field_name == "x").unwrap();
        assert!(removed.is_removed());
        let added = diffs.iter().find(|d| d.field_name == "y").unwrap();
        assert!(added.is_added());
    }

    #[test]
    fn dyn_message_equals() {
        let mut a = DynMessage::new("test");
        a.set("x", DynValue::Int32(1));
        let b = a.clone();
        assert!(a.equals(&b));

        let mut c = DynMessage::new("test");
        c.set("x", DynValue::Int32(2));
        assert!(!a.equals(&c));
    }

    #[test]
    fn field_diff_display() {
        let diff = FieldDiff {
            field_name: "count".to_string(),
            left: Some(DynValue::Int32(1)),
            right: Some(DynValue::Int32(2)),
        };
        assert_eq!(diff.to_string(), "count: 1 -> 2");
    }

    #[test]
    fn enum_descriptor() {
        let mut ed = EnumDescriptor::new("example.Status");
        ed.add_value("UNKNOWN", 0);
        ed.add_value("ACTIVE", 1);
        ed.add_value("INACTIVE", 2);

        assert_eq!(ed.name_by_number(1), Some("ACTIVE"));
        assert_eq!(ed.number_by_name("INACTIVE"), Some(2));
        assert_eq!(ed.name_by_number(99), None);
        assert_eq!(ed.number_by_name("MISSING"), None);
    }

    #[test]
    fn descriptor_registry_basic() {
        let mut reg = DescriptorRegistry::new();
        reg.register_message(person_descriptor());

        let mut status_enum = EnumDescriptor::new("example.Status");
        status_enum.add_value("UNKNOWN", 0);
        reg.register_enum(status_enum);

        assert_eq!(reg.type_count(), 2);
        assert!(reg.get_message("example.Person").is_some());
        assert!(reg.get_enum("example.Status").is_some());
    }

    #[test]
    fn descriptor_registry_names() {
        let mut reg = DescriptorRegistry::new();
        reg.register_message(MessageDescriptor::new("B"));
        reg.register_message(MessageDescriptor::new("A"));

        let names = reg.message_names();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn descriptor_registry_new_message() {
        let mut reg = DescriptorRegistry::new();
        reg.register_message(person_descriptor());

        let msg = reg.new_message("example.Person").unwrap();
        assert_eq!(msg.type_name, "example.Person");
        assert_eq!(msg.field_count(), 4);
        assert!(reg.new_message("missing").is_none());
    }

    #[test]
    fn descriptor_registry_validate_ok() {
        let mut reg = DescriptorRegistry::new();
        let mut desc = MessageDescriptor::new("Outer");
        desc.add_field(FieldDescriptor::new("inner", 1, FieldKind::Message("Inner".into())));
        reg.register_message(desc);
        reg.register_message(MessageDescriptor::new("Inner"));

        let errors = reg.validate_references();
        assert!(errors.is_empty());
    }

    #[test]
    fn descriptor_registry_validate_missing() {
        let mut reg = DescriptorRegistry::new();
        let mut desc = MessageDescriptor::new("Outer");
        desc.add_field(FieldDescriptor::new("inner", 1, FieldKind::Message("Missing".into())));
        reg.register_message(desc);

        let errors = reg.validate_references();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Missing"));
    }

    #[test]
    fn dyn_value_float_accessors() {
        let f = DynValue::Float(3.14);
        assert!((f.as_f32().unwrap() - 3.14).abs() < 0.01);

        let d = DynValue::Double(2.718);
        assert!((d.as_f64().unwrap() - 2.718).abs() < 0.001);
    }

    #[test]
    fn dyn_value_message_accessor() {
        let inner = DynMessage::new("Inner");
        let val = DynValue::Message(Box::new(inner));
        assert!(val.as_message().is_some());
        assert_eq!(val.as_message().unwrap().type_name, "Inner");
    }

    #[test]
    fn message_descriptor_json_name_lookup() {
        let mut desc = MessageDescriptor::new("Test");
        desc.add_field(
            FieldDescriptor::new("my_field", 1, FieldKind::Int32)
                .with_json_name("myField"),
        );
        assert!(desc.field_by_json_name("myField").is_some());
        assert!(desc.field_by_json_name("my_field").is_none());
    }

    #[test]
    fn enum_names_sorted() {
        let mut reg = DescriptorRegistry::new();
        reg.register_enum(EnumDescriptor::new("Z"));
        reg.register_enum(EnumDescriptor::new("A"));
        assert_eq!(reg.enum_names(), vec!["A", "Z"]);
    }
}
