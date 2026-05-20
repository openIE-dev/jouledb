//! Protocol buffer schema model — message, field, enum, oneof, service definitions.
//!
//! Pure-Rust protobuf schema representation for proto3 syntax. Supports
//! message/field/enum/oneof/service/method definitions, field types (scalar,
//! message, enum), field rules (singular/repeated/map), nested messages,
//! and schema validation.

use std::collections::HashMap;
use std::fmt;

// ── Field Types ──────────────────────────────────────────────

/// Protobuf scalar types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Int32,
    Int64,
    Uint32,
    Uint64,
    Sint32,
    Sint64,
    Fixed32,
    Fixed64,
    Sfixed32,
    Sfixed64,
    Float,
    Double,
    Bool,
    String,
    Bytes,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
            Self::Sint32 => "sint32",
            Self::Sint64 => "sint64",
            Self::Fixed32 => "fixed32",
            Self::Fixed64 => "fixed64",
            Self::Sfixed32 => "sfixed32",
            Self::Sfixed64 => "sfixed64",
            Self::Float => "float",
            Self::Double => "double",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Bytes => "bytes",
        };
        f.write_str(s)
    }
}

impl ScalarType {
    /// Parse a scalar type name from proto syntax.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "int32" => Some(Self::Int32),
            "int64" => Some(Self::Int64),
            "uint32" => Some(Self::Uint32),
            "uint64" => Some(Self::Uint64),
            "sint32" => Some(Self::Sint32),
            "sint64" => Some(Self::Sint64),
            "fixed32" => Some(Self::Fixed32),
            "fixed64" => Some(Self::Fixed64),
            "sfixed32" => Some(Self::Sfixed32),
            "sfixed64" => Some(Self::Sfixed64),
            "float" => Some(Self::Float),
            "double" => Some(Self::Double),
            "bool" => Some(Self::Bool),
            "string" => Some(Self::String),
            "bytes" => Some(Self::Bytes),
            _ => None,
        }
    }

    /// Wire type used to encode this scalar.
    pub fn wire_type(&self) -> WireFormat {
        match self {
            Self::Int32 | Self::Int64 | Self::Uint32 | Self::Uint64
            | Self::Sint32 | Self::Sint64 | Self::Bool => WireFormat::Varint,
            Self::Fixed64 | Self::Sfixed64 | Self::Double => WireFormat::Fixed64,
            Self::Fixed32 | Self::Sfixed32 | Self::Float => WireFormat::Fixed32,
            Self::String | Self::Bytes => WireFormat::LengthDelimited,
        }
    }
}

/// Wire format categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    Varint,
    Fixed32,
    Fixed64,
    LengthDelimited,
}

impl WireFormat {
    /// Wire type number (0, 1, 2, 5).
    pub fn number(self) -> u8 {
        match self {
            Self::Varint => 0,
            Self::Fixed64 => 1,
            Self::LengthDelimited => 2,
            Self::Fixed32 => 5,
        }
    }
}

/// A field's type — scalar, message reference, or enum reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    Scalar(ScalarType),
    /// Reference to a message type by fully-qualified name.
    Message(String),
    /// Reference to an enum type by fully-qualified name.
    Enum(String),
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(s) => write!(f, "{s}"),
            Self::Message(name) => write!(f, "{name}"),
            Self::Enum(name) => write!(f, "{name}"),
        }
    }
}

// ── Field Rule ───────────────────────────────────────────────

/// Field cardinality rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRule {
    /// Default in proto3 — a singular field.
    Singular,
    /// Repeated field (list).
    Repeated,
    /// Map field (key → value).
    Map,
}

// ── Field Definition ─────────────────────────────────────────

/// A field within a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
    /// Field name.
    pub name: String,
    /// Field number (tag).
    pub number: u32,
    /// Field type.
    pub field_type: FieldType,
    /// Cardinality.
    pub rule: FieldRule,
    /// For map fields — the key type.
    pub map_key_type: Option<ScalarType>,
    /// Whether this field belongs to a oneof.
    pub oneof_index: Option<usize>,
    /// Field options/annotations.
    pub options: HashMap<String, String>,
    /// Deprecated flag.
    pub deprecated: bool,
    /// JSON name override (if different from field name).
    pub json_name: Option<String>,
}

impl FieldDef {
    /// Create a new scalar field.
    pub fn scalar(name: impl Into<String>, number: u32, scalar: ScalarType) -> Self {
        Self {
            name: name.into(),
            number,
            field_type: FieldType::Scalar(scalar),
            rule: FieldRule::Singular,
            map_key_type: None,
            oneof_index: None,
            options: HashMap::new(),
            deprecated: false,
            json_name: None,
        }
    }

    /// Create a new message-typed field.
    pub fn message(name: impl Into<String>, number: u32, msg_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            number,
            field_type: FieldType::Message(msg_type.into()),
            rule: FieldRule::Singular,
            map_key_type: None,
            oneof_index: None,
            options: HashMap::new(),
            deprecated: false,
            json_name: None,
        }
    }

    /// Create a repeated field.
    pub fn repeated(name: impl Into<String>, number: u32, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            number,
            field_type,
            rule: FieldRule::Repeated,
            map_key_type: None,
            oneof_index: None,
            options: HashMap::new(),
            deprecated: false,
            json_name: None,
        }
    }

    /// Create a map field.
    pub fn map(
        name: impl Into<String>,
        number: u32,
        key_type: ScalarType,
        value_type: FieldType,
    ) -> Self {
        Self {
            name: name.into(),
            number,
            field_type: value_type,
            rule: FieldRule::Map,
            map_key_type: Some(key_type),
            oneof_index: None,
            options: HashMap::new(),
            deprecated: false,
            json_name: None,
        }
    }

    /// Set deprecated.
    pub fn with_deprecated(mut self, deprecated: bool) -> Self {
        self.deprecated = deprecated;
        self
    }

    /// Set JSON name.
    pub fn with_json_name(mut self, json_name: impl Into<String>) -> Self {
        self.json_name = Some(json_name.into());
        self
    }

    /// Add an option.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Effective JSON name (json_name or field name).
    pub fn effective_json_name(&self) -> &str {
        self.json_name.as_deref().unwrap_or(&self.name)
    }

    /// Wire format for this field.
    pub fn wire_format(&self) -> WireFormat {
        match &self.field_type {
            FieldType::Scalar(s) => s.wire_type(),
            FieldType::Message(_) => WireFormat::LengthDelimited,
            FieldType::Enum(_) => WireFormat::Varint,
        }
    }
}

// ── Oneof Definition ─────────────────────────────────────────

/// A oneof group within a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OneofDef {
    /// Oneof name.
    pub name: String,
    /// Field numbers belonging to this oneof.
    pub field_numbers: Vec<u32>,
}

impl OneofDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_numbers: Vec::new(),
        }
    }

    /// Add a field number to this oneof.
    pub fn add_field(&mut self, number: u32) {
        if !self.field_numbers.contains(&number) {
            self.field_numbers.push(number);
        }
    }
}

// ── Enum Value ───────────────────────────────────────────────

/// A single enum value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumValueDef {
    /// Value name.
    pub name: String,
    /// Numeric value.
    pub number: i32,
    /// Deprecated flag.
    pub deprecated: bool,
}

impl EnumValueDef {
    pub fn new(name: impl Into<String>, number: i32) -> Self {
        Self {
            name: name.into(),
            number,
            deprecated: false,
        }
    }
}

// ── Enum Definition ──────────────────────────────────────────

/// A protobuf enum definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDef {
    /// Enum name.
    pub name: String,
    /// Values in declaration order.
    pub values: Vec<EnumValueDef>,
    /// Whether aliases are allowed.
    pub allow_alias: bool,
    /// Containing message name (empty for top-level).
    pub parent: String,
}

impl EnumDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            allow_alias: false,
            parent: String::new(),
        }
    }

    /// Add a value.
    pub fn add_value(&mut self, name: impl Into<String>, number: i32) {
        self.values.push(EnumValueDef::new(name, number));
    }

    /// Fully-qualified name.
    pub fn full_name(&self) -> String {
        if self.parent.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.parent, self.name)
        }
    }

    /// Find value by name.
    pub fn value_by_name(&self, name: &str) -> Option<&EnumValueDef> {
        self.values.iter().find(|v| v.name == name)
    }

    /// Find value by number.
    pub fn value_by_number(&self, number: i32) -> Option<&EnumValueDef> {
        self.values.iter().find(|v| v.number == number)
    }

    /// Validate: proto3 requires first value == 0, no duplicate numbers unless alias.
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.values.is_empty() {
            return Err(SchemaError::EmptyEnum(self.name.clone()));
        }
        if self.values[0].number != 0 {
            return Err(SchemaError::EnumZeroValueRequired(self.name.clone()));
        }
        if !self.allow_alias {
            let mut seen = HashMap::new();
            for v in &self.values {
                if let Some(prev) = seen.insert(v.number, &v.name) {
                    return Err(SchemaError::DuplicateEnumNumber {
                        enum_name: self.name.clone(),
                        number: v.number,
                        first: prev.clone(),
                        second: v.name.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ── Message Definition ───────────────────────────────────────

/// A protobuf message definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDef {
    /// Message name.
    pub name: String,
    /// Fields in declaration order.
    pub fields: Vec<FieldDef>,
    /// Oneof definitions.
    pub oneofs: Vec<OneofDef>,
    /// Nested message definitions.
    pub nested_messages: Vec<MessageDef>,
    /// Nested enum definitions.
    pub nested_enums: Vec<EnumDef>,
    /// Containing message name (empty for top-level).
    pub parent: String,
    /// Message-level options.
    pub options: HashMap<String, String>,
    /// Reserved field numbers.
    pub reserved_numbers: Vec<u32>,
    /// Reserved field names.
    pub reserved_names: Vec<String>,
}

impl MessageDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            oneofs: Vec::new(),
            nested_messages: Vec::new(),
            nested_enums: Vec::new(),
            parent: String::new(),
            options: HashMap::new(),
            reserved_numbers: Vec::new(),
            reserved_names: Vec::new(),
        }
    }

    /// Add a field.
    pub fn add_field(&mut self, field: FieldDef) {
        self.fields.push(field);
    }

    /// Add a oneof.
    pub fn add_oneof(&mut self, oneof: OneofDef) {
        self.oneofs.push(oneof);
    }

    /// Add a nested message.
    pub fn add_nested_message(&mut self, mut msg: MessageDef) {
        msg.parent = self.full_name();
        self.nested_messages.push(msg);
    }

    /// Add a nested enum.
    pub fn add_nested_enum(&mut self, mut e: EnumDef) {
        e.parent = self.full_name();
        self.nested_enums.push(e);
    }

    /// Fully-qualified name.
    pub fn full_name(&self) -> String {
        if self.parent.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.parent, self.name)
        }
    }

    /// Lookup field by number.
    pub fn field_by_number(&self, number: u32) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.number == number)
    }

    /// Lookup field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// All field numbers in use.
    pub fn used_field_numbers(&self) -> Vec<u32> {
        self.fields.iter().map(|f| f.number).collect()
    }

    /// Reserve a field number.
    pub fn reserve_number(&mut self, n: u32) {
        if !self.reserved_numbers.contains(&n) {
            self.reserved_numbers.push(n);
        }
    }

    /// Reserve a field name.
    pub fn reserve_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.reserved_names.contains(&name) {
            self.reserved_names.push(name);
        }
    }

    /// Validate the message definition.
    pub fn validate(&self) -> Result<(), SchemaError> {
        // Check for duplicate field numbers.
        let mut seen_numbers: HashMap<u32, &str> = HashMap::new();
        for field in &self.fields {
            if let Some(prev) = seen_numbers.insert(field.number, &field.name) {
                return Err(SchemaError::DuplicateFieldNumber {
                    message: self.name.clone(),
                    number: field.number,
                    first: prev.to_string(),
                    second: field.name.clone(),
                });
            }
        }

        // Check for duplicate field names.
        let mut seen_names: HashMap<&str, u32> = HashMap::new();
        for field in &self.fields {
            if let Some(prev) = seen_names.insert(&field.name, field.number) {
                return Err(SchemaError::DuplicateFieldName {
                    message: self.name.clone(),
                    name: field.name.clone(),
                    first_number: prev,
                    second_number: field.number,
                });
            }
        }

        // Check reserved numbers not used.
        for field in &self.fields {
            if self.reserved_numbers.contains(&field.number) {
                return Err(SchemaError::ReservedFieldNumber {
                    message: self.name.clone(),
                    number: field.number,
                    field_name: field.name.clone(),
                });
            }
        }

        // Check reserved names not used.
        for field in &self.fields {
            if self.reserved_names.contains(&field.name) {
                return Err(SchemaError::ReservedFieldName {
                    message: self.name.clone(),
                    name: field.name.clone(),
                });
            }
        }

        // Field numbers must be 1..=536870911 (2^29 - 1) and not in 19000..=19999.
        for field in &self.fields {
            if field.number == 0 || field.number > 536_870_911 {
                return Err(SchemaError::InvalidFieldNumber {
                    message: self.name.clone(),
                    field_name: field.name.clone(),
                    number: field.number,
                });
            }
            if (19000..=19999).contains(&field.number) {
                return Err(SchemaError::ReservedRange {
                    message: self.name.clone(),
                    field_name: field.name.clone(),
                    number: field.number,
                });
            }
        }

        // Validate nested.
        for nested in &self.nested_messages {
            nested.validate()?;
        }
        for nested in &self.nested_enums {
            nested.validate()?;
        }

        Ok(())
    }
}

// ── Method Definition ────────────────────────────────────────

/// Streaming mode for an RPC method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    Unary,
    ServerStreaming,
    ClientStreaming,
    BidiStreaming,
}

/// An RPC method definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDef {
    /// Method name.
    pub name: String,
    /// Input message type (fully-qualified).
    pub input_type: String,
    /// Output message type (fully-qualified).
    pub output_type: String,
    /// Whether client streams input.
    pub client_streaming: bool,
    /// Whether server streams output.
    pub server_streaming: bool,
    /// Method options.
    pub options: HashMap<String, String>,
}

impl MethodDef {
    pub fn new(
        name: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            input_type: input_type.into(),
            output_type: output_type.into(),
            client_streaming: false,
            server_streaming: false,
            options: HashMap::new(),
        }
    }

    /// Create a server-streaming method.
    pub fn server_streaming(
        name: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
    ) -> Self {
        Self {
            server_streaming: true,
            ..Self::new(name, input_type, output_type)
        }
    }

    /// Create a client-streaming method.
    pub fn client_streaming(
        name: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
    ) -> Self {
        Self {
            client_streaming: true,
            ..Self::new(name, input_type, output_type)
        }
    }

    /// Create a bidi-streaming method.
    pub fn bidi_streaming(
        name: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
    ) -> Self {
        Self {
            client_streaming: true,
            server_streaming: true,
            ..Self::new(name, input_type, output_type)
        }
    }

    /// Streaming mode.
    pub fn streaming_mode(&self) -> StreamingMode {
        match (self.client_streaming, self.server_streaming) {
            (false, false) => StreamingMode::Unary,
            (false, true) => StreamingMode::ServerStreaming,
            (true, false) => StreamingMode::ClientStreaming,
            (true, true) => StreamingMode::BidiStreaming,
        }
    }
}

// ── Service Definition ───────────────────────────────────────

/// A service definition containing RPC methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceDef {
    /// Service name.
    pub name: String,
    /// RPC methods.
    pub methods: Vec<MethodDef>,
    /// Service options.
    pub options: HashMap<String, String>,
}

impl ServiceDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            methods: Vec::new(),
            options: HashMap::new(),
        }
    }

    /// Add a method.
    pub fn add_method(&mut self, method: MethodDef) {
        self.methods.push(method);
    }

    /// Find method by name.
    pub fn method_by_name(&self, name: &str) -> Option<&MethodDef> {
        self.methods.iter().find(|m| m.name == name)
    }
}

// ── Proto Syntax ─────────────────────────────────────────────

/// Protobuf syntax version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtoSyntax {
    Proto2,
    Proto3,
}

impl fmt::Display for ProtoSyntax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proto2 => f.write_str("proto2"),
            Self::Proto3 => f.write_str("proto3"),
        }
    }
}

// ── File Definition ──────────────────────────────────────────

/// A complete proto file schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtoFile {
    /// Syntax version.
    pub syntax: ProtoSyntax,
    /// Package name.
    pub package: String,
    /// Import paths.
    pub imports: Vec<String>,
    /// Top-level message definitions.
    pub messages: Vec<MessageDef>,
    /// Top-level enum definitions.
    pub enums: Vec<EnumDef>,
    /// Service definitions.
    pub services: Vec<ServiceDef>,
    /// File-level options.
    pub options: HashMap<String, String>,
}

impl ProtoFile {
    pub fn new(package: impl Into<String>) -> Self {
        Self {
            syntax: ProtoSyntax::Proto3,
            package: package.into(),
            imports: Vec::new(),
            messages: Vec::new(),
            enums: Vec::new(),
            services: Vec::new(),
            options: HashMap::new(),
        }
    }

    /// Add a top-level message.
    pub fn add_message(&mut self, msg: MessageDef) {
        self.messages.push(msg);
    }

    /// Add a top-level enum.
    pub fn add_enum(&mut self, e: EnumDef) {
        self.enums.push(e);
    }

    /// Add a service.
    pub fn add_service(&mut self, svc: ServiceDef) {
        self.services.push(svc);
    }

    /// Add an import.
    pub fn add_import(&mut self, path: impl Into<String>) {
        self.imports.push(path.into());
    }

    /// Find message by name (top-level only).
    pub fn message_by_name(&self, name: &str) -> Option<&MessageDef> {
        self.messages.iter().find(|m| m.name == name)
    }

    /// Find enum by name (top-level only).
    pub fn enum_by_name(&self, name: &str) -> Option<&EnumDef> {
        self.enums.iter().find(|e| e.name == name)
    }

    /// Find service by name.
    pub fn service_by_name(&self, name: &str) -> Option<&ServiceDef> {
        self.services.iter().find(|s| s.name == name)
    }

    /// Validate all definitions.
    pub fn validate(&self) -> Result<(), SchemaError> {
        for msg in &self.messages {
            msg.validate()?;
        }
        for e in &self.enums {
            e.validate()?;
        }
        // Validate service method type references exist.
        for svc in &self.services {
            for method in &svc.methods {
                if !self.type_exists(&method.input_type) {
                    return Err(SchemaError::UnresolvedType {
                        context: format!("{}.{}", svc.name, method.name),
                        type_name: method.input_type.clone(),
                    });
                }
                if !self.type_exists(&method.output_type) {
                    return Err(SchemaError::UnresolvedType {
                        context: format!("{}.{}", svc.name, method.name),
                        type_name: method.output_type.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check whether a type name exists (messages or enums, top-level or nested).
    fn type_exists(&self, name: &str) -> bool {
        if self.messages.iter().any(|m| m.name == name || m.full_name() == name) {
            return true;
        }
        if self.enums.iter().any(|e| e.name == name || e.full_name() == name) {
            return true;
        }
        // Check nested in messages.
        for msg in &self.messages {
            if Self::type_exists_in_message(msg, name) {
                return true;
            }
        }
        false
    }

    fn type_exists_in_message(msg: &MessageDef, name: &str) -> bool {
        for nested in &msg.nested_messages {
            if nested.name == name || nested.full_name() == name {
                return true;
            }
            if Self::type_exists_in_message(nested, name) {
                return true;
            }
        }
        for nested in &msg.nested_enums {
            if nested.name == name || nested.full_name() == name {
                return true;
            }
        }
        false
    }

    /// Collect all type names defined in this file.
    pub fn all_type_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for msg in &self.messages {
            Self::collect_message_names(msg, &mut names);
        }
        for e in &self.enums {
            names.push(e.full_name());
        }
        names
    }

    fn collect_message_names(msg: &MessageDef, names: &mut Vec<String>) {
        names.push(msg.full_name());
        for nested in &msg.nested_messages {
            Self::collect_message_names(nested, names);
        }
        for nested in &msg.nested_enums {
            names.push(nested.full_name());
        }
    }

    /// Total number of fields across all messages.
    pub fn total_field_count(&self) -> usize {
        self.messages.iter().map(|m| Self::count_fields(m)).sum()
    }

    fn count_fields(msg: &MessageDef) -> usize {
        let own = msg.fields.len();
        let nested: usize = msg.nested_messages.iter().map(|m| Self::count_fields(m)).sum();
        own + nested
    }
}

// ── Errors ───────────────────────────────────────────────────

/// Schema validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    DuplicateFieldNumber {
        message: String,
        number: u32,
        first: String,
        second: String,
    },
    DuplicateFieldName {
        message: String,
        name: String,
        first_number: u32,
        second_number: u32,
    },
    ReservedFieldNumber {
        message: String,
        number: u32,
        field_name: String,
    },
    ReservedFieldName {
        message: String,
        name: String,
    },
    InvalidFieldNumber {
        message: String,
        field_name: String,
        number: u32,
    },
    ReservedRange {
        message: String,
        field_name: String,
        number: u32,
    },
    EmptyEnum(String),
    EnumZeroValueRequired(String),
    DuplicateEnumNumber {
        enum_name: String,
        number: i32,
        first: String,
        second: String,
    },
    UnresolvedType {
        context: String,
        type_name: String,
    },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateFieldNumber { message, number, first, second } => {
                write!(f, "message {message}: duplicate field number {number} ({first}, {second})")
            }
            Self::DuplicateFieldName { message, name, first_number, second_number } => {
                write!(f, "message {message}: duplicate field name '{name}' (numbers {first_number}, {second_number})")
            }
            Self::ReservedFieldNumber { message, number, field_name } => {
                write!(f, "message {message}: field '{field_name}' uses reserved number {number}")
            }
            Self::ReservedFieldName { message, name } => {
                write!(f, "message {message}: field name '{name}' is reserved")
            }
            Self::InvalidFieldNumber { message, field_name, number } => {
                write!(f, "message {message}: field '{field_name}' has invalid number {number}")
            }
            Self::ReservedRange { message, field_name, number } => {
                write!(f, "message {message}: field '{field_name}' uses reserved range number {number}")
            }
            Self::EmptyEnum(name) => write!(f, "enum {name} has no values"),
            Self::EnumZeroValueRequired(name) => {
                write!(f, "enum {name}: first value must be 0 in proto3")
            }
            Self::DuplicateEnumNumber { enum_name, number, first, second } => {
                write!(f, "enum {enum_name}: duplicate number {number} ({first}, {second})")
            }
            Self::UnresolvedType { context, type_name } => {
                write!(f, "{context}: unresolved type '{type_name}'")
            }
        }
    }
}

impl std::error::Error for SchemaError {}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_type_from_name() {
        assert_eq!(ScalarType::from_name("int32"), Some(ScalarType::Int32));
        assert_eq!(ScalarType::from_name("string"), Some(ScalarType::String));
        assert_eq!(ScalarType::from_name("bytes"), Some(ScalarType::Bytes));
        assert_eq!(ScalarType::from_name("float"), Some(ScalarType::Float));
        assert_eq!(ScalarType::from_name("double"), Some(ScalarType::Double));
        assert_eq!(ScalarType::from_name("bool"), Some(ScalarType::Bool));
        assert_eq!(ScalarType::from_name("unknown"), None);
    }

    #[test]
    fn scalar_type_display() {
        assert_eq!(ScalarType::Int32.to_string(), "int32");
        assert_eq!(ScalarType::Double.to_string(), "double");
        assert_eq!(ScalarType::Bytes.to_string(), "bytes");
    }

    #[test]
    fn scalar_wire_type() {
        assert_eq!(ScalarType::Int32.wire_type(), WireFormat::Varint);
        assert_eq!(ScalarType::Fixed32.wire_type(), WireFormat::Fixed32);
        assert_eq!(ScalarType::Fixed64.wire_type(), WireFormat::Fixed64);
        assert_eq!(ScalarType::String.wire_type(), WireFormat::LengthDelimited);
        assert_eq!(ScalarType::Double.wire_type(), WireFormat::Fixed64);
        assert_eq!(ScalarType::Float.wire_type(), WireFormat::Fixed32);
    }

    #[test]
    fn wire_format_number() {
        assert_eq!(WireFormat::Varint.number(), 0);
        assert_eq!(WireFormat::Fixed64.number(), 1);
        assert_eq!(WireFormat::LengthDelimited.number(), 2);
        assert_eq!(WireFormat::Fixed32.number(), 5);
    }

    #[test]
    fn field_def_scalar() {
        let f = FieldDef::scalar("age", 1, ScalarType::Int32);
        assert_eq!(f.name, "age");
        assert_eq!(f.number, 1);
        assert_eq!(f.rule, FieldRule::Singular);
        assert_eq!(f.wire_format(), WireFormat::Varint);
    }

    #[test]
    fn field_def_message_type() {
        let f = FieldDef::message("address", 2, "Address");
        assert_eq!(f.wire_format(), WireFormat::LengthDelimited);
        assert_eq!(f.field_type, FieldType::Message("Address".into()));
    }

    #[test]
    fn field_def_repeated() {
        let f = FieldDef::repeated("tags", 5, FieldType::Scalar(ScalarType::String));
        assert_eq!(f.rule, FieldRule::Repeated);
    }

    #[test]
    fn field_def_map() {
        let f = FieldDef::map("attrs", 6, ScalarType::String, FieldType::Scalar(ScalarType::String));
        assert_eq!(f.rule, FieldRule::Map);
        assert_eq!(f.map_key_type, Some(ScalarType::String));
    }

    #[test]
    fn field_json_name() {
        let f = FieldDef::scalar("my_field", 1, ScalarType::Int32)
            .with_json_name("myField");
        assert_eq!(f.effective_json_name(), "myField");
        let f2 = FieldDef::scalar("my_field", 1, ScalarType::Int32);
        assert_eq!(f2.effective_json_name(), "my_field");
    }

    #[test]
    fn field_deprecated_option() {
        let f = FieldDef::scalar("old", 1, ScalarType::Int32)
            .with_deprecated(true)
            .with_option("custom", "value");
        assert!(f.deprecated);
        assert_eq!(f.options.get("custom").unwrap(), "value");
    }

    #[test]
    fn oneof_definition() {
        let mut oneof = OneofDef::new("choice");
        oneof.add_field(3);
        oneof.add_field(4);
        oneof.add_field(3); // duplicate — no-op
        assert_eq!(oneof.field_numbers, vec![3, 4]);
    }

    #[test]
    fn enum_definition_valid() {
        let mut e = EnumDef::new("Status");
        e.add_value("UNKNOWN", 0);
        e.add_value("ACTIVE", 1);
        e.add_value("INACTIVE", 2);
        assert!(e.validate().is_ok());
        assert_eq!(e.value_by_name("ACTIVE").unwrap().number, 1);
        assert_eq!(e.value_by_number(2).unwrap().name, "INACTIVE");
    }

    #[test]
    fn enum_must_start_with_zero() {
        let mut e = EnumDef::new("Bad");
        e.add_value("FIRST", 1);
        let err = e.validate().unwrap_err();
        assert!(matches!(err, SchemaError::EnumZeroValueRequired(_)));
    }

    #[test]
    fn enum_no_duplicate_numbers() {
        let mut e = EnumDef::new("Dup");
        e.add_value("A", 0);
        e.add_value("B", 1);
        e.add_value("C", 1);
        let err = e.validate().unwrap_err();
        assert!(matches!(err, SchemaError::DuplicateEnumNumber { .. }));
    }

    #[test]
    fn enum_alias_allowed() {
        let mut e = EnumDef::new("Aliased");
        e.allow_alias = true;
        e.add_value("A", 0);
        e.add_value("B", 1);
        e.add_value("C", 1); // alias
        assert!(e.validate().is_ok());
    }

    #[test]
    fn message_def_basic() {
        let mut msg = MessageDef::new("Person");
        msg.add_field(FieldDef::scalar("name", 1, ScalarType::String));
        msg.add_field(FieldDef::scalar("age", 2, ScalarType::Int32));
        assert!(msg.validate().is_ok());
        assert_eq!(msg.field_by_number(1).unwrap().name, "name");
        assert_eq!(msg.field_by_name("age").unwrap().number, 2);
        assert_eq!(msg.used_field_numbers(), vec![1, 2]);
    }

    #[test]
    fn message_duplicate_field_number() {
        let mut msg = MessageDef::new("Bad");
        msg.add_field(FieldDef::scalar("a", 1, ScalarType::Int32));
        msg.add_field(FieldDef::scalar("b", 1, ScalarType::String));
        let err = msg.validate().unwrap_err();
        assert!(matches!(err, SchemaError::DuplicateFieldNumber { .. }));
    }

    #[test]
    fn message_duplicate_field_name() {
        let mut msg = MessageDef::new("Bad");
        msg.add_field(FieldDef::scalar("x", 1, ScalarType::Int32));
        msg.add_field(FieldDef::scalar("x", 2, ScalarType::String));
        let err = msg.validate().unwrap_err();
        assert!(matches!(err, SchemaError::DuplicateFieldName { .. }));
    }

    #[test]
    fn message_reserved_number() {
        let mut msg = MessageDef::new("Reserv");
        msg.reserve_number(5);
        msg.add_field(FieldDef::scalar("a", 5, ScalarType::Int32));
        let err = msg.validate().unwrap_err();
        assert!(matches!(err, SchemaError::ReservedFieldNumber { .. }));
    }

    #[test]
    fn message_reserved_name() {
        let mut msg = MessageDef::new("Reserv");
        msg.reserve_name("old_field");
        msg.add_field(FieldDef::scalar("old_field", 1, ScalarType::Int32));
        let err = msg.validate().unwrap_err();
        assert!(matches!(err, SchemaError::ReservedFieldName { .. }));
    }

    #[test]
    fn message_reserved_range_19000() {
        let mut msg = MessageDef::new("Range");
        msg.add_field(FieldDef::scalar("a", 19500, ScalarType::Int32));
        let err = msg.validate().unwrap_err();
        assert!(matches!(err, SchemaError::ReservedRange { .. }));
    }

    #[test]
    fn message_field_number_zero() {
        let mut msg = MessageDef::new("Zero");
        msg.add_field(FieldDef::scalar("a", 0, ScalarType::Int32));
        let err = msg.validate().unwrap_err();
        assert!(matches!(err, SchemaError::InvalidFieldNumber { .. }));
    }

    #[test]
    fn nested_message() {
        let mut inner = MessageDef::new("Inner");
        inner.add_field(FieldDef::scalar("x", 1, ScalarType::Int32));

        let mut outer = MessageDef::new("Outer");
        outer.add_field(FieldDef::scalar("id", 1, ScalarType::Int32));
        outer.add_nested_message(inner);

        assert!(outer.validate().is_ok());
        assert_eq!(outer.nested_messages[0].full_name(), "Outer.Inner");
    }

    #[test]
    fn nested_enum() {
        let mut e = EnumDef::new("Color");
        e.add_value("UNSPECIFIED", 0);
        e.add_value("RED", 1);

        let mut msg = MessageDef::new("Palette");
        msg.add_nested_enum(e);

        assert_eq!(msg.nested_enums[0].full_name(), "Palette.Color");
    }

    #[test]
    fn method_streaming_modes() {
        let unary = MethodDef::new("Get", "Req", "Resp");
        assert_eq!(unary.streaming_mode(), StreamingMode::Unary);

        let ss = MethodDef::server_streaming("List", "Req", "Resp");
        assert_eq!(ss.streaming_mode(), StreamingMode::ServerStreaming);

        let cs = MethodDef::client_streaming("Upload", "Req", "Resp");
        assert_eq!(cs.streaming_mode(), StreamingMode::ClientStreaming);

        let bidi = MethodDef::bidi_streaming("Chat", "Req", "Resp");
        assert_eq!(bidi.streaming_mode(), StreamingMode::BidiStreaming);
    }

    #[test]
    fn service_def() {
        let mut svc = ServiceDef::new("Greeter");
        svc.add_method(MethodDef::new("SayHello", "HelloRequest", "HelloReply"));
        assert!(svc.method_by_name("SayHello").is_some());
        assert!(svc.method_by_name("Missing").is_none());
    }

    #[test]
    fn proto_file_validation() {
        let mut file = ProtoFile::new("example.v1");
        let mut msg = MessageDef::new("HelloRequest");
        msg.add_field(FieldDef::scalar("name", 1, ScalarType::String));
        file.add_message(msg);

        let mut resp = MessageDef::new("HelloReply");
        resp.add_field(FieldDef::scalar("message", 1, ScalarType::String));
        file.add_message(resp);

        let mut svc = ServiceDef::new("Greeter");
        svc.add_method(MethodDef::new("SayHello", "HelloRequest", "HelloReply"));
        file.add_service(svc);

        assert!(file.validate().is_ok());
    }

    #[test]
    fn proto_file_unresolved_type() {
        let mut file = ProtoFile::new("bad");
        let mut svc = ServiceDef::new("S");
        svc.add_method(MethodDef::new("M", "Missing", "AlsoMissing"));
        file.add_service(svc);
        let err = file.validate().unwrap_err();
        assert!(matches!(err, SchemaError::UnresolvedType { .. }));
    }

    #[test]
    fn proto_file_all_type_names() {
        let mut file = ProtoFile::new("test");
        let mut msg = MessageDef::new("Outer");
        let inner = MessageDef::new("Inner");
        msg.add_nested_message(inner);
        file.add_message(msg);

        let mut e = EnumDef::new("Status");
        e.add_value("UNKNOWN", 0);
        file.add_enum(e);

        let names = file.all_type_names();
        assert!(names.contains(&"Outer".to_string()));
        assert!(names.contains(&"Outer.Inner".to_string()));
        assert!(names.contains(&"Status".to_string()));
    }

    #[test]
    fn proto_file_total_fields() {
        let mut file = ProtoFile::new("test");
        let mut msg = MessageDef::new("A");
        msg.add_field(FieldDef::scalar("x", 1, ScalarType::Int32));
        msg.add_field(FieldDef::scalar("y", 2, ScalarType::Int32));

        let mut inner = MessageDef::new("B");
        inner.add_field(FieldDef::scalar("z", 1, ScalarType::String));
        msg.add_nested_message(inner);

        file.add_message(msg);
        assert_eq!(file.total_field_count(), 3);
    }

    #[test]
    fn proto_syntax_display() {
        assert_eq!(ProtoSyntax::Proto2.to_string(), "proto2");
        assert_eq!(ProtoSyntax::Proto3.to_string(), "proto3");
    }

    #[test]
    fn proto_file_imports() {
        let mut file = ProtoFile::new("test");
        file.add_import("google/protobuf/timestamp.proto");
        file.add_import("other.proto");
        assert_eq!(file.imports.len(), 2);
    }

    #[test]
    fn field_type_display() {
        let ft = FieldType::Scalar(ScalarType::Int32);
        assert_eq!(ft.to_string(), "int32");
        let ft2 = FieldType::Message("Foo".into());
        assert_eq!(ft2.to_string(), "Foo");
        let ft3 = FieldType::Enum("Status".into());
        assert_eq!(ft3.to_string(), "Status");
    }

    #[test]
    fn schema_error_display() {
        let err = SchemaError::EmptyEnum("E".into());
        assert_eq!(err.to_string(), "enum E has no values");
    }

    #[test]
    fn enum_empty_error() {
        let e = EnumDef::new("Empty");
        let err = e.validate().unwrap_err();
        assert!(matches!(err, SchemaError::EmptyEnum(_)));
    }

    #[test]
    fn enum_full_name_nested() {
        let mut e = EnumDef::new("Kind");
        e.parent = "Outer.Inner".to_string();
        assert_eq!(e.full_name(), "Outer.Inner.Kind");
    }
}
