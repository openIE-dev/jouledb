//! GraphQL schema definition — types, fields, arguments, enums, interfaces,
//! unions, input types, directives, introspection, schema validation.
//!
//! Pure-Rust replacement for graphql-js schema builder, juniper, async-graphql, etc.

use std::collections::BTreeMap;
use std::fmt;

// ── Type references ───────────────────────────────────────────────

/// A reference to a GraphQL type (possibly wrapped in NonNull/List).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRef {
    Named(String),
    NonNull(Box<TypeRef>),
    List(Box<TypeRef>),
}

impl TypeRef {
    pub fn named(name: &str) -> Self { Self::Named(name.into()) }
    pub fn non_null(inner: TypeRef) -> Self { Self::NonNull(Box::new(inner)) }
    pub fn list(inner: TypeRef) -> Self { Self::List(Box::new(inner)) }

    /// Shorthand for `String!`.
    pub fn non_null_named(name: &str) -> Self {
        Self::NonNull(Box::new(Self::Named(name.into())))
    }

    /// Shorthand for `[T!]!`.
    pub fn non_null_list_of_non_null(name: &str) -> Self {
        Self::NonNull(Box::new(Self::List(Box::new(
            Self::NonNull(Box::new(Self::Named(name.into())))
        ))))
    }

    /// The innermost named type.
    pub fn inner_name(&self) -> &str {
        match self {
            Self::Named(n) => n,
            Self::NonNull(inner) | Self::List(inner) => inner.inner_name(),
        }
    }

    pub fn is_non_null(&self) -> bool { matches!(self, Self::NonNull(_)) }
    pub fn is_list(&self) -> bool { matches!(self, Self::List(_)) }
}

impl fmt::Display for TypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(n) => write!(f, "{n}"),
            Self::NonNull(inner) => write!(f, "{inner}!"),
            Self::List(inner) => write!(f, "[{inner}]"),
        }
    }
}

// ── Argument / InputValue ─────────────────────────────────────────

/// A field argument or input value.
#[derive(Debug, Clone, PartialEq)]
pub struct InputValue {
    pub name: String,
    pub type_ref: TypeRef,
    pub description: Option<String>,
    pub default_value: Option<String>,
}

impl InputValue {
    pub fn new(name: &str, type_ref: TypeRef) -> Self {
        Self { name: name.into(), type_ref, description: None, default_value: None }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.into()); self
    }

    pub fn with_default(mut self, val: &str) -> Self {
        self.default_value = Some(val.into()); self
    }
}

// ── Field ─────────────────────────────────────────────────────────

/// A field on an object or interface type.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub type_ref: TypeRef,
    pub description: Option<String>,
    pub arguments: Vec<InputValue>,
    pub is_deprecated: bool,
    pub deprecation_reason: Option<String>,
}

impl Field {
    pub fn new(name: &str, type_ref: TypeRef) -> Self {
        Self {
            name: name.into(), type_ref, description: None,
            arguments: Vec::new(), is_deprecated: false, deprecation_reason: None,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.into()); self
    }

    pub fn with_arg(mut self, arg: InputValue) -> Self {
        self.arguments.push(arg); self
    }

    pub fn deprecated(mut self, reason: &str) -> Self {
        self.is_deprecated = true;
        self.deprecation_reason = Some(reason.into());
        self
    }
}

// ── Enum value ────────────────────────────────────────────────────

/// A value in a GraphQL enum type.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumValue {
    pub name: String,
    pub description: Option<String>,
    pub is_deprecated: bool,
    pub deprecation_reason: Option<String>,
}

impl EnumValue {
    pub fn new(name: &str) -> Self {
        Self { name: name.into(), description: None, is_deprecated: false, deprecation_reason: None }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.into()); self
    }

    pub fn deprecated(mut self, reason: &str) -> Self {
        self.is_deprecated = true;
        self.deprecation_reason = Some(reason.into());
        self
    }
}

// ── Directive ─────────────────────────────────────────────────────

/// Where a directive can be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectiveLocation {
    Query, Mutation, Subscription, Field, FragmentDefinition,
    FragmentSpread, InlineFragment, Schema, Scalar, Object,
    FieldDefinition, ArgumentDefinition, Interface, Union,
    Enum, EnumValue, InputObject, InputFieldDefinition,
}

impl fmt::Display for DirectiveLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query => write!(f, "QUERY"),
            Self::Mutation => write!(f, "MUTATION"),
            Self::Subscription => write!(f, "SUBSCRIPTION"),
            Self::Field => write!(f, "FIELD"),
            Self::FragmentDefinition => write!(f, "FRAGMENT_DEFINITION"),
            Self::FragmentSpread => write!(f, "FRAGMENT_SPREAD"),
            Self::InlineFragment => write!(f, "INLINE_FRAGMENT"),
            Self::Schema => write!(f, "SCHEMA"),
            Self::Scalar => write!(f, "SCALAR"),
            Self::Object => write!(f, "OBJECT"),
            Self::FieldDefinition => write!(f, "FIELD_DEFINITION"),
            Self::ArgumentDefinition => write!(f, "ARGUMENT_DEFINITION"),
            Self::Interface => write!(f, "INTERFACE"),
            Self::Union => write!(f, "UNION"),
            Self::Enum => write!(f, "ENUM"),
            Self::EnumValue => write!(f, "ENUM_VALUE"),
            Self::InputObject => write!(f, "INPUT_OBJECT"),
            Self::InputFieldDefinition => write!(f, "INPUT_FIELD_DEFINITION"),
        }
    }
}

/// A directive definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Directive {
    pub name: String,
    pub description: Option<String>,
    pub locations: Vec<DirectiveLocation>,
    pub arguments: Vec<InputValue>,
    pub is_repeatable: bool,
}

impl Directive {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(), description: None,
            locations: Vec::new(), arguments: Vec::new(),
            is_repeatable: false,
        }
    }

    pub fn with_location(mut self, loc: DirectiveLocation) -> Self {
        self.locations.push(loc); self
    }

    pub fn with_arg(mut self, arg: InputValue) -> Self {
        self.arguments.push(arg); self
    }
}

// ── Type definitions ──────────────────────────────────────────────

/// A type definition in the schema.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDef {
    Scalar {
        name: String,
        description: Option<String>,
    },
    Object {
        name: String,
        description: Option<String>,
        fields: Vec<Field>,
        interfaces: Vec<String>,
    },
    Interface {
        name: String,
        description: Option<String>,
        fields: Vec<Field>,
    },
    Union {
        name: String,
        description: Option<String>,
        members: Vec<String>,
    },
    Enum {
        name: String,
        description: Option<String>,
        values: Vec<EnumValue>,
    },
    InputObject {
        name: String,
        description: Option<String>,
        fields: Vec<InputValue>,
    },
}

impl TypeDef {
    pub fn name(&self) -> &str {
        match self {
            Self::Scalar { name, .. } | Self::Object { name, .. }
            | Self::Interface { name, .. } | Self::Union { name, .. }
            | Self::Enum { name, .. } | Self::InputObject { name, .. } => name,
        }
    }

    pub fn kind(&self) -> &str {
        match self {
            Self::Scalar { .. } => "SCALAR",
            Self::Object { .. } => "OBJECT",
            Self::Interface { .. } => "INTERFACE",
            Self::Union { .. } => "UNION",
            Self::Enum { .. } => "ENUM",
            Self::InputObject { .. } => "INPUT_OBJECT",
        }
    }
}

// ── Schema ────────────────────────────────────────────────────────

/// A complete GraphQL schema.
#[derive(Debug, Clone)]
pub struct GraphQLSchema {
    pub types: BTreeMap<String, TypeDef>,
    pub query_type: Option<String>,
    pub mutation_type: Option<String>,
    pub subscription_type: Option<String>,
    pub directives: Vec<Directive>,
}

impl Default for GraphQLSchema {
    fn default() -> Self {
        let mut schema = Self {
            types: BTreeMap::new(),
            query_type: None,
            mutation_type: None,
            subscription_type: None,
            directives: Vec::new(),
        };
        // Add built-in scalar types
        for name in &["String", "Int", "Float", "Boolean", "ID"] {
            schema.types.insert(name.to_string(), TypeDef::Scalar {
                name: name.to_string(),
                description: Some(format!("Built-in {name} scalar")),
            });
        }
        // Add built-in directives
        schema.directives.push(Directive {
            name: "skip".into(),
            description: Some("Skip this field if argument is true".into()),
            locations: vec![DirectiveLocation::Field, DirectiveLocation::FragmentSpread, DirectiveLocation::InlineFragment],
            arguments: vec![InputValue::new("if", TypeRef::non_null_named("Boolean"))],
            is_repeatable: false,
        });
        schema.directives.push(Directive {
            name: "include".into(),
            description: Some("Include this field if argument is true".into()),
            locations: vec![DirectiveLocation::Field, DirectiveLocation::FragmentSpread, DirectiveLocation::InlineFragment],
            arguments: vec![InputValue::new("if", TypeRef::non_null_named("Boolean"))],
            is_repeatable: false,
        });
        schema.directives.push(Directive {
            name: "deprecated".into(),
            description: Some("Marks an element as deprecated".into()),
            locations: vec![DirectiveLocation::FieldDefinition, DirectiveLocation::EnumValue],
            arguments: vec![
                InputValue::new("reason", TypeRef::named("String"))
                    .with_default("\"No longer supported\""),
            ],
            is_repeatable: false,
        });
        schema
    }
}

impl GraphQLSchema {
    pub fn new() -> Self { Self::default() }

    pub fn add_type(&mut self, td: TypeDef) {
        self.types.insert(td.name().to_string(), td);
    }

    pub fn set_query(&mut self, name: &str) { self.query_type = Some(name.into()); }
    pub fn set_mutation(&mut self, name: &str) { self.mutation_type = Some(name.into()); }
    pub fn set_subscription(&mut self, name: &str) { self.subscription_type = Some(name.into()); }

    pub fn add_directive(&mut self, d: Directive) { self.directives.push(d); }

    pub fn get_type(&self, name: &str) -> Option<&TypeDef> { self.types.get(name) }

    /// Validate the schema. Returns a list of error strings.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        // Check query type exists
        if let Some(ref qt) = self.query_type {
            if !self.types.contains_key(qt) {
                errors.push(format!("query type '{qt}' not defined"));
            }
        }
        if let Some(ref mt) = self.mutation_type {
            if !self.types.contains_key(mt) {
                errors.push(format!("mutation type '{mt}' not defined"));
            }
        }
        if let Some(ref st) = self.subscription_type {
            if !self.types.contains_key(st) {
                errors.push(format!("subscription type '{st}' not defined"));
            }
        }
        // Check all type references resolve
        for td in self.types.values() {
            match td {
                TypeDef::Object { name, fields, interfaces, .. } => {
                    for field in fields {
                        self.check_type_ref(&field.type_ref, name, &field.name, &mut errors);
                        for arg in &field.arguments {
                            self.check_type_ref(&arg.type_ref, name, &format!("{}.{}", field.name, arg.name), &mut errors);
                        }
                    }
                    for iface in interfaces {
                        if !self.types.contains_key(iface) {
                            errors.push(format!("{name}: interface '{iface}' not defined"));
                        } else if let Some(TypeDef::Interface { .. }) = self.types.get(iface) {
                            // ok
                        } else {
                            errors.push(format!("{name}: '{iface}' is not an interface"));
                        }
                    }
                }
                TypeDef::Interface { name, fields, .. } => {
                    for field in fields {
                        self.check_type_ref(&field.type_ref, name, &field.name, &mut errors);
                    }
                }
                TypeDef::Union { name, members, .. } => {
                    if members.is_empty() {
                        errors.push(format!("{name}: union must have at least one member"));
                    }
                    for member in members {
                        if !self.types.contains_key(member) {
                            errors.push(format!("{name}: union member '{member}' not defined"));
                        }
                    }
                }
                TypeDef::InputObject { name, fields, .. } => {
                    for field in fields {
                        self.check_type_ref(&field.type_ref, name, &field.name, &mut errors);
                    }
                }
                TypeDef::Enum { name, values, .. } => {
                    if values.is_empty() {
                        errors.push(format!("{name}: enum must have at least one value"));
                    }
                }
                TypeDef::Scalar { .. } => {}
            }
        }
        errors
    }

    fn check_type_ref(&self, tr: &TypeRef, context: &str, field: &str, errors: &mut Vec<String>) {
        let name = tr.inner_name();
        if !self.types.contains_key(name) {
            errors.push(format!("{context}.{field}: type '{name}' not defined"));
        }
    }

    /// Generate the SDL (Schema Definition Language) string.
    pub fn to_sdl(&self) -> String {
        let mut out = String::new();
        // Schema definition
        let has_schema = self.query_type.is_some()
            || self.mutation_type.is_some()
            || self.subscription_type.is_some();
        if has_schema {
            out.push_str("schema {\n");
            if let Some(ref q) = self.query_type {
                out.push_str(&format!("  query: {q}\n"));
            }
            if let Some(ref m) = self.mutation_type {
                out.push_str(&format!("  mutation: {m}\n"));
            }
            if let Some(ref s) = self.subscription_type {
                out.push_str(&format!("  subscription: {s}\n"));
            }
            out.push_str("}\n\n");
        }
        // Types (skip built-in scalars)
        let builtins = ["String", "Int", "Float", "Boolean", "ID"];
        for td in self.types.values() {
            if builtins.contains(&td.name()) { continue; }
            self.write_type_sdl(td, &mut out);
            out.push('\n');
        }
        // Custom directives (skip built-ins)
        let builtin_directives = ["skip", "include", "deprecated"];
        for d in &self.directives {
            if builtin_directives.contains(&d.name.as_str()) { continue; }
            self.write_directive_sdl(d, &mut out);
            out.push('\n');
        }
        out
    }

    fn write_type_sdl(&self, td: &TypeDef, out: &mut String) {
        match td {
            TypeDef::Scalar { name, description } => {
                if let Some(d) = description {
                    out.push_str(&format!("\"{d}\"\n"));
                }
                out.push_str(&format!("scalar {name}\n"));
            }
            TypeDef::Object { name, description, fields, interfaces } => {
                if let Some(d) = description {
                    out.push_str(&format!("\"{d}\"\n"));
                }
                out.push_str(&format!("type {name}"));
                if !interfaces.is_empty() {
                    out.push_str(" implements ");
                    out.push_str(&interfaces.join(" & "));
                }
                out.push_str(" {\n");
                for f in fields { self.write_field_sdl(f, out); }
                out.push_str("}\n");
            }
            TypeDef::Interface { name, description, fields } => {
                if let Some(d) = description {
                    out.push_str(&format!("\"{d}\"\n"));
                }
                out.push_str(&format!("interface {name} {{\n"));
                for f in fields { self.write_field_sdl(f, out); }
                out.push_str("}\n");
            }
            TypeDef::Union { name, description, members } => {
                if let Some(d) = description {
                    out.push_str(&format!("\"{d}\"\n"));
                }
                out.push_str(&format!("union {name} = {}\n", members.join(" | ")));
            }
            TypeDef::Enum { name, description, values } => {
                if let Some(d) = description {
                    out.push_str(&format!("\"{d}\"\n"));
                }
                out.push_str(&format!("enum {name} {{\n"));
                for v in values {
                    out.push_str(&format!("  {}", v.name));
                    if v.is_deprecated {
                        if let Some(ref reason) = v.deprecation_reason {
                            out.push_str(&format!(" @deprecated(reason: \"{reason}\")"));
                        } else {
                            out.push_str(" @deprecated");
                        }
                    }
                    out.push('\n');
                }
                out.push_str("}\n");
            }
            TypeDef::InputObject { name, description, fields } => {
                if let Some(d) = description {
                    out.push_str(&format!("\"{d}\"\n"));
                }
                out.push_str(&format!("input {name} {{\n"));
                for f in fields {
                    out.push_str(&format!("  {}: {}", f.name, f.type_ref));
                    if let Some(ref dv) = f.default_value {
                        out.push_str(&format!(" = {dv}"));
                    }
                    out.push('\n');
                }
                out.push_str("}\n");
            }
        }
    }

    fn write_field_sdl(&self, f: &Field, out: &mut String) {
        out.push_str(&format!("  {}", f.name));
        if !f.arguments.is_empty() {
            out.push('(');
            let args: Vec<String> = f.arguments.iter().map(|a| {
                let mut s = format!("{}: {}", a.name, a.type_ref);
                if let Some(ref dv) = a.default_value {
                    s.push_str(&format!(" = {dv}"));
                }
                s
            }).collect();
            out.push_str(&args.join(", "));
            out.push(')');
        }
        out.push_str(&format!(": {}", f.type_ref));
        if f.is_deprecated {
            if let Some(ref reason) = f.deprecation_reason {
                out.push_str(&format!(" @deprecated(reason: \"{reason}\")"));
            } else {
                out.push_str(" @deprecated");
            }
        }
        out.push('\n');
    }

    fn write_directive_sdl(&self, d: &Directive, out: &mut String) {
        out.push_str("directive @");
        out.push_str(&d.name);
        if !d.arguments.is_empty() {
            out.push('(');
            let args: Vec<String> = d.arguments.iter().map(|a| {
                format!("{}: {}", a.name, a.type_ref)
            }).collect();
            out.push_str(&args.join(", "));
            out.push(')');
        }
        if d.is_repeatable { out.push_str(" repeatable"); }
        out.push_str(" on ");
        let locs: Vec<String> = d.locations.iter().map(|l| l.to_string()).collect();
        out.push_str(&locs.join(" | "));
        out.push('\n');
    }

    /// Introspection: return the __schema result as JSON.
    pub fn introspect(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        // queryType
        if let Some(ref qt) = self.query_type {
            let mut t = serde_json::Map::new();
            t.insert("name".into(), serde_json::Value::String(qt.clone()));
            m.insert("queryType".into(), serde_json::Value::Object(t));
        } else {
            m.insert("queryType".into(), serde_json::Value::Null);
        }
        // mutationType
        if let Some(ref mt) = self.mutation_type {
            let mut t = serde_json::Map::new();
            t.insert("name".into(), serde_json::Value::String(mt.clone()));
            m.insert("mutationType".into(), serde_json::Value::Object(t));
        } else {
            m.insert("mutationType".into(), serde_json::Value::Null);
        }
        // types
        let types_arr: Vec<serde_json::Value> = self.types.values()
            .map(|td| self.introspect_type(td))
            .collect();
        m.insert("types".into(), serde_json::Value::Array(types_arr));
        // directives
        let dirs: Vec<serde_json::Value> = self.directives.iter()
            .map(|d| {
                let mut dm = serde_json::Map::new();
                dm.insert("name".into(), serde_json::Value::String(d.name.clone()));
                let locs: Vec<serde_json::Value> = d.locations.iter()
                    .map(|l| serde_json::Value::String(l.to_string())).collect();
                dm.insert("locations".into(), serde_json::Value::Array(locs));
                serde_json::Value::Object(dm)
            }).collect();
        m.insert("directives".into(), serde_json::Value::Array(dirs));
        serde_json::Value::Object(m)
    }

    fn introspect_type(&self, td: &TypeDef) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("kind".into(), serde_json::Value::String(td.kind().into()));
        m.insert("name".into(), serde_json::Value::String(td.name().into()));
        match td {
            TypeDef::Object { fields, interfaces, .. } => {
                let fields_arr: Vec<serde_json::Value> = fields.iter()
                    .map(|f| {
                        let mut fm = serde_json::Map::new();
                        fm.insert("name".into(), serde_json::Value::String(f.name.clone()));
                        fm.insert("type".into(), serde_json::Value::String(f.type_ref.to_string()));
                        fm.insert("isDeprecated".into(), serde_json::Value::Bool(f.is_deprecated));
                        serde_json::Value::Object(fm)
                    }).collect();
                m.insert("fields".into(), serde_json::Value::Array(fields_arr));
                let ifaces: Vec<serde_json::Value> = interfaces.iter()
                    .map(|i| serde_json::Value::String(i.clone())).collect();
                m.insert("interfaces".into(), serde_json::Value::Array(ifaces));
            }
            TypeDef::Enum { values, .. } => {
                let vals: Vec<serde_json::Value> = values.iter()
                    .map(|v| {
                        let mut vm = serde_json::Map::new();
                        vm.insert("name".into(), serde_json::Value::String(v.name.clone()));
                        vm.insert("isDeprecated".into(), serde_json::Value::Bool(v.is_deprecated));
                        serde_json::Value::Object(vm)
                    }).collect();
                m.insert("enumValues".into(), serde_json::Value::Array(vals));
            }
            TypeDef::Union { members, .. } => {
                let mbrs: Vec<serde_json::Value> = members.iter()
                    .map(|mm| serde_json::Value::String(mm.clone())).collect();
                m.insert("possibleTypes".into(), serde_json::Value::Array(mbrs));
            }
            _ => {}
        }
        serde_json::Value::Object(m)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_ref_display() {
        assert_eq!(TypeRef::named("String").to_string(), "String");
        assert_eq!(TypeRef::non_null_named("String").to_string(), "String!");
        assert_eq!(TypeRef::list(TypeRef::named("Int")).to_string(), "[Int]");
        assert_eq!(TypeRef::non_null_list_of_non_null("User").to_string(), "[User!]!");
    }

    #[test]
    fn type_ref_inner_name() {
        let tr = TypeRef::non_null_list_of_non_null("Post");
        assert_eq!(tr.inner_name(), "Post");
    }

    #[test]
    fn type_ref_predicates() {
        assert!(TypeRef::non_null_named("X").is_non_null());
        assert!(TypeRef::list(TypeRef::named("X")).is_list());
        assert!(!TypeRef::named("X").is_non_null());
    }

    #[test]
    fn field_builder() {
        let f = Field::new("name", TypeRef::non_null_named("String"))
            .with_description("The user's name")
            .with_arg(InputValue::new("uppercase", TypeRef::named("Boolean")))
            .deprecated("Use fullName");
        assert_eq!(f.name, "name");
        assert!(f.is_deprecated);
        assert_eq!(f.arguments.len(), 1);
    }

    #[test]
    fn enum_value_deprecated() {
        let v = EnumValue::new("OLD_VALUE").deprecated("Use NEW_VALUE");
        assert!(v.is_deprecated);
        assert_eq!(v.deprecation_reason.as_deref(), Some("Use NEW_VALUE"));
    }

    #[test]
    fn directive_builder() {
        let d = Directive::new("cache")
            .with_location(DirectiveLocation::FieldDefinition)
            .with_arg(InputValue::new("maxAge", TypeRef::non_null_named("Int")));
        assert_eq!(d.name, "cache");
        assert_eq!(d.locations.len(), 1);
        assert_eq!(d.arguments.len(), 1);
    }

    #[test]
    fn schema_built_in_scalars() {
        let schema = GraphQLSchema::new();
        assert!(schema.get_type("String").is_some());
        assert!(schema.get_type("Int").is_some());
        assert!(schema.get_type("Float").is_some());
        assert!(schema.get_type("Boolean").is_some());
        assert!(schema.get_type("ID").is_some());
    }

    #[test]
    fn schema_built_in_directives() {
        let schema = GraphQLSchema::new();
        assert!(schema.directives.iter().any(|d| d.name == "skip"));
        assert!(schema.directives.iter().any(|d| d.name == "include"));
        assert!(schema.directives.iter().any(|d| d.name == "deprecated"));
    }

    #[test]
    fn schema_add_object_type() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "User".into(),
            description: Some("A user".into()),
            fields: vec![
                Field::new("id", TypeRef::non_null_named("ID")),
                Field::new("name", TypeRef::non_null_named("String")),
            ],
            interfaces: Vec::new(),
        });
        schema.set_query("User");
        assert!(schema.get_type("User").is_some());
        assert_eq!(schema.get_type("User").unwrap().kind(), "OBJECT");
    }

    #[test]
    fn schema_validate_valid() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![
                Field::new("hello", TypeRef::non_null_named("String")),
            ],
            interfaces: Vec::new(),
        });
        schema.set_query("Query");
        let errors = schema.validate();
        assert!(errors.is_empty(), "errors: {errors:?}");
    }

    #[test]
    fn schema_validate_missing_query_type() {
        let mut schema = GraphQLSchema::new();
        schema.set_query("Query");
        let errors = schema.validate();
        assert!(errors.iter().any(|e| e.contains("Query")));
    }

    #[test]
    fn schema_validate_unresolved_field_type() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![
                Field::new("user", TypeRef::named("User")),
            ],
            interfaces: Vec::new(),
        });
        schema.set_query("Query");
        let errors = schema.validate();
        assert!(errors.iter().any(|e| e.contains("User")));
    }

    #[test]
    fn schema_validate_interface_reference() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Interface {
            name: "Node".into(),
            description: None,
            fields: vec![Field::new("id", TypeRef::non_null_named("ID"))],
        });
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![Field::new("hello", TypeRef::named("String"))],
            interfaces: vec!["Node".into()],
        });
        schema.set_query("Query");
        let errors = schema.validate();
        assert!(errors.is_empty(), "errors: {errors:?}");
    }

    #[test]
    fn schema_validate_bad_interface() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![Field::new("x", TypeRef::named("String"))],
            interfaces: vec!["NotAnInterface".into()],
        });
        let errors = schema.validate();
        assert!(errors.iter().any(|e| e.contains("NotAnInterface")));
    }

    #[test]
    fn schema_validate_empty_union() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Union {
            name: "Empty".into(),
            description: None,
            members: Vec::new(),
        });
        let errors = schema.validate();
        assert!(errors.iter().any(|e| e.contains("at least one member")));
    }

    #[test]
    fn schema_validate_empty_enum() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Enum {
            name: "Empty".into(),
            description: None,
            values: Vec::new(),
        });
        let errors = schema.validate();
        assert!(errors.iter().any(|e| e.contains("at least one value")));
    }

    #[test]
    fn sdl_simple_schema() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![
                Field::new("hello", TypeRef::non_null_named("String")),
            ],
            interfaces: Vec::new(),
        });
        schema.set_query("Query");
        let sdl = schema.to_sdl();
        assert!(sdl.contains("schema {"));
        assert!(sdl.contains("query: Query"));
        assert!(sdl.contains("type Query {"));
        assert!(sdl.contains("hello: String!"));
    }

    #[test]
    fn sdl_enum() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Enum {
            name: "Status".into(),
            description: None,
            values: vec![
                EnumValue::new("ACTIVE"),
                EnumValue::new("INACTIVE").deprecated("Use DISABLED"),
            ],
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("enum Status {"));
        assert!(sdl.contains("ACTIVE"));
        assert!(sdl.contains("INACTIVE @deprecated"));
    }

    #[test]
    fn sdl_union() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Union {
            name: "SearchResult".into(),
            description: None,
            members: vec!["User".into(), "Post".into()],
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("union SearchResult = User | Post"));
    }

    #[test]
    fn sdl_input_type() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::InputObject {
            name: "CreateUser".into(),
            description: None,
            fields: vec![
                InputValue::new("name", TypeRef::non_null_named("String")),
                InputValue::new("age", TypeRef::named("Int")).with_default("18"),
            ],
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("input CreateUser {"));
        assert!(sdl.contains("name: String!"));
        assert!(sdl.contains("age: Int = 18"));
    }

    #[test]
    fn sdl_interface() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Interface {
            name: "Node".into(),
            description: None,
            fields: vec![Field::new("id", TypeRef::non_null_named("ID"))],
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("interface Node {"));
        assert!(sdl.contains("id: ID!"));
    }

    #[test]
    fn sdl_implements() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "User".into(),
            description: None,
            fields: vec![
                Field::new("id", TypeRef::non_null_named("ID")),
                Field::new("name", TypeRef::non_null_named("String")),
            ],
            interfaces: vec!["Node".into(), "Named".into()],
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("type User implements Node & Named {"));
    }

    #[test]
    fn sdl_field_with_args() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![
                Field::new("user", TypeRef::named("User"))
                    .with_arg(InputValue::new("id", TypeRef::non_null_named("ID"))),
            ],
            interfaces: Vec::new(),
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("user(id: ID!): User"));
    }

    #[test]
    fn introspection_basic() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Object {
            name: "Query".into(),
            description: None,
            fields: vec![Field::new("hello", TypeRef::non_null_named("String"))],
            interfaces: Vec::new(),
        });
        schema.set_query("Query");
        let intro = schema.introspect();
        assert_eq!(intro["queryType"]["name"], "Query");
        assert!(intro["mutationType"].is_null());
        let types = intro["types"].as_array().unwrap();
        assert!(types.iter().any(|t| t["name"] == "Query"));
        assert!(types.iter().any(|t| t["name"] == "String"));
    }

    #[test]
    fn introspection_enum() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Enum {
            name: "Color".into(),
            description: None,
            values: vec![EnumValue::new("RED"), EnumValue::new("GREEN")],
        });
        let intro = schema.introspect();
        let types = intro["types"].as_array().unwrap();
        let color = types.iter().find(|t| t["name"] == "Color").unwrap();
        assert_eq!(color["kind"], "ENUM");
        assert_eq!(color["enumValues"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn introspection_directives() {
        let schema = GraphQLSchema::new();
        let intro = schema.introspect();
        let dirs = intro["directives"].as_array().unwrap();
        assert!(dirs.iter().any(|d| d["name"] == "skip"));
        assert!(dirs.iter().any(|d| d["name"] == "include"));
    }

    #[test]
    fn typedef_kind() {
        assert_eq!(TypeDef::Scalar { name: "X".into(), description: None }.kind(), "SCALAR");
        assert_eq!(TypeDef::Object { name: "X".into(), description: None, fields: vec![], interfaces: vec![] }.kind(), "OBJECT");
        assert_eq!(TypeDef::Interface { name: "X".into(), description: None, fields: vec![] }.kind(), "INTERFACE");
        assert_eq!(TypeDef::Union { name: "X".into(), description: None, members: vec![] }.kind(), "UNION");
        assert_eq!(TypeDef::Enum { name: "X".into(), description: None, values: vec![] }.kind(), "ENUM");
        assert_eq!(TypeDef::InputObject { name: "X".into(), description: None, fields: vec![] }.kind(), "INPUT_OBJECT");
    }

    #[test]
    fn input_value_builder() {
        let iv = InputValue::new("limit", TypeRef::named("Int"))
            .with_description("Max items")
            .with_default("10");
        assert_eq!(iv.default_value.as_deref(), Some("10"));
        assert_eq!(iv.description.as_deref(), Some("Max items"));
    }

    #[test]
    fn directive_location_display() {
        assert_eq!(DirectiveLocation::Query.to_string(), "QUERY");
        assert_eq!(DirectiveLocation::FieldDefinition.to_string(), "FIELD_DEFINITION");
        assert_eq!(DirectiveLocation::InputObject.to_string(), "INPUT_OBJECT");
    }

    #[test]
    fn custom_scalar() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Scalar {
            name: "DateTime".into(),
            description: Some("ISO 8601 date-time".into()),
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("scalar DateTime"));
    }

    #[test]
    fn custom_directive_sdl() {
        let mut schema = GraphQLSchema::new();
        schema.add_directive(Directive {
            name: "auth".into(),
            description: None,
            locations: vec![DirectiveLocation::FieldDefinition, DirectiveLocation::Object],
            arguments: vec![InputValue::new("requires", TypeRef::non_null_named("String"))],
            is_repeatable: false,
        });
        let sdl = schema.to_sdl();
        assert!(sdl.contains("directive @auth(requires: String!) on FIELD_DEFINITION | OBJECT"));
    }

    #[test]
    fn validate_union_members() {
        let mut schema = GraphQLSchema::new();
        schema.add_type(TypeDef::Union {
            name: "Result".into(),
            description: None,
            members: vec!["Missing".into()],
        });
        let errors = schema.validate();
        assert!(errors.iter().any(|e| e.contains("Missing")));
    }
}
