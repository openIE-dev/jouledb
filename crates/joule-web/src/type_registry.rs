//! Runtime type registry — type descriptors, registration, lookup by name/id,
//! type hierarchy (inheritance), generic type instantiation, type metadata,
//! and reflection-like capabilities.
//!
//! Replaces TypeScript's reflect-metadata, Java's Class<T>, and C#'s
//! System.Type with a pure-Rust runtime type registry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors produced by the type registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRegistryError {
    /// Type with this name already registered.
    DuplicateType(String),
    /// Type not found by name.
    TypeNotFound(String),
    /// Type not found by id.
    TypeIdNotFound(u64),
    /// Circular inheritance detected.
    CircularInheritance { child: String, ancestor: String },
    /// Parent type does not exist.
    ParentNotFound { child: String, parent: String },
    /// Generic parameter count mismatch.
    GenericArityMismatch { type_name: String, expected: usize, got: usize },
    /// Method not found on type.
    MethodNotFound { type_name: String, method: String },
    /// Field not found on type.
    FieldNotFound { type_name: String, field: String },
    /// Invalid metadata key.
    MetadataKeyNotFound { type_name: String, key: String },
}

impl fmt::Display for TypeRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateType(n) => write!(f, "type already registered: {n}"),
            Self::TypeNotFound(n) => write!(f, "type not found: {n}"),
            Self::TypeIdNotFound(id) => write!(f, "type id not found: {id}"),
            Self::CircularInheritance { child, ancestor } => {
                write!(f, "circular inheritance: {child} -> {ancestor}")
            }
            Self::ParentNotFound { child, parent } => {
                write!(f, "parent type {parent} not found for {child}")
            }
            Self::GenericArityMismatch { type_name, expected, got } => {
                write!(f, "generic arity mismatch for {type_name}: expected {expected}, got {got}")
            }
            Self::MethodNotFound { type_name, method } => {
                write!(f, "method {method} not found on {type_name}")
            }
            Self::FieldNotFound { type_name, field } => {
                write!(f, "field {field} not found on {type_name}")
            }
            Self::MetadataKeyNotFound { type_name, key } => {
                write!(f, "metadata key {key} not found on {type_name}")
            }
        }
    }
}

impl std::error::Error for TypeRegistryError {}

// ── Type Kind ───────────────────────────────────────────────────

/// The kind of a registered type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeKind {
    /// A primitive type (i32, f64, bool, String, etc.).
    Primitive,
    /// A struct/record with named fields.
    Struct,
    /// An enum with named variants.
    Enum,
    /// An interface/trait-like type.
    Interface,
    /// A generic type that requires parameters to instantiate.
    Generic,
    /// An alias for another type.
    Alias,
}

// ── Field Descriptor ────────────────────────────────────────────

/// Describes a field on a struct-like type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldDescriptor {
    /// Field name.
    pub name: String,
    /// Type name of this field.
    pub type_name: String,
    /// Whether the field is optional.
    pub optional: bool,
    /// Default value as JSON, if any.
    pub default_value: Option<String>,
    /// Doc string.
    pub doc: Option<String>,
}

impl FieldDescriptor {
    /// Create a required field.
    pub fn required(name: impl Into<String>, type_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_name: type_name.into(),
            optional: false,
            default_value: None,
            doc: None,
        }
    }

    /// Create an optional field.
    pub fn optional(name: impl Into<String>, type_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_name: type_name.into(),
            optional: true,
            default_value: None,
            doc: None,
        }
    }

    /// Set the default value (as JSON string).
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_value = Some(default.into());
        self
    }

    /// Set the doc string.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }
}

// ── Method Descriptor ───────────────────────────────────────────

/// Describes a method on a type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MethodDescriptor {
    /// Method name.
    pub name: String,
    /// Parameter types (name, type_name).
    pub parameters: Vec<(String, String)>,
    /// Return type name, or `None` for void.
    pub return_type: Option<String>,
    /// Whether the method is static (no self).
    pub is_static: bool,
    /// Doc string.
    pub doc: Option<String>,
}

impl MethodDescriptor {
    /// Create a new instance method descriptor.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: Vec::new(),
            return_type: None,
            is_static: false,
            doc: None,
        }
    }

    /// Mark the method as static.
    pub fn make_static(mut self) -> Self {
        self.is_static = true;
        self
    }

    /// Add a parameter.
    pub fn with_param(mut self, name: impl Into<String>, type_name: impl Into<String>) -> Self {
        self.parameters.push((name.into(), type_name.into()));
        self
    }

    /// Set the return type.
    pub fn with_return(mut self, type_name: impl Into<String>) -> Self {
        self.return_type = Some(type_name.into());
        self
    }

    /// Set the doc string.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }
}

// ── Type Descriptor ─────────────────────────────────────────────

/// Full descriptor of a registered type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDescriptor {
    /// Unique numeric id.
    pub id: u64,
    /// Unique name.
    pub name: String,
    /// Kind of type.
    pub kind: TypeKind,
    /// Fields (for Struct kinds).
    pub fields: Vec<FieldDescriptor>,
    /// Methods.
    pub methods: Vec<MethodDescriptor>,
    /// Parent type name (single inheritance).
    pub parent: Option<String>,
    /// Implemented interface names.
    pub interfaces: Vec<String>,
    /// Generic type parameter names.
    pub generic_params: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
    /// Doc string.
    pub doc: Option<String>,
}

impl TypeDescriptor {
    /// Create a new type descriptor.
    pub fn new(id: u64, name: impl Into<String>, kind: TypeKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            fields: Vec::new(),
            methods: Vec::new(),
            parent: None,
            interfaces: Vec::new(),
            generic_params: Vec::new(),
            metadata: HashMap::new(),
            doc: None,
        }
    }

    /// Add a field.
    pub fn with_field(mut self, field: FieldDescriptor) -> Self {
        self.fields.push(field);
        self
    }

    /// Add a method.
    pub fn with_method(mut self, method: MethodDescriptor) -> Self {
        self.methods.push(method);
        self
    }

    /// Set the parent type.
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    /// Add an implemented interface.
    pub fn with_interface(mut self, iface: impl Into<String>) -> Self {
        self.interfaces.push(iface.into());
        self
    }

    /// Add a generic parameter.
    pub fn with_generic_param(mut self, param: impl Into<String>) -> Self {
        self.generic_params.push(param.into());
        self
    }

    /// Set metadata key/value.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the doc string.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    /// Check whether this type has a given field.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    /// Get a field descriptor by name.
    pub fn get_field(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Check whether this type has a given method.
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.iter().any(|m| m.name == name)
    }

    /// Get a method descriptor by name.
    pub fn get_method(&self, name: &str) -> Option<&MethodDescriptor> {
        self.methods.iter().find(|m| m.name == name)
    }

    /// Whether this is a generic type.
    pub fn is_generic(&self) -> bool {
        !self.generic_params.is_empty()
    }

    /// Count of required (non-optional) fields.
    pub fn required_field_count(&self) -> usize {
        self.fields.iter().filter(|f| !f.optional).count()
    }
}

// ── Type Registry ───────────────────────────────────────────────

/// Runtime type registry: register, look up, query, and instantiate types.
#[derive(Debug, Clone)]
pub struct TypeRegistry {
    /// Types by name.
    types: HashMap<String, TypeDescriptor>,
    /// Name by id for fast id lookup.
    id_to_name: HashMap<u64, String>,
    /// Next auto-id.
    next_id: u64,
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            id_to_name: HashMap::new(),
            next_id: 1,
        }
    }

    /// How many types are registered.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Register a type descriptor (auto-assigns id if descriptor.id == 0).
    pub fn register(&mut self, mut desc: TypeDescriptor) -> Result<u64, TypeRegistryError> {
        if self.types.contains_key(&desc.name) {
            return Err(TypeRegistryError::DuplicateType(desc.name.clone()));
        }
        // Validate parent exists if set.
        if let Some(parent) = &desc.parent {
            if !self.types.contains_key(parent) {
                return Err(TypeRegistryError::ParentNotFound {
                    child: desc.name.clone(),
                    parent: parent.clone(),
                });
            }
            // Check for circular inheritance.
            if self.is_ancestor(&desc.name, parent) {
                return Err(TypeRegistryError::CircularInheritance {
                    child: desc.name.clone(),
                    ancestor: parent.clone(),
                });
            }
        }
        if desc.id == 0 {
            desc.id = self.next_id;
            self.next_id += 1;
        } else if desc.id >= self.next_id {
            self.next_id = desc.id + 1;
        }
        let id = desc.id;
        self.id_to_name.insert(id, desc.name.clone());
        self.types.insert(desc.name.clone(), desc);
        Ok(id)
    }

    /// Register a simple primitive type by name.
    pub fn register_primitive(&mut self, name: impl Into<String>) -> Result<u64, TypeRegistryError> {
        let name = name.into();
        let desc = TypeDescriptor::new(0, &name, TypeKind::Primitive);
        self.register(desc)
    }

    /// Look up a type by name.
    pub fn get(&self, name: &str) -> Option<&TypeDescriptor> {
        self.types.get(name)
    }

    /// Look up a type by id.
    pub fn get_by_id(&self, id: u64) -> Option<&TypeDescriptor> {
        self.id_to_name.get(&id).and_then(|name| self.types.get(name))
    }

    /// Check whether a type with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Remove a type from the registry.
    pub fn unregister(&mut self, name: &str) -> Result<TypeDescriptor, TypeRegistryError> {
        let desc = self.types.remove(name).ok_or_else(|| TypeRegistryError::TypeNotFound(name.to_string()))?;
        self.id_to_name.remove(&desc.id);
        Ok(desc)
    }

    /// Get all type names (sorted for determinism).
    pub fn type_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.types.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get all type descriptors (sorted by name for determinism).
    pub fn all_types(&self) -> Vec<&TypeDescriptor> {
        let mut names: Vec<&String> = self.types.keys().collect();
        names.sort();
        names.iter().filter_map(|n| self.types.get(*n)).collect()
    }

    /// Get types of a specific kind.
    pub fn types_of_kind(&self, kind: &TypeKind) -> Vec<&TypeDescriptor> {
        let mut result: Vec<&TypeDescriptor> = self.types.values().filter(|t| &t.kind == kind).collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    // ── Hierarchy ───────────────────────────────────────────────

    /// Whether `ancestor_name` is an ancestor of the type named `type_name`.
    fn is_ancestor(&self, child_name: &str, ancestor_name: &str) -> bool {
        // Walk the parent chain starting from ancestor_name to see if we loop back to child_name.
        let mut current = Some(ancestor_name.to_string());
        while let Some(name) = current {
            if name == child_name {
                return true;
            }
            current = self.types.get(&name).and_then(|t| t.parent.clone());
        }
        false
    }

    /// Get the full ancestor chain for a type (parent, grandparent, ...).
    pub fn ancestors(&self, name: &str) -> Result<Vec<String>, TypeRegistryError> {
        if !self.types.contains_key(name) {
            return Err(TypeRegistryError::TypeNotFound(name.to_string()));
        }
        let mut result = Vec::new();
        let mut current = self.types.get(name).and_then(|t| t.parent.clone());
        while let Some(parent_name) = current {
            result.push(parent_name.clone());
            current = self.types.get(&parent_name).and_then(|t| t.parent.clone());
        }
        Ok(result)
    }

    /// Check whether `name` is a subtype of `ancestor` (directly or transitively).
    pub fn is_subtype_of(&self, name: &str, ancestor: &str) -> Result<bool, TypeRegistryError> {
        let ancestors = self.ancestors(name)?;
        Ok(ancestors.contains(&ancestor.to_string()))
    }

    /// Get all direct children of a type.
    pub fn children(&self, name: &str) -> Result<Vec<String>, TypeRegistryError> {
        if !self.types.contains_key(name) {
            return Err(TypeRegistryError::TypeNotFound(name.to_string()));
        }
        let target = name.to_string();
        let mut result: Vec<String> = self
            .types
            .values()
            .filter(|t| t.parent.as_ref() == Some(&target))
            .map(|t| t.name.clone())
            .collect();
        result.sort();
        Ok(result)
    }

    /// Get all descendants (children, grandchildren, ...) of a type.
    pub fn descendants(&self, name: &str) -> Result<Vec<String>, TypeRegistryError> {
        if !self.types.contains_key(name) {
            return Err(TypeRegistryError::TypeNotFound(name.to_string()));
        }
        let mut result = Vec::new();
        let mut stack = vec![name.to_string()];
        while let Some(current) = stack.pop() {
            let child_names: Vec<String> = self
                .types
                .values()
                .filter(|t| t.parent.as_ref() == Some(&current))
                .map(|t| t.name.clone())
                .collect();
            for c in child_names {
                result.push(c.clone());
                stack.push(c);
            }
        }
        result.sort();
        Ok(result)
    }

    // ── Generics ────────────────────────────────────────────────

    /// Instantiate a generic type with concrete type arguments.
    /// Creates a new non-generic type descriptor with parameters substituted in field types.
    pub fn instantiate_generic(
        &self,
        name: &str,
        type_args: &[&str],
    ) -> Result<TypeDescriptor, TypeRegistryError> {
        let base = self.types.get(name).ok_or_else(|| TypeRegistryError::TypeNotFound(name.to_string()))?;
        if base.generic_params.len() != type_args.len() {
            return Err(TypeRegistryError::GenericArityMismatch {
                type_name: name.to_string(),
                expected: base.generic_params.len(),
                got: type_args.len(),
            });
        }
        // Build substitution map: generic_param -> concrete type.
        let subst: HashMap<&str, &str> = base
            .generic_params
            .iter()
            .zip(type_args.iter())
            .map(|(p, a)| (p.as_str(), *a))
            .collect();

        let inst_name = format!("{}<{}>", name, type_args.join(", "));
        let fields = base
            .fields
            .iter()
            .map(|f| {
                let fallback = f.type_name.as_str();
                let resolved = subst.get(fallback).copied().unwrap_or(fallback);
                FieldDescriptor {
                    name: f.name.clone(),
                    type_name: resolved.to_string(),
                    optional: f.optional,
                    default_value: f.default_value.clone(),
                    doc: f.doc.clone(),
                }
            })
            .collect();

        let methods = base
            .methods
            .iter()
            .map(|m| {
                let params = m
                    .parameters
                    .iter()
                    .map(|(pn, pt)| {
                        let fallback = pt.as_str();
                        let resolved = subst.get(fallback).copied().unwrap_or(fallback);
                        (pn.clone(), resolved.to_string())
                    })
                    .collect();
                let ret = m.return_type.as_ref().map(|rt| {
                    let fallback = rt.as_str();
                    subst.get(fallback).copied().unwrap_or(fallback).to_string()
                });
                MethodDescriptor {
                    name: m.name.clone(),
                    parameters: params,
                    return_type: ret,
                    is_static: m.is_static,
                    doc: m.doc.clone(),
                }
            })
            .collect();

        Ok(TypeDescriptor {
            id: 0,
            name: inst_name,
            kind: base.kind.clone(),
            fields,
            methods,
            parent: base.parent.clone(),
            interfaces: base.interfaces.clone(),
            generic_params: Vec::new(),
            metadata: base.metadata.clone(),
            doc: base.doc.clone(),
        })
    }

    // ── Reflection-like queries ─────────────────────────────────

    /// Get all fields for a type including inherited fields (parent fields first).
    pub fn all_fields(&self, name: &str) -> Result<Vec<FieldDescriptor>, TypeRegistryError> {
        let desc = self.types.get(name).ok_or_else(|| TypeRegistryError::TypeNotFound(name.to_string()))?;
        let mut result = Vec::new();
        if let Some(parent) = &desc.parent {
            result.extend(self.all_fields(parent)?);
        }
        result.extend(desc.fields.clone());
        Ok(result)
    }

    /// Get all methods for a type including inherited methods (parent methods first).
    pub fn all_methods(&self, name: &str) -> Result<Vec<MethodDescriptor>, TypeRegistryError> {
        let desc = self.types.get(name).ok_or_else(|| TypeRegistryError::TypeNotFound(name.to_string()))?;
        let mut result = Vec::new();
        if let Some(parent) = &desc.parent {
            result.extend(self.all_methods(parent)?);
        }
        result.extend(desc.methods.clone());
        Ok(result)
    }

    /// Get metadata value for a type.
    pub fn get_metadata(&self, name: &str, key: &str) -> Result<&str, TypeRegistryError> {
        let desc = self.types.get(name).ok_or_else(|| TypeRegistryError::TypeNotFound(name.to_string()))?;
        desc.metadata
            .get(key)
            .map(|s| s.as_str())
            .ok_or_else(|| TypeRegistryError::MetadataKeyNotFound {
                type_name: name.to_string(),
                key: key.to_string(),
            })
    }

    /// Set metadata on an already-registered type.
    pub fn set_metadata(
        &mut self,
        name: &str,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), TypeRegistryError> {
        let desc = self.types.get_mut(name).ok_or_else(|| TypeRegistryError::TypeNotFound(name.to_string()))?;
        desc.metadata.insert(key.into(), value.into());
        Ok(())
    }

    /// Find all types that implement a given interface name.
    pub fn implementors_of(&self, interface: &str) -> Vec<String> {
        let mut result: Vec<String> = self
            .types
            .values()
            .filter(|t| t.interfaces.contains(&interface.to_string()))
            .map(|t| t.name.clone())
            .collect();
        result.sort();
        result
    }

    /// Search for types whose name contains a substring.
    pub fn search(&self, query: &str) -> Vec<&TypeDescriptor> {
        let lower = query.to_lowercase();
        let mut result: Vec<&TypeDescriptor> = self
            .types
            .values()
            .filter(|t| t.name.to_lowercase().contains(&lower))
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> TypeRegistry {
        let mut reg = TypeRegistry::new();
        reg.register_primitive("i32").unwrap();
        reg.register_primitive("String").unwrap();
        reg.register_primitive("bool").unwrap();
        reg
    }

    #[test]
    fn test_register_and_lookup() {
        let reg = make_registry();
        assert_eq!(reg.len(), 3);
        assert!(reg.contains("i32"));
        assert!(!reg.contains("f64"));
        let desc = reg.get("i32").unwrap();
        assert_eq!(desc.kind, TypeKind::Primitive);
    }

    #[test]
    fn test_lookup_by_id() {
        let mut reg = TypeRegistry::new();
        let id = reg.register_primitive("i32").unwrap();
        let desc = reg.get_by_id(id).unwrap();
        assert_eq!(desc.name, "i32");
    }

    #[test]
    fn test_duplicate_registration() {
        let mut reg = make_registry();
        let err = reg.register_primitive("i32").unwrap_err();
        assert_eq!(err, TypeRegistryError::DuplicateType("i32".to_string()));
    }

    #[test]
    fn test_unregister() {
        let mut reg = make_registry();
        let desc = reg.unregister("i32").unwrap();
        assert_eq!(desc.name, "i32");
        assert!(!reg.contains("i32"));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_unregister_missing() {
        let mut reg = TypeRegistry::new();
        let err = reg.unregister("nope").unwrap_err();
        assert_eq!(err, TypeRegistryError::TypeNotFound("nope".to_string()));
    }

    #[test]
    fn test_type_names_sorted() {
        let reg = make_registry();
        let names = reg.type_names();
        assert_eq!(names, vec!["String", "bool", "i32"]);
    }

    #[test]
    fn test_struct_with_fields() {
        let mut reg = make_registry();
        let desc = TypeDescriptor::new(0, "User", TypeKind::Struct)
            .with_field(FieldDescriptor::required("name", "String"))
            .with_field(FieldDescriptor::optional("age", "i32").with_default("0"));
        reg.register(desc).unwrap();

        let user = reg.get("User").unwrap();
        assert_eq!(user.fields.len(), 2);
        assert!(user.has_field("name"));
        assert!(!user.has_field("email"));
        assert_eq!(user.required_field_count(), 1);
    }

    #[test]
    fn test_struct_with_methods() {
        let mut reg = TypeRegistry::new();
        let desc = TypeDescriptor::new(0, "Calc", TypeKind::Struct)
            .with_method(
                MethodDescriptor::new("add")
                    .with_param("a", "i32")
                    .with_param("b", "i32")
                    .with_return("i32"),
            )
            .with_method(MethodDescriptor::new("create").make_static().with_return("Calc"));
        reg.register(desc).unwrap();

        let calc = reg.get("Calc").unwrap();
        assert!(calc.has_method("add"));
        let add = calc.get_method("add").unwrap();
        assert_eq!(add.parameters.len(), 2);
        assert!(!add.is_static);
        assert!(calc.get_method("create").unwrap().is_static);
    }

    #[test]
    fn test_inheritance_ancestors() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeDescriptor::new(0, "Animal", TypeKind::Struct)
            .with_field(FieldDescriptor::required("name", "String")))
            .unwrap();
        reg.register(TypeDescriptor::new(0, "Dog", TypeKind::Struct)
            .with_parent("Animal")
            .with_field(FieldDescriptor::required("breed", "String")))
            .unwrap();
        reg.register(TypeDescriptor::new(0, "Puppy", TypeKind::Struct)
            .with_parent("Dog")
            .with_field(FieldDescriptor::required("toy", "String")))
            .unwrap();

        let ancestors = reg.ancestors("Puppy").unwrap();
        assert_eq!(ancestors, vec!["Dog", "Animal"]);
        assert!(reg.is_subtype_of("Puppy", "Animal").unwrap());
        assert!(!reg.is_subtype_of("Animal", "Puppy").unwrap());
    }

    #[test]
    fn test_parent_not_found() {
        let mut reg = TypeRegistry::new();
        let desc = TypeDescriptor::new(0, "Orphan", TypeKind::Struct).with_parent("Missing");
        let err = reg.register(desc).unwrap_err();
        match err {
            TypeRegistryError::ParentNotFound { child, parent } => {
                assert_eq!(child, "Orphan");
                assert_eq!(parent, "Missing");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_children_and_descendants() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeDescriptor::new(0, "A", TypeKind::Struct)).unwrap();
        reg.register(TypeDescriptor::new(0, "B", TypeKind::Struct).with_parent("A")).unwrap();
        reg.register(TypeDescriptor::new(0, "C", TypeKind::Struct).with_parent("A")).unwrap();
        reg.register(TypeDescriptor::new(0, "D", TypeKind::Struct).with_parent("B")).unwrap();

        let children = reg.children("A").unwrap();
        assert_eq!(children, vec!["B", "C"]);

        let desc = reg.descendants("A").unwrap();
        assert_eq!(desc, vec!["B", "C", "D"]);
    }

    #[test]
    fn test_generic_instantiation() {
        let mut reg = TypeRegistry::new();
        let list = TypeDescriptor::new(0, "List", TypeKind::Generic)
            .with_generic_param("T")
            .with_field(FieldDescriptor::required("items", "T"))
            .with_method(MethodDescriptor::new("push").with_param("item", "T"));
        reg.register(list).unwrap();

        let inst = reg.instantiate_generic("List", &["i32"]).unwrap();
        assert_eq!(inst.name, "List<i32>");
        assert!(inst.generic_params.is_empty());
        assert_eq!(inst.fields[0].type_name, "i32");
        assert_eq!(inst.methods[0].parameters[0].1, "i32");
    }

    #[test]
    fn test_generic_arity_mismatch() {
        let mut reg = TypeRegistry::new();
        let map = TypeDescriptor::new(0, "Map", TypeKind::Generic)
            .with_generic_param("K")
            .with_generic_param("V");
        reg.register(map).unwrap();

        let err = reg.instantiate_generic("Map", &["String"]).unwrap_err();
        match err {
            TypeRegistryError::GenericArityMismatch { expected, got, .. } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_all_fields_inherited() {
        let mut reg = TypeRegistry::new();
        reg.register(
            TypeDescriptor::new(0, "Base", TypeKind::Struct)
                .with_field(FieldDescriptor::required("id", "i32")),
        )
        .unwrap();
        reg.register(
            TypeDescriptor::new(0, "Child", TypeKind::Struct)
                .with_parent("Base")
                .with_field(FieldDescriptor::required("name", "String")),
        )
        .unwrap();

        let all = reg.all_fields("Child").unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].name, "id");
        assert_eq!(all[1].name, "name");
    }

    #[test]
    fn test_all_methods_inherited() {
        let mut reg = TypeRegistry::new();
        reg.register(
            TypeDescriptor::new(0, "Base", TypeKind::Struct)
                .with_method(MethodDescriptor::new("base_method")),
        )
        .unwrap();
        reg.register(
            TypeDescriptor::new(0, "Child", TypeKind::Struct)
                .with_parent("Base")
                .with_method(MethodDescriptor::new("child_method")),
        )
        .unwrap();

        let all = reg.all_methods("Child").unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].name, "base_method");
        assert_eq!(all[1].name, "child_method");
    }

    #[test]
    fn test_metadata() {
        let mut reg = TypeRegistry::new();
        reg.register(
            TypeDescriptor::new(0, "T", TypeKind::Struct)
                .with_metadata("table", "users"),
        )
        .unwrap();
        assert_eq!(reg.get_metadata("T", "table").unwrap(), "users");
        assert!(reg.get_metadata("T", "nope").is_err());

        reg.set_metadata("T", "version", "2").unwrap();
        assert_eq!(reg.get_metadata("T", "version").unwrap(), "2");
    }

    #[test]
    fn test_implementors_of() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeDescriptor::new(0, "Serializable", TypeKind::Interface)).unwrap();
        reg.register(
            TypeDescriptor::new(0, "User", TypeKind::Struct).with_interface("Serializable"),
        )
        .unwrap();
        reg.register(
            TypeDescriptor::new(0, "Product", TypeKind::Struct).with_interface("Serializable"),
        )
        .unwrap();
        reg.register(TypeDescriptor::new(0, "Log", TypeKind::Struct)).unwrap();

        let impls = reg.implementors_of("Serializable");
        assert_eq!(impls, vec!["Product", "User"]);
    }

    #[test]
    fn test_search() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeDescriptor::new(0, "UserProfile", TypeKind::Struct)).unwrap();
        reg.register(TypeDescriptor::new(0, "UserSettings", TypeKind::Struct)).unwrap();
        reg.register(TypeDescriptor::new(0, "Product", TypeKind::Struct)).unwrap();

        let found = reg.search("user");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_types_of_kind() {
        let mut reg = make_registry();
        reg.register(TypeDescriptor::new(0, "User", TypeKind::Struct)).unwrap();
        let prims = reg.types_of_kind(&TypeKind::Primitive);
        assert_eq!(prims.len(), 3);
        let structs = reg.types_of_kind(&TypeKind::Struct);
        assert_eq!(structs.len(), 1);
    }

    #[test]
    fn test_field_descriptor_builder() {
        let f = FieldDescriptor::required("age", "i32")
            .with_default("18")
            .with_doc("The user's age");
        assert_eq!(f.name, "age");
        assert!(!f.optional);
        assert_eq!(f.default_value.as_deref(), Some("18"));
        assert_eq!(f.doc.as_deref(), Some("The user's age"));
    }

    #[test]
    fn test_empty_registry() {
        let reg = TypeRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.type_names().is_empty());
    }

    #[test]
    fn test_custom_id() {
        let mut reg = TypeRegistry::new();
        let desc = TypeDescriptor::new(100, "Custom", TypeKind::Struct);
        let id = reg.register(desc).unwrap();
        assert_eq!(id, 100);
        assert!(reg.get_by_id(100).is_some());
    }

    #[test]
    fn test_generic_with_return_type() {
        let mut reg = TypeRegistry::new();
        let desc = TypeDescriptor::new(0, "Container", TypeKind::Generic)
            .with_generic_param("T")
            .with_method(MethodDescriptor::new("get").with_return("T"));
        reg.register(desc).unwrap();

        let inst = reg.instantiate_generic("Container", &["String"]).unwrap();
        assert_eq!(inst.methods[0].return_type.as_deref(), Some("String"));
    }

    #[test]
    fn test_doc_on_type() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeDescriptor::new(0, "Foo", TypeKind::Struct).with_doc("A foo"))
            .unwrap();
        assert_eq!(reg.get("Foo").unwrap().doc.as_deref(), Some("A foo"));
    }
}
