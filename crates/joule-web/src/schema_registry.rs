//! Schema registry — versioned schemas with compatibility checking, evolution
//! rules, caching, and fingerprint-based lookup.

use std::collections::HashMap;

/// Schema compatibility mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    None,
    Backward,
    Forward,
    Full,
}

/// A field in a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaField {
    pub name: String,
    pub field_type: FieldType,
    pub required: bool,
    pub default_value: Option<String>,
}

/// Supported field types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    String,
    Int,
    Long,
    Float,
    Double,
    Bool,
    Bytes,
    Array(Box<FieldType>),
    Map(Box<FieldType>),
    Nullable(Box<FieldType>),
}

impl FieldType {
    /// Check if this type is promotable to another.
    pub fn is_promotable_to(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (FieldType::Int, FieldType::Long)
                | (FieldType::Int, FieldType::Float)
                | (FieldType::Int, FieldType::Double)
                | (FieldType::Long, FieldType::Double)
                | (FieldType::Float, FieldType::Double)
        )
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other || self.is_promotable_to(other)
    }
}

/// A versioned schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub name: String,
    pub version: u32,
    pub fields: Vec<SchemaField>,
    pub fingerprint: u64,
}

impl Schema {
    pub fn new(name: &str, version: u32, fields: Vec<SchemaField>) -> Self {
        let fingerprint = compute_fingerprint(name, &fields);
        Self { name: name.to_string(), version, fields, fingerprint }
    }

    pub fn get_field(&self, name: &str) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn required_fields(&self) -> Vec<&SchemaField> {
        self.fields.iter().filter(|f| f.required).collect()
    }
}

fn compute_fingerprint(name: &str, fields: &[SchemaField]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for f in fields {
        for b in f.name.bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= f.required as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Result of a compatibility check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatResult {
    pub compatible: bool,
    pub errors: Vec<String>,
}

/// Check backward compatibility (new schema can read old data).
pub fn check_backward(old: &Schema, new: &Schema) -> CompatResult {
    let mut errors = Vec::new();

    // New required fields without defaults are breaking.
    for nf in &new.fields {
        if nf.required && nf.default_value.is_none() {
            if old.get_field(&nf.name).is_none() {
                errors.push(format!("New required field '{}' without default", nf.name));
            }
        }
    }

    // Type changes must be compatible.
    for of in &old.fields {
        if let Some(nf) = new.get_field(&of.name) {
            if !of.field_type.is_compatible_with(&nf.field_type) {
                errors.push(format!("Incompatible type change for '{}'", of.name));
            }
        }
        // Removed fields are OK for backward compat (old data has it, new schema ignores it).
    }

    CompatResult { compatible: errors.is_empty(), errors }
}

/// Check forward compatibility (old schema can read new data).
pub fn check_forward(old: &Schema, new: &Schema) -> CompatResult {
    let mut errors = Vec::new();

    // Removing required fields breaks forward compat (new data may omit it).
    for of in &old.fields {
        if of.required {
            if new.get_field(&of.name).is_none() {
                errors.push(format!("Required field '{}' removed", of.name));
            }
        }
    }

    // Type changes.
    for nf in &new.fields {
        if let Some(of) = old.get_field(&nf.name) {
            if !nf.field_type.is_compatible_with(&of.field_type) {
                errors.push(format!("Incompatible type change for '{}'", nf.name));
            }
        }
    }

    CompatResult { compatible: errors.is_empty(), errors }
}

/// Check full compatibility (both directions).
pub fn check_full(old: &Schema, new: &Schema) -> CompatResult {
    let bw = check_backward(old, new);
    let fw = check_forward(old, new);
    let mut errors = bw.errors;
    errors.extend(fw.errors);
    CompatResult { compatible: errors.is_empty(), errors }
}

/// The schema registry.
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    subjects: HashMap<String, SubjectEntry>,
    by_fingerprint: HashMap<u64, (String, u32)>,
}

#[derive(Debug)]
struct SubjectEntry {
    schemas: Vec<Schema>,
    compatibility: Compatibility,
}

impl Default for SubjectEntry {
    fn default() -> Self {
        Self { schemas: Vec::new(), compatibility: Compatibility::None }
    }
}

impl SchemaRegistry {
    pub fn new() -> Self { Self::default() }

    /// Set compatibility mode for a subject.
    pub fn set_compatibility(&mut self, subject: &str, mode: Compatibility) {
        self.subjects.entry(subject.to_string()).or_default().compatibility = mode;
    }

    /// Get the compatibility mode for a subject.
    pub fn get_compatibility(&self, subject: &str) -> Compatibility {
        self.subjects.get(subject).map(|e| e.compatibility).unwrap_or(Compatibility::None)
    }

    /// Register a new schema version. Returns Ok(version) or Err with compat errors.
    pub fn register(&mut self, subject: &str, fields: Vec<SchemaField>) -> Result<u32, Vec<String>> {
        let entry = self.subjects.entry(subject.to_string()).or_default();
        let version = entry.schemas.len() as u32 + 1;
        let schema = Schema::new(subject, version, fields);

        // Check compatibility against last version.
        if let Some(last) = entry.schemas.last() {
            let result = match entry.compatibility {
                Compatibility::None => CompatResult { compatible: true, errors: vec![] },
                Compatibility::Backward => check_backward(last, &schema),
                Compatibility::Forward => check_forward(last, &schema),
                Compatibility::Full => check_full(last, &schema),
            };
            if !result.compatible {
                return Err(result.errors);
            }
        }

        self.by_fingerprint.insert(schema.fingerprint, (subject.to_string(), version));
        entry.schemas.push(schema);
        Ok(version)
    }

    /// Get a schema by subject and version.
    pub fn get(&self, subject: &str, version: u32) -> Option<&Schema> {
        self.subjects.get(subject)?.schemas.get(version as usize - 1)
    }

    /// Get the latest schema for a subject.
    pub fn get_latest(&self, subject: &str) -> Option<&Schema> {
        self.subjects.get(subject)?.schemas.last()
    }

    /// Look up a schema by fingerprint.
    pub fn get_by_fingerprint(&self, fp: u64) -> Option<&Schema> {
        let (subject, version) = self.by_fingerprint.get(&fp)?;
        self.get(subject, *version)
    }

    /// List all versions for a subject.
    pub fn versions(&self, subject: &str) -> Vec<u32> {
        self.subjects.get(subject)
            .map(|e| (1..=e.schemas.len() as u32).collect())
            .unwrap_or_default()
    }

    /// List all subjects.
    pub fn subjects(&self) -> Vec<String> {
        let mut s: Vec<_> = self.subjects.keys().cloned().collect();
        s.sort();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, ft: FieldType, required: bool) -> SchemaField {
        SchemaField { name: name.to_string(), field_type: ft, required, default_value: None }
    }

    fn field_with_default(name: &str, ft: FieldType, default: &str) -> SchemaField {
        SchemaField { name: name.to_string(), field_type: ft, required: true, default_value: Some(default.to_string()) }
    }

    #[test]
    fn register_first_version() {
        let mut reg = SchemaRegistry::new();
        let v = reg.register("user", vec![field("name", FieldType::String, true)]).unwrap();
        assert_eq!(v, 1);
    }

    #[test]
    fn register_multiple_versions() {
        let mut reg = SchemaRegistry::new();
        reg.register("user", vec![field("name", FieldType::String, true)]).unwrap();
        let v2 = reg.register("user", vec![
            field("name", FieldType::String, true),
            field("email", FieldType::String, false),
        ]).unwrap();
        assert_eq!(v2, 2);
    }

    #[test]
    fn backward_compat_add_optional_field() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("user", Compatibility::Backward);
        reg.register("user", vec![field("name", FieldType::String, true)]).unwrap();
        // Adding optional field is backward compatible.
        let r = reg.register("user", vec![
            field("name", FieldType::String, true),
            field("age", FieldType::Int, false),
        ]);
        assert!(r.is_ok());
    }

    #[test]
    fn backward_compat_add_required_no_default_fails() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("user", Compatibility::Backward);
        reg.register("user", vec![field("name", FieldType::String, true)]).unwrap();
        let r = reg.register("user", vec![
            field("name", FieldType::String, true),
            field("age", FieldType::Int, true), // required, no default
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn backward_compat_add_required_with_default_ok() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("user", Compatibility::Backward);
        reg.register("user", vec![field("name", FieldType::String, true)]).unwrap();
        let r = reg.register("user", vec![
            field("name", FieldType::String, true),
            field_with_default("age", FieldType::Int, "0"),
        ]);
        assert!(r.is_ok());
    }

    #[test]
    fn forward_compat_remove_required_fails() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("user", Compatibility::Forward);
        reg.register("user", vec![
            field("name", FieldType::String, true),
            field("email", FieldType::String, true),
        ]).unwrap();
        let r = reg.register("user", vec![
            field("name", FieldType::String, true),
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn type_promotion_int_to_long() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("data", Compatibility::Backward);
        reg.register("data", vec![field("count", FieldType::Int, true)]).unwrap();
        let r = reg.register("data", vec![field("count", FieldType::Long, true)]);
        assert!(r.is_ok());
    }

    #[test]
    fn incompatible_type_change() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("data", Compatibility::Backward);
        reg.register("data", vec![field("count", FieldType::Int, true)]).unwrap();
        let r = reg.register("data", vec![field("count", FieldType::String, true)]);
        assert!(r.is_err());
    }

    #[test]
    fn get_schema_by_version() {
        let mut reg = SchemaRegistry::new();
        reg.register("x", vec![field("a", FieldType::String, true)]).unwrap();
        reg.register("x", vec![field("a", FieldType::String, true), field("b", FieldType::Int, false)]).unwrap();
        let s1 = reg.get("x", 1).unwrap();
        assert_eq!(s1.fields.len(), 1);
        let s2 = reg.get("x", 2).unwrap();
        assert_eq!(s2.fields.len(), 2);
    }

    #[test]
    fn get_latest() {
        let mut reg = SchemaRegistry::new();
        reg.register("x", vec![field("a", FieldType::String, true)]).unwrap();
        reg.register("x", vec![field("a", FieldType::String, true), field("b", FieldType::Int, false)]).unwrap();
        let latest = reg.get_latest("x").unwrap();
        assert_eq!(latest.version, 2);
    }

    #[test]
    fn fingerprint_lookup() {
        let mut reg = SchemaRegistry::new();
        reg.register("event", vec![field("id", FieldType::String, true)]).unwrap();
        let s = reg.get("event", 1).unwrap();
        let fp = s.fingerprint;
        let found = reg.get_by_fingerprint(fp).unwrap();
        assert_eq!(found.name, "event");
    }

    #[test]
    fn list_versions() {
        let mut reg = SchemaRegistry::new();
        reg.register("x", vec![field("a", FieldType::String, true)]).unwrap();
        reg.register("x", vec![field("a", FieldType::String, true)]).unwrap();
        reg.register("x", vec![field("a", FieldType::String, true)]).unwrap();
        assert_eq!(reg.versions("x"), vec![1, 2, 3]);
    }

    #[test]
    fn list_subjects() {
        let mut reg = SchemaRegistry::new();
        reg.register("beta", vec![]).unwrap();
        reg.register("alpha", vec![]).unwrap();
        assert_eq!(reg.subjects(), vec!["alpha", "beta"]);
    }

    #[test]
    fn no_compat_allows_anything() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility("x", Compatibility::None);
        reg.register("x", vec![field("a", FieldType::String, true)]).unwrap();
        let r = reg.register("x", vec![field("b", FieldType::Int, true)]);
        assert!(r.is_ok());
    }

    #[test]
    fn full_compat_both_directions() {
        let old = Schema::new("s", 1, vec![field("a", FieldType::String, true)]);
        let new = Schema::new("s", 2, vec![
            field("a", FieldType::String, true),
            field("b", FieldType::String, true), // required, no default — fails backward
        ]);
        let r = check_full(&old, &new);
        assert!(!r.compatible);
    }
}
