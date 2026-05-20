//! Schema evolution tracking.
//!
//! Replaces Avro schema registries, Protobuf schema evolution tools, and similar
//! libraries with a pure-Rust schema registry. Tracks schema versions, checks
//! backward/forward compatibility, tracks field addition/removal/rename, generates
//! migrations, computes schema diffs, and maintains a version history.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors from schema evolution operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaEvolutionError {
    /// Schema not found.
    SchemaNotFound(String),
    /// Version not found.
    VersionNotFound { schema: String, version: u32 },
    /// Incompatible schema change.
    IncompatibleChange(String),
    /// Duplicate schema name.
    DuplicateName(String),
    /// Invalid field definition.
    InvalidField(String),
}

impl fmt::Display for SchemaEvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaNotFound(name) => write!(f, "schema not found: {name}"),
            Self::VersionNotFound { schema, version } => {
                write!(f, "version {version} not found for schema {schema}")
            }
            Self::IncompatibleChange(msg) => write!(f, "incompatible change: {msg}"),
            Self::DuplicateName(name) => write!(f, "duplicate schema name: {name}"),
            Self::InvalidField(msg) => write!(f, "invalid field: {msg}"),
        }
    }
}

impl std::error::Error for SchemaEvolutionError {}

// ── Field type ───────────────────────────────────────────────────

/// Data type of a schema field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Bytes,
    Array(Box<FieldType>),
    Map(Box<FieldType>),
    Optional(Box<FieldType>),
    Record(String), // name of another schema
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Integer => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::Boolean => write!(f, "boolean"),
            Self::Bytes => write!(f, "bytes"),
            Self::Array(inner) => write!(f, "array<{inner}>"),
            Self::Map(inner) => write!(f, "map<{inner}>"),
            Self::Optional(inner) => write!(f, "optional<{inner}>"),
            Self::Record(name) => write!(f, "record<{name}>"),
        }
    }
}

// ── Field definition ─────────────────────────────────────────────

/// A field in a schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    /// Field name.
    pub name: String,
    /// Field type.
    pub field_type: FieldType,
    /// Whether this field is required.
    pub required: bool,
    /// Default value (if any).
    pub default_value: Option<serde_json::Value>,
    /// Documentation.
    pub doc: Option<String>,
}

impl FieldDef {
    /// Create a new required field.
    pub fn required(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: true,
            default_value: None,
            doc: None,
        }
    }

    /// Create a new optional field.
    pub fn optional(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: false,
            default_value: None,
            doc: None,
        }
    }

    /// Set a default value.
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default_value = Some(default);
        self
    }

    /// Set documentation.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }
}

// ── Schema version ───────────────────────────────────────────────

/// A specific version of a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Schema name.
    pub name: String,
    /// Version number (1-based).
    pub version: u32,
    /// Fields in this version.
    pub fields: Vec<FieldDef>,
    /// Timestamp of creation (ISO 8601 string).
    pub created_at: String,
    /// Optional description of changes.
    pub description: Option<String>,
}

impl SchemaVersion {
    /// Get a field by name.
    pub fn get_field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }
}

// ── Schema diff ──────────────────────────────────────────────────

/// A change between two schema versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaChange {
    /// Field was added.
    FieldAdded(FieldDef),
    /// Field was removed.
    FieldRemoved(FieldDef),
    /// Field was renamed.
    FieldRenamed { old_name: String, new_name: String },
    /// Field type changed.
    TypeChanged {
        field: String,
        old_type: FieldType,
        new_type: FieldType,
    },
    /// Field changed from required to optional.
    MadeOptional(String),
    /// Field changed from optional to required.
    MadeRequired(String),
    /// Default value changed.
    DefaultChanged {
        field: String,
        old_default: Option<serde_json::Value>,
        new_default: Option<serde_json::Value>,
    },
}

impl fmt::Display for SchemaChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldAdded(fd) => write!(f, "add field '{}' ({})", fd.name, fd.field_type),
            Self::FieldRemoved(fd) => write!(f, "remove field '{}' ({})", fd.name, fd.field_type),
            Self::FieldRenamed { old_name, new_name } => {
                write!(f, "rename '{old_name}' -> '{new_name}'")
            }
            Self::TypeChanged { field, old_type, new_type } => {
                write!(f, "change type of '{field}': {old_type} -> {new_type}")
            }
            Self::MadeOptional(name) => write!(f, "make '{name}' optional"),
            Self::MadeRequired(name) => write!(f, "make '{name}' required"),
            Self::DefaultChanged { field, .. } => write!(f, "change default of '{field}'"),
        }
    }
}

/// A diff between two schema versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDiff {
    /// Schema name.
    pub schema_name: String,
    /// From version.
    pub from_version: u32,
    /// To version.
    pub to_version: u32,
    /// List of changes.
    pub changes: Vec<SchemaChange>,
}

impl SchemaDiff {
    /// Whether there are any changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Number of changes.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Fields that were added.
    pub fn added_fields(&self) -> Vec<&FieldDef> {
        self.changes
            .iter()
            .filter_map(|c| match c {
                SchemaChange::FieldAdded(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    /// Fields that were removed.
    pub fn removed_fields(&self) -> Vec<&FieldDef> {
        self.changes
            .iter()
            .filter_map(|c| match c {
                SchemaChange::FieldRemoved(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

// ── Compatibility ────────────────────────────────────────────────

/// Schema compatibility level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compatibility {
    /// New schema can read data written with old schema.
    Backward,
    /// Old schema can read data written with new schema.
    Forward,
    /// Both backward and forward compatible.
    Full,
    /// No compatibility check.
    None,
}

/// Result of a compatibility check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// Whether the schemas are compatible.
    pub compatible: bool,
    /// Compatibility level checked.
    pub level: Compatibility,
    /// Violations found.
    pub violations: Vec<String>,
}

// ── Migration step ───────────────────────────────────────────────

/// A migration step to transform data from one schema version to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStep {
    /// Add a field with a default value.
    AddField { name: String, default: serde_json::Value },
    /// Remove a field.
    RemoveField(String),
    /// Rename a field.
    RenameField { from: String, to: String },
    /// Cast a field to a new type.
    CastField { name: String, new_type: FieldType },
    /// Set a default value for missing fields.
    SetDefault { name: String, value: serde_json::Value },
}

impl fmt::Display for MigrationStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddField { name, .. } => write!(f, "ADD FIELD {name}"),
            Self::RemoveField(name) => write!(f, "REMOVE FIELD {name}"),
            Self::RenameField { from, to } => write!(f, "RENAME {from} -> {to}"),
            Self::CastField { name, new_type } => write!(f, "CAST {name} TO {new_type}"),
            Self::SetDefault { name, value } => write!(f, "SET DEFAULT {name} = {value}"),
        }
    }
}

/// A migration plan from one version to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Schema name.
    pub schema_name: String,
    /// Source version.
    pub from_version: u32,
    /// Target version.
    pub to_version: u32,
    /// Ordered list of migration steps.
    pub steps: Vec<MigrationStep>,
}

// ── Schema Registry ──────────────────────────────────────────────

/// The schema registry that stores and manages schema versions.
#[derive(Debug, Clone)]
pub struct SchemaRegistry {
    /// All schema versions, keyed by (schema_name, version).
    schemas: HashMap<String, Vec<SchemaVersion>>,
    /// Default compatibility level for the registry.
    compatibility: Compatibility,
}

impl SchemaRegistry {
    /// Create a new schema registry.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            compatibility: Compatibility::Backward,
        }
    }

    /// Set the default compatibility level.
    pub fn set_compatibility(&mut self, compatibility: Compatibility) {
        self.compatibility = compatibility;
    }

    /// Register a new schema (first version).
    pub fn register(
        &mut self,
        name: impl Into<String>,
        fields: Vec<FieldDef>,
    ) -> Result<&SchemaVersion, SchemaEvolutionError> {
        let name = name.into();
        if self.schemas.contains_key(&name) {
            return Err(SchemaEvolutionError::DuplicateName(name));
        }

        let version = SchemaVersion {
            name: name.clone(),
            version: 1,
            fields,
            created_at: "2026-03-09T00:00:00Z".to_string(),
            description: Some("Initial version".into()),
        };

        let versions = self.schemas.entry(name).or_default();
        versions.push(version);
        Ok(versions.last().unwrap())
    }

    /// Evolve a schema to a new version.
    pub fn evolve(
        &mut self,
        name: &str,
        new_fields: Vec<FieldDef>,
        description: Option<String>,
    ) -> Result<&SchemaVersion, SchemaEvolutionError> {
        let versions = self
            .schemas
            .get(name)
            .ok_or_else(|| SchemaEvolutionError::SchemaNotFound(name.to_string()))?;

        let latest = versions.last().unwrap();
        let new_version_num = latest.version + 1;

        // Check compatibility if not None.
        if self.compatibility != Compatibility::None {
            let compat = self.check_compatibility_internal(latest, &new_fields);
            if !compat.compatible {
                return Err(SchemaEvolutionError::IncompatibleChange(
                    compat.violations.join("; "),
                ));
            }
        }

        let version = SchemaVersion {
            name: name.to_string(),
            version: new_version_num,
            fields: new_fields,
            created_at: "2026-03-09T00:00:00Z".to_string(),
            description,
        };

        let versions = self.schemas.get_mut(name).unwrap();
        versions.push(version);
        Ok(versions.last().unwrap())
    }

    /// Get a specific version of a schema.
    pub fn get_version(
        &self,
        name: &str,
        version: u32,
    ) -> Result<&SchemaVersion, SchemaEvolutionError> {
        let versions = self
            .schemas
            .get(name)
            .ok_or_else(|| SchemaEvolutionError::SchemaNotFound(name.to_string()))?;

        versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| SchemaEvolutionError::VersionNotFound {
                schema: name.to_string(),
                version,
            })
    }

    /// Get the latest version of a schema.
    pub fn get_latest(&self, name: &str) -> Result<&SchemaVersion, SchemaEvolutionError> {
        let versions = self
            .schemas
            .get(name)
            .ok_or_else(|| SchemaEvolutionError::SchemaNotFound(name.to_string()))?;

        versions
            .last()
            .ok_or_else(|| SchemaEvolutionError::SchemaNotFound(name.to_string()))
    }

    /// Get all versions of a schema.
    pub fn get_versions(&self, name: &str) -> Result<&[SchemaVersion], SchemaEvolutionError> {
        self.schemas
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| SchemaEvolutionError::SchemaNotFound(name.to_string()))
    }

    /// Number of schemas registered.
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    /// Compute diff between two versions.
    pub fn diff(
        &self,
        name: &str,
        from_version: u32,
        to_version: u32,
    ) -> Result<SchemaDiff, SchemaEvolutionError> {
        let from = self.get_version(name, from_version)?;
        let to = self.get_version(name, to_version)?;

        let changes = compute_changes(&from.fields, &to.fields);

        Ok(SchemaDiff {
            schema_name: name.to_string(),
            from_version,
            to_version,
            changes,
        })
    }

    /// Check compatibility between old and new schemas.
    pub fn check_compatibility(
        &self,
        name: &str,
        new_fields: &[FieldDef],
    ) -> Result<CompatibilityResult, SchemaEvolutionError> {
        let latest = self.get_latest(name)?;
        Ok(self.check_compatibility_internal(latest, new_fields))
    }

    /// Generate a migration plan between two versions.
    pub fn generate_migration(
        &self,
        name: &str,
        from_version: u32,
        to_version: u32,
    ) -> Result<MigrationPlan, SchemaEvolutionError> {
        let diff = self.diff(name, from_version, to_version)?;

        let mut steps = Vec::new();
        for change in &diff.changes {
            match change {
                SchemaChange::FieldAdded(fd) => {
                    let default = fd
                        .default_value
                        .clone()
                        .unwrap_or(serde_json::Value::Null);
                    steps.push(MigrationStep::AddField {
                        name: fd.name.clone(),
                        default,
                    });
                }
                SchemaChange::FieldRemoved(fd) => {
                    steps.push(MigrationStep::RemoveField(fd.name.clone()));
                }
                SchemaChange::FieldRenamed { old_name, new_name } => {
                    steps.push(MigrationStep::RenameField {
                        from: old_name.clone(),
                        to: new_name.clone(),
                    });
                }
                SchemaChange::TypeChanged { field, new_type, .. } => {
                    steps.push(MigrationStep::CastField {
                        name: field.clone(),
                        new_type: new_type.clone(),
                    });
                }
                SchemaChange::DefaultChanged { field, new_default, .. } => {
                    if let Some(val) = new_default {
                        steps.push(MigrationStep::SetDefault {
                            name: field.clone(),
                            value: val.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(MigrationPlan {
            schema_name: name.to_string(),
            from_version,
            to_version,
            steps,
        })
    }

    // ── Private helpers ──

    fn check_compatibility_internal(
        &self,
        old: &SchemaVersion,
        new_fields: &[FieldDef],
    ) -> CompatibilityResult {
        let mut violations = Vec::new();
        let level = self.compatibility;

        let old_map: HashMap<&str, &FieldDef> =
            old.fields.iter().map(|f| (f.name.as_str(), f)).collect();
        let new_map: HashMap<&str, &FieldDef> =
            new_fields.iter().map(|f| (f.name.as_str(), f)).collect();

        match level {
            Compatibility::Backward => {
                // New schema can read old data.
                // Removing a required field without default breaks backward compat.
                // Adding a new required field without default breaks backward compat.
                for new_f in new_fields {
                    if !old_map.contains_key(new_f.name.as_str())
                        && new_f.required
                        && new_f.default_value.is_none()
                    {
                        violations.push(format!(
                            "new required field '{}' without default",
                            new_f.name
                        ));
                    }
                }
            }
            Compatibility::Forward => {
                // Old schema can read new data.
                // Removing a field that old schema expects breaks forward compat.
                for old_f in &old.fields {
                    if !new_map.contains_key(old_f.name.as_str()) && old_f.required {
                        violations.push(format!(
                            "required field '{}' removed",
                            old_f.name
                        ));
                    }
                }
            }
            Compatibility::Full => {
                // Both backward and forward.
                for new_f in new_fields {
                    if !old_map.contains_key(new_f.name.as_str())
                        && new_f.required
                        && new_f.default_value.is_none()
                    {
                        violations.push(format!(
                            "new required field '{}' without default (backward)",
                            new_f.name
                        ));
                    }
                }
                for old_f in &old.fields {
                    if !new_map.contains_key(old_f.name.as_str()) && old_f.required {
                        violations.push(format!(
                            "required field '{}' removed (forward)",
                            old_f.name
                        ));
                    }
                }
            }
            Compatibility::None => {}
        }

        // Type changes.
        for new_f in new_fields {
            if let Some(old_f) = old_map.get(new_f.name.as_str()) {
                if old_f.field_type != new_f.field_type {
                    violations.push(format!(
                        "type of '{}' changed from {} to {}",
                        new_f.name, old_f.field_type, new_f.field_type
                    ));
                }
            }
        }

        CompatibilityResult {
            compatible: violations.is_empty(),
            level,
            violations,
        }
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the list of changes between old and new fields.
fn compute_changes(old_fields: &[FieldDef], new_fields: &[FieldDef]) -> Vec<SchemaChange> {
    let mut changes = Vec::new();

    let old_map: HashMap<&str, &FieldDef> =
        old_fields.iter().map(|f| (f.name.as_str(), f)).collect();
    let new_map: HashMap<&str, &FieldDef> =
        new_fields.iter().map(|f| (f.name.as_str(), f)).collect();

    // Added fields (in new but not in old).
    for nf in new_fields {
        if !old_map.contains_key(nf.name.as_str()) {
            changes.push(SchemaChange::FieldAdded(nf.clone()));
        }
    }

    // Removed fields (in old but not in new).
    for of in old_fields {
        if !new_map.contains_key(of.name.as_str()) {
            changes.push(SchemaChange::FieldRemoved(of.clone()));
        }
    }

    // Changed fields (in both).
    for nf in new_fields {
        if let Some(of) = old_map.get(nf.name.as_str()) {
            if of.field_type != nf.field_type {
                changes.push(SchemaChange::TypeChanged {
                    field: nf.name.clone(),
                    old_type: of.field_type.clone(),
                    new_type: nf.field_type.clone(),
                });
            }
            if of.required && !nf.required {
                changes.push(SchemaChange::MadeOptional(nf.name.clone()));
            }
            if !of.required && nf.required {
                changes.push(SchemaChange::MadeRequired(nf.name.clone()));
            }
            if of.default_value != nf.default_value {
                changes.push(SchemaChange::DefaultChanged {
                    field: nf.name.clone(),
                    old_default: of.default_value.clone(),
                    new_default: nf.default_value.clone(),
                });
            }
        }
    }

    changes
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_fields() -> Vec<FieldDef> {
        vec![
            FieldDef::required("id", FieldType::Integer),
            FieldDef::required("name", FieldType::String),
        ]
    }

    #[test]
    fn register_schema() {
        let mut reg = SchemaRegistry::new();
        let v = reg.register("user", basic_fields()).unwrap();
        assert_eq!(v.version, 1);
        assert_eq!(v.name, "user");
        assert_eq!(v.fields.len(), 2);
    }

    #[test]
    fn register_duplicate_fails() {
        let mut reg = SchemaRegistry::new();
        reg.register("user", basic_fields()).unwrap();
        let err = reg.register("user", basic_fields()).unwrap_err();
        assert!(matches!(err, SchemaEvolutionError::DuplicateName(_)));
    }

    #[test]
    fn evolve_schema() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();

        let mut new_fields = basic_fields();
        new_fields.push(FieldDef::optional("email", FieldType::String));

        let v = reg.evolve("user", new_fields, Some("Add email".into())).unwrap();
        assert_eq!(v.version, 2);
        assert_eq!(v.fields.len(), 3);
    }

    #[test]
    fn get_specific_version() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();
        reg.evolve("user", basic_fields(), None).unwrap();

        let v1 = reg.get_version("user", 1).unwrap();
        assert_eq!(v1.version, 1);
        let v2 = reg.get_version("user", 2).unwrap();
        assert_eq!(v2.version, 2);
    }

    #[test]
    fn get_version_not_found() {
        let mut reg = SchemaRegistry::new();
        reg.register("user", basic_fields()).unwrap();
        let err = reg.get_version("user", 99).unwrap_err();
        assert!(matches!(err, SchemaEvolutionError::VersionNotFound { .. }));
    }

    #[test]
    fn get_latest() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();
        reg.evolve("user", basic_fields(), None).unwrap();
        reg.evolve("user", basic_fields(), None).unwrap();

        let latest = reg.get_latest("user").unwrap();
        assert_eq!(latest.version, 3);
    }

    #[test]
    fn schema_not_found() {
        let reg = SchemaRegistry::new();
        let err = reg.get_latest("nonexistent").unwrap_err();
        assert!(matches!(err, SchemaEvolutionError::SchemaNotFound(_)));
    }

    #[test]
    fn diff_added_field() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();

        let mut v2_fields = basic_fields();
        v2_fields.push(FieldDef::optional("email", FieldType::String));
        reg.evolve("user", v2_fields, None).unwrap();

        let diff = reg.diff("user", 1, 2).unwrap();
        assert!(!diff.is_empty());
        assert_eq!(diff.added_fields().len(), 1);
        assert_eq!(diff.added_fields()[0].name, "email");
    }

    #[test]
    fn diff_removed_field() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();

        let v2_fields = vec![FieldDef::required("id", FieldType::Integer)];
        reg.evolve("user", v2_fields, None).unwrap();

        let diff = reg.diff("user", 1, 2).unwrap();
        assert_eq!(diff.removed_fields().len(), 1);
        assert_eq!(diff.removed_fields()[0].name, "name");
    }

    #[test]
    fn diff_type_changed() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();

        let v2_fields = vec![
            FieldDef::required("id", FieldType::String), // was Integer
            FieldDef::required("name", FieldType::String),
        ];
        reg.evolve("user", v2_fields, None).unwrap();

        let diff = reg.diff("user", 1, 2).unwrap();
        let type_changes: Vec<_> = diff
            .changes
            .iter()
            .filter(|c| matches!(c, SchemaChange::TypeChanged { .. }))
            .collect();
        assert_eq!(type_changes.len(), 1);
    }

    #[test]
    fn diff_made_optional() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();

        let v2_fields = vec![
            FieldDef::required("id", FieldType::Integer),
            FieldDef::optional("name", FieldType::String), // was required
        ];
        reg.evolve("user", v2_fields, None).unwrap();

        let diff = reg.diff("user", 1, 2).unwrap();
        let optional_changes: Vec<_> = diff
            .changes
            .iter()
            .filter(|c| matches!(c, SchemaChange::MadeOptional(_)))
            .collect();
        assert_eq!(optional_changes.len(), 1);
    }

    #[test]
    fn backward_compatibility_ok() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::Backward);
        reg.register("user", basic_fields()).unwrap();

        // Adding an optional field is backward compatible.
        let mut v2_fields = basic_fields();
        v2_fields.push(FieldDef::optional("email", FieldType::String));

        let result = reg.check_compatibility("user", &v2_fields).unwrap();
        assert!(result.compatible);
    }

    #[test]
    fn backward_compatibility_fails() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::Backward);
        reg.register("user", basic_fields()).unwrap();

        // Adding a required field without default breaks backward compat.
        let mut v2_fields = basic_fields();
        v2_fields.push(FieldDef::required("email", FieldType::String));

        let result = reg.check_compatibility("user", &v2_fields).unwrap();
        assert!(!result.compatible);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn forward_compatibility_fails() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::Forward);
        reg.register("user", basic_fields()).unwrap();

        // Removing a required field breaks forward compat.
        let v2_fields = vec![FieldDef::required("id", FieldType::Integer)];

        let result = reg.check_compatibility("user", &v2_fields).unwrap();
        assert!(!result.compatible);
    }

    #[test]
    fn evolve_blocked_by_compatibility() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::Backward);
        reg.register("user", basic_fields()).unwrap();

        let mut v2_fields = basic_fields();
        v2_fields.push(FieldDef::required("email", FieldType::String));

        let err = reg.evolve("user", v2_fields, None).unwrap_err();
        assert!(matches!(err, SchemaEvolutionError::IncompatibleChange(_)));
    }

    #[test]
    fn generate_migration() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();

        let mut v2_fields = basic_fields();
        v2_fields.push(
            FieldDef::optional("email", FieldType::String)
                .with_default(serde_json::json!("")),
        );
        reg.evolve("user", v2_fields, None).unwrap();

        let plan = reg.generate_migration("user", 1, 2).unwrap();
        assert_eq!(plan.from_version, 1);
        assert_eq!(plan.to_version, 2);
        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn field_type_display() {
        assert_eq!(format!("{}", FieldType::String), "string");
        assert_eq!(
            format!("{}", FieldType::Array(Box::new(FieldType::Integer))),
            "array<integer>"
        );
        assert_eq!(
            format!("{}", FieldType::Optional(Box::new(FieldType::Boolean))),
            "optional<boolean>"
        );
    }

    #[test]
    fn schema_change_display() {
        let c = SchemaChange::FieldAdded(FieldDef::required("email", FieldType::String));
        assert!(format!("{c}").contains("add field"));
    }

    #[test]
    fn migration_step_display() {
        let s = MigrationStep::AddField {
            name: "email".into(),
            default: serde_json::json!(""),
        };
        assert!(format!("{s}").contains("ADD FIELD"));
    }

    #[test]
    fn get_all_versions() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();
        reg.evolve("user", basic_fields(), None).unwrap();
        reg.evolve("user", basic_fields(), None).unwrap();

        let versions = reg.get_versions("user").unwrap();
        assert_eq!(versions.len(), 3);
    }

    #[test]
    fn schema_count() {
        let mut reg = SchemaRegistry::new();
        reg.register("user", basic_fields()).unwrap();
        reg.register("order", basic_fields()).unwrap();
        assert_eq!(reg.schema_count(), 2);
    }

    #[test]
    fn field_def_with_doc() {
        let f = FieldDef::required("id", FieldType::Integer)
            .with_doc("Primary key");
        assert_eq!(f.doc, Some("Primary key".to_string()));
    }

    #[test]
    fn error_display() {
        let e = SchemaEvolutionError::SchemaNotFound("x".into());
        assert!(format!("{e}").contains("schema not found"));
    }

    #[test]
    fn diff_no_changes() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::None);
        reg.register("user", basic_fields()).unwrap();
        reg.evolve("user", basic_fields(), None).unwrap();

        let diff = reg.diff("user", 1, 2).unwrap();
        assert!(diff.is_empty());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn full_compatibility_check() {
        let mut reg = SchemaRegistry::new();
        reg.set_compatibility(Compatibility::Full);
        reg.register("user", basic_fields()).unwrap();

        // Adding optional field is fully compatible.
        let mut v2_fields = basic_fields();
        v2_fields.push(FieldDef::optional("email", FieldType::String));

        let result = reg.check_compatibility("user", &v2_fields).unwrap();
        assert!(result.compatible);
    }
}
