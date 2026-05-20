//! Contract testing — consumer/provider contracts, pact-like format, contract
//! verification, schema compatibility, and breaking change detection.
//!
//! Replaces JS contract testing libraries (Pact, Spring Cloud Contract) with a
//! pure-Rust consumer-driven contract framework for verifying API compatibility.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Contract testing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// Contract not found.
    NotFound(String),
    /// Duplicate contract ID.
    Duplicate(String),
    /// Verification failed with details.
    VerificationFailed(Vec<ContractViolation>),
    /// Incompatible schema change.
    IncompatibleSchema(String),
    /// Missing required field.
    MissingField(String),
    /// Type mismatch.
    TypeMismatch { field: String, expected: String, actual: String },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "contract not found: {id}"),
            Self::Duplicate(id) => write!(f, "duplicate contract: {id}"),
            Self::VerificationFailed(violations) => {
                write!(f, "contract verification failed ({} violations)", violations.len())
            }
            Self::IncompatibleSchema(msg) => write!(f, "incompatible schema: {msg}"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::TypeMismatch { field, expected, actual } => {
                write!(f, "type mismatch for '{field}': expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for ContractError {}

// ── Schema Types ───────────────────────────────────────────────

/// Schema field type for contract validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array(Box<SchemaType>),
    Object(Vec<SchemaField>),
    Nullable(Box<SchemaType>),
    Any,
}

impl fmt::Display for SchemaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Number => write!(f, "number"),
            Self::Integer => write!(f, "integer"),
            Self::Boolean => write!(f, "boolean"),
            Self::Array(inner) => write!(f, "array<{inner}>"),
            Self::Object(_) => write!(f, "object"),
            Self::Nullable(inner) => write!(f, "nullable<{inner}>"),
            Self::Any => write!(f, "any"),
        }
    }
}

/// A field in a schema definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub field_type: SchemaType,
    pub required: bool,
    pub description: Option<String>,
}

// ── Interaction ────────────────────────────────────────────────

/// HTTP method for contract interactions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContractMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl fmt::Display for ContractMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Patch => write!(f, "PATCH"),
            Self::Delete => write!(f, "DELETE"),
        }
    }
}

/// A request in a contract interaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractRequest {
    pub method: ContractMethod,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body_schema: Option<SchemaType>,
    pub query_params: HashMap<String, String>,
}

/// A response in a contract interaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body_schema: Option<SchemaType>,
}

/// A single interaction (request-response pair) in a contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interaction {
    pub description: String,
    pub provider_state: Option<String>,
    pub request: ContractRequest,
    pub response: ContractResponse,
}

// ── Contract ───────────────────────────────────────────────────

/// A consumer-provider contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub id: String,
    pub consumer: String,
    pub provider: String,
    pub version: String,
    pub interactions: Vec<Interaction>,
    pub metadata: HashMap<String, String>,
}

impl Contract {
    /// Create a new contract.
    pub fn new(
        id: impl Into<String>,
        consumer: impl Into<String>,
        provider: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            consumer: consumer.into(),
            provider: provider.into(),
            version: version.into(),
            interactions: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add an interaction.
    pub fn add_interaction(&mut self, interaction: Interaction) {
        self.interactions.push(interaction);
    }

    /// Set metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ── Violation ──────────────────────────────────────────────────

/// A single contract violation found during verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractViolation {
    pub interaction: String,
    pub kind: ViolationKind,
    pub message: String,
}

/// Kind of contract violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
    /// Missing expected field.
    MissingField,
    /// Field type mismatch.
    TypeMismatch,
    /// Unexpected status code.
    StatusMismatch,
    /// Missing header.
    MissingHeader,
    /// Path not found.
    PathNotFound,
    /// Extra unexpected field.
    UnexpectedField,
}

impl fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField => write!(f, "missing_field"),
            Self::TypeMismatch => write!(f, "type_mismatch"),
            Self::StatusMismatch => write!(f, "status_mismatch"),
            Self::MissingHeader => write!(f, "missing_header"),
            Self::PathNotFound => write!(f, "path_not_found"),
            Self::UnexpectedField => write!(f, "unexpected_field"),
        }
    }
}

// ── Breaking Change Detection ──────────────────────────────────

/// A detected breaking change between contract versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BreakingChange {
    pub interaction: String,
    pub description: String,
    pub severity: ChangeSeverity,
}

/// Severity of a schema change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeSeverity {
    /// Backward-compatible (e.g., new optional field).
    Compatible,
    /// Potentially breaking (e.g., new required field with default).
    Warning,
    /// Definitely breaking (e.g., removed field, type change).
    Breaking,
}

/// Check schema compatibility between old and new versions.
pub fn check_schema_compatibility(old: &SchemaType, new: &SchemaType) -> Vec<BreakingChange> {
    let mut changes = Vec::new();
    check_compat_recursive(old, new, "", &mut changes);
    changes
}

fn check_compat_recursive(
    old: &SchemaType,
    new: &SchemaType,
    path: &str,
    changes: &mut Vec<BreakingChange>,
) {
    match (old, new) {
        (SchemaType::Object(old_fields), SchemaType::Object(new_fields)) => {
            // Check removed fields
            for of in old_fields {
                let field_path = if path.is_empty() {
                    of.name.clone()
                } else {
                    format!("{path}.{}", of.name)
                };
                let found = new_fields.iter().find(|nf| nf.name == of.name);
                match found {
                    None => {
                        changes.push(BreakingChange {
                            interaction: field_path.clone(),
                            description: format!("field '{}' was removed", of.name),
                            severity: ChangeSeverity::Breaking,
                        });
                    }
                    Some(nf) => {
                        // Check type change
                        if of.field_type != nf.field_type {
                            check_compat_recursive(&of.field_type, &nf.field_type, &field_path, changes);
                        }
                        // Check required change: optional -> required is breaking
                        if !of.required && nf.required {
                            changes.push(BreakingChange {
                                interaction: field_path,
                                description: format!("field '{}' changed from optional to required", of.name),
                                severity: ChangeSeverity::Breaking,
                            });
                        }
                    }
                }
            }
            // Check new required fields (breaking for consumers)
            for nf in new_fields {
                let exists = old_fields.iter().any(|of| of.name == nf.name);
                if !exists && nf.required {
                    let field_path = if path.is_empty() {
                        nf.name.clone()
                    } else {
                        format!("{path}.{}", nf.name)
                    };
                    changes.push(BreakingChange {
                        interaction: field_path,
                        description: format!("new required field '{}' added", nf.name),
                        severity: ChangeSeverity::Warning,
                    });
                }
            }
        }
        (SchemaType::Array(old_inner), SchemaType::Array(new_inner)) => {
            check_compat_recursive(old_inner, new_inner, &format!("{path}[]"), changes);
        }
        (SchemaType::Nullable(old_inner), SchemaType::Nullable(new_inner)) => {
            check_compat_recursive(old_inner, new_inner, path, changes);
        }
        (_, SchemaType::Any) => {
            // Widening to Any is compatible
        }
        (SchemaType::Any, _) => {
            // Narrowing from Any is a warning
            changes.push(BreakingChange {
                interaction: path.to_string(),
                description: format!("type narrowed from any to {new}"),
                severity: ChangeSeverity::Warning,
            });
        }
        _ if old != new => {
            changes.push(BreakingChange {
                interaction: path.to_string(),
                description: format!("type changed from {old} to {new}"),
                severity: ChangeSeverity::Breaking,
            });
        }
        _ => {}
    }
}

// ── Verification ───────────────────────────────────────────────

/// Validate a JSON value against a schema type.
pub fn validate_value(value: &serde_json::Value, schema: &SchemaType) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    validate_recursive(value, schema, "", &mut violations);
    violations
}

fn validate_recursive(
    value: &serde_json::Value,
    schema: &SchemaType,
    path: &str,
    violations: &mut Vec<ContractViolation>,
) {
    match schema {
        SchemaType::String => {
            if !value.is_string() {
                violations.push(ContractViolation {
                    interaction: path.to_string(),
                    kind: ViolationKind::TypeMismatch,
                    message: format!("expected string at '{path}', got {}", type_name(value)),
                });
            }
        }
        SchemaType::Number => {
            if !value.is_number() {
                violations.push(ContractViolation {
                    interaction: path.to_string(),
                    kind: ViolationKind::TypeMismatch,
                    message: format!("expected number at '{path}', got {}", type_name(value)),
                });
            }
        }
        SchemaType::Integer => {
            if !value.is_i64() && !value.is_u64() {
                violations.push(ContractViolation {
                    interaction: path.to_string(),
                    kind: ViolationKind::TypeMismatch,
                    message: format!("expected integer at '{path}', got {}", type_name(value)),
                });
            }
        }
        SchemaType::Boolean => {
            if !value.is_boolean() {
                violations.push(ContractViolation {
                    interaction: path.to_string(),
                    kind: ViolationKind::TypeMismatch,
                    message: format!("expected boolean at '{path}', got {}", type_name(value)),
                });
            }
        }
        SchemaType::Array(inner) => {
            if let Some(arr) = value.as_array() {
                for (i, item) in arr.iter().enumerate() {
                    let item_path = format!("{path}[{i}]");
                    validate_recursive(item, inner, &item_path, violations);
                }
            } else {
                violations.push(ContractViolation {
                    interaction: path.to_string(),
                    kind: ViolationKind::TypeMismatch,
                    message: format!("expected array at '{path}', got {}", type_name(value)),
                });
            }
        }
        SchemaType::Object(fields) => {
            if let Some(obj) = value.as_object() {
                for field in fields {
                    let field_path = if path.is_empty() {
                        field.name.clone()
                    } else {
                        format!("{path}.{}", field.name)
                    };
                    match obj.get(&field.name) {
                        Some(v) => validate_recursive(v, &field.field_type, &field_path, violations),
                        None if field.required => {
                            violations.push(ContractViolation {
                                interaction: field_path,
                                kind: ViolationKind::MissingField,
                                message: format!("missing required field '{}'", field.name),
                            });
                        }
                        None => {}
                    }
                }
            } else {
                violations.push(ContractViolation {
                    interaction: path.to_string(),
                    kind: ViolationKind::TypeMismatch,
                    message: format!("expected object at '{path}', got {}", type_name(value)),
                });
            }
        }
        SchemaType::Nullable(inner) => {
            if !value.is_null() {
                validate_recursive(value, inner, path, violations);
            }
        }
        SchemaType::Any => {}
    }
}

fn type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

// ── Contract Registry ──────────────────────────────────────────

/// Registry managing multiple contracts.
#[derive(Debug, Clone, Default)]
pub struct ContractRegistry {
    contracts: HashMap<String, Contract>,
}

impl ContractRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self { contracts: HashMap::new() }
    }

    /// Register a contract.
    pub fn register(&mut self, contract: Contract) -> Result<(), ContractError> {
        if self.contracts.contains_key(&contract.id) {
            return Err(ContractError::Duplicate(contract.id));
        }
        self.contracts.insert(contract.id.clone(), contract);
        Ok(())
    }

    /// Get a contract by ID.
    pub fn get(&self, id: &str) -> Option<&Contract> {
        self.contracts.get(id)
    }

    /// Remove a contract.
    pub fn remove(&mut self, id: &str) -> Option<Contract> {
        self.contracts.remove(id)
    }

    /// List all contract IDs (sorted).
    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.contracts.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Find contracts by consumer name.
    pub fn find_by_consumer(&self, consumer: &str) -> Vec<&Contract> {
        let mut results: Vec<&Contract> = self.contracts.values()
            .filter(|c| c.consumer == consumer)
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    /// Find contracts by provider name.
    pub fn find_by_provider(&self, provider: &str) -> Vec<&Contract> {
        let mut results: Vec<&Contract> = self.contracts.values()
            .filter(|c| c.provider == provider)
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    /// Count of registered contracts.
    pub fn len(&self) -> usize {
        self.contracts.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_schema() -> SchemaType {
        SchemaType::Object(vec![
            SchemaField {
                name: "id".to_string(),
                field_type: SchemaType::Integer,
                required: true,
                description: None,
            },
            SchemaField {
                name: "name".to_string(),
                field_type: SchemaType::String,
                required: true,
                description: None,
            },
            SchemaField {
                name: "email".to_string(),
                field_type: SchemaType::String,
                required: false,
                description: Some("optional email".to_string()),
            },
        ])
    }

    fn sample_contract() -> Contract {
        let mut c = Contract::new("c1", "web-app", "user-api", "1.0.0");
        c.add_interaction(Interaction {
            description: "get user by id".to_string(),
            provider_state: Some("user exists".to_string()),
            request: ContractRequest {
                method: ContractMethod::Get,
                path: "/users/1".to_string(),
                headers: HashMap::new(),
                body_schema: None,
                query_params: HashMap::new(),
            },
            response: ContractResponse {
                status: 200,
                headers: HashMap::new(),
                body_schema: Some(user_schema()),
            },
        });
        c
    }

    #[test]
    fn test_validate_valid_object() {
        let schema = user_schema();
        let value = json!({"id": 1, "name": "Alice"});
        let violations = validate_value(&value, &schema);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let schema = user_schema();
        let value = json!({"id": 1});
        let violations = validate_value(&value, &schema);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].kind, ViolationKind::MissingField);
    }

    #[test]
    fn test_validate_type_mismatch() {
        let schema = user_schema();
        let value = json!({"id": "not_a_number", "name": "Alice"});
        let violations = validate_value(&value, &schema);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].kind, ViolationKind::TypeMismatch);
    }

    #[test]
    fn test_validate_optional_missing_ok() {
        let schema = user_schema();
        let value = json!({"id": 1, "name": "Alice"});
        let violations = validate_value(&value, &schema);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_validate_optional_present() {
        let schema = user_schema();
        let value = json!({"id": 1, "name": "Alice", "email": "a@b.com"});
        let violations = validate_value(&value, &schema);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_validate_array() {
        let schema = SchemaType::Array(Box::new(SchemaType::String));
        let value = json!(["a", "b", "c"]);
        assert!(validate_value(&value, &schema).is_empty());
    }

    #[test]
    fn test_validate_array_bad_item() {
        let schema = SchemaType::Array(Box::new(SchemaType::String));
        let value = json!(["a", 42, "c"]);
        let violations = validate_value(&value, &schema);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_validate_nullable_null() {
        let schema = SchemaType::Nullable(Box::new(SchemaType::String));
        let value = json!(null);
        assert!(validate_value(&value, &schema).is_empty());
    }

    #[test]
    fn test_validate_nullable_value() {
        let schema = SchemaType::Nullable(Box::new(SchemaType::String));
        let value = json!("hello");
        assert!(validate_value(&value, &schema).is_empty());
    }

    #[test]
    fn test_validate_any() {
        let schema = SchemaType::Any;
        assert!(validate_value(&json!("anything"), &schema).is_empty());
        assert!(validate_value(&json!(42), &schema).is_empty());
        assert!(validate_value(&json!(null), &schema).is_empty());
    }

    #[test]
    fn test_validate_boolean() {
        let schema = SchemaType::Boolean;
        assert!(validate_value(&json!(true), &schema).is_empty());
        assert!(!validate_value(&json!("true"), &schema).is_empty());
    }

    #[test]
    fn test_validate_number() {
        let schema = SchemaType::Number;
        assert!(validate_value(&json!(3.14), &schema).is_empty());
        assert!(validate_value(&json!(42), &schema).is_empty());
        assert!(!validate_value(&json!("42"), &schema).is_empty());
    }

    #[test]
    fn test_compat_identical_schemas() {
        let schema = user_schema();
        let changes = check_schema_compatibility(&schema, &schema);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_compat_removed_field_breaking() {
        let old = user_schema();
        let new = SchemaType::Object(vec![
            SchemaField { name: "id".to_string(), field_type: SchemaType::Integer, required: true, description: None },
        ]);
        let changes = check_schema_compatibility(&old, &new);
        let breaking = changes.iter().filter(|c| c.severity == ChangeSeverity::Breaking).count();
        assert!(breaking >= 1); // "name" removed
    }

    #[test]
    fn test_compat_new_required_field_warning() {
        let old = user_schema();
        let mut new_fields = vec![
            SchemaField { name: "id".to_string(), field_type: SchemaType::Integer, required: true, description: None },
            SchemaField { name: "name".to_string(), field_type: SchemaType::String, required: true, description: None },
            SchemaField { name: "role".to_string(), field_type: SchemaType::String, required: true, description: None },
        ];
        // Keep email as optional
        new_fields.push(SchemaField { name: "email".to_string(), field_type: SchemaType::String, required: false, description: None });
        let new = SchemaType::Object(new_fields);
        let changes = check_schema_compatibility(&old, &new);
        let warnings = changes.iter().filter(|c| c.severity == ChangeSeverity::Warning).count();
        assert!(warnings >= 1); // "role" is new required
    }

    #[test]
    fn test_compat_type_change_breaking() {
        let old = SchemaType::Object(vec![
            SchemaField { name: "count".to_string(), field_type: SchemaType::Integer, required: true, description: None },
        ]);
        let new = SchemaType::Object(vec![
            SchemaField { name: "count".to_string(), field_type: SchemaType::String, required: true, description: None },
        ]);
        let changes = check_schema_compatibility(&old, &new);
        assert!(!changes.is_empty());
        assert!(changes.iter().any(|c| c.severity == ChangeSeverity::Breaking));
    }

    #[test]
    fn test_compat_optional_to_required_breaking() {
        let old = SchemaType::Object(vec![
            SchemaField { name: "x".to_string(), field_type: SchemaType::String, required: false, description: None },
        ]);
        let new = SchemaType::Object(vec![
            SchemaField { name: "x".to_string(), field_type: SchemaType::String, required: true, description: None },
        ]);
        let changes = check_schema_compatibility(&old, &new);
        assert!(changes.iter().any(|c| c.severity == ChangeSeverity::Breaking));
    }

    #[test]
    fn test_compat_widen_to_any() {
        let old = SchemaType::String;
        let new = SchemaType::Any;
        let changes = check_schema_compatibility(&old, &new);
        assert!(changes.is_empty()); // Widening is compatible
    }

    #[test]
    fn test_compat_narrow_from_any() {
        let old = SchemaType::Any;
        let new = SchemaType::String;
        let changes = check_schema_compatibility(&old, &new);
        assert!(changes.iter().any(|c| c.severity == ChangeSeverity::Warning));
    }

    #[test]
    fn test_compat_array_inner_change() {
        let old = SchemaType::Array(Box::new(SchemaType::String));
        let new = SchemaType::Array(Box::new(SchemaType::Integer));
        let changes = check_schema_compatibility(&old, &new);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_contract_creation() {
        let c = sample_contract();
        assert_eq!(c.consumer, "web-app");
        assert_eq!(c.provider, "user-api");
        assert_eq!(c.interactions.len(), 1);
    }

    #[test]
    fn test_contract_metadata() {
        let c = Contract::new("c1", "a", "b", "1.0")
            .with_metadata("tool", "joule-web");
        assert_eq!(c.metadata.get("tool").unwrap(), "joule-web");
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ContractRegistry::new();
        reg.register(sample_contract()).unwrap();
        assert!(reg.get("c1").is_some());
    }

    #[test]
    fn test_registry_duplicate() {
        let mut reg = ContractRegistry::new();
        reg.register(sample_contract()).unwrap();
        let err = reg.register(sample_contract()).unwrap_err();
        assert!(matches!(err, ContractError::Duplicate(_)));
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = ContractRegistry::new();
        reg.register(sample_contract()).unwrap();
        assert!(reg.remove("c1").is_some());
        assert!(reg.get("c1").is_none());
    }

    #[test]
    fn test_registry_find_by_consumer() {
        let mut reg = ContractRegistry::new();
        reg.register(sample_contract()).unwrap();
        let found = reg.find_by_consumer("web-app");
        assert_eq!(found.len(), 1);
        let empty = reg.find_by_consumer("other");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_registry_find_by_provider() {
        let mut reg = ContractRegistry::new();
        reg.register(sample_contract()).unwrap();
        let found = reg.find_by_provider("user-api");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_registry_ids_sorted() {
        let mut reg = ContractRegistry::new();
        reg.register(Contract::new("z", "a", "b", "1")).unwrap();
        reg.register(Contract::new("a", "x", "y", "1")).unwrap();
        assert_eq!(reg.ids(), vec!["a", "z"]);
    }

    #[test]
    fn test_registry_empty() {
        let reg = ContractRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_schema_type_display() {
        assert_eq!(format!("{}", SchemaType::String), "string");
        assert_eq!(format!("{}", SchemaType::Integer), "integer");
        assert_eq!(format!("{}", SchemaType::Array(Box::new(SchemaType::String))), "array<string>");
        assert_eq!(format!("{}", SchemaType::Nullable(Box::new(SchemaType::Boolean))), "nullable<boolean>");
    }

    #[test]
    fn test_violation_kind_display() {
        assert_eq!(format!("{}", ViolationKind::MissingField), "missing_field");
        assert_eq!(format!("{}", ViolationKind::TypeMismatch), "type_mismatch");
    }

    #[test]
    fn test_error_display() {
        let err = ContractError::NotFound("x".to_string());
        assert!(format!("{err}").contains("not found"));
        let err = ContractError::TypeMismatch {
            field: "f".to_string(),
            expected: "string".to_string(),
            actual: "number".to_string(),
        };
        assert!(format!("{err}").contains("type mismatch"));
    }

    #[test]
    fn test_validate_not_object_for_object_schema() {
        let schema = user_schema();
        let value = json!("not an object");
        let violations = validate_value(&value, &schema);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].kind, ViolationKind::TypeMismatch);
    }

    #[test]
    fn test_validate_not_array_for_array_schema() {
        let schema = SchemaType::Array(Box::new(SchemaType::String));
        let value = json!(42);
        let violations = validate_value(&value, &schema);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_contract_method_display() {
        assert_eq!(format!("{}", ContractMethod::Get), "GET");
        assert_eq!(format!("{}", ContractMethod::Post), "POST");
        assert_eq!(format!("{}", ContractMethod::Delete), "DELETE");
    }
}
