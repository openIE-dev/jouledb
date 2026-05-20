//! Test fixture factory — builder pattern for test data, sequences, traits/attributes,
//! associations, lazy vs eager evaluation, and factory registry.
//!
//! Replaces JS fixture libraries (factory-bot, fishery, faker.js patterns) with a
//! pure-Rust test data factory supporting builder chains, sequences, and overrides.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Fixture factory errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactoryError {
    /// Factory not found.
    NotFound(String),
    /// Duplicate factory name.
    Duplicate(String),
    /// Required attribute missing.
    MissingAttribute(String),
    /// Trait not found.
    TraitNotFound { factory: String, trait_name: String },
    /// Sequence overflow.
    SequenceOverflow(String),
    /// Invalid attribute type.
    InvalidType { attribute: String, expected: String },
    /// Association target factory not found.
    AssociationNotFound { source: String, target: String },
}

impl fmt::Display for FactoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "factory not found: {name}"),
            Self::Duplicate(name) => write!(f, "duplicate factory: {name}"),
            Self::MissingAttribute(attr) => write!(f, "missing required attribute: {attr}"),
            Self::TraitNotFound { factory, trait_name } => {
                write!(f, "trait '{trait_name}' not found in factory '{factory}'")
            }
            Self::SequenceOverflow(name) => write!(f, "sequence overflow: {name}"),
            Self::InvalidType { attribute, expected } => {
                write!(f, "invalid type for '{attribute}': expected {expected}")
            }
            Self::AssociationNotFound { source, target } => {
                write!(f, "association target '{target}' not found from '{source}'")
            }
        }
    }
}

impl std::error::Error for FactoryError {}

// ── Attribute Value ────────────────────────────────────────────

/// How an attribute value is resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    /// Static value.
    Static(Value),
    /// Sequence-based (template with `{n}` placeholder).
    Sequence { template: String },
    /// Derived from another attribute (attribute name).
    Derived { from: String, transform: DeriveTransform },
    /// Reference to another factory (association).
    Association { factory_name: String },
}

/// Transform applied to derived values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeriveTransform {
    /// No transform — use as-is.
    Identity,
    /// Convert to uppercase string.
    Uppercase,
    /// Convert to lowercase string.
    Lowercase,
    /// Prefix with a string.
    Prefix(String),
    /// Suffix with a string.
    Suffix(String),
}

impl DeriveTransform {
    /// Apply the transform to a JSON value.
    pub fn apply(&self, value: &Value) -> Value {
        match self {
            Self::Identity => value.clone(),
            Self::Uppercase => {
                Value::String(value.as_str().unwrap_or("").to_uppercase())
            }
            Self::Lowercase => {
                Value::String(value.as_str().unwrap_or("").to_lowercase())
            }
            Self::Prefix(p) => {
                let s = value.as_str().unwrap_or("");
                Value::String(format!("{p}{s}"))
            }
            Self::Suffix(s_suf) => {
                let s = value.as_str().unwrap_or("");
                Value::String(format!("{s}{s_suf}"))
            }
        }
    }
}

// ── Evaluation Mode ────────────────────────────────────────────

/// Whether attributes are resolved immediately or on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvalMode {
    /// Resolve all attributes immediately.
    Eager,
    /// Store attribute definitions, resolve on access.
    Lazy,
}

impl Default for EvalMode {
    fn default() -> Self {
        Self::Eager
    }
}

// ── Factory Trait (named attribute overrides) ──────────────────

/// A named set of attribute overrides (like factory_bot traits).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryTrait {
    pub name: String,
    pub overrides: HashMap<String, AttributeValue>,
}

// ── Factory Definition ─────────────────────────────────────────

/// A factory definition for building test fixtures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryDefinition {
    /// Factory name.
    pub name: String,
    /// Default attributes and their value strategies.
    pub attributes: HashMap<String, AttributeValue>,
    /// Named traits for easy overrides.
    pub traits: HashMap<String, FactoryTrait>,
    /// Attributes that are required (must be in final output).
    pub required: Vec<String>,
    /// Evaluation mode.
    pub eval_mode: EvalMode,
}

impl FactoryDefinition {
    /// Create a new factory definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: HashMap::new(),
            traits: HashMap::new(),
            required: Vec::new(),
            eval_mode: EvalMode::default(),
        }
    }

    /// Add a static attribute.
    pub fn attr_static(mut self, name: impl Into<String>, value: Value) -> Self {
        self.attributes.insert(name.into(), AttributeValue::Static(value));
        self
    }

    /// Add a sequence attribute.
    pub fn attr_sequence(mut self, name: impl Into<String>, template: impl Into<String>) -> Self {
        self.attributes.insert(
            name.into(),
            AttributeValue::Sequence { template: template.into() },
        );
        self
    }

    /// Add a derived attribute.
    pub fn attr_derived(
        mut self,
        name: impl Into<String>,
        from: impl Into<String>,
        transform: DeriveTransform,
    ) -> Self {
        self.attributes.insert(
            name.into(),
            AttributeValue::Derived {
                from: from.into(),
                transform,
            },
        );
        self
    }

    /// Add an association attribute.
    pub fn attr_association(mut self, name: impl Into<String>, factory_name: impl Into<String>) -> Self {
        self.attributes.insert(
            name.into(),
            AttributeValue::Association { factory_name: factory_name.into() },
        );
        self
    }

    /// Add a named trait.
    pub fn add_trait(mut self, trait_def: FactoryTrait) -> Self {
        self.traits.insert(trait_def.name.clone(), trait_def);
        self
    }

    /// Mark an attribute as required.
    pub fn require(mut self, name: impl Into<String>) -> Self {
        self.required.push(name.into());
        self
    }

    /// Set evaluation mode.
    pub fn with_eval_mode(mut self, mode: EvalMode) -> Self {
        self.eval_mode = mode;
        self
    }
}

// ── Built Fixture ──────────────────────────────────────────────

/// A built fixture (resolved attributes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fixture {
    /// Factory name that produced this fixture.
    pub factory: String,
    /// Resolved attribute values.
    pub attributes: HashMap<String, Value>,
}

impl Fixture {
    /// Get an attribute value.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.attributes.get(name)
    }

    /// Get a string attribute.
    pub fn get_str(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).and_then(|v| v.as_str())
    }

    /// Get an integer attribute.
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        self.attributes.get(name).and_then(|v| v.as_i64())
    }

    /// Get a boolean attribute.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.attributes.get(name).and_then(|v| v.as_bool())
    }

    /// Convert to JSON Value.
    pub fn to_json(&self) -> Value {
        Value::Object(
            self.attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }
}

// ── Factory Registry ───────────────────────────────────────────

/// Registry of factory definitions with sequence counters.
#[derive(Debug, Clone)]
pub struct FactoryRegistry {
    factories: HashMap<String, FactoryDefinition>,
    sequences: HashMap<String, u64>,
}

impl FactoryRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            sequences: HashMap::new(),
        }
    }

    /// Register a factory definition.
    pub fn register(&mut self, factory: FactoryDefinition) -> Result<(), FactoryError> {
        if self.factories.contains_key(&factory.name) {
            return Err(FactoryError::Duplicate(factory.name));
        }
        self.factories.insert(factory.name.clone(), factory);
        Ok(())
    }

    /// Get a factory definition.
    pub fn get(&self, name: &str) -> Option<&FactoryDefinition> {
        self.factories.get(name)
    }

    /// Remove a factory.
    pub fn remove(&mut self, name: &str) -> Option<FactoryDefinition> {
        self.factories.remove(name)
    }

    /// List all factory names (sorted).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered factories.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Get the next sequence number for a factory.
    fn next_sequence(&mut self, factory_name: &str) -> u64 {
        let counter = self.sequences.entry(factory_name.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Build a fixture from a factory with optional overrides and traits.
    pub fn build(
        &mut self,
        factory_name: &str,
        overrides: &HashMap<String, Value>,
        trait_names: &[&str],
    ) -> Result<Fixture, FactoryError> {
        let factory = self
            .factories
            .get(factory_name)
            .ok_or_else(|| FactoryError::NotFound(factory_name.to_string()))?
            .clone();

        let seq = self.next_sequence(factory_name);

        // Start with default attributes
        let mut resolved: HashMap<String, Value> = HashMap::new();

        // Resolve non-derived attributes first, then derived ones,
        // so that derived attributes can find their source values.
        let mut non_derived: Vec<(String, AttributeValue)> = Vec::new();
        let mut derived: Vec<(String, AttributeValue)> = Vec::new();
        for (name, attr_val) in &factory.attributes {
            if matches!(attr_val, AttributeValue::Derived { .. }) {
                derived.push((name.clone(), attr_val.clone()));
            } else {
                non_derived.push((name.clone(), attr_val.clone()));
            }
        }
        for (name, attr_val) in &non_derived {
            let value = self.resolve_attribute(attr_val, seq, &resolved, factory_name, overrides)?;
            resolved.insert(name.clone(), value);
        }
        for (name, attr_val) in &derived {
            let value = self.resolve_attribute(attr_val, seq, &resolved, factory_name, overrides)?;
            resolved.insert(name.clone(), value);
        }

        // Apply traits in order
        for trait_name in trait_names {
            let trait_def = factory.traits.get(*trait_name).ok_or_else(|| {
                FactoryError::TraitNotFound {
                    factory: factory_name.to_string(),
                    trait_name: trait_name.to_string(),
                }
            })?;
            for (name, attr_val) in &trait_def.overrides {
                let value = self.resolve_attribute(attr_val, seq, &resolved, factory_name, overrides)?;
                resolved.insert(name.clone(), value);
            }
        }

        // Apply explicit overrides last
        for (name, value) in overrides {
            resolved.insert(name.clone(), value.clone());
        }

        // Check required attributes
        for req in &factory.required {
            if !resolved.contains_key(req) {
                return Err(FactoryError::MissingAttribute(req.clone()));
            }
        }

        Ok(Fixture {
            factory: factory_name.to_string(),
            attributes: resolved,
        })
    }

    /// Build N fixtures from the same factory.
    pub fn build_list(
        &mut self,
        factory_name: &str,
        count: usize,
        overrides: &HashMap<String, Value>,
        trait_names: &[&str],
    ) -> Result<Vec<Fixture>, FactoryError> {
        let mut fixtures = Vec::with_capacity(count);
        for _ in 0..count {
            fixtures.push(self.build(factory_name, overrides, trait_names)?);
        }
        Ok(fixtures)
    }

    /// Resolve a single attribute value.
    fn resolve_attribute(
        &mut self,
        attr_val: &AttributeValue,
        seq: u64,
        resolved: &HashMap<String, Value>,
        factory_name: &str,
        overrides: &HashMap<String, Value>,
    ) -> Result<Value, FactoryError> {
        match attr_val {
            AttributeValue::Static(v) => Ok(v.clone()),
            AttributeValue::Sequence { template } => {
                Ok(Value::String(template.replace("{n}", &seq.to_string())))
            }
            AttributeValue::Derived { from, transform } => {
                // Check overrides first, then resolved
                let source = overrides
                    .get(from)
                    .or_else(|| resolved.get(from))
                    .ok_or_else(|| FactoryError::MissingAttribute(from.clone()))?;
                Ok(transform.apply(source))
            }
            AttributeValue::Association { factory_name: target } => {
                if !self.factories.contains_key(target) {
                    return Err(FactoryError::AssociationNotFound {
                        source: factory_name.to_string(),
                        target: target.to_string(),
                    });
                }
                let fixture = self.build(target, &HashMap::new(), &[])?;
                Ok(fixture.to_json())
            }
        }
    }

    /// Reset all sequence counters.
    pub fn reset_sequences(&mut self) {
        self.sequences.clear();
    }
}

impl Default for FactoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_factory() -> FactoryDefinition {
        FactoryDefinition::new("user")
            .attr_sequence("id", "user-{n}")
            .attr_sequence("email", "user{n}@example.com")
            .attr_static("name", json!("Test User"))
            .attr_static("active", json!(true))
            .require("id")
            .require("name")
    }

    #[test]
    fn test_build_basic() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let fixture = reg.build("user", &HashMap::new(), &[]).unwrap();
        assert_eq!(fixture.get_str("id").unwrap(), "user-1");
        assert_eq!(fixture.get_str("email").unwrap(), "user1@example.com");
        assert_eq!(fixture.get_str("name").unwrap(), "Test User");
        assert_eq!(fixture.get_bool("active").unwrap(), true);
    }

    #[test]
    fn test_build_sequence_increments() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let f1 = reg.build("user", &HashMap::new(), &[]).unwrap();
        let f2 = reg.build("user", &HashMap::new(), &[]).unwrap();
        assert_eq!(f1.get_str("id").unwrap(), "user-1");
        assert_eq!(f2.get_str("id").unwrap(), "user-2");
    }

    #[test]
    fn test_build_with_overrides() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let overrides = HashMap::from([("name".to_string(), json!("Alice"))]);
        let fixture = reg.build("user", &overrides, &[]).unwrap();
        assert_eq!(fixture.get_str("name").unwrap(), "Alice");
    }

    #[test]
    fn test_build_with_trait() {
        let mut reg = FactoryRegistry::new();
        let factory = user_factory().add_trait(FactoryTrait {
            name: "admin".to_string(),
            overrides: HashMap::from([
                ("role".to_string(), AttributeValue::Static(json!("admin"))),
                ("active".to_string(), AttributeValue::Static(json!(true))),
            ]),
        });
        reg.register(factory).unwrap();
        let fixture = reg.build("user", &HashMap::new(), &["admin"]).unwrap();
        assert_eq!(fixture.get_str("role").unwrap(), "admin");
    }

    #[test]
    fn test_build_trait_not_found() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let result = reg.build("user", &HashMap::new(), &["nonexistent"]);
        assert!(matches!(result, Err(FactoryError::TraitNotFound { .. })));
    }

    #[test]
    fn test_build_factory_not_found() {
        let mut reg = FactoryRegistry::new();
        let result = reg.build("unknown", &HashMap::new(), &[]);
        assert!(matches!(result, Err(FactoryError::NotFound(_))));
    }

    #[test]
    fn test_build_list() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let fixtures = reg.build_list("user", 3, &HashMap::new(), &[]).unwrap();
        assert_eq!(fixtures.len(), 3);
        assert_eq!(fixtures[0].get_str("id").unwrap(), "user-1");
        assert_eq!(fixtures[1].get_str("id").unwrap(), "user-2");
        assert_eq!(fixtures[2].get_str("id").unwrap(), "user-3");
    }

    #[test]
    fn test_derived_attribute() {
        let mut reg = FactoryRegistry::new();
        let factory = FactoryDefinition::new("item")
            .attr_static("name", json!("widget"))
            .attr_derived("code", "name", DeriveTransform::Uppercase);
        reg.register(factory).unwrap();
        let fixture = reg.build("item", &HashMap::new(), &[]).unwrap();
        assert_eq!(fixture.get_str("code").unwrap(), "WIDGET");
    }

    #[test]
    fn test_derived_lowercase() {
        let mut reg = FactoryRegistry::new();
        let factory = FactoryDefinition::new("item")
            .attr_static("name", json!("HELLO"))
            .attr_derived("slug", "name", DeriveTransform::Lowercase);
        reg.register(factory).unwrap();
        let fixture = reg.build("item", &HashMap::new(), &[]).unwrap();
        assert_eq!(fixture.get_str("slug").unwrap(), "hello");
    }

    #[test]
    fn test_derived_prefix() {
        let mut reg = FactoryRegistry::new();
        let factory = FactoryDefinition::new("item")
            .attr_static("name", json!("widget"))
            .attr_derived("display", "name", DeriveTransform::Prefix("Item: ".to_string()));
        reg.register(factory).unwrap();
        let fixture = reg.build("item", &HashMap::new(), &[]).unwrap();
        assert_eq!(fixture.get_str("display").unwrap(), "Item: widget");
    }

    #[test]
    fn test_derived_suffix() {
        let mut reg = FactoryRegistry::new();
        let factory = FactoryDefinition::new("item")
            .attr_static("name", json!("test"))
            .attr_derived("full", "name", DeriveTransform::Suffix("_v2".to_string()));
        reg.register(factory).unwrap();
        let fixture = reg.build("item", &HashMap::new(), &[]).unwrap();
        assert_eq!(fixture.get_str("full").unwrap(), "test_v2");
    }

    #[test]
    fn test_derived_identity() {
        let transform = DeriveTransform::Identity;
        let v = json!("hello");
        assert_eq!(transform.apply(&v), json!("hello"));
    }

    #[test]
    fn test_association() {
        let mut reg = FactoryRegistry::new();
        let team_factory = FactoryDefinition::new("team")
            .attr_static("name", json!("Engineering"));
        let member_factory = FactoryDefinition::new("member")
            .attr_static("name", json!("Alice"))
            .attr_association("team", "team");
        reg.register(team_factory).unwrap();
        reg.register(member_factory).unwrap();

        let fixture = reg.build("member", &HashMap::new(), &[]).unwrap();
        let team = fixture.get("team").unwrap();
        assert!(team.is_object());
        assert_eq!(team.get("name").unwrap(), "Engineering");
    }

    #[test]
    fn test_association_not_found() {
        let mut reg = FactoryRegistry::new();
        let factory = FactoryDefinition::new("item")
            .attr_association("ref", "nonexistent");
        reg.register(factory).unwrap();
        let result = reg.build("item", &HashMap::new(), &[]);
        assert!(matches!(result, Err(FactoryError::AssociationNotFound { .. })));
    }

    #[test]
    fn test_registry_names() {
        let mut reg = FactoryRegistry::new();
        reg.register(FactoryDefinition::new("zebra")).unwrap();
        reg.register(FactoryDefinition::new("alpha")).unwrap();
        assert_eq!(reg.names(), vec!["alpha", "zebra"]);
    }

    #[test]
    fn test_registry_duplicate() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let err = reg.register(user_factory()).unwrap_err();
        assert!(matches!(err, FactoryError::Duplicate(_)));
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        assert!(reg.remove("user").is_some());
        assert!(reg.get("user").is_none());
    }

    #[test]
    fn test_registry_empty() {
        let reg = FactoryRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_reset_sequences() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        reg.build("user", &HashMap::new(), &[]).unwrap();
        reg.build("user", &HashMap::new(), &[]).unwrap();
        reg.reset_sequences();
        let fixture = reg.build("user", &HashMap::new(), &[]).unwrap();
        assert_eq!(fixture.get_str("id").unwrap(), "user-1");
    }

    #[test]
    fn test_fixture_to_json() {
        let mut reg = FactoryRegistry::new();
        reg.register(user_factory()).unwrap();
        let fixture = reg.build("user", &HashMap::new(), &[]).unwrap();
        let json_val = fixture.to_json();
        assert!(json_val.is_object());
        assert!(json_val.get("id").is_some());
        assert!(json_val.get("name").is_some());
    }

    #[test]
    fn test_fixture_accessors() {
        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), json!("Alice"));
        attrs.insert("age".to_string(), json!(30));
        attrs.insert("active".to_string(), json!(true));
        let fixture = Fixture {
            factory: "user".to_string(),
            attributes: attrs,
        };
        assert_eq!(fixture.get_str("name").unwrap(), "Alice");
        assert_eq!(fixture.get_i64("age").unwrap(), 30);
        assert_eq!(fixture.get_bool("active").unwrap(), true);
        assert!(fixture.get("missing").is_none());
    }

    #[test]
    fn test_error_display() {
        let err = FactoryError::NotFound("x".to_string());
        assert!(format!("{err}").contains("not found"));
        let err = FactoryError::TraitNotFound {
            factory: "user".to_string(),
            trait_name: "admin".to_string(),
        };
        assert!(format!("{err}").contains("admin"));
    }

    #[test]
    fn test_eval_mode_default() {
        assert_eq!(EvalMode::default(), EvalMode::Eager);
    }

    #[test]
    fn test_overrides_win_over_traits() {
        let mut reg = FactoryRegistry::new();
        let factory = user_factory().add_trait(FactoryTrait {
            name: "admin".to_string(),
            overrides: HashMap::from([
                ("name".to_string(), AttributeValue::Static(json!("Admin User"))),
            ]),
        });
        reg.register(factory).unwrap();
        let overrides = HashMap::from([("name".to_string(), json!("Override"))]);
        let fixture = reg.build("user", &overrides, &["admin"]).unwrap();
        assert_eq!(fixture.get_str("name").unwrap(), "Override");
    }
}
