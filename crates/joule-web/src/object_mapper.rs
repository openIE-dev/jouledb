//! Object mapper — JSON-based field mapping between types.
//!
//! Maps fields between `serde_json::Value` objects using configurable
//! profiles. Supports auto-mapping by name, custom converters, nested
//! object mapping, collection mapping, and validation.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────────

/// Mapping error.
#[derive(Debug, Clone, PartialEq)]
pub enum MapError {
    /// A required source field is missing.
    MissingField(String),
    /// A converter failed.
    ConversionFailed { field: String, reason: String },
    /// Validation failed.
    ValidationFailed(Vec<String>),
    /// Source is not a JSON object.
    NotAnObject,
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::MissingField(name) => write!(f, "missing field: {name}"),
            MapError::ConversionFailed { field, reason } => {
                write!(f, "conversion failed for '{field}': {reason}")
            }
            MapError::ValidationFailed(reasons) => {
                write!(f, "validation failed: {}", reasons.join(", "))
            }
            MapError::NotAnObject => write!(f, "source is not a JSON object"),
        }
    }
}

// ── Field mapping rule ─────────────────────────────────────────────

/// How a single field is mapped from source to destination.
#[derive(Clone)]
pub struct FieldRule {
    /// Source field path (dot-separated for nested access, e.g. "address.city").
    pub source: String,
    /// Destination field name.
    pub dest: String,
    /// Whether the field is required.
    pub required: bool,
    /// Optional default value if source is missing and not required.
    pub default_value: Option<Value>,
    /// Converter index in the profile's converter list.
    pub converter_idx: Option<usize>,
}

impl FieldRule {
    pub fn new(source: impl Into<String>, dest: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            dest: dest.into(),
            required: false,
            default_value: None,
            converter_idx: None,
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }

    pub fn with_converter(mut self, idx: usize) -> Self {
        self.converter_idx = Some(idx);
        self
    }
}

impl fmt::Debug for FieldRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FieldRule")
            .field("source", &self.source)
            .field("dest", &self.dest)
            .field("required", &self.required)
            .finish()
    }
}

// ── Converter ──────────────────────────────────────────────────────

/// A named value converter.
pub struct Converter {
    pub name: String,
    pub func: Box<dyn Fn(&Value) -> Result<Value, String>>,
}

impl Converter {
    pub fn new(
        name: impl Into<String>,
        func: impl Fn(&Value) -> Result<Value, String> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            func: Box::new(func),
        }
    }
}

impl fmt::Debug for Converter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Converter")
            .field("name", &self.name)
            .finish()
    }
}

// ── Mapping Profile ────────────────────────────────────────────────

/// A named mapping profile containing rules and converters.
pub struct MappingProfile {
    pub name: String,
    rules: Vec<FieldRule>,
    converters: Vec<Converter>,
}

impl MappingProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rules: Vec::new(),
            converters: Vec::new(),
        }
    }

    /// Add a field mapping rule.
    pub fn add_rule(&mut self, rule: FieldRule) {
        self.rules.push(rule);
    }

    /// Add a converter and return its index.
    pub fn add_converter(&mut self, converter: Converter) -> usize {
        let idx = self.converters.len();
        self.converters.push(converter);
        idx
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Number of converters.
    pub fn converter_count(&self) -> usize {
        self.converters.len()
    }

    /// Auto-map: for every field in the source that also exists as a rule
    /// source, the mapping is already defined. This method adds identity
    /// rules for all top-level fields in a source template that don't yet
    /// have rules.
    pub fn auto_map_from(&mut self, source: &Value) {
        if let Value::Object(map) = source {
            for key in map.keys() {
                let already = self.rules.iter().any(|r| r.source == *key);
                if !already {
                    self.rules.push(FieldRule::new(key.clone(), key.clone()));
                }
            }
        }
    }

    /// Map a source JSON object to a destination JSON object.
    pub fn map(&self, source: &Value) -> Result<Value, MapError> {
        let src_obj = source.as_object().ok_or(MapError::NotAnObject)?;
        let mut dest = serde_json::Map::new();

        for rule in &self.rules {
            // Resolve source value (support dot-separated paths).
            let src_val = resolve_path(source, &rule.source);

            match src_val {
                Some(val) => {
                    let final_val = if let Some(idx) = rule.converter_idx {
                        if let Some(conv) = self.converters.get(idx) {
                            (conv.func)(val).map_err(|reason| MapError::ConversionFailed {
                                field: rule.source.clone(),
                                reason,
                            })?
                        } else {
                            val.clone()
                        }
                    } else {
                        val.clone()
                    };
                    set_path(&mut dest, &rule.dest, final_val);
                }
                None => {
                    if let Some(default) = &rule.default_value {
                        set_path(&mut dest, &rule.dest, default.clone());
                    } else if rule.required {
                        return Err(MapError::MissingField(rule.source.clone()));
                    }
                    // Optional field with no default — just skip.
                }
            }
        }

        // Include any source fields not covered by rules (pass-through is opt-in
        // via auto_map_from, so we only include fields that have rules).
        let _ = src_obj; // used in the NotAnObject check above

        Ok(Value::Object(dest))
    }

    /// Map a collection (JSON array) of source objects.
    pub fn map_collection(&self, source: &Value) -> Result<Value, MapError> {
        match source {
            Value::Array(arr) => {
                let mapped: Result<Vec<Value>, MapError> =
                    arr.iter().map(|item| self.map(item)).collect();
                Ok(Value::Array(mapped?))
            }
            _ => Err(MapError::NotAnObject),
        }
    }

    /// Validate that a source can be mapped without errors.
    pub fn validate(&self, source: &Value) -> Vec<String> {
        let mut issues = Vec::new();

        if !source.is_object() {
            issues.push("source is not a JSON object".to_string());
            return issues;
        }

        for rule in &self.rules {
            let val = resolve_path(source, &rule.source);
            if val.is_none() && rule.required && rule.default_value.is_none() {
                issues.push(format!("missing required field: {}", rule.source));
            }
        }

        issues
    }
}

impl fmt::Debug for MappingProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappingProfile")
            .field("name", &self.name)
            .field("rules", &self.rules.len())
            .field("converters", &self.converters.len())
            .finish()
    }
}

// ── Helpers: dot-path resolution ───────────────────────────────────

/// Resolve a dot-separated path in a JSON value.
fn resolve_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        match current {
            Value::Object(map) => {
                current = map.get(segment)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Set a value at a dot-separated path in a JSON map, creating intermediate
/// objects as needed.
fn set_path(map: &mut serde_json::Map<String, Value>, path: &str, value: Value) {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.len() == 1 {
        map.insert(segments[0].to_string(), value);
        return;
    }

    // Walk/create intermediate objects.
    let mut current = map;
    for seg in &segments[..segments.len() - 1] {
        let key = seg.to_string();
        if !current.contains_key(&key) {
            current.insert(key.clone(), Value::Object(serde_json::Map::new()));
        }
        current = current
            .get_mut(&key)
            .and_then(|v| v.as_object_mut())
            .expect("path conflict: intermediate is not an object");
    }
    current.insert(segments[segments.len() - 1].to_string(), value);
}

// ── Profile Registry ───────────────────────────────────────────────

/// Registry of named mapping profiles.
pub struct ProfileRegistry {
    profiles: HashMap<String, MappingProfile>,
}

impl ProfileRegistry {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Register a profile.
    pub fn register(&mut self, profile: MappingProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Get a profile by name.
    pub fn get(&self, name: &str) -> Option<&MappingProfile> {
        self.profiles.get(name)
    }

    /// Get a mutable reference.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut MappingProfile> {
        self.profiles.get_mut(name)
    }

    /// Remove a profile.
    pub fn remove(&mut self, name: &str) -> bool {
        self.profiles.remove(name).is_some()
    }

    /// Number of profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// All profile names (sorted for determinism).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.profiles.keys().cloned().collect();
        names.sort();
        names
    }

    /// Map a source using a named profile.
    pub fn map(&self, profile_name: &str, source: &Value) -> Result<Value, MapError> {
        self.profiles
            .get(profile_name)
            .ok_or_else(|| MapError::MissingField(format!("profile not found: {profile_name}")))?
            .map(source)
    }
}

impl Default for ProfileRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_source() -> Value {
        json!({
            "first_name": "Alice",
            "last_name": "Smith",
            "age": 30,
            "address": {
                "city": "Sarasota",
                "state": "FL"
            }
        })
    }

    #[test]
    fn simple_field_mapping() {
        let mut profile = MappingProfile::new("user_to_dto");
        profile.add_rule(FieldRule::new("first_name", "name"));
        profile.add_rule(FieldRule::new("age", "age"));

        let result = profile.map(&user_source()).unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn nested_source_path() {
        let mut profile = MappingProfile::new("addr");
        profile.add_rule(FieldRule::new("address.city", "city"));
        profile.add_rule(FieldRule::new("address.state", "state"));

        let result = profile.map(&user_source()).unwrap();
        assert_eq!(result["city"], json!("Sarasota"));
        assert_eq!(result["state"], json!("FL"));
    }

    #[test]
    fn nested_dest_path() {
        let mut profile = MappingProfile::new("nested_dest");
        profile.add_rule(FieldRule::new("first_name", "user.name"));
        profile.add_rule(FieldRule::new("age", "user.age"));

        let result = profile.map(&user_source()).unwrap();
        assert_eq!(result["user"]["name"], json!("Alice"));
        assert_eq!(result["user"]["age"], json!(30));
    }

    #[test]
    fn required_field_missing() {
        let mut profile = MappingProfile::new("strict");
        profile.add_rule(FieldRule::new("email", "email").required());

        let err = profile.map(&user_source()).unwrap_err();
        assert_eq!(err, MapError::MissingField("email".to_string()));
    }

    #[test]
    fn optional_field_missing() {
        let mut profile = MappingProfile::new("lenient");
        profile.add_rule(FieldRule::new("email", "email")); // not required
        profile.add_rule(FieldRule::new("first_name", "name"));

        let result = profile.map(&user_source()).unwrap();
        assert!(result.get("email").is_none());
        assert_eq!(result["name"], json!("Alice"));
    }

    #[test]
    fn default_value() {
        let mut profile = MappingProfile::new("defaults");
        profile.add_rule(
            FieldRule::new("email", "email").with_default(json!("unknown@example.com")),
        );

        let result = profile.map(&user_source()).unwrap();
        assert_eq!(result["email"], json!("unknown@example.com"));
    }

    #[test]
    fn custom_converter() {
        let mut profile = MappingProfile::new("convert");
        let idx = profile.add_converter(Converter::new("uppercase", |v| {
            v.as_str()
                .map(|s| Value::String(s.to_uppercase()))
                .ok_or_else(|| "not a string".to_string())
        }));
        profile.add_rule(FieldRule::new("first_name", "name").with_converter(idx));

        let result = profile.map(&user_source()).unwrap();
        assert_eq!(result["name"], json!("ALICE"));
    }

    #[test]
    fn converter_failure() {
        let mut profile = MappingProfile::new("fail_conv");
        let idx = profile.add_converter(Converter::new("fail", |_| {
            Err("always fails".to_string())
        }));
        profile.add_rule(FieldRule::new("age", "age").with_converter(idx));

        let err = profile.map(&user_source()).unwrap_err();
        match err {
            MapError::ConversionFailed { field, reason } => {
                assert_eq!(field, "age");
                assert_eq!(reason, "always fails");
            }
            _ => panic!("expected ConversionFailed"),
        }
    }

    #[test]
    fn auto_map() {
        let mut profile = MappingProfile::new("auto");
        profile.auto_map_from(&user_source());

        let result = profile.map(&user_source()).unwrap();
        assert_eq!(result["first_name"], json!("Alice"));
        assert_eq!(result["age"], json!(30));
    }

    #[test]
    fn auto_map_no_duplicates() {
        let mut profile = MappingProfile::new("auto");
        profile.add_rule(FieldRule::new("first_name", "custom_name"));
        profile.auto_map_from(&user_source());

        // "first_name" already has a rule — auto_map should not add another.
        let count = profile
            .rules
            .iter()
            .filter(|r| r.source == "first_name")
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn collection_mapping() {
        let mut profile = MappingProfile::new("items");
        profile.add_rule(FieldRule::new("name", "label"));

        let source = json!([
            {"name": "A"},
            {"name": "B"},
            {"name": "C"}
        ]);

        let result = profile.map_collection(&source).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["label"], json!("A"));
        assert_eq!(arr[2]["label"], json!("C"));
    }

    #[test]
    fn collection_mapping_error() {
        let profile = MappingProfile::new("items");
        let err = profile.map_collection(&json!("not an array")).unwrap_err();
        assert_eq!(err, MapError::NotAnObject);
    }

    #[test]
    fn validate_passes() {
        let mut profile = MappingProfile::new("v");
        profile.add_rule(FieldRule::new("first_name", "name").required());
        let issues = profile.validate(&user_source());
        assert!(issues.is_empty());
    }

    #[test]
    fn validate_missing_required() {
        let mut profile = MappingProfile::new("v");
        profile.add_rule(FieldRule::new("email", "email").required());
        let issues = profile.validate(&user_source());
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("email"));
    }

    #[test]
    fn validate_not_an_object() {
        let profile = MappingProfile::new("v");
        let issues = profile.validate(&json!("string"));
        assert!(!issues.is_empty());
    }

    #[test]
    fn source_not_an_object() {
        let profile = MappingProfile::new("test");
        let err = profile.map(&json!(42)).unwrap_err();
        assert_eq!(err, MapError::NotAnObject);
    }

    #[test]
    fn profile_counts() {
        let mut profile = MappingProfile::new("p");
        profile.add_rule(FieldRule::new("a", "b"));
        profile.add_converter(Converter::new("c", |v| Ok(v.clone())));
        assert_eq!(profile.rule_count(), 1);
        assert_eq!(profile.converter_count(), 1);
    }

    #[test]
    fn registry_register_and_get() {
        let mut reg = ProfileRegistry::new();
        let profile = MappingProfile::new("user");
        reg.register(profile);
        assert_eq!(reg.len(), 1);
        assert!(reg.get("user").is_some());
    }

    #[test]
    fn registry_remove() {
        let mut reg = ProfileRegistry::new();
        reg.register(MappingProfile::new("x"));
        assert!(reg.remove("x"));
        assert!(reg.is_empty());
        assert!(!reg.remove("x"));
    }

    #[test]
    fn registry_names_sorted() {
        let mut reg = ProfileRegistry::new();
        reg.register(MappingProfile::new("beta"));
        reg.register(MappingProfile::new("alpha"));
        assert_eq!(reg.names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn registry_map() {
        let mut reg = ProfileRegistry::new();
        let mut p = MappingProfile::new("simple");
        p.add_rule(FieldRule::new("first_name", "name"));
        reg.register(p);

        let result = reg.map("simple", &user_source()).unwrap();
        assert_eq!(result["name"], json!("Alice"));
    }

    #[test]
    fn registry_map_unknown_profile() {
        let reg = ProfileRegistry::new();
        let err = reg.map("nope", &user_source()).unwrap_err();
        match err {
            MapError::MissingField(msg) => assert!(msg.contains("nope")),
            _ => panic!("expected MissingField"),
        }
    }

    #[test]
    fn deeply_nested_source() {
        let source = json!({
            "a": {
                "b": {
                    "c": 42
                }
            }
        });
        let mut profile = MappingProfile::new("deep");
        profile.add_rule(FieldRule::new("a.b.c", "value"));
        let result = profile.map(&source).unwrap();
        assert_eq!(result["value"], json!(42));
    }

    #[test]
    fn deeply_nested_dest() {
        let source = json!({"val": 99});
        let mut profile = MappingProfile::new("deep_dest");
        profile.add_rule(FieldRule::new("val", "x.y.z"));
        let result = profile.map(&source).unwrap();
        assert_eq!(result["x"]["y"]["z"], json!(99));
    }
}
