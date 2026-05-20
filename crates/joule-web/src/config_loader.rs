//! Layered configuration loading.
//!
//! Supports defaults, file (JSON), environment variables, and CLI overrides.
//! Provides typed access (string, int, bool, float), nested dot-separated keys,
//! hot-reload signaling via version tracking, secret masking, validation rules,
//! and environment profiles. Pure Rust — no filesystem access; layers are fed
//! as in-memory maps.

use std::collections::HashMap;
use std::fmt;

// ── Config value ──────────────────────────────────────────────────

/// A typed configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(s) => write!(f, "{s}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Null => write!(f, "null"),
        }
    }
}

impl ConfigValue {
    /// Try to interpret as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Str(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to interpret as a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(i) => Some(*i as f64),
            Self::Str(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to interpret as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::Str(s) => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            },
            Self::Int(i) => match i {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            },
            _ => None,
        }
    }

    /// Return true if this is Null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Mask the value for display (secrets).
    pub fn masked(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            _ => "****".to_string(),
        }
    }
}

// ── Config source ─────────────────────────────────────────────────

/// Priority-ordered source layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfigSource {
    /// Built-in defaults (lowest priority).
    Default = 0,
    /// Loaded from a configuration file.
    File = 1,
    /// From environment variables.
    Env = 2,
    /// CLI arguments (highest priority).
    Cli = 3,
}

impl fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::File => write!(f, "file"),
            Self::Env => write!(f, "env"),
            Self::Cli => write!(f, "cli"),
        }
    }
}

// ── Validation rule ───────────────────────────────────────────────

/// Validation constraint for a config key.
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// Key must be present and not Null.
    Required,
    /// Integer must be within range.
    IntRange { min: i64, max: i64 },
    /// Float must be within range.
    FloatRange { min: f64, max: f64 },
    /// String must match one of the given values.
    OneOf(Vec<String>),
    /// String must not be empty.
    NonEmpty,
    /// Custom validation via a description (always passes in evaluation;
    /// the caller checks the description-based rule externally).
    Custom(String),
}

/// Validation error for a single key.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub key_path: String,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key_path, self.message)
    }
}

// ── Config entry ──────────────────────────────────────────────────

/// An entry tracking value and source.
#[derive(Debug, Clone)]
struct ConfigEntry {
    value: ConfigValue,
    source: ConfigSource,
    secret: bool,
}

// ── Config profile ────────────────────────────────────────────────

/// A named configuration profile (e.g. "development", "staging", "production").
#[derive(Debug, Clone)]
pub struct ConfigProfile {
    pub name: String,
    pub values: HashMap<String, ConfigValue>,
}

impl ConfigProfile {
    /// Create a new profile with given values.
    pub fn new(name: &str, values: HashMap<String, ConfigValue>) -> Self {
        Self {
            name: name.to_string(),
            values,
        }
    }
}

// ── Config loader ─────────────────────────────────────────────────

/// Layered configuration loader.
#[derive(Debug, Clone)]
pub struct ConfigLoader {
    entries: HashMap<String, ConfigEntry>,
    secret_keys: Vec<String>,
    validations: HashMap<String, Vec<ValidationRule>>,
    version: u64,
    active_profile: Option<String>,
}

impl ConfigLoader {
    /// Create a new empty config.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            secret_keys: Vec::new(),
            validations: HashMap::new(),
            version: 0,
            active_profile: None,
        }
    }

    /// Set a value at the given dot-separated key from the given source.
    pub fn set(&mut self, key_path: &str, value: ConfigValue, source: ConfigSource) {
        let is_secret = self.secret_keys.iter().any(|k| k == key_path);
        let entry = self.entries.entry(key_path.to_string()).or_insert(ConfigEntry {
            value: ConfigValue::Null,
            source: ConfigSource::Default,
            secret: is_secret,
        });
        // Only overwrite if new source has higher or equal priority.
        if source >= entry.source {
            entry.value = value;
            entry.source = source;
            entry.secret = is_secret;
        }
    }

    /// Load a full map of values from a given source.
    pub fn load_map(&mut self, source: ConfigSource, values: &HashMap<String, ConfigValue>) {
        for (k, v) in values {
            self.set(k, v.clone(), source);
        }
    }

    /// Apply a configuration profile (values go in at File level).
    pub fn apply_profile(&mut self, profile: &ConfigProfile) {
        self.active_profile = Some(profile.name.clone());
        self.load_map(ConfigSource::File, &profile.values);
    }

    /// Mark a key path as containing a secret (will be masked in output).
    pub fn mark_secret(&mut self, key_path: &str) {
        self.secret_keys.push(key_path.to_string());
        if let Some(entry) = self.entries.get_mut(key_path) {
            entry.secret = true;
        }
    }

    /// Add a validation rule for a key.
    pub fn add_validation(&mut self, key_path: &str, rule: ValidationRule) {
        self.validations
            .entry(key_path.to_string())
            .or_default()
            .push(rule);
    }

    /// Get a value by dot-separated key path.
    pub fn get(&self, key_path: &str) -> Option<&ConfigValue> {
        self.entries.get(key_path).map(|e| &e.value)
    }

    /// Get a string value.
    pub fn get_str(&self, key_path: &str) -> Option<&str> {
        self.get(key_path).and_then(|v| v.as_str())
    }

    /// Get an integer value.
    pub fn get_int(&self, key_path: &str) -> Option<i64> {
        self.get(key_path).and_then(|v| v.as_int())
    }

    /// Get a float value.
    pub fn get_float(&self, key_path: &str) -> Option<f64> {
        self.get(key_path).and_then(|v| v.as_float())
    }

    /// Get a bool value.
    pub fn get_bool(&self, key_path: &str) -> Option<bool> {
        self.get(key_path).and_then(|v| v.as_bool())
    }

    /// Get the source of a value.
    pub fn get_source(&self, key_path: &str) -> Option<ConfigSource> {
        self.entries.get(key_path).map(|e| e.source)
    }

    /// Get a display-safe string (secrets are masked).
    pub fn get_display(&self, key_path: &str) -> String {
        match self.entries.get(key_path) {
            Some(entry) if entry.secret => entry.value.masked(),
            Some(entry) => entry.value.to_string(),
            None => "undefined".to_string(),
        }
    }

    /// Get the current version (incremented on reload).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Signal a hot-reload: increment version.
    pub fn signal_reload(&mut self) {
        self.version = self.version.saturating_add(1);
    }

    /// Get the active profile name, if any.
    pub fn active_profile(&self) -> Option<&str> {
        self.active_profile.as_deref()
    }

    /// Run all registered validations and return errors.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        let mut keys: Vec<&String> = self.validations.keys().collect();
        keys.sort();

        for key_path in keys {
            let rules = &self.validations[key_path];
            let value = self.get(key_path);

            for rule in rules {
                match rule {
                    ValidationRule::Required => {
                        if value.is_none() || value == Some(&ConfigValue::Null) {
                            errors.push(ValidationError {
                                key_path: key_path.clone(),
                                message: "required but missing or null".into(),
                            });
                        }
                    }
                    ValidationRule::IntRange { min, max } => {
                        if let Some(v) = value.and_then(|v| v.as_int()) {
                            if v < *min || v > *max {
                                errors.push(ValidationError {
                                    key_path: key_path.clone(),
                                    message: format!("value {v} not in range [{min}, {max}]"),
                                });
                            }
                        }
                    }
                    ValidationRule::FloatRange { min, max } => {
                        if let Some(v) = value.and_then(|v| v.as_float()) {
                            if v < *min || v > *max {
                                errors.push(ValidationError {
                                    key_path: key_path.clone(),
                                    message: format!("value {v} not in range [{min}, {max}]"),
                                });
                            }
                        }
                    }
                    ValidationRule::OneOf(allowed) => {
                        if let Some(v) = value.and_then(|v| v.as_str()) {
                            if !allowed.iter().any(|a| a == v) {
                                errors.push(ValidationError {
                                    key_path: key_path.clone(),
                                    message: format!("value '{v}' not in allowed set"),
                                });
                            }
                        }
                    }
                    ValidationRule::NonEmpty => {
                        if let Some(v) = value.and_then(|v| v.as_str()) {
                            if v.is_empty() {
                                errors.push(ValidationError {
                                    key_path: key_path.clone(),
                                    message: "must not be empty".into(),
                                });
                            }
                        }
                    }
                    ValidationRule::Custom(_) => {
                        // Custom rules are checked externally.
                    }
                }
            }
        }

        errors
    }

    /// Return all key paths (sorted).
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.entries.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Dump all entries as display-safe key=value pairs.
    pub fn dump(&self) -> Vec<(String, String, String)> {
        let mut out: Vec<(String, String, String)> = self
            .entries
            .iter()
            .map(|(k, e)| {
                let display = if e.secret {
                    e.value.masked()
                } else {
                    e.value.to_string()
                };
                (k.clone(), display, e.source.to_string())
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_value_display() {
        assert_eq!(ConfigValue::Str("hello".into()).to_string(), "hello");
        assert_eq!(ConfigValue::Int(42).to_string(), "42");
        assert_eq!(ConfigValue::Bool(true).to_string(), "true");
        assert_eq!(ConfigValue::Null.to_string(), "null");
    }

    #[test]
    fn config_value_as_str() {
        assert_eq!(ConfigValue::Str("abc".into()).as_str(), Some("abc"));
        assert_eq!(ConfigValue::Int(1).as_str(), None);
    }

    #[test]
    fn config_value_as_int() {
        assert_eq!(ConfigValue::Int(42).as_int(), Some(42));
        assert_eq!(ConfigValue::Str("99".into()).as_int(), Some(99));
        assert_eq!(ConfigValue::Str("bad".into()).as_int(), None);
        assert_eq!(ConfigValue::Bool(true).as_int(), None);
    }

    #[test]
    fn config_value_as_float() {
        assert_eq!(ConfigValue::Float(3.14).as_float(), Some(3.14));
        assert_eq!(ConfigValue::Int(7).as_float(), Some(7.0));
        assert_eq!(ConfigValue::Str("2.5".into()).as_float(), Some(2.5));
    }

    #[test]
    fn config_value_as_bool() {
        assert_eq!(ConfigValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ConfigValue::Str("yes".into()).as_bool(), Some(true));
        assert_eq!(ConfigValue::Str("off".into()).as_bool(), Some(false));
        assert_eq!(ConfigValue::Str("maybe".into()).as_bool(), None);
        assert_eq!(ConfigValue::Int(1).as_bool(), Some(true));
        assert_eq!(ConfigValue::Int(0).as_bool(), Some(false));
        assert_eq!(ConfigValue::Int(2).as_bool(), None);
    }

    #[test]
    fn config_value_is_null() {
        assert!(ConfigValue::Null.is_null());
        assert!(!ConfigValue::Int(0).is_null());
    }

    #[test]
    fn config_value_masked() {
        assert_eq!(ConfigValue::Str("secret".into()).masked(), "****");
        assert_eq!(ConfigValue::Null.masked(), "null");
    }

    #[test]
    fn source_ordering() {
        assert!(ConfigSource::Default < ConfigSource::File);
        assert!(ConfigSource::File < ConfigSource::Env);
        assert!(ConfigSource::Env < ConfigSource::Cli);
    }

    #[test]
    fn source_display() {
        assert_eq!(ConfigSource::Default.to_string(), "default");
        assert_eq!(ConfigSource::Cli.to_string(), "cli");
    }

    #[test]
    fn set_and_get() {
        let mut cfg = ConfigLoader::new();
        cfg.set("db.host", ConfigValue::Str("localhost".into()), ConfigSource::Default);
        assert_eq!(cfg.get_str("db.host"), Some("localhost"));
    }

    #[test]
    fn higher_source_wins() {
        let mut cfg = ConfigLoader::new();
        cfg.set("port", ConfigValue::Int(8080), ConfigSource::Default);
        cfg.set("port", ConfigValue::Int(9090), ConfigSource::Env);
        assert_eq!(cfg.get_int("port"), Some(9090));
        assert_eq!(cfg.get_source("port"), Some(ConfigSource::Env));
    }

    #[test]
    fn lower_source_does_not_override() {
        let mut cfg = ConfigLoader::new();
        cfg.set("port", ConfigValue::Int(9090), ConfigSource::Cli);
        cfg.set("port", ConfigValue::Int(8080), ConfigSource::Default);
        assert_eq!(cfg.get_int("port"), Some(9090));
    }

    #[test]
    fn load_map() {
        let mut cfg = ConfigLoader::new();
        let mut map = HashMap::new();
        map.insert("a".into(), ConfigValue::Str("val_a".into()));
        map.insert("b".into(), ConfigValue::Int(10));
        cfg.load_map(ConfigSource::File, &map);
        assert_eq!(cfg.get_str("a"), Some("val_a"));
        assert_eq!(cfg.get_int("b"), Some(10));
    }

    #[test]
    fn secret_masking() {
        let mut cfg = ConfigLoader::new();
        cfg.mark_secret("db.password");
        cfg.set(
            "db.password",
            ConfigValue::Str("s3cr3t".into()),
            ConfigSource::File,
        );
        assert_eq!(cfg.get_display("db.password"), "****");
        assert_eq!(cfg.get_str("db.password"), Some("s3cr3t"));
    }

    #[test]
    fn secret_on_existing_key() {
        let mut cfg = ConfigLoader::new();
        cfg.set(
            "api_key",
            ConfigValue::Str("abc123".into()),
            ConfigSource::Env,
        );
        cfg.mark_secret("api_key");
        assert_eq!(cfg.get_display("api_key"), "****");
    }

    #[test]
    fn get_display_undefined() {
        let cfg = ConfigLoader::new();
        assert_eq!(cfg.get_display("missing"), "undefined");
    }

    #[test]
    fn version_and_reload() {
        let mut cfg = ConfigLoader::new();
        assert_eq!(cfg.version(), 0);
        cfg.signal_reload();
        assert_eq!(cfg.version(), 1);
        cfg.signal_reload();
        assert_eq!(cfg.version(), 2);
    }

    #[test]
    fn apply_profile() {
        let mut cfg = ConfigLoader::new();
        cfg.set("db.host", ConfigValue::Str("prod-db".into()), ConfigSource::Default);

        let mut dev_vals = HashMap::new();
        dev_vals.insert("db.host".into(), ConfigValue::Str("localhost".into()));
        dev_vals.insert("debug".into(), ConfigValue::Bool(true));
        let profile = ConfigProfile::new("development", dev_vals);

        cfg.apply_profile(&profile);
        assert_eq!(cfg.active_profile(), Some("development"));
        assert_eq!(cfg.get_str("db.host"), Some("localhost"));
        assert_eq!(cfg.get_bool("debug"), Some(true));
    }

    #[test]
    fn validate_required_present() {
        let mut cfg = ConfigLoader::new();
        cfg.set("name", ConfigValue::Str("app".into()), ConfigSource::Default);
        cfg.add_validation("name", ValidationRule::Required);
        assert!(cfg.validate().is_empty());
    }

    #[test]
    fn validate_required_missing() {
        let mut cfg = ConfigLoader::new();
        cfg.add_validation("missing_key", ValidationRule::Required);
        let errs = cfg.validate();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("required"));
    }

    #[test]
    fn validate_required_null() {
        let mut cfg = ConfigLoader::new();
        cfg.set("x", ConfigValue::Null, ConfigSource::Default);
        cfg.add_validation("x", ValidationRule::Required);
        let errs = cfg.validate();
        assert_eq!(errs.len(), 1);
    }

    #[test]
    fn validate_int_range_ok() {
        let mut cfg = ConfigLoader::new();
        cfg.set("port", ConfigValue::Int(8080), ConfigSource::Default);
        cfg.add_validation("port", ValidationRule::IntRange { min: 1, max: 65535 });
        assert!(cfg.validate().is_empty());
    }

    #[test]
    fn validate_int_range_fail() {
        let mut cfg = ConfigLoader::new();
        cfg.set("port", ConfigValue::Int(0), ConfigSource::Default);
        cfg.add_validation("port", ValidationRule::IntRange { min: 1, max: 65535 });
        let errs = cfg.validate();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("not in range"));
    }

    #[test]
    fn validate_float_range() {
        let mut cfg = ConfigLoader::new();
        cfg.set("ratio", ConfigValue::Float(1.5), ConfigSource::Default);
        cfg.add_validation("ratio", ValidationRule::FloatRange { min: 0.0, max: 1.0 });
        let errs = cfg.validate();
        assert_eq!(errs.len(), 1);
    }

    #[test]
    fn validate_one_of_ok() {
        let mut cfg = ConfigLoader::new();
        cfg.set(
            "env",
            ConfigValue::Str("production".into()),
            ConfigSource::Default,
        );
        cfg.add_validation(
            "env",
            ValidationRule::OneOf(vec![
                "development".into(),
                "staging".into(),
                "production".into(),
            ]),
        );
        assert!(cfg.validate().is_empty());
    }

    #[test]
    fn validate_one_of_fail() {
        let mut cfg = ConfigLoader::new();
        cfg.set("env", ConfigValue::Str("test".into()), ConfigSource::Default);
        cfg.add_validation(
            "env",
            ValidationRule::OneOf(vec!["dev".into(), "prod".into()]),
        );
        let errs = cfg.validate();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("not in allowed set"));
    }

    #[test]
    fn validate_non_empty_ok() {
        let mut cfg = ConfigLoader::new();
        cfg.set("name", ConfigValue::Str("app".into()), ConfigSource::Default);
        cfg.add_validation("name", ValidationRule::NonEmpty);
        assert!(cfg.validate().is_empty());
    }

    #[test]
    fn validate_non_empty_fail() {
        let mut cfg = ConfigLoader::new();
        cfg.set("name", ConfigValue::Str(String::new()), ConfigSource::Default);
        cfg.add_validation("name", ValidationRule::NonEmpty);
        let errs = cfg.validate();
        assert_eq!(errs.len(), 1);
    }

    #[test]
    fn validate_custom_always_passes() {
        let mut cfg = ConfigLoader::new();
        cfg.add_validation("x", ValidationRule::Custom("must be prime".into()));
        assert!(cfg.validate().is_empty());
    }

    #[test]
    fn keys_sorted() {
        let mut cfg = ConfigLoader::new();
        cfg.set("z", ConfigValue::Int(1), ConfigSource::Default);
        cfg.set("a", ConfigValue::Int(2), ConfigSource::Default);
        cfg.set("m", ConfigValue::Int(3), ConfigSource::Default);
        assert_eq!(cfg.keys(), vec!["a", "m", "z"]);
    }

    #[test]
    fn dump_sorted_and_masked() {
        let mut cfg = ConfigLoader::new();
        cfg.mark_secret("password");
        cfg.set("host", ConfigValue::Str("localhost".into()), ConfigSource::Default);
        cfg.set(
            "password",
            ConfigValue::Str("s3cr3t".into()),
            ConfigSource::Env,
        );
        let dump = cfg.dump();
        assert_eq!(dump.len(), 2);
        assert_eq!(dump[0], ("host".into(), "localhost".into(), "default".into()));
        assert_eq!(dump[1], ("password".into(), "****".into(), "env".into()));
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError {
            key_path: "db.port".into(),
            message: "out of range".into(),
        };
        assert_eq!(e.to_string(), "db.port: out of range");
    }

    #[test]
    fn profile_does_not_override_cli() {
        let mut cfg = ConfigLoader::new();
        cfg.set("port", ConfigValue::Int(9999), ConfigSource::Cli);
        let mut vals = HashMap::new();
        vals.insert("port".into(), ConfigValue::Int(3000));
        cfg.apply_profile(&ConfigProfile::new("dev", vals));
        // CLI has higher priority than File (profile).
        assert_eq!(cfg.get_int("port"), Some(9999));
    }

    #[test]
    fn get_missing_key_returns_none() {
        let cfg = ConfigLoader::new();
        assert!(cfg.get("nope").is_none());
        assert!(cfg.get_str("nope").is_none());
        assert!(cfg.get_int("nope").is_none());
        assert!(cfg.get_float("nope").is_none());
        assert!(cfg.get_bool("nope").is_none());
        assert!(cfg.get_source("nope").is_none());
    }

    #[test]
    fn bool_string_variants() {
        let trues = ["true", "1", "yes", "on", "TRUE", "Yes", "ON"];
        let falses = ["false", "0", "no", "off", "FALSE", "No", "OFF"];
        for t in &trues {
            assert_eq!(
                ConfigValue::Str(t.to_string()).as_bool(),
                Some(true),
                "expected true for '{t}'"
            );
        }
        for f in &falses {
            assert_eq!(
                ConfigValue::Str(f.to_string()).as_bool(),
                Some(false),
                "expected false for '{f}'"
            );
        }
    }

    #[test]
    fn float_display() {
        let v = ConfigValue::Float(3.14);
        let s = v.to_string();
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn equal_source_overwrites() {
        let mut cfg = ConfigLoader::new();
        cfg.set("x", ConfigValue::Int(1), ConfigSource::File);
        cfg.set("x", ConfigValue::Int(2), ConfigSource::File);
        assert_eq!(cfg.get_int("x"), Some(2));
    }
}
