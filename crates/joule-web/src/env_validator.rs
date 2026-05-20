//! Environment validation — required env vars, type checking, default values,
//! secret detection, .env file parsing, validation report.
//!
//! Replaces JS env tools (dotenv, envalid, env-var, joi for env) with a
//! pure-Rust environment validator that tracks every check with energy awareness.

use std::collections::BTreeMap;

// ── Errors ──────────────────────────────────────────────────────

/// Environment validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    /// Required variable missing.
    Missing(String),
    /// Type validation failed.
    TypeMismatch { name: String, expected: String, actual: String },
    /// Value not in allowed set.
    InvalidChoice { name: String, value: String, allowed: Vec<String> },
    /// Value out of range.
    OutOfRange { name: String, value: String, min: Option<String>, max: Option<String> },
    /// Parse error in .env file.
    ParseError { line: usize, reason: String },
    /// Secret leaked in non-secret variable.
    PotentialSecret { name: String, reason: String },
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(name) => write!(f, "required variable missing: {name}"),
            Self::TypeMismatch { name, expected, actual } => {
                write!(f, "{name}: expected {expected}, got \"{actual}\"")
            }
            Self::InvalidChoice { name, value, allowed } => {
                write!(f, "{name}: \"{value}\" not in [{}]", allowed.join(", "))
            }
            Self::OutOfRange { name, value, min, max } => {
                write!(
                    f,
                    "{name}: \"{value}\" out of range [{}, {}]",
                    min.as_deref().unwrap_or(".."),
                    max.as_deref().unwrap_or("..")
                )
            }
            Self::ParseError { line, reason } => {
                write!(f, "parse error at line {line}: {reason}")
            }
            Self::PotentialSecret { name, reason } => {
                write!(f, "potential secret in {name}: {reason}")
            }
        }
    }
}

// ── Types ───────────────────────────────────────────────────────

/// Expected type for an environment variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvType {
    Str,
    Int,
    Float,
    Bool,
    Url,
    Port,
    Email,
    Choice(Vec<String>),
}

impl EnvType {
    fn name(&self) -> String {
        match self {
            Self::Str => "string".to_string(),
            Self::Int => "integer".to_string(),
            Self::Float => "float".to_string(),
            Self::Bool => "boolean".to_string(),
            Self::Url => "url".to_string(),
            Self::Port => "port (1-65535)".to_string(),
            Self::Email => "email".to_string(),
            Self::Choice(opts) => format!("one of [{}]", opts.join(", ")),
        }
    }

    /// Validate a string value against this type.
    fn validate(&self, value: &str) -> bool {
        match self {
            Self::Str => true,
            Self::Int => value.parse::<i64>().is_ok(),
            Self::Float => value.parse::<f64>().is_ok(),
            Self::Bool => matches!(
                value.to_lowercase().as_str(),
                "true" | "false" | "1" | "0" | "yes" | "no"
            ),
            Self::Url => {
                value.starts_with("http://")
                    || value.starts_with("https://")
                    || value.starts_with("ftp://")
            }
            Self::Port => {
                value.parse::<u16>().map(|p| p >= 1).unwrap_or(false)
            }
            Self::Email => value.contains('@') && value.contains('.'),
            Self::Choice(opts) => opts.iter().any(|o| o == value),
        }
    }
}

/// Definition of an expected environment variable.
#[derive(Debug, Clone)]
pub struct EnvVarDef {
    pub name: String,
    pub env_type: EnvType,
    pub required: bool,
    pub default: Option<String>,
    pub description: String,
    pub secret: bool,
}

impl EnvVarDef {
    pub fn new(name: impl Into<String>, env_type: EnvType) -> Self {
        Self {
            name: name.into(),
            env_type,
            required: true,
            default: None,
            description: String::new(),
            secret: false,
        }
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn with_default(mut self, val: impl Into<String>) -> Self {
        self.default = Some(val.into());
        self.required = false;
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn secret(mut self) -> Self {
        self.secret = true;
        self
    }
}

// ── .env Parser ─────────────────────────────────────────────────

/// Parse a .env file content into key-value pairs.
pub fn parse_dotenv(content: &str) -> Result<BTreeMap<String, String>, EnvError> {
    let mut vars = BTreeMap::new();

    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();

        // Skip empty lines and comments.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Handle export prefix.
        let line = line.strip_prefix("export ").unwrap_or(line);

        let eq_pos = line.find('=').ok_or_else(|| EnvError::ParseError {
            line: line_idx + 1,
            reason: "missing '=' separator".to_string(),
        })?;

        let key = line[..eq_pos].trim().to_string();
        if key.is_empty() {
            return Err(EnvError::ParseError {
                line: line_idx + 1,
                reason: "empty variable name".to_string(),
            });
        }

        let raw_value = line[eq_pos + 1..].trim();

        // Strip surrounding quotes.
        let value = if (raw_value.starts_with('"') && raw_value.ends_with('"'))
            || (raw_value.starts_with('\'') && raw_value.ends_with('\''))
        {
            raw_value[1..raw_value.len() - 1].to_string()
        } else {
            // Strip inline comments (only for unquoted values).
            raw_value
                .split_once(" #")
                .map(|(v, _)| v.trim())
                .unwrap_or(raw_value)
                .to_string()
        };

        vars.insert(key, value);
    }

    Ok(vars)
}

// ── Secret Detection ────────────────────────────────────────────

/// Common patterns that suggest a value is a secret.
const SECRET_PATTERNS: &[&str] = &[
    "password", "secret", "token", "api_key", "apikey",
    "private_key", "access_key", "auth",
];

/// Heuristic patterns in values that suggest secrets.
fn looks_like_secret_value(value: &str) -> bool {
    // Long random-looking strings.
    if value.len() >= 32
        && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return true;
    }
    // Base64-ish.
    if value.len() >= 20 && value.ends_with('=') {
        return true;
    }
    false
}

/// Check if a variable name looks like it holds a secret.
fn name_suggests_secret(name: &str) -> bool {
    let lower = name.to_lowercase();
    SECRET_PATTERNS.iter().any(|p| lower.contains(p))
}

// ── Validator ───────────────────────────────────────────────────

/// Environment validator that checks variables against definitions.
#[derive(Debug, Clone)]
pub struct EnvValidator {
    definitions: Vec<EnvVarDef>,
    detect_secrets: bool,
}

impl EnvValidator {
    pub fn new() -> Self {
        Self { definitions: Vec::new(), detect_secrets: true }
    }

    pub fn disable_secret_detection(mut self) -> Self {
        self.detect_secrets = false;
        self
    }

    pub fn add(&mut self, def: EnvVarDef) {
        self.definitions.push(def);
    }

    /// Validate environment variables against definitions.
    pub fn validate(
        &self,
        env: &BTreeMap<String, String>,
    ) -> ValidationReport {
        let mut errors = Vec::new();
        let mut resolved = BTreeMap::new();

        for def in &self.definitions {
            let value = env.get(&def.name).cloned().or_else(|| def.default.clone());

            match &value {
                None if def.required => {
                    errors.push(EnvError::Missing(def.name.clone()));
                    continue;
                }
                None => continue,
                Some(val) => {
                    // Type checking.
                    if !def.env_type.validate(val) {
                        errors.push(EnvError::TypeMismatch {
                            name: def.name.clone(),
                            expected: def.env_type.name(),
                            actual: val.clone(),
                        });
                    }

                    // Secret detection.
                    if self.detect_secrets && !def.secret && name_suggests_secret(&def.name) {
                        errors.push(EnvError::PotentialSecret {
                            name: def.name.clone(),
                            reason: "variable name suggests a secret but not marked as such"
                                .to_string(),
                        });
                    }

                    if self.detect_secrets
                        && !def.secret
                        && looks_like_secret_value(val)
                        && !name_suggests_secret(&def.name)
                    {
                        errors.push(EnvError::PotentialSecret {
                            name: def.name.clone(),
                            reason: "value looks like a secret (long random string)".to_string(),
                        });
                    }

                    resolved.insert(def.name.clone(), val.clone());
                }
            }
        }

        ValidationReport {
            valid: errors.is_empty(),
            errors,
            resolved,
            total_checked: self.definitions.len(),
        }
    }

    /// Get all definitions.
    pub fn definitions(&self) -> &[EnvVarDef] {
        &self.definitions
    }

    /// Generate a template .env file from definitions.
    pub fn generate_template(&self) -> String {
        let mut lines = Vec::new();
        lines.push("# Environment Configuration".to_string());
        lines.push(String::new());

        for def in &self.definitions {
            if !def.description.is_empty() {
                lines.push(format!("# {}", def.description));
            }
            let required_tag = if def.required { " (required)" } else { " (optional)" };
            let type_tag = def.env_type.name();
            lines.push(format!("# Type: {type_tag}{required_tag}"));
            let default_val = def.default.as_deref().unwrap_or("");
            if def.secret {
                lines.push(format!("# {}=<secret>", def.name));
            } else {
                lines.push(format!("{}={default_val}", def.name));
            }
            lines.push(String::new());
        }

        lines.join("\n")
    }
}

impl Default for EnvValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of environment validation.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub valid: bool,
    pub errors: Vec<EnvError>,
    pub resolved: BTreeMap<String, String>,
    pub total_checked: usize,
}

impl ValidationReport {
    /// Format the report as a readable string.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Environment Validation: {}\n",
            if self.valid { "PASSED" } else { "FAILED" }
        ));
        out.push_str(&format!(
            "Checked: {} variables, Resolved: {}\n",
            self.total_checked,
            self.resolved.len()
        ));
        if !self.errors.is_empty() {
            out.push_str(&format!("Errors ({}):\n", self.errors.len()));
            for err in &self.errors {
                out.push_str(&format!("  - {err}\n"));
            }
        }
        out
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn env_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn test_parse_dotenv_basic() {
        let content = "FOO=bar\nBAZ=qux";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.get("FOO").map(|s| s.as_str()), Some("bar"));
        assert_eq!(vars.get("BAZ").map(|s| s.as_str()), Some("qux"));
    }

    #[test]
    fn test_parse_dotenv_comments_and_blank() {
        let content = "# comment\n\nFOO=bar\n# another comment";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars.get("FOO").map(|s| s.as_str()), Some("bar"));
    }

    #[test]
    fn test_parse_dotenv_quoted() {
        let content = "A=\"hello world\"\nB='single quoted'";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.get("A").map(|s| s.as_str()), Some("hello world"));
        assert_eq!(vars.get("B").map(|s| s.as_str()), Some("single quoted"));
    }

    #[test]
    fn test_parse_dotenv_export() {
        let content = "export FOO=bar";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.get("FOO").map(|s| s.as_str()), Some("bar"));
    }

    #[test]
    fn test_parse_dotenv_inline_comment() {
        let content = "FOO=bar # this is a comment";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.get("FOO").map(|s| s.as_str()), Some("bar"));
    }

    #[test]
    fn test_parse_dotenv_missing_equals() {
        let content = "NO_EQUALS";
        let err = parse_dotenv(content).unwrap_err();
        assert!(matches!(err, EnvError::ParseError { line: 1, .. }));
    }

    #[test]
    fn test_parse_dotenv_empty_key() {
        let content = "=value";
        let err = parse_dotenv(content).unwrap_err();
        assert!(matches!(err, EnvError::ParseError { .. }));
    }

    #[test]
    fn test_env_type_str() {
        assert!(EnvType::Str.validate("anything"));
        assert!(EnvType::Str.validate(""));
    }

    #[test]
    fn test_env_type_int() {
        assert!(EnvType::Int.validate("42"));
        assert!(EnvType::Int.validate("-10"));
        assert!(!EnvType::Int.validate("abc"));
        assert!(!EnvType::Int.validate("3.14"));
    }

    #[test]
    fn test_env_type_float() {
        assert!(EnvType::Float.validate("3.14"));
        assert!(EnvType::Float.validate("-2.5"));
        assert!(EnvType::Float.validate("42"));
        assert!(!EnvType::Float.validate("abc"));
    }

    #[test]
    fn test_env_type_bool() {
        for val in &["true", "false", "1", "0", "yes", "no", "True", "FALSE"] {
            assert!(EnvType::Bool.validate(val), "should accept: {val}");
        }
        assert!(!EnvType::Bool.validate("maybe"));
    }

    #[test]
    fn test_env_type_url() {
        assert!(EnvType::Url.validate("https://example.com"));
        assert!(EnvType::Url.validate("http://localhost:8080"));
        assert!(!EnvType::Url.validate("not-a-url"));
    }

    #[test]
    fn test_env_type_port() {
        assert!(EnvType::Port.validate("8080"));
        assert!(EnvType::Port.validate("1"));
        assert!(EnvType::Port.validate("65535"));
        assert!(!EnvType::Port.validate("0"));
        assert!(!EnvType::Port.validate("99999"));
        assert!(!EnvType::Port.validate("abc"));
    }

    #[test]
    fn test_env_type_email() {
        assert!(EnvType::Email.validate("user@example.com"));
        assert!(!EnvType::Email.validate("not-email"));
        assert!(!EnvType::Email.validate("@no-dot"));
    }

    #[test]
    fn test_env_type_choice() {
        let c = EnvType::Choice(vec!["dev".into(), "staging".into(), "prod".into()]);
        assert!(c.validate("dev"));
        assert!(c.validate("prod"));
        assert!(!c.validate("test"));
    }

    #[test]
    fn test_validate_required_present() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port));
        let env = env_map(&[("PORT", "8080")]);
        let report = v.validate(&env);
        assert!(report.valid);
        assert_eq!(report.resolved.get("PORT").map(|s| s.as_str()), Some("8080"));
    }

    #[test]
    fn test_validate_required_missing() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port));
        let env = BTreeMap::new();
        let report = v.validate(&env);
        assert!(!report.valid);
        assert!(matches!(&report.errors[0], EnvError::Missing(n) if n == "PORT"));
    }

    #[test]
    fn test_validate_optional_missing() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("DEBUG", EnvType::Bool).optional());
        let env = BTreeMap::new();
        let report = v.validate(&env);
        assert!(report.valid);
    }

    #[test]
    fn test_validate_with_default() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port).with_default("3000"));
        let env = BTreeMap::new();
        let report = v.validate(&env);
        assert!(report.valid);
        assert_eq!(report.resolved.get("PORT").map(|s| s.as_str()), Some("3000"));
    }

    #[test]
    fn test_validate_type_mismatch() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port));
        let env = env_map(&[("PORT", "not-a-number")]);
        let report = v.validate(&env);
        assert!(!report.valid);
        assert!(matches!(&report.errors[0], EnvError::TypeMismatch { name, .. } if name == "PORT"));
    }

    #[test]
    fn test_validate_choice() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new(
            "ENV",
            EnvType::Choice(vec!["dev".into(), "prod".into()]),
        ));
        let env = env_map(&[("ENV", "staging")]);
        let report = v.validate(&env);
        assert!(!report.valid);
    }

    #[test]
    fn test_secret_detection_name() {
        let mut v = EnvValidator::new();
        v.add(EnvVarDef::new("API_KEY", EnvType::Str));
        let env = env_map(&[("API_KEY", "abc123")]);
        let report = v.validate(&env);
        assert!(!report.valid);
        assert!(matches!(
            &report.errors[0],
            EnvError::PotentialSecret { name, .. } if name == "API_KEY"
        ));
    }

    #[test]
    fn test_secret_detection_marked() {
        let mut v = EnvValidator::new();
        v.add(EnvVarDef::new("API_KEY", EnvType::Str).secret());
        let env = env_map(&[("API_KEY", "abc123")]);
        let report = v.validate(&env);
        assert!(report.valid);
    }

    #[test]
    fn test_secret_detection_value() {
        let mut v = EnvValidator::new();
        v.add(EnvVarDef::new("SOME_CONFIG", EnvType::Str));
        let long_random = "abcdefghijklmnopqrstuvwxyz0123456789AB";
        let env = env_map(&[("SOME_CONFIG", long_random)]);
        let report = v.validate(&env);
        assert!(!report.valid);
        assert!(matches!(
            &report.errors[0],
            EnvError::PotentialSecret { .. }
        ));
    }

    #[test]
    fn test_secret_detection_disabled() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("API_KEY", EnvType::Str));
        let env = env_map(&[("API_KEY", "abc123")]);
        let report = v.validate(&env);
        assert!(report.valid);
    }

    #[test]
    fn test_generate_template() {
        let mut v = EnvValidator::new();
        v.add(
            EnvVarDef::new("PORT", EnvType::Port)
                .with_default("3000")
                .with_description("Server port"),
        );
        v.add(
            EnvVarDef::new("DB_PASSWORD", EnvType::Str)
                .secret()
                .with_description("Database password"),
        );
        let template = v.generate_template();
        assert!(template.contains("PORT=3000"));
        assert!(template.contains("# DB_PASSWORD=<secret>"));
        assert!(template.contains("Server port"));
    }

    #[test]
    fn test_validation_report_format() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port));
        let env = BTreeMap::new();
        let report = v.validate(&env);
        let formatted = report.format_report();
        assert!(formatted.contains("FAILED"));
        assert!(formatted.contains("Errors (1)"));
    }

    #[test]
    fn test_validation_report_passed() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port));
        let env = env_map(&[("PORT", "8080")]);
        let report = v.validate(&env);
        let formatted = report.format_report();
        assert!(formatted.contains("PASSED"));
    }

    #[test]
    fn test_parse_dotenv_empty_value() {
        let content = "FOO=";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.get("FOO").map(|s| s.as_str()), Some(""));
    }

    #[test]
    fn test_parse_dotenv_spaces_around_equals() {
        let content = "FOO = bar";
        let vars = parse_dotenv(content).unwrap();
        assert_eq!(vars.get("FOO").map(|s| s.as_str()), Some("bar"));
    }

    #[test]
    fn test_multiple_validations() {
        let mut v = EnvValidator::new().disable_secret_detection();
        v.add(EnvVarDef::new("PORT", EnvType::Port));
        v.add(EnvVarDef::new("HOST", EnvType::Str));
        v.add(EnvVarDef::new("DEBUG", EnvType::Bool).with_default("false"));
        v.add(
            EnvVarDef::new("ENV", EnvType::Choice(vec!["dev".into(), "prod".into()]))
                .with_default("dev"),
        );

        let env = env_map(&[("PORT", "8080"), ("HOST", "localhost")]);
        let report = v.validate(&env);
        assert!(report.valid);
        assert_eq!(report.resolved.len(), 4);
        assert_eq!(report.resolved.get("DEBUG").map(|s| s.as_str()), Some("false"));
    }

    #[test]
    fn test_env_var_def_description() {
        let def = EnvVarDef::new("PORT", EnvType::Port)
            .with_description("The server port");
        assert_eq!(def.description, "The server port");
    }
}
