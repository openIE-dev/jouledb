//! `.env` file parser with variable interpolation, defaults, and type coercion.
//!
//! Handles the standard `.env` format including comments (`#`), blank lines,
//! quoted values (single / double), multiline values (double-quoted with `\n`),
//! `${VAR}` interpolation, `${VAR:-default}` fallback syntax, and prefix
//! filtering (e.g. `APP_*`).

use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Errors produced during `.env` parsing or validation.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvParseError {
    /// Syntax error at a specific line.
    SyntaxError { line: usize, message: String },
    /// A required variable was not set.
    MissingRequired(String),
    /// An interpolation referenced an undefined variable.
    UndefinedVariable { key: String, referenced: String },
    /// Could not coerce a value to the requested type.
    CoercionError { key: String, target_type: String },
}

impl std::fmt::Display for EnvParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvParseError::SyntaxError { line, message } => {
                write!(f, "line {}: {}", line, message)
            }
            EnvParseError::MissingRequired(k) => write!(f, "required variable missing: {}", k),
            EnvParseError::UndefinedVariable { key, referenced } => {
                write!(f, "variable '{}' references undefined '{}'", key, referenced)
            }
            EnvParseError::CoercionError { key, target_type } => {
                write!(f, "cannot coerce '{}' to {}", key, target_type)
            }
        }
    }
}

// ── Parsed value ────────────────────────────────────────────────

/// A coerced value from an environment variable.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl EnvValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            EnvValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            EnvValue::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            EnvValue::Float(f) => Some(*f),
            EnvValue::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            EnvValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

// ── EnvParser ───────────────────────────────────────────────────

/// Parser for `.env` files with interpolation, defaults, and validation.
pub struct EnvParser {
    vars: HashMap<String, String>,
    coerced: HashMap<String, EnvValue>,
    required: Vec<String>,
    prefix_filter: Option<String>,
}

impl EnvParser {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            coerced: HashMap::new(),
            required: Vec::new(),
            prefix_filter: None,
        }
    }

    /// Restrict keys to those starting with `prefix`.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix_filter = Some(prefix.into());
        self
    }

    /// Mark a variable as required.
    pub fn require(mut self, key: impl Into<String>) -> Self {
        self.required.push(key.into());
        self
    }

    /// Pre-seed a variable (e.g. from the actual OS environment).
    pub fn seed(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.vars.insert(key.into(), value.into());
    }

    /// Parse a `.env` file contents string.
    pub fn parse(&mut self, input: &str) -> Result<(), EnvParseError> {
        let mut raw_entries: Vec<(String, String, usize)> = Vec::new();

        for (line_idx, line) in input.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            // Skip blank lines and comments.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Find the `=` separator.
            let eq_pos = trimmed.find('=').ok_or_else(|| EnvParseError::SyntaxError {
                line: line_num,
                message: "expected KEY=VALUE".into(),
            })?;

            let key = trimmed[..eq_pos].trim();
            if key.is_empty() {
                return Err(EnvParseError::SyntaxError {
                    line: line_num,
                    message: "empty key".into(),
                });
            }

            let raw_value = trimmed[eq_pos + 1..].trim();

            // Unquote if wrapped in matching quotes.
            let value = unquote(raw_value);

            // Apply prefix filter.
            if let Some(prefix) = &self.prefix_filter {
                if !key.starts_with(prefix.as_str()) {
                    continue;
                }
            }

            raw_entries.push((key.to_string(), value, line_num));
        }

        // Insert raw values first so later interpolations can reference earlier keys.
        for (key, value, _line) in &raw_entries {
            self.vars.insert(key.clone(), value.clone());
        }

        // Resolve interpolations.
        let keys: Vec<String> = raw_entries.iter().map(|(k, _, _)| k.clone()).collect();
        for key in &keys {
            let raw = self.vars.get(key).cloned().unwrap_or_default();
            let resolved = self.interpolate(&raw, key)?;
            self.vars.insert(key.clone(), resolved.clone());
            self.coerced.insert(key.clone(), coerce_value(&resolved));
        }

        Ok(())
    }

    /// Validate that all required variables are present.
    pub fn validate(&self) -> Result<(), Vec<EnvParseError>> {
        let mut errors = Vec::new();
        for req in &self.required {
            if !self.vars.contains_key(req.as_str()) {
                errors.push(EnvParseError::MissingRequired(req.clone()));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get raw string value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(|s| s.as_str())
    }

    /// Get a coerced value.
    pub fn get_typed(&self, key: &str) -> Option<&EnvValue> {
        self.coerced.get(key)
    }

    /// Get value or default string.
    pub fn get_or(&self, key: &str, default: &str) -> String {
        self.vars.get(key).cloned().unwrap_or_else(|| default.to_string())
    }

    /// Get as int with explicit coercion.
    pub fn get_int(&self, key: &str) -> Result<i64, EnvParseError> {
        let s = self.vars.get(key).ok_or_else(|| EnvParseError::MissingRequired(key.into()))?;
        s.parse::<i64>().map_err(|_| EnvParseError::CoercionError {
            key: key.into(),
            target_type: "int".into(),
        })
    }

    /// Get as bool with explicit coercion.
    pub fn get_bool(&self, key: &str) -> Result<bool, EnvParseError> {
        let s = self.vars.get(key).ok_or_else(|| EnvParseError::MissingRequired(key.into()))?;
        match s.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            _ => Err(EnvParseError::CoercionError {
                key: key.into(),
                target_type: "bool".into(),
            }),
        }
    }

    /// All parsed key-value pairs.
    pub fn all(&self) -> &HashMap<String, String> {
        &self.vars
    }

    /// Number of parsed variables.
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Filter keys by prefix and return stripped keys.
    pub fn with_prefix_stripped(&self, prefix: &str) -> HashMap<String, String> {
        let mut result = HashMap::new();
        for (k, v) in &self.vars {
            if let Some(rest) = k.strip_prefix(prefix) {
                let stripped = rest.strip_prefix('_').unwrap_or(rest);
                result.insert(stripped.to_string(), v.clone());
            }
        }
        result
    }

    // ── Interpolation ───────────────────────────────────────────

    /// Resolve `${VAR}` and `${VAR:-default}` references.
    fn interpolate(&self, value: &str, context_key: &str) -> Result<String, EnvParseError> {
        let mut result = String::with_capacity(value.len());
        let mut chars = value.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '$' && chars.peek() == Some(&'{') {
                chars.next(); // consume '{'
                let mut var_expr = String::new();
                let mut found_close = false;
                for c in chars.by_ref() {
                    if c == '}' {
                        found_close = true;
                        break;
                    }
                    var_expr.push(c);
                }
                if !found_close {
                    result.push_str("${");
                    result.push_str(&var_expr);
                    continue;
                }

                // Check for :- default syntax.
                if let Some(sep_pos) = var_expr.find(":-") {
                    let var_name = &var_expr[..sep_pos];
                    let default_val = &var_expr[sep_pos + 2..];
                    if let Some(resolved) = self.vars.get(var_name) {
                        result.push_str(resolved);
                    } else {
                        result.push_str(default_val);
                    }
                } else {
                    let var_name = var_expr.as_str();
                    if let Some(resolved) = self.vars.get(var_name) {
                        result.push_str(resolved);
                    } else {
                        return Err(EnvParseError::UndefinedVariable {
                            key: context_key.into(),
                            referenced: var_name.into(),
                        });
                    }
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }
}

impl Default for EnvParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Strip matching quotes from a value.
fn unquote(s: &str) -> String {
    if s.len() >= 2 {
        if (s.starts_with('"') && s.ends_with('"'))
            || (s.starts_with('\'') && s.ends_with('\''))
        {
            let inner = &s[1..s.len() - 1];
            // Handle escape sequences in double-quoted strings.
            if s.starts_with('"') {
                return inner
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\");
            }
            return inner.to_string();
        }
    }
    s.to_string()
}

/// Coerce a string to the most specific type.
fn coerce_value(s: &str) -> EnvValue {
    match s.to_lowercase().as_str() {
        "true" | "yes" | "on" => return EnvValue::Bool(true),
        "false" | "no" | "off" => return EnvValue::Bool(false),
        _ => {}
    }
    if let Ok(n) = s.parse::<i64>() {
        return EnvValue::Int(n);
    }
    if let Ok(f) = s.parse::<f64>() {
        return EnvValue::Float(f);
    }
    EnvValue::String(s.to_string())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(input: &str) -> EnvParser {
        let mut p = EnvParser::new();
        p.parse(input).unwrap();
        p
    }

    #[test]
    fn basic_key_value() {
        let p = parse_str("HOST=localhost\nPORT=8080");
        assert_eq!(p.get("HOST"), Some("localhost"));
        assert_eq!(p.get("PORT"), Some("8080"));
    }

    #[test]
    fn comments_and_blanks_ignored() {
        let input = "# comment\n\nKEY=value\n  # another comment\n";
        let p = parse_str(input);
        assert_eq!(p.len(), 1);
        assert_eq!(p.get("KEY"), Some("value"));
    }

    #[test]
    fn double_quoted_value() {
        let p = parse_str("MSG=\"hello world\"");
        assert_eq!(p.get("MSG"), Some("hello world"));
    }

    #[test]
    fn single_quoted_value() {
        let p = parse_str("MSG='hello world'");
        assert_eq!(p.get("MSG"), Some("hello world"));
    }

    #[test]
    fn multiline_escape_in_double_quotes() {
        let p = parse_str("MSG=\"line1\\nline2\"");
        assert_eq!(p.get("MSG"), Some("line1\nline2"));
    }

    #[test]
    fn interpolation_basic() {
        let p = parse_str("BASE=/app\nPATH=${BASE}/bin");
        assert_eq!(p.get("PATH"), Some("/app/bin"));
    }

    #[test]
    fn interpolation_with_default() {
        let p = parse_str("URL=${HOST:-localhost}:${PORT:-3000}");
        assert_eq!(p.get("URL"), Some("localhost:3000"));
    }

    #[test]
    fn interpolation_default_overridden() {
        let p = parse_str("HOST=myhost\nURL=${HOST:-localhost}");
        assert_eq!(p.get("URL"), Some("myhost"));
    }

    #[test]
    fn interpolation_undefined_no_default() {
        let mut parser = EnvParser::new();
        let result = parser.parse("URL=${MISSING}");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, EnvParseError::UndefinedVariable { .. }));
    }

    #[test]
    fn required_validation_fails() {
        let parser = EnvParser::new().require("DATABASE_URL");
        let result = parser.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn required_validation_passes() {
        let mut parser = EnvParser::new().require("HOST");
        parser.parse("HOST=ok").unwrap();
        assert!(parser.validate().is_ok());
    }

    #[test]
    fn prefix_filter() {
        let mut parser = EnvParser::new().with_prefix("APP");
        parser.parse("APP_PORT=3000\nOTHER=ignore").unwrap();
        assert!(parser.get("APP_PORT").is_some());
        assert!(parser.get("OTHER").is_none());
    }

    #[test]
    fn type_coercion_bool() {
        let p = parse_str("A=true\nB=false\nC=yes\nD=no");
        assert_eq!(p.get_typed("A").unwrap().as_bool(), Some(true));
        assert_eq!(p.get_typed("B").unwrap().as_bool(), Some(false));
        assert_eq!(p.get_typed("C").unwrap().as_bool(), Some(true));
        assert_eq!(p.get_typed("D").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn type_coercion_int() {
        let p = parse_str("N=42");
        assert_eq!(p.get_typed("N").unwrap().as_int(), Some(42));
    }

    #[test]
    fn type_coercion_float() {
        let p = parse_str("PI=3.14");
        let val = p.get_typed("PI").unwrap().as_float().unwrap();
        assert!((val - 3.14).abs() < 1e-10);
    }

    #[test]
    fn get_int_explicit() {
        let p = parse_str("PORT=8080");
        assert_eq!(p.get_int("PORT").unwrap(), 8080);
    }

    #[test]
    fn get_bool_explicit() {
        let p = parse_str("DEBUG=true");
        assert!(p.get_bool("DEBUG").unwrap());
    }

    #[test]
    fn get_or_default() {
        let p = parse_str("A=hello");
        assert_eq!(p.get_or("A", "default"), "hello");
        assert_eq!(p.get_or("MISSING", "fallback"), "fallback");
    }

    #[test]
    fn syntax_error_no_equals() {
        let mut parser = EnvParser::new();
        let result = parser.parse("BAD LINE");
        assert!(matches!(result, Err(EnvParseError::SyntaxError { .. })));
    }

    #[test]
    fn empty_key_error() {
        let mut parser = EnvParser::new();
        let result = parser.parse("=value");
        assert!(matches!(result, Err(EnvParseError::SyntaxError { .. })));
    }

    #[test]
    fn seed_before_parse() {
        let mut parser = EnvParser::new();
        parser.seed("BASE", "/opt");
        parser.parse("FULL=${BASE}/data").unwrap();
        assert_eq!(parser.get("FULL"), Some("/opt/data"));
    }

    #[test]
    fn prefix_stripped() {
        let p = parse_str("APP_HOST=localhost\nAPP_PORT=3000\nOTHER=x");
        let stripped = p.with_prefix_stripped("APP");
        assert_eq!(stripped.get("HOST").map(|s| s.as_str()), Some("localhost"));
        assert_eq!(stripped.get("PORT").map(|s| s.as_str()), Some("3000"));
        assert!(!stripped.contains_key("OTHER"));
    }

    #[test]
    fn is_empty_and_len() {
        let p = EnvParser::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);

        let p2 = parse_str("K=V");
        assert!(!p2.is_empty());
        assert_eq!(p2.len(), 1);
    }

    #[test]
    fn escape_sequences_double_quote() {
        let p = parse_str("V=\"tab\\there\\\\\"");
        assert_eq!(p.get("V"), Some("tab\there\\"));
    }

    #[test]
    fn coercion_error_on_bad_int() {
        let p = parse_str("X=not_a_number");
        assert!(matches!(p.get_int("X"), Err(EnvParseError::CoercionError { .. })));
    }
}
