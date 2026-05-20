//! Validation engine — rule definitions, field validators, chained/conditional/cross-field validation.
//!
//! Replaces Yup, Zod, Joi, and class-validator with a pure-Rust validation
//! engine that composes rules declaratively and produces structured error messages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// A single validation error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationError {
    /// Field path (e.g. "user.email").
    pub field: String,
    /// Rule that failed.
    pub rule: String,
    /// Human-readable message.
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.field, self.message, self.rule)
    }
}

impl std::error::Error for ValidationError {}

/// Aggregated validation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn ok() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn errors_for(&self, field: &str) -> Vec<&ValidationError> {
        self.errors.iter().filter(|e| e.field == field).collect()
    }

    pub fn merge(&mut self, other: ValidationResult) {
        self.errors.extend(other.errors);
    }

    pub fn error_messages(&self) -> Vec<String> {
        self.errors.iter().map(|e| e.message.clone()).collect()
    }
}

// ── Rules ───────────────────────────────────────────────────────

/// A single validation rule.
#[derive(Debug, Clone)]
pub enum Rule {
    /// Field must not be empty / missing.
    Required,
    /// String minimum length.
    MinLength(usize),
    /// String maximum length.
    MaxLength(usize),
    /// Numeric range (inclusive).
    Range { min: f64, max: f64 },
    /// Simple email pattern check.
    Email,
    /// URL pattern check.
    Url,
    /// Custom regex pattern.
    Pattern { regex: String, description: String },
    /// Value must be one of the given options.
    OneOf(Vec<String>),
    /// Custom predicate with description.
    Custom {
        description: String,
        predicate: fn(&str) -> bool,
    },
}

impl Rule {
    /// Validate a single string value against this rule.
    pub fn validate(&self, value: &str) -> Result<(), String> {
        match self {
            Rule::Required => {
                if value.trim().is_empty() {
                    Err("field is required".into())
                } else {
                    Ok(())
                }
            }
            Rule::MinLength(min) => {
                if value.len() < *min {
                    Err(format!("must be at least {min} characters"))
                } else {
                    Ok(())
                }
            }
            Rule::MaxLength(max) => {
                if value.len() > *max {
                    Err(format!("must be at most {max} characters"))
                } else {
                    Ok(())
                }
            }
            Rule::Range { min, max } => {
                let v: f64 = value
                    .parse()
                    .map_err(|_| "must be a valid number".to_string())?;
                if v < *min || v > *max {
                    Err(format!("must be between {min} and {max}"))
                } else {
                    Ok(())
                }
            }
            Rule::Email => {
                // Simplified email check: contains exactly one @, has text before and after,
                // and has a dot in the domain part.
                let parts: Vec<&str> = value.split('@').collect();
                if parts.len() != 2
                    || parts[0].is_empty()
                    || parts[1].is_empty()
                    || !parts[1].contains('.')
                {
                    Err("must be a valid email address".into())
                } else {
                    Ok(())
                }
            }
            Rule::Url => {
                if value.starts_with("http://") || value.starts_with("https://") {
                    let rest = value.split("://").nth(1).unwrap_or("");
                    if rest.contains('.') && rest.len() > 3 {
                        Ok(())
                    } else {
                        Err("must be a valid URL".into())
                    }
                } else {
                    Err("must be a valid URL (http:// or https://)".into())
                }
            }
            Rule::Pattern { regex, description } => {
                // Simple pattern matching without regex crate:
                // We support a subset: ^...$, literal chars, \d, \w, .*, .+
                if simple_pattern_match(regex, value) {
                    Ok(())
                } else {
                    Err(format!("must match pattern: {description}"))
                }
            }
            Rule::OneOf(options) => {
                if options.iter().any(|o| o == value) {
                    Ok(())
                } else {
                    Err(format!("must be one of: {}", options.join(", ")))
                }
            }
            Rule::Custom {
                description,
                predicate,
            } => {
                if predicate(value) {
                    Ok(())
                } else {
                    Err(description.clone())
                }
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            Rule::Required => "required",
            Rule::MinLength(_) => "min_length",
            Rule::MaxLength(_) => "max_length",
            Rule::Range { .. } => "range",
            Rule::Email => "email",
            Rule::Url => "url",
            Rule::Pattern { .. } => "pattern",
            Rule::OneOf(_) => "one_of",
            Rule::Custom { .. } => "custom",
        }
    }
}

/// Simple pattern matching for common patterns without regex crate.
fn simple_pattern_match(pattern: &str, value: &str) -> bool {
    // Handle a few common cases
    let pat = pattern.trim_start_matches('^').trim_end_matches('$');

    if pat == ".*" || pat == ".+" && !value.is_empty() {
        return true;
    }

    // Check if pattern is purely literal
    if !pat.contains('\\') && !pat.contains('.') && !pat.contains('*') && !pat.contains('+') {
        return value.contains(pat);
    }

    // For digit-only pattern like \d+
    if pat == r"\d+" {
        return !value.is_empty() && value.chars().all(|c| c.is_ascii_digit());
    }

    // For word chars \w+
    if pat == r"\w+" {
        return !value.is_empty() && value.chars().all(|c| c.is_alphanumeric() || c == '_');
    }

    // Fallback: treat as literal contains
    value.contains(pat)
}

// ── Field Validator ─────────────────────────────────────────────

/// Validator for a single named field with chained rules.
#[derive(Debug, Clone)]
pub struct FieldValidator {
    pub field: String,
    pub rules: Vec<Rule>,
    /// Optional custom message override per rule index.
    pub custom_messages: HashMap<usize, String>,
}

impl FieldValidator {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            rules: Vec::new(),
            custom_messages: HashMap::new(),
        }
    }

    pub fn rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn required(self) -> Self {
        self.rule(Rule::Required)
    }

    pub fn min_length(self, min: usize) -> Self {
        self.rule(Rule::MinLength(min))
    }

    pub fn max_length(self, max: usize) -> Self {
        self.rule(Rule::MaxLength(max))
    }

    pub fn email(self) -> Self {
        self.rule(Rule::Email)
    }

    pub fn url(self) -> Self {
        self.rule(Rule::Url)
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        let idx = self.rules.len().saturating_sub(1);
        self.custom_messages.insert(idx, message.into());
        self
    }

    pub fn validate(&self, value: &str) -> ValidationResult {
        let mut result = ValidationResult::ok();
        for (i, rule) in self.rules.iter().enumerate() {
            if let Err(default_msg) = rule.validate(value) {
                let message = self
                    .custom_messages
                    .get(&i)
                    .cloned()
                    .unwrap_or(default_msg);
                result.errors.push(ValidationError {
                    field: self.field.clone(),
                    rule: rule.name().to_string(),
                    message,
                });
            }
        }
        result
    }
}

// ── Conditional Validation ──────────────────────────────────────

/// Condition for conditional validation.
pub enum Condition {
    /// Validate only if the given field equals the specified value.
    FieldEquals { field: String, value: String },
    /// Validate only if the given field is non-empty.
    FieldPresent(String),
    /// Custom condition.
    Predicate(fn(&HashMap<String, String>) -> bool),
}

impl Condition {
    pub fn check(&self, data: &HashMap<String, String>) -> bool {
        match self {
            Condition::FieldEquals { field, value } => {
                data.get(field).map_or(false, |v| v == value)
            }
            Condition::FieldPresent(field) => {
                data.get(field).map_or(false, |v| !v.trim().is_empty())
            }
            Condition::Predicate(f) => f(data),
        }
    }
}

// ── Cross-Field Validation ──────────────────────────────────────

/// A cross-field validation rule that examines multiple fields.
pub struct CrossFieldRule {
    pub fields: Vec<String>,
    pub description: String,
    pub check: fn(&HashMap<String, String>) -> Result<(), String>,
}

impl CrossFieldRule {
    pub fn validate(&self, data: &HashMap<String, String>) -> ValidationResult {
        let mut result = ValidationResult::ok();
        if let Err(msg) = (self.check)(data) {
            result.errors.push(ValidationError {
                field: self.fields.join(", "),
                rule: "cross_field".to_string(),
                message: msg,
            });
        }
        result
    }
}

// ── Form Validator ──────────────────────────────────────────────

/// A complete form validator with multiple field validators, conditional rules,
/// and cross-field constraints.
pub struct FormValidator {
    pub field_validators: Vec<FieldValidator>,
    pub conditional_validators: Vec<(Condition, FieldValidator)>,
    pub cross_field_rules: Vec<CrossFieldRule>,
}

impl FormValidator {
    pub fn new() -> Self {
        Self {
            field_validators: Vec::new(),
            conditional_validators: Vec::new(),
            cross_field_rules: Vec::new(),
        }
    }

    pub fn field(mut self, validator: FieldValidator) -> Self {
        self.field_validators.push(validator);
        self
    }

    pub fn conditional(mut self, condition: Condition, validator: FieldValidator) -> Self {
        self.conditional_validators.push((condition, validator));
        self
    }

    pub fn cross_field(mut self, rule: CrossFieldRule) -> Self {
        self.cross_field_rules.push(rule);
        self
    }

    pub fn validate(&self, data: &HashMap<String, String>) -> ValidationResult {
        let mut result = ValidationResult::ok();

        // Validate each field.
        for fv in &self.field_validators {
            let value = data.get(&fv.field).map(|s| s.as_str()).unwrap_or("");
            result.merge(fv.validate(value));
        }

        // Conditional validators.
        for (condition, fv) in &self.conditional_validators {
            if condition.check(data) {
                let value = data.get(&fv.field).map(|s| s.as_str()).unwrap_or("");
                result.merge(fv.validate(value));
            }
        }

        // Cross-field rules.
        for rule in &self.cross_field_rules {
            result.merge(rule.validate(data));
        }

        result
    }
}

impl Default for FormValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_rule_rejects_empty() {
        let r = Rule::Required;
        assert!(r.validate("").is_err());
        assert!(r.validate("   ").is_err());
        assert!(r.validate("hello").is_ok());
    }

    #[test]
    fn min_max_length() {
        assert!(Rule::MinLength(3).validate("ab").is_err());
        assert!(Rule::MinLength(3).validate("abc").is_ok());
        assert!(Rule::MaxLength(5).validate("abcdef").is_err());
        assert!(Rule::MaxLength(5).validate("abcde").is_ok());
    }

    #[test]
    fn range_rule() {
        let r = Rule::Range {
            min: 1.0,
            max: 100.0,
        };
        assert!(r.validate("0").is_err());
        assert!(r.validate("50").is_ok());
        assert!(r.validate("101").is_err());
        assert!(r.validate("abc").is_err());
    }

    #[test]
    fn email_rule() {
        assert!(Rule::Email.validate("user@example.com").is_ok());
        assert!(Rule::Email.validate("user@localhost").is_err());
        assert!(Rule::Email.validate("@example.com").is_err());
        assert!(Rule::Email.validate("user@").is_err());
        assert!(Rule::Email.validate("plaintext").is_err());
    }

    #[test]
    fn url_rule() {
        assert!(Rule::Url.validate("https://example.com").is_ok());
        assert!(Rule::Url.validate("http://foo.bar").is_ok());
        assert!(Rule::Url.validate("ftp://example.com").is_err());
        assert!(Rule::Url.validate("not-a-url").is_err());
    }

    #[test]
    fn one_of_rule() {
        let r = Rule::OneOf(vec!["a".into(), "b".into(), "c".into()]);
        assert!(r.validate("a").is_ok());
        assert!(r.validate("d").is_err());
    }

    #[test]
    fn custom_rule() {
        let r = Rule::Custom {
            description: "must start with X".into(),
            predicate: |v| v.starts_with('X'),
        };
        assert!(r.validate("X123").is_ok());
        assert!(r.validate("Y123").is_err());
    }

    #[test]
    fn field_validator_chaining() {
        let fv = FieldValidator::new("username")
            .required()
            .min_length(3)
            .max_length(20);

        let res = fv.validate("");
        assert!(!res.is_valid());
        assert_eq!(res.errors.len(), 2); // required + min_length

        let res = fv.validate("ab");
        assert!(!res.is_valid());
        assert_eq!(res.errors.len(), 1); // min_length

        let res = fv.validate("alice");
        assert!(res.is_valid());
    }

    #[test]
    fn custom_message_override() {
        let fv = FieldValidator::new("email")
            .required()
            .with_message("Please enter your email");
        let res = fv.validate("");
        assert_eq!(res.errors[0].message, "Please enter your email");
    }

    #[test]
    fn form_validator_basic() {
        let form = FormValidator::new()
            .field(FieldValidator::new("name").required().min_length(2))
            .field(FieldValidator::new("email").required().email());

        let mut data = HashMap::new();
        data.insert("name".into(), "".into());
        data.insert("email".into(), "bad".into());

        let res = form.validate(&data);
        assert!(!res.is_valid());
        assert!(!res.errors_for("name").is_empty());
        assert!(!res.errors_for("email").is_empty());
    }

    #[test]
    fn conditional_validation() {
        let form = FormValidator::new().conditional(
            Condition::FieldEquals {
                field: "type".into(),
                value: "business".into(),
            },
            FieldValidator::new("company").required(),
        );

        let mut data = HashMap::new();
        data.insert("type".into(), "personal".into());
        let res = form.validate(&data);
        assert!(res.is_valid()); // condition not met, skip

        data.insert("type".into(), "business".into());
        let res = form.validate(&data);
        assert!(!res.is_valid()); // company missing
    }

    #[test]
    fn cross_field_validation() {
        let form = FormValidator::new().cross_field(CrossFieldRule {
            fields: vec!["password".into(), "confirm".into()],
            description: "passwords must match".into(),
            check: |data| {
                let pw = data.get("password").map(|s| s.as_str()).unwrap_or("");
                let cf = data.get("confirm").map(|s| s.as_str()).unwrap_or("");
                if pw == cf {
                    Ok(())
                } else {
                    Err("passwords do not match".into())
                }
            },
        });

        let mut data = HashMap::new();
        data.insert("password".into(), "secret".into());
        data.insert("confirm".into(), "different".into());
        assert!(!form.validate(&data).is_valid());

        data.insert("confirm".into(), "secret".into());
        assert!(form.validate(&data).is_valid());
    }

    #[test]
    fn validation_result_merge() {
        let mut r1 = ValidationResult::ok();
        r1.errors.push(ValidationError {
            field: "a".into(),
            rule: "required".into(),
            message: "a is required".into(),
        });
        let mut r2 = ValidationResult::ok();
        r2.errors.push(ValidationError {
            field: "b".into(),
            rule: "required".into(),
            message: "b is required".into(),
        });
        r1.merge(r2);
        assert_eq!(r1.errors.len(), 2);
    }

    #[test]
    fn pattern_digit_only() {
        let r = Rule::Pattern {
            regex: r"^\d+$".into(),
            description: "digits only".into(),
        };
        assert!(r.validate("12345").is_ok());
        assert!(r.validate("abc").is_err());
    }

    #[test]
    fn field_present_condition() {
        let cond = Condition::FieldPresent("phone".into());
        let mut data = HashMap::new();
        assert!(!cond.check(&data));
        data.insert("phone".into(), "  ".into());
        assert!(!cond.check(&data));
        data.insert("phone".into(), "555-1234".into());
        assert!(cond.check(&data));
    }
}
