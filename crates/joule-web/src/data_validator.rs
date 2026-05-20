//! Data validation rules engine.
//!
//! Replaces `Joi`, `Yup`, `Cerberus`, and similar validation libraries with a
//! pure-Rust rules engine. Supports rule definitions (required, type check, range,
//! regex, custom), row-level validation, validation reports, field-level errors,
//! severity levels (error/warning/info), and validation summary statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors from the validation engine itself (not validation failures).
#[derive(Debug, Clone, PartialEq)]
pub enum ValidatorError {
    /// Rule already exists with this name.
    DuplicateRule(String),
    /// Field name is empty.
    EmptyFieldName,
    /// Invalid regex pattern.
    InvalidPattern(String),
}

impl fmt::Display for ValidatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRule(name) => write!(f, "duplicate rule: {name}"),
            Self::EmptyFieldName => write!(f, "field name cannot be empty"),
            Self::InvalidPattern(p) => write!(f, "invalid pattern: {p}"),
        }
    }
}

impl std::error::Error for ValidatorError {}

// ── Severity ─────────────────────────────────────────────────────

/// Severity level for a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ── Expected type ────────────────────────────────────────────────

/// Expected JSON value type for type-check rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

impl ExpectedType {
    /// Check if a JSON value matches this expected type.
    fn matches(&self, value: &serde_json::Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Number => value.is_number(),
            Self::Integer => value.is_i64() || value.is_u64(),
            Self::Boolean => value.is_boolean(),
            Self::Array => value.is_array(),
            Self::Object => value.is_object(),
            Self::Null => value.is_null(),
        }
    }
}

impl fmt::Display for ExpectedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Number => write!(f, "number"),
            Self::Integer => write!(f, "integer"),
            Self::Boolean => write!(f, "boolean"),
            Self::Array => write!(f, "array"),
            Self::Object => write!(f, "object"),
            Self::Null => write!(f, "null"),
        }
    }
}

// ── Rule kind ────────────────────────────────────────────────────

/// The kind of validation rule.
#[derive(Debug, Clone)]
pub enum RuleKind {
    /// Field must be present and not null.
    Required,
    /// Field must match the expected type.
    TypeCheck(ExpectedType),
    /// Numeric value must be in [min, max].
    Range { min: f64, max: f64 },
    /// String value must match the pattern (simple contains/prefix/suffix).
    Pattern(PatternRule),
    /// String length must be in [min, max].
    StringLength { min: usize, max: usize },
    /// Array length must be in [min, max].
    ArrayLength { min: usize, max: usize },
    /// Value must be one of the allowed values.
    OneOf(Vec<serde_json::Value>),
    /// Value must NOT be one of the disallowed values.
    NoneOf(Vec<serde_json::Value>),
    /// Custom validation with a description and predicate.
    Custom {
        description: String,
        validator: fn(&serde_json::Value) -> bool,
    },
}

/// Simple pattern matching rule (no regex crate).
#[derive(Debug, Clone)]
pub struct PatternRule {
    /// The pattern kind.
    pub kind: PatternKind,
    /// The pattern string.
    pub pattern: String,
}

/// Pattern matching strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// String contains the pattern.
    Contains,
    /// String starts with the pattern.
    StartsWith,
    /// String ends with the pattern.
    EndsWith,
    /// String exactly matches the pattern.
    Exact,
}

impl PatternRule {
    fn matches(&self, s: &str) -> bool {
        match self.kind {
            PatternKind::Contains => s.contains(&self.pattern),
            PatternKind::StartsWith => s.starts_with(&self.pattern),
            PatternKind::EndsWith => s.ends_with(&self.pattern),
            PatternKind::Exact => s == self.pattern,
        }
    }
}

// ── Validation rule ──────────────────────────────────────────────

/// A named validation rule applied to a specific field.
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name/id.
    pub name: String,
    /// Target field name.
    pub field: String,
    /// The kind of check.
    pub kind: RuleKind,
    /// Severity if the rule fails.
    pub severity: Severity,
    /// Custom error message override.
    pub message: Option<String>,
}

impl ValidationRule {
    /// Create a new validation rule.
    pub fn new(
        name: impl Into<String>,
        field: impl Into<String>,
        kind: RuleKind,
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            kind,
            severity: Severity::Error,
            message: None,
        }
    }

    /// Set severity.
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Set custom error message.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Validate a record (key-value map).
    pub fn validate(&self, record: &HashMap<String, serde_json::Value>) -> Option<ValidationFinding> {
        let value = record.get(&self.field);

        let failed = match &self.kind {
            RuleKind::Required => value.is_none() || value == Some(&serde_json::Value::Null),
            RuleKind::TypeCheck(expected) => {
                match value {
                    None => false, // Missing handled by Required rule
                    Some(v) if v.is_null() => false,
                    Some(v) => !expected.matches(v),
                }
            }
            RuleKind::Range { min, max } => {
                match value.and_then(|v| v.as_f64()) {
                    None => false,
                    Some(n) => n < *min || n > *max,
                }
            }
            RuleKind::Pattern(pat) => {
                match value.and_then(|v| v.as_str()) {
                    None => false,
                    Some(s) => !pat.matches(s),
                }
            }
            RuleKind::StringLength { min, max } => {
                match value.and_then(|v| v.as_str()) {
                    None => false,
                    Some(s) => s.len() < *min || s.len() > *max,
                }
            }
            RuleKind::ArrayLength { min, max } => {
                match value.and_then(|v| v.as_array()) {
                    None => false,
                    Some(a) => a.len() < *min || a.len() > *max,
                }
            }
            RuleKind::OneOf(allowed) => {
                match value {
                    None => false,
                    Some(v) => !allowed.contains(v),
                }
            }
            RuleKind::NoneOf(disallowed) => {
                match value {
                    None => false,
                    Some(v) => disallowed.contains(v),
                }
            }
            RuleKind::Custom { validator, .. } => {
                match value {
                    None => false,
                    Some(v) => !validator(v),
                }
            }
        };

        if failed {
            let default_msg = self.default_message(value);
            Some(ValidationFinding {
                rule_name: self.name.clone(),
                field: self.field.clone(),
                severity: self.severity,
                message: self.message.clone().unwrap_or(default_msg),
                value: value.cloned(),
            })
        } else {
            None
        }
    }

    fn default_message(&self, value: Option<&serde_json::Value>) -> String {
        match &self.kind {
            RuleKind::Required => format!("field '{}' is required", self.field),
            RuleKind::TypeCheck(expected) => {
                format!("field '{}' expected type {expected}", self.field)
            }
            RuleKind::Range { min, max } => {
                let val = value.and_then(|v| v.as_f64()).unwrap_or(0.0);
                format!(
                    "field '{}' value {val} out of range [{min}, {max}]",
                    self.field
                )
            }
            RuleKind::Pattern(pat) => {
                format!(
                    "field '{}' does not match pattern '{}'",
                    self.field, pat.pattern
                )
            }
            RuleKind::StringLength { min, max } => {
                format!(
                    "field '{}' string length not in [{min}, {max}]",
                    self.field
                )
            }
            RuleKind::ArrayLength { min, max } => {
                format!(
                    "field '{}' array length not in [{min}, {max}]",
                    self.field
                )
            }
            RuleKind::OneOf(_) => format!("field '{}' value not in allowed set", self.field),
            RuleKind::NoneOf(_) => format!("field '{}' value is in disallowed set", self.field),
            RuleKind::Custom { description, .. } => {
                format!("field '{}' failed custom check: {description}", self.field)
            }
        }
    }
}

// ── Validation finding ───────────────────────────────────────────

/// A single validation finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFinding {
    /// Which rule triggered this.
    pub rule_name: String,
    /// Which field failed.
    pub field: String,
    /// Severity.
    pub severity: Severity,
    /// Human-readable message.
    pub message: String,
    /// The offending value (if present).
    pub value: Option<serde_json::Value>,
}

// ── Row validation result ────────────────────────────────────────

/// Result of validating a single row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowResult {
    /// Row index.
    pub row_index: usize,
    /// Findings for this row.
    pub findings: Vec<ValidationFinding>,
}

impl RowResult {
    /// Whether this row has any error-severity findings.
    pub fn has_errors(&self) -> bool {
        self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    /// Whether this row passed validation (no errors).
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// Count findings by severity.
    pub fn count_by_severity(&self, severity: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity == severity).count()
    }
}

// ── Validation summary ──────────────────────────────────────────

/// Summary statistics for a validation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total rows validated.
    pub total_rows: usize,
    /// Rows that passed (no errors).
    pub valid_rows: usize,
    /// Rows with at least one error.
    pub invalid_rows: usize,
    /// Total error-level findings.
    pub error_count: usize,
    /// Total warning-level findings.
    pub warning_count: usize,
    /// Total info-level findings.
    pub info_count: usize,
    /// Per-rule failure counts.
    pub rule_failures: HashMap<String, usize>,
    /// Per-field failure counts.
    pub field_failures: HashMap<String, usize>,
}

impl ValidationSummary {
    /// Validity rate as a fraction [0.0, 1.0].
    pub fn validity_rate(&self) -> f64 {
        if self.total_rows == 0 {
            1.0
        } else {
            self.valid_rows as f64 / self.total_rows as f64
        }
    }
}

// ── Validation report ────────────────────────────────────────────

/// Full validation report with per-row results and summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Per-row results.
    pub rows: Vec<RowResult>,
    /// Summary statistics.
    pub summary: ValidationSummary,
}

impl ValidationReport {
    /// Get all findings with a specific severity.
    pub fn findings_by_severity(&self, severity: Severity) -> Vec<&ValidationFinding> {
        self.rows
            .iter()
            .flat_map(|r| r.findings.iter())
            .filter(|f| f.severity == severity)
            .collect()
    }

    /// Get all findings for a specific field.
    pub fn findings_for_field<'a>(&'a self, field: &str) -> Vec<&'a ValidationFinding> {
        self.rows
            .iter()
            .flat_map(|r| r.findings.iter())
            .filter(|f| f.field == field)
            .collect()
    }

    /// Whether the entire dataset passed validation.
    pub fn is_valid(&self) -> bool {
        self.summary.invalid_rows == 0
    }
}

// ── Data validator ───────────────────────────────────────────────

/// The validation engine that holds rules and validates datasets.
#[derive(Debug)]
pub struct DataValidator {
    /// Ordered list of validation rules.
    rules: Vec<ValidationRule>,
    /// Whether to stop validating a row after the first error.
    fail_fast: bool,
}

impl DataValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            fail_fast: false,
        }
    }

    /// Set fail-fast mode.
    pub fn set_fail_fast(&mut self, fail_fast: bool) {
        self.fail_fast = fail_fast;
    }

    /// Add a validation rule.
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Validate a single row.
    pub fn validate_row(
        &self,
        row_index: usize,
        record: &HashMap<String, serde_json::Value>,
    ) -> RowResult {
        let mut findings = Vec::new();

        for rule in &self.rules {
            if let Some(finding) = rule.validate(record) {
                let is_error = finding.severity == Severity::Error;
                findings.push(finding);
                if self.fail_fast && is_error {
                    break;
                }
            }
        }

        RowResult { row_index, findings }
    }

    /// Validate a dataset (slice of records).
    pub fn validate(
        &self,
        data: &[HashMap<String, serde_json::Value>],
    ) -> ValidationReport {
        let mut rows = Vec::with_capacity(data.len());
        let mut summary = ValidationSummary {
            total_rows: data.len(),
            ..Default::default()
        };

        for (i, record) in data.iter().enumerate() {
            let row_result = self.validate_row(i, record);

            if row_result.is_valid() {
                summary.valid_rows += 1;
            } else {
                summary.invalid_rows += 1;
            }

            for finding in &row_result.findings {
                match finding.severity {
                    Severity::Error => summary.error_count += 1,
                    Severity::Warning => summary.warning_count += 1,
                    Severity::Info => summary.info_count += 1,
                }
                *summary.rule_failures.entry(finding.rule_name.clone()).or_insert(0) += 1;
                *summary.field_failures.entry(finding.field.clone()).or_insert(0) += 1;
            }

            rows.push(row_result);
        }

        ValidationReport { rows, summary }
    }
}

impl Default for DataValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Convenience constructors ─────────────────────────────────────

/// Create a "required" rule for a field.
pub fn required(field: impl Into<String>) -> ValidationRule {
    let f: String = field.into();
    ValidationRule::new(format!("{f}_required"), f, RuleKind::Required)
}

/// Create a type-check rule.
pub fn type_check(field: impl Into<String>, expected: ExpectedType) -> ValidationRule {
    let f: String = field.into();
    ValidationRule::new(format!("{f}_type"), f, RuleKind::TypeCheck(expected))
}

/// Create a range rule.
pub fn range(field: impl Into<String>, min: f64, max: f64) -> ValidationRule {
    let f: String = field.into();
    ValidationRule::new(format!("{f}_range"), f, RuleKind::Range { min, max })
}

/// Create a string-length rule.
pub fn string_length(field: impl Into<String>, min: usize, max: usize) -> ValidationRule {
    let f: String = field.into();
    ValidationRule::new(format!("{f}_strlen"), f, RuleKind::StringLength { min, max })
}

/// Create a pattern rule.
pub fn pattern(
    field: impl Into<String>,
    kind: PatternKind,
    pat: impl Into<String>,
) -> ValidationRule {
    let f: String = field.into();
    let p: String = pat.into();
    ValidationRule::new(
        format!("{f}_pattern"),
        f,
        RuleKind::Pattern(PatternRule {
            kind,
            pattern: p,
        }),
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn required_field_present() {
        let v = DataValidator::new();
        let r = required("name");
        let record = row(&[("name", serde_json::json!("Alice"))]);
        assert!(r.validate(&record).is_none());
    }

    #[test]
    fn required_field_missing() {
        let r = required("name");
        let record = row(&[("age", serde_json::json!(30))]);
        let finding = r.validate(&record).unwrap();
        assert_eq!(finding.severity, Severity::Error);
        assert_eq!(finding.field, "name");
    }

    #[test]
    fn required_field_null() {
        let r = required("name");
        let record = row(&[("name", serde_json::Value::Null)]);
        assert!(r.validate(&record).is_some());
    }

    #[test]
    fn type_check_string() {
        let r = type_check("name", ExpectedType::String);
        let good = row(&[("name", serde_json::json!("Alice"))]);
        let bad = row(&[("name", serde_json::json!(42))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn type_check_number() {
        let r = type_check("score", ExpectedType::Number);
        let good = row(&[("score", serde_json::json!(3.14))]);
        let bad = row(&[("score", serde_json::json!("three"))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn type_check_boolean() {
        let r = type_check("active", ExpectedType::Boolean);
        let good = row(&[("active", serde_json::json!(true))]);
        assert!(r.validate(&good).is_none());
    }

    #[test]
    fn range_in_bounds() {
        let r = range("age", 0.0, 150.0);
        let good = row(&[("age", serde_json::json!(25))]);
        assert!(r.validate(&good).is_none());
    }

    #[test]
    fn range_out_of_bounds() {
        let r = range("age", 0.0, 150.0);
        let bad = row(&[("age", serde_json::json!(200))]);
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn range_below_min() {
        let r = range("temp", -40.0, 60.0);
        let bad = row(&[("temp", serde_json::json!(-50.0))]);
        let finding = r.validate(&bad).unwrap();
        assert!(finding.message.contains("out of range"));
    }

    #[test]
    fn string_length_valid() {
        let r = string_length("code", 2, 10);
        let good = row(&[("code", serde_json::json!("ABC"))]);
        assert!(r.validate(&good).is_none());
    }

    #[test]
    fn string_length_too_short() {
        let r = string_length("code", 2, 10);
        let bad = row(&[("code", serde_json::json!("A"))]);
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn pattern_contains() {
        let r = pattern("email", PatternKind::Contains, "@");
        let good = row(&[("email", serde_json::json!("user@example.com"))]);
        let bad = row(&[("email", serde_json::json!("no-at-sign"))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn pattern_starts_with() {
        let r = pattern("url", PatternKind::StartsWith, "https://");
        let good = row(&[("url", serde_json::json!("https://example.com"))]);
        let bad = row(&[("url", serde_json::json!("http://example.com"))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn pattern_ends_with() {
        let r = pattern("file", PatternKind::EndsWith, ".csv");
        let good = row(&[("file", serde_json::json!("data.csv"))]);
        assert!(r.validate(&good).is_none());
    }

    #[test]
    fn one_of_valid() {
        let r = ValidationRule::new(
            "status_check",
            "status",
            RuleKind::OneOf(vec![
                serde_json::json!("active"),
                serde_json::json!("inactive"),
            ]),
        );
        let good = row(&[("status", serde_json::json!("active"))]);
        let bad = row(&[("status", serde_json::json!("deleted"))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn none_of_valid() {
        let r = ValidationRule::new(
            "no_banned",
            "role",
            RuleKind::NoneOf(vec![serde_json::json!("banned")]),
        );
        let good = row(&[("role", serde_json::json!("user"))]);
        let bad = row(&[("role", serde_json::json!("banned"))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn custom_rule() {
        fn is_even(v: &serde_json::Value) -> bool {
            v.as_u64().map_or(false, |n| n % 2 == 0)
        }
        let r = ValidationRule::new(
            "even_check",
            "count",
            RuleKind::Custom {
                description: "must be even".into(),
                validator: is_even,
            },
        );
        let good = row(&[("count", serde_json::json!(4))]);
        let bad = row(&[("count", serde_json::json!(3))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&bad).is_some());
    }

    #[test]
    fn validate_dataset() {
        let mut validator = DataValidator::new();
        validator.add_rule(required("name"));
        validator.add_rule(type_check("name", ExpectedType::String));
        validator.add_rule(range("age", 0.0, 150.0));

        let data = vec![
            row(&[("name", serde_json::json!("Alice")), ("age", serde_json::json!(30))]),
            row(&[("age", serde_json::json!(25))]), // missing name
            row(&[("name", serde_json::json!("Bob")), ("age", serde_json::json!(200))]), // age out of range
        ];

        let report = validator.validate(&data);
        assert_eq!(report.summary.total_rows, 3);
        assert_eq!(report.summary.valid_rows, 1);
        assert_eq!(report.summary.invalid_rows, 2);
        assert!(report.summary.error_count >= 2);
    }

    #[test]
    fn validation_report_queries() {
        let mut validator = DataValidator::new();
        validator.add_rule(required("name"));
        validator.add_rule(
            range("score", 0.0, 100.0).with_severity(Severity::Warning),
        );

        let data = vec![
            row(&[("score", serde_json::json!(150))]), // missing name (error) + score out of range (warning)
        ];

        let report = validator.validate(&data);
        assert_eq!(report.findings_by_severity(Severity::Error).len(), 1);
        assert_eq!(report.findings_by_severity(Severity::Warning).len(), 1);
        assert!(!report.is_valid());
    }

    #[test]
    fn findings_for_field() {
        let mut validator = DataValidator::new();
        validator.add_rule(required("email"));
        validator.add_rule(pattern("email", PatternKind::Contains, "@"));

        let data = vec![
            row(&[("email", serde_json::json!("bad"))]),
        ];

        let report = validator.validate(&data);
        let email_findings = report.findings_for_field("email");
        // Pattern fails because "bad" doesn't contain "@"
        assert!(!email_findings.is_empty());
    }

    #[test]
    fn fail_fast_stops_at_first_error() {
        let mut validator = DataValidator::new();
        validator.set_fail_fast(true);
        validator.add_rule(required("a"));
        validator.add_rule(required("b"));
        validator.add_rule(required("c"));

        let data = vec![row(&[])]; // all three fields missing
        let report = validator.validate(&data);
        // In fail-fast mode, should stop after first error
        assert_eq!(report.rows[0].findings.len(), 1);
    }

    #[test]
    fn custom_message_override() {
        let r = required("name").with_message("Name is mandatory");
        let record = row(&[]);
        let finding = r.validate(&record).unwrap();
        assert_eq!(finding.message, "Name is mandatory");
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn validity_rate() {
        let summary = ValidationSummary {
            total_rows: 10,
            valid_rows: 8,
            invalid_rows: 2,
            ..Default::default()
        };
        assert!((summary.validity_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn validity_rate_empty() {
        let summary = ValidationSummary::default();
        assert!((summary.validity_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn row_result_count_by_severity() {
        let rr = RowResult {
            row_index: 0,
            findings: vec![
                ValidationFinding {
                    rule_name: "r1".into(),
                    field: "f1".into(),
                    severity: Severity::Error,
                    message: "err".into(),
                    value: None,
                },
                ValidationFinding {
                    rule_name: "r2".into(),
                    field: "f2".into(),
                    severity: Severity::Warning,
                    message: "warn".into(),
                    value: None,
                },
                ValidationFinding {
                    rule_name: "r3".into(),
                    field: "f3".into(),
                    severity: Severity::Error,
                    message: "err2".into(),
                    value: None,
                },
            ],
        };
        assert_eq!(rr.count_by_severity(Severity::Error), 2);
        assert_eq!(rr.count_by_severity(Severity::Warning), 1);
        assert_eq!(rr.count_by_severity(Severity::Info), 0);
    }

    #[test]
    fn array_length_rule() {
        let r = ValidationRule::new(
            "tags_len",
            "tags",
            RuleKind::ArrayLength { min: 1, max: 5 },
        );
        let good = row(&[("tags", serde_json::json!(["a", "b"]))]);
        let too_many = row(&[("tags", serde_json::json!(["a", "b", "c", "d", "e", "f"]))]);
        let empty = row(&[("tags", serde_json::json!([]))]);
        assert!(r.validate(&good).is_none());
        assert!(r.validate(&too_many).is_some());
        assert!(r.validate(&empty).is_some());
    }

    #[test]
    fn missing_field_skips_type_check() {
        let r = type_check("optional_field", ExpectedType::String);
        let record = row(&[]);
        // Missing field should not trigger type check failure
        assert!(r.validate(&record).is_none());
    }

    #[test]
    fn error_display() {
        let e = ValidatorError::DuplicateRule("r1".into());
        assert!(format!("{e}").contains("duplicate rule"));
    }

    #[test]
    fn severity_display() {
        assert_eq!(format!("{}", Severity::Error), "error");
        assert_eq!(format!("{}", Severity::Warning), "warning");
        assert_eq!(format!("{}", Severity::Info), "info");
    }

    #[test]
    fn expected_type_display() {
        assert_eq!(format!("{}", ExpectedType::String), "string");
        assert_eq!(format!("{}", ExpectedType::Number), "number");
        assert_eq!(format!("{}", ExpectedType::Integer), "integer");
    }
}
