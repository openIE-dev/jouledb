//! Specification pattern — boolean specifications (and/or/not composition),
//! parameterized specs, evaluation, specification-based queries, candidate
//! filtering, spec serialization, and composite specifications.
//!
//! Replaces ad-hoc predicate patterns in JS/TS with a pure-Rust specification
//! pattern that supports composable, testable business rules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Specification domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    /// Evaluation error.
    EvaluationError(String),
    /// Invalid parameter.
    InvalidParameter { name: String, reason: String },
    /// Missing parameter.
    MissingParameter(String),
    /// Serialization error.
    SerializationError(String),
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvaluationError(msg) => write!(f, "evaluation error: {msg}"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter {name}: {reason}")
            }
            Self::MissingParameter(name) => write!(f, "missing parameter: {name}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl std::error::Error for SpecError {}

// ── Candidate ───────────────────────────────────────────────────

/// A candidate that can be evaluated against specifications.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub fields: HashMap<String, String>,
}

impl Candidate {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), fields: HashMap::new() }
    }

    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    pub fn get_field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    pub fn get_field_as_i64(&self, key: &str) -> Option<i64> {
        self.fields.get(key).and_then(|v| v.parse().ok())
    }

    pub fn get_field_as_f64(&self, key: &str) -> Option<f64> {
        self.fields.get(key).and_then(|v| v.parse().ok())
    }
}

// ── Comparison ──────────────────────────────────────────────────

/// Comparison operators for field-based specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
}

impl ComparisonOp {
    /// Evaluate a string comparison.
    pub fn compare_str(&self, left: &str, right: &str) -> bool {
        match self {
            Self::Equal => left == right,
            Self::NotEqual => left != right,
            Self::GreaterThan => left > right,
            Self::GreaterThanOrEqual => left >= right,
            Self::LessThan => left < right,
            Self::LessThanOrEqual => left <= right,
        }
    }

    /// Evaluate a numeric comparison.
    pub fn compare_i64(&self, left: i64, right: i64) -> bool {
        match self {
            Self::Equal => left == right,
            Self::NotEqual => left != right,
            Self::GreaterThan => left > right,
            Self::GreaterThanOrEqual => left >= right,
            Self::LessThan => left < right,
            Self::LessThanOrEqual => left <= right,
        }
    }

    /// Evaluate a float comparison.
    pub fn compare_f64(&self, left: f64, right: f64) -> bool {
        match self {
            Self::Equal => (left - right).abs() < f64::EPSILON,
            Self::NotEqual => (left - right).abs() >= f64::EPSILON,
            Self::GreaterThan => left > right,
            Self::GreaterThanOrEqual => left >= right,
            Self::LessThan => left < right,
            Self::LessThanOrEqual => left <= right,
        }
    }
}

// ── Spec ────────────────────────────────────────────────────────

/// A composable specification that can evaluate candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Spec {
    /// Field equals a string value.
    FieldEquals { field: String, value: String },
    /// Field compared with an operator.
    FieldCompare { field: String, op: ComparisonOp, value: String },
    /// Field is numeric and compared with an operator.
    NumericCompare { field: String, op: ComparisonOp, value: i64 },
    /// Field contains a substring.
    Contains { field: String, substring: String },
    /// Field starts with a prefix.
    StartsWith { field: String, prefix: String },
    /// Field ends with a suffix.
    EndsWith { field: String, suffix: String },
    /// Field exists (non-null).
    FieldExists(String),
    /// Field is one of a set of values.
    In { field: String, values: Vec<String> },
    /// Conjunction: all must match.
    And(Vec<Spec>),
    /// Disjunction: at least one must match.
    Or(Vec<Spec>),
    /// Negation.
    Not(Box<Spec>),
    /// Always true.
    True,
    /// Always false.
    False,
}

impl Spec {
    // ── Factory methods ──

    pub fn field_equals(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::FieldEquals { field: field.into(), value: value.into() }
    }

    pub fn field_compare(
        field: impl Into<String>,
        op: ComparisonOp,
        value: impl Into<String>,
    ) -> Self {
        Self::FieldCompare { field: field.into(), op, value: value.into() }
    }

    pub fn numeric_compare(field: impl Into<String>, op: ComparisonOp, value: i64) -> Self {
        Self::NumericCompare { field: field.into(), op, value }
    }

    pub fn contains(field: impl Into<String>, substring: impl Into<String>) -> Self {
        Self::Contains { field: field.into(), substring: substring.into() }
    }

    pub fn starts_with(field: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self::StartsWith { field: field.into(), prefix: prefix.into() }
    }

    pub fn ends_with(field: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self::EndsWith { field: field.into(), suffix: suffix.into() }
    }

    pub fn field_exists(field: impl Into<String>) -> Self {
        Self::FieldExists(field.into())
    }

    pub fn is_in(field: impl Into<String>, values: Vec<String>) -> Self {
        Self::In { field: field.into(), values }
    }

    // ── Composition ──

    /// Logical AND.
    pub fn and(self, other: Spec) -> Spec {
        match self {
            Spec::And(mut specs) => {
                specs.push(other);
                Spec::And(specs)
            }
            _ => Spec::And(vec![self, other]),
        }
    }

    /// Logical OR.
    pub fn or(self, other: Spec) -> Spec {
        match self {
            Spec::Or(mut specs) => {
                specs.push(other);
                Spec::Or(specs)
            }
            _ => Spec::Or(vec![self, other]),
        }
    }

    /// Logical NOT.
    pub fn not(self) -> Spec {
        Spec::Not(Box::new(self))
    }

    // ── Evaluation ──

    /// Evaluate this spec against a candidate.
    pub fn is_satisfied_by(&self, candidate: &Candidate) -> bool {
        match self {
            Self::FieldEquals { field, value } => {
                candidate.get_field(field).map(|v| v == value).unwrap_or(false)
            }
            Self::FieldCompare { field, op, value } => {
                candidate.get_field(field).map(|v| op.compare_str(v, value)).unwrap_or(false)
            }
            Self::NumericCompare { field, op, value } => {
                candidate.get_field_as_i64(field).map(|v| op.compare_i64(v, *value)).unwrap_or(false)
            }
            Self::Contains { field, substring } => {
                candidate.get_field(field).map(|v| v.contains(substring.as_str())).unwrap_or(false)
            }
            Self::StartsWith { field, prefix } => {
                candidate.get_field(field).map(|v| v.starts_with(prefix.as_str())).unwrap_or(false)
            }
            Self::EndsWith { field, suffix } => {
                candidate.get_field(field).map(|v| v.ends_with(suffix.as_str())).unwrap_or(false)
            }
            Self::FieldExists(field) => candidate.fields.contains_key(field),
            Self::In { field, values } => {
                candidate.get_field(field).map(|v| values.iter().any(|val| val == v)).unwrap_or(false)
            }
            Self::And(specs) => specs.iter().all(|s| s.is_satisfied_by(candidate)),
            Self::Or(specs) => specs.iter().any(|s| s.is_satisfied_by(candidate)),
            Self::Not(spec) => !spec.is_satisfied_by(candidate),
            Self::True => true,
            Self::False => false,
        }
    }

    /// Count satisfied candidates.
    pub fn count_satisfied(&self, candidates: &[Candidate]) -> usize {
        candidates.iter().filter(|c| self.is_satisfied_by(c)).count()
    }

    /// Filter candidates that satisfy this spec.
    pub fn filter<'a>(&self, candidates: &'a [Candidate]) -> Vec<&'a Candidate> {
        candidates.iter().filter(|c| self.is_satisfied_by(c)).collect()
    }

    /// Find the first satisfying candidate.
    pub fn find_first<'a>(&self, candidates: &'a [Candidate]) -> Option<&'a Candidate> {
        candidates.iter().find(|c| self.is_satisfied_by(c))
    }

    /// Whether any candidate satisfies this spec.
    pub fn any_satisfied(&self, candidates: &[Candidate]) -> bool {
        candidates.iter().any(|c| self.is_satisfied_by(c))
    }

    /// Whether all candidates satisfy this spec.
    pub fn all_satisfied(&self, candidates: &[Candidate]) -> bool {
        candidates.iter().all(|c| self.is_satisfied_by(c))
    }

    /// Partition candidates into satisfying and non-satisfying.
    pub fn partition<'a>(&self, candidates: &'a [Candidate]) -> (Vec<&'a Candidate>, Vec<&'a Candidate>) {
        candidates.iter().partition(|c| self.is_satisfied_by(c))
    }
}

// ── Serializable Spec ───────────────────────────────────────────

/// Serialize a spec to JSON.
pub fn serialize_spec(spec: &Spec) -> Result<String, SpecError> {
    serde_json::to_string(spec).map_err(|e| SpecError::SerializationError(e.to_string()))
}

/// Deserialize a spec from JSON.
pub fn deserialize_spec(json: &str) -> Result<Spec, SpecError> {
    serde_json::from_str(json).map_err(|e| SpecError::SerializationError(e.to_string()))
}

// ── ParameterizedSpec ───────────────────────────────────────────

/// A parameterized specification that binds parameters at evaluation time.
#[derive(Debug, Clone)]
pub struct ParameterizedSpec {
    template: Spec,
    parameters: HashMap<String, String>,
}

impl ParameterizedSpec {
    pub fn new(template: Spec) -> Self {
        Self { template, parameters: HashMap::new() }
    }

    pub fn bind(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(name.into(), value.into());
        self
    }

    /// Build the final spec by substituting parameters.
    pub fn build(&self) -> Result<Spec, SpecError> {
        self.substitute(&self.template)
    }

    fn substitute(&self, spec: &Spec) -> Result<Spec, SpecError> {
        match spec {
            Spec::FieldEquals { field, value } => {
                let resolved = self.resolve_value(value)?;
                Ok(Spec::FieldEquals { field: field.clone(), value: resolved })
            }
            Spec::FieldCompare { field, op, value } => {
                let resolved = self.resolve_value(value)?;
                Ok(Spec::FieldCompare { field: field.clone(), op: *op, value: resolved })
            }
            Spec::NumericCompare { field, op, value } => {
                Ok(Spec::NumericCompare { field: field.clone(), op: *op, value: *value })
            }
            Spec::Contains { field, substring } => {
                let resolved = self.resolve_value(substring)?;
                Ok(Spec::Contains { field: field.clone(), substring: resolved })
            }
            Spec::StartsWith { field, prefix } => {
                let resolved = self.resolve_value(prefix)?;
                Ok(Spec::StartsWith { field: field.clone(), prefix: resolved })
            }
            Spec::EndsWith { field, suffix } => {
                let resolved = self.resolve_value(suffix)?;
                Ok(Spec::EndsWith { field: field.clone(), suffix: resolved })
            }
            Spec::FieldExists(f) => Ok(Spec::FieldExists(f.clone())),
            Spec::In { field, values } => {
                let resolved: Result<Vec<String>, _> =
                    values.iter().map(|v| self.resolve_value(v)).collect();
                Ok(Spec::In { field: field.clone(), values: resolved? })
            }
            Spec::And(specs) => {
                let resolved: Result<Vec<Spec>, _> =
                    specs.iter().map(|s| self.substitute(s)).collect();
                Ok(Spec::And(resolved?))
            }
            Spec::Or(specs) => {
                let resolved: Result<Vec<Spec>, _> =
                    specs.iter().map(|s| self.substitute(s)).collect();
                Ok(Spec::Or(resolved?))
            }
            Spec::Not(inner) => {
                let resolved = self.substitute(inner)?;
                Ok(Spec::Not(Box::new(resolved)))
            }
            Spec::True => Ok(Spec::True),
            Spec::False => Ok(Spec::False),
        }
    }

    fn resolve_value(&self, value: &str) -> Result<String, SpecError> {
        if let Some(stripped) = value.strip_prefix(':') {
            // Parameter reference.
            self.parameters.get(stripped)
                .cloned()
                .ok_or_else(|| SpecError::MissingParameter(stripped.to_string()))
        } else {
            Ok(value.to_string())
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candidates() -> Vec<Candidate> {
        vec![
            Candidate::new("c1")
                .with_field("name", "Alice")
                .with_field("age", "30")
                .with_field("city", "New York"),
            Candidate::new("c2")
                .with_field("name", "Bob")
                .with_field("age", "25")
                .with_field("city", "San Francisco"),
            Candidate::new("c3")
                .with_field("name", "Charlie")
                .with_field("age", "35")
                .with_field("city", "New York"),
        ]
    }

    #[test]
    fn test_field_equals() {
        let spec = Spec::field_equals("name", "Alice");
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0]));
        assert!(!spec.is_satisfied_by(&candidates[1]));
    }

    #[test]
    fn test_numeric_compare() {
        let spec = Spec::numeric_compare("age", ComparisonOp::GreaterThan, 28);
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0])); // 30
        assert!(!spec.is_satisfied_by(&candidates[1])); // 25
        assert!(spec.is_satisfied_by(&candidates[2])); // 35
    }

    #[test]
    fn test_contains() {
        let spec = Spec::contains("city", "New");
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0]));
        assert!(!spec.is_satisfied_by(&candidates[1]));
    }

    #[test]
    fn test_starts_with() {
        let spec = Spec::starts_with("name", "Al");
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0]));
        assert!(!spec.is_satisfied_by(&candidates[1]));
    }

    #[test]
    fn test_ends_with() {
        let spec = Spec::ends_with("name", "lie");
        let candidates = sample_candidates();
        assert!(!spec.is_satisfied_by(&candidates[0]));
        assert!(!spec.is_satisfied_by(&candidates[1]));
        assert!(spec.is_satisfied_by(&candidates[2]));
    }

    #[test]
    fn test_field_exists() {
        let spec = Spec::field_exists("city");
        let c = Candidate::new("c1").with_field("name", "test");
        assert!(!spec.is_satisfied_by(&c));
        let c2 = Candidate::new("c2").with_field("city", "LA");
        assert!(spec.is_satisfied_by(&c2));
    }

    #[test]
    fn test_in_spec() {
        let spec = Spec::is_in("name", vec!["Alice".to_string(), "Bob".to_string()]);
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0]));
        assert!(spec.is_satisfied_by(&candidates[1]));
        assert!(!spec.is_satisfied_by(&candidates[2]));
    }

    #[test]
    fn test_and_composition() {
        let spec = Spec::field_equals("city", "New York")
            .and(Spec::numeric_compare("age", ComparisonOp::GreaterThan, 28));
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0])); // NYC, 30
        assert!(!spec.is_satisfied_by(&candidates[1])); // SF, 25
        assert!(spec.is_satisfied_by(&candidates[2])); // NYC, 35
    }

    #[test]
    fn test_or_composition() {
        let spec = Spec::field_equals("name", "Alice")
            .or(Spec::field_equals("name", "Bob"));
        let candidates = sample_candidates();
        assert!(spec.is_satisfied_by(&candidates[0]));
        assert!(spec.is_satisfied_by(&candidates[1]));
        assert!(!spec.is_satisfied_by(&candidates[2]));
    }

    #[test]
    fn test_not_composition() {
        let spec = Spec::field_equals("name", "Alice").not();
        let candidates = sample_candidates();
        assert!(!spec.is_satisfied_by(&candidates[0]));
        assert!(spec.is_satisfied_by(&candidates[1]));
    }

    #[test]
    fn test_filter() {
        let spec = Spec::field_equals("city", "New York");
        let candidates = sample_candidates();
        let filtered = spec.filter(&candidates);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_count_satisfied() {
        let spec = Spec::numeric_compare("age", ComparisonOp::GreaterThanOrEqual, 30);
        let candidates = sample_candidates();
        assert_eq!(spec.count_satisfied(&candidates), 2);
    }

    #[test]
    fn test_find_first() {
        let spec = Spec::field_equals("name", "Bob");
        let candidates = sample_candidates();
        let found = spec.find_first(&candidates).unwrap();
        assert_eq!(found.id, "c2");
    }

    #[test]
    fn test_any_and_all() {
        let spec = Spec::field_exists("name");
        let candidates = sample_candidates();
        assert!(spec.all_satisfied(&candidates));
        assert!(spec.any_satisfied(&candidates));

        let spec2 = Spec::field_equals("name", "Nobody");
        assert!(!spec2.any_satisfied(&candidates));
    }

    #[test]
    fn test_partition() {
        let spec = Spec::field_equals("city", "New York");
        let candidates = sample_candidates();
        let (matching, non_matching) = spec.partition(&candidates);
        assert_eq!(matching.len(), 2);
        assert_eq!(non_matching.len(), 1);
    }

    #[test]
    fn test_true_false_specs() {
        let c = Candidate::new("c1");
        assert!(Spec::True.is_satisfied_by(&c));
        assert!(!Spec::False.is_satisfied_by(&c));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let spec = Spec::field_equals("name", "Alice")
            .and(Spec::numeric_compare("age", ComparisonOp::GreaterThan, 20));
        let json = serialize_spec(&spec).unwrap();
        let deserialized = deserialize_spec(&json).unwrap();
        let c = Candidate::new("c1").with_field("name", "Alice").with_field("age", "30");
        assert!(deserialized.is_satisfied_by(&c));
    }

    #[test]
    fn test_parameterized_spec() {
        let template = Spec::field_equals("status", ":target_status");
        let pspec = ParameterizedSpec::new(template)
            .bind("target_status", "active");
        let built = pspec.build().unwrap();
        let c = Candidate::new("c1").with_field("status", "active");
        assert!(built.is_satisfied_by(&c));
    }

    #[test]
    fn test_parameterized_spec_missing() {
        let template = Spec::field_equals("status", ":missing");
        let pspec = ParameterizedSpec::new(template);
        let result = pspec.build();
        assert!(matches!(result, Err(SpecError::MissingParameter(_))));
    }

    #[test]
    fn test_comparison_operators() {
        assert!(ComparisonOp::Equal.compare_i64(5, 5));
        assert!(!ComparisonOp::Equal.compare_i64(5, 6));
        assert!(ComparisonOp::NotEqual.compare_i64(5, 6));
        assert!(ComparisonOp::LessThan.compare_i64(5, 6));
        assert!(ComparisonOp::LessThanOrEqual.compare_i64(5, 5));
        assert!(ComparisonOp::GreaterThan.compare_i64(6, 5));
        assert!(ComparisonOp::GreaterThanOrEqual.compare_i64(5, 5));
    }

    #[test]
    fn test_and_flattening() {
        // Chaining .and() should flatten into a single And.
        let spec = Spec::field_equals("a", "1")
            .and(Spec::field_equals("b", "2"))
            .and(Spec::field_equals("c", "3"));
        match spec {
            Spec::And(specs) => assert_eq!(specs.len(), 3),
            _ => panic!("expected And variant"),
        }
    }

    #[test]
    fn test_missing_field_returns_false() {
        let spec = Spec::field_equals("nonexistent", "value");
        let c = Candidate::new("c1");
        assert!(!spec.is_satisfied_by(&c));
    }
}
