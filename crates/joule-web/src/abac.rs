//! Attribute-based access control (ABAC) — subject/resource/action/environment
//! attributes, policy rules with conditions, policy evaluation (permit/deny/not-applicable),
//! and policy combining algorithms (deny-overrides/permit-overrides/first-applicable).
//!
//! Replaces Open Policy Agent (OPA), Cedar, and casbin with a pure-Rust ABAC engine
//! supporting rich attribute matching and composable policy sets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// ABAC engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbacError {
    /// Policy not found.
    PolicyNotFound(String),
    /// Duplicate policy ID.
    DuplicatePolicy(String),
    /// Invalid condition expression.
    InvalidCondition(String),
    /// Attribute missing from request context.
    MissingAttribute { category: String, key: String },
    /// Policy set not found.
    PolicySetNotFound(String),
    /// Circular policy reference.
    CircularReference(String),
}

impl fmt::Display for AbacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyNotFound(id) => write!(f, "policy not found: {id}"),
            Self::DuplicatePolicy(id) => write!(f, "duplicate policy: {id}"),
            Self::InvalidCondition(msg) => write!(f, "invalid condition: {msg}"),
            Self::MissingAttribute { category, key } => {
                write!(f, "missing {category} attribute: {key}")
            }
            Self::PolicySetNotFound(id) => write!(f, "policy set not found: {id}"),
            Self::CircularReference(id) => write!(f, "circular policy reference: {id}"),
        }
    }
}

impl std::error::Error for AbacError {}

// ── Types ──────────────────────────────────────────────────────

/// The decision from evaluating a policy against a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    /// Access is permitted.
    Permit,
    /// Access is denied.
    Deny,
    /// Policy does not apply to this request.
    NotApplicable,
    /// Policy evaluation encountered an error.
    Indeterminate,
}

impl Decision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Permit => "permit",
            Self::Deny => "deny",
            Self::NotApplicable => "not_applicable",
            Self::Indeterminate => "indeterminate",
        }
    }

    pub fn is_permit(&self) -> bool {
        matches!(self, Self::Permit)
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny)
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// How to combine multiple policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombiningAlgorithm {
    /// If any policy denies, the result is deny.
    DenyOverrides,
    /// If any policy permits, the result is permit.
    PermitOverrides,
    /// Use the result of the first applicable policy.
    FirstApplicable,
    /// All policies must permit for the result to be permit.
    UnanimousPermit,
}

impl CombiningAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DenyOverrides => "deny_overrides",
            Self::PermitOverrides => "permit_overrides",
            Self::FirstApplicable => "first_applicable",
            Self::UnanimousPermit => "unanimous_permit",
        }
    }
}

/// A comparison operator for conditions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    /// Attribute equals a value.
    Equals,
    /// Attribute does not equal a value.
    NotEquals,
    /// Attribute contains a substring (string) or element (list).
    Contains,
    /// Attribute starts with a prefix.
    StartsWith,
    /// Attribute ends with a suffix.
    EndsWith,
    /// Attribute is in a set of values.
    In,
    /// Numeric greater-than.
    GreaterThan,
    /// Numeric less-than.
    LessThan,
    /// Numeric greater-than-or-equal.
    GreaterThanOrEqual,
    /// Numeric less-than-or-equal.
    LessThanOrEqual,
    /// Attribute exists (is present).
    Exists,
    /// Glob/wildcard pattern match.
    Matches,
}

/// A single condition in a policy rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Condition {
    /// Which attribute category: "subject", "resource", "action", "environment".
    pub category: String,
    /// The attribute key within the category.
    pub key: String,
    /// The comparison operator.
    pub operator: Operator,
    /// The value(s) to compare against.
    pub value: Vec<String>,
}

impl Condition {
    /// Create a simple equals condition.
    pub fn equals(category: &str, key: &str, value: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::Equals,
            value: vec![value.to_string()],
        }
    }

    /// Create an "in" condition.
    pub fn is_in(category: &str, key: &str, values: &[&str]) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::In,
            value: values.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create an exists condition.
    pub fn exists(category: &str, key: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::Exists,
            value: Vec::new(),
        }
    }

    /// Create a contains condition.
    pub fn contains(category: &str, key: &str, substring: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::Contains,
            value: vec![substring.to_string()],
        }
    }

    /// Create a starts-with condition.
    pub fn starts_with(category: &str, key: &str, prefix: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::StartsWith,
            value: vec![prefix.to_string()],
        }
    }

    /// Create a greater-than condition.
    pub fn greater_than(category: &str, key: &str, value: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::GreaterThan,
            value: vec![value.to_string()],
        }
    }

    /// Create a less-than condition.
    pub fn less_than(category: &str, key: &str, value: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::LessThan,
            value: vec![value.to_string()],
        }
    }

    /// Create a matches (glob) condition.
    pub fn matches(category: &str, key: &str, pattern: &str) -> Self {
        Self {
            category: category.to_string(),
            key: key.to_string(),
            operator: Operator::Matches,
            value: vec![pattern.to_string()],
        }
    }

    /// Evaluate this condition against an attribute set.
    pub fn evaluate(&self, attrs: &AttributeSet) -> bool {
        let map = match self.category.as_str() {
            "subject" => &attrs.subject,
            "resource" => &attrs.resource,
            "action" => &attrs.action,
            "environment" => &attrs.environment,
            _ => return false,
        };

        match self.operator {
            Operator::Exists => map.contains_key(&self.key),
            _ => {
                let attr_val = match map.get(&self.key) {
                    Some(v) => v,
                    None => return false,
                };
                match self.operator {
                    Operator::Equals => {
                        self.value.first().map_or(false, |v| v == attr_val)
                    }
                    Operator::NotEquals => {
                        self.value.first().map_or(true, |v| v != attr_val)
                    }
                    Operator::Contains => {
                        self.value.first().map_or(false, |v| attr_val.contains(v.as_str()))
                    }
                    Operator::StartsWith => {
                        self.value.first().map_or(false, |v| attr_val.starts_with(v.as_str()))
                    }
                    Operator::EndsWith => {
                        self.value.first().map_or(false, |v| attr_val.ends_with(v.as_str()))
                    }
                    Operator::In => self.value.contains(attr_val),
                    Operator::GreaterThan => {
                        self.value.first().and_then(|v| {
                            let a: f64 = attr_val.parse().ok()?;
                            let b: f64 = v.parse().ok()?;
                            Some(a > b)
                        }).unwrap_or(false)
                    }
                    Operator::LessThan => {
                        self.value.first().and_then(|v| {
                            let a: f64 = attr_val.parse().ok()?;
                            let b: f64 = v.parse().ok()?;
                            Some(a < b)
                        }).unwrap_or(false)
                    }
                    Operator::GreaterThanOrEqual => {
                        self.value.first().and_then(|v| {
                            let a: f64 = attr_val.parse().ok()?;
                            let b: f64 = v.parse().ok()?;
                            Some(a >= b)
                        }).unwrap_or(false)
                    }
                    Operator::LessThanOrEqual => {
                        self.value.first().and_then(|v| {
                            let a: f64 = attr_val.parse().ok()?;
                            let b: f64 = v.parse().ok()?;
                            Some(a <= b)
                        }).unwrap_or(false)
                    }
                    Operator::Matches => {
                        self.value.first().map_or(false, |pat| glob_match(pat, attr_val))
                    }
                    Operator::Exists => unreachable!(),
                }
            }
        }
    }
}

/// Simple glob matching: `*` matches any sequence of characters,
/// `?` matches exactly one character, other chars match literally.
fn glob_match(pattern: &str, value: &str) -> bool {
    let p_bytes = pattern.as_bytes();
    let v_bytes = value.as_bytes();
    let pn = p_bytes.len();
    let vn = v_bytes.len();

    // dp[i][j] = pattern[..i] matches value[..j]
    let mut prev = vec![false; vn + 1];
    let mut curr = vec![false; vn + 1];
    prev[0] = true;

    for i in 1..=pn {
        curr[0] = if p_bytes[i - 1] == b'*' { prev[0] } else { false };
        for j in 1..=vn {
            if p_bytes[i - 1] == b'*' {
                // * can match zero chars (prev[j]) or one more char (curr[j-1])
                curr[j] = prev[j] || curr[j - 1];
            } else if p_bytes[i - 1] == b'?' || p_bytes[i - 1] == v_bytes[j - 1] {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = false;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        for x in curr.iter_mut() {
            *x = false;
        }
    }
    prev[vn]
}

/// A set of attributes for an ABAC request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributeSet {
    /// Subject attributes (e.g., role, department, clearance).
    pub subject: HashMap<String, String>,
    /// Resource attributes (e.g., type, owner, classification).
    pub resource: HashMap<String, String>,
    /// Action attributes (e.g., type, method).
    pub action: HashMap<String, String>,
    /// Environment attributes (e.g., time, ip, location).
    pub environment: HashMap<String, String>,
}

impl AttributeSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_subject(mut self, key: &str, value: &str) -> Self {
        self.subject.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_resource(mut self, key: &str, value: &str) -> Self {
        self.resource.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_action(mut self, key: &str, value: &str) -> Self {
        self.action.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_environment(mut self, key: &str, value: &str) -> Self {
        self.environment.insert(key.to_string(), value.to_string());
        self
    }

    /// Get an attribute from any category.
    pub fn get(&self, category: &str, key: &str) -> Option<&str> {
        let map = match category {
            "subject" => &self.subject,
            "resource" => &self.resource,
            "action" => &self.action,
            "environment" => &self.environment,
            _ => return None,
        };
        map.get(key).map(|s| s.as_str())
    }
}

/// A policy rule: a set of conditions that must hold, plus the decision if they do.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Unique rule ID.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// The target attributes this rule applies to (all must match for rule to be applicable).
    pub target: Vec<Condition>,
    /// Additional conditions that must be true for the effect to apply.
    pub conditions: Vec<Condition>,
    /// The effect if target matches and conditions hold.
    pub effect: Decision,
    /// Priority (lower number = higher priority).
    pub priority: u32,
}

impl PolicyRule {
    pub fn new(id: &str, effect: Decision) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            target: Vec::new(),
            conditions: Vec::new(),
            effect,
            priority: 100,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_target(mut self, condition: Condition) -> Self {
        self.target.push(condition);
        self
    }

    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Evaluate this rule against an attribute set.
    /// Returns the effect if the rule applies, or NotApplicable.
    pub fn evaluate(&self, attrs: &AttributeSet) -> Decision {
        // Check target: all target conditions must match for the rule to be applicable.
        for cond in &self.target {
            if !cond.evaluate(attrs) {
                return Decision::NotApplicable;
            }
        }
        // Check conditions: all must hold for the effect to be returned.
        for cond in &self.conditions {
            if !cond.evaluate(attrs) {
                return Decision::NotApplicable;
            }
        }
        self.effect
    }
}

/// A named policy containing multiple rules and a combining algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Unique policy ID.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// The combining algorithm for rules within this policy.
    pub combining: CombiningAlgorithm,
    /// The rules in evaluation order.
    pub rules: Vec<PolicyRule>,
    /// Version number.
    pub version: u32,
    /// Whether this policy is enabled.
    pub enabled: bool,
}

impl Policy {
    pub fn new(id: &str, combining: CombiningAlgorithm) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            combining,
            rules: Vec::new(),
            version: 1,
            enabled: true,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_rule(mut self, rule: PolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.rules.push(rule);
    }

    /// Evaluate this policy against an attribute set.
    pub fn evaluate(&self, attrs: &AttributeSet) -> Decision {
        if !self.enabled {
            return Decision::NotApplicable;
        }
        let decisions: Vec<Decision> = self.rules.iter().map(|r| r.evaluate(attrs)).collect();
        combine(&decisions, self.combining)
    }
}

/// A policy set containing multiple policies, also with a combining algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySet {
    /// Unique policy set ID.
    pub id: String,
    /// Description.
    pub description: String,
    /// The combining algorithm for policies in this set.
    pub combining: CombiningAlgorithm,
    /// The policies in this set.
    pub policies: Vec<Policy>,
}

impl PolicySet {
    pub fn new(id: &str, combining: CombiningAlgorithm) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            combining,
            policies: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_policy(mut self, policy: Policy) -> Self {
        self.policies.push(policy);
        self
    }

    pub fn add_policy(&mut self, policy: Policy) {
        self.policies.push(policy);
    }

    /// Evaluate the policy set against an attribute set.
    pub fn evaluate(&self, attrs: &AttributeSet) -> Decision {
        let decisions: Vec<Decision> = self.policies.iter().map(|p| p.evaluate(attrs)).collect();
        combine(&decisions, self.combining)
    }
}

/// Combine multiple decisions using the given algorithm.
pub fn combine(decisions: &[Decision], algorithm: CombiningAlgorithm) -> Decision {
    match algorithm {
        CombiningAlgorithm::DenyOverrides => {
            let mut has_permit = false;
            let mut has_indeterminate = false;
            for d in decisions {
                match d {
                    Decision::Deny => return Decision::Deny,
                    Decision::Permit => has_permit = true,
                    Decision::Indeterminate => has_indeterminate = true,
                    Decision::NotApplicable => {}
                }
            }
            if has_indeterminate {
                Decision::Indeterminate
            } else if has_permit {
                Decision::Permit
            } else {
                Decision::NotApplicable
            }
        }
        CombiningAlgorithm::PermitOverrides => {
            let mut has_deny = false;
            let mut has_indeterminate = false;
            for d in decisions {
                match d {
                    Decision::Permit => return Decision::Permit,
                    Decision::Deny => has_deny = true,
                    Decision::Indeterminate => has_indeterminate = true,
                    Decision::NotApplicable => {}
                }
            }
            if has_indeterminate {
                Decision::Indeterminate
            } else if has_deny {
                Decision::Deny
            } else {
                Decision::NotApplicable
            }
        }
        CombiningAlgorithm::FirstApplicable => {
            for d in decisions {
                match d {
                    Decision::Permit | Decision::Deny => return *d,
                    Decision::Indeterminate => return Decision::Indeterminate,
                    Decision::NotApplicable => {}
                }
            }
            Decision::NotApplicable
        }
        CombiningAlgorithm::UnanimousPermit => {
            let mut found_applicable = false;
            for d in decisions {
                match d {
                    Decision::Deny => return Decision::Deny,
                    Decision::Indeterminate => return Decision::Indeterminate,
                    Decision::Permit => found_applicable = true,
                    Decision::NotApplicable => {}
                }
            }
            if found_applicable {
                Decision::Permit
            } else {
                Decision::NotApplicable
            }
        }
    }
}

/// The ABAC engine: manages policies and evaluates access requests.
pub struct AbacEngine {
    policies: Vec<Policy>,
    policy_sets: Vec<PolicySet>,
    /// Default decision when no policy applies.
    pub default_decision: Decision,
    /// Top-level combining algorithm.
    pub combining: CombiningAlgorithm,
    /// Evaluation history for auditing.
    history: Vec<EvaluationRecord>,
}

/// Record of a policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRecord {
    /// The attributes that were evaluated.
    pub attributes: AttributeSet,
    /// The final decision.
    pub decision: Decision,
    /// Decisions from individual policies (policy_id -> decision).
    pub policy_decisions: Vec<(String, Decision)>,
    /// Timestamp (epoch millis).
    pub timestamp_ms: u64,
}

impl AbacEngine {
    /// Create a new engine with a default deny-overrides combining algorithm.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            policy_sets: Vec::new(),
            default_decision: Decision::Deny,
            combining: CombiningAlgorithm::DenyOverrides,
            history: Vec::new(),
        }
    }

    /// Add a policy.
    pub fn add_policy(&mut self, policy: Policy) -> Result<(), AbacError> {
        if self.policies.iter().any(|p| p.id == policy.id) {
            return Err(AbacError::DuplicatePolicy(policy.id));
        }
        self.policies.push(policy);
        Ok(())
    }

    /// Remove a policy by ID.
    pub fn remove_policy(&mut self, id: &str) -> Result<Policy, AbacError> {
        let idx = self
            .policies
            .iter()
            .position(|p| p.id == id)
            .ok_or_else(|| AbacError::PolicyNotFound(id.to_string()))?;
        Ok(self.policies.remove(idx))
    }

    /// Get a policy by ID.
    pub fn get_policy(&self, id: &str) -> Option<&Policy> {
        self.policies.iter().find(|p| p.id == id)
    }

    /// Get a mutable reference to a policy.
    pub fn get_policy_mut(&mut self, id: &str) -> Option<&mut Policy> {
        self.policies.iter_mut().find(|p| p.id == id)
    }

    /// Add a policy set.
    pub fn add_policy_set(&mut self, ps: PolicySet) -> Result<(), AbacError> {
        if self.policy_sets.iter().any(|p| p.id == ps.id) {
            return Err(AbacError::DuplicatePolicy(ps.id));
        }
        self.policy_sets.push(ps);
        Ok(())
    }

    /// Evaluate an access request.
    pub fn evaluate(&mut self, attrs: &AttributeSet, timestamp_ms: u64) -> Decision {
        let mut all_decisions: Vec<(String, Decision)> = Vec::new();

        for policy in &self.policies {
            let d = policy.evaluate(attrs);
            all_decisions.push((policy.id.clone(), d));
        }
        for ps in &self.policy_sets {
            let d = ps.evaluate(attrs);
            all_decisions.push((ps.id.clone(), d));
        }

        let decisions: Vec<Decision> = all_decisions.iter().map(|(_, d)| *d).collect();
        let final_decision = if decisions.is_empty() {
            self.default_decision
        } else {
            let combined = combine(&decisions, self.combining);
            if combined == Decision::NotApplicable {
                self.default_decision
            } else {
                combined
            }
        };

        self.history.push(EvaluationRecord {
            attributes: attrs.clone(),
            decision: final_decision,
            policy_decisions: all_decisions,
            timestamp_ms,
        });

        final_decision
    }

    /// Evaluate without recording history (dry-run).
    pub fn evaluate_dry_run(&self, attrs: &AttributeSet) -> Decision {
        let mut decisions: Vec<Decision> = Vec::new();

        for policy in &self.policies {
            decisions.push(policy.evaluate(attrs));
        }
        for ps in &self.policy_sets {
            decisions.push(ps.evaluate(attrs));
        }

        if decisions.is_empty() {
            self.default_decision
        } else {
            let combined = combine(&decisions, self.combining);
            if combined == Decision::NotApplicable {
                self.default_decision
            } else {
                combined
            }
        }
    }

    /// Get evaluation history.
    pub fn history(&self) -> &[EvaluationRecord] {
        &self.history
    }

    /// Clear evaluation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Number of policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    /// Number of policy sets.
    pub fn policy_set_count(&self) -> usize {
        self.policy_sets.len()
    }

    /// List all policy IDs.
    pub fn policy_ids(&self) -> Vec<&str> {
        self.policies.iter().map(|p| p.id.as_str()).collect()
    }

    /// Enable or disable a policy.
    pub fn set_policy_enabled(&mut self, id: &str, enabled: bool) -> Result<(), AbacError> {
        let policy = self
            .policies
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| AbacError::PolicyNotFound(id.to_string()))?;
        policy.enabled = enabled;
        Ok(())
    }
}

impl Default for AbacEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admin_request() -> AttributeSet {
        AttributeSet::new()
            .with_subject("role", "admin")
            .with_subject("department", "engineering")
            .with_resource("type", "document")
            .with_resource("classification", "internal")
            .with_action("type", "read")
            .with_environment("time", "14:00")
    }

    fn user_request() -> AttributeSet {
        AttributeSet::new()
            .with_subject("role", "user")
            .with_subject("department", "marketing")
            .with_resource("type", "document")
            .with_resource("classification", "public")
            .with_action("type", "read")
    }

    #[test]
    fn test_condition_equals() {
        let cond = Condition::equals("subject", "role", "admin");
        assert!(cond.evaluate(&admin_request()));
        assert!(!cond.evaluate(&user_request()));
    }

    #[test]
    fn test_condition_not_equals() {
        let cond = Condition {
            category: "subject".into(),
            key: "role".into(),
            operator: Operator::NotEquals,
            value: vec!["admin".into()],
        };
        assert!(!cond.evaluate(&admin_request()));
        assert!(cond.evaluate(&user_request()));
    }

    #[test]
    fn test_condition_in() {
        let cond = Condition::is_in("subject", "role", &["admin", "superadmin"]);
        assert!(cond.evaluate(&admin_request()));
        assert!(!cond.evaluate(&user_request()));
    }

    #[test]
    fn test_condition_contains() {
        let cond = Condition::contains("subject", "department", "engine");
        assert!(cond.evaluate(&admin_request()));
        assert!(!cond.evaluate(&user_request()));
    }

    #[test]
    fn test_condition_starts_with() {
        let cond = Condition::starts_with("subject", "department", "eng");
        assert!(cond.evaluate(&admin_request()));
        assert!(!cond.evaluate(&user_request()));
    }

    #[test]
    fn test_condition_exists() {
        let cond = Condition::exists("environment", "time");
        assert!(cond.evaluate(&admin_request()));
        assert!(!cond.evaluate(&user_request())); // user_request has no environment
    }

    #[test]
    fn test_condition_numeric_greater_than() {
        let attrs = AttributeSet::new().with_subject("clearance", "5");
        let cond = Condition::greater_than("subject", "clearance", "3");
        assert!(cond.evaluate(&attrs));
        let cond2 = Condition::greater_than("subject", "clearance", "5");
        assert!(!cond2.evaluate(&attrs));
    }

    #[test]
    fn test_condition_numeric_less_than() {
        let attrs = AttributeSet::new().with_subject("clearance", "2");
        let cond = Condition::less_than("subject", "clearance", "5");
        assert!(cond.evaluate(&attrs));
    }

    #[test]
    fn test_condition_glob_match() {
        let cond = Condition::matches("resource", "path", "/api/*/data");
        let attrs = AttributeSet::new().with_resource("path", "/api/users/data");
        assert!(cond.evaluate(&attrs));
        let attrs2 = AttributeSet::new().with_resource("path", "/api/data");
        assert!(!cond2_evaluate(&cond, &attrs2));
    }

    fn cond2_evaluate(cond: &Condition, attrs: &AttributeSet) -> bool {
        cond.evaluate(attrs)
    }

    #[test]
    fn test_policy_rule_permit() {
        let rule = PolicyRule::new("r1", Decision::Permit)
            .with_target(Condition::equals("subject", "role", "admin"))
            .with_condition(Condition::equals("action", "type", "read"));
        assert_eq!(rule.evaluate(&admin_request()), Decision::Permit);
        assert_eq!(rule.evaluate(&user_request()), Decision::NotApplicable);
    }

    #[test]
    fn test_policy_rule_deny() {
        let rule = PolicyRule::new("r1", Decision::Deny)
            .with_target(Condition::equals("resource", "classification", "secret"));
        let attrs = AttributeSet::new()
            .with_resource("classification", "secret")
            .with_subject("role", "admin");
        assert_eq!(rule.evaluate(&attrs), Decision::Deny);
    }

    #[test]
    fn test_deny_overrides_combining() {
        let decisions = vec![Decision::Permit, Decision::Deny, Decision::Permit];
        assert_eq!(combine(&decisions, CombiningAlgorithm::DenyOverrides), Decision::Deny);
    }

    #[test]
    fn test_permit_overrides_combining() {
        let decisions = vec![Decision::Deny, Decision::NotApplicable, Decision::Permit];
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::PermitOverrides),
            Decision::Permit
        );
    }

    #[test]
    fn test_first_applicable_combining() {
        let decisions = vec![Decision::NotApplicable, Decision::Deny, Decision::Permit];
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::FirstApplicable),
            Decision::Deny
        );
    }

    #[test]
    fn test_unanimous_permit() {
        let d1 = vec![Decision::Permit, Decision::Permit];
        assert_eq!(combine(&d1, CombiningAlgorithm::UnanimousPermit), Decision::Permit);
        let d2 = vec![Decision::Permit, Decision::Deny];
        assert_eq!(combine(&d2, CombiningAlgorithm::UnanimousPermit), Decision::Deny);
    }

    #[test]
    fn test_policy_evaluation() {
        let policy = Policy::new("p1", CombiningAlgorithm::DenyOverrides)
            .with_rule(
                PolicyRule::new("allow-admin-read", Decision::Permit)
                    .with_target(Condition::equals("subject", "role", "admin"))
                    .with_condition(Condition::equals("action", "type", "read")),
            )
            .with_rule(
                PolicyRule::new("deny-secret", Decision::Deny)
                    .with_target(Condition::equals("resource", "classification", "secret")),
            );
        assert_eq!(policy.evaluate(&admin_request()), Decision::Permit);
    }

    #[test]
    fn test_engine_evaluate() {
        let mut engine = AbacEngine::new();
        let policy = Policy::new("p1", CombiningAlgorithm::FirstApplicable)
            .with_rule(
                PolicyRule::new("allow-read", Decision::Permit)
                    .with_target(Condition::equals("action", "type", "read")),
            );
        engine.add_policy(policy).unwrap();
        let result = engine.evaluate(&admin_request(), 1000);
        assert_eq!(result, Decision::Permit);
        assert_eq!(engine.history().len(), 1);
    }

    #[test]
    fn test_engine_default_deny() {
        let mut engine = AbacEngine::new();
        engine.default_decision = Decision::Deny;
        let attrs = AttributeSet::new().with_action("type", "delete");
        let result = engine.evaluate(&attrs, 1000);
        assert_eq!(result, Decision::Deny);
    }

    #[test]
    fn test_engine_dry_run() {
        let mut engine = AbacEngine::new();
        let policy = Policy::new("p1", CombiningAlgorithm::PermitOverrides)
            .with_rule(
                PolicyRule::new("allow-all", Decision::Permit)
                    .with_target(Condition::equals("action", "type", "read")),
            );
        engine.add_policy(policy).unwrap();
        let result = engine.evaluate_dry_run(&admin_request());
        assert_eq!(result, Decision::Permit);
        assert!(engine.history().is_empty());
    }

    #[test]
    fn test_duplicate_policy_error() {
        let mut engine = AbacEngine::new();
        let p1 = Policy::new("p1", CombiningAlgorithm::DenyOverrides);
        let p2 = Policy::new("p1", CombiningAlgorithm::DenyOverrides);
        engine.add_policy(p1).unwrap();
        assert_eq!(
            engine.add_policy(p2),
            Err(AbacError::DuplicatePolicy("p1".into()))
        );
    }

    #[test]
    fn test_remove_policy() {
        let mut engine = AbacEngine::new();
        engine
            .add_policy(Policy::new("p1", CombiningAlgorithm::DenyOverrides))
            .unwrap();
        assert_eq!(engine.policy_count(), 1);
        engine.remove_policy("p1").unwrap();
        assert_eq!(engine.policy_count(), 0);
    }

    #[test]
    fn test_disabled_policy() {
        let mut engine = AbacEngine::new();
        let policy = Policy::new("p1", CombiningAlgorithm::FirstApplicable)
            .with_rule(
                PolicyRule::new("allow-all", Decision::Permit)
                    .with_target(Condition::equals("action", "type", "read")),
            );
        engine.add_policy(policy).unwrap();
        engine.set_policy_enabled("p1", false).unwrap();
        let result = engine.evaluate(&admin_request(), 1000);
        // With policy disabled, no applicable -> default deny
        assert_eq!(result, Decision::Deny);
    }

    #[test]
    fn test_policy_set_evaluation() {
        let p1 = Policy::new("p-allow", CombiningAlgorithm::FirstApplicable)
            .with_rule(
                PolicyRule::new("r1", Decision::Permit)
                    .with_target(Condition::equals("subject", "role", "admin")),
            );
        let p2 = Policy::new("p-deny-secret", CombiningAlgorithm::FirstApplicable)
            .with_rule(
                PolicyRule::new("r2", Decision::Deny)
                    .with_target(Condition::equals("resource", "classification", "secret")),
            );
        let ps = PolicySet::new("ps1", CombiningAlgorithm::DenyOverrides)
            .with_policy(p1)
            .with_policy(p2);

        let attrs = AttributeSet::new()
            .with_subject("role", "admin")
            .with_resource("classification", "secret");
        assert_eq!(ps.evaluate(&attrs), Decision::Deny);
    }

    #[test]
    fn test_glob_matching_patterns() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*.txt", "file.txt"));
        assert!(!glob_match("*.txt", "file.rs"));
        assert!(glob_match("?at", "cat"));
        assert!(glob_match("?at", "bat"));
        assert!(!glob_match("?at", "at"));
        assert!(glob_match("/api/*/items", "/api/v1/items"));
    }

    #[test]
    fn test_all_not_applicable() {
        let decisions = vec![Decision::NotApplicable, Decision::NotApplicable];
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::DenyOverrides),
            Decision::NotApplicable
        );
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::PermitOverrides),
            Decision::NotApplicable
        );
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::FirstApplicable),
            Decision::NotApplicable
        );
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::UnanimousPermit),
            Decision::NotApplicable
        );
    }

    #[test]
    fn test_indeterminate_propagation() {
        let decisions = vec![Decision::Indeterminate, Decision::Permit];
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::DenyOverrides),
            Decision::Indeterminate
        );
        assert_eq!(
            combine(&decisions, CombiningAlgorithm::PermitOverrides),
            Decision::Permit
        );
    }

    #[test]
    fn test_attribute_set_get() {
        let attrs = admin_request();
        assert_eq!(attrs.get("subject", "role"), Some("admin"));
        assert_eq!(attrs.get("resource", "type"), Some("document"));
        assert_eq!(attrs.get("unknown", "key"), None);
    }

    #[test]
    fn test_decision_display() {
        assert_eq!(Decision::Permit.to_string(), "permit");
        assert_eq!(Decision::Deny.to_string(), "deny");
        assert!(Decision::Permit.is_permit());
        assert!(!Decision::Permit.is_deny());
    }

    #[test]
    fn test_engine_clear_history() {
        let mut engine = AbacEngine::new();
        engine.evaluate(&admin_request(), 1000);
        engine.evaluate(&admin_request(), 2000);
        assert_eq!(engine.history().len(), 2);
        engine.clear_history();
        assert!(engine.history().is_empty());
    }

    #[test]
    fn test_condition_ends_with() {
        let cond = Condition {
            category: "resource".into(),
            key: "path".into(),
            operator: Operator::EndsWith,
            value: vec![".json".into()],
        };
        let attrs = AttributeSet::new().with_resource("path", "/data/config.json");
        assert!(cond.evaluate(&attrs));
        let attrs2 = AttributeSet::new().with_resource("path", "/data/config.yaml");
        assert!(!cond.evaluate(&attrs2));
    }

    #[test]
    fn test_condition_missing_attribute() {
        let cond = Condition::equals("subject", "clearance", "top-secret");
        let attrs = AttributeSet::new().with_subject("role", "admin");
        assert!(!cond.evaluate(&attrs));
    }

    #[test]
    fn test_condition_invalid_category() {
        let cond = Condition::equals("invalid_category", "key", "value");
        assert!(!cond.evaluate(&admin_request()));
    }
}
