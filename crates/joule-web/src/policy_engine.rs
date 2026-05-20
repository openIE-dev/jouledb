//! Policy engine — JSON-based policy DSL, condition evaluation, effect (allow/deny),
//! policy sets with combining, obligation/advice actions, policy versioning, and
//! policy simulation (dry-run).
//!
//! Replaces OPA Rego, Cedar, and casbin with a pure-Rust policy engine supporting
//! a JSON rule DSL, versioned policies, and full dry-run simulation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Policy engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyEngineError {
    /// Policy not found.
    PolicyNotFound(String),
    /// Duplicate policy ID.
    DuplicatePolicy(String),
    /// Invalid rule syntax.
    InvalidRule(String),
    /// Condition evaluation error.
    ConditionError(String),
    /// Version conflict.
    VersionConflict { policy_id: String, expected: u64, actual: u64 },
    /// Policy set not found.
    PolicySetNotFound(String),
    /// Invalid JSON in rule definition.
    InvalidJson(String),
}

impl fmt::Display for PolicyEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyNotFound(id) => write!(f, "policy not found: {id}"),
            Self::DuplicatePolicy(id) => write!(f, "duplicate policy: {id}"),
            Self::InvalidRule(msg) => write!(f, "invalid rule: {msg}"),
            Self::ConditionError(msg) => write!(f, "condition error: {msg}"),
            Self::VersionConflict { policy_id, expected, actual } => {
                write!(f, "version conflict for policy {policy_id}: expected {expected}, actual {actual}")
            }
            Self::PolicySetNotFound(id) => write!(f, "policy set not found: {id}"),
            Self::InvalidJson(msg) => write!(f, "invalid JSON: {msg}"),
        }
    }
}

impl std::error::Error for PolicyEngineError {}

// ── Types ──────────────────────────────────────────────────────

/// The effect of a policy rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effect {
    Allow,
    Deny,
}

impl Effect {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Evaluation result from a single rule or policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvalResult {
    /// Rule matched with this effect.
    Matched(Effect),
    /// Rule did not match (conditions not met).
    NotMatched,
    /// Rule had an evaluation error.
    Error(String),
}

impl EvalResult {
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Matched(Effect::Allow))
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Matched(Effect::Deny))
    }
}

/// How to combine multiple rule/policy results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombineStrategy {
    /// First deny wins.
    DenyOverrides,
    /// First allow wins.
    AllowOverrides,
    /// First matched result wins.
    FirstMatch,
    /// All must allow.
    AllMustAllow,
}

/// Comparison operators for conditions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
    StartsWith,
    EndsWith,
    In,
    Exists,
    NotExists,
    Regex,
}

/// A single condition in a rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    /// The field path (dot-separated, e.g. "subject.role").
    pub field: String,
    /// Comparison operator.
    pub op: ConditionOp,
    /// Value to compare against.
    pub value: Value,
}

impl RuleCondition {
    pub fn new(field: &str, op: ConditionOp, value: Value) -> Self {
        Self {
            field: field.to_string(),
            op,
            value,
        }
    }

    /// Evaluate this condition against a context.
    pub fn evaluate(&self, context: &Value) -> bool {
        let field_val = resolve_path(context, &self.field);

        match self.op {
            ConditionOp::Exists => field_val.is_some(),
            ConditionOp::NotExists => field_val.is_none(),
            _ => {
                let field_val = match field_val {
                    Some(v) => v,
                    None => return false,
                };
                match self.op {
                    ConditionOp::Eq => values_equal(field_val, &self.value),
                    ConditionOp::Neq => !values_equal(field_val, &self.value),
                    ConditionOp::Gt => compare_numeric(field_val, &self.value, |a, b| a > b),
                    ConditionOp::Lt => compare_numeric(field_val, &self.value, |a, b| a < b),
                    ConditionOp::Gte => compare_numeric(field_val, &self.value, |a, b| a >= b),
                    ConditionOp::Lte => compare_numeric(field_val, &self.value, |a, b| a <= b),
                    ConditionOp::Contains => {
                        if let (Some(haystack), Some(needle)) = (field_val.as_str(), self.value.as_str()) {
                            haystack.contains(needle)
                        } else if let Some(arr) = field_val.as_array() {
                            arr.contains(&self.value)
                        } else {
                            false
                        }
                    }
                    ConditionOp::StartsWith => {
                        match (field_val.as_str(), self.value.as_str()) {
                            (Some(s), Some(prefix)) => s.starts_with(prefix),
                            _ => false,
                        }
                    }
                    ConditionOp::EndsWith => {
                        match (field_val.as_str(), self.value.as_str()) {
                            (Some(s), Some(suffix)) => s.ends_with(suffix),
                            _ => false,
                        }
                    }
                    ConditionOp::In => {
                        if let Some(arr) = self.value.as_array() {
                            arr.contains(field_val)
                        } else {
                            false
                        }
                    }
                    ConditionOp::Regex => {
                        // Simple regex: just check for basic patterns
                        match (field_val.as_str(), self.value.as_str()) {
                            (Some(s), Some(pattern)) => simple_pattern_match(pattern, s),
                            _ => false,
                        }
                    }
                    ConditionOp::Exists | ConditionOp::NotExists => unreachable!(),
                }
            }
        }
    }
}

/// Resolve a dot-separated path in a JSON value.
fn resolve_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for part in path.split('.') {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Number(a), Value::Number(b)) => {
            a.as_f64().zip(b.as_f64()).map_or(false, |(x, y)| (x - y).abs() < f64::EPSILON)
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        _ => a == b,
    }
}

fn compare_numeric(a: &Value, b: &Value, cmp: fn(f64, f64) -> bool) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => cmp(x, y),
        _ => false,
    }
}

/// Simple pattern matching for the Regex operator.
/// Supports `*` as wildcard and `^`/`$` for anchoring.
fn simple_pattern_match(pattern: &str, value: &str) -> bool {
    let anchored_start = pattern.starts_with('^');
    let anchored_end = pattern.ends_with('$');
    let core = pattern
        .strip_prefix('^').unwrap_or(pattern)
        .strip_suffix('$').unwrap_or_else(|| {
            pattern.strip_prefix('^').unwrap_or(pattern)
        });

    if core.contains('*') {
        // Split on `*` and check each part appears in order
        let parts: Vec<&str> = core.split('*').collect();
        let mut pos = 0;
        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            if let Some(found) = value[pos..].find(part) {
                if i == 0 && anchored_start && found != 0 {
                    return false;
                }
                pos += found + part.len();
            } else {
                return false;
            }
        }
        if anchored_end {
            if let Some(last) = parts.last() {
                if !last.is_empty() {
                    return value.ends_with(last);
                }
            }
        }
        true
    } else if anchored_start && anchored_end {
        value == core
    } else if anchored_start {
        value.starts_with(core)
    } else if anchored_end {
        value.ends_with(core)
    } else {
        value.contains(core)
    }
}

/// An obligation or advice action to execute when a rule matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obligation {
    /// Unique ID.
    pub id: String,
    /// The action type (e.g., "log", "notify", "audit").
    pub action: String,
    /// Parameters for the action.
    pub params: HashMap<String, String>,
    /// Whether this is mandatory (obligation) or optional (advice).
    pub mandatory: bool,
}

impl Obligation {
    pub fn new(id: &str, action: &str, mandatory: bool) -> Self {
        Self {
            id: id.to_string(),
            action: action.to_string(),
            params: HashMap::new(),
            mandatory,
        }
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }
}

/// A single policy rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Unique rule ID.
    pub id: String,
    /// Description.
    pub description: String,
    /// Conditions that must all be true.
    pub conditions: Vec<RuleCondition>,
    /// Effect when conditions match.
    pub effect: Effect,
    /// Priority (lower = higher priority).
    pub priority: u32,
    /// Obligations to fulfill when this rule matches.
    pub obligations: Vec<Obligation>,
}

impl Rule {
    pub fn new(id: &str, effect: Effect) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            conditions: Vec::new(),
            effect,
            priority: 100,
            obligations: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_condition(mut self, condition: RuleCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_obligation(mut self, obligation: Obligation) -> Self {
        self.obligations.push(obligation);
        self
    }

    /// Evaluate this rule against a context.
    pub fn evaluate(&self, context: &Value) -> EvalResult {
        for cond in &self.conditions {
            if !cond.evaluate(context) {
                return EvalResult::NotMatched;
            }
        }
        EvalResult::Matched(self.effect)
    }

    /// Create a rule from a JSON definition.
    pub fn from_json(json: &Value) -> Result<Self, PolicyEngineError> {
        let id = json.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PolicyEngineError::InvalidRule("missing 'id'".into()))?;

        let effect_str = json.get("effect")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PolicyEngineError::InvalidRule("missing 'effect'".into()))?;

        let effect = match effect_str {
            "allow" => Effect::Allow,
            "deny" => Effect::Deny,
            other => return Err(PolicyEngineError::InvalidRule(format!("unknown effect: {other}"))),
        };

        let mut rule = Rule::new(id, effect);

        if let Some(desc) = json.get("description").and_then(|v| v.as_str()) {
            rule.description = desc.to_string();
        }

        if let Some(priority) = json.get("priority").and_then(|v| v.as_u64()) {
            rule.priority = priority as u32;
        }

        if let Some(conditions) = json.get("conditions").and_then(|v| v.as_array()) {
            for cond_json in conditions {
                let field = cond_json.get("field")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| PolicyEngineError::InvalidRule("condition missing 'field'".into()))?;
                let op_str = cond_json.get("op")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| PolicyEngineError::InvalidRule("condition missing 'op'".into()))?;
                let op = parse_op(op_str)?;
                let value = cond_json.get("value").cloned().unwrap_or(Value::Null);
                rule.conditions.push(RuleCondition::new(field, op, value));
            }
        }

        Ok(rule)
    }
}

fn parse_op(s: &str) -> Result<ConditionOp, PolicyEngineError> {
    match s {
        "eq" | "==" => Ok(ConditionOp::Eq),
        "neq" | "!=" => Ok(ConditionOp::Neq),
        "gt" | ">" => Ok(ConditionOp::Gt),
        "lt" | "<" => Ok(ConditionOp::Lt),
        "gte" | ">=" => Ok(ConditionOp::Gte),
        "lte" | "<=" => Ok(ConditionOp::Lte),
        "contains" => Ok(ConditionOp::Contains),
        "starts_with" => Ok(ConditionOp::StartsWith),
        "ends_with" => Ok(ConditionOp::EndsWith),
        "in" => Ok(ConditionOp::In),
        "exists" => Ok(ConditionOp::Exists),
        "not_exists" => Ok(ConditionOp::NotExists),
        "regex" | "matches" => Ok(ConditionOp::Regex),
        other => Err(PolicyEngineError::InvalidRule(format!("unknown operator: {other}"))),
    }
}

/// A versioned policy containing rules and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Unique ID.
    pub id: String,
    /// Description.
    pub description: String,
    /// Version number.
    pub version: u64,
    /// Rules in this policy.
    pub rules: Vec<Rule>,
    /// Combining strategy for rules.
    pub combine: CombineStrategy,
    /// Whether the policy is enabled.
    pub enabled: bool,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Creation timestamp.
    pub created_at: Option<DateTime<Utc>>,
    /// Last modified timestamp.
    pub updated_at: Option<DateTime<Utc>>,
}

impl Policy {
    pub fn new(id: &str, combine: CombineStrategy) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            version: 1,
            rules: Vec::new(),
            combine,
            enabled: true,
            tags: Vec::new(),
            created_at: None,
            updated_at: None,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Bump the version.
    pub fn bump_version(&mut self) {
        self.version += 1;
        self.updated_at = Some(Utc::now());
    }

    /// Evaluate this policy against a context.
    pub fn evaluate(&self, context: &Value) -> PolicyResult {
        if !self.enabled {
            return PolicyResult {
                effect: None,
                matched_rules: Vec::new(),
                obligations: Vec::new(),
            };
        }

        let mut matched_rules: Vec<(String, Effect)> = Vec::new();
        let mut obligations: Vec<Obligation> = Vec::new();
        let mut sorted_rules: Vec<&Rule> = self.rules.iter().collect();
        sorted_rules.sort_by_key(|r| r.priority);

        for rule in &sorted_rules {
            match rule.evaluate(context) {
                EvalResult::Matched(effect) => {
                    matched_rules.push((rule.id.clone(), effect));
                    obligations.extend(rule.obligations.iter().cloned());
                }
                EvalResult::NotMatched | EvalResult::Error(_) => {}
            }
        }

        let effects: Vec<Effect> = matched_rules.iter().map(|(_, e)| *e).collect();
        let final_effect = combine_effects(&effects, self.combine);

        PolicyResult {
            effect: final_effect,
            matched_rules,
            obligations,
        }
    }

    /// Parse a policy from JSON.
    pub fn from_json(json: &Value) -> Result<Self, PolicyEngineError> {
        let id = json.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PolicyEngineError::InvalidJson("missing 'id'".into()))?;

        let combine_str = json.get("combine")
            .and_then(|v| v.as_str())
            .unwrap_or("deny_overrides");

        let combine = match combine_str {
            "deny_overrides" => CombineStrategy::DenyOverrides,
            "allow_overrides" => CombineStrategy::AllowOverrides,
            "first_match" => CombineStrategy::FirstMatch,
            "all_must_allow" => CombineStrategy::AllMustAllow,
            other => {
                return Err(PolicyEngineError::InvalidJson(
                    format!("unknown combine strategy: {other}"),
                ));
            }
        };

        let mut policy = Policy::new(id, combine);

        if let Some(desc) = json.get("description").and_then(|v| v.as_str()) {
            policy.description = desc.to_string();
        }

        if let Some(version) = json.get("version").and_then(|v| v.as_u64()) {
            policy.version = version;
        }

        if let Some(rules) = json.get("rules").and_then(|v| v.as_array()) {
            for rule_json in rules {
                policy.rules.push(Rule::from_json(rule_json)?);
            }
        }

        Ok(policy)
    }
}

/// Result of evaluating a policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    /// The combined effect, or None if no rules matched.
    pub effect: Option<Effect>,
    /// Rules that matched: (rule_id, effect).
    pub matched_rules: Vec<(String, Effect)>,
    /// Obligations from matched rules.
    pub obligations: Vec<Obligation>,
}

impl PolicyResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self.effect, Some(Effect::Allow))
    }

    pub fn is_denied(&self) -> bool {
        matches!(self.effect, Some(Effect::Deny))
    }
}

/// Combine multiple effects using the given strategy.
fn combine_effects(effects: &[Effect], strategy: CombineStrategy) -> Option<Effect> {
    if effects.is_empty() {
        return None;
    }
    match strategy {
        CombineStrategy::DenyOverrides => {
            if effects.iter().any(|e| *e == Effect::Deny) {
                Some(Effect::Deny)
            } else {
                Some(Effect::Allow)
            }
        }
        CombineStrategy::AllowOverrides => {
            if effects.iter().any(|e| *e == Effect::Allow) {
                Some(Effect::Allow)
            } else {
                Some(Effect::Deny)
            }
        }
        CombineStrategy::FirstMatch => effects.first().copied(),
        CombineStrategy::AllMustAllow => {
            if effects.iter().all(|e| *e == Effect::Allow) {
                Some(Effect::Allow)
            } else {
                Some(Effect::Deny)
            }
        }
    }
}

/// A named policy set: a collection of policies with a combining strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySet {
    /// Unique ID.
    pub id: String,
    /// Description.
    pub description: String,
    /// Policies in this set.
    pub policies: Vec<Policy>,
    /// Combining strategy across policies.
    pub combine: CombineStrategy,
}

impl PolicySet {
    pub fn new(id: &str, combine: CombineStrategy) -> Self {
        Self {
            id: id.to_string(),
            description: String::new(),
            policies: Vec::new(),
            combine,
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

    pub fn evaluate(&self, context: &Value) -> PolicyResult {
        let mut all_effects: Vec<Effect> = Vec::new();
        let mut all_matched: Vec<(String, Effect)> = Vec::new();
        let mut all_obligations: Vec<Obligation> = Vec::new();

        for policy in &self.policies {
            let result = policy.evaluate(context);
            if let Some(effect) = result.effect {
                all_effects.push(effect);
                all_matched.extend(result.matched_rules);
                all_obligations.extend(result.obligations);
            }
        }

        PolicyResult {
            effect: combine_effects(&all_effects, self.combine),
            matched_rules: all_matched,
            obligations: all_obligations,
        }
    }
}

/// Simulation result from a dry-run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// The policy that was evaluated.
    pub policy_id: String,
    /// The result.
    pub result: PolicyResult,
    /// The context used.
    pub context: Value,
    /// Per-rule evaluation details.
    pub rule_details: Vec<RuleDetail>,
}

/// Per-rule detail in a simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleDetail {
    pub rule_id: String,
    pub matched: bool,
    pub effect: Option<Effect>,
    /// Which conditions passed/failed.
    pub condition_results: Vec<(String, bool)>,
}

/// The main policy engine.
pub struct PolicyEngine {
    policies: HashMap<String, Policy>,
    policy_sets: HashMap<String, PolicySet>,
    /// Default effect when no policy matches.
    pub default_effect: Effect,
    /// Evaluation log.
    log: Vec<EvalLogEntry>,
}

/// An entry in the evaluation log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalLogEntry {
    pub policy_id: String,
    pub effect: Option<Effect>,
    pub timestamp: DateTime<Utc>,
    pub context_summary: String,
}

impl PolicyEngine {
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            policy_sets: HashMap::new(),
            default_effect: Effect::Deny,
            log: Vec::new(),
        }
    }

    /// Add a policy.
    pub fn add_policy(&mut self, policy: Policy) -> Result<(), PolicyEngineError> {
        if self.policies.contains_key(&policy.id) {
            return Err(PolicyEngineError::DuplicatePolicy(policy.id));
        }
        self.policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Update a policy with version check.
    pub fn update_policy(
        &mut self,
        policy: Policy,
        expected_version: u64,
    ) -> Result<(), PolicyEngineError> {
        let existing = self
            .policies
            .get(&policy.id)
            .ok_or_else(|| PolicyEngineError::PolicyNotFound(policy.id.clone()))?;
        if existing.version != expected_version {
            return Err(PolicyEngineError::VersionConflict {
                policy_id: policy.id,
                expected: expected_version,
                actual: existing.version,
            });
        }
        self.policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Remove a policy.
    pub fn remove_policy(&mut self, id: &str) -> Result<Policy, PolicyEngineError> {
        self.policies
            .remove(id)
            .ok_or_else(|| PolicyEngineError::PolicyNotFound(id.to_string()))
    }

    /// Get a policy by ID.
    pub fn get_policy(&self, id: &str) -> Option<&Policy> {
        self.policies.get(id)
    }

    /// Add a policy set.
    pub fn add_policy_set(&mut self, ps: PolicySet) -> Result<(), PolicyEngineError> {
        if self.policy_sets.contains_key(&ps.id) {
            return Err(PolicyEngineError::DuplicatePolicy(ps.id));
        }
        self.policy_sets.insert(ps.id.clone(), ps);
        Ok(())
    }

    /// Evaluate a policy by ID against a context.
    pub fn evaluate(
        &mut self,
        policy_id: &str,
        context: &Value,
    ) -> Result<PolicyResult, PolicyEngineError> {
        let policy = self
            .policies
            .get(policy_id)
            .ok_or_else(|| PolicyEngineError::PolicyNotFound(policy_id.to_string()))?;

        let result = policy.evaluate(context);

        self.log.push(EvalLogEntry {
            policy_id: policy_id.to_string(),
            effect: result.effect,
            timestamp: Utc::now(),
            context_summary: summarize_context(context),
        });

        Ok(result)
    }

    /// Evaluate a policy set by ID.
    pub fn evaluate_set(
        &mut self,
        set_id: &str,
        context: &Value,
    ) -> Result<PolicyResult, PolicyEngineError> {
        let ps = self
            .policy_sets
            .get(set_id)
            .ok_or_else(|| PolicyEngineError::PolicySetNotFound(set_id.to_string()))?;
        let result = ps.evaluate(context);

        self.log.push(EvalLogEntry {
            policy_id: set_id.to_string(),
            effect: result.effect,
            timestamp: Utc::now(),
            context_summary: summarize_context(context),
        });

        Ok(result)
    }

    /// Simulate (dry-run) a policy evaluation with detailed per-rule results.
    pub fn simulate(
        &self,
        policy_id: &str,
        context: &Value,
    ) -> Result<SimulationResult, PolicyEngineError> {
        let policy = self
            .policies
            .get(policy_id)
            .ok_or_else(|| PolicyEngineError::PolicyNotFound(policy_id.to_string()))?;

        let mut rule_details: Vec<RuleDetail> = Vec::new();

        for rule in &policy.rules {
            let mut condition_results: Vec<(String, bool)> = Vec::new();
            let mut all_passed = true;
            for cond in &rule.conditions {
                let passed = cond.evaluate(context);
                let desc = format!("{} {:?} {:?}", cond.field, cond.op, cond.value);
                condition_results.push((desc, passed));
                if !passed {
                    all_passed = false;
                }
            }
            rule_details.push(RuleDetail {
                rule_id: rule.id.clone(),
                matched: all_passed,
                effect: if all_passed { Some(rule.effect) } else { None },
                condition_results,
            });
        }

        let result = policy.evaluate(context);

        Ok(SimulationResult {
            policy_id: policy_id.to_string(),
            result,
            context: context.clone(),
            rule_details,
        })
    }

    /// Get the evaluation log.
    pub fn log(&self) -> &[EvalLogEntry] {
        &self.log
    }

    /// Clear the log.
    pub fn clear_log(&mut self) {
        self.log.clear();
    }

    /// Number of policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    /// Enable/disable a policy.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) -> Result<(), PolicyEngineError> {
        let policy = self
            .policies
            .get_mut(id)
            .ok_or_else(|| PolicyEngineError::PolicyNotFound(id.to_string()))?;
        policy.enabled = enabled;
        Ok(())
    }

    /// Load a policy from JSON DSL.
    pub fn load_policy_json(&mut self, json: &Value) -> Result<(), PolicyEngineError> {
        let policy = Policy::from_json(json)?;
        self.add_policy(policy)
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn summarize_context(context: &Value) -> String {
    if let Some(obj) = context.as_object() {
        let keys: Vec<&String> = obj.keys().take(5).collect();
        format!("keys: [{}]", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", "))
    } else {
        "non-object context".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_context() -> Value {
        json!({
            "subject": {
                "role": "admin",
                "department": "engineering",
                "clearance": 5
            },
            "resource": {
                "type": "document",
                "owner": "alice",
                "classification": "internal"
            },
            "action": {
                "type": "read"
            }
        })
    }

    #[test]
    fn test_condition_eq() {
        let cond = RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin"));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_neq() {
        let cond = RuleCondition::new("subject.role", ConditionOp::Neq, json!("user"));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_gt() {
        let cond = RuleCondition::new("subject.clearance", ConditionOp::Gt, json!(3));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_lt() {
        let cond = RuleCondition::new("subject.clearance", ConditionOp::Lt, json!(10));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_in() {
        let cond = RuleCondition::new(
            "subject.role",
            ConditionOp::In,
            json!(["admin", "superadmin"]),
        );
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_contains() {
        let cond = RuleCondition::new("subject.department", ConditionOp::Contains, json!("engine"));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_starts_with() {
        let cond = RuleCondition::new("subject.department", ConditionOp::StartsWith, json!("eng"));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_exists() {
        let cond = RuleCondition::new("subject.role", ConditionOp::Exists, json!(null));
        assert!(cond.evaluate(&sample_context()));
        let cond2 = RuleCondition::new("subject.missing", ConditionOp::Exists, json!(null));
        assert!(!cond2.evaluate(&sample_context()));
    }

    #[test]
    fn test_condition_not_exists() {
        let cond = RuleCondition::new("subject.missing", ConditionOp::NotExists, json!(null));
        assert!(cond.evaluate(&sample_context()));
    }

    #[test]
    fn test_rule_evaluate_match() {
        let rule = Rule::new("r1", Effect::Allow)
            .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin")))
            .with_condition(RuleCondition::new("action.type", ConditionOp::Eq, json!("read")));
        assert!(rule.evaluate(&sample_context()).is_allow());
    }

    #[test]
    fn test_rule_evaluate_no_match() {
        let rule = Rule::new("r1", Effect::Allow)
            .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("viewer")));
        assert_eq!(rule.evaluate(&sample_context()), EvalResult::NotMatched);
    }

    #[test]
    fn test_policy_deny_overrides() {
        let policy = Policy::new("p1", CombineStrategy::DenyOverrides)
            .with_rule(
                Rule::new("allow-admin", Effect::Allow)
                    .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin"))),
            )
            .with_rule(
                Rule::new("deny-classified", Effect::Deny)
                    .with_condition(RuleCondition::new(
                        "resource.classification",
                        ConditionOp::Eq,
                        json!("internal"),
                    )),
            );
        let result = policy.evaluate(&sample_context());
        assert!(result.is_denied());
    }

    #[test]
    fn test_policy_allow_overrides() {
        let policy = Policy::new("p1", CombineStrategy::AllowOverrides)
            .with_rule(
                Rule::new("allow-admin", Effect::Allow)
                    .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin"))),
            )
            .with_rule(
                Rule::new("deny-all", Effect::Deny)
                    .with_condition(RuleCondition::new("action.type", ConditionOp::Eq, json!("read"))),
            );
        let result = policy.evaluate(&sample_context());
        assert!(result.is_allowed());
    }

    #[test]
    fn test_policy_first_match() {
        let policy = Policy::new("p1", CombineStrategy::FirstMatch)
            .with_rule(
                Rule::new("deny-first", Effect::Deny)
                    .with_condition(RuleCondition::new("action.type", ConditionOp::Eq, json!("read")))
                    .with_priority(1),
            )
            .with_rule(
                Rule::new("allow-admin", Effect::Allow)
                    .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin")))
                    .with_priority(2),
            );
        let result = policy.evaluate(&sample_context());
        assert!(result.is_denied());
    }

    #[test]
    fn test_engine_add_and_evaluate() {
        let mut engine = PolicyEngine::new();
        let policy = Policy::new("p1", CombineStrategy::FirstMatch)
            .with_rule(
                Rule::new("allow-read", Effect::Allow)
                    .with_condition(RuleCondition::new("action.type", ConditionOp::Eq, json!("read"))),
            );
        engine.add_policy(policy).unwrap();
        let result = engine.evaluate("p1", &sample_context()).unwrap();
        assert!(result.is_allowed());
        assert_eq!(engine.log().len(), 1);
    }

    #[test]
    fn test_engine_duplicate_policy() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(Policy::new("p1", CombineStrategy::DenyOverrides)).unwrap();
        let err = engine.add_policy(Policy::new("p1", CombineStrategy::DenyOverrides)).unwrap_err();
        assert_eq!(err, PolicyEngineError::DuplicatePolicy("p1".into()));
    }

    #[test]
    fn test_engine_remove_policy() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(Policy::new("p1", CombineStrategy::DenyOverrides)).unwrap();
        engine.remove_policy("p1").unwrap();
        assert_eq!(engine.policy_count(), 0);
    }

    #[test]
    fn test_engine_version_conflict() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(Policy::new("p1", CombineStrategy::DenyOverrides)).unwrap();
        let mut updated = Policy::new("p1", CombineStrategy::AllowOverrides);
        updated.version = 2;
        let err = engine.update_policy(updated, 99).unwrap_err();
        match err {
            PolicyEngineError::VersionConflict { expected: 99, actual: 1, .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_engine_simulation() {
        let mut engine = PolicyEngine::new();
        let policy = Policy::new("p1", CombineStrategy::FirstMatch)
            .with_rule(
                Rule::new("r1", Effect::Allow)
                    .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin"))),
            )
            .with_rule(
                Rule::new("r2", Effect::Deny)
                    .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("guest"))),
            );
        engine.add_policy(policy).unwrap();
        let sim = engine.simulate("p1", &sample_context()).unwrap();
        assert!(sim.result.is_allowed());
        assert_eq!(sim.rule_details.len(), 2);
        assert!(sim.rule_details[0].matched);
        assert!(!sim.rule_details[1].matched);
        // Simulation should NOT add to the log
        assert!(engine.log().is_empty());
    }

    #[test]
    fn test_disabled_policy() {
        let mut engine = PolicyEngine::new();
        let policy = Policy::new("p1", CombineStrategy::FirstMatch)
            .with_rule(
                Rule::new("r1", Effect::Allow)
                    .with_condition(RuleCondition::new("action.type", ConditionOp::Eq, json!("read"))),
            );
        engine.add_policy(policy).unwrap();
        engine.set_enabled("p1", false).unwrap();
        let result = engine.evaluate("p1", &sample_context()).unwrap();
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_obligations() {
        let rule = Rule::new("r1", Effect::Allow)
            .with_condition(RuleCondition::new("action.type", ConditionOp::Eq, json!("read")))
            .with_obligation(
                Obligation::new("log-access", "log", true)
                    .with_param("level", "info"),
            );
        let policy = Policy::new("p1", CombineStrategy::FirstMatch).with_rule(rule);
        let result = policy.evaluate(&sample_context());
        assert!(result.is_allowed());
        assert_eq!(result.obligations.len(), 1);
        assert_eq!(result.obligations[0].action, "log");
        assert!(result.obligations[0].mandatory);
    }

    #[test]
    fn test_policy_from_json() {
        let json = json!({
            "id": "json-policy",
            "combine": "deny_overrides",
            "version": 3,
            "rules": [
                {
                    "id": "rule1",
                    "effect": "allow",
                    "conditions": [
                        {"field": "subject.role", "op": "eq", "value": "admin"}
                    ]
                }
            ]
        });
        let policy = Policy::from_json(&json).unwrap();
        assert_eq!(policy.id, "json-policy");
        assert_eq!(policy.version, 3);
        assert_eq!(policy.rules.len(), 1);
    }

    #[test]
    fn test_rule_from_json_invalid() {
        let json = json!({"effect": "allow"});
        let err = Rule::from_json(&json).unwrap_err();
        match err {
            PolicyEngineError::InvalidRule(msg) => assert!(msg.contains("id")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_load_policy_json() {
        let mut engine = PolicyEngine::new();
        let json = json!({
            "id": "p-json",
            "rules": [
                {
                    "id": "r1",
                    "effect": "deny",
                    "conditions": [
                        {"field": "action.type", "op": "eq", "value": "delete"}
                    ]
                }
            ]
        });
        engine.load_policy_json(&json).unwrap();
        assert!(engine.get_policy("p-json").is_some());
    }

    #[test]
    fn test_policy_set_evaluation() {
        let p1 = Policy::new("p1", CombineStrategy::FirstMatch)
            .with_rule(
                Rule::new("r1", Effect::Allow)
                    .with_condition(RuleCondition::new("subject.role", ConditionOp::Eq, json!("admin"))),
            );
        let p2 = Policy::new("p2", CombineStrategy::FirstMatch)
            .with_rule(
                Rule::new("r2", Effect::Deny)
                    .with_condition(RuleCondition::new(
                        "resource.classification",
                        ConditionOp::Eq,
                        json!("internal"),
                    )),
            );
        let ps = PolicySet::new("ps1", CombineStrategy::DenyOverrides)
            .with_policy(p1)
            .with_policy(p2);
        let result = ps.evaluate(&sample_context());
        assert!(result.is_denied());
    }

    #[test]
    fn test_all_must_allow() {
        let effects = vec![Effect::Allow, Effect::Allow, Effect::Deny];
        assert_eq!(combine_effects(&effects, CombineStrategy::AllMustAllow), Some(Effect::Deny));
        let effects2 = vec![Effect::Allow, Effect::Allow];
        assert_eq!(combine_effects(&effects2, CombineStrategy::AllMustAllow), Some(Effect::Allow));
    }

    #[test]
    fn test_resolve_nested_path() {
        let ctx = json!({"a": {"b": {"c": 42}}});
        assert_eq!(resolve_path(&ctx, "a.b.c"), Some(&json!(42)));
        assert_eq!(resolve_path(&ctx, "a.b.missing"), None);
    }

    #[test]
    fn test_empty_effects() {
        assert_eq!(combine_effects(&[], CombineStrategy::DenyOverrides), None);
    }

    #[test]
    fn test_condition_regex_simple() {
        let cond = RuleCondition::new("subject.department", ConditionOp::Regex, json!("^eng"));
        assert!(cond.evaluate(&sample_context()));
        let cond2 = RuleCondition::new("subject.department", ConditionOp::Regex, json!("^mark"));
        assert!(!cond2.evaluate(&sample_context()));
    }

    #[test]
    fn test_engine_clear_log() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(Policy::new("p1", CombineStrategy::DenyOverrides)).unwrap();
        let _ = engine.evaluate("p1", &sample_context());
        assert!(!engine.log().is_empty());
        engine.clear_log();
        assert!(engine.log().is_empty());
    }
}
