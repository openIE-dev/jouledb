//! Business rules engine — rule definitions, priority, fact-based evaluation,
//! chaining, conflict resolution, rule groups, and audit trails.
//!
//! Replaces JS rule engines (json-rules-engine, Nools) with a pure-Rust
//! forward-chaining rules engine that tracks every rule firing.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Rule engine domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleError {
    /// Rule not found.
    RuleNotFound(String),
    /// Duplicate rule ID.
    DuplicateRule(String),
    /// Fact not found.
    FactNotFound(String),
    /// Condition evaluation failure.
    ConditionError(String),
    /// Rule group not found.
    GroupNotFound(String),
    /// Max chain depth exceeded.
    MaxChainDepthExceeded(u32),
}

impl std::fmt::Display for RuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleNotFound(id) => write!(f, "rule not found: {id}"),
            Self::DuplicateRule(id) => write!(f, "duplicate rule: {id}"),
            Self::FactNotFound(k) => write!(f, "fact not found: {k}"),
            Self::ConditionError(msg) => write!(f, "condition error: {msg}"),
            Self::GroupNotFound(g) => write!(f, "group not found: {g}"),
            Self::MaxChainDepthExceeded(d) => write!(f, "max chain depth exceeded: {d}"),
        }
    }
}

impl std::error::Error for RuleError {}

// ── Fact Value ──────────────────────────────────────────────────

/// A fact value in the working memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FactValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<FactValue>),
}

impl FactValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(n) => Some(*n),
            Self::Integer(n) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

impl std::fmt::Display for FactValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Boolean(b) => write!(f, "{b}"),
            Self::List(l) => write!(f, "{l:?}"),
        }
    }
}

// ── Conditions ──────────────────────────────────────────────────

/// Comparison operator for conditions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
    Contains,
    Exists,
    NotExists,
}

/// A single condition comparing a fact to a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionClause {
    pub fact_key: String,
    pub operator: Operator,
    pub value: Option<FactValue>,
}

/// Composite condition with AND/OR logic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Condition {
    Clause(ConditionClause),
    All(Vec<Condition>),
    Any(Vec<Condition>),
    Not(Box<Condition>),
}

// ── Actions ─────────────────────────────────────────────────────

/// An action to perform when a rule fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Set a fact to a value.
    SetFact { key: String, value: FactValue },
    /// Remove a fact.
    RemoveFact { key: String },
    /// Emit a named event with data.
    Emit { event: String, data: HashMap<String, FactValue> },
    /// Halt further rule evaluation.
    Halt,
}

// ── Conflict Resolution ─────────────────────────────────────────

/// Strategy for resolving conflicts when multiple rules match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictStrategy {
    /// Fire all matching rules in priority order.
    Priority,
    /// Fire only the highest priority rule.
    FirstMatch,
    /// Fire all matching rules (no ordering).
    FireAll,
}

// ── Rule ────────────────────────────────────────────────────────

/// A business rule with conditions and actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub priority: i32,
    pub group: Option<String>,
    pub condition: Condition,
    pub actions: Vec<Action>,
    pub enabled: bool,
}

impl Rule {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            priority: 0,
            group: None,
            condition: Condition::All(Vec::new()),
            actions: Vec::new(),
            enabled: true,
        }
    }

    pub fn with_priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }

    pub fn with_group(mut self, g: impl Into<String>) -> Self {
        self.group = Some(g.into());
        self
    }

    pub fn with_condition(mut self, c: Condition) -> Self {
        self.condition = c;
        self
    }

    pub fn with_action(mut self, a: Action) -> Self {
        self.actions.push(a);
        self
    }
}

// ── Audit Trail ─────────────────────────────────────────────────

/// Record of a rule firing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleFiring {
    pub rule_id: String,
    pub timestamp: DateTime<Utc>,
    pub facts_snapshot: HashMap<String, FactValue>,
    pub actions_executed: Vec<Action>,
}

// ── Working Memory ──────────────────────────────────────────────

/// The fact store / working memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub facts: HashMap<String, FactValue>,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: FactValue) {
        self.facts.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&FactValue> {
        self.facts.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<FactValue> {
        self.facts.remove(key)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.facts.contains_key(key)
    }
}

// ── Condition Evaluator ─────────────────────────────────────────

fn compare_values(left: &FactValue, op: &Operator, right: &FactValue) -> bool {
    match op {
        Operator::Equal => left == right,
        Operator::NotEqual => left != right,
        Operator::GreaterThan => {
            left.as_f64().zip(right.as_f64()).map_or(false, |(l, r)| l > r)
        }
        Operator::GreaterOrEqual => {
            left.as_f64().zip(right.as_f64()).map_or(false, |(l, r)| l >= r)
        }
        Operator::LessThan => {
            left.as_f64().zip(right.as_f64()).map_or(false, |(l, r)| l < r)
        }
        Operator::LessOrEqual => {
            left.as_f64().zip(right.as_f64()).map_or(false, |(l, r)| l <= r)
        }
        Operator::Contains => {
            if let (FactValue::String(haystack), FactValue::String(needle)) = (left, right) {
                haystack.contains(needle.as_str())
            } else {
                false
            }
        }
        Operator::Exists | Operator::NotExists => true, // handled at clause level
    }
}

fn evaluate_condition(cond: &Condition, mem: &WorkingMemory) -> bool {
    match cond {
        Condition::Clause(clause) => {
            match &clause.operator {
                Operator::Exists => mem.contains(&clause.fact_key),
                Operator::NotExists => !mem.contains(&clause.fact_key),
                op => {
                    let fact = match mem.get(&clause.fact_key) {
                        Some(v) => v,
                        None => return false,
                    };
                    match &clause.value {
                        Some(val) => compare_values(fact, op, val),
                        None => false,
                    }
                }
            }
        }
        Condition::All(conds) => conds.iter().all(|c| evaluate_condition(c, mem)),
        Condition::Any(conds) => conds.iter().any(|c| evaluate_condition(c, mem)),
        Condition::Not(c) => !evaluate_condition(c, mem),
    }
}

// ── Rule Engine ─────────────────────────────────────────────────

/// The rule engine that evaluates rules against working memory.
#[derive(Debug)]
pub struct RuleEngine {
    pub rules: Vec<Rule>,
    pub memory: WorkingMemory,
    pub audit: Vec<RuleFiring>,
    pub conflict_strategy: ConflictStrategy,
    pub max_chain_depth: u32,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            memory: WorkingMemory::new(),
            audit: Vec::new(),
            conflict_strategy: ConflictStrategy::Priority,
            max_chain_depth: 100,
        }
    }

    pub fn with_strategy(mut self, s: ConflictStrategy) -> Self {
        self.conflict_strategy = s;
        self
    }

    /// Add a rule.
    pub fn add_rule(&mut self, rule: Rule) -> Result<(), RuleError> {
        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(RuleError::DuplicateRule(rule.id));
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Remove a rule by ID.
    pub fn remove_rule(&mut self, id: &str) -> Result<Rule, RuleError> {
        let pos = self.rules.iter().position(|r| r.id == id)
            .ok_or_else(|| RuleError::RuleNotFound(id.to_string()))?;
        Ok(self.rules.remove(pos))
    }

    /// Set a fact in working memory.
    pub fn set_fact(&mut self, key: impl Into<String>, value: FactValue) {
        self.memory.set(key, value);
    }

    /// Get matching rules sorted by priority.
    fn matching_rules(&self) -> Vec<&Rule> {
        let mut matched: Vec<&Rule> = self.rules.iter()
            .filter(|r| r.enabled && evaluate_condition(&r.condition, &self.memory))
            .collect();
        matched.sort_by(|a, b| b.priority.cmp(&a.priority));
        matched
    }

    /// Fire rules once (no chaining).
    pub fn fire_once(&mut self) -> Vec<String> {
        let matched: Vec<Rule> = self.matching_rules().into_iter().cloned().collect();
        let to_fire = match self.conflict_strategy {
            ConflictStrategy::FirstMatch => matched.into_iter().take(1).collect::<Vec<_>>(),
            ConflictStrategy::Priority | ConflictStrategy::FireAll => matched,
        };

        let mut fired_ids = Vec::new();
        for rule in &to_fire {
            let firing = RuleFiring {
                rule_id: rule.id.clone(),
                timestamp: Utc::now(),
                facts_snapshot: self.memory.facts.clone(),
                actions_executed: rule.actions.clone(),
            };
            self.audit.push(firing);

            let mut halt = false;
            for action in &rule.actions {
                match action {
                    Action::SetFact { key, value } => {
                        self.memory.set(key.clone(), value.clone());
                    }
                    Action::RemoveFact { key } => {
                        self.memory.remove(key);
                    }
                    Action::Emit { .. } => { /* events collected via audit */ }
                    Action::Halt => { halt = true; }
                }
            }
            fired_ids.push(rule.id.clone());
            if halt {
                break;
            }
        }
        fired_ids
    }

    /// Fire rules with forward chaining until no new rules fire or max depth.
    pub fn fire_all(&mut self) -> Result<Vec<String>, RuleError> {
        let mut all_fired = Vec::new();
        for _ in 0..self.max_chain_depth {
            let fired = self.fire_once();
            if fired.is_empty() {
                break;
            }
            all_fired.extend(fired);
        }
        Ok(all_fired)
    }

    /// Get rules in a specific group.
    pub fn rules_in_group(&self, group: &str) -> Vec<&Rule> {
        self.rules.iter()
            .filter(|r| r.group.as_deref() == Some(group))
            .collect()
    }

    /// Get the audit trail.
    pub fn audit_trail(&self) -> &[RuleFiring] {
        &self.audit
    }

    /// Clear the audit trail.
    pub fn clear_audit(&mut self) {
        self.audit.clear();
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn age_rule() -> Rule {
        Rule::new("age-check", "Age Check")
            .with_priority(10)
            .with_condition(Condition::Clause(ConditionClause {
                fact_key: "age".to_string(),
                operator: Operator::GreaterOrEqual,
                value: Some(FactValue::Integer(18)),
            }))
            .with_action(Action::SetFact {
                key: "eligible".to_string(),
                value: FactValue::Boolean(true),
            })
    }

    #[test]
    fn test_rule_creation() {
        let r = age_rule();
        assert_eq!(r.id, "age-check");
        assert_eq!(r.priority, 10);
        assert!(r.enabled);
    }

    #[test]
    fn test_fact_operations() {
        let mut mem = WorkingMemory::new();
        mem.set("name", FactValue::String("Alice".into()));
        assert_eq!(mem.get("name").unwrap().as_str(), Some("Alice"));
        assert!(mem.contains("name"));
        mem.remove("name");
        assert!(!mem.contains("name"));
    }

    #[test]
    fn test_condition_evaluation_basic() {
        let mut mem = WorkingMemory::new();
        mem.set("age", FactValue::Integer(21));
        let cond = Condition::Clause(ConditionClause {
            fact_key: "age".to_string(),
            operator: Operator::GreaterOrEqual,
            value: Some(FactValue::Integer(18)),
        });
        assert!(evaluate_condition(&cond, &mem));
    }

    #[test]
    fn test_condition_not() {
        let mut mem = WorkingMemory::new();
        mem.set("age", FactValue::Integer(15));
        let cond = Condition::Not(Box::new(Condition::Clause(ConditionClause {
            fact_key: "age".to_string(),
            operator: Operator::GreaterOrEqual,
            value: Some(FactValue::Integer(18)),
        })));
        assert!(evaluate_condition(&cond, &mem));
    }

    #[test]
    fn test_condition_exists() {
        let mut mem = WorkingMemory::new();
        let exists = Condition::Clause(ConditionClause {
            fact_key: "x".to_string(),
            operator: Operator::Exists,
            value: None,
        });
        assert!(!evaluate_condition(&exists, &mem));
        mem.set("x", FactValue::Boolean(true));
        assert!(evaluate_condition(&exists, &mem));
    }

    #[test]
    fn test_fire_once() {
        let mut engine = RuleEngine::new();
        engine.add_rule(age_rule()).unwrap();
        engine.set_fact("age", FactValue::Integer(21));
        let fired = engine.fire_once();
        assert_eq!(fired, vec!["age-check"]);
        assert_eq!(engine.memory.get("eligible").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_fire_no_match() {
        let mut engine = RuleEngine::new();
        engine.add_rule(age_rule()).unwrap();
        engine.set_fact("age", FactValue::Integer(15));
        let fired = engine.fire_once();
        assert!(fired.is_empty());
    }

    #[test]
    fn test_first_match_strategy() {
        let mut engine = RuleEngine::new().with_strategy(ConflictStrategy::FirstMatch);
        engine.add_rule(
            Rule::new("r1", "R1").with_priority(5)
                .with_condition(Condition::All(vec![]))
                .with_action(Action::SetFact { key: "x".into(), value: FactValue::Integer(1) }),
        ).unwrap();
        engine.add_rule(
            Rule::new("r2", "R2").with_priority(10)
                .with_condition(Condition::All(vec![]))
                .with_action(Action::SetFact { key: "y".into(), value: FactValue::Integer(2) }),
        ).unwrap();
        let fired = engine.fire_once();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0], "r2"); // highest priority
    }

    #[test]
    fn test_duplicate_rule_rejected() {
        let mut engine = RuleEngine::new();
        engine.add_rule(age_rule()).unwrap();
        assert!(matches!(engine.add_rule(age_rule()), Err(RuleError::DuplicateRule(_))));
    }

    #[test]
    fn test_remove_rule() {
        let mut engine = RuleEngine::new();
        engine.add_rule(age_rule()).unwrap();
        let r = engine.remove_rule("age-check").unwrap();
        assert_eq!(r.id, "age-check");
        assert!(engine.remove_rule("age-check").is_err());
    }

    #[test]
    fn test_audit_trail() {
        let mut engine = RuleEngine::new();
        engine.add_rule(age_rule()).unwrap();
        engine.set_fact("age", FactValue::Integer(21));
        engine.fire_once();
        assert_eq!(engine.audit_trail().len(), 1);
        assert_eq!(engine.audit_trail()[0].rule_id, "age-check");
        engine.clear_audit();
        assert!(engine.audit_trail().is_empty());
    }

    #[test]
    fn test_rule_groups() {
        let mut engine = RuleEngine::new();
        engine.add_rule(age_rule().with_group("eligibility")).unwrap();
        engine.add_rule(
            Rule::new("name-check", "Name Check")
                .with_group("validation")
                .with_condition(Condition::All(vec![])),
        ).unwrap();
        assert_eq!(engine.rules_in_group("eligibility").len(), 1);
        assert_eq!(engine.rules_in_group("validation").len(), 1);
        assert_eq!(engine.rules_in_group("other").len(), 0);
    }

    #[test]
    fn test_halt_action() {
        let mut engine = RuleEngine::new();
        engine.add_rule(
            Rule::new("r1", "R1").with_priority(10)
                .with_condition(Condition::All(vec![]))
                .with_action(Action::Halt),
        ).unwrap();
        engine.add_rule(
            Rule::new("r2", "R2").with_priority(5)
                .with_condition(Condition::All(vec![]))
                .with_action(Action::SetFact { key: "x".into(), value: FactValue::Integer(1) }),
        ).unwrap();
        let fired = engine.fire_once();
        assert_eq!(fired, vec!["r1"]); // r2 not fired due to halt
    }

    #[test]
    fn test_chaining() {
        let mut engine = RuleEngine::new();
        // Rule 1: if age >= 18 → set eligible = true
        engine.add_rule(
            Rule::new("r1", "R1").with_priority(10)
                .with_condition(Condition::All(vec![
                    Condition::Clause(ConditionClause {
                        fact_key: "age".into(),
                        operator: Operator::GreaterOrEqual,
                        value: Some(FactValue::Integer(18)),
                    }),
                    Condition::Clause(ConditionClause {
                        fact_key: "eligible".into(),
                        operator: Operator::NotExists,
                        value: None,
                    }),
                ]))
                .with_action(Action::SetFact {
                    key: "eligible".into(),
                    value: FactValue::Boolean(true),
                }),
        ).unwrap();
        // Rule 2: if eligible → set approved = true
        engine.add_rule(
            Rule::new("r2", "R2").with_priority(5)
                .with_condition(Condition::All(vec![
                    Condition::Clause(ConditionClause {
                        fact_key: "eligible".into(),
                        operator: Operator::Equal,
                        value: Some(FactValue::Boolean(true)),
                    }),
                    Condition::Clause(ConditionClause {
                        fact_key: "approved".into(),
                        operator: Operator::NotExists,
                        value: None,
                    }),
                ]))
                .with_action(Action::SetFact {
                    key: "approved".into(),
                    value: FactValue::Boolean(true),
                }),
        ).unwrap();

        engine.set_fact("age", FactValue::Integer(21));
        let fired = engine.fire_all().unwrap();
        assert!(fired.contains(&"r1".to_string()));
        assert!(fired.contains(&"r2".to_string()));
        assert_eq!(engine.memory.get("approved").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_contains_operator() {
        let mut mem = WorkingMemory::new();
        mem.set("email", FactValue::String("user@example.com".into()));
        let cond = Condition::Clause(ConditionClause {
            fact_key: "email".into(),
            operator: Operator::Contains,
            value: Some(FactValue::String("@example".into())),
        });
        assert!(evaluate_condition(&cond, &mem));
    }
}
