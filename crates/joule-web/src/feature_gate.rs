//! Feature gates / feature flags.
//!
//! Percentage rollout, user targeting, kill switches, gradual rollout,
//! flag evaluation with dependencies, and override rules. Pure Rust —
//! no external service needed; all evaluation is deterministic.

use std::collections::HashMap;
use std::fmt;

// ── Flag status ───────────────────────────────────────────────────

/// Whether a flag is enabled, disabled, or conditionally evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagStatus {
    /// Always on for all users.
    Enabled,
    /// Always off for all users (kill switch).
    Disabled,
    /// Evaluated based on rules (percentage, targeting, etc.).
    Conditional,
}

impl fmt::Display for FlagStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enabled => write!(f, "enabled"),
            Self::Disabled => write!(f, "disabled"),
            Self::Conditional => write!(f, "conditional"),
        }
    }
}

// ── Targeting rule ────────────────────────────────────────────────

/// A single targeting rule that can match against user attributes.
#[derive(Debug, Clone, PartialEq)]
pub enum TargetRule {
    /// Match if user ID is in the allow list.
    UserAllowList(Vec<String>),
    /// Match if user ID is in the deny list (override to off).
    UserDenyList(Vec<String>),
    /// Match if attribute equals a specific value.
    AttributeEquals { attribute: String, value: String },
    /// Match if attribute is in a set of values.
    AttributeIn { attribute: String, values: Vec<String> },
    /// Percentage rollout: user_id is hashed to decide inclusion.
    Percentage(u8),
    /// Always evaluates to a fixed result (useful in rule chains).
    Static(bool),
}

impl TargetRule {
    /// Evaluate this rule against a user context.
    pub fn evaluate(&self, ctx: &EvalContext) -> Option<bool> {
        match self {
            Self::UserAllowList(ids) => {
                if ids.iter().any(|id| id == &ctx.user_id) {
                    Some(true)
                } else {
                    None // no opinion
                }
            }
            Self::UserDenyList(ids) => {
                if ids.iter().any(|id| id == &ctx.user_id) {
                    Some(false)
                } else {
                    None
                }
            }
            Self::AttributeEquals { attribute, value } => {
                ctx.attributes
                    .get(attribute)
                    .map(|v| v == value)
            }
            Self::AttributeIn { attribute, values } => {
                ctx.attributes
                    .get(attribute)
                    .map(|v| values.iter().any(|val| val == v))
            }
            Self::Percentage(pct) => {
                let hash = deterministic_hash(&ctx.user_id);
                let bucket = (hash % 100) as u8;
                Some(bucket < *pct)
            }
            Self::Static(val) => Some(*val),
        }
    }
}

/// Deterministic hash for percentage rollout. Not cryptographic.
fn deterministic_hash(input: &str) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ── Evaluation context ────────────────────────────────────────────

/// Context for evaluating feature flags against a specific user/request.
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Unique user identifier.
    pub user_id: String,
    /// Arbitrary key-value attributes (e.g. "plan"="pro", "region"="us-east").
    pub attributes: HashMap<String, String>,
}

impl EvalContext {
    /// Create a context with just a user ID.
    pub fn new(user_id: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute.
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }
}

// ── Feature flag definition ───────────────────────────────────────

/// A single feature flag with its evaluation rules.
#[derive(Debug, Clone)]
pub struct FeatureFlag {
    /// Unique flag name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Top-level status (kill switch takes priority).
    pub status: FlagStatus,
    /// Ordered list of targeting rules (first match wins).
    pub rules: Vec<TargetRule>,
    /// Default value if no rules match.
    pub default_value: bool,
    /// Flags this depends on (all must be enabled for this to evaluate).
    pub dependencies: Vec<String>,
    /// Sticky: if true, evaluation is deterministic per user.
    pub sticky: bool,
}

impl FeatureFlag {
    /// Create a new flag with sensible defaults.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            status: FlagStatus::Conditional,
            rules: Vec::new(),
            default_value: false,
            dependencies: Vec::new(),
            sticky: true,
        }
    }

    /// Set the flag to always-on.
    pub fn enabled(mut self) -> Self {
        self.status = FlagStatus::Enabled;
        self
    }

    /// Set the flag to always-off (kill switch).
    pub fn disabled(mut self) -> Self {
        self.status = FlagStatus::Disabled;
        self
    }

    /// Add a targeting rule.
    pub fn with_rule(mut self, rule: TargetRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Set the default value when no rules match.
    pub fn with_default(mut self, val: bool) -> Self {
        self.default_value = val;
        self
    }

    /// Add a dependency on another flag.
    pub fn depends_on(mut self, flag_name: &str) -> Self {
        self.dependencies.push(flag_name.to_string());
        self
    }
}

// ── Evaluation result ─────────────────────────────────────────────

/// Result of evaluating a feature flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalResult {
    /// The flag name.
    pub flag_name: String,
    /// Whether the flag is on or off for this context.
    pub enabled: bool,
    /// Which rule index matched (None if default or status override).
    pub matched_rule: Option<usize>,
    /// Reason for the decision.
    pub reason: EvalReason,
}

/// Why a flag resolved to a particular value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalReason {
    /// Flag status is Enabled (always on).
    FlagEnabled,
    /// Flag status is Disabled (kill switch).
    FlagDisabled,
    /// A targeting rule matched.
    RuleMatch(usize),
    /// No rules matched; using default.
    Default,
    /// A dependency was not met.
    DependencyNotMet(String),
    /// Flag not found in the registry.
    FlagNotFound,
}

impl fmt::Display for EvalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FlagEnabled => write!(f, "flag_enabled"),
            Self::FlagDisabled => write!(f, "flag_disabled"),
            Self::RuleMatch(i) => write!(f, "rule_match:{i}"),
            Self::Default => write!(f, "default"),
            Self::DependencyNotMet(dep) => write!(f, "dependency_not_met:{dep}"),
            Self::FlagNotFound => write!(f, "flag_not_found"),
        }
    }
}

// ── Override ──────────────────────────────────────────────────────

/// A manual override for a specific user and flag.
#[derive(Debug, Clone)]
pub struct FlagOverride {
    pub flag_name: String,
    pub user_id: String,
    pub value: bool,
}

// ── Feature gate registry ─────────────────────────────────────────

/// Registry holding all feature flags and evaluating them.
#[derive(Debug, Clone)]
pub struct FeatureGate {
    flags: HashMap<String, FeatureFlag>,
    overrides: Vec<FlagOverride>,
}

impl FeatureGate {
    /// Create a new empty gate.
    pub fn new() -> Self {
        Self {
            flags: HashMap::new(),
            overrides: Vec::new(),
        }
    }

    /// Register a feature flag.
    pub fn register(&mut self, flag: FeatureFlag) {
        self.flags.insert(flag.name.clone(), flag);
    }

    /// Add a manual override.
    pub fn add_override(&mut self, flag_override: FlagOverride) {
        self.overrides.push(flag_override);
    }

    /// Remove all overrides for a specific flag and user.
    pub fn remove_override(&mut self, flag_name: &str, user_id: &str) {
        self.overrides
            .retain(|o| !(o.flag_name == flag_name && o.user_id == user_id));
    }

    /// Get a flag by name.
    pub fn get_flag(&self, name: &str) -> Option<&FeatureFlag> {
        self.flags.get(name)
    }

    /// Evaluate a flag for a given context.
    pub fn evaluate(&self, flag_name: &str, ctx: &EvalContext) -> EvalResult {
        let flag = match self.flags.get(flag_name) {
            Some(f) => f,
            None => {
                return EvalResult {
                    flag_name: flag_name.to_string(),
                    enabled: false,
                    matched_rule: None,
                    reason: EvalReason::FlagNotFound,
                };
            }
        };

        // Check manual overrides first.
        for ovr in &self.overrides {
            if ovr.flag_name == flag_name && ovr.user_id == ctx.user_id {
                return EvalResult {
                    flag_name: flag_name.to_string(),
                    enabled: ovr.value,
                    matched_rule: None,
                    reason: if ovr.value {
                        EvalReason::FlagEnabled
                    } else {
                        EvalReason::FlagDisabled
                    },
                };
            }
        }

        // Check flag status (kill switch).
        match flag.status {
            FlagStatus::Enabled => {
                return EvalResult {
                    flag_name: flag_name.to_string(),
                    enabled: true,
                    matched_rule: None,
                    reason: EvalReason::FlagEnabled,
                };
            }
            FlagStatus::Disabled => {
                return EvalResult {
                    flag_name: flag_name.to_string(),
                    enabled: false,
                    matched_rule: None,
                    reason: EvalReason::FlagDisabled,
                };
            }
            FlagStatus::Conditional => {}
        }

        // Check dependencies.
        for dep in &flag.dependencies {
            let dep_result = self.evaluate(dep, ctx);
            if !dep_result.enabled {
                return EvalResult {
                    flag_name: flag_name.to_string(),
                    enabled: false,
                    matched_rule: None,
                    reason: EvalReason::DependencyNotMet(dep.clone()),
                };
            }
        }

        // Evaluate targeting rules.
        for (i, rule) in flag.rules.iter().enumerate() {
            if let Some(result) = rule.evaluate(ctx) {
                return EvalResult {
                    flag_name: flag_name.to_string(),
                    enabled: result,
                    matched_rule: Some(i),
                    reason: EvalReason::RuleMatch(i),
                };
            }
        }

        // Default.
        EvalResult {
            flag_name: flag_name.to_string(),
            enabled: flag.default_value,
            matched_rule: None,
            reason: EvalReason::Default,
        }
    }

    /// Evaluate and return just the boolean result.
    pub fn is_enabled(&self, flag_name: &str, ctx: &EvalContext) -> bool {
        self.evaluate(flag_name, ctx).enabled
    }

    /// Get all registered flag names.
    pub fn flag_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.flags.keys().cloned().collect();
        names.sort();
        names
    }

    /// Evaluate all flags for a given context.
    pub fn evaluate_all(&self, ctx: &EvalContext) -> Vec<EvalResult> {
        let mut names: Vec<&String> = self.flags.keys().collect();
        names.sort();
        names
            .iter()
            .map(|name| self.evaluate(name, ctx))
            .collect()
    }

    /// Count how many flags are in each status.
    pub fn status_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for flag in self.flags.values() {
            *counts.entry(flag.status.to_string()).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for FeatureGate {
    fn default() -> Self {
        Self::new()
    }
}

// ── Gradual rollout helper ────────────────────────────────────────

/// Configuration for a gradual rollout that increases percentage over stages.
#[derive(Debug, Clone)]
pub struct GradualRollout {
    /// Flag name to control.
    pub flag_name: String,
    /// Stages as percentages (e.g. [1, 5, 10, 25, 50, 100]).
    pub stages: Vec<u8>,
    /// Current stage index.
    pub current_stage: usize,
}

impl GradualRollout {
    /// Create a new gradual rollout.
    pub fn new(flag_name: &str, stages: Vec<u8>) -> Self {
        Self {
            flag_name: flag_name.to_string(),
            stages,
            current_stage: 0,
        }
    }

    /// Get the current rollout percentage.
    pub fn current_percentage(&self) -> u8 {
        self.stages.get(self.current_stage).copied().unwrap_or(0)
    }

    /// Advance to the next stage. Returns the new percentage, or None if complete.
    pub fn advance(&mut self) -> Option<u8> {
        if self.current_stage + 1 < self.stages.len() {
            self.current_stage += 1;
            Some(self.stages[self.current_stage])
        } else {
            None
        }
    }

    /// Roll back to the previous stage. Returns the new percentage, or None at start.
    pub fn rollback(&mut self) -> Option<u8> {
        if self.current_stage > 0 {
            self.current_stage -= 1;
            Some(self.stages[self.current_stage])
        } else {
            None
        }
    }

    /// Is the rollout at 100%?
    pub fn is_complete(&self) -> bool {
        self.current_percentage() == 100
    }

    /// Build a TargetRule for the current stage.
    pub fn to_rule(&self) -> TargetRule {
        TargetRule::Percentage(self.current_percentage())
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_status_display() {
        assert_eq!(FlagStatus::Enabled.to_string(), "enabled");
        assert_eq!(FlagStatus::Disabled.to_string(), "disabled");
        assert_eq!(FlagStatus::Conditional.to_string(), "conditional");
    }

    #[test]
    fn eval_reason_display() {
        assert_eq!(EvalReason::FlagEnabled.to_string(), "flag_enabled");
        assert_eq!(EvalReason::RuleMatch(3).to_string(), "rule_match:3");
        assert_eq!(
            EvalReason::DependencyNotMet("x".into()).to_string(),
            "dependency_not_met:x"
        );
    }

    #[test]
    fn simple_enabled_flag() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("f1", "test").enabled());
        let ctx = EvalContext::new("user1");
        let res = gate.evaluate("f1", &ctx);
        assert!(res.enabled);
        assert_eq!(res.reason, EvalReason::FlagEnabled);
    }

    #[test]
    fn simple_disabled_flag() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("f1", "test").disabled());
        let ctx = EvalContext::new("user1");
        assert!(!gate.is_enabled("f1", &ctx));
    }

    #[test]
    fn flag_not_found() {
        let gate = FeatureGate::new();
        let ctx = EvalContext::new("user1");
        let res = gate.evaluate("nonexistent", &ctx);
        assert!(!res.enabled);
        assert_eq!(res.reason, EvalReason::FlagNotFound);
    }

    #[test]
    fn user_allow_list() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("beta", "beta feature")
                .with_rule(TargetRule::UserAllowList(vec!["alice".into(), "bob".into()])),
        );

        assert!(gate.is_enabled("beta", &EvalContext::new("alice")));
        assert!(gate.is_enabled("beta", &EvalContext::new("bob")));
        assert!(!gate.is_enabled("beta", &EvalContext::new("charlie")));
    }

    #[test]
    fn user_deny_list() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("f1", "test")
                .with_rule(TargetRule::UserDenyList(vec!["banned".into()]))
                .with_default(true),
        );

        assert!(!gate.is_enabled("f1", &EvalContext::new("banned")));
        assert!(gate.is_enabled("f1", &EvalContext::new("ok_user")));
    }

    #[test]
    fn attribute_equals() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("pro_feature", "pro only").with_rule(TargetRule::AttributeEquals {
                attribute: "plan".into(),
                value: "pro".into(),
            }),
        );

        let pro_ctx = EvalContext::new("u1").with_attribute("plan", "pro");
        let free_ctx = EvalContext::new("u2").with_attribute("plan", "free");
        let no_plan = EvalContext::new("u3");

        assert!(gate.is_enabled("pro_feature", &pro_ctx));
        assert!(!gate.is_enabled("pro_feature", &free_ctx));
        // No attribute -> no opinion -> falls to default (false).
        assert!(!gate.is_enabled("pro_feature", &no_plan));
    }

    #[test]
    fn attribute_in_set() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("us_feature", "US regions").with_rule(TargetRule::AttributeIn {
                attribute: "region".into(),
                values: vec!["us-east".into(), "us-west".into()],
            }),
        );

        let east = EvalContext::new("u1").with_attribute("region", "us-east");
        let eu = EvalContext::new("u2").with_attribute("region", "eu-west");

        assert!(gate.is_enabled("us_feature", &east));
        assert!(!gate.is_enabled("us_feature", &eu));
    }

    #[test]
    fn percentage_rollout_deterministic() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("pct", "percentage test").with_rule(TargetRule::Percentage(50)),
        );

        // Same user always gets same result.
        let ctx = EvalContext::new("user_42");
        let r1 = gate.is_enabled("pct", &ctx);
        let r2 = gate.is_enabled("pct", &ctx);
        assert_eq!(r1, r2);
    }

    #[test]
    fn percentage_100_always_on() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("all", "everyone").with_rule(TargetRule::Percentage(100)),
        );

        for i in 0..50 {
            let ctx = EvalContext::new(&format!("user_{i}"));
            assert!(gate.is_enabled("all", &ctx));
        }
    }

    #[test]
    fn percentage_0_always_off() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("none", "nobody").with_rule(TargetRule::Percentage(0)),
        );

        for i in 0..50 {
            let ctx = EvalContext::new(&format!("user_{i}"));
            assert!(!gate.is_enabled("none", &ctx));
        }
    }

    #[test]
    fn static_rule() {
        let rule_on = TargetRule::Static(true);
        let rule_off = TargetRule::Static(false);
        let ctx = EvalContext::new("anyone");
        assert_eq!(rule_on.evaluate(&ctx), Some(true));
        assert_eq!(rule_off.evaluate(&ctx), Some(false));
    }

    #[test]
    fn first_rule_wins() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("f", "first match")
                .with_rule(TargetRule::UserAllowList(vec!["vip".into()]))
                .with_rule(TargetRule::Static(false)),
        );

        let vip = EvalContext::new("vip");
        let res = gate.evaluate("f", &vip);
        assert!(res.enabled);
        assert_eq!(res.matched_rule, Some(0));

        let other = EvalContext::new("other");
        let res2 = gate.evaluate("f", &other);
        assert!(!res2.enabled);
        assert_eq!(res2.matched_rule, Some(1));
    }

    #[test]
    fn default_value_when_no_rules() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("f", "test").with_default(true));
        let res = gate.evaluate("f", &EvalContext::new("u"));
        assert!(res.enabled);
        assert_eq!(res.reason, EvalReason::Default);
    }

    #[test]
    fn dependency_not_met() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("base", "base").disabled());
        gate.register(
            FeatureFlag::new("child", "depends on base")
                .depends_on("base")
                .with_default(true),
        );

        let ctx = EvalContext::new("u1");
        let res = gate.evaluate("child", &ctx);
        assert!(!res.enabled);
        assert_eq!(res.reason, EvalReason::DependencyNotMet("base".into()));
    }

    #[test]
    fn dependency_met() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("base", "base").enabled());
        gate.register(
            FeatureFlag::new("child", "depends on base")
                .depends_on("base")
                .with_default(true),
        );

        let ctx = EvalContext::new("u1");
        assert!(gate.is_enabled("child", &ctx));
    }

    #[test]
    fn manual_override() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("f", "test").disabled());
        gate.add_override(FlagOverride {
            flag_name: "f".into(),
            user_id: "admin".into(),
            value: true,
        });

        assert!(gate.is_enabled("f", &EvalContext::new("admin")));
        assert!(!gate.is_enabled("f", &EvalContext::new("regular")));
    }

    #[test]
    fn remove_override() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("f", "test").disabled());
        gate.add_override(FlagOverride {
            flag_name: "f".into(),
            user_id: "u1".into(),
            value: true,
        });
        assert!(gate.is_enabled("f", &EvalContext::new("u1")));

        gate.remove_override("f", "u1");
        assert!(!gate.is_enabled("f", &EvalContext::new("u1")));
    }

    #[test]
    fn flag_names_sorted() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("zebra", "z"));
        gate.register(FeatureFlag::new("alpha", "a"));
        gate.register(FeatureFlag::new("middle", "m"));
        assert_eq!(gate.flag_names(), vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn evaluate_all_returns_all_flags() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("a", "aa").enabled());
        gate.register(FeatureFlag::new("b", "bb").disabled());
        let results = gate.evaluate_all(&EvalContext::new("u"));
        assert_eq!(results.len(), 2);
        let a_res = results.iter().find(|r| r.flag_name == "a").unwrap();
        let b_res = results.iter().find(|r| r.flag_name == "b").unwrap();
        assert!(a_res.enabled);
        assert!(!b_res.enabled);
    }

    #[test]
    fn status_counts() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("a", "").enabled());
        gate.register(FeatureFlag::new("b", "").enabled());
        gate.register(FeatureFlag::new("c", "").disabled());
        gate.register(FeatureFlag::new("d", ""));
        let counts = gate.status_counts();
        assert_eq!(counts.get("enabled").copied(), Some(2));
        assert_eq!(counts.get("disabled").copied(), Some(1));
        assert_eq!(counts.get("conditional").copied(), Some(1));
    }

    #[test]
    fn gradual_rollout_stages() {
        let mut rollout = GradualRollout::new("feature_x", vec![1, 5, 10, 25, 50, 100]);
        assert_eq!(rollout.current_percentage(), 1);
        assert!(!rollout.is_complete());

        assert_eq!(rollout.advance(), Some(5));
        assert_eq!(rollout.advance(), Some(10));
        assert_eq!(rollout.current_percentage(), 10);

        assert_eq!(rollout.rollback(), Some(5));
        assert_eq!(rollout.current_percentage(), 5);
    }

    #[test]
    fn gradual_rollout_completion() {
        let mut rollout = GradualRollout::new("f", vec![50, 100]);
        assert_eq!(rollout.advance(), Some(100));
        assert!(rollout.is_complete());
        assert_eq!(rollout.advance(), None);
    }

    #[test]
    fn gradual_rollout_at_start_no_rollback() {
        let mut rollout = GradualRollout::new("f", vec![10, 50]);
        assert_eq!(rollout.rollback(), None);
        assert_eq!(rollout.current_percentage(), 10);
    }

    #[test]
    fn gradual_rollout_to_rule() {
        let rollout = GradualRollout::new("f", vec![25, 50, 75, 100]);
        let rule = rollout.to_rule();
        match rule {
            TargetRule::Percentage(p) => assert_eq!(p, 25),
            _ => panic!("expected Percentage rule"),
        }
    }

    #[test]
    fn get_flag_returns_registered() {
        let mut gate = FeatureGate::new();
        gate.register(FeatureFlag::new("f1", "flag one").enabled());
        let f = gate.get_flag("f1").unwrap();
        assert_eq!(f.name, "f1");
        assert_eq!(f.status, FlagStatus::Enabled);
    }

    #[test]
    fn get_flag_returns_none_for_missing() {
        let gate = FeatureGate::new();
        assert!(gate.get_flag("nonexistent").is_none());
    }

    #[test]
    fn deny_list_takes_precedence_before_allow_list() {
        let mut gate = FeatureGate::new();
        gate.register(
            FeatureFlag::new("f", "test")
                .with_rule(TargetRule::UserDenyList(vec!["banned".into()]))
                .with_rule(TargetRule::UserAllowList(vec!["banned".into()])),
        );
        // Deny list is rule 0, matched first.
        assert!(!gate.is_enabled("f", &EvalContext::new("banned")));
    }

    #[test]
    fn hash_consistency() {
        let h1 = deterministic_hash("test_user");
        let h2 = deterministic_hash("test_user");
        assert_eq!(h1, h2);
        let h3 = deterministic_hash("other_user");
        assert_ne!(h1, h3);
    }

    #[test]
    fn gradual_rollout_empty_stages() {
        let rollout = GradualRollout::new("f", vec![]);
        assert_eq!(rollout.current_percentage(), 0);
    }

    #[test]
    fn eval_context_with_multiple_attributes() {
        let ctx = EvalContext::new("u1")
            .with_attribute("plan", "pro")
            .with_attribute("region", "us-east");
        assert_eq!(ctx.attributes.len(), 2);
        assert_eq!(ctx.attributes["plan"], "pro");
        assert_eq!(ctx.attributes["region"], "us-east");
    }
}
