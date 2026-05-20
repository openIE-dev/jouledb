//! Data masking/redaction — field-level masking rules, masking strategies
//! (full/partial/hash/tokenize/null), PII detection patterns, mask on serialize,
//! configurable by role, and audit trail of mask operations.
//!
//! Replaces `data-mask`, `sensitive-data-masker`, and PII redaction services with
//! a pure-Rust masking engine supporting role-based field redaction and audit logging.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Data masking errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskingError {
    /// Invalid field path.
    InvalidPath(String),
    /// Rule not found.
    RuleNotFound(String),
    /// Duplicate rule ID.
    DuplicateRule(String),
    /// Invalid masking configuration.
    InvalidConfig(String),
}

impl fmt::Display for MaskingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(p) => write!(f, "invalid field path: {p}"),
            Self::RuleNotFound(id) => write!(f, "masking rule not found: {id}"),
            Self::DuplicateRule(id) => write!(f, "duplicate masking rule: {id}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for MaskingError {}

// ── Types ──────────────────────────────────────────────────────

/// How to mask a field value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskStrategy {
    /// Replace entirely with a fixed string (default: "***").
    Full { replacement: String },
    /// Show first N and last M characters, mask the rest.
    Partial { show_first: usize, show_last: usize, mask_char: char },
    /// Replace with a deterministic hash (first 16 hex chars of a simple hash).
    Hash,
    /// Replace with a random-looking token that maps to the original (deterministic from value).
    Tokenize { prefix: String },
    /// Replace with null.
    Null,
    /// Redact: remove the field entirely.
    Redact,
    /// Replace with a constant value.
    Constant { value: String },
    /// Preserve first N characters.
    TruncateAfter { keep: usize },
}

impl Default for MaskStrategy {
    fn default() -> Self {
        Self::Full { replacement: "***".to_string() }
    }
}

impl MaskStrategy {
    /// Apply this strategy to a string value.
    pub fn apply(&self, value: &str) -> MaskResult {
        match self {
            Self::Full { replacement } => MaskResult::Replaced(replacement.clone()),
            Self::Partial { show_first, show_last, mask_char } => {
                let chars: Vec<char> = value.chars().collect();
                let len = chars.len();
                let first = *show_first;
                let last = *show_last;
                if first + last >= len {
                    // Not enough chars to mask; mask all
                    return MaskResult::Replaced(std::iter::repeat(*mask_char).take(len).collect());
                }
                let mut result = String::with_capacity(len);
                for (i, ch) in chars.iter().enumerate() {
                    if i < first || i >= len - last {
                        result.push(*ch);
                    } else {
                        result.push(*mask_char);
                    }
                }
                MaskResult::Replaced(result)
            }
            Self::Hash => {
                let hash = simple_hash(value.as_bytes());
                MaskResult::Replaced(format!("{hash:016x}"))
            }
            Self::Tokenize { prefix } => {
                let hash = simple_hash(value.as_bytes());
                MaskResult::Replaced(format!("{prefix}{hash:012x}"))
            }
            Self::Null => MaskResult::Null,
            Self::Redact => MaskResult::Removed,
            Self::Constant { value: cval } => MaskResult::Replaced(cval.clone()),
            Self::TruncateAfter { keep } => {
                let truncated: String = value.chars().take(*keep).collect();
                MaskResult::Replaced(truncated)
            }
        }
    }
}

/// Result of applying a mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskResult {
    /// Field value replaced with this string.
    Replaced(String),
    /// Field set to null.
    Null,
    /// Field removed entirely.
    Removed,
}

/// Simple non-cryptographic hash for deterministic masking.
fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// PII category for auto-detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiiCategory {
    Email,
    Phone,
    Ssn,
    CreditCard,
    IpAddress,
    Name,
    Address,
}

impl PiiCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Phone => "phone",
            Self::Ssn => "ssn",
            Self::CreditCard => "credit_card",
            Self::IpAddress => "ip_address",
            Self::Name => "name",
            Self::Address => "address",
        }
    }

    /// Detect PII category from a string value using simple heuristics.
    pub fn detect(value: &str) -> Vec<PiiCategory> {
        let mut detected = Vec::new();
        let trimmed = value.trim();

        // Email: contains @ and a dot after @
        if trimmed.contains('@') && trimmed.split('@').nth(1).map_or(false, |d| d.contains('.')) {
            detected.push(PiiCategory::Email);
        }

        // SSN: XXX-XX-XXXX pattern
        if is_ssn_pattern(trimmed) {
            detected.push(PiiCategory::Ssn);
        }

        // Credit card: 13-19 digits (possibly with spaces/dashes)
        let digits_only: String = trimmed.chars().filter(|c| c.is_ascii_digit()).collect();
        if (13..=19).contains(&digits_only.len()) && luhn_check(&digits_only) {
            detected.push(PiiCategory::CreditCard);
        }

        // Phone: starts with + and has 10-15 digits, or 10 digits with dashes/parens
        let phone_digits: String = trimmed.chars().filter(|c| c.is_ascii_digit()).collect();
        if (10..=15).contains(&phone_digits.len())
            && (trimmed.starts_with('+')
                || trimmed.starts_with('(')
                || phone_digits.len() == 10)
            && !detected.contains(&PiiCategory::CreditCard)
        {
            detected.push(PiiCategory::Phone);
        }

        // IP address: X.X.X.X
        if is_ipv4(trimmed) {
            detected.push(PiiCategory::IpAddress);
        }

        detected
    }
}

fn is_ssn_pattern(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    parts[0].len() == 3
        && parts[1].len() == 2
        && parts[2].len() == 4
        && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
}

fn luhn_check(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut double = false;
    for ch in digits.chars().rev() {
        let d = match ch.to_digit(10) {
            Some(d) => d,
            None => return false,
        };
        let val = if double {
            let doubled = d * 2;
            if doubled > 9 { doubled - 9 } else { doubled }
        } else {
            d
        };
        sum += val;
        double = !double;
    }
    sum % 10 == 0
}

fn is_ipv4(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    parts.iter().all(|p| p.parse::<u8>().is_ok())
}

/// A masking rule: which fields to mask and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskingRule {
    /// Unique rule ID.
    pub id: String,
    /// Dot-separated field path (e.g., "user.email", "payment.card_number").
    pub field_path: String,
    /// The masking strategy.
    pub strategy: MaskStrategy,
    /// Roles that see the unmasked value (bypass masking).
    pub exempt_roles: Vec<String>,
    /// Description.
    pub description: String,
    /// Whether PII auto-detection applies to this field.
    pub pii_category: Option<PiiCategory>,
}

impl MaskingRule {
    pub fn new(id: &str, field_path: &str, strategy: MaskStrategy) -> Self {
        Self {
            id: id.to_string(),
            field_path: field_path.to_string(),
            strategy,
            exempt_roles: Vec::new(),
            description: String::new(),
            pii_category: None,
        }
    }

    pub fn with_exempt_role(mut self, role: &str) -> Self {
        self.exempt_roles.push(role.to_string());
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_pii_category(mut self, cat: PiiCategory) -> Self {
        self.pii_category = Some(cat);
        self
    }

    /// Check if a role is exempt from this rule.
    pub fn is_exempt(&self, role: &str) -> bool {
        self.exempt_roles.iter().any(|r| r == role)
    }
}

/// Record of a mask operation for auditing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskAuditEntry {
    /// Field that was masked.
    pub field_path: String,
    /// Rule that was applied.
    pub rule_id: String,
    /// Strategy used.
    pub strategy_name: String,
    /// Whether the caller was exempt.
    pub was_exempt: bool,
    /// Timestamp (epoch millis).
    pub timestamp_ms: u64,
    /// Role of the caller.
    pub caller_role: String,
}

/// The masking engine.
pub struct MaskingEngine {
    rules: Vec<MaskingRule>,
    /// Audit log.
    audit_log: Vec<MaskAuditEntry>,
    /// Whether to auto-detect PII even without explicit rules.
    pub auto_detect_pii: bool,
    /// Default strategy for auto-detected PII.
    pub default_pii_strategy: MaskStrategy,
}

impl MaskingEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            audit_log: Vec::new(),
            auto_detect_pii: false,
            default_pii_strategy: MaskStrategy::Full { replacement: "[REDACTED]".to_string() },
        }
    }

    /// Add a masking rule.
    pub fn add_rule(&mut self, rule: MaskingRule) -> Result<(), MaskingError> {
        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(MaskingError::DuplicateRule(rule.id));
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Remove a masking rule by ID.
    pub fn remove_rule(&mut self, id: &str) -> Result<MaskingRule, MaskingError> {
        let idx = self
            .rules
            .iter()
            .position(|r| r.id == id)
            .ok_or_else(|| MaskingError::RuleNotFound(id.to_string()))?;
        Ok(self.rules.remove(idx))
    }

    /// Get all rules.
    pub fn rules(&self) -> &[MaskingRule] {
        &self.rules
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Mask a JSON value in-place based on rules and the caller's role.
    pub fn mask_value(
        &mut self,
        value: &mut Value,
        caller_role: &str,
        timestamp_ms: u64,
    ) {
        // Collect rules and their data to avoid borrow issues.
        let rule_data: Vec<(String, String, MaskStrategy, bool)> = self
            .rules
            .iter()
            .map(|r| {
                let exempt = r.is_exempt(caller_role);
                (r.id.clone(), r.field_path.clone(), r.strategy.clone(), exempt)
            })
            .collect();

        for (rule_id, field_path, strategy, exempt) in &rule_data {
            let strategy_name = strategy_label(strategy);
            self.audit_log.push(MaskAuditEntry {
                field_path: field_path.clone(),
                rule_id: rule_id.clone(),
                strategy_name: strategy_name.clone(),
                was_exempt: *exempt,
                timestamp_ms,
                caller_role: caller_role.to_string(),
            });

            if *exempt {
                continue;
            }

            apply_mask_at_path(value, field_path, strategy);
        }
    }

    /// Mask a single string value using a specific strategy.
    pub fn mask_string(&self, value: &str, strategy: &MaskStrategy) -> MaskResult {
        strategy.apply(value)
    }

    /// Detect PII in a JSON value and return (path, categories) pairs.
    pub fn detect_pii(&self, value: &Value) -> Vec<(String, Vec<PiiCategory>)> {
        let mut results = Vec::new();
        detect_pii_recursive(value, "", &mut results);
        results
    }

    /// Mask all detected PII in a JSON value using the default PII strategy.
    pub fn mask_detected_pii(&mut self, value: &mut Value, caller_role: &str, timestamp_ms: u64) {
        let detections = self.detect_pii(value);
        let strategy = self.default_pii_strategy.clone();
        for (path, categories) in detections {
            if categories.is_empty() {
                continue;
            }
            let strategy_name = strategy_label(&strategy);
            self.audit_log.push(MaskAuditEntry {
                field_path: path.clone(),
                rule_id: format!("auto-pii-{}", categories[0].as_str()),
                strategy_name,
                was_exempt: false,
                timestamp_ms,
                caller_role: caller_role.to_string(),
            });
            apply_mask_at_path(value, &path, &strategy);
        }
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &[MaskAuditEntry] {
        &self.audit_log
    }

    /// Clear the audit log.
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }
}

impl Default for MaskingEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn strategy_label(s: &MaskStrategy) -> String {
    match s {
        MaskStrategy::Full { .. } => "full".to_string(),
        MaskStrategy::Partial { .. } => "partial".to_string(),
        MaskStrategy::Hash => "hash".to_string(),
        MaskStrategy::Tokenize { .. } => "tokenize".to_string(),
        MaskStrategy::Null => "null".to_string(),
        MaskStrategy::Redact => "redact".to_string(),
        MaskStrategy::Constant { .. } => "constant".to_string(),
        MaskStrategy::TruncateAfter { .. } => "truncate".to_string(),
    }
}

/// Apply a masking strategy to a field at a dot-separated path in a JSON value.
fn apply_mask_at_path(value: &mut Value, path: &str, strategy: &MaskStrategy) {
    let parts: Vec<&str> = path.split('.').collect();
    apply_mask_recursive(value, &parts, strategy);
}

fn apply_mask_recursive(value: &mut Value, parts: &[&str], strategy: &MaskStrategy) {
    if parts.is_empty() {
        return;
    }
    if parts.len() == 1 {
        if let Value::Object(map) = value {
            let key = parts[0];
            if let Some(field_val) = map.get(key) {
                let str_val = match field_val {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let result = strategy.apply(&str_val);
                match result {
                    MaskResult::Replaced(s) => {
                        map.insert(key.to_string(), Value::String(s));
                    }
                    MaskResult::Null => {
                        map.insert(key.to_string(), Value::Null);
                    }
                    MaskResult::Removed => {
                        map.remove(key);
                    }
                }
            }
        }
        return;
    }
    if let Value::Object(map) = value {
        let key = parts[0];
        if let Some(child) = map.get_mut(key) {
            apply_mask_recursive(child, &parts[1..], strategy);
        }
    }
}

fn detect_pii_recursive(value: &Value, path: &str, results: &mut Vec<(String, Vec<PiiCategory>)>) {
    match value {
        Value::String(s) => {
            let categories = PiiCategory::detect(s);
            if !categories.is_empty() {
                results.push((path.to_string(), categories));
            }
        }
        Value::Object(map) => {
            for (key, val) in map {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                detect_pii_recursive(val, &child_path, results);
            }
        }
        Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                let child_path = format!("{path}[{i}]");
                detect_pii_recursive(val, &child_path, results);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_full_mask() {
        let s = MaskStrategy::Full { replacement: "***".to_string() };
        assert_eq!(s.apply("hello"), MaskResult::Replaced("***".to_string()));
    }

    #[test]
    fn test_partial_mask() {
        let s = MaskStrategy::Partial { show_first: 2, show_last: 2, mask_char: '*' };
        assert_eq!(s.apply("abcdefgh"), MaskResult::Replaced("ab****gh".to_string()));
    }

    #[test]
    fn test_partial_mask_short_value() {
        let s = MaskStrategy::Partial { show_first: 3, show_last: 3, mask_char: '*' };
        assert_eq!(s.apply("abc"), MaskResult::Replaced("***".to_string()));
    }

    #[test]
    fn test_hash_mask() {
        let s = MaskStrategy::Hash;
        let r1 = s.apply("test@example.com");
        let r2 = s.apply("test@example.com");
        assert_eq!(r1, r2); // deterministic
        match r1 {
            MaskResult::Replaced(h) => assert_eq!(h.len(), 16),
            _ => panic!("expected Replaced"),
        }
    }

    #[test]
    fn test_tokenize_mask() {
        let s = MaskStrategy::Tokenize { prefix: "tok_".to_string() };
        let r = s.apply("sensitive-data");
        match r {
            MaskResult::Replaced(t) => assert!(t.starts_with("tok_")),
            _ => panic!("expected Replaced"),
        }
    }

    #[test]
    fn test_null_mask() {
        assert_eq!(MaskStrategy::Null.apply("anything"), MaskResult::Null);
    }

    #[test]
    fn test_redact_mask() {
        assert_eq!(MaskStrategy::Redact.apply("anything"), MaskResult::Removed);
    }

    #[test]
    fn test_constant_mask() {
        let s = MaskStrategy::Constant { value: "N/A".to_string() };
        assert_eq!(s.apply("secret"), MaskResult::Replaced("N/A".to_string()));
    }

    #[test]
    fn test_truncate_mask() {
        let s = MaskStrategy::TruncateAfter { keep: 4 };
        assert_eq!(s.apply("abcdefgh"), MaskResult::Replaced("abcd".to_string()));
    }

    #[test]
    fn test_pii_detect_email() {
        let cats = PiiCategory::detect("user@example.com");
        assert!(cats.contains(&PiiCategory::Email));
    }

    #[test]
    fn test_pii_detect_ssn() {
        let cats = PiiCategory::detect("123-45-6789");
        assert!(cats.contains(&PiiCategory::Ssn));
    }

    #[test]
    fn test_pii_detect_credit_card() {
        // Valid Luhn: 4111111111111111
        let cats = PiiCategory::detect("4111111111111111");
        assert!(cats.contains(&PiiCategory::CreditCard));
    }

    #[test]
    fn test_pii_detect_ipv4() {
        let cats = PiiCategory::detect("192.168.1.1");
        assert!(cats.contains(&PiiCategory::IpAddress));
    }

    #[test]
    fn test_pii_no_false_positive() {
        let cats = PiiCategory::detect("hello world");
        assert!(cats.is_empty());
    }

    #[test]
    fn test_mask_json_value() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new(
                "mask-email",
                "user.email",
                MaskStrategy::Full { replacement: "[MASKED]".to_string() },
            ))
            .unwrap();

        let mut val = json!({
            "user": {
                "name": "Alice",
                "email": "alice@example.com"
            }
        });
        engine.mask_value(&mut val, "viewer", 1000);
        assert_eq!(val["user"]["email"], "[MASKED]");
        assert_eq!(val["user"]["name"], "Alice");
    }

    #[test]
    fn test_mask_exempt_role() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(
                MaskingRule::new(
                    "mask-email",
                    "email",
                    MaskStrategy::Full { replacement: "[MASKED]".to_string() },
                )
                .with_exempt_role("admin"),
            )
            .unwrap();

        let mut val = json!({"email": "alice@example.com"});
        engine.mask_value(&mut val, "admin", 1000);
        assert_eq!(val["email"], "alice@example.com");

        let mut val2 = json!({"email": "alice@example.com"});
        engine.mask_value(&mut val2, "viewer", 2000);
        assert_eq!(val2["email"], "[MASKED]");
    }

    #[test]
    fn test_mask_redact_removes_field() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new("redact-ssn", "ssn", MaskStrategy::Redact))
            .unwrap();
        let mut val = json!({"ssn": "123-45-6789", "name": "Bob"});
        engine.mask_value(&mut val, "viewer", 1000);
        assert!(val.get("ssn").is_none());
        assert_eq!(val["name"], "Bob");
    }

    #[test]
    fn test_mask_null_strategy() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new("null-field", "secret", MaskStrategy::Null))
            .unwrap();
        let mut val = json!({"secret": "hidden"});
        engine.mask_value(&mut val, "viewer", 1000);
        assert!(val["secret"].is_null());
    }

    #[test]
    fn test_duplicate_rule() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new("r1", "f", MaskStrategy::Null))
            .unwrap();
        let err = engine
            .add_rule(MaskingRule::new("r1", "g", MaskStrategy::Null))
            .unwrap_err();
        assert_eq!(err, MaskingError::DuplicateRule("r1".into()));
    }

    #[test]
    fn test_remove_rule() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new("r1", "f", MaskStrategy::Null))
            .unwrap();
        engine.remove_rule("r1").unwrap();
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_audit_log() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new(
                "r1",
                "email",
                MaskStrategy::Full { replacement: "***".to_string() },
            ))
            .unwrap();
        let mut val = json!({"email": "test@test.com"});
        engine.mask_value(&mut val, "user", 5000);
        let log = engine.audit_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].rule_id, "r1");
        assert_eq!(log[0].caller_role, "user");
        assert!(!log[0].was_exempt);
    }

    #[test]
    fn test_audit_log_exempt() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(
                MaskingRule::new("r1", "email", MaskStrategy::Null)
                    .with_exempt_role("admin"),
            )
            .unwrap();
        let mut val = json!({"email": "a@b.com"});
        engine.mask_value(&mut val, "admin", 1000);
        assert!(engine.audit_log()[0].was_exempt);
    }

    #[test]
    fn test_detect_pii_in_json() {
        let engine = MaskingEngine::new();
        let val = json!({
            "contact": {
                "email": "user@example.com",
                "phone": "+12025551234",
                "name": "Alice"
            }
        });
        let detections = engine.detect_pii(&val);
        let paths: Vec<&str> = detections.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"contact.email"));
    }

    #[test]
    fn test_nested_path_masking() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new(
                "r1",
                "a.b.c",
                MaskStrategy::Full { replacement: "X".to_string() },
            ))
            .unwrap();
        let mut val = json!({"a": {"b": {"c": "secret", "d": "visible"}}});
        engine.mask_value(&mut val, "user", 1000);
        assert_eq!(val["a"]["b"]["c"], "X");
        assert_eq!(val["a"]["b"]["d"], "visible");
    }

    #[test]
    fn test_clear_audit_log() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new("r1", "f", MaskStrategy::Null))
            .unwrap();
        let mut val = json!({"f": "data"});
        engine.mask_value(&mut val, "user", 1000);
        assert_eq!(engine.audit_log().len(), 1);
        engine.clear_audit_log();
        assert!(engine.audit_log().is_empty());
    }

    #[test]
    fn test_luhn_validation() {
        assert!(luhn_check("4111111111111111")); // Valid Visa test card
        assert!(!luhn_check("4111111111111112")); // Invalid
    }

    #[test]
    fn test_hash_deterministic() {
        let s1 = MaskStrategy::Hash;
        let s2 = MaskStrategy::Hash;
        assert_eq!(s1.apply("same-input"), s2.apply("same-input"));
    }

    #[test]
    fn test_mask_nonexistent_field() {
        let mut engine = MaskingEngine::new();
        engine
            .add_rule(MaskingRule::new(
                "r1",
                "nonexistent",
                MaskStrategy::Full { replacement: "X".to_string() },
            ))
            .unwrap();
        let mut val = json!({"other": "data"});
        engine.mask_value(&mut val, "user", 1000);
        // Should not crash, value unchanged
        assert_eq!(val["other"], "data");
    }
}
