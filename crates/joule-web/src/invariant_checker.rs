//! Runtime invariant checking — invariant definitions with predicates,
//! check points, violation reporting, invariant groups, enabled/disabled
//! invariants, violation history, and assert-like API with rich context.

use std::collections::HashMap;

// ── Invariant Severity ───────────────────────────────────────────

/// How critical an invariant violation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InvariantSeverity {
    /// Informational — log but do not halt.
    Info,
    /// Warning — something unexpected but recoverable.
    Warning,
    /// Error — the system is in a bad state.
    Error,
    /// Critical — the system must halt.
    Critical,
}

// ── Violation ────────────────────────────────────────────────────

/// A record of a single invariant violation.
#[derive(Debug, Clone)]
pub struct Violation {
    pub id: u64,
    pub invariant_name: String,
    pub group: Option<String>,
    pub severity: InvariantSeverity,
    pub message: String,
    pub context: HashMap<String, String>,
    pub timestamp_us: u64,
    pub checkpoint: Option<String>,
}

impl Violation {
    /// Short summary: "invariant_name: message".
    pub fn summary(&self) -> String {
        format!("{}: {}", self.invariant_name, self.message)
    }
}

// ── Invariant Definition ─────────────────────────────────────────

/// A named invariant with a predicate function.
pub struct InvariantDef {
    pub name: String,
    pub description: String,
    pub group: Option<String>,
    pub severity: InvariantSeverity,
    pub enabled: bool,
    /// The predicate: returns Ok(()) if the invariant holds,
    /// or Err(message) if violated.
    predicate: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
}

impl InvariantDef {
    pub fn new<F>(name: &str, description: &str, severity: InvariantSeverity, predicate: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            group: None,
            severity,
            enabled: true,
            predicate: Box::new(predicate),
        }
    }

    pub fn with_group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }

    /// Check the invariant. Returns Ok(()) or Err(message).
    pub fn check(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        (self.predicate)()
    }
}

impl std::fmt::Debug for InvariantDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvariantDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("group", &self.group)
            .field("severity", &self.severity)
            .field("enabled", &self.enabled)
            .finish()
    }
}

// ── Check Result ─────────────────────────────────────────────────

/// Result of checking one invariant.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub invariant_name: String,
    pub passed: bool,
    pub message: Option<String>,
    pub severity: InvariantSeverity,
}

// ── Check Point Result ───────────────────────────────────────────

/// Result of running all invariants at a checkpoint.
#[derive(Debug, Clone)]
pub struct CheckPointResult {
    pub checkpoint: String,
    pub timestamp_us: u64,
    pub results: Vec<CheckResult>,
    pub total_checked: usize,
    pub total_passed: usize,
    pub total_failed: usize,
}

impl CheckPointResult {
    /// True if all invariants passed.
    pub fn all_passed(&self) -> bool {
        self.total_failed == 0
    }

    /// Get only the failed results.
    pub fn failures(&self) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Highest severity among failures, if any.
    pub fn max_severity(&self) -> Option<InvariantSeverity> {
        self.results
            .iter()
            .filter(|r| !r.passed)
            .map(|r| r.severity)
            .max()
    }
}

// ── Invariant Checker ────────────────────────────────────────────

/// Runtime invariant checker with grouping, history, and rich context.
pub struct InvariantChecker {
    invariants: Vec<InvariantDef>,
    violations: Vec<Violation>,
    next_violation_id: u64,
    checkpoint_history: Vec<CheckPointResult>,
}

impl InvariantChecker {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
            violations: Vec::new(),
            next_violation_id: 0,
            checkpoint_history: Vec::new(),
        }
    }

    /// Register an invariant.
    pub fn register(&mut self, def: InvariantDef) {
        self.invariants.push(def);
    }

    /// Number of registered invariants.
    pub fn invariant_count(&self) -> usize {
        self.invariants.len()
    }

    /// Number of enabled invariants.
    pub fn enabled_count(&self) -> usize {
        self.invariants.iter().filter(|i| i.enabled).count()
    }

    /// Enable an invariant by name. Returns true if found.
    pub fn enable(&mut self, name: &str) -> bool {
        for inv in &mut self.invariants {
            if inv.name == name {
                inv.enabled = true;
                return true;
            }
        }
        false
    }

    /// Disable an invariant by name. Returns true if found.
    pub fn disable(&mut self, name: &str) -> bool {
        for inv in &mut self.invariants {
            if inv.name == name {
                inv.enabled = false;
                return true;
            }
        }
        false
    }

    /// Enable all invariants in a group.
    pub fn enable_group(&mut self, group: &str) {
        for inv in &mut self.invariants {
            if inv.group.as_deref() == Some(group) {
                inv.enabled = true;
            }
        }
    }

    /// Disable all invariants in a group.
    pub fn disable_group(&mut self, group: &str) {
        for inv in &mut self.invariants {
            if inv.group.as_deref() == Some(group) {
                inv.enabled = false;
            }
        }
    }

    /// Enable all invariants.
    pub fn enable_all(&mut self) {
        for inv in &mut self.invariants {
            inv.enabled = true;
        }
    }

    /// Disable all invariants.
    pub fn disable_all(&mut self) {
        for inv in &mut self.invariants {
            inv.enabled = false;
        }
    }

    /// Check a single invariant by name. Returns the check result.
    pub fn check_one(&mut self, name: &str, timestamp_us: u64) -> Option<CheckResult> {
        // Find the invariant and check it, collecting the result first
        let check_data: Option<(String, InvariantSeverity, Option<String>, Result<(), String>)> = self
            .invariants
            .iter()
            .find(|i| i.name == name)
            .map(|inv| {
                let result = inv.check();
                let group = inv.group.clone();
                (inv.name.clone(), inv.severity, group, result)
            });

        match check_data {
            Some((inv_name, severity, group, result)) => {
                let (passed, message) = match &result {
                    Ok(()) => (true, None),
                    Err(msg) => (false, Some(msg.clone())),
                };

                if !passed {
                    let violation = Violation {
                        id: self.next_violation_id,
                        invariant_name: inv_name.clone(),
                        group,
                        severity,
                        message: message.clone().unwrap_or_default(),
                        context: HashMap::new(),
                        timestamp_us,
                        checkpoint: None,
                    };
                    self.next_violation_id += 1;
                    self.violations.push(violation);
                }

                Some(CheckResult {
                    invariant_name: inv_name,
                    passed,
                    message,
                    severity,
                })
            }
            None => None,
        }
    }

    /// Run all enabled invariants at a named checkpoint.
    pub fn checkpoint(&mut self, name: &str, timestamp_us: u64) -> CheckPointResult {
        // Collect check data for all enabled invariants
        let check_data: Vec<(String, InvariantSeverity, Option<String>, Result<(), String>)> = self
            .invariants
            .iter()
            .filter(|i| i.enabled)
            .map(|inv| {
                let result = inv.check();
                (inv.name.clone(), inv.severity, inv.group.clone(), result)
            })
            .collect();

        let mut results = Vec::new();
        let mut total_passed = 0;
        let mut total_failed = 0;

        for (inv_name, severity, group, result) in check_data {
            let (passed, message) = match &result {
                Ok(()) => (true, None),
                Err(msg) => (false, Some(msg.clone())),
            };

            if passed {
                total_passed += 1;
            } else {
                total_failed += 1;
                let violation = Violation {
                    id: self.next_violation_id,
                    invariant_name: inv_name.clone(),
                    group,
                    severity,
                    message: message.clone().unwrap_or_default(),
                    context: HashMap::new(),
                    timestamp_us,
                    checkpoint: Some(name.to_string()),
                };
                self.next_violation_id += 1;
                self.violations.push(violation);
            }

            results.push(CheckResult {
                invariant_name: inv_name,
                passed,
                message,
                severity,
            });
        }

        let total_checked = results.len();
        let cp_result = CheckPointResult {
            checkpoint: name.to_string(),
            timestamp_us,
            results,
            total_checked,
            total_passed,
            total_failed,
        };

        self.checkpoint_history.push(cp_result.clone());
        cp_result
    }

    /// Run only invariants in a specific group.
    pub fn check_group(&mut self, group: &str, timestamp_us: u64) -> CheckPointResult {
        let check_data: Vec<(String, InvariantSeverity, Option<String>, Result<(), String>)> = self
            .invariants
            .iter()
            .filter(|i| i.enabled && i.group.as_deref() == Some(group))
            .map(|inv| {
                let result = inv.check();
                (inv.name.clone(), inv.severity, inv.group.clone(), result)
            })
            .collect();

        let mut results = Vec::new();
        let mut total_passed = 0;
        let mut total_failed = 0;

        for (inv_name, severity, inv_group, result) in check_data {
            let (passed, message) = match &result {
                Ok(()) => (true, None),
                Err(msg) => (false, Some(msg.clone())),
            };

            if passed {
                total_passed += 1;
            } else {
                total_failed += 1;
                let violation = Violation {
                    id: self.next_violation_id,
                    invariant_name: inv_name.clone(),
                    group: inv_group,
                    severity,
                    message: message.clone().unwrap_or_default(),
                    context: HashMap::new(),
                    timestamp_us,
                    checkpoint: Some(group.to_string()),
                };
                self.next_violation_id += 1;
                self.violations.push(violation);
            }

            results.push(CheckResult {
                invariant_name: inv_name,
                passed,
                message,
                severity,
            });
        }

        let total_checked = results.len();
        CheckPointResult {
            checkpoint: group.to_string(),
            timestamp_us,
            results,
            total_checked,
            total_passed,
            total_failed,
        }
    }

    /// Get all violations recorded so far.
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    /// Get violation count.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Get violations filtered by severity.
    pub fn violations_by_severity(&self, min_severity: InvariantSeverity) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.severity >= min_severity)
            .collect()
    }

    /// Get violations for a specific invariant.
    pub fn violations_for(&self, name: &str) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.invariant_name == name)
            .collect()
    }

    /// Get the checkpoint history.
    pub fn checkpoint_history(&self) -> &[CheckPointResult] {
        &self.checkpoint_history
    }

    /// Clear the violation history.
    pub fn clear_violations(&mut self) {
        self.violations.clear();
    }

    /// Assert-like API: check an invariant and return a rich error.
    pub fn assert_invariant(
        &mut self,
        name: &str,
        timestamp_us: u64,
        context: HashMap<String, String>,
    ) -> Result<(), Violation> {
        let check_data: Option<(String, InvariantSeverity, Option<String>, Result<(), String>)> = self
            .invariants
            .iter()
            .find(|i| i.name == name)
            .map(|inv| {
                let result = inv.check();
                (inv.name.clone(), inv.severity, inv.group.clone(), result)
            });

        match check_data {
            Some((inv_name, severity, group, Err(msg))) => {
                let violation = Violation {
                    id: self.next_violation_id,
                    invariant_name: inv_name,
                    group,
                    severity,
                    message: msg,
                    context,
                    timestamp_us,
                    checkpoint: None,
                };
                self.next_violation_id += 1;
                self.violations.push(violation.clone());
                Err(violation)
            }
            Some(_) => Ok(()),
            None => {
                let violation = Violation {
                    id: self.next_violation_id,
                    invariant_name: name.to_string(),
                    group: None,
                    severity: InvariantSeverity::Error,
                    message: format!("unknown invariant: {}", name),
                    context,
                    timestamp_us,
                    checkpoint: None,
                };
                self.next_violation_id += 1;
                self.violations.push(violation.clone());
                Err(violation)
            }
        }
    }

    /// Get invariant names in a group.
    pub fn group_members(&self, group: &str) -> Vec<&str> {
        self.invariants
            .iter()
            .filter(|i| i.group.as_deref() == Some(group))
            .map(|i| i.name.as_str())
            .collect()
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_register_invariant() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "positive_balance",
            "Balance must be positive",
            InvariantSeverity::Error,
            || Ok(()),
        ));
        assert_eq!(checker.invariant_count(), 1);
    }

    #[test]
    fn test_check_passing_invariant() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "always_ok",
            "Always passes",
            InvariantSeverity::Info,
            || Ok(()),
        ));
        let result = checker.check_one("always_ok", 1000).unwrap();
        assert!(result.passed);
        assert_eq!(checker.violation_count(), 0);
    }

    #[test]
    fn test_check_failing_invariant() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "always_fail",
            "Always fails",
            InvariantSeverity::Error,
            || Err("failed!".to_string()),
        ));
        let result = checker.check_one("always_fail", 1000).unwrap();
        assert!(!result.passed);
        assert_eq!(result.message.as_deref(), Some("failed!"));
        assert_eq!(checker.violation_count(), 1);
    }

    #[test]
    fn test_check_unknown_invariant() {
        let mut checker = InvariantChecker::new();
        let result = checker.check_one("nonexistent", 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_checkpoint_all_pass() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())));
        checker.register(InvariantDef::new("b", "", InvariantSeverity::Info, || Ok(())));

        let result = checker.checkpoint("init", 1000);
        assert!(result.all_passed());
        assert_eq!(result.total_checked, 2);
        assert_eq!(result.total_passed, 2);
        assert_eq!(result.total_failed, 0);
    }

    #[test]
    fn test_checkpoint_with_failure() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new("ok", "", InvariantSeverity::Info, || Ok(())));
        checker.register(InvariantDef::new(
            "bad",
            "",
            InvariantSeverity::Error,
            || Err("broken".to_string()),
        ));

        let result = checker.checkpoint("check", 1000);
        assert!(!result.all_passed());
        assert_eq!(result.total_failed, 1);
        assert_eq!(result.failures().len(), 1);
    }

    #[test]
    fn test_disable_invariant() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "flaky",
            "",
            InvariantSeverity::Warning,
            || Err("flaky".to_string()),
        ));
        checker.disable("flaky");
        assert_eq!(checker.enabled_count(), 0);

        let result = checker.checkpoint("check", 1000);
        assert!(result.all_passed());
        assert_eq!(result.total_checked, 0);
    }

    #[test]
    fn test_enable_invariant() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new("x", "", InvariantSeverity::Info, || Ok(())));
        checker.disable("x");
        assert_eq!(checker.enabled_count(), 0);
        checker.enable("x");
        assert_eq!(checker.enabled_count(), 1);
    }

    #[test]
    fn test_invariant_groups() {
        let mut checker = InvariantChecker::new();
        checker.register(
            InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())).with_group("core"),
        );
        checker.register(
            InvariantDef::new("b", "", InvariantSeverity::Info, || Ok(())).with_group("core"),
        );
        checker.register(
            InvariantDef::new("c", "", InvariantSeverity::Info, || Ok(())).with_group("optional"),
        );

        assert_eq!(checker.group_members("core").len(), 2);
        assert_eq!(checker.group_members("optional").len(), 1);
    }

    #[test]
    fn test_disable_group() {
        let mut checker = InvariantChecker::new();
        checker.register(
            InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())).with_group("g"),
        );
        checker.register(
            InvariantDef::new("b", "", InvariantSeverity::Info, || Ok(())).with_group("g"),
        );
        checker.register(InvariantDef::new("c", "", InvariantSeverity::Info, || Ok(())));

        checker.disable_group("g");
        assert_eq!(checker.enabled_count(), 1); // only "c"
    }

    #[test]
    fn test_enable_group() {
        let mut checker = InvariantChecker::new();
        checker.register(
            InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())).with_group("g"),
        );
        checker.disable_all();
        checker.enable_group("g");
        assert_eq!(checker.enabled_count(), 1);
    }

    #[test]
    fn test_check_group() {
        let mut checker = InvariantChecker::new();
        checker.register(
            InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())).with_group("core"),
        );
        checker.register(
            InvariantDef::new("b", "", InvariantSeverity::Info, || {
                Err("fail".to_string())
            })
            .with_group("optional"),
        );

        let result = checker.check_group("core", 1000);
        assert!(result.all_passed());
        assert_eq!(result.total_checked, 1);
    }

    #[test]
    fn test_violation_history() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "fail",
            "",
            InvariantSeverity::Error,
            || Err("bad".to_string()),
        ));
        checker.checkpoint("cp1", 1000);
        checker.checkpoint("cp2", 2000);
        assert_eq!(checker.violation_count(), 2);
        assert_eq!(checker.violations()[0].checkpoint.as_deref(), Some("cp1"));
        assert_eq!(checker.violations()[1].checkpoint.as_deref(), Some("cp2"));
    }

    #[test]
    fn test_violations_by_severity() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "warn",
            "",
            InvariantSeverity::Warning,
            || Err("w".to_string()),
        ));
        checker.register(InvariantDef::new(
            "err",
            "",
            InvariantSeverity::Error,
            || Err("e".to_string()),
        ));
        checker.checkpoint("check", 1000);

        let errors = checker.violations_by_severity(InvariantSeverity::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].invariant_name, "err");
    }

    #[test]
    fn test_violations_for_invariant() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "x",
            "",
            InvariantSeverity::Error,
            || Err("fail".to_string()),
        ));
        checker.checkpoint("cp1", 1000);
        checker.checkpoint("cp2", 2000);
        assert_eq!(checker.violations_for("x").len(), 2);
    }

    #[test]
    fn test_clear_violations() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "x",
            "",
            InvariantSeverity::Error,
            || Err("fail".to_string()),
        ));
        checker.checkpoint("cp", 1000);
        assert_eq!(checker.violation_count(), 1);
        checker.clear_violations();
        assert_eq!(checker.violation_count(), 0);
    }

    #[test]
    fn test_assert_invariant_pass() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new("ok", "", InvariantSeverity::Info, || Ok(())));
        let result = checker.assert_invariant("ok", 1000, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_invariant_fail_with_context() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "bad",
            "",
            InvariantSeverity::Critical,
            || Err("invariant violated".to_string()),
        ));
        let mut ctx = HashMap::new();
        ctx.insert("user_id".to_string(), "42".to_string());
        let result = checker.assert_invariant("bad", 1000, ctx);
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.severity, InvariantSeverity::Critical);
        assert_eq!(violation.context.get("user_id").unwrap(), "42");
    }

    #[test]
    fn test_assert_unknown_invariant() {
        let mut checker = InvariantChecker::new();
        let result = checker.assert_invariant("nope", 1000, HashMap::new());
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert!(violation.message.contains("unknown"));
    }

    #[test]
    fn test_violation_summary() {
        let v = Violation {
            id: 0,
            invariant_name: "positive_balance".to_string(),
            group: None,
            severity: InvariantSeverity::Error,
            message: "balance is -5".to_string(),
            context: HashMap::new(),
            timestamp_us: 1000,
            checkpoint: None,
        };
        assert_eq!(v.summary(), "positive_balance: balance is -5");
    }

    #[test]
    fn test_checkpoint_max_severity() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "w",
            "",
            InvariantSeverity::Warning,
            || Err("w".to_string()),
        ));
        checker.register(InvariantDef::new(
            "c",
            "",
            InvariantSeverity::Critical,
            || Err("c".to_string()),
        ));
        let result = checker.checkpoint("cp", 1000);
        assert_eq!(result.max_severity(), Some(InvariantSeverity::Critical));
    }

    #[test]
    fn test_dynamic_predicate() {
        let flag = Arc::new(AtomicBool::new(true));
        let flag_clone = flag.clone();
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new(
            "dynamic",
            "",
            InvariantSeverity::Error,
            move || {
                if flag_clone.load(Ordering::Relaxed) {
                    Ok(())
                } else {
                    Err("flag is false".to_string())
                }
            },
        ));

        let r1 = checker.check_one("dynamic", 1000).unwrap();
        assert!(r1.passed);

        flag.store(false, Ordering::Relaxed);
        let r2 = checker.check_one("dynamic", 2000).unwrap();
        assert!(!r2.passed);
    }

    #[test]
    fn test_checkpoint_history() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())));
        checker.checkpoint("cp1", 1000);
        checker.checkpoint("cp2", 2000);
        assert_eq!(checker.checkpoint_history().len(), 2);
        assert_eq!(checker.checkpoint_history()[0].checkpoint, "cp1");
    }

    #[test]
    fn test_enable_disable_all() {
        let mut checker = InvariantChecker::new();
        checker.register(InvariantDef::new("a", "", InvariantSeverity::Info, || Ok(())));
        checker.register(InvariantDef::new("b", "", InvariantSeverity::Info, || Ok(())));
        checker.disable_all();
        assert_eq!(checker.enabled_count(), 0);
        checker.enable_all();
        assert_eq!(checker.enabled_count(), 2);
    }

    #[test]
    fn test_empty_checker() {
        let mut checker = InvariantChecker::new();
        let result = checker.checkpoint("empty", 0);
        assert!(result.all_passed());
        assert_eq!(result.total_checked, 0);
    }
}
