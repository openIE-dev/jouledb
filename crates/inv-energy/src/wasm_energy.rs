//! WASM function-level energy tracking and budget enforcement.
//!
//! Tracks energy consumption per WASM module and function invocation,
//! enforcing per-module energy budgets.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FunctionProfile
// ---------------------------------------------------------------------------

/// Energy profile for a single function within a WASM module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionProfile {
    /// Name of the WASM module containing this function.
    pub module: String,
    /// Name of the function being profiled.
    pub func_name: String,
    /// Cumulative energy consumed across all invocations (joules).
    pub total_joules: f64,
    /// Number of times this function has been invoked.
    pub call_count: u64,
}

impl FunctionProfile {
    /// Returns the average energy per invocation.
    ///
    /// If `call_count` is zero the result is `0.0`.
    pub fn avg_joules(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_joules / self.call_count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// WasmEnergyTracker
// ---------------------------------------------------------------------------

/// Collects per-function energy profiles keyed by `(module, func_name)`.
pub struct WasmEnergyTracker {
    profiles: HashMap<(String, String), FunctionProfile>,
}

impl WasmEnergyTracker {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Records an energy measurement for a specific function in a WASM module.
    pub fn record(&mut self, module: &str, func_name: &str, joules: f64) {
        let key = (module.to_string(), func_name.to_string());
        let profile = self.profiles.entry(key).or_insert_with(|| FunctionProfile {
            module: module.to_string(),
            func_name: func_name.to_string(),
            total_joules: 0.0,
            call_count: 0,
        });
        profile.total_joules += joules;
        profile.call_count += 1;
    }

    /// Returns the profile for a specific function, if recorded.
    pub fn get_function_profile(&self, module: &str, func_name: &str) -> Option<&FunctionProfile> {
        let key = (module.to_string(), func_name.to_string());
        self.profiles.get(&key)
    }

    /// Returns the total energy consumed by all functions in the given module.
    pub fn get_module_total(&self, module: &str) -> f64 {
        self.profiles
            .values()
            .filter(|p| p.module == module)
            .map(|p| p.total_joules)
            .sum()
    }

    /// Returns references to all recorded function profiles.
    pub fn list_profiles(&self) -> Vec<&FunctionProfile> {
        self.profiles.values().collect()
    }

    /// Sum of `total_joules` across all function profiles.
    pub fn total_energy(&self) -> f64 {
        self.profiles.values().map(|p| p.total_joules).sum()
    }

    /// Clears all recorded profiles.
    pub fn reset(&mut self) {
        self.profiles.clear();
    }
}

impl Default for WasmEnergyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BudgetViolation
// ---------------------------------------------------------------------------

/// Describes a module that has exceeded its energy budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetViolation {
    /// Module that exceeded its budget.
    pub module: String,
    /// Energy consumed so far (joules).
    pub consumed: f64,
    /// Configured budget limit (joules).
    pub budget: f64,
}

// ---------------------------------------------------------------------------
// WasmEnergyBudget
// ---------------------------------------------------------------------------

/// Manages per-module energy budgets and checks them against a
/// [`WasmEnergyTracker`].
pub struct WasmEnergyBudget {
    budgets: HashMap<String, f64>,
}

impl WasmEnergyBudget {
    /// Creates a budget manager with no budgets configured.
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
        }
    }

    /// Sets the maximum allowed energy (joules) for `module`.
    pub fn set_budget(&mut self, module: &str, max_joules: f64) {
        self.budgets.insert(module.to_string(), max_joules);
    }

    /// Checks whether `module` is within its budget.
    ///
    /// Returns `Ok(())` if no budget is set or consumption is within limits.
    /// Returns `Err(BudgetViolation)` when the budget is exceeded.
    pub fn check(&self, tracker: &WasmEnergyTracker, module: &str) -> Result<(), BudgetViolation> {
        if let Some(&budget) = self.budgets.get(module) {
            let consumed = tracker.get_module_total(module);
            if consumed > budget {
                return Err(BudgetViolation {
                    module: module.to_string(),
                    consumed,
                    budget,
                });
            }
        }
        Ok(())
    }

    /// Checks all budgeted modules and returns a list of violations.
    ///
    /// Modules without a configured budget are not checked.
    pub fn check_all(&self, tracker: &WasmEnergyTracker) -> Vec<BudgetViolation> {
        let mut violations = Vec::new();
        for (module, &budget) in &self.budgets {
            let consumed = tracker.get_module_total(module);
            if consumed > budget {
                violations.push(BudgetViolation {
                    module: module.clone(),
                    consumed,
                    budget,
                });
            }
        }
        violations
    }
}

impl Default for WasmEnergyBudget {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_function() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("mod_a", "compute", 2.5);

        let profile = tracker
            .get_function_profile("mod_a", "compute")
            .expect("profile should exist");
        assert_eq!(profile.module, "mod_a");
        assert_eq!(profile.func_name, "compute");
        assert!((profile.total_joules - 2.5).abs() < 1e-10);
        assert_eq!(profile.call_count, 1);
    }

    #[test]
    fn record_multiple_calls() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("mod_a", "hash", 1.0);
        tracker.record("mod_a", "hash", 2.0);
        tracker.record("mod_a", "hash", 3.0);

        let profile = tracker.get_function_profile("mod_a", "hash").unwrap();
        assert!((profile.total_joules - 6.0).abs() < 1e-10);
        assert_eq!(profile.call_count, 3);
    }

    #[test]
    fn get_module_total() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("mod_a", "f1", 1.0);
        tracker.record("mod_a", "f2", 2.0);
        tracker.record("mod_b", "f1", 10.0);

        assert!((tracker.get_module_total("mod_a") - 3.0).abs() < 1e-10);
        assert!((tracker.get_module_total("mod_b") - 10.0).abs() < 1e-10);
        assert!((tracker.get_module_total("mod_c")).abs() < 1e-10);
    }

    #[test]
    fn list_profiles() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("m1", "f1", 1.0);
        tracker.record("m1", "f2", 2.0);
        tracker.record("m2", "f1", 3.0);

        assert_eq!(tracker.list_profiles().len(), 3);
    }

    #[test]
    fn total_energy() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("m", "a", 5.0);
        tracker.record("m", "b", 10.0);
        tracker.record("n", "a", 15.0);

        assert!((tracker.total_energy() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn reset_tracker() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("m", "f", 1.0);
        assert!(!tracker.list_profiles().is_empty());

        tracker.reset();
        assert!(tracker.list_profiles().is_empty());
        assert!((tracker.total_energy()).abs() < 1e-10);
    }

    #[test]
    fn budget_within_limit() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("mod_a", "f1", 3.0);

        let mut budget = WasmEnergyBudget::new();
        budget.set_budget("mod_a", 10.0);

        assert!(budget.check(&tracker, "mod_a").is_ok());
    }

    #[test]
    fn budget_exceeded() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("mod_a", "f1", 12.0);

        let mut budget = WasmEnergyBudget::new();
        budget.set_budget("mod_a", 10.0);

        let result = budget.check(&tracker, "mod_a");
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.module, "mod_a");
        assert!((violation.consumed - 12.0).abs() < 1e-10);
        assert!((violation.budget - 10.0).abs() < 1e-10);
    }

    #[test]
    fn check_all_budgets() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("ok_mod", "f", 3.0);
        tracker.record("bad_mod", "f", 20.0);

        let mut budget = WasmEnergyBudget::new();
        budget.set_budget("ok_mod", 10.0);
        budget.set_budget("bad_mod", 5.0);

        let violations = budget.check_all(&tracker);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].module, "bad_mod");
    }

    #[test]
    fn no_budget_no_violation() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("unbudgeted", "f", 999.0);

        let budget = WasmEnergyBudget::new();
        assert!(budget.check(&tracker, "unbudgeted").is_ok());
        assert!(budget.check_all(&tracker).is_empty());
    }

    #[test]
    fn function_avg() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("m", "f", 4.0);
        tracker.record("m", "f", 6.0);

        let profile = tracker.get_function_profile("m", "f").unwrap();
        assert!((profile.avg_joules() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn profile_serialization() {
        let mut tracker = WasmEnergyTracker::new();
        tracker.record("mod_s", "func_s", 7.5);
        tracker.record("mod_s", "func_s", 2.5);

        let profile = tracker.get_function_profile("mod_s", "func_s").unwrap();
        let json = serde_json::to_string(profile).expect("serialize");
        let deserialized: FunctionProfile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.module, "mod_s");
        assert_eq!(deserialized.func_name, "func_s");
        assert!((deserialized.total_joules - 10.0).abs() < 1e-10);
        assert_eq!(deserialized.call_count, 2);
    }
}
