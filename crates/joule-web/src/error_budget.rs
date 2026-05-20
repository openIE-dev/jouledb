//! Error budget tracking (SRE concept) — budget definition with target SLO,
//! budget consumption, remaining budget, alerts, burn rate calculation,
//! budget reset periods, and budget history.
//!
//! Pure Rust implementation of SRE error budgets for reliability tracking.
//! No external dependencies beyond the workspace set.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── SLO Target ──────────────────────────────────────────────────

/// A Service Level Objective target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloTarget {
    /// Name of the SLO (e.g. "availability", "latency_p99").
    pub name: String,
    /// Target as a ratio (e.g. 0.999 = 99.9%).
    pub target: f64,
    /// Rolling window in seconds for budget calculation.
    pub window_secs: u64,
}

impl SloTarget {
    pub fn new(name: impl Into<String>, target: f64, window_secs: u64) -> Self {
        Self {
            name: name.into(),
            target: target.clamp(0.0, 1.0),
            window_secs,
        }
    }

    /// Error budget as a ratio (e.g. 0.001 for 99.9% SLO).
    pub fn error_budget_ratio(&self) -> f64 {
        1.0 - self.target
    }

    /// Display the target as a percentage string.
    pub fn target_pct(&self) -> f64 {
        self.target * 100.0
    }
}

// ── Budget Status ───────────────────────────────────────────────

/// Current status of an error budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetStatus {
    /// Budget is healthy — plenty remaining.
    Healthy,
    /// Budget is being consumed faster than expected.
    Warning,
    /// Budget is nearly exhausted.
    Critical,
    /// Budget is exhausted.
    Exhausted,
}

impl BudgetStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Exhausted => "exhausted",
        }
    }
}

// ── Budget Alert ────────────────────────────────────────────────

/// An alert triggered by error budget state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    pub slo_name: String,
    pub status: BudgetStatus,
    pub remaining_pct: f64,
    pub burn_rate: f64,
    pub message: String,
}

// ── Budget Snapshot ─────────────────────────────────────────────

/// A point-in-time snapshot of the budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub total_requests: u64,
    pub total_errors: u64,
    pub error_rate: f64,
    pub budget_remaining_pct: f64,
    pub burn_rate: f64,
    pub status: BudgetStatus,
    pub window_elapsed_pct: f64,
}

// ── Error Budget ────────────────────────────────────────────────

/// Tracks the error budget for a single SLO.
#[derive(Debug, Clone)]
pub struct ErrorBudget {
    slo: SloTarget,
    total_requests: u64,
    total_errors: u64,
    /// Alert thresholds: (remaining_pct, status).
    alert_thresholds: Vec<(f64, BudgetStatus)>,
    /// Historical snapshots.
    history: Vec<BudgetSnapshot>,
    /// Maximum history entries to keep.
    max_history: usize,
    /// Window start time (epoch seconds or monotonic).
    window_start_secs: u64,
}

impl ErrorBudget {
    pub fn new(slo: SloTarget) -> Self {
        Self {
            slo,
            total_requests: 0,
            total_errors: 0,
            alert_thresholds: vec![
                (50.0, BudgetStatus::Warning),
                (10.0, BudgetStatus::Critical),
                (0.0, BudgetStatus::Exhausted),
            ],
            history: Vec::new(),
            max_history: 100,
            window_start_secs: 0,
        }
    }

    /// Set custom alert thresholds.
    pub fn with_thresholds(mut self, thresholds: Vec<(f64, BudgetStatus)>) -> Self {
        self.alert_thresholds = thresholds;
        self
    }

    /// Set the window start time.
    pub fn with_window_start(mut self, start_secs: u64) -> Self {
        self.window_start_secs = start_secs;
        self
    }

    /// Record a successful request.
    pub fn record_success(&mut self) {
        self.total_requests += 1;
    }

    /// Record a batch of successes.
    pub fn record_successes(&mut self, count: u64) {
        self.total_requests += count;
    }

    /// Record a failed request (consumes error budget).
    pub fn record_error(&mut self) {
        self.total_requests += 1;
        self.total_errors += 1;
    }

    /// Record a batch of errors.
    pub fn record_errors(&mut self, count: u64) {
        self.total_requests += count;
        self.total_errors += count;
    }

    /// Current error rate.
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_errors as f64 / self.total_requests as f64
    }

    /// Total error budget allowed (number of errors allowed given total requests).
    pub fn total_budget(&self) -> f64 {
        self.total_requests as f64 * self.slo.error_budget_ratio()
    }

    /// Errors that can still be "spent" before exceeding the SLO.
    pub fn remaining_budget(&self) -> f64 {
        self.total_budget() - self.total_errors as f64
    }

    /// Remaining budget as a percentage (0-100). Can go negative.
    pub fn remaining_pct(&self) -> f64 {
        let budget = self.total_budget();
        if budget <= 0.0 {
            if self.total_errors == 0 {
                return 100.0;
            }
            return 0.0;
        }
        (self.remaining_budget() / budget) * 100.0
    }

    /// Whether the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining_budget() < 1e-9 && self.total_requests > 0
    }

    /// Calculate burn rate: how fast is the budget being consumed relative
    /// to what's expected? A burn rate of 1.0 means consuming exactly
    /// at the SLO pace. >1 means faster. <1 means slower.
    pub fn burn_rate(&self, window_elapsed_secs: u64) -> f64 {
        if self.slo.window_secs == 0 || window_elapsed_secs == 0 || self.total_requests == 0 {
            return 0.0;
        }
        let window_fraction = window_elapsed_secs as f64 / self.slo.window_secs as f64;
        if window_fraction <= 0.0 {
            return 0.0;
        }
        let expected_budget_consumption = window_fraction; // Fraction of budget we "should" have consumed.
        let actual_budget_consumption = if self.total_budget() > 0.0 {
            self.total_errors as f64 / self.total_budget()
        } else {
            0.0
        };
        actual_budget_consumption / expected_budget_consumption
    }

    /// Time to budget exhaustion at current burn rate (seconds).
    pub fn time_to_exhaustion_secs(&self, window_elapsed_secs: u64) -> Option<f64> {
        let rate = self.burn_rate(window_elapsed_secs);
        if rate <= 0.0 || self.is_exhausted() {
            return None;
        }
        let remaining_fraction = self.remaining_pct() / 100.0;
        if remaining_fraction <= 0.0 {
            return None;
        }
        let remaining_window_secs = self.slo.window_secs.saturating_sub(window_elapsed_secs) as f64;
        Some(remaining_window_secs / rate)
    }

    /// Get current budget status.
    pub fn status(&self) -> BudgetStatus {
        if self.total_requests == 0 {
            return BudgetStatus::Healthy;
        }
        let remaining = self.remaining_pct();
        let mut sorted = self.alert_thresholds.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (threshold, status) in &sorted {
            if remaining <= *threshold + 1e-9 {
                return *status;
            }
        }
        BudgetStatus::Healthy
    }

    /// Check alerts based on current state.
    pub fn check_alerts(&self, window_elapsed_secs: u64) -> Option<BudgetAlert> {
        let status = self.status();
        if status == BudgetStatus::Healthy {
            return None;
        }
        Some(BudgetAlert {
            slo_name: self.slo.name.clone(),
            status,
            remaining_pct: self.remaining_pct(),
            burn_rate: self.burn_rate(window_elapsed_secs),
            message: format!(
                "SLO '{}' error budget at {:.1}% remaining (status: {})",
                self.slo.name,
                self.remaining_pct(),
                status.as_str()
            ),
        })
    }

    /// Take a snapshot and append to history.
    pub fn snapshot(&mut self, window_elapsed_secs: u64) -> BudgetSnapshot {
        let window_elapsed_pct = if self.slo.window_secs > 0 {
            (window_elapsed_secs as f64 / self.slo.window_secs as f64) * 100.0
        } else {
            0.0
        };
        let snap = BudgetSnapshot {
            total_requests: self.total_requests,
            total_errors: self.total_errors,
            error_rate: self.error_rate(),
            budget_remaining_pct: self.remaining_pct(),
            burn_rate: self.burn_rate(window_elapsed_secs),
            status: self.status(),
            window_elapsed_pct,
        };
        self.history.push(snap.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        snap
    }

    /// Reset the budget for a new window.
    pub fn reset(&mut self) {
        self.total_requests = 0;
        self.total_errors = 0;
    }

    pub fn slo(&self) -> &SloTarget {
        &self.slo
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    pub fn total_errors(&self) -> u64 {
        self.total_errors
    }

    pub fn history(&self) -> &[BudgetSnapshot] {
        &self.history
    }
}

// ── Error Budget Registry ───────────────────────────────────────

/// Manages multiple error budgets.
#[derive(Debug)]
pub struct ErrorBudgetRegistry {
    budgets: HashMap<String, ErrorBudget>,
}

impl ErrorBudgetRegistry {
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
        }
    }

    pub fn register(&mut self, budget: ErrorBudget) {
        self.budgets.insert(budget.slo().name.clone(), budget);
    }

    pub fn get(&self, name: &str) -> Option<&ErrorBudget> {
        self.budgets.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut ErrorBudget> {
        self.budgets.get_mut(name)
    }

    /// Check all budgets for alerts.
    pub fn check_all_alerts(&self, window_elapsed_secs: u64) -> Vec<BudgetAlert> {
        let mut alerts = Vec::new();
        for budget in self.budgets.values() {
            if let Some(alert) = budget.check_alerts(window_elapsed_secs) {
                alerts.push(alert);
            }
        }
        alerts.sort_by(|a, b| a.slo_name.cmp(&b.slo_name));
        alerts
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.budgets.keys().map(|k| k.as_str()).collect();
        names.sort();
        names
    }

    pub fn len(&self) -> usize {
        self.budgets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.budgets.is_empty()
    }
}

impl Default for ErrorBudgetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slo() -> SloTarget {
        SloTarget::new("availability", 0.999, 30 * 24 * 3600) // 99.9%, 30-day window
    }

    #[test]
    fn test_slo_target() {
        let slo = make_slo();
        assert_eq!(slo.name, "availability");
        assert!((slo.target - 0.999).abs() < f64::EPSILON);
        assert!((slo.error_budget_ratio() - 0.001).abs() < 1e-9);
        assert!((slo.target_pct() - 99.9).abs() < 0.01);
    }

    #[test]
    fn test_new_budget_healthy() {
        let budget = ErrorBudget::new(make_slo());
        assert_eq!(budget.total_requests(), 0);
        assert_eq!(budget.total_errors(), 0);
        assert_eq!(budget.status(), BudgetStatus::Healthy);
    }

    #[test]
    fn test_record_success() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_success();
        budget.record_success();
        assert_eq!(budget.total_requests(), 2);
        assert_eq!(budget.total_errors(), 0);
        assert_eq!(budget.error_rate(), 0.0);
    }

    #[test]
    fn test_record_error() {
        let mut budget = ErrorBudget::new(make_slo());
        for _ in 0..999 {
            budget.record_success();
        }
        budget.record_error();
        assert_eq!(budget.total_requests(), 1000);
        assert_eq!(budget.total_errors(), 1);
        assert!((budget.error_rate() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_remaining_budget() {
        let mut budget = ErrorBudget::new(make_slo());
        // SLO = 99.9%, so error budget = 0.1%.
        // With 10000 requests, we can have 10 errors.
        budget.record_successes(10000);
        assert!((budget.total_budget() - 10.0).abs() < 0.01);
        assert!((budget.remaining_budget() - 10.0).abs() < 0.01);
        assert!((budget.remaining_pct() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_budget_consumption() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9995);
        budget.record_errors(5);
        // Budget = 10000 * 0.001 = 10, used 5 => 50% remaining.
        assert!((budget.remaining_pct() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_budget_exhaustion() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9990);
        budget.record_errors(10);
        // Budget = 10000 * 0.001 = 10, used 10 => 0% remaining.
        assert!(budget.remaining_pct() <= 0.01);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_budget_over_exhaustion() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9985);
        budget.record_errors(15);
        // Budget = 10000 * 0.001 = 10, used 15 => negative.
        assert!(budget.remaining_pct() < 0.0);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_status_healthy() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(10000);
        assert_eq!(budget.status(), BudgetStatus::Healthy);
    }

    #[test]
    fn test_status_warning() {
        let mut budget = ErrorBudget::new(make_slo());
        // SLO 99.9%, budget = 10 errors per 10000.
        // To get to <50%, need >5 errors.
        budget.record_successes(9994);
        budget.record_errors(6); // 40% remaining.
        assert_eq!(budget.status(), BudgetStatus::Warning);
    }

    #[test]
    fn test_status_critical() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9991);
        budget.record_errors(9); // 10% remaining.
        assert_eq!(budget.status(), BudgetStatus::Critical);
    }

    #[test]
    fn test_status_exhausted() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9989);
        budget.record_errors(11); // Over budget.
        assert_eq!(budget.status(), BudgetStatus::Exhausted);
    }

    #[test]
    fn test_burn_rate() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9990);
        budget.record_errors(10);
        // All budget consumed. In 30 days we should use 1.0x.
        // If we consumed it all in 15 days, burn rate = 2.0.
        let half_window = 15 * 24 * 3600;
        let rate = budget.burn_rate(half_window);
        assert!((rate - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_burn_rate_zero_requests() {
        let budget = ErrorBudget::new(make_slo());
        assert_eq!(budget.burn_rate(1000), 0.0);
    }

    #[test]
    fn test_check_alerts_none_when_healthy() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(10000);
        assert!(budget.check_alerts(1000).is_none());
    }

    #[test]
    fn test_check_alerts_fires() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(9989);
        budget.record_errors(11);
        let alert = budget.check_alerts(1000).unwrap();
        assert_eq!(alert.status, BudgetStatus::Exhausted);
        assert!(alert.message.contains("availability"));
    }

    #[test]
    fn test_snapshot() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(10000);
        budget.record_errors(5);
        let snap = budget.snapshot(1000);
        assert_eq!(snap.total_requests, 10005);
        assert_eq!(snap.total_errors, 5);
        assert_eq!(budget.history().len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(1000);
        budget.record_errors(10);
        budget.reset();
        assert_eq!(budget.total_requests(), 0);
        assert_eq!(budget.total_errors(), 0);
    }

    #[test]
    fn test_registry() {
        let mut reg = ErrorBudgetRegistry::new();
        reg.register(ErrorBudget::new(SloTarget::new("avail", 0.999, 3600)));
        reg.register(ErrorBudget::new(SloTarget::new("latency", 0.99, 3600)));
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.names(), vec!["avail", "latency"]);
        assert!(reg.get("avail").is_some());
    }

    #[test]
    fn test_registry_alerts() {
        let mut reg = ErrorBudgetRegistry::new();
        let mut budget = ErrorBudget::new(SloTarget::new("test_slo", 0.999, 3600));
        budget.record_successes(9989);
        budget.record_errors(11);
        reg.register(budget);
        let alerts = reg.check_all_alerts(1000);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].slo_name, "test_slo");
    }

    #[test]
    fn test_budget_status_as_str() {
        assert_eq!(BudgetStatus::Healthy.as_str(), "healthy");
        assert_eq!(BudgetStatus::Warning.as_str(), "warning");
        assert_eq!(BudgetStatus::Critical.as_str(), "critical");
        assert_eq!(BudgetStatus::Exhausted.as_str(), "exhausted");
    }

    #[test]
    fn test_batch_operations() {
        let mut budget = ErrorBudget::new(make_slo());
        budget.record_successes(500);
        budget.record_errors(3);
        assert_eq!(budget.total_requests(), 503);
        assert_eq!(budget.total_errors(), 3);
    }
}
