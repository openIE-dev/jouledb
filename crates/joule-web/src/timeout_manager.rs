//! Timeout management — per-operation timeouts, cascading timeout budgets,
//! timeout inheritance, deadline propagation, timeout statistics, and adaptive
//! timeouts based on observed p99 latency.
//!
//! Pure Rust timeout infrastructure for coordinating time budgets across
//! nested operations without an async runtime.

use std::collections::HashMap;

// ── Timeout Error ───────────────────────────────────────────────

/// Errors from the timeout system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeoutError {
    Expired { operation: String, budget_ms: u64, elapsed_ms: u64 },
    NoBudgetRemaining { operation: String },
    OperationNotFound(String),
    DeadlineExceeded { deadline_ms: u64, now_ms: u64 },
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expired { operation, budget_ms, elapsed_ms } => {
                write!(f, "timeout expired for {operation}: {elapsed_ms}ms > {budget_ms}ms")
            }
            Self::NoBudgetRemaining { operation } => {
                write!(f, "no budget remaining for {operation}")
            }
            Self::OperationNotFound(op) => write!(f, "operation not found: {op}"),
            Self::DeadlineExceeded { deadline_ms, now_ms } => {
                write!(f, "deadline exceeded: now={now_ms}ms > deadline={deadline_ms}ms")
            }
        }
    }
}

impl std::error::Error for TimeoutError {}

// ── Timeout Config ──────────────────────────────────────────────

/// Configuration for a single operation's timeout.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    pub operation: String,
    pub timeout_ms: u64,
    /// Whether this timeout inherits from a parent budget.
    pub inherits_parent: bool,
}

impl TimeoutConfig {
    pub fn new(operation: impl Into<String>, timeout_ms: u64) -> Self {
        Self {
            operation: operation.into(),
            timeout_ms,
            inherits_parent: false,
        }
    }

    pub fn with_inheritance(mut self) -> Self {
        self.inherits_parent = true;
        self
    }
}

// ── Timeout Entry ───────────────────────────────────────────────

/// Runtime state for a single timeout.
#[derive(Debug, Clone)]
struct TimeoutEntry {
    config: TimeoutConfig,
    started_at_ms: u64,
    elapsed_ms: u64,
    completed: bool,
    timed_out: bool,
}

impl TimeoutEntry {
    fn remaining_ms(&self) -> u64 {
        self.config.timeout_ms.saturating_sub(self.elapsed_ms)
    }

    fn is_expired(&self) -> bool {
        self.elapsed_ms >= self.config.timeout_ms
    }
}

// ── Deadline ────────────────────────────────────────────────────

/// An absolute deadline that propagates across operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline {
    /// Absolute deadline in ms (relative to an epoch / monotonic clock).
    pub deadline_ms: u64,
}

impl Deadline {
    pub fn new(deadline_ms: u64) -> Self {
        Self { deadline_ms }
    }

    /// Create from a timeout starting at `now_ms`.
    pub fn from_timeout(now_ms: u64, timeout_ms: u64) -> Self {
        Self {
            deadline_ms: now_ms.saturating_add(timeout_ms),
        }
    }

    /// How much time is left given the current time.
    pub fn remaining_ms(&self, now_ms: u64) -> u64 {
        self.deadline_ms.saturating_sub(now_ms)
    }

    /// Whether the deadline has passed.
    pub fn is_exceeded(&self, now_ms: u64) -> bool {
        now_ms >= self.deadline_ms
    }

    /// Take the earlier of two deadlines.
    pub fn min(self, other: Deadline) -> Deadline {
        if self.deadline_ms <= other.deadline_ms {
            self
        } else {
            other
        }
    }
}

// ── Timeout Statistics ──────────────────────────────────────────

/// Aggregated timeout statistics per operation.
#[derive(Debug, Clone, Default)]
pub struct TimeoutStats {
    pub total_started: u64,
    pub total_completed: u64,
    pub total_timed_out: u64,
    /// Observed latencies for adaptive timeout.
    latencies_ms: Vec<u64>,
    max_latency_ms: u64,
    sum_latency_ms: u64,
}

impl TimeoutStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_completion(&mut self, latency_ms: u64) {
        self.total_started += 1;
        self.total_completed += 1;
        self.latencies_ms.push(latency_ms);
        self.sum_latency_ms += latency_ms;
        if latency_ms > self.max_latency_ms {
            self.max_latency_ms = latency_ms;
        }
    }

    pub fn record_timeout(&mut self, latency_ms: u64) {
        self.total_started += 1;
        self.total_timed_out += 1;
        self.latencies_ms.push(latency_ms);
        self.sum_latency_ms += latency_ms;
        if latency_ms > self.max_latency_ms {
            self.max_latency_ms = latency_ms;
        }
    }

    pub fn timeout_rate(&self) -> f64 {
        if self.total_started == 0 {
            return 0.0;
        }
        self.total_timed_out as f64 / self.total_started as f64
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        self.sum_latency_ms as f64 / self.latencies_ms.len() as f64
    }

    pub fn max_latency_ms(&self) -> u64 {
        self.max_latency_ms
    }

    /// Calculate the p99 latency.
    pub fn p99_latency_ms(&self) -> Option<u64> {
        if self.latencies_ms.is_empty() {
            return None;
        }
        let mut sorted = self.latencies_ms.clone();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len().saturating_sub(1));
        Some(sorted[idx])
    }

    /// Calculate the p95 latency.
    pub fn p95_latency_ms(&self) -> Option<u64> {
        if self.latencies_ms.is_empty() {
            return None;
        }
        let mut sorted = self.latencies_ms.clone();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len().saturating_sub(1));
        Some(sorted[idx])
    }

    pub fn count(&self) -> usize {
        self.latencies_ms.len()
    }
}

// ── Cascading Timeout Budget ────────────────────────────────────

/// A cascading timeout budget. Parent allocates a total budget; children
/// consume from it. When the parent budget is exhausted, all children fail.
#[derive(Debug)]
pub struct CascadingBudget {
    name: String,
    total_budget_ms: u64,
    consumed_ms: u64,
    children: Vec<(String, u64)>, // (name, consumed_ms)
}

impl CascadingBudget {
    pub fn new(name: impl Into<String>, total_budget_ms: u64) -> Self {
        Self {
            name: name.into(),
            total_budget_ms,
            consumed_ms: 0,
            children: Vec::new(),
        }
    }

    /// Remaining budget.
    pub fn remaining_ms(&self) -> u64 {
        self.total_budget_ms.saturating_sub(self.consumed_ms)
    }

    /// Whether the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.consumed_ms >= self.total_budget_ms
    }

    /// Allocate budget for a child operation. Returns the allowed timeout.
    pub fn allocate(&self, child_name: &str, requested_ms: u64) -> u64 {
        let remaining = self.remaining_ms();
        remaining.min(requested_ms)
    }

    /// Record that a child consumed some time.
    pub fn consume(&mut self, child_name: impl Into<String>, elapsed_ms: u64) {
        let name = child_name.into();
        self.consumed_ms += elapsed_ms;
        if let Some(entry) = self.children.iter_mut().find(|(n, _)| *n == name) {
            entry.1 += elapsed_ms;
        } else {
            self.children.push((name, elapsed_ms));
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn total_budget_ms(&self) -> u64 {
        self.total_budget_ms
    }

    pub fn consumed_ms(&self) -> u64 {
        self.consumed_ms
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Get consumption for a specific child.
    pub fn child_consumed(&self, name: &str) -> u64 {
        self.children
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }
}

// ── Timeout Manager ─────────────────────────────────────────────

/// Central timeout manager: tracks per-operation timeouts, cascading budgets,
/// deadlines, and collects statistics.
#[derive(Debug)]
pub struct TimeoutManager {
    /// Per-operation timeout configurations.
    configs: HashMap<String, TimeoutConfig>,
    /// Active timeouts.
    active: HashMap<String, TimeoutEntry>,
    /// Per-operation statistics.
    stats: HashMap<String, TimeoutStats>,
    /// Global deadline (if set).
    global_deadline: Option<Deadline>,
    /// Cascading budgets.
    budgets: HashMap<String, CascadingBudget>,
    /// Adaptive timeout multiplier (e.g. 1.5x p99).
    adaptive_multiplier: f64,
}

impl TimeoutManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            active: HashMap::new(),
            stats: HashMap::new(),
            global_deadline: None,
            budgets: HashMap::new(),
            adaptive_multiplier: 1.5,
        }
    }

    /// Set the adaptive multiplier (applied to p99 for adaptive timeouts).
    pub fn with_adaptive_multiplier(mut self, multiplier: f64) -> Self {
        self.adaptive_multiplier = multiplier.max(1.0);
        self
    }

    /// Register a timeout configuration for an operation.
    pub fn register(&mut self, config: TimeoutConfig) {
        self.configs.insert(config.operation.clone(), config);
    }

    /// Set a global deadline.
    pub fn set_deadline(&mut self, deadline: Deadline) {
        self.global_deadline = Some(deadline);
    }

    /// Register a cascading budget.
    pub fn register_budget(&mut self, budget: CascadingBudget) {
        self.budgets.insert(budget.name().to_string(), budget);
    }

    /// Start an operation. Returns the effective timeout in ms.
    pub fn start(
        &mut self,
        operation: &str,
        now_ms: u64,
    ) -> Result<u64, TimeoutError> {
        let config = self
            .configs
            .get(operation)
            .cloned()
            .ok_or_else(|| TimeoutError::OperationNotFound(operation.to_string()))?;

        // Check global deadline.
        if let Some(deadline) = &self.global_deadline {
            if deadline.is_exceeded(now_ms) {
                return Err(TimeoutError::DeadlineExceeded {
                    deadline_ms: deadline.deadline_ms,
                    now_ms,
                });
            }
        }

        let effective_timeout = self.effective_timeout(operation, now_ms);

        let entry = TimeoutEntry {
            config,
            started_at_ms: now_ms,
            elapsed_ms: 0,
            completed: false,
            timed_out: false,
        };
        self.active.insert(operation.to_string(), entry);
        Ok(effective_timeout)
    }

    /// Check if an active operation has timed out given the current time.
    pub fn check(&mut self, operation: &str, now_ms: u64) -> Result<u64, TimeoutError> {
        let entry = self
            .active
            .get_mut(operation)
            .ok_or_else(|| TimeoutError::OperationNotFound(operation.to_string()))?;

        entry.elapsed_ms = now_ms.saturating_sub(entry.started_at_ms);

        // Check global deadline.
        if let Some(deadline) = &self.global_deadline {
            if deadline.is_exceeded(now_ms) {
                entry.timed_out = true;
                return Err(TimeoutError::DeadlineExceeded {
                    deadline_ms: deadline.deadline_ms,
                    now_ms,
                });
            }
        }

        if entry.is_expired() {
            entry.timed_out = true;
            Err(TimeoutError::Expired {
                operation: operation.to_string(),
                budget_ms: entry.config.timeout_ms,
                elapsed_ms: entry.elapsed_ms,
            })
        } else {
            Ok(entry.remaining_ms())
        }
    }

    /// Complete an operation successfully.
    pub fn complete(&mut self, operation: &str, now_ms: u64) {
        if let Some(entry) = self.active.remove(operation) {
            let latency = now_ms.saturating_sub(entry.started_at_ms);
            let stats = self.stats.entry(operation.to_string()).or_default();
            if entry.timed_out {
                stats.record_timeout(latency);
            } else {
                stats.record_completion(latency);
            }
        }
    }

    /// Record a timeout for an operation.
    pub fn record_timeout(&mut self, operation: &str, now_ms: u64) {
        if let Some(mut entry) = self.active.remove(operation) {
            entry.timed_out = true;
            let latency = now_ms.saturating_sub(entry.started_at_ms);
            let stats = self.stats.entry(operation.to_string()).or_default();
            stats.record_timeout(latency);
        }
    }

    /// Get the effective timeout for an operation, considering global deadline
    /// and adaptive adjustment.
    pub fn effective_timeout(&self, operation: &str, now_ms: u64) -> u64 {
        let base = self
            .configs
            .get(operation)
            .map(|c| c.timeout_ms)
            .unwrap_or(u64::MAX);

        // Adaptive: use p99 * multiplier if available.
        let adaptive = self.adaptive_timeout(operation).unwrap_or(base);
        let effective = base.min(adaptive);

        // Constrain by global deadline.
        if let Some(deadline) = &self.global_deadline {
            let remaining = deadline.remaining_ms(now_ms);
            effective.min(remaining)
        } else {
            effective
        }
    }

    /// Calculate adaptive timeout based on observed p99.
    pub fn adaptive_timeout(&self, operation: &str) -> Option<u64> {
        self.stats
            .get(operation)
            .and_then(|s| s.p99_latency_ms())
            .map(|p99| (p99 as f64 * self.adaptive_multiplier) as u64)
    }

    /// Get statistics for an operation.
    pub fn stats(&self, operation: &str) -> Option<&TimeoutStats> {
        self.stats.get(operation)
    }

    /// Get a cascading budget.
    pub fn budget(&self, name: &str) -> Option<&CascadingBudget> {
        self.budgets.get(name)
    }

    /// Get a mutable cascading budget.
    pub fn budget_mut(&mut self, name: &str) -> Option<&mut CascadingBudget> {
        self.budgets.get_mut(name)
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn registered_count(&self) -> usize {
        self.configs.len()
    }
}

impl Default for TimeoutManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_config() {
        let cfg = TimeoutConfig::new("fetch", 5000);
        assert_eq!(cfg.operation, "fetch");
        assert_eq!(cfg.timeout_ms, 5000);
        assert!(!cfg.inherits_parent);

        let cfg2 = cfg.with_inheritance();
        assert!(cfg2.inherits_parent);
    }

    #[test]
    fn test_deadline_basic() {
        let dl = Deadline::new(1000);
        assert_eq!(dl.remaining_ms(500), 500);
        assert!(!dl.is_exceeded(500));
        assert!(dl.is_exceeded(1000));
        assert!(dl.is_exceeded(1500));
    }

    #[test]
    fn test_deadline_from_timeout() {
        let dl = Deadline::from_timeout(100, 500);
        assert_eq!(dl.deadline_ms, 600);
        assert_eq!(dl.remaining_ms(300), 300);
    }

    #[test]
    fn test_deadline_min() {
        let a = Deadline::new(1000);
        let b = Deadline::new(500);
        assert_eq!(a.min(b).deadline_ms, 500);
        assert_eq!(b.min(a).deadline_ms, 500);
    }

    #[test]
    fn test_manager_start_and_complete() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("op", 1000));
        let timeout = mgr.start("op", 100).unwrap();
        assert_eq!(timeout, 1000);
        assert_eq!(mgr.active_count(), 1);
        mgr.complete("op", 300);
        assert_eq!(mgr.active_count(), 0);
        let stats = mgr.stats("op").unwrap();
        assert_eq!(stats.total_completed, 1);
    }

    #[test]
    fn test_manager_check_ok() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("op", 1000));
        mgr.start("op", 0).unwrap();
        let remaining = mgr.check("op", 500).unwrap();
        assert_eq!(remaining, 500);
    }

    #[test]
    fn test_manager_check_expired() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("op", 100));
        mgr.start("op", 0).unwrap();
        let result = mgr.check("op", 200);
        assert!(result.is_err());
        match result.unwrap_err() {
            TimeoutError::Expired { operation, budget_ms, elapsed_ms } => {
                assert_eq!(operation, "op");
                assert_eq!(budget_ms, 100);
                assert_eq!(elapsed_ms, 200);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_manager_unknown_operation() {
        let mut mgr = TimeoutManager::new();
        let result = mgr.start("unknown", 0);
        assert!(matches!(result, Err(TimeoutError::OperationNotFound(_))));
    }

    #[test]
    fn test_global_deadline() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("op", 5000));
        mgr.set_deadline(Deadline::new(1000));
        let timeout = mgr.start("op", 0).unwrap();
        assert_eq!(timeout, 1000); // Constrained by deadline.
    }

    #[test]
    fn test_global_deadline_exceeded() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("op", 5000));
        mgr.set_deadline(Deadline::new(100));
        let result = mgr.start("op", 200);
        assert!(matches!(result, Err(TimeoutError::DeadlineExceeded { .. })));
    }

    #[test]
    fn test_cascading_budget() {
        let mut budget = CascadingBudget::new("request", 1000);
        assert_eq!(budget.remaining_ms(), 1000);
        assert!(!budget.is_exhausted());
        let alloc = budget.allocate("db_query", 500);
        assert_eq!(alloc, 500);
        budget.consume("db_query", 300);
        assert_eq!(budget.remaining_ms(), 700);
        assert_eq!(budget.child_consumed("db_query"), 300);
    }

    #[test]
    fn test_cascading_budget_exhaustion() {
        let mut budget = CascadingBudget::new("request", 100);
        budget.consume("step1", 80);
        let alloc = budget.allocate("step2", 50);
        assert_eq!(alloc, 20); // Only 20 remaining.
        budget.consume("step2", 20);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_timeout_stats() {
        let mut stats = TimeoutStats::new();
        stats.record_completion(100);
        stats.record_completion(200);
        stats.record_timeout(300);
        assert_eq!(stats.total_started, 3);
        assert_eq!(stats.total_completed, 2);
        assert_eq!(stats.total_timed_out, 1);
        assert!((stats.timeout_rate() - 1.0 / 3.0).abs() < 0.01);
        assert!((stats.avg_latency_ms() - 200.0).abs() < 0.01);
        assert_eq!(stats.max_latency_ms(), 300);
    }

    #[test]
    fn test_timeout_stats_percentiles() {
        let mut stats = TimeoutStats::new();
        for i in 1..=100 {
            stats.record_completion(i);
        }
        let p99 = stats.p99_latency_ms().unwrap();
        assert!(p99 >= 98);
        let p95 = stats.p95_latency_ms().unwrap();
        assert!(p95 >= 93);
    }

    #[test]
    fn test_adaptive_timeout() {
        let mut mgr = TimeoutManager::new().with_adaptive_multiplier(2.0);
        mgr.register(TimeoutConfig::new("op", 10000));
        // Manually add stats.
        let stats = mgr.stats.entry("op".to_string()).or_default();
        for i in 1..=100 {
            stats.record_completion(i);
        }
        let adaptive = mgr.adaptive_timeout("op").unwrap();
        // p99 ~ 99, * 2.0 = ~198
        assert!(adaptive >= 190 && adaptive <= 210);
    }

    #[test]
    fn test_record_timeout() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("op", 100));
        mgr.start("op", 0).unwrap();
        mgr.record_timeout("op", 150);
        let stats = mgr.stats("op").unwrap();
        assert_eq!(stats.total_timed_out, 1);
    }

    #[test]
    fn test_registered_count() {
        let mut mgr = TimeoutManager::new();
        mgr.register(TimeoutConfig::new("a", 100));
        mgr.register(TimeoutConfig::new("b", 200));
        assert_eq!(mgr.registered_count(), 2);
    }

    #[test]
    fn test_cascading_budget_child_count() {
        let mut budget = CascadingBudget::new("req", 1000);
        budget.consume("a", 100);
        budget.consume("b", 200);
        assert_eq!(budget.child_count(), 2);
    }

    #[test]
    fn test_empty_stats() {
        let stats = TimeoutStats::new();
        assert_eq!(stats.timeout_rate(), 0.0);
        assert_eq!(stats.avg_latency_ms(), 0.0);
        assert!(stats.p99_latency_ms().is_none());
    }
}
