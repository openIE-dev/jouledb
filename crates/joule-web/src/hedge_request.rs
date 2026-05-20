//! Hedged requests — send duplicate requests after delay, take first response,
//! cancel slower request, percentile-based hedge delay, hedge budget, and statistics.
//!
//! Pure Rust hedged request infrastructure for latency reduction.
//! Models the hedge request pattern without async runtime dependencies.

use std::collections::VecDeque;

// ── Hedge Decision ──────────────────────────────────────────────

/// Whether to send a hedge request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgeDecision {
    /// Send the hedge.
    Hedge,
    /// Do not hedge (budget exhausted or delay not reached).
    Skip,
    /// Budget exhausted.
    BudgetExhausted,
}

// ── Hedge Outcome ───────────────────────────────────────────────

/// Outcome of a hedged request pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgeOutcome {
    /// Primary responded first.
    PrimaryWon,
    /// Hedge responded first.
    HedgeWon,
    /// Only primary was sent (no hedge).
    NoHedge,
}

// ── Percentile Calculator ───────────────────────────────────────

/// Rolling percentile calculator for latency-based hedge delay.
#[derive(Debug)]
pub struct PercentileTracker {
    window: VecDeque<u64>,
    max_window: usize,
}

impl PercentileTracker {
    pub fn new(max_window: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_window),
            max_window: max_window.max(1),
        }
    }

    /// Record a latency observation.
    pub fn record(&mut self, latency_ms: u64) {
        if self.window.len() >= self.max_window {
            self.window.pop_front();
        }
        self.window.push_back(latency_ms);
    }

    /// Calculate the given percentile (0.0 - 1.0).
    pub fn percentile(&self, p: f64) -> Option<u64> {
        if self.window.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.window.iter().copied().collect();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64 * p.clamp(0.0, 1.0)) as usize)
            .min(sorted.len().saturating_sub(1));
        Some(sorted[idx])
    }

    /// P50 (median).
    pub fn p50(&self) -> Option<u64> {
        self.percentile(0.5)
    }

    /// P95.
    pub fn p95(&self) -> Option<u64> {
        self.percentile(0.95)
    }

    /// P99.
    pub fn p99(&self) -> Option<u64> {
        self.percentile(0.99)
    }

    pub fn count(&self) -> usize {
        self.window.len()
    }

    pub fn clear(&mut self) {
        self.window.clear();
    }
}

// ── Hedge Budget ────────────────────────────────────────────────

/// Controls how many outstanding hedges are allowed.
#[derive(Debug)]
pub struct HedgeBudget {
    /// Maximum outstanding hedges.
    max_outstanding: usize,
    /// Current outstanding hedges.
    outstanding: usize,
    /// Maximum hedge rate (hedges per window).
    max_hedges_per_window: u64,
    /// Hedges issued in the current window.
    window_hedges: u64,
    /// Total hedges ever issued.
    total_hedges: u64,
    /// Total hedges skipped due to budget.
    total_skipped: u64,
}

impl HedgeBudget {
    pub fn new(max_outstanding: usize, max_hedges_per_window: u64) -> Self {
        Self {
            max_outstanding: max_outstanding.max(1),
            outstanding: 0,
            max_hedges_per_window,
            window_hedges: 0,
            total_hedges: 0,
            total_skipped: 0,
        }
    }

    /// Try to acquire a hedge permit.
    pub fn try_acquire(&mut self) -> bool {
        if self.outstanding >= self.max_outstanding {
            self.total_skipped += 1;
            return false;
        }
        if self.window_hedges >= self.max_hedges_per_window {
            self.total_skipped += 1;
            return false;
        }
        self.outstanding += 1;
        self.window_hedges += 1;
        self.total_hedges += 1;
        true
    }

    /// Release a hedge permit (hedge completed or cancelled).
    pub fn release(&mut self) {
        self.outstanding = self.outstanding.saturating_sub(1);
    }

    /// Reset the window counter.
    pub fn reset_window(&mut self) {
        self.window_hedges = 0;
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    pub fn total_hedges(&self) -> u64 {
        self.total_hedges
    }

    pub fn total_skipped(&self) -> u64 {
        self.total_skipped
    }

    pub fn window_hedges(&self) -> u64 {
        self.window_hedges
    }
}

// ── Hedge Statistics ────────────────────────────────────────────

/// Aggregated statistics for hedged requests.
#[derive(Debug, Clone, Default)]
pub struct HedgeStatistics {
    pub total_requests: u64,
    pub total_hedges_sent: u64,
    pub primary_wins: u64,
    pub hedge_wins: u64,
    pub no_hedge: u64,
    pub hedge_savings_ms: u64,
}

impl HedgeStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, outcome: HedgeOutcome, savings_ms: u64) {
        self.total_requests += 1;
        match outcome {
            HedgeOutcome::PrimaryWon => {
                self.primary_wins += 1;
                self.total_hedges_sent += 1;
            }
            HedgeOutcome::HedgeWon => {
                self.hedge_wins += 1;
                self.total_hedges_sent += 1;
                self.hedge_savings_ms += savings_ms;
            }
            HedgeOutcome::NoHedge => {
                self.no_hedge += 1;
            }
        }
    }

    pub fn hedge_win_rate(&self) -> f64 {
        if self.total_hedges_sent == 0 {
            return 0.0;
        }
        self.hedge_wins as f64 / self.total_hedges_sent as f64
    }

    pub fn hedge_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_hedges_sent as f64 / self.total_requests as f64
    }

    pub fn avg_savings_ms(&self) -> f64 {
        if self.hedge_wins == 0 {
            return 0.0;
        }
        self.hedge_savings_ms as f64 / self.hedge_wins as f64
    }
}

// ── Hedge Controller ────────────────────────────────────────────

/// Controller for hedged request decisions.
#[derive(Debug)]
pub struct HedgeController {
    /// Fixed delay in ms before sending hedge (used when no percentile data).
    fixed_delay_ms: u64,
    /// Percentile to use for adaptive delay (e.g. 0.95 = p95).
    delay_percentile: f64,
    /// Latency tracker for adaptive delay.
    latency_tracker: PercentileTracker,
    /// Budget controller.
    budget: HedgeBudget,
    /// Aggregated statistics.
    stats: HedgeStatistics,
}

impl HedgeController {
    pub fn new(fixed_delay_ms: u64, max_outstanding: usize, max_per_window: u64) -> Self {
        Self {
            fixed_delay_ms,
            delay_percentile: 0.95,
            latency_tracker: PercentileTracker::new(1000),
            budget: HedgeBudget::new(max_outstanding, max_per_window),
            stats: HedgeStatistics::new(),
        }
    }

    /// Set the percentile used for adaptive hedge delay.
    pub fn with_delay_percentile(mut self, percentile: f64) -> Self {
        self.delay_percentile = percentile.clamp(0.0, 1.0);
        self
    }

    /// Get the hedge delay in ms (adaptive if enough data, else fixed).
    pub fn hedge_delay_ms(&self) -> u64 {
        self.latency_tracker
            .percentile(self.delay_percentile)
            .unwrap_or(self.fixed_delay_ms)
    }

    /// Should we hedge? Call this after the primary has been running for `elapsed_ms`.
    pub fn should_hedge(&mut self, elapsed_ms: u64) -> HedgeDecision {
        let delay = self.hedge_delay_ms();
        if elapsed_ms < delay {
            return HedgeDecision::Skip;
        }
        if self.budget.try_acquire() {
            HedgeDecision::Hedge
        } else {
            HedgeDecision::BudgetExhausted
        }
    }

    /// Record a completed request (primary or hedge) latency.
    pub fn record_latency(&mut self, latency_ms: u64) {
        self.latency_tracker.record(latency_ms);
    }

    /// Record the outcome of a hedged pair.
    pub fn record_outcome(&mut self, outcome: HedgeOutcome, savings_ms: u64) {
        self.stats.record(outcome, savings_ms);
        if outcome != HedgeOutcome::NoHedge {
            self.budget.release();
        }
    }

    /// Resolve a hedged pair: given primary and hedge latencies, determine winner.
    pub fn resolve(
        &mut self,
        primary_latency_ms: u64,
        hedge_latency_ms: Option<u64>,
    ) -> HedgeOutcome {
        self.record_latency(primary_latency_ms);

        match hedge_latency_ms {
            None => {
                let outcome = HedgeOutcome::NoHedge;
                self.stats.record(outcome, 0);
                outcome
            }
            Some(hedge_lat) => {
                self.record_latency(hedge_lat);
                if primary_latency_ms <= hedge_lat {
                    let outcome = HedgeOutcome::PrimaryWon;
                    self.stats.record(outcome, 0);
                    self.budget.release();
                    outcome
                } else {
                    let savings = primary_latency_ms - hedge_lat;
                    let outcome = HedgeOutcome::HedgeWon;
                    self.stats.record(outcome, savings);
                    self.budget.release();
                    outcome
                }
            }
        }
    }

    pub fn stats(&self) -> &HedgeStatistics {
        &self.stats
    }

    pub fn budget(&self) -> &HedgeBudget {
        &self.budget
    }

    pub fn latency_p50(&self) -> Option<u64> {
        self.latency_tracker.p50()
    }

    pub fn latency_p95(&self) -> Option<u64> {
        self.latency_tracker.p95()
    }

    pub fn latency_p99(&self) -> Option<u64> {
        self.latency_tracker.p99()
    }

    /// Reset the per-window budget.
    pub fn reset_window(&mut self) {
        self.budget.reset_window();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_basic() {
        let mut pt = PercentileTracker::new(10);
        for i in 1..=10 {
            pt.record(i * 10);
        }
        // Sorted: 10,20,30,40,50,60,70,80,90,100
        let p50 = pt.p50().unwrap();
        assert!(p50 >= 40 && p50 <= 60);
    }

    #[test]
    fn test_percentile_p99() {
        let mut pt = PercentileTracker::new(100);
        for i in 1..=100 {
            pt.record(i);
        }
        let p99 = pt.p99().unwrap();
        assert!(p99 >= 98);
    }

    #[test]
    fn test_percentile_empty() {
        let pt = PercentileTracker::new(10);
        assert!(pt.p50().is_none());
        assert!(pt.p99().is_none());
    }

    #[test]
    fn test_percentile_single() {
        let mut pt = PercentileTracker::new(10);
        pt.record(42);
        assert_eq!(pt.p50(), Some(42));
        assert_eq!(pt.p99(), Some(42));
    }

    #[test]
    fn test_percentile_window_eviction() {
        let mut pt = PercentileTracker::new(3);
        pt.record(100);
        pt.record(200);
        pt.record(300);
        pt.record(400); // Evicts 100.
        assert_eq!(pt.count(), 3);
        // Should have 200, 300, 400.
        let p50 = pt.p50().unwrap();
        assert_eq!(p50, 300);
    }

    #[test]
    fn test_hedge_budget_acquire() {
        let mut budget = HedgeBudget::new(2, 10);
        assert!(budget.try_acquire());
        assert!(budget.try_acquire());
        assert!(!budget.try_acquire()); // Max outstanding.
        assert_eq!(budget.outstanding(), 2);
        assert_eq!(budget.total_skipped(), 1);
    }

    #[test]
    fn test_hedge_budget_release() {
        let mut budget = HedgeBudget::new(1, 10);
        assert!(budget.try_acquire());
        budget.release();
        assert!(budget.try_acquire()); // Can acquire again.
    }

    #[test]
    fn test_hedge_budget_window_limit() {
        let mut budget = HedgeBudget::new(10, 2);
        assert!(budget.try_acquire());
        budget.release();
        assert!(budget.try_acquire());
        budget.release();
        assert!(!budget.try_acquire()); // Window limit.
        budget.reset_window();
        assert!(budget.try_acquire()); // New window.
    }

    #[test]
    fn test_hedge_controller_skip_early() {
        let mut hc = HedgeController::new(100, 5, 100);
        assert_eq!(hc.should_hedge(50), HedgeDecision::Skip);
    }

    #[test]
    fn test_hedge_controller_hedge() {
        let mut hc = HedgeController::new(100, 5, 100);
        assert_eq!(hc.should_hedge(150), HedgeDecision::Hedge);
    }

    #[test]
    fn test_hedge_controller_budget_exhausted() {
        let mut hc = HedgeController::new(10, 1, 100);
        hc.should_hedge(20); // Takes the one permit.
        assert_eq!(hc.should_hedge(20), HedgeDecision::BudgetExhausted);
    }

    #[test]
    fn test_hedge_resolve_primary_wins() {
        let mut hc = HedgeController::new(100, 5, 100);
        let outcome = hc.resolve(50, Some(80));
        assert_eq!(outcome, HedgeOutcome::PrimaryWon);
        assert_eq!(hc.stats().primary_wins, 1);
    }

    #[test]
    fn test_hedge_resolve_hedge_wins() {
        let mut hc = HedgeController::new(100, 5, 100);
        let outcome = hc.resolve(200, Some(80));
        assert_eq!(outcome, HedgeOutcome::HedgeWon);
        assert_eq!(hc.stats().hedge_wins, 1);
        assert_eq!(hc.stats().hedge_savings_ms, 120);
    }

    #[test]
    fn test_hedge_resolve_no_hedge() {
        let mut hc = HedgeController::new(100, 5, 100);
        let outcome = hc.resolve(50, None);
        assert_eq!(outcome, HedgeOutcome::NoHedge);
        assert_eq!(hc.stats().no_hedge, 1);
    }

    #[test]
    fn test_hedge_statistics() {
        let mut stats = HedgeStatistics::new();
        stats.record(HedgeOutcome::HedgeWon, 50);
        stats.record(HedgeOutcome::PrimaryWon, 0);
        stats.record(HedgeOutcome::NoHedge, 0);
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.total_hedges_sent, 2);
        assert!((stats.hedge_win_rate() - 0.5).abs() < 0.01);
        assert!((stats.hedge_rate() - 2.0 / 3.0).abs() < 0.01);
        assert!((stats.avg_savings_ms() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_hedge_statistics_empty() {
        let stats = HedgeStatistics::new();
        assert_eq!(stats.hedge_win_rate(), 0.0);
        assert_eq!(stats.hedge_rate(), 0.0);
        assert_eq!(stats.avg_savings_ms(), 0.0);
    }

    #[test]
    fn test_adaptive_delay() {
        let mut hc = HedgeController::new(100, 5, 100);
        // Record some latencies.
        for i in 1..=100 {
            hc.record_latency(i);
        }
        // Adaptive delay should use p95.
        let delay = hc.hedge_delay_ms();
        assert!(delay >= 90); // p95 of 1..100
    }

    #[test]
    fn test_fixed_delay_when_no_data() {
        let hc = HedgeController::new(200, 5, 100);
        assert_eq!(hc.hedge_delay_ms(), 200);
    }

    #[test]
    fn test_latency_percentiles() {
        let mut hc = HedgeController::new(100, 5, 100);
        for i in 1..=100 {
            hc.record_latency(i);
        }
        assert!(hc.latency_p50().is_some());
        assert!(hc.latency_p95().is_some());
        assert!(hc.latency_p99().is_some());
    }

    #[test]
    fn test_percentile_clear() {
        let mut pt = PercentileTracker::new(10);
        pt.record(100);
        pt.clear();
        assert_eq!(pt.count(), 0);
        assert!(pt.p50().is_none());
    }

    #[test]
    fn test_hedge_budget_totals() {
        let mut budget = HedgeBudget::new(1, 10);
        budget.try_acquire();
        budget.try_acquire(); // Skipped.
        assert_eq!(budget.total_hedges(), 1);
        assert_eq!(budget.total_skipped(), 1);
    }
}
