//! Rate controller / governor — adaptive rate control with AIMD and congestion window.
//!
//! Replaces ad-hoc rate limiters with a principled controller inspired by TCP
//! congestion control. Supports additive increase / multiplicative decrease (AIMD),
//! slow start, congestion window management, rate history, feedback integration,
//! and fair rate limiting across multiple flows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Rate Phase ────────────────────────────────────────────────

/// Current phase of the rate controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RatePhase {
    /// Exponential growth until threshold.
    SlowStart,
    /// Linear growth after slow-start threshold.
    CongestionAvoidance,
    /// Rate was cut due to congestion signal.
    Recovery,
}

// ── Congestion Event ──────────────────────────────────────────

/// A signal that congestion was detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionEvent {
    /// Timestamp (ms) when congestion was detected.
    pub timestamp_ms: u64,
    /// The window size at the time of congestion.
    pub window_at_event: f64,
    /// Optional reason label.
    pub reason: String,
}

// ── Rate Snapshot ─────────────────────────────────────────────

/// A snapshot of the controller's rate at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateSnapshot {
    pub timestamp_ms: u64,
    pub window: f64,
    pub phase: RatePhase,
    pub throughput: f64,
}

// ── Rate Controller Config ────────────────────────────────────

/// Configuration for the rate controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateControllerConfig {
    /// Initial congestion window.
    pub initial_window: f64,
    /// Minimum window (floor).
    pub min_window: f64,
    /// Maximum window (ceiling).
    pub max_window: f64,
    /// Slow-start threshold — switch from slow start to congestion avoidance.
    pub ss_threshold: f64,
    /// Additive increase per successful round.
    pub additive_increase: f64,
    /// Multiplicative decrease factor on congestion (0.0 - 1.0).
    pub multiplicative_decrease: f64,
    /// Number of rate snapshots to keep in history.
    pub history_capacity: usize,
}

impl Default for RateControllerConfig {
    fn default() -> Self {
        Self {
            initial_window: 1.0,
            min_window: 1.0,
            max_window: 1024.0,
            ss_threshold: 64.0,
            additive_increase: 1.0,
            multiplicative_decrease: 0.5,
            history_capacity: 100,
        }
    }
}

// ── Rate Controller ───────────────────────────────────────────

/// Adaptive rate controller using AIMD (Additive Increase / Multiplicative Decrease).
#[derive(Debug)]
pub struct RateController {
    config: RateControllerConfig,
    /// Current congestion window.
    window: f64,
    /// Current phase.
    phase: RatePhase,
    /// Slow-start threshold (adapts on congestion).
    ss_threshold: f64,
    /// Total successful operations.
    total_success: u64,
    /// Total congestion events.
    total_congestion: u64,
    /// Rate history.
    history: Vec<RateSnapshot>,
    /// Congestion events log.
    congestion_log: Vec<CongestionEvent>,
    /// Current throughput measurement (ops/sec or ops/round).
    throughput: f64,
}

impl RateController {
    /// Create a new rate controller with the given configuration.
    pub fn new(config: RateControllerConfig) -> Self {
        let window = config.initial_window;
        let ss_threshold = config.ss_threshold;
        Self {
            config,
            window,
            phase: RatePhase::SlowStart,
            ss_threshold,
            total_success: 0,
            total_congestion: 0,
            history: Vec::new(),
            congestion_log: Vec::new(),
            throughput: 0.0,
        }
    }

    /// Create a rate controller with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RateControllerConfig::default())
    }

    /// Get the current window size (how many operations are allowed concurrently).
    pub fn window(&self) -> f64 {
        self.window
    }

    /// Get the current window as an integer (floor).
    pub fn window_int(&self) -> u64 {
        self.window.max(1.0) as u64
    }

    /// Get the current phase.
    pub fn phase(&self) -> RatePhase {
        self.phase
    }

    /// Get current throughput.
    pub fn throughput(&self) -> f64 {
        self.throughput
    }

    /// Record a successful operation and increase the window.
    pub fn on_success(&mut self, timestamp_ms: u64) {
        self.total_success += 1;

        match self.phase {
            RatePhase::SlowStart => {
                // Exponential growth: double each round.
                self.window = (self.window + 1.0).min(self.config.max_window);
                if self.window >= self.ss_threshold {
                    self.phase = RatePhase::CongestionAvoidance;
                }
            }
            RatePhase::CongestionAvoidance | RatePhase::Recovery => {
                // Additive increase.
                let increment = self.config.additive_increase / self.window;
                self.window = (self.window + increment).min(self.config.max_window);
                if self.phase == RatePhase::Recovery {
                    self.phase = RatePhase::CongestionAvoidance;
                }
            }
        }

        self.record_snapshot(timestamp_ms);
    }

    /// Record a congestion signal and decrease the window.
    pub fn on_congestion(&mut self, timestamp_ms: u64, reason: &str) {
        self.total_congestion += 1;

        let event = CongestionEvent {
            timestamp_ms,
            window_at_event: self.window,
            reason: reason.to_string(),
        };
        self.congestion_log.push(event);

        // Multiplicative decrease.
        self.ss_threshold = (self.window * self.config.multiplicative_decrease)
            .max(self.config.min_window);
        self.window = (self.window * self.config.multiplicative_decrease)
            .max(self.config.min_window);
        self.phase = RatePhase::Recovery;

        self.record_snapshot(timestamp_ms);
    }

    /// Update throughput measurement (e.g. ops completed / time window).
    pub fn update_throughput(&mut self, ops_per_second: f64) {
        self.throughput = ops_per_second;
    }

    /// Provide explicit feedback to adjust the window.
    /// Positive feedback increases, negative decreases.
    pub fn feedback(&mut self, adjustment: f64, timestamp_ms: u64) {
        self.window = (self.window + adjustment)
            .max(self.config.min_window)
            .min(self.config.max_window);
        self.record_snapshot(timestamp_ms);
    }

    /// Check if the controller allows issuing a new operation.
    /// `in_flight` is the number of currently active operations.
    pub fn allows(&self, in_flight: u64) -> bool {
        in_flight < self.window_int()
    }

    /// Get the rate history.
    pub fn history(&self) -> &[RateSnapshot] {
        &self.history
    }

    /// Get the congestion event log.
    pub fn congestion_log(&self) -> &[CongestionEvent] {
        &self.congestion_log
    }

    /// Total successful operations recorded.
    pub fn total_success(&self) -> u64 {
        self.total_success
    }

    /// Total congestion events recorded.
    pub fn total_congestion(&self) -> u64 {
        self.total_congestion
    }

    /// Reset the controller to initial state.
    pub fn reset(&mut self) {
        self.window = self.config.initial_window;
        self.phase = RatePhase::SlowStart;
        self.ss_threshold = self.config.ss_threshold;
        self.total_success = 0;
        self.total_congestion = 0;
        self.history.clear();
        self.congestion_log.clear();
        self.throughput = 0.0;
    }

    fn record_snapshot(&mut self, timestamp_ms: u64) {
        let snapshot = RateSnapshot {
            timestamp_ms,
            window: self.window,
            phase: self.phase,
            throughput: self.throughput,
        };
        self.history.push(snapshot);
        if self.history.len() > self.config.history_capacity {
            self.history.remove(0);
        }
    }
}

// ── Fair Rate Limiter ─────────────────────────────────────────

/// Fair rate limiter that distributes capacity across multiple flows.
#[derive(Debug)]
pub struct FairRateLimiter {
    /// Total capacity (operations per round).
    total_capacity: u64,
    /// Per-flow usage in the current window.
    flow_usage: HashMap<String, u64>,
    /// Per-flow weight (higher = more allocation). Default 1.
    flow_weights: HashMap<String, u64>,
    /// Total operations issued this round.
    total_issued: u64,
}

impl FairRateLimiter {
    /// Create a new fair rate limiter with the given total capacity.
    pub fn new(total_capacity: u64) -> Self {
        Self {
            total_capacity,
            flow_usage: HashMap::new(),
            flow_weights: HashMap::new(),
            total_issued: 0,
        }
    }

    /// Register a flow with an optional weight (default 1).
    pub fn register_flow(&mut self, flow_id: &str, weight: u64) {
        let w = if weight == 0 { 1 } else { weight };
        self.flow_weights.insert(flow_id.to_string(), w);
        self.flow_usage.entry(flow_id.to_string()).or_insert(0);
    }

    /// Calculate the fair share for a given flow.
    pub fn fair_share(&self, flow_id: &str) -> u64 {
        let total_weight: u64 = self.flow_weights.values().sum();
        if total_weight == 0 {
            return 0;
        }
        let flow_weight = self.flow_weights.get(flow_id).copied().unwrap_or(1);
        (self.total_capacity * flow_weight) / total_weight
    }

    /// Check if a flow is allowed to issue an operation.
    pub fn allows(&self, flow_id: &str) -> bool {
        if self.total_issued >= self.total_capacity {
            return false;
        }
        let share = self.fair_share(flow_id);
        let used = self.flow_usage.get(flow_id).copied().unwrap_or(0);
        used < share
    }

    /// Record an operation for a flow.
    pub fn record(&mut self, flow_id: &str) -> bool {
        if !self.allows(flow_id) {
            return false;
        }
        *self.flow_usage.entry(flow_id.to_string()).or_insert(0) += 1;
        self.total_issued += 1;
        true
    }

    /// Reset all usage counters for a new round.
    pub fn reset_round(&mut self) {
        for v in self.flow_usage.values_mut() {
            *v = 0;
        }
        self.total_issued = 0;
    }

    /// Get usage for a flow.
    pub fn usage(&self, flow_id: &str) -> u64 {
        self.flow_usage.get(flow_id).copied().unwrap_or(0)
    }

    /// Get remaining capacity for a flow.
    pub fn remaining(&self, flow_id: &str) -> u64 {
        let share = self.fair_share(flow_id);
        let used = self.usage(flow_id);
        share.saturating_sub(used)
    }

    /// Total remaining capacity across all flows.
    pub fn total_remaining(&self) -> u64 {
        self.total_capacity.saturating_sub(self.total_issued)
    }

    /// Number of registered flows.
    pub fn flow_count(&self) -> usize {
        self.flow_weights.len()
    }

    /// Update total capacity.
    pub fn set_capacity(&mut self, capacity: u64) {
        self.total_capacity = capacity;
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let ctrl = RateController::with_defaults();
        assert_eq!(ctrl.window(), 1.0);
        assert_eq!(ctrl.phase(), RatePhase::SlowStart);
        assert_eq!(ctrl.total_success(), 0);
    }

    #[test]
    fn test_slow_start_growth() {
        let mut ctrl = RateController::with_defaults();
        // Each success in slow start adds 1 to window.
        ctrl.on_success(0);
        assert_eq!(ctrl.window(), 2.0);
        ctrl.on_success(1);
        assert_eq!(ctrl.window(), 3.0);
    }

    #[test]
    fn test_slow_start_to_congestion_avoidance() {
        let config = RateControllerConfig {
            initial_window: 1.0,
            ss_threshold: 4.0,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        assert_eq!(ctrl.phase(), RatePhase::SlowStart);
        // Grow window: 1 -> 2 -> 3 -> 4 (hits threshold).
        ctrl.on_success(0);
        ctrl.on_success(1);
        ctrl.on_success(2);
        assert_eq!(ctrl.phase(), RatePhase::CongestionAvoidance);
    }

    #[test]
    fn test_congestion_decreases_window() {
        let config = RateControllerConfig {
            initial_window: 10.0,
            multiplicative_decrease: 0.5,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        ctrl.on_congestion(100, "timeout");
        assert_eq!(ctrl.window(), 5.0);
        assert_eq!(ctrl.phase(), RatePhase::Recovery);
    }

    #[test]
    fn test_min_window_floor() {
        let config = RateControllerConfig {
            initial_window: 2.0,
            min_window: 1.0,
            multiplicative_decrease: 0.1,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        ctrl.on_congestion(0, "err");
        assert!(ctrl.window() >= 1.0);
        ctrl.on_congestion(1, "err");
        assert!(ctrl.window() >= 1.0);
    }

    #[test]
    fn test_max_window_ceiling() {
        let config = RateControllerConfig {
            initial_window: 1.0,
            max_window: 5.0,
            ss_threshold: 100.0,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        for i in 0..20 {
            ctrl.on_success(i);
        }
        assert!(ctrl.window() <= 5.0);
    }

    #[test]
    fn test_allows_check() {
        let config = RateControllerConfig {
            initial_window: 3.0,
            ..Default::default()
        };
        let ctrl = RateController::new(config);
        assert!(ctrl.allows(0));
        assert!(ctrl.allows(2));
        assert!(!ctrl.allows(3));
        assert!(!ctrl.allows(10));
    }

    #[test]
    fn test_congestion_log() {
        let mut ctrl = RateController::with_defaults();
        ctrl.on_congestion(100, "timeout");
        ctrl.on_congestion(200, "overload");
        let log = ctrl.congestion_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].reason, "timeout");
        assert_eq!(log[1].timestamp_ms, 200);
    }

    #[test]
    fn test_history_recording() {
        let config = RateControllerConfig {
            history_capacity: 5,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        for i in 0..10 {
            ctrl.on_success(i as u64);
        }
        assert!(ctrl.history().len() <= 5);
    }

    #[test]
    fn test_feedback_adjustment() {
        let mut ctrl = RateController::with_defaults();
        ctrl.feedback(5.0, 0);
        assert_eq!(ctrl.window(), 6.0);
        ctrl.feedback(-3.0, 1);
        assert_eq!(ctrl.window(), 3.0);
    }

    #[test]
    fn test_feedback_respects_bounds() {
        let config = RateControllerConfig {
            initial_window: 5.0,
            min_window: 1.0,
            max_window: 10.0,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        ctrl.feedback(-100.0, 0);
        assert_eq!(ctrl.window(), 1.0);
        ctrl.feedback(100.0, 1);
        assert_eq!(ctrl.window(), 10.0);
    }

    #[test]
    fn test_throughput_update() {
        let mut ctrl = RateController::with_defaults();
        assert_eq!(ctrl.throughput(), 0.0);
        ctrl.update_throughput(42.5);
        assert_eq!(ctrl.throughput(), 42.5);
    }

    #[test]
    fn test_reset() {
        let mut ctrl = RateController::with_defaults();
        ctrl.on_success(0);
        ctrl.on_success(1);
        ctrl.on_congestion(2, "err");
        ctrl.reset();
        assert_eq!(ctrl.window(), 1.0);
        assert_eq!(ctrl.phase(), RatePhase::SlowStart);
        assert_eq!(ctrl.total_success(), 0);
        assert!(ctrl.history().is_empty());
    }

    #[test]
    fn test_recovery_to_avoidance() {
        let config = RateControllerConfig {
            initial_window: 10.0,
            ..Default::default()
        };
        let mut ctrl = RateController::new(config);
        ctrl.on_congestion(0, "drop");
        assert_eq!(ctrl.phase(), RatePhase::Recovery);
        ctrl.on_success(1);
        assert_eq!(ctrl.phase(), RatePhase::CongestionAvoidance);
    }

    // ── Fair Rate Limiter Tests ──────────────────────────────

    #[test]
    fn test_fair_limiter_equal_shares() {
        let mut limiter = FairRateLimiter::new(100);
        limiter.register_flow("a", 1);
        limiter.register_flow("b", 1);
        assert_eq!(limiter.fair_share("a"), 50);
        assert_eq!(limiter.fair_share("b"), 50);
    }

    #[test]
    fn test_fair_limiter_weighted_shares() {
        let mut limiter = FairRateLimiter::new(100);
        limiter.register_flow("heavy", 3);
        limiter.register_flow("light", 1);
        assert_eq!(limiter.fair_share("heavy"), 75);
        assert_eq!(limiter.fair_share("light"), 25);
    }

    #[test]
    fn test_fair_limiter_allows_within_share() {
        let mut limiter = FairRateLimiter::new(10);
        limiter.register_flow("a", 1);
        limiter.register_flow("b", 1);
        // Each has 5 ops.
        for _ in 0..5 {
            assert!(limiter.record("a"));
        }
        assert!(!limiter.allows("a"));
        assert!(limiter.allows("b"));
    }

    #[test]
    fn test_fair_limiter_reset_round() {
        let mut limiter = FairRateLimiter::new(10);
        limiter.register_flow("a", 1);
        for _ in 0..10 {
            limiter.record("a");
        }
        limiter.reset_round();
        assert_eq!(limiter.usage("a"), 0);
        assert!(limiter.allows("a"));
    }

    #[test]
    fn test_fair_limiter_remaining() {
        let mut limiter = FairRateLimiter::new(20);
        limiter.register_flow("x", 1);
        limiter.record("x");
        limiter.record("x");
        assert_eq!(limiter.remaining("x"), 18);
    }

    #[test]
    fn test_fair_limiter_total_remaining() {
        let mut limiter = FairRateLimiter::new(10);
        limiter.register_flow("a", 1);
        limiter.record("a");
        assert_eq!(limiter.total_remaining(), 9);
    }

    #[test]
    fn test_fair_limiter_set_capacity() {
        let mut limiter = FairRateLimiter::new(10);
        limiter.register_flow("a", 1);
        limiter.set_capacity(20);
        assert_eq!(limiter.fair_share("a"), 20);
    }
}
