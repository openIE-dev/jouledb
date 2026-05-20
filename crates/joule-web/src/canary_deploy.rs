//! Canary deployment: traffic splitting, metric comparison, automatic
//! rollback rules, deployment stages, health threshold checks, and
//! progressive delivery. Pure Rust — no I/O or real traffic routing.

use std::collections::HashMap;
use std::fmt;

// ── Deployment version ────────────────────────────────────────────

/// Identifies a deployment version.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeployVersion {
    pub name: String,
    pub tag: String,
}

impl DeployVersion {
    pub fn new(name: &str, tag: &str) -> Self {
        Self {
            name: name.to_string(),
            tag: tag.to_string(),
        }
    }
}

impl fmt::Display for DeployVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.name, self.tag)
    }
}

// ── Deployment stage ──────────────────────────────────────────────

/// Stage of a canary deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployStage {
    /// Not started.
    Pending,
    /// Canary is receiving a fraction of traffic.
    Canary,
    /// Canary is being promoted to full traffic.
    Promoting,
    /// Deployment succeeded and canary is now stable.
    Stable,
    /// Canary was rolled back.
    RolledBack,
    /// Deployment was aborted due to errors.
    Aborted,
}

impl fmt::Display for DeployStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Canary => write!(f, "canary"),
            Self::Promoting => write!(f, "promoting"),
            Self::Stable => write!(f, "stable"),
            Self::RolledBack => write!(f, "rolled_back"),
            Self::Aborted => write!(f, "aborted"),
        }
    }
}

// ── Metric snapshot ───────────────────────────────────────────────

/// Collected metrics for a deployment version.
#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    /// Total requests handled.
    pub request_count: u64,
    /// Number of errors (5xx).
    pub error_count: u64,
    /// Sum of latencies in microseconds (for average calculation).
    pub latency_sum_us: u64,
    /// P99 latency in microseconds.
    pub p99_latency_us: u64,
    /// Custom metrics (e.g. "cpu_pct", "memory_mb").
    pub custom: HashMap<String, f64>,
}

impl MetricSnapshot {
    /// Create a new empty snapshot.
    pub fn new() -> Self {
        Self {
            request_count: 0,
            error_count: 0,
            latency_sum_us: 0,
            p99_latency_us: 0,
            custom: HashMap::new(),
        }
    }

    /// Error rate as a fraction (0.0 to 1.0).
    pub fn error_rate(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.error_count as f64 / self.request_count as f64
        }
    }

    /// Average latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.latency_sum_us as f64 / self.request_count as f64
        }
    }

    /// Record a request.
    pub fn record_request(&mut self, latency_us: u64, is_error: bool) {
        self.request_count += 1;
        self.latency_sum_us = self.latency_sum_us.saturating_add(latency_us);
        if is_error {
            self.error_count += 1;
        }
    }
}

impl Default for MetricSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ── Rollback rule ─────────────────────────────────────────────────

/// A condition that triggers automatic rollback.
#[derive(Debug, Clone)]
pub enum RollbackRule {
    /// Rollback if canary error rate exceeds this fraction.
    MaxErrorRate(f64),
    /// Rollback if canary p99 latency exceeds this value (microseconds).
    MaxP99Latency(u64),
    /// Rollback if canary error rate exceeds baseline by this relative factor.
    ErrorRateIncrease(f64),
    /// Rollback if canary avg latency exceeds baseline by this relative factor.
    LatencyIncrease(f64),
    /// Rollback if a custom metric exceeds a threshold.
    CustomMetricThreshold { metric: String, max_value: f64 },
}

impl RollbackRule {
    /// Check if this rule triggers a rollback given canary and baseline metrics.
    pub fn should_rollback(&self, canary: &MetricSnapshot, baseline: &MetricSnapshot) -> bool {
        match self {
            Self::MaxErrorRate(max) => canary.error_rate() > *max,
            Self::MaxP99Latency(max) => canary.p99_latency_us > *max,
            Self::ErrorRateIncrease(factor) => {
                let base_rate = baseline.error_rate();
                if base_rate == 0.0 {
                    canary.error_rate() > 0.0 && *factor < f64::MAX
                } else {
                    canary.error_rate() > base_rate * (1.0 + factor)
                }
            }
            Self::LatencyIncrease(factor) => {
                let base_lat = baseline.avg_latency_us();
                if base_lat == 0.0 {
                    false
                } else {
                    canary.avg_latency_us() > base_lat * (1.0 + factor)
                }
            }
            Self::CustomMetricThreshold { metric, max_value } => {
                canary
                    .custom
                    .get(metric)
                    .map(|v| *v > *max_value)
                    .unwrap_or(false)
            }
        }
    }
}

// ── Traffic split ─────────────────────────────────────────────────

/// Traffic split configuration.
#[derive(Debug, Clone)]
pub struct TrafficSplit {
    /// Percentage of traffic to canary (0-100).
    pub canary_pct: u8,
}

impl TrafficSplit {
    /// Create a new split.
    pub fn new(canary_pct: u8) -> Self {
        Self {
            canary_pct: canary_pct.min(100),
        }
    }

    /// Percentage going to baseline.
    pub fn baseline_pct(&self) -> u8 {
        100 - self.canary_pct
    }

    /// Route a request given a hash value (0-99). Returns true if canary.
    pub fn route_to_canary(&self, hash_bucket: u8) -> bool {
        hash_bucket < self.canary_pct
    }
}

// ── Progressive delivery stages ───────────────────────────────────

/// Configuration for progressive canary delivery.
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Stages as canary percentages.
    pub stages: Vec<u8>,
    /// Minimum observation period per stage (in abstract time units).
    pub min_observation_units: u64,
}

impl ProgressiveConfig {
    /// Common pattern: 1% → 5% → 10% → 25% → 50% → 100%.
    pub fn standard() -> Self {
        Self {
            stages: vec![1, 5, 10, 25, 50, 100],
            min_observation_units: 300,
        }
    }

    /// Custom stages.
    pub fn custom(stages: Vec<u8>, min_observation: u64) -> Self {
        Self {
            stages,
            min_observation_units: min_observation,
        }
    }
}

// ── Canary deployment ─────────────────────────────────────────────

/// A canary deployment managing traffic split and rollback decisions.
#[derive(Debug, Clone)]
pub struct CanaryDeployment {
    /// Baseline version.
    pub baseline: DeployVersion,
    /// Canary version.
    pub canary: DeployVersion,
    /// Current deployment stage.
    pub stage: DeployStage,
    /// Current traffic split.
    pub split: TrafficSplit,
    /// Rollback rules.
    pub rules: Vec<RollbackRule>,
    /// Baseline metrics.
    pub baseline_metrics: MetricSnapshot,
    /// Canary metrics.
    pub canary_metrics: MetricSnapshot,
    /// Progressive delivery config.
    pub progressive: ProgressiveConfig,
    /// Current progressive stage index.
    pub current_stage_idx: usize,
    /// Time units elapsed in current stage.
    pub stage_elapsed: u64,
    /// Event log.
    events: Vec<DeployEvent>,
}

/// A deployment lifecycle event.
#[derive(Debug, Clone)]
pub struct DeployEvent {
    pub stage: DeployStage,
    pub message: String,
    pub timestamp_unit: u64,
}

impl CanaryDeployment {
    /// Create a new canary deployment.
    pub fn new(
        baseline: DeployVersion,
        canary: DeployVersion,
        rules: Vec<RollbackRule>,
        progressive: ProgressiveConfig,
    ) -> Self {
        let initial_pct = progressive.stages.first().copied().unwrap_or(0);
        Self {
            baseline,
            canary,
            stage: DeployStage::Pending,
            split: TrafficSplit::new(initial_pct),
            rules,
            baseline_metrics: MetricSnapshot::new(),
            canary_metrics: MetricSnapshot::new(),
            progressive,
            current_stage_idx: 0,
            stage_elapsed: 0,
            events: Vec::new(),
        }
    }

    /// Start the canary deployment.
    pub fn start(&mut self, timestamp: u64) {
        self.stage = DeployStage::Canary;
        let pct = self.progressive.stages.first().copied().unwrap_or(0);
        self.split = TrafficSplit::new(pct);
        self.log_event(timestamp, format!("started canary at {pct}%"));
    }

    /// Record a metric observation and check rollback rules.
    /// Returns true if rollback was triggered.
    pub fn observe(&mut self, timestamp: u64) -> bool {
        if self.stage != DeployStage::Canary {
            return false;
        }

        for rule in &self.rules {
            if rule.should_rollback(&self.canary_metrics, &self.baseline_metrics) {
                self.stage = DeployStage::RolledBack;
                self.split = TrafficSplit::new(0);
                self.log_event(timestamp, "rollback triggered by rule violation".into());
                return true;
            }
        }

        false
    }

    /// Advance time by given units and potentially promote to next stage.
    /// Returns the new canary percentage if promoted.
    pub fn advance_time(&mut self, units: u64, timestamp: u64) -> Option<u8> {
        if self.stage != DeployStage::Canary {
            return None;
        }

        self.stage_elapsed = self.stage_elapsed.saturating_add(units);

        if self.stage_elapsed >= self.progressive.min_observation_units {
            // Check rollback first.
            if self.observe(timestamp) {
                return None;
            }

            // Try to advance to next stage.
            let next_idx = self.current_stage_idx + 1;
            if next_idx < self.progressive.stages.len() {
                self.current_stage_idx = next_idx;
                let pct = self.progressive.stages[next_idx];
                self.split = TrafficSplit::new(pct);
                self.stage_elapsed = 0;

                if pct == 100 {
                    self.stage = DeployStage::Promoting;
                    self.log_event(timestamp, "promoting canary to 100%".into());
                } else {
                    self.log_event(timestamp, format!("advanced to {pct}%"));
                }

                return Some(pct);
            }
        }

        None
    }

    /// Finalize promotion (canary becomes stable).
    pub fn finalize(&mut self, timestamp: u64) {
        if self.stage == DeployStage::Promoting || self.split.canary_pct == 100 {
            self.stage = DeployStage::Stable;
            self.log_event(timestamp, "deployment finalized as stable".into());
        }
    }

    /// Manually abort the deployment.
    pub fn abort(&mut self, reason: &str, timestamp: u64) {
        self.stage = DeployStage::Aborted;
        self.split = TrafficSplit::new(0);
        self.log_event(timestamp, format!("aborted: {reason}"));
    }

    /// Get the event log.
    pub fn events(&self) -> &[DeployEvent] {
        &self.events
    }

    fn log_event(&mut self, timestamp_unit: u64, message: String) {
        self.events.push(DeployEvent {
            stage: self.stage,
            message,
            timestamp_unit,
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_v() -> DeployVersion {
        DeployVersion::new("myapp", "v1.0.0")
    }

    fn canary_v() -> DeployVersion {
        DeployVersion::new("myapp", "v1.1.0-canary")
    }

    #[test]
    fn deploy_version_display() {
        let v = DeployVersion::new("app", "v2");
        assert_eq!(v.to_string(), "app:v2");
    }

    #[test]
    fn deploy_stage_display() {
        assert_eq!(DeployStage::Pending.to_string(), "pending");
        assert_eq!(DeployStage::Canary.to_string(), "canary");
        assert_eq!(DeployStage::Promoting.to_string(), "promoting");
        assert_eq!(DeployStage::Stable.to_string(), "stable");
        assert_eq!(DeployStage::RolledBack.to_string(), "rolled_back");
        assert_eq!(DeployStage::Aborted.to_string(), "aborted");
    }

    #[test]
    fn metric_snapshot_empty() {
        let m = MetricSnapshot::new();
        assert_eq!(m.error_rate(), 0.0);
        assert_eq!(m.avg_latency_us(), 0.0);
    }

    #[test]
    fn metric_snapshot_record() {
        let mut m = MetricSnapshot::new();
        m.record_request(100, false);
        m.record_request(200, true);
        m.record_request(300, false);
        assert_eq!(m.request_count, 3);
        assert_eq!(m.error_count, 1);
        assert!((m.error_rate() - 1.0 / 3.0).abs() < 1e-10);
        assert!((m.avg_latency_us() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn traffic_split_basic() {
        let split = TrafficSplit::new(10);
        assert_eq!(split.canary_pct, 10);
        assert_eq!(split.baseline_pct(), 90);
    }

    #[test]
    fn traffic_split_clamped() {
        let split = TrafficSplit::new(200);
        assert_eq!(split.canary_pct, 100);
    }

    #[test]
    fn traffic_split_routing() {
        let split = TrafficSplit::new(50);
        assert!(split.route_to_canary(0));
        assert!(split.route_to_canary(49));
        assert!(!split.route_to_canary(50));
        assert!(!split.route_to_canary(99));
    }

    #[test]
    fn rollback_max_error_rate() {
        let rule = RollbackRule::MaxErrorRate(0.05);
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.error_count = 10; // 10%
        let baseline = MetricSnapshot::new();
        assert!(rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn rollback_max_error_rate_ok() {
        let rule = RollbackRule::MaxErrorRate(0.05);
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.error_count = 3; // 3%
        let baseline = MetricSnapshot::new();
        assert!(!rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn rollback_max_p99() {
        let rule = RollbackRule::MaxP99Latency(5000);
        let mut canary = MetricSnapshot::new();
        canary.p99_latency_us = 6000;
        assert!(rule.should_rollback(&canary, &MetricSnapshot::new()));
    }

    #[test]
    fn rollback_error_rate_increase() {
        let rule = RollbackRule::ErrorRateIncrease(0.5); // 50% increase
        let mut baseline = MetricSnapshot::new();
        baseline.request_count = 1000;
        baseline.error_count = 10; // 1%
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.error_count = 2; // 2% > 1% * 1.5 = 1.5%
        assert!(rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn rollback_error_rate_increase_ok() {
        let rule = RollbackRule::ErrorRateIncrease(1.0); // 100% increase
        let mut baseline = MetricSnapshot::new();
        baseline.request_count = 1000;
        baseline.error_count = 10; // 1%
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.error_count = 1; // 1% <= 1% * 2.0 = 2%
        assert!(!rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn rollback_latency_increase() {
        let rule = RollbackRule::LatencyIncrease(0.5);
        let mut baseline = MetricSnapshot::new();
        baseline.request_count = 100;
        baseline.latency_sum_us = 10_000; // avg 100us
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.latency_sum_us = 20_000; // avg 200us > 100 * 1.5 = 150us
        assert!(rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn rollback_custom_metric() {
        let rule = RollbackRule::CustomMetricThreshold {
            metric: "cpu_pct".into(),
            max_value: 80.0,
        };
        let mut canary = MetricSnapshot::new();
        canary.custom.insert("cpu_pct".into(), 90.0);
        assert!(rule.should_rollback(&canary, &MetricSnapshot::new()));
    }

    #[test]
    fn rollback_custom_metric_missing() {
        let rule = RollbackRule::CustomMetricThreshold {
            metric: "cpu_pct".into(),
            max_value: 80.0,
        };
        let canary = MetricSnapshot::new();
        assert!(!rule.should_rollback(&canary, &MetricSnapshot::new()));
    }

    #[test]
    fn progressive_standard() {
        let p = ProgressiveConfig::standard();
        assert_eq!(p.stages, vec![1, 5, 10, 25, 50, 100]);
        assert_eq!(p.min_observation_units, 300);
    }

    #[test]
    fn canary_deploy_start() {
        let mut deploy = CanaryDeployment::new(
            baseline_v(),
            canary_v(),
            vec![],
            ProgressiveConfig::standard(),
        );
        assert_eq!(deploy.stage, DeployStage::Pending);
        deploy.start(0);
        assert_eq!(deploy.stage, DeployStage::Canary);
        assert_eq!(deploy.split.canary_pct, 1);
        assert_eq!(deploy.events().len(), 1);
    }

    #[test]
    fn canary_deploy_progressive_advance() {
        let config = ProgressiveConfig::custom(vec![10, 50, 100], 100);
        let mut deploy = CanaryDeployment::new(baseline_v(), canary_v(), vec![], config);
        deploy.start(0);
        assert_eq!(deploy.split.canary_pct, 10);

        // Not enough time yet.
        assert_eq!(deploy.advance_time(50, 50), None);

        // Now enough time.
        let pct = deploy.advance_time(50, 100);
        assert_eq!(pct, Some(50));
        assert_eq!(deploy.split.canary_pct, 50);

        // Advance to 100% (promoting).
        let pct = deploy.advance_time(100, 200);
        assert_eq!(pct, Some(100));
        assert_eq!(deploy.stage, DeployStage::Promoting);
    }

    #[test]
    fn canary_deploy_rollback_on_observe() {
        let rules = vec![RollbackRule::MaxErrorRate(0.05)];
        let config = ProgressiveConfig::custom(vec![10, 50], 100);
        let mut deploy = CanaryDeployment::new(baseline_v(), canary_v(), rules, config);
        deploy.start(0);

        // Simulate bad canary.
        deploy.canary_metrics.request_count = 100;
        deploy.canary_metrics.error_count = 10; // 10% > 5%

        let rolled_back = deploy.observe(10);
        assert!(rolled_back);
        assert_eq!(deploy.stage, DeployStage::RolledBack);
        assert_eq!(deploy.split.canary_pct, 0);
    }

    #[test]
    fn canary_deploy_rollback_during_advance() {
        let rules = vec![RollbackRule::MaxErrorRate(0.05)];
        let config = ProgressiveConfig::custom(vec![10, 50], 100);
        let mut deploy = CanaryDeployment::new(baseline_v(), canary_v(), rules, config);
        deploy.start(0);

        deploy.canary_metrics.request_count = 100;
        deploy.canary_metrics.error_count = 10;

        let result = deploy.advance_time(100, 100);
        assert!(result.is_none());
        assert_eq!(deploy.stage, DeployStage::RolledBack);
    }

    #[test]
    fn canary_deploy_finalize() {
        let config = ProgressiveConfig::custom(vec![100], 0);
        let mut deploy = CanaryDeployment::new(baseline_v(), canary_v(), vec![], config);
        deploy.start(0);
        deploy.advance_time(0, 0);
        deploy.stage = DeployStage::Promoting;
        deploy.finalize(1);
        assert_eq!(deploy.stage, DeployStage::Stable);
    }

    #[test]
    fn canary_deploy_abort() {
        let mut deploy = CanaryDeployment::new(
            baseline_v(),
            canary_v(),
            vec![],
            ProgressiveConfig::standard(),
        );
        deploy.start(0);
        deploy.abort("manual abort", 5);
        assert_eq!(deploy.stage, DeployStage::Aborted);
        assert_eq!(deploy.split.canary_pct, 0);
        let last_event = deploy.events().last().unwrap();
        assert!(last_event.message.contains("manual abort"));
    }

    #[test]
    fn observe_does_nothing_when_not_canary() {
        let mut deploy = CanaryDeployment::new(
            baseline_v(),
            canary_v(),
            vec![RollbackRule::MaxErrorRate(0.0)],
            ProgressiveConfig::standard(),
        );
        // Still Pending.
        assert!(!deploy.observe(0));
    }

    #[test]
    fn advance_does_nothing_when_not_canary() {
        let mut deploy = CanaryDeployment::new(
            baseline_v(),
            canary_v(),
            vec![],
            ProgressiveConfig::standard(),
        );
        assert!(deploy.advance_time(1000, 0).is_none());
    }

    #[test]
    fn error_rate_increase_zero_baseline() {
        let rule = RollbackRule::ErrorRateIncrease(0.5);
        let baseline = MetricSnapshot::new();
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.error_count = 1;
        // Baseline has 0 errors, canary has some → rollback.
        assert!(rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn latency_increase_zero_baseline() {
        let rule = RollbackRule::LatencyIncrease(0.5);
        let baseline = MetricSnapshot::new();
        let mut canary = MetricSnapshot::new();
        canary.request_count = 100;
        canary.latency_sum_us = 10_000;
        // Baseline has 0 avg latency → no rollback (can't divide by zero).
        assert!(!rule.should_rollback(&canary, &baseline));
    }

    #[test]
    fn event_log_grows() {
        let config = ProgressiveConfig::custom(vec![5, 10, 25], 10);
        let mut deploy = CanaryDeployment::new(baseline_v(), canary_v(), vec![], config);
        deploy.start(0);
        deploy.advance_time(10, 10);
        deploy.advance_time(10, 20);
        assert!(deploy.events().len() >= 3);
    }

    #[test]
    fn metric_snapshot_default() {
        let m = MetricSnapshot::default();
        assert_eq!(m.request_count, 0);
    }
}
