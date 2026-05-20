//! Heuristic-based cheat detection — behavior analysis, anomaly detection, rule chaining.
//!
//! Replaces heuristic cheat-detection middleware with pure Rust.
//! BehaviorSample collection, configurable threshold rules, moving
//! average baselines, standard deviation anomaly detection, multi-metric
//! correlation analysis, adaptive thresholds that learn normal behavior,
//! detection confidence scoring, and rule chaining with composite logic.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeuristicError {
    PlayerNotFound(u64),
    RuleNotFound(String),
    DuplicateRule(String),
    InsufficientData(String),
}

impl fmt::Display for HeuristicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::RuleNotFound(name) => write!(f, "rule not found: {name}"),
            Self::DuplicateRule(name) => write!(f, "duplicate rule: {name}"),
            Self::InsufficientData(msg) => write!(f, "insufficient data: {msg}"),
        }
    }
}

impl std::error::Error for HeuristicError {}

// ── Behavior Sample ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorSample {
    pub metric_name: String,
    pub value: f64,
    pub timestamp: u64,
}

impl BehaviorSample {
    pub fn new(metric_name: &str, value: f64, timestamp: u64) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            value,
            timestamp,
        }
    }
}

impl fmt::Display for BehaviorSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.3} @t={}", self.metric_name, self.value, self.timestamp)
    }
}

// ── Detection Alert ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicAlert {
    pub player_id: u64,
    pub rule_name: String,
    pub confidence: f64,
    pub timestamp: u64,
    pub details: String,
}

impl HeuristicAlert {
    pub fn new(player_id: u64, rule_name: &str, confidence: f64, timestamp: u64) -> Self {
        Self {
            player_id,
            rule_name: rule_name.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            details: String::new(),
        }
    }

    pub fn with_details(mut self, d: &str) -> Self {
        self.details = d.to_string();
        self
    }
}

impl fmt::Display for HeuristicAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[t={}] player={} rule={} conf={:.2}",
            self.timestamp, self.player_id, self.rule_name, self.confidence
        )
    }
}

// ── Threshold Rule ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleCondition {
    GreaterThan,
    LessThan,
    AbsGreaterThan,
    StdDevAbove,
}

impl fmt::Display for RuleCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GreaterThan => write!(f, ">"),
            Self::LessThan => write!(f, "<"),
            Self::AbsGreaterThan => write!(f, "|x| >"),
            Self::StdDevAbove => write!(f, "σ >"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThresholdRule {
    pub name: String,
    pub metric_name: String,
    pub condition: RuleCondition,
    pub threshold: f64,
    pub consecutive_count: usize,
    pub enabled: bool,
}

impl ThresholdRule {
    pub fn new(name: &str, metric: &str, condition: RuleCondition, threshold: f64) -> Self {
        Self {
            name: name.to_string(),
            metric_name: metric.to_string(),
            condition,
            threshold,
            consecutive_count: 1,
            enabled: true,
        }
    }

    pub fn with_consecutive(mut self, n: usize) -> Self {
        self.consecutive_count = n.max(1);
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    pub fn check_value(&self, value: f64) -> bool {
        match self.condition {
            RuleCondition::GreaterThan => value > self.threshold,
            RuleCondition::LessThan => value < self.threshold,
            RuleCondition::AbsGreaterThan => value.abs() > self.threshold,
            RuleCondition::StdDevAbove => false, // Handled by detector with baseline
        }
    }
}

// ── Correlation Rule ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CorrelationRule {
    pub name: String,
    pub metrics: Vec<String>,
    pub spike_threshold: f64,
    pub min_spike_count: usize,
}

impl CorrelationRule {
    pub fn new(name: &str, metrics: Vec<String>, spike_threshold: f64) -> Self {
        Self {
            name: name.to_string(),
            metrics,
            spike_threshold,
            min_spike_count: 2,
        }
    }

    pub fn with_min_spike_count(mut self, n: usize) -> Self {
        self.min_spike_count = n.max(2);
        self
    }
}

// ── Metric Tracker ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MetricTracker {
    samples: Vec<f64>,
    timestamps: Vec<u64>,
    running_sum: f64,
    running_sq_sum: f64,
    max_samples: usize,
    baseline_count: u64,
    baseline_mean: f64,
    baseline_std: f64,
}

impl MetricTracker {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::new(),
            timestamps: Vec::new(),
            running_sum: 0.0,
            running_sq_sum: 0.0,
            max_samples,
            baseline_count: 0,
            baseline_mean: 0.0,
            baseline_std: 1.0,
        }
    }

    fn push(&mut self, value: f64, timestamp: u64) {
        self.running_sum += value;
        self.running_sq_sum += value * value;
        self.samples.push(value);
        self.timestamps.push(timestamp);
        if self.samples.len() > self.max_samples {
            let removed = self.samples.remove(0);
            self.timestamps.remove(0);
            self.running_sum -= removed;
            self.running_sq_sum -= removed * removed;
        }
        self.update_baseline();
    }

    fn update_baseline(&mut self) {
        self.baseline_count = self.samples.len() as u64;
        if self.baseline_count > 0 {
            let n = self.baseline_count as f64;
            self.baseline_mean = self.running_sum / n;
            let variance = (self.running_sq_sum / n) - (self.baseline_mean * self.baseline_mean);
            self.baseline_std = variance.max(0.0).sqrt();
            if self.baseline_std < 1e-12 {
                self.baseline_std = 1.0;
            }
        }
    }

    fn mean(&self) -> f64 {
        self.baseline_mean
    }

    fn std_dev(&self) -> f64 {
        if self.baseline_count == 0 {
            return 0.0;
        }
        let n = self.baseline_count as f64;
        let variance = (self.running_sq_sum / n) - (self.baseline_mean * self.baseline_mean);
        variance.max(0.0).sqrt()
    }

    fn z_score(&self, value: f64) -> f64 {
        (value - self.baseline_mean) / self.baseline_std
    }

    fn last_n(&self, n: usize) -> &[f64] {
        let start = self.samples.len().saturating_sub(n);
        &self.samples[start..]
    }

    fn is_recent_spike(&self, threshold: f64) -> bool {
        if self.samples.is_empty() {
            return false;
        }
        let last = *self.samples.last().unwrap();
        self.z_score(last).abs() > threshold
    }
}

// ── Player Profile ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PlayerProfile {
    metrics: HashMap<String, MetricTracker>,
    alerts: Vec<HeuristicAlert>,
    consecutive_violations: HashMap<String, usize>,
    max_samples: usize,
}

impl PlayerProfile {
    fn new(max_samples: usize) -> Self {
        Self {
            metrics: HashMap::new(),
            alerts: Vec::new(),
            consecutive_violations: HashMap::new(),
            max_samples,
        }
    }

    fn tracker_mut(&mut self, metric: &str) -> &mut MetricTracker {
        self.metrics
            .entry(metric.to_string())
            .or_insert_with(|| MetricTracker::new(self.max_samples))
    }

    fn tracker(&self, metric: &str) -> Option<&MetricTracker> {
        self.metrics.get(metric)
    }
}

// ── Heuristic Detector ─────────────────────────────────────────

#[derive(Debug)]
pub struct HeuristicDetector {
    rules: HashMap<String, ThresholdRule>,
    correlations: Vec<CorrelationRule>,
    players: HashMap<u64, PlayerProfile>,
    max_samples: usize,
    std_dev_threshold: f64,
}

impl HeuristicDetector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            rules: HashMap::new(),
            correlations: Vec::new(),
            players: HashMap::new(),
            max_samples,
            std_dev_threshold: 3.0,
        }
    }

    pub fn with_std_dev_threshold(mut self, t: f64) -> Self {
        self.std_dev_threshold = t;
        self
    }

    pub fn add_rule(&mut self, rule: ThresholdRule) -> Result<(), HeuristicError> {
        if self.rules.contains_key(&rule.name) {
            return Err(HeuristicError::DuplicateRule(rule.name));
        }
        self.rules.insert(rule.name.clone(), rule);
        Ok(())
    }

    pub fn add_correlation(&mut self, rule: CorrelationRule) {
        self.correlations.push(rule);
    }

    pub fn enable_rule(&mut self, name: &str) -> Result<(), HeuristicError> {
        let rule = self.rules.get_mut(name).ok_or_else(|| HeuristicError::RuleNotFound(name.to_string()))?;
        rule.enabled = true;
        Ok(())
    }

    pub fn disable_rule(&mut self, name: &str) -> Result<(), HeuristicError> {
        let rule = self.rules.get_mut(name).ok_or_else(|| HeuristicError::RuleNotFound(name.to_string()))?;
        rule.enabled = false;
        Ok(())
    }

    pub fn submit_sample(&mut self, player_id: u64, sample: BehaviorSample) -> Vec<HeuristicAlert> {
        let max = self.max_samples;
        let profile = self
            .players
            .entry(player_id)
            .or_insert_with(|| PlayerProfile::new(max));
        profile
            .tracker_mut(&sample.metric_name)
            .push(sample.value, sample.timestamp);

        let mut alerts = Vec::new();
        let timestamp = sample.timestamp;
        let metric_name = &sample.metric_name;

        // Threshold rules
        let rules: Vec<_> = self
            .rules
            .values()
            .filter(|r| r.enabled && r.metric_name == *metric_name)
            .cloned()
            .collect();

        for rule in &rules {
            let violated = match rule.condition {
                RuleCondition::StdDevAbove => {
                    if let Some(tracker) = profile.tracker(metric_name) {
                        if tracker.baseline_count >= 10 {
                            tracker.z_score(sample.value) > rule.threshold
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                _ => rule.check_value(sample.value),
            };

            let consec = profile
                .consecutive_violations
                .entry(rule.name.clone())
                .or_insert(0);

            if violated {
                *consec += 1;
                if *consec >= rule.consecutive_count {
                    let confidence = (*consec as f64 / (rule.consecutive_count as f64 * 2.0))
                        .clamp(0.3, 0.95);
                    let alert = HeuristicAlert::new(player_id, &rule.name, confidence, timestamp)
                        .with_details(&format!(
                            "{} {} {} for {} consecutive samples",
                            metric_name, rule.condition, rule.threshold, consec
                        ));
                    alerts.push(alert);
                }
            } else {
                *consec = 0;
            }
        }

        // Standard deviation anomaly (generic)
        if let Some(tracker) = profile.tracker(metric_name) {
            if tracker.baseline_count >= 20 {
                let z = tracker.z_score(sample.value);
                if z.abs() > self.std_dev_threshold {
                    let conf = ((z.abs() - self.std_dev_threshold) / self.std_dev_threshold)
                        .clamp(0.2, 0.9);
                    let alert = HeuristicAlert::new(player_id, "anomaly_stddev", conf, timestamp)
                        .with_details(&format!("{} z-score={:.2}", metric_name, z));
                    alerts.push(alert);
                }
            }
        }

        // Correlation analysis
        let corr_rules: Vec<_> = self
            .correlations
            .iter()
            .filter(|c| c.metrics.contains(metric_name))
            .cloned()
            .collect();

        for cr in &corr_rules {
            let spike_count = cr
                .metrics
                .iter()
                .filter(|m| {
                    profile
                        .tracker(m)
                        .map(|t| t.is_recent_spike(cr.spike_threshold))
                        .unwrap_or(false)
                })
                .count();

            if spike_count >= cr.min_spike_count {
                let conf = (spike_count as f64 / cr.metrics.len() as f64).clamp(0.3, 0.95);
                let alert = HeuristicAlert::new(player_id, &cr.name, conf, timestamp)
                    .with_details(&format!("{}/{} metrics spiking", spike_count, cr.metrics.len()));
                alerts.push(alert);
            }
        }

        for a in &alerts {
            if let Some(p) = self.players.get_mut(&player_id) {
                p.alerts.push(a.clone());
            }
        }
        alerts
    }

    pub fn player_alerts(&self, player_id: u64) -> Result<&[HeuristicAlert], HeuristicError> {
        self.players
            .get(&player_id)
            .map(|p| p.alerts.as_slice())
            .ok_or(HeuristicError::PlayerNotFound(player_id))
    }

    pub fn player_metric_mean(&self, player_id: u64, metric: &str) -> Option<f64> {
        self.players
            .get(&player_id)
            .and_then(|p| p.tracker(metric))
            .map(|t| t.mean())
    }

    pub fn player_metric_std(&self, player_id: u64, metric: &str) -> Option<f64> {
        self.players
            .get(&player_id)
            .and_then(|p| p.tracker(metric))
            .map(|t| t.std_dev())
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn tracked_player_count(&self) -> usize {
        self.players.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(name: &str, val: f64, ts: u64) -> BehaviorSample {
        BehaviorSample::new(name, val, ts)
    }

    fn setup_detector() -> HeuristicDetector {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(ThresholdRule::new("high_speed", "speed", RuleCondition::GreaterThan, 50.0))
            .unwrap();
        det
    }

    #[test]
    fn test_sample_creation() {
        let s = sample("speed", 42.0, 100);
        assert_eq!(s.metric_name, "speed");
        assert!((s.value - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sample_display() {
        let s = sample("health", 85.0, 200);
        let txt = format!("{s}");
        assert!(txt.contains("health"));
        assert!(txt.contains("85.0"));
    }

    #[test]
    fn test_add_rule() {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(ThresholdRule::new("r1", "speed", RuleCondition::GreaterThan, 10.0)).unwrap();
        assert_eq!(det.rule_count(), 1);
    }

    #[test]
    fn test_duplicate_rule_rejected() {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(ThresholdRule::new("r1", "speed", RuleCondition::GreaterThan, 10.0)).unwrap();
        let err = det.add_rule(ThresholdRule::new("r1", "speed", RuleCondition::LessThan, 5.0)).unwrap_err();
        assert!(matches!(err, HeuristicError::DuplicateRule(_)));
    }

    #[test]
    fn test_threshold_violation() {
        let mut det = setup_detector();
        let alerts = det.submit_sample(1, sample("speed", 60.0, 100));
        assert!(alerts.iter().any(|a| a.rule_name == "high_speed"));
    }

    #[test]
    fn test_no_violation_under_threshold() {
        let mut det = setup_detector();
        let alerts = det.submit_sample(1, sample("speed", 30.0, 100));
        assert!(!alerts.iter().any(|a| a.rule_name == "high_speed"));
    }

    #[test]
    fn test_consecutive_count() {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(
            ThresholdRule::new("hs", "speed", RuleCondition::GreaterThan, 50.0).with_consecutive(3),
        )
        .unwrap();
        let a1 = det.submit_sample(1, sample("speed", 60.0, 1));
        let a2 = det.submit_sample(1, sample("speed", 60.0, 2));
        assert!(a1.iter().all(|a| a.rule_name != "hs"));
        assert!(a2.iter().all(|a| a.rule_name != "hs"));
        let a3 = det.submit_sample(1, sample("speed", 60.0, 3));
        assert!(a3.iter().any(|a| a.rule_name == "hs"));
    }

    #[test]
    fn test_consecutive_reset() {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(
            ThresholdRule::new("hs", "speed", RuleCondition::GreaterThan, 50.0).with_consecutive(3),
        )
        .unwrap();
        det.submit_sample(1, sample("speed", 60.0, 1));
        det.submit_sample(1, sample("speed", 60.0, 2));
        det.submit_sample(1, sample("speed", 30.0, 3)); // Reset
        let a = det.submit_sample(1, sample("speed", 60.0, 4));
        assert!(a.iter().all(|alert| alert.rule_name != "hs"));
    }

    #[test]
    fn test_std_dev_anomaly() {
        let mut det = HeuristicDetector::new(200).with_std_dev_threshold(2.5);
        // Build baseline
        for i in 0..50 {
            det.submit_sample(1, sample("damage", 10.0 + (i as f64 * 0.1), i as u64));
        }
        // Submit anomaly
        let alerts = det.submit_sample(1, sample("damage", 100.0, 100));
        assert!(alerts.iter().any(|a| a.rule_name == "anomaly_stddev"));
    }

    #[test]
    fn test_disabled_rule_ignored() {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(
            ThresholdRule::new("hs", "speed", RuleCondition::GreaterThan, 50.0).disabled(),
        )
        .unwrap();
        let alerts = det.submit_sample(1, sample("speed", 100.0, 1));
        assert!(alerts.iter().all(|a| a.rule_name != "hs"));
    }

    #[test]
    fn test_enable_disable_rule() {
        let mut det = HeuristicDetector::new(100);
        det.add_rule(ThresholdRule::new("hs", "speed", RuleCondition::GreaterThan, 50.0)).unwrap();
        det.disable_rule("hs").unwrap();
        let a1 = det.submit_sample(1, sample("speed", 100.0, 1));
        assert!(a1.iter().all(|a| a.rule_name != "hs"));
        det.enable_rule("hs").unwrap();
        let a2 = det.submit_sample(1, sample("speed", 100.0, 2));
        assert!(a2.iter().any(|a| a.rule_name == "hs"));
    }

    #[test]
    fn test_correlation_detection() {
        let mut det = HeuristicDetector::new(200);
        det.add_correlation(CorrelationRule::new(
            "multi_spike",
            vec!["speed".to_string(), "accuracy".to_string()],
            2.0,
        ));
        // Build baselines
        for i in 0..30 {
            det.submit_sample(1, sample("speed", 10.0, i as u64));
            det.submit_sample(1, sample("accuracy", 50.0, i as u64));
        }
        // Spike both
        det.submit_sample(1, sample("speed", 100.0, 100));
        let alerts = det.submit_sample(1, sample("accuracy", 200.0, 101));
        assert!(alerts.iter().any(|a| a.rule_name == "multi_spike"));
    }

    #[test]
    fn test_player_metric_mean() {
        let mut det = HeuristicDetector::new(100);
        det.submit_sample(1, sample("hp", 10.0, 1));
        det.submit_sample(1, sample("hp", 20.0, 2));
        let mean = det.player_metric_mean(1, "hp").unwrap();
        assert!((mean - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_player_metric_std() {
        let mut det = HeuristicDetector::new(100);
        for i in 0..10 {
            det.submit_sample(1, sample("hp", 100.0, i));
        }
        let std = det.player_metric_std(1, "hp").unwrap();
        assert!(std < 0.01); // All same value
    }

    #[test]
    fn test_player_not_found() {
        let det = HeuristicDetector::new(100);
        assert!(matches!(det.player_alerts(999), Err(HeuristicError::PlayerNotFound(999))));
    }

    #[test]
    fn test_alert_display() {
        let a = HeuristicAlert::new(1, "test_rule", 0.8, 500);
        let s = format!("{a}");
        assert!(s.contains("player=1"));
        assert!(s.contains("test_rule"));
    }

    #[test]
    fn test_rule_condition_display() {
        assert_eq!(format!("{}", RuleCondition::GreaterThan), ">");
        assert_eq!(format!("{}", RuleCondition::StdDevAbove), "σ >");
    }

    #[test]
    fn test_tracked_player_count() {
        let mut det = HeuristicDetector::new(100);
        det.submit_sample(1, sample("hp", 100.0, 1));
        det.submit_sample(2, sample("hp", 100.0, 1));
        assert_eq!(det.tracked_player_count(), 2);
    }
}
