//! Alert engine: threshold/rate-of-change/absence rules, alert states
//! (pending/firing/resolved), notification channels, silencing/inhibition,
//! alert grouping, escalation, and cooldown periods.

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// Kind of alert rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleKind {
    /// Fires when a value crosses a threshold.
    Threshold,
    /// Fires when the rate of change exceeds a limit.
    RateOfChange,
    /// Fires when no data is received within a window.
    Absence,
}

impl RuleKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuleKind::Threshold => "threshold",
            RuleKind::RateOfChange => "rate_of_change",
            RuleKind::Absence => "absence",
        }
    }
}

/// Comparison operator for threshold rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
}

impl Comparator {
    pub fn evaluate(&self, value: f64, threshold: f64) -> bool {
        match self {
            Comparator::GreaterThan => value > threshold,
            Comparator::GreaterThanOrEqual => value >= threshold,
            Comparator::LessThan => value < threshold,
            Comparator::LessThanOrEqual => value <= threshold,
            Comparator::Equal => (value - threshold).abs() < f64::EPSILON,
        }
    }
}

/// An alert rule definition.
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub kind: RuleKind,
    pub metric_name: String,
    pub comparator: Comparator,
    pub threshold: f64,
    /// How long the condition must hold before firing (seconds).
    pub pending_duration_secs: u64,
    /// For absence rules: how many seconds without data to trigger.
    pub absence_window_secs: u64,
    pub labels: HashMap<String, String>,
    pub severity: String,
}

impl AlertRule {
    pub fn threshold(name: &str, metric: &str, cmp: Comparator, threshold: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            kind: RuleKind::Threshold,
            metric_name: metric.to_string(),
            comparator: cmp,
            threshold,
            pending_duration_secs: 0,
            absence_window_secs: 0,
            labels: HashMap::new(),
            severity: "warning".to_string(),
        }
    }

    pub fn rate_of_change(name: &str, metric: &str, cmp: Comparator, threshold: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            kind: RuleKind::RateOfChange,
            metric_name: metric.to_string(),
            comparator: cmp,
            threshold,
            pending_duration_secs: 0,
            absence_window_secs: 0,
            labels: HashMap::new(),
            severity: "warning".to_string(),
        }
    }

    pub fn absence(name: &str, metric: &str, window_secs: u64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            kind: RuleKind::Absence,
            metric_name: metric.to_string(),
            comparator: Comparator::GreaterThan, // unused
            threshold: 0.0,
            pending_duration_secs: 0,
            absence_window_secs: window_secs,
            labels: HashMap::new(),
            severity: "critical".to_string(),
        }
    }

    pub fn with_severity(mut self, severity: &str) -> Self {
        self.severity = severity.to_string();
        self
    }

    pub fn with_pending(mut self, secs: u64) -> Self {
        self.pending_duration_secs = secs;
        self
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Evaluate a threshold or rate-of-change rule against a value.
    pub fn evaluate_value(&self, value: f64) -> bool {
        self.comparator.evaluate(value, self.threshold)
    }

    /// Evaluate an absence rule: true if last_data_time is too old.
    pub fn evaluate_absence(&self, last_data_time: DateTime<Utc>, now: DateTime<Utc>) -> bool {
        let elapsed = (now - last_data_time).num_seconds();
        elapsed > self.absence_window_secs as i64
    }
}

/// Alert lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertState {
    Pending,
    Firing,
    Resolved,
}

impl AlertState {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertState::Pending => "pending",
            AlertState::Firing => "firing",
            AlertState::Resolved => "resolved",
        }
    }
}

/// An active alert instance.
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub rule_id: String,
    pub rule_name: String,
    pub state: AlertState,
    pub severity: String,
    pub value: f64,
    pub started_at: DateTime<Utc>,
    pub fired_at: Option<DateTime<Utc>>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

impl Alert {
    pub fn from_rule(rule: &AlertRule, value: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            rule_id: rule.id.clone(),
            rule_name: rule.name.clone(),
            state: AlertState::Pending,
            severity: rule.severity.clone(),
            value,
            started_at: Utc::now(),
            fired_at: None,
            resolved_at: None,
            labels: rule.labels.clone(),
            annotations: HashMap::new(),
        }
    }

    pub fn fire(&mut self) {
        self.state = AlertState::Firing;
        self.fired_at = Some(Utc::now());
    }

    pub fn resolve(&mut self) {
        self.state = AlertState::Resolved;
        self.resolved_at = Some(Utc::now());
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "rule_name": self.rule_name,
            "state": self.state.as_str(),
            "severity": self.severity,
            "value": self.value,
            "started_at": self.started_at.to_rfc3339(),
            "fired_at": self.fired_at.map(|t| t.to_rfc3339()),
            "resolved_at": self.resolved_at.map(|t| t.to_rfc3339()),
            "labels": self.labels,
        })
    }
}

/// A notification channel for alert delivery.
#[derive(Debug, Clone)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: String, // e.g. "email", "slack", "pagerduty"
    pub config: HashMap<String, String>,
}

impl NotificationChannel {
    pub fn new(name: &str, channel_type: &str) -> Self {
        Self {
            name: name.to_string(),
            channel_type: channel_type.to_string(),
            config: HashMap::new(),
        }
    }

    pub fn with_config(mut self, key: &str, value: &str) -> Self {
        self.config.insert(key.to_string(), value.to_string());
        self
    }
}

/// A silencing rule: suppresses alerts matching certain labels during a time range.
#[derive(Debug, Clone)]
pub struct Silence {
    pub id: String,
    pub matchers: HashMap<String, String>,
    pub starts_at: DateTime<Utc>,
    pub ends_at: DateTime<Utc>,
    pub comment: String,
}

impl Silence {
    pub fn new(matchers: HashMap<String, String>, ends_at: DateTime<Utc>, comment: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            matchers,
            starts_at: Utc::now(),
            ends_at,
            comment: comment.to_string(),
        }
    }

    /// Check if this silence matches an alert.
    pub fn matches(&self, alert: &Alert, now: DateTime<Utc>) -> bool {
        if now < self.starts_at || now > self.ends_at {
            return false;
        }
        self.matchers
            .iter()
            .all(|(k, v)| alert.labels.get(k).map(|av| av == v).unwrap_or(false))
    }

    pub fn is_active(&self, now: DateTime<Utc>) -> bool {
        now >= self.starts_at && now <= self.ends_at
    }
}

/// Escalation level.
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    pub level: u32,
    pub after_secs: u64,
    pub channel_name: String,
}

/// Cooldown tracker: prevents re-firing an alert within a cooldown period.
#[derive(Debug)]
pub struct CooldownTracker {
    pub cooldown_secs: u64,
    last_fired: HashMap<String, DateTime<Utc>>,
}

impl CooldownTracker {
    pub fn new(cooldown_secs: u64) -> Self {
        Self {
            cooldown_secs,
            last_fired: HashMap::new(),
        }
    }

    pub fn can_fire(&self, rule_id: &str, now: DateTime<Utc>) -> bool {
        match self.last_fired.get(rule_id) {
            None => true,
            Some(last) => {
                (now - *last).num_seconds() >= self.cooldown_secs as i64
            }
        }
    }

    pub fn record_fire(&mut self, rule_id: &str) {
        self.last_fired.insert(rule_id.to_string(), Utc::now());
    }
}

/// Alert grouping: groups alerts by a set of label keys.
pub fn group_alerts<'a>(alerts: &'a [Alert], group_by: &[&str]) -> HashMap<String, Vec<&'a Alert>> {
    let mut groups: HashMap<String, Vec<&Alert>> = HashMap::new();
    for alert in alerts {
        let key: String = group_by
            .iter()
            .map(|k| {
                alert
                    .labels
                    .get(*k)
                    .cloned()
                    .unwrap_or_else(|| "".to_string())
            })
            .collect::<Vec<_>>()
            .join("|");
        groups.entry(key).or_default().push(alert);
    }
    groups
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparator_evaluate() {
        assert!(Comparator::GreaterThan.evaluate(5.0, 3.0));
        assert!(!Comparator::GreaterThan.evaluate(3.0, 5.0));
        assert!(Comparator::LessThan.evaluate(1.0, 2.0));
        assert!(Comparator::Equal.evaluate(3.0, 3.0));
        assert!(Comparator::GreaterThanOrEqual.evaluate(3.0, 3.0));
        assert!(Comparator::LessThanOrEqual.evaluate(3.0, 3.0));
    }

    #[test]
    fn test_threshold_rule() {
        let rule = AlertRule::threshold("high_cpu", "cpu_usage", Comparator::GreaterThan, 80.0)
            .with_severity("critical");
        assert!(rule.evaluate_value(90.0));
        assert!(!rule.evaluate_value(70.0));
        assert_eq!(rule.severity, "critical");
    }

    #[test]
    fn test_rate_of_change_rule() {
        let rule =
            AlertRule::rate_of_change("spike", "rps", Comparator::GreaterThan, 100.0);
        assert_eq!(rule.kind, RuleKind::RateOfChange);
        assert!(rule.evaluate_value(150.0));
    }

    #[test]
    fn test_absence_rule() {
        let rule = AlertRule::absence("no_data", "heartbeat", 300);
        let now = Utc::now();
        let old = now - Duration::seconds(600);
        assert!(rule.evaluate_absence(old, now));
        let recent = now - Duration::seconds(100);
        assert!(!rule.evaluate_absence(recent, now));
    }

    #[test]
    fn test_alert_lifecycle() {
        let rule = AlertRule::threshold("test", "m", Comparator::GreaterThan, 50.0);
        let mut alert = Alert::from_rule(&rule, 75.0);
        assert_eq!(alert.state, AlertState::Pending);
        alert.fire();
        assert_eq!(alert.state, AlertState::Firing);
        assert!(alert.fired_at.is_some());
        alert.resolve();
        assert_eq!(alert.state, AlertState::Resolved);
        assert!(alert.resolved_at.is_some());
    }

    #[test]
    fn test_alert_to_json() {
        let rule = AlertRule::threshold("test", "m", Comparator::GreaterThan, 50.0)
            .with_label("service", "web");
        let alert = Alert::from_rule(&rule, 80.0);
        let j = alert.to_json();
        assert_eq!(j["state"], "pending");
        assert_eq!(j["severity"], "warning");
        assert_eq!(j["labels"]["service"], "web");
    }

    #[test]
    fn test_silence_matching() {
        let mut matchers = HashMap::new();
        matchers.insert("service".to_string(), "web".to_string());
        let silence = Silence::new(matchers, Utc::now() + Duration::hours(1), "maintenance");
        let rule = AlertRule::threshold("test", "m", Comparator::GreaterThan, 50.0)
            .with_label("service", "web");
        let alert = Alert::from_rule(&rule, 80.0);
        assert!(silence.matches(&alert, Utc::now()));
    }

    #[test]
    fn test_silence_expired() {
        let mut matchers = HashMap::new();
        matchers.insert("service".to_string(), "web".to_string());
        let mut silence = Silence::new(matchers, Utc::now() + Duration::hours(1), "maint");
        silence.ends_at = Utc::now() - Duration::hours(1); // expired
        let rule = AlertRule::threshold("t", "m", Comparator::GreaterThan, 0.0)
            .with_label("service", "web");
        let alert = Alert::from_rule(&rule, 1.0);
        assert!(!silence.matches(&alert, Utc::now()));
    }

    #[test]
    fn test_cooldown_tracker() {
        let mut tracker = CooldownTracker::new(300);
        assert!(tracker.can_fire("rule-1", Utc::now()));
        tracker.record_fire("rule-1");
        assert!(!tracker.can_fire("rule-1", Utc::now()));
        // After cooldown passes:
        assert!(tracker.can_fire("rule-1", Utc::now() + Duration::seconds(301)));
    }

    #[test]
    fn test_alert_grouping() {
        let rule = AlertRule::threshold("t", "m", Comparator::GreaterThan, 0.0)
            .with_label("service", "web")
            .with_label("env", "prod");
        let a1 = Alert::from_rule(&rule, 1.0);

        let rule2 = AlertRule::threshold("t", "m", Comparator::GreaterThan, 0.0)
            .with_label("service", "api")
            .with_label("env", "prod");
        let a2 = Alert::from_rule(&rule2, 2.0);

        let alerts = vec![a1, a2];
        let groups = group_alerts(&alerts, &["service"]);
        assert_eq!(groups.len(), 2);
        assert!(groups.contains_key("web"));
        assert!(groups.contains_key("api"));
    }

    #[test]
    fn test_notification_channel() {
        let ch = NotificationChannel::new("ops-slack", "slack")
            .with_config("webhook_url", "https://hooks.slack.com/xxx");
        assert_eq!(ch.channel_type, "slack");
        assert_eq!(ch.config["webhook_url"], "https://hooks.slack.com/xxx");
    }

    #[test]
    fn test_escalation_level() {
        let esc = EscalationLevel {
            level: 1,
            after_secs: 600,
            channel_name: "pagerduty".to_string(),
        };
        assert_eq!(esc.level, 1);
        assert_eq!(esc.after_secs, 600);
    }

    #[test]
    fn test_rule_kind_strings() {
        assert_eq!(RuleKind::Threshold.as_str(), "threshold");
        assert_eq!(RuleKind::RateOfChange.as_str(), "rate_of_change");
        assert_eq!(RuleKind::Absence.as_str(), "absence");
    }
}
